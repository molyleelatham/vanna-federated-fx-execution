"""Participant-side Flower app; each invocation sees one desk partition.

Default path: FedXgbBagging — each client trains local XGBoost trees and
returns updated model bytes.

Secure path (``secure-aggregation=true``): FedAvg over the transparent
NumPy logistic model under SecAgg+. The ``secaggplus_mod`` masks the
update before it leaves the desk, so the server only recovers the
weighted average across desks.

Includes desk partition persistence.
"""

import numpy as np
from flwr.app import ArrayRecord, Context, Message, MetricRecord, RecordDict
from flwr.app.message_type import MessageType
from flwr.client.mod import secaggplus_mod
from flwr.clientapp import ClientApp
from flwr.clientapp.typing import ClientAppCallable
from flwr.common import (
    Code,
    EvaluateRes,
    FitRes,
    Status,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
)
from flwr.common.secure_aggregation.secaggplus_constants import RECORD_KEY_CONFIGS
from flwr.compat.common.recorddict_compat import (
    arrayrecord_to_parameters,
    evaluateres_to_recorddict,
    fitres_to_recorddict,
)

from .data import (
    DeskData,
    generate_desk_data_legacy,
    require_local_vault,
    vault_from_partition,
)
from .model import (
    accuracy as logistic_accuracy,
    binary_cross_entropy,
    train as logistic_train,
)
from .xgboost_federated import (
    train_xgboost_local,
    evaluate_xgboost,
    bytes_to_xgb_model,
)
from .persistence import save_desk_partition, load_desk_partition
from .privacy import validate_shared_payload


def secure_aggregation_mod(
    msg: Message, ctxt: Context, call_next: ClientAppCallable
) -> Message:
    """Apply SecAgg+ masking only to messages carrying SecAgg+ configs.

    Plain FedXgbBagging train/evaluate messages pass through unchanged.
    """
    if (
        msg.metadata.message_type == MessageType.TRAIN
        and RECORD_KEY_CONFIGS in msg.content.config_records
    ):
        return secaggplus_mod(msg, ctxt, call_next)
    return call_next(msg, ctxt)


app = ClientApp(mods=[secure_aggregation_mod])


def _resolve_desk_data(partition_id: int, persist: bool) -> DeskData:
    """Load the persisted desk partition, or generate (and persist) it."""
    loaded = load_desk_partition(partition_id)
    if loaded:
        _, x_train, y_train, x_test, y_test = loaded
        # The npz holds features only. Rebuild the vault in this process.
        return DeskData(
            x_train=x_train,
            y_train=y_train,
            x_test=x_test,
            y_test=y_test,
            vault=vault_from_partition(partition_id, x_train, x_test),
        )
    data = generate_desk_data_legacy(partition_id)
    if persist:
        from .desk_config import DEFAULT_DESK_CONFIGS

        save_desk_partition(
            partition_id,
            DEFAULT_DESK_CONFIGS[partition_id].to_dict(),
            data.x_train,
            data.y_train,
            data.x_test,
            data.y_test,
        )
    return data


def _global_parameters(msg: Message, record_key: str) -> list[np.ndarray]:
    """Decode legacy Parameters from a compat ArrayRecord (bytes-encoded)."""
    parameters = parameters_to_ndarrays(
        arrayrecord_to_parameters(msg.content.array_records[record_key], keep_input=True)
    )
    return [np.asarray(array, dtype=np.float64) for array in parameters]


def _train_secure(msg: Message, context: Context, data: DeskData) -> Message:
    """Logistic FedAvg step; the SecAgg+ mod masks the reply parameters."""
    parameters = _global_parameters(msg, "fitins.parameters")
    updated, loss = logistic_train(
        data.x_train,
        data.y_train,
        parameters,
        epochs=int(context.run_config["local-epochs"]),
        learning_rate=float(context.run_config["learning-rate"]),
    )

    # Only weights + numeric metrics leave the desk; assert no raw identifiers.
    shared_metrics = {
        "train_loss": loss,
        "raw-records-shared": 0,
    }
    validate_shared_payload(shared_metrics)

    fit_res = FitRes(
        status=Status(Code.OK, ""),
        parameters=ndarrays_to_parameters(updated),
        num_examples=len(data.y_train),
        metrics=shared_metrics,
    )
    return Message(
        content=fitres_to_recorddict(fit_res, keep_input=True), reply_to=msg
    )


def _evaluate_secure(msg: Message, context: Context, data: DeskData) -> Message:
    parameters = _global_parameters(msg, "evaluateins.parameters")
    loss = binary_cross_entropy(data.x_test, data.y_test, parameters)
    acc = logistic_accuracy(data.x_test, data.y_test, parameters)

    shared_metrics = {
        "eval_loss": loss,
        "eval_accuracy": acc,
    }
    validate_shared_payload(shared_metrics)

    evaluate_res = EvaluateRes(
        status=Status(Code.OK, ""),
        loss=loss,
        num_examples=len(data.y_test),
        metrics=shared_metrics,
    )
    return Message(content=evaluateres_to_recorddict(evaluate_res), reply_to=msg)


@app.train()
def train(msg: Message, context: Context) -> Message:
    partition_id = int(context.node_config["partition-id"])
    data = _resolve_desk_data(partition_id, persist=True)
    # Vault stays in this process. It is not attached to the reply.
    require_local_vault(data, partition_id)

    if bool(context.run_config["secure-aggregation"]):
        return _train_secure(msg, context, data)

    # Receive global XGBoost model as bytes
    global_model_bytes = msg.content["arrays"].to_numpy_ndarrays()[0].tobytes()

    # Train locally
    updated_bytes, loss = train_xgboost_local(
        data.x_train,
        data.y_train,
        global_model_bytes,
        num_local_trees=int(context.run_config["local-trees"]),
    )

    # Only weights + numeric metrics leave the desk; assert no raw identifiers.
    shared_metrics = {
        "train_loss": loss,
        "num-examples": len(data.y_train),
        "raw-records-shared": 0,
    }
    validate_shared_payload(shared_metrics)

    # Return updated model as bytes array
    content = RecordDict(
        {
            "arrays": ArrayRecord([np.frombuffer(updated_bytes, dtype=np.uint8)]),
            "metrics": MetricRecord(shared_metrics),
        }
    )
    return Message(content=content, reply_to=msg)


@app.evaluate()
def evaluate(msg: Message, context: Context) -> Message:
    partition_id = int(context.node_config["partition-id"])
    data = _resolve_desk_data(partition_id, persist=False)
    # Vault stays in this process. It is not attached to the reply.
    require_local_vault(data, partition_id)

    if bool(context.run_config["secure-aggregation"]):
        return _evaluate_secure(msg, context, data)

    global_model_bytes = msg.content["arrays"].to_numpy_ndarrays()[0].tobytes()
    metrics_dict = evaluate_xgboost(global_model_bytes, data.x_test, data.y_test)

    shared_metrics = {
        "eval_loss": metrics_dict["logloss"],
        "eval_accuracy": metrics_dict["accuracy"],
        "num-examples": len(data.y_test),
    }
    validate_shared_payload(shared_metrics)

    return Message(content=RecordDict({"metrics": MetricRecord(shared_metrics)}), reply_to=msg)
