import json
from pathlib import Path

import numpy as np
import pytest

from vanna_federation.data import (
    DeskData,
    generate_desk_data_legacy,
    global_test_data_legacy,
    require_local_vault,
    vault_from_partition,
)
from vanna_federation.model import (
    accuracy,
    binary_cross_entropy,
    initial_parameters,
    train,
)
from vanna_federation.persistence import TrainingManifest, save_desk_partition
from vanna_federation.privacy import validate_shared_payload
from vanna_federation.server_app import export_approved_evidence


def test_five_partitions_are_distinct_and_reproducible() -> None:
    partitions = [generate_desk_data_legacy(index) for index in range(5)]
    assert all(len(partition.y_train) == 360 for partition in partitions)
    assert np.array_equal(partitions[0].x_train, generate_desk_data_legacy(0).x_train)
    assert not np.array_equal(partitions[0].x_train, partitions[1].x_train)


def test_one_fedavg_round_improves_global_model() -> None:
    initial = initial_parameters()
    updates = []
    for index in range(5):
        data = generate_desk_data_legacy(index)
        updated, _ = train(
            data.x_train,
            data.y_train,
            initial,
            epochs=8,
            learning_rate=0.15,
        )
        updates.append(updated)
    aggregated = [
        np.mean([update[array_index] for update in updates], axis=0)
        for array_index in range(len(initial))
    ]
    x_test, y_test = global_test_data_legacy()
    assert binary_cross_entropy(x_test, y_test, aggregated) < binary_cross_entropy(
        x_test, y_test, initial
    )
    assert accuracy(x_test, y_test, aggregated) >= accuracy(x_test, y_test, initial)


def test_desk_vault_stays_local_and_export_has_no_identifiers(tmp_path, monkeypatch) -> None:
    data = generate_desk_data_legacy(0)
    require_local_vault(data, 0)
    assert data.vault.desk_id == 0
    assert data.vault.entries
    first = data.vault.entries[0]
    assert data.vault.reveal(first.hash_id) == first
    assert data.vault.reveal("missing") is None
    assert first.local_order_id.startswith("DESK0-ORD-")
    assert first.uti.startswith("UTI-0-")
    assert first.hash_id != first.local_order_id
    assert first.local_order_id not in first.hash_id

    shared = {
        "train_loss": 0.5,
        "num-examples": len(data.y_train),
        "raw-records-shared": 0,
    }
    validate_shared_payload(shared)
    with pytest.raises(ValueError, match="prohibited"):
        validate_shared_payload({"train_loss": 0.5, "local_order_id": first.local_order_id})
    with pytest.raises(ValueError, match="prohibited"):
        validate_shared_payload({"secret": data.vault.secret, "vault": "desk"})

    empty = DeskData(data.x_train, data.y_train, data.x_test, data.y_test)
    with pytest.raises(RuntimeError, match="desk vault"):
        require_local_vault(empty, 0)

    monkeypatch.chdir(tmp_path)
    from vanna_federation.desk_config import DEFAULT_DESK_CONFIGS

    saved = save_desk_partition(
        0,
        DEFAULT_DESK_CONFIGS[0].to_dict(),
        data.x_train,
        data.y_train,
        data.x_test,
        data.y_test,
    )
    blob = Path(saved).read_bytes()
    assert data.vault.secret.encode() not in blob
    assert first.local_order_id.encode() not in blob
    assert first.uti.encode() not in blob
    assert first.client_id.encode() not in blob
    rebuilt = vault_from_partition(0, data.x_train, data.x_test)
    assert rebuilt.secret == data.vault.secret
    assert rebuilt.entries[0] == first
    assert rebuilt.reveal(first.hash_id) == first

    path = export_approved_evidence(
        np.array([0.30, 0.88, 0.11]),
        "fed-vault-test",
        TrainingManifest(),
    )
    evidence = path.read_text(encoding="utf-8")
    payload = json.loads(evidence)
    assert first.local_order_id not in evidence
    assert first.uti not in evidence
    assert first.client_id not in evidence
    assert first.hash_id not in evidence
    assert data.vault.secret not in evidence
    assert payload["raw_records_shared"] == 0
    assert payload["client_identities_shared"] == 0
