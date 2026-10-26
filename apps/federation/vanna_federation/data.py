"""Reproducible synthetic FX histories for isolated desks."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

from .desk_config import DeskConfig, DEFAULT_DESK_CONFIGS
from .privacy import bucket_latency, bucket_notional, pseudonymize

PROVIDERS = ("LP_A", "LP_B", "LP_C")
FEATURE_NAMES = (
    "is_lp_a",
    "is_lp_b",
    "is_lp_c",
    "is_high_volatility",
    "is_lp_a_high_volatility",
    "is_lp_c_high_volatility",
    "size_scaled",
    "quote_age_scaled",
)

_SIZE_COLUMN = FEATURE_NAMES.index("size_scaled")
_QUOTE_AGE_COLUMN = FEATURE_NAMES.index("quote_age_scaled")


@dataclass(frozen=True)
class LocalAuditEntry:
    hash_id: str
    local_order_id: str
    uti: str
    client_id: str
    notional_bucket: str
    latency_bucket: str


@dataclass(frozen=True)
class DeskVault:
    """In-memory audit map. Never serialize this into a Flower message."""

    desk_id: int
    secret: str
    entries: tuple[LocalAuditEntry, ...] = ()

    def reveal(self, hash_id: str) -> LocalAuditEntry | None:
        for entry in self.entries:
            if entry.hash_id == hash_id:
                return entry
        return None


@dataclass(frozen=True)
class DeskData:
    x_train: NDArray[np.float64]
    y_train: NDArray[np.float64]
    x_test: NDArray[np.float64]
    y_test: NDArray[np.float64]
    vault: DeskVault = field(default_factory=lambda: DeskVault(desk_id=-1, secret=""))


def build_desk_vault(
    partition_id: int,
    size_scaled: NDArray[np.float64],
    quote_age_scaled: NDArray[np.float64],
) -> DeskVault:
    """HMAC-pseudonymize local order ids. The secret and raw ids stay on the desk."""
    if size_scaled.shape != quote_age_scaled.shape:
        raise ValueError("size and quote-age series must match")
    secret = f"desk-{partition_id}-local-secret"
    entries: list[LocalAuditEntry] = []
    for index, (size, quote_age) in enumerate(
        zip(size_scaled, quote_age_scaled, strict=True)
    ):
        local_order_id = f"DESK{partition_id}-ORD-{index:06d}"
        entries.append(
            LocalAuditEntry(
                hash_id=pseudonymize(secret, local_order_id),
                local_order_id=local_order_id,
                uti=f"UTI-{partition_id}-{index:06d}",
                client_id=f"CLIENT-{partition_id}-{(index % 20):03d}",
                notional_bucket=bucket_notional(float(size) * 10_000_000),
                latency_bucket=bucket_latency(20.0 + 90.0 * float(quote_age)),
            )
        )
    return DeskVault(desk_id=partition_id, secret=secret, entries=tuple(entries))


def vault_from_partition(
    partition_id: int,
    x_train: NDArray[np.float64],
    x_test: NDArray[np.float64],
) -> DeskVault:
    """Rebuild the in-memory vault from features. It is not stored on disk."""
    features = np.concatenate([x_train, x_test])
    return build_desk_vault(
        partition_id,
        features[:, _SIZE_COLUMN],
        features[:, _QUOTE_AGE_COLUMN],
    )


def require_local_vault(data: DeskData, partition_id: int) -> None:
    """Check the vault belongs to this desk and stays out of shared replies."""
    vault = data.vault
    if vault.desk_id != partition_id or not vault.secret or not vault.entries:
        raise RuntimeError("desk vault does not match partition")


def _sigmoid(value: NDArray[np.float64]) -> NDArray[np.float64]:
    return 1.0 / (1.0 + np.exp(-np.clip(value, -30, 30)))


def generate_desk_data(
    config: DeskConfig,
    samples: int = 450,
) -> DeskData:
    """Generate one private partition from a DeskConfig."""
    rng = np.random.default_rng(20260826 + config.partition_id)
    
    # Provider mix (desk-specific, rotated)
    provider = rng.choice(3, size=samples, p=config.rotated_provider_probs)
    high_volatility = rng.binomial(1, config.volatility_prob, samples)
    size_scaled = rng.uniform(0.05, 1.0, samples)
    quote_age_scaled = rng.uniform(0.0, 1.0, samples)
    one_hot = np.eye(3, dtype=np.float64)[provider]
    x = np.column_stack(
        (
            one_hot,
            high_volatility,
            one_hot[:, 0] * high_volatility,
            one_hot[:, 2] * high_volatility,
            size_scaled,
            quote_age_scaled,
        )
    )

    # Desk-specific logit coefficients from config
    logits = (
        config.base_logit
        + config.lp_a_coeff * one_hot[:, 0]
        + config.lp_b_coeff * one_hot[:, 1]
        + config.lp_c_coeff * one_hot[:, 2]
        + config.volatility_coeff * high_volatility
        + config.lp_a_vol_interaction * one_hot[:, 0] * high_volatility
        + config.lp_c_vol_interaction * one_hot[:, 2] * high_volatility
        + config.size_coeff * size_scaled
        + config.quote_age_coeff * quote_age_scaled
        + rng.normal(0, config.noise_std, samples)
    )
    y = rng.binomial(1, _sigmoid(logits)).astype(np.float64)
    split = int(samples * 0.8)
    vault = build_desk_vault(config.partition_id, size_scaled, quote_age_scaled)
    return DeskData(x[:split], y[:split], x[split:], y[split:], vault=vault)


def generate_all_desks(
    configs: list[DeskConfig] | None = None,
    samples: int = 450,
) -> list[DeskData]:
    """Generate data for all configured desks."""
    if configs is None:
        configs = DEFAULT_DESK_CONFIGS
    return [generate_desk_data(config, samples) for config in configs]


def global_test_data(configs: list[DeskConfig] | None = None) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Global test data from all desks."""
    desks = generate_all_desks(configs)
    return (
        np.concatenate([desk.x_test for desk in desks]),
        np.concatenate([desk.y_test for desk in desks]),
    )


# Backward compatibility
def generate_desk_data_legacy(partition_id: int, samples: int = 450) -> DeskData:
    """Legacy function for backward compatibility."""
    config = DEFAULT_DESK_CONFIGS[partition_id]
    return generate_desk_data(config, samples)


def global_test_data_legacy() -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Legacy function for backward compatibility."""
    return global_test_data(DEFAULT_DESK_CONFIGS)