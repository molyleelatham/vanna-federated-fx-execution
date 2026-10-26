"""Stable agent interfaces for SuperGrid wiring.

Agents implement assess(). An optional connectors argument carries live
SuperLink tools when they are configured; callers omit it on the local path.
"""

from __future__ import annotations

from datetime import datetime
from typing import Protocol

from ..connectors import ConnectorClient
from ..domain import OrderRequest, ProviderEvidence, Recommendation
from .contracts import (
    CounterpartyRiskAssessment,
    GovernanceAssessment,
    GovernanceContext,
    LastLookAssessment,
    ManipulationAssessment,
    MarginAssessment,
    MarginContext,
    MarketPatternContext,
)


class VannaLike(Protocol):
    name: str

    def assess(
        self,
        order: OrderRequest,
        evidence: list[ProviderEvidence],
        connectors: ConnectorClient | None = None,
        *,
        now: datetime | None = None,
    ) -> Recommendation: ...


class LastLookLike(Protocol):
    name: str

    def assess(
        self,
        order: OrderRequest,
        evidence: list[ProviderEvidence],
        connectors: ConnectorClient | None = None,
    ) -> LastLookAssessment: ...


class CounterpartyRiskLike(Protocol):
    name: str

    def assess(
        self,
        recommendation: Recommendation,
        evidence: list[ProviderEvidence],
        connectors: ConnectorClient | None = None,
    ) -> CounterpartyRiskAssessment: ...


class MarginLike(Protocol):
    name: str

    def assess(
        self,
        context: MarginContext,
        connectors: ConnectorClient | None = None,
    ) -> MarginAssessment: ...


class ManipulationWatchLike(Protocol):
    name: str

    def assess(
        self,
        context: MarketPatternContext,
        connectors: ConnectorClient | None = None,
    ) -> ManipulationAssessment: ...


class GovernanceLike(Protocol):
    name: str

    def assess(
        self,
        context: GovernanceContext,
        connectors: ConnectorClient | None = None,
    ) -> GovernanceAssessment: ...


HANDOFF_CHAIN = (
    "Vanna",
    "LastLookAgent",
    "CounterpartyRiskAgent",
    "MarginAgent",
    "ManipulationWatch",
    "GovernanceAgent",
)
