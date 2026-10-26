"""Orchestrator: sequences all six Vanna agents and builds the final GovernanceContext."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from ..connectors import ConnectorClient
from .contracts import (
    AgentContribution,
    CounterpartyRiskAssessment,
    GovernanceAssessment,
    GovernanceContext,
    LastLookAssessment,
    ManipulationAssessment,
    MarginAssessment,
    MarginContext,
    MarketPatternContext,
)
from .counterparty_risk import CounterpartyRiskAgent
from .governance import GovernanceAgent
from .interfaces import (
    CounterpartyRiskLike,
    GovernanceLike,
    LastLookLike,
    ManipulationWatchLike,
    MarginLike,
    VannaLike,
)
from .last_look import LastLookAgent
from .manipulation_watch import ManipulationWatch
from .margin import MarginAgent
from .vanna import VannaAgent
from ..domain import OrderRequest, ProviderEvidence, Recommendation


class OrchestratorAgent:
    name = "OrchestratorAgent"

    def __init__(self, connectors: ConnectorClient | None = None) -> None:
        self.vanna: VannaLike = VannaAgent()
        self.last_look: LastLookLike = LastLookAgent()
        self.counterparty_risk: CounterpartyRiskLike = CounterpartyRiskAgent()
        self.margin: MarginLike = MarginAgent()
        self.manipulation_watch: ManipulationWatchLike = ManipulationWatch()
        self.governance: GovernanceLike = GovernanceAgent()
        self.connectors = connectors

    def assess(
        self,
        order: OrderRequest,
        evidence: list[ProviderEvidence],
        *,
        now: datetime | None = None,
    ) -> dict[str, Any]:
        current_time = now or datetime.now(UTC)

        # 1. Vanna: execution-value ranking (with live execution history)
        recommendation = self.vanna.assess(order, evidence, connectors=self.connectors, now=current_time)

        # 2. LastLook: conditional rejection asymmetry on displayed-quote leader (with live order flow)
        last_look = self.last_look.assess(order, evidence, connectors=self.connectors)

        # 3. CounterpartyRisk: reliability on recommended provider (with live execution history)
        counterparty_risk = self.counterparty_risk.assess(recommendation, evidence, connectors=self.connectors)

        # 4. Margin: advisory pressure from order context (with live risk metrics)
        margin_context = self._build_margin_context(order, current_time)
        margin = self.margin.assess(margin_context, connectors=self.connectors)

        # 5. ManipulationWatch: market-pattern surveillance on displayed-quote leader (with live surveillance)
        available = [item for item in evidence if item.provider in order.available_providers]
        displayed_quote_leader = max(
            available,
            key=lambda item: item.displayed_price_benefit_bps,
        )
        manipulation_context = self._build_manipulation_context(displayed_quote_leader)
        manipulation = self.manipulation_watch.assess(manipulation_context, connectors=self.connectors)

        # 6. Governance: final decision consuming all assessments (with live federation metrics)
        governance_context = GovernanceContext(
            recommendation=recommendation,
            last_look=last_look,
            counterparty_risk=counterparty_risk,
            margin=margin,
            manipulation=manipulation,
            cohort_size=5,  # from federation artifact
            synchronized_routing_ratio=0.0,  # will be overridden by live metrics if available
            rare_participant_query=False,
            anomalous_model_update=False,
        )
        governance = self.governance.assess(governance_context, connectors=self.connectors)

        # Per-agent visible contributions, in handoff order, for streaming/narration
        contributions = [
            AgentContribution(
                agent=self.vanna.name,
                summary=self.vanna.explain(recommendation),
                assessment=recommendation.model_dump(mode="json"),
            ),
            AgentContribution(
                agent=self.last_look.name,
                summary=self.last_look.explain(last_look),
                assessment=last_look.model_dump(mode="json"),
            ),
            AgentContribution(
                agent=self.counterparty_risk.name,
                summary=self.counterparty_risk.explain(counterparty_risk),
                assessment=counterparty_risk.model_dump(mode="json"),
            ),
            AgentContribution(
                agent=self.margin.name,
                summary=self.margin.explain(margin),
                assessment=margin.model_dump(mode="json"),
            ),
            AgentContribution(
                agent=self.manipulation_watch.name,
                summary=self.manipulation_watch.explain(manipulation),
                assessment=manipulation.model_dump(mode="json"),
            ),
            AgentContribution(
                agent=self.governance.name,
                summary=self.governance.explain(governance),
                assessment=governance.model_dump(mode="json"),
            ),
        ]

        return {
            "recommendation": recommendation,
            "last_look": last_look,
            "counterparty_risk": counterparty_risk,
            "margin": margin,
            "manipulation": manipulation,
            "governance": governance,
            "contributions": contributions,
        }

    def _build_margin_context(self, order: OrderRequest, now: datetime) -> MarginContext:
        """Build MarginContext from order and current market conditions.

        In production this would come from a risk system. For the demo we derive
        sensible defaults from the order's volatility and size bucket.
        """
        volatility_pressure = {"calm": 0.2, "normal": 0.5, "high": 0.8}[order.volatility]
        size_pressure = {"<1m": 0.1, "1m-5m": 0.4, "5m-10m": 0.7, ">10m": 0.9}[order.size_bucket]

        return MarginContext(
            pair=order.pair,
            volatility=order.volatility,
            margin_utilization=min(0.3 + 0.5 * volatility_pressure, 0.95),
            leverage_ratio=5.0 + 15.0 * size_pressure,
            correlated_exposure=0.2 + 0.6 * volatility_pressure,
            settlement_pressure=0.1 + 0.4 * volatility_pressure,
        )

    def _build_manipulation_context(self, leader: ProviderEvidence) -> MarketPatternContext:
        """Build MarketPatternContext from the displayed-quote leader's evidence.

        In production this would come from a surveillance feed. For the demo we
        derive signals from the provider's rejection asymmetry and fill probability.
        """
        # Higher rejection asymmetry and lower fill prob → more suspicious quoting
        quote_activity = 10.0 + 40.0 * abs(leader.rejection_asymmetry)
        cancellation_rate = 0.1 + 0.5 * (1.0 - leader.fill_probability)
        sync_score = 0.1 + 0.4 * abs(leader.rejection_asymmetry)
        cross_pair = 0.05 + 0.3 * (1.0 - leader.fill_probability)
        pre_movement = 0.05 + 0.2 * abs(leader.rejection_asymmetry)

        return MarketPatternContext(
            provider=leader.provider,
            quote_to_trade_ratio=quote_activity,
            cancellation_rate=cancellation_rate,
            synchronized_quote_score=sync_score,
            cross_pair_anomaly_score=cross_pair,
            pre_movement_activity_score=pre_movement,
            sample_count=leader.sample_count,
        )