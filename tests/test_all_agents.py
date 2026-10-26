from datetime import UTC, datetime

from vanna_agent.agents import (
    AGENT_REGISTRY,
    HANDOFF_CHAIN,
    CounterpartyRiskAgent,
    GovernanceAgent,
    LastLookAgent,
    ManipulationWatch,
    MarginAgent,
    OrchestratorAgent,
    VannaAgent,
)
from vanna_agent.agents.interfaces import (
    CounterpartyRiskLike,
    GovernanceLike,
    LastLookLike,
    ManipulationWatchLike,
    MarginLike,
    VannaLike,
)
from vanna_agent.agents.contracts import (
    GovernanceContext,
    MarginContext,
    MarketPatternContext,
)
from vanna_agent.domain import OrderRequest, ProviderEvidence


def provider(
    name: str,
    fill: float,
    slippage: float,
    latency: float,
    benefit: float,
    asymmetry: float,
) -> ProviderEvidence:
    return ProviderEvidence(
        provider=name,
        sample_count=450,
        fill_probability=fill,
        rejection_probability=1 - fill,
        expected_slippage_bps=slippage,
        expected_latency_ms=latency,
        displayed_price_benefit_bps=benefit,
        rejection_asymmetry=asymmetry,
        model_version="test-fed-v1",
        generated_at=datetime(2026, 8, 26, 11, 30, tzinfo=UTC),
    )


def scenario():
    order = OrderRequest(
        pair="EUR/USD",
        side="BUY",
        size_bucket="1m-5m",
        volatility="high",
        available_providers=["LP_A", "LP_B", "LP_C"],
    )
    evidence = [
        provider("LP_A", 0.42, 1.10, 78, 0.45, 0.22),
        provider("LP_B", 0.78, 0.42, 31, 0.10, 0.03),
        provider("LP_C", 0.51, 0.84, 86, 0.20, 0.08),
    ]
    recommendation = VannaAgent().assess(
        order,
        evidence,
        now=datetime(2026, 8, 26, 11, 35, tzinfo=UTC),
    )
    return order, evidence, recommendation


def test_orchestrator_wires_main_agent_interfaces() -> None:
    orchestrator = OrchestratorAgent()
    assert isinstance(orchestrator.vanna, VannaAgent)
    assert isinstance(orchestrator.last_look, LastLookAgent)
    assert isinstance(orchestrator.counterparty_risk, CounterpartyRiskAgent)
    assert isinstance(orchestrator.margin, MarginAgent)
    assert isinstance(orchestrator.manipulation_watch, ManipulationWatch)
    assert isinstance(orchestrator.governance, GovernanceAgent)
    vanna: VannaLike = orchestrator.vanna
    last_look: LastLookLike = orchestrator.last_look
    counterparty: CounterpartyRiskLike = orchestrator.counterparty_risk
    margin: MarginLike = orchestrator.margin
    manipulation: ManipulationWatchLike = orchestrator.manipulation_watch
    governance: GovernanceLike = orchestrator.governance
    assert vanna.name == "Vanna"
    assert last_look.name == "LastLookAgent"
    assert counterparty.name == "CounterpartyRiskAgent"
    assert margin.name == "MarginAgent"
    assert manipulation.name == "ManipulationWatch"
    assert governance.name == "GovernanceAgent"
    assert HANDOFF_CHAIN[0] == "Vanna"
    assert HANDOFF_CHAIN[-1] == "GovernanceAgent"


def test_registry_contains_every_documented_agent() -> None:
    assert set(AGENT_REGISTRY) == {
        "Vanna",
        "LastLookAgent",
        "CounterpartyRiskAgent",
        "MarginAgent",
        "ManipulationWatch",
        "GovernanceAgent",
        "OrchestratorAgent",
    }


def test_execution_last_look_and_counterparty_agents() -> None:
    order, evidence, recommendation = scenario()
    assert recommendation.provider == "LP_B"

    last_look = LastLookAgent().assess(order, evidence)
    assert last_look.provider == "LP_A"
    assert last_look.review_required is True

    counterparty = CounterpartyRiskAgent().assess(recommendation, evidence)
    assert counterparty.provider == "LP_B"
    assert counterparty.reliability_score > 0.65
    assert counterparty.automatic_exclusion is False


def test_margin_agent_is_advisory() -> None:
    assessment = MarginAgent().assess(
        MarginContext(
            pair="EUR/USD",
            volatility="high",
            margin_utilization=0.92,
            leverage_ratio=24,
            correlated_exposure=0.86,
            settlement_pressure=0.81,
        )
    )
    assert assessment.pressure == "high"
    assert assessment.recommended_size_multiplier == 0.5
    assert assessment.human_review_required is True
    assert assessment.automatic_liquidation is False


def test_manipulation_watch_never_makes_a_misconduct_finding() -> None:
    assessment = ManipulationWatch().assess(
        MarketPatternContext(
            provider="LP_A",
            quote_to_trade_ratio=48,
            cancellation_rate=0.94,
            synchronized_quote_score=0.86,
            cross_pair_anomaly_score=0.79,
            pre_movement_activity_score=0.88,
            sample_count=500,
        )
    )
    assert assessment.signal == "review"
    assert assessment.human_review_required is True
    assert assessment.misconduct_finding is False


def test_governance_suppresses_collective_routing_output() -> None:
    order, evidence, recommendation = scenario()
    last_look = LastLookAgent().assess(order, evidence)
    counterparty = CounterpartyRiskAgent().assess(recommendation, evidence)
    margin = MarginAgent().assess(
        MarginContext(
            pair="EUR/USD",
            volatility="normal",
            margin_utilization=0.3,
            leverage_ratio=4,
            correlated_exposure=0.2,
            settlement_pressure=0.1,
        )
    )
    manipulation = ManipulationWatch().assess(
        MarketPatternContext(
            provider="LP_A",
            quote_to_trade_ratio=4,
            cancellation_rate=0.1,
            synchronized_quote_score=0.1,
            cross_pair_anomaly_score=0.1,
            pre_movement_activity_score=0.1,
            sample_count=500,
        )
    )
    decision = GovernanceAgent().assess(
        GovernanceContext(
            recommendation=recommendation,
            last_look=last_look,
            counterparty_risk=counterparty,
            margin=margin,
            manipulation=manipulation,
            cohort_size=5,
            synchronized_routing_ratio=0.8,
        )
    )
    assert decision.action == "SUPPRESS_COLLECTIVE_OUTPUT"
    assert decision.collective_instruction is False
    assert decision.automatic_blacklist is False
    assert decision.automatic_execution is False


def test_explain_methods_narrate_their_own_assessments() -> None:
    order, evidence, recommendation = scenario()

    vanna_text = VannaAgent().explain(recommendation)
    assert "LP_B" in vanna_text
    assert f"{recommendation.expected_cost_bps:.2f} bps" in vanna_text

    last_look = LastLookAgent().assess(order, evidence)
    last_look_text = LastLookAgent().explain(last_look)
    assert "LP_A" in last_look_text
    assert "review required: yes" in last_look_text

    counterparty = CounterpartyRiskAgent().assess(recommendation, evidence)
    counterparty_text = CounterpartyRiskAgent().explain(counterparty)
    assert str(counterparty.reliability_score) in counterparty_text
    assert counterparty.route_posture in counterparty_text
    assert "no automatic exclusion" in counterparty_text

    margin = MarginAgent().assess(
        MarginContext(
            pair="EUR/USD",
            volatility="high",
            margin_utilization=0.92,
            leverage_ratio=24,
            correlated_exposure=0.86,
            settlement_pressure=0.81,
        )
    )
    margin_text = MarginAgent().explain(margin)
    assert "high" in margin_text
    assert "0.50" in margin_text
    assert "No automatic liquidation" in margin_text

    manipulation = ManipulationWatch().assess(
        MarketPatternContext(
            provider="LP_A",
            quote_to_trade_ratio=48,
            cancellation_rate=0.94,
            synchronized_quote_score=0.86,
            cross_pair_anomaly_score=0.79,
            pre_movement_activity_score=0.88,
            sample_count=500,
        )
    )
    manipulation_text = ManipulationWatch().explain(manipulation)
    assert "review" in manipulation_text
    assert "not a misconduct finding" in manipulation_text

    governance = GovernanceAgent().assess(
        GovernanceContext(
            recommendation=recommendation,
            last_look=last_look,
            counterparty_risk=counterparty,
            margin=margin,
            manipulation=manipulation,
            cohort_size=5,
            synchronized_routing_ratio=0.0,
        )
    )
    governance_text = GovernanceAgent().explain(governance)
    assert governance.action in governance_text
    assert "No automatic execution, blacklist, or collective instruction" in governance_text


def test_orchestrator_collects_contributions_in_handoff_order() -> None:
    order, evidence, _ = scenario()
    result = OrchestratorAgent().assess(
        order,
        evidence,
        now=datetime(2026, 8, 26, 11, 35, tzinfo=UTC),
    )
    contributions = result["contributions"]
    assert [c.agent for c in contributions] == [
        "Vanna",
        "LastLookAgent",
        "CounterpartyRiskAgent",
        "MarginAgent",
        "ManipulationWatch",
        "GovernanceAgent",
    ]
    for contribution in contributions:
        assert contribution.summary
        assert contribution.assessment
    # Governance contribution carries the final decision payload
    assert contributions[-1].assessment["action"] == result["governance"].action
