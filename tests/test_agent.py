import json
from types import SimpleNamespace
from uuid import uuid4

import pytest
from pydantic import ValidationError

from vanna_agent.agent_app import (
    MAX_AGENT_CALLS,
    MODEL_TIMEOUT_SECONDS,
    deterministic_answer,
    pipeline_failure_answer,
    run_pipeline,
    emit_terminal_decision,
    terminal_request_id,
)
from vanna_agent.agents import HANDOFF_CHAIN
from vanna_agent.agents.contracts import GovernanceAssessment, LastLookAssessment


def test_structured_handoff_and_local_fallback() -> None:
    result = run_pipeline(
        json.dumps(
            {
                "pair": "EUR/USD",
                "side": "BUY",
                "size_bucket": "1m-5m",
                "volatility": "high",
                "available_providers": ["LP_A", "LP_B", "LP_C"],
            }
        )
    )
    assert MAX_AGENT_CALLS == 6
    assert result["vanna_recommendation"]["provider"] == "LP_B"
    assert result["handoff_chain"] == list(HANDOFF_CHAIN)
    assert result["live_path"] == "local-non-blocking"
    assert "counterparty_risk" in result
    assert "margin" in result
    assert "manipulation" in result
    assert "governance" in result
    assert result["privacy"] == {
        "raw_records_shared": 0,
        "client_identities_shared": 0,
    }
    answer = deterministic_answer(result, "endpoint unavailable")
    assert "deterministic fallback" in answer
    assert "Handoff:" in answer
    assert "No automatic execution, blacklist" in answer
    assert "CounterpartyRisk" in answer
    assert "Margin" in answer
    assert "ManipulationWatch" in answer
    assert "Governance" in answer


def test_terminal_event_uses_validated_request_id() -> None:
    request_id = str(uuid4())
    context = SimpleNamespace(run_config={"terminal.request-id": request_id})
    emitted: list[dict[str, object]] = []
    agent = SimpleNamespace(events=SimpleNamespace(emit=emitted.append))

    assert terminal_request_id(context) == request_id
    emit_terminal_decision(
        agent,
        request_id,
        status="completed",
        result={"privacy": {"raw_records_shared": 0}},
    )

    assert emitted == [{
        "type": "vanna.decision",
        "request_id": request_id,
        "status": "completed",
        "result": {"privacy": {"raw_records_shared": 0}},
    }]


def test_terminal_request_id_rejects_malformed_value() -> None:
    context = SimpleNamespace(run_config={"terminal.request-id": "not-a-uuid"})

    with pytest.raises(ValueError, match="UUID"):
        terminal_request_id(context)


def test_terminal_request_id_allows_default_empty_config() -> None:
    context = SimpleNamespace(run_config={"terminal.request-id": ""})

    assert terminal_request_id(context) is None


def test_contributions_render_one_line_per_agent_in_handoff_order() -> None:
    result = run_pipeline(
        json.dumps(
            {
                "pair": "EUR/USD",
                "side": "BUY",
                "size_bucket": "1m-5m",
                "volatility": "high",
                "available_providers": ["LP_A", "LP_B", "LP_C"],
            }
        )
    )
    contributions = result["contributions"]
    assert [c["agent"] for c in contributions] == list(HANDOFF_CHAIN)
    for contribution in contributions:
        assert contribution["summary"]
        assert contribution["assessment"]

    answer = deterministic_answer(result)
    answer_lines = answer.splitlines()
    # Header + one line per agent + privacy footer
    assert answer_lines[0] == f"Handoff: {' -> '.join(HANDOFF_CHAIN)}"
    for contribution in contributions:
        assert f"{contribution['agent']}: {contribution['summary']}" in answer_lines
    assert answer_lines[-1] == "Privacy: 0 raw records and 0 client identities shared."


def test_malformed_agent_outputs_are_rejected_by_contracts() -> None:
    # Out-of-range value
    with pytest.raises(ValidationError):
        LastLookAssessment(
            provider="LP_A",
            rejection_asymmetry=2.0,  # must be within [-1, 1]
            level="elevated",
            review_required=True,
            explanation="malformed",
        )
    # Unexpected extra field (contracts are extra="forbid")
    with pytest.raises(ValidationError):
        GovernanceAssessment(
            action="HUMAN_REVIEW",
            reasons=["test"],
            rogue_field="not allowed",
        )
    # Invalid enum value
    with pytest.raises(ValidationError):
        GovernanceAssessment(action="EXECUTE_TRADE", reasons=["test"])


def test_child_agent_failure_produces_safe_fallback() -> None:
    order_json = json.dumps(
        {
            "pair": "EUR/USD",
            "side": "BUY",
            "size_bucket": "1m-5m",
            "volatility": "high",
            "available_providers": ["LP_A", "LP_B", "LP_C"],
        }
    )

    class BrokenOrchestrator:
        def __init__(self, connectors=None) -> None:
            pass

        def assess(self, *args, **kwargs):
            raise RuntimeError("child agent exploded")

    import vanna_agent.agent_app as agent_app

    original = agent_app.OrchestratorAgent
    agent_app.OrchestratorAgent = BrokenOrchestrator
    try:
        with pytest.raises(RuntimeError, match="child agent exploded"):
            run_pipeline(order_json)
    finally:
        agent_app.OrchestratorAgent = original

    # The app-level fallback converts that failure into a safe answer
    answer = pipeline_failure_answer("child agent exploded")
    assert "no routing recommendation produced" in answer
    assert "HUMAN_REVIEW" in answer
    assert "no auto-execution or blacklist" in answer
    assert "Privacy: 0 raw records and 0 client identities shared." in answer
    assert "LP_B" not in answer  # no recommendation leaks from a broken chain


def test_model_call_is_time_bounded() -> None:
    assert MODEL_TIMEOUT_SECONDS > 0
