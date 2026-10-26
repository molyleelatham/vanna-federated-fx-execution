"""The local demo must show the handoff without calling SuperGrid."""

import runpy
from pathlib import Path


def test_local_demo_prints_handoff_and_zero_raw_records(capsys) -> None:
    script = Path(__file__).resolve().parents[1] / "scripts" / "local_demo.py"
    runpy.run_path(str(script), run_name="__main__")
    output = capsys.readouterr().out

    assert "does not call SuperGrid" in output
    assert "LP_B" in output
    assert "Handoff:" in output
    assert "Handoff JSON keys: Vanna, LastLookAgent, CounterpartyRiskAgent" in output
    assert "Raw records shared: 0" in output
    assert "client identities shared: 0" in output
