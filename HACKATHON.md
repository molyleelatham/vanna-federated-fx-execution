# Vanna at the Flower Cambridge Federated AI hackathon

**Result: top 5.**

This is the write-up of the hackathon submission. For a guide to the code, see [README.md](README.md).

## The event

The project was built at the Flower Cambridge Federated AI hackathon on 26 August 2026. The commit history and the checked-in evidence file are both dated that day, in UK time.

Two tracks were entered:

- **Track 1, SuperGrid.** Show that collaboration changes what an agent can conclude.
- **Track 2, Infrastructure.** A collaboration layer that can be reused: typed evidence between Flower apps, an optional secure-aggregation mode, and a human approval step.

The runs used Flower SuperGrid and the hackathon's AMD-hosted Qwen model behind a Responses-compatible API. The gateway's default model id for that endpoint is `/models/Qwen3.5-397B-A17B-FP8`.

## What was submitted

Five simulated FX desks learn which liquidity provider actually fills, without any desk sharing a raw order. A six-agent Flower chain turns that shared evidence into a governed recommendation, and a person remains in control.

The submission had four parts.

- **A SuperGrid federation.** Five SuperNodes, one per desk, on the `@molyleela/Vanna` deployment federation, training federated XGBoost (`FedXgbBagging`). The start scripts pass a private key per desk, a distinct port, and Flower's runtime dependency installation flag. The SuperLink address in those scripts is `fleet-supergrid.flower.ai:443`. Melanie ran the five SuperNodes.
- **An optional SecAgg+ mode**, switched on with `secure-aggregation=true`. A conditional client mod applies `secaggplus_mod` only when the message carries SecAgg+ config, so the XGBoost path and the secure path share one ClientApp. Secure mode trains a NumPy logistic model with FedAvg, because a summation protocol cannot merge XGBoost trees. The run config sets `num-shares` to 5 and `reconstruction-threshold` to 3. The strategy still requires five available clients.
- **A six-agent chain** — Vanna, LastLook, CounterpartyRisk, Margin, ManipulationWatch, Governance — running as a Flower AgentApp. Qwen narrated the finished chain through six tools, at most three turns, with a 120-second timeout. If the model call failed, a deterministic fallback printed the pipeline's own result.
- **Unsafe outputs made unrepresentable.** On the agent contracts, `automatic_execution`, `automatic_blacklist`, `collective_instruction` and `misconduct_finding` are `Literal[False]`, so an unsafe value fails validation.

## An example from the day

The checked-in evidence file, `apps/agent/vanna_agent/artifacts/provider_evidence.json`, was generated on 26 August 2026. It is synthetic. LP_A's displayed price benefit is the fixed value 0.45 bps, and LP_C's federated fill probability is 0.105751. Scoring that file with `expected_execution_cost` in `decisioning.py` returns 5.5868 bps for LP_A and 2.1113 bps for LP_B, so LP_B ranks ahead. Last-look treats LP_A's rejection asymmetry as a review signal.

The submission described the same moment in round numbers: LP_A showed the best displayed quote and an executable cost of about 6.0 bps, and the chain preferred LP_B at about 2.1 bps. Those rounded figures, and the table below, are illustrative example results from the 26 August 2026 run on synthetic data. They are not benchmarks.

| What was reported | Illustrative example from 26 August 2026 |
|---|---|
| Federation on SuperGrid | Run `12309076582906127164`: 3 rounds, 5/5 nodes, 0 failures, 0 raw records, about 2.5 minutes |
| AgentApp on SuperGrid | Runs `1896158749138907396` and `17460595724030593185`: full chain, about 22 seconds |
| Live Qwen narration | Local SuperLink runs, about 65 seconds, with a full six-agent write-up. The README of that day also recorded about 69 seconds for one narration |
| SecAgg+ mode | Simulation: 3 rounds, 5/5 nodes, 16.4 seconds, loss 0.6931 to 0.6528 |
| One desk against five | Desk 0 alone: LP_C fills about 60%. Five desks federated: about 11%. Held-out loss on LP_B: 0.50 federated, 0.62 single-desk |
| LP_A against LP_B | Displayed benefit +0.45 bps for LP_A, executable cost about 6.0 bps, against about 2.1 bps for LP_B |
| Tests at submission | 46 passing |

The federated "about 11%" is the write-up's rounding of the 0.105751 stored in the evidence file. The desk-0-only "about 60%", the about 6.0 bps, the run IDs and the timings are the figures reported from that day's run. The loss figures for SecAgg+ and for LP_B are from the same report. The training manifest checked in with the evidence file records a local XGBoost run (`fed-xgb-20260826-161034`), not the SuperGrid run id.

## What the model and the agents do

- **Federated XGBoost.** Each desk adds locally trained trees and the server merges boosters with `FedXgbBagging`. `CheckpointingFedXgbBagging` saves each round and writes `artifacts/training_manifest.json`.
- **Where the evidence numbers come from.** Fill probability comes from the federated ensemble. Slippage, latency and rejection probability come from XGBoost regressions whose targets are synthetic desk profiles. Displayed price benefit, rejection asymmetry and sample count are constants on those profiles. The provenance block says so.
- **A one-desk comparison.** `build_model_comparison` trains on desk 0's saved partition only, then scores that model and the five-desk ensemble on desk 0's held-out rows. That is a comparison on one desk's own split.
- **Feature importance.** The saved export is led by `is_high_volatility`, then by `is_lp_a_high_volatility`.
- **The language model narrates.** Every cost, confidence grade and governance action is computed before the model is called. The model explains the finished chain. A failed call falls back to the deterministic text.

## What stays on the desk

Client apps return model bytes and metrics. `validate_shared_payload`, in both `packages/vanna-core` and the federation app, rejects prohibited fields before a payload is shared. The exported file sets `raw_records_shared` and `client_identities_shared` to 0. The AgentApp scans the file again before the agents read it.

Shared analytics can use an HMAC-SHA256 pseudonym of a local order id, and bucketed notional and latency. The mapping back to a client stays on the desk. With `secure-aggregation=true`, the server is meant to see only the reconstructed weighted sum of desk updates.

Governance, in `apps/agent/vanna_agent/agents/governance.py`, can suppress collective output. A cohort smaller than 3, or evidence older than 900 seconds, selects the local fallback. A synchronised-routing ratio of 0.60 or higher, or a query that would isolate a rare participant, suppresses collective output. Last-look review, low confidence, margin pressure and the manipulation watch can each force human review on their own. `collective_instruction` cannot be set to true.

## Problems solved on the day

- **SuperGrid on Flower 1.35.0.** Five one-command scripts stage a desk partition and start a SuperNode with its own key, port and partition id. The 14:55 commit records that five nodes were connected to SuperGrid.
- **A quiet failure in narration.** The endpoint was rejecting a chat-completions tool schema, and the run was falling back without saying why. The tool list is now the flat Responses format, and a tool whose live data is missing returns `"available": false`.
- **Fallback constants leaking into scores.** Agents now blend connector data only when the connector reports that the data is live. Otherwise they keep the evidence file as it stands.

## The day, from the commit history

Times are UK (UTC+1).

| When | What landed |
|---|---|
| 11:57–12:34 | Repository created. Early notes, then an agent guide. |
| 12:47–13:28 | Flower prototype, the full agent suite, and `OrchestratorAgent` wiring all six agents. |
| 13:35–14:05 | SuperNode notes for Melanie, the Alpha Vantage connector, and configurable desk simulations. |
| 14:12–14:37 | Switch to `FedXgbBagging`, checkpointing and a training manifest, then evidence taken from XGBoost regression rather than hard-coded provider scores. |
| 14:42–15:08 | SuperNode scripts for Flower 1.35.0. The history records five nodes on SuperGrid. Desk-local privacy and the agent hand-off followed. |
| 15:45–16:28 | Federation evidence generated. Model-id override for SuperGrid AgentApp runs. Per-agent `explain()`. |
| 16:41–17:16 | Responses API tool-format fix, SecAgg+ mode, reliability fixes, then the approval gateway and dashboard. |
| 17:23–18:50 | Further notes on the design, gateway and dashboard tests, and internal notes removed from the tree. |
| 27 August, 22:46 | The first version of this write-up. |

## Tracks

**Track 1.** The agent chain runs on evidence assembled from five desks, on the `@molyleela/Vanna` federation. Each agent contributes one line: cost, rejection asymmetry, reliability, margin, surveillance, then the governance action. The illustrative comparison from 26 August is the one in the table above, including the desk-0-only LP_C estimate of about 60%.

**Track 2.** The two Flower apps meet at one typed evidence file. SecAgg+ is a run-config flag on the same ClientApp. The local gateway turns a governed decision into a `PENDING_HUMAN_APPROVAL` audit record and does not send an order.

## Limitations at submission

- All desk data is synthetic. Displayed price benefit, rejection asymmetry and sample count are fixed values, and the provenance block says so.
- SecAgg+ uses a logistic model. The default path still aggregates XGBoost trees where the server can see the merged model. The code does not add differential privacy.
- The prototype recommends. It does not execute a trade, blacklist a provider, or instruct desks together.
- The run IDs, timings, the about-60% local LP_C estimate and the about-6.0 bps LP_A cost are illustrative examples from the 26 August 2026 synthetic run, not benchmarks.
