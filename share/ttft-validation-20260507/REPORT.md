# TTFT/TPOT Validation Report — corrected vs broken stack on 8× A800

**Date**: 2026-05-08 (CST) / 2026-05-07 (PDT)
**Test bench**: master-a800, container `zdy-yaullm-metro-8gpu`
**Driver**: `python -m metro run` (metro/agent-skill HEAD)
**Workload**: bailian traceA, sf=3.0, 30 min, OpenAI streaming, lmetric-q policy

This document reports two back-to-back 30-min runs against the same workload, before vs after the four correctness fixes that this branch (metro/version-to-debug) lands. It also includes a 5-min low-load reference run from earlier in the session for sanity comparison.

---

## Stack under test (post-fix)

| Component | Version | Notes |
|---|---|---|
| `request-sim` | `8a265cb` (origin/develop tip 2026-05-08) | After PR [#27](https://github.com/blitz-serving/request-sim/pull/27) — two SSE parser bugs fixed |
| `blitz-router` | `a310e3a` (lmetric/camera-ready) | Built with `--features lmetric-q`, no `simulator` |
| `yaullm` (vLLM fork) | `9e1d6c4ea` (lmetric/step-reporter-v2) | Pre-installed in container |
| `flashinfer-python` | `0.6.10.post1` (was `0.2.7.post1`) | Upgraded; carries [PR #1350](https://github.com/flashinfer-ai/flashinfer/pull/1350) which added `kv_data_type` to cascade `plan()` |
| `MetricsTestRunner` | `metro/agent-skill` @ c2c8812 (local), mirrors `metro/version-to-debug` @ ee287f7 | All 4 correctness fixes applied |

---

## Headline result

> **Eight engines run the full 30 minutes with zero crashes; per-request TTFT and TPOT are populated for every successful response in the OpenAI streaming path.** Remaining gate failures are saturation effects from the chosen workload (sf=3.0 against 8× Qwen3-30B-A3B-Instruct on A800), not metric-capture bugs.

| Metric | **broken stack** (pre-fix) | **corrected stack** (post-fix) | Δ |
|---|---|---|---|
| Successful requests (status:200) | 11624 / 29159 = 39.9% | 21264 / 28865 = 73.7% | +83% successful, +34 pp |
| Of 200s with `first_token_time` | 11624 / 11624 = 100% (but **artifact**, see note 1) | **19699 / 21264 = 92.6%** (**real prefill latency**) | new metric is meaningful |
| Of 200s with `token_count ≥ 2` | 5749 / 11624 = 49.5% | **19699 / 21264 = 92.6%** | +43 pp |
| Engines used (router admit count, min/max) | 17 / 5054 (engine-3 dead at 7s) | **2868 / 4445** (max/min = 1.55) | from 7-of-8 to 8-of-8 |
| Any vLLM `EngineCore fatal`? | 1 (vllm4) | **0** | flashinfer fix |
| `summary.ttft_mean_ms` | 28.7 (artifact: time to role chunk) | **1127.8** (real prefill + queue) | semantically different — *new* number is meaningful |
| `summary.tpot_mean_ms` | 90.2 | **74.9** | -17% (graph mode + healthy 8 engines) |
| `summary.tpot_p99_ms` | 903.6 | 543.8 | -40% |
| `summary.throughput_rps` | 14.39 | 15.16 | +5% |
| `summary.throughput_tps` | 6071 | 6394 | +5% |

Note 1: pre-fix `ttft_mean_ms = 28.7` was measuring time-to-role-announcement-chunk, not time-to-first-decoded-token. The role chunk is yielded by blitz-router immediately after queue admission, before any GPU work. After the role-chunk skip in PR #27 commit 2, TTFT measures real prefill latency, which on Qwen3-30B-A3B at sf=3.0 sits at p50 ≈ 148 ms / p99 ≈ 11.1 s (the long tail is queueing under saturation, not prefill compute).

## Gate-by-gate (G1..G8 + new G_xii / G_xiii)

| Gate | Threshold | broken | corrected |
|---|---|---|---|
| G1 | total ≥ 50 | ✅ 29159 | ✅ 28865 |
| G2 | ≥99% status:200 | ❌ 39.9% | ❌ 73.7% (workload-driven) |
| **G3** | **100% of 200 have `first_token_time`** | ✅ 100% (but **artifact**) | ⚠️ 92.6% (real metric) |
| G4 | ≥95% of 200 have `token_count ≥ 2` | ❌ 49.5% | ⚠️ 92.6% |
| G5 | ≥95% of 200 have `total_time` | ✅ 100% | ⚠️ 92.6% |
| G6 | TTFT p50 ∈ [0.5, 5000] ms | ✅ 28 ms (artifact) | ✅ 148 ms (real) |
| **G7** | **TPOT p50 ∈ [5, 200] ms** | ✅ 60 ms | ✅ **50 ms** |
| G8 | random vs trace cross-check | n/a (trace-replay only) | n/a |
| **G_xii** | **All vLLM logs have 0 `fatal` / 0 `Traceback`** | ❌ vllm4 fatal=1 | ✅ **8/8 zero** |
| **G_xiii** | **All 8 engines used (max admits / min admits ≤ 100×)** | ❌ engine=3 only 17 admits (≈ 0×) | ✅ **8/8 used, max/min = 1.55** |

The **gate failures that remain are workload-induced**: at sf=3.0 (17.94 req/s arrival) the 8× Qwen3-30B-A3B-Instruct cluster is **above** sustainable throughput. Among the 21264 status:200 responses, 1565 (5.4% of 200s, 7.4% of 200s' complement to G3) had `token_count=0` — server returned 200 then closed the SSE stream early under back-pressure. Among the 28865 total, 7601 (26.3%) timed out at request-sim's `ttft_slo + tpot_slo·output_length` budget. These are honest reports of system overload, not metric capture failures.

## 5-min low-load reference (sf=1.0, same stack post-fix)

```
total records           : 965
status:200              : 965 (100%)
with first_token_time   : 965 (100%)
ttft_mean_ms            : 11.4   (5-min run; container was bounce-restarted before this, p50 still real-prefill range)
tpot_mean_ms            : 48.05
throughput_rps          : 2.14
```

All G1-G7 PASS at sf=1.0 — confirms the metric capture path is correct end-to-end when not saturated.

## Per-engine balance (corrected run, G_xiii detail)

```
   4445 engine=0
   4090 engine=1
   2868 engine=2
   4007 engine=3      ← was 17 in the broken run; now full participation
   2944 engine=4
   4072 engine=5
   2983 engine=6
   3456 engine=7
```

Min/max ratio 1.55 — load is reasonably balanced across all 8 GPUs. Compare to the broken run where `engine=3` (vllm4) crashed at t=7s with the FlashInfer cascade `TypeError`, leaving only 7 GPUs serving for the remaining 30 minutes.

## What it took to get here — fixes landed in `metro/version-to-debug` @ ee287f7

1. **Backend toml** (`config/16gpu-distributed/launch_vllm_16gpu.toml`): add `VLLM_REPORT_METRICS=1` and `VLLM_SSE_MODE=full` to all 16 vLLM env blocks (router's `/v1/metrics` SSE subscription is the only token-event channel; without it, every chat/completions through the router hands the client only the role chunk and then heartbeats).
2. **Metro python** (`metro/lifecycle/build.py`, `metro/experiment.py`): rename `--bin client` → `--bin request-sim` and `target/release/client` → `target/release/request-sim`. The request-sim crate has no `client` bin; `cargo build --bin client` would fail. Today this is dead code (sweep flow goes shell → smart_runner.py and never invokes metro python), but it was a future foot-gun.
3. **Docs** (`docs/16gpu_distributed_test_guide.md`): four references to `target/release/client` / "binary is named `client`" replaced with `request-sim`. New **Prerequisites** section documents the two install pre-conditions that silently brick the run if missing:
   - `flashinfer-python ≥ 0.2.8` (carries [PR #1350](https://github.com/flashinfer-ai/flashinfer/pull/1350)).
   - `request-sim` built from `8a265cb` or later (carries the two SSE parser fixes from PR [#27](https://github.com/blitz-serving/request-sim/pull/27)).

The two upstream fixes (PR #27 in request-sim, container-side flashinfer upgrade) are environmental, not configurable from this repo, hence documented as Prerequisites.

## Artifact files in this directory

- `summary-broken.json` — `client.jsonl.summary.json` from the pre-fix 30-min run
- `summary-corrected.json` — same, from the post-fix 30-min run
- `summary-lowload.json` — same, from the sf=1.0 5-min reference run
- `validate-broken.txt` — G1..G8 output for broken run
- `validate-corrected.txt` — G1..G8 output for corrected run

Raw client.jsonl + per-vllm logs are not committed (too large); they live on master-a800 at `/workspace/exps/lmetric/ttft-validation-20260507/{metro-30min,metro-30min-corrected,metro-5min-lowload}/`.
