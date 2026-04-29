============================================================
Camera-Ready 7-Policy Sweep — qwen_traceA_blksz_16.jsonl
============================================================
Date: 2026-04-29 (sweep ran 16:35 - 22:14 UTC)
Model: Qwen3-30B-A3B
GPUs: 8 A800-SXM4-80GB (one yaullm per GPU, TP=1)
Router: blitz-router lmetric/camera-ready
Engine: yaullm lmetric/step-reporter-v2
Trace: qwen_traceA_blksz_16.jsonl, SF=3.9, target time=1900s
Streaming: enabled (--stream); TTFT capture broken (request-sim summary writes 0.000)
============================================================

Per-policy results (sorted by throughput_rps desc):

Policy                   Validation    rps    TPOT_mean  TPOT_p99   E2E_mean  Success/Total      OutTokens
------                   ----------    ----   ---------  --------   --------  -------------      ---------
preble-q                 PASS          19.06  86.4ms     1023ms     24.7s     18573/38301 (49%)  16.4M
dynamo-decoupled-q       PASS *        18.37  76.6ms     672ms      23.1s      7106/12184 (58%)   4.9M
join-shortest-q-weight   PASS          15.87  119.3ms    1393ms     29.0s     11469/25306 (45%)  10.7M
lmetric-q                PASS          15.46  100.1ms    1028ms     30.7s     14034/33359 (42%)  14.2M
bailian-impl-q           PASS          14.59  105.5ms    1192ms     32.7s     12674/31857 (40%)  13.6M
aibrix-q                 PASS          13.49  128.2ms    1248ms     35.3s     13379/30018 (45%)  12.9M
dynamo-q                 FAIL **       12.26  204.6ms    2221ms     44.5s     13077/27193 (48%)  11.6M

* dynamo-decoupled-q exited early (11 min vs 31 min target) - fewer total reqs, but metrics computed.
** dynamo-q has 6 P-ii/P-iii staleness violations: corrections with non-zero diff but
   zero epoch gap (decision_epoch == current_epoch). Indicates a router cache-tracking
   bug specific to the dynamo-q policy. Sample violations:
     request_id=1627 engine=6: diff=16 but decision_epoch=854 >= current_epoch=854
     request_id=1873 engine=6: diff=624 but decision_epoch=1207 >= current_epoch=1207
     ... (4 more)

KEY FINDINGS
============
1. preble-q is the top performer on this trace (highest rps, low E2E).
2. dynamo-decoupled-q has the best per-token latency (TPOT 76ms) - if its early-exit
   issue is investigated, it could be a strong candidate.
3. lmetric-q is competitive (3rd in TPOT mean, 4th in rps).
4. dynamo-q is BOTH the slowest AND has staleness violations - serious candidate to debug.
5. TTFT measurement is broken in request-sim's summary output (always 0.000) despite
   --stream being enabled. Per-request first_token_time is captured in client.jsonl
   but not surfaced into the summary.
6. Success rates 40-58% across the board indicate SF=3.9 is near or above saturation
   for 8 A800s with Qwen3-30B-A3B. For accurate latency comparison, lower SF is needed.

DATA LOCATIONS
==============
- Per-policy dirs: /workspace/exps/metro/sf3.9_traceA_camready/<ts>_<policy>/
- Each contains: client.jsonl, client.jsonl.summary.json, router_v2.log,
  vllm1..vllm8.log, backend.toml, router.toml, client.toml
- The bailian-impl-q dir 20260429100835 is from the FIRST sweep attempt (client.jsonl
  was wiped by a now-fixed merge bug; router log + summary are intact).

DECISION
========
Per user's plan: "qwen_thinking only if Trace 1 passes without bugs".
Trace 1 has dynamo-q staleness FAIL. Therefore: NOT proceeding to Trace 2 (qwen_thinking)
without explicit user direction.

METRO PATCHES APPLIED THIS SESSION
==================================
1. metro/lifecycle/build.py: changed "router_v2" to "router", added per-features
   binary cache mechanism so pre-built router_<policy> binaries are reused.
2. metro/lifecycle/cleanup.py: changed router pkill pattern to "target/release/router";
   added cleanup for vllm multiprocessing.spawn / resource_tracker workers.
3. metro/experiment.py: inject "time-in-secs" into variables dict so client TOML
   ${time-in-secs} macro resolves (was always defaulting to 60s).
4. metro/results/collector.py: merge_client_jsonl no longer truncates client.jsonl
   when no other client*.jsonl sources exist (was wiping data when only one file).
5. config/ali-a800/launch_vllm_dp8_container.toml: --max-model-len 8192 -> 40960,
   --max-num-seqs 4096 -> 256, base_vllm_port 59190 -> 8100, added --block-size 16.
6. config/ali-a800/vllm_router_8gpu_container.toml: context_length 8191 -> 40959,
   total_tokens 8192 -> 40960, mbpt 16384 -> 81920, LOG_LEVEL added cache_tracking=info,
   router runs background=true with block_keyword="Blitz router is ready",
   client-config uses auto-generated config-stubs.json.
7. config/ali-a800/client_bailian_container.toml: --api tgi -> openai,
   endpoint /generate -> /v1/chat/completions, added model_name macro.
8. clusters/a800-8gpu.toml: model paths /models/ -> /nvme/models/, base_vllm_port 59190 -> 8100.
9. New file: sweeps/lmetric_camera_ready_a800.toml + lmetric_camera_ready_a800_redo.toml.
