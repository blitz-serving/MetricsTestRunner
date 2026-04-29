---
name: setup-metro-node
description: Set up a fresh Node (container/VM) for running metro distributed inference tests. Installs system deps, yaullm (without rebuilding C++), blitz-router, request-sim; writes TOML configs; launches via tmux. Invoke when: a Node has only the base image (CUDA + PyTorch + stock vLLM) and needs the full metro stack ready.
---

# Setup Metro Node

This skill prepares a single Node to run metro experiments. A "Node" is one container/VM that simulates one node in the (potentially distributed) NFS-shared cluster. Multiple Nodes share `/workspace/`; each Node has its own `/nvme/models/` and `/usr/local/lib/.../vllm/`.

## Phase 0: Pre-flight (read-only checks)

```bash
# GPU visibility
nvidia-smi --query-gpu=index,memory.total --format=csv,noheader

# Workspace mount (must show all 4)
ls /workspace/{yaullm,blitz-router,request-sim,MetricsTestRunner} | head -10

# Model — must exist locally per Node
ls /nvme/models/Qwen3-30B-A3B/tokenizer.json

# Trace data
ls /workspace/tmp/bailian-traces/*.jsonl | head
```

If any check fails: do NOT proceed. The base image, NFS mount, or model copy is incomplete — fix at the infrastructure level first.

## Phase 1: System dependencies

These tools are needed by launch scripts (`fuser`, `ss`), router build (`protoc`), and tmux orchestration. The standard metro-node base image lacks them.

```bash
apt-get update -qq && apt-get install -y --no-install-recommends \
    iproute2 psmisc protobuf-compiler tmux
```

Rust toolchain check (should already be in `/opt/cargo/bin/cargo`):
```bash
which cargo rustc || curl --proto '=https' --tlsv1.2 -sSf https://mirrors.ustc.edu.cn/misc/rustup-install.sh \
    | sh -s -- -y --default-toolchain stable --profile minimal
```

## Phase 2: yaullm install — wholesale .py overlay

**Critical**: do NOT use `pip install -e /workspace/yaullm`. Editable install redirects the package path away from the pre-built `.so` files in site-packages → `vllm.__file__ = None` → `No module named vllm.entrypoints`.

Instead, copy all `.py` files from the consistent yaullm tree into site-packages (preserves `.so` files):

```bash
SITE=/usr/local/lib/python3.12/dist-packages/vllm
SRC=/workspace/yaullm/vllm
cd $SRC && find . -name "*.py" -exec install -D -m 644 {} $SITE/{} \;
```

Verify (these all must succeed — failure means image's `.so` files are too old for the yaullm Python code):
```bash
python3 -c "
from vllm.v1.request import Request
from vllm.v1.core.kv_cache_manager import KVCacheManager
from vllm import envs
assert hasattr(Request, 'mark_as_waiting')
assert hasattr(KVCacheManager, 'poll_evicted_block_hashes')
print('VLLM_SSE_MODE:', envs.VLLM_SSE_MODE)  # must print 'full' or whatever the env var is set to
"
```

If verification fails, the base image is too old and a new metro-node image must be built (see `workspace/blitz-router/scripts/e2e/Dockerfile.metro-node`).

## Phase 3: Build blitz-router (per policy)

```bash
cd /workspace/blitz-router
# Pick one or all 7 policies (each produces a separate binary):
for policy in lmetric-q join-shortest-q-weight bailian-impl-q aibrix-q dynamo-q dynamo-decoupled-q preble-q; do
    cargo build -p router --release --features "$policy"
    cp target/release/router target/release/router_${policy}
done
```

Default features (`ngrok,vllm-backend,radixtree-blockhash,default-hash-algo,determinent-schedule`) are inherited from `Cargo.toml`. Helper script: `bash scripts/e2e/build_all_policies.sh [single-policy] [log-file]`.

## Phase 4: Build request-sim

```bash
cd /workspace/request-sim && cargo build --release
# Binary: /workspace/request-sim/target/release/request-sim
```

## Phase 5: Write TOML configs

### Single-Node 8-GPU template

`/workspace/MetricsTestRunner/config/<cluster>/launch_yaullm_8gpu.toml`:
```toml
[variables]
model_path = "/nvme/models/Qwen3-30B-A3B"
base_vllm_port = 8100              # NOT 59180 — that range often has stuck listeners
gpu_memory_utilization = 0.85
report_metrics = "1"
sse_mode = "full"

[runtime.raw]
config = [
    { app = "vllm_template", envs = { CUDA_VISIBLE_DEVICES = "0", VLLM_REPORT_METRICS = "${report_metrics}", VLLM_SSE_MODE = "${sse_mode}" }, background = true,  port_offset = 0 },
    { app = "vllm_template", envs = { CUDA_VISIBLE_DEVICES = "1", VLLM_REPORT_METRICS = "${report_metrics}", VLLM_SSE_MODE = "${sse_mode}" }, background = true,  port_offset = 1 },
    { app = "vllm_template", envs = { CUDA_VISIBLE_DEVICES = "2", VLLM_REPORT_METRICS = "${report_metrics}", VLLM_SSE_MODE = "${sse_mode}" }, background = true,  port_offset = 2 },
    { app = "vllm_template", envs = { CUDA_VISIBLE_DEVICES = "3", VLLM_REPORT_METRICS = "${report_metrics}", VLLM_SSE_MODE = "${sse_mode}" }, background = true,  port_offset = 3 },
    { app = "vllm_template", envs = { CUDA_VISIBLE_DEVICES = "4", VLLM_REPORT_METRICS = "${report_metrics}", VLLM_SSE_MODE = "${sse_mode}" }, background = true,  port_offset = 4 },
    { app = "vllm_template", envs = { CUDA_VISIBLE_DEVICES = "5", VLLM_REPORT_METRICS = "${report_metrics}", VLLM_SSE_MODE = "${sse_mode}" }, background = true,  port_offset = 5 },
    { app = "vllm_template", envs = { CUDA_VISIBLE_DEVICES = "6", VLLM_REPORT_METRICS = "${report_metrics}", VLLM_SSE_MODE = "${sse_mode}" }, background = true,  port_offset = 6 },
    { app = "vllm_template", envs = { CUDA_VISIBLE_DEVICES = "7", VLLM_REPORT_METRICS = "${report_metrics}", VLLM_SSE_MODE = "${sse_mode}" }, background = false, port_offset = 7 },
]

[app.vllm_template]
executable = ["python3", "-m", "vllm.entrypoints.openai.api_server"]
extra_args = [
    "--model", "${model_path}",
    "--port", "${port}",                 # = base_vllm_port + port_offset
    "--tensor-parallel-size", "1",
    "--gpu-memory-utilization", "${gpu_memory_utilization}",
    "--enable-prefix-caching",
    "--max-num-seqs", "256",
    "--block-size", "16",                # NOT --kv-cache-block-size (router-only flag)
    "--trust-remote-code",
]
```

### Router config

`vllm_router.toml`:
```toml
[variables]
model_name = "/nvme/models/Qwen3-30B-A3B"   # MUST equal yaullm's --model exactly (404 otherwise)
context_length = 40960                       # match the model's max_position_embeddings, NOT 8192
router_port = 58009
base_vllm_port = 8100                        # consume from backend TOML
num_engines = 8

[runtime.raw]
config = [
    { app = "router", envs = { RUST_BACKTRACE = "full", LOG_LEVEL = "info" }, background = false },
]

[app.router]
executable = ["/workspace/blitz-router/target/release/router_lmetric-q"]
extra_args = [
    "--port", "${router_port}",
    "--client-config", "${client_config_json}",   # JSON file with engine URLs
    "--model-name", "${model_name}",
    "--tokenizer-name", "${model_name}",
    "--use-tokenizer",
    "--max-input-length", "40959",                # context_length - 1
    "--max-total-tokens", "${context_length}",
    "--max-batch-prefill-tokens", "81920",        # MUST be >= max-input-length
    "--max-concurrent-requests", "4096",
    "--kvcache-block-size", "16",
]
```

### Client config (request-sim)

```toml
[variables]
endpoint = "http://localhost:58009/v1/chat/completions"   # OpenAI API, NOT TGI /generate
api = "openai"
model_name = "/nvme/models/Qwen3-30B-A3B"
trace_path = "/workspace/tmp/bailian-traces/qwen_traceA_blksz_16.jsonl"
scale_factor = 3.0

[runtime.raw]
config = [
    { app = "client", background = false },
]

[app.client]
executable = ["/workspace/request-sim/target/release/request-sim"]
extra_args = [
    "--endpoint", "${endpoint}",
    "--api", "${api}",
    "--model-name", "${model_name}",
    "--dataset", "bailian",
    "--dataset-path", "${trace_path}",
    "--mode", "trace-replay",
    "--scale-factor", "${scale_factor}",
    "--tokenizer", "/nvme/models/Qwen3-30B-A3B/tokenizer.json",
    "--tokenizer-config", "/nvme/models/Qwen3-30B-A3B/tokenizer_config.json",
    "--context-length", "40960",
    "--output-path", "/workspace/tmp/exp/results.jsonl",
]
```

### Multi-Node distributed config

For N Nodes sharing the cluster, use `remote_*` variables. Each remote Node must have its own local model:
```toml
[variables]
base_vllm_port = 8100                          # this Node
remote_base_vllm_port = 8100                   # remote Nodes
remote_ips = ["172.27.21.64", "172.27.21.65"]
remote_ssh_port = 10022
remote_model_path = "/nvme/models/Qwen3-30B-A3B"  # path on remote Node (local to it)
```

`smart_runner.py` will SSH to each remote IP, launch via the same TOML, and aggregate logs.

## Phase 6: Launch via tmux

```bash
# Three TOMLs: backend, router, client
cd /workspace/MetricsTestRunner
tmux new-session -d -s metro -x 200 -y 50 \
    "python3 smart_runner.py \
        --backend config/<cluster>/launch_yaullm_8gpu.toml \
        --router  config/<cluster>/vllm_router.toml \
        --client  config/<cluster>/clients.toml \
        --output-dir /workspace/tmp/exp/$(date +%Y%m%d-%H%M%S); read"

# Attach later: tmux attach -t metro
# Status:      tmux capture-pane -t metro -p | tail -20
```

For ad-hoc 8-GPU runs without TOML, use `bash /workspace/blitz-router/scripts/e2e/run_metro_8gpu.sh` directly (it embeds the same defaults).

## Phase 7: Health check

```bash
# All 8 engines healthy?
for i in 0 1 2 3 4 5 6 7; do
    p=$((8100+i))
    curl -sf http://localhost:$p/health > /dev/null && echo "gpu$i:OK" || echo "gpu$i:DOWN"
done

# Router responsive?
curl -sf http://localhost:58009/info | head -c 100

# Requests succeeding?
tail -3 /workspace/tmp/exp/*/results.jsonl | grep -oE '"status":"[0-9]+"'
```

## Phase 8: Trace validation (optional, for staleness experiments)

```bash
python3 /workspace/blitz-router/scripts/e2e/verify_staleness.py \
    /workspace/tmp/exp/<run-id>/router.log
# Exit 0 = P-i, P-ii, P-iii all PASS
# If 0 events parsed, the router log uses a different format — update the regexes
```

## Common Pitfalls (lessons from prior debugging)

| # | Pitfall | Fix |
|---|---|---|
| 1 | Sibling container `-v /workspace:/workspace` mounts host's `/workspace` (wrong) | Use host-absolute path: `-v /ssd/agents/zdy/claude/master-workspace:/workspace` |
| 2 | `pip install -e /workspace/yaullm` breaks C++ extensions | Use wholesale .py overlay (Phase 2) |
| 3 | Image built from inconsistent commits (e.g., scheduler.py from one commit, request.py from another) → `'Request' object has no attribute 'mark_as_waiting'` | Verify all files come from one commit. Use the verification script in Phase 2 |
| 4 | Ports 59180-59187 stuck (TIME_WAIT or other containers using `--network host`) | Use 8100-8107 for run 1, 8200-8207 for run 2 (rotation) |
| 5 | `python` not found in container (only `python3`) | Use `python3` in scripts |
| 6 | `--kv-cache-block-size` is a router flag, not a vLLM flag | yaullm uses `--block-size`, router uses `--kvcache-block-size` |
| 7 | `--api tgi --endpoint /generate` works but breaks parity with vanilla vLLM tests | Always use `--api openai --endpoint /v1/chat/completions` |
| 8 | Router's `--model-name` ≠ yaullm's `--model` → 404 NotFoundError | yaullm's served_model_name = its `--model` value verbatim. Set router `--model-name` to the same string |
| 9 | `--max-batch-prefill-tokens 19999 < --max-input-length 40959` → router refuses to start | Set `--max-batch-prefill-tokens` to ≥ 2 × `--max-input-length` |
| 10 | `CONTEXT_LENGTH=8192` for Qwen3-30B-A3B → 422 errors on long requests | Match the model's `max_position_embeddings` (40960 for Qwen3-30B-A3B) |
| 11 | `fuser` / `ss` missing in container → silent failures in port-cleanup loop | Install `iproute2` + `psmisc` (Phase 1) |
| 12 | request-sim has no `--seed` flag | OS-entropy auto-seeds via `rand::thread_rng()` — restart = new seed |

## Done

After all phases, you have a Node ready to run metro experiments end-to-end. Check progress via `tmux attach -t metro`.
