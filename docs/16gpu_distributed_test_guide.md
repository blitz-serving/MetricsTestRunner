# 16-GPU Distributed Test Setup Guide

This document describes how to set up and run 16-GPU distributed LLM inference tests using MetricsTestRunner across two machines (local + remote Peer).

---

## Architecture

```
request-sim (client, local)
    ↓ HTTP
blitz-router (router, local)
    ↓ HTTP
yaullm/vLLM (16 instances: 8 local + 8 remote via SSH)
```

Each test requires **3 TOML files**:
- **Backend**: `launch_vllm_*.toml` — vLLM instances (per-GPU + per-Node)
- **Router**: `vllm_router_*.toml` — blitz-router (one per cluster)
- **Client**: `*_clients.toml` — request-sim invocations

---

## SSH Connection (Peer Node)

The remote machine is configured in `~/.ssh/config` as `Peer`:

```
Host Peer
    HostName 172.27.122.6
    User root
    Port 10022
    IdentityFile ~/.ssh/id_ed25519
```

**Always verify connectivity before running tests:**
```bash
ssh -p 10022 root@172.27.122.6 "echo OK"
```

---

## Path Conventions (Current Setup)

| Component | Local Path | Remote (Peer) Path |
|-----------|-----------|-------------------|
| **Model** | `/home/admin/cpfs/zkx/Qwen--Qwen3-30B-A3B-Instruct-2507/` | Same (NFS/shared) |
| **vLLM/yaullm venv** | `/home/admin/cpfs/zkx/code/yaullm/.venv/` | Same |
| **vLLM executable** | `/home/admin/cpfs/zkx/code/yaullm/.venv/bin/vllm` | Same |
| **Router source** | `/home/admin/cpfs/zkx/code/blitz-router/` | Same |
| **Router binary** | `${WORK_DIR}/target/release/router_<policy>` | N/A (runs local) |
| **Client binary** | `${WORK_DIR}/target/release/client` | N/A (runs local) |
| **Trace data** | `/home/admin/cpfs/zkx/qwen-bailian-usagetraces-anon/` | Same |
| **Output (local)** | `/home/admin/cpfs/xmetric/logs/node1/<timestamp>_<policy>/` | N/A |
| **Output (remote)** | N/A | `/home/admin/cpfs/xmetric/logs/node2/<timestamp>_<policy>/` |

---

## Routing Policies

Available policies (from sweep results on `qwen_traceA_blksz_16.jsonl`):

| Policy | RPS | TPOT_mean | Status |
|--------|-----|-----------|--------|
| `preble-q` | 19.06 | 86.4ms | ✅ Best throughput |
| `dynamo-decoupled-q` | 18.37 | 76.6ms | ✅ Lowest latency (exits early) |
| `join-shortest-q-weight` | 15.87 | 119.3ms | ✅ Balanced |
| `lmetric-q` | 15.46 | 100.1ms | ✅ Project policy |
| `bailian-impl-q` | 14.59 | 105.5ms | ✅ |
| `aibrix-q` | 13.49 | 128.2ms | ✅ |
| `dynamo-q` | 12.26 | 204.6ms | ❌ Has staleness bug |

---

## Repository Branches

| Repository | Branch | Notes |
|-----------|--------|-------|
| `blitz-router` (router) | `lmetric/camera-ready` | Build with `cargo build -p router --release --features <policy>` |
| `yaullm` (vLLM fork) | `lmetric/step-reporter-v2` | Baseline commit: `9e1d6c4ea` |
| `request-sim` | `develop` | Part of `blitz-router` workspace |
| `MetricsTestRunner` | `metro/agent-skill` | Current working branch |

---

## How to Generate Configs for a New Setup

### Step 1: Fill in the config template

Copy `config_16gpu_template.txt` and fill in all values. Key fields:

```bash
# SSH (if not using existing "Peer" entry)
PEER_HOST=
PEER_USER=
PEER_SSH_PORT=

# Model
MODEL_PATH_LOCAL=
MODEL_PATH_REMOTE=

# Software
VENV_PATH_LOCAL=
VENV_PATH_REMOTE=
VLLM_EXECUTABLE_LOCAL=
VLLM_EXECUTABLE_REMOTE=

# Router & Client
WORK_DIR_LOCAL=
WORK_DIR_REMOTE=

# Test
DATASET_DIR_LOCAL=
TRACE_NAME=
POLICY=
TIME_IN_SEC=
SCALE_FACTOR=

# Output
OUTPUT_DIR_LOCAL=
OUTPUT_DIR_REMOTE=

# Ports & Settings
BASE_VLLM_PORT_LOCAL=
BASE_VLLM_PORT_REMOTE=
ROUTER_PORT=
CONTEXT_LENGTH=
GPU_MEMORY_UTILIZATION=
ATTENTION_BACKEND=
```

### Step 2: Infer binary paths

From `WORK_DIR_LOCAL`, infer:
- Router binary: `${WORK_DIR_LOCAL}/target/release/router`
- Client binary: `${WORK_DIR_LOCAL}/target/release/client` (NOT `request-sim` — the binary is named `client`)

### Step 3: Verify remote paths

Before generating configs, verify the remote machine has all required paths:

```bash
ssh -p ${PEER_SSH_PORT} ${PEER_USER}@${PEER_HOST} "
    ls ${MODEL_PATH_REMOTE}/tokenizer.json &&
    ls ${VENV_PATH_REMOTE}/bin/vllm &&
    echo 'Remote paths OK'
"
```

### Step 4: Create TOML configs

Create 3 files in `config/<cluster-name>/`:

> **⚠️ Critical Rules for Backend TOML**
>
> 1. **Required environment variables** — Every vLLM app in `[runtime.raw]` and `[runtime.ssh_1]` must include these envs:
>    ```
>    TORCH_CUDA_ARCH_LIST = "9.0"
>    PYTHONHASHSEED = 42
>    VLLM_USE_FLASHINFER_SAMPLER = 0
>    OMP_NUM_THREADS = 1
>    ```
> 2. **Last app must be foreground** — The last app in each runtime's `config` list must have `background = false` so `smart_runner.py` blocks and the launch script does not exit prematurely.
> 3. **Do NOT use `--enforce-eager`** — This flag must not appear in `app.vllm_template`'s `extra_args`.

#### Backend TOML (`launch_vllm_16gpu.toml`)

```toml
# 8 local GPUs + 8 remote GPUs via SSH
[runtime.raw]
config = [
    { app = "vllm_template", envs = { CUDA_VISIBLE_DEVICES = "0", VLLM_ATTENTION_BACKEND = "FLASHINFER", TORCH_CUDA_ARCH_LIST = "9.0", PYTHONHASHSEED = "42", VLLM_USE_FLASHINFER_SAMPLER = "0", OMP_NUM_THREADS = "1" }, background = true, port_offset = 0 },
    { app = "vllm_template", envs = { CUDA_VISIBLE_DEVICES = "1", VLLM_ATTENTION_BACKEND = "FLASHINFER", TORCH_CUDA_ARCH_LIST = "9.0", PYTHONHASHSEED = "42", VLLM_USE_FLASHINFER_SAMPLER = "0", OMP_NUM_THREADS = "1" }, background = true, port_offset = 1 },
    # ... ports 2-6 ...
    { app = "vllm_template", envs = { CUDA_VISIBLE_DEVICES = "7", VLLM_ATTENTION_BACKEND = "FLASHINFER", TORCH_CUDA_ARCH_LIST = "9.0", PYTHONHASHSEED = "42", VLLM_USE_FLASHINFER_SAMPLER = "0", OMP_NUM_THREADS = "1" }, background = false, port_offset = 7 },
]

[runtime.ssh_1]
config = [
    { app = "vllm_remote_template", envs = { CUDA_VISIBLE_DEVICES = "0", VLLM_ATTENTION_BACKEND = "FLASHINFER", TORCH_CUDA_ARCH_LIST = "9.0", PYTHONHASHSEED = "42", VLLM_USE_FLASHINFER_SAMPLER = "0", OMP_NUM_THREADS = "1" }, background = true, port_offset = 0 },
    { app = "vllm_remote_template", envs = { CUDA_VISIBLE_DEVICES = "1", VLLM_ATTENTION_BACKEND = "FLASHINFER", TORCH_CUDA_ARCH_LIST = "9.0", PYTHONHASHSEED = "42", VLLM_USE_FLASHINFER_SAMPLER = "0", OMP_NUM_THREADS = "1" }, background = true, port_offset = 1 },
    # ... ports 2-6 ...
    { app = "vllm_remote_template", envs = { CUDA_VISIBLE_DEVICES = "7", VLLM_ATTENTION_BACKEND = "FLASHINFER", TORCH_CUDA_ARCH_LIST = "9.0", PYTHONHASHSEED = "42", VLLM_USE_FLASHINFER_SAMPLER = "0", OMP_NUM_THREADS = "1" }, background = false, port_offset = 7 },
]

[variables]
model_path = "${MODEL_PATH_LOCAL}"
vllm_executable = "${VLLM_EXECUTABLE_LOCAL}"
work_dir = "${WORK_DIR_LOCAL}"
base_vllm_port = ${BASE_VLLM_PORT_LOCAL}
remote_base_vllm_port = ${BASE_VLLM_PORT_REMOTE}
remote_ips = ["${PEER_HOST}"]
remote_user = "${PEER_USER}"
remote_ssh_port = ${PEER_SSH_PORT}

[app.vllm_template]
executable = ["${vllm_executable}", "serve"]
extra_args = [
    "${model_path}",
    "--port", "${port}",
    "--max-model-len", "${CONTEXT_LENGTH}",
    "--max-num-seqs", "256",
    "--block-size", "16",
    "--enable-prefix-caching",
    "--gpu-memory-utilization", "${GPU_MEMORY_UTILIZATION}",
    "--trust-remote-code",
    "--disable-log-requests",
    "> ${log_file} 2>&1"
]

[app.vllm_remote_template]
inherit = "vllm_template"
```

#### Router TOML (`vllm_router_16gpu.toml`)

```toml
[runtime.raw]
config = [
    { app = "router", envs = { RUST_BACKTRACE = "full", LOG_LEVEL = "info,cache_tracking=info" }, background = true, block_keyword = "Blitz router is ready" },
]

[variables]
model_path = "${MODEL_PATH_LOCAL}"
work_dir = "${WORK_DIR_LOCAL}"

[app.router]
executable = ["${work_dir}/target/release/router"]
extra_args = ["--use-tokenizer"]

[app.router.macro_rules]
"--hostname" = "hostname"
"--port" = "port"
"--client-config" = "cli_stub_cfg"
"--tokenizer-name" = "tokenizer"
"--model-name" = "model"
"--log-path" = "log_path"
"--max-input-length" = "context_length"
"--max-total-tokens" = "total_tokens"
"--max-concurrent-requests" = "concurrent_req"
"--max-batch-prefill-tokens" = "mbpt"
"--kvcache-block-size" = "block_size"

[app.router.config]
hostname = "localhost"
port = "${ROUTER_PORT}"
model = "${model_path}"
tokenizer = "${model_path}"
cli_stub_cfg = "${work_dir}/exps/blitz-run/configs/config-stubs.json"
context_length = ${CONTEXT_LENGTH} - 1
total_tokens = ${CONTEXT_LENGTH}
concurrent_req = 4096
mbpt = ${CONTEXT_LENGTH} * 2
block_size = 16
```

#### Client TOML (`client_bailian_16gpu.toml`)

```toml
[runtime.raw]
config = [
    { app = "client", background = false },
]

[variables]
model_path = "${MODEL_PATH_LOCAL}"
work_dir = "${WORK_DIR_LOCAL}"
dataset_dir = "${DATASET_DIR_LOCAL}"
trace_name = "${TRACE_NAME}"
client_executable = "${work_dir}/target/release/client"

[app.client_template]
executable = ["${client_executable}"]
extra_args = ["--mode", "trace-replay", "--stream"]

[app.client_template.macro_rules]
"--tokenizer" = "tokenizer"
"--tokenizer-config" = "tokenizer_config"
"--endpoint" = "endpoint"
"--api" = "api"
"--scale-factor" = "scale_factor"
"--dataset" = "dataset_type"
"--dataset-path" = "dataset_path"
"--time-in-secs" = "time_in_secs"
"--output-path" = "output_path"
"--model-name" = "model_name"

[app.client]
inherit = "client_template"
config.tokenizer = "${model_path}/tokenizer.json"
config.tokenizer_config = "${model_path}/tokenizer_config.json"
config.endpoint = "http://localhost:${ROUTER_PORT}/v1/chat/completions"
config.api = "openai"
config.model_name = "${model_path}"
config.scale_factor = "${SCALE_FACTOR}"
config.dataset_type = "bailian"
config.dataset_path = "${dataset_dir}/${trace_name}"
config.time_in_secs = "${time-in-secs}"
config.output_path = "${output_dir}/client.jsonl"
```

### Step 5: Create launch script

See `launch_16gpu.sh` for the complete template. Key sections:
1. **Validation** — check configs, directories, trace files
2. **Build** — compile router with selected policy + build request-sim client
3. **Launch** — tmux session with 3 windows (backend/router/client)
4. **Health check** — wait for local + remote vLLM startup
5. **Post-process** — merge client logs, generate figures
6. **Cleanup** — kill all processes on exit

---

## How to Run

```bash
cd /home/admin/cpfs/zkx/code/MetricsTestRunner

# Run with a specific policy
bash launch_16gpu.sh lmetric-q
```

The script will:
1. Validate all paths and configs
2. Build router with the selected policy feature
3. Build request-sim client
4. Create timestamped output directories
5. Launch 16 vLLM instances (8 local + 8 remote) in tmux window "backend"
6. Wait for all instances to report "Application startup complete"
7. Launch router in tmux window "router"
8. Launch client in tmux window "client"
9. Wait for experiment duration (TIME_IN_SEC)
10. Merge logs and generate figures
11. Cleanup all processes

---

## Common Pitfalls

| # | Issue | Fix |
|---|-------|-----|
| 1 | `pip install -e /workspace/yaullm` breaks C++ extensions | Use wholesale .py overlay instead |
| 2 | Ports 59180-59187 stuck | Use 8100-8107 or 51000-51007 with rotation |
| 3 | `--max-batch-prefill-tokens < --max-input-length` | Set `mbpt >= context_length * 2` |
| 4 | Router's `--model-name` ≠ vLLM's `--model` | Must match exactly (404 otherwise) |
| 5 | `--kv-cache-block-size` is router-only | vLLM uses `--block-size` |
| 6 | Client binary name | It's `client`, NOT `request-sim` |
| 7 | Remote vLLM startup timeout | Check remote logs, ensure model exists on remote |
| 8 | CONTEXT_LENGTH mismatch | Must match model's `max_position_embeddings` |
| 9 | **Missing required env vars** | Every vLLM app needs `TORCH_CUDA_ARCH_LIST`, `PYTHONHASHSEED`, `VLLM_USE_FLASHINFER_SAMPLER`, `OMP_NUM_THREADS` |
| 10 | **All apps background=true** | Last app in each runtime's config must be `background = false` to block script exit |
| 11 | **`--enforce-eager` in extra_args** | This flag must NOT appear in `app.vllm_template`'s `extra_args` |

---

## Verification Checklist

Before running a test:

- [ ] SSH to Peer works: `ssh -p 10022 root@172.27.122.6 "echo OK"`
- [ ] Model exists on both machines
- [ ] vLLM venv exists on both machines
- [ ] Router source exists locally
- [ ] Trace file exists locally
- [ ] Ports are free: `ss -tlnp | grep 5100`
- [ ] Output directories exist or are creatable
- [ ] TOML configs have correct paths
- [ ] Every vLLM app includes required env vars (`TORCH_CUDA_ARCH_LIST`, `PYTHONHASHSEED`, `VLLM_USE_FLASHINFER_SAMPLER`, `OMP_NUM_THREADS`)
- [ ] Last app in each runtime's config has `background = false`
- [ ] `--enforce-eager` is NOT in `app.vllm_template`'s `extra_args`

---

## Key Files Reference

| File | Purpose |
|------|---------|
| `smart_runner.py` | Main execution script — parses TOML, launches apps |
| `launch_16gpu.sh` | One-click launch script for 16-GPU test |
| `config/16gpu-distributed/launch_vllm_16gpu.toml` | Backend config (8 local + 8 remote) |
| `config/16gpu-distributed/vllm_router_16gpu.toml` | Router config |
| `config/16gpu-distributed/client_bailian_16gpu.toml` | Client config |
| `config_16gpu_template.txt` | Config template (fill in for new setups) |
