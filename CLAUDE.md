# MetricsTestRunner (metro)

Distributed test harness for the **lmetric** project. Coordinates `request-sim` (load gen) → `blitz-router` (Rust router) → `yaullm` (vLLM fork) on one or more **Nodes**.

## Working branch (LOCKED)

**Branch contract**: All agent-driven work using this CLAUDE.md MUST happen on **`metro/agent-skill`**. Do NOT switch to `metro/dev`, `main`, or any other branch without explicit user instruction.

```bash
git -C /workspace/MetricsTestRunner branch --show-current  # must print: metro/agent-skill
```

If on the wrong branch, stop and ask the user before any further work. Branch mismatch is a known recurring failure mode (agent assumes one branch state, human is on another).

## What is a Node?

A "Node" is a virtualized compute unit (VM or container) — multiple Nodes share a filesystem via NFS. On the A800 bare-metal server, **one Docker container simulates one Node**. The agent runs inside `zdy-master-agent`; sibling containers (e.g., `zdy-yaullm-metro-8gpu`) act as test Nodes.

## Directory Layout

| Path (in Node) | Purpose | Notes |
|---|---|---|
| `/workspace/yaullm/` | yaullm source (Python + C++/CUDA) | Editable mount, NFS-shared across Nodes |
| `/workspace/blitz-router/` | router source (Rust) | NFS-shared |
| `/workspace/request-sim/` | load generator (Rust) | NFS-shared |
| `/workspace/MetricsTestRunner/` | this harness (TOML configs + smart_runner.py) | NFS-shared |
| `/workspace/MetricsTestRunner/config/<cluster>/` | per-cluster TOML configs | e.g., `ali-a800/`, `dash-h20-1/` |
| `/nvme/models/` | model weights | **Local per Node** (NOT NFS) — must exist on each Node |
| `/workspace/tmp/` | scratch + experiment outputs | NFS-shared |
| `/usr/local/lib/python3.12/dist-packages/vllm/` | installed yaullm | **Local per Node** — wholesale .py overlay (see skill) |

## TOML Config Convention

Every test = three TOML files driven by `smart_runner.py`:
- **Backend**: `launch_vllm_*.toml` — yaullm instances (per-GPU + per-Node)
- **Router**: `vllm_router.toml` — blitz-router (one per cluster)
- **Client**: `*_clients.toml` — request-sim invocations

Ports use `base_vllm_port` + `port_offset` pattern:
```toml
[variables]
base_vllm_port = 59180        # local
remote_base_vllm_port = 59180 # remote Nodes
remote_ips = ["172.27.21.64"]

[runtime.raw]
config = [
    { app = "vllm_template", port_offset = 0, ... },  # → port 59180
    { app = "vllm_template", port_offset = 1, ... },  # → port 59181
]
```

## Setup on a Fresh Node

Use the skill `setup-metro-node` — it documents installing rust, system deps, yaullm overlay, building router/request-sim, writing TOML, launching via tmux. Trigger explicitly via `Skill(skill="setup-metro-node")`.

## Branches (locked)

- **blitz-router**: `lmetric/camera-ready`
- **yaullm**: `lmetric/step-reporter-v2` (commit `9e1d6c4ea` is the validated baseline)
- **request-sim**: `develop` (unpinned — internal lib)

## Common Pitfalls

1. Sibling containers spawned from `zdy-master-agent` need **host-absolute** workspace path (`/ssd/agents/zdy/claude/master-workspace`), NOT `/workspace`.
2. Never `pip install -e /workspace/yaullm` — breaks C++ extensions. Use wholesale .py overlay.
3. Ports 59180-59187 may have stuck listeners on host — prefer 8100-8107 / 8200-8207 with rotation between runs.
4. `--max-batch-prefill-tokens` must be ≥ `--max-input-length` (router validation).
5. yaullm's served_model_name = the path you passed to `--model`. Router's `--model-name` must match exactly.
