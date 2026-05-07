#!/bin/bash
# =============================================================================
# Comprehensive 16-GPU Policy Sweep Script
# =============================================================================
# Runs all 7 policies × 3 traces × 5 scale factors = 105 test combinations.
# Each run: start vLLM → router → client → wait → cleanup → repeat.
#
# Usage: bash sweep_test_robust.sh [start_index]
#   start_index: Resume from this index (0-based), default 0
# =============================================================================

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$SCRIPT_DIR"

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
MODEL_PATH="/home/admin/cpfs/zkx/Qwen--Qwen3-30B-A3B-Instruct-2507/"
VENV_PATH="/home/admin/cpfs/zkx/code/yaullm/.venv"
WORK_DIR="/home/admin/cpfs/zkx/code/blitz-router"
DATASET_DIR="/home/admin/cpfs/zkx/qwen-bailian-usagetraces-anon"
OUTPUT_BASE="/home/admin/cpfs/xmetric/logs"

# SSH
REMOTE_HOST="172.27.122.6"
REMOTE_USER="root"
REMOTE_SSH_PORT=10022

# Timing per run
TIME_IN_SEC=${TIME_IN_SEC:-1200}
ROUTER_WAIT_TIME=60
VLLM_STARTUP_MAX_WAIT=600
VLLM_STARTUP_MAX_RETRIES=5
SWEEP_TEST_LIMIT=${SWEEP_TEST_LIMIT:-}
FORCE_GPU_RESET=${FORCE_GPU_RESET:-0}

# Config files
CONFIG_BACKEND="$SCRIPT_DIR/config/16gpu-distributed/launch_vllm_16gpu.toml"
CONFIG_ROUTER="$SCRIPT_DIR/config/16gpu-distributed/vllm_router_16gpu.toml"
CONFIG_CLIENT="$SCRIPT_DIR/config/16gpu-distributed/client_bailian_16gpu.toml"

# -----------------------------------------------------------------------------
# Test Matrix: 7 policies × 4 traces × 5 scale factors = 133 runs
# -----------------------------------------------------------------------------
# Format: POLICY|TRACE_NAME|SCALE_FACTOR|TIME_IN_SEC
TESTS=(
    # TraceA (toC) - qwen_traceA_blksz_16.jsonl
    "lmetric-q|qwen_traceA_blksz_16.jsonl|5.6|${TIME_IN_SEC}"
    "lmetric-q|qwen_traceA_blksz_16.jsonl|6.0|${TIME_IN_SEC}"
    "lmetric-q|qwen_traceA_blksz_16.jsonl|6.4|${TIME_IN_SEC}"
    "lmetric-q|qwen_traceA_blksz_16.jsonl|6.8|${TIME_IN_SEC}"
    "lmetric-q|qwen_traceA_blksz_16.jsonl|7.2|${TIME_IN_SEC}"
    "preble-q|qwen_traceA_blksz_16.jsonl|5.6|${TIME_IN_SEC}"
    "preble-q|qwen_traceA_blksz_16.jsonl|6.0|${TIME_IN_SEC}"
    "preble-q|qwen_traceA_blksz_16.jsonl|6.4|${TIME_IN_SEC}"
    "preble-q|qwen_traceA_blksz_16.jsonl|6.8|${TIME_IN_SEC}"
    "preble-q|qwen_traceA_blksz_16.jsonl|7.2|${TIME_IN_SEC}"
    "dynamo-q|qwen_traceA_blksz_16.jsonl|5.6|${TIME_IN_SEC}"
    "dynamo-q|qwen_traceA_blksz_16.jsonl|6.0|${TIME_IN_SEC}"
    "dynamo-q|qwen_traceA_blksz_16.jsonl|6.4|${TIME_IN_SEC}"
    "dynamo-q|qwen_traceA_blksz_16.jsonl|6.8|${TIME_IN_SEC}"
    "dynamo-q|qwen_traceA_blksz_16.jsonl|7.2|${TIME_IN_SEC}"
    "dynamo-decoupled-q|qwen_traceA_blksz_16.jsonl|5.6|${TIME_IN_SEC}"
    "dynamo-decoupled-q|qwen_traceA_blksz_16.jsonl|6.0|${TIME_IN_SEC}"
    "dynamo-decoupled-q|qwen_traceA_blksz_16.jsonl|6.4|${TIME_IN_SEC}"
    "dynamo-decoupled-q|qwen_traceA_blksz_16.jsonl|6.8|${TIME_IN_SEC}"
    "dynamo-decoupled-q|qwen_traceA_blksz_16.jsonl|7.2|${TIME_IN_SEC}"
    "aibrix-q|qwen_traceA_blksz_16.jsonl|5.6|${TIME_IN_SEC}"
    "aibrix-q|qwen_traceA_blksz_16.jsonl|6.0|${TIME_IN_SEC}"
    "aibrix-q|qwen_traceA_blksz_16.jsonl|6.4|${TIME_IN_SEC}"
    "aibrix-q|qwen_traceA_blksz_16.jsonl|6.8|${TIME_IN_SEC}"
    "aibrix-q|qwen_traceA_blksz_16.jsonl|7.2|${TIME_IN_SEC}"
    "bailian-impl-q|qwen_traceA_blksz_16.jsonl|5.6|${TIME_IN_SEC}"
    "bailian-impl-q|qwen_traceA_blksz_16.jsonl|6.0|${TIME_IN_SEC}"
    "bailian-impl-q|qwen_traceA_blksz_16.jsonl|6.4|${TIME_IN_SEC}"
    "bailian-impl-q|qwen_traceA_blksz_16.jsonl|6.8|${TIME_IN_SEC}"
    "bailian-impl-q|qwen_traceA_blksz_16.jsonl|7.2|${TIME_IN_SEC}"
    "join-shortest-q-weight|qwen_traceA_blksz_16.jsonl|5.6|${TIME_IN_SEC}"
    "join-shortest-q-weight|qwen_traceA_blksz_16.jsonl|6.0|${TIME_IN_SEC}"
    "join-shortest-q-weight|qwen_traceA_blksz_16.jsonl|6.4|${TIME_IN_SEC}"
    "join-shortest-q-weight|qwen_traceA_blksz_16.jsonl|6.8|${TIME_IN_SEC}"
    "join-shortest-q-weight|qwen_traceA_blksz_16.jsonl|7.2|${TIME_IN_SEC}"

    # TraceB (toB) - qwen_traceB_blksz_16.jsonl
    "lmetric-q|qwen_traceB_blksz_16.jsonl|5.5|${TIME_IN_SEC}"
    "lmetric-q|qwen_traceB_blksz_16.jsonl|6.0|${TIME_IN_SEC}"
    "lmetric-q|qwen_traceB_blksz_16.jsonl|6.5|${TIME_IN_SEC}"
    "lmetric-q|qwen_traceB_blksz_16.jsonl|7.0|${TIME_IN_SEC}"
    "lmetric-q|qwen_traceB_blksz_16.jsonl|7.5|${TIME_IN_SEC}"
    "preble-q|qwen_traceB_blksz_16.jsonl|5.5|${TIME_IN_SEC}"
    "preble-q|qwen_traceB_blksz_16.jsonl|6.0|${TIME_IN_SEC}"
    "preble-q|qwen_traceB_blksz_16.jsonl|6.5|${TIME_IN_SEC}"
    "preble-q|qwen_traceB_blksz_16.jsonl|7.0|${TIME_IN_SEC}"
    "preble-q|qwen_traceB_blksz_16.jsonl|7.5|${TIME_IN_SEC}"
    "dynamo-q|qwen_traceB_blksz_16.jsonl|5.5|${TIME_IN_SEC}"
    "dynamo-q|qwen_traceB_blksz_16.jsonl|6.0|${TIME_IN_SEC}"
    "dynamo-q|qwen_traceB_blksz_16.jsonl|6.5|${TIME_IN_SEC}"
    "dynamo-q|qwen_traceB_blksz_16.jsonl|7.0|${TIME_IN_SEC}"
    "dynamo-q|qwen_traceB_blksz_16.jsonl|7.5|${TIME_IN_SEC}"
    "dynamo-decoupled-q|qwen_traceB_blksz_16.jsonl|5.5|${TIME_IN_SEC}"
    "dynamo-decoupled-q|qwen_traceB_blksz_16.jsonl|6.0|${TIME_IN_SEC}"
    "dynamo-decoupled-q|qwen_traceB_blksz_16.jsonl|6.5|${TIME_IN_SEC}"
    "dynamo-decoupled-q|qwen_traceB_blksz_16.jsonl|7.0|${TIME_IN_SEC}"
    "dynamo-decoupled-q|qwen_traceB_blksz_16.jsonl|7.5|${TIME_IN_SEC}"
    "aibrix-q|qwen_traceB_blksz_16.jsonl|5.5|${TIME_IN_SEC}"
    "aibrix-q|qwen_traceB_blksz_16.jsonl|6.0|${TIME_IN_SEC}"
    "aibrix-q|qwen_traceB_blksz_16.jsonl|6.5|${TIME_IN_SEC}"
    "aibrix-q|qwen_traceB_blksz_16.jsonl|7.0|${TIME_IN_SEC}"
    "aibrix-q|qwen_traceB_blksz_16.jsonl|7.5|${TIME_IN_SEC}"
    "bailian-impl-q|qwen_traceB_blksz_16.jsonl|5.5|${TIME_IN_SEC}"
    "bailian-impl-q|qwen_traceB_blksz_16.jsonl|6.0|${TIME_IN_SEC}"
    "bailian-impl-q|qwen_traceB_blksz_16.jsonl|6.5|${TIME_IN_SEC}"
    "bailian-impl-q|qwen_traceB_blksz_16.jsonl|7.0|${TIME_IN_SEC}"
    "bailian-impl-q|qwen_traceB_blksz_16.jsonl|7.5|${TIME_IN_SEC}"
    "join-shortest-q-weight|qwen_traceB_blksz_16.jsonl|5.5|${TIME_IN_SEC}"
    "join-shortest-q-weight|qwen_traceB_blksz_16.jsonl|6.0|${TIME_IN_SEC}"
    "join-shortest-q-weight|qwen_traceB_blksz_16.jsonl|6.5|${TIME_IN_SEC}"
    "join-shortest-q-weight|qwen_traceB_blksz_16.jsonl|7.0|${TIME_IN_SEC}"
    "join-shortest-q-weight|qwen_traceB_blksz_16.jsonl|7.5|${TIME_IN_SEC}"

    # Coder - anony-qwen3-coder-20251118-14-16.jsonl
    "lmetric-q|anony-qwen3-coder-20251118-14-16.jsonl|2.0|${TIME_IN_SEC}"
    "lmetric-q|anony-qwen3-coder-20251118-14-16.jsonl|2.2|${TIME_IN_SEC}"
    "lmetric-q|anony-qwen3-coder-20251118-14-16.jsonl|2.4|${TIME_IN_SEC}"
    "lmetric-q|anony-qwen3-coder-20251118-14-16.jsonl|2.6|${TIME_IN_SEC}"
    "lmetric-q|anony-qwen3-coder-20251118-14-16.jsonl|2.8|${TIME_IN_SEC}"
    "preble-q|anony-qwen3-coder-20251118-14-16.jsonl|2.0|${TIME_IN_SEC}"
    "preble-q|anony-qwen3-coder-20251118-14-16.jsonl|2.2|${TIME_IN_SEC}"
    "preble-q|anony-qwen3-coder-20251118-14-16.jsonl|2.4|${TIME_IN_SEC}"
    "preble-q|anony-qwen3-coder-20251118-14-16.jsonl|2.6|${TIME_IN_SEC}"
    "preble-q|anony-qwen3-coder-20251118-14-16.jsonl|2.8|${TIME_IN_SEC}"
    "dynamo-q|anony-qwen3-coder-20251118-14-16.jsonl|2.0|${TIME_IN_SEC}"
    "dynamo-q|anony-qwen3-coder-20251118-14-16.jsonl|2.2|${TIME_IN_SEC}"
    "dynamo-q|anony-qwen3-coder-20251118-14-16.jsonl|2.4|${TIME_IN_SEC}"
    "dynamo-q|anony-qwen3-coder-20251118-14-16.jsonl|2.6|${TIME_IN_SEC}"
    "dynamo-q|anony-qwen3-coder-20251118-14-16.jsonl|2.8|${TIME_IN_SEC}"
    "dynamo-decoupled-q|anony-qwen3-coder-20251118-14-16.jsonl|2.0|${TIME_IN_SEC}"
    "dynamo-decoupled-q|anony-qwen3-coder-20251118-14-16.jsonl|2.2|${TIME_IN_SEC}"
    "dynamo-decoupled-q|anony-qwen3-coder-20251118-14-16.jsonl|2.4|${TIME_IN_SEC}"
    "dynamo-decoupled-q|anony-qwen3-coder-20251118-14-16.jsonl|2.6|${TIME_IN_SEC}"
    "dynamo-decoupled-q|anony-qwen3-coder-20251118-14-16.jsonl|2.8|${TIME_IN_SEC}"
    "aibrix-q|anony-qwen3-coder-20251118-14-16.jsonl|2.0|${TIME_IN_SEC}"
    "aibrix-q|anony-qwen3-coder-20251118-14-16.jsonl|2.2|${TIME_IN_SEC}"
    "aibrix-q|anony-qwen3-coder-20251118-14-16.jsonl|2.4|${TIME_IN_SEC}"
    "aibrix-q|anony-qwen3-coder-20251118-14-16.jsonl|2.6|${TIME_IN_SEC}"
    "aibrix-q|anony-qwen3-coder-20251118-14-16.jsonl|2.8|${TIME_IN_SEC}"
    "bailian-impl-q|anony-qwen3-coder-20251118-14-16.jsonl|2.0|${TIME_IN_SEC}"
    "bailian-impl-q|anony-qwen3-coder-20251118-14-16.jsonl|2.2|${TIME_IN_SEC}"
    "bailian-impl-q|anony-qwen3-coder-20251118-14-16.jsonl|2.4|${TIME_IN_SEC}"
    "bailian-impl-q|anony-qwen3-coder-20251118-14-16.jsonl|2.6|${TIME_IN_SEC}"
    "bailian-impl-q|anony-qwen3-coder-20251118-14-16.jsonl|2.8|${TIME_IN_SEC}"
    "join-shortest-q-weight|anony-qwen3-coder-20251118-14-16.jsonl|2.0|${TIME_IN_SEC}"
    "join-shortest-q-weight|anony-qwen3-coder-20251118-14-16.jsonl|2.2|${TIME_IN_SEC}"
    "join-shortest-q-weight|anony-qwen3-coder-20251118-14-16.jsonl|2.4|${TIME_IN_SEC}"
    "join-shortest-q-weight|anony-qwen3-coder-20251118-14-16.jsonl|2.6|${TIME_IN_SEC}"
    "join-shortest-q-weight|anony-qwen3-coder-20251118-14-16.jsonl|2.8|${TIME_IN_SEC}"

    # Mooncake Tool - mooncake_toolagent_trace_poissoned.jsonl
    "lmetric-q|mooncake_toolagent_trace_poissoned.jsonl|1.2|${TIME_IN_SEC}"
    "lmetric-q|mooncake_toolagent_trace_poissoned.jsonl|1.4|${TIME_IN_SEC}"
    "lmetric-q|mooncake_toolagent_trace_poissoned.jsonl|1.6|${TIME_IN_SEC}"
    "lmetric-q|mooncake_toolagent_trace_poissoned.jsonl|1.8|${TIME_IN_SEC}"
    "preble-q|mooncake_toolagent_trace_poissoned.jsonl|1.2|${TIME_IN_SEC}"
    "preble-q|mooncake_toolagent_trace_poissoned.jsonl|1.4|${TIME_IN_SEC}"
    "preble-q|mooncake_toolagent_trace_poissoned.jsonl|1.6|${TIME_IN_SEC}"
    "preble-q|mooncake_toolagent_trace_poissoned.jsonl|1.8|${TIME_IN_SEC}"
    "dynamo-q|mooncake_toolagent_trace_poissoned.jsonl|1.2|${TIME_IN_SEC}"
    "dynamo-q|mooncake_toolagent_trace_poissoned.jsonl|1.4|${TIME_IN_SEC}"
    "dynamo-q|mooncake_toolagent_trace_poissoned.jsonl|1.6|${TIME_IN_SEC}"
    "dynamo-q|mooncake_toolagent_trace_poissoned.jsonl|1.8|${TIME_IN_SEC}"
    "dynamo-decoupled-q|mooncake_toolagent_trace_poissoned.jsonl|1.2|${TIME_IN_SEC}"
    "dynamo-decoupled-q|mooncake_toolagent_trace_poissoned.jsonl|1.4|${TIME_IN_SEC}"
    "dynamo-decoupled-q|mooncake_toolagent_trace_poissoned.jsonl|1.6|${TIME_IN_SEC}"
    "dynamo-decoupled-q|mooncake_toolagent_trace_poissoned.jsonl|1.8|${TIME_IN_SEC}"
    "aibrix-q|mooncake_toolagent_trace_poissoned.jsonl|1.2|${TIME_IN_SEC}"
    "aibrix-q|mooncake_toolagent_trace_poissoned.jsonl|1.4|${TIME_IN_SEC}"
    "aibrix-q|mooncake_toolagent_trace_poissoned.jsonl|1.6|${TIME_IN_SEC}"
    "aibrix-q|mooncake_toolagent_trace_poissoned.jsonl|1.8|${TIME_IN_SEC}"
    "bailian-impl-q|mooncake_toolagent_trace_poissoned.jsonl|1.2|${TIME_IN_SEC}"
    "bailian-impl-q|mooncake_toolagent_trace_poissoned.jsonl|1.4|${TIME_IN_SEC}"
    "bailian-impl-q|mooncake_toolagent_trace_poissoned.jsonl|1.6|${TIME_IN_SEC}"
    "bailian-impl-q|mooncake_toolagent_trace_poissoned.jsonl|1.8|${TIME_IN_SEC}"
    "join-shortest-q-weight|mooncake_toolagent_trace_poissoned.jsonl|1.2|${TIME_IN_SEC}"
    "join-shortest-q-weight|mooncake_toolagent_trace_poissoned.jsonl|1.4|${TIME_IN_SEC}"
    "join-shortest-q-weight|mooncake_toolagent_trace_poissoned.jsonl|1.6|${TIME_IN_SEC}"
    "join-shortest-q-weight|mooncake_toolagent_trace_poissoned.jsonl|1.8|${TIME_IN_SEC}"
)

TOTAL=${#TESTS[@]}
START_INDEX=${1:-0}

# -----------------------------------------------------------------------------
# Helper functions
# -----------------------------------------------------------------------------
log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"
}

cleanup_vllm() {
    log "Cleaning up vLLM processes (local)..."
    if command -v nvidia-smi >/dev/null 2>&1; then
        local gpu_pids
        gpu_pids=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null | awk '/^[0-9]+$/ {print $1}' | sort -u)
        if [ -n "$gpu_pids" ]; then
            log "Killing local GPU compute PIDs: $gpu_pids"
            kill -9 $gpu_pids 2>/dev/null || true
        fi
    fi

    pkill -9 -f "vllm serve" 2>/dev/null || true
    pkill -9 -f "VLLM::Worker" 2>/dev/null || true
    pkill -9 -f "multiprocessing.spawn.*--multiprocessing-fork" 2>/dev/null || true
    pkill -9 -f "multiprocessing.resource_tracker" 2>/dev/null || true
    pkill -9 -f "smart_runner.py" 2>/dev/null || true
    pkill -9 -f "router --hostname" 2>/dev/null || true
    pkill -9 -f "request-sim" 2>/dev/null || true
    pkill -9 -f "$VENV_PATH/bin/python" 2>/dev/null || true

    for port in $(seq 51000 51015) 58009 22281; do
        if command -v fuser >/dev/null 2>&1; then
            fuser -k -9 "${port}/tcp" 2>/dev/null || true
        fi
    done

    if [ "$FORCE_GPU_RESET" = "1" ] && command -v nvidia-smi >/dev/null 2>&1; then
        log "FORCE_GPU_RESET=1; resetting local GPUs 0-7..."
        nvidia-smi --gpu-reset -i 0,1,2,3,4,5,6,7 2>/dev/null || true
    fi

    sleep 5

    log "Cleaning up vLLM processes (remote)..."
    ssh -p "$REMOTE_SSH_PORT" "${REMOTE_USER}@${REMOTE_HOST}" \
        "if command -v nvidia-smi >/dev/null 2>&1; then
             gpu_pids=\$(nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null | awk '/^[0-9]+$/ {print \$1}' | sort -u);
             [ -n \"\$gpu_pids\" ] && kill -9 \$gpu_pids 2>/dev/null || true;
         fi;
         tmux kill-server 2>/dev/null || true;
         pkill -9 -f 'vllm serve' 2>/dev/null || true;
         pkill -9 -f 'VLLM::Worker' 2>/dev/null || true;
         pkill -9 -f 'multiprocessing.spawn.*--multiprocessing-fork' 2>/dev/null || true;
         pkill -9 -f 'multiprocessing.resource_tracker' 2>/dev/null || true;
         pkill -9 -f 'python.*vllm' 2>/dev/null || true;
         pkill -9 -f 'smart_runner.py' 2>/dev/null || true;
         pkill -9 -f 'router --hostname' 2>/dev/null || true;
         pkill -9 -f 'request-sim' 2>/dev/null || true;
         pkill -9 -f '$VENV_PATH/bin/python' 2>/dev/null || true;
         for port in \$(seq 51000 51015) 58009 22281; do
             command -v fuser >/dev/null 2>&1 && fuser -k -9 \${port}/tcp 2>/dev/null || true;
         done;
         if [ '$FORCE_GPU_RESET' = '1' ] && command -v nvidia-smi >/dev/null 2>&1; then
             nvidia-smi --gpu-reset -i 0,1,2,3,4,5,6,7 2>/dev/null || true;
         fi" || true
    sleep 8

    log "vLLM cleanup complete."
}

wait_for_vllm_startup() {
    local output_dir="$1"
    local max_wait="$VLLM_STARTUP_MAX_WAIT"
    local elapsed=0

    log "Waiting for local vLLM instances to start..."
    sleep 30
    elapsed=30

    while [ $elapsed -lt $max_wait ]; do
        local all_ready=true
        for logfile in "$output_dir"/vllm*.log; do
            [ -f "$logfile" ] || { all_ready=false; break; }
            if grep -Eq "^(RuntimeError|ValueError):|EngineCore failed to start|Traceback" "$logfile" 2>/dev/null; then
                log "ERROR: Startup error in $logfile"
                tail -20 "$logfile"
                return 1
            fi
            if ! grep -q "Application startup complete" "$logfile" 2>/dev/null; then
                all_ready=false
                break
            fi
        done

        if [ "$all_ready" = true ]; then
            log "All local vLLM instances started."
            return 0
        fi

        sleep 10
        elapsed=$((elapsed + 10))
    done

    log "ERROR: Local vLLM startup timeout after ${max_wait}s"
    return 1
}

wait_for_remote_vllm_startup() {
    local output_dir="$1"
    local max_wait="$VLLM_STARTUP_MAX_WAIT"
    local elapsed=0

    log "Waiting for remote vLLM instances on Peer..."

    while [ $elapsed -lt $max_wait ]; do
        local all_ready=true
        local remote_logs
        remote_logs=$(ssh -p "$REMOTE_SSH_PORT" "${REMOTE_USER}@${REMOTE_HOST}" \
            "ls ${output_dir}/vllm*.log 2>/dev/null" 2>/dev/null) || { all_ready=false; sleep 5; elapsed=$((elapsed + 5)); continue; }

        if [ -z "$remote_logs" ]; then
            all_ready=false
        else
            for rlog in $remote_logs; do
                if ssh -p "$REMOTE_SSH_PORT" "${REMOTE_USER}@${REMOTE_HOST}" \
                    "grep -Eq '^(RuntimeError|ValueError):|EngineCore failed to start|Traceback' '$rlog'" 2>/dev/null; then
                    log "ERROR: Remote startup error in $rlog"
                    ssh -p "$REMOTE_SSH_PORT" "${REMOTE_USER}@${REMOTE_HOST}" \
                        "tail -20 '$rlog'" 2>/dev/null || true
                    return 1
                fi
                if ! ssh -p "$REMOTE_SSH_PORT" "${REMOTE_USER}@${REMOTE_HOST}" \
                    "grep -q 'Application startup complete' '$rlog'" 2>/dev/null; then
                    all_ready=false
                    break
                fi
            done
        fi

        if [ "$all_ready" = true ]; then
            log "All remote vLLM instances started."
            return 0
        fi

        sleep 5
        elapsed=$((elapsed + 5))
    done

    log "ERROR: Remote vLLM startup timeout"
    return 1
}

run_single_test() {
    local policy="$1"
    local trace_name="$2"
    local scale_factor="$3"
    local time_in_sec="$4"
    local run_index="$5"

    local timestamp
    timestamp=$(date +%Y%m%d-%H%M%S)
    local trace_short
    trace_short=$(echo "$trace_name" | sed 's/\.jsonl$//' | sed 's/_blksz_16//' | sed 's/anony-//' | sed 's/qwen_//' | sed 's/-/./g')
    local output_base_run="${OUTPUT_BASE}/${timestamp}_${trace_short}_${policy}"
    local output_dir_local=""
    local output_dir_remote=""

    log "============================================================"
    log "Run $((run_index + 1))/$TOTAL: policy=$policy trace=$trace_short SF=$scale_factor"
    log "Output base: $output_base_run"
    log "============================================================"

    # Build router for current policy
    log "Building router for policy=$policy..."
    if ! (cd "$WORK_DIR" && cargo build -p router --release --features "$policy" --quiet); then
        log "ERROR: Router build failed for policy=$policy. Skipping this run."
        cleanup_vllm
        return 1
    fi

    # Update client TOML with current trace and scale factor
    if ! sed -i "s|trace_name = \".*\"|trace_name = \"${trace_name}\"|" "$CONFIG_CLIENT"; then
        log "ERROR: Failed to update trace_name in $CONFIG_CLIENT"
        return 1
    fi
    if ! sed -i "s|config.scale_factor = .*|config.scale_factor = ${scale_factor}|" "$CONFIG_CLIENT"; then
        log "ERROR: Failed to update scale_factor in $CONFIG_CLIENT"
        return 1
    fi

    local backend_pid=""
    local backend_started=false
    for attempt in $(seq 1 "$VLLM_STARTUP_MAX_RETRIES"); do
        output_dir_local="${output_base_run}/attempt${attempt}/node1"
        output_dir_remote="${output_base_run}/attempt${attempt}/node2"

        log "Pre-attempt cleanup (${attempt}/${VLLM_STARTUP_MAX_RETRIES})..."
        cleanup_vllm

        log "Starting vLLM backends (attempt ${attempt}/${VLLM_STARTUP_MAX_RETRIES})..."
        SMART_RUNNER_NO_CLEANUP=1 "$VENV_PATH/bin/python" "$PROJECT_ROOT/smart_runner.py" \
            --toml "$CONFIG_BACKEND" \
            --output-dir="$output_dir_local" \
            --remote-output-dir="$output_dir_remote" &
        backend_pid=$!

        if wait_for_vllm_startup "$output_dir_local" && wait_for_remote_vllm_startup "$output_dir_remote"; then
            backend_started=true
            break
        fi

        log "ERROR: vLLM startup failed on attempt ${attempt}/${VLLM_STARTUP_MAX_RETRIES}. Cleaning up before retry."
        kill "$backend_pid" 2>/dev/null || true
        cleanup_vllm

        if [ "$attempt" -lt "$VLLM_STARTUP_MAX_RETRIES" ]; then
            log "Retrying vLLM startup after 20s..."
            sleep 20
        fi
    done

    if [ "$backend_started" != true ]; then
        log "ERROR: vLLM startup failed after ${VLLM_STARTUP_MAX_RETRIES} attempts. Skipping this run."
        return 1
    fi

    # Start router
    log "Starting router..."
    "$VENV_PATH/bin/python" "$PROJECT_ROOT/smart_runner.py" \
        --toml "$CONFIG_ROUTER" \
        --output-dir="$output_dir_local" &
    local router_pid=$!

    sleep $ROUTER_WAIT_TIME

    if ! ss -tlnp | grep -q ":58009"; then
        log "ERROR: Router failed to start. Skipping this run."
        kill $backend_pid $router_pid 2>/dev/null || true
        cleanup_vllm
        return 1
    fi
    log "Router is running."

    # Start client
    log "Starting client (request-sim)..."
    "$VENV_PATH/bin/python" "$PROJECT_ROOT/smart_runner.py" \
        --toml "$CONFIG_CLIENT" \
        --output-dir="$output_dir_local" \
        --time_in_secs="$time_in_sec" &
    local client_pid=$!

    # Wait for experiment
    log "Experiment running for ${time_in_sec}s..."
    sleep "$time_in_sec"

    # Wait for client to finish
    local wait_count=0
    while pgrep -f "request-sim" > /dev/null 2>&1; do
        log "Client still running, waiting..."
        sleep 5
        wait_count=$((wait_count + 1))
        if [ $wait_count -gt 120 ]; then
            log "Client timeout, killing..."
            pkill -9 -f "request-sim" 2>/dev/null || true
            break
        fi
    done

    log "Client finished."

    # Cleanup all processes
    kill $backend_pid 2>/dev/null || true
    kill $router_pid 2>/dev/null || true
    sleep 3
    cleanup_vllm

    # Merge client logs. Exclude client.jsonl itself; redirecting to a file that
    # is also matched by the input glob truncates the only client log.
    local client_parts=()
    mapfile -t client_parts < <(
        find "$output_dir_local" -maxdepth 1 -type f -name 'client*.jsonl' ! -name 'client.jsonl' | sort
    )
    if [ ${#client_parts[@]} -gt 0 ]; then
        local merged_client="${output_dir_local}/client.jsonl"
        local merged_tmp="${merged_client}.tmp"
        cat "${client_parts[@]}" > "$merged_tmp"
        mv "$merged_tmp" "$merged_client"
        local req_count
        req_count=$(wc -l < "$merged_client")
        log "Merged client.jsonl: $req_count requests"
    elif [ -f "$output_dir_local/client.jsonl" ]; then
        local req_count
        req_count=$(wc -l < "$output_dir_local/client.jsonl")
        log "Keeping existing client.jsonl: $req_count requests"
    fi

    log "Run $((run_index + 1))/$TOTAL complete."
    return 0
}

# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
log "============================================================"
log "Starting comprehensive sweep: $TOTAL runs"
log "Policies: lmetric-q, preble-q, dynamo-q, dynamo-decoupled-q, aibrix-q, bailian-impl-q, join-shortest-q-weight"
log "Traces: TraceA, TraceB, Coder"
log "Starting from index: $START_INDEX"
[ -n "$SWEEP_TEST_LIMIT" ] && log "Test limit: $SWEEP_TEST_LIMIT run(s)"
log "============================================================"

# Save sweep config
mkdir -p "$OUTPUT_BASE"
echo "Sweep started: $(date)" > "$OUTPUT_BASE/sweep_log.txt"
echo "Total runs: $TOTAL" >> "$OUTPUT_BASE/sweep_log.txt"
echo "Start index: $START_INDEX" >> "$OUTPUT_BASE/sweep_log.txt"
[ -n "$SWEEP_TEST_LIMIT" ] && echo "Test limit: $SWEEP_TEST_LIMIT" >> "$OUTPUT_BASE/sweep_log.txt"

END_INDEX=$((TOTAL - 1))
if [ -n "$SWEEP_TEST_LIMIT" ]; then
    END_INDEX=$((START_INDEX + SWEEP_TEST_LIMIT - 1))
    if [ "$END_INDEX" -ge "$TOTAL" ]; then
        END_INDEX=$((TOTAL - 1))
    fi
fi

for i in $(seq 0 "$END_INDEX"); do
    # Skip already completed runs
    if [ $i -lt $START_INDEX ]; then
        continue
    fi

    IFS='|' read -r policy trace_name scale_factor time_in_sec <<< "${TESTS[$i]}"

    run_single_test "$policy" "$trace_name" "$scale_factor" "$time_in_sec" "$i"
    local_result=$?

    # Log result
    if [ $local_result -eq 0 ]; then
        echo "$(date '+%Y-%m-%d %H:%M:%S') SUCCESS run=$((i+1)) policy=$policy trace=$trace_name SF=$scale_factor" >> "$OUTPUT_BASE/sweep_log.txt"
    else
        echo "$(date '+%Y-%m-%d %H:%M:%S') FAILED run=$((i+1)) policy=$policy trace=$trace_name SF=$scale_factor" >> "$OUTPUT_BASE/sweep_log.txt"
    fi

    # Brief pause between runs
    log "Pausing 10s before next run..."
    sleep 10
done

log "============================================================"
log "Sweep complete! Results in: $OUTPUT_BASE"
log "Sweep log: $OUTPUT_BASE/sweep_log.txt"
log "============================================================"
