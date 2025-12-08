#!/bin/bash

# =============================================================================
# Client-Only vLLM Distributed Inference Runner (v2)
# =============================================================================
# Launches:
#   - vLLM backends (local + remote, data-parallel) → each in its own tmux window
#   - Client via smart_runner.py → in a separate tmux window
# No router. No policy. No --no-backend.
# Supports --output-dir / --remote-output-dir (overrides timestamped path).
# =============================================================================

# -----------------------------------------------------------------------------
# Configuration - Edit as needed
# -----------------------------------------------------------------------------

MODEL_PATH='/home/admin/resource/model/464482ce.Qwen2.5-7B-Instruct/1.0/'
REMOTE_MODEL_PATH='/home/admin/resource/model/464482ce.Qwen2.5-7B-Instruct/1.0/'

VENV_PATH='/mnt/debugger/hjb/node1/original_vllm/.venv'
REMOTE_VENV_PATH='/mnt/debugger/hjb/node2/original_vllm/.venv'

WORK_DIR='/mnt/debugger/hjb/node1/blitz-infer-pack'
DATASET_DIR="/mnt/debugger/hjb/node1/qwen-bailian-usagetraces-anon"

OUTPUT_BASE="/tmp/node1/lmmetric-logs"
REMOTE_OUTPUT_BASE="/tmp/node2/lmmetric-logs"

REMOTE_IPS="172.27.21.162"
USE_REMOTE=True
SSH_PORT=10022
HEAD_IP="172.27.69.15"

TIME_IN_SEC=$((1200 + 10))
SESSION_NAME="client_only_vllm"

CLIENT_WAIT_TIME=20

# -----------------------------------------------------------------------------
# Derived Configuration
# -----------------------------------------------------------------------------

TIMESTAMP=$(date +%Y%m%d%H%M%S)
OUTPUT_DIR="${OUTPUT_BASE}/${TIMESTAMP}"
REMOTE_OUTPUT_DIR="${REMOTE_OUTPUT_BASE}/${TIMESTAMP}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# -----------------------------------------------------------------------------
# Helper Functions
# -----------------------------------------------------------------------------

print_usage() {
    echo "Usage: $0 [--work-dir DIR] [--venv-path PATH] [--output-dir DIR] [--remote-output-dir DIR] [-v] <client-cfg>"
    echo ""
    echo "Options:"
    echo "  --work-dir DIR        Working directory (default: $WORK_DIR)"
    echo "  --venv-path PATH      Local venv path (default: $VENV_PATH)"
    echo "  --output-dir DIR      Use this as OUTPUT_DIR (no timestamp suffix)"
    echo "  --remote-output-dir DIR  Use this as REMOTE_OUTPUT_DIR"
    echo "  -v                    Verbose build"
}

validate_file_exists() {
    local f=$1; local desc=$2
    [ -f "$f" ] || { echo "Error: $desc not found: $f"; exit 1; }
}

validate_directory_exists() {
    local d=$1; local desc=$2
    [ -d "$d" ] || { echo "Error: $desc not found: $d"; exit 1; }
}

cleanup_tmux_session() {
    tmux has-session -t "$SESSION_NAME" 2>/dev/null && tmux kill-session -t "$SESSION_NAME"
}

kill_processes_by_pattern() {
    pkill -f "$1" 2>/dev/null && pkill -9 -f "$1" 2>/dev/null || true
}

kill_remote_processes_by_pattern() {
    local pattern=$1; local remote_ips=$2
    IFS=',' read -ra IPS <<< "$remote_ips"
    for ip in "${IPS[@]}"; do
        ssh "-p${SSH_PORT}" "$ip" "pkill -f '$pattern' 2>/dev/null && sleep 3 && pkill -9 -f '$pattern' 2>/dev/null || true" 2>/dev/null || true
    done
}

build_client() {
    local wd=$1
    cd "$wd" || exit 1
    if [ "$VERBOSE" = true ]; then
        RUSTFLAGS="-Awarnings" cargo build -p request-sim --release --bin client -j64
    else
        RUSTFLAGS="-Awarnings" cargo build -p request-sim --release --bin client -j64 --quiet
    fi
    [ $? -eq 0 ] || { echo "Build client failed"; exit 1; }
}

setup_output_directory() {
    local out_dir=$1; local client_cfg=$2
    mkdir -p "$out_dir"
    cp "$client_cfg" "$out_dir/client.toml"
}

wait_for_vllm_startup() {
    local log_dir=$1
    local max_wait=600
    local elapsed=0
    local interval=5

    echo "🔍 [Local vLLM] Waiting for startup in '$log_dir' (max ${max_wait}s)..."

    while [ $elapsed -lt $max_wait ]; do
        all_ready=true
        found_any_log=false

        for logfile in "$log_dir"/vllm_local*.log; do
            if [ ! -f "$logfile" ]; then
                echo "⏳ [Local vLLM] Log file not yet created: $logfile"
                all_ready=false
                break
            fi

            found_any_log=true

            if grep -q "^RuntimeError:" "$logfile" 2>/dev/null || grep -q "^OSError:" "$logfile" 2>/dev/null; then
                echo "💥 [Local vLLM] FATAL error detected in $logfile"
                exit 1
            fi

            if ! grep -Fq "INFO:     Application startup complete." "$logfile" 2>/dev/null; then
                echo "⏳ [Local vLLM] Still waiting for startup completion in $logfile"
                all_ready=false
                break
            else
                echo "✅ [Local vLLM] Startup complete in $logfile"
            fi
        done

        if [ "$found_any_log" = false ]; then
            echo "⏳ [Local vLLM] No vllm_local*.log files found yet in $log_dir"
            all_ready=false
        fi

        if [ "$all_ready" = true ]; then
            echo "✨ [Local vLLM] All local instances started."
            return 0
        fi

        sleep $interval
        elapsed=$((elapsed + interval))
        echo "⏳ [Local vLLM] Still waiting... (${elapsed}s elapsed)"
    done

    echo "❌ [Local vLLM] Startup timeout after ${max_wait} seconds."
    exit 1
}

wait_for_remote_vllm_startup() {
    local remote_dir=$1
    local remote_ips=$2
    local max_wait=100
    local elapsed=0
    local interval=5

    IFS=',' read -ra IPS <<< "$remote_ips"
    echo "🔍 [Remote vLLM] Waiting for startup on remote nodes ${IPS[*]} in '$remote_dir' (max ${max_wait}s)..."

    while [ $elapsed -lt $max_wait ]; do
        all_ready=true

        for ip in "${IPS[@]}"; do
            echo "📡 [Remote vLLM] Checking $ip..."

            logs=$(ssh "-p${SSH_PORT}" "$ip" "ls ${remote_dir}/vllm_remote*.log 2>/dev/null" 2>/dev/null)
            if [ -z "$logs" ]; then
                echo "⏳ [Remote vLLM] No vllm_remote*.log files found yet on $ip"
                all_ready=false
                break
            fi

            for f in $logs; do
                if ssh "-p${SSH_PORT}" "$ip" "grep -q '^RuntimeError:\|^OSError:' '$f' 2>/dev/null"; then
                    echo "💥 [Remote vLLM] FATAL error detected in $f on $ip"
                    exit 1
                fi

                if ! ssh "-p${SSH_PORT}" "$ip" "grep -Fq 'INFO:     Application startup complete.' '$f' 2>/dev/null"; then
                    echo "⏳ [Remote vLLM] Still waiting for startup in $f on $ip"
                    all_ready=false
                    break 2
                else
                    echo "✅ [Remote vLLM] Startup complete in $f on $ip"
                fi
            done
        done

        if [ "$all_ready" = true ]; then
            echo "✨ [Remote vLLM] All remote instances started."
            return 0
        fi

        sleep $interval
        elapsed=$((elapsed + interval))
        echo "⏳ [Remote vLLM] Still waiting... (${elapsed}s elapsed)"
    done

    echo "❌ [Remote vLLM] Startup timeout after ${max_wait} seconds."
    exit 1
}

# # Launch vLLM in tmux windows (local + remote)
# launch_vllm_in_tmux() {
#     # Ensure session exists
#     if ! tmux has-session -t "$SESSION_NAME" 2>/dev/null; then
#         tmux new-session -d -s "$SESSION_NAME"
#     fi

#     echo "Luanching local vllms via tmux..."
#     # Local vLLM (rank 0-7)
#     local vllm_local_cmd="source '$VENV_PATH/bin/activate' && '$VENV_PATH/bin/vllm' serve '$MODEL_PATH' \
#         --data-parallel-size 16 \
#         --data-parallel-size-local 8 \
#         --data-parallel-address '$HEAD_IP' \
#         --data-parallel-rpc-port 13345 \
#         --max-num-batched-tokens 4096 \
#         --gpu-memory-utilization 0.9 \
#         > '$OUTPUT_DIR/vllm_local.log' 2>&1"

#     tmux new-window -t "$SESSION_NAME" -n vllm_local
#     tmux send-keys -t "$SESSION_NAME:vllm_local" "$vllm_local_cmd" C-m

#     echo "Luanching remote vllms via ssh..."

#     # Remote vLLM (rank 8-15)
#     if [[ "$USE_REMOTE" == "True" ]]; then
#         IFS=',' read -ra IPS <<< "$REMOTE_IPS"
#         for ip in "${IPS[@]}"; do
#             local vllm_remote_cmd="source '$REMOTE_VENV_PATH/bin/activate' && '$REMOTE_VENV_PATH/bin/vllm' serve '$REMOTE_MODEL_PATH' \
#                 --headless \
#                 --data-parallel-size 16 \
#                 --data-parallel-size-local 8 \
#                 --data-parallel-start-rank 8 \
#                 --data-parallel-address '$HEAD_IP' \
#                 --data-parallel-rpc-port 13345 \
#                 --max-num-batched-tokens 4096 \
#                 --gpu-memory-utilization 0.9 \
#                 > '$REMOTE_OUTPUT_DIR/vllm_remote.log' 2>&1 &"

#             ssh "-p ${SSH_PORT}" "$ip" "mkdir -p '$REMOTE_OUTPUT_DIR' && ($vllm_remote_cmd)" 2>/dev/null || true
#         done
#     fi
# }

launch_vllm_in_tmux() {
    # Ensure session exists
    if ! tmux has-session -t "$SESSION_NAME" 2>/dev/null; then
        tmux new-session -d -s "$SESSION_NAME"
    fi

    echo "Launching local vLLM via tmux (non-blocking)..."
    # Local vLLM (rank 0-7) — tmux is inherently non-blocking
    local vllm_local_cmd="source '$VENV_PATH/bin/activate' && '$VENV_PATH/bin/vllm' serve '$MODEL_PATH' \
        --data-parallel-size 16 \
        --data-parallel-size-local 8 \
        --data-parallel-address '$HEAD_IP' \
        --data-parallel-rpc-port 13345 \
        --max-num-batched-tokens 4096 \
        --gpu-memory-utilization 0.9 \
        > '$OUTPUT_DIR/vllm_local.log' 2>&1"

    tmux new-window -t "$SESSION_NAME" -n vllm_local
    tmux send-keys -t "$SESSION_NAME:vllm_local" "$vllm_local_cmd" C-m

    # Launch remote vLLM(s) in fully detached background (no blocking)
    if [[ "$USE_REMOTE" == "True" ]]; then
        echo "Launching remote vLLM(s) via SSH in background..."
        (
            IFS=',' read -ra IPS <<< "$REMOTE_IPS"
            for ip in "${IPS[@]}"; do
                # Ensure remote output dir exists
                ssh "-p${SSH_PORT}" "$ip" "mkdir -p '$REMOTE_OUTPUT_DIR'" >/dev/null 2>&1 || true

                # Launch vLLM in background on remote (double-backgrounding for safety)
                ssh "-p${SSH_PORT}" "$ip" " \
                    source '$REMOTE_VENV_PATH/bin/activate' 2>/dev/null || true; \
                    nohup '$REMOTE_VENV_PATH/bin/vllm' serve '$REMOTE_MODEL_PATH' \
                        --headless \
                        --data-parallel-size 16 \
                        --data-parallel-size-local 8 \
                        --data-parallel-start-rank 8 \
                        --data-parallel-address '$HEAD_IP' \
                        --data-parallel-rpc-port 13345 \
                        --max-num-batched-tokens 4096 \
                        --gpu-memory-utilization 0.9 \
                        > '$REMOTE_OUTPUT_DIR/vllm_remote.log' 2>&1 \
                        < /dev/null & \
                " >/dev/null 2>&1 &
            done
            wait  # Wait for all SSH background jobs in this subshell
        ) >/dev/null 2>&1 &
    fi

    # Function returns immediately — all work is backgrounded
}

# Launch client in its own tmux window
launch_client_in_tmux() {
    if ! tmux has-session -t "$SESSION_NAME" 2>/dev/null; then
        tmux new-session -d -s "$SESSION_NAME"
    fi

    tmux new-window -t "$SESSION_NAME" -n client
    local cmd="source '$VENV_PATH/bin/activate' && cd '$SCRIPT_DIR' && python ../../smart_runner.py --toml '$1' --output-dir='$OUTPUT_DIR' --model-path='$MODEL_PATH' --venv-path='$VENV_PATH' --work-dir='$WORK_DIR' --dataset-dir='$DATASET_DIR'"
    tmux send-keys -t "$SESSION_NAME:client" "$cmd" C-m
}

post_process_results() {
    cat "$OUTPUT_DIR"/client*.jsonl > "$OUTPUT_DIR/client.jsonl" 2>/dev/null || true

    "$VENV_PATH/bin/python" "../../figures/draw_send_gap.py" "$OUTPUT_DIR/" --time-window=2.0
    "$VENV_PATH/bin/python" "../../figures/draw_cdf.py" "$OUTPUT_DIR/"
    "$VENV_PATH/bin/python" "../../figures/draw_cumu.py" "$OUTPUT_DIR/"
    "$VENV_PATH/bin/python" "../../figures/analyze_statistics.py" "$OUTPUT_DIR/"
    "$VENV_PATH/bin/python" "../../figures/analyze_statistics_smooth.py" "$OUTPUT_DIR/"
    "$VENV_PATH/bin/python" "../../figures/analyze_statistics_smooth.py" "$OUTPUT_DIR/" --smooth-window 5 --instances {0..15}
}

cleanup_all() {
    trap - INT TERM
    echo "Cleaning up processes..."
    kill_processes_by_pattern "$VENV_PATH/bin/vllm"
    kill_processes_by_pattern "smart_runner.py"
    kill_processes_by_pattern "$WORK_DIR/target/release/client"
    pkill -f -9 "VLLM::EngineCore"
    pkill -9 -f "cpu_monitor.py" 2>/dev/null || true

    if [[ "$USE_REMOTE" == "True" ]] && [ -n "$REMOTE_IPS" ]; then
        kill_remote_processes_by_pattern "$REMOTE_VENV_PATH/bin/vllm" "$REMOTE_IPS"
        kill_remote_processes_by_pattern "VLLM::EngineCore" "$REMOTE_IPS"
    fi
    cleanup_tmux_session
}

trap cleanup_all INT
trap cleanup_all USR1
trap cleanup_all TERM

# -----------------------------------------------------------------------------
# Argument Parsing
# -----------------------------------------------------------------------------

POSITIONAL=()
VERBOSE=false
USER_SPECIFIED_OUTPUT=false
USER_SPECIFIED_REMOTE_OUTPUT=false

while [[ $# -gt 0 ]]; do
    case $1 in
        -v) VERBOSE=true; shift ;;
        --work-dir) WORK_DIR="$2"; shift 2 ;;
        --venv-path) VENV_PATH="$2"; shift 2 ;;
        --output-dir) OUTPUT_DIR="$2"; USER_SPECIFIED_OUTPUT=true; shift 2 ;;
        --remote-output-dir) REMOTE_OUTPUT_DIR="$2"; USER_SPECIFIED_REMOTE_OUTPUT=true; shift 2 ;;
        -h|--help) print_usage; exit 0 ;;
        *) POSITIONAL+=("$1"); shift ;;
    esac
done
set -- "${POSITIONAL[@]}"

if [ "$#" -ne 1 ]; then
    echo "Error: Missing <client-cfg>"
    print_usage
    exit 1
fi

CLIENT_CFG="$1"

validate_file_exists "$CLIENT_CFG" "Client config"
validate_directory_exists "$WORK_DIR" "Work dir"
validate_directory_exists "$DATASET_DIR" "Dataset dir"

# Apply timestamp only if user didn't override
[ "$USER_SPECIFIED_OUTPUT" = false ] && OUTPUT_DIR="${OUTPUT_BASE}/${TIMESTAMP}"
[ "$USER_SPECIFIED_REMOTE_OUTPUT" = false ] && REMOTE_OUTPUT_DIR="${REMOTE_OUTPUT_BASE}/${TIMESTAMP}"

# -----------------------------------------------------------------------------
# Main Execution
# -----------------------------------------------------------------------------


# Build client
build_client "$WORK_DIR"

cd $SCRIPT_DIR
echo "cd back to $SCRIPT_DIR"

# Setup output
setup_output_directory "$OUTPUT_DIR" "$CLIENT_CFG"

# Cleanup leftovers
cleanup_all

# Create output dirs
mkdir -p "$OUTPUT_DIR"
[ "$USE_REMOTE" == "True" ] && ssh "-p$SSH_PORT" "$REMOTE_IPS" "mkdir -p '$REMOTE_OUTPUT_DIR'" 2>/dev/null || true

# Launch vLLM in tmux
echo "🚀 Launching vLLM backends in tmux windows..."
launch_vllm_in_tmux
wait_for_vllm_startup "$OUTPUT_DIR"

if [[ "$USE_REMOTE" == "True" ]]; then
    wait_for_remote_vllm_startup "$REMOTE_OUTPUT_DIR" "$REMOTE_IPS"
fi

# Launch client in tmux
echo "📤 Launching client in tmux window..."
launch_client_in_tmux "$CLIENT_CFG"
sleep $CLIENT_WAIT_TIME

# Wait for client to finish
client_bin="$WORK_DIR/target/release/client"
if pgrep -f "$client_bin" > /dev/null; then
    echo "⏳ Client still running; waiting up to 60s..."
    timeout=60
    while [ $timeout -gt 0 ] && pgrep -f "$client_bin" > /dev/null; do
        sleep 5; timeout=$((timeout - 5))
    done
fi

# Final cleanup
cleanup_all

# Post-process
post_process_results

echo "✅ Experiment completed!"
echo "📁 Results: $OUTPUT_DIR"


# Run on node1
# VLLM_ATTENTION_BACKEND = "FLASHINFER" VLLM_USE_FLASHINFER_SAMPLER=0 OMP_NUM_THREADS=1 /mnt/debugger/hjb/node1/original_vllm/.venv/bin/python3 /mnt/debugger/hjb/node1/original_vllm/.venv/bin/vllm serve /home/admin/resource/model/464482ce.Qwen2.5-7B-Instruct/1.0/ --data-parallel-size 16 --data-parallel-size-local 8 --data-parallel-address 172.27.69.15 --data-parallel-rpc-port 13345 --max-num-batched-tokens 4096 --gpu-memory-utilization 0.9

# Run on node2
# VLLM_ATTENTION_BACKEND = "FLASHINFER" VLLM_USE_FLASHINFER_SAMPLER=0 OMP_NUM_THREADS=1 /mnt/debugger/hjb/node2/original_vllm/.venv/bin/python3 /mnt/debugger/hjb/node2/original_vllm/.venv/bin/vllm serve /home/admin/resource/model/464482ce.Qwen2.5-7B-Instruct/1.0/ --headless --data-parallel-size 16 --data-parallel-size-local 8 --data-parallel-start-rank 8 --data-parallel-address 172.27.69.15 --data-parallel-rpc-port 13345 --max-num-batched-tokens 4096 --gpu-memory-utilization 0.9