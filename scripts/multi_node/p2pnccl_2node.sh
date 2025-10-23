#!/bin/bash

# =============================================================================
# vLLM Disaggregated Serving Script - P2P NCCL XpYd Architecture (tmux version)
# =============================================================================
# This script demonstrates disaggregated prefill and decode serving using
# P2P NCCL communication, with all servers launched inside a single 'kvtest'
# tmux session per machine (local and remote).
# =============================================================================

# Configuration - can be overridden via environment variables
MODEL='/home/hanjinbo.hjb/Qwen2.5-7B-Instruct'
VENV_PATH='/home/hanjinbo.hjb/yaullm/.venv'
LOCAL_SCRIPT_PATH="$(dirname "${BASH_SOURCE[0]}")"
DECODE_MODEL='/home/hanjinbo.hjb/Qwen2.5-7B-Instruct'
DECODE_VENV_PATH='/home/hanjinbo.hjb/yaullm/.venv'
DECODE_SCRIPT_PATH='/home/hanjinbo.hjb/blitz-infer-pack/scripts/e2e'
TIMEOUT_SECONDS=${TIMEOUT_SECONDS:-1200}
PROXY_PORT=${PROXY_PORT:-30001}

# 3P1D configuration
# PREFILL_GPUS=${PREFILL_GPUS:-0,1,2}
# DECODE_GPUS=${DECODE_GPUS:-7}
# PREFILL_PORTS=${PREFILL_PORTS:-20005,20007,20009}
# DECODE_PORTS=${DECODE_PORTS:-20003}
# PREFILL_IP="33.254.160.17"
# DECODE_IP="33.254.160.19"
# CONFIG_NAME="3p1d_2node"

# 1P1D config
PREFILL_GPUS=${PREFILL_GPUS:-7}
DECODE_GPUS=${DECODE_GPUS:-2}
PREFILL_PORTS=${PREFILL_PORTS:-20005}
DECODE_PORTS=${DECODE_PORTS:-20003}
PREFILL_IP="33.254.160.17"
DECODE_IP="33.254.160.19"
CONFIG_NAME="1p1d_2node"

echo "Warning: P2P NCCL disaggregated prefill XpYd support for vLLM v1 is experimental."
echo ""
echo "Architecture Configuration:"
echo "  Model: $MODEL"
echo "  Prefill GPUs: $PREFILL_GPUS, Ports: $PREFILL_PORTS"
echo "  Decode GPUs: $DECODE_GPUS, Ports: $DECODE_PORTS"
echo "  Proxy Port: $PROXY_PORT"
echo "  Prefill IP: $PREFILL_IP Decode IP: $DECODE_IP"
echo "  Timeout: ${TIMEOUT_SECONDS}s"
echo ""

PIDS=()

# Global arrays for cleanup
declare -a PREFILL_GPU_ARRAY
declare -a DECODE_GPU_ARRAY

# Switch to script dir
cd "$(dirname "${BASH_SOURCE[0]}")"

check_required_files() {
    local files=("disagg_proxy_p2p_nccl_xpyd.py")
    for file in "${files[@]}"; do
        if [[ ! -f "$file" ]]; then
            echo "Required file $file not found in $(pwd)"
            exit 1
        fi
    done
}

ensure_python_library_installed() {
    echo "Checking if $1 is installed..."
    if ! python3 -c "import $1" > /dev/null 2>&1; then
        echo "$1 is not installed. Please install it via pip install $1."
        exit 1
    else
        echo "$1 is installed."
    fi
}

cleanup() {
    echo "Stopping everything…"

    # Prevent re-entrancy
    trap - INT TERM

    # Kill proxy
    pkill -9 -f "disagg_proxy_p2p_nccl_xpyd.py"

    # Kill local kvtest tmux session

    # tmux kill-session -t kvtest 2>/dev/null || true

    # Kill remote kvtest tmux session

    # if [[ -n "$DECODE_IP" && "$DECODE_IP" != "127.0.0.1" && "$DECODE_IP" != "localhost" ]]; then
    #     echo "Terminating remote kvtest tmux session on $DECODE_IP..."
    #     ssh -o ConnectTimeout=5 -o StrictHostKeyChecking=no "$DECODE_IP" \
    #         "tmux kill-session -t kvtest 2>/dev/null || true" 2>/dev/null || true
    # fi

    # Kill any leftover vLLM processes (fallback)
    pkill -9 -f "$VENV_PATH/bin/vllm" 2>/dev/null || true

    if [[ -n "$DECODE_IP" && "$DECODE_IP" != "127.0.0.1" ]]; then
        ssh -o ConnectTimeout=5 -o StrictHostKeyChecking=no "$DECODE_IP" \
            "pkill -9 -f '$DECODE_VENV_PATH/bin/vllm'" 2>/dev/null || true
    fi

    wait
    exit 0
}

wait_for_server() {
  local ip=$1
  local port=$2
  local timeout_seconds=$TIMEOUT_SECONDS
  local start_time=$(date +%s)

  echo "Waiting for server on port $port..."

  while true; do
    if curl -s "${ip}:${port}/v1/completions" > /dev/null; then
      echo "Server on port $port is ready."
      return 0
    fi

    local now=$(date +%s)
    if (( now - start_time >= timeout_seconds )); then
      echo "Timeout waiting for server on port $port"
      return 1
    fi

    sleep 3
  done
}

main() {
    check_required_files
    ensure_python_library_installed pandas
    ensure_python_library_installed datasets
    ensure_python_library_installed vllm
    ensure_python_library_installed quart

    trap cleanup INT
    trap cleanup USR1
    trap cleanup TERM

    echo "Launching servers in single 'kvtest' tmux session per machine..."

    # =============================================================================
    # Launch Proxy Server
    # =============================================================================
    echo ""
    echo "Starting proxy server on port $PROXY_PORT..."
    
    # Handmake router
    python3 disagg_proxy_p2p_nccl_xpyd.py &

    # vLLM router
    # TOOD, is it correct?
    # $VENV_PATH/bin/python -m vllm.entrypoints.disaggregated_api_server \
    #     --role router \
    #     --router-host 0.0.0.0 \
    #     --router-port 10001 \
    #     --proxy-port 30001 &
        
    PIDS+=($!)

    # Parse GPU and port arrays (global for cleanup)
    IFS=',' read -ra PREFILL_GPU_ARRAY <<< "$PREFILL_GPUS"
    IFS=',' read -ra DECODE_GPU_ARRAY <<< "$DECODE_GPUS"
    IFS=',' read -ra PREFILL_PORT_ARRAY <<< "$PREFILL_PORTS"
    IFS=',' read -ra DECODE_PORT_ARRAY <<< "$DECODE_PORTS"

    # =============================================================================
    # Setup LOCAL kvtest tmux session for Prefill Servers
    # =============================================================================
    echo ""
    echo "Setting up local 'kvtest' tmux session for ${#PREFILL_GPU_ARRAY[@]} prefill server(s)..."

    # Kill existing session
    tmux kill-session -t kvtest 2>/dev/null || true

    # Create base session (first window will be replaced)
    tmux new-session -d -s kvtest -n base "sleep infinity"

    for i in "${!PREFILL_GPU_ARRAY[@]}"; do
        local gpu_id=${PREFILL_GPU_ARRAY[$i]}
        local port=${PREFILL_PORT_ARRAY[$i]}
        local kv_port=$((24001 + i))
        local log_file="prefill$((i+1)).log"
        local window_name="prefill-$i"

        echo "  Adding window '$window_name': GPU $gpu_id, Port $port"

        # Add new window and run command
        tmux new-window -t kvtest -n "$window_name" \
            "CUDA_VISIBLE_DEVICES=$gpu_id VLLM_USE_V1=1 $VENV_PATH/bin/vllm serve $MODEL \
            --enforce-eager \
            --host 0.0.0.0 \
            --port $port \
            --tensor-parallel-size 1 \
            --seed 1024 \
            --dtype float16 \
            --max-model-len 10000 \
            --max-num-batched-tokens 10000 \
            --max-num-seqs 256 \
            --trust-remote-code \
            --gpu-memory-utilization 0.9 \
            --kv-transfer-config '{\"kv_connector\":\"P2pNcclConnector\",\"kv_role\":\"kv_producer\",\"kv_buffer_size\":\"1e1\",\"kv_port\":\"$kv_port\",\"kv_connector_extra_config\":{\"proxy_ip\":\"0.0.0.0\",\"proxy_port\":\"$PROXY_PORT\",\"http_port\":\"$port\",\"send_type\":\"PUT_ASYNC\",\"nccl_num_channels\":\"16\"}}' \
            2>&1 | tee '$log_file'"
    done

    # Remove the initial 'base' window
    tmux kill-window -t kvtest:base 2>/dev/null || true

    # =============================================================================
    # Setup REMOTE kvtest tmux session for Decode Servers
    # =============================================================================
    echo ""
    echo "Setting up remote 'kvtest' tmux session on $DECODE_IP for ${#DECODE_GPU_ARRAY[@]} decode server(s)..."

    local remote_script_path="./remote_decode_launcher.sh"

    # Build argument list: gpu_id port kv_port ...
    local args=()
    for i in "${!DECODE_GPU_ARRAY[@]}"; do
        local gpu_id=${DECODE_GPU_ARRAY[$i]}
        local port=${DECODE_PORT_ARRAY[$i]}
        local kv_port=$((22001 + i))
        args+=("$gpu_id" "$port" "$kv_port")
    done

    echo "Uploading and executing remote launcher on $DECODE_IP..."
    # upload script
    scp -o StrictHostKeyChecking=no "$remote_script_path" "$DECODE_IP:$DECODE_SCRIPT_PATH/"
    # executing script
    ssh -o StrictHostKeyChecking=no "$DECODE_IP" \
        "PROXY_IP='$PREFILL_IP' PROXY_PORT='$PROXY_PORT' bash $DECODE_SCRIPT_PATH/$(basename "$remote_script_path") ${args[*]}"

    echo "  Added remote window '$window_name': GPU $gpu_id, Port $port"

    # =============================================================================
    # Wait for All Servers to Start
    # =============================================================================
    echo ""
    echo "Waiting for all servers to start..."
    for port in "${PREFILL_PORT_ARRAY[@]}"; do
        if ! wait_for_server "$PREFILL_IP" "$port"; then
            echo "Failed to start prefill server on port $port"
            cleanup
            exit 1
        fi
    done
    for port in "${DECODE_PORT_ARRAY[@]}"; do
        if ! wait_for_server "$DECODE_IP" "$port"; then
            echo "Failed to start decode server on port $port"
            cleanup
            exit 1
        fi
    done

    echo ""
    echo "All servers are up. Starting benchmark..."

    # =============================================================================
    # Run Benchmark
    # =============================================================================
    # --burstiness 100
    $VENV_PATH/bin/vllm bench serve --port 10001 --seed $(date +%s) \
        --model "$MODEL" \
        --dataset-name random --random-input-len 1000 --random-output-len 10 \
        --num-prompts 1 --request-rate 0.5 | tee benchmark_$CONFIG_NAME.log

    echo "Benchmarking done. Cleaning up..."
    #sleep 9000
    cleanup
}

main "$@"