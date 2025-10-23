#!/bin/bash
# remote_decode_launcher.sh - executed on DECODE_IP

set -euo pipefail

DECODE_MODEL='/home/hanjinbo.hjb/Qwen2.5-7B-Instruct'
DECODE_VENV_PATH='/home/hanjinbo.hjb/yaullm/.venv'
DECODE_SCRIPT_PATH='/home/hanjinbo.hjb/blitz-infer-pack/scripts/e2e'

if [ $# -eq 0 ] || [ $(( $# % 3 )) -ne 0 ]; then
    echo "Usage: $0 <gpu_id1> <port1> <kv_port1> [<gpu_id2> <port2> <kv_port2> ...]"
    exit 1
fi

cd "$DECODE_SCRIPT_PATH"
source "$DECODE_VENV_PATH/bin/activate"

sleep 5
echo "$PWD"

# Kill and recreate kvtest session
tmux kill-session -t kvtest 2>/dev/null || true
tmux new-session -d -s kvtest -n base 'sleep infinity'

# Use shift to consume arguments in groups of 3
index=0
while [ $# -gt 0 ]; do
    gpu_id="$1"
    port="$2"
    kv_port="$3"
    shift 3

    window_name="decode-$index"
    log_file="$DECODE_SCRIPT_PATH/decode$((index + 1)).log"

    # Build kv_transfer_config JSON (escape double quotes for shell)
    kv_config="{\"kv_connector\":\"P2pNcclConnector\",\"kv_role\":\"kv_consumer\",\"kv_buffer_size\":\"1e1\",\"kv_port\":\"$kv_port\",\"kv_connector_extra_config\":{\"proxy_ip\":\"$PROXY_IP\",\"proxy_port\":\"$PROXY_PORT\",\"http_port\":\"$port\",\"nccl_num_channels\":\"16\"}}"

    # Launch in tmux window
    tmux new-window -t kvtest -n "$window_name" \
        "CUDA_VISIBLE_DEVICES=$gpu_id VLLM_USE_V1=1 $DECODE_VENV_PATH/bin/vllm serve '$DECODE_MODEL' \
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
        --gpu-memory-utilization 0.7 \
        --kv-transfer-config '$kv_config' \
        2>&1 | tee '$log_file'"

    index=$((index + 1))
done

# Clean up base window
# tmux kill-window -t kvtest:base 2>/dev/null || true

echo "Remote decode servers launched in tmux session 'kvtest'."