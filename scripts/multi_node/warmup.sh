#!/bin/bash

set -e

# === 配置 ===
MODEL_PATH="/mnt/debugger/hjb/models/Qwen3-30B-A3B/"
REMOTE_IP="172.27.21.64"
SSH_PORT=10022

LOCAL_LOG="/tmp/node1/vllm_local.log"
REMOTE_LOG="/tmp/node2/vllm_remote.log"
LOCAL_DONE="/tmp/vllm_warmup_local_done"
REMOTE_DONE="/tmp/vllm_warmup_remote_done"

# 清理旧状态
rm -f "$LOCAL_DONE" "$REMOTE_DONE"
mkdir -p /tmp/node1 /tmp/node2

# === 清理可能残留的旧进程（关键！）===
echo "[CLEAN] Cleaning up old vLLM and tail processes..."

# 本地：清理 vLLM + 本地 tail
pkill -9 -f "vllm serve" 2>/dev/null || true
pkill -9 -f "tail -n0 -F $LOCAL_LOG" 2>/dev/null || true

# 远程：清理 vLLM + 远程 tail
ssh -p "$SSH_PORT" "$REMOTE_IP" "
  pkill -9 -f 'vllm serve' 2>/dev/null || true
  pkill -9 -f 'tail -n0 -F $REMOTE_LOG' 2>/dev/null || true
"

sleep 2  # 确保 kill 生效

# === 1. 启动本地 vLLM ===
echo "[LOCAL] Starting vLLM serve..."
nohup /mnt/debugger/hjb/node1/yaullm/.venvflashinfer/bin/vllm serve "$MODEL_PATH" > "$LOCAL_LOG" 2>&1 &
echo "[LOCAL] Launched."

# === 2. 启动远程 vLLM ===
echo "[REMOTE] Starting vLLM serve on $REMOTE_IP..."
ssh -p "$SSH_PORT" "$REMOTE_IP" "
  mkdir -p /tmp/node2
  nohup /mnt/debugger/hjb/node2/yaullm/.venvflashinfer/bin/vllm serve '$MODEL_PATH' > $REMOTE_LOG 2>&1 &
"
echo "[REMOTE] Launched."

# === 3. 启动本地日志监控 ===
echo "[MONITOR] Starting local log monitor..."
nohup tail -n0 -F "$LOCAL_LOG" | while read line; do
  if [[ "$line" == *"INFO:     Application startup complete."* ]]; then
    echo "[LOCAL] Startup complete detected."
    touch "$LOCAL_DONE"
    break
  fi
done &
# 注意：这里不再用 $!，因为后面的 pkill 不依赖它

# === 4. 启动远程日志监控 ===
echo "[MONITOR] Starting remote log monitor..."
nohup ssh -p "$SSH_PORT" "$REMOTE_IP" "tail -n0 -F $REMOTE_LOG" | while read line; do
  if [[ "$line" == *"INFO:     Application startup complete."* ]]; then
    echo "[REMOTE] Startup complete detected."
    touch "$REMOTE_DONE"
    break
  fi
done &

# === 5. 等待双方完成（最多 8 分钟）===
echo "[WAIT] Waiting for both sides to complete startup..."
if ! timeout 480 bash -c "while [ ! -f '$LOCAL_DONE' ] || [ ! -f '$REMOTE_DONE' ]; do sleep 1; done"; then
  echo "[WARN] Timeout (8 min). Proceeding to cleanup."
fi

# === 6. 强制清理所有相关进程（不依赖 PID）===
echo "[CLEANUP] Killing vLLM and tail processes..."

# 本地清理
pkill -9 -f "vllm serve" || true
pkill -9 -f "tail -n0 -F $LOCAL_LOG" || true

# 远程清理
ssh -p "$SSH_PORT" "$REMOTE_IP" "
  pkill -9 -f 'vllm serve' || true
  pkill -9 -f 'tail -n0 -F $REMOTE_LOG' || true
"

# === 7. 清理临时文件 ===
rm -f "$LOCAL_DONE" "$REMOTE_DONE"

echo "Warmup completed."