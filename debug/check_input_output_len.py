import re
import json
import matplotlib.pyplot as plt
import os

LOG_FILE = None
JSONL_FILE = None

assert(LOG_FILE is not None)
assert(JSONL_FILE is not None)

log_pattern = re.compile(r"Request_(\d+).*?input length (\d+).*?output length (\d+)")

log_data = {}
json_data = {}

# ---------- 解析 log ----------
with open(LOG_FILE, "r") as f:
    for line in f:
        m = log_pattern.search(line)
        if m:
            req_id = int(m.group(1))
            log_data[req_id] = {
                "input": int(m.group(2)),
                "output": int(m.group(3)),
            }

# ---------- 解析 jsonl ----------
with open(JSONL_FILE, "r") as f:
    for line in f:
        j = json.loads(line)
        if j["status"] != "200":
            continue
        req_id = int(j["request_id"])
        json_data[req_id] = {
            "input": int(j["input_length"]),
            "output": int(j["output_length"]),
        }

# ---------- 合并 ----------
matched_ids = sorted(set(log_data.keys()) & set(json_data.keys()))
print(f"Matched {len(matched_ids)} requests")

if not matched_ids:
    print("No matching request_id between log and jsonl.")
    exit()

log_inputs = [log_data[i]["input"] for i in matched_ids]
json_inputs = [json_data[i]["input"] for i in matched_ids]

log_outputs = [log_data[i]["output"] for i in matched_ids]
json_outputs = [json_data[i]["output"] for i in matched_ids]

# ---------- 作图：两个 subplots ----------
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# --- subplot 1: input length 对比 ---
axes[0].scatter(log_inputs, json_inputs, alpha=0.7)
axes[0].plot(
    [min(log_inputs), max(log_inputs)],
    [min(log_inputs), max(log_inputs)],
    linestyle="--",
)
axes[0].set_title("Input Length: log vs jsonl")
axes[0].set_xlabel("log input length")
axes[0].set_ylabel("jsonl input length")
axes[0].grid(True)

# --- subplot 2: output length 对比 ---
axes[1].scatter(log_outputs, json_outputs, alpha=0.7)
axes[1].plot(
    [min(log_outputs), max(log_outputs)],
    [min(log_outputs), max(log_outputs)],
    linestyle="--",
)
axes[1].set_title("Output Length: log vs jsonl")
axes[1].set_xlabel("log output length")
axes[1].set_ylabel("jsonl output length")
axes[1].grid(True)

plt.tight_layout()

# ---------- 保存 ----------
save_path = os.path.join(os.path.dirname(__file__), "input_output_len.png")
plt.savefig(save_path)
print(f"Saved plot to {save_path}")
