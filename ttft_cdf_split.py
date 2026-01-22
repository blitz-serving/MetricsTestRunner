import re
import argparse
import matplotlib.pyplot as plt
import json
import numpy as np

def parse_predicted_ttft(log_lines):
    """
    解析日志中的预测 TTFT，只保留「实际被指派的实例」的那一条预测。
    """
    pred_pattern = re.compile(
        r"Request_(\d+)\s+estimated ttft:\s*([0-9.]+)\s*ms.*?on\s+Vllm#(\d+)"
    )

    assign_pattern = re.compile(
        r"Assigning Request_(\d+)\s+to Replica#(\d+)"
    )

    preds_all = {}
    assigned_replica = {}

    for line in log_lines:
        m_pred = pred_pattern.search(line)
        if m_pred:
            rid = int(m_pred.group(1))
            ttft = float(m_pred.group(2))
            replica_id = int(m_pred.group(3))
            preds_all[(rid, replica_id)] = ttft
            continue

        m_assign = assign_pattern.search(line)
        if m_assign:
            rid = int(m_assign.group(1))
            replica_id = int(m_assign.group(2))
            assigned_replica[rid] = replica_id

    preds = {}
    missing = 0
    for rid, replica_id in assigned_replica.items():
        key = (rid, replica_id)
        if key in preds_all:
            preds[rid] = preds_all[key]
        else:
            missing += 1

    if assigned_replica == {}:
        for (rid, replica_id), ttft in preds_all.items():
            preds[rid] = ttft

    print(f"✅ 解析到预测行数量（含所有实例）：{len(preds_all)}")
    print(f"✅ 解析到指派行数量：{len(assigned_replica)}")
    print(f"✅ 成功匹配到 (request_id, assigned_replica) 的预测条目：{len(preds)}")
    if missing > 0:
        print(f"⚠ 有 {missing} 个请求找不到对应实例的预测记录，请检查日志是否完整。")

    return preds


def parse_real_ttft_from_jsonl(path):
    """从 client.jsonl 读取真实执行 TTFT"""
    reals = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                if "request_id" in obj and "first_token_time" in obj:
                    rid = int(obj["request_id"])
                    real_ttft = float(obj["first_token_time"])
                    reals[rid] = real_ttft
            except Exception:
                continue
    return reals


def plot_error_cdf(preds, reals, threshold=2000):
    """
    按真实 TTFT 是否大于阈值分组计算误差比例 CDF
    横轴强制扩展到 [0, 1]，避免缩放过小
    """
    short_errors, long_errors = [], []
    table_rows = []

    short_errors, long_errors = [], []

    print("\n=== TTFT 对比明细（单位：ms） ===")
    print(f"{'request_id':>10} | {'pred(ms)':>10} | {'real(ms)':>10} | {'diff(ms)':>10} | {'error_ratio':>12}")
    print("-" * 60)

    for rid, real_val in sorted(reals.items()):
        if rid in preds:
            pred_val = preds[rid]
            diff = pred_val - real_val
            error_ratio = abs(diff) / real_val
            table_rows.append((rid, pred_val, real_val, diff, error_ratio))

            if real_val <= threshold:
                short_errors.append(error_ratio)
            else:
                long_errors.append(error_ratio)

            print(f"{rid:10d} | {pred_val:10.2f} | {real_val:10.2f} | {diff:10.2f} | {error_ratio:12.4f}")

    if not short_errors and not long_errors:
        print("❌ 未找到匹配的 request_id，请检查日志。")
        return

    # === 绘制 CDF ===
    plt.figure(figsize=(8, 5))
    colors = {"short": "tab:blue", "long": "tab:orange"}

    if short_errors:
        short_errors = np.array(sorted(short_errors))
        short_cdf = np.arange(1, len(short_errors) + 1) / len(short_errors)
        plt.plot(short_errors, short_cdf, color=colors["short"],
                 label=f"Real ≤ {threshold}ms (n={len(short_errors)})")

    if long_errors:
        long_errors = np.array(sorted(long_errors))
        long_cdf = np.arange(1, len(long_errors) + 1) / len(long_errors)
        plt.plot(long_errors, long_cdf, color=colors["long"],
                 label=f"Real > {threshold}ms (n={len(long_errors)})")

    # 🔧 新增：强制 X 轴缩放到 1
    plt.xlim(0, 1)

    plt.xlabel("Error ratio (|pred - real| / real)")
    plt.ylabel("Cumulative Probability")
    plt.title(f"TTFT Error Ratio CDF (split by real_ttft={threshold}ms)")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()
    plt.savefig("comparison.png")

    # === 导出明细 CSV ===
    with open("ttft_error_details_split.csv", "w", encoding="utf-8") as f:
        f.write("request_id,pred_ms,real_ms,diff_ms,error_ratio,group\n")
        for rid, p, r, d, e in table_rows:
            group = "short" if r <= threshold else "long"
            f.write(f"{rid},{p:.4f},{r:.4f},{d:.4f},{e:.6f},{group}\n")

    print("\n=== 🔺 误差最大的 Top 10 请求（按 error_ratio 降序） ===")
    top_k = sorted(table_rows, key=lambda x: x[4], reverse=True)[:30]
    print(f"{'request_id':>10} | {'pred(ms)':>10} | {'real(ms)':>10} | {'diff(ms)':>10} | {'error_ratio':>12}")
    print("-" * 60)
    for rid, p, r, d, e in top_k:
        print(f"{rid:10d} | {p:10.2f} | {r:10.2f} | {d:10.2f} | {e:12.4f}")


def main():
    parser = argparse.ArgumentParser(description="Compare predicted and real TTFT, plot CDF by real_ttft threshold.")
    parser.add_argument("--log", required=True, help="Path to simulator log file (for predicted TTFT).")
    parser.add_argument("--client", required=True, help="Path to client.jsonl file (for real TTFT).")
    parser.add_argument("--threshold", type=float, default=100,
                        help="Threshold (ms) to split real_ttft groups (default=100ms).")
    args = parser.parse_args()

    with open(args.log, "r", encoding="utf-8") as f:
        sim_lines = f.readlines()

    preds = parse_predicted_ttft(sim_lines)
    reals = parse_real_ttft_from_jsonl(args.client)

    print(f"✅ 提取到预测TTFT条目: {len(preds)}")
    print(f"✅ 提取到真实TTFT条目: {len(reals)}")

    matched = set(preds.keys()) & set(reals.keys())
    print(f"✅ 匹配成功的 request_id 数量: {len(matched)}")

    plot_error_cdf(preds, reals, args.threshold)


if __name__ == "__main__":
    main()
