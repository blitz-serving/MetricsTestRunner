import re
import argparse
import matplotlib.pyplot as plt
import json
from collections import defaultdict
import numpy as np

def parse_predicted_ttft(log_lines):
    """
    解析预测的 TTFT
    示例：
    Simulator AddRequest completed for request_id: 3, system_metrics: 198.38895
    """
    pred_pattern = re.compile(r"request_id:\s*(\d+),\s*system_metrics:\s*([0-9.]+)")
    preds = {}
    for line in log_lines:
        m = pred_pattern.search(line)
        if m:
            rid = int(m.group(1))
            preds[rid] = float(m.group(2))  # 单位 ms
    return preds


def parse_real_ttft_from_jsonl(path):
    """
    从 client.jsonl 读取真实执行的 TTFT
    示例：
    {"request_id":"255","first_token_time":"941", ...}
    """
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


def plot_error_cdf(preds, reals):
    """
    绘制误差比例的 CDF 曲线，并输出每个 request 的误差信息
    """
    errors = []
    table_rows = []

    print("\n=== TTFT 对比明细（单位：ms） ===")
    print(f"{'request_id':>10} | {'pred(ms)':>10} | {'real(ms)':>10} | {'diff(ms)':>10} | {'error_ratio':>12}")
    print("-" * 60)

    for rid, real_val in sorted(reals.items()):
        if rid in preds:
            pred_val = preds[rid]
            diff = pred_val - real_val
            error_ratio = abs(diff) / real_val
            errors.append(error_ratio)
            table_rows.append((rid, pred_val, real_val, diff, error_ratio))
            print(f"{rid:10d} | {pred_val:10.2f} | {real_val:10.2f} | {diff:10.2f} | {error_ratio:12.4f}")

    if not errors:
        print("❌ 未找到匹配的 request_id，请检查日志。")
        return

    # === 计算 CDF ===
    errors = np.array(sorted(errors))
    cdf = np.arange(1, len(errors) + 1) / len(errors)

    plt.figure(figsize=(8, 5))
    plt.plot(errors, cdf, linestyle="-", color="blue")
    plt.xlabel("Error ratio (|pred - real| / real)")
    plt.ylabel("Cumulative Probability")
    plt.title("TTFT Error Ratio CDF")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()

    # 导出明细 CSV
    with open("ttft_error_details.csv", "w", encoding="utf-8") as f:
        f.write("request_id,pred_ms,real_ms,diff_ms,error_ratio\n")
        for rid, p, r, d, e in table_rows:
            f.write(f"{rid},{p:.4f},{r:.4f},{d:.4f},{e:.6f}\n")
    print("\n✅ 明细已导出至 ttft_error_details.csv")


def main():
    parser = argparse.ArgumentParser(
        description="Compare predicted and real TTFT from simulator + client.jsonl logs, and plot CDF."
    )
    parser.add_argument("--log", required=True, help="Path to simulator log file (for predicted TTFT).")
    parser.add_argument("--client", required=True, help="Path to client.jsonl file (for real TTFT).")
    args = parser.parse_args()

    with open(args.log, "r", encoding="utf-8") as f:
        sim_lines = f.readlines()

    preds = parse_predicted_ttft(sim_lines)
    reals = parse_real_ttft_from_jsonl(args.client)

    print(f"✅ 提取到预测TTFT条目: {len(preds)}")
    print(f"✅ 提取到真实TTFT条目: {len(reals)}")

    matched = set(preds.keys()) & set(reals.keys())
    print(f"✅ 匹配成功的 request_id 数量: {len(matched)}")

    plot_error_cdf(preds, reals)


if __name__ == "__main__":
    main()
