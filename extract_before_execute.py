import re
import argparse
from collections import defaultdict
import matplotlib.pyplot as plt
import csv


def parse_by_line(log_path):
    """逐行解析 StepSync 日志（适合每条记录为单行的情况）"""
    outputs_pat = re.compile(r"VllmRequestStatus\s*\{")
    before_exec_pat = re.compile(r"before_execute=([0-9.]+)")
    trigger = "Simulator received StepSync command:"

    mapping = defaultdict(list)
    x_vals, y_vals = [], []

    with open(log_path, "r", encoding="utf-8") as f:
        for line in f:
            if trigger not in line:
                continue
            num_outputs = len(outputs_pat.findall(line))
            m = before_exec_pat.search(line)
            if m:
                before_exec = float(m.group(1))
                mapping[num_outputs].append(before_exec)
                x_vals.append(num_outputs)
                y_vals.append(before_exec)

    return mapping, x_vals, y_vals


def parse_fallback_block(log_path):
    """跨行解析，用于日志分多行的情况"""
    with open(log_path, "r", encoding="utf-8") as f:
        content = f.read()

    block_pat = re.compile(
        r"Simulator received StepSync command:(.*?op_exec_log:\s*Some\(\".*?\"\))",
        re.DOTALL,
    )
    outputs_pat = re.compile(r"VllmRequestStatus\s*\{")
    before_exec_pat = re.compile(r"before_execute=([0-9.]+)")

    mapping = defaultdict(list)
    x_vals, y_vals = [], []

    for m in block_pat.finditer(content):
        block = m.group(1)
        num_outputs = len(outputs_pat.findall(block))
        be = before_exec_pat.search(block)
        if be:
            val = float(be.group(1))
            mapping[num_outputs].append(val)
            x_vals.append(num_outputs)
            y_vals.append(val)

    return mapping, x_vals, y_vals


def parse_log(log_path):
    """自动选择解析方式"""
    mapping, x_vals, y_vals = parse_by_line(log_path)
    if not x_vals:
        mapping, x_vals, y_vals = parse_fallback_block(log_path)
    return mapping, x_vals, y_vals


def plot_scatter(x_vals, y_vals, output_file="before_execute_scatter.png"):
    """绘制散点图"""
    if not x_vals:
        return False

    plt.figure(figsize=(8, 6))
    plt.scatter(x_vals, y_vals, alpha=0.7, edgecolors="black")
    plt.title("Outputs Count vs before_execute")
    plt.xlabel("Number of VllmRequestStatus (outputs)")
    plt.ylabel("before_execute (ms)")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(output_file, dpi=150)
    return True


def main():
    parser = argparse.ArgumentParser(
        description="从 router_v2 日志中提取 outputs 数量与 before_execute 的映射，并绘制散点图"
    )
    parser.add_argument("--log", required=True, help="日志文件路径")
    parser.add_argument("--out", default="before_execute_scatter.png", help="输出图像文件名")
    parser.add_argument("--csv", default="before_execute_avg.csv", help="输出 CSV 文件名")
    args = parser.parse_args()

    mapping, x_vals, y_vals = parse_log(args.log)

    print("=== 输出数量 -> before_execute(ms) ===")
    if not x_vals:
        print("❌ 未提取到任何 before_execute 数据，请检查日志格式。")
        return

    # 写 CSV
    with open(args.csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["batch_size", "predictions"])

        for n, vals in sorted(mapping.items()):
            avg = sum(vals) / len(vals)
            writer.writerow([n, avg])
            print(f"{n:4d} 个 outputs -> 平均 before_execute: {avg:.3f} ms (共 {len(vals)} 条)")

    print(f"\n✅ 已保存平均时长到 CSV: {args.csv}")

    if plot_scatter(x_vals, y_vals, args.out):
        print(f"✅ 已生成散点图: {args.out}（点数: {len(x_vals)}）")


if __name__ == "__main__":
    main()
