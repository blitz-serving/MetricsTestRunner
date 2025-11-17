import re
import argparse
from collections import defaultdict
import matplotlib.pyplot as plt


def parse_by_line(log_path):
    """
    优先：逐行解析。适用于 StepSync 全部落在同一行的日志（常见）。
    返回：
      mapping: {outputs_count: [after_execute, ...]}
      points: (x_vals, y_vals)
    """
    outputs_pat = re.compile(r"VllmRequestStatus\s*\{")
    after_exec_pat = re.compile(r"after_execute=([0-9.]+)")
    trigger = "Simulator received StepSync command:"

    mapping = defaultdict(list)
    x_vals, y_vals = [], []

    with open(log_path, "r", encoding="utf-8") as f:
        for line in f:
            if trigger not in line:
                continue
            num_outputs = len(outputs_pat.findall(line))
            m = after_exec_pat.search(line)
            if m:
                after_exec = float(m.group(1))
                mapping[num_outputs].append(after_exec)
                x_vals.append(num_outputs)
                y_vals.append(after_exec)

    return mapping, x_vals, y_vals


def parse_fallback_block(log_path):
    """
    回退：跨行解析。匹配从 StepSync 到 op_exec_log 的块，避免早停。
    """
    with open(log_path, "r", encoding="utf-8") as f:
        content = f.read()

    # 捕获每个 StepSync 块，并确保包含 op_exec_log
    # 使用懒惰匹配直到出现 op_exec_log 的右引号后再结束
    block_pat = re.compile(
        r"Simulator received StepSync command:(.*?op_exec_log:\s*Some\(\".*?\"\))",
        re.DOTALL,
    )
    outputs_pat = re.compile(r"VllmRequestStatus\s*\{")
    after_exec_pat = re.compile(r"after_execute=([0-9.]+)")

    mapping = defaultdict(list)
    x_vals, y_vals = [], []

    for m in block_pat.finditer(content):
        block = m.group(1)
        num_outputs = len(outputs_pat.findall(block))
        ae = after_exec_pat.search(block)
        if ae:
            val = float(ae.group(1))
            mapping[num_outputs].append(val)
            x_vals.append(num_outputs)
            y_vals.append(val)

    return mapping, x_vals, y_vals


def parse_log(log_path):
    """
    先按行解析；若无结果，再用跨行回退解析。
    """
    mapping, x_vals, y_vals = parse_by_line(log_path)
    if not x_vals:
        mapping, x_vals, y_vals = parse_fallback_block(log_path)
    return mapping, x_vals, y_vals


def plot_scatter(x_vals, y_vals, output_file="after_execute_scatter.png"):
    """
    绘制“输出数量 vs after_execute”的散点图
    """
    if not x_vals:
        return False

    plt.figure(figsize=(8, 6))
    # 不指定颜色风格，避免环境差异；edgecolors 提升可读性
    plt.scatter(x_vals, y_vals, alpha=0.7, edgecolors="black")
    plt.title("Outputs Count vs after_execute")
    plt.xlabel("Number of VllmRequestStatus (outputs)")
    plt.ylabel("after_execute (ms)")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(output_file, dpi=150)
    # 在命令行脚本中展示图窗可选；默认保存即可
    # plt.show()
    return True


def main():
    parser = argparse.ArgumentParser(
        description="从 router_v2 日志中提取 outputs 数量与 after_execute 的映射，并绘制散点图"
    )
    parser.add_argument("--log", required=True, help="日志文件路径")
    parser.add_argument("--out", default="after_execute_scatter.png", help="散点图输出文件名")
    args = parser.parse_args()

    mapping, x_vals, y_vals = parse_log(args.log)

    print("=== 输出数量 -> after_execute(ms) ===")
    if not x_vals:
        print("❌ 未提取到任何 after_execute 数据，请检查日志格式。")
        return

    import csv

    with open("sampler.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["batch_size", "predictions"])

        for n, vals in sorted(mapping.items()):
            avg = sum(vals) / len(vals)
            writer.writerow([n, avg])
            print(f"{n:4d} 个 outputs -> 平均 after_execute: {avg:.3f} ms (共 {len(vals)} 条)")

    if plot_scatter(x_vals, y_vals, args.out):
        print(f"\n✅ 已生成散点图: {args.out}（点数: {len(x_vals)}）")


if __name__ == "__main__":
    main()
