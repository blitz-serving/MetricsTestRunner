import re
import argparse
import numpy as np
import matplotlib.pyplot as plt

def main():
    # === 命令行参数 ===
    parser = argparse.ArgumentParser(description="Plot signed error distribution from predictor logs.")
    parser.add_argument("--log", required=True, help="Path to the predictor log file.")
    parser.add_argument("--bucket", type=float, default=0.05, help="Bucket size for error grouping (default: 0.05).")
    args = parser.parse_args()

    log_file = args.log
    bucket_size = args.bucket

    # === 1. 提取 error 值（带符号）===
    pattern = re.compile(r"error\s*=\s*(-?[0-9.]+)")
    errors = []

    with open(log_file, "r", encoding="utf-8") as f:
        for line in f:
            match = pattern.search(line)
            if match:
                try:
                    val = float(match.group(1))
                    errors.append(val)  # ✅ 保留符号
                except ValueError:
                    pass

    if not errors:
        print("❌ 未找到任何 error 值，请检查日志文件格式。")
        return

    # === 2. 分桶 ===
    max_abs_err = max(abs(e) for e in errors)
    # 从 -max 到 +max，中心对齐 0
    num_buckets = int(np.ceil(max_abs_err / bucket_size))
    bins = np.arange(-num_buckets * bucket_size, (num_buckets + 1) * bucket_size, bucket_size)

    # === 3. 统计分布 ===
    hist, _ = np.histogram(errors, bins=bins)
    ratios = hist / len(errors)

    # === 4. 绘制对称图表 ===
    plt.figure(figsize=(10, 6))
    plt.bar(bins[:-1], ratios, width=bucket_size * 0.9, align='edge', edgecolor='black')

    plt.title("LinearRegressionPredictor | Signed Error Distribution")
    plt.xlabel(f"Error bucket start (size={bucket_size})")
    plt.ylabel("Proportion of records")

    # 视觉优化
    plt.axvline(0, color='red', linestyle='--', linewidth=1)  # 中心线
    plt.grid(axis='y', linestyle='--', alpha=0.6)
    plt.xlim(-max_abs_err, max_abs_err)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
