import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import argparse
import numpy as np

def main():
    parser = argparse.ArgumentParser(description="Visualize mean time error between patterns as readable heatmaps.")
    parser.add_argument("--csv", required=True, help="Path to CSV file.")
    parser.add_argument("--save", default=None, help="Optional path to save figure.")
    args = parser.parse_args()

    # === 1. 读取数据 ===
    df = pd.read_csv(args.csv)

    # === 2. 对比组合 ===
    pairs = [
        ("uniform", "avg_linear"),
        ("uniform", "extreme"),
    ]

    # === 3. 计算误差 ===
    results = []
    for gkey, group in df.groupby("group_key"):
        pattern_map = group.set_index("pattern")["time_stats.attn_decode.mean"].to_dict()
        bs = group["batch_size"].iloc[0]
        kv = group["kv_cache_size"].iloc[0]
        for p1, p2 in pairs:
            if p1 in pattern_map and p2 in pattern_map:
                results.append({
                    "batch_size": bs,
                    "kv_cache_size": kv,
                    "pattern_pair": f"{p1} vs {p2}",
                    "error": pattern_map[p2] - pattern_map[p1]
                })

    result_df = pd.DataFrame(results)
    if result_df.empty:
        print("❌ 未找到可比较的pattern对。")
        return

    sns.set_theme(style="whitegrid", font_scale=1.1)
    fig, axes = plt.subplots(1, len(pairs), figsize=(7*len(pairs), 5))

    if len(pairs) == 1:
        axes = [axes]

    for ax, (p1, p2) in zip(axes, pairs):
        sub = result_df[result_df["pattern_pair"] == f"{p1} vs {p2}"]
        pivot = sub.pivot(index="kv_cache_size", columns="batch_size", values="error")

        # 对数据进行插值，确保画面更平滑
        pivot_interp = pivot.sort_index().interpolate(axis=0).interpolate(axis=1)

        sns.heatmap(
            pivot_interp,
            cmap="coolwarm",
            center=0,
            cbar_kws={'label': 'Error (mean_time_diff)'},
            ax=ax,
            annot=False
        )

        ax.set_title(f"{p1} vs {p2} mean time error")
        ax.set_xlabel("Batch Size")
        ax.set_ylabel("KV Cache Size")

        # 减少坐标标签数量
        ax.set_xticks(ax.get_xticks()[::max(1, len(ax.get_xticks()) // 10)])
        ax.set_yticks(ax.get_yticks()[::max(1, len(ax.get_yticks()) // 10)])

    plt.tight_layout()

    if args.save:
        plt.savefig(args.save, dpi=300)
        print(f"✅ 图像已保存至: {args.save}")
    else:
        plt.show()

if __name__ == "__main__":
    main()
