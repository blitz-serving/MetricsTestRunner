import pandas as pd
import argparse

def main():
    parser = argparse.ArgumentParser(description="Filter CSV to keep only attention decode-related fields.")
    parser.add_argument("--input", required=True, help="Path to the input CSV file.")
    parser.add_argument("--output", required=True, help="Path to save filtered CSV.")
    args = parser.parse_args()

    # === 1. 读取 CSV ===
    df = pd.read_csv(args.input)

    # === 2. 要保留的列 ===
    keep_cols = [
        # "time_stats.attn_decode.mean",
        # "batch_size",
        # "kv_cache_size",
        # "pattern",
        "time_stats.attn_post_proj.mean",
        # "time_stats.attn_post_proj.mean",
        "num_tokens",
    ] 
    df = df[[c for c in keep_cols if c in df.columns]]

    # === 4️⃣ 按 batch_size 和 kv_cache_size 分组排序 ===
    # df_sorted = df.sort_values(by=["batch_size", "kv_cache_size"]).reset_index(drop=True)

    # === 5️⃣ 可选：把 (batch_size, kv_cache_size) 组合视作二维索引 ===
    # df_sorted["group_key"] = list(zip(df_sorted["batch_size"], df_sorted["kv_cache_size"]))

    # === 6️⃣ 导出 ===
    # df_sorted.to_csv(args.output, index=False)
    df.to_csv(args.output, index=False)
    print(f"✅ Grouped CSV saved to: {args.output}")
    # print(f"✅ Unique (batch_size, kv_cache_size) groups: {df_sorted['group_key'].nunique()}")

if __name__ == "__main__":
    main()
