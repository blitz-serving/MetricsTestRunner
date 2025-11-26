#!/bin/bash

# 配置参数
sc="5.0"
tag="bailian_hybrid_lwl"
policies=("least-wait-token-random" "bailian-impl-06" "bailian-impl-lwl-00" "bailian-impl-lwl-01" "bailian-impl-lwl-02" "bailian-impl-lwl-03" "bailian-impl-lwl-04" "bailian-impl-lwl-05" "bailian-impl-lwl-06" "bailian-impl-lwl-07" "bailian-impl-lwl-08" "bailian-impl-lwl-09" "bailian-impl-lwl-10")

base_src="/mnt/debugger/hjb/node1/lmmetric-logs/${sc}_batch1024_u0.9_${tag}"
output_tar="/mnt/debugger/hjb/node1/xmetric-paper/data/${sc}_batch1024_u0.9_${tag}.tgz"

echo "🎯 Target output: $output_tar"
mkdir -p "$(dirname "$output_tar")"

# 创建临时根目录
tmp_root=$(mktemp -d)

# 👇 关键：在 tmp_root 下创建目标顶层目录
top_dir_name="${sc}_batch1024_u0.9_${tag}"
archive_root="$tmp_root/$top_dir_name"
mkdir -p "$archive_root"

any_found=false

# 遍历每个策略
for policy in "${policies[@]}"; do
    echo "🔍 Processing policy: ${policy}"

    dirs=("${base_src}"/*_"${policy}")
    valid_dirs=()
    for d in "${dirs[@]}"; do
        if [[ -d "$d" ]]; then
            valid_dirs+=("$d")
        fi
    done

    if [[ ${#valid_dirs[@]} -eq 0 ]]; then
        echo "  ⚠️  No directory found ending with '${policy}' in ${base_src}"
        continue
    fi

    latest_dir=$(printf '%s\n' "${valid_dirs[@]}" | sort -r | head -n1)
    dirname=$(basename "$latest_dir")
    target_dir="$archive_root/$dirname"  # 👈 复制到 archive_root 下

    echo "  → Latest dir: $latest_dir"
    echo "  → Copying to: $target_dir"

    mkdir -p "$target_dir"

    [[ -f "$latest_dir/client.jsonl"  ]] && cp "$latest_dir/client.jsonl"  "$target_dir/"
    [[ -f "$latest_dir/statistic.log" ]] && cp "$latest_dir/statistic.log" "$target_dir/"
    [[ -f "$latest_dir/router_v2.log" ]] && cp "$latest_dir/router_v2.log" "$target_dir/"

    if [[ -n "$(ls -A "$target_dir")" ]]; then
        any_found=true
        echo "  → Files copied for ${policy}"
    else
        echo "  → No files to copy for ${policy}, skipping."
        rmdir "$target_dir"
    fi
done

# 打包
if [[ "$any_found" == true ]]; then
    echo "📦 Creating unified archive: $output_tar"
    # 👇 从 tmp_root 打包，这样顶层目录会包含在 tar 中
    #tar -czf "$output_tar" -C "$tmp_root" "$top_dir_name"
    tar --use-compress-program='pigz -9 -p32' -cf "$output_tar" -C "$tmp_root" "$top_dir_name"
    echo "✅ Successfully created: $output_tar"
else
    echo "❌ No valid data found for any policy. Archive not created."
fi

# 清理
rm -rf "$tmp_root"