#!/bin/bash

# 配置参数
sc="5.0"

# moonckae
# /mnt/debugger/hjb/node1/lmmetric-logs/5.0_batch1024_u0.9_flashinfer_1126_predictions_lasttry_r1/20251126205103_join-shortest-q-ttft

tag="flashinfer_1126_predictions_lasttry_r1"
policies=("join-shortest-q-ttft")  # ← 在这里添加你的策略列表

base_src="/mnt/debugger/hjb/node1/lmmetric-logs/${sc}_batch1024_u0.9_${tag}"

# 遍历每个策略
for policy in "${policies[@]}"; do
    echo "Processing policy: ${policy}"

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

    dest_tar="/mnt/debugger/hjb/node1/xmetric-paper/data/${dirname}.tgz"
    mkdir -p "$(dirname "$dest_tar")"

    echo "  → Latest dir: $latest_dir"
    echo "  → Packing to: $dest_tar"

    # 创建临时根目录，用于构建 dirname/ 文件结构
    tmp_root=$(mktemp -d)
    target_dir="$tmp_root/$dirname"
    mkdir -p "$target_dir"

    # 拷贝需要的文件（仅当存在时）
    [[ -f "$latest_dir/client.jsonl"    ]] && cp "$latest_dir/client.jsonl"    "$target_dir/"
    [[ -f "$latest_dir/statistic.log"   ]] && cp "$latest_dir/statistic.log"   "$target_dir/"
    #[[ -f "$latest_dir/router_v2.log"   ]] && cp "$latest_dir/router_v2.log"   "$target_dir/"

    # 检查是否有文件被复制
    if [[ -z "$(ls -A "$target_dir")" ]]; then
        echo "  → No files to pack for ${policy}, skipping."
        rm -rf "$tmp_root"
        continue
    fi

    # 打包：进入 tmp_root，打包 dirname 目录
    tar -czf "$dest_tar" -C "$tmp_root" "$dirname"

    # 清理
    rm -rf "$tmp_root"

    echo "  → Done: $dest_tar"
done

echo "✅ All policies processed."