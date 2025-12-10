#!/bin/bash

# 配置参数
sc="1.5"
bs="4096"
tag="flashinfer_1203_mooncake_conv_searchbailianpara_qwen7b_r1"
show_tag="qwen7b_bailian_search_parameter_mooncake_conv"
NODE="1"
#policies=("bailian-impl-00" "bailian-impl-01" "bailian-impl-02" "bailian-impl-03" "bailian-impl-04" "bailian-impl-05" "bailian-impl-06" "bailian-impl-07" "bailian-impl-08" "bailian-impl-09" "bailian-impl-10")
policies=("bailian-impl-01-deterministic" "bailian-impl-03-deterministic" "bailian-impl-05-deterministic" "bailian-impl-07-deterministic" "bailian-impl-09-deterministic")

# /mnt/debugger/hjb/node1/lmmetric-logs/1.5_batch4096_u0.9_flashinfer_1203_mooncake_conv_searchbailianpara_qwen7b_r1
#/mnt/debugger/hjb/node1/lmmetric-logs/5.0_batch1024_u0.9_lwl_evo1_plus_bailiankv
# /mnt/debugger/hjb/node1/lmmetric-logs/5.0_batch1024_u0.9_ugly4
base_src="/mnt/debugger/hjb/node${NODE}/lmmetric-logs/${sc}_batch${bs}_u0.9_${tag}"
output_tar="/mnt/debugger/hjb/node1/xmetric-paper/data/${sc}_batch${bs}_u0.9_${show_tag}.tgz"

echo "🎯 Target output: $output_tar"
mkdir -p "$(dirname "$output_tar")"

# 创建临时根目录
tmp_root=$(mktemp -d)

# 👇 关键：在 tmp_root 下创建目标顶层目录
top_dir_name="${sc}_batch${bs}_u0.9_${show_tag}"
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

#/mnt/debugger/hjb/node1/lmmetric-logs/5.0_batch1024_u0.9_
# tag="lwl_evo1_plus_bailiankv"
# policies=("least-wait-token-random")
# base_src="/mnt/debugger/hjb/node${NODE}/lmmetric-logs/${sc}_batch1024_u0.9_${tag}"

# # 遍历每个策略
# for policy in "${policies[@]}"; do
#     echo "🔍 Processing policy: ${policy}"

#     dirs=("${base_src}"/*_"${policy}")
#     valid_dirs=()
#     for d in "${dirs[@]}"; do
#         if [[ -d "$d" ]]; then
#             valid_dirs+=("$d")
#         fi
#     done

#     if [[ ${#valid_dirs[@]} -eq 0 ]]; then
#         echo "  ⚠️  No directory found ending with '${policy}' in ${base_src}"
#         continue
#     fi

#     latest_dir=$(printf '%s\n' "${valid_dirs[@]}" | sort -r | head -n1)
#     dirname=$(basename "$latest_dir")
#     target_dir="$archive_root/$dirname"  # 👈 复制到 archive_root 下

#     echo "  → Latest dir: $latest_dir"
#     echo "  → Copying to: $target_dir"

#     mkdir -p "$target_dir"

#     [[ -f "$latest_dir/client.jsonl"  ]] && cp "$latest_dir/client.jsonl"  "$target_dir/"
#     [[ -f "$latest_dir/statistic.log" ]] && cp "$latest_dir/statistic.log" "$target_dir/"
#     #[[ -f "$latest_dir/router_v2.log" ]] && cp "$latest_dir/router_v2.log" "$target_dir/"

#     if [[ -n "$(ls -A "$target_dir")" ]]; then
#         any_found=true
#         echo "  → Files copied for ${policy}"
#     else
#         echo "  → No files to copy for ${policy}, skipping."
#         rmdir "$target_dir"
#     fi
# done

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