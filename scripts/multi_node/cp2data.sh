#!/bin/bash

# 配置参数
sc="5.6"
bs="4096"
# moonckae
# /mnt/debugger/hjb/node1/lmmetric-logs/5.0_batch1024_u0.9_flashinfer_1126_predictions_lasttry_r1/20251126205103_join-shortest-q-ttft
# 
# /mnt/debugger/hjb/node1/lmmetric-logs/3.0_batch4096_u0.9_flashinfer_1201_coder_qwen7b_r1
# /mnt/debugger/hjb/node1/lmmetric-logs/1.5_batch4096_u0.9_flashinfer_1202_mooncake_conv_qwen7b_r1
# /mnt/debugger/hjb/node1/lmmetric-logs/1.0_batch4096_u0.9_flashinfer_1202_mooncake_tool_qwen7b_r1
# /mnt/debugger/hjb/node3/lmmetric-logs/5.0_batch4096_u0.9_flashinfer_1201_toB_qwen30b_r1
# /mnt/debugger/hjb/node3/lmmetric-logs/2.0_batch4096_u0.9_flashinfer_1201_coder_qwen30b_r1
# /mnt/debugger/hjb/node1/lmmetric-logs/1.5_batch4096_u0.9_flashinfer_1203_mooncake_conv_searchbailianpara_qwen7b_r1/20251204000418_bailian-impl-05-deterministic
# /mnt/debugger/hjb/node1/lmmetric-logs/5.0_batch1024_u0.9_1119_lwl_gated_param_r1
# /mnt/debugger/hjb/node3/lmmetric-logs/1.5_batch4096_u0.9_flashinfer_1204_mooncake_conv_final_fix_qwen30b_r1
# /mnt/debugger/hjb/node1/lmmetric-logs/2.2_batch4096_u0.9_flashinfer_1205_coder_scaling_qwen30b_r1
# /mnt/debugger/hjb/node1/lmmetric-logs/5.0_batch1024_u0.9_flashinfer_redoall_r
# /mnt/debugger/hjb/node1/lmmetric-logs/5.0_batch1024_u0.9_flashinfer_redoall_r
# /mnt/debugger/hjb/node3/lmmetric-logs/5.0_batch1024_u0.9_flashinfer_1205_4.2_addtests_r1
# /mnt/debugger/hjb/node3/lmmetric-logs/5.5_batch4096_u0.9_flashinfer_1206_scaling_toB_r1
# /mnt/debugger/hjb/node1/lmmetric-logs/1.4_batch4096_u0.9_flashinfer_1206_scaling_mooncake_tool_r1
# /mnt/debugger/hjb/node1/lmmetric-logs/2.0_batch4096_u0.9_flashinfer_1205_try_scale_mooncake_tool_r1
# /mnt/debugger/hjb/node3/lmmetric-logs/2.4_batch4096_u0.9_flashinfer_1206_coder_scaling_fixmooncakecrash_r1/20251206203202_join-shortest-q-ttft
# /mnt/debugger/hjb/node3/lmmetric-logs/5.6_batch4096_u0.9_flashinfer_1206_redoSection4.2_r1
# /mnt/debugger/hjb/node1/lmmetric-logs/5.6_batch4096_u0.9_flashinfer_1205_toC_scaling_5_6_qwen30b_r1
# /mnt/debugger/hjb/node1/lmmetric-logs/1.6_batch4096_u0.9_flashinfer_1206_scaling_mooncake_conv_r1
# /mnt/debugger/hjb/node3/lmmetric-logs/7.2_batch4096_u0.9_flashinfer_1206_scaling_toC_r1
# /mnt/debugger/hjb/node1/lmmetric-logs/5.6_batch4096_u0.9_flashinfer_1205_toC_scaling_5_6_qwen30b_r1
# /mnt/debugger/hjb/node1/lmmetric-logs/1.2_batch4096_u0.9_flashinfer_1206_scaling_mooncake_tool_r1
# /mnt/debugger/hjb/node3/lmmetric-logs/1.2_batch4096_u0.9_flashinfer_1207_scaling_mooncake_tool_bailian05_r1/20251207173451_bailian-impl-05-deterministic
# /mnt/debugger/hjb/node1/lmmetric-logs/2.0_batch4096_u0.9_flashinfer_1206_scaling_mooncake_conv_r1
# /mnt/debugger/hjb/node1/lmmetric-logs/5.6_batch4096_u0.9_flashinfer_1208_toC_onlyttft_r1

suffix="sc${sc}-toC-cpsize${bs}-u0.9-qwen30b"
NODE1="1"
tag="flashinfer_1208_toC_onlyttft_r1"
# "join-shortest-q-weight"  "bailian-impl-06" 
# bailian-impl-05-deterministic"
# "dynamo-deterministic" "join-shortest-q-ttft" "least-wait-token-mul-bs" "join-shortest-q-weight" "bailian-impl-06" 
# "bailian-impl-06" "join-shortest-q-ttft" "dynamo-deterministic" "least-wait-token-mul-bs" "join-shortest-q-weight"
# Mooncake  bailian-impl-05-deterministic" "join-shortest-q-ttft" "dynamo-deterministic" "least-wait-token-mul-bs" "join-shortest-q-weight"
# 4.2 test1 "least-bs-random" "least-wait-token-random" "least-wait-token-bs" "bailian-impl-06" "join-shortest-q-tuple" "join-shortest-q-weight" "join-shortest-q-ttft"
# "bailian-impl-06" "join-shortest-q-ttft" "dynamo-deterministic" "least-wait-token-mul-bs" "join-shortest-q-weight"
policies=( "ttft-only" )  # ← 在这里添加你的策略列表

base_src="/mnt/debugger/hjb/node${NODE1}/lmmetric-logs/${sc}_batch${bs}_u0.9_${tag}"

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
    dirname="${dirname}_${suffix}"

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
    [[ -f "$latest_dir/router_v2.log"   ]] && cp "$latest_dir/router_v2.log"   "$target_dir/"

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