#!/bin/bash

# 定义参数
qps_list=("4.0" "5.0" "5.5" "6.0")
bs_list=("1024")
tag="n3"

# 遍历所有组合
for qps in "${qps_list[@]}"; do
  for bs in "${bs_list[@]}"; do
    dirname="${qps}_batch${bs}_u0.9_${tag}"
    full_path="/mnt/debugger/hjb/node1/lmmetric-logs/${dirname}"
    
    # 检查目录是否存在
    if [ -d "$full_path" ]; then
      # 遍历该目录下的所有子目录
      for dir in "$full_path"/*/; do
        # 确保是目录（避免空展开）
        if [ -d "$dir" ]; then
          # 去掉末尾的斜杠（可选，使路径更干净）
          dir="${dir%/}"
          echo "Analyzing: $dir"
          python ../../figures/analyze_statistics.py "$dir"

          python ../../figures/analyze_statistics_smooth.py "$dir" --smooth-window 10
        fi
      done
    else
      echo "Warning: Directory $full_path does not exist."
    fi
  done
done