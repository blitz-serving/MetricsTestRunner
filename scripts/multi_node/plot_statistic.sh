#!/bin/bash

# 定义参数
qps_list=("5.0")
bs_list=("1024")
tag="lwl_pdinterference"

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
          # python ../../figures/analyze_statistics.py "$dir"

          # python ../../figures/analyze_statistics_smooth.py "$dir" --smooth-window 5

          # python ../../figures/analyze_statistics_smooth.py "$dir" --smooth-window 5 --instances 2 3 10 11 12

          # python ../../figures/analyze_load_with_time.py "$dir" --smooth-window 5 --instances 0 7 8 15

          # python ../../figures/analyze_load_with_time.py "$dir" --smooth-window 5 --instances 3 11

          # python ../../figures/analyze_load_with_time.py "$dir" --smooth-window 5 --instances 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15
          python ../../figures/plot_interference.py "$dir" --smooth-window 15
        fi
      done
    else
      echo "Warning: Directory $full_path does not exist."
    fi
  done
done