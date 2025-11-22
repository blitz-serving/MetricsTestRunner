#!/bin/bash

# 定义参数
qps_list=("5.0")
bs_list=("1024")
#tag="dynamo_bailian_round2"
#policy="join-shortest-q-tuple"  # 可选值: "all" 或 具体策略名，如 "join-shortest-q-tuple"
#policy="bailian-impl-06"
#policy="least-wait-token-random"
policy="least-wait-token-bs"

# 遍历所有组合
for r in 1; do
  #tag="flashinfer_predictions_r$r"
  #tag="dynamo_bailian_round2"
  tag="1118_lwl_bs_tuple_r1"
  for qps in "${qps_list[@]}"; do
    for bs in "${bs_list[@]}"; do
      dirname="${qps}_batch${bs}_u0.9_${tag}"
      full_path="/mnt/debugger/hjb/node1/lmmetric-logs/${dirname}"
      #full_path="/tmp/node1/lmmetric-logs/${dirname}"
      # 检查目录是否存在
      if [ -d "$full_path" ]; then
        # 遍历该目录下的所有子目录
        for dir in "$full_path"/*/; do
          if [ -d "$dir" ]; then
            dir="${dir%/}"  # 去掉末尾斜杠
            dir_name=$(basename "$dir")

            # 判断是否应处理该目录
            should_process=false
            if [[ "$policy" == "all" ]]; then
              should_process=true
            elif [[ "$dir_name" == *_"$policy" ]]; then
              should_process=true
            fi

            if [ "$should_process" = true ]; then
              echo "Analyzing: $dir"
              # --start-time 1140 --end-time 1200 

              # python ../../figures/analyze_statistics.py "$dir"

              python ../../figures/analyze_statistics_smooth.py "$dir" --smooth-window 5 --instances 8

              # python ../../figures/analyze_statistics_smooth.py "$dir" --smooth-window 5 --instances 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15

              # python ../../figures/analyze_load_with_time.py "$dir" --smooth-window 5 --instances 0 7 8 15

              # python ../../figures/analyze_load_with_time.py "$dir" --smooth-window 5 --instances 3 11
                
              #python ../../figures/analyze_actual_prefill_tokens.py "$dir" --smooth-window 15 --instances 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 --actual-prefill-token-ylim 7500 --start-time 600 --end-time 720

              # python ../../figures/analyze_actual_prefill_tokens.py "$dir" --smooth-window 15 --instances 0  8 9 10 15 --actual-prefill-token-ylim 7500 --start-time 600 --end-time 720
              #python ../../figures/analyze_load_with_time.py "$dir" --smooth-window 5 --instances 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 --prefill-request-ylim 16 --prefill-throughput-ylim 20000 --start-time 1140 --end-time 1200
              # 创建目标目录（保留原目录名
              #python ../../figures/analyze_overall_hit_rate.py "$dir" --smooth-window 10

              #python ../../figures/analyze_overall_hit_rate.py "$dir" --smooth-window 15
              #python ../../figures/plot_interference_merged_by_time.py "$dir" --start-time 0 --end-time 1400 --instances 3 15 --smooth-window 10
              #python ../../figures/cp_cycle.py "$dir" --start-time 600 --end-time 10000 --instances 0 3 8 15 --smooth-window 5
              #python ../../figures/analyze_tpot_cluster.py "$dir" --instances 0 3 7 8 12 15
              # Plotting waiting-p-tks to see some zeros
              # python ../../figures/waiting-p-tks-timeline.py "$dir"
              # Plotting bailian hit ratio when scheduler query, show it is zero manytimes
              #python ../../figures/hit_ratio_timeline_bailian.py "$dir" --replicas 6 12 --smooth-window 50
              dest_dir="/root/figs/$dir_name"
              mkdir -p "$dest_dir"

      # 复制该目录下所有 .png 文件到目标目录
              #find "$dir" -maxdepth 1 -name "*.png" -exec cp {} "$dest_dir/" \;
              ## 打包该目录下所有 .png 文件为一个 tar 包，存放到 dest_dir/ 下
              png_files=("$dir"/*.png)
              # 进入目标目录，再执行 tar
              (cd "$dir" && tar -cf "$dest_dir/${dir_name}.tar" *.png)
              #if [ -e "${png_files[0]}" ]; then
  # 如果存在至少一个 .png 文件，则打包
              #  tar -cf "$dest_dir/${dir_name}.tar" -C "$dir" -- *.png
              #else
              #    echo "No .png files in $dir, skipping tar."
              #fi

            else
              echo "Skipped (policy mismatch): $dir"
            fi
          fi
        done
      else
        echo "Warning: Directory $full_path does not exist."
      fi
    done
  done
done

 

