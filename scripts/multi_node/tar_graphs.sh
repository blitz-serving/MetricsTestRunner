#!/bin/bash

# 定义参数
qps_list=("5.0")
bs_list=("1024")
tag="dynamo_bailian_round2"

# 遍历所有组合
for qps in "${qps_list[@]}"; do
  for bs in "${bs_list[@]}"; do
    dirname="${qps}_batch${bs}_u0.9_${tag}"
    full_path="/mnt/debugger/hjb/node1/lmmetric-logs/${dirname}"

    if [ ! -d "$full_path" ]; then
      echo "Warning: Directory $full_path does not exist."
      continue
    fi

    tar_file="${full_path}_filtered.tar"
    echo "Creating tarball: $tar_file"

    # 使用 tar 的 -r（追加）方式效率低，改用 find + tar -c -T /dev/stdin 方式更可靠
    # 但更简洁的方式是：在内存中构建文件列表，然后用 tar -c -T

    # 创建一个临时文件存放要打包的文件列表（带相对路径）
    list_file=$(mktemp)

    # 遍历所有一级子目录
    while IFS= read -r -d '' sub_dir; do
      sub_name=$(basename "$sub_dir")
      # 查找该子目录下的 *.png 和 client.jsonl，记录相对于 $full_path 的路径
      find "$sub_dir" -maxdepth 1 -name "*.png" -printf "$sub_name/%P\n" >> "$list_file"
      if [ -f "$sub_dir/client.jsonl" ]; then
        echo "$sub_name/client.jsonl" >> "$list_file"
      fi
      if [ -f "$sub_dir/statistic.log" ]; then
        echo "$sub_name/statistic.log" >> "$list_file"
      fi
      if [ -f "$sub_dir/top3.log" ]; then
        echo "$sub_name/top3.log" >> "$list_file"
      fi
    done < <(find "$full_path" -mindepth 1 -maxdepth 1 -type d -print0)

    # 如果列表为空，跳过打包
    if [ ! -s "$list_file" ]; then
      echo "No matching files found in $full_path. Skipping tar creation."
      rm -f "$list_file"
      continue
    fi

    # 在 $full_path 目录下，根据相对路径打包
    tar -cf "$tar_file" -C "$full_path" -T "$list_file"

    echo "Done: $tar_file created with $(wc -l < "$list_file") entries."

    # 清理
    rm -f "$list_file"
  done
done