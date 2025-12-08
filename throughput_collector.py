import re
import sys
import subprocess
from pathlib import Path
from datetime import datetime
from statistics import mean
from collections import defaultdict

# 匹配精确到秒的时间戳
timestamp_pattern = re.compile(r"^(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2})")

# 新日志格式：INFO 12-08 05:52:55 ...
timestamp_pattern2 = re.compile(r"^\S+\s+(\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2})")


def filter_log(input_log: Path, output_log: Path):
    """ 用 grep 'Scheduling' 过滤 router_v2.log → filter.log """
    print(f"Filtering Scheduling lines into {output_log} ...")
    subprocess.run(
        ["grep", "Scheduling", str(input_log)],
        stdout=open(output_log, "w"),
        stderr=subprocess.DEVNULL,
        text=True
    )
    print("Filtering done.\n")


def parse_tps(logfile: Path):
    per_second_counts = defaultdict(int)
    base_time = None

    with open(logfile, "r") as f:
        for line in f:
            # 尝试匹配旧格式：2024-12-08T05:52:55
            m = timestamp_pattern.match(line)
            if m:
                ts_str = m.group(1)
                ts = datetime.strptime(ts_str, "%Y-%m-%dT%H:%M:%S")
            else:
                # 尝试匹配新格式：INFO 12-08 05:52:55
                m2 = timestamp_pattern2.match(line)
                if not m2:
                    continue
                ts_str = m2.group(1)

                # 缺少年份 → 自动补当前年份（也可改为固定年份）
                current_year = datetime.now().year
                ts = datetime.strptime(f"{current_year}-{ts_str}", "%Y-%m-%d %H:%M:%S")

            # 初始化基准时间
            if base_time is None:
                base_time = ts

            delta_s = int((ts - base_time).total_seconds())
            per_second_counts[delta_s] += 1
    
    return per_second_counts


def compute_window_average(per_second_counts, start_s, end_s):
    """ 统计 start_s 到 end_s 的平均 TPS """
    values = [per_second_counts.get(s, 0) for s in range(start_s, end_s+1)]
    if len(values) == 0:
        return 0
    return mean(values)


def main():
    if len(sys.argv) != 2:
        print("Usage: python throughput_collector.py <directory>")
        sys.exit(1)

    directory = Path(sys.argv[1]).expanduser().resolve()
    router_log = directory / "router_v2.log"
    filtered_log = directory / "filter.log"

    if not router_log.exists():
        print(f"Error: {router_log} not found")
        sys.exit(1)

    # Step 1: grep 过滤
    filter_log(router_log, filtered_log)

    # Step 2: 解析过滤后的 log
    per_second_counts = parse_tps(filtered_log)

    # Step 3: 计算窗口统计
    avg_65 = compute_window_average(per_second_counts, 15, 75)
    avg_125 = compute_window_average(per_second_counts, 15, 135)

    print("=== Throughput Statistics ===")
    print(f"Average TPS (15s ~ 75s):  {avg_65:.2f}")
    print(f"Average TPS (15s ~ 135s): {avg_125:.2f}\n")
    # 可选：打印每秒 TPS
    print("Per-second TPS:")
    for sec in sorted(per_second_counts.keys()):
        print(f"{sec:4d}s: {per_second_counts[sec]}")



if __name__ == "__main__":
    main()
