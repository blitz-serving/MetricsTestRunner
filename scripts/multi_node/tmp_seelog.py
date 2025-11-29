import json
import sys
import numpy as np
from collections import defaultdict

def main(input_file):
    success_data = {
        'queue_time': [],
        'avg_time_between_tokens': [],
        'first_token_time': [],
        'total_time': [],
        'input_length': [],
        'output_length': []
    }
    status_200_count = 0
    non_200_counts = defaultdict(int)  # 统计每种非200状态的出现次数

    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue

                status = record.get("status")
                if status == "200":
                    status_200_count += 1
                    for key in success_data:
                        val = record.get(key)
                        if val is not None and val != "nil":
                            try:
                                success_data[key].append(float(val))
                            except ValueError:
                                pass  # skip unparseable values
                else:
                    # 注意：status 可能是 None 或其他非字符串类型
                    status_key = str(status) if status is not None else "null"
                    non_200_counts[status_key] += 1

    except FileNotFoundError:
        print(f"Error: File '{input_file}' not found.", file=sys.stderr)
        sys.exit(1)

    # 输出 status 统计
    print(f"Status 200 count: {status_200_count}")
    print(f"Non-200 status count: {sum(non_200_counts.values())}")
    if non_200_counts:
        print("\nBreakdown of non-200 statuses:")
        for status, cnt in sorted(non_200_counts.items()):
            print(f"  {status}: {cnt}")
    print()

    # 统计并输出性能指标
    metrics = ['queue_time', 'avg_time_between_tokens', 'first_token_time', 'total_time', 'input_length', 'output_length']
    for metric in metrics:
        values = success_data[metric]
        if not values:
            print(f"No valid data for {metric}")
            continue

        arr = np.array(values)
        mean = np.mean(arr)
        p50 = np.percentile(arr, 50)
        p90 = np.percentile(arr, 90)
        p99 = np.percentile(arr, 99)
        mini = np.min(arr)
        maxx = np.max(arr)

        print(f"{metric}:")
        print(f"  mean: {mean:.2f}")
        print(f"  p50:  {p50:.2f}")
        print(f"  p90:  {p90:.2f}")
        print(f"  p99:  {p99:.2f}")
        print(f"  min:  {mini:.2f}")
        print(f"  maxx:  {maxx:.2f}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python script.py <client.jsonl>", file=sys.stderr)
        sys.exit(1)
    main(sys.argv[1])