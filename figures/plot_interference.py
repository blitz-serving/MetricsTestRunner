#!/usr/bin/env python3
import argparse
import os
import re
import json
from collections import defaultdict
import matplotlib.pyplot as plt


def parse_log_line(line):
    pattern = (
        r'(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d+Z)\s+INFO.+?'
        r'vLLM#(\d+)\s+prefill tokens\s+(\d+)\s+budget\s+\d+\s+'
        r'request states:\s+({.*})'
    )
    match = re.search(pattern, line)
    if not match:
        return None

    timestamp_str, instance_id, prefill_tokens_str, states_str = match.groups()
    instance_id = int(instance_id)
    prefill_tokens = int(prefill_tokens_str)

    try:
        states = json.loads(states_str.replace("'", '"'))
        states = {str(k): str(v).upper() for k, v in states.items()}
    except json.JSONDecodeError:
        return None

    return {
        'timestamp': timestamp_str,
        'instance': instance_id,
        'prefill_tokens': prefill_tokens,
        'states': states,
        'has_prefill': prefill_tokens > 0
    }


def smooth_data(data, window):
    if window <= 1:
        return data
    smoothed = []
    half = window // 2
    n = len(data)
    for i in range(n):
        start = max(0, i - half)
        end = min(n, i + half + 1)
        avg = sum(data[start:end]) / (end - start)
        smoothed.append(avg)
    return smoothed


def main():
    parser = argparse.ArgumentParser(description="Analyze vLLM interference from router_v2.log")
    parser.add_argument('dir', type=str, help='Directory containing router_v2.log')
    parser.add_argument('--smooth-window', type=int, default=1, help='Smoothing window size (default: 1)')
    parser.add_argument(
        '--instances',
        type=int,
        nargs='+',
        default=[0, 7, 8, 15],
        help='List of vLLM instance IDs to analyze (default: 0 7 8 15)'
    )
    args = parser.parse_args()

    log_path = os.path.join(args.dir, 'router_v2.log')
    if not os.path.isfile(log_path):
        raise FileNotFoundError(f"Log file not found: {log_path}")

    # 用整型计数器存储统计值
    per_request = defaultdict(lambda: {'total': 0, 'interfered': 0})
    per_instance = defaultdict(lambda: defaultdict(lambda: {'total': 0, 'interfered': 0}))
    
    # 记录每个请求首次出现的时间（用于排序）
    first_appearance = {}

    all_entries = []
    with open(log_path, 'r', encoding='utf-8') as f:
        for line in f:
            parsed = parse_log_line(line)
            if parsed is not None:
                all_entries.append(parsed)

    all_entries.sort(key=lambda x: x['timestamp'])

    for entry in all_entries:
        instance = entry['instance']
        if instance not in args.instances:
            continue
        has_prefill = entry['has_prefill']
        for req_id, state in entry['states'].items():
            if state == 'DECODE':
                # 记录首次出现时间
                if req_id not in first_appearance:
                    first_appearance[req_id] = entry['timestamp']
                # 更新总计数
                per_request[req_id]['total'] += 1
                per_instance[instance][req_id]['total'] += 1
                if has_prefill:
                    per_request[req_id]['interfered'] += 1
                    per_instance[instance][req_id]['interfered'] += 1

    if not per_request:
        print("No DECODE events found.")
        return

    # 按首次出现时间排序请求
    sorted_reqs = sorted(per_request.keys(), key=lambda r: first_appearance[r])
    
    # 提取总计数值序列
    y_total = [per_request[req]['total'] for req in sorted_reqs]
    y_interfered = [per_request[req]['interfered'] for req in sorted_reqs]
    x = list(range(1, len(sorted_reqs) + 1))

    # 平滑处理
    smooth_win = max(1, args.smooth_window)
    y_total_s = smooth_data(y_total, smooth_win)
    y_interfered_s = smooth_data(y_interfered, smooth_win)

    # 绘制总体图
    plt.figure(figsize=(12, 6))
    plt.plot(x, y_interfered_s, label='Interfered Decode Count', marker='o')
    plt.plot(x, y_total_s, label='Total Decode Count', marker='x')
    plt.xlabel('Request (ordered by first appearance)')
    plt.ylabel('Count (smoothed)')
    plt.title('vLLM Request Decode Interference (Overall)')
    plt.legend()
    plt.grid(True)
    overall_png = os.path.join(args.dir, 'decode_interference_overall.png')
    plt.savefig(overall_png)
    plt.close()
    print(f"Saved overall plot: {overall_png}")

    # 总体干扰比例
    total_decode_all = sum(y_total)
    total_interfered_all = sum(y_interfered)
    ratio_all = total_interfered_all / total_decode_all if total_decode_all > 0 else 0
    print(f"\n📊 Overall interference ratio: {total_interfered_all} / {total_decode_all} = {ratio_all:.4f} ({ratio_all*100:.2f}%)")

    # 每个实例的干扰比例和绘图
    print("\n📉 Per-instance interference ratios:")
    for instance in sorted(args.instances):
        inst_reqs = per_instance[instance]
        if not inst_reqs:
            print(f"  Instance {instance}: No DECODE events")
            continue

        # 获取该实例的请求首次出现时间（复用全局的first_appearance）
        inst_sorted_reqs = sorted(inst_reqs.keys(), key=lambda r: first_appearance.get(r, ''))
        y_t = [inst_reqs[req]['total'] for req in inst_sorted_reqs]
        y_i = [inst_reqs[req]['interfered'] for req in inst_sorted_reqs]
        
        inst_total = sum(y_t)
        inst_interfered = sum(y_i)
        ratio_inst = inst_interfered / inst_total if inst_total > 0 else 0
        print(f"  Instance {instance}: {inst_interfered} / {inst_total} = {ratio_inst:.4f} ({ratio_inst*100:.2f}%)")

        # 绘图
        x_inst = list(range(1, len(inst_sorted_reqs) + 1))
        y_t_s = smooth_data(y_t, smooth_win)
        y_i_s = smooth_data(y_i, smooth_win)
        plt.figure(figsize=(12, 6))
        plt.plot(x_inst, y_i_s, label='Interfered Decode Count', marker='o')
        plt.plot(x_inst, y_t_s, label='Total Decode Count', marker='x')
        plt.xlabel('Request (ordered by first appearance)')
        plt.ylabel('Count (smoothed)')
        plt.title(f'vLLM Request Decode Interference (Instance {instance})')
        plt.legend()
        plt.grid(True)
        inst_png = os.path.join(args.dir, f'decode_interference_instance_{instance}.png')
        plt.savefig(inst_png)
        plt.close()
        print(f"  → Saved plot: {inst_png}")

    print("\n✅ Analysis complete.")


if __name__ == '__main__':
    main()