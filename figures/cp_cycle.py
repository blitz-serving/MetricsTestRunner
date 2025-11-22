#!/usr/bin/env python3
"""
Analyze vLLM decode interference ratio per request using route_time as s_time.
Plots a single line: interference ratio per request over time.
"""

import argparse
import os
import re
import json
from collections import defaultdict
import matplotlib.pyplot as plt
from datetime import datetime, timezone

LOG_FILENAME = "router_v2.log"

def parse_iso_zulu(ts_str):
    if ts_str.endswith('Z'):
        ts_str = ts_str[:-1] + '+00:00'
    return datetime.fromisoformat(ts_str).astimezone(timezone.utc)

def parse_prefill_log_line(line):
    pattern = (
        r'(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d+Z)\s+INFO.+?'
        r'vLLM#(\d+)\s+prefill tokens\s+(\d+)\s+budget\s+\d+\s+'
        r'request states:\s+({.*})'
    )
    match = re.search(pattern, line)
    if not match:
        return None

    ts_str, instance_id, prefill_tokens_str, states_str = match.groups()
    try:
        states = json.loads(states_str.replace("'", '"'))
        states = {str(k): str(v).upper() for k, v in states.items()}
    except json.JSONDecodeError:
        return None

    return {
        'timestamp': parse_iso_zulu(ts_str),
        'instance': int(instance_id),
        'prefill_tokens': int(prefill_tokens_str),
        'states': states
    }

def parse_route_log_line(line):
    route_pattern = re.compile(
        r'(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d+Z).*?'
        r'Request_(\d+)\s+queued\s+\d+us,\s+with\s+input\s+length\s+(\d+)\s+output\s+length\s+(\d+),\s+added\s+to\s+vLLM#(\d+)'
    )
    match = route_pattern.search(line)
    if not match:
        return None
    ts_str, req_id, input_len, output_len, instance_id = match.groups()
    return {
        'timestamp': parse_iso_zulu(ts_str),
        'request_id': int(req_id),
        'input_length': int(input_len),
        'output_length': int(output_len),
        'instance_id': int(instance_id)
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
    parser = argparse.ArgumentParser(description="Plot per-request decode interference ratio over s_time")
    parser.add_argument('dir', type=str, help='Directory containing router_v2.log')
    parser.add_argument('--smooth-window', type=int, default=1, help='Smoothing window size (default: 1)')
    parser.add_argument('--instances', type=int, nargs='+', default=[0, 7, 8, 15], help='vLLM instance IDs to analyze')
    parser.add_argument('--start-time', type=float, default=None, help='Start time in seconds (relative to first route)')
    parser.add_argument('--end-time', type=float, default=None, help='End time in seconds (relative to first route)')
    args = parser.parse_args()

    log_path = os.path.join(args.dir, LOG_FILENAME)
    if not os.path.isfile(log_path):
        raise FileNotFoundError(f"Log file not found: {log_path}")

    # Step 1: Parse route lines to get s_time
    route_time = {}
    route_instance = {}
    all_route_timestamps = []

    with open(log_path, 'r', encoding='utf-8') as f:
        for line in f:
            route = parse_route_log_line(line)
            if route:
                req_id = route['request_id']
                route_time[req_id] = route['timestamp']
                route_instance[req_id] = route['instance_id']
                all_route_timestamps.append(route['timestamp'])

    if not route_time:
        raise ValueError("No route log entries found – cannot determine s_time.")

    t0 = min(all_route_timestamps)
    route_time_rel = {req: (ts - t0).total_seconds() for req, ts in route_time.items()}

    # Step 2: Parse and filter prefill/decode state lines
    prefill_entries = []
    with open(log_path, 'r', encoding='utf-8') as f:
        for line in f:
            entry = parse_prefill_log_line(line)
            if entry:
                rel_time = (entry['timestamp'] - t0).total_seconds()
                if args.start_time is not None and rel_time < args.start_time:
                    continue
                if args.end_time is not None and rel_time >= args.end_time:
                    continue
                prefill_entries.append(entry)

    if not prefill_entries:
        print("No relevant log entries in time window.")
        return

    # Step 3: Count total and interfered decode steps per request
    per_request = defaultdict(lambda: {'total': 0, 'interfered': 0})
    per_instance = defaultdict(lambda: defaultdict(lambda: {'total': 0, 'interfered': 0}))

    for entry in prefill_entries:
        instance = entry['instance']
        if instance not in args.instances:
            continue
        has_prefill = entry['prefill_tokens'] > 0
        for req_id_str, state in entry['states'].items():
            try:
                req_id = int(req_id_str)
            except ValueError:
                continue
            if req_id not in route_time:
                continue
            if route_instance[req_id] != instance:
                continue
            if state == 'DECODE':
                per_request[req_id]['total'] += 1
                per_instance[instance][req_id]['total'] += 1
                if has_prefill:
                    per_request[req_id]['interfered'] += 1
                    per_instance[instance][req_id]['interfered'] += 1

    if not per_request:
        print("No DECODE events found for routed requests.")
        return

    # Step 4: Compute per-request interference ratio
    sorted_reqs = sorted(per_request.keys(), key=lambda r: route_time_rel[r])
    x_rel = [route_time_rel[r] for r in sorted_reqs]
    y_ratio = []
    valid_reqs = []  # only requests with total > 0
    for req in sorted_reqs:
        total = per_request[req]['total']
        interfered = per_request[req]['interfered']
        if total > 0:
            ratio = interfered / total
            y_ratio.append(ratio)
            valid_reqs.append(req)
        else:
            # Should not happen, but skip
            pass

    if not y_ratio:
        print("No valid requests with decode steps.")
        return

    x_rel_valid = [route_time_rel[r] for r in valid_reqs]

    smooth_win = max(1, args.smooth_window)
    y_ratio_s = smooth_data(y_ratio, smooth_win)

    # Plot overall interference ratio
    plt.figure(figsize=(12, 6))
    plt.plot(x_rel_valid, y_ratio_s, label='Interference Ratio', color='tab:blue', marker='o', markersize=3, linewidth=2)
    plt.xlabel('Request s_time (seconds since first route)')
    plt.ylabel('Interference Ratio (interfered decode steps / total decode steps)')
    plt.title('vLLM Per-Request Decode Interference Ratio (Overall)')
    plt.ylim(-0.05, 1.05)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    overall_png = os.path.join(args.dir, 'interference_ratio_overall.png')
    plt.savefig(overall_png)
    plt.close()
    print(f"Saved overall interference ratio plot: {overall_png}")

    # Overall aggregate ratio (weighted by decode steps)
    total_decode_all = sum(per_request[r]['total'] for r in per_request)
    total_interfered_all = sum(per_request[r]['interfered'] for r in per_request)
    ratio_all = total_interfered_all / total_decode_all if total_decode_all > 0 else 0
    print(f"\n📊 Overall weighted interference ratio: {total_interfered_all} / {total_decode_all} = {ratio_all:.4f} ({ratio_all*100:.2f}%)")

    # Per-instance plots and stats
    print("\n📉 Per-instance interference ratios:")
    for instance in sorted(args.instances):
        inst_reqs = per_instance[instance]
        if not inst_reqs:
            print(f"  Instance {instance}: No DECODE events in time window")
            continue

        # Compute per-request ratio for this instance
        inst_req_list = []
        inst_ratios = []
        inst_x = []
        for req, counts in inst_reqs.items():
            if counts['total'] > 0 and req in route_time_rel:
                ratio = counts['interfered'] / counts['total']
                inst_ratios.append(ratio)
                inst_req_list.append(req)
                inst_x.append(route_time_rel[req])

        if not inst_ratios:
            print(f"  Instance {instance}: No valid decode steps")
            continue

        # Sort by s_time
        sorted_pairs = sorted(zip(inst_x, inst_ratios), key=lambda x: x[0])
        x_inst, y_inst = zip(*sorted_pairs)
        y_inst_s = smooth_data(list(y_inst), smooth_win)

        plt.figure(figsize=(12, 6))
        plt.plot(x_inst, y_inst_s, label='Interference Ratio', color='tab:orange', marker='o', markersize=3, linewidth=2)
        plt.xlabel('Request s_time (seconds since first route)')
        plt.ylabel('Interference Ratio')
        plt.title(f'vLLM Per-Request Decode Interference Ratio (Instance {instance})')
        plt.ylim(-0.05, 1.05)
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        inst_png = os.path.join(args.dir, f'interference_ratio_instance_{instance}.png')
        plt.savefig(inst_png)
        plt.close()
        print(f"  → Saved plot: {inst_png}")

        # Weighted per-instance ratio
        inst_total = sum(inst_reqs[r]['total'] for r in inst_reqs)
        inst_interfered = sum(inst_reqs[r]['interfered'] for r in inst_reqs)
        ratio_inst = inst_interfered / inst_total if inst_total > 0 else 0
        print(f"  Instance {instance}: weighted ratio = {ratio_inst:.4f} ({ratio_inst*100:.2f}%)")

    print("\n✅ Analysis complete.")

if __name__ == '__main__':
    main()
