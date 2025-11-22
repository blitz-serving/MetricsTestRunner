#!/usr/bin/env python3
"""
Plot per-request decode interference ratio for multiple instances on a single chart.
Each instance is a line. X-axis = request s_time (route_time, relative seconds).
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
    if window <= 1 or len(data) == 0:
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
    parser = argparse.ArgumentParser(description="Plot merged interference ratio for multiple instances")
    parser.add_argument('dir', type=str, help='Directory containing router_v2.log')
    parser.add_argument('--smooth-window', type=int, default=1, help='Smoothing window size (default: 1)')
    parser.add_argument('--instances', type=int, nargs='+', default=[0, 7, 8, 15], help='vLLM instance IDs to plot (merged)')
    parser.add_argument('--start-time', type=float, default=None, help='Start time in seconds (relative to first route)')
    parser.add_argument('--end-time', type=float, default=None, help='End time in seconds (relative to first route)')
    args = parser.parse_args()

    log_path = os.path.join(args.dir, LOG_FILENAME)
    if not os.path.isfile(log_path):
        raise FileNotFoundError(f"Log file not found: {log_path}")

    # Step 1: Parse route lines
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

    # Step 2: Parse and filter state lines
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

    # Step 3: Build per-instance per-request stats
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
                per_instance[instance][req_id]['total'] += 1
                if has_prefill:
                    per_instance[instance][req_id]['interfered'] += 1

    # Check if any data
    has_data = any(per_instance[inst] for inst in args.instances)
    if not has_data:
        print("No DECODE events found for the specified instances in the time window.")
        return

    # Step 4: Plot all instances on one chart
    plt.figure(figsize=(14, 7))

    plotted_any = False
    for instance in sorted(args.instances):
        reqs = per_instance[instance]
        if not reqs:
            print(f"  Instance {instance}: no data (skipped in plot)")
            continue

        # Build list of (s_time, ratio)
        points = []
        for req, counts in reqs.items():
            if counts['total'] > 0 and req in route_time_rel:
                ratio = counts['interfered'] / counts['total']
                points.append((route_time_rel[req], ratio))

        if not points:
            continue

        points.sort(key=lambda x: x[0])  # sort by s_time
        x_vals, y_vals = zip(*points)
        y_vals_smooth = smooth_data(list(y_vals), args.smooth_window)

        plt.plot(x_vals, y_vals_smooth,
                 label=f'Instance #{instance}',
                 marker='o',
                 markersize=2,
                 linewidth=2)
        plotted_any = True

    if not plotted_any:
        print("No valid data to plot.")
        return

    plt.xlabel('Request s_time (seconds since first route)', fontsize=12)
    plt.ylabel('Interference Ratio (interfered / total decode steps)', fontsize=12)
    plt.title('vLLM Per-Request Decode Interference Ratio (Merged Instances)', fontsize=14)
    plt.ylim(-0.05, 1.05)
    plt.grid(True, alpha=0.3)
    plt.legend(title='vLLM Instance', bbox_to_anchor=(1.02, 1), loc='upper left')
    plt.tight_layout()

    # Build output filename
    inst_str = '_'.join(map(str, sorted(args.instances)))
    output_path = os.path.join(args.dir, f'interference_ratio_merged_instance_{inst_str}.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved merged interference ratio plot: {output_path}")

    # === Optional: print weighted stats per instance ===
    print("\n📊 Weighted interference ratios per instance:")
    total_all_decode = 0
    total_all_interfered = 0
    for instance in sorted(args.instances):
        reqs = per_instance[instance]
        if not reqs:
            print(f"  Instance {instance}: no decode steps")
            continue
        total = sum(c['total'] for c in reqs.values())
        interfered = sum(c['interfered'] for c in reqs.values())
        ratio = interfered / total if total > 0 else 0
        print(f"  Instance {instance}: {interfered} / {total} = {ratio:.4f} ({ratio*100:.2f}%)")
        total_all_decode += total
        total_all_interfered += interfered

    if total_all_decode > 0:
        overall_ratio = total_all_interfered / total_all_decode
        print(f"\n✅ Overall (across selected instances): {total_all_interfered} / {total_all_decode} = {overall_ratio:.4f} ({overall_ratio*100:.2f}%)")

    print("\n✅ Analysis complete.")

if __name__ == '__main__':
    main()
