#!/usr/bin/env python3
"""
Analyze vLLM decode interference with s_time = route_time (from request routing log)
Supports relative time window filtering (e.g., --start-time 600 --end-time 800)
"""

import argparse
import os
import re
import json
from collections import defaultdict
import matplotlib.pyplot as plt
from datetime import datetime, timezone

LOG_FILENAME = "router_v2.log"
TIME_BIN = 0.5  # Not used here, but kept for consistency

def parse_iso_zulu(ts_str):
    if ts_str.endswith('Z'):
        ts_str = ts_str[:-1] + '+00:00'
    return datetime.fromisoformat(ts_str).astimezone(timezone.utc)

# ---- Parsers for different log lines ----

def parse_prefill_log_line(line):
    """Parse prefill/decode state log line (original pattern)"""
    pattern = (
        r'(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d+Z)\s+INFO.+?'
        r'vLLM#(\d+)\s+prefill tokens\s+(\d+)\s+budget\s+\d+\s+'
        r'request states:\s+({.*})'
    )
    match = re.search(pattern, line)
    if not match:
        return None

    ts_str, instance_id, prefill_tokens_str, states_str = match.groups()
    return {
        'timestamp': parse_iso_zulu(ts_str),
        'instance': int(instance_id),
        'prefill_tokens': int(prefill_tokens_str),
        'states': {str(k): str(v).upper() for k, v in json.loads(states_str.replace("'", '"')).items()}
    }

def parse_route_log_line(line):
    """Parse request routing line to get s_time = route_time"""
    # Example: ... Request_123 queued ... with input length 100 output length 50, added to vLLM#0
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
    parser = argparse.ArgumentParser(description="Analyze vLLM interference using route_time as s_time")
    parser.add_argument('dir', type=str, help='Directory containing router_v2.log')
    parser.add_argument('--smooth-window', type=int, default=1, help='Smoothing window size (default: 1)')
    parser.add_argument('--instances', type=int, nargs='+', default=[0, 7, 8, 15], help='vLLM instance IDs to analyze')
    parser.add_argument('--start-time', type=float, default=None, help='Start time in seconds (relative to first route)')
    parser.add_argument('--end-time', type=float, default=None, help='End time in seconds (relative to first route)')
    args = parser.parse_args()

    log_path = os.path.join(args.dir, LOG_FILENAME)
    if not os.path.isfile(log_path):
        raise FileNotFoundError(f"Log file not found: {log_path}")

    # Step 1: Parse all route lines to get s_time = route_time
    route_time = {}  # req_id -> datetime
    route_instance = {}  # req_id -> instance_id
    all_route_timestamps = []

    prefill_entries = []  # List of {'timestamp', 'instance', 'prefill_tokens', 'states'}

    with open(log_path, 'r', encoding='utf-8') as f:
        for line in f:
            # Try route line first
            route = parse_route_log_line(line)
            if route:
                req_id = route['request_id']
                route_time[req_id] = route['timestamp']
                route_instance[req_id] = route['instance_id']
                all_route_timestamps.append(route['timestamp'])
                continue

            # Try prefill/decode state line
            entry = parse_prefill_log_line(line)
            if entry:
                prefill_entries.append(entry)

    if not route_time:
        raise ValueError("No route log entries found – cannot determine s_time for requests.")

    # Determine t0 = earliest route time
    t0 = min(all_route_timestamps)

    # Convert route_time to relative seconds
    route_time_rel = {req: (ts - t0).total_seconds() for req, ts in route_time.items()}

    # Step 2: Filter prefill_entries by relative time window
    filtered_entries = []
    for entry in prefill_entries:
        rel_time = (entry['timestamp'] - t0).total_seconds()
        if args.start_time is not None and rel_time < args.start_time:
            continue
        if args.end_time is not None and rel_time >= args.end_time:
            continue
        filtered_entries.append(entry)

    if not filtered_entries:
        print("No prefill/decode state log entries in the specified time window.")
        return

    # Step 3: Build per-request interference stats, but only for requests with known route_time
    per_request = defaultdict(lambda: {'total': 0, 'interfered': 0})
    per_instance = defaultdict(lambda: defaultdict(lambda: {'total': 0, 'interfered': 0}))

    for entry in filtered_entries:
        instance = entry['instance']
        if instance not in args.instances:
            continue
        has_prefill = entry['prefill_tokens'] > 0
        for req_id_str, state in entry['states'].items():
            try:
                req_id = int(req_id_str)
            except ValueError:
                continue
            # Only consider requests that were routed (i.e., we know their s_time)
            if req_id not in route_time:
                continue
            if route_instance[req_id] != instance:
                # Skip if instance mismatch (should not happen, but safe)
                continue
            if state == 'DECODE':
                per_request[req_id]['total'] += 1
                per_instance[instance][req_id]['total'] += 1
                if has_prefill:
                    per_request[req_id]['interfered'] += 1
                    per_instance[instance][req_id]['interfered'] += 1

    if not per_request:
        print("No DECODE events found for routed requests in the time window.")
        return

    # Step 4: Sort by s_time = route_time_rel[req_id]
    sorted_reqs = sorted(per_request.keys(), key=lambda r: route_time_rel[r])
    x_rel = [route_time_rel[r] for r in sorted_reqs]
    y_total = [per_request[r]['total'] for r in sorted_reqs]
    y_interfered = [per_request[r]['interfered'] for r in sorted_reqs]

    smooth_win = max(1, args.smooth_window)
    y_total_s = smooth_data(y_total, smooth_win)
    y_interfered_s = smooth_data(y_interfered, smooth_win)

    # Plot overall
    plt.figure(figsize=(12, 6))
    plt.plot(x_rel, y_interfered_s, label='Interfered Decode Count', marker='o', markersize=3)
    plt.plot(x_rel, y_total_s, label='Total Decode Count', marker='x', markersize=3)
    plt.xlabel('Request s_time (seconds since first route)')
    plt.ylabel('Count (smoothed)')
    plt.title('vLLM Request Decode Interference (Overall)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    overall_png = os.path.join(args.dir, 'decode_interference_overall.png')
    plt.savefig(overall_png)
    plt.close()
    print(f"Saved overall plot: {overall_png}")

    # Overall ratio
    total_decode_all = sum(y_total)
    total_interfered_all = sum(y_interfered)
    ratio_all = total_interfered_all / total_decode_all if total_decode_all > 0 else 0
    print(f"\n📊 Overall interference ratio: {total_interfered_all} / {total_decode_all} = {ratio_all:.4f} ({ratio_all*100:.2f}%)")

    # Per instance
    print("\n📉 Per-instance interference ratios:")
    for instance in sorted(args.instances):
        inst_reqs = per_instance[instance]
        if not inst_reqs:
            print(f"  Instance {instance}: No DECODE events in time window")
            continue

        inst_sorted_reqs = sorted(inst_reqs.keys(), key=lambda r: route_time_rel[r])
        x_inst = [route_time_rel[r] for r in inst_sorted_reqs]
        y_t = [inst_reqs[r]['total'] for r in inst_sorted_reqs]
        y_i = [inst_reqs[r]['interfered'] for r in inst_sorted_reqs]

        inst_total = sum(y_t)
        inst_interfered = sum(y_i)
        ratio_inst = inst_interfered / inst_total if inst_total > 0 else 0
        print(f"  Instance {instance}: {inst_interfered} / {inst_total} = {ratio_inst:.4f} ({ratio_inst*100:.2f}%)")

        y_t_s = smooth_data(y_t, smooth_win)
        y_i_s = smooth_data(y_i, smooth_win)

        plt.figure(figsize=(12, 6))
        plt.plot(x_inst, y_i_s, label='Interfered Decode Count', marker='o', markersize=3)
        plt.plot(x_inst, y_t_s, label='Total Decode Count', marker='x', markersize=3)
        plt.xlabel('Request s_time (seconds since first route)')
        plt.ylabel('Count (smoothed)')
        plt.title(f'vLLM Request Decode Interference (Instance {instance})')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        inst_png = os.path.join(args.dir, f'decode_interference_instance_{instance}.png')
        plt.savefig(inst_png)
        plt.close()
        print(f"  → Saved plot: {inst_png}")

    print("\n✅ Analysis complete.")

if __name__ == '__main__':
    main()
