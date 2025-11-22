#!/usr/bin/env python3
"""
generate_actual_prefill_tokens.py

This script analyzes router_v2.log to generate a single timeline chart:
Actual Prefill Tokens Over Time = input_length - hit_cnt.

Usage:
    python generate_actual_prefill_tokens.py <directory_path>
"""

import argparse
import re
import os
from datetime import datetime
from collections import defaultdict, namedtuple
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Constants
TIME_BIN = 0.5  # time bin window in seconds
LOG_FILENAME = "router_v2.log"

# Data structures
RequestRoute = namedtuple('RequestRoute', ['timestamp', 'request_id', 'input_length', 'decode_length', 'instance_id'])
PrefillDone = namedtuple('PrefillDone', ['timestamp', 'request_id', 'hit_cnt', 'instance_id'])
DecodeDone = namedtuple('DecodeDone', ['timestamp', 'request_id', 'instance_id'])
RequestLifecycle = namedtuple('RequestLifecycle', ['request_id', 'instance_id', 'input_length', 'hit_cnt', 'prefill_time'])

def parse_arguments():
    parser = argparse.ArgumentParser(description='Generate timeline chart for actual prefill tokens (input - cache hits)')
    parser.add_argument('directory', type=str, help='Directory containing router_v2.log file')
    parser.add_argument('--smooth-window', type=int, default=5,
                        help='Sliding window size for smoothing time series (default: 5)')
    parser.add_argument('--instances', type=int, nargs='+', default=[0, 7, 8, 15],
                        help='List of vLLM instance IDs to plot (default: 0 7 8 15)')
    parser.add_argument('--start-time', type=float, default=0.0,
                        help='Start time (in seconds) for timeline display (default: 0)')
    parser.add_argument('--end-time', type=float, default=1400.0,
                        help='End time (in seconds) for timeline display (default: 1400)')
    parser.add_argument('--actual-prefill-token-ylim', type=float, default=None,
                        help='Force y-axis limit for actual prefill token chart (e.g., 1500)')
    return parser.parse_args()

def parse_timestamp(log_timestamp):
    clean_timestamp = log_timestamp.replace('Z', '')
    return datetime.fromisoformat(clean_timestamp)

def extract_log_data(log_file_path):
    request_routes = {}
    prefill_dones = {}
    decode_dones = {}
    earliest_time = None
    latest_time = None

    route_pattern = re.compile(
        r'(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d+Z).*?'
        r'Request_(\d+)\s+queued\s+\d+us,\s+with\s+input\s+length\s+(\d+)\s+output\s+length\s+(\d+),\s+added\s+to\s+vLLM#(\d+)'
    )
    prefill_pattern = re.compile(
        r'(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d+Z).*?'
        r'Vllm#(\d+)::Request_(\d+)\s+prefill\s+done\s+with\s+(\d+)\s+actual\s+hit\s+tokens!'
    )
    backup_prefill_pattern = re.compile(
        r'(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d+Z).*?'
        r'Vllm#(\d+)::Request_(\d+)\s+prefill\s+with\s+(\d+)\s+actual\s+hit\s+tokens\s+done!'
    )
    decode_pattern = re.compile(
        r'(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d+Z).*?'
        r'Vllm#(\d+)::Request_(\d+)\s+is\s+finished'
    )

    with open(log_file_path, 'r') as file:
        for line in file:
            line = line.strip()
            if not line:
                continue

            route_match = route_pattern.search(line)
            if route_match:
                ts, rid, il, dl, iid = (
                    route_match.group(1),
                    int(route_match.group(2)),
                    int(route_match.group(3)),
                    int(route_match.group(4)),
                    int(route_match.group(5)),
                )
                timestamp = parse_timestamp(ts)
                request_routes[rid] = RequestRoute(timestamp, rid, il, dl, iid)
                if earliest_time is None or timestamp < earliest_time:
                    earliest_time = timestamp
                if latest_time is None or timestamp > latest_time:
                    latest_time = timestamp
                continue

            prefill_match = prefill_pattern.search(line) or backup_prefill_pattern.search(line)
            if prefill_match:
                ts, iid, rid, hit = (
                    prefill_match.group(1),
                    int(prefill_match.group(2)),
                    int(prefill_match.group(3)),
                    int(prefill_match.group(4)),
                )
                timestamp = parse_timestamp(ts)
                prefill_dones[rid] = PrefillDone(timestamp, rid, hit, iid)
                if latest_time is None or timestamp > latest_time:
                    latest_time = timestamp
                continue

            decode_match = decode_pattern.search(line)
            if decode_match:
                ts, iid, rid = (
                    decode_match.group(1),
                    int(decode_match.group(2)),
                    int(decode_match.group(3)),
                )
                timestamp = parse_timestamp(ts)
                decode_dones[rid] = DecodeDone(timestamp, rid, iid)
                if latest_time is None or timestamp > latest_time:
                    latest_time = timestamp

    if earliest_time is None or latest_time is None:
        raise ValueError("No valid log entries found")

    return request_routes, prefill_dones, decode_dones, earliest_time, latest_time

def build_request_lifecycles(request_routes, prefill_dones, decode_dones):
    lifecycles = []
    incomplete = 0
    mismatch = 0
    for req_id, route in request_routes.items():
        if req_id not in prefill_dones or req_id not in decode_dones:
            incomplete += 1
            continue
        prefill = prefill_dones[req_id]
        decode = decode_dones[req_id]
        if not (route.instance_id == prefill.instance_id == decode.instance_id):
            mismatch += 1
            continue
        lifecycles.append(RequestLifecycle(
            request_id=req_id,
            instance_id=route.instance_id,
            input_length=route.input_length,
            hit_cnt=prefill.hit_cnt,
            prefill_time=prefill.timestamp
        ))
    print(f"Warning: {incomplete=}, {mismatch=}")
    return lifecycles

def calculate_time_bins(start_time, end_time, time_bin=TIME_BIN):
    total_duration = (end_time - start_time).total_seconds()
    num_bins = int(np.ceil(total_duration / time_bin))
    bin_edges = [i * time_bin for i in range(num_bins + 1)]
    bin_centers = [(bin_edges[i] + bin_edges[i+1]) / 2 for i in range(num_bins)]
    return bin_centers

def compute_actual_prefill_tokens(lifecycles, bin_centers, start_time):
    instance_ids = sorted(set(l.instance_id for l in lifecycles))
    n_bins = len(bin_centers)
    actual_prefill = defaultdict(lambda: [0] * n_bins)

    for lc in lifecycles:
        actual_tokens = lc.input_length - lc.hit_cnt
        if actual_tokens < 0:
            actual_tokens = 0  # safety
        prefill_time_rel = (lc.prefill_time - start_time).total_seconds()
        bin_idx = int(prefill_time_rel / TIME_BIN)
        if 0 <= bin_idx < n_bins:
            actual_prefill[lc.instance_id][bin_idx] += actual_tokens

    # Convert bins to tokens/sec
    for inst in actual_prefill:
        actual_prefill[inst] = [v / TIME_BIN for v in actual_prefill[inst]]

    return dict(actual_prefill)

def plot_actual_prefill_tokens(actual_prefill, bin_centers, instance_ids, selected_instances,
                               output_dir, smooth_window, time_range, ylim):
    start_clip, end_clip = time_range
    bin_centers = np.array(bin_centers)
    valid = (bin_centers >= start_clip) & (bin_centers < end_clip)
    if not np.any(valid):
        print("No data in specified time range.")
        return

    clipped_time = bin_centers[valid]
    filtered_insts = [i for i in selected_instances if i in instance_ids]
    if not filtered_insts:
        print("No selected instances found in data.")
        return

    plt.figure(figsize=(12, 6))
    for inst in filtered_insts:
        values = np.array(actual_prefill[inst], dtype=float)
        clipped_values = values[valid]

        if smooth_window > 1 and len(clipped_values) >= smooth_window:
            smoothed = np.convolve(clipped_values, np.ones(smooth_window)/smooth_window, mode='same')
            y_vals = smoothed
        else:
            y_vals = clipped_values

        plt.plot(clipped_time, y_vals, label=f'Instance #{inst}',
                 linewidth=2, marker='o', markersize=3, alpha=0.8)

    plt.title('Actual Prefill Tokens Over Time (input_length - hit_cnt)', fontsize=14, fontweight='bold')
    plt.xlabel('Time (seconds)', fontsize=12)
    plt.ylabel('Actual Prefill Throughput (tokens/sec)', fontsize=12)
    plt.xlim(start_clip, min(end_clip, clipped_time[-1] if len(clipped_time) > 0 else end_clip))
    if ylim is not None:
        plt.ylim(0, ylim)
    plt.grid(True, alpha=0.3)
    plt.legend(title='vLLM Instance', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()

    suffix = '_'.join(map(str, sorted(filtered_insts))) + f"_smooth{smooth_window}"
    if start_clip != 0 or end_clip != 1400:
        suffix += f"_t{int(start_clip)}-{int(end_clip)}"
    output_path = os.path.join(output_dir, f"actual_prefill_tokens_{suffix}.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved chart: {output_path}")

def main():
    args = parse_arguments()
    log_path = os.path.join(args.directory, LOG_FILENAME)
    if not os.path.exists(log_path):
        raise FileNotFoundError(f"Log file not found: {log_path}")

    print(f"Processing {log_path}")
    routes, prefill, decode, start_time, end_time = extract_log_data(log_path)
    lifecycles = build_request_lifecycles(routes, prefill, decode)
    if not lifecycles:
        print("No complete lifecycles found.")
        return

    print(f"Built {len(lifecycles)} lifecycles")
    instance_ids = sorted(set(l.instance_id for l in lifecycles))
    bin_centers = calculate_time_bins(start_time, end_time)
    actual_prefill = compute_actual_prefill_tokens(lifecycles, bin_centers, start_time)

    plot_actual_prefill_tokens(
        actual_prefill, bin_centers, instance_ids, args.instances,
        args.directory, args.smooth_window,
        (args.start_time, args.end_time),
        args.actual_prefill_token_ylim
    )
    print("Done.")

if __name__ == "__main__":
    main()
