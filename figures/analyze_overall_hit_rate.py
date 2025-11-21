#!/usr/bin/env python3
"""
generate_overall_cache_hit_rate.py

This script analyzes router_v2.log to generate a single timeline chart:
**Overall Prefix Cache Hit Rate** across all vLLM instances.

It aggregates input_length and hit_cnt (cache hit tokens) from all requests
into time bins, then computes hit rate = sum(hit_cnt) / sum(input_length)
per bin.

Usage:
    python generate_overall_cache_hit_rate.py <directory_path>
"""

import argparse
import re
import os
from datetime import datetime
from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Constants
TIME_BIN = 0.5  # time bin window in seconds
LOG_FILENAME = "router_v2.log"

def parse_arguments():
    parser = argparse.ArgumentParser(description='Generate overall cache hit rate timeline from router_v2.log')
    parser.add_argument('directory', type=str, help='Directory containing router_v2.log file')
    parser.add_argument('--smooth-window', type=int, default=5,
                        help='Sliding window size for smoothing time series (default: 5)')
    parser.add_argument('--start-time', type=float, default=0.0,
                        help='Start time (in seconds) for timeline display (default: 0)')
    parser.add_argument('--end-time', type=float, default=1400.0,
                        help='End time (in seconds) for timeline display (default: 1400)')
    return parser.parse_args()

def parse_timestamp(log_timestamp):
    clean_timestamp = log_timestamp.replace('Z', '')
    return datetime.fromisoformat(clean_timestamp)

def extract_log_data(log_file_path):
    request_routes = {}
    prefill_dones = {}
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

    with open(log_file_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            route_match = route_pattern.search(line)
            if route_match:
                ts_str, req_id, input_len, _, instance_id = (
                    route_match.group(1),
                    int(route_match.group(2)),
                    int(route_match.group(3)),
                    int(route_match.group(4)),
                    int(route_match.group(5)),
                )
                ts = parse_timestamp(ts_str)
                request_routes[req_id] = (ts, input_len, instance_id)
                if earliest_time is None or ts < earliest_time:
                    earliest_time = ts
                if latest_time is None or ts > latest_time:
                    latest_time = ts
                continue

            prefill_match = prefill_pattern.search(line) or backup_prefill_pattern.search(line)
            if prefill_match:
                ts_str, instance_id, req_id, hit_cnt = (
                    prefill_match.group(1),
                    int(prefill_match.group(2)),
                    int(prefill_match.group(3)),
                    int(prefill_match.group(4)),
                )
                ts = parse_timestamp(ts_str)
                prefill_dones[req_id] = (ts, hit_cnt, instance_id)
                if latest_time is None or ts > latest_time:
                    latest_time = ts

    if earliest_time is None or latest_time is None:
        raise ValueError("No valid log entries found")

    return request_routes, prefill_dones, earliest_time, latest_time

def build_overall_data(request_routes, prefill_dones):
    """
    Build list of (prefill_time, input_length, hit_cnt) for all complete requests,
    ignoring instance_id.
    """
    data = []
    matched = 0
    for req_id, (route_time, input_len, _) in request_routes.items():
        if req_id in prefill_dones:
            prefill_time, hit_cnt, _ = prefill_dones[req_id]
            data.append((prefill_time, input_len, hit_cnt))
            matched += 1
    print(f"Matched {matched} requests with prefill completion.")
    return data

def compute_overall_hit_rate(data, start_time, smooth_window=5, time_range=(0.0, 1400.0)):
    """
    Compute overall cache hit rate per time bin.
    Returns: (clipped_bin_centers, clipped_hit_rates)
    """
    total_duration = max((prefill_time - start_time).total_seconds() for prefill_time, _, _ in data)
    num_bins = int(np.ceil(total_duration / TIME_BIN)) + 1
    bin_edges = np.arange(0, (num_bins + 1) * TIME_BIN, TIME_BIN)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    sum_input = np.zeros(num_bins)
    sum_hit = np.zeros(num_bins)

    for prefill_time, input_len, hit_cnt in data:
        t_rel = (prefill_time - start_time).total_seconds()
        bin_idx = min(int(t_rel / TIME_BIN), num_bins - 1)
        sum_input[bin_idx] += input_len
        sum_hit[bin_idx] += hit_cnt

    # Compute hit rate per bin (avoid division by zero)
    hit_rate = np.divide(sum_hit, sum_input, out=np.zeros_like(sum_hit), where=sum_input != 0)

    # Clip to time range
    start_clip, end_clip = time_range
    valid = (bin_centers >= start_clip) & (bin_centers < end_clip)
    if not np.any(valid):
        raise ValueError(f"No data in time range [{start_clip}, {end_clip})")

    clipped_centers = bin_centers[valid]
    clipped_rates = hit_rate[valid]

    # Apply smoothing
    if smooth_window > 1 and len(clipped_rates) >= smooth_window:
        smoothed = np.convolve(clipped_rates, np.ones(smooth_window) / smooth_window, mode='same')
        # Preserve edges (optional, but 'same' already does)
        clipped_rates = smoothed

    return clipped_centers, clipped_rates

def plot_and_save(clipped_centers, clipped_rates, output_dir, smooth_window, time_range):
    start_clip, end_clip = time_range
    plt.figure(figsize=(12, 6))
    plt.plot(clipped_centers, clipped_rates, linewidth=2, color='tab:blue', alpha=0.9)

    plt.title('Overall Prefix Cache Hit Rate Over Time', fontsize=14, fontweight='bold')
    plt.xlabel('Time (seconds)', fontsize=12)
    plt.ylabel('Cache Hit Rate', fontsize=12)
    plt.ylim(0, 1)
    plt.xlim(start_clip, min(end_clip, clipped_centers[-1] if len(clipped_centers) > 0 else end_clip))
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    # Build filename suffix
    suffix = f"smooth{smooth_window}_t{int(start_clip)}-{int(end_clip)}"
    output_path = os.path.join(output_dir, f"overall_cache_hit_rate_{suffix}.png")

    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved overall cache hit rate chart: {output_path}")

def main():
    args = parse_arguments()
    log_path = os.path.join(args.directory, LOG_FILENAME)
    if not os.path.exists(log_path):
        raise FileNotFoundError(f"Log file not found: {log_path}")

    print(f"Processing {log_path}")
    request_routes, prefill_dones, start_time, end_time = extract_log_data(log_path)
    data = build_overall_data(request_routes, prefill_dones)

    if not data:
        print("No complete request-prefill pairs found. Exiting.")
        return

    clipped_centers, clipped_rates = compute_overall_hit_rate(
        data,
        start_time,
        smooth_window=args.smooth_window,
        time_range=(args.start_time, args.end_time)
    )

    plot_and_save(
        clipped_centers,
        clipped_rates,
        args.directory,
        args.smooth_window,
        (args.start_time, args.end_time)
    )

    print("Overall cache hit rate chart generation completed.")

if __name__ == "__main__":
    main()
