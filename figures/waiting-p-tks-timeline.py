#!/usr/bin/env python3
"""
generate_waiting_tokens_timeline.py

This script analyzes router_v2.log files to generate a timeline chart showing
the 'waiting_tokens' value at each scheduling decision per replica.

Usage:
    python generate_waiting_tokens_timeline.py <directory_path>
"""

import argparse
import re
import os
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
from pathlib import Path

# Constants
LOG_FILENAME = "router_v2.log"
TIME_BIN = 0.5  # for optional binning if needed (not used for direct point plotting)

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Generate waiting_tokens timeline from router_v2.log')
    parser.add_argument('directory', type=str, help='Directory containing router_v2.log file')
    parser.add_argument('--smooth-window', type=int, default=5,
                        help='Sliding window size for smoothing time series (default: 5)')
    parser.add_argument('--replicas', type=int, nargs='+', default=None,
                        help='List of replica IDs to plot (default: all found)')
    parser.add_argument('--start-time', type=float, default=0.0,
                        help='Start time (in seconds) for timeline display (default: 0)')
    parser.add_argument('--end-time', type=float, default=1400.0,
                        help='End time (in seconds) for timeline display (default: 1400)')
    parser.add_argument('--ylim', type=float, default=None,
                        help='Force y-axis limit for waiting tokens chart (e.g., 1000)')
    return parser.parse_args()

def parse_timestamp(log_timestamp):
    """Parse ISO 8601 timestamp from log entries."""
    clean_timestamp = log_timestamp.replace('Z', '')
    return datetime.fromisoformat(clean_timestamp)

def extract_waiting_tokens_data(log_file_path):
    """
    Extract waiting_tokens events from log.

    Returns:
        dict: {replica_id: [(relative_time_sec, waiting_tokens), ...]}
    """
    # Pattern: "2025-11-04T06:36:22.830882Z ... Replica id: 3 waiting_tokens 1250 num_new_tokens 32"
    pattern = re.compile(
        r'(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d+Z).*?'
        r'Replica id:\s+(\d+)\s+waiting_tokens\s+(\d+)\s+num_new_tokens\s+\d+'
    )

    replica_data = {}
    earliest_time = None

    with open(log_file_path, 'r') as f:
        for line in f:
            match = pattern.search(line)
            if not match:
                continue
            timestamp_str, replica_id_str, waiting_tokens_str = match.groups()
            replica_id = int(replica_id_str)
            waiting_tokens = int(waiting_tokens_str)
            timestamp = parse_timestamp(timestamp_str)

            if earliest_time is None:
                earliest_time = timestamp

            rel_time = (timestamp - earliest_time).total_seconds()
            if replica_id not in replica_data:
                replica_data[replica_id] = []
            replica_data[replica_id].append((rel_time, waiting_tokens))

    if earliest_time is None:
        raise ValueError("No 'Replica id: ... waiting_tokens ...' log entries found.")

    return replica_data, earliest_time

def smooth_series(times, values, window):
    """Apply moving average smoothing."""
    if len(values) < window or window <= 1:
        return times, values
    smoothed = np.convolve(values, np.ones(window) / window, mode='same')
    return times, smoothed

def generate_and_save_chart(replica_data, output_dir, smooth_window=5, time_range=(0.0, 1400.0), selected_replicas=None, ylim=None):
    """Generate and save the waiting_tokens timeline chart."""
    start_time_clip, end_time_clip = time_range

    all_replica_ids = sorted(replica_data.keys())
    if selected_replicas is None:
        plot_replicas = all_replica_ids
    else:
        plot_replicas = [rid for rid in selected_replicas if rid in all_replica_ids]
        if not plot_replicas:
            print(f"Warning: None of the specified replicas {selected_replicas} found in log.")
            return

    plt.style.use('seaborn-v0_8')
    plt.figure(figsize=(12, 6))

    for replica_id in plot_replicas:
        times, tokens = zip(*replica_data[replica_id])
        times = np.array(times)
        tokens = np.array(tokens)

        # Clip to time range
        mask = (times >= start_time_clip) & (times < end_time_clip)
        if not np.any(mask):
            continue
        clipped_times = times[mask]
        clipped_tokens = tokens[mask]

        smoothed_times, smoothed_tokens = smooth_series(clipped_times, clipped_tokens, smooth_window)

        plt.plot(smoothed_times, smoothed_tokens,
                 label=f'Replica #{replica_id}',
                 linewidth=2,
                 marker='o',
                 markersize=2,
                 alpha=0.8)

    plt.title('Waiting Tokens Over Time per Replica', fontsize=14, fontweight='bold')
    plt.xlabel('Time (seconds)', fontsize=12)
    plt.ylabel('Waiting Tokens', fontsize=12)
    plt.xlim(start_time_clip, end_time_clip)
    if ylim is not None:
        plt.ylim(0, ylim)
    plt.grid(True, alpha=0.3)
    plt.legend(title='Replica', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()

    # Build suffix
    suffix_parts = [f"r{'_'.join(map(str, sorted(plot_replicas)))}", f"smooth{smooth_window}"]
    if start_time_clip != 0 or end_time_clip != 1400:
        suffix_parts.append(f"t{int(start_time_clip)}-{int(end_time_clip)}")
    suffix = "_".join(suffix_parts)

    output_path = os.path.join(output_dir, f"waiting-p-tks_{suffix}.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Saved waiting tokens timeline: {output_path}")

def main():
    args = parse_arguments()

    if not os.path.isdir(args.directory):
        raise ValueError(f"Directory not found: {args.directory}")

    log_file_path = os.path.join(args.directory, LOG_FILENAME)
    if not os.path.exists(log_file_path):
        raise FileNotFoundError(f"Log file not found: {log_file_path}")

    print(f"Processing log file: {log_file_path}")
    replica_data, start_time = extract_waiting_tokens_data(log_file_path)

    print(f"Found data for replicas: {sorted(replica_data.keys())}")
    total_points = sum(len(v) for v in replica_data.values())
    print(f"Total scheduling decision points: {total_points}")

    generate_and_save_chart(
        replica_data,
        output_dir=args.directory,
        smooth_window=args.smooth_window,
        time_range=(args.start_time, args.end_time),
        selected_replicas=args.replicas,
        ylim=args.ylim
    )

    print("Waiting tokens timeline generation complete!")

if __name__ == "__main__":
    main()
