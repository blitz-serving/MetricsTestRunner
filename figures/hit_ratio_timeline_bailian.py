#!/usr/bin/env python3
"""
generate_hit_ratio_timeline.py

This script analyzes router_v2.log files to generate a timeline chart showing
the 'hit_ratio' value per replica over time, extracted from scheduler decision logs.

Usage:
    python generate_hit_ratio_timeline.py <directory_path>
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

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Generate hit_ratio timeline from router_v2.log')
    parser.add_argument('directory', type=str, help='Directory containing router_v2.log file')
    parser.add_argument('--smooth-window', type=int, default=5,
                        help='Sliding window size for smoothing time series (default: 5)')
    parser.add_argument('--replicas', type=int, nargs='+', default=None,
                        help='List of replica IDs to plot (default: all found)')
    parser.add_argument('--start-time', type=float, default=0.0,
                        help='Start time (in seconds) for timeline display (default: 0)')
    parser.add_argument('--end-time', type=float, default=1200.0,
                        help='End time (in seconds) for timeline display (default: 1200)')
    parser.add_argument('--ylim', type=float, default=None,
                        help='Force y-axis limit for hit ratio chart (e.g., 1.0)')
    return parser.parse_args()

def parse_timestamp(log_timestamp):
    """Parse ISO 8601 timestamp from log entries."""
    clean_timestamp = log_timestamp.replace('Z', '')
    return datetime.fromisoformat(clean_timestamp)

def extract_hit_ratio_data(log_file_path):
    """
    Extract hit_ratio events from log.

    Returns:
        dict: {replica_id: [(relative_time_sec, hit_ratio), ...]}
    """
    # Pattern: "2025-11-21T18:31:00.149069Z ... Replica id: 0 hit_ratio 0.5714 num_requests ..."
    pattern = re.compile(
        r'(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d+Z).*?'
        r'Replica id:\s+(\d+)\s+hit_ratio\s+([0-9.]+)\s+num_requests'
    )

    replica_data = {}
    earliest_time = None

    with open(log_file_path, 'r') as f:
        for line in f:
            match = pattern.search(line)
            if not match:
                continue
            timestamp_str, replica_id_str, hit_ratio_str = match.groups()
            replica_id = int(replica_id_str)
            hit_ratio = float(hit_ratio_str)
            timestamp = parse_timestamp(timestamp_str)

            if earliest_time is None:
                earliest_time = timestamp

            rel_time = (timestamp - earliest_time).total_seconds()
            if replica_id not in replica_data:
                replica_data[replica_id] = []
            replica_data[replica_id].append((rel_time, hit_ratio))

    if earliest_time is None:
        raise ValueError("No 'Replica id: ... hit_ratio ...' log entries found.")

    return replica_data, earliest_time

def smooth_series(times, values, window):
    """Apply moving average smoothing."""
    if len(values) < window or window <= 1:
        return times, values
    smoothed = np.convolve(values, np.ones(window) / window, mode='same')
    return times, smoothed

def generate_and_save_chart(replica_data, output_dir, smooth_window=5, time_range=(0.0, 1400.0), selected_replicas=None, ylim=None):
    """Generate and save the hit_ratio timeline chart."""
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
        times, ratios = zip(*replica_data[replica_id])
        times = np.array(times)
        ratios = np.array(ratios)

        # Clip to time range
        mask = (times >= start_time_clip) & (times < end_time_clip)
        if not np.any(mask):
            continue
        clipped_times = times[mask]
        clipped_ratios = ratios[mask]

        smoothed_times, smoothed_ratios = smooth_series(clipped_times, clipped_ratios, smooth_window)

        plt.plot(smoothed_times, smoothed_ratios,
                 label=f'Replica #{replica_id}',
                 linewidth=2,
                 marker='o',
                 markersize=2,
                 alpha=0.8)

    plt.title('Hit Ratio Over Time per Replica', fontsize=14, fontweight='bold')
    plt.xlabel('Time (seconds)', fontsize=12)
    plt.ylabel('Hit Ratio', fontsize=12)
    plt.xlim(start_time_clip, end_time_clip)
    plt.ylim(0, 1.0 if ylim is None else ylim)
    plt.grid(True, alpha=0.3)
    plt.legend(title='Replica', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()

    # Build suffix
    suffix_parts = [f"r{'_'.join(map(str, sorted(plot_replicas)))}", f"smooth{smooth_window}"]
    if start_time_clip != 0 or end_time_clip != 1400:
        suffix_parts.append(f"t{int(start_time_clip)}-{int(end_time_clip)}")
    suffix = "_".join(suffix_parts)

    output_path = os.path.join(output_dir, f"hit_ratio_perinstances_timeline_{suffix}.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Saved hit ratio timeline: {output_path}")

def main():
    args = parse_arguments()

    if not os.path.isdir(args.directory):
        raise ValueError(f"Directory not found: {args.directory}")

    log_file_path = os.path.join(args.directory, LOG_FILENAME)
    if not os.path.exists(log_file_path):
        raise FileNotFoundError(f"Log file not found: {log_file_path}")

    print(f"Processing log file: {log_file_path}")
    replica_data, start_time = extract_hit_ratio_data(log_file_path)

    print(f"Found data for replicas: {sorted(replica_data.keys())}")
    total_points = sum(len(v) for v in replica_data.values())
    print(f"Total hit_ratio log points: {total_points}")

    generate_and_save_chart(
        replica_data,
        output_dir=args.directory,
        smooth_window=args.smooth_window,
        time_range=(args.start_time, args.end_time),
        selected_replicas=args.replicas,
        ylim=args.ylim
    )

    print("Hit ratio timeline generation complete!")

if __name__ == "__main__":
    main()
