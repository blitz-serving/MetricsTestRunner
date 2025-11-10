#!/usr/bin/env python3
"""
analyze_statistics.py

This script reads a 'statistic.log' file from a given directory,
parses JSON-formatted log lines, and generates four timeline plots:
1. all_tokens over time
2. batch size (bs) over time
3. prefill_tokens over time
4. tokens per second (tps) over time

Only the instance IDs specified by the --instances argument (default: [0,7,8,15])
will be plotted to keep the charts clean.

Each log entry is assumed to be reported every TS seconds (default: 2s).
Plots are saved as PNG files in the input directory.

Now supports smoothing via moving average (--smooth-window).
"""

import os
import json
import argparse
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict

# Default time step between log entries (in seconds)
DEFAULT_TS = 2  # seconds

# Default instance IDs to plot
DEFAULT_INSTANCES = [0, 7, 8, 15]

# Default smoothing window
DEFAULT_SMOOTH_WINDOW = 5


def load_logs(log_file_path):
    """
    Load and parse JSON log lines from the given file.

    Returns:
        list[dict]: List of parsed log entries.
    """
    logs = []
    with open(log_file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
                logs.append(entry)
            except json.JSONDecodeError as e:
                print(f"Warning: Skipping invalid JSON line: {line[:100]}... ({e})")
    return logs


def group_logs_by_id(logs):
    """
    Group logs by 'id' field.

    Returns:
        dict[int, list[dict]]: Mapping from instance id to list of log entries.
    """
    grouped = defaultdict(list)
    for log in logs:
        if 'id' not in log:
            print(f"Warning: Skipping log without 'id': {log}")
            continue
        grouped[log['id']].append(log)
    return grouped


def smooth_data_with_time(values, times, window):
    """
    Apply a simple moving average and return smoothed values with correctly aligned timestamps.
    
    Parameters:
        values (list): original values
        times (list): original timestamps
        window (int): window size for moving average

    Returns:
        (smoothed_values, aligned_times): both as lists
    """
    if len(values) <= window:
        return values, times
    smoothed = np.convolve(values, np.ones(window) / window, mode='valid')
    aligned_times = times[window - 1:]  # critical: 'valid' starts at index (window - 1)
    return smoothed.tolist(), aligned_times


def plot_timeline(grouped_logs, ts, output_dir, metric_key, ylabel, title, filename, instance_ids, smooth_window):
    """
    Plot a timeline for a given metric, only for specified instance IDs.
    Uses small markers and thin lines to reduce visual clutter.
    Applies moving average smoothing if smooth_window > 1.
    """
    plt.figure(figsize=(12, 6))

    plotted_any = False
    for instance_id in sorted(instance_ids):
        if instance_id not in grouped_logs:
            print(f"Info: Instance ID {instance_id} not found in logs. Skipping.")
            continue
        logs = grouped_logs[instance_id]
        values = [log.get(metric_key, 0) for log in logs]
        times = [i * ts for i in range(len(values))]

        if smooth_window > 1 and len(values) > smooth_window:
            plot_values, plot_times = smooth_data_with_time(values, times, smooth_window)
        else:
            plot_values, plot_times = values, times

        plt.plot(
            plot_times, plot_values,
            marker='.',        # tiny dot
            markersize=4,      # small marker size
            linewidth=1.2,
            label=f'ID {instance_id}'
        )
        plotted_any = True

    if not plotted_any:
        print(f"Warning: No data to plot for {metric_key}.")
        plt.text(0.5, 0.5, 'No data', horizontalalignment='center',
                 verticalalignment='center', transform=plt.gca().transAxes)

    plt.xlabel('Time (seconds)')
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)  # lighter grid
    plt.tight_layout()

    output_path = os.path.join(output_dir, filename)
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Saved plot: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Analyze statistic.log and generate smoothed timeline plots for selected instances.")
    parser.add_argument('dir', help='Directory containing statistic.log')
    parser.add_argument('--ts', type=float, default=DEFAULT_TS,
                        help=f'Time step between log entries in seconds (default: {DEFAULT_TS})')
    parser.add_argument('--instances', nargs='+', type=int, default=DEFAULT_INSTANCES,
                        help=f'List of instance IDs to plot (default: {DEFAULT_INSTANCES})')
    parser.add_argument('--smooth-window', type=int, default=DEFAULT_SMOOTH_WINDOW,
                        help=f'Moving average window size for smoothing (default: {DEFAULT_SMOOTH_WINDOW}, set <=1 to disable)')

    args = parser.parse_args()

    log_file = os.path.join(args.dir, 'statistic.log')
    if not os.path.isfile(log_file):
        print(f"Error: statistic.log not found in {args.dir}")
        return

    logs = load_logs(log_file)
    if not logs:
        print("Error: No valid logs found.")
        return

    grouped_logs = group_logs_by_id(logs)
    if not grouped_logs:
        print("Error: No logs with valid 'id' found.")
        return

    plot_configs = [
        ('all_tokens', 'All Tokens', 'All Tokens Over Time', 'all_tokens_timeline_smooth.png'),
        ('bs', 'Batch Size (bs)', 'Batch Size Over Time', 'batch_size_timeline_smooth.png'),
        ('prefill_tokens', 'Prefill Tokens', 'Prefill Tokens Over Time', 'prefill_tokens_timeline_smooth.png'),
        ('tps', 'Chunked Prefill Number Per Second (TPS)', 'Chunked Prefill Number Per Second (TPS) Over Time', 'tps_timeline_smooth.png'),
    ]

    for metric_key, ylabel, title, filename in plot_configs:
        plot_timeline(
            grouped_logs=grouped_logs,
            ts=args.ts,
            output_dir=args.dir,
            metric_key=metric_key,
            ylabel=ylabel,
            title=title,
            filename=filename,
            instance_ids=args.instances,
            smooth_window=args.smooth_window
        )

    print(f"All 4 smoothed plots generated for instances: {args.instances} (smooth window = {args.smooth_window})")


if __name__ == '__main__':
    main()