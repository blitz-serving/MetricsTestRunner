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
"""

import os
import json
import argparse
import matplotlib.pyplot as plt
from collections import defaultdict

# Default time step between log entries (in seconds)
DEFAULT_TS = 2  # seconds

# Default instance IDs to plot
DEFAULT_INSTANCES = [0, 7, 8, 15]


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

def plot_timeline(grouped_logs, ts, output_dir, metric_key, ylabel, title, filename, instance_ids):
    """
    Plot a timeline for a given metric, only for specified instance IDs.
    Uses small markers and thin lines to reduce visual clutter.
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
        # Use small dots and thin lines to avoid clutter
        plt.plot(
            times, values,
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
    plt.savefig(output_path, dpi=150)  # slightly higher DPI for clarity
    plt.close()
    print(f"Saved plot: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Analyze statistic.log and generate timeline plots for selected instances.")
    parser.add_argument('dir', help='Directory containing statistic.log')
    parser.add_argument('--ts', type=float, default=DEFAULT_TS,
                        help=f'Time step between log entries in seconds (default: {DEFAULT_TS})')
    parser.add_argument('--instances', nargs='+', type=int, default=DEFAULT_INSTANCES,
                        help=f'List of instance IDs to plot (default: {DEFAULT_INSTANCES})')

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
        ('all_tokens', 'All Tokens', 'All Tokens Over Time', 'all_tokens_timeline.png'),
        ('bs', 'Batch Size (bs)', 'Batch Size Over Time', 'batch_size_timeline.png'),
        ('prefill_tokens', 'Prefill Tokens', 'Prefill Tokens Over Time', 'prefill_tokens_timeline.png'),
        ('tps', 'Chunked Prefill Number Per Second (TPS)', 'Chunked Prefill Number Per Second (TPS) Over Time', 'tps_timeline.png'),
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
            instance_ids=args.instances
        )

    print(f"All 4 plots generated for instances: {args.instances}")


if __name__ == '__main__':
    main()