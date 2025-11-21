#!/usr/bin/env python3
"""
generate_load_with_time.py

This script analyzes router_v2.log files to generate timeline charts showing
various metrics across different vLLM instances. It processes log entries to
track request lifecycles and generates 6 timeline charts, including prefill token throughput.

Usage:
    python generate_load_with_time.py <directory_path>
"""

import argparse
import re
import os
import json
from datetime import datetime
from collections import defaultdict, namedtuple
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Constants
TIME_BIN = 0.5  # time bin window in seconds
LOG_FILENAME = "router_v2.log"

# Define data structures for better organization
RequestRoute = namedtuple('RequestRoute', ['timestamp', 'request_id', 'input_length', 'decode_length', 'instance_id'])
PrefillDone = namedtuple('PrefillDone', ['timestamp', 'request_id', 'hit_cnt', 'instance_id'])
DecodeDone = namedtuple('DecodeDone', ['timestamp', 'request_id', 'instance_id'])
RequestLifecycle = namedtuple('RequestLifecycle', ['request_id', 'instance_id', 'input_length', 'decode_length',
                                                  'route_time', 'prefill_time', 'decode_time', 'hit_cnt'])

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Generate timeline charts from router_v2.log')
    parser.add_argument('directory', type=str, help='Directory containing router_v2.log file')
    parser.add_argument('--smooth-window', type=int, default=5,
                        help='Sliding window size for smoothing time series (default: 5)')
    parser.add_argument('--instances', type=int, nargs='+', default=[0, 7, 8, 15],
                        help='List of vLLM instance IDs to plot (default: 0 7 8 15)')
    parser.add_argument('--start-time', type=float, default=0.0,
                        help='Start time (in seconds) for timeline display (default: 0)')
    parser.add_argument('--end-time', type=float, default=1400.0,
                        help='End time (in seconds) for timeline display (default: 1400)')
    parser.add_argument('--prefill-request-ylim', type=float, default=None,
                        help='Force y-axis limit for prefill request count chart (e.g., 50)')
    parser.add_argument('--prefill-throughput-ylim', type=float, default=None,
                        help='Force y-axis limit for prefill token throughput chart (e.g., 2000)')
    return parser.parse_args()

def parse_timestamp(log_timestamp):
    """
    Parse ISO 8601 timestamp from log entries.
    
    Args:
        log_timestamp (str): Timestamp string from log (e.g., '2025-11-04T06:36:22.830882Z')
    
    Returns:
        datetime: Parsed datetime object
    """
    clean_timestamp = log_timestamp.replace('Z', '')
    return datetime.fromisoformat(clean_timestamp)

def extract_log_data(log_file_path):
    """
    Extract relevant data from router_v2.log file.
    
    Args:
        log_file_path (str): Path to the log file
    
    Returns:
        tuple: (request_routes, prefill_dones, decode_dones, start_time, end_time)
    """
    request_routes = {}
    prefill_dones = {}
    decode_dones = {}
    earliest_time = None
    latest_time = None
    
    # Pattern 1: Request routing with input_length and decode_length
    route_pattern = re.compile(
        r'(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d+Z).*?'
        r'Request_(\d+)\s+queued\s+\d+us,\s+with\s+input\s+length\s+(\d+)\s+output\s+length\s+(\d+),\s+added\s+to\s+vLLM#(\d+)'
    )

    # Pattern 2: Prefill done with hit_cnt
    prefill_pattern = re.compile(
        r'(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d+Z).*?'
        r'Vllm#(\d+)::Request_(\d+)\s+prefill\s+done\s+with\s+(\d+)\s+actual\s+hit\s+tokens!'
    )
    
    backup_prefill_pattern = re.compile(
        r'(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d+Z).*?'
        r'Vllm#(\d+)::Request_(\d+)\s+prefill\s+with\s+(\d+)\s+actual\s+hit\s+tokens\s+done!'
    )

    # Pattern 3: Decode done
    decode_pattern = re.compile(
        r'(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d+Z).*?'
        r'Vllm#(\d+)::Request_(\d+)\s+is\s+finished\s+generating\s+(\d+)\s+tokens'
    )
    
    with open(log_file_path, 'r') as file:
        for line in file:
            line = line.strip()
            if not line:
                continue
            
            # Try to match routing information
            route_match = route_pattern.search(line)
            if route_match:
                timestamp_str, req_id, input_length, decode_length, instance_id = (
                    route_match.group(1),
                    int(route_match.group(2)),
                    int(route_match.group(3)),
                    int(route_match.group(4)),
                    int(route_match.group(5)),
                )
                timestamp = parse_timestamp(timestamp_str)
                request_routes[req_id] = RequestRoute(
                    timestamp=timestamp,
                    request_id=req_id,
                    input_length=input_length,
                    decode_length=decode_length,
                    instance_id=instance_id
                )
                if earliest_time is None or timestamp < earliest_time:
                    earliest_time = timestamp
                if latest_time is None or timestamp > latest_time:
                    latest_time = timestamp
                continue
            
            # Try to match prefill completion
            prefill_match = prefill_pattern.search(line)
            if not prefill_match:
                prefill_match = backup_prefill_pattern.search(line)
            if prefill_match:
                timestamp_str, instance_id, req_id, hit_cnt = (
                    prefill_match.group(1),
                    int(prefill_match.group(2)),
                    int(prefill_match.group(3)),
                    int(prefill_match.group(4)),
                )
                timestamp = parse_timestamp(timestamp_str)
                prefill_dones[req_id] = PrefillDone(
                    timestamp=timestamp,
                    request_id=req_id,
                    hit_cnt=hit_cnt,
                    instance_id=instance_id
                )
                if latest_time is None or timestamp > latest_time:
                    latest_time = timestamp
                continue
            
            # Try to match decode completion
            decode_match = decode_pattern.search(line)
            if decode_match:
                timestamp_str, instance_id, req_id = (
                    decode_match.group(1),
                    int(decode_match.group(2)),
                    int(decode_match.group(3)),
                )
                timestamp = parse_timestamp(timestamp_str)
                decode_dones[req_id] = DecodeDone(
                    timestamp=timestamp,
                    request_id=req_id,
                    instance_id=instance_id
                )
                if latest_time is None or timestamp > latest_time:
                    latest_time = timestamp
    
    if earliest_time is None or latest_time is None:
        raise ValueError("No valid log entries found to determine time range")
    
    return request_routes, prefill_dones, decode_dones, earliest_time, latest_time

def build_request_lifecycles(request_routes, prefill_dones, decode_dones):
    """
    Build complete request lifecycles from extracted log data.
    
    Args:
        request_routes (dict): Request routing information
        prefill_dones (dict): Prefill completion information
        decode_dones (dict): Decode completion information
    
    Returns:
        list: List of RequestLifecycle objects
    """
    lifecycles = []
    incomplete_lifectyle_cnt = 0
    wrong_instance_id_cnt = 0
    incomplete_lifectyle_set = set()
    for req_id, route in request_routes.items():
        if req_id not in prefill_dones or req_id not in decode_dones:
            incomplete_lifectyle_cnt += 1
            incomplete_lifectyle_set.add(req_id)
            continue  # Skip incomplete lifecycles
        
        prefill = prefill_dones[req_id]
        decode = decode_dones[req_id]
        
        # Validate that the instance IDs match across the lifecycle
        if route.instance_id != prefill.instance_id or route.instance_id != decode.instance_id:
            wrong_instance_id_cnt += 1
            continue
        
        lifecycle = RequestLifecycle(
            request_id=req_id,
            instance_id=route.instance_id,
            input_length=route.input_length,
            decode_length=route.decode_length,
            route_time=route.timestamp,
            prefill_time=prefill.timestamp,
            decode_time=decode.timestamp,
            hit_cnt=prefill.hit_cnt
        )
        lifecycles.append(lifecycle)
    
    print(f"warning: {incomplete_lifectyle_cnt=} {incomplete_lifectyle_set=} {wrong_instance_id_cnt=}")
    return lifecycles

def calculate_time_bins(start_time, end_time, time_bin=TIME_BIN):
    """
    Calculate time bins and their centers.
    
    Args:
        start_time (datetime): Start timestamp
        end_time (datetime): End timestamp
        time_bin (float): Time bin duration in seconds
    
    Returns:
        tuple: (bin_edges, bin_centers, total_duration)
    """
    total_duration = (end_time - start_time).total_seconds()
    num_bins = int(np.ceil(total_duration / time_bin))
    
    bin_edges = [i * time_bin for i in range(num_bins + 1)]
    bin_centers = [(bin_edges[i] + bin_edges[i + 1]) / 2 for i in range(num_bins)]
    
    return bin_edges, bin_centers, total_duration

def compute_metrics(lifecycles, bin_edges, start_time):
    """
    Compute metrics for each time bin and instance.
    
    Args:
        lifecycles (list): List of RequestLifecycle objects
        bin_edges (list): Time bin edges in seconds
        start_time (datetime): Reference start time
    
    Returns:
        dict: Metrics organized by instance and time bin
    """
    instance_ids = sorted(set(lifecycle.instance_id for lifecycle in lifecycles))
    
    metrics = {
        'prefill_req_number': defaultdict(lambda: [0] * (len(bin_edges) - 1)),
        'decode_req_number': defaultdict(lambda: [0] * (len(bin_edges) - 1)),
        'total_prefill_length': defaultdict(lambda: [0] * (len(bin_edges) - 1)),
        'total_decode_length': defaultdict(lambda: [0] * (len(bin_edges) - 1)),
        'sum_input_length': defaultdict(lambda: [0] * (len(bin_edges) - 1)),
        'sum_cache_length': defaultdict(lambda: [0] * (len(bin_edges) - 1))
    }
    
    for lifecycle in lifecycles:
        instance_id = lifecycle.instance_id
        
        route_time_rel = (lifecycle.route_time - start_time).total_seconds()
        prefill_time_rel = (lifecycle.prefill_time - start_time).total_seconds()
        decode_time_rel = (lifecycle.decode_time - start_time).total_seconds()
        
        # Prefill phase: from route_time to prefill_time
        prefill_start_bin = max(0, int(route_time_rel / TIME_BIN))
        prefill_end_bin = min(len(bin_edges) - 2, int(prefill_time_rel / TIME_BIN))
        for bin_idx in range(prefill_start_bin, prefill_end_bin + 1):
            bin_start = bin_idx * TIME_BIN
            bin_end = (bin_idx + 1) * TIME_BIN
            if bin_end > route_time_rel and bin_start < prefill_time_rel:
                metrics['prefill_req_number'][instance_id][bin_idx] += 1
                metrics['total_prefill_length'][instance_id][bin_idx] += lifecycle.input_length
                metrics['sum_input_length'][instance_id][bin_idx] += lifecycle.input_length
                metrics['sum_cache_length'][instance_id][bin_idx] += lifecycle.hit_cnt
        
        # Decode phase: from prefill_time to decode_time
        decode_start_bin = max(0, int(prefill_time_rel / TIME_BIN))
        decode_end_bin = min(len(bin_edges) - 2, int(decode_time_rel / TIME_BIN))
        for bin_idx in range(decode_start_bin, decode_end_bin + 1):
            bin_start = bin_idx * TIME_BIN
            bin_end = (bin_idx + 1) * TIME_BIN
            if bin_end > prefill_time_rel and bin_start < decode_time_rel:
                metrics['decode_req_number'][instance_id][bin_idx] += 1
                metrics['total_decode_length'][instance_id][bin_idx] += lifecycle.decode_length

    # Calculate prefix cache rate
    metrics['prefix_cache_rate'] = defaultdict(lambda: [0.0] * (len(bin_edges) - 1))
    for instance_id in instance_ids:
        for bin_idx in range(len(bin_edges) - 1):
            sum_input = metrics['sum_input_length'][instance_id][bin_idx]
            sum_cache = metrics['sum_cache_length'][instance_id][bin_idx]
            if sum_input > 0:
                cache_rate = sum_cache / sum_input
                metrics['prefix_cache_rate'][instance_id][bin_idx] = round(cache_rate, 2)
            else:
                metrics['prefix_cache_rate'][instance_id][bin_idx] = 0.0

    # NEW METRIC: prefill token throughput (at prefill completion time), in tokens per bin
    metrics['prefill_token_throughput'] = defaultdict(lambda: [0] * (len(bin_edges) - 1))
    for lifecycle in lifecycles:
        instance_id = lifecycle.instance_id
        prefill_time_rel = (lifecycle.prefill_time - start_time).total_seconds()
        bin_idx = int(prefill_time_rel / TIME_BIN)
        if 0 <= bin_idx < len(bin_edges) - 1:
            metrics['prefill_token_throughput'][instance_id][bin_idx] += lifecycle.input_length

    # Convert defaultdicts to dicts
    for metric_name in metrics:
        metrics[metric_name] = dict(metrics[metric_name])
    
    return metrics

def generate_and_save_charts(metrics, bin_centers, all_instance_ids, selected_instances, output_dir, smooth_window=5, time_range=(0.0, 1400.0), prefill_request_ylim=None, prefill_throughput_ylim=None):
    """
    Generate and save the timeline charts with optional smoothing, time range clipping, and y-limits.
    
    Args:
        metrics (dict): Computed metrics
        bin_centers (list): Time bin centers in seconds
        all_instance_ids (list): All instance IDs present in data
        selected_instances (list): User-specified instance IDs to plot
        output_dir (str): Directory to save the charts
        smooth_window (int): Size of the moving average window
        time_range (tuple): (start_time, end_time) in seconds to clip the timeline
        prefill_request_ylim (float or None): y-limit for prefill request chart
        prefill_throughput_ylim (float or None): y-limit for prefill token throughput chart
    """
    start_time_clip, end_time_clip = time_range
    bin_centers = np.array(bin_centers)
    
    valid_indices = (bin_centers >= start_time_clip) & (bin_centers < end_time_clip)
    if not np.any(valid_indices):
        print(f"Warning: No data in time range [{start_time_clip}, {end_time_clip}). Skipping charts.")
        return

    clipped_bin_centers = bin_centers[valid_indices]
    
    chart_configs = [
        {
            'metric_name': 'prefill_req_number',
            'title': 'Prefill Request Count Over Time',
            'ylabel': 'Number of Requests in Prefill',
            'filename': 'prefill_req_number.png',
        },
        {
            'metric_name': 'decode_req_number',
            'title': 'Decode Request Count Over Time',
            'ylabel': 'Number of Requests in Decode',
            'filename': 'decode_req_number.png',
        },
        {
            'metric_name': 'total_prefill_length',
            'title': 'Total Prefill Length Over Time',
            'ylabel': 'Cumulative Input Length',
            'filename': 'total_prefill_length.png',
        },
        {
            'metric_name': 'prefill_token_throughput',
            'title': 'Prefill Token Throughput Over Time',
            'ylabel': 'Prefill Throughput (tokens/sec)',
            'filename': 'prefill_token_throughput.png',
        },
        {
            'metric_name': 'total_decode_length',
            'title': 'Total Decode Length Over Time',
            'ylabel': 'Cumulative Decode Length',
            'filename': 'total_decode_length.png',
        },
        {
            'metric_name': 'prefix_cache_rate',
            'title': 'Prefix Cache Hit Rate Over Time',
            'ylabel': 'Cache Hit Rate',
            'filename': 'prefix_cache_rate.png',
        }
    ]
    
    filtered_instance_ids = [inst for inst in selected_instances if inst in all_instance_ids]
    if not filtered_instance_ids:
        print(f"Warning: No instances from {selected_instances} found in data. No charts will be plotted.")
        return

    print(f"Selected instance ids = {filtered_instance_ids}")
    print(f"Clipping timeline to [{start_time_clip}, {end_time_clip}) seconds")

    plt.style.use('seaborn-v0_8')
    
    for config in chart_configs:
        plt.figure(figsize=(12, 6))
        
        metric_name = config['metric_name']
        if metric_name not in metrics:
            print(f"Warning: Metric '{metric_name}' not found in computed metrics")
            continue
        
        for instance_id in filtered_instance_ids:
            if instance_id in metrics[metric_name]:
                values = np.array(metrics[metric_name][instance_id], dtype=float)
                clipped_values = values[valid_indices]
                # Convert throughput from tokens/bin to tokens/sec
                if metric_name == 'prefill_token_throughput':
                    clipped_values = clipped_values / TIME_BIN
                
                if smooth_window > 1 and len(clipped_values) >= smooth_window:
                    smoothed = np.convolve(clipped_values, np.ones(smooth_window)/smooth_window, mode='same')
                    y_vals = smoothed
                else:
                    y_vals = clipped_values
                
                plt.plot(clipped_bin_centers, y_vals,
                        label=f'Instance #{instance_id}',
                        linewidth=2,
                        marker='o',
                        markersize=3,
                        alpha=0.8)
        
        plt.title(config['title'], fontsize=14, fontweight='bold')
        plt.xlabel('Time (seconds)', fontsize=12)
        plt.ylabel(config['ylabel'], fontsize=12)
        plt.xlim(start_time_clip, min(end_time_clip, clipped_bin_centers[-1] if len(clipped_bin_centers) > 0 else end_time_clip))
        plt.grid(True, alpha=0.3)
        plt.legend(title='vLLM Instance', bbox_to_anchor=(1.05, 1), loc='upper left')
        
        # Apply y-limits for specific charts
        if metric_name == 'prefill_req_number' and prefill_request_ylim is not None:
            plt.ylim(0, prefill_request_ylim)
        elif metric_name == 'prefill_token_throughput' and prefill_throughput_ylim is not None:
            plt.ylim(0, prefill_throughput_ylim)
        
        plt.tight_layout()
        
        # Add suffix to filename
        instance_suffix = '_'.join(map(str, sorted(filtered_instance_ids)))
        instance_suffix += f"_smooth{smooth_window}"
        if start_time_clip != 0 or end_time_clip != 1400:
            instance_suffix += f"_t{int(start_time_clip)}-{int(end_time_clip)}"
        
        base = Path(config['filename']).stem
        ext = Path(config['filename']).suffix
        output_path = os.path.join(output_dir, f"{base}_{instance_suffix}{ext}")
        
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Saved chart (smoothed with window={smooth_window}, time range=[{start_time_clip}, {end_time_clip})): {output_path}")

def main():
    """Main function to orchestrate the analysis."""
    args = parse_arguments()
    
    if not os.path.isdir(args.directory):
        raise ValueError(f"Directory not found: {args.directory}")
    
    log_file_path = os.path.join(args.directory, LOG_FILENAME)
    if not os.path.exists(log_file_path):
        raise FileNotFoundError(f"Log file not found: {log_file_path}")
    
    print(f"Processing log file: {log_file_path}")
    
    request_routes, prefill_dones, decode_dones, start_time, end_time = extract_log_data(log_file_path)
    
    print(f"Found {len(request_routes)} route events, {len(prefill_dones)} prefill events, {len(decode_dones)} decode events")
    
    lifecycles = build_request_lifecycles(request_routes, prefill_dones, decode_dones)
    
    print(f"Built {len(lifecycles)} complete request lifecycles")
    
    if not lifecycles:
        print("No complete request lifecycles found. Cannot generate charts.")
        return
    
    bin_edges, bin_centers, total_duration = calculate_time_bins(start_time, end_time)
    
    print(f"Time range: {total_duration:.2f} seconds, Number of bins: {len(bin_edges)-1}")
    
    instance_ids = sorted(set(lifecycle.instance_id for lifecycle in lifecycles))
    
    print(f"Found instances: {instance_ids}")
    
    metrics = compute_metrics(lifecycles, bin_edges, start_time)
    
    generate_and_save_charts(
        metrics, bin_centers, instance_ids, args.instances, args.directory,
        smooth_window=args.smooth_window,
        time_range=(args.start_time, args.end_time),
        prefill_request_ylim=args.prefill_request_ylim,
        prefill_throughput_ylim=args.prefill_throughput_ylim
    )
    
    print("Analysis complete! All charts have been saved.")

if __name__ == "__main__":
    main()
