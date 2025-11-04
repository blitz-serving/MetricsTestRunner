#!/usr/bin/env python3
"""
generate_load_with_time.py

This script analyzes router_v2.log files to generate timeline charts showing
various metrics across different vLLM instances. It processes log entries to
track request lifecycles and generates 5 different timeline charts.

Usage:
    python generate_load_with_time.py <directory_path>
"""

import argparse
import re
import os
import json
from datetime import datetime, timedelta
from collections import defaultdict, namedtuple
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Constants
TIME_BIN = 15  # time bin window, <10s will cause too much data point number;
LOG_FILENAME = "router_v2.log"
INSTANCE_LIST = [0, 7, 8, 15] # selected instance to plot

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
    return parser.parse_args()

def parse_timestamp(log_timestamp):
    """
    Parse ISO 8601 timestamp from log entries.
    
    Args:
        log_timestamp (str): Timestamp string from log (e.g., '2025-11-04T06:36:22.830882Z')
    
    Returns:
        datetime: Parsed datetime object
    """
    # Remove the 'Z' suffix and parse the timestamp
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
    
    # Enhanced regex patterns to handle the actual log format and the described format
    # Pattern 1: Request routing with input_length and decode_length (as described in requirements)
    route_pattern = re.compile(
        r'(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d+Z).*?'
        r'Request\((\d+)\).*?input_length (\d+) decode_length (\d+).*?added to vLLM#(\d+)'
    )
    
    # Pattern 2: Prefill done with hit_cnt (as described in requirements)
    prefill_pattern = re.compile(
        r'(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d+Z).*?'
        r'Vllm#(\d+)::Request#\((\d+)\) prefill done hit_cnt (\d+)!'
    )
    
    # Pattern 3: Decode done (as described in requirements)
    decode_pattern = re.compile(
        r'(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d+Z).*?'
        r'Vllm#(\d+)::Request\((\d+)\) is finished generating \d+ tokens'
    )
    
    with open(log_file_path, 'r') as file:
        for line in file:
            line = line.strip()
            if not line:
                continue
            
            # Try to match routing information (enhanced pattern first)
            route_match = route_pattern.search(line)
            
            if route_match:
                timestamp_str, req_id, instance_id = route_match.group(1), int(route_match.group(2)), int(route_match.group(5))
                timestamp = parse_timestamp(timestamp_str)
                
                # For fallback pattern, use default values
                input_length = int(route_match.group(3)) if len(route_match.groups()) > 3 else -1
                decode_length = int(route_match.group(4)) if len(route_match.groups()) > 4 else -1
                
                if input_length != -1 and decode_length != -1:
                    request_routes[req_id] = RequestRoute(
                        timestamp=timestamp,
                        request_id=req_id,
                        input_length=input_length,
                        decode_length=decode_length,
                        instance_id=instance_id
                    )
                    
                    # Update time bounds
                    if earliest_time is None or timestamp < earliest_time:
                        earliest_time = timestamp
                    if latest_time is None or timestamp > latest_time:
                        latest_time = timestamp
                    continue
                else:
                    print(f"warning: {req_id=} {instance_id=} {input_length=} {decode_length=}")
            
            # Try to match prefill completion (enhanced pattern first)
            prefill_match = prefill_pattern.search(line)
            
            if prefill_match:
                timestamp_str, instance_id, req_id = prefill_match.group(1), int(prefill_match.group(2)), int(prefill_match.group(3))
                timestamp = parse_timestamp(timestamp_str)
                
                # For fallback pattern, use default hit count
                hit_cnt = int(prefill_match.group(4)) if len(prefill_match.groups()) > 3 else 0
                
                prefill_dones[req_id] = PrefillDone(
                    timestamp=timestamp,
                    request_id=req_id,
                    hit_cnt=hit_cnt,
                    instance_id=instance_id
                )
                
                # Update time bounds
                if latest_time is None or timestamp > latest_time:
                    latest_time = timestamp
                continue
            
            # Try to match decode completion
            decode_match = decode_pattern.search(line)
          
            if decode_match:
                timestamp_str, instance_id, req_id = decode_match.group(1), int(decode_match.group(2)), int(decode_match.group(3))
                timestamp = parse_timestamp(timestamp_str)
                
                decode_dones[req_id] = DecodeDone(
                    timestamp=timestamp,
                    request_id=req_id,
                    instance_id=instance_id
                )
                
                # Update time bounds
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
    # Get all unique instance IDs
    instance_ids = sorted(set(lifecycle.instance_id for lifecycle in lifecycles))
    
    # Initialize metrics dictionary
    metrics = {
        'prefill_req_number': defaultdict(lambda: [0] * (len(bin_edges) - 1)),
        'decode_req_number': defaultdict(lambda: [0] * (len(bin_edges) - 1)),
        'total_prefill_length': defaultdict(lambda: [0] * (len(bin_edges) - 1)),
        'total_decode_length': defaultdict(lambda: [0] * (len(bin_edges) - 1)),
        'sum_input_length': defaultdict(lambda: [0] * (len(bin_edges) - 1)),
        'sum_cache_length': defaultdict(lambda: [0] * (len(bin_edges) - 1))
    }
    
    # Process each request lifecycle
    for lifecycle in lifecycles:
        instance_id = lifecycle.instance_id
        
        # Convert timestamps to relative seconds
        route_time_rel = (lifecycle.route_time - start_time).total_seconds()
        prefill_time_rel = (lifecycle.prefill_time - start_time).total_seconds()
        decode_time_rel = (lifecycle.decode_time - start_time).total_seconds()
        
        # Find bins that overlap with [route_time, prefill_time] for prefill phase
        prefill_start_bin = max(0, int(route_time_rel / TIME_BIN))
        prefill_end_bin = min(len(bin_edges) - 2, int(prefill_time_rel / TIME_BIN))
        
        for bin_idx in range(prefill_start_bin, prefill_end_bin + 1):
            bin_start = bin_idx * TIME_BIN
            bin_end = (bin_idx + 1) * TIME_BIN
            
            # Check if there's overlap between the bin and the prefill phase
            if bin_end > route_time_rel and bin_start < prefill_time_rel:
                metrics['prefill_req_number'][instance_id][bin_idx] += 1
                metrics['total_prefill_length'][instance_id][bin_idx] += lifecycle.input_length
                metrics['sum_input_length'][instance_id][bin_idx] += lifecycle.input_length
                metrics['sum_cache_length'][instance_id][bin_idx] += lifecycle.hit_cnt
        
        # Find bins that overlap with [prefill_time, decode_time] for decode phase
        decode_start_bin = max(0, int(prefill_time_rel / TIME_BIN))
        decode_end_bin = min(len(bin_edges) - 2, int(decode_time_rel / TIME_BIN))
        
        for bin_idx in range(decode_start_bin, decode_end_bin + 1):
            bin_start = bin_idx * TIME_BIN
            bin_end = (bin_idx + 1) * TIME_BIN
            
            # Check if there's overlap between the bin and the decode phase
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
    
    # Convert defaultdicts to regular dicts for easier handling
    for metric_name in metrics:
        metrics[metric_name] = dict(metrics[metric_name])
    
    return metrics

def generate_and_save_charts(metrics, bin_centers, instance_ids, output_dir):
    """
    Generate and save the 5 timeline charts.
    
    Args:
        metrics (dict): Computed metrics
        bin_centers (list): Time bin centers in seconds
        instance_ids (list): List of instance IDs (all available)
        output_dir (str): Directory to save the charts
    """
    chart_configs = [
        {
            'metric_name': 'prefill_req_number',
            'title': 'Prefill Request Count Over Time',
            'ylabel': 'Number of Requests in Prefill',
            'filename': 'prefill_req_number.png'
        },
        {
            'metric_name': 'decode_req_number',
            'title': 'Decode Request Count Over Time',
            'ylabel': 'Number of Requests in Decode',
            'filename': 'decode_req_number.png'
        },
        {
            'metric_name': 'total_prefill_length',
            'title': 'Total Prefill Length Over Time',
            'ylabel': 'Cumulative Input Length',
            'filename': 'total_prefill_length.png'
        },
        {
            'metric_name': 'total_decode_length',
            'title': 'Total Decode Length Over Time',
            'ylabel': 'Cumulative Decode Length',
            'filename': 'total_decode_length.png'
        },
        {
            'metric_name': 'prefix_cache_rate',
            'title': 'Prefix Cache Hit Rate Over Time',
            'ylabel': 'Cache Hit Rate',
            'filename': 'prefix_cache_rate.png'
        }
    ]
    
    # 🔹 仅绘制 INSTANCE_LIST 中存在的实例
    filtered_instance_ids = [inst for inst in INSTANCE_LIST if inst in instance_ids]
    if not filtered_instance_ids:
        print(f"Warning: No instances from INSTANCE_LIST {INSTANCE_LIST} found in data. No charts will be plotted.")
        return

    print(f"Selected instance ids = {filtered_instance_ids}")

    # Set up plot style
    plt.style.use('seaborn-v0_8')
    
    for config in chart_configs:
        plt.figure(figsize=(12, 6))
        
        metric_name = config['metric_name']
        if metric_name not in metrics:
            print(f"Warning: Metric '{metric_name}' not found in computed metrics")
            continue
        
        for instance_id in filtered_instance_ids:
            if instance_id in metrics[metric_name]:
                values = metrics[metric_name][instance_id]
                plt.plot(bin_centers, values, 
                        label=f'Instance #{instance_id}', 
                        linewidth=2,
                        marker='o', 
                        markersize=4,
                        alpha=0.8)
        
        plt.title(config['title'], fontsize=14, fontweight='bold')
        plt.xlabel('Time (seconds)', fontsize=12)
        plt.ylabel(config['ylabel'], fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.legend(title='vLLM Instance', bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        
        # Save the figure as PNG
        output_path = os.path.join(output_dir, config['filename'])
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Saved chart: {output_path}")

def main():
    """Main function to orchestrate the analysis."""
    args = parse_arguments()
    
    # Validate directory exists
    if not os.path.isdir(args.directory):
        raise ValueError(f"Directory not found: {args.directory}")
    
    # Construct log file path
    log_file_path = os.path.join(args.directory, LOG_FILENAME)
    if not os.path.exists(log_file_path):
        raise FileNotFoundError(f"Log file not found: {log_file_path}")
    
    print(f"Processing log file: {log_file_path}")
    
    # Extract log data
    request_routes, prefill_dones, decode_dones, start_time, end_time = extract_log_data(log_file_path)
    
    print(f"Found {len(request_routes)} route events, {len(prefill_dones)} prefill events, {len(decode_dones)} decode events")
    
    print(f"{request_routes[1]} {prefill_dones[1]} {decode_dones[1]}")
    # Build request lifecycles
    lifecycles = build_request_lifecycles(request_routes, prefill_dones, decode_dones)
    
    print(f"Built {len(lifecycles)} complete request lifecycles")
    
    if not lifecycles:
        print("No complete request lifecycles found. Cannot generate charts.")
        return
    
    # Calculate time bins
    bin_edges, bin_centers, total_duration = calculate_time_bins(start_time, end_time)
    
    print(f"Time range: {total_duration:.2f} seconds, Number of bins: {len(bin_edges)-1}")
    
    # Get all unique instance IDs
    instance_ids = sorted(set(lifecycle.instance_id for lifecycle in lifecycles))
    
    print(f"Found instances: {instance_ids}")
    
    # Compute metrics
    metrics = compute_metrics(lifecycles, bin_edges, start_time)
    
    # Generate and save charts
    generate_and_save_charts(metrics, bin_centers, instance_ids, args.directory)
    
    print("Analysis complete! All charts have been saved.")

if __name__ == "__main__":
    main()