#!/usr/bin/env python3
"""
Script to draw send gap vs s_time from client.jsonl files.

This script reads a client.jsonl file from a specified directory and plots
send_gap (ms) against s_time (ms) with data aggregated over T-second intervals.
"""

import argparse
import json
import os
from typing import List, Tuple, Dict
import matplotlib.pyplot as plt


def read_client_jsonl(file_path: str) -> List[Dict]:
    """
    Read and parse the client.jsonl file.
    
    Args:
        file_path: Path to the client.jsonl file
        
    Returns:
        List of parsed JSON objects
    """
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    data.append(json.loads(line))
                except json.JSONDecodeError as e:
                    print(f"Warning: Skipping invalid JSON line: {e}")
    return data


def aggregate_data(data: List[Dict], time_window: float) -> List[Tuple[float, float]]:
    """
    Aggregate data points over time windows and calculate average send_gap.
    
    Args:
        data: List of parsed JSON objects containing timing metrics
        time_window: Time window in seconds to aggregate over
        
    Returns:
        List of tuples (s_time, avg_send_gap) for plotting
    """
    # Convert s_time and send_gap to float and filter out invalid values
    valid_data = []
    for entry in data:
        try:
            s_time = float(entry.get('s_time', 0))
            send_gap = float(entry.get('s_time_drift', 0))
            if send_gap != 0:  # Only consider non-zero send_gap values
                valid_data.append((s_time, send_gap))
        except (ValueError, TypeError):
            continue
    
    if not valid_data:
        return []
    
    # Sort by s_time
    valid_data.sort(key=lambda x: x[0])
    
    # Aggregate data over time windows
    aggregated = []
    current_window_start = valid_data[0][0]
    window_send_gaps = []
    
    for s_time, send_gap in valid_data:
        # If current point is within the time window
        if s_time - current_window_start <= time_window * 1000:  # Convert seconds to ms
            window_send_gaps.append(send_gap)
        else:
            # Calculate average for current window and start new window
            if window_send_gaps:
                avg_send_gap = sum(window_send_gaps) / len(window_send_gaps)
                # Use the midpoint of the window as the x-coordinate
                window_midpoint = current_window_start + (time_window * 1000) / 2
                aggregated.append((window_midpoint, avg_send_gap))
            
            # Start new window
            current_window_start = s_time
            window_send_gaps = [send_gap]
    
    # Handle the last window
    if window_send_gaps:
        avg_send_gap = sum(window_send_gaps) / len(window_send_gaps)
        window_midpoint = current_window_start + (time_window * 1000) / 2
        aggregated.append((window_midpoint, avg_send_gap))
    
    return aggregated


def plot_send_gap_vs_stime(aggregated_data: List[Tuple[float, float]], output_path: str = None):
    """
    Plot send_gap vs s_time.
    
    Args:
        aggregated_data: List of tuples (s_time, avg_send_gap)
        output_path: Optional path to save the plot
    """
    if not aggregated_data:
        print("No data to plot.")
        return
    
    s_times, send_gaps = zip(*aggregated_data)
    
    # Convert s_time from milliseconds to seconds
    s_times_seconds = [t / 1000.0 for t in s_times]
    
    plt.figure(figsize=(12, 6))
    plt.plot(s_times_seconds, send_gaps, 'b-', linewidth=1, alpha=0.7)
    plt.xlabel('s_time (s)')
    plt.ylabel('send_gap (ms)')
    plt.title('Send Gap vs Start Time')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {output_path}")
    else:
        plt.show()


def main():
    """
    Main function to parse arguments and run the script.
    """
    parser = argparse.ArgumentParser(
        description='Draw send gap vs s_time from client.jsonl files'
    )
    parser.add_argument(
        'dir',
        type=str,
        help='Directory containing client.jsonl file'
    )
    parser.add_argument(
        '--time-window',
        type=float,
        default=1.0,
        help='Time window in seconds for data aggregation (default: 1.0)'
    )
    parser.add_argument(
        '--output',
        type=str,
        help='Output file path for the plot (optional)'
    )
    
    args = parser.parse_args()
    
    # Check if the directory exists
    if not os.path.isdir(args.dir):
        print(f"Error: Directory {args.dir} does not exist.")
        return
    
    # Check if client.jsonl exists in the directory
    client_jsonl_path = os.path.join(args.dir, 'client.jsonl')
    if not os.path.isfile(client_jsonl_path):
        print(f"Error: {client_jsonl_path} does not exist.")
        return
    
    # Read the client.jsonl file
    print(f"Reading {client_jsonl_path}...")
    data = read_client_jsonl(client_jsonl_path)
    print(f"Read {len(data)} entries from the file.")
    
    # Aggregate the data
    print(f"Aggregating data with time window of {args.time_window} seconds...")
    aggregated_data = aggregate_data(data, args.time_window)
    print(f"Aggregated to {len(aggregated_data)} data points.")
    
    # Plot the data
    output_path = args.output
    if not output_path:
        # Generate default output path
        dir_name = os.path.basename(os.path.normpath(args.dir))
        output_path = os.path.join(args.dir, f"send_gap_vs_stime_{dir_name}.png")
    
    plot_send_gap_vs_stime(aggregated_data, output_path)


if __name__ == '__main__':
    main()
