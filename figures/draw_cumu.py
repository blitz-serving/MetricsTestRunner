#!/usr/bin/env python3
"""
Script to draw cumulative alive requests over time from client.jsonl logs.
"""

import sys
import json
import os
from typing import List, Dict, Tuple

# Set matplotlib to use Agg backend for headless environments
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def read_client_logs(log_dir: str) -> List[Dict]:
    """
    Read and parse client.jsonl file from the given directory.
    Returns a list of valid log entries (status == "200").
    """
    log_file = os.path.join(log_dir, "client.jsonl")
    valid_entries = []
    
    try:
        with open(log_file, 'r') as f:
            for line in f:
                try:
                    entry = json.loads(line.strip())
                    if entry.get("status") == "200":
                        # Convert time strings to integers
                        entry["s_time"] = int(entry["s_time"])
                        entry["e_time"] = int(entry["e_time"])
                        valid_entries.append(entry)
                except (json.JSONDecodeError, KeyError, ValueError):
                    # Skip malformed or incomplete entries
                    continue
    except FileNotFoundError:
        print(f"Error: {log_file} not found.")
        sys.exit(1)
    except Exception as e:
        print(f"Error reading log file: {e}")
        sys.exit(1)
        
    return valid_entries


def calculate_alive_requests(entries: List[Dict], bin_size_ms: int = 200) -> Tuple[List[int], List[int]]:
    """
    Calculate the number of alive requests over time, aggregated by time bins.
    Returns tuple of (time_points, alive_counts).
    """
    if not entries:
        return [], []
    
    # Find the overall time range
    min_time = min(entry["s_time"] for entry in entries)
    max_time = max(entry["e_time"] for entry in entries)
    
    # Create time bins
    bin_start = min_time
    time_points = []
    alive_counts = []
    
    while bin_start <= max_time:
        bin_end = bin_start + bin_size_ms
        time_points.append(bin_start)
        
        # Count requests alive during this bin
        count = 0
        for entry in entries:
            # Request is alive if it started before bin_end and ended after bin_start
            if entry["s_time"] < bin_end and entry["e_time"] > bin_start:
                count += 1
                
        alive_counts.append(count)
        bin_start = bin_end
    
    return time_points, alive_counts


def plot_alive_requests(time_points: List[int], alive_counts: List[int], output_file: str = None):
    """
    Plot the number of alive requests over time.
    """
    plt.figure(figsize=(12, 6))
    # Convert time points from ms to s for better readability
    time_points_seconds = [t / 1000.0 for t in time_points]
    plt.plot(time_points_seconds, alive_counts, linewidth=2, marker='o', markersize=3)
    plt.title('Number of Alive Requests Over Time')
    plt.xlabel('Time (s)')
    plt.ylabel('Number of Alive Requests')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {output_file}")
    else:
        plt.show()


def main():
    """Main function to process logs and generate plot."""
    if len(sys.argv) != 2:
        print("Usage: python draw_cumu.py <log_directory>")
        sys.exit(1)
    
    log_dir = sys.argv[1]
    
    # Read and filter log entries
    print(f"Reading logs from {log_dir}...")
    entries = read_client_logs(log_dir)
    print(f"Found {len(entries)} valid entries (status 200).")
    
    if not entries:
        print("No valid entries found. Exiting.")
        sys.exit(0)
    
    # Calculate alive requests over time
    print("Calculating alive requests over time...")
    time_points, alive_counts = calculate_alive_requests(entries)
    
    # Generate plot
    print("Generating plot...")
    output_file = os.path.join(log_dir, "req_num_cumu.png")
    plot_alive_requests(time_points, alive_counts, output_file)
    
    print("Done!")


if __name__ == "__main__":
    main()
