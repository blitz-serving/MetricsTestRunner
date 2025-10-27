#!/usr/bin/env python3
"""
Script to generate comparative CDF plots for LLM performance metrics across different strategies.

This script reads client.jsonl files from multiple directories and generates CDF plots for:
- TTFT (Time To First Token)
- TPOT (Time Per Output Token)
- Inference Time
- Total Time

Each plot will show multiple lines, one for each strategy, allowing for direct comparison.
"""

import argparse
import json
import os
import sys
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np


def load_jsonl_data(file_path: str) -> List[Dict]:
    """
    Load data from a JSONL file, filtering out records with status != '200'.
    
    Args:
        file_path: Path to the JSONL file
        
    Returns:
        List of dictionaries containing the JSON data (filtered)
    """
    data = []
    try:
        with open(file_path, 'r') as f:
            line_num = 0
            for line in f:
                line_num += 1
                if line.strip():  # Skip empty lines
                    try:
                        record = json.loads(line.strip())
                        # Filter out records where status is not '200'
                        if record.get('status', '') == '200':
                            data.append(record)
                        else:
                            print(f"Warning: Skipping record on line {line_num} with status {record.get('status', 'unknown')}")
                    except json.JSONDecodeError as e:
                        print(f"Warning: Skipping invalid JSON on line {line_num}: {e}")
                        continue
    except FileNotFoundError:
        print(f"Error: File {file_path} not found")
        sys.exit(1)
    
    return data


def extract_metrics(data: List[Dict]) -> Dict[str, np.ndarray]:
    """
    Extract relevant metrics from the loaded data.
    
    Args:
        data: List of dictionaries containing the JSON data
        
    Returns:
        Dictionary with arrays of extracted metrics
    """
    metrics = {
        'ttft': [],
        'tpot': [],
        'inference_time': [],
        'total_time': []
    }
    
    for item in data:
        # Extract metrics, converting strings to numbers where needed
        metrics['ttft'].append(float(item.get('first_token_time', 0)))
        metrics['tpot'].append(float(item.get('avg_time_between_tokens', 0)))
        metrics['inference_time'].append(float(item.get('inference_time', 0)))
        metrics['total_time'].append(float(item.get('total_time', 0)))
    
    # Convert lists to numpy arrays for easier calculations
    for key in metrics:
        metrics[key] = np.array(metrics[key])
    
    return metrics


def calculate_statistics(data: np.ndarray) -> Dict[str, float]:
    """
    Calculate summary statistics for a dataset.
    
    Args:
        data: Numpy array of numeric data
        
    Returns:
        Dictionary with mean, p50, p95, p99 statistics
    """
    return {
        'mean': np.mean(data),
        'p50': np.percentile(data, 50),
        'p95': np.percentile(data, 95),
        'p99': np.percentile(data, 99)
    }


def plot_comparative_cdf(all_metrics: Dict[str, Dict[str, np.ndarray]], 
                        strategy_names: List[str], 
                        title: str, 
                        xlabel: str, 
                        output_files: List[str]):
    """
    Generate a comparative CDF plot for the given data across multiple strategies.
    
    Args:
        all_metrics: Dictionary mapping strategy names to their metrics
        strategy_names: List of strategy names
        title: Title for the plot
        xlabel: Label for x-axis
        output_files: List of file paths to save the plot (one for each strategy directory)
    """
    plt.figure(figsize=(12, 8))
    
    # Plot CDF for each strategy
    for strategy in strategy_names:
        metrics = all_metrics[strategy]
        sorted_data = np.sort(metrics)
        yvals = np.arange(1, len(sorted_data) + 1) / len(sorted_data)
        plt.plot(sorted_data, yvals, linewidth=2, label=strategy)
    
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel('Cumulative Probability')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # Set x-axis limit based on maximum value across all strategies
    max_val = max([np.max(all_metrics[strategy]) for strategy in strategy_names])
    plt.xlim(0, max_val * 1.05)
    
    # Save plot to each specified output file
    for output_file in output_files:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"  Comparative CDF plot saved to {output_file}")
    
    plt.close()


def main():
    """Main function to process data and generate comparative plots."""
    parser = argparse.ArgumentParser(description='Generate comparative CDF plots for LLM metrics across different strategies')
    parser.add_argument('dirs', nargs='+', help='Directories containing client.jsonl files (each directory should be in format xxx_yyy where yyy is the strategy name)')
    args = parser.parse_args()
    
    # Extract strategy names from directory paths
    strategy_names = []
    for dir_path in args.dirs:
        dir_name = os.path.basename(dir_path.rstrip('/'))
        if '_' in dir_name:
            strategy = dir_name.split('_')[-1]  # Get the part after the last underscore
        else:
            strategy = dir_name  # Use full directory name if no underscore
        strategy_names.append(strategy)
    
    # Load and process data for each directory
    print("Loading and processing data from specified directories...")
    all_data = {}
    all_metrics = {}
    
    for i, dir_path in enumerate(args.dirs):
        # Construct path to client.jsonl file
        jsonl_path = os.path.join(dir_path, 'client.jsonl')
        
        # Load and process data
        print(f"\nProcessing data from {jsonl_path}")
        data = load_jsonl_data(jsonl_path)
        all_data[strategy_names[i]] = data
        metrics = extract_metrics(data)
        all_metrics[strategy_names[i]] = metrics
        
        # Print statistics for each metric
        metric_info = {
            'ttft': {'title': 'Time To First Token (TTFT)', 'xlabel': 'TTFT (ms)'},
            'tpot': {'title': 'Time Per Output Token (TPOT)', 'xlabel': 'TPOT (ms)'},
            'inference_time': {'title': 'Inference Time', 'xlabel': 'Inference Time (ms)'},
            'total_time': {'title': 'Total Time', 'xlabel': 'Total Time (ms)'}
        }
        
        for metric_name, info in metric_info.items():
            stats = calculate_statistics(metrics[metric_name])
            print(f"\n{strategy_names[i]} - {metric_name.upper()} Statistics:")
            print(f"  Mean: {stats['mean']:.2f}")
            print(f"  P50: {stats['p50']:.2f}")
            print(f"  P95: {stats['p95']:.2f}")
            print(f"  P99: {stats['p99']:.2f}")
    
    # Define metric information for plotting
    metric_info = {
        'ttft': {'title': 'Comparative CDF of Time To First Token (TTFT)', 'xlabel': 'TTFT (ms)'},
        'tpot': {'title': 'Comparative CDF of Time Per Output Token (TPOT)', 'xlabel': 'TPOT (ms)'},
        'inference_time': {'title': 'Comparative CDF of Inference Time', 'xlabel': 'Inference Time (ms)'},
        'total_time': {'title': 'Comparative CDF of Total Time', 'xlabel': 'Total Time (ms)'}
    }
    
    # Generate comparative CDF plots for each metric
    for metric_name, info in metric_info.items():
        print(f"\nGenerating comparative CDF plot for {metric_name}...")
        
        # Prepare data for this metric across all strategies
        metric_data = {strategy: all_metrics[strategy][metric_name] for strategy in strategy_names}
        
        # Generate output file paths (save to each directory)
        output_files = [os.path.join(args.dirs[i], f"{metric_name}_comparative_cdf.png") for i in range(len(args.dirs))]
        
        # Generate and save the comparative plot
        plot_comparative_cdf(metric_data, strategy_names, info['title'], info['xlabel'], output_files)


if __name__ == "__main__":
    main()
