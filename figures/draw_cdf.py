#!/usr/bin/env python3
"""
Script to generate CDF plots and statistics for LLM performance metrics.

This script reads client.jsonl files and generates CDF plots for:
- TTFT (Time To First Token)
- TPOT (Time Per Output Token) 
- Inference Time
- Total Time

It also prints summary statistics (mean, p50, p95, p99) for each metric.
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
    Load data from a JSONL file.
    
    Args:
        file_path: Path to the JSONL file
        
    Returns:
        List of dictionaries containing the JSON data
    """
    data = []
    try:
        with open(file_path, 'r') as f:
            line_num = 0
            for line in f:
                line_num += 1
                if line.strip():  # Skip empty lines
                    try:
                        data.append(json.loads(line.strip()))
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


def plot_cdf(data: np.ndarray, title: str, xlabel: str, output_file: str):
    """
    Generate a CDF plot for the given data.
    
    Args:
        data: Numpy array of numeric data
        title: Title for the plot
        xlabel: Label for x-axis
        output_file: File path to save the plot
    """
    # Sort data for CDF
    sorted_data = np.sort(data)
    yvals = np.arange(1, len(sorted_data) + 1) / len(sorted_data)
    
    # Create plot
    plt.figure(figsize=(10, 6))
    plt.plot(sorted_data, yvals, linewidth=2)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel('Cumulative Probability')
    plt.grid(True, alpha=0.3)
    plt.xlim(0, max(sorted_data) * 1.05)
    
    # Save plot
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()


def main():
    """Main function to process data and generate plots."""
    parser = argparse.ArgumentParser(description='Generate CDF plots for LLM metrics')
    parser.add_argument('dir', help='Directory containing client.jsonl file')
    args = parser.parse_args()
    
    # Construct path to client.jsonl file
    jsonl_path = os.path.join(args.dir, 'client.jsonl')
    
    # Load and process data
    print(f"Loading data from {jsonl_path}")
    data = load_jsonl_data(jsonl_path)
    metrics = extract_metrics(data)
    
    # Define metric information for plotting
    metric_info = {
        'ttft': {'title': 'CDF of Time To First Token (TTFT)', 'xlabel': 'TTFT (ms)'},
        'tpot': {'title': 'CDF of Time Per Output Token (TPOT)', 'xlabel': 'TPOT (ms)'},
        'inference_time': {'title': 'CDF of Inference Time', 'xlabel': 'Inference Time (ms)'},
        'total_time': {'title': 'CDF of Total Time', 'xlabel': 'Total Time (ms)'}
    }
    
    # Process each metric
    for metric_name, info in metric_info.items():
        # Calculate and print statistics
        stats = calculate_statistics(metrics[metric_name])
        print(f"\n{metric_name.upper()} Statistics:")
        print(f"  Mean: {stats['mean']:.2f}")
        print(f"  P50: {stats['p50']:.2f}")
        print(f"  P95: {stats['p95']:.2f}")
        print(f"  P99: {stats['p99']:.2f}")
        
        # Generate CDF plot
        output_file = os.path.join(args.dir, f"{metric_name}_cdf.png")
        plot_cdf(metrics[metric_name], info['title'], info['xlabel'], output_file)
        print(f"  CDF plot saved to {output_file}")


if __name__ == "__main__":
    main()
