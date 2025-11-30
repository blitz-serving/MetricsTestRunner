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


ZOOM_IN = False

def load_jsonl_data(file_path: str) -> Tuple[List[Dict], bool]:
    """
    Load data from a JSONL file, filtering out records with status != '200'.
    
    Args:
        file_path: Path to the JSONL file
        
    Returns:
        List of dictionaries containing the JSON data (filtered)
    """
    data = []
    err_rate = 0
    ok_number = 0
    try:
        with open(file_path, 'r') as f:
            line_num = 0
            skip_line_num = 0
            for line in f:
                line_num += 1
                if line.strip():  # Skip empty lines
                    try:
                        record = json.loads(line.strip())
                        # Filter out records where status is not '200'
                        status = record.get('status', '')
                        if status != '422': # only ignore these
                            if status == '200':
                                ok_number += 1
                            data.append(record)
                        else:
                            skip_line_num += 1
                    except json.JSONDecodeError as e:
                        print(f"Warning: Skipping invalid JSON on line {line_num}: {e}")
                        continue
            print(f"Warning: skipped {skip_line_num=} requests (422) lasting {len(data)}")
            print(f"Warning: Success Rate {ok_number / len(data)} (200) / (not 422s)")
            err_rate = 1.0 - ok_number / len(data)
    except FileNotFoundError:
        print(f"Error: File {file_path} not found")
        sys.exit(1)
    
    return data, err_rate


def extract_metrics(data: List[Dict]) -> Dict[str, np.ndarray]:
    """
    Extract relevant metrics from the loaded data.
    
    Args:
        data: List of dictionaries containing the JSON data
        
    Returns:
        Dictionary with arrays of extracted metrics
    """
    metrics = {
        'rid': [],
        'ttft': [],
        'tpot': [],
        'inference_time': [],
        'total_time': []
    }
    err_cnt = 0
    for item in data:
        # Extract metrics, converting strings to numbers where needed
        status = item.get("status")
        rid = str(item.get('request_id', ""))
        if rid != "":
            metrics['rid'].append(rid)
        else:
            metrics['rid'].append(f"err_{err_cnt}")
            err_cnt += 1
        if status == '200':
            ttft = float(item.get('first_token_time', 0))
            tpot = float(item.get('avg_time_between_tokens', 0))
            inference_time = float(item.get('inference_time', 0))
            total_time = float(item.get('total_time', 0))
            metrics['ttft'].append(ttft)
            metrics['tpot'].append(tpot)
            metrics['inference_time'].append(inference_time)
            metrics['total_time'].append(total_time)
        else:
            metrics['ttft'].append(-1)
            metrics['tpot'].append(-1)
            metrics['inference_time'].append(-1)
            metrics['total_time'].append(-1)


    # Convert lists to numpy arrays for easier calculations
    for key in metrics:
        if key != "rid":
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


def plot_comparative_cdf_zoom_in(all_metrics: Dict[str, np.ndarray], 
                                strategy_names: List[str], 
                                title: str, 
                                xlabel: str, 
                                output_files: List[str]):
    """
    Generate a comparative CDF plot with explicit color mapping per strategy.
    Colors are assigned based on strategy name (not list index) for robustness.
    """
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # === 为每个 strategy name 分配唯一颜色（顺序固定）===
    unique_strategies = sorted(set(strategy_names))  # 确保顺序确定（可选：也可保持原顺序但去重）
    # 如果你希望保持输入顺序（不去重排序），可改用：
    # unique_strategies = []
    # for s in strategy_names:
    #     if s not in unique_strategies:
    #         unique_strategies.append(s)

    n = len(unique_strategies)
    tab20_colors = plt.get_cmap('tab20').colors
    tab20b_colors = plt.get_cmap('tab20b').colors
    tab20c_colors = plt.get_cmap('tab20c').colors
    all_tab_colors = list(tab20_colors) + list(tab20b_colors) + list(tab20c_colors)  # 60 colors

    if n <= len(all_tab_colors):
        selected_colors = all_tab_colors[:n]
    else:
        print(f"Warning: {n} unique strategies exceed 60. Falling back to 'hsv' colormap.")
        cmap = plt.get_cmap('hsv')
        selected_colors = [cmap(i / n) for i in range(n)]
    
    # 创建 strategy -> color 的映射字典
    strategy_colors = {strategy: selected_colors[i] for i, strategy in enumerate(unique_strategies)}
    
    # Plot main CDF curves using the color map
    for strategy in strategy_names:  # 保持调用方期望的绘制顺序（可能有重复？通常不应有）
        if strategy not in all_metrics:
            continue
        color = strategy_colors[strategy]
        data = all_metrics[strategy]
        sorted_data = np.sort(data)
        yvals = np.arange(1, len(sorted_data) + 1) / len(sorted_data)
        ax.plot(sorted_data, yvals, linewidth=1.5, label=strategy, color=color)

    # Compute and plot P99 lines
    p99_values = {}
    for strategy in strategy_names:
        if strategy not in all_metrics:
            continue
        data = all_metrics[strategy]
        p99 = np.percentile(data, 99)
        p99_values[strategy] = p99
        ax.axvline(p99, color=strategy_colors[strategy], linestyle='--', linewidth=1.2, alpha=0.7,
                   label=f"{strategy} P99")

    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel('Cumulative Probability')
    ax.grid(True, alpha=0.3)
    
    # Legend handling
    if len(strategy_names) > 10:
        ax.legend(fontsize='x-small', ncol=2 if len(strategy_names) <= 20 else 3, loc='lower right')
    else:
        ax.legend(fontsize='small')

    # Set x-axis limit
    max_val = max(np.max(all_metrics[strategy]) for strategy in strategy_names)
    ax.set_xlim(0, max_val * 1.05)

    # --- Inset zoom-in ---
    if ZOOM_IN:
        from mpl_toolkits.axes_grid1.inset_locator import inset_axes
        axins = inset_axes(
            ax,
            width="100%",  
            height="100%",  
            bbox_to_anchor=(0.57, 0.4, 0.4, 0.3), 
            bbox_transform=ax.transAxes,
            loc='lower left'
        )

        for strategy in strategy_names:
            if strategy not in all_metrics:
                continue
            color = strategy_colors[strategy]
            data = all_metrics[strategy]
            sorted_data = np.sort(data)
            yvals = np.arange(1, len(sorted_data) + 1) / len(sorted_data)
            mask = (yvals >= 0.8) & (yvals <= 1.0)
            if np.any(mask):
                axins.plot(sorted_data[mask], yvals[mask], linewidth=1.5, color=color)

            p99 = p99_values[strategy]
            axins.axvline(p99, color=color, linestyle='--', linewidth=1, alpha=0.7)

        axins.set_ylim(0.8, 1.0)
        axins.set_xlim(ax.get_xlim())
        axins.grid(True, alpha=0.3)
        axins.set_facecolor('white')
        axins.tick_params(labelsize=8)
        for spine in axins.spines.values():
            spine.set_edgecolor('gray')
            spine.set_linewidth(0.8)

    # Save
    for output_file in output_files:
        fig.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"  Comparative CDF plot with P99 lines saved to {output_file}")
    
    plt.close(fig)

def main():
    """Main function to process data and generate comparative plots."""
    parser = argparse.ArgumentParser(description='Generate comparative CDF plots for LLM metrics across different strategies')
    parser.add_argument('dirs', nargs='+', help='Directories containing client.jsonl files (each directory should be in format xxx_yyy where yyy is the strategy name)')
    parser.add_argument('--label', type=str, default=None, help='Optional label to append to all plot titles as suffix {label}')
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
    
    report_errs = []
    for i, dir_path in enumerate(args.dirs):
        # Construct path to client.jsonl file
        jsonl_path = os.path.join(dir_path, 'client.jsonl')
        
        # Load and process data
        print(f"\nProcessing data from {jsonl_path}")
        data, err_rate = load_jsonl_data(jsonl_path)
        if (err_rate > 0.05):
            report_errs.append({strategy_names[i]:err_rate})
        all_data[strategy_names[i]] = data
        metrics = extract_metrics(data)
        all_metrics[strategy_names[i]] = metrics
    

    # Count data points per strategy and identify outliers with significantly fewer samples
    data_counts = {strategy: len(data) for strategy, data in all_data.items()}
    max_count = max(data_counts.values()) if data_counts else 0

    strategies_to_remove = []
    for strategy, count in data_counts.items():
        if max_count > 0 and (max_count - count) / max_count > 0.01:  # more than 1% fewer
            strategies_to_remove.append(strategy)
            report_errs.append({strategy: f"insufficient_data_points ({count}/{max_count})"})

    # Remove strategies with too few data points
    for strategy in strategies_to_remove:
        del all_data[strategy]
        del all_metrics[strategy]
        # Also remove from strategy_names if you use it later
        if strategy in strategy_names:
            strategy_names.remove(strategy)


    def red(text):
        return f"\033[91m{text}\033[0m"

    if strategies_to_remove:
        print(red(f"\nWarning: Removed strategies due to insufficient data points: {strategies_to_remove}"))
    
    # Correction function for timeout handling
    print("\nApplying timeout correction...")
    
    # Get maximum values across all policies
    max_values = {
        'ttft': 0,
        'tpot': 0,
        'inference_time': 0,
        'total_time': 0
    }
    
    for strategy in strategy_names:
        for metric in ['ttft', 'tpot', 'inference_time', 'total_time']:
            # Only consider non-negative values (ignore -1 timeout markers)
            valid_values = [v for v in all_metrics[strategy][metric] if v >= 0]
            if valid_values:
                max_values[metric] = max(max_values[metric], max(valid_values))
    
    # Apply correction to each strategy
    for strategy in strategy_names:
        metrics = all_metrics[strategy]
        all_rids = list(all_metrics[strategy]['rid'])
        corrected_metrics = {
            'rid': all_rids,
            'ttft': [],
            'tpot': [],
            'inference_time': [],
            'total_time': []
        }
    
        print(f"\n all requests numbers {len(all_rids)} for strategy {strategy}")
        
        success_rate = 0
        # Fill in corrected values
        for id, rid in enumerate(all_rids):
            if rid.startswith("err"): # time out or other err codes
                for metric in ['ttft', 'tpot', 'inference_time', 'total_time']:
                    assert all_metrics[strategy][metric][id] == -1
                    corrected_metrics[metric].append(max_values[metric])
            else: # ok
                success_rate += 1
                for metric in ['ttft', 'tpot', 'inference_time', 'total_time']:
                    assert all_metrics[strategy][metric][id] != -1
                    corrected_metrics[metric].append(all_metrics[strategy][metric][id])
        
        print(f"Valid data point for {strategy} is {success_rate}, Rate = {success_rate / len(list(all_metrics[strategy]['rid']))}")

        # Convert lists to numpy arrays
        for key in ['ttft', 'tpot', 'inference_time', 'total_time']:
            corrected_metrics[key] = np.array(corrected_metrics[key])
        
        # Replace original metrics with corrected ones
        all_metrics[strategy] = corrected_metrics
        
        # Assert all strategies have same number of data points
        assert len(corrected_metrics['rid']) == len(all_rids), f"Strategy {strategy} has incorrect number of data points"
        for metric in ['ttft', 'tpot', 'inference_time', 'total_time']:
            assert len(corrected_metrics[metric]) == len(all_rids), f"Strategy {strategy} has incorrect number of {metric} data points"
    
    print(f"Correction applied. All strategies now have {len(all_rids)} data points.")

    # Print statistics for each metric (moved outside the loop since we're using corrected metrics)
    metric_info = {
        'ttft': {'title': 'Time To First Token (TTFT)', 'xlabel': 'TTFT (ms)'},
        'tpot': {'title': 'Time Per Output Token (TPOT)', 'xlabel': 'TPOT (ms)'},
        'inference_time': {'title': 'Inference Time', 'xlabel': 'Inference Time (ms)'},
        'total_time': {'title': 'Total Time', 'xlabel': 'Total Time (ms)'}
    }
    
    for i, strategy in enumerate(strategy_names):
        print(f"\nStatistics for {strategy}:")
        for metric_name, info in metric_info.items():
            stats = calculate_statistics(all_metrics[strategy][metric_name])
            print(f"  {metric_name.upper()}:")
            print(f"    Mean: {stats['mean']:.2f}")
            print(f"    P50: {stats['p50']:.2f}")
            print(f"    P95: {stats['p95']:.2f}")
            print(f"    P99: {stats['p99']:.2f}")

    for dir_path in args.dirs:
        perf_log_path = os.path.join(dir_path, 'perf.log')
        with open(perf_log_path, "w") as file:
            for i, strategy in enumerate(strategy_names):
                file.write(f"\nStatistics for {strategy}:\n")
                for metric_name, info in metric_info.items():
                    stats = calculate_statistics(all_metrics[strategy][metric_name])
                    file.write(f"  {metric_name.upper()}:\n")
                    file.write(f"    Mean: {stats['mean']:.2f}\n")
                    file.write(f"    P50: {stats['p50']:.2f}\n")
                    file.write(f"    P95: {stats['p95']:.2f}\n")
                    file.write(f"    P99: {stats['p99']:.2f}\n")
    
    # ==============================
    # Calculate TTFT, TPOT, Total Time  Top3 and print
    # ==============================
    metrics_to_rank = ['ttft', 'tpot', 'total_time']
    ranking_results = {}

    for metric in metrics_to_rank:
        strategy_stats = []
        for strategy in strategy_names:
            data = all_metrics[strategy][metric]
            mean_val = np.mean(data)
            p99_val = np.percentile(data, 99)
            strategy_stats.append((strategy, mean_val, p99_val))
        
        # Sort by mean (ascending: lower is better)
        strategy_stats.sort(key=lambda x: x[1])
        top3_mean = strategy_stats[:3]

        # Sort by p99 (ascending)
        strategy_stats_p99 = sorted(strategy_stats, key=lambda x: x[2])
        top3_p99 = strategy_stats_p99[:3]

        ranking_results[metric] = {
            'by_mean': top3_mean,
            'by_p99': top3_p99
        }

    # Output to console and top3.log in each directory
    console_output = []
    console_output.append("\n=== Top-3 Strategies (Lower is Better) ===")
    for metric in metrics_to_rank:
        display_name = metric.replace('_', ' ').upper()
        console_output.append(f"\nTop-3 for {display_name} (by Mean):")
        for i, (name, mean_val, _) in enumerate(ranking_results[metric]['by_mean'], 1):
            console_output.append(f"  {i}. {name}: Mean = {mean_val:.2f}")

        console_output.append(f"\nTop-3 for {display_name} (by P99):")
        for i, (name, _, p99_val) in enumerate(ranking_results[metric]['by_p99'], 1):
            console_output.append(f"  {i}. {name}: P99 = {p99_val:.2f}")

    # Print to console
    for line in console_output:
        print(line)

    # Write to top3.log in each directory
    for dir_path in args.dirs:
        top3_log_path = os.path.join(dir_path, 'top3.log')
        with open(top3_log_path, 'w') as f:
            for line in console_output:
                f.write(line + '\n')
        print(f"  Top-3 ranking saved to {top3_log_path}")

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
        
        # Append label to title if provided
        plot_title = info['title']
        if args.label:
            plot_title += f" {args.label}"

        # Generate and save the comparative plot
        #plot_comparative_cdf(metric_data, strategy_names, plot_title, info['xlabel'], output_files)
        plot_comparative_cdf_zoom_in(metric_data, strategy_names, plot_title, info['xlabel'], output_files)

    print(red(f"warning: this errcode is too high! should ignore these {report_errs}"))

if __name__ == "__main__":
    main()
