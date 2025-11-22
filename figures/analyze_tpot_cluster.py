#!/usr/bin/env python3
"""
generate_ttft_tpot_cdf.py

Generates two CDF charts from router_v2.log:
  1. TTFT (Time To First Token) CDF per vLLM instance
  2. TPOT (Time Per Output Token) CDF per vLLM instance

TPOT is computed as:
    (decode_time - prefill_time) / actual_generated_tokens
where actual_generated_tokens comes from the decode log line.

Usage:
    python generate_ttft_tpot_cdf.py <directory_path>
"""

import argparse
import re
import os
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime, timedelta
from collections import defaultdict, namedtuple
from pathlib import Path

# Log filename
LOG_FILENAME = "router_v2.log"

# Data structures
RequestRoute = namedtuple('RequestRoute', ['timestamp', 'request_id', 'input_length', 'decode_length', 'instance_id'])
PrefillDone = namedtuple('PrefillDone', ['timestamp', 'request_id', 'hit_cnt', 'instance_id'])
DecodeDone = namedtuple('DecodeDone', ['timestamp', 'request_id', 'instance_id', 'generated_tokens'])

def parse_arguments():
    parser = argparse.ArgumentParser(description='Generate TTFT and TPOT CDF charts from router_v2.log')
    parser.add_argument('directory', type=str, help='Directory containing router_v2.log')
    parser.add_argument('--instances', type=int, nargs='+', default=[0, 7, 8, 15],
                        help='List of vLLM instance IDs to plot (default: 0 7 8 15)')
    parser.add_argument('--start-time', type=float, default=0.0,
                        help='Start time (seconds from first log) for filtering requests (default: 0)')
    parser.add_argument('--end-time', type=float, default=1400.0,
                        help='End time (seconds from first log) for filtering requests (default: 1400)')
    return parser.parse_args()

def parse_timestamp(log_timestamp):
    clean = log_timestamp.replace('Z', '')
    return datetime.fromisoformat(clean)

def extract_log_data(log_file_path):
    request_routes = {}
    prefill_dones = {}
    decode_dones = {}
    earliest_time = None

    # Pattern: Request routing
    route_pattern = re.compile(
        r'(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d+Z).*?'
        r'Request_(\d+)\s+queued\s+\d+us,\s+with\s+input\s+length\s+(\d+)\s+output\s+length\s+(\d+),\s+added\s+to\s+vLLM#(\d+)'
    )

    # Pattern: Prefill done (primary)
    prefill_pattern = re.compile(
        r'(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d+Z).*?'
        r'Vllm#(\d+)::Request_(\d+)\s+prefill\s+done\s+with\s+(\d+)\s+actual\s+hit\s+tokens!'
    )

    # Pattern: Prefill done (backup wording)
    backup_prefill_pattern = re.compile(
        r'(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d+Z).*?'
        r'Vllm#(\d+)::Request_(\d+)\s+prefill\s+with\s+(\d+)\s+actual\s+hit\s+tokens\s+done!'
    )

    # Pattern: Decode done — includes actual generated token count
    decode_pattern = re.compile(
        r'(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d+Z).*?'
        r'Vllm#(\d+)::Request_(\d+)\s+is\s+finished\s+generating\s+(\d+)\s+tokens'
    )

    with open(log_file_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            # --- Match request route ---
            m = route_pattern.search(line)
            if m:
                ts_str, rid, inp, out, inst = m.groups()
                ts = parse_timestamp(ts_str)
                request_routes[int(rid)] = RequestRoute(ts, int(rid), int(inp), int(out), int(inst))
                if earliest_time is None or ts < earliest_time:
                    earliest_time = ts
                continue

            # --- Match prefill done ---
            m = prefill_pattern.search(line) or backup_prefill_pattern.search(line)
            if m:
                ts_str, inst, rid, hit = m.groups()
                ts = parse_timestamp(ts_str)
                prefill_dones[int(rid)] = PrefillDone(ts, int(rid), int(hit), int(inst))
                continue

            # --- Match decode done ---
            m = decode_pattern.search(line)
            if m:
                ts_str, inst, rid, gen_tokens = m.groups()  # <-- 4 groups: timestamp, instance, request_id, generated_tokens
                ts = parse_timestamp(ts_str)
                decode_dones[int(rid)] = DecodeDone(ts, int(rid), int(inst), int(gen_tokens))
                continue

    if earliest_time is None:
        raise ValueError("No valid route events found in log.")
    return request_routes, prefill_dones, decode_dones, earliest_time

def build_lifecycles(request_routes, prefill_dones, decode_dones):
    lifecycles = []
    for rid, route in request_routes.items():
        if rid not in prefill_dones or rid not in decode_dones:
            continue
        prefill = prefill_dones[rid]
        decode = decode_dones[rid]
        # Ensure consistent instance ID across lifecycle
        if not (route.instance_id == prefill.instance_id == decode.instance_id):
            continue
        lifecycles.append({
            'request_id': rid,
            'instance_id': route.instance_id,
            'route_time': route.timestamp,
            'prefill_time': prefill.timestamp,
            'decode_time': decode.timestamp,
            'input_length': route.input_length,
            'actual_decode_tokens': decode.generated_tokens,  # Use actual generated tokens
        })
    return lifecycles

def compute_ttft_tpot_per_instance(lifecycles, start_abs, end_abs):
    """
    Compute TTFT and TPOT for each instance, filtering by prefill_time in [start_abs, end_abs].
    """
    data = defaultdict(lambda: {'ttft': [], 'tpot': []})
    for lc in lifecycles:
        prefill_time = lc['prefill_time']
        if not (start_abs <= prefill_time <= end_abs):
            continue

        # TTFT: from route to prefill completion
        ttft = (prefill_time - lc['route_time']).total_seconds()
        # Decode duration: from prefill completion to decode completion
        decode_duration = (lc['decode_time'] - prefill_time).total_seconds()
        actual_tokens = lc['actual_decode_tokens']

        # Skip invalid or degenerate cases
        if ttft < 0 or decode_duration < 0 or actual_tokens <= 0:
            continue

        tpot = decode_duration / actual_tokens

        iid = lc['instance_id']
        data[iid]['ttft'].append(ttft)
        data[iid]['tpot'].append(tpot)

    return dict(data)

def plot_cdf(metric_name, data_dict, selected_instances, output_dir, suffix):
    plt.figure(figsize=(10, 6))
    plotted_any = False
    for inst in selected_instances:
        if inst not in data_dict:
            continue
        values = data_dict[inst][metric_name]
        if not values:
            continue
        values = np.array(values)
        sorted_vals = np.sort(values)
        y = np.arange(1, len(sorted_vals) + 1) / len(sorted_vals)
        plt.plot(sorted_vals, y, label=f'Instance #{inst}', linewidth=2)
        plotted_any = True

    if not plotted_any:
        plt.close()
        return False

    xlabel = "TTFT (seconds)" if metric_name == 'ttft' else "TPOT (seconds/token)"
    title = f"CDF of {'TTFT' if metric_name == 'ttft' else 'TPOT'}"
    plt.title(title, fontsize=14, fontweight='bold')
    plt.xlabel(xlabel, fontsize=12)
    plt.ylabel('Cumulative Probability', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.legend(title='vLLM Instance', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.xlim(left=0)
    plt.tight_layout()

    filename = f"{metric_name}_cdf_{suffix}.png"
    output_path = os.path.join(output_dir, filename)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {filename}")
    return True

def main():
    args = parse_arguments()
    log_path = os.path.join(args.directory, LOG_FILENAME)
    if not os.path.exists(log_path):
        raise FileNotFoundError(f"Log file not found: {log_path}")

    print(f"Parsing log: {log_path}")
    routes, prefill, decode, start_time = extract_log_data(log_path)
    lifecycles = build_lifecycles(routes, prefill, decode)
    print(f"Built {len(lifecycles)} complete request lifecycles")

    if not lifecycles:
        print("No complete lifecycles. Exiting.")
        return

    # Convert relative time window to absolute datetime
    start_abs = start_time + timedelta(seconds=args.start_time)
    end_abs = start_time + timedelta(seconds=args.end_time)

    # Compute metrics
    data = compute_ttft_tpot_per_instance(lifecycles, start_abs, end_abs)

    # Build filename suffix
    present_instances = [i for i in args.instances if i in data and data[i]['ttft']]
    if not present_instances:
        present_instances = [i for i in args.instances if i in data]
    suffix = '_'.join(map(str, sorted(present_instances))) if present_instances else 'none'
    suffix += f"_t{int(args.start_time)}-{int(args.end_time)}"

    # Generate plots
    plot_cdf('ttft', data, args.instances, args.directory, suffix)
    plot_cdf('tpot', data, args.instances, args.directory, suffix)

    print("CDF charts generation complete.")

if __name__ == "__main__":
    main()
