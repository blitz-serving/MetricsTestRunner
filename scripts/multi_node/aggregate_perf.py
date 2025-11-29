import sys
import re
import csv
import argparse

# ✅ 硬编码：只分析以下策略
TARGET_POLICIES = {
    "least-wait-token-gated-bs",
    "least-wait-token-bs",
    "dynamo-deterministic",
    "bailian-impl-06",
    "join-shortest-q-tuple",
    "join-shortest-q-weight",
    "least-wait-token-random",
    "round-robin-q",
    "least-bs-random",
    "llumnix-linear-04",
    "least-ttft-bs-tuple",
    "least-wait-token-mul-bs",
    "least-wait-token-mul-tbt",
}
# "join-shortest-q-ttft",
# 
#     "llmd-impl-q",
#     "slo-serve-impl-q"
def parse_perf_log(file_path):
    results = []
    current_policy = None
    current_metric = None
    metrics = {}

    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.rstrip('\n\r')

            policy_match = re.match(r'^Statistics for ([\w\-_]+):$', line.strip())
            if policy_match:
                if current_policy is not None:
                    results.append((current_policy, dict(metrics)))
                current_policy = policy_match.group(1)
                metrics = {}
                current_metric = None
                continue

            if current_policy is None:
                continue

            metric_match = re.match(r'^\s{2}(\w+):$', line)
            if metric_match:
                current_metric = metric_match.group(1)
                metrics[current_metric] = {}
                continue

            stat_match = re.match(r'^\s{4}(\w+):\s*([\d\.]+)', line)
            if stat_match and current_metric:
                stat_name = stat_match.group(1)
                try:
                    value = float(stat_match.group(2))
                    metrics[current_metric][stat_name] = value
                except ValueError:
                    pass
                continue

        if current_policy is not None:
            results.append((current_policy, dict(metrics)))

    return results

def main():
    parser = argparse.ArgumentParser(description="Analyze perf.log files and output comparison table + CSV.")
    parser.add_argument('log_files', nargs='+', help='Input perf.log files')
    parser.add_argument('--csv', default='perf_summary.csv', help='Output CSV file (default: perf_summary.csv)')
    args = parser.parse_args()

    best_records = {}

    for log_file in args.log_files:
        parsed_list = parse_perf_log(log_file)
        for policy, metrics in parsed_list:
            ttft_mean = metrics.get('TTFT', {}).get('Mean')
            if ttft_mean is None:
                continue
            if policy not in TARGET_POLICIES:
                continue

            if policy not in best_records:
                best_records[policy] = metrics
            else:
                current_best_ttft = best_records[policy].get('TTFT', {}).get('Mean', float('inf'))
                if ttft_mean < current_best_ttft:
                    best_records[policy] = metrics

    if not best_records:
        print("No valid target policy records found.")
        return

    summary = {}
    for policy, metrics in best_records.items():
        ttft = metrics.get('TTFT', {}).get('Mean')
        tpot = metrics.get('TPOT', {}).get('Mean')
        total = metrics.get('TOTAL_TIME', {}).get('Mean')
        if None not in (ttft, tpot, total):
            summary[policy] = {
                'mean_ttft': ttft,
                'mean_tpot': tpot,
                'mean_total_time': total
            }

    if not summary:
        print("No complete metrics found for target policies.")
        return

    baseline_policy = max(summary, key=lambda p: summary[p]['mean_tpot'])
    baseline = summary[baseline_policy]

    print(f"Baseline policy (worst TTFT among targets): {baseline_policy}")
    print(f"  TTFT: {baseline['mean_ttft']:.2f}, TPOT: {baseline['mean_tpot']:.2f}, Total: {baseline['mean_total_time']:.2f}")
    print()

    output_rows = []
    for policy, data in summary.items():
        ttft_ratio = baseline['mean_ttft'] / data['mean_ttft']
        tpot_ratio = baseline['mean_tpot'] / data['mean_tpot']
        total_ratio = baseline['mean_total_time'] / data['mean_total_time']
        output_rows.append({
            'Policy': policy,
            'TTFT': data['mean_ttft'],
            'TPOT': data['mean_tpot'],
            'Total': data['mean_total_time'],
            'R_TTFT': ttft_ratio,
            'R_TPOT': tpot_ratio,
            'R_Total': total_ratio
        })

    # Sort by TTFT descending (worst on top)
    output_rows.sort(key=lambda x: x['TPOT'], reverse=True)

    # --- Print human-readable table ---
    header = f"{'Policy':<28} {'TTFT':>10} {'TPOT':>8} {'Total':>10} {'R_TTFT':>8} {'R_TPOT':>8} {'R_Total':>8}"
    print(header)
    print("-" * len(header))
    for row in output_rows:
        print(f"{row['Policy']:<28} "
              f"{row['TTFT']:>10.2f} "
              f"{row['TPOT']:>8.2f} "
              f"{row['Total']:>10.2f} "
              f"{row['R_TTFT']:>8.2f} "
              f"{row['R_TPOT']:>8.2f} "
              f"{row['R_Total']:>8.2f}")

    # --- Write CSV ---
    csv_fields = ['Policy', 'TTFT', 'TPOT', 'Total', 'R_TTFT', 'R_TPOT', 'R_Total']
    with open(args.csv, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=csv_fields)
        writer.writeheader()
        for row in output_rows:
            # Round floats to 2 decimal places for CSV as well
            out_row = {
                'Policy': row['Policy'],
                'TTFT': f"{row['TTFT']:.2f}",
                'TPOT': f"{row['TPOT']:.2f}",
                'Total': f"{row['Total']:.2f}",
                'R_TTFT': f"{row['R_TTFT']:.2f}",
                'R_TPOT': f"{row['R_TPOT']:.2f}",
                'R_Total': f"{row['R_Total']:.2f}",
            }
            writer.writerow(out_row)

    print(f"\nCSV output written to: {args.csv}")

if __name__ == "__main__":
    main()
