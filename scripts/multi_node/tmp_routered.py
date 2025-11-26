#!/usr/bin/env python3
import re
import sys
from collections import defaultdict

def count_assigned_requests(log_lines):
    """
    统计每个 vLLM 实例被分配的请求数。
    匹配模式: ... Request_<id> queued ... added to vLLM#<N>
    """
    pattern = re.compile(
        r"Request_(\d+)\s+queued\s+(\d+)us,\s+"
        r"with\s+input\s+length\s+(\d+)\s+"
        r"output\s+length\s+(\d+),\s+"
        r"added\s+to\s+vLLM#(\d+)"
    )
    instance_counts = defaultdict(int)
    for line in log_lines:
        match = pattern.search(line)
        if match:
            instance_id = f"vLLM#{match.group(5)}"
            instance_counts[instance_id] += 1
    return instance_counts


def aggregate_hit_tokens(log_lines):
    """
    统计每个 vLLM 实例的 hit tokens 总数。
    匹配模式: ... Vllm#<N>::Request_<id> prefill done with <num> actual hit tokens!
    注意：日志中是 "Vllm#"（大写 V），但我们统一输出为 "vLLM#"
    """
    pattern = re.compile(
        r"Vllm#(\d+)::Request_\d+\s+prefill\s+done\s+with\s+(\d+)\s+actual\s+hit\s+tokens!"
    )
    instance_hits = defaultdict(int)
    for line in log_lines:
        match = pattern.search(line)
        if match:
            instance_num = match.group(1)
            hit_tokens = int(match.group(2))
            instance_id = f"vLLM#{instance_num}"
            instance_hits[instance_id] += hit_tokens
    return instance_hits


def read_log_lines(log_path):
    with open(log_path, 'r', encoding='utf-8') as f:
        return f.readlines()


def main():
    if len(sys.argv) != 2:
        print("Usage: python3 analyze_vllm_log.py <logfile.log>", file=sys.stderr)
        sys.exit(1)

    log_file = sys.argv[1]
    try:
        log_lines = read_log_lines(log_file)
    except FileNotFoundError:
        print(f"Error: Log file '{log_file}' not found.", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Error reading log file: {e}", file=sys.stderr)
        sys.exit(1)

    # 功能1: 请求分配统计
    request_counts = count_assigned_requests(log_lines)

    # 功能2: Hit tokens 统计
    hit_tokens_per_instance = aggregate_hit_tokens(log_lines)
    total_hit_tokens = sum(hit_tokens_per_instance.values())

    # === 输出结果 ===
    print("=== Request Assignment Counts ===")
    if request_counts:
        for inst in sorted(request_counts.keys(), key=lambda x: int(x.split('#')[1])):
            print(f"{inst}: {request_counts[inst]} requests")
    else:
        print("No request assignment logs found.")

    print("\n=== Hit Tokens per Instance ===")
    if hit_tokens_per_instance:
        for inst in sorted(hit_tokens_per_instance.keys(), key=lambda x: int(x.split('#')[1])):
            print(f"{inst}: {hit_tokens_per_instance[inst]} hit tokens")
    else:
        print("No hit token logs found.")

    print(f"\n=== Total Hit Tokens (Global) ===\n{total_hit_tokens}")

if __name__ == "__main__":
    main()