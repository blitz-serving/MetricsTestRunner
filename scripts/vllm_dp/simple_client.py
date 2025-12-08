#!/usr/bin/env python3
import argparse
import asyncio
import json
import time
import numpy as np
from tqdm.asyncio import tqdm
import aiohttp

BLOCK_SIZE = 16
MODEL = "/home/admin/resource/model/464482ce.Qwen2.5-7B-Instruct/1.0/"

def reconstruct_token_ids(hash_ids, target_length):
    """
    Reconstruct token_ids of length == target_length using hash_ids.
    Each hash_id is expanded to BLOCK_SIZE tokens by repeating it as token values.
    """
    tokens = []
    for h in hash_ids:
        tokens.extend([h] * BLOCK_SIZE)
        if len(tokens) >= target_length:
            break
    # Truncate to exact target_length
    return tokens[:target_length]

async def send_request(session, prompt_token_ids, min_tokens, max_tokens):
    start = time.perf_counter()
    ttft = None
    output_tokens = 0
    total_generated_text = ""

    # For Chat API, we need to provide messages format
    # We'll use a placeholder message and override with prompt_token_ids via extra_body
    async with session.post(
        "http://0.0.0.0:8000/v1/chat/completions",
        json={
            "model": MODEL,  # This is required but will be ignored if prompt_token_ids is provided
            "messages": [
                {"role": "user", "content": "placeholder"}  # Placeholder content
            ],
            "extra_body": {
                "prompt_token_ids": prompt_token_ids,
                "ignore_eos": True,
                "min_tokens": min_tokens,
            },
            "max_tokens": max_tokens,
            "stream": True,
            "stream_options": {"include_usage": True}
        }
    ) as resp:
        async for line in resp.content:
            line = line.decode("utf-8").strip()
            if not line.startswith(""):
                continue
            data = line[5:].strip()
            if data == "[DONE]":
                break
            try:
                chunk = json.loads(data)
            except json.JSONDecodeError:
                continue

            # First token arrival time
            if ttft is None and 'choices' in chunk and chunk['choices']:
                delta = chunk['choices'][0].get('delta', {})
                if delta.get('content'):
                    ttft = time.perf_counter() - start
            
            # Count output tokens from usage if available
            if 'usage' in chunk and chunk['usage']:
                output_tokens = chunk['usage'].get('completion_tokens', 0)
            
            # Accumulate generated text for fallback token counting
            if 'choices' in chunk and chunk['choices']:
                delta = chunk['choices'][0].get('delta', {})
                content = delta.get('content', '')
                if content:
                    total_generated_text += content

    total_time = time.perf_counter() - start
    
    # Fallback token counting if usage not available
    if output_tokens == 0 and total_generated_text:
        # Rough estimate - this is not accurate but better than nothing
        output_tokens = len(total_generated_text.split())
    
    # Calculate TPOT (Time Per Output Token)
    if output_tokens <= 1:
        tpot = float('inf')
    else:
        tpot = (total_time - ttft) / (output_tokens - 1) if ttft else total_time / output_tokens
    
    return {
        "ttft": ttft,
        "tpot": tpot,
        "total_time": total_time,
        "output_tokens": output_tokens,
        "total_text": total_generated_text
    }

async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--jsonl", required=True, help="Path to input JSONL trace file")
    parser.add_argument("--qps", type=float, required=True, help="Target queries per second")
    parser.add_argument("--duration", type=int, required=True, help="Test duration in seconds")
    args = parser.parse_args()

    n_requests = int(args.qps * args.duration)
    if n_requests <= 0:
        print("No requests to send.")
        return

    # Load traces
    traces = []
    with open(args.jsonl, "r") as f:
        for i, line in enumerate(f):
            if i >= n_requests:
                break
            traces.append(json.loads(line))
    if len(traces) < n_requests:
        print(f"Warning: only {len(traces)} traces available, but {n_requests} requested.")
        n_requests = len(traces)

    print(f"Loaded {n_requests} traces for benchmarking at {args.qps} QPS for {args.duration}s.")

    # Precompute prompts
    requests = []
    for trace in traces:
        input_len = trace["input_length"]
        output_len = trace["output_length"]
        hash_ids = trace["hash_ids"]
        prompt_tokens = reconstruct_token_ids(hash_ids, input_len)
        assert len(prompt_tokens) == input_len, f"Length mismatch: {len(prompt_tokens)} vs {input_len}"
        requests.append((prompt_tokens, output_len))

    # Inter-arrival time
    interval = 1.0 / args.qps

    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=300)) as session:
        tasks = []
        start_time = time.perf_counter()
        for i, (prompt, output_len) in enumerate(requests):
            # Sleep to control QPS
            elapsed = time.perf_counter() - start_time
            target_time = i * interval
            if elapsed < target_time:
                await asyncio.sleep(target_time - elapsed)
            task = asyncio.create_task(send_request(session, prompt, output_len, output_len))
            tasks.append(task)

        results = []
        for f in tqdm.as_completed(tasks, total=len(tasks), desc="Processing"):
            res = await f
            results.append(res)

    # === Statistics ===
    ttfts = [r["ttft"] * 1000 for r in results if r["ttft"] is not None]          # Convert to ms
    tpots = [r["tpot"] * 1000 for r in results if r["tpot"] != float('inf') and r["tpot"] > 0]  # Convert to ms
    total_times = [r["total_time"] * 1000 for r in results]                       # Convert to ms
    output_tokens_list = [r["output_tokens"] for r in results if r["output_tokens"] > 0]

    def fmt_stats(arr, name, unit="ms"):
        if not arr:
            print(f"{name}: no valid data")
            return
        arr = np.array(arr)
        mean = np.mean(arr)
        p50 = np.percentile(arr, 50)
        p90 = np.percentile(arr, 90)
        p99 = np.percentile(arr, 99)
        std = np.std(arr)
        print(f"{name} ({unit}): mean={mean:.4f}, std={std:.4f}, p50={p50:.4f}, p90={p90:.4f}, p99={p99:.4f}, min={np.min(arr):.4f}, max={np.max(arr):.4f}")

    print("\n=== Benchmark Results ===")
    fmt_stats(ttfts, "TTFT (Time To First Token)")
    fmt_stats(tpots, "TPOT (Time Per Output Token)")
    fmt_stats(total_times, "Total Latency")
    if output_tokens_list:
        fmt_stats(output_tokens_list, "Output Tokens", "tokens")

    # Calculate throughput (unchanged: tokens/second is standard)
    total_output_tokens = sum(r["output_tokens"] for r in results if r["output_tokens"] > 0)
    total_input_tokens = sum(len(r[0]) for r in requests[:len(results)])
    total_time_sec = max(r["total_time"] for r in results) if results else 0  # Keep in seconds for throughput

    if total_time_sec > 0:
        tokens_per_second = total_output_tokens / total_time_sec
        print(f"\nThroughput: {tokens_per_second:.2f} tokens/second")
        print(f"Total input tokens processed: {total_input_tokens}")
        print(f"Total output tokens generated: {total_output_tokens}")

    print(f"\nCompleted {len(results)} requests.")

if __name__ == "__main__":
    asyncio.run(main())

# python simple_client.py --jsonl /mnt/debugger/hjb/node1/qwen-bailian-usagetraces-anon/qwen_traceA_blksz_16.jsonl --qps 6 --duration 10