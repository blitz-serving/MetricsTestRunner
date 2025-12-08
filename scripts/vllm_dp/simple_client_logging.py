#!/usr/bin/env python3
import argparse
import asyncio
import json
import time
import os
import numpy as np
from tqdm.asyncio import tqdm
import aiohttp

BLOCK_SIZE = 16
MODEL = "/home/admin/resource/model/464482ce.Qwen2.5-7B-Instruct/1.0/"

def reconstruct_token_ids(hash_ids, target_length):
    tokens = []
    for h in hash_ids:
        tokens.extend([h] * BLOCK_SIZE)
        if len(tokens) >= target_length:
            break
    return tokens[:target_length]


async def send_request(session, prompt_token_ids, min_tokens, max_tokens, request_id, output_file):
    start = time.perf_counter()
    ttft = None
    output_tokens = 0
    total_generated_text = ""
    status = 200  # default
    encountered_json_error = False
    first_token_time = None

    try:
        async with session.post(
            "http://0.0.0.0:8000/v1/chat/completions",
            json={
                "model": MODEL,
                "messages": [{"role": "user", "content": "placeholder"}],
                # 关键参数直接放在顶层
                "prompt_token_ids": prompt_token_ids,
                "ignore_eos": True,
                "min_tokens": min_tokens,
                "max_tokens": max_tokens,
                "temperature": 0.0,  # 确保确定性输出
                "top_p": 1.0,
                "stream": True,
                "stream_options": {"include_usage": True}
            },
            timeout=aiohttp.ClientTimeout(total=300)
        ) as resp:
            
            if resp.status != 200:
                status = resp.status
                try:
                    error_body = await resp.text()
                    print(f"Request {request_id} failed with status {resp.status}: {error_body[:200]}")
                except Exception as e:
                    error_body = str(e)
                # 仍然记录失败请求
                total_time = time.perf_counter() - start
                entry = {
                    "request_id": request_id,
                    "input_length": len(prompt_token_ids),
                    "output_length": 0,
                    "ttft": None,
                    "tpot": None,
                    "total_time": round(total_time * 1000, 4),
                    "status": status,
                    "error": error_body[:200] if isinstance(error_body, str) else str(error_body)
                }
                with open(output_file, "a") as f:
                    f.write(json.dumps(entry) + "\n")
                return {
                    "ttft": None,
                    "tpot": float('inf'),
                    "total_time": total_time,
                    "output_tokens": 0,
                }

            # 处理SSE流式响应
            async for line in resp.content:
                line = line.decode("utf-8").strip()
                if not line:
                    continue
                
                # 处理SSE格式: "data: {...}"
                if line.startswith("data:"):
                    data = line[5:].strip()
                else:
                    # 有些实现可能直接返回JSON
                    data = line
                
                if data == "[DONE]":
                    break
                
                try:
                    chunk = json.loads(data)
                except json.JSONDecodeError as e:
                    encountered_json_error = True
                    print(f"JSON decode error in chunk for request {request_id}: {e}")
                    print(f"Problematic line: {line[:200]}")
                    continue

                # 记录TTFT - 第一个有效token的时间
                if ttft is None and 'choices' in chunk and chunk['choices']:
                    delta = chunk['choices'][0].get('delta', {})
                    if delta.get('content'):
                        ttft = time.perf_counter() - start
                        first_token_time = time.perf_counter()

                # 累计生成的文本
                if 'choices' in chunk and chunk['choices']:
                    delta = chunk['choices'][0].get('delta', {})
                    content = delta.get('content', '')
                    if content:
                        total_generated_text += content

                # 从usage获取token数量
                if 'usage' in chunk and chunk['usage']:
                    output_tokens = chunk['usage'].get('completion_tokens', output_tokens)

            # 检查是否达到预期输出长度
            if output_tokens < min_tokens and not encountered_json_error:
                print(f"Warning: Request {request_id} expected {min_tokens} tokens but got {output_tokens} tokens")
                print(f"Generated text preview: {total_generated_text[:100]}...")

            # 标记JSON错误但无有效输出的请求为429
            if encountered_json_error and output_tokens == 0:
                status = 429

            total_time = time.perf_counter() - start

    except asyncio.TimeoutError:
        print(f"Request {request_id} timed out after 300 seconds")
        status = 408
        total_time = time.perf_counter() - start
        entry = {
            "request_id": request_id,
            "input_length": len(prompt_token_ids),
            "output_length": 0,
            "ttft": None,
            "tpot": None,
            "total_time": round(total_time * 1000, 4),
            "status": 408,
            "error": "Request timeout after 300 seconds"
        }
        with open(output_file, "a") as f:
            f.write(json.dumps(entry) + "\n")
        return {
            "ttft": None,
            "tpot": float('inf'),
            "total_time": total_time,
            "output_tokens": 0,
        }
    except Exception as e:
        print(f"Unexpected error for request {request_id}: {str(e)}")
        status = 500
        total_time = time.perf_counter() - start
        entry = {
            "request_id": request_id,
            "input_length": len(prompt_token_ids),
            "output_length": 0,
            "ttft": None,
            "tpot": None,
            "total_time": round(total_time * 1000, 4),
            "status": 500,
            "error": str(e)
        }
        with open(output_file, "a") as f:
            f.write(json.dumps(entry) + "\n")
        return {
            "ttft": None,
            "tpot": float('inf'),
            "total_time": total_time,
            "output_tokens": 0,
        }

    # 如果没有从usage获取到token数，使用文本估算（不准确但有总比没有好）
    if output_tokens == 0 and total_generated_text:
        # 更准确的token估算：按空格分割，但考虑标点符号
        output_tokens = max(1, len(total_generated_text.split()))

    # 计算TPOT (Time Per Output Token)
    tpot = float('inf')
    if output_tokens > 0:
        if ttft is not None and output_tokens > 1:
            # TPOT = (总时间 - TTFT) / (输出token数 - 1)
            tpot = (total_time - ttft) / (output_tokens - 1)
        elif ttft is not None:
            # 只有一个token，TTFT就是总时间
            tpot = total_time
        else:
            # 没有记录到TTFT，使用平均时间
            tpot = total_time / output_tokens

    # 验证输出长度是否符合要求
    expected_length = min_tokens
    actual_length = output_tokens
    length_ratio = actual_length / expected_length if expected_length > 0 else 0
    
    # 记录结果
    entry = {
        "request_id": request_id,
        "input_length": len(prompt_token_ids),
        "output_length": output_tokens,
        "min_tokens_requested": min_tokens,
        "max_tokens_requested": max_tokens,
        "ttft": round(ttft * 1000, 4) if ttft is not None else None,
        "tpot": round(tpot * 1000, 4) if tpot != float('inf') else None,
        "total_time": round(total_time * 1000, 4),
        "status": status,
        "length_ratio": round(length_ratio, 4),
        "generated_text_preview": total_generated_text[:200] if total_generated_text else None
    }

    with open(output_file, "a") as f:
        f.write(json.dumps(entry) + "\n")

    return {
        "ttft": ttft,
        "tpot": tpot,
        "total_time": total_time,
        "output_tokens": output_tokens,
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

    requests = []
    request_ids = []
    for trace in traces:
        input_len = trace["input_length"]
        output_len = trace["output_length"]
        hash_ids = trace["hash_ids"]
        chat_id = trace["chat_id"]
        prompt_tokens = reconstruct_token_ids(hash_ids, input_len)
        assert len(prompt_tokens) == input_len, f"Length mismatch: {len(prompt_tokens)} vs {input_len}"
        requests.append((prompt_tokens, output_len))
        request_ids.append(chat_id)

    interval = 1.0 / args.qps

    # Create single output file for this run
    os.makedirs("outputs", exist_ok=True)
    timestamp_str = time.strftime("%Y%m%d_%H%M%S", time.localtime())
    output_file = f"outputs/{timestamp_str}.jsonl"
    print(f"Per-request results will be saved to: {output_file}")

    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=300)) as session:
        tasks = []
        start_time = time.perf_counter()
        for i, (prompt, output_len) in enumerate(requests):
            elapsed = time.perf_counter() - start_time
            target_time = i * interval
            if elapsed < target_time:
                await asyncio.sleep(target_time - elapsed)
            task = asyncio.create_task(
                send_request(session, prompt, output_len, output_len, request_ids[i], output_file)
            )
            tasks.append(task)

        results = []
        for f in tqdm.as_completed(tasks, total=len(tasks), desc="Processing"):
            res = await f
            results.append(res)

    # === Statistics ===
    ttfts = [r["ttft"] * 1000 for r in results if r["ttft"] is not None]
    tpots = [r["tpot"] * 1000 for r in results if r["tpot"] != float('inf') and r["tpot"] > 0]
    total_times = [r["total_time"] * 1000 for r in results]
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

    total_output_tokens = sum(r["output_tokens"] for r in results if r["output_tokens"] > 0)
    total_input_tokens = sum(len(r[0]) for r in requests[:len(results)])
    total_time_sec = max(r["total_time"] for r in results) if results else 0

    if total_time_sec > 0:
        tokens_per_second = total_output_tokens / total_time_sec
        print(f"\nThroughput: {tokens_per_second:.2f} tokens/second")
        print(f"Total input tokens processed: {total_input_tokens}")
        print(f"Total output tokens generated: {total_output_tokens}")

    print(f"\nCompleted {len(results)} requests. Results saved to: {output_file}")

if __name__ == "__main__":
    asyncio.run(main())

# python simple_client_logging.py --jsonl /mnt/debugger/hjb/node1/qwen-bailian-usagetraces-anon/qwen_traceA_blksz_16.jsonl --qps 6 --duration 10