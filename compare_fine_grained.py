#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
对比 simulator 真实执行与 predictor 预测的逐算子执行时间
- 按时间段绘制图像（默认每120s）
- 随机抽取若干段（默认6段）
- 剔除 latency > 300 的异常点
- 阶段分类以预测值为准
"""

import os
import re
import sys
import random
from datetime import datetime, timedelta
import pandas as pd
import matplotlib.pyplot as plt


# ------------------- 正则匹配 -------------------
real_re = re.compile(
    r'(?P<time>\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d+Z).*?VllmMetric\s*\{\s*'
    r'prefill_tokens:\s*(?P<prefill_tokens>\d+),.*?latency:\s*(?P<latency>\d+),.*?'
    r'outputs:\s*(?P<outputs>.*?)op_exec_log:\s*Some\("(?P<ops>.*?)"\)\s*\}',
    re.S,
)

pred_re = re.compile(
    r'(?P<time>\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d+Z).*?Log info from base predictor:\s*(?P<ops>.*?)(?=$|\n)',
    re.S,
)


# ------------------- 工具函数 -------------------
def parse_ops(ops_str: str) -> dict[str, float]:
    """解析类似 a=1.23, b=0.45 的字符串为字典"""
    res = {}
    for kv in ops_str.split(","):
        if "=" not in kv:
            continue
        k, v = kv.strip().split("=", 1)
        k = k.strip()
        v = v.strip().replace("ms", "")
        try:
            res[k] = float(v)
        except ValueError:
            continue
    return res


def load_logs(file_path: str):
    """加载日志并提取真实执行与预测数据"""
    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
        text = f.read()

    real_logs = []
    for m in real_re.finditer(text):
        t = datetime.fromisoformat(m.group("time").replace("Z", "+00:00")).replace(tzinfo=None)
        latency = float(m.group("latency"))
        real_ops = parse_ops(m.group("ops"))
        real_logs.append({"time": t, "real_ops": real_ops, "latency": latency})

    pred_logs = []
    for m in pred_re.finditer(text):
        t = datetime.fromisoformat(m.group("time").replace("Z", "+00:00")).replace(tzinfo=None)
        pred_ops = parse_ops(m.group("ops"))
        pred_logs.append({"time": t, "pred_ops": pred_ops})

    return real_logs, pred_logs


def determine_stage(pred_ops: dict) -> str:
    """以预测值为准判定阶段"""
    attn_prefill = pred_ops.get("attn_prefill", 0.0)
    attn_decode = pred_ops.get("attn_decode", 0.0)
    if attn_prefill != 0 and attn_decode == 0:
        return "PREFILL"
    elif attn_decode != 0 and attn_prefill == 0:
        return "DECODE"
    else:
        return "COLO"


def merge_logs(real_logs, pred_logs):
    """按顺序配对真实与预测记录并剔除异常"""
    records = []
    for r, p in zip(real_logs, pred_logs):
        # 剔除异常：latency > 300
        if r["latency"] > 300:
            continue

        stage = determine_stage(p["pred_ops"])
        real_ops = r["real_ops"]
        pred_ops = p["pred_ops"]

        op_pairs = {
            "norm": ("norm", ["attn_norm", "mlp_norm"]),
            "qkv_proj": ("qkv_proj", ["qkv_proj"]),
            "rotary_emb": ("rotary_emb", ["rotary_emb"]),
            "attention": ("attention", ["attn_kv_cache_save", "attn_prefill", "attn_decode"]),
            "o_proj": ("o_proj", ["o_proj"]),
            "mlp_gate_up_proj": ("mlp_gate_up_proj", ["mlp_gate_up_proj"]),
            "mlp_activation": ("mlp_activation", ["mlp_activation"]),
            "mlp_down_proj": ("mlp_down_proj", ["mlp_down_proj"]),
        }

        for real_key, (target_name, pred_keys) in op_pairs.items():
            if real_key not in real_ops:
                continue

            real_v = real_ops[real_key]
            pred_v = sum(pred_ops.get(k, 0.0) for k in pred_keys)

            if real_key == "attention":
                # 根据阶段决定绘图归类
                if stage == "PREFILL":
                    op_name = "attn_prefill"
                    pred_v = pred_ops.get("attn_prefill", 0.0)
                elif stage == "DECODE":
                    op_name = "attn_decode"
                    pred_v = pred_ops.get("attn_kv_cache_save", 0.0) + pred_ops.get("attn_decode", 0.0)
                else:
                    op_name = "attn_colo"
                    pred_v = (
                        pred_ops.get("attn_prefill", 0.0)
                        + pred_ops.get("attn_kv_cache_save", 0.0)
                        + pred_ops.get("attn_decode", 0.0)
                    )
            else:
                op_name = real_key

            records.append({
                "timestamp": r["time"],
                "op": op_name,
                "real_ms": real_v,
                "pred_ms": pred_v,
                "stage": stage,
                "latency": r["latency"],
            })

    return pd.DataFrame(records)


# ------------------- 绘图 -------------------
def plot_by_segments(df: pd.DataFrame, output_dir: str):
    """每 120 秒为一段绘图，随机抽取若干段"""
    if df.empty:
        print("❌ 没有可绘制的数据")
        return

    df = df.sort_values("timestamp")
    start_time = df["timestamp"].min()
    end_time = df["timestamp"].max()
    total_seconds = (end_time - start_time).total_seconds()
    segment_len = 120  # 每段120秒

    n_segments = int(total_seconds // segment_len) + 1
    seg_indices = sorted(random.sample(range(n_segments), min(6, n_segments)))

    for seg_idx in seg_indices:
        seg_start = start_time + timedelta(seconds=seg_idx * segment_len)
        seg_end = seg_start + timedelta(seconds=segment_len)
        seg_dir = os.path.join(output_dir, f"segment_{seg_start.strftime('%H%M%S')}_{seg_end.strftime('%H%M%S')}")
        os.makedirs(seg_dir, exist_ok=True)
        sub = df[(df["timestamp"] >= seg_start) & (df["timestamp"] < seg_end)]
        if sub.empty:
            continue

        for op, group in sub.groupby("op"):
            plt.figure(figsize=(10, 5))
            plt.plot(group["timestamp"], group["real_ms"], label="Real", linewidth=1.5)
            plt.plot(group["timestamp"], group["pred_ms"], label="Pred", linewidth=1.5, linestyle="--")
            plt.title(f"{op} ({seg_start.strftime('%H:%M:%S')} – {seg_end.strftime('%H:%M:%S')})")
            plt.xlabel("Time")
            plt.ylabel("Exec Time (ms)")
            plt.legend()
            plt.grid(True, linestyle="--", alpha=0.6)
            plt.tight_layout()
            plt.savefig(os.path.join(seg_dir, f"{op}.png"))
            plt.close()

        print(f"✅ 绘制完成: {seg_dir} ({len(sub)} 条记录)")


# ------------------- 主程序 -------------------
def main():
    if len(sys.argv) < 2:
        print("用法: python compare_fine_grained.py <log_file_path>")
        return

    log_file = sys.argv[1]
    if not os.path.isfile(log_file):
        print(f"❌ 文件不存在: {log_file}")
        return

    output_dir = os.path.dirname(log_file)
    print(f"📂 输出目录: {output_dir}")

    real_logs, pred_logs = load_logs(log_file)
    print(f"🔍 匹配到 {len(real_logs)} 条真实日志, {len(pred_logs)} 条预测日志")

    df = merge_logs(real_logs, pred_logs)
    print(f"📈 数据条数: {len(df)} (ops={df['op'].nunique()})")
    print("📊 数据统计（过滤后）:")
    print(df.groupby("op")[["real_ms", "pred_ms"]].describe().round(3))

    plot_by_segments(df, output_dir)


if __name__ == "__main__":
    main()
