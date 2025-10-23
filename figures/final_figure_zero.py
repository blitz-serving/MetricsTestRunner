import re
import json
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.colors as mcolors
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from collections import defaultdict
from matplotlib.lines import Line2D
import matplotlib.patches as mpatches
import os
import argparse
import subprocess

parser = argparse.ArgumentParser()
parser.add_argument("--output-dir", type=str, required=True)
args = parser.parse_args()


# ========= 基本路径设置 =========
PREFIX = args.output_dir
RAW_LOG_FILE = f"{PREFIX}/router_v2.log"
TRUNC_LOG_FILE = f"{PREFIX}/router_truncated.log"
processed_file = f"{PREFIX}/processed.log"

CLIENT_FILE = f"{PREFIX}/client.jsonl"
TMP_LOG_FILE = f"{PREFIX}/router_v2.tmp"

# Define alert_file variable (even though we're not using it anymore, 
# just to prevent NameError in case it's referenced elsewhere)
alert_file = f"{PREFIX}/alert.log"
alert_inst_pattern = re.compile(r"")
alert_global_pattern = re.compile(r"")
alert_array_pattern = re.compile(r"")

def keep_from_first(pattern: str, infile: str, outfile: str):
    with open(outfile, "wb") as out:
        subprocess.run(
            ["awk", "-v", f"p={pattern}", r'f||$0~p{f=1} f', infile],
            check=True,
            stdout=out
        )

def keep_to_last(pattern2: str, infile: str, outfile: str):
    with open(outfile, "wb") as out:
        subprocess.run(
            [
                "awk",
                "-v", f"p={pattern2}",
                r"$0~p{last=NR}{buf[NR]=$0} END{for(i=1;i<=last;i++) print buf[i]}",
                infile,
            ],
            check=True,
            stdout=out,
        )



keep_from_first("added request", RAW_LOG_FILE, TMP_LOG_FILE)
keep_to_last("VllmMetric", TMP_LOG_FILE, TRUNC_LOG_FILE)
# subprocess.run(['rm', TMP_LOG_FILE])
# exit(0)

# ========= 预处理 =========
os.makedirs(PREFIX, exist_ok=True)
with open(processed_file, 'w') as outfile:
    subprocess.run(['grep', 'VllmMetric', TRUNC_LOG_FILE], stdout=outfile)

GRAPH_PATH = os.path.join(os.path.dirname(processed_file), "instance_fig/")
os.makedirs(GRAPH_PATH, exist_ok=True)

# ========= 颜色与样式 =========
salmon_cmap = mcolors.LinearSegmentedColormap.from_list("prefill_cmap", ["white", "salmon"])
COLOR_PREFILL_TOKENS = "#1f77b4"  # 蓝
COLOR_REQUEST_PERSEC  = "#ff7f0e"  # 橙
COLOR_BATCH_INST      = "#9467bd"  # 紫
COLOR_DECODE_LAT      = "#2ca02c"  # 绿
COLOR_DECODE_COUNT    = "#d62728"  # 红
COLOR_TTFT            = "#17becf"  # 青（Total 第三图：左轴）
COLOR_TPOT            = "#8c564b"  # 褐（Total 第三图：右轴）

LW_MAIN   = 2.0     # 主要曲线粗细
LW_THIN   = 1.2     # 全局与 TTFT/TPOT 的细线

def set_time_formatter(ax):
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
    ax.grid(True)

# ========= 正则（processed.log: VllmMetric）=========
vm_pattern = re.compile(
    r"(?P<timestamp>[\d\-:T\.]+Z).*?Vllm#(?P<instance_id>\d+)::Event::data received VllmMetric { prefill_tokens: (?P<prefill_tokens>\d+), prefill_token_budget: \d+, latency: (?P<latency>\d+), outputs: \[(?P<outputs>.*?)\], new_block_hashes: \[.*?\], evicted_block_ids: \[.*?\], cur_used_block_ids: .*? }"
)
output_pattern = re.compile(
    r"request_id: (?P<request_id>\d+), new_token_ids: \[.*?\], state: \"(?P<state>PREFILL|DECODE)\", is_finished: (true|false)"
)


# ========= 解析 processed.log（细粒度事件 -> 每秒每实例）=========
instance_events = defaultdict(list)
# 记录 request_id==0 的 PREFILL 候选，用于对齐 s_time=0 的绝对时间
req0_prefill_candidates = []  # [(timestamp, latency_ms)]

with open(processed_file, "r") as f:
    for line in f:
        m = vm_pattern.search(line)
        if not m:
            continue
        ts = datetime.fromisoformat(m.group("timestamp").replace("Z", "+00:00"))  # aware UTC
        latency = int(m.group("latency"))
        inst = f"Vllm#{m.group('instance_id')}"
        outputs = m.group("outputs")

        for out in output_pattern.finditer(outputs):
            state = out.group("state")
            req_id = int(out.group("request_id"))
            instance_events[inst].append({
                "time": ts,
                "second": pd.to_datetime(ts).floor("s"),
                "state": state,
                "latency": latency,
                "request_id": req_id
            })
            if state == "PREFILL" and req_id == 0:
                req0_prefill_candidates.append((ts, latency))

# processed 每秒聚合
per_inst_proc = {}
all_prefill_candidates_any = []  # 作为兜底
for inst, events in instance_events.items():
    df = pd.DataFrame(events)
    if df.empty:
        continue
    # 兜底：任意 PREFILL 候选
    any_prefill = df[df["state"] == "PREFILL"][["time", "latency"]]
    for _, r in any_prefill.iterrows():
        all_prefill_candidates_any.append((r["time"], int(r["latency"])))

    g = df.groupby("second")

    def decode_mean(s):
        idx = s.index
        return df.loc[idx][df.loc[idx, "state"] == "DECODE"]["latency"].mean()

    decode_latency = g["latency"].apply(decode_mean)

    prefill_lat_sum = df[df["state"] == "PREFILL"].groupby("second")["latency"].sum()
    prefill_ratio = (prefill_lat_sum / 1000.0).clip(upper=1.0)

    decode_df = df[df["state"] == "DECODE"]
    decode_count = (
        decode_df.drop_duplicates(subset=["second", "request_id"])
                .groupby("second").size()
    )

    merged = pd.concat([
        decode_latency.rename("decode_latency_avg"),
        prefill_ratio.rename("prefill_ratio"),
        decode_count.rename("decode_count")
    ], axis=1).sort_index()

    per_inst_proc[inst] = merged


# ========= 计算 s_time=0 的绝对时间（对齐 client.jsonl 与 processed.log）=========
s0_abs = None
if req0_prefill_candidates:
    # 取时间最早的 request_id=0 的 PREFILL 事件
    t0, lat0 = sorted(req0_prefill_candidates, key=lambda x: x[0])[0]
    s0_abs = t0 - timedelta(milliseconds=int(lat0))
elif all_prefill_candidates_any:
    # 兜底：任意 PREFILL 事件最早的一条
    t0, lat0 = sorted(all_prefill_candidates_any, key=lambda x: x[0])[0]
    s0_abs = t0 - timedelta(milliseconds=int(lat0))
else:
    # 最弱兜底：任意事件最早时间
    any_events = [r["time"] for lst in instance_events.values() for r in lst]
    if any_events:
        s0_abs = min(any_events)
# 若仍为空，后续 TTFT/TPOT 将为空处理

# ========= 解析“请求级 JSON 指标”（TTFT/TPOT，按绝对日志时间聚合）=========
# 以 统计时间 = s_time(秒) + first_token_time(毫秒)/1000，映射到 绝对时间 = s0_abs + 统计时间
ttft_tpot_rows = []
if s0_abs is not None and os.path.exists(CLIENT_FILE):
    with open(CLIENT_FILE, "r") as f:
        for line in f:
            if '"avg_time_between_tokens"' not in line:
                continue
            # 抓取 JSON
            try:
                start = line.index("{")
                end = line.rindex("}") + 1
                j = json.loads(line[start:end])
            except Exception:
                continue

            s_time = j.get("s_time", j.get("stime"))
            ttft_ms = j.get("first_token_time")
            tpot_ms = j.get("avg_time_between_tokens")
            if s_time is None or ttft_ms is None or tpot_ms is None:
                continue

            try:
                s_time_ms = float(s_time)     # 秒
                ttft_ms = float(ttft_ms)   # 毫秒
                tpot_ms = float(tpot_ms)   # 毫秒
            except Exception:
                continue

            abs_ts = s0_abs + timedelta(milliseconds=s_time_ms + ttft_ms)
            bucket = pd.to_datetime(abs_ts).floor("s")

            ttft_tpot_rows.append({
                "second": bucket,
                "TTFT_ms": ttft_ms,
                "TPOT_ms": tpot_ms
            })

# 每秒聚合：把“统计时间”落在这一秒的请求的 TTFT/TPOT 求均值
if ttft_tpot_rows:
    ttft_df = pd.DataFrame(ttft_tpot_rows).set_index("second").sort_index()
    ttft_agg = ttft_df.groupby(ttft_df.index).agg(
        avg_TTFT_ms=("TTFT_ms", "mean"),
        avg_TPOT_ms=("TPOT_ms", "mean"),
        count=("TTFT_ms", "size"),
    ).sort_index()
else:
    ttft_agg = pd.DataFrame(columns=["avg_TTFT_ms", "avg_TPOT_ms", "count"])

# ========= 实例名 =========
inst_names = sorted(per_inst_proc.keys(), key=lambda x: int(x.split("#")[1])) if per_inst_proc else []

# ========= 绘制：每实例一张图（1 子图）=========
for inst in inst_names:
    proc_df = per_inst_proc.get(inst, pd.DataFrame())
    s_decode_lat = proc_df.get("decode_latency_avg")
    s_prefill_ratio = proc_df.get("prefill_ratio")
    s_decode_cnt = proc_df.get("decode_count")

    # Only use processed log data for plotting
    idx_list = [s.index for s in [s_decode_lat, s_prefill_ratio, s_decode_cnt] if s is not None and not s.empty]
    if not idx_list:
        continue
    time_index = idx_list[0]
    for idx in idx_list[1:]:
        time_index = time_index.union(idx)
    time_index = time_index.sort_values()

    # Create a single panel for each instance with prefill/decode metrics
    fig, ax = plt.subplots(1, 1, figsize=(14, 6))

    # ----- Panel: salmon background + decode_latency(left) + decode_count(right)
    if s_prefill_ratio is not None and not s_prefill_ratio.empty:
        for t in s_prefill_ratio.index:
            ratio = s_prefill_ratio.loc[t]
            if pd.notna(ratio):
                ax.axvspan(t, t + pd.Timedelta(seconds=1), color=salmon_cmap(ratio), alpha=1.0)

    left_line = right_line = None
    if s_decode_lat is not None and not s_decode_lat.empty:
        left_line, = ax.plot(s_decode_lat.index, s_decode_lat.values, linewidth=LW_MAIN,
                             label="Avg Decode Latency (left)", color=COLOR_DECODE_LAT)
        ax.set_ylabel("Decode Latency (ms)")

    ax_r = ax.twinx()
    if s_decode_cnt is not None and not s_decode_cnt.empty:
        right_line, = ax_r.plot(s_decode_cnt.index, s_decode_cnt.values, linewidth=LW_MAIN,
                                label="Total Decode Count (right)", color=COLOR_DECODE_COUNT)
        ax_r.set_ylabel("Decode Count (req/s)")

    handles = [mpatches.Patch(color="salmon", label="Prefill Ratio (bg)")]
    if left_line is not None:
        handles.append(Line2D([0], [0], color=COLOR_DECODE_LAT, lw=LW_MAIN, label="Avg Decode Latency (left)"))
    if right_line is not None:
        handles.append(Line2D([0], [0], color=COLOR_DECODE_COUNT, lw=LW_MAIN, label="Decode Count (right)"))
    ax.legend(handles=handles, loc="upper left", fontsize=9, ncol=2)
    set_time_formatter(ax)

    # Ensure y-axis starts from 0 for both left and right axes
    ax.set_ylim(bottom=0)
    ax_r.set_ylim(bottom=0)

    ax.set_xlim(time_index.min(), time_index.max())
    ax.set_xlabel("Time (seconds)")
    fig.suptitle(f"{inst} — Prefill/Decode Metrics")
    fig.autofmt_xdate()
    plt.tight_layout(rect=[0, 0, 1, 0.96])

    out_path = os.path.join(GRAPH_PATH, f"instance_panel_{inst}.png")
    plt.savefig(out_path)
    print(f"Saved per-instance panel: {out_path}")
    plt.close(fig)

# ========= Total 汇总 =========
def to_wide_from_per_inst(colname):
    cols = {}
    for inst in inst_names:
        df = per_inst_proc.get(inst, pd.DataFrame())
        s = df.get(colname) if not df.empty else None
        if s is not None and not s.empty:
            cols[inst] = s.rename(inst)
    if not cols: return pd.DataFrame()
    return pd.concat(cols.values(), axis=1).sort_index()

decode_latency_wide = to_wide_from_per_inst("decode_latency_avg")
decode_count_wide   = to_wide_from_per_inst("decode_count")
prefill_ratio_wide  = to_wide_from_per_inst("prefill_ratio")

def safe_sum(df):  return df.sum(axis=1, min_count=1) if not df.empty else pd.Series(dtype=float)
def safe_mean(df): return df.mean(axis=1)            if not df.empty else pd.Series(dtype=float)

# total 指标
all_idx = []
for df in [decode_latency_wide, decode_count_wide, prefill_ratio_wide, ttft_agg]:
    if isinstance(df, pd.DataFrame) and not df.empty:
        all_idx.append(df.index)
    elif isinstance(df, pd.Series) and not df.empty:
        all_idx.append(df.index)
if all_idx:
    idx_union = all_idx[0]
    for idx in all_idx[1:]:
        idx_union = idx_union.union(idx)
    idx_union = idx_union.sort_values()
else:
    idx_union = pd.Index([], name="second")

total_df = pd.DataFrame(index=idx_union)
total_df["avg_decode_latency"] = safe_mean(decode_latency_wide).reindex(idx_union)
total_df["total_decode_count"] = safe_sum(decode_count_wide).reindex(idx_union)
total_df["avg_prefill_ratio"]  = safe_mean(prefill_ratio_wide).reindex(idx_union)

# ========= Total 图（2 子图：上合并面板 / 下 TTFT&TPOT（绝对时间，实线细线））=========
fig2, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
ax_top2, ax_ttft = axes  # 上 / 下

# --- 顶部：三文鱼背景( avg_prefill_ratio ) + avg_decode_latency(左) + total_decode_count(右)
if "avg_prefill_ratio" in total_df and total_df["avg_prefill_ratio"].notna().any():
    for t, ratio in total_df["avg_prefill_ratio"].dropna().items():
        ax_top2.axvspan(t, t + pd.Timedelta(seconds=1), color=salmon_cmap(ratio), alpha=1.0)

if "avg_decode_latency" in total_df:
    ax_top2.plot(total_df.index, total_df["avg_decode_latency"], linewidth=LW_MAIN,
                 label="Avg Decode Latency (left)", color=COLOR_DECODE_LAT)
    ax_top2.set_ylabel("Decode Latency (ms)")
ax_top2_r = ax_top2.twinx()
if "total_decode_count" in total_df:
    ax_top2_r.plot(total_df.index, total_df["total_decode_count"], linewidth=LW_MAIN,
                   label="Total Decode Count (right)", color=COLOR_DECODE_COUNT)
    ax_top2_r.set_ylabel("Decode Count (req/s)")

handles2 = [mpatches.Patch(color="salmon", label="Avg Prefill Ratio (bg)"),
            Line2D([0], [0], color=COLOR_DECODE_LAT, lw=LW_MAIN, label="Avg Decode Latency (left)"),
            Line2D([0], [0], color=COLOR_DECODE_COUNT, lw=LW_MAIN, label="Total Decode Count (right)")]
ax_top2.legend(handles=handles2, loc="upper left", fontsize=9, ncol=2)
set_time_formatter(ax_top2)

# Ensure y-axis starts from 0 for both left and right axes in the top panel
ax_top2.set_ylim(bottom=0)
ax_top2_r.set_ylim(bottom=0)

# --- 底部：TTFT/TPOT（使用绝对日志时间；两条实线细线；每秒桶聚合均值）
if not ttft_agg.empty:
    ax_ttft_l = ax_ttft
    ax_ttft_r = ax_ttft.twinx()

    ax_ttft_l.plot(ttft_agg.index, ttft_agg["avg_TTFT_ms"], color=COLOR_TTFT, linewidth=LW_THIN, label="Avg TTFT (ms)")
    ax_ttft_r.plot(ttft_agg.index, ttft_agg["avg_TPOT_ms"], color=COLOR_TPOT, linewidth=LW_THIN, label="Avg TPOT (ms)")

    ax_ttft_l.set_ylabel("Avg TTFT (ms)")
    ax_ttft_r.set_ylabel("Avg TPOT (ms)")
    ax_ttft_l.legend(handles=[
        Line2D([0],[0], color=COLOR_TTFT, lw=LW_THIN, label="Avg TTFT (ms)"),
        Line2D([0],[0], color=COLOR_TPOT, lw=LW_THIN, label="Avg TPOT (ms)")
    ], loc="upper left", fontsize=9)
    set_time_formatter(ax_ttft_l)
    ax_ttft.set_xlabel("Time (seconds)")
    
    # Ensure y-axis starts from 0 for both left and right axes in the bottom panel
    ax_ttft_l.set_ylim(bottom=0)
    ax_ttft_r.set_ylim(bottom=0)
else:
    ax_ttft.text(0.5, 0.5, "No TTFT/TPOT data (or s_time=0 anchor not found)", ha="center", va="center", transform=ax_ttft.transAxes)
    ax_ttft.axis("off")

fig2.suptitle("Total Metrics — Combined Panels (TTFT/TPOT aligned to log time)")
fig2.autofmt_xdate()
plt.tight_layout(rect=[0, 0, 1, 0.97])
out_global = os.path.join(GRAPH_PATH, "global_metrics.png")
plt.savefig(out_global)
print(f"Saved total figure: {out_global}")
print("All done.")
