import re
import argparse
import matplotlib.pyplot as plt
import json
import numpy as np
import os
import tarfile
import sys
# ====== 以下是你原本模板的 import 与配置 ======
import matplotlib.patches as patches
import matplotlib
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.font_manager import FontProperties
import matplotlib.patheffects as pe
import pandas as pd
import matplotlib as mpl
from matplotlib.ticker import MultipleLocator
import matplotlib.gridspec as gridspec
from pyparsing import alphas
import scienceplots
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.ticker import MultipleLocator
from matplotlib.ticker import MaxNLocator
from common import *
from matplotlib import rc

fontsize = 7.0
latex_col = 241.02039
ylabel_sz = fontsize - 1
xlabel_sz = fontsize - 2
yticks_sz = fontsize - 2

mpl.rcParams["hatch.linewidth"] = 0.2
plt.rcParams["lines.markersize"] = 3


def adjust_ax_style(ax: Axes, disable=True):
    """你原本模板的 axis 美化逻辑"""
    if disable:
        ax.tick_params(which="major", length=0, axis="x")
        ax.tick_params(which="minor", length=0, axis="x")
        ax.tick_params(which="minor", length=0, axis="y")
        ax.tick_params(which="major", length=2, axis="y")
    ax.tick_params(axis="y", labelsize=yticks_sz)
    ax.tick_params(axis="y", which="major", pad=3)
    ax.tick_params(axis="x", pad=2)

    ax.yaxis.set_tick_params(width=0.2)

    for s in ax.spines:
        ax.spines[s].set_linewidth(0.1)
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)

    ax.yaxis.grid(color="black", linestyle=(0, (5, 10)), linewidth=0.1, zorder=0)


def parse_predicted_ttft(log_lines):
    pred_pattern = re.compile(
        r"Request_(\d+)\s+estimated ttft:\s*([0-9.]+)\s*ms.*?on\s+Vllm#(\d+)"
    )
    assign_pattern = re.compile(
        r"Assigning Request_(\d+)\s+to Replica#(\d+)"
    )

    preds_all = {}
    assigned_replica = {}

    # === 新增：计数器 ===
    pred_count = 0
    assign_count = 0

    for line in log_lines:
        m_pred = pred_pattern.search(line)
        if m_pred:
            pred_count += 1
            rid = int(m_pred.group(1))
            ttft = float(m_pred.group(2))
            replica_id = int(m_pred.group(3))
            preds_all[(rid, replica_id)] = ttft
            continue

        m_assign = assign_pattern.search(line)
        if m_assign:
            assign_count += 1
            rid = int(m_assign.group(1))
            replica_id = int(m_assign.group(2))
            assigned_replica[rid] = replica_id

    preds = {}
    for rid, rep in assigned_replica.items():
        if (rid, rep) in preds_all:
            preds[rid] = preds_all[(rid, rep)]

    # fallback
    if assigned_replica == {}:
        for (rid, rep), ttft in preds_all.items():
            preds[rid] = ttft

    # === 新增：输出日志匹配数 ===
    print("------------- Log Parse Summary -------------")
    print(f"Estimate lines matched:        {pred_count}")
    print(f"Assign lines matched:          {assign_count}")
    print(f"Predicted TTFT entries saved:  {len(preds)}")
    print("----------------------------------------------")

    return preds



def parse_real_ttft_from_jsonl(path):
    reals = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                obj = json.loads(line.strip())
                if "request_id" in obj and "first_token_time" in obj:
                    reals[int(obj["request_id"])] = float(obj["first_token_time"])
            except:
                pass
    return reals


# =============== 新的 CDF 绘制逻辑（仅 threshold 以上） =================
def collect_errors(preds, reals, threshold):
    """
    返回 real_ttft >= threshold 的 error_ratio 数组
    """
    errors = []

    for rid, real in reals.items():
        if real >= threshold and rid in preds:
            pred = preds[rid]
            err = abs(pred - real) / real
            errors.append(err)

    return np.array(sorted(errors))


# =============== 主绘图：对比 correct vs wrong Impl =================
def plot_two_cdfs(correct_errs, wrong_errs, threshold, fig, spec):
    ax = fig.add_subplot(spec[:, :])

    # CDF 生成
    def plot_cdf(ax, data, label, color):
        if len(data) == 0:
            return
        y = np.arange(1, len(data) + 1) / len(data)
        ax.plot(data, y, label=label, lw=0.6, color=color)

    # 画
    plot_cdf(ax, correct_errs, f"Correct Predictor (n={len(correct_errs)})", "tab:blue")
    plot_cdf(ax, wrong_errs, f"Wrong Predictor (n={len(wrong_errs)})", "tab:red")

    # X 轴固定为 [0,1]
    ax.set_xlim(0, 1)

    ax.set_xlabel("Error ratio = |pred - real| / real", fontsize=xlabel_sz)
    ax.set_ylabel("CDF", fontsize=ylabel_sz)
    ax.legend(frameon=False, fontsize=fontsize - 1)

    adjust_ax_style(ax)

def get_figsize(column_width_pt, wf=1.0, hf=1.0):
    """
    将 LaTeX 列宽（pt）转换成 Matplotlib figure size（英寸）
    """
    inches_per_pt = 1.0 / 72.27   # TeX 的 pt → inch
    fig_width = column_width_pt * inches_per_pt * wf
    fig_height = fig_width * hf
    return (fig_width, fig_height)

def extract(base_path, archive_path):
    if not os.path.isfile(archive_path):
        print(f"Error: archive file not found at {archive_path}", file=sys.stderr)
        sys.exit(1)

    archive_name = archive_path.split(".")[0]

    print(f"Archive dir name {archive_name}")

    if not os.path.isdir(archive_name):
        print(f"Extracting {archive_path} to {archive_name}...")
        with tarfile.open(archive_path, 'r:gz') as tar:
            tar.extractall(path=base_path)

        print("Extraction completed.")




# =========================== 主入口 ===============================
def parse_accuracy(threshold = 200):
    work_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    base_path = os.path.join(work_dir, 'data')

    correct_path = f"{base_path}/20251207124818_join-shortest-q-ttft_sc5.6-toC-cpsize4096-u0.9-qwen30b-right-parameter"
    wrong_path = f"{base_path}/20251209185133_join-shortest-q-ttft_sc5.6-toC-cpsize4096-u0.9-qwen30b-wrong-parameter"

    for path in [correct_path, wrong_path]:
        extract(base_path, f"{path}.tgz")


    base1 = correct_path
    base2 = wrong_path

    correct_log = f"{base1}/router_v2.log"
    wrong_log = f"{base2}/router_v2.log"
    
    correct_client = f"{base1}/client.jsonl"
    wrong_client = f"{base2}/client.jsonl"

    # 载入两套 TTFT
    with open(correct_log, "r") as f:
        correct_preds = parse_predicted_ttft(f.readlines())
    with open(correct_client, "r") as f:
        correct_reals = parse_real_ttft_from_jsonl(correct_client)

    with open(wrong_log, "r") as f:
        wrong_preds = parse_predicted_ttft(f.readlines())
    wrong_reals = parse_real_ttft_from_jsonl(wrong_client)

    # 提取 threshold 以上的请求误差
    correct_errs = collect_errors(correct_preds, correct_reals, threshold)
    wrong_errs = collect_errors(wrong_preds, wrong_reals, threshold)

    return correct_errs, wrong_errs
    # print("\n========== Final Data Summary ==========")
    # print(f"Correct: raw pred={len(correct_preds)}, real={len(correct_reals)}, used_in_cdf={len(correct_errs)}")
    # print(f"Wrong  : raw pred={len(wrong_preds)},  real={len(wrong_reals)},  used_in_cdf={len(wrong_errs)}")
    # print("========================================\n")
    # # ======== 进入你原来模板的绘图环境 ========
    # with plt.style.context(["science", "high-vis", "no-latex"]):
    #     fig = plt.figure(constrained_layout=False)
    #     spec = gridspec.GridSpec(ncols=6, nrows=1, figure=fig)

    #     fig.set_size_inches(get_figsize(latex_col, wf=2.1, hf=0.28))
    #     fig.subplots_adjust(wspace=0.25, hspace=0.20)

    #     # 绘制 CDF
    #     plot_two_cdfs(correct_errs, wrong_errs, threshold, fig, spec)

    #     # 输出 PDF
    #     fig.savefig("ttft_compare_cdf.pdf", dpi=1000, bbox_inches="tight")


if __name__ == "__main__":
    parse_accuracy(threshold=0)
