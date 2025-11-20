import json
import math
import numpy as np
import os
import pandas as pd
import re

import matplotlib.dates as mdates
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt

from matplotlib.ticker import MultipleLocator
from matplotlib.ticker import MaxNLocator
from matplotlib.legend_handler import HandlerBase
from matplotlib.dates import DateFormatter
from matplotlib.font_manager import FontProperties

from matplotlib import gridspec
from matplotlib import rc

from typing import List, Dict, Optional

fontsize = 5.5
latex_col = 241.02039  ## pt
marker_sz = 2.0
M = 1024 * 1024
read_epoch_num = 1000
linewidth = 0.5
leg_font = fontsize - 0.5
markeredgewidth_sz = 0.5

ylabel_sz = fontsize - 0.5
xlabel_sz = fontsize - 0.5
xtick_sz = fontsize - 0.5

title_font = FontProperties(family="sans-serif", weight="bold", size=xlabel_sz)

import matplotlib.font_manager as fm
import pkg_resources

scicenplot_pkg = pkg_resources.get_distribution("scienceplots")
if int(scicenplot_pkg.version.split(".")[0]) >= 2:
    print("scienceplots version:", scicenplot_pkg.version)
    import scienceplots

font = fm.FontProperties(fname="/System/Library/Fonts/Helvetica.ttc")
con_font = fm.FontProperties(fname="/System/Library/Fonts/Courier.ttc")
supply_font = fm.FontProperties(fname="./SimHei.ttf")

# color_list = ["#E24A33","#348ABD","#988ED5","#777777","#FBC15E","#8EBA42","#FFB5B8"]
color_list = [
    "#268BD2",
    "#2AA198",
    "#859900",
    "#B58900",
    "#CB4B16",
    "#DC322F",
    "#D33682",
    "#6C71C4",
]
bar_color_list = [
    "#F7FBFF",
    "#DEEBF7",
    "#C6DBEF",
    "#9ECAE1",
    "#6BAED6",
    "#4292C6",
    "#2171B5",
    "#084594",
]
color_map = {"dark_red": "#d73027", "dark_orange": "#fc8d59", "dark_blue": "#4575b4"}
golden_ratio = 0.618


def sumzip(*items):
    return [sum(values) for values in zip(*items)]


def set_axis_formats(axises, fontsize):
    for ax in axises:
        for axis in ["top", "bottom", "left", "right"]:
            ax.spines[axis].set_linewidth(0.3)
        ax.tick_params(which="major", width=0.3, length=1.5)
        ax.tick_params(which="minor", width=0)
        ax.tick_params(axis="x", labelsize=fontsize)
        ax.tick_params(axis="y", labelsize=fontsize)


def fixed_aspect_ratio(ratio, ax):
    """
    Set a fixed aspect ratio on matplotlib plots
    regardless of axis units
    """
    xvals, yvals = ax.axes.get_xlim(), ax.axes.get_ylim()

    xrange = xvals[1] - xvals[0]
    yrange = yvals[1] - yvals[0]
    ax.set_aspect(ratio * (xrange / yrange), adjustable="box")


def scale_sz(width, origin):
    w, h = origin
    return (width, h * width / w)


def mult_ratio(orign, ratio):
    x, y = orign
    rx, ry = ratio
    return (x * rx, y * ry)


def reverse_legends(leg, render):
    vp = leg._legend_box._children[-1]._children[0]
    for c in vp._children:
        c._children.reverse()
    vp.align = "right"


def transfrom_ticks(x, tick_vals):
    # count from 1 to n
    # result value [0, x]
    n = len(tick_vals)

    anchor = [num + 1 for num in range(n)]  # 1 ~ n
    res = []

    for item in x:
        for i in range(n):
            tick = tick_vals[i]
            if item < tick_vals[i]:
                base = i
                if i == 0:
                    res.append(0 + item / tick)
                else:
                    pre_tick = tick_vals[i - 1]
                    res.append(base + (item - pre_tick) / (tick - pre_tick))
                break
            elif item == tick_vals[i]:
                res.append(i + 1)
                break
            elif item > tick_vals[i]:
                base = i + 1
                if i == n - 1:  # no bigger
                    res.append(base + abs(item - tick_vals[i]) / (tick_vals[i]))
                    break
                else:
                    pass  # next one
    return res


def get_figsize(
    columnwidth,
    wf=0.5,
    hf=(5.0**0.5 - 1.0) / 2.0,
):
    r"""
    Parameters:
    - wf [float]:  width fraction in columnwidth units
    - hf [float]:  height fraction in columnwidth units.
                       Set by default to golden ratio.
    - columnwidth [float]: width of the column in latex. Get this from LaTeX
                               using \showthe\columnwidth
    Returns:  [fig_width,fig_height]: that should be given to matplotlib
    """
    fig_width_pt = columnwidth * wf
    inches_per_pt = 1.0 / 72.27  # Convert pt to inch
    fig_width = fig_width_pt * inches_per_pt  # width in inches
    #    fig_height = fig_width*hf      # height in inches
    fig_height = columnwidth * hf * inches_per_pt
    return [fig_width, fig_height]


def cdf_transform(data):
    """
    return back the (x,y) of cdf graph
    :param data:
    :return:
    """
    data_sorted = np.sort(data)

    # calculate the proportional values of samples
    p = 1.0 * np.arange(len(data)) / (len(data) - 1)
    return data_sorted, p


def get_data(name, file, entries):
    sheet_name = name

    res = pd.read_excel(file, sheet_name=sheet_name, na_filter=False, nrows=entries)

    # Without NaN if all rows have NaN
    res = res.dropna(
        axis=0,
        how="all",
        #        inplace=True,
    )
    return res


def adjust_ax_style(ax, disable=True):
    ax.tick_params(axis="y", labelsize=fontsize)
    ax.tick_params(axis="y", which="major", pad=2)
    ax.yaxis.set_tick_params(width=0.2)

    ax.tick_params(axis="x", labelsize=xtick_sz)

    ax.tick_params(which="major", length=0, axis="x", pad=2, labelcolor="black")
    ax.tick_params(which="minor", length=0, axis="x")
    ax.tick_params(which="major", length=2, axis="y", pad=2, labelcolor="black")
    ax.tick_params(which="minor", length=0, axis="y")

    for s in ax.spines:
        ax.spines[s].set_linewidth(0.1)

    ax.minorticks_off()
    ax.yaxis.grid(color="grey", linestyle=(0, (5, 10)), linewidth=0.1, zorder=0)
    ax.xaxis.grid(color="grey", linestyle=(0, (5, 10)), linewidth=0.1, zorder=0)


def plot_ttft_cdf(
    fig, spec, a_name, b_name, a_ttft, a_plus_b_ttft, xlabel, ylabel=True
):
    ax = fig.add_subplot(spec)

    a_ttft = np.sort(a_ttft)
    a_plus_b_ttft = np.sort(a_plus_b_ttft)

    a_cdf = np.arange(1, len(a_ttft) + 1) / len(a_ttft)
    a_plus_b_cdf = np.arange(1, len(a_plus_b_ttft) + 1) / len(a_plus_b_ttft)
    ax.plot(
        a_ttft,
        a_cdf,
        marker=".",
        linestyle="-",
        markersize=0,
        color=color_map["dark_red"],
        linewidth=linewidth,
        markeredgewidth=0,
        label=a_name,
        zorder=2,
    )
    ax.plot(
        a_plus_b_ttft,
        a_plus_b_cdf,
        marker=".",
        linestyle="-",
        markersize=0,
        color=color_map["dark_blue"],
        linewidth=linewidth,
        markeredgewidth=0,
        label=f"{a_name}+{b_name}",
        zorder=1,
    )
    if ylabel:
        ax.set_ylabel("Percentage", fontsize=ylabel_sz, labelpad=1)
    ax.set_xlabel(f"{xlabel}", fontsize=xlabel_sz, labelpad=0)

    ax.tick_params(axis="x", labelsize=ylabel_sz)
    ax.tick_params(axis="y", labelsize=xtick_sz)

    adjust_ax_style(ax)
    return


def plot_tpot_cdf(
    fig, spec, a_name, b_name, a_tpot, a_plus_b_tpot, xlabel, ylabel=True
):
    ax = fig.add_subplot(spec)

    a_tpot = np.sort(a_tpot)
    a_plus_b_tpot = np.sort(a_plus_b_tpot)
    a_cdf = np.arange(1, len(a_tpot) + 1) / len(a_tpot)
    a_plus_b_cdf = np.arange(1, len(a_plus_b_tpot) + 1) / len(a_plus_b_tpot)

    ax.plot(
        a_tpot,
        a_cdf,
        marker=".",
        linestyle="-",
        markersize=0,
        color=color_map["dark_red"],
        linewidth=linewidth,
        markeredgewidth=0,
        label=a_name,
        zorder=2,
    )
    ax.plot(
        a_plus_b_tpot,
        a_plus_b_cdf,
        marker=".",
        linestyle="-",
        markersize=0,
        color=color_map["dark_blue"],
        linewidth=linewidth,
        markeredgewidth=0,
        label=f"{a_name}+{b_name}",
        zorder=1,
    )
    if ylabel:
        ax.set_ylabel("Percentage", fontsize=ylabel_sz, labelpad=1)
    ax.set_xlabel(f"{xlabel}", fontsize=xlabel_sz, labelpad=0)

    ax.tick_params(axis="x", labelsize=ylabel_sz)
    ax.tick_params(axis="y", labelsize=xtick_sz)

    adjust_ax_style(ax)
    return


def plot_slo_scale(fig, spec, baseline_df, blitz_df, tbt_key, feat, ttft_slo, tbt_slo):
    ax = fig.add_subplot(spec)

    baseline_ttft_slo = []
    blitz_ttft_slo = []
    baseline_tbt_slo = []
    blitz_tbt_slo = []
    scale = [1.6, 1.4, 1.2, 1.0, 0.8, 0.6, 0.4]
    # scale = [0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6]

    # tbt_key = 'max_time_between_tokens'
    # tbt_key = 'p90_time_between_tokens'
    # tbt_key = 'avg_time_between_tokens'
    assert (
        tbt_key == "max_time_between_tokens"
        or tbt_key == "p90_time_between_tokens"
        or tbt_key == "avg_time_between_tokens"
    )

    for s in scale:
        baseline_atta = baseline_df["first_token_time"] < (ttft_slo * s)
        blitz_atta = blitz_df["first_token_time"] < (ttft_slo * s)

        baseline = baseline_df[baseline_atta].shape[0] / baseline_df.shape[0]
        blitz = blitz_df[blitz_atta].shape[0] / blitz_df.shape[0]

        baseline_ttft_slo.append(baseline * 100)
        blitz_ttft_slo.append(blitz * 100)

        baseline_atta = baseline_df[tbt_key].astype(int) < (tbt_slo * s)
        blitz_atta = blitz_df[tbt_key].astype(int) < (tbt_slo * s)

        baseline = baseline_df[baseline_atta].shape[0] / baseline_df.shape[0]
        blitz = blitz_df[blitz_atta].shape[0] / blitz_df.shape[0]

        baseline_tbt_slo.append(baseline * 100)
        blitz_tbt_slo.append(blitz * 100)

    ax.plot(
        scale,
        baseline_ttft_slo,
        marker=".",
        linestyle="-",
        label="S-LLM-ttft",
        color=color_map["dark_red"],
        markersize=marker_sz,
        linewidth=linewidth,
        markeredgewidth=markeredgewidth_sz,
        zorder=1024,
    )
    ax.plot(
        scale,
        blitz_ttft_slo,
        marker=".",
        linestyle="-",
        label="BlitzScale-ttft",
        color=color_map["dark_blue"],
        markersize=marker_sz,
        linewidth=linewidth,
        markeredgewidth=markeredgewidth_sz,
        zorder=1024,
    )
    ax.plot(
        scale,
        baseline_tbt_slo,
        marker=".",
        linestyle="--",
        label="S-LLM-tbt",
        color=color_map["dark_red"],
        markersize=marker_sz,
        linewidth=linewidth,
        markeredgewidth=markeredgewidth_sz,
        zorder=1024,
    )
    ax.plot(
        scale,
        blitz_tbt_slo,
        marker=".",
        linestyle="--",
        label="BlitzScale-tbt",
        color=color_map["dark_blue"],
        markersize=marker_sz,
        linewidth=linewidth,
        markeredgewidth=markeredgewidth_sz,
        zorder=1024,
    )
    ax.set_ylim(0, 100)

    ax.set_ylabel("SLO attainment", fontsize=ylabel_sz, labelpad=1)
    ax.set_xlabel("SLO Scale", fontsize=xlabel_sz, labelpad=0)

    ax.tick_params(axis="x", labelsize=ylabel_sz)
    ax.tick_params(axis="y", labelsize=xtick_sz)

    ax.xaxis.set_major_locator(MaxNLocator(nbins=5))

    ax.legend(
        fontsize=xlabel_sz,
        frameon=False,
        loc="best",
        handlelength=0.5,
        ncol=1,
        facecolor="white",
        edgecolor="white",
        framealpha=1,
    )

    adjust_ax_style(ax)
    return


def load_jsonl_data(file_path: str) -> List[Dict]:
    data = []
    try:
        with open(file_path, "r") as f:
            line_num = 0
            skip_line_num = 0
            for line in f:
                line_num += 1
                if line.strip():
                    try:
                        record = json.loads(line.strip())
                        status = record.get("status", "")
                        if status == "200" or status == "timeout":
                            data.append(record)
                        else:
                            skip_line_num += 1
                    except json.JSONDecodeError as e:
                        print(f"Warning: Skipping invalid JSON on line {line_num}: {e}")
                        continue
            if skip_line_num > 0:
                print(f"Warning: skipped {skip_line_num} requests (not 200 or timeout)")
    except FileNotFoundError:
        print(f"Error: File {file_path} not found")
        sys.exit(1)
    return data


def extract_metrics(data: List[Dict]) -> Dict[str, np.ndarray]:
    metrics = {"rid": [], "ttft": [], "tpot": []}
    for item in data:
        rid = str(item.get("request_id", ""))
        metrics["rid"].append(rid)
        status = item.get("status")
        if status == "200":
            ttft = int(item.get("first_token_time", 0))
            tpot = int(item.get("avg_time_between_tokens", 0))
            metrics["ttft"].append(ttft)
            metrics["tpot"].append(tpot)
        elif status == "timeout":
            metrics["ttft"].append(-1)
            metrics["tpot"].append(-1)

    # Convert to numpy, filter out -1 (timeouts)
    ttft_arr = np.array(metrics["ttft"])
    tpot_arr = np.array(metrics["tpot"])
    valid_ttft = ttft_arr[ttft_arr >= 0]
    valid_tpot = tpot_arr[tpot_arr >= 0]

    return {"ttft": valid_ttft, "tpot": valid_tpot}


def plot_null_figure(fig, spec):
    ax = fig.add_subplot(spec)
    ax.axis("off")


def get_parameterization_data(
    base_name, base_path, earliest_timestamp, latest_timestamp
):
    pp = r"^(\d+)_([A-Za-z0-9\-]+)-(\d+)"
    ret_all_data = {"ttft": [], "tpot": []}

    def is_parameterizd(target_path: str) -> Optional[str]:
        mm = re.match(pp, target_path)
        if mm:
            ts, name, idx = mm.groups()
            if int(ts) < int(earliest_timestamp) or int(ts) > int(latest_timestamp):
                return None
            if name != base_name:
                return None
            return idx
        else:
            return None

    for entry in os.listdir(base_path):
        idx = is_parameterizd(entry)
        if idx is not None:
            print(f"Find parameterized test entry {entry}")
            file_path = os.path.join(base_path, entry, "client.jsonl")
            data = extract_metrics(load_jsonl_data(file_path))
            for k, v in data.items():
                ret_all_data[k].append(v)

    return ret_all_data


def plot_parameterization(fig, spec, data_list, xlabel, ylabel, color_set):
    color_sublist = []
    if color_set == "red":
        color_sublist = [color_list[5], color_list[6], color_list[3]]
    elif color_set == "blue":
        color_sublist = [color_list[0], color_list[1], color_list[-1]]
    else:
        raise ValueError("Invalid color set! Must be either 'red' or 'blue'")

    ax = fig.add_subplot(spec)

    idx = np.arange(len(data_list))

    vv = {"avg": [], "p90": [], "p99": []}

    fn_list = {
        "avg": lambda x: np.mean(x),
        "p90": lambda x: np.percentile(x, 90),
        "p99": lambda x: np.percentile(x, 99),
    }

    for arr in data_list:
        for fn, v in vv.items():
            v.append(fn_list[fn](arr))

    for v, c in zip(vv.values(), color_sublist):
        ax.plot(
            idx,
            v,
            "-",
            color=c,
            marker="o",
            markerfacecolor="none",
            linewidth=0.6,
            markersize=2,
            markeredgewidth=0.5,
        )

    plt.text(
        x=0, y=-0, s="metric-A", ha="left", va="top", transform=ax.transAxes, fontsize=4
    )
    plt.text(
        x=1,
        y=-0,
        s="metric-B",
        ha="right",
        va="top",
        transform=ax.transAxes,
        fontsize=4,
    )
    ax.set_ylabel(f"{ylabel}", fontsize=xlabel_sz, labelpad=0)
    ax.tick_params(axis="x", labelsize=ylabel_sz)
    ax.set_xticks([])
    # ax.tick_params(axis="y", labelsize=xtick_sz)

    adjust_ax_style(ax)


if __name__ == "__main__":
    prefix = "e2e"

    eval_case = ["baseline", "blitz"]
    feat = "mean"

    with plt.style.context(["science", "high-vis", "no-latex"]):
        plt.rcParams["lines.markersize"] = 3
        #        plt.rc('text', usetex=True)
        plt.rcParams["font.family"] = font.get_name()

        client_log_path = f"./{prefix}/baseline/client.3.jsonl"
        sllm_data = prepare_data(
            client_log_path,
            zoom_out_millis_min=69.5 * 1000,
            zoom_out_millis_max=110 * 1000,
        )
        client_log_path = f"./{prefix}/blitz/client.3.jsonl"
        blitz_data = prepare_data(
            client_log_path,
            zoom_out_millis_min=69.5 * 1000,
            zoom_out_millis_max=110 * 1000,
        )

        fig = plt.figure(constrained_layout=False)
        spec = gridspec.GridSpec(
            ncols=10,
            nrows=1,
            figure=fig,
            width_ratios=[0.8, 0.05, 0.9, 0.05, 0.9, 0.05, 0.5, 0.5, 0.05, 0.5],
        )

        plot_request(fig, spec[0, 0], sllm_data, "BurstGPT")
        plot_null_figure(fig, spec[0, 1])
        plot(
            fig,
            spec[0, 2],
            baseline_df=sllm_data,
            blitz_df=blitz_data,
            watermark=300,
            feat=feat,
            ylabel="TTFT",
        )
        plot_null_figure(fig, spec[0, 3])

        plot(
            fig,
            spec[0, 4],
            baseline_df=sllm_data,
            blitz_df=blitz_data,
            watermark=300,
            feat=feat,
            ylabel="TBT",
            leg_start=1,
        )
        plot_null_figure(fig, spec[0, 5])

        plot_cdf(
            fig,
            spec[0, 6],
            baseline_df=sllm_data,
            blitz_df=blitz_data,
            feat=None,
            xlabel="TTFT (ms)",
        )
        plot_cdf(
            fig,
            spec[0, 7],
            baseline_df=sllm_data,
            blitz_df=blitz_data,
            feat=None,
            xlabel="TBT (ms)",
            ylabel=False,
        )

        plot_null_figure(fig, spec[0, 8])
        plot_slo_scale(
            fig, spec[0, 9], baseline_df=sllm_data, blitz_df=blitz_data, feat=None
        )

        fig.set_size_inches(get_figsize(latex_col, wf=2.1, hf=0.2))
        fig.subplots_adjust(wspace=0.35, hspace=0.22)

        fig.savefig(
            os.path.splitext(__file__)[0] + ".pdf",
            dpi=1000,
            format="pdf",
            bbox_inches="tight",
        )
