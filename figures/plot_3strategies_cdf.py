import os
import sys
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from matplotlib.ticker import MaxNLocator, FuncFormatter
from matplotlib import gridspec
from typing import List, Dict
import scienceplots  # ←←← 必须加这一行！

# ----------------------------
# Global Configuration Flags
# ----------------------------
ENABLE_ZOOM_IN = True  # Set to False to disable zoom-in inset
ZOOM_IN_PERCENTILE = 90  # Zoom into top X% (e.g., 90 means show 0 ~ p90)

fontsize = 8
marker_sz = 2.0
linewidth = 0.5
leg_font = fontsize - 0.5
markeredgewidth_sz = 0.5
ylabel_sz = fontsize - 0.5
xlabel_sz = fontsize - 0.5
xtick_sz = fontsize - 1

# Color and marker mapping for strategies
STRATEGY_STYLE = {
    "dynamo": {"color": "red", "marker": "o"},
    "lwl-bs-tuple": {"color": "blue", "marker": "s"},
    "lwl-bs-linear": {"color": "green", "marker": "^"},
}

def adjust_ax_style(ax):
    ax.tick_params(axis="y", labelsize=xtick_sz)
    ax.tick_params(axis="y", which="major", pad=2)
    ax.yaxis.set_tick_params(width=0.2)

    ax.tick_params(axis="x", labelsize=xtick_sz)
    ax.tick_params(which="major", length=0, axis="x", pad=2, labelcolor="black")
    ax.tick_params(which="minor", length=0, axis="x")
    ax.tick_params(which="major", length=2, axis="y", pad=2, labelcolor="black")
    ax.tick_params(which="minor", length=0, axis="y")

    for spine in ax.spines.values():
        spine.set_linewidth(0.1)

    ax.minorticks_off()
    ax.yaxis.grid(color="grey", linestyle=(0, (5, 10)), linewidth=0.1, zorder=0)
    ax.xaxis.grid(color="grey", linestyle=(0, (5, 10)), linewidth=0.1, zorder=0)


def load_jsonl_data(file_path: str) -> List[Dict]:
    data = []
    try:
        with open(file_path, 'r') as f:
            line_num = 0
            skip_line_num = 0
            for line in f:
                line_num += 1
                if line.strip():
                    try:
                        record = json.loads(line.strip())
                        status = record.get('status', '')
                        if status == '200' or status == 'timeout':
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
    metrics = {'rid': [], 'ttft': [], 'tpot': []}
    for item in data:
        rid = str(item.get('request_id', ""))
        metrics['rid'].append(rid)
        status = item.get("status")
        if status == '200':
            ttft = float(item.get('first_token_time', 0))
            tpot = float(item.get('avg_time_between_tokens', 0))
            metrics['ttft'].append(ttft)
            metrics['tpot'].append(tpot)
        elif status == "timeout":
            metrics['ttft'].append(-1)
            metrics['tpot'].append(-1)

    # Convert to numpy, filter out -1 (timeouts)
    ttft_arr = np.array(metrics['ttft'])
    tpot_arr = np.array(metrics['tpot'])
    valid_ttft = ttft_arr[ttft_arr >= 0]
    valid_tpot = tpot_arr[tpot_arr >= 0]

    return {
        "ttft": valid_ttft,
        "tpot": valid_tpot
    }

def plot_cdf_single(ax, data_list, labels, metric_name, ylabel="Cumulative Fraction"):
    """
    metric_name: 'TTFT' or 'TPOT'
    """
    all_valid_data = np.concatenate([d for d in data_list if len(d) > 0])
    if len(all_valid_data) == 0:
        ax.set_visible(False)
        return

    # Determine x-axis ticks: use percentiles for better coverage
    percentiles = [0, 25, 50, 75, 90, 95, 99]
    x_ticks = np.percentile(all_valid_data, percentiles)
    # Remove duplicates and sort
    x_ticks = np.unique(np.round(x_ticks, decimals=2))
    # Keep reasonable number (max 6)
    if len(x_ticks) > 6:
        x_ticks = x_ticks[::max(1, len(x_ticks)//6)][:6]

    for data, label in zip(data_list, labels):
        if len(data) == 0:
            continue
        sorted_data = np.sort(data)
        if len(sorted_data) > 20000:
            step = len(sorted_data) // 20000
            sorted_data = sorted_data[::step]
        cdf = np.arange(1, len(sorted_data) + 1) / len(sorted_data)

        style = STRATEGY_STYLE.get(label, {"color": "black", "marker": "x"})
        ax.plot(
            sorted_data,
            cdf,
            color=style["color"],
            marker=style["marker"],
            markersize=marker_sz * 0.8,          # slightly smaller
            linewidth=0.3,                       # thinner line!
            linestyle="solid",
            markeredgewidth=markeredgewidth_sz * 0.7,
            label=label,
            zorder=1024,
        )

    # Set xlabel as "TTFT (ms)" or "TPOT (ms)"
    ax.set_xlabel(f"{metric_name} (ms)", fontsize=xlabel_sz, labelpad=1)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=ylabel_sz, labelpad=2)

    # Remove title
    # ax.set_title(...)  # 👈 removed

    # === 恢复原始风格：使用 MaxNLocator 控制 x 轴刻度数量 ===
    ax.xaxis.set_major_locator(MaxNLocator(nbins=5, steps=[1, 2, 2.5, 5, 10]))
    ax.set_xlim(left=0)

    # 保持 y 轴不变
    ax.yaxis.set_major_locator(MaxNLocator(nbins=5))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f'{y:.1f}'))

    adjust_ax_style(ax)

    # --- Zoom-in inset (if enabled) ---
    if ENABLE_ZOOM_IN:
        zoom_x_max = np.percentile(all_valid_data, ZOOM_IN_PERCENTILE)
        inset_ax = inset_axes(ax, width="45%", height="30%", loc="lower right",
                              bbox_to_anchor=(0.05, 0.05, 1, 1), bbox_transform=ax.transAxes)
        for data, label in zip(data_list, labels):
            if len(data) == 0:
                continue
            sorted_data = np.sort(data)
            if len(sorted_data) > 20000:
                step = len(sorted_data) // 20000
                sorted_data = sorted_data[::step]
            cdf = np.arange(1, len(sorted_data) + 1) / len(sorted_data)
            style = STRATEGY_STYLE.get(label, {"color": "black", "marker": "x"})
            inset_ax.plot(
                sorted_data,
                cdf,
                color=style["color"],
                marker=style["marker"],
                markersize=marker_sz * 0.5,
                linewidth=0.2,
                linestyle="solid",
                markeredgewidth=markeredgewidth_sz * 0.5,
                zorder=1024,
            )
        inset_ax.set_xlim(0, zoom_x_max)
        inset_ax.xaxis.set_major_locator(MaxNLocator(nbins=3))
        inset_ax.yaxis.set_major_locator(MaxNLocator(nbins=3))
        inset_ax.tick_params(labelsize=xtick_sz - 1)
        for spine in inset_ax.spines.values():
            spine.set_linewidth(0.08)
        inset_ax.minorticks_off()
        inset_ax.yaxis.grid(color="grey", linestyle=(0, (3, 6)), linewidth=0.08, zorder=0)
        inset_ax.xaxis.grid(color="grey", linestyle=(0, (3, 6)), linewidth=0.08, zorder=0)

def save_fig(fig, filename: str, dpi=600):
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    fig.savefig(filename, dpi=dpi, bbox_inches="tight", pad_inches=0.01)


def get_figsize(column_width_pt, wf=1, hf=0.27):
    inches_per_pt = 1.0 / 72.27
    fig_width = column_width_pt * inches_per_pt * wf
    fig_height = fig_width * hf
    return (fig_width, fig_height)


def main():
    parser = argparse.ArgumentParser(description="Plot TTFT and TPOT CDFs for three strategies.")
    parser.add_argument("dynamo_dir", help="Directory containing dynamo/client.jsonl")
    parser.add_argument("tuple_dir", help="Directory containing lwl-bs-tuple/client.jsonl")
    parser.add_argument("linear_dir", help="Directory containing lwl-bs-linear/client.jsonl")
    args = parser.parse_args()

    strategy_dirs = {
        "dynamo": args.dynamo_dir,
        "lwl-bs-tuple": args.tuple_dir,
        "lwl-bs-linear": args.linear_dir
    }

    all_ttft = {}
    all_tpot = {}

    for name, d in strategy_dirs.items():
        jsonl_path = os.path.join(d, "client.jsonl")
        print(f"Loading {name} from {jsonl_path}")
        raw_data = load_jsonl_data(jsonl_path)
        metrics = extract_metrics(raw_data)
        all_ttft[name] = metrics["ttft"]
        all_tpot[name] = metrics["tpot"]
        print(f"  {name}: TTFT count={len(metrics['ttft'])}, TPOT count={len(metrics['tpot'])}")

    # Plot
    with plt.style.context(["science", "high-vis", "no-latex"]):
        plt.rcParams["font.family"] = "sans-serif"
        fig = plt.figure(constrained_layout=False)
        gs = gridspec.GridSpec(ncols=2, nrows=1, figure=fig, width_ratios=[1, 1])
        fig.set_size_inches(get_figsize(241.02039, wf=1, hf=0.27))

        labels = list(strategy_dirs.keys())
        ttft_data = [all_ttft[name] for name in labels]
        tpot_data = [all_tpot[name] for name in labels]

        plot_cdf_single(fig.add_subplot(gs[0, 0]), ttft_data, labels, "TTFT")
        plot_cdf_single(fig.add_subplot(gs[0, 1]), tpot_data, labels, "TPOT", ylabel=False)

        # Legend
        handles, _ = fig.axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="upper center", ncol=3, fontsize=leg_font,
                   bbox_to_anchor=(0.5, 1.02), frameon=True, fancybox=False, shadow=False,
                   columnspacing=1.0, handletextpad=0.3)

        fig.subplots_adjust(top=0.85, bottom=0.2, wspace=0.3)

        # Save
        output_name = "_".join(labels)
        output_dir = os.path.join(os.path.dirname(__file__), "figs", "ttft_tpot_cdf")
        os.makedirs(output_dir, exist_ok=True)
        save_path = os.path.join(output_dir, f"{output_name}.png")
        save_fig(fig, save_path)
        print(f"Saved figure to {save_path}")

        plt.close(fig)


if __name__ == "__main__":
    main()

# python plot_3strategies_cdf.py /mnt/debugger/hjb/node1/lmmetric-logs/5.0_batch1024_u0.9_bailian_hybrid_lwl/20251117181949_dynamo-deterministic /mnt/debugger/hjb/node1/lmmetric-logs/5.0_batch1024_u0.9_bailian_hybrid_lwl/20251117171110_least-wait-token-bs /mnt/debugger/hjb/node1/lmmetric-logs/5.0_batch1024_u0.9_1118_lwl_bs_linear/20251119002601_least-wait-token-bs-linear-05
