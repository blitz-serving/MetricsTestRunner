from paper_fmt import plot_null_figure, plot_ttft_cdf, plot_tpot_cdf, plot_parameterization
from paper_fmt import load_jsonl_data, extract_metrics
from paper_fmt import get_figsize, get_parameterization_data
from paper_fmt import latex_col, gridspec, plt

import argparse
import os


def main(
    fst_name,
    fst_path,
    snd_name,
    snd_path,
    earliest_timestamp,
    latest_timestamp,
    base_name,
    base_path,
):
    # hold all metrics
    all_ttft = {}
    all_tpot = {}

    # ==== Load Data ====
    for name, d in [(fst_name, fst_path), (snd_name, snd_path)]:
        jsonl_path = os.path.join(d, "client.jsonl")
        print(f"[Loading] {name} from {jsonl_path}")
        raw = load_jsonl_data(jsonl_path)
        metrics = extract_metrics(raw)

        all_ttft[name] = metrics["ttft"]
        all_tpot[name] = metrics["tpot"]

        print(f"  {name}: TTFT={len(metrics['ttft'])}, TPOT={len(metrics['tpot'])}")

    # ==== Begin plotting ====
    with plt.style.context(["science", "high-vis", "no-latex"]):
        plt.rcParams["lines.markersize"] = 3
        plt.rcParams["font.family"] = "sans-serif"

        fig = plt.figure(constrained_layout=False)
        spec = gridspec.GridSpec(
            ncols=7,
            nrows=1,
            figure=fig,
            width_ratios=[0.8, 0.00, 0.8, 0.00, 0.6, 0.00, 0.6],
        )

        # ---------------- TTFT CDF ----------------
        plot_ttft_cdf(
            fig,
            spec[0, 0],
            a_name=fst_name,
            b_name=snd_name,
            a_ttft=all_ttft[fst_name],
            a_plus_b_ttft=all_ttft[snd_name],
            xlabel="TTFT (ms)",
            ylabel=True,
        )
        plot_null_figure(fig, spec[0, 1])

        # ---------------- TPOT CDF ----------------
        plot_tpot_cdf(
            fig,
            spec[0, 2],
            fst_name,
            snd_name,
            a_tpot=all_tpot[fst_name],
            a_plus_b_tpot=all_tpot[snd_name],
            xlabel="TPOT (ms)",
            ylabel=False,
        )
        plot_null_figure(fig, spec[0, 3])

        # --------- TTFT Parameterized cmp ---------
        data_list = get_parameterization_data(
            base_name, base_path, earliest_timestamp, latest_timestamp
        )
        plot_parameterization(
            fig,
            spec[0, 4],
            data_list['ttft'],
            xlabel='λ',
            ylabel='TTFT (ms)',
            color_set='red'
        )
        plot_null_figure(fig, spec[0, 5])

        # --------- TPOT Parameterized cmp ---------
        plot_parameterization(
            fig,
            spec[0, 6],
            data_list['tpot'],
            xlabel='λ',
            ylabel='TPOT (ms)',
            color_set='blue'
        )

        fig.set_size_inches(get_figsize(latex_col, wf=2.1, hf=0.2))
        fig.subplots_adjust(wspace=0.4, hspace=0.22)
        out_path = os.path.splitext(__file__)[0] + ".pdf"
        fig.savefig(
            out_path,
            dpi=1000,
            format="pdf",
            bbox_inches="tight",
        )
        print(f"[Saved] {out_path}")

        plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Plot A + B')
    parser.add_argument('--fst-name', type=str, default='A')
    parser.add_argument('--fst-path', type=str, required=True, 
                       help='data of A, contains client.jsonl')
    parser.add_argument('--snd-name', type=str, default='B')
    parser.add_argument('--snd-path', type=str, required=True,
                       help='data of A, contains client.jsonl')

    # Plot parameterization
    parser.add_argument('--base-path', type=str,
                       help='Directory contains all parameterization data subdirectories')
    parser.add_argument('--base-name', type=str,
                       help='Infix of each subdirectories')
    parser.add_argument('--earliest-timestamp', type=str,
                       help='Earliest prefix of all subdirectories')
    parser.add_argument('--latest-timestamp', type=str,
                       help='Latest prefix of all subdirectories')

    args = parser.parse_args()
    # Just for demo
    # base_path = "5.0_batch1024_u0.9_bailian_hybrid_lwl_filtered"
    # base_name = "bailian-impl-lwl"
    # fst_path = f"{base_path}/20251117193347_least-wait-token-bs"
    # snd_path = f"{base_path}/20251117181949_dynamo-deterministic"
    # earliest_timestamp = "20251117200136"
    # latest_timestamp = "20251118012726"
    main(
        args.fst_name,
        args.fst_path,
        args.snd_name,
        args.snd_path,
        args.earliest_timestamp,
        args.latest_timestamp,
        args.base_name,
        args.base_path,
    )
