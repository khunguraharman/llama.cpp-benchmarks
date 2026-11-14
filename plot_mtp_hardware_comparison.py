"""Compare M4 and RTX 3080 weighted throughput in one two-panel figure."""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

from summarize_mtp_throughput import ROOT, summarize
from plot_localai_style import (
    BACKGROUND, HEADING, MUTED, TEXT, SERIES_COLORS, _font,
    apply_dashboard_style, set_dashboard_title,
)


def main(*, input_dirs=None, output_subdir="mtp", description=__doc__, method=None):
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--bench", choices=("throughput_1k", "throughput_2k"), default="throughput_1k")
    parser.add_argument("--osl", type=int, default=1024)
    parser.add_argument("--output-dir", type=Path,
                        default=ROOT / "benchmark_plots/hardware-comparison" / output_subdir)
    args = parser.parse_args()
    if input_dirs is None:
        input_dirs = (
            "results/nvidia-rtx-3080/mtp-speed-bench-results",
            "results/m4-mac-mini/mtp/mtp-speed-bench-results",
        )
    for name in ("DMSans-static.ttf", "SpaceGrotesk-static.ttf"):
        font_manager.fontManager.addfont(str(ROOT / "benchmark_plots/fonts" / name))
    plt.rcParams.update({
        "font.family": _font("DM Sans"), "font.size": 10,
        "text.color": TEXT, "figure.facecolor": BACKGROUND,
        "savefig.facecolor": BACKGROUND, "lines.linewidth": 2,
        "lines.markersize": 5,
    })
    fig, axes = plt.subplots(1, 2, figsize=(11, 5.4), sharex=True)
    for ax, label, directory in zip(axes, ("RTX 3080", "M4 Mac mini"), input_dirs):
        rows = [r for r in summarize(ROOT / directory)
                if r["bench"] == args.bench and r["osl"] == args.osl]
        if not rows:
            raise ValueError(f"No matching results for {label}")
        apply_dashboard_style(fig, ax)
        set_dashboard_title(ax, label)
        for entropy, title, color, marker, linestyle in (
            ("low_entropy", "Low entropy", SERIES_COLORS[0], "o", "-"),
            ("high_entropy", "High entropy", SERIES_COLORS[1], "s", "--"),
        ):
            series = sorted((r for r in rows if r["entropy"] == entropy), key=lambda r: r["n_predict"])
            ax.errorbar([r["n_predict"] for r in series], [r["mean"] for r in series],
                        yerr=[r["sd"] for r in series], color=color, marker=marker,
                        linestyle=linestyle, capsize=3, alpha=0.6, label=title)
        lower = min(r["mean"] - r["sd"] for r in rows)
        upper = max(r["mean"] + r["sd"] for r in rows)
        padding = max((upper - lower) * .1, 1)
        ax.set_ylim(max(0, lower - padding), upper + padding)
        ax.set_xticks(sorted({r["n_predict"] for r in rows}))
        ax.set_xlabel("Max Predicted Tokens", labelpad=8)
        ax.set_ylabel("Effective Throughput [tokens/s]", labelpad=8)
    title = "Effective Throughput vs. Max Predicted Tokens"
    if method:
        title = f"{method} {title}"
    fig.suptitle(title, x=.08, y=.98, ha="left",
                 fontsize=18, color=HEADING, fontfamily=_font("Space Grotesk"), fontweight="medium")
    fig.text(.08, .875, f"{args.bench}  |  OSL {args.osl}", fontsize=10, color=MUTED)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right", bbox_to_anchor=(.98, .915),
               ncol=2, frameon=False, handlelength=2.4, labelcolor=TEXT)
    fig.subplots_adjust(left=.08, right=.975, bottom=.17, top=.72, wspace=.28)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    prefix = f"{method.lower()}_" if method else ""
    stem = args.output_dir / f"{prefix}weighted_throughput_{args.bench}_osl{args.osl}_m4_vs_rtx3080"
    for suffix in (".png", ".svg"):
        path = stem.with_suffix(suffix)
        fig.savefig(path, dpi=200)
        print(path.resolve())
    plt.close(fig)


if __name__ == "__main__":
    main()
