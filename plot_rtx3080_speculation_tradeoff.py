"""Compare throughput and peak VRAM at the fastest tested n_predict."""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.ticker import PercentFormatter

from plot_localai_style import (
    HEADING, MUTED, SERIES_COLORS, _font, apply_dashboard_style,
    set_dashboard_title, style_dashboard_legend,
)
from plot_mtp_peak_memory import load_peaks
from summarize_mtp_throughput import DEFAULT_INPUT, ROOT, summarize


BENCH = "throughput_1k"
ENTROPIES = ("low_entropy", "high_entropy")
METHODS = ("MTP", "Eagle3")
OUTPUT = ROOT / "benchmark_plots/nvidia-rtx-3080/mtp-vs-eagle3"


def compare(directory, osl):
    speeds = {(r["entropy"], r["n_predict"]): r["mean"]
              for r in summarize(directory) if (r["bench"], r["osl"]) == (BENCH, osl)}
    peaks = {(r["entropy"], r["n_predict"]): r["peak_memory_used_mib"] / 1024
             for r in load_peaks(directory) if (r["bench"], r["osl"]) == (BENCH, osl)}
    comparisons = {}
    for entropy in ENTROPIES:
        candidates = [n for e, n in speeds if e == entropy and n > 0]
        best = max(candidates, key=lambda n: (speeds[entropy, n], -n))
        comparisons[entropy] = (
            best, tuple(speeds[entropy, n] for n in (0, best)),
            tuple(peaks[entropy, n] for n in (0, best)),
        )
    return comparisons


def figure(title, subtitle, osl):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8))
    for ax in axes:
        apply_dashboard_style(fig, ax)
        ax.grid(False, axis="x")
        ax.set_xticks(range(len(ENTROPIES)), ["Low entropy", "High entropy"])
    fig.suptitle(title, x=.09, y=.97, ha="left", fontsize=18, color=HEADING,
                 fontfamily=_font("Space Grotesk"), fontweight="medium")
    fig.text(.09, .865, f"RTX 3080  |  {BENCH}  |  OSL {osl}  |  {subtitle}",
             fontsize=10, color=MUTED)
    fig.subplots_adjust(left=.09, right=.97, bottom=.14, top=.70, wspace=.26)
    return fig, axes


def render(data, output, osl):
    gain, gain_axes = figure(
        "MTP vs. Eagle3 · Throughput reward and VRAM cost",
        "Fastest tested setting compared with each method’s n_predict 0 baseline",
        osl,
    )
    for index, (method, color) in enumerate(zip(METHODS, SERIES_COLORS)):
        offset = (-.19, .19)[index]
        points = [data[method][entropy] for entropy in ENTROPIES]
        best = [point[0] for point in points]
        speed_gain = [speed[1] / speed[0] - 1 for _, speed, _ in points]
        extra_vram = [vram[1] - vram[0] for _, _, vram in points]
        for ax, values, labels in (
            (gain_axes[0], speed_gain, [f"{v:+.1%}\n(n={n})" for v, n in zip(speed_gain, best)]),
            (gain_axes[1], extra_vram, [f"{v:+.2f}" for v in extra_vram]),
        ):
            bars = ax.bar([x + offset for x in range(len(points))], values,
                          width=.32, color=color, label=method)
            ax.bar_label(bars, labels=labels, padding=5, fontsize=10, color=HEADING)

    for ax, title, ylabel in (
        (gain_axes[0], "Throughput reward", "Effective throughput gain"),
        (gain_axes[1], "VRAM cost", "Additional peak VRAM [GiB]"),
    ):
        set_dashboard_title(ax, title)
        ax.set_ylabel(ylabel, labelpad=8)
        ax.margins(y=.28)
        ax.set_xlim(-.5, 1.5)
    gain_axes[0].yaxis.set_major_formatter(PercentFormatter(xmax=1))
    style_dashboard_legend(gain_axes[1], loc="lower right", bbox_to_anchor=(1, 1.03),
                           ncol=2, borderaxespad=0, handlelength=1.2)

    output.mkdir(parents=True, exist_ok=True)
    for suffix in ("png", "svg"):
        path = output / f"throughput_vram_tradeoff_{BENCH}_osl{osl}.{suffix}"
        gain.savefig(path, dpi=200)
        print(path.resolve())
    plt.close(gain)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--osl", type=int, choices=(1024, 2048, 4096), default=1024)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT)
    args = parser.parse_args()
    for name in ("DMSans-static.ttf", "SpaceGrotesk-static.ttf"):
        font_manager.fontManager.addfont(str(ROOT / "benchmark_plots/fonts" / name))
    data = {
        "MTP": compare(DEFAULT_INPUT, args.osl),
        "Eagle3": compare(DEFAULT_INPUT.with_name("eagle3-speed-bench-results"), args.osl),
    }
    render(data, args.output_dir, args.osl)


if __name__ == "__main__":
    main()
