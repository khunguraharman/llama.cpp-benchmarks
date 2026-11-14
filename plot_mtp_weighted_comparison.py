"""Render individual entropy comparison SVG charts from weighted summaries."""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

from summarize_mtp_throughput import DEFAULT_INPUT, ROOT, summarize
from plot_localai_style import (
    BACKGROUND, TEXT, HEADING, SERIES_COLORS,
    _font, apply_dashboard_style, set_dashboard_title, style_dashboard_legend,
)


def render_comparison(rows, output_dir, *, series_alpha=1.0):
    output_dir.mkdir(parents=True, exist_ok=True)
    for name in ("DMSans-static.ttf", "SpaceGrotesk-static.ttf"):
        font_manager.fontManager.addfont(str(ROOT / "benchmark_plots/fonts" / name))
    benches = sorted({r["bench"] for r in rows})
    lengths = sorted({r["osl"] for r in rows})
    lower = min(r["mean"] - r["sd"] for r in rows)
    upper = max(r["mean"] + r["sd"] for r in rows)
    padding = max((upper - lower) * .08, 1)
    plt.rcParams.update({
        "font.family": _font("DM Sans"), "font.size": 10,
        "text.color": TEXT, "axes.titlecolor": HEADING,
        "axes.titleweight": "medium", "figure.facecolor": BACKGROUND,
        "axes.facecolor": BACKGROUND, "savefig.facecolor": BACKGROUND,
        "lines.linewidth": 2, "lines.markersize": 5,
    })
    fig, axes = plt.subplots(len(benches), len(lengths), figsize=(16, 9), squeeze=False)
    for i, bench in enumerate(benches):
        for j, osl in enumerate(lengths):
            ax = axes[i, j]
            subset = [r for r in rows if r["bench"] == bench and r["osl"] == osl]
            set_dashboard_title(ax, f"{bench} | OSL {osl}")
            apply_dashboard_style(fig, ax)
            for entropy, label, color, marker, linestyle in (
                ("low_entropy", "Low entropy", SERIES_COLORS[0], "o", "-"),
                ("high_entropy", "High entropy", SERIES_COLORS[1], "s", "--"),
            ):
                series = sorted((r for r in subset if r["entropy"] == entropy), key=lambda r: r["n_predict"])
                ax.errorbar([r["n_predict"] for r in series], [r["mean"] for r in series],
                            yerr=[r["sd"] for r in series], color=color, marker=marker,
                            linestyle=linestyle, capsize=3, alpha=series_alpha, label=label)
            ax.set_xticks(sorted({r["n_predict"] for r in subset}))
            ax.set_xlabel("n_predict")
            ax.set_ylabel("Tokens per sec")
            ax.set_ylim(max(0, lower - padding), upper + padding)
            style_dashboard_legend(ax, loc="best")
    fig.suptitle("RTX 3080 · Weighted generation throughput", fontsize=19, y=.98,
                 color=HEADING, fontfamily=_font("Space Grotesk"), fontweight="medium")
    fig.subplots_adjust(left=.055, right=.985, top=.9, bottom=.09, hspace=.42, wspace=.27)
    # Export each panel with its own title, axes, and legend.
    fig._suptitle.set_visible(False)
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    for i, bench in enumerate(benches):
        for j, osl in enumerate(lengths):
            bounds = axes[i, j].get_tightbbox(renderer).transformed(fig.dpi_scale_trans.inverted())
            svg_path = output_dir / f"weighted_throughput_{bench}_osl{osl}.svg"
            fig.savefig(svg_path, bbox_inches=bounds.padded(.15), format="svg")
            fig.savefig(svg_path.with_suffix(".png"), bbox_inches=bounds.padded(.15), dpi=200)
            print(svg_path.resolve())
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "benchmark_plots/nvidia-rtx-3080/mtp/mtp-speed-bench-results")
    args = parser.parse_args()
    render_comparison(summarize(args.input_dir), args.output_dir, series_alpha=0.6)


if __name__ == "__main__":
    main()
