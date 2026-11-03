"""Compare low-entropy MTP and Eagle3 generation throughput by n_predict."""

import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

from plot_localai_style import (
    SERIES_COLORS, apply_dashboard_style, save_dashboard_figure,
    set_dashboard_title, style_dashboard_legend,
)
from summarize_mtp_throughput import DEFAULT_INPUT, ROOT, summarize


def render_comparison(mtp_dir, eagle3_dir, output_dir):
    series = []
    for label, directory in (("Deepseek/Qwen-3.5", mtp_dir), ("Eagle3", eagle3_dir)):
        rows = [dict(series=label, **row) for row in summarize(directory)
                if row["entropy"] == "low_entropy"]
        if not rows:
            raise ValueError(f"No low_entropy results in {directory}")
        series.append((label, rows))

    keys = lambda rows: {(r["bench"], r["osl"], r["n_predict"]) for r in rows}
    if keys(series[0][1]) != keys(series[1][1]):
        raise ValueError("MTP and Eagle3 low_entropy configurations do not match")

    output_dir.mkdir(parents=True, exist_ok=True)
    for name in ("DMSans-static.ttf", "SpaceGrotesk-static.ttf"):
        font_manager.fontManager.addfont(str(ROOT / "benchmark_plots/fonts" / name))
    all_rows = [row for _, rows in series for row in rows]
    with (output_dir / "low_entropy_throughput.csv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=list(all_rows[0]))
        writer.writeheader()
        writer.writerows(all_rows)

    for bench, osl in sorted({(r["bench"], r["osl"]) for r in all_rows}):
        fig, ax = plt.subplots(figsize=(9, 5.5))
        apply_dashboard_style(fig, ax)
        for index, (label, rows) in enumerate(series):
            points = sorted((r for r in rows if (r["bench"], r["osl"]) == (bench, osl)),
                            key=lambda r: r["n_predict"])
            ax.errorbar(
                [r["n_predict"] for r in points], [r["mean"] for r in points],
                yerr=[r["sd"] for r in points], label=label,
                color=SERIES_COLORS[index], marker=("o", "s")[index],
                linestyle=("-", "--")[index], linewidth=2, capsize=3,
            )
        set_dashboard_title(
            ax, f"RTX 3080 · {bench.replace('throughput_', '')} prompt · OSL {osl}"
            "\nLow entropy · Token-weighted generation throughput"
        )
        ax.set_xlabel("n_predict")
        ax.set_ylabel("Tokens per second")
        ax.set_xticks(sorted({r["n_predict"] for r in all_rows
                             if (r["bench"], r["osl"]) == (bench, osl)}))
        ax.set_ylim(bottom=0)
        ax.margins(x=.05, y=.12)
        style_dashboard_legend(ax, loc="best")
        path = output_dir / f"low_entropy_{bench}_osl{osl}.png"
        save_dashboard_figure(fig, ax, path)
        fig.savefig(path.with_suffix(".svg"), facecolor=fig.get_facecolor(), edgecolor="none")
        plt.close(fig)
        print(path.resolve())
    return all_rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mtp-dir", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--eagle3-dir", type=Path,
                        default=ROOT / "results/nvidia-rtx-3080/eagle3-speed-bench-results")
    parser.add_argument("--output-dir", type=Path,
                        default=ROOT / "benchmark_plots/nvidia-rtx-3080/mtp/low-entropy-comparison")
    args = parser.parse_args()
    render_comparison(args.mtp_dir, args.eagle3_dir, args.output_dir)


if __name__ == "__main__":
    main()
