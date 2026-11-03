"""Plot peak-to-n=5 drop and n=0-to-peak gain against output length."""

import argparse
import csv
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.ticker import PercentFormatter
from pathlib import Path

from summarize_mtp_throughput import DEFAULT_INPUT, ROOT, summarize
from plot_localai_style import (
    SERIES_COLORS, apply_dashboard_style, save_dashboard_figure,
    set_dashboard_title, style_dashboard_legend,
)


def calculate_changes(rows):
    groups = defaultdict(dict)
    for row in rows:
        groups[row["bench"], row["entropy"], row["osl"]][row["n_predict"]] = row["mean"]
    changes = []
    for (bench, entropy, osl), speeds in sorted(groups.items()):
        if set(speeds) != set(range(6)):
            raise ValueError(f"Expected n_predict=0..5 for {bench}/{entropy}/OSL {osl}")
        peak = max(speeds.values())
        changes.append(dict(
            bench=bench, entropy=entropy, osl=osl,
            baseline_speed=speeds[0], peak_speed=peak, n5_speed=speeds[5],
            peak_n_predict=",".join(str(n) for n in sorted(speeds) if speeds[n] == peak),
            drop_percent=100 * (peak - speeds[5]) / peak,
            gain_percent=100 * (peak - speeds[0]) / speeds[0],
        ))
    return changes


def render(rows, output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)
    for name in ("DMSans-static.ttf", "SpaceGrotesk-static.ttf"):
        font_manager.fontManager.addfont(str(ROOT / "benchmark_plots/fonts" / name))
    changes = calculate_changes(rows)
    with (output_dir / "speed_changes_by_osl.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(changes[0]))
        writer.writeheader()
        writer.writerows(changes)
    for bench in sorted({r["bench"] for r in changes}):
        for metric, title, ylabel in (
            ("drop_percent", "Speed drop from peak to n_predict=5", "Drop from peak (%)"),
            ("gain_percent", "Speed gain from n_predict=0 to peak", "Gain over n_predict=0 (%)"),
        ):
            fig, ax = plt.subplots(figsize=(9, 5.5))
            apply_dashboard_style(fig, ax)
            selected = [r for r in changes if r["bench"] == bench]
            upper = max(r[metric] for r in selected)
            for entropy, label, color, marker, linestyle in (
                ("low_entropy", "Low entropy", SERIES_COLORS[0], "o", "-"),
                ("high_entropy", "High entropy", SERIES_COLORS[1], "s", "--"),
            ):
                points = sorted((r for r in selected if r["entropy"] == entropy), key=lambda r: r["osl"])
                ax.plot([r["osl"] for r in points], [r[metric] for r in points],
                        color=color, marker=marker, linestyle=linestyle, linewidth=2, label=label)
                for r in points:
                    other_values = [p[metric] for p in selected if p["osl"] == r["osl"] and p["entropy"] != entropy]
                    above = not other_values or r[metric] > max(other_values)
                    ax.annotate(f'{r[metric]:.1f}%', (r["osl"], r[metric]),
                                xytext=(0, 10 if above else -17),
                                textcoords="offset points", ha="center", color=color, fontsize=9)
            lengths = sorted({r["osl"] for r in selected})
            peak_lines = ["Peak n_predict by OSL " + " / ".join(str(osl) for osl in lengths) + ":"]
            for entropy, label in (("low_entropy", "Low entropy"), ("high_entropy", "High entropy")):
                peaks = {r["osl"]: r["peak_n_predict"] for r in selected if r["entropy"] == entropy}
                peak_lines.append(label + ": " + " / ".join(peaks.get(osl, "—") for osl in lengths))
            set_dashboard_title(ax, f"{bench.replace('throughput_', '')} prompt · {title}\n" + "\n".join(peak_lines))
            ax.set_xlabel("Output sequence length (OSL)")
            ax.set_ylabel(ylabel)
            ax.set_xticks(sorted({r["osl"] for r in selected}))
            ax.yaxis.set_major_formatter(PercentFormatter(xmax=100))
            ax.set_ylim(-max(upper * .08, 1), max(upper * 1.2, 5))
            ax.margins(x=.12)
            style_dashboard_legend(ax, loc="best")
            path = output_dir / f"{bench}_{metric}_by_osl.png"
            save_dashboard_figure(fig, ax, path)
            fig.savefig(path.with_suffix(".svg"), facecolor=fig.get_facecolor(), edgecolor="none")
            plt.close(fig)
            print(path.resolve())
    return changes


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "benchmark_plots/nvidia-rtx-3080/mtp/mtp-speed-bench-results")
    args = parser.parse_args()
    render(summarize(args.input_dir), args.output_dir)


if __name__ == "__main__":
    main()
