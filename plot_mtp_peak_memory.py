"""Compare peak GPU memory across MTP n_predict, with one series per OSL.

Each point is the maximum logged sample across the entire benchmark file.
Allocated VRAM is reported in MiB; prompt length and entropy are never pooled.
"""

import argparse
import csv
import math
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from plot_localai_style import (
    HEADING, SERIES_COLORS, apply_dashboard_style,
    set_dashboard_title,
)

ROOT = Path(__file__).resolve().parent
DEFAULT_INPUT = ROOT / "results/nvidia-rtx-3080/mtp-speed-bench-results"
NAME = re.compile(r"mtp_n(\d+)_(throughput_\d+k)_(high_entropy|low_entropy)_osl(\d+)\.gpu\.csv")
MARKERS = ("o", "s", "^", "D", "v", "P")


def load_peaks(input_dir):
    rows = []
    for path in sorted(input_dir.glob("*.gpu.csv")):
        match = NAME.fullmatch(path.name)
        if not match:
            continue
        n, bench, entropy, osl = match.groups()
        samples = []
        with path.open(encoding="utf-8-sig", newline="") as stream:
            for raw in csv.DictReader(stream, skipinitialspace=True):
                sample = {}
                for key in ("memory_used_mib",):
                    try:
                        value = float(raw[key])
                        if math.isfinite(value):
                            sample[key] = value
                    except (KeyError, TypeError, ValueError):
                        pass
                samples.append(sample)
        row = dict(n_predict=int(n), bench=bench, entropy=entropy, osl=int(osl), source=path.name)
        for source, target in (("memory_used_mib", "peak_memory_used_mib"),):
            values = [s[source] for s in samples if source in s]
            if not values:
                raise ValueError(f"No valid {source} samples in {path}")
            row[target] = max(values)
        rows.append(row)
    if not rows:
        raise ValueError(f"No matching GPU logs found in {input_dir}")
    return rows


def render(rows, output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "peak_memory.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    benches = sorted({r["bench"] for r in rows})
    entropies = [e for e in ("low_entropy", "high_entropy") if any(r["entropy"] == e for r in rows)]
    ns = sorted({r["n_predict"] for r in rows})
    lengths = sorted({r["osl"] for r in rows})
    for metric, ylabel, filename, title in (
        ("peak_memory_used_mib", "Peak VRAM used [MiB]", "peak_vram_used_vs_n_predict.png", "Peak GPU memory footprint"),
    ):
        fig, axes = plt.subplots(len(benches), len(entropies), figsize=(13, 8), squeeze=False, sharex=True, sharey=True)
        for i, bench in enumerate(benches):
            for j, entropy in enumerate(entropies):
                ax = axes[i, j]
                apply_dashboard_style(fig, ax)
                set_dashboard_title(ax, f"{bench.replace('throughput_', '')} prompt · {entropy.replace('_', ' ')}")
                for index, osl in enumerate(lengths):
                    points = sorted((r for r in rows if r["bench"] == bench and r["entropy"] == entropy and r["osl"] == osl), key=lambda r: r["n_predict"])
                    ax.plot([r["n_predict"] for r in points], [r[metric] for r in points],
                            color=SERIES_COLORS[index % len(SERIES_COLORS)], marker=MARKERS[index % len(MARKERS)],
                            linewidth=1.8, markersize=6, label=f"OSL={osl:,}")
                ax.set_xticks(ns)
                ax.tick_params(labelbottom=True)
                ax.set_xlabel("n_predict")
                ax.set_ylabel(ylabel)
        values = [r[metric] for r in rows]
        padding = max((max(values) - min(values)) * .15, 1)
        axes[0, 0].set_ylim(max(0, min(values) - padding), max(values) + padding)
        handles, labels = axes[0, 0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(.5, .925), ncol=len(lengths), frameon=False)
        fig.suptitle(f"{title} vs n_predict", fontsize=18, color=HEADING, y=.98)
        fig.subplots_adjust(left=.085, right=.98, top=.82, bottom=.08, hspace=.5, wspace=.22)
        path = output_dir / filename
        svg_path = path.with_suffix(".svg")
        fig.savefig(svg_path, facecolor="white")
        plt.close(fig)
        print(svg_path.resolve())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "benchmark_plots/nvidia-rtx-3080/mtp-speed-bench-results")
    args = parser.parse_args()
    rows = load_peaks(args.input_dir)
    render(rows, args.output_dir)
    print(f"Read {len(rows)} benchmark GPU logs.")


if __name__ == "__main__":
    main()
