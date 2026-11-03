"""Compare peak GPU memory across speculative n_predict, with one series per OSL.

Each point is the maximum logged sample across the entire benchmark file.
Allocated VRAM is reported in MiB; prompt length and entropy are never pooled.
"""

import argparse
import csv
import json
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
NAME = re.compile(r"(?:mtp|eagle3)_n(\d+)_(throughput_\d+k)_(high_entropy|low_entropy)_osl(\d+)\.gpu\.csv")
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


def memory_components(spec):
    """Estimate draft tensor payload and allocated dense KV capacity, in MiB."""
    kv_bytes = (spec["sequences"] * spec["context_tokens"] * spec["draft_layers"]
                * spec["kv_heads"] * (spec["key_head_dim"] * spec["key_bytes"]
                                      + spec["value_head_dim"] * spec["value_bytes"]))
    return spec["weight_tensor_bytes"] / 2**20, kv_bytes / 2**20


def annotate_eagle3_memory(ax, selected, spec, metric):
    weights, kv = memory_components(spec)
    # Average across OSL only for the annotation's common reference levels.
    # The measured curves retain their individual OSL values.
    baseline = sum(r[metric] for r in selected if r["n_predict"] == 0)
    baseline /= sum(r["n_predict"] == 0 for r in selected)
    enabled = sum(r[metric] for r in selected if r["n_predict"] == 1)
    enabled /= sum(r["n_predict"] == 1 for r in selected)
    weights_top = baseline + weights
    accounted = weights_top + kv
    ax.hlines(baseline, 1.15, 5.35, color="#7b8c98", linestyle="--", linewidth=1)
    ax.fill_between([1.15, 5.35], baseline, weights_top, color="#82684d", alpha=.09)
    ax.hlines(weights_top, 1.15, 5.35, color="#82684d", linestyle="--", linewidth=1)
    ax.hlines(accounted, 1.15, 5.35, color="#c17b50", linestyle=":", linewidth=1.3)
    ax.annotate("", xy=(1.55, weights_top), xytext=(1.55, baseline),
                arrowprops=dict(arrowstyle="<->", color="#82684d", lw=1.5))
    box = dict(facecolor="white", alpha=.9, edgecolor="none", pad=2)
    ax.text(1.72, baseline + weights / 2, f"Est. draft weights: +{weights:,.0f} MiB",
            va="center", fontsize=8.5, color="#82684d", bbox=box)
    # KV's 32 MiB band is too short for an internal text label.
    ax.plot([2.35, 2.35], [weights_top, accounted], color="#c17b50", lw=2.5)
    ax.annotate(f"Est. {spec['context_tokens']:,}-token draft KV: +{kv:,.0f} MiB",
                xy=(2.35, accounted), xytext=(2.0, accounted + 360),
                fontsize=8.5, color="#c17b50", bbox=box,
                arrowprops=dict(arrowstyle="-", color="#c17b50", lw=1))
    ax.annotate("", xy=(3.5, enabled), xytext=(3.5, accounted),
                arrowprops=dict(arrowstyle="<->", color="#16705c", lw=1.5))
    ax.text(3.65, (accounted + enabled) / 2,
            f"Unattributed runtime\ndifference: ~{enabled-accounted:,.0f} MiB",
            va="center", fontsize=8.5, color="#16705c", bbox=box)
    ax.text(5.3, baseline - 120, "n=0 baseline (mean across OSL)",
            ha="right", va="top", fontsize=8, color="#68756f", bbox=box)
    ax.set_xlim(-.3, 5.5)


def render(rows, output_dir, memory_spec=None):
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "peak_memory.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    benches = sorted({r["bench"] for r in rows})
    entropies = [e for e in ("low_entropy", "high_entropy") if any(r["entropy"] == e for r in rows)]
    ns = sorted({r["n_predict"] for r in rows})
    lengths = sorted({r["osl"] for r in rows})
    if memory_spec is not None:
        weights, kv = memory_components(memory_spec)
        breakdown = []
        indexed = {(r["bench"], r["entropy"], r["osl"], r["n_predict"]): r for r in rows}
        for row in rows:
            if row["n_predict"] != 1:
                continue
            baseline = indexed[row["bench"], row["entropy"], row["osl"], 0]["peak_memory_used_mib"]
            jump = row["peak_memory_used_mib"] - baseline
            breakdown.append(dict(bench=row["bench"], entropy=row["entropy"], osl=row["osl"],
                baseline_peak_mib=baseline, n1_peak_mib=row["peak_memory_used_mib"],
                measured_jump_mib=jump, estimated_draft_weights_mib=weights,
                estimated_draft_kv_mib=kv, unattributed_difference_mib=jump-weights-kv))
        with (output_dir / "eagle3_memory_breakdown.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(breakdown[0]))
            writer.writeheader()
            writer.writerows(breakdown)
        (output_dir / "eagle3_memory_methodology.json").write_text(
            json.dumps(dict(spec=memory_spec,
                kv_formula="sequences * context_tokens * draft_layers * kv_heads * (key_head_dim * key_bytes + value_head_dim * value_bytes)",
                estimated_draft_weights_mib=weights, estimated_draft_kv_mib=kv,
                annotation_reference="n=0 and n=1 mean peak VRAM across OSL within each panel",
                residual="Difference of device-wide peaks minus estimates; not a measured buffer allocation. Includes changed compute/feature buffers, allocator effects and other device usage; exact attribution unavailable.",
                breakdown=breakdown), indent=2), encoding="utf-8")
    for metric, ylabel, filename, title in (
        ("peak_memory_used_mib", "Peak VRAM used [MiB]", "peak_vram_used_vs_n_predict.png", "Peak GPU memory footprint"),
    ):
        fig, axes = plt.subplots(len(benches), len(entropies), figsize=(15, 10) if memory_spec else (13, 8), squeeze=False, sharex=True, sharey=True)
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
                if memory_spec is not None:
                    selected = [r for r in rows if r["bench"] == bench and r["entropy"] == entropy]
                    annotate_eagle3_memory(ax, selected, memory_spec, metric)
        values = [r[metric] for r in rows]
        padding = max((max(values) - min(values)) * .15, 1)
        axes[0, 0].set_ylim(max(0, min(values) - padding), max(values) + padding)
        handles, labels = axes[0, 0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(.5, .925), ncol=len(lengths), frameon=False)
        fig.suptitle(f"{title} vs n_predict", fontsize=18, color=HEADING, y=.98)
        fig.subplots_adjust(left=.085, right=.98, top=.82, bottom=.08, hspace=.5, wspace=.22)
        if memory_spec is not None:
            fig.suptitle("Eagle3 · Peak GPU memory footprint vs n_predict", fontsize=18, color=HEADING, y=.98)
            fig.subplots_adjust(bottom=.13)
            fig.text(.5, .025,
                "Estimates: local GGUF tensor payloads; full offload; 1 sequence; FP16 draft KV; 8,192-token context.\n"
                "Shaded band and vertical brackets show added draft memory above n=0. Residual is unattributed, not a measured allocation.",
                ha="center", fontsize=9, color="#68756f")
        path = output_dir / filename
        svg_path = path.with_suffix(".svg")
        fig.savefig(path, dpi=200, facecolor="white")
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
