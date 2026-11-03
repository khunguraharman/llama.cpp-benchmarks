"""Generate Eagle3 throughput, speed-change, and peak-VRAM plots in the MTP style."""

import argparse
import csv
import json
from pathlib import Path

from plot_mtp_peak_memory import load_peaks, render as render_memory
from plot_mtp_speed_changes import render as render_changes
from plot_mtp_weighted_comparison import render_comparison
from summarize_mtp_throughput import ROOT, summarize


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-dir", type=Path,
        default=ROOT / "results/nvidia-rtx-3080/eagle3-speed-bench-results",
    )
    parser.add_argument(
        "--output-dir", type=Path,
        default=ROOT / "benchmark_plots/nvidia-rtx-3080/eagle3-speed-bench-results",
    )
    parser.add_argument("--memory-only", action="store_true",
                        help="Update only the annotated VRAM chart and memory summaries")
    args = parser.parse_args()
    memory_spec = json.loads((ROOT / "eagle3_memory_spec.json").read_text(encoding="utf-8"))
    if args.memory_only:
        render_memory(load_peaks(args.input_dir), args.output_dir, memory_spec=memory_spec)
        return
    rows = summarize(args.input_dir)
    peaks = load_peaks(args.input_dir)
    keys = lambda records: {
        (r["n_predict"], r["bench"], r["entropy"], r["osl"]) for r in records
    }
    if keys(rows) != keys(peaks):
        raise ValueError("Throughput results and GPU logs cover different configurations")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "weighted_throughput.csv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    render_comparison(rows, args.output_dir)
    render_changes(rows, args.output_dir)
    render_memory(peaks, args.output_dir, memory_spec=memory_spec)
    print(f"Plotted {len(rows)} configurations, {sum(r['runs'] for r in rows)} samples, "
          f"and {len(peaks)} GPU logs.")


if __name__ == "__main__":
    main()
