"""Export throughput changes by output sequence length."""

import argparse
import csv
from collections import defaultdict

from pathlib import Path

from summarize_mtp_throughput import DEFAULT_INPUT, ROOT, summarize


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


def write_speed_changes(rows, output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)
    changes = calculate_changes(rows)
    with (output_dir / "speed_changes_by_osl.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(changes[0]))
        writer.writeheader()
        writer.writerows(changes)
    return changes


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "benchmark_plots/nvidia-rtx-3080/mtp/mtp-speed-bench-results")
    args = parser.parse_args()
    write_speed_changes(summarize(args.input_dir), args.output_dir)


if __name__ == "__main__":
    main()
