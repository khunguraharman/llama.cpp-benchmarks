"""Export generation tokens per second as mean +/- one sample standard deviation."""

import argparse
import csv
import json
import statistics
from pathlib import Path, PureWindowsPath


DEFAULT_INPUT = Path(__file__).resolve().parent / "results/nvidia-rtx-3080/throughput"


def summarize(input_dir):
    files = sorted(input_dir.glob("*.json"))
    if not files:
        raise ValueError(f"No JSON files found in {input_dir}")
    results = []
    for path in files:
        with path.open(encoding="utf-8") as source:
            records = json.load(source)
        for index, record in enumerate(records, start=1):
            if record["n_gen"] <= 0:
                continue
            samples = record["samples_ts"]
            if len(samples) < 2:
                raise ValueError(f"{path.name}, record {index}: sample SD requires at least two repetitions")
            mean = statistics.mean(samples)
            sd = statistics.stdev(samples)
            results.append({
                "model_filename": PureWindowsPath(record["model_filename"]).name,
                "n_gen": record["n_gen"],
                "mean_tokens_per_second": mean,
                "sample_sd_tokens_per_second": sd,
                "mean_plus_minus_sd": f"{mean:.2f} +/- {sd:.2f}",
            })
    if not results:
        raise ValueError("No matching benchmark records found")
    return results, len(files)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, help="Default: throughput_summary.csv inside the input directory")
    args = parser.parse_args()
    rows, file_count = summarize(args.input_dir)
    output = args.output or args.input_dir / "throughput_summary.csv"
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as destination:
        writer = csv.DictWriter(destination, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Exported {len(rows)} generation throughput results from {file_count} files to {output.resolve()}")


if __name__ == "__main__":
    main()
