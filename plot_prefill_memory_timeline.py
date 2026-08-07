"""Plot GPU memory utilization timelines for llama.cpp prefill benchmarks.

The script reads results/<device>/prefill-bench-results/*.gpu.csv and creates a
memory-utilization-over-time plot. Each GPU CSV is plotted as its own series,
with time normalized so 0 ms is the first log entry in that CSV.
"""

from __future__ import annotations

import argparse
import csv
import re
from datetime import datetime
from math import ceil
from pathlib import Path

import matplotlib.pyplot as plt


RUN_NAME_RE = re.compile(r"_p(?P<prompt>\d+)_b(?P<batch>\d+)_ub(?P<ubatch>\d+)_")
TIMESTAMP_FORMATS = (
    "%Y/%m/%d %H:%M:%S.%f",
    "%Y-%m-%d %H:%M:%S.%f",
    "%Y/%m/%d %H:%M:%S",
    "%Y-%m-%d %H:%M:%S",
)


def parse_run_values(path: Path) -> tuple[int | None, int | None, int | None]:
    match = RUN_NAME_RE.search(path.name)
    if not match:
        return None, None, None

    return (
        int(match.group("prompt")),
        int(match.group("batch")),
        int(match.group("ubatch")),
    )


def parse_timestamp(value: str) -> datetime:
    value = value.strip()
    for timestamp_format in TIMESTAMP_FORMATS:
        try:
            return datetime.strptime(value, timestamp_format)
        except ValueError:
            pass

    raise ValueError(f"Unsupported timestamp format: {value!r}")


def series_label(csv_path: Path) -> str:
    prompt_size, batch_size, ubatch_size = parse_run_values(csv_path)
    if prompt_size is None or batch_size is None or ubatch_size is None:
        return csv_path.stem

    return f"prompt={prompt_size}, batch={batch_size}, microbatch={ubatch_size}"


def read_memory_series(csv_path: Path) -> tuple[list[float], list[float]]:
    timestamps: list[datetime] = []
    memory_values: list[float] = []

    with csv_path.open("r", encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file, skipinitialspace=True)
        for row in reader:
            cleaned_row = {
                str(key).strip(): value.strip() if isinstance(value, str) else value
                for key, value in row.items()
                if key is not None
            }
            timestamp = cleaned_row.get("timestamp")
            memory_utilization = cleaned_row.get("memory_utilization_percent")
            if not timestamp or not memory_utilization:
                continue

            try:
                timestamps.append(parse_timestamp(timestamp))
                memory_values.append(float(memory_utilization))
            except ValueError:
                continue

    if not timestamps:
        return [], []

    start_time = timestamps[0]
    elapsed_ms = [
        (timestamp - start_time).total_seconds() * 1000.0
        for timestamp in timestamps
    ]
    return elapsed_ms, memory_values


def sort_key(csv_path: Path) -> tuple[int, int, int, str]:
    prompt_size, batch_size, ubatch_size = parse_run_values(csv_path)
    return (
        prompt_size if prompt_size is not None else -1,
        batch_size if batch_size is not None else -1,
        ubatch_size if ubatch_size is not None else -1,
        csv_path.name,
    )


def chunks(values: list[Path], chunk_size: int) -> list[list[Path]]:
    if chunk_size <= 0:
        return [values]

    return [values[index : index + chunk_size] for index in range(0, len(values), chunk_size)]


def default_prefill_dirs() -> list[Path]:
    results_dir = Path.cwd() / "results"
    if results_dir.exists():
        return sorted(path for path in results_dir.glob("*/prefill-bench-results") if path.is_dir())

    return [Path.cwd()]


def prefill_output_dir(prefill_dir: Path, output_dir: Path) -> Path:
    results_dir = Path.cwd() / "results"
    try:
        relative_path = prefill_dir.resolve().relative_to(results_dir.resolve())
    except ValueError:
        return output_dir

    if relative_path.parts:
        return output_dir / relative_path.parts[0] / "prefill"

    return output_dir


def plot_prefill_memory_timelines(
    prefill_dir: Path,
    csv_paths: list[Path],
    output_path: Path,
    page_number: int | None = None,
    page_count: int | None = None,
) -> bool:
    figure_height = max(8.0, min(18.0, 4.5 + len(csv_paths) * 0.12))
    fig, ax = plt.subplots(figsize=(14, figure_height))
    plotted_count = 0

    for csv_path in csv_paths:
        elapsed_ms, memory_values = read_memory_series(csv_path)
        if not elapsed_ms:
            continue

        ax.plot(
            elapsed_ms,
            memory_values,
            linewidth=1.2,
            alpha=0.85,
            label=series_label(csv_path),
        )
        plotted_count += 1

    if plotted_count == 0:
        plt.close(fig)
        return False

    title = f"{prefill_dir.parent.name}: prefill memory utilization over time"
    if page_number is not None and page_count is not None and page_count > 1:
        title = f"{title} ({page_number}/{page_count})"

    ax.set_title(title)
    ax.set_xlabel("Time since logging started (ms)")
    ax.set_ylabel("Memory utilization (%)")
    ax.grid(True, alpha=0.3)
    ax.legend(
        title="Run parameters",
        fontsize="xx-small",
        ncols=2,
        loc="center left",
        bbox_to_anchor=(1.01, 0.5),
        borderaxespad=0,
    )
    fig.subplots_adjust(right=0.58)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    return True


def save_plots(prefill_dir: Path, output_dir: Path, series_per_plot: int) -> list[Path]:
    csv_paths = sorted(prefill_dir.glob("*.gpu.csv"), key=sort_key)
    if not csv_paths:
        return []

    csv_path_chunks = chunks(csv_paths, series_per_plot)
    page_count = len(csv_path_chunks)
    saved_paths: list[Path] = []

    for index, csv_path_chunk in enumerate(csv_path_chunks, start=1):
        if page_count == 1:
            output_path = output_dir / "prefill_memory_utilization_over_time.png"
            page_number = None
        else:
            output_path = output_dir / f"prefill_memory_utilization_over_time_{index:03d}.png"
            page_number = index

        if plot_prefill_memory_timelines(prefill_dir, csv_path_chunk, output_path, page_number, page_count):
            saved_paths.append(output_path)

    return saved_paths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot prefill GPU memory utilization timelines.")
    parser.add_argument(
        "prefill_dirs",
        nargs="*",
        type=Path,
        help="Prefill result directories. Defaults to results/<device>/prefill-bench-results.",
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        type=Path,
        default=Path("benchmark_plots"),
        help="Directory for generated plots. Defaults to benchmark_plots.",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Open an interactive matplotlib window after saving the plots.",
    )
    parser.add_argument(
        "--series-per-plot",
        type=int,
        default=80,
        help="Maximum CSV series per output plot. Use 0 to force one plot. Defaults to 80.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    prefill_dirs = args.prefill_dirs or default_prefill_dirs()
    saved_paths: list[Path] = []

    for prefill_dir in prefill_dirs:
        output_dir = prefill_output_dir(prefill_dir, args.output_dir)
        saved_paths.extend(save_plots(prefill_dir, output_dir, args.series_per_plot))

    for path in saved_paths:
        print(f"Saved plot to {path.resolve()}")

    if args.show:
        plt.show()


if __name__ == "__main__":
    main()
