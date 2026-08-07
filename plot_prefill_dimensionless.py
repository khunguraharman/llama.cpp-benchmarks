"""Plot dimensionless llama.cpp prefill benchmark results.

The script reads results/<device>/prefill-bench-results and creates a plot of:

    x = prompt tokens / n_batch
    y = prefill tokens per second / reference tokens per second

By default the reference throughput is the best observed avg_ts value for each
model family in the selected result directory. This makes the y-axis a
dimensionless fraction of peak observed throughput.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path
from statistics import mean

import matplotlib.pyplot as plt

from plot_prefill_benchmarks import (
    MARKERS,
    RunRow,
    SeriesRows,
    choose_series,
    default_prefill_dirs,
    group_rows_by_model,
    load_prefill_rows,
    prefill_output_dir,
    slugify,
)


def add_dimensionless_values(rows: list[RunRow], tps_reference: float) -> list[RunRow]:
    dimensionless_rows: list[RunRow] = []

    for row in rows:
        batch_size = float(row["batch_size"])
        avg_ts = float(row["avg_ts"])
        stddev_ts = float(row.get("stddev_ts", 0.0))
        if batch_size <= 0 or tps_reference <= 0:
            continue

        dimensionless_row = dict(row)
        dimensionless_row["prompt_over_batch"] = float(row["prompt_tokens"]) / batch_size
        dimensionless_row["relative_tps"] = avg_ts / tps_reference
        dimensionless_row["relative_stddev_ts"] = stddev_ts / tps_reference
        dimensionless_rows.append(dimensionless_row)

    return dimensionless_rows


def average_duplicate_x_rows(rows: list[RunRow]) -> list[dict[str, float]]:
    rows_by_x: dict[float, list[RunRow]] = defaultdict(list)
    for row in rows:
        rows_by_x[float(row["prompt_over_batch"])].append(row)

    averaged_rows: list[dict[str, float]] = []
    for x_value, x_rows in sorted(rows_by_x.items()):
        averaged_rows.append(
            {
                "prompt_over_batch": x_value,
                "relative_tps": mean(float(row["relative_tps"]) for row in x_rows),
                "relative_stddev_ts": mean(float(row["relative_stddev_ts"]) for row in x_rows),
            }
        )

    return averaged_rows


def plot_dimensionless_prefill(
    series_rows: SeriesRows,
    output_path: Path,
    title: str,
    tps_reference: float,
) -> None:
    fig, ax = plt.subplots(figsize=(11, 7))

    for index, ((batch_size, ubatch_size), rows) in enumerate(series_rows.items()):
        averaged_rows = average_duplicate_x_rows(rows)
        if not averaged_rows:
            continue

        x_values = [row["prompt_over_batch"] for row in averaged_rows]
        y_values = [row["relative_tps"] for row in averaged_rows]
        y_errors = [row["relative_stddev_ts"] for row in averaged_rows]
        label = f"batch={batch_size}, ubatch={ubatch_size}"

        (line,) = ax.plot(
            x_values,
            y_values,
            marker=MARKERS[index % len(MARKERS)],
            linewidth=1.5,
            markersize=6,
            label=label,
        )
        ax.fill_between(
            x_values,
            [y_value - y_error for y_value, y_error in zip(y_values, y_errors)],
            [y_value + y_error for y_value, y_error in zip(y_values, y_errors)],
            color=line.get_color(),
            alpha=0.12,
            linewidth=0,
        )
        ax.errorbar(
            x_values,
            y_values,
            yerr=y_errors,
            color=line.get_color(),
            fmt="none",
            marker=MARKERS[index % len(MARKERS)],
            capsize=4,
        )

    ax.axhline(1.0, color="black", linewidth=1.0, alpha=0.35, linestyle="--")
    ax.set_title(title)
    ax.set_xlabel("Prompt tokens / n_batch")
    ax.set_ylabel("Prefill throughput / peak observed prefill throughput")
    ax.grid(True, alpha=0.3)
    ax.legend(title="Batch / microbatch", fontsize="small")
    ax.text(
        0.99,
        0.02,
        f"Reference throughput: {tps_reference:,.0f} tokens/s",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize="small",
        alpha=0.75,
    )
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def save_plots(
    prefill_dir: Path,
    output_dir: Path,
    max_lines: int | None,
    tps_normalizer: float | None,
) -> list[Path]:
    rows = load_prefill_rows(prefill_dir)
    if not rows:
        return []

    output_dir.mkdir(parents=True, exist_ok=True)
    saved_paths: list[Path] = []

    for model_family, model_rows in sorted(group_rows_by_model(rows).items()):
        reference_tps = tps_normalizer or max(float(row["avg_ts"]) for row in model_rows)
        dimensionless_rows = add_dimensionless_values(model_rows, reference_tps)
        series_rows = choose_series(dimensionless_rows, max_lines)
        if not series_rows:
            continue

        plot_path = output_dir / f"{slugify(model_family)}_prefill_dimensionless_throughput.png"
        plot_dimensionless_prefill(
            series_rows,
            plot_path,
            f"{model_family}: dimensionless prefill throughput",
            reference_tps,
        )
        saved_paths.append(plot_path)

    return saved_paths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot dimensionless llama.cpp prefill benchmark results.")
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
        "--max-lines",
        type=int,
        default=None,
        help="Maximum number of batch/microbatch lines per plot. Defaults to all unique combinations.",
    )
    parser.add_argument(
        "--tps-normalizer",
        type=float,
        default=None,
        help=(
            "Reference tokens/sec for the y-axis. Defaults to each model family's "
            "best observed avg_ts in the selected result directory."
        ),
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Open an interactive matplotlib window after saving the plots.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    prefill_dirs = args.prefill_dirs or default_prefill_dirs()
    saved_paths: list[Path] = []

    for prefill_dir in prefill_dirs:
        output_dir = prefill_output_dir(prefill_dir, args.output_dir)
        saved_paths.extend(save_plots(prefill_dir, output_dir, args.max_lines, args.tps_normalizer))

    for path in saved_paths:
        print(f"Saved plot to {path.resolve()}")

    if args.show:
        plt.show()


if __name__ == "__main__":
    main()
