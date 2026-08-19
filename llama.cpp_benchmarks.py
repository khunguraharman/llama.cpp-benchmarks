
"""Plot llama.cpp benchmark results from JSON files.

The script reads llama-bench style JSON files and creates:

1. Average generated tokens per second vs generated tokens.
2. Time-to-first-token estimate from each generation run.
3. 99th percentile time between generated tokens.

Vertical error bars show one standard deviation.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev
from typing import Any

import matplotlib.pyplot as plt

from platform_paths import result_directory, results_root
from plot_style import (
    apply_dashboard_style,
    save_dashboard_figure,
    set_dashboard_title,
    style_dashboard_legend,
)


MARKERS = ["o", "s", "^", "D", "v", "P", "X", "*", "<", ">", "h", "8"]


NestedRows = dict[str, dict[str, list[dict[str, float]]]]
PathGroups = dict[Path, list[Path]]


def stddev_or_zero(values: list[float]) -> float:
    return stdev(values) if len(values) > 1 else 0.0


def percentile(values: list[float], percent: float) -> float:
    """Return a percentile using linear interpolation."""

    if not values:
        raise ValueError("Cannot calculate a percentile from an empty list.")

    sorted_values = sorted(values)
    if len(sorted_values) == 1:
        return sorted_values[0]

    rank = (len(sorted_values) - 1) * (percent / 100)
    lower_index = int(rank)
    upper_index = min(lower_index + 1, len(sorted_values) - 1)
    weight = rank - lower_index

    return sorted_values[lower_index] * (1 - weight) + sorted_values[upper_index] * weight


def split_model_name(model_name: str) -> tuple[str, str]:
    """Split a model name into its base model family and quantization label."""

    parts = model_name.split()
    for index, part in enumerate(parts):
        if re.match(r"^Q\d", part):
            return " ".join(parts[:index]), " ".join(parts[index:])

    return model_name, model_name


def slugify(value: str) -> str:
    """Create a readable filename-safe slug."""

    value = value.lower()
    value = re.sub(r"[^a-z0-9]+", "_", value)
    return value.strip("_")


def as_rows(json_paths: list[Path]) -> list[dict[str, Any]]:
    """Load all benchmark rows from JSON files."""

    rows: list[dict[str, Any]] = []

    for json_path in json_paths:
        with json_path.open("r", encoding="utf-8") as file:
            data: Any = json.load(file)

        if isinstance(data, dict):
            data = [data]

        for row in data:
            row["_source_file"] = json_path.name
            row["_model_name"] = row.get("model_type") or row.get("model_filename") or json_path.stem
            row["_model_family"], row["_model_variation"] = split_model_name(str(row["_model_name"]))
            rows.append(row)

    return rows


def load_generation_speed_rows(rows: list[dict[str, Any]]) -> NestedRows:
    """Load generation speed rows grouped by model family and quantization."""

    rows_by_family: NestedRows = defaultdict(lambda: defaultdict(list))

    for row in rows:
        n_gen = int(row.get("n_gen", 0))
        avg_ts = row.get("avg_ts")
        if n_gen == 0 or avg_ts is None:
            continue

        samples_ts = [float(value) for value in row.get("samples_ts", [])]
        stddev_ts = row.get("stddev_ts")
        if stddev_ts is None:
            stddev_ts = stddev_or_zero(samples_ts)

        rows_by_family[str(row["_model_family"])][str(row["_model_variation"])].append(
            {
                "n_gen": float(n_gen),
                "avg_ts": float(avg_ts),
                "stddev_ts": float(stddev_ts),
            }
        )

    return rows_by_family


def load_p99_time_between_tokens_rows(rows: list[dict[str, Any]]) -> NestedRows:
    """Load p99 time-between-token rows grouped by model family and quantization.

    samples_ts are tokens/sec samples, so each sample's time between tokens is
    1 / samples_ts. The plot uses the 99th percentile of those sample times.
    """

    rows_by_family: NestedRows = defaultdict(lambda: defaultdict(list))

    for row in rows:
        n_gen = int(row.get("n_gen", 0))
        avg_ts = row.get("avg_ts")
        if n_gen <= 0 or avg_ts is None:
            continue

        samples_ts = [float(value) for value in row.get("samples_ts", []) if float(value) > 0]
        if samples_ts:
            token_times_ms = [1000 / sample_ts for sample_ts in samples_ts]
            p99_token_time_ms = percentile(token_times_ms, 99)
        else:
            p99_token_time_ms = 1000 / float(avg_ts)

        rows_by_family[str(row["_model_family"])][str(row["_model_variation"])].append(
            {
                "n_gen": float(n_gen),
                "p99_token_time_ms": p99_token_time_ms,
            }
        )

    return rows_by_family


def load_ttft_from_generation_rows(rows: list[dict[str, Any]]) -> NestedRows:
    """Estimate TTFT from the generation run itself.

    Estimate:
        total generation run time - decode time for tokens after the first

    Because avg_ts is tokens/sec, the per-token time is 1 / avg_ts.
    """

    rows_by_family: NestedRows = defaultdict(lambda: defaultdict(list))

    for row in rows:
        n_gen = int(row.get("n_gen", 0))
        avg_ts = row.get("avg_ts")
        avg_ns = row.get("avg_ns")
        if n_gen <= 0 or avg_ts is None:
            continue

        samples_ns = [float(value) for value in row.get("samples_ns", [])]
        samples_ts = [float(value) for value in row.get("samples_ts", [])]

        estimates_s: list[float] = []
        if samples_ns and samples_ts:
            for sample_ns, sample_ts in zip(samples_ns, samples_ts):
                if sample_ts > 0:
                    total_time_s = sample_ns / 1_000_000_000
                    remaining_decode_time_s = (n_gen - 1) / sample_ts
                    estimates_s.append(total_time_s - remaining_decode_time_s)

        if estimates_s:
            ttft_ms = mean(estimates_s) * 1000
            ttft_stddev_ms = stddev_or_zero(estimates_s) * 1000
        else:
            total_time_s = (float(avg_ns) / 1_000_000_000) if avg_ns is not None else n_gen / float(avg_ts)
            ttft_ms = (total_time_s - ((n_gen - 1) / float(avg_ts))) * 1000
            ttft_stddev_ms = 0.0

        rows_by_family[str(row["_model_family"])][str(row["_model_variation"])].append(
            {
                "n_gen": float(n_gen),
                "ttft_ms": ttft_ms,
                "ttft_stddev_ms": ttft_stddev_ms,
            }
        )

    return rows_by_family


def plot_generation_speed(rows_by_variation: dict[str, list[dict[str, float]]], output_path: Path, family_name: str) -> None:
    """Create and save the benchmark plot."""

    if not rows_by_variation:
        raise SystemExit("No benchmark rows found to plot.")

    fig, ax = plt.subplots(figsize=(11, 7))
    apply_dashboard_style(fig, ax)

    for index, (variation_name, rows) in enumerate(sorted(rows_by_variation.items())):
        rows = sorted(rows, key=lambda item: item["n_gen"])
        x_values = [row["n_gen"] for row in rows]
        y_values = [row["avg_ts"] for row in rows]
        y_errors = [row["stddev_ts"] for row in rows]

        ax.errorbar(
            x_values,
            y_values,
            yerr=y_errors,
            marker=MARKERS[index % len(MARKERS)],
            linestyle="-",
            linewidth=1.5,
            markersize=6,
            capsize=4,
            label=variation_name,
        )

    set_dashboard_title(ax, f"{family_name}: generation speed")
    ax.set_xlabel("Generated tokens")
    ax.set_ylabel("Average tokens per second")
    style_dashboard_legend(ax, title="Quantization", fontsize="small", ncols=3)
    save_dashboard_figure(fig, ax, output_path)
    plt.close(fig)


def plot_p99_time_between_tokens(
    rows_by_variation: dict[str, list[dict[str, float]]], output_path: Path, title: str
) -> None:
    """Create and save a p99 time-between-tokens plot."""

    if not rows_by_variation:
        raise SystemExit("No time-between-tokens rows found to plot.")

    fig, ax = plt.subplots(figsize=(11, 7))
    apply_dashboard_style(fig, ax)

    for index, (variation_name, rows) in enumerate(sorted(rows_by_variation.items())):
        rows = sorted(rows, key=lambda item: item["n_gen"])
        x_values = [row["n_gen"] for row in rows]
        y_values = [row["p99_token_time_ms"] for row in rows]

        ax.plot(
            x_values,
            y_values,
            marker=MARKERS[index % len(MARKERS)],
            linestyle="-",
            linewidth=1.5,
            markersize=6,
            label=variation_name,
        )

    set_dashboard_title(ax, title)
    ax.set_xlabel("Generated tokens")
    ax.set_ylabel("P99 time between tokens (ms/token)")
    style_dashboard_legend(ax, title="Quantization", fontsize="small", ncols=3)
    save_dashboard_figure(fig, ax, output_path)
    plt.close(fig)


def plot_ttft(rows_by_variation: dict[str, list[dict[str, float]]], output_path: Path, title: str) -> None:
    """Create and save a time-to-first-token plot."""

    if not rows_by_variation:
        raise SystemExit("No TTFT rows found to plot.")

    fig, ax = plt.subplots(figsize=(11, 7))
    apply_dashboard_style(fig, ax)

    for index, (variation_name, rows) in enumerate(sorted(rows_by_variation.items())):
        rows = sorted(rows, key=lambda item: item["n_gen"])
        x_values = [row["n_gen"] for row in rows]
        y_values = [row["ttft_ms"] for row in rows]
        y_errors = [row["ttft_stddev_ms"] for row in rows]

        ax.errorbar(
            x_values,
            y_values,
            yerr=y_errors,
            marker=MARKERS[index % len(MARKERS)],
            linestyle="-",
            linewidth=1.5,
            markersize=6,
            capsize=4,
            label=variation_name,
        )

    set_dashboard_title(ax, title)
    ax.set_xlabel("Generated tokens")
    ax.set_ylabel("Estimated time to first token (ms)")
    style_dashboard_legend(ax, title="Quantization", fontsize="small", ncols=3)
    save_dashboard_figure(fig, ax, output_path)
    plt.close(fig)


def plot_all_families(
    rows_by_family: NestedRows,
    output_dir: Path,
    metric_slug: str,
    metric_title: str,
    plot_function,
) -> list[Path]:
    output_paths: list[Path] = []

    for family_name, rows_by_variation in sorted(rows_by_family.items()):
        output_path = output_dir / f"{slugify(family_name)}_{metric_slug}.png"
        plot_function(rows_by_variation, output_path, f"{family_name}: {metric_title}")
        output_paths.append(output_path)

    return output_paths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot llama.cpp benchmark JSON files."
    )
    parser.add_argument(
        "json_files",
        nargs="*",
        type=Path,
        help="JSON files to parse. Defaults to all *.json files in the current device's throughput results.",
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
        help="Open an interactive matplotlib window after saving the plot.",
    )
    return parser.parse_args()


def default_json_paths() -> list[Path]:
    """Find benchmark JSON files using the repository's results folder layout."""

    results_dir = result_directory()
    if results_dir.exists():
        if results_dir.name == "results":
            return sorted(results_dir.glob("*/throughput/*.json"))

        throughput_paths = sorted((results_dir / "throughput").glob("*.json"))
        if throughput_paths:
            return throughput_paths

        # Also accept the older macOS layout where files lived directly in the
        # device result directory.
        return sorted(results_dir.glob("*.json"))

    return sorted(Path.cwd().glob("*.json"))


def device_plot_dir(json_path: Path, output_dir: Path) -> Path:
    """Map result files under results/<device>/ to benchmark_plots/<device>/."""

    results_dir = results_root()
    try:
        relative_path = json_path.resolve().relative_to(results_dir.resolve())
    except ValueError:
        return output_dir

    if len(relative_path.parts) > 1:
        return output_dir / relative_path.parts[0]

    return output_dir


def json_paths_by_output_dir(json_paths: list[Path], output_dir: Path) -> PathGroups:
    grouped_paths: PathGroups = defaultdict(list)

    for json_path in json_paths:
        grouped_paths[device_plot_dir(json_path, output_dir)].append(json_path)

    return dict(grouped_paths)


def save_plots(json_paths: list[Path], output_dir: Path) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = as_rows(json_paths)
    saved_paths: list[Path] = []

    speed_rows_by_model = load_generation_speed_rows(rows)
    for family_name, rows_by_variation in sorted(speed_rows_by_model.items()):
        output_path = output_dir / f"{slugify(family_name)}_tokens_per_second.png"
        plot_generation_speed(rows_by_variation, output_path, family_name)
        saved_paths.append(output_path)

    ttft_run_rows_by_model = load_ttft_from_generation_rows(rows)
    saved_paths.extend(plot_all_families(
        ttft_run_rows_by_model,
        output_dir,
        "ttft_from_generation_run",
        "estimated TTFT from generation run",
        plot_ttft,
    ))

    p99_time_between_tokens_rows = load_p99_time_between_tokens_rows(rows)
    saved_paths.extend(plot_all_families(
        p99_time_between_tokens_rows,
        output_dir,
        "p99_time_between_tokens",
        "p99 time between generated tokens",
        plot_p99_time_between_tokens,
    ))

    return saved_paths


def main() -> None:
    args = parse_args()
    json_paths = args.json_files or default_json_paths()
    if not json_paths:
        raise SystemExit(f"No benchmark JSON files found under {result_directory()}")

    saved_paths: list[Path] = []

    for output_dir, grouped_json_paths in sorted(json_paths_by_output_dir(json_paths, args.output_dir).items()):
        saved_paths.extend(save_plots(grouped_json_paths, output_dir))

    for path in saved_paths:
        print(f"Saved plot to {path.resolve()}")

    if args.show:
        plt.show()


if __name__ == "__main__":
    main()
