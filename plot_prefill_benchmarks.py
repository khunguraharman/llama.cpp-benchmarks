"""Plot llama.cpp prefill benchmark results.

The script reads results/<device>/prefill-bench-results and creates one prefill
tokens-per-second plot for every available n_batch size. Each plot includes a
series for every n_ubatch size available at that n_batch size.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev
from typing import Any

import matplotlib.pyplot as plt


MARKERS = ["o", "s", "^", "D", "v", "P", "X", "*", "<", ">", "h", "8"]
RUN_NAME_RE = re.compile(r"_p(?P<prompt>\d+)_b(?P<batch>\d+)_ub(?P<ubatch>\d+)_")


RunRow = dict[str, Any]
SeriesRows = dict[tuple[int, int], list[RunRow]]


def stddev_or_zero(values: list[float]) -> float:
    return stdev(values) if len(values) > 1 else 0.0


def slugify(value: str) -> str:
    value = value.lower()
    value = re.sub(r"[^a-z0-9]+", "_", value)
    return value.strip("_")


def split_model_name(model_name: str) -> tuple[str, str]:
    model_stem = Path(model_name).stem
    model_stem = re.sub(r"(?i)\.gguf$", "", model_stem)

    quant_match = re.search(r"(?i)(?:^|[-_\s])(Q\d(?:_[A-Z0-9]+)?|IQ\d_[A-Z0-9]+)$", model_stem)
    if quant_match:
        family = model_stem[: quant_match.start()].strip("-_ ")
        variation = model_stem[quant_match.start() :].strip("-_ ")
        return family or model_stem, variation or model_stem

    parts = model_stem.split()
    for index, part in enumerate(parts):
        if re.match(r"^Q\d", part, flags=re.IGNORECASE):
            return " ".join(parts[:index]), " ".join(parts[index:])

    return model_stem, model_stem


def load_json_rows(json_path: Path) -> list[dict[str, Any]]:
    with json_path.open("r", encoding="utf-8") as file:
        try:
            data: Any = json.load(file)
        except json.JSONDecodeError:
            return []

    if isinstance(data, dict):
        data = [data]

    return [row for row in data if isinstance(row, dict)]


def parse_run_values(path: Path) -> tuple[int | None, int | None, int | None]:
    match = RUN_NAME_RE.search(path.name)
    if not match:
        return None, None, None

    return (
        int(match.group("prompt")),
        int(match.group("batch")),
        int(match.group("ubatch")),
    )


def maximum_memory_utilization(gpu_csv_path: Path) -> float | None:
    if not gpu_csv_path.exists():
        return None

    values: list[float] = []
    with gpu_csv_path.open("r", encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file, skipinitialspace=True)
        for row in reader:
            value = row.get("memory_utilization_percent")
            if value is None:
                continue
            try:
                values.append(float(value))
            except ValueError:
                continue

    return max(values) if values else None


def read_csv_dicts(csv_path: Path) -> list[dict[str, str]]:
    with csv_path.open("r", encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file, skipinitialspace=True)
        return [
            {
                str(key).strip(): value.strip() if isinstance(value, str) else value
                for key, value in row.items()
                if key is not None
            }
            for row in reader
        ]


def resolve_run_path(prefill_dir: Path, manifest_value: str) -> Path:
    path = Path(manifest_value.strip())
    if path.is_absolute():
        return path

    if path.parts and path.parts[0] == prefill_dir.name:
        return prefill_dir.parent / path

    return prefill_dir / path


def run_rows_from_manifest(prefill_dir: Path, runs_csv_path: Path) -> list[RunRow]:
    rows: list[RunRow] = []

    for manifest_row in read_csv_dicts(runs_csv_path):
        json_path = resolve_run_path(prefill_dir, manifest_row["bench_json"])
        gpu_csv_path = resolve_run_path(prefill_dir, manifest_row["gpu_csv"])
        if not json_path.exists():
            continue

        for json_row in load_json_rows(json_path):
            prompt_tokens = int(json_row.get("n_prompt") or manifest_row["prompt_tokens"])
            batch_size = int(json_row.get("n_batch") or manifest_row["batch_size"])
            ubatch_size = int(json_row.get("n_ubatch") or manifest_row["ubatch_size"])
            samples_ts = [float(value) for value in json_row.get("samples_ts", [])]
            stddev_ts = json_row.get("stddev_ts")
            if stddev_ts is None:
                stddev_ts = stddev_or_zero(samples_ts)

            rows.append(
                {
                    "model_name": str(json_row.get("model_filename") or manifest_row["model"] or json_row.get("model_type") or json_path.stem),
                    "prompt_tokens": prompt_tokens,
                    "batch_size": batch_size,
                    "ubatch_size": ubatch_size,
                    "avg_ts": float(json_row["avg_ts"]),
                    "stddev_ts": float(stddev_ts),
                    "max_memory_utilization": maximum_memory_utilization(gpu_csv_path),
                    "json_path": json_path,
                    "gpu_csv_path": gpu_csv_path,
                }
            )

    return rows


def run_rows_from_json_files(prefill_dir: Path) -> list[RunRow]:
    rows: list[RunRow] = []

    for json_path in sorted(prefill_dir.glob("*.json")):
        prompt_from_name, batch_from_name, ubatch_from_name = parse_run_values(json_path)
        gpu_csv_path = json_path.with_suffix(".gpu.csv")

        for json_row in load_json_rows(json_path):
            samples_ts = [float(value) for value in json_row.get("samples_ts", [])]
            stddev_ts = json_row.get("stddev_ts")
            if stddev_ts is None:
                stddev_ts = stddev_or_zero(samples_ts)

            rows.append(
                {
                    "model_name": str(json_row.get("model_filename") or json_row.get("model_type") or json_path.stem),
                    "prompt_tokens": int(json_row.get("n_prompt") or prompt_from_name),
                    "batch_size": int(json_row.get("n_batch") or batch_from_name),
                    "ubatch_size": int(json_row.get("n_ubatch") or ubatch_from_name),
                    "avg_ts": float(json_row["avg_ts"]),
                    "stddev_ts": float(stddev_ts),
                    "max_memory_utilization": maximum_memory_utilization(gpu_csv_path),
                    "json_path": json_path,
                    "gpu_csv_path": gpu_csv_path,
                }
            )

    return rows


def load_prefill_rows(prefill_dir: Path) -> list[RunRow]:
    runs_csv_path = prefill_dir / "runs.csv"
    rows: list[RunRow] = []
    if runs_csv_path.exists():
        rows.extend(run_rows_from_manifest(prefill_dir, runs_csv_path))

    rows.extend(run_rows_from_json_files(prefill_dir))

    deduplicated_rows: dict[tuple[Path, int, int, int], RunRow] = {}
    for row in rows:
        key = (
            row["json_path"].resolve(),
            int(row["prompt_tokens"]),
            int(row["batch_size"]),
            int(row["ubatch_size"]),
        )
        deduplicated_rows.setdefault(key, row)

    return list(deduplicated_rows.values())


def group_rows_by_model(rows: list[RunRow]) -> dict[str, list[RunRow]]:
    grouped_rows: dict[str, list[RunRow]] = defaultdict(list)

    for row in rows:
        model_family, model_variation = split_model_name(row["model_name"])
        row["model_family"] = model_family
        row["model_variation"] = model_variation
        grouped_rows[model_family].append(row)

    return dict(grouped_rows)


def choose_series(rows: list[RunRow], batch_size: int) -> SeriesRows:
    rows_by_pair: SeriesRows = defaultdict(list)
    for row in rows:
        if row["batch_size"] == batch_size:
            rows_by_pair[(row["batch_size"], row["ubatch_size"])].append(row)

    return {
        pair: rows_by_pair[pair]
        for pair in sorted(rows_by_pair, key=lambda pair: pair[1])
    }


def average_duplicate_prompt_rows(rows: list[RunRow], metric_key: str) -> list[dict[str, float]]:
    rows_by_prompt: dict[int, list[RunRow]] = defaultdict(list)
    for row in rows:
        if row.get(metric_key) is not None:
            rows_by_prompt[row["prompt_tokens"]].append(row)

    averaged_rows: list[dict[str, float]] = []
    for prompt_tokens, prompt_rows in sorted(rows_by_prompt.items()):
        y_values = [float(row[metric_key]) for row in prompt_rows]
        averaged_rows.append(
            {
                "prompt_tokens": float(prompt_tokens),
                metric_key: mean(y_values),
                "stddev_ts": mean(float(row.get("stddev_ts", 0.0)) for row in prompt_rows),
            }
        )

    return averaged_rows


def plot_metric(
    series_rows: SeriesRows,
    output_path: Path,
    title: str,
    metric_key: str,
    y_label: str,
    include_error_bars: bool = False,
    y_limits: tuple[float, float] | None = None,
) -> None:
    fig, ax = plt.subplots(figsize=(11, 7))

    for index, ((_batch_size, ubatch_size), rows) in enumerate(series_rows.items()):
        averaged_rows = average_duplicate_prompt_rows(rows, metric_key)
        if not averaged_rows:
            continue

        x_values = [row["prompt_tokens"] for row in averaged_rows]
        y_values = [row[metric_key] for row in averaged_rows]
        label = f"ubatch={ubatch_size}"

        if include_error_bars:
            y_errors = [row["stddev_ts"] for row in averaged_rows]
            y_lower = [y_value - y_error for y_value, y_error in zip(y_values, y_errors)]
            y_upper = [y_value + y_error for y_value, y_error in zip(y_values, y_errors)]
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
                y_lower,
                y_upper,
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
        else:
            ax.plot(
                x_values,
                y_values,
                marker=MARKERS[index % len(MARKERS)],
                linewidth=1.5,
                markersize=6,
                label=label,
            )

    ax.set_title(title)
    ax.set_xlabel("Prompt tokens")
    ax.set_ylabel(y_label)
    if y_limits is not None:
        ax.set_ylim(*y_limits)
    ax.grid(True, alpha=0.3)
    ax.legend(title="Microbatch size", fontsize="small")
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)


def metric_limits(
    rows: list[RunRow],
    metric_key: str,
    include_error_bars: bool = False,
) -> tuple[float, float] | None:
    bounds: list[float] = []
    for row in rows:
        metric_value = row.get(metric_key)
        if metric_value is None:
            continue

        value = float(metric_value)
        error = float(row.get("stddev_ts", 0.0)) if include_error_bars else 0.0
        bounds.extend((value - error, value + error))

    if not bounds:
        return None

    lower = min(bounds)
    upper = max(bounds)
    padding = max((upper - lower) * 0.05, 1.0)
    return lower - padding, upper + padding


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


def save_plots(
    prefill_dir: Path,
    output_dir: Path,
    requested_batch_sizes: list[int] | None,
) -> list[Path]:
    rows = load_prefill_rows(prefill_dir)
    if not rows:
        return []

    output_dir.mkdir(parents=True, exist_ok=True)
    saved_paths: list[Path] = []

    for model_family, model_rows in sorted(group_rows_by_model(rows).items()):
        model_slug = slugify(model_family)
        available_batch_sizes = sorted({int(row["batch_size"]) for row in model_rows})
        if requested_batch_sizes:
            requested = set(requested_batch_sizes)
            batch_sizes = [size for size in available_batch_sizes if size in requested]
        else:
            batch_sizes = available_batch_sizes

        y_limits = metric_limits(model_rows, "avg_ts", include_error_bars=True)
        for batch_size in batch_sizes:
            series_rows = choose_series(model_rows, batch_size)
            if not series_rows:
                continue

            speed_path = output_dir / f"{model_slug}_prefill_tokens_per_second_b{batch_size}.png"
            plot_metric(
                series_rows,
                speed_path,
                f"{model_family}: prefill tokens per second (batch={batch_size})",
                "avg_ts",
                "Tokens per second (+/- 1 std dev)",
                include_error_bars=True,
                y_limits=y_limits,
            )
            saved_paths.append(speed_path)

    return saved_paths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot llama.cpp prefill benchmark results.")
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
        "--batch-size",
        dest="batch_sizes",
        action="append",
        type=int,
        help="Only plot this batch size. May be repeated; defaults to every available batch size.",
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
        saved_paths.extend(save_plots(prefill_dir, output_dir, args.batch_sizes))

    for path in saved_paths:
        print(f"Saved plot to {path.resolve()}")

    if args.show:
        plt.show()


if __name__ == "__main__":
    main()
