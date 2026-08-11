"""Plot throughput batch and microbatch benchmark results.

The script reads JSON files from throughput-batch-bench-results datasets and
creates one tokens-per-second plot for each model family. Each plot has one
series per varied batch parameter and includes +/- 1 standard deviation error
bars computed from samples_ts.
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


MARKERS = ["o", "s", "^", "D", "v", "P", "X", "*", "<", ">", "h", "8"]
RUN_NAME_RE = re.compile(r"_p(?P<prompt>\d+)_b(?P<batch>\d+)_ub(?P<ubatch>\d+)_n(?P<gen>\d+)_")

RunRow = dict[str, Any]


class Sweep:
    def __init__(self, key: str, label: str, constant_key: str, constant_label: str, slug: str) -> None:
        self.key = key
        self.label = label
        self.constant_key = constant_key
        self.constant_label = constant_label
        self.slug = slug


def stddev_or_zero(values: list[float]) -> float:
    return stdev(values) if len(values) > 1 else 0.0


def average_and_stddev(json_row: dict[str, Any]) -> tuple[float, float, list[float]]:
    samples_ts = [float(value) for value in json_row.get("samples_ts", [])]
    if samples_ts:
        return mean(samples_ts), stddev_or_zero(samples_ts), samples_ts

    return float(json_row["avg_ts"]), float(json_row.get("stddev_ts", 0.0)), []


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

    return model_stem, model_stem


def parse_run_values(path: Path) -> tuple[int | None, int | None, int | None, int | None]:
    match = RUN_NAME_RE.search(path.name)
    if not match:
        return None, None, None, None

    return (
        int(match.group("prompt")),
        int(match.group("batch")),
        int(match.group("ubatch")),
        int(match.group("gen")),
    )


def load_json_rows(json_path: Path) -> list[dict[str, Any]]:
    with json_path.open("r", encoding="utf-8") as file:
        try:
            data: Any = json.load(file)
        except json.JSONDecodeError:
            return []

    if isinstance(data, dict):
        data = [data]

    return [row for row in data if isinstance(row, dict)]


def load_rows(results_dir: Path) -> list[RunRow]:
    rows: list[RunRow] = []

    for json_path in sorted(results_dir.glob("*.json")):
        prompt_from_name, batch_from_name, ubatch_from_name, gen_from_name = parse_run_values(json_path)

        for json_row in load_json_rows(json_path):
            row_n_gen = json_row.get("n_gen")
            generated_tokens = int(row_n_gen if row_n_gen is not None else gen_from_name or 0)
            if generated_tokens <= 0:
                continue

            avg_ts, stddev_ts, samples_ts = average_and_stddev(json_row)

            model_family, model_variation = split_model_name(
                str(json_row.get("model_filename") or json_row.get("model_type") or json_path.stem)
            )

            rows.append(
                {
                    "model_family": model_family,
                    "model_variation": model_variation,
                    "tokens": generated_tokens,
                    "prompt_tokens": int(json_row.get("n_prompt") or prompt_from_name),
                    "batch_size": int(json_row.get("n_batch") or batch_from_name),
                    "microbatch_size": int(json_row.get("n_ubatch") or ubatch_from_name),
                    "avg_ts": avg_ts,
                    "stddev_ts": stddev_ts,
                    "samples_ts": samples_ts,
                    "json_path": json_path,
                }
            )

    return rows


def average_duplicate_token_rows(rows: list[RunRow]) -> list[RunRow]:
    rows_by_tokens: dict[int, list[RunRow]] = defaultdict(list)
    for row in rows:
        rows_by_tokens[int(row["tokens"])].append(row)

    averaged_rows: list[RunRow] = []
    for tokens, token_rows in sorted(rows_by_tokens.items()):
        samples_ts = [
            sample
            for row in token_rows
            for sample in row.get("samples_ts", [])
        ]
        if samples_ts:
            avg_ts = mean(samples_ts)
            stddev_ts = stddev_or_zero(samples_ts)
        else:
            avg_ts = mean(float(row["avg_ts"]) for row in token_rows)
            stddev_ts = stddev_or_zero([float(row["avg_ts"]) for row in token_rows])

        averaged_rows.append(
            {
                "tokens": tokens,
                "avg_ts": avg_ts,
                "stddev_ts": stddev_ts,
            }
        )

    return averaged_rows


def infer_sweep(rows: list[RunRow], results_dir: Path) -> Sweep:
    batch_sizes = {int(row["batch_size"]) for row in rows}
    microbatch_sizes = {int(row["microbatch_size"]) for row in rows}
    dir_name = results_dir.name.lower()

    if len(microbatch_sizes) > 1 and (len(batch_sizes) == 1 or "ubatch" in dir_name):
        return Sweep("microbatch_size", "Microbatch size (n_ubatch)", "batch_size", "n_batch", "ubatch")

    return Sweep("batch_size", "Batch size (n_batch)", "microbatch_size", "n_ubatch", "batch")


def y_limits(rows: list[RunRow]) -> tuple[float, float]:
    bounds: list[float] = []
    for row in rows:
        value = float(row["avg_ts"])
        error = float(row["stddev_ts"])
        bounds.extend((value - error, value + error))

    lower = min(bounds)
    upper = max(bounds)
    padding = max((upper - lower) * 0.05, 1.0)
    return lower - padding, upper + padding


def plot_model_family(model_family: str, model_rows: list[RunRow], output_path: Path, sweep: Sweep) -> None:
    fig, ax = plt.subplots(figsize=(11, 7))

    constant_values = sorted({int(row[sweep.constant_key]) for row in model_rows})
    series_values = sorted({int(row[sweep.key]) for row in model_rows})

    for index, series_value in enumerate(series_values):
        series_rows = [
            row for row in model_rows if int(row[sweep.key]) == series_value
        ]
        averaged_rows = average_duplicate_token_rows(series_rows)
        x_values = [int(row["tokens"]) for row in averaged_rows]
        y_values = [float(row["avg_ts"]) for row in averaged_rows]
        y_errors = [float(row["stddev_ts"]) for row in averaged_rows]

        label = f"{series_value}"
        (line,) = ax.plot(
            x_values,
            y_values,
            marker=MARKERS[index % len(MARKERS)],
            linewidth=1.5,
            markersize=6,
            label=label,
        )
        ax.errorbar(
            x_values,
            y_values,
            yerr=y_errors,
            color=line.get_color(),
            fmt="none",
            capsize=4,
        )

    ax.set_title(f"{model_family}: throughput by generated tokens ({sweep.slug} sweep)")
    ax.set_xlabel("Tokens generated")
    ax.set_ylabel("Tokens per second (+/- 1 std dev)")
    ax.set_ylim(*y_limits(model_rows))
    ax.grid(True, alpha=0.3)
    if len(constant_values) == 1:
        ax.text(
            0.99,
            0.02,
            f"{sweep.constant_label}={constant_values[0]}",
            transform=ax.transAxes,
            ha="right",
            va="bottom",
            fontsize="small",
            alpha=0.75,
        )
    ax.legend(title=sweep.label, fontsize="small")
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def output_dir_for(results_dir: Path, output_root: Path) -> Path:
    cwd_results = results_root()
    try:
        relative_path = results_dir.resolve().relative_to(cwd_results.resolve())
    except ValueError:
        return output_root

    if not relative_path.parts:
        return output_root

    device = relative_path.parts[0]
    if len(relative_path.parts) >= 3:
        return output_root / device / "throughput-batch" / relative_path.parts[-1]

    return output_root / device / "throughput-batch"


def save_plots(results_dir: Path, output_root: Path) -> list[Path]:
    rows = load_rows(results_dir)
    if not rows:
        return []

    sweep = infer_sweep(rows, results_dir)
    output_dir = output_dir_for(results_dir, output_root)
    output_dir.mkdir(parents=True, exist_ok=True)

    saved_paths: list[Path] = []
    rows_by_family: dict[str, list[RunRow]] = defaultdict(list)
    for row in rows:
        rows_by_family[str(row["model_family"])].append(row)

    for model_family, model_rows in sorted(rows_by_family.items()):
        output_path = output_dir / f"{slugify(model_family)}_tokens_per_second_by_{sweep.slug}_size.png"
        plot_model_family(model_family, model_rows, output_path, sweep)
        saved_paths.append(output_path)

    return saved_paths


def has_json_files(path: Path) -> bool:
    return any(path.glob("*.json"))


def expand_result_dirs(result_dirs: list[Path]) -> list[Path]:
    expanded_dirs: list[Path] = []
    for result_dir in result_dirs:
        if has_json_files(result_dir):
            expanded_dirs.append(result_dir)
            continue

        child_dirs = sorted(path for path in result_dir.iterdir() if path.is_dir() and has_json_files(path))
        if child_dirs:
            expanded_dirs.extend(child_dirs)
        else:
            expanded_dirs.append(result_dir)

    return expanded_dirs


def default_result_dirs() -> list[Path]:
    results_dir = result_directory()
    if results_dir.exists():
        if results_dir.name != "results":
            throughput_dir = results_dir / "throughput-batch-bench-results"
            return expand_result_dirs([throughput_dir]) if throughput_dir.is_dir() else []
        return expand_result_dirs(
            sorted(path for path in results_dir.glob("*/throughput-batch-bench-results") if path.is_dir())
        )

    return [Path.cwd()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot llama.cpp throughput batch benchmark results.")
    parser.add_argument(
        "result_dirs",
        nargs="*",
        type=Path,
        help="Result directories. Defaults to results/<device>/throughput-batch-bench-results.",
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        type=Path,
        default=Path("benchmark_plots"),
        help="Directory for generated plots. Defaults to benchmark_plots.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    saved_paths: list[Path] = []
    result_dirs = expand_result_dirs(args.result_dirs) if args.result_dirs else default_result_dirs()

    for results_dir in result_dirs:
        saved_paths.extend(save_plots(results_dir, args.output_dir))

    for path in saved_paths:
        print(f"Saved plot to {path.resolve()}")


if __name__ == "__main__":
    main()
