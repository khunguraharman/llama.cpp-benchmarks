"""Plot maximum GPU memory usage summaries for benchmark runs.

Creates four plots:
1. prefill max memory vs prompt size, grouped by constant n_batch
2. prefill max memory vs prompt size, grouped by constant n_ubatch
3. throughput-batch max memory vs generated tokens, grouped by constant n_batch
4. throughput-batch max memory vs generated tokens, grouped by constant n_ubatch
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


MARKERS = ["o", "s", "^", "D", "v", "P", "X", "*", "<", ">", "h", "8"]
RUN_NAME_RE = re.compile(r"_p(?P<prompt>\d+)_b(?P<batch>\d+)_ub(?P<ubatch>\d+)_n(?P<gen>\d+)_")

RunRow = dict[str, Any]


def slugify(value: str) -> str:
    value = value.lower()
    value = re.sub(r"[^a-z0-9]+", "_", value)
    return value.strip("_")


def split_model_name(model_name: str) -> str:
    model_stem = Path(model_name).stem
    model_stem = re.sub(r"(?i)\.gguf$", "", model_stem)
    quant_match = re.search(r"(?i)(?:^|[-_\s])(Q\d(?:_[A-Z0-9]+)?|IQ\d_[A-Z0-9]+)$", model_stem)
    if quant_match:
        return model_stem[: quant_match.start()].strip("-_ ") or model_stem

    return model_stem


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


def gpu_csv_path_for(json_path: Path) -> Path | None:
    exact_path = json_path.with_suffix(".gpu.csv")
    if exact_path.exists():
        return exact_path

    candidates = sorted(json_path.parent.glob(f"{json_path.stem}.gpu*.csv"))
    return candidates[0] if candidates else None


def maximum_memory_used_mib(gpu_csv_path: Path | None) -> float | None:
    if gpu_csv_path is None or not gpu_csv_path.exists():
        return None

    values: list[float] = []
    with gpu_csv_path.open("r", encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file, skipinitialspace=True)
        for row in reader:
            cleaned_row = {
                str(key).strip(): value.strip() if isinstance(value, str) else value
                for key, value in row.items()
                if key is not None
            }
            memory_used = cleaned_row.get("memory_used_mib")
            if memory_used is None:
                continue

            try:
                values.append(float(memory_used))
            except ValueError:
                continue

    return max(values) if values else None


def load_rows(results_dir: Path) -> list[RunRow]:
    rows: list[RunRow] = []

    for json_path in sorted(results_dir.rglob("*.json")):
        prompt_from_name, batch_from_name, ubatch_from_name, gen_from_name = parse_run_values(json_path)
        gpu_csv_path = gpu_csv_path_for(json_path)
        max_memory_used_mib = maximum_memory_used_mib(gpu_csv_path)
        if max_memory_used_mib is None:
            continue

        for json_row in load_json_rows(json_path):
            model_name = str(json_row.get("model_filename") or json_row.get("model_type") or json_path.stem)
            prompt_tokens = int(json_row.get("n_prompt") or prompt_from_name or 0)
            generated_tokens = int(json_row.get("n_gen") or gen_from_name or 0)
            batch_size = int(json_row.get("n_batch") or batch_from_name or 0)
            ubatch_size = int(json_row.get("n_ubatch") or ubatch_from_name or 0)
            if prompt_tokens <= 0 or batch_size <= 0 or ubatch_size <= 0:
                continue

            rows.append(
                {
                    "model_family": split_model_name(model_name),
                    "prompt_tokens": prompt_tokens,
                    "generated_tokens": generated_tokens,
                    "batch_size": batch_size,
                    "ubatch_size": ubatch_size,
                    "max_memory_used_mib": max_memory_used_mib,
                    "json_path": json_path,
                    "gpu_csv_path": gpu_csv_path,
                }
            )

    return rows


def average_duplicate_points(rows: list[RunRow], x_key: str) -> list[tuple[int, float]]:
    values_by_x: dict[int, list[float]] = defaultdict(list)
    for row in rows:
        values_by_x[int(row[x_key])].append(float(row["max_memory_used_mib"]))

    return [
        (x_value, mean(values))
        for x_value, values in sorted(values_by_x.items())
    ]


def plot_grouped_summary(
    rows: list[RunRow],
    output_path: Path,
    title: str,
    x_key: str,
    x_label: str,
    group_key: str,
    group_label: str,
) -> None:
    fig, ax = plt.subplots(figsize=(13, 8))

    grouped_rows: dict[tuple[str, int], list[RunRow]] = defaultdict(list)
    for row in rows:
        grouped_rows[(str(row["model_family"]), int(row[group_key]))].append(row)

    for index, ((model_family, group_value), series_rows) in enumerate(sorted(grouped_rows.items())):
        points = average_duplicate_points(series_rows, x_key)
        if not points:
            continue

        x_values = [x_value for x_value, _memory in points]
        y_values = [memory for _x_value, memory in points]
        ax.plot(
            x_values,
            y_values,
            marker=MARKERS[index % len(MARKERS)],
            linewidth=1.4,
            markersize=5,
            label=f"{model_family}, {group_label}={group_value}",
        )

    ax.set_title(title)
    ax.set_xlabel(x_label)
    ax.set_ylabel("Maximum GPU memory used (MiB)")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize="x-small", ncols=2, loc="center left", bbox_to_anchor=(1.01, 0.5))
    fig.subplots_adjust(right=0.62)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def output_dir_for(output_root: Path, device: str) -> Path:
    return output_root / device / "memory-summaries"


def save_summary_plots(
    prefill_dir: Path,
    throughput_batch_dir: Path,
    output_root: Path,
    device: str,
) -> list[Path]:
    prefill_rows = load_rows(prefill_dir)
    throughput_rows = [
        row
        for row in load_rows(throughput_batch_dir)
        if int(row["generated_tokens"]) > 0
    ]

    output_dir = output_dir_for(output_root, device)
    plots = [
        (
            prefill_rows,
            output_dir / "prefill_max_memory_vs_prompt_by_batch_size.png",
            "Prefill: maximum GPU memory usage vs prompt size by batch size",
            "prompt_tokens",
            "Prompt tokens",
            "batch_size",
            "n_batch",
        ),
        (
            prefill_rows,
            output_dir / "prefill_max_memory_vs_prompt_by_ubatch_size.png",
            "Prefill: maximum GPU memory usage vs prompt size by microbatch size",
            "prompt_tokens",
            "Prompt tokens",
            "ubatch_size",
            "n_ubatch",
        ),
        (
            throughput_rows,
            output_dir / "throughput_batch_max_memory_vs_generated_tokens_by_batch_size.png",
            "Throughput batch: maximum GPU memory usage vs generated tokens by batch size",
            "generated_tokens",
            "Generated tokens",
            "batch_size",
            "n_batch",
        ),
        (
            throughput_rows,
            output_dir / "throughput_batch_max_memory_vs_generated_tokens_by_ubatch_size.png",
            "Throughput batch: maximum GPU memory usage vs generated tokens by microbatch size",
            "generated_tokens",
            "Generated tokens",
            "ubatch_size",
            "n_ubatch",
        ),
    ]

    saved_paths: list[Path] = []
    for rows, output_path, title, x_key, x_label, group_key, group_label in plots:
        if not rows:
            print(f"No rows available for {output_path.name}", flush=True)
            continue

        print(f"Saving {output_path}", flush=True)
        plot_grouped_summary(rows, output_path, title, x_key, x_label, group_key, group_label)
        saved_paths.append(output_path)

    return saved_paths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot max GPU memory usage benchmark summaries.")
    parser.add_argument(
        "--prefill-dir",
        type=Path,
        default=Path("results/nvidia-rtx-3080/prefill-bench-results"),
        help="Prefill result directory.",
    )
    parser.add_argument(
        "--throughput-batch-dir",
        type=Path,
        default=Path("results/nvidia-rtx-3080/throughput-batch-bench-results"),
        help="Throughput batch result directory.",
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        type=Path,
        default=Path("benchmark_plots"),
        help="Directory for generated plots. Defaults to benchmark_plots.",
    )
    parser.add_argument(
        "--device",
        default="nvidia-rtx-3080",
        help="Device name used in the output path.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    saved_paths = save_summary_plots(
        args.prefill_dir,
        args.throughput_batch_dir,
        args.output_dir,
        args.device,
    )

    print(f"Done. Saved {len(saved_paths)} summary plot(s).", flush=True)
    for path in saved_paths:
        print(f"Saved plot to {path.resolve()}", flush=True)


if __name__ == "__main__":
    main()
