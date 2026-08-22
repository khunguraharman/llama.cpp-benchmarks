"""Plot MTP GPU memory utilization timelines and peak summaries.

The script reads the memory timeline data captured for MTP runs, with one
series per MTP ``n`` value. It creates:

1. memory utilization over time for each benchmark/category/output length
2. peak memory utilization vs prompt tokens
3. peak memory utilization vs generated tokens
4. peak memory utilization vs estimated context window
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from statistics import mean
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from platform_paths import device_name, result_directory, results_root
from plot_style import (
    apply_dashboard_style,
    save_dashboard_figure,
    set_dashboard_title,
    style_dashboard_legend,
)


MARKERS = ["o", "s", "^", "D", "v", "P", "X", "*", "<", ">", "h", "8"]
TIMESTAMP_FORMATS = (
    "%Y/%m/%d %H:%M:%S.%f",
    "%Y-%m-%d %H:%M:%S.%f",
    "%Y/%m/%d %H:%M:%S",
    "%Y-%m-%d %H:%M:%S",
)
MTP_CSV_RE = re.compile(
    r"^mtp_n(?P<mtp_n>\d+)_(?P<bench>.+?)_(?P<category>.+?)_osl(?P<osl>\d+)\.gpu\.csv$",
    re.IGNORECASE,
)
THROUGHPUT_PROMPT_RE = re.compile(r"throughput_(?P<prompt_k>\d+)k$", re.IGNORECASE)

RunRow = dict[str, Any]


def slugify(value: str) -> str:
    value = value.lower()
    value = re.sub(r"[^a-z0-9]+", "_", value)
    return value.strip("_")


def parse_timestamp(value: str) -> datetime:
    value = value.strip()
    for timestamp_format in TIMESTAMP_FORMATS:
        try:
            return datetime.strptime(value, timestamp_format)
        except ValueError:
            pass

    raise ValueError(f"Unsupported timestamp format: {value!r}")


def parse_mtp_csv_filename(path: Path) -> tuple[int, str, str, int] | None:
    match = MTP_CSV_RE.match(path.name)
    if not match:
        return None

    return (
        int(match.group("mtp_n")),
        match.group("bench"),
        match.group("category"),
        int(match.group("osl")),
    )


def load_manifest_rows(results_dir: Path) -> list[RunRow]:
    manifest_path = results_dir / "runs.csv"
    if not manifest_path.exists():
        return []

    rows: list[RunRow] = []
    with manifest_path.open("r", encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file, skipinitialspace=True)
        for row in reader:
            cleaned_row = {
                str(key).strip(): value.strip() if isinstance(value, str) else value
                for key, value in row.items()
                if key is not None
            }
            try:
                rows.append(
                    {
                        "mtp_n": int(cleaned_row["mtp_n_max"]),
                        "bench": str(cleaned_row["bench"]),
                        "category": str(cleaned_row["category"]),
                        "osl": int(cleaned_row["osl"]),
                        "json_ref": str(cleaned_row.get("json") or ""),
                        "gpu_csv_ref": str(cleaned_row.get("gpu_csv") or ""),
                    }
                )
            except (KeyError, TypeError, ValueError):
                continue

    return rows


def resolve_result_file(results_dir: Path, reference: str, fallback_name: str) -> Path | None:
    candidates: list[Path] = []
    if reference:
        reference_path = Path(reference)
        candidates.append(results_dir / reference_path)
        candidates.append(results_dir / reference_path.name)
    candidates.append(results_dir / fallback_name)

    for candidate in candidates:
        if candidate.exists():
            return candidate

    return None


def load_token_summary(
    json_path: Path | None,
    bench: str,
    generated_tokens: int,
    context_window_limit: int | None,
) -> tuple[int, int, int]:
    prompt_tokens = prompt_tokens_from_bench(bench)
    completion_tokens = generated_tokens
    context_window = prompt_tokens + completion_tokens

    if json_path is None or not json_path.exists():
        return prompt_tokens, completion_tokens, capped_context_window(context_window, context_window_limit)

    with json_path.open("r", encoding="utf-8") as file:
        try:
            data = json.load(file)
        except json.JSONDecodeError:
            return prompt_tokens, completion_tokens, capped_context_window(context_window, context_window_limit)

    results = data.get("results") if isinstance(data, dict) else None
    if not isinstance(results, list):
        return prompt_tokens, completion_tokens, capped_context_window(context_window, context_window_limit)

    prompt_values: list[int] = []
    completion_values: list[int] = []
    context_values: list[int] = []
    for result in results:
        if not isinstance(result, dict) or not result.get("ok", False):
            continue

        if not is_single_turn_result(result):
            continue

        try:
            prompt = int(result.get("prompt_tokens") or 0)
            completion = int(result.get("completion_tokens") or 0)
            total = int(result.get("total_tokens") or (prompt + completion))
        except (TypeError, ValueError):
            continue

        if prompt > 0:
            prompt_values.append(prompt)
        if completion > 0:
            completion_values.append(completion)
        if total > 0:
            context_values.append(total)

    if prompt_values:
        prompt_tokens = max(prompt_values)
    if completion_values:
        completion_tokens = max(completion_values)
    if context_values:
        context_window = max(context_values)
    else:
        context_window = prompt_tokens + completion_tokens

    return prompt_tokens, completion_tokens, capped_context_window(context_window, context_window_limit)


def is_single_turn_result(result: dict[str, Any]) -> bool:
    try:
        return int(result.get("turns") or 1) <= 1
    except (TypeError, ValueError):
        return True


def has_multi_turn_samples(json_path: Path | None) -> bool:
    if json_path is None or not json_path.exists():
        return False

    with json_path.open("r", encoding="utf-8") as file:
        try:
            data = json.load(file)
        except json.JSONDecodeError:
            return False

    results = data.get("results") if isinstance(data, dict) else None
    if not isinstance(results, list):
        return False

    return any(
        isinstance(result, dict)
        and result.get("ok", False)
        and not is_single_turn_result(result)
        for result in results
    )


def capped_context_window(context_window: int, context_window_limit: int | None) -> int:
    if context_window_limit is None or context_window_limit <= 0:
        return context_window

    return min(context_window, context_window_limit)


def prompt_tokens_from_bench(bench: str) -> int:
    match = THROUGHPUT_PROMPT_RE.match(bench)
    if match:
        return int(match.group("prompt_k")) * 1024

    return 0


def read_memory_series(csv_path: Path) -> tuple[list[float], list[float], float | None]:
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
        return [], [], None

    start_time = timestamps[0]
    elapsed_s = [
        (timestamp - start_time).total_seconds()
        for timestamp in timestamps
    ]
    return elapsed_s, memory_values, max(memory_values)


def load_rows(results_dir: Path, context_window_limit: int | None = 8192) -> list[RunRow]:
    rows: list[RunRow] = []
    manifest_rows = load_manifest_rows(results_dir)

    if manifest_rows:
        source_rows = manifest_rows
    else:
        source_rows = []
        for csv_path in sorted(results_dir.glob("mtp_n*.gpu.csv")):
            parsed_values = parse_mtp_csv_filename(csv_path)
            if parsed_values is None:
                continue
            mtp_n, bench, category, osl = parsed_values
            source_rows.append(
                {
                    "mtp_n": mtp_n,
                    "bench": bench,
                    "category": category,
                    "osl": osl,
                    "json_ref": "",
                    "gpu_csv_ref": csv_path.name,
                }
            )

    for source_row in source_rows:
        mtp_n = int(source_row["mtp_n"])
        bench = str(source_row["bench"])
        category = str(source_row["category"])
        osl = int(source_row["osl"])
        fallback_stem = f"mtp_n{mtp_n}_{bench}_{category}_osl{osl}"

        csv_path = resolve_result_file(results_dir, str(source_row["gpu_csv_ref"]), f"{fallback_stem}.gpu.csv")
        if csv_path is None:
            continue

        json_path = resolve_result_file(results_dir, str(source_row["json_ref"]), f"{fallback_stem}_tokens.json")
        if has_multi_turn_samples(json_path):
            continue

        elapsed_s, memory_values, peak_memory_utilization = read_memory_series(csv_path)
        if peak_memory_utilization is None:
            continue

        prompt_tokens, generated_tokens, context_window = load_token_summary(
            json_path,
            bench,
            osl,
            context_window_limit,
        )
        rows.append(
            {
                "mtp_n": mtp_n,
                "bench": bench,
                "category": category,
                "osl": osl,
                "prompt_tokens": prompt_tokens,
                "generated_tokens": generated_tokens,
                "context_window": context_window,
                "elapsed_s": elapsed_s,
                "memory_utilization": memory_values,
                "peak_memory_utilization": peak_memory_utilization,
                "csv_path": csv_path,
                "json_path": json_path,
            }
        )

    return rows


def output_dir_for(results_dir: Path, output_root: Path, device: str) -> Path:
    try:
        relative_path = results_dir.resolve().relative_to(results_root().resolve())
    except ValueError:
        return output_root / device / "mtp-memory-utilization"

    if relative_path.parts:
        return output_root / relative_path.parts[0] / "mtp-memory-utilization"

    return output_root / device / "mtp-memory-utilization"


def average_duplicate_points(rows: list[RunRow], x_key: str) -> list[tuple[int, float]]:
    values_by_x: dict[int, list[float]] = defaultdict(list)
    for row in rows:
        x_value = int(row[x_key])
        if x_value <= 0:
            continue
        values_by_x[x_value].append(float(row["peak_memory_utilization"]))

    return [
        (x_value, mean(values))
        for x_value, values in sorted(values_by_x.items())
    ]


def plot_timeline(rows: list[RunRow], output_path: Path, title: str) -> bool:
    fig, ax = plt.subplots(figsize=(12, 6.5))
    apply_dashboard_style(fig, ax)
    plotted_count = 0

    for index, row in enumerate(sorted(rows, key=lambda value: int(value["mtp_n"]))):
        elapsed_s = row["elapsed_s"]
        memory_values = row["memory_utilization"]
        if not elapsed_s or not memory_values:
            continue

        ax.scatter(
            elapsed_s,
            memory_values,
            marker=MARKERS[index % len(MARKERS)],
            alpha=0.9,
            s=16,
            label=f"n={row['mtp_n']}",
        )
        plotted_count += 1

    if plotted_count == 0:
        plt.close(fig)
        return False

    set_dashboard_title(ax, title)
    ax.set_xlabel("Time since logging started (s)")
    ax.set_ylabel("Memory utilization (%)")
    style_dashboard_legend(ax, title="MTP")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_dashboard_figure(fig, ax, output_path)
    plt.close(fig)
    return True


def plot_peak_summary(
    rows: list[RunRow],
    output_path: Path,
    x_key: str,
    x_label: str,
    title: str,
) -> bool:
    fig, ax = plt.subplots(figsize=(11.5, 6.5))
    apply_dashboard_style(fig, ax)
    plotted_count = 0

    for index, mtp_n in enumerate(sorted({int(row["mtp_n"]) for row in rows})):
        series_rows = [row for row in rows if int(row["mtp_n"]) == mtp_n]
        points = average_duplicate_points(series_rows, x_key)
        if not points:
            continue

        x_values = [x_value for x_value, _memory in points]
        y_values = [memory for _x_value, memory in points]
        ax.scatter(
            x_values,
            y_values,
            marker=MARKERS[index % len(MARKERS)],
            s=48,
            label=f"n={mtp_n}",
        )
        plotted_count += 1

    if plotted_count == 0:
        plt.close(fig)
        return False

    set_dashboard_title(ax, title)
    ax.set_xlabel(x_label)
    ax.set_ylabel("Peak memory utilization (%)")
    style_dashboard_legend(ax, title="MTP")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_dashboard_figure(fig, ax, output_path)
    plt.close(fig)
    return True


def save_plots(
    results_dir: Path,
    output_root: Path,
    device: str,
    context_window_limit: int | None,
) -> list[Path]:
    rows = load_rows(results_dir, context_window_limit)
    if not rows:
        return []

    output_dir = output_dir_for(results_dir, output_root, device)
    saved_paths: list[Path] = []

    timeline_rows: dict[tuple[str, str, int], list[RunRow]] = defaultdict(list)
    for row in rows:
        timeline_rows[(str(row["bench"]), str(row["category"]), int(row["osl"]))].append(row)

    for (bench, category, osl), plot_rows in sorted(timeline_rows.items()):
        output_path = output_dir / "timelines" / (
            f"{slugify(bench)}_{slugify(category)}_osl{osl}_memory_utilization_over_time.png"
        )
        title = f"{bench} / {category} / osl={osl}: memory utilization over time"
        if plot_timeline(plot_rows, output_path, title):
            saved_paths.append(output_path)

    summary_specs = [
        (
            "prompt_tokens",
            "Prompt tokens",
            "mtp_peak_memory_utilization_vs_prompt_tokens.png",
            "MTP peak memory utilization vs prompt tokens",
        ),
        (
            "generated_tokens",
            "Tokens generated",
            "mtp_peak_memory_utilization_vs_generated_tokens.png",
            "MTP peak memory utilization vs generated tokens",
        ),
        (
            "context_window",
            "Estimated active context tokens",
            "mtp_peak_memory_utilization_vs_context_window.png",
            "MTP peak memory utilization vs active context",
        ),
    ]

    for x_key, x_label, filename, title in summary_specs:
        output_path = output_dir / filename
        if plot_peak_summary(rows, output_path, x_key, x_label, title):
            saved_paths.append(output_path)

    return saved_paths


def default_results_dir() -> Path:
    selected_results_dir = result_directory()
    if selected_results_dir.name == "results":
        selected_results_dir = selected_results_dir / device_name()

    return selected_results_dir / "MTP memory utilization"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot MTP GPU memory utilization data.")
    parser.add_argument(
        "results_dir",
        nargs="?",
        type=Path,
        default=default_results_dir(),
        help="MTP memory utilization result directory.",
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
        default=device_name(),
        help="Device name used in the output path.",
    )
    parser.add_argument(
        "--context-window-limit",
        type=int,
        default=8192,
        help=(
            "Cap estimated active context tokens at the server context window. "
            "Use 0 to disable the cap. Defaults to 8192."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    context_window_limit = args.context_window_limit if args.context_window_limit > 0 else None
    saved_paths = save_plots(args.results_dir, args.output_dir, args.device, context_window_limit)

    print(f"Done. Saved {len(saved_paths)} MTP memory utilization plot(s).", flush=True)
    for path in saved_paths:
        print(f"Saved plot to {path.resolve()}", flush=True)


if __name__ == "__main__":
    main()
