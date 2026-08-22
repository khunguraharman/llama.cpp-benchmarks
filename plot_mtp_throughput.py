"""Plot MTP accepted-token throughput by generated tokens.

The script reads MTP benchmark JSON files named like
``mtp_n3_throughput_1k_low_entropy_osl2048.json`` and creates one plot for
each benchmark/category pair. Each plot shows accepted draft tokens per second
versus generated tokens, with one series per MTP ``n`` value.
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
MTP_FILE_RE = re.compile(
    r"^mtp_n(?P<mtp_n>\d+)_(?P<bench>.+?)_(?P<category>.+?)_osl(?P<osl>\d+)\.json$",
    re.IGNORECASE,
)

RunRow = dict[str, Any]


def slugify(value: str) -> str:
    value = value.lower()
    value = re.sub(r"[^a-z0-9]+", "_", value)
    return value.strip("_")


def stddev_or_zero(values: list[float]) -> float:
    return stdev(values) if len(values) > 1 else 0.0


def load_json(path: Path) -> dict[str, Any] | None:
    with path.open("r", encoding="utf-8") as file:
        try:
            data = json.load(file)
        except json.JSONDecodeError:
            return None

    return data if isinstance(data, dict) else None


def parse_mtp_filename(path: Path) -> tuple[int, str, str, int] | None:
    match = MTP_FILE_RE.match(path.name)
    if not match:
        return None

    return (
        int(match.group("mtp_n")),
        match.group("bench"),
        match.group("category"),
        int(match.group("osl")),
    )


def accepted_tokens_per_second(result: dict[str, Any], metric: str) -> float | None:
    accepted_tokens = float(result.get("draft_n_accepted") or 0)
    if accepted_tokens <= 0:
        return None

    if metric == "server":
        predicted_ms = float(result.get("predicted_ms") or 0)
        if predicted_ms <= 0:
            return None
        return accepted_tokens / (predicted_ms / 1000)

    latency_s = float(result.get("latency_s") or 0)
    if latency_s <= 0:
        return None
    return accepted_tokens / latency_s


def is_single_turn_result(result: dict[str, Any]) -> bool:
    try:
        return int(result.get("turns") or 1) <= 1
    except (TypeError, ValueError):
        return True


def load_rows(results_dir: Path, metric: str) -> list[RunRow]:
    rows: list[RunRow] = []

    for json_path in sorted(results_dir.glob("mtp_n*.json")):
        filename_values = parse_mtp_filename(json_path)
        if filename_values is None:
            continue

        mtp_n, filename_bench, filename_category, osl = filename_values
        data = load_json(json_path)
        if data is None:
            continue

        config = data.get("config") if isinstance(data.get("config"), dict) else {}
        bench = str(config.get("bench") or filename_bench)
        configured_category = str(config.get("category") or filename_category)

        for index, result in enumerate(data.get("results", [])):
            if not isinstance(result, dict) or not result.get("ok", False):
                continue

            if not is_single_turn_result(result):
                continue

            completion_tokens = int(result.get("completion_tokens") or 0)
            if completion_tokens <= 0:
                continue

            throughput = accepted_tokens_per_second(result, metric)
            if throughput is None:
                continue

            result_category = str(result.get("category") or configured_category)
            category = result_category if configured_category == "all" else configured_category

            rows.append(
                {
                    "bench": bench,
                    "category": category,
                    "mtp_n": mtp_n,
                    "osl": osl,
                    "tokens": completion_tokens,
                    "throughput": throughput,
                    "json_path": json_path,
                    "result_index": index,
                }
            )

    return rows


def average_duplicate_token_rows(rows: list[RunRow]) -> list[RunRow]:
    rows_by_tokens: dict[int, list[RunRow]] = defaultdict(list)
    for row in rows:
        rows_by_tokens[int(row["tokens"])].append(row)

    averaged_rows: list[RunRow] = []
    for tokens, token_rows in sorted(rows_by_tokens.items()):
        values = [float(row["throughput"]) for row in token_rows]
        averaged_rows.append(
            {
                "tokens": tokens,
                "throughput": mean(values),
                "stddev": stddev_or_zero(values),
                "samples": len(values),
            }
        )

    return averaged_rows


def y_limits(rows: list[RunRow]) -> tuple[float, float]:
    bounds = [float(row["throughput"]) for row in rows]
    lower = min(bounds)
    upper = max(bounds)
    padding = max((upper - lower) * 0.06, 1.0)
    return lower - padding, upper + padding


def metric_label(metric: str) -> str:
    if metric == "server":
        return "Accepted draft tokens/s (server decode time)"
    return "Accepted draft tokens/s (end-to-end time)"


def plot_bench_category(
    bench: str,
    category: str,
    rows: list[RunRow],
    output_path: Path,
    metric: str,
    token_range_label: str,
) -> None:
    fig, ax = plt.subplots(figsize=(11, 7))
    apply_dashboard_style(fig, ax)

    mtp_values = sorted({int(row["mtp_n"]) for row in rows})
    for index, mtp_n in enumerate(mtp_values):
        series_rows = [row for row in rows if int(row["mtp_n"]) == mtp_n]
        averaged_rows = average_duplicate_token_rows(series_rows)
        x_values = [int(row["tokens"]) for row in averaged_rows]
        y_values = [float(row["throughput"]) for row in averaged_rows]
        ax.scatter(
            x_values,
            y_values,
            marker=MARKERS[index % len(MARKERS)],
            s=46,
            label=f"n={mtp_n}",
        )

    set_dashboard_title(
        ax,
        f"{bench} / {category}: accepted-token rate by generated tokens ({token_range_label})",
    )
    ax.set_xlabel("Tokens generated")
    ax.set_ylabel(metric_label(metric))
    ax.set_ylim(*y_limits(rows))
    style_dashboard_legend(ax, title="MTP")
    save_dashboard_figure(fig, ax, output_path)
    plt.close(fig)


def output_dir_for(results_dir: Path, output_root: Path) -> Path:
    cwd_results = results_root()
    try:
        relative_path = results_dir.resolve().relative_to(cwd_results.resolve())
    except ValueError:
        return output_root / "mtp-throughput"

    if relative_path.parts:
        return output_root / relative_path.parts[0] / "mtp-throughput"

    return output_root / "mtp-throughput"


def save_plots(results_dir: Path, output_root: Path, metric: str) -> list[Path]:
    rows = load_rows(results_dir, metric)
    if not rows:
        return []

    output_dir = output_dir_for(results_dir, output_root)
    output_dir.mkdir(parents=True, exist_ok=True)

    saved_paths: list[Path] = []
    rows_by_plot: dict[tuple[str, str], list[RunRow]] = defaultdict(list)
    for row in rows:
        rows_by_plot[(str(row["bench"]), str(row["category"]))].append(row)

    token_ranges = [
        ("under_1000_tokens", "under 1000 tokens", lambda row: int(row["tokens"]) < 1000),
        ("over_1000_tokens", "over 1000 tokens", lambda row: int(row["tokens"]) > 1000),
    ]

    for (bench, category), plot_rows in sorted(rows_by_plot.items()):
        for range_slug, range_label, row_matches_range in token_ranges:
            range_rows = [row for row in plot_rows if row_matches_range(row)]
            if not range_rows:
                continue

            output_path = (
                output_dir
                / f"{slugify(bench)}_{slugify(category)}_{metric}_accepted_tokens_per_second_{range_slug}.png"
            )
            plot_bench_category(bench, category, range_rows, output_path, metric, range_label)
            saved_paths.append(output_path)

    return saved_paths


def has_mtp_json_files(path: Path) -> bool:
    return any(path.glob("mtp_n*.json"))


def default_result_dirs() -> list[Path]:
    results_dir = result_directory()
    candidate_names = ("MTP throughput", "mtp")

    if results_dir.exists() and results_dir.name != "results":
        return [path for name in candidate_names if (path := results_dir / name).is_dir()]

    if results_dir.exists():
        return [
            path
            for device_dir in sorted(path for path in results_dir.iterdir() if path.is_dir())
            for name in candidate_names
            if (path := device_dir / name).is_dir()
        ]

    return [Path.cwd()]


def expand_result_dirs(result_dirs: list[Path]) -> list[Path]:
    expanded_dirs: list[Path] = []
    for result_dir in result_dirs:
        if has_mtp_json_files(result_dir):
            expanded_dirs.append(result_dir)
            continue

        child_dirs = sorted(path for path in result_dir.iterdir() if path.is_dir() and has_mtp_json_files(path))
        expanded_dirs.extend(child_dirs or [result_dir])

    return expanded_dirs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot MTP accepted-token rate versus generated tokens.")
    parser.add_argument(
        "result_dirs",
        nargs="*",
        type=Path,
        help="MTP result directories. Defaults to results/<device>/MTP throughput when present.",
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        type=Path,
        default=Path("benchmark_plots"),
        help="Directory for generated plots. Defaults to benchmark_plots.",
    )
    parser.add_argument(
        "--metric",
        choices=("server", "e2e"),
        default="server",
        help="Time basis for accepted-token rate. Defaults to server decode time.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result_dirs = expand_result_dirs(args.result_dirs) if args.result_dirs else default_result_dirs()
    saved_paths: list[Path] = []

    for results_dir in result_dirs:
        saved_paths.extend(save_plots(results_dir, args.output_dir, args.metric))

    for path in saved_paths:
        print(f"Saved plot to {path.resolve()}")


if __name__ == "__main__":
    main()
