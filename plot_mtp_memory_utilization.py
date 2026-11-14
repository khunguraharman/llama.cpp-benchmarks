"""Plot full MTP VRAM timelines for n_predict=0 and n_predict=3."""

from __future__ import annotations

import argparse
import csv
import json
from math import isfinite
import re
from datetime import datetime
from pathlib import Path
from statistics import mean
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

from platform_paths import results_root
from plot_model_vram_timeline import IDLE_MIB
from plot_localai_style import (
    MUTED,
    apply_dashboard_style,
    save_dashboard_figure,
    set_dashboard_title,
    style_dashboard_legend,
)


DEFAULT_RESULTS = Path("results/nvidia-rtx-3080/mtp-speed-bench-results")
REPRESENTATIVE_N = 3
CONTEXT_TOKENS = 8192
# Base tensor payload of Jackrong/DeepSeek-V4-Pro-Qwen3.5-4B-MTP-GGUF,
# Q6_K file, excluding the final blk.32 MTP block. Full GPU offload.
BASE_WEIGHTS_MIB = 3453087744 / 2**20
# Eight full-attention layers, four KV heads, 256 elements per K or V head.
# Recurrent state and the separate MTP block's cache are outside the base KV.
BASE_KV_MIB = CONTEXT_TOKENS * 8 * 4 * (256 + 256) * 2 / 2**20
FONT_DIR = Path(__file__).resolve().parent / "benchmark_plots" / "fonts"

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


def read_memory_series(csv_path: Path) -> tuple[list[float], list[float], float | None]:
    timestamps: list[datetime] = []
    memory_values: list[float] = []
    capacity_mib = None

    with csv_path.open("r", encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file, skipinitialspace=True)
        for row in reader:
            cleaned_row = {
                str(key).strip(): value.strip() if isinstance(value, str) else value
                for key, value in row.items()
                if key is not None
            }
            try:
                timestamp = parse_timestamp(cleaned_row["timestamp"])
                memory_used = float(cleaned_row["memory_used_mib"])
                total = float(cleaned_row["memory_total_mib"])
            except (KeyError, TypeError, ValueError):
                continue
            if not (isfinite(memory_used) and isfinite(total) and 0 <= memory_used <= total and total > 0):
                continue
            if capacity_mib is not None and total != capacity_mib:
                raise ValueError(f"GPU capacity changed within {csv_path}")
            timestamps.append(timestamp)
            memory_values.append(memory_used)
            capacity_mib = total

    if not timestamps:
        return [], [], None

    start_time = timestamps[0]
    elapsed_s = [
        (timestamp - start_time).total_seconds()
        for timestamp in timestamps
    ]
    return elapsed_s, memory_values, capacity_mib


def load_rows(results_dir: Path) -> list[RunRow]:
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

        json_path = resolve_result_file(results_dir, str(source_row["json_ref"]), f"{fallback_stem}.json")
        if json_path is None:
            json_path = resolve_result_file(results_dir, "", f"{fallback_stem}_tokens.json")
        if has_multi_turn_samples(json_path):
            continue

        elapsed_s, memory_values, capacity_mib = read_memory_series(csv_path)
        if capacity_mib is None:
            continue

        rows.append(
            {
                "mtp_n": mtp_n,
                "bench": bench,
                "category": category,
                "osl": osl,
                "elapsed_s": elapsed_s,
                "memory_used_mib": memory_values,
                "capacity_mib": capacity_mib,
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


def annotate_base_memory(ax, baseline_mib: float) -> None:
    """Draw the base model's memory breakdown up to the measured mean."""
    components = [
        (IDLE_MIB, "#7b8c98", f"Idle system (1.4 GB) · {IDLE_MIB:,.0f} MiB", .08, "<->"),
        (BASE_WEIGHTS_MIB, "#82684d",
         f"Base weights (Q6_K) · {BASE_WEIGHTS_MIB:,.0f} MiB", .25, "<->"),
        (BASE_KV_MIB, "#c17b50",
         f"Base KV (8,192 tokens, FP16) · {BASE_KV_MIB:,.0f} MiB", .42, "|-|"),
    ]
    residual = baseline_mib - (IDLE_MIB + BASE_WEIGHTS_MIB + BASE_KV_MIB)
    if residual > 0:
        components.append(
            (residual, "#52635a", f"Misc. compute buffers and other VRAM · ~{residual:,.0f} MiB",
             .51, "|-|")
        )

    transform = ax.get_yaxis_transform()
    low = 0
    for size, color, label, x, arrowstyle in components:
        high = low + size
        ax.axhspan(low, high, color=color, alpha=.09, linewidth=0)
        ax.axhline(high, color=color, linewidth=1, linestyle="--")
        ax.annotate("", xy=(x, high), xytext=(x, low),
                    xycoords=transform, textcoords=transform,
                    arrowprops=dict(arrowstyle=arrowstyle, color=color, lw=1.5,
                                    mutation_scale=7, shrinkA=0, shrinkB=0))
        ax.text(x + .02, (low + high) / 2, label, transform=transform, va="center",
                fontsize=9, color=color,
                bbox=dict(facecolor="white", alpha=.9, edgecolor="none", pad=2))
        low = high


def plot_timeline(row: RunRow, output_path: Path, title: str, baseline_mib: float) -> None:
    fig, ax = plt.subplots(figsize=(11, 6.4))
    apply_dashboard_style(fig, ax)
    total = row["capacity_mib"]
    ax.plot(row["elapsed_s"], row["memory_used_mib"], linewidth=1.8,
            label="Measured device VRAM")
    if row["mtp_n"] == 0:
        annotate_base_memory(ax, baseline_mib)
        style_dashboard_legend(ax, loc="upper left")
    else:
        ax.axhspan(0, baseline_mib, color=MUTED, alpha=.09, linewidth=0)
        ax.axhline(baseline_mib, color=MUTED, linewidth=1.4, linestyle="--")
        transform = ax.get_yaxis_transform()
        ax.annotate("", xy=(.12, baseline_mib), xytext=(.12, 0),
                    xycoords=transform, textcoords=transform,
                    arrowprops=dict(arrowstyle="<->", color=MUTED, lw=1.5))
        ax.text(.14, baseline_mib / 2,
                f"Base model · {baseline_mib:,.0f} MiB mean\n"
                "8,192-token FP16 KV cache, Q6_K weights,\n"
                "idle system usage and compute buffers",
                transform=transform, va="center", color=MUTED, fontsize=10)
        trace = ax.lines[0]
        for label, height, color, offset in (
            ("n=3", trace.get_ydata()[-1], trace.get_color(), 4),
            ("n=0", baseline_mib, MUTED, -4),
        ):
            ax.annotate(label, xy=(.98, height), xycoords=transform,
                        xytext=(0, offset), textcoords="offset points", ha="right",
                        va="bottom" if offset > 0 else "top", color=color, fontsize=10)
    ax.set(xlabel="Time Since First GPU Sample [s]", ylabel="VRAM Used [MiB]",
           xlim=(0, max(row["elapsed_s"][-1], 1)), ylim=(0, total * 1.08))
    percent = ax.secondary_yaxis("right", functions=(lambda v: v / total * 100,
                                                    lambda p: p * total / 100))
    percent.set_ylabel("VRAM Capacity Used [%]", color=MUTED, fontsize=10)
    percent.tick_params(colors=MUTED, labelsize=9)
    set_dashboard_title(ax, title)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_dashboard_figure(fig, ax, output_path)
    fig.savefig(output_path.with_suffix(".svg"), facecolor=fig.get_facecolor(), edgecolor="none")
    plt.close(fig)


def save_plots(
    results_dir: Path,
    output_root: Path,
    device: str,
) -> list[Path]:
    rows = [row for row in load_rows(results_dir)
            if row["mtp_n"] in (0, REPRESENTATIVE_N)]
    if not rows:
        return []

    output_dir = output_dir_for(results_dir, output_root, device)
    saved_paths: list[Path] = []

    baseline_by_run = {
        (row["bench"], row["category"], row["osl"]): mean(row["memory_used_mib"])
        for row in rows if row["mtp_n"] == 0
    }
    missing_baselines = {
        (row["bench"], row["category"], row["osl"])
        for row in rows if (row["bench"], row["category"], row["osl"]) not in baseline_by_run
    }
    if missing_baselines:
        raise ValueError(f"Missing n=0 GPU logs for {sorted(missing_baselines)}")

    for row in rows:
        bench, category, osl = row["bench"], row["category"], row["osl"]
        n = row["mtp_n"]
        output_path = output_dir / "timelines" / (
            f"{slugify(bench)}_{slugify(category)}_osl{osl}_n{n}_vram_over_time.png"
        )
        title = (f"MTP n_predict={n} · {bench.replace('throughput_', '')} prompt · "
                 f"{category.replace('_', ' ')} · OSL {osl:,}")
        plot_timeline(row, output_path, title, baseline_by_run[bench, category, osl])
        saved_paths.append(output_path)

    return saved_paths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot MTP GPU memory utilization data.")
    parser.add_argument(
        "results_dir",
        nargs="?",
        type=Path,
        default=Path(__file__).resolve().parent / DEFAULT_RESULTS,
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
        default="nvidia-rtx-3080",
        help="Device name used in the output path.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    for font in FONT_DIR.glob("*-static.ttf"):
        font_manager.fontManager.addfont(str(font))
    saved_paths = save_plots(args.results_dir, args.output_dir, args.device)

    print(f"Done. Saved {len(saved_paths)} MTP memory utilization plot(s).", flush=True)
    for path in saved_paths:
        print(f"Saved plot to {path.resolve()}", flush=True)


if __name__ == "__main__":
    main()
