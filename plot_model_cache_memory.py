"""Plot observed RAM and VRAM during the model-speed runs."""

import csv
import json
from bisect import bisect_left, bisect_right
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

from plot_localai_style import (
    BACKGROUND, MUTED, SERIES_COLORS, apply_dashboard_style,
    set_dashboard_title, style_dashboard_legend,
)


ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "results/nvidia-rtx-3080/model-speed-bench-results"
OUTPUT = ROOT / "benchmark_plots/nvidia-rtx-3080/model-speed-memory"
MODELS = [
    ("Qwen3-4B-Q8_0", "Qwen3 4B"),
    ("Phi-3.5-mini-instruct-Q8_0", "Phi 3.5 mini"),
    ("Llama-3.2-3B-Instruct-Q8_0", "Llama 3.2 3B"),
]
# (layers, KV heads, head dimension); keys and values both use FP16.
KV_DIMENSIONS = {"Qwen3 4B": (36, 8, 128), "Phi 3.5 mini": (32, 32, 96),
                 "Llama 3.2 3B": (28, 8, 128)}


def style_reference_typography(ax):
    """Match the exported LocalAI memory/3080 titles and axes explicitly.

    Those SVGs use DejaVu Sans, even when dashboard fonts are installed.
    Keep this local so other dashboard plots retain their existing styling.
    """
    ax.title.set_fontfamily("DejaVu Sans")
    ax._left_title.set_fontfamily("DejaVu Sans")
    for label in (ax.xaxis.label, ax.yaxis.label):
        label.set_fontfamily("DejaVu Sans")
        label.set_fontsize(10)
    for label in (*ax.get_xticklabels(), *ax.get_yticklabels()):
        label.set_fontfamily("DejaVu Sans")
        label.set_fontsize(9)


def smooth_memory(times, values, window_seconds=9):
    """Centered time-window mean for display; raw values drive summaries."""
    prefix = [0.0]
    for value in values:
        prefix.append(prefix[-1] + value)
    result = []
    for time in times:
        left = bisect_left(times, time - window_seconds / 2)
        right = bisect_right(times, time + window_seconds / 2)
        result.append((prefix[right] - prefix[left]) / (right - left))
    return result


def sample_kv_reference(results, start, baseline, dims):
    """Accumulate each prompt at its cumulative request-latency endpoint.

    JSON has no request timestamps. Reconstruct elapsed time from sequential
    latencies, anchored to the first RAM sample, as for the average reference.
    This remains an uncapped reference: no eviction or reuse is subtracted.
    """
    layers, heads, dim = dims
    mib_per_token = 2 * layers * heads * dim * 2 / 2**20
    times, memory = [start], [baseline]
    for result in results:
        times.append(times[-1] + result["latency_s"])
        memory.append(memory[-1] + result["prompt_tokens"] * mib_per_token)
    return times, memory


def align_sample_reference(times, memory, ram_times, ram_memory):
    """Align the first prompt endpoint to the first cache-sized RAM increase.

    Use half the first prompt's theoretical KV size to distinguish the initial
    cache allocation from small startup fluctuations. Shift time only, without
    fitting the storage amounts or the subsequent growth rate.
    """
    threshold = ram_memory[0] + (memory[1] - memory[0]) / 2
    first_growth = next(t for t, value in zip(ram_times, ram_memory)
                        if value >= threshold)
    shift = first_growth - times[1]
    shifted_times = [t + shift for t in times]
    if shifted_times[0] > ram_times[0]:
        return [ram_times[0], *shifted_times], [memory[0], *memory], shift
    return shifted_times, memory, shift


def rows(path):
    with path.open(encoding="utf-8-sig", newline="") as handle:
        result = [{k.strip(): v.strip() for k, v in row.items()}
                  for row in csv.DictReader(handle)]
    if result and any("PDH-CSV" in key for key in result[0]):
        keys = list(result[0])
        memory_key = next(k for k in keys if "Working Set - Private" in k)
        return [{"timestamp_local": r[keys[0]], "private_working_set_bytes": r[memory_key]} for r in result]
    return result


def nearest_sample_index(times, target):
    """Return the closest recorded timestamp; prefer the earlier one on ties."""
    right = bisect_left(times, target)
    candidates = range(max(0, right - 1), min(len(times), right + 1))
    return min(candidates, key=lambda index: (abs(times[index] - target), index))


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    for font_path in (OUTPUT / "fonts").glob("*.ttf"):
        font_manager.fontManager.addfont(str(font_path))
    summaries = []
    series_data = []
    sample_references = []
    sample_time_shifts = []
    token_series = []
    token_records = []
    for (model, label), marker, color in zip(MODELS, ("o", "s", "^"), SERIES_COLORS):
        stem = SOURCE / f"{model}_throughput_1k_low_entropy_osl1"
        # Model names include a dot: append suffixes instead of replacing them.
        gpu = rows(Path(str(stem) + ".gpu.csv"))
        ram = rows(Path(str(stem) + ".ram.csv"))
        data = json.loads(Path(str(stem) + ".json").read_text())
        origin = datetime.strptime(gpu[0]["timestamp"], "%Y/%m/%d %H:%M:%S.%f")
        gx = [(datetime.strptime(r["timestamp"], "%Y/%m/%d %H:%M:%S.%f") - origin).total_seconds() for r in gpu]
        rx = [(datetime.strptime(r["timestamp_local"], "%m/%d/%Y %H:%M:%S.%f") - origin).total_seconds() for r in ram]
        rv = [int(r["private_working_set_bytes"]) / 2**20 for r in ram]
        gv = [float(r["memory_used_mib"]) for r in gpu]
        series_data.append((label, marker, rx, rv, gx, gv))
        successful = [r for r in data["results"] if r.get("ok")]
        dims = KV_DIMENSIONS[label]
        layers, heads, dim = dims
        sample_times, sample_memory = sample_kv_reference(successful, rx[0], rv[0], dims)
        sample_times, sample_memory, shift = align_sample_reference(
            sample_times, sample_memory, rx, rv)
        sample_time_shifts.append(dict(model=label, time_shift_seconds=shift))
        sample_references.append((label, color, sample_times, sample_memory))
        # Map each cumulative input-token count to measured RAM at the same
        # aligned sample endpoint using the nearest actual RAM reading.
        # Keep the original measured value; do not interpolate or smooth it.
        endpoint = rx[0] + shift
        cumulative_tokens = 0
        token_x, token_y = [0], [rv[0]]
        mib_per_token = 2 * layers * heads * dim * 2 / 2**20
        for sample, result in enumerate(successful, 1):
            endpoint += result["latency_s"]
            cumulative_tokens += result["prompt_tokens"]
            if rx[0] <= endpoint <= rx[-1]:
                ram_index = nearest_sample_index(rx, endpoint)
                measured = rv[ram_index]
                token_x.append(cumulative_tokens)
                token_y.append(measured)
                token_records.append(dict(model=label, sample=sample,
                    aligned_time_seconds=endpoint,
                    ram_sample_time_seconds=rx[ram_index],
                    ram_sample_time_error_seconds=rx[ram_index] - endpoint,
                    cumulative_input_tokens=cumulative_tokens,
                    measured_private_working_set_mib=measured,
                    calculated_private_working_set_mib=rv[0] + cumulative_tokens * mib_per_token))
        token_series.append((label, color, token_x, token_y,
                             cumulative_tokens, rv[0], mib_per_token))
        summaries.append(dict(model=label, completed_requests=data["completed_samples"],
                              ram_start_mib=rv[0], ram_end_mib=rv[-1],
                              ram_growth_mib=rv[-1]-rv[0],
                              vram_start_mib=gv[0], vram_end_mib=gv[-1],
                              vram_growth_mib=gv[-1]-gv[0],
                              vram_min_mib=min(gv), vram_max_mib=max(gv)))
    duration = max(max(s[2][-1], s[4][-1]) for s in series_data) * 1.02
    ram_ceiling = max(max(s[3]) for s in series_data) * 1.15
    fig, ax = plt.subplots(figsize=(12, 8))
    apply_dashboard_style(fig, ax)
    for label, marker, rx, rv, gx, gv in series_data:
        ax.plot(rx, smooth_memory(rx, rv), linewidth=2,
                label=f"{label}: Measured RAM")
    for label, color, times, memory in sample_references:
        ax.plot(times, memory, color=color, linestyle="--", linewidth=1.6,
                alpha=.85, label=f"{label}: Calculated FP16 KV (aligned)")
    ax.set(xlabel="Time Since First GPU Sample [s]", xlim=(0, duration),
           ylabel="Process Memory [MiB]", ylim=(0, ram_ceiling))
    set_dashboard_title(ax, "Windows Desktop: Process RAM vs. Time")
    ax.text(.23, .12,
            r"$M_{\mathrm{process}}(t_i) = M_{\mathrm{process},0} + m\sum_{j=1}^{i} N_j(t_i)$"
            "\n"
            r"$t_i = \Delta t_0 + \sum_{j=1}^{i} \ell_j$"
            "\n"
            r"$m = \frac{2\,L\,H_{\mathrm{KV}}\,d_{\mathrm{head}}\,s_{\mathrm{KV}}}{2^{20}}\quad[\mathrm{MiB/token}]$",
            transform=ax.transAxes, ha="left", va="bottom", fontsize=13,
            color=MUTED, linespacing=2.5)
    ax.text(.23, .035,
            r"$N_j$: prompt tokens; $\ell_j$: response latency; $\Delta t_0$: fixed initial time offset",
            transform=ax.transAxes, ha="left", va="bottom", fontsize=9,
            color=MUTED)
    style_reference_typography(ax)
    style_dashboard_legend(ax, loc="lower right", fontsize=12)
    fig.tight_layout()
    for extension in ("png", "svg"):
        fig.savefig(OUTPUT / f"ram-vs-time-per-sample.{extension}", dpi=180,
                    facecolor=BACKGROUND)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(12, 8))
    apply_dashboard_style(fig, ax)
    for label, color, tokens, memory, total, baseline, slope in token_series:
        ax.plot(tokens, memory, color=color, linewidth=2,
                label=f"{label}: Measured RAM")
    for label, color, tokens, memory, total, baseline, slope in token_series:
        ax.plot([0, total], [baseline, baseline + total * slope],
                color=color, linestyle="--", linewidth=1.6, alpha=.85,
                label=f"{label}: Calculated FP16 KV (m = {slope:.6f} MiB/token)")
    ax.set(xlabel="Cumulative Input Tokens",
           ylabel="Process Memory [MiB]",
           xlim=(0, max(s[4] for s in token_series) * 1.02),
           ylim=(0, ram_ceiling))
    ax.xaxis.set_major_formatter(matplotlib.ticker.StrMethodFormatter("{x:,.0f}"))
    set_dashboard_title(ax, "Windows Desktop: Process RAM vs. Cumulative Input Tokens")
    ax.text(.08, .085,
            r"$M_{\mathrm{process}}(N) = M_{\mathrm{process},0} + mN$"
            "\n"
            r"$m = \frac{2\,L\,H_{\mathrm{KV}}\,d_{\mathrm{head}}\,s_{\mathrm{KV}}}{2^{20}}\quad[\mathrm{MiB/token}]$",
            transform=ax.transAxes, ha="left", va="bottom", fontsize=13,
            color=MUTED, linespacing=1.7)
    style_reference_typography(ax)
    style_dashboard_legend(ax, loc="lower right", fontsize=12)
    fig.tight_layout()
    for extension in ("png", "svg"):
        fig.savefig(OUTPUT / f"ram-vs-input-tokens.{extension}", dpi=180,
                    facecolor=BACKGROUND)
    plt.close(fig)
    with (OUTPUT / "ram-vs-input-tokens.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(token_records[0]))
        writer.writeheader()
        writer.writerows(token_records)
    with (OUTPUT / "per-sample-time-shifts.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(sample_time_shifts[0]))
        writer.writeheader()
        writer.writerows(sample_time_shifts)
    with (OUTPUT / "memory-summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summaries[0]))
        writer.writeheader()
        writer.writerows(summaries)
    print(json.dumps(summaries, indent=2))
    print(OUTPUT)


if __name__ == "__main__":
    main()
