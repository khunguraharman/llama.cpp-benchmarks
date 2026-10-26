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
    BACKGROUND, HEADING, MUTED, apply_dashboard_style,
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


def add_kv_references(ax, references, ceiling):
    """Show an uncapped accumulation reference, not a fitted RAM prediction."""
    for label, color, start, baseline, rate, prompt, dims in references:
        end = start + (ceiling - baseline) / rate
        ax.plot([start, end], [baseline, ceiling], color=color, linestyle="--",
                linewidth=1.6, alpha=.85,
                label=f"{label} FP16 KV reference: {rate:.1f} [MiB/s]")


def reference_caption(references):
    lines = ["FP16 KV reference slope = r [prompts/s] × T_mean [tokens/prompt] × 2 × L × H_KV × D × 2 / 2^20 [MiB/s]"]
    for label, color, start, baseline, rate, prompt, dims in references:
        layers, heads, dim = dims
        lines.append(f"{label}: T_mean = {prompt:.2f}; L × H_KV × D = {layers} × {heads} × {dim}")
    return "\n".join(lines)


def rows(path):
    with path.open(encoding="utf-8-sig", newline="") as handle:
        result = [{k.strip(): v.strip() for k, v in row.items()}
                  for row in csv.DictReader(handle)]
    if result and any("PDH-CSV" in key for key in result[0]):
        keys = list(result[0])
        memory_key = next(k for k in keys if "Working Set - Private" in k)
        return [{"timestamp_local": r[keys[0]], "private_working_set_bytes": r[memory_key]} for r in result]
    return result


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    for font_path in (OUTPUT / "fonts").glob("*.ttf"):
        font_manager.fontManager.addfont(str(font_path))
    fig, axes = plt.subplots(2, 1, figsize=(12, 11), sharex=True)
    for ax in axes:
        apply_dashboard_style(fig, ax)
    summaries = []
    series_data = []
    references = []
    for (model, label), marker in zip(MODELS, ("o", "s", "^")):
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
        observed, = axes[0].plot(rx, smooth_memory(rx, rv), linewidth=2,
                     label=f"{label}: {rv[0]:,.0f} → {rv[-1]:,.0f} [MiB]")
        successful = [r for r in data["results"] if r.get("ok")]
        prompt = sum(r["prompt_tokens"] for r in successful) / len(successful)
        request_rate = len(successful) / sum(r["latency_s"] for r in successful)
        dims = KV_DIMENSIONS[label]
        layers, heads, dim = dims
        rate = request_rate * prompt * 2 * layers * heads * dim * 2 / 2**20
        references.append((label, observed.get_color(), rx[0], rv[0], rate, prompt, dims))
        axes[1].plot(gx, smooth_memory(gx, gv), linewidth=2,
                     label=f"{label}: {gv[0]:,.0f} → {gv[-1]:,.0f} [MiB]")
        summaries.append(dict(model=label, completed_requests=data["completed_samples"],
                              ram_start_mib=rv[0], ram_end_mib=rv[-1],
                              ram_growth_mib=rv[-1]-rv[0],
                              vram_start_mib=gv[0], vram_end_mib=gv[-1],
                              vram_growth_mib=gv[-1]-gv[0],
                              vram_min_mib=min(gv), vram_max_mib=max(gv)))
    duration = max(max(s[2][-1], s[4][-1]) for s in series_data) * 1.02
    ram_ceiling = max(max(s[3]) for s in series_data) * 1.15
    add_kv_references(axes[0], references, ram_ceiling)
    counts = ", ".join(f"{s['model']}: {s['completed_requests']}" for s in summaries)
    axes[0].set(ylabel="Private Working Set [MiB]", ylim=(0, ram_ceiling))
    axes[1].set(ylabel="Device Memory Used [MiB]", ylim=(0, 11000),
                xlabel="Time Since First GPU Sample [s]")
    set_dashboard_title(axes[0], "Process RAM vs. Time")
    set_dashboard_title(axes[1], "GPU Memory vs. Time")
    axes[1].axhline(10240, color=MUTED, linewidth=1, linestyle="--")
    axes[1].text(duration * .99, 10370, "10 GiB capacity", ha="right", color=MUTED)
    for ax in axes:
        ax.set_xlim(0, duration)
        style_reference_typography(ax)
        style_dashboard_legend(ax, loc="lower right" if ax is axes[0] else "lower left")
    title = fig.suptitle("RTX 3080: Memory During Sequential Prompt Processing", fontsize=12,
                 fontfamily="DejaVu Sans", fontweight="medium", color=HEADING,
                 x=0, ha="left", y=0.98)
    fig.text(0.09, 0.025,
             reference_caption(references), fontsize=9, color=MUTED, linespacing=1.5)
    fig.tight_layout(rect=(0, 0.13, 1, 0.95), h_pad=2.2)
    # Match the reference titles' alignment with the left edge of the axes.
    title.set_x(axes[0].get_position().x0)
    fig.savefig(OUTPUT / "memory-timelines.png", dpi=170, facecolor=BACKGROUND)
    plt.close(fig)
    for kind in ("ram", "vram"):
        fig, ax = plt.subplots(figsize=(12, 8) if kind == "ram" else (10, 5.2))
        apply_dashboard_style(fig, ax)
        for label, marker, rx, rv, gx, gv in series_data:
            x, y = (rx, rv) if kind == "ram" else (gx, gv)
            ax.plot(x, smooth_memory(x, y), linewidth=2, label=label)
        ax.set(xlabel="Time Since First GPU Sample [s]", xlim=(0, duration),
               ylabel="Private Working Set [MiB]" if kind == "ram" else "Device Memory Used [MiB]",
               ylim=(0, ram_ceiling if kind == "ram" else 11000))
        set_dashboard_title(ax, "RTX 3080: " + ("Process RAM" if kind == "ram" else "GPU Memory") + " vs. Time")
        style_reference_typography(ax)
        if kind == "vram":
            ax.axhline(10240, color=MUTED, linewidth=1, linestyle="--")
            ax.text(duration * .99, 10370, "10 GiB capacity", ha="right", color=MUTED)
        else:
            add_kv_references(ax, references, ram_ceiling)
        style_dashboard_legend(ax, loc="lower right" if kind == "ram" else "lower left", ncol=1 if kind == "ram" else 3)
        fig.text(0.1, 0.035,
                 (reference_caption(references) if kind == "ram" else "Device-wide GPU memory allocation. Measured curves: centered 9 s mean."),
                 color=MUTED, fontsize=9, linespacing=1.5)
        fig.tight_layout(rect=(0, 0.18 if kind == "ram" else 0.12, 1, 1))
        fig.savefig(OUTPUT / f"{kind}-vs-time-localai.png", dpi=180, facecolor=BACKGROUND)
        fig.savefig(OUTPUT / f"{kind}-vs-time-localai.svg", facecolor=BACKGROUND)
        if kind == "ram":
            for name in ("ram-vs-time-bracketed-units.png", "ram-vs-time-reference-updated.png"):
                fig.savefig(OUTPUT / name, dpi=180, facecolor=BACKGROUND)
        plt.close(fig)
    with (OUTPUT / "memory-summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summaries[0]))
        writer.writeheader()
        writer.writerows(summaries)
    print(json.dumps(summaries, indent=2))
    print(OUTPUT / "memory-timelines.png")


if __name__ == "__main__":
    main()
