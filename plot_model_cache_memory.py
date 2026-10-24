"""Plot observed RAM and VRAM during the model-speed runs."""

import csv
import json
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
    fig, axes = plt.subplots(2, 1, figsize=(11, 8), sharex=True)
    for ax in axes:
        apply_dashboard_style(fig, ax)
    summaries = []
    series_data = []
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
        axes[0].plot(rx, rv, marker=marker, markevery=max(1, len(rx)//15), markersize=4, linewidth=2,
                     label=f"{label}: {rv[0]:,.0f} → {rv[-1]:,.0f} MiB")
        axes[1].plot(gx, gv, marker=marker, markevery=20, markersize=4, linewidth=2,
                     label=f"{label}: {gv[0]:,.0f} → {gv[-1]:,.0f} MiB")
        summaries.append(dict(model=label, completed_requests=data["completed_samples"],
                              ram_start_mib=rv[0], ram_end_mib=rv[-1],
                              ram_growth_mib=rv[-1]-rv[0],
                              vram_start_mib=gv[0], vram_end_mib=gv[-1],
                              vram_growth_mib=gv[-1]-gv[0],
                              vram_min_mib=min(gv), vram_max_mib=max(gv)))
    duration = max(max(s[2][-1], s[4][-1]) for s in series_data) * 1.02
    ram_ceiling = max(max(s[3]) for s in series_data) * 1.15
    counts = ", ".join(f"{s['model']}: {s['completed_requests']}" for s in summaries)
    axes[0].set(ylabel="Private working set (MiB)", ylim=(0, ram_ceiling))
    axes[1].set(ylabel="Device memory used (MiB)", ylim=(0, 11000),
                xlabel="Seconds since first GPU sample in each run")
    set_dashboard_title(axes[0], "Process RAM vs time")
    set_dashboard_title(axes[1], "GPU memory: nearly constant")
    axes[1].axhline(10240, color=MUTED, linewidth=1, linestyle="--")
    axes[1].text(duration * .99, 10370, "10 GiB capacity", ha="right", color=MUTED)
    for ax in axes:
        ax.set_xlim(0, duration)
        style_dashboard_legend(ax, loc="lower right" if ax is axes[0] else "lower left")
    fig.suptitle("Memory during sequential prompt processing · RTX 3080", fontsize=16,
                 fontfamily="Space Grotesk", fontweight="medium", color=HEADING, y=0.98)
    fig.text(0.09, 0.025,
             "Completed requests — " + counts + "\n"
             "RAM is process-specific; GPU memory is device-wide. Concurrency 1 · one output token · Q8_0.",
             fontsize=10, color=MUTED)
    fig.tight_layout(rect=(0, 0.09, 1, 0.95), h_pad=2.2)
    fig.savefig(OUTPUT / "memory-timelines.png", dpi=170, facecolor=BACKGROUND)
    plt.close(fig)
    for kind in ("ram", "vram"):
        fig, ax = plt.subplots(figsize=(10, 5.2))
        apply_dashboard_style(fig, ax)
        for label, marker, rx, rv, gx, gv in series_data:
            x, y = (rx, rv) if kind == "ram" else (gx, gv)
            ax.plot(x, y, marker=marker, markevery=max(1, len(x)//15),
                    markersize=4, linewidth=2, label=label)
        ax.set(xlabel="Seconds since first GPU sample in each run", xlim=(0, duration),
               ylabel="Private working set (MiB)" if kind == "ram" else "Device memory used (MiB)",
               ylim=(0, ram_ceiling if kind == "ram" else 11000))
        set_dashboard_title(ax, ("Process RAM" if kind == "ram" else "GPU memory") + " vs time · RTX 3080")
        if kind == "vram":
            ax.axhline(10240, color=MUTED, linewidth=1, linestyle="--")
            ax.text(duration * .99, 10370, "10 GiB capacity", ha="right", color=MUTED)
        style_dashboard_legend(ax, loc="lower right" if kind == "ram" else "lower left", ncol=3)
        fig.text(0.1, 0.035, "Completed requests — " + counts + "\n" +
                 ("Process-specific private working set." if kind == "ram" else "Device-wide GPU memory allocation."),
                 color=MUTED, fontsize=10)
        fig.tight_layout(rect=(0, 0.12, 1, 1))
        fig.savefig(OUTPUT / f"{kind}-vs-time-localai.png", dpi=180, facecolor=BACKGROUND)
        fig.savefig(OUTPUT / f"{kind}-vs-time-localai.svg", facecolor=BACKGROUND)
        plt.close(fig)
    with (OUTPUT / "memory-summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summaries[0]))
        writer.writeheader()
        writer.writerows(summaries)
    print(json.dumps(summaries, indent=2))
    print(OUTPUT / "memory-timelines.png")


if __name__ == "__main__":
    main()
