"""Plot device VRAM and estimated Q8_0 weights + configured-context FP16 KV cache.

Estimates assume full GPU offload, one sequence, and FP16 keys/values.
Published GGUF file sizes proxy weight bytes (including file metadata).
They exclude compute buffers, allocator overhead, and other GPU applications.
"""

import csv
import json
from datetime import datetime
from pathlib import Path
from statistics import mean

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from plot_localai_style import (
    BACKGROUND, MUTED, apply_dashboard_style, set_dashboard_title,
    style_dashboard_legend,
)

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "results/nvidia-rtx-3080/model-speed-bench-results"
OUTPUT = ROOT / "benchmark_plots/nvidia-rtx-3080/model-vram-timelines"
CONTEXT_TOKENS = 8172
IDLE_MIB = 1.4 * 10**9 / 2**20  # 1.4 decimal GB; fixed user-supplied baseline.
# Sizes retrieved from Hugging Face model APIs on 2026-10-26.
MODELS = {
    "Qwen3-4B-Q8_0": dict(label="Qwen3 4B", weight_bytes=4280404704,
        layers=36, kv_heads=8, head_dim=128, repo="Qwen/Qwen3-4B-GGUF",
        config_repo="Qwen/Qwen3-4B"),
    "Phi-3.5-mini-instruct-Q8_0": dict(label="Phi 3.5 mini", weight_bytes=4061222688,
        layers=32, kv_heads=32, head_dim=96, repo="bartowski/Phi-3.5-mini-instruct-GGUF",
        config_repo="microsoft/Phi-3.5-mini-instruct"),
    "Llama-3.2-3B-Instruct-Q8_0": dict(label="Llama 3.2 3B", weight_bytes=3421899296,
        layers=28, kv_heads=8, head_dim=128, repo="bartowski/Llama-3.2-3B-Instruct-GGUF",
        config_repo="unsloth/Llama-3.2-3B-Instruct"),
}


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    for font in (ROOT / "benchmark_plots/nvidia-rtx-3080/model-speed-memory/fonts").glob("*.ttf"):
        font_manager.fontManager.addfont(str(font))
    summaries = []
    for model, spec in MODELS.items():
        stem = f"{model}_throughput_1k_low_entropy_osl1"
        data = json.loads((SOURCE / f"{stem}.json").read_text())
        results = [r for r in data["results"] if r.get("ok")]
        assert len(results) == data["completed_samples"]
        # User-configured llama-server maximum context window.
        context = CONTEXT_TOKENS
        assert all(r.get("turns", 1) == 1 for r in results)
        kv_bytes_per_token = 2 * 2 * spec["layers"] * spec["kv_heads"] * spec["head_dim"]
        weights = spec["weight_bytes"] / 2**20
        kv = context * kv_bytes_per_token / 2**20
        with (SOURCE / f"{stem}.gpu.csv").open(newline="", encoding="utf-8-sig") as f:
            gpu = list(csv.DictReader(f, skipinitialspace=True))
        times = [datetime.strptime(r["timestamp"], "%Y/%m/%d %H:%M:%S.%f") for r in gpu]
        seconds = [(t - times[0]).total_seconds() for t in times]
        used = [float(r["memory_used_mib"]) for r in gpu]
        total = float(gpu[0]["memory_total_mib"])
        assert all(float(r["memory_total_mib"]) == total for r in gpu)
        fig, ax = plt.subplots(figsize=(11, 6.4))
        apply_dashboard_style(fig, ax)
        ax.plot(seconds, used, linewidth=1.8, label="Measured device VRAM")
        accounted = IDLE_MIB + weights + kv
        measured_mean = mean(used)
        levels = [0, IDLE_MIB, IDLE_MIB + weights, accounted]
        colors = ["#7b8c98", "#82684d", "#c17b50"]
        labels = [f"Idle system: 1.4 GB ({IDLE_MIB:,.0f} MiB)",
                  f"Q8_0 weights: +{weights:,.0f} MiB",
                  f"8,172-token KV: +{kv:,.0f} MiB"]
        for low, high, color, label, fraction in zip(
                levels, levels[1:], colors, labels, (.08, .30, .55)):
            ax.axhspan(low, high, color=color, alpha=.09, linewidth=0)
            ax.axhline(high, color=color, linewidth=1.6, linestyle="--")
            x = seconds[-1] * fraction
            ax.annotate("", xy=(x, high), xytext=(x, low),
                        arrowprops=dict(arrowstyle="<->", color=color, lw=1.6))
            ax.text(x + seconds[-1]*.015, (low+high)/2, label,
                    va="center", fontsize=9, color=color,
                    bbox=dict(facecolor="white", alpha=.85, edgecolor="none", pad=3))
        x = seconds[-1] * .78
        ax.annotate("", xy=(x, measured_mean), xytext=(x, accounted),
                    arrowprops=dict(arrowstyle="<->", color="#16705c", lw=1.6))
        ax.text(x + seconds[-1]*.015, (accounted+measured_mean)/2,
                f"Est. runtime buffers: {measured_mean-accounted:,.0f} MiB",
                va="center", fontsize=9, color="#16705c",
                bbox=dict(facecolor="white", alpha=.85, edgecolor="none", pad=3))
        ax.axhline(total, color=MUTED, linewidth=1, linestyle="--")
        ax.text(seconds[-1]*.99, total+110, "10 GiB capacity", ha="right", color=MUTED, fontsize=9)
        ax.set(xlabel="Time Since First GPU Sample [s]", ylabel="VRAM Used [MiB]",
               xlim=(0, seconds[-1]), ylim=(0, total*1.08))
        percent = ax.secondary_yaxis("right", functions=(lambda v: v/total*100, lambda p: p*total/100))
        percent.set_ylabel("VRAM Capacity Used [%]", color=MUTED,
                           fontfamily="DejaVu Sans", fontsize=10)
        percent.tick_params(colors=MUTED, labelsize=9, length=3, width=.7, pad=6)
        set_dashboard_title(ax, f"{spec['label']} · Q8_0 · VRAM vs. Time · RTX 3080")
        # The reference memory/3080 SVGs use DejaVu Sans for titles and axes.
        # Set it explicitly so installed dashboard fonts cannot change the match.
        ax.set_title(ax.get_title(loc="left"), loc="left", fontfamily="DejaVu Sans")
        for label in (ax.xaxis.label, ax.yaxis.label):
            label.set_fontfamily("DejaVu Sans")
            label.set_fontsize(10)
        for label in (*ax.get_xticklabels(), *ax.get_yticklabels(), *percent.get_yticklabels()):
            label.set_fontfamily("DejaVu Sans")
            label.set_fontsize(9)
        style_dashboard_legend(ax, loc="upper left", fontsize=9)
        fig.text(.09, .035,
            f"Accounted VRAM: {accounted:,.0f} MiB = idle system + Q8_0 weights + {context:,}-token FP16 KV cache.\n"
            "Fixed idle offset: 1.4 GB (decimal). Weight estimate uses published GGUF size; full GPU offload, one sequence.\n"
            "Runtime buffers estimated from the remaining gap; also includes other runtime overhead and estimation error.",
            fontsize=9, color=MUTED, linespacing=1.6)
        fig.tight_layout(rect=(0, .16, 1, 1))
        for ext in ("png", "svg"):
            fig.savefig(OUTPUT / f"{model}-vram-vs-time.{ext}", dpi=180, facecolor=BACKGROUND)
        plt.close(fig)
        summaries.append(dict(model=model, successful_requests=len(results),
            configured_context_tokens=context, estimated_weight_mib=weights,
            fp16_kv_bytes_per_token=kv_bytes_per_token, estimated_configured_context_kv_mib=kv,
            estimated_weights_plus_context_mib=weights+kv,
            idle_baseline_mib=IDLE_MIB, estimated_accounted_vram_mib=accounted,
            measured_mean_vram_mib=measured_mean, estimated_runtime_buffers_and_overhead_mib=measured_mean-accounted,
            measured_vram_min_mib=min(used), measured_vram_max_mib=max(used),
            duration_seconds=seconds[-1],
            weights_source=f"https://huggingface.co/{spec['repo']}/blob/main/{model}.gguf",
            config_source=f"https://huggingface.co/{spec['config_repo']}/blob/main/config.json"))
    with (OUTPUT / "memory-estimates.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(summaries[0]))
        writer.writeheader()
        writer.writerows(summaries)
    (OUTPUT / "methodology.json").write_text(json.dumps(dict(
        metadata_retrieved="2026-10-26", models=MODELS,
        configured_context_tokens=CONTEXT_TOKENS,
        idle_baseline_mib=IDLE_MIB, idle_baseline_units="1.4 decimal GB converted to MiB",
        context_source="User-configured llama-server maximum context window",
        kv_formula="context_tokens * layers * kv_heads * head_dim * 2 (K,V) * 2 (FP16 bytes)",
        assumptions="Full GPU offload; one sequence; FP16 KV; GGUF file size proxies weights; excludes compute buffers and overhead",
        utilization="memory_used_mib / memory_total_mib; memory_utilization_percent is memory-controller activity, not capacity used",
        runs=summaries), indent=2), encoding="utf-8")
    print(json.dumps(summaries, indent=2))


if __name__ == "__main__":
    main()
