"""Plot per-sample prompt throughput and throughput divided by input tokens."""

import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.ticker import MaxNLocator

from plot_localai_style import BACKGROUND, MUTED, apply_dashboard_style, set_dashboard_title, style_dashboard_legend


ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "results/nvidia-rtx-3080/model-speed-bench-results"
OUTPUT = ROOT / "benchmark_plots/nvidia-rtx-3080/model-prompt-speed"
MODELS = [
    ("Qwen3-4B-Q8_0", "Qwen3 4B"),
    ("Phi-3.5-mini-instruct-Q8_0", "Phi 3.5 mini"),
    ("Llama-3.2-3B-Instruct-Q8_0", "Llama 3.2 3B"),
]


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    for font in (ROOT / "benchmark_plots/nvidia-rtx-3080/model-speed-memory/fonts").glob("*.ttf"):
        font_manager.fontManager.addfont(str(font))
    records = []
    for model, label in MODELS:
        path = SOURCE / f"{model}_throughput_1k_low_entropy_osl1.json"
        data = json.loads(path.read_text())
        for sample, result in enumerate(data["results"], start=1):
            records.append(dict(model=label, sample=sample,
                                input_tokens=result["prompt_tokens"],
                                prompt_tokens_per_second=result["prompt_per_second"],
                                normalized_speed_per_second=result["prompt_per_second"] / result["prompt_tokens"]))
    plots = [
        ("prompt_tokens_per_second", "Prompt processing speed by sample",
         "Prompt processing speed (tokens/s)", "prompt-speed.png",
         "Reported prompt_per_second: throughput for tokens actually processed."),
        ("normalized_speed_per_second", "Prompt processing speed / input-token count",
         "Throughput / full input tokens (1/s)", "prompt-speed-normalized.png",
         "For each sample: prompt_per_second / prompt_tokens (including reused input tokens)."),
    ]
    for field, title, ylabel, filename, note in plots:
        fig, ax = plt.subplots(figsize=(10, 5.1))
        apply_dashboard_style(fig, ax)
        for (_, label), marker in zip(MODELS, ("o", "s", "^")):
            series = [r for r in records if r["model"] == label]
            ax.plot([r["sample"] for r in series], [r[field] for r in series],
                    marker=marker, markevery=max(1,len(series)//15), linewidth=1.2, markersize=4, label=label, alpha=.85)
        n = max(r['sample'] for r in records)
        ax.set(xlabel="Sample # (JSON result order)", ylabel=ylabel, xlim=(1, n))
        ax.xaxis.set_major_locator(MaxNLocator(nbins=7, integer=True))
        set_dashboard_title(ax, title + " · RTX 3080")
        ax.set_ylim(bottom=0, top=max(r[field] for r in records) * 1.16)
        style_dashboard_legend(ax, loc="lower left", ncol=3)
        fig.text(0.1, 0.035, note + f"\nUp to {n} samples per model · Q8_0 · concurrency 1 · low entropy · one output token", fontsize=10, color=MUTED)
        fig.tight_layout(rect=(0, 0.12, 1, 1))
        fig.savefig(OUTPUT / filename, dpi=180, facecolor=BACKGROUND)
        fig.savefig((OUTPUT / filename).with_suffix('.svg'), facecolor=BACKGROUND)
        plt.close(fig)
        print(OUTPUT / filename)
    with (OUTPUT / "prompt-speed-by-sample.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    for _, label in MODELS:
        series = [r for r in records if r["model"] == label]
        print(label, "mean tokens/s", round(sum(r["prompt_tokens_per_second"] for r in series)/len(series), 2),
              "mean normalized", round(sum(r["normalized_speed_per_second"] for r in series)/len(series), 3))


if __name__ == "__main__":
    main()
