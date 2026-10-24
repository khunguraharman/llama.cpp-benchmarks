"""Plot tokens whose evaluation was avoided, verified against server logs."""

import csv
import json
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.ticker import MaxNLocator

from plot_localai_style import (
    BACKGROUND, MUTED, SERIES_COLORS, apply_dashboard_style,
    set_dashboard_title, style_dashboard_legend,
)
from plot_model_prompt_speed import MODELS, ROOT, SOURCE


OUTPUT = ROOT / "benchmark_plots/nvidia-rtx-3080/model-cache-hits"


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    for font in (ROOT / "benchmark_plots/nvidia-rtx-3080/model-speed-memory/fonts").glob("*.ttf"):
        font_manager.fontManager.addfont(str(font))
    records = []
    for model, label in MODELS:
        results = json.loads((SOURCE / f"{model}_throughput_1k_low_entropy_osl1.json").read_text())["results"]
        log = (SOURCE / f"server_{model}.log").read_text()
        evaluations = [int(n) for n in re.findall(r"prompt eval time\s*=\s*[\d.]+ ms\s*/\s*(\d+) tokens", log)]
        if len(evaluations) != len(results):
            raise ValueError(f"{model}: log/JSON sample count mismatch")
        cumulative_reused = 0
        for sample, (result, evaluated) in enumerate(zip(results, evaluations), 1):
            implied = result["prompt_ms"] * result["prompt_per_second"] / 1000
            if abs(implied - evaluated) > 0.01:
                raise ValueError(f"{model} sample {sample}: log/JSON evaluated tokens mismatch")
            total = result["prompt_tokens"]
            cached = total - evaluated
            if not 0 <= cached <= total:
                raise ValueError(f"{model} sample {sample}: invalid cache count")
            cumulative_reused += cached
            records.append(dict(model=label, sample=sample, input_tokens=total,
                                evaluated_tokens=evaluated, reused_tokens=cached,
                                cumulative_reused_tokens=cumulative_reused,
                                evaluation_avoided_percent=100 * cached / total))
    fig, axes = plt.subplots(2, 1, figsize=(11, 8.5), sharex=True)
    for ax in axes:
        apply_dashboard_style(fig, ax)
    for (_, label), color, marker in zip(MODELS, SERIES_COLORS, ("o", "s", "^")):
        series = [r for r in records if r["model"] == label]
        for ax, field in zip(axes, ("reused_tokens", "evaluation_avoided_percent")):
            ax.scatter([r["sample"] for r in series], [r[field] for r in series],
                       color=color, marker=marker, s=16, alpha=.65, linewidths=0, label=label)
        total = sum(r['input_tokens'] for r in series)
        cached = sum(r['reused_tokens'] for r in series)
        print(f"{label}: {cached:,}/{total:,} tokens reused ({100*cached/total:.2f}%), {len(series)} samples")
    set_dashboard_title(axes[0], "Tokens reused from cache by sample · RTX 3080")
    set_dashboard_title(axes[1], "Reduction in evaluated input tokens")
    axes[0].set_ylabel("Input tokens − evaluated tokens")
    axes[0].set_ylim(0, max(r['reused_tokens'] for r in records) * 1.15)
    axes[1].set(ylabel="Input-token evaluation avoided (%)", ylim=(0, 105),
                xlabel="Sample # (JSON result order)")
    n = max(r['sample'] for r in records)
    axes[1].set_xlim(0, n + 2)
    axes[1].xaxis.set_major_locator(MaxNLocator(nbins=7, integer=True))
    style_dashboard_legend(axes[0], loc="upper left", ncol=3, markerscale=1.5)
    fig.text(.1, .025,
             "Evaluated-token counts verified against server logs for every sample.\n"
             "Avoided evaluation (%) = 100 × (input tokens − evaluated tokens) / input tokens.",
             color=MUTED, fontsize=10)
    fig.tight_layout(rect=(0, .08, 1, 1), h_pad=2.2)
    for extension in ("png", "svg"):
        fig.savefig(OUTPUT / f"cached-tokens-by-sample.{extension}", dpi=180, facecolor=BACKGROUND)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(10, 5.6))
    apply_dashboard_style(fig, ax)
    for (_, label), color, marker in zip(MODELS, SERIES_COLORS, ("o", "s", "^")):
        series = [r for r in records if r["model"] == label]
        x = [0] + [r['sample'] for r in series]
        y = [0] + [r['cumulative_reused_tokens'] for r in series]
        ax.plot(x, y, color=color, linewidth=2,
                label=f"{label}: {y[-1]:,} tokens")
    n = max(r['sample'] for r in records)
    ax.set(xlabel="Requests handled by the server", ylabel="Cumulative input-token evaluations avoided",
           xlim=(0, n), ylim=(0, max(r['cumulative_reused_tokens'] for r in records)*1.08))
    ax.xaxis.set_major_locator(MaxNLocator(nbins=7, integer=True))
    ax.yaxis.set_major_formatter(matplotlib.ticker.StrMethodFormatter('{x:,.0f}'))
    set_dashboard_title(ax, "Cumulative tokens reused from cache · RTX 3080")
    style_dashboard_legend(ax, loc='upper left')
    fig.text(.1, .035,
             "Running sum of (full input tokens − evaluated tokens), verified against server logs.\n"
             "Counts avoided evaluations across requests; repeated reuse of a token is counted each time.",
             fontsize=10, color=MUTED)
    fig.tight_layout(rect=(0, .12, 1, 1))
    for extension in ('png', 'svg'):
        fig.savefig(OUTPUT / f'cumulative-cached-tokens.{extension}', dpi=180, facecolor=BACKGROUND)
    plt.close(fig)
    with (OUTPUT / "cached-tokens-by-sample.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    print(OUTPUT / "cached-tokens-by-sample.png")


if __name__ == "__main__":
    main()
