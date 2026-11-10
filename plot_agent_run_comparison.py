"""Compare observed agent-run totals from the cached and no-cache server logs."""

import csv

from matplotlib import font_manager
from matplotlib.patches import Patch
from matplotlib.ticker import StrMethodFormatter

from plot_agent_run import LOG, OUTPUT, ROOT, parse_log, plt
from plot_localai_style import (
    BACKGROUND, HEADING, MUTED, SERIES_COLORS, apply_dashboard_style,
    set_dashboard_title,
)


def split_conversations(requests):
    """Split these two-conversation runs at the first prompt-context reset.

    Later short prompts belong to conversation 2 (including its no-cache
    retry after context exhaustion), so they must not create new series.
    Server logs do not contain explicit conversation IDs.
    """
    ordered = sorted(requests, key=lambda request: request.start)
    boundary = next((i for i in range(1, len(ordered))
                     if ordered[i].prompt_tokens < ordered[i - 1].prompt_tokens), None)
    if boundary is None:
        raise ValueError("Cannot identify conversation 2: no prompt-context reset")
    return ordered[:boundary], ordered[boundary:]


def main():
    records = []
    for label, path in (
        ("Cache enabled", LOG),
        ("No cache", LOG.with_name(f"{LOG.stem}-no-cache.log")),
    ):
        requests, _ = parse_log(path)
        for conversation, group in enumerate(split_conversations(requests), start=1):
            records.append(dict(
                run=label,
                conversation=conversation,
                requests=len(group),
                tool_calls=None,
                prompt_tokens_processed=sum(r.prefill[0] for r in group),
                prompt_processing_seconds=sum(r.prefill_seconds for r in group),
                output_tokens=sum(r.generation[0] for r in group),
                elapsed_seconds=max(r.end for r in group) - min(r.start for r in group),
            ))

    for font in (ROOT / "benchmark_plots/fonts").glob("*-static.ttf"):
        font_manager.fontManager.addfont(str(font))
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    metrics = (
        ("prompt_processing_seconds", "Prompt processing time", "Seconds"),
        ("elapsed_seconds", "Time to completion", "Seconds"),
        ("prompt_tokens_processed", "Total prompt tokens processed", "Tokens, excluding reused input"),
        ("output_tokens", "Total output tokens", "Generated tokens"),
    )
    for ax, (field, title, unit) in zip(axes.flat, metrics):
        apply_dashboard_style(fig, ax)
        set_dashboard_title(ax, title)
        all_values = [record[field] for record in records]
        for conversation, color in enumerate(SERIES_COLORS[:2], start=1):
            series = [record for record in records if record["conversation"] == conversation]
            values = [record[field] for record in series]
            positions = [x + (conversation - 1.5) * .36 for x in (0, 1)]
            bars = ax.bar(positions, values, width=.34, color=color)
            labels = [f"{v:,.2f} s" if field.endswith("_seconds") else f"{v:,}" for v in values]
            ax.bar_label(bars, labels=labels, padding=6, fontsize=10, color=HEADING)
        ax.set_xticks([0, 1], ["Cache enabled", "No cache"])
        ax.set_ylabel(unit)
        ax.set_ylim(0, max(all_values) * 1.25)
        ax.yaxis.set_major_formatter(StrMethodFormatter(
            "{x:,.1f}" if field == "prompt_processing_seconds" else "{x:,.0f}"
        ))

    fig.text(.065, .95, "Qwen3-4B-Q8_0 · agent run comparison", fontsize=23,
             fontfamily="Space Grotesk", fontweight="medium", color=HEADING)
    fig.text(.065, .905, "RTX 3080  /  per-conversation totals from the two server logs", fontsize=11, color=MUTED)
    fig.legend(handles=[Patch(color=color, label=f"Conversation {conversation}")
                        for conversation, color in enumerate(SERIES_COLORS[:2], start=1)],
               loc="lower center", ncol=2, frameon=False, bbox_to_anchor=(.5, .01))
    fig.subplots_adjust(left=.085, right=.97, top=.80, bottom=.12, wspace=.30, hspace=.55)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    for extension in ("png", "svg"):
        fig.savefig(OUTPUT / f"qwen3-4b-q8_0-run-comparison.{extension}",
                    dpi=180, facecolor=BACKGROUND)
    plt.close(fig)
    with (OUTPUT / "qwen3-4b-q8_0-run-comparison.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=records[0].keys())
        writer.writeheader()
        writer.writerows(records)
    for record in records:
        print(record)


if __name__ == "__main__":
    main()
