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


def main():
    records = []
    for label, path in (
        ("Cache enabled", LOG),
        ("No cache", LOG.with_name(f"{LOG.stem}-no-cache.log")),
    ):
        requests, _ = parse_log(path)
        records.append(dict(
            run=label,
            requests=len(requests),
            tool_calls=None,
            input_tokens=sum(r.prompt_tokens for r in requests),
            output_tokens=sum(r.generation[0] for r in requests),
            elapsed_seconds=max(r.end for r in requests) - min(r.start for r in requests),
        ))

    for font in (ROOT / "benchmark_plots/fonts").glob("*-static.ttf"):
        font_manager.fontManager.addfont(str(font))
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    metrics = (
        ("requests", "Requests to server", "Completed inference requests"),
        ("tool_calls", "Tool calls", ""),
        ("elapsed_seconds", "Time to completion", "Seconds"),
        ("input_tokens", "Total input tokens", "Tokens, including reused input"),
        ("output_tokens", "Total output tokens", "Generated tokens"),
    )
    for ax, (field, title, unit) in zip(axes.flat, metrics):
        apply_dashboard_style(fig, ax)
        set_dashboard_title(ax, title)
        if field == "tool_calls":
            ax.set_axis_off()
            ax.text(.5, .56, "Unavailable for both runs", ha="center", va="center",
                    transform=ax.transAxes, fontsize=14, color=HEADING)
            ax.text(.5, .35, "Server logs do not record tool calls", ha="center",
                    transform=ax.transAxes, fontsize=10, color=MUTED)
            continue
        values = [record[field] for record in records]
        bars = ax.bar([0, 1], values, width=.5, color=SERIES_COLORS[:2])
        ax.set_xticks([0, 1], [record["run"] for record in records])
        ax.set_ylabel(unit)
        ax.set_ylim(0, max(values) * 1.3)
        ax.yaxis.set_major_formatter(StrMethodFormatter("{x:,.0f}"))
        labels = [f"{v:,.2f} s" if field == "elapsed_seconds" else f"{v:,}" for v in values]
        ax.bar_label(bars, labels=labels, padding=6, fontsize=12, color=HEADING)
        change = 100 * (values[0] / values[1] - 1)
        ax.text(.5, .96, f"Cache enabled vs no cache: {change:+.1f}%",
                ha="center", va="top", transform=ax.transAxes, fontsize=9, color=MUTED)

    note = axes.flat[-1]
    note.set_axis_off()
    note.text(0, .92, "How to read these totals", fontsize=13, color=HEADING,
              transform=note.transAxes)
    note.text(0, .77,
              "Requests count completed model calls.\n"
              "Input sums full prompts across requests;\n"
              "repeated context is counted each time.\n\n"
              "Time spans first prompt to last response,\n"
              "including gaps between requests.\n"
              "Agent work after that is not recorded.\n\n"
              "The runs produced different request and\n"
              "output totals; this is an observed-run\n"
              "comparison, not an isolated cache speedup.",
              fontsize=9, color=MUTED, linespacing=1.2, va="top", transform=note.transAxes)
    fig.text(.065, .95, "Qwen3-4B-Q8_0 · agent run comparison", fontsize=23,
             fontfamily="Space Grotesk", fontweight="medium", color=HEADING)
    fig.text(.065, .905, "RTX 3080  /  totals from the two server logs", fontsize=11, color=MUTED)
    fig.legend(handles=[Patch(color=color, label=record["run"])
                        for color, record in zip(SERIES_COLORS, records)],
               loc="lower center", ncol=2, frameon=False, bbox_to_anchor=(.5, .01))
    fig.subplots_adjust(left=.065, right=.97, top=.80, bottom=.12, wspace=.35, hspace=.55)
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
