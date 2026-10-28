"""Illustrate generation latency against an approximate English reading pace."""

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np

from plot_localai_style import apply_dashboard_style, TEXT as TEXT_COLOR, MUTED as MUTED_TEXT_COLOR, SERIES_COLORS, HEADING


def main():
    for font in (Path(__file__).parent / "benchmark_plots" / "fonts").glob("*-static.ttf"):
        font_manager.fontManager.addfont(str(font))
    plt.rcParams.update({"font.family": "DM Sans", "text.color": TEXT_COLOR, "svg.fonttype": "path"})
    rates = np.array([2, 5, 10, 20])
    tokens = 500
    reading_rate = 238 / 60 / 0.75
    fig, ax = plt.subplots(figsize=(12, 8.2))
    apply_dashboard_style(fig, ax)
    ax.grid(False, axis="y")
    fig.subplots_adjust(left=0.16, right=0.78, bottom=0.33, top=0.65)
    fig.text(0.07, 0.93, "Why tokens per second matter", fontsize=25, weight="medium", fontfamily="Space Grotesk", color=HEADING)
    fig.text(0.07, 0.84, "~5.3 tokens/sec", fontsize=32, weight="medium", fontfamily="Space Grotesk", color=SERIES_COLORS[0])
    fig.text(0.07, 0.785, "Estimated average adult reading speed · English nonfiction", fontsize=13, color=HEADING)
    fig.text(0.07, 0.738, "238 words/min ÷ 60 ÷ ~0.75 words/token = ~5.3 tokens/sec", fontsize=11, color=MUTED_TEXT_COLOR)
    positions = np.arange(len(rates))
    colors = [SERIES_COLORS[1] if r < reading_rate else SERIES_COLORS[0] for r in rates]
    ax.barh(positions, rates, height=0.52, color=colors, zorder=3)
    ax.set_yticks(positions, [f"{r} tokens/sec" for r in rates], fontsize=12)
    ax.invert_yaxis()
    ax.set_xlim(0, 22)
    ax.set_xticks([0, 5, 10, 15, 20])
    ax.set_xlabel("Generation speed (tokens/sec)", labelpad=12, fontsize=12)
    ax.grid(axis="x", zorder=0)
    ax.set_axisbelow(True)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(axis="y", length=0, pad=12)
    ax.axvline(reading_rate, color=HEADING, linestyle=(0, (4, 3)), linewidth=1.8, zorder=4)
    ax.text(reading_rate, 1.06, "Average reading\n~5.3 tokens/sec", transform=ax.get_xaxis_transform(),
            ha="center", color=HEADING, fontsize=11, linespacing=1.4)
    ax.text(1.06, 1.06, "Time to generate\n500 tokens (~375 words)", transform=ax.transAxes,
            color=MUTED_TEXT_COLOR, fontsize=10, linespacing=1.4)
    for y, rate in zip(positions, rates):

        ax.text(1.06, y, f"{tokens / rate:.0f} sec", transform=ax.get_yaxis_transform(),
                va="center", fontsize=12, color=TEXT_COLOR)
    fig.text(0.07, 0.225, "Below ~5 tokens/sec, readers can outpace the stream.", fontsize=16, weight="medium", fontfamily="Space Grotesk", color=HEADING)
    fig.text(0.07, 0.18, "Above it, text arrives faster than typical reading; complete answers are ready sooner.",
             fontsize=11, color=MUTED_TEXT_COLOR)
    fig.text(0.07, 0.12,
             "Reading reference: 238 words/min (adult English nonfiction; Brysbaert, 2019). At ~0.75 words/token: ~5.3 tokens/sec.\n"
             "Sources: doi.org/10.1016/j.jml.2019.104047  |  OpenAI, Understanding and counting tokens (0.75 words/token).\n"
             "Illustrative rates, not benchmark results. Constant output speed; excludes prompt processing and time to first token.\n"
             "Reading pace and tokenization vary by person, text, language, and tokenizer.",
             fontsize=8.5, color=MUTED_TEXT_COLOR, va="top", linespacing=1.25)
    output = Path(__file__).parent / "benchmark_plots"
    output.mkdir(exist_ok=True)
    for extension in ("png", "svg"):
        fig.savefig(output / f"why_tokens_per_second.{extension}", dpi=200, facecolor=fig.get_facecolor())
    plt.close(fig)


if __name__ == "__main__":
    main()
