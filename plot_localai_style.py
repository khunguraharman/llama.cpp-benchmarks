"""Light chart styling based on LocalAI's src/styles/global.css."""

from pathlib import Path
from typing import Any

import matplotlib as mpl
from matplotlib import font_manager
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.legend import Legend


BACKGROUND = "#ffffff"
TEXT = "#283531"
HEADING = "#1c3028"
MUTED = "#68756f"
BORDER = "#e1e8e3"
# Green and terracotta are the CSS's platform accents; the additional shades
# keep the remaining series distinct on white, alongside different markers.
SERIES_COLORS = ("#16705c", "#d89477", "#82684d", "#52635a", "#2c916d", "#7a8b80")


def _font(preferred: str) -> str:
    available = {font.name for font in font_manager.fontManager.ttflist}
    return preferred if preferred in available else "DejaVu Sans"


def apply_dashboard_style(fig: Figure, ax: Axes) -> None:
    """Apply the site's light surfaces, muted labels, and fine sage rules."""
    body_font = _font("DM Sans")
    mpl.rcParams.update({"font.family": body_font, "text.color": TEXT})
    fig.patch.set_facecolor(BACKGROUND)
    ax.set_facecolor(BACKGROUND)
    ax.set_prop_cycle(color=SERIES_COLORS)
    ax.set_axisbelow(True)
    ax.grid(True, which="major", color=BORDER, linewidth=0.7)
    ax.grid(False, which="minor")
    ax.tick_params(colors=MUTED, labelsize=9, length=3, width=0.7, pad=6)
    for label in (ax.xaxis.label, ax.yaxis.label):
        label.set_color(MUTED)
        label.set_fontfamily(body_font)
        label.set_fontsize(10)
    for label in (*ax.get_xticklabels(), *ax.get_yticklabels()):
        label.set_fontfamily(body_font)
    for name, spine in ax.spines.items():
        spine.set_color(BORDER)
        spine.set_linewidth(0.8)
        spine.set_visible(name in ("left", "bottom"))


def set_dashboard_title(ax: Axes, title: str) -> None:
    ax.set_title(
        title, loc="left", pad=20, fontsize=12, fontweight="medium",
        fontfamily=_font("Space Grotesk"), color=HEADING,
    )


def style_dashboard_legend(ax: Axes, **kwargs: Any) -> Legend:
    kwargs.setdefault("frameon", False)
    kwargs.setdefault("labelspacing", 0.8)
    kwargs.setdefault("handlelength", 2.4)
    legend = ax.legend(**kwargs)
    for label in legend.get_texts():
        label.set_color(TEXT)
        label.set_fontfamily(_font("DM Sans"))
    legend.get_title().set_color(MUTED)
    return legend


def save_dashboard_figure(fig: Figure, ax: Axes, output_path: Path) -> None:
    fig.tight_layout(pad=2)
    fig.savefig(output_path, dpi=200, facecolor=BACKGROUND, edgecolor="none")
