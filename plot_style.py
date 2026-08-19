"""Shared dark report styling for benchmark plots.

The charts in this project are saved as standalone PNGs, so this module makes
them feel like the charts inside a cohesive benchmark report rather than
Matplotlib defaults.  It deliberately contains no benchmark-specific data or
layout logic; every plotting script can opt into the same visual language.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib as mpl
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.legend import Legend
from matplotlib.patches import FancyBboxPatch


PAGE_BACKGROUND = "#06131F"
PANEL_BACKGROUND = "#0B1E2B"
PANEL_BORDER = "#385567"
GRID_COLOR = "#274352"
TEXT_COLOR = "#F3F7FA"
MUTED_TEXT_COLOR = "#AABAC6"
LEGEND_BACKGROUND = "#0C2231"

# The first three colors mirror the cyan, violet, and amber series used in the
# visual reference.  The remaining colors keep larger series distinguishable.
SERIES_COLORS = (
    "#19C2F1",
    "#AD7AFF",
    "#F7B844",
    "#59D0A2",
    "#F47F98",
    "#7FC7FF",
    "#F39A58",
    "#C3D96B",
)


def apply_dashboard_theme() -> None:
    """Set shared Matplotlib defaults for the dark benchmark-report theme."""

    mpl.rcParams.update(
        {
            "figure.facecolor": PAGE_BACKGROUND,
            "axes.facecolor": PANEL_BACKGROUND,
            "savefig.facecolor": PAGE_BACKGROUND,
            "savefig.edgecolor": PAGE_BACKGROUND,
            "text.color": TEXT_COLOR,
            "axes.labelcolor": MUTED_TEXT_COLOR,
            "axes.titlecolor": TEXT_COLOR,
            "xtick.color": MUTED_TEXT_COLOR,
            "ytick.color": MUTED_TEXT_COLOR,
            "axes.edgecolor": PANEL_BORDER,
            "axes.prop_cycle": mpl.cycler(color=SERIES_COLORS),
            "font.family": ["Segoe UI", "DejaVu Sans"],
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.titleweight": "semibold",
            "axes.labelsize": 10,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 9,
            "legend.title_fontsize": 9,
            "lines.linewidth": 2.15,
            "lines.markersize": 5.5,
            "grid.color": GRID_COLOR,
            "grid.linewidth": 0.7,
            "grid.alpha": 0.9,
        }
    )


def apply_dashboard_style(fig: Figure, ax: Axes) -> None:
    """Style one chart as a compact dark report panel."""

    apply_dashboard_theme()
    fig.patch.set_facecolor(PAGE_BACKGROUND)
    ax.set_facecolor(PANEL_BACKGROUND)
    ax.set_prop_cycle(color=SERIES_COLORS)
    ax.set_axisbelow(True)
    ax.xaxis.label.set_color(MUTED_TEXT_COLOR)
    ax.yaxis.label.set_color(MUTED_TEXT_COLOR)
    ax.xaxis.label.set_fontsize(10)
    ax.yaxis.label.set_fontsize(10)
    ax.grid(True, which="major", color=GRID_COLOR, linewidth=0.7, alpha=0.9)
    ax.grid(False, which="minor")
    ax.tick_params(
        axis="both",
        which="major",
        colors=MUTED_TEXT_COLOR,
        length=3.5,
        width=0.8,
        pad=5,
        labelsize=9,
    )

    for spine in ax.spines.values():
        spine.set_color(PANEL_BORDER)
        spine.set_linewidth(0.8)


def set_dashboard_title(ax: Axes, title: str) -> None:
    """Apply the left-aligned title treatment used by the report cards."""

    title_text = ax.set_title(title, loc="left", pad=15, fontsize=12, fontweight="semibold")
    title_text.set_color(TEXT_COLOR)


def style_dashboard_legend(ax: Axes, **kwargs: Any) -> Legend | None:
    """Create a compact legend that remains readable on the dark panel."""

    kwargs.setdefault("frameon", False)
    kwargs.setdefault("borderpad", 0.35)
    kwargs.setdefault("labelspacing", 0.55)
    kwargs.setdefault("handlelength", 2.2)
    legend = ax.legend(**kwargs)
    if legend is None:
        return None

    frame = legend.get_frame()
    frame.set_facecolor(LEGEND_BACKGROUND)
    frame.set_edgecolor(PANEL_BORDER)
    frame.set_linewidth(0.8)

    for text in legend.get_texts():
        text.set_color(TEXT_COLOR)
    legend.get_title().set_color(MUTED_TEXT_COLOR)
    return legend


def save_dashboard_figure(
    fig: Figure,
    ax: Axes,
    output_path: Path,
    *,
    use_tight_layout: bool = True,
) -> None:
    """Finish a themed chart and save it with its dark page background.

    A light rounded border around the plotting area echoes the panel framing in
    the reference dashboard.  It is added only after layout has settled so it
    hugs the final chart dimensions and any externally anchored legend.
    """

    if use_tight_layout:
        fig.tight_layout(pad=1.35)

    fig.canvas.draw()
    _add_panel_outline(fig, ax)
    fig.savefig(output_path, dpi=200, facecolor=fig.get_facecolor(), edgecolor="none")


def _add_panel_outline(fig: Figure, ax: Axes) -> None:
    """Add one rounded panel outline around an axes and its legend."""

    for artist in list(fig.artists):
        if getattr(artist, "_benchmark_panel_outline", False):
            artist.remove()

    x0, y0, width, height = ax.get_position().bounds
    x1 = x0 + width
    y1 = y0 + height

    legend = ax.get_legend()
    if legend is not None and legend.get_visible():
        legend_box = legend.get_window_extent(fig.canvas.get_renderer()).transformed(
            fig.transFigure.inverted()
        )
        x0 = min(x0, legend_box.x0)
        y0 = min(y0, legend_box.y0)
        x1 = max(x1, legend_box.x1)
        y1 = max(y1, legend_box.y1)

    for title in (ax.title, getattr(ax, "_left_title", None), getattr(ax, "_right_title", None)):
        if title is None or not title.get_visible() or not title.get_text():
            continue
        title_box = title.get_window_extent(fig.canvas.get_renderer()).transformed(fig.transFigure.inverted())
        x0 = min(x0, title_box.x0)
        y0 = min(y0, title_box.y0)
        x1 = max(x1, title_box.x1)
        y1 = max(y1, title_box.y1)

    horizontal_padding = max(0.0, min(0.014, x0, 1.0 - x1))
    vertical_padding = max(0.0, min(0.018, y0, 1.0 - y1))
    panel = FancyBboxPatch(
        (x0 - horizontal_padding, y0 - vertical_padding),
        (x1 - x0) + horizontal_padding * 2,
        (y1 - y0) + vertical_padding * 2,
        boxstyle="round,pad=0.004,rounding_size=0.012",
        transform=fig.transFigure,
        clip_on=False,
        facecolor=PANEL_BACKGROUND,
        edgecolor=PANEL_BORDER,
        linewidth=0.85,
        zorder=-10,
    )
    panel._benchmark_panel_outline = True
    fig.add_artist(panel)
