"""Reusable Streamlit UI blocks.

Every card the dashboard draws is produced here, so spacing, shadows, icon
treatment and colour semantics stay identical on all seven pages. Icons are
inline SVG (stroke-based, ``currentColor``) which keeps the app fully
self-contained — no external font or CDN request.
"""

from __future__ import annotations

import base64
import html
import math
from dataclasses import dataclass
from itertools import count
from typing import Iterable, Literal, Sequence

import numpy as np
import streamlit as st

from dashboard.config import Palette

_UID = count()

Tone = Literal["primary", "success", "warning", "danger", "neutral"]

# --------------------------------------------------------------------------
# Icon set (24x24, stroke-based)
# --------------------------------------------------------------------------
_ICON_BODIES: dict[str, str] = {
    "database": '<ellipse cx="12" cy="5" rx="9" ry="3"/><path d="M21 12c0 1.66-4 3-9 3s-9-1.34-9-3"/><path d="M3 5v14c0 1.66 4 3 9 3s9-1.34 9-3V5"/>',
    "calendar": '<rect x="3" y="4" width="18" height="18" rx="2"/><path d="M16 2v4M8 2v4M3 10h18"/>',
    "clock": '<circle cx="12" cy="12" r="10"/><path d="M12 6v6l4 2"/>',
    "award": '<circle cx="12" cy="8" r="6"/><path d="M8.5 13.5L7 23l5-3 5 3-1.5-9.5"/>',
    "shield": '<path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"/>',
    "price": '<path d="M12 1v22"/><path d="M17 5H9.5a3.5 3.5 0 0 0 0 7h5a3.5 3.5 0 0 1 0 7H6"/>',
    "trend_up": '<polyline points="23 6 13.5 15.5 8.5 10.5 1 18"/><polyline points="17 6 23 6 23 12"/>',
    "trend_down": '<polyline points="23 18 13.5 8.5 8.5 13.5 1 6"/><polyline points="17 18 23 18 23 12"/>',
    "trend_flat": '<path d="M4 12h16"/><polyline points="16 8 20 12 16 16"/>',
    "percent": '<path d="M19 5L5 19"/><circle cx="6.5" cy="6.5" r="2.5"/><circle cx="17.5" cy="17.5" r="2.5"/>',
    "activity": '<path d="M22 12h-4l-3 9L9 3l-3 9H2"/>',
    "chart": '<path d="M18 20V10M12 20V4M6 20v-4"/>',
    "alert": '<path d="M10.29 3.86L1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0z"/><path d="M12 9v4M12 17h.01"/>',
    "check": '<path d="M22 11.08V12a10 10 0 1 1-5.93-9.14"/><polyline points="22 4 12 14.01 9 11.01"/>',
    "bulb": '<path d="M9 18h6M10 22h4"/><path d="M12 2a7 7 0 0 0-4 12.7V17h8v-2.3A7 7 0 0 0 12 2z"/>',
    "policy": '<path d="M12 3v18M5 21h14"/><path d="M3 8l4-4 4 4M3 8c0 2.2 1.8 4 4 4s4-1.8 4-4"/><path d="M13 8l4-4 4 4M13 8c0 2.2 1.8 4 4 4s4-1.8 4-4"/>',
    "cart": '<circle cx="9" cy="21" r="1"/><circle cx="20" cy="21" r="1"/><path d="M1 1h4l2.68 13.39a2 2 0 0 0 2 1.61h9.72a2 2 0 0 0 2-1.61L23 6H6"/>',
    "download": '<path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/><polyline points="7 10 12 15 17 10"/><path d="M12 15V3"/>',
    "layers": '<path d="M12 2L2 7l10 5 10-5-10-5z"/><path d="M2 17l10 5 10-5"/><path d="M2 12l10 5 10-5"/>',
    "zap": '<path d="M13 2L3 14h9l-1 8 10-12h-9l1-8z"/>',
    "target": '<circle cx="12" cy="12" r="10"/><circle cx="12" cy="12" r="6"/><circle cx="12" cy="12" r="2"/>',
    "compass": '<circle cx="12" cy="12" r="10"/><polygon points="16.24 7.76 14.12 14.12 7.76 16.24 9.88 9.88 16.24 7.76"/>',
    "file": '<path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/><polyline points="14 2 14 8 20 8"/><path d="M8 13h8M8 17h5"/>',
    "grid": '<rect x="3" y="3" width="7" height="7" rx="1"/><rect x="14" y="3" width="7" height="7" rx="1"/><rect x="3" y="14" width="7" height="7" rx="1"/><rect x="14" y="14" width="7" height="7" rx="1"/>',
    "leaf": '<path d="M11 20A7 7 0 0 1 9.8 6.1C15.5 5 17 4.48 19 2c1 2 2 4.18 2 8 0 5.5-4.78 10-10 10z"/><path d="M2 21c0-3 1.85-5.36 5.08-6"/>',
    "drop": '<path d="M12 2.7l5.3 5.3a7.5 7.5 0 1 1-10.6 0z"/>',
}


def _data_uri(svg: str) -> str:
    """Encode an SVG document as a base64 ``data:`` URI.

    Streamlit sanitises HTML with DOMPurify's HTML profile, which strips inline
    ``<svg>`` elements. Wrapping the graphic in an ``<img src="data:…">``
    survives sanitisation and keeps the app fully self-contained — no CDN, no
    icon font, no external request.
    """
    return "data:image/svg+xml;base64," + base64.b64encode(svg.encode("utf-8")).decode("ascii")


def icon(name: str, size: int = 18, color: str = "currentColor", stroke: float = 1.9) -> str:
    """Return an icon as a self-contained ``<img>`` element.

    Args:
        name: Key from the built-in icon set; unknown names fall back to a dot.
        size: Rendered edge length in pixels.
        color: Stroke colour, baked into the SVG (a data URI cannot inherit
            ``currentColor`` from the host document).
        stroke: Stroke width in user units.

    Returns:
        An ``<img>`` element carrying a base64 SVG.
    """
    body = _ICON_BODIES.get(name, '<circle cx="12" cy="12" r="4"/>')
    svg = (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{size}" height="{size}" '
        f'viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="{stroke}" '
        f'stroke-linecap="round" stroke-linejoin="round">{body}</svg>'
    )
    return (
        f'<img src="{_data_uri(svg)}" width="{size}" height="{size}" alt="" '
        f'style="display:block">'
    )


# --------------------------------------------------------------------------
# Small primitives
# --------------------------------------------------------------------------
def _tone_colors(tone: Tone, palette: Palette) -> tuple[str, str]:
    """Return ``(foreground, translucent background)`` for a semantic tone."""
    mapping = {
        "primary": (palette.primary, "rgba(37,99,235,.12)"),
        "success": (palette.success, "rgba(22,163,74,.13)"),
        "warning": (palette.warning, "rgba(245,158,11,.15)"),
        "danger": (palette.danger, "rgba(220,38,38,.12)"),
        "neutral": (palette.text_muted, "rgba(100,116,139,.14)"),
    }
    return mapping.get(tone, mapping["primary"])


def sparkline(
    values: Sequence[float],
    color: str,
    width: int = 220,
    height: int = 34,
) -> str:
    """Render a mini sparkline as a self-contained SVG.

    Args:
        values: Ordered numeric series; fewer than two points renders nothing.
        color: Stroke colour.
        width: viewBox width (the element itself stretches to its container).
        height: viewBox height.

    Returns:
        An ``<svg>`` string, or an empty string when there is no data.
    """
    clean = [float(v) for v in values if v is not None and math.isfinite(float(v))]
    if len(clean) < 2:
        return ""

    lo, hi = min(clean), max(clean)
    span = hi - lo or 1.0
    pad = 3.0
    step = width / (len(clean) - 1)
    points = [
        (i * step, height - pad - (v - lo) / span * (height - 2 * pad))
        for i, v in enumerate(clean)
    ]
    line = " ".join(f"{x:.1f},{y:.1f}" for x, y in points)
    area = f"0,{height} {line} {width},{height}"
    gid = f"bdpspark{next(_UID)}"

    svg = (
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" '
        f'preserveAspectRatio="none">'
        f'<defs><linearGradient id="{gid}" x1="0" y1="0" x2="0" y2="1">'
        f'<stop offset="0%" stop-color="{color}" stop-opacity=".26"/>'
        f'<stop offset="100%" stop-color="{color}" stop-opacity="0"/>'
        f"</linearGradient></defs>"
        f'<polygon points="{area}" fill="url(#{gid})"/>'
        f'<polyline points="{line}" fill="none" stroke="{color}" stroke-width="1.8" '
        f'stroke-linecap="round" stroke-linejoin="round"/>'
        f"</svg>"
    )
    return f'<img class="bdp-kpi-spark" src="{_data_uri(svg)}" alt="">'


def page_header(title: str, subtitle: str) -> None:
    """Render the standard page title block.

    Args:
        title: Page title.
        subtitle: One-line description of what the page answers.
    """
    st.html(
        f'<div class="bdp-page-head"><h1>{html.escape(title)}</h1>'
        f"<p>{html.escape(subtitle)}</p></div>"
    )


def section(title: str, hint: str = "") -> None:
    """Render a section heading with an optional explanatory hint.

    Args:
        title: Section name.
        hint: The analytical question the section answers.
    """
    hint_html = f"<span>{html.escape(hint)}</span>" if hint else ""
    st.html(f'<div class="bdp-section"><h3>{html.escape(title)}</h3>{hint_html}</div>')


def pill(text: str, level: Literal["low", "mod", "high", "info"] = "info") -> str:
    """Return a coloured status pill as an HTML string."""
    return f'<span class="bdp-pill bdp-pill-{level}">{html.escape(text)}</span>'


# --------------------------------------------------------------------------
# Status strip
# --------------------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class StatusCard:
    """One tile in the always-visible dashboard status strip."""

    label: str
    value: str
    sub: str = ""
    icon: str = "database"
    tone: Tone = "primary"


def status_strip(cards: Iterable[StatusCard], palette: Palette) -> None:
    """Render the dashboard status strip.

    Args:
        cards: Tiles to display, in order.
        palette: Active palette.
    """
    tiles: list[str] = []
    for card in cards:
        fg, bg = _tone_colors(card.tone, palette)
        sub = f'<div class="bdp-status-sub">{html.escape(card.sub)}</div>' if card.sub else ""
        tiles.append(
            f'<div class="bdp-status">'
            f'<div class="bdp-status-icon" style="background:{bg}">{icon(card.icon, 18, fg)}</div>'
            f"<div><div class=\"bdp-status-label\">{html.escape(card.label)}</div>"
            f'<div class="bdp-status-value">{html.escape(card.value)}</div>{sub}</div>'
            f"</div>"
        )
    st.html(f'<div class="bdp-status-grid">{"".join(tiles)}</div>')


# --------------------------------------------------------------------------
# KPI cards
# --------------------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class KpiCard:
    """One KPI tile: value, trend, sparkline and hover tooltip."""

    label: str
    value: str
    unit: str = ""
    delta: float | None = None
    delta_suffix: str = "%"
    #: ``True`` when a rising value is bad (prices, inflation).
    higher_is_worse: bool = True
    footnote: str = ""
    tooltip: str = ""
    icon: str = "price"
    tone: Tone = "primary"
    spark: Sequence[float] = ()
    #: Free-form status pill (e.g. "↓ Decreasing") that overrides the
    #: delta-vs-baseline pill below. Leave empty to use ``delta`` instead.
    status_text: str = ""
    status_tone: Literal["up", "down", "flat"] = "flat"
    #: Hide the pill entirely when neither ``status_text`` nor ``delta`` applies.
    hide_delta: bool = False


def _delta_html(card: KpiCard, palette: Palette) -> str:
    """Render the trend arrow / delta pill for a KPI card."""
    if card.status_text:
        cls = {"up": "bdp-up", "down": "bdp-down", "flat": "bdp-flat"}[card.status_tone]
        return f'<span class="bdp-kpi-delta {cls}">{html.escape(card.status_text)}</span>'
    if card.hide_delta:
        return ""
    if card.delta is None or not np.isfinite(card.delta):
        return '<span class="bdp-kpi-delta bdp-flat">no change data</span>'

    if abs(card.delta) < 0.05:
        cls, arrow = "bdp-flat", "→"
    elif card.delta > 0:
        cls, arrow = ("bdp-up" if card.higher_is_worse else "bdp-down"), "▲"
    else:
        cls, arrow = ("bdp-down" if card.higher_is_worse else "bdp-up"), "▼"
    return (
        f'<span class="bdp-kpi-delta {cls}">{arrow} {card.delta:+.2f}'
        f"{html.escape(card.delta_suffix)}</span>"
    )


def kpi_row(cards: Sequence[KpiCard], palette: Palette) -> None:
    """Render a responsive row of KPI cards.

    Args:
        cards: Cards to display, in order.
        palette: Active palette.
    """
    tiles: list[str] = []
    for card in cards:
        fg, bg = _tone_colors(card.tone, palette)
        unit = f"<small>{html.escape(card.unit)}</small>" if card.unit else ""
        foot = f'<div class="bdp-kpi-foot">{html.escape(card.footnote)}</div>' if card.footnote else ""
        tooltip = html.escape(card.tooltip or f"{card.label}: {card.value} {card.unit}".strip())
        tiles.append(
            f'<div class="bdp-kpi" title="{tooltip}">'
            f'<div class="bdp-kpi-top">'
            f'<div class="bdp-kpi-label">{html.escape(card.label)}</div>'
            f'<div class="bdp-kpi-icon" style="background:{bg}">{icon(card.icon, 16, fg)}</div>'
            f"</div>"
            f'<div class="bdp-kpi-value">{html.escape(card.value)}{unit}</div>'
            f"{_delta_html(card, palette)}{foot}"
            f"{sparkline(card.spark, fg)}"
            f"</div>"
        )
    st.html(f'<div class="bdp-kpi-grid">{"".join(tiles)}</div>')


# --------------------------------------------------------------------------
# Insight / recommendation cards
# --------------------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class InsightCard:
    """A narrative card used on the Insights & Recommendations page."""

    title: str
    body: str
    icon: str = "bulb"
    tone: Tone = "primary"


def insight_card(card: InsightCard, palette: Palette) -> None:
    """Render a single insight card. ``card.body`` may contain ``<b>`` tags."""
    fg, bg = _tone_colors(card.tone, palette)
    st.html(
        f'<div class="bdp-insight" style="border-left-color:{fg}">'
        f'<div class="bdp-insight-head">'
        f'<div class="bdp-kpi-icon" style="background:{bg}">{icon(card.icon, 15, fg)}</div>'
        f'<div class="bdp-insight-title">{html.escape(card.title)}</div></div>'
        f'<div class="bdp-insight-body">{card.body}</div></div>'
    )


def insight_grid(cards: Sequence[InsightCard], palette: Palette, columns: int = 3) -> None:
    """Lay insight cards out on a fixed grid that wraps on narrow screens.

    Args:
        cards: Cards to render.
        palette: Active palette.
        columns: Cards per row on wide screens.
    """
    if not cards:
        empty_state("Nothing to report for the current selection.")
        return
    for start in range(0, len(cards), columns):
        row = cards[start : start + columns]
        cols = st.columns(len(row), gap="small")
        for col, card in zip(cols, row):
            with col:
                insight_card(card, palette)


def footer(text: str, right: str = "") -> None:
    """Render the shared page footer."""
    st.html(
        f'<div class="bdp-foot"><span>{html.escape(text)}</span>'
        f"<span>{html.escape(right)}</span></div>"
    )


def empty_state(message: str, hint: str = "") -> None:
    """Render a friendly placeholder when a section has no data to show."""
    st.info(f"**{message}**\n\n{hint}" if hint else f"**{message}**", icon=":material/info:")
