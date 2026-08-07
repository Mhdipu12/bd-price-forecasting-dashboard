"""Presentation layer: CSS design tokens and the shared Plotly template.

The dashboard intentionally uses a small, explicit CSS layer (the brief asks
for BI-style cards with soft shadows, rounded corners and mini sparklines,
which Streamlit's native widgets do not provide). Everything is driven from
:data:`dashboard.config.PALETTES`, so the light and dark modes stay in sync.
"""

from __future__ import annotations

from functools import lru_cache

import plotly.graph_objects as go
import plotly.io as pio
import streamlit as st

from dashboard.config import COMMODITY_COLORS, DARK_PALETTE, LIGHT_PALETTE, PALETTES, Palette

_TEMPLATE_LIGHT = "bdp_light"
_TEMPLATE_DARK = "bdp_dark"


def get_palette(mode: str) -> Palette:
    """Return the palette for a theme mode, defaulting to light.

    Args:
        mode: ``"Light"`` or ``"Dark"``.
    """
    return PALETTES.get(mode, LIGHT_PALETTE)


# --------------------------------------------------------------------------
# Plotly
# --------------------------------------------------------------------------
def _build_template(palette: Palette) -> go.layout.Template:
    """Create a Plotly template that matches one palette."""
    return go.layout.Template(
        layout=go.Layout(
            font={"family": "Inter, 'Segoe UI', system-ui, sans-serif", "size": 13,
                  "color": palette.text},
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            colorway=[
                palette.primary,
                COMMODITY_COLORS["Onion"],
                COMMODITY_COLORS["Potato"],
                "#7C3AED",
                palette.warning,
                palette.danger,
            ],
            margin={"l": 8, "r": 8, "t": 48, "b": 8},
            hoverlabel={
                "bgcolor": palette.surface,
                "bordercolor": palette.border,
                "font": {"color": palette.text, "size": 12},
            },
            hovermode="x unified",
            title={"font": {"size": 15, "color": palette.text}, "x": 0, "xanchor": "left"},
            legend={
                "orientation": "h",
                "yanchor": "bottom",
                "y": 1.02,
                "xanchor": "left",
                "x": 0,
                "bgcolor": "rgba(0,0,0,0)",
                "font": {"size": 12},
            },
            xaxis={
                "gridcolor": palette.grid,
                "zerolinecolor": palette.grid,
                "linecolor": palette.border,
                "tickfont": {"color": palette.text_muted, "size": 11},
                "title": {"font": {"color": palette.text_muted, "size": 12}},
            },
            yaxis={
                "gridcolor": palette.grid,
                "zerolinecolor": palette.grid,
                "linecolor": palette.border,
                "tickfont": {"color": palette.text_muted, "size": 11},
                "title": {"font": {"color": palette.text_muted, "size": 12}},
            },
        )
    )


@lru_cache(maxsize=2)
def _register_templates() -> tuple[str, str]:
    """Register both Plotly templates exactly once per process."""
    pio.templates[_TEMPLATE_LIGHT] = _build_template(LIGHT_PALETTE)
    pio.templates[_TEMPLATE_DARK] = _build_template(DARK_PALETTE)
    return _TEMPLATE_LIGHT, _TEMPLATE_DARK


def plotly_template(mode: str) -> str:
    """Return the registered Plotly template name for a theme mode."""
    light, dark = _register_templates()
    return dark if mode == "Dark" else light


def plotly_config(filename: str = "chart") -> dict[str, object]:
    """Return a Plotly ``config`` dict with a clean, download-friendly toolbar.

    Args:
        filename: Base file name used when the user downloads the chart image.
    """
    return {
        "displaylogo": False,
        "modeBarButtonsToRemove": ["lasso2d", "select2d", "autoScale2d"],
        "toImageButtonOptions": {
            "format": "png",
            "filename": filename,
            "height": 640,
            "width": 1280,
            "scale": 2,
        },
        "scrollZoom": False,
        "responsive": True,
    }


# --------------------------------------------------------------------------
# CSS
# --------------------------------------------------------------------------
def _css(palette: Palette, mode: str) -> str:
    """Render the design-token stylesheet for one palette."""
    dark = mode == "Dark"
    header_bg = palette.background
    return f"""
<style>
:root {{
    --bdp-bg: {palette.background};
    --bdp-surface: {palette.surface};
    --bdp-surface-alt: {palette.surface_alt};
    --bdp-border: {palette.border};
    --bdp-text: {palette.text};
    --bdp-muted: {palette.text_muted};
    --bdp-primary: {palette.primary};
    --bdp-success: {palette.success};
    --bdp-warning: {palette.warning};
    --bdp-danger: {palette.danger};
    --bdp-shadow: {palette.shadow};
    --bdp-radius: 14px;
}}

/* ---------- app shell ---------- */
.stApp {{
    background: var(--bdp-bg);
    color: var(--bdp-text);
}}
[data-testid="stHeader"] {{
    background: {header_bg};
    border-bottom: 1px solid var(--bdp-border);
}}
.block-container {{
    padding-top: 2.1rem;
    padding-bottom: 3rem;
    max-width: 1480px;
}}
.stApp, .stApp p, .stApp li, .stApp label, .stApp span,
.stApp h1, .stApp h2, .stApp h3, .stApp h4, .stApp h5 {{
    color: var(--bdp-text);
}}
.stApp h1 {{ font-size: 1.85rem; font-weight: 700; letter-spacing: -.02em; }}
.stApp h2 {{ font-size: 1.25rem; font-weight: 650; }}
.stApp h3 {{ font-size: 1.05rem; font-weight: 620; }}

/* ---------- sticky sidebar ---------- */
section[data-testid="stSidebar"] {{
    background: {palette.sidebar};
    border-right: 1px solid rgba(148,163,184,.16);
}}
section[data-testid="stSidebar"] > div {{
    position: sticky;
    top: 0;
    max-height: 100vh;
    overflow-y: auto;
}}
section[data-testid="stSidebar"] * {{ color: {palette.sidebar_text}; }}
section[data-testid="stSidebar"] label {{
    font-size: .78rem;
    font-weight: 600;
    letter-spacing: .04em;
    text-transform: uppercase;
    color: #94A3B8 !important;
}}
section[data-testid="stSidebar"] [data-testid="stWidgetLabel"] p {{
    color: #94A3B8 !important;
}}
section[data-testid="stSidebar"] hr {{ border-color: rgba(148,163,184,.18); }}

/* ---------- brand block ---------- */
.bdp-brand {{
    display: flex; gap: .7rem; align-items: center;
    padding: .35rem 0 .9rem 0;
}}
.bdp-brand-mark {{
    width: 38px; height: 38px; border-radius: 11px; flex: 0 0 38px;
    background: linear-gradient(140deg, #2563EB, #7C3AED);
    display: flex; align-items: center; justify-content: center;
    color: #fff; font-weight: 700; font-size: .95rem;
}}
.bdp-brand-text b {{ display: block; font-size: .92rem; line-height: 1.2; color: #F8FAFC; }}
.bdp-brand-text span {{ font-size: .72rem; color: #94A3B8; }}

/* ---------- page header ---------- */
.bdp-page-head {{ margin: 0 0 1.15rem 0; }}
.bdp-page-head h1 {{ margin: 0 0 .2rem 0; }}
.bdp-page-head p {{ margin: 0; color: var(--bdp-muted); font-size: .9rem; }}

/* ---------- generic card ---------- */
.bdp-card {{
    background: var(--bdp-surface);
    border: 1px solid var(--bdp-border);
    border-radius: var(--bdp-radius);
    box-shadow: var(--bdp-shadow);
    padding: 1.05rem 1.15rem;
    height: 100%;
}}

/* ---------- status strip ---------- */
.bdp-status-grid {{
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(190px, 1fr));
    gap: .8rem;
    margin-bottom: 1.25rem;
}}
.bdp-status {{
    background: var(--bdp-surface);
    border: 1px solid var(--bdp-border);
    border-radius: var(--bdp-radius);
    box-shadow: var(--bdp-shadow);
    padding: .85rem 1rem;
    display: flex; gap: .75rem; align-items: flex-start;
}}
.bdp-status-icon {{
    width: 34px; height: 34px; border-radius: 10px; flex: 0 0 34px;
    display: flex; align-items: center; justify-content: center;
    font-size: 18px;
}}
.bdp-status-label {{
    font-size: .68rem; font-weight: 700; letter-spacing: .07em;
    text-transform: uppercase; color: var(--bdp-muted); margin-bottom: .18rem;
}}
.bdp-status-value {{ font-size: .95rem; font-weight: 650; line-height: 1.25; }}
.bdp-status-sub {{ font-size: .72rem; color: var(--bdp-muted); margin-top: .12rem; }}

/* ---------- KPI card ---------- */
.bdp-kpi-grid {{
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(215px, 1fr));
    gap: .8rem;
    margin-bottom: 1.25rem;
}}
.bdp-kpi {{
    position: relative;
    background: var(--bdp-surface);
    border: 1px solid var(--bdp-border);
    border-radius: var(--bdp-radius);
    box-shadow: var(--bdp-shadow);
    padding: .95rem 1.05rem 0.7rem 1.05rem;
    transition: transform .16s ease, box-shadow .16s ease;
    overflow: hidden;
}}
.bdp-kpi:hover {{
    transform: translateY(-2px);
    box-shadow: 0 2px 4px rgba(15,23,42,.08), 0 18px 32px -18px rgba(37,99,235,.45);
}}
.bdp-kpi-top {{ display: flex; justify-content: space-between; align-items: center; gap: .5rem; }}
.bdp-kpi-label {{
    font-size: .7rem; font-weight: 700; letter-spacing: .06em;
    text-transform: uppercase; color: var(--bdp-muted);
}}
.bdp-kpi-icon {{
    width: 28px; height: 28px; border-radius: 8px; flex: 0 0 28px;
    display: flex; align-items: center; justify-content: center; font-size: 15px;
}}
.bdp-kpi-value {{
    font-size: 1.55rem; font-weight: 700; letter-spacing: -.02em;
    margin: .35rem 0 .15rem 0; line-height: 1.15;
}}
.bdp-kpi-value small {{ font-size: .72rem; font-weight: 600; color: var(--bdp-muted); margin-left: .25rem; }}
.bdp-kpi-delta {{
    display: inline-flex; align-items: center; gap: .25rem;
    font-size: .78rem; font-weight: 650;
    padding: .1rem .45rem; border-radius: 999px;
}}
.bdp-kpi-foot {{ font-size: .72rem; color: var(--bdp-muted); margin-top: .3rem; }}
.bdp-kpi-spark {{ margin-top: .45rem; height: 34px; display: block; width: 100%; }}

.bdp-up   {{ color: var(--bdp-danger);  background: rgba(220,38,38,.10); }}
.bdp-down {{ color: var(--bdp-success); background: rgba(22,163,74,.12); }}
.bdp-flat {{ color: var(--bdp-muted);   background: rgba(100,116,139,.14); }}

/* ---------- risk / severity pills ---------- */
.bdp-pill {{
    display: inline-block; padding: .16rem .6rem; border-radius: 999px;
    font-size: .72rem; font-weight: 700; letter-spacing: .02em;
}}
.bdp-pill-low  {{ color: var(--bdp-success); background: rgba(22,163,74,.13); }}
.bdp-pill-mod  {{ color: var(--bdp-warning); background: rgba(245,158,11,.15); }}
.bdp-pill-high {{ color: var(--bdp-danger);  background: rgba(220,38,38,.13); }}
.bdp-pill-info {{ color: var(--bdp-primary); background: rgba(37,99,235,.13); }}

/* ---------- insight card ---------- */
.bdp-insight {{
    background: var(--bdp-surface);
    border: 1px solid var(--bdp-border);
    border-left: 4px solid var(--bdp-primary);
    border-radius: var(--bdp-radius);
    box-shadow: var(--bdp-shadow);
    padding: 1rem 1.15rem;
    height: 100%;
}}
.bdp-insight-head {{ display: flex; align-items: center; gap: .55rem; margin-bottom: .4rem; }}
.bdp-insight-head .bdp-kpi-icon {{ width: 26px; height: 26px; flex: 0 0 26px; font-size: 14px; }}
.bdp-insight-title {{ font-size: .88rem; font-weight: 700; }}
.bdp-insight-body {{ font-size: .86rem; line-height: 1.55; color: var(--bdp-text); opacity: .92; }}
.bdp-insight-body b {{ font-weight: 700; }}

/* ---------- section heading ---------- */
.bdp-section {{
    display: flex; align-items: baseline; gap: .6rem;
    margin: 1.55rem 0 .65rem 0;
}}
.bdp-section h3 {{ margin: 0; font-size: 1.02rem; font-weight: 680; }}
.bdp-section span {{ font-size: .8rem; color: var(--bdp-muted); }}

/* ---------- native widget polish ---------- */
[data-testid="stMetric"] {{
    background: var(--bdp-surface);
    border: 1px solid var(--bdp-border);
    border-radius: var(--bdp-radius);
    box-shadow: var(--bdp-shadow);
    padding: .85rem 1rem;
}}
div[data-testid="stDataFrame"], div[data-testid="stTable"] {{
    border-radius: 12px; overflow: hidden;
}}
.stTabs [data-baseweb="tab-list"] {{ gap: .35rem; border-bottom: 1px solid var(--bdp-border); }}
.stTabs [data-baseweb="tab"] {{
    border-radius: 9px 9px 0 0; padding: .35rem .85rem; font-weight: 600; font-size: .87rem;
}}
div[data-testid="stExpander"] details {{
    background: var(--bdp-surface);
    border: 1px solid var(--bdp-border);
    border-radius: 12px;
}}
.js-plotly-plot .plotly .modebar {{ opacity: .35; transition: opacity .2s ease; }}
.js-plotly-plot:hover .plotly .modebar {{ opacity: 1; }}

/* ---------- footer ---------- */
.bdp-foot {{
    margin-top: 2.4rem; padding-top: 1rem;
    border-top: 1px solid var(--bdp-border);
    font-size: .76rem; color: var(--bdp-muted);
    display: flex; justify-content: space-between; flex-wrap: wrap; gap: .5rem;
}}

/* ---------- responsive ---------- */
@media (max-width: 900px) {{
    .block-container {{ padding-left: 1rem; padding-right: 1rem; }}
    .bdp-kpi-value {{ font-size: 1.32rem; }}
    .bdp-kpi-grid {{ grid-template-columns: repeat(auto-fit, minmax(165px, 1fr)); }}
    .bdp-status-grid {{ grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); }}
}}
{"" if not dark else _DARK_EXTRAS}
</style>
"""


#: Dark-mode overrides for Streamlit's own widgets.
#:
#: The app's Light/Dark control restyles everything this dashboard draws. Two
#: Streamlit surfaces need explicit help: BaseWeb form controls (which read
#: their colours from the native theme) and ``st.dataframe``, which paints to a
#: canvas that CSS cannot reach. Data grids are therefore presented as a light
#: "paper" panel on the dark canvas — a deliberate BI convention rather than a
#: half-applied theme.
_DARK_EXTRAS = """
.bdp-kpi:hover { box-shadow: 0 2px 4px rgba(0,0,0,.5), 0 18px 32px -18px rgba(96,165,250,.55); }

[data-baseweb="select"] > div,
[data-baseweb="base-input"],
.stTextInput input,
.stNumberInput input,
.stDateInput input {
    background-color: #16243D !important;
    border-color: #1E2E4A !important;
    color: #E2E8F0 !important;
}
[data-baseweb="select"] svg { fill: #94A3B8 !important; }
[data-baseweb="popover"] [role="listbox"],
[data-baseweb="popover"] ul {
    background-color: #16243D !important;
    border: 1px solid #1E2E4A !important;
}
[data-baseweb="popover"] li { color: #E2E8F0 !important; }
[data-baseweb="popover"] li:hover { background-color: #1E2E4A !important; }
[data-baseweb="calendar"] { background-color: #16243D !important; color: #E2E8F0 !important; }

.stTabs [data-baseweb="tab"] { color: #94A3B8; }
.stTabs [aria-selected="true"] { color: #E2E8F0; }

.stButton button, .stDownloadButton button {
    border-color: #1E2E4A;
}

/* Data grids stay on a light panel so their canvas-rendered text keeps
   contrast; the frame below makes that read as intentional. */
div[data-testid="stDataFrame"] {
    background: #F8FAFC;
    padding: 6px;
    border: 1px solid #1E2E4A;
    border-radius: 12px;
}
"""


def apply_theme(mode: str) -> Palette:
    """Inject the stylesheet for the requested mode and return its palette.

    Args:
        mode: ``"Light"`` or ``"Dark"``.

    Returns:
        The active :class:`~dashboard.config.Palette`.
    """
    palette = get_palette(mode)
    st.html(_css(palette, mode))
    return palette
