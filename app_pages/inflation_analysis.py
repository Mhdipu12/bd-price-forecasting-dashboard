"""Inflation analysis — realised and projected rates at every time scale."""

from __future__ import annotations

import numpy as np
import streamlit as st

from dashboard import charts
from dashboard.components import KpiCard, empty_state, footer, kpi_row, page_header, section
from dashboard.config import COMMODITY_BY_KEY, SETTINGS
from dashboard.context import get_context
from dashboard.theme import plotly_config
from dashboard.transforms import (
    commodity_forecast,
    commodity_inflation,
    commodity_prices,
    inflation_heatmap_matrix,
    periodic_inflation,
)

ctx = get_context()
bundle, filters, palette, mode = ctx.bundle, ctx.filters, ctx.palette, ctx.mode
meta = COMMODITY_BY_KEY[ctx.product]

page_header(
    f"{meta.label} — inflation analysis",
    "Realised inflation at daily, weekly and monthly scales, plus the rate implied by the "
    f"{ctx.model} forecast over the next {filters.horizon} days.",
)

prices = commodity_prices(bundle, ctx.product)
inflation = commodity_inflation(bundle, ctx.product)
path = commodity_forecast(bundle, ctx.product, ctx.model, filters.horizon)
snapshot = ctx.snapshot()

if inflation.empty and prices.empty:
    empty_state("No inflation data for this commodity.")
    st.stop()

monthly_series = (
    inflation["inflation_monthly"].dropna()
    if "inflation_monthly" in inflation.columns
    else prices["price"].pct_change(30).dropna() * 100
)

# --------------------------------------------------------------------------
# Summary cards
# --------------------------------------------------------------------------
if monthly_series.empty:
    average = float("nan")
else:
    average = float(monthly_series.mean())


def _direction(value: float) -> tuple[str, str, str]:
    """Classify a 30-day rate as decreasing/stable/increasing.

    Returns ``(arrow_and_word, status_tone, icon_name)``. The stable band
    reuses the app-wide "broadly stable" threshold so this page agrees with
    the regime classification shown elsewhere in the dashboard.
    """
    threshold = SETTINGS.thresholds.stable_change
    if not np.isfinite(value):
        return "→ Stable", "flat", "trend_flat"
    if value >= threshold:
        return "↑ Increasing", "up", "trend_up"
    if value <= -threshold:
        return "↓ Decreasing", "down", "trend_down"
    return "→ Stable", "flat", "trend_flat"


current_status, current_tone, current_icon = _direction(snapshot.current_inflation)
forecast_status, forecast_tone, forecast_icon = _direction(snapshot.forecast_inflation)
_tone_for_status = {"up": "danger", "down": "success", "flat": "neutral"}

kpi_row(
    [
        KpiCard(
            label="Current 30-day inflation",
            value=f"{snapshot.current_inflation:+.2f}" if np.isfinite(snapshot.current_inflation) else "—",
            unit="%",
            footnote="vs previous 30 days",
            tooltip="Latest realised 30-day price change for this commodity, versus the prior 30-day period.",
            icon=current_icon,
            tone=_tone_for_status[current_tone],
            status_text=current_status,
            status_tone=current_tone,
        ),
        KpiCard(
            label="Historical average",
            value=f"{average:+.2f}" if np.isfinite(average) else "—",
            unit="%",
            footnote="historical 30-day average",
            tooltip=f"Mean realised 30-day price-change rate for {meta.label} across the full sample.",
            icon="percent",
            tone="neutral",
            hide_delta=True,
            spark=monthly_series.tail(SETTINGS.sparkline_days).tolist(),
        ),
        KpiCard(
            label="Forecast inflation",
            value=f"{snapshot.forecast_inflation:+.2f}" if np.isfinite(snapshot.forecast_inflation) else "—",
            unit="%",
            footnote=f"next 30 days · {ctx.model}",
            tooltip="Forward 30-day price-change rate implied by the selected forecast model.",
            icon="target",
            tone="primary",
            hide_delta=True,
            spark=((path["forecast"] / snapshot.current_price - 1) * 100).tolist()
            if not path.empty
            else [],
        ),
        KpiCard(
            label="Expected price trend",
            value=forecast_status,
            footnote=f"{snapshot.forecast_inflation:+.2f}% expected over 30 days"
            if np.isfinite(snapshot.forecast_inflation)
            else "no forecast available",
            tooltip="Human-readable direction implied by the forecast inflation rate.",
            icon=forecast_icon,
            tone=_tone_for_status[forecast_tone],
            hide_delta=True,
        ),
    ],
    palette,
)

# --------------------------------------------------------------------------
# Historical vs forecast
# --------------------------------------------------------------------------
section(
    "Realised versus projected inflation",
    "Does the forecast imply a continuation of recent price pressure, or a break from it?",
)
st.plotly_chart(
    charts.inflation_history_vs_forecast(
        inflation, path, snapshot.current_price, ctx.product, mode, palette, height=380
    ),
    width="stretch",
    config=plotly_config(f"{ctx.product}_inflation_realised_vs_forecast"),
)

section(
    "Forecast daily inflation",
    "Is the projected move steady day by day, or concentrated in a few sharp steps?",
)
st.plotly_chart(
    charts.forecast_daily_inflation(path, snapshot.current_price, mode, palette, height=320),
    width="stretch",
    config=plotly_config(f"{ctx.product}_forecast_daily_inflation"),
)

# --------------------------------------------------------------------------
# Weekly and monthly
# --------------------------------------------------------------------------
section(
    "Weekly and monthly inflation",
    "Which time scale carries the signal, and which is mostly noise?",
)
left, right = st.columns(2, gap="medium")
with left:
    st.markdown("**Week-on-week** — short-run shocks and how quickly they reverse")
    st.plotly_chart(
        charts.periodic_inflation_bars(
            periodic_inflation(prices, "W"), "Weekly", mode, palette, tail=52
        ),
        width="stretch",
        config=plotly_config(f"{ctx.product}_weekly_inflation_page"),
    )
with right:
    st.markdown("**Month-on-month** — the measure comparable to official BBS reporting")
    st.plotly_chart(
        charts.periodic_inflation_bars(
            periodic_inflation(prices, "MS"), "Monthly", mode, palette, tail=36
        ),
        width="stretch",
        config=plotly_config(f"{ctx.product}_monthly_inflation_page"),
    )

# --------------------------------------------------------------------------
# Seasonality heatmap
# --------------------------------------------------------------------------
section(
    "Seasonal inflation heatmap",
    "Which calendar months carry Bangladesh's recurring price pressure for this commodity?",
)
matrix = inflation_heatmap_matrix(prices)
st.plotly_chart(
    charts.inflation_heatmap(matrix, mode, palette, height=max(240, 52 * max(len(matrix), 1))),
    width="stretch",
    config=plotly_config(f"{ctx.product}_inflation_heatmap"),
)

if not matrix.empty:
    monthly_profile = matrix.mean(axis=0).dropna()
    if not monthly_profile.empty:
        names = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                 "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
        hottest = names[int(monthly_profile.idxmax()) - 1]
        coolest = names[int(monthly_profile.idxmin()) - 1]
        st.caption(
            f"Averaged across all years, **{hottest}** is the strongest inflation month for "
            f"{meta.label} ({monthly_profile.max():+.2f}%) and **{coolest}** the weakest "
            f"({monthly_profile.min():+.2f}%). Use this to time procurement and buffer-stock "
            "releases rather than reacting after the fact."
        )

footer(
    "Inflation is computed as the percentage change in the cleaned daily retail price over the "
    "stated period; the 30-day measure is the closest daily analogue to official monthly figures.",
    f"{meta.label} · {ctx.model} · {filters.horizon}-day horizon",
)
