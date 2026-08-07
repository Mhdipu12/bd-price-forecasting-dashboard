"""Inflation analysis — realised and projected rates at every time scale."""

from __future__ import annotations

import streamlit as st

from dashboard import charts
from dashboard.components import KpiCard, empty_state, footer, kpi_row, page_header, section
from dashboard.config import COMMODITY_BY_KEY, SETTINGS, format_date
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
    highest = lowest = average = volatility = float("nan")
    highest_date = lowest_date = None
else:
    highest = float(monthly_series.max())
    lowest = float(monthly_series.min())
    average = float(monthly_series.mean())
    volatility = float(monthly_series.std())
    highest_date = inflation.loc[monthly_series.idxmax(), "date"] if not inflation.empty else None
    lowest_date = inflation.loc[monthly_series.idxmin(), "date"] if not inflation.empty else None

kpi_row(
    [
        KpiCard(
            label="Highest inflation",
            value=f"{highest:+.2f}",
            unit="% / 30d",
            delta=None,
            footnote=f"peak on {format_date(highest_date)}",
            tooltip="Largest realised 30-day price increase in the sample.",
            icon="alert",
            tone="danger",
        ),
        KpiCard(
            label="Lowest inflation",
            value=f"{lowest:+.2f}",
            unit="% / 30d",
            delta=None,
            footnote=f"trough on {format_date(lowest_date)}",
            tooltip="Largest realised 30-day price decline in the sample.",
            icon="trend_down",
            tone="success",
        ),
        KpiCard(
            label="Average inflation",
            value=f"{average:+.2f}",
            unit="% / 30d",
            delta=snapshot.current_inflation - average,
            footnote="sample mean · delta shows today versus normal",
            tooltip="Mean realised 30-day inflation across the whole sample.",
            icon="percent",
            tone="primary",
            spark=monthly_series.tail(SETTINGS.sparkline_days).tolist(),
        ),
        KpiCard(
            label="Inflation volatility",
            value=f"{volatility:,.2f}",
            unit="pp",
            delta=None,
            footnote="standard deviation of the 30-day rate",
            tooltip=(
                "How much the inflation rate itself swings. High values mean price shocks "
                "arrive suddenly rather than building gradually."
            ),
            icon="activity",
            tone="warning",
        ),
        KpiCard(
            label="Forecast inflation",
            value=f"{snapshot.forecast_inflation:+.2f}",
            unit="% / 30d",
            delta=snapshot.forecast_inflation - snapshot.current_inflation,
            footnote=f"at {format_date(snapshot.forecast_date)}",
            tooltip=(
                "Forward 30-day rate implied by the forecast path. The delta compares it "
                "with the realised rate today."
            ),
            icon="target",
            tone="danger"
            if snapshot.forecast_inflation > SETTINGS.thresholds.inflation_alert
            else "primary",
            spark=((path["forecast"] / snapshot.current_price - 1) * 100).tolist()
            if not path.empty
            else [],
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
