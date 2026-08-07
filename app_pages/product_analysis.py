"""Product analysis — the diagnostic view of a single commodity's history."""

from __future__ import annotations

import streamlit as st

from dashboard import charts
from dashboard.components import KpiCard, empty_state, footer, kpi_row, page_header, section
from dashboard.config import COMMODITY_BY_KEY, format_date
from dashboard.context import get_context
from dashboard.theme import plotly_config
from dashboard.transforms import (
    commodity_inflation,
    commodity_prices,
    descriptive_statistics,
    periodic_inflation,
    realised_volatility,
    resample_prices,
)

ctx = get_context()
bundle, filters, palette, mode = ctx.bundle, ctx.filters, ctx.palette, ctx.mode
meta = COMMODITY_BY_KEY[ctx.product]

page_header(
    f"{meta.label} — historical analysis",
    "Every chart below answers one question about how this market has behaved. "
    "Forecasts are on the Price forecast page.",
)

prices = commodity_prices(bundle, ctx.product)
inflation = commodity_inflation(bundle, ctx.product)

if prices.empty:
    empty_state(
        "No price history for this commodity.",
        "Re-run the export so `historical_price.csv` contains this series.",
    )
    st.stop()

# --------------------------------------------------------------------------
# Descriptive KPIs
# --------------------------------------------------------------------------
series = prices["price"]
weekly_prices = resample_prices(prices, "W")
monthly_prices = resample_prices(prices, "MS")
weekly_inflation = periodic_inflation(prices, "W")
monthly_inflation = periodic_inflation(prices, "MS")

total_change = (series.iloc[-1] / series.iloc[0] - 1) * 100
years = max((prices["date"].iloc[-1] - prices["date"].iloc[0]).days / 365.25, 1e-9)
annualised = ((series.iloc[-1] / series.iloc[0]) ** (1 / years) - 1) * 100

kpi_row(
    [
        KpiCard(
            label="Sample period",
            value=f"{len(series):,}",
            unit="days",
            delta=None,
            footnote=f"{format_date(prices['date'].iloc[0])} → {format_date(prices['date'].iloc[-1])}",
            tooltip="Number of daily observations after cleaning and calendar reindexing.",
            icon="calendar",
            tone="primary",
        ),
        KpiCard(
            label="Average price",
            value=f"{series.mean():,.2f}",
            unit=meta.unit,
            delta=(series.iloc[-1] / series.mean() - 1) * 100,
            footnote="today versus the sample mean",
            tooltip="Mean retail price across the whole cleaned sample.",
            icon="price",
            tone="primary",
            spark=series.tail(90).tolist(),
        ),
        KpiCard(
            label="Price range",
            value=f"{series.min():,.0f}–{series.max():,.0f}",
            unit=meta.unit,
            delta=(series.max() / series.min() - 1) * 100,
            footnote="peak versus trough over the sample",
            tooltip="Lowest and highest observed price, and the peak-to-trough spread.",
            icon="chart",
            tone="warning",
        ),
        KpiCard(
            label="Total change",
            value=f"{total_change:+.1f}",
            unit="%",
            delta=annualised,
            delta_suffix="% / yr",
            footnote="whole sample · annualised delta",
            tooltip="Cumulative change from the first to the last observation.",
            icon="trend_up" if total_change > 0 else "trend_down",
            tone="danger" if total_change > 0 else "success",
        ),
        KpiCard(
            label="Volatility (30d)",
            value=f"{realised_volatility(prices):,.2f}",
            unit="%",
            delta=None,
            footnote="standard deviation of daily returns",
            tooltip="Trailing 30-day standard deviation of daily percentage changes.",
            icon="activity",
            tone="warning",
        ),
    ],
    palette,
)

# --------------------------------------------------------------------------
# Trend
# --------------------------------------------------------------------------
section("Historical price trend", "What is the long-run direction beneath the daily noise?")
st.plotly_chart(
    charts.price_trend(prices, ctx.product, mode, palette, height=380),
    width="stretch",
    config=plotly_config(f"{ctx.product}_price_trend"),
)

section("Historical inflation trend", "How persistent has month-on-month price growth been?")
st.plotly_chart(
    charts.inflation_trend(
        inflation, "inflation_monthly", "30-day inflation", mode, palette, height=320
    ),
    width="stretch",
    config=plotly_config(f"{ctx.product}_inflation_trend"),
)

# --------------------------------------------------------------------------
# Weekly and monthly views
# --------------------------------------------------------------------------
section(
    "Weekly and monthly price levels",
    "Once daily noise is averaged away, which direction is the market really taking?",
)
left, right = st.columns(2, gap="medium")
with left:
    st.markdown("**Weekly average price** — last 52 weeks, shaded by the week's low–high range")
    st.plotly_chart(
        charts.periodic_price_bars(weekly_prices, ctx.product, "Weekly", mode, palette, tail=52),
        width="stretch",
        config=plotly_config(f"{ctx.product}_weekly_price"),
    )
with right:
    st.markdown("**Monthly average price** — last 36 months, shaded by the month's low–high range")
    st.plotly_chart(
        charts.periodic_price_bars(monthly_prices, ctx.product, "Monthly", mode, palette, tail=36),
        width="stretch",
        config=plotly_config(f"{ctx.product}_monthly_price"),
    )

section(
    "Weekly and monthly inflation",
    "Which periods actually pushed prices up, and by how much?",
)
left, right = st.columns(2, gap="medium")
with left:
    st.markdown("**Week-on-week inflation** — short-run shocks and their reversals")
    st.plotly_chart(
        charts.periodic_inflation_bars(weekly_inflation, "Weekly", mode, palette, tail=52),
        width="stretch",
        config=plotly_config(f"{ctx.product}_weekly_inflation"),
    )
with right:
    st.markdown("**Month-on-month inflation** — the measure closest to official BBS practice")
    st.plotly_chart(
        charts.periodic_inflation_bars(monthly_inflation, "Monthly", mode, palette, tail=36),
        width="stretch",
        config=plotly_config(f"{ctx.product}_monthly_inflation"),
    )

# --------------------------------------------------------------------------
# Distribution and statistics
# --------------------------------------------------------------------------
section(
    "Distribution and descriptive statistics",
    "At which price levels has this market spent most of its time?",
)
left, right = st.columns([1.35, 1], gap="medium")
with left:
    st.plotly_chart(
        charts.price_distribution(prices, ctx.product, mode, palette, height=380),
        width="stretch",
        config=plotly_config(f"{ctx.product}_distribution"),
    )
with right:
    st.dataframe(
        descriptive_statistics(prices, meta.unit),
        hide_index=True,
        width="stretch",
        height=380,
        column_config={
            "Statistic": st.column_config.TextColumn("Statistic", width="medium"),
            "Value": st.column_config.TextColumn("Value", width="small"),
        },
    )

footer(
    f"{meta.label} · {len(series):,} cleaned daily observations · "
    f"{format_date(prices['date'].iloc[0])} to {format_date(prices['date'].iloc[-1])}",
    "Outliers winsorised with a rolling median/MAD rule; gaps interpolated on the time index.",
)
