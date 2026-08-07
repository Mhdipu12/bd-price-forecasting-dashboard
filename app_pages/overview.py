"""Overview — the always-on situation report for the selected commodity."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from dashboard import charts
from dashboard.components import (
    KpiCard,
    StatusCard,
    footer,
    kpi_row,
    page_header,
    section,
    status_strip,
)
from dashboard.config import COMMODITY_BY_KEY, SETTINGS, format_date
from dashboard.context import get_context
from dashboard.insights import market_headline, risk_rankings
from dashboard.theme import plotly_config
from dashboard.transforms import commodity_forecast, commodity_inflation, commodity_prices

ctx = get_context()
bundle, filters, palette, mode = ctx.bundle, ctx.filters, ctx.palette, ctx.mode
meta = COMMODITY_BY_KEY[ctx.product]

page_header(
    "Market overview",
    f"{meta.label} · {filters.horizon}-day outlook · positions as of "
    f"{format_date(bundle.last_observation)}",
)

# --------------------------------------------------------------------------
# Dashboard status strip — always visible, driven by dataset dates only
# --------------------------------------------------------------------------
snapshot = ctx.snapshot()
comparison = ctx.comparison_table()
riskiest, steadiest = risk_rankings(comparison)

risk_tone = {"Low": "success", "Moderate": "warning", "High": "danger"}[snapshot.risk_level]

status_strip(
    [
        StatusCard(
            "Last dataset update",
            format_date(bundle.last_observation),
            f"{len(bundle.prices):,} daily observations",
            icon="database",
            tone="primary",
        ),
        StatusCard(
            "Forecast window",
            f"{format_date(bundle.forecast_start)} → "
            f"{format_date(bundle.forecast_start + pd.Timedelta(days=filters.horizon - 1))}",
            f"full export runs to {format_date(bundle.forecast_end)}",
            icon="calendar",
            tone="primary",
        ),
        StatusCard(
            "Forecast horizon",
            f"{filters.horizon} days",
            f"selected date: {format_date(snapshot.forecast_date)}",
            icon="clock",
            tone="primary",
        ),
        StatusCard(
            "Best model",
            bundle.best_model(ctx.product),
            "lowest test-set MAE" + (" · auto" if filters.model_is_auto else " · manual override"),
            icon="award",
            tone="success",
        ),
        StatusCard(
            "Market risk",
            snapshot.risk_level,
            f"volatility {snapshot.volatility:,.2f}% · move {snapshot.expected_change_pct:+.2f}%",
            icon="shield",
            tone=risk_tone,  # type: ignore[arg-type]
        ),
    ],
    palette,
)

# --------------------------------------------------------------------------
# Product KPI cards
# --------------------------------------------------------------------------
prices = commodity_prices(bundle, ctx.product)
inflation = commodity_inflation(bundle, ctx.product)
path = commodity_forecast(bundle, ctx.product, ctx.model, filters.horizon)

recent_change = float("nan")
if len(prices) > 7:
    recent_change = (prices["price"].iloc[-1] / prices["price"].iloc[-8] - 1) * 100

inflation_tail = (
    inflation["inflation_monthly"].dropna().tail(SETTINGS.sparkline_days).tolist()
    if "inflation_monthly" in inflation.columns
    else []
)
forward_inflation_path = (
    ((path["forecast"] / snapshot.current_price - 1) * 100).tolist() if not path.empty else []
)

kpi_row(
    [
        KpiCard(
            label="Current price",
            value=f"{snapshot.current_price:,.2f}",
            unit=meta.unit,
            delta=recent_change,
            footnote=f"last observation · {format_date(snapshot.as_of)}",
            tooltip=(
                f"Closing retail price of {meta.label} on {format_date(snapshot.as_of)}. "
                "Delta compares with seven days earlier."
            ),
            icon="price",
            tone="primary",
            spark=snapshot.sparkline,
        ),
        KpiCard(
            label="Forecast price",
            value=f"{snapshot.forecast_price:,.2f}",
            unit=meta.unit,
            delta=snapshot.expected_change_pct,
            footnote=f"{snapshot.model} · {format_date(snapshot.forecast_date)}",
            tooltip=(
                f"{snapshot.model} central forecast for {format_date(snapshot.forecast_date)}. "
                f"95% interval {snapshot.lower:,.2f}–{snapshot.upper:,.2f} {meta.unit}."
            ),
            icon="target",
            tone="warning" if snapshot.expected_change_pct > 0 else "success",
            spark=snapshot.forecast_sparkline,
        ),
        KpiCard(
            label="Current inflation",
            value=f"{snapshot.current_inflation:+.2f}",
            unit="% / 30d",
            delta=snapshot.current_inflation,
            footnote="realised 30-day rate",
            tooltip="Percentage change in price over the trailing 30 days, as observed.",
            icon="percent",
            tone="primary",
            spark=inflation_tail,
        ),
        KpiCard(
            label="Forecast inflation",
            value=f"{snapshot.forecast_inflation:+.2f}",
            unit="% / 30d",
            delta=snapshot.forecast_inflation - snapshot.current_inflation,
            footnote="forward 30-day rate at selected date",
            tooltip=(
                "Forward 30-day inflation implied by the forecast path, scaled from the "
                f"{snapshot.horizon_days}-day horizon move. Delta shows acceleration versus "
                "the realised rate."
            ),
            icon="activity",
            tone="danger"
            if snapshot.forecast_inflation > SETTINGS.thresholds.inflation_alert
            else "primary",
            spark=forward_inflation_path,
        ),
        KpiCard(
            label="Expected price change",
            value=f"{snapshot.expected_change_pct:+.2f}",
            unit="%",
            delta=snapshot.expected_change_pct,
            footnote=f"over {snapshot.horizon_days} days",
            tooltip=(
                f"Move from {snapshot.current_price:,.2f} to {snapshot.forecast_price:,.2f} "
                f"{meta.unit} between {format_date(snapshot.as_of)} and "
                f"{format_date(snapshot.forecast_date)}."
            ),
            icon="trend_up" if snapshot.direction == "up" else
                 "trend_down" if snapshot.direction == "down" else "trend_flat",
            tone="danger" if snapshot.direction == "up" else
                 "success" if snapshot.direction == "down" else "neutral",
            spark=snapshot.forecast_sparkline,
        ),
    ],
    palette,
)

# --------------------------------------------------------------------------
# Price and inflation trajectory
# --------------------------------------------------------------------------
section(
    "Price trajectory",
    "Where has the price been over the past year, and where does the model take it next?",
)
st.plotly_chart(
    charts.price_history_vs_forecast(
        prices,
        path,
        ctx.product,
        ctx.model,
        mode,
        palette,
        lookback_days=365,
        marker_date=snapshot.forecast_date,
        height=410,
    ),
    width="stretch",
    config=plotly_config(f"{ctx.product}_price_vs_forecast"),
)

section(
    "Inflation trajectory",
    "Is the pace of price growth accelerating or cooling relative to the recent past?",
)
st.plotly_chart(
    charts.inflation_history_vs_forecast(
        inflation, path, snapshot.current_price, ctx.product, mode, palette, height=360
    ),
    width="stretch",
    config=plotly_config(f"{ctx.product}_inflation_vs_forecast"),
)

# --------------------------------------------------------------------------
# Cross-commodity comparison
# --------------------------------------------------------------------------
section("Commodity comparison", "How does the whole essential-goods basket look side by side?")
st.html(f'<div class="bdp-insight-body" style="margin-bottom:.6rem">'
        f"{market_headline(bundle, comparison, filters.horizon)}</div>")

left, right = st.columns(2, gap="medium")
with left:
    st.plotly_chart(
        charts.commodity_price_comparison(comparison, mode, palette),
        width="stretch",
        config=plotly_config("commodity_price_comparison"),
    )
with right:
    st.plotly_chart(
        charts.commodity_inflation_comparison(comparison, mode, palette),
        width="stretch",
        config=plotly_config("commodity_inflation_comparison"),
    )

display = comparison.copy()
display["Commodity"] = [COMMODITY_BY_KEY[c].label for c in display["commodity"]]
display = display[
    [
        "Commodity",
        "model",
        "Current price",
        "Forecast price",
        "Expected change",
        "Current inflation",
        "Forecast inflation",
        "Volatility",
        "Risk",
    ]
].rename(columns={"model": "Model"})

st.dataframe(
    display,
    hide_index=True,
    width="stretch",
    column_config={
        "Current price": st.column_config.NumberColumn("Current price", format="%.2f"),
        "Forecast price": st.column_config.NumberColumn("Forecast price", format="%.2f"),
        "Expected change": st.column_config.NumberColumn("Expected change", format="%+.2f%%"),
        "Current inflation": st.column_config.NumberColumn("Current inflation", format="%+.2f%%"),
        "Forecast inflation": st.column_config.NumberColumn("Forecast inflation", format="%+.2f%%"),
        "Volatility": st.column_config.NumberColumn("Volatility (30d)", format="%.2f%%"),
    },
)

footer(
    f"Historical KPIs read the last observation ({format_date(bundle.last_observation)}); "
    f"forecast KPIs read {format_date(snapshot.forecast_date)}.",
    f"Highest risk: {COMMODITY_BY_KEY[riskiest].label if riskiest else '—'} · "
    f"Most stable: {COMMODITY_BY_KEY[steadiest].label if steadiest else '—'}",
)
