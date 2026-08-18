"""Price forecast — the interactive forward view and the downloadable path."""

from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd
import streamlit as st

from dashboard import charts
from dashboard.components import KpiCard, empty_state, footer, kpi_row, page_header, section
from dashboard.config import COMMODITY_BY_KEY, SETTINGS, format_date
from dashboard.context import get_context
from dashboard.theme import plotly_config
from dashboard.transforms import (
    commodity_forecast,
    commodity_prices,
    forecast_row,
    forecast_table,
    price_row,
)

ctx = get_context()
bundle, filters, palette, mode = ctx.bundle, ctx.filters, ctx.palette, ctx.mode
meta = COMMODITY_BY_KEY[ctx.product]

page_header(
    f"{meta.label} — price forecast",
    f"{ctx.model} · {filters.horizon}-day horizon · forecast window "
    f"{format_date(bundle.forecast_start)} → "
    f"{format_date(bundle.forecast_start + pd.Timedelta(days=filters.horizon - 1))}",
)

prices = commodity_prices(bundle, ctx.product)
path = commodity_forecast(bundle, ctx.product, ctx.model, filters.horizon)
snapshot = ctx.snapshot()

if path.empty:
    empty_state(
        "No forecast path is available for this commodity and model.",
        "Check that `forecast.csv` contains rows for this combination.",
    )
    st.stop()

# --------------------------------------------------------------------------
# Price on a specific date
# --------------------------------------------------------------------------
KEY_LOOKUP_DATE = "price_lookup_date"

lookup_min = prices["date"].min().date()
lookup_max = path["date"].max().date()
_lookup_default = st.session_state.get(KEY_LOOKUP_DATE, snapshot.as_of.date())
if not isinstance(_lookup_default, date):
    _lookup_default = snapshot.as_of.date()
st.session_state[KEY_LOOKUP_DATE] = min(max(_lookup_default, lookup_min), lookup_max)

section(
    "Price on a specific date",
    "Pick any day, past or future, to read its recorded or forecast price.",
)
left, right = st.columns([1, 2], gap="large")
with left:
    picked = st.date_input(
        "Date",
        min_value=lookup_min,
        max_value=lookup_max,
        key=KEY_LOOKUP_DATE,
        format="DD-MM-YYYY",
        help="Dates on or before today return the recorded price; later dates "
        "return this model's forecast.",
    )
    st.caption(f"Range: {format_date(pd.Timestamp(lookup_min))} → {format_date(pd.Timestamp(lookup_max))}")

lookup_target = pd.Timestamp(picked)
if lookup_target <= snapshot.as_of:
    lookup_source = price_row(prices, lookup_target)
    lookup_date_resolved = pd.Timestamp(lookup_source["date"])
    lookup_value = float(lookup_source["price"])
    lookup_card = KpiCard(
        label="Price on selected date",
        value=f"{lookup_value:,.2f}",
        unit=meta.unit,
        delta=(lookup_value / snapshot.current_price - 1) * 100,
        footnote=f"Actual · {format_date(lookup_date_resolved)}",
        tooltip=f"Recorded price on {format_date(lookup_date_resolved)}.",
        icon="clock",
        tone="neutral",
    )
else:
    lookup_source = forecast_row(path, lookup_target)
    lookup_date_resolved = pd.Timestamp(lookup_source["date"])
    lookup_value = float(lookup_source["forecast"])
    lookup_lower = float(lookup_source["lower"])
    lookup_upper = float(lookup_source["upper"])
    lookup_card = KpiCard(
        label="Price on selected date",
        value=f"{lookup_value:,.2f}",
        unit=meta.unit,
        delta=(lookup_value / snapshot.current_price - 1) * 100,
        footnote=(
            f"{ctx.model} forecast · {format_date(lookup_date_resolved)} · "
            f"95% CI {lookup_lower:,.2f}–{lookup_upper:,.2f}"
        ),
        tooltip=f"{ctx.model} forecast for {format_date(lookup_date_resolved)}.",
        icon="target",
        tone="primary",
    )
with right:
    kpi_row([lookup_card], palette)

# --------------------------------------------------------------------------
# Forecast KPIs
# --------------------------------------------------------------------------
peak_idx = int(path["forecast"].idxmax())
trough_idx = int(path["forecast"].idxmin())

kpi_row(
    [
        KpiCard(
            label="Forecast at selected date",
            value=f"{snapshot.forecast_price:,.2f}",
            unit=meta.unit,
            delta=snapshot.expected_change_pct,
            footnote=format_date(snapshot.forecast_date),
            tooltip=f"{ctx.model} central forecast for {format_date(snapshot.forecast_date)}.",
            icon="target",
            tone="primary",
            spark=path["forecast"].tolist(),
        ),
        KpiCard(
            label="Average over horizon",
            value=f"{path['forecast'].mean():,.2f}",
            unit=meta.unit,
            delta=(path["forecast"].mean() / snapshot.current_price - 1) * 100,
            footnote=f"mean of all {len(path)} forecast days",
            tooltip=(
                "Average of the whole forecast path. Less sensitive to a single day than "
                "the horizon-end number, and the better basis for procurement planning."
            ),
            icon="chart",
            tone="warning",
        ),
        KpiCard(
            label="Projected peak",
            value=f"{path['forecast'].iloc[peak_idx]:,.2f}",
            unit=meta.unit,
            delta=(path["forecast"].iloc[peak_idx] / snapshot.current_price - 1) * 100,
            footnote=format_date(path["date"].iloc[peak_idx]),
            tooltip="Highest point of the forecast path within the selected horizon.",
            icon="alert",
            tone="danger",
        ),
        KpiCard(
            label="Projected trough",
            value=f"{path['forecast'].iloc[trough_idx]:,.2f}",
            unit=meta.unit,
            delta=(path["forecast"].iloc[trough_idx] / snapshot.current_price - 1) * 100,
            footnote=format_date(path["date"].iloc[trough_idx]),
            tooltip="Lowest point of the forecast path within the selected horizon.",
            icon="trend_down",
            tone="success",
        ),
    ],
    palette,
)

# --------------------------------------------------------------------------
# Interactive chart
# --------------------------------------------------------------------------
section(
    "Historical versus forecast price",
    "Hover for values, drag to pan, use the range buttons or the slider to zoom.",
)
figure = charts.price_history_vs_forecast(
    prices,
    path,
    ctx.product,
    ctx.model,
    mode,
    palette,
    lookback_days=0,
    marker_date=snapshot.forecast_date,
    lookup_date=lookup_date_resolved,
    range_selector=True,
    height=520,
)
st.plotly_chart(
    figure,
    width="stretch",
    config=plotly_config(f"{ctx.product}_{ctx.model}_forecast"),
)
st.caption(
    "Use the camera icon on the chart toolbar to download a 2× PNG. The shaded band is the "
    "95% prediction interval — SARIMA's is analytic, the other models' are residual-based."
)

# --------------------------------------------------------------------------
# Model agreement
# --------------------------------------------------------------------------
available_models = bundle.models_for(ctx.product)
if len(available_models) > 1:
    section("Model agreement", "How much do the competing models disagree about the road ahead?")
    paths = {
        model: commodity_forecast(bundle, ctx.product, model, filters.horizon)
        for model in available_models
    }
    st.plotly_chart(
        charts.forecast_fan(
            paths, prices, ctx.product, bundle.best_model(ctx.product), mode, palette, height=400
        ),
        width="stretch",
        config=plotly_config(f"{ctx.product}_model_agreement"),
    )
    spread = pd.DataFrame(
        {
            "Model": list(paths),
            f"Horizon-end price ({meta.unit})": [
                round(float(p["forecast"].iloc[-1]), 2) if not p.empty else np.nan for p in paths.values()
            ],
        }
    )
    spread["Change vs today (%)"] = (
        (spread[f"Horizon-end price ({meta.unit})"] / snapshot.current_price - 1) * 100
    ).round(2)
    spread["Best model"] = spread["Model"] == bundle.best_model(ctx.product)
    st.dataframe(
        spread,
        hide_index=True,
        width="stretch",
        column_config={
            "Change vs today (%)": st.column_config.NumberColumn(
                "Change vs today", format="%+.2f%%"
            ),
            "Best model": st.column_config.CheckboxColumn("Best on test set"),
        },
    )
    band = float(
        spread[f"Horizon-end price ({meta.unit})"].max()
        - spread[f"Horizon-end price ({meta.unit})"].min()
    )
    st.caption(
        f"Spread between the most and least bullish model at the horizon end: "
        f"**{band:,.2f} {meta.unit}** "
        f"({band / snapshot.current_price * 100:,.2f}% of today's price). A wide spread is "
        "itself a signal — it says the models do not agree, so plan against the interval."
    )

# --------------------------------------------------------------------------
# Forecast table
# --------------------------------------------------------------------------
section("Forecast table", "Search, sort and export the day-by-day path.")

table = forecast_table(path, snapshot.current_price, SETTINGS.currency)
price_col = f"Forecast price ({SETTINGS.currency})"

controls = st.container(horizontal=True, vertical_alignment="bottom")
with controls:
    query = st.text_input(
        "Search",
        placeholder="Filter by date, e.g. 2026-07 or Jul",
        key="forecast_search",
    )
    show_interval = st.toggle("Show 95% interval columns", value=True, key="forecast_interval")

view = table.copy()
if query:
    mask = view.astype(str).apply(lambda col: col.str.contains(query, case=False, na=False))
    view = view[mask.any(axis=1)]
    if view.empty:
        st.warning(f"No forecast rows match “{query}”.", icon=":material/search_off:")

if not show_interval:
    view = view.drop(
        columns=[c for c in view.columns if c.startswith(("Lower 95%", "Upper 95%"))],
        errors="ignore",
    )

st.dataframe(
    view,
    hide_index=True,
    width="stretch",
    height=min(430, 42 + 35 * max(len(view), 1)),
    column_config={
        "Date": st.column_config.DateColumn("Date", format="DD MMM YYYY"),
        "Day": st.column_config.NumberColumn("Horizon day", format="%d"),
        price_col: st.column_config.NumberColumn(price_col, format="%.2f"),
        "Expected change (%)": st.column_config.NumberColumn(
            "Change vs today", format="%+.2f%%"
        ),
        "Day-on-day (%)": st.column_config.NumberColumn("Day-on-day", format="%+.3f%%"),
    },
)

st.download_button(
    "Download forecast table (CSV)",
    data=table.to_csv(index=False).encode("utf-8"),
    file_name=f"forecast_{ctx.product.replace(' ', '_').lower()}_{ctx.model.lower()}_{filters.horizon}d.csv",
    mime="text/csv",
    icon=":material/download:",
    type="primary",
)

footer(
    f"Base price {snapshot.current_price:,.2f} {meta.unit} on {format_date(snapshot.as_of)}; "
    f"all percentage changes are measured against it.",
    f"{len(path)} forecast days · {ctx.model}",
)
