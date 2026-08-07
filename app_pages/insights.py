"""Insights & recommendations — the decision layer of the dashboard."""

from __future__ import annotations

import streamlit as st

from dashboard.components import (
    InsightCard,
    footer,
    insight_card,
    insight_grid,
    page_header,
    section,
)
from dashboard.config import COMMODITY_BY_KEY, format_date
from dashboard.context import get_context
from dashboard.insights import (
    build_insight_cards,
    build_recommendations,
    classify_regime,
    market_headline,
    risk_rankings,
)

ctx = get_context()
bundle, filters, palette, mode = ctx.bundle, ctx.filters, ctx.palette, ctx.mode
meta = COMMODITY_BY_KEY[ctx.product]

page_header(
    "Insights & recommendations",
    f"Generated from the {ctx.model} forecast for {meta.label} at "
    f"{format_date(ctx.snapshot().forecast_date)} — every sentence updates with the sidebar filters.",
)

snapshot = ctx.snapshot()
comparison = ctx.comparison_table()
regime = classify_regime(snapshot)
riskiest, steadiest = risk_rankings(comparison)

# --------------------------------------------------------------------------
# Basket headline
# --------------------------------------------------------------------------
insight_card(
    InsightCard(
        title=f"Basket outlook — next {filters.horizon} days",
        body=market_headline(bundle, comparison, filters.horizon),
        icon="compass",
        tone="primary",
    ),
    palette,
)

# --------------------------------------------------------------------------
# Six headline insights
# --------------------------------------------------------------------------
section("Market insights", f"{meta.label}, read against the whole essential-goods basket.")
insight_grid(build_insight_cards(bundle, snapshot, comparison, filters.horizon), palette, columns=3)

# --------------------------------------------------------------------------
# Recommendations
# --------------------------------------------------------------------------
section(
    "Recommendations",
    "Actions implied by the forecast, separated by who has to take them.",
)

policy_tab, consumer_tab = st.tabs(["Policy makers", "Consumers"])

with policy_tab:
    st.markdown(
        f"**Context.** {meta.label} is in a **{regime.replace('_', ' ')}** regime with a "
        f"**{snapshot.risk_level.lower()}** risk rating. The projected move is "
        f"**{snapshot.expected_change_pct:+.2f}%** over {filters.horizon} days, with a 95% "
        f"interval of {snapshot.lower:,.2f}–{snapshot.upper:,.2f} {meta.unit}."
    )
    insight_grid(
        build_recommendations(bundle, snapshot, filters.horizon, "policy"), palette, columns=3
    )

with consumer_tab:
    st.markdown(
        f"**Context.** Buying {meta.label} today costs **{snapshot.current_price:,.2f} "
        f"{meta.unit}**. Buying on {format_date(snapshot.forecast_date)} is projected to cost "
        f"**{snapshot.forecast_price:,.2f} {meta.unit}**."
    )
    insight_grid(
        build_recommendations(bundle, snapshot, filters.horizon, "consumer"), palette, columns=3
    )

# --------------------------------------------------------------------------
# Basket-wide risk table
# --------------------------------------------------------------------------
section("Risk ranking across the basket", "Where should attention and budget go first?")

table = comparison.copy()
table["Commodity"] = [COMMODITY_BY_KEY[c].label for c in table["commodity"]]
table["Risk score"] = table["Volatility"].fillna(0) + table["Expected change"].abs().fillna(0) / 2
table = table.sort_values("Risk score", ascending=False)[
    ["Commodity", "model", "Expected change", "Volatility", "Risk score", "Risk"]
].rename(columns={"model": "Model"})

st.dataframe(
    table,
    hide_index=True,
    width="stretch",
    column_config={
        "Expected change": st.column_config.NumberColumn("Expected change", format="%+.2f%%"),
        "Volatility": st.column_config.NumberColumn("Volatility (30d)", format="%.2f%%"),
        "Risk score": st.column_config.ProgressColumn(
            "Risk score",
            format="%.2f",
            min_value=0.0,
            max_value=float(max(table["Risk score"].max(), 1.0)),
        ),
    },
)
st.caption(
    "Risk score blends realised 30-day volatility with half the absolute projected move, so a "
    "quiet commodity that is about to jump still ranks as risky."
)

footer(
    "Recommendations are rule-based readings of the exported forecast, not financial advice. "
    "They are intended to frame a decision, not to replace market judgement.",
    f"Highest risk: {COMMODITY_BY_KEY[riskiest].label if riskiest else '—'} · "
    f"Most stable: {COMMODITY_BY_KEY[steadiest].label if steadiest else '—'}",
)
