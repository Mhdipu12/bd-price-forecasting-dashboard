"""Model comparison — which forecaster earns the right to be trusted."""

from __future__ import annotations

import streamlit as st

from dashboard import charts
from dashboard.components import (
    InsightCard,
    empty_state,
    footer,
    insight_card,
    page_header,
    section,
)
from dashboard.config import COMMODITY_BY_KEY, MODEL_DESCRIPTIONS, MODEL_ORDER
from dashboard.context import get_context
from dashboard.theme import plotly_config
from dashboard.transforms import error_summary

ctx = get_context()
bundle, filters, palette, mode = ctx.bundle, ctx.filters, ctx.palette, ctx.mode
meta = COMMODITY_BY_KEY[ctx.product]

page_header(
    f"{meta.label} — model comparison",
    "All models are scored on the same held-out test set using the same one-step-ahead "
    "protocol, so the comparison is like for like.",
)

metrics = bundle.metrics[bundle.metrics["commodity"] == ctx.product].copy()
if metrics.empty:
    empty_state(
        "No model metrics were exported for this commodity.",
        "Check that `metrics.csv` contains rows for it.",
    )
    st.stop()

metrics["_order"] = metrics["model"].map({m: i for i, m in enumerate(MODEL_ORDER)}).fillna(99)
metrics = metrics.sort_values("_order").drop(columns="_order")
best_model = bundle.best_model(ctx.product)

scored = metrics[metrics["model"] != "Seasonal Naive"].sort_values("MAE")
best_row = scored.iloc[0] if not scored.empty else metrics.iloc[0]
runner_up = scored.iloc[1] if len(scored) > 1 else None
baseline = metrics[metrics["model"] == "Seasonal Naive"]

# --------------------------------------------------------------------------
# Winner callout
# --------------------------------------------------------------------------
margin_text = ""
if runner_up is not None and runner_up["MAE"]:
    margin = (runner_up["MAE"] - best_row["MAE"]) / runner_up["MAE"] * 100
    margin_text = (
        f" That is <b>{margin:.1f}%</b> lower MAE than the runner-up "
        f"({runner_up['model']}, {runner_up['MAE']:.3f})."
    )

skill_text = ""
if not baseline.empty and float(baseline.iloc[0]["MAE"]):
    skill = (1 - best_row["MAE"] / float(baseline.iloc[0]["MAE"])) * 100
    verdict = "beats" if skill > 0 else "fails to beat"
    skill_text = (
        f" Against the seasonal-naive baseline it {verdict} doing nothing by "
        f"<b>{abs(skill):.1f}%</b>."
    )

insight_card(
    InsightCard(
        title=f"Best model for {meta.label}: {best_row['model']}",
        body=(
            f"MAE <b>{best_row['MAE']:.3f}</b> · RMSE <b>{best_row['RMSE']:.3f}</b> · "
            f"MAPE <b>{best_row['MAPE']:.2f}%</b>.{margin_text}{skill_text} "
            f"{MODEL_DESCRIPTIONS.get(str(best_row['model']), '')}"
        ),
        icon="award",
        tone="success",
    ),
    palette,
)

# --------------------------------------------------------------------------
# Metrics table
# --------------------------------------------------------------------------
section("Accuracy metrics", "Lower is better on all three; the winner is highlighted.")

display = metrics.copy()
display["Best"] = display["model"] == best_model
columns = ["model", "MAE", "RMSE", "MAPE", "Best"]
if "skill_vs_naive" in display.columns:
    columns.insert(4, "skill_vs_naive")
display = display[columns].rename(
    columns={"model": "Model", "skill_vs_naive": "Skill vs naive"}
)

st.dataframe(
    display,
    hide_index=True,
    width="stretch",
    column_config={
        "Model": st.column_config.TextColumn("Model", width="medium"),
        "MAE": st.column_config.NumberColumn("MAE (BDT)", format="%.4f"),
        "RMSE": st.column_config.NumberColumn("RMSE (BDT)", format="%.4f"),
        "MAPE": st.column_config.NumberColumn("MAPE", format="%.2f%%"),
        "Skill vs naive": st.column_config.NumberColumn("Skill vs naive", format="%+.1f%%"),
        "Best": st.column_config.CheckboxColumn("Best", width="small"),
    },
)

cols = st.columns(3, gap="small")
for col, metric in zip(cols, ["MAE", "RMSE", "MAPE"]):
    with col:
        st.plotly_chart(
            charts.metric_bars(metrics, metric, best_model, mode, palette, height=300),
            width="stretch",
            config=plotly_config(f"{ctx.product}_{metric.lower()}"),
        )

# --------------------------------------------------------------------------
# Actual vs predicted
# --------------------------------------------------------------------------
section(
    "Actual versus predicted",
    "Where does each model track reality on the held-out test set, and where does it drift?",
)
models_in_comparison = [
    m for m in MODEL_ORDER
    if m in set(bundle.comparison.loc[bundle.comparison["commodity"] == ctx.product, "model"])
]
chosen = st.multiselect(
    "Models to overlay",
    options=models_in_comparison,
    default=[m for m in models_in_comparison if m != "Seasonal Naive"],
    key="mc_models",
)
st.plotly_chart(
    charts.actual_vs_predicted(bundle.comparison, ctx.product, chosen, mode, palette, height=420),
    width="stretch",
    config=plotly_config(f"{ctx.product}_actual_vs_predicted"),
)

# --------------------------------------------------------------------------
# Error comparison
# --------------------------------------------------------------------------
section(
    "Error comparison",
    "Accuracy on average is not enough — a model must also be consistently accurate.",
)
left, right = st.columns([1.25, 1], gap="medium")
with left:
    st.plotly_chart(
        charts.error_distribution(
            bundle.comparison, ctx.product, chosen or models_in_comparison, mode, palette, height=380
        ),
        width="stretch",
        config=plotly_config(f"{ctx.product}_error_distribution"),
    )
with right:
    errors = error_summary(bundle.comparison, ctx.product)
    if errors.empty:
        empty_state("No back-test predictions were exported.")
    else:
        errors = errors.rename(columns={"model": "Model"})
        st.dataframe(
            errors,
            hide_index=True,
            width="stretch",
            height=380,
            column_config={
                "Bias": st.column_config.NumberColumn("Bias", format="%+.3f"),
                "MAE": st.column_config.NumberColumn("MAE", format="%.3f"),
                "Max error": st.column_config.NumberColumn("Max error", format="%.3f"),
                "Error sd": st.column_config.NumberColumn("Error sd", format="%.3f"),
            },
        )
        st.caption(
            "**Bias** is the mean error: a positive value means the model systematically "
            "under-forecasts. **Max error** is tail risk — for a policy audience that number "
            "often matters more than the average."
        )

# --------------------------------------------------------------------------
# Feature importance
# --------------------------------------------------------------------------
section(
    "XGBoost feature importance",
    "Which engineered signals actually drive the machine learning forecast?",
)
st.plotly_chart(
    charts.feature_importance_chart(bundle.feature_importance, ctx.product, mode, palette),
    width="stretch",
    config=plotly_config(f"{ctx.product}_feature_importance"),
)

with st.expander("What each model brings to the study"):
    for model in MODEL_ORDER:
        if model in set(metrics["model"]):
            st.markdown(f"**{model}** — {MODEL_DESCRIPTIONS.get(model, '')}")

footer(
    "Metrics are one-step-ahead errors on the final 180 days, which no model saw during training.",
    f"Winner for {meta.label}: {best_model}",
)
