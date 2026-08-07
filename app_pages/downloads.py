"""Downloads — every artifact and report the dashboard can hand over."""

from __future__ import annotations

from datetime import datetime

import pandas as pd
import streamlit as st

from dashboard import charts
from dashboard.components import footer, page_header, section
from dashboard.config import COMMODITY_BY_KEY, SETTINGS, format_date
from dashboard.context import get_context
from dashboard.insights import build_insight_cards, build_recommendations
from dashboard.logging_utils import get_logger
from dashboard.reporting import figures_archive, summary_pdf
from dashboard.transforms import (
    commodity_forecast,
    commodity_inflation,
    commodity_prices,
    forecast_table,
    periodic_inflation,
    resample_prices,
)

logger = get_logger(__name__)

ctx = get_context()
bundle, filters, palette, mode = ctx.bundle, ctx.filters, ctx.palette, ctx.mode
meta = COMMODITY_BY_KEY[ctx.product]
slug = ctx.product.replace(" ", "_").lower()
stamp = datetime.now().strftime("%Y%m%d")

page_header(
    "Downloads",
    f"Exports reflect the current selection: {meta.label} · {ctx.model} · "
    f"{filters.horizon}-day horizon.",
)

snapshot = ctx.snapshot()
prices = commodity_prices(bundle, ctx.product)
inflation = commodity_inflation(bundle, ctx.product)
path = commodity_forecast(bundle, ctx.product, ctx.model, filters.horizon)
comparison = ctx.comparison_table()
metrics = bundle.metrics[bundle.metrics["commodity"] == ctx.product]


def _csv(frame: pd.DataFrame) -> bytes:
    """Encode a frame as UTF-8 CSV bytes."""
    return frame.to_csv(index=False).encode("utf-8")


# --------------------------------------------------------------------------
# Data exports
# --------------------------------------------------------------------------
section("Data exports", "Tabular artifacts, filtered to the current selection where relevant.")

row1 = st.columns(3, gap="small")
with row1[0]:
    with st.container(border=True):
        st.markdown("**Forecast path**")
        st.caption(
            f"{len(path)} days · {ctx.model} · central forecast with the 95% interval and "
            "expected change against the last observation."
        )
        st.download_button(
            "Download forecast CSV",
            data=_csv(forecast_table(path, snapshot.current_price, SETTINGS.currency)),
            file_name=f"forecast_{slug}_{ctx.model.lower()}_{filters.horizon}d_{stamp}.csv",
            mime="text/csv",
            icon=":material/download:",
            width="stretch",
            type="primary",
        )

with row1[1]:
    with st.container(border=True):
        st.markdown("**Accuracy metrics**")
        st.caption(
            "MAE, RMSE and MAPE for every model on the held-out test set, for all commodities."
        )
        st.download_button(
            "Download metrics CSV",
            data=_csv(bundle.metrics),
            file_name=f"metrics_all_commodities_{stamp}.csv",
            mime="text/csv",
            icon=":material/download:",
            width="stretch",
        )

with row1[2]:
    with st.container(border=True):
        st.markdown("**Model comparison**")
        st.caption(
            "Day-by-day actual versus predicted values on the test set — the raw material "
            "behind every accuracy chart."
        )
        st.download_button(
            "Download comparison CSV",
            data=_csv(bundle.comparison),
            file_name=f"model_comparison_{stamp}.csv",
            mime="text/csv",
            icon=":material/download:",
            width="stretch",
        )

row2 = st.columns(3, gap="small")
with row2[0]:
    with st.container(border=True):
        st.markdown("**Cleaned price history**")
        st.caption(f"{len(bundle.prices):,} daily observations across all commodities.")
        st.download_button(
            "Download history CSV",
            data=_csv(bundle.prices),
            file_name=f"historical_price_{stamp}.csv",
            mime="text/csv",
            icon=":material/download:",
            width="stretch",
        )

with row2[1]:
    with st.container(border=True):
        st.markdown("**Inflation series**")
        st.caption("Daily, weekly, monthly and annual inflation for every commodity.")
        st.download_button(
            "Download inflation CSV",
            data=_csv(bundle.inflation),
            file_name=f"historical_inflation_{stamp}.csv",
            mime="text/csv",
            icon=":material/download:",
            width="stretch",
        )

with row2[2]:
    with st.container(border=True):
        st.markdown("**Basket snapshot**")
        st.caption(
            f"Current versus forecast price and inflation for every commodity at "
            f"{format_date(snapshot.forecast_date)}."
        )
        st.download_button(
            "Download snapshot CSV",
            data=_csv(comparison),
            file_name=f"basket_snapshot_{filters.horizon}d_{stamp}.csv",
            mime="text/csv",
            icon=":material/download:",
            width="stretch",
        )

if not bundle.feature_importance.empty:
    with st.container(border=True):
        st.markdown("**XGBoost feature importance**")
        st.caption("Gain importance per feature, per commodity.")
        st.download_button(
            "Download feature importance CSV",
            data=_csv(bundle.feature_importance),
            file_name=f"feature_importance_{stamp}.csv",
            mime="text/csv",
            icon=":material/download:",
        )

# --------------------------------------------------------------------------
# Figures
# --------------------------------------------------------------------------
section("Figure bundle", "Publication-quality versions of the charts for the current selection.")


def _build_figures() -> dict:
    """Assemble the figure set exported for the current selection."""
    return {
        f"01_{slug}_price_vs_forecast": charts.price_history_vs_forecast(
            prices, path, ctx.product, ctx.model, mode, palette, lookback_days=365,
            marker_date=snapshot.forecast_date, height=560,
        ),
        f"02_{slug}_inflation_vs_forecast": charts.inflation_history_vs_forecast(
            inflation, path, snapshot.current_price, ctx.product, mode, palette, height=520
        ),
        f"03_{slug}_price_trend": charts.price_trend(
            prices, ctx.product, mode, palette, height=520
        ),
        f"04_{slug}_weekly_price": charts.periodic_price_bars(
            resample_prices(prices, "W"), ctx.product, "Weekly", mode, palette, tail=52, height=460
        ),
        f"05_{slug}_monthly_inflation": charts.periodic_inflation_bars(
            periodic_inflation(prices, "MS"), "Monthly", mode, palette, tail=36, height=460
        ),
        f"06_{slug}_distribution": charts.price_distribution(
            prices, ctx.product, mode, palette, height=460
        ),
        "07_basket_price_comparison": charts.commodity_price_comparison(
            comparison, mode, palette, height=460
        ),
        "08_basket_inflation_comparison": charts.commodity_inflation_comparison(
            comparison, mode, palette, height=460
        ),
        f"09_{slug}_actual_vs_predicted": charts.actual_vs_predicted(
            bundle.comparison, ctx.product,
            sorted(set(bundle.comparison.loc[bundle.comparison["commodity"] == ctx.product, "model"])),
            mode, palette, height=520,
        ),
        f"10_{slug}_feature_importance": charts.feature_importance_chart(
            bundle.feature_importance, ctx.product, mode, palette, height=620
        ),
    }


left, right = st.columns([1, 1.6], gap="medium")
with left:
    if st.button("Prepare figure bundle", icon=":material/image:", width="stretch"):
        try:
            with st.spinner("Rendering figures…"):
                figures = _build_figures()
                data, used = figures_archive(figures)
            st.session_state["figures_zip"] = data
            st.session_state["figures_format"] = used
            st.success(f"{len(figures)} figures rendered as {used.upper()}.",
                       icon=":material/check_circle:")
        except Exception as exc:  # keep the page alive on export failure
            logger.exception("Figure export failed")
            st.error(f"Figure export failed: {exc}", icon=":material/error:")

    if st.session_state.get("figures_zip"):
        st.download_button(
            "Download figures (ZIP)",
            data=st.session_state["figures_zip"],
            file_name=f"figures_{slug}_{stamp}.zip",
            mime="application/zip",
            icon=":material/download:",
            width="stretch",
            type="primary",
        )
with right:
    st.info(
        "Figures are exported as 2× PNG when a Chrome runtime is available to Kaleido, and as "
        "self-contained interactive HTML otherwise. Individual charts can always be saved from "
        "the camera icon on each chart's toolbar.",
        icon=":material/info:",
    )

# --------------------------------------------------------------------------
# PDF report
# --------------------------------------------------------------------------
section("PDF summary report", "A one-page decision brief for the current selection.")

left, right = st.columns([1, 1.6], gap="medium")
with left:
    if st.button("Generate PDF report", icon=":material/picture_as_pdf:", width="stretch"):
        try:
            with st.spinner("Building report…"):
                insight_text = [
                    f"{card.title}: "
                    + card.body.replace("<b>", "").replace("</b>", "")
                    for card in build_insight_cards(bundle, snapshot, comparison, filters.horizon)
                ]
                recommendations = {
                    "Policy makers": [
                        f"{c.title} — {c.body}"
                        for c in build_recommendations(bundle, snapshot, filters.horizon, "policy")
                    ],
                    "Consumers": [
                        f"{c.title} — {c.body}"
                        for c in build_recommendations(
                            bundle, snapshot, filters.horizon, "consumer"
                        )
                    ],
                }
                st.session_state["summary_pdf"] = summary_pdf(
                    snapshot, path, metrics, comparison, insight_text, recommendations,
                    filters.horizon,
                )
            st.success("Report ready.", icon=":material/check_circle:")
        except ImportError:
            st.error(
                "PDF export needs `reportlab`. Install it with `pip install reportlab` and "
                "try again.",
                icon=":material/error:",
            )
        except Exception as exc:
            logger.exception("PDF export failed")
            st.error(f"Report generation failed: {exc}", icon=":material/error:")

    if st.session_state.get("summary_pdf"):
        st.download_button(
            "Download PDF report",
            data=st.session_state["summary_pdf"],
            file_name=f"price_brief_{slug}_{filters.horizon}d_{stamp}.pdf",
            mime="application/pdf",
            icon=":material/download:",
            width="stretch",
            type="primary",
        )
with right:
    st.info(
        "The brief contains the headline indicators, model accuracy, the basket comparison, "
        "the generated insights, both recommendation sets and the first fifteen forecast days — "
        "sized for a single page a minister or a market officer can actually read.",
        icon=":material/description:",
    )

footer(
    f"All exports are generated live from the artifacts in `data/processed` "
    f"(history to {format_date(bundle.last_observation)}).",
    f"{meta.label} · {ctx.model} · {filters.horizon}-day horizon",
)
