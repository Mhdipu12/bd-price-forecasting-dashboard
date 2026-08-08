"""The sticky sidebar: brand block, navigation slot and the global filter set.

Filters live in ``st.session_state`` under stable keys, so every page reads the
same selection and any change reruns the whole app — KPIs, charts, tables and
narrative text update together.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta

import pandas as pd
import streamlit as st

from dashboard.config import COMMODITY_BY_KEY, SETTINGS, format_date
from dashboard.data_loader import DataBundle
from dashboard.logging_utils import get_logger

logger = get_logger(__name__)

KEY_PRODUCT = "filter_product"
KEY_HORIZON = "filter_horizon"
KEY_DATE = "filter_date"
KEY_MODEL = "filter_model"
KEY_THEME = "filter_theme"

AUTO_MODEL = "Best model (auto)"


@dataclass(frozen=True, slots=True)
class Filters:
    """The resolved sidebar selection shared by every page."""

    product: str
    horizon: int
    forecast_date: pd.Timestamp
    model: str
    model_is_auto: bool
    theme: str

    def model_for(self, commodity: str, bundle: DataBundle) -> str:
        """Resolve the model to use for one commodity.

        In auto mode each commodity uses its own best model; otherwise the
        explicitly selected model is used wherever it exists.

        Args:
            commodity: Canonical commodity key.
            bundle: Loaded artifacts.
        """
        if self.model_is_auto:
            return bundle.best_model(commodity)
        available = bundle.models_for(commodity)
        return self.model if self.model in available else bundle.best_model(commodity)


def render_brand() -> None:
    """Draw the sidebar brand block above the navigation."""
    st.sidebar.html(
        '<div class="bdp-brand">'
        '<div class="bdp-brand-mark">BD</div>'
        '<div class="bdp-brand-text"><b>Commodity Price Monitoring Dashboard</b>'
        "<span>Essential goods · Bangladesh</span></div></div>"
    )


def _horizon_options(bundle: DataBundle) -> list[int]:
    """Return the horizons the exported forecast actually supports."""
    available = [h for h in SETTINGS.horizons if h <= bundle.max_horizon]
    if not available:
        available = [bundle.max_horizon]
    if bundle.max_horizon not in available:
        available.append(bundle.max_horizon)
    return sorted(set(available))


def render_filters(bundle: DataBundle) -> Filters:
    """Render every sidebar filter and return the resolved selection.

    Args:
        bundle: Loaded artifacts, used to bound each control to real data.

    Returns:
        The active :class:`Filters`.
    """
    sidebar = st.sidebar
    sidebar.divider()
    sidebar.markdown("##### Filters")

    # ---- product -------------------------------------------------------
    commodities = bundle.commodities
    product = sidebar.selectbox(
        "Product",
        options=commodities,
        format_func=lambda key: COMMODITY_BY_KEY[key].label,
        key=KEY_PRODUCT,
        help="Every KPI, chart, table and insight on the dashboard follows this selection.",
    )

    # ---- horizon -------------------------------------------------------
    options = _horizon_options(bundle)
    if KEY_HORIZON not in st.session_state:
        st.session_state[KEY_HORIZON] = (
            SETTINGS.default_horizon if SETTINGS.default_horizon in options else options[-1]
        )
    horizon = sidebar.select_slider(
        "Forecast horizon (days)",
        options=options,
        key=KEY_HORIZON,
        help="Length of the forward window used by every forecast view.",
    )

    # ---- forecast date -------------------------------------------------
    start = bundle.forecast_start.date()
    end = (bundle.forecast_start + timedelta(days=int(horizon) - 1)).date()
    _clamp_date(start, end)
    forecast_date = sidebar.date_input(
        "Forecast date",
        min_value=start,
        max_value=end,
        key=KEY_DATE,
        format="DD-MM-YYYY",
        help="Forecast KPIs read this date; historical KPIs always read the last observation.",
    )

    # ---- model ---------------------------------------------------------
    model_options = [AUTO_MODEL, *bundle.models_for(product)]
    if KEY_MODEL not in st.session_state or st.session_state[KEY_MODEL] not in model_options:
        st.session_state[KEY_MODEL] = AUTO_MODEL
    model_choice = sidebar.selectbox(
        "Model",
        options=model_options,
        key=KEY_MODEL,
        help=f"Auto uses the best test-set model per commodity "
        f"({bundle.best_model(product)} for {COMMODITY_BY_KEY[product].label}).",
    )

    # ---- theme ---------------------------------------------------------
    sidebar.divider()
    if KEY_THEME not in st.session_state:
        st.session_state[KEY_THEME] = "Light"
    theme = sidebar.segmented_control(
        "Theme",
        options=["Light", "Dark"],
        key=KEY_THEME,
        width="stretch",
        help="Restyles the whole dashboard instantly without losing your filters. "
        "Data grids stay on a light panel in dark mode because Streamlit renders "
        "them to a canvas.",
    ) or "Light"

    resolved_model = (
        bundle.best_model(product) if model_choice == AUTO_MODEL else str(model_choice)
    )
    filters = Filters(
        product=str(product),
        horizon=int(horizon),
        forecast_date=pd.Timestamp(forecast_date),
        model=resolved_model,
        model_is_auto=model_choice == AUTO_MODEL,
        theme=str(theme),
    )
    _render_footer(bundle, filters)
    logger.debug("Filters resolved: %s", filters)
    return filters


def _clamp_date(low: date, high: date) -> None:
    """Keep the stored forecast date inside the window the horizon allows."""
    if KEY_DATE not in st.session_state:
        st.session_state[KEY_DATE] = high
        return
    current = st.session_state[KEY_DATE]
    if isinstance(current, tuple):  # defensive: range mode was never requested
        current = current[0]
    if not isinstance(current, date):
        st.session_state[KEY_DATE] = high
        return
    st.session_state[KEY_DATE] = min(max(current, low), high)


def _render_footer(bundle: DataBundle, filters: Filters) -> None:
    """Show the active data context at the bottom of the sidebar."""
    st.sidebar.divider()
    st.sidebar.html(
        '<div style="font-size:.72rem;line-height:1.6;color:#94A3B8">'
        f"<div><b style='color:#E2E8F0'>Data through</b> {format_date(bundle.last_observation)}</div>"
        f"<div><b style='color:#E2E8F0'>Forecast window</b> {format_date(bundle.forecast_start)} → "
        f"{format_date(bundle.forecast_end)}</div>"
        f"<div><b style='color:#E2E8F0'>Active model</b> {filters.model}</div>"
        "</div>"
    )
