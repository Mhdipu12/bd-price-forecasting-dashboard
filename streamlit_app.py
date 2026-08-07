"""Entry point for the Bangladesh essential-goods price intelligence dashboard.

Run with::

    streamlit run streamlit_app.py

The app is a *reader* of exported research artifacts — it never trains a model.
Generate the artifacts once with ``python scripts/build_artifacts.py`` (or the
Phase 10 export cell of the research notebook) before starting the dashboard.
"""

from __future__ import annotations

import streamlit as st

from dashboard.config import SETTINGS

st.set_page_config(
    page_title=SETTINGS.app_short_title,
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        "About": (
            f"**{SETTINGS.app_title}**\n\n{SETTINGS.subtitle}\n\n"
            "Comparative study of SARIMA, XGBoost, LSTM and a weighted ensemble."
        )
    },
)

from dashboard.context import AppContext, set_context  # noqa: E402
from dashboard.data_loader import ArtifactError, load_bundle  # noqa: E402
from dashboard.logging_utils import configure_logging, get_logger  # noqa: E402
from dashboard.sidebar import KEY_THEME, render_brand, render_filters  # noqa: E402
from dashboard.theme import apply_theme  # noqa: E402

configure_logging()
logger = get_logger(__name__)

# --------------------------------------------------------------------------
# Theme — read before anything renders so the whole run is styled coherently.
# --------------------------------------------------------------------------
mode = st.session_state.get(KEY_THEME, "Light")
palette = apply_theme(mode)

# --------------------------------------------------------------------------
# Data
# --------------------------------------------------------------------------
try:
    with st.spinner("Loading forecast artifacts…", show_time=False):
        bundle = load_bundle()
except ArtifactError as exc:
    st.error(f"**Dashboard artifacts could not be loaded.**\n\n{exc}", icon=":material/error:")
    st.markdown(
        "#### How to fix this\n"
        "1. Generate the artifacts from the raw data:\n"
        "   ```\n   python scripts/build_artifacts.py\n   ```\n"
        "2. Or copy the files exported by Phase 10 of the research notebook into "
        "`data/processed/`.\n\n"
        "Expected files: `historical_price.csv`, `historical_inflation.csv`, "
        "`forecast.csv`, `metrics.csv`, `model_comparison.csv`, "
        "`feature_importance.csv`, `insights.json`, `recommendations.json`."
    )
    logger.error("Startup failed: %s", exc)
    st.stop()
except Exception as exc:  # unexpected — surface it rather than showing a blank page
    st.error(f"**Unexpected error while loading data:** {exc}", icon=":material/error:")
    logger.exception("Unhandled startup error")
    st.stop()

# --------------------------------------------------------------------------
# Sidebar: brand → navigation → filters
# --------------------------------------------------------------------------
render_brand()

navigation = st.navigation(
    {
        "Analysis": [
            st.Page("app_pages/overview.py", title="Overview", icon=":material/dashboard:",
                    default=True),
            st.Page("app_pages/product_analysis.py", title="Product analysis",
                    icon=":material/query_stats:"),
            st.Page("app_pages/price_forecast.py", title="Price forecast",
                    icon=":material/timeline:"),
            st.Page("app_pages/inflation_analysis.py", title="Inflation analysis",
                    icon=":material/percent:"),
        ],
        "Models & decisions": [
            st.Page("app_pages/model_comparison.py", title="Model comparison",
                    icon=":material/compare_arrows:"),
            st.Page("app_pages/insights.py", title="Insights",
                    icon=":material/lightbulb:"),
            st.Page("app_pages/downloads.py", title="Downloads",
                    icon=":material/download:"),
        ],
    },
    position="sidebar",
)

filters = render_filters(bundle)
set_context(AppContext(bundle=bundle, filters=filters, palette=palette, mode=mode))

if bundle.missing:
    st.toast(
        f"Optional artifact(s) not found: {', '.join(bundle.missing)} — using computed fallbacks.",
        icon=":material/info:",
    )

navigation.run()
