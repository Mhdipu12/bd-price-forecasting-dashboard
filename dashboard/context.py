"""The per-run application context shared between the entry point and pages."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
import streamlit as st

from dashboard.config import Palette
from dashboard.data_loader import DataBundle
from dashboard.sidebar import Filters
from dashboard.transforms import Snapshot, build_snapshot, snapshot_table

_CONTEXT_KEY = "_bdp_context"


@dataclass(frozen=True, slots=True)
class AppContext:
    """Everything a page needs, assembled once per script run.

    Attributes:
        bundle: Loaded artifacts.
        filters: Resolved sidebar selection.
        palette: Active colour palette.
        mode: ``"Light"`` or ``"Dark"``.
    """

    bundle: DataBundle
    filters: Filters
    palette: Palette
    mode: str

    @property
    def product(self) -> str:
        """Canonical key of the selected commodity."""
        return self.filters.product

    @property
    def model(self) -> str:
        """Model resolved for the selected commodity."""
        return self.filters.model_for(self.filters.product, self.bundle)

    def snapshot(self, commodity: str | None = None) -> Snapshot:
        """Build the KPI snapshot for one commodity (defaults to the selection)."""
        key = commodity or self.product
        return build_snapshot(
            self.bundle,
            key,
            self.filters.model_for(key, self.bundle),
            self.filters.forecast_date,
            self.filters.horizon,
        )

    def comparison_table(self) -> pd.DataFrame:
        """Build the cross-commodity comparison table for the current filters."""
        return snapshot_table(
            self.bundle,
            lambda key: self.filters.model_for(key, self.bundle),
            self.filters.forecast_date,
            self.filters.horizon,
        )


def set_context(context: AppContext) -> None:
    """Store the context for the current script run."""
    st.session_state[_CONTEXT_KEY] = context


def get_context() -> AppContext:
    """Return the context assembled by the entry point.

    Raises:
        RuntimeError: If a page is executed outside ``streamlit_app.py``.
    """
    context = st.session_state.get(_CONTEXT_KEY)
    if context is None:
        raise RuntimeError(
            "Application context is not initialised. Run the dashboard with "
            "`streamlit run streamlit_app.py`."
        )
    return context
