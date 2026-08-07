"""Decision-support dashboard package for the Bangladesh price-forecasting study.

The package is deliberately split so that every layer has exactly one job:

``config``      immutable settings, palette and the commodity registry
``logging_utils`` one place that configures logging for app and scripts
``data_loader`` cached, validated readers for the exported model artifacts
``transforms``  pure pandas helpers that derive weekly/monthly/statistical views
``theme``       CSS + Plotly template for the light/dark presentation layer
``components``  reusable Streamlit UI blocks (status cards, KPI cards, ...)
``charts``      Plotly figure builders — one function per analytical question
``insights``    turns loaded artifacts into date-specific narrative text
``reporting``   PDF / figure export helpers used by the Downloads page

Nothing in this package trains a model. The dashboard is a *reader* of the
artifacts exported by the research notebook (see ``scripts/``).
"""

from __future__ import annotations

__all__ = ["__version__"]

__version__ = "1.0.0"
