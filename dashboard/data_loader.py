"""Cached, validated readers for the artifacts exported by the research notebook.

The dashboard never trains a model. It reads the eight files listed in
:data:`dashboard.config.REQUIRED_ARTIFACTS` and turns them into a single
:class:`DataBundle` that every page consumes.

All public loaders are wrapped in ``st.cache_data`` so a filter change costs a
dictionary lookup rather than a disk read.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

from dashboard.config import (
    COMMODITY_KEYS,
    DATA_DIR,
    REQUIRED_ARTIFACTS,
    SETTINGS,
)
from dashboard.logging_utils import get_logger

logger = get_logger(__name__)


class ArtifactError(RuntimeError):
    """Raised when a required artifact is missing, empty or malformed."""


@dataclass(slots=True)
class DataBundle:
    """Everything the dashboard knows, loaded once and cached.

    Attributes:
        prices: ``date, commodity, price`` long table of cleaned history.
        inflation: ``date, commodity, inflation_daily/weekly/monthly/annual``.
        forecast: ``date, commodity, model, horizon_day, forecast, lower, upper``.
        metrics: One row per commodity × model with MAE / RMSE / MAPE.
        comparison: Test-set ``actual`` vs ``predicted`` per commodity × model.
        feature_importance: XGBoost gain importance per commodity × feature.
        insights: Rule base and per-commodity facts from ``insights.json``.
        recommendations: Policy / consumer guidance from ``recommendations.json``.
        missing: Names of optional artifacts that were not found.
    """

    prices: pd.DataFrame
    inflation: pd.DataFrame
    forecast: pd.DataFrame
    metrics: pd.DataFrame
    comparison: pd.DataFrame
    feature_importance: pd.DataFrame
    insights: dict[str, Any] = field(default_factory=dict)
    recommendations: dict[str, Any] = field(default_factory=dict)
    missing: tuple[str, ...] = ()

    # -- convenience accessors ---------------------------------------------
    @property
    def commodities(self) -> list[str]:
        """Commodities present in the artifacts, in registry order."""
        available = set(self.prices["commodity"].unique())
        return [c for c in COMMODITY_KEYS if c in available]

    @property
    def last_observation(self) -> pd.Timestamp:
        """Date of the most recent historical observation across all series."""
        return pd.Timestamp(self.prices["date"].max())

    @property
    def forecast_start(self) -> pd.Timestamp:
        """First forecast date."""
        return pd.Timestamp(self.forecast["date"].min())

    @property
    def forecast_end(self) -> pd.Timestamp:
        """Last forecast date across every horizon."""
        return pd.Timestamp(self.forecast["date"].max())

    @property
    def max_horizon(self) -> int:
        """Longest horizon (in days) present in ``forecast.csv``."""
        return int(self.forecast["horizon_day"].max())

    def best_model(self, commodity: str) -> str:
        """Return the winning model for one commodity.

        Selection uses the ``is_best_model`` flag written by the export script
        and falls back to the lowest test-set MAE.

        Args:
            commodity: Canonical commodity key.

        Returns:
            Model name, or ``"Ensemble"`` when nothing can be determined.
        """
        flagged = self.metrics[
            (self.metrics["commodity"] == commodity) & self.metrics["is_best_model"]
        ]
        if not flagged.empty:
            return str(flagged.iloc[0]["model"])

        scored = self.metrics[
            (self.metrics["commodity"] == commodity)
            & (self.metrics["model"] != "Seasonal Naive")
        ].sort_values("MAE")
        if not scored.empty:
            return str(scored.iloc[0]["model"])
        logger.warning("No metrics found for %s; defaulting to Ensemble.", commodity)
        return "Ensemble"

    def models_for(self, commodity: str) -> list[str]:
        """Forecast models available for one commodity, in display order."""
        from dashboard.config import FORECAST_MODELS

        present = set(self.forecast.loc[self.forecast["commodity"] == commodity, "model"])
        return [m for m in FORECAST_MODELS if m in present]


# --------------------------------------------------------------------------
# Low-level readers
# --------------------------------------------------------------------------
def _resolve(name: str, data_dir: Path) -> Path:
    """Return the path of an artifact, raising a helpful error when absent."""
    path = data_dir / name
    if not path.exists():
        raise ArtifactError(
            f"Required artifact '{name}' was not found in {data_dir}. "
            "Run `python scripts/build_artifacts.py` (or export Phase 10 of the "
            "research notebook) to generate it."
        )
    return path


def _read_csv(name: str, data_dir: Path, parse_dates: list[str] | None = None) -> pd.DataFrame:
    """Read one artifact CSV with consistent error handling."""
    path = _resolve(name, data_dir)
    try:
        frame = pd.read_csv(path, parse_dates=parse_dates or [])
    except (OSError, pd.errors.ParserError) as exc:
        raise ArtifactError(f"Could not parse '{name}': {exc}") from exc
    if frame.empty:
        raise ArtifactError(f"Artifact '{name}' is empty.")
    logger.debug("Loaded %s (%d rows).", name, len(frame))
    return frame


def _require_columns(frame: pd.DataFrame, columns: set[str], name: str) -> None:
    """Assert that a frame carries every column the dashboard depends on."""
    missing = columns - set(frame.columns)
    if missing:
        raise ArtifactError(f"Artifact '{name}' is missing column(s): {sorted(missing)}")


def _read_json(name: str, data_dir: Path) -> dict[str, Any]:
    """Read an optional JSON artifact, returning ``{}`` when unavailable."""
    path = data_dir / name
    if not path.exists():
        logger.info("Optional artifact '%s' not found; using computed fallbacks.", name)
        return {}
    try:
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning("Could not read '%s': %s", name, exc)
        return {}
    return payload if isinstance(payload, dict) else {}


# --------------------------------------------------------------------------
# Public API
# --------------------------------------------------------------------------
@st.cache_data(ttl=SETTINGS.cache_ttl, show_spinner=False)
def load_bundle(data_dir: str | None = None) -> DataBundle:
    """Load and validate every dashboard artifact.

    Args:
        data_dir: Optional override for ``data/processed``. Passed as a string
            so the cache key stays hashable.

    Returns:
        A fully populated :class:`DataBundle`.

    Raises:
        ArtifactError: If a mandatory artifact is missing or malformed.
    """
    directory = Path(data_dir) if data_dir else DATA_DIR
    logger.info("Loading dashboard artifacts from %s", directory)

    if not directory.exists():
        raise ArtifactError(
            f"Artifact directory {directory} does not exist. "
            "Run `python scripts/build_artifacts.py` first."
        )

    prices = _read_csv("historical_price.csv", directory, ["date"])
    _require_columns(prices, {"date", "commodity", "price"}, "historical_price.csv")
    prices = prices.sort_values(["commodity", "date"]).reset_index(drop=True)

    inflation = _read_csv("historical_inflation.csv", directory, ["date"])
    _require_columns(inflation, {"date", "commodity", "inflation_daily"}, "historical_inflation.csv")
    inflation = inflation.sort_values(["commodity", "date"]).reset_index(drop=True)

    forecast = _read_csv("forecast.csv", directory, ["date"])
    _require_columns(
        forecast,
        {"date", "commodity", "model", "horizon_day", "forecast", "lower", "upper"},
        "forecast.csv",
    )
    forecast = forecast.sort_values(["commodity", "model", "date"]).reset_index(drop=True)

    metrics = _read_csv("metrics.csv", directory)
    _require_columns(metrics, {"commodity", "model", "MAE", "RMSE", "MAPE"}, "metrics.csv")
    if "is_best_model" not in metrics.columns:
        metrics["is_best_model"] = False
    metrics["is_best_model"] = metrics["is_best_model"].astype(bool)

    comparison = _read_csv("model_comparison.csv", directory, ["date"])
    _require_columns(
        comparison, {"date", "commodity", "model", "actual", "predicted"}, "model_comparison.csv"
    )

    try:
        importance = _read_csv("feature_importance.csv", directory)
        _require_columns(importance, {"commodity", "feature", "importance"}, "feature_importance.csv")
    except ArtifactError as exc:
        logger.info("Feature importance unavailable: %s", exc)
        importance = pd.DataFrame(columns=["commodity", "feature", "importance"])

    insights = _read_json("insights.json", directory)
    recommendations = _read_json("recommendations.json", directory)

    missing = tuple(
        name
        for name, required in REQUIRED_ARTIFACTS.items()
        if not required and not (directory / name).exists()
    )

    bundle = DataBundle(
        prices=prices,
        inflation=inflation,
        forecast=forecast,
        metrics=metrics,
        comparison=comparison,
        feature_importance=importance,
        insights=insights,
        recommendations=recommendations,
        missing=missing,
    )
    logger.info(
        "Artifacts loaded: %d commodities, history to %s, forecast to %s.",
        len(bundle.commodities),
        bundle.last_observation.date(),
        bundle.forecast_end.date(),
    )
    return bundle


def artifact_signature(data_dir: Path | None = None) -> str:
    """Build a cheap cache key from artifact modification times.

    Useful when an external process rewrites ``data/processed`` while the app is
    running: pass the signature into a cached function to invalidate it.

    Args:
        data_dir: Optional artifact directory override.

    Returns:
        A string that changes whenever any artifact on disk changes.
    """
    directory = data_dir or DATA_DIR
    parts: list[str] = []
    for name in sorted(REQUIRED_ARTIFACTS):
        path = directory / name
        parts.append(f"{name}:{int(path.stat().st_mtime) if path.exists() else 0}")
    return "|".join(parts)
