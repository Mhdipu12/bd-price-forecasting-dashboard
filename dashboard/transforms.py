"""Pure pandas helpers that derive every analytical view the dashboard shows.

These functions contain the project's *data logic* and nothing else — no
Streamlit calls, no plotting. Two rules are enforced throughout:

* historical KPIs always read the **last historical observation**;
* forecast KPIs always read the **selected forecast date**.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd

from dashboard.config import SETTINGS
from dashboard.data_loader import DataBundle
from dashboard.logging_utils import get_logger

logger = get_logger(__name__)

Frequency = Literal["W", "MS"]


# --------------------------------------------------------------------------
# Slicing helpers
# --------------------------------------------------------------------------
def commodity_prices(bundle: DataBundle, commodity: str) -> pd.DataFrame:
    """Return ``date, price`` history for one commodity."""
    frame = bundle.prices.loc[bundle.prices["commodity"] == commodity, ["date", "price"]]
    return frame.sort_values("date").reset_index(drop=True)


def commodity_inflation(bundle: DataBundle, commodity: str) -> pd.DataFrame:
    """Return the historical inflation frame for one commodity."""
    frame = bundle.inflation.loc[bundle.inflation["commodity"] == commodity]
    return frame.sort_values("date").reset_index(drop=True)


def commodity_forecast(
    bundle: DataBundle,
    commodity: str,
    model: str,
    horizon: int | None = None,
) -> pd.DataFrame:
    """Return the forecast path for one commodity / model, trimmed to a horizon.

    Args:
        bundle: Loaded artifacts.
        commodity: Canonical commodity key.
        model: Model name; falls back to the commodity's best model when the
            requested model is unavailable.
        horizon: Keep only the first ``horizon`` forecast days. ``None`` keeps all.

    Returns:
        ``date, horizon_day, forecast, lower, upper`` sorted chronologically.
    """
    frame = bundle.forecast[bundle.forecast["commodity"] == commodity]
    if model not in set(frame["model"]):
        fallback = bundle.best_model(commodity)
        logger.warning("Model %s unavailable for %s; using %s.", model, commodity, fallback)
        model = fallback
    frame = frame[frame["model"] == model]
    if horizon is not None:
        frame = frame[frame["horizon_day"] <= horizon]
    columns = ["date", "horizon_day", "forecast", "lower", "upper"]
    return frame[columns].sort_values("date").reset_index(drop=True)


def forecast_row(forecast: pd.DataFrame, target: pd.Timestamp) -> pd.Series | None:
    """Return the forecast row on ``target``, or the nearest earlier row.

    Args:
        forecast: Output of :func:`commodity_forecast`.
        target: Selected forecast date.

    Returns:
        The matching row, or ``None`` when the path is empty.
    """
    if forecast.empty:
        return None
    target = pd.Timestamp(target).normalize()
    exact = forecast[forecast["date"].dt.normalize() == target]
    if not exact.empty:
        return exact.iloc[0]
    earlier = forecast[forecast["date"] <= target]
    if not earlier.empty:
        return earlier.iloc[-1]
    return forecast.iloc[0]


# --------------------------------------------------------------------------
# Resampling and aggregation
# --------------------------------------------------------------------------
def resample_prices(prices: pd.DataFrame, freq: Frequency) -> pd.DataFrame:
    """Aggregate a daily price frame to weekly or monthly means.

    Args:
        prices: ``date, price`` frame.
        freq: ``"W"`` for week-ending or ``"MS"`` for month-start buckets.

    Returns:
        ``date, price, low, high`` where low/high are the bucket min/max.
    """
    if prices.empty:
        return prices.assign(low=[], high=[])
    grouped = prices.set_index("date")["price"].resample(freq)
    frame = pd.DataFrame(
        {
            "price": grouped.mean(),
            "low": grouped.min(),
            "high": grouped.max(),
        }
    ).dropna()
    return frame.reset_index()


def periodic_inflation(prices: pd.DataFrame, freq: Frequency) -> pd.DataFrame:
    """Compute period-over-period inflation from a daily price frame.

    Args:
        prices: ``date, price`` frame.
        freq: ``"W"`` (week-on-week) or ``"MS"`` (month-on-month).

    Returns:
        ``date, inflation`` in percent.
    """
    resampled = resample_prices(prices, freq)
    if resampled.empty:
        return pd.DataFrame(columns=["date", "inflation"])
    resampled["inflation"] = resampled["price"].pct_change() * 100
    return resampled.dropna(subset=["inflation"])[["date", "inflation"]].reset_index(drop=True)


def inflation_heatmap_matrix(prices: pd.DataFrame) -> pd.DataFrame:
    """Build a year × month matrix of month-on-month inflation.

    Args:
        prices: ``date, price`` frame.

    Returns:
        A pivot table with years as rows and month numbers as columns.
    """
    monthly = periodic_inflation(prices, "MS")
    if monthly.empty:
        return pd.DataFrame()
    monthly = monthly.assign(
        year=monthly["date"].dt.year,
        month=monthly["date"].dt.month,
    )
    return monthly.pivot_table(index="year", columns="month", values="inflation", aggfunc="mean")


def descriptive_statistics(prices: pd.DataFrame, unit: str) -> pd.DataFrame:
    """Summarise a price series for the Product Analysis page.

    Args:
        prices: ``date, price`` frame.
        unit: Unit label appended to price-denominated rows.

    Returns:
        A two-column ``Statistic, Value`` frame ready for ``st.dataframe``.
    """
    if prices.empty:
        return pd.DataFrame(columns=["Statistic", "Value"])

    series = prices["price"]
    daily_returns = series.pct_change() * 100
    rows: list[tuple[str, str]] = [
        ("Observations", f"{len(series):,}"),
        ("First observation", prices['date'].iloc[0].strftime(SETTINGS.date_format)),
        ("Last observation", prices['date'].iloc[-1].strftime(SETTINGS.date_format)),
        ("Mean", f"{series.mean():,.2f} {unit}"),
        ("Median", f"{series.median():,.2f} {unit}"),
        ("Standard deviation", f"{series.std():,.2f} {unit}"),
        ("Minimum", f"{series.min():,.2f} {unit}"),
        ("Maximum", f"{series.max():,.2f} {unit}"),
        ("Range", f"{series.max() - series.min():,.2f} {unit}"),
        ("Coefficient of variation", f"{series.std() / series.mean() * 100:,.2f} %"),
        ("Skewness", f"{series.skew():,.3f}"),
        ("Kurtosis", f"{series.kurtosis():,.3f}"),
        ("Mean daily change", f"{daily_returns.mean():+,.3f} %"),
        ("Daily volatility (sd of returns)", f"{daily_returns.std():,.3f} %"),
        ("Total change over sample", f"{(series.iloc[-1] / series.iloc[0] - 1) * 100:+,.2f} %"),
    ]
    return pd.DataFrame(rows, columns=["Statistic", "Value"])


def realised_volatility(prices: pd.DataFrame, window: int | None = None) -> float:
    """Return the trailing standard deviation of daily returns, in percent.

    Args:
        prices: ``date, price`` frame.
        window: Number of trailing days; defaults to the configured window.

    Returns:
        Realised volatility in percent, or ``nan`` when there is too little data.
    """
    window = window or SETTINGS.volatility_window
    if len(prices) < window + 1:
        return float("nan")
    returns = prices["price"].pct_change().tail(window)
    return float(returns.std() * 100)


# --------------------------------------------------------------------------
# KPI snapshot
# --------------------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class Snapshot:
    """Every headline number for one commodity at one forecast date.

    Historical fields come from the last observation; forecast fields come from
    the user's selected forecast date.
    """

    commodity: str
    model: str
    as_of: pd.Timestamp
    forecast_date: pd.Timestamp
    current_price: float
    forecast_price: float
    lower: float
    upper: float
    current_inflation: float
    forecast_inflation: float
    expected_change_pct: float
    volatility: float
    horizon_days: int
    sparkline: list[float]
    forecast_sparkline: list[float]

    @property
    def direction(self) -> Literal["up", "down", "flat"]:
        """Direction of the expected price move."""
        if abs(self.expected_change_pct) < SETTINGS.thresholds.stable_change:
            return "flat"
        return "up" if self.expected_change_pct > 0 else "down"

    @property
    def risk_level(self) -> Literal["Low", "Moderate", "High"]:
        """Blend volatility and expected move into a three-level risk rating."""
        thresholds = SETTINGS.thresholds
        score = 0
        if self.volatility >= thresholds.high_volatility:
            score += 2
        elif self.volatility >= thresholds.low_volatility:
            score += 1
        if abs(self.expected_change_pct) >= thresholds.material_change:
            score += 2
        elif abs(self.expected_change_pct) >= thresholds.stable_change:
            score += 1
        if score >= 3:
            return "High"
        return "Moderate" if score >= 2 else "Low"


def build_snapshot(
    bundle: DataBundle,
    commodity: str,
    model: str,
    forecast_date: pd.Timestamp,
    horizon: int,
) -> Snapshot:
    """Assemble the KPI snapshot that drives every card on the dashboard.

    Args:
        bundle: Loaded artifacts.
        commodity: Canonical commodity key.
        model: Selected model.
        forecast_date: Date chosen in the sidebar.
        horizon: Selected forecast horizon in days.

    Returns:
        A populated :class:`Snapshot`.
    """
    prices = commodity_prices(bundle, commodity)
    inflation = commodity_inflation(bundle, commodity)
    path = commodity_forecast(bundle, commodity, model, horizon)

    current_price = float(prices["price"].iloc[-1]) if not prices.empty else float("nan")
    as_of = pd.Timestamp(prices["date"].iloc[-1]) if not prices.empty else pd.NaT

    current_inflation = float("nan")
    if not inflation.empty and "inflation_monthly" in inflation.columns:
        recent = inflation["inflation_monthly"].dropna()
        if not recent.empty:
            current_inflation = float(recent.iloc[-1])

    row = forecast_row(path, forecast_date)
    if row is None:
        logger.warning("No forecast path for %s / %s.", commodity, model)
        forecast_price = lower = upper = float("nan")
        horizon_day = 0
        resolved_date = pd.Timestamp(forecast_date)
    else:
        forecast_price = float(row["forecast"])
        lower = float(row["lower"])
        upper = float(row["upper"])
        horizon_day = int(row["horizon_day"])
        resolved_date = pd.Timestamp(row["date"])

    expected_change = (
        (forecast_price - current_price) / current_price * 100
        if current_price and np.isfinite(forecast_price)
        else float("nan")
    )

    # Forecast inflation is the annualised rate implied by the horizon move,
    # expressed on the same monthly basis as the historical figure.
    if np.isfinite(expected_change) and horizon_day > 0:
        forecast_inflation = ((forecast_price / current_price) ** (30.0 / horizon_day) - 1) * 100
    else:
        forecast_inflation = float("nan")

    tail = prices["price"].tail(SETTINGS.sparkline_days).tolist()

    return Snapshot(
        commodity=commodity,
        model=model,
        as_of=as_of,
        forecast_date=resolved_date,
        current_price=current_price,
        forecast_price=forecast_price,
        lower=lower,
        upper=upper,
        current_inflation=current_inflation,
        forecast_inflation=forecast_inflation,
        expected_change_pct=expected_change,
        volatility=realised_volatility(prices),
        horizon_days=horizon_day,
        sparkline=tail,
        forecast_sparkline=path["forecast"].tolist(),
    )


def snapshot_table(
    bundle: DataBundle,
    model_selector,
    forecast_date: pd.Timestamp,
    horizon: int,
) -> pd.DataFrame:
    """Build the cross-commodity comparison table used on the Overview page.

    Args:
        bundle: Loaded artifacts.
        model_selector: Callable mapping a commodity key to a model name.
        forecast_date: Selected forecast date.
        horizon: Selected horizon in days.

    Returns:
        One row per commodity with current/forecast price and inflation.
    """
    rows: list[dict[str, object]] = []
    for commodity in bundle.commodities:
        snap = build_snapshot(bundle, commodity, model_selector(commodity), forecast_date, horizon)
        rows.append(
            {
                "commodity": commodity,
                "model": snap.model,
                "Current price": snap.current_price,
                "Forecast price": snap.forecast_price,
                "Current inflation": snap.current_inflation,
                "Forecast inflation": snap.forecast_inflation,
                "Expected change": snap.expected_change_pct,
                "Volatility": snap.volatility,
                "Risk": snap.risk_level,
            }
        )
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# Forecast presentation
# --------------------------------------------------------------------------
def forecast_table(
    path: pd.DataFrame,
    current_price: float,
    unit_label: str = "BDT",
) -> pd.DataFrame:
    """Format a forecast path for display and CSV download.

    Args:
        path: Output of :func:`commodity_forecast`.
        current_price: Last historical observation, used as the change base.
        unit_label: Currency label for column headers.

    Returns:
        A tidy frame with date, forecast, interval and expected change.
    """
    if path.empty:
        return pd.DataFrame(
            columns=["Date", f"Forecast price ({unit_label})", "Expected change (%)"]
        )

    frame = pd.DataFrame(
        {
            "Date": path["date"].dt.date,
            "Day": path["horizon_day"].astype(int),
            f"Forecast price ({unit_label})": path["forecast"].round(2),
            f"Lower 95% ({unit_label})": path["lower"].round(2),
            f"Upper 95% ({unit_label})": path["upper"].round(2),
            "Expected change (%)": ((path["forecast"] - current_price) / current_price * 100).round(2),
        }
    )
    frame["Day-on-day (%)"] = (path["forecast"].pct_change() * 100).round(2).values
    return frame


def error_summary(comparison: pd.DataFrame, commodity: str) -> pd.DataFrame:
    """Compute per-model error diagnostics on the held-out test set.

    Args:
        comparison: ``date, commodity, model, actual, predicted`` frame.
        commodity: Canonical commodity key.

    Returns:
        One row per model with bias, mean absolute error and worst-case error.
    """
    subset = comparison[comparison["commodity"] == commodity].copy()
    if subset.empty:
        return pd.DataFrame(columns=["model", "Bias", "MAE", "Max error"])

    subset["error"] = subset["actual"] - subset["predicted"]
    grouped = subset.groupby("model")["error"]
    summary = pd.DataFrame(
        {
            "Bias": grouped.mean(),
            "MAE": grouped.apply(lambda s: s.abs().mean()),
            "Max error": grouped.apply(lambda s: s.abs().max()),
            "Error sd": grouped.std(),
        }
    ).reset_index()
    return summary
