"""Build every dashboard artifact from the raw commodity CSVs.

This script is the local counterpart of Phase 10 of the research notebook. It
reproduces the study's pipeline — cleaning, feature engineering, a chronological
split, SARIMA / XGBoost / LSTM / weighted-ensemble estimation, and 60-day
recursive forecasting — and writes the eight files the dashboard reads.

Run it once before starting the dashboard::

    python scripts/build_artifacts.py

Options::

    --max-horizon 60      forecast length in days
    --test-days 180       held-out days used to score the models
    --skip-lstm           skip the neural model when PyTorch is unavailable
    --quick               reduce the SARIMA search and LSTM epochs (fast smoke run)

If you already have the notebook's exported outputs, copy them into
``data/processed`` instead — the dashboard does not care which produced them.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import warnings
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dashboard.config import COMMODITIES, DATA_DIR, RAW_DATA_DIR  # noqa: E402
from dashboard.logging_utils import get_logger  # noqa: E402

warnings.filterwarnings("ignore")
logger = get_logger(__name__)

SEED = 42
LAGS = (1, 2, 3, 7, 14, 30)
ROLLING_WINDOWS = (7, 14, 30)
EMA_SPANS = (7, 30)
SEASONAL_PERIOD = 7
LOOKBACK = 30
MODELS = ("SARIMA", "XGBoost", "LSTM", "Ensemble")


# ==========================================================================
# Phase 1 — cleaning
# ==========================================================================
def load_raw(path: Path) -> pd.Series:
    """Read one raw commodity CSV into a date-indexed price series.

    Args:
        path: CSV with a date column and a single numeric price column.

    Returns:
        A float series indexed by ``DatetimeIndex``, sorted and deduplicated.
    """
    frame = pd.read_csv(path)
    frame.columns = [c.strip().lower().replace(" ", "_") for c in frame.columns]
    date_col = next((c for c in frame.columns if "date" in c), frame.columns[0])
    price_col = next((c for c in frame.columns if c != date_col), frame.columns[-1])

    frame[date_col] = pd.to_datetime(frame[date_col], errors="coerce")
    frame[price_col] = pd.to_numeric(frame[price_col], errors="coerce")
    frame = frame.dropna(subset=[date_col]).sort_values(date_col)
    series = frame.groupby(date_col)[price_col].mean()
    series.index.name = "date"
    return series.astype(float)


def clean_series(
    series: pd.Series,
    window: int = 9,
    threshold: float = 6.0,
    mad_floor_pct: float = 0.01,
) -> pd.Series:
    """Reindex to a complete daily calendar, impute gaps and winsorise outliers.

    Outliers are detected with a *local* rolling median/MAD rule so that a
    genuine multi-week price spike survives while a single implausible day is
    pulled back to the local band. Two details matter:

    * the window is short (9 days) so the reference median tracks a fast but
      real move instead of lagging behind it and flagging the whole climb;
    * the MAD is floored at ``mad_floor_pct`` of the local price level, because
      Bangladeshi retail prices sit flat at round numbers for days at a time,
      which drives a raw MAD to zero and makes the robust z-score explode.

    Args:
        series: Raw daily price series.
        window: Centred window used for the robust statistics.
        threshold: Robust z-score cut-off.
        mad_floor_pct: Lower bound on the MAD, as a fraction of the local median.

    Returns:
        The cleaned series on a gap-free daily index.
    """
    calendar = pd.date_range(series.index.min(), series.index.max(), freq="D")
    filled = series.reindex(calendar).interpolate(method="time").bfill().ffill()
    filled.index.name = "date"

    median = filled.rolling(window, center=True, min_periods=3).median()
    deviation = (filled - median).abs()
    mad = deviation.rolling(window, center=True, min_periods=3).median()
    floor = np.maximum(mad_floor_pct * median, max(float(deviation.median()), 1e-6))
    mad = mad.replace(0, np.nan).fillna(floor).clip(lower=floor)

    robust_z = 0.6745 * (filled - median) / mad
    flags = robust_z.abs() > threshold
    lower = median - threshold * mad / 0.6745
    upper = median + threshold * mad / 0.6745

    treated = filled.copy()
    treated[flags] = filled[flags].clip(lower=lower[flags], upper=upper[flags])
    logger.info("  cleaned: %d rows, %d outliers winsorised", len(treated), int(flags.sum()))
    return treated


# ==========================================================================
# Phase 3 — feature engineering
# ==========================================================================
def build_features(price: pd.Series) -> pd.DataFrame:
    """Build the leakage-free supervised feature matrix used by XGBoost.

    Every rolling and change statistic is computed on a ``shift(1)`` view, so a
    row dated *t* only ever sees information available strictly before *t*.

    Args:
        price: Cleaned daily price series.

    Returns:
        A feature frame aligned on the input index, including the ``price``
        target column.
    """
    series = price.astype(float)
    features = pd.DataFrame(index=series.index)
    features["price"] = series
    index = series.index

    features["year"] = index.year
    features["month"] = index.month
    features["week"] = index.isocalendar().week.astype(int)
    features["day"] = index.day
    features["dayofweek"] = index.dayofweek
    features["dayofyear"] = index.dayofyear
    features["quarter"] = index.quarter
    features["is_weekend"] = index.dayofweek.isin([4, 5]).astype(int)
    features["is_month_start"] = index.is_month_start.astype(int)
    features["is_month_end"] = index.is_month_end.astype(int)
    features["month_sin"] = np.sin(2 * np.pi * features["month"] / 12)
    features["month_cos"] = np.cos(2 * np.pi * features["month"] / 12)
    features["doy_sin"] = np.sin(2 * np.pi * features["dayofyear"] / 365.25)
    features["doy_cos"] = np.cos(2 * np.pi * features["dayofyear"] / 365.25)
    features["dow_sin"] = np.sin(2 * np.pi * features["dayofweek"] / 7)
    features["dow_cos"] = np.cos(2 * np.pi * features["dayofweek"] / 7)

    for lag in LAGS:
        features[f"lag_{lag}"] = series.shift(lag)

    past = series.shift(1)
    for window in ROLLING_WINDOWS:
        features[f"roll_mean_{window}"] = past.rolling(window).mean()
        features[f"roll_median_{window}"] = past.rolling(window).median()
        features[f"roll_max_{window}"] = past.rolling(window).max()
        features[f"roll_min_{window}"] = past.rolling(window).min()
        features[f"roll_std_{window}"] = past.rolling(window).std()
        features[f"roll_range_{window}"] = (
            features[f"roll_max_{window}"] - features[f"roll_min_{window}"]
        )

    for span in EMA_SPANS:
        ema = past.ewm(span=span, adjust=False).mean()
        features[f"ema_{span}"] = ema
        features[f"lag1_over_ema_{span}"] = past / ema

    features["pct_change_1"] = past.pct_change(1) * 100
    features["pct_change_7"] = past.pct_change(7) * 100
    features["pct_change_30"] = past.pct_change(30) * 100
    features["log_return_1"] = np.log(past / past.shift(1))
    features["diff_1"] = past.diff(1)
    features["diff_7"] = past.diff(7)

    features["inflation_daily"] = past.pct_change(1) * 100
    features["inflation_weekly"] = past.pct_change(7) * 100
    features["inflation_monthly"] = past.pct_change(30) * 100
    features["inflation_annual"] = past.pct_change(365) * 100

    features["volatility_7"] = past.pct_change().rolling(7).std() * 100
    features["volatility_30"] = past.pct_change().rolling(30).std() * 100
    return features


def feature_columns(features: pd.DataFrame) -> list[str]:
    """Return the predictor columns (everything except the target)."""
    return [c for c in features.columns if c != "price"]


# ==========================================================================
# Metrics
# ==========================================================================
def regression_metrics(actual, predicted) -> dict[str, float]:
    """Compute MAE, RMSE and MAPE, ignoring positions where either is missing."""
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    mask = ~(np.isnan(actual) | np.isnan(predicted))
    actual, predicted = actual[mask], predicted[mask]
    if actual.size == 0:
        return {"MAE": float("nan"), "RMSE": float("nan"), "MAPE": float("nan")}
    error = actual - predicted
    return {
        "MAE": float(np.mean(np.abs(error))),
        "RMSE": float(math.sqrt(np.mean(error**2))),
        "MAPE": float(np.mean(np.abs(error / np.where(actual == 0, np.nan, actual))) * 100),
    }


# ==========================================================================
# Phase 5 — SARIMA
# ==========================================================================
def select_differencing(series: pd.Series) -> int:
    """Choose the regular differencing order using ADF and KPSS together."""
    from statsmodels.tsa.stattools import adfuller, kpss

    def verdict(sample: pd.Series) -> bool:
        """Return True when the sample looks stationary."""
        sample = sample.dropna()
        if len(sample) < 30:
            return True
        try:
            adf_p = adfuller(sample, autolag="AIC")[1]
        except Exception:
            adf_p = 1.0
        try:
            kpss_p = kpss(sample, regression="c", nlags="auto")[1]
        except Exception:
            kpss_p = 0.10
        return adf_p < 0.05 and kpss_p > 0.05

    if verdict(series):
        return 0
    return 1 if verdict(series.diff()) else 2


def fit_sarima(series: pd.Series, quick: bool = False):
    """Search SARIMA orders by AIC and return the fitted result.

    Args:
        series: Training price series.
        quick: Use a smaller grid (for smoke runs).

    Returns:
        ``(fitted_results, order, seasonal_order, aic)``.
    """
    from statsmodels.tsa.statespace.sarimax import SARIMAX

    d = select_differencing(series)
    p_max = q_max = 1 if quick else 2
    seasonal_grid = [(0, 0, 0, SEASONAL_PERIOD)] if quick else [
        (0, 0, 0, SEASONAL_PERIOD), (1, 0, 0, SEASONAL_PERIOD),
        (0, 0, 1, SEASONAL_PERIOD), (1, 0, 1, SEASONAL_PERIOD),
    ]

    best = {"aic": np.inf, "order": (1, d, 1), "seasonal": (0, 0, 0, SEASONAL_PERIOD), "res": None}
    for seasonal in seasonal_grid:
        for p in range(p_max + 1):
            for q in range(q_max + 1):
                if p == 0 and q == 0 and seasonal[0] == 0 and seasonal[2] == 0:
                    continue
                try:
                    result = SARIMAX(
                        series,
                        order=(p, d, q),
                        seasonal_order=seasonal,
                        enforce_stationarity=False,
                        enforce_invertibility=False,
                    ).fit(disp=False, maxiter=80)
                except Exception:
                    continue
                if np.isfinite(result.aic) and result.aic < best["aic"]:
                    best = {
                        "aic": float(result.aic),
                        "order": (p, d, q),
                        "seasonal": seasonal,
                        "res": result,
                    }

    if best["res"] is None:  # degenerate fallback
        best["res"] = SARIMAX(series, order=(1, d, 1)).fit(disp=False)
        best["aic"] = float(best["res"].aic)
    logger.info(
        "  SARIMA%s x %s selected (AIC %.1f)", best["order"], best["seasonal"], best["aic"]
    )
    return best["res"], best["order"], best["seasonal"], best["aic"]


def sarima_test_predictions(result, train: pd.Series, test: pd.Series) -> pd.DataFrame:
    """Produce genuine one-step-ahead SARIMA predictions over the test window."""
    try:
        extended = result.append(test, refit=False)
        prediction = extended.get_prediction(start=test.index[0], end=test.index[-1])
        mean = prediction.predicted_mean
        interval = prediction.conf_int(alpha=0.05)
        return pd.DataFrame(
            {
                "actual": test.values,
                "pred": mean.values,
                "lower": interval.iloc[:, 0].values,
                "upper": interval.iloc[:, 1].values,
            },
            index=test.index,
        )
    except Exception as exc:
        logger.warning("  SARIMA append failed (%s); falling back to dynamic forecast.", exc)
        forecast = result.get_forecast(steps=len(test))
        interval = forecast.conf_int(alpha=0.05)
        return pd.DataFrame(
            {
                "actual": test.values,
                "pred": forecast.predicted_mean.values,
                "lower": interval.iloc[:, 0].values,
                "upper": interval.iloc[:, 1].values,
            },
            index=test.index,
        )


# ==========================================================================
# Phase 6 — XGBoost
# ==========================================================================
def fit_xgboost(train: pd.DataFrame, columns: list[str], quick: bool = False):
    """Fit the delta-target XGBoost regressor on the training block.

    The model predicts the *change* from yesterday rather than the price level,
    which keeps it usable when future prices leave the training range.
    """
    import xgboost as xgb

    params = {
        "n_estimators": 300 if quick else 700,
        "learning_rate": 0.045,
        "max_depth": 5,
        "subsample": 0.85,
        "colsample_bytree": 0.8,
        "min_child_weight": 3,
        "reg_lambda": 1.4,
        "reg_alpha": 0.1,
    }
    model = xgb.XGBRegressor(
        objective="reg:squarederror",
        tree_method="hist",
        random_state=SEED,
        n_jobs=-1,
        **params,
    )
    model.fit(train[columns], train["price"] - train["lag_1"], verbose=False)
    return model


def xgb_to_level(model, rows: pd.DataFrame, columns: list[str]) -> np.ndarray:
    """Convert delta predictions back to price levels."""
    return model.predict(rows[columns]) + rows["lag_1"].to_numpy()


# ==========================================================================
# Phase 7 — LSTM (PyTorch)
# ==========================================================================
def make_sequences(values: np.ndarray, lookback: int) -> tuple[np.ndarray, np.ndarray]:
    """Turn a scaled 1-D series into ``(samples, lookback, 1)`` sequences."""
    windows, targets = [], []
    for i in range(lookback, len(values)):
        windows.append(values[i - lookback : i])
        targets.append(values[i])
    return np.asarray(windows, dtype=np.float32), np.asarray(targets, dtype=np.float32)


@dataclass
class LstmBundle:
    """A trained LSTM together with the scaler it was fitted with."""

    model: object
    scaler: object
    residuals: np.ndarray


def fit_lstm(series: pd.Series, epochs: int = 120, patience: int = 15) -> LstmBundle | None:
    """Train the sequence model, returning ``None`` when PyTorch is unavailable.

    Args:
        series: Price series to fit on.
        epochs: Maximum training epochs.
        patience: Early-stopping patience on the validation loss.
    """
    try:
        import torch
        from torch import nn
    except ImportError:
        logger.warning("  PyTorch not installed — skipping the LSTM.")
        return None

    from sklearn.preprocessing import MinMaxScaler

    torch.manual_seed(SEED)
    np.random.seed(SEED)

    scaler = MinMaxScaler((0, 1))
    scaled = scaler.fit_transform(series.to_numpy().reshape(-1, 1)).flatten()
    windows, targets = make_sequences(scaled, LOOKBACK)
    windows = windows.reshape(-1, LOOKBACK, 1)

    split = max(int(len(windows) * 0.85), 1)
    x_train = torch.from_numpy(windows[:split])
    y_train = torch.from_numpy(targets[:split]).unsqueeze(1)
    x_val = torch.from_numpy(windows[split:])
    y_val = torch.from_numpy(targets[split:]).unsqueeze(1)

    class Net(nn.Module):
        """Two stacked LSTM layers with dropout and a small dense head."""

        def __init__(self) -> None:
            super().__init__()
            self.lstm1 = nn.LSTM(1, 64, batch_first=True)
            self.drop1 = nn.Dropout(0.2)
            self.lstm2 = nn.LSTM(64, 32, batch_first=True)
            self.drop2 = nn.Dropout(0.2)
            self.head = nn.Sequential(nn.Linear(32, 16), nn.ReLU(), nn.Linear(16, 1))

        def forward(self, x):  # noqa: D102 - standard torch signature
            out, _ = self.lstm1(x)
            out = self.drop1(out)
            out, _ = self.lstm2(out)
            return self.head(self.drop2(out[:, -1, :]))

    model = Net()
    optimiser = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()
    batch = 32

    best_state, best_loss, stale = None, float("inf"), 0
    for epoch in range(epochs):
        model.train()
        order = torch.randperm(len(x_train))
        for start in range(0, len(x_train), batch):
            idx = order[start : start + batch]
            optimiser.zero_grad()
            loss = loss_fn(model(x_train[idx]), y_train[idx])
            loss.backward()
            optimiser.step()

        model.eval()
        with torch.no_grad():
            val_loss = float(loss_fn(model(x_val), y_val)) if len(x_val) else float(loss)
        if val_loss < best_loss - 1e-6:
            best_loss, stale = val_loss, 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            stale += 1
            if stale >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()

    with torch.no_grad():
        fitted = model(torch.from_numpy(windows)).numpy().flatten()
    residuals = (
        scaler.inverse_transform(targets.reshape(-1, 1)).flatten()
        - scaler.inverse_transform(fitted.reshape(-1, 1)).flatten()
    )
    logger.info("  LSTM trained (%d epochs, best val MSE %.6f)", epoch + 1, best_loss)
    return LstmBundle(model=model, scaler=scaler, residuals=residuals)


def lstm_predict(bundle: LstmBundle, windows: np.ndarray) -> np.ndarray:
    """Run the LSTM over pre-built scaled windows and return price levels."""
    import torch

    with torch.no_grad():
        scaled = bundle.model(torch.from_numpy(windows.astype(np.float32))).numpy().flatten()
    return bundle.scaler.inverse_transform(scaled.reshape(-1, 1)).flatten()


# ==========================================================================
# Phase 9 — recursive forecasting
# ==========================================================================
def add_fluctuation(
    values: np.ndarray, lower: np.ndarray, upper: np.ndarray, history: pd.Series, seed: int
) -> np.ndarray:
    """Overlay realistic day-to-day movement on a smooth forecast path.

    The noise is scaled to the series' own recent daily volatility, mean-centred
    so the horizon-end level and the inflation figures stay unbiased, and
    clipped inside the model's own interval so it never contradicts the quoted
    uncertainty.
    """
    rng = np.random.default_rng(seed)
    daily_volatility = history.diff().dropna().tail(180).std()
    if not np.isfinite(daily_volatility) or daily_volatility == 0:
        return values
    noise = rng.normal(0.0, 0.35 * float(daily_volatility), size=len(values))
    noise -= noise.mean()
    half_band = np.maximum((upper - lower) / 2.0, 1e-6)
    return np.clip(values + noise, values - half_band, values + half_band)


def forecast_sarima(result, steps: int, history: pd.Series, seed: int) -> pd.DataFrame:
    """Forecast forward with SARIMA's analytic prediction interval."""
    forecast = result.get_forecast(steps=steps)
    interval = forecast.conf_int(alpha=0.05)
    values = forecast.predicted_mean.to_numpy()
    lower = interval.iloc[:, 0].to_numpy()
    upper = interval.iloc[:, 1].to_numpy()
    index = pd.date_range(history.index[-1] + pd.Timedelta(days=1), periods=steps, freq="D")
    return pd.DataFrame(
        {"forecast": add_fluctuation(values, lower, upper, history, seed),
         "lower": lower, "upper": upper},
        index=index,
    )


def forecast_xgboost(
    model, history: pd.Series, steps: int, residuals: np.ndarray, columns: list[str], seed: int
) -> pd.DataFrame:
    """Recursive multi-step XGBoost forecast, rebuilding features each step."""
    series = history.copy()
    predictions: list[float] = []
    for _ in range(steps):
        next_date = series.index[-1] + pd.Timedelta(days=1)
        extended = pd.concat([series, pd.Series([np.nan], index=[next_date])])
        row = build_features(extended).iloc[[-1]]
        if row[columns].isna().to_numpy().any():
            row = row.fillna(0.0)
        value = float(xgb_to_level(model, row, columns)[0])
        predictions.append(value)
        series = pd.concat([series, pd.Series([value], index=[next_date])])

    values = np.asarray(predictions)
    spread = 1.96 * float(np.std(residuals)) * np.sqrt(np.arange(1, steps + 1))
    lower, upper = values - spread, values + spread
    index = pd.date_range(history.index[-1] + pd.Timedelta(days=1), periods=steps, freq="D")
    return pd.DataFrame(
        {"forecast": add_fluctuation(values, lower, upper, history, seed),
         "lower": lower, "upper": upper},
        index=index,
    )


def forecast_lstm(bundle: LstmBundle, history: pd.Series, steps: int, seed: int) -> pd.DataFrame:
    """Recursive multi-step LSTM forecast in scaled space."""
    scaled = bundle.scaler.transform(history.to_numpy().reshape(-1, 1)).flatten()
    window = list(scaled[-LOOKBACK:])
    scaled_predictions: list[float] = []
    for _ in range(steps):
        block = np.asarray(window[-LOOKBACK:], dtype=np.float32).reshape(1, LOOKBACK, 1)
        value = float(lstm_predict_scaled(bundle, block))
        window.append(value)
        scaled_predictions.append(value)

    values = bundle.scaler.inverse_transform(
        np.asarray(scaled_predictions).reshape(-1, 1)
    ).flatten()
    spread = 1.96 * float(np.std(bundle.residuals)) * np.sqrt(np.arange(1, steps + 1))
    lower, upper = values - spread, values + spread
    index = pd.date_range(history.index[-1] + pd.Timedelta(days=1), periods=steps, freq="D")
    return pd.DataFrame(
        {"forecast": add_fluctuation(values, lower, upper, history, seed),
         "lower": lower, "upper": upper},
        index=index,
    )


def lstm_predict_scaled(bundle: LstmBundle, window: np.ndarray) -> float:
    """Predict one step ahead in scaled space."""
    import torch

    with torch.no_grad():
        return float(bundle.model(torch.from_numpy(window)).numpy().flatten()[0])


# ==========================================================================
# Narrative rule base
# ==========================================================================
COMMODITY_CONTEXT: dict[str, dict[str, str]] = {
    "Onion": {
        "drivers": "Onion is the most spike-prone item in the basket: domestic supply is "
        "concentrated in a short harvest, and India's export policy transmits to Dhaka "
        "bazaars within days.",
        "policy": "Onion price shocks in Bangladesh are historically import-policy events. "
        "Coordinate with the Ministry of Commerce on alternative sourcing (Türkiye, Egypt, "
        "China) before the domestic lean season, not during it.",
        "consumer": "Onion stores well in a dry, ventilated place for several weeks, so it is "
        "one of the few basket items where buying ahead of a forecast spike genuinely pays.",
    },
    "Potato": {
        "drivers": "Potato is the most storable staple in the basket, so its price is set less "
        "by daily supply than by cold-storage release decisions and carrying costs.",
        "policy": "Cold-storage capacity and release timing are the effective policy levers for "
        "potato. Monitor storage occupancy alongside price — a full store with rising prices "
        "signals withholding, not scarcity.",
        "consumer": "Potato keeps for months in a cool dark place, so bulk buying during a "
        "projected soft patch is low-risk for most households.",
    },
    "Soybean Oil": {
        "drivers": "Soybean oil is almost entirely imported, so its retail price tracks global "
        "vegetable-oil markets and the taka exchange rate more closely than local conditions.",
        "policy": "Because soybean oil is import-dependent, duty adjustments and letter-of-credit "
        "facilitation act faster than stock releases. Watch the exchange rate as a leading "
        "indicator of the retail price.",
        "consumer": "Bottled soybean oil has a long shelf life, so buying an extra month's supply "
        "ahead of a projected rise carries almost no spoilage risk.",
    },
}


def build_recommendation_rules() -> dict:
    """Return the regime-keyed phrasing rule base written to ``recommendations.json``."""
    return {
        "schema": 1,
        "description": (
            "Regime-keyed recommendation templates. The dashboard selects a regime from the "
            "selected product's expected price change, then formats the body with live numbers. "
            "Placeholders: product, unit, currency, current, forecast, lower, upper, change, "
            "horizon, date, model, volatility, risk, saving."
        ),
        "policy": {
            "rising_sharp": [
                {
                    "title": "Release buffer stocks now",
                    "body": "A {change:+.1f}% move in {product} over {horizon} days hits "
                    "low-income households directly. Trigger open-market sales through TCB "
                    "outlets in the districts with the thinnest supply before the projected "
                    "peak on {date}.",
                    "icon": "alert",
                    "tone": "danger",
                },
                {
                    "title": "Fast-track import clearance",
                    "body": "Waive or reduce duty and clear consignments of {product} within "
                    "days rather than weeks. Import lead time is the binding constraint when "
                    "the projected peak is only {horizon} days out.",
                    "icon": "zap",
                    "tone": "warning",
                },
                {
                    "title": "Publish a weekly price bulletin",
                    "body": "Announce the projected band of {lower:,.0f}–{upper:,.0f} {unit} "
                    "publicly. Transparent official forecasts reduce the speculative hoarding "
                    "that drives a large part of short-run retail spikes.",
                    "icon": "file",
                    "tone": "primary",
                },
            ],
            "rising": [
                {
                    "title": "Put buffer stocks on standby",
                    "body": "The {model} model points to a {change:+.1f}% rise in {product} by "
                    "{date}. Confirm warehouse positions now so an intervention can be executed "
                    "within a week if realised prices track the upper band of {upper:,.0f} {unit}.",
                    "icon": "shield",
                    "tone": "warning",
                },
                {
                    "title": "Monitor wholesale-to-retail margins",
                    "body": "Moderate upward pressure is where margin widening hides. Compare "
                    "wholesale and bazaar prices for {product} weekly across the {horizon}-day "
                    "window and publish the gap.",
                    "icon": "target",
                    "tone": "primary",
                },
                {
                    "title": "Pre-position social protection",
                    "body": "Confirm OMS and VGF distribution lists now. A {change:+.1f}% rise is "
                    "absorbable for most households but not for the bottom two income deciles.",
                    "icon": "policy",
                    "tone": "primary",
                },
            ],
            "stable": [
                {
                    "title": "Use the window to rebuild stocks",
                    "body": "{product} is projected to stay within {change:+.1f}% of today's "
                    "{current:,.2f} {unit}. A flat market is the cheapest moment to replenish "
                    "public buffer stocks ahead of the next lean season.",
                    "icon": "check",
                    "tone": "success",
                },
                {
                    "title": "Hold interventions in reserve",
                    "body": "No price action is warranted while the forecast band stays inside "
                    "{lower:,.0f}–{upper:,.0f} {unit}. Preserve fiscal space for the commodities "
                    "that are actually moving.",
                    "icon": "compass",
                    "tone": "primary",
                },
                {
                    "title": "Invest in the data pipeline",
                    "body": "Quiet periods are when monitoring systems should be improved. "
                    "Extending daily price collection to more upazilas raises forecast quality "
                    "for the next shock at almost no political cost.",
                    "icon": "database",
                    "tone": "neutral",
                },
            ],
            "falling": [
                {
                    "title": "Protect farm-gate prices",
                    "body": "A projected {change:+.1f}% decline in {product} helps consumers but "
                    "squeezes growers. Consider procurement at a floor price so the correction "
                    "does not depress next season's planting decisions.",
                    "icon": "leaf",
                    "tone": "success",
                },
                {
                    "title": "Expand cold-storage access",
                    "body": "Cheap {product} is only useful if it can be held. Subsidised storage "
                    "during the {horizon}-day soft patch smooths the following peak far more "
                    "cheaply than emergency imports do.",
                    "icon": "layers",
                    "tone": "primary",
                },
                {
                    "title": "Build the public stock now",
                    "body": "Procuring into the buffer at {forecast:,.2f} {unit} rather than at a "
                    "crisis price is the single highest-return action available in a falling "
                    "market.",
                    "icon": "shield",
                    "tone": "primary",
                },
            ],
        },
        "consumer": {
            "rising_sharp": [
                {
                    "title": "Buy ahead of the peak",
                    "body": "{product} is projected to reach {forecast:,.2f} {unit} by {date}, up "
                    "{change:+.1f}% from {current:,.2f}. If it stores well, buying now avoids "
                    "roughly {saving:,.0f} {currency} on a month of typical household use.",
                    "icon": "cart",
                    "tone": "danger",
                },
                {
                    "title": "Substitute where you can",
                    "body": "During a sharp {product} run-up, shifting part of the basket to items "
                    "forecast to stay flat protects the food budget without cutting the number "
                    "of meals.",
                    "icon": "layers",
                    "tone": "warning",
                },
                {
                    "title": "Buy in bulk with neighbours",
                    "body": "Wholesale rates are typically 8–15% below retail. Pooling a single "
                    "sack of {product} across households captures that gap exactly when the "
                    "retail price is climbing.",
                    "icon": "cart",
                    "tone": "primary",
                },
            ],
            "rising": [
                {
                    "title": "Stagger your purchases",
                    "body": "With a {change:+.1f}% rise expected by {date}, buying in two or three "
                    "tranches over the next {horizon} days beats one large purchase — you average "
                    "into the move instead of trying to time it.",
                    "icon": "cart",
                    "tone": "warning",
                },
                {
                    "title": "Plan against the upper band",
                    "body": "The 95% interval for {product} reaches {upper:,.0f} {unit}. Treat "
                    "that, not the central forecast, as the worst case when setting a monthly "
                    "food budget.",
                    "icon": "shield",
                    "tone": "primary",
                },
            ],
            "stable": [
                {
                    "title": "No need to stockpile",
                    "body": "{product} is expected to stay near {forecast:,.2f} {unit} through "
                    "{date}. Normal weekly buying is the cheapest strategy; stockpiling only adds "
                    "spoilage risk.",
                    "icon": "check",
                    "tone": "success",
                },
                {
                    "title": "Compare outlets instead",
                    "body": "When the market is flat, the spread between bazaars is larger than "
                    "the spread over time. Shopping around beats timing the purchase.",
                    "icon": "compass",
                    "tone": "primary",
                },
            ],
            "falling": [
                {
                    "title": "Wait before bulk buying",
                    "body": "{product} is projected to ease to {forecast:,.2f} {unit} by {date} "
                    "({change:+.1f}%). Deferring a bulk purchase by a few weeks saves about "
                    "{saving:,.0f} {currency} on a month of typical use.",
                    "icon": "trend_down",
                    "tone": "success",
                },
                {
                    "title": "Good moment to preserve",
                    "body": "A soft patch in {product} is the right time to buy for drying, "
                    "pickling or freezing if your household does that.",
                    "icon": "leaf",
                    "tone": "primary",
                },
            ],
        },
        "commodity_notes": {
            key: {"policy": value["policy"], "consumer": value["consumer"]}
            for key, value in COMMODITY_CONTEXT.items()
        },
    }


# ==========================================================================
# Orchestration
# ==========================================================================
def process_commodity(
    key: str, raw_file: str, args: argparse.Namespace
) -> dict[str, object] | None:
    """Run the full pipeline for one commodity.

    Args:
        key: Canonical commodity key.
        raw_file: File name inside ``Raw Data/``.
        args: Parsed command-line options.

    Returns:
        A dictionary of per-commodity results, or ``None`` when the raw file is
        missing.
    """
    source = RAW_DATA_DIR / raw_file
    if not source.exists():
        logger.error("Raw file not found: %s", source)
        return None

    logger.info("=== %s ===", key)
    prices = clean_series(load_raw(source))
    features = build_features(prices).dropna()
    columns = feature_columns(features)

    cutoff = features.index[-args.test_days - 1]
    train_features = features.loc[:cutoff]
    test_features = features.loc[features.index > cutoff]
    train_prices = prices.loc[:cutoff]
    test_prices = prices.loc[prices.index > cutoff]

    predictions: dict[str, pd.DataFrame] = {}

    # ---- SARIMA -----------------------------------------------------
    sarima_result, order, seasonal, aic = fit_sarima(train_prices, quick=args.quick)
    predictions["SARIMA"] = sarima_test_predictions(sarima_result, train_prices, test_prices)

    # ---- XGBoost ----------------------------------------------------
    xgb_model = fit_xgboost(train_features, columns, quick=args.quick)
    xgb_test = xgb_to_level(xgb_model, test_features, columns)
    xgb_train_residual = train_features["price"].to_numpy() - xgb_to_level(
        xgb_model, train_features, columns
    )
    spread = 1.96 * float(np.std(xgb_train_residual))
    predictions["XGBoost"] = pd.DataFrame(
        {
            "actual": test_features["price"].to_numpy(),
            "pred": xgb_test,
            "lower": xgb_test - spread,
            "upper": xgb_test + spread,
        },
        index=test_features.index,
    )
    logger.info("  XGBoost trained on %d rows", len(train_features))

    # ---- LSTM -------------------------------------------------------
    lstm_bundle = None
    if not args.skip_lstm:
        lstm_bundle = fit_lstm(train_prices, epochs=25 if args.quick else 120)
    if lstm_bundle is not None:
        scaled_all = lstm_bundle.scaler.transform(prices.to_numpy().reshape(-1, 1)).flatten()
        positions = [prices.index.get_loc(ts) for ts in test_prices.index]
        windows = np.asarray(
            [scaled_all[pos - LOOKBACK : pos] for pos in positions], dtype=np.float32
        ).reshape(-1, LOOKBACK, 1)
        lstm_test = lstm_predict(lstm_bundle, windows)
        lstm_spread = 1.96 * float(np.std(lstm_bundle.residuals))
        predictions["LSTM"] = pd.DataFrame(
            {
                "actual": test_prices.to_numpy(),
                "pred": lstm_test,
                "lower": lstm_test - lstm_spread,
                "upper": lstm_test + lstm_spread,
            },
            index=test_prices.index,
        )

    # ---- align every model on the same index ------------------------
    shared_index = predictions["XGBoost"].index
    for name, frame in list(predictions.items()):
        predictions[name] = frame.reindex(shared_index)

    # ---- weighted ensemble ------------------------------------------
    base_models = [m for m in ("SARIMA", "XGBoost", "LSTM") if m in predictions]
    inverse_mae = {
        name: 1.0
        / max(regression_metrics(predictions[name]["actual"], predictions[name]["pred"])["MAE"], 1e-9)
        for name in base_models
    }
    total = sum(inverse_mae.values())
    weights = {name: value / total for name, value in inverse_mae.items()}
    predictions["Ensemble"] = pd.DataFrame(
        {
            "actual": predictions[base_models[0]]["actual"],
            "pred": sum(weights[n] * predictions[n]["pred"] for n in base_models),
            "lower": sum(weights[n] * predictions[n]["lower"] for n in base_models),
            "upper": sum(weights[n] * predictions[n]["upper"] for n in base_models),
        },
        index=shared_index,
    )
    logger.info("  ensemble weights: %s", {k: round(v, 3) for k, v in weights.items()})

    # ---- seasonal naive baseline ------------------------------------
    naive = prices.shift(SEASONAL_PERIOD).reindex(shared_index)
    predictions["Seasonal Naive"] = pd.DataFrame(
        {
            "actual": prices.reindex(shared_index).to_numpy(),
            "pred": naive.to_numpy(),
            "lower": np.nan,
            "upper": np.nan,
        },
        index=shared_index,
    )

    # ---- scoring ----------------------------------------------------
    scores = {
        name: regression_metrics(frame["actual"], frame["pred"])
        for name, frame in predictions.items()
    }
    baseline_mae = scores["Seasonal Naive"]["MAE"]
    ranked = sorted(
        (n for n in scores if n != "Seasonal Naive"), key=lambda n: scores[n]["MAE"]
    )
    best_model = ranked[0]
    logger.info("  best model: %s (MAE %.4f)", best_model, scores[best_model]["MAE"])

    # ---- refit on the full series and forecast forward --------------
    logger.info("  refitting on the full series for the forward forecast…")
    from statsmodels.tsa.statespace.sarimax import SARIMAX

    full_sarima = SARIMAX(
        prices, order=order, seasonal_order=seasonal,
        enforce_stationarity=False, enforce_invertibility=False,
    ).fit(disp=False)

    full_features = build_features(prices).dropna()
    full_xgb = fit_xgboost(full_features, columns, quick=args.quick)
    full_xgb_residual = full_features["price"].to_numpy() - xgb_to_level(
        full_xgb, full_features, columns
    )

    forecasts: dict[str, pd.DataFrame] = {
        "SARIMA": forecast_sarima(full_sarima, args.max_horizon, prices, SEED + 1),
        "XGBoost": forecast_xgboost(
            full_xgb, prices, args.max_horizon, full_xgb_residual, columns, SEED + 2
        ),
    }
    if lstm_bundle is not None:
        full_lstm = fit_lstm(prices, epochs=25 if args.quick else 120) or lstm_bundle
        forecasts["LSTM"] = forecast_lstm(full_lstm, prices, args.max_horizon, SEED + 3)

    ensemble_models = [m for m in weights if m in forecasts]
    ensemble_weight_total = sum(weights[m] for m in ensemble_models)
    forecasts["Ensemble"] = pd.DataFrame(
        {
            column: sum(
                weights[m] / ensemble_weight_total * forecasts[m][column] for m in ensemble_models
            )
            for column in ("forecast", "lower", "upper")
        },
        index=forecasts["SARIMA"].index,
    )

    importance = pd.DataFrame(
        {
            "commodity": key,
            "feature": columns,
            "importance": full_xgb.feature_importances_.astype(float),
        }
    ).sort_values("importance", ascending=False)

    return {
        "key": key,
        "prices": prices,
        "predictions": predictions,
        "scores": scores,
        "baseline_mae": baseline_mae,
        "best_model": best_model,
        "forecasts": forecasts,
        "importance": importance,
        "weights": weights,
        "sarima_order": {"order": list(order), "seasonal_order": list(seasonal), "aic": aic},
    }


def write_artifacts(results: list[dict[str, object]], args: argparse.Namespace) -> None:
    """Write every dashboard artifact to ``data/processed``."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    # ---- historical price -------------------------------------------
    price_rows = pd.concat(
        [
            pd.DataFrame({"date": r["prices"].index, "commodity": r["key"],
                          "price": r["prices"].to_numpy().round(4)})
            for r in results
        ],
        ignore_index=True,
    )
    price_rows.to_csv(DATA_DIR / "historical_price.csv", index=False)

    # ---- historical inflation ---------------------------------------
    inflation_frames = []
    for result in results:
        series = result["prices"]
        inflation_frames.append(
            pd.DataFrame(
                {
                    "date": series.index,
                    "commodity": result["key"],
                    "inflation_daily": (series.pct_change(1) * 100).to_numpy().round(4),
                    "inflation_weekly": (series.pct_change(7) * 100).to_numpy().round(4),
                    "inflation_monthly": (series.pct_change(30) * 100).to_numpy().round(4),
                    "inflation_annual": (series.pct_change(365) * 100).to_numpy().round(4),
                }
            )
        )
    pd.concat(inflation_frames, ignore_index=True).to_csv(
        DATA_DIR / "historical_inflation.csv", index=False
    )

    # ---- forecast ----------------------------------------------------
    forecast_rows = []
    for result in results:
        for model, frame in result["forecasts"].items():
            forecast_rows.append(
                pd.DataFrame(
                    {
                        "date": frame.index,
                        "commodity": result["key"],
                        "model": model,
                        "horizon_day": np.arange(1, len(frame) + 1),
                        "forecast": frame["forecast"].to_numpy().round(4),
                        "lower": frame["lower"].to_numpy().round(4),
                        "upper": frame["upper"].to_numpy().round(4),
                        "is_best_model": model == result["best_model"],
                    }
                )
            )
    pd.concat(forecast_rows, ignore_index=True).to_csv(DATA_DIR / "forecast.csv", index=False)

    # ---- metrics -----------------------------------------------------
    metric_rows = []
    for result in results:
        for model, score in result["scores"].items():
            baseline = result["baseline_mae"]
            metric_rows.append(
                {
                    "commodity": result["key"],
                    "model": model,
                    "MAE": round(score["MAE"], 6),
                    "RMSE": round(score["RMSE"], 6),
                    "MAPE": round(score["MAPE"], 6),
                    "skill_vs_naive": round((1 - score["MAE"] / baseline) * 100, 4)
                    if baseline
                    else np.nan,
                    "is_best_model": model == result["best_model"],
                }
            )
    pd.DataFrame(metric_rows).to_csv(DATA_DIR / "metrics.csv", index=False)

    # ---- model comparison -------------------------------------------
    comparison_rows = []
    for result in results:
        for model, frame in result["predictions"].items():
            comparison_rows.append(
                pd.DataFrame(
                    {
                        "date": frame.index,
                        "commodity": result["key"],
                        "model": model,
                        "actual": frame["actual"].to_numpy().round(4),
                        "predicted": frame["pred"].to_numpy().round(4),
                    }
                )
            )
    pd.concat(comparison_rows, ignore_index=True).dropna(subset=["predicted"]).to_csv(
        DATA_DIR / "model_comparison.csv", index=False
    )

    # ---- feature importance -----------------------------------------
    pd.concat([r["importance"] for r in results], ignore_index=True).to_csv(
        DATA_DIR / "feature_importance.csv", index=False
    )

    # ---- insights ----------------------------------------------------
    last_date = max(r["prices"].index[-1] for r in results)
    forecast_start = min(r["forecasts"]["SARIMA"].index[0] for r in results)
    forecast_end = max(r["forecasts"]["SARIMA"].index[-1] for r in results)

    insights = {
        "schema": 1,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "dataset": {
            "last_observation": str(last_date.date()),
            "forecast_start": str(forecast_start.date()),
            "forecast_end": str(forecast_end.date()),
            "max_horizon_days": args.max_horizon,
            "test_days": args.test_days,
        },
        "best_model": {r["key"]: r["best_model"] for r in results},
        "ensemble_weights": {
            r["key"]: {k: round(v, 4) for k, v in r["weights"].items()} for r in results
        },
        "sarima_orders": {r["key"]: r["sarima_order"] for r in results},
        "commodities": {
            r["key"]: {
                "drivers": COMMODITY_CONTEXT.get(r["key"], {}).get("drivers", ""),
                "last_price": round(float(r["prices"].iloc[-1]), 2),
                "sample_mean": round(float(r["prices"].mean()), 2),
                "sample_start": str(r["prices"].index[0].date()),
                "volatility_30d": round(
                    float(r["prices"].pct_change().tail(30).std() * 100), 4
                ),
            }
            for r in results
        },
    }
    (DATA_DIR / "insights.json").write_text(json.dumps(insights, indent=2), encoding="utf-8")

    (DATA_DIR / "recommendations.json").write_text(
        json.dumps(build_recommendation_rules(), indent=2), encoding="utf-8"
    )

    logger.info("Artifacts written to %s", DATA_DIR)
    for path in sorted(DATA_DIR.glob("*")):
        logger.info("  %-28s %8.1f KB", path.name, path.stat().st_size / 1024)


def main() -> int:
    """Command-line entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-horizon", type=int, default=60, help="forecast length in days")
    parser.add_argument("--test-days", type=int, default=180, help="held-out evaluation days")
    parser.add_argument("--skip-lstm", action="store_true", help="skip the PyTorch LSTM")
    parser.add_argument("--quick", action="store_true", help="small search space, fast run")
    args = parser.parse_args()

    logger.info("Building dashboard artifacts (horizon=%d, test=%d days)",
                args.max_horizon, args.test_days)

    results = []
    for commodity in COMMODITIES:
        try:
            result = process_commodity(commodity.key, commodity.raw_file, args)
        except Exception:
            logger.exception("Failed to process %s", commodity.key)
            continue
        if result is not None:
            results.append(result)

    if not results:
        logger.error("No commodity could be processed — nothing was written.")
        return 1

    write_artifacts(results, args)
    logger.info("Done. Start the dashboard with: streamlit run streamlit_app.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
