"""Central configuration for the price-forecasting dashboard.

Every tunable value — paths, palette, commodity registry, thresholds — lives
here so that no other module hard-codes a colour, a file name or a business
rule. Import ``SETTINGS`` (a frozen dataclass instance) rather than reaching
for module-level globals in application code.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Final

# --------------------------------------------------------------------------
# Filesystem layout
# --------------------------------------------------------------------------
PROJECT_ROOT: Final[Path] = Path(__file__).resolve().parent.parent
DATA_DIR: Final[Path] = PROJECT_ROOT / "data" / "processed"
RAW_DATA_DIR: Final[Path] = PROJECT_ROOT / "Raw Data"
MODEL_DIR: Final[Path] = PROJECT_ROOT / "data" / "models"
EXPORT_DIR: Final[Path] = PROJECT_ROOT / "data" / "exports"
LOG_DIR: Final[Path] = PROJECT_ROOT / "logs"

#: Artifact files the dashboard consumes. ``True`` marks a hard requirement —
#: the app refuses to start without it; ``False`` marks an optional enrichment.
REQUIRED_ARTIFACTS: Final[dict[str, bool]] = {
    "historical_price.csv": True,
    "forecast.csv": True,
    "historical_inflation.csv": True,
    "metrics.csv": True,
    "model_comparison.csv": True,
    "feature_importance.csv": False,
    "insights.json": False,
    "recommendations.json": False,
}


# --------------------------------------------------------------------------
# Commodity registry
# --------------------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class Commodity:
    """A single tracked essential good.

    Attributes:
        key: Canonical name used inside every exported artifact.
        label: Human-readable name rendered in the UI.
        icon: Material Symbols identifier used on cards and navigation.
        color: Hex colour that identifies the commodity across all charts.
        unit: Quoted retail unit, appended to price labels.
        raw_file: Source CSV in ``Raw Data/`` (used only by the export script).
    """

    key: str
    label: str
    icon: str
    color: str
    unit: str
    raw_file: str


COMMODITIES: Final[tuple[Commodity, ...]] = (
    Commodity("Onion", "Onion", ":material/nutrition:", "#DC6803", "BDT / kg", "Onion.csv"),
    Commodity("Potato", "Potato", ":material/agriculture:", "#16A34A", "BDT / kg", "Potato.csv"),
    Commodity(
        "Soybean Oil",
        "Soybean oil",
        ":material/water_drop:",
        "#2563EB",
        "BDT / litre",
        "Soybean.csv",
    ),
)

COMMODITY_KEYS: Final[tuple[str, ...]] = tuple(c.key for c in COMMODITIES)
COMMODITY_BY_KEY: Final[dict[str, Commodity]] = {c.key: c for c in COMMODITIES}
COMMODITY_COLORS: Final[dict[str, str]] = {c.key: c.color for c in COMMODITIES}


# --------------------------------------------------------------------------
# Models
# --------------------------------------------------------------------------
#: Display order for every model comparison table and chart.
MODEL_ORDER: Final[tuple[str, ...]] = (
    "SARIMA",
    "XGBoost",
    "LSTM",
    "Ensemble",
    "Seasonal Naive",
)

#: The four models the study actually compares; the naive series is a baseline.
FORECAST_MODELS: Final[tuple[str, ...]] = ("SARIMA", "XGBoost", "LSTM", "Ensemble")

MODEL_COLORS: Final[dict[str, str]] = {
    "SARIMA": "#DC2626",
    "XGBoost": "#16A34A",
    "LSTM": "#7C3AED",
    "Ensemble": "#2563EB",
    "Seasonal Naive": "#94A3B8",
}

MODEL_DESCRIPTIONS: Final[dict[str, str]] = {
    "SARIMA": "Seasonal ARIMA — linear statistical benchmark; the only model with "
    "analytically derived prediction intervals.",
    "XGBoost": "Gradient-boosted trees over 60+ engineered calendar, lag, rolling "
    "and volatility features.",
    "LSTM": "Recurrent neural network reading a 30-day sequence window of scaled prices.",
    "Ensemble": "Inverse-MAE weighted blend of SARIMA, XGBoost and LSTM.",
    "Seasonal Naive": "Baseline: tomorrow equals the value seven days ago.",
}


# --------------------------------------------------------------------------
# Colour system
# --------------------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class Palette:
    """Semantic colours for one presentation mode."""

    background: str
    surface: str
    surface_alt: str
    sidebar: str
    sidebar_text: str
    border: str
    text: str
    text_muted: str
    primary: str
    success: str
    warning: str
    danger: str
    grid: str
    shadow: str


LIGHT_PALETTE: Final[Palette] = Palette(
    background="#F8FAFC",
    surface="#FFFFFF",
    surface_alt="#F1F5F9",
    sidebar="#0F172A",
    sidebar_text="#E2E8F0",
    border="#E2E8F0",
    text="#1E293B",
    text_muted="#64748B",
    primary="#2563EB",
    success="#16A34A",
    warning="#F59E0B",
    danger="#DC2626",
    grid="#E9EEF5",
    shadow="0 1px 2px rgba(15,23,42,.06), 0 8px 24px -12px rgba(15,23,42,.18)",
)

DARK_PALETTE: Final[Palette] = Palette(
    background="#0B1220",
    surface="#111C31",
    surface_alt="#16243D",
    sidebar="#070E1B",
    sidebar_text="#CBD5E1",
    border="#1E2E4A",
    text="#E2E8F0",
    text_muted="#94A3B8",
    primary="#60A5FA",
    success="#4ADE80",
    warning="#FBBF24",
    danger="#F87171",
    grid="#1E2E4A",
    shadow="0 1px 2px rgba(0,0,0,.4), 0 10px 28px -14px rgba(0,0,0,.7)",
)

PALETTES: Final[dict[str, Palette]] = {"Light": LIGHT_PALETTE, "Dark": DARK_PALETTE}


# --------------------------------------------------------------------------
# Business rules
# --------------------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class Thresholds:
    """Cut-offs that turn continuous numbers into decision language.

    All values are percentages unless stated otherwise.
    """

    #: |expected change| below this is reported as "broadly stable".
    stable_change: float = 1.0
    #: |expected change| above this is reported as a material move.
    material_change: float = 5.0
    #: 30-day realised volatility above this marks a commodity as high risk.
    high_volatility: float = 2.5
    #: 30-day realised volatility below this marks a commodity as stable.
    low_volatility: float = 1.0
    #: Horizon inflation above this triggers an intervention recommendation.
    inflation_alert: float = 5.0


@dataclass(frozen=True, slots=True)
class Settings:
    """Application-wide settings bundle."""

    app_title: str = "Commodity Price Monitoring Dashboard"
    app_short_title: str = "Commodity Price Monitoring Dashboard"
    app_icon: str = ":material/monitoring:"
    subtitle: str = (
        "Forecasting price inflation of daily essential goods — a comparative study "
        "of statistical and machine learning models"
    )
    data_dir: Path = DATA_DIR
    export_dir: Path = EXPORT_DIR
    #: Horizons offered in the sidebar; clipped to what the artifacts contain.
    horizons: tuple[int, ...] = (7, 14, 30, 60)
    default_horizon: int = 30
    #: Number of trailing days used for KPI sparklines.
    sparkline_days: int = 45
    #: Rolling window (days) for the realised-volatility risk metric.
    volatility_window: int = 30
    currency: str = "BDT"
    date_format: str = "%d %b %Y"
    cache_ttl: int = 3600
    thresholds: Thresholds = field(default_factory=Thresholds)


SETTINGS: Final[Settings] = Settings()


def format_date(value) -> str:
    """Format a date-like value using the dashboard's display convention.

    Args:
        value: Anything ``pandas.Timestamp`` accepts.

    Returns:
        A ``dd Mon yyyy`` string, or ``"—"`` when the value is missing.
    """
    import pandas as pd

    if value is None:
        return "—"
    try:
        ts = pd.Timestamp(value)
    except (ValueError, TypeError):
        return "—"
    if pd.isna(ts):
        return "—"
    return ts.strftime(SETTINGS.date_format)
