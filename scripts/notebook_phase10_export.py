"""Phase 10 export cell — paste this into the research notebook and run it.

Place this after Phase 9 (once ``cleaned``, ``ALL_PRED``, ``comparison``,
``BEST_MODEL``, ``future_forecasts``, ``ensemble_weights`` and ``final_models``
all exist in memory). It writes the eight artifacts the Streamlit dashboard
reads and, on Colab, zips them for download.

Nothing is retrained here — the cell only serialises what the notebook already
computed, so the dashboard shows exactly the numbers reported in the thesis.

Copy the resulting files into ``data/processed/`` next to ``streamlit_app.py``.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------
#: Notebook commodity key -> dashboard commodity key.
NAME_MAP = {
    "Onion": "Onion",
    "Potato": "Potato",
    "Soybean": "Soybean Oil",
    "Soybean_Oil": "Soybean Oil",
}

#: Notebook metric column -> dashboard metric column.
METRIC_MAP = {"MAE": "MAE", "RMSE": "RMSE", "MAPE_%": "MAPE", "skill_vs_naive_%": "skill_vs_naive"}

OUT = Path("/content/dashboard_artifacts")
OUT.mkdir(parents=True, exist_ok=True)


def dash_name(key: str) -> str:
    """Translate a notebook commodity key to the dashboard's canonical key."""
    return NAME_MAP.get(key, key)


# --------------------------------------------------------------------------
# 1. historical_price.csv
# --------------------------------------------------------------------------
price_rows = []
for key, frame in cleaned.items():  # noqa: F821 - notebook global
    series = frame["price"]
    price_rows.append(
        pd.DataFrame({"date": series.index, "commodity": dash_name(key),
                      "price": series.to_numpy().round(4)})
    )
prices_long = pd.concat(price_rows, ignore_index=True)
prices_long.to_csv(OUT / "historical_price.csv", index=False)

# --------------------------------------------------------------------------
# 2. historical_inflation.csv
# --------------------------------------------------------------------------
inflation_rows = []
for key, frame in cleaned.items():  # noqa: F821
    series = frame["price"]
    inflation_rows.append(
        pd.DataFrame(
            {
                "date": series.index,
                "commodity": dash_name(key),
                "inflation_daily": (series.pct_change(1) * 100).to_numpy().round(4),
                "inflation_weekly": (series.pct_change(7) * 100).to_numpy().round(4),
                "inflation_monthly": (series.pct_change(30) * 100).to_numpy().round(4),
                "inflation_annual": (series.pct_change(365) * 100).to_numpy().round(4),
            }
        )
    )
pd.concat(inflation_rows, ignore_index=True).to_csv(OUT / "historical_inflation.csv", index=False)

# --------------------------------------------------------------------------
# 3. forecast.csv
# --------------------------------------------------------------------------
forecast_rows = []
for key, models in future_forecasts.items():  # noqa: F821
    for model, frame in models.items():
        forecast_rows.append(
            pd.DataFrame(
                {
                    "date": frame.index,
                    "commodity": dash_name(key),
                    "model": model,
                    "horizon_day": np.arange(1, len(frame) + 1),
                    "forecast": frame["forecast"].to_numpy().round(4),
                    "lower": frame["lower"].to_numpy().round(4),
                    "upper": frame["upper"].to_numpy().round(4),
                    "is_best_model": model == BEST_MODEL[key],  # noqa: F821
                }
            )
        )
pd.concat(forecast_rows, ignore_index=True).to_csv(OUT / "forecast.csv", index=False)

# --------------------------------------------------------------------------
# 4. metrics.csv
# --------------------------------------------------------------------------
metrics = comparison.copy()  # noqa: F821
metrics = metrics.rename(columns=METRIC_MAP)
metrics["commodity"] = metrics["commodity"].map(dash_name)
metrics["is_best_model"] = [
    row["model"] == BEST_MODEL.get(  # noqa: F821
        next(k for k, v in NAME_MAP.items() if v == row["commodity"]), None
    )
    or row["model"] == BEST_MODEL.get(row["commodity"], None)  # noqa: F821
    for _, row in metrics.iterrows()
]
keep = [c for c in ["commodity", "model", "MAE", "RMSE", "MAPE", "skill_vs_naive",
                    "is_best_model"] if c in metrics.columns]
metrics[keep].to_csv(OUT / "metrics.csv", index=False)

# --------------------------------------------------------------------------
# 5. model_comparison.csv
# --------------------------------------------------------------------------
comparison_rows = []
for key, models in ALL_PRED.items():  # noqa: F821
    for model, frame in models.items():
        comparison_rows.append(
            pd.DataFrame(
                {
                    "date": frame.index,
                    "commodity": dash_name(key),
                    "model": model,
                    "actual": frame["actual"].to_numpy().round(4),
                    "predicted": frame["pred"].to_numpy().round(4),
                }
            )
        )
pd.concat(comparison_rows, ignore_index=True).dropna(subset=["predicted"]).to_csv(
    OUT / "model_comparison.csv", index=False
)

# --------------------------------------------------------------------------
# 6. feature_importance.csv
# --------------------------------------------------------------------------
importance_rows = []
for key, bundle in final_models.items():  # noqa: F821
    model = bundle["xgb"]
    importance_rows.append(
        pd.DataFrame(
            {
                "commodity": dash_name(key),
                "feature": FEATURE_COLS,  # noqa: F821
                "importance": model.feature_importances_.astype(float),
            }
        )
    )
pd.concat(importance_rows, ignore_index=True).to_csv(OUT / "feature_importance.csv", index=False)

# --------------------------------------------------------------------------
# 7. insights.json
# --------------------------------------------------------------------------
last_date = max(frame["price"].index[-1] for frame in cleaned.values())  # noqa: F821
first_forecast = min(f["SARIMA"].index[0] for f in future_forecasts.values())  # noqa: F821
last_forecast = max(f["SARIMA"].index[-1] for f in future_forecasts.values())  # noqa: F821

insights = {
    "schema": 1,
    "generated_at": datetime.now().isoformat(timespec="seconds"),
    "source": "research notebook Phase 10",
    "dataset": {
        "last_observation": str(last_date.date()),
        "forecast_start": str(first_forecast.date()),
        "forecast_end": str(last_forecast.date()),
        "max_horizon_days": int(max(CONFIG["horizons"])),  # noqa: F821
        "test_days": int(CONFIG["test_days"]),  # noqa: F821
    },
    "best_model": {dash_name(k): v for k, v in BEST_MODEL.items()},  # noqa: F821
    "ensemble_weights": {
        dash_name(k): {m: round(float(w), 4) for m, w in v.items()}
        for k, v in ensemble_weights.items()  # noqa: F821
    },
    "sarima_orders": {
        dash_name(k): {
            "order": list(v["order"]),
            "seasonal_order": list(v["seasonal_order"]),
            "aic": float(v["aic"]),
        }
        for k, v in sarima_orders.items()  # noqa: F821
    },
    "commodities": {
        dash_name(k): {
            "drivers": "",
            "last_price": round(float(frame["price"].iloc[-1]), 2),
            "sample_mean": round(float(frame["price"].mean()), 2),
            "sample_start": str(frame["price"].index[0].date()),
            "volatility_30d": round(float(frame["price"].pct_change().tail(30).std() * 100), 4),
        }
        for k, frame in cleaned.items()  # noqa: F821
    },
}
(OUT / "insights.json").write_text(json.dumps(insights, indent=2), encoding="utf-8")

# --------------------------------------------------------------------------
# 8. recommendations.json — reuse the rule base shipped with the dashboard
# --------------------------------------------------------------------------
# The dashboard ships a default rule base and falls back to it when this file is
# absent, so copying `data/processed/recommendations.json` from the repository
# is enough. Regenerate it with:
#     python scripts/build_artifacts.py --quick   (then keep only this one file)

# --------------------------------------------------------------------------
# Bundle for download
# --------------------------------------------------------------------------
print("Artifacts written to", OUT)
for path in sorted(OUT.glob("*")):
    print(f"  {path.name:<28} {path.stat().st_size / 1024:8.1f} KB")

try:
    import shutil

    from google.colab import files  # type: ignore

    archive = shutil.make_archive("/content/dashboard_artifacts", "zip", OUT)
    files.download(archive)
except Exception as exc:  # running outside Colab
    print(f"(skipping download: {exc})")
