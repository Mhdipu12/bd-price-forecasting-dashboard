# Bangladesh Essential Goods Price Intelligence

A decision-support dashboard for **Forecasting Price Inflation of Daily Essential Goods in
Bangladesh: A Comparative Study of Statistical and Machine Learning Models**.

The app visualises cleaned price history, model forecasts with prediction intervals, inflation
at every time scale, a like-for-like model comparison (SARIMA · XGBoost · LSTM · weighted
ensemble) and rule-based recommendations for policymakers and consumers.

> **The dashboard never trains a model.** It reads exported artifacts. Training happens once, in
> the research notebook or in `scripts/build_artifacts.py`.

---

## Quick start

```bash
pip install -r requirements.txt
python scripts/build_artifacts.py
streamlit run streamlit_app.py
```

The first command installs everything; the second produces `data/processed/` (a few minutes —
it fits SARIMA, XGBoost and an LSTM per commodity); the third opens the dashboard at
<http://localhost:8501>.

Add `--quick` to `build_artifacts.py` for a fast smoke run, or `--skip-lstm` if PyTorch is not
available on your Python version.

---

## Project layout

```
streamlit_app.py            entry point: theme, data load, navigation, sidebar filters
app_pages/
    overview.py             status strip, product KPIs, price/inflation trajectory, basket
    product_analysis.py     trend, weekly/monthly views, distribution, descriptive statistics
    price_forecast.py       interactive forecast chart, model agreement, searchable table
    inflation_analysis.py   realised vs projected inflation, seasonality heatmap
    model_comparison.py     metrics, actual vs predicted, error diagnostics, feature importance
    insights.py             generated insights and policy / consumer recommendations
    downloads.py            CSV, figure bundle and PDF brief exports
dashboard/
    config.py               palette, commodity registry, thresholds, paths
    data_loader.py          cached, validated artifact readers
    transforms.py           all data logic (KPI snapshots, resampling, statistics)
    charts.py               Plotly figure builders — one per analytical question
    components.py           status cards, KPI cards, insight cards, inline SVG icons
    theme.py                CSS design tokens and the Plotly template
    insights.py             regime classification and narrative generation
    reporting.py            figure ZIP and PDF report builders
    sidebar.py              filters and their interdependencies
    context.py              the per-run application context shared by pages
scripts/
    build_artifacts.py            local pipeline → data/processed/
    notebook_phase10_export.py    paste-in cell that exports from the notebook instead
data/processed/             the artifacts the dashboard reads
Raw Data/                   source CSVs (Onion, Potato, Soybean)
Notebook/                   the research notebook
```

---

## Data contract

`data/processed/` must contain these files. The dashboard validates every column on startup and
tells you exactly what is missing.

| File | Columns | Required |
|---|---|---|
| `historical_price.csv` | `date, commodity, price` | yes |
| `historical_inflation.csv` | `date, commodity, inflation_daily, inflation_weekly, inflation_monthly, inflation_annual` | yes |
| `forecast.csv` | `date, commodity, model, horizon_day, forecast, lower, upper, is_best_model` | yes |
| `metrics.csv` | `commodity, model, MAE, RMSE, MAPE, skill_vs_naive, is_best_model` | yes |
| `model_comparison.csv` | `date, commodity, model, actual, predicted` | yes |
| `feature_importance.csv` | `commodity, feature, importance` | optional |
| `insights.json` | dataset metadata, best model, ensemble weights, commodity context | optional |
| `recommendations.json` | regime-keyed policy / consumer templates | optional |

`commodity` values are `Onion`, `Potato` and `Soybean Oil`. Optional files have built-in
fallbacks, so the app still runs without them.

### Using the notebook's own outputs

`scripts/build_artifacts.py` reproduces the study's pipeline locally, which is convenient but is
*not* the authoritative source for the thesis. To publish the exact numbers you report, paste
`scripts/notebook_phase10_export.py` into the notebook after Phase 9 and run it — it serialises
the objects already in memory (`cleaned`, `ALL_PRED`, `comparison`, `BEST_MODEL`,
`future_forecasts`, `ensemble_weights`, `final_models`) without retraining anything. Copy the
resulting files into `data/processed/`.

---

## Data logic

Two rules are enforced everywhere and are worth stating explicitly, because they are the source
of most dashboard bugs:

* **Historical KPIs always read the last historical observation** — never today's date.
* **Forecast KPIs always read the selected forecast date** from the sidebar.

Changing any sidebar filter reruns the whole app, so every KPI, chart, table and generated
sentence updates together. The forecast date is automatically clamped into the window the
selected horizon allows.

Derived measures:

* **Inflation** — percentage change in the cleaned daily price over 1 / 7 / 30 / 365 days. The
  30-day measure is the daily analogue of the monthly figure BBS publishes.
* **Forecast inflation** — the horizon move rescaled to a 30-day rate, so it is directly
  comparable with the realised figure.
* **Volatility** — standard deviation of daily returns over the trailing 30 days.
* **Risk rating** — blends volatility with the absolute projected move, so a quiet commodity
  about to jump still ranks as risky.

---

## Cleaning notes

`build_artifacts.py` reindexes each series to a gap-free daily calendar, interpolates on the time
index, and winsorises outliers with a **local** rolling median/MAD rule (9-day centred window,
robust-z cut-off 6, MAD floored at 1% of the local price level).

The short window and the MAD floor matter: Bangladeshi retail prices sit flat at round numbers
for days, which drives a raw MAD to zero and makes a naive robust z-score explode — a 31-day
window with a zero-floored MAD flags ~15% of observations and shaves genuine crisis peaks by
double digits. The settings above flag ~1% and leave the real spikes intact.

---

## Configuration

Everything tunable lives in `dashboard/config.py`:

* `COMMODITIES` — registry of tracked goods (key, label, icon, colour, unit, source file)
* `LIGHT_PALETTE` / `DARK_PALETTE` — the full colour system
* `Thresholds` — the cut-offs that turn numbers into decision language
* `Settings` — horizons, sparkline length, volatility window, cache TTL, date format

Streamlit's own theming (fonts, radii, widget colours, light/dark variants) is in
`.streamlit/config.toml`.

---

## Performance

* Artifacts are read once through `st.cache_data` with a one-hour TTL; filter changes are
  dictionary lookups, not disk reads.
* Figures are built per run from already-loaded frames — cheap, and always consistent with the
  active theme.
* The expensive exports (figure ZIP, PDF) are behind explicit buttons so no page pays for them.

## Logging

Logs go to the console and to `logs/dashboard.log`. Raise or lower the level in
`dashboard/logging_utils.py`.

---

## Known limitations

* Multi-step forecasts are recursive, so uncertainty compounds — the interval widens roughly with
  the square root of the horizon. Treat the band, not the central line, as the planning number.
* SARIMA's interval is analytic and assumes Gaussian errors; the XGBoost and LSTM intervals are
  residual-based. They are not strictly comparable.
* Recommendations are rule-based readings of the forecast. They frame a decision; they are not
  financial advice.
