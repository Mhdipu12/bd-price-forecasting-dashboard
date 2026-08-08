"""Plotly figure builders — one function per analytical question.

Every function returns a ready-to-render :class:`plotly.graph_objects.Figure`
themed through :func:`dashboard.theme.plotly_template`. No function reads
Streamlit state, which keeps the charts unit-testable and reusable by the PDF
report builder.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from dashboard.config import COMMODITY_BY_KEY, COMMODITY_COLORS, MODEL_COLORS, Palette
from dashboard.logging_utils import get_logger
from dashboard.theme import plotly_template

logger = get_logger(__name__)

_RANGE_SELECTOR = {
    "buttons": [
        {"count": 3, "label": "3M", "step": "month", "stepmode": "backward"},
        {"count": 6, "label": "6M", "step": "month", "stepmode": "backward"},
        {"count": 1, "label": "1Y", "step": "year", "stepmode": "backward"},
        {"count": 2, "label": "2Y", "step": "year", "stepmode": "backward"},
        {"step": "all", "label": "All"},
    ],
    "y": 1.12,
    "x": 0,
    "bgcolor": "rgba(148,163,184,.12)",
    "activecolor": "rgba(37,99,235,.28)",
    "font": {"size": 11},
}


def _base(mode: str, title: str, height: int = 380, **layout) -> go.Figure:
    """Return an empty figure pre-loaded with the active template."""
    figure = go.Figure()
    figure.update_layout(
        template=plotly_template(mode),
        title=title,
        height=height,
        **layout,
    )
    return figure


def _rgba(hex_color: str, alpha: float) -> str:
    """Convert ``#RRGGBB`` to an ``rgba()`` string."""
    value = hex_color.lstrip("#")
    r, g, b = (int(value[i : i + 2], 16) for i in (0, 2, 4))
    return f"rgba({r},{g},{b},{alpha})"


def empty_figure(mode: str, message: str, height: int = 320) -> go.Figure:
    """Return a placeholder figure carrying an explanatory message."""
    figure = _base(mode, "", height=height)
    figure.add_annotation(
        text=message,
        showarrow=False,
        font={"size": 13, "color": "#94A3B8"},
        xref="paper",
        yref="paper",
        x=0.5,
        y=0.5,
    )
    figure.update_xaxes(visible=False)
    figure.update_yaxes(visible=False)
    return figure


# --------------------------------------------------------------------------
# Price: history vs forecast
# --------------------------------------------------------------------------
def price_history_vs_forecast(
    prices: pd.DataFrame,
    forecast: pd.DataFrame,
    commodity: str,
    model: str,
    mode: str,
    palette: Palette,
    lookback_days: int = 365,
    marker_date: pd.Timestamp | None = None,
    range_selector: bool = False,
    height: int = 400,
) -> go.Figure:
    """Plot observed prices, the forecast path and its 95% interval.

    Answers: *where has the price been, and where is it heading?*

    Args:
        prices: ``date, price`` history.
        forecast: ``date, forecast, lower, upper`` path.
        commodity: Canonical commodity key, used for colour and units.
        model: Model name shown in the legend.
        mode: Active theme mode.
        palette: Active palette.
        lookback_days: How much history to show; ``0`` shows everything.
        marker_date: Optional vertical marker for the selected forecast date.
        range_selector: Add zoom/pan range buttons and a range slider.
        height: Figure height in pixels.
    """
    meta = COMMODITY_BY_KEY[commodity]
    figure = _base(mode, "", height=height)

    if prices.empty:
        return empty_figure(mode, "No price history available.", height)

    window = prices.tail(lookback_days) if lookback_days else prices
    figure.add_trace(
        go.Scatter(
            x=window["date"],
            y=window["price"],
            name="Observed price",
            mode="lines",
            line={"color": meta.color, "width": 2},
            hovertemplate="%{y:,.2f} " + meta.unit + "<extra>Observed</extra>",
        )
    )

    if not forecast.empty:
        bridge_x = [window["date"].iloc[-1], *forecast["date"]]
        bridge_y = [window["price"].iloc[-1], *forecast["forecast"]]
        band_color = _rgba(palette.primary, 0.16)

        figure.add_trace(
            go.Scatter(
                x=list(forecast["date"]) + list(forecast["date"][::-1]),
                y=list(forecast["upper"]) + list(forecast["lower"][::-1]),
                fill="toself",
                fillcolor=band_color,
                line={"width": 0},
                name="95% interval",
                hoverinfo="skip",
                showlegend=True,
            )
        )
        figure.add_trace(
            go.Scatter(
                x=bridge_x,
                y=bridge_y,
                name=f"{model} forecast",
                mode="lines",
                line={"color": palette.primary, "width": 2.4, "dash": "dot"},
                hovertemplate="%{y:,.2f} " + meta.unit + f"<extra>{model}</extra>",
            )
        )
        figure.add_vline(
            x=window["date"].iloc[-1],
            line={"color": palette.primary, "width": 2.5, "dash": "dash"},
        )

    if marker_date is not None and not forecast.empty:
        target = pd.Timestamp(marker_date)
        row = forecast[forecast["date"] == target]
        if not row.empty:
            figure.add_trace(
                go.Scatter(
                    x=[target],
                    y=[float(row["forecast"].iloc[0])],
                    mode="markers",
                    marker={
                        "size": 11,
                        "color": palette.primary,
                        "line": {"color": palette.surface, "width": 2},
                    },
                    name="Selected date",
                    hovertemplate="%{y:,.2f} " + meta.unit + "<extra>Selected date</extra>",
                )
            )

    figure.update_yaxes(title_text=f"Price ({meta.unit})")
    if range_selector:
        figure.update_xaxes(rangeselector=_RANGE_SELECTOR, rangeslider={"visible": True, "thickness": 0.07})
        figure.update_layout(margin={"t": 78})
    return figure


def inflation_history_vs_forecast(
    inflation: pd.DataFrame,
    forecast: pd.DataFrame,
    current_price: float,
    commodity: str,
    mode: str,
    palette: Palette,
    lookback_days: int = 365,
    height: int = 380,
) -> go.Figure:
    """Plot realised 30-day inflation against forecast-implied inflation.

    Answers: *is the pace of price growth accelerating or cooling?*

    Args:
        inflation: Historical inflation frame with ``inflation_monthly``.
        forecast: Forecast path used to derive forward inflation.
        current_price: Last observation, the base of the forward calculation.
        commodity: Canonical commodity key.
        mode: Active theme mode.
        palette: Active palette.
        lookback_days: Days of history to show.
        height: Figure height in pixels.
    """
    meta = COMMODITY_BY_KEY[commodity]
    figure = _base(mode, "", height=height)

    if inflation.empty and forecast.empty:
        return empty_figure(mode, "No inflation data available.", height)

    if not inflation.empty and "inflation_monthly" in inflation.columns:
        window = inflation.tail(lookback_days) if lookback_days else inflation
        figure.add_trace(
            go.Scatter(
                x=window["date"],
                y=window["inflation_monthly"],
                name="Observed 30-day inflation",
                mode="lines",
                line={"color": meta.color, "width": 2},
                hovertemplate="%{y:+.2f}%<extra>Observed</extra>",
            )
        )

    if not forecast.empty and current_price:
        forward = (forecast["forecast"] / current_price - 1) * 100
        figure.add_trace(
            go.Scatter(
                x=forecast["date"],
                y=forward,
                name="Forecast cumulative inflation",
                mode="lines",
                line={"color": palette.primary, "width": 2.4, "dash": "dot"},
                hovertemplate="%{y:+.2f}%<extra>Forecast</extra>",
            )
        )
        upper = (forecast["upper"] / current_price - 1) * 100
        lower = (forecast["lower"] / current_price - 1) * 100
        figure.add_trace(
            go.Scatter(
                x=list(forecast["date"]) + list(forecast["date"][::-1]),
                y=list(upper) + list(lower[::-1]),
                fill="toself",
                fillcolor=_rgba(palette.primary, 0.13),
                line={"width": 0},
                name="95% interval",
                hoverinfo="skip",
            )
        )

    figure.add_hline(y=0, line={"color": palette.text_muted, "width": 1, "dash": "dot"})
    figure.update_yaxes(title_text="Inflation (%)", ticksuffix="%")
    return figure


# --------------------------------------------------------------------------
# Cross-commodity comparison
# --------------------------------------------------------------------------
def commodity_price_comparison(
    table: pd.DataFrame, mode: str, palette: Palette, height: int = 340
) -> go.Figure:
    """Compare current and forecast price levels across all commodities.

    Answers: *which basket item is most expensive today and at the horizon?*
    """
    if table.empty:
        return empty_figure(mode, "No comparison data available.", height)

    labels = [COMMODITY_BY_KEY[c].label for c in table["commodity"]]
    figure = _base(mode, "", height=height, barmode="group", hovermode="x")
    figure.add_trace(
        go.Bar(
            x=labels,
            y=table["Current price"],
            name="Current price",
            marker_color=[COMMODITY_COLORS[c] for c in table["commodity"]],
            marker_line_width=0,
            text=[f"{v:,.1f}" for v in table["Current price"]],
            textposition="outside",
            hovertemplate="%{y:,.2f}<extra>Current</extra>",
        )
    )
    figure.add_trace(
        go.Bar(
            x=labels,
            y=table["Forecast price"],
            name="Forecast price",
            marker_color=[_rgba(COMMODITY_COLORS[c], 0.42) for c in table["commodity"]],
            marker_line={"width": 1.4, "color": [COMMODITY_COLORS[c] for c in table["commodity"]]},
            text=[f"{v:,.1f}" for v in table["Forecast price"]],
            textposition="outside",
            hovertemplate="%{y:,.2f}<extra>Forecast</extra>",
        )
    )
    figure.update_yaxes(title_text="Price (BDT)")
    figure.update_layout(margin={"t": 46, "b": 8})
    return figure


def commodity_inflation_comparison(
    table: pd.DataFrame, mode: str, palette: Palette, height: int = 340
) -> go.Figure:
    """Compare realised and forecast inflation across all commodities.

    Answers: *where is inflationary pressure building fastest?*
    """
    if table.empty:
        return empty_figure(mode, "No comparison data available.", height)

    labels = [COMMODITY_BY_KEY[c].label for c in table["commodity"]]
    figure = _base(mode, "", height=height, barmode="group", hovermode="x")
    figure.add_trace(
        go.Bar(
            x=labels,
            y=table["Current inflation"],
            name="Current 30-day inflation",
            marker_color=palette.text_muted,
            marker_line_width=0,
            hovertemplate="%{y:+.2f}%<extra>Current</extra>",
        )
    )
    colors = [
        palette.danger if v > 0 else palette.success for v in table["Forecast inflation"].fillna(0)
    ]
    figure.add_trace(
        go.Bar(
            x=labels,
            y=table["Forecast inflation"],
            name="Forecast 30-day inflation",
            marker_color=colors,
            marker_line_width=0,
            text=[f"{v:+.2f}%" for v in table["Forecast inflation"]],
            textposition="outside",
            hovertemplate="%{y:+.2f}%<extra>Forecast</extra>",
        )
    )
    figure.add_hline(y=0, line={"color": palette.text_muted, "width": 1})
    figure.update_yaxes(title_text="Inflation (%)", ticksuffix="%")
    figure.update_layout(margin={"t": 46, "b": 8})
    return figure


# --------------------------------------------------------------------------
# Product analysis
# --------------------------------------------------------------------------
def price_trend(
    prices: pd.DataFrame,
    commodity: str,
    mode: str,
    palette: Palette,
    show_bands: bool = True,
    height: int = 360,
) -> go.Figure:
    """Plot the full price history with 30/90-day moving averages.

    Answers: *what is the long-run trend beneath the daily noise?*
    """
    meta = COMMODITY_BY_KEY[commodity]
    if prices.empty:
        return empty_figure(mode, "No price history available.", height)

    figure = _base(mode, "", height=height)
    figure.add_trace(
        go.Scatter(
            x=prices["date"],
            y=prices["price"],
            name="Daily price",
            mode="lines",
            line={"color": _rgba(meta.color, 0.45), "width": 1.2},
            hovertemplate="%{y:,.2f}<extra>Daily</extra>",
        )
    )
    if show_bands and len(prices) > 90:
        figure.add_trace(
            go.Scatter(
                x=prices["date"],
                y=prices["price"].rolling(30).mean(),
                name="30-day average",
                mode="lines",
                line={"color": meta.color, "width": 2},
                hovertemplate="%{y:,.2f}<extra>30-day MA</extra>",
            )
        )
        figure.add_trace(
            go.Scatter(
                x=prices["date"],
                y=prices["price"].rolling(90).mean(),
                name="90-day average",
                mode="lines",
                line={"color": palette.primary, "width": 2, "dash": "dash"},
                hovertemplate="%{y:,.2f}<extra>90-day MA</extra>",
            )
        )
    figure.update_yaxes(title_text=f"Price ({meta.unit})")
    figure.update_xaxes(rangeselector=_RANGE_SELECTOR)
    figure.update_layout(margin={"t": 74})
    return figure


def inflation_trend(
    inflation: pd.DataFrame,
    column: str,
    label: str,
    mode: str,
    palette: Palette,
    height: int = 320,
) -> go.Figure:
    """Plot one historical inflation series as a zero-centred area chart.

    Answers: *how volatile has this measure of price growth been?*
    """
    if inflation.empty or column not in inflation.columns:
        return empty_figure(mode, "No inflation history available.", height)

    frame = inflation.dropna(subset=[column])
    if frame.empty:
        return empty_figure(mode, "No inflation history available.", height)

    figure = _base(mode, "", height=height)
    figure.add_trace(
        go.Scatter(
            x=frame["date"],
            y=frame[column],
            name=label,
            mode="lines",
            line={"color": palette.primary, "width": 1.6},
            fill="tozeroy",
            fillcolor=_rgba(palette.primary, 0.14),
            hovertemplate="%{y:+.2f}%<extra>" + label + "</extra>",
        )
    )
    figure.add_hline(y=0, line={"color": palette.text_muted, "width": 1})
    figure.update_yaxes(title_text="Inflation (%)", ticksuffix="%")
    return figure


def periodic_price_bars(
    frame: pd.DataFrame,
    commodity: str,
    label: str,
    mode: str,
    palette: Palette,
    tail: int = 52,
    height: int = 320,
) -> go.Figure:
    """Plot weekly or monthly average prices with their intra-period range.

    Answers: *how does the average price move once daily noise is removed?*
    """
    meta = COMMODITY_BY_KEY[commodity]
    if frame.empty:
        return empty_figure(mode, f"No {label.lower()} data available.", height)

    window = frame.tail(tail)
    figure = _base(mode, "", height=height, hovermode="x")
    figure.add_trace(
        go.Scatter(
            x=list(window["date"]) + list(window["date"][::-1]),
            y=list(window["high"]) + list(window["low"][::-1]),
            fill="toself",
            fillcolor=_rgba(meta.color, 0.14),
            line={"width": 0},
            name="Period low–high",
            hoverinfo="skip",
        )
    )
    figure.add_trace(
        go.Scatter(
            x=window["date"],
            y=window["price"],
            name=f"{label} average",
            mode="lines+markers",
            line={"color": meta.color, "width": 2.2},
            marker={"size": 5},
            hovertemplate="%{y:,.2f}<extra>" + label + " average</extra>",
        )
    )
    figure.update_yaxes(title_text=f"Price ({meta.unit})")
    return figure


def periodic_inflation_bars(
    frame: pd.DataFrame,
    label: str,
    mode: str,
    palette: Palette,
    tail: int = 52,
    height: int = 320,
) -> go.Figure:
    """Plot period-over-period inflation as diverging bars.

    Answers: *which periods pushed prices up, and by how much?*
    """
    if frame.empty:
        return empty_figure(mode, f"No {label.lower()} inflation available.", height)

    window = frame.tail(tail)
    colors = [palette.danger if v > 0 else palette.success for v in window["inflation"]]
    figure = _base(mode, "", height=height, hovermode="x")
    figure.add_trace(
        go.Bar(
            x=window["date"],
            y=window["inflation"],
            marker_color=colors,
            marker_line_width=0,
            name=f"{label} inflation",
            hovertemplate="%{y:+.2f}%<extra>" + label + "</extra>",
        )
    )
    figure.add_hline(y=0, line={"color": palette.text_muted, "width": 1})
    figure.update_yaxes(title_text="Inflation (%)", ticksuffix="%")
    return figure


def price_distribution(
    prices: pd.DataFrame,
    commodity: str,
    mode: str,
    palette: Palette,
    height: int = 320,
) -> go.Figure:
    """Plot the price histogram with mean and median reference lines.

    Answers: *around which price levels has the market spent most of its time?*
    """
    meta = COMMODITY_BY_KEY[commodity]
    if prices.empty:
        return empty_figure(mode, "No price history available.", height)

    series = prices["price"]
    figure = _base(mode, "", height=height, hovermode="closest")
    figure.add_trace(
        go.Histogram(
            x=series,
            nbinsx=45,
            marker_color=_rgba(meta.color, 0.75),
            marker_line={"width": 1, "color": "rgba(255,255,255,.35)"},
            name="Days",
            hovertemplate="%{x:,.1f} → %{y} days<extra></extra>",
        )
    )
    figure.add_vline(
        x=float(series.mean()),
        line={"color": palette.danger, "width": 2, "dash": "dash"},
        annotation_text=f"mean {series.mean():,.1f}",
        annotation_position="top right",
        annotation_font_size=11,
    )
    figure.add_vline(
        x=float(series.median()),
        line={"color": palette.success, "width": 2, "dash": "dot"},
        annotation_text=f"median {series.median():,.1f}",
        annotation_position="top left",
        annotation_font_size=11,
    )
    figure.update_xaxes(title_text=f"Price ({meta.unit})")
    figure.update_yaxes(title_text="Number of days")
    return figure


def inflation_heatmap(
    matrix: pd.DataFrame, mode: str, palette: Palette, height: int = 320
) -> go.Figure:
    """Plot month-on-month inflation as a year × month heatmap.

    Answers: *which calendar months carry Bangladesh's seasonal price pressure?*
    """
    if matrix.empty:
        return empty_figure(mode, "Not enough history for a seasonal heatmap.", height)

    month_names = [
        "Jan", "Feb", "Mar", "Apr", "May", "Jun",
        "Jul", "Aug", "Sep", "Oct", "Nov", "Dec",
    ]
    columns = [month_names[int(m) - 1] for m in matrix.columns]
    bound = float(np.nanmax(np.abs(matrix.to_numpy()))) or 1.0

    figure = _base(mode, "", height=height, hovermode="closest")
    figure.add_trace(
        go.Heatmap(
            z=matrix.to_numpy(),
            x=columns,
            y=[str(int(y)) for y in matrix.index],
            colorscale=[
                [0.0, "#15803D"], [0.35, "#86EFAC"], [0.5, "#F1F5F9"],
                [0.65, "#FCA5A5"], [1.0, "#B91C1C"],
            ],
            zmid=0,
            zmin=-bound,
            zmax=bound,
            colorbar={"title": {"text": "%", "font": {"size": 11}}, "thickness": 12},
            hovertemplate="%{y} %{x}: %{z:+.2f}%<extra></extra>",
        )
    )
    figure.update_yaxes(autorange="reversed", title_text="Year")
    return figure


# --------------------------------------------------------------------------
# Model comparison
# --------------------------------------------------------------------------
def metric_bars(
    metrics: pd.DataFrame,
    metric: str,
    best_model: str,
    mode: str,
    palette: Palette,
    height: int = 300,
) -> go.Figure:
    """Rank models on one error metric, highlighting the winner.

    Answers: *which model is most accurate on this criterion?*
    """
    if metrics.empty:
        return empty_figure(mode, "No metrics available.", height)

    frame = metrics.sort_values(metric)
    colors = [
        MODEL_COLORS.get(m, palette.primary) if m == best_model else _rgba(
            MODEL_COLORS.get(m, palette.primary), 0.45
        )
        for m in frame["model"]
    ]
    suffix = "%" if metric.upper() == "MAPE" else ""
    figure = _base(mode, "", height=height, hovermode="closest")
    figure.add_trace(
        go.Bar(
            x=frame["model"],
            y=frame[metric],
            marker_color=colors,
            marker_line_width=0,
            text=[f"{v:,.3f}{suffix}" for v in frame[metric]],
            textposition="outside",
            hovertemplate="%{x}: %{y:,.4f}" + suffix + "<extra></extra>",
        )
    )
    figure.update_yaxes(title_text=metric, ticksuffix=suffix)
    figure.update_layout(title=metric, margin={"t": 44})
    return figure


def actual_vs_predicted(
    comparison: pd.DataFrame,
    commodity: str,
    models: Sequence[str],
    mode: str,
    palette: Palette,
    height: int = 400,
) -> go.Figure:
    """Overlay every model's test-set prediction on the observed series.

    Answers: *where does each model track reality, and where does it drift?*
    """
    subset = comparison[comparison["commodity"] == commodity]
    if subset.empty:
        return empty_figure(mode, "No back-test predictions available.", height)

    meta = COMMODITY_BY_KEY[commodity]
    figure = _base(mode, "", height=height)

    reference = subset[subset["model"] == subset["model"].iloc[0]].sort_values("date")
    figure.add_trace(
        go.Scatter(
            x=reference["date"],
            y=reference["actual"],
            name="Actual",
            mode="lines",
            line={"color": palette.text, "width": 2.6},
            hovertemplate="%{y:,.2f}<extra>Actual</extra>",
        )
    )
    for model in models:
        rows = subset[subset["model"] == model].sort_values("date")
        if rows.empty:
            continue
        figure.add_trace(
            go.Scatter(
                x=rows["date"],
                y=rows["predicted"],
                name=model,
                mode="lines",
                line={"color": MODEL_COLORS.get(model, palette.primary), "width": 1.5},
                opacity=0.9,
                hovertemplate="%{y:,.2f}<extra>" + model + "</extra>",
            )
        )
    figure.update_yaxes(title_text=f"Price ({meta.unit})")
    return figure


def error_distribution(
    comparison: pd.DataFrame,
    commodity: str,
    models: Sequence[str],
    mode: str,
    palette: Palette,
    height: int = 360,
) -> go.Figure:
    """Compare the spread and bias of each model's forecast errors.

    Answers: *which model is not just accurate on average, but reliably so?*
    """
    subset = comparison[comparison["commodity"] == commodity]
    if subset.empty:
        return empty_figure(mode, "No back-test predictions available.", height)

    figure = _base(mode, "", height=height, hovermode="closest")
    for model in models:
        rows = subset[subset["model"] == model]
        if rows.empty:
            continue
        figure.add_trace(
            go.Box(
                y=rows["actual"] - rows["predicted"],
                name=model,
                marker_color=MODEL_COLORS.get(model, palette.primary),
                boxmean="sd",
                hovertemplate="%{y:+.2f}<extra>" + model + "</extra>",
            )
        )
    figure.add_hline(y=0, line={"color": palette.text_muted, "width": 1, "dash": "dot"})
    figure.update_yaxes(title_text="Forecast error (actual − predicted)")
    return figure


def feature_importance_chart(
    importance: pd.DataFrame,
    commodity: str,
    mode: str,
    palette: Palette,
    top_n: int = 18,
    height: int = 460,
) -> go.Figure:
    """Rank the XGBoost features that drive the forecast.

    Answers: *what information is the machine learning model actually using?*
    """
    subset = importance[importance["commodity"] == commodity]
    if subset.empty:
        return empty_figure(mode, "Feature importance was not exported.", height)

    frame = subset.nlargest(top_n, "importance").sort_values("importance")
    total = subset["importance"].sum() or 1.0
    share = frame["importance"] / total * 100

    figure = _base(mode, "", height=height, hovermode="closest")
    figure.add_trace(
        go.Bar(
            x=frame["importance"],
            y=frame["feature"],
            orientation="h",
            marker_color=[
                _rgba(palette.primary, 0.35 + 0.65 * (v / frame["importance"].max()))
                for v in frame["importance"]
            ],
            marker_line_width=0,
            text=[f"{s:.1f}%" for s in share],
            textposition="outside",
            hovertemplate="%{y}: %{x:,.4f}<extra></extra>",
        )
    )
    figure.update_xaxes(title_text="Gain importance")
    figure.update_layout(margin={"l": 8, "r": 60})
    return figure


def forecast_fan(
    forecasts: dict[str, pd.DataFrame],
    prices: pd.DataFrame,
    commodity: str,
    best_model: str,
    mode: str,
    palette: Palette,
    lookback_days: int = 120,
    height: int = 400,
) -> go.Figure:
    """Overlay every model's forward path so disagreement becomes visible.

    Answers: *how much do the models disagree about the road ahead?*
    """
    meta = COMMODITY_BY_KEY[commodity]
    figure = _base(mode, "", height=height)

    if not prices.empty:
        window = prices.tail(lookback_days)
        figure.add_trace(
            go.Scatter(
                x=window["date"],
                y=window["price"],
                name="Observed",
                mode="lines",
                line={"color": palette.text, "width": 2.2},
                hovertemplate="%{y:,.2f}<extra>Observed</extra>",
            )
        )

    for model, path in forecasts.items():
        if path.empty:
            continue
        is_best = model == best_model
        figure.add_trace(
            go.Scatter(
                x=path["date"],
                y=path["forecast"],
                name=f"{model}{' (best)' if is_best else ''}",
                mode="lines",
                line={
                    "color": MODEL_COLORS.get(model, palette.primary),
                    "width": 2.6 if is_best else 1.4,
                    "dash": "solid" if is_best else "dot",
                },
                hovertemplate="%{y:,.2f}<extra>" + model + "</extra>",
            )
        )
    figure.update_yaxes(title_text=f"Price ({meta.unit})")
    return figure


def forecast_daily_inflation(
    forecast: pd.DataFrame,
    current_price: float,
    mode: str,
    palette: Palette,
    height: int = 320,
) -> go.Figure:
    """Plot day-on-day inflation implied by the forecast path.

    Answers: *is the projected rise steady, or front-loaded?*
    """
    if forecast.empty or len(forecast) < 2:
        return empty_figure(mode, "Not enough forecast days for a daily series.", height)

    daily = forecast["forecast"].pct_change() * 100
    if current_price:
        daily.iloc[0] = (forecast["forecast"].iloc[0] / current_price - 1) * 100

    colors = [palette.danger if v > 0 else palette.success for v in daily.fillna(0)]
    figure = _base(mode, "", height=height, hovermode="x")
    figure.add_trace(
        go.Bar(
            x=forecast["date"],
            y=daily,
            marker_color=colors,
            marker_line_width=0,
            name="Daily inflation",
            hovertemplate="%{y:+.3f}%<extra>Day-on-day</extra>",
        )
    )
    figure.add_hline(y=0, line={"color": palette.text_muted, "width": 1})
    figure.update_yaxes(title_text="Day-on-day inflation (%)", ticksuffix="%")
    return figure
