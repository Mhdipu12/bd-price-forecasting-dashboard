"""Turn loaded artifacts into decision language.

``insights.json`` and ``recommendations.json`` ship a *rule base* — phrasing
templates keyed by market regime plus commodity-specific context. This module
selects the right rule for the currently selected product, forecast date and
horizon, and fills it with live numbers, so the narrative changes whenever a
sidebar filter changes.

If the JSON artifacts are missing the module falls back to built-in templates,
so the page never renders empty.
"""

from __future__ import annotations

import html
from typing import Any, Literal

import numpy as np
import pandas as pd

from dashboard.components import InsightCard
from dashboard.config import COMMODITY_BY_KEY, SETTINGS, format_date
from dashboard.data_loader import DataBundle
from dashboard.logging_utils import get_logger
from dashboard.transforms import Snapshot

logger = get_logger(__name__)

Regime = Literal["rising_sharp", "rising", "stable", "falling", "falling_sharp"]

# --------------------------------------------------------------------------
# Fallback rule base (used when the JSON artifacts are absent)
# --------------------------------------------------------------------------
_FALLBACK_POLICY: dict[str, list[dict[str, str]]] = {
    "rising_sharp": [
        {
            "title": "Release buffer stocks now",
            "body": "A {change:+.1f}% move over {horizon} days is large enough to hit "
            "low-income households directly. Trigger open-market sales through TCB "
            "outlets in the districts with the thinnest supply before the peak.",
            "icon": "alert",
            "tone": "danger",
        },
        {
            "title": "Fast-track import clearance",
            "body": "Waive or reduce duty and clear consignments of {product} at port "
            "within days rather than weeks. Import lead time is the binding constraint "
            "when the projected peak is only {horizon} days out.",
            "icon": "zap",
            "tone": "warning",
        },
        {
            "title": "Publish a weekly price bulletin",
            "body": "Announce the projected band of {lower:,.0f}–{upper:,.0f} {unit} "
            "publicly. Transparent official forecasts reduce speculative hoarding, "
            "which is a large part of short-run spikes in Bangladesh's retail markets.",
            "icon": "file",
            "tone": "primary",
        },
    ],
    "rising": [
        {
            "title": "Put buffer stocks on standby",
            "body": "The model points to a {change:+.1f}% rise by {date}. Confirm "
            "warehouse positions now so an intervention can be executed within a week "
            "if realised prices track the upper band of {upper:,.0f} {unit}.",
            "icon": "shield",
            "tone": "warning",
        },
        {
            "title": "Monitor wholesale-retail margins",
            "body": "Moderate upward pressure is where margin widening usually hides. "
            "Compare mill-gate and bazaar prices weekly for {product} over the "
            "{horizon}-day window.",
            "icon": "target",
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
            "{lower:,.0f}–{upper:,.0f} {unit}. Keep the monitoring cadence and preserve "
            "fiscal space for the commodities that are moving.",
            "icon": "compass",
            "tone": "primary",
        },
    ],
    "falling": [
        {
            "title": "Protect farm-gate prices",
            "body": "A projected {change:+.1f}% decline helps consumers but squeezes "
            "growers. Consider procurement at a floor price so the correction does not "
            "depress next season's planting decisions for {product}.",
            "icon": "leaf",
            "tone": "success",
        },
        {
            "title": "Expand cold-storage access",
            "body": "Cheap {product} is only useful if it can be held. Subsidised "
            "storage during the {horizon}-day soft patch smooths the following peak "
            "far more cheaply than emergency imports do.",
            "icon": "layers",
            "tone": "primary",
        },
    ],
}

_FALLBACK_CONSUMER: dict[str, list[dict[str, str]]] = {
    "rising_sharp": [
        {
            "title": "Buy ahead of the peak",
            "body": "{product} is projected to reach {forecast:,.2f} {unit} by {date}, "
            "up {change:+.1f}% from {current:,.2f}. If it stores well, buying now avoids "
            "roughly {saving:,.0f} {currency} on a month of typical household use.",
            "icon": "cart",
            "tone": "danger",
        },
        {
            "title": "Substitute where you can",
            "body": "During a sharp {product} run-up, shifting part of the basket to "
            "commodities forecast to stay flat protects the food budget without cutting "
            "the number of meals.",
            "icon": "layers",
            "tone": "warning",
        },
    ],
    "rising": [
        {
            "title": "Stagger your purchases",
            "body": "With a mild {change:+.1f}% rise expected by {date}, buying in two "
            "or three tranches over the next {horizon} days beats a single large "
            "purchase — you average into the move instead of timing it.",
            "icon": "cart",
            "tone": "warning",
        },
        {
            "title": "Watch the upper band",
            "body": "The 95% interval reaches {upper:,.0f} {unit}. Treat that, not the "
            "central forecast, as your worst-case planning number.",
            "icon": "shield",
            "tone": "primary",
        },
    ],
    "stable": [
        {
            "title": "No need to stockpile",
            "body": "{product} is expected to stay near {forecast:,.2f} {unit} through "
            "{date}. Normal weekly buying is the cheapest strategy; stockpiling only "
            "adds spoilage risk.",
            "icon": "check",
            "tone": "success",
        },
    ],
    "falling": [
        {
            "title": "Wait before bulk buying",
            "body": "Prices are projected to ease to {forecast:,.2f} {unit} by {date} "
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
}


# --------------------------------------------------------------------------
# Regime classification
# --------------------------------------------------------------------------
def classify_regime(snapshot: Snapshot) -> Regime:
    """Map an expected price change onto a named market regime.

    Args:
        snapshot: KPI snapshot for the selected product and forecast date.

    Returns:
        One of ``rising_sharp``, ``rising``, ``stable``, ``falling`` or
        ``falling_sharp``.
    """
    change = snapshot.expected_change_pct
    thresholds = SETTINGS.thresholds
    if not np.isfinite(change):
        return "stable"
    if change >= 2 * thresholds.material_change:
        return "rising_sharp"
    if change >= thresholds.stable_change:
        return "rising" if change < thresholds.material_change else "rising_sharp"
    if change <= -2 * thresholds.material_change:
        return "falling_sharp"
    if change <= -thresholds.stable_change:
        return "falling"
    return "stable"


def _rule_key(regime: Regime) -> str:
    """Collapse the five regimes onto the four keys used by the rule base."""
    if regime == "falling_sharp":
        return "falling"
    return regime


# --------------------------------------------------------------------------
# Template context
# --------------------------------------------------------------------------
def _context(snapshot: Snapshot, horizon: int) -> dict[str, Any]:
    """Build the substitution context shared by every narrative template."""
    meta = COMMODITY_BY_KEY[snapshot.commodity]
    change = snapshot.expected_change_pct
    monthly_quantity = 8.0  # kg or litres a typical household buys per month
    saving = abs(snapshot.forecast_price - snapshot.current_price) * monthly_quantity
    return {
        "product": meta.label,
        "unit": meta.unit,
        "currency": SETTINGS.currency,
        "current": snapshot.current_price,
        "forecast": snapshot.forecast_price,
        "lower": snapshot.lower,
        "upper": snapshot.upper,
        "change": change if np.isfinite(change) else 0.0,
        "horizon": horizon,
        "date": format_date(snapshot.forecast_date),
        "model": snapshot.model,
        "volatility": snapshot.volatility if np.isfinite(snapshot.volatility) else 0.0,
        "risk": snapshot.risk_level,
        "saving": saving,
    }


def _render(template: str, context: dict[str, Any]) -> str:
    """Format a template, degrading gracefully when a placeholder is unknown."""
    try:
        return html.escape(template.format(**context)).replace("&lt;b&gt;", "<b>").replace(
            "&lt;/b&gt;", "</b>"
        )
    except (KeyError, IndexError, ValueError) as exc:
        logger.warning("Could not render narrative template: %s", exc)
        return html.escape(template)


# --------------------------------------------------------------------------
# Insight cards
# --------------------------------------------------------------------------
def _trend_word(regime: Regime) -> str:
    return {
        "rising_sharp": "sharply rising",
        "rising": "gradually rising",
        "stable": "broadly stable",
        "falling": "easing",
        "falling_sharp": "falling sharply",
    }[regime]


def build_insight_cards(
    bundle: DataBundle,
    snapshot: Snapshot,
    comparison: pd.DataFrame,
    horizon: int,
) -> list[InsightCard]:
    """Generate the six headline insights for the selected product.

    Args:
        bundle: Loaded artifacts.
        snapshot: KPI snapshot for the selected product and forecast date.
        comparison: Cross-commodity table from
            :func:`dashboard.transforms.snapshot_table`.
        horizon: Selected horizon in days.

    Returns:
        Ready-to-render insight cards, in presentation order.
    """
    meta = COMMODITY_BY_KEY[snapshot.commodity]
    regime = classify_regime(snapshot)
    trend = _trend_word(regime)

    riskiest, steadiest = risk_rankings(comparison)
    risk_label = COMMODITY_BY_KEY[riskiest].label if riskiest else "—"
    stable_label = COMMODITY_BY_KEY[steadiest].label if steadiest else "—"

    notes = (
        bundle.insights.get("commodities", {}).get(snapshot.commodity, {})
        if bundle.insights
        else {}
    )
    driver_note = notes.get("drivers", "")

    tone_for_change = (
        "danger" if regime in ("rising_sharp",) else
        "warning" if regime == "rising" else
        "success" if regime in ("falling", "falling_sharp") else
        "primary"
    )

    cards = [
        InsightCard(
            title="Market summary",
            body=(
                f"{meta.label} last traded at <b>{snapshot.current_price:,.2f} {meta.unit}</b> "
                f"on {format_date(snapshot.as_of)}. Over the selected {horizon}-day window the "
                f"<b>{snapshot.model}</b> model — the best performer for this commodity on the "
                f"held-out test set — describes the market as <b>{trend}</b>, with realised "
                f"30-day volatility of {snapshot.volatility:,.2f}%."
                + (f" {html.escape(driver_note)}" if driver_note else "")
            ),
            icon="compass",
            tone="primary",
        ),
        InsightCard(
            title="Price outlook",
            body=(
                f"By <b>{format_date(snapshot.forecast_date)}</b> the central forecast is "
                f"<b>{snapshot.forecast_price:,.2f} {meta.unit}</b>, a change of "
                f"<b>{snapshot.expected_change_pct:+.2f}%</b> against today. The 95% interval "
                f"spans {snapshot.lower:,.2f}–{snapshot.upper:,.2f} {meta.unit}; plan against "
                f"the band, not the central line."
            ),
            icon="trend_up" if snapshot.direction == "up" else
                 "trend_down" if snapshot.direction == "down" else "trend_flat",
            tone=tone_for_change,
        ),
        InsightCard(
            title="Inflation outlook",
            body=(
                f"Realised 30-day inflation stands at <b>{snapshot.current_inflation:+.2f}%</b>. "
                f"The forecast path implies a forward 30-day rate of "
                f"<b>{snapshot.forecast_inflation:+.2f}%</b>, i.e. price growth is "
                f"<b>{'accelerating' if snapshot.forecast_inflation > snapshot.current_inflation else 'decelerating'}</b> "
                f"relative to the recent past."
            ),
            icon="percent",
            tone="danger" if snapshot.forecast_inflation > SETTINGS.thresholds.inflation_alert
            else "primary",
        ),
        InsightCard(
            title="Highest-risk commodity",
            body=(
                f"<b>{risk_label}</b> carries the widest combination of realised volatility and "
                f"projected movement across the basket, so it is where a household budget or a "
                f"market-stabilisation fund is most exposed over the next {horizon} days."
            ),
            icon="alert",
            tone="danger",
        ),
        InsightCard(
            title="Most stable commodity",
            body=(
                f"<b>{stable_label}</b> shows the narrowest movement and the lowest volatility, "
                f"making it the dependable anchor of the essential-goods basket for this window."
            ),
            icon="shield",
            tone="success",
        ),
        InsightCard(
            title="Expected trend",
            body=(
                f"Direction: <b>{trend}</b>. Risk rating: <b>{snapshot.risk_level}</b>. "
                f"The projected move of {snapshot.expected_change_pct:+.2f}% over {horizon} days "
                f"is "
                f"{'material enough to warrant action' if abs(snapshot.expected_change_pct) >= SETTINGS.thresholds.material_change else 'within normal market variation'}"
                f" for {meta.label}."
            ),
            icon="activity",
            tone=tone_for_change,
        ),
    ]
    return cards


def risk_rankings(comparison: pd.DataFrame) -> tuple[str | None, str | None]:
    """Identify the riskiest and steadiest commodity in the basket.

    Risk blends realised volatility with the absolute projected move, so a
    commodity that is quiet but about to jump still scores as risky.

    Args:
        comparison: Cross-commodity snapshot table.

    Returns:
        ``(highest_risk_key, most_stable_key)``; either may be ``None``.
    """
    if comparison.empty:
        return None, None
    frame = comparison.copy()
    frame["score"] = frame["Volatility"].fillna(0) + frame["Expected change"].abs().fillna(0) / 2
    ordered = frame.sort_values("score")
    return str(ordered.iloc[-1]["commodity"]), str(ordered.iloc[0]["commodity"])


# --------------------------------------------------------------------------
# Recommendations
# --------------------------------------------------------------------------
def build_recommendations(
    bundle: DataBundle,
    snapshot: Snapshot,
    horizon: int,
    audience: Literal["policy", "consumer"],
) -> list[InsightCard]:
    """Select and render audience-specific recommendations.

    Args:
        bundle: Loaded artifacts (supplies the JSON rule base when present).
        snapshot: KPI snapshot for the selected product and forecast date.
        horizon: Selected horizon in days.
        audience: ``"policy"`` for policymakers, ``"consumer"`` for households.

    Returns:
        Rendered recommendation cards.
    """
    regime = classify_regime(snapshot)
    key = _rule_key(regime)
    ctx = _context(snapshot, horizon)

    fallback = _FALLBACK_POLICY if audience == "policy" else _FALLBACK_CONSUMER
    rules = bundle.recommendations.get(audience, {}) if bundle.recommendations else {}
    entries = rules.get(key) or fallback.get(key) or fallback["stable"]

    cards: list[InsightCard] = []
    for entry in entries:
        cards.append(
            InsightCard(
                title=str(entry.get("title", "Recommendation")),
                body=_render(str(entry.get("body", "")), ctx),
                icon=str(entry.get("icon", "bulb")),
                tone=str(entry.get("tone", "primary")),  # type: ignore[arg-type]
            )
        )

    note = (
        bundle.recommendations.get("commodity_notes", {})
        .get(snapshot.commodity, {})
        .get(audience)
        if bundle.recommendations
        else None
    )
    if note:
        cards.append(
            InsightCard(
                title=f"{COMMODITY_BY_KEY[snapshot.commodity].label} — market context",
                body=_render(str(note), ctx),
                icon="bulb",
                tone="neutral",
            )
        )
    return cards


def _join(names: list[str]) -> str:
    """Join names into readable English: ``a``, ``a and b``, ``a, b and c``."""
    if len(names) <= 1:
        return names[0] if names else ""
    return f"{', '.join(names[:-1])} and {names[-1]}"


def market_headline(bundle: DataBundle, comparison: pd.DataFrame, horizon: int) -> str:
    """Compose a one-paragraph summary of the whole basket.

    Args:
        bundle: Loaded artifacts.
        comparison: Cross-commodity snapshot table.
        horizon: Selected horizon in days.

    Returns:
        An HTML-safe sentence pair describing basket-wide direction.
    """
    if comparison.empty:
        return "No commodity data is available for the selected window."

    rising = comparison[comparison["Expected change"] > SETTINGS.thresholds.stable_change]
    falling = comparison[comparison["Expected change"] < -SETTINGS.thresholds.stable_change]
    average = comparison["Expected change"].mean()

    parts = [
        f"Across the tracked basket the average projected move over {horizon} days is "
        f"<b>{average:+.2f}%</b>."
    ]
    if not rising.empty:
        names = _join([COMMODITY_BY_KEY[c].label for c in rising["commodity"]])
        parts.append(f"Upward pressure is concentrated in <b>{names}</b>.")
    if not falling.empty:
        names = _join([COMMODITY_BY_KEY[c].label for c in falling["commodity"]])
        parts.append(f"<b>{names}</b> {'is' if len(falling) == 1 else 'are'} projected to ease.")
    if rising.empty and falling.empty:
        parts.append("No commodity is projected to move materially in either direction.")
    return " ".join(parts)
