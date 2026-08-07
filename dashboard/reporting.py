"""Export helpers for the Downloads page: figure bundles and a PDF summary.

Both builders degrade gracefully. If ``kaleido`` cannot rasterise a Plotly
figure the ZIP falls back to self-contained interactive HTML; if ``reportlab``
is unavailable the PDF button explains what to install instead of crashing.
"""

from __future__ import annotations

import io
import zipfile
from datetime import datetime
from typing import Iterable, Sequence

import pandas as pd
import plotly.graph_objects as go

from dashboard.config import COMMODITY_BY_KEY, SETTINGS, format_date
from dashboard.logging_utils import get_logger
from dashboard.transforms import Snapshot

logger = get_logger(__name__)


def figures_archive(figures: dict[str, go.Figure], scale: int = 2) -> tuple[bytes, str]:
    """Bundle figures into a ZIP of PNGs, falling back to standalone HTML.

    Args:
        figures: Mapping of file stem to figure.
        scale: Raster scale factor when PNG export is available.

    Returns:
        ``(zip_bytes, format_used)`` where format is ``"png"`` or ``"html"``.
    """
    import plotly.io as pio

    buffer = io.BytesIO()
    used = "png"
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
        for name, figure in figures.items():
            try:
                # Round-trip through Plotly's own JSON encoder first: it knows how
                # to serialise pandas Timestamps, which the raster backend does not.
                plain = pio.from_json(figure.to_json())
                archive.writestr(f"{name}.png", plain.to_image(format="png", scale=scale))
            except Exception as exc:  # kaleido missing, or no Chrome runtime available
                logger.info("PNG export unavailable for %s (%s); writing HTML.", name, exc)
                used = "html"
                archive.writestr(
                    f"{name}.html",
                    figure.to_html(include_plotlyjs="cdn", full_html=True),
                )
    return buffer.getvalue(), used


def _pdf_styles():
    """Return the paragraph styles used by the PDF report."""
    from reportlab.lib import colors
    from reportlab.lib.enums import TA_LEFT
    from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet

    base = getSampleStyleSheet()
    return {
        "title": ParagraphStyle(
            "BdpTitle", parent=base["Title"], fontSize=17, leading=21,
            textColor=colors.HexColor("#0F172A"), alignment=TA_LEFT, spaceAfter=4,
        ),
        "subtitle": ParagraphStyle(
            "BdpSubtitle", parent=base["Normal"], fontSize=9.5, leading=13,
            textColor=colors.HexColor("#64748B"), spaceAfter=14,
        ),
        "h2": ParagraphStyle(
            "BdpH2", parent=base["Heading2"], fontSize=11.5, leading=15,
            textColor=colors.HexColor("#1E293B"), spaceBefore=13, spaceAfter=5,
        ),
        "body": ParagraphStyle(
            "BdpBody", parent=base["Normal"], fontSize=9.2, leading=13.6,
            textColor=colors.HexColor("#1E293B"), spaceAfter=5,
        ),
        "small": ParagraphStyle(
            "BdpSmall", parent=base["Normal"], fontSize=7.8, leading=10,
            textColor=colors.HexColor("#94A3B8"),
        ),
    }


def _table(data: Sequence[Sequence[object]], widths: Sequence[float] | None = None):
    """Build a styled reportlab table from a header row plus body rows."""
    from reportlab.lib import colors
    from reportlab.platypus import Table, TableStyle

    table = Table([list(map(str, row)) for row in data], colWidths=widths, hAlign="LEFT")
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#0F172A")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, -1), 8),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
                ("TOPPADDING", (0, 0), (-1, -1), 5),
                ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#E2E8F0")),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F8FAFC")]),
                ("ALIGN", (1, 1), (-1, -1), "RIGHT"),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ]
        )
    )
    return table


def summary_pdf(
    snapshot: Snapshot,
    forecast: pd.DataFrame,
    metrics: pd.DataFrame,
    comparison: pd.DataFrame,
    insights: Iterable[str],
    recommendations: dict[str, list[str]],
    horizon: int,
) -> bytes:
    """Render the one-page decision brief as a PDF.

    Args:
        snapshot: KPI snapshot for the selected product and forecast date.
        forecast: Forecast path for the selected product / model / horizon.
        metrics: Model metrics for the selected product.
        comparison: Cross-commodity snapshot table.
        insights: Plain-text insight sentences.
        recommendations: ``{"Policy makers": [...], "Consumers": [...]}``.
        horizon: Selected horizon in days.

    Returns:
        The PDF file as bytes.

    Raises:
        ImportError: If ``reportlab`` is not installed.
    """
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.units import mm
    from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer

    meta = COMMODITY_BY_KEY[snapshot.commodity]
    styles = _pdf_styles()
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        leftMargin=16 * mm,
        rightMargin=16 * mm,
        topMargin=15 * mm,
        bottomMargin=15 * mm,
        title=f"{meta.label} forecast brief",
        author=SETTINGS.app_title,
    )

    story: list[object] = [
        Paragraph(f"{meta.label} — price and inflation brief", styles["title"]),
        Paragraph(
            f"{SETTINGS.app_title} &middot; data to {format_date(snapshot.as_of)} &middot; "
            f"{horizon}-day horizon &middot; model: {snapshot.model} &middot; "
            f"generated {datetime.now().strftime('%d %b %Y %H:%M')}",
            styles["subtitle"],
        ),
        Paragraph("Headline indicators", styles["h2"]),
        _table(
            [
                ["Indicator", "Value"],
                ["Current price (last observation)", f"{snapshot.current_price:,.2f} {meta.unit}"],
                [f"Forecast price ({format_date(snapshot.forecast_date)})",
                 f"{snapshot.forecast_price:,.2f} {meta.unit}"],
                ["95% interval", f"{snapshot.lower:,.2f} – {snapshot.upper:,.2f} {meta.unit}"],
                ["Expected change", f"{snapshot.expected_change_pct:+.2f} %"],
                ["Current 30-day inflation", f"{snapshot.current_inflation:+.2f} %"],
                ["Forecast 30-day inflation", f"{snapshot.forecast_inflation:+.2f} %"],
                ["Realised volatility (30d)", f"{snapshot.volatility:,.2f} %"],
                ["Market risk rating", snapshot.risk_level],
            ],
            widths=[95 * mm, 63 * mm],
        ),
    ]

    if not metrics.empty:
        story += [
            Paragraph("Model accuracy on the held-out test set", styles["h2"]),
            _table(
                [["Model", "MAE", "RMSE", "MAPE (%)"]]
                + [
                    [
                        row["model"],
                        f"{row['MAE']:.3f}",
                        f"{row['RMSE']:.3f}",
                        f"{row['MAPE']:.2f}",
                    ]
                    for _, row in metrics.iterrows()
                ],
                widths=[52 * mm, 35 * mm, 35 * mm, 36 * mm],
            ),
        ]

    if not comparison.empty:
        story += [
            Paragraph("Basket comparison", styles["h2"]),
            _table(
                [["Commodity", "Current", "Forecast", "Change (%)", "Risk"]]
                + [
                    [
                        COMMODITY_BY_KEY[row["commodity"]].label,
                        f"{row['Current price']:,.2f}",
                        f"{row['Forecast price']:,.2f}",
                        f"{row['Expected change']:+.2f}",
                        row["Risk"],
                    ]
                    for _, row in comparison.iterrows()
                ],
                widths=[42 * mm, 29 * mm, 29 * mm, 30 * mm, 28 * mm],
            ),
        ]

    story.append(Paragraph("Key insights", styles["h2"]))
    for line in insights:
        story.append(Paragraph(f"&bull; {line}", styles["body"]))

    for audience, items in recommendations.items():
        story.append(Paragraph(f"Recommendations — {audience}", styles["h2"]))
        for item in items:
            story.append(Paragraph(f"&bull; {item}", styles["body"]))

    if not forecast.empty:
        head = forecast.head(15)
        story += [
            Paragraph(f"Forecast path (first {len(head)} days)", styles["h2"]),
            _table(
                [["Date", f"Forecast ({meta.unit})", "Lower 95%", "Upper 95%", "Change (%)"]]
                + [
                    [
                        format_date(row["date"]),
                        f"{row['forecast']:,.2f}",
                        f"{row['lower']:,.2f}",
                        f"{row['upper']:,.2f}",
                        f"{(row['forecast'] / snapshot.current_price - 1) * 100:+.2f}",
                    ]
                    for _, row in head.iterrows()
                ],
                widths=[34 * mm, 34 * mm, 30 * mm, 30 * mm, 30 * mm],
            ),
        ]

    story += [
        Spacer(1, 8),
        Paragraph(
            "Forecasts are model estimates carrying the uncertainty stated above and should "
            "support, not replace, market judgement. Source: research artifacts exported from "
            "the SARIMA / XGBoost / LSTM / weighted-ensemble study.",
            styles["small"],
        ),
    ]

    doc.build(story)
    return buffer.getvalue()
