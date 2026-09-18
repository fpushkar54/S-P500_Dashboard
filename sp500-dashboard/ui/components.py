"""Reusable presentation helpers shared by every tab.

Nothing here computes anything: the functions take already-derived values and
turn them into markup. Keeping the formatting in one place is what stops the
four tabs from drifting apart visually.
"""

from __future__ import annotations

import os
from datetime import datetime, timezone
from typing import Dict, List, Optional, Sequence

import pandas as pd
import streamlit as st

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CSS_PATH = os.path.join(PROJECT_ROOT, "assets", "style.css")

PLOTLY_TEMPLATE = "plotly_dark"
# Single source of truth for chart colour. These mirror the CSS custom
# properties in assets/style.css — change both together.
CHART_COLORS = {
    "accent": "#70D603",
    "up": "#70D603",
    "down": "#FF4B5C",
    "neutral": "#5B8FD9",
    "warning": "#FFB020",
    "line": "#0E2F63",
    "text": "#DCE6F7",
    "muted": "#8FA5C8",
    "ground": "#011133",
    "panel": "#041A45",
    "panel_raised": "#072253",
}

# Eleven hues for eleven GICS sectors, all legible on the navy ground and
# ordered so neighbouring slices never sit at the same lightness.
SECTOR_PALETTE = [
    "#70D603",
    "#4FA8F5",
    "#FFB020",
    "#D96BE0",
    "#2ED9C3",
    "#FF4B5C",
    "#A9C23F",
    "#FF8A4C",
    "#7F9DC7",
    "#9B7BFF",
    "#56D9F0",
]

# Sequential ramp for the treemap: ground navy through to the accent.
SEQUENTIAL_SCALE = ["#041A45", "#0F4A52", "#2E8A2E", "#70D603"]

# The recessive fill used for "everything else" slices.
MUTED_FILL = "#0B2A57"


def with_alpha(color: str, alpha: float) -> str:
    """Convert a palette hex to an rgba() string.

    Plotly fills need transparency that CSS variables cannot provide, so this
    keeps the tabs from hardcoding colour a second time.
    """
    value = CHART_COLORS.get(color, color).lstrip("#")
    red, green, blue = (int(value[i : i + 2], 16) for i in (0, 2, 4))
    return f"rgba({red}, {green}, {blue}, {alpha})"


# --------------------------------------------------------------------------- #
# Formatting
# --------------------------------------------------------------------------- #


def format_number(value, decimals: int = 2, dash: str = "--") -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return dash
    try:
        return f"{float(value):,.{decimals}f}"
    except (TypeError, ValueError):
        return dash


def format_percent(value, decimals: int = 1, signed: bool = False, dash: str = "--") -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return dash
    try:
        number = float(value)
    except (TypeError, ValueError):
        return dash
    sign = "+" if signed and number > 0 else ""
    return f"{sign}{number:,.{decimals}f}%"


def format_market_cap(value, dash: str = "--") -> str:
    """Compact currency: 3.21T, 845.0B, 92.4M."""
    if value is None or pd.isna(value):
        return dash
    try:
        number = float(value)
    except (TypeError, ValueError):
        return dash

    for threshold, suffix in ((1e12, "T"), (1e9, "B"), (1e6, "M"), (1e3, "K")):
        if abs(number) >= threshold:
            return f"${number / threshold:,.2f}{suffix}"
    return f"${number:,.0f}"


def format_timestamp(stamp: Optional[datetime], dash: str = "never") -> str:
    if stamp is None:
        return dash
    if stamp.tzinfo is None:
        stamp = stamp.replace(tzinfo=timezone.utc)
    return stamp.astimezone().strftime("%d %b %Y, %H:%M")


# --------------------------------------------------------------------------- #
# Chrome
# --------------------------------------------------------------------------- #


def load_css(path: str = CSS_PATH) -> None:
    """Inject the stylesheet. Missing file degrades to Streamlit defaults."""
    try:
        with open(path, "r", encoding="utf-8") as handle:
            st.markdown(f"<style>{handle.read()}</style>", unsafe_allow_html=True)
    except OSError:
        st.markdown(
            "<style>.block-container{padding-top:2rem;max-width:1500px;}</style>",
            unsafe_allow_html=True,
        )


def render_masthead(last_refresh: Optional[datetime], universe_size: int) -> None:
    st.markdown(
        f"""
        <div class="masthead">
          <h1>S&amp;P 500 Health Dashboard</h1>
          <p>Valuation, breadth, concentration and forecasting across
             {universe_size} tracked constituents.</p>
          <p class="stamp">Data cached locally &middot; last refreshed
             {format_timestamp(last_refresh)}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def section_header(title: str, note: str = "") -> None:
    st.markdown(
        f"""
        <div class="section-header">
          <h3>{title}</h3>
          <span>{note}</span>
        </div>
        """,
        unsafe_allow_html=True,
    )


def kpi_card(label: str, value: str, delta: Optional[str] = None, tone: str = "") -> str:
    """Return the markup for a single KPI card.

    ``tone`` is one of "", "up", "down", "accent" and drives the left rule.
    """
    tone_class = f" is-{tone}" if tone else ""
    delta_class = "up" if tone == "up" else "down" if tone == "down" else ""
    delta_html = (
        f'<div class="kpi-delta {delta_class}">{delta}</div>' if delta else '<div class="kpi-delta">&nbsp;</div>'
    )
    return (
        f'<div class="kpi-card{tone_class}">'
        f'<div class="kpi-label">{label}</div>'
        f'<div class="kpi-value">{value}</div>'
        f"{delta_html}"
        f"</div>"
    )


def render_kpi_bar(kpis: Sequence[Dict[str, object]]) -> None:
    """Lay out KPI cards across one row of equal columns."""
    if not kpis:
        return
    columns = st.columns(len(kpis), gap="small")
    for column, kpi in zip(columns, kpis):
        with column:
            st.markdown(
                kpi_card(
                    label=str(kpi.get("label", "")),
                    value=str(kpi.get("value", "--")),
                    delta=kpi.get("delta"),
                    tone=str(kpi.get("tone", "")),
                ),
                unsafe_allow_html=True,
            )


def build_summary_kpis(
    macro: Dict[str, pd.Series],
    breadth_snapshot: Dict[str, Optional[float]],
    concentration: Dict[str, object],
    valuation: Dict[str, Optional[float]],
) -> List[Dict[str, object]]:
    """Assemble the five headline numbers shown above the tabs."""

    def last_and_change(series: Optional[pd.Series]):
        if series is None or len(series) == 0:
            return None, None
        latest = float(series.iloc[-1])
        if len(series) < 2:
            return latest, None
        previous = float(series.iloc[-2])
        return latest, latest - previous

    sp500, sp500_change = last_and_change(macro.get("SP500"))
    vix, vix_change = last_and_change(macro.get("VIX"))
    yield_10y, yield_change = last_and_change(macro.get("DGS10"))

    kpis: List[Dict[str, object]] = []

    sp_pct = (sp500_change / (sp500 - sp500_change) * 100.0) if sp500 and sp500_change else None
    kpis.append(
        {
            "label": "S&P 500",
            "value": format_number(sp500) if sp500 else "--",
            "delta": format_percent(sp_pct, signed=True) + " today" if sp_pct is not None else None,
            "tone": "up" if (sp_pct or 0) > 0 else "down" if (sp_pct or 0) < 0 else "",
        }
    )
    kpis.append(
        {
            "label": "10-year Treasury",
            "value": f"{format_number(yield_10y)}%" if yield_10y else "--",
            "delta": f"{yield_change:+.2f} pts" if yield_change is not None else None,
            "tone": "down" if (yield_change or 0) > 0 else "up" if (yield_change or 0) < 0 else "",
        }
    )
    kpis.append(
        {
            "label": "VIX",
            "value": format_number(vix) if vix else "--",
            "delta": f"{vix_change:+.2f}" if vix_change is not None else None,
            "tone": "down" if (vix_change or 0) > 0 else "up" if (vix_change or 0) < 0 else "",
        }
    )

    pct_above_200 = breadth_snapshot.get("pct_above_200")
    delta_200 = breadth_snapshot.get("delta_200")
    kpis.append(
        {
            "label": "Above 200-day average",
            "value": format_percent(pct_above_200, decimals=1),
            "delta": f"{delta_200:+.1f} pts vs 1m ago" if delta_200 is not None else None,
            "tone": "up" if (delta_200 or 0) > 0 else "down" if (delta_200 or 0) < 0 else "",
        }
    )

    top_10 = (concentration.get("weights") or {}).get("top_10")
    forward_pe = valuation.get("forward_pe")
    kpis.append(
        {
            "label": "Top 10 share of index",
            "value": format_percent(top_10, decimals=1),
            "delta": f"Median forward P/E {format_number(forward_pe, 1)}"
            if forward_pe
            else None,
            "tone": "accent",
        }
    )
    return kpis


def render_sidebar(stats: Dict[str, object], last_refresh: Optional[datetime]) -> Dict[str, object]:
    """Draw the sidebar and return the chosen settings."""
    with st.sidebar:
        st.markdown("## Data controls")

        universe = st.selectbox(
            "Universe size",
            options=["Full index (500)", "Top 250", "Top 100", "Top 50"],
            index=0,
            help="A smaller universe downloads faster. Useful for a first run.",
        )
        limit_map = {
            "Full index (500)": None,
            "Top 250": 250,
            "Top 100": 100,
            "Top 50": 50,
        }

        period = st.selectbox(
            "History depth",
            options=["2y", "3y", "5y"],
            index=0,
            help="Two years is the minimum for a reliable 200-day average.",
        )

        workers = st.slider(
            "Parallel downloads",
            min_value=2,
            max_value=24,
            value=12,
            help="Higher is faster until Yahoo starts throttling. 8-12 is a good range.",
        )

        refresh = st.button("Refresh market data", width="stretch", type="primary")
        reset = st.button("Clear cached data", width="stretch")

        st.markdown("### Cache status")
        st.markdown(
            f"""
            <div class="sidebar-note">
              Tickers stored: {stats.get('tickers', 0)}<br>
              Price rows: {int(stats.get('price_rows', 0)):,}<br>
              Coverage: {stats.get('first_date') or '--'} to {stats.get('last_date') or '--'}<br>
              Database size: {stats.get('size_mb', 0)} MB<br>
              Last refresh: {format_timestamp(last_refresh)}
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown("### About")
        st.markdown(
            """
            <div class="sidebar-note">
              Prices and fundamentals come from Yahoo Finance, the 10-year yield
              from FRED, and index membership from Wikipedia. Figures are
              delayed and are not investment advice.
            </div>
            """,
            unsafe_allow_html=True,
        )

    return {
        "limit": limit_map[universe],
        "period": period,
        "workers": workers,
        "refresh": refresh,
        "reset": reset,
    }


def render_disclaimer(text: Optional[str] = None) -> None:
    default = (
        "These projections are a statistical exercise on historical prices, not a "
        "forecast of what markets will do. Model output ignores earnings, policy, "
        "liquidity and every other thing that actually moves prices. Do not trade on it."
    )
    st.markdown(f'<div class="disclaimer">{text or default}</div>', unsafe_allow_html=True)


def render_footnote(text: str) -> None:
    st.markdown(f'<div class="footnote">{text}</div>', unsafe_allow_html=True)


def empty_state(message: str, hint: str = "") -> None:
    st.markdown(
        f"""
        <div class="panel">
          <div style="font-weight:600;margin-bottom:0.3rem;">{message}</div>
          <div style="color:var(--text-muted);font-size:0.86rem;">{hint}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def style_figure(figure, height: int = 420, legend_top: bool = True):
    """Apply the shared dark styling to any Plotly figure."""
    figure.update_layout(
        template=PLOTLY_TEMPLATE,
        height=height,
        margin=dict(l=10, r=10, t=40, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color=CHART_COLORS["text"], size=12),
        hovermode="x unified",
    )
    if legend_top:
        figure.update_layout(
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="left",
                x=0,
                title_text="",
            )
        )
    figure.update_xaxes(gridcolor=CHART_COLORS["line"], zerolinecolor=CHART_COLORS["line"])
    figure.update_yaxes(gridcolor=CHART_COLORS["line"], zerolinecolor=CHART_COLORS["line"])
    return figure
