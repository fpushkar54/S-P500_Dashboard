"""Tab 1 - Valuation & Macro.

Answers one question: what are investors paying for these earnings, and what
is the risk-free alternative doing? The dual-axis chart is the anchor, since
index level and 10-year yield moving in opposite directions is the single most
useful macro tell on the page.
"""

from __future__ import annotations

from typing import Dict, Optional

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from ui.components import (
    CHART_COLORS,
    empty_state,
    format_number,
    format_percent,
    render_footnote,
    section_header,
    style_figure,
    with_alpha,
)


def _index_vs_yield_chart(sp500: pd.Series, yields: pd.Series) -> go.Figure:
    figure = make_subplots(specs=[[{"secondary_y": True}]])

    figure.add_trace(
        go.Scatter(
            x=sp500.index,
            y=sp500.values,
            name="S&P 500",
            mode="lines",
            line=dict(color=CHART_COLORS["accent"], width=2),
            hovertemplate="S&P 500: %{y:,.0f}<extra></extra>",
        ),
        secondary_y=False,
    )

    if yields is not None and len(yields) > 0:
        figure.add_trace(
            go.Scatter(
                x=yields.index,
                y=yields.values,
                name="10-year Treasury yield",
                mode="lines",
                line=dict(color=CHART_COLORS["neutral"], width=1.8, dash="dot"),
                hovertemplate="10Y: %{y:.2f}%<extra></extra>",
            ),
            secondary_y=True,
        )

    figure.update_yaxes(title_text="Index level", secondary_y=False)
    figure.update_yaxes(title_text="Yield (%)", secondary_y=True, showgrid=False)
    return style_figure(figure, height=440)


def _sector_pe_chart(sector_valuation: pd.DataFrame, use_weighted: bool) -> go.Figure:
    column = "forward_pe_weighted" if use_weighted else "forward_pe_median"
    frame = sector_valuation.dropna(subset=[column]).sort_values(column)

    index_median = frame[column].median()
    colors = [
        CHART_COLORS["down"] if value > index_median else CHART_COLORS["neutral"]
        for value in frame[column]
    ]

    figure = go.Figure(
        go.Bar(
            x=frame[column],
            y=frame["sector"],
            orientation="h",
            marker=dict(color=colors),
            text=[f"{value:.1f}" for value in frame[column]],
            textposition="outside",
            customdata=frame[["companies", "trailing_pe_median"]].values,
            hovertemplate=(
                "%{y}<br>Forward P/E: %{x:.1f}"
                "<br>Trailing P/E: %{customdata[1]:.1f}"
                "<br>Companies: %{customdata[0]}<extra></extra>"
            ),
        )
    )
    figure.add_vline(
        x=index_median,
        line=dict(color=CHART_COLORS["accent"], width=1.5, dash="dash"),
        annotation_text=f"Index median {index_median:.1f}",
        annotation_position="top",
        annotation_font_color=CHART_COLORS["accent"],
    )
    figure.update_xaxes(title_text="Forward price / earnings")
    return style_figure(figure, height=440, legend_top=False)


def _vix_gauge(vix_level: float) -> go.Figure:
    figure = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=vix_level,
            number=dict(font=dict(size=42, color=CHART_COLORS["text"])),
            gauge=dict(
                axis=dict(range=[0, 55], tickcolor=CHART_COLORS["muted"]),
                bar=dict(color=CHART_COLORS["text"], thickness=0.22),
                bgcolor="rgba(0,0,0,0)",
                borderwidth=0,
                steps=[
                    dict(range=[0, 15], color=with_alpha("up", 0.28)),
                    dict(range=[15, 20], color=with_alpha("neutral", 0.28)),
                    dict(range=[20, 30], color=with_alpha("warning", 0.30)),
                    dict(range=[30, 55], color=with_alpha("down", 0.33)),
                ],
                threshold=dict(
                    line=dict(color=CHART_COLORS["accent"], width=3),
                    thickness=0.8,
                    value=20,
                ),
            ),
        )
    )
    return style_figure(figure, height=260, legend_top=False)


def _vix_regime(vix_level: float) -> str:
    if vix_level < 15:
        return "Calm. Options markets are pricing small daily moves."
    if vix_level < 20:
        return "Normal. Volatility sits near its long-run average."
    if vix_level < 30:
        return "Elevated. Hedging demand is rising."
    return "Stressed. This range historically clusters around drawdowns."


def render(
    macro: Dict[str, pd.Series],
    sector_valuation: pd.DataFrame,
    valuation_summary: Dict[str, Optional[float]],
) -> None:
    """Render the valuation and macro tab."""
    sp500 = macro.get("SP500")
    vix = macro.get("VIX")
    yields = macro.get("DGS10")

    if sp500 is None or len(sp500) == 0:
        empty_state(
            "No index history cached yet.",
            "Use 'Refresh market data' in the sidebar to download S&P 500, VIX and Treasury series.",
        )
        return

    window = st.radio(
        "Window",
        options=["6 months", "1 year", "2 years", "Everything"],
        index=1,
        horizontal=True,
        label_visibility="collapsed",
    )
    sessions = {"6 months": 126, "1 year": 252, "2 years": 504, "Everything": len(sp500)}[window]

    sp500_window = sp500.tail(sessions)
    yields_window = (
        yields.reindex(sp500.index).ffill().tail(sessions)
        if yields is not None and len(yields)
        else pd.Series(dtype=float)
    )

    change_pct = (
        (sp500_window.iloc[-1] / sp500_window.iloc[0] - 1.0) * 100.0 if len(sp500_window) > 1 else None
    )
    section_header(
        "Index level against the risk-free rate",
        f"{window} &middot; index {format_percent(change_pct, signed=True)}",
    )
    st.plotly_chart(_index_vs_yield_chart(sp500_window, yields_window), width="stretch")

    left, right = st.columns([2, 1], gap="medium")

    with left:
        if sector_valuation is None or sector_valuation.empty:
            empty_state("No sector valuation data.", "Fundamentals load with a full refresh.")
        else:
            weighted = st.toggle(
                "Weight by market cap",
                value=False,
                help="Off: plain median across the sector. On: cap-weighted median, so "
                "megacaps drive the reading.",
            )
            forward_pe = valuation_summary.get("forward_pe")
            trailing_pe = valuation_summary.get("trailing_pe")
            section_header(
                "Forward P/E by GICS sector",
                f"Index median forward {format_number(forward_pe, 1)} &middot; "
                f"trailing {format_number(trailing_pe, 1)}",
            )
            st.plotly_chart(_sector_pe_chart(sector_valuation, weighted), width="stretch")

    with right:
        section_header("Volatility", "CBOE VIX, latest close")
        if vix is None or len(vix) == 0:
            empty_state("No VIX data cached.")
        else:
            level = float(vix.iloc[-1])
            st.plotly_chart(_vix_gauge(level), width="stretch")
            st.markdown(
                f'<div class="panel" style="margin-top:-0.5rem;">{_vix_regime(level)}</div>',
                unsafe_allow_html=True,
            )

            if len(vix) > 252:
                percentile = float((vix.tail(252) < level).mean() * 100.0)
                st.markdown(
                    f'<div class="sidebar-note">Higher than {percentile:.0f}% of closes '
                    f"over the past year.</div>",
                    unsafe_allow_html=True,
                )

    if sector_valuation is not None and not sector_valuation.empty:
        section_header("Sector detail", "Median and cap-weighted multiples")
        display = sector_valuation.copy()
        display = display.rename(
            columns={
                "sector": "Sector",
                "companies": "Companies",
                "weight_pct": "Index weight %",
                "trailing_pe_median": "Trailing P/E (median)",
                "forward_pe_median": "Forward P/E (median)",
                "trailing_pe_weighted": "Trailing P/E (weighted)",
                "forward_pe_weighted": "Forward P/E (weighted)",
            }
        ).drop(columns=["market_cap"], errors="ignore")
        st.dataframe(
            display,
            width="stretch",
            hide_index=True,
            column_config={
                "Index weight %": st.column_config.NumberColumn(format="%.2f"),
                "Trailing P/E (median)": st.column_config.NumberColumn(format="%.1f"),
                "Forward P/E (median)": st.column_config.NumberColumn(format="%.1f"),
                "Trailing P/E (weighted)": st.column_config.NumberColumn(format="%.1f"),
                "Forward P/E (weighted)": st.column_config.NumberColumn(format="%.1f"),
            },
        )

    render_footnote(
        "P/E ratios outside 0-200 are dropped as data errors. Forward multiples use "
        "analyst consensus from Yahoo Finance and change as estimates are revised."
    )
