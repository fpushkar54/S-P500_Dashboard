"""Tab 4 - Screener & Sector Drill-Down.

A sortable table of every tracked constituent, with the filters that matter for
index health work: sector, market cap, valuation, and position relative to the
200-day average.
"""

from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from ui.components import (
    CHART_COLORS,
    empty_state,
    format_market_cap,
    format_percent,
    render_footnote,
    section_header,
    style_figure,
)

COLUMN_LABELS = {
    "ticker": "Ticker",
    "name": "Company",
    "sector": "Sector",
    "market_cap": "Market cap",
    "price": "Price",
    "trailing_pe": "Trailing P/E",
    "forward_pe": "Forward P/E",
    "dist_from_200d_pct": "vs 200d MA %",
    "dist_from_50d_pct": "vs 50d MA %",
    "return_1m_pct": "1m return %",
    "return_6m_pct": "6m return %",
}


def _sector_performance_chart(performance: pd.DataFrame, label: str) -> go.Figure:
    frame = performance.sort_values("return_pct")
    colors = [
        CHART_COLORS["up"] if value >= 0 else CHART_COLORS["down"] for value in frame["return_pct"]
    ]
    figure = go.Figure(
        go.Bar(
            x=frame["return_pct"],
            y=frame["sector"],
            orientation="h",
            marker=dict(color=colors),
            text=[f"{value:+.1f}%" for value in frame["return_pct"]],
            textposition="outside",
            hovertemplate="%{y}: %{x:+.2f}%<extra></extra>",
        )
    )
    figure.update_xaxes(title_text=f"Cap-weighted {label} return")
    return style_figure(figure, height=400, legend_top=False)


def _distance_histogram(table: pd.DataFrame) -> go.Figure:
    values = table["dist_from_200d_pct"].dropna()
    figure = go.Figure(
        go.Histogram(
            x=values,
            nbinsx=40,
            marker=dict(color=CHART_COLORS["neutral"], line=dict(width=0)),
            hovertemplate="%{x:.0f}%: %{y} stocks<extra></extra>",
        )
    )
    figure.add_vline(
        x=0,
        line=dict(color=CHART_COLORS["accent"], width=2),
        annotation_text="200-day average",
        annotation_position="top",
        annotation_font_color=CHART_COLORS["accent"],
    )
    figure.update_xaxes(title_text="Distance from 200-day average (%)")
    figure.update_yaxes(title_text="Constituents")
    return style_figure(figure, height=400, legend_top=False)


def render(screener: pd.DataFrame, sector_performance: pd.DataFrame) -> None:
    """Render the screener tab."""
    if screener is None or screener.empty:
        empty_state(
            "No constituent data cached.",
            "Refresh market data from the sidebar to populate the screener.",
        )
        return

    sectors = sorted(screener["sector"].dropna().unique().tolist())

    filter_row = st.columns([3, 2, 2, 2], gap="medium")
    with filter_row[0]:
        chosen_sectors = st.multiselect(
            "GICS sector", options=sectors, default=[], placeholder="All sectors"
        )
    with filter_row[1]:
        trend = st.selectbox(
            "Trend position", options=["Any", "Above 200d MA", "Below 200d MA"], index=0
        )
    with filter_row[2]:
        max_pe = st.number_input(
            "Max forward P/E", min_value=0.0, max_value=200.0, value=200.0, step=5.0
        )
    with filter_row[3]:
        min_cap_b = st.number_input(
            "Min market cap ($B)", min_value=0.0, max_value=5000.0, value=0.0, step=10.0
        )

    search = st.text_input(
        "Search", placeholder="Ticker or company name", label_visibility="collapsed"
    )

    filtered = screener.copy()
    if chosen_sectors:
        filtered = filtered[filtered["sector"].isin(chosen_sectors)]
    if trend == "Above 200d MA":
        filtered = filtered[filtered["above_200d"] == True]  # noqa: E712
    elif trend == "Below 200d MA":
        filtered = filtered[filtered["above_200d"] == False]  # noqa: E712
    if max_pe < 200.0:
        filtered = filtered[filtered["forward_pe"].notna() & (filtered["forward_pe"] <= max_pe)]
    if min_cap_b > 0:
        filtered = filtered[filtered["market_cap"].fillna(0) >= min_cap_b * 1e9]
    if search:
        needle = search.strip().lower()
        mask = filtered["ticker"].str.lower().str.contains(needle, na=False) | filtered[
            "name"
        ].astype(str).str.lower().str.contains(needle, na=False)
        filtered = filtered[mask]

    above = int(filtered["above_200d"].sum()) if "above_200d" in filtered.columns else 0
    total = len(filtered)
    section_header(
        "Constituents",
        f"{total} of {len(screener)} shown &middot; {above} above their 200-day average "
        f"&middot; {format_market_cap(filtered['market_cap'].sum())} combined cap",
    )

    display = filtered[list(COLUMN_LABELS.keys())].rename(columns=COLUMN_LABELS)
    # Billions keeps the column sortable while staying readable.
    display["Market cap"] = pd.to_numeric(display["Market cap"], errors="coerce") / 1e9
    display = display.rename(columns={"Market cap": "Market cap ($B)"})

    # Symbols carry the accent so the eye can scan the column without reading it.
    styled = display.style.map(
        lambda _: f"color: {CHART_COLORS['accent']}; font-weight: 600;", subset=["Ticker"]
    )

    st.dataframe(
        styled,
        width="stretch",
        hide_index=True,
        height=520,
        column_config={
            "Market cap ($B)": st.column_config.NumberColumn(
                format="%.1f", help="Current market capitalisation in billions of dollars"
            ),
            "Price": st.column_config.NumberColumn(format="$%.2f"),
            "Trailing P/E": st.column_config.NumberColumn(format="%.1f"),
            "Forward P/E": st.column_config.NumberColumn(format="%.1f"),
            "vs 200d MA %": st.column_config.NumberColumn(format="%.1f%%"),
            "vs 50d MA %": st.column_config.NumberColumn(format="%.1f%%"),
            "1m return %": st.column_config.NumberColumn(format="%.1f%%"),
            "6m return %": st.column_config.NumberColumn(format="%.1f%%"),
        },
    )

    csv = display.to_csv(index=False).encode("utf-8")
    st.download_button(
        "Download this view as CSV",
        data=csv,
        file_name="sp500_screener.csv",
        mime="text/csv",
    )

    st.write("")
    left, right = st.columns(2, gap="medium")

    with left:
        if sector_performance is None or sector_performance.empty:
            empty_state("Not enough history for sector returns.")
        else:
            section_header("Sector momentum", "Trailing month, weighted by market cap")
            st.plotly_chart(
                _sector_performance_chart(sector_performance, "1-month"), width="stretch"
            )

    with right:
        section_header("Trend distribution", "How far each stock sits from its 200-day average")
        st.plotly_chart(_distance_histogram(filtered), width="stretch")

    if chosen_sectors:
        section_header("Sector detail", " &middot; ".join(chosen_sectors))
        stats = []
        for sector in chosen_sectors:
            group = screener[screener["sector"] == sector]
            stats.append(
                {
                    "Sector": sector,
                    "Companies": len(group),
                    "Above 200d MA": int(group["above_200d"].sum()),
                    "Median forward P/E": group["forward_pe"].median(),
                    "Median 1m return %": group["return_1m_pct"].median(),
                    "Combined cap": format_market_cap(group["market_cap"].sum()),
                }
            )
        st.dataframe(
            pd.DataFrame(stats),
            width="stretch",
            hide_index=True,
            column_config={
                "Median forward P/E": st.column_config.NumberColumn(format="%.1f"),
                "Median 1m return %": st.column_config.NumberColumn(format="%.1f%%"),
            },
        )

    render_footnote(
        "Distance from the 200-day average uses the latest cached close. Stocks with fewer "
        "than 200 sessions of history — recent index additions — show blank trend columns."
    )
