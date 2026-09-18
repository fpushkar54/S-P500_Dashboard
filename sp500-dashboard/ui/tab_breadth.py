"""Tab 2 - Market Breadth & Concentration.

Breadth answers "how many stocks are actually participating", concentration
answers "how much of the index is really just a handful of names". Read
together they explain most cases where the headline index and the median
stock tell different stories.
"""

from __future__ import annotations

from typing import Dict

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from backend import metrics
from ui.components import (
    CHART_COLORS,
    MUTED_FILL,
    SECTOR_PALETTE,
    SEQUENTIAL_SCALE,
    empty_state,
    format_market_cap,
    format_percent,
    render_footnote,
    section_header,
    style_figure,
    with_alpha,
)


def _breadth_chart(breadth: pd.DataFrame) -> go.Figure:
    figure = go.Figure()

    figure.add_trace(
        go.Scatter(
            x=breadth.index,
            y=breadth["pct_above_50"],
            name="Above 50-day average",
            mode="lines",
            line=dict(color=CHART_COLORS["accent"], width=1.8),
            hovertemplate="50d: %{y:.1f}%<extra></extra>",
        )
    )
    figure.add_trace(
        go.Scatter(
            x=breadth.index,
            y=breadth["pct_above_200"],
            name="Above 200-day average",
            mode="lines",
            line=dict(color=CHART_COLORS["neutral"], width=2),
            hovertemplate="200d: %{y:.1f}%<extra></extra>",
        )
    )

    for level, label in ((50, "Half the index"), (80, "Overbought zone"), (20, "Washed out")):
        figure.add_hline(
            y=level,
            line=dict(color=CHART_COLORS["line"], width=1, dash="dash"),
            annotation_text=label,
            annotation_position="right",
            annotation_font_color=CHART_COLORS["muted"],
            annotation_font_size=10,
        )

    figure.update_yaxes(title_text="% of constituents", range=[0, 100])
    return style_figure(figure, height=420)


def _ad_line_chart(ad_line: pd.DataFrame, sp500: pd.Series) -> go.Figure:
    from plotly.subplots import make_subplots

    figure = make_subplots(specs=[[{"secondary_y": True}]])

    figure.add_trace(
        go.Scatter(
            x=ad_line.index,
            y=ad_line["ad_line"],
            name="Cumulative advance-decline",
            mode="lines",
            line=dict(color=CHART_COLORS["up"], width=2),
            fill="tozeroy",
            fillcolor=with_alpha("up", 0.10),
            hovertemplate="A/D line: %{y:,.0f}<extra></extra>",
        ),
        secondary_y=False,
    )

    if sp500 is not None and len(sp500) > 0:
        aligned = sp500.reindex(ad_line.index).ffill()
        figure.add_trace(
            go.Scatter(
                x=aligned.index,
                y=aligned.values,
                name="S&P 500",
                mode="lines",
                line=dict(color=CHART_COLORS["muted"], width=1.4, dash="dot"),
                hovertemplate="S&P 500: %{y:,.0f}<extra></extra>",
            ),
            secondary_y=True,
        )

    figure.update_yaxes(title_text="Cumulative net advances", secondary_y=False)
    figure.update_yaxes(title_text="Index level", secondary_y=True, showgrid=False)
    return style_figure(figure, height=400)


def _net_advances_chart(ad_line: pd.DataFrame) -> go.Figure:
    recent = ad_line.tail(60)
    colors = [
        CHART_COLORS["up"] if value >= 0 else CHART_COLORS["down"] for value in recent["net"]
    ]
    figure = go.Figure(
        go.Bar(
            x=recent.index,
            y=recent["net"],
            marker=dict(color=colors),
            hovertemplate="Net advances: %{y:+,.0f}<extra></extra>",
        )
    )
    figure.update_yaxes(title_text="Advancers minus decliners")
    return style_figure(figure, height=280, legend_top=False)


def _concentration_donut(donut_frame: pd.DataFrame, top_n: int) -> go.Figure:
    colors = SECTOR_PALETTE[: len(donut_frame)]
    if len(donut_frame) > len(colors):
        colors = (colors * (len(donut_frame) // len(colors) + 1))[: len(donut_frame)]
    if len(donut_frame):
        colors[-1] = MUTED_FILL  # the "rest of index" slice recedes

    figure = go.Figure(
        go.Pie(
            labels=donut_frame["label"],
            values=donut_frame["weight_pct"],
            hole=0.62,
            sort=False,
            marker=dict(colors=colors, line=dict(color=CHART_COLORS["panel"], width=2)),
            textinfo="label",
            textfont=dict(size=11),
            hovertemplate="%{label}: %{value:.2f}% of index<extra></extra>",
        )
    )
    top_weight = donut_frame["weight_pct"].head(top_n).sum()
    figure.add_annotation(
        text=f"<b>{top_weight:.1f}%</b><br><span style='font-size:11px'>top {top_n}</span>",
        showarrow=False,
        font=dict(size=24, color=CHART_COLORS["text"]),
    )
    return style_figure(figure, height=400, legend_top=False)


def _treemap(ranked: pd.DataFrame, top_n: int = 60) -> go.Figure:
    frame = ranked.head(top_n).copy()
    frame["sector"] = frame["sector"].fillna("Unclassified")
    frame["label"] = frame["ticker"]

    figure = px.treemap(
        frame,
        path=["sector", "label"],
        values="market_cap",
        color="weight_pct",
        color_continuous_scale=SEQUENTIAL_SCALE,
        custom_data=["weight_pct"],
    )
    figure.update_traces(
        marker=dict(line=dict(color=CHART_COLORS["panel"], width=1)),
        hovertemplate="%{label}<br>%{customdata[0]:.2f}% of index<extra></extra>",
        textfont=dict(size=12),
    )
    figure.update_layout(coloraxis_showscale=False)
    return style_figure(figure, height=440, legend_top=False)


def render(
    breadth: pd.DataFrame,
    ad_line: pd.DataFrame,
    concentration: Dict[str, object],
    macro: Dict[str, pd.Series],
) -> None:
    """Render the breadth and concentration tab."""
    if breadth is None or breadth.empty:
        empty_state(
            "Breadth needs at least 200 sessions of history per ticker.",
            "Refresh with a 2-year window to populate the moving averages.",
        )
    else:
        latest = breadth.dropna(subset=["pct_above_200"]).tail(1)
        note = ""
        if not latest.empty:
            row = latest.iloc[0]
            note = (
                f"{format_percent(row['pct_above_50'])} above 50d &middot; "
                f"{format_percent(row['pct_above_200'])} above 200d &middot; "
                f"{int(row['universe_200'])} tickers with full history"
            )
        section_header("Participation over time", note)
        st.plotly_chart(_breadth_chart(breadth), width="stretch")

    if ad_line is not None and not ad_line.empty:
        left, right = st.columns([3, 2], gap="medium")
        with left:
            section_header(
                "Cumulative advance-decline line",
                "A rising index with a falling A/D line means a narrowing rally",
            )
            st.plotly_chart(
                _ad_line_chart(ad_line, macro.get("SP500")), width="stretch"
            )
        with right:
            section_header("Daily net advances", "Last 60 sessions")
            st.plotly_chart(_net_advances_chart(ad_line), width="stretch")

            up_days = int((ad_line["net"].tail(60) > 0).sum())
            st.markdown(
                f'<div class="panel">Advancers beat decliners on <b>{up_days}</b> of the '
                f"last 60 sessions.</div>",
                unsafe_allow_html=True,
            )

    weights = concentration.get("weights") or {}
    ranked = concentration.get("ranked")
    if not weights or ranked is None or len(ranked) == 0:
        return

    section_header(
        "Concentration",
        f"{concentration.get('constituents', 0)} constituents &middot; "
        f"{format_market_cap(concentration.get('total_market_cap'))} total market cap",
    )

    tier_columns = st.columns(4, gap="small")
    tiers = [("Top 5", weights.get("top_5")), ("Top 10", weights.get("top_10")), ("Top 20", weights.get("top_20"))]
    for column, (label, value) in zip(tier_columns, tiers):
        with column:
            st.markdown(
                f'<div class="kpi-card is-accent"><div class="kpi-label">{label} weight</div>'
                f'<div class="kpi-value">{format_percent(value)}</div>'
                f'<div class="kpi-delta">of total index market cap</div></div>',
                unsafe_allow_html=True,
            )
    with tier_columns[3]:
        largest = ranked.iloc[0]
        st.markdown(
            f'<div class="kpi-card"><div class="kpi-label">Largest constituent</div>'
            f'<div class="kpi-value ticker">{largest["ticker"]}</div>'
            f'<div class="kpi-delta">{format_percent(largest["weight_pct"], 2)} of the index</div></div>',
            unsafe_allow_html=True,
        )

    st.write("")
    donut_column, treemap_column = st.columns([2, 3], gap="medium")

    with donut_column:
        top_n = st.select_slider("Holdings shown", options=[5, 10, 20], value=10)
        donut_frame = metrics.concentration_donut_frame(concentration, top_n=top_n)
        st.plotly_chart(_concentration_donut(donut_frame, top_n), width="stretch")

    with treemap_column:
        section_header("Market cap map", "Largest 60 constituents, grouped by sector")
        st.plotly_chart(_treemap(ranked), width="stretch")

    top_holdings = concentration.get("top_holdings")
    if top_holdings is not None and not top_holdings.empty:
        display = top_holdings.copy()
        display["market_cap"] = display["market_cap"].map(format_market_cap)
        display = display.rename(
            columns={
                "ticker": "Ticker",
                "name": "Company",
                "sector": "Sector",
                "market_cap": "Market cap",
                "weight_pct": "Weight %",
                "cumulative_weight_pct": "Cumulative %",
            }
        )
        st.dataframe(
            display.style.map(
                lambda _: f"color: {CHART_COLORS['accent']}; font-weight: 600;",
                subset=["Ticker"],
            ),
            width="stretch",
            hide_index=True,
            column_config={
                "Weight %": st.column_config.NumberColumn(format="%.2f"),
                "Cumulative %": st.column_config.NumberColumn(format="%.2f"),
            },
        )

    render_footnote(
        "Weights are computed from current market caps across the tickers in the cache, "
        "so they approximate but do not exactly reproduce the official float-adjusted "
        "index weights."
    )
