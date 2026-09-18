"""Tab 3 - Predictive Analytics.

A Random Forest turns today's technical and volatility state into a directional
probability, and an ARIMA fit draws the projection cone. Both are framed as
what they are: pattern-matching on price history with wide error bars.
"""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from backend import ml_engine
from ui.components import (
    CHART_COLORS,
    empty_state,
    format_number,
    format_percent,
    render_disclaimer,
    render_footnote,
    section_header,
    style_figure,
    with_alpha,
)

HORIZON_LABELS = {5: "1 week ahead", 20: "1 month ahead", 60: "1 quarter ahead"}


# --------------------------------------------------------------------------- #
# Cached model runs
# --------------------------------------------------------------------------- #


@st.cache_data(show_spinner=False, ttl=60 * 60)
def _cached_features(
    index_close: pd.Series, vix_close: Optional[pd.Series], breadth: Optional[pd.DataFrame]
) -> pd.DataFrame:
    return ml_engine.build_feature_frame(index_close, vix_close=vix_close, breadth=breadth)


@st.cache_data(show_spinner=False, ttl=60 * 60)
def _cached_models(features: pd.DataFrame, horizons: tuple) -> Dict[int, Dict[str, object]]:
    return ml_engine.train_direction_models(features, horizons=horizons)


@st.cache_data(show_spinner=False, ttl=60 * 60)
def _cached_forecast(index_close: pd.Series, steps: int) -> Dict[str, object]:
    return ml_engine.arima_forecast(index_close, steps=steps)


# --------------------------------------------------------------------------- #
# Charts
# --------------------------------------------------------------------------- #


def _probability_gauge(probability: float) -> go.Figure:
    color = (
        CHART_COLORS["up"]
        if probability >= 0.58
        else CHART_COLORS["down"]
        if probability <= 0.42
        else CHART_COLORS["neutral"]
    )
    figure = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=probability * 100.0,
            number=dict(suffix="%", font=dict(size=34, color=CHART_COLORS["text"])),
            gauge=dict(
                axis=dict(range=[0, 100], tickcolor=CHART_COLORS["muted"]),
                bar=dict(color=color, thickness=0.3),
                bgcolor="rgba(0,0,0,0)",
                borderwidth=0,
                steps=[
                    dict(range=[0, 42], color=with_alpha("down", 0.18)),
                    dict(range=[42, 58], color=with_alpha("neutral", 0.18)),
                    dict(range=[58, 100], color=with_alpha("up", 0.18)),
                ],
                threshold=dict(
                    line=dict(color=CHART_COLORS["text"], width=2), thickness=0.75, value=50
                ),
            ),
        )
    )
    return style_figure(figure, height=230, legend_top=False)


def _importance_chart(importances: pd.DataFrame, top_n: int = 12) -> go.Figure:
    frame = importances.head(top_n).iloc[::-1]
    figure = go.Figure(
        go.Bar(
            x=frame["importance"],
            y=frame["label"],
            orientation="h",
            marker=dict(color=CHART_COLORS["accent"]),
            hovertemplate="%{y}: %{x:.3f}<extra></extra>",
        )
    )
    figure.update_xaxes(title_text="Mean decrease in impurity")
    return style_figure(figure, height=400, legend_top=False)


def _forecast_chart(history: pd.Series, forecast: pd.DataFrame, history_days: int = 120) -> go.Figure:
    recent = history.tail(history_days)
    figure = go.Figure()

    # Bands are drawn widest-first so the 80% cone sits on top of the 95% cone.
    for lower, upper, fill, name in (
        ("lower_95", "upper_95", with_alpha("neutral", 0.16), "95% band"),
        ("lower_80", "upper_80", with_alpha("accent", 0.20), "80% band"),
    ):
        figure.add_trace(
            go.Scatter(
                x=list(forecast["date"]) + list(forecast["date"][::-1]),
                y=list(forecast[upper]) + list(forecast[lower][::-1]),
                fill="toself",
                fillcolor=fill,
                line=dict(width=0),
                name=name,
                hoverinfo="skip",
            )
        )

    figure.add_trace(
        go.Scatter(
            x=recent.index,
            y=recent.values,
            name="Actual",
            mode="lines",
            line=dict(color=CHART_COLORS["text"], width=2),
            hovertemplate="%{y:,.0f}<extra></extra>",
        )
    )
    figure.add_trace(
        go.Scatter(
            x=forecast["date"],
            y=forecast["forecast"],
            name="Central projection",
            mode="lines",
            line=dict(color=CHART_COLORS["accent"], width=2, dash="dash"),
            hovertemplate="%{y:,.0f}<extra></extra>",
        )
    )

    figure.update_yaxes(title_text="Index level")
    return style_figure(figure, height=460)


# --------------------------------------------------------------------------- #
# Blocks
# --------------------------------------------------------------------------- #


def _render_signal_block(model: Dict[str, object], horizon: int) -> None:
    signal = str(model.get("signal", "UNAVAILABLE"))
    probability = model.get("probability_up")
    tone = {"BULLISH": "up", "BEARISH": "down"}.get(signal, "flat")

    st.markdown(
        f"""
        <div class="signal-block">
          <span class="badge badge-{signal.lower() if signal in ('BULLISH', 'BEARISH', 'NEUTRAL') else 'neutral'}">
            {HORIZON_LABELS.get(horizon, f'{horizon} sessions')}
          </span>
          <div class="headline {tone}">{signal}</div>
          <div class="caption">
            {format_percent((probability or 0) * 100, 1)} modelled probability the index
            closes higher in {horizon} sessions
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _render_model_stats(model: Dict[str, object]) -> None:
    accuracy = model.get("accuracy")
    cv_accuracy = model.get("cv_accuracy")
    auc = model.get("roc_auc")
    baseline = model.get("baseline")

    rows = [
        ("Holdout accuracy", format_percent((accuracy or 0) * 100, 1) if accuracy else "--"),
        (
            "Walk-forward accuracy",
            format_percent((cv_accuracy or 0) * 100, 1) if cv_accuracy else "--",
        ),
        ("ROC AUC", format_number(auc, 3) if auc else "--"),
        (
            "Always-up baseline",
            format_percent((baseline or 0) * 100, 1) if baseline else "--",
        ),
        ("Training rows", f"{model.get('train_rows', 0):,}"),
    ]
    body = "".join(
        f'<div style="display:flex;justify-content:space-between;padding:0.3rem 0;'
        f'border-bottom:1px solid var(--line);"><span style="color:var(--text-muted);">{label}</span>'
        f"<span><b>{value}</b></span></div>"
        for label, value in rows
    )
    st.markdown(f'<div class="panel">{body}</div>', unsafe_allow_html=True)


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #


def render(
    macro: Dict[str, pd.Series],
    breadth: Optional[pd.DataFrame] = None,
    horizons: tuple = (5, 20, 60),
) -> None:
    """Render the prediction tab."""
    index_close = macro.get("SP500")
    if index_close is None or len(index_close) < 260:
        empty_state(
            "The models need at least one year of index history.",
            "Refresh market data, then come back to this tab.",
        )
        return

    render_disclaimer()
    st.write("")

    with st.spinner("Building features and fitting models..."):
        features = _cached_features(index_close, macro.get("VIX"), breadth)
        models = _cached_models(features, tuple(horizons))
        forecast_result = _cached_forecast(index_close, ml_engine.FORECAST_STEPS)

    if features.empty:
        empty_state("Feature engineering produced no usable rows.", "Try a longer history window.")
        return

    horizon = st.radio(
        "Forecast horizon",
        options=list(horizons),
        format_func=lambda value: HORIZON_LABELS.get(value, f"{value} sessions"),
        index=1 if len(horizons) > 1 else 0,
        horizontal=True,
    )
    model = models.get(int(horizon), {})

    if model.get("error"):
        st.warning(model["error"])
    else:
        signal_column, gauge_column, stats_column = st.columns([2, 2, 2], gap="medium")

        with signal_column:
            _render_signal_block(model, int(horizon))
        with gauge_column:
            probability = model.get("probability_up")
            if probability is not None:
                st.plotly_chart(_probability_gauge(float(probability)), width="stretch")
        with stats_column:
            _render_model_stats(model)

        importances = model.get("importances")
        if importances is not None and not importances.empty:
            left, right = st.columns([3, 2], gap="medium")
            with left:
                section_header(
                    "What the model is watching",
                    f"Feature importance, {len(features.columns) - 1} inputs",
                )
                st.plotly_chart(_importance_chart(importances), width="stretch")
            with right:
                section_header("All horizons", "Probability the index closes higher")
                summary_rows = []
                for key in horizons:
                    entry = models.get(int(key), {})
                    summary_rows.append(
                        {
                            "Horizon": HORIZON_LABELS.get(key, f"{key} sessions"),
                            "Signal": entry.get("signal", "--"),
                            "P(up)": (entry.get("probability_up") or np.nan) * 100,
                            "Accuracy": (entry.get("accuracy") or np.nan) * 100,
                        }
                    )
                st.dataframe(
                    pd.DataFrame(summary_rows),
                    width="stretch",
                    hide_index=True,
                    column_config={
                        "P(up)": st.column_config.NumberColumn(format="%.1f%%"),
                        "Accuracy": st.column_config.NumberColumn(format="%.1f%%"),
                    },
                )
                st.markdown(
                    '<div class="sidebar-note">Accuracy is measured on the most recent '
                    "20% of sessions, which the model never saw during training. Anything "
                    "close to the always-up baseline is noise, not skill.</div>",
                    unsafe_allow_html=True,
                )

    st.write("")
    if forecast_result.get("error"):
        st.warning(forecast_result["error"])
        return

    forecast = forecast_result.get("forecast")
    if forecast is None or forecast.empty:
        return

    order = forecast_result.get("order")
    expected_change = forecast_result.get("expected_change_pct")
    section_header(
        f"{len(forecast)}-session projection cone",
        f"ARIMA{order} on log prices &middot; central path "
        f"{format_percent(expected_change, 1, signed=True)}",
    )
    st.plotly_chart(_forecast_chart(index_close, forecast), width="stretch")

    final = forecast.iloc[-1]
    cone_columns = st.columns(4, gap="small")
    cards = [
        ("Last close", format_number(forecast_result.get("last_price"), 0), ""),
        ("Central projection", format_number(final["forecast"], 0), "accent"),
        (
            "80% range",
            f"{format_number(final['lower_80'], 0)} - {format_number(final['upper_80'], 0)}",
            "",
        ),
        (
            "95% range",
            f"{format_number(final['lower_95'], 0)} - {format_number(final['upper_95'], 0)}",
            "",
        ),
    ]
    for column, (label, value, tone) in zip(cone_columns, cards):
        with column:
            tone_class = f" is-{tone}" if tone else ""
            st.markdown(
                f'<div class="kpi-card{tone_class}"><div class="kpi-label">{label}</div>'
                f'<div class="kpi-value" style="font-size:1.25rem;">{value}</div>'
                f'<div class="kpi-delta">at horizon end</div></div>',
                unsafe_allow_html=True,
            )

    render_footnote(
        "The cone widens with the square root of time because the model is fitted on log "
        "returns. A band that contains a 10% move in either direction is telling you the "
        "honest answer: 30 sessions out, the distribution is wide."
    )
