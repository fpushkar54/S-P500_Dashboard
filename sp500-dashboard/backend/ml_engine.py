"""Predictive analytics for the index.

Two models, both fitted on demand:

* ``train_direction_model`` - a RandomForestClassifier that maps today's
  technical and volatility state to the probability that the index is higher
  N sessions from now. Evaluation is strictly chronological: the test block is
  always the most recent slice, never a shuffled sample.

* ``arima_forecast`` - an ARIMA fit on log prices that produces a 30-session
  projection with 80% and 95% bands.

Both return plain dicts with an ``error`` key rather than raising, so a failed
fit renders a message in the UI instead of a stack trace.
"""

from __future__ import annotations

import warnings
from typing import Dict, Iterable, List, Optional, Sequence

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

DEFAULT_HORIZONS: Sequence[int] = (5, 20, 60)
FORECAST_STEPS = 30
MIN_TRAINING_ROWS = 260  # roughly one year of sessions


# --------------------------------------------------------------------------- #
# Technical indicators
# --------------------------------------------------------------------------- #


def relative_strength_index(close: pd.Series, window: int = 14) -> pd.Series:
    """Wilder's RSI."""
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    avg_gain = gain.ewm(alpha=1 / window, min_periods=window, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / window, min_periods=window, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    return rsi.fillna(50.0)


def macd(
    close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9
) -> pd.DataFrame:
    """MACD line, signal line and histogram."""
    ema_fast = close.ewm(span=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, adjust=False).mean()
    line = ema_fast - ema_slow
    signal_line = line.ewm(span=signal, adjust=False).mean()
    return pd.DataFrame(
        {"macd": line, "macd_signal": signal_line, "macd_hist": line - signal_line}
    )


def bollinger_bands(close: pd.Series, window: int = 20, num_std: float = 2.0) -> pd.DataFrame:
    """Upper/lower bands, %B position and normalised band width."""
    middle = close.rolling(window, min_periods=window).mean()
    deviation = close.rolling(window, min_periods=window).std()
    upper = middle + num_std * deviation
    lower = middle - num_std * deviation
    span = (upper - lower).replace(0.0, np.nan)
    return pd.DataFrame(
        {
            "bb_upper": upper,
            "bb_middle": middle,
            "bb_lower": lower,
            "bb_percent_b": (close - lower) / span,
            "bb_width": span / middle,
        }
    )


# --------------------------------------------------------------------------- #
# Feature engineering
# --------------------------------------------------------------------------- #


FEATURE_LABELS: Dict[str, str] = {
    "return_1d": "1-day return",
    "return_5d": "5-day return",
    "return_20d": "20-day return",
    "rsi_14": "RSI (14)",
    "macd_hist": "MACD histogram",
    "bb_percent_b": "Bollinger %B",
    "bb_width": "Bollinger width",
    "ma_ratio_50": "Price / 50d MA",
    "ma_ratio_200": "Price / 200d MA",
    "golden_ratio": "50d MA / 200d MA",
    "volatility_20d": "20-day volatility",
    "drawdown": "Drawdown from high",
    "vix_level": "VIX level",
    "vix_change_1d": "VIX 1-day change",
    "vix_change_5d": "VIX 5-day change",
    "vix_ma_ratio": "VIX / 20d average",
    "breadth_50": "% above 50d MA",
    "breadth_200": "% above 200d MA",
}


def build_feature_frame(
    index_close: pd.Series,
    vix_close: Optional[pd.Series] = None,
    breadth: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """Assemble the model matrix from an index price series plus context.

    Returns a date-indexed frame whose last column is ``close`` (kept so the
    labelling step can compute forward returns without a second join).
    """
    if index_close is None or len(index_close) < 60:
        return pd.DataFrame()

    close = pd.Series(index_close).astype(float).sort_index()
    close = close[~close.index.duplicated(keep="last")].dropna()

    features = pd.DataFrame(index=close.index)
    features["return_1d"] = close.pct_change()
    features["return_5d"] = close.pct_change(5)
    features["return_20d"] = close.pct_change(20)
    features["rsi_14"] = relative_strength_index(close)
    features = features.join(macd(close)[["macd_hist"]])
    features = features.join(bollinger_bands(close)[["bb_percent_b", "bb_width"]])

    sma_50 = close.rolling(50, min_periods=50).mean()
    sma_200 = close.rolling(200, min_periods=100).mean()
    features["ma_ratio_50"] = close / sma_50
    features["ma_ratio_200"] = close / sma_200
    features["golden_ratio"] = sma_50 / sma_200
    features["volatility_20d"] = close.pct_change().rolling(20, min_periods=20).std() * np.sqrt(252)
    features["drawdown"] = close / close.cummax() - 1.0

    if vix_close is not None and len(vix_close) > 20:
        vix = pd.Series(vix_close).astype(float).sort_index()
        vix = vix[~vix.index.duplicated(keep="last")].reindex(close.index).ffill()
        features["vix_level"] = vix
        features["vix_change_1d"] = vix.pct_change()
        features["vix_change_5d"] = vix.pct_change(5)
        features["vix_ma_ratio"] = vix / vix.rolling(20, min_periods=20).mean()

    if breadth is not None and not breadth.empty:
        aligned = breadth.copy()
        aligned.index = pd.to_datetime(aligned.index)
        aligned = aligned.reindex(close.index).ffill()
        if "pct_above_50" in aligned.columns:
            features["breadth_50"] = aligned["pct_above_50"]
        if "pct_above_200" in aligned.columns:
            features["breadth_200"] = aligned["pct_above_200"]

    features["close"] = close
    features = features.replace([np.inf, -np.inf], np.nan)
    return features.dropna()


def feature_columns(features: pd.DataFrame) -> List[str]:
    return [column for column in features.columns if column != "close"]


def make_labels(close: pd.Series, horizon: int) -> pd.Series:
    """1 when the index is higher ``horizon`` sessions ahead, else 0."""
    forward_return = close.shift(-horizon) / close - 1.0
    return (forward_return > 0).astype(int).where(forward_return.notna())


# --------------------------------------------------------------------------- #
# Direction model
# --------------------------------------------------------------------------- #


def _signal_from_probability(probability: float) -> str:
    if probability >= 0.58:
        return "BULLISH"
    if probability <= 0.42:
        return "BEARISH"
    return "NEUTRAL"


def train_direction_model(
    features: pd.DataFrame,
    horizon: int = 20,
    test_fraction: float = 0.2,
    n_estimators: int = 400,
    random_state: int = 42,
) -> Dict[str, object]:
    """Fit the classifier for one horizon and score the latest observation."""
    result: Dict[str, object] = {
        "horizon": horizon,
        "probability_up": None,
        "signal": "UNAVAILABLE",
        "accuracy": None,
        "cv_accuracy": None,
        "roc_auc": None,
        "baseline": None,
        "train_rows": 0,
        "test_rows": 0,
        "importances": pd.DataFrame(columns=["feature", "label", "importance"]),
        "as_of": None,
        "error": None,
    }

    if features is None or features.empty:
        result["error"] = "No feature data available. Refresh the market data first."
        return result

    try:
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.metrics import accuracy_score, roc_auc_score
        from sklearn.model_selection import TimeSeriesSplit, cross_val_score
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler
    except Exception as exc:  # pragma: no cover - import guard
        result["error"] = f"scikit-learn is not available: {exc}"
        return result

    try:
        columns = feature_columns(features)
        labels = make_labels(features["close"], horizon)
        frame = features[columns].join(labels.rename("target")).dropna()

        if len(frame) < MIN_TRAINING_ROWS:
            result["error"] = (
                f"Only {len(frame)} usable rows for a {horizon}-day horizon; "
                f"at least {MIN_TRAINING_ROWS} are needed. Load more history."
            )
            return result

        X = frame[columns]
        y = frame["target"].astype(int)
        if y.nunique() < 2:
            result["error"] = "The target has a single class over this window."
            return result

        split = max(int(len(frame) * (1 - test_fraction)), MIN_TRAINING_ROWS // 2)
        split = min(split, len(frame) - 20)
        X_train, X_test = X.iloc[:split], X.iloc[split:]
        y_train, y_test = y.iloc[:split], y.iloc[split:]

        pipeline = Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "forest",
                    RandomForestClassifier(
                        n_estimators=n_estimators,
                        max_depth=6,
                        min_samples_leaf=20,
                        max_features="sqrt",
                        class_weight="balanced_subsample",
                        random_state=random_state,
                        n_jobs=-1,
                    ),
                ),
            ]
        )
        pipeline.fit(X_train, y_train)

        predictions = pipeline.predict(X_test)
        result["accuracy"] = float(accuracy_score(y_test, predictions))
        try:
            probabilities = pipeline.predict_proba(X_test)[:, 1]
            result["roc_auc"] = float(roc_auc_score(y_test, probabilities))
        except Exception:
            result["roc_auc"] = None

        try:
            scores = cross_val_score(
                pipeline, X, y, cv=TimeSeriesSplit(n_splits=4), scoring="accuracy"
            )
            result["cv_accuracy"] = float(np.mean(scores))
        except Exception:
            result["cv_accuracy"] = None

        # Refit on everything before scoring today's observation.
        pipeline.fit(X, y)
        latest = features[columns].iloc[[-1]]
        probability = float(pipeline.predict_proba(latest)[0][1])

        forest = pipeline.named_steps["forest"]
        importances = pd.DataFrame(
            {
                "feature": columns,
                "label": [FEATURE_LABELS.get(column, column) for column in columns],
                "importance": forest.feature_importances_,
            }
        ).sort_values("importance", ascending=False, ignore_index=True)

        result.update(
            {
                "probability_up": probability,
                "signal": _signal_from_probability(probability),
                "baseline": float(y.mean()),
                "train_rows": int(len(X_train)),
                "test_rows": int(len(X_test)),
                "importances": importances,
                "as_of": features.index[-1],
            }
        )
    except Exception as exc:
        result["error"] = f"Model fit failed: {exc}"

    return result


def train_direction_models(
    features: pd.DataFrame, horizons: Iterable[int] = DEFAULT_HORIZONS
) -> Dict[int, Dict[str, object]]:
    """Convenience wrapper: one fitted model per horizon."""
    return {int(horizon): train_direction_model(features, horizon=int(horizon)) for horizon in horizons}


# --------------------------------------------------------------------------- #
# ARIMA forecast
# --------------------------------------------------------------------------- #


def _select_arima_order(series: pd.Series, candidates: Sequence[tuple]) -> tuple:
    """Pick the candidate order with the lowest AIC."""
    from statsmodels.tsa.arima.model import ARIMA

    best_order, best_aic = (1, 1, 1), np.inf
    for order in candidates:
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                aic = ARIMA(series, order=order).fit().aic
        except Exception:
            continue
        if np.isfinite(aic) and aic < best_aic:
            best_order, best_aic = order, aic
    return best_order


def arima_forecast(
    close: pd.Series,
    steps: int = FORECAST_STEPS,
    lookback: int = 500,
    order: Optional[tuple] = None,
) -> Dict[str, object]:
    """Project the index forward with 80% and 95% bands.

    The model is fitted on log prices, so the bands are asymmetric in price
    space — wider on the upside, which is the right shape for an equity index.
    """
    result: Dict[str, object] = {
        "forecast": pd.DataFrame(),
        "order": None,
        "last_price": None,
        "last_date": None,
        "expected_price": None,
        "expected_change_pct": None,
        "error": None,
    }

    if close is None or len(close) < 120:
        result["error"] = "Not enough price history to fit a forecast (120+ sessions needed)."
        return result

    try:
        from statsmodels.tsa.arima.model import ARIMA
    except Exception as exc:  # pragma: no cover - import guard
        result["error"] = f"statsmodels is not available: {exc}"
        return result

    try:
        series = pd.Series(close).astype(float).sort_index().dropna()
        series = series[~series.index.duplicated(keep="last")].tail(int(lookback))
        log_series = np.log(series)
        log_series.index = pd.RangeIndex(len(log_series))  # avoid irregular-frequency warnings

        selected_order = order or _select_arima_order(
            log_series, [(1, 1, 1), (2, 1, 1), (1, 1, 2), (2, 1, 2), (0, 1, 1), (1, 1, 0)]
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fitted = ARIMA(log_series, order=selected_order).fit()
            prediction = fitted.get_forecast(steps=int(steps))

        mean_log = prediction.predicted_mean
        ci_95 = prediction.conf_int(alpha=0.05)
        ci_80 = prediction.conf_int(alpha=0.20)

        last_date = pd.to_datetime(series.index[-1])
        future_dates = pd.bdate_range(
            start=last_date + pd.Timedelta(days=1), periods=int(steps)
        )

        forecast = pd.DataFrame(
            {
                "date": future_dates,
                "forecast": np.exp(np.asarray(mean_log, dtype=float)),
                "lower_95": np.exp(np.asarray(ci_95.iloc[:, 0], dtype=float)),
                "upper_95": np.exp(np.asarray(ci_95.iloc[:, 1], dtype=float)),
                "lower_80": np.exp(np.asarray(ci_80.iloc[:, 0], dtype=float)),
                "upper_80": np.exp(np.asarray(ci_80.iloc[:, 1], dtype=float)),
            }
        )

        last_price = float(series.iloc[-1])
        expected = float(forecast["forecast"].iloc[-1])
        result.update(
            {
                "forecast": forecast,
                "order": selected_order,
                "last_price": last_price,
                "last_date": last_date,
                "expected_price": expected,
                "expected_change_pct": (expected / last_price - 1.0) * 100.0,
                "aic": float(fitted.aic),
            }
        )
    except Exception as exc:
        result["error"] = f"ARIMA fit failed: {exc}"

    return result


def run_full_analysis(
    index_close: pd.Series,
    vix_close: Optional[pd.Series] = None,
    breadth: Optional[pd.DataFrame] = None,
    horizons: Iterable[int] = DEFAULT_HORIZONS,
    steps: int = FORECAST_STEPS,
) -> Dict[str, object]:
    """Features + classifiers + forecast in one call, for the prediction tab."""
    features = build_feature_frame(index_close, vix_close=vix_close, breadth=breadth)
    return {
        "features": features,
        "models": train_direction_models(features, horizons=horizons),
        "forecast": arima_forecast(index_close, steps=steps),
    }
