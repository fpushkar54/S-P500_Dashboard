"""Index health math: breadth, participation, concentration and valuation.

Everything here is pure pandas/numpy. No network, no Streamlit — which keeps
the functions unit-testable and cheap enough to sit behind ``@st.cache_data``.

The canonical input is the long price frame produced by ``data_loader``:

    ticker | date | open | high | low | close | volume
"""

from __future__ import annotations

from typing import Dict, Optional, Sequence

import numpy as np
import pandas as pd

SHORT_WINDOW = 50
LONG_WINDOW = 200

# P/E ratios outside this band are almost always data errors or artefacts of a
# near-zero earnings base, and they wreck a sector median.
PE_LOWER_BOUND = 0.0
PE_UPPER_BOUND = 200.0


# --------------------------------------------------------------------------- #
# Shaping helpers
# --------------------------------------------------------------------------- #


def pivot_closes(prices: pd.DataFrame) -> pd.DataFrame:
    """Wide matrix of closing prices: rows = date, columns = ticker."""
    if prices is None or prices.empty:
        return pd.DataFrame()
    frame = prices.copy()
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
    frame = frame.dropna(subset=["date", "close"])
    wide = frame.pivot_table(index="date", columns="ticker", values="close", aggfunc="last")
    return wide.sort_index()


def moving_averages(closes: pd.DataFrame, window: int) -> pd.DataFrame:
    """Simple moving average of each column, requiring a full window."""
    if closes.empty:
        return closes
    return closes.rolling(window=window, min_periods=window).mean()


def add_moving_averages(
    prices: pd.DataFrame, windows: Sequence[int] = (SHORT_WINDOW, LONG_WINDOW)
) -> pd.DataFrame:
    """Append ``sma_<window>`` columns to the long price frame, per ticker."""
    if prices is None or prices.empty:
        return prices

    frame = prices.copy()
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
    frame = frame.sort_values(["ticker", "date"])
    grouped = frame.groupby("ticker")["close"]
    for window in windows:
        frame[f"sma_{window}"] = grouped.transform(
            lambda series, w=window: series.rolling(w, min_periods=w).mean()
        )
    return frame


# --------------------------------------------------------------------------- #
# Breadth
# --------------------------------------------------------------------------- #


def calculate_breadth(
    prices: pd.DataFrame, windows: Sequence[int] = (SHORT_WINDOW, LONG_WINDOW)
) -> pd.DataFrame:
    """Percentage of constituents trading above each moving average, by date.

    Tickers without a full moving-average window are excluded from both the
    numerator and the denominator on that date.
    """
    closes = pivot_closes(prices)
    if closes.empty:
        return pd.DataFrame()

    result = pd.DataFrame(index=closes.index)
    for window in windows:
        sma = moving_averages(closes, window)
        valid = closes.notna() & sma.notna()
        above = (closes > sma) & valid
        counts = valid.sum(axis=1)
        pct = np.where(counts > 0, above.sum(axis=1) / counts.replace(0, np.nan) * 100.0, np.nan)
        result[f"pct_above_{window}"] = pct
        result[f"count_above_{window}"] = above.sum(axis=1)
        result[f"universe_{window}"] = counts

    result.index.name = "date"
    return result.dropna(how="all")


def breadth_snapshot(breadth: pd.DataFrame) -> Dict[str, Optional[float]]:
    """Latest breadth values plus the one-month change, for the KPI bar."""
    empty = {
        "pct_above_50": None,
        "pct_above_200": None,
        "delta_50": None,
        "delta_200": None,
        "as_of": None,
    }
    if breadth is None or breadth.empty:
        return empty

    latest = breadth.dropna(subset=["pct_above_200"], how="all").tail(1)
    if latest.empty:
        return empty

    row = latest.iloc[0]
    lookback = breadth.iloc[-22] if len(breadth) > 22 else breadth.iloc[0]

    def delta(column: str) -> Optional[float]:
        if column not in breadth.columns:
            return None
        current, previous = row.get(column), lookback.get(column)
        if pd.isna(current) or pd.isna(previous):
            return None
        return float(current - previous)

    return {
        "pct_above_50": float(row.get("pct_above_50")) if pd.notna(row.get("pct_above_50")) else None,
        "pct_above_200": float(row.get("pct_above_200"))
        if pd.notna(row.get("pct_above_200"))
        else None,
        "delta_50": delta("pct_above_50"),
        "delta_200": delta("pct_above_200"),
        "as_of": latest.index[0],
    }


# --------------------------------------------------------------------------- #
# Advance / decline
# --------------------------------------------------------------------------- #


def advance_decline_line(prices: pd.DataFrame) -> pd.DataFrame:
    """Daily advancers, decliners, net breadth and the cumulative A/D line."""
    closes = pivot_closes(prices)
    if closes.empty:
        return pd.DataFrame()

    changes = closes.diff()
    advances = (changes > 0).sum(axis=1)
    declines = (changes < 0).sum(axis=1)
    unchanged = ((changes == 0) & changes.notna()).sum(axis=1)

    frame = pd.DataFrame(
        {
            "advances": advances,
            "declines": declines,
            "unchanged": unchanged,
        }
    )
    frame["net"] = frame["advances"] - frame["declines"]
    frame = frame.iloc[1:]  # first diff row is all NaN
    frame["ad_line"] = frame["net"].cumsum()

    participating = (frame["advances"] + frame["declines"]).replace(0, np.nan)
    frame["advance_ratio"] = frame["advances"] / participating * 100.0
    frame.index.name = "date"
    return frame


# --------------------------------------------------------------------------- #
# Concentration
# --------------------------------------------------------------------------- #


def calculate_concentration(
    fundamentals: pd.DataFrame, tiers: Sequence[int] = (5, 10, 20)
) -> Dict[str, object]:
    """Share of total index market cap held by the largest N constituents."""
    result: Dict[str, object] = {
        "weights": {},
        "total_market_cap": 0.0,
        "constituents": 0,
        "top_holdings": pd.DataFrame(),
    }
    if fundamentals is None or fundamentals.empty or "market_cap" not in fundamentals.columns:
        return result

    frame = fundamentals.copy()
    frame["market_cap"] = pd.to_numeric(frame["market_cap"], errors="coerce")
    frame = frame.dropna(subset=["market_cap"])
    frame = frame[frame["market_cap"] > 0].sort_values("market_cap", ascending=False)
    if frame.empty:
        return result

    total = float(frame["market_cap"].sum())
    frame["weight_pct"] = frame["market_cap"] / total * 100.0
    frame["cumulative_weight_pct"] = frame["weight_pct"].cumsum()

    result["weights"] = {
        f"top_{tier}": float(frame["weight_pct"].head(tier).sum()) for tier in tiers
    }
    result["total_market_cap"] = total
    result["constituents"] = int(len(frame))
    result["top_holdings"] = frame[
        [column for column in ("ticker", "name", "sector", "market_cap", "weight_pct", "cumulative_weight_pct") if column in frame.columns]
    ].head(max(tiers)).reset_index(drop=True)
    result["ranked"] = frame.reset_index(drop=True)
    return result


def concentration_donut_frame(concentration: Dict[str, object], top_n: int = 10) -> pd.DataFrame:
    """Top N holdings plus a single 'rest of index' slice, ready to plot."""
    ranked = concentration.get("ranked")
    if ranked is None or len(ranked) == 0:
        return pd.DataFrame(columns=["label", "weight_pct"])

    head = ranked.head(top_n)
    rest_weight = float(ranked["weight_pct"].iloc[top_n:].sum()) if len(ranked) > top_n else 0.0

    labels = head["ticker"].tolist()
    weights = head["weight_pct"].tolist()
    if rest_weight > 0:
        labels.append(f"Other {len(ranked) - top_n}")
        weights.append(rest_weight)

    return pd.DataFrame({"label": labels, "weight_pct": weights})


def sector_weights(fundamentals: pd.DataFrame) -> pd.DataFrame:
    """Market-cap weight of each GICS sector."""
    if fundamentals is None or fundamentals.empty:
        return pd.DataFrame(columns=["sector", "market_cap", "weight_pct"])

    frame = fundamentals.copy()
    frame["market_cap"] = pd.to_numeric(frame["market_cap"], errors="coerce")
    frame = frame.dropna(subset=["market_cap", "sector"])
    if frame.empty:
        return pd.DataFrame(columns=["sector", "market_cap", "weight_pct"])

    grouped = frame.groupby("sector", as_index=False)["market_cap"].sum()
    total = grouped["market_cap"].sum()
    grouped["weight_pct"] = grouped["market_cap"] / total * 100.0
    return grouped.sort_values("weight_pct", ascending=False).reset_index(drop=True)


# --------------------------------------------------------------------------- #
# Valuation
# --------------------------------------------------------------------------- #


def _clean_pe(series: pd.Series) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce")
    return values.where((values > PE_LOWER_BOUND) & (values < PE_UPPER_BOUND))


def weighted_median(values: pd.Series, weights: pd.Series) -> Optional[float]:
    """Median of ``values`` where each observation carries ``weights`` mass."""
    frame = pd.DataFrame({"value": values, "weight": weights}).dropna()
    frame = frame[frame["weight"] > 0]
    if frame.empty:
        return None

    frame = frame.sort_values("value")
    cumulative = frame["weight"].cumsum()
    cutoff = frame["weight"].sum() / 2.0
    selected = frame.loc[cumulative >= cutoff, "value"]
    return float(selected.iloc[0]) if len(selected) else float(frame["value"].iloc[-1])


def sector_valuation(fundamentals: pd.DataFrame) -> pd.DataFrame:
    """Per-sector trailing and forward P/E, plain median and cap-weighted median."""
    columns = [
        "sector",
        "companies",
        "market_cap",
        "weight_pct",
        "trailing_pe_median",
        "forward_pe_median",
        "trailing_pe_weighted",
        "forward_pe_weighted",
    ]
    if fundamentals is None or fundamentals.empty:
        return pd.DataFrame(columns=columns)

    frame = fundamentals.copy()
    frame["market_cap"] = pd.to_numeric(frame["market_cap"], errors="coerce")
    frame["trailing_pe"] = _clean_pe(frame.get("trailing_pe", pd.Series(dtype=float)))
    frame["forward_pe"] = _clean_pe(frame.get("forward_pe", pd.Series(dtype=float)))
    frame = frame.dropna(subset=["sector"])
    if frame.empty:
        return pd.DataFrame(columns=columns)

    rows = []
    for sector, group in frame.groupby("sector"):
        rows.append(
            {
                "sector": sector,
                "companies": int(len(group)),
                "market_cap": float(group["market_cap"].sum(skipna=True)),
                "trailing_pe_median": float(group["trailing_pe"].median())
                if group["trailing_pe"].notna().any()
                else np.nan,
                "forward_pe_median": float(group["forward_pe"].median())
                if group["forward_pe"].notna().any()
                else np.nan,
                "trailing_pe_weighted": weighted_median(group["trailing_pe"], group["market_cap"]),
                "forward_pe_weighted": weighted_median(group["forward_pe"], group["market_cap"]),
            }
        )

    result = pd.DataFrame(rows)
    total = result["market_cap"].sum()
    result["weight_pct"] = result["market_cap"] / total * 100.0 if total else np.nan
    return result[columns].sort_values("forward_pe_median", ascending=False).reset_index(drop=True)


def index_valuation_summary(fundamentals: pd.DataFrame) -> Dict[str, Optional[float]]:
    """Index-level P/E medians across every constituent."""
    if fundamentals is None or fundamentals.empty:
        return {"trailing_pe": None, "forward_pe": None, "coverage": 0}

    trailing = _clean_pe(fundamentals.get("trailing_pe", pd.Series(dtype=float)))
    forward = _clean_pe(fundamentals.get("forward_pe", pd.Series(dtype=float)))
    return {
        "trailing_pe": float(trailing.median()) if trailing.notna().any() else None,
        "forward_pe": float(forward.median()) if forward.notna().any() else None,
        "coverage": int(forward.notna().sum()),
    }


# --------------------------------------------------------------------------- #
# Screener
# --------------------------------------------------------------------------- #


def build_screener_table(prices: pd.DataFrame, fundamentals: pd.DataFrame) -> pd.DataFrame:
    """One row per constituent with price, valuation and trend position."""
    columns = [
        "ticker",
        "name",
        "sector",
        "market_cap",
        "price",
        "trailing_pe",
        "forward_pe",
        "dist_from_200d_pct",
        "dist_from_50d_pct",
        "return_1m_pct",
        "return_6m_pct",
        "above_200d",
    ]
    if fundamentals is None or fundamentals.empty:
        return pd.DataFrame(columns=columns)

    table = fundamentals.copy()
    table["market_cap"] = pd.to_numeric(table["market_cap"], errors="coerce")
    table["trailing_pe"] = _clean_pe(table.get("trailing_pe", pd.Series(dtype=float)))
    table["forward_pe"] = _clean_pe(table.get("forward_pe", pd.Series(dtype=float)))

    closes = pivot_closes(prices)
    if not closes.empty:
        latest = closes.ffill().iloc[-1]
        sma50 = moving_averages(closes, SHORT_WINDOW).iloc[-1]
        sma200 = moving_averages(closes, LONG_WINDOW).iloc[-1]

        trend = pd.DataFrame(
            {
                "ticker": latest.index,
                "last_close": latest.values,
                "sma_50": sma50.reindex(latest.index).values,
                "sma_200": sma200.reindex(latest.index).values,
            }
        )
        for label, periods in (("return_1m_pct", 21), ("return_6m_pct", 126)):
            if len(closes) > periods:
                prior = closes.ffill().iloc[-1 - periods]
                trend[label] = ((latest / prior - 1.0) * 100.0).reindex(latest.index).values
            else:
                trend[label] = np.nan

        table = table.merge(trend, on="ticker", how="left")
        table["price"] = pd.to_numeric(table.get("price"), errors="coerce").fillna(
            table["last_close"]
        )
        table["dist_from_50d_pct"] = (table["price"] / table["sma_50"] - 1.0) * 100.0
        table["dist_from_200d_pct"] = (table["price"] / table["sma_200"] - 1.0) * 100.0
        table["above_200d"] = table["price"] > table["sma_200"]
    else:
        for column in (
            "dist_from_50d_pct",
            "dist_from_200d_pct",
            "return_1m_pct",
            "return_6m_pct",
        ):
            table[column] = np.nan
        table["above_200d"] = False

    for column in columns:
        if column not in table.columns:
            table[column] = np.nan

    return table[columns].sort_values("market_cap", ascending=False).reset_index(drop=True)


def sector_performance(prices: pd.DataFrame, fundamentals: pd.DataFrame, periods: int = 21) -> pd.DataFrame:
    """Cap-weighted sector return over the trailing ``periods`` sessions."""
    closes = pivot_closes(prices)
    if closes.empty or fundamentals is None or fundamentals.empty:
        return pd.DataFrame(columns=["sector", "return_pct", "market_cap"])
    if len(closes) <= periods:
        return pd.DataFrame(columns=["sector", "return_pct", "market_cap"])

    filled = closes.ffill()
    returns = (filled.iloc[-1] / filled.iloc[-1 - periods] - 1.0) * 100.0
    frame = pd.DataFrame({"ticker": returns.index, "return_pct": returns.values})

    meta = fundamentals[["ticker", "sector", "market_cap"]].copy()
    meta["market_cap"] = pd.to_numeric(meta["market_cap"], errors="coerce")
    frame = frame.merge(meta, on="ticker", how="inner").dropna(subset=["sector", "return_pct"])
    if frame.empty:
        return pd.DataFrame(columns=["sector", "return_pct", "market_cap"])

    frame["market_cap"] = frame["market_cap"].fillna(0.0)

    def weighted(group: pd.DataFrame) -> float:
        weights = group["market_cap"]
        if weights.sum() <= 0:
            return float(group["return_pct"].mean())
        return float(np.average(group["return_pct"], weights=weights))

    rows = [
        {"sector": sector, "return_pct": weighted(group), "market_cap": float(group["market_cap"].sum())}
        for sector, group in frame.groupby("sector")
    ]
    return pd.DataFrame(rows).sort_values("return_pct", ascending=False).reset_index(drop=True)
