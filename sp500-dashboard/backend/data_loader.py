"""Data ingestion for the S&P 500 health dashboard.

Three sources feed the cache:

1. Wikipedia  - the current index membership (ticker, company, GICS sector).
2. yfinance   - two-plus years of daily OHLCV plus per-ticker fundamentals,
                pulled in parallel with a ThreadPoolExecutor.
3. FRED/yahoo - macro context: ^GSPC, ^VIX and the 10-year Treasury yield.

Every network call is defensive. A handful of dead tickers, a throttled API or
an offline FRED endpoint degrades the dataset rather than crashing the app.
"""

from __future__ import annotations

import io
import os
import time
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta, timezone
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import pandas as pd

from backend import database

warnings.filterwarnings("ignore", category=FutureWarning)

WIKI_SP500_URL = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
)

DEFAULT_PERIOD = "2y"
DEFAULT_MAX_WORKERS = 12

# Price source. Google Finance is deliberately absent: Google shut its Finance
# API down in October 2012 and the June 2026 consumer relaunch did not restore
# it, so the only ways to get Google's numbers are the GOOGLEFINANCE spreadsheet
# formula or scraping a page whose markup rotates. Neither belongs in an
# ingestion pipeline. "yahoo" is the default; "stooq" is keyless and needs no
# account, and is also used automatically when Yahoo returns nothing for a
# ticker. To add a keyed provider (Alpha Vantage, Finnhub, Tiingo), write a
# fetch_price_history_<name> with the same signature and add a branch in
# fetch_price_history.
PRIMARY_PRICE_SOURCE = os.environ.get("SP500_PRICE_SOURCE", "yahoo").lower()

MACRO_SYMBOLS = {"SP500": "^GSPC", "VIX": "^VIX"}
TREASURY_FRED_ID = "DGS10"
TREASURY_FALLBACK_SYMBOL = "^TNX"  # 10Y yield x10, used if FRED is unreachable

# Used only when Wikipedia is unreachable, so the app still starts with a
# representative slice of the index instead of an empty screen.
FALLBACK_CONSTITUENTS: List[Tuple[str, str, str]] = [
    ("AAPL", "Apple Inc.", "Information Technology"),
    ("MSFT", "Microsoft Corporation", "Information Technology"),
    ("NVDA", "NVIDIA Corporation", "Information Technology"),
    ("AVGO", "Broadcom Inc.", "Information Technology"),
    ("ORCL", "Oracle Corporation", "Information Technology"),
    ("CRM", "Salesforce Inc.", "Information Technology"),
    ("AMD", "Advanced Micro Devices", "Information Technology"),
    ("ADBE", "Adobe Inc.", "Information Technology"),
    ("CSCO", "Cisco Systems", "Information Technology"),
    ("ACN", "Accenture plc", "Information Technology"),
    ("AMZN", "Amazon.com Inc.", "Consumer Discretionary"),
    ("TSLA", "Tesla Inc.", "Consumer Discretionary"),
    ("HD", "Home Depot", "Consumer Discretionary"),
    ("MCD", "McDonald's Corporation", "Consumer Discretionary"),
    ("NKE", "Nike Inc.", "Consumer Discretionary"),
    ("GOOGL", "Alphabet Inc. Class A", "Communication Services"),
    ("META", "Meta Platforms Inc.", "Communication Services"),
    ("NFLX", "Netflix Inc.", "Communication Services"),
    ("DIS", "Walt Disney Company", "Communication Services"),
    ("CMCSA", "Comcast Corporation", "Communication Services"),
    ("BRK-B", "Berkshire Hathaway Class B", "Financials"),
    ("JPM", "JPMorgan Chase & Co.", "Financials"),
    ("V", "Visa Inc.", "Financials"),
    ("MA", "Mastercard Inc.", "Financials"),
    ("BAC", "Bank of America", "Financials"),
    ("GS", "Goldman Sachs Group", "Financials"),
    ("LLY", "Eli Lilly and Company", "Health Care"),
    ("UNH", "UnitedHealth Group", "Health Care"),
    ("JNJ", "Johnson & Johnson", "Health Care"),
    ("ABBV", "AbbVie Inc.", "Health Care"),
    ("MRK", "Merck & Co.", "Health Care"),
    ("PFE", "Pfizer Inc.", "Health Care"),
    ("TMO", "Thermo Fisher Scientific", "Health Care"),
    ("PG", "Procter & Gamble", "Consumer Staples"),
    ("COST", "Costco Wholesale", "Consumer Staples"),
    ("WMT", "Walmart Inc.", "Consumer Staples"),
    ("KO", "Coca-Cola Company", "Consumer Staples"),
    ("PEP", "PepsiCo Inc.", "Consumer Staples"),
    ("XOM", "Exxon Mobil Corporation", "Energy"),
    ("CVX", "Chevron Corporation", "Energy"),
    ("COP", "ConocoPhillips", "Energy"),
    ("CAT", "Caterpillar Inc.", "Industrials"),
    ("GE", "GE Aerospace", "Industrials"),
    ("HON", "Honeywell International", "Industrials"),
    ("UNP", "Union Pacific Corporation", "Industrials"),
    ("RTX", "RTX Corporation", "Industrials"),
    ("LIN", "Linde plc", "Materials"),
    ("SHW", "Sherwin-Williams", "Materials"),
    ("NEE", "NextEra Energy", "Utilities"),
    ("DUK", "Duke Energy", "Utilities"),
    ("SO", "Southern Company", "Utilities"),
    ("AMT", "American Tower", "Real Estate"),
    ("PLD", "Prologis Inc.", "Real Estate"),
    ("SPG", "Simon Property Group", "Real Estate"),
]


# --------------------------------------------------------------------------- #
# Constituents
# --------------------------------------------------------------------------- #


def normalize_ticker(ticker: str) -> str:
    """Yahoo uses dashes where Wikipedia uses dots (BRK.B -> BRK-B)."""
    return str(ticker).strip().upper().replace(".", "-")


def _fallback_constituents() -> pd.DataFrame:
    frame = pd.DataFrame(FALLBACK_CONSTITUENTS, columns=["ticker", "name", "sector"])
    frame["sub_industry"] = None
    frame["source"] = "fallback"
    return frame


def get_sp500_constituents(timeout: int = 20) -> pd.DataFrame:
    """Scrape the index membership table from Wikipedia.

    Returns columns: ticker, name, sector, sub_industry, source.
    Falls back to a built-in list if the page cannot be read.
    """
    try:
        import requests

        response = requests.get(
            WIKI_SP500_URL, headers={"User-Agent": USER_AGENT}, timeout=timeout
        )
        response.raise_for_status()
        tables = pd.read_html(io.StringIO(response.text))
    except Exception:
        try:
            tables = pd.read_html(WIKI_SP500_URL)
        except Exception:
            return _fallback_constituents()

    table = None
    for candidate in tables:
        columns = {str(col).strip().lower() for col in candidate.columns}
        if "symbol" in columns and ("security" in columns or "company" in columns):
            table = candidate
            break

    if table is None or table.empty:
        return _fallback_constituents()

    table = table.rename(
        columns={
            "Symbol": "ticker",
            "Security": "name",
            "GICS Sector": "sector",
            "GICS Sub-Industry": "sub_industry",
        }
    )

    for column in ("ticker", "name", "sector", "sub_industry"):
        if column not in table.columns:
            table[column] = None

    frame = table[["ticker", "name", "sector", "sub_industry"]].copy()
    frame["ticker"] = frame["ticker"].map(normalize_ticker)
    frame = frame[frame["ticker"].str.len() > 0].drop_duplicates(subset="ticker")
    frame["source"] = "wikipedia"
    return frame.reset_index(drop=True)


# --------------------------------------------------------------------------- #
# Per-ticker fetching
# --------------------------------------------------------------------------- #


def fetch_price_history_stooq(ticker: str, period: str = DEFAULT_PERIOD) -> pd.DataFrame:
    """Daily OHLCV from Stooq, which needs no API key.

    Used as an automatic fallback when the primary source returns nothing for a
    ticker, which on a 500-name download usually means throttling rather than a
    genuinely dead symbol.
    """
    try:
        from pandas_datareader import data as pdr

        years = int(str(period).rstrip("y") or 2) if str(period).endswith("y") else 2
        start = datetime.now(timezone.utc) - timedelta(days=365 * years)
        history = pdr.DataReader(f"{ticker}.US", "stooq", start)
        if history is None or history.empty:
            return pd.DataFrame(columns=database.PRICE_COLUMNS)

        history = history.sort_index().reset_index()
        return pd.DataFrame(
            {
                "ticker": ticker,
                "date": history["Date"],
                "open": history.get("Open"),
                "high": history.get("High"),
                "low": history.get("Low"),
                "close": history.get("Close"),
                "volume": history.get("Volume"),
            }
        ).dropna(subset=["close"])
    except Exception:
        return pd.DataFrame(columns=database.PRICE_COLUMNS)


def fetch_price_history(
    ticker: str,
    period: str = DEFAULT_PERIOD,
    retries: int = 2,
    source: str = PRIMARY_PRICE_SOURCE,
    allow_fallback: bool = True,
) -> pd.DataFrame:
    """Daily OHLCV for one ticker as a long frame. Empty frame on failure."""
    if source == "stooq":
        return fetch_price_history_stooq(ticker, period=period)

    frame = _fetch_price_history_yahoo(ticker, period=period, retries=retries)
    if frame.empty and allow_fallback:
        return fetch_price_history_stooq(ticker, period=period)
    return frame


def _fetch_price_history_yahoo(
    ticker: str, period: str = DEFAULT_PERIOD, retries: int = 2
) -> pd.DataFrame:
    import yfinance as yf

    for attempt in range(retries + 1):
        try:
            history = yf.Ticker(ticker).history(
                period=period, interval="1d", auto_adjust=True, actions=False
            )
            if history is None or history.empty:
                return pd.DataFrame(columns=database.PRICE_COLUMNS)

            history = history.reset_index()
            date_column = "Date" if "Date" in history.columns else history.columns[0]
            frame = pd.DataFrame(
                {
                    "ticker": ticker,
                    "date": history[date_column],
                    "open": history.get("Open"),
                    "high": history.get("High"),
                    "low": history.get("Low"),
                    "close": history.get("Close"),
                    "volume": history.get("Volume"),
                }
            )
            return frame.dropna(subset=["close"])
        except Exception:
            if attempt >= retries:
                return pd.DataFrame(columns=database.PRICE_COLUMNS)
            time.sleep(0.6 * (attempt + 1))

    return pd.DataFrame(columns=database.PRICE_COLUMNS)


def fetch_fundamentals(ticker: str) -> Dict[str, Optional[float]]:
    """Market cap, P/E ratios and a few extras for one ticker."""
    import yfinance as yf

    record: Dict[str, Optional[float]] = {"ticker": ticker}
    try:
        handle = yf.Ticker(ticker)
    except Exception:
        return record

    info: Dict = {}
    try:
        info = handle.get_info() or {}
    except Exception:
        try:
            info = handle.info or {}
        except Exception:
            info = {}

    def pick(*keys):
        for key in keys:
            value = info.get(key)
            if value is not None and not isinstance(value, str):
                try:
                    numeric = float(value)
                except (TypeError, ValueError):
                    continue
                if pd.notna(numeric):
                    return numeric
        return None

    record.update(
        {
            "name": info.get("longName") or info.get("shortName"),
            "sector": info.get("sector"),
            "market_cap": pick("marketCap"),
            "price": pick("currentPrice", "regularMarketPrice", "previousClose"),
            "trailing_pe": pick("trailingPE"),
            "forward_pe": pick("forwardPE"),
            "price_to_book": pick("priceToBook"),
            "dividend_yield": pick("dividendYield"),
            "beta": pick("beta"),
            "fifty_two_week_high": pick("fiftyTwoWeekHigh"),
            "fifty_two_week_low": pick("fiftyTwoWeekLow"),
        }
    )

    # fast_info is cheap and fills gaps left by a throttled .info call.
    if record.get("price") is None or record.get("market_cap") is None:
        try:
            fast = handle.fast_info
            record["price"] = record.get("price") or float(fast.get("last_price"))
            record["market_cap"] = record.get("market_cap") or float(fast.get("market_cap"))
        except Exception:
            pass

    return record


def _fetch_one(ticker: str, period: str, with_fundamentals: bool):
    """Fetch both halves for one ticker. The halves fail independently: a
    throttled fundamentals call must not throw away good price history."""
    try:
        prices = fetch_price_history(ticker, period=period)
    except Exception:
        prices = pd.DataFrame(columns=database.PRICE_COLUMNS)

    fundamentals: Dict = {"ticker": ticker}
    if with_fundamentals:
        try:
            fundamentals = fetch_fundamentals(ticker) or {"ticker": ticker}
        except Exception:
            fundamentals = {"ticker": ticker}

    return ticker, prices, fundamentals


def download_universe(
    tickers: Sequence[str],
    period: str = DEFAULT_PERIOD,
    max_workers: int = DEFAULT_MAX_WORKERS,
    with_fundamentals: bool = True,
    progress_callback: Optional[Callable[[int, int, str], None]] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, List[str]]:
    """Download prices and fundamentals for many tickers in parallel.

    Returns ``(prices, fundamentals, failed_tickers)``.
    """
    tickers = [normalize_ticker(t) for t in tickers if str(t).strip()]
    price_frames: List[pd.DataFrame] = []
    fundamental_rows: List[Dict] = []
    failed: List[str] = []
    total = len(tickers)
    completed = 0

    if total == 0:
        return (
            pd.DataFrame(columns=database.PRICE_COLUMNS),
            pd.DataFrame(columns=database.FUNDAMENTAL_COLUMNS),
            failed,
        )

    with ThreadPoolExecutor(max_workers=max(1, int(max_workers))) as pool:
        futures = {
            pool.submit(_fetch_one, ticker, period, with_fundamentals): ticker
            for ticker in tickers
        }
        for future in as_completed(futures):
            ticker = futures[future]
            try:
                _, prices, fundamentals = future.result()
                if prices is not None and not prices.empty:
                    price_frames.append(prices)
                else:
                    failed.append(ticker)
                if fundamentals:
                    fundamental_rows.append(fundamentals)
            except Exception:
                failed.append(ticker)
            finally:
                completed += 1
                if progress_callback is not None:
                    try:
                        progress_callback(completed, total, ticker)
                    except Exception:
                        pass

    prices = (
        pd.concat(price_frames, ignore_index=True)
        if price_frames
        else pd.DataFrame(columns=database.PRICE_COLUMNS)
    )
    fundamentals = (
        pd.DataFrame(fundamental_rows)
        if fundamental_rows
        else pd.DataFrame(columns=database.FUNDAMENTAL_COLUMNS)
    )
    return prices, fundamentals, failed


# --------------------------------------------------------------------------- #
# Macro indicators
# --------------------------------------------------------------------------- #


def fetch_treasury_yield(start: datetime) -> pd.DataFrame:
    """10-year Treasury constant maturity yield (percent), FRED first."""
    try:
        from pandas_datareader import data as pdr

        series = pdr.DataReader(TREASURY_FRED_ID, "fred", start)
        frame = series.reset_index()
        frame.columns = ["date", "value"]
        frame = frame.dropna(subset=["value"])
        if not frame.empty:
            frame["series_id"] = TREASURY_FRED_ID
            return frame[database.MACRO_COLUMNS]
    except Exception:
        pass

    # Yahoo's ^TNX quotes the yield multiplied by ten.
    try:
        import yfinance as yf

        history = yf.Ticker(TREASURY_FALLBACK_SYMBOL).history(period="5y", interval="1d")
        if history is None or history.empty:
            return pd.DataFrame(columns=database.MACRO_COLUMNS)
        history = history.reset_index()
        frame = pd.DataFrame(
            {
                "series_id": TREASURY_FRED_ID,
                "date": history[history.columns[0]],
                "value": history["Close"].astype(float) / 10.0,
            }
        )
        return frame.dropna(subset=["value"])
    except Exception:
        return pd.DataFrame(columns=database.MACRO_COLUMNS)


def fetch_macro_indicators(years: int = 5) -> pd.DataFrame:
    """Long-format frame of SP500, VIX and DGS10."""
    start = datetime.now(timezone.utc) - timedelta(days=365 * years)
    frames: List[pd.DataFrame] = []

    try:
        import yfinance as yf
    except Exception:
        yf = None

    if yf is not None:
        for series_id, symbol in MACRO_SYMBOLS.items():
            try:
                history = yf.Ticker(symbol).history(
                    period=f"{years}y", interval="1d", auto_adjust=False
                )
                if history is None or history.empty:
                    continue
                history = history.reset_index()
                frames.append(
                    pd.DataFrame(
                        {
                            "series_id": series_id,
                            "date": history[history.columns[0]],
                            "value": history["Close"].astype(float),
                        }
                    ).dropna(subset=["value"])
                )
            except Exception:
                continue

    treasury = fetch_treasury_yield(start)
    if not treasury.empty:
        frames.append(treasury)

    if not frames:
        return pd.DataFrame(columns=database.MACRO_COLUMNS)
    return pd.concat(frames, ignore_index=True)[database.MACRO_COLUMNS]


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #


def refresh_market_data(
    period: str = DEFAULT_PERIOD,
    max_workers: int = DEFAULT_MAX_WORKERS,
    limit: Optional[int] = None,
    progress_callback: Optional[Callable[[int, int, str], None]] = None,
) -> Dict:
    """Full pipeline: scrape members, download everything, write to SQLite.

    ``limit`` caps the universe (handy for a fast first run or a demo).
    Returns a summary dict for the UI.
    """
    database.initialize_database()

    constituents = get_sp500_constituents()
    if limit:
        constituents = constituents.head(int(limit))

    tickers = constituents["ticker"].tolist()
    prices, fundamentals, failed = download_universe(
        tickers,
        period=period,
        max_workers=max_workers,
        progress_callback=progress_callback,
    )

    # Wikipedia sector/name is authoritative; yfinance fills the numbers.
    if fundamentals.empty:
        fundamentals = constituents[["ticker", "name", "sector", "sub_industry"]].copy()
    else:
        fundamentals = fundamentals.drop(columns=["name", "sector"], errors="ignore")
        fundamentals = constituents[["ticker", "name", "sector", "sub_industry"]].merge(
            fundamentals, on="ticker", how="left"
        )

    # Backfill missing market prices from the freshly downloaded history.
    if not prices.empty:
        last_close = (
            prices.sort_values("date")
            .groupby("ticker")["close"]
            .last()
            .rename("last_close")
            .reset_index()
        )
        fundamentals = fundamentals.merge(last_close, on="ticker", how="left")
        if "price" not in fundamentals.columns:
            fundamentals["price"] = None
        fundamentals["price"] = fundamentals["price"].fillna(fundamentals["last_close"])
        fundamentals = fundamentals.drop(columns=["last_close"])

    macro = fetch_macro_indicators()

    price_rows = database.upsert_daily_prices(prices)
    fundamental_rows = database.upsert_sector_fundamentals(fundamentals)
    macro_rows = database.upsert_macro_indicators(macro)
    stamp = database.mark_refreshed()

    return {
        "tickers_requested": len(tickers),
        "tickers_loaded": int(prices["ticker"].nunique()) if not prices.empty else 0,
        "failed_tickers": failed,
        "price_rows": price_rows,
        "fundamental_rows": fundamental_rows,
        "macro_rows": macro_rows,
        "refreshed_at": stamp,
        "constituent_source": constituents["source"].iloc[0] if not constituents.empty else "none",
    }


def load_market_data(start: Optional[str] = None) -> Dict[str, object]:
    """Read the cached dataset out of SQLite into memory."""
    database.initialize_database()
    prices = database.read_daily_prices(start=start)
    fundamentals = database.read_sector_fundamentals()

    macro = {
        "SP500": database.macro_series("SP500"),
        "VIX": database.macro_series("VIX"),
        TREASURY_FRED_ID: database.macro_series(TREASURY_FRED_ID),
    }

    return {
        "prices": prices,
        "fundamentals": fundamentals,
        "macro": macro,
        "last_refresh": database.get_last_refresh(),
    }
