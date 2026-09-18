"""SQLite caching layer for the S&P 500 health dashboard.

Everything the app fetches from the network lands here first, so a restart of
Streamlit never means a re-download of 500 tickers. Three tables carry the load:

    daily_prices        OHLCV history, one row per ticker per session
    sector_fundamentals one row per ticker: name, sector, market cap, P/E ratios
    macro_indicators    long-format series (SP500, VIX, DGS10)

A fourth table, ``meta``, stores small key/value bookkeeping such as the
timestamp of the last successful refresh.
"""

from __future__ import annotations

import os
import sqlite3
import threading
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Iterable, Optional, Sequence

import numpy as np
import pandas as pd

# sqlite3 cannot bind numpy scalars out of the box; teach it how.
for _np_int in (np.int8, np.int16, np.int32, np.int64):
    sqlite3.register_adapter(_np_int, int)
for _np_float in (np.float16, np.float32, np.float64):
    sqlite3.register_adapter(_np_float, float)
sqlite3.register_adapter(np.bool_, int)

# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB_FILENAME = "sp500_health.db"
DB_PATH = os.environ.get("SP500_DB_PATH", os.path.join(PROJECT_ROOT, DB_FILENAME))

# SQLite is happy with concurrent readers but a single writer. Every write goes
# through this lock so the threaded downloader cannot interleave transactions.
_WRITE_LOCK = threading.Lock()

_SCHEMA_STATEMENTS: Sequence[str] = (
    """
    CREATE TABLE IF NOT EXISTS daily_prices (
        ticker  TEXT    NOT NULL,
        date    TEXT    NOT NULL,
        open    REAL,
        high    REAL,
        low     REAL,
        close   REAL,
        volume  REAL,
        PRIMARY KEY (ticker, date)
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_daily_prices_date ON daily_prices (date)",
    "CREATE INDEX IF NOT EXISTS idx_daily_prices_ticker ON daily_prices (ticker)",
    """
    CREATE TABLE IF NOT EXISTS sector_fundamentals (
        ticker        TEXT PRIMARY KEY,
        name          TEXT,
        sector        TEXT,
        sub_industry  TEXT,
        market_cap    REAL,
        price         REAL,
        trailing_pe   REAL,
        forward_pe    REAL,
        price_to_book REAL,
        dividend_yield REAL,
        beta          REAL,
        fifty_two_week_high REAL,
        fifty_two_week_low  REAL,
        updated_at    TEXT
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_fundamentals_sector ON sector_fundamentals (sector)",
    """
    CREATE TABLE IF NOT EXISTS macro_indicators (
        series_id TEXT NOT NULL,
        date      TEXT NOT NULL,
        value     REAL,
        PRIMARY KEY (series_id, date)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS meta (
        key   TEXT PRIMARY KEY,
        value TEXT
    )
    """,
)

PRICE_COLUMNS = ["ticker", "date", "open", "high", "low", "close", "volume"]

FUNDAMENTAL_COLUMNS = [
    "ticker",
    "name",
    "sector",
    "sub_industry",
    "market_cap",
    "price",
    "trailing_pe",
    "forward_pe",
    "price_to_book",
    "dividend_yield",
    "beta",
    "fifty_two_week_high",
    "fifty_two_week_low",
    "updated_at",
]

MACRO_COLUMNS = ["series_id", "date", "value"]


# --------------------------------------------------------------------------- #
# Connection handling
# --------------------------------------------------------------------------- #


@contextmanager
def get_connection(db_path: str = DB_PATH):
    """Yield a SQLite connection with sane pragmas, always closed afterwards."""
    directory = os.path.dirname(os.path.abspath(db_path))
    if directory and not os.path.isdir(directory):
        os.makedirs(directory, exist_ok=True)

    conn = sqlite3.connect(db_path, timeout=30, check_same_thread=False)
    try:
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def initialize_database(db_path: str = DB_PATH) -> None:
    """Create the schema if it does not already exist. Safe to call repeatedly."""
    with get_connection(db_path) as conn:
        for statement in _SCHEMA_STATEMENTS:
            conn.execute(statement)


# --------------------------------------------------------------------------- #
# Internal helpers
# --------------------------------------------------------------------------- #


def _normalize_dates(frame: pd.DataFrame, column: str = "date") -> pd.DataFrame:
    """Coerce a date column to naive ISO ``YYYY-MM-DD`` strings."""
    out = frame.copy()
    parsed = pd.to_datetime(out[column], errors="coerce", utc=True)
    out[column] = parsed.dt.tz_convert(None).dt.strftime("%Y-%m-%d")
    return out.dropna(subset=[column])


def _prepare(frame: pd.DataFrame, columns: Iterable[str]) -> pd.DataFrame:
    """Return a frame containing exactly ``columns`` (missing ones become NaN)."""
    columns = list(columns)
    out = frame.copy()
    for column in columns:
        if column not in out.columns:
            out[column] = None
    return out[columns]


def _upsert(frame: pd.DataFrame, table: str, columns: Sequence[str], db_path: str) -> int:
    """INSERT OR REPLACE ``frame`` into ``table``; returns the row count written."""
    if frame is None or frame.empty:
        return 0

    prepared = _prepare(frame, columns).astype(object)
    prepared = prepared.where(pd.notnull(prepared), None)
    rows = list(prepared.itertuples(index=False, name=None))
    placeholders = ", ".join("?" for _ in columns)
    sql = f"INSERT OR REPLACE INTO {table} ({', '.join(columns)}) VALUES ({placeholders})"

    with _WRITE_LOCK:
        with get_connection(db_path) as conn:
            conn.executemany(sql, rows)
    return len(rows)


# --------------------------------------------------------------------------- #
# Writes
# --------------------------------------------------------------------------- #


def upsert_daily_prices(frame: pd.DataFrame, db_path: str = DB_PATH) -> int:
    """Store long-format OHLCV data: ticker, date, open, high, low, close, volume."""
    if frame is None or frame.empty:
        return 0
    frame = _normalize_dates(frame)
    return _upsert(frame, "daily_prices", PRICE_COLUMNS, db_path)


def upsert_sector_fundamentals(frame: pd.DataFrame, db_path: str = DB_PATH) -> int:
    """Store one fundamentals row per ticker."""
    if frame is None or frame.empty:
        return 0
    frame = frame.copy()
    if "updated_at" not in frame.columns or frame["updated_at"].isna().all():
        frame["updated_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    return _upsert(frame, "sector_fundamentals", FUNDAMENTAL_COLUMNS, db_path)


def upsert_macro_indicators(frame: pd.DataFrame, db_path: str = DB_PATH) -> int:
    """Store long-format macro series: series_id, date, value."""
    if frame is None or frame.empty:
        return 0
    frame = _normalize_dates(frame)
    return _upsert(frame, "macro_indicators", MACRO_COLUMNS, db_path)


def set_meta(key: str, value: str, db_path: str = DB_PATH) -> None:
    with _WRITE_LOCK:
        with get_connection(db_path) as conn:
            conn.execute(
                "INSERT OR REPLACE INTO meta (key, value) VALUES (?, ?)", (key, str(value))
            )


def mark_refreshed(db_path: str = DB_PATH) -> str:
    """Record 'now' as the last successful refresh and return the stamp."""
    stamp = datetime.now(timezone.utc).isoformat(timespec="seconds")
    set_meta("last_refresh", stamp, db_path=db_path)
    return stamp


# --------------------------------------------------------------------------- #
# Reads
# --------------------------------------------------------------------------- #


def get_meta(key: str, default: Optional[str] = None, db_path: str = DB_PATH) -> Optional[str]:
    try:
        with get_connection(db_path) as conn:
            row = conn.execute("SELECT value FROM meta WHERE key = ?", (key,)).fetchone()
    except sqlite3.Error:
        return default
    return row[0] if row else default


def get_last_refresh(db_path: str = DB_PATH) -> Optional[datetime]:
    raw = get_meta("last_refresh", db_path=db_path)
    if not raw:
        return None
    try:
        return datetime.fromisoformat(raw)
    except ValueError:
        return None


def read_daily_prices(
    tickers: Optional[Sequence[str]] = None,
    start: Optional[str] = None,
    db_path: str = DB_PATH,
) -> pd.DataFrame:
    """Read OHLCV history, optionally filtered by ticker list and start date."""
    clauses, params = [], []
    if tickers:
        clauses.append(f"ticker IN ({', '.join('?' for _ in tickers)})")
        params.extend(list(tickers))
    if start:
        clauses.append("date >= ?")
        params.append(str(start)[:10])

    sql = "SELECT ticker, date, open, high, low, close, volume FROM daily_prices"
    if clauses:
        sql += " WHERE " + " AND ".join(clauses)
    sql += " ORDER BY date ASC, ticker ASC"

    with get_connection(db_path) as conn:
        frame = pd.read_sql_query(sql, conn, params=params)

    if not frame.empty:
        frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
        frame = frame.dropna(subset=["date"])
    return frame


def read_sector_fundamentals(db_path: str = DB_PATH) -> pd.DataFrame:
    with get_connection(db_path) as conn:
        return pd.read_sql_query(
            "SELECT * FROM sector_fundamentals ORDER BY market_cap DESC", conn
        )


def read_macro_indicators(
    series_id: Optional[str] = None, db_path: str = DB_PATH
) -> pd.DataFrame:
    sql = "SELECT series_id, date, value FROM macro_indicators"
    params: list = []
    if series_id:
        sql += " WHERE series_id = ?"
        params.append(series_id)
    sql += " ORDER BY date ASC"

    with get_connection(db_path) as conn:
        frame = pd.read_sql_query(sql, conn, params=params)

    if not frame.empty:
        frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
        frame = frame.dropna(subset=["date"])
    return frame


def macro_series(series_id: str, db_path: str = DB_PATH) -> pd.Series:
    """Return a single macro indicator as a date-indexed Series."""
    frame = read_macro_indicators(series_id, db_path=db_path)
    if frame.empty:
        return pd.Series(dtype="float64", name=series_id)
    series = frame.set_index("date")["value"].astype(float)
    series.name = series_id
    return series.sort_index()


def database_stats(db_path: str = DB_PATH) -> dict:
    """Row counts and coverage, used by the sidebar status panel."""
    stats = {
        "price_rows": 0,
        "tickers": 0,
        "fundamentals": 0,
        "macro_rows": 0,
        "first_date": None,
        "last_date": None,
        "db_path": db_path,
        "size_mb": 0.0,
    }
    try:
        with get_connection(db_path) as conn:
            stats["price_rows"] = conn.execute("SELECT COUNT(*) FROM daily_prices").fetchone()[0]
            stats["tickers"] = conn.execute(
                "SELECT COUNT(DISTINCT ticker) FROM daily_prices"
            ).fetchone()[0]
            stats["fundamentals"] = conn.execute(
                "SELECT COUNT(*) FROM sector_fundamentals"
            ).fetchone()[0]
            stats["macro_rows"] = conn.execute("SELECT COUNT(*) FROM macro_indicators").fetchone()[0]
            bounds = conn.execute("SELECT MIN(date), MAX(date) FROM daily_prices").fetchone()
            stats["first_date"], stats["last_date"] = bounds
    except sqlite3.Error:
        return stats

    if os.path.exists(db_path):
        stats["size_mb"] = round(os.path.getsize(db_path) / (1024 * 1024), 2)
    return stats


def has_data(db_path: str = DB_PATH) -> bool:
    """True when the cache holds enough to render the dashboard."""
    stats = database_stats(db_path)
    return stats["price_rows"] > 0 and stats["fundamentals"] > 0


def reset_database(db_path: str = DB_PATH) -> None:
    """Drop every row but keep the schema."""
    with _WRITE_LOCK:
        with get_connection(db_path) as conn:
            for table in ("daily_prices", "sector_fundamentals", "macro_indicators", "meta"):
                conn.execute(f"DELETE FROM {table}")
        with get_connection(db_path) as conn:
            conn.execute("VACUUM")
