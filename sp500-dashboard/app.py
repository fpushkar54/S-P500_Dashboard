"""S&P 500 Health Dashboard - application entry point.

Run with:

    streamlit run app.py

The file does four things and nothing else: configure the page, run the cached
data pipeline, draw the KPI bar, and hand each tab the slice of data it needs.
All computation lives in ``backend`` and all markup lives in ``ui``.
"""

from __future__ import annotations

from typing import Dict

import pandas as pd
import streamlit as st

from backend import data_loader, database, metrics
from ui import components, tab_breadth, tab_prediction, tab_screener, tab_valuation

st.set_page_config(
    page_title="S&P 500 Health Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

CACHE_TTL = 60 * 30  # half an hour


# --------------------------------------------------------------------------- #
# Cached pipeline
# --------------------------------------------------------------------------- #


@st.cache_data(show_spinner=False, ttl=CACHE_TTL)
def load_dataset(cache_key: str) -> Dict[str, object]:
    """Read the cached dataset from SQLite.

    ``cache_key`` carries the last-refresh stamp so a new download invalidates
    this cache without needing a global clear.
    """
    return data_loader.load_market_data()


@st.cache_data(show_spinner=False, ttl=CACHE_TTL)
def compute_analytics(prices: pd.DataFrame, fundamentals: pd.DataFrame) -> Dict[str, object]:
    """Derive every metric the tabs render."""
    breadth = metrics.calculate_breadth(prices)
    concentration = metrics.calculate_concentration(fundamentals)
    return {
        "breadth": breadth,
        "breadth_snapshot": metrics.breadth_snapshot(breadth),
        "ad_line": metrics.advance_decline_line(prices),
        "concentration": concentration,
        "sector_valuation": metrics.sector_valuation(fundamentals),
        "valuation_summary": metrics.index_valuation_summary(fundamentals),
        "screener": metrics.build_screener_table(prices, fundamentals),
        "sector_performance": metrics.sector_performance(prices, fundamentals),
    }


# --------------------------------------------------------------------------- #
# Refresh
# --------------------------------------------------------------------------- #


def run_refresh(settings: Dict[str, object]) -> None:
    """Download everything and write it to SQLite, with live progress."""
    progress = st.progress(0.0, text="Fetching index membership...")
    status = st.empty()

    def on_progress(completed: int, total: int, ticker: str) -> None:
        fraction = completed / max(total, 1)
        progress.progress(
            min(fraction, 1.0), text=f"Downloading {completed} of {total} tickers ({ticker})"
        )

    try:
        summary = data_loader.refresh_market_data(
            period=str(settings["period"]),
            max_workers=int(settings["workers"]),
            limit=settings["limit"],
            progress_callback=on_progress,
        )
    except Exception as exc:
        progress.empty()
        st.error(f"Refresh failed: {exc}")
        return

    progress.empty()
    st.cache_data.clear()

    failed = summary.get("failed_tickers") or []
    status.success(
        f"Loaded {summary['tickers_loaded']} of {summary['tickers_requested']} tickers "
        f"({summary['price_rows']:,} price rows, {summary['macro_rows']:,} macro rows)."
    )
    if failed:
        with st.expander(f"{len(failed)} tickers returned no data"):
            st.write(", ".join(sorted(failed)))
    if summary.get("constituent_source") == "fallback":
        st.warning(
            "Wikipedia was unreachable, so a built-in subset of the index was used. "
            "Refresh again later for the full 500."
        )


def render_first_run(settings: Dict[str, object]) -> None:
    """Landing screen shown while the cache is empty."""
    st.markdown(
        """
        <div class="masthead">
          <h1>S&amp;P 500 Health Dashboard</h1>
          <p>Nothing is cached yet. Download the index to get started.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    components.empty_state(
        "First run downloads prices, fundamentals and macro series.",
        "The full 500 takes a few minutes. Pick a smaller universe in the sidebar for a "
        "faster start — you can always widen it later.",
    )
    if st.button("Download market data now", type="primary"):
        run_refresh(settings)
        st.rerun()


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #


def main() -> None:
    components.load_css()
    database.initialize_database()

    stats = database.database_stats()
    last_refresh = database.get_last_refresh()
    settings = components.render_sidebar(stats, last_refresh)

    if settings["reset"]:
        database.reset_database()
        st.cache_data.clear()
        st.rerun()

    if settings["refresh"]:
        run_refresh(settings)
        st.rerun()

    if not database.has_data():
        render_first_run(settings)
        return

    cache_key = str(last_refresh) if last_refresh else "bootstrap"
    with st.spinner("Loading cached market data..."):
        dataset = load_dataset(cache_key)

    prices: pd.DataFrame = dataset["prices"]
    fundamentals: pd.DataFrame = dataset["fundamentals"]
    macro: Dict[str, pd.Series] = dataset["macro"]

    with st.spinner("Computing index metrics..."):
        analytics = compute_analytics(prices, fundamentals)

    components.render_masthead(last_refresh, len(fundamentals))
    components.render_kpi_bar(
        components.build_summary_kpis(
            macro=macro,
            breadth_snapshot=analytics["breadth_snapshot"],
            concentration=analytics["concentration"],
            valuation=analytics["valuation_summary"],
        )
    )
    st.write("")

    valuation_tab, breadth_tab, prediction_tab, screener_tab = st.tabs(
        [
            "Valuation & macro",
            "Breadth & concentration",
            "Predictions",
            "Screener",
        ]
    )

    with valuation_tab:
        tab_valuation.render(
            macro=macro,
            sector_valuation=analytics["sector_valuation"],
            valuation_summary=analytics["valuation_summary"],
        )

    with breadth_tab:
        tab_breadth.render(
            breadth=analytics["breadth"],
            ad_line=analytics["ad_line"],
            concentration=analytics["concentration"],
            macro=macro,
        )

    with prediction_tab:
        tab_prediction.render(macro=macro, breadth=analytics["breadth"])

    with screener_tab:
        tab_screener.render(
            screener=analytics["screener"],
            sector_performance=analytics["sector_performance"],
        )

    components.render_footnote(
        "Sources: Yahoo Finance (prices, fundamentals, VIX), FRED (10-year Treasury yield), "
        "Wikipedia (index membership). Data is delayed and provided for research only. "
        "Nothing here is investment advice."
    )


if __name__ == "__main__":
    main()
