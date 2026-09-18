# S&P 500 Health Dashboard

A Streamlit application that tracks all 500 index constituents and turns them
into four views: valuation against the macro backdrop, market breadth and
concentration, a machine-learning prediction panel, and a full screener.

Everything downloaded is cached in a local SQLite file, so a restart costs a
second rather than a fresh download of 500 tickers.

---

## Quick start

```bash
# 1. Clone or copy the project, then move into it
cd sp500-dashboard

# 2. Create and activate a virtual environment
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

# 3. Install dependencies
pip install --upgrade pip
pip install -r requirements.txt

# 4. Run
streamlit run app.py
```

The app opens at <http://localhost:8501>.

On first launch the cache is empty and you'll see a download prompt. Pick a
universe size in the sidebar first — **Top 50** finishes in under a minute and
is the right way to confirm everything works before pulling the full index,
which takes a few minutes depending on how aggressively Yahoo throttles.

Requires Python 3.10 or newer.

---

## What each tab shows

**Valuation & macro.** The S&P 500 plotted against the 10-year Treasury yield on
a dual axis, forward P/E by GICS sector (plain median or cap-weighted), and a
VIX gauge with its one-year percentile.

**Breadth & concentration.** Percentage of constituents above their 50- and
200-day moving averages over time, the cumulative advance-decline line overlaid
on the index, and top-5/10/20 market cap concentration as a donut and a treemap.

**Predictions.** A Random Forest classifier estimates the probability the index
closes higher 5, 20 or 60 sessions out, with feature importances and
walk-forward accuracy. An ARIMA model fitted on log prices draws a 30-session
projection cone with 80% and 95% bands.

**Screener.** Every tracked constituent with ticker, company, sector, market
cap, price, trailing and forward P/E, and distance from the 200-day average.
Filter by sector, trend position, valuation and size; export the filtered view
as CSV.

---

## Architecture

```text
sp500-dashboard/
├── app.py                  Page config, cached pipeline, tab orchestration
├── requirements.txt
├── README.md
├── backend/                No Streamlit imports — pure, testable logic
│   ├── __init__.py
│   ├── data_loader.py      Wikipedia scrape, threaded yfinance pulls, FRED macro
│   ├── database.py         SQLite schema, upserts, read queries
│   ├── metrics.py          Breadth, A/D line, concentration, valuation, screener
│   └── ml_engine.py        Feature engineering, Random Forest, ARIMA
├── ui/                     No data fetching — presentation only
│   ├── __init__.py
│   ├── components.py       CSS loader, KPI cards, sidebar, formatters
│   ├── tab_valuation.py
│   ├── tab_breadth.py
│   ├── tab_prediction.py
│   └── tab_screener.py
├── assets/
│   └── style.css           Dark theme and colour tokens
└── images/                 Screenshots for documentation
```

The split is deliberate: `backend` never imports Streamlit, so the metric and
model functions can be imported into a notebook or a test suite unchanged. `ui`
never fetches or computes — it receives finished frames and renders them.

### Data flow

```text
Wikipedia ──► constituents ─┐
                            ├──► ThreadPoolExecutor ──► SQLite ──► @st.cache_data ──► tabs
yfinance  ──► OHLCV + fundamentals
FRED      ──► DGS10, plus ^GSPC and ^VIX from Yahoo
```

### Database schema

`sp500_health.db` is created in the project root on first run.

| Table | Key | Contents |
|---|---|---|
| `daily_prices` | (ticker, date) | Adjusted OHLCV |
| `sector_fundamentals` | ticker | Name, GICS sector, market cap, P/E ratios, beta |
| `macro_indicators` | (series_id, date) | `SP500`, `VIX`, `DGS10` |
| `meta` | key | Last refresh timestamp |

Writes use `INSERT OR REPLACE`, so re-running a refresh updates rows in place
rather than duplicating them.

---

## Data sources and their quirks

| Source | Used for | Watch out for |
|---|---|---|
| Wikipedia | Index membership and GICS sectors | Page structure changes occasionally; a built-in fallback list keeps the app running |
| Yahoo Finance (`yfinance`) | Prices, fundamentals, `^GSPC`, `^VIX` | Unofficial API — throttles under heavy parallelism, and some tickers return partial fundamentals |
| Stooq | Automatic price fallback | No API key needed; fundamentals not available, prices only |
| FRED | `DGS10` 10-year Treasury yield | Falls back to Yahoo's `^TNX` (which quotes the yield ×10) if FRED is unreachable |

Ticker symbols are normalised for Yahoo: `BRK.B` becomes `BRK-B`.

### Why not Google Finance

There is no Google Finance API. Google deprecated it in May 2011, shut it down
in October 2012, and the June 2026 consumer relaunch did not bring it back. The
only official access to Google's numbers is the `GOOGLEFINANCE` formula inside
Google Sheets, which is a spreadsheet function rather than a callable endpoint.
The alternative — scraping the Google Finance page — depends on CSS classes that
rotate, which is not something to put underneath a data pipeline.

The price layer is pluggable instead. `PRIMARY_PRICE_SOURCE` in
`backend/data_loader.py` (or the `SP500_PRICE_SOURCE` environment variable)
selects the provider, and Stooq is used automatically whenever the primary
source returns nothing for a ticker — on a 500-name download that usually means
throttling rather than a dead symbol.

To add a keyed provider such as Alpha Vantage, Finnhub or Tiingo, write a
`fetch_price_history_<name>` with the same signature as the existing two and add
a branch to `fetch_price_history`. Nothing else in the codebase needs to change,
since every downstream function consumes the same long-format frame.

---

## Tuning

- **Parallel downloads** (sidebar): 8–12 workers is the sweet spot. Push it to
  24 and Yahoo starts returning empty frames, which show up in the "tickers
  returned no data" list after a refresh.
- **History depth**: 2 years is the minimum for a trustworthy 200-day average.
  5 years gives the ARIMA fit more to work with.
- **Cache TTL**: `CACHE_TTL` in `app.py`, 30 minutes by default.
- **Database location**: set the `SP500_DB_PATH` environment variable to move
  the SQLite file somewhere else.
- **Price provider**: set `SP500_PRICE_SOURCE` to `yahoo` (default) or `stooq`.

---

## Theming

The palette lives in two places that must be changed together: the CSS custom
properties at the top of `assets/style.css`, and `CHART_COLORS` in
`ui/components.py`, which Plotly needs because charts cannot read CSS variables.
No tab hardcodes a colour — transparency comes from `with_alpha()`.

| Token | Value | Role |
|---|---|---|
| `--ground` | `#011133` | Page background |
| `--panel` / `--panel-raised` | `#041A45` / `#072253` | Card and container surfaces |
| `--accent` | `#70D603` | Tickers, KPI values, active states |
| `--up` / `--down` | `#70D603` / `#FF4B5C` | Market direction |
| `--neutral` | `#5B8FD9` | Non-directional series |
| `--warning` | `#FFB020` | Disclaimers and cautions |
| `--text` / `--text-muted` / `--text-faint` | `#DCE6F7` / `#8FA5C8` / `#7A92C2` | Type scale |

Three decisions worth knowing before you change anything:

**Green is both the brand and "up".** They reinforce each other rather than
competing, which is why down needed its own red instead of inheriting a
complementary hue.

**Warnings are amber, not green.** If the disclaimer banner used the accent, a
caution would read as good news.

**Prose stays pale blue-white.** The accent carries numbers, symbols and state —
long-form text in saturated green is tiring to read and dilutes the signal. Every
foreground colour clears WCAG AA against both the ground and the panel surfaces;
the accent, body text and warning clear AAA.

---

## On the prediction tab

The models are a demonstration of the technique, not a trading signal.

A directional classifier on index prices will usually land in the 50–60%
accuracy range, and much of that comes from the simple fact that equities drift
upward — which is why the panel prints the always-up baseline next to the model
accuracy. If the model isn't clearly beating that baseline on held-out data, it
has found nothing.

Evaluation is chronological throughout: the test block is always the most recent
slice of history, and cross-validation uses `TimeSeriesSplit`. Shuffling would
leak future information into training and produce accuracy figures that look
impressive and mean nothing.

---

## Troubleshooting

**Empty or partial data after a refresh.** Yahoo throttled the request. Lower
the worker count and refresh again; existing rows are kept.

**`ModuleNotFoundError: No module named 'backend'`.** Run `streamlit run app.py`
from the project root, not from inside a subdirectory.

**Wikipedia returns a 403.** The scraper sends a browser user-agent and falls
back to a built-in constituent list. Refreshing later usually resolves it.

**The prediction tab says there aren't enough rows.** The classifier needs about
a year of clean feature rows. Refresh with a longer history window.

---

## Licence and disclaimer

Provided as-is for research and education. Market data belongs to its
respective providers and is subject to their terms. Nothing in this application
is investment advice.
