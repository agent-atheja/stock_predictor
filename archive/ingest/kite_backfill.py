"""Deep historical backfill via Kite → Bronze. RELIABLE alternative to jugaad/NSE scraping.

Why this exists: the free jugaad path scrapes NSE's AJAX endpoint, which IP-throttles any batch
(blocks after a handful of requests). Kite is an authenticated, paid API that serves deep daily
history without blocking — the right tool for bulk backfill when you have a Kite subscription.

Kite's daily-candle API returns at most ~2000 bars per call, so multi-year history is fetched in
chunked date windows per symbol, parallelized under the 3 req/s rate limit. All 200 names × ~10y
≈ a few hundred calls ≈ ~3–5 min. Idempotent (per-date partitions overwrite).

Run:  python -m ingest.kite_backfill        (requires a valid Kite token)
"""
from __future__ import annotations

from datetime import date, timedelta

import pandas as pd

from core.config import load_config
from core.io import write_partitioned
from core.logging_setup import get_logger, stage
from core.parallel import io_map
from ingest.constituents import all_symbols_ever

log = get_logger(__name__)

_BRONZE_COLS = ["symbol", "date", "open", "high", "low", "close", "volume"]
_MAX_DAYS = 1900  # stay under Kite's ~2000-bar daily-candle cap per request


def _chunks(start: date, end: date) -> list[tuple[date, date]]:
    out, cur = [], start
    while cur <= end:
        hi = min(cur + timedelta(days=_MAX_DAYS), end)
        out.append((cur, hi))
        cur = hi + timedelta(days=1)
    return out


def _fetch_symbol(sym: str, start: date, end: date) -> pd.DataFrame:
    """Fetch a symbol's full daily history in Kite-sized chunks (sequential within the symbol;
    symbols themselves are parallelized upstream)."""
    from archive.ingest.kite_client import historical

    frames = []
    for lo, hi in _chunks(start, end):
        candles = historical(sym, lo, hi, interval="day")
        if candles:
            frames.append(pd.DataFrame(candles))
    if not frames:
        return pd.DataFrame(columns=_BRONZE_COLS)
    df = pd.concat(frames, ignore_index=True)
    df["date"] = pd.to_datetime(df["date"])  # Kite returns IST-correct dates
    return df.drop_duplicates(subset=["date"])[_BRONZE_COLS]


def backfill(start: str | None = None, end: str | None = None,
             symbols: list[str] | None = None) -> int:
    cfg = load_config()
    start_d = pd.Timestamp(start or cfg.data.history_start).date()
    end_d = pd.Timestamp(end or date.today()).date()
    symbols = symbols if symbols is not None else all_symbols_ever()
    log.info("Kite backfill: %d symbols, %s → %s (%d chunk(s)/symbol)",
             len(symbols), start_d, end_d, len(_chunks(start_d, end_d)))

    def _job(sym: str) -> pd.DataFrame:
        return _fetch_symbol(sym, start_d, end_d)

    with stage(log, "bronze-kite-backfill"):
        # I/O-bound; honor Kite's 3 req/s across chunked calls.
        frames = io_map(_job, symbols, rate_limit_per_sec=cfg.concurrency.kite_rate_limit_per_sec,
                        desc="kite-backfill")
        frames = [f for f in frames if f is not None and not f.empty]
        if not frames:
            log.error("No data — is the Kite token valid? Run: python main.py kite-login <token>")
            return 0
        allrows = pd.concat(frames, ignore_index=True).drop_duplicates(subset=["symbol", "date"])
        allrows = allrows.sort_values(["symbol", "date"])
        n = write_partitioned(allrows, cfg.data.bronze, "equity_ohlcv", date_col="date")
    log.info("Kite backfill complete: %d rows, %d symbols", n, allrows["symbol"].nunique())
    return n


if __name__ == "__main__":
    backfill()
