"""Daily incremental ingestion → Bronze (authoritative recent data via Kite).

Runs post-close: for the CURRENT index members, fetch the last N days of daily candles from
Kite (the authoritative, corporate-action-adjusted-at-source feed) and upsert into Bronze.
Idempotent — re-running the same day overwrites its partition, never duplicates.

Parallelized with the rate-limited thread pool (Kite ~3 req/s hard cap).

Run:  python -m ingest.daily_incremental [lookback_days]
"""
from __future__ import annotations

import sys
from datetime import date, timedelta

import pandas as pd

from core.config import load_config
from core.io import write_partitioned
from core.logging_setup import get_logger, stage
from core.parallel import io_map
from ingest.constituents import members_on

log = get_logger(__name__)

_BRONZE_COLS = ["symbol", "date", "open", "high", "low", "close", "volume"]


def _fetch_one_kite(symbol: str, start: date, end: date) -> pd.DataFrame:
    from archive.ingest.kite_client import historical

    candles = historical(symbol, start, end, interval="day")
    if not candles:
        return pd.DataFrame(columns=_BRONZE_COLS)
    df = pd.DataFrame(candles)
    df["date"] = pd.to_datetime(df["date"])
    return df[_BRONZE_COLS]


def run(lookback_days: int = 7) -> int:
    cfg = load_config()
    today = date.today()
    start = today - timedelta(days=lookback_days)
    symbols = members_on(today)
    log.info("Daily incremental: %d current members, %s → %s", len(symbols), start, today)

    def _job(sym: str) -> pd.DataFrame:
        return _fetch_one_kite(sym, start, today)

    with stage(log, "bronze-incremental"):
        frames = io_map(
            _job, symbols,
            rate_limit_per_sec=cfg.concurrency.kite_rate_limit_per_sec,
            desc="daily",
        )
        frames = [f for f in frames if f is not None and not f.empty]
        if not frames:
            log.error("No candles fetched — is the Kite token valid? (python -m ingest.kite_client login)")
            return 0
        allrows = pd.concat(frames, ignore_index=True).drop_duplicates(subset=["symbol", "date"])
        n = write_partitioned(allrows, cfg.data.bronze, "equity_ohlcv", date_col="date")
    log.info("Daily incremental complete: %d rows", n)
    return n


if __name__ == "__main__":
    run(int(sys.argv[1]) if len(sys.argv) > 1 else 7)
