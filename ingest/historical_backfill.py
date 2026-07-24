"""Deep historical backfill → Bronze (raw, immutable, append-only).

Primary path: jugaad-data pulls per-symbol NSE history (which under the hood aggregates the
bhavcopy archive) for the full membership set, INCLUDING delisted names. Parallelized with a
rate-limited thread pool (I/O-bound). Bronze is an exact raw copy — no adjustment here; that
happens in Silver.

Run:  python -m ingest.historical_backfill
"""
from __future__ import annotations

from datetime import date, datetime

import pandas as pd

from core.config import load_config
from core.io import write_partitioned
from core.logging_setup import get_logger, stage
from core.parallel import io_map
from ingest.constituents import all_symbols_ever

log = get_logger(__name__)

_BRONZE_COLS = ["symbol", "date", "open", "high", "low", "close", "volume"]


def _fetch_one_jugaad(symbol: str, start: date, end: date) -> pd.DataFrame:
    """Fetch one symbol's daily history via jugaad-data. Returns Bronze-shaped frame."""
    from jugaad_data.nse import stock_df

    df = stock_df(symbol=symbol, from_date=start, to_date=end, series="EQ")
    if df is None or df.empty:
        return pd.DataFrame(columns=_BRONZE_COLS)
    df = df.rename(
        columns={
            "DATE": "date",
            "OPEN": "open",
            "HIGH": "high",
            "LOW": "low",
            "CLOSE": "close",
            "VOLUME": "volume",
        }
    )
    df["symbol"] = symbol
    # jugaad stores IST-midnight as naive UTC (18:30 = 00:00 IST next day). Add the +5:30 IST
    # offset and normalize to recover the TRUE calendar trade date — otherwise every bar is
    # shifted back a day and the forward-return label misaligns.
    df["date"] = (pd.to_datetime(df["date"]) + pd.Timedelta(hours=5, minutes=30)).dt.normalize()
    return df[_BRONZE_COLS]


def backfill(start: str | None = None, end: str | None = None) -> int:
    cfg = load_config()
    start_d = pd.Timestamp(start or cfg.data.history_start).date()
    end_d = pd.Timestamp(end or datetime.today().date()).date()
    symbols = all_symbols_ever()
    log.info("Backfilling %d symbols from %s to %s", len(symbols), start_d, end_d)

    def _job(sym: str) -> pd.DataFrame:
        return _fetch_one_jugaad(sym, start_d, end_d)

    with stage(log, "bronze-backfill"):
        # I/O-bound → threaded, rate-limited so we don't hammer NSE.
        frames = io_map(_job, symbols, rate_limit_per_sec=2.5, desc="backfill")
        frames = [f for f in frames if f is not None and not f.empty]
        if not frames:
            log.error("No data fetched — check network / jugaad-data install / symbols.")
            return 0
        allrows = pd.concat(frames, ignore_index=True)
        # De-dup on (symbol, date); Bronze is append-only but re-runs must stay idempotent.
        allrows = allrows.drop_duplicates(subset=["symbol", "date"]).sort_values(["symbol", "date"])
        n = write_partitioned(allrows, cfg.data.bronze, "equity_ohlcv", date_col="date")
    log.info("Bronze backfill complete: %d rows, %d symbols", n, allrows["symbol"].nunique())
    return n


if __name__ == "__main__":
    backfill()
