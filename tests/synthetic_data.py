"""Synthetic OHLCV generator — lets the whole pipeline run/verify without live credentials.

Produces geometric-Brownian-motion daily bars for a set of symbols over a date range, with a
small injected cross-sectional signal (momentum persistence) so the model has *something* real
to learn — good enough to prove wiring, metrics, and leakage controls end-to-end.

Usage:  python -m tests.synthetic_data              # writes bronze for stub/universe symbols
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from core.config import load_config
from core.io import write_partitioned
from core.logging_setup import get_logger
from ingest.constituents import all_symbols_ever

log = get_logger(__name__)


def generate(symbols: list[str], start: str, end: str, seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(start, end)
    frames = []
    for i, sym in enumerate(symbols):
        n = len(dates)
        mu = 0.0003 + 0.0002 * rng.standard_normal()          # small per-symbol drift
        vol = 0.012 + 0.006 * rng.random()                    # daily vol
        shocks = rng.standard_normal(n) * vol + mu
        # inject weak momentum autocorrelation so forward return is partly predictable
        for t in range(1, n):
            shocks[t] += 0.05 * shocks[t - 1]
        close = 100 * (1 + i * 0.1) * np.exp(np.cumsum(shocks))
        intraday = np.abs(rng.standard_normal(n)) * vol
        high = close * (1 + intraday)
        low = close * (1 - intraday)
        open_ = np.concatenate([[close[0]], close[:-1]])
        volume = rng.integers(1e5, 5e6, n)
        frames.append(
            pd.DataFrame(
                {
                    "symbol": sym,
                    "date": dates,
                    "open": open_,
                    "high": np.maximum.reduce([high, open_, close]),
                    "low": np.minimum.reduce([low, open_, close]),
                    "close": close,
                    "volume": volume,
                }
            )
        )
    return pd.concat(frames, ignore_index=True)


def write_bronze(start: str = "2016-01-01", end: str = "2024-12-31") -> int:
    cfg = load_config()
    symbols = all_symbols_ever()
    df = generate(symbols, start, end)
    n = write_partitioned(df, cfg.data.bronze, "equity_ohlcv", date_col="date")
    log.info("Synthetic bronze written: %d rows, %d symbols, %s→%s", n, len(symbols), start, end)
    return n


if __name__ == "__main__":
    write_bronze()
