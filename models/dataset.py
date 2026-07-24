"""Model dataset — Gold → (X, y, groups) with the tradable-universe filter applied.

The universe filter is a POINT-IN-TIME gate: on each date keep only rows that were index
members AND cleared the liquidity/price floors on that date. This is what makes the backtest
honest (we never train or trade on names we couldn't actually have held then).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from core.config import load_config
from core.io import read_dataset
from core.logging_setup import get_logger
from features.assembler import FEATURE_COLS

log = get_logger(__name__)


def load_gold() -> pd.DataFrame:
    cfg = load_config()
    gold = read_dataset(cfg.data.gold, "features")
    if gold.empty:
        raise RuntimeError("Gold is empty — run features.build_gold first.")
    gold["date"] = pd.to_datetime(gold["date"])
    return gold.sort_values(["date", "symbol"]).reset_index(drop=True)


def apply_universe_filter(df: pd.DataFrame) -> pd.DataFrame:
    """Keep PIT-tradable rows: index member, price ≥ floor, turnover ≥ floor."""
    cfg = load_config().universe
    mask = (
        df["is_member"].fillna(False)
        & (df["adj_close"] >= cfg.min_price_inr)
        & (df["turnover_cr_20d"].fillna(0) >= cfg.min_avg_turnover_cr)
    )
    out = df[mask].copy()
    log.info("Universe filter: %d → %d rows (%.0f%% kept)", len(df), len(out), 100 * len(out) / max(len(df), 1))
    return out


def make_xy(df: pd.DataFrame, require_label: bool = True) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    """Return (X features, y target, meta[date,symbol,fwd_ret_5d]). Drops rows with NaN features."""
    d = df.copy()
    d[FEATURE_COLS] = d[FEATURE_COLS].replace([np.inf, -np.inf], np.nan)
    d = d.dropna(subset=FEATURE_COLS)
    if require_label:
        d = d.dropna(subset=["y"])
    X = d[FEATURE_COLS]
    y = d["y"] if "y" in d else pd.Series(index=d.index, dtype=float)
    meta = d[["date", "symbol", "fwd_ret_5d"]].copy()
    return X, y, meta


def day_groups(dates: pd.Series) -> list[int]:
    """LightGBM ranker query groups = contiguous counts per date. Requires date-sorted input."""
    return dates.groupby(dates).size().tolist()
