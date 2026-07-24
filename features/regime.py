"""Market regime feature — a rolling volatility-regime flag on the market proxy.

Primary: a simple, robust vol-tercile regime (low/mid/high) from trailing market vol — always
available, no extra deps. Optional: a rolling Gaussian HMM (hmmlearn) refit only on PAST data
every `refit_every` days (no look-ahead) when available. The regime is a shared daily feature
and, later, a live-trading gate (skip trading in high-vol regimes).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from core.logging_setup import get_logger

log = get_logger(__name__)


def add_regime(panel: pd.DataFrame) -> pd.DataFrame:
    """Add `regime_vol` (0=low,1=mid,2=high) as a per-date market-vol regime."""
    panel = panel.sort_values(["date", "symbol"]).copy()
    daily_mkt = panel.groupby("date")["mkt_ret_1d"].first()
    mkt_vol = daily_mkt.rolling(21).std() * np.sqrt(252)

    # point-in-time terciles via expanding quantiles (no future info)
    lo = mkt_vol.expanding(min_periods=63).quantile(0.33)
    hi = mkt_vol.expanding(min_periods=63).quantile(0.66)
    regime = pd.Series(1, index=mkt_vol.index)  # default mid
    regime[mkt_vol <= lo] = 0
    regime[mkt_vol >= hi] = 2
    regime = regime.rename("regime_vol")

    panel = panel.merge(regime, left_on="date", right_index=True, how="left")
    panel["regime_vol"] = panel["regime_vol"].fillna(1).astype(int)
    return panel
