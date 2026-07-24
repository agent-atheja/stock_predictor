"""Label: 5-trading-day forward return, the prediction target.

  fwd_ret_5d = adj_close[t+h]/adj_close[t] - 1        (h = config label.horizon_days)

Processing:
  1. Winsorize per day at 1/99 pct (kill corporate-action / data spikes).
  2. Cross-sectional z-score per day → the ranking target (regime-robust, scale-free).

The forward window is the ONLY intentionally future-looking computation in the pipeline; the
walk-forward embargo (> horizon) is what prevents it from leaking into training. We also emit a
binary `fwd_up_5d` for a classification cross-check and keep raw `fwd_ret_5d` for backtest PnL.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from core.config import load_config


def add_labels(panel: pd.DataFrame) -> pd.DataFrame:
    cfg = load_config().label
    h = cfg.horizon_days
    panel = panel.sort_values(["symbol", "date"]).copy()

    # raw forward return per symbol
    panel["fwd_ret_5d"] = (
        panel.groupby("symbol")["adj_close"].shift(-h) / panel["adj_close"] - 1
    )

    # winsorize per day
    if cfg.winsorize_pct and cfg.winsorize_pct > 0:
        p = cfg.winsorize_pct

        def _wins(s: pd.Series) -> pd.Series:
            lo, hi = s.quantile(p), s.quantile(1 - p)
            return s.clip(lo, hi)

        panel["fwd_ret_5d_w"] = panel.groupby("date")["fwd_ret_5d"].transform(_wins)
    else:
        panel["fwd_ret_5d_w"] = panel["fwd_ret_5d"]

    # cross-sectional z-score per day → the ranking target
    if cfg.cross_sectional_zscore:
        grp = panel.groupby("date")["fwd_ret_5d_w"]
        panel["y"] = (panel["fwd_ret_5d_w"] - grp.transform("mean")) / grp.transform("std")
    else:
        panel["y"] = panel["fwd_ret_5d_w"]

    panel["fwd_up_5d"] = (panel["fwd_ret_5d"] > 0).astype("Int8")
    return panel
