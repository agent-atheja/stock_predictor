"""Cross-sectional / market-structure features computed ACROSS symbols per day.

These need the whole panel (not one symbol), so they run after per-symbol technicals:
  • beta to the market (rolling regression of stock ret on index ret)
  • market-relative return (stock ret minus equal-weight universe ret)
  • cross-sectional rank of momentum (breadth-aware positioning)

Still strictly point-in-time: everything uses data up to bar t only.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def add_market_structure(panel: pd.DataFrame, beta_window: int = 63) -> pd.DataFrame:
    """panel: long frame (date, symbol, adj_close, ret_1d, mom_12_1, ...) for the universe."""
    panel = panel.sort_values(["date", "symbol"]).copy()

    # equal-weight universe daily return = market proxy (breadth-consistent, self-contained)
    mkt = panel.groupby("date")["ret_1d"].transform("mean")
    panel["mkt_ret_1d"] = mkt
    panel["rel_ret_1d"] = panel["ret_1d"] - mkt

    # rolling beta per symbol: cov(stock, mkt) / var(mkt) over trailing window
    def _beta(g: pd.DataFrame) -> pd.Series:
        s, m = g["ret_1d"], g["mkt_ret_1d"]
        cov = s.rolling(beta_window).cov(m)
        var = m.rolling(beta_window).var()
        return cov / var

    panel["beta_63d"] = (
        panel.groupby("symbol", group_keys=False).apply(_beta).reset_index(level=0, drop=True)
    )

    # cross-sectional percentile ranks (0..1) — regime-robust relative positioning
    for col in ("mom_12_1", "ret_21d", "vol_21d"):
        if col in panel.columns:
            panel[f"xs_rank_{col}"] = panel.groupby("date")[col].rank(pct=True)

    return panel
