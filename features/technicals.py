"""Per-symbol technical features — pure pandas/numpy (no TA-Lib; keeps the project
self-contained). Every feature uses ONLY data up to and including bar t (point-in-time);
no forward-looking windows. The assembler asserts this invariant downstream.

Input: one symbol's adjusted OHLCV sorted by date. Output: same frame + feature columns.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from core.config import load_config


def _rsi(close: pd.Series, period: int) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - 100 / (1 + rs)


def _atr(df: pd.DataFrame, period: int) -> pd.Series:
    hl = df["adj_high"] - df["adj_low"]
    hc = (df["adj_high"] - df["adj_close"].shift()).abs()
    lc = (df["adj_low"] - df["adj_close"].shift()).abs()
    tr = pd.concat([hl, hc, lc], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()


def add_technicals(df: pd.DataFrame) -> pd.DataFrame:
    """Add the technical feature block for a single symbol (df sorted by date)."""
    cfg = load_config().features
    df = df.sort_values("date").copy()
    c = df["adj_close"]

    # returns / momentum (log returns are stationary)
    logret = np.log(c).diff()
    for w in cfg.return_windows:
        df[f"ret_{w}d"] = c.pct_change(w)
    df["mom_12_1"] = c.shift(21) / c.shift(252) - 1  # 12-1 momentum (skip most recent month)

    # trend
    ema12, ema26 = c.ewm(span=12, adjust=False).mean(), c.ewm(span=26, adjust=False).mean()
    df["ema_ratio_12_26"] = ema12 / ema26 - 1
    df["sma_ratio_20_50"] = c.rolling(20).mean() / c.rolling(50).mean() - 1
    macd = ema12 - ema26
    df["macd_hist"] = macd - macd.ewm(span=9, adjust=False).mean()
    df["dist_52w_high"] = c / c.rolling(252, min_periods=60).max() - 1

    # mean-reversion
    df["rsi"] = _rsi(c, cfg.rsi_period)
    bb_mid = c.rolling(cfg.bb_period).mean()
    bb_std = c.rolling(cfg.bb_period).std()
    df["bb_pctb"] = (c - (bb_mid - 2 * bb_std)) / (4 * bb_std)
    df["zscore_20d"] = (c - bb_mid) / bb_std

    # volatility
    for w in cfg.vol_windows:
        df[f"vol_{w}d"] = logret.rolling(w).std() * np.sqrt(252)
    df["atr_pct"] = _atr(df, cfg.atr_period) / c
    df["vol_of_vol"] = logret.rolling(21).std().rolling(21).std()

    # volume / liquidity
    vol = df["volume"]
    df["vol_z_20d"] = (vol - vol.rolling(20).mean()) / vol.rolling(20).std()
    df["turnover_z_20d"] = (
        df["turnover"] - df["turnover"].rolling(20).mean()
    ) / df["turnover"].rolling(20).std()
    # Amihud illiquidity: |ret| / turnover, higher = more illiquid
    df["amihud_21d"] = (logret.abs() / df["turnover"].replace(0, np.nan)).rolling(21).mean()

    return df
