"""Backtest performance metrics from a period-return series (non-overlapping h-day periods)."""
from __future__ import annotations

import numpy as np
import pandas as pd


def performance(period_returns: pd.Series, periods_per_year: float) -> dict:
    r = period_returns.dropna()
    if r.empty:
        return {}
    equity = (1 + r).cumprod()
    total = equity.iloc[-1] - 1
    years = len(r) / periods_per_year
    cagr = (equity.iloc[-1]) ** (1 / years) - 1 if years > 0 else np.nan
    vol = r.std() * np.sqrt(periods_per_year)
    sharpe = (r.mean() * periods_per_year) / vol if vol > 0 else np.nan
    downside = r[r < 0].std() * np.sqrt(periods_per_year)
    sortino = (r.mean() * periods_per_year) / downside if downside > 0 else np.nan
    dd = equity / equity.cummax() - 1
    return {
        "total_return": float(total),
        "cagr": float(cagr),
        "ann_vol": float(vol),
        "sharpe": float(sharpe),
        "sortino": float(sortino),
        "max_drawdown": float(dd.min()),
        "hit_rate": float((r > 0).mean()),
        "n_periods": int(len(r)),
    }


def sharpe_ratio(period_returns: pd.Series, periods_per_year: float) -> float:
    r = period_returns.dropna()
    vol = r.std() * np.sqrt(periods_per_year)
    return float(r.mean() * periods_per_year / vol) if vol > 0 else np.nan


def equity_curve(period_returns: pd.Series) -> pd.Series:
    return (1 + period_returns.fillna(0)).cumprod()
