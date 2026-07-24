"""Predictive metrics — do the scores actually rank forward returns?

Rank IC (daily Spearman corr between score and realized forward return) is the workhorse. A
strategy is only worth backtesting if mean Rank IC > 0 with a decent IC IR (mean/std) and the
top decile out-earns the bottom decile monotonically.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import spearmanr


def daily_rank_ic(scores: pd.Series, fwd_ret: pd.Series, dates: pd.Series) -> pd.Series:
    """Spearman rank correlation per day between score and realized forward return."""
    df = pd.DataFrame({"s": scores.values, "r": fwd_ret.values, "d": dates.values}).dropna()

    def _ic(g: pd.DataFrame) -> float:
        if len(g) < 5 or g["s"].std() == 0 or g["r"].std() == 0:
            return np.nan
        return spearmanr(g["s"], g["r"]).statistic

    return df.groupby("d").apply(_ic).dropna()


def ic_summary(ic: pd.Series) -> dict:
    n = len(ic)
    std = ic.std()
    # t-stat of the daily IC series: mean / std * sqrt(n). |t| > 2 ≈ significant at 5%.
    tstat = float(ic.mean() / std * np.sqrt(n)) if std > 0 and n > 1 else np.nan
    return {
        "mean_ic": float(ic.mean()),
        "ic_ir": float(ic.mean() / std) if std > 0 else np.nan,
        "ic_tstat": tstat,
        "ic_significant": bool(abs(tstat) > 2) if not np.isnan(tstat) else False,
        "hit_rate": float((ic > 0).mean()),
        "n_days": int(n),
    }


def ic_by_year(ic: pd.Series) -> dict:
    """Mean IC per calendar year — the slice that reveals regime failures / alpha decay."""
    return {int(y): float(v) for y, v in ic.groupby(ic.index.year).mean().items()}


def bootstrap_ci(series: pd.Series, stat_fn, n_boot: int = 2000, seed: int = 0, alpha: float = 0.05) -> tuple[float, float]:
    """Percentile bootstrap CI for a statistic of a return series (e.g. Sharpe). Resamples periods
    i.i.d. — a reasonable approximation for non-overlapping period returns."""
    rng = np.random.default_rng(seed)
    vals = series.dropna().to_numpy()
    if len(vals) < 10:
        return (np.nan, np.nan)
    boots = [stat_fn(pd.Series(rng.choice(vals, size=len(vals), replace=True))) for _ in range(n_boot)]
    lo, hi = np.percentile(boots, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return (float(lo), float(hi))


def decile_spread(scores: pd.Series, fwd_ret: pd.Series, dates: pd.Series, q: int = 10) -> pd.DataFrame:
    """Mean forward return by score-decile, per day then averaged. Checks monotonicity."""
    df = pd.DataFrame({"s": scores.values, "r": fwd_ret.values, "d": dates.values}).dropna()

    def _bucket(g: pd.DataFrame) -> pd.DataFrame:
        g = g.copy()
        g["decile"] = pd.qcut(g["s"].rank(method="first"), q, labels=False, duplicates="drop")
        return g

    df = df.groupby("d", group_keys=False).apply(_bucket)
    tbl = df.groupby("decile")["r"].mean().to_frame("mean_fwd_ret")
    tbl["monotonic"] = tbl["mean_fwd_ret"].is_monotonic_increasing
    return tbl
