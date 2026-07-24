"""Portfolio construction from cross-sectional scores.

On each rebalance date: rank the day's universe by score, go long the top-N and (optionally)
short the bottom-N. Weights are equal or inverse-vol ("vol_target") within each leg, capped at
max_position_pct. Long/short legs are dollar-neutral when shorting is enabled.

Returns a per-date frame of target weights (positive long, negative short), summing to +1 on the
long leg and -1 on the short leg (or +1 long-only).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from functools import lru_cache

from core.config import load_config, resolve


@lru_cache(maxsize=1)
def _sector_map() -> dict[str, str]:
    """symbol → sector from config/sectors.csv (empty if absent; then caps are a no-op)."""
    path = resolve("config/sectors.csv")
    if not path.exists():
        return {}
    df = pd.read_csv(path)
    return dict(zip(df["symbol"], df["sector"]))


def _select_with_sector_cap(ranked: pd.DataFrame, n: int, max_sector_pct: float) -> pd.DataFrame:
    """Greedily pick the top-n by score while capping names-per-sector at max_sector_pct·n.
    Enforces diversification so a low-vol tilt can't pile into one defensive sector."""
    sectors = _sector_map()
    if not sectors:
        return ranked.head(n)
    per_sector_cap = max(1, int(max_sector_pct * n))
    picked, counts = [], {}
    for _, row in ranked.iterrows():
        sec = sectors.get(row["symbol"], "UNKNOWN")
        if counts.get(sec, 0) >= per_sector_cap:
            continue
        picked.append(row)
        counts[sec] = counts.get(sec, 0) + 1
        if len(picked) >= n:
            break
    return pd.DataFrame(picked) if picked else ranked.head(n)


def build_book(day: pd.DataFrame) -> pd.DataFrame:
    """day: rows for one rebalance date with columns [symbol, score, fwd_ret_5d, vol_21d?].

    Returns [symbol, weight] target weights for that date, respecting the sector cap.
    """
    cfg = load_config().backtest
    d = day.dropna(subset=["score"]).sort_values("score", ascending=False)
    if d.empty:
        return pd.DataFrame(columns=["symbol", "weight"])

    top = _select_with_sector_cap(d, cfg.top_n, cfg.max_sector_pct)
    legs = [_weight_leg(top, +1.0, cfg)]
    if cfg.bottom_n and cfg.bottom_n > 0 and len(d) > cfg.top_n:
        bottom = _select_with_sector_cap(d.iloc[::-1], cfg.bottom_n, cfg.max_sector_pct)
        legs.append(_weight_leg(bottom, -1.0, cfg))
    book = pd.concat(legs, ignore_index=True)
    return book[["symbol", "weight"]]


def _weight_leg(leg: pd.DataFrame, sign: float, cfg) -> pd.DataFrame:
    leg = leg.copy()
    if cfg.weighting == "vol_target" and "vol_21d" in leg and leg["vol_21d"].notna().any():
        inv = 1.0 / leg["vol_21d"].replace(0, np.nan)
        w = inv / inv.sum()
    else:
        w = pd.Series(1.0 / len(leg), index=leg.index)
    w = w.clip(upper=cfg.max_position_pct)
    w = w / w.sum()  # renormalize the leg to 1
    leg["weight"] = sign * w
    return leg


def turnover(prev: pd.DataFrame | None, curr: pd.DataFrame) -> float:
    """One-way turnover between two books = 0.5 * Σ|w_curr - w_prev| (fraction of NAV)."""
    c = curr.set_index("symbol")["weight"]
    if prev is None or prev.empty:
        return float(c.abs().sum())  # initial build = full turnover
    p = prev.set_index("symbol")["weight"]
    all_syms = c.index.union(p.index)
    diff = c.reindex(all_syms, fill_value=0.0) - p.reindex(all_syms, fill_value=0.0)
    return float(0.5 * diff.abs().sum())
