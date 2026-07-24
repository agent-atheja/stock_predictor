"""Sliced evaluation — disaggregate IC by fold and by market regime.

Aggregate metrics hide regime-specific failures (e.g. the model losing money in high-vol
crashes). These slices are mandatory eval hygiene: a stable edge should hold across folds and
not collapse in any single regime.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from core.config import load_config
from core.io import read_dataset
from validation.metrics import daily_rank_ic


def ic_by_fold(oos: pd.DataFrame) -> dict:
    """Per-fold mean IC + cross-fold stability (mean/std). Low std = a dependable edge."""
    if "fold" not in oos.columns:
        return {}
    per_fold = {}
    for fold, g in oos.groupby("fold"):
        ic = daily_rank_ic(g["score"], g["fwd_ret_5d"], g["date"])
        per_fold[int(fold)] = float(ic.mean()) if len(ic) else np.nan
    vals = np.array([v for v in per_fold.values() if not np.isnan(v)])
    return {
        "per_fold_mean_ic": per_fold,
        "n_folds": len(vals),
        "pct_folds_positive": float((vals > 0).mean()) if len(vals) else np.nan,
        "fold_ic_stability": float(vals.mean() / vals.std()) if len(vals) > 1 and vals.std() > 0 else np.nan,
    }


def ic_by_regime(oos: pd.DataFrame) -> dict:
    """Mean IC within each volatility regime (0=low,1=mid,2=high) from the Gold regime flag."""
    cfg = load_config()
    regime = read_dataset(cfg.data.gold, "features")[["date", "symbol", "regime_vol"]]
    regime["date"] = pd.to_datetime(regime["date"])
    m = oos.merge(regime, on=["date", "symbol"], how="left")
    out = {}
    for r, g in m.groupby("regime_vol"):
        ic = daily_rank_ic(g["score"], g["fwd_ret_5d"], g["date"])
        out[int(r)] = {"mean_ic": float(ic.mean()) if len(ic) else np.nan, "n_days": int(len(ic))}
    return out
