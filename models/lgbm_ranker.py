"""LightGBM cross-sectional model.

Two objectives, selectable via config.model.type:
  • lgbm_ranker : LambdaRank with per-day query groups; relevance = per-day decile of y (0..9).
                  Optimizes the ordering within each day — exactly our cross-sectional framing.
  • lgbm_reg    : regression on the z-scored forward return; the raw prediction is the score.

Seed-averaged (config.model.n_ensemble_seeds) for stability — single-seed GBMs on noisy
financial data are jittery. Returns a small wrapper exposing .predict(X) → per-row score.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import lightgbm as lgb

from core.config import load_config
from core.logging_setup import get_logger
from models.dataset import day_groups

log = get_logger(__name__)

_BASE_PARAMS = dict(
    n_estimators=600,
    learning_rate=0.03,
    num_leaves=31,
    min_child_samples=80,     # regularize — noisy labels overfit easily
    subsample=0.8,
    subsample_freq=1,
    colsample_bytree=0.8,
    reg_lambda=5.0,
    reg_alpha=1.0,
    max_depth=-1,
    n_jobs=-1,                # use all cores (24 threads on this host)
    verbosity=-1,
)


def _deciles_per_day(y: pd.Series, dates: pd.Series) -> np.ndarray:
    """Relevance grades for LambdaRank: per-day decile rank of y (higher return → higher grade)."""
    df = pd.DataFrame({"y": y.values, "d": dates.values})
    grade = df.groupby("d")["y"].transform(
        lambda s: pd.qcut(s.rank(method="first"), q=min(10, max(2, s.notna().sum())), labels=False, duplicates="drop")
    )
    return grade.fillna(0).astype(int).to_numpy()


def _standardize(v: np.ndarray) -> np.ndarray:
    """Z-score a prediction vector so heterogeneous models (LGB vs XGB, ranker vs reg) combine on
    a common scale before averaging — a raw mean would let the larger-scale model dominate."""
    s = v.std()
    return (v - v.mean()) / s if s > 0 else v - v.mean()


class CrossSectionalModel:
    def __init__(self, models: list, params: dict, mode: str):
        self.models = models
        self.params = params
        self.mode = mode

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        # standardize each member's output, then average — robust to scale differences
        preds = np.column_stack([_standardize(m.predict(X)) for m in self.models])
        return preds.mean(axis=1)

    @property
    def feature_importance(self) -> pd.Series:
        imp = np.mean([m.feature_importances_ for m in self.models], axis=0)
        return pd.Series(imp, index=self.models[0].feature_name_).sort_values(ascending=False)


def _recency_weights(dates: pd.Series) -> np.ndarray:
    """Exponential-decay sample weights by age so recent regimes count more. Half-life in days
    from config (0/None → uniform weights)."""
    hl = getattr(load_config().model, "recency_halflife_days", 0)
    if not hl:
        return np.ones(len(dates))
    age_days = (dates.max() - pd.to_datetime(dates)).dt.days.to_numpy()
    return np.power(0.5, age_days / float(hl))


def train_model(
    X: pd.DataFrame,
    y: pd.Series,
    dates: pd.Series,
    params: dict | None = None,
) -> CrossSectionalModel:
    cfg = load_config().model
    mode = cfg.type
    p = {**_BASE_PARAMS, **(params or {})}
    seeds = list(range(cfg.n_ensemble_seeds))
    w = _recency_weights(dates)
    models = []

    if mode == "lgbm_ranker":
        groups = day_groups(dates)
        relevance = _deciles_per_day(y, dates)
        for s in seeds:
            m = lgb.LGBMRanker(objective="lambdarank", random_state=s, **p)
            m.fit(X, relevance, group=groups, sample_weight=w)
            models.append(m)
    else:  # lgbm_reg
        for s in seeds:
            m = lgb.LGBMRegressor(objective="regression", random_state=s, **p)
            m.fit(X, y, sample_weight=w)
            models.append(m)

    if getattr(cfg, "ensemble_xgb", False):
        models.append(_fit_xgb(X, y, w))

    return CrossSectionalModel(models, p, mode)


def _fit_xgb(X: pd.DataFrame, y: pd.Series, weights: np.ndarray):
    """XGBoost regression member on the z-scored return target — diversifies the LGB ranker."""
    import xgboost as xgb

    m = xgb.XGBRegressor(
        n_estimators=600, learning_rate=0.03, max_depth=5, subsample=0.8,
        colsample_bytree=0.8, reg_lambda=5.0, n_jobs=-1, random_state=0,
    )
    m.fit(X, y, sample_weight=weights)
    return m
