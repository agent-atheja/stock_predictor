"""Walk-forward splitter with PURGE + EMBARGO — the core anti-leakage machinery.

For a horizon-h label, a training bar at date t has its outcome realized at t+h. If the test
window starts too soon after training, those maturing labels leak. We therefore:
  • PURGE: drop training bars whose [t, t+h] forward window overlaps the test window.
  • EMBARGO: additionally skip `embargo_days` between train end and test start (embargo > h).

Expanding scheme: train grows from the start; test rolls forward `test_step_days` at a time.
Yields (train_dates, test_dates) as boolean-maskable date ranges.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from core.config import load_config
from core.logging_setup import get_logger

log = get_logger(__name__)


@dataclass
class Fold:
    idx: int
    train_start: pd.Timestamp
    train_end: pd.Timestamp     # inclusive; already purged/embargoed vs test
    test_start: pd.Timestamp
    test_end: pd.Timestamp      # inclusive


def make_folds(all_dates: pd.Series) -> list[Fold]:
    """Build expanding/rolling walk-forward folds over the sorted unique trading dates."""
    cfg = load_config()
    wf = cfg.walkforward
    h = cfg.label.horizon_days
    dates = pd.Index(pd.to_datetime(pd.Series(all_dates).unique())).sort_values()
    embargo = wf.embargo_days
    if embargo <= h:
        log.warning("embargo_days(%d) ≤ horizon(%d) — bumping to horizon+1 to prevent leakage", embargo, h)
        embargo = h + 1

    folds: list[Fold] = []
    n = len(dates)
    start = 0
    train_min = wf.train_min_days
    step = wf.test_step_days
    i = 0
    test_lo = train_min + embargo
    while test_lo < n:
        test_hi = min(test_lo + step, n)
        # train end is embargo days before test start (purge maturing labels)
        train_end_idx = test_lo - embargo
        train_start_idx = start if wf.scheme == "expanding" else max(start, train_end_idx - train_min)
        if train_end_idx - train_start_idx < train_min // 2:
            test_lo = test_hi
            continue
        folds.append(
            Fold(
                idx=i,
                train_start=dates[train_start_idx],
                train_end=dates[train_end_idx - 1],
                test_start=dates[test_lo],
                test_end=dates[test_hi - 1],
            )
        )
        i += 1
        test_lo = test_hi
    log.info("Walk-forward: %d folds (scheme=%s, embargo=%d, step=%d)", len(folds), wf.scheme, embargo, step)
    return folds


def purge_train_mask(train_dates: pd.Series, test_start: pd.Timestamp, horizon: int) -> np.ndarray:
    """Extra safety: within the train slice, drop bars whose forward window reaches test_start.

    A bar at date t matures at ~t + horizon business days; if that ≥ test_start, purge it.
    """
    t = pd.to_datetime(train_dates)
    maturity = t + pd.tseries.offsets.BDay(horizon)
    return (maturity < test_start).to_numpy()
