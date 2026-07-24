"""Unit tests for the leakage-critical and correctness-critical logic.

Run:  python -m pytest tests/test_pipeline.py -q
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from core.config import load_config


# ── Walk-forward embargo: no train bar within embargo of test start ──────────────
def test_embargo_gap_exceeds_horizon():
    from validation.walkforward import make_folds

    dates = pd.Series(pd.bdate_range("2016-01-01", "2022-01-01"))
    folds = make_folds(dates)
    cfg = load_config()
    embargo = max(cfg.walkforward.embargo_days, cfg.label.horizon_days + 1)
    assert folds, "expected at least one fold"
    for f in folds:
        gap_days = np.busday_count(f.train_end.date(), f.test_start.date())
        assert gap_days >= embargo, f"fold {f.idx}: gap {gap_days} < embargo {embargo}"


def test_purge_removes_maturing_labels():
    from validation.walkforward import purge_train_mask

    train_dates = pd.Series(pd.bdate_range("2020-01-01", "2020-02-01"))
    test_start = train_dates.iloc[-1]  # last train day matures AFTER test start
    mask = purge_train_mask(train_dates, test_start, horizon=5)
    # the final ~5 business days before test_start must be purged
    assert not mask[-1], "bar maturing into the test window should be purged"
    assert mask[0], "an early bar should be kept"


# ── Label correctness: forward return points to the right future bar ─────────────
def test_forward_return_label():
    from labels.forward_return import add_labels

    n = 40
    df = pd.DataFrame({
        "symbol": "AAA",
        "date": pd.bdate_range("2021-01-01", periods=n),
        "adj_close": np.linspace(100, 139, n),
    })
    out = add_labels(df.assign(fwd_ret_5d=np.nan)).sort_values("date").reset_index(drop=True)
    h = load_config().label.horizon_days
    expected = df["adj_close"].iloc[h] / df["adj_close"].iloc[0] - 1
    assert out["fwd_ret_5d"].iloc[0] == pytest.approx(expected, rel=1e-9)
    # last h rows have no future → NaN
    assert out["fwd_ret_5d"].iloc[-1] != out["fwd_ret_5d"].iloc[-1]  # NaN


# ── Cost model: round-trip cost is positive and ordered by liquidity ─────────────
def test_cost_model_monotonic_in_liquidity():
    from backtest.costs import roundtrip_cost_rate

    high = roundtrip_cost_rate("high", "high")
    low = roundtrip_cost_rate("low", "low")
    assert 0 < high < low, "low-liquidity round-trip must cost more than high-liquidity"


# ── Parallel helpers: order preserved, resilient to failures ─────────────────────
def _square(x):
    return x * x


def test_cpu_map_preserves_order():
    from core.parallel import cpu_map

    items = list(range(20))
    assert cpu_map(_square, items, workers=4, desc="t") == [i * i for i in items]


def test_io_map_handles_failure_gracefully():
    from core.parallel import io_map

    def flaky(x):
        if x == 3:
            raise ValueError("boom")
        return x

    res = io_map(flaky, list(range(6)), workers=3, rate_limit_per_sec=0, desc="t")
    assert res[3] is None and res[0] == 0 and res[5] == 5
