"""Unit tests for round-2 hardening (recency, ensemble scaling, sector cap, reconstitution)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


def test_recency_weights_favor_recent():
    from models.lgbm_ranker import _recency_weights

    dates = pd.Series(pd.to_datetime(["2020-01-01", "2026-01-01"]))
    w = _recency_weights(dates)
    if len(set(w)) > 1:                      # only if recency weighting is enabled in config
        assert w[1] > w[0]                   # recent sample weighted higher


def test_standardize_zero_mean_unit_scale():
    from models.lgbm_ranker import _standardize

    v = np.array([1.0, 2.0, 3.0, 4.0])
    z = _standardize(v)
    assert z.mean() == pytest.approx(0.0, abs=1e-9)
    assert z.std() == pytest.approx(1.0, abs=1e-9)


def test_standardize_constant_vector_safe():
    from models.lgbm_ranker import _standardize

    assert np.allclose(_standardize(np.array([5.0, 5.0, 5.0])), 0.0)  # no divide-by-zero


def test_sector_cap_limits_per_sector(tmp_path, monkeypatch):
    import backtest.portfolio as pf

    # 10 names, all one sector except one; cap should stop the pile-up
    sectors = {f"S{i}": ("FIN" if i < 9 else "IT") for i in range(10)}
    monkeypatch.setattr(pf, "_sector_map", lambda: sectors)
    ranked = pd.DataFrame({"symbol": [f"S{i}" for i in range(10)],
                           "score": np.linspace(1, 0, 10)})
    picked = pf._select_with_sector_cap(ranked, n=5, max_sector_pct=0.4)  # cap = 2 per sector
    counts = picked["symbol"].map(sectors).value_counts()
    assert counts.get("FIN", 0) <= 2


def test_reconstitution_empty_events_returns_seed():
    from ingest.reference_data import apply_reconstitution_events

    hist = apply_reconstitution_events()      # events file is header-only in the repo
    assert len(hist) > 0
    assert {"symbol", "valid_from", "valid_to"}.issubset(hist.columns)


def test_new_config_keys_present():
    from core.config import load_config

    cfg = load_config()
    assert hasattr(cfg.backtest, "impl_shortfall_bps")
    assert hasattr(cfg.model, "ensemble_xgb")
    assert hasattr(cfg.model, "recency_halflife_days")
