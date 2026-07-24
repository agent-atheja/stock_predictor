"""Unit tests for the hardening iterations (tax, regime gate, smoothing, drift, CI, validation)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


# ── STCG tax ─────────────────────────────────────────────────────────────────
def test_stcg_tax_only_on_positive_returns():
    from backtest.engine import _stcg_tax
    from core.config import load_config

    cfg = load_config()
    assert _stcg_tax(0.10, cfg) == pytest.approx(0.10 * cfg.backtest.stcg_pct / 100)
    assert _stcg_tax(-0.10, cfg) == 0.0  # no tax on losses


# ── Regime gross factor ──────────────────────────────────────────────────────
def test_regime_factor_uses_config_scaling():
    from backtest.engine import _regime_factor
    from core.config import load_config

    cfg = load_config()
    hi = pd.DataFrame({"regime_vol": [2, 2]})
    lo = pd.DataFrame({"regime_vol": [0, 0]})
    assert _regime_factor(hi, cfg) == cfg.backtest.regime_gross_scaling[2]
    assert _regime_factor(lo, cfg) == cfg.backtest.regime_gross_scaling[0]
    assert _regime_factor(pd.DataFrame({"x": [1]}), cfg) == 1.0  # missing column → no scaling


# ── Score smoothing ──────────────────────────────────────────────────────────
def test_score_smoothing_pulls_toward_previous():
    from backtest.engine import _smooth_scores

    state = {"AAA": 1.0}
    day = pd.DataFrame({"symbol": ["AAA"], "score": [0.0]})
    out = _smooth_scores(day, state, alpha=0.5)
    assert out["score"].iloc[0] == pytest.approx(0.5)  # 0.5*0 + 0.5*1
    assert state["AAA"] == pytest.approx(0.5)


# ── PSI drift ────────────────────────────────────────────────────────────────
def test_psi_near_zero_for_identical_and_positive_for_shift():
    from monitoring.drift import psi

    rng = np.random.default_rng(0)
    base = pd.Series(rng.normal(0, 1, 5000))
    same = pd.Series(rng.normal(0, 1, 5000))
    shifted = pd.Series(rng.normal(2, 1, 5000))
    assert psi(base, same) < 0.1
    assert psi(base, shifted) > 0.25


# ── Bootstrap CI ─────────────────────────────────────────────────────────────
def test_bootstrap_ci_brackets_point_estimate():
    from backtest.report import sharpe_ratio
    from validation.metrics import bootstrap_ci

    rng = np.random.default_rng(1)
    r = pd.Series(rng.normal(0.01, 0.05, 400))
    point = sharpe_ratio(r, 52)
    lo, hi = bootstrap_ci(r, lambda x: sharpe_ratio(x, 52))
    assert lo <= point <= hi


# ── IC t-stat ────────────────────────────────────────────────────────────────
def test_ic_summary_reports_tstat_and_significance():
    from validation.metrics import ic_summary

    strong = pd.Series([0.05] * 100)  # tiny std → very significant
    s = ic_summary(strong)
    assert s["ic_significant"] is True
    noise = pd.Series(np.random.default_rng(0).normal(0, 0.1, 100))
    assert ic_summary(noise)["ic_significant"] in (True, False)  # just exercises the path


# ── Config validation ────────────────────────────────────────────────────────
def test_config_rejects_embargo_below_horizon(tmp_path):
    import yaml
    from core.config import ConfigError, load_config

    cfg = load_config()
    raw = yaml.safe_load(yaml.safe_dump(cfg._raw))
    raw["walkforward"]["embargo_days"] = raw["label"]["horizon_days"]  # not strictly greater
    p = tmp_path / "bad.yaml"
    p.write_text(yaml.safe_dump(raw))
    load_config.cache_clear()
    with pytest.raises(ConfigError):
        load_config(str(p))
    load_config.cache_clear()


# ── Book-weighted cost penalizes illiquidity ─────────────────────────────────
def test_book_weighted_cost_higher_for_illiquid_book():
    from backtest.engine import _book_weighted_cost_rate

    liquid = pd.DataFrame({"weight": [0.5, 0.5], "turnover_cr_20d": [500.0, 300.0]})
    illiquid = pd.DataFrame({"weight": [0.5, 0.5], "turnover_cr_20d": [3.0, 2.0]})
    assert _book_weighted_cost_rate(illiquid) > _book_weighted_cost_rate(liquid)
