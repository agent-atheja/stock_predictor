"""Cost-aware backtest engine.

Uses the OUT-OF-SAMPLE walk-forward predictions (never in-sample) and trades non-overlapping
h-day periods so the 5-day forward-return label maps cleanly to one holding period (no
double-counting from overlapping windows).

Per rebalance date:
  gross_period_ret = Σ weightᵢ · fwd_ret_5dᵢ
  cost             = turnover · roundtrip_cost_rate(liquidity)
  net_period_ret   = gross - cost
Equity compounds the net period returns. Reported against buy&hold and momentum baselines.
"""
from __future__ import annotations

import pandas as pd

from core.config import load_config
from core.io import read_dataset
from core.logging_setup import get_logger, stage
from backtest.costs import liquidity_bucket, roundtrip_cost_rate
from backtest.portfolio import build_book, turnover
from backtest.report import performance

log = get_logger(__name__)


def enrich(oos: pd.DataFrame) -> pd.DataFrame:
    """Attach vol_21d + turnover_cr_20d from Gold for weighting and cost liquidity buckets."""
    cfg = load_config()
    gold = read_dataset(cfg.data.gold, "features")[["date", "symbol", "vol_21d", "turnover_cr_20d", "regime_vol"]]
    gold["date"] = pd.to_datetime(gold["date"])
    oos = oos.copy()
    oos["date"] = pd.to_datetime(oos["date"])
    return oos.merge(gold, on=["date", "symbol"], how="left")


def _rebalance_dates(dates: pd.Series, horizon: int) -> list[pd.Timestamp]:
    """Non-overlapping rebalance dates spaced `horizon` trading days apart."""
    uniq = pd.Index(sorted(pd.to_datetime(dates.unique())))
    return list(uniq[::horizon])


def run_strategy(oos: pd.DataFrame, score_col: str = "score", long_only: bool = False) -> tuple[pd.Series, dict]:
    """Backtest the ranked strategy. long_only=True drops the (often unshortable in India) short
    leg and holds only the top book — the realistically deployable variant."""
    cfg = load_config()
    h = cfg.label.horizon_days
    oos = oos.rename(columns={score_col: "score"}) if score_col != "score" else oos
    rebs = _rebalance_dates(oos["date"], h)

    if "turnover_cr_20d" not in oos.columns:
        oos = oos.assign(turnover_cr_20d=float("nan"))

    prev_book = None
    smoothed: dict[str, float] = {}
    alpha = float(getattr(cfg.backtest, "score_smoothing_alpha", 1.0))
    records = []
    for dt in rebs:
        day = oos[oos["date"] == dt]
        if day.empty:
            continue
        if alpha < 1.0:
            day = _smooth_scores(day, smoothed, alpha)   # EWMA-smooth to reduce turnover
        book = build_book(day)
        if long_only:
            book = book[book["weight"] > 0].copy()
            if not book.empty:
                book["weight"] /= book["weight"].sum()   # renormalize the long leg to 1.0
        if book.empty:
            continue
        merged = book.merge(day[["symbol", "fwd_ret_5d", "turnover_cr_20d"]], on="symbol", how="left")
        gross = float((merged["weight"] * merged["fwd_ret_5d"].fillna(0)).sum())
        tno = turnover(prev_book, book)
        # regime de-grossing: scale exposure (and thus return + turnover) by the day's regime factor
        factor = _regime_factor(day, cfg)
        gross *= factor
        tno *= factor
        # book-weighted cost rate + implementation-shortfall haircut (close-t signal, t+1 fill)
        impl_bps = getattr(cfg.backtest, "impl_shortfall_bps", 0) * 1e-4
        cost = tno * (_book_weighted_cost_rate(merged) + impl_bps)
        net_pretax = gross - cost
        tax = _stcg_tax(net_pretax, cfg)
        records.append({
            "date": dt, "gross": gross, "cost": cost,
            "net_pretax": net_pretax, "tax": tax, "net": net_pretax - tax, "turnover": tno,
        })
        prev_book = book

    curve = pd.DataFrame(records).set_index("date")
    net = curve["net"]                      # headline series is AFTER cost AND tax
    ppy = 252 / h
    perf = performance(net, ppy)
    perf["avg_turnover"] = float(curve["turnover"].mean()) if not curve.empty else 0.0
    perf["gross_sharpe"] = performance(curve["gross"], ppy).get("sharpe")
    perf["pretax_sharpe"] = performance(curve["net_pretax"], ppy).get("sharpe")
    perf["total_tax_drag"] = float(curve["tax"].sum())
    return net, perf


def _book_weighted_cost_rate(merged: pd.DataFrame) -> float:
    """Round-trip cost rate weighted by |position|, using each name's own liquidity bucket —
    so an illiquidity tilt pays the higher slippage it actually incurs."""
    w = merged["weight"].abs()
    if w.sum() == 0:
        return roundtrip_cost_rate("high", "high")
    buckets = merged["turnover_cr_20d"].apply(
        lambda t: liquidity_bucket(t) if pd.notna(t) else "high"
    )
    rates = buckets.apply(lambda b: roundtrip_cost_rate(b, b))
    return float((w * rates).sum() / w.sum())


def _smooth_scores(day: pd.DataFrame, state: dict[str, float], alpha: float) -> pd.DataFrame:
    """EWMA-smooth each symbol's score against its prior smoothed value; updates state in place."""
    day = day.copy()
    new_scores = []
    for sym, raw in zip(day["symbol"], day["score"]):
        prev = state.get(sym, raw)
        val = alpha * raw + (1 - alpha) * prev
        state[sym] = val
        new_scores.append(val)
    day["score"] = new_scores
    return day


def _regime_factor(day: pd.DataFrame, cfg) -> float:
    """Gross-exposure multiplier for the day's volatility regime (1.0 if regime info absent)."""
    scaling = getattr(cfg.backtest, "regime_gross_scaling", None)
    if scaling is None or "regime_vol" not in day.columns:
        return 1.0
    r = day["regime_vol"].iloc[0]
    if pd.isna(r):
        return 1.0
    r = int(r)
    return float(scaling[r]) if 0 <= r < len(scaling) else 1.0


def _stcg_tax(period_return: float, cfg) -> float:
    """Short-term capital gains on a positive period return. 5-day holds are always STCG.
    Tax is portfolio-level (not a per-trade cost), so it reads from backtest.* not backtest.costs.*."""
    bt = cfg.backtest
    if not getattr(bt, "apply_stcg_tax", False):
        return 0.0
    return max(period_return, 0.0) * (bt.stcg_pct / 100.0)


def run_baselines(oos: pd.DataFrame) -> dict:
    """Baselines to beat, on the same dates/universe: equal-weight (buy&hold) and 12-1 momentum."""
    cfg = load_config()
    gold = read_dataset(cfg.data.gold, "features")[["date", "symbol", "mom_12_1"]]
    gold["date"] = pd.to_datetime(gold["date"])
    base = oos.merge(gold, on=["date", "symbol"], how="left")
    h = cfg.label.horizon_days
    ppy = 252 / h

    # equal-weight universe (market proxy / buy&hold)
    ew = base.groupby("date")["fwd_ret_5d"].mean()
    ew = ew.iloc[::h]
    ew_perf = performance(ew, ppy)

    # momentum long-short using the same portfolio construction.
    # Build a clean frame whose `score` IS the momentum signal (avoid colliding with the
    # model's score column already present on `base`).
    mom = base[["date", "symbol", "fwd_ret_5d", "mom_12_1"]].rename(columns={"mom_12_1": "score"})
    _, mom_perf = run_strategy(mom)
    return {"equal_weight": ew_perf, "momentum_12_1": mom_perf}
