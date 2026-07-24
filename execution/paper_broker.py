"""Paper-trading broker — the deployment dry-run before any real capital.

Stateful simulation that runs once per rebalance:
  1. Mark existing positions to the latest prices → update NAV.
  2. Kill switch: if drawdown from peak exceeds the limit, go flat and stop.
  3. Build the target long-only, sector-capped book from the model's latest signals.
  4. Apply the regime de-gross factor, generate orders, charge round-trip costs.
  5. Persist positions + NAV history so P&L can be reconciled against the backtest over time.

Deliberately long-only (deployable without shorting) and reuses the SAME portfolio construction
as the backtest, so paper P&L should track the backtest for the same dates (a key validation).

Run:  python -m execution.paper_broker            # one rebalance step (idempotent per date)
"""
from __future__ import annotations

import json
from datetime import date

import numpy as np
import pandas as pd

from backtest.costs import liquidity_bucket, roundtrip_cost_rate
from backtest.portfolio import build_book
from core.config import load_config, resolve
from core.logging_setup import get_logger, stage
from models.dataset import apply_universe_filter, load_gold, make_xy
from models.loader import load_final_model

log = get_logger(__name__)

STARTING_CAPITAL = 1_000_000.0   # ₹10L notional paper book
KILL_SWITCH_DD = 0.20            # go flat if drawdown from peak exceeds 20%


def _state_path():
    return resolve(load_config().model.registry_dir) / "paper_portfolio.json"


def _load_state() -> dict:
    p = _state_path()
    if p.exists():
        return json.loads(p.read_text())
    return {"nav": STARTING_CAPITAL, "peak_nav": STARTING_CAPITAL, "cash": STARTING_CAPITAL,
            "positions": {}, "last_prices": {}, "history": [], "halted": False}


def _save_state(state: dict) -> None:
    _state_path().write_text(json.dumps(state, indent=2, default=str))


def _score_universe() -> pd.DataFrame:
    """Score the latest tradable universe with the served model → [date, symbol, score, price]."""
    gold = load_gold()
    univ = apply_universe_filter(gold)
    as_of = univ["date"].max()
    today = univ[univ["date"] == as_of].copy()
    X, _, meta = make_xy(today, require_label=False)
    meta = meta.assign(score=load_final_model().predict(X))
    prices = today.set_index("symbol")["adj_close"]
    meta["price"] = meta["symbol"].map(prices)
    meta["turnover_cr_20d"] = meta["symbol"].map(today.set_index("symbol")["turnover_cr_20d"])
    meta["vol_21d"] = meta["symbol"].map(today.set_index("symbol")["vol_21d"])
    meta["regime_vol"] = today["regime_vol"].iloc[0]
    return meta.assign(as_of=as_of)


def _mark_to_market(state: dict, prices: dict[str, float]) -> None:
    """Update each position's value by its price change since the last mark; refresh NAV."""
    pos_value = 0.0
    for sym, val in list(state["positions"].items()):
        last = state["last_prices"].get(sym)
        cur = prices.get(sym)
        if last and cur and last > 0:
            state["positions"][sym] = val * (cur / last)
        pos_value += state["positions"][sym]
    state["nav"] = state["cash"] + pos_value
    state["peak_nav"] = max(state["peak_nav"], state["nav"])


def rebalance() -> dict:
    cfg = load_config()
    with stage(log, "paper-rebalance"):
        state = _load_state()
        scored = _score_universe()
        as_of = str(pd.Timestamp(scored["as_of"].iloc[0]).date())
        prices = dict(zip(scored["symbol"], scored["price"]))

        if state["history"] and state["history"][-1]["date"] == as_of:
            log.info("Already rebalanced for %s — idempotent no-op.", as_of)
            return state

        _mark_to_market(state, prices)

        drawdown = state["nav"] / state["peak_nav"] - 1
        if drawdown <= -KILL_SWITCH_DD:
            state["halted"] = True
        target = {} if state["halted"] else _target_book(scored, state["nav"], cfg)
        if state["halted"]:
            log.warning("KILL SWITCH active (drawdown %.1f%%) — going flat.", drawdown * 100)

        cost = _apply_orders(state, target, scored, prices)
        state["last_prices"] = {**state["last_prices"], **prices}
        state["history"].append({"date": as_of, "nav": round(state["nav"], 2),
                                 "drawdown": round(drawdown, 4), "cost": round(cost, 2),
                                 "n_positions": len(state["positions"]), "halted": state["halted"]})
        _save_state(state)
        log.info("Paper NAV %s: ₹%.0f (dd %.1f%%, %d positions, cost ₹%.0f)",
                 as_of, state["nav"], drawdown * 100, len(state["positions"]), cost)
    return state


def _target_book(scored: pd.DataFrame, nav: float, cfg) -> dict[str, float]:
    """Long-only sector-capped book (rupee value per symbol), scaled by the regime factor."""
    book = build_book(scored.rename(columns={"score": "score"}))
    book = book[book["weight"] > 0]
    if book.empty:
        return {}
    book["weight"] /= book["weight"].sum()
    regime = int(scored["regime_vol"].iloc[0]) if pd.notna(scored["regime_vol"].iloc[0]) else 1
    scaling = getattr(cfg.backtest, "regime_gross_scaling", [1, 1, 1])
    factor = scaling[regime] if 0 <= regime < len(scaling) else 1.0
    return {row["symbol"]: row["weight"] * nav * factor for _, row in book.iterrows()}


def _apply_orders(state: dict, target: dict[str, float], scored: pd.DataFrame, prices: dict) -> float:
    """Move positions to target, charging per-name round-trip cost on traded value."""
    liq = dict(zip(scored["symbol"], scored["turnover_cr_20d"]))
    current = state["positions"]
    symbols = set(current) | set(target)
    total_cost = 0.0
    for sym in symbols:
        traded = abs(target.get(sym, 0.0) - current.get(sym, 0.0))
        if traded <= 0:
            continue
        bucket = liquidity_bucket(liq.get(sym, 100.0) if pd.notna(liq.get(sym, np.nan)) else 100.0)
        total_cost += traded * roundtrip_cost_rate(bucket, bucket)
    state["positions"] = {s: v for s, v in target.items() if v > 0}
    invested = sum(state["positions"].values())
    state["cash"] = state["nav"] - invested - total_cost
    state["nav"] -= total_cost
    return total_cost


def status() -> dict:
    """Print a compact paper-book readout (NAV, return, drawdown, top holdings, recent history).
    Read-only — safe to run any time, including in monitoring/cron."""
    state = _load_state()
    nav, peak = state["nav"], state["peak_nav"]
    ret = nav / STARTING_CAPITAL - 1
    dd = nav / peak - 1
    hist = state["history"]
    last = hist[-1]["date"] if hist else "—"
    print(f"\n== Paper book ==")
    print(f"  as of         : {last}   ({len(hist)} rebalance(s))")
    print(f"  NAV           : ₹{nav:,.0f}   (start ₹{STARTING_CAPITAL:,.0f})")
    print(f"  total return  : {ret:+.2%}")
    print(f"  drawdown      : {dd:+.2%}   (peak ₹{peak:,.0f})")
    print(f"  kill switch   : {'HALTED' if state.get('halted') else 'armed'} "
          f"(trips at {-KILL_SWITCH_DD:.0%})")
    print(f"  positions     : {len(state['positions'])}")
    if state["positions"]:
        top = sorted(state["positions"].items(), key=lambda kv: -kv[1])[:5]
        for sym, val in top:
            print(f"      {sym:12s} ₹{val:,.0f}  ({val/nav:.1%})")
    if hist:
        print("  recent NAV    :")
        for h in hist[-5:]:
            print(f"      {h['date']}  ₹{h['nav']:,.0f}  dd {h['drawdown']:+.2%}  "
                  f"cost ₹{h['cost']:,.0f}  n={h['n_positions']}")
    print()
    return state


if __name__ == "__main__":
    import sys
    cmd = sys.argv[1] if len(sys.argv) > 1 else "rebalance"
    if cmd == "status":
        status()
    else:
        rebalance()
