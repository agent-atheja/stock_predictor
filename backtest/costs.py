"""Indian equity round-trip transaction-cost model (delivery/CNC).

Charges modeled (from config.backtest.costs), applied on traded notional:
  • brokerage        (both sides)
  • STT              (sell side, delivery)
  • exchange txn     (both sides)
  • GST              (on brokerage + exchange txn)
  • stamp duty       (buy side)
  • slippage         (both sides, by liquidity bucket)

Returned as a fraction of traded notional so the engine can charge cost = turnover_fraction × cost_rate.
This is what makes the backtest honest — a raw-return edge that dies after this model is not real.
"""
from __future__ import annotations

from core.config import load_config

BPS = 1e-4


def liquidity_bucket(turnover_cr_20d: float) -> str:
    if turnover_cr_20d >= 100:
        return "high"
    if turnover_cr_20d >= 20:
        return "mid"
    return "low"


def roundtrip_cost_rate(buy_liquidity: str = "high", sell_liquidity: str = "high") -> float:
    """Total round-trip cost as a fraction of notional traded (buy notional ≈ sell notional)."""
    c = load_config().backtest.costs
    gst = c.gst_pct_on_charges / 100.0

    # buy side
    buy = (
        c.brokerage_bps * BPS
        + c.exchange_txn_bps * BPS
        + c.stamp_duty_bps * BPS
        + c.slippage_bps_by_liquidity.__dict__[buy_liquidity] * BPS
    )
    buy += (c.brokerage_bps + c.exchange_txn_bps) * BPS * gst

    # sell side (STT here)
    sell = (
        c.brokerage_bps * BPS
        + c.exchange_txn_bps * BPS
        + c.stt_bps * BPS
        + c.slippage_bps_by_liquidity.__dict__[sell_liquidity] * BPS
    )
    sell += (c.brokerage_bps + c.exchange_txn_bps) * BPS * gst

    return buy + sell


def cost_for_turnover(turnover_fraction: float, liquidity: str = "high") -> float:
    """Cost charged to the portfolio for a given one-way turnover fraction of NAV."""
    # turnover_fraction is one-way (fraction of NAV bought OR sold); a rebalance does both.
    return turnover_fraction * roundtrip_cost_rate(liquidity, liquidity)
