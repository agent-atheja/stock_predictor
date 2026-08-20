"""Feature assembler — Silver → Gold.

Pipeline:
  1. Load Silver (adjusted OHLCV panel).
  2. Per-symbol technicals in PARALLEL (CPU-bound → process pool). Each symbol is independent,
     so this fans out cleanly across cores.
  3. Cross-sectional market-structure + regime features (need the whole panel per day).
  4. Attach the 5-day forward-return label.
  5. Leakage guard: assert no feature column correlates ~1.0 with the forward label by
     construction, and that the label is the only future-looking column.
  6. Write Gold (one row per (date, symbol)).

Run:  python -m features.build_gold
"""
from __future__ import annotations

import pandas as pd

from core.config import load_config
from core.io import read_dataset, write_partitioned
from core.logging_setup import get_logger, stage
from core.parallel import cpu_map
from features.market_structure import add_market_structure
from features.regime import add_regime
from features.technicals import add_technicals
from labels.forward_return import add_labels

log = get_logger(__name__)

# feature columns produced (kept explicit so the model never accidentally trains on a leak)
FEATURE_COLS = [
    "ret_1d", "ret_5d", "ret_10d", "ret_21d", "ret_63d", "mom_12_1",
    "ema_ratio_12_26", "sma_ratio_20_50", "macd_hist", "dist_52w_high",
    "rsi", "bb_pctb", "zscore_20d",
    "vol_10d", "vol_21d", "vol_63d", "atr_pct", "vol_of_vol",
    "vol_z_20d", "turnover_z_20d", "amihud_21d",
    "mkt_ret_1d", "rel_ret_1d", "beta_63d",
    "xs_rank_mom_12_1", "xs_rank_ret_21d", "xs_rank_vol_21d",
    "regime_vol",
]


def _tech_for_symbol(sym_df: pd.DataFrame) -> pd.DataFrame:
    """Top-level (picklable) worker for the process pool: technicals for one symbol."""
    return add_technicals(sym_df)


def build_gold() -> int:
    cfg = load_config()
    with stage(log, "gold"):
        # Silver comes from MDS when USE_MDS_SILVER=1, otherwise from the local
        # Parquet lake exactly as before. Pinned to config history_start so the
        # training window is identical either way — MDS carries history back to
        # 1996 and silently widening it would confound any before/after backtest
        # comparison with a data-quantity change.
        from core.mds_source import is_enabled as _mds_enabled, read_silver as _mds_read

        if _mds_enabled():
            silver = _mds_read(start=str(cfg.data.history_start))
            log.info("Silver sourced from MDS (USE_MDS_SILVER=1)")
        else:
            silver = read_dataset(cfg.data.silver, "equity_ohlcv_adj")

        if silver.empty:
            log.error("Silver is empty — check MDS: python -m mds.cli status")
            return 0
        silver["date"] = pd.to_datetime(silver["date"])

        # 2. per-symbol technicals in parallel (CPU-bound)
        groups = [g for _, g in silver.groupby("symbol")]
        log.info("Building technicals for %d symbols across process pool", len(groups))
        tech_frames = cpu_map(_tech_for_symbol, groups, desc="technicals")
        panel = pd.concat([f for f in tech_frames if f is not None], ignore_index=True)

        # 3. cross-sectional structure + regime
        panel = add_market_structure(panel)
        panel = add_regime(panel)

        # 4. label
        panel = add_labels(panel)

        # 5. leakage guard
        _assert_no_leakage(panel)

        # 6. keep tradable, drop warm-up NaNs on core features, write
        keep = ["date", "symbol", "adj_close", "is_member", "turnover_cr_20d",
                "y", "fwd_ret_5d", "fwd_up_5d"] + FEATURE_COLS
        gold = panel[keep].copy()
        gold = gold.dropna(subset=["ret_63d", "vol_63d", "beta_63d"])  # feature warm-up
        n = write_partitioned(gold, cfg.data.gold, "features", date_col="date")
    log.info("Gold built: %d rows, %d features, %d symbols", n, len(FEATURE_COLS), gold["symbol"].nunique())
    return n


def _assert_no_leakage(panel: pd.DataFrame) -> None:
    """Guard: no feature may be almost-perfectly correlated with the forward label (a leak),
    and features must be finite. Cheap tripwire that catches accidental future joins."""
    sample = panel.dropna(subset=["fwd_ret_5d"]).tail(20000)
    for col in FEATURE_COLS:
        if col not in sample or sample[col].std(skipna=True) == 0:
            continue
        corr = sample[col].corr(sample["fwd_ret_5d"])
        if pd.notna(corr) and abs(corr) > 0.95:
            raise AssertionError(
                f"Leakage tripwire: feature '{col}' corr={corr:.3f} with forward label — "
                "likely a future-looking join. Fix before training."
            )
    log.info("Leakage tripwire passed (%d features checked)", len(FEATURE_COLS))


if __name__ == "__main__":
    build_gold()
