"""Bronze → Silver: adjust, clean, type, PIT-tag. Idempotent, incremental-friendly.

Steps:
  1. Load Bronze OHLCV (raw).
  2. Split/bonus back-adjustment via a corporate-actions file (cumulative factor). Kite is
     adjusted-at-source, but jugaad bhavcopy is NOT — so we always apply adjustment in Silver
     and cross-check. Absent CA file → pass-through with a loud warning.
  3. Dedup, type, drop non-positive-price rows.
  4. Data-quality assertions (hard checks raise).
  5. Add liquidity fields (20d turnover) used by the universe filter.
  6. PIT membership flag so downstream can filter to the tradable set per day.

Run:  python -m ingest.bronze_to_silver
"""
from __future__ import annotations

import pandas as pd

from core.config import load_config, resolve
from core.io import read_dataset, write_partitioned
from core.logging_setup import get_logger, stage
from core.quality import check_ohlcv
from ingest.constituents import load_membership

log = get_logger(__name__)


def _load_corporate_actions() -> pd.DataFrame:
    """CA file schema: symbol,ex_date,ratio  (ratio = old_shares/new_shares equivalent price factor).

    For a 1:2 split (price halves) ratio=0.5; for 1:1 bonus ratio=0.5; dividends ignored here.
    """
    path = resolve("config/corporate_actions.csv")
    if not path.exists():
        log.warning(
            "No corporate_actions.csv — Silver prices are UNADJUSTED. Splits/bonuses will show "
            "as spurious jumps. Supply the file for correct backtests."
        )
        return pd.DataFrame(columns=["symbol", "ex_date", "ratio"])
    ca = pd.read_csv(path)
    ca["ex_date"] = pd.to_datetime(ca["ex_date"])
    return ca


def _apply_adjustment(df: pd.DataFrame, ca: pd.DataFrame) -> pd.DataFrame:
    """Back-adjust OHLC by cumulative factor of all CAs strictly AFTER each bar's date.

    Point-in-time note: adjustment uses only actions dated after the bar, so a bar's adjusted
    value is stable as new history arrives (no future leakage into the *relative* returns we model).
    """
    if ca.empty:
        df["adj_close"] = df["close"]
        for c in ("open", "high", "low"):
            df[f"adj_{c}"] = df[c]
        return df

    df = df.sort_values(["symbol", "date"]).copy()
    for c in ("open", "high", "low", "close"):
        df[f"adj_{c}"] = df[c].astype(float)

    for sym, actions in ca.groupby("symbol"):
        sym_mask = df["symbol"] == sym
        if not sym_mask.any():
            continue
        for _, act in actions.iterrows():
            # bars strictly before ex_date get multiplied by the ratio (cumulative)
            m = sym_mask & (df["date"] < act["ex_date"])
            for c in ("open", "high", "low", "close"):
                df.loc[m, f"adj_{c}"] *= act["ratio"]
    return df


def build_silver() -> int:
    cfg = load_config()
    with stage(log, "silver"):
        bronze = read_dataset(cfg.data.bronze, "equity_ohlcv")
        if bronze.empty:
            log.error("Bronze is empty — run ingest.historical_backfill first.")
            return 0

        # 1. clean & type
        bronze["date"] = pd.to_datetime(bronze["date"])
        bronze = bronze.drop_duplicates(subset=["symbol", "date"])
        bronze = bronze[bronze["close"] > 0].sort_values(["symbol", "date"])

        # 2. adjust — but ONLY if the source is raw. Kite is adjusted-at-source; re-applying
        # the corporate-actions file on top would double-adjust pre-split bars.
        if getattr(cfg.data, "source_adjusted", False):
            log.info("Source is pre-adjusted (e.g. Kite) — skipping corporate-action adjustment.")
            adj = bronze.copy()
            for c in ("open", "high", "low", "close"):
                adj[f"adj_{c}"] = adj[c].astype(float)
        else:
            ca = _load_corporate_actions()
            adj = _apply_adjustment(bronze, ca)

        # 3. liquidity (20d avg turnover in ₹ crore) — used by universe filter
        adj["turnover"] = adj["adj_close"] * adj["volume"]
        adj["turnover_cr_20d"] = (
            adj.groupby("symbol")["turnover"].transform(lambda s: s.rolling(20, min_periods=5).mean()) / 1e7
        )

        # 4. PIT membership flag
        mem = load_membership()
        adj = _tag_membership(adj, mem)

        # 5. quality gate (hard checks raise)
        check_ohlcv(adj.rename(columns={}), stage="silver").raise_if_failed()

        n = write_partitioned(adj, cfg.data.silver, "equity_ohlcv_adj", date_col="date")
    log.info("Silver built: %d rows, %d symbols", n, adj["symbol"].nunique())
    return n


def _tag_membership(df: pd.DataFrame, mem: pd.DataFrame) -> pd.DataFrame:
    """Add boolean `is_member` = symbol was an index constituent on that bar's date.
    Vectorized interval join (O(n·intervals_per_symbol)) instead of a per-interval full scan."""
    df = df.copy()
    joined = df.reset_index().merge(mem, on="symbol", how="left")
    hit = (joined["date"] >= joined["valid_from"]) & (joined["date"] < joined["valid_to"])
    member_by_row = hit.groupby(joined["index"]).any()
    df["is_member"] = df.index.map(member_by_row).fillna(False).astype(bool)
    return df


if __name__ == "__main__":
    build_silver()
