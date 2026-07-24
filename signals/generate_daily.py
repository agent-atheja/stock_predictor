"""Daily ranked signals — score the latest tradable universe with the served model.

Produces a ranked watchlist for the most recent available Gold date and splits it into
long / short buckets (top_n / bottom_n from config). Written to registry as parquet + CSV.
This is the NEXT-phase serving path; it reuses the exact universe filter + feature set as
training, so a signal for date D equals what the backtest used for D (verifiable).

Run:  python -m signals.generate_daily
"""
from __future__ import annotations

import pandas as pd

from core.config import load_config, resolve
from core.logging_setup import get_logger, stage
from models.dataset import apply_universe_filter, load_gold, make_xy
from models.loader import load_final_model

log = get_logger(__name__)


def main() -> pd.DataFrame:
    cfg = load_config()
    with stage(log, "signals"):
        gold = load_gold()
        univ = apply_universe_filter(gold)
        as_of = univ["date"].max()
        today = univ[univ["date"] == as_of]
        X, _, meta = make_xy(today, require_label=False)
        model = load_final_model()
        meta = meta.assign(score=model.predict(X))
        ranked = meta.sort_values("score", ascending=False).reset_index(drop=True)
        ranked["rank"] = ranked.index + 1
        ranked["bucket"] = "hold"
        ranked.loc[ranked["rank"] <= cfg.backtest.top_n, "bucket"] = "long"
        if cfg.backtest.bottom_n:
            ranked.loc[ranked["rank"] > len(ranked) - cfg.backtest.bottom_n, "bucket"] = "short"

    out = resolve(cfg.model.registry_dir)
    ranked.to_parquet(out / "signals_latest.parquet", index=False)
    ranked[["rank", "symbol", "score", "bucket"]].to_csv(out / "signals_latest.csv", index=False)
    log.info("Signals for %s: %d longs / %d shorts (of %d names)",
             as_of.date(), (ranked["bucket"] == "long").sum(),
             (ranked["bucket"] == "short").sum(), len(ranked))
    log.info("Top 5 longs: %s", ", ".join(ranked.head(5)["symbol"]))
    return ranked


if __name__ == "__main__":
    main()
