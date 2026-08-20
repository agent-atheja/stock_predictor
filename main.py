"""Unified CLI for the Indian Stock Predictor. Self-contained — no external project deps.

Acquisition lives in MDS, not here. This project reads `silver.equity_ohlcv_adj`
and waits on `day_ready`; it makes no external calls of its own. The Kite and
bhavcopy fetchers that used to sit behind `kite-login`, `backfill`,
`kite-backfill`, `daily` and `silver` are in `archive/` — kept for reference,
deliberately not wired.

Examples:
  python main.py gold                           # silver (from MDS) → gold
  python main.py train [--hpo]                  # walk-forward train + validate
  python main.py backtest                       # cost-aware backtest
  python main.py signals                        # today's ranked signals
  python main.py synth                          # generate synthetic data (dev/verify)
  python main.py all                            # gold → train → backtest
"""
from __future__ import annotations

import argparse

from core.logging_setup import get_logger

log = get_logger("cli")


def main() -> None:
    ap = argparse.ArgumentParser(prog="stock-predictor")
    ap.add_argument("cmd", choices=[
        "gold", "train", "backtest", "signals", "refdata",
        "monitor", "paper", "synth", "all",
    ])
    ap.add_argument("extra", nargs="*")
    ap.add_argument("--hpo", action="store_true", help="enable Optuna HPO in training")
    args = ap.parse_args()

    if args.cmd == "refdata":
        from ingest.reference_data import build_pit_membership, fetch_corporate_actions, fetch_index_constituents
        fetch_index_constituents(); build_pit_membership(); fetch_corporate_actions()
    elif args.cmd == "gold":
        from features.build_gold import build_gold
        build_gold()
    elif args.cmd == "train":
        from models.train import run
        run(do_hpo=args.hpo)
    elif args.cmd == "backtest":
        from backtest.run import main as bt
        bt()
    elif args.cmd == "signals":
        from signals.generate_daily import main as sig
        sig()
    elif args.cmd == "monitor":
        from monitoring.drift import run as monitor
        monitor()
    elif args.cmd == "paper":
        from execution.paper_broker import rebalance
        rebalance()
    elif args.cmd == "synth":
        from tests.synthetic_data import write_bronze
        write_bronze()
    elif args.cmd == "all":
        # No build_silver step: Silver is MDS's, read through core.mds_source.
        from features.build_gold import build_gold
        from models.train import run as train
        from backtest.run import main as bt
        build_gold(); train(do_hpo=args.hpo); bt()


if __name__ == "__main__":
    main()
