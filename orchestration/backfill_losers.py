"""Auto-backfill the survivorship losers as soon as Kite recovers — idempotent, self-throttled.

The Wayback safe-subset staged ~110 delisted/demoted names into all_symbols_ever(), but they have
no prices yet, so the backtest universe is still survivor-only until they're backfilled. This job
backfills ONLY the missing names, then rebuilds Silver + Gold so is_member/features pick them up.

Safe to run on a timer:
  • If nothing is missing → exits 0 (and disables its own systemd timer if present).
  • If the Kite session/instruments endpoint is still cooled down → logs and exits 0 to retry later,
    never hammering (the instruments dump is disk-cached per day once it succeeds once).

Run:  python -m orchestration.backfill_losers
"""
from __future__ import annotations

import glob
import os

import pandas as pd

from core.config import load_config, resolve
from core.logging_setup import get_logger, stage
from ingest.constituents import all_symbols_ever

log = get_logger(__name__)


def _symbols_in_bronze() -> set[str]:
    root = resolve(load_config().data.bronze)
    have: set[str] = set()
    for p in glob.glob(str(root / "**" / "*.parquet"), recursive=True):
        try:
            have |= set(pd.read_parquet(p, columns=["symbol"])["symbol"].astype(str).str.upper())
        except Exception:  # noqa: BLE001
            continue
    return have


def _disable_own_timer() -> None:
    """Best-effort: stop retrying once the backfill is complete."""
    os.system("systemctl --user disable --now backfill-losers.timer >/dev/null 2>&1")


def main() -> int:
    with stage(log, "backfill-losers"):
        missing = sorted(set(all_symbols_ever()) - _symbols_in_bronze())
        if not missing:
            log.info("No missing symbols — survivorship backfill already complete. Standing down.")
            _disable_own_timer()
            return 0
        log.info("%d symbols missing from Bronze (survivorship losers): %s%s",
                 len(missing), ", ".join(missing[:15]), " …" if len(missing) > 15 else "")

        # Gentle Kite readiness check (profile call is not the rate-limited endpoint).
        from ingest.kite_client import verify
        if not verify():
            log.warning("Kite session not live yet — will retry on the next timer tick.")
            return 0

        # Probe the instruments endpoint ONCE before any batch. It has its own rate limit and can
        # be cooled down even when the session/profile works. One attempt disk-caches on success;
        # on failure we bail immediately so the batch never hammers it 110×.
        from ingest.kite_client import _instruments_nse
        try:
            _instruments_nse()
        except Exception as exc:  # noqa: BLE001
            log.warning("Kite instruments endpoint still cooled down (%s) — retry next tick.", exc)
            return 0

        # Backfill only the missing names. If the instruments dump is still cooled down, the first
        # call fails and backfill returns 0 rows — we detect that and retry later, no hammering.
        from ingest.kite_backfill import backfill
        try:
            n = backfill(symbols=missing)
        except Exception as exc:  # noqa: BLE001
            log.warning("Backfill attempt failed (%s) — retry next tick.", exc)
            return 0
        if n == 0:
            log.warning("Backfill returned 0 rows (Kite likely still cooled down) — retry next tick.")
            return 0

        # Propagate the new names into Silver (re-tags is_member) and Gold (features).
        from features.assembler import build_gold
        from ingest.bronze_to_silver import build_silver
        build_silver()
        build_gold()

        remaining = sorted(set(all_symbols_ever()) - _symbols_in_bronze())
        log.info("Backfill tick done: +%d rows. %d symbols still missing (delisted names Kite may "
                 "not serve).", n, len(remaining))
        if not remaining:
            _disable_own_timer()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
