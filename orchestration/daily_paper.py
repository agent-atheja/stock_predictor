"""Daily paper-trading driver — the MVP operational loop.

Runs once per trading day (cron/systemd-timer). Chains: refresh data → rebuild silver/gold →
paper rebalance → drift check → status. Designed to be SAFE to run unattended:

  • If Kite is not authenticated, it SKIPS the data refresh and still rebalances on the latest
    available Gold (idempotent — a no-op if already done for that date). No crash, loud warning.
  • Every stage is wrapped so one failure is logged and does not abort the rest where safe.

This is the deployment dry-run before live capital: no real orders are ever placed here.

Cron (weekdays ~18:30 IST, after NSE close + EOD data settles):
    30 18 * * 1-5  cd /mnt/stock_nvme_new/stock_predictor && \
        .venv/bin/python -m orchestration.daily_paper >> logs/daily_paper.out 2>&1

Run manually:  python -m orchestration.daily_paper
"""
from __future__ import annotations

from core.logging_setup import get_logger, stage

log = get_logger(__name__)


def _refresh_data() -> bool:
    """Bring Gold up to date. Returns False if the refresh was skipped.

    Two paths. With ``USE_MDS_SILVER=1`` this project owns no acquisition at all:
    it waits for MDS to publish ``day_ready`` and rebuilds Gold from Silver. The
    legacy path below — pull from Kite, rebuild the local Parquet lake — is kept
    only for running without MDS.

    Dropping the Kite pull is not a saving, it is a correctness fix. Once
    ``build_gold`` reads MDS, the local pull maintains a Parquet lake that
    nothing reads: `features/assembler.py` takes it only in the
    ``USE_MDS_SILVER=0`` branch. So it was a full daily Kite fetch producing dead
    data — and, on a box where the broker exists specifically to make MDS the
    only Kite consumer, a third one nobody had counted.
    """
    from core.mds_source import is_enabled as _mds_enabled

    if _mds_enabled():
        from core.mds_source import wait_for_day_ready
        from features.assembler import build_gold

        with stage(log, "daily-data-refresh"):
            if wait_for_day_ready() is None:
                log.warning("MDS has not published day_ready — SKIPPING the refresh and "
                            "rebalancing on the latest existing Gold. Check: "
                            "python -m mds.cli status")
                return False
            build_gold()      # features → gold, sourced from MDS Silver
        return True

    from ingest.kite_client import verify

    if not verify():
        log.warning("Kite session not live — SKIPPING daily data refresh. Rebalancing on the "
                    "latest existing Gold. Run: python -m ingest.kite_client login")
        return False
    from features.assembler import build_gold
    from ingest.bronze_to_silver import build_silver
    from ingest.daily_incremental import run as daily_pull

    with stage(log, "daily-data-refresh"):
        daily_pull()      # incremental OHLCV → bronze
        build_silver()    # adjust + re-tag PIT membership → silver
        build_gold()      # features → gold
    return True


def main() -> int:
    from core.mds_source import is_enabled as _mds_enabled

    with stage(log, "daily-paper-driver"):
        refreshed = False
        try:
            refreshed = _refresh_data()
        except Exception as exc:  # noqa: BLE001 — never let a data hiccup skip the rebalance
            log.exception("Data refresh failed (%s) — proceeding to rebalance on existing Gold.", exc)

        # If Kite is live, top up any missing survivorship losers (idempotent, self-throttled;
        # a fast no-op once they're all backfilled). This is the recovery trigger for the
        # loser-backfill — it lands the day Kite's instruments endpoint is healthy again.
        if refreshed and not _mds_enabled():
            try:
                from orchestration.backfill_losers import main as backfill_losers
                backfill_losers()
            except Exception as exc:  # noqa: BLE001 — never let backfill block the rebalance
                log.warning("Loser backfill skipped (%s).", exc)

        # Rebalance is the point of the loop; run it regardless of refresh outcome (idempotent).
        from execution.paper_broker import rebalance, status
        rebalance()

        # Best-effort drift/IC-decay monitor — informational, never blocks.
        try:
            from monitoring.drift import run as drift_run
            drift_run()
        except Exception as exc:  # noqa: BLE001
            log.warning("Drift monitor skipped (%s).", exc)

        status()
        log.info("Daily paper driver complete (data %s).",
                 "refreshed" if refreshed else "NOT refreshed — Kite offline")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
