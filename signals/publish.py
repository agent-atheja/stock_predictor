"""Publish the daily book to the shared `stock_predictions` database.

Until now this project kept no signal history at all: ``generate_daily`` wrote
``signals_latest.csv`` and overwrote it on the next run. There was no way to ask
how last month's book performed, or whether the model's view of a name had
changed. This is that history, starting now — it cannot recover the past.

Two writes per name, matching the two-layer contract:

``common.signal``
    The conformed row every prediction app publishes, through
    ``common.publish_signal`` rather than a raw INSERT so the contract cannot be
    bypassed. Keyed on (app, symbol, as_of_date, horizon, model_version), so
    re-running a day updates in place instead of doubling the book.

``pred_xsrank.ranking``
    The native detail — raw LambdaRank score, rank, bucket, book side, and the
    point-in-time membership flag — keyed back by (signal_id, as_of_date).

Every signal records ``data_watermark`` and ``adj_vintage`` from MDS. Without
them a signal cannot be reproduced once a corporate action rewrites the adjusted
history it was computed from, and "the model changed" becomes indistinguishable
from "the data underneath it changed".

**Opt-in and non-blocking.** Set ``PUBLISH_SIGNALS=1``. A database problem must
never stop the book being generated — the CSV is still written either way.
"""
from __future__ import annotations

import logging
import os
from typing import Optional

import pandas as pd

log = logging.getLogger(__name__)

APP_CODE = "xsrank"
HORIZON_DAYS = 5          # this project ranks by 5-trading-day forward return
_ENV_FLAG = "PUBLISH_SIGNALS"


def is_enabled() -> bool:
    from core.mds_source import _load_env
    if _ENV_FLAG not in os.environ:
        _load_env()
    return os.environ.get(_ENV_FLAG, "").strip().lower() in ("1", "true", "yes", "on")


def _mds_provenance(as_of) -> tuple[Optional[str], Optional[str]]:
    """The MDS watermark and adjustment vintage behind this book.

    Read from the manifest rather than assumed: it is the difference between a
    reproducible signal and one that merely looks reproducible.
    """
    from core.mds_source import _load_env
    _load_env()
    dsn = os.environ.get("MDS_READER_DSN")
    if not dsn:
        return None, None
    import psycopg2
    conn = psycopg2.connect(dsn)
    try:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT watermark FROM meta.partition_manifest "
                "WHERE dataset='silver.equity_ohlcv_adj' AND trade_date=%s", (as_of,))
            r = cur.fetchone()
            watermark = r[0] if r else None
            cur.execute(
                "SELECT max(adj_vintage) FROM silver.equity_ohlcv_adj WHERE trade_date=%s",
                (as_of,))
            r = cur.fetchone()
            vintage = r[0] if r else None
    finally:
        conn.close()
    return watermark, vintage


def publish(ranked: pd.DataFrame, as_of, model_version: str = "lgbm_ranker_v1") -> int:
    """Write the book to stock_predictions. Returns rows published."""
    dsn = os.environ.get("PRED_XSRANK_DSN")
    if not dsn:
        log.warning("PRED_XSRANK_DSN not set — skipping publish. It lives in "
                    "/mnt/stock_nvme_new/predictions_db/secrets/.env")
        return 0

    watermark, vintage = _mds_provenance(as_of)
    if watermark is None or vintage is None:
        # Refuse rather than invent. A signal whose provenance is fabricated is
        # worse than one that was never recorded.
        log.error("no MDS manifest for %s — refusing to publish signals without "
                  "provenance", as_of)
        return 0

    import psycopg2

    # Only actionable names carry a direction; the held middle is recorded too,
    # so the book can be reconstructed exactly rather than inferred from the
    # extremes that happened to be traded.
    side = {"long": "long", "short": "short", "hold": "neutral"}

    conn = psycopg2.connect(dsn)
    published = 0
    try:
        conn.autocommit = False
        with conn.cursor() as cur:
            cur.execute(
                "INSERT INTO meta.prediction_run (app_code, job, as_of_date) "
                "VALUES (%s,'generate_daily',%s) RETURNING run_id", (APP_CODE, as_of))
            run_id = cur.fetchone()[0]

            for row in ranked.itertuples(index=False):
                cur.execute(
                    "SELECT common.publish_signal(%s,%s,%s,%s,%s,%s,%s,%s,%s,NULL,%s,%s,%s)",
                    (APP_CODE, model_version, row.symbol, as_of, HORIZON_DAYS,
                     side.get(row.bucket, "neutral"), watermark, vintage,
                     float(row.score), int(row.rank),
                     row.bucket in ("long", "short"), run_id),
                )
                signal_id = cur.fetchone()[0]
                cur.execute(
                    """
                    INSERT INTO pred_xsrank.ranking
                        (signal_id, as_of_date, symbol, raw_score, rank, bucket,
                         book_side, is_member)
                    VALUES (%s,%s,%s,%s,%s,%s,%s,%s)
                    ON CONFLICT (signal_id, as_of_date) DO UPDATE SET
                        raw_score = EXCLUDED.raw_score, rank = EXCLUDED.rank,
                        bucket = EXCLUDED.bucket, book_side = EXCLUDED.book_side,
                        is_member = EXCLUDED.is_member
                    """,
                    (signal_id, as_of, row.symbol, float(row.score), int(row.rank),
                     row.bucket, row.bucket if row.bucket in ("long", "short") else None,
                     bool(getattr(row, "is_member", True))),
                )
                published += 1

            cur.execute(
                "UPDATE meta.prediction_run SET ended_at=now(), status='succeeded', "
                "rows_written=%s WHERE run_id=%s", (published, run_id))
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()

    log.info("published %s signals to stock_predictions for %s "
             "(watermark=%s, adj_vintage=%s)", published, as_of, watermark, vintage)
    return published


def publish_safely(ranked: pd.DataFrame, as_of, **kw) -> int:
    """Publish, but never let a database problem stop the book being produced."""
    if not is_enabled():
        return 0
    try:
        return publish(ranked, as_of, **kw)
    except Exception as exc:                           # noqa: BLE001
        log.error("signal publish failed (book still written to CSV): %s", exc)
        return 0
