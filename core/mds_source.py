"""Read Silver from the Market Data Service instead of the local Parquet lake.

Drop-in replacement for ``read_dataset(cfg.data.silver, "equity_ohlcv_adj")``.
It returns the identical frame — same columns, same names, same dtypes — so
``features/assembler.py`` needs a source swap and nothing else.

**Inert until switched on.** Set ``USE_MDS_SILVER=1`` (and ``MDS_READER_DSN``);
otherwise nothing here runs and the Parquet path is untouched.

Two derived fields are reproduced server-side rather than in pandas:

``turnover_cr_20d``
    20-session mean of ``adj_close * volume``, in ₹ crore. Matches
    ``ingest/bronze_to_silver.py``'s ``rolling(20, min_periods=5)`` exactly,
    including the NULL for a symbol's first four bars — a window function with
    a plain 19-preceding frame would silently emit a value there and quietly
    widen the tradable universe at every symbol's listing.

``is_member``
    Point-in-time index membership, joined from ``dim.index_membership`` on a
    half-open ``[valid_from, valid_to)`` interval. In MDS that table is
    protected by an exclusion constraint, so a symbol cannot hold two
    overlapping intervals in one index — the survivorship guarantee this
    project enforces in Python is enforced by the database here.

What this replaces: the whole ``ingest/`` path — Kite client, bhavcopy
backfill, reference data, Wayback membership, bronze→silver. MDS does that
once, for every predictor.
"""
from __future__ import annotations

import logging
import os
from typing import Optional

import pandas as pd

log = logging.getLogger(__name__)

_ENV_FLAG = "USE_MDS_SILVER"

# The exact column set and order that read_dataset(silver) yields today.
SILVER_COLUMNS = [
    "symbol", "date", "open", "high", "low", "close", "volume",
    "adj_open", "adj_high", "adj_low", "adj_close",
    "turnover", "turnover_cr_20d", "is_member",
]

_SQL = """
WITH members AS (
    SELECT DISTINCT symbol FROM dim.index_membership WHERE index_name = %(index_name)s
),
base AS (
    SELECT s.symbol, s.trade_date,
           s.open, s.high, s.low, s.close, s.volume,
           s.adj_open, s.adj_high, s.adj_low, s.adj_close,
           (s.adj_close * s.volume) AS turnover
      FROM silver.v_equity_ohlcv_published s
     WHERE (NOT %(members_only)s OR s.symbol IN (SELECT symbol FROM members))
       AND (%(start)s::date IS NULL OR s.trade_date >= %(start)s)
       AND (%(end)s::date   IS NULL OR s.trade_date <= %(end)s)
),
liq AS (
    SELECT b.*,
           CASE WHEN count(*) OVER w >= 5
                THEN avg(b.turnover) OVER w / 1e7
           END AS turnover_cr_20d
      FROM base b
    WINDOW w AS (PARTITION BY b.symbol ORDER BY b.trade_date
                 ROWS BETWEEN 19 PRECEDING AND CURRENT ROW)
)
SELECT l.symbol,
       l.trade_date AS date,
       l.open, l.high, l.low, l.close, l.volume,
       l.adj_open, l.adj_high, l.adj_low, l.adj_close,
       l.turnover, l.turnover_cr_20d,
       (m.symbol IS NOT NULL) AS is_member
  FROM liq l
  LEFT JOIN dim.index_membership m
         ON m.symbol      = l.symbol
        AND m.index_name  = %(index_name)s
        AND l.trade_date >= m.valid_from
        AND l.trade_date <  m.valid_to
 ORDER BY l.symbol, l.trade_date
"""


def _load_env() -> None:
    """Populate the environment from secrets/.env, self-contained.

    Today the only thing that loads that file is ``ingest/kite_client.py`` —
    which this module exists to make redundant. Depending on it would mean the
    cutover works right up until ``ingest/`` is retired and then silently stops
    reading its own configuration. So the load is duplicated here rather than
    imported, mirroring how kite_client does it.

    ``override=False``: an explicitly exported variable must win over the file,
    or pointing a run at a different database becomes impossible.
    """
    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    from core.config import resolve
    try:
        load_dotenv(resolve("secrets/.env"), override=False)
    except Exception:                                  # noqa: BLE001
        pass


def is_enabled() -> bool:
    if _ENV_FLAG not in os.environ:
        _load_env()
    return os.environ.get(_ENV_FLAG, "").strip().lower() in ("1", "true", "yes", "on")


def read_silver(start: Optional[str] = None, end: Optional[str] = None,
                index_name: str = "NIFTY200",
                members_only: bool = True) -> pd.DataFrame:
    """Return the Silver panel from MDS, shaped exactly like the Parquet one.

    ``members_only`` restricts to symbols that have EVER been in ``index_name``
    — the equivalent of this project's ``all_symbols_ever()``. MDS publishes the
    full ~3,100-symbol NSE universe, but ``apply_universe_filter`` discards
    non-members downstream anyway, so building technicals for the other ~2,900
    is pure cost. Set it False only to deliberately widen the universe.

    Raises on failure rather than returning empty: an empty Silver frame makes
    ``build_gold`` log "run bronze_to_silver first" and exit 0, which looks like
    a configuration problem and hides a database outage. A loud failure is the
    correct outcome for a missing data source.
    """
    # Env override for the upper bound. Needed to reproduce a historical run
    # exactly: MDS carries data past whatever date a baseline was built on, and
    # an unpinned end silently extends the test window, which shows up as a
    # model-quality change when it is really a different period.
    end = end or os.environ.get("MDS_SILVER_END") or None

    if "MDS_READER_DSN" not in os.environ:
        _load_env()
    dsn = os.environ.get("MDS_READER_DSN")
    if not dsn:
        raise RuntimeError(
            "MDS_READER_DSN is not set. It lives in "
            "/mnt/stock_nvme_new/marketdata/secrets/.env")

    import psycopg2

    conn = psycopg2.connect(dsn)
    try:
        with conn.cursor() as cur:
            cur.execute(_SQL, {"start": start, "end": end, "index_name": index_name,
                               "members_only": members_only})
            cols = [d[0] for d in cur.description]
            df = pd.DataFrame(cur.fetchall(), columns=cols)
    finally:
        conn.close()

    if df.empty:
        raise RuntimeError(
            f"MDS returned no Silver rows for {start}..{end}. Check "
            f"`python -m mds.cli status` — is day_ready published?")

    # Parquet yields float64 for prices and bool for is_member; psycopg2 hands
    # back Decimal for NUMERIC, which would silently poison every downstream
    # technical indicator with object-dtype arithmetic.
    for c in ("open", "high", "low", "close", "adj_open", "adj_high",
              "adj_low", "adj_close", "turnover", "turnover_cr_20d"):
        df[c] = pd.to_numeric(df[c], errors="coerce").astype("float64")
    df["volume"] = pd.to_numeric(df["volume"], errors="coerce").fillna(0).astype("int64")
    df["is_member"] = df["is_member"].fillna(False).astype(bool)
    df["date"] = pd.to_datetime(df["date"])

    # MDS's canonical form is SYMBOL.NS; this project's internal form is bare
    # (`RELIANCE`, not `RELIANCE.NS`). The rule in both CLAUDE.mds is "convert
    # only at boundaries", and this function IS the boundary — the docstring
    # above promises a frame "shaped exactly like the Parquet one", and the
    # Parquet lake is bare.
    #
    # Without this the suffix leaks all the way through Gold into signals and
    # the paper portfolio, which ended up keying open positions as
    # "TVSMOTOR.NS" while last_prices kept the bare "TVSMOTOR" — the same
    # instrument spelled two ways in one file, with the bare half frozen at the
    # cutover date.
    df["symbol"] = df["symbol"].str.replace(r"\.NS$", "", regex=True)

    log.info("MDS silver: %s rows, %s symbols, %s..%s, %s member-rows",
             f"{len(df):,}", df["symbol"].nunique(),
             df["date"].min().date(), df["date"].max().date(),
             f"{int(df['is_member'].sum()):,}")

    return df[SILVER_COLUMNS]


def wait_for_day_ready(timeout_s: int = 1800, poll_s: int = 60):
    """Block until MDS publishes `day_ready` for the latest trading session.

    Returns the published date, or None on timeout.

    This replaces guessing at cron times. The old arrangement had this project
    pull its own OHLCV from Kite at 07:45 and hope MDS's 07:00 run had finished;
    a slow or failed MDS run produced a *stale-data rebalance* rather than a
    blocked one, which is the failure mode that hides. On 2026-08-18 the MDS
    pipeline died at 07:00:28 and Silver stayed three days old — exactly the case
    this guards.

    Waiting is also what makes dropping the local Kite pull safe: the reason to
    keep an independent refresh was never the data, it was not knowing when the
    shared data was ready.
    """
    import time

    import psycopg2

    if "MDS_READER_DSN" not in os.environ:
        _load_env()
    dsn = os.environ.get("MDS_READER_DSN")
    if not dsn:
        raise RuntimeError("MDS_READER_DSN is not set")

    deadline = time.time() + timeout_s
    while True:
        conn = psycopg2.connect(dsn)
        try:
            with conn.cursor() as cur:
                # The last COMPLETED session — strictly before today. This job
                # runs pre-open (07:45), when today's session has not happened,
                # so `<= current_date` would demand a date MDS cannot publish and
                # block until timeout every single day. Never a weekday
                # heuristic either: Indian market holidays are irregular and
                # dim.calendar also carries the weekend Muhurat sessions.
                cur.execute("""
                    SELECT max(c.trade_date)
                      FROM dim.calendar c
                     WHERE c.is_trading_day AND c.trade_date < current_date
                """)
                (want,) = cur.fetchone()
                cur.execute("""
                    SELECT max(trade_date) FROM meta.partition_manifest
                     WHERE dataset = 'day_ready'
                """)
                (have,) = cur.fetchone()
        finally:
            conn.close()

        if have is not None and want is not None and have >= want:
            log.info("MDS day_ready published for %s", have)
            return have

        if time.time() >= deadline:
            log.error("MDS day_ready still at %s, expected %s — gave up after %ss",
                      have, want, timeout_s)
            return None

        log.info("waiting for MDS day_ready: have=%s want=%s", have, want)
        time.sleep(poll_s)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)-7s %(message)s")
    d = read_silver(start="2026-06-01")
    print(d.dtypes)
    print(d.tail(3).to_string(index=False))
