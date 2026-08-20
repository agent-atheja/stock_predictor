# archive/ — retired acquisition layer

This project no longer acquires market data. It reads `silver.equity_ohlcv_adj`
from MDS and waits on the `day_ready` contract; on the MDS path it makes **no
external calls of its own**.

Everything here fetched data from Kite or NSE and has been superseded. It is kept
for reference — the rate-limit discipline, the bhavcopy parsing and the
bronze→silver adjustment logic are all worth reading before anyone writes
something similar again — but it is **deliberately not wired**. Nothing in the
live tree imports it.

| Module | What it did | Replaced by |
|---|---|---|
| `ingest/kite_client.py` | Authenticated Kite handle, token login, `historical()` | MDS quota broker |
| `ingest/daily_incremental.py` | Post-close OHLCV pull for current members | MDS `daily_ohlcv` |
| `ingest/historical_backfill.py` | Deep history → bronze | MDS Bronze |
| `ingest/kite_backfill.py` | Kite history → bronze | MDS Bronze |
| `ingest/bhavcopy_backfill.py` | NSE bhavcopy → bronze | MDS `silver.delivery` |
| `ingest/bronze_to_silver.py` | Adjustment + PIT membership tagging → silver | MDS `conform.silver` |
| `orchestration/backfill_losers.py` | Kite top-up for survivorship losers | not needed — MDS carries delisted history |

`ingest/constituents.py`, `ingest/reference_data.py` and `ingest/wayback_membership.py`
stayed in the live tree: they build index membership and corporate actions rather
than fetching prices, and `reference_data` is still exercised by the test suite.

## Why it was retired rather than deleted

The daily pull was doing real work that nothing consumed. Once `build_gold()`
read Silver from MDS, `bronze_to_silver` maintained a Parquet lake that
`features/assembler.py` only touches in the `USE_MDS_SILVER=0` branch — so a full
Kite fetch ran every morning to produce data no one read, on an account the MDS
broker exists to protect.

The Parquet lake itself is still on disk and still readable with
`USE_MDS_SILVER=0`. That is deliberate: it is the control arm for comparing MDS
against the legacy source, and it is how the 2026-08-18 matched control was run.
What is gone is the ability to *advance* it.

## If you need this back

Set `USE_MDS_SILVER=0` and re-wire the imports; `orchestration/daily_paper.py`
will tell you it has been retired rather than silently rebalancing on stale Gold.
Before re-enabling any of it, read `marketdata/CLAUDE.md` on the instrument dump —
a burst of `kite.instruments("NSE")` trips an account-level block that looks like
an ingest finishing instantly with `ok=0`.
