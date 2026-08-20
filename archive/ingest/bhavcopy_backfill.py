"""Authoritative delisted-name backfill from raw NSE bhavcopy (EQ cash segment).

jugaad-data interleaves a *different* same-ticker security into delisted names' history (e.g. HDFC
alternates real ₹2,702 @ 4.4M vol with a fake ₹552 @ 184k vol; also DHFL/PEL/IBULHSGFIN/IBVENTURES).
Raw bhavcopy has exactly one EQ row per (symbol, day), ISIN-stamped, so there is no interleaving.
This tool downloads the daily bhavcopy for every existing Bronze trading day, extracts the target
symbols, pins each to its **dominant ISIN** (rejecting a later reuse of the ticker by another
company), and merges the result into Bronze partition-safely (never clobbering current symbols).

Two NSE formats, auto-selected by date with fallback to the other:
  legacy (~2015 .. 2024-07): content/historical/EQUITIES/YYYY/MON/cmDDMONYYYYbhav.csv.zip
  UDiFF  (2024-07 ..)      : content/cm/BhavCopy_NSE_CM_0_0_0_YYYYMMDD_F_0000.csv.zip
Per-day extracted target rows are disk-cached (datalake/cache/bhavcopy/) so re-runs are cheap.

Run:  python -m ingest.bhavcopy_backfill              # missing = all_symbols_ever() - Bronze
      python -m ingest.bhavcopy_backfill HDFC DHFL    # explicit subset
"""
from __future__ import annotations

import io
import sys
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
import requests

from core.config import load_config, resolve
from core.io import read_dataset, write_partitioned
from core.logging_setup import get_logger, stage
from ingest.constituents import all_symbols_ever
from archive.ingest.historical_backfill import _BRONZE_COLS

log = get_logger(__name__)

_UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
       "(KHTML, like Gecko) Chrome/120 Safari/537.36")
_MON = ["JAN", "FEB", "MAR", "APR", "MAY", "JUN", "JUL", "AUG", "SEP", "OCT", "NOV", "DEC"]
_ARCH = "https://nsearchives.nseindia.com"


def _urls_for(d: pd.Timestamp) -> list[str]:
    """Candidate bhavcopy URLs for a date, ordered by the format most likely for that era."""
    ymd = d.strftime("%Y%m%d")
    udiff = f"{_ARCH}/content/cm/BhavCopy_NSE_CM_0_0_0_{ymd}_F_0000.csv.zip"
    legacy = (f"{_ARCH}/content/historical/EQUITIES/{d.year}/{_MON[d.month-1]}/"
              f"cm{d.day:02d}{_MON[d.month-1]}{d.year}bhav.csv.zip")
    return [legacy, udiff] if d < pd.Timestamp("2024-07-01") else [udiff, legacy]


def _parse(content: bytes) -> pd.DataFrame:
    """Parse a bhavcopy zip (either format) → normalized [symbol,date,ohlc,volume,isin], EQ only."""
    with zipfile.ZipFile(io.BytesIO(content)) as z:
        raw = pd.read_csv(z.open(z.namelist()[0]))
    raw.columns = [c.strip() for c in raw.columns]
    if "TckrSymb" in raw.columns:      # UDiFF
        df = raw[(raw["SctySrs"] == "EQ") & (raw.get("FinInstrmTp", "STK") == "STK")]
        out = pd.DataFrame({
            "symbol": df["TckrSymb"].str.strip(), "date": pd.to_datetime(df["TradDt"]),
            "open": df["OpnPric"], "high": df["HghPric"], "low": df["LwPric"],
            "close": df["ClsPric"], "volume": df["TtlTradgVol"], "isin": df["ISIN"].str.strip(),
        })
    else:                               # legacy
        df = raw[raw["SERIES"].str.strip() == "EQ"]
        # NSE's TIMESTAMP is usually "13-JUL-2020" but some days use a 2-digit year ("13-Jul-20");
        # parse tolerantly so a single odd file doesn't abort the whole backfill.
        out = pd.DataFrame({
            "symbol": df["SYMBOL"].str.strip(),
            "date": pd.to_datetime(df["TIMESTAMP"].str.strip(), format="mixed", dayfirst=True),
            "open": df["OPEN"], "high": df["HIGH"], "low": df["LOW"],
            "close": df["CLOSE"], "volume": df["TOTTRDQTY"], "isin": df["ISIN"].str.strip(),
        })
    return out


def _fetch_day(d: pd.Timestamp, targets: set[str], cache_dir, session: requests.Session,
               retries: int = 3) -> pd.DataFrame:
    """Download+parse one day's bhavcopy, filtered to target symbols. Disk-cached (small parquet)."""
    cache = cache_dir / f"dt={d.strftime('%Y-%m-%d')}.parquet"
    if cache.exists():
        return pd.read_parquet(cache)
    for url in _urls_for(d):
        for a in range(retries):
            try:
                r = session.get(url, headers={"User-Agent": _UA, "Accept": "*/*",
                                              "Referer": "https://www.nseindia.com/"}, timeout=30)
                if r.status_code == 404:
                    break                        # wrong format for this date → try the other url
                r.raise_for_status()
                sub = _parse(r.content)
                sub = sub[sub["symbol"].isin(targets)].copy()
                # Only a genuine HTTP 200 parse is cached — even if 0 target rows (legit absence).
                # A throttle/timeout failure must NOT be cached, or it silently loses that day.
                sub.to_parquet(cache, index=False)
                return sub
            except Exception:
                time.sleep(2.0 * (a + 1))        # backoff on throttle/timeout
    # every attempt failed (throttled) — signal failure (None) so the caller can retry; never cache
    return None


def backfill_bhavcopy(symbols: list[str] | None = None, workers: int = 8) -> int:
    cfg = load_config()
    existing = read_dataset(cfg.data.bronze, "equity_ohlcv")
    have = set(existing["symbol"].unique())
    targets = set(symbols) if symbols else (set(all_symbols_ever()) - have)
    if not targets:
        log.info("Nothing to backfill — Bronze already covers all_symbols_ever().")
        return 0
    # trading calendar = the dates Bronze already has (real NSE sessions), so the merge lines up
    days = pd.to_datetime(sorted(pd.to_datetime(existing["date"]).dt.normalize().unique()))
    cache_dir = resolve(cfg.data.bronze).parent / "cache" / "bhavcopy"
    cache_dir.mkdir(parents=True, exist_ok=True)
    log.info("Bhavcopy backfill: %d target symbols over %d trading days: %s",
             len(targets), len(days), sorted(targets))

    with stage(log, "download-bhavcopy"):
        frames: list[pd.DataFrame] = []
        pending = list(days)
        with requests.Session() as s:
            # Pass 1 concurrent; subsequent passes retry only the throttle-failed days, ever gentler
            # (fewer workers, longer pacing), so a full clean set is assembled without gaps.
            for pass_no, w in enumerate([workers, 3, 1, 1], 1):
                if not pending:
                    break
                failed, done = [], 0
                with ThreadPoolExecutor(max_workers=w) as ex:
                    futs = {ex.submit(_fetch_day, d, targets, cache_dir, s): d for d in pending}
                    for f in as_completed(futs):
                        d = futs[f]
                        sub = f.result()
                        if sub is None:
                            failed.append(d)                 # throttled → retry next pass
                        elif not sub.empty:
                            frames.append(sub)
                        done += 1
                        if done % 250 == 0:
                            log.info("  pass %d: %d/%d done (%d ok, %d failed)",
                                     pass_no, done, len(pending), len(frames), len(failed))
                log.info("Pass %d complete: %d failed (workers=%d)", pass_no, len(failed), w)
                pending = failed
                if pending and pass_no >= 2:
                    time.sleep(15)                            # cool-off before the next gentle pass
        if pending:
            log.error("ABORTING (Bronze untouched): %d days still unfetchable after retries: %s ...",
                      len(pending), [d.strftime('%Y-%m-%d') for d in pending[:10]])
            return 0
        if not frames:
            log.error("No bhavcopy rows for any target — aborting (Bronze untouched).")
            return 0
        new = pd.concat(frames, ignore_index=True)

    with stage(log, "isin-pin-and-clean"):
        # pin each symbol to its DOMINANT ISIN — rejects a later reuse of the ticker by another company
        keep = []
        for sym, g in new.groupby("symbol"):
            dom = g["isin"].value_counts().idxmax()
            k = g[g["isin"] == dom]
            dropped = len(g) - len(k)
            if dropped:
                log.info("  %s: pinned ISIN %s, dropped %d rows on other ISINs", sym, dom, dropped)
            keep.append(k)
        new = (pd.concat(keep, ignore_index=True)
               .drop_duplicates(subset=["symbol", "date"])[_BRONZE_COLS])
        # types to match Bronze
        new["volume"] = pd.to_numeric(new["volume"], errors="coerce").fillna(0).astype("int64")
        for c in ["open", "high", "low", "close"]:
            new[c] = pd.to_numeric(new[c], errors="coerce")
        new = new.dropna(subset=["close"])
        got = sorted(new["symbol"].unique())
        log.info("Assembled %d clean rows for %d/%d symbols: %s", len(new), len(got), len(targets), got)
        still = sorted(targets - set(got))
        if still:
            log.warning("No bhavcopy data for %d symbols: %s", len(still), still)

    with stage(log, "merge-into-bronze"):
        touched = set(pd.to_datetime(new["date"]).dt.strftime("%Y-%m-%d"))
        existing_touched = existing[
            pd.to_datetime(existing["date"]).dt.strftime("%Y-%m-%d").isin(touched)
        ][_BRONZE_COLS]
        combined = (pd.concat([existing_touched, new], ignore_index=True)
                    .drop_duplicates(subset=["symbol", "date"], keep="first")
                    .sort_values(["date", "symbol"]))
        n = write_partitioned(combined, cfg.data.bronze, "equity_ohlcv", date_col="date")
        log.info("Merged %d partitions: %d existing + %d new = %d rows",
                 len(touched), len(existing_touched), len(new), n)
    return len(new)


if __name__ == "__main__":
    args = [a.upper() for a in sys.argv[1:]] or None
    backfill_bhavcopy(args)
