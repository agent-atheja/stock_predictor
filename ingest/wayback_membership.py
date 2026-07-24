"""Reconstruct NIFTY 200 membership history from Wayback Machine snapshots of the index
constituent CSV — a free, self-contained (partial) survivorship fix.

Method:
  1. Query the Wayback CDX API for every archived snapshot of ind_nifty200list.csv across the
     hosts NSE has used (niftyindices.com, *.nseindia.com), dedup by content digest.
  2. Download each snapshot's raw CSV → the exact member set on that date.
  3. Diff consecutive snapshots to derive reconstitution EVENTS, mapped onto our seed+replay model:
       • current member absent early then present later → 'add' at first-seen (join-date fix)
       • non-current name seen in a snapshot (delisted/demoted) → 'add' at first-seen +
         'remove' at the snapshot after last-seen (reinstates the loser into the universe)
  4. Write config/reconstitution_events.csv, then build_pit_membership() replays them + clamps.

HONEST LIMITS (Wayback coverage is sparse — ~8 snapshots, 2017–2024):
  • effective dates are APPROXIMATE (= snapshot date, not the true review date).
  • names added AND removed between two snapshots are invisible; pre-first-snapshot is uncovered.
  • current members present in the EARLIEST snapshot are left at the seed start (we cannot prove a
    later join within coverage) — refined only by the listing-bound clamp.
This is a real but partial improvement; see config/RESIDUAL_BIAS.md.

Run:  python -m ingest.wayback_membership          # fetch, diff, write events, rebuild membership
      python -m ingest.wayback_membership --dry     # just print the derived events, write nothing
"""
from __future__ import annotations

import io
import sys
import time

import pandas as pd
import requests

from core.config import load_config, resolve
from core.logging_setup import get_logger, stage

log = get_logger(__name__)

_UA = {"User-Agent": "Mozilla/5.0 (X11; Linux x86_64) research/1.0"}
_CDX = "http://web.archive.org/cdx/search/cdx"
# niftyindices.com holds all usable snapshots; the nseindia.com CDX endpoints reliably time out
# and add nothing, so they are omitted to keep refreshes fast. Re-add if you need wider coverage.
_HOSTS = [
    "niftyindices.com/IndexConstituent/ind_nifty200list.csv",
]


def _cdx_snapshots(url: str) -> list[tuple[str, str]]:
    """Return [(timestamp, original_url)] of archived snapshots for `url`, deduped by digest."""
    try:
        r = requests.get(_CDX, params={"url": url, "output": "json",
                                       "fl": "timestamp,original,digest", "collapse": "digest"},
                          headers=_UA, timeout=60)
        rows = r.json()
    except Exception as exc:  # noqa: BLE001
        log.warning("CDX query failed for %s: %s", url, exc)
        return []
    return [(row[0], row[1]) for row in rows[1:]] if len(rows) > 1 else []


def _fetch_snapshot_symbols(timestamp: str, original: str) -> set[str] | None:
    """Download one raw (id_) snapshot CSV and return its symbol set, or None on failure."""
    raw = f"https://web.archive.org/web/{timestamp}id_/{original}"
    for attempt in range(3):
        try:
            r = requests.get(raw, headers=_UA, timeout=45)
            if r.status_code != 200 or not r.text.strip():
                return None
            df = pd.read_csv(io.StringIO(r.text))
            col = next((c for c in df.columns if c.strip().lower() in ("symbol", "symbol ")), None)
            if col is None:
                return None
            syms = {str(s).strip().upper() for s in df[col].dropna()}
            return syms or None
        except Exception:  # noqa: BLE001
            time.sleep(2 * (attempt + 1))
    return None


def _snapshot_cache_path():
    return resolve("datalake/cache") / "wayback_nifty200_snapshots.json"


def collect_snapshots(use_cache: bool = True) -> list[tuple[pd.Timestamp, set[str]]]:
    """Gather (date, member_set) for every distinct snapshot across all known hosts, sorted.
    Cached to disk (Wayback is slow and we should not re-hammer it) — delete the cache to refresh."""
    import json

    cache = _snapshot_cache_path()
    if use_cache and cache.exists():
        raw = json.loads(cache.read_text())
        log.info("Loaded %d Wayback snapshots from cache %s", len(raw), cache.name)
        return [(pd.Timestamp(d), set(syms)) for d, syms in raw]

    seen_digests: dict[str, tuple[pd.Timestamp, set[str]]] = {}
    for host in _HOSTS:
        for ts, original in _cdx_snapshots(host):
            date = pd.to_datetime(ts[:8], format="%Y%m%d")
            syms = _fetch_snapshot_symbols(ts, original)
            if syms and len(syms) >= 150:  # sanity: a real NIFTY200 dump is ~200 names
                key = f"{date:%Y%m%d}"
                # keep the first good snapshot per calendar day
                if key not in seen_digests:
                    seen_digests[key] = (date, syms)
                    log.info("snapshot %s: %d members (%s)", date.date(), len(syms), host)
    snaps = sorted(seen_digests.values(), key=lambda x: x[0])
    if snaps:  # persist so we don't re-hit Wayback on every iteration
        import json
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_text(json.dumps([[f"{d:%Y-%m-%d}", sorted(s)] for d, s in snaps]))
        log.info("Cached %d snapshots → %s", len(snaps), cache.name)
    return snaps


# Known NSE symbol renames / share-class cancellations whose successor is (or maps to) a current
# member. Excluded from "reinstate" so we don't double-count a company or stage an un-backfillable
# legacy ticker. NOT an exhaustive corporate-actions table — just the ones this diff surfaces.
_RENAME_PREDECESSORS = {
    "LTI", "MINDTREE",       # → LTIM (LTIMindtree)
    "MOTHERSUMI",            # → MSUMI / MOTHERSON (demerger)
    "SRTRANSFIN",            # → SHRIRAMFIN
    "TATAGLOBAL",            # → TATACONSUM
    "TATAMTRDVR",            # DVR share class, cancelled (not a distinct company)
    "ZOMATO",                # → ETERNAL
    "CADILAHC",              # → ZYDUSLIFE
    "AMARAJABAT",            # → ARE&M (Amara Raja Energy & Mobility)
    "IDFCBANK", "IDFC",      # → IDFCFIRSTB (rename + holdco merger)
    "ADANITRANS",            # → ADANIENSOL (Adani Energy Solutions)
    "L&TFH",                 # → LTF (L&T Finance)
    "MCDOWELL-N",            # → UNITDSPR (United Spirits)
}


def derive_events(snaps: list[tuple[pd.Timestamp, set[str]]], current: set[str],
                  safe_only: bool = True) -> pd.DataFrame:
    """Turn the snapshot timeline into reconstitution events on the seed+replay model.

    safe_only=True (default) emits the trustworthy subset: every non-current name that was ever a
    member (a real LOSER — delisted, demoted, or merged out), excluding known renames and DUMMY*
    settlement placeholders. Two exit cases:
      • left within coverage (has a later snapshot without it) → close at that snapshot date;
      • still present in the LAST snapshot but not current (a recent demotion, e.g. LICI/LTIM on a
        free-float basis) → close conservatively at the last snapshot date (it left sometime after,
        exact date unknown without newer snapshots).
    It drops all join-date corrections (coarse/late dates; the listing-bound clamp handles the
    verifiable part with exact dates)."""
    if len(snaps) < 2:
        log.warning("Only %d usable snapshot(s) — need ≥2 to diff. No events derived.", len(snaps))
        return pd.DataFrame(columns=["symbol", "action", "effective_date", "source"])

    # first/last snapshot-date each symbol is present, and presence in the earliest snapshot
    first_seen: dict[str, pd.Timestamp] = {}
    last_seen: dict[str, pd.Timestamp] = {}
    earliest_date = snaps[0][0]
    for date, syms in snaps:
        for s in syms:
            first_seen.setdefault(s, date)
            last_seen[s] = date

    # snapshot dates in order, to pick "the snapshot after last-seen" as an approx exit date
    dates = [d for d, _ in snaps]

    def next_snapshot_after(d: pd.Timestamp) -> pd.Timestamp | None:
        for x in dates:
            if x > d:
                return x
        return None

    rows: list[dict] = []
    for sym in sorted(first_seen):
        fs, ls = first_seen[sym], last_seen[sym]
        in_current = sym in current
        present_in_earliest = fs == earliest_date
        exit_date = next_snapshot_after(ls)  # None → persisted to the last snapshot

        if in_current:
            # Current member join-date correction. UNSAFE (coarse/late dates) → only when not safe_only.
            if not safe_only and not present_in_earliest:
                rows.append({"symbol": sym, "action": "add", "effective_date": fs,
                             "source": "wayback:join"})
        else:
            # Real loser (non-current). Skip renames-to-current and settlement placeholders.
            if sym in _RENAME_PREDECESSORS or sym.startswith("DUMMY"):
                continue
            # Close at the observed exit, or (recent demotion) conservatively at the last snapshot.
            close = exit_date if exit_date is not None else last_seen[sym]
            rows.append({"symbol": sym, "action": "add", "effective_date": fs,
                         "source": "wayback:reinstate"})
            rows.append({"symbol": sym, "action": "remove", "effective_date": close,
                         "source": "wayback:demote"})

    ev = pd.DataFrame(rows)
    if not ev.empty:
        ev["effective_date"] = pd.to_datetime(ev["effective_date"]).dt.strftime("%Y-%m-%d")
        ev = ev.sort_values(["effective_date", "symbol"]).reset_index(drop=True)
    return ev


def run(dry: bool = False) -> pd.DataFrame:
    cfg = load_config()
    with stage(log, "wayback-membership"):
        snaps = collect_snapshots()
        log.info("Collected %d usable snapshots spanning %s..%s", len(snaps),
                 snaps[0][0].date() if snaps else "—", snaps[-1][0].date() if snaps else "—")
        current = set(pd.read_csv(resolve(f"config/{cfg.universe.index.lower()}_current.csv"))
                      ["symbol"].astype(str).str.strip().str.upper())
        events = derive_events(snaps, current, safe_only=True)
        src = events.get("source", pd.Series(dtype=str))
        losers = sorted(events.loc[src == "wayback:reinstate", "symbol"]) if not events.empty else []
        log.info("Safe subset: %d events reinstating %d delisted/demoted losers: %s",
                 len(events), len(losers), ", ".join(losers))

        if dry:
            print(events.to_string(index=False) if not events.empty else "(no events derived)")
            return events

        out = resolve("config/reconstitution_events.csv")
        events.to_csv(out, index=False)
        log.info("Wrote %d reconstitution events → %s", len(events), out.name)

        # Rebuild PIT membership: seed → these events → listing-bound clamp.
        from ingest.reference_data import build_pit_membership
        build_pit_membership()
    return events


if __name__ == "__main__":
    run(dry="--dry" in sys.argv)
