"""Reference-data fetchers — the inputs that turn an honest-but-empty backtest into a real one.

Produces three config artifacts:
  1. config/<index>_current.csv        — real constituents + sector (from niftyindices.com)
  2. config/<index>_membership_history.csv — PIT membership (seeded from current; augmentable)
  3. config/corporate_actions.csv       — split/bonus factors (best-effort via NSE; else template)
  4. config/sectors.csv                 — symbol → sector map (for sector-exposure limits)

Honest caveat on PIT membership: NSE does not publish a clean free API of historical index
reconstitutions. We SEED membership from the current list (valid_from = history_start) and give
`add_membership_change()` + a documented CSV so you can append real reconstitution events (each
semi-annual review) to remove survivorship bias properly. Until you do, the seed is survivorship-
biased and the pipeline says so loudly.

Run:  python -m ingest.reference_data              # fetch all for config.universe.index
      python -m ingest.reference_data constituents # just the constituent list
"""
from __future__ import annotations

import io
import sys
import time

import pandas as pd
import requests

from core.config import load_config, resolve
from core.logging_setup import get_logger

log = get_logger(__name__)

# Sentinel for an open (still-current) membership interval. pandas 3.x holds this at `us`
# resolution without overflow, so it round-trips through to_datetime cleanly.
_FAR_FUTURE = pd.Timestamp("9999-12-31")
_VALID_ACTIONS = {"add", "remove"}

_UA = "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124 Safari/537.36"
_INDEX_CSV = {
    "NIFTY50": "ind_nifty50list.csv",
    "NIFTY100": "ind_nifty100list.csv",
    "NIFTY200": "ind_nifty200list.csv",
    "NIFTY500": "ind_nifty500list.csv",
}


# ── 1. constituents + sector ─────────────────────────────────────────────────
def fetch_index_constituents(index: str | None = None) -> pd.DataFrame:
    cfg = load_config()
    index = (index or cfg.universe.index).upper()
    if index not in _INDEX_CSV:
        raise ValueError(f"Unsupported index {index}; choose {list(_INDEX_CSV)}")
    url = f"https://niftyindices.com/IndexConstituent/{_INDEX_CSV[index]}"
    r = requests.get(url, headers={"User-Agent": _UA, "Accept": "text/csv,*/*"}, timeout=20)
    r.raise_for_status()
    df = pd.read_csv(io.StringIO(r.text))
    df = df.rename(columns={"Symbol": "symbol", "Industry": "sector", "Company Name": "company"})
    df = df[["symbol", "company", "sector"]].dropna(subset=["symbol"])

    current = resolve(f"config/{index.lower()}_current.csv")
    df[["symbol"]].to_csv(current, index=False)
    df[["symbol", "sector"]].to_csv(resolve("config/sectors.csv"), index=False)
    log.info("Fetched %d constituents for %s → %s (+ sectors.csv)", len(df), index, current.name)
    return df


# ── 2. PIT membership (seed + augment) ───────────────────────────────────────
def seed_membership_history(index: str | None = None) -> pd.DataFrame:
    cfg = load_config()
    index = (index or cfg.universe.index).upper()
    cons = fetch_index_constituents(index)
    hist = pd.DataFrame(
        {
            "symbol": cons["symbol"],
            "valid_from": cfg.data.history_start,
            "valid_to": "9999-12-31",
        }
    )
    out = resolve(cfg.universe.pit_membership_csv)
    hist.to_csv(out, index=False)
    log.warning(
        "Seeded membership history from CURRENT %s constituents → %s. This is SURVIVORSHIP-"
        "BIASED until you append historical reconstitution events (see add_membership_change).",
        index, out.name,
    )
    return hist


def _validate_events(events: pd.DataFrame, events_name: str) -> pd.DataFrame:
    """Schema + value guard on a reconstitution-events log. Fail loud on structural problems
    (missing columns, bad action, unparseable date) — a silently-wrong membership file corrupts
    every downstream backtest. Returns the cleaned, date-sorted frame."""
    required = {"symbol", "action", "effective_date"}
    missing = required - set(events.columns)
    if missing:
        raise ValueError(f"{events_name}: missing columns {sorted(missing)} (need symbol,action,effective_date)")

    events = events.copy()
    events["symbol"] = events["symbol"].astype(str).str.strip().str.upper()
    events["action"] = events["action"].astype(str).str.strip().str.lower()

    bad_action = events.loc[~events["action"].isin(_VALID_ACTIONS), "action"].unique()
    if len(bad_action):
        raise ValueError(f"{events_name}: invalid action(s) {list(bad_action)} — must be 'add' or 'remove'")

    parsed = pd.to_datetime(events["effective_date"], errors="coerce")
    if parsed.isna().any():
        bad = events.loc[parsed.isna(), "effective_date"].tolist()
        raise ValueError(f"{events_name}: unparseable effective_date(s) {bad}")
    events["effective_date"] = parsed

    dupes = events.duplicated(subset=["symbol", "action", "effective_date"])
    if dupes.any():
        log.warning("%s: dropping %d duplicate event row(s).", events_name, int(dupes.sum()))
        events = events[~dupes]
    return events.sort_values("effective_date").reset_index(drop=True)


def _replay_events(seed: pd.DataFrame, events: pd.DataFrame, history_start: pd.Timestamp) -> pd.DataFrame:
    """Replay validated add/remove events onto the seed, in date order. Returns membership rows."""
    hist = seed.copy()
    hist["symbol"] = hist["symbol"].astype(str).str.strip().str.upper()
    hist["valid_from"] = pd.to_datetime(hist["valid_from"])
    hist["valid_to"] = pd.to_datetime(hist["valid_to"])

    applied = 0
    for _, ev in events.iterrows():
        sym, action, eff = ev["symbol"], ev["action"], ev["effective_date"]
        if action == "remove":
            m = (hist["symbol"] == sym) & (hist["valid_to"] == _FAR_FUTURE)
            if not m.any():
                log.warning("remove %s @ %s: no open membership interval — event ignored.",
                            sym, eff.date())
                continue
            hist.loc[m, "valid_to"] = eff
        else:  # add
            # Two distinct cases:
            #  (a) join-date correction — the seed opened this current member at history_start,
            #      but it verifiably joined later. Move valid_from up instead of appending a
            #      second interval (which would double-count the name).
            #  (b) genuine re-add — no open interval (it was removed earlier), so open a new one.
            seed_open = (hist["symbol"] == sym) & (hist["valid_from"] == history_start) & \
                        (hist["valid_to"] == _FAR_FUTURE)
            open_now = (hist["symbol"] == sym) & (hist["valid_to"] == _FAR_FUTURE)
            if seed_open.any():
                hist.loc[seed_open, "valid_from"] = eff  # (a) correct the seeded join date
            elif open_now.any():
                log.warning("add %s @ %s: already an open member — event ignored.", sym, eff.date())
                continue
            else:
                hist = pd.concat(  # (b) genuine re-add
                    [hist, pd.DataFrame([{"symbol": sym, "valid_from": eff, "valid_to": _FAR_FUTURE}])],
                    ignore_index=True,
                )
        applied += 1
    log.info("Applied %d/%d reconstitution events.", applied, len(events))
    return hist


def _listing_bounds_from_bronze() -> pd.Series:
    """First observed price-bar date per symbol from the Bronze layer — a hard, self-contained
    LOWER BOUND on index membership (a stock cannot be in an index before it is listed/traded).
    Empty Series if Bronze is absent. Provenance: our own ingested data, not a third party."""
    import glob

    cfg = load_config()
    root = resolve(cfg.data.bronze)
    parts = glob.glob(str(root / "**" / "*.parquet"), recursive=True)
    if not parts:
        log.warning("No Bronze parquet found under %s — skipping listing-bound clamp.", root)
        return pd.Series(dtype="datetime64[ns]")
    frames = []
    for p in parts:
        try:
            frames.append(pd.read_parquet(p, columns=["date", "symbol"]))
        except Exception as exc:  # noqa: BLE001
            log.warning("listing-bounds: skipping unreadable %s (%s)", p, exc)
    if not frames:
        return pd.Series(dtype="datetime64[ns]")
    b = pd.concat(frames, ignore_index=True)
    b["date"] = pd.to_datetime(b["date"])
    b["symbol"] = b["symbol"].astype(str).str.strip().str.upper()
    return b.groupby("symbol")["date"].min()


def _clamp_to_listing_bounds(hist: pd.DataFrame, bounds: pd.Series) -> pd.DataFrame:
    """Raise each interval's valid_from up to the symbol's first-traded date (listing bound).
    Deterministic survivorship firewall: kills the 'IPO winner backfilled to history_start'
    artifact. Drops any interval whose window collapses (valid_from >= valid_to) after clamping."""
    if bounds.empty:
        return hist
    hist = hist.copy()
    lb = hist["symbol"].map(bounds)
    clamp = lb.notna() & (lb > hist["valid_from"])
    n = int(clamp.sum())
    hist.loc[clamp, "valid_from"] = lb[clamp]
    collapsed = hist["valid_from"] >= hist["valid_to"]
    if collapsed.any():
        log.info("listing-bound clamp dropped %d collapsed interval(s).", int(collapsed.sum()))
        hist = hist[~collapsed]
    log.info("Clamped %d interval(s) to listing bounds (self-contained survivorship firewall).", n)
    return hist


def build_pit_membership(events_csv: str = "config/reconstitution_events.csv",
                         clamp_listing: bool = True) -> pd.DataFrame:
    """Build point-in-time index membership — the survivorship fix, end to end.

    Layers, in order:
      1. Seed from the CURRENT constituents (valid_from = history_start).
      2. Replay curated NSE reconstitution events (add/remove) if the events CSV has any.
      3. Clamp each interval's valid_from up to the symbol's first-traded date from Bronze
         (a hard, self-contained lower bound — a stock can't be an index member before listing).

    Idempotent: always rebuilds from seed + full event log + bronze bounds, so re-running yields
    the same membership. Writes config.universe.pit_membership_csv and returns the frame.
    """
    cfg = load_config()
    history_start = pd.to_datetime(cfg.data.history_start)
    seed = seed_membership_history()  # current constituents, valid_from = history_start

    events_path = resolve(events_csv)
    if events_path.exists() and not (raw := pd.read_csv(events_path)).empty:
        hist = _replay_events(seed, _validate_events(raw, events_path.name), history_start)
    else:
        log.warning(
            "No curated reconstitution events (%s) — relying on the listing-bound clamp only. "
            "Append NSE semi-annual review circulars (symbol,action,effective_date) to correct "
            "names that graduated INTO the index with a long pre-existing price history.",
            events_path,
        )
        hist = seed.copy()
        hist["valid_from"] = pd.to_datetime(hist["valid_from"])
        hist["valid_to"] = pd.to_datetime(hist["valid_to"])

    if clamp_listing:
        hist = _clamp_to_listing_bounds(hist, _listing_bounds_from_bronze())

    out = resolve(cfg.universe.pit_membership_csv)
    hist.sort_values(["symbol", "valid_from"]).to_csv(out, index=False)
    log.info("PIT membership built: %d rows, %d symbols → %s",
             len(hist), hist["symbol"].nunique(), out.name)
    return hist


def apply_reconstitution_events(events_csv: str = "config/reconstitution_events.csv") -> pd.DataFrame:
    """Backward-compatible alias. Prefer build_pit_membership(), which also applies the
    self-contained listing-bound survivorship clamp."""
    return build_pit_membership(events_csv=events_csv, clamp_listing=True)


def add_membership_change(symbol: str, action: str, effective_date: str) -> None:
    """Append one reconstitution event. action='add' → new valid_from row; action='remove' →
    close the open row's valid_to. Lets you build true PIT membership incrementally from NSE
    semi-annual review press releases."""
    cfg = load_config()
    path = resolve(cfg.universe.pit_membership_csv)
    df = pd.read_csv(path)
    # Parse valid_to so the open-interval test works whether the CSV stored "9999-12-31" or a
    # full "9999-12-31 00:00:00" timestamp (apply_reconstitution_events writes the latter).
    df["valid_to"] = pd.to_datetime(df["valid_to"])
    symbol = symbol.strip().upper()
    eff = pd.Timestamp(effective_date)
    if action == "remove":
        m = (df["symbol"] == symbol) & (df["valid_to"] == _FAR_FUTURE)
        df.loc[m, "valid_to"] = eff
    elif action == "add":
        df = pd.concat(
            [df, pd.DataFrame([{"symbol": symbol, "valid_from": eff, "valid_to": _FAR_FUTURE}])],
            ignore_index=True,
        )
    else:
        raise ValueError("action must be 'add' or 'remove'")
    df.to_csv(path, index=False)
    log.info("Recorded membership %s: %s @ %s", action, symbol, eff.date())


# ── 3. corporate actions (best-effort) ───────────────────────────────────────
def fetch_corporate_actions(symbols: list[str] | None = None) -> pd.DataFrame:
    """Split/bonus factors. PRIMARY source is yfinance (reliable for .NS tickers), with NSE as a
    fallback; on total failure writes a documented empty template. Ratio = price multiplier for
    bars BEFORE ex_date (a k-for-1 split → 1/k; e.g. 10-for-1 → 0.1)."""
    cfg = load_config()
    out = resolve("config/corporate_actions.csv")
    if symbols is None:
        cur = resolve(f"config/{cfg.universe.index.lower()}_current.csv")
        symbols = pd.read_csv(cur)["symbol"].tolist() if cur.exists() else []

    rows = _yf_corporate_actions(symbols)
    if rows.empty:
        try:
            rows = _nse_corporate_actions()
        except Exception as exc:  # noqa: BLE001
            log.warning("NSE CA fallback failed: %s", exc)

    if rows.empty:
        pd.DataFrame(columns=["symbol", "ex_date", "ratio", "purpose"]).to_csv(out, index=False)
        log.warning("No corporate actions fetched — wrote empty template %s. Prices stay UNADJUSTED.", out.name)
        return rows
    rows = rows.sort_values(["symbol", "ex_date"])
    rows.to_csv(out, index=False)
    log.info("Fetched %d corporate actions for %d symbols → %s",
             len(rows), rows["symbol"].nunique(), out.name)
    return rows


def _yf_corporate_actions(symbols: list[str]) -> pd.DataFrame:
    """Pull split history from yfinance for each symbol (parallel, I/O-bound). A k-for-1 split
    has yfinance value k; our price-adjustment ratio for pre-ex bars is 1/k."""
    import yfinance as yf

    from core.parallel import io_map

    def _one(sym: str) -> list[dict]:
        sp = yf.Ticker(f"{sym}.NS").splits
        recs = []
        for ts, val in sp.items():
            if val and val > 0:
                recs.append(
                    {
                        "symbol": sym,
                        "ex_date": pd.Timestamp(ts).tz_localize(None).normalize(),
                        "ratio": 1.0 / float(val),
                        "purpose": f"split {val:g}:1",
                    }
                )
        return recs

    results = io_map(_one, symbols, workers=8, rate_limit_per_sec=0, desc="yf-splits")
    flat = [r for sub in results if sub for r in sub]
    return pd.DataFrame(flat)


def _nse_corporate_actions() -> pd.DataFrame:
    """Prime NSE session cookies, then hit the corporate-actions API. Parses splits/bonuses into
    price-adjustment ratios."""
    s = requests.Session()
    s.headers.update({"User-Agent": _UA, "Accept": "application/json"})
    s.get("https://www.nseindia.com", timeout=15)
    time.sleep(1)
    r = s.get("https://www.nseindia.com/api/corporates-corporateActions?index=equities", timeout=20)
    r.raise_for_status()
    data = r.json()
    recs = []
    for it in data:
        purpose = str(it.get("subject", "")).lower()
        ratio = _parse_ratio(purpose)
        if ratio is None:
            continue
        recs.append(
            {
                "symbol": it.get("symbol"),
                "ex_date": pd.to_datetime(it.get("exDate"), dayfirst=True, errors="coerce"),
                "ratio": ratio,
                "purpose": purpose,
            }
        )
    return pd.DataFrame(recs).dropna(subset=["symbol", "ex_date"])


def _parse_ratio(purpose: str) -> float | None:
    """Turn a corporate-action subject into a price-adjustment ratio (splits & bonuses only)."""
    import re

    if "split" in purpose:
        m = re.search(r"(\d+)\D+(\d+)", purpose)  # 'from Rs 10 to Rs 2' → new/old face value
        if m:
            new, old = float(m.group(2)), float(m.group(1))
            return new / old if old else None
    if "bonus" in purpose:
        m = re.search(r"(\d+)\s*:\s*(\d+)", purpose)  # '1:1' → 1 new per 1 held
        if m:
            a, b = float(m.group(1)), float(m.group(2))
            return b / (a + b)  # price multiplier for pre-ex bars
    return None


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    if which in ("all", "constituents"):
        fetch_index_constituents()
    if which in ("all", "membership"):
        build_pit_membership()
    if which in ("all", "corporate"):
        fetch_corporate_actions()
