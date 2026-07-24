"""Point-in-time index membership — the survivorship-bias firewall.

The tradable universe on any date must be the stocks that were ACTUALLY in the index THEN,
including names later delisted or demoted. We keep a membership-history CSV with effective
date ranges and answer `members_on(date)` from it.

CSV schema (config/nifty200_membership_history.csv):
    symbol,valid_from,valid_to        # valid_to blank/9999-12-31 = still a member

If the history CSV is absent we fall back to a current-snapshot file so the pipeline still
runs in dev — but that path is survivorship-BIASED and logs a loud warning.
"""
from __future__ import annotations

from datetime import date

import pandas as pd

from core.config import load_config, resolve
from core.logging_setup import get_logger

log = get_logger(__name__)

_FAR_FUTURE = pd.Timestamp("9999-12-31")


def load_membership() -> pd.DataFrame:
    cfg = load_config()
    path = resolve(cfg.universe.pit_membership_csv)
    if not path.exists():
        log.warning(
            "PIT membership CSV missing (%s) — falling back to current snapshot. "
            "This introduces SURVIVORSHIP BIAS; supply the history file for real backtests.",
            path,
        )
        return _current_snapshot_fallback()
    df = pd.read_csv(path)
    df["valid_from"] = pd.to_datetime(df["valid_from"])
    df["valid_to"] = pd.to_datetime(df["valid_to"]).fillna(_FAR_FUTURE)
    return df


def members_on(as_of: date | str) -> list[str]:
    """Symbols that were index members on `as_of` (inclusive of valid_from, exclusive of valid_to)."""
    as_of = pd.Timestamp(as_of)
    df = load_membership()
    mask = (df["valid_from"] <= as_of) & (as_of < df["valid_to"])
    return sorted(df.loc[mask, "symbol"].unique().tolist())


def all_symbols_ever() -> list[str]:
    """Every symbol that was ever a member — the set we must backfill (incl. delisted)."""
    return sorted(load_membership()["symbol"].unique().tolist())


def _current_snapshot_fallback() -> pd.DataFrame:
    """Dev fallback: read a plain list of current constituents if present, else a tiny stub."""
    cfg = load_config()
    snap = resolve("config/nifty200_current.csv")
    if snap.exists():
        syms = pd.read_csv(snap)["symbol"].tolist()
    else:
        # Minimal stub so `python -m ...` doesn't crash before data is wired up.
        syms = ["RELIANCE", "TCS", "HDFCBANK", "INFY", "ICICIBANK"]
        log.warning("Using a 5-symbol STUB universe. Provide %s or the membership history.", snap)
    return pd.DataFrame(
        {"symbol": syms, "valid_from": pd.Timestamp(cfg.data.history_start), "valid_to": _FAR_FUTURE}
    )
