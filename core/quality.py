"""Data-quality assertion layer — fail fast, alert loud (DE principle).

Lightweight, dependency-free assertions run at each pipeline stage. Layered per the standard
hierarchy: schema → volume → value → business → freshness. A failed HARD check raises; a SOFT
check logs a warning and records it. Returns a report dict for the stage to log/persist.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta

import pandas as pd

from core.logging_setup import get_logger

log = get_logger(__name__)


class DataQualityError(AssertionError):
    """Raised when a HARD data-quality assertion fails."""


@dataclass
class QualityReport:
    stage: str
    checks: list[dict] = field(default_factory=list)

    def add(self, name: str, passed: bool, detail: str, hard: bool) -> None:
        self.checks.append({"name": name, "passed": passed, "detail": detail, "hard": hard})
        level = log.info if passed else (log.error if hard else log.warning)
        level("DQ[%s] %s: %s — %s", self.stage, name, "PASS" if passed else "FAIL", detail)

    @property
    def failed_hard(self) -> list[dict]:
        return [c for c in self.checks if not c["passed"] and c["hard"]]

    def raise_if_failed(self) -> "QualityReport":
        if self.failed_hard:
            names = ", ".join(c["name"] for c in self.failed_hard)
            raise DataQualityError(f"{self.stage}: hard checks failed → {names}")
        return self


def check_ohlcv(df: pd.DataFrame, stage: str, freshness_days: int = 5) -> QualityReport:
    """Run the standard OHLCV assertion battery. Returns a report (call raise_if_failed())."""
    r = QualityReport(stage=stage)

    # L1 schema
    required = {"symbol", "date", "open", "high", "low", "close", "volume"}
    missing = required - set(df.columns)
    r.add("schema.columns", not missing, f"missing={missing or 'none'}", hard=True)
    if missing:
        return r  # nothing else is meaningful

    # L2 volume (row count sanity)
    r.add("volume.nonempty", len(df) > 0, f"rows={len(df)}", hard=True)

    # L3 value — nulls on keys, non-negative prices/volume
    key_nulls = int(df[["symbol", "date", "close"]].isna().any(axis=1).sum())
    r.add("value.key_nulls", key_nulls == 0, f"null key rows={key_nulls}", hard=True)
    neg = int((df[["open", "high", "low", "close"]] < 0).any(axis=1).sum())
    r.add("value.non_negative_price", neg == 0, f"negative-price rows={neg}", hard=True)
    neg_vol = int((df["volume"] < 0).sum())
    r.add("value.non_negative_volume", neg_vol == 0, f"negative-volume rows={neg_vol}", hard=False)

    # L4 business — OHLC internal consistency (high ≥ max(open,close) ≥ min ≥ low)
    bad_hl = int((df["high"] < df["low"]).sum())
    r.add("business.high_ge_low", bad_hl == 0, f"high<low rows={bad_hl}", hard=True)
    bad_range = int(
        ((df["high"] < df[["open", "close"]].max(axis=1)) | (df["low"] > df[["open", "close"]].min(axis=1))).sum()
    )
    r.add("business.oc_within_hl", bad_range == 0, f"OC-outside-HL rows={bad_range}", hard=False)

    # L4 duplicates
    dups = int(df.duplicated(subset=["symbol", "date"]).sum())
    r.add("business.no_duplicates", dups == 0, f"dup (symbol,date) rows={dups}", hard=True)

    # L5 freshness
    max_date = pd.to_datetime(df["date"]).max()
    stale = (datetime.today() - max_date) > timedelta(days=freshness_days + 3)  # +3 for weekends/holidays
    r.add("freshness.recent", not stale, f"max_date={max_date.date()}", hard=False)

    return r
