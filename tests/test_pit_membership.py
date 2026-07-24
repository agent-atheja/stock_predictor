"""Tests for the PIT-membership survivorship firewall (ingest/reference_data).

Covers the pure logic — event validation, event replay (remove / join-date correction / re-add),
and the listing-bound clamp — without touching the real config files.
"""
from __future__ import annotations

import pandas as pd
import pytest

from ingest import reference_data as rd

_HS = pd.Timestamp("2015-01-01")


def _seed(*symbols: str) -> pd.DataFrame:
    return pd.DataFrame({"symbol": list(symbols), "valid_from": _HS, "valid_to": rd._FAR_FUTURE})


def test_validate_events_rejects_bad_action():
    bad = pd.DataFrame([{"symbol": "X", "action": "delete", "effective_date": "2020-01-01"}])
    with pytest.raises(ValueError, match="invalid action"):
        rd._validate_events(bad, "t")


def test_validate_events_rejects_unparseable_date():
    bad = pd.DataFrame([{"symbol": "X", "action": "add", "effective_date": "not-a-date"}])
    with pytest.raises(ValueError, match="unparseable"):
        rd._validate_events(bad, "t")


def test_validate_events_requires_columns():
    with pytest.raises(ValueError, match="missing columns"):
        rd._validate_events(pd.DataFrame({"symbol": ["X"]}), "t")


def test_replay_remove_closes_open_interval():
    seed = _seed("A")
    ev = rd._validate_events(
        pd.DataFrame([{"symbol": "A", "action": "remove", "effective_date": "2019-03-29"}]), "t")
    hist = rd._replay_events(seed, ev, _HS)
    row = hist[hist.symbol == "A"].iloc[0]
    assert row.valid_from == _HS and row.valid_to == pd.Timestamp("2019-03-29")


def test_replay_add_corrects_seeded_join_date_not_duplicate():
    # A current member seeded at 2015 that verifiably joined in 2021 → correct valid_from, one row.
    seed = _seed("B")
    ev = rd._validate_events(
        pd.DataFrame([{"symbol": "B", "action": "add", "effective_date": "2021-09-30"}]), "t")
    hist = rd._replay_events(seed, ev, _HS)
    rows = hist[hist.symbol == "B"]
    assert len(rows) == 1
    assert rows.iloc[0].valid_from == pd.Timestamp("2021-09-30")


def test_replay_remove_then_add_is_two_intervals():
    seed = _seed("C")
    ev = rd._validate_events(pd.DataFrame([
        {"symbol": "C", "action": "remove", "effective_date": "2017-09-29"},
        {"symbol": "C", "action": "add", "effective_date": "2022-03-31"},
    ]), "t")
    hist = rd._replay_events(seed, ev, _HS).sort_values("valid_from")
    rows = hist[hist.symbol == "C"]
    assert len(rows) == 2
    assert list(rows.valid_to) == [pd.Timestamp("2017-09-29"), rd._FAR_FUTURE]


def test_clamp_raises_valid_from_to_listing_date():
    seed = _seed("IPO")
    bounds = pd.Series({"IPO": pd.Timestamp("2021-07-23")})
    hist = rd._clamp_to_listing_bounds(seed, bounds)
    assert hist[hist.symbol == "IPO"].iloc[0].valid_from == pd.Timestamp("2021-07-23")


def test_clamp_drops_collapsed_interval():
    # An interval that ends before the listing bound collapses and must be dropped.
    seed = pd.DataFrame([{"symbol": "OLD", "valid_from": _HS, "valid_to": pd.Timestamp("2016-01-01")}])
    bounds = pd.Series({"OLD": pd.Timestamp("2020-01-01")})
    hist = rd._clamp_to_listing_bounds(seed, bounds)
    assert hist.empty


def test_clamp_noop_when_listed_before_history_start():
    seed = _seed("LEGACY")
    bounds = pd.Series({"LEGACY": pd.Timestamp("2010-01-01")})  # listed before window
    hist = rd._clamp_to_listing_bounds(seed, bounds)
    assert hist[hist.symbol == "LEGACY"].iloc[0].valid_from == _HS
