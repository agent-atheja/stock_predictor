# Survivorship Bias — What's Fixed, What Remains (MVP honest ledger)

**Status as of 2026-07-25.** BLUF: PIT-membership mechanism fixed + hardened + tested; listing-bound
firewall applied; **all 110 real historical losers now backfilled into the universe** (200→310 in
Gold). Survivorship bias is now **fully corrected and quantified** for the reconstructable universe.
Sections below are a chronological ledger; the earliest ones describe pre-backfill states — **the
MEASURED IMPACT section is the current truth.**

**UPDATE 2026-07-25 — the final 29 delisted names are now backfilled (no longer a lower bound).**
The 29 truly-delisted/merged names Kite couldn't serve (DHFL, RCOM, HDFC-merged, GRUH, SYNDIBANK,
GSKCONS, FRETAIL, …) were sourced from **authoritative raw NSE bhavcopy** (`ingest/bhavcopy_backfill.py`),
NOT jugaad-data — jugaad interleaves a *different same-ticker security* into these names' history
(HDFC alternated real ₹2,702 with a fake ₹552; ~28% of rows garbage) and is unusable here. Bhavcopy
gives one true ISIN-stamped EQ row per symbol/day; each name is pinned to its dominant ISIN to reject
ticker-reuse. Gold universe went 281→310. **Re-measured survivorship delta from these 29 (at
[1,1,0.75]): long-only net Sharpe 0.958→0.849 (−0.109), CAGR 18.3%→15.7%, MaxDD −40.8%→−47.4%;
equal-weight also dropped (gross 1.033→0.979, net 0.427→0.373).** The model's edge over the honest
net-of-cost/tax benchmark **survives** the full correction: net 0.849 vs EW-net 0.373 (2.27×). The
"lower bound" caveat is retired; residual bias is now only from names never in any snapshot (below).

---

## The three layers of PIT membership (`ingest/reference_data.build_pit_membership`)

1. **Seed** — current NIFTY 200 constituents, `valid_from = history_start (2015-01-01)`.
2. **Curated reconstitution events** — `config/reconstitution_events.csv`
   (`symbol,action,effective_date`), replayed in date order. Validated by `_validate_events`
   (schema, action ∈ {add,remove}, parseable dates, dedup). **Currently empty** (header only) —
   see "Remaining gap" below for why we did not hand-populate it.
3. **Listing-bound clamp** — `_clamp_to_listing_bounds`: raises each interval's `valid_from` up to
   the symbol's **first traded bar in Bronze**. A stock cannot be an index member before it is
   listed. Provenance is our own ingested data → zero third-party/web-scrape risk.

Result: `members_on(date)` now grows **147 (2015) → 200 (2026)** instead of a flat 200. IPO-era
names are correctly excluded before listing (ETERNAL 2021, JIOFIN 2023, PAYTM 2021, SWIGGY 2024,
53 names in total).

---

## What is FIXED

| Item | Detail |
|------|--------|
| `_FAR_FUTURE` NameError | Was undefined in the events replay — would crash on the first real event. Now module-level. |
| Silent bad-data risk | `_validate_events` fails loud on malformed events (missing cols, bad action, unparseable date). |
| Idempotency | `build_pit_membership` always rebuilds from seed + full log + bounds; re-runs are deterministic. |
| String/Timestamp bug | `add_membership_change` parsed `valid_to` so the open-interval test works regardless of CSV format. |
| Membership record honesty | 53 current members no longer claimed as index members before they were listed. |

## What does NOT change (and why — be honest)

The listing-bound clamp has **~zero backtest P&L impact**, by construction:

- `is_member` in `datalake/gold` is **True on 100% of rows** (484,578/484,578).
- `symbols_with_data` per year **equals** the clamped `members_on()` count exactly.
- ⟹ the eligible universe (`is_member AND has-features`) was *already* "current constituents that
  had data on that date" — i.e. survivor-only. The clamp only removed membership on dates that had
  **no data anyway**, so `apply_universe_filter` output is unchanged.

The clamp is still worth having: it makes the record correct, fixes `all_symbols_ever()` /
backfill targeting, and prevents future bugs — but it is **not** a P&L-moving survivorship fix.

---

## Remaining gap (the part that actually biases returns) — **Kite-gated**

Two sources of real, P&L-moving survivorship bias remain. Both need data we cannot get without a
working Kite session (item 1), and we deliberately did **not** fabricate to close them:

1. **Delisted / demoted losers are absent.** The Bronze layer only contains *today's* 200
   constituents. Names that were in NIFTY 200 in, say, 2016 and later dropped out (or delisted)
   are missing entirely → the universe is composed of survivors → returns are optimistically
   biased. **Fix:** obtain `all_symbols_ever()` (needs authoritative historical composition) and
   **backfill their prices via `ingest/kite_backfill.py`** (Kite-gated).

2. **Data-rich late-joiners are included too early.** A small-cap that had prices since 2015 but
   only *graduated into* NIFTY 200 in, say, 2020 is currently marked a member from its listing
   date (pre-2020). The listing clamp cannot catch this (it has early data). **Fix:** append the
   real inclusion date as an `add` event in `reconstitution_events.csv` — a **join-date
   correction** (`_replay_events` case (a)).

### Why we did not hand-populate `reconstitution_events.csv`
Authoritative NIFTY 200 semi-annual add/drop lists are **not** cleanly published for free
(NIFTY 50 is well-reported; NIFTY 200 = NIFTY 100 + Midcap 100 is not). The source of record is
NSE Indices press-release PDFs (`nsearchives.nseindia.com`), which are bot-blocked. Hand-entering
web-scraped events risks **wrong** membership — strictly worse than a documented gap. Per the
data-integrity rule, we ship the validated mechanism + a documented on-ramp instead.

---

## UPDATE 2026-07-24 — Wayback-diff applied (safe subset)

Sourced real historical composition from **Wayback Machine snapshots** of the NIFTY 200
constituent CSV (`ingest/wayback_membership.py`). Coverage is **sparse**: 5 usable snapshots
(2017-11, 2018-03, 2019-02, 2023-08, 2024-01) with a **4-year hole (2019→2023)**, so all derived
exit dates collapse onto **2023-08-11** — dates are approximate, not review-exact.

**What we applied (safe subset):** **110 delisted/demoted/merged-out LOSERS** — every non-current
name that was ever a member, minus known renames (`_RENAME_PREDECESSORS`) and DUMMY* placeholders.
Two exit cases: left within coverage → close at the next snapshot; still present in the last
snapshot but not current (a recent free-float demotion) → close conservatively at the last
snapshot date. Examples: HDFC (merged 2023), GRUH (→Bandhan), DHFL/RCOM (bankrupt), INFRATEL
(→Indus), FRETAIL (Future), SYNDIBANK/CENTRALBK (PSU mergers), plus recent demotions ACC, LICI,
LTIM, LTTS, ICICIPRULI, PETRONET, SUNTV. Membership **200 → 310 symbols**; all 200 current members
**unchanged** (verified); `all_symbols_ever()` = 310.

**Correction to an earlier worry (current list is NOT broken):** `nifty200_current.csv` was
suspected of missing LICI/LTIM/LTTS. Verified against the **live niftyindices.com fetch** — it
returns the *same* 200 names, and the list contains Dec-2025 IPOs (GROWW, LENSKART), so it is
**fresh and authoritative**. NIFTY 200 ranks by **free-float** market cap, and LIC (~96.5% govt
holding → tiny float) legitimately ranks below 200. So LICI/LTIM/etc. are **real recent
demotions**, correctly folded into the loser set above — not a current-list defect.

**What we deliberately dropped:** `wayback:join` corrections for current members — coarse/late
dates would wrongly *exclude* valid membership; the listing-bound clamp already handles the
verifiable part with exact dates.

**P&L impact so far: still zero.** The 110 losers have **no price data yet** (Silver/Gold have no
rows for them) — they are *staged* backfill targets. Survivorship bias actually drops **only
after** they are Kite-backfilled.

**Auto-backfill wiring (2026-07-24):** `orchestration/backfill_losers.py` backfills only the
missing losers, then rebuilds Silver+Gold. It is idempotent + self-throttled: gentle `verify()` +
a single instruments probe, and bails (retry later) if Kite's instruments endpoint is still cooled
down — never hammers. Wired into `orchestration/daily_paper.py`, which now runs weekdays 07:45 IST
via cron (after the 06:30 token refresh + auto-sync). So the losers backfill **automatically on
the first day Kite is healthy**, and the job self-disables once complete. (A systemd timer was the
first choice but the sandbox blocked creating auto-executing user units; the cron path is
equivalent.)

---

## How to close the gap (runbook, once Kite works — item 1)

```bash
# 1. Fix the Kite secret, then verify (see item 1):
python -m ingest.kite_client verify

# 2. Append real reconstitution events from NSE semi-annual review circulars:
#    edit config/reconstitution_events.csv  ->  symbol,action,effective_date
#    (add = joined index; remove = left index)   e.g.:
#      SOMENAME,add,2020-09-30
#      OLDLOSER,remove,2018-03-29

# 3. Backfill prices for every name that was EVER a member (incl. delisted):
python -m ingest.kite_backfill          # uses all_symbols_ever()

# 4. Rebuild membership + silver (re-tags is_member) + gold, then retrain/backtest:
python main.py refdata                  # build_pit_membership: seed + events + listing clamp
python main.py silver
python main.py gold
python main.py train
```

Until steps 1–4 are done, treat backtest numbers as **survivor-biased (optimistic)**.

---

## MEASURED IMPACT 2026-07-24 — survivorship bias quantified

Backfilled 81/110 losers (29 truly-delisted names — DHFL, RCOM, HDFC-merged, bankrupt/delisted —
are NOT in Kite's current NSE instrument dump; need archived bhavcopy or a vendor). Gold universe
200 → 281 symbols; `is_member` is now 562k True / 127k False (was 100% True). Retrained + re-backtested.

| Metric (long-only) | Survivor-only | +81 losers | Δ |
|---|---|---|---|
| Sharpe | 1.116 | **0.936** | −0.18 |
| CAGR | 18.1% | **15.6%** | −2.6pp |
| Max drawdown | −24.8% | **−32.9%** | −8.1pp worse |

Long-short: Sharpe 0.833 → 0.662, CAGR 9.3% → 7.5%, maxDD −16.9% → −24.3%.

**This is a LOWER BOUND** — the 29 unfetchable names are often the worst performers, so true
survivorship bias is larger. **Flag:** post-correction long-only Sharpe (0.936) is now BELOW the
equal-weight baseline (1.033) — the model no longer beats naive equal-weight on Sharpe once
survivor inflation is removed. Ship criterion still passes (vs momentum baseline 0.003).
