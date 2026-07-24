# TODO — stock_predictor

Living backlog. Last updated 2026-07-24. Priorities: 🔴 high · 🟡 medium · 🟢 low.
Context for finished work lives in `config/RESIDUAL_BIAS.md` and git history.

---

## 🔴 P0 — Substantive / strategic

### 0. Git repository CORRUPTED → REPAIRED (2026-07-24). ✅
Power loss (~Jul 24 20:06) left 3 zero-byte objects incl. `master`/`HEAD` `c434678…` (an aborted 6th
commit) → "bad object HEAD". Reflog was intact; last-good commit `cf241ae` and all 5 ancestors verified
whole (full tree readable, corrupt objects belonged only to the aborted commit). Repaired by pointing
`master` at `cf241ae`, deleting the 3 zero-byte stubs, and `git reset --mixed`. `git fsck` now clean,
full history restored, zero code loss. (Broken ref backed up to `scratchpad/master.broken.bak`.)
- Note: project `CLAUDE.md` is currently **untracked** (was likely in the aborted commit) — commit it.

### 1. The alpha question — RESOLVED (2026-07-24): GO. Genuine selection alpha.
**Decision: GO.** The "below equal-weight" headline was an *apples-to-oranges* artifact — `equal_weight`
is costless & tax-free and holds the whole universe, while `long_only` pays costs, 20% STCG on every
5-day gain, an impl-shortfall haircut, and is de-grossed to 0.5× in stormy regimes.
On identical footing the model wins decisively (all figures reconcile exactly to backtest_report.json):
- **Gross vs gross:** long-only Sharpe **1.81 vs EW 1.03**; ann return **53% vs 19%**.
- **Selection alpha** (LO_gross − EW_gross): **+26.7%/yr, Information Ratio 2.19, t-stat 6.22** (p≪0.01),
  positive in 63% of periods. This is real, statistically overwhelming edge, not survivor inflation.
- **The bridge (46%→16% net):** regime de-gross −12pp (Sharpe-neutral, pure return dilution) →
  costs −5pp → **20% STCG −13pp (the killer)**. Net Sharpe 0.936 is a *tax/construction* number.
- **Fix levers (counterfactual, weights reconciled):** drop return-dilutive de-grossing → net Sharpe
  0.936→0.950 but **+5.7pp return** (15.9%→21.7%); at 12.5% (LTCG-equiv) tax → **net Sharpe 1.18–1.19,
  beats EW**. Pretax Sharpe 1.53 ≫ EW.
- **Edge is monotonic in vol regime** (IC calm +0.005 → stormy +0.034) and **strongest in low-liquidity
  names** (IC +0.039 vs +0.019 high) and in IT/Consumer Svcs/Realty/Chemicals/Financials; negative in
  Construction/Telecom/Services. → de-grossing cuts exposure exactly where edge is best.
- **Follow-ups (moved to P1):** (a) remove/soften regime de-grossing; (b) benchmark net-vs-net (cost the
  EW rebalance too) so the scorecard is honest; (c) turnover/horizon work to cut the STCG hit; consider
  a sector-neutral or liquidity-aware book. Repro: `scratchpad/attribution.py`.

### 2. Backfill the delisted-name tail (29 names) — tighten the bias estimate
Kite's NSE instrument dump can't serve fully-delisted/merged names (DHFL, RCOM, HDFC-merged,
GRUH, INFRATEL, SYNDIBANK, GSKCONS, …). These are often the *worst* performers, so the current
survivorship delta (Sharpe −0.18) is a **lower bound**.
- [ ] Source their OHLCV from archived NSE bhavcopy (per-day zips, bulk-downloadable) or a vendor.
- [ ] Load → Bronze → rebuild Silver/Gold → retrain + re-backtest.
- **Acceptance:** all/most of the 29 have data; re-measured survivorship delta (expected larger).

---

## 🟡 P1 — Operational / hygiene

### 3. Verify first unattended daily run
`orchestration/daily_paper.py` is scheduled **weekdays 07:45 IST** (cron). Tomorrow is its first
real run (token refresh 06:30 → auto-sync → daily pull → loser top-up → rebalance → status).
- [ ] After first run, check `logs/daily_paper.out` — confirm the full chain worked unattended.

### 4. Retire old-project crons (if `claude_stock_predictor` is dead)
That project's `training.monitor` (running now), 07:30 pipeline, Sat full-train, and sentiment
prefetch are still active in crontab and consuming CPU/GPU/RAM.
- [ ] Confirm the old project is retired, then comment out its crontab entries (reversible).

### 5. Confirm daily_paper cron timing
Currently 07:45 IST (morning) → uses previous trading day's settled EOD bars (correct for a
next-morning signal). If post-close (evening) is preferred, change the cron time.

---

## 🟢 P2 — Lower priority / passive

### 6. Paper-trade forward track record
Let the daily loop accumulate; compare paper P&L vs backtest for matching dates (built-in
cross-check). This is the survivorship-free proof of the system.

### 7. Improve Wayback membership date resolution
Loser membership windows are approximate (4-year snapshot gap 2019→2023, exit dates collapse to
2023-08-11). More snapshot sources (or NSE circulars) would sharpen dates. Diminishing returns.

### 8. Data-quality follow-ups
- [ ] `TATAMOTORS`/`LTIM` missing from backfill — rename/symbol artifacts; add to
  `_RENAME_PREDECESSORS` or map to current symbols if their history matters.
- [ ] Consider archived-bhavcopy universe (top-N by turnover) as a fully survivorship-free
  alternative to index-membership reconstruction (evaluated; larger rebuild).

---

## ✅ Done (2026-07-24) — see git log + RESIDUAL_BIAS.md
Kite auth fixed + daily auto-login (kitePoc token → systemd .path sync); PIT membership mechanism
(bug fix, validation, listing-bound firewall); Wayback-diff → 81 losers backfilled; survivorship
bias quantified (long-only Sharpe 1.12→0.94); paper-trade operationalized; auto-backfill wired;
client-level Kite rate-gate + instruments disk-cache; git initialized (5 commits).
