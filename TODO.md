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
- **Follow-ups:** (a)✅ soften regime de-grossing; (b)✅ honest net-vs-net benchmark; (c) turnover/horizon
  work to cut STCG hit (bigger effort); consider sector-neutral or liquidity-aware book. Repro:
  `scratchpad/attribution.py`.

**Fix-lever PROTOTYPE (2026-07-24, branch `exp/fix-levers`, real engine, OOS predictions):**
- (b) Added `equal_weight_net` baseline — EW charged the *same* cost + 20% STCG. **EW-net Sharpe = 0.427**
  (turnover only 1.3%/period, so the drop from gross 1.03 is almost pure tax). Honest scorecard:
  **model long-only NET 0.936 vs EW-net 0.427 — model wins >2×.** The fair EW sits between the costless
  1.03 (too generous) and taxed-rebalanced 0.43 (too harsh, a real holder wouldn't rebalance q5d); model
  net clears the midpoint. The "below equal-weight" headline is dead.
- (a) `regime_gross_scaling` grid — it's a risk/return dial, not a free lunch:
    `[1,1,0.5]`(cur) Sharpe 0.936 / CAGR 15.6% / MaxDD −32.9%
    `[1,1,0.75]`     Sharpe **0.958** / CAGR 18.3% / MaxDD −40.8%   ← Sharpe-best, RECOMMENDED
    `[1,1,1.0]`      Sharpe 0.950 / CAGR 20.9% / MaxDD −48.4%
  Branch left at `[1,1,0.75]` pending a risk-appetite sign-off; revert is a one-line dial. Code change
  (EW-net baseline) is keep-regardless.

### 2. Backfill the delisted-name tail (29 names) — DATA DONE (2026-07-25); retrain running
Kite can't serve fully-delisted/merged names (DHFL, RCOM, HDFC-merged, …) — the *worst* performers,
so the pre-backfill survivorship delta was a **lower bound**. Now backfilled from the authoritative source.
**What was done:**
- **jugaad-data is UNUSABLE for the heavy delisted names** — it *interleaves a different same-ticker
  security* day-by-day (HDFC alternated real ₹2,702 @ vol 4.4M with a fake ₹552 @ 184k; PEL fake +4243%;
  also DHFL/IBULHSGFIN/IBVENTURES — HDFC ~28% of rows garbage). Simple outlier/median cleaning can't
  salvage it. Superseded module deleted.
- **Built `ingest/bhavcopy_backfill.py`** — pulls raw NSE bhavcopy (one true EQ row per symbol per day,
  ISIN-stamped → no interleaving). Handles both formats (legacy `cmDDMONYYYYbhav` + UDiFF), disk-cached
  per day, multi-pass gentle retry (aborts rather than merge a gap), pins each name to its **dominant
  ISIN** (rejected 813 GUJGASLTD + 303 WELSPUNIND rows from a *different* company reusing the ticker),
  tolerant date parse (one day used a 2-digit year), partition-safe merge into Bronze.
- **All 29/29 backfilled clean**: 52,603 authoritative rows → Bronze now 310 symbols / 760,468 rows.
  HDFC ₹2,724 @ merger (42M vol, real) max 1-day 13% (was 992%); PEL max 45% (was 4243%). Silver + Gold
  rebuilt (DQ PASS, leakage tripwire PASS). Delisted names now carry `is_member` in their PIT windows
  (HDFC 1,403 member-days → 2023 merger, DHFL 886 → 2021, etc.).
- **Retrain + re-backtest RUNNING** (bg): 97 walk-forward folds. Baseline to beat (pre-backfill [1,1,0.75]):
  long-only Sharpe 0.958 / CAGR 18.3% / MaxDD −40.8% (EW 1.033, EW-net 0.427). **Delta TBD on completion.**
- **Acceptance:** ✅ trustworthy OHLCV for all 29; re-measured survivorship delta — PENDING retrain.

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
