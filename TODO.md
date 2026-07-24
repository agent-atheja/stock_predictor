# TODO — stock_predictor

Living backlog. Last updated 2026-07-24. Priorities: 🔴 high · 🟡 medium · 🟢 low.
Context for finished work lives in `config/RESIDUAL_BIAS.md` and git history.

---

## 🔴 P0 — Substantive / strategic

### 1. The alpha question — does the model actually beat naive baselines?
After survivorship correction, **long-only Sharpe fell to 0.936, now BELOW equal-weight (1.033)**.
The model no longer beats a naive equal-weight portfolio on risk-adjusted return once survivor
inflation is removed. Ship criterion still passes only vs the (near-zero) momentum baseline.
- [ ] Attribution: where does the model add vs lose return relative to equal-weight?
- [ ] Segment/regime slices: is there edge in specific regimes, sectors, or liquidity buckets?
- [ ] Decide: improve (features/model), reframe as smart-beta, or park the strategy.
- **Acceptance:** a clear go/no-go with evidence — genuine alpha vs equal-weight, or an honest "no."

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
