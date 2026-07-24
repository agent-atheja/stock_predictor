# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

Clean-slate, **fully self-contained** cross-sectional predictor for Indian equities: ranks the
NIFTY 100/200 universe daily by **5-trading-day forward return** (LightGBM `LambdaRank`), validated
with leakage-safe walk-forward and a cost-aware backtest. Do **not** import from or depend on the
sibling project `/mnt/stock_nvme_new/claude_stock_predictor` (deliberately not reused). See
`README.md` for the design paragraph and directory table; this file covers what isn't obvious from
a single file.

## Commands

Everything routes through the `main.py` CLI (run inside the venv: `source .venv/bin/activate`):

```bash
python main.py refdata        # constituents + sectors + PIT membership + corporate actions
python main.py kite-backfill  # deep history via Kite (reliable; needs a valid token)
python main.py all            # silver → gold → train → backtest
python main.py train [--hpo]  # walk-forward train (+ optional Optuna HPO)
python main.py backtest       # cost/tax-aware backtest → models/registry/backtest_report.json
python main.py signals        # today's ranked long/short book
python main.py synth          # write SYNTHETIC bronze → run the whole pipeline with no real data
```

Kite auth (daily token) and self-check:
```bash
python -m ingest.kite_client login PASTE_REQUEST_TOKEN   # exchange; maps 'Invalid checksum' etc.
python -m ingest.kite_client verify                       # one-line PASS/FAIL on the live session
```

Tests (pytest; no config file — just run the dir). Run a single test:
```bash
python -m pytest tests/ -q
python -m pytest tests/test_pit_membership.py::test_replay_remove_closes_open_interval -q
```
`tests/synthetic_data.py` generates a fake universe so the full pipeline (silver→gold→train→
backtest) can be exercised end-to-end with **no Kite and no real data** — use it to validate
pipeline changes.

## Configuration & paths

- **Single source of truth:** `config/config.yaml` (sections: `universe, data, label, features,
  model, walkforward, backtest, concurrency, kite, logging`). Load via `core.config.load_config()`.
- `core.config.resolve(path)` resolves relative paths against the **repo root** (`ROOT`), **not the
  cwd**. Config values like `datalake/bronze` are repo-relative — this is why cron/systemd jobs must
  still work regardless of working directory. Prefer `resolve(cfg...path)` over raw paths.
- Secrets: `core.config.secret(env_name)` reads from `secrets/.env`, loaded with `override=True` so
  the project's `.env` **wins over any stale `KITE_*` exported in the shell/bashrc**.

## Data architecture (medallion) & the one data-safety invariant

`Kite/bhavcopy → Bronze (raw OHLCV) → Silver (split-adjusted + PIT is_member tag) → Gold
(features + 5-day forward-return label)`, all Hive-partitioned parquet under `datalake/` (gitignored,
regenerable). Written via `core.io.write_partitioned`.

**⚠️ `write_partitioned` OVERWRITES each touched `dt=` partition wholesale.** Therefore you must
only ever write the **full universe** for a given date. Backfilling a *subset* of symbols wipes the
other symbols out of those date-partitions. `ingest/kite_backfill.py` and
`orchestration/backfill_losers.py` rewrite the full `all_symbols_ever()` set for this reason;
`ingest/daily_incremental.py` is safe because it writes all current members for recent dates only.

## Point-in-time membership & survivorship (correctness-critical)

The tradable universe on a date must be who was *actually* in the index then. `is_member` is tagged
per (date, symbol) in Silver from `config/nifty200_membership_history.csv`, and
`models.dataset.apply_universe_filter` filters Gold on it.

`ingest.reference_data.build_pit_membership()` builds that CSV in three layers:
**seed (current constituents) → replay `config/reconstitution_events.csv` (validated add/remove) →
listing-bound clamp** (a symbol can't be a member before its first Bronze bar).
`ingest/wayback_membership.py` sources real historical composition from Wayback snapshots of the
constituent CSV and writes reconstitution events. `all_symbols_ever()` (incl. delisted losers) is
the backfill target set. Read `config/RESIDUAL_BIAS.md` before touching anything survivorship-
related — it is the honest ledger of what's corrected, what's a lower bound, and why.

## Leakage safety (do not break)

`validation/walkforward.py` `make_folds` enforces **purge + embargo**: train ends `embargo_days`
before each test window, and embargo is auto-bumped to `horizon+1` if configured too small. Any
change to feature timing, label horizon, or fold construction must preserve "no train label matures
into the test window." This is the model's core correctness property.

## Kite integration specifics

- **Token is daily.** In production it's auto-refreshed by the sibling `kitePoc` (Playwright+TOTP,
  06:30 cron) and copied into `secrets/access_token.txt` by a systemd `--user` path watcher
  (`~/.config/systemd/user/kite-token-sync.path`). Don't reimplement login here.
- **Rate limits are real and punishing.** `ingest/kite_client.py` has a process-wide `RateLimiter`
  that gates *every* Kite call **including tenacity retries** (retries otherwise bypass `io_map`'s
  limiter and cause `Too many requests` cascades). The instruments dump is **disk-cached per day**
  (`datalake/cache/`) — fetch it at most once/day. Aggressive calling triggers a temporary
  account-level cooldown; back off, don't hammer.

## Concurrency

`core.parallel`: `io_map` = rate-limited **thread pool** for I/O-bound multi-stock fetches;
`cpu_map` = **process pool** for per-symbol feature builds / walk-forward folds. Inference is a
single vectorized batch over the whole universe — never a per-stock loop. RAM is the binding host
constraint (~30GB) — avoid loading all partitions into one frame when a scan or per-partition pass
works.

## Operational notes

- `orchestration/daily_paper.py` is the unattended daily driver (cron, weekdays 07:45 IST):
  Kite-refresh → silver/gold → loser top-up → paper rebalance → drift → status; it **skips
  gracefully when Kite is offline**. `orchestration/backfill_losers.py` is idempotent + self-
  throttled (bails if the instruments endpoint is cooled down).
- Backtest/model artifacts land in `models/registry/` (gitignored). `config/*.csv` (constituents,
  membership, reconstitution events, corporate actions) **are** tracked — they're inputs, not data.
- Python here is externally-managed (PEP 668): use the repo `.venv`, or
  `pip install --user --break-system-packages`.
