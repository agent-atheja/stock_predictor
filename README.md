# Indian Stock Predictor

Cross-sectional predictor for Indian equities. Ranks the **NIFTY 100/200** universe each
day by **5-trading-day forward return** and trades the spread. Built clean-slate, fully
self-contained (no dependencies on any other project on this host).

## Design in one paragraph
Predict standardized **forward returns** (not price), framed as a **cross-sectional ranking**
problem, learned by a **LightGBM `LambdaRank`** model over ~40–70 point-in-time technical /
market-structure features. Validated with **leakage-safe walk-forward** (purge + embargo) and
a **cost-aware backtest** using a realistic Indian round-trip cost model. Deep sequence models
are an optional later add-on, not the foundation.

## Data flow (medallion)
```
Kite + jugaad bhavcopy ─► Bronze (raw OHLCV) ─► Silver (adjusted, PIT) ─► Gold (features+labels)
                                                                            │
                                                     Training / Backtest / Daily signals
```

## Layout
| Dir | Purpose |
|-----|---------|
| `core/` | config, logging, **parallelism** (`parallel.py`), datalake IO |
| `ingest/` | Kite client, jugaad deep backfill, daily incremental, PIT constituents |
| `datalake/` | `bronze/ silver/ gold/` partitioned parquet |
| `features/` | technicals, market structure, regime, assembler |
| `labels/` | 5-day forward-return label |
| `models/` | LightGBM ranker, training, registry |
| `validation/` | walk-forward (purge+embargo), metrics |
| `backtest/` | cost model, portfolio, engine, report |
| `signals/` | daily ranked signal generation |
| `execution/` | (LATER) paper → guarded live trading |

## Concurrency model
- **I/O-bound** multi-stock fetches → rate-limited **thread pool** (`core.parallel.io_map`,
  respects Kite's ~3 req/s).
- **CPU-bound** per-symbol feature builds / walk-forward folds → **process pool**
  (`core.parallel.cpu_map`, uses the box's cores).
- **Inference** is a single vectorized batch call over the whole universe — no per-stock loop.

## Setup

This host uses an **externally-managed** Python (PEP 668), so a plain `pip install` is blocked.
Pick ONE of these:

**Option A — venv (recommended, fully isolated):**
```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

**Option B — user site (matches the pre-installed libs in ~/.local):**
```bash
pip install --user --break-system-packages -r requirements.txt
# or just the runtime core:
pip install --user --break-system-packages jugaad-data kiteconnect tenacity python-dotenv duckdb
```

Then configure Kite (note: `<request_token>` is a placeholder — paste the real token from the
login redirect, no angle brackets):
```bash
cp secrets/.env.example secrets/.env      # fill in KITE_API_KEY / KITE_API_SECRET
python main.py kite-login                  # prints the login URL
python main.py kite-login PASTE_REAL_TOKEN # exchanges it for the daily access token
```

## Quickstart (NOW phase)
```bash
python main.py refdata          # real constituents + sectors + membership + corporate actions
python main.py kite-backfill    # deep history via Kite (RELIABLE; needs a valid token)   ← recommended
#   or, no Kite (best-effort, NSE throttles batches — may fail):
#   python main.py backfill
python main.py all              # silver → gold → train → backtest
python main.py signals          # today's ranked long/short book
```

### Backfill source: Kite vs jugaad
- **`kite-backfill`** (recommended): authenticated Kite API, chunked, ~3–4 min for 200×10y, does
  not get IP-blocked. Requires a working Kite token.
- **`backfill`** (jugaad/NSE): free, no auth, but NSE **IP-throttles batches** — fine for a few
  symbols, unreliable for the full universe. Use only if you have no Kite subscription.
