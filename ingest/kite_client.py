"""Self-contained Zerodha Kite Connect client.

Everything (keys, access token) lives inside this project's `secrets/` dir — no external
project paths. Kite access tokens expire daily; run `python -m ingest.kite_client login`
once per day (or wire it into the scheduler) to refresh.

Responsibilities:
  • Authenticated `KiteConnect` handle (token cached at secrets/access_token.txt).
  • `historical(...)` — OHLCV candles for one instrument, retried with backoff.
  • Instrument-token lookup for NSE equity symbols.

Deep multi-year backfill goes through jugaad bhavcopy (see historical_backfill.py); Kite is
the authoritative source for recent/ongoing daily data and (LATER) intraday + live.
"""
from __future__ import annotations

import sys
from datetime import date, datetime
from functools import lru_cache
from pathlib import Path

from core.config import load_config, resolve, secret
from core.logging_setup import get_logger
from core.parallel import RateLimiter

log = get_logger(__name__)

# Global gate on ALL Kite REST calls (historical + instruments), shared across worker threads.
# Kite's historical endpoint hard-caps ~3 req/s; crucially this gates tenacity RETRIES too, which
# otherwise bypass io_map's limiter and cause 429 ("Too many requests") cascades under load.
_kite_limiter: "RateLimiter | None" = None


def _rate_gate() -> RateLimiter:
    global _kite_limiter
    if _kite_limiter is None:
        _kite_limiter = RateLimiter(load_config().concurrency.kite_rate_limit_per_sec)
    return _kite_limiter


def _load_env() -> None:
    """Load secrets/.env into the environment (self-contained, dotenv optional)."""
    try:
        from dotenv import load_dotenv

        # override=True so THIS project's secrets/.env always wins over any stale KITE_* vars
        # exported in the shell (~/.bashrc from other projects). Self-contained = .env is truth.
        load_dotenv(resolve("secrets/.env"), override=True)
    except Exception:  # dotenv not installed / no file — fall back to real env
        pass


def _token_path() -> Path:
    return resolve(load_config().kite.access_token_path)


def _read_token() -> str | None:
    p = _token_path()
    if p.exists():
        tok = p.read_text().strip()
        return tok or None
    return None


def login(request_token: str | None = None) -> str:
    """Interactive-ish daily login. Prints the login URL; exchanges request_token for an
    access token and caches it. Pass request_token= or set KITE_REQUEST_TOKEN in secrets/.env.
    """
    from kiteconnect import KiteConnect

    _load_env()
    cfg = load_config()
    api_key = secret(cfg.kite.api_key_env)
    api_secret = secret(cfg.kite.api_secret_env)
    if not api_key or not api_secret:
        raise RuntimeError("Set KITE_API_KEY / KITE_API_SECRET in secrets/.env")

    api_key = api_key.strip()
    api_secret = api_secret.strip()
    kite = KiteConnect(api_key=api_key)
    request_token = (request_token or secret("KITE_REQUEST_TOKEN") or "").strip()
    if not request_token:
        print("\n1) Open this URL, log in, and copy the `request_token` from the redirect:")
        print("   ", kite.login_url())
        print("2) Re-run:  python -m ingest.kite_client login <request_token>\n")
        raise SystemExit(2)

    try:
        data = kite.generate_session(request_token, api_secret=api_secret)
    except Exception as exc:  # noqa: BLE001 — turn Kite's opaque errors into an actionable one
        msg = str(exc).lower()
        if "checksum" in msg:
            raise SystemExit(
                "\nKite login failed: 'Invalid checksum'.\n"
                "This is NOT a code bug — it means KITE_API_SECRET in secrets/.env does not match "
                f"KITE_API_KEY (…{api_key[-4:]}).\n"
                "Fix: open https://developers.kite.trade/apps → your app → copy the API secret "
                "EXACTLY (no spaces/newlines) into secrets/.env, then re-run login with a FRESH "
                "request_token (they are single-use and expire in minutes).\n"
            ) from exc
        if "token" in msg or "expired" in msg:
            raise SystemExit(
                "\nKite login failed: request_token invalid or expired.\n"
                "request_tokens are single-use and expire within minutes. Re-open the login URL, "
                "grab a new request_token, and re-run login immediately.\n"
            ) from exc
        raise SystemExit(f"\nKite login failed: {exc}\n") from exc

    access_token = data["access_token"]
    tp = _token_path()
    tp.parent.mkdir(parents=True, exist_ok=True)
    tp.write_text(access_token)
    log.info("Kite access token refreshed and cached at %s (user: %s)",
             tp, data.get("user_name", "?"))
    return access_token


def verify() -> bool:
    """One-command credential self-check. Confirms the cached daily token is live by hitting the
    authenticated profile endpoint. Prints a clear PASS/FAIL and the reason. Returns True on success.
    """
    _load_env()
    cfg = load_config()
    api_key = secret(cfg.kite.api_key_env)
    if not api_key:
        print("FAIL: KITE_API_KEY not set in secrets/.env")
        return False
    if not _read_token():
        print("FAIL: no cached access token. Run: python -m ingest.kite_client login")
        return False
    try:
        prof = get_kite().profile()
    except Exception as exc:  # noqa: BLE001
        print(f"FAIL: token rejected by Kite ({exc}). Re-run daily login: "
              "python -m ingest.kite_client login")
        return False
    print(f"PASS: Kite session live — user '{prof.get('user_name','?')}' "
          f"({prof.get('user_id','?')}), api_key …{api_key[-4:]}")
    return True


@lru_cache(maxsize=1)
def get_kite():
    """Return an authenticated KiteConnect handle using the cached daily token."""
    from kiteconnect import KiteConnect

    _load_env()
    cfg = load_config()
    api_key = secret(cfg.kite.api_key_env)
    token = _read_token()
    if not api_key or not token:
        raise RuntimeError(
            "No Kite session. Set keys in secrets/.env and run: python -m ingest.kite_client login"
        )
    kite = KiteConnect(api_key=api_key)
    kite.set_access_token(token)
    return kite


def _instruments_cache_path() -> Path:
    return resolve("datalake/cache") / f"instruments_NSE_{date.today():%Y-%m-%d}.parquet"


@lru_cache(maxsize=1)
def _instruments_nse():
    """NSE instrument dump (tradingsymbol → row), cached to DISK per day.

    Kite's instruments endpoint is a heavy full-exchange dump with its own rate limit; the docs say
    fetch it at most once a day and cache. We persist today's dump to datalake/cache and reuse it
    across processes — this is what stops repeated runs from tripping 'Too many requests'."""
    import pandas as pd

    cache = _instruments_cache_path()
    if cache.exists():
        df = pd.read_parquet(cache)
        log.info("Loaded NSE instruments from daily disk cache (%d rows) %s", len(df), cache.name)
    else:
        _rate_gate().acquire()  # gate the heavy dump too
        rows = get_kite().instruments("NSE")
        df = pd.DataFrame(rows)
        cache.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(cache, index=False)
        log.info("Fetched + disk-cached NSE instruments (%d rows) → %s", len(df), cache.name)
    return {r["tradingsymbol"]: r for r in df.to_dict("records")}


def instrument_token(symbol: str) -> int | None:
    row = _instruments_nse().get(symbol)
    return int(row["instrument_token"]) if row else None


def historical(symbol: str, start: date, end: date, interval: str = "day") -> list[dict]:
    """Fetch OHLCV candles for one symbol between dates. Retries with exponential backoff.

    Returns list of dicts: {date, open, high, low, close, volume}. Empty on failure so a
    single bad symbol never breaks a parallel batch (caller logs the gap).
    """
    from tenacity import retry, stop_after_attempt, wait_exponential

    token = instrument_token(symbol)
    if token is None:
        log.warning("historical: no instrument token for %s", symbol)
        return []
    kite = get_kite()

    @retry(stop=stop_after_attempt(4), wait=wait_exponential(multiplier=1, min=1, max=20), reraise=True)
    def _call():
        _rate_gate().acquire()  # gate every attempt (incl. retries) → true ≤rate/s to Kite
        return kite.historical_data(token, start, end, interval)

    candles = _call()
    for c in candles:
        c["symbol"] = symbol
        c["date"] = c["date"].date() if isinstance(c["date"], datetime) else c["date"]
    return candles


if __name__ == "__main__":
    # CLI:  python -m ingest.kite_client login [request_token]
    #       python -m ingest.kite_client verify
    cmd = sys.argv[1] if len(sys.argv) >= 2 else ""
    if cmd == "login":
        login(sys.argv[2] if len(sys.argv) > 2 else None)
    elif cmd == "verify":
        raise SystemExit(0 if verify() else 1)
    else:
        print("usage: python -m ingest.kite_client [login [request_token] | verify]")
