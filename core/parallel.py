"""Concurrency layer — parallel/async workers for multi-stock operations.

Two distinct workloads, two distinct pools (picking the wrong one is a classic mistake):

  • I/O-bound  (Kite / jugaad / yfinance network fetches across hundreds of symbols)
      → ThreadPoolExecutor + a token-bucket rate limiter. Threads are right here because
        the work is waiting on the network, and we MUST respect Kite's ~3 req/s cap or get
        throttled/banned. GIL is not a bottleneck for I/O.

  • CPU-bound  (per-symbol feature engineering, walk-forward fold training)
      → ProcessPoolExecutor across the box's cores. True parallelism, sidesteps the GIL.

Inference itself is NOT parallelized per-stock: the cross-sectional LightGBM model scores
the entire universe in ONE vectorized call (predict on the full day's feature matrix), which
is faster than any per-stock fan-out. Use these pools for ingestion and feature build.
"""
from __future__ import annotations

import threading
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from typing import Callable, Iterable, Sequence, TypeVar

from core.config import load_config
from core.logging_setup import get_logger

log = get_logger(__name__)

T = TypeVar("T")
R = TypeVar("R")


class RateLimiter:
    """Thread-safe token bucket. Blocks callers so aggregate rate ≤ max_per_sec."""

    def __init__(self, max_per_sec: float):
        self._min_interval = 1.0 / max_per_sec if max_per_sec > 0 else 0.0
        self._lock = threading.Lock()
        self._next_allowed = 0.0

    def acquire(self) -> None:
        if self._min_interval <= 0:
            return
        with self._lock:
            now = time.monotonic()
            wait = self._next_allowed - now
            if wait > 0:
                time.sleep(wait)
                now = time.monotonic()
            self._next_allowed = max(now, self._next_allowed) + self._min_interval


def io_map(
    fn: Callable[[T], R],
    items: Sequence[T],
    *,
    workers: int | None = None,
    rate_limit_per_sec: float | None = None,
    desc: str = "io_map",
) -> list[R]:
    """Run an I/O-bound fn over items on a rate-limited thread pool. Order preserved.

    Failures are logged and returned as None so one bad symbol never kills the batch
    (idempotent re-runs pick up the gaps).
    """
    cfg = load_config()
    workers = workers or cfg.concurrency.io_workers
    rate = rate_limit_per_sec if rate_limit_per_sec is not None else cfg.concurrency.kite_rate_limit_per_sec
    limiter = RateLimiter(rate)

    def _wrapped(item: T) -> R | None:
        limiter.acquire()
        try:
            return fn(item)
        except Exception as exc:  # noqa: BLE001 — resilience is the point here
            log.warning("%s: item %r failed: %s", desc, item, exc)
            return None

    results: list[R | None] = [None] * len(items)
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futures = {ex.submit(_wrapped, it): i for i, it in enumerate(items)}
        done = 0
        for fut in as_completed(futures):
            results[futures[fut]] = fut.result()
            done += 1
            if done % 50 == 0 or done == len(items):
                log.info("%s: %d/%d", desc, done, len(items))
    return results


def cpu_map(
    fn: Callable[[T], R],
    items: Sequence[T],
    *,
    workers: int | None = None,
    desc: str = "cpu_map",
) -> list[R]:
    """Run a CPU-bound, picklable fn over items on a process pool. Order preserved.

    `fn` and its args must be top-level/picklable (no lambdas/closures) for the
    ProcessPoolExecutor. Use for per-symbol feature builds and walk-forward folds.
    """
    cfg = load_config()
    workers = workers or cfg.concurrency.cpu_workers
    results: list[R | None] = [None] * len(items)
    with ProcessPoolExecutor(max_workers=workers) as ex:
        futures = {ex.submit(fn, it): i for i, it in enumerate(items)}
        done = 0
        for fut in as_completed(futures):
            try:
                results[futures[fut]] = fut.result()
            except Exception as exc:  # noqa: BLE001
                log.error("%s: item %d failed: %s", desc, futures[fut], exc)
            done += 1
            if done % 20 == 0 or done == len(items):
                log.info("%s: %d/%d", desc, done, len(items))
    return results


def chunked(items: Sequence[T], size: int) -> Iterable[Sequence[T]]:
    """Yield fixed-size chunks — batch symbols per worker task to amortize overhead."""
    for i in range(0, len(items), size):
        yield items[i : i + size]
