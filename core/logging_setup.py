"""Structured, consistent logging. Every stage logs row counts / timings via helpers here."""
from __future__ import annotations

import logging
import sys
import time
from contextlib import contextmanager
from pathlib import Path

from core.config import load_config, resolve

_CONFIGURED = False


def get_logger(name: str) -> logging.Logger:
    global _CONFIGURED
    if not _CONFIGURED:
        cfg = load_config()
        log_dir = resolve(cfg.logging.dir)
        log_dir.mkdir(parents=True, exist_ok=True)
        fmt = "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"
        handlers = [
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(log_dir / "pipeline.log"),
        ]
        logging.basicConfig(level=getattr(logging, cfg.logging.level), format=fmt, handlers=handlers)
        _CONFIGURED = True
    return logging.getLogger(name)


@contextmanager
def stage(logger: logging.Logger, name: str):
    """Time a pipeline stage and log start/end. Usage: `with stage(log, 'silver'): ...`."""
    logger.info("▶ %s — start", name)
    t0 = time.perf_counter()
    try:
        yield
    finally:
        logger.info("✔ %s — done in %.2fs", name, time.perf_counter() - t0)
