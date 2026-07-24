"""Central config loader.

Single source of truth: config/config.yaml. Everything else reads through here so
there is exactly one place to change a tunable. Self-contained — no paths outside
this project except the datalake root (which is inside the project by default).
"""
from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import yaml

# Project root = parent of this file's package (…/stock_predictor)
ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "config" / "config.yaml"


def _to_ns(obj: Any) -> Any:
    """Recursively turn dicts into attribute-accessible namespaces (cfg.model.type)."""
    if isinstance(obj, dict):
        return SimpleNamespace(**{k: _to_ns(v) for k, v in obj.items()})
    if isinstance(obj, list):
        return [_to_ns(v) for v in obj]
    return obj


@lru_cache(maxsize=1)
def load_config(path: str | os.PathLike | None = None) -> SimpleNamespace:
    """Load and cache the project config. Pass a path to override (mainly for tests)."""
    cfg_path = Path(path) if path else DEFAULT_CONFIG
    with open(cfg_path, "r") as fh:
        raw = yaml.safe_load(fh)
    ns = _to_ns(raw)
    ns._raw = raw          # keep raw dict for serialization / logging
    ns._root = ROOT
    _validate(ns)
    return ns


class ConfigError(ValueError):
    """Raised when config values are missing or out of range — fail fast at load time."""


def _validate(cfg) -> None:
    """Guard against silently-wrong config. Fail loudly at the boundary, not deep in a pipeline."""
    errors: list[str] = []
    if cfg.label.horizon_days <= 0:
        errors.append("label.horizon_days must be > 0")
    if cfg.walkforward.embargo_days <= cfg.label.horizon_days:
        errors.append("walkforward.embargo_days must exceed label.horizon_days (leakage risk)")
    if cfg.backtest.top_n <= 0:
        errors.append("backtest.top_n must be > 0")
    if cfg.backtest.weighting not in ("equal", "vol_target"):
        errors.append("backtest.weighting must be 'equal' or 'vol_target'")
    if not (0 <= cfg.backtest.stcg_pct <= 100):
        errors.append("backtest.stcg_pct must be in [0, 100]")
    if not (0 < cfg.backtest.score_smoothing_alpha <= 1.0):
        errors.append("backtest.score_smoothing_alpha must be in (0, 1]")
    if len(cfg.backtest.regime_gross_scaling) != 3:
        errors.append("backtest.regime_gross_scaling must have 3 entries [low, mid, high]")
    if errors:
        raise ConfigError("Invalid config:\n  - " + "\n  - ".join(errors))


def resolve(path_like: str) -> Path:
    """Resolve a config-relative path against the project root (keeps things portable)."""
    p = Path(path_like)
    return p if p.is_absolute() else (ROOT / p)


def secret(env_name: str, default: str | None = None) -> str | None:
    """Read a secret from the environment (populated from secrets/.env by dotenv)."""
    return os.environ.get(env_name, default)
