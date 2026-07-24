"""Run manifest — reproducibility record written beside every artifact set.

Captures what's needed to reproduce a run from logs alone: config hash + snapshot, data
versions (row counts + latest date per layer), key library versions, and a UTC timestamp.
"""
from __future__ import annotations

import hashlib
import json
import platform
from datetime import datetime, timezone
from pathlib import Path

import yaml

from core.config import load_config, resolve
from core.logging_setup import get_logger

log = get_logger(__name__)


def _config_hash(cfg) -> str:
    canonical = yaml.safe_dump(cfg._raw, sort_keys=True).encode()
    return hashlib.sha256(canonical).hexdigest()[:16]


def _data_versions() -> dict:
    import pyarrow.dataset as ds

    cfg = load_config()
    out = {}
    for layer, dataset in (("bronze", "equity_ohlcv"), ("silver", "equity_ohlcv_adj"), ("gold", "features")):
        base = resolve(getattr(cfg.data, layer)) / dataset
        if not base.exists():
            out[layer] = None
            continue
        try:
            parts = sorted(p.name for p in base.glob("dt=*"))
            out[layer] = {"n_partitions": len(parts), "latest": parts[-1][3:] if parts else None}
        except Exception:  # noqa: BLE001
            out[layer] = None
    return out


def _lib_versions() -> dict:
    import importlib

    vers = {}
    for mod in ("pandas", "numpy", "lightgbm", "pyarrow"):
        try:
            vers[mod] = importlib.import_module(mod).__version__
        except Exception:  # noqa: BLE001
            vers[mod] = None
    vers["python"] = platform.python_version()
    return vers


def write_manifest(registry_dir: str, extra: dict | None = None) -> Path:
    cfg = load_config()
    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "config_hash": _config_hash(cfg),
        "config_snapshot": cfg._raw,
        "data_versions": _data_versions(),
        "lib_versions": _lib_versions(),
    }
    if extra:
        manifest.update(extra)
    out = resolve(registry_dir) / "run_manifest.json"
    out.write_text(json.dumps(manifest, indent=2, default=str))
    log.info("Run manifest written (config_hash=%s) → %s", manifest["config_hash"], out)
    return out
