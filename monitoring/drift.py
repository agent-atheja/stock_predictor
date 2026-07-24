"""Drift & IC-decay monitoring — the silent-degradation tripwires.

Two health checks, each emitting a status and a retrain recommendation:
  • Feature drift (PSI): current feature distribution vs a training-reference window.
        PSI < 0.1 stable · 0.1–0.25 moderate · > 0.25 significant (retrain).
  • IC decay: recent out-of-sample Rank IC vs the model's historical mean; alert if the edge
        has faded below a floor.

These are the L1→L2 MLOps gap: without them, degradation surfaces via P&L (too late).

Run:  python -m monitoring.drift
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from core.config import load_config, resolve
from core.io import read_dataset
from core.logging_setup import get_logger
from features.assembler import FEATURE_COLS
from validation.metrics import daily_rank_ic

log = get_logger(__name__)

PSI_MODERATE, PSI_SIGNIFICANT = 0.10, 0.25


def psi(reference: pd.Series, current: pd.Series, bins: int = 10) -> float:
    """Population Stability Index between two distributions using reference-quantile bins."""
    ref, cur = reference.dropna(), current.dropna()
    if len(ref) < bins or len(cur) < bins:
        return np.nan
    edges = np.unique(np.quantile(ref, np.linspace(0, 1, bins + 1)))
    if len(edges) < 3:
        return np.nan
    ref_pct = np.histogram(ref, bins=edges)[0] / len(ref)
    cur_pct = np.histogram(cur, bins=edges)[0] / len(cur)
    eps = 1e-6
    ref_pct, cur_pct = ref_pct + eps, cur_pct + eps
    return float(np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct)))


def feature_drift(reference_days: int = 756, recent_days: int = 63) -> dict:
    """PSI per feature: recent window vs an older training-reference window from Gold."""
    gold = read_dataset(load_config().data.gold, "features")
    gold["date"] = pd.to_datetime(gold["date"])
    dates = np.sort(gold["date"].unique())
    if len(dates) < reference_days + recent_days:
        log.warning("Not enough history for drift check")
        return {}
    ref_cut, rec_cut = dates[-(recent_days + reference_days)], dates[-recent_days]
    ref = gold[(gold["date"] >= ref_cut) & (gold["date"] < rec_cut)]
    rec = gold[gold["date"] >= rec_cut]
    out = {}
    for f in FEATURE_COLS:
        val = psi(ref[f], rec[f])
        status = "stable" if val < PSI_MODERATE else ("moderate" if val < PSI_SIGNIFICANT else "significant")
        out[f] = {"psi": round(val, 4) if not np.isnan(val) else None, "status": status}
    n_sig = sum(1 for v in out.values() if v["status"] == "significant")
    return {"per_feature": out, "n_significant": n_sig, "retrain_recommended": n_sig >= 3}


def ic_decay(recent_days: int = 126, ic_floor: float = 0.005) -> dict:
    """Compare recent OOS Rank IC to the model's historical mean; flag if the edge has faded."""
    reg = resolve(load_config().model.registry_dir)
    path = reg / "oos_predictions.parquet"
    if not path.exists():
        return {"error": "no oos_predictions — train first"}
    oos = pd.read_parquet(path)
    oos["date"] = pd.to_datetime(oos["date"])
    ic = daily_rank_ic(oos["score"], oos["fwd_ret_5d"], oos["date"])
    recent = ic[ic.index >= ic.index.max() - pd.Timedelta(days=recent_days)]
    hist_mean, recent_mean = float(ic.mean()), float(recent.mean()) if len(recent) else np.nan
    decayed = not np.isnan(recent_mean) and recent_mean < ic_floor
    return {
        "historical_mean_ic": round(hist_mean, 4),
        "recent_mean_ic": round(recent_mean, 4) if not np.isnan(recent_mean) else None,
        "ic_floor": ic_floor,
        "edge_decayed": decayed,
        "retrain_recommended": decayed,
    }


def run() -> dict:
    report = {"feature_drift": feature_drift(), "ic_decay": ic_decay()}
    report["retrain_recommended"] = bool(
        report["feature_drift"].get("retrain_recommended") or report["ic_decay"].get("retrain_recommended")
    )
    out = resolve(load_config().model.registry_dir) / "monitoring_report.json"
    out.write_text(json.dumps(report, indent=2, default=str))
    log.info("Monitoring: %d features drifting significantly | recent IC=%s | retrain=%s",
             report["feature_drift"].get("n_significant", 0),
             report["ic_decay"].get("recent_mean_ic"), report["retrain_recommended"])
    return report


if __name__ == "__main__":
    run()
