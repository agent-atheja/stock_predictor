"""Walk-forward training orchestrator.

For each purged/embargoed fold:
  1. (optional) Optuna HPO on an INNER time-split of the train window only — never sees test.
  2. Train the seed-averaged cross-sectional model on the full train window.
  3. Predict on the test window → collect OUT-OF-SAMPLE scores.
Then aggregate OOS predictions, compute Rank IC / decile spread, fit a final model on ALL data
for live serving, and dump SHAP importance.

Folds run SEQUENTIALLY on purpose: LightGBM already saturates all cores, so parallelizing folds
would oversubscribe the 24 threads and slow things down. (The parallel pools are for the
embarrassingly-parallel, single-core-bound ingestion/feature stages.)

Run:  python -m models.train [--hpo]
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from core.config import load_config, resolve
from core.logging_setup import get_logger, stage
from models.dataset import apply_universe_filter, load_gold, make_xy
from models.lgbm_ranker import CrossSectionalModel, train_model
from validation.metrics import daily_rank_ic, decile_spread, ic_summary
from validation.walkforward import make_folds, purge_train_mask

log = get_logger(__name__)


def _tune(X_tr, y_tr, d_tr, n_trials: int = 25) -> dict:
    """Optuna HPO on an inner temporal split of the training window (no test leakage)."""
    import optuna

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    cut = d_tr.quantile(0.8)
    itr, ival = d_tr <= cut, d_tr > cut
    if ival.sum() < 200:  # too small to tune — keep defaults
        return {}

    def objective(trial: "optuna.Trial") -> float:
        params = dict(
            num_leaves=trial.suggest_int("num_leaves", 15, 127),
            learning_rate=trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
            min_child_samples=trial.suggest_int("min_child_samples", 20, 200),
            reg_lambda=trial.suggest_float("reg_lambda", 1.0, 20.0),
            colsample_bytree=trial.suggest_float("colsample_bytree", 0.5, 1.0),
        )
        m = train_model(X_tr[itr], y_tr[itr], d_tr[itr], params=params)
        ic = daily_rank_ic(pd.Series(m.predict(X_tr[ival])), y_tr[ival], d_tr[ival])
        return ic.mean() if len(ic) else -1.0

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    log.info("HPO best inner IC=%.4f params=%s", study.best_value, study.best_params)
    return study.best_params


def run(do_hpo: bool = False) -> dict:
    cfg = load_config()
    np.random.seed(0)                       # global determinism for reproducibility
    with stage(log, "train-walkforward"):
        gold = load_gold()
        univ = apply_universe_filter(gold)
        folds = make_folds(univ["date"])
        if not folds:
            raise RuntimeError("No walk-forward folds — need more history (extend history_start).")

        h = cfg.label.horizon_days
        oos_frames = []
        for fold in folds:
            tr = univ[(univ["date"] >= fold.train_start) & (univ["date"] <= fold.train_end)]
            te = univ[(univ["date"] >= fold.test_start) & (univ["date"] <= fold.test_end)]
            X_tr, y_tr, m_tr = make_xy(tr)
            # extra purge: drop train bars whose forward label reaches the test window
            keep = purge_train_mask(m_tr["date"], fold.test_start, h)
            X_tr, y_tr, m_tr = X_tr[keep], y_tr[keep], m_tr[keep]
            X_te, y_te, m_te = make_xy(te)
            if len(X_tr) < 500 or len(X_te) == 0:
                continue

            params = _tune(X_tr, y_tr, m_tr["date"]) if do_hpo else {}
            model = train_model(X_tr, y_tr, m_tr["date"], params=params)
            m_te = m_te.assign(score=model.predict(X_te), fold=fold.idx)
            oos_frames.append(m_te)
            ic = daily_rank_ic(m_te["score"], m_te["fwd_ret_5d"], m_te["date"])
            log.info("fold %d | train=%d test=%d | OOS mean IC=%.4f",
                     fold.idx, len(X_tr), len(X_te), ic.mean() if len(ic) else np.nan)

        oos = pd.concat(oos_frames, ignore_index=True)
        summary = _report_and_persist(oos, univ)
    return summary


def _report_and_persist(oos: pd.DataFrame, univ: pd.DataFrame) -> dict:
    cfg = load_config()
    reg = resolve(cfg.model.registry_dir)
    reg.mkdir(parents=True, exist_ok=True)

    ic = daily_rank_ic(oos["score"], oos["fwd_ret_5d"], oos["date"])
    summ = ic_summary(ic)
    deciles = decile_spread(oos["score"], oos["fwd_ret_5d"], oos["date"])
    ls_spread = float(deciles["mean_fwd_ret"].iloc[-1] - deciles["mean_fwd_ret"].iloc[0])
    summ["long_short_decile_spread_5d"] = ls_spread
    summ["decile_monotonic"] = bool(deciles["monotonic"].iloc[-1])

    log.info("── OOS predictive summary ──")
    for k, v in summ.items():
        log.info("  %-28s %s", k, f"{v:.4f}" if isinstance(v, float) else v)

    # persist OOS predictions + metrics + final full-data model for serving
    oos.to_parquet(reg / "oos_predictions.parquet", index=False)
    (reg / "metrics.json").write_text(json.dumps(summ, indent=2, default=str))

    X_all, y_all, m_all = make_xy(univ)
    final = train_model(X_all, y_all, m_all["date"])
    _save_model(final, reg / "final_model")
    (reg / "feature_importance.json").write_text(
        json.dumps(final.feature_importance.round(1).to_dict(), indent=2)
    )
    _dump_shap(final, X_all.sample(min(2000, len(X_all)), random_state=0), reg)
    from core.manifest import write_manifest
    write_manifest(cfg.model.registry_dir, extra={"stage": "train", "oos_summary": summ})
    log.info("Saved final model + OOS predictions + metrics to %s", reg)
    return summ


def _save_model(model: CrossSectionalModel, path: Path) -> None:
    """Persist the whole ensemble via joblib — handles mixed member types (LGB ranker + XGB)."""
    import joblib

    path.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path / "model.joblib")
    (path / "meta.json").write_text(json.dumps({"mode": model.mode, "n_models": len(model.models)}))


def _dump_shap(model: CrossSectionalModel, X_sample: pd.DataFrame, reg: Path) -> None:
    """SHAP global importance on the first seed model — trust gate + leakage detector."""
    try:
        import shap

        expl = shap.TreeExplainer(model.models[0].booster_)
        sv = expl.shap_values(X_sample)
        mean_abs = pd.Series(np.abs(sv).mean(axis=0), index=X_sample.columns).sort_values(ascending=False)
        (reg / "shap_importance.json").write_text(json.dumps(mean_abs.round(4).to_dict(), indent=2))
        log.info("SHAP top-5: %s", ", ".join(mean_abs.head(5).index))
    except Exception as exc:  # noqa: BLE001 — SHAP is diagnostic, not load-bearing
        log.warning("SHAP dump skipped: %s", exc)


if __name__ == "__main__":
    run(do_hpo="--hpo" in sys.argv)
