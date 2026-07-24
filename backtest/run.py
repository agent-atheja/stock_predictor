"""Backtest entry point — loads OOS predictions, runs the strategy + baselines, writes a report.

Run:  python -m backtest.run
"""
from __future__ import annotations

import json

import pandas as pd

from core.config import load_config, resolve
from core.logging_setup import get_logger, stage
from core.manifest import write_manifest
from backtest.engine import enrich, run_baselines, run_strategy
from backtest.report import equity_curve, sharpe_ratio
from validation.metrics import bootstrap_ci, daily_rank_ic, ic_by_year
from validation.slices import ic_by_fold, ic_by_regime

log = get_logger(__name__)


def main() -> dict:
    cfg = load_config()
    reg = resolve(cfg.model.registry_dir)
    oos_path = reg / "oos_predictions.parquet"
    if not oos_path.exists():
        raise RuntimeError("No OOS predictions — run models.train first.")
    oos = pd.read_parquet(oos_path)
    oos["date"] = pd.to_datetime(oos["date"])

    with stage(log, "backtest"):
        enriched = enrich(oos)              # attach vol_21d + turnover_cr_20d for weighting/costs
        net, perf = run_strategy(enriched)
        net_lo, perf_lo = run_strategy(enriched, long_only=True)   # deployable no-short variant
        baselines = run_baselines(oos)

        # statistical rigor: bootstrap 95% CI on net Sharpe + per-year OOS IC (regime slice)
        ppy = 252 / cfg.label.horizon_days
        perf["sharpe_ci95"] = bootstrap_ci(net, lambda r: sharpe_ratio(r, ppy))
        perf_lo["sharpe_ci95"] = bootstrap_ci(net_lo, lambda r: sharpe_ratio(r, ppy))
        ic = daily_rank_ic(oos["score"], oos["fwd_ret_5d"], oos["date"])
        perf["ic_by_year"] = ic_by_year(ic)

        # market beta: is the edge real alpha or just repackaged market exposure?
        mkt = oos.groupby("date")["fwd_ret_5d"].mean()
        perf["beta_to_market"] = _beta(net, mkt)
        perf_lo["beta_to_market"] = _beta(net_lo, mkt)

    slices = {"by_fold": ic_by_fold(oos), "by_regime": ic_by_regime(oos)}
    report = {"strategy_long_short": perf, "strategy_long_only": perf_lo,
              "baselines": baselines, "slices": slices}
    _log_report(report)

    out = resolve(cfg.model.registry_dir) / "backtest_report.json"
    out.write_text(json.dumps(report, indent=2, default=str))
    equity_curve(net).to_frame("equity").to_csv(resolve(cfg.model.registry_dir) / "equity_curve.csv")
    write_manifest(cfg.model.registry_dir, extra={"stage": "backtest"})
    log.info("Backtest report written to %s", out)
    return report


def _beta(strategy_returns: pd.Series, market: pd.Series) -> float:
    """Beta of strategy period returns to the equal-weight market, aligned on rebalance dates."""
    m = market.reindex(strategy_returns.index)
    df = pd.concat([strategy_returns, m], axis=1).dropna()
    if len(df) < 3 or df.iloc[:, 1].var() == 0:
        return float("nan")
    return float(df.cov().iloc[0, 1] / df.iloc[:, 1].var())


def _log_report(report: dict) -> None:
    for name in ("strategy_long_short", "strategy_long_only"):
        s = report[name]
        log.info("── %s (net of cost + STCG tax) ──", name)
        for k in ("cagr", "sharpe", "pretax_sharpe", "gross_sharpe", "sortino",
                  "max_drawdown", "hit_rate", "avg_turnover", "total_tax_drag"):
            v = s.get(k)
            log.info("  %-16s %s", k, f"{v:.4f}" if isinstance(v, float) else v)
        ci = s.get("sharpe_ci95")
        if ci:
            log.info("  %-16s [%.2f, %.2f]", "sharpe_ci95", ci[0], ci[1])

    mom = report["baselines"]["momentum_12_1"].get("sharpe", float("nan"))
    log.info("── Baselines (Sharpe) ──")
    log.info("  equal_weight     %.4f  (gross, costless/tax-free)",
             report["baselines"]["equal_weight"].get("sharpe", float("nan")))
    ewn = report["baselines"].get("equal_weight_net", {})
    log.info("  equal_weight_net %.4f  (net of same cost+STCG — the honest benchmark)",
             ewn.get("sharpe", float("nan")))
    log.info("  momentum_12_1    %.4f", mom)
    # Ship on the DEPLOYABLE (long-only) net-of-tax Sharpe beating the momentum baseline.
    lo_sharpe = report["strategy_long_only"].get("sharpe")
    ship = isinstance(lo_sharpe, float) and lo_sharpe > mom
    log.info("SHIP CRITERION (long-only net-of-tax Sharpe > momentum): %s", "PASS ✅" if ship else "NOT MET ❌")


if __name__ == "__main__":
    main()
