"""
train.py: download data, build features, run walk-forward tests, train the
final model, save it, and write the report.

    python train.py              # real data (Yahoo Finance + FRED)
    python train.py --synthetic  # fake random-walk data, to test the code offline
    python train.py --fast       # bigger test folds (fewer retrains), quicker

Outputs:
    models/model.joblib   the trained models + metadata, used by predict.py
    docs/report.html      the full report (published by GitHub Pages)
    docs/backtest.png     backtest chart (also shown in REPORT.md)
    REPORT.md             short version of the report for the GitHub app
"""
import argparse
import json
import time
from datetime import date

import joblib
import numpy as np
import pandas as pd

from stockpred.config import ROOT, load_config
from stockpred import evaluation as ev
from stockpred.features import FEATURES, build_panel, leak_check
from stockpred.models import fit_classifiers, fit_models
from stockpred.report import build_report


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--synthetic", action="store_true", help="use fake data (offline test)")
    ap.add_argument("--fast", action="store_true", help="half-year test folds instead of quarterly")
    ap.add_argument("--force-download", action="store_true", help="ignore the local cache")
    args = ap.parse_args()
    t0 = time.time()
    cfg = load_config()
    if args.fast:
        cfg["model"]["fold_days"] = 126
    horizon = cfg["model"]["horizon"]

    # ---------------------------------------------------------------- data
    print("1) Loading data...")
    if args.synthetic:
        from stockpred.synthetic import make_synthetic
        prices, earnings, fred = make_synthetic(cfg)
    else:
        from stockpred.data import load_all
        prices, earnings, fred = load_all(cfg, force=args.force_download)
    targets = [t for t in cfg["targets"] if t in prices]
    pool = [t for t in cfg["pool"] if t in prices]
    missing = sorted(set(cfg["targets"]) - set(targets))
    if missing:
        print(f"   WARNING: no data for {missing}; they are skipped.")
    print(f"   {len(targets)} stocks to predict, {len(pool) - len(targets)} extra training-only stocks")

    # ---------------------------------------------------------- leak check
    print("2) Checking that features use only past data...")
    ok, leak_msg = leak_check(prices, earnings, fred, pool, horizon)
    print("   " + leak_msg)
    if not ok:
        raise SystemExit("Leak check FAILED. Fix features.py before trusting anything.")

    # ------------------------------------------------------------- features
    print("3) Building features...")
    panel = build_panel(prices, earnings, fred, pool, horizon, targets=targets)
    hist_days = panel[panel["is_target"]].groupby("ticker")["n_days_history"].max()
    short = sorted(hist_days[hist_days < cfg["data"]["short_history_days"]].index)
    print(f"   {len(panel):,} rows, {panel['ticker'].nunique()} tickers, "
          f"{panel['date'].min().date()} -> {panel['date'].max().date()}")
    print(f"   Limited history (<{cfg['data']['short_history_days']} days): {short}")

    # --------------------------------------------------------- walk-forward
    print("4) Walk-forward testing (this is the slow part)...")
    oos, resid, resid_1d = ev.walk_forward(panel, cfg)

    dir_tab = ev.direction_table(oos)
    auc = dir_tab.set_index("col")["auc"]
    primary = "p_lgbm" if auc["p_lgbm"] >= auc["p_logit"] else "p_logit"
    primary_name = ev.DIRECTION_MODELS[primary]
    print("\n   Direction results (out-of-sample):")
    print(dir_tab.drop(columns="col").to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    acc_primary = dir_tab.set_index("col").loc[primary, "accuracy"]
    if acc_primary > 0.60:
        print("\n" + "!" * 70 + f"\n   ACCURACY {acc_primary:.1%} > 60%: almost certainly a bug or leak.\n"
              "   Do not trust these results; investigate features/targets first.\n" + "!" * 70)

    boot = ev.bootstrap(oos, primary)
    cal = ev.calibration_table(oos, primary)
    rng_tab = ev.range_table(oos)
    rng1_tab = ev.range_table_1d(oos)
    print("\n   Range results (1 week):")
    print(rng_tab.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print("\n   Range results (tomorrow):")
    print(rng1_tab.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    # ------------------------------------------------------------- backtest
    print("5) Backtest...")
    bench = pd.DataFrame({t: prices[t]["Close"].shift(-horizon) / prices[t]["Close"] - 1
                          for t in ["SPY", "SMH"] if t in prices})
    bt_cfg = cfg["backtest"]
    bt = ev.backtest(oos, primary, bt_cfg["threshold"], bt_cfg["cost_per_trade"], bench)
    bt_sum = ev.backtest_summary(bt)
    sweep = ev.threshold_sweep(oos, primary, bt_cfg["thresholds_to_test"], bt_cfg["cost_per_trade"], bench)
    print(bt_sum.to_string(float_format=lambda x: f"{x:.3f}"))

    v = ev.verdict(dir_tab, boot, rng_tab, bt_sum, primary)
    print("\n   VERDICT: " + v["headline"])
    for d in v["details"]:
        print("    - " + d)

    # ---------------------------------------------------------- experiments
    print("6) Experiments: easier questions (beat peers / 1 month)...")
    exps = ev.run_experiments(panel, cfg, oos)
    for name, e in exps.items():
        d, x = e["dev"], e["exam"]
        print(f"   {e['desc']:<45} dev AUC {d['auc']:.3f} IC {d['ic']:+.3f} (t={d['ic_t']:.1f}) | "
              + (f"exam AUC {x['auc']:.3f} IC {x['ic']:+.3f}" if x else "exam n/a") + f" -> {e['label']}")

    # ---------------------------------------------------------- final model
    print("7) Training the final models on all data...")
    labeled = panel[panel["fwd_ret"].notna()]
    final = fit_models(labeled)
    # Final models for the "beat peers" questions (shown in LATEST.md with their verdicts)
    exp_models = {}
    for name in ("rel_5", "rel_20"):
        if name in exps:
            ycol = exps[name]["ycol"]
            m = fit_classifiers(panel[panel[ycol].notna()], ycol)
            exp_models[name] = {"model": m[exps[name]["best"].replace("p_", "")],
                                "level": exps[name]["level"], "label": exps[name]["label"]}
    gain = final["lgbm"].booster_.feature_importance("gain")
    imp = pd.Series(gain / gain.sum(), index=FEATURES).sort_values(ascending=False)
    coef = pd.Series(final["logit"].named_steps["lr"].coef_[0], index=FEATURES)
    top_coefs = coef.reindex(coef.abs().sort_values(ascending=False).index).head(8).round(3).to_dict()

    bundle = {
        "models": final,
        "primary": primary,
        "primary_name": primary_name,
        "features": FEATURES,
        "cqr_adj": float(np.quantile(resid, 0.8)),   # range widening learned out-of-sample
        "cqr_adj_1d": float(np.quantile(resid_1d, 0.8)),
        "experiments": exp_models,
        "verdict": v,
        "trained_through": str(labeled["date"].max().date()),
        "trained_on": date.today().isoformat(),
        "short_history": short,
        "synthetic": args.synthetic,
        "metrics": {
            "direction": dir_tab.drop(columns="col").to_dict("records"),
            "range": rng_tab.to_dict("records"),
            "range_1d": rng1_tab.to_dict("records"),
            "experiments": {k: {x: e[x] for x in ["desc", "best", "dev", "exam", "level", "label"]}
                            for k, e in exps.items()},
        },
    }
    (ROOT / "models").mkdir(exist_ok=True)
    joblib.dump(bundle, ROOT / "models" / "model.joblib", compress=3)
    with open(ROOT / "models" / "summary.json", "w") as f:
        json.dump({k: bundle[k] for k in ["primary_name", "trained_through", "trained_on", "cqr_adj",
                                          "short_history", "synthetic", "metrics"]}
                  | {"verdict": {k: v[k] for k in ["level", "headline", "details"]}}, f, indent=2, default=float)

    # --------------------------------------------------------------- report
    print("8) Writing report...")
    (ROOT / "docs").mkdir(exist_ok=True)
    ctx = dict(verdict=v, direction=dir_tab, calibration=cal, range=rng_tab, backtest=bt,
               backtest_summary=bt_sum, sweep=sweep, yearly=ev.yearly_table(oos, primary),
               importance=imp, coefs=top_coefs, primary_name=primary_name,
               coverage_by_group=ev.coverage_by_group(oos, cfg["groups"]),
               data_end=str(panel["date"].max().date()), test_start=str(oos["date"].min().date()),
               n_rows=len(panel), n_tickers=len(targets), n_pool=len(pool), range_1d=rng1_tab,
               experiments=exps, exam_start=cfg["experiments"]["final_exam_start"], gap_days=cfg["model"]["gap_days"],
               threshold=bt_cfg["threshold"], cost=bt_cfg["cost_per_trade"], leak_msg=leak_msg,
               short_history=short, synthetic=args.synthetic)
    build_report(ctx, ROOT / "docs" / "report.html", ROOT / "REPORT.md", ROOT / "docs" / "backtest.png")
    print(f"Done in {time.time() - t0:.0f}s. Open docs/report.html")


if __name__ == "__main__":
    main()
