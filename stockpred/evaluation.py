"""
Walk-forward testing, metrics, and the backtest.

Walk-forward = simulate the past honestly:
    fold 1: train on 2011..2015      -> test on Q1 2016
    fold 2: train on 2011..Q1 2016   -> test on Q2 2016
    ... and so on, rolling forward. The model never sees the period it is tested on.

Purge gap: a training row from day t has a label that uses prices up to t+5.
If t sits just before the test period, its label overlaps test-period prices.
So we drop the last `gap_days` training days before each test period.
"""
import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, roc_auc_score

from .models import fit_classifiers, fit_models, predict_all, predict_classifiers

DIRECTION_MODELS = {
    "p_always_up": "Baseline: always up",
    "p_last_week": "Baseline: last week repeats",
    "p_hist_avg": "Baseline: historical average",
    "p_logit": "Logistic regression",
    "p_lgbm": "LightGBM",
}


# ----------------------------------------------------------------------------
# Walk-forward
# ----------------------------------------------------------------------------
def walk_forward(panel, cfg, verbose=True):
    mc = cfg["model"]
    lab = panel[panel["fwd_ret"].notna()].copy()
    dates = np.sort(lab["date_pos"].unique())
    pos_of = {d: i for i, d in enumerate(dates)}
    lab["k"] = lab["date_pos"].map(pos_of)          # 0..N-1 index over labeled days
    date_of_k = lab.groupby("k")["date"].first()

    start_k = int(np.searchsorted(date_of_k.values, np.datetime64(mc["first_test_date"])))
    start_k = max(start_k, int(252 * mc["min_train_years"]))
    fold, gap = mc["fold_days"], mc["gap_days"]

    results, prior_resid, prior_resid_1d = [], [], []
    n_folds = int(np.ceil((len(dates) - start_k) / fold))
    for f, i in enumerate(range(start_k, len(dates), fold)):
        train = lab[lab["k"] < i - gap]                 # <-- the purge gap
        # Train on the whole pool, but grade only on the stocks we report on.
        test = lab[(lab["k"] >= i) & (lab["k"] < i + fold) & lab["is_target"]]
        if len(test) == 0:
            continue
        m = fit_models(train)
        p = predict_all(m, test)
        p["fold"] = f
        # Interval calibration: use errors from EARLIER folds only (honest),
        # i.e. "how far outside the 10-90 band did reality land in the past?"
        p["cqr_adj"] = float(np.quantile(prior_resid, 0.8)) if len(prior_resid) > 500 else 0.0
        resid = np.maximum(p["q10_z"] - test["z_target"], test["z_target"] - p["q90_z"])
        prior_resid.extend(resid.tolist())
        p["cqr_adj_1d"] = float(np.quantile(prior_resid_1d, 0.8)) if len(prior_resid_1d) > 500 else 0.0
        resid1 = np.maximum(p["q10_1d_z"] - test["z_target_1"], test["z_target_1"] - p["q90_1d_z"])
        prior_resid_1d.extend(resid1.dropna().tolist())
        results.append(pd.concat([test[["date", "ticker", "date_pos", "fwd_ret", "y_up", "scale",
                                        "z_target", "n_days_history", "fwd_ret_1", "scale_1"]], p], axis=1))
        if verbose:
            d0, d1 = test["date"].min().date(), test["date"].max().date()
            print(f"  fold {f + 1:>2}/{n_folds}: train {len(train):>7,} rows | test {d0} -> {d1}")
    oos = pd.concat(results).reset_index(drop=True)
    # Convert z-unit quantiles back into % returns.
    for c in ["q10", "q50", "q90", "bq10", "bq50", "bq90"]:
        oos[c] = oos[c + "_z"] * oos["scale"]
    oos["q10_adj"] = (oos["q10_z"] - oos["cqr_adj"]) * oos["scale"]
    oos["q90_adj"] = (oos["q90_z"] + oos["cqr_adj"]) * oos["scale"]
    # Tomorrow's range
    oos["bq10_1d"] = oos["bq10_1d_z"] * oos["scale_1"]
    oos["bq90_1d"] = oos["bq90_1d_z"] * oos["scale_1"]
    oos["q10_1d"] = oos["q10_1d_z"] * oos["scale_1"]
    oos["q90_1d"] = oos["q90_1d_z"] * oos["scale_1"]
    oos["q10_1d_adj"] = (oos["q10_1d_z"] - oos["cqr_adj_1d"]) * oos["scale_1"]
    oos["q90_1d_adj"] = (oos["q90_1d_z"] + oos["cqr_adj_1d"]) * oos["scale_1"]
    return oos, np.array(prior_resid), np.array(prior_resid_1d)


# ----------------------------------------------------------------------------
# Direction metrics
# ----------------------------------------------------------------------------
def _auc(y, p):
    return roc_auc_score(y, p) if len(np.unique(p)) > 1 else 0.5


def direction_table(oos):
    y = oos["y_up"].astype(int)
    rows = []
    for col, name in DIRECTION_MODELS.items():
        p = oos[col]
        rows.append({
            "model": name, "col": col,
            "accuracy": float(((p > 0.5) == y).mean()) if col != "p_always_up" else float(y.mean()),
            # A constant guess has no ranking ability: AUC is 0.5 by definition.
            "auc": 0.5 if col == "p_always_up" else float(_auc(y, p)),
            "brier": float(brier_score_loss(y, p.clip(0, 1))),
        })
    return pd.DataFrame(rows)


def bootstrap(oos, col, n_boot=200, seed=0):
    """
    Block bootstrap by WEEK. Rows are NOT independent: 33 stocks on the same day
    move together, and overlapping 5-day windows share days. Resampling whole
    weeks gives honest (wider) uncertainty than pretending we have 100k
    independent coin flips.
    Returns 90% intervals for AUC and for (accuracy - always-up accuracy).
    """
    rng = np.random.default_rng(seed)
    y = oos["y_up"].astype(int).values
    p = oos[col].values
    correct = ((p > 0.5) == y).astype(float)
    block = (oos["date_pos"].values // 5)
    ub, inv = np.unique(block, return_inverse=True)
    idx_by_block = pd.Series(np.arange(len(y))).groupby(inv).apply(np.array).values
    aucs, diffs = [], []
    for _ in range(n_boot):
        pick = rng.integers(0, len(ub), len(ub))
        idx = np.concatenate([idx_by_block[b] for b in pick])
        aucs.append(_auc(y[idx], p[idx]))
        diffs.append(correct[idx].mean() - y[idx].mean())
    return {"auc_lo": float(np.quantile(aucs, 0.05)), "auc_hi": float(np.quantile(aucs, 0.95)),
            "accdiff": float(correct.mean() - y.mean()),
            "accdiff_lo": float(np.quantile(diffs, 0.05)), "accdiff_hi": float(np.quantile(diffs, 0.95))}


def calibration_table(oos, col, edges=(0, .40, .45, .48, .50, .52, .55, .60, 1.0001)):
    """When the model says X%, how often does the stock actually go up?"""
    b = pd.cut(oos[col], edges, right=False)
    t = oos.groupby(b, observed=True).agg(predicted=(col, "mean"), actual=("y_up", "mean"),
                                          n=("y_up", "size"))
    t.index = [f"{iv.left:.0%}-{min(iv.right, 1):.0%}" for iv in t.index]
    return t[t["n"] > 0]


def yearly_table(oos, col):
    g = oos.assign(year=oos["date"].dt.year).groupby("year")
    return pd.DataFrame({
        "model_accuracy": g.apply(lambda d: ((d[col] > .5) == d["y_up"]).mean()),
        "always_up_accuracy": g["y_up"].mean(),
        "auc": g.apply(lambda d: _auc(d["y_up"].astype(int), d[col])),
        "rows": g.size(),
    })


# ----------------------------------------------------------------------------
# Range metrics
# ----------------------------------------------------------------------------
def pinball(y, q, a):
    d = y - q
    return float(np.mean(np.maximum(a * d, (a - 1) * d)))


def range_table(oos):
    y = oos["fwd_ret"]
    def cov(lo, hi):
        return float(((y >= oos[lo]) & (y <= oos[hi])).mean())
    def width(lo, hi):
        return float((oos[hi] - oos[lo]).median())
    rows = [
        {"model": "Baseline (volatility only)",
         "coverage": cov("bq10", "bq90"), "median_width": width("bq10", "bq90"),
         "pinball": np.mean([pinball(y, oos[c], a) for c, a in [("bq10", .1), ("bq50", .5), ("bq90", .9)]])},
        {"model": "LightGBM (raw)",
         "coverage": cov("q10", "q90"), "median_width": width("q10", "q90"),
         "pinball": np.mean([pinball(y, oos[c], a) for c, a in [("q10", .1), ("q50", .5), ("q90", .9)]])},
        {"model": "LightGBM (calibrated)",
         "coverage": cov("q10_adj", "q90_adj"), "median_width": width("q10_adj", "q90_adj"),
         "pinball": np.nan},
    ]
    return pd.DataFrame(rows)


def range_table_1d(oos):
    """Same coverage check for tomorrow's range."""
    y = oos["fwd_ret_1"]
    rows = []
    for name, lo, hi in [("Baseline (volatility only)", "bq10_1d", "bq90_1d"),
                         ("LightGBM (raw)", "q10_1d", "q90_1d"),
                         ("LightGBM (calibrated)", "q10_1d_adj", "q90_1d_adj")]:
        rows.append({"model": name, "coverage": float(((y >= oos[lo]) & (y <= oos[hi])).mean()),
                     "median_width": float((oos[hi] - oos[lo]).median())})
    return pd.DataFrame(rows)


def coverage_by_group(oos, groups):
    g = oos.assign(group=oos["ticker"].map(groups))
    inr = (g["fwd_ret"] >= g["q10_adj"]) & (g["fwd_ret"] <= g["q90_adj"])
    return inr.groupby(g["group"]).mean()


# ----------------------------------------------------------------------------
# Backtest
# ----------------------------------------------------------------------------
def backtest(oos, col, threshold, cost, bench_fwd):
    """
    Rule: every 5 trading days, hold (equal weight) each stock with P(up) > threshold
    for the next week; hold cash (0%) if none qualify. Costs: `cost` x turnover,
    so 0.1% for each buy and each sell. Keeping a stock you already hold is free.
    Only out-of-sample predictions are used.

    bench_fwd: DataFrame indexed by date with 5-day forward returns of SPY/SMH.
    """
    reb_dates = np.sort(oos["date_pos"].unique())[::5]
    rows, prev_w = [], {}
    for d in reb_dates:
        day = oos[oos["date_pos"] == d]
        pick = day[day[col] > threshold]
        w = {t: 1 / len(pick) for t in pick["ticker"]} if len(pick) else {}
        turnover = sum(abs(w.get(t, 0) - prev_w.get(t, 0)) for t in set(w) | set(prev_w))
        gross = float((pick["fwd_ret"]).mean()) if len(pick) else 0.0
        ew = float(day["fwd_ret"].mean())
        date = day["date"].iloc[0]
        rows.append({"date": date, "strategy": gross - cost * turnover, "gross": gross,
                     "n_held": len(pick), "turnover": turnover, "equal_weight_universe": ew})
        prev_w = w
    bt = pd.DataFrame(rows).set_index("date")
    bt = bt.join(bench_fwd, how="left").fillna({"SPY": 0.0, "SMH": 0.0})
    return bt


def perf_stats(r, periods=52):
    r = pd.Series(r).dropna()
    eq = (1 + r).cumprod()
    years = len(r) / periods
    cagr = eq.iloc[-1] ** (1 / years) - 1 if years > 0 else np.nan
    vol = r.std() * np.sqrt(periods)
    dd = (eq / eq.cummax() - 1).min()
    return {"CAGR": cagr, "Volatility": vol, "Sharpe": (r.mean() * periods) / vol if vol else np.nan,
            "Max drawdown": dd, "Total return": eq.iloc[-1] - 1}


def backtest_summary(bt):
    cols = {"strategy": "Model rule (after costs)", "equal_weight_universe": "Hold all stocks (equal weight)",
            "SPY": "Buy & hold SPY", "SMH": "Buy & hold SMH"}
    t = pd.DataFrame({name: perf_stats(bt[c]) for c, name in cols.items()}).T
    t["% weeks invested"] = np.nan
    t.loc["Model rule (after costs)", "% weeks invested"] = float((bt["n_held"] > 0).mean())
    return t


def threshold_sweep(oos, col, thresholds, cost, bench_fwd):
    rows = []
    for th in thresholds:
        bt = backtest(oos, col, th, cost, bench_fwd)
        s = perf_stats(bt["strategy"])
        rows.append({"threshold": th, "CAGR": s["CAGR"], "Sharpe": s["Sharpe"],
                     "Max drawdown": s["Max drawdown"], "avg stocks held": bt["n_held"].mean(),
                     "% weeks invested": (bt["n_held"] > 0).mean(),
                     "avg weekly cost": (bt["gross"] - bt["strategy"]).mean()})
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------
# The verdict
# ----------------------------------------------------------------------------
def verdict(dir_tab, boot, rng_tab, bt_sum, primary):
    """Turn the numbers into a plain-English judgement."""
    r = dir_tab.set_index("col")
    acc, auc = r.loc[primary, "accuracy"], r.loc[primary, "auc"]
    up_acc = r.loc["p_always_up", "accuracy"]
    best_base_auc = r.loc[["p_last_week", "p_hist_avg"], "auc"].max()
    strat = bt_sum.loc["Model rule (after costs)"]
    ew = bt_sum.loc["Hold all stocks (equal weight)"]

    v = {"accuracy": acc, "auc": auc, "always_up_accuracy": up_acc, **boot}
    if acc > 0.60 or auc > 0.65:
        v["level"] = "suspicious"
        v["headline"] = ("SUSPICIOUS: accuracy/AUC is too good for weekly stock moves. "
                         "Assume a bug or data leak; do not trust these predictions.")
    elif boot["auc_lo"] <= 0.5 or auc < 0.52:
        v["level"] = "no_edge"
        v["headline"] = ("No reliable edge. The model is not distinguishable from a coin "
                         "flip / simple baselines once you account for noise.")
    elif auc < 0.55:
        v["level"] = "tiny_edge"
        v["headline"] = ("A tiny, statistically detectable edge, too small to matter much for "
                         "any single stock-week. Treat predictions as a faint tilt, not a call.")
    else:
        v["level"] = "modest_edge"
        v["headline"] = ("A modest edge in testing. Still mostly noise per stock-week. "
                         "Confirm with the paper log before risking money.")

    v["details"] = [
        f"Out-of-sample accuracy {acc:.1%} vs {up_acc:.1%} for simply guessing 'up' every time "
        f"(difference {boot['accdiff']:+.1%}, 90% range {boot['accdiff_lo']:+.1%} to {boot['accdiff_hi']:+.1%}).",
        f"ROC AUC {auc:.3f} (0.5 = no skill; 90% range {boot['auc_lo']:.3f}-{boot['auc_hi']:.3f}); "
        f"best simple baseline AUC {best_base_auc:.3f}.",
        f"The 10-90% range caught the actual move {rng_tab.iloc[2]['coverage']:.0%} of the time "
        f"after calibration (target 80%).",
        f"Backtest after costs: {strat['CAGR']:.1%}/yr (Sharpe {strat['Sharpe']:.2f}) vs "
        f"{ew['CAGR']:.1%}/yr (Sharpe {ew['Sharpe']:.2f}) for just holding all the stocks. "
        + ("The rule beat that on risk-adjusted terms." if strat["Sharpe"] > ew["Sharpe"]
           else "The rule did NOT beat simply holding the stocks on risk-adjusted terms."),
    ]
    return v


# ----------------------------------------------------------------------------
# Experiments: "can we ask an easier question?"
# ----------------------------------------------------------------------------
EXPERIMENTS = {
    # name: (label column, forward-return column, horizon in days, description)
    "up_5": ("y_up_5", "fwd_ret_5", 5, "Up or down, 1 week (the original question)"),
    "rel_5": ("y_rel_5", "fwd_ret_5", 5, "Beats its peers, 1 week"),
    "up_20": ("y_up_20", "fwd_ret_20", 20, "Up or down, 1 month"),
    "rel_20": ("y_rel_20", "fwd_ret_20", 20, "Beats its peers, 1 month"),
}


def walk_forward_clf(panel, cfg, ycol, retcol, horizon, verbose=True):
    """Walk-forward for one yes/no question. The purge gap equals the horizon."""
    mc, fold = cfg["model"], cfg["experiments"]["fold_days"]
    lab = panel[panel[ycol].notna()].copy()
    dates = np.sort(lab["date_pos"].unique())
    lab["k"] = lab["date_pos"].map({d: i for i, d in enumerate(dates)})
    date_of_k = lab.groupby("k")["date"].first()
    start_k = int(np.searchsorted(date_of_k.values, np.datetime64(mc["first_test_date"])))
    start_k = max(start_k, int(252 * mc["min_train_years"]))
    gap = max(mc["gap_days"], horizon)
    out = []
    for i in range(start_k, len(dates), fold):
        train = lab[lab["k"] < i - gap]
        test = lab[(lab["k"] >= i) & (lab["k"] < i + fold) & lab["is_target"]]
        if len(test) == 0:
            continue
        p = predict_classifiers(fit_classifiers(train, ycol), test)
        out.append(pd.concat([test[["date", "ticker", "date_pos"]],
                              test[ycol].rename("y"), test[retcol].rename("ret"), p], axis=1))
    if verbose:
        print(f"    {ycol}: {len(out)} folds done")
    return pd.concat(out).reset_index(drop=True)


def _boot_auc(y, p, blocks, n_boot=100, seed=0):
    rng = np.random.default_rng(seed)
    ub, inv = np.unique(blocks, return_inverse=True)
    idx_by_block = pd.Series(np.arange(len(y))).groupby(inv).apply(np.array).values
    aucs = []
    for _ in range(n_boot):
        idx = np.concatenate([idx_by_block[b] for b in rng.integers(0, len(ub), len(ub))])
        if len(np.unique(y[idx])) > 1:
            aucs.append(_auc(y[idx], p[idx]))
    return float(np.quantile(aucs, 0.05)), float(np.quantile(aucs, 0.95))


def experiment_metrics(d, pcol, horizon, cost, top_n):
    """
    Metrics for one experiment over one time period.
      accuracy / naive : model accuracy vs always guessing the more common answer
      auc              : 0.5 = no skill (with a 90% block-bootstrap range)
      ic               : "information coefficient": each day, do higher-scored
                         stocks have higher forward returns? (rank correlation,
                         averaged). 0 = no skill; 0.03+ is considered decent.
      ic_t             : how many standard errors the IC is from zero (>2 = unlikely luck)
      top_excess       : yearly return of "hold the top N" minus "hold everything",
                         after trading costs
    """
    if len(d) < 200:
        return None
    y, p = d["y"].astype(int).values, d[pcol].values
    lo, hi = _boot_auc(y, p, d["date_pos"].values // horizon)
    ic_day = d.groupby("date_pos").apply(lambda g: g[pcol].corr(g["ret"], method="spearman")
                                         if len(g) >= 8 else np.nan).dropna()
    ic_blk = ic_day.groupby(ic_day.index // horizon).mean()      # non-overlapping blocks
    ic_t = ic_blk.mean() / (ic_blk.std() / np.sqrt(len(ic_blk))) if len(ic_blk) > 3 else np.nan

    reb, prev, ex = np.sort(d["date_pos"].unique())[::horizon], set(), []
    for dp in reb:
        day = d[d["date_pos"] == dp]
        if len(day) < 2 * top_n:
            continue
        top = day.nlargest(top_n, pcol)
        held = set(top["ticker"])
        turnover = len(held ^ prev) / top_n                     # buys + sells, as a fraction
        ex.append(top["ret"].mean() - day["ret"].mean() - cost * turnover)
        prev = held
    ex = pd.Series(ex)
    per_year = 252 / horizon
    return {"n": len(d), "accuracy": float(((p > 0.5) == y).mean()),
            "naive": float(max(y.mean(), 1 - y.mean())),
            "auc": float(_auc(y, p)), "auc_lo": lo, "auc_hi": hi,
            "ic": float(ic_day.mean()), "ic_t": float(ic_t),
            "top_excess": float(ex.mean() * per_year) if len(ex) else np.nan,
            "top_sharpe": float(ex.mean() / ex.std() * np.sqrt(per_year)) if len(ex) > 3 and ex.std() else np.nan}


def run_experiments(panel, cfg, main_oos):
    """Run every experiment; return {name: result dict}."""
    ec = cfg["experiments"]
    split = pd.Timestamp(ec["final_exam_start"])
    cost, top_n = cfg["backtest"]["cost_per_trade"], ec["top_n"]
    results = {}
    for name, (ycol, retcol, h, desc) in EXPERIMENTS.items():
        if name == "up_5":      # already computed by the main walk-forward: reuse it
            oos = main_oos.rename(columns={"y_up": "y", "fwd_ret": "ret"})[
                ["date", "ticker", "date_pos", "y", "ret", "p_logit", "p_lgbm"]]
        else:
            oos = walk_forward_clf(panel, cfg, ycol, retcol, h)
        dev, exam = oos[oos["date"] < split], oos[oos["date"] >= split]
        # Pick the better model using DEVELOPMENT data only. The final exam is
        # never used to choose anything.
        best = max(["p_logit", "p_lgbm"], key=lambda c: _auc(dev["y"].astype(int), dev[c]))
        md = experiment_metrics(dev, best, h, cost, top_n)
        me = experiment_metrics(exam, best, h, cost, top_n)
        if md is None:
            continue
        if md["auc"] > 0.65:
            level, label = "suspicious", "SUSPICIOUS (too good: likely a bug)"
        elif md["auc_lo"] > 0.5 and md["ic_t"] > 2:
            if me and me["auc"] > 0.5 and me["ic"] > 0:
                level, label = "confirmed", "Small edge: passed development AND the final exam"
            else:
                level, label = "failed_exam", "Looked good in development, FAILED the final exam"
        else:
            level, label = "no_edge", "No edge"
        results[name] = {"desc": desc, "horizon": h, "ycol": ycol, "best": best,
                         "dev": md, "exam": me, "level": level, "label": label}
    return results
