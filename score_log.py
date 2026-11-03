"""
score_log.py: fill in what actually happened for paper_log.csv predictions that
are at least 5 trading days old, then print a live scorecard (and add it to
LATEST.md).

This is the REAL test. The backtest can fool you, but predictions written
down before the fact can't. Give it 4-8 weeks before trusting anything.

    python score_log.py
    python score_log.py --synthetic
"""
import argparse
import re

import numpy as np
import pandas as pd

from stockpred.config import ROOT, load_config

LOG_PATH = ROOT / "paper_log.csv"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--synthetic", action="store_true")
    args = ap.parse_args()
    cfg = load_config()
    if not LOG_PATH.exists():
        print("No paper_log.csv yet. Run predict.py first.")
        return
    log = pd.read_csv(LOG_PATH, dtype={"asof_date": str, "target_date": str})
    todo = log["actual_ret"].isna()
    if todo.any():
        if args.synthetic:
            from stockpred.synthetic import make_synthetic
            prices = make_synthetic(cfg)[0]
        else:
            from stockpred.data import download_prices
            d = cfg["data"]
            tickers = sorted(set(log.loc[todo, "ticker"]) | {"SPY"})
            # Same history length as train.py so we never overwrite the shared cache with less data.
            prices = download_prices(tickers, d["years"], d["cache_dir"], 1)
        cal = prices["SPY"].index
        h = cfg["model"]["horizon"]
        for i in log.index[todo]:
            t, a = log.at[i, "ticker"], pd.Timestamp(log.at[i, "asof_date"])
            if t not in prices or a not in cal:
                continue
            k = cal.get_loc(a) + h
            if k >= len(cal):
                continue               # the week isn't over yet
            tgt = cal[k]
            c = prices[t]["Close"]
            if a not in c.index or tgt not in c.index:
                continue
            ret = c[tgt] / c[a] - 1
            log.at[i, "target_date"] = str(tgt.date())
            log.at[i, "actual_ret"] = round(float(ret), 5)
            log.at[i, "actual_up"] = int(ret > 0)
            log.at[i, "direction_correct"] = int((log.at[i, "p_up"] > 0.5) == (ret > 0))
            log.at[i, "in_range"] = int(log.at[i, "q10"] <= ret <= log.at[i, "q90"])
        log.to_csv(LOG_PATH, index=False)

    s = log.dropna(subset=["actual_ret"])
    md = scorecard(s)
    print(md)
    latest = ROOT / "LATEST.md"
    if latest.exists():
        txt = latest.read_text()
        txt = re.sub(r"<!-- SCORE START -->.*<!-- SCORE END -->",
                     "<!-- SCORE START -->\n" + md + "\n<!-- SCORE END -->", txt, flags=re.S)
        latest.write_text(txt)


def scorecard(s):
    if len(s) == 0:
        return "## Paper-trading score\n\n_No predictions are old enough to score yet (each needs 5 trading days)._"
    n_days = s["asof_date"].nunique()
    acc = s["direction_correct"].mean()
    up_acc = s["actual_up"].mean()          # what "always up" would have scored
    brier = np.mean((s["p_up"] - s["actual_up"]) ** 2)
    cov = s["in_range"].mean()
    lines = ["## Paper-trading score (live, predictions made before the fact)", "",
             f"Scored {len(s):,} predictions from {n_days} trading days "
             f"({s['asof_date'].min()} to {s['asof_date'].max()}).", "",
             "| Measure | Model | Benchmark |", "|---|---|---|",
             f"| Direction accuracy | {acc:.1%} | {up_acc:.1%} (always guess up) |",
             f"| Brier score (lower = better) | {brier:.4f} | 0.2500 (always say 50%) |",
             f"| Actual move inside 10-90% range | {cov:.0%} | 80% target |"]
    h = s[s["is_holding"] == 1]
    for t, g in h.groupby("ticker"):
        lines.append(f"| {t}: accuracy / in range | {g['direction_correct'].mean():.0%} / "
                     f"{g['in_range'].mean():.0%} | n={len(g)} |")
    weeks = n_days / 5
    lines += ["", f"_About {weeks:.1f} weeks of data. Daily predictions overlap and stocks move together, "
              "so this is far fewer independent tests than the row count suggests. "
              + ("Too early to judge. Keep waiting._" if weeks < 6 else
                 "Getting meaningful, but a handful of market weeks can still dominate._")]
    return "\n".join(lines)


if __name__ == "__main__":
    main()
