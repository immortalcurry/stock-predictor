"""
score_log.py: fill in what actually happened for paper_log.csv predictions, then
print a live scorecard (and add it to LATEST.md).

  * weekly predictions are scored once they are 5 trading days old
  * tomorrow's price range is scored after 1 trading day
  * "beats peers" is scored against the median of the stocks in the log that day

This is the REAL test. The backtest can fool you, but predictions written
down before the fact can't. Give it 4-8 weeks before trusting anything.

    python score_log.py
    python score_log.py --synthetic
"""
import argparse
import re

import numpy as np
import pandas as pd

from predict import LOG_COLUMNS
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
    log = log.reindex(columns=LOG_COLUMNS)      # adds any columns from newer versions

    todo_w = log["actual_ret"].isna()
    todo_d = log["actual_ret_1d"].isna() & log["q10_1d"].notna()
    if (todo_w | todo_d).any():
        if args.synthetic:
            from stockpred.synthetic import make_synthetic
            prices = make_synthetic(cfg)[0]
        else:
            from stockpred.data import download_prices
            d = cfg["data"]
            tickers = sorted(set(log.loc[todo_w | todo_d, "ticker"]) | {"SPY"})
            # Same history length as train.py so we never overwrite the shared cache with less data.
            prices = download_prices(tickers, d["years"], d["cache_dir"], 1)
        cal = prices["SPY"].index
        h = cfg["model"]["horizon"]

        def realized(ticker, asof, days):
            """Return (target_date, return) `days` trading days after asof, or None if not known yet."""
            a = pd.Timestamp(asof)
            if ticker not in prices or a not in cal:
                return None
            k = cal.get_loc(a) + days
            c = prices[ticker]["Close"]
            if k >= len(cal) or a not in c.index or cal[k] not in c.index:
                return None
            return cal[k], float(c[cal[k]] / c[a] - 1)

        for i in log.index[todo_w]:
            res = realized(log.at[i, "ticker"], log.at[i, "asof_date"], h)
            if res is None:
                continue               # the week isn't over yet
            tgt, ret = res
            log.at[i, "target_date"] = str(tgt.date())
            log.at[i, "actual_ret"] = round(ret, 5)
            log.at[i, "actual_up"] = int(ret > 0)
            log.at[i, "direction_correct"] = int((log.at[i, "p_up"] > 0.5) == (ret > 0))
            log.at[i, "in_range"] = int(log.at[i, "q10"] <= ret <= log.at[i, "q90"])
        for i in log.index[todo_d]:
            res = realized(log.at[i, "ticker"], log.at[i, "asof_date"], 1)
            if res is None:
                continue
            log.at[i, "actual_ret_1d"] = round(res[1], 5)
            log.at[i, "in_range_1d"] = int(log.at[i, "q10_1d"] <= res[1] <= log.at[i, "q90_1d"])

        # "Beats peers": compare each stock with that day's median, once every
        # stock logged that day has a weekly result.
        for asof, g in log.groupby("asof_date"):
            if g["actual_ret"].isna().any() or g["p_beat_peers_5d"].isna().all():
                continue
            rel = g["actual_ret"] - g["actual_ret"].median()
            ok = g["p_beat_peers_5d"].notna()
            log.loc[g.index[ok], "beat_peers_correct"] = ((g["p_beat_peers_5d"] > 0.5) == (rel > 0))[ok].astype(int)
        log.to_csv(LOG_PATH, index=False)

    md = scorecard(log)
    print(md)
    latest = ROOT / "LATEST.md"
    if latest.exists():
        txt = latest.read_text()
        txt = re.sub(r"<!-- SCORE START -->.*<!-- SCORE END -->",
                     lambda m: "<!-- SCORE START -->\n" + md + "\n<!-- SCORE END -->", txt, flags=re.S)
        latest.write_text(txt)


def scorecard(log):
    s = log.dropna(subset=["actual_ret"])
    d1 = log.dropna(subset=["actual_ret_1d"])
    head = "## Paper-trading score (live, predictions made before the fact)"
    if len(s) == 0 and len(d1) == 0:
        return head + "\n\n_No predictions are old enough to score yet._"
    lines = [head, "", "| Measure | Model | Benchmark |", "|---|---|---|"]
    if len(d1):
        lines.append(f"| Tomorrow's close inside the predicted range | {d1['in_range_1d'].mean():.0%} "
                     f"| 80% target (n={len(d1):,}) |")
    if len(s):
        acc, up_acc = s["direction_correct"].mean(), s["actual_up"].mean()
        brier = np.mean((s["p_up"] - s["actual_up"]) ** 2)
        lines += [f"| 1-week close inside the predicted range | {s['in_range'].mean():.0%} | 80% target (n={len(s):,}) |",
                  f"| 1-week direction accuracy | {acc:.1%} | {up_acc:.1%} (always guess up) |",
                  f"| Brier score (lower = better) | {brier:.4f} | 0.2500 (always say 50%) |"]
        bp = s.dropna(subset=["beat_peers_correct"])
        if len(bp):
            lines.append(f"| \"Beats peers\" accuracy (1 week) | {bp['beat_peers_correct'].mean():.1%} | 50% (coin flip) |")
        for t, g in s[s["is_holding"] == 1].groupby("ticker"):
            lines.append(f"| {t}: direction right / in range | {g['direction_correct'].mean():.0%} / "
                         f"{g['in_range'].mean():.0%} | n={len(g)} |")
    n_days = log.loc[log["actual_ret"].notna() | log["actual_ret_1d"].notna(), "asof_date"].nunique()
    weeks = n_days / 5
    lines += ["", f"_About {weeks:.1f} weeks of scored data. Daily predictions overlap and stocks move together, "
              "so this is far fewer independent tests than the row count suggests. "
              + ("Too early to judge. Keep waiting._" if weeks < 6 else
                 "Getting meaningful, but a handful of market weeks can still dominate._")]
    return "\n".join(lines)


if __name__ == "__main__":
    main()
