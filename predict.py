"""
predict.py: run after the market closes. For each ticker it prints:
  * P(up over the next 5 trading days)
  * a predicted range: 10th / 50th / 90th percentile of the 5-day return
  * a plain-English confidence note
  * whether earnings fall inside the week
Holdings (NBIS, CRWV) are listed first.

It also writes LATEST.md (for the GitHub app) and appends to paper_log.csv.

    python predict.py
    python predict.py --synthetic   # offline test with fake data
"""
import argparse
from datetime import datetime, timezone

import joblib
import numpy as np
import pandas as pd

from stockpred.config import ROOT, load_config
from stockpred.features import build_panel
from stockpred.models import predict_all

LOG_PATH = ROOT / "paper_log.csv"
LOG_COLUMNS = ["logged_at_utc", "asof_date", "ticker", "is_holding", "p_up", "q10", "q50", "q90",
               "earnings_in_week", "model_trained_on", "target_date", "actual_ret", "actual_up",
               "direction_correct", "in_range"]


def confidence_note(p, verdict_level, earnings_week, short, stale):
    lean = abs(p - 0.5)
    if lean < 0.03:
        note = "coin flip"
    elif lean < 0.06:
        note = f"slight lean {'up' if p > .5 else 'down'}"
    else:
        note = f"lean {'up' if p > .5 else 'down'}"
    if verdict_level in ("no_edge", "suspicious"):
        note += " (model has no proven edge)"
    extras = []
    if earnings_week == 1:
        extras.append("earnings this week: bigger move likely")
    if short:
        extras.append("limited history")
    if stale:
        extras.append("STALE data")
    return note + ("; " + "; ".join(extras) if extras else "")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--synthetic", action="store_true")
    args = ap.parse_args()
    cfg = load_config()
    path = ROOT / "models" / "model.joblib"
    if not path.exists():
        raise SystemExit("No trained model found. Run `python train.py` first.")
    b = joblib.load(path)

    if args.synthetic:
        from stockpred.synthetic import make_synthetic
        prices, earnings, fred = make_synthetic(cfg)
    else:
        from stockpred.data import load_all
        prices, earnings, fred = load_all(cfg, max_age_hours=1)   # we want today's close

    targets = [t for t in cfg["targets"] if t in prices]
    panel = build_panel(prices, earnings, fred, targets, cfg["model"]["horizon"])
    last_market_day = prices["SPY"].index.max()
    latest = panel.loc[panel.groupby("ticker")["date"].idxmax()].copy()

    p = predict_all(b["models"], latest)
    adj = b["cqr_adj"]
    latest["p_up"] = p[b["primary"]].values
    latest["q10"] = ((p["q10_z"] - adj) * latest["scale"]).values
    latest["q50"] = (p["q50_z"] * latest["scale"]).values
    latest["q90"] = ((p["q90_z"] + adj) * latest["scale"]).values

    hist = panel.groupby("ticker")["n_days_history"].max()
    short_days = cfg["data"]["short_history_days"]
    holdings = cfg["holdings"]
    level = b["verdict"]["level"]

    rows = []
    for _, r in latest.iterrows():
        t = r["ticker"]
        stale = r["date"] < last_market_day
        ew = r["earnings_in_week"] if not np.isnan(r["days_to_earnings"]) else np.nan
        rows.append({
            "ticker": t, "holding": t in holdings, "group": cfg["groups"].get(t, ""),
            "date": r["date"].date(), "p_up": r["p_up"], "q10": r["q10"], "q50": r["q50"], "q90": r["q90"],
            "earnings": "unknown" if np.isnan(ew) else ("YES" if ew == 1 else "no"),
            "earnings_flag": ew,
            "short": hist.get(t, 0) < short_days,
            "note": confidence_note(r["p_up"], level, ew, hist.get(t, 0) < short_days, stale),
        })
    out = pd.DataFrame(rows)
    out["hold_order"] = out["ticker"].map({t: i for i, t in enumerate(holdings)}).fillna(99)
    out = out.sort_values(["hold_order", "p_up"], ascending=[True, False]).reset_index(drop=True)

    # ------------------------------------------------------------- console
    asof = last_market_day.date()
    print(f"\nPredictions for the 5 trading days after {asof}   (model: {b['primary_name']}, "
          f"trained {b['trained_on']})")
    print(f"Model verdict from testing: {b['verdict']['headline']}\n")
    print(f"{'':2}{'Ticker':<7}{'P(up)':>7}   {'Range 10%..90%':<18}{'Middle':>8}  {'Earnings':<9} Note")
    for _, r in out.iterrows():
        star = "* " if r["holding"] else "  "
        print(f"{star}{r['ticker']:<7}{r['p_up']:>6.0%}   {r['q10']:>+6.1%} .. {r['q90']:<+7.1%}  "
              f"{r['q50']:>+6.1%}  {r['earnings']:<9} {r['note']}")
    print(f"\nAverage P(up) across all stocks: {out['p_up'].mean():.0%}. Predictions move together, "
          "so they are partly a single market call.")

    # ------------------------------------------------------------ LATEST.md
    write_latest_md(out, b, asof, args.synthetic)

    # ------------------------------------------------------- paper_log.csv
    append_log(out, b, asof)


def write_latest_md(out, b, asof, synthetic):
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    lines = [f"# Weekly move predictions, as of {asof} close", "",
             f"_Generated {now}. Covers the next 5 trading days. Model: {b['primary_name']}, "
             f"trained {b['trained_on']}._", ""]
    if synthetic:
        lines += ["> **SYNTHETIC TEST DATA.** These are not real predictions.", ""]
    lines += [f"> **Honesty check:** {b['verdict']['headline']}", "",
              "| | Ticker | P(up) | Range (10%–90%) | Middle | Earnings in week | Note |",
              "|---|---|---|---|---|---|---|"]
    for _, r in out.iterrows():
        lines.append(f"| {'★' if r['holding'] else ''} | **{r['ticker']}** | {r['p_up']:.0%} | "
                     f"{r['q10']:+.1%} to {r['q90']:+.1%} | {r['q50']:+.1%} | {r['earnings']} | {r['note']} |")
    lines += ["", f"Average P(up) across all stocks: **{out['p_up'].mean():.0%}**.", "",
              "**How to read this:** P(up) is the chance the stock is higher 5 trading days from now. "
              "The range should contain the actual move about 80% of the time. "
              "Anything between 47% and 53% is a coin flip. Not investment advice.", "",
              "<!-- SCORE START -->", "_Paper-trading score appears here once predictions are a week old._",
              "<!-- SCORE END -->", ""]
    (ROOT / "LATEST.md").write_text("\n".join(lines))
    print(f"\nWrote LATEST.md")


def append_log(out, b, asof):
    new = pd.DataFrame({
        "logged_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M"),
        "asof_date": str(asof), "ticker": out["ticker"], "is_holding": out["holding"].astype(int),
        "p_up": out["p_up"].round(4), "q10": out["q10"].round(4), "q50": out["q50"].round(4),
        "q90": out["q90"].round(4), "earnings_in_week": out["earnings_flag"],
        "model_trained_on": b["trained_on"],
    })
    if LOG_PATH.exists():
        log = pd.read_csv(LOG_PATH, dtype={"asof_date": str})
        # Don't log the same day twice (e.g. a holiday or a manual re-run)
        key = set(zip(log["asof_date"], log["ticker"]))
        new = new[[(a, t) not in key for a, t in zip(new["asof_date"], new["ticker"])]]
        log = pd.concat([log, new], ignore_index=True)
    else:
        log = new
    log = log.reindex(columns=LOG_COLUMNS)
    log.to_csv(LOG_PATH, index=False)
    print(f"paper_log.csv: added {len(new)} rows ({len(log)} total)")


if __name__ == "__main__":
    main()
