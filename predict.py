"""
predict.py: run after the market closes. For each ticker it prints:
  * price estimates: likely low / middle / high for TOMORROW's close and for the
    close 5 trading days from now
  * P(up over the next 5 trading days) and P(beats its peers)
  * the predicted range: 10th / 50th / 90th percentile of the 5-day return
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
               "direction_correct", "in_range",
               # added in v2
               "last_close", "q10_1d", "q90_1d", "p_beat_peers_5d", "p_beat_peers_20d",
               "actual_ret_1d", "in_range_1d", "beat_peers_correct"]


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
    pool = [t for t in cfg["pool"] if t in prices]
    # Features are built on the whole pool (peer ranks need everyone); we only
    # report on the target stocks.
    panel = build_panel(prices, earnings, fred, pool, cfg["model"]["horizon"], targets=targets)
    panel = panel[panel["is_target"]]
    last_market_day = prices["SPY"].index.max()
    latest = panel.loc[panel.groupby("ticker")["date"].idxmax()].copy()

    p = predict_all(b["models"], latest)
    adj = b["cqr_adj"]
    latest["p_up"] = p[b["primary"]].values
    latest["q10"] = ((p["q10_z"] - adj) * latest["scale"]).values
    latest["q50"] = (p["q50_z"] * latest["scale"]).values
    latest["q90"] = ((p["q90_z"] + adj) * latest["scale"]).values
    # Tomorrow's range
    adj1 = b.get("cqr_adj_1d", 0.0)
    latest["q10_1d"] = ((p["q10_1d_z"] - adj1) * latest["scale_1"]).values
    latest["q50_1d"] = (p["q50_1d_z"] * latest["scale_1"]).values
    latest["q90_1d"] = ((p["q90_1d_z"] + adj1) * latest["scale_1"]).values
    # "Beats its peers" probabilities from the experiment models
    exps = b.get("experiments", {})
    for name, col in [("rel_5", "p_rel_5"), ("rel_20", "p_rel_20")]:
        latest[col] = (exps[name]["model"].predict_proba(latest[b["features"]])[:, 1]
                       if name in exps else np.nan)

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
            "close": r["close"], "q10_1d": r["q10_1d"], "q50_1d": r["q50_1d"], "q90_1d": r["q90_1d"],
            "p_rel_5": r["p_rel_5"], "p_rel_20": r["p_rel_20"],
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
    print(f"{'':2}{'Ticker':<7}{'Close':>9}  {'Tomorrow (low..high)':<22}{'In 1 week (low / mid / high)':<32}"
          f"{'P(up)':>6} {'P(beat)':>8}  {'Earn':<8} Note")
    for _, r in out.iterrows():
        star = "* " if r["holding"] else "  "
        c = r["close"]
        print(f"{star}{r['ticker']:<7}{money(c):>9}  "
              f"{money(c * (1 + r['q10_1d'])) + ' .. ' + money(c * (1 + r['q90_1d'])):<22}"
              f"{money(c * (1 + r['q10'])) + ' / ' + money(c * (1 + r['q50'])) + ' / ' + money(c * (1 + r['q90'])):<32}"
              f"{r['p_up']:>6.0%} {pct(r['p_rel_5']):>8}  {r['earnings']:<8} {r['note']}")
    print(f"\nAverage P(up) across all stocks: {out['p_up'].mean():.0%}. Predictions move together, "
          "so they are partly a single market call.")

    # ------------------------------------------------------------ LATEST.md
    write_latest_md(out, b, asof, args.synthetic)

    # ------------------------------------------------------- paper_log.csv
    append_log(out, b, asof)


def money(x):
    """$1,234 for big prices, $12.34 for small ones."""
    return f"${x:,.0f}" if x >= 1000 else f"${x:.2f}"


def pct(x):
    return "n/a" if pd.isna(x) else f"{x:.0%}"


def write_latest_md(out, b, asof, synthetic):
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    lines = [f"# Weekly move predictions, as of {asof} close", "",
             f"_Generated {now}. Covers the next 5 trading days. Model: {b['primary_name']}, "
             f"trained {b['trained_on']}._", ""]
    if synthetic:
        lines += ["> **SYNTHETIC TEST DATA.** These are not real predictions.", ""]
    exps = b.get("experiments", {})
    lines += [f"> **Honesty check on direction (P(up)):** {b['verdict']['headline']}", "",
              "## Price estimates", "",
              "Likely low and high are the 10th and 90th percentiles: the close should land "
              "between them about 4 times out of 5.", "",
              "| | Ticker | Last close | Tomorrow: low – high | In 1 week: low | Middle | High |",
              "|---|---|---|---|---|---|---|"]
    for _, r in out.iterrows():
        c = r["close"]
        lines.append(f"| {'★' if r['holding'] else ''} | **{r['ticker']}** | {money(c)} | "
                     f"{money(c * (1 + r['q10_1d']))} – {money(c * (1 + r['q90_1d']))} | "
                     f"{money(c * (1 + r['q10']))} | {money(c * (1 + r['q50']))} | {money(c * (1 + r['q90']))} |")
    lines += ["", "## Direction and ranking", "",
              "| | Ticker | P(up) 1 wk | Range 1 wk | P(beats peers) 1 wk | P(beats peers) 1 mo | Earnings in week | Note |",
              "|---|---|---|---|---|---|---|---|"]
    for _, r in out.iterrows():
        lines.append(f"| {'★' if r['holding'] else ''} | **{r['ticker']}** | {r['p_up']:.0%} | "
                     f"{r['q10']:+.1%} to {r['q90']:+.1%} | {pct(r['p_rel_5'])} | {pct(r['p_rel_20'])} | "
                     f"{r['earnings']} | {r['note']} |")
    lines += ["", f"Average P(up) across all stocks: **{out['p_up'].mean():.0%}**.", ""]
    if exps:
        lines += ["**How much to trust \"beats peers\"** (chance the stock does better than the average "
                  "of ~100 chip/tech stocks):", ""]
        for name, label in [("rel_5", "1 week"), ("rel_20", "1 month")]:
            if name in exps:
                lines.append(f"- {label}: {exps[name]['label']}")
        lines.append("")
    lines += ["**How to read this:** the price ranges are the most reliable part. P(up) is the chance the "
              "stock is higher 5 trading days from now; anything between 47% and 53% is a coin flip. "
              "The middle price is usually close to the last close because nobody can reliably call "
              "direction. Not investment advice.", "",
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
        "last_close": out["close"].round(4), "q10_1d": out["q10_1d"].round(4),
        "q90_1d": out["q90_1d"].round(4), "p_beat_peers_5d": out["p_rel_5"].round(4),
        "p_beat_peers_20d": out["p_rel_20"].round(4),
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
