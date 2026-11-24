"""
The beginner-friendly "Simple view".

It answers three plain questions per stock, using only price history:

  1. Which way is it heading?   The TREND over months (not a prediction of
                                 next week, which nobody can reliably call).
  2. How long might that last?  How long past trends like this one lasted,
                                 measured across ~100 chip/tech stocks over 15 years.
  3. Is today's move normal?    Compares the latest week with this stock's usual
                                 week, so a -10% week in a wild stock doesn't cause panic.

Definitions (kept simple on purpose):
  * Uptrend   = the 50-day average price is ABOVE the 200-day average price.
  * Downtrend = the 50-day average price is BELOW the 200-day average price.
  * "Dipping" / "bouncing" = today's price is on the other side of the
    50-day average, i.e. a short-term move against the bigger trend.
These averages smooth out daily noise, so the trend only flips every few months.
"""
import html
from datetime import datetime, timezone

import numpy as np
import pandas as pd

QUARTER = 63   # trading days in ~3 months
MONTH = 21     # trading days in ~1 month


# ----------------------------------------------------------------------------
# Trend + trend history
# ----------------------------------------------------------------------------
def trend_series(close):
    ma50 = close.rolling(50).mean()
    ma200 = close.rolling(200).mean()
    state = pd.Series(np.where(ma50 > ma200, 1.0, -1.0), index=close.index).where(ma200.notna())
    return state, ma50, ma200


def trend_runs(state):
    """Each unbroken stretch of the same trend: its direction, length (trading days)
    and whether it's still going. The first stretch is dropped because its true
    start happened before we had enough data to see it."""
    s = state.dropna()
    if s.empty:
        return pd.DataFrame(columns=["state", "length", "ongoing"])
    run_id = s.ne(s.shift()).cumsum()
    g = s.groupby(run_id)
    df = pd.DataFrame({"state": g.first(), "length": g.size()}).reset_index(drop=True)
    df["ongoing"] = False
    df.loc[df.index[-1], "ongoing"] = True
    return df.iloc[1:]


def history_stats(prices, tickers):
    """What happened after uptrends and downtrends, pooled over many stocks."""
    rows, runs = [], []
    for t in tickers:
        c = prices[t]["Close"].astype(float)
        if len(c) < 260:
            continue
        state, ma50, _ = trend_series(c)
        rows.append(pd.DataFrame({
            "state": state,
            "dip": (c < ma50).astype(float),
            "week": c.shift(-5) / c - 1,
            "quarter": c.shift(-QUARTER) / c - 1,
        }).dropna(subset=["state"]))
        runs.append(trend_runs(state))
    allr = pd.concat(rows)
    out = {"runs": pd.concat(runs, ignore_index=True), "by_state": {}}
    for s in (1.0, -1.0):
        d = allr[allr["state"] == s]
        q, w = d["quarter"].dropna(), d["week"].dropna()
        out["by_state"][s] = {
            "p_up_quarter": float((q > 0).mean()), "median_quarter": float(q.median()),
            "p_up_week": float((w > 0).mean()), "n": len(q),
        }
    return out


def remaining_trend(runs, state, age):
    """Among past trends of the same direction that had already lasted `age` days:
    how many more days did they typically last, and how many were still going
    a month later? Returns (median_more_days, p_still_in_1_month) or Nones."""
    same = runs[(runs["state"] == state) & (runs["length"] > age)]
    done = same[~same["ongoing"]]
    if len(done) < 15:
        return None, None
    more = float((done["length"] - age).median())
    # Ongoing trends that are younger than age+1 month can't tell us yet; skip them.
    decided = same[~same["ongoing"] | (same["length"] > age + MONTH)]
    p_month = float((decided["length"] > age + MONTH).mean())
    return more, p_month


# ----------------------------------------------------------------------------
# Plain-English helpers
# ----------------------------------------------------------------------------
def human_days(d):
    """Trading days -> 'about 3 weeks'."""
    if d is None or np.isnan(d):
        return "unknown"
    if d < 7:
        n = max(1, round(d))
        return f"about {n} day{'s' if n != 1 else ''}"
    if d < 40:
        n = max(1, round(d / 5))
        return f"about {n} week{'s' if n != 1 else ''}"
    n = max(1, round(d / 21))
    return f"about {n} month{'s' if n != 1 else ''}"


def short_days(d):
    """Trading days -> '3 mo' / '2 wk' / '4 d' for tight spaces."""
    if d is None or np.isnan(d):
        return ""
    return f"{round(d)} d" if d < 7 else f"{round(d / 5)} wk" if d < 40 else f"{round(d / 21)} mo"


def money(x):
    return f"${x:,.0f}" if x >= 1000 else f"${x:,.2f}"


def wildness(typical_week):
    if typical_week < 0.04:
        return "Calm"
    if typical_week < 0.07:
        return "Normal"
    if typical_week < 0.11:
        return "Wild"
    return "Very wild"


def move_size_word(ret, typical):
    z = abs(ret) / typical if typical > 0 else 0
    if z < 1:
        return "a normal-sized move"
    if z < 2:
        return "bigger than usual"
    return "an unusually big move"


# ----------------------------------------------------------------------------
# One card per stock
# ----------------------------------------------------------------------------
def stock_card(ticker, close, row, hist, holding, short_history, cal_days_to_date):
    """Everything the Simple view says about one stock, as a dict."""
    c = close.astype(float)
    price = float(c.iloc[-1])
    state, ma50, _ = trend_series(c)
    vol = float(row["vol_20"])
    typical_week = vol * np.sqrt(5)
    ret_5 = float(c.iloc[-1] / c.iloc[-6] - 1) if len(c) > 6 else np.nan
    card = {
        "ticker": ticker, "holding": holding, "price": price, "short_history": short_history,
        "ret_5": ret_5, "typical_week": typical_week, "wild": wildness(typical_week),
        "move_word": move_size_word(ret_5, typical_week) if not np.isnan(ret_5) else "",
        "low_week": price * (1 + row["q10"]), "high_week": price * (1 + row["q90"]),
        # A random-walk rule of thumb: a move of size m usually needs about (m / daily vol)^2 days
        "days_for_10pct": (0.10 / vol) ** 2 if vol > 0 else np.nan,
        "earn_days": row["days_to_earnings"], "earn_date": None,
    }
    if not np.isnan(card["earn_days"]):
        card["earn_date"] = cal_days_to_date(int(card["earn_days"]))

    s = state.iloc[-1]
    if np.isnan(s):
        card.update(trend=0, arrow="⚪", label="Not enough history yet",
                    trend_text="It needs about 10 months of prices before a trend can be measured.")
        return card

    run_id = state.ne(state.shift()).cumsum()
    age = int((run_id == run_id.iloc[-1]).sum())
    first_run = run_id.iloc[-1] == run_id.dropna().iloc[0]   # trend started before our data
    dip = price < float(ma50.iloc[-1])
    up = s > 0
    card["trend"] = 1 if up else -1
    card["arrow"] = "🟢 ⬆" if up else "🔴 ⬇"
    card["label"] = ("Uptrend" if up else "Downtrend") + (
        (", but dipping right now" if dip else ", still going") if up else
        (", but bouncing right now" if not dip else ", still going"))
    since = c.index[-age].strftime("%b %Y")
    card["age_text"] = (f"{'at least ' if first_run else ''}{human_days(age)} "
                        f"(since {since})")
    card["age_short"] = ("≥" if first_run else "") + short_days(age)
    more, p_month = remaining_trend(hist["runs"], s, age)
    card["more_days"], card["p_month"] = more, p_month
    word = "uptrends" if up else "downtrends"
    if more is not None:
        card["how_long"] = (f"Past {word} in these stocks that had already lasted this long went on for "
                            f"{human_days(more)} more (middle case). Chance it's still "
                            f"{'an uptrend' if up else 'a downtrend'} in 1 month: {p_month:.0%}.")
    else:
        card["how_long"] = f"Not enough past {word} this old to say how long it usually lasts."
    return card


def market_weather(prices):
    spy = prices["SPY"]["Close"].astype(float)
    state, _, _ = trend_series(spy)
    vix = float(prices["^VIX"]["Close"].iloc[-1]) if "^VIX" in prices else np.nan
    if np.isnan(vix):
        mood = "unknown"
    elif vix < 15:
        mood = "Calm"
    elif vix < 20:
        mood = "Normal"
    elif vix < 30:
        mood = "Nervous"
    else:
        mood = "Fearful"
    smh = prices["SMH"]["Close"].astype(float) if "SMH" in prices else spy
    return {"spy_trend": "Uptrend" if state.iloc[-1] > 0 else "Downtrend",
            "spy_week": float(spy.iloc[-1] / spy.iloc[-6] - 1),
            "chips_week": float(smh.iloc[-1] / smh.iloc[-6] - 1),
            "vix": vix, "mood": mood}


# ----------------------------------------------------------------------------
# Rendering
# ----------------------------------------------------------------------------
def _card_lines(cd):
    lines = []
    lines.append(f"**Price:** {money(cd['price'])} · last 5 days: {cd['ret_5']:+.1%} "
                 f"({cd['move_word']}; a normal week for this stock is about ±{cd['typical_week']:.0%})")
    if cd.get("trend"):
        lines.append(f"**Direction (big picture):** {cd['label'].lower()}. It has been "
                     f"{'up' if cd['trend'] > 0 else 'down'}trending for {cd['age_text']}.")
        lines.append(f"**How long it might last:** {cd['how_long']}")
    else:
        lines.append(f"**Direction:** {cd['trend_text']}")
    lines.append(f"**Next week:** nobody can reliably call up or down. Likely somewhere between "
                 f"{money(cd['low_week'])} and {money(cd['high_week'])} (4 times out of 5).")
    lines.append(f"**How wild:** {cd['wild']}. A 10% swing can happen in "
                 f"{human_days(cd['days_for_10pct'])}.")
    if cd["earn_date"] is not None:
        soon = " ⚠️ expect a bigger-than-usual move" if cd["earn_days"] <= 5 else ""
        lines.append(f"**Next earnings report:** in {human_days(cd['earn_days'])} "
                     f"(around {cd['earn_date']}){soon}.")
    if cd["short_history"]:
        lines.append("_Newer stock: less history, so treat everything here as rougher._")
    return lines


def history_sentence(hist):
    u, d = hist["by_state"][1.0], hist["by_state"][-1.0]
    return (f"Across ~100 chip and tech stocks since 2011: when a stock was in an **uptrend**, it was higher "
            f"3 months later **{u['p_up_quarter']:.0%}** of the time (middle result {u['median_quarter']:+.0%}). "
            f"In a **downtrend**: **{d['p_up_quarter']:.0%}** (middle result {d['median_quarter']:+.0%}). "
            f"For just the **next week** it's {u['p_up_week']:.0%} vs {d['p_up_week']:.0%}, "
            f"barely better than a coin flip either way. That's why this page shows the trend and how long "
            f"it tends to last, not a next-week call.")


def render_markdown(cards, market, hist, asof, synthetic):
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    L = [f"# Your stocks, in plain English: {asof} close", "",
         f"_Updated {now}. Not investment advice._", ""]
    if synthetic:
        L += ["> **SYNTHETIC TEST DATA.** Not real prices.", ""]
    L += ["## The market right now", "",
          f"- **Stock market overall (SPY):** {market['spy_trend'].lower()}, {market['spy_week']:+.1%} this week",
          f"- **Chip stocks (SMH):** {market['chips_week']:+.1%} this week",
          f"- **Mood (VIX fear gauge {market['vix']:.0f}):** {market['mood']}", "",
          "> " + history_sentence(hist), ""]
    for cd in [c for c in cards if c["holding"]]:
        L += [f"## ★ {cd['ticker']}: {cd['arrow']} {cd['label']}", ""]
        L += [f"- {x}" for x in _card_lines(cd)] + [""]
    L += ["## Everything else", "",
          "| Stock | Direction | For how long | Last 5 days | How wild | Earnings |", "|---|---|---|---|---|---|"]
    for cd in [c for c in cards if not c["holding"]]:
        age = cd.get("age_text", "–").split(" (")[0]
        earn = (f"in {human_days(cd['earn_days'])}" + (" ⚠️" if cd["earn_days"] <= 5 else "")
                if cd["earn_date"] is not None else "–")
        L.append(f"| **{cd['ticker']}** | {cd['arrow']} {cd['label']} | {age} | {cd['ret_5']:+.1%} | "
                 f"{cd['wild']} | {earn} |")
    L += ["", "<details><summary><b>What do these words mean?</b></summary>", "",
          "- **Uptrend / downtrend:** whether the stock's average price over the last ~2½ months is above or "
          "below its average over the last ~10 months. It changes slowly, every few months.",
          "- **Dipping / bouncing:** a short-term move against the bigger trend. Dips inside an uptrend are common "
          "and often don't end the uptrend.",
          "- **Normal week:** how much this stock usually moves in a week. A move bigger than that is "
          "unusual; a move 2× bigger is rare.",
          "- **How wild:** Calm (under ±4% a week), Normal (±4–7%), Wild (±7–11%), Very wild (more).",
          "- **Earnings report:** the company's quarterly results. The stock often jumps or drops a lot "
          "the next day.", "", "</details>", ""]
    return L


CSS = """
:root{--bg:#f6f6f4;--card:#fff;--ink:#111;--muted:#5b5a56;--line:#e6e5e1;--up:#0f7a4a;--up-bg:#e5f4ec;
--down:#b3261e;--down-bg:#fbe9e7;--mid:#8a6d00;--mid-bg:#fff4d6;--accent:#2a78d6}
@media (prefers-color-scheme: dark){:root:not([data-theme="light"]){--bg:#171716;--card:#222220;--ink:#f0efec;
--muted:#bdbcb3;--line:#36352f;--up:#6fd3a0;--up-bg:#173527;--down:#f2958c;--down-bg:#3c1d1a;--mid:#f0cf6a;
--mid-bg:#3a3014;--accent:#3987e5}}
:root[data-theme="dark"]{--bg:#171716;--card:#222220;--ink:#f0efec;--muted:#bdbcb3;--line:#36352f;--up:#6fd3a0;
--up-bg:#173527;--down:#f2958c;--down-bg:#3c1d1a;--mid:#f0cf6a;--mid-bg:#3a3014;--accent:#3987e5}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--ink);font:16px/1.5 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif}
main{max-width:640px;margin:0 auto;padding:16px 16px 48px}
h1{font-size:1.35rem;margin:8px 0 2px}h2{font-size:1.05rem;margin:28px 0 10px;color:var(--muted);font-weight:600}
.sub{color:var(--muted);font-size:.85rem}
.card{background:var(--card);border:1px solid var(--line);border-radius:14px;padding:14px 16px;margin:12px 0}
.top{display:flex;justify-content:space-between;align-items:baseline;gap:8px;flex-wrap:wrap}
.tk{font-size:1.3rem;font-weight:700}.px{font-size:1.1rem;font-weight:600}
.badge{display:inline-block;border-radius:999px;padding:4px 12px;font-weight:600;font-size:.95rem;margin:8px 0 4px}
.b-up{background:var(--up-bg);color:var(--up)}.b-down{background:var(--down-bg);color:var(--down)}
.b-none{background:var(--mid-bg);color:var(--mid)}
.row{margin:8px 0;font-size:.95rem}.row b{display:block;font-size:.78rem;color:var(--muted);text-transform:uppercase;letter-spacing:.03em}
.warn{color:var(--mid)}
.mini{display:flex;justify-content:space-between;align-items:center;gap:8px;padding:10px 0;border-bottom:1px solid var(--line)}
.mini:last-child{border-bottom:0}.mini .tk{font-size:1rem}.mini .r{text-align:right;font-size:.85rem;color:var(--muted);white-space:nowrap}
summary.mini{list-style:none}summary.mini::-webkit-details-marker{display:none}
.note{background:var(--card);border-left:4px solid var(--accent);border-radius:8px;padding:12px 14px;font-size:.92rem}
details{margin:16px 0}summary{cursor:pointer;font-weight:600}
a{color:var(--accent)}
"""


def _badge(cd):
    cls = "b-up" if cd.get("trend") == 1 else "b-down" if cd.get("trend") == -1 else "b-none"
    return f"<span class='badge {cls}'>{cd['arrow']} {html.escape(cd['label'])}</span>"


def render_html(cards, market, hist, asof, synthetic, model_line):
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    e = html.escape

    def full_card(cd):
        rows = []
        if cd.get("trend"):
            rows.append(("Which way (big picture)",
                         f"{'Up' if cd['trend'] > 0 else 'Down'}trending for {e(cd['age_text'])}."))
            rows.append(("How long it might last", e(cd["how_long"])))
        else:
            rows.append(("Which way", e(cd["trend_text"])))
        rows.append(("Last 5 days", f"{cd['ret_5']:+.1%}: {e(cd['move_word'])} "
                                    f"(a normal week is about ±{cd['typical_week']:.0%})"))
        rows.append(("Next week", f"Nobody can reliably call it. Likely between <b style='display:inline;font-size:inherit;"
                                  f"color:inherit;text-transform:none'>{money(cd['low_week'])}</b> and "
                                  f"<b style='display:inline;font-size:inherit;color:inherit;text-transform:none'>"
                                  f"{money(cd['high_week'])}</b> (4 times out of 5)."))
        rows.append(("How wild", f"{cd['wild']}: a 10% swing can happen in {human_days(cd['days_for_10pct'])}."))
        if cd["earn_date"] is not None:
            w = " <span class='warn'>⚠️ expect a big move</span>" if cd["earn_days"] <= 5 else ""
            rows.append(("Next earnings", f"In {human_days(cd['earn_days'])} (around {cd['earn_date']}){w}"))
        extra = "<div class='row sub'>Newer stock: less history, so treat this as rougher.</div>" if cd["short_history"] else ""
        return (f"<div class='card'><div class='top'><span class='tk'>{'★ ' if cd['holding'] else ''}{e(cd['ticker'])}</span>"
                f"<span class='px'>{money(cd['price'])}</span></div>{_badge(cd)}"
                + "".join(f"<div class='row'><b>{k}</b>{v}</div>" for k, v in rows) + extra + "</div>")

    def mini(cd):
        age = e(cd.get("age_short", ""))
        earn = "<br>⚠️ earnings" if cd["earn_date"] is not None and cd["earn_days"] <= 5 else ""
        return (f"<details class='mini-wrap'><summary class='mini'><span><span class='tk'>{e(cd['ticker'])}</span> "
                f"{_badge(cd)}</span><span class='r'>{money(cd['price'])}<br>{cd['ret_5']:+.1%} wk"
                f"{' · trend ' + age if age else ''}{earn}</span></summary>{full_card(cd)}</details>")

    holds = [c for c in cards if c["holding"]]
    rest = [c for c in cards if not c["holding"]]
    ups = sum(1 for c in rest + holds if c.get("trend") == 1)
    banner = ("<div class='note'><b>SYNTHETIC TEST DATA.</b> Not real prices.</div>" if synthetic else "")
    return f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<meta name="apple-mobile-web-app-capable" content="yes">
<title>My Stocks</title><style>{CSS}</style></head><body><main>
<h1>Your stocks, in plain English</h1>
<div class="sub">As of the {asof} close · updated {now} · not investment advice</div>
{banner}
<h2>THE MARKET RIGHT NOW</h2>
<div class="card">
<div class="row"><b>Stock market (SPY)</b>{market['spy_trend']}, {market['spy_week']:+.1%} this week</div>
<div class="row"><b>Chip stocks (SMH)</b>{market['chips_week']:+.1%} this week</div>
<div class="row"><b>Mood (VIX fear gauge {market['vix']:.0f})</b>{market['mood']}</div>
<div class="row"><b>Your list</b>{ups} of {len(cards)} stocks are in an uptrend</div>
</div>
<h2>YOUR HOLDINGS</h2>
{''.join(full_card(c) for c in holds)}
<h2>EVERYTHING ELSE (TAP TO OPEN)</h2>
<div class="card">{''.join(mini(c) for c in rest)}</div>
<h2>WHAT THE HISTORY SAYS</h2>
<div class="note">{e(history_sentence(hist)).replace('**', '')}</div>
<details><summary>What do these words mean?</summary><div class="card">
<div class="row"><b>Uptrend / downtrend</b>Whether the stock's average price over the last ~2½ months is above or below
its average over the last ~10 months. It changes slowly, every few months.</div>
<div class="row"><b>Dipping / bouncing</b>A short-term move against the bigger trend. Dips inside an uptrend are common
and often don't end it.</div>
<div class="row"><b>Normal week</b>How much this stock usually moves in a week. Bigger than that is unusual; twice that is rare.</div>
<div class="row"><b>How wild</b>Calm (under ±4% a week), Normal (±4–7%), Wild (±7–11%), Very wild (more).</div>
<div class="row"><b>Earnings</b>The company's quarterly results. The stock often jumps or drops a lot the next day.</div>
</div></details>
<p class="sub">{e(model_line)} · <a href="report.html">Model test results (advanced)</a></p>
</main></body></html>"""
