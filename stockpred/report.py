"""
Builds docs/report.html (published with GitHub Pages) and REPORT.md (readable
in the GitHub phone app). Charts are made with matplotlib and embedded in the
HTML as images, so the page is a single self-contained file.
"""
import base64
import html
import io
from datetime import datetime, timezone

import matplotlib

matplotlib.use("Agg")   # no screen needed (GitHub Actions has none)
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

# Fixed colors: one per series, the same in every chart
C_MODEL, C_SPY, C_SMH, C_EW = "#2a78d6", "#eb6834", "#1baf7a", "#b07a00"
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"


def _style(ax):
    for s in ["top", "right"]:
        ax.spines[s].set_visible(False)
    for s in ["left", "bottom"]:
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=9)
    ax.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def _png(fig, path=None):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=130, bbox_inches="tight", facecolor="white")
    if path:
        fig.savefig(path, dpi=130, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode()


def chart_backtest(bt, path):
    fig, ax = plt.subplots(figsize=(8, 4.2))
    series = [("strategy", "Model rule", C_MODEL, 2.2), ("equal_weight_universe", "Hold all stocks", C_EW, 1.6),
              ("SMH", "SMH", C_SMH, 1.6), ("SPY", "SPY", C_SPY, 1.6)]
    for col, label, color, lw in series:
        eq = (1 + bt[col].fillna(0)).cumprod()
        # The final multiple goes in the legend, e.g. "SPY (2.1x)", so no labels get clipped
        ax.plot(eq.index, eq.values, color=color, linewidth=lw, label=f"{label} ({eq.iloc[-1]:.1f}x)")
    ax.set_yscale("log")
    ax.set_ylabel("Growth of $1 (log scale)", color=MUTED, fontsize=9)
    ax.legend(frameon=False, fontsize=9, loc="upper left")
    _style(ax)
    return _png(fig, path)


def chart_calibration(cal):
    fig, ax = plt.subplots(figsize=(4.6, 4.2))
    lo = min(cal["predicted"].min(), cal["actual"].min()) - 0.03
    hi = max(cal["predicted"].max(), cal["actual"].max()) + 0.03
    ax.plot([lo, hi], [lo, hi], color=MUTED, linestyle="--", linewidth=1, label="Perfect calibration")
    ax.scatter(cal["predicted"], cal["actual"], s=np.clip(cal["n"] / cal["n"].max() * 300, 30, 300),
               color=C_MODEL, edgecolor="white", linewidth=1.5, zorder=3, label="Model (dot size = # predictions)")
    ax.set_xlabel("Model said P(up)", color=MUTED, fontsize=9)
    ax.set_ylabel("Actually went up", color=MUTED, fontsize=9)
    ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1, 0))
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1, 0))
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    _style(ax)
    return _png(fig)


def chart_importance(imp):
    top = imp.head(15)[::-1]
    fig, ax = plt.subplots(figsize=(6.5, 4.6))
    ax.barh(top.index, top.values, color=C_MODEL, height=0.6)
    ax.set_xlabel("Share of LightGBM split gain", color=MUTED, fontsize=9)
    ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1, 0))
    _style(ax)
    ax.grid(axis="y", visible=False)
    return _png(fig)


def _pct(x, d=1):
    return "" if pd.isna(x) else f"{x:.{d}%}"


def _tbl(df, fmt):
    """DataFrame -> HTML with per-column formatters."""
    df = df.copy()
    for c, f in fmt.items():
        if c in df:
            df[c] = df[c].map(f)
    return df.to_html(border=0, classes="t", escape=True)


CSS = """
:root{--bg:#fcfcfb;--card:#ffffff;--ink:#0b0b0b;--muted:#52514e;--line:#e4e3df;--accent:#2a78d6;
--warn-bg:#fff4e0;--warn-ink:#6b4500;--ok-bg:#e7f5ee;--ok-ink:#0d5a3a}
@media (prefers-color-scheme: dark){:root:not([data-theme="light"]){--bg:#1a1a19;--card:#232321;--ink:#f0efec;
--muted:#c3c2b7;--line:#3a3936;--accent:#3987e5;--warn-bg:#3b2d12;--warn-ink:#f3d79c;--ok-bg:#163327;--ok-ink:#a8e2c6}}
:root[data-theme="dark"]{--bg:#1a1a19;--card:#232321;--ink:#f0efec;--muted:#c3c2b7;--line:#3a3936;--accent:#3987e5;
--warn-bg:#3b2d12;--warn-ink:#f3d79c;--ok-bg:#163327;--ok-ink:#a8e2c6}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--ink);font:16px/1.55 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif}
main{max-width:880px;margin:0 auto;padding:24px 16px 64px}
h1{font-size:1.6rem;margin:0 0 4px}h2{font-size:1.2rem;margin:36px 0 8px;border-top:1px solid var(--line);padding-top:20px}
.sub{color:var(--muted);font-size:.9rem}
.verdict{border-radius:10px;padding:14px 16px;margin:18px 0;background:var(--warn-bg);color:var(--warn-ink)}
.verdict.good{background:var(--ok-bg);color:var(--ok-ink)}
.verdict b{font-size:1.05rem}
.scroll{overflow-x:auto;-webkit-overflow-scrolling:touch}
table.t{border-collapse:collapse;font-size:.88rem;margin:8px 0;min-width:100%}
table.t th,table.t td{padding:6px 10px;border-bottom:1px solid var(--line);text-align:right;white-space:nowrap}
table.t th:first-child,table.t td:first-child{text-align:left}
table.t thead th{color:var(--muted);font-weight:600}
img{max-width:100%;height:auto;border-radius:8px;background:#fff;display:block;margin:10px 0}
.note{color:var(--muted);font-size:.9rem}
li{margin:4px 0}
"""


def build_report(ctx, html_path, md_path, png_path):
    """ctx: dict of results assembled by train.py."""
    v, dir_tab, cal = ctx["verdict"], ctx["direction"], ctx["calibration"]
    rng, bt, bt_sum, sweep = ctx["range"], ctx["backtest"], ctx["backtest_summary"], ctx["sweep"]
    yearly, imp, coefs = ctx["yearly"], ctx["importance"], ctx["coefs"]
    primary_name = ctx["primary_name"]
    synthetic = ctx.get("synthetic", False)
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    img_bt = chart_backtest(bt, png_path)
    img_cal = chart_calibration(cal)
    img_imp = chart_importance(imp)
    good = v["level"] in ("tiny_edge", "modest_edge")

    # ---- experiments table ----
    exps = ctx.get("experiments", {})
    erows = []
    for e in exps.values():
        d, x = e["dev"], e["exam"] or {}
        erows.append({
            "question": e["desc"],
            "model": "LightGBM" if e["best"] == "p_lgbm" else "Logistic",
            "dev accuracy": f"{d['accuracy']:.1%} (naive {d['naive']:.1%})",
            "dev AUC": f"{d['auc']:.3f} ({d['auc_lo']:.3f}-{d['auc_hi']:.3f})",
            "dev ranking IC": f"{d['ic']:+.3f} (t={d['ic_t']:.1f})",
            "dev top-5 edge/yr": _pct(d["top_excess"]),
            "exam AUC": f"{x['auc']:.3f}" if x else "n/a",
            "exam ranking IC": f"{x['ic']:+.3f}" if x else "n/a",
            "exam top-5 edge/yr": _pct(x["top_excess"]) if x else "n/a",
            "result": e["label"],
        })
    etab = pd.DataFrame(erows).set_index("question") if erows else pd.DataFrame()
    r1 = ctx["range_1d"].set_index("model") if "range_1d" in ctx else pd.DataFrame()
    exam_start = ctx.get("exam_start", "")
    exp_html = f"""<h2>Experiments: can we ask an easier question?</h2>
<p>"Will it go up?" is mostly a bet on the whole market. These experiments try easier questions:
<b>will the stock beat the average of its peers</b> (which removes the market's move), and a
<b>one-month</b> horizon. Each idea is developed on data before {exam_start} ("dev") and then graded once
on data from {exam_start} onward (the "final exam"). An idea counts only if it passes both.</p>
<ul><li><b>AUC</b>: 0.50 = no skill (90% range in brackets).</li>
<li><b>Ranking IC</b>: each day, do the stocks the model scores higher actually do better? 0 = no skill,
+0.03 or more is decent; <i>t</i> above 2 means it's unlikely to be luck.</li>
<li><b>Top-5 edge/yr</b>: yearly return of holding the model's 5 favourite stocks minus holding all of them,
after trading costs.</li></ul>
<div class="scroll">{etab.to_html(border=0, classes="t", escape=True)}</div>""" if erows else ""
    r1_html = f"""<h3>Tomorrow's range</h3>
<p class="note">The same check for the 1-day price range shown in LATEST.md.</p>
<div class="scroll">{_tbl(r1, {'coverage': _pct, 'median_width': _pct})}</div>""" if len(r1) else ""

    dt = dir_tab.drop(columns="col").set_index("model")
    rt = rng.set_index("model")
    banner = ("<div class='verdict'><b>SYNTHETIC TEST DATA.</b> These results come from fake random-walk "
              "prices used to test the code. They say nothing about real stocks.</div>") if synthetic else ""

    h = f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Weekly Move Model</title><style>{CSS}</style></head><body><main>
<h1>Weekly stock-move model: test report</h1>
<div class="sub">Generated {now} · data through {ctx['data_end']} · {ctx['n_rows']:,} stock-days from
{ctx['n_tickers']} reported stocks (trained on {ctx.get('n_pool', ctx['n_tickers'])}) · walk-forward test {ctx['test_start']} → {ctx['data_end']}</div>
{banner}
<div class="verdict {'good' if good else ''}"><b>Verdict: {html.escape(v['headline'])}</b>
<ul>{''.join(f'<li>{html.escape(d)}</li>' for d in v['details'])}</ul></div>

<h2>1. Can it call the direction?</h2>
<p>Every number here is <b>out-of-sample</b>: each prediction came from a model trained only on
earlier data, with a {ctx['gap_days']}-day gap so the test week can't leak into training. "Accuracy" counts
P(up) &gt; 50% as an "up" call. <b>AUC</b> measures whether higher probabilities go with more up-weeks
(0.50 = no skill, 1.0 = perfect). <b>Brier</b> is the probability error (lower is better; 0.25 = always saying 50%).</p>
<div class="scroll">{_tbl(dt, {'accuracy': _pct, 'auc': lambda x: f'{x:.3f}', 'brier': lambda x: f'{x:.4f}'})}</div>
<p class="note">Why "always up" is hard to beat: stocks rise in a bit more than half of all weeks, and this
universe of semis/AI names rose a lot over the test period. A model has to beat <i>that</i>, not 50%.
The primary model used for live predictions is <b>{html.escape(primary_name)}</b> (higher test AUC).</p>

<h3>Is it stable year to year?</h3>
<div class="scroll">{_tbl(yearly, {'model_accuracy': _pct, 'always_up_accuracy': _pct, 'auc': lambda x: f'{x:.3f}', 'rows': lambda x: f'{x:,}'})}</div>

<h2>2. Are the probabilities honest? (calibration)</h2>
<p>If the model says 55%, the stock should go up about 55% of the time. Dots near the dashed line = honest
probabilities. Note how narrow the range of probabilities is. A sensible model on weekly data rarely
says anything far from 50%.</p>
<img alt="Calibration chart" src="data:image/png;base64,{img_cal}">
<div class="scroll">{_tbl(cal, {'predicted': _pct, 'actual': _pct, 'n': lambda x: f'{x:,}'})}</div>

<h2>3. Are the ranges right? (10th-90th percentile)</h2>
<p>The predicted range should contain the actual 5-day return about <b>80%</b> of the time. "Calibrated"
widens the raw LightGBM range using only errors from <i>earlier</i> test periods (no peeking). Pinball loss
scores the percentile forecasts (lower is better). If LightGBM doesn't beat the simple volatility baseline,
the range really comes from "how volatile has this stock been lately".</p>
<div class="scroll">{_tbl(rt, {'coverage': _pct, 'median_width': _pct, 'pinball': lambda x: '' if pd.isna(x) else f'{x:.5f}'})}</div>
<p class="note">Coverage by group (calibrated): {', '.join(f'{k} {val:.0%}' for k, val in ctx['coverage_by_group'].items())}</p>
{r1_html}

<h2>4. Backtest: would trading it have made money?</h2>
<p>Rule: every week, hold an equal-weight basket of stocks with P(up) &gt; {ctx['threshold']:.0%}; hold cash
otherwise. Costs: {ctx['cost']:.1%} per buy and per sell. Out-of-sample predictions only. The fair comparison
is <b>"hold all stocks"</b>: if the rule only beats SPY, that's just because chip stocks boomed, not skill.</p>
<img alt="Backtest equity curves" src="data:image/png;base64,{img_bt}">
<div class="scroll">{_tbl(bt_sum, {c: _pct for c in ['CAGR', 'Volatility', 'Max drawdown', 'Total return', '% weeks invested']} | {'Sharpe': lambda x: f'{x:.2f}'})}</div>
<h3>Does the threshold matter?</h3>
<p class="note">If results swing wildly between nearby thresholds, any "good" result is probably luck.</p>
<div class="scroll">{_tbl(sweep.set_index('threshold'), {'CAGR': _pct, 'Sharpe': lambda x: f'{x:.2f}', 'Max drawdown': _pct, 'avg stocks held': lambda x: f'{x:.1f}', '% weeks invested': _pct, 'avg weekly cost': lambda x: f'{x:.2%}'})}</div>

{exp_html}

<h2>5. What does the model look at? (feature importance)</h2>
<p>Share of the tree model's "split gain" by feature. This shows what the model <i>uses</i>, not that the
feature truly predicts anything. Market-wide features (VIX, SPY trend) tend to dominate, which means many
predictions are really a market call applied to every stock at once.</p>
<img alt="Feature importance" src="data:image/png;base64,{img_imp}">
<p class="note">Largest logistic-regression coefficients (standardized; positive = higher value → more likely up):
{', '.join(f'{k} {val:+.3f}' for k, val in coefs.items())}</p>

<h2>6. Checks for bugs and leaks</h2>
<ul>
<li><b>Future-data test:</b> {html.escape(ctx['leak_msg'])}</li>
<li><b>Purge gap:</b> {ctx['gap_days']} trading days between training and each test period.</li>
<li><b>Too-good-to-be-true alarm:</b> accuracy &gt; 60% or AUC &gt; 0.65 is flagged as a probable leak.
{'<b>TRIGGERED, investigate!</b>' if v['level'] == 'suspicious' else 'Not triggered.'}</li>
<li><b>Overlapping data:</b> confidence ranges use a week-block bootstrap, because 30+ stocks on the same day
are not independent experiments.</li>
</ul>

<h2>7. Caveats</h2>
<ul>
<li>The ticker list was chosen <b>today</b>, knowing these companies are AI winners (survivorship bias). This flatters
"hold all stocks" and anything that is usually long.</li>
<li>Past earnings dates are the actual report dates; companies usually announce them only weeks in advance.</li>
<li>Short-history names ({', '.join(ctx['short_history']) or 'none'}) get predictions from patterns learned on other stocks.</li>
<li>Backtest ignores taxes, slippage on very volatile names, and the fact that you'd trade at the next open, not the close.</li>
</ul>
<p class="note">Not investment advice. Use the paper log (paper_log.csv) for 4-8 weeks to see live accuracy before trusting it.</p>
</main></body></html>"""
    with open(html_path, "w") as f:
        f.write(h)

    # ---- REPORT.md (for the GitHub app) ----
    def md_table(df, fmt):
        df = df.copy()
        for c, fn in fmt.items():
            if c in df:
                df[c] = df[c].map(fn)
        cols = [df.index.name or ""] + list(df.columns)
        lines = ["| " + " | ".join(map(str, cols)) + " |", "|" + "---|" * len(cols)]
        for i, r in df.iterrows():
            lines.append("| " + " | ".join([str(i)] + [str(x) for x in r.values]) + " |")
        return "\n".join(lines)

    md = [f"# Model test report\n\n_Generated {now}. Data through {ctx['data_end']}._\n"]
    if synthetic:
        md.append("> **SYNTHETIC TEST DATA.** Fake prices, used only to check the code.\n")
    md.append(f"## Verdict\n\n**{v['headline']}**\n")
    md += [f"- {d}" for d in v["details"]]
    md.append("\n## Direction (out-of-sample)\n")
    md.append(md_table(dt, {'accuracy': _pct, 'auc': lambda x: f'{x:.3f}', 'brier': lambda x: f'{x:.4f}'}))
    md.append("\n## Ranges (target coverage 80%)\n")
    md.append(md_table(rt, {'coverage': _pct, 'median_width': _pct, 'pinball': lambda x: '' if pd.isna(x) else f'{x:.5f}'}))
    if len(r1):
        md.append("\n## Tomorrow's range (target coverage 80%)\n")
        md.append(md_table(r1, {'coverage': _pct, 'median_width': _pct}))
    if erows:
        md.append("\n## Experiments: easier questions\n")
        md.append(f"Developed on data before {exam_start}; graded once on the final exam ({exam_start} onward).\n")
        for r in erows:
            md.append(f"- **{r['question']}**: {r['result']}. Dev AUC {r['dev AUC']}, ranking IC {r['dev ranking IC']}, "
                      f"top-5 edge {r['dev top-5 edge/yr']}/yr. Final exam: AUC {r['exam AUC']}, IC {r['exam ranking IC']}, "
                      f"top-5 edge {r['exam top-5 edge/yr']}/yr.")
    md.append("\n## Backtest\n\n![backtest](docs/backtest.png)\n")
    md.append(md_table(bt_sum, {c: _pct for c in ['CAGR', 'Volatility', 'Max drawdown', 'Total return', '% weeks invested']} | {'Sharpe': lambda x: f'{x:.2f}'}))
    md.append("\nFull report with charts: see `docs/report.html` / your GitHub Pages link.\n")
    with open(md_path, "w") as f:
        f.write("\n".join(md))
