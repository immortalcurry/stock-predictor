# Weekly Move Model: honest 1-week predictions for memory and tech stocks

Predicts the **next 5 trading days** for ~31 memory, semiconductor, AI-infrastructure and
big-tech stocks:

- **Direction:** P(up), the probability the stock is higher in 5 trading days
- **Size:** a range of 10th / 50th / 90th percentiles, e.g. "-5% to +7%, middle +1%"

It runs entirely on GitHub Actions, so you don't need a laptop. You read the results on your phone:

| What | Where |
|---|---|
| Today's predictions (holdings on top) | [`LATEST.md`](LATEST.md) |
| Full test report with charts | your GitHub Pages link (`https://<you>.github.io/<repo>/`) |
| Short report for the GitHub app | [`REPORT.md`](REPORT.md) |
| Every prediction ever made + what happened | [`paper_log.csv`](paper_log.csv) |
| Tickers and settings | [`config.toml`](config.toml) |

> **Expect "no edge" or "tiny edge".** Weekly stock moves are mostly noise. The point of
> this project is to *measure* that honestly. The report states the verdict up front, and
> every prediction comes with a plain-English confidence note. Not investment advice.

---

## 1. One-time GitHub setup (from your phone)

You'll do all of this in a phone **browser** (Safari/Chrome) at github.com. The GitHub
app can't change settings. Tip: in the browser menu pick "Request desktop site" if a
menu is hidden.

### Step 1: Get the code into a repo
If Claude already pushed it for you, skip this step. Otherwise create a new repository
(**+** → *New repository*), then upload these files (*Add file → Upload files*).

> **Public or private?** GitHub Pages is free only for **public** repos (private repos
> need GitHub Pro). If you want to keep the repo private, skip Step 3 and read
> `REPORT.md` in the app instead. It has the key numbers and the backtest chart.

### Step 2: Let Actions run and commit
1. Repo → **Settings** → **Actions** → **General**.
2. Under *Actions permissions*, select **Allow all actions and reusable workflows**.
3. Scroll to *Workflow permissions*, select **Read and write permissions**, and tap **Save**.
   (The workflows commit `LATEST.md`, `paper_log.csv` and the model back to the repo.)

### Step 3: Turn on GitHub Pages
1. Repo → **Settings** → **Pages**.
2. Under *Build and deployment → Source*, choose **GitHub Actions**. That's all; no branch to pick.

### Step 4: Run it the first time (don't wait for Sunday)
1. Repo → **Actions** tab. If you see "Workflows aren't being run on this repository",
   tap **I understand my workflows, go ahead and enable them**.
2. Tap **Weekly retrain** → **Run workflow** → **Run workflow**. It takes ~10-20 minutes.
   This downloads 15 years of data, runs all the tests, and publishes the report.
3. When it's green, tap **Daily predictions** → **Run workflow** to get your first `LATEST.md`.
4. Your report link is shown under **Settings → Pages** ("Your site is live at ...").
   Bookmark it.

After that, everything is automatic:

| Schedule | Workflow | What it does |
|---|---|---|
| Sundays ~10 AM ET | `weekly-retrain.yml` | Retrains, re-tests, rebuilds + publishes the report |
| Weekdays 21:30 UTC (4:30 PM ET in winter, 5:30 PM in summer) | `daily-predict.yml` | Predicts, writes `LATEST.md`, appends + scores `paper_log.csv` |

GitHub can start scheduled jobs 5-30 minutes late. That's normal. GitHub also pauses
schedules in repos with no activity for 60 days, but the daily commits count as activity.

**If a run fails** (red ❌): tap it → tap the failed step to read the log. The most common cause
is Yahoo Finance rate-limiting GitHub's servers. Just re-run it later (*Re-run jobs*).

---

## 2. How it works (and why it's built this way)

```
config.toml ──► download (yfinance + FRED, cached) ──► features ──► walk-forward test ──► report.html
                                                          │                                   │
                                                          └──► final model ──► predict.py ──► LATEST.md
                                                                                    │
                                                                    paper_log.csv ◄─┘ ◄── score_log.py
```

### Data
- **Adjusted prices** (`auto_adjust=True`): splits and dividends are folded into history. Without this,
  NVDA's 10:1 split would look like a 90% crash.
- **NBIS starts 2024-10-21**. Before that, the ticker's history is Yandex, a different business.
  CRWV and SNDK also get explicit start dates (`[data.start_overrides]` in config.toml).
- **10-year yield from FRED's public CSV** (no API key), shifted by one day because FRED publishes
  each day's value the *next* day. Using it same-day would be a tiny leak.
- **Cache:** downloads are saved to `data/cache/` and reused for 12 hours.

### Features (all computed from data available at that day's close)
| Group | Features |
|---|---|
| Momentum | returns over 1, 5, 20, 60 days; 5- and 20-day returns **divided by the stock's own volatility** |
| Technicals | 20-day volatility, RSI(14), MACD and MACD histogram (as % of price), distance from 50/200-day average |
| Volume | log(volume / 20-day average volume) |
| Relative | return minus SPY's and minus SMH's, over 5 and 20 days |
| Peer ranks | where the stock's 5/20/60-day return, volatility, and volume rank among all stocks *that day* (0-1) |
| Market | VIX and its 5-day change, SPY above its 200-day average?, SPY/QQQ/SMH/SOXX returns, 10y-yield change (5 and 20 days) |
| Earnings | trading days until the next earnings reaction; flag if it falls in the next 5 days |

**Why volatility-scaling and ranks?** A +10% week is enormous for AAPL and ordinary for IREN.
Dividing returns by each stock's own volatility (a z-score), and ranking stocks against each
other, puts every stock on the same scale. That's what lets one pooled model learn from all of them.

**Why no "which ticker is this" feature?** It would let the model memorize "NVDA went up a
lot" (hindsight), and it can't help CRWV or SNDK, which have little history. Without it, the
model has to learn patterns that apply to any stock. That's also how the short-history
stocks still get predictions.

### Targets
- Direction: is the 5-day forward return > 0?
- Size: the 5-day forward return, **predicted in volatility units** and converted back to %.
  So the range automatically widens for wild stocks and around earnings.

### Models (simplest first)
1. **Baselines:** "always up", "last week repeats", "each stock's historical up-rate".
   A model is only interesting if it beats these.
2. **Logistic regression**, heavily regularized (`C=0.05`): a straight-line model that only trusts
   strong evidence. Outliers are clipped first.
3. **LightGBM** (gradient-boosted trees), kept deliberately small: depth 3, 8 leaves,
   at least 500 examples per leaf, slow learning rate. Big flexible models memorize noise.
4. **LightGBM quantile regression** for the 10th/50th/90th percentiles, compared to a baseline
   of "the typical-week distribution × the stock's current volatility".

Whichever direction model has the better walk-forward AUC is used for live predictions.

### Testing: the most important part
- **Walk-forward:** train on everything before a quarter, test on that quarter, then roll forward
  one quarter and repeat (2016 → today, ~43 retrains). The model is never tested on data it trained on,
  and data is never shuffled across time.
- **5-day purge gap:** a training row from day *t* has a label using prices up to *t+5*, so the
  last 5 days before each test period are dropped from training.
- **Leak test:** `train.py` rebuilds every feature with the last year of data deleted and checks
  that nothing about the past changes. If a feature secretly peeked ahead, this fails and training stops.
- **Too-good alarm:** accuracy > 60% (or AUC > 0.65) is flagged as a probable bug, not celebrated.
- **Honest error bars:** 30 stocks on the same day aren't 30 independent experiments, so confidence
  intervals use a *block bootstrap* that resamples whole weeks.
- **Metrics:** accuracy, ROC AUC (does a higher P(up) go with more up-weeks?), Brier score, a calibration
  table ("when it says 55%, does it go up 55% of the time?"), and range **coverage** (target 80%).
- **Range calibration:** if the raw 10-90% ranges catch too few outcomes, they are widened using
  errors from *earlier* test periods only (a simple version of "conformal" calibration).
- **Backtest:** each week, hold an equal-weight basket of the stocks with P(up) > 55%; hold cash
  otherwise; pay 0.1% per buy and per sell. It's compared with buy-and-hold **SPY**, **SMH**, and,
  most importantly, **holding all the stocks**. A rule that beats SPY only because chip stocks
  boomed has no skill. It's the universe doing the work.

### Offline self-test
`python train.py --synthetic` runs the whole pipeline on fake random-walk prices, where the
future is unpredictable by construction. A correct pipeline must report about 50% accuracy, AUC of about 0.50,
and "No reliable edge". It does. That's evidence the testing code isn't fooling itself.

---

## 3. Reading the outputs

**LATEST.md:** one row per stock, ★ = your holdings:

| | Ticker | P(up) | Range (10%–90%) | Middle | Earnings in week | Note |
|---|---|---|---|---|---|---|
| ★ | NBIS | 53% | -11.8% to +13.5% | +0.4% | no | coin flip (model has no proven edge) |

- **P(up)** between 47% and 53% means coin flip. Weekly models rarely go beyond 45-58%.
- The **range** should contain the real move 4 times out of 5. That's the most useful part:
  it tells you how big a normal week is for that stock right now.
- **"limited history"** = under ~2 years of data (CRWV, SNDK, NBIS for now). Their predictions
  come entirely from patterns learned on the other stocks.
- If most stocks show the same lean, it's really one market call, not 31 separate ones.

**paper_log.csv / the score in LATEST.md:** every prediction is written down *before* the week happens,
then `score_log.py` fills in the result 5 trading days later. This is the only test that
can't be accidentally overfit. **Watch it for 4-8 weeks** and compare its accuracy with
"always guess up" before trusting it with money.

---

## 4. Running on a laptop (optional)

```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python train.py          # ~5-15 min the first time (downloads + ~43 retrains); add --fast for fewer
python predict.py        # after 4 PM ET
python score_log.py
```

## 5. Files

| File | Purpose |
|---|---|
| `config.toml` | tickers, groups, dates, thresholds: **edit this** |
| `train.py` | data → leak check → features → walk-forward → backtest → final model → report |
| `predict.py` | today's predictions → console, `LATEST.md`, `paper_log.csv` |
| `score_log.py` | fills in actual outcomes and writes the live scorecard |
| `stockpred/data.py` | yfinance / FRED downloads + cache |
| `stockpred/features.py` | all features, plus the leak check |
| `stockpred/models.py` | baselines, logistic regression, LightGBM |
| `stockpred/evaluation.py` | walk-forward, metrics, bootstrap, backtest, verdict |
| `stockpred/report.py` | `docs/report.html` and `REPORT.md` |
| `stockpred/synthetic.py` | fake data for the offline self-test |
| `.github/workflows/` | the Sunday retrain and the weekday predictions |

## 6. Known limitations
- **Survivorship bias:** the ticker list was picked today, knowing these are AI-era winners. That flatters
  any strategy that's usually long, and the "hold all stocks" benchmark.
- **Earnings dates:** past dates are actual report dates, and Yahoo's earnings data is sometimes
  missing (shown as "unknown").
- The backtest assumes you trade at the close the signal is computed on; in reality you'd get the next open.
- No taxes, and no extra slippage for very volatile names.
