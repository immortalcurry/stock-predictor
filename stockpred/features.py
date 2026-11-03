"""
Feature engineering.

GOLDEN RULE: every feature for day t may use ONLY data known at day t's close.
In pandas that means only backward-looking operations (pct_change(n), rolling,
ewm, shift(+k)). The one forward-looking operation, shift(-5), is used ONLY
for the target. train.py runs `leak_check()`, which proves the rule holds:
it recomputes the features with the future cut off and checks they don't change.

The output is a "long" panel with one row per (date, ticker).
"""
import numpy as np
import pandas as pd

SQRT5 = np.sqrt(5)
VOL_FLOOR = 0.005  # minimum daily vol used for scaling (stops divide-by-tiny)

# Features the models see. Ticker identity is deliberately NOT a feature: we
# want rules that generalize across stocks, including new ones like CRWV.
STOCK_FEATURES = [
    "ret_1", "ret_5", "ret_20", "ret_60",
    "ret_5_z", "ret_20_z",              # returns divided by the stock's own volatility
    "vol_20", "rsi_14", "macd", "macd_hist",
    "dist_ma50", "dist_ma200", "rel_volume_20",
    "rel_spy_5", "rel_spy_20", "rel_smh_5", "rel_smh_20",
    "days_to_earnings", "earnings_in_week",
]
RANK_FEATURES = ["rank_ret_5", "rank_ret_20", "rank_ret_60", "rank_vol_20", "rank_rel_volume"]
MARKET_FEATURES = [
    "vix", "vix_chg_5", "spy_above_200", "spy_ret_5", "spy_ret_20",
    "qqq_ret_5", "smh_ret_20", "soxx_ret_5", "y10_chg_5", "y10_chg_20",
]
FEATURES = STOCK_FEATURES + RANK_FEATURES + MARKET_FEATURES


# ----------------------------------------------------------------------------
# Indicators
# ----------------------------------------------------------------------------
def rsi(close, n=14):
    """Wilder's RSI: 0-100. Above 70 is 'overbought', below 30 is 'oversold'."""
    delta = close.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / n, adjust=False, min_periods=n).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / n, adjust=False, min_periods=n).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - 100 / (1 + rs)


def trading_calendar(spy_index, extra_days=90):
    """Past trading days (from SPY) plus approximate future weekdays, used to count
    trading days until upcoming earnings. Future holidays are ignored, which is
    close enough."""
    future = pd.bdate_range(spy_index[-1] + pd.Timedelta(days=1), periods=extra_days)
    return spy_index.append(future)


def earnings_features(index, earn, calendar, cap=65):
    """
    days_to_earnings: trading days from day t until the first trading day whose
                      price REACTS to the next report (1 = tomorrow).
    earnings_in_week: 1 if that reaction falls inside the next 5 trading days,
                      i.e. inside the prediction window.

    Small caveat (also in the README): for past dates we use the actual report
    dates. Companies usually announce these a few weeks ahead, so the lookahead
    is small, but it isn't exactly zero for the "60 days ahead" part.
    """
    n = len(index)
    if earn is None or len(earn) == 0:
        return np.full(n, np.nan), np.zeros(n)
    dates = pd.to_datetime(earn["date"]).values
    after = earn["after_close"].astype(bool).values
    pos = calendar.searchsorted(dates)              # first trading day >= report date
    on_trading_day = (pos < len(calendar)) & (calendar[np.minimum(pos, len(calendar) - 1)] == dates)
    react = pos + (after & on_trading_day)          # after-close on a trading day: react next day
    react = np.unique(react[react < len(calendar)])

    t_pos = calendar.get_indexer(index)
    j = np.searchsorted(react, t_pos, side="right")  # first reaction strictly after t
    days = np.full(n, np.nan)
    ok = j < len(react)
    days[ok] = react[j[ok]] - t_pos[ok]
    days = np.minimum(days, cap)
    return days, ((days >= 1) & (days <= 5)).astype(float)


# ----------------------------------------------------------------------------
# Building blocks
# ----------------------------------------------------------------------------
def market_context(prices, fred, cal_index):
    """Market-wide features, one row per trading day."""
    def close(t):
        return prices[t]["Close"].reindex(cal_index) if t in prices else pd.Series(np.nan, cal_index)

    spy, qqq, smh, soxx, vix = (close(t) for t in ["SPY", "QQQ", "SMH", "SOXX", "^VIX"])
    m = pd.DataFrame(index=cal_index)
    m["vix"] = vix
    m["vix_chg_5"] = vix - vix.shift(5)
    m["spy_above_200"] = (spy > spy.rolling(200).mean()).astype(float)
    m["spy_ret_5"] = spy.pct_change(5)
    m["spy_ret_20"] = spy.pct_change(20)
    m["qqq_ret_5"] = qqq.pct_change(5)
    m["smh_ret_20"] = smh.pct_change(20)
    m["soxx_ret_5"] = soxx.pct_change(5)
    if fred is not None and len(fred):
        # FRED posts day t's yield the next day, so shift by one trading day to
        # use only what was actually published by day t's close.
        y = fred.reindex(cal_index.union(fred.index)).ffill().reindex(cal_index).shift(1)
        m["y10_chg_5"] = y - y.shift(5)
        m["y10_chg_20"] = y - y.shift(20)
    else:
        m["y10_chg_5"] = np.nan
        m["y10_chg_20"] = np.nan
    return m


def stock_frame(df, spy_close, smh_close, earn, calendar, horizon=5):
    """Per-stock features plus targets."""
    c = df["Close"].astype(float)
    v = df["Volume"].astype(float).replace(0, np.nan)
    r1 = c.pct_change()
    f = pd.DataFrame(index=c.index)

    for n in (1, 5, 20, 60):
        f[f"ret_{n}"] = c.pct_change(n)
    f["vol_20"] = r1.rolling(20).std()
    vol = f["vol_20"].clip(lower=VOL_FLOOR)
    # Volatility-scaled returns: +10% in a week is huge for AAPL and ordinary for
    # IREN. Dividing by each stock's own vol makes them comparable.
    f["ret_5_z"] = (f["ret_5"] / (vol * SQRT5)).clip(-6, 6)
    f["ret_20_z"] = (f["ret_20"] / (vol * np.sqrt(20))).clip(-6, 6)
    f["rsi_14"] = rsi(c)
    ema12, ema26 = c.ewm(span=12, adjust=False).mean(), c.ewm(span=26, adjust=False).mean()
    macd = (ema12 - ema26) / c          # divided by price so it's comparable across stocks
    f["macd"] = macd
    f["macd_hist"] = macd - macd.ewm(span=9, adjust=False).mean()
    f["dist_ma50"] = c / c.rolling(50).mean() - 1
    f["dist_ma200"] = c / c.rolling(200).mean() - 1
    f["rel_volume_20"] = np.log(v / v.rolling(20).mean())

    spy = spy_close.reindex(c.index)
    smh = smh_close.reindex(c.index)
    for n in (5, 20):
        f[f"rel_spy_{n}"] = f[f"ret_{n}"] - spy.pct_change(n)
        f[f"rel_smh_{n}"] = f[f"ret_{n}"] - smh.pct_change(n)

    f["days_to_earnings"], f["earnings_in_week"] = earnings_features(c.index, earn, calendar)

    # ---- Targets (the ONLY place we look forward) ----
    f["close"] = c
    f["fwd_ret"] = c.shift(-horizon) / c - 1
    f["scale"] = vol * SQRT5                      # this stock's "typical" 1-week move
    f["z_target"] = f["fwd_ret"] / f["scale"]     # forward move in units of typical move
    f["y_up"] = (f["fwd_ret"] > 0).astype(float).where(f["fwd_ret"].notna())
    f["n_days_history"] = np.arange(1, len(c) + 1)
    return f


def build_panel(prices, earnings, fred, targets, horizon=5, calendar=None):
    """Build the full (date, ticker) panel with features + targets."""
    spy_idx = prices["SPY"].index
    if calendar is None:
        calendar = trading_calendar(spy_idx)
    ctx = market_context(prices, fred, spy_idx)
    spy_c = prices["SPY"]["Close"]
    smh_c = prices["SMH"]["Close"] if "SMH" in prices else spy_c

    frames = []
    for t in targets:
        if t not in prices or len(prices[t]) < 25:
            continue
        df = prices[t][prices[t].index.isin(spy_idx)]   # align to market trading days
        f = stock_frame(df, spy_c, smh_c, earnings.get(t), calendar, horizon)
        f["ticker"] = t
        frames.append(f)
    panel = pd.concat(frames)
    panel.index.name = "date"
    panel = panel.reset_index()

    # Cross-sectional percentile ranks (0..1) within each day: "is this stock's
    # week strong compared to its peers today?" Ranks are robust to outliers.
    g = panel.groupby("date")
    panel["rank_ret_5"] = g["ret_5"].rank(pct=True)
    panel["rank_ret_20"] = g["ret_20"].rank(pct=True)
    panel["rank_ret_60"] = g["ret_60"].rank(pct=True)
    panel["rank_vol_20"] = g["vol_20"].rank(pct=True)
    panel["rank_rel_volume"] = g["rel_volume_20"].rank(pct=True)

    panel = panel.merge(ctx, left_on="date", right_index=True, how="left")
    # A row is usable once the stock has enough history for the basic features.
    panel = panel[panel["vol_20"].notna() & panel["ret_5"].notna()]
    panel["date_pos"] = panel["date"].map(pd.Series(np.arange(len(spy_idx)), index=spy_idx))
    return panel.sort_values(["date", "ticker"]).reset_index(drop=True)


# ----------------------------------------------------------------------------
# Leak check
# ----------------------------------------------------------------------------
def leak_check(prices, earnings, fred, targets, horizon=5, cut_back=250):
    """
    Rebuild the panel using only data up to an earlier cutoff date, and confirm
    that every feature on or before the cutoff is identical to the full-data
    version. If any feature changes, it was secretly using future data.
    Returns (passed: bool, message: str).
    """
    spy_idx = prices["SPY"].index
    cutoff = spy_idx[-cut_back]
    trunc = {t: df[df.index <= cutoff] for t, df in prices.items()}
    fred_t = fred[fred.index <= cutoff] if fred is not None and len(fred) else fred
    # The exchange holiday calendar is published in advance, so both builds may
    # share the same calendar for counting days until earnings.
    cal = trading_calendar(spy_idx)
    full = build_panel(prices, earnings, fred, targets, horizon, cal)
    part = build_panel(trunc, earnings, fred_t, targets, horizon, cal)
    a = full[full["date"] <= cutoff].set_index(["date", "ticker"])[FEATURES]
    b = part.set_index(["date", "ticker"])[FEATURES]
    a, b = a.align(b, join="inner")
    diff = (a - b).abs()
    both_nan = a.isna() & b.isna()
    bad = (diff > 1e-9) | (a.isna() != b.isna())
    bad = bad & ~both_nan
    if bad.values.any():
        cols = list(bad.columns[bad.any()])
        return False, f"Features changed when future data was removed: {cols}"
    # Targets near the cutoff SHOULD become unknown in the truncated data.
    return True, (f"Passed: {len(a):,} rows x {len(FEATURES)} features are identical "
                  f"with data after {cutoff.date()} removed.")
