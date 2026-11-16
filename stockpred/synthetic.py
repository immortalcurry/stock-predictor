"""
Fake market data for testing the pipeline offline (`python train.py --synthetic`).

The prices are pure random walks: a shared market factor plus stock-specific
noise, with volatility that drifts over time. By construction the future is
UNPREDICTABLE, so a correct pipeline must report about 50% accuracy, AUC of about
0.50, and "no edge". If it reports skill on this data, there's a leak.
"""
import numpy as np
import pandas as pd


def make_synthetic(cfg, seed=7):
    rng = np.random.default_rng(seed)
    n = int(252 * cfg["data"]["years"])
    idx = pd.bdate_range(end=pd.Timestamp.today().normalize(), periods=n)

    # Market factor with slowly varying volatility (volatility clustering)
    mvol = 0.01 * np.exp(np.cumsum(rng.normal(0, 0.03, n)).clip(-1, 1))
    mkt = rng.normal(0.0003, 1, n) * mvol

    def series(beta, idio, drift=0.0003):
        vol = idio * np.exp(np.cumsum(rng.normal(0, 0.03, n)).clip(-1, 1))
        r = drift + beta * mkt + rng.normal(0, 1, n) * vol
        close = 50 * np.exp(np.cumsum(r))
        volu = rng.lognormal(15, 0.4, n)
        return pd.DataFrame({"Open": close, "High": close, "Low": close,
                             "Close": close, "Volume": volu}, index=idx)

    prices = {}
    for t in ["SPY", "QQQ", "SMH", "SOXX"]:
        prices[t] = series(1.0 if t == "SPY" else 1.3, 0.003)
    vix = 15 + 600 * (mvol - mvol.mean()) + rng.normal(0, 0.5, n)
    prices["^VIX"] = pd.DataFrame({"Open": vix, "High": vix, "Low": vix, "Close": vix,
                                   "Volume": 0.0}, index=idx)
    earnings = {}
    for t in cfg["pool"]:
        prices[t] = series(rng.uniform(0.8, 1.8), rng.uniform(0.01, 0.035))
        q = pd.date_range(idx[0], idx[-1] + pd.Timedelta(days=90), freq="91D") + pd.Timedelta(days=int(rng.integers(0, 60)))
        earnings[t] = pd.DataFrame({"date": q, "after_close": True,
                                    "surprise": rng.normal(0, 10, len(q))})
    for t, d in cfg["data"]["start_overrides"].items():
        if t in prices:
            prices[t] = prices[t][prices[t].index >= pd.Timestamp(d)]
    fred = pd.Series(3 + np.cumsum(rng.normal(0, 0.03, n)), index=idx, name="DGS10")
    return prices, earnings, fred
