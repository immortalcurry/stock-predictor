"""
Models and baselines.

Direction (will the 5-day return be > 0?):
  * baseline_always_up : always guesses "up". Its probability is the training up-rate.
  * baseline_last_week : guesses that last week's direction repeats.
  * baseline_hist_avg  : each stock's historical up-rate (shrunk toward the
                         overall rate when a stock has little history).
  * logit              : regularized logistic regression (a linear model)
  * lgbm               : small, heavily regularized LightGBM tree model

Size of move (10th/50th/90th percentile of the 5-day return):
  * baseline: "typical week" percentiles from history, scaled by the stock's
    current volatility
  * LightGBM quantile regression, one model per percentile

Both range models predict in VOLATILITY UNITS (z = return / typical weekly move)
and convert back to % at the end. Without this, NBIS-style high-volatility
names would dominate the loss and the AAPL ranges would be nonsense.
"""
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier, LGBMRegressor
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .features import FEATURES

QUANTILES = (0.1, 0.5, 0.9)

# Deliberately small, slow-learning trees with large leaves. Financial data has
# a tiny signal-to-noise ratio. A flexible model memorizes noise and then looks
# great in-sample and terrible out-of-sample.
LGBM_PARAMS = dict(
    n_estimators=250,
    learning_rate=0.02,
    num_leaves=8,
    max_depth=3,
    min_child_samples=500,   # every leaf must hold 500+ examples
    subsample=0.7,
    subsample_freq=1,
    colsample_bytree=0.7,
    reg_lambda=10.0,
    random_state=42,
    n_jobs=-1,
    verbose=-1,
)


class Clipper(BaseEstimator, TransformerMixin):
    """Clip each feature to its 1st-99th percentile (from training data) so one
    crazy day can't dominate the linear model."""

    def fit(self, X, y=None):
        self.lo_ = np.nanpercentile(X, 1, axis=0)
        self.hi_ = np.nanpercentile(X, 99, axis=0)
        return self

    def transform(self, X):
        return np.clip(X, self.lo_, self.hi_)


def make_logit():
    return Pipeline([
        ("impute", SimpleImputer(strategy="median")),   # e.g. no 200-day MA yet for new stocks
        ("clip", Clipper()),
        ("scale", StandardScaler()),
        # C is the inverse of regularization strength. A small C pulls coefficients
        # toward zero, i.e. "only believe a feature if the evidence is strong".
        ("lr", LogisticRegression(C=0.05, max_iter=2000)),
    ])


def fit_models(train):
    """Fit all models + baseline statistics on a training DataFrame."""
    X, y, z = train[FEATURES], train["y_up"].astype(int), train["z_target"].clip(-6, 6)
    m = {}
    m["logit"] = make_logit().fit(X, y)
    m["lgbm"] = LGBMClassifier(**LGBM_PARAMS).fit(X, y)
    m["quantile"] = {
        a: LGBMRegressor(objective="quantile", alpha=a, **LGBM_PARAMS).fit(X, z)
        for a in QUANTILES
    }
    # --- baseline statistics ---
    base_rate = float(y.mean())
    by_t = train.groupby("ticker")["y_up"].agg(["sum", "count"])
    k = 100  # shrinkage: a stock needs ~100 weeks of data before its own rate dominates
    m["hist_rate"] = ((by_t["sum"] + k * base_rate) / (by_t["count"] + k)).to_dict()
    m["base_rate"] = base_rate
    m["z_quantiles"] = {a: float(np.nanquantile(z, a)) for a in QUANTILES}
    return m


def predict_all(m, df):
    """Return a DataFrame of probabilities and quantiles (in z units) for df rows."""
    X = df[FEATURES]
    out = pd.DataFrame(index=df.index)
    out["p_always_up"] = m["base_rate"]
    out["p_last_week"] = (df["ret_5"] > 0).astype(float)
    out["p_hist_avg"] = df["ticker"].map(m["hist_rate"]).fillna(m["base_rate"])
    out["p_logit"] = m["logit"].predict_proba(X)[:, 1]
    out["p_lgbm"] = m["lgbm"].predict_proba(X)[:, 1]
    q = np.column_stack([m["quantile"][a].predict(X) for a in QUANTILES])
    q.sort(axis=1)   # quantile models are fit separately and can "cross"; sorting fixes that
    out["q10_z"], out["q50_z"], out["q90_z"] = q[:, 0], q[:, 1], q[:, 2]
    for a, name in zip(QUANTILES, ["bq10_z", "bq50_z", "bq90_z"]):
        out[name] = m["z_quantiles"][a]
    return out
