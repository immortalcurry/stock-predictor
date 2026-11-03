"""
Data download + local caching.

Sources (all free, no API keys):
  * yfinance : daily prices/volume (split- and dividend-ADJUSTED) and earnings dates
  * FRED     : 10-year Treasury yield via the public CSV link (no key needed)

Caching: each ticker is saved as a CSV under data/cache/. If the file is newer
than `cache_max_age_hours` we reuse it, so reruns are fast. If a download fails
and an older cached file exists, we fall back to it with a warning instead of
crashing.
"""
import io
import time
import warnings
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import requests

FRED_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv?id={series}"


def _safe(ticker):
    """'^VIX' -> 'VIX' so it's a friendly file name."""
    return ticker.replace("^", "").replace("/", "_")


def _is_fresh(path, max_age_hours):
    return path.exists() and (time.time() - path.stat().st_mtime) < max_age_hours * 3600


def _read_price_csv(path):
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    df.index = pd.to_datetime(df.index).tz_localize(None)
    return df


# ----------------------------------------------------------------------------
# Prices
# ----------------------------------------------------------------------------
def download_prices(tickers, years, cache_dir, max_age_hours, force=False):
    """Return {ticker: DataFrame[Open, High, Low, Close, Volume]} of ADJUSTED daily data."""
    cache = Path(cache_dir) / "prices"
    cache.mkdir(parents=True, exist_ok=True)
    start = (date.today() - timedelta(days=int(365.25 * years))).isoformat()

    out, need = {}, []
    for t in tickers:
        p = cache / f"{_safe(t)}.csv"
        if not force and _is_fresh(p, max_age_hours):
            out[t] = _read_price_csv(p)
        else:
            need.append(t)

    if need:
        import yfinance as yf

        print(f"Downloading prices for {len(need)} tickers from Yahoo Finance...")
        raw = None
        for attempt in range(3):
            try:
                # auto_adjust=True gives split/dividend-adjusted OHLC. That matters:
                # an unadjusted 4:1 split would look like a -75% "crash".
                raw = yf.download(need, start=start, auto_adjust=True, group_by="ticker",
                                  progress=False, threads=True)
                if raw is not None and len(raw):
                    break
            except Exception as e:  # network hiccups happen
                print(f"  download attempt {attempt + 1} failed: {e}")
            time.sleep(2 ** (attempt + 1))

        for t in need:
            p = cache / f"{_safe(t)}.csv"
            df = None
            try:
                if raw is not None and isinstance(raw.columns, pd.MultiIndex):
                    if t in raw.columns.get_level_values(0):
                        df = raw[t]
                elif raw is not None and len(need) == 1:
                    df = raw
                if df is not None:
                    df = df[["Open", "High", "Low", "Close", "Volume"]].dropna(subset=["Close"])
                    df.index = pd.to_datetime(df.index).tz_localize(None)
            except Exception as e:
                print(f"  could not parse {t}: {e}")
                df = None

            if df is not None and len(df) > 0:
                df.to_csv(p)
                out[t] = df
            elif p.exists():
                warnings.warn(f"{t}: download failed, using older cached data")
                out[t] = _read_price_csv(p)
            else:
                warnings.warn(f"{t}: no data available, skipping this ticker")
    return out


def apply_start_overrides(prices, overrides):
    """Drop rows before a ticker's allowed start date (e.g. NBIS before 2024-10-21)."""
    for t, d in overrides.items():
        if t in prices:
            prices[t] = prices[t][prices[t].index >= pd.Timestamp(d)]
    return prices


# ----------------------------------------------------------------------------
# Earnings dates
# ----------------------------------------------------------------------------
def download_earnings(tickers, cache_dir, max_age_hours, force=False):
    """
    Return {ticker: DataFrame[date, after_close]}, covering past and scheduled
    future earnings dates.

    after_close=True means the report comes after the bell, so the price reacts
    the NEXT trading day. Yahoo's earnings scraping is sometimes flaky. If it
    fails we return an empty table and the earnings features become "unknown"
    (NaN) for that stock.
    """
    cache = Path(cache_dir) / "earnings"
    cache.mkdir(parents=True, exist_ok=True)
    out = {}
    for t in tickers:
        p = cache / f"{_safe(t)}.csv"
        if not force and _is_fresh(p, max_age_hours):
            df = pd.read_csv(p, parse_dates=["date"])
            out[t] = df
            continue

        import yfinance as yf

        rows = []
        try:
            ed = yf.Ticker(t).get_earnings_dates(limit=80)
            if ed is not None and len(ed):
                idx = pd.DatetimeIndex(ed.index)
                if idx.tz is not None:
                    idx = idx.tz_convert("America/New_York")
                for ts in idx:
                    rows.append({"date": pd.Timestamp(ts.date()), "after_close": ts.hour >= 12})
        except Exception as e:
            print(f"  earnings dates for {t} unavailable ({e})")
        # Backup source for the NEXT date: Ticker.calendar
        try:
            cal = yf.Ticker(t).calendar
            nxt = cal.get("Earnings Date") if isinstance(cal, dict) else None
            for d in (nxt or []):
                d = pd.Timestamp(d)
                if all(r["date"] != d for r in rows):
                    # Time unknown: assume after the close, which is typical for US tech.
                    rows.append({"date": d, "after_close": True})
        except Exception:
            pass

        df = pd.DataFrame(rows, columns=["date", "after_close"])
        if len(df) == 0 and p.exists():
            df = pd.read_csv(p, parse_dates=["date"])   # keep the old cache
        else:
            df = df.drop_duplicates("date").sort_values("date")
            df.to_csv(p, index=False)
        out[t] = df
        time.sleep(0.3)   # be polite to Yahoo
    return out


# ----------------------------------------------------------------------------
# FRED (10y yield)
# ----------------------------------------------------------------------------
def download_fred(series, cache_dir, max_age_hours, force=False):
    """Return a pd.Series of the FRED series indexed by date (no API key)."""
    cache = Path(cache_dir)
    cache.mkdir(parents=True, exist_ok=True)
    p = cache / f"fred_{series}.csv"
    if force or not _is_fresh(p, max_age_hours):
        try:
            r = requests.get(FRED_URL.format(series=series), timeout=30)
            r.raise_for_status()
            p.write_text(r.text)
        except Exception as e:
            if not p.exists():
                warnings.warn(f"FRED download failed ({e}); yield features will be missing")
                return pd.Series(dtype=float, name=series)
            warnings.warn(f"FRED download failed ({e}); using older cached data")
    df = pd.read_csv(p)
    date_col = df.columns[0]               # 'observation_date' (new) or 'DATE' (old)
    s = pd.to_numeric(df[series], errors="coerce")   # FRED uses '.' for missing days
    s.index = pd.to_datetime(df[date_col])
    return s.dropna().rename(series)


# ----------------------------------------------------------------------------
# One call to get everything
# ----------------------------------------------------------------------------
def load_all(cfg, max_age_hours=None, force=False):
    d = cfg["data"]
    age = d["cache_max_age_hours"] if max_age_hours is None else max_age_hours
    tickers = cfg["targets"] + cfg["context"]["tickers"]
    prices = download_prices(tickers, d["years"], d["cache_dir"], age, force)
    prices = apply_start_overrides(prices, d["start_overrides"])
    if "SPY" not in prices:
        raise RuntimeError("SPY data is required (it defines the trading calendar). "
                           "Is Yahoo Finance reachable?")
    targets = [t for t in cfg["targets"] if t in prices]
    earnings = download_earnings(targets, d["cache_dir"], max(age, 24), force)
    fred = download_fred(cfg["context"]["fred_series"], d["cache_dir"], age, force)
    return prices, earnings, fred
