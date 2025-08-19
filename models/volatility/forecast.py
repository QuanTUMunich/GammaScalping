# models/volatility/forecast.py

import pandas as pd
import numpy as np
from arch import arch_model

from models.volatility.targets import log_returns

ann_days  = 365           
ann_hours  = 24 * ann_days

def make_vol_data(df_spot: pd.DataFrame, window: int, *, freq: str = "daily") -> pd.DataFrame:
    """
    Build a small table with close, log_returns, and a rolling vol estimate (annualized, %).

    Parameters
    ----------
    df_spot : DataFrame
        Hourly OHLC dataframe with DatetimeIndex and column 'close'.
        (Already cleaned: datetime index set & sorted.)
    window : int
        Rolling window length in periods of `freq` (days if daily, hours if hourly).
    freq : {'daily','hourly'}
        Frequency at which to compute returns & rolling vol.

    Returns
    -------
    DataFrame indexed by time with columns:
      - 'close'        : close at chosen frequency
      - 'log_returns'  : log returns at chosen frequency
      - 'rv_est_pct'   : rolling std * sqrt(ann_factor) * 100 (annualized %)
    """
    freq = freq.lower()
    if freq not in {"daily", "hourly"}:
        raise ValueError("freq must be 'daily' or 'hourly'")

    df_spot = df_spot.sort_index()

    # 1) returns via your helper
    r = log_returns(df_spot, out_freq=freq)

    # 2) matching close series and annualization
    if freq == "hourly":
        ann = ann_hours
        close = df_spot["close"].asfreq("h")
    else:
        ann = ann_days
        close = df_spot["close"].resample("1D").last()

    # 3) assemble output
    out = pd.DataFrame(index=r.index)
    out["close"] = close.reindex(out.index)
    out["log_returns"] = r

    # 4) rolling realized-vol estimate (annualized, %)
    rv = out["log_returns"].rolling(window).std(ddof=0)
    out["rv_est_pct"] = rv * np.sqrt(ann) * 100.0

    return out

def naive_forecast(rv_est_pct: pd.Series) -> pd.Series:
    """
    Naive volatility forecast (annualized, %):
    simply use the last available volatility estimate as the forecast for the next step.
    """
    return rv_est_pct.shift(1).rename("fcst_naive_pct")

def garch_forecast(
    df_spot: pd.DataFrame,
    *,
    freq: str = "daily",      # 'daily' or 'hourly'
    horizon: int = 1,         # forecast horizon in periods of `freq`
    annualization: int | None = None,  # default: 365 (daily) or 8760 (hourly)
    mean: str = "constant",
    vol: str = "GARCH",
    p: int = 1, q: int = 1, o: int = 0,
    dist: str = "normal",     # 'normal' | 't' | 'skewt'
) -> pd.Series:
    """
    Rolling (dynamic) GARCH forecast of volatility.

    Returns an *annualized percent* volatility forecast aligned at time t:
      - horizon=1: sqrt(Var_{t+1}) * sqrt(ann) * 100
      - horizon>1: sqrt((ann/h) * sum_{i=1..h} Var_{t+i}) * 100

    Parameters
    ----------
    df_spot : hourly OHLC with DatetimeIndex and 'close' column (already cleaned)
    freq    : 'daily' or 'hourly' (controls the returns frequency)
    horizon : steps ahead (days if daily, hours if hourly)
    annualization : override annualization (default: 365 or 8760)
    mean, vol, p,q,o, dist : arch model settings

    Returns
    -------
    pd.Series : forecast in annualized %, named 'fcst_garch_{h}{unit}_pct'
    """
    freq = freq.lower()
    if freq not in {"daily", "hourly"}:
        raise ValueError("freq must be 'daily' or 'hourly'")
    if horizon < 1:
        raise ValueError("horizon must be >= 1")

    # build returns at the chosen frequency
    r = log_returns(df_spot.sort_index(), out_freq=freq).dropna()

    if annualization is None:
        annualization = ann_hours if freq == "hourly" else ann_days

    # fit GARCH on the full sample (one-shot) and get dynamic forecasts
    am = arch_model(r, mean=mean, vol=vol, p=p, o=o, q=q, dist=dist, rescale=False)
    res = am.fit(disp="off")  # silent fit

    # 4) dynamic 1..h-step-ahead forecasts for *each* t (start=0)
    f = res.forecast(horizon=horizon, start=0, reindex=True)  # variance forecasts

    # f.variance has one column per step (step 1 = next period)
    if horizon == 1:
        var_h = f.variance.iloc[:, 0]
        vol_pct = np.sqrt(var_h * annualization) * 100.0
    else:
        var_sum = f.variance.iloc[:, :horizon].sum(axis=1)   # sum_{i=1..h} Var_{t+i}
        vol_pct = np.sqrt((annualization / horizon) * var_sum) * 100.0

    unit = "h" if freq == "hourly" else "d"
    vol_pct.name = f"fcst_garch_{horizon}{unit}_pct"
    return vol_pct