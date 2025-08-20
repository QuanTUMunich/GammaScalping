# models/volatility/forecast.py

import pandas as pd
import numpy as np
from arch import arch_model

from .target import log_returns

ann_days  = 365           
ann_hours  = 24 * ann_days

# Naive forecast from any annualized % estimator series
def naive_forecast(est_pct: pd.Series, horizon: int = 1) -> pd.Series:
    """
    Naive volatility forecast (annualized, %).
    Uses today's estimator as the forecast for the next h days.
    Output remains annualized, %.

    Parameters
    ----------
    est_pct : pd.Series
        Annualized % volatility estimator (e.g. rv_est_pct / rv_parkinson_pct / rv_rs_pct).
    horizon : int, default=1
        Forecast horizon in calendar days (or hours if your estimator is hourly).

    Returns
    -------
    pd.Series
        Series of forecasts, annualized %, named f'fcst_naive_h{h}_pct'.
    """
    if horizon < 1:
        raise ValueError("horizon must be >= 1")

    # Forecast = current estimate shifted forward to align with t+h target
    f = est_pct.shift(1) * np.sqrt(horizon)
    f.name = f"fcst_naive_h{h}_pct"
    return f


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