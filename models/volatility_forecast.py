 #!/usr/bin/env python3
"""
This script aims to produce horizon-adjustable volatility forecasts (without ML).

Models:
- Naive (use today's vol estimate as forecast)
- EWMA (RiskMetrics) ... here explain more.

Targets:
- Realized volatility over the next h days (annualized, in %)

Input:

Outputs:

Usage example:
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime

def make_vol_data(path, window=30, ann_factor=365):
    df = pd.read_feather(path)
    df['datetime'] = pd.to_datetime(df['datetime'])
    daily = (df.set_index('datetime')['close']
               .resample('1D')
               .last()
               .dropna()
               .to_frame('close'))
    daily['log_returns'] = np.log(daily['close']).diff().fillna(0)
    rv_daily = daily['log_returns'].rolling(window).std(ddof=0)
    rv_pct = rv_daily * np.sqrt(ann_factor) * 100
    daily['rv_est_pct'] = rv_pct
    return daily

def realized_future_vol(log_returns, h=1, ann=365):
    """
    Computes the realized volatility target for forecast evaluation. This is what we will compare our forecasts against.

    For each day t, this function looks ahead to the next h days of log returns
    (t+1 ... t+h), calculates their volatility, annualizes it, and aligns the
    result back at index t.
    """
    sq = (log_returns ** 2).rolling(h).sum().shift(-h)
    return np.sqrt((ann / h) * sq) * 100

def naive_forecast(rv_est_pct, h=1):
    """
    Naive volatility forecast:
    use today's vol estimate as forecast for next h days and scale appropriately.
    """
    return rv_est_pct.shift(1) * np.sqrt(h)

def ewma_vol(log_returns, lam=0.94, ann=365):
    var = []
    prev = log_returns.var()
    for r in log_returns.fillna(0):
        prev = lam * prev + (1 - lam) * r**2
        var.append(prev)
    vol = (pd.Series(var, index=log_returns.index)**0.5) * np.sqrt(ann) * 100
    return vol
