import numpy as np
import pandas as pd
import pytest

from models.options_pricing import bs_price
from simulation.gamma_scalping_simulator import GammaScalpingSimulator

SIGMA, K, R = 0.6, 100_000.0, 0.01


def _market(option_every_n_hours=1, seed=0):
    """Hourly GBM spot and a Deribit-style call quoted in BTC, priced at SIGMA."""
    rng = np.random.default_rng(seed)
    ts = pd.date_range("2025-01-01", periods=24 * 30, freq="h")
    expiry = ts[-1] + pd.Timedelta(hours=1)
    dt = 1 / (365 * 24)
    spot = K * np.exp(np.cumsum((R - 0.5 * SIGMA**2) * dt + SIGMA * np.sqrt(dt) * rng.standard_normal(len(ts))))
    tte = (expiry - ts).total_seconds().to_numpy() / (365 * 24 * 3600)
    price_btc = bs_price(spot, K, tte, R, SIGMA, "call") / spot
    spot_df = pd.DataFrame({"timestamp": ts, "close": spot})
    opt_df = pd.DataFrame({"timestamp": ts, "close": price_btc, "strike": K,
                           "expiry": expiry, "option_type": "call"}).iloc[::option_every_n_hours]
    return spot_df, opt_df.reset_index(drop=True)


def _sim(**kw):
    return GammaScalpingSimulator(commission_rate=0.0, slippage_bps=0.0, risk_free_rate=R, **kw)


def test_btc_quoted_prices_recover_the_true_implied_vol():
    spot_df, opt_df = _market()
    res = _sim().simulate(spot_df, opt_df, hedge_threshold=0.0)
    np.testing.assert_allclose(res.greeks_history["iv"], SIGMA, atol=1e-4)


def test_hedged_pnl_is_small_when_realized_equals_implied():
    spot_df, opt_df = _market()
    res = _sim().simulate(spot_df, opt_df, hedge_threshold=0.0)
    premium_usd = opt_df["close"].iloc[0] * spot_df["close"].iloc[0]
    assert abs(res.total_pnl) < 0.1 * premium_usd
    assert res.total_pnl == pytest.approx(res.option_pnl + res.hedge_pnl, abs=1e-6)


def test_hedges_hourly_even_when_the_option_trades_rarely():
    spot_df, opt_df = _market(option_every_n_hours=24)
    res = _sim().simulate(spot_df, opt_df, hedge_threshold=0.0)
    assert len(res.pnl_history) > 24 * 25
