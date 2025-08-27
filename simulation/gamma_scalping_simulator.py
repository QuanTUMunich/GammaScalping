"""
Gamma Scalping Simulator
Simulates gamma scalping strategy for individual options with clear P&L attribution
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.options_pricing import bs_price, delta, gamma, theta, vega, implied_volatility

@dataclass
class GammaScalpingResult:
    """Results from simulating gamma scalping on an option"""
    option_symbol: str
    strike: float
    expiry: pd.Timestamp
    option_type: str
    
    # P&L components
    option_pnl: float
    hedge_pnl: float
    total_pnl: float
    commission_cost: float
    slippage_cost: float
    
    # Trading statistics
    num_hedges: int
    total_hedge_volume: float
    avg_hedge_size: float
    
    # Greeks statistics
    avg_delta: float
    avg_gamma: float
    total_gamma_pnl: float
    total_theta_cost: float
    
    # Time series data
    pnl_history: pd.DataFrame
    greeks_history: pd.DataFrame
    trades: pd.DataFrame
    

class GammaScalpingSimulator:
    """
    Simulates gamma scalping strategy for options.
    Provides clear P&L attribution between option and hedge components.
    """
    
    def __init__(
        self,
        commission_rate: float = 0.0005,
        slippage_bps: float = 10,
        risk_free_rate: float = 0.01
    ):
        self.commission_rate = commission_rate
        self.slippage_bps = slippage_bps / 10000
        self.risk_free_rate = risk_free_rate
        
    def simulate(
        self,
        spot_prices: pd.DataFrame,
        option_data: pd.DataFrame,
        hedge_threshold: float = 0.1,
        position_size: float = 1.0
    ) -> GammaScalpingResult:
        """
        Simulate gamma scalping for a single option using delta-band hedging.
        
        Args:
            spot_prices: DataFrame with columns ['timestamp', 'close']
            option_data: DataFrame with option prices for a single option
            hedge_threshold: Delta threshold for rehedging
            position_size: Number of option contracts
        
        Returns:
            GammaScalpingResult with detailed P&L and Greeks analysis
        """
        
        if option_data.empty or spot_prices.empty:
            return None
            
        # Get option details from first row
        first_row = option_data.iloc[0]
        strike = first_row['strike']
        expiry = first_row['expiry']
        option_type = first_row['option_type']
        symbol = first_row.get('symbol', f"{strike}_{option_type}")
        
        # Initialize tracking variables
        hedge_position = 0  # Current hedge position in underlying
        last_hedge_time = None
        last_hedge_delta = 0
        
        # Storage for results
        pnl_history = []
        greeks_history = []
        trades = []
        
        # P&L tracking
        option_cost = 0
        hedge_cost_basis = 0
        total_commissions = 0
        total_slippage = 0
        
        # Get initial option price and open position
        initial_timestamp = option_data.iloc[0]['timestamp']
        initial_spot = spot_prices[spot_prices['timestamp'] == initial_timestamp]['close'].iloc[0]
        initial_option_price = option_data.iloc[0]['close']
        
        # Calculate initial IV
        tte = (expiry - initial_timestamp).total_seconds() / (365 * 24 * 3600)
        initial_iv = implied_volatility(
            initial_option_price, initial_spot, strike, 
            tte, self.risk_free_rate, option_type
        )
        
        if np.isnan(initial_iv):
            initial_iv = 0.5  # Default IV if calculation fails
            
        # Open option position
        option_cost = position_size * initial_option_price
        option_commission = abs(option_cost) * self.commission_rate
        option_slippage = abs(option_cost) * self.slippage_bps
        total_commissions += option_commission
        total_slippage += option_slippage
        
        trades.append({
            'timestamp': initial_timestamp,
            'type': 'option',
            'action': 'buy',
            'quantity': position_size,
            'price': initial_option_price,
            'commission': option_commission,
            'slippage': option_slippage
        })
        
        # Main simulation loop
        for idx, row in option_data.iterrows():
            timestamp = row['timestamp']
            option_price = row['close']
            
            # Get spot price at this timestamp
            spot_rows = spot_prices[spot_prices['timestamp'] == timestamp]
            if spot_rows.empty:
                continue
            spot_price = spot_rows['close'].iloc[0]
            
            # Calculate time to expiry
            tte = (expiry - timestamp).total_seconds() / (365 * 24 * 3600)
            if tte <= 0:
                break  # Option expired
                
            # Calculate current IV
            current_iv = implied_volatility(
                option_price, spot_price, strike,
                tte, self.risk_free_rate, option_type
            )
            if np.isnan(current_iv):
                current_iv = initial_iv  # Use last known IV
                
            # Calculate Greeks
            current_delta = delta(spot_price, strike, tte, self.risk_free_rate, current_iv, option_type)
            current_gamma = gamma(spot_price, strike, tte, self.risk_free_rate, current_iv)
            current_theta = theta(spot_price, strike, tte, self.risk_free_rate, current_iv, option_type)
            current_vega = vega(spot_price, strike, tte, self.risk_free_rate, current_iv)
            
            # Scale by position size
            position_delta = current_delta * position_size
            position_gamma = current_gamma * position_size
            position_theta = current_theta * position_size
            position_vega = current_vega * position_size
            
            # Determine if we need to hedge using delta-band method
            # Hedge if delta has moved beyond threshold
            delta_diff = abs(position_delta - last_hedge_delta)
            should_hedge = delta_diff > hedge_threshold
                
            # Execute hedge if needed
            hedge_trade_size = 0
            if should_hedge:
                # Calculate hedge size to neutralize delta
                target_hedge = -position_delta  # Negative because we hedge opposite
                hedge_trade_size = target_hedge - hedge_position
                
                if abs(hedge_trade_size) > 0.001:  # Minimum trade size
                    # Execute hedge trade
                    hedge_notional = abs(hedge_trade_size * spot_price)
                    hedge_commission = hedge_notional * self.commission_rate
                    hedge_slippage = hedge_notional * self.slippage_bps
                    
                    total_commissions += hedge_commission
                    total_slippage += hedge_slippage
                    
                    # Update hedge cost basis for P&L calculation
                    hedge_cost_basis += hedge_trade_size * spot_price
                    hedge_cost_basis += hedge_commission + hedge_slippage
                    
                    # Update position
                    hedge_position = target_hedge
                    last_hedge_delta = position_delta
                    last_hedge_time = timestamp
                    
                    trades.append({
                        'timestamp': timestamp,
                        'type': 'hedge',
                        'action': 'buy' if hedge_trade_size > 0 else 'sell',
                        'quantity': abs(hedge_trade_size),
                        'price': spot_price,
                        'commission': hedge_commission,
                        'slippage': hedge_slippage
                    })
            
            # Calculate current P&L
            option_mtm = position_size * option_price
            option_pnl = option_mtm - option_cost
            
            hedge_mtm = hedge_position * spot_price
            hedge_pnl = hedge_mtm - hedge_cost_basis
            
            total_pnl = option_pnl + hedge_pnl - total_commissions - total_slippage
            
            # Store history
            pnl_history.append({
                'timestamp': timestamp,
                'spot_price': spot_price,
                'option_price': option_price,
                'option_pnl': option_pnl,
                'hedge_pnl': hedge_pnl,
                'commission_cost': total_commissions,
                'slippage_cost': total_slippage,
                'total_pnl': total_pnl,
                'hedge_position': hedge_position,
                'net_delta': position_delta + hedge_position
            })
            
            greeks_history.append({
                'timestamp': timestamp,
                'delta': position_delta,
                'gamma': position_gamma,
                'theta': position_theta,
                'vega': position_vega,
                'iv': current_iv
            })
        
        # Close positions at expiry
        if pnl_history:
            final_spot = pnl_history[-1]['spot_price']
            
            # Option payoff at expiry
            if option_type == 'call':
                option_payoff = max(0, final_spot - strike) * position_size
            else:
                option_payoff = max(0, strike - final_spot) * position_size
                
            final_option_pnl = option_payoff - option_cost
            
            # Close hedge position
            if abs(hedge_position) > 0.001:
                hedge_close_notional = abs(hedge_position * final_spot)
                hedge_close_commission = hedge_close_notional * self.commission_rate
                hedge_close_slippage = hedge_close_notional * self.slippage_bps
                
                total_commissions += hedge_close_commission
                total_slippage += hedge_close_slippage
                
                # Final hedge P&L
                final_hedge_pnl = -hedge_cost_basis  # Closed at expiry
                
                trades.append({
                    'timestamp': expiry,
                    'type': 'hedge',
                    'action': 'close',
                    'quantity': abs(hedge_position),
                    'price': final_spot,
                    'commission': hedge_close_commission,
                    'slippage': hedge_close_slippage
                })
            else:
                final_hedge_pnl = 0
                
            final_total_pnl = final_option_pnl + final_hedge_pnl - total_commissions - total_slippage
            
            # Calculate statistics
            pnl_df = pd.DataFrame(pnl_history)
            greeks_df = pd.DataFrame(greeks_history)
            trades_df = pd.DataFrame(trades)
            
            # Estimate gamma P&L (from rehedging gains)
            if len(pnl_df) > 1:
                spot_returns = pnl_df['spot_price'].pct_change()
                spot_moves = pnl_df['spot_price'].diff()
                gamma_pnl = 0.5 * (greeks_df['gamma'] * spot_moves**2).sum()
            else:
                gamma_pnl = 0
                
            # Estimate theta cost
            theta_cost = greeks_df['theta'].sum() / (24 * 365)  # Convert to hourly
            
            return GammaScalpingResult(
                option_symbol=symbol,
                strike=strike,
                expiry=expiry,
                option_type=option_type,
                option_pnl=final_option_pnl,
                hedge_pnl=final_hedge_pnl,
                total_pnl=final_total_pnl,
                commission_cost=total_commissions,
                slippage_cost=total_slippage,
                num_hedges=len(trades_df[trades_df['type'] == 'hedge']),
                total_hedge_volume=trades_df[trades_df['type'] == 'hedge']['quantity'].sum(),
                avg_hedge_size=trades_df[trades_df['type'] == 'hedge']['quantity'].mean() if len(trades_df[trades_df['type'] == 'hedge']) > 0 else 0,
                avg_delta=greeks_df['delta'].mean(),
                avg_gamma=greeks_df['gamma'].mean(),
                total_gamma_pnl=gamma_pnl,
                total_theta_cost=theta_cost,
                pnl_history=pnl_df,
                greeks_history=greeks_df,
                trades=trades_df
            )
        
        return None