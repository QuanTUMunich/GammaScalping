import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.options_pricing import (
    bs_price, delta, gamma, theta, vega, rho, implied_volatility
)

@dataclass
class OptionPosition:
    """Enhanced option position with full Greeks tracking"""
    symbol: str
    strike: float
    expiry: pd.Timestamp
    option_type: str  # 'call' or 'put'
    quantity: float
    entry_price: float
    entry_spot: float
    entry_iv: float
    entry_time: pd.Timestamp
    
    # Current values
    current_price: float = 0
    current_spot: float = 0
    current_iv: float = 0
    time_to_expiry: float = 0
    
    # Greeks at entry
    entry_delta: float = 0
    entry_gamma: float = 0
    entry_theta: float = 0
    entry_vega: float = 0
    
    # Current Greeks
    current_delta: float = 0
    current_gamma: float = 0
    current_theta: float = 0
    current_vega: float = 0
    
    def update_greeks(self, spot: float, current_time: pd.Timestamp, iv: float, r: float = 0.01):
        """Update current Greeks based on new market conditions"""
        self.current_spot = spot
        self.current_iv = iv
        self.time_to_expiry = max(0, (self.expiry - current_time).total_seconds() / (365 * 24 * 3600))
        
        if self.time_to_expiry > 0:
            # Calculate current Greeks
            self.current_delta = delta(spot, self.strike, self.time_to_expiry, r, iv, self.option_type)
            self.current_gamma = gamma(spot, self.strike, self.time_to_expiry, r, iv)
            self.current_theta = theta(spot, self.strike, self.time_to_expiry, r, iv, self.option_type)
            self.current_vega = vega(spot, self.strike, self.time_to_expiry, r, iv)
            
            # Calculate theoretical price
            self.current_price = bs_price(spot, self.strike, self.time_to_expiry, r, iv, self.option_type)
        else:
            # Option expired
            self.current_delta = 0
            self.current_gamma = 0
            self.current_theta = 0
            self.current_vega = 0
            
            # Intrinsic value at expiry
            if self.option_type == 'call':
                self.current_price = max(0, spot - self.strike)
            else:
                self.current_price = max(0, self.strike - spot)
    
    def get_pnl(self) -> float:
        """Calculate P&L for this position"""
        return self.quantity * (self.current_price - self.entry_price)
    
    def get_position_greeks(self) -> Dict[str, float]:
        """Get position-weighted Greeks"""
        return {
            'delta': self.quantity * self.current_delta,
            'gamma': self.quantity * self.current_gamma,
            'theta': self.quantity * self.current_theta,
            'vega': self.quantity * self.current_vega
        }

@dataclass
class StockPosition:
    """Stock/underlying position for hedging"""
    symbol: str
    quantity: float
    avg_price: float
    current_price: float = 0
    
    def add_trade(self, quantity: float, price: float):
        """Add a trade to the position"""
        total_value = self.quantity * self.avg_price + quantity * price
        self.quantity += quantity
        if self.quantity != 0:
            self.avg_price = total_value / self.quantity
        else:
            self.avg_price = 0
    
    def get_pnl(self) -> float:
        """Calculate P&L for stock position"""
        return self.quantity * (self.current_price - self.avg_price)
    
    def get_market_value(self) -> float:
        """Get current market value"""
        return self.quantity * self.current_price

class PortfolioDynamics:
    """
    Manages the complete portfolio dynamics for gamma scalping.
    Tracks options, hedges, and provides real-time Greeks and P&L.
    """
    
    def __init__(self, risk_free_rate: float = 0.01):
        self.risk_free_rate = risk_free_rate
        self.option_positions: Dict[str, OptionPosition] = {}
        self.stock_positions: Dict[str, StockPosition] = {}
        self.cash: float = 0
        
        # History tracking
        self.greek_history: List[Dict] = []
        self.pnl_history: List[Dict] = []
        self.position_history: List[Dict] = []
        
    def add_option_position(
        self,
        symbol: str,
        strike: float,
        expiry: pd.Timestamp,
        option_type: str,
        quantity: float,
        price: float,
        spot: float,
        iv: float,
        timestamp: pd.Timestamp
    ) -> OptionPosition:
        """Add a new option position to the portfolio"""
        
        position_id = f"{symbol}_{strike}_{expiry.strftime('%Y%m%d')}_{option_type}"
        
        # Create position
        position = OptionPosition(
            symbol=symbol,
            strike=strike,
            expiry=expiry,
            option_type=option_type,
            quantity=quantity,
            entry_price=price,
            entry_spot=spot,
            entry_iv=iv,
            entry_time=timestamp,
            current_price=price,
            current_spot=spot,
            current_iv=iv
        )
        
        # Calculate initial Greeks
        position.update_greeks(spot, timestamp, iv, self.risk_free_rate)
        position.entry_delta = position.current_delta
        position.entry_gamma = position.current_gamma
        position.entry_theta = position.current_theta
        position.entry_vega = position.current_vega
        
        # Add to portfolio
        if position_id in self.option_positions:
            # Aggregate position
            existing = self.option_positions[position_id]
            total_quantity = existing.quantity + quantity
            avg_price = (existing.quantity * existing.entry_price + quantity * price) / total_quantity
            existing.quantity = total_quantity
            existing.entry_price = avg_price
        else:
            self.option_positions[position_id] = position
        
        # Update cash
        self.cash -= quantity * price
        
        return position
    
    def add_stock_hedge(
        self,
        symbol: str,
        quantity: float,
        price: float
    ) -> StockPosition:
        """Add a stock hedge position"""
        
        if symbol in self.stock_positions:
            self.stock_positions[symbol].add_trade(quantity, price)
        else:
            self.stock_positions[symbol] = StockPosition(
                symbol=symbol,
                quantity=quantity,
                avg_price=price,
                current_price=price
            )
        
        # Update cash
        self.cash -= quantity * price
        
        return self.stock_positions[symbol]
    
    def update_market_data(
        self,
        timestamp: pd.Timestamp,
        spot_prices: Dict[str, float],
        option_ivs: Optional[Dict[str, float]] = None,
        option_prices: Optional[Dict[str, float]] = None
    ):
        """Update all positions with new market data"""
        
        # Update stock positions
        for symbol, position in self.stock_positions.items():
            if symbol in spot_prices:
                position.current_price = spot_prices[symbol]
        
        # Update option positions
        for position_id, position in self.option_positions.items():
            if position.symbol in spot_prices:
                spot = spot_prices[position.symbol]
                
                # Use market IV if available, otherwise calculate from price
                if option_ivs and position_id in option_ivs:
                    iv = option_ivs[position_id]
                elif option_prices and position_id in option_prices:
                    market_price = option_prices[position_id]
                    time_to_expiry = max(0, (position.expiry - timestamp).total_seconds() / (365 * 24 * 3600))
                    if time_to_expiry > 0:
                        iv = implied_volatility(
                            market_price, spot, position.strike, 
                            time_to_expiry, self.risk_free_rate, position.option_type
                        )
                    else:
                        iv = position.current_iv
                else:
                    iv = position.current_iv
                
                # Update Greeks and theoretical price
                position.update_greeks(spot, timestamp, iv, self.risk_free_rate)
                
                # If market price available, use it
                if option_prices and position_id in option_prices:
                    position.current_price = option_prices[position_id]
        
        # Record state
        self._record_state(timestamp)
    
    def get_portfolio_greeks(self) -> Dict[str, float]:
        """Calculate total portfolio Greeks"""
        total_greeks = {
            'delta': 0,
            'gamma': 0,
            'theta': 0,
            'vega': 0
        }
        
        # Sum option Greeks
        for position in self.option_positions.values():
            position_greeks = position.get_position_greeks()
            for greek in total_greeks:
                total_greeks[greek] += position_greeks[greek]
        
        # Add stock delta
        for position in self.stock_positions.values():
            total_greeks['delta'] += position.quantity
        
        return total_greeks
    
    def get_pnl_breakdown(self) -> Dict[str, float]:
        """Get detailed P&L breakdown"""
        option_pnl = sum(pos.get_pnl() for pos in self.option_positions.values())
        stock_pnl = sum(pos.get_pnl() for pos in self.stock_positions.values())
        
        return {
            'option_pnl': option_pnl,
            'stock_pnl': stock_pnl,
            'total_pnl': option_pnl + stock_pnl,
            'unrealized_pnl': option_pnl + stock_pnl,
            'cash': self.cash
        }
    
    def calculate_hedge_requirement(
        self,
        target_delta: float = 0,
        hedge_ratio: float = 1.0
    ) -> float:
        """Calculate required hedge quantity to achieve target delta"""
        current_greeks = self.get_portfolio_greeks()
        current_delta = current_greeks['delta']
        
        # Calculate hedge needed
        delta_gap = target_delta - current_delta
        hedge_quantity = delta_gap * hedge_ratio
        
        return hedge_quantity
    
    def decompose_pnl(
        self,
        start_time: pd.Timestamp,
        end_time: pd.Timestamp
    ) -> Dict[str, float]:
        """
        Decompose P&L into components (gamma, theta, vega, etc.)
        This is a simplified version - full implementation would use
        Taylor series expansion of option price
        """
        
        pnl_components = {
            'delta_pnl': 0,
            'gamma_pnl': 0,
            'theta_pnl': 0,
            'vega_pnl': 0,
            'unexplained': 0
        }
        
        time_decay = (end_time - start_time).total_seconds() / (365 * 24 * 3600)
        
        for position in self.option_positions.values():
            # Simplified P&L attribution
            spot_change = position.current_spot - position.entry_spot
            iv_change = position.current_iv - position.entry_iv
            
            # Delta P&L (first-order spot move)
            pnl_components['delta_pnl'] += position.quantity * position.entry_delta * spot_change
            
            # Gamma P&L (second-order spot move)
            pnl_components['gamma_pnl'] += 0.5 * position.quantity * position.entry_gamma * spot_change ** 2
            
            # Theta P&L (time decay)
            pnl_components['theta_pnl'] += position.quantity * position.entry_theta * time_decay
            
            # Vega P&L (IV change)
            pnl_components['vega_pnl'] += position.quantity * position.entry_vega * iv_change
        
        # Calculate unexplained P&L
        total_pnl = self.get_pnl_breakdown()['total_pnl']
        explained_pnl = sum(pnl_components.values()) - pnl_components['unexplained']
        pnl_components['unexplained'] = total_pnl - explained_pnl
        
        return pnl_components
    
    def _record_state(self, timestamp: pd.Timestamp):
        """Record current portfolio state for analysis"""
        
        # Record Greeks
        greeks = self.get_portfolio_greeks()
        greeks['timestamp'] = timestamp
        self.greek_history.append(greeks)
        
        # Record P&L
        pnl = self.get_pnl_breakdown()
        pnl['timestamp'] = timestamp
        self.pnl_history.append(pnl)
        
        # Record positions
        position_snapshot = {
            'timestamp': timestamp,
            'num_options': len(self.option_positions),
            'num_stocks': len(self.stock_positions),
            'total_option_quantity': sum(abs(p.quantity) for p in self.option_positions.values()),
            'total_stock_quantity': sum(abs(p.quantity) for p in self.stock_positions.values())
        }
        self.position_history.append(position_snapshot)
    
    def get_history_dataframe(self) -> pd.DataFrame:
        """Convert history to DataFrame for analysis"""
        
        # Combine all history
        df_greeks = pd.DataFrame(self.greek_history)
        df_pnl = pd.DataFrame(self.pnl_history)
        df_positions = pd.DataFrame(self.position_history)
        
        # Merge on timestamp
        if not df_greeks.empty and not df_pnl.empty:
            df = df_greeks.merge(df_pnl, on='timestamp', how='outer')
            if not df_positions.empty:
                df = df.merge(df_positions, on='timestamp', how='outer')
            return df.sort_values('timestamp').reset_index(drop=True)
        
        return pd.DataFrame()
    
    def calculate_risk_metrics(self) -> Dict[str, float]:
        """Calculate portfolio risk metrics"""
        
        greeks = self.get_portfolio_greeks()
        pnl = self.get_pnl_breakdown()
        
        # Calculate key risk metrics
        metrics = {
            'delta_exposure': greeks['delta'],
            'gamma_exposure': greeks['gamma'],
            'theta_decay_daily': greeks['theta'],
            'vega_exposure': greeks['vega'],
            'total_market_value': sum(p.quantity * p.current_price for p in self.option_positions.values()),
            'hedge_value': sum(p.get_market_value() for p in self.stock_positions.values()),
            'net_exposure': greeks['delta'],
            'gamma_scalp_potential': abs(greeks['gamma']),  # Simplified metric
        }
        
        return metrics