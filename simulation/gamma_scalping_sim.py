import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field
from collections import defaultdict
import warnings

@dataclass
class Position:
    """Represents a single position (option or stock)"""
    instrument_type: str  # 'option' or 'stock'
    quantity: float
    entry_price: float
    entry_time: pd.Timestamp
    strike: Optional[float] = None
    expiry: Optional[pd.Timestamp] = None
    option_type: Optional[str] = None  # 'call' or 'put'
    
@dataclass
class Trade:
    """Records a single trade execution"""
    timestamp: pd.Timestamp
    instrument_type: str
    action: str  # 'buy' or 'sell'
    quantity: float
    price: float
    commission: float
    slippage: float
    trade_type: str  # 'open', 'hedge', 'close'
    
@dataclass
class PositionSnapshot:
    """Snapshot of position state at a point in time"""
    timestamp: pd.Timestamp
    spot_price: float
    positions: Dict[str, Position]
    cash: float
    # Greeks
    delta: float
    gamma: float
    theta: float
    vega: float
    # P&L components
    option_pnl: float
    hedge_pnl: float
    total_pnl: float
    realized_pnl: float
    unrealized_pnl: float

class GammaScalpingSimulator:
    """
    Core simulation engine for gamma scalping strategies.
    Handles position management, execution simulation, and P&L tracking.
    """
    
    def __init__(
        self,
        initial_capital: float = 100000,
        commission_rate: float = 0.0005,  # 5 bps
        slippage_bps: float = 10,  # 10 bps slippage
        borrow_rate: float = 0.0,  # Annual borrow rate for short stock
        risk_free_rate: float = 0.01,  # Risk-free rate for options pricing
    ):
        self.initial_capital = initial_capital
        self.cash = initial_capital
        self.commission_rate = commission_rate
        self.slippage_bps = slippage_bps / 10000
        self.borrow_rate = borrow_rate
        self.risk_free_rate = risk_free_rate
        
        # Position tracking
        self.positions = {}  # Dict of Position objects
        self.trades = []  # List of Trade objects
        self.snapshots = []  # List of PositionSnapshot objects
        
        # P&L tracking
        self.realized_pnl = 0
        self.commissions_paid = 0
        self.slippage_paid = 0
        
        # Greeks tracking
        self.current_delta = 0
        self.current_gamma = 0
        self.current_theta = 0
        self.current_vega = 0
        
    def open_option_position(
        self,
        timestamp: pd.Timestamp,
        strike: float,
        expiry: pd.Timestamp,
        option_type: str,
        quantity: float,
        price: float,
        spot_price: float,
        implied_vol: float
    ) -> Trade:
        """Open a new option position"""
        
        # Calculate execution costs
        notional = abs(quantity * price)
        commission = notional * self.commission_rate
        slippage = notional * self.slippage_bps
        total_cost = notional + commission + slippage
        
        if total_cost > self.cash:
            raise ValueError(f"Insufficient cash: need {total_cost:.2f}, have {self.cash:.2f}")
        
        # Create position
        position_id = f"{option_type}_{strike}_{expiry.strftime('%Y%m%d')}"
        self.positions[position_id] = Position(
            instrument_type='option',
            quantity=quantity,
            entry_price=price,
            entry_time=timestamp,
            strike=strike,
            expiry=expiry,
            option_type=option_type
        )
        
        # Update cash
        self.cash -= total_cost
        self.commissions_paid += commission
        self.slippage_paid += slippage
        
        # Record trade
        trade = Trade(
            timestamp=timestamp,
            instrument_type='option',
            action='buy' if quantity > 0 else 'sell',
            quantity=quantity,
            price=price,
            commission=commission,
            slippage=slippage,
            trade_type='open'
        )
        self.trades.append(trade)
        
        # Update Greeks (would need Black-Scholes here)
        self._update_greeks(spot_price, implied_vol, timestamp)
        
        return trade
    
    def hedge_delta(
        self,
        timestamp: pd.Timestamp,
        spot_price: float,
        target_delta: float = 0
    ) -> Optional[Trade]:
        """Hedge delta exposure by trading the underlying"""
        
        # Calculate hedge requirement
        hedge_quantity = target_delta - self.current_delta
        
        if abs(hedge_quantity) < 0.001:  # Minimum trade size
            return None
        
        # Calculate execution costs
        notional = abs(hedge_quantity * spot_price)
        commission = notional * self.commission_rate
        slippage = notional * self.slippage_bps
        
        # Execute hedge
        execution_price = spot_price * (1 + self.slippage_bps if hedge_quantity > 0 else 1 - self.slippage_bps)
        total_cost = hedge_quantity * execution_price + commission
        
        # Update or create stock position
        if 'stock' in self.positions:
            self.positions['stock'].quantity += hedge_quantity
        else:
            self.positions['stock'] = Position(
                instrument_type='stock',
                quantity=hedge_quantity,
                entry_price=execution_price,
                entry_time=timestamp
            )
        
        # Update cash
        self.cash -= total_cost
        self.commissions_paid += commission
        self.slippage_paid += slippage
        
        # Record trade
        trade = Trade(
            timestamp=timestamp,
            instrument_type='stock',
            action='buy' if hedge_quantity > 0 else 'sell',
            quantity=hedge_quantity,
            price=execution_price,
            commission=commission,
            slippage=slippage,
            trade_type='hedge'
        )
        self.trades.append(trade)
        
        # Update delta
        self.current_delta += hedge_quantity
        
        return trade
    
    def update_position_values(
        self,
        timestamp: pd.Timestamp,
        spot_price: float,
        option_prices: Dict[str, float],
        implied_vols: Dict[str, float]
    ):
        """Update position values and calculate P&L"""
        
        option_pnl = 0
        hedge_pnl = 0
        
        # Calculate option P&L
        for position_id, position in self.positions.items():
            if position.instrument_type == 'option':
                current_price = option_prices.get(position_id, 0)
                position_pnl = position.quantity * (current_price - position.entry_price)
                option_pnl += position_pnl
            elif position.instrument_type == 'stock':
                position_pnl = position.quantity * (spot_price - position.entry_price)
                hedge_pnl += position_pnl
        
        # Update Greeks
        self._update_greeks(spot_price, implied_vols, timestamp)
        
        # Calculate total P&L
        total_pnl = option_pnl + hedge_pnl - self.commissions_paid - self.slippage_paid
        
        # Create snapshot
        snapshot = PositionSnapshot(
            timestamp=timestamp,
            spot_price=spot_price,
            positions=self.positions.copy(),
            cash=self.cash,
            delta=self.current_delta,
            gamma=self.current_gamma,
            theta=self.current_theta,
            vega=self.current_vega,
            option_pnl=option_pnl,
            hedge_pnl=hedge_pnl,
            total_pnl=total_pnl,
            realized_pnl=self.realized_pnl,
            unrealized_pnl=total_pnl - self.realized_pnl
        )
        self.snapshots.append(snapshot)
        
        return snapshot
    
    def _update_greeks(
        self,
        spot_price: float,
        implied_vols: Dict[str, float],
        timestamp: pd.Timestamp
    ):
        """Update portfolio Greeks (placeholder - would integrate with options_pricing.py)"""
        # This would call the actual Greeks calculation from models/options_pricing.py
        # For now, using placeholder values
        total_delta = 0
        total_gamma = 0
        total_theta = 0
        total_vega = 0
        
        for position_id, position in self.positions.items():
            if position.instrument_type == 'option':
                # Would calculate actual Greeks here
                # delta = calculate_delta(spot_price, position.strike, ...)
                # For now, placeholder
                pass
            elif position.instrument_type == 'stock':
                total_delta += position.quantity
        
        self.current_delta = total_delta
        self.current_gamma = total_gamma
        self.current_theta = total_theta
        self.current_vega = total_vega
    
    def calculate_performance_metrics(self) -> Dict:
        """Calculate key performance metrics"""
        if not self.snapshots:
            return {}
        
        # Convert snapshots to DataFrame for easier analysis
        df = pd.DataFrame([
            {
                'timestamp': s.timestamp,
                'total_pnl': s.total_pnl,
                'option_pnl': s.option_pnl,
                'hedge_pnl': s.hedge_pnl,
                'delta': s.delta,
                'gamma': s.gamma
            }
            for s in self.snapshots
        ])
        
        # Calculate returns
        df['returns'] = df['total_pnl'].diff() / self.initial_capital
        
        # Key metrics
        total_return = df['total_pnl'].iloc[-1] / self.initial_capital
        sharpe_ratio = np.sqrt(252) * df['returns'].mean() / df['returns'].std() if len(df) > 1 and df['returns'].std() > 0 else 0
        
        # Fix max drawdown calculation - should be percentage of capital, not absolute
        cumulative_returns = (1 + df['returns']).cumprod()
        running_max = cumulative_returns.expanding().max()
        drawdown_series = (cumulative_returns - running_max) / running_max
        max_drawdown = abs(drawdown_series.min()) if len(drawdown_series) > 0 else 0
        
        win_rate = (df['returns'] > 0).mean() if len(df) > 1 else 0
        
        # P&L attribution
        option_pnl_pct = df['option_pnl'].iloc[-1] / self.initial_capital if len(df) > 0 else 0
        hedge_pnl_pct = df['hedge_pnl'].iloc[-1] / self.initial_capital if len(df) > 0 else 0
        
        # Transaction costs
        total_commissions = self.commissions_paid
        total_slippage = self.slippage_paid
        num_trades = len(self.trades)
        
        return {
            'total_return': total_return,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'win_rate': win_rate,
            'option_pnl_pct': option_pnl_pct,
            'hedge_pnl_pct': hedge_pnl_pct,
            'total_commissions': total_commissions,
            'total_slippage': total_slippage,
            'num_trades': num_trades,
            'avg_delta': df['delta'].mean() if len(df) > 0 else 0,
            'avg_gamma': df['gamma'].mean() if len(df) > 0 else 0
        }
    
    def generate_report(self) -> pd.DataFrame:
        """Generate comprehensive simulation report"""
        metrics = self.calculate_performance_metrics()
        
        # Create summary DataFrame
        summary = pd.DataFrame([metrics])
        
        # Add trade statistics
        if self.trades:
            trades_df = pd.DataFrame([
                {
                    'timestamp': t.timestamp,
                    'type': t.trade_type,
                    'instrument': t.instrument_type,
                    'action': t.action,
                    'quantity': t.quantity,
                    'price': t.price,
                    'cost': t.commission + t.slippage
                }
                for t in self.trades
            ])
            
            # Calculate additional trade stats
            hedge_trades = trades_df[trades_df['type'] == 'hedge']
            if not hedge_trades.empty:
                summary['avg_hedge_size'] = hedge_trades['quantity'].abs().mean()
                summary['hedge_frequency'] = len(hedge_trades)
        
        return summary