import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.options_pricing import delta, gamma, theta, vega

@dataclass
class PnLComponent:
    """Represents a single component of P&L"""
    component_type: str  # 'delta', 'gamma', 'theta', 'vega', 'rho', 'hedge', 'transaction'
    value: float
    timestamp: pd.Timestamp
    description: str

class PnLDecomposition:
    """
    Advanced P&L decomposition for gamma scalping strategies.
    Breaks down P&L into Greeks components using Taylor series expansion.
    """
    
    def __init__(self):
        self.pnl_components: List[PnLComponent] = []
        self.daily_breakdown: Dict[pd.Timestamp, Dict[str, float]] = {}
        
    def decompose_option_pnl(
        self,
        initial_spot: float,
        final_spot: float,
        initial_iv: float,
        final_iv: float,
        initial_time: pd.Timestamp,
        final_time: pd.Timestamp,
        strike: float,
        option_type: str,
        quantity: float,
        initial_price: float,
        final_price: float,
        risk_free_rate: float = 0.01
    ) -> Dict[str, float]:
        """
        Decompose option P&L using Greeks-based attribution.
        Uses first and second-order Taylor expansion.
        """
        
        # Calculate time decay
        time_diff = (final_time - initial_time).total_seconds() / (365 * 24 * 3600)
        initial_ttm = max(0.001, time_diff)  # Avoid zero time
        
        # Calculate initial Greeks
        initial_delta = delta(initial_spot, strike, initial_ttm, risk_free_rate, initial_iv, option_type)
        initial_gamma = gamma(initial_spot, strike, initial_ttm, risk_free_rate, initial_iv)
        initial_theta = theta(initial_spot, strike, initial_ttm, risk_free_rate, initial_iv, option_type)
        initial_vega = vega(initial_spot, strike, initial_ttm, risk_free_rate, initial_iv)
        
        # Price changes
        spot_change = final_spot - initial_spot
        iv_change = final_iv - initial_iv
        
        # P&L components (per unit)
        delta_pnl = initial_delta * spot_change
        gamma_pnl = 0.5 * initial_gamma * spot_change ** 2
        theta_pnl = initial_theta * time_diff
        vega_pnl = initial_vega * iv_change * 100  # Vega is per 1% IV change
        
        # Higher-order terms and cross-effects
        vanna_pnl = 0  # dDelta/dVol * spot_change * iv_change
        volga_pnl = 0  # dVega/dVol * 0.5 * iv_change^2
        charm_pnl = 0  # dDelta/dTime * time_diff * spot_change
        
        # Total explained P&L
        explained_pnl = delta_pnl + gamma_pnl + theta_pnl + vega_pnl + vanna_pnl + volga_pnl + charm_pnl
        
        # Actual P&L
        actual_pnl = final_price - initial_price
        
        # Unexplained P&L (model error, discrete hedging error, etc.)
        unexplained_pnl = actual_pnl - explained_pnl
        
        # Scale by quantity
        components = {
            'delta_pnl': quantity * delta_pnl,
            'gamma_pnl': quantity * gamma_pnl,
            'theta_pnl': quantity * theta_pnl,
            'vega_pnl': quantity * vega_pnl,
            'higher_order_pnl': quantity * (vanna_pnl + volga_pnl + charm_pnl),
            'unexplained_pnl': quantity * unexplained_pnl,
            'total_pnl': quantity * actual_pnl
        }
        
        return components
    
    def analyze_gamma_scalping_pnl(
        self,
        option_pnl_components: Dict[str, float],
        hedge_trades: pd.DataFrame,
        spot_prices: pd.Series
    ) -> Dict[str, float]:
        """
        Analyze the complete gamma scalping P&L including hedging.
        Separates gamma gains from hedging losses.
        """
        
        # Calculate hedge P&L
        hedge_pnl = 0
        if not hedge_trades.empty:
            for _, trade in hedge_trades.iterrows():
                # Find exit price (next hedge or final spot)
                entry_price = trade['price']
                entry_time = trade['timestamp']
                
                # Find next trade or use final spot
                next_trades = hedge_trades[hedge_trades['timestamp'] > entry_time]
                if not next_trades.empty:
                    exit_price = next_trades.iloc[0]['price']
                else:
                    exit_price = spot_prices.iloc[-1] if not spot_prices.empty else entry_price
                
                trade_pnl = trade['quantity'] * (exit_price - entry_price)
                hedge_pnl += trade_pnl
        
        # Gamma scalping P&L components
        gamma_scalping_pnl = {
            'option_delta_pnl': option_pnl_components.get('delta_pnl', 0),
            'option_gamma_pnl': option_pnl_components.get('gamma_pnl', 0),
            'option_theta_pnl': option_pnl_components.get('theta_pnl', 0),
            'option_vega_pnl': option_pnl_components.get('vega_pnl', 0),
            'hedge_pnl': hedge_pnl,
            'net_gamma_capture': option_pnl_components.get('gamma_pnl', 0) + hedge_pnl,
            'total_strategy_pnl': option_pnl_components.get('total_pnl', 0) + hedge_pnl
        }
        
        return gamma_scalping_pnl
    
    def calculate_daily_attribution(
        self,
        portfolio_history: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Calculate daily P&L attribution from portfolio history.
        """
        
        if portfolio_history.empty:
            return pd.DataFrame()
        
        # Ensure we have required columns
        required_cols = ['timestamp', 'total_pnl', 'option_pnl', 'stock_pnl']
        if not all(col in portfolio_history.columns for col in required_cols):
            return pd.DataFrame()
        
        # Set timestamp as index
        df = portfolio_history.set_index('timestamp').sort_index()
        
        # Calculate daily changes
        daily_attribution = pd.DataFrame(index=df.index)
        
        # Total P&L change
        daily_attribution['total_pnl_change'] = df['total_pnl'].diff()
        daily_attribution['option_pnl_change'] = df['option_pnl'].diff()
        daily_attribution['hedge_pnl_change'] = df['stock_pnl'].diff()
        
        # Greeks attribution (if available)
        if 'delta' in df.columns and 'gamma' in df.columns:
            daily_attribution['avg_delta'] = df['delta'].rolling(2).mean()
            daily_attribution['avg_gamma'] = df['gamma'].rolling(2).mean()
            
            # Estimate Greeks contribution (simplified)
            if 'spot' in portfolio_history.columns:
                spot_returns = df['spot'].pct_change()
                daily_attribution['delta_contribution'] = daily_attribution['avg_delta'] * spot_returns
                daily_attribution['gamma_contribution'] = 0.5 * daily_attribution['avg_gamma'] * (spot_returns ** 2)
        
        # Calculate cumulative attribution
        daily_attribution['cumulative_pnl'] = daily_attribution['total_pnl_change'].cumsum()
        daily_attribution['cumulative_option_pnl'] = daily_attribution['option_pnl_change'].cumsum()
        daily_attribution['cumulative_hedge_pnl'] = daily_attribution['hedge_pnl_change'].cumsum()
        
        return daily_attribution.dropna()
    
    def calculate_trade_statistics(
        self,
        trades: pd.DataFrame
    ) -> Dict[str, float]:
        """
        Calculate detailed statistics for gamma scalping trades.
        """
        
        if trades.empty:
            return {}
        
        # Separate by trade type
        option_trades = trades[trades['instrument_type'] == 'option']
        hedge_trades = trades[trades['instrument_type'] == 'stock']
        
        stats = {}
        
        # Option trade statistics
        if not option_trades.empty:
            stats['num_option_trades'] = len(option_trades)
            stats['avg_option_size'] = option_trades['quantity'].abs().mean()
            stats['total_option_premium'] = (option_trades['quantity'] * option_trades['price']).sum()
        
        # Hedge trade statistics
        if not hedge_trades.empty:
            stats['num_hedge_trades'] = len(hedge_trades)
            stats['avg_hedge_size'] = hedge_trades['quantity'].abs().mean()
            stats['avg_hedge_frequency_hours'] = 0
            
            if len(hedge_trades) > 1:
                time_diffs = hedge_trades['timestamp'].diff().dropna()
                avg_time_diff = time_diffs.mean()
                stats['avg_hedge_frequency_hours'] = avg_time_diff.total_seconds() / 3600
            
            # Calculate hedge efficiency
            buy_hedges = hedge_trades[hedge_trades['quantity'] > 0]
            sell_hedges = hedge_trades[hedge_trades['quantity'] < 0]
            
            if not buy_hedges.empty and not sell_hedges.empty:
                avg_buy_price = (buy_hedges['quantity'] * buy_hedges['price']).sum() / buy_hedges['quantity'].sum()
                avg_sell_price = (sell_hedges['quantity'].abs() * sell_hedges['price']).sum() / sell_hedges['quantity'].abs().sum()
                stats['hedge_efficiency'] = (avg_sell_price - avg_buy_price) / avg_buy_price
        
        # Transaction costs
        if 'commission' in trades.columns and 'slippage' in trades.columns:
            stats['total_commission'] = trades['commission'].sum()
            stats['total_slippage'] = trades['slippage'].sum()
            stats['total_transaction_costs'] = stats['total_commission'] + stats['total_slippage']
            stats['avg_cost_per_trade'] = stats['total_transaction_costs'] / len(trades)
        
        return stats
    
    def generate_attribution_report(
        self,
        portfolio_history: pd.DataFrame,
        trades: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Generate comprehensive P&L attribution report.
        """
        
        # Daily attribution
        daily_attr = self.calculate_daily_attribution(portfolio_history)
        
        # Trade statistics
        trade_stats = self.calculate_trade_statistics(trades)
        
        # Summary statistics
        if not daily_attr.empty:
            summary = {
                'total_pnl': daily_attr['cumulative_pnl'].iloc[-1] if 'cumulative_pnl' in daily_attr else 0,
                'option_pnl': daily_attr['cumulative_option_pnl'].iloc[-1] if 'cumulative_option_pnl' in daily_attr else 0,
                'hedge_pnl': daily_attr['cumulative_hedge_pnl'].iloc[-1] if 'cumulative_hedge_pnl' in daily_attr else 0,
                'daily_pnl_mean': daily_attr['total_pnl_change'].mean() if 'total_pnl_change' in daily_attr else 0,
                'daily_pnl_std': daily_attr['total_pnl_change'].std() if 'total_pnl_change' in daily_attr else 0,
                'sharpe_ratio': (daily_attr['total_pnl_change'].mean() / daily_attr['total_pnl_change'].std() * np.sqrt(252)) 
                               if 'total_pnl_change' in daily_attr and daily_attr['total_pnl_change'].std() > 0 else 0
            }
        else:
            summary = {}
        
        # Combine all statistics
        summary.update(trade_stats)
        
        # Create report DataFrame
        report = pd.DataFrame([summary])
        
        # Add percentage breakdowns
        if 'total_pnl' in summary and summary['total_pnl'] != 0:
            report['option_pnl_pct'] = summary.get('option_pnl', 0) / summary['total_pnl'] * 100
            report['hedge_pnl_pct'] = summary.get('hedge_pnl', 0) / summary['total_pnl'] * 100
            
            if 'total_transaction_costs' in summary:
                report['transaction_cost_pct'] = summary['total_transaction_costs'] / abs(summary['total_pnl']) * 100
        
        return report
    
    def plot_pnl_attribution(
        self,
        daily_attribution: pd.DataFrame
    ) -> Dict:
        """
        Create data for P&L attribution visualization.
        Returns dict with plot data.
        """
        
        if daily_attribution.empty:
            return {}
        
        plot_data = {
            'timestamps': daily_attribution.index.tolist(),
            'cumulative_pnl': daily_attribution['cumulative_pnl'].tolist() if 'cumulative_pnl' in daily_attribution else [],
            'option_pnl': daily_attribution['cumulative_option_pnl'].tolist() if 'cumulative_option_pnl' in daily_attribution else [],
            'hedge_pnl': daily_attribution['cumulative_hedge_pnl'].tolist() if 'cumulative_hedge_pnl' in daily_attribution else [],
            'daily_pnl': daily_attribution['total_pnl_change'].tolist() if 'total_pnl_change' in daily_attribution else []
        }
        
        return plot_data