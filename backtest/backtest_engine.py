import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Callable, Any
from dataclasses import dataclass
from abc import ABC, abstractmethod
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from simulation.gamma_scalping_sim import GammaScalpingSimulator
from simulation.portfolio_dynamics import PortfolioDynamics
from analytics.pnl_decomposition import PnLDecomposition
from models.options_pricing import implied_volatility

@dataclass
class BacktestConfig:
    """Configuration for backtest run"""
    initial_capital: float = 100000
    commission_rate: float = 0.0005
    slippage_bps: float = 10
    risk_free_rate: float = 0.01
    start_date: Optional[pd.Timestamp] = None
    end_date: Optional[pd.Timestamp] = None
    
@dataclass
class StrategyConfig:
    """Configuration for strategy variant"""
    name: str
    hedge_threshold: float = 0.1  # Delta threshold for rebalancing
    hedge_method: str = 'delta_band'  # 'delta_band', 'time_based', 'gamma_scaled'
    rebalance_frequency: str = '1H'  # For time-based hedging
    option_selection: str = 'atm'  # 'atm', 'otm_call', 'otm_put', 'straddle'
    otm_percent: float = 0.05  # For OTM selection (5% OTM)
    position_size: float = 1.0  # Number of option contracts
    max_gamma_exposure: float = 100  # Maximum gamma exposure allowed
    vol_adjustment: bool = False  # Adjust thresholds based on volatility
    
class HedgingStrategy(ABC):
    """Abstract base class for hedging strategies"""
    
    @abstractmethod
    def should_hedge(self, portfolio: PortfolioDynamics, config: StrategyConfig) -> bool:
        """Determine if hedging is needed"""
        pass
    
    @abstractmethod
    def calculate_hedge_size(self, portfolio: PortfolioDynamics, config: StrategyConfig) -> float:
        """Calculate required hedge size"""
        pass

class DeltaBandHedging(HedgingStrategy):
    """Hedge when delta exceeds threshold bands"""
    
    def should_hedge(self, portfolio: PortfolioDynamics, config: StrategyConfig) -> bool:
        greeks = portfolio.get_portfolio_greeks()
        return abs(greeks['delta']) > config.hedge_threshold
    
    def calculate_hedge_size(self, portfolio: PortfolioDynamics, config: StrategyConfig) -> float:
        return portfolio.calculate_hedge_requirement(target_delta=0)

class TimeBasedHedging(HedgingStrategy):
    """Hedge at fixed time intervals"""
    
    def __init__(self):
        self.last_hedge_time = None
    
    def should_hedge(self, portfolio: PortfolioDynamics, config: StrategyConfig) -> bool:
        # Implementation would check time since last hedge
        return True  # Simplified
    
    def calculate_hedge_size(self, portfolio: PortfolioDynamics, config: StrategyConfig) -> float:
        return portfolio.calculate_hedge_requirement(target_delta=0)

class GammaScaledHedging(HedgingStrategy):
    """Hedge with thresholds scaled by gamma exposure"""
    
    def should_hedge(self, portfolio: PortfolioDynamics, config: StrategyConfig) -> bool:
        greeks = portfolio.get_portfolio_greeks()
        gamma_adjusted_threshold = config.hedge_threshold * max(0.5, 1 / (1 + abs(greeks['gamma'])))
        return abs(greeks['delta']) > gamma_adjusted_threshold
    
    def calculate_hedge_size(self, portfolio: PortfolioDynamics, config: StrategyConfig) -> float:
        greeks = portfolio.get_portfolio_greeks()
        # Partial hedge based on gamma
        hedge_ratio = min(1.0, abs(greeks['gamma']) / config.max_gamma_exposure)
        return portfolio.calculate_hedge_requirement(target_delta=0, hedge_ratio=hedge_ratio)

class BacktestEngine:
    """
    Main backtesting engine for gamma scalping strategies.
    Supports multiple strategy variants and comprehensive analysis.
    """
    
    def __init__(self, config: BacktestConfig):
        self.config = config
        self.hedging_strategies = {
            'delta_band': DeltaBandHedging(),
            'time_based': TimeBasedHedging(),
            'gamma_scaled': GammaScaledHedging()
        }
        
    def select_options(
        self,
        spot_price: float,
        available_options: pd.DataFrame,
        config: StrategyConfig
    ) -> pd.DataFrame:
        """Select options based on strategy configuration"""
        
        if available_options.empty:
            return pd.DataFrame()
        
        # Filter by days to expiry (prefer 7-30 DTE)
        available_options['dte'] = (available_options['expiry'] - pd.Timestamp.now()).dt.days
        options = available_options[(available_options['dte'] >= 7) & (available_options['dte'] <= 30)]
        
        if options.empty:
            options = available_options
        
        selected = pd.DataFrame()
        
        if config.option_selection == 'atm':
            # Select ATM option
            options['moneyness'] = abs(options['strike'] - spot_price) / spot_price
            selected = options.nsmallest(1, 'moneyness')
            
        elif config.option_selection == 'otm_call':
            # Select OTM call
            target_strike = spot_price * (1 + config.otm_percent)
            calls = options[options['option_type'] == 'call']
            calls['distance'] = abs(calls['strike'] - target_strike)
            selected = calls.nsmallest(1, 'distance')
            
        elif config.option_selection == 'otm_put':
            # Select OTM put
            target_strike = spot_price * (1 - config.otm_percent)
            puts = options[options['option_type'] == 'put']
            puts['distance'] = abs(puts['strike'] - target_strike)
            selected = puts.nsmallest(1, 'distance')
            
        elif config.option_selection == 'straddle':
            # Select ATM straddle (both call and put)
            options['moneyness'] = abs(options['strike'] - spot_price) / spot_price
            atm_strike = options.nsmallest(1, 'moneyness')['strike'].iloc[0]
            selected = options[options['strike'] == atm_strike]
        
        return selected
    
    def run_backtest(
        self,
        spot_data: pd.DataFrame,
        options_data: pd.DataFrame,
        strategy_config: StrategyConfig
    ) -> Dict[str, Any]:
        """
        Run backtest for a specific strategy configuration.
        
        Args:
            spot_data: DataFrame with columns ['timestamp', 'close']
            options_data: DataFrame with columns ['timestamp', 'strike', 'expiry', 
                         'option_type', 'close', 'volume', 'open_interest']
            strategy_config: Strategy configuration
        
        Returns:
            Dictionary with backtest results
        """
        
        # Initialize simulator and portfolio
        simulator = GammaScalpingSimulator(
            initial_capital=self.config.initial_capital,
            commission_rate=self.config.commission_rate,
            slippage_bps=self.config.slippage_bps,
            risk_free_rate=self.config.risk_free_rate
        )
        
        portfolio = PortfolioDynamics(risk_free_rate=self.config.risk_free_rate)
        pnl_decomposer = PnLDecomposition()
        
        # Get hedging strategy
        hedge_strategy = self.hedging_strategies[strategy_config.hedge_method]
        
        # Filter data by date range
        if self.config.start_date:
            spot_data = spot_data[spot_data['timestamp'] >= self.config.start_date]
            options_data = options_data[options_data['timestamp'] >= self.config.start_date]
        if self.config.end_date:
            spot_data = spot_data[spot_data['timestamp'] <= self.config.end_date]
            options_data = options_data[options_data['timestamp'] <= self.config.end_date]
        
        # Main backtest loop
        position_opened = False
        current_options = []
        
        for idx, spot_row in spot_data.iterrows():
            timestamp = spot_row['timestamp']
            spot_price = spot_row['close']
            
            # Get available options at this timestamp
            current_option_data = options_data[options_data['timestamp'] == timestamp]
            
            # Open position if not already opened
            if not position_opened and not current_option_data.empty:
                selected_options = self.select_options(spot_price, current_option_data, strategy_config)
                
                for _, opt in selected_options.iterrows():
                    # Calculate implied volatility
                    time_to_expiry = (opt['expiry'] - timestamp).total_seconds() / (365 * 24 * 3600)
                    if time_to_expiry > 0:
                        # Option price is already in USD (converted by data loader)
                        option_price_usd = opt['close']
                        
                        iv = implied_volatility(
                            option_price_usd, spot_price, opt['strike'],
                            time_to_expiry, self.config.risk_free_rate, opt['option_type']
                        )
                        
                        if not np.isnan(iv):
                            # Open option position
                            portfolio.add_option_position(
                                symbol='BTC',
                                strike=opt['strike'],
                                expiry=opt['expiry'],
                                option_type=opt['option_type'],
                                quantity=strategy_config.position_size,
                                price=option_price_usd,
                                spot=spot_price,
                                iv=iv,
                                timestamp=timestamp
                            )
                            current_options.append(opt)
                            position_opened = True
            
            # Update portfolio with current market data
            if position_opened:
                # Get current option prices and IVs
                option_prices = {}
                option_ivs = {}
                
                for opt in current_options:
                    opt_data = current_option_data[
                        (current_option_data['strike'] == opt['strike']) &
                        (current_option_data['expiry'] == opt['expiry']) &
                        (current_option_data['option_type'] == opt['option_type'])
                    ]
                    
                    if not opt_data.empty:
                        position_id = f"BTC_{opt['strike']}_{opt['expiry'].strftime('%Y%m%d')}_{opt['option_type']}"
                        
                        # Option price is already in USD (converted by data loader)
                        option_price_usd = opt_data['close'].iloc[0]
                        option_prices[position_id] = option_price_usd
                        
                        # Calculate IV
                        time_to_expiry = (opt['expiry'] - timestamp).total_seconds() / (365 * 24 * 3600)
                        if time_to_expiry > 0:
                            iv = implied_volatility(
                                option_price_usd, spot_price, opt['strike'],
                                time_to_expiry, self.config.risk_free_rate, opt['option_type']
                            )
                            if not np.isnan(iv):
                                option_ivs[position_id] = iv
                
                # Update portfolio
                portfolio.update_market_data(
                    timestamp=timestamp,
                    spot_prices={'BTC': spot_price},
                    option_ivs=option_ivs,
                    option_prices=option_prices
                )
                
                # Check if hedging is needed
                if hedge_strategy.should_hedge(portfolio, strategy_config):
                    hedge_size = hedge_strategy.calculate_hedge_size(portfolio, strategy_config)
                    
                    if abs(hedge_size) > 0.001:  # Minimum hedge size
                        portfolio.add_stock_hedge('BTC', hedge_size, spot_price)
        
        # Generate results
        portfolio_history = portfolio.get_history_dataframe()
        
        # Calculate performance metrics
        if not portfolio_history.empty:
            final_pnl = portfolio.get_pnl_breakdown()
            greeks_history = portfolio.get_portfolio_greeks()
            
            # P&L decomposition
            pnl_report = pnl_decomposer.generate_attribution_report(
                portfolio_history,
                pd.DataFrame()  # Would pass actual trades here
            )
            
            # Calculate key metrics
            total_return = final_pnl['total_pnl'] / self.config.initial_capital
            
            if 'total_pnl' in portfolio_history.columns:
                returns = portfolio_history['total_pnl'].diff() / self.config.initial_capital
                sharpe = np.sqrt(252) * returns.mean() / returns.std() if returns.std() > 0 else 0
                
                # Fix max drawdown calculation - use percentage not absolute
                cumulative_value = self.config.initial_capital + portfolio_history['total_pnl']
                running_max = cumulative_value.expanding().max()
                drawdown = (cumulative_value - running_max) / running_max
                max_dd = abs(drawdown.min()) if len(drawdown) > 0 else 0
            else:
                sharpe = 0
                max_dd = 0
            
            results = {
                'strategy_name': strategy_config.name,
                'total_return': total_return,
                'sharpe_ratio': sharpe,
                'max_drawdown': max_dd,
                'final_pnl': final_pnl,
                'portfolio_history': portfolio_history,
                'pnl_attribution': pnl_report,
                'config': strategy_config
            }
        else:
            results = {
                'strategy_name': strategy_config.name,
                'total_return': 0,
                'sharpe_ratio': 0,
                'max_drawdown': 0,
                'final_pnl': {},
                'portfolio_history': pd.DataFrame(),
                'pnl_attribution': pd.DataFrame(),
                'config': strategy_config
            }
        
        return results
    
    def run_multiple_strategies(
        self,
        spot_data: pd.DataFrame,
        options_data: pd.DataFrame,
        strategy_configs: List[StrategyConfig]
    ) -> pd.DataFrame:
        """Run backtest for multiple strategy configurations"""
        
        results = []
        
        for config in strategy_configs:
            print(f"Running backtest for strategy: {config.name}")
            result = self.run_backtest(spot_data, options_data, config)
            
            # Extract key metrics
            summary = {
                'strategy': config.name,
                'hedge_method': config.hedge_method,
                'hedge_threshold': config.hedge_threshold,
                'option_selection': config.option_selection,
                'total_return': result['total_return'],
                'sharpe_ratio': result['sharpe_ratio'],
                'max_drawdown': result['max_drawdown']
            }
            
            # Add P&L breakdown if available
            if 'final_pnl' in result and result['final_pnl']:
                summary.update({
                    'option_pnl': result['final_pnl'].get('option_pnl', 0),
                    'hedge_pnl': result['final_pnl'].get('stock_pnl', 0)
                })
            
            results.append(summary)
        
        return pd.DataFrame(results)
    
    def parameter_optimization(
        self,
        spot_data: pd.DataFrame,
        options_data: pd.DataFrame,
        base_config: StrategyConfig,
        param_grid: Dict[str, List[Any]]
    ) -> pd.DataFrame:
        """
        Optimize strategy parameters using grid search.
        
        Args:
            param_grid: Dictionary of parameters to optimize
                       e.g., {'hedge_threshold': [0.05, 0.1, 0.15]}
        """
        
        results = []
        
        # Generate all parameter combinations
        import itertools
        param_names = list(param_grid.keys())
        param_values = list(param_grid.values())
        
        for values in itertools.product(*param_values):
            # Create config with current parameters
            config = StrategyConfig(
                name=f"opt_{'-'.join(map(str, values))}",
                hedge_threshold=base_config.hedge_threshold,
                hedge_method=base_config.hedge_method,
                option_selection=base_config.option_selection,
                position_size=base_config.position_size
            )
            
            # Update with optimization parameters
            for name, value in zip(param_names, values):
                setattr(config, name, value)
            
            # Run backtest
            result = self.run_backtest(spot_data, options_data, config)
            
            # Store results
            param_result = {param: value for param, value in zip(param_names, values)}
            param_result.update({
                'total_return': result['total_return'],
                'sharpe_ratio': result['sharpe_ratio'],
                'max_drawdown': result['max_drawdown']
            })
            results.append(param_result)
        
        return pd.DataFrame(results)