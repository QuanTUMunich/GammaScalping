"""
Main gamma scalping strategy implementation and runner.
This module ties together all components for research and analysis.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.data_loader import DataLoader
from backtest.backtest_engine import BacktestEngine, BacktestConfig, StrategyConfig
from analytics.pnl_decomposition import PnLDecomposition
from simulation.portfolio_dynamics import PortfolioDynamics

def create_strategy_variants():
    """Create different strategy configurations to test"""
    
    strategies = [
        # Delta band strategies with different thresholds
        StrategyConfig(
            name="Conservative Delta Band",
            hedge_threshold=0.15,
            hedge_method='delta_band',
            option_selection='atm',
            position_size=1.0
        ),
        StrategyConfig(
            name="Moderate Delta Band",
            hedge_threshold=0.10,
            hedge_method='delta_band',
            option_selection='atm',
            position_size=1.0
        ),
        StrategyConfig(
            name="Aggressive Delta Band",
            hedge_threshold=0.05,
            hedge_method='delta_band',
            option_selection='atm',
            position_size=1.0
        ),
        
        # Time-based hedging
        StrategyConfig(
            name="Hourly Rebalance",
            hedge_threshold=0.10,
            hedge_method='time_based',
            rebalance_frequency='1H',
            option_selection='atm',
            position_size=1.0
        ),
        
        # Gamma-scaled hedging
        StrategyConfig(
            name="Gamma Scaled",
            hedge_threshold=0.10,
            hedge_method='gamma_scaled',
            option_selection='atm',
            position_size=1.0,
            max_gamma_exposure=50
        ),
        
        # Different option selections
        StrategyConfig(
            name="OTM Call Strategy",
            hedge_threshold=0.10,
            hedge_method='delta_band',
            option_selection='otm_call',
            otm_percent=0.05,
            position_size=1.0
        ),
        StrategyConfig(
            name="OTM Put Strategy",
            hedge_threshold=0.10,
            hedge_method='delta_band',
            option_selection='otm_put',
            otm_percent=0.05,
            position_size=1.0
        ),
        StrategyConfig(
            name="Straddle Strategy",
            hedge_threshold=0.10,
            hedge_method='delta_band',
            option_selection='straddle',
            position_size=0.5  # Half size for each leg
        )
    ]
    
    return strategies

def run_gamma_scalping_research():
    """Main function to run gamma scalping research"""
    
    print("=" * 60)
    print("GAMMA SCALPING STRATEGY RESEARCH")
    print("=" * 60)
    
    # Initialize data loader
    data_loader = DataLoader()
    
    # Get data summary
    print("\nChecking available data...")
    summary = data_loader.get_data_summary()
    print(f"Found {len(summary)} data files")
    print(f"- Spot data files: {len(summary[summary['type'] == 'spot'])}")
    print(f"- Option data files: {len(summary[summary['type'] == 'option'])}")
    
    # Load synchronized data
    print("\nLoading synchronized spot and options data...")
    
    # Use a specific date range for testing (we have data from 2019-2025)
    start_date = pd.Timestamp('2020-04-01')
    end_date = pd.Timestamp('2020-04-30')
    
    spot_data, options_data = data_loader.create_synchronized_dataset(
        spot_symbol='BTCUSDT',
        underlying='BTC',
        start_date=start_date,
        end_date=end_date,
        option_filters={'min_volume': 10}
    )
    
    if spot_data.empty or options_data.empty:
        print("Warning: No synchronized data found for the specified period")
        print("Attempting to load any available data...")
        
        # Try without date restrictions
        spot_data, options_data = data_loader.create_synchronized_dataset(
            spot_symbol='BTCUSDT',
            underlying='BTC',
            option_filters={'min_volume': 0}
        )
    
    if not spot_data.empty and not options_data.empty:
        print(f"Loaded {len(spot_data)} spot price points")
        print(f"Loaded {len(options_data)} option data points")
        print(f"Date range: {spot_data['timestamp'].min()} to {spot_data['timestamp'].max()}")
        
        # Initialize backtest engine
        backtest_config = BacktestConfig(
            initial_capital=100000,
            commission_rate=0.0005,
            slippage_bps=10,
            risk_free_rate=0.01
        )
        
        engine = BacktestEngine(backtest_config)
        
        # Create strategy variants
        strategies = create_strategy_variants()
        
        # Run backtests
        print(f"\nRunning backtests for {len(strategies)} strategy variants...")
        results = engine.run_multiple_strategies(spot_data, options_data, strategies)
        
        # Display results
        print("\n" + "=" * 60)
        print("BACKTEST RESULTS")
        print("=" * 60)
        
        if not results.empty:
            # Sort by Sharpe ratio
            results = results.sort_values('sharpe_ratio', ascending=False)
            
            # Display summary
            print("\nStrategy Performance Summary:")
            print("-" * 60)
            
            for idx, row in results.iterrows():
                print(f"\n{row['strategy']}:")
                print(f"  Total Return: {row['total_return']:.2%}")
                print(f"  Sharpe Ratio: {row['sharpe_ratio']:.2f}")
                print(f"  Max Drawdown: {row['max_drawdown']:.2%}")
                
                if 'option_pnl' in row and 'hedge_pnl' in row:
                    print(f"  Option P&L: ${row['option_pnl']:.2f}")
                    print(f"  Hedge P&L: ${row['hedge_pnl']:.2f}")
            
            # Best performing strategy
            best = results.iloc[0]
            print("\n" + "=" * 60)
            print("BEST PERFORMING STRATEGY")
            print("=" * 60)
            print(f"Strategy: {best['strategy']}")
            print(f"Configuration:")
            print(f"  - Hedge Method: {best['hedge_method']}")
            print(f"  - Hedge Threshold: {best['hedge_threshold']}")
            print(f"  - Option Selection: {best['option_selection']}")
            print(f"Performance:")
            print(f"  - Total Return: {best['total_return']:.2%}")
            print(f"  - Sharpe Ratio: {best['sharpe_ratio']:.2f}")
            print(f"  - Max Drawdown: {best['max_drawdown']:.2%}")
            
            # Save results
            output_dir = Path('backtest/results')
            output_dir.mkdir(parents=True, exist_ok=True)
            
            results_file = output_dir / 'gamma_scalping_results.csv'
            results.to_csv(results_file, index=False)
            print(f"\nResults saved to: {results_file}")
            
            # Parameter optimization example
            print("\n" + "=" * 60)
            print("PARAMETER OPTIMIZATION")
            print("=" * 60)
            
            # Optimize delta threshold for best strategy
            base_config = StrategyConfig(
                name="Optimization",
                hedge_method='delta_band',
                option_selection='atm',
                position_size=1.0
            )
            
            param_grid = {
                'hedge_threshold': [0.02, 0.05, 0.08, 0.10, 0.12, 0.15, 0.20]
            }
            
            print("Optimizing hedge threshold...")
            opt_results = engine.parameter_optimization(
                spot_data, options_data, base_config, param_grid
            )
            
            if not opt_results.empty:
                opt_results = opt_results.sort_values('sharpe_ratio', ascending=False)
                
                print("\nOptimal Parameters:")
                print("-" * 40)
                best_params = opt_results.iloc[0]
                print(f"Best Hedge Threshold: {best_params['hedge_threshold']:.2f}")
                print(f"Sharpe Ratio: {best_params['sharpe_ratio']:.2f}")
                print(f"Total Return: {best_params['total_return']:.2%}")
                
                # Save optimization results
                opt_file = output_dir / 'optimization_results.csv'
                opt_results.to_csv(opt_file, index=False)
                print(f"\nOptimization results saved to: {opt_file}")
        
    else:
        print("\nError: Unable to load data for backtesting")
        print("Please ensure data files are present in data/parsed/")
    
    print("\n" + "=" * 60)
    print("RESEARCH COMPLETE")
    print("=" * 60)

def analyze_single_backtest(spot_data, options_data, strategy_config):
    """Detailed analysis of a single strategy backtest"""
    
    # Initialize components
    backtest_config = BacktestConfig(
        initial_capital=100000,
        commission_rate=0.0005,
        slippage_bps=10,
        risk_free_rate=0.01
    )
    
    engine = BacktestEngine(backtest_config)
    pnl_decomposer = PnLDecomposition()
    
    # Run backtest
    result = engine.run_backtest(spot_data, options_data, strategy_config)
    
    # Analyze P&L attribution
    if 'portfolio_history' in result and not result['portfolio_history'].empty:
        portfolio_history = result['portfolio_history']
        
        # Calculate daily attribution
        daily_attr = pnl_decomposer.calculate_daily_attribution(portfolio_history)
        
        # Generate detailed report
        print(f"\nDetailed Analysis: {strategy_config.name}")
        print("-" * 60)
        
        if not daily_attr.empty:
            print("\nP&L Attribution:")
            if 'cumulative_option_pnl' in daily_attr:
                print(f"  Option P&L: ${daily_attr['cumulative_option_pnl'].iloc[-1]:.2f}")
            if 'cumulative_hedge_pnl' in daily_attr:
                print(f"  Hedge P&L: ${daily_attr['cumulative_hedge_pnl'].iloc[-1]:.2f}")
            
            print("\nRisk Metrics:")
            if 'delta' in portfolio_history.columns:
                print(f"  Avg Delta Exposure: {portfolio_history['delta'].mean():.3f}")
            if 'gamma' in portfolio_history.columns:
                print(f"  Avg Gamma Exposure: {portfolio_history['gamma'].mean():.3f}")
            
            # Calculate win rate
            if 'total_pnl_change' in daily_attr:
                daily_returns = daily_attr['total_pnl_change'].dropna()
                win_rate = (daily_returns > 0).mean()
                print(f"  Daily Win Rate: {win_rate:.1%}")
                print(f"  Avg Win: ${daily_returns[daily_returns > 0].mean():.2f}")
                print(f"  Avg Loss: ${daily_returns[daily_returns < 0].mean():.2f}")
    
    return result

if __name__ == "__main__":
    run_gamma_scalping_research()