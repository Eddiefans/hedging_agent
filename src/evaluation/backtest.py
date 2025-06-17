import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from stable_baselines3 import PPO, SAC
from datetime import datetime, timedelta

# Add the project root directory to the system path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(project_root)

# Import the environment
from src.environment.hedging_env import PortfolioHedgingEnv

def run_hedging_backtest(
    model_path="models/best_model/best_model",
    data_path="data/processed/NVDA_hedging_features.csv",
    results_dir="results",
    episode_length_months=6,
    window_size=5,
    dead_zone=0.09,
    initial_portfolio=1_000_000,
    commission=0.00125,
    test_start_date=None,
    test_end_date=None,
    n_test_episodes=50,
    verbose=True,
    save=True
):
    """
    Run a comprehensive backtest of the hedging strategy.
    
    Args:
        model_path: Path to the trained model
        data_path: Path to the processed dataset
        results_dir: Directory to save results
        episode_length_months: Length of each episode
        window_size: Observation window size
        dead_zone: Dead zone for no-action
        initial_portfolio: Initial portfolio value
        commission: Commission rate
        test_start_date: Start date for testing (if None, uses last 20% of data)
        test_end_date: End date for testing
        n_test_episodes: Number of random episodes to test
        verbose: Whether to print detailed information
        save: Whether to save results
    
    Returns:
        dict: Comprehensive performance metrics
    """
    # Create results directories
    figures_dir = os.path.join(results_dir, "figures")
    metrics_dir = os.path.join(results_dir, "metrics")
    os.makedirs(figures_dir, exist_ok=True)
    os.makedirs(metrics_dir, exist_ok=True)
    
    # Load the dataset
    if verbose:
        print("Loading dataset from {}...".format(data_path))
    df = pd.read_csv(data_path)
    df['Date'] = pd.to_datetime(df['Date'])
    
    # Determine test period
    if test_start_date is None:
        # Use last 20% of data for testing
        split_idx = int(len(df) * 0.8)
        test_start_date = df.iloc[split_idx]['Date']
    
    if test_end_date is None:
        test_end_date = df['Date'].max()
    
    # Filter test data
    test_mask = (df['Date'] >= test_start_date) & (df['Date'] <= test_end_date)
    test_df = df[test_mask].reset_index(drop=True)
    
    if verbose:
        print(f"Test period: {test_start_date} to {test_end_date}")
        print(f"Test data points: {len(test_df)}")
    
    # Extract test data
    dates = test_df['Date'].values
    prices = test_df['Close'].astype(np.float32).values
    feature_columns = [col for col in test_df.columns if col not in ['Date', 'Close']]
    features = test_df[feature_columns].astype(np.float32).values
    
    # Load the trained model
    if verbose:
        print("Loading model from {}...".format(model_path))
    
    try:
        model = SAC.load(model_path)
        model_type = "SAC"
    except:
        try:
            model = PPO.load(model_path)
            model_type = "PPO"
        except Exception as e:
            raise ValueError(f"Could not load model: {e}")
    
    if verbose:
        print(f"Loaded {model_type} model")
    
    # Create test environment
    env = PortfolioHedgingEnv(
        features=features,
        prices=prices,
        dates=dates,
        episode_length_months=episode_length_months,
        window_size=window_size,
        dead_zone=dead_zone,
        portfolio_value=initial_portfolio,
        commission=commission
    )
    
    # Run multiple episodes for robust evaluation
    episode_results = []
    all_portfolio_histories = []
    all_position_histories = []
    all_dates_histories = []
    
    if verbose:
        print(f"Running {n_test_episodes} test episodes...")
    
    for episode in range(n_test_episodes):
        obs = env.reset()
        done = False
        step_count = 0
        episode_actions = []
        
        # Store episode start info
        episode_start_idx = env.current_episode_start
        episode_start_date = dates[episode_start_idx]
        
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)
            episode_actions.append(action[0])
            step_count += 1
        
        # Get episode statistics
        stats = env.get_episode_stats()
        stats.update({
            'episode': episode,
            'start_date': episode_start_date,
            'start_idx': episode_start_idx,
            'steps': step_count,
            'avg_action': np.mean(episode_actions),
            'action_std': np.std(episode_actions)
        })
        
        episode_results.append(stats)
        
        # Store histories for detailed analysis
        if episode < 5:  # Store first 5 episodes for plotting
            episode_dates = dates[episode_start_idx:episode_start_idx + len(env.portfolio_history)]
            all_portfolio_histories.append(env.portfolio_history.copy())
            all_position_histories.append(env.position_history.copy())
            all_dates_histories.append(episode_dates)
        
        if verbose and (episode + 1) % 10 == 0:
            print(f"Completed {episode + 1}/{n_test_episodes} episodes")
    
    # Aggregate results
    results_df = pd.DataFrame(episode_results)
    
    # Calculate benchmark (buy and hold)
    benchmark_returns = []
    for _, row in results_df.iterrows():
        start_idx = row['start_idx']
        end_idx = min(start_idx + episode_length_months * 21, len(prices) - 1)
        start_price = prices[start_idx]
        end_price = prices[end_idx]
        benchmark_return = (end_price - start_price) / start_price
        benchmark_returns.append(benchmark_return)
    
    results_df['benchmark_return'] = benchmark_returns
    results_df['excess_return'] = results_df['total_return'] - results_df['benchmark_return']
    
    # Performance metrics
    performance_metrics = {
        'model_type': model_type,
        'test_episodes': n_test_episodes,
        'test_period': f"{test_start_date} to {test_end_date}",
        
        # Returns
        'avg_return': results_df['total_return'].mean(),
        'std_return': results_df['total_return'].std(),
        'median_return': results_df['total_return'].median(),
        'min_return': results_df['total_return'].min(),
        'max_return': results_df['total_return'].max(),
        
        # Risk metrics
        'avg_sharpe': results_df['sharpe_ratio'].mean(),
        'avg_max_drawdown': results_df['max_drawdown'].mean(),
        'avg_volatility': results_df['volatility'].mean(),
        
        # Benchmark comparison
        'avg_benchmark_return': results_df['benchmark_return'].mean(),
        'avg_excess_return': results_df['excess_return'].mean(),
        'win_rate_vs_benchmark': (results_df['excess_return'] > 0).mean(),
        
        # Trading behavior
        'avg_final_position': results_df['final_position'].mean(),
        'avg_num_trades': results_df['num_trades'].mean(),
        'position_std': results_df['final_position'].std(),
        
        # Success rates
        'positive_return_rate': (results_df['total_return'] > 0).mean(),
        'sharpe_above_1_rate': (results_df['sharpe_ratio'] > 1.0).mean(),
    }
    
    # Print results
    if verbose:
        print("\n" + "="*60)
        print("HEDGING STRATEGY BACKTEST RESULTS")
        print("="*60)
        print(f"Model: {model_type}")
        print(f"Test Period: {test_start_date.strftime('%Y-%m-%d')} to {test_end_date.strftime('%Y-%m-%d')}")
        print(f"Episodes: {n_test_episodes}")
        print(f"Episode Length: {episode_length_months} months")
        print()
        print("RETURNS:")
        print(f"  Average Return: {performance_metrics['avg_return']*100:.2f}%")
        print(f"  Median Return: {performance_metrics['median_return']*100:.2f}%")
        print(f"  Return Std: {performance_metrics['std_return']*100:.2f}%")
        print(f"  Best Episode: {performance_metrics['max_return']*100:.2f}%")
        print(f"  Worst Episode: {performance_metrics['min_return']*100:.2f}%")
        print()
        print("RISK METRICS:")
        print(f"  Average Sharpe Ratio: {performance_metrics['avg_sharpe']:.3f}")
        print(f"  Average Max Drawdown: {performance_metrics['avg_max_drawdown']*100:.2f}%")
        print(f"  Average Volatility: {performance_metrics['avg_volatility']*100:.2f}%")
        print()
        print("BENCHMARK COMPARISON:")
        print(f"  Benchmark Avg Return: {performance_metrics['avg_benchmark_return']*100:.2f}%")
        print(f"  Average Excess Return: {performance_metrics['avg_excess_return']*100:.2f}%")
        print(f"  Win Rate vs Benchmark: {performance_metrics['win_rate_vs_benchmark']*100:.1f}%")
        print()
        print("TRADING BEHAVIOR:")
        print(f"  Average Final Position: {performance_metrics['avg_final_position']:.3f}")
        print(f"  Average Trades per Episode: {performance_metrics['avg_num_trades']:.1f}")
        print(f"  Position Std Dev: {performance_metrics['position_std']:.3f}")
        print()
        print("SUCCESS RATES:")
        print(f"  Positive Return Rate: {performance_metrics['positive_return_rate']*100:.1f}%")
        print(f"  Sharpe > 1.0 Rate: {performance_metrics['sharpe_above_1_rate']*100:.1f}%")
        print("="*60)
    
    if save:
        # Create comprehensive plots
        fig, axes = plt.subplots(2, 3, figsize=(20, 12))
        fig.suptitle(f'{model_type} Hedging Strategy Backtest Results', fontsize=16)
        
        # 1. Return distribution
        axes[0, 0].hist(results_df['total_return'] * 100, bins=20, alpha=0.7, edgecolor='black')
        axes[0, 0].axvline(results_df['total_return'].mean() * 100, color='red', linestyle='--', label='Mean')
        axes[0, 0].axvline(results_df['benchmark_return'].mean() * 100, color='blue', linestyle='--', label='Benchmark')
        axes[0, 0].set_xlabel('Episode Return (%)')
        axes[0, 0].set_ylabel('Frequency')
        axes[0, 0].set_title('Distribution of Episode Returns')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. Sharpe ratio distribution
        axes[0, 1].hist(results_df['sharpe_ratio'], bins=20, alpha=0.7, edgecolor='black')
        axes[0, 1].axvline(results_df['sharpe_ratio'].mean(), color='red', linestyle='--', label='Mean')
        axes[0, 1].set_xlabel('Sharpe Ratio')
        axes[0, 1].set_ylabel('Frequency')
        axes[0, 1].set_title('Distribution of Sharpe Ratios')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. Excess returns vs benchmark
        axes[0, 2].scatter(results_df['benchmark_return'] * 100, results_df['total_return'] * 100, alpha=0.6)
        axes[0, 2].plot([-10, 15], [-10, 15], 'r--', label='Equal Performance')
        axes[0, 2].set_xlabel('Benchmark Return (%)')
        axes[0, 2].set_ylabel('Strategy Return (%)')
        axes[0, 2].set_title('Strategy vs Benchmark Returns')
        axes[0, 2].legend()
        axes[0, 2].grid(True, alpha=0.3)
        
        # 4. Position distribution
        axes[1, 0].hist(results_df['final_position'], bins=20, alpha=0.7, edgecolor='black')
        axes[1, 0].axvline(1.0, color='red', linestyle='--', label='Neutral')
        axes[1, 0].axvline(results_df['final_position'].mean(), color='orange', linestyle='--', label='Mean')
        axes[1, 0].set_xlabel('Final Position')
        axes[1, 0].set_ylabel('Frequency')
        axes[1, 0].set_title('Distribution of Final Positions')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # 5. Sample episode portfolio evolution
        if all_portfolio_histories:
            for i, (portfolio_hist, dates_hist) in enumerate(zip(all_portfolio_histories[:3], all_dates_histories[:3])):
                normalized_portfolio = np.array(portfolio_hist) / portfolio_hist[0] * 100
                axes[1, 1].plot(dates_hist[:len(normalized_portfolio)], normalized_portfolio, 
                              label=f'Episode {i+1}', alpha=0.7)
            axes[1, 1].set_xlabel('Date')
            axes[1, 1].set_ylabel('Portfolio Value (Normalized to 100)')
            axes[1, 1].set_title('Sample Episode Portfolio Evolution')
            axes[1, 1].legend()
            axes[1, 1].grid(True, alpha=0.3)
            axes[1, 1].tick_params(axis='x', rotation=45)
        
        # 6. Max drawdown vs return
        axes[1, 2].scatter(results_df['max_drawdown'] * 100, results_df['total_return'] * 100, alpha=0.6)
        axes[1, 2].set_xlabel('Max Drawdown (%)')
        axes[1, 2].set_ylabel('Total Return (%)')
        axes[1, 2].set_title('Risk-Return Profile')
        axes[1, 2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save plot
        figure_path = os.path.join(figures_dir, "hedging_backtest_results.png")
        plt.savefig(figure_path, dpi=300, bbox_inches='tight')
        if verbose:
            print(f"Saved comprehensive results plot to {figure_path}")
        plt.show()
        
        # Save detailed results
        results_path = os.path.join(metrics_dir, "episode_results.csv")
        results_df.to_csv(results_path, index=False)
        
        metrics_path = os.path.join(metrics_dir, "performance_metrics.csv")
        pd.DataFrame([performance_metrics]).to_csv(metrics_path, index=False)
        
        if verbose:
            print(f"Saved episode results to {results_path}")
            print(f"Saved performance metrics to {metrics_path}")
    
    return performance_metrics, results_df

def compare_models(model_paths, data_path, **kwargs):
    """Compare multiple trained models."""
    all_results = {}
    
    for name, path in model_paths.items():
        print(f"\nEvaluating {name}...")
        metrics, episodes_df = run_hedging_backtest(
            model_path=path,
            data_path=data_path,
            verbose=False,
            save=False,
            **kwargs
        )
        all_results[name] = metrics
    
    # Create comparison DataFrame
    comparison_df = pd.DataFrame(all_results).T
    
    print("\n" + "="*80)
    print("MODEL COMPARISON")
    print("="*80)
    print(comparison_df[['avg_return', 'avg_sharpe', 'avg_max_drawdown', 
                        'win_rate_vs_benchmark', 'positive_return_rate']].round(4))
    
    return comparison_df

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Backtest hedging strategy')
    parser.add_argument('--model_path', default="models/best_model/best_model",
                       help='Path to trained model')
    parser.add_argument('--data_path', default="data/processed/NVDA_hedging_features.csv",
                       help='Path to processed dataset')
    parser.add_argument('--n_episodes', type=int, default=50,
                       help='Number of test episodes')
    parser.add_argument('--episode_months', type=int, default=6,
                       help='Episode length in months')
    
    args = parser.parse_args()
    
    # Run backtest
    performance_metrics, episode_results = run_hedging_backtest(
        model_path=args.model_path,
        data_path=args.data_path,
        n_test_episodes=args.n_episodes,
        episode_length_months=args.episode_months
    )