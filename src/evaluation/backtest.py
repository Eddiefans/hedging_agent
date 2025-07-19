import os 
import sys
import pandas as pd 
import numpy as np 
from stable_baselines3 import PPO, DDPG

# Add the project root directory to the system path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(project_root)

from src.environment.hedging_env import PortfolioHedgingEnv

def run_backtest(
    model_path = "models/best_model/best_model",
    data_path = "data/processed/NVDA_hedging_features",
    results_path = "results",
    commission: float = 0.00125,
    dead_zone: float = 0.03,
    initial_capital: float = 2_000_000.0,
    window_size: int = 5,
    episode_months: int = 6,
    algorithm = "PPO",
    n_episodes = 10,
    verbose = True
) -> None:
    
    if verbose: 
        print("Reading data from {}".format(data_path))
        
    # Read the data and section by datees, prices and features
    df = pd.read_csv(data_path)
    df['Date'] = pd.to_datetime(df['Date'])
    prices = df['Close'].astype(np.float32).values
    features_columns = [col for col in df.columns if col not in ["Date", "Close"]]
    features = df[features_columns].astype(np.float32).values

    # Check that there are no negative prices
    if np.any(prices <= 0):  
        raise ValueError("Prices contain non-positive values")
    
    # Create environment 
    env = PortfolioHedgingEnv(
        features=features, 
        prices=prices,
        episode_months=episode_months,
        window_size=window_size,
        dead_zone=dead_zone,
        commission=commission,
        initial_capital=initial_capital,
        dates = df['Date'].values
    )
    
    if verbose: 
        print("Loading {} model from {}".format(algorithm, model_path))
    try:
        if algorithm.lower() == "ppo":
            model = PPO.load(model_path)
        else: 
            model = DDPG.load(model_path)
    except Exception as e: 
        raise RuntimeError("Failed to laod {} model from {}".format(algorithm, model_path))
    
    
    episodes_stats = []
    episodes_history = []
    
    for episode in range(n_episodes):
        obs, info = env.reset()
        done = False
        
        while not done: 
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated 
            
        stat = env.get_episode_stats()
        history = env.get_history()
        
        print(f"\nEpisode {episode + 1:2d}:")
        print(f"  Strategy  | Return: {stat['total_return']*100:6.2f}% | Sharpe: {stat['sharpe_ratio']:6.3f} | Sortino: {stat['sortino_ratio']:6.3f} | MaxDD: {stat['max_drawdown']*100:5.2f}% | Vol: {stat['volatility']*100:5.2f}%")
        print(f"  Benchmark | Return: {stat['benchmark_return']*100:6.2f}% | Sharpe: {stat['benchmark_sharpe_ratio']:6.3f} | Sortino: {stat['benchmark_sortino_ratio']:6.3f} | MaxDD: {stat['benchmark_max_drawdown']*100:5.2f}% | Vol: {stat['benchmark_volatility']*100:5.2f}%")
        print(f"  Actions   | Average: L:{stat["avg_l_action"]:.3f} S:{stat["avg_s_action"]:.3f} | Range: L:[{stat["min_l_action"]:.3f}, {stat["max_l_action"]:.3f}] S:[{stat["min_s_action"]:.3f}, {stat["max_s_action"]:.3f}] | Std: L:{stat["action_l_std"]:.3f} S:{stat["action_s_std"]:.3f}")
        
        episodes_stats.append(stat)
        episodes_history.append(history)
    
    stats = pd.DataFrame(episodes_stats)
    periods_per_year = 12 / stats['months'].iloc[-1]
    
    average_return = stats['benchmark_return'].mean()
    closest_episode_idx = (stats['benchmark_return'] - average_return).abs().idxmin()
    episode_history = episodes_history[closest_episode_idx]
    
    

    
    print("\n" + "="*100)
    print("STRATEGY SUMMARY".center(100))
    print("="*100)
    print("\nPERFORMANCE:")
    print(f"  Average Return: {stats['total_return'].mean()*100:.2f}% ± {stats['total_return'].std()*100:.2f}%")
    print(f"  Annualized Average Return: {((1 + stats['total_return'].mean() ) ** periods_per_year - 1)*100:.2f}%")
    print(f"  Best Episode: {stats['total_return'].max()*100:.2f}%")
    print(f"  Worst Episode: {stats['total_return'].min()*100:.2f}%")
    print(f"  Win Rate: {(stats['total_return'] > 0).mean()*100:.1f}%")
    print("\nRISK METRICS:")
    print(f"  Average Sharpe Ratio: {stats['sharpe_ratio'].mean():.3f} ± {stats['sharpe_ratio'].std():.3f}")
    print(f"  Average Sortino Ratio: {stats['sortino_ratio'].mean():.3f} ± {stats['sortino_ratio'].std():.3f}")
    print(f"  Average Max Drawdown: {stats['max_drawdown'].mean()*100:.2f}%")
    print(f"  Average Volatility: {stats['volatility'].mean()*100:.2f}%")
    print("\nTRADING BEHAVIOR:")
    print(f"  Average Action:     L: {stats["avg_l_action"].mean():.3f} S: {stats["avg_s_action"].mean():.3f}")
    print(f"  Action Range:       L: [{stats["min_l_action"].min():.3f}, {stats["max_l_action"].max():.3f}] S: [{stats["min_s_action"].min():.3f}, {stats["max_s_action"].max():.3f}]")
    print(f"  Average Action Std: L: {stats["action_l_std"].std():.3f} S: {stats["action_s_std"].std():.3f}")
    
    print("\n" + "="*100)
    print("BENCHMARK SUMMARY".center(100))
    print("="*100)
    print("\nPERFORMANCE:")
    print(f"  Average Return: {stats['benchmark_return'].mean()*100:.2f}% ± {stats['benchmark_return'].std()*100:.2f}%")
    print(f"  Annualized Average Return: {((1 + stats['benchmark_return'].mean() ) ** periods_per_year - 1)*100:.2f}%")
    print(f"  Best Episode: {stats['benchmark_return'].max()*100:.2f}%")
    print(f"  Worst Episode: {stats['benchmark_return'].min()*100:.2f}%")
    print(f"  Win Rate: {(stats['benchmark_return'] > 0).mean()*100:.1f}%")
    print("\nRISK METRICS:")
    print(f"  Average Sharpe Ratio: {stats['benchmark_sharpe_ratio'].mean():.3f} ± {stats['benchmark_sharpe_ratio'].std():.3f}")
    print(f"  Average Sortino Ratio: {stats['benchmark_sortino_ratio'].mean():.3f} ± {stats['benchmark_sortino_ratio'].std():.3f}")
    print(f"  Average Max Drawdown: {stats['benchmark_max_drawdown'].mean()*100:.2f}%")
    print(f"  Average Volatility: {stats['benchmark_volatility'].mean()*100:.2f}%")
    print("\nTRADING BEHAVIOR:")
    print("  Buy and Hold")
    
    print("\n" + "="*100)
        
    # Create average row for stats
    average_stats = {
        'total_return': stats['total_return'].mean(),
        'months': stats['months'].iloc[0],  # Same for all episodes
        'annualized_return': stats['annualized_return'].mean(),
        'benchmark_return': stats['benchmark_return'].mean(),
        'annualized_benchmark_return': stats['annualized_benchmark_return'].mean(),
        'benchmark_sharpe_ratio': stats['benchmark_sharpe_ratio'].mean(),
        'benchmark_sortino_ratio': stats['benchmark_sortino_ratio'].mean(),
        'benchmark_max_drawdown': stats['benchmark_max_drawdown'].mean(),
        'benchmark_volatility': stats['benchmark_volatility'].mean(),
        'volatility': stats['volatility'].mean(),
        'sharpe_ratio': stats['sharpe_ratio'].mean(),
        'num_trades': stats['num_trades'].mean(),
        'max_drawdown': stats['max_drawdown'].mean(),
        'sortino_ratio': stats['sortino_ratio'].mean(),
        'final_portfolio_value': stats['final_portfolio_value'].mean(),
        'final_cash': stats['final_cash'].mean(),
        'final_long_shares': stats['final_long_shares'].mean(),
        'final_short_shares': stats['final_short_shares'].mean(),
        'avg_l_action': stats['avg_l_action'].mean(),
        'min_l_action': stats['min_l_action'].mean(),
        'max_l_action': stats['max_l_action'].mean(),
        'action_l_std': stats['action_l_std'].mean(),
        'avg_s_action': stats['avg_s_action'].mean(),
        'min_s_action': stats['min_s_action'].mean(),
        'max_s_action': stats['max_s_action'].mean(),
        'action_s_std': stats['action_s_std'].mean()
    }
    
    stats_with_avg = pd.concat([stats, pd.DataFrame([average_stats])], ignore_index=True)
    
    episode_numbers = list(range(1, n_episodes + 1)) + ['AVERAGE']
    stats_with_avg.insert(0, 'episode', episode_numbers)
    
    data_path = os.path.join(results_path, "data")
    os.makedirs(data_path, exist_ok=True)
    
    # Save metrics with average
    metrics_path = os.path.join(data_path, "metrics.csv")
    stats_with_avg.to_csv(metrics_path, index=False)
    if verbose:
        print(f"Metrics saved to: {metrics_path}")
        
    # Save representative episode history
    episode_history_path = os.path.join(data_path, "avg_episode_performance.csv")
    episode_history.to_csv(episode_history_path) 
    if verbose:
        print(f"Representative episode data saved to: {episode_history_path}")
    
            
    
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Backtest hedging model")
    
    parser.add_argument('--model_path', default="models/best_model/best_model", help='Path to model for evaluation')
    parser.add_argument('--data_path', default="data/processed/NVDA_hedging_features.csv", help='Path to processed dataset')
    parser.add_argument("--results_path", default="results", help="Path to results directory")
    parser.add_argument('--algorithm', choices=['PPO', 'DDPG'], default='PPO', help='RL algorithm to use')
    parser.add_argument('--episode_length', type=int, default=6, help='Episode length in months')
    parser.add_argument("--n_episodes", type=int, default=10, help="Number of episodes to backtest with")
    
    args = parser.parse_args()
    
    run_backtest(
        model_path=args.model_path,
        data_path=args.data_path,
        results_path=args.results_path,
        episode_months=args.episode_length,
        n_episodes = args.n_episodes
    )