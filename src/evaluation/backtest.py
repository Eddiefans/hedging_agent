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
    restuls_path = "results",
    commission: float = 0.00125,
    dead_zone: float = 0.03,
    initial_capital: float = 2_000_000.0,
    window_size: int = 5,
    episode_months: int = 6,
    algorithm = "PPO",
    n_episodes = 10,
    verbose = False
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
        initial_capital=initial_capital
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
    for episode in range(n_episodes):
        obs, info = env.reset()
        done = False
        
        if verbose: 
            print("\nEpisode {}".format(episode + 1))
        
        while not done: 
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated 
        
        episodes_stats.append(env.get_episode_stats())
    
    stats = pd.DataFrame(episodes_stats)
    
    print("\n" + "="*80)
    print("STRATEGY SUMMARY")
    print("="*80)
    print("PERFORMANCE:")
    print(f"  Annualized Average Return: {stats['annualized_return'].mean()*100:.2f}% ± {stats['annualized_return'].std()*100:.2f}%")
    print(f"  Average Return: {stats['total_return'].mean()*100:.2f}% ± {stats['total_return'].std()*100:.2f}%")
    print(f"  Best Episode: {stats['total_return'].max()*100:.2f}%")
    print(f"  Worst Episode: {stats['total_return'].min()*100:.2f}%")
    print(f"  Win Rate: {(stats['total_return'] > 0).mean()*100:.1f}%")
    print("RISK METRICS:")
    print(f"  Average Sharpe Ratio: {stats['sharpe_ratio'].mean():.3f} ± {stats['sharpe_ratio'].std():.3f}")
    print(f"  Average Sortino Ratio: {stats['sortino_ratio'].mean():.3f} ± {stats['sortino_ratio'].std():.3f}")
    print(f"  Average Max Drawdown: {stats['max_drawdown'].mean():.3f} ± {stats['max_drawdown'].std():.3f}")
    print(f"  Strategy Volatility: {stats['volatility'].mean()*100:.2f}%")
    print("TRADING BEHAVIOR:")
    # print(f"  Average Action: {stats['avg_action'].mean():.3f}")
    # print(f"  Action Range: [{stats['min_action'].min():.3f}, {stats['max_action'].max():.3f}]")
    # print(f"  Average Action Std: {stats['action_std'].mean():.3f}")
    # print(f"  Average Reward: {stats['total_reward'].mean():.2f} ± {stats['total_reward'].std():.2f}")
    
    print("\n" + "="*80)
    print("BENCHMARK SUMMARY")
    print("="*80)
    print("PERFORMANCE:")
    print(f"  Annualized Average Return: {stats['annualized_benchmark_return'].mean()*100:.2f}% ± {stats['annualized_benchmark_return'].std()*100:.2f}%")
    print(f"  Average Return: {stats['benchmark_return'].mean()*100:.2f}% ± {stats['benchmark_return'].std()*100:.2f}%")
    print(f"  Best Episode: {stats['benchmark_return'].max()*100:.2f}%")
    print(f"  Worst Episode: {stats['benchmark_return'].min()*100:.2f}%")
    print(f"  Win Rate: {(stats['benchmark_return'] > 0).mean()*100:.1f}%")
    print("RISK METRICS:")
    print(f"  Average Sharpe Ratio: {stats['benchmark_sharpe_ratio'].mean():.3f} ± {stats['benchmark_sharpe_ratio'].std():.3f}")
    print(f"  Average Sortino Ratio: {stats['benchmark_sortino_ratio'].mean():.3f} ± {stats['benchmark_sortino_ratio'].std():.3f}")
    print(f"  Average Max Drawdown: {stats['benchmark_max_drawdown'].mean():.3f} ± {stats['benchmark_max_drawdown'].std():.3f}")
    print(f"  Benchmark Volatility: {stats['volatility'].mean()*100:.2f}%")
    print("TRADING BEHAVIOR: Buy and Hold")
    
    print("\n" + "="*80)
        
    
        
            
    
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
        restuls_path=args.results_path,
        episode_months=args.episode_length,
        n_episodes = args.n_episodes
    )