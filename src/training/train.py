import os
import sys
import pandas as pd
import numpy as np
from stable_baselines3 import PPO, DDPG
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.logger import configure
from stable_baselines3.common.noise import OrnsteinUhlenbeckActionNoise, NormalActionNoise

# Add the project root directory to the system path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(project_root)

# Import the NEW hedging environment
from src.environment.hedging_env import PortfolioHedgingEnv

def train_hedging_model(
    data_path="data/processed/NVDA_hedging_features.csv",
    log_dir="logs/tensorboard",
    checkpoints_dir="checkpoints",
    best_model_dir="models/best_model",
    eval_log_dir="logs/evaluation",
    total_timesteps=10_000_000,
    episode_months=6,
    window_size=5,
    dead_zone=0.03,  
    initial_capital=2_000_000,
    commission=0.00125,
    algorithm="PPO",
    verbose=True
):
    
    # Create directories if they don't exist
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(checkpoints_dir, exist_ok=True)
    os.makedirs(best_model_dir, exist_ok=True)
    os.makedirs(eval_log_dir, exist_ok=True)
    
    # Load data
    if verbose:
        print("Loading data from {}...".format(data_path))
    df = pd.read_csv(data_path)
    
    # Convert dates
    df['Date'] = pd.to_datetime(df['Date'])
    dates = df['Date'].values
    
    # Extract prices and features
    prices = df['Close'].astype(np.float32).values
    if np.any(prices <= 0):
        raise ValueError("Prices contain non-positive values")
    
    # Remove Date and Close from features
    feature_columns = [col for col in df.columns if col not in ['Date', 'Close']]
    features = df[feature_columns].astype(np.float32).values
    
    if verbose:
        print(f"Dataset shape: {df.shape}")
        print(f"Features shape: {features.shape}")
        print(f"Price data points: {len(prices)}")
        print(f"Date range: {dates[0]} to {dates[-1]}")
        print(f"Dead zone: ±{dead_zone*100}%")
    
    # Create environments using the NEW environment
    train_env = PortfolioHedgingEnv(
        features=features,
        prices=prices,
        episode_months=episode_months,
        window_size=window_size,
        dead_zone=dead_zone,
        commission=commission,
        initial_capital=initial_capital
    )
    
    eval_env = PortfolioHedgingEnv(
        features=features,
        prices=prices,
        episode_months=episode_months,
        window_size=window_size,
        dead_zone=dead_zone,
        commission=commission,
        initial_capital=initial_capital
    )
    
    # Configure logging
    new_logger = configure(log_dir, ["stdout", "tensorboard"])
    
    
    if algorithm.upper() == "DDPG":
        
        if verbose:
            print("Using DDPG (Deep Deterministic Policy Gradient) for continuous hedging control...")
        
        n_actions = train_env.action_space.shape[-1]
        # action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))
        action_noise = OrnsteinUhlenbeckActionNoise(
            mean=np.zeros(n_actions),
            sigma=0.2 * np.ones(n_actions)  
        )
        
        model = DDPG(
            "MlpPolicy",
            train_env,
            action_noise=action_noise,
            verbose=1,
            tensorboard_log=log_dir,
            learning_rate=1e-4,
            buffer_size=1_000_000,
            learning_starts=10_000,
            batch_size=256,
            tau=0.005,
            gamma=0.99,
            train_freq=(1, "episode"),
            gradient_steps=1,
            policy_kwargs=dict(net_arch=dict(pi=[256, 256], qf=[256, 256]))
            # ent_coef='auto',
            # target_update_interval=1,
            # target_entropy='auto'
        )
        model_prefix = "ddpg_hedging"
        
    else:  # PPO
        if verbose:
            print("Using PPO (Proximal Policy Optimization) for hedging...")
        
        model = PPO(
            "MlpPolicy",
            train_env,
            verbose=1,
            tensorboard_log=log_dir,
            learning_rate=3e-4,
            n_steps=2048,
            batch_size=64,
            n_epochs=10,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            # clip_range_vf=None,
            ent_coef=0.01,
            vf_coef=0.5,
            max_grad_norm=0.5,
            policy_kwargs=dict(net_arch=[dict(pi=[256, 256], vf=[256, 256])])
        )
        model_prefix = "ppo_hedging"
    
    model.set_logger(new_logger)
    
    # Set up callbacks
    checkpoint_callback = CheckpointCallback(
        save_freq=10_000,
        save_path=checkpoints_dir,
        name_prefix=model_prefix
    )
    
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=best_model_dir,
        log_path=eval_log_dir,
        eval_freq=5_000,
        n_eval_episodes=10,
        deterministic=True,
        render=False
    )
    
    # Train model
    if verbose:
        print(f"Training {algorithm} for {total_timesteps} timesteps...")
        print(f"Episode length: {episode_months} months")
        print(f"Portfolio value: ${initial_capital:,}")
        print(f"Dead zone: ±{dead_zone*100:.1f}%")
        print(f"Action space: [0.0 = no holdings, 1.0 = max long (no hedging), 2.0 = max long with max hedging]")
    
    model.learn(
        total_timesteps=total_timesteps,
        callback=[checkpoint_callback, eval_callback],
        tb_log_name=f"{model_prefix}_run"
    )
    
    # Save final model
    final_model_path = os.path.join("models", f"{model_prefix}_final")
    model.save(final_model_path)
    if verbose:
        print("Final model saved to {}".format(final_model_path))
    
    return model


def evaluate_model_sample_episodes(model_path, data_path, n_episodes=10, verbose=True):
    
    df = pd.read_csv(data_path)
    df['Date'] = pd.to_datetime(df['Date'])
    prices = df['Close'].astype(np.float32).values
    if np.any(prices <= 0):  
        raise ValueError("Prices contain non-positive values")
    
    feature_columns = [col for col in df.columns if col not in ['Date', 'Close']]
    features = df[feature_columns].astype(np.float32).values
    
    env = PortfolioHedgingEnv(
        features=features,
        prices=prices,
        episode_months=6,
        window_size=5,
        dead_zone=0.02,
        commission=0.00125,
        initial_capital=2_000_000
    )
    
    # Load model
    if "ddpg" in model_path.lower():
        model = DDPG.load(model_path)
    else:
        model = PPO.load(model_path)
    
    # Run episodes
    episode_stats = []
    for episode in range(n_episodes):
        obs, info = env.reset()
        done = False
        step_count = 0
        total_reward = 0.0
        
        # Get episode start info
        start_idx = info['current_index']
        start_price = prices[start_idx]
        
        # DEBUG: Print episode start info
        if verbose and episode < 3:
            print(f"\nEpisode {episode+1} DEBUG:")
            print(f"  Start index: {start_idx}")
            print(f"  Start price: ${start_price:.2f}")
            print(f"  Initial portfolio: ${info['portfolio_value']:,.2f}")
        
        action_history = []
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            action_history.append(action[0])
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            done = terminated or truncated
            step_count += 1
        
        # Get final episode info
        final_idx = start_idx + step_count
        if final_idx >= len(prices):
            final_idx = len(prices) - 1
        final_price = prices[final_idx]
        
        # Calculate benchmark (buy-and-hold) return
        benchmark_return = (final_price - start_price) / start_price
        
        stats = env.get_episode_stats()
        stats.update({
            'episode': episode,
            'start_idx': start_idx,
            'start_price': start_price,
            'final_price': final_price,
            'steps': step_count,
            'total_reward': total_reward,
            'benchmark_return': benchmark_return,
            'excess_return': stats['total_return'] - benchmark_return,
            'avg_action': np.mean(action_history),
            'action_std': np.std(action_history),
            'min_action': np.min(action_history),
            'max_action': np.max(action_history)
        })
        
        episode_stats.append(stats)
        
        if verbose:
            print(f"Episode {episode+1}: "
                  f"Strategy={stats['total_return']*100:.2f}%, "
                  f"Benchmark={benchmark_return*100:.2f}%, "
                  f"Excess={stats['excess_return']*100:.2f}%, "
                  f"Sharpe={stats['sharpe_ratio']:.3f}, "
                  f"Actions: avg={stats['avg_action']:.3f}, "
                  f"range=[{stats['min_action']:.3f}, {stats['max_action']:.3f}]")
    
    # Summary statistics
    stats_df = pd.DataFrame(episode_stats)
    
    if verbose:
        print("\n" + "="*80)
        print("PERFORMANCE SUMMARY")
        print("="*80)
        print("STRATEGY PERFORMANCE:")
        print(f"  Average Return: {stats_df['total_return'].mean()*100:.2f}% ± {stats_df['total_return'].std()*100:.2f}%")
        print(f"  Median Return: {stats_df['total_return'].median()*100:.2f}%")
        print(f"  Best Episode: {stats_df['total_return'].max()*100:.2f}%")
        print(f"  Worst Episode: {stats_df['total_return'].min()*100:.2f}%")
        print(f"  Win Rate: {(stats_df['total_return'] > 0).mean()*100:.1f}%")
        
        print("\nBENCHMARK COMPARISON:")
        print(f"  Benchmark Avg Return: {stats_df['benchmark_return'].mean()*100:.2f}% ± {stats_df['benchmark_return'].std()*100:.2f}%")
        print(f"  Average Excess Return: {stats_df['excess_return'].mean()*100:.2f}% ± {stats_df['excess_return'].std()*100:.2f}%")
        print(f"  Outperformance Rate: {(stats_df['excess_return'] > 0).mean()*100:.1f}%")
        print(f"  Average Outperformance: {stats_df['excess_return'][stats_df['excess_return'] > 0].mean()*100:.2f}%")
        
        print("\nRISK METRICS:")
        print(f"  Average Sharpe Ratio: {stats_df['sharpe_ratio'].mean():.3f} ± {stats_df['sharpe_ratio'].std():.3f}")
        print(f"  Strategy Volatility: {stats_df['volatility'].mean()*100:.2f}%")
        
        print("\nTRADING BEHAVIOR:")
        print(f"  Average Action: {stats_df['avg_action'].mean():.3f}")
        print(f"  Action Range: [{stats_df['min_action'].min():.3f}, {stats_df['max_action'].max():.3f}]")
        print(f"  Average Action Std: {stats_df['action_std'].mean():.3f}")
        print(f"  Average Reward: {stats_df['total_reward'].mean():.2f} ± {stats_df['total_reward'].std():.2f}")
        
        print("="*80)
    
    return stats_df

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Train portfolio hedging model')
    parser.add_argument('--data_path', default="data/processed/NVDA_hedging_features.csv",
                       help='Path to processed dataset')
    parser.add_argument('--algorithm', choices=['PPO', 'DDPG'], default='PPO',
                       help='RL algorithm to use')
    parser.add_argument('--timesteps', type=int, default=10_000_000,
                       help='Total training timesteps')
    parser.add_argument('--episode_months', type=int, default=6,
                       help='Episode length in months')
    parser.add_argument('--evaluate', action='store_true',
                        help='Evaluate existing model instead of training')
    parser.add_argument('--model_path', default="models/best_model/best_model",
                        help='Path to model for evaluation')
    
    args = parser.parse_args()
    
    if args.evaluate:
        print("Evaluating model...")
        evaluate_model_sample_episodes(args.model_path, args.data_path)
    else:
        print("Training model...")
        trained_model = train_hedging_model(
            data_path=args.data_path,
            total_timesteps=args.timesteps,
            episode_months=args.episode_months,
            algorithm=args.algorithm
        )