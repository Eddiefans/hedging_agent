import os
import sys
import pandas as pd
import numpy as np
from stable_baselines3 import PPO, DDPG
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.logger import configure
from stable_baselines3.common.noise import OrnsteinUhlenbeckActionNoise
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.preprocessing import preprocess_obs

# Add the project root directory to the system path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(project_root)

# Import the hedging environment
from src.environment.hedging_env import PortfolioHedgingEnv

def train_hedging_model(
    data_path="data/processed/NVDA_hedging_features.csv",
    log_dir="logs/tensorboard",
    checkpoints_dir="checkpoints",
    best_model_dir="models/best_model",
    eval_log_dir="logs/evaluation",
    total_timesteps=5_000_000,
    episode_length_months=6,
    window_size=5,
    dead_zone=0.005,
    initial_long_capital=1_000_000,
    initial_short_capital=1_000_000,
    commission=0.00125,
    action_change_penalty_threshold=0.1,
    max_shares_per_trade=0.5,
    algorithm="DDPG",
    n_envs=4,
    verbose=True
):
    """
    Train a reinforcement learning agent for portfolio hedging.
    
    Args:
        data_path: Path to the processed dataset
        log_dir: Directory for TensorBoard logs
        checkpoints_dir: Directory for model checkpoints
        best_model_dir: Directory to save the best model
        eval_log_dir: Directory for evaluation logs
        total_timesteps: Total number of timesteps to train for
        episode_length_months: Length of each episode in months
        window_size: Observation window size
        dead_zone: Dead zone around action changes
        initial_long_capital: Initial capital for long positions (1M)
        initial_short_capital: Initial capital for short positions (1M)
        commission: Commission rate for trades
        action_change_penalty_threshold: Threshold for penalizing large action changes
        max_shares_per_trade: Maximum proportion of portfolio value per trade
        algorithm: RL algorithm to use ("PPO" or "DDPG")
        n_envs: Number of parallel environments for training
        verbose: Whether to print progress messages
    
    Returns:
        model: Trained RL model
    """
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
    
    # Vectorizar el entorno para entrenamiento paralelo
    def make_env():
        return PortfolioHedgingEnv(
            features=features,
            prices=prices,
            dates=dates,
            episode_length_months=episode_length_months,
            window_size=window_size,
            dead_zone=dead_zone,
            initial_long_capital=initial_long_capital,
            initial_short_capital=initial_short_capital,
            commission=commission,
            action_change_penalty_threshold=action_change_penalty_threshold,
            max_shares_per_trade=max_shares_per_trade
        )
    
    train_env = make_vec_env(lambda: make_env(), n_envs=n_envs, seed=0)
    eval_env = PortfolioHedgingEnv(
        features=features,
        prices=prices,
        dates=dates,
        episode_length_months=episode_length_months,
        window_size=window_size,
        dead_zone=dead_zone,
        initial_long_capital=initial_long_capital,
        initial_short_capital=initial_short_capital,
        commission=commission,
        action_change_penalty_threshold=action_change_penalty_threshold,
        max_shares_per_trade=max_shares_per_trade
    )
    
    # Configure logging
    new_logger = configure(log_dir, ["stdout", "tensorboard"])
    
    # Create model based on algorithm choice
    if algorithm.upper() == "DDPG":
        if verbose:
            print("Using DDPG (Deep Deterministic Policy Gradient) for continuous hedging control...")
        
        n_actions = train_env.action_space.shape[-1]
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
            n_steps=2048 // n_envs,
            batch_size=64,
            n_epochs=10,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=0.01,
            vf_coef=0.5,
            max_grad_norm=0.5,
            policy_kwargs=dict(net_arch=[dict(pi=[256, 256], vf=[256, 256])])
        )
        model_prefix = "ppo_hedging"
    
    model.set_logger(new_logger)
    
    # Set up callbacks
    checkpoint_callback = CheckpointCallback(
        save_freq=max(10_000 // n_envs, 1),
        save_path=checkpoints_dir,
        name_prefix=model_prefix
    )
    
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=best_model_dir,
        log_path=eval_log_dir,
        eval_freq=max(5_000 // n_envs, 1),
        n_eval_episodes=10,
        deterministic=True,
        render=False
    )
    
    # Train model
    if verbose:
        print(f"Training {algorithm} for {total_timesteps} timesteps...")
        print(f"Episode length: {episode_length_months} months")
        print(f"Portfolio value: ${initial_long_capital + initial_short_capital:,}")
        print(f"Dead zone: ±{dead_zone*100:.1f}%")
        print(f"Action change penalty threshold: {action_change_penalty_threshold:.2f}")
    
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
    """
    Evaluate a trained model on sample episodes and print statistics.
    """
    # Load data
    df = pd.read_csv(data_path)
    df['Date'] = pd.to_datetime(df['Date'])
    dates = df['Date'].values
    prices = df['Close'].astype(np.float32).values
    if np.any(prices <= 0):
        raise ValueError("Prices contain non-positive values")
    
    feature_columns = [col for col in df.columns if col not in ['Date', 'Close']]
    features = df[feature_columns].astype(np.float32).values
    
    # Create environment
    env = PortfolioHedgingEnv(
        features=features,
        prices=prices,
        dates=dates,
        episode_length_months=6,
        window_size=5,
        dead_zone=0.005,
        initial_portfolio_value=2_000_000,
        initial_long_capital=1_000_000,
        initial_short_capital=1_000_000,
        commission=0.00125,
        action_change_penalty_threshold=0.1,
        max_shares_per_trade=0.5
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
        
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            done = terminated or truncated
            step_count += 1
        
        stats = env.get_episode_stats()
        stats['episode'] = episode
        stats['steps'] = step_count
        stats['total_reward'] = total_reward
        episode_stats.append(stats)
        
        if verbose:
            print(f"Episode {episode+1}: Return={stats['total_return']*100:.2f}%, "
                  f"Sharpe={stats['sharpe_ratio']:.3f}, "
                  f"Max DD={stats['max_drawdown']*100:.2f}%, "
                  f"Total Reward={stats['total_reward']:.2f}, "
                  f"Final Pos Net Shares={stats['final_position_net_shares']:.2f}")
    
    # Summary statistics
    stats_df = pd.DataFrame(episode_stats)
    if verbose:
        print("\n=== Summary Statistics ===")
        print(f"Average Return: {stats_df['total_return'].mean()*100:.2f}% ± {stats_df['total_return'].std()*100:.2f}%")
        print(f"Average Sharpe Ratio: {stats_df['sharpe_ratio'].mean():.3f} ± {stats_df['sharpe_ratio'].std():.3f}")
        print(f"Average Max Drawdown: {stats_df['max_drawdown'].mean()*100:.2f}% ± {stats_df['max_drawdown'].std()*100:.2f}%")
        print(f"Average Total Reward: {stats_df['total_reward'].mean():.2f} ± {stats_df['total_reward'].std():.2f}")
        print(f"Win Rate: {(stats_df['total_return'] > 0).mean()*100:.1f}%")
    
    return stats_df

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Train portfolio hedging model')
    parser.add_argument('--data_path', default="data/processed/NVDA_hedging_features.csv",
                        help='Path to processed dataset')
    parser.add_argument('--algorithm', choices=['PPO', 'DDPG'], default='DDPG',
                        help='RL algorithm to use')
    parser.add_argument('--timesteps', type=int, default=5_000_000,
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
            episode_length_months=args.episode_months,
            algorithm=args.algorithm
        )