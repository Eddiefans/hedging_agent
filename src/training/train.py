import os
import sys
import pandas as pd
import numpy as np
from stable_baselines3 import PPO, SAC
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.logger import configure
from stable_baselines3.common.noise import NormalActionNoise

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
    episode_length_months=6,
    window_size=5,
    dead_zone=0.05,  # 5% dead zone
    initial_portfolio=1_000_000,
    reserve_cash_pct=0.15,  # 15% reserve cash
    commission=0.00125,
    max_position_change_penalty=0.20,  # 20% position change penalty threshold
    algorithm="SAC",  # or "PPO" for policy gradient
    verbose=True
):
    """
    Train a reinforcement learning agent for portfolio hedging using the environment.
    
    Args:
        data_path: Path to the processed dataset
        log_dir: Directory for TensorBoard logs
        checkpoints_dir: Directory for model checkpoints
        best_model_dir: Directory to save the best model
        eval_log_dir: Directory for evaluation logs
        total_timesteps: Total number of timesteps to train for
        episode_length_months: Length of each episode in months
        window_size: Observation window size
        dead_zone: Dead zone around 1.0 for no-action (5%)
        initial_portfolio: Initial portfolio value
        reserve_cash_pct: Percentage of portfolio to keep as cash reserve (15%)
        commission: Commission rate for trades
        max_position_change_penalty: Threshold for position change penalty (20%)
        algorithm: RL algorithm to use ("PPO" or "SAC")
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
    
    # Remove Date and Close from features
    feature_columns = [col for col in df.columns if col not in ['Date', 'Close']]
    features = df[feature_columns].astype(np.float32).values
    
    if verbose:
        print(f"Dataset shape: {df.shape}")
        print(f"Features shape: {features.shape}")
        print(f"Price data points: {len(prices)}")
        print(f"Date range: {dates[0]} to {dates[-1]}")
        print(f"Reserve cash: {reserve_cash_pct*100}%")
        print(f"Dead zone: ±{dead_zone*100}%")
    
    # Create environments using the NEW environment
    train_env = PortfolioHedgingEnv(
        features=features,
        prices=prices,
        dates=dates,
        episode_length_months=episode_length_months,
        window_size=window_size,
        dead_zone=dead_zone,
        initial_portfolio=initial_portfolio,
        reserve_cash_pct=reserve_cash_pct,
        commission=commission,
        max_position_change_penalty=max_position_change_penalty
    )
    
    eval_env = PortfolioHedgingEnv(
        features=features,
        prices=prices,
        dates=dates,
        episode_length_months=episode_length_months,
        window_size=window_size,
        dead_zone=dead_zone,
        initial_portfolio=initial_portfolio,
        reserve_cash_pct=reserve_cash_pct,
        commission=commission,
        max_position_change_penalty=max_position_change_penalty
    )
    
    # Configure logging
    new_logger = configure(log_dir, ["stdout", "tensorboard"])
    
    # Create model based on algorithm choice
    if algorithm.upper() == "SAC":
        # SAC is better for continuous control
        if verbose:
            print("Using SAC (Soft Actor-Critic) for continuous hedging control...")
        
        # Add action noise for exploration
        n_actions = train_env.action_space.shape[-1]
        action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))
        
        model = SAC(
            "MlpPolicy",
            train_env,
            action_noise=action_noise,
            verbose=1,
            tensorboard_log=log_dir,
            learning_rate=3e-4,
            buffer_size=100000,
            learning_starts=1000,
            batch_size=256,
            tau=0.005,
            gamma=0.99,
            train_freq=1,
            gradient_steps=1,
            ent_coef='auto',
            target_update_interval=1,
            target_entropy='auto'
        )
        model_prefix = "sac_hedging"
        
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
            clip_range_vf=None,
            ent_coef=0.0,
            vf_coef=0.5,
            max_grad_norm=0.5
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
        print(f"Episode length: {episode_length_months} months")
        print(f"Portfolio value: ${initial_portfolio:,}")
        print(f"Action space: [0.0 = max long, 1.0 = neutral, 2.0 = max short]")
    
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

def quick_test_environment(data_path="data/processed/NVDA_hedging_features.csv"):
    """
    Quick test to make sure the environment works correctly.
    """
    print("Testing PortfolioHedgingEnv...")
    
    # Load data
    df = pd.read_csv(data_path)
    df['Date'] = pd.to_datetime(df['Date'])
    dates = df['Date'].values
    prices = df['Close'].astype(np.float32).values
    feature_columns = [col for col in df.columns if col not in ['Date', 'Close']]
    features = df[feature_columns].astype(np.float32).values
    
    # Create environment
    env = PortfolioHedgingEnv(
        features=features,
        prices=prices,
        dates=dates,
        episode_length_months=1,  # Short episode for testing
        window_size=5
    )
    
    # Test a few steps
    obs = env.reset()
    print(f"Initial observation shape: {obs.shape}")
    print(f"Initial portfolio value: ${env._calculate_portfolio_value():,.2f}")
    
    # Test different actions
    test_actions = [0.5, 1.0, 1.5, 0.95, 1.05]  # Long, neutral, short, dead zone
    
    for i, action in enumerate(test_actions):
        if env.episode_done:
            obs = env.reset()
        
        obs, reward, done, info = env.step([action])
        print(f"Step {i+1}: Action={action:.2f}, Reward={reward:.4f}, "
              f"Portfolio=${info['portfolio_value']:,.2f}, Cash=${info['cash']:,.2f}")
        
        if done:
            stats = env.get_episode_stats()
            print(f"Episode finished. Total return: {stats['total_return']*100:.2f}%")
            break
    
    print("Environment test completed successfully!")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Train portfolio hedging model')
    parser.add_argument('--data_path', default="data/processed/NVDA_hedging_features.csv",
                       help='Path to processed dataset')
    parser.add_argument('--algorithm', choices=['PPO', 'SAC'], default='SAC',
                       help='RL algorithm to use')
    parser.add_argument('--timesteps', type=int, default=1_000_000,
                       help='Total training timesteps')
    parser.add_argument('--episode_months', type=int, default=6,
                       help='Episode length in months')
    parser.add_argument('--test', action='store_true',
                       help='Run environment test instead of training')
    
    args = parser.parse_args()
    
    if args.test:
        print("Running environment test...")
        quick_test_environment(args.data_path)
    else:
        print("Training model...")
        trained_model = train_hedging_model(
            data_path=args.data_path,
            total_timesteps=args.timesteps,
            episode_length_months=args.episode_months,
            algorithm=args.algorithm
        )