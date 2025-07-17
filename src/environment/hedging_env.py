import numpy as np
import gymnasium as gym
from gymnasium import spaces  # Added import for spaces module
import pandas as pd

class PortfolioHedgingEnv(gym.Env):
    def __init__(
        self, 
        features: np.ndarray, 
        prices: np.ndarray, 
        dates: np.ndarray = None,
        episode_length_months=6,
        window_size=5,
        dead_zone=0.005,
        initial_long_capital=1_000_000,
        initial_short_capital=1_000_000,
        commission=0.00125,
        action_change_penalty_threshold=0.1,
        max_shares_per_trade=0.5
    ):
        
        # Initialize the environment with input data and parameters
        super().__init__()
        
        # Validate input data and parameters
        assert len(features) == len(prices), "Features and prices must have the same length"
        assert len(features) > window_size, f"Features length ({len(features)}) must be greater than window_size ({window_size})"
        assert all(prices >= 0), "Prices must be non-negative"
        assert initial_long_capital > 0 and initial_short_capital > 0, "Initial capitals must be positive"
        assert 0 <= commission <= 1, "Commission must be between 0 and 1"
        assert 0 < max_shares_per_trade <= 1, "max_shares_per_trade must be between 0 and 1"

        # Store environment parameters
        self.features = features
        self.prices = prices
        self.dates = pd.to_datetime(dates) if dates is not None else np.arange(len(prices))
        self.window_size = window_size
        self.dead_zone = dead_zone
        self.initial_portfolio_value = initial_long_capital + initial_short_capital
        self.initial_long_capital = initial_long_capital
        self.initial_short_capital = initial_short_capital
        self.commission = commission
        self.action_change_penalty_threshold = action_change_penalty_threshold
        self.max_shares_per_trade = max_shares_per_trade
        self.episode_length = episode_length_months * 21  # Convert months to trading days (21 days/month)

        # Calculate valid start indices for episodes
        self.min_start_idx = self.window_size
        self.max_start_idx = len(features) - self.episode_length - 1
        assert self.max_start_idx > self.min_start_idx, "Not enough data for specified episode length"

        # Define action and observation spaces
        self.action_space = spaces.Box(low=0.0, high=2.0, shape=(1,), dtype=np.float32)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(window_size * features.shape[1] + 4,),
            dtype=np.float32
        )

        # Initialize state variables
        
        initial_price = prices[0]  # Use first price as initial reference
        
        self.current_episode_start = 0
        self.current_step = 0
        self.episode_done = False
        self.current_long_shares = self.initial_long_capital / initial_price  # Initial long position
        self.current_short_shares = 0.0  # No initial short positions
        self.cash = self.initial_short_capital
        self.current_portfolio_value = 0.0
        self.last_action = 0.0
        self.portfolio_history = []
        self.action_history = []
        self.price_history = []
        self.returns_history = []
        self.np_random = np.random.default_rng()
        self.reset()

    def _get_observation(self):
        
        # Get current observation based on market features and portfolio state
        current_idx = self.current_episode_start + self.current_step
        end_idx = current_idx + self.window_size

        # Fill with padding if there is not enough data to cover the range [current_idx:end_idx]
        if end_idx > len(self.features):
            valid_idx = len(self.features) - 1
            obs_window = self.features[current_idx:valid_idx + 1]
            padding = np.tile(self.features[valid_idx], (self.window_size - (valid_idx - current_idx + 1), 1))
            market_features = np.vstack((obs_window, padding))
        else:
            market_features = self.features[current_idx:end_idx]

        # Normalize market features
        market_features = (market_features - market_features.mean(axis=0)) / (market_features.std(axis=0) + 1e-8)
        market_features = market_features.flatten()

        # Get current price and portfolio state
        portfolio_state = np.array([
            self.current_long_shares,
            self.current_short_shares,
            self.cash / self.initial_portfolio_value,
            self.current_portfolio_value / self.initial_portfolio_value
        ], dtype=np.float32)

        return np.concatenate((market_features, portfolio_state))

    def _calculate_portfolio_value(self, current_price):
        
        # Calculate current portfolio value based on long and short positions
        long_value = self.current_long_shares * current_price
        short_value = self.current_short_shares * current_price
        return self.cash + long_value - short_value

    def reset(self, seed=None):
        # Reset the environment to a new episode
        super().reset(seed=seed)
        
        initial_price = self.prices[self.current_episode_start]
        
        self.np_random = np.random.default_rng(seed)
        self.current_episode_start = self.np_random.integers(self.min_start_idx, self.max_start_idx)
        self.current_step = 0
        self.episode_done = False
        self.current_long_shares = self.initial_long_capital / initial_price  # Fully invest initial long capital
        self.current_short_shares = 0.0  # No short positions at start
        self.cash = self.initial_short_capital
        self.current_portfolio_value = self._calculate_portfolio_value(initial_price)
        self.last_action = 1.0  # Start with neutral action value
        self.portfolio_history = [self.current_portfolio_value]
        self.action_history = [self.last_action]
        self.price_history = [initial_price]
        self.returns_history = []
        
        return self._get_observation(), self._get_info()

    def step(self, action):
        # Execute a single step in the environment based on the action
        if self.episode_done:
            raise ValueError("Episode is done, call reset()")

        action_value = np.clip(action[0], 0.0, 2.0)
        current_idx = self.current_episode_start + self.current_step
        current_price = self.prices[current_idx]
        prev_value = self.current_portfolio_value

        # Determine target shares based on action value
        if 0.0 <= action_value <= 1.0:
            
            target_long_shares = (self.initial_long_capital * action_value) / current_price
            target_short_shares = 0.0
            
        else:
            
            target_long_shares = self.initial_long_capital / current_price
            target_short_shares = (action_value - 1.0) * self.initial_short_capital / current_price

        # Calculate shares to execute with maximum limit
        current_total = self.current_long_shares + self.current_short_shares
        target_total = target_long_shares + target_short_shares
        max_shares = (self.current_portfolio_value / current_price) * self.max_shares_per_trade
        shares_to_execute = np.clip(target_total - current_total, -max_shares, max_shares)

        # Apply dead zone to avoid small trades
        if (abs(action_value - self.last_action) < self.dead_zone or 
            abs(shares_to_execute * current_price) < self.dead_zone * self.current_portfolio_value):
            shares_to_execute = 0.0

        # Execute trades
        if shares_to_execute > 0:
            
            long_increase = min(max(0, target_long_shares - self.current_long_shares), max_shares)
            short_increase = min(max(0, target_short_shares - self.current_short_shares), max_shares - long_increase)
            self.current_long_shares += long_increase
            self.current_short_shares += short_increase 
            self.cash += (short_increase - long_increase) * current_price * (1 - self.commission)
            
        elif shares_to_execute < 0:
            
            abs_execute = abs(shares_to_execute)
            short_decrease = min(abs_execute, self.current_short_shares)
            long_decrease = min(abs_execute - short_decrease, self.current_long_shares) if abs_execute > short_decrease else 0
            self.current_short_shares -= short_decrease
            self.current_long_shares -= long_decrease
            self.cash += (long_decrease - short_decrease) * current_price * (1 - self.commission)

        self.current_step += 1

        # Check if episode is done
        self.episode_done = (self.current_step >= self.episode_length or 
                            current_idx >= len(self.prices) - 1)

        # Calculate next price and update portfolio
        next_price = np.mean(self.prices[max(0, current_idx - 3):current_idx + 3]) if not self.episode_done else current_price
        self.current_portfolio_value = self._calculate_portfolio_value(next_price)
        step_return = (self.current_portfolio_value - prev_value) / prev_value if prev_value else 0
        self.last_action = self.action_history[-1]
        reward = self._calculate_reward(step_return, action_value)

        # Update history
        self.portfolio_history.append(self.current_portfolio_value)
        self.action_history.append(action_value)
        self.price_history.append(next_price)
        self.returns_history.append(step_return)

        return self._get_observation(), reward, self.episode_done, False, self._get_info()

    def _calculate_reward(self, step_return, current_action_value):
        # Base reward calculated as a scaled step return
        reward = step_return * 100.0
        
        # Calculate penalty for large action changes
        action_diff = abs(current_action_value - self.last_action)
        if action_diff > self.action_change_penalty_threshold:
            penalty = (action_diff - self.action_change_penalty_threshold) * 50.0
            reward -= penalty
        
        # Apply penalties based on portfolio history if available
        if len(self.portfolio_history) > 1:
            drawdown = self._calculate_max_drawdown(np.array(self.portfolio_history))
            if drawdown > 0.10:  # Penalize if drawdown exceeds 10%
                reward -= (drawdown - 0.10) * 20.0
            if self.current_long_shares > 2 * (self.initial_long_capital / self.prices[self.current_episode_start]):  # Penalize excessive long shares
                excess_shares = self.current_long_shares - 2 * (self.initial_long_capital / self.prices[self.current_episode_start])
                reward -= excess_shares * 0.01
        
        # Clip reward to ensure it stays within bounds
        return max(min(reward, 100.0), -100.0)

    def _get_info(self):
        # Provide additional information about the current state
        current_idx = self.current_episode_start + self.current_step
        date = self.dates[min(current_idx, len(self.dates) - 1)]
        return {
            'current_price': self.price_history[-1] if self.price_history else self.prices[self.current_episode_start],
            'current_long_shares': self.current_long_shares,
            'current_short_shares': self.current_short_shares,
            'cash': self.cash,
            'portfolio_value': self.current_portfolio_value,
            'total_return_episode_so_far': (self.current_portfolio_value - self.initial_portfolio_value) / self.initial_portfolio_value,
            'date': date
        }

    def get_episode_stats(self):
        # Calculate episode statistics if enough data is available
        if len(self.portfolio_history) < 2:
            return {}
        returns = np.array(self.returns_history)
        values = np.array(self.portfolio_history)
        total_return = (values[-1] - values[0]) / values[0]
        volatility = returns.std() if len(returns) > 1 else 0.0
        sharpe_ratio = returns.mean() / volatility if volatility > 0 else 0.0
        max_drawdown = self._calculate_max_drawdown(values)
        return {
            'total_return': total_return,
            'volatility': volatility,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'final_position_net_shares': self.current_long_shares - self.current_short_shares,
            'num_trades': sum(1 for i in range(1, len(self.action_history)) 
                             if abs(self.action_history[i] - self.action_history[i-1]) > 1e-6)
        }

    def _calculate_max_drawdown(self, values):
        # Calculate the maximum drawdown from a series of portfolio values
        peak = values[0]
        max_dd = 0.0
        for value in values:
            if value > peak:
                peak = value
            if peak > 0:
                drawdown = (peak - value) / peak
                max_dd = max(max_dd, drawdown)
        return max_dd

    def render(self, mode='human'):
        # Render the current state in human-readable format
        if mode == 'human':
            current_value = self.portfolio_history[-1] if self.portfolio_history else self.initial_portfolio_value
            stats = self.get_episode_stats()
            print(f"Step: {self.current_step}")
            print(f"Portfolio Value: ${current_value:,.2f}")
            print(f"Long Shares: {self.current_long_shares:.2f}, Short Shares: {self.current_short_shares:.2f}")
            print(f"Cash: ${self.cash:,.2f}")
            if stats:
                print(f"Total Return: {stats['total_return']:.4f}")
                print(f"Volatility: {stats['volatility']:.4f}")
                print(f"Sharpe Ratio: {stats['sharpe_ratio']:.4f}")
                print(f"Max Drawdown: {stats['max_drawdown']:.4f}")