import numpy as np
import gym
from gym import spaces

class PortfolioHedgingEnv(gym.Env):
    """
    Gym environment for a portfolio hedging agent:
    - Agent manages hedging for a portfolio of a single stock (NVIDIA)
    - Output is continuous between 0 and 2, where 1 is neutral
    - Values below 1 indicate buying more (e.g., 0.9 = buy 10%)
    - Values above 1 indicate short selling (e.g., 1.1 = sell 10% more)
    - Dead zone of ±10% around 1.0 (0.91-1.09) results in no action
    - Each episode randomly selects a 6-month period from the dataset
    - Rewards based on hedging performance over the episode period
    """

    def __init__(self, features: np.ndarray, prices: np.ndarray, dates: np.ndarray = None,
                 episode_length_months=6, window_size=5, dead_zone=0.09,
                 portfolio_value=1_000_000, commission=0.00125, 
                 max_position=2.0, min_position=-1.0):
        super().__init__()
        
        assert len(features) == len(prices), "Features and prices must have same length"
        
        self.features = features
        self.prices = prices
        self.dates = dates if dates is not None else np.arange(len(prices))
        self.window_size = window_size
        self.dead_zone = dead_zone  # ±9% dead zone around 1.0
        self.portfolio_value = portfolio_value
        self.commission = commission
        self.max_position = max_position  # Maximum 200% long
        self.min_position = min_position  # Maximum 100% short
        
        # Calculate episode length in trading days (assuming ~21 trading days per month)
        self.episode_length = episode_length_months * 21
        
        # Ensure we have enough data for episodes
        self.min_start_idx = window_size
        self.max_start_idx = len(features) - self.episode_length - 1
        
        if self.max_start_idx <= self.min_start_idx:
            raise ValueError(f"Not enough data for {episode_length_months}-month episodes")
        
        # Action space: continuous value between 0 and 2
        self.action_space = spaces.Box(low=0.0, high=2.0, shape=(1,), dtype=np.float32)
        
        # Observation space: window of features
        n_features = features.shape[1]
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(window_size, n_features),
            dtype=np.float32
        )
        
        # Episode tracking
        self.current_episode_start = 0
        self.current_step = 0
        self.episode_done = False
        
        # Portfolio tracking
        self.current_position = 1.0  # Start neutral (100% of portfolio value)
        self.cash = 0.0
        self.stock_shares = 0.0
        self.portfolio_history = []
        self.position_history = []
        self.action_history = []
        self.returns_history = []
        
        self.reset()

    def reset(self):
        """Reset environment for a new episode with random 6-month period"""
        # Randomly select start of episode
        self.current_episode_start = np.random.randint(
            self.min_start_idx, self.max_start_idx
        )
        self.current_step = 0
        self.episode_done = False
        
        # Reset portfolio state
        self.current_position = 1.0  # Start neutral
        initial_price = self.prices[self.current_episode_start]
        self.stock_shares = self.portfolio_value / initial_price
        self.cash = 0.0
        
        # Clear history
        self.portfolio_history = [self.portfolio_value]
        self.position_history = [self.current_position]
        self.action_history = []
        self.returns_history = []
        
        return self._get_observation()

    def step(self, action):
        """Execute one step in the environment"""
        if self.episode_done:
            raise ValueError("Episode is done, call reset()")
        
        # Clamp action to valid range
        target_action_value = np.clip(action[0], 0.0, 2.0)
        
        # Apply dead zone
        if abs(target_action_value - 1.0) <= self.dead_zone:
            target_position = self.current_position  # No change
        else:
            target_position = 2.0 - target_action_value
        
        # Clamp to position limits
        target_position = np.clip(target_position, self.min_position, self.max_position)
        
        # Calculate current portfolio value before action
        current_idx = self.current_episode_start + self.current_step
        current_price = self.prices[current_idx]
        portfolio_value_before = self.stock_shares * current_price + self.cash
        
        # Execute position change if needed
        position_change = target_position - self.current_position
        self._execute_trade(position_change, current_price, portfolio_value_before)
        
        # Move to next step
        self.current_step += 1
        next_idx = self.current_episode_start + self.current_step
        
        # Check if episode is done
        if self.current_step >= self.episode_length or next_idx >= len(self.prices):
            self.episode_done = True
        
        # Calculate reward and portfolio value after price change
        if not self.episode_done:
            next_price = self.prices[next_idx]
        else:
            next_price = current_price  # Use current price if episode ended
            
        portfolio_value_after = self.stock_shares * next_price + self.cash
        
        # Calculate step return
        step_return = (portfolio_value_after - portfolio_value_before) / portfolio_value_before
        self.returns_history.append(step_return)
        
        # Store tracking data
        self.portfolio_history.append(portfolio_value_after)
        self.position_history.append(self.current_position)
        self.action_history.append(target_position)
        
        # Calculate reward (could be step return or cumulative performance)
        reward = self._calculate_reward(step_return, portfolio_value_after)
        
        # Get next observation
        obs = self._get_observation() if not self.episode_done else self._get_observation()
        
        # Info dictionary
        info = {
            'portfolio_value': portfolio_value_after,
            'position': self.current_position,
            'step_return': step_return,
            'episode_step': self.current_step,
            'episode_start_date': self.dates[self.current_episode_start] if hasattr(self.dates[0], 'strftime') else self.current_episode_start
        }
        
        return obs, reward, self.episode_done, info

    def _execute_trade(self, position_change, current_price, current_portfolio_value):
        """Execute the trade to achieve target position"""
        if abs(position_change) < 1e-6:  # No significant change
            return
        
        # Calculate the dollar amount of position change needed
        target_dollar_change = position_change * current_portfolio_value
        shares_to_trade = target_dollar_change / current_price
        
        # Apply commission
        commission_cost = abs(target_dollar_change) * self.commission
        
        # Update portfolio
        self.stock_shares += shares_to_trade
        self.cash -= (shares_to_trade * current_price + commission_cost)
        self.current_position += position_change

    def _calculate_reward(self, step_return, current_portfolio_value):
        """Calculate reward for the current step"""
        # Simple reward: step return adjusted for volatility
        # You could make this more sophisticated based on your hedging objectives
        
        # Reward step return
        reward = step_return
        
        # Penalty for extreme positions (encourage reasonable hedging)
        position_penalty = 0
        if self.current_position > 1.5:  # Very long
            position_penalty = -0.001 * (self.current_position - 1.5)
        elif self.current_position < 0.5:  # Very short
            position_penalty = -0.001 * (0.5 - self.current_position)
        
        reward += position_penalty
        
        return reward

    def _get_observation(self):
        """Get current observation window"""
        if self.episode_done:
            # Return last valid observation
            end_idx = self.current_episode_start + min(self.current_step, len(self.features) - 1)
        else:
            end_idx = self.current_episode_start + self.current_step
            
        start_idx = max(0, end_idx - self.window_size + 1)
        
        # Pad if needed
        obs_window = self.features[start_idx:end_idx + 1]
        if len(obs_window) < self.window_size:
            padding = np.zeros((self.window_size - len(obs_window), self.features.shape[1]))
            obs_window = np.vstack([padding, obs_window])
        
        return obs_window.astype(np.float32)

    def get_episode_stats(self):
        """Get statistics for the current episode"""
        if len(self.portfolio_history) < 2:
            return {}
        
        returns = np.array(self.returns_history)
        portfolio_values = np.array(self.portfolio_history)
        
        total_return = (portfolio_values[-1] - portfolio_values[0]) / portfolio_values[0]
        volatility = returns.std() if len(returns) > 1 else 0.0
        sharpe_ratio = returns.mean() / volatility if volatility > 0 else 0.0
        max_drawdown = self._calculate_max_drawdown(portfolio_values)
        
        return {
            'total_return': total_return,
            'volatility': volatility,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'final_position': self.current_position,
            'num_trades': sum(1 for i in range(1, len(self.position_history)) 
                            if abs(self.position_history[i] - self.position_history[i-1]) > 1e-6)
        }

    def _calculate_max_drawdown(self, portfolio_values):
        """Calculate maximum drawdown"""
        peak = portfolio_values[0]
        max_dd = 0.0
        
        for value in portfolio_values:
            if value > peak:
                peak = value
            drawdown = (peak - value) / peak
            max_dd = max(max_dd, drawdown)
        
        return max_dd

    def render(self, mode='human'):
        """Render the environment (optional)"""
        if mode == 'human':
            current_value = self.portfolio_history[-1] if self.portfolio_history else self.portfolio_value
            print(f"Step: {self.current_step}, Portfolio: ${current_value:,.2f}, Position: {self.current_position:.3f}")
