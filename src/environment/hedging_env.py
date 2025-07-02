import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
import datetime

class PortfolioHedgingEnv(gym.Env):
    """
    Gymnasium environment for a portfolio hedging agent.

    Manages a portfolio against asset price movements.
    - Capital: $1M for long, $1M for short positions.
    - Action Space (0.0 to 2.0):
    - 0.0: 0% invested (all cash).
    - 1.0: 100% of long capital in long positions.
    - 2.0: 100% long capital in long, AND 100% short capital in short positions.
    - Dead Zone: +/- 5% of target investment (relative to initial long/short capital).
    - Penalization: For >10% change in total exposure (relative to initial total portfolio value) per step.
    - Episode: Random 6-month period.
    """
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}

    def __init__(
        self,
        features: np.ndarray,
        prices: np.ndarray,
        dates: np.ndarray = None,
        episode_length_months: int = 6,
        window_size: int = 5,
        dead_zone: float = 0.05, # Dead zone: 5%
        commission: float = 0.00125,
        render_mode: str = None,
    ):
        super().__init__()

        assert len(features) == len(prices), "Features and prices must have the same length."

        self.features = features
        self.prices = prices
        self.dates = dates if dates is not None else np.arange(len(prices))
        self.episode_length_months = episode_length_months
        self.window_size = window_size
        self.dead_zone = dead_zone
        self.commission = commission
        self.render_mode = render_mode

        # --- Capital Allocation ---
        self.initial_total_portfolio_value = 2_000_000.0
        self.initial_long_capital = 1_000_000.0
        self.initial_short_capital = 1_000_000.0

        self.episode_length_days = self.episode_length_months * 21 # ~21 trading days/month

        # --- Episode Start Index Range ---
        self.min_start_idx = self.window_size
        self.max_start_idx = len(self.features) - self.episode_length_days - 1
        if self.max_start_idx <= self.min_start_idx:
            raise ValueError(f"Not enough data for {episode_length_months}-month episodes with a window size of {window_size}.")

        # --- Action Space ---
        # 0.0: No investment, 1.0: 100% long capital, 2.0: 100% long + 100% short capital.
        self.action_space = spaces.Box(low=0.0, high=2.0, shape=(1,), dtype=np.float32)

        # --- Observation Space ---
        # Window of market features + 4 portfolio state variables.
        # The 4 portfolio state variables are: current_long_shares, current_short_shares, cash, portfolio_value
        n_features_total = self.features.shape[1] + 4 # CORRECTED: Added 4 for portfolio state variables
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(self.window_size, n_features_total),
            dtype=np.float32
        )

        # --- Episode & Portfolio State Variables (initialized in reset) ---
        self.current_episode_start_idx = 0
        self.current_step = 0
        self.episode_done = False

        self.portfolio_value = 0.0
        self.cash = 0.0
        self.current_long_shares = 0.0
        self.current_short_shares = 0.0
        self.last_total_exposure = 0.0

        # --- History Tracking ---
        self.portfolio_history = []
        self.long_shares_history = []
        self.short_shares_history = []
        self.cash_history = []
        self.action_history = []
        self.returns_history = []

        # Call reset at the end of init to set initial state
        self.reset()

    def reset(self, seed: int = None, options: dict = None) -> tuple[np.ndarray, dict]:
        """Resets the environment for a new episode."""
        # CORRECTED: Call super().reset() to handle the seed
        super().reset(seed=seed)

        # Use self.np_random for randomness, which is initialized by super().reset()
        self.current_episode_start_idx = self.np_random.integers(
            low=self.min_start_idx, high=self.max_start_idx
        )
        self.current_step = 0
        self.episode_done = False

        # --- Initialize Portfolio State ---
        self.portfolio_value = self.initial_total_portfolio_value
        self.cash = self.initial_total_portfolio_value
        self.current_long_shares = 0.0
        self.current_short_shares = 0.0
        self.last_total_exposure = 0.0 # Reset exposure for the new episode

        # --- Clear History ---
        self.portfolio_history = [self.portfolio_value]
        self.long_shares_history = [self.current_long_shares]
        self.short_shares_history = [self.current_short_shares]
        self.cash_history = [self.cash]
        self.action_history = []
        self.returns_history = []

        # CORRECTED: Return (observation, info) as required by Gymnasium
        return self._get_observation(), self._get_info()

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict]:
        """
        Executes one step in the environment given an agent's action.

        Returns: (observation, reward, terminated, truncated, info)
        """
        if self.episode_done:
            raise ValueError("Episode is done, call reset() to start a new one.")

        current_data_idx = self.current_episode_start_idx + self.current_step
        current_price = self.prices[current_data_idx]

        # Portfolio value before trade, used for step return calculation
        portfolio_value_before_trade = self.cash + \
                                       (self.current_long_shares * current_price) - \
                                       (self.current_short_shares * current_price)

        # Previous total exposure for penalty calculation
        previous_total_exposure = (self.current_long_shares * current_price) + \
                                  (self.current_short_shares * current_price)

        # --- Interpret and Clamp Action ---
        action_value = np.clip(action[0], self.action_space.low[0], self.action_space.high[0])

        target_long_investment = 0.0
        target_short_investment = 0.0

        if 0.0 <= action_value <= 1.0:
            target_long_investment = action_value * self.initial_long_capital
            target_short_investment = 0.0 # Close short positions
        elif 1.0 < action_value <= 2.0:
            target_long_investment = self.initial_long_capital # Max long
            target_short_investment = (action_value - 1.0) * self.initial_short_capital # Scale short

        target_long_shares = target_long_investment / current_price if current_price > 0 else 0.0
        target_short_shares = target_short_investment / current_price if current_price > 0 else 0.0

        # --- Dead Zone Logic ---
        desired_change_long_shares = target_long_shares - self.current_long_shares
        desired_change_short_shares = target_short_shares - self.current_short_shares

        # Apply dead zone based on desired change value vs initial capital allocation
        actual_change_long_shares = 0.0
        if abs(desired_change_long_shares * current_price) > (self.dead_zone * self.initial_long_capital):
            actual_change_long_shares = desired_change_long_shares

        actual_change_short_shares = 0.0
        if abs(desired_change_short_shares * current_price) > (self.dead_zone * self.initial_short_capital):
            actual_change_short_shares = desired_change_short_shares

        # Calculate actual shares to trade
        buy_shares = max(0.0, actual_change_long_shares)
        sell_shares = max(0.0, -actual_change_long_shares)
        open_short_shares = max(0.0, actual_change_short_shares)
        cover_short_shares = max(0.0, -actual_change_short_shares)

        # --- Execute Trades & Deduct Commissions ---
        # Update cash
        self.cash -= (buy_shares * current_price)
        self.cash += (sell_shares * current_price)
        self.cash += (open_short_shares * current_price)
        self.cash -= (cover_short_shares * current_price)

        # Calculate and deduct total commissions
        total_transaction_value_for_commissions = (buy_shares + sell_shares + open_short_shares + cover_short_shares) * current_price
        commissions_cost = total_transaction_value_for_commissions * self.commission
        self.cash -= commissions_cost

        # Update share positions
        self.current_long_shares += (buy_shares - sell_shares)
        self.current_short_shares += (open_short_shares - cover_short_shares)
        self.current_long_shares = max(0.0, self.current_long_shares)
        self.current_short_shares = max(0.0, self.current_short_shares)

        # --- Advance Time ---
        self.current_step += 1
        next_data_idx = self.current_episode_start_idx + self.current_step

        # --- Check Termination ---
        terminated = self.current_step >= self.episode_length_days or next_data_idx >= len(self.prices)
        truncated = False

        # --- Calculate Portfolio Value & Step Return ---
        # Use next_data_idx for portfolio value if not terminated, otherwise use current_price
        price_for_portfolio_value = self.prices[next_data_idx] if not terminated and next_data_idx < len(self.prices) else current_price
            
        self.portfolio_value = self.cash + \
                               (self.current_long_shares * price_for_portfolio_value) - \
                               (self.current_short_shares * price_for_portfolio_value)
        
        step_return = (self.portfolio_value - portfolio_value_before_trade) / portfolio_value_before_trade if portfolio_value_before_trade != 0 else 0

        # --- Penalize for Big Exposure Swings ---
        current_total_exposure = (self.current_long_shares * price_for_portfolio_value) + \
                                 (self.current_short_shares * price_for_portfolio_value)
        
        penalty_for_big_move = 0.0
        if self.current_step > 1: # Only penalize after the first step where prior exposure exists
            # Avoid division by zero if previous exposure was effectively zero
            if previous_total_exposure > 1e-6 or current_total_exposure > 1e-6:
                change_in_exposure_amount = abs(current_total_exposure - previous_total_exposure)
                # Normalize change by initial total capital, not just current exposure, for consistent penalty scale
                change_percentage = change_in_exposure_amount / self.initial_total_portfolio_value
                
                max_allowed_change_ratio = 0.1 # 10% max allowed change
                if change_percentage > max_allowed_change_ratio:
                    excess_change = change_percentage - max_allowed_change_ratio
                    penalty_for_big_move = -20.0 * excess_change
                    # Ensure penalty is negative
                    if penalty_for_big_move > 0: penalty_for_big_move *= -1
        
        self.last_total_exposure = current_total_exposure

        # --- Update Histories ---
        self.portfolio_history.append(self.portfolio_value)
        self.long_shares_history.append(self.current_long_shares)
        self.short_shares_history.append(self.current_short_shares)
        self.cash_history.append(self.cash)
        self.action_history.append(action_value)
        self.returns_history.append(step_return)

        # --- Calculate Reward ---
        reward = self._calculate_reward(step_return, self.portfolio_value, penalty_for_big_move)

        # --- Get Next Observation & Info ---
        obs = self._get_observation()
        # CORRECTED: Call _get_info() for consistent info dictionary
        info = self._get_info() 

        return obs, reward, terminated, truncated, info

    def _calculate_reward(self, step_return: float, current_portfolio_value: float, penalty_for_big_move: float = 0.0) -> float:
        """Calculates the reward for the current step, including penalties."""
        reward = step_return * 100.0 # Scale return for reward

        # --- Penalize Extreme Positions ---
        position_penalty = 0.0
        # Check current exposure relative to initial capital for long/short
        # Use price from the previous step (current_data_idx) or next step (next_data_idx) for consistency with portfolio value calculation
        # It's better to use the price at which the position was valued for the portfolio_value calculation.
        # The price for current_price was current_data_idx, but portfolio_value uses price_for_portfolio_value (next_data_idx)
        # For leverage, the price at the END of the step makes most sense for the current exposure ratio
        price_for_leverage_calc = self.prices[self.current_episode_start_idx + self.current_step] if (self.current_episode_start_idx + self.current_step) < len(self.prices) else self.prices[-1]

        long_exposure_ratio = (self.current_long_shares * price_for_leverage_calc) / self.initial_long_capital if self.initial_long_capital > 0 else 0
        short_exposure_ratio = (self.current_short_shares * price_for_leverage_calc) / self.initial_short_capital if self.initial_short_capital > 0 else 0

        leverage_threshold = 1.05 # E.g., 5% above 100% allocation
        if long_exposure_ratio > leverage_threshold:
            position_penalty -= 0.01 * (long_exposure_ratio - leverage_threshold)
        if short_exposure_ratio > leverage_threshold:
            position_penalty -= 0.01 * (short_exposure_ratio - leverage_threshold)

        reward += position_penalty
        reward += penalty_for_big_move # Add big move penalty
        
        return reward

    def _get_observation(self) -> np.ndarray:
        """Generates the current observation window for the agent."""
        end_idx = self.current_episode_start_idx + self.current_step
        start_idx = max(0, end_idx - self.window_size + 1)
        
        features_window = self.features[start_idx:end_idx + 1]

        # Pad with zeros if window is not full at the beginning
        if len(features_window) < self.window_size:
            padding = np.zeros((self.window_size - len(features_window), self.features.shape[1]), dtype=np.float32)
            features_window = np.vstack([padding, features_window])
        
        # --- CORRECTED: Add Portfolio State Variables (current state, replicated for window_size) ---
        # The 4 portfolio state variables are: current_long_shares, current_short_shares, cash, portfolio_value
        current_state_obs = np.array([
            self.current_long_shares,
            self.current_short_shares,
            self.cash,
            self.portfolio_value
        ], dtype=np.float32)

        # Tile the current state to match the window size for consistency
        current_state_window = np.tile(current_state_obs, (self.window_size, 1))

        # Concatenate market features with portfolio state variables
        observation = np.concatenate((features_window, current_state_window), axis=1)
        
        return observation.astype(np.float32)

    def _get_info(self) -> dict:
        """Provides additional environment information for logging/debugging."""
        current_data_idx = self.current_episode_start_idx + self.current_step
        if current_data_idx >= len(self.prices):
            current_data_idx = len(self.prices) - 1

        date_val = self.dates[current_data_idx]
        formatted_date = date_val.strftime("%Y-%m-%d") if isinstance(date_val, (datetime.date, datetime.datetime)) else date_val

        return {
            'current_date': formatted_date,
            'current_price': self.prices[current_data_idx],
            'portfolio_value': self.portfolio_value,
            'cash': self.cash,
            'current_long_shares': self.current_long_shares,
            'current_short_shares': self.current_short_shares,
            'last_total_exposure_for_penalty': self.last_total_exposure,
            'episode_step': self.current_step,
            'episode_length_days': self.episode_length_days
        }

    def get_episode_stats(self) -> dict:
        """Calculates and returns key performance statistics for the current episode."""
        if len(self.portfolio_history) < 2:
            return {}
        
        returns = np.array(self.returns_history)
        portfolio_values = np.array(self.portfolio_history)
        
        total_return = (portfolio_values[-1] - portfolio_values[0]) / portfolio_values[0]
        volatility = returns.std() if len(returns) > 1 else 0.0
        sharpe_ratio = returns.mean() / volatility if volatility > 0 else 0.0
        max_drawdown = self._calculate_max_drawdown(portfolio_values)
        
        # Count trades (significant changes in long/short shares)
        num_trades = sum(1 for i in range(1, len(self.long_shares_history))
                         if abs(self.long_shares_history[i] - self.long_shares_history[i-1]) > 1e-6 or \
                            abs(self.short_shares_history[i] - self.short_shares_history[i-1]) > 1e-6)

        return {
            'total_return': total_return,
            'volatility': volatility,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'final_portfolio_value': self.portfolio_history[-1], # Added final portfolio value back
            'final_long_shares': self.current_long_shares,
            'final_short_shares': self.current_short_shares,
            'final_cash': self.cash, # Added final cash back
            'num_trades': num_trades
        }

    def _calculate_max_drawdown(self, portfolio_values: np.ndarray) -> float:
        """Calculates the maximum drawdown for the portfolio value series."""
        peak = portfolio_values[0]
        max_dd = 0.0
        
        for value in portfolio_values:
            if value > peak:
                peak = value
            drawdown = (peak - value) / peak
            max_dd = max(max_dd, drawdown)
            
        return max_dd

    def render(self) -> None:
        """Renders the environment state. Prints portfolio info to console in 'human' mode."""
        if self.render_mode == 'human':
            current_data_idx = self.current_episode_start_idx + self.current_step
            if current_data_idx >= len(self.prices):
                current_data_idx = len(self.prices) - 1

            current_price = self.prices[current_data_idx]
            current_portfolio_value_for_display = self.cash + \
                                                  (self.current_long_shares * current_price) - \
                                                  (self.current_short_shares * current_price)
                                                  
            date_val = self.dates[current_data_idx]
            formatted_date = date_val.strftime("%Y-%m-%d") if isinstance(date_val, (datetime.date, datetime.datetime)) else date_val

            print(f"Date: {formatted_date}, "
                  f"Step: {self.current_step}/{self.episode_length_days}, "
                  f"Price: ${current_price:,.2f}, "
                  f"Portfolio: ${current_portfolio_value_for_display:,.2f}, "
                  f"Cash: ${self.cash:,.2f}, "
                  f"Long Shares: {self.current_long_shares:,.2f}, "
                  f"Short Shares: {self.current_short_shares:,.2f}")

    def close(self) -> None:
        """Closes any resources opened by the environment."""
        pass