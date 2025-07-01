import numpy as np
import gym
from gym import spaces

class PortfolioHedgingEnv(gym.Env):
    """
    Reworked Gym environment for portfolio hedging:
    - Action 0.0 = Maximum long hedge (buy additional shares)
    - Action 2.0 = Maximum short hedge (short against holdings)  
    - Action 1.0 = Neutral position
    - 15% reserve cash for additional purchases
    - 5% dead zone around 1.0 (0.95-1.05)
    - Short positions based on current total portfolio value
    - Reward based on hedging performance and position change penalty
    """

    def __init__(self, features: np.ndarray, prices: np.ndarray, dates: np.ndarray = None,
                 episode_length_months=6, window_size=5, dead_zone=0.05,
                 initial_portfolio=1_000_000, reserve_cash_pct=0.15, commission=0.00125,
                 max_position_change_penalty=0.20):
        super().__init__()
        
        assert len(features) == len(prices), "Features and prices must have same length"
        
        self.features = features
        self.prices = prices
        self.dates = dates if dates is not None else np.arange(len(prices))
        self.window_size = window_size
        self.dead_zone = dead_zone  # ±5% dead zone around 1.0
        self.initial_portfolio = initial_portfolio
        self.reserve_cash_pct = reserve_cash_pct
        self.commission = commission
        self.max_position_change_penalty = max_position_change_penalty
        
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
        
        # Portfolio state tracking
        self.initial_stock_value = 0.0  # Value of base stock holdings
        self.cash = 0.0
        self.additional_long_shares = 0.0  # Additional shares beyond base position
        self.short_shares = 0.0  # Shares sold short
        self.short_entry_price = 0.0  # Average entry price for short positions
        
        # Position tracking
        self.current_action = 1.0  # Start neutral
        self.previous_action = 1.0
        
        # History for analysis
        self.portfolio_history = []
        self.action_history = []
        self.position_history = []
        self.cash_history = []
        self.returns_history = []
        
        self.reset()

    def reset(self):
        """Reset environment for a new episode with random period"""
        # Randomly select start of episode
        self.current_episode_start = np.random.randint(
            self.min_start_idx, self.max_start_idx
        )
        self.current_step = 0
        self.episode_done = False
        
        # Initialize portfolio state
        initial_price = self.prices[self.current_episode_start]
        
        # 85% in stock, 15% in cash reserve
        stock_allocation = self.initial_portfolio * (1 - self.reserve_cash_pct)
        self.initial_stock_value = stock_allocation
        self.cash = self.initial_portfolio * self.reserve_cash_pct
        
        # Reset positions
        self.additional_long_shares = 0.0
        self.short_shares = 0.0
        self.short_entry_price = 0.0
        
        # Reset tracking
        self.current_action = 1.0
        self.previous_action = 1.0
        
        # Clear history
        initial_portfolio_value = self._calculate_portfolio_value()
        self.portfolio_history = [initial_portfolio_value]
        self.action_history = [1.0]
        self.position_history = [1.0]
        self.cash_history = [self.cash]
        self.returns_history = []
        
        return self._get_observation()

    def step(self, action):
        """Execute one step in the environment"""
        if self.episode_done:
            raise ValueError("Episode is done, call reset()")
        
        # Clamp action to valid range
        target_action = np.clip(action[0], 0.0, 2.0)
        
        # Apply dead zone - if within 5% of 1.0, no action
        if abs(target_action - 1.0) <= self.dead_zone:
            target_action = self.current_action  # Keep current position
            commission_cost = 0.0  # No trades, no commission
        else:
            # Execute position change
            commission_cost = self._execute_position_change(target_action)
        
        # Store previous portfolio value for return calculation
        current_idx = self.current_episode_start + self.current_step
        portfolio_value_before = self._calculate_portfolio_value()
        
        # Move to next step
        self.current_step += 1
        next_idx = self.current_episode_start + self.current_step
        
        # Check if episode is done
        if self.current_step >= self.episode_length or next_idx >= len(self.prices):
            self.episode_done = True
        
        # Calculate portfolio value after price change
        portfolio_value_after = self._calculate_portfolio_value()
        
        # Calculate step return
        step_return = (portfolio_value_after - portfolio_value_before) / portfolio_value_before
        self.returns_history.append(step_return)
        
        # Calculate benchmark return (buy and hold)
        if not self.episode_done:
            current_price = self.prices[current_idx]
            next_price = self.prices[next_idx]
            benchmark_return = (next_price - current_price) / current_price
        else:
            benchmark_return = 0.0
        
        # Calculate reward
        reward = self._calculate_reward(step_return, benchmark_return, target_action)
        
        # Update tracking
        self.previous_action = self.current_action
        self.current_action = target_action
        self.portfolio_history.append(portfolio_value_after)
        self.action_history.append(target_action)
        self.position_history.append(self._get_current_position_ratio())
        self.cash_history.append(self.cash)
        
        # Get next observation
        obs = self._get_observation()
        
        # Info dictionary
        info = {
            'portfolio_value': portfolio_value_after,
            'cash': self.cash,
            'action': target_action,
            'actual_position': self._get_current_position_ratio(),
            'step_return': step_return,
            'benchmark_return': benchmark_return,
            'commission_cost': commission_cost,
            'episode_step': self.current_step
        }
        
        return obs, reward, self.episode_done, info

    def _execute_position_change(self, target_action):
        """Execute the position change and return commission cost"""
        current_price = self.prices[self.current_episode_start + self.current_step]
        total_commission = 0.0
        
        # Calculate current portfolio value for short position sizing
        current_portfolio_value = self._calculate_portfolio_value()
        
        # Determine target position
        if target_action < 1.0:
            # Want to go long (action 0.0 = maximum long)
            long_percentage = (1.0 - target_action)  # 0.7 -> 30% additional long
            target_additional_value = long_percentage * current_portfolio_value
            
            # Check how much we can actually buy with available cash
            max_buyable_value = self.cash / (1 + self.commission)  # Account for commission
            actual_additional_value = min(target_additional_value, max_buyable_value)
            
            # Close any existing short positions first
            if self.short_shares > 0:
                total_commission += self._close_short_positions(current_price)
            
            # Buy additional shares if we have cash
            if actual_additional_value > 0:
                shares_to_buy = actual_additional_value / current_price
                commission_cost = actual_additional_value * self.commission
                
                self.additional_long_shares += shares_to_buy
                self.cash -= (actual_additional_value + commission_cost)
                total_commission += commission_cost
        
        elif target_action > 1.0:
            # Want to go short (action 2.0 = maximum short)
            short_percentage = (target_action - 1.0)  # 1.3 -> 30% short
            target_short_value = short_percentage * current_portfolio_value
            
            # Close any additional long positions first if needed
            if self.additional_long_shares > 0:
                total_commission += self._close_additional_long_positions(current_price)
            
            # Calculate how much we need to short
            current_short_value = self.short_shares * current_price
            additional_short_needed = target_short_value - current_short_value
            
            if additional_short_needed > 0:
                # Need to short more
                shares_to_short = additional_short_needed / current_price
                commission_cost = additional_short_needed * self.commission
                
                # Update short position with weighted average entry price
                if self.short_shares > 0:
                    total_short_value = (self.short_shares * self.short_entry_price + 
                                       shares_to_short * current_price)
                    total_short_shares = self.short_shares + shares_to_short
                    self.short_entry_price = total_short_value / total_short_shares
                else:
                    self.short_entry_price = current_price
                
                self.short_shares += shares_to_short
                self.cash += (additional_short_needed - commission_cost)  # Receive cash from short sale
                total_commission += commission_cost
            
            elif additional_short_needed < 0:
                # Need to cover some short positions
                shares_to_cover = abs(additional_short_needed) / current_price
                shares_to_cover = min(shares_to_cover, self.short_shares)
                
                if shares_to_cover > 0:
                    cover_value = shares_to_cover * current_price
                    commission_cost = cover_value * self.commission
                    
                    # Calculate P&L from covering
                    pnl = shares_to_cover * (self.short_entry_price - current_price)
                    
                    self.short_shares -= shares_to_cover
                    self.cash -= (cover_value + commission_cost - pnl)
                    total_commission += commission_cost
                    
                    # Reset entry price if all positions closed
                    if self.short_shares <= 0:
                        self.short_entry_price = 0.0
        
        return total_commission

    def _close_short_positions(self, current_price):
        """Close all short positions"""
        if self.short_shares <= 0:
            return 0.0
        
        cover_value = self.short_shares * current_price
        commission_cost = cover_value * self.commission
        
        # Calculate P&L
        pnl = self.short_shares * (self.short_entry_price - current_price)
        
        # Update cash and positions
        self.cash -= (cover_value + commission_cost - pnl)
        self.short_shares = 0.0
        self.short_entry_price = 0.0
        
        return commission_cost

    def _close_additional_long_positions(self, current_price):
        """Close all additional long positions"""
        if self.additional_long_shares <= 0:
            return 0.0
        
        sale_value = self.additional_long_shares * current_price
        commission_cost = sale_value * self.commission
        
        # Update cash and positions
        self.cash += (sale_value - commission_cost)
        self.additional_long_shares = 0.0
        
        return commission_cost

    def _calculate_portfolio_value(self):
        """Calculate total portfolio value"""
        current_price = self.prices[self.current_episode_start + self.current_step]
        
        # Base stock holdings value
        base_shares = self.initial_stock_value / self.prices[self.current_episode_start]
        base_value = base_shares * current_price
        
        # Additional long positions value
        additional_long_value = self.additional_long_shares * current_price
        
        # Short positions P&L (add to cash conceptually)
        if self.short_shares > 0:
            short_pnl = self.short_shares * (self.short_entry_price - current_price)
        else:
            short_pnl = 0.0
        
        total_value = base_value + additional_long_value + self.cash + short_pnl
        return total_value

    def _get_current_position_ratio(self):
        """Get current position as a ratio relative to neutral (1.0)"""
        current_price = self.prices[self.current_episode_start + self.current_step]
        current_portfolio_value = self._calculate_portfolio_value()
        
        if current_portfolio_value <= 0:
            return 1.0
        
        # Calculate additional long exposure
        additional_long_value = self.additional_long_shares * current_price
        additional_long_ratio = additional_long_value / current_portfolio_value
        
        # Calculate short exposure
        short_value = self.short_shares * current_price
        short_ratio = short_value / current_portfolio_value
        
        # Position ratio: 1.0 = neutral, <1.0 = net long, >1.0 = net short
        position_ratio = 1.0 - additional_long_ratio + short_ratio
        return position_ratio

    def _calculate_reward(self, step_return, benchmark_return, current_action):
        """Calculate reward based on hedging performance and position change penalty"""
        # Hedging performance: how much better/worse than buy-and-hold
        hedging_performance = step_return - benchmark_return
        
        # Position change penalty
        position_change = abs(current_action - self.previous_action)
        if position_change > self.max_position_change_penalty:
            position_penalty = (position_change - self.max_position_change_penalty) * 0.01
        else:
            position_penalty = 0.0
        
        # Combine rewards
        reward = hedging_performance - position_penalty
        
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
        
        # Calculate benchmark returns
        start_price = self.prices[self.current_episode_start]
        end_price = self.prices[self.current_episode_start + len(self.returns_history)]
        benchmark_total_return = (end_price - start_price) / start_price
        
        # Portfolio performance
        total_return = (portfolio_values[-1] - portfolio_values[0]) / portfolio_values[0]
        excess_return = total_return - benchmark_total_return
        
        volatility = returns.std() if len(returns) > 1 else 0.0
        sharpe_ratio = returns.mean() / volatility if volatility > 0 else 0.0
        max_drawdown = self._calculate_max_drawdown(portfolio_values)
        
        # Trading behavior
        position_changes = np.diff(self.action_history)
        avg_position_change = np.mean(np.abs(position_changes)) if len(position_changes) > 0 else 0.0
        num_significant_trades = np.sum(np.abs(position_changes) > self.dead_zone)
        
        return {
            'total_return': total_return,
            'benchmark_return': benchmark_total_return,
            'excess_return': excess_return,
            'volatility': volatility,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'final_action': self.current_action,
            'avg_position_change': avg_position_change,
            'num_significant_trades': num_significant_trades,
            'final_cash': self.cash,
            'final_portfolio_value': portfolio_values[-1]
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
            current_value = self.portfolio_history[-1] if self.portfolio_history else self.initial_portfolio
            current_pos = self._get_current_position_ratio()
            print(f"Step: {self.current_step}, Portfolio: ${current_value:,.2f}, "
                  f"Action: {self.current_action:.3f}, Position: {current_pos:.3f}, "
                  f"Cash: ${self.cash:,.2f}")