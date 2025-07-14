import gymnasium as gym 
from gymnasium import spaces
import numpy as np 
import pandas as pd
import datetime

class PortfolioHedgingEnv(gym.Env):
    def __init__(
        self, 
        features: np.ndarray, 
        prices: np.ndarray, 
        episode_months: int = 6, 
        window_size: int = 5, 
        dead_zone: float = 0.01, 
        commission: float = 0.00125, 
        initial_capital: float = 2_000_000.0,
        render_mode: str | None = "human"
    ):
        super().__init__()
        
        if len(features) != len(prices):
            raise ValueError("Features and prices must have the same length. Your features length is {}, and prices length is {}.".format(len(features), len(prices)))
        
        assert len(features) > window_size, f"Features length ({len(features)}) must be greater than window_size ({window_size})"
        assert all(prices >= 0), "Prices must be non-negative"
        
        self.features = features, 
        self.prices = prices, 
        self.episode_months = episode_months
        self.window_size = window_size, 
        self.dead_zone = dead_zone,
        self.commission = commission, 
        self.initial_capital = initial_capital
        self.render_mode = render_mode
        
        self.episode_days = episode_months * 21 # Assuming 21 trading days per month
        
        self.action_space = spaces.Box(low=0.0, high=2.0, shape=(1,), dtype=np.float32)
        self.observation_space = spaces.Box(
            low=-np.inf, 
            high=np.inf, 
            shape=(self.window_size, len(self.features[0]) + 4), # + 4 for additional portfolio state variables: current_long_sahres, current_short_shares, cash, portfolio_value 
            dtype=np.float32
        )      
        
        self.min_start_index = self.window_size
        self.max_start_index = len(self.prices) - self.episode_days - 1
        if self.min_start_index > self.max_start_index:
            raise ValueError("The length of the prices array is too short for {} months episode and window size of {}.".format(episode_months, window_size))
        
        self.start_index = None
        self.current_index = None
        self.episode_done = False
        
        self.total_portfolio_value = self.initial_capital
        self.long_portfolio_value = self.total_portfolio_value / 2
        self.short_portfolio_value = self.total_portfolio_value / 2
        
        self.total_cash = self.initial_capital
        self.long_cash = self.total_capital / 2
        self.short_cash = self.total_capital / 2
        
        self.long_shares = 0
        self.short_shares = 0
        
        self.historical_portfolio = [self.total_portfolio_value]
        self.historical_long_shares = [0]
        self.historical_short_shares = [0]
        self.historical_cash = [self.total_cash]
        self.historical_actions = []
        self.historical_returns = []
        
        self.reset()
        
        
    def reset(self, seed : int = 42):
        super().reset(seed=seed)
        
        self.start_index = self.np_random.integers(
            low = self.min_start_index, high = self.max_start_index
        )
        self.current_index = self.start_index
        self.episode_done = False
        
        self.total_portfolio_value = self.initial_capital
        self.long_portfolio_value = self.total_portfolio_value / 2
        self.short_portfolio_value = self.total_portfolio_value / 2
        
        self.total_cash = self.initial_capital
        self.long_cash = self.total_capital / 2
        self.short_cash = self.total_capital / 2
        
        self.long_shares = 0
        self.short_shares = 0
        
        self.historical_portfolio = [self.total_portfolio_value]
        self.historical_long_shares = [0]
        self.historical_short_shares = [0]
        self.historical_cash = [self.total_cash]
        self.historical_actions = []
        self.historical_returns = []
        
        return self._get_observation(), self._get_info()
    
    
    def step(
        self, action: np.ndarray
    ) -> tuple[np.ndarray, float, bool, bool, dict]:
        
        if self.episode_done:
            raise RuntimeError("Episode is done. Please reset the environment.")
        
        current_price = self.prices[self.current_index]
        long_portfolio_before = current_price * self.long_shares + self.long_cash
        short_portfolio_before = current_price * self.short_shares + self.short_cash
        portfolio_before = long_portfolio_before + short_portfolio_before
        
        action = np.clip(action[0], self.action_space.low[0], self.action_space.high[0])
        
        trade = True
        
        if self.historical_actions:
            last_action = self.historical_actions[-1]
            if abs(action - last_action) < self.dead_zone:
                trade = False
                action = last_action
            diff_action = abs(action - last_action)
        else: 
            diff_action = 0
        
        if trade:
            
            if action <= 1:
                
                target_long_shares = int(action * long_portfolio_before / current_price)
                target_short_shares = 0
            else:
                target_long_shares = int(long_portfolio_before / current_price)
                target_short_shares = int((action - 1) * short_portfolio_before / current_price)
                
            long_shares_to_trade = target_long_shares - self.long_shares
            short_shares_to_trade = target_short_shares - self.short_shares
            
            long_volume_to_trade = long_shares_to_trade * current_price
            short_volume_to_trade = short_shares_to_trade * current_price
            
            long_commission_to_pay = abs(long_volume_to_trade) * self.commission
            short_commission_to_pay = abs(short_volume_to_trade) * self.commission
            
            self.long_cash -= (long_volume_to_trade + long_commission_to_pay)
            self.short_cash += (short_volume_to_trade - short_commission_to_pay)
            
            self.long_shares += long_shares_to_trade
            self.short_shares += short_shares_to_trade
            
            self.total_cash = self.long_cash + self.short_cash
            
        self.current_index += 1
        self.episode_done = self.current_index >= (self.start_index + self.episode_days)
        
        next_price = self.prices[self.current_index]
        self.long_portfolio_value = self.long_shares * next_price + self.long_cash
        self.short_portfolio_value = self.short_shares * next_price + self.short_cash   
        self.total_portfolio_value = self.long_portfolio_value + self.short_portfolio_value
        
        step_return = (self.total_portfolio_value - portfolio_before) / self.portfolio_before
        
        self.historical_portfolio.append(self.total_portfolio_value)
        self.historical_long_shares.append(self.long_shares)
        self.historical_short_shares.append(self.short_shares)
        self.historical_cash.append(self.total_cash)
        self.historical_actions.append(action)
        self.historical_returns.append(step_return)
            
        reward = self._calculate_reward(step_return, diff_action)
        obs = self._get_observation()
        info = self._get_info()
        
        return obs, reward, self.episode_done, False, info
        
    
    def _calculate_reward(
        self, step_return: float, diff_action: float
    ) -> float:
        
        reward = step_return * 100.0
        if abs(diff_action) > 0.1:
            reward -= abs(diff_action) * 10.0
            
        return reward
    
    
    def _get_observation(self):
        start_index = self.current_index - self.window_size + 1
        end_index = self.current_index
        
        features_window = self.features[start_index:end_index + 1]
        # features_window = (features_window - features_window.mean(axis=0)) / (features_window.std(axis=0) + 1e-8)
        
        if len(self.historical_portfolio) >= self.window_size:
            stats_window = np.column_stack([
                self.historical_portfolio[-self.window_size:],
                self.historical_long_shares[-self.window_size:],
                self.historical_short_shares[-self.window_size:],
                self.historical_cash[-self.window_size:]
            ])
        else:
            stats_window = np.column_stack([
                self.historical_portfolio,
                self.historical_long_shares,
                self.historical_short_shares,
                self.historical_cash
            ])
            padding = np.array([[
                    self.historical_portfolio[0], 
                    self.historical_long_shares[0], 
                    self.historical_short_shares[0], 
                    self.historical_cash[0]
                ]] * (self.window_size - len(self.historical_portfolio)), 
                dtype=np.float32
            )
            stats_window = np.vstack((padding, stats_window))
        
        obs = np.hstack((
            features_window,
            stats_window
        ))
        
        return obs
    
    def _get_info(self) -> dict:
        index = self.current_index
        if index >= len(self.prices):
            index = len(self.prices) - 1
        
        return {
            "current_index": index,
            "current_price": self.prices[index],
            "portfolio_value": self.total_portfolio_value,
            "cash": self.total_cash,
            "long_shares": self.long_shares,
            "short_shares": self.short_shares,
            "episode_step": self.current_index - self.start_index,
            "episode_length_days": self.episode_days
        }
        
    
    def get_episode_stats(self):
        
        returns = np.array(self.historical_returns)
        
        total_return = (self.historical_portfolio[-1] - self.historical_portfolio[0]) / self.historical_portfolio[0]
        volatility = returns.std() if len(returns) > 1 else 0.0
        sharpe_ratio = (returns.mean() / volatility) if volatility > 0 else 0.0
        num_trades = sum(1 for i in range(1, len(self.historical_long_shares))
                         if abs(self.historical_long_shares[i] - self.historical_long_shares[i-1]) > 1e-6 or \
                            abs(self.historical_short_shares[i] - self.historical_short_shares[i-1]) > 1e-6)
        
        return {
            "total_return": total_return,
            "volatility": volatility,
            "sharpe_ratio": sharpe_ratio,
            "num_trades": num_trades,
            "final_portfolio_value": self.historical_portfolio[-1],
            "final_cash": self.total_cash,
            "final_long_shares": self.long_shares,
            "final_short_shares": self.short_shares
        }



    def render(self) -> None:
        if self.render_mode == "human":
            index = self.current_index
            if index >= len(self.prices):
                index = len(self.prices) - 1
                
            current_price = self.prices[index]
            long_portfolio_value = self.long_shares * current_price + self.long_cash
            short_portfolio_value = self.short_shares * current_price + self.short_cash   
            portfolio_value =  long_portfolio_value + short_portfolio_value
            
            print(f"Current index: {index}, "
                  f"Step: {self.current_index - self.start_index}/{self.episode_days}, "
                  f"Price: ${current_price:,.2f}, "
                  f"Portfolio: ${portfolio_value:,.2f}, "
                  f"Cash: ${self.total_cash:,.2f}, "
                  f"Long Shares: {self.long_shares:,.2f}, "
                  f"Short Shares: {self.short_shares:,.2f}")


    def close(self) -> None:
        if self.render_mode == "human":
            print("Environment closed.")