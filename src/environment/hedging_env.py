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
        dead_zone: float = 0.03, 
        commission: float = 0.00125, 
        initial_capital: float = 2_000_000.0,
        render_mode: str | None = "human"
    ):
        super().__init__()
        
        if len(features) != len(prices):
            raise ValueError("Features and prices must have the same length. Your features length is {}, and prices length is {}.".format(len(features), len(prices)))
        
        assert len(features) > window_size, f"Features length ({len(features)}) must be greater than window_size ({window_size})"
        assert all(prices >= 0), "Prices must be non-negative"
        
        self.features = features
        self.prices = prices
        self.episode_months = episode_months
        self.window_size = window_size
        self.dead_zone = dead_zone
        self.commission = commission
        self.initial_capital = initial_capital
        self.render_mode = render_mode
        
        self.episode_days = episode_months * 21 # Assuming 21 trading days per month
        
        self.action_space = spaces.Box(low=0.0, high=2.0, shape=(1,), dtype=np.float32)
        self.observation_space = spaces.Box(
            low=-np.inf, 
            high=np.inf, 
            shape=(self.window_size, self.features.shape[1]  + 4), # + 4 for additional portfolio state variables: current_long_sahres, current_short_shares, cash, portfolio_value 
            dtype=np.float32
        )      
        
        self.min_start_index = self.window_size
        self.max_start_index = len(self.prices) - self.episode_days - 1
        if self.min_start_index > self.max_start_index:
            raise ValueError("The length of the prices array is too short for {} months episode and window size of {}.".format(episode_months, window_size))
        
        self.start_index = None
        self.current_index = None
        self.episode_done = False
        
        self.total_cash = self.initial_capital
        self.long_cash = self.total_cash / 2
        self.short_cash = self.total_cash / 2
        
        self.long_shares = 0
        self.short_shares = 0
        
        self.refresh_portfolio(price = 0.0)
        
        
        self.historical_portfolio = []
        self.historical_long_shares = []
        self.historical_short_shares = []
        self.historical_cash = []
        self.historical_actions = []
        self.historical_returns = []
        
        self.reset()
        
        
    def reset(self, seed: int | None = None):
        super().reset(seed=seed)
        
        self.start_index = self.np_random.integers(
            low=self.min_start_index, 
            high=self.max_start_index + 1
        )
        
        self.current_index = self.start_index
        self.episode_done = False
        
        self.total_cash = self.initial_capital
        self.long_cash = self.total_cash / 2
        self.short_cash = self.total_cash / 2
        
        self.long_shares = 0
        self.short_shares = 0
        
        self.refresh_portfolio()
        
        self.historical_portfolio = []
        self.historical_long_shares = []
        self.historical_short_shares = []
        self.historical_cash = []
        self.historical_actions = []
        self.historical_returns = []
        
        # Start with 100% long, no hedging
        self.trade(action=1.0)
        self.update_historical(action=1.0, returnp=0.00)
        
        return self._get_observation(), self._get_info()
        
    
    def trade(self, action: float):
        
        # Get current price based on the current index
        current_price = self.prices[self.current_index]
        
        # Get the cost of share
        cost_per_share = current_price * (1 + self.commission)
            
        # Refresh values
        self.refresh_portfolio()

        if action >= 1:
            
            # Open all long positions:
            shares_to_buy = int(self.long_cash / cost_per_share)
            position_to_buy = shares_to_buy * cost_per_share
            
            self.long_cash -= position_to_buy
            self.long_shares += shares_to_buy
            
            # Open <action>% of short 
            
            if action != 1:
                 
                # Check if we need to sell in short more or buy to cover some
                shares_to_hold = int((self.short_portfolio_value * action) / current_price)
                if self.short_shares > shares_to_hold:
                    
                    # Buy to cover some
                    shares_to_buy_to_cover = self.short_shares - shares_to_hold
                    position_to_buy_to_cover = shares_to_buy_to_cover * cost_per_share
                    
                    # Check if I have enough cash to deduct commissions
                    if position_to_buy_to_cover > self.short_cash:
                        missing_shares = np.ceil((position_to_buy_to_cover - self.short_cash) / cost_per_share) # Number of shares that we don't have money for
                        shares_to_buy_to_cover -= missing_shares
                        position_to_buy_to_cover = shares_to_buy_to_cover * cost_per_share
                        
                    self.short_cash -= position_to_buy_to_cover
                    self.short_shares -= shares_to_buy_to_cover
                
                elif self.short_shares < shares_to_hold:
                    
                    # Sell in short more
                    shares_to_short = shares_to_hold - self.short_shares
                    position_to_short = shares_to_short * current_price
                    
                    self.short_cash += position_to_short * (1 - self.commission)
                    self.short_shares += shares_to_short
                
            
        else:
            
            # Close all short positions
            shares_to_buy_to_cover = self.short_shares
            position_to_buy_to_cover = shares_to_buy_to_cover * cost_per_share
            
            self.short_cash -= position_to_buy_to_cover
            self.short_shares = 0
            
            # Open <action>% of long positions
            
            # Check if we need to buy more or sell some
            shares_to_hold = int((self.long_portfolio_value * action) / current_price)
            if self.long_shares > shares_to_hold:
                
                # Sell some
                shares_to_sell = self.long_shares - shares_to_hold
                position_to_sell = shares_to_sell * current_price
                
                self.long_cash += position_to_sell * (1 - self.commission)
                self.long_shares -= shares_to_sell
                
            elif self.long_shares < shares_to_hold:
                # Buy more
                shares_to_buy = shares_to_hold - self.long_shares
                position_to_buy = shares_to_buy * cost_per_share
                
                # Check if I have enough cash to deduct commissions
                if position_to_buy > self.long_cash:
                    missing_shares = np.ceil((position_to_buy - self.long_cash) / cost_per_share) # Number of shares that we don't have money for
                    shares_to_buy -= missing_shares
                    position_to_buy = shares_to_buy * cost_per_share
                
                self.long_cash -= position_to_buy
                self.long_shares += shares_to_buy
                 
        self.refresh_portfolio(price = current_price)
        
        
            
    def update_historical(self, action: float, returnp: float):
        
        self.historical_portfolio.append(self.total_portfolio_value)
        self.historical_long_shares.append(self.long_shares)
        self.historical_short_shares.append(self.short_shares)
        self.historical_cash.append(self.total_cash)
        self.historical_actions.append(action)
        self.historical_returns.append(returnp)
        
            
    def refresh_portfolio(self, price: float | None = None):
        """
        Refresh portfolio values based on current shares and cash.
        """
        if price is None:
            price = self.prices[self.current_index]
        
        self.long_position = self.long_shares * price
        self.short_position =  self.short_shares * price
        
        self.long_portfolio_value = self.long_cash + self.long_position
        self.short_portfolio_value = self.short_cash - self.short_position
        self.total_portfolio_value = self.long_portfolio_value + self.short_portfolio_value
            
            
            
    
    def step(
        self, action: np.ndarray
    ) -> tuple[np.ndarray, float, bool, bool, dict]:
        
        if self.episode_done:
            raise RuntimeError("Episode is done. Please reset the environment.")
        
        # Clip the action to fit the space box
        action = np.clip(action[0], self.action_space.low[0], self.action_space.high[0])
        
        # Get the difference with respect to the previous action
        diff_action = abs(action - self.historical_actions[-1])
        
        # Trade only if the difference is higher than the dead zone
        if diff_action > self.dead_zone:
            self.trade(action)
        
        # Take a step
        self.current_index += 1
        self.episode_done = self.current_index >= (self.start_index + self.episode_days)
        
        # Get the return in the portfolio value
        self.refresh_portfolio()
        step_return = (self.total_portfolio_value - self.historical_portfolio[-1]) / self.historical_portfolio[-1]
        
        # Update historical data
        self.update_historical(action, step_return)
        
            
        reward = self._calculate_reward(step_return, diff_action)
        obs = self._get_observation()
        info = self._get_info()
        
        return obs, reward, self.episode_done, False, info
    
    
    def _calculate_reward(self, step_return: float, diff_action: float) -> float:
    
        reward = step_return * 5.0
        
        if step_return > 0:
            reward += 0.1
        
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
        
        obs = obs.astype(np.float32)
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
            "portfolio": self.historical_portfolio,
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
            short_portfolio_value = -(self.short_shares * current_price) + self.short_cash   
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
            
