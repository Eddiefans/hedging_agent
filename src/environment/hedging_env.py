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
        render_mode: str | None = "human",
        curriculum_stage: str = "random"
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
            shape=(self.window_size * self.features.shape[1] + 5,), # Flattened window + 5 stats
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
        
        self.curriculum_stage = curriculum_stage
        self.curriculum_episodes = []  
        self._prepare_curriculum_episodes()  
        
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
        
        if self.curriculum_episodes:
            episode_idx = self.np_random.choice(len(self.curriculum_episodes))
            self.start_index = self.curriculum_episodes[episode_idx]
        else:
            # Fallback to normal
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
        
            
        reward = self._calculate_reward(step_return, action, diff_action)
        obs = self._get_observation()
        info = self._get_info()
        
        return obs, reward, self.episode_done, False, info
    
    
    def _calculate_reward(self, step_return, action, diff_action):
        
        # Base return
        base_reward = step_return * 10.0
        
        # Benchmark comparison
        benchmark_return = (self.prices[self.current_index] - self.prices[self.current_index-1]) / self.prices[self.current_index-1]
        outperformance_bonus = (step_return - benchmark_return) * 15.0
        
        # Market regime awareness
        lookback = min(20, self.current_index)  # 20-day rolling window
        if lookback > 1:
            recent_prices = self.prices[self.current_index-lookback:self.current_index+1]
            returns = np.diff(recent_prices) / recent_prices[:-1]
            market_volatility = np.std(returns) * np.sqrt(252)  # Annualized volatility
        else:
            market_volatility = 0.15  # Default to low volatility
        
        if market_volatility > 0.25:  # High volatility market
            # Reward hedging in volatile markets
            hedging_bonus = (2.0 - action) * 0.3 if action > 1.0 else 0
        else:  # Low volatility market  
            # Penalize over-hedging in calm markets
            hedging_penalty = -abs(action - 1.0) * 0.5
            hedging_bonus = hedging_penalty
        
        # Position change penalty (prevent overtrading)
        position_change = abs(diff_action)
        if position_change > self.dead_zone:
            trading_penalty = -position_change * 0.2
        else:
            trading_penalty = 0
        
        return base_reward + outperformance_bonus + hedging_bonus + trading_penalty
    
    
    def _get_observation(self):
        start_index = self.current_index - self.window_size + 1
        end_index = self.current_index
        
        # Get features window and flatten it
        features_window = self.features[start_index:end_index + 1]
        flattened_features = features_window.flatten()
        
        # Get current stats (5 values: portfolio, long_shares, short_shares, cash, last_action)
        current_stats = np.array([
            self.total_portfolio_value,
            self.long_shares,
            self.short_shares,
            self.total_cash,
            self.historical_actions[-1] if self.historical_actions else 1.0
        ], dtype=np.float32)
        
        # Concatenate flattened features + stats
        obs = np.concatenate([flattened_features, current_stats])
        
        return obs.astype(np.float32)
    
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
        
        risk_free_rate = 0.0823 # TIE 17 Jul 2025
        daily_risk_free_rate = (1 + risk_free_rate) ** (1/252) - 1 # 252 trading days in a year
        
        target_return = daily_risk_free_rate 
        
        returns = np.array(self.historical_returns)
        portfolio_values = np.array(self.historical_portfolio)
        
        total_return = (self.historical_portfolio[-1] - self.historical_portfolio[0]) / self.historical_portfolio[0]
        periods_per_year = 12 / self.episode_months
        annualized_return = (1 + total_return ) ** periods_per_year - 1
        
        benchmark_return = (self.prices[self.current_index] - self.prices[self.start_index]) / self.prices[self.start_index]
        annualized_benchmark_return = (1 + benchmark_return ) ** periods_per_year - 1
        
        # Benchmark calculations
        benchmark_prices = self.prices[self.start_index:self.current_index+1]
        benchmark_returns = np.diff(benchmark_prices) / benchmark_prices[:-1]  # Daily returns
        benchmark_volatility = benchmark_returns.std() if len(benchmark_returns) > 1 else 0.0
        benchmark_sharpe_ratio = (benchmark_returns.mean() - daily_risk_free_rate) / benchmark_volatility if benchmark_volatility > 0 else 0.0
        
        # Benchmark Sortino
        benchmark_downside_returns = np.minimum(benchmark_returns - target_return, 0.0)
        benchmark_downside_deviation = np.sqrt(np.mean(benchmark_downside_returns**2)) if len(benchmark_returns) > 1 else 0.0
        benchmark_sortino_ratio = (benchmark_returns.mean() - target_return) / benchmark_downside_deviation if benchmark_downside_deviation > 0 else 0.0
        
        # Benchmark Max Drawdown
        benchmark_peak = np.maximum.accumulate(benchmark_prices)
        benchmark_drawdown = (benchmark_prices - benchmark_peak) / benchmark_peak
        benchmark_max_drawdown = np.abs(benchmark_drawdown.min()) if len(benchmark_drawdown) > 0 else 0.0
        
        volatility = returns.std() if len(returns) > 1 else 0.0
        sharpe_ratio = (returns.mean() - daily_risk_free_rate) / volatility if volatility > 0 else 0.0
        
        # Downside deviation (target = risk free rate)
        downside_returns = np.minimum(returns - target_return, 0.0)  # Only negative deviations
        downside_deviation = np.sqrt(np.mean(downside_returns**2)) if len(returns) > 1 else 0.0
        
        # Sortino ratio
        sortino_ratio = (returns.mean() - target_return) / downside_deviation if downside_deviation > 0 else 0.0
        
        # Max drawdown
        peak = np.maximum.accumulate(portfolio_values)
        drawdown = (portfolio_values - peak) / peak
        max_drawdown = np.abs(drawdown.min()) if len(drawdown) > 0 else 0.0
        
        num_trades = sum(1 for i in range(1, len(self.historical_long_shares))
                         if abs(self.historical_long_shares[i] - self.historical_long_shares[i-1]) > 1e-6 or \
                            abs(self.historical_short_shares[i] - self.historical_short_shares[i-1]) > 1e-6)
        
        return {
            "total_return": total_return,
            "annualized_return": annualized_return,
            "benchmark_return": benchmark_return, 
            "annualized_benchmark_return": annualized_benchmark_return,
            "benchmark_sharpe_ratio": benchmark_sharpe_ratio, 
            "benchmark_sortino_ratio": benchmark_sortino_ratio, 
            "benchmark_max_drawdown": benchmark_max_drawdown, 
            "benchmark_volatility": benchmark_volatility,
            "volatility": volatility,
            "sharpe_ratio": sharpe_ratio,
            "num_trades": num_trades,
            "max_drawdown": max_drawdown, 
            "sortino_ratio": sortino_ratio,
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
            short_portfolio_value = -(self.short_shares * current_price) + self.short_cash   
            portfolio_value =  long_portfolio_value + short_portfolio_value
            
            print(f"Current index: {index}, "
                  f"Step: {self.current_index - self.start_index}/{self.episode_days}, "
                  f"Price: ${current_price:,.2f}, "
                  f"Portfolio: ${portfolio_value:,.2f}, "
                  f"Cash: ${self.total_cash:,.2f}, "
                  f"Long Shares: {self.long_shares:,.2f}, "
                  f"Short Shares: {self.short_shares:,.2f}")
            
            
    def _prepare_curriculum_episodes(self):
        """Prepare valid episode indices based on curriculum stage."""
        if self.curriculum_stage == "random":
            # Use all available episodes
            self.curriculum_episodes = list(range(self.min_start_index, self.max_start_index + 1))
            return
        
        # Analyze each potential 6-month period to classify market regime
        episode_length = self.episode_days
        valid_episodes = []
        
        for start_idx in range(self.min_start_index, self.max_start_index + 1):
            end_idx = min(start_idx + episode_length, len(self.prices) - 1)
            
            # Calculate period return
            start_price = self.prices[start_idx]
            end_price = self.prices[end_idx]
            period_return = (end_price - start_price) / start_price
            
            # Calculate volatility during period
            period_prices = self.prices[start_idx:end_idx+1]
            period_returns = np.diff(period_prices) / period_prices[:-1]
            volatility = np.std(period_returns) * np.sqrt(252)  # Annualized
            
            # Classify market regime
            is_bull = period_return > 0.15 and volatility < 0.4  # Strong positive return, low volatility
            is_bear = period_return < -0.10  # Negative return
            is_mixed = not is_bull and not is_bear  # Everything else
            
            # Add to curriculum episodes based on stage
            if (self.curriculum_stage == "bull" and is_bull) or \
               (self.curriculum_stage == "bear" and is_bear) or \
               (self.curriculum_stage == "mixed" and is_mixed):
                valid_episodes.append(start_idx)
        
        self.curriculum_episodes = valid_episodes
        
        # Fallback to random if no episodes found for the stage
        if not self.curriculum_episodes:
            print(f"Warning: No episodes found for {self.curriculum_stage} stage, using random sampling")
            self.curriculum_episodes = list(range(self.min_start_index, self.max_start_index + 1))
    
    
    def set_curriculum_stage(self, stage: str):
        self.curriculum_stage = stage
        self._prepare_curriculum_episodes()
        print(f"Curriculum stage set to '{stage}' with {len(self.curriculum_episodes)} available episodes")


    def close(self) -> None:
        if self.render_mode == "human":
            print("Environment closed.")
            
