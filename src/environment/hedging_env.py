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
        dates: np.ndarray = None,
        episode_months: int = 6, 
        window_size: int = 5, 
        dead_zone: float = 0.01, 
        commission: float = 0.00125, 
        initial_capital: float = 2_000_000.0,
        render_mode: str | None = "human",
        action_change_penalty_threshold=0.2,
        max_shares_per_trade=0.5 
    ):
        super().__init__()
        
        # Validate input data and parameters
        assert len(features) == len(prices), "Features and prices must have the same length"
        assert len(features) > window_size, f"Features length ({len(features)}) must be greater than window_size ({window_size})"
        assert np.all(prices >= 0), "Prices must be non-negative"
        assert initial_capital >= 0, "Initial capital must be positive or zero"
        assert 0 <= commission <= 1, "Commission must be between 0 and 1"
        assert 0 < max_shares_per_trade <= 1, "max_shares_per_trade must be between 0 and 1"
        assert dead_zone >= 0, "Dead zone must be non-negative"
        
        # Store environment parameters
        self.features = features
        self.prices = prices
        self.dates = pd.to_datetime(dates) if dates is not None else np.arange(len(prices))
        self.episode_months = episode_months
        self.window_size = window_size
        self.dead_zone = dead_zone
        self.commission = commission
        self.initial_capital = initial_capital
        self.initial_long_capital = initial_capital / 2
        self.initial_short_capital = initial_capital / 2
        self.action_change_penalty_threshold = action_change_penalty_threshold
        self.max_shares_per_trade = max_shares_per_trade
        self.render_mode = render_mode
        
        self.episode_days = episode_months * 21 # Assuming 21 trading days per month
        
        self.action_space = spaces.Box(low=0.0, high=1.0, shape=(2,), dtype=np.float32)
        
        num_features = features.shape[1]
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(window_size * num_features + 4,), 
            dtype=np.float32
        )
        
        self.min_start_index = self.window_size
        self.max_start_index = len(features) - self.episode_days - 1
        if self.min_start_index > self.max_start_index:
            raise ValueError("The length of the prices array is too short for {} months episode and window size of {}.".format(episode_months, window_size))
        
        self.current_episode_start = 0
        self.current_step = 0
        self.episode_done = False
        
        self.current_long_shares = 0.0
        self.current_short_shares = 0.0
        self.cash = 0.0
        self.current_portfolio_value = 0.0
        self.last_action = np.array([1.0, 0.0], dtype=np.float32)
        
        self.historical_portfolio = []
        self.historical_long_shares = []
        self.historical_short_shares = []
        self.historical_cash = []
        self.historical_prices = []
        self.historical_actions = []
        self.historical_returns = []
        self.np_random = None 
        
        self.reset()
        
        
    def reset(self, seed: int | None = None):
        super().reset(seed=seed)
        
        self.current_episode_start = self.np_random.integers(self.min_start_index, self.max_start_index + 1)
        self.current_step = 0
        self.episode_done = False
        
        initial_price_for_reset = self.prices[self.current_episode_start]
        
        self.current_long_shares = self.initial_long_capital / initial_price_for_reset if initial_price_for_reset > 0 else 0.0
        self.current_short_shares = 0.0 
        self.cash = self.initial_short_capital 
        
        self.current_portfolio_value = self._calculate_portfolio_value(initial_price_for_reset)
        
        self.last_action = np.array([1.0, 0.0], dtype=np.float32) 
        
        self.historical_portfolio = [self.current_portfolio_value]
        self.historical_long_shares = [self.current_long_shares]
        self.historical_short_shares = [self.current_short_shares]
        self.historical_cash = [self.cash]
        self.historical_actions = [self.last_action.copy()]
        self.historical_prices = [initial_price_for_reset]
        self.historical_returns = []
        
        return self._get_observation(), self._get_info()    
            
    
    def step(
        self, action: np.ndarray
    ) -> tuple[np.ndarray, float, bool, bool, dict]:
        
        if self.episode_done:
            raise RuntimeError("Episode is done. Please reset the environment.")
        
        target_long_ratio = np.clip(action[0], self.action_space.low[0], self.action_space.high[0])
        target_short_ratio = np.clip(action[1], self.action_space.low[1], self.action_space.high[1])
        
        current_idx = self.current_episode_start + self.current_step
        current_price = self.prices[current_idx]
        prev_portfolio_value = self.current_portfolio_value
        
        if current_price <= 0: 
            self.current_portfolio_value = 0.0
            self.episode_done = True
            return self._get_observation(), -1000.0, True, False, self._get_info() 
        
        target_long_capital_value = self.initial_long_capital * target_long_ratio
        target_short_capital_value = self.initial_short_capital * target_short_ratio
        
        target_long_shares = target_long_capital_value / current_price 
        target_short_shares = target_short_capital_value / current_price 
        
        delta_long_shares = target_long_shares - self.current_long_shares
        delta_short_shares = target_short_shares - self.current_short_shares
        
        total_desired_trade_value = abs(delta_long_shares * current_price) + abs(delta_short_shares * current_price)
        
        is_significant_action_change_long = abs(target_long_ratio - self.last_action[0]) > self.dead_zone
        is_significant_action_change_short = abs(target_short_ratio - self.last_action[1]) > self.dead_zone
        is_significant_trade_value = total_desired_trade_value > (self.dead_zone * self.current_portfolio_value)
        
        max_trade_capital_per_side = self.current_portfolio_value * self.max_shares_per_trade
        max_trade_shares_long = max_trade_capital_per_side / current_price
        max_trade_shares_short = max_trade_capital_per_side / current_price
        
        if is_significant_action_change_long or is_significant_action_change_short or is_significant_trade_value:
            
            if delta_long_shares > 0:
                shares_to_buy = min(delta_long_shares, max_trade_shares_long)
                cost = shares_to_buy * current_price * (1 + self.commission)

                if self.cash >= cost:
                    self.current_long_shares += shares_to_buy
                    self.cash -= cost

                elif self.current_portfolio_value > 0: 
                    affordable_shares = self.cash / (current_price * (1 + self.commission)) if current_price > 0 else 0
                    self.current_long_shares += affordable_shares
                    self.cash -= affordable_shares * current_price * (1 + self.commission)

            elif delta_long_shares < 0: 
                shares_to_sell = min(abs(delta_long_shares), self.current_long_shares, max_trade_shares_long)
                revenue = shares_to_sell * current_price * (1 - self.commission)
                self.current_long_shares -= shares_to_sell
                self.cash += revenue
            
            if delta_short_shares > 0:
                shares_to_short = min(delta_short_shares, max_trade_shares_short)
                revenue_from_short = shares_to_short * current_price * (1 - self.commission)
                self.current_short_shares += shares_to_short
                self.cash += revenue_from_short

            elif delta_short_shares < 0:
                shares_to_cover = min(abs(delta_short_shares), self.current_short_shares, max_trade_shares_short)
                cost_to_cover = shares_to_cover * current_price * (1 + self.commission)

                if self.cash >= cost_to_cover:
                    self.current_short_shares -= shares_to_cover
                    self.cash -= cost_to_cover

                elif self.current_portfolio_value > 0: 
                    affordable_shares = self.cash / (current_price * (1 + self.commission)) if current_price > 0 else 0
                    actual_shares_to_cover = min(shares_to_cover, self.current_short_shares, affordable_shares)
                    self.current_short_shares -= actual_shares_to_cover
                    self.cash -= actual_shares_to_cover * current_price * (1 + self.commission)
                    
        self.current_long_shares = max(0.0, self.current_long_shares)
        self.current_short_shares = max(0.0, self.current_short_shares)
        
        self.current_step += 1

        self.episode_done = (self.current_step >= self.episode_days or 
                             (current_idx + 1) >= len(self.prices))
        
        self.current_portfolio_value = self._calculate_portfolio_value(current_price)
        
        reward = 0.0
        step_return = 0.0
        
        if self.current_portfolio_value <= 0:
            reward = -200.0 
            self.episode_done = True
        else:
            step_return = (self.current_portfolio_value - prev_portfolio_value) / prev_portfolio_value if prev_portfolio_value != 0 else 0.0 
            reward = self._calculate_reward(step_return, action)
            
            
        
        self.historical_portfolio.append(self.current_portfolio_value)
        self.historical_long_shares.append(self.current_long_shares)
        self.historical_short_shares.append(self.current_short_shares)
        self.historical_cash.append(self.cash)
        self.historical_actions.append(action.copy())
        self.historical_returns.append(step_return)
        self.historical_prices.append(current_price)
            
        self.last_action = action.copy()
        
        obs = self._get_observation() 
        info = self._get_info()
        
        return obs, reward, self.episode_done, False, info
    
    
    def _calculate_reward(self, step_return, current_action_array):
        
        reward = step_return * 200.0
        
        if step_return > 0:
            reward += step_return * 500.0
            
        action_diff = np.linalg.norm(current_action_array - self.last_action) 
        
        if action_diff > self.action_change_penalty_threshold:
            penalty = (action_diff - self.action_change_penalty_threshold) * 10.0 
            reward -= penalty
        
        if len(self.historical_portfolio) > 1:
            drawdown = self._calculate_max_drawdown(np.array(self.historical_portfolio))
            if drawdown > 0.10:
                reward -= (drawdown - 0.10) * 20.0
    
        return max(min(reward, 100.0), -100.0)
    
    def _get_observation(self):
        current_idx = self.current_episode_start + self.current_step

        feature_window_start_idx = current_idx - self.window_size + 1
        feature_window_end_idx = current_idx + 1

        if feature_window_start_idx < 0: 
            padding_needed = abs(feature_window_start_idx)
            market_features = np.zeros((self.window_size, self.features.shape[1]), dtype=np.float32)
            market_features[padding_needed:] = self.features[0:feature_window_end_idx]
        else:
            market_features = self.features[feature_window_start_idx:feature_window_end_idx]

        if market_features.shape[0] > 0 and market_features.std(axis=0).sum() > 1e-8:
            market_features = (market_features - market_features.mean(axis=0)) / (market_features.std(axis=0) + 1e-8)
        else:
            market_features = np.zeros_like(market_features)

        market_features = market_features.flatten()

        portfolio_value_for_norm = max(self.current_portfolio_value, 1e-6) 

        portfolio_state = np.array([
            (self.current_long_shares * self.prices[current_idx]) / portfolio_value_for_norm, 
            (self.current_short_shares * self.prices[current_idx]) / portfolio_value_for_norm, 
            self.cash / portfolio_value_for_norm, 
            self.current_portfolio_value / self.initial_capital 
        ], dtype=np.float32)

        return np.concatenate((market_features, portfolio_state))
    
    def _calculate_portfolio_value(self, current_price):
        long_value = self.current_long_shares * current_price
        short_value_debt = self.current_short_shares * current_price 
        return self.cash + long_value - short_value_debt
        
    
    def _get_info(self) -> dict:
        current_idx = self.current_episode_start + self.current_step
        date_idx = min(current_idx, len(self.dates) - 1)
        date = self.dates[date_idx]

        current_price_for_info = self.prices[min(current_idx, len(self.prices) - 1)]

        return {
            'current_price': current_price_for_info,
            'current_long_shares': self.current_long_shares,
            'current_short_shares': self.current_short_shares,
            'cash': self.cash,
            'portfolio_value': self.current_portfolio_value,
            'total_return_episode_so_far': (self.current_portfolio_value - self.initial_capital) / self.initial_capital if self.initial_capital > 0 else 0.0,
            'date': date
        }
        
    
    def get_episode_stats(self):
        
        risk_free_rate = 0.0823 # TIE 17 Jul 2025
        daily_risk_free_rate = (1 + risk_free_rate) ** (1/252) - 1 # 252 trading days in a year
        
        target_return = daily_risk_free_rate 
        
        returns = np.array(self.historical_returns)
        portfolio_values = np.array(self.historical_portfolio)
        actions = np.array(self.historical_actions)
        
        total_return = (self.historical_portfolio[-1] - self.historical_portfolio[0]) / self.historical_portfolio[0]
        periods_per_year = 12 / self.episode_months
        annualized_return = (1 + total_return ) ** periods_per_year - 1
        
        benchmark_return = (self.historical_prices[-1] - self.historical_prices[0]) / self.historical_prices[0]
        annualized_benchmark_return = (1 + benchmark_return ) ** periods_per_year - 1
        
        # Benchmark calculations
        benchmark_prices = self.historical_prices
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
            "months": self.episode_months,
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
            "final_cash": self.cash,
            "final_long_shares": self.current_long_shares,
            "final_short_shares": self.current_short_shares,
            "avg_action": actions.mean(),
            "min_action": actions.min(),
            "max_action": actions.max(),
            "action_std": actions.std()
        }
    
    
    def _calculate_max_drawdown(self, values):
        if not values.size or len(values) < 2: 
            return 0.0
        
        peak_values = np.maximum.accumulate(values)
        drawdowns = (peak_values - values) / (peak_values + 1e-6) 
        return np.max(drawdowns)
    

    def render(self) -> None:
        if self.render_mode == "human":
            current_value = self.historical_portfolio[-1] if self.historical_portfolio else self.initial_capital
            stats = self.get_episode_stats()
            print(f"Step: {self.current_step}")
            print(f"Portfolio Value: ${current_value:,.2f}")
            print(f"Long Shares: {self.current_long_shares:.2f}, Short Shares: {self.current_short_shares:.2f}")
            print(f"Cash: ${self.cash:,.2f}")
            print(f"Last Action: Long Ratio={self.last_action[0]:.2f}, Short Ratio={self.last_action[1]:.2f}") # Mostrar ambas
            if stats:
                print(f"Total Return: {stats['total_return']:.4f}")
                print(f"Volatility: {stats['volatility']:.4f}")
                print(f"Sharpe Ratio: {stats['sharpe_ratio']:.4f}")
                print(f"Max Drawdown: {stats['max_drawdown']:.4f}")
            print("-" * 30)
            
            

    def close(self) -> None:
        if self.render_mode == "human":
            print("Environment closed.")
            
