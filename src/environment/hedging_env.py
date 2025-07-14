import numpy as np
import gymnasium as gym
from gymnasium import spaces
import pandas as pd

class PortfolioHedgingEnv(gym.Env):
    def __init__(self, features: np.ndarray, prices: np.ndarray, dates: np.ndarray = None,
                 episode_length_months=6, window_size=5, dead_zone=0.005,
                 initial_portfolio_value=2_000_000, initial_long_capital=1_000_000,
                 initial_short_capital=1_000_000, commission=0.00125,
                 max_shares_per_trade=1.0, action_change_penalty_threshold=0.1):
        super().__init__()

        # Validaciones
        assert len(features) == len(prices), "Features and prices must have same length"
        assert len(features) > window_size, f"Features length ({len(features)}) must be greater than window_size ({window_size})"
        assert all(prices >= 0), "Prices must be non-negative"
        assert initial_portfolio_value > 0, "Initial portfolio value must be positive"
        assert initial_long_capital > 0 and initial_short_capital > 0, "Initial capitals must be positive"
        assert 0 <= commission <= 1, "Commission must be between 0 and 1"
        assert 0 <= max_shares_per_trade <= 1, "Max shares per trade must be between 0 and 1"

        self.features = features
        self.prices = prices
        self.dates = pd.to_datetime(dates) if dates is not None else np.arange(len(prices))
        self.window_size = window_size
        self.dead_zone = dead_zone
        self.initial_portfolio_value = initial_portfolio_value
        self.initial_long_capital = initial_long_capital
        self.initial_short_capital = initial_short_capital
        self.commission = commission
        self.max_shares_per_trade = max_shares_per_trade
        self.action_change_penalty_threshold = action_change_penalty_threshold
        self.episode_length = episode_length_months * 21

        min_feature_lag = 0
        self.min_start_idx = self.window_size + min_feature_lag
        self.max_start_idx = len(features) - self.episode_length - 1

        if self.max_start_idx <= self.min_start_idx:
            raise ValueError(f"Not enough data for {episode_length_months}-month episodes.")

        self.action_space = spaces.Box(low=0.0, high=2.0, shape=(1,), dtype=np.float32)
        n_features = features.shape[1]
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(window_size * n_features + 4,),
            dtype=np.float32
        )

        self.current_episode_start = 0
        self.current_step = 0
        self.episode_done = False
        self.current_long_shares = 0.0
        self.current_short_shares = 0.0
        self.cash = 0.0
        self.current_portfolio_value = 0.0
        self.last_action = 0.0
        self.portfolio_history = []
        self.action_history = []
        self.price_history = []
        self.returns_history = []
        self.np_random = np.random.default_rng()
        self.reset()

    def _get_observation(self):
        current_data_idx = self.current_episode_start + self.current_step
        end_features_idx = current_data_idx + self.window_size

        if end_features_idx > len(self.features):
            last_valid_idx = len(self.features) - 1
            obs_window = self.features[current_data_idx:last_valid_idx + 1]
            last_features = self.features[last_valid_idx]
            padding = np.repeat(last_features[np.newaxis, :], self.window_size - (last_valid_idx - current_data_idx + 1), axis=0)
            market_features = np.concatenate((obs_window, padding), axis=0)
        else:
            market_features = self.features[current_data_idx:end_features_idx]

        market_features = (market_features - market_features.mean(axis=0)) / (market_features.std(axis=0) + 1e-8)
        market_features = market_features.flatten()

        current_price = self.prices[current_data_idx] if current_data_idx < len(self.prices) else self.prices[-1]
        portfolio_state = np.array([
            self.current_long_shares,
            self.current_short_shares,
            self.cash / self.initial_portfolio_value,
            self.current_portfolio_value / self.initial_portfolio_value
        ], dtype=np.float32)

        return np.concatenate((market_features, portfolio_state))

    def _calculate_portfolio_value(self, current_price):
        long_value = self.current_long_shares * current_price
        short_value = self.current_short_shares * current_price
        return self.cash + long_value - short_value

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.np_random = np.random.default_rng(seed)
        self.current_episode_start = self.np_random.integers(self.min_start_idx, self.max_start_idx)
        self.current_step = 0
        self.episode_done = False
        self.current_long_shares = 0.0
        self.current_short_shares = 0.0
        self.cash = self.initial_portfolio_value
        self.current_portfolio_value = self.initial_portfolio_value
        self.last_action = 0.0
        self.portfolio_history = [self.current_portfolio_value]
        self.action_history = [self.last_action]
        self.price_history = [self.prices[self.current_episode_start]]
        self.returns_history = []
        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.episode_done:
            raise ValueError("Episode is done, call reset()")

        action_value = np.clip(action[0], 0.0, 2.0)
        current_data_idx = self.current_episode_start + self.current_step
        current_price = self.prices[current_data_idx]
        portfolio_value_before_action = self.current_portfolio_value

        # Usar el capital actual para las posiciones deseadas
        current_long_capital = min(self.current_portfolio_value, self.initial_long_capital)  # Limitar al capital inicial para evitar sobreapalancamiento
        current_short_capital = min(self.current_portfolio_value, self.initial_short_capital)

        # Calcular posición deseada
        if action_value >= 0.0 and action_value <= 1.0:
            desired_long_capital = action_value * current_long_capital
            target_long_shares = desired_long_capital / current_price
            target_short_shares = 0.0
        elif action_value > 1.0 and action_value <= 2.0:
            desired_long_capital = current_long_capital  # Mantener 100% de posiciones largas
            target_long_shares = desired_long_capital / current_price
            short_proportion = (action_value - 1.0)
            desired_short_capital = short_proportion * current_short_capital
            target_short_shares = desired_short_capital / current_price
        else:
            target_long_shares = self.current_long_shares
            target_short_shares = self.current_short_shares

        # Calcular cambio de acciones netos
        net_shares_current = self.current_long_shares - self.current_short_shares
        net_shares_target = target_long_shares - target_short_shares
        shares_to_execute = net_shares_target - net_shares_current

        # Dead zone basada en el capital actual
        action_diff = abs(action_value - self.last_action)
        if action_diff < self.dead_zone or abs(shares_to_execute * current_price) < self.dead_zone * self.current_portfolio_value:
            shares_to_execute = 0.0
            commission_cost = 0.0
        else:
            max_dollar_trade = self.max_shares_per_trade * self.current_portfolio_value  # Usar capital actual
            max_shares_trade_units = max_dollar_trade / current_price
            shares_to_execute = np.clip(shares_to_execute, -max_shares_trade_units, max_shares_trade_units)
            commission_cost = abs(shares_to_execute) * current_price * self.commission
            self.cash -= commission_cost

        # Ejecutar transacciones
        if shares_to_execute > 0:
            if self.current_short_shares > 0:
                shares_to_cover = min(shares_to_execute, self.current_short_shares)
                self.current_short_shares -= shares_to_cover
                self.cash += shares_to_cover * current_price
                shares_to_execute -= shares_to_cover
            self.current_long_shares += shares_to_execute
            self.cash -= shares_to_execute * current_price
        elif shares_to_execute < 0:
            abs_shares_to_execute = np.abs(shares_to_execute)
            if self.current_long_shares > 0:
                shares_to_sell = min(abs_shares_to_execute, self.current_long_shares)
                self.current_long_shares -= shares_to_sell
                self.cash += shares_to_sell * current_price
                abs_shares_to_execute -= shares_to_sell
            self.current_short_shares += abs_shares_to_execute
            self.cash += abs_shares_to_execute * current_price

        self.current_long_shares = max(0.0, self.current_long_shares)
        self.current_short_shares = max(0.0, self.current_short_shares)
        self.current_step += 1

        self.episode_done = (self.current_step >= self.episode_length) or \
                            (current_data_idx >= len(self.prices) - 1)

        next_price = self.prices[current_data_idx + 1] if not self.episode_done else current_price
        self.current_portfolio_value = self._calculate_portfolio_value(next_price)
        step_return = (self.current_portfolio_value - portfolio_value_before_action) / portfolio_value_before_action if portfolio_value_before_action != 0 else 0
        reward = self._calculate_reward(step_return, action_value, shares_to_execute)

        self.portfolio_history.append(self.current_portfolio_value)
        self.action_history.append(action_value)
        self.price_history.append(next_price)
        self.returns_history.append(step_return)

        obs, info = self._get_observation(), self._get_info()
        self.last_action = action_value
        
        return obs, reward, self.episode_done, False, info

    def _calculate_reward(self, step_return, current_action_value, shares_to_execute):
        reward = 0.0
        if (1 + step_return) > 0:
            reward = np.log(1 + step_return) * 1000
        else:
            reward = -10.0

        action_diff = np.abs(current_action_value - self.last_action)
        if action_diff > self.action_change_penalty_threshold:
            penalty = (action_diff - self.action_change_penalty_threshold) ** 2 * 0.1
            reward -= penalty

        if len(self.returns_history) > 5:
            recent_returns = np.array(self.returns_history[-5:])
            volatility = recent_returns.std()
            reward -= volatility * 10

        return reward

    def _get_info(self):
        current_data_idx = self.current_episode_start + self.current_step
        current_date_val = self.dates[current_data_idx] if current_data_idx < len(self.dates) else self.dates[-1]
        return {
            'current_price': self.price_history[-1] if self.price_history else self.prices[self.current_episode_start],
            'current_long_shares': self.current_long_shares,
            'current_short_shares': self.current_short_shares,
            'cash': self.cash,
            'portfolio_value': self.current_portfolio_value,
            'total_return_episode_so_far': (self.current_portfolio_value - self.initial_portfolio_value) / self.initial_portfolio_value,
            'date': current_date_val
        }

    def get_episode_stats(self):
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
            'final_position_net_shares': self.current_long_shares - self.current_short_shares,
            'num_trades': sum(1 for i in range(1, len(self.action_history))
                             if abs(self.action_history[i] - self.action_history[i-1]) > 1e-6)
        }

    def _calculate_max_drawdown(self, portfolio_values):
        peak = portfolio_values[0]
        max_dd = 0.0
        for value in portfolio_values:
            if value > peak:
                peak = value
            if peak > 0:
                drawdown = (peak - value) / peak
                max_dd = max(max_dd, drawdown)
        return max_dd

    def render(self, mode='human'):
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