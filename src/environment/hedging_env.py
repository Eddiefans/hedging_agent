import numpy as np
import gymnasium as gym
from gymnasium import spaces
import pandas as pd 

class PortfolioHedgingEnv(gym.Env):
    """
    Gym environment for a portfolio hedging agent:
    - Agent manages hedging for a portfolio of a single stock (NVIDIA)
    - Output is continuous between 0 and 2, where:
        - 0.0 means no investment.
        - Values from 0.0 to 1.0 indicate long positions, with 1.0 being fully invested long.
        - Values from 1.0 to 2.0 indicate short positions, with 2.0 being fully invested short.
    - Dead zone of ±0.5% around 1.0 results in no action.
    - Each episode randomly selects a 6-month period from the dataset.
    - Rewards based on daily portfolio return, with a simple penalty for large action changes.
    """

    def __init__(self, features: np.ndarray, prices: np.ndarray, dates: np.ndarray = None,
                 episode_length_months=6, window_size=5,
                 dead_zone=0.005,
                 initial_portfolio_value=2_000_000,
                 initial_long_capital=1_000_000,
                 initial_short_capital=1_000_000,
                 commission=0.00125,
                 max_shares_per_trade=1.0, # Mantener para limitar tamaño de trade
                 action_change_penalty_threshold=0.1 # Nuevo umbral para penalización de acción
                 ):
        super().__init__()

        assert len(features) == len(prices), "Features and prices must have same length"

        self.features = features
        self.prices = prices
        self.dates = pd.to_datetime(dates) if dates is not None else np.arange(len(prices))
        self.window_size = window_size
        self.dead_zone = dead_zone  # ±0.5% dead zone around 1.0
        
        self.initial_portfolio_value = initial_portfolio_value
        self.initial_long_capital = initial_long_capital
        self.initial_short_capital = initial_short_capital
        self.commission = commission
        self.max_shares_per_trade = max_shares_per_trade
        self.action_change_penalty_threshold = action_change_penalty_threshold # Umbral para la penalización

        self.episode_length = episode_length_months * 21

        min_feature_lag = 0 
        self.min_start_idx = self.window_size + min_feature_lag
        self.max_start_idx = len(features) - self.episode_length - 1

        if self.max_start_idx <= self.min_start_idx:
            raise ValueError(f"Not enough data for {episode_length_months}-month episodes. "
                             f"Required: {self.min_start_idx + self.episode_length + 1}, Available: {len(features)}")

        self.action_space = spaces.Box(low=0.0, high=2.0, shape=(1,), dtype=np.float32)

        n_features = features.shape[1]
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(window_size * n_features + 4,), # features + (shares_long, shares_short, cash, portfolio_value)
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
        """Get current observation window including portfolio state"""
        current_data_idx = self.current_episode_start + self.current_step
        
        end_features_idx = current_data_idx + self.window_size
        
        if end_features_idx > len(self.features):
            obs_window = self.features[current_data_idx:len(self.features)].flatten()
            padding = np.zeros(self.window_size * self.features.shape[1] - len(obs_window))
            market_features = np.concatenate((obs_window, padding))
        else:
            market_features = self.features[current_data_idx : end_features_idx].flatten()

        current_price = self.prices[current_data_idx] if current_data_idx < len(self.prices) else self.prices[-1]

        portfolio_state = np.array([
            self.current_long_shares,
            self.current_short_shares,
            self.cash,
            self.current_portfolio_value
        ], dtype=np.float32)

        return np.concatenate((market_features, portfolio_state))

    def _calculate_portfolio_value(self, current_price):
        """Calcula el valor total actual del portafolio."""
        long_value = self.current_long_shares * current_price
        short_value = self.current_short_shares * current_price
        return self.cash + long_value - short_value

    def reset(self, seed=None, options=None):
        """Reset environment for a new episode with random 6-month period"""
        super().reset(seed=seed)
        self.np_random = np.random.default_rng(seed)

        self.current_episode_start = self.np_random.integers(
            self.min_start_idx, self.max_start_idx
        )
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

        obs, info = self._get_observation(), self._get_info()
        return obs, info

    def step(self, action):
        """Execute one step in the environment"""
        if self.episode_done:
            raise ValueError("Episode is done, call reset()")

        action_value = np.clip(action[0], 0.0, 2.0)

        current_data_idx = self.current_episode_start + self.current_step
        current_price = self.prices[current_data_idx]

        portfolio_value_before_action = self.current_portfolio_value

        # Calcular posicion deseada
        if action_value >= 0.0 and action_value <= 1.0: # Desired long position
            desired_long_capital = action_value * self.initial_long_capital
            target_long_shares = desired_long_capital / current_price
            target_short_shares = 0.0
        elif action_value > 1.0 and action_value <= 2.0: # Desired short position
            short_proportion = (action_value - 1.0)
            desired_short_capital = short_proportion * self.initial_short_capital
            target_short_shares = desired_short_capital / current_price
            target_long_shares = 0.0
        else: # Fallback, should not happen due to np.clip
            target_long_shares = self.current_long_shares
            target_short_shares = self.current_short_shares

        # --- Calcular Cambios de Acciones Netos ---
        net_shares_current = self.current_long_shares - self.current_short_shares
        net_shares_target = target_long_shares - target_short_shares
        shares_to_execute = net_shares_target - net_shares_current

        # --- Lógica de la Dead Zone  ---
        if abs(shares_to_execute * current_price) < self.dead_zone * self.initial_portfolio_value: # Si el valor en dolares del trade deseado es menor al 1% del portafolio inicial
            shares_to_execute = 0.0 # No ejecutar trade
            commission_cost = 0.0 # No hay comision
        else:
            # Limitar el cambio máximo de acciones por operación
            max_dollar_trade = self.max_shares_per_trade * self.initial_portfolio_value
            max_shares_trade_units = max_dollar_trade / current_price
            shares_to_execute = np.clip(shares_to_execute, -max_shares_trade_units, max_shares_trade_units)
            # --- Ejecutar Transacciones y Aplicar Comisiones ---
            commission_cost = abs(shares_to_execute) * current_price * self.commission
            self.cash -= commission_cost

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

        # --- Actualizar Paso del Tiempo ---
        self.current_step += 1
        
        self.episode_done = (self.current_step >= self.episode_length) or \
                            ((self.current_episode_start + self.current_step) >= len(self.prices))

        if not self.episode_done:
            next_price = self.prices[self.current_episode_start + self.current_step]
        else:
            next_price = current_price 

        # --- Recalcular Valor del Portafolio y Recompensa ---
        self.current_portfolio_value = self._calculate_portfolio_value(next_price)

        # Calcular el retorno del paso para la recompensa
        step_return = (self.current_portfolio_value - portfolio_value_before_action) / portfolio_value_before_action if portfolio_value_before_action != 0 else 0

        # Llama a la función de recompensa simplificada
        reward = self._calculate_reward(step_return, action_value)

        # --- Almacenar Historial ---
        self.portfolio_history.append(self.current_portfolio_value)
        self.action_history.append(action_value)
        self.price_history.append(next_price)
        self.returns_history.append(step_return)

        obs, info = self._get_observation(), self._get_info()

        self.last_action = action_value

        return obs, reward, self.episode_done, False, info

    def _calculate_reward(self, step_return, current_action_value):
        """
        Calcula la recompensa para el paso actual, de forma simplificada.
        """
        reward = 0.0

        # 1. Recompensa principal: Retorno diario (log-retorno)
        if (1 + step_return) > 0:
            reward = np.log(1 + step_return) * 1000
        else:
            reward = -10.0 # Penalización si hay grandes pérdidas

        # 2. Penalización por cambio de acción significativo (ej. > 0.1 en el rango 0-2)
        action_diff = np.abs(current_action_value - self.last_action)
        
        # Penaliza si el cambio es mayor al umbral definido
        if action_diff > self.action_change_penalty_threshold:
            # La penalización aumenta linealmente con la magnitud del cambio por encima del umbral
            penalty_amount = (action_diff - self.action_change_penalty_threshold) * 0.05 # Ajusta el 0.1 para la intensidad
            reward -= penalty_amount
            
        return reward

    def _get_info(self):
        """Devuelve información adicional sobre el estado del entorno."""
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
            'final_position_net_shares': self.current_long_shares - self.current_short_shares,
            'num_trades': sum(1 for i in range(1, len(self.action_history))
                               if abs(self.action_history[i] - self.action_history[i-1]) > 1e-6)
        }

    def _calculate_max_drawdown(self, portfolio_values):
        """Calculate maximum drawdown"""
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
        """Render the environment (optional)"""
        if mode == 'human':
            current_value = self.portfolio_history[-1] if self.portfolio_history else self.initial_portfolio_value
            print(f"Step: {self.current_step}, Portfolio: ${current_value:,.2f}, Long Shares: {self.current_long_shares:.2f}, Short Shares: {self.current_short_shares:.2f}, Cash: ${self.cash:,.2f}")