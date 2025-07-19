import numpy as np
import gymnasium as gym
from gymnasium import spaces
import pandas as pd

class PortfolioHedgingEnv(gym.Env):
    def __init__(
        self, 
        features: np.ndarray, 
        prices: np.ndarray, 
        dates: np.ndarray = None,
        episode_length_months=6,
        window_size=5,
        dead_zone=0.01,  
        initial_long_capital=1_000_000,
        initial_short_capital=1_000_000,
        commission=0.00125,
        action_change_penalty_threshold=0.2,
        max_shares_per_trade=0.5 
    ):
        
        super().__init__()
        
        # Validate input data and parameters
        assert len(features) == len(prices), "Features and prices must have the same length"
        assert len(features) > window_size, f"Features length ({len(features)}) must be greater than window_size ({window_size})"
        assert np.all(prices >= 0), "Prices must be non-negative"
        assert initial_long_capital >= 0 and initial_short_capital >= 0, "Initial capitals must be positive or zero"
        assert 0 <= commission <= 1, "Commission must be between 0 and 1"
        assert 0 < max_shares_per_trade <= 1, "max_shares_per_trade must be between 0 and 1"
        assert dead_zone >= 0, "Dead zone must be non-negative"

        # Store environment parameters
        self.features = features
        self.prices = prices
        self.dates = pd.to_datetime(dates) if dates is not None else np.arange(len(prices))
        self.window_size = window_size
        self.dead_zone = dead_zone
        self.initial_long_capital = initial_long_capital
        self.initial_short_capital = initial_short_capital
        
        self.initial_portfolio_value = initial_long_capital + initial_short_capital
        if self.initial_portfolio_value == 0:
            self.initial_portfolio_value = 1.0 

        self.commission = commission
        self.action_change_penalty_threshold = action_change_penalty_threshold
        self.max_shares_per_trade = max_shares_per_trade
        self.episode_length = episode_length_months * 21  

        self.min_start_idx = self.window_size
        self.max_start_idx = len(features) - self.episode_length - 1
        if self.max_start_idx < self.min_start_idx:
            raise ValueError(f"Not enough data for specified episode length ({self.episode_length} days) and window size ({self.window_size}). Total data points: {len(self.prices)}. Need at least {self.window_size + self.episode_length} data points.")

        # --- MODIFICACIÓN CLAVE: Action Space de 2 dimensiones ---
        # action[0]: Target % of initial_long_capital for long position (0.0 to 1.0)
        # action[1]: Target % of initial_short_capital for short position (0.0 to 1.0)
        self.action_space = spaces.Box(low=0.0, high=1.0, shape=(2,), dtype=np.float32)
        
        num_features = features.shape[1]
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(window_size * num_features + 4,), 
            dtype=np.float32
        )

        self.current_episode_start = 0
        self.current_step = 0
        self.episode_done = False
        self.current_long_shares = 0.0
        self.current_short_shares = 0.0
        self.cash = 0.0
        self.current_portfolio_value = 0.0
        self.last_action = np.array([1.0, 0.0], dtype=np.float32) # Iniciar con 100% largo, 0% corto
        self.portfolio_history = []
        self.action_history = []
        self.price_history = []
        self.returns_history = []
        self.np_random = None 

        self.reset() 

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
            self.current_portfolio_value / self.initial_portfolio_value 
        ], dtype=np.float32)

        return np.concatenate((market_features, portfolio_state))

    def _calculate_portfolio_value(self, current_price):
        long_value = self.current_long_shares * current_price
        short_value_debt = self.current_short_shares * current_price 
        return self.cash + long_value - short_value_debt

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.current_episode_start = self.np_random.integers(self.min_start_idx, self.max_start_idx + 1)
        self.current_step = 0
        self.episode_done = False
        
        initial_price_for_reset = self.prices[self.current_episode_start]
        
        # Iniciar con 100% de la posición larga, y sin posición corta.
        # El capital para corto está en efectivo. Esto corresponde a last_action = [1.0, 0.0].
        self.current_long_shares = self.initial_long_capital / initial_price_for_reset if initial_price_for_reset > 0 else 0.0
        self.current_short_shares = 0.0 
        self.cash = self.initial_short_capital 
        
        self.current_portfolio_value = self._calculate_portfolio_value(initial_price_for_reset)
        
        self.last_action = np.array([1.0, 0.0], dtype=np.float32) # Estado inicial: 100% largo, 0% corto
        self.portfolio_history = [self.current_portfolio_value]
        self.action_history = [self.last_action.copy()] # Guardar una copia para evitar referencias
        self.price_history = [initial_price_for_reset]
        self.returns_history = []
        
        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.episode_done:
            raise ValueError("Episode is done, call reset()")

        # --- MODIFICACIÓN CLAVE: Interpretar dos acciones ---
        # action[0]: % objetivo de capital largo
        # action[1]: % objetivo de capital corto
        target_long_ratio = np.clip(action[0], self.action_space.low[0], self.action_space.high[0])
        target_short_ratio = np.clip(action[1], self.action_space.low[1], self.action_space.high[1])

        current_idx = self.current_episode_start + self.current_step
        current_price = self.prices[current_idx]
        prev_portfolio_value = self.current_portfolio_value

        if current_price <= 0: 
            self.current_portfolio_value = 0.0
            self.episode_done = True
            return self._get_observation(), -1000.0, True, False, self._get_info() 

        # Calcular valores de capital objetivo basados en las ratios de acción
        target_long_capital_value = self.initial_long_capital * target_long_ratio
        target_short_capital_value = self.initial_short_capital * target_short_ratio
        
        # Convertir valores de capital objetivo a shares objetivo
        target_long_shares = target_long_capital_value / current_price 
        target_short_shares = target_short_capital_value / current_price 

        # Calcular la diferencia de shares desde la posición actual a la objetivo
        delta_long_shares = target_long_shares - self.current_long_shares
        delta_short_shares = target_short_shares - self.current_short_shares

        # Calcular el valor monetario del cambio para la dead_zone
        total_desired_trade_value = abs(delta_long_shares * current_price) + abs(delta_short_shares * current_price)

        # Aplicamos dead zone
        is_significant_action_change_long = abs(target_long_ratio - self.last_action[0]) > self.dead_zone
        is_significant_action_change_short = abs(target_short_ratio - self.last_action[1]) > self.dead_zone
        is_significant_trade_value = total_desired_trade_value > (self.dead_zone * self.current_portfolio_value)

        # Cap trades by max_shares_per_trade (as ratio of current portfolio value)
        max_trade_capital_per_side = self.current_portfolio_value * self.max_shares_per_trade
        max_trade_shares_long = max_trade_capital_per_side / current_price
        max_trade_shares_short = max_trade_capital_per_side / current_price

        trade_executed = False

        if is_significant_action_change_long or is_significant_action_change_short or is_significant_trade_value:
            # Manejar ajustes de Posición Larga
            if delta_long_shares > 0: # Comprar más acciones largas
                shares_to_buy = min(delta_long_shares, max_trade_shares_long)
                cost = shares_to_buy * current_price * (1 + self.commission)

                if self.cash >= cost:
                    self.current_long_shares += shares_to_buy
                    self.cash -= cost
                    trade_executed = True

                elif self.current_portfolio_value > 0: 
                    affordable_shares = self.cash / (current_price * (1 + self.commission)) if current_price > 0 else 0
                    self.current_long_shares += affordable_shares
                    self.cash -= affordable_shares * current_price * (1 + self.commission)

                    if affordable_shares > 0: trade_executed = True

            elif delta_long_shares < 0: # Vender acciones largas
                shares_to_sell = min(abs(delta_long_shares), self.current_long_shares, max_trade_shares_long)
                revenue = shares_to_sell * current_price * (1 - self.commission)
                self.current_long_shares -= shares_to_sell
                self.cash += revenue
                trade_executed = True

            # Manejar ajustes de Posición Corta (Hedging)
            if delta_short_shares > 0: # Abrir/Aumentar posición corta
                shares_to_short = min(delta_short_shares, max_trade_shares_short)
                revenue_from_short = shares_to_short * current_price * (1 - self.commission)
                self.current_short_shares += shares_to_short
                self.cash += revenue_from_short
                trade_executed = True

            elif delta_short_shares < 0: # Cubrir/Disminuir posición corta
                shares_to_cover = min(abs(delta_short_shares), self.current_short_shares, max_trade_shares_short)
                cost_to_cover = shares_to_cover * current_price * (1 + self.commission)

                if self.cash >= cost_to_cover:
                    self.current_short_shares -= shares_to_cover
                    self.cash -= cost_to_cover
                    trade_executed = True

                elif self.current_portfolio_value > 0: 
                    affordable_shares = self.cash / (current_price * (1 + self.commission)) if current_price > 0 else 0
                    actual_shares_to_cover = min(shares_to_cover, self.current_short_shares, affordable_shares)
                    self.current_short_shares -= actual_shares_to_cover
                    self.cash -= actual_shares_to_cover * current_price * (1 + self.commission)

                    if actual_shares_to_cover > 0: trade_executed = True
        
        self.current_long_shares = max(0.0, self.current_long_shares)
        self.current_short_shares = max(0.0, self.current_short_shares)

        self.current_step += 1

        self.episode_done = (self.current_step >= self.episode_length or 
                             (current_idx + 1) >= len(self.prices))

        self.current_portfolio_value = self._calculate_portfolio_value(current_price)
        
        reward = 0.0
        step_return = 0.0

        if self.current_portfolio_value <= 0:
            reward = -200.0 
            self.episode_done = True
        else:
            step_return = (self.current_portfolio_value - prev_portfolio_value) / prev_portfolio_value if prev_portfolio_value != 0 else 0.0 
            reward = self._calculate_reward(step_return, action) # Pasar el array de acción
            
        self.portfolio_history.append(self.current_portfolio_value)
        self.action_history.append(action.copy()) # Guardar una copia del array de acción
        self.price_history.append(current_price) 
        self.returns_history.append(step_return)
        
        self.last_action = action.copy() # Actualizar last_action con el nuevo array

        observation = self._get_observation() 
        info = self._get_info()

        return observation, reward, self.episode_done, False, info

    def _calculate_reward(self, step_return, current_action_array): # Recibe el array de acción
        reward = step_return * 200.0
        
        # Si el retorno es positivo, darle un boost
        if step_return > 0:
            reward += step_return * 500.0

        # Calcular la diferencia de acción para cada componente y sumarlas (o usar la norma L2)
        action_diff = np.linalg.norm(current_action_array - self.last_action) # Norma euclidiana del cambio de acción
        
        if action_diff > self.action_change_penalty_threshold:
            penalty = (action_diff - self.action_change_penalty_threshold) * 10.0 
            reward -= penalty
        
        if len(self.portfolio_history) > 1:
            drawdown = self._calculate_max_drawdown(np.array(self.portfolio_history))
            if drawdown > 0.10:
                reward -= (drawdown - 0.10) * 20.0
            
            initial_long_shares_at_start = self.initial_long_capital / self.prices[self.current_episode_start] if self.prices[self.current_episode_start] > 0 else 0
            if self.current_long_shares > 2 * initial_long_shares_at_start and initial_long_shares_at_start > 0:
                excess_shares = self.current_long_shares - 2 * initial_long_shares_at_start
                reward -= excess_shares * 0.01
        
        return max(min(reward, 100.0), -100.0)

    # Resto de métodos (_get_info, get_episode_stats, _calculate_max_drawdown, render, close)
    # permanecen igual ya que no dependen directamente del cambio en el action_space,
    # aunque la representación de 'last_action' en render/info cambiará a un array.
    def _get_info(self):
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
            'total_return_episode_so_far': (self.current_portfolio_value - self.initial_portfolio_value) / self.initial_portfolio_value if self.initial_portfolio_value > 0 else 0.0,
            'date': date
        }

    def get_episode_stats(self):
        if len(self.portfolio_history) < 2:
            return {
                'total_return': 0.0,
                'volatility': 0.0,
                'sharpe_ratio': 0.0,
                'max_drawdown': 0.0,
                'final_position_net_shares': self.current_long_shares - self.current_short_shares,
                'num_trades': 0
            }
            
        returns = np.array(self.returns_history)
        values = np.array(self.portfolio_history)
        
        total_return = (values[-1] - values[0]) / values[0] if values[0] != 0 else 0.0
        
        volatility = returns.std() if len(returns) > 0 else 0.0
        
        sharpe_ratio = returns.mean() / volatility if volatility > 0 else 0.0
        max_drawdown = self._calculate_max_drawdown(values)
        
        # Contar trades basado en cambios en cualquiera de las acciones
        num_trades = sum(1 for i in range(1, len(self.action_history)) 
                           if np.linalg.norm(self.action_history[i] - self.action_history[i-1]) > 1e-6)


        return {
            'total_return': total_return,
            'volatility': volatility,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'final_position_net_shares': self.current_long_shares - self.current_short_shares,
            'num_trades': num_trades
        }

    def _calculate_max_drawdown(self, values):
        if not values.size or len(values) < 2: 
            return 0.0
        
        peak_values = np.maximum.accumulate(values)
        drawdowns = (peak_values - values) / (peak_values + 1e-6) 
        return np.max(drawdowns)

    def render(self, mode='human'):
        if mode == 'human':
            current_value = self.portfolio_history[-1] if self.portfolio_history else self.initial_portfolio_value
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

    def close(self):
        pass

# Example usage for quick unit testing of the environment
if __name__ == '__main__':
    # Sample data
    sample_dates = pd.to_datetime(pd.date_range(start='2023-01-01', periods=500))
    np.random.seed(0) 
    initial_price = 100
    price_changes = np.random.normal(0, 0.5, 500).cumsum() 
    sample_prices = (initial_price + price_changes).astype(np.float32)
    sample_prices = np.maximum(sample_prices, 10.0) 

    sample_features = np.random.rand(500, 3).astype(np.float32) 

    # Environment parameters
    env_params = {
        "features": sample_features,
        "prices": sample_prices,
        "dates": sample_dates,
        "episode_length_months": 6, 
        "window_size": 5,
        "dead_zone": 0.001, 
        "initial_long_capital": 700_000, 
        "initial_short_capital": 700_000, 
        "commission": 0.005,
        "action_change_penalty_threshold": 0.1,
        "max_shares_per_trade": 0.20, 
    }

    print("--- Probando PortfolioHedgingEnv: Dos Acciones (Largo y Corto) ---")
    try:
        env = PortfolioHedgingEnv(**env_params)
        env.render_mode = "human"

        obs, info = env.reset(seed=42)
        print("\n--- Estado Inicial (Reset) ---")
        env.render()
        print("Shape de la Observación Inicial:", obs.shape)
        print("Info Inicial:", info)
        print("Valor del Portfolio Inicial:", info['portfolio_value'])

        done_internal = False 
        total_reward = 0
        step_count = 0
        
        # --- NUEVAS ACCIONES DE PRUEBA: Cada una es un array [largo_ratio, corto_ratio] ---
        # largo_ratio: 0.0 (0% largo) a 1.0 (100% largo de initial_long_capital)
        # corto_ratio: 0.0 (0% corto) a 1.0 (100% corto de initial_short_capital)
        action_plan = [
            np.array([1.0, 0.0]), # 100% largo, 0% corto (posición inicial)
            np.array([1.0, 0.0]), # Mantiene
            np.array([0.8, 0.0]), # 80% largo, 0% corto
            np.array([0.8, 0.0]), 
            np.array([1.0, 0.2]), # 100% largo, 20% corto (para cubrir)
            np.array([1.0, 0.5]), # 100% largo, 50% corto
            np.array([0.5, 0.1]), # 50% largo, 10% corto (reduce ambos)
            np.array([0.0, 0.0]), # Cierra todo (solo cash)
            np.array([0.0, 0.5]), # ¡Solo corto! Ojo: si no hay largo, esta cobertura puede no tener sentido para el RL.
                                  # Si quieres que el agente pueda tomar posiciones cortas especulativas sin largo,
                                  # necesitaríamos ajustar la interpretación de initial_short_capital y la recompensa.
            np.array([1.0, 1.0]), # 100% largo, 100% corto (fully hedged si los capitales son iguales)
        ]
        
        steps_per_action_phase = 5 

        while not done_internal:
            phase_idx = step_count // steps_per_action_phase
            if phase_idx < len(action_plan):
                action_to_take = action_plan[phase_idx]
            else:
                action_to_take = np.array([1.0, 0.0]) # Acción neutral por defecto

            print(f"\n--- Paso {step_count + 1} ---")
            print(f"Acción del Agente: Largo={action_to_take[0]:.2f}, Corto={action_to_take[1]:.2f}")
            
            obs, reward, terminated, truncated, info = env.step(action_to_take)
            total_reward += reward
            
            done_internal = terminated or truncated 
            
            env.render() 
            print(f"Recompensa: {reward:.4f}, Terminado: {terminated}, Truncado: {truncated}")
            
            step_count += 1
            if done_internal:
                break 

        print(f"\nEpisodio finalizado después de {step_count} pasos.")
        print(f"Recompensa Total Acumulada: {total_reward:.2f}")
        final_stats = env.get_episode_stats()
        print("\n=== Estadísticas Finales del Episodio ===")
        for k, v in final_stats.items():
            if isinstance(v, (float, np.float32)):
                print(f"  {k}: {v:.4f}")
            else:
                print(f"  {k}: {v}")

    except ValueError as e:
        print(f"Error al crear/ejecutar el entorno: {e}")
    except Exception as e:
        print(f"Se produjo un error inesperado: {e}")