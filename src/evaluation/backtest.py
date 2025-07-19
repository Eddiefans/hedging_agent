import os 
import sys
import pandas as pd 
import numpy as np 
from stable_baselines3 import PPO, DDPG

# Add the project root directory to the system path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(project_root)

# Importa tu entorno, que contiene la lógica principal
from src.environment.hedging_env import PortfolioHedgingEnv

# --- INICIO DE CÓDIGO NECESARIO PARA EL BENCHMARK ---
# Función para calcular Max Drawdown (copiada de tu entorno)
# Es crucial tener esta función definida globalmente en este script
# porque la necesitas para calcular el drawdown del benchmark directamente.

def _calculate_max_drawdown_for_benchmark(values: np.ndarray) -> float:
    if not values.size or len(values) < 2: 
        return 0.0
    
    peak_values = np.maximum.accumulate(values)
    drawdowns = (peak_values - values) / (np.where(peak_values > 1e-6, peak_values, 1e-6)) 
    return np.max(drawdowns)


# Función para calcular las métricas del benchmark (Buy and Hold) para un episodio
def _calculate_benchmark_metrics(prices_data: np.ndarray, initial_total_capital: float, episode_start_idx: int, current_episode_length: int) -> dict:
    """
    Calcula las métricas de rendimiento para una estrategia de Buy and Hold
    durante un período específico (equivalente a un episodio del agente).
    """
    
    episode_end_idx = episode_start_idx + current_episode_length - 1
    # Asegúrate de que el end_idx no exceda los límites de los datos
    if episode_end_idx >= len(prices_data):
        episode_end_idx = len(prices_data) - 1

    benchmark_prices_segment = prices_data[episode_start_idx : episode_end_idx + 1]

    if len(benchmark_prices_segment) < 2:
        return {
            'benchmark_return': 0.0,
            'annualized_benchmark_return': 0.0,
            'benchmark_sharpe_ratio': 0.0,
            'benchmark_sortino_ratio': 0.0,
            'benchmark_max_drawdown': 0.0,
            'benchmark_volatility': 0.0
        }

    initial_price_bh = benchmark_prices_segment[0]
    
    # Simular Buy and Hold: comprar acciones al inicio con el capital inicial total
    # Si initial_price_bh es 0, no se pueden comprar acciones.
    initial_shares_bh = initial_total_capital / initial_price_bh if initial_price_bh > 0 else 0.0
    
    # Valor del portafolio Buy and Hold a lo largo del tiempo
    benchmark_portfolio_values = initial_shares_bh * benchmark_prices_segment
    
    # Calcular Retorno Total
    benchmark_return = (benchmark_portfolio_values[-1] - benchmark_portfolio_values[0]) / benchmark_portfolio_values[0] if benchmark_portfolio_values[0] != 0 else 0.0

    # Calcular Retornos Diarios para Sharpe, Sortino y Volatilidad
    # Evitar divisiones por cero en retornos diarios
    daily_returns_bh = (benchmark_portfolio_values[1:] - benchmark_portfolio_values[:-1]) / (benchmark_portfolio_values[:-1] + 1e-9)
    
    volatility_bh = daily_returns_bh.std()
    
    # Anualización (asumiendo 252 días de trading en un año)
    annualization_factor = np.sqrt(252) if len(daily_returns_bh) > 0 else 1.0
    
    sharpe_ratio_bh = daily_returns_bh.mean() / volatility_bh if volatility_bh > 0 else 0.0
    sharpe_ratio_bh_annualized = sharpe_ratio_bh * annualization_factor

    negative_returns_bh = daily_returns_bh[daily_returns_bh < 0]
    downside_std_bh = negative_returns_bh.std() if len(negative_returns_bh) > 0 else 0.0
    sortino_ratio_bh = daily_returns_bh.mean() / downside_std_bh if downside_std_bh > 0 else 0.0
    sortino_ratio_bh_annualized = sortino_ratio_bh * annualization_factor

    max_drawdown_bh = _calculate_max_drawdown_for_benchmark(benchmark_portfolio_values)
    
    # Calcular retorno anualizado aproximado para el benchmark
    # total_returns_factor = (1 + benchmark_return)
    # days_in_episode = current_episode_length
    # annualized_return_bh = (total_returns_factor ** (252.0 / days_in_episode)) - 1 if days_in_episode > 0 else 0.0
    
    # Un retorno anualizado simple (para la impresión que ya tienes)
    # Asumiendo que `get_episode_stats` también devuelve `annualized_return` de una forma similar
    annualized_return_bh = benchmark_return * (252 / current_episode_length) if current_episode_length > 0 else 0.0

    return {
        'benchmark_return': benchmark_return,
        'annualized_benchmark_return': annualized_return_bh,
        'benchmark_sharpe_ratio': sharpe_ratio_bh_annualized,
        'benchmark_sortino_ratio': sortino_ratio_bh_annualized,
        'benchmark_max_drawdown': max_drawdown_bh,
        'benchmark_volatility': volatility_bh * annualization_factor # Volatilidad anualizada del benchmark
    }
# --- FIN DE CÓDIGO NECESARIO PARA EL BENCHMARK ---


def run_backtest(
    model_path = "models/best_model/best_model",
    data_path = "data/processed/NVDA_hedging_features",
    results_path = "results", # Corregido de 'restuls_path'
    commission: float = 0.00125,
    dead_zone: float = 0.03,
    initial_capital: float = 2_000_000.0,
    window_size: int = 5,
    episode_months: int = 6,
    algorithm = "PPO",
    n_episodes = 10,
    verbose = True
) -> None:
    
    if verbose: 
        print("Reading data from {}".format(data_path))
        
    df = pd.read_csv(data_path)
    df['Date'] = pd.to_datetime(df['Date'])
    prices = df['Close'].astype(np.float32).values
    features_columns = [col for col in df.columns if col not in ["Date", "Close"]]
    features = df[features_columns].astype(np.float32).values

    if np.any(prices <= 0):  
        raise ValueError("Prices contain non-positive values")
    
    # MODIFICACIÓN: Pasar el capital inicial correctamente al entorno.
    # Tu entorno espera initial_long_capital y initial_short_capital.
    # Asumo que initial_capital se divide entre ambos. Ajusta si tu lógica es diferente.
    env = PortfolioHedgingEnv(
        features=features, 
        prices=prices,
        episode_length_months=episode_months, # Nombre del parámetro corregido
        window_size=window_size,
        dead_zone=dead_zone,
        commission=commission,
        initial_long_capital=initial_capital / 2.0,  # Ejemplo de división del capital
        initial_short_capital=initial_capital / 2.0 # Ajusta según tu necesidad
    )
    
    if verbose: 
        print("Loading {} model from {}".format(algorithm, model_path))
    try:
        if algorithm.lower() == "ppo":
            model = PPO.load(model_path)
        else: 
            model = DDPG.load(model_path)
    except Exception as e: 
        raise RuntimeError(f"Failed to load {algorithm} model from {model_path}. Error: {e}")
        
    
    episodes_stats = []
    for episode in range(n_episodes):
        obs, info = env.reset()
        done = False
        
        # Guardar el índice de inicio del episodio para el cálculo del benchmark
        episode_start_idx = env.current_episode_start 
        current_episode_length_actual = 0 # Para contar los pasos reales en este episodio
        
        if verbose: 
            print("\nEpisode {}".format(episode + 1))
        
        while not done: 
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated 
            current_episode_length_actual += 1
            
        # Obtener estadísticas del agente para el episodio actual
        agent_episode_stats = env.get_episode_stats()
        
        # Calcular métricas del benchmark para el MISMO PERÍODO exacto
        # Usa el capital inicial TOTAL que el entorno maneja (long + short)
        total_initial_capital_env = env.initial_long_capital + env.initial_short_capital
        benchmark_metrics = _calculate_benchmark_metrics(
            prices_data=prices, 
            initial_total_capital=total_initial_capital_env,
            episode_start_idx=episode_start_idx,
            current_episode_length=current_episode_length_actual # Usar la longitud real del episodio
        )
        
        # Combina las estadísticas del agente y del benchmark en un solo diccionario
        combined_stats = {**agent_episode_stats, **benchmark_metrics}
        episodes_stats.append(combined_stats)
    
    stats = pd.DataFrame(episodes_stats)
    
    # Impresión de resultados (sin cambios aquí, ya que esperas estas columnas)
    print("\n" + "="*80)
    print("STRATEGY SUMMARY")
    print("="*80)
    print("PERFORMANCE:")
    print(f"  Annualized Average Return: {stats['annualized_return'].mean()*100:.2f}% ± {stats['annualized_return'].std()*100:.2f}%")
    print(f"  Average Return: {stats['total_return'].mean()*100:.2f}% ± {stats['total_return'].std()*100:.2f}%")
    print(f"  Best Episode: {stats['total_return'].max()*100:.2f}%")
    print(f"  Worst Episode: {stats['total_return'].min()*100:.2f}%")
    print(f"  Win Rate: {(stats['total_return'] > 0).mean()*100:.1f}%")
    print("RISK METRICS:")
    print(f"  Average Sharpe Ratio: {stats['sharpe_ratio'].mean():.3f} ± {stats['sharpe_ratio'].std():.3f}")
    print(f"  Average Sortino Ratio: {stats['sortino_ratio'].mean():.3f} ± {stats['sortino_ratio'].std():.3f}")
    print(f"  Average Max Drawdown: {stats['max_drawdown'].mean():.3f} ± {stats['max_drawdown'].std():.3f}")
    print(f"  Strategy Volatility: {stats['volatility'].mean()*100:.2f}%")
    print("TRADING BEHAVIOR:")
    # print(f"  Average Action: {stats['avg_action'].mean():.3f}")
    # print(f"  Action Range: [{stats['min_action'].min():.3f}, {stats['max_action'].max():.3f}]")
    # print(f"  Average Action Std: {stats['action_std'].mean():.3f}")
    # print(f"  Average Reward: {stats['total_reward'].mean():.2f} ± {stats['total_reward'].std():.2f}")
    
    print("\n" + "="*80)
    print("BENCHMARK SUMMARY")
    print("="*80)
    print("PERFORMANCE:")
    print(f"  Annualized Average Return: {stats['annualized_benchmark_return'].mean()*100:.2f}% ± {stats['annualized_benchmark_return'].std()*100:.2f}%")
    print(f"  Average Return: {stats['benchmark_return'].mean()*100:.2f}% ± {stats['benchmark_return'].std()*100:.2f}%")
    print(f"  Best Episode: {stats['benchmark_return'].max()*100:.2f}%")
    print(f"  Worst Episode: {stats['benchmark_return'].min()*100:.2f}%")
    print(f"  Win Rate: {(stats['benchmark_return'] > 0).mean()*100:.1f}%")
    print("RISK METRICS:")
    print(f"  Average Sharpe Ratio: {stats['benchmark_sharpe_ratio'].mean():.3f} ± {stats['benchmark_sharpe_ratio'].std():.3f}")
    print(f"  Average Sortino Ratio: {stats['benchmark_sortino_ratio'].mean():.3f} ± {stats['benchmark_sortino_ratio'].std():.3f}")
    print(f"  Average Max Drawdown: {stats['benchmark_max_drawdown'].mean():.3f} ± {stats['benchmark_max_drawdown'].std():.3f}")
    print(f"  Benchmark Volatility: {stats['benchmark_volatility'].mean()*100:.2f}%") # Corregido
    print("TRADING BEHAVIOR: Buy and Hold")
    
    print("\n" + "="*80)
        
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Backtest hedging model")
    
    parser.add_argument('--model_path', default="models/best_model/best_model", help='Path to model for evaluation')
    parser.add_argument('--data_path', default="data/processed/NVDA_hedging_features.csv", help='Path to processed dataset')
    parser.add_argument("--results_path", default="results", help="Path to results directory") # Corregido
    parser.add_argument('--algorithm', choices=['PPO', 'DDPG'], default='PPO', help='RL algorithm to use')
    parser.add_argument('--episode_length', type=int, default=6, help='Episode length in months')
    parser.add_argument("--n_episodes", type=int, default=10, help="Number of episodes to backtest with")
    
    args = parser.parse_args()
    
    run_backtest(
        model_path=args.model_path,
        data_path=args.data_path,
        results_path=args.results_path, # Corregido
        episode_months=args.episode_length,
        n_episodes = args.n_episodes
    )