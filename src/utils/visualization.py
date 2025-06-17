import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd
import numpy as np
import seaborn as sns
from datetime import datetime, timedelta

def plot_portfolio_evolution(portfolio_history, position_history, dates, prices, 
                           title="Portfolio Hedging Evolution", save_path=None):
    """
    Plot the evolution of portfolio value and positions over time.
    
    Args:
        portfolio_history: List of portfolio values over time
        position_history: List of positions over time
        dates: Array of dates corresponding to the history
        prices: Array of underlying asset prices
        title: Plot title
        save_path: Path to save the plot
    """
    fig, axes = plt.subplots(3, 1, figsize=(14, 10))
    
    # Ensure we have matching lengths
    min_len = min(len(portfolio_history), len(position_history), len(dates), len(prices))
    portfolio_history = portfolio_history[:min_len]
    position_history = position_history[:min_len]
    dates = dates[:min_len]
    prices = prices[:min_len]
    
    # Plot 1: Portfolio value vs benchmark
    axes[0].plot(dates, portfolio_history, label='Hedged Portfolio', linewidth=2, color='blue')
    
    # Calculate buy-and-hold benchmark
    initial_portfolio = portfolio_history[0]
    benchmark = initial_portfolio * (prices / prices[0])
    axes[0].plot(dates, benchmark, label='Buy & Hold Benchmark', linewidth=2, color='red', alpha=0.7)
    
    axes[0].set_ylabel('Portfolio Value ($)')
    axes[0].set_title(f'{title} - Portfolio Performance')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    axes[0].yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))
    
    # Plot 2: Position over time
    axes[1].plot(dates, position_history, label='Position', linewidth=2, color='green')
    axes[1].axhline(y=1.0, color='black', linestyle='--', alpha=0.5, label='Neutral (1.0)')
    axes[1].axhline(y=1.09, color='orange', linestyle=':', alpha=0.5, label='Dead Zone')
    axes[1].axhline(y=0.91, color='orange', linestyle=':', alpha=0.5)
    axes[1].fill_between(dates, 0.91, 1.09, alpha=0.2, color='yellow', label='Dead Zone')
    
    axes[1].set_ylabel('Position')
    axes[1].set_title('Position Over Time')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    # Plot 3: Underlying asset price
    axes[2].plot(dates, prices, label='NVDA Price', linewidth=2, color='purple')
    axes[2].set_ylabel('Price ($)')
    axes[2].set_xlabel('Date')
    axes[2].set_title('Underlying Asset Price')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)
    
    # Format x-axis
    for ax in axes:
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
    plt.show()

def plot_performance_heatmap(results_df, save_path=None):
    """
    Create a heatmap showing performance across different time periods.
    
    Args:
        results_df: DataFrame with episode results including start_date and returns
        save_path: Path to save the plot
    """
    # Extract year and month from start dates
    results_df['year'] = pd.to_datetime(results_df['start_date']).dt.year
    results_df['month'] = pd.to_datetime(results_df['start_date']).dt.month
    
    # Create pivot table for heatmap
    heatmap_data = results_df.pivot_table(
        values='total_return', 
        index='year', 
        columns='month', 
        aggfunc='mean'
    )
    
    plt.figure(figsize=(12, 8))
    sns.heatmap(
        heatmap_data * 100,  # Convert to percentage
        annot=True, 
        fmt='.1f', 
        cmap='RdYlGn', 
        center=0,
        cbar_kws={'label': 'Average Return (%)'}
    )
    
    plt.title('Average Returns by Start Month and Year')
    plt.xlabel('Month')
    plt.ylabel('Year')
    
    # Set month labels
    month_labels = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                   'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    plt.xticks(range(1, 13), month_labels)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
    plt.show()

def plot_risk_return_scatter(results_df, benchmark_col='benchmark_return', save_path=None):
    """
    Create a risk-return scatter plot.
    
    Args:
        results_df: DataFrame with episode results
        benchmark_col: Column name for benchmark returns
        save_path: Path to save the plot
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Risk-Return scatter
    ax1.scatter(results_df['volatility'] * 100, results_df['total_return'] * 100, 
               alpha=0.6, s=50, c=results_df['sharpe_ratio'], cmap='viridis')
    
    ax1.set_xlabel('Volatility (%)')
    ax1.set_ylabel('Return (%)')
    ax1.set_title('Risk-Return Profile (colored by Sharpe Ratio)')
    ax1.grid(True, alpha=0.3)
    
    # Add colorbar
    cbar = plt.colorbar(ax1.collections[0], ax=ax1)
    cbar.set_label('Sharpe Ratio')
    
    # Return vs Benchmark scatter
    ax2.scatter(results_df[benchmark_col] * 100, results_df['total_return'] * 100, 
               alpha=0.6, s=50)
    
    # Add diagonal line for equal performance
    min_ret = min(results_df[benchmark_col].min(), results_df['total_return'].min()) * 100
    max_ret = max(results_df[benchmark_col].max(), results_df['total_return'].max()) * 100
    ax2.plot([min_ret, max_ret], [min_ret, max_ret], 'r--', alpha=0.7, label='Equal Performance')
    
    ax2.set_xlabel('Benchmark Return (%)')
    ax2.set_ylabel('Strategy Return (%)')
    ax2.set_title('Strategy vs Benchmark Performance')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
    plt.show()

def plot_position_distribution(position_history, save_path=None):
    """
    Plot the distribution of positions taken by the agent.
    
    Args:
        position_history: List or array of position values
        save_path: Path to save the plot
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Histogram of positions
    ax1.hist(position_history, bins=30, alpha=0.7, edgecolor='black')
    ax1.axvline(1.0, color='red', linestyle='--', label='Neutral')
    ax1.axvline(0.91, color='orange', linestyle=':', label='Dead Zone')
    ax1.axvline(1.09, color='orange', linestyle=':')
    ax1.set_xlabel('Position')
    ax1.set_ylabel('Frequency')
    ax1.set_title('Distribution of Positions')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Time series of positions
    ax2.plot(position_history, alpha=0.7)
    ax2.axhline(1.0, color='red', linestyle='--', alpha=0.5, label='Neutral')
    ax2.axhline(0.91, color='orange', linestyle=':', alpha=0.5)
    ax2.axhline(1.09, color='orange', linestyle=':', alpha=0.5)
    ax2.fill_between(range(len(position_history)), 0.91, 1.09, alpha=0.2, color='yellow')
    ax2.set_xlabel('Time Step')
    ax2.set_ylabel('Position')
    ax2.set_title('Position Over Time')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
    plt.show()

def plot_feature_importance_heatmap(features_df, target_col='returns', n_features=20, save_path=None):
    """
    Plot correlation heatmap between features and target variable.
    
    Args:
        features_df: DataFrame with features
        target_col: Target column name
        n_features: Number of top features to show
        save_path: Path to save the plot
    """
    # Calculate correlations with target
    correlations = features_df.corr()[target_col].abs().sort_values(ascending=False)
    
    # Select top features
    top_features = correlations.head(n_features).index.tolist()
    if target_col not in top_features:
        top_features.append(target_col)
    
    # Create correlation matrix for top features
    corr_matrix = features_df[top_features].corr()
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, 
                square=True, fmt='.2f', cbar_kws={'label': 'Correlation'})
    
    plt.title(f'Feature Correlation Heatmap (Top {n_features} features)')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
    plt.show()

def plot_monthly_returns_boxplot(results_df, save_path=None):
    """
    Create a boxplot of returns by month.
    
    Args:
        results_df: DataFrame with episode results
        save_path: Path to save the plot
    """
    # Extract month from start dates
    results_df['month'] = pd.to_datetime(results_df['start_date']).dt.month
    
    plt.figure(figsize=(12, 6))
    
    # Create boxplot
    box_data = [results_df[results_df['month'] == month]['total_return'] * 100 
                for month in range(1, 13)]
    
    plt.boxplot(box_data, labels=['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                                 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
    
    plt.axhline(y=0, color='red', linestyle='--', alpha=0.5)
    plt.ylabel('Episode Return (%)')
    plt.xlabel('Start Month')
    plt.title('Distribution of Returns by Start Month')
    plt.grid(True, alpha=0.3)
    
    # Add mean line
    overall_mean = results_df['total_return'].mean() * 100
    plt.axhline(y=overall_mean, color='blue', linestyle='-', alpha=0.7, 
                label=f'Overall Mean: {overall_mean:.1f}%')
    plt.legend()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
    plt.show()

def create_performance_dashboard(results_df, portfolio_history=None, position_history=None, 
                               dates=None, prices=None, save_dir=None):
    """
    Create a comprehensive performance dashboard.
    
    Args:
        results_df: DataFrame with episode results
        portfolio_history: Optional portfolio evolution data
        position_history: Optional position evolution data  
        dates: Optional dates array
        prices: Optional prices array
        save_dir: Directory to save plots
    """
    if save_dir:
        import os
        os.makedirs(save_dir, exist_ok=True)
    
    # 1. Risk-return analysis
    plot_risk_return_scatter(
        results_df, 
        save_path=f"{save_dir}/risk_return_analysis.png" if save_dir else None
    )
    
    # 2. Performance heatmap
    plot_performance_heatmap(
        results_df,
        save_path=f"{save_dir}/performance_heatmap.png" if save_dir else None
    )
    
    # 3. Monthly returns distribution
    plot_monthly_returns_boxplot(
        results_df,
        save_path=f"{save_dir}/monthly_returns.png" if save_dir else None
    )
    
    # 4. Portfolio evolution (if data provided)
    if all(x is not None for x in [portfolio_history, position_history, dates, prices]):
        plot_portfolio_evolution(
            portfolio_history, position_history, dates, prices,
            save_path=f"{save_dir}/portfolio_evolution.png" if save_dir else None
        )
    
    # 5. Position distribution (if data provided)
    if position_history is not None:
        plot_position_distribution(
            position_history,
            save_path=f"{save_dir}/position_distribution.png" if save_dir else None
        )
    
    print("Performance dashboard created successfully!")
    if save_dir:
        print(f"All plots saved to: {save_dir}")
