#!/usr/bin/env python3
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import csv
import os

# Import the trading strategy
from trading_strategy import BilibiliTradingStrategy

def run_backtest(strategy, start_date, end_date, period_name):
    """Run a backtest for a specific period and return results"""
    print(f"\n===== 测试期间: {period_name} ({start_date} 至 {end_date}) =====")
    
    try:
        # Download data
        data = strategy.download_data(start_date, end_date)
        print(f"下载了 {len(data)} 天的 BILI 数据")
        
        # Calculate indicators
        strategy.calculate_indicators()
        
        # Generate signals
        strategy.generate_signals()
        
        # Run backtest
        strategy.backtest()
        
        # Analyze performance
        results = strategy.analyze_performance()
        
        # Print results
        print("\n🔹 策略性能指标:")
        for key, value in results.items():
            print(f"{key}: {value:.2f}" if isinstance(value, float) else f"{key}: {value}")
        
        # Print trade summary
        num_trades = len([t for t in strategy.all_trades if t['action'].startswith('卖出')])
        winning_trades = [t for t in strategy.all_trades if t['action'].startswith('卖出') and t.get('pnl_pct', 0) > 0]
        win_rate = len(winning_trades) / num_trades * 100 if num_trades > 0 else 0
        
        print(f"\n🔹 交易摘要:")
        print(f"总交易次数: {len(strategy.all_trades)}")
        print(f"完成交易: {num_trades}")
        print(f"胜率: {win_rate:.2f}%")
        
        # Plot results
        fig = strategy.plot_results()
        
        # Save plot
        plot_filename = f"BILI_backtest_{period_name}.png"
        plt.savefig(plot_filename)
        print(f"📊 图表已保存至 {plot_filename}")
        
        # Return results for comparison
        results['Period'] = period_name
        results['Start Date'] = start_date
        results['End Date'] = end_date
        results['Number of Days'] = len(data)
        results['Win Rate'] = win_rate
        
        return results
    
    except Exception as e:
        print(f"❌ 测试期间出错 {period_name}: {str(e)}")
        return None

def save_results_to_csv(results_list, filename="backtest_results.csv"):
    """Save results to CSV file"""
    if not results_list:
        print("没有结果可保存")
        return
    
    # Ensure all results have the same keys
    all_keys = set()
    for result in results_list:
        if result:
            all_keys.update(result.keys())
    
    # Write to CSV
    with open(filename, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=list(all_keys))
        writer.writeheader()
        for result in results_list:
            if result:
                writer.writerow(result)
    
    print(f"📝 结果已保存至 {filename}")

def main():
    """Run backtests for different time periods"""
    end_date = (datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d')  # Yesterday
    
    # Define test periods (3 months, 6 months, 12 months)
    periods = [
        ("3M", (datetime.now() - timedelta(days=90)).strftime('%Y-%m-%d'), end_date),
        ("6M", (datetime.now() - timedelta(days=180)).strftime('%Y-%m-%d'), end_date),
        ("12M", (datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d'), end_date)
    ]
    
    results_list = []
    
    # Run tests for each period
    for period_name, start_date, end_date in periods:
        # Create new strategy instance for each test
        strategy = BilibiliTradingStrategy()
        result = run_backtest(strategy, start_date, end_date, period_name)
        if result:
            results_list.append(result)
    
    # Compare results
    if results_list:
        print("\n===== 多周期测试比较 =====")
        comparison_df = pd.DataFrame(results_list)
        comparison_df = comparison_df.set_index('Period')
        
        # Select key metrics for comparison
        key_metrics = ['Total Return (%)', 'Annualized Return (%)', 'Sharpe Ratio', 
                       'Max Drawdown (%)', 'Number of Trades', 'Win Rate', 'Profit Factor']
        
        comparison = comparison_df[key_metrics]
        print(comparison)
        
        # Save all results to CSV
        save_results_to_csv(results_list)
        
        # Plot comparison
        plt.figure(figsize=(12, 8))
        
        # Plot returns
        ax1 = plt.subplot(2, 2, 1)
        comparison[['Total Return (%)', 'Annualized Return (%)']].plot(kind='bar', ax=ax1)
        ax1.set_title('Returns Comparison')
        ax1.set_ylabel('Percentage (%)')
        ax1.grid(True, axis='y')
        
        # Plot Sharpe Ratio
        ax2 = plt.subplot(2, 2, 2)
        comparison[['Sharpe Ratio']].plot(kind='bar', ax=ax2, color='green')
        ax2.set_title('Sharpe Ratio Comparison')
        ax2.grid(True, axis='y')
        
        # Plot drawdown
        ax3 = plt.subplot(2, 2, 3)
        comparison[['Max Drawdown (%)']].plot(kind='bar', ax=ax3, color='red')
        ax3.set_title('Max Drawdown Comparison')
        ax3.set_ylabel('Percentage (%)')
        ax3.grid(True, axis='y')
        
        # Plot win rate
        ax4 = plt.subplot(2, 2, 4)
        comparison[['Win Rate']].plot(kind='bar', ax=ax4, color='purple')
        ax4.set_title('Win Rate Comparison')
        ax4.set_ylabel('Percentage (%)')
        ax4.grid(True, axis='y')
        
        plt.tight_layout()
        plt.savefig('period_comparison.png')
        print(f"📊 比较图表已保存至 period_comparison.png")

if __name__ == "__main__":
    main()