import pandas as pd
from datetime import datetime, timedelta
from trading_strategy import BilibiliTradingStrategy
import matplotlib.pyplot as plt

def run_backtest(period_months):
    """Run backtest for a specific time period"""
    strategy = BilibiliTradingStrategy()
    
    # Calculate date range
    end_date = datetime.now() - timedelta(days=1)
    start_date = end_date - timedelta(days=30*period_months)
    
    # Format dates
    start_date_str = start_date.strftime('%Y-%m-%d')
    end_date_str = end_date.strftime('%Y-%m-%d')
    
    print(f"\n=== Testing {period_months} Month(s) Period ===")
    print(f"Date Range: {start_date_str} to {end_date_str}")
    
    try:
        # Download and process data
        data = strategy.download_data(start_date_str, end_date_str)
        strategy.calculate_indicators()
        strategy.generate_signals()
        strategy.backtest()
        
        # Get results
        results = strategy.analyze_performance()
        
        # Print detailed results
        print("\nPerformance Metrics:")
        for key, value in results.items():
            print(f"{key}: {value:.2f}" if isinstance(value, float) else f"{key}: {value}")
        
        # Print trade details
        print("\nTrade Details:")
        for i, trade in enumerate(strategy.all_trades, 1):
            print(f"Trade {i}: {trade['date']} - {trade['action']} at ${trade['price']:.2f}")
            if 'pnl_pct' in trade:
                print(f"  P&L: {trade['pnl_pct']:.2f}%")
        
        # Plot results
        fig = strategy.plot_results()
        plt.savefig(f'backtest_results_{period_months}m.png')
        plt.close()
        
        return results
        
    except Exception as e:
        print(f"Error during {period_months}-month backtest: {str(e)}")
        return None

def main():
    """Run backtests for different time periods and save results"""
    periods = [3, 6, 12]
    results = {}
    
    for period in periods:
        results[period] = run_backtest(period)
    
    # Create summary DataFrame
    summary_data = []
    for period, result in results.items():
        if result:
            summary_data.append({
                'Period (Months)': period,
                'Total Return (%)': result['Total Return (%)'],
                'Annualized Return (%)': result['Annualized Return (%)'],
                'Sharpe Ratio': result['Sharpe Ratio'],
                'Max Drawdown (%)': result['Max Drawdown (%)'],
                'Number of Trades': result['Number of Trades'],
                'Win Rate (%)': result['Win Rate (%)'],
                'Profit Factor': result['Profit Factor']
            })
    
    summary_df = pd.DataFrame(summary_data)
    summary_df.to_csv('backtest_summary.csv', index=False)
    
    # Print summary table
    print("\n=== Backtest Summary ===")
    print(summary_df.to_string(index=False))

if __name__ == "__main__":
    main()