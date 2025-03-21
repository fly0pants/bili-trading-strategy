#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Bilibili Trading Strategy - Main Program Entry
This file serves as the main entry point for running the Bilibili trading strategy.
"""

import sys
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from src.trading_strategy import BilibiliTradingStrategy

def run_single_backtest(start_date=None, end_date=None):
    """
    Run a single backtest for the specified period.
    
    Args:
        start_date (str): Start date in format 'YYYY-MM-DD'
        end_date (str): End date in format 'YYYY-MM-DD'
    """
    # Use default dates if none provided
    if not start_date:
        start_date = (datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d')
    if not end_date:
        end_date = (datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d')

    # Initialize strategy
    strategy = BilibiliTradingStrategy()

    # Download data and run backtest
    print(f"Running backtest from {start_date} to {end_date}...")
    data = strategy.download_data(start_date, end_date)
    
    # Calculate indicators and generate signals
    strategy.calculate_indicators()
    strategy.generate_signals()
    
    # Run backtest
    strategy.backtest()
    
    # Analyze performance
    results = strategy.analyze_performance()
    
    # Plot results
    strategy.plot_results()
    plt.show()
    
    return results

def main():
    """
    Main function to run the trading strategy based on command line arguments.
    """
    if len(sys.argv) > 1:
        # Check if user wants to run multi-period tests
        if sys.argv[1] == "multi":
            from src.test_strategy import run_multi_period_tests
            run_multi_period_tests()
        # Check if user wants a specific time period
        elif len(sys.argv) >= 3:
            start_date = sys.argv[1]
            end_date = sys.argv[2]
            run_single_backtest(start_date, end_date)
        else:
            print("Invalid arguments. Usage: python main.py [multi] or python main.py [start_date] [end_date]")
    else:
        # Run default single backtest for the past year
        run_single_backtest()

if __name__ == "__main__":
    main() 