# Bilibili Trading Strategy

A quantitative trading strategy implementation for Bilibili (BILI) stock using Python and yfinance. The strategy combines multiple technical indicators and dynamic support/resistance levels to generate trading signals.

## Features

- Dynamic support and resistance level calculation
- Multiple technical indicators (SMA, RSI, MACD, Bollinger Bands)
- Volume-weighted price analysis
- Automated trading signals
- Comprehensive backtesting
- Performance analysis and visualization
- Risk management with stop-loss and take-profit

## Strategy Performance

Based on recent backtesting results:

- Total Return: 4.71%
- Annualized Return: 16.67%
- Sharpe Ratio: 1.47
- Maximum Drawdown: -3.96%
- Win Rate: 66.67%
- Profit Factor: 2.24

## Requirements

- Python 3.8+
- yfinance
- pandas
- numpy
- matplotlib
- scikit-learn
- scipy

## Installation

1. Clone the repository:
```bash
git clone https://github.com/fly0pants/bili-trading-strategy.git
cd bili-trading-strategy
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

```python
from trading_strategy import BilibiliTradingStrategy

# Initialize strategy
strategy = BilibiliTradingStrategy()

# Download data and run backtest
start_date = '2023-01-01'
end_date = (datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d')
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
```

## Strategy Components

1. Technical Indicators:
   - Simple Moving Averages (5 and 20-day)
   - Relative Strength Index (RSI)
   - Moving Average Convergence Divergence (MACD)
   - Bollinger Bands

2. Entry Signals:
   - Support/Resistance levels
   - Momentum with volume confirmation
   - Moving average crossovers
   - RSI oversold/overbought
   - Bollinger Band breakouts
   - Price reversal patterns

3. Risk Management:
   - Position sizing (50% of capital)
   - Stop-loss (5%)
   - Take-profit (15%)

## License

MIT License

## Disclaimer

This trading strategy is for educational purposes only. Past performance does not guarantee future results. Always do your own research and consider your risk tolerance before trading.