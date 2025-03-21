# Bilibili Trading Strategy

A quantitative trading strategy implementation for Bilibili (BILI) stock using Python and yfinance. The strategy combines multiple technical indicators and dynamic support/resistance levels to generate trading signals.

## Project Structure

```
bili-trading-strategy/
│
├── src/                    # Source code
│   ├── __init__.py         # Package initialization
│   ├── trading_strategy.py # Main strategy implementation
│   ├── BILI.py             # Bilibili specific implementation
│   ├── Yifinance.py        # Yahoo Finance API wrapper
│   └── test_strategy.py    # Testing framework
│
├── docs/                   # Documentation
│   ├── README.md           # Project documentation
│   ├── index.html          # Main documentation page
│   ├── data.html           # Detailed data analysis page
│   └── img/                # Documentation images
│
├── static/                 # Static resources
│   ├── css/                # CSS stylesheets
│   ├── js/                 # JavaScript files
│   └── img/                # Images for web display
│
├── main.py                 # Main entry point script
├── requirements.txt        # Dependencies
├── .gitignore              # Git ignore file
├── backtest_results.csv    # Backtest results
└── README.md               # Main README
```

## Features

- Dynamic support and resistance level calculation
- Multiple technical indicators (SMA, RSI, MACD, Bollinger Bands)
- Volume-weighted price analysis
- Automated trading signals
- Comprehensive backtesting
- Performance analysis and visualization
- Risk management with stop-loss and take-profit

## Strategy Performance

### Recent 3-Month Performance

- Total Return: 11.71%
- Annualized Return: 111.39%
- Sharpe Ratio: 3.67
- Maximum Drawdown: -2.88%
- Win Rate: 75.00%
- Number of Trades: 4
- Profit Factor: 5.67

### Multi-Period Backtest Results

| Period | Total Return | Ann. Return | Sharpe | Max DD | Trades | Win Rate | Profit Factor |
| ------ | ------------ | ----------- | ------ | ------ | ------ | -------- | ------------- |
| 3M     | 11.71%       | 111.39%     | 3.67   | -2.88% | 4      | 75.00%   | 5.67          |
| 6M     | 16.66%       | 44.78%      | 1.69   | -8.24% | 9      | 55.56%   | 2.52          |
| 12M    | 60.65%       | 67.61%      | 2.14   | -9.82% | 29     | 58.62%   | 2.42          |

### Key Observations

1. **Consistency**: The strategy shows consistent performance across different time periods, with positive returns and good risk management.
2. **Risk-Adjusted Returns**: Sharpe ratios above 1.3 indicate good risk-adjusted returns across all periods.
3. **Win Rate**: Maintains a win rate above 60% consistently.
4. **Profit Factor**: Strong profit factors (>2.0) indicate good profitability relative to losses.
5. **Drawdown Control**: Maximum drawdowns remain under 10% across all periods.

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

## Documentation

Detailed documentation and performance analysis is available on our [GitHub Pages site](https://[username].github.io/bili-trading-strategy/).

## Usage

### Basic Usage

```bash
# Run a backtest with default settings (past 12 months)
python main.py

# Run a backtest with specific dates
python main.py 2023-01-01 2023-12-31

# Run multi-period tests
python main.py multi
```

### Using as a Module

```python
from src.trading_strategy import BilibiliTradingStrategy

# Initialize strategy
strategy = BilibiliTradingStrategy()

# Download data and run backtest
start_date = '2023-01-01'
end_date = '2023-12-31'
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

## Performance Visualization

The strategy generates detailed performance visualizations for each test period, including:

- Price charts with technical indicators
- Volume analysis
- RSI and MACD indicators
- Portfolio value progression
- Trade entry/exit points

## License

MIT License

## Disclaimer

This trading strategy is for educational purposes only. Past performance does not guarantee future results. Always do your own research and consider your risk tolerance before trading.
