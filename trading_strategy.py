import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import matplotlib.dates as mdates
from pandas.plotting import register_matplotlib_converters
import os
from scipy.signal import argrelextrema
from sklearn.cluster import KMeans

# Set proxy environment variables
os.environ['HTTPS_PROXY'] = 'http://127.0.0.1:7890'
os.environ['HTTP_PROXY'] = 'http://127.0.0.1:7890'
os.environ['ALL_PROXY'] = 'socks5://127.0.0.1:7890'

register_matplotlib_converters()

class BilibiliTradingStrategy:
    def __init__(self):
        # Strategy name
        self.name = "哔哩哔哩高级交易策略"
        
        # Set proxy configuration for yfinance
        self.proxies = {
            'https': 'http://127.0.0.1:7890',
            'http': 'http://127.0.0.1:7890',
            'socks5': 'socks5://127.0.0.1:7890'
        }
        
        # Initial parameters
        self.initial_capital = 10000
        self.position_size = 0.5       # 50% of capital per trade
        self.stop_loss_pct = 0.05      # 5% stop loss
        self.take_profit_pct = 0.15    # 15% take profit
        
        # Support and resistance parameters
        self.support_resistance_window = 15
        self.price_cluster_count = 5    
        self.support_resistance_threshold = 0.03  # 3% threshold
        self.volume_weight_factor = 0.8
        
        # Support and resistance levels will be calculated dynamically
        self.support_levels = []
        self.resistance_levels = []
        
        # Momentum and volume thresholds
        self.momentum_threshold = 0.8  # 0.8% price change as momentum
        self.volume_threshold = 1.1    # 10% above average volume
        
        # Trading state
        self.entry_price = 0
        self.stop_loss = 0
        self.take_profit = 0
        self.position = 0
        self.capital = self.initial_capital
        self.portfolio_value = self.initial_capital
        
        # Trade history
        self.all_trades = []
        
        # Performance tracking
        self.equity_curve = []
        self.daily_returns = []
        
    def calculate_support_resistance_levels(self):
        """Calculate dynamic support and resistance levels using multiple methods"""
        if len(self.data) < self.support_resistance_window:
            raise ValueError("Not enough data to calculate support and resistance levels")

        # Method 1: Local extrema
        local_min_indices = argrelextrema(self.data['Low'].values, np.less_equal, 
                                        order=self.support_resistance_window)[0]
        local_max_indices = argrelextrema(self.data['High'].values, np.greater_equal, 
                                        order=self.support_resistance_window)[0]
        
        local_mins = self.data['Low'].iloc[local_min_indices]
        local_maxs = self.data['High'].iloc[local_max_indices]
        
        # Method 2: Price clusters using K-means
        price_data = np.column_stack([
            self.data['Close'].values,
            self.data['Volume'].values * self.volume_weight_factor
        ])
        
        kmeans = KMeans(n_clusters=self.price_cluster_count, random_state=42)
        kmeans.fit(price_data)
        cluster_centers = kmeans.cluster_centers_[:, 0]  # Get only price values
        
        # Method 3: Volume-weighted price levels
        volume_profile = pd.DataFrame({
            'price': self.data['Close'],
            'volume': self.data['Volume']
        })
        volume_profile = volume_profile.groupby('price')['volume'].sum()
        high_volume_prices = volume_profile.nlargest(5).index.values
        
        # Combine all methods
        all_levels = np.concatenate([
            local_mins.values,
            local_maxs.values,
            cluster_centers,
            high_volume_prices
        ])
        
        # Remove duplicates and sort
        unique_levels = np.unique(all_levels)
        
        # Calculate median price
        median_price = self.data['Close'].median()
        
        # Separate into support and resistance levels
        self.support_levels = sorted([
            level for level in unique_levels 
            if level < median_price and 
            self.is_significant_level(level)
        ])
        
        self.resistance_levels = sorted([
            level for level in unique_levels 
            if level > median_price and 
            self.is_significant_level(level)
        ])
        
        # Keep only the most significant levels (top 3-5)
        self.support_levels = self.support_levels[-3:] if len(self.support_levels) > 3 else self.support_levels
        self.resistance_levels = self.resistance_levels[:3] if len(self.resistance_levels) > 3 else self.resistance_levels
        
        print("Calculated Support Levels:", [f"{level:.2f}" for level in self.support_levels])
        print("Calculated Resistance Levels:", [f"{level:.2f}" for level in self.resistance_levels])
        
    def is_significant_level(self, level):
        """Check if a price level is significant based on historical data"""
        # Calculate how many times price has bounced off this level
        price_touches = 0
        for i in range(len(self.data)):
            if (abs(self.data['High'].iloc[i] - level) / level <= self.support_resistance_threshold or
                abs(self.data['Low'].iloc[i] - level) / level <= self.support_resistance_threshold):
                price_touches += 1
        
        # Calculate volume around this level
        volume_around_level = self.data[
            (abs(self.data['Close'] - level) / level <= self.support_resistance_threshold)
        ]['Volume'].mean()
        
        # Level is significant if it has been touched multiple times or has high volume
        return price_touches >= 3 or volume_around_level > self.data['Volume'].mean() * 1.5

    def download_data(self, start_date, end_date):
        """Download historical data for BILI using yfinance with proxy"""
        self.ticker = "BILI"
        # Create ticker object with proxy settings
        ticker_obj = yf.Ticker(self.ticker)
        
        try:
            # Download data with proxy configuration
            self.data = ticker_obj.history(
                start=start_date,
                end=end_date,
                proxy=self.proxies
            )
            
            if self.data.empty:
                raise ValueError(f"No data retrieved for {self.ticker} in the specified period")
                
            # Make sure we have all required columns
            required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
            if not all(col in self.data.columns for col in required_columns):
                missing = [col for col in required_columns if col not in self.data.columns]
                raise ValueError(f"Missing required columns: {missing}")
            
            # Calculate support and resistance levels
            self.calculate_support_resistance_levels()
                
            return self.data
            
        except Exception as e:
            print(f"Error downloading data: {str(e)}")
            print("Attempting alternative download method...")
            try:
                # Alternative download method using yf.download directly
                self.data = yf.download(
                    self.ticker,
                    start=start_date,
                    end=end_date,
                    proxy=self.proxies
                )
                
                # Calculate support and resistance levels
                self.calculate_support_resistance_levels()
                
                return self.data
            except Exception as e2:
                raise Exception(f"Both download methods failed. Error: {str(e2)}")
        
    def calculate_indicators(self):
        """Calculate all technical indicators used by the strategy"""
        # Price moving averages
        self.data['SMA5'] = self.data['Close'].rolling(window=5).mean()
        self.data['SMA20'] = self.data['Close'].rolling(window=20).mean()
        
        # Volume indicators
        self.data['VolumeSMA5'] = self.data['Volume'].rolling(window=5).mean()
        
        # RSI (14-day)
        delta = self.data['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        self.data['RSI'] = 100 - (100 / (1 + rs))
        
        # Bollinger Bands (20-day, 2 standard deviations)
        self.data['BB_Middle'] = self.data['Close'].rolling(window=20).mean()
        std_dev = self.data['Close'].rolling(window=20).std()
        self.data['BB_Upper'] = self.data['BB_Middle'] + (std_dev * 2)
        self.data['BB_Lower'] = self.data['BB_Middle'] - (std_dev * 2)
        
        # MACD (12, 26, 9)
        self.data['EMA12'] = self.data['Close'].ewm(span=12, adjust=False).mean()
        self.data['EMA26'] = self.data['Close'].ewm(span=26, adjust=False).mean()
        self.data['MACD'] = self.data['EMA12'] - self.data['EMA26']
        self.data['Signal'] = self.data['MACD'].ewm(span=9, adjust=False).mean()
        self.data['Histogram'] = self.data['MACD'] - self.data['Signal']
        
        # Previous closes for momentum calculation
        self.data['PrevClose1'] = self.data['Close'].shift(1)
        self.data['PrevClose3'] = self.data['Close'].shift(3)
        
        # Daily change percentage
        self.data['DailyChange'] = (self.data['Close'] / self.data['PrevClose1'] - 1) * 100
        
        # Momentum (3-day percentage change)
        self.data['Momentum'] = (self.data['Close'] / self.data['PrevClose3'] - 1) * 100
        
        # Volume ratio
        self.data['VolumeRatio'] = self.data['Volume'] / self.data['VolumeSMA5']
        
        # Drop NaN values that resulted from calculations
        self.data = self.data.dropna()
        
        return self.data
    
    def check_support_resistance(self, price):
        """Check if price is near support or resistance levels"""
        near_support = False
        closest_support = 0
        for level in self.support_levels:
            if abs(price - level) / level <= 0.02:  # Price within 2% of support
                near_support = True
                closest_support = level
                break
                
        near_resistance = False
        closest_resistance = 0
        for level in self.resistance_levels:
            if abs(price - level) / level <= 0.02:  # Price within 2% of resistance
                near_resistance = True
                closest_resistance = level
                break
                
        return near_support, closest_support, near_resistance, closest_resistance
    
    def generate_signals(self):
        """Generate trading signals based on technical indicators"""
        self.data['Signal'] = 0  # Initialize signal column
        self.data['SignalReasons'] = ""  # To store reasons for signals
        
        for i in range(3, len(self.data)):
            current_idx = self.data.index[i]
            prev_idx = self.data.index[i-1]
            prev2_idx = self.data.index[i-2]
            prev3_idx = self.data.index[i-3]
            
            # Current values
            current_price = self.data.loc[current_idx, 'Close']
            momentum = self.data.loc[current_idx, 'Momentum']
            volume_ratio = self.data.loc[current_idx, 'VolumeRatio']
            daily_change = self.data.loc[current_idx, 'DailyChange']
            
            # Initialize signal and reasons
            signal = 0
            signal_reasons = []
            
            # 1. Support and resistance strategy
            near_support, support_level, near_resistance, resistance_level = self.check_support_resistance(current_price)
            
            if near_support:
                signal += 1
                signal_reasons.append(f"接近支撑位 {support_level}")
                
            if near_resistance:
                signal -= 1
                signal_reasons.append(f"接近阻力位 {resistance_level}")
                
            # 2. Momentum strategy with volume confirmation
            if momentum > self.momentum_threshold and volume_ratio > self.volume_threshold:
                signal += 1
                signal_reasons.append(f"正动量 {momentum:.2f}% 配合放量 {volume_ratio:.2f}x")
                
            if momentum < -self.momentum_threshold and volume_ratio > self.volume_threshold:
                signal -= 1
                signal_reasons.append(f"负动量 {momentum:.2f}% 配合放量 {volume_ratio:.2f}x")
                
            # 3. Moving average strategy
            # Golden cross (5-day MA crosses above 20-day MA)
            if (self.data.loc[current_idx, 'SMA5'] > self.data.loc[current_idx, 'SMA20'] and 
                self.data.loc[prev_idx, 'SMA5'] <= self.data.loc[prev_idx, 'SMA20']):
                signal += 1
                signal_reasons.append("均线金叉")
                
            # Death cross (5-day MA crosses below 20-day MA)
            if (self.data.loc[current_idx, 'SMA5'] < self.data.loc[current_idx, 'SMA20'] and 
                self.data.loc[prev_idx, 'SMA5'] >= self.data.loc[prev_idx, 'SMA20']):
                signal -= 1
                signal_reasons.append("均线死叉")
                
            # 4. RSI strategy
            # Oversold region
            if self.data.loc[current_idx, 'RSI'] < 30:
                signal += 1
                signal_reasons.append(f"RSI超卖 {self.data.loc[current_idx, 'RSI']:.2f}")
                
            # Overbought region
            if self.data.loc[current_idx, 'RSI'] > 70:
                signal -= 1
                signal_reasons.append(f"RSI超买 {self.data.loc[current_idx, 'RSI']:.2f}")
                
            # 5. Bollinger Band strategy
            # Price touches lower band
            if current_price <= self.data.loc[current_idx, 'BB_Lower']:
                signal += 1
                signal_reasons.append("价格触及布林带下轨")
                
            # Price touches upper band
            if current_price >= self.data.loc[current_idx, 'BB_Upper']:
                signal -= 1
                signal_reasons.append("价格触及布林带上轨")
                
            # 6. Price reversal strategy
            # Rebound after two consecutive down days
            if (daily_change > 0 and 
                self.data.loc[prev_idx, 'DailyChange'] < 0 and 
                self.data.loc[prev2_idx, 'DailyChange'] < 0):
                signal += 1
                signal_reasons.append("价格反转向上")
                
            # Correction after two consecutive up days
            if (daily_change < 0 and 
                self.data.loc[prev_idx, 'DailyChange'] > 0 and 
                self.data.loc[prev2_idx, 'DailyChange'] > 0):
                signal -= 1
                signal_reasons.append("价格反转向下")
                
            # Normalize signal to -1, 0, 1
            # Require less confirming signals
            if signal >= 1:
                signal = 1
            elif signal <= -1:
                signal = -1
            else:
                signal = 0
                
            # Store signal and reasons
            self.data.loc[current_idx, 'Signal'] = signal
            self.data.loc[current_idx, 'SignalReasons'] = ", ".join(signal_reasons)
    
    def backtest(self):
        """Run backtest on the strategy"""
        # Initialize columns for trade tracking
        self.data['Position'] = 0
        self.data['Capital'] = self.initial_capital
        self.data['Holdings'] = 0
        self.data['PortfolioValue'] = self.initial_capital
        
        # Trade state
        position = 0
        entry_price = 0
        stop_loss = 0
        take_profit = 0
        capital = self.initial_capital
        
        for i in range(1, len(self.data)):
            current_idx = self.data.index[i]
            prev_idx = self.data.index[i-1]
            
            # Current values
            current_price = self.data.loc[current_idx, 'Close']
            signal = self.data.loc[current_idx, 'Signal']
            signal_reasons = self.data.loc[current_idx, 'SignalReasons']
            
            # Carry forward values by default
            self.data.loc[current_idx, 'Position'] = position
            self.data.loc[current_idx, 'Capital'] = capital
            self.data.loc[current_idx, 'Holdings'] = position * current_price
            self.data.loc[current_idx, 'PortfolioValue'] = capital + (position * current_price)
            
            # Check if we have an existing position
            if position > 0:
                # Check for stop loss
                if current_price <= stop_loss:
                    # Sell (stop loss hit)
                    loss_pct = (current_price / entry_price - 1) * 100
                    capital += position * current_price
                    
                    # Record trade
                    self.all_trades.append({
                        'date': current_idx.strftime("%Y-%m-%d"),
                        'action': '卖出 (止损)',
                        'price': current_price,
                        'shares': position,
                        'pnl_pct': loss_pct,
                        'reason': '触及止损位'
                    })
                    
                    # Update state
                    position = 0
                    entry_price = 0
                    stop_loss = 0
                    take_profit = 0
                    
                    # Update data
                    self.data.loc[current_idx, 'Position'] = position
                    self.data.loc[current_idx, 'Capital'] = capital
                    self.data.loc[current_idx, 'Holdings'] = 0
                    self.data.loc[current_idx, 'PortfolioValue'] = capital
                    
                # Check for take profit
                elif current_price >= take_profit:
                    # Sell (take profit hit)
                    profit_pct = (current_price / entry_price - 1) * 100
                    capital += position * current_price
                    
                    # Record trade
                    self.all_trades.append({
                        'date': current_idx.strftime("%Y-%m-%d"),
                        'action': '卖出 (止盈)',
                        'price': current_price,
                        'shares': position,
                        'pnl_pct': profit_pct,
                        'reason': '触及止盈位'
                    })
                    
                    # Update state
                    position = 0
                    entry_price = 0
                    stop_loss = 0
                    take_profit = 0
                    
                    # Update data
                    self.data.loc[current_idx, 'Position'] = position
                    self.data.loc[current_idx, 'Capital'] = capital
                    self.data.loc[current_idx, 'Holdings'] = 0
                    self.data.loc[current_idx, 'PortfolioValue'] = capital
                    
                # Check for sell signal
                elif signal == -1:
                    # Sell (signal)
                    profit_pct = (current_price / entry_price - 1) * 100
                    capital += position * current_price
                    
                    # Record trade
                    self.all_trades.append({
                        'date': current_idx.strftime("%Y-%m-%d"),
                        'action': '卖出 (信号)',
                        'price': current_price,
                        'shares': position,
                        'pnl_pct': profit_pct,
                        'reason': signal_reasons
                    })
                    
                    # Update state
                    position = 0
                    entry_price = 0
                    stop_loss = 0
                    take_profit = 0
                    
                    # Update data
                    self.data.loc[current_idx, 'Position'] = position
                    self.data.loc[current_idx, 'Capital'] = capital
                    self.data.loc[current_idx, 'Holdings'] = 0
                    self.data.loc[current_idx, 'PortfolioValue'] = capital
                    
            # No position - check for buy signal
            elif position == 0 and signal == 1:
                # Calculate position size
                invest_amount = capital * self.position_size
                shares_to_buy = int(invest_amount / current_price)
                
                if shares_to_buy > 0:
                    # Buy
                    entry_price = current_price
                    stop_loss = current_price * (1 - self.stop_loss_pct)
                    take_profit = current_price * (1 + self.take_profit_pct)
                    capital -= shares_to_buy * current_price
                    position = shares_to_buy
                    
                    # Record trade
                    self.all_trades.append({
                        'date': current_idx.strftime("%Y-%m-%d"),
                        'action': '买入',
                        'price': current_price,
                        'shares': shares_to_buy,
                        'stop_loss': stop_loss,
                        'take_profit': take_profit,
                        'reason': signal_reasons
                    })
                    
                    # Update data
                    self.data.loc[current_idx, 'Position'] = position
                    self.data.loc[current_idx, 'Capital'] = capital
                    self.data.loc[current_idx, 'Holdings'] = position * current_price
                    self.data.loc[current_idx, 'PortfolioValue'] = capital + (position * current_price)
        
        # Store the equity curve
        self.equity_curve = self.data['PortfolioValue']
        
        # Calculate daily returns
        self.data['DailyReturn'] = self.data['PortfolioValue'].pct_change() * 100
        self.daily_returns = self.data['DailyReturn'].dropna()
        
        return self.data
    
    def analyze_performance(self):
        """Calculate and display performance metrics"""
        # Calculate performance metrics
        initial_value = self.initial_capital
        final_value = self.data['PortfolioValue'].iloc[-1]
        total_return = (final_value / initial_value - 1) * 100
        
        # Calculate annualized return
        days = (self.data.index[-1] - self.data.index[0]).days
        annualized_return = ((1 + total_return/100) ** (365.0/days) - 1) * 100
        
        # Calculate Sharpe ratio (assuming risk-free rate of 0%)
        if len(self.daily_returns) > 0:
            sharpe_ratio = self.daily_returns.mean() / self.daily_returns.std() * (252 ** 0.5)
        else:
            sharpe_ratio = 0
            
        # Calculate maximum drawdown
        rolling_max = self.data['PortfolioValue'].cummax()
        drawdown = (self.data['PortfolioValue'] / rolling_max - 1) * 100
        max_drawdown = drawdown.min()
        
        # Trade statistics
        num_trades = len([t for t in self.all_trades if t['action'].startswith('卖出')])
        
        if num_trades > 0:
            winning_trades = [t for t in self.all_trades if t['action'].startswith('卖出') and t.get('pnl_pct', 0) > 0]
            losing_trades = [t for t in self.all_trades if t['action'].startswith('卖出') and t.get('pnl_pct', 0) <= 0]
            
            win_rate = len(winning_trades) / num_trades * 100
            
            avg_win = sum([t.get('pnl_pct', 0) for t in winning_trades]) / len(winning_trades) if winning_trades else 0
            avg_loss = sum([t.get('pnl_pct', 0) for t in losing_trades]) / len(losing_trades) if losing_trades else 0
            
            # Profit factor
            gross_profit = sum([t.get('pnl_pct', 0) for t in winning_trades]) if winning_trades else 0
            gross_loss = abs(sum([t.get('pnl_pct', 0) for t in losing_trades])) if losing_trades else 0
            profit_factor = gross_profit / gross_loss if gross_loss != 0 else float('inf')
        else:
            win_rate = 0
            avg_win = 0
            avg_loss = 0
            profit_factor = 0
            
        # Create results dictionary
        results = {
            'Initial Capital': initial_value,
            'Final Value': final_value,
            'Total Return (%)': total_return,
            'Annualized Return (%)': annualized_return,
            'Sharpe Ratio': sharpe_ratio,
            'Max Drawdown (%)': max_drawdown,
            'Number of Trades': num_trades,
            'Win Rate (%)': win_rate,
            'Average Win (%)': avg_win,
            'Average Loss (%)': avg_loss,
            'Profit Factor': profit_factor
        }
        
        return results
    
    def plot_results(self):
        """Plot the backtest results"""
        # Create a 3x2 grid of subplots
        fig = plt.figure(figsize=(15, 15))
        
        # 1. Price chart with indicators
        ax1 = plt.subplot2grid((3, 2), (0, 0), colspan=2)
        ax1.set_title('BILI Price with Indicators')
        ax1.plot(self.data.index, self.data['Close'], label='Price')
        ax1.plot(self.data.index, self.data['SMA5'], label='SMA5')
        ax1.plot(self.data.index, self.data['SMA20'], label='SMA20')
        ax1.plot(self.data.index, self.data['BB_Upper'], label='BB Upper', linestyle='--')
        ax1.plot(self.data.index, self.data['BB_Lower'], label='BB Lower', linestyle='--')
        
        # Plot buy and sell signals
        buy_signals = self.data[self.data['SignalReasons'].str.contains('买入', na=False)]
        sell_signals = self.data[self.data['SignalReasons'].str.contains('卖出', na=False)]
        
        for trade in self.all_trades:
            if trade['action'] == '买入':
                date = pd.to_datetime(trade['date'])
                ax1.plot(date, trade['price'], '^', markersize=10, color='g')
            elif '卖出' in trade['action']:
                date = pd.to_datetime(trade['date'])
                ax1.plot(date, trade['price'], 'v', markersize=10, color='r')
        
        # Add support and resistance levels
        for level in self.support_levels:
            ax1.axhline(y=level, color='g', linestyle='--', alpha=0.5, label=f'Support {level}')
        for level in self.resistance_levels:
            ax1.axhline(y=level, color='r', linestyle='--', alpha=0.5, label=f'Resistance {level}')
            
        ax1.legend()
        ax1.grid(True)
        
        # 2. Volume
        ax2 = plt.subplot2grid((3, 2), (1, 0))
        ax2.set_title('Volume')
        ax2.bar(self.data.index, self.data['Volume'])
        ax2.plot(self.data.index, self.data['VolumeSMA5'], color='r')
        ax2.grid(True)
        
        # 3. RSI
        ax3 = plt.subplot2grid((3, 2), (1, 1))
        ax3.set_title('RSI')
        ax3.plot(self.data.index, self.data['RSI'])
        ax3.axhline(y=70, color='r', linestyle='--')
        ax3.axhline(y=30, color='g', linestyle='--')
        ax3.grid(True)
        
        # 4. MACD
        ax4 = plt.subplot2grid((3, 2), (2, 0))
        ax4.set_title('MACD')
        ax4.plot(self.data.index, self.data['MACD'], label='MACD')
        ax4.plot(self.data.index, self.data['Signal'], label='Signal')
        ax4.bar(self.data.index, self.data['Histogram'], label='Histogram')
        ax4.legend()
        ax4.grid(True)
        
        # 5. Portfolio Value
        ax5 = plt.subplot2grid((3, 2), (2, 1))
        ax5.set_title('Portfolio Value')
        ax5.plot(self.data.index, self.data['PortfolioValue'])
        ax5.grid(True)
        
        # Format x-axis dates
        for ax in [ax1, ax2, ax3, ax4, ax5]:
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
            ax.xaxis.set_major_locator(mdates.WeekdayLocator(interval=2))
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
        
        plt.tight_layout()
        return fig