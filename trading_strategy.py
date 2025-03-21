import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import matplotlib.dates as mdates
from pandas.plotting import register_matplotlib_converters
from scipy.signal import argrelextrema
from sklearn.cluster import KMeans

register_matplotlib_converters()

class BilibiliTradingStrategy:
    def __init__(self):
        # Strategy name
        self.name = "哔哩哔哩高级交易策略"
        
        # Initial parameters
        self.initial_capital = 10000
        self.position_size = 0.5       # Use 50% of capital per trade
        self.stop_loss_pct = 0.05      # 5% stop loss
        self.take_profit_pct = 0.15    # 15% take profit
        
        # Support and resistance parameters
        self.support_resistance_window = 15
        self.price_cluster_count = 5
        self.support_resistance_threshold = 0.03
        self.volume_weight_factor = 0.8
        
        # Support and resistance levels will be calculated dynamically
        self.support_levels = []
        self.resistance_levels = []
        
        # Momentum and volume thresholds
        self.momentum_threshold = 0.8
        self.volume_threshold = 1.1
        
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