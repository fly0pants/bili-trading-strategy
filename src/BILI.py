from clr import AddReference
AddReference("System")
AddReference("QuantConnect.Algorithm")
AddReference("QuantConnect.Common")
AddReference("QuantConnect.Indicators")

from System import *
from QuantConnect import *
from QuantConnect.Algorithm import *
from QuantConnect.Data import *
from QuantConnect.Indicators import *
from QuantConnect.Securities import *
from QuantConnect.Orders import *
from QuantConnect.Brokerages import BrokerageName

import numpy as np
from datetime import datetime, timedelta

class BilibiliTradingAlgorithm(QCAlgorithm):
    def Initialize(self):
        """初始化算法设置"""
        self.Debug("==== 开始初始化算法 ====")
        
        # 设置回测期间 - 使用2022年的历史数据
        self.SetStartDate(2025, 1, 1)  # 设置开始日期为2025年2月21日
        self.SetEndDate(2025, 2, 20)    # 设置结束日期为2025年3月21日

        self.Debug(f"设置回测时段: {self.StartDate.strftime('%Y-%m-%d')} 到 {self.EndDate.strftime('%Y-%m-%d')}")
        
        # 设置初始资金
        self.SetCash(10000)
        
        # 添加BILI股票
        self.bili = self.AddEquity("BILI", Resolution.Daily)
        self.bili.SetDataNormalizationMode(DataNormalizationMode.Raw)
        self.Debug(f"添加BILI股票，Resolution: {Resolution.Daily}")
        
        # 尝试获取历史数据进行诊断
        self.Debug("尝试获取BILI历史数据...")
        try:
            history = self.History(["BILI"], 0, Resolution.Daily)
            
            if history.empty:
                self.Debug("警告: 未找到任何BILI的历史数据!")
            else:
                self.Debug(f"成功获取BILI历史数据: {len(history)} 条记录")
                
                # 提取开盘价和收盘价
                open_prices = history['open']
                close_prices = history['close']
                
                # 输出数据以验证
                self.Debug(f"获取到最近30个交易日的数据：{len(history)} 条记录")
                self.Debug(f"第一天开盘价：{open_prices.iloc[0]}, 收盘价：{close_prices.iloc[0]}")
                self.Debug(f"最后一天开盘价：{open_prices.iloc[-1]}, 收盘价：{close_prices.iloc[-1]}")
                
                # 如果需要DataFrame格式的数据
                price_data = history[['open', 'close']]
                self.Debug(f"价格数据头部：\n{price_data.head()}")
                
                # 安全地检查和打印日期信息
                try:
                    # 检查索引结构并打印更详细的信息
                    self.Debug(f"历史数据索引类型: {type(history.index)}")
                    self.Debug(f"历史数据索引第一个元素类型: {type(history.index[0])}")
                    
                    # 打印更详细的索引信息
                    self.Debug(f"历史数据索引前几个元素: {list(history.index[:3])}")
                    
                    # 获取日期信息 (假设是复合索引的第二个元素是时间)
                    if hasattr(history.index, 'levels') and len(history.index.levels) > 1:
                        # 多级索引
                        first_date = history.index.levels[1][0]
                        last_date = history.index.levels[1][-1]
                        self.Debug(f"数据时间范围: {first_date} 到 {last_date}")
                    elif isinstance(history.index[0], tuple) and len(history.index[0]) > 1:
                        # 元组索引
                        dates = [idx[1] for idx in history.index]
                        self.Debug(f"数据时间范围: {min(dates)} 到 {max(dates)}")
                    else:
                        # 尝试直接获取时间戳
                        self.Debug(f"索引值范围: {history.index.min()} 到 {history.index.max()}")
                except Exception as e:
                    self.Debug(f"打印索引信息时出错: {str(e)}")
                
                # 记录历史数据的第一条和最后一条，以检查数据质量
                try:
                    first_record = history.iloc[0]
                    last_record = history.iloc[-1]
                    
                    self.Debug(f"第一条数据: 收盘价={first_record['close']}, 成交量={first_record['volume']}")
                    self.Debug(f"最后一条数据: 收盘价={last_record['close']}, 成交量={last_record['volume']}")
                    
                    # 保存最近的收盘价用于后续计算
                    self.previous_closes = list(history['close'].tail(10))
                    self.Debug(f"初始化previous_closes: {len(self.previous_closes)} 条记录")
                except Exception as e:
                    self.Debug(f"打印数据记录时出错: {str(e)}")
        except Exception as e:
            self.Debug(f"获取历史数据时出错: {str(e)}")
        
        # 初始化交易参数
        self.position_size = 0.3       # 每笔交易使用30%资金
        self.stop_loss_pct = 0.03      # 3%止损
        self.take_profit_pct = 0.05    # 5%止盈
        
        # 设置特定的支撑位和阻力位 - 根据2023年BILI的价格区间调整
        # 设置特定的支撑位和阻力位
        self.support_levels = [20.55, 20.12, 19.81]  # 报告中提到的支撑位
        self.resistance_levels = [21.30, 21.61, 22.04]  # 报告中提到的阻力位
    
        self.Debug(f"配置支撑位: {self.support_levels}")
        self.Debug(f"配置阻力位: {self.resistance_levels}")
        
        # 动量和成交量参数
        self.momentum_threshold = 1.0   # 1%价格变动视为动量
        self.volume_threshold = 1.2     # 成交量超过5日均线的20%
        
        # 注册技术指标
        self.Debug("注册技术指标...")
        # 价格移动平均线
        self.sma5 = self.SMA("BILI", 5, Resolution.Daily)
        self.sma20 = self.SMA("BILI", 20, Resolution.Daily)
        
        # 成交量指标
        self.volume_sma5 = self.SMA("BILI", 5, Resolution.Daily, Field.Volume)
        
        # 初始化状态变量
        if not hasattr(self, 'previous_closes'):
            self.previous_closes = []
        self.stop_loss = 0
        self.take_profit = 0
        self.entry_price = 0
        
        # 存储所有交易信息
        self.all_trades = []
        
        # 添加计数器跟踪OnData调用次数
        self.data_count = 0
        
        # 设置基准
        self.SetBenchmark("BILI")
        
        # 放宽交易信号阈值，便于测试
        self.signal_threshold = 1  # 只需要1个信号即可触发交易
        
        # 每日记录投资组合状态
        self.Schedule.On(self.DateRules.EveryDay("BILI"), 
                         self.TimeRules.AfterMarketOpen("BILI", 0), 
                         self.LogPortfolioStatus)
        
        # Enable precise end time to properly handle daily bar timing
        self.settings.daily_precise_end_time = True
        
        # 尝试获取BILI特定日期范围的历史数据
        self.Debug("计算2025.1.1至2025.2.20期间的支撑位和阻力位...")
        try:
            # 获取指定日期范围的历史数据
            start_date = datetime(2025, 1, 1)
            end_date = datetime(2025, 2, 20)
            history_period = self.History(["BILI"], start_date, end_date, Resolution.Daily)
            
            if history_period.empty:
                self.Debug("警告: 未找到指定时间范围内的BILI历史数据!")
            else:
                self.Debug(f"成功获取指定范围BILI历史数据: {len(history_period)} 条记录")
                
                # 提取高低价数据
                highs = history_period['high']
                lows = history_period['low']
                closes = history_period['close']
                
                # 计算支撑位和阻力位
                # 方法1: 使用期间内的最低点作为支撑位
                sorted_lows = sorted(lows.values)
                support_candidates = sorted_lows[:int(len(sorted_lows) * 0.3)]  # 取最低的30%
                
                # 方法2: 使用期间内的最高点作为阻力位
                sorted_highs = sorted(highs.values, reverse=True)
                resistance_candidates = sorted_highs[:int(len(sorted_highs) * 0.3)]  # 取最高的30%
                
                # 筛选有意义的支撑位和阻力位(避免太接近的价格)
                filtered_supports = []
                for s in support_candidates:
                    # 检查是否与已有支撑位相距太近
                    if not any(abs(s - existing) / existing < 0.01 for existing in filtered_supports):
                        filtered_supports.append(s)
                    if len(filtered_supports) >= 3:  # 最多保留3个支撑位
                        break
                
                filtered_resistances = []
                for r in resistance_candidates:
                    # 检查是否与已有阻力位相距太近
                    if not any(abs(r - existing) / existing < 0.01 for existing in filtered_resistances):
                        filtered_resistances.append(r)
                    if len(filtered_resistances) >= 3:  # 最多保留3个阻力位
                        break
                
                # 更新支撑位和阻力位
                self.support_levels = [round(s, 2) for s in filtered_supports]
                self.resistance_levels = [round(r, 2) for r in filtered_resistances]
                
                self.Debug(f"计算得到的支撑位: {self.support_levels}")
                self.Debug(f"计算得到的阻力位: {self.resistance_levels}")
                
                # 计算回测期间的价格范围
                price_min = min(lows.values)
                price_max = max(highs.values)
                price_last = closes.values[-1]
                self.Debug(f"价格范围: 最低 ${price_min:.2f}, 最高 ${price_max:.2f}, 最新 ${price_last:.2f}")
        except Exception as e:
            self.Debug(f"计算支撑位和阻力位时出错: {str(e)}")
        
        self.Debug("==== 算法初始化完成 ====")

    def OnData(self, data):
        """接收新数据时的主处理函数"""
        self.data_count += 1
        self.Debug(f"\n==== OnData 调用 #{self.data_count} - 日期: {self.Time.strftime('%Y-%m-%d')} ====")
        
        if not data.ContainsKey("BILI"):
            self.Debug("未收到BILI数据")
            return
            
        # 记录接收到的数据
        bili_bar = data["BILI"]
        current_price = bili_bar.Close
        current_volume = bili_bar.Volume
        
        self.Debug(f"接收到BILI数据: 价格=${current_price}, 成交量={current_volume}")
        
        # 检查指标就绪状态
        self.Debug("检查指标就绪状态:")
        self.Debug(f"  SMA5 就绪: {self.sma5.IsReady}, 当前值: {self.sma5.Current.Value if self.sma5.IsReady else 'N/A'}")
        self.Debug(f"  SMA20 就绪: {self.sma20.IsReady}, 当前值: {self.sma20.Current.Value if self.sma20.IsReady else 'N/A'}")
        self.Debug(f"  Volume SMA5 就绪: {self.volume_sma5.IsReady}, 当前值: {self.volume_sma5.Current.Value if self.volume_sma5.IsReady else 'N/A'}")
        
        # 等待所有指标就绪
        if not self.sma5.IsReady:
            self.Debug("等待SMA5就绪 - 需要至少5个数据点")
            return
            
        if not self.sma20.IsReady:
            self.Debug("等待SMA20就绪 - 需要至少20个数据点")
            return
            
        if not self.volume_sma5.IsReady:
            self.Debug("等待Volume SMA5就绪 - 需要至少5个数据点")
            return
        
        self.Debug("所有指标已就绪，继续执行交易逻辑")
            
        # 存储收盘价用于计算动量
        self.previous_closes.append(current_price)
        if len(self.previous_closes) > 10:
            self.previous_closes.pop(0)
        
        self.Debug(f"历史收盘价数据: {len(self.previous_closes)} 条记录")
            
        # 确保我们有足够的数据来生成信号
        if len(self.previous_closes) < 4:
            self.Debug(f"历史价格数据不足，需要至少4条记录，当前: {len(self.previous_closes)}")
            return
            
        # 计算交易指标
        daily_change_pct = (current_price / self.previous_closes[-2] - 1) * 100
        momentum = (current_price / self.previous_closes[-4] - 1) * 100
        volume_ratio = current_volume / self.volume_sma5.Current.Value
        
        self.Debug(f"计算交易指标:")
        self.Debug(f"  日涨跌幅: {daily_change_pct:.2f}%")
        self.Debug(f"  三日动量: {momentum:.2f}%")
        self.Debug(f"  成交量比率: {volume_ratio:.2f}x")
        
        # 检查是否接近支撑位和阻力位
        near_support = False
        closest_support = 0
        near_resistance = False
        closest_resistance = 0
        
        self.Debug("检查支撑位和阻力位:")
        for level in self.support_levels:
            price_diff_pct = abs(current_price - level) / level * 100
            self.Debug(f"  支撑位 ${level}: 当前价格偏差 {price_diff_pct:.2f}%")
            if price_diff_pct <= 2:  # 价格在支撑位2%范围内
                near_support = True
                closest_support = level
                self.Debug(f"  价格 ${current_price} 接近支撑位 ${level}")
                break
                
        for level in self.resistance_levels:
            price_diff_pct = abs(current_price - level) / level * 100
            self.Debug(f"  阻力位 ${level}: 当前价格偏差 {price_diff_pct:.2f}%")
            if price_diff_pct <= 2:  # 价格在阻力位2%范围内
                near_resistance = True
                closest_resistance = level
                self.Debug(f"  价格 ${current_price} 接近阻力位 ${level}")
                break
                
        # 生成交易信号
        signal = 0
        signal_reasons = []
        
        # 1. 支撑位和阻力位策略
        if near_support:
            signal += 1
            signal_reasons.append(f"接近支撑位 {closest_support}")
            
        if near_resistance:
            signal -= 1
            signal_reasons.append(f"接近阻力位 {closest_resistance}")
            
        # 2. 动量策略(配合成交量确认)
        if momentum > self.momentum_threshold and volume_ratio > self.volume_threshold:
            signal += 1
            signal_reasons.append(f"正动量 {momentum:.2f}% 配合放量 {volume_ratio:.2f}x")
            
        if momentum < -self.momentum_threshold and volume_ratio > self.volume_threshold:
            signal -= 1
            signal_reasons.append(f"负动量 {momentum:.2f}% 配合放量 {volume_ratio:.2f}x")
            
        # 3. 均线策略
        # 金叉信号 (5日均线上穿20日均线)
        if self.sma5.Current.Value > self.sma20.Current.Value and self.sma5.Previous.Value <= self.sma20.Previous.Value:
            signal += 1
            signal_reasons.append("均线金叉")
            
        # 死叉信号 (5日均线下穿20日均线)
        if self.sma5.Current.Value < self.sma20.Current.Value and self.sma5.Previous.Value >= self.sma20.Previous.Value:
            signal -= 1
            signal_reasons.append("均线死叉")
        
        # 信号强度原始值
        raw_signal = signal
                
        # 信号归一化为 -1, 0, 1 (使用可配置的阈值)
        if signal >= self.signal_threshold:  # 降低为只需要1个信号
            signal = 1
        elif signal <= -self.signal_threshold:  # 降低为只需要1个信号
            signal = -1
        else:
            signal = 0
        
        self.Debug(f"交易信号: 原始强度={raw_signal}, 归一化={signal}, 原因: {signal_reasons}")
            
        # 检查现有仓位
        existing_position = self.Portfolio["BILI"].Quantity
        self.Debug(f"当前BILI持仓: {existing_position} 股")
        
        # 根据信号执行交易
        if existing_position > 0:
            self.Debug(f"当前有多头仓位, 入场价: ${self.entry_price}, 止损: ${self.stop_loss}, 止盈: ${self.take_profit}")
            
            # 检查是否触及止损
            if current_price <= self.stop_loss:
                self.Debug(f"触发止损: 当前价格 ${current_price} <= 止损价 ${self.stop_loss}")
                self.Liquidate("BILI")
                loss_pct = (current_price / self.entry_price - 1) * 100
                
                self.Debug(f"卖出 (止损) 价格 ${current_price} - 触及止损位 ${self.stop_loss}")
                self.Debug(f"亏损: ${(current_price - self.entry_price) * existing_position} ({loss_pct:.2f}%)")
                
                # 记录交易
                self.all_trades.append({
                    'date': self.Time.strftime("%Y-%m-%d"),
                    'action': '卖出 (止损)',
                    'price': current_price,
                    'shares': existing_position,
                    'pnl_pct': loss_pct,
                    'reason': '触及止损位'
                })
                
                self.stop_loss = 0
                self.take_profit = 0
                self.entry_price = 0
                return
                
            # 检查是否触及止盈
            if current_price >= self.take_profit:
                self.Debug(f"触发止盈: 当前价格 ${current_price} >= 止盈价 ${self.take_profit}")
                self.Liquidate("BILI")
                profit_pct = (current_price / self.entry_price - 1) * 100
                
                self.Debug(f"卖出 (止盈) 价格 ${current_price} - 触及止盈位 ${self.take_profit}")
                self.Debug(f"盈利: ${(current_price - self.entry_price) * existing_position} ({profit_pct:.2f}%)")
                
                # 记录交易
                self.all_trades.append({
                    'date': self.Time.strftime("%Y-%m-%d"),
                    'action': '卖出 (止盈)',
                    'price': current_price,
                    'shares': existing_position,
                    'pnl_pct': profit_pct,
                    'reason': '触及止盈位'
                })
                
                self.stop_loss = 0
                self.take_profit = 0
                self.entry_price = 0
                return
                
            # 检查卖出信号
            if signal == -1:
                self.Debug(f"触发卖出信号: {signal_reasons}")
                self.Liquidate("BILI")
                profit_pct = (current_price / self.entry_price - 1) * 100
                reasons = ", ".join(signal_reasons)
                
                self.Debug(f"卖出 (信号) 价格 ${current_price} - 原因: {reasons}")
                self.Debug(f"盈亏: ${(current_price - self.entry_price) * existing_position} ({profit_pct:.2f}%)")
                
                # 记录交易
                self.all_trades.append({
                    'date': self.Time.strftime("%Y-%m-%d"),
                    'action': '卖出 (信号)',
                    'price': current_price,
                    'shares': existing_position,
                    'pnl_pct': profit_pct,
                    'reason': reasons
                })
                
                self.stop_loss = 0
                self.take_profit = 0
                self.entry_price = 0
                
        # 无仓位 - 检查买入信号
        elif existing_position == 0 and signal == 1:
            self.Debug(f"触发买入信号: {signal_reasons}")
            
            # 计算仓位大小
            invest_amount = self.Portfolio.Cash * self.position_size
            shares_to_buy = int(invest_amount / current_price)
            
            if shares_to_buy > 0:
                # 执行买入订单
                self.Debug(f"执行买入订单: {shares_to_buy} 股, 价格 ${current_price}")
                self.MarketOrder("BILI", shares_to_buy)
                
                # 设置止损和止盈水平
                self.entry_price = current_price
                self.stop_loss = current_price * (1 - self.stop_loss_pct)
                self.take_profit = current_price * (1 + self.take_profit_pct)
                
                reasons = ", ".join(signal_reasons)
                self.Debug(f"买入 价格 ${current_price} - {shares_to_buy} 股 - 原因: {reasons}")
                self.Debug(f"止损: ${self.stop_loss}, 止盈: ${self.take_profit}")
                
                # 记录交易
                self.all_trades.append({
                    'date': self.Time.strftime("%Y-%m-%d"),
                    'action': '买入',
                    'price': current_price,
                    'shares': shares_to_buy,
                    'stop_loss': self.stop_loss,
                    'take_profit': self.take_profit,
                    'reason': reasons
                })
        else:
            if existing_position == 0:
                self.Debug(f"无仓位且无交易信号: signal={signal}")
            else:
                self.Debug(f"持有仓位但无卖出信号: signal={signal}")
    
    def OnOrderEvent(self, orderEvent):
        """订单事件处理函数"""
        self.Debug(f"订单事件: {orderEvent.Status} - {orderEvent.OrderId}")
        if orderEvent.Status == OrderStatus.Filled:
            try:
                order = self.Transactions.GetOrderById(orderEvent.OrderId)
                self.Debug(f"订单 {order.Type} 成交: {orderEvent}")
            except Exception as e:
                self.Debug(f"处理订单事件时出错: {str(e)}")
        
    def LogPortfolioStatus(self):
        """记录每日投资组合状态"""
        self.Debug(f"日期: {self.Time.strftime('%Y-%m-%d')} - 投资组合价值: ${self.Portfolio.TotalPortfolioValue}")
        
        if self.Portfolio.Invested:
            position = self.Portfolio["BILI"]
            unrealized_profit = position.UnrealizedProfit
            unrealized_profit_pct = (position.Price / position.AveragePrice - 1) * 100
            
            self.Debug(f"BILI仓位: {position.Quantity} 股，均价: ${position.AveragePrice}")
            self.Debug(f"未实现盈亏: ${unrealized_profit} ({unrealized_profit_pct:.2f}%)")
            self.Debug(f"止损: ${self.stop_loss}, 止盈: ${self.take_profit}")
    
    def OnEndOfAlgorithm(self):
        """算法结束时的处理"""
        self.Debug("\n==== 算法结束 ====")
        self.Debug(f"OnData 总调用次数: {self.data_count}")
        
        # 计算和显示最终结果
        final_equity = self.Portfolio.TotalPortfolioValue
        initial_capital = 10000  # 初始资金
        total_return = (final_equity / initial_capital - 1) * 100
        
        self.Debug(f"最终投资组合价值: ${final_equity}")
        self.Debug(f"初始资金: ${initial_capital}")
        self.Debug(f"总收益: {total_return:.2f}%")
        
        # 交易统计
        num_trades = len([t for t in self.all_trades if t['action'].startswith('卖出')])
        self.Debug(f"总交易次数: {len(self.all_trades)}, 完成交易次数: {num_trades}")
        
        if num_trades > 0:
            winning_trades = [t for t in self.all_trades if t['action'].startswith('卖出') and t.get('pnl_pct', 0) > 0]
            losing_trades = [t for t in self.all_trades if t['action'].startswith('卖出') and t.get('pnl_pct', 0) <= 0]
            
            win_rate = len(winning_trades) / num_trades * 100
            
            avg_win = sum([t.get('pnl_pct', 0) for t in winning_trades]) / len(winning_trades) if winning_trades else 0
            avg_loss = sum([t.get('pnl_pct', 0) for t in losing_trades]) / len(losing_trades) if losing_trades else 0
            
            self.Debug(f"\n===== 哔哩哔哩交易策略回测结果 =====")
            self.Debug(f"初始资金: ${initial_capital}")
            self.Debug(f"最终价值: ${final_equity}")
            self.Debug(f"总收益: {total_return:.2f}%")
            self.Debug(f"交易次数: {num_trades}")
            self.Debug(f"胜率: {win_rate:.2f}%")
            self.Debug(f"平均盈利: {avg_win:.2f}%")
            self.Debug(f"平均亏损: {avg_loss:.2f}%")
            
            # 输出所有交易明细
            self.Debug("\n===== 交易明细 =====")
            for i, trade in enumerate(self.all_trades):
                self.Debug(f"交易 {i+1}: {trade['date']} - {trade['action']} 价格 ${trade['price']}, {trade['shares']} 股")
                if 'reason' in trade:
                    self.Debug(f"  原因: {trade['reason']}")
                if 'pnl_pct' in trade:
                    self.Debug(f"  盈亏: {trade['pnl_pct']:.2f}%")
        else:
            self.Debug("回测期间没有执行交易。")
            self.Debug("\n可能原因:")
            self.Debug("1) 数据完全缺失 - 检查回测期间是否有BILI数据")
            self.Debug("2) 数据量不足 - SMA20需要至少20个数据点才能就绪")
            self.Debug("3) 交易信号未触发 - 价格可能未接近支撑位/阻力位")
            self.Debug("4) 信号强度不足 - 调整信号阈值或支撑位/阻力位范围")