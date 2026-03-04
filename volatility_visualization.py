#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
通貨ペア ボラティリティ & 買い圧力 可視化ツール
"""

import pandas as pd
import MetaTrader5 as mt5
from datetime import timedelta
import seaborn as sns
import matplotlib.pyplot as plt

# MT5への接続
mt5.initialize()

def fetch_and_prepare(symbol, timeframe, count, timezone_hours, ratio_calculation, column_name):
    # データの取得
    rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, count)

    # DataFrameに変換
    df = pd.DataFrame(rates)

    # 日時をPandasの日時形式に変換し、GMT+9に変換
    df['time'] = pd.to_datetime(df['time'], unit='s') + timedelta(hours=timezone_hours)

    df[column_name] = ratio_calculation(df)

    return df

def high_low_ratio(df):
    return (df['high'] - df['low']) / df['low']

def close_open_ratio(df):
    return (df['close'] - df['open']) / df['open']

def plot_avg_return(df, column_name):
    # インデックスを日付に変換
    df.set_index('time', inplace=True)

    # 5分ごとの平均リターンを計算
    df['5min_avg_return'] = df[column_name].rolling(window=5).mean()

    # 時刻をグループ化して平均を計算
    grouped_df = df.groupby(df.index.time).mean()

    # 時刻を文字列に変換
    grouped_df.index = grouped_df.index.astype(str)

    # グラフの作成
    plt.figure(figsize=(50, 10))
    sns.barplot(x=grouped_df.index, y=grouped_df['5min_avg_return'])
    plt.xlabel('Time')
    plt.ylabel('5min Average Return')
    plt.title('5-minute Average Return (Grouped by Time)')

    # X軸のラベルを斜めに表示
    plt.xticks(rotation=90)

    plt.show()

# 読み込む通貨ペアと時間枠、データの件数を指定
symbol = 'USDJPYm'
timeframe = mt5.TIMEFRAME_M5
count = 100000
timezone_hours = 9

df = fetch_and_prepare(symbol, timeframe, count, timezone_hours, high_low_ratio, 'high_low_ratio')
plot_avg_return(df, 'high_low_ratio')

df = fetch_and_prepare(symbol, timeframe, count, timezone_hours, close_open_ratio, 'close_open_ratio')
plot_avg_return(df, 'close_open_ratio')
