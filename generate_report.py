#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
分析結果からレポート用のMarkdownと画像を生成
"""

import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

import pandas as pd
import MetaTrader5 as mt5
from datetime import timedelta
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # GUI不要のバックエンド
import numpy as np
import warnings
import os
warnings.filterwarnings('ignore')

# =============================================================================
# 設定
# =============================================================================

TARGET_SYMBOLS = [
    'EURUSD', 'USDJPY', 'GBPUSD', 'USDCHF', 'AUDUSD', 'USDCAD', 'NZDUSD',
    'EURJPY', 'GBPJPY', 'AUDJPY', 'NZDJPY', 'CADJPY', 'CHFJPY',
    'EURGBP', 'EURAUD', 'EURNZD', 'EURCAD', 'EURCHF',
    'GBPAUD', 'GBPNZD', 'GBPCAD', 'GBPCHF',
    'AUDNZD', 'AUDCAD', 'AUDCHF',
    'NZDCAD', 'NZDCHF', 'CADCHF',
    'XAUUSD',
]

DEFAULT_SPREAD_MULTIPLIER = 5.0
DATA_COUNT = 10000
MIN_TRADES = 20
TIMEZONE_HOURS = 9
WEEKDAY_NAMES = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']

OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))
IMG_DIR = os.path.join(OUTPUT_DIR, 'images')
os.makedirs(IMG_DIR, exist_ok=True)

plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 150

# =============================================================================
# MT5接続
# =============================================================================

print("MT5に接続中...")
if not mt5.initialize():
    print("MT5への接続失敗")
    quit()
print("MT5接続成功！")

# =============================================================================
# 関数定義
# =============================================================================

def detect_actual_symbols(target_symbols):
    actual_symbols = []
    all_symbols = mt5.symbols_get()
    if not all_symbols:
        return []
    for target in target_symbols:
        found = False
        for symbol in all_symbols:
            if symbol.name.upper() == target.upper():
                actual_symbols.append(symbol.name)
                found = True
                break
        if not found:
            for symbol in all_symbols:
                if target.upper() in symbol.name.upper():
                    actual_symbols.append(symbol.name)
                    found = True
                    break
        if not found and target == 'XAUUSD':
            for gold in ['GOLD', 'Gold', 'XAU']:
                for symbol in all_symbols:
                    if gold.upper() in symbol.name.upper():
                        actual_symbols.append(symbol.name)
                        found = True
                        break
                if found:
                    break
    return actual_symbols

def fetch_data(symbol, count=DATA_COUNT):
    symbol_info = mt5.symbol_info(symbol)
    if not symbol_info:
        return None
    if not symbol_info.visible:
        if not mt5.symbol_select(symbol, True):
            return None
    rates = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_M5, 0, count)
    if rates is None:
        return None
    df = pd.DataFrame(rates)
    df['time'] = pd.to_datetime(df['time'], unit='s') + timedelta(hours=TIMEZONE_HOURS)
    df['hour'] = df['time'].dt.hour
    df['minute'] = df['time'].dt.minute
    df['weekday'] = df['time'].dt.weekday
    df['date'] = df['time'].dt.date
    digits = symbol_info.digits
    if digits >= 3:
        pip_value = 10 ** (-(digits - 1))
    else:
        pip_value = 10 ** (-digits)
    if 'spread' not in df.columns:
        df['spread'] = 20
    if digits == 5 or digits == 3:
        df['spread_pips'] = df['spread'] / 10
    else:
        df['spread_pips'] = df['spread']
    df.attrs['symbol'] = symbol
    df.attrs['pip_value'] = pip_value
    df.attrs['digits'] = digits
    df.attrs['spread_mean'] = df['spread_pips'].mean()
    return df

def backtest_time_range(df, entry_h, entry_m, exit_h, exit_m, direction='long'):
    max_spread = df.attrs['spread_mean'] * DEFAULT_SPREAD_MULTIPLIER
    entry_mask = (df['hour'] == entry_h) & (df['minute'] == entry_m) & (df['spread_pips'] <= max_spread)
    exit_mask = (df['hour'] == exit_h) & (df['minute'] == exit_m)
    entries = df[entry_mask][['date', 'open', 'spread_pips', 'weekday']].copy()
    exits = df[exit_mask][['date', 'close', 'spread_pips']].copy()
    trades = pd.merge(entries, exits, on='date', suffixes=('_entry', '_exit'))
    if len(trades) == 0:
        return None
    pip_value = df.attrs['pip_value']
    if direction == 'long':
        trades['entry_price'] = trades['open'] + trades['spread_pips_entry'] * pip_value
        trades['exit_price'] = trades['close']
        trades['profit'] = (trades['exit_price'] - trades['entry_price']) / trades['entry_price']
    else:
        trades['entry_price'] = trades['open']
        trades['exit_price'] = trades['close'] + trades['spread_pips_exit'] * pip_value
        trades['profit'] = (trades['entry_price'] - trades['exit_price']) / trades['entry_price']
    if len(trades) > 0:
        return {
            'count': len(trades),
            'win_rate': (trades['profit'] > 0).mean(),
            'avg_return': trades['profit'].mean(),
            'total_return': trades['profit'].sum(),
            'sharpe': trades['profit'].mean() / trades['profit'].std() if trades['profit'].std() > 0 else 0,
            'trades': trades
        }
    return None

def find_best_times(df, symbol):
    results = []
    for entry_h in range(0, 24):
        for entry_m in [0, 30]:
            for hold_h in range(3, 13):
                exit_h = (entry_h + hold_h) % 24
                exit_m = entry_m
                long_result = backtest_time_range(df, entry_h, entry_m, exit_h, exit_m, 'long')
                if long_result and long_result['count'] >= MIN_TRADES:
                    results.append({
                        'symbol': symbol, 'direction': 'long',
                        'entry': f"{entry_h:02d}:{entry_m:02d}",
                        'exit': f"{exit_h:02d}:{exit_m:02d}",
                        'hold_hours': hold_h,
                        'count': long_result['count'],
                        'win_rate': long_result['win_rate'],
                        'avg_return': long_result['avg_return'],
                        'sharpe': long_result['sharpe']
                    })
                short_result = backtest_time_range(df, entry_h, entry_m, exit_h, exit_m, 'short')
                if short_result and short_result['count'] >= MIN_TRADES:
                    results.append({
                        'symbol': symbol, 'direction': 'short',
                        'entry': f"{entry_h:02d}:{entry_m:02d}",
                        'exit': f"{exit_h:02d}:{exit_m:02d}",
                        'hold_hours': hold_h,
                        'count': short_result['count'],
                        'win_rate': short_result['win_rate'],
                        'avg_return': short_result['avg_return'],
                        'sharpe': short_result['sharpe']
                    })
    results_df = pd.DataFrame(results)
    if len(results_df) > 0:
        return results_df[results_df['avg_return'] > 0].sort_values('sharpe', ascending=False)
    return pd.DataFrame()

def save_detailed_charts(df, entry_h, entry_m, exit_h, exit_m, direction, symbol, rank):
    """詳細チャートを画像として保存"""
    result = backtest_time_range(df, entry_h, entry_m, exit_h, exit_m, direction)
    if not result:
        return None

    trades = result['trades']
    trades['cumulative'] = (1 + trades['profit']).cumprod() - 1

    weekday_stats = trades.groupby('weekday').agg({
        'profit': ['mean', 'count', lambda x: (x > 0).mean()]
    })
    weekday_stats.columns = ['avg_return', 'count', 'win_rate']

    entry_time = f"{entry_h:02d}:{entry_m:02d}"
    exit_time = f"{exit_h:02d}:{exit_m:02d}"

    fig, axes = plt.subplots(2, 2, figsize=(16, 11))
    fig.suptitle(f'#{rank} {symbol} {direction.upper()}: {entry_time} -> {exit_time}', fontsize=16, fontweight='bold')

    # 累積リターン
    ax = axes[0, 0]
    ax.plot(range(len(trades)), trades['cumulative'].values * 100, linewidth=2, color='#2196F3')
    ax.fill_between(range(len(trades)), 0, trades['cumulative'].values * 100, alpha=0.1, color='#2196F3')
    ax.set_title('Cumulative Return', fontsize=13)
    ax.set_ylabel('Return %')
    ax.set_xlabel('Trade #')
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0, color='r', linestyle='--', alpha=0.5)

    # リターン分布
    ax = axes[0, 1]
    profits = trades['profit'] * 100
    n, bins, patches = ax.hist(profits, bins=30, alpha=0.7, color='#4CAF50', edgecolor='white')
    ax.axvline(x=profits.mean(), color='r', linestyle='-', linewidth=2, label=f'Mean: {profits.mean():.4f}%')
    ax.set_title('Return Distribution', fontsize=13)
    ax.set_xlabel('Return %')
    ax.set_ylabel('Frequency')
    ax.legend()

    # 曜日別リターン
    ax = axes[1, 0]
    weekday_returns = weekday_stats['avg_return'] * 100
    weekday_names = [WEEKDAY_NAMES[i] for i in weekday_returns.index]
    colors = ['#4CAF50' if x > 0 else '#F44336' for x in weekday_returns.values]
    bars = ax.bar(weekday_names, weekday_returns.values, color=colors, alpha=0.8, edgecolor='white')
    ax.set_title('Average Return by Weekday', fontsize=13)
    ax.set_ylabel('Avg Return %')
    ax.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    for bar, val in zip(bars, weekday_returns.values):
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height(),
                f'{val:.4f}%', ha='center', va='bottom' if val >= 0 else 'top', fontsize=9)

    # サマリー
    ax = axes[1, 1]
    ax.axis('off')
    win_rate = (trades['profit'] > 0).mean()
    avg_return = trades['profit'].mean()
    total_return = trades['cumulative'].iloc[-1]
    sharpe = trades['profit'].mean() / trades['profit'].std() if trades['profit'].std() > 0 else 0
    max_dd = (trades['cumulative'] - trades['cumulative'].cummax()).min()

    summary_text = (
        f"Performance Summary\n"
        f"{'='*30}\n\n"
        f"Symbol:       {symbol}\n"
        f"Direction:    {direction.upper()}\n"
        f"Entry:        {entry_time}\n"
        f"Exit:         {exit_time}\n"
        f"Hold:         {(exit_h - entry_h) % 24}h\n\n"
        f"Trades:       {len(trades)}\n"
        f"Win Rate:     {win_rate:.1%}\n"
        f"Avg Return:   {avg_return:.4%}\n"
        f"Total Return: {total_return:.2%}\n"
        f"Sharpe Ratio: {sharpe:.3f}\n"
        f"Max Drawdown: {max_dd:.2%}"
    )
    ax.text(0.1, 0.9, summary_text, transform=ax.transAxes, fontsize=12,
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='#f0f0f0', alpha=0.8))

    plt.tight_layout()
    filename = f"top{rank}_{symbol}_{direction}_{entry_time.replace(':','')}_{exit_time.replace(':','')}.png"
    filepath = os.path.join(IMG_DIR, filename)
    plt.savefig(filepath, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  Saved: {filename}")
    return filename

def save_overview_chart(top_strategies):
    """TOP20のSharpeランキング横棒グラフ"""
    top20 = top_strategies.head(20).iloc[::-1]  # 逆順で下から上に

    fig, ax = plt.subplots(figsize=(14, 10))
    labels = [f"{r['symbol']} {r['direction'].upper()} {r['entry']}->{r['exit']}" for _, r in top20.iterrows()]
    sharpes = top20['sharpe'].values
    colors = ['#2196F3' if d == 'long' else '#F44336' for d in top20['direction'].values]

    bars = ax.barh(range(len(labels)), sharpes, color=colors, alpha=0.8, edgecolor='white')
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=10)
    ax.set_xlabel('Sharpe Ratio', fontsize=12)
    ax.set_title('TOP 20 Strategies by Sharpe Ratio (Blue=Long, Red=Short)', fontsize=14, fontweight='bold')
    ax.grid(True, axis='x', alpha=0.3)

    for bar, val in zip(bars, sharpes):
        ax.text(val + 0.005, bar.get_y() + bar.get_height()/2., f'{val:.3f}',
                ha='left', va='center', fontsize=9)

    plt.tight_layout()
    filepath = os.path.join(IMG_DIR, 'top20_sharpe_ranking.png')
    plt.savefig(filepath, bbox_inches='tight', facecolor='white')
    plt.close()
    print("Saved: top20_sharpe_ranking.png")

def save_heatmap(top_strategies):
    """通貨ペア x 方向の戦略数ヒートマップ"""
    pivot = top_strategies.head(100).groupby(['symbol', 'direction']).size().unstack(fill_value=0)
    fig, ax = plt.subplots(figsize=(8, 12))
    import seaborn as sns
    sns.heatmap(pivot, annot=True, fmt='d', cmap='YlOrRd', ax=ax, linewidths=0.5)
    ax.set_title('Strategy Count in TOP100 by Symbol & Direction', fontsize=14, fontweight='bold')
    ax.set_ylabel('Symbol')
    ax.set_xlabel('Direction')
    plt.tight_layout()
    filepath = os.path.join(IMG_DIR, 'strategy_heatmap.png')
    plt.savefig(filepath, bbox_inches='tight', facecolor='white')
    plt.close()
    print("Saved: strategy_heatmap.png")

def save_hold_hours_distribution(top_strategies):
    """保持時間分布"""
    top100 = top_strategies.head(100)
    fig, ax = plt.subplots(figsize=(10, 6))
    hours = top100['hold_hours'].value_counts().sort_index()
    ax.bar(hours.index, hours.values, color='#9C27B0', alpha=0.8, edgecolor='white')
    ax.set_xlabel('Hold Hours', fontsize=12)
    ax.set_ylabel('Count in TOP100', fontsize=12)
    ax.set_title('Distribution of Hold Hours in TOP100 Strategies', fontsize=14, fontweight='bold')
    ax.set_xticks(range(3, 13))
    ax.grid(True, axis='y', alpha=0.3)
    for x, y in zip(hours.index, hours.values):
        ax.text(x, y + 0.3, str(y), ha='center', fontsize=11, fontweight='bold')
    plt.tight_layout()
    filepath = os.path.join(IMG_DIR, 'hold_hours_distribution.png')
    plt.savefig(filepath, bbox_inches='tight', facecolor='white')
    plt.close()
    print("Saved: hold_hours_distribution.png")

def save_winrate_vs_sharpe(top_strategies):
    """勝率 vs Sharpe散布図"""
    top100 = top_strategies.head(100)
    fig, ax = plt.subplots(figsize=(10, 8))
    colors = ['#2196F3' if d == 'long' else '#F44336' for d in top100['direction'].values]
    sizes = top100['count'] * 3
    scatter = ax.scatter(top100['win_rate'] * 100, top100['sharpe'], c=colors, s=sizes, alpha=0.6, edgecolors='white')
    ax.set_xlabel('Win Rate %', fontsize=12)
    ax.set_ylabel('Sharpe Ratio', fontsize=12)
    ax.set_title('Win Rate vs Sharpe Ratio (Size=Trade Count, Blue=Long, Red=Short)', fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3)

    # TOP5にラベル付け
    for i in range(min(5, len(top100))):
        row = top100.iloc[i]
        ax.annotate(f"{row['symbol']}", (row['win_rate']*100, row['sharpe']),
                    fontsize=8, ha='left', va='bottom')

    plt.tight_layout()
    filepath = os.path.join(IMG_DIR, 'winrate_vs_sharpe.png')
    plt.savefig(filepath, bbox_inches='tight', facecolor='white')
    plt.close()
    print("Saved: winrate_vs_sharpe.png")

def save_entry_time_heatmap(top_strategies):
    """エントリー時間帯の分布"""
    top100 = top_strategies.head(100)
    top100_copy = top100.copy()
    top100_copy['entry_hour'] = top100_copy['entry'].apply(lambda x: int(x.split(':')[0]))
    fig, ax = plt.subplots(figsize=(12, 6))
    counts = top100_copy['entry_hour'].value_counts().sort_index()
    all_hours = pd.Series(0, index=range(24))
    all_hours.update(counts)
    ax.bar(all_hours.index, all_hours.values, color='#FF9800', alpha=0.8, edgecolor='white')
    ax.set_xlabel('Entry Hour (JST)', fontsize=12)
    ax.set_ylabel('Count in TOP100', fontsize=12)
    ax.set_title('Entry Time Distribution in TOP100 Strategies', fontsize=14, fontweight='bold')
    ax.set_xticks(range(24))
    ax.set_xticklabels([f'{h:02d}' for h in range(24)])
    ax.grid(True, axis='y', alpha=0.3)
    plt.tight_layout()
    filepath = os.path.join(IMG_DIR, 'entry_time_distribution.png')
    plt.savefig(filepath, bbox_inches='tight', facecolor='white')
    plt.close()
    print("Saved: entry_time_distribution.png")

# =============================================================================
# メイン処理
# =============================================================================

def main():
    symbols = detect_actual_symbols(TARGET_SYMBOLS)
    if not symbols:
        print("通貨ペアが見つかりません")
        mt5.shutdown()
        return

    print(f"\n分析対象: {len(symbols)}通貨ペア")

    all_results = []
    symbol_data = {}

    for symbol in symbols:
        print(f"  {symbol} ...", end=" ")
        df = fetch_data(symbol)
        if df is None:
            print("NG")
            continue
        profitable = find_best_times(df, symbol)
        if len(profitable) > 0:
            print(f"{len(profitable)} strategies found")
            all_results.append({'symbol': symbol, 'strategies': profitable})
            symbol_data[symbol] = df
        else:
            print("0 strategies")

    if not all_results:
        print("戦略なし")
        mt5.shutdown()
        return

    # 全戦略を統合
    all_strategies = []
    for result in all_results:
        all_strategies.extend(result['strategies'].to_dict('records'))
    all_strategies_df = pd.DataFrame(all_strategies)
    top_strategies = all_strategies_df.sort_values('sharpe', ascending=False).reset_index(drop=True)

    print(f"\n合計 {len(top_strategies)} 戦略を発見")

    # --- 画像生成 ---
    print("\n=== 画像生成 ===")

    # 1. 概要チャート
    save_overview_chart(top_strategies)
    save_heatmap(top_strategies)
    save_hold_hours_distribution(top_strategies)
    save_winrate_vs_sharpe(top_strategies)
    save_entry_time_heatmap(top_strategies)

    # 2. TOP10の詳細チャート
    detail_files = []
    for i in range(min(10, len(top_strategies))):
        strategy = top_strategies.iloc[i]
        df = symbol_data.get(strategy['symbol'])
        if df is None:
            continue
        entry_h, entry_m = map(int, strategy['entry'].split(':'))
        exit_h, exit_m = map(int, strategy['exit'].split(':'))
        filename = save_detailed_charts(df, entry_h, entry_m, exit_h, exit_m,
                                         strategy['direction'], strategy['symbol'], i + 1)
        if filename:
            detail_files.append((i + 1, strategy, filename))

    # --- Markdown生成 ---
    print("\n=== Markdown生成 ===")

    top100 = top_strategies.head(100)

    md = []
    md.append("# FX Anomaly Auto-Search Report")
    md.append("")
    md.append(f"> Generated from MT5 data | Hold: 3-12 hours | Min Trades: {MIN_TRADES} | Spread Filter: {DEFAULT_SPREAD_MULTIPLIER}x avg")
    md.append("")

    md.append("## Overview")
    md.append("")
    md.append(f"- **Analyzed Pairs**: {len(symbols)}")
    md.append(f"- **Total Profitable Strategies**: {len(top_strategies)}")
    md.append(f"- **Hold Time Range**: 3 - 12 hours")
    md.append(f"- **Data**: {DATA_COUNT} x 5min candles per pair (~35 days)")
    md.append(f"- **Timezone**: GMT+{TIMEZONE_HOURS} (JST)")
    md.append("")

    # 概要チャート
    md.append("## Strategy Overview Charts")
    md.append("")
    md.append("### TOP 20 Sharpe Ranking")
    md.append("![TOP20 Sharpe Ranking](images/top20_sharpe_ranking.png)")
    md.append("")
    md.append("### Win Rate vs Sharpe Ratio")
    md.append("![Win Rate vs Sharpe](images/winrate_vs_sharpe.png)")
    md.append("")
    md.append("### Hold Hours Distribution")
    md.append("![Hold Hours](images/hold_hours_distribution.png)")
    md.append("")
    md.append("### Entry Time Distribution")
    md.append("![Entry Time](images/entry_time_distribution.png)")
    md.append("")
    md.append("### Strategy Heatmap (Symbol x Direction)")
    md.append("![Heatmap](images/strategy_heatmap.png)")
    md.append("")

    # TOP10詳細
    md.append("## TOP 10 Strategy Details")
    md.append("")
    for rank, strategy, filename in detail_files:
        md.append(f"### #{rank} {strategy['symbol']} {strategy['direction'].upper()}: {strategy['entry']} -> {strategy['exit']} ({strategy['hold_hours']}h)")
        md.append("")
        md.append(f"| Metric | Value |")
        md.append(f"|--------|-------|")
        md.append(f"| Sharpe Ratio | {strategy['sharpe']:.3f} |")
        md.append(f"| Win Rate | {strategy['win_rate']:.1%} |")
        md.append(f"| Avg Return | {strategy['avg_return']:.4%} |")
        md.append(f"| Trades | {strategy['count']} |")
        md.append(f"| Hold | {strategy['hold_hours']}h |")
        md.append("")
        md.append(f"![#{rank} Detail](images/{filename})")
        md.append("")

    # TOP100テーブル
    md.append("## TOP 100 Strategies Ranking")
    md.append("")
    md.append("| Rank | Symbol | Dir | Entry | Exit | Hold | Trades | Win% | AvgReturn% | Sharpe |")
    md.append("|------|--------|-----|-------|------|------|--------|------|------------|--------|")
    for i, (_, row) in enumerate(top100.iterrows()):
        md.append(f"| {i+1} | {row['symbol']} | {row['direction'].upper()} | {row['entry']} | {row['exit']} | {row['hold_hours']}h | {row['count']} | {row['win_rate']:.1%} | {row['avg_return']:.4%} | {row['sharpe']:.3f} |")
    md.append("")

    # 通貨ペア別サマリー
    md.append("## Summary by Symbol")
    md.append("")
    md.append("| Symbol | Strategies in TOP100 | Best Sharpe | Best Direction |")
    md.append("|--------|---------------------|-------------|----------------|")
    for sym in top100['symbol'].unique():
        sym_data = top100[top100['symbol'] == sym]
        best = sym_data.iloc[0]
        md.append(f"| {sym} | {len(sym_data)} | {best['sharpe']:.3f} | {best['direction'].upper()} |")
    md.append("")

    md.append("---")
    md.append("*This report was auto-generated by the FX Anomaly Search System.*")

    # Markdown書き出し
    md_path = os.path.join(OUTPUT_DIR, 'REPORT.md')
    with open(md_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(md))
    print(f"Saved: REPORT.md")

    mt5.shutdown()
    print("\nDone!")

if __name__ == "__main__":
    main()
