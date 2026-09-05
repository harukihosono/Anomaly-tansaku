#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
FXマルチ通貨ペア時間帯別トレード分析システム
保持時間: 3〜12時間、ランキングTOP100
"""

import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

import pandas as pd
import MetaTrader5 as mt5
from datetime import timedelta
import matplotlib.pyplot as plt
import numpy as np
import time
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# 設定パラメータ（ここを編集してカスタマイズ）
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

FORCE_ENGLISH = True
DEFAULT_SPREAD_MULTIPLIER = 5.0
DATA_COUNT = 10000
MIN_TRADES = 20
TIMEZONE_HOURS = 9
SPECIFIC_ENTRY_TIME = None
SPECIFIC_EXIT_TIME = None

# =============================================================================
# グローバル変数とフォント設定
# =============================================================================

use_japanese = False
WEEKDAY_NAMES = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']

def setup_japanese_font():
    """日本語フォントを設定"""
    global use_japanese, WEEKDAY_NAMES

    try:
        import matplotlib.font_manager as fm

        if 'win' in sys.platform:
            fonts = ['MS Gothic', 'MS Mincho', 'Yu Gothic', 'Yu Mincho', 'Meiryo']
            for font in fonts:
                if font in [f.name for f in fm.fontManager.ttflist]:
                    plt.rcParams['font.family'] = font
                    print(f"フォント設定: {font}")
                    use_japanese = True
                    WEEKDAY_NAMES = ['月', '火', '水', '木', '金', '土', '日']
                    return True

        elif 'darwin' in sys.platform:
            fonts = ['Hiragino Sans', 'Hiragino Mincho Pro', 'Osaka']
            for font in fonts:
                if font in [f.name for f in fm.fontManager.ttflist]:
                    plt.rcParams['font.family'] = font
                    print(f"フォント設定: {font}")
                    use_japanese = True
                    WEEKDAY_NAMES = ['月', '火', '水', '木', '金', '土', '日']
                    return True

        else:
            fonts = ['IPAGothic', 'IPAPGothic', 'TakaoGothic', 'Noto Sans CJK JP']
            for font in fonts:
                if font in [f.name for f in fm.fontManager.ttflist]:
                    plt.rcParams['font.family'] = font
                    print(f"フォント設定: {font}")
                    use_japanese = True
                    WEEKDAY_NAMES = ['月', '火', '水', '木', '金', '土', '日']
                    return True

        print("日本語フォントが見つかりません。英語表示にします。")
        return False

    except Exception as e:
        print(f"フォント設定エラー: {e}")
        return False

if not FORCE_ENGLISH:
    setup_japanese_font()
else:
    print("英語表示モードが有効です")

plt.rcParams['axes.unicode_minus'] = False

# =============================================================================
# MT5接続と初期化
# =============================================================================

print("MT5に接続中...")
if not mt5.initialize():
    print("MT5への接続失敗。MT5が起動していることを確認してください。")
    quit()
print("MT5接続成功！")

# =============================================================================
# 関数定義
# =============================================================================

def detect_actual_symbols(target_symbols):
    """ブローカーで使用されている実際のシンボル名を検出"""
    actual_symbols = []
    all_symbols = mt5.symbols_get()

    if not all_symbols:
        print("利用可能なシンボルが取得できません")
        return []

    print("\nシンボル名を確認中...")

    for target in target_symbols:
        found = False

        for symbol in all_symbols:
            if symbol.name.upper() == target.upper():
                actual_symbols.append(symbol.name)
                print(f"{target} -> {symbol.name} [OK]")
                found = True
                break

        if not found:
            for symbol in all_symbols:
                if target.upper() in symbol.name.upper():
                    actual_symbols.append(symbol.name)
                    print(f"{target} -> {symbol.name} [OK]")
                    found = True
                    break

        if not found:
            print(f"{target} -> not found [NG]")
            if target == 'XAUUSD':
                for gold in ['GOLD', 'Gold', 'XAU']:
                    for symbol in all_symbols:
                        if gold.upper() in symbol.name.upper():
                            actual_symbols.append(symbol.name)
                            print(f"{target} -> {symbol.name} [OK] (GOLD)")
                            found = True
                            break
                    if found:
                        break

    return actual_symbols

def fetch_data(symbol, count=DATA_COUNT):
    """データ取得とpip計算"""
    print(f"\n{symbol}のデータ取得中...")

    symbol_info = mt5.symbol_info(symbol)
    if not symbol_info:
        print(f"エラー: {symbol}の情報が取得できません")
        return None

    if not symbol_info.visible:
        if not mt5.symbol_select(symbol, True):
            print(f"エラー: {symbol}を有効化できません")
            return None

    rates = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_M5, 0, count)
    if rates is None:
        print(f"エラー: {symbol}のデータ取得失敗")
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

    print(f"取得完了: {len(df)}件、期間: {df['time'].min()} ～ {df['time'].max()}")
    print(f"平均スプレッド: {df['spread_pips'].mean():.1f} pips")

    return df

def backtest_time_range(df, entry_h, entry_m, exit_h, exit_m, direction='long'):
    """特定時間帯のバックテスト"""
    max_spread = df.attrs['spread_mean'] * DEFAULT_SPREAD_MULTIPLIER

    entry_mask = (df['hour'] == entry_h) & (df['minute'] == entry_m) & (df['spread_pips'] <= max_spread)
    # 保持時間（分）: エグジット時刻がエントリー時刻より前なら翌日扱い
    hold_minutes = ((exit_h * 60 + exit_m) - (entry_h * 60 + entry_m)) % (24 * 60)
    if hold_minutes == 0:
        hold_minutes = 24 * 60

    entries = df[entry_mask][['time', 'date', 'open', 'spread_pips', 'weekday']].copy()
    entries['exit_time'] = entries['time'] + timedelta(minutes=hold_minutes)

    # エントリー時刻 + 保持時間 のタイムスタンプでエグジット足を引く（同日結合による先読みを防止）
    exits = df[['time', 'close', 'spread_pips']].rename(columns={'time': 'exit_time'})
    trades = pd.merge(entries, exits, on='exit_time', suffixes=('_entry', '_exit'))

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
    """最適な時間帯を探索（3〜12時間保持）"""
    print(f"\n{symbol}: 最適時間帯を探索中（保持時間: 3〜12時間）...")

    results = []
    total_tests = 0

    for entry_h in range(0, 24):
        for entry_m in [0, 30]:
            for hold_h in range(3, 13):  # 3〜12時間
                exit_h = (entry_h + hold_h) % 24
                exit_m = entry_m

                total_tests += 2

                if total_tests % 100 == 0:
                    print(f"  テスト中... {total_tests}個の戦略を検証")

                long_result = backtest_time_range(df, entry_h, entry_m, exit_h, exit_m, 'long')
                if long_result and long_result['count'] >= MIN_TRADES:
                    results.append({
                        'symbol': symbol,
                        'direction': 'long',
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
                        'symbol': symbol,
                        'direction': 'short',
                        'entry': f"{entry_h:02d}:{entry_m:02d}",
                        'exit': f"{exit_h:02d}:{exit_m:02d}",
                        'hold_hours': hold_h,
                        'count': short_result['count'],
                        'win_rate': short_result['win_rate'],
                        'avg_return': short_result['avg_return'],
                        'sharpe': short_result['sharpe']
                    })

    print(f"  完了: {total_tests}個の戦略をテスト")

    results_df = pd.DataFrame(results)

    if len(results_df) > 0:
        profitable = results_df[results_df['avg_return'] > 0].sort_values('sharpe', ascending=False)
        return profitable

    return pd.DataFrame()

def detailed_backtest(df, entry_h, entry_m, exit_h, exit_m, direction):
    """詳細なバックテストと曜日別分析"""
    result = backtest_time_range(df, entry_h, entry_m, exit_h, exit_m, direction)

    if not result:
        return None, None

    trades = result['trades']
    trades['cumulative'] = (1 + trades['profit']).cumprod() - 1

    weekday_stats = trades.groupby('weekday').agg({
        'profit': ['mean', 'count', lambda x: (x > 0).mean()]
    })
    weekday_stats.columns = ['avg_return', 'count', 'win_rate']

    return trades, weekday_stats

def plot_results(trades, weekday_stats, symbol, entry_time, exit_time, direction):
    """結果の可視化"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    ax = axes[0, 0]
    ax.plot(trades['cumulative'] * 100, linewidth=2)
    ax.set_title(f'{symbol} {direction.upper()}: {entry_time} → {exit_time}')
    ax.set_ylabel('Cumulative Return %' if not use_japanese else '累積リターン %')
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0, color='r', linestyle='--')

    ax = axes[0, 1]
    profits = trades['profit'] * 100
    ax.hist(profits, bins=30, alpha=0.7)
    ax.axvline(x=profits.mean(), color='r', linestyle='-', linewidth=2)
    ax.set_title('Return Distribution' if not use_japanese else 'リターン分布')
    ax.set_xlabel('Return %' if not use_japanese else 'リターン %')

    ax = axes[1, 0]
    weekday_returns = weekday_stats['avg_return'] * 100
    weekday_names = [WEEKDAY_NAMES[i] for i in weekday_returns.index]
    colors = ['green' if x > 0 else 'red' for x in weekday_returns.values]
    ax.bar(weekday_names, weekday_returns.values, color=colors, alpha=0.7)
    ax.set_title('Average Return by Weekday' if not use_japanese else '曜日別平均リターン')
    ax.set_ylabel('Average Return %' if not use_japanese else '平均リターン %')
    ax.axhline(y=0, color='black', linestyle='-')

    ax = axes[1, 1]
    ax.axis('off')

    win_rate = (trades['profit'] > 0).mean()
    avg_return = trades['profit'].mean()
    total_return = trades['cumulative'].iloc[-1]
    sharpe = trades['profit'].mean() / trades['profit'].std() if trades['profit'].std() > 0 else 0

    summary_text = f"""
=== Performance Summary ===

Symbol: {symbol}
Direction: {direction.upper()}
Entry: {entry_time}
Exit: {exit_time}

Trades: {len(trades)}
Win Rate: {win_rate:.1%}
Avg Return: {avg_return:.4%}
Total Return: {total_return:.2%}
Sharpe Ratio: {sharpe:.3f}
"""

    ax.text(0.1, 0.9, summary_text, transform=ax.transAxes, fontsize=12,
            verticalalignment='top', fontfamily='monospace')

    plt.tight_layout()
    plt.show()

# =============================================================================
# メイン処理
# =============================================================================

def main():
    """メイン分析処理"""
    symbols = detect_actual_symbols(TARGET_SYMBOLS)

    if not symbols:
        print("\n指定された通貨ペアが見つかりません")
        print("ブローカーのシンボル名を確認してください")
        mt5.shutdown()
        return

    print(f"\n分析対象通貨ペア: {symbols}")

    if SPECIFIC_ENTRY_TIME and SPECIFIC_EXIT_TIME:
        print(f"\n特定時間帯のテスト: {SPECIFIC_ENTRY_TIME} → {SPECIFIC_EXIT_TIME}")
        entry_h, entry_m = map(int, SPECIFIC_ENTRY_TIME.split(':'))
        exit_h, exit_m = map(int, SPECIFIC_EXIT_TIME.split(':'))

        for symbol in symbols:
            df = fetch_data(symbol)
            if df is None:
                continue

            print(f"\n{symbol}:")

            long_result = backtest_time_range(df, entry_h, entry_m, exit_h, exit_m, 'long')
            if long_result:
                print(f"  Long: Trades={long_result['count']}, WinRate={long_result['win_rate']:.1%}, AvgReturn={long_result['avg_return']:.4%}")

            short_result = backtest_time_range(df, entry_h, entry_m, exit_h, exit_m, 'short')
            if short_result:
                print(f"  Short: Trades={short_result['count']}, WinRate={short_result['win_rate']:.1%}, AvgReturn={short_result['avg_return']:.4%}")

        mt5.shutdown()
        return

    all_results = []

    for symbol in symbols:
        df = fetch_data(symbol)
        if df is None:
            continue

        profitable = find_best_times(df, symbol)

        if len(profitable) > 0:
            print(f"\n{symbol}: {len(profitable)}個の利益戦略を発見")
            all_results.append({
                'symbol': symbol,
                'data': df,
                'strategies': profitable
            })
        else:
            print(f"\n{symbol}: 利益が出る戦略が見つかりませんでした")

    if all_results:
        print("\n" + "="*80)
        print("全通貨ペア 最良戦略 TOP100（保持時間: 3〜12時間）")
        print("="*80)

        all_strategies = []
        for result in all_results:
            all_strategies.extend(result['strategies'].to_dict('records'))

        all_strategies_df = pd.DataFrame(all_strategies)
        top_strategies = all_strategies_df.sort_values('sharpe', ascending=False).head(100)

        print("\n")
        display_df = top_strategies[['symbol', 'direction', 'entry', 'exit', 'hold_hours', 'count', 'win_rate', 'avg_return', 'sharpe']].copy()
        display_df['win_rate'] = (display_df['win_rate'] * 100).round(1)
        display_df['avg_return'] = (display_df['avg_return'] * 100).round(4)
        display_df['sharpe'] = display_df['sharpe'].round(3)

        display_df.columns = ['Symbol', 'Direction', 'Entry', 'Exit', 'Hold(h)', 'Trades', 'Win%', 'AvgReturn%', 'Sharpe']

        # 全100件を表示
        pd.set_option('display.max_rows', 120)
        pd.set_option('display.width', 120)
        print(display_df.to_string(index=False))

        print("\n" + "="*80)
        print("TOP3戦略の詳細分析")
        print("="*80)

        for i in range(min(3, len(top_strategies))):
            strategy = top_strategies.iloc[i]

            df = None
            for result in all_results:
                if result['symbol'] == strategy['symbol']:
                    df = result['data']
                    break

            if df is None:
                continue

            entry_h, entry_m = map(int, strategy['entry'].split(':'))
            exit_h, exit_m = map(int, strategy['exit'].split(':'))

            print(f"\n【{i+1}位】 {strategy['symbol']} {strategy['direction'].upper()}: {strategy['entry']} → {strategy['exit']} (保持{strategy['hold_hours']}時間)")
            print(f"Sharpe Ratio: {strategy['sharpe']:.3f}, Avg Return: {strategy['avg_return']:.4%}")

            trades, weekday_stats = detailed_backtest(df, entry_h, entry_m, exit_h, exit_m, strategy['direction'])

            if trades is not None:
                plot_results(trades, weekday_stats, strategy['symbol'], strategy['entry'], strategy['exit'], strategy['direction'])

    else:
        print("\n利益が出る戦略が見つかりませんでした")
        print("以下を確認してください:")
        print("1. 十分なヒストリカルデータがあるか")
        print(f"2. スプレッドフィルタが厳しすぎないか（現在: 平均の{DEFAULT_SPREAD_MULTIPLIER}倍）")
        print(f"3. 最小トレード数の条件（{MIN_TRADES}回）が厳しすぎないか")

    mt5.shutdown()
    print("\n分析完了！")

# =============================================================================
# 実行
# =============================================================================

if __name__ == "__main__":
    main()
