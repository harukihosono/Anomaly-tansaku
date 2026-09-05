#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
FXアノマリー自動探索装置（高速版・期間別レポート対応）

- 全通貨ペア × エントリー時刻（30分刻み）× 保持時間（3〜12時間）× Long/Short を総当たり
- 1ヶ月 / 3ヶ月 / 6ヶ月 の3期間で実行し、期間ごとに画像と Markdown レポートを出力
- TOP100 ランキング表を画像として保存
- バックテストは NumPy で完全ベクトル化（1ペア・1期間あたり 0.1 秒程度）
- データは最長期間を1回だけ取得し、短い期間はスライスして再利用

使い方:
    python generate_report.py                                  # 全ペア・全期間
    python generate_report.py --periods 1m                     # 1ヶ月だけ
    python generate_report.py --symbols EURUSD,USDJPY --periods 1m,3m
    python generate_report.py --skip-images                    # 画像なし（速度確認用）
"""

import sys
import io
import os
import time
import argparse
import warnings

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace', line_buffering=True)

import numpy as np
import pandas as pd
import MetaTrader5 as mt5
from datetime import datetime, timedelta, timezone
import matplotlib
matplotlib.use('Agg')  # GUI不要のバックエンド
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings('ignore')

# =============================================================================
# 設定（ここを編集してカスタマイズ）
# =============================================================================

# 分析対象通貨ペア（ブローカーのサフィックスは自動検出。Exness なら EURUSDm など）
TARGET_SYMBOLS = [
    # メジャー
    'EURUSD', 'USDJPY', 'GBPUSD', 'USDCHF', 'AUDUSD', 'USDCAD', 'NZDUSD',
    # クロス円
    'EURJPY', 'GBPJPY', 'AUDJPY', 'NZDJPY', 'CADJPY', 'CHFJPY',
    # EURクロス
    'EURGBP', 'EURAUD', 'EURNZD', 'EURCAD', 'EURCHF',
    # GBPクロス
    'GBPAUD', 'GBPNZD', 'GBPCAD', 'GBPCHF',
    # AUDクロス
    'AUDNZD', 'AUDCAD', 'AUDCHF',
    # その他クロス
    'NZDCAD', 'NZDCHF', 'CADCHF',
    # 貴金属
    'XAUUSD',"JP225","US30"
]

# 分析期間（ラベル, 画像用の英語表記, 日数）。直近 N 日を対象にする
PERIODS = [
    ('1m', '1 Month', 30),
    ('3m', '3 Months', 90),
    ('6m', '6 Months', 180),
]

ENTRY_MINUTES = [0, 30]            # エントリー時刻の分（00分と30分）
HOLD_HOURS = range(3, 13)          # 保持時間 3〜12時間
DEFAULT_SPREAD_MULTIPLIER = 5.0    # 平均スプレッドの何倍までエントリーを許容するか
MIN_TRADES_RATIO = 0.5             # 最小トレード数 = 取引日数 × この割合（期間に応じて自動調整）
MIN_TRADES_FLOOR = 15              # 最小トレード数の下限
TIMEZONE_HOURS = 9                 # サーバー時間(UTC)からのシフト。日本時間 = +9
TOP_N_TABLE = 100                  # ランキング表の件数
TOP_N_DETAIL = 10                  # 詳細チャートを出す件数
WEEKDAY_NAMES = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']

OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))
IMG_ROOT = os.path.join(OUTPUT_DIR, 'images')
CSV_DIR = os.path.join(OUTPUT_DIR, 'results')

plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 150

KEY_COLS = ['symbol', 'direction', 'entry', 'exit']

# =============================================================================
# MT5 接続とシンボル検出
# =============================================================================

def connect_mt5():
    print("MT5に接続中...")
    if not mt5.initialize():
        print("MT5への接続失敗。MT5が起動していることを確認してください。", mt5.last_error())
        sys.exit(1)
    ti = mt5.terminal_info()
    ai = mt5.account_info()
    print(f"MT5接続成功: {ti.company} / server={ai.server if ai else '?'}")


def detect_actual_symbols(targets):
    """ブローカーで使われている実際のシンボル名を検出（サフィックス付きにも対応）"""
    names = [s.name for s in (mt5.symbols_get() or [])]
    if not names:
        print("利用可能なシンボルが取得できません")
        return []
    found = []
    print("\nシンボル名を確認中...")
    for t in targets:
        exact = [n for n in names if n.upper() == t.upper()]
        cands = exact or sorted([n for n in names if t.upper() in n.upper()], key=len)
        if not cands and t == 'XAUUSD':
            cands = sorted([n for n in names if 'GOLD' in n.upper()], key=len)
        if cands:
            found.append(cands[0])
            print(f"  {t} -> {cands[0]} [OK]")
        else:
            print(f"  {t} -> not found [NG]")
    return found

# =============================================================================
# データ取得
# =============================================================================

def fetch_full(symbol, days):
    """直近 days 日分の M5 データを取得（最長期間を1回だけ取り、短い期間はスライスする）"""
    info = mt5.symbol_info(symbol)
    if info is None:
        return None
    if not info.visible and not mt5.symbol_select(symbol, True):
        return None

    utc_to = datetime.now(timezone.utc)
    utc_from = utc_to - timedelta(days=days)
    rates = mt5.copy_rates_range(symbol, mt5.TIMEFRAME_M5, utc_from, utc_to)
    if rates is None or len(rates) == 0:
        return None

    df = pd.DataFrame(rates)
    df['time'] = pd.to_datetime(df['time'], unit='s') + timedelta(hours=TIMEZONE_HOURS)
    df = df.drop_duplicates('time').sort_values('time').reset_index(drop=True)

    digits = info.digits
    pip_value = 10 ** (-(digits - 1)) if digits >= 3 else 10 ** (-digits)
    df['spread_pips'] = df['spread'] / 10 if digits in (3, 5) else df['spread']

    df.attrs['symbol'] = symbol
    df.attrs['pip_value'] = pip_value
    df.attrs['digits'] = digits
    return df


def slice_period(df_full, days):
    """取得済みデータから直近 days 日分を切り出す"""
    now_local = pd.Timestamp(datetime.now(timezone.utc)).tz_localize(None) + timedelta(hours=TIMEZONE_HOURS)
    cutoff = now_local - timedelta(days=days)
    df = df_full[df_full['time'] >= cutoff].reset_index(drop=True)
    df.attrs.update(df_full.attrs)
    df.attrs['spread_mean'] = float(df['spread_pips'].mean()) if len(df) else 0.0
    return df

# =============================================================================
# 高速バックテスト（NumPy ベクトル化）
# =============================================================================

class FastBacktester:
    """1ペア・1期間分の価格配列を保持し、NumPy で高速にバックテストする

    エグジットは「エントリー時刻 + 保持時間」のタイムスタンプで厳密に照合する。
    足が存在しない（週末・休場）場合はそのトレードを除外するので、先読みは起きない。
    """

    def __init__(self, df):
        self.symbol = df.attrs['symbol']
        self.pip_value = df.attrs['pip_value']
        self.max_spread = df.attrs['spread_mean'] * DEFAULT_SPREAD_MULTIPLIER

        t = df['time'].to_numpy()
        self.time = t
        self.t_min = t.astype('datetime64[m]').astype(np.int64)  # 分単位の整数（照合用）
        self.open = df['open'].to_numpy(dtype=float)
        self.close = df['close'].to_numpy(dtype=float)
        self.spread = df['spread_pips'].to_numpy(dtype=float)

        dt = df['time'].dt
        self.hour = dt.hour.to_numpy()
        self.minute = dt.minute.to_numpy()
        self.weekday = dt.weekday.to_numpy()

        weekday_rows = df.loc[dt.weekday < 5, 'time'].dt.normalize()
        self.trading_days = int(weekday_rows.nunique())
        self.min_trades = max(MIN_TRADES_FLOOR, int(self.trading_days * MIN_TRADES_RATIO))
        self.date_min = pd.Timestamp(t[0]) if len(t) else None
        self.date_max = pd.Timestamp(t[-1]) if len(t) else None

    # --- 基本部品 ---
    def entry_indices(self, entry_h, entry_m):
        mask = (self.hour == entry_h) & (self.minute == entry_m) & (self.spread <= self.max_spread)
        return np.flatnonzero(mask)

    def match_exits(self, entry_idx, hold_minutes):
        """エントリー行 + 保持時間 に一致するエグジット行を二分探索で引く（無ければ除外）"""
        target = self.t_min[entry_idx] + hold_minutes
        pos = np.searchsorted(self.t_min, target)
        pos = np.minimum(pos, len(self.t_min) - 1)
        ok = self.t_min[pos] == target
        return entry_idx[ok], pos[ok]

    def profits(self, ei, xi, direction):
        if direction == 'long':
            entry_price = self.open[ei] + self.spread[ei] * self.pip_value
            return (self.close[xi] - entry_price) / entry_price
        entry_price = self.open[ei]
        exit_price = self.close[xi] + self.spread[xi] * self.pip_value
        return (entry_price - exit_price) / entry_price

    @staticmethod
    def hold_minutes(entry_h, entry_m, exit_h, exit_m):
        hm = ((exit_h * 60 + exit_m) - (entry_h * 60 + entry_m)) % (24 * 60)
        return hm if hm else 24 * 60

    # --- 総当たり ---
    def run_grid(self):
        rows = []
        for entry_h in range(24):
            for entry_m in ENTRY_MINUTES:
                entry_idx = self.entry_indices(entry_h, entry_m)
                if len(entry_idx) < self.min_trades:
                    continue
                for hold_h in HOLD_HOURS:
                    ei, xi = self.match_exits(entry_idx, hold_h * 60)
                    if len(ei) < self.min_trades:
                        continue
                    exit_h = (entry_h + hold_h) % 24
                    for direction in ('long', 'short'):
                        p = self.profits(ei, xi, direction)
                        std = p.std(ddof=1)
                        rows.append({
                            'symbol': self.symbol,
                            'direction': direction,
                            'entry': f"{entry_h:02d}:{entry_m:02d}",
                            'exit': f"{exit_h:02d}:{entry_m:02d}",
                            'hold_hours': hold_h,
                            'count': int(len(p)),
                            'win_rate': float((p > 0).mean()),
                            'avg_return': float(p.mean()),
                            'sharpe': float(p.mean() / std) if std > 0 else 0.0,
                        })
        return pd.DataFrame(rows)

    # --- 個別戦略のトレード明細（詳細チャート用） ---
    def trades(self, entry_h, entry_m, exit_h, exit_m, direction):
        hm = self.hold_minutes(entry_h, entry_m, exit_h, exit_m)
        ei, xi = self.match_exits(self.entry_indices(entry_h, entry_m), hm)
        if len(ei) == 0:
            return None
        p = self.profits(ei, xi, direction)
        tr = pd.DataFrame({
            'time': self.time[ei],
            'exit_time': self.time[xi],
            'weekday': self.weekday[ei],
            'profit': p,
        })
        tr['cumulative'] = (1 + tr['profit']).cumprod() - 1
        return tr


def rank_strategies(all_df):
    """利益が出た戦略だけを Sharpe 順に並べる"""
    if all_df is None or all_df.empty:
        return pd.DataFrame(columns=KEY_COLS + ['hold_hours', 'count', 'win_rate', 'avg_return', 'sharpe'])
    return all_df[all_df['avg_return'] > 0].sort_values('sharpe', ascending=False).reset_index(drop=True)

# =============================================================================
# 画像出力
# =============================================================================

def _save(fig, img_dir, filename):
    path = os.path.join(img_dir, filename)
    fig.savefig(path, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"    Saved: {os.path.relpath(path, OUTPUT_DIR)}")
    return filename


def save_top100_table(top, img_dir, plabel):
    """TOP100 ランキング表を画像化（左右2パネル × 50行）"""
    n = min(TOP_N_TABLE, len(top))
    if n == 0:
        return None
    cols = ['Rank', 'Symbol', 'Dir', 'Entry', 'Exit', 'Hold', 'Trades', 'Win%', 'AvgRet%', 'Sharpe']
    rows = []
    for i, r in top.head(n).iterrows():
        rows.append([i + 1, r['symbol'], r['direction'].upper(), r['entry'], r['exit'], f"{r['hold_hours']}h",
                     r['count'], f"{r['win_rate'] * 100:.1f}", f"{r['avg_return'] * 100:.4f}", f"{r['sharpe']:.3f}"])
    half = (n + 1) // 2
    panels = [rows[:half], rows[half:]]

    col_widths = [0.07, 0.15, 0.09, 0.09, 0.09, 0.07, 0.09, 0.09, 0.13, 0.10]
    fig, axes = plt.subplots(1, 2, figsize=(16, 0.26 * half + 1.0))
    fig.subplots_adjust(left=0.01, right=0.99, top=0.95, bottom=0.01, wspace=0.04)
    for ax, panel in zip(axes, panels):
        ax.axis('off')
        if not panel:
            continue
        # bbox=[0,0,1,1] でパネル全体に表を広げる
        tbl = ax.table(cellText=panel, colLabels=cols, colWidths=col_widths, cellLoc='center', bbox=[0, 0, 1, 1])
        tbl.auto_set_font_size(False)
        tbl.set_fontsize(9)
        for (row, col), cell in tbl.get_celld().items():
            cell.set_edgecolor('#DDDDDD')
            if row == 0:
                cell.set_facecolor('#37474F')
                cell.set_text_props(color='white', fontweight='bold')
            else:
                is_long = panel[row - 1][2] == 'LONG'
                cell.set_facecolor('#E3F2FD' if is_long else '#FFEBEE')
                if col == 9:
                    cell.set_text_props(fontweight='bold')
    fig.suptitle(f'TOP{n} Strategies by Sharpe Ratio ({plabel})   Blue = Long / Red = Short   Time = JST',
                 fontsize=15, fontweight='bold', y=0.985)
    return _save(fig, img_dir, 'top100_ranking.png')


def save_overview_chart(top, img_dir, plabel):
    """TOP20 の Sharpe 横棒グラフ"""
    top20 = top.head(20).iloc[::-1]
    fig, ax = plt.subplots(figsize=(14, 10))
    labels = [f"{r['symbol']} {r['direction'].upper()} {r['entry']}->{r['exit']}" for _, r in top20.iterrows()]
    colors = ['#2196F3' if d == 'long' else '#F44336' for d in top20['direction']]
    bars = ax.barh(range(len(labels)), top20['sharpe'].values, color=colors, alpha=0.85, edgecolor='white')
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=10)
    ax.set_xlabel('Sharpe Ratio (per trade)', fontsize=12)
    ax.set_title(f'TOP 20 Strategies by Sharpe Ratio ({plabel})  Blue=Long, Red=Short', fontsize=14, fontweight='bold')
    ax.grid(True, axis='x', alpha=0.3)
    for bar, val in zip(bars, top20['sharpe'].values):
        ax.text(val + 0.005, bar.get_y() + bar.get_height() / 2., f'{val:.3f}', ha='left', va='center', fontsize=9)
    fig.tight_layout()
    return _save(fig, img_dir, 'top20_sharpe_ranking.png')


def save_heatmap(top, img_dir, plabel):
    """通貨ペア × 方向 の TOP100 入り数ヒートマップ"""
    pivot = top.head(TOP_N_TABLE).groupby(['symbol', 'direction']).size().unstack(fill_value=0)
    fig, ax = plt.subplots(figsize=(8, max(4, 0.45 * len(pivot) + 1)))
    sns.heatmap(pivot, annot=True, fmt='d', cmap='YlOrRd', ax=ax, linewidths=0.5)
    ax.set_title(f'Strategy Count in TOP{TOP_N_TABLE} by Symbol & Direction ({plabel})', fontsize=13, fontweight='bold')
    ax.set_ylabel('Symbol')
    ax.set_xlabel('Direction')
    fig.tight_layout()
    return _save(fig, img_dir, 'strategy_heatmap.png')


def save_hold_hours_distribution(top, img_dir, plabel):
    top100 = top.head(TOP_N_TABLE)
    fig, ax = plt.subplots(figsize=(10, 6))
    hours = top100['hold_hours'].value_counts().sort_index()
    ax.bar(hours.index, hours.values, color='#9C27B0', alpha=0.85, edgecolor='white')
    ax.set_xlabel('Hold Hours', fontsize=12)
    ax.set_ylabel(f'Count in TOP{TOP_N_TABLE}', fontsize=12)
    ax.set_title(f'Distribution of Hold Hours in TOP{TOP_N_TABLE} ({plabel})', fontsize=14, fontweight='bold')
    ax.set_xticks(list(HOLD_HOURS))
    ax.grid(True, axis='y', alpha=0.3)
    for x, y in zip(hours.index, hours.values):
        ax.text(x, y + 0.3, str(y), ha='center', fontsize=11, fontweight='bold')
    fig.tight_layout()
    return _save(fig, img_dir, 'hold_hours_distribution.png')


def save_winrate_vs_sharpe(top, img_dir, plabel):
    top100 = top.head(TOP_N_TABLE)
    fig, ax = plt.subplots(figsize=(10, 8))
    colors = ['#2196F3' if d == 'long' else '#F44336' for d in top100['direction']]
    ax.scatter(top100['win_rate'] * 100, top100['sharpe'], c=colors, s=top100['count'] * 2, alpha=0.6, edgecolors='white')
    ax.set_xlabel('Win Rate %', fontsize=12)
    ax.set_ylabel('Sharpe Ratio', fontsize=12)
    ax.set_title(f'Win Rate vs Sharpe ({plabel})  Size=Trades, Blue=Long, Red=Short', fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3)
    for i in range(min(5, len(top100))):
        r = top100.iloc[i]
        ax.annotate(r['symbol'], (r['win_rate'] * 100, r['sharpe']), fontsize=8, ha='left', va='bottom')
    fig.tight_layout()
    return _save(fig, img_dir, 'winrate_vs_sharpe.png')


def save_entry_time_distribution(top, img_dir, plabel):
    top100 = top.head(TOP_N_TABLE).copy()
    top100['entry_hour'] = top100['entry'].str.slice(0, 2).astype(int)
    counts = top100['entry_hour'].value_counts().reindex(range(24), fill_value=0)
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(counts.index, counts.values, color='#FF9800', alpha=0.85, edgecolor='white')
    ax.set_xlabel('Entry Hour (JST)', fontsize=12)
    ax.set_ylabel(f'Count in TOP{TOP_N_TABLE}', fontsize=12)
    ax.set_title(f'Entry Time Distribution in TOP{TOP_N_TABLE} ({plabel})', fontsize=14, fontweight='bold')
    ax.set_xticks(range(24))
    ax.set_xticklabels([f'{h:02d}' for h in range(24)])
    ax.grid(True, axis='y', alpha=0.3)
    fig.tight_layout()
    return _save(fig, img_dir, 'entry_time_distribution.png')


def save_detailed_chart(bt, strategy, rank, img_dir, plabel):
    """1戦略の詳細チャート（累積リターン・分布・曜日別・サマリー）"""
    entry_h, entry_m = map(int, strategy['entry'].split(':'))
    exit_h, exit_m = map(int, strategy['exit'].split(':'))
    direction = strategy['direction']
    trades = bt.trades(entry_h, entry_m, exit_h, exit_m, direction)
    if trades is None or trades.empty:
        return None

    weekday_stats = trades.groupby('weekday')['profit'].agg(['mean', 'count'])
    symbol = bt.symbol
    entry_time, exit_time = strategy['entry'], strategy['exit']

    fig, axes = plt.subplots(2, 2, figsize=(16, 11))
    fig.suptitle(f"#{rank} {symbol} {direction.upper()}: {entry_time} -> {exit_time} ({plabel})", fontsize=16, fontweight='bold')

    ax = axes[0, 0]
    x = np.arange(len(trades))
    ax.plot(x, trades['cumulative'].values * 100, linewidth=2, color='#2196F3')
    ax.fill_between(x, 0, trades['cumulative'].values * 100, alpha=0.1, color='#2196F3')
    ax.set_title('Cumulative Return', fontsize=13)
    ax.set_ylabel('Return %')
    ax.set_xlabel('Trade #')
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0, color='r', linestyle='--', alpha=0.5)

    ax = axes[0, 1]
    profits = trades['profit'] * 100
    ax.hist(profits, bins=30, alpha=0.7, color='#4CAF50', edgecolor='white')
    ax.axvline(x=profits.mean(), color='r', linewidth=2, label=f'Mean: {profits.mean():.4f}%')
    ax.set_title('Return Distribution', fontsize=13)
    ax.set_xlabel('Return %')
    ax.set_ylabel('Frequency')
    ax.legend()

    ax = axes[1, 0]
    wr = weekday_stats['mean'] * 100
    names = [WEEKDAY_NAMES[i] for i in wr.index]
    colors = ['#4CAF50' if v > 0 else '#F44336' for v in wr.values]
    bars = ax.bar(names, wr.values, color=colors, alpha=0.85, edgecolor='white')
    ax.set_title('Average Return by Weekday', fontsize=13)
    ax.set_ylabel('Avg Return %')
    ax.axhline(y=0, color='black', alpha=0.3)
    for bar, val, cnt in zip(bars, wr.values, weekday_stats['count'].values):
        ax.text(bar.get_x() + bar.get_width() / 2., bar.get_height(), f'{val:.4f}%\n(n={cnt})',
                ha='center', va='bottom' if val >= 0 else 'top', fontsize=9)

    ax = axes[1, 1]
    ax.axis('off')
    win_rate = (trades['profit'] > 0).mean()
    avg_return = trades['profit'].mean()
    total_return = trades['cumulative'].iloc[-1]
    std = trades['profit'].std()
    sharpe = avg_return / std if std > 0 else 0
    max_dd = (trades['cumulative'] - trades['cumulative'].cummax()).min()
    summary = (
        f"Performance Summary ({plabel})\n{'=' * 32}\n\n"
        f"Symbol:       {symbol}\nDirection:    {direction.upper()}\n"
        f"Entry:        {entry_time} JST\nExit:         {exit_time} JST\n"
        f"Hold:         {strategy['hold_hours']}h\n\n"
        f"Period:       {trades['time'].min():%Y-%m-%d} - {trades['time'].max():%Y-%m-%d}\n"
        f"Trades:       {len(trades)}\nWin Rate:     {win_rate:.1%}\n"
        f"Avg Return:   {avg_return:.4%}\nTotal Return: {total_return:.2%}\n"
        f"Sharpe Ratio: {sharpe:.3f}\nMax Drawdown: {max_dd:.2%}"
    )
    ax.text(0.05, 0.95, summary, transform=ax.transAxes, fontsize=12, verticalalignment='top',
            fontfamily='monospace', bbox=dict(boxstyle='round,pad=0.5', facecolor='#f0f0f0', alpha=0.8))

    fig.tight_layout()
    filename = f"top{rank}_{symbol}_{direction}_{entry_time.replace(':', '')}_{exit_time.replace(':', '')}.png"
    return _save(fig, img_dir, filename)

# =============================================================================
# 期間横断の分析
# =============================================================================

def cross_period_analysis(period_results, labels):
    """複数期間の結果を突き合わせる

    戻り値:
      common_df : 全期間で TOP100 に入った戦略
      robust_df : 最長期間の TOP100 のうち、全期間で平均リターンがプラスだった戦略
      overlap   : 期間ペアごとの TOP100 重複数
    """
    labels = [l for l in labels if l in period_results]
    if len(labels) < 2:
        return None, None, {}

    tops = {l: period_results[l]['ranked'].head(TOP_N_TABLE) for l in labels}
    sets = {l: set(map(tuple, tops[l][KEY_COLS].to_numpy())) for l in labels}
    indexed = {l: period_results[l]['all'].set_index(KEY_COLS) for l in labels}

    def lookup(l, key, col):
        try:
            v = indexed[l].loc[key, col]
            return float(v.iloc[0]) if isinstance(v, pd.Series) else float(v)
        except KeyError:
            return np.nan

    overlap = {}
    for i, a in enumerate(labels):
        for b in labels[i + 1:]:
            overlap[(a, b)] = len(sets[a] & sets[b])

    longest = labels[-1]

    common_rows = []
    for key in set.intersection(*sets.values()):
        row = {'symbol': key[0], 'direction': key[1], 'entry': key[2], 'exit': key[3],
               'hold_hours': int(lookup(longest, key, 'hold_hours'))}
        for l in labels:
            row[f'sharpe_{l}'] = lookup(l, key, 'sharpe')
            row[f'trades_{l}'] = lookup(l, key, 'count')
        row['min_sharpe'] = min(row[f'sharpe_{l}'] for l in labels)
        common_rows.append(row)
    common_df = pd.DataFrame(common_rows)
    if not common_df.empty:
        common_df = common_df.sort_values('min_sharpe', ascending=False).reset_index(drop=True)

    robust_rows = []
    for _, r in tops[longest].iterrows():
        key = tuple(r[KEY_COLS])
        rets = {l: lookup(l, key, 'avg_return') for l in labels}
        if all(pd.notna(v) and v > 0 for v in rets.values()):
            row = {'symbol': key[0], 'direction': key[1], 'entry': key[2], 'exit': key[3], 'hold_hours': int(r['hold_hours'])}
            for l in labels:
                row[f'sharpe_{l}'] = lookup(l, key, 'sharpe')
                row[f'win_{l}'] = lookup(l, key, 'win_rate')
            robust_rows.append(row)
    robust_df = pd.DataFrame(robust_rows)
    return common_df, robust_df, overlap

# =============================================================================
# Markdown レポート
# =============================================================================

def md_table(df, columns, formats):
    """DataFrame を Markdown 表に変換。columns は (列名, 見出し) のリスト、formats は列名 -> フォーマット関数"""
    lines = ['| ' + ' | '.join(h for _, h in columns) + ' |',
             '|' + '|'.join('---' for _ in columns) + '|']
    for _, r in df.iterrows():
        cells = []
        for col, _ in columns:
            v = r[col]
            f = formats.get(col)
            cells.append(f(v) if f else str(v))
        lines.append('| ' + ' | '.join(cells) + ' |')
    return lines


def period_label_ja(label, days):
    return {'1m': '1ヶ月', '3m': '3ヶ月', '6m': '6ヶ月'}.get(label, label) + f'（直近{days}日）'


def build_report(period_results, periods, symbols, elapsed_total):
    md = []
    now_jst = datetime.now(timezone.utc) + timedelta(hours=TIMEZONE_HOURS)
    md += ["# FX Anomaly Auto-Search Report", "",
           f"> Generated: {now_jst:%Y-%m-%d %H:%M} JST | Hold: {HOLD_HOURS.start}-{HOLD_HOURS.stop - 1}h | "
           f"Spread filter: {DEFAULT_SPREAD_MULTIPLIER}x avg | Min trades: max({MIN_TRADES_FLOOR}, trading days x {MIN_TRADES_RATIO}) | "
           f"Timezone: GMT+{TIMEZONE_HOURS} (JST) | Elapsed: {elapsed_total:.0f}s", ""]

    labels = [p[0] for p in periods if p[0] in period_results]

    # --- 期間サマリー ---
    md += ["## Summary", "", f"- **Analyzed Pairs**: {len(symbols)}", "",
           "| Period | Data range (JST) | Bars/pair | Trading days | Min trades | Tested | Profitable | #1 strategy | #1 Sharpe |",
           "|---|---|---|---|---|---|---|---|---|"]
    for l in labels:
        m = period_results[l]['meta']
        top1 = period_results[l]['ranked'].head(1)
        s1 = (f"{top1.iloc[0]['symbol']} {top1.iloc[0]['direction'].upper()} {top1.iloc[0]['entry']}->{top1.iloc[0]['exit']}"
              if len(top1) else '-')
        sh1 = f"{top1.iloc[0]['sharpe']:.3f}" if len(top1) else '-'
        md.append(f"| {period_label_ja(l, m['days'])} | {m['date_min']:%Y-%m-%d} - {m['date_max']:%Y-%m-%d} | {m['bars_avg']:,} | "
                  f"{m['trading_days']} | {m['min_trades']} | {m['n_tested']:,} | {m['n_profitable']:,} | {s1} | {sh1} |")
    md.append("")

    # --- 期間横断 ---
    common_df, robust_df, overlap = cross_period_analysis(period_results, labels)
    if common_df is not None:
        md += ["## Cross-Period Analysis（期間横断）", ""]
        md += ["### TOP100 overlap between periods", "", "| Periods | Common strategies in TOP100 |", "|---|---|"]
        for (a, b), n in overlap.items():
            md.append(f"| {a} & {b} | {n} |")
        md.append("")
        md += [f"### Strategies in TOP{TOP_N_TABLE} of ALL periods ({', '.join(labels)})", ""]
        if common_df.empty:
            md.append("_None_")
        else:
            cols = [('symbol', 'Symbol'), ('direction', 'Dir'), ('entry', 'Entry'), ('exit', 'Exit'), ('hold_hours', 'Hold')]
            cols += [(f'sharpe_{l}', f'Sharpe {l}') for l in labels] + [(f'trades_{l}', f'Trades {l}') for l in labels]
            fmt = {'direction': lambda v: v.upper(), 'hold_hours': lambda v: f'{v}h'}
            fmt.update({f'sharpe_{l}': (lambda v: f'{v:.3f}') for l in labels})
            fmt.update({f'trades_{l}': (lambda v: f'{int(v)}') for l in labels})
            md += md_table(common_df, cols, fmt)
        md.append("")
        md += [f"### {labels[-1]} TOP{TOP_N_TABLE} strategies that were also profitable in every shorter period", ""]
        if robust_df.empty:
            md.append("_None_")
        else:
            md.append(f"{len(robust_df)} of the {labels[-1]} TOP{TOP_N_TABLE} strategies had a positive average return in all periods. Top 30 by {labels[-1]} Sharpe:")
            md.append("")
            cols = [('symbol', 'Symbol'), ('direction', 'Dir'), ('entry', 'Entry'), ('exit', 'Exit'), ('hold_hours', 'Hold')]
            cols += [(f'sharpe_{l}', f'Sharpe {l}') for l in labels] + [(f'win_{l}', f'Win% {l}') for l in labels]
            fmt = {'direction': lambda v: v.upper(), 'hold_hours': lambda v: f'{v}h'}
            fmt.update({f'sharpe_{l}': (lambda v: f'{v:.3f}') for l in labels})
            fmt.update({f'win_{l}': (lambda v: f'{v * 100:.1f}') for l in labels})
            md += md_table(robust_df.head(30), cols, fmt)
        md.append("")

    # --- 期間ごとのセクション ---
    for l in labels:
        pr = period_results[l]
        m, ranked = pr['meta'], pr['ranked']
        top100 = ranked.head(TOP_N_TABLE)
        img = f"images/{l}"
        md += [f"## {period_label_ja(l, m['days'])}", "",
               f"- **Data range (JST)**: {m['date_min']:%Y-%m-%d %H:%M} - {m['date_max']:%Y-%m-%d %H:%M}",
               f"- **Bars per pair (avg)**: {m['bars_avg']:,} x 5min",
               f"- **Trading days**: {m['trading_days']} / **Min trades**: {m['min_trades']}",
               f"- **Strategies tested**: {m['n_tested']:,} / **Profitable**: {m['n_profitable']:,}",
               f"- **Search time**: {m['search_sec']:.1f}s", ""]
        if pr.get('images'):
            md += [f"### TOP{TOP_N_TABLE} Ranking ({l})", "", f"![TOP100 {l}]({img}/top100_ranking.png)", "",
                   f"### Overview Charts ({l})", "",
                   f"![TOP20 Sharpe {l}]({img}/top20_sharpe_ranking.png)", "",
                   f"![Win Rate vs Sharpe {l}]({img}/winrate_vs_sharpe.png)", "",
                   f"![Hold Hours {l}]({img}/hold_hours_distribution.png)", "",
                   f"![Entry Time {l}]({img}/entry_time_distribution.png)", "",
                   f"![Heatmap {l}]({img}/strategy_heatmap.png)", ""]
            md += [f"### TOP{TOP_N_DETAIL} Strategy Details ({l})", ""]
            for rank, strategy, filename in pr['detail_files']:
                md += [f"#### #{rank} {strategy['symbol']} {strategy['direction'].upper()}: {strategy['entry']} -> {strategy['exit']} ({strategy['hold_hours']}h)", "",
                       "| Metric | Value |", "|---|---|",
                       f"| Sharpe Ratio | {strategy['sharpe']:.3f} |", f"| Win Rate | {strategy['win_rate']:.1%} |",
                       f"| Avg Return | {strategy['avg_return']:.4%} |", f"| Trades | {strategy['count']} |", "",
                       f"![#{rank} {l}]({img}/{filename})", ""]
        md += [f"### TOP{TOP_N_TABLE} Table ({l})", "",
               "| Rank | Symbol | Dir | Entry | Exit | Hold | Trades | Win% | AvgReturn% | Sharpe |",
               "|---|---|---|---|---|---|---|---|---|---|"]
        for i, r in top100.iterrows():
            md.append(f"| {i + 1} | {r['symbol']} | {r['direction'].upper()} | {r['entry']} | {r['exit']} | {r['hold_hours']}h | "
                      f"{r['count']} | {r['win_rate']:.1%} | {r['avg_return']:.4%} | {r['sharpe']:.3f} |")
        md += ["", f"### Summary by Symbol ({l})", "",
               "| Symbol | Strategies in TOP100 | Best Sharpe | Best Direction |", "|---|---|---|---|"]
        for sym in top100['symbol'].unique():
            sd = top100[top100['symbol'] == sym]
            md.append(f"| {sym} | {len(sd)} | {sd.iloc[0]['sharpe']:.3f} | {sd.iloc[0]['direction'].upper()} |")
        md.append("")

    md += ["---", "*This report was auto-generated by the FX Anomaly Search System.*"]
    return '\n'.join(md)

# =============================================================================
# メイン
# =============================================================================

def parse_args():
    ap = argparse.ArgumentParser(description='FX anomaly auto-search (multi-period, fast)')
    ap.add_argument('--periods', default=','.join(p[0] for p in PERIODS), help='例: 1m,3m,6m')
    ap.add_argument('--symbols', default='', help='例: EURUSD,USDJPY（省略時は TARGET_SYMBOLS）')
    ap.add_argument('--skip-images', action='store_true', help='画像を生成しない（速度確認用）')
    return ap.parse_args()


def main():
    args = parse_args()
    wanted = [p.strip() for p in args.periods.split(',') if p.strip()]
    periods = [p for p in PERIODS if p[0] in wanted]
    if not periods:
        print(f"期間の指定が不正です: {args.periods}")
        sys.exit(1)
    targets = [s.strip() for s in args.symbols.split(',') if s.strip()] or TARGET_SYMBOLS

    t_start = time.time()
    connect_mt5()
    symbols = detect_actual_symbols(targets)
    if not symbols:
        print("通貨ペアが見つかりません")
        mt5.shutdown()
        sys.exit(1)

    # 1) データ取得: 最長期間を1回だけ取得
    max_days = max(p[2] for p in periods)
    print(f"\nデータ取得中（直近{max_days}日 / M5）...")
    t0 = time.time()
    full = {}
    for s in symbols:
        df = fetch_full(s, max_days)
        if df is None:
            print(f"  {s}: 取得失敗")
            continue
        full[s] = df
    print(f"データ取得完了: {len(full)}ペア, {sum(len(d) for d in full.values()):,}本, {time.time() - t0:.1f}s")
    mt5.shutdown()  # 以降はオフラインで計算できる

    os.makedirs(CSV_DIR, exist_ok=True)
    period_results = {}

    # 2) 期間ごとに探索
    for label, plabel, days in periods:
        print(f"\n=== {period_label_ja(label, days)} ===")
        t0 = time.time()
        bts, frames = {}, []
        for s, df_full in full.items():
            df = slice_period(df_full, days)
            if len(df) < 100:
                continue
            bt = FastBacktester(df)
            bts[s] = bt
            frames.append(bt.run_grid())
        all_df = pd.concat([f for f in frames if not f.empty], ignore_index=True) if frames else pd.DataFrame()
        ranked = rank_strategies(all_df)
        search_sec = time.time() - t0

        meta = {
            'label': label, 'plabel': plabel, 'days': days, 'n_symbols': len(bts),
            'date_min': min(bt.date_min for bt in bts.values()),
            'date_max': max(bt.date_max for bt in bts.values()),
            'bars_avg': int(np.mean([len(bt.time) for bt in bts.values()])),
            'trading_days': int(np.median([bt.trading_days for bt in bts.values()])),
            'min_trades': int(np.median([bt.min_trades for bt in bts.values()])),
            'n_tested': int(len(all_df)), 'n_profitable': int(len(ranked)), 'search_sec': search_sec,
        }
        print(f"  探索完了: {meta['n_tested']:,}戦略をテスト, 利益戦略 {meta['n_profitable']:,}件, {search_sec:.1f}s")
        if len(ranked):
            r = ranked.iloc[0]
            print(f"  #1: {r['symbol']} {r['direction'].upper()} {r['entry']}->{r['exit']} Sharpe={r['sharpe']:.3f} Win={r['win_rate']:.1%} n={r['count']}")

        csv_path = os.path.join(CSV_DIR, f'strategies_{label}.csv')
        all_df.sort_values('sharpe', ascending=False).to_csv(csv_path, index=False)
        print(f"  Saved: {os.path.relpath(csv_path, OUTPUT_DIR)}")

        detail_files = []
        if not args.skip_images and len(ranked):
            t1 = time.time()
            img_dir = os.path.join(IMG_ROOT, label)
            os.makedirs(img_dir, exist_ok=True)
            for f in os.listdir(img_dir):  # 古い画像を掃除
                if f.endswith('.png'):
                    os.remove(os.path.join(img_dir, f))
            print("  画像生成中...")
            save_top100_table(ranked, img_dir, plabel)
            save_overview_chart(ranked, img_dir, plabel)
            save_heatmap(ranked, img_dir, plabel)
            save_hold_hours_distribution(ranked, img_dir, plabel)
            save_winrate_vs_sharpe(ranked, img_dir, plabel)
            save_entry_time_distribution(ranked, img_dir, plabel)
            for i in range(min(TOP_N_DETAIL, len(ranked))):
                strategy = ranked.iloc[i]
                fn = save_detailed_chart(bts[strategy['symbol']], strategy, i + 1, img_dir, plabel)
                if fn:
                    detail_files.append((i + 1, strategy, fn))
            print(f"  画像生成完了: {time.time() - t1:.1f}s")

        period_results[label] = {'all': all_df, 'ranked': ranked, 'bts': bts, 'meta': meta,
                                 'detail_files': detail_files, 'images': not args.skip_images}

    # 3) レポート
    elapsed = time.time() - t_start
    report = build_report(period_results, periods, symbols, elapsed)
    md_path = os.path.join(OUTPUT_DIR, 'REPORT.md')
    with open(md_path, 'w', encoding='utf-8') as f:
        f.write(report)
    print(f"\nSaved: REPORT.md")
    print(f"合計時間: {elapsed:.1f}s")
    print("\nDone!")


if __name__ == '__main__':
    main()
