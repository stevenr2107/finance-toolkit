"""
Day 17 - professionelles backtesting framework

die drei fehler die 99 % aller backtests ruinieren:

1. keine transaktionskosten -> Strategie sieht 30 % besser aus als sie ist 
2. Kein walk-forward test -> overfitting auf historische daten 
3. kein slippage-modell -> in der realtität kaufst du nie zum schlusskors

heute fixt du alle drei
"""

import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from itertools import product
from dataclasses import dataclass, field
from typing import Callable 
import warnings
warnings.filterwarnings("ignore")


def load_data(ticker:str, period:str = "10y") -> pd.DataFrame:
    df = yf.download(ticker, period=period, 
    auto_adjust=True, progress=False)
    df.columns = df.columns.get_level_values(0)
    return df.dropna()

@dataclass
class TransactionsCosts:
    """
    Realistisches Kostenmodell - kein backtest oihne das 

    die meisten backtests scheitern in der realität weil 
    kosten ignoriert werden. Das hier macht es richtug.

    Retail Broker ( Interactive Brokers etc.):
        comission: 0.005% bis 0.1% pro trade 
        slippage: 0.05% bis 0.2% je nach liquidät 
        spread: bei liquiden aktien vernachlässigbar 

    Faustregel:
        wenn deine strategie nach kosten nicht mehr profitabel ust 
        war sie es vorher auch nicht wirklich 
    """

    comission_pct: float= 0.001 # 0.1% realistisch für retail 
    slippage_pct: float= 0.001 # 0.1% slippage
    min_comission: float=1.0 # Mindestgebühr in dollar 
    spread_pct: float = 0.0005 # 0.05% bid ask spread

    def total_cost(self,
                    trade_value: float, 
                    is_entry: bool=True) -> float:
        """
        Gesamtkosten eines Trades in dollar 
        entry und exit kosten beide - Roundtrip = 2x
        """
        commission = max(
            trade_value * self.comission_pct,
            self.min_comission
        )
        slippage = trade_value * self.slippage_pct
        spread = trade_value * self.spread_pct * 0.5
    
        return commission + slippage + spread
    
    def adjusted_entry(self, price: float) -> float:
        """ Realer Kaufpreis: höher als Schlusskurs"""
        return price * (1 + self.slippage_pct + self.spread_pct * 0.5)

    def adjusted_exit(self, price: float) -> float:
        """ Realer Verkaufspreis: niedriger als Schlusskurs"""
        return price * (1 - self.slippage_pct - self.spread_pct * 0.5)

def compute_rsi(close: pd.Series, window: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.ewm(com=window - 1, adjust=False).mean()
    avg_loss = loss.ewm(com=window - 1, adjust=False).mean()

    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))


# --- Generischer strategie runner ----
def run_strategy(df: pd.DataFrame,
                 signal_fn: Callable,
                  costs: TransactionsCosts,
                   capital: float = 10_000) -> dict:
    """
    Generischer backtest runner

    signal_fn gibt für jeden tag zurückk
        +1 -> Long 
        0 -> Cash
        -1 -> Short ( Falls strategie das unterstützt )

    Look-ahead Bias Prevention:
        Signal von tag x -> order auf open von tag x+1
        das ist die einig korrekte art zu backtesten 
    """
    close = df["Close"].squeeze()
    opens = df["Open"].squeeze()
    highs = df["High"].squeeze()
    lows = df["Low"].squeeze()

    # Signale berechnen - auf schlusskurs basis
    signals = signal_fn(df)

    # Shift: Signal gestern -> position heute 
    positions = signals.shift(1).fillna(0)

    equity = [capital]
    cash = capital
    shares = 0
    trade_log = []
    in_trade = False
    entry_px = 0.0
    entry_date = None 

    for i in range(1, len(df)):
        date = df.index[i]
        pos_today = positions.iloc[i]
        pos_yesterday = positions.iloc[i-1]
        open_px = float(opens.iloc[i])
        close_px = float(close.iloc[i])

        # --- Position wechseln? ---
        if pos_today != pos_yesterday:

            # Aus long raus 
            if pos_yesterday== 1 and shares > 0:
                exit_px = costs.adjusted_exit(close_px)
                proceeds = shares * exit_px
                fee = costs.total_cost(proceeds, is_entry = False)
                cash += proceeds - fee

                pnl = (exit_px - entry_px) * shares - fee
                pnl_pct = (exit_px / entry_px -1) * 100

                trade_log.append({
                    "entry_date": entry_date,
                    "exit_date": date,
                    "entry_price": round(entry_px,2),
                    "exit_price": round(exit_px,2),
                    "shares": shares,
                    "pnl": round(pnl,2),
                    "pnl_pct": round(pnl_pct,2),
                    "duration": (date - entry_date).days
                })
                shares = 0 
                in_trade = False

            # In Long rein 
            if pos_today == 1 and cash > 0:
                entry_px = costs.adjusted_entry(open_px)
                fee = costs.total_cost(cash, is_entry = True)
                shares = int((cash - fee) / entry_px)

                if shares > 0 :
                    cash -= shares * entry_px + fee
                    in_trade = True
                    entry_date = date
        
        # Equity: Cash + offene positionen 
        position_value = shares * close_px if in_trade else 0 
        equity.append(round(cash + position_value,2))

    # Letzten Trade schließne 
    if in_trade and shares > 0:
        exit_px = costs.adjusted_exit(float(close.iloc[-1]))
        proceeds = shares * exit_px
        fee = costs.total_cost(proceeds, is_entry = False)
        cash += proceeds - fee
        pnl = (exit_px - entry_px) * shares - fee
        trade_log.append({
            "entry_date": entry_date,
            "exit_date": df.index[-1],
            "entry_price": round(entry_px,2),
            "exit_price": round(exit_px,2),
            "shares": shares,
            "pnl": round(pnl,2),
            "pnl_pct": round((exit_px / entry_px -1) * 100,2),
            "duration": (df.index[-1] - entry_date).days,
        })
    equity_series = pd.Series(equity, index= df.index)
    trades_df = pd.DataFrame(trade_log)

    return {
        "equity": equity_series,
        "trades": trades_df,
        "capital": capital
    }

# --- Strategie signale ---
# jede funktion gibt eine signal.series zurück (0 oder 1)

def sma_crossover_signal(df: pd.DataFrame,
                         fast: int = 50,
                         slow: int = 200) -> pd.Series:
    
    close = df["Close"].squeeze()
    sma_fast = close.rolling(fast).mean()
    sma_slow = close.rolling(slow).mean()
    return (sma_fast > sma_slow).astype(int)


def rsi_reversion_signal(df: pd.DataFrame,
                         entry: int = 30,
                         exit_lvl: int = 65,
                         window: int=14) -> pd.Series:
    close = df["Close"].squeeze()
    rsi = compute_rsi(close, window)
    signal = pd.Series(0, index=df.index)

    position = 0 
    for i in range(len(df)):
        if rsi.iloc[i] < entry:
            position = 1
        elif rsi.iloc[i] > exit_lvl:
            position = 0
        signal.iloc[i] = position

    return signal

def buy_and_hold_signal(df: pd.DataFrame) -> pd.Series:
    """Immer investiert - Benchmark """
    return pd.Series(1, index=df.index)

# --- Performance Metriken ---
def compute_metrics(result: dict,
                    name: str="Strategie") -> dict:
    """
    Vollständige set an performance metriken 
    das ist was ein professionelles tearsheez enthält
    """
    equity = result["equity"]
    trades = result["trades"]
    capital = result["capital"]

    returns = equity.pct_change().dropna()
    years = len(equity) / 252
    total_ret = (equity.iloc[-1] / equity.iloc[0] -1) * 100
    cagr = ((equity.iloc[-1] / equity.iloc[0]) ** (1/max(years, 0.01)) - 1) * 100
    vol = returns.std() * np.sqrt(252) * 100
    sharpe = (returns.mean() / returns.std()) * np.sqrt (252) if returns.std() > 0 else 0

    #Drawdown 
    rolling_max = equity.cummax()
    dd_series = (equity - rolling_max) / rolling_max * 100
    max_dd = dd_series.min()
    calmar = cagr / abs(max_dd) if max_dd != 0 else 0

    # Sortino - wie sharpe aber nur downside volatilität 
    downside = returns[returns < 0 ].std() * np.sqrt(252)
    sortino = (returns.mean() * 252) / downside if downside > 0 else 0

    # Trade stats 
    if not trades.empty and "pnl" in trades.columns:
        completed = trades.dropna(subset=["pnl"])
        wins = completed[completed["pnl"] > 0]
        losses = completed[completed["pnl"] <=0]

        win_rate = len(wins) / len(completed) * 100 if len(completed) > 0 else 0
        avg_win = wins["pnl_pct"].mean() if not wins.empty else 0 
        avg_loss = losses["pnl_pct"].mean() if not losses.empty else 0
        profit_factor = (
            wins["pnl"].sum() / abs(losses["pnl"].sum())
            if not losses.empty and losses["pnl"].sum() != 0 else 0
        )
        avg_duration = completed["duration"].mean() if "duration" in completed else 0
        n_trades = len(completed)
    else: 
        win_rate = avg_win = avg_loss = 0
        profit_factor = avg_duration =  0
        n_trades = 1 # Buy & Hold

    return {
        "Name": name,
        "Total Return (%)": round(total_ret, 2),
        "CAGR (%)": round(cagr, 2),
        "Volatilität (%)": round(vol, 2),
        "Sharpe" : round(sharpe, 2),
        "Sortino" : round(sortino, 2),
        "Calmar" : round(calmar, 2),
        "Max DD (%)": round(max_dd, 2),
        "Win Rate (%)": round(win_rate, 1),
        "Avg Win": round(avg_win, 2),
        "Avg Loss": round(avg_loss, 2),
        "Profit Factor": round(profit_factor, 2),
        "Trades": n_trades,
        "Avg Duration": round(avg_duration, 1)
    }

def walk_forward_test(df: pd.DataFrame,
                      signal_fn: Callable,
                      costs: TransactionsCosts,
                      train_years: int = 2,
                      test_years: int = 1,
                      capital: float = 10_000) -> dict:
    """
    Walk forward test - der standard test für robuste strategien 
    
    logik 
        train: 2 jahre -> optimierte parameter
        test: 1jahr -> trade mit optimierten parametern 
        schiebe das fenster um 1 jahr weiter -> repeat 

    warum das so wichtig ist:
        in sample backtest sieht immmer gut aus 
        walk-forward zeigt ob die strategie wirklich funktioniert 
        oder nur auf historischen daten gefitted wurde 

        wenn out of sample performance >> schlechter als in sample 
        -> overfitting. Strategie ist nicht handelbar 

        wenn beide ähnlich -> robuste edge vorhanden 
    """
    train_days = train_years * 252
    test_days = test_years * 252

    if len(df) < train_days + test_days:
        print ( "Nicht genug daten fpr walk forward test")
        return {}

    periods = []
    all_equity = []
    start = 0

    while start + train_days + test_days <= len(df):
        train_df = df.iloc[start : start + train_days]
        test_df = df.iloc[start + train_days : start + train_days + test_days]

        # In sample 
        train_result = run_strategy(train_df, signal_fn, costs, capital)
        train_metrics = compute_metrics(train_result, "Train")

        # Out of sample test 
        test_result = run_strategy(test_df, signal_fn, costs, capital)
        test_metrics = compute_metrics(test_result, "Test")

        period_info = {
            "train_start": train_df.index[0].strftime("%Y-%m"),
            "train_end": train_df.index[-1].strftime("%Y-%m"),
            "test_start": test_df.index[0].strftime("%Y-%m"),
            "test_end": test_df.index[-1].strftime("%Y-%m"),
            "train_sharpe": train_metrics["Sharpe"],
            "test_sharpe": test_metrics["Sharpe"],
            "train_cagr": train_metrics["CAGR (%)"],
            "test_cagr": test_metrics["CAGR (%)"],
            "train_dd": train_metrics["Max DD (%)"],
            "test_dd": test_metrics["Max DD (%)"],
        }
        periods.append(period_info)

        # Equity aus test periode sammeln 
        all_equity.append(test_result["equity"])

        start += test_days

    # Gesamte out of sample equity curve 
    if all_equity:
        combined = pd.concat(all_equity)
        # Normalisieren: jede periode startet wo die letzte aufgehört hat 
        combined_normalized = combined / combined.iloc[0] * capital
    else: combined_normalized = pd.Series()

    periods_df = pd.DataFrame(periods)

    # Effizienz score: wie viel in sampke performance überlebt out of sample ?
    if not periods_df.empty:
        efficiency = (
            periods_df["test_sharpe"].mean() /
            periods_df["train_sharpe"].mean()
            if periods_df["train_sharpe"].mean() != 0 else 0
        )
    else: 
        efficiency = 0

    return {
        "periods": periods_df,
        "equity": combined_normalized,
        "efficiency": round(efficiency,3)
    }

def print_walk_forward_results(wf_result: dict) -> None: # wf steht einfach für walkforward 
    """GIbt walk forward ergebnusse strukturiert aus"""
    if not wf_result:
        return 
    
    periods = wf_result["periods"]
    efficiency = wf_result["efficiency"]

    print("\n" + "="*65)
    print("  WALK-FORWARD TEST ERGEBNISSE")
    print("="*65)
    print(f"\n  {'Periode':<12} {'Train CAGR':>11} "
          f"{'Test CAGR':>10} {'Train Sharpe':>13} "
          f"{'Test Sharpe':>12}")
    print("  " + "-"*58)

    for _, row in periods.iterrows():
        print(f" {row['test_start']:<12}"
              f" {row['train_cagr']:>9.1f}%"
              f" {row['test_cagr']:>9.1f}%"
              f" {row['train_sharpe']:>12.2f}"
              f" {row['test_sharpe']:>11.2f}")
        
    print(" " + "-" * 58)
    avg_test_sharpe = periods["test_sharpe"].mean()
    avg_test_cagr = periods["test_cagr"].mean()

    print(f" {'Durchschnitt':<12}"
          f" {'':>10}"
          f" {avg_test_cagr:9.1f}%"
          f" {'':>12}"
          f" {avg_test_sharpe:11.2f}")
    
    print(f"\n Effizienz-Score: {efficiency:.3f}")
    if efficiency > 0.7:
        print("  Strategie ist robust. Strategie überlebt out of sample gut ")
    elif efficiency >0.4:
        print(" Moderat robust. Etwas overfitting vorhanden. ")
    else:
        print("  Schwache robustheit. Strategie ist stark overfitted.")
    print("="*65)

#--- Grid Search Optimierung 

def grid_search(df: pd.DataFrame,
                param_grid: dict,
                signal_fn: Callable,
                costs: TransactionsCosts,
                metric: str = "Sharpe",
                capital: float = 10_000) -> pd.DataFrame:
    """
    Grid search über strategie parameter 

    wichtig: Grid search immer NUR auf Train-Daten.
    Niemals auf dem vollen datensatz optimieren 
    Das ist Overfitting per definition 

    paam_grid Beispiel:
        {
            "fast": [20, 50],
            "slow": [100, 200],
        }
    -> Testet alle 4 kombinationen 

    zurückgegeben wird ein dataframe aller kombinationen 
    sortiert nach der gewählten metric.
    """
    param_names = list(param_grid.keys())
    param_values = list(param_grid.values())
    combinations = list(product(*param_values))

    print(f"Grid Search : {len(combinations)} Kombinationen...")
    results = []

    for combo in combinations:
        params = dict(zip(param_names, combo)) # zip führt mehrere listen zusammen

        # signal funktion mit aktuellen parametern 
        def signal_with_params(df, p=params):
            return signal_fn(df, **p)
        
        try: 
            result = run_strategy(df, signal_with_params, costs, capital)
            metrics = compute_metrics(result)

            row = {**params, **{
                "Sharpe": metrics["Sharpe"],
                "CAGR (%)": metrics["CAGR (%)"],
                "Max DD (%)": metrics["Max DD (%)"],
                "Calmar" : metrics["Calmar"],
                "Win Rate (%)": metrics["Win Rate (%)"],
                "Trades": metrics["Trades"],
            }}
            results.append(row)

        except Exception:
            pass

    results_df = pd.DataFrame(results).sort_values(metric, ascending=False).reset_index(drop=True)

    return results_df

def plot_grid_search_heatmap(grid_results: pd.DataFrame,
                             x_param: str,
                             y_param: str,
                             metric: str = "Sharpe") -> None :
    """
    Visulaisiert grid searcg ergebnisse als heatmap 
    zeigt sofort welche parameter-kombinition am besten ist 
    """ 
    pivot = grid_results.pivot(
        index = y_param, columns=x_param, values=metric
    )

    fig = go.Figure(go.Heatmap(
        z=pivot.values.round(3),
        x=[str(c) for c in pivot.columns],
        y=[str(i) for i in pivot.index],
        colorscale=[
            [0.0, "#dc2626"],
            [0.4, "#fca5a5"],
            [0.5, "#f9fafb"],
            [0.6, "#86efac"],
            [1.0, "#16a34a"]
        ],
        text = [[f"{v:.2f}" for v in row]
                for row in pivot.values],
                texttemplate="%{text}",
                textfont=dict(size=12),
                showscale=True,
                zmid= pivot.values.mean()
    ))

    fig.update_layout(
        title=f"Grid Search - {metric} nach {x_param} / {y_param}",
        xaxis_title=x_param,
        yaxis_title=y_param,
        template = "plotly_white",
        height = 450,
        margin=dict(l=0, r=0, t=40, b=0)
    )

    fig.show()

#--- Equity curve visualisierung 

def plot_walk_forward_equity(wf_result: dict,
                             bah_equity: pd.Series,
                             ticker: str) -> None:
    """
    Vergleicht walk forward out of sample equity mit buy &Hold
    Das ist das ehrlichste Chart bis jetzt 
    """
    if not wf_result or wf_result["equity"].empty:
        print("Keine walk forward equity vorhanden ")
        return 
    
    wf_equity = wf_result["equity"]
    periods = wf_result["periods"]

    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        row_heights=[0.65,0.35],
        subplot_titles=[f" {ticker} - Walk forward out of sample vs buy&Hold ",
                        "Rollendes Sharpe Ratio (126 Tage)"]
    )
    # Walk forward equity 
    fig.add_trace(go.Scatter(
        x = wf_equity.index,
        y = wf_equity.round(2),
        name="WF Out of Sample",
        line=dict(color="#2563eb", width= 2)
    ), row=1, col=1)

    # Buy&Hold zum vergleich auf wf zeitraum trimmen 
    bah_trimmed = bah_equity.loc[
        wf_equity.index[0]:wf_equity.index[-1]
    ]
    bah_norm = (bah_trimmed / bah_trimmed.iloc[0] * wf_equity.iloc[0])

    fig.add_trace (go.Scatter(
        x = bah_norm.index,
        y = bah_norm.round(2),
        name="Buy&Hold",
        line=dict(color="#94a3b8", width= 1.5, dash = "dot")
    ), row=1, col=1)


    # Test perioden als hintergrund bänder 
    for i, row in periods.iterrows():
        color = "rgba(37,99,235,0.04)" if i % 2 == 0 \
            else "rgba(37,99,235,0.08)"
        fig.add_vrect(
            x0=row["test_start"], x1=row["test_end"],
            fillcolor=color,
            layer="below", line_width=0,
            row=1, col=1
        )

    # Rollendes Sharpe 
    wf_returns = wf_equity.pct_change().dropna()
    roll_sharpe = (
        wf_returns.rolling(126).apply(lambda x: (x.mean() / x.std()) * np.sqrt(252) if x.std() > 0 else 0)
    ).round(3)


    fig.add_trace(go.Scatter(
        x = roll_sharpe.index,
        y = roll_sharpe,
        name="Rolling Sharpe",
        line=dict(color="#2563eb", width= 1.5)
    ), row=2, col=1)

    fig.add_hline(
        y=1.0, line_dash = "dash",
        line_color="#16a34a",
        opacity = 0.5, row=2, col=1
    )
    fig.add_hline(
        y=0.0, line_dash="dot",
        line_color="#ef4444",
        opacity=0.5, row=2, col=1
    )

    fig.update_layout(
        height=620,
        template="plotly_white",
        hovermode="x unified",
        legend=dict(orientation="h", y=1.02),
        margin=dict(l=0, r=0, t=50, b=0)
    )

    fig.update_yaxes(title_text="Kapital ($)", row=1, col=1)
    fig.update_yaxes(title_text="Sharpe",      row=2, col=1)

    fig.show()

if __name__ =="__main__":

    TICKER = "HIMS"
    CAPITAL = 10_000
    costs = TransactionsCosts()

    print(f"Professionelles Backtesting: {TICKER}")
    print("="*55)

    df= load_data(TICKER, "10y")
    print(f"Daten: {len(df)} Handelstage ({TICKER})")

    #--- Einfacher Vergleich mit vs. ohne Kosten ---
    print("\n--- Kosteneinfluss auf SMA Crossover ---")

    def sma_50_200(df):
        return sma_crossover_signal(df, fast=50, slow=200)

    # Mit Kosten 
    result_costs = run_strategy(df, sma_50_200, costs, CAPITAL)
    metrics_costs = compute_metrics(result_costs, "SMA mit Kosten")

    # Ohne Kosten 
    zero_costs = TransactionsCosts(0.0, 0.0, 0.0, 0.0)
    result_free = run_strategy(df, sma_50_200, zero_costs, CAPITAL)
    metrics_free = compute_metrics(result_free, "SMA ohne Kosten")

    # Buy & Hold 
    bah_result = run_strategy(df, buy_and_hold_signal, zero_costs, CAPITAL)
    metrics_bah = compute_metrics(bah_result, "Buy & Hold")

    summary = pd.DataFrame([metrics_free, metrics_costs, metrics_bah])
    cols = ["Name", "Total Return (%)", "CAGR (%)",
               "Sharpe", "Max DD (%)", "Trades"]
    print(summary[cols].to_string(index=False))

    print("\n→ Der Unterschied zeigt wie viel Kosten wirklich fressen.")

    # --- Grid Search (auf ersten 5 Jahren = Train) ---
    print("\n--- Grid Search: SMA Parameter (Train: erste 5 Jahre) ---")

    train_df = df.iloc[:252*5]

    param_grid = {
        "fast": [10,20, 50],
        "slow": [100, 150, 200]
    }

    grid_results = grid_search(
        df=train_df,
        param_grid=param_grid,
        signal_fn=sma_crossover_signal,
        costs=costs,
        metric = "Sharpe",
        capital=CAPITAL
    )

    print("\nTop 5 Parameter Kombinationen (Train):")
    print(grid_results.head(5)[
        ["fast", "slow", "Sharpe",
         "CAGR (%)", "Max DD (%)", "Trades"]
    ].to_string(index=False))

    # Heatmap der Grid Search 
    plot_grid_search_heatmap(
        grid_results,
        x_param = "fast",
        y_param = "slow",
        metric = "Sharpe"
    )

    # Beste Parameter
    best = grid_results.iloc[0]
    best_fast = int(best["fast"])
    best_slow = int(best["slow"])
    print(f"\nBeste Oarameter (Train): fast={best_fast}, slow={best_slow}")

    #--- Walk Forward Test ---
    print("\n--- Walk Forward Test (2y Train / 1y Test) ---")

    def best_sma_signal(df):
        return sma_crossover_signal(df, fast=best_fast, slow=best_slow)
    
    wf_result = walk_forward_test (
        df = df,
        signal_fn = best_sma_signal,
        costs = costs,
        train_years = 2,
        test_years = 1,
        capital = CAPITAL
    )

    print_walk_forward_results(wf_result)

    # Walk-Forward Chart 
    plot_walk_forward_equity(
        wf_result=wf_result,
        bah_equity = bah_result["equity"],
        ticker = TICKER
    )

    # --- RSI Walk-Forward ---
    print("\n--- Walk Forward Test: RSI Mean Reversion ---")

    wf_rsi = walk_forward_test(
        df = df,
        signal_fn = rsi_reversion_signal,
        costs = costs,
        train_years = 2,
        test_years = 1,
        capital = CAPITAL
    )

    print_walk_forward_results(wf_rsi)

    # Effizienz vergleich 
    print("\n" + "="*45)
    print("ROBUSTHEIT VERGLEICH")
    print("="*45)
    print(f" SMA Crossover Effizienz: {wf_result['efficiency']:.3f}")
    print(f" RSI Reversion Effizienz: {wf_rsi['efficiency']:.3f}")
    print(" (> 0.7 = robust | 0.4-0.7 = moderat | < 0.4 = overfittet)")
