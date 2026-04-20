"""
Day 13 Momentum Strategie + Strategy comparison 
momentum: kaufe was stark war, verkaufe was schwach wird
kombiniert: mit rsi filter um überkauft situationen zu vermeiden
final: Alle strategien in einem Verglecihs dashboard 
"""

import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from dataclasses import dataclass
from typing import Optional 
import warnings
warnings.filterwarnings("ignore") # Ignoriert Warnungen, damit die Ausgabe sauber bleibt

# Wiederverwendung aus Tag 12
from strategy_rsi import (
    load_data, compute_rsi, fixed_fractional_size
    , generate_tearsheet, StrategyConfig, Trade
)

@dataclass
class MomentumConfig:
    """ Momentum Berechnung """
    # Momentum Berechnung 
    momentum_window: int = 126 # 6 Monate Momentum 
    signal_window: int = 20 # Glätttung des Signals 

    # Filter
    rsi_max: int = 70 # Nicht kaufen wenn überkauft 
    sma_filter: int = 200 # Nur long wenn über 200 sma
    min_momentum: float = 0.05 # mindestens +5% in 6M

    # Risiko Management
    stop_loss_pct: float = 0.07 # 7% stop loss
    take_profit_pct: float = 0.2 # 20% take profit
    max_hold_days: int = 60 # Maximal 60 Tage halten
    
    # Sizing
    risk_per_trade: float = 0.02 # 2% Risiko pro Trade
    max_position: float = 0.25

    # Kosten
    commission: float=0.001
    slippage: float = 0.001

    initial_capital: float = 10_000

# --- Momentum indikatoren ---
def compute_momentum_indicators (df: pd.DataFrame,
                      cfg: MomentumConfig) -> pd.DataFrame:
    """
    Berechnet alle Momentum relevanten indikatoren 

    Momentum Score: 
    Nicht einfach 6m return - das ist zu ruckartig 
    neuere perioden bekommen mehr gewicht 

    idee aus der literatur:
        Jegadeesh & Titman (1993): 12-1 Momentum
        Letzte 12 Monate minus letzten Monat (Reversal)
    """ 
    close = df["Close"].squeeze()

    # Returns verschiedener Perioden
    df["ret_1m"]  = close.pct_change(21)
    df["ret_3m"]  = close.pct_change(63)
    df["ret_6m"]  = close.pct_change(126)
    df["ret_12m"] = close.pct_change(252)

    # Jegadeesh-Titman: 12M Return minus letzter Monat 
    df["jt_momentum"] = df["ret_12m"] - df["ret_1m"]

    # Gewichteter Momentum Score 
    df["momentum_score"] = (
        df["ret_1m"] * 0.2 +
        df["ret_3m"] * 0.3 +
        df["ret_6m"] * 0.5
    )

    # Geglättetes Signal 
    df["momentum_signal"] = (
        df["momentum_score"].rolling(cfg.signal_window).mean()
    )

    # Trend filter 
    df["sma_200"] = close.rolling(200).mean()
    df["above_sma200"] = close > df["sma_200"]

    # RSI 
    df["RSI"] =compute_rsi(close)

    # Volatilität für sizing 
    df["volatility"] = (
        close.pct_change().rolling(21).std() * np.sqrt(252)
    )
    return df

# --- Momentum Backtest ---

def run_momentum_backtest(ticker: str, cfg: MomentumConfig) -> dict:
    """
    Momentum strategie Backtest 
    entry: momentum signal > min_momentum 
    und über sma 200
    und rsi < rsi_max ( nicht überkauft )


    Exit: momentum_signal dreht negativ 
    oder stop loss / take profit 
    oder rsi > 80 (extrem überkauft)
    """
    df = load_data(ticker)
    if len(df) < 260:
        return{}
    
    df = compute_momentum_indicators(df, cfg)
    df = df.dropna()

    close = df["Close"].squeeze()
    opens = df["Open"].squeeze()
    high = df["High"].squeeze()
    low = df["Low"].squeeze()

    capital = cfg.initial_capital
    equity_curve = [capital]
    dates = [df.index[0]]
    trades = []
    current_trade = None 

    for i in range(1, len(df)):
        date = df.index[i]
        prev = df.iloc[i -1] # Signale von gestern 
        open_price = float(opens.iloc[i])
        high_price = float(high.iloc[i])
        low_price = float(low.iloc[i])
        close_price = float(close.iloc[i])

        # --- Entry Signal ( alle bedingungen müssen erfüllt sein) ---
        entry_signal = (
            prev["momentum_signal"] > cfg.min_momentum and 
            prev["above_sma200"] and 
            prev["RSI"] < cfg.rsi_max
        )
        
        # --- Exit Signal ---
        exit_signal = (
            prev["momentum_signal"] < 0 or prev["RSI"] > 80
        )

        # --- Offene Positionen managen ---
        if current_trade is not None:
            days_held = (date - current_trade.entry_date).days
            exit_price = None
            exit_reason = None

            if low_price <= current_trade.stop_loss:
                exit_price = current_trade.stop_loss
                exit_reason = "Stop Loss"
            elif high_price >= current_trade.take_profit:
                exit_price = current_trade.take_profit
                exit_reason = "Take Profit"
            elif exit_signal:
                exit_price = open_price
                exit_reason = "Momentum Exit"
            elif days_held >= cfg.max_hold_days:
                exit_price = open_price
                exit_reason = "Max Hold"
            
            if exit_price is not None:
                proceeds= exit_price * current_trade.shares * (1- cfg.commission)
                entry_cost = current_trade.entry_price * current_trade.shares * (1 + cfg.commission)
                pnl = proceeds - entry_cost
                pnl_pct = (pnl/entry_cost) * 100

                capital += proceeds

                current_trade.exit_date = date
                current_trade.exit_price = round(exit_price, 2)
                current_trade.exit_reason = exit_reason 
                current_trade.pnl = round(pnl, 2)
                current_trade.pnl_pct = round(pnl_pct, 2)
                current_trade.duration = days_held

                trades.append(current_trade)
                current_trade = None

        # --- Neues entry ---
        if current_trade is None and entry_signal:
            entry_price = open_price * (1+cfg.slippage)
            stop_loss = entry_price * (1 - cfg.stop_loss_pct)
            take_profit = entry_price * (1 + cfg.take_profit_pct)

            shares = fixed_fractional_size(
                capital = capital ,
                entry_price= entry_price,
                stop_loss_price= stop_loss,
                risk_per_trade= cfg.risk_per_trade,
                max_position= cfg.max_position
            )

            if shares > 0:
                cost = entry_price * shares * (1+cfg.commission)
                if cost <= capital:
                    capital -= cost
                    current_trade = Trade(
                        ticker = ticker,
                        entry_date = date,
                        entry_price = round(entry_price, 2),
                        shares = shares,
                        stop_loss = round(stop_loss, 2),
                        take_profit = round(take_profit, 2)
                    )
        # Equity Update 
        open_value =(float(close.iloc[i]) * current_trade.shares
                     if current_trade else 0)
        equity_curve.append(round(capital + open_value, 2))
        dates.append(date)

    # Letzten Trade schließen 
    if current_trade:
        last_price = float(close.iloc[-1])
        pnl = (last_price - current_trade.entry_price) * current_trade.shares
        current_trade.exit_date = df.index[-1]
        current_trade.exit_price = last_price
        current_trade.exit_reason = "End of Data"
        current_trade.pnl = round(pnl, 2)
        current_trade.pnl_pct = round(pnl / (current_trade.entry_price * current_trade.shares) * 100, 2)
        current_trade.duration = (df.index[-1] - current_trade.entry_date).days
        trades.append(current_trade)

    equity = pd.Series(equity_curve, index=dates)
    trades_df = pd.DataFrame([t.__dict__ for t in trades]) if trades else pd.DataFrame()

    return{
        "ticker": ticker,
        "equity": equity,
        "trades": trades_df,
        "price": close,
        "df": df,
        "config": cfg
    }

def run_buy_and_hold(ticker: str,
                     capital: float = 10_000) -> dict:
     """
     Buy & hold benchmarj 
     jede strategie muss sich hieran messen 
     """
     df = load_data(ticker)
     close = df["Close"].squeeze()

     equity = (close / close.iloc[0]) * capital
     returns = close.pct_change().dropna()

     return {
         "ticker": ticker, 
         "equity": equity,
         "trades": pd.DataFrame(),
         "price": close,
     }


def compare_strategies(results: dict) -> pd.DataFrame:
    """
    Vergleicht alle strategien anhand derselben 
    results = {"Strategy Name": result_dict}
    """
    rows = []

    for name, result in results.items():
        equity = result["equity"]
        returns = equity.pct_change().dropna()

        years = len(equity) / 252
        total_ret = (equity.iloc[-1] / equity.iloc[0] -1) * 100
        cagr_val = ((equity.iloc[-1] / equity.iloc[0]) ** (1/max(years, 0.01)) - 1) * 100
        vol = returns.std() * np.sqrt(252) * 100
        sharpe = (returns.mean() / returns.std()) * np.sqrt (252)

        rolling_max = equity.cummax()
        max_dd = ((equity - rolling_max) / rolling_max).min() * 100
        calmar = cagr_val / abs(max_dd) if max_dd != 0 else 0 

        trades_df = result.get("trades", pd.DataFrame())
        if not trades_df.empty and "pnl" in trades_df.columns:
            completed = trades_df.dropna(subset=["pnl"])
            wins = completed[completed["pnl"] > 0]
            losses = completed[completed["pnl"] <=0]
            win_rate = len(wins) / len(completed) * 100 if len(completed) > 0 else 0
            pf = wins["pnl"].sum() / abs(losses["pnl"].sum()) if not losses.empty and losses["pnl"].sum() != 0 else 0
            n_trades = len(completed)
        else: 
            win_rate = pf = 0
            n_trades = 1

        rows.append({
            "Strategie":         name,
            "Total Return (%)":  round(total_ret, 2),
            "CAGR (%)":          round(cagr_val, 2),
            "Volatilität (%)":   round(vol, 2),
            "Sharpe Ratio":      round(sharpe, 2),
            "Calmar Ratio":      round(calmar, 2),
            "Max Drawdown (%)":  round(max_dd, 2),
            "Win Rate (%)":      round(win_rate, 1),
            "Profit Factor":     round(pf, 2),
            "Trades":            n_trades,
        })

    return pd.DataFrame(rows).sort_values(
        "Sharpe Ratio", ascending=False
    ).reset_index(drop = True)

def plot_strategy_comparison( results: dict,
                             ticker: str)-> None:
    """
    Das finale Dashboard - alle strategien auf einem blivk 

    panel1: normalisierte equity curve 
    panel 2: Rollender Sharpe (252 Tage)
    Panel 3: Drawdown vergleich 
    Panel 4: Monatliche Return heatmap ( beste strategie)
    """
    colors = {
        "Buy & Hold":     "#94a3b8",
        "RSI Reversion":  "#2563eb",
        "Momentum":       "#16a34a",
        "SMA Crossover":  "#f59e0b",
    }

    fig = make_subplots(
        rows = 3, cols = 1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        row_heights= [0.45,0.3,0.25],
        subplot_titles = [
            f"{ticker} - Strategy comparison ( normalisiert auf 10k $)",
            "Rolling sharpe Ratio ( 252 Tage)", 
            "Drawdown (%)"
        ]
    )

    for name, result in results.items():
        equity = result["equity"]
        color = colors.get(name, "#8b5cf6")
        lw = 1.5 if name == "Buy & Hold" else 2 
        dash = "dot" if name == "Buy & Hold" else "solid"

        returns = equity.pct_change().dropna()

        # --- Panel 1 Equity ---
        fig.add_trace(go.Scatter(
            x = equity.index,
            y=equity.round(2),
            name=name,
            line=dict(color=color, width=lw, dash=dash)
        ),row = 1, col=1)

        # --- Panel 2 Rolling sharpe ---
        roll_sharpe = (
            returns.rolling(252).apply(lambda x: (x.mean() / x.std()) * np.sqrt(252)
                                       if x.std() > 0 else 0)
        ).round(3)

        fig.add_trace(go.Scatter(
            x=roll_sharpe.index,
            y=roll_sharpe,
            name=f"{name} Sharpe",
            line=dict(color=color, width=1.5, dash=dash),
            showlegend=False
        ), row=2, col=1)


        # --- Panel 3 Drawdown ---
        rolling_max = equity.cummax()
        dd = ((equity - rolling_max) / rolling_max * 100).round(2)

        fig.add_trace(go.Scatter(
            x=dd.index,
            y=dd,
            name=f"{name} DD",
            line=dict(color=color, width=1, dash=dash),
            fill="tozeroy" if name != "Buy & Hold" else None,
            fillcolor=f"rgba{tuple(list(int(color.lstrip('#')[i:i+2], 16) for i in (0,2,4)) + [0.08])}",
            showlegend=False
        ), row=3, col=1)


        # Referenzlinien 
        fig.add_hline( y= 0, line_dash="dot",
                      line_color="#1e293b", opacity=0.3, row=2, col=1)
        fig.add_hline(y=1.0, line_dash="dash",
                      line_color="#16a34a", opacity=0.4, row=2, col=1)

        fig.update_layout(
            height=750,
            template="plotly_white",
            hovermode="x unified",
            legend=dict(orientation="h", y=1.02),
            margin=dict(l=0, r=0, t=50, b=0)
        ) 


        fig.update_yaxes(title_text="Kapital($)", row=1, col=1)
        fig.update_yaxes(title_text="Sharpe", row=2, col=1)
        fig.update_yaxes(title_text="DD (%)", row=3, col=1)

        fig.show()

def plot_comparison_bars(summary_df: pd.DataFrame) -> None:
    """
    Balken vergleich aller kennzahlen 
    sofort klar welche strategie wo besser ist
    """
    metrics = [
        ("CAGR (%)", "CAGR", "#2563eb"),
        ("Sharpe Ratio", "Sharpe", "#16a34a"),
        ("Max Drawdown (%)", "Max DD", "#ef4444"),
        ("Win Rate (%)", "Win Rate", "#f59e0b"),
    ]
    
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=[m[1] for m in metrics],
        vertical_spacing = 0.15,
        horizontal_spacing = 0.1
    )

    positions = [(1,1), (1,2), (2,1), (2,2)]

    for (col,label, color), (row, c) in zip(metrics, positions):
        vals = summary_df[col]

        bar_colors = []
        for v in vals:
            if col == "Max Drawdown (%)":
                bar_colors.append("#ef4444" if v < -10 else "#f59e0b")
            else:
                bar_colors.append(color)

        fig.add_trace(go.Bar(
            x=summary_df["Strategie"],
            y=vals,
            marker_color=bar_colors,
            text = [f"{v:.2f}" for v in vals],
            textposition="outside",
            name=label,
            showlegend=False
        ), row=row, col=c)

    fig.update_layout(
        height = 550,
        template = "plotly_white",
        margin = dict(l=0, r=0, t=50, b=0)
    )
    fig.show()

def plot_monthly_heatmap(result: dict, name:str) -> None:
    """
    Monatliche returns der besten strategie als heatmap 
    das ist das was ein hedge fund report zeigt 
    """
    equity = result["equity"]
    returns = equity.pct_change().dropna()

    monthly = returns.resample("ME").apply(
        lambda x: (1 + x).prod() - 1
    ) * 100

    monthly_df = monthly.to_frame("Return")
    monthly_df["Jahr"] = monthly_df.index.year
    monthly_df["Monat"] = monthly_df.index.month

    pivot = monthly_df.pivot(
        index="Jahr", columns="Monat", values="Return"
    )
    pivot.columns = ["Jan","Feb","Mär","Apr","Mai","Jun",
                     "Jul","Aug","Sep","Okt","Nov","Dez"]
    
    # Jares Return hinzufügen 
    pivot["Jahr"] = pivot.sum(axis=1).round(1)

    fig = go.Figure(data=go.Heatmap(
        z=pivot.values.round(1),
        x=pivot.columns.tolist(),
        y=pivot.index.tolist(),
        colorscale=[
            [0.0,  "#dc2626"],
            [0.4,  "#fca5a5"],
            [0.5,  "#f9fafb"],
            [0.6,  "#86efac"],
            [1.0,  "#16a34a"]
        ],
        text = [[f"{v:.1f}%" if not np.isnan(v) else "" for v in row] for row in pivot.values],
        texttemplate="%{text}",
        textfont=dict(size=10),
        showscale=True, zmid=0
    ))

    fig.update_layout(
        title = f"Monatliche Returns - {name}",
        template = "plotly_white",
        height = max(250, len(pivot) * 45 + 100),
        margin = dict(l=0, r=0, t=40, b=0)
    )

    fig.show()

# --- SMA Corssover wiederverwenden ---

def run_sma_crossover(ticker: str,
                      fast: int=50,
                      slow: int = 200,
                      capital: float = 10_000) -> dict:
    """
    SMA crossover aus tag 8  hier direkt eingebaut 
    damit der vergleich komplett ist 
    """

    df = load_data(ticker)
    close = df["Close"].squeeze()

    df["SMA_fast"] = close.rolling(fast).mean()
    df["SMA_slow"] = close.rolling(slow).mean()
    df["signal"] = np.where(df["SMA_fast"] > df["SMA_slow"], 1, 0)
    df["position"] = df["signal"].shift(1).fillna(0)

    df["market_ret"] = close.pct_change()
    df["strategy_ret"] = df["market_ret"] * df["position"]
    df["trade"] = df["position"].diff().abs()
    df["strategy_ret"] -= df["trade"] * 0.001

    equity = (1+ df["strategy_ret"].fillna(0)).cumprod() * capital
    trades = int(df["trade"].sum())

    return {
        "ticker": ticker,
        "equity": equity,
        "trades": pd.DataFrame({"pnl": [0] * trades}),
        "price": close
    }


# --- MAIN alles zusammen ---

if __name__ == "__main__":
    TICKER = "AAPL"
    CAPITAL = 10_000

    print(f" Strategy Comparison === {TICKER} ===")
    print("="*50)

    # --- Alle strategien laufen lassen ---
    print("Running Buy & Hold")
    bah = run_buy_and_hold(TICKER, CAPITAL)

    print("Running SMA Crossover")
    sma = run_sma_crossover(TICKER, CAPITAL)

    print("Running RSI Reversion")
    from strategy_rsi import run_backtest as rsi_backtest
    rsi_cfg = StrategyConfig(initial_capital=CAPITAL)
    rsi_result = rsi_backtest(TICKER, rsi_cfg)

    print("Running Momentum")
    mom_cfg = MomentumConfig(initial_capital=CAPITAL)
    mom_result = run_momentum_backtest(TICKER, mom_cfg)

    # --- Zusammenführen ---

    results = {
        "Buy & Hold": bah,
        "SMA Crossover": sma,
        "RSI Reversion": rsi_result,
        "Momentum": mom_result
    }

    #--- Vergleichstabelle ---
    summary = compare_strategies(results)

    print("\nSTRATEGY COMPARISON SUMMARY")
    print("="*80)
    print(summary.to_string(index=False))
    print("="*80)

    # --- Charts ---
    plot_strategy_comparison(results, TICKER)
    plot_comparison_bars(summary)

    # beste strategie + Heatmap 
    best_name = summary.iloc[0]["Strategie"]
    best_result = results[best_name]
    print(f"\nBeste Strategie: {best_name}")
    plot_monthly_heatmap(best_result, best_name)


    # --- Multi Ticker Test ---
    print ("\n---Multi Ticker Momentum Test ---")
    tickers = ["AAPL", "MSFT", "NVDA", "JPM", "XOM", "SPY"]

    momentum_results = []
    for t in tickers:
        r = run_momentum_backtest(t, MomentumConfig(initial_capital=CAPITAL))
        if r and not r["equity"].empty:
            eq      = r["equity"]
            returns = eq.pct_change().dropna()
            sharpe  = (returns.mean() / returns.std()) * np.sqrt(252)
            total   = (eq.iloc[-1] / eq.iloc[0] - 1) * 100
            dd      = ((eq - eq.cummax()) / eq.cummax()).min() * 100
            trades  = len(r["trades"]) if not r["trades"].empty else 0

            momentum_results.append({
                "Ticker":         t,
                "Total Ret (%)":  round(total, 1),
                "Sharpe":         round(sharpe, 2),
                "Max DD (%)":     round(dd, 1),
                "Trades":         trades,
            })

    if momentum_results:
        mom_df = pd.DataFrame(momentum_results).sort_values(
            "Sharpe", ascending=False
        )
        print(mom_df.to_string(index=False))

    # --- Tearsheet für beste Momentum Aktie ---
    best_ticker = mom_df.iloc[0]["Ticker"] if momentum_results else TICKER
    print(f"\nDetail Tearsheet: {best_ticker}")
    best_mom = run_momentum_backtest(
        best_ticker, MomentumConfig(initial_capital=CAPITAL)
    )
    
    if not best_mom["trades"].empty:
        generate_tearsheet(best_mom)
    else:
        print(f"\nKeine Trades für {best_ticker} gefunden. Tearsheet kann nicht erstellt werden.")