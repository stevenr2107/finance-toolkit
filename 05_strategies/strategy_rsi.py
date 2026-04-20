"""
Day 12 - RSI Mean Reversion strategie 
vollständige strategie mit:
- Entry / exit 
- Stopp loss & Take Profit 
- Position Siziing  (Kelly + Fixed % )
- Multi- Stock Backtest 
- Professionelles Tearsheet 
"""

import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from dataclasses import dataclass # macht es einfacher klassen zu erstellen die nur daten halten sollen (wie eine Tabelle)
from typing import Optional # noch nicht klare parameter dürfen auch None sein statt einem int bsp. Aktienpreis der noch nicht sicher ist
import warnings
warnings.filterwarnings("ignore")


@dataclass
class StrategyConfig:
    """
    Alle Parameter an einem Ort
    Nie Magic Numbers im Code verstreuen 
    """

    # Entry/exit 
    rsi_entry: int = 30 # RSI unter diesem Wert -> Long
    rsi_exit: int = 65  # RSI über diesem Wert -> Exit
    rsi_window: int = 14 # RSI Berechnung über 14 Tage

    # Risikomanagement 
    stop_loss_pct: float = 0.05 # 5% Stop Loss
    take_profit_pct: float = 0.15 # 15% Take Profit
    max_hold_days: int = 30 # Maximal 30 Tage halten

    # Position Sizing
    risk_per_trade: float = 0.02 # 2% des Kapitals pro Trade riskieren
    max_position: float = 0.2 # Maximal 20% des Kapitals in einer Aktie halten

    # Kosten
    commission: float =0.001 # 0.1% pro Trade
    slippage: float = 0.001 # 0.1% Slippage pro Trade

    # Kapital
    initial_capital: float = 10_000 # Startkapital 10k

def load_data(ticker: str, period = "5y") -> pd.DataFrame:
    df = yf.download(ticker, period=period, auto_adjust=True, progress=False # fortschrittsbalken ausschalten) 
    )
    df.columns = df.columns.get_level_values(0) # hier wird alles in einem Index gespeichert statt in mehreren Spalten
    return df.dropna() # fehlende Werte entfernen


def compute_rsi(close: pd.Series, window: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.ewm(com=window - 1, adjust=False).mean()
    avg_loss = loss.ewm(com=window - 1, adjust=False).mean()

    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs)) # rsi berechnen und zurückgeben

def fixed_fractional_size(capital: float,
                          entry_price: float,
                          stop_loss_price: float,
                          risk_per_trade: float = 0.02,
                          max_position: float=0.2) -> int:
        """
        Fixed fractiional position sizing 

        wie viele shares kaufen?
        antwort: So viele dass bei stop loss genau risk_per_trade % verloren wird.

        Beispiel:
        Kapital: 10k
        Risk per trade: 2%
        entry : 100$
        stop loss: 95$
        risk pro share: 5$
        -> Shares: 200$ / 5§ = 40 Shares = 4000$ ( 40% zu viel)
        -> Gedeckelt auf max_position: 20% = 2k /1000 = 20 shares
        """
        if entry_price <= stop_loss_price:
             return 0 # kein trade wenn stop loss über entry price liegt
        
        dollar_risk = capital * risk_per_trade
        risk_per_share = entry_price - stop_loss_price
        shares_by_risk = int(dollar_risk / risk_per_share)

        max_shares = int((capital * max_position) / entry_price)
        shares = min(shares_by_risk, max_shares) # wie viele shares kaufen
        
        return max(shares, 0) # sicherstellen dass es nicht negativ wird


def kelly_criterion(win_rate: float,
                    avg_win: float,
                    avg_loss: float) -> float:
     """
     Kelly Criterion - mathematisch optimale positionsgröße

     formel: f= W - (1-W) / (avg_win/avg_loss)

     WICHTIG: immer half kelly nutzen (/2) - full kelly
     ist theoretisch optimal aber in der praxis zu aggressiv 
     kein professional nutzt full kelly, da es zu großen drawdowns führen kann 
     """
     if avg_loss == 0:
          return 0 
     b = avg_win/avg_loss
     kelly = win_rate - (1 - win_rate) / b
     return max(min(kelly / 2, 0.25), 0) # half kelly nutzen, max 25% des kapitals riskieren, nicht negativ

@dataclass
class Trade:
     """ Alle Informationen zu einem Trade in einer Klasse bündeln """
     ticker: str
     entry_date: pd.Timestamp
     entry_price: float
     shares: int
     stop_loss: float
     take_profit: float
     exit_date: Optional[pd.Timestamp] = None
     exit_price: Optional[float] = None
     pnl: Optional[float] = None
     pnl_pct: Optional[float] = None
     duration: Optional[int] = None

def run_backtest(ticker: str,
                 cfg: StrategyConfig) -> dict:
     """
     Event-driven Backtest mit vollständigem Risikomanagement 

     Ablauf pro Tag:
     1. Prüfe ob offene Position gestoppt/exitiert wird
     2. prüfe ob neuees Entry signal vorliegt 
     3. update kapital und equity curve 
     """

     df = load_data(ticker)
     if len(df) < 200:
          return{}
     close = df["Close"].squeeze()
     high= df["High"].squeeze()
     low = df["Low"].squeeze()
     opens = df["Open"].squeeze()

     df["RSI"] = compute_rsi(close, cfg.rsi_window)

     capital = cfg.initial_capital
     equity_curve = [capital]
     dates = [df.index[0]]
     trades = []
     current_trade = None

     for i in range(1, len(df)): # wiederhole bis zum letzten Tag
        date = df.index[i]
        rsi_prev = df["RSI"].iloc[i-1] # Signal von gestern 
        open_price = float(opens.iloc[i]) # heute eröffnungs kurs
        high_price = float(high.iloc[i]) # heute höchster kurs
        low_price = float(low.iloc[i]) # heute niedrigster kurs
        close_price = float(close.iloc[i]) # heute schluss kurs

        # --- Offene Positionen managen ----
        if current_trade is not None: # wenn es eine offene position gibt 
             # prüfe ob offene position gestoppt/exited wird
             days_held = (date - current_trade.entry_date).days
             exit_price = None
             exit_reason = None

             # stop loss prüfen
             if low_price <= current_trade.stop_loss:
                  exit_price = current_trade.stop_loss 
                  exit_reason = "Stop Loss"
                  
             # take profit prüfen # bei take profit gibt es keine stop loss
             elif high_price >= current_trade.take_profit: # wenn der kurs heute über dem take profit liegt
                  exit_price = current_trade.take_profit
                  exit_reason = "Take Profit"
            
            # RSI Exit prüfen
             elif rsi_prev >= cfg.rsi_exit:
                  exit_price = open_price # wenn rsi exit signal gibt -> zum eröffnungs kurs verkaufen 
                  exit_reason = "RSI Exit"
                
            # Max haltedauer 
             elif days_held >= cfg.max_hold_days:
                    exit_price = open_price # zum eröffnungs kurs verkaufen 
                    exit_reason = "Max Hold "

            # Trade schließen wenn ein exit grund vorliegt
             if exit_price is not None:
                  cost = exit_price * current_trade.shares
                  proceeds = cost * (1- cfg.commission)
                  entry_cost = (current_trade.entry_price *
                                current_trade.shares * 
                                (1 + cfg.commission))
                  pnl = proceeds - entry_cost
                  pnl_pct = (pnl / entry_cost) * 100

                  capital += proceeds

                  current_trade.exit_date = date
                  current_trade.exit_price = round(exit_price,2)
                  current_trade.exit_reason = exit_reason
                  current_trade.pnl = round(pnl,2)
                  current_trade.pnl_pct = round(pnl_pct,2)
                  current_trade.duration = days_held

                  trades.append(current_trade)
                  current_trade = None

        # --- Neues entry signal ---
        if current_trade is None and rsi_prev < cfg.rsi_entry: # wenn es kein offenes trade gibt und rsi entry signal von gestern vorliegt
             # Slipppage auf entry 
             entry_price = open_price * (1 + cfg.slippage) # zum eröffnungs kurs kaufen + slippage
             stop_loss = entry_price * (1 - cfg.stop_loss_pct)
             take_profit = entry_price * (1 + cfg.take_profit_pct)

             shares = fixed_fractional_size(
                 capital = capital,
                 entry_price = entry_price,
                 stop_loss_price = stop_loss,
                 risk_per_trade = cfg.risk_per_trade,
                 max_position = cfg.max_position
             )
             if shares > 0: # wenn die berechnung der shares ergibt dass wir mindestens 1 share kaufen können
                  cost = entry_price * shares * (1 + cfg.commission) # entry kosten mit kommission
                  if cost <= capital: # wenn entry kosten weniger als kapital sind
                       capital -= cost
                       current_trade = Trade(
                            ticker=ticker,
                            entry_date=date,
                            entry_price=round(entry_price,2),
                            shares=shares,
                            stop_loss=round(stop_loss,2),
                            take_profit=round(take_profit,2)
                       ) 
        # Equity cash + offene positionen zum tagesschluss
         # Offene Positionen
        if current_trade is not None:
             open_value = close_price * current_trade.shares # offene position zum schlusskurs bewerten
        else:
             open_value = 0 # keine offene position
             
        equity_curve.append(round(capital + open_value, 2))
        dates.append(date)

    # Offene Posiotionen am ende schließen 
     if current_trade is not None: # wenn es am ende des backtests noch eine offene position gibt -> zum schlusskurs verkaufen
          last_price = float(close.iloc[-1]) # zum schlusskurs verkaufen
          pnl = (last_price - current_trade.entry_price) * current_trade.shares  # Gewinn/Verlust berechnen
          current_trade.exit_date = df.index[-1] # zum schlusskurs verkaufen
          current_trade.exit_price = last_price # letzten price anzeigen 
          current_trade.exit_reason = "End of Data" # exit grund am ende des backtests
          current_trade.pnl = round(pnl,2) # Gewinn/Verlust in Dollar
          current_trade.pnl_pct = round(pnl / (current_trade.entry_price * current_trade.shares) * 100, 2)
          current_trade.duration = (df.index[-1] - current_trade.entry_date).days
          trades.append(current_trade)

     equity = pd.Series(equity_curve, index=dates) # equity curve als zeitreihe speichern
     trades_df = pd.DataFrame([t.__dict__ for t in trades]) if trades else pd.DataFrame() # alle trades in einem dataframe speichern, wenn es keine trades gibt -> leeres dataframe
     """ Alle Ergebnisse in einem Dictionary zurückgeben """
     return {
             "ticker": ticker,
             "equity": equity,
             "trades": trades_df,
             "price": close,
             "rsi": df["RSI"],
             "config": cfg
        }

# --- TEARSHEET ---
def generate_tearsheet(result: dict) -> dict:
    """
    Alle wichtigen Kennzahlen und Charts in einem Dictionary bündeln
    """
    equity = result["equity"]
    trades_df = result["trades"]
    ticker = result["ticker"]

    returns = equity.pct_change().dropna()

    # Basis Kennzahlen 

    total_ret = ((equity.iloc[-1] / equity.iloc[0]) - 1) * 100
    years = len(equity) / 252
    cagr_val = ((equity.iloc[-1] / equity.iloc[0]) ** (1/years) - 1) * 100
    vol = returns.std() * np.sqrt(252) * 100
    sharpe = (returns.mean() / returns.std()) * np.sqrt(252) 
    
    # Drawdown berechnen
    rolling_max = equity.cummax() # kumulatives Maximum der equity curve
    dd_series = (equity - rolling_max) / rolling_max * 100 # drawdown in Prozent
    max_dd = dd_series.min() # maximaler drawdown
    
    # Calmar Ratio : CAGR / Max Drawdown
    calmar = cagr_val / abs(max_dd) if max_dd != 0 else 0

    # Trade Statistiken 
    if not trades_df.empty:
        completed = trades_df.dropna(subset=["pnl"]) # nur abgeschlossene trades
        wins = completed[completed["pnl"] > 0] # gewinn trades
        losses = completed[completed["pnl"] <= 0]

        win_rate = len(wins) / len(completed) * 100 if len(completed) > 0 else 0
        avg_win = wins["pnl_pct"].mean() if not wins.empty else 0
        avg_loss = losses["pnl_pct"].mean() if not losses.empty else 0
        profit_factor = (wins["pnl"].sum() / abs(losses["pnl"].sum())
                         if not losses.empty and losses["pnl"].sum() != 0 else 0)
        avg_duration = completed["duration"].mean() if "duration" in completed else 0

        # Kelly für zukünftige Positionsgröße
        kelly = kelly_criterion(
            win_rate = win_rate / 100,
            avg_win = abs(avg_win),
            avg_loss = abs(avg_loss)
        ) * 100

        exit_reasons = completed["exit_reason"].value_counts()

    else:
        win_rate = avg_win = avg = loss = profit_factor = 0
        avg_duration = kelly = 0
        exit_reasons = pd.Series()
    
    stats = { # alle berechneten Kennzahlen in einem Dictionary speichern
         "Ticker": ticker,
         "Total Return (%)": round(total_ret,2),
         "CAGR (%)": round(cagr_val,2),
         "Volatilität (%)": round(vol,2),
         "Sharpe Ratio": round(sharpe,2),
         "Max Drawdown (%)": round(max_dd,2),
         "Calmar Ratio": round(calmar,2),
         "Anzahl Trades": len(trades_df) if not trades_df.empty else 0,
         "Win Rate (%)": round(win_rate,2),
         "Avg Win (%)": round(avg_win,2),
         "Avg Loss (%)": round(avg_loss,2),
         "Profit Factor": round(profit_factor,2),
         "Avg Duration (d)": round(avg_duration,1),
         "Kelly Size (%)": round(kelly,1),
    }

    # Print
    print(f"\n{'='*50}")
    print(f"Tearsheet - {ticker}")
    print(f"{'='*50}")
    for k, v in stats.items():
        print(f"{k:20}: {v}")
    
    if not exit_reasons.empty:
        print("\nExit Reasons:")
        for reason, count in exit_reasons.items():
            print(f"{reason:<20} {count}x")
    print(f"{'='*50}")

    return stats

def plot_strategy(result: dict) -> None:
     """
     4- Pnale strategy chart 
     1. equity curve
     2. drawdown 
     3. RSI + Entry/Exit Signale 
     4. Trade Dauer Verteilung 
     """
     equity = result["equity"]
     trades_df = result["trades"]
     ticker = result["ticker"]
     rsi = result["rsi"]
     price = result["price"]
     cfg = result["config"]

     fig = make_subplots(
          rows=4, cols=1, shared_xaxes=False,
          vertical_spacing=0.06,
          row_heights=[0.35,0.2,0.3,0.15],
          subplot_titles=[
               f"{ticker} - Equity Curve",
               "Drawdown (%)",
               f"RSI ({cfg.rsi_window}) + Trade Signale",
               "Trade PnL Verteilung (%)"
          ]
     )
        # Panel 1: Equity Curve
     fig.add_trace(go.Scatter(
        x=equity.index, y=equity,
        name="Equity",
        line=dict(color="#2563eb", width=2),
        fill="tozeroy", fillcolor="rgba(37,99,235,0.08)"
    ), row=1, col=1)
     
     # Startkapital Linie
     fig.add_hline(
        y=cfg.initial_capital, line_dash="dot",
        line_color="#94a3b8", opacity=0.6, row=1, col=1
     )

     # Panel 2: Drawdown
     rolling_max = equity.cummax()
     dd = (equity - rolling_max) / rolling_max * 100
     fig.add_trace(go.Scatter(
        x=dd.index, y=dd.round(2),
        name="Drawdown",
        line=dict(color="#ef4444", width=1),
        fill="tozeroy", fillcolor="rgba(239,68,68,0.15)"
     ), row=2, col=1)
     
     # ---  Panel 3: RSI + Signale---
     fig.add_trace(go.Scatter(
        x=rsi.index, y=rsi.round(1),
        name="RSI", line=dict(color="#0891b2", width=1.5)
     ), row=3, col=1)

     fig.add_hline(y=cfg.rsi_entry, line_dash="dash",
                    line_color="#16a34a", opacity=0.7, row=3, col=1)
     fig.add_hline(y=cfg.rsi_exit, line_dash="dash",
                    line_color="#ef4444", opacity=0.7, row=3, col=1)
     
     # Entry / Exit Marker
     if not trades_df.empty:
          completed = trades_df.dropna(subset=["exit_date"])

          # Entries
          entry_rsi = [
               rsi.asof(d) if d in rsi.index or d > rsi.index[0] # asof gibt den letzten gültigen Wert zurück, wenn das Datum im RSI Index liegt, sonst wird der erste Wert zurückgegeben
               else None for d in completed["entry_date"]
          ]
          fig.add_trace(go.Scatter(
               x=completed["entry_date"], y=entry_rsi,
               mode="markers", name="Entry",
               marker=dict(color="#16a34a", symbol="triangle-up", size=10)
          ), row=3, col=1)

          # Exits Farbe nach Ergebnis 
          exit_colors = [
               "#16a34a" if p > 0 else "#ef4444" for p in completed["pnl"]
          ]
          exit_rsi = [
               rsi.asof(d) if d in rsi.index or d > rsi.index[0]
               else None for d in completed["exit_date"]
          ]
          fig.add_trace(go.Scatter(
               x=completed["exit_date"], y=exit_rsi,
               mode="markers", name="Exit",
               marker=dict(color=exit_colors, symbol="triangle-down", size=10, line=dict(width=1, color="white"))
          ), row=3, col=1)
     # --- Panel 4 Trade PnL Distribution ---
     if not trades_df.empty and "pnl_pct" in trades_df.columns:
          completed = trades_df.dropna(subset=["pnl_pct"])
          colors = ["#16a34a" if v > 0 else "#ef4444" for v in completed["pnl_pct"]]
        
          fig.add_trace(go.Bar(
               x=list(range(len(completed))),
               y=completed["pnl_pct"].round(2),
               name=" Trade PnL %",
               marker_color=colors,
               opacity=0.8
          ), row=4, col=1)

          fig.add_hline(y=0, line_color="#1e293b",
                        line_width=1, row=4, col=1)

     fig.update_layout(
          height=900,
          template="plotly_white",
          showlegend=True,
          legend=dict(orientation="h", y=1.02,),
          margin = dict(l=0, r=0, t=40, b=0)
     )
     fig.update_yaxes(title_text="Kapital ($)", range=[equity.min() * 0.98, equity.max() * 1.02], row=1, col=1)
     fig.update_yaxes(title_text="DD (%)", row=2, col=1)
     fig.update_yaxes(title_text="RSI", row=3, col=1, range=[0,100])
     fig.update_yaxes(title_text="PnL (%)", row=4, col=1)

     fig.show()

def multi_stock_backtest(tickers: list,
                         cfg: StrategyConfig) -> pd.DataFrame:
     """
     Testet die Strategie auf mehreren Aktien
     out of sample test - die entscheidende frage: funktioniert die strategie nicht nur auf einer aktie sondern auf vielen?
     """
     all_stats = []

     for ticker in tickers: 
          print(f"Backtesting {ticker}...", end= " ")
          result = run_backtest(ticker, cfg)
          if result:
               stats = generate_tearsheet(result)
               all_stats.append(stats)
          else :
               print("Nicht genügend Daten, überspringe...")
     if not all_stats:
          return pd.DataFrame()
     
     df = pd.DataFrame(all_stats)
     df = df.sort_values("Sharpe Ratio", ascending=False).reset_index(drop=True)
     return df

if __name__ == "__main__": # Hier kann man an den Parametern spielen 
     # --- Konfiguration ---
     cfg = StrategyConfig(
          rsi_entry=40, # normal bei 30
          rsi_exit=70, # normal bei 65
          stop_loss_pct=0.1, # normal bei 5%
          take_profit_pct=0.3, # normal bei 15%
          max_hold_days=60, # normal bei 30
          risk_per_trade=0.05, # normal bei 2%
          max_position=1.0, # normal bei 20%
          initial_capital=10_000
     )

     # --- Single Stock Deep Dive ---
     TICKER = "AAPL"
     print(f"Strategie BAcktest: {TICKER}")

     result = run_backtest(TICKER, cfg)
     stats = generate_tearsheet(result)
     plot_strategy(result)

     # --- Parameter Sensitivität ---
     print("\n --- RSI Entry Level Sensitivität ---")
     for entry in [25,30,35, 40]:
          cfg_test = StrategyConfig(rsi_entry=entry,
                                    initial_capital=10_000)
          r = run_backtest(TICKER, cfg_test)
          if r and not r["trades"].empty:
               eq = r["equity"]   # equity curve
               ret = eq.pct_change().dropna() # returns
               sh = (ret.mean() / ret.std()) * np.sqrt(252) # sharpe
               tot = (eq.iloc[-1] / eq.iloc[0] -1) * 100  # total return
               tr = len(r["trades"]) # Anzahl trades
               print(f" RSI Entry {entry}:"
                     f"Return={tot:.1f}% "
                     f"Sharpe={sh:.2f} "
                     f"Trades={tr}")
     # --- Multi Stock Backtest ---
     print("\n--- Multi Stock Backtest ---")
     test_universe = [
          "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA",
          "XOM", "JNJ", "JPM", "SPY"
     ]

     results_df = multi_stock_backtest(test_universe, cfg)

     if not results_df.empty:
          print("\n RANKING nach sharpe ratio:")
          cols = ["Ticker", "Total Return (%)", "Sharpe Ratio", "Max Drawdown (%)"
                  ,"Win Rate (%)", "Profit Factor", "Anzahl Trades"]
          print(results_df[cols].to_string(index=False))

          # Top Performer plotten
          best = results_df.iloc[0]["Ticker"]
          print(f"\nBester Ticker: {best} - Detail Chart:")
          best_result = run_backtest(best, cfg)
          plot_strategy(best_result)
     

"""
Kann man sehr gut dran rumspielen und posten. 
"""