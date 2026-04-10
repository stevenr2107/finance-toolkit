"""
Day 08 - Backtesting Grundlagen
Erste Strategie: Moving Average Crossover Buy&Hold benchmark.
Kennzahlen: Sharpe Ratio, Max Drawdown, CAGR
"""

# TODO: SMA Crossover Strategie implementieren 
# TODO: Benchmark: Buy& Hold SPY
# TODO Equity  Curve plotten 
# TODO: Sharpe, Drawdown, CAGR berechnen

"""
Die größten Fehler
1. Survivorship-Bias:  Es werden nur Firmen getestet, die es jetzt noch gibt und alte die Bankrupt gegangen sind, werden ausgelassen
2. Look-ahead Bias:    Man nutzt Daten, die nicht verfügbar waren zu dem Zeitpunkt -> man kauft nachts was, ist aber erst am nächsten morggen drin 
3. Overfitting:        Man optimiert an historische Daten, die jetzt versagen 
"""

import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# --- Daten laden ---
def load_data(ticker:str, period: str = "5y") -> pd.DataFrame:
    df = yf.download(ticker, period=period, auto_adjust=True)
    df.columns = df.columns.get_level_values(0)
    return df[["Open", "High", "Low", "Close", "Volume"]].dropna() # doppelte Klammer, da man 2 Listen nimmt und jedes Element eine Series in sich ist
    # dropna sorgt dafür, dass das program nicht abstürzt wenn etwas keine nummer ist 

# --- Kennzahlen berechnen ---
def cagr(equity_curve: pd.Series) -> float:
    """
    Compound Annual Growth Rate - Jährliches Wachstum in % um auf unseren endwert zu kommen
    Formel: (Endwert/Startwert) ^(1/jahre ) - 1
    """

    years = len(equity_curve) /252 # anzahl der insgesamten Tage durch die handelstage -> jahre laufzeit
    total_return = equity_curve.iloc[-1] / equity_curve.iloc[0] # letzten wert der kurve / ersten -> 1.6 = 50% gewinn
    return ( total_return ** (1/years) -1 ) * 100 # wachstum pro jahr zurückberechnet

def sharpe_ratio(returns: pd.Series, risk_free: float = 0.05) -> float: # Qualität
    """
    Sharpe Ratio - Return pro Risikoeinheit. - wieviel ist die aktie geschwankt -> bsp Aktie a schwankt und b nicht aber gleicher return  -> b hat höheres sharpe_ratio
    > 1.0: gut, >1.5: sehr gut, > 2.0 exzellent
    Risikofrei: aktuell ca. 5% (US Treasury )
    """

    excess = returns - risk_free / 252  # man zieht von den returns die am ende rauskommen, wieviel man bei us bonds bekommen hätte (5%) -> hat börse mehr eingebracht als sicher?
    return(excess.mean() / returns.std()) *np.sqrt(252) # excess -> durchschnittliche überrendite / die schwankungen und die hochrechnung aufs jahr

def max_drawdown(equity_curve: pd.Series) -> float:
    """
    Maximaler Drawdown - größter möglicher Verlust in %
    Das ist die Zahl, die dich nachts wachhält wenn du bei novo -80% bist
    """

    rolling_max = equity_curve.cummax() # kummuliertes maximum
    drawdown    =  (equity_curve - rolling_max) /rolling_max
    return drawdown.min() * 100 # wir haben negative zahlen deshalb minimum

def print_stats(name: str, equity: pd.Series, returns: pd.Series) -> dict:
    """ Alle Kommazahlen auf einem Blick"""
    stats={
        "Strategie": name,
        "CAGR (%)" : round(cagr(equity), 2),
        "Sharpe"   : round(sharpe_ratio(returns), 2),
        "Max DD (%)": round(max_drawdown(equity), 2),
        "Total Return": f"{((equity.iloc[-1] / equity.iloc[0]) - 1) * 100:.1f}%",
        "Trades": None # wird später ersetzt
    }
    print(f"\n{"=" * 40}")
    print(f" {name}")
    print(f"{"=" * 40}")
    for k, v in stats.items():
        if v is not None:
            print(f" {k:<20} {v}")
    return stats
    

# --- Benchmark: Buy&Hold ---
def buy_and_hold(df: pd.DataFrame, capital: float = 10_000) -> dict:
    """
    Simpelste Strategie: einmal kaufen, nie verkaufen.
    Alles muss sich gegen das messen.
    """
    # Normalisierte Equity Curve: startet bei capital
    equity = (df["Close"] / df["Close"].iloc[0]) * capital
    returns = df["Close"].pct_change().dropna()

    stats = print_stats("Buy & Hold", equity, returns)
    stats["Trades"] = 1
    stats["equity"] = equity
    stats["returns"] = returns

    return stats

def sma_crossover(df: pd.DataFrame,
                  fast: int = 50,
                  slow: int = 200,
                  capital: float = 10_000) -> dict:
    """
    Golden Cross / Death Cross Strategie.

    Signal:
        SMA_fast > SMA_slow → Long (invested)
        SMA_fast < SMA_slow → Cash (out of market)

    Look-ahead Bias vermeiden:
        Signal entsteht am Tag X → Kauf am Tag X+1 (Open)
    """
    df = df.copy()
    df["SMA_fast"] = df["Close"].rolling(fast).mean()
    df["SMA_slow"] = df["Close"].rolling(slow).mean()

    # Signal 1 = long, 0 = cash
    # .shift(1) verhindert look-ahead bias - Signal von gestern 
    df["signal"] = np.where( # where hat 2 bedingung und wenn 1 dann = 1 wenn nicht = 0
        df["SMA_fast"] > df["SMA_slow"], 1, 0
    )
    df["position"] = df["signal"].shift(1) # <- kritisch  - look ahead bias - man weiß erst am nächsten tag was am vorherigen passiert ist

    # Returns nur wenn invested
    df["market_return"] = df["Close"].pct_change()
    df["strategy_return"] = df["market_return"] * df["position"]

    # Transaktionskosten 0.1% pro Trade 
    df["trade"]  = df["position"].diff().abs()
    df["cost"]  = df["trade"] * 0.001
    df["strategy_return"] -= df["trade"] * 0.001 # speicher das neue in der alten variable ab wie x-y = x

    # Equity Curve 
    equity = (1 + df["strategy_return"].fillna(0)).cumprod() * capital # fillna -> lücken werden mit 0 aufgefüllt - cumprod -> x*y = z -> z*a = b...
    returns = df["strategy_return"].dropna()

    # Trade-Anzahl zählen
    trades = int(df["trade"].sum())

    stats = print_stats(
        f"SMA Crossover ({fast}/{slow})", equity, returns
    )
    stats["Trades"] = trades
    stats["equity"] = equity
    stats["returns"] = returns
    stats["df"] = df

    return stats


def rsi_mean_reversion(df: pd.DataFrame,
                       rsi_low: int = 30,
                       rsi_high: int = 70,
                       window : int= 14,
                       capital: float = 10_000) -> dict:
    """
    RSI mean reversion strategie 

    signal:
        RSI < rsi_low -> long (Überverkauft - Erholung erwartet)
        RSI > rsi_high -> Cash ( Überverkauft - rückgang möglich)
    """
    df=df.copy()

    # RSI berechnen
    delta = df["Close"].diff()
    gain = delta.clip(lower=0) # alle negativen werte werden auf 0 gesetzt -> nur gewinne
    loss = -delta.clip(upper=0) # negative werte werden positiv gemacht um besser zu rechnen später
    avg_gain = gain.ewm(com=window - 1, adjust=False).mean() # com =window -> eine einstellung, die heutige kurse höher gewichtet als alte
    avg_loss = loss.ewm(com=window - 1, adjust=False).mean()
    rs = avg_gain/avg_loss # durchschnittliche gewinne/verluste
    df["RSI"] = 100- (100/(1+rs)) # formel

    # Signal mit look ahead schutz 
    df["signal"] =          np.where(df["RSI"] < rsi_low, 1, # where war die bedingung und was dann passiert
                            np.where(df["RSI"] > rsi_high, 0, np.nan)) # np.nan = not a number - rsi ist nicht zu hoch nicht zu tief -> keine meinung 
    df["signal"] = df["signal"].ffill().fillna(0) # forward fill -> wenn ein nan wird das letzte signal angeschaut und nach vorne kopiert 
    # diese zwei funktionen sorgen dafür, das man hält solange man kein signal bekommt 
    df["position"] = df["signal"].shift(1) # wieder nur die daten nehmen die man auch kennt zu dem zeitpunkt 

    # Returns
    df["market_return"] = df["Close"].pct_change()
    df["strategy_return"] = df["market_return"] * df["position"]

    # Transaktionskosten
    df["trade"] = df["position"].diff().abs()
    df["strategy_return"] -= df["trade"] * 0.001

    equity = (1 + df["strategy_return"].fillna(0)).cumprod() * capital
    returns = df["strategy_return"].dropna()
    trades = int(df["trade"].sum())

    stats= print_stats(
        f"RSI Mean Reversion ({rsi_low}/{rsi_high})", equity, returns
    )
    stats["Trades"] = trades
    stats["equity"] = equity
    stats["returns"] = returns

    return stats


# --- Equity curve vergleich ---
def plot_equity_curves(*strategies: dict, title: str = "Strategy Comparison") -> None:  # der stern sorgt daf+r, das strategies eine sammlung wird 
    """
    Vergleicht mehrere equity kurven auf einem chart 
    darunter : Drawdown der besten strategie
    """
    colors = ["#2563eb", "#16a34a", "#f59e0b", "#8b5cf6", "#ef4444"]

    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        row_heights=[0.7,0.3],
        subplot_titles=[title, "Drawdown"]
    )

    for i, strat in enumerate(strategies):
        equity = strat["equity"]
        name= strat["Strategie"]
        color= colors[i % len(colors)]

        # Equity curve 
        fig.add_trace(go.Scatter(
            x = equity.index,
            y=equity.round(2),
            name=name,
            line=dict(color=color, width=2)
        ), row=1, col=1)

        # Drawdown
        rolling_max = equity.cummax()
        drawdown = ((equity - rolling_max) / rolling_max * 100).round(2)

        fig.add_trace(go.Scatter(
            x=drawdown.index,
            y=drawdown,
            name=f"{name} DD",
            line=dict(color=color, width=1, dash="dot"),
            fill= "tozeroy", # fülle zwischen x linie und chart aus 
            fillcolor=color.replace(")", ", 0.1").replace("rgb", "rgba"), # normal rgb xyz,opacity. die 0.1 ist opacity und rgb -> rgba a = alpha = opacity
            opacity=0.3,
            showlegend=False
        ), row=2, col=1)

        fig.update_layout(
            height=650,
            template="plotly_white",
            hovermode="x unified",
            legend=dict(orientation="h", y=1.02),
            margin=dict(l=0, r=0, t=40, b=0)
        )

        fig.update_yaxes(title_text="Portfolio Wert ($)", row=1, col=1)
        fig.update_yaxes(title_text="Drawdown (%)", row=2, col=1)

        fig.show()


# --- Alles zusammenführen und laufen lassen ---

if __name__ == "__main__":
    #--- Setup ---
    TICKER = "SPY" # S&P 500 als benchmark test
    PERIOD = "10y" # genug daten
    CAPITAL = 10_000 # Startkapital

    print(f"\n Backtest: {TICKER} | {PERIOD} | ${CAPITAL:,} Startkapital")

    df = load_data(TICKER, PERIOD)
    print(f"Daten geladen. {len(df)} Handelstage")

    # --- Strategien ---
    bah = buy_and_hold(df, CAPITAL)
    sma = sma_crossover(df, fast=50, slow=200, capital =CAPITAL)
    rsi_st = rsi_mean_reversion(df, rsi_low=30, rsi_high=70, capital = CAPITAL)

    # --- Vergleichstabelle ---
    print("\n" + "="*60)
    print("ZUSAMMENFASSUNG")
    print("="*60)

    summary=pd.DataFrame([
        {k: v for k,v in s.items() # k: v zeigt name und wert sollen als paar bleiben 
         if k not in ["equity", "returns", "df"]} # kopiere alles außer der schlüssel heißt equity...
         for s in [bah, sma, rsi_st] # für alle 3 strategien ausführen
    ])
    print(summary.to_string(index=False))

    # --- Chart ---
    plot_equity_curves(bah, sma, rsi_st,
                       title=f"{TICKER} - Strategy Comparison (10Y)")
    
    #--- Teste verschiedene Parameter ---
    print("\n --- Parameter Sensitivität: SMA Crossover ---")
    for fast, slow in [(20, 50), (50, 200), (10, 30)]:
        s = sma_crossover(df, fast=fast, slow=slow, capital = CAPITAL)


    # Nicht nur auf SPY testen - das ist Overfitting 
    print("\n--- Out-of-Sample Test: verschiedene Ticker ---")

    test_tickers = ["QQQ", "AAPL", "MSFT", "GLD"]

    results = []
    for t in test_tickers:
        d = load_data(t, "5y")
        bh = buy_and_hold(d, CAPITAL)
        sm = sma_crossover(d, capital=CAPITAL)

        results.append({
            "Ticker": t,
            "B&H CAGR": bh["CAGR (%)"],
            "SMA CAGR": sm["CAGR (%)"],
            "B&H Sharpe": bh["Sharpe"],
            "SMA Sharpe": sm["Sharpe"],
            "SMA schlägt B&H": sm["CAGR (%)"] > bh["CAGR (%)"],
        })

    results_df = pd.DataFrame(results)
    print(results_df.to_string(index=False))