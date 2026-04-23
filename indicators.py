"""
Eigene Indikatorfunktionen um sie zu verstehen und easy zu importieren 
"""

"""
indicators.py - Custom Technical Analysis Library

Selbst implementierte Indikatoren ohne externe TA-Libraries
Wiederverwendbar in allen anderen Modulen 

Verfügbare Funktionen:
sma(series, window) -> Simple Moving Average
ema(series, window) -> Exponential Moving Average
rsi(series, window) -> Relative Strength Index
macd(series, fast, slow, signal) -> Moving Average Convergence Divergence
find_crossover(df) -> golden/Death Cross Signale 

Verwendung:
from indicators import sma, ema, rsi, macd
"""

import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots



def sma(series: pd.Series, window:int ) -> pd.Series: # series ist nur eine spalte 
    """
    Simple Moving Average - Durchschnitt der letzten N Tage.
    Jeder Tag gleich gewichtet
    """
    return series.rolling(window=window).mean() # mean ist durchschnitt
# window = window -> computer soll meinen Wert annehmen  kann man auch in window = anzahl_tage umbenennen
# rolling -> man nimmt das window und rutscht immer eins rüber und nimmt dann die neuen zahlen (kamera kann 5 leute erfassen und wir haben 30 -> wir müssen so oft rollen bis wir bei 25-30 sind )

def ema(series: pd.Series, window:int) -> pd.Series:
    """
    EMA - neuere Tage werden stärker gewichtet 
    Formel: EMA = Kurs * k + EMA_gestern * (1-k)
    wobei k=2 / (window + 1)
    ewm = exponential weighted moving
    """
    return series.ewm(span=window, adjust=False).mean()

def rsi(series: pd.Series, window: int=14 ) -> pd.Series:
    """
    RSI misst ob eine Aktie überkauft (>70) oder überverkauft (<30) ist.

    Formel:
    RS = Durchschnitt der Gewinntage / Durchschnitt der Verlusttage 
    RSI = 100-(100/(1+RS))

    """
    delta = series.diff()       #tägliche Veränderung 

    gain = delta.clip(lower=0)  # nur positive tage nehmen >0
    loss = -delta.clip(upper=0) # nur negative Tage 

    avg_gain = gain.ewm(com=window - 1, adjust=False).mean()
    avg_loss = loss.ewm(com=window - 1, adjust = False).mean()

    rs = avg_gain / avg_loss
    rsi = 100- (100 / (1+rs))

    return rsi 


def macd(series: pd.Series,
         fast: int= 12,
         slow: int = 26,
         signal: int = 9) -> pd.DataFrame:
    """
    MACD = EMA(12) - EMA(26)
    Signal Line = EMA(9) des MACD
    Histogram = MACD - Signal 
    
    Bullish Signal: MACD kreuzt Signal von unten 
    Bearish Signal: MACD kreuzt Signal von oben 
    """

    ema_fast = ema(series,fast) # oben steht fast and slow 
    ema_slow = ema(series, slow)

    macd_line = ema_fast - ema_slow
    signal_line = ema(macd_line, signal)
    histogram = macd_line - signal_line 

    return pd.DataFrame({
        "macd":     macd_line,
        "signal":   signal_line,
        "histogram": histogram 
    })

# Alles auf einem Chart visualisieren 

def plot_full_analysis(ticker: str, period: str="1y") -> None:
    # Daten laden 
    df = yf.download(ticker, period = period, auto_adjust=True)
    df.columns = df.columns.get_level_values(0) # nur die values nehmen statt 5x apple apple apple 
    close= df["Close"].squeeze() # alles in eine liste statt in mehrere spalten 

    # Indikatoren berechnen
    df["SMA20"] = sma(close, 20)
    df["SMA50"] = sma(close, 50)
    df["SMA200"]= sma(close, 200)
    df["EMA20"] = ema(close, 20)
    df["RSI"]   = rsi(close)
    macd_df     = macd(close)

    # 3 Panel Chart: Kurs/ RSI / MACD
    fig = make_subplots(
        rows=3, cols = 1,
        shared_xaxes=True,
        vertical_spacing=0.04,
        row_heights=[0.55,0.22,0.23],
        subplot_titles=[
            f"{ticker} - Kurs + Moving Averages",
            "RSI (14)",
            "MACD (12/26/9)"
        ]
    )

    # --- Panel 1: Kurs + MAs ---
    fig.add_trace(go.Scatter(
        x=df.index, y=close,
        name="Kurs", line=dict(color="#1e293b", width=1.5)
    ), row=1, col=1)

    fig.add_trace(go.Scatter(
        x = df.index, y=df["SMA20"],
        name="SMA 20", line=dict(color="#3b82f6", width=1, dash="dot")
    ), row=1, col=1)    
    
    fig.add_trace(go.Scatter(
        x = df.index, y=df["SMA50"],
        name="SMA 50", line=dict(color="#f59e0b", width=1.2)
    ), row=1, col=1)    
    
    fig.add_trace(go.Scatter(
        x = df.index, y=df["SMA200"],
        name="SMA 200", line=dict(color="#ef4444", width=1.5)
    ), row=1, col=1)    
    
    fig.add_trace(go.Scatter(
        x = df.index, y=df["EMA20"],
        name="EMA 20", line=dict(color="#8b5cf6", width=1, dash="dash")
    ), row=1, col=1)    
    
    # --- Panel 2: RSI ---

    fig.add_trace(go.Scatter(
        x = df.index, y=df["RSI"],
        name="RSI", line=dict(color="#0891b2", width=1.5)
    ), row=2, col=1) 
    
     

    # Überkauft/Überverkauft Linien 
    fig.add_hline(y=70, line_dash="dash",
                  line_color="#ef4444", opacity=0.7, row=2, col=1)  
    fig.add_hline(y=30, line_dash="dash",
                  line_color="#16a34a", opacity=0.7, row=2, col=1)
    fig.add_hline(y=50, line_dash="dash",
                  line_color="#94a3b8", opacity=0.5, row=2, col=1)
    
    # --- Panel 3: MACD ---
    # Histogramm grün wenn positiv rot wenn negativ 
    colors = ["#16a34a" if v >= 0 else "#ef4444"
              for v in macd_df["histogram"]]
    
    fig.add_trace(go.Bar(
        x=df.index, y=macd_df["histogram"],
        name="Histogram", marker_color=colors, opacity=0.6
    ), row=3, col=1)

    fig.add_trace(go.Scatter(
        x=df.index, y=macd_df["macd"],
        name="MACD", line=dict(color="#2563eb", width=1.5)
    ), row=3, col=1)

    fig.add_trace(go.Scatter(
        x=df.index, y=macd_df["signal"],
        name="Signal", line=dict(color="#f59e0b", width=1.5)
    ), row=3, col=1)

    fig.update_layout(
        height=1200,
        template="plotly_white",
        hovermode="x unified",
        legend=dict(orientation="h", y=1.02)
    )

    fig.update_yaxes(title_text="Kurs (USD)", row=1, col=1)
    fig.update_yaxes(title_text="RSI", row=2, col=1, range=[0,100])
    fig.update_yaxes(title_text="MACD", row=3, col=1)

    fig.show()

# --- Main ---
if __name__ == "__main__":
    # Teste mit verschiedenen Aktien 
    plot_full_analysis("AAPL")
    plot_full_analysis("NVDA") # Sehr unterschiedlicher RSI verlauf 

# --  Golden cross --
def find_crossovers(df:pd.DataFrame) -> pd.DataFrame:
    """
    Golden Cross: SMA 50 kreuzt SMA 200 von unten : bullish 
    Death Cross: SMA 50 kreuzt SMA 200 von oben : bearish 
    """

    close = df["Close"].squeeze()
    df = df.copy()
    df["SMA50"] = sma(close,50)
    df["SMA200"] = sma(close,200)

    # 1 wenn sma50 > sma200 sonst 0 
    df["above"] = (df["SMA50"] > df["SMA200"]).astype(int) # zeigt 1/ 0 wer oben ist

    # Wenn sich das ändert = crossover
    df["cross"] = df["above"].diff() # neue spalte mit heute - gestern -> viele 0 ab und zu ne 1 /-1

    golden = df[df["cross"] == 1][["Close", "SMA50", "SMA200"]] # zeige nur wo df[cross] = 1 und dann printe nur close, sma und sma200
    death  = df[df["cross"] == -1][["Close", "SMA50", "SMA200"]] # zeige nur wo df[cross] = -1

    print("=== Golden Cross Signale (Bullish) ===")
    print(golden.round(2))
    print("=== Death Cross Signale (Bearish) ===")
    print(death.round(2))

    return df

if __name__ == "__main__":
    raw = yf.download("SPY", period="5y", auto_adjust=True)
    raw.columns = raw.columns.get_level_values(0)
    find_crossovers(raw)