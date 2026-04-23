"""
Day 09 - Stock Screener 
Filtert den kompletten S&P 500 nach technischen 
fundamentalen Kriterien. Kein bezahltes Tool nötig
"""

import yfinance as yf
import pandas as pd 
import numpy as np
import plotly.graph_objects as go
from datetime import datetime
import time
import warnings
warnings.filterwarnings("ignore")

def get_sp500_tickers() -> list:
    """
    Lädt aktuelle S&P 500 Bestandteile direkt von Wikipedia
    Kein API Key nötig - pandas liest die HTML-Tabelle
    """
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    table = pd.read_html(url, storage_options={'User-Agent': 'Mozilla/5.0'})[0]

    # BRK.B -> BRK-B (Yahoo Finance Format)
    tickers = table["Symbol"].str.replace(".", "-", regex=False).tolist()

    print(f"S&P 500 geladen: {len(tickers)} Ticker")
    return tickers, table

def compute_indicators(ticker: str) -> dict | None:
    """
    Berechnet alle screening-kriterien für einen einzelnen Ticker
    Gibt none zurück wenn daten fehlen oder fehlerhaft sind 
    """
    try:
        df = yf.download(ticker, period="1y",
                         auto_adjust=True, progress=False)
        if len(df) < 200: # zu wenig Daten
            return None
        
        df.columns = df.columns.get_level_values(0)
        close = df["Close"].squeeze()

        # --- Technische Indikatoren ---
        sma20 = close.rolling(20).mean().iloc[-1]
        sma50 = close.rolling(50).mean().iloc[-1]
        sma200 = close.rolling(200).mean().iloc[-1]

        # RSI
        delta = close.diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)
        avg_gain = gain.ewm(com = 13, adjust=False).mean() 
        avg_loss = loss.ewm(com = 13, adjust=False).mean() 
        rs = avg_gain / avg_loss
        rsi_val = (100- (100 / (1+rs))).iloc[-1]

        # Preise
        current = close.iloc[-1]
        high_52w = close.max()
        low_52w = close.min()

        # Performance 
        ret_1m = (current / close.iloc[-21] - 1) *100
        ret_3m = (current / close.iloc[-63] - 1) *100
        ret_6m = (current / close.iloc[-126] - 1) *100
        ret_1y = (current / close.iloc[0] - 1) *100

        # Volatilität
        returns = close.pct_change().dropna()
        volatility = returns.std() * np.sqrt(252) * 100

        # Volumen 
        avg_vol = df["Volume"].mean()
        vol_ratio = df["Volume"].iloc[-1] / avg_vol # heute vs durchschnitt

        # Abstand von High/Lows
        pct_from_high = (current / high_52w - 1) * 100
        pct_from_low = (current / low_52w - 1) * 100

        # Trend-Score: Wie viele MAs ist der Kurs drüber?
        trend_score = sum([
            current > sma20,
            current > sma50,
            current > sma200,
            sma20 > sma50,
            sma50 > sma200,
        ])

        return {
            "Ticker": ticker,
            "Kurs": round(float(current), 2),
            "RSI": round(float(rsi_val), 1),
            "SMA20": round(float(sma20), 2),
            "SMA50": round(float(sma50), 2),
            "SMA200": round(float(sma200), 2),
            "1M %": round(float(ret_1m), 1),
            "3M %": round(float(ret_3m), 1),
            "6M %": round(float(ret_6m), 1),
            "1Y %": round(float(ret_1y), 1),
            "Volatilität": round(float(volatility), 1),
            "Ø Volumen": int(avg_vol),
            "Vol Ratio": round(float(vol_ratio), 2),
            "% vom Hoch": round(float(pct_from_high), 1),
            "% vom Tief": round(float(pct_from_low), 1),
            "Trend Score": int(trend_score),
        }
    
    except Exception:
        return None
    
def run_screener(tickers: list,
                 max_stocks: int = 503,
                 delay: float = 0.3) -> pd.DataFrame:
    
    """
    Lädt alle ticker und baut die screener tabelle

    args:
    max_stocks für tests erst 50 nehmen später alle 500
    delay: Pause zwischen requests - yahoo mag kein spamming 
    """
    results = []
    failed = []
    subset = tickers[:max_stocks]

    print(f"\nScreene {len(subset)} Aktien ...")
    print("-" * 40)

    for i, ticker in enumerate(subset):
        data = compute_indicators(ticker)

        if data: # wenn data also nicht leer oder so , resultat anhängen mit rsi und haken 
            results.append(data)
            status = f"✓ {ticker:<8} RSI: {data['RSI']:.0f}"
        else:
            failed.append(ticker)
            status = f"✗ {ticker:<8} (übersprungen)"

        # Fortschritt anzeigen um zu wissen ob programm arbeitet oder abstürzt
        if ( i + 1) % 10 == 0: # nur bei jeder 10. aktie wird ausgedruckt und wenn rest 0, printe 
            print(f"  {i+1} / {len(subset)} - {status}") # printe 10/länge datensatz(500) - status

        time.sleep(delay) # damit man nicht rausgeworfen wird - > delay = 0.3 ( oben)
    
    print(f"\nFertig {len(results)} erfolgreich, {len(failed)} fehlgeschlagen")
    return pd.DataFrame(results)

def filter_oversold(df: pd.DataFrame, rsi_max: int = 35) -> pd.DataFrame:
    """ 
    überverkaufte aktien rsi unter schwellenwert
    klassisches mean reversion setup 
    """
    mask = (
        (df["RSI"] < rsi_max) &
        (df["Trend Score"] >= 2) & # noch im übergeordneten aufwärtstrend
        (df["Ø Volumen"] > 500_000) # ausreichend liquide (500k aktien gehandelt pro tag )
    )
    return df[mask].sort_values("RSI").reset_index(drop=True) # Nach rsi geordnet 
# Aktie wird rausgeworfen und deshalb reihenfolge falsch -> neue nummerierug mit drop = True

def filter_momentum(df: pd.DataFrame) -> pd.DataFrame:
    """
    Momentum aktien - stark in allen zeitfenstern
    trend following setup
    """
    mask = (
        (df["1M %"] > 3) & 
        (df["3M %"] > 8) & 
        (df["6M %"] > 15) & 
        (df["Trend Score"] == 5) & # über allen MAs
        (df["RSI"] < 75)  # doch nicht extrem überkauft  
    )
    return df[mask].sort_values("6M %", ascending=False).reset_index(drop=True)


def filter_near_52w_high(df: pd.DataFrame,
                         pct: float = -5.0) -> pd.DataFrame: # nur aktien behalten, die max 5% vom hochb entfernt sind 
    """
    Aktien nahe 52 wochen hoch - breakout setup
    stärke zeigt sich oft an neuen highs 
    """
    mask = (
        (df["% vom Hoch"] >= pct) &
        (df["Trend Score"] >= 4) &
        (df["Vol Ratio"] > 1.2) # überdurschnittliches Volumen 
    )
    return df[mask].sort_values("% vom Hoch", ascending=False).reset_index(drop=True)

def filter_high_volume(df: pd.DataFrame,
                       ratio: float = 2.0) -> pd.DataFrame:
    """
    Ungewöhnlich hohes Volumen - mögliche News, institutional Activity.
    Volumen ist der Footprint des Smart Money
    """
    mask = (df["Vol Ratio"] > ratio)
    return df[mask].sort_values("Vol Ratio", ascending=False).reset_index(drop=True)

def filter_custom(df: pd.DataFrame,
                  rsi_min: float = 0,
                  rsi_max: float = 100,
                  ret_1m_min: float = -100,
                  trend_min: int = 0,
                  vol_min: int = 0) -> pd.DataFrame:
    """Vollständig anpassbare Filter"""
    mask = (
        (df["RSI"] >= rsi_min) &
        (df["RSI"] <= rsi_max) &
        (df["1M %"] >= ret_1m_min) &
        (df["Trend Score"] >= trend_min) &
        (df["Ø Volumen"] >= vol_min)
    )
    return df[mask].reset_index(drop=True)

# --- Visualisierung ---

def plot_screener_results(df: pd.DataFrame,
                          title: str = "Screener Ergebnisse") -> None:
    """
    Scatter Plot: RSI vs 1M performance 
    Quadranten zeigen sofort, welche Aktien interessant sind
    """
    if df.empty:
        print("Keine Ergebnisse für diesen Filter")
        return
    
    # Farbe nach trend score 
    colors = df["Trend Score"].map({
        5: "#16a34a",   # alle MAs — stark bullish
        4: "#86efac",
        3: "#fbbf24",
        2: "#f97316",
        1: "#ef4444",
        0: "#dc2626"
    }).fillna("#94a3b8")

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=df["RSI"],
        y=df["1M %"],
        mode = "markers+text",
        text = df["Ticker"],
        textposition = "top center",
        textfont=dict(size=10),
        marker=dict(
            size=df["Trend Score"] * 4 + 8,
            color = colors,
            opacity = 0.8,
            line=dict(width=1, color="white")
        ),
        hovertemplate=(
        "<b>%{text}</b><br>"
        "RSI: %{x:.1f}<br>"
        "1M Return: %{y:.1f}%<br>"
        "<extra></extra>"
        )
    ))

    # Quadrantenlinien
    fig.add_vline(x=50, line_dash="dot", line_color="#94a3b8", opacity=0.5)
    fig.add_hline(y=0, line_dash="dot", line_color="#94a3b8", opacity=0.5)
    fig.add_vline(x=70, line_dash="dash", line_color="#ef4444", opacity=0.4)
    fig.add_vline(x=30, line_dash="dash", line_color="#16a34a", opacity=0.4)

    #Quadranten beschriften
    fig.add_annotation(x=20, y=df["1M %"].max() * 0.9,
                       text="Überverkauft +<br>Stark",
                       showarrow=False, font=dict(color="#16a34a", size=11))
    fig.add_annotation(x=80, y=df["1M %"].max() * 0.9,
                       text="Momentum<br>Zone",
                       showarrow=False, font=dict(color="#2563eb", size=11))
    fig.add_annotation(x=80, y=df["1M %"].min() * 0.9,
                       text="Überkauft +<br>Schwach",
                       showarrow=False, font=dict(color="#ef4444", size=11))
    
    fig.update_layout(
        title=title,
        xaxis_title="RSI (14)",
        yaxis_title="1M Performance (%)",
        template="plotly_white",
        height=550,
        margin=dict(l=0, r=0, t=40, b=0)
    )

    fig.show()

def plot_top_momentum(df: pd.DataFrame, top_n: int = 10) -> None:
    """Bar Chart der top momentum aktien über verschiedene Zeiträume"""
    if df.empty:
        return
    
    top = df.nlargest(top_n, "6M %")

    fig = go.Figure()

    for col, color, name in [
        ("1M %",  "#93c5fd", "1 Monat"),
        ("3M %",  "#3b82f6", "3 Monate"),
        ("6M %",  "#1d4ed8", "6 Monate"),
    ]:
        fig.add_trace(go.Bar(
            name=name,
            x=top["Ticker"],
            y=top[col],
            marker_color=color,
            text=[f"{v:.1f}%" for v in top[col]],
            textposition="outside",
            textfont=dict(size=9)
        ))

    fig.update_layout(
        barmode="group",
        title=f"Top {top_n} Momentum Aktien",
        yaxis_title="Performance (%)",
        template="plotly_white",
        height=420,
        legend=dict(orientation="h", y=1.02),
        margin=dict(l=0,r=0,t=40, b=0)
    )

    fig.show()

# --- Alles zusammen ---

if __name__ == "__main__":

    # --- 1. EINSTELLUNGEN ---
    LOAD_FROM_CSV = True # Auf False wenn morgens frische Daten ziehen, sonst zum updaten auf True 
    CSV_FILENAME = "screener_results_20260413_1207.csv" # echten Dateinamen des heutigen Tages eingeben

    # --- 2. Daten beschaffen oder laden ---
    if LOAD_FROM_CSV:
        print(f"Lade gespeicherte Daten aus {CSV_FILENAME}...")
        df = pd.read_csv(CSV_FILENAME)
    else:
        print("Starte frischen S&P 500 Screener...")
        tickers, sp500_info = get_sp500_tickers()

        #kompletten s&p 500 screenen

        df = run_screener(tickers, max_stocks =503, delay=0.3)

        # Für den ersten Run: 50 Aktien ( dauert ca. 3 min)
        # Später: max_stocks=503 für kompletten s&p 500

        # Optional: ergebnisse speichern damit du nicht immer neu laden musst
        timestamp = datetime.now().strftime("%Y%m%d_%H%M")
        df.to_csv(f"screener_results_{timestamp}.csv", index=False)
        print(f"\nErgebnisse gespeichert: screener_results_{timestamp}.csv")

    # --- Filter anwenden ---
    print("\n" + "=" * 50)
    print("ÜBERVERKAUFTE AKTIEN (RSI < 35, Uptrend)")
    print("="*50)
    oversold = filter_oversold(df, rsi_max=35)
    print(oversold[["Ticker", "Kurs", "RSI",
                    "1M %", "Trend Score"]].to_string(index=False))
    
    print("\n" + "=" *50)
    print("MOMENTUM AKTIEN")
    print("=" *50)
    momentum = filter_momentum(df)
    print(momentum[["Ticker", "Kurs", "RSI",
                    "1M %", "3M %", "6M %"]].to_string(index=False))
    
    print("\n" + "="*50)
    print("NAHE 52-WOCHEN HOCH (max. -5%)")
    print("="*50)
    near_high = filter_near_52w_high(df)
    print(near_high[["Ticker", "Kurs",
                    "% vom Hoch", "Vol Ratio"]].to_string(index=False))
    
    print("\n" + "="*50)
    print("UNGEWÖHNLICHES VOLUMEN (2x Durchschnitt)")
    print("="*50)
    high_vol = filter_high_volume(df)
    print(high_vol[["Ticker", "Kurs",
                     "Vol Ratio", "RSI", "1M %"]].to_string(index=False))
    
    # --- Charts ---
    plot_screener_results(momentum, "Momentum Screener - S&P 500")
    plot_top_momentum(momentum)
