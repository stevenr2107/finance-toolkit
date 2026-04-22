"""
Day 16  - VWAP, OBV & Stochastic oscillator
volumen- basierte indikatoren und preis-range analyse

Warum Volumen?
    Volumen ist der Footprint des Smart Money
    Preis lügt manchmal. Volumen lügt nie 
    Ein anstieg ohne volumen ist schwach  
"""

import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
import numpy as np
from plotly.subplots import make_subplots
import warnings 
warnings.filterwarnings("ignore")

def load_data(ticker: str, period: str = "1y") -> pd.DataFrame:
    df = yf.download(ticker, period=period, auto_adjust=True, progress=False # fortschrittsbalken ausschalten) 
    )
    df.columns = df.columns.get_level_values(0) # hier wird alles in einem Index gespeichert statt in mehreren Spalten
    return df.dropna()

def vwap(df: pd.DataFrame,
         window: int = None ) -> pd.Series:
    """
    Volume weighted average price - 
    der wichtigste institutionelle Benchmark 

    Formel: 
        Typical Price = (high + low + close) / 3
        VWAP = sum(Typical Price * Volume) / sum(Volume)

        warum instis vwap lieben:
            jede große order wird gegen vwap gemessen.
            "Ich habe besser als VWAP gekauft" = gute Ausführung 
            Goldman, Blackrock, Vanguard alle tracken VWAP

            wenn du weißt wo vwap ist, weißt du wo die 
            großen ihre orders platzieren 
            
        window=None -> kumulativer VWAP ( Standard für intraday)
        window=20 -> Rolling vwap (besser für tagesdaten)
     """
    typical_price = (
        df["High"].squeeze() +
        df["Low"].squeeze() +
        df["Close"].squeeze()
    ) / 3

    volume = df["Volume"].squeeze()
    tp_vol = typical_price * volume

    if window is None:
        return tp_vol.cumsum() / volume.cumsum()
    else:
        return (
            tp_vol.rolling(window).sum() / 
            volume.rolling(window).sum()
        )
    
def vwap_bands(df: pd.DataFrame,
               window: int=20,
               num_std: float = 1.5) -> pd.DataFrame:
    """
    VWAP mit standardabweichungs-Bändern 

    Ähnlich wie bollingert bands aber volumengewichtet 
    instis nutzen diese bänder als 
    dynamische support/resistance levels 

    kurs kehrt vom unteren vwap band um -> bullischer reversal 
    Kurs brucht über oberes vwap band -> starkes momentum
    """
    vwap_series = vwap(df, window)
    typical_price = (
        df["High"].squeeze() +
        df["Low"].squeeze() +
        df["Close"].squeeze()
    ) / 3

    volume = df["Volume"].squeeze()

    # Volumengewichtete Varianz 
    variance = (
        ((typical_price - vwap_series) ** 2 * volume).rolling(window).sum() / volume.rolling(window).sum()
    )
    std = np.sqrt(variance)

    upper = vwap_series + num_std * std
    lower = vwap_series - num_std * std

    # Kurs vw vwap in % 
    close = df["Close"].squeeze()
    vwap_dist = ((close - vwap_series) / vwap_series * 100).round(3)

    return pd.DataFrame({
        "vwap": vwap_series.round(4),
        "vwap_upper": upper.round(4),
        "vwap_lower": lower.round(4),
        "vwap_dist": vwap_dist
    })

def obv(df: pd.DataFrame) -> pd.Series:
    """
    On balance Volume - Volumen-Momentum-Indikator

    Logik:
        Kurs steigt heute -> OBV += heutiges Volumen 
        Kurs fallt heute -> OBV -= heutiges Volumen
        Kurs gleich -> OBV bleibt gleich

    Was obv dir zeigt:
        OBV steigt mit Kurs -> Trend bestätigt. Volumen folgt preis.
        OBV steigt, kurs fällt -> Bullische divergenz. Kurs folgt bald.
        OBV fällt, Kurs steigt -> Bärische Divergenz. Warnsignal

    Divergenzen sind das wertvollste Signal das OBV gibt.
    Wenn kurs und OBV sich trennen - einer von beiden lügt 
    Fast immer ist es der Kurs der lügt 
    """
    close = df["Close"].squeeze()
    volume = df["Volume"].squeeze()
    direction = np.sign(close.diff()).fillna(0)
    return (direction * volume).cumsum()

def obv_analysis(df: pd.DataFrame,
                 smooth: int = 20) -> pd.DataFrame:
    
    """
    OBV mit Signal Linie und divergenz detektor 

    signal-Linie = EMA(OBV, smooth)
    Histogram = OBV- Signal 

    Wenn Histogram positiv und wächst -> Volumen Momentum Bullish 
    Wenn Histogram negativ und sinkt -> Volumen Momentum Bearish

    Divergenz Detektor:
        Preis macht neues Hoch aber OBV nicht -> bearish divergence
        Preis macht neues Tief aber OBV nicht -> bullish divergence
    """
    close = df["Close"].squeeze()
    obv_series = obv(df)
    signal = obv_series.ewm(span=smooth, adjust=False).mean()
    histogram = obv_series - signal

    # Divergenz erkennen - Rolling Window von 20 Tagen 
    w = 20
    price_high = close.rolling(w).max()
    obv_high = obv_series.rolling(w).max()

    price_low = close.rolling(w).min()
    obv_low = obv_series.rolling(w).min()

    # Bullishe divergenz  Kurs nahe Tief, OBV nicht 

    bullish_div = ((close <= price_low * 1.02) & 
                   (obv_series < obv_low * 1.05)
    )
    # bearishe divergenz  Kurs nahe Hoch, OBV nicht
    bearish_div = ((close >= price_high * 0.98) & 
                   (obv_series < obv_high * 0.95))

    return pd.DataFrame({
        "obv": obv_series,
        "obv_signal": signal,
        "obv_hist": histogram,
        "bullish_div": bullish_div,
        "bearish_div": bearish_div
    })

def stochastic(df: pd.DataFrame,
               k_window: int=14,
               d_window: int=3) -> pd.DataFrame:
    """
    Stochastic Oscillator - wo ist der kurs im range der letzten n tage?

    Formel:
        %K = (Close - lowestLow(N)) / (HighestHigh(N) - LowestLow(N)) * 100
        %D = SMA(%K, d_window) <- Signal Linie 

    Interpretation:
        %K > 80 -> überkauft
        %K < 20 -> überverkauft
        %K kreuzt %D von unten -> bullish crossover 
        %K kreuzt %D von oben -> bearish crossover

    Vorteil vs. RSI:
        RSI misst Geschwindigkeit der Preisänderung 
        Stochastic misst Position im Preis-Range
        Kombiniert: wenn beide übereinstimmen ist das Signal stark 
    """
    close = df["Close"].squeeze()
    high = df["High"].squeeze()
    low = df["Low"].squeeze()

    lowest_low = low.rolling(k_window).min()
    highest_high = high.rolling(k_window).max()

    # 1e-10 verhindetr division durch null bei flachen märkten 
    k = (
        (close - lowest_low) /
        (highest_high - lowest_low + 1e-10) * 100
    )
    d= k.rolling(d_window).mean()

    # Crossover Signale 
    prev_k = k.shift(1)
    prev_d = d.shift(1)
    bull_cross = (k>d) & (prev_k <= prev_d) & (k < 30)
    bear_cross = (k<d) & (prev_k >= prev_d) & (k > 70)

    return pd.DataFrame({
        "stoch_k": k.round(2),
        "stoch_d": d.round(2),
        "bull_cross": bull_cross,
        "bear_cross": bear_cross,
    })

def stochastic_rsi(series: pd.Series,
                   rsi_window: int = 14,
                   stoch_window: int = 14,
                   k_smooth: int = 3,
                   d_smooth: int = 3) -> pd.DataFrame:
    """
    Stochastic rsi - stochastig angewendet auf den rsi wert.

    viel sensitiver als normaler stochastic 
    gibt frühere signale aber auch mehr false Positives 
    Gut für kurzfristiges Trading , nicht für langfristige Trends.

    Formel: 
        RSI berechnen 
        --- Stochastic auf RSI anwenden statt auf preis ---
    """
    # RSI 
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.ewm(com=rsi_window - 1, adjust=False).mean()
    avg_loss = loss.ewm(com=rsi_window - 1, adjust=False).mean()

    rs = avg_gain / avg_loss
    rsi_vals = 100 - (100 / (1 + rs))

    # Stochastic auf RSI 
    lowest_rsi = rsi_vals.rolling(stoch_window).min()
    highest_rsi = rsi_vals.rolling(stoch_window).max()

    stoch_rsi_k = (
        (rsi_vals - lowest_rsi) /
        (highest_rsi - lowest_rsi + 1e-10) * 100
    )
    stoch_rsi_d = stoch_rsi_k.rolling(d_smooth).mean()

    return pd.DataFrame({
        "rsi": rsi_vals.round(2),
        "stoch_rsi_k": stoch_rsi_k.round(2),
        "stoch_rsi_d": stoch_rsi_d.round(2)
    })

# --- Volumen analyse ---

def volume_analysis(df: pd.DataFrame, window: int = 20) -> pd.DataFrame:
    """
    Analysiert volumen annomalien 

    volume ratio : heute vs durchschnitt
    > 2.0 = ungewöhnlich hohes Volumen -> mögliche News, institutional Activity.
    < 0.5 = ungewöhnlich niedriges Volumen -> kein Interesse, Vorsicht bei Signalen

    Positive Volumen Index (PVI):
        Ändert sich nur an tagen mit höherem Volumen als gestern 
        Zeigt was passiert wenn "das Volk" (dumb money) kauft (hohes Volumen)

        Negative Volume Index (NVI):
            Ändert sich nur an tagen mit niedrigerem Volumen
            Zeigt was Instis machen (niedrigeres Volumen = ruhig)
    """

    close = df["Close"].squeeze()
    volume = df["Volume"].squeeze()

    avg_vol = volume.rolling(window).mean()
    vol_ratio = (volume / avg_vol).round(3)
    vol_change = volume.pct_change()

    # PVI - steigt wenn Volumen steigt 
    pvi = pd.Series(index = df.index, dtype=float)
    pvi.iloc[0] = 1000
    for i in range(1, len(df)):
        if volume.iloc[i] > volume.iloc[i-1]:
            pvi.iloc[i] = pvi.iloc[i-1] * (1+close.pct_change().iloc[i])
        else:
            pvi.iloc[i] = pvi.iloc[i-1]

    # NVI - steigt wenn Volumen sinkt 
    nvi = pd.Series(index = df.index, dtype=float)
    nvi.iloc[0] = 1000
    for i in range(1, len(df)):
        if volume.iloc[i] < volume.iloc[i-1]:
            nvi.iloc[i] = nvi.iloc[i-1] * (1+close.pct_change().iloc[i])
        else:
            nvi.iloc[i] = nvi.iloc[i-1]

    # Volumen Anomalien flaggen 
    high_vol = vol_ratio > 2.0
    low_vol = vol_ratio < 0.5

    return pd.DataFrame({
        "volume": volume,
        "avg_volume": avg_vol.round(0),
        "vol_ratio": vol_ratio,
        "high_vol": high_vol,
        "low_vol": low_vol,
        "pvi": pvi.round(2),
        "nvi": nvi.round(2)
    })

# --- Vollständiges Dashboard ---

def plot_volume_dashboard(ticker: str,
                          period: str = "6mo") -> None:
    """
    4-Panel Volume Intelligence Dashboard.

    Panel 1: Kurs + VWAP Bänder 
    Panel 2: OBV + Divergenz Signale 
    Panel 3: Stochastic %K / %D mit Crossovers 
    Panel 4: Volumen + Anomalie-Markierung 
    """
    df = load_data(ticker, period)
    close = df["Close"].squeeze()

    vwap_df = vwap_bands(df)
    obv_df = obv_analysis(df)
    stoch_df = stochastic(df)
    vol_df = volume_analysis(df)

    fig = make_subplots(
        rows=4, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.04,
        row_heights=[0.38, 0.22, 0.22, 0.18],
        subplot_titles=[
            f"{ticker} — Kurs + VWAP Bänder",
            "OBV + Signal + Divergenzen",
            "Stochastic Oscillator (%K / %D)",
            "Volumen + Anomalien"
        ]
    )

    # Panel 1: Kurs + VWAP Bänder
    fig.add_trace(go.Scatter(
        x=df.index, y=vwap_df["vwap_upper"], name="VWAP Upper",
        line=dict(color="#f59e0b", width=1, dash = "dot")
    ), row=1, col=1)

    fig.add_trace(go.Scatter(
        x=df.index, 
        y=vwap_df["vwap_lower"], 
        name="VWAP Lower",
        line=dict(color="#f59e0b", width=1, dash = "dot"),
        fill ="tonexty",
        fillcolor = "rgba(245, 158, 11, 0.05)"
    ), row=1, col=1)

    fig.add_trace(go.Scatter(
        x = df.index,
        y=vwap_df["vwap"],
        name="VWAP",
        line=dict(color="#f59e0b", width=2)
    ),row=1, col=1)

    fig.add_trace(go.Scatter(
        x=df.index, 
        y=close.round(2),
        name = "Kurs",
        line=dict(color="#1e293b", width=2)
    ), row=1, col=1)

    # --- Panel 2: OBV ---
    fig.add_trace(go.Scatter(
        x=df.index, 
        y=obv_df["obv"],
        name="OBV",
        line=dict(color="#2563eb", width=1.5)
    ), row=2, col=1)

    fig.add_trace(go.Scatter(
        x=df.index, 
        y=obv_df["obv_signal"],
        name="OBV Signal",
        line=dict(color="#f59e0b", width=1.5)
    ), row=2, col=1)

    # Bullishe divergenzen 
    bull_divs = obv_df[obv_df["bullish_div"]]
    if not bull_divs.empty:
        fig.add_trace(go.Scatter(
            x=bull_divs.index, 
            y=obv_df.loc[bull_divs.index, "obv"],
            name="Bullish Div",
            mode="markers",
            marker=dict(color="#16a34a", size=10,
                        symbol="triangle-up",
                        line=dict(width=1, color="white"))
        ), row=2, col=1)

    # Bärishe Divergenzen 
    bear_divs = obv_df[obv_df["bearish_div"]]
    if not bear_divs.empty:
        fig.add_trace(go.Scatter(
            x=bear_divs.index, 
            y=obv_df.loc[bear_divs.index, "obv"],
            name="Bearish Div",
            mode="markers",
            marker=dict(color="#ef4444", size=10,
                        symbol="triangle-down",
                        line=dict(width=1, color="white"))
        ), row=2, col=1)

    # --- Panel 3: Stochastic ---
    fig.add_trace(go.Scatter(
        x=df.index, 
        y=stoch_df["stoch_k"],
        name="%K",
        line=dict(color="#8b5cf6", width=1.5)
    ), row=3, col=1)

    fig.add_trace(go.Scatter(
        x=df.index, 
        y=stoch_df["stoch_d"],
        name="%D Signal",
        line=dict(color="#f59e0b", width=1.5)
    ),row=3, col=1)

    # Crossover Marker 
    bull_cross = stoch_df[stoch_df["bull_cross"]]
    if not bull_cross.empty:
        fig.add_trace(go.Scatter(
            x=bull_cross.index, 
            y=stoch_df.loc[bull_cross.index, "stoch_k"],
            name="Bull Cross",
            mode="markers",
            marker=dict(color="#16a34a", size=10,
                        symbol="triangle-up",
                        line=dict(width=1, color="white"))
        ), row=3, col=1)

    bear_cross = stoch_df[stoch_df["bear_cross"]]
    if not bear_cross.empty:
        fig.add_trace(go.Scatter(
            x=bear_cross.index, 
            y=stoch_df.loc[bear_cross.index, "stoch_k"],
            name="Bear Cross",
            mode="markers",
            marker=dict(color="#ef4444", size=10,
                        symbol="triangle-down",
                        line=dict(width=1, color="white"))
        ), row=3, col=1)

    for y_val, color in [(80, "#ef4444"), (20, "#16a34a")]:
            fig.add_hline(y=y_val, line_dash="dash", line_color=color,
                          opacity=0.5, row=3, col=1)
            
    # --- Panel 4: Volumen + Anomalien ---
    vol_colors = []
    for i in range(len(df)):
        if vol_df["high_vol"].iloc[i]:
            vol_colors.append("#f59e0b") # anomalie gelb
        elif close.iloc[i] >= df["Open"].squeeze().iloc[i]:
            vol_colors.append("#16a34a") # Up-Tag grün
        else:
            vol_colors.append("#ef4444") # Down Tag rot

    fig.add_trace(go.Bar(
        x=df.index,
        y=vol_df["volume"],
        name="Volumen",
        marker_color=vol_colors,
        opacity=0.8
    ), row=4, col=1)

    # Durchschnitt
    fig.add_trace(go.Scatter(
        x=df.index,
        y=vol_df["avg_volume"],
        name="Durchschnittl. Volumen",
        line=dict(color="#94a3b8", width=1.5, dash="dot"),
    ), row=4, col=1)

    fig.update_layout(
        height=850,
        template = "plotly_white",
        hovermode="x unified",
        legend=dict(orientation="h", y=1.02),
        margin=dict(l=0, r=0, t=50, b=0)
    )

    fig.update_yaxes(title_text="Kurs ($)",  row=1, col=1)
    fig.update_yaxes(title_text="OBV",       row=2, col=1)
    fig.update_yaxes(title_text="Stoch %",   row=3, col=1,
                     range=[0, 100])
    fig.update_yaxes(title_text="Volumen",   row=4, col=1)

    fig.show()

# --- Signal Kombination ---

def combined_volume_signal(df: pd.DataFrame) -> pd.DataFrame:
    """
    Kombiniert alle volumen signale zu einem score 

    score logik ( 0-5 punkte bullish, 0- -5 bearish):
        +1 Kurs über VWAP
        +1 OBV über Signal linie 
        +1 Stochastic %K unter 30 (überverkauft)
        +1 bullisge OBV-Divergenz erkannt
        +1 Volumen überdurchschnittlich bei Up-Tag

    Score > 3 -> starkes bullishes Setup
    Score < -3 -> starkes bearishes Setup
    """

    close = df["Close"].squeeze()
    opens = df["Open"].squeeze()
    volume = df["Volume"].squeeze()

    vwap_val = vwap(df, window=20)
    obv_df = obv_analysis(df)
    stoch_df = stochastic(df)
    vol_df = volume_analysis(df)

    score = pd.Series(0, index=df.index)

    # +1 Kurs über VWAP
    score += (close > vwap_val).astype(int)

    # +1 OBV ober Signal linie
    score += (obv_df["obv"] > obv_df["obv_signal"]).astype(int)

    # +1 Stochastic überverkauft
    score += (stoch_df["stoch_k"] < 30).astype(int)

    # +1 bullishe OBV-Divergenz erkannt
    score += (obv_df["bullish_div"]).astype(int)

    # -1 Bärishe Divergenz 
    score -= obv_df["bearish_div"].astype(int)

    # +1 hohes Volumen  bei Up-Tag
    up_day = close >= opens
    score += ((vol_df["vol_ratio"] > 1.5) & up_day).astype(int) 

    # -1 hohes Volumen bei down tag 
    score -= ((vol_df["vol_ratio"] > 1.5) & ~up_day).astype(int)

    result = pd.DataFrame({
        "close": close.round(2),
        "score": score,
        "signal": score.apply(
            lambda x: "Stark Bullish" if x >= 3
                 else ("Bullish"      if x >= 1
                 else ("Neutral"      if x == 0
                 else ("Bearish"      if x >= -2
                 else  "Stark Bearish")))    
        ),
        "vwap_dist": ((close - vwap_val) / vwap_val * 100).round(2),
        "vol_ratio": vol_df["vol_ratio"],
        "stoch_k": stoch_df["stoch_k"],
    })

    return result 


def print_signal_summary(ticker: str,
                         result: pd.DataFrame) -> None:
    """Gibt den aktuellen Signal-Status im Terminal aus."""
    latest = result.iloc[-1]

    print(f"\n{'='*45}")
    print(f"  VOLUME SIGNAL SUMMARY — {ticker}")
    print(f"{'='*45}")
    print(f"  Signal:          {latest['signal']}")
    print(f"  Score:           {latest['score']:+.0f} / 5")
    print(f"  Kurs:            ${latest['close']:.2f}")
    print(f"  VWAP Abstand:    {latest['vwap_dist']:+.2f}%")
    print(f"  Volumen Ratio:   {latest['vol_ratio']:.2f}x")
    print(f"  Stochastic %K:   {latest['stoch_k']:.1f}")
    print(f"{'='*45}")

    # Letzte 5 Signale
    print("\n  Letzte 5 Tage:")
    recent = result[["close", "score",
                      "signal"]].tail(5)
    print(recent.to_string())

# --- Main ---
if __name__ == "__main__":

    TICKER = "AAPL"

    print(f"Tag 16 - Volume Intelligence: {TICKER}")
    print("=" * 50)

    df = load_data(TICKER, "1y")
    close = df["Close"].squeeze()

    # --- VWAP ---
    vwap_df = vwap_bands(df, window=20)
    current = float(close.iloc[-1])
    vwap_val = float(vwap_df["vwap"].iloc[-1])
    vwap_dist = float(vwap_df["vwap_dist"].iloc[-1])

    print(f"\nVWAP (20) ${vwap_val:.2f}")
    print(f"Kurs:       ${current:.2f}")
    print(f"Abstand:    {vwap_dist:+.2f}%")
    print("->", "Über VWAP - Käufer in Kontrolle"
          if vwap_dist > 0
          else "Unter VWAP - Verkäufer in Kontrolle")
    

    # --- OBV ---
    obv_df = obv_analysis(df)
    print(f"\nOBV aktuell: {obv_df['obv'].iloc[-1]:,.0f}")
    print(f"OBV Signal: {obv_df['obv_signal'].iloc[-1]:,.0f}")
    print(f"Bullish Divs: {obv_df['bullish_div'].sum()}")
    print(f"Bearish Divs: {obv_df['bearish_div'].sum()}")

    # --- Stochastic ---
    stoch_df = stochastic(df)
    k_val = stoch_df["stoch_k"].iloc[-1]
    d_val = stoch_df["stoch_d"].iloc[-1]

    print(f"\nStochastic %K: {k_val:.1f}")
    print(f"Stochastic %D: {d_val:.1f}")
    if k_val > 80:
        print(" -> Überkauft Zone")
    elif k_val < 20:
        print(" -> Überverkauft Zone")
    else:
        print("-> Neutrale Zone")

    #--- StochRSI ---
    srsi = stochastic_rsi(close)
    print(f"\nStoch RSI %K: {srsi['stoch_rsi_k'].iloc[-1]:.1f}")
    print(f"RSI aktuell: {srsi['rsi'].iloc[-1]:.1f}")

    # --- Kombiniertes Signal ---
    signal_df = combined_volume_signal(df)
    print_signal_summary(TICKER, signal_df)

    # --- Chart ---
    plot_volume_dashboard(TICKER, "6mo")

    # --- Multi-Ticker Vergleich ---
    print("\n" + "="*55)
    print("VOLUME SIGNAL VERGLEICH - Magnificent 7")
    print("="*55)

    mag7 =["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA"]
    rows = []

    for t in mag7:
        try:
            d = load_data(t, "3mo")
            sig = combined_volume_signal(d)
            latest = sig.iloc[-1]

            vw = vwap_bands(d)
            stoch = stochastic(d)

            rows.append({
                "Ticker": t,
                "Score": f"{latest['score']:+.0f}",
                "Signal": latest["signal"],
                "VWAP Dist": f"{latest['vwap_dist']:+.1f}%",
                "Vol Ratio": f"{latest['vol_ratio']:.2f}x",
                "Stoch %K": f"{stoch['stoch_k'].iloc[-1]:.0f}"
            })
        except Exception as e:
            print(f" {t}: Fehler - {e}")

    import pandas as pd 
    result_df = pd.DataFrame(rows)
    print(result_df.to_string(index=False))

    # Export 
    result_df.to_csv("day16_volume_signals.csv", index = False)
    print("\nGespeichert: day16_volume_signals.csv")