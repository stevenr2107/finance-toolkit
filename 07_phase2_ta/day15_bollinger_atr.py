"""
Day 15 - Bollinger Bands + ATR
Zwei professionelle Indikatoren von Scratch 
Bollinger Bands: Volatilitätskanal + Squeeze Detektor 
ATR: Marktvolatilität in dollar - basis für jeden stop loss
"""

import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings 
warnings.filterwarnings("ignore")

def load_data(ticker: str, period = "1y") -> pd.DataFrame:
    df = yf.download(ticker, period=period, auto_adjust=True, progress=False # fortschrittsbalken ausschalten) 
    )
    df.columns = df.columns.get_level_values(0) # hier wird alles in einem Index gespeichert statt in mehreren Spalten
    return df.dropna()

def bollinger_bands(series: pd.Series,
                    window: int = 20,
                    num_std: float = 2.0) -> pd.DataFrame: # Berechnet den abstand zwischen oberen und unteren 
    
    """
    Bollinger bands - volatitaetskanal um einen moving average 

    formel:
    Middle = sma(window)
    Upper = sma(window) + num_std * stdDev(window)
    Lower = sma(window) - num_std * stdDev(window)

    %B: Wo im band ist der kurs gerade?
    %B = (Price - Lower) / (Upper - Lower)
    > 1.0 -> über dem oberen band -> overbought
    = 0.5 -> im band -> neutral
    < 0.0 -> unter dem unteren band -> oversold

    Bandwidth: wie weit sind die bänder auseinander?
    eng = volatilität niedrig , squeeze möglich 
    weit = volatilität hoch , Trend läuft 
    """

    middle = series.rolling(window=window).mean()
    std = series.rolling(window=window).std()
    upper = middle + num_std * std
    lower = middle - num_std * std
    bandwidth = ((upper-lower) / middle * 100).round(4)
    pct_b = ((series - lower) / (upper - lower)).round(4)

    return pd.DataFrame({
        "bb_upper": upper.round(4),
        "bb_middle": middle.round(4),
        "bb_lower": lower.round(4),
        "bb_pct_b": pct_b,
        "bb_bandwidth": bandwidth,
    })


def detect_squeeze(series: pd.Series, 
                   window: int = 20, 
                   squeeze_pct: float = 0.15) -> pd.DataFrame:
    """
    BB Squeeze - das Setzp kurz vor einem explosiven move.

    Logik:

    Wenn die aktuelle Bandwith nahe am 6-Monats-Tief liegt
    -> Volatilität ist komprimiert -> Energie baut sich auf -> Squeeze

    Richtung des Ausbruchs:

    Momentum > 0 -> wahrscheinlich nach oben 
    Momentum < 0 -> wahrscheinlich nach unten

    Das ist kein geheimnis aber die wenigsten könnenn es quantifizieren. 
    """

    bb = bollinger_bands(series, window)
    min_bw = bb["bb_bandwidth"].rolling(126).min() 
    # schaut wo in den letzten 6 monaten die niedrigste bandwith war

    # Squeeze wenn aktuelle BW nahe am historischen Minimum 
    squeeze = bb["bb_bandwidth"] <= min_bw * (1+squeeze_pct) 
    # es wir dnicht nur gefragt ob gleich dem minimum ist 
    # sondern mit einer toleranz von 15 %
    momentum = series - series.rolling(window).mean() # aktueller preis minus durchschnitt
    # -> Auf/abwärtstrend? je nach wert pos oder neg.

    df = pd.DataFrame({
        "close": series,
        "bandwidth": bb["bb_bandwidth"],
        "squeeze": squeeze,
        "momentum": momentum.round(4),
        "pct_b": bb["bb_pct_b"],

    })

    n_squeeze = squeeze.sum()
    if n_squeeze > 0:
        print(f"Squeeze aktiv an {n_squeeze} von "
              f"{len(series)} Tagen ({n_squeeze / len(series) * 100:.1f}%)")
        print("aktueller status ",
              "🔴 SQUEEZE AKTIV" if squeeze.iloc[-1]
              else "✅ Kein Squeeze")

    return df

def atr(df: pd.DataFrame, window: int = 14) -> pd.Series:
    """
    Average True Range - misst volatilität in kurs einheiten 

    true range berücksichtigt gaps:
    TR = max(
        High - Low,  Normale Tagesrange 
        High - Previous Close, Gap nach oben 
        low - close_prev , Gap nach unten 
        )
    ATR = EMA(TR, window)

    Warum das wichtig ist:
        Ein 5% stop loss auf nvda ( ATR = 8%) -> du wirst beim ersten normalen Rauschen gestoppt
        Ein 5% stop auf JNJ (ATR = 1%) viel zu weit 

        ATR- basierter Stop: Entry - 2* ATR
        Passt sich automatisch an die volatilität an 
        Das ist professionelles Risk Management 
    """

    high = df["High"].squeeze()
    low = df["Low"].squeeze()
    close = df["Close"].squeeze()
    prev_close = close.shift(1)

    tr = pd.concat([ # drei true range szenarien werden in einer neuen tabelle mit 3 spalten angelegt 
        high -low, # normaler handelstag 
        (high - prev_close).abs(), # gap nach oben
        (low - prev_close).abs() # gap nach unten
    ], axis=1).max(axis=1) 

    return tr.ewm(com=window - 1, adjust = False).mean() #aus den letzten 14 tagen wird der durchschnitt berechnet

def atr_levels(df: pd.DataFrame, # sicherheitsnetz bei kauf aktie durch atr
               multiplier: float = 2.0, # durchschnittliche volatilität * multiplier 
               window: int = 14) -> pd.DataFrame:
    """
    ATR - basierter stop loss und take profit levels

    Long Setup:
        Stop Loss = Kurs - multiplier * ATR
        Take Profit = Kurs + multiplier * ATR * 1.5 (asymmetrisch)

    Risk/Reward:
        1:1.5 ist minimum für positive erwartungswert bei 40% win rate
        1:2.0 ist gut - du kannst in 34% der Fälle richtig liegen und trotzdem profitabel sein 
    """

    close = df["Close"].squeeze()
    atr_series = atr(df, window)

    stop_loss = close - multiplier * atr_series 
    # stop loss -> aktie fällt um 2* ihrer durchschnittl. Vol.
    take_profit = close + multiplier * atr_series * 1.5
    # take profit -> aktie steigt um 3* ihrer durchschnittl. Vol. -> 1:1.5 rr
    atr_pct = (atr_series / close * 100).round(3)

    # Risk/reward verhältnis 
    rr_ratio = (take_profit - close) / ( close - stop_loss)

    return pd.DataFrame({
        "close": close.round(2),
        "atr": atr_series.round(3),
        "atr_pct": atr_pct,
        "stop_loss": stop_loss.round(2),
        "take_profit": take_profit.round(2),
        "rr_ratio": rr_ratio.round(2),
    })

def plot_bb_atr(ticker: str, period: str = "6mo") -> None:
    """
    3-panel chart 
    1. Kurs + BB + ATR Stop Loss
    2. Bandwidth + Squeeze Signale 
    3. %B Position + Überkauft / *berverkauft Zonen
    """

    df = load_data(ticker, period)
    close = df["Close"].squeeze()

    bb = bollinger_bands(close)
    squeeze_df = detect_squeeze(close)
    atr_df = atr_levels(df)

    fig = make_subplots(
        rows=3,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.04,
        row_heights=[2.00, 0.25, 0.25],
        subplot_titles=[
            f"{ticker} - Bollinger Bands + ATR Stopp Loss",
            "Bandwidth (%) + Squeeze Detektor",
            "%B Position - Position im Band ",
        ]
    )

    # --- Panel 1 Kurs + BB + Stop Loss

    #BB Füllung zwischen upper und lower 
    fig.add_trace(go.Scatter(
        x=df.index,
        y=bb["bb_upper"],
        name="BB upper",
        line=dict(color="#3b82f6", width=1, dash="dot"),
    ), row=1, col=1)

    fig.add_trace(go.Scatter(
        x=df.index,
        y=bb["bb_lower"],
        name="BB lower",
        line=dict(color="#3b82f6", width=1, dash="dot"),
        fill="tonexty",
        fillcolor="rgba(59, 130, 246, 0.06)",
    ), row=1, col=1)

    # Middle Band
    fig.add_trace(go.Scatter(
        x=df.index,
        y=bb["bb_middle"],
        name="SMA 20",
        line=dict(color="#93c5fd", width=1.),
    ), row=1, col=1)

    # Kurs
    fig.add_trace(go.Scatter(
        x=df.index,
        y=close.round(2),
        name="Kurs",
        line=dict(color="#1e293b", width=2,),
    ), row=1, col=1)

    # ATR Stop Loss
    fig.add_trace(go.Scatter(
        x=df.index,
        y=atr_df["stop_loss"],
        name="ATR Stop (2x)",
        line=dict(color="#ef4444", width=1, dash="dash"),
        opacity=0.8
    ), row=1, col=1)

    # Squeeze Marker auf Kurs-Level
    squeeze_days = squeeze_df[squeeze_df["squeeze"]]
    if not squeeze_days.empty:
        sq_prices = close[squeeze_days.index]
        fig.add_trace(go.Scatter(
            x=squeeze_days.index,
            y=sq_prices * 0.995, # verdeckt sonst den kursverlauf 
            name="Squeeze",
            mode="markers",
            marker=dict(
                symbol="diamond",
                size=6,
                color="#f59e0b",
                line=dict(width=1, color="white")
            ),
        ), row=1, col=1)

    # --- Panel 2: Bandwidth 

    bw_colors= [
        "#ef4444" if sq else "#94a3b8"
        for sq in squeeze_df["squeeze"]
    ]

    fig.add_trace(go.Bar(
        x=df.index,
        y=squeeze_df["bandwidth"],
        name="Bandwidth",
        marker_color=bw_colors,
        opacity=0.8,
        showlegend=True
    ), row=2, col=1)

    # Durchschnittliche bandwidth als referent 
    avg_bw = squeeze_df["bandwidth"].mean()
    fig.add_hline(
        y=avg_bw,
        line_dash="dot",
        line_color="#64748b",
        opacity=0.6,
        row=2, col=1
    )

    # --- Panel 3: %B Position
    pct_b = bb["bb_pct_b"]

    pct_colors = [
        "#ef4444" if v > 1.0 
        else ("#16a34a" if v < 0.0 else "#3b82f6")
        for v in pct_b
    ]


    fig.add_trace(go.Scatter(
        x=df.index,
        y=pct_b,
        name="%B",
        line=dict(color="#8b5cf6", width=1.5),
    ), row=3, col=1)

    # Referenzlinie für %B
    for y_val, color, label in [
        (1.0, "#ef4444", "Über BB"),
        (0.5, "#94a3b8", "Mitte"),
        (0.0, "#16a34a", "Unter BB"),
    ]:
        fig.add_hline(
            y=y_val,
            line_dash="dot" if y_val == 0.5 else "dash",
            line_color=color,
            opacity=0.5,
            row=3, col=1
        )

    fig.update_layout(
        height=750,
        template="plotly_white",
        hovermode="x unified",
        legend=dict(orientation="h", y=1.02),
        margin = dict(l=0, r=0, t=50, b=0)
    )

    fig.update_yaxes(title_text="Kurs ($)", row=1, col=1)
    fig.update_yaxes(title_text="BW (%)",  row=2, col=1)
    fig.update_yaxes(title_text="%B",  row=3, col=1)

    fig.show()

def compare_atr(tickers: list) -> pd.DataFrame:
    """
    Vergleicht atr und vollinger bw quer über den ticker 
    beantwortet welche aktie ist gerade am volatilsten?
    und wo ist der atr-stop relativ zum aktuellen Kurs?
    """
    results = []

    for ticker in tickers:
        try:
            df = load_data(ticker, "3mo")
            close = df["Close"].squeeze()
            atr_df = atr_levels(df)
            bb = bollinger_bands(close)
            sq = detect_squeeze(close)
            
            current = float(close.iloc[-1])
            current_atr = float(atr_df["atr"].iloc[-1])
            current_stop = float(atr_df["stop_loss"].iloc[-1])
            current_tp = float(atr_df["take_profit"].iloc[-1])

            results.append({
                "Ticker": ticker,
                "Kurs": round(current, 2),
                "ATR ($)": round(current_atr, 2),
                "ATR (%)": round(float(atr_df["atr_pct"].iloc[-1]), 2),
                "Stop ($)": round(current_stop, 2),
                "Stop Abstand": f"{((current_stop/current)-1)* 100:.1f}%",
                "BB Width (%)": round(float(bb["bb_bandwidth"].iloc[-1]), 2),
                "%B": round(float(bb["bb_pct_b"].iloc[-1]), 2),
                "Squeeze": "🔴 Ja" if sq["squeeze"].iloc[-1] else "✅ Nein",
            })
        
        except Exception as e:
            print(f"{ticker}: Fehler - {e}")

    df_result = pd.DataFrame(results).sort_values(
        "ATR (%)", ascending=False
    ).reset_index(drop=True)
    return df_result


if __name__ == "__main__":

    TICKER = "KO"

    print(f"Tag 15 - Bollinger bands & ATR {TICKER}")
    print( "=" * 50)

    df = load_data(TICKER, "1y")
    close = df["Close"].squeeze()

    #--- Bollinger Bands ---
    bb = bollinger_bands(close)
    print("\nBollinger Bands - letzte 5 Tage:")
    print(bb[["bb_upper", "bb_middle", "bb_lower", "bb_pct_b"]].tail().round(2))

    current_pct_b = bb["bb_pct_b"].iloc[-1]
    print(f"\nAktuelle %B: {current_pct_b:.3f}")
    if current_pct_b > 0.8:
        print(" Kurs nahe am oberen Band - überkauft Zone")
    elif current_pct_b < 0.2:
        print(" Kurs nahe am unteren Band - überverkauft Zone")
    else:
        print(" Kurs im Band - neutral Zone")

    #--- Squeeze ---
    print("\n--- BB Squeeze Analyse ---")
    squeeze_df = detect_squeeze(close)

    #--- ATR ---
    atr_df = atr_levels(df)
    print("\nATR Levels heute:")
    latest = atr_df.iloc[-1]
    print(f"  Kurs:         ${latest['close']:.2f}")
    print(f"  ATR (14):     ${latest['atr']:.2f} "
          f"({latest['atr_pct']:.2f}% des Kurses)")
    print(f"  Stop Loss:    ${latest['stop_loss']:.2f}")
    print(f"  Take Profit:  ${latest['take_profit']:.2f}")
    print(f"  R/R Ratio:    1:{latest['rr_ratio']:.2f}")

    #--- Chart ---
    plot_bb_atr(TICKER, "4y")

    #--- Multi Ticker ATR Vergleich ---
    print("\n" + "=" * 55)
    print("ATR & SQUEEZE VERGLEICH")
    print("=" * 55)

    universe = ["AAPL", "NVDA", "TSLA", "MSFT", "JPM", "SPY", "GME"]
    comparison = compare_atr(universe)
    print(comparison.to_string(index=False))

    # Wer hat den engsten ATR-Stop? (relativ zum Kurs)
    best_risk = comparison.nsmallest(1, "ATR (%)")
    print(f"\nGeringste Volatilität: {best_risk['Ticker'].iloc[0]}")
    print(f"→ Engster Stop möglich ohne viel Rauschen zu riskieren")

    # Wer ist im Squeeze?
    squeezed = comparison[comparison["Squeeze"] == "🔴 Ja"]
    if not squeezed.empty:
        print(f"\nAktive Squeezes: {', '.join(squeezed['Ticker'].tolist())}")
        print("→ Explosive Moves erwartet — Richtung unklar bis zum Ausbruch")
    else:
        print("\nKeine aktiven Squeezes im Universe")

    # Export
    comparison.to_csv("day15_atr_comparison.csv", index=False)
    print("\nGespeichert: day15_atr_comparison.csv")

# Man sieht hier, das die aktien sehr oft am unteren band kaufen und am 
# oberen band drehen, da da gewinnmitnahmen stattfinden 

# Drei Erkenntnisse die dich von anderen abheben:
"""
ATR ist die Basis von professionellem Risk Management. 
Jeder Fixed-Prozent-Stop ist falsch. 
Ein Trader der TSLA mit demselben Stop wie JNJ handelt versteht Volatilität nicht. 
ATR löst das automatisch — kein Denken nötig, der Markt sagt dir wie weit der Stop sein muss.
%B ist ehrlicher als der Kurs selbst. 
Du siehst nicht wo der Kurs ist — du siehst wo er relativ zur aktuellen Volatilität ist. 
%B von 0.95 auf NVDA bedeutet etwas völlig anderes als 0.95 auf SPY. 
Das ist der Kontext den rohe Kurszahlen nicht haben.
Squeeze + Momentum-Richtung = einer der stärksten Setups. 
Die Energie baut sich auf, du weißt in welche Richtung der Druck zeigt, und wartest auf den Ausbruch. 
Das ist keine Raketenwissenschaft — aber die wenigsten quantifizieren es so klar wie du jetzt.
"""