"""
Day 20 - Pairs Trading & Cointegration 

die idee dahinter:
    Zwei aktien die wirtschaftlich zusammenhängen
    (Coca-Cola / Pepsi, Gold / Silber, Shell / BP)
    bewegen sich langfristig zusammen.

    Wenn sie sich kurzfristig trennen → Mean Reversion erwartet.
    Du kaufst die günstigere, verkaufst (shortest) die teurere.
    Wenn sie zurückkonvergieren → Profit.

    Das ist marktunabhängig. Bull oder Bear — egal.
    Das nennt sich Market-Neutral Strategy.

    Hedge Funds wie Renaissance Technologies haben damit
    Milliarden verdient. Du baust es heute in Python.

Mathematik dahinter:
    Cointegration: zwei nicht-stationäre Zeitreihen deren
    Linearkombination stationär ist.
    Einfach gesagt: sie driften zusammen auch wenn
    sie einzeln random walks sind.
    
    Z-Score: wie weit ist die Spread vom Mittelwert?
    Entry bei Z > 2 oder Z < -2 (2 Standardabweichungen)
    Exit bei Z → 0 (Konvergenz)


"""

import yfinance as yf 
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy import stats # normalverteilung und standardisieren x-mü / sigma 
from statsmodels.tsa.stattools import coint, adfuller 
# tsa = time series analysis - ökonometrie teil 
# coint = prüft ob zwei zeitreihen cointegrated sind
# adfuller = prüft ob eine zeitreihe stationär ist
# stationär = keine trend und keine saisonalität
from statsmodels.regression.linear_model import OLS
# OLS aus öko -> klassische lineare regression
from statsmodels.tools import add_constant
# ohne würde die Regression eine Linie durch den ursprung machen aber so konstanta 
# b -> b+ ax1 + ax2...
from itertools import combinations
# n über k um alle möglichkeiten zu berechnen
import warnings
warnings.filterwarnings("ignore")

"""
---- 1. Code ----
"""

def load_data(tickers: list,
              period:  str = "5y") -> pd.DataFrame:
    """Lädt mehrere Ticker gleichzeitig."""
    df = yf.download(tickers, period=period,
                     auto_adjust=True, progress=False)["Close"]
    # progress sorgt dafür, das er beim laden eines Tickers nicht 1/20 completed schreibt 
    if len(tickers) == 1:
        df = df.to_frame(name=tickers[0])
    df.columns = df.columns.get_level_values(0)
    return df.dropna()


"""
---- Stationaritäts-Test ----
"""

def adf_test(series: pd.Series,
             name: str = "") -> dict:
    
    """
    Augmented Dickey-Fuller Test — ist die Zeitreihe stationär?

    Was ist Stationarität?
        Eine stationäre Zeitreihe hat konstanten Mittelwert
        und konstante Varianz über die Zeit.
        Aktienkurse sind NICHT stationär (random walk).
        Returns SIND stationär (annähernd).

    Warum das für Pairs Trading wichtig ist:
        Der Spread zweier cointegrierten Aktien ist stationär.
        Das bedeutet er kehrt immer zum Mittelwert zurück.
        Das ist die mathematische Grundlage unserer Strategie.

    ADF Null-Hypothese: Zeitreihe ist NICHT stationär.
    p-value < 0.05 → Null-Hypothese ablehnen → stationär.
    """
    result = adfuller(series.dropna(), autolag="AIC") # autolag -> bestraft zu viele parameter 
    # Adfuller gibt ein Tuple zurück -> wie eine schachtel mit verschiedenen fächern
    """
    result [0] ADF-Statistik der wert selbst
    result[1] p-wert 
    """
    output = {
        "name":       name,
        "adf_stat":   round(result[0], 4), # neg. Wert -> je negativer desto stärker Evidenz gegen H0
        "p_value":    round(result[1], 4), # p=0.003 nur 0.3% wsk das H0 stimmt -> stationär
        "stationary": result[1] < 0.05, # vergleicht hier result mit 0.05 evidenz 
        "critical_1": round(result[4]["1%"], 4), # ADF-Stat muss kleiner sein als das für 99% sicherheit
        "critical_5": round(result[4]["5%"], 4), # adf stat muss kleiner sein als das für 95% sicherheit
    }

    # adf muss kleiner als critical sein um x% Sicherheit zu haben -> wie normalvert
    # critical 5% = -2.86 -> 95% stationär auf 5% Niveau 

    status = "✅ Stationär" if output["stationary"] \
             else "❌ Nicht Stationär"
    print(f"  ADF Test — {name}: {status} "
          f"(p={output['p_value']:.4f})")

    return output

def cointegration_test(series_a: pd.Series,
                       series_b: pd.Series,
                       name_a:   str = "A",
                       name_b:   str = "B") -> dict:
    """
    Engle-Granger Cointegration Test.

    Was testet das?
        Ob eine Linearkombination der beiden Serien stationär ist.
        Konkret: series_a - hedge_ratio × series_b = stationär?

    p-value < 0.05 → Cointegration vorhanden → Pair ist handelbar.
    p-value > 0.05 → Kein stabiles Verhältnis → Pair vermeiden. # sie cointegraten nicht 

    Hedge Ratio:
        Regression von A auf B gibt den Koeffizienten.
        Du kaufst 1 Einheit A und shortest hedge_ratio Einheiten B.
        Damit ist die Position delta-neutral (marktunabhängig).
    """
    # Engle-Granger Test
    score, p_value, _ = coint(series_a, series_b) # coint gibt 3 wert und 3. wird ignoriert
    # 1. OLS Regression von A auf B
    # 2. ADF-Test auf die Residuen 

    # Hedge Ratio via OLS Regression
    X   = add_constant(series_b) # spalte mit length hinzufügen 
    reg = OLS(series_a, X).fit() # OLS definiert Modell mit series_a (unabhängig), x(abhängig))
    # fit() = schätzen -> Y = a * bX indem summe der quad. resiuden 
    hedge_ratio = reg.params.iloc[1] # B ist die hedge ratio also die konstante mit der a mult. werden muss für b 
    #eine hegge ration von 0.87 -> kaufe 1 aktie von x und short 0.87 von y
    intercept   = reg.params.iloc[0] # a -> achsenabschnitt
    r_squared   = reg.rsquared # wie gut erklärt x das y? (0-1) also die stärke der beziehung 

    # Spread berechnen
    spread = series_a - hedge_ratio * series_b - intercept
    # spread_t = A_t - b*B_t - a

    # ADF auf Spread
    spread_adf = adfuller(spread.dropna(), autolag="AIC")
    # testet ob der spread stationär ist

    """
    Wenn beide p-value < 0.05 starke Evidenz gegen H0, dass die beiden Serien stationär sind.
    """

    result = {
        "pair":         f"{name_a}/{name_b}", # schaut sich 2 werte an 
        "coint_score":  round(score, 4), # erstellt den coint-score
        "p_value":      round(p_value, 4), # < 0.05 -> cointegriert
        "cointegrated": p_value < 0.05, # True or False 
        "hedge_ratio":  round(hedge_ratio, 4), # wieviel aktien von x shorten wenn y kauf
        "intercept":    round(intercept, 4), # a aus der Regression 
        "r_squared":    round(r_squared, 4), # stärke erklärungskraft
        "spread_adf_p": round(spread_adf[1], 4), # Schaut ob Spread stationär 
        "spread_mean":  round(spread.mean(), 4), # Mittelwert des Spreads (ca 0 nach Regression)
        "spread_std":   round(spread.std(), 4), # Standardabweichung des Spreads
    }


    """
series_a (AAPL) ──┐
                  ├─→ coint() ──→ p_value: cointegriert?
series_b (MSFT) ──┘
                  │
                  ├─→ OLS() ───→ hedge_ratio β + intercept α
                  │
                  ├─→ spread = A - β·B - α
                  │
                  └─→ adfuller(spread) ──→ Double-Check Stationarität
                                           spread_mean + spread_std
                                           → später: Z-Score → Trade Signal
    """



    status = "✅ Cointegrated" if result["cointegrated"] \
             else "❌ Not Cointegrated"
    print(f"  {name_a}/{name_b}: {status} "
          f"(p={p_value:.4f}, "
          f"hedge={hedge_ratio:.3f}, "
          f"R²={r_squared:.3f})")

    return result

def compute_spread(series_a:    pd.Series,
                   series_b:    pd.Series,
                   hedge_ratio: float,
                   intercept:   float = 0,
                   window:      int   = 60) -> pd.DataFrame:
    """
    Berechnet Spread und Z-Score für ein Pair.

    Spread = A - hedge_ratio × B - intercept

    Z-Score:
        (Spread - Rolling Mean) / Rolling Std

    Warum rolling statt gesamt?
        Das Verhältnis zwischen zwei Aktien verändert sich
        langsam über Zeit. Rolling Window passt sich an.
        60 Tage = guter Balance zwischen Anpassung und Stabilität.

    ------------ Trading Signale: ----------------
        Z > +2.0  → A zu teuer vs B → Short A, Long B
        Z < -2.0  → A zu günstig vs B → Long A, Short B
        Z → 0     → Konvergenz → Position schließen
        |Z| > 3.0 → Warnsignal → Pair könnte auseinanderbrechen
    """
    spread = series_a - hedge_ratio * series_b - intercept

    roll_mean = spread.rolling(window).mean() # rolling rollt mit der zeit mit 
    roll_std  = spread.rolling(window).std()

    z_score = (spread - roll_mean) / roll_std # standardisierungsformel
    # formatiert spread in eine einheitslose zahl -> 1.0 = 1 standardabweichung

    # Half-Life: wie lange dauert es von Z=" auf Z=0 zu kommen?"
    # Ornstein-Uhlenbeck Prozess: AR(1) Regression Formel 
    spread_lag  = spread.shift(1).dropna()
    spread_diff = spread.diff().dropna()

    aligned = pd.concat([spread_lag, spread_diff], axis=1).dropna()
    # man brauch beide serien gleich lang und zeitlich aligned -> concat + dropna
    if len(aligned) > 10:
        reg        = stats.linregress( # erstellt die regression
            aligned.iloc[:, 0],
            aligned.iloc[:, 1]
        )
        lambda_val = -reg.slope
        half_life  = np.log(2) / lambda_val if lambda_val > 0 else np.inf 
        # lambda_val > 0 -> der wert konvergiert gegen den anderen
    else:
        half_life = np.inf # wie lange bis der spread auf die hälfte seines aktuellen abstands zum mittel zurück ist

    return pd.DataFrame({
        "spread":      spread,
        "roll_mean":   roll_mean, # 60 Tage mittelwert des spreads
        "roll_std":    roll_std, # 60 Tage standardabweichung
        "z_score":     z_score.round(4), # signal
        "half_life":   half_life, # skalare Zahl 
    })

def backtest_pairs(series_a:     pd.Series,
                   series_b:     pd.Series,
                   hedge_ratio:  float,
                   intercept:    float = 0,
                   entry_z:      float = 2.0, # trade wird erst bei z = 2 eröffnet 
                   exit_z:       float = 0.5, # bei 0.5 wird trade geschlossen 
                   stop_z:       float = 3.5, # aktien brechen auseinadner stop loss
                   window:       int   = 60,
                   capital:      float = 10_000,
                   commission:   float = 0.001) -> dict:
    """
    Vollständiger Pairs Trading Backtest.

    Position Setup:
        Long A / Short B wenn Z < -entry_z
            → A ist relativ günstig vs B
        Short A / Long B wenn Z > +entry_z
            → A ist relativ teuer vs B

    Exit:
        |Z| < exit_z  → Spread hat konvergiert → Profit nehmen
        |Z| > stop_z  → Spread divergiert zu stark → Stop Loss

    Capital Allocation:
        50% in Long-Leg, 50% in Short-Leg
        Dollar-neutral: Long und Short in Dollar gleich groß

    Commission:
        Beide Seiten zahlen Commission (4 Trades pro Roundtrip)

        Stop Loss          Entry           Mittelwert        Entry          Stop Loss
        ←─────────────────┼─────────────────┼─────────────────┼─────────────────→
       -3.5             -2.0             0.0             +2.0             +3.5

                     Long A/Short B ←─────────────────→ Short A/Long B
                                     Exit bei ±0.5
    """
    spread_df = compute_spread(
        series_a, series_b, hedge_ratio, intercept, window
    )
    z_score = spread_df["z_score"]

    equity       = [capital]# portfolio wert über zeit 
    dates        = [series_a.index[0]]
    trades       = []       # alle abgeschlossenen Trades 
    position     = 0        # +1 Long A/Short B | -1 Short A/Long B
    entry_z_val  = 0.0      # z-score beim anstieg 
    entry_date   = None
    entry_price_a = 0.0     # -> Kaufpreis inkl. Commission für pnl
    entry_price_b = 0.0
    shares_a     = 0        # Wie viele aktien halten wir?
    shares_b     = 0
    cash         = capital  # Verfügbares Cash 

    for i in range(window + 1, len(series_a)): # ersten 60 tage kein z score 
        date    = series_a.index[i]
        z       = float(z_score.iloc[i - 1])  # Signal von gestern -> Look ahead bias
        price_a = float(series_a.iloc[i]) # Preis von heute 
        price_b = float(series_b.iloc[i])

        if np.isnan(z):
            equity.append(equity[-1]) # kein signal -> equity bleibt gleich 
            dates.append(date)
            continue

        # --- Position schließen ---
        close_signal = (
            position != 0 and (
                abs(z) < exit_z or      # Spread konvergiert -> profit nehmen exit bei 0.5
                abs(z) > stop_z         # Spread divergiert zu stark -> Stop Loss
            )
        )

        if close_signal:
            # Long A / Short B schließen
            #
            #Entry: Kaufe 100 AAPL à 150$ = 15.000$  ausgegeben
            #       Shorte 87 MSFT à 172$ = 14.964$  eingenommen
            #       Netto Entry Cost = 36$

            #Exit:   Verkaufe 100 AAPL à 158$ = 15.800$ eingenommen
            #        Kaufe zurück 87 MSFT à 168$ = 14.616$ ausgegeben
            #        Netto Exit = 1.184$

            #PnL = 1.184$ - 36$ = 1.148$ Gewinn ✅
            if position == 1:
                proceeds_a = shares_a * price_a * (1 - commission) # verkauf a : comission abziehen 
                cost_b     = shares_b * price_b * (1 + commission) # Rückkauf B: comission drauf 
                pnl        = (proceeds_a - cost_b) - \
                             (entry_price_a * shares_a -
                              entry_price_b * shares_b)
                cash      += proceeds_a - cost_b

            # Short A / Long B schließen also andersrum 
            else:
                proceeds_b = shares_b * price_b * (1 - commission)
                cost_a     = shares_a * price_a * (1 + commission)
                pnl        = (proceeds_b - cost_a) - \
                             (entry_price_b * shares_b -
                              entry_price_a * shares_a)
                cash      += proceeds_b - cost_a

            exit_reason = "Konvergenz" if abs(z) < exit_z \
                          else "Stop Loss"

            trades.append({
                "entry_date":  entry_date,
                "exit_date":   date,
                "position":    "Long A/Short B" if position == 1
                               else "Short A/Long B",
                "entry_z":     round(entry_z_val, 3), # z beim einstieg
                "exit_z":      round(z, 3),           # z beim ausstieg
                "pnl":         round(pnl, 2),
                "pnl_pct":     round(pnl / (capital * 0.5) * 100, 2), # pnl in % vom eingesetzeten kapital
                "duration":    (date - entry_date).days,
                "exit_reason": exit_reason, # Konvergenz oder Stopp Loss
            })

            position  = 0 # zurück auf flat 
            shares_a  = 0
            shares_b  = 0

        # --- Neue Position öffnen ---
        if position == 0: # nur wenn wir flat sind und keine position offen ist
            alloc = cash * 0.45   # 45% pro Seite (Rest als Buffer)
            # commission kosten und margin anforderungen beim shorten 

            if z < -entry_z:
                # Long A / Short B
                shares_a    = int(alloc / price_a) # wie viele ganze aktien 
                shares_b    = int(alloc / price_b)
                cost        = (shares_a * price_a * (1 + commission) - # Kauf A teurer
                               shares_b * price_b * (1 - commission)) # Short B günstiger
                cash       -= cost
                position    = 1
                entry_z_val = z
                entry_date  = date
                entry_price_a = price_a * (1 + commission)
                entry_price_b = price_b * (1 - commission)
                # ---- Comission Logik ----
                #   Kaufen  → (1 + 0.001) → du zahlst 0.1% mehr
                #   Shorten → (1 - 0.001) → du erhältst 0.1% weniger
            elif z > entry_z:
                # Short A / Long B
                shares_a    = int(alloc / price_a)
                shares_b    = int(alloc / price_b)
                cost        = (shares_b * price_b * (1 + commission) -
                               shares_a * price_a * (1 - commission))
                cash       -= cost
                position    = -1
                entry_z_val = z
                entry_date  = date
                entry_price_a = price_a * (1 - commission)
                entry_price_b = price_b * (1 + commission)

        # Equity Update
        if position == 1:
            open_value = (shares_a * price_a -
                          shares_b * price_b)
        elif position == -1:
            open_value = (shares_b * price_b -
                          shares_a * price_a)
        else:
            open_value = 0

        equity.append(round(cash + open_value, 2))
        dates.append(date)

    equity_series = pd.Series(equity, index=dates)
    trades_df     = pd.DataFrame(trades)

    return {
        "equity":    equity_series,
        "trades":    trades_df,
        "spread_df": spread_df,
        "capital":   capital,
    }

"""
--- COMISSIONS ---
- Gebühr pro Trade 
Aktie kostet 100$

Kaufen:    100 × 1.001 = 100.10$  → Broker nimmt 0.10$
Verkaufen: 100 × 0.999 =  99.90$  → Broker nimmt 0.10$

ENTRY:          EXIT:
Kauf A    ──→   Verkauf A
Short B   ──→   Rückkauf B -> 4 Beine bei Pairs Trade 
"""

def screen_pairs(tickers:    list,
                 period:     str   = "5y",
                 p_threshold: float = 0.05) -> pd.DataFrame:
    """
    Screent alle möglichen Ticker-Kombinationen auf Cointegration.

    Für N Ticker gibt es N×(N-1)/2 mögliche Paare. n über k
    10 Ticker → 45 Paare.
    20 Ticker → 190 Paare.

    Wir testen alle und geben die besten zurück.
    """
    print(f"Lade Daten für {len(tickers)} Ticker...")
    df = load_data(tickers, period)

    # Nur Ticker mit vollständigen Daten
    df = df.dropna(axis=1, thresh=int(len(df) * 0.95)) # eine spalte bleibt nur wenn sie 95% der werte hat
    # es wird nur ticker b genommen wenn er 95% von ticker a hat
    available = df.columns.tolist()

    pairs        = list(combinations(available, 2)) # kombinationen erzeugen 
    results      = []

    print(f"Teste {len(pairs)} mögliche Paare...")

    for ticker_a, ticker_b in pairs:
        series_a = df[ticker_a].dropna()
        series_b = df[ticker_b].dropna()

        # Align
        aligned = pd.concat(
            [series_a, series_b], axis=1
        ).dropna() # nur vollstændige daten bleiben

        if len(aligned) < 252: # mindestens 252 Tage da sonst nicht aussagekräftig 
            continue

        try:
            result = cointegration_test(
                aligned.iloc[:, 0],
                aligned.iloc[:, 1],
                ticker_a, ticker_b
            )

            # Half-Life berechnen
            spread_df = compute_spread(
                aligned.iloc[:, 0], # spalte 0 ist aktie x
                aligned.iloc[:, 1], # spalte 1 ist aktie y 
                result["hedge_ratio"],
                result["intercept"]
            )
            half_life = spread_df["half_life"].iloc[-1]

            result["half_life_days"] = round(
                float(half_life) if not np.isinf(half_life) # is.inf schaut ob unendlich ist 
                else 999, 1
            )
            result["tradeable"] = (
                result["cointegrated"] and
                5 < result["half_life_days"] < 120 # bedingungen die erfüllt sein müssen 
                # mindestens 5 Tage (transaktionskosten) und maximal 120 Tage (kapital zu lange ebunden)
            )
            results.append(result)

        except Exception:
            pass

    results_df = pd.DataFrame(results)

    if results_df.empty:
        print("Keine Ergebnisse gefunden.")
        return results_df

    # Sortieren: beste Pairs zuerst
    results_df = results_df.sort_values(
        "p_value"
    ).reset_index(drop=True)

    tradeable  = results_df[results_df["tradeable"]]
    print(f"\nHandelbare Pairs gefunden: {len(tradeable)}")

    return results_df

def plot_pair_analysis(series_a:     pd.Series,
                       series_b:     pd.Series,
                       spread_df:    pd.DataFrame,
                       name_a:       str,
                       name_b:       str,
                       entry_z:      float = 2.0) -> None:
    """
    4-Panel Pairs Analysis Dashboard.

    Panel 1: Normalisierte Kurse beider Aktien
    Panel 2: Spread mit Rolling Mean
    Panel 3: Z-Score + Entry/Exit Levels
    Panel 4: Z-Score Verteilung (sollte normalverteilt sein)
    """
    fig = make_subplots(
        rows=4, cols=1,
        shared_xaxes=False,
        vertical_spacing=0.06,
        row_heights=[0.30, 0.25, 0.25, 0.20],
        subplot_titles=[
            f"{name_a} vs {name_b} — Normalisiert",
            f"Spread ({name_a} - hedge × {name_b})",
            "Z-Score + Entry/Exit Signale",
            "Z-Score Verteilung"
        ]
    )

    # Panel 1: Normalisierte Kurse
    norm_a = series_a / series_a.iloc[0] * 100
    norm_b = series_b / series_b.iloc[0] * 100

    fig.add_trace(go.Scatter(
        x=series_a.index,
        y=norm_a.round(2),
        name=name_a,
        line=dict(color="#2563eb", width=2)
    ), row=1, col=1)

    fig.add_trace(go.Scatter(
        x=series_b.index,
        y=norm_b.round(2),
        name=name_b,
        line=dict(color="#ef4444", width=2)
    ), row=1, col=1)

    # Panel 2: Spread
    fig.add_trace(go.Scatter(
        x=spread_df.index,
        y=spread_df["spread"].round(4),
        name="Spread",
        line=dict(color="#8b5cf6", width=1.5)
    ), row=2, col=1)

    fig.add_trace(go.Scatter(
        x=spread_df.index,
        y=spread_df["roll_mean"].round(4),
        name="Roll. Mean",
        line=dict(color="#f59e0b", width=1.5,
                  dash="dash")
    ), row=2, col=1)

    # Spread Bänder (±1 Std)
    upper_band = spread_df["roll_mean"] + spread_df["roll_std"]
    lower_band = spread_df["roll_mean"] - spread_df["roll_std"]

    fig.add_trace(go.Scatter(
        x=spread_df.index,
        y=upper_band.round(4),
        name="+1σ",
        line=dict(color="#94a3b8", width=1, dash="dot")
    ), row=2, col=1)

    fig.add_trace(go.Scatter(
        x=spread_df.index,
        y=lower_band.round(4),
        name="-1σ",
        line=dict(color="#94a3b8", width=1, dash="dot"),
        fill="tonexty",
        fillcolor="rgba(148,163,184,0.08)"
    ), row=2, col=1)

    # Panel 3: Z-Score
    z = spread_df["z_score"]

    z_colors = []
    for val in z:
        if val > entry_z:
            z_colors.append("#ef4444")
        elif val < -entry_z:
            z_colors.append("#16a34a")
        else:
            z_colors.append("#94a3b8")

    fig.add_trace(go.Scatter(
        x=z.index,
        y=z.round(3),
        name="Z-Score",
        line=dict(color="#2563eb", width=1.5)
    ), row=3, col=1)

    # Entry/Exit Levels
    for level, color, label in [
        (entry_z,  "#ef4444", f"Short A (Z={entry_z})"),
        (-entry_z, "#16a34a", f"Long A (Z=-{entry_z})"),
        (0,        "#94a3b8", "Mean"),
        (3.5,      "#dc2626", "Stop Loss"),
        (-3.5,     "#dc2626", "Stop Loss"),
    ]:
        fig.add_hline(
            y=level,
            line_dash="dash" if level != 0 else "dot",
            line_color=color,
            opacity=0.6,
            row=3, col=1
        )

    # Entry Signale markieren
    long_entries  = z[z < -entry_z]
    short_entries = z[z > entry_z]

    if not long_entries.empty:
        fig.add_trace(go.Scatter(
            x=long_entries.index,
            y=long_entries.values,
            mode="markers",
            name="Long Entry",
            marker=dict(
                color="#16a34a", size=8,
                symbol="triangle-up",
                line=dict(width=1, color="white")
            )
        ), row=3, col=1)

    if not short_entries.empty:
        fig.add_trace(go.Scatter(
            x=short_entries.index,
            y=short_entries.values,
            mode="markers",
            name="Short Entry",
            marker=dict(
                color="#ef4444", size=8,
                symbol="triangle-down",
                line=dict(width=1, color="white")
            )
        ), row=3, col=1)

    # Panel 4: Z-Score Histogramm
    z_clean = z.dropna()
    x_range = np.linspace(z_clean.min(), z_clean.max(), 100)

    # Normalverteilung als Referenz
    norm_curve = stats.norm.pdf(
        x_range, z_clean.mean(), z_clean.std()
    )
    norm_curve = norm_curve / norm_curve.max() * \
                 (len(z_clean) / 20)

    fig.add_trace(go.Histogram(
        x=z_clean,
        nbinsx=50,
        name="Z Verteilung",
        marker_color="#3b82f6",
        opacity=0.7,
        showlegend=False
    ), row=4, col=1)

    fig.add_trace(go.Scatter(
        x=x_range,
        y=norm_curve,
        name="Normal Ref",
        line=dict(color="#ef4444", width=2),
        showlegend=False
    ), row=4, col=1)

    fig.update_layout(
        height=900,
        template="plotly_white",
        hovermode="x unified",
        legend=dict(orientation="h", y=1.02),
        margin=dict(l=0, r=0, t=60, b=0)
    )

    fig.update_yaxes(title_text="Normalisiert", row=1, col=1)
    fig.update_yaxes(title_text="Spread ($)",   row=2, col=1)
    fig.update_yaxes(title_text="Z-Score",      row=3, col=1)
    fig.update_yaxes(title_text="Häufigkeit",   row=4, col=1)

    fig.show()


def plot_pairs_backtest(result: dict,
                        name_a: str,
                        name_b: str) -> None:
    """
    Backtest Ergebnisse visualisieren.
    """
    equity   = result["equity"]
    trades   = result["trades"]
    capital  = result["capital"]

    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=[
            "Equity Curve",
            "P&L pro Trade ($)",
            "Trade Duration (Tage)",
            "Exit Reasons"
        ],
        vertical_spacing=0.12,
        horizontal_spacing=0.10
    )

    # Equity Curve
    fig.add_trace(go.Scatter(
        x=equity.index,
        y=equity.round(2),
        name="Equity",
        line=dict(color="#2563eb", width=2),
        fill="tozeroy",
        fillcolor="rgba(37,99,235,0.08)"
    ), row=1, col=1)

    fig.add_hline(
        y=capital,
        line_dash="dot",
        line_color="#94a3b8",
        opacity=0.6,
        row=1, col=1
    )

    if not trades.empty and "pnl" in trades.columns:
        completed = trades.dropna(subset=["pnl"])

        # P&L Bars
        pnl_colors = [
            "#16a34a" if v > 0 else "#ef4444"
            for v in completed["pnl"]
        ]
        fig.add_trace(go.Bar(
            x=list(range(len(completed))),
            y=completed["pnl"].round(2),
            marker_color=pnl_colors,
            name="Trade P&L",
            showlegend=False
        ), row=1, col=2)

        fig.add_hline(
            y=0, line_color="#1e293b",
            line_width=1, row=1, col=2
        )

        # Duration Histogram
        if "duration" in completed.columns:
            fig.add_trace(go.Histogram(
                x=completed["duration"],
                nbinsx=20,
                name="Duration",
                marker_color="#8b5cf6",
                opacity=0.8,
                showlegend=False
            ), row=2, col=1)

        # Exit Reasons
        if "exit_reason" in completed.columns:
            reasons = completed["exit_reason"].value_counts()
            reason_colors = {
                "Konvergenz": "#16a34a",
                "Stop Loss":  "#ef4444",
            }
            fig.add_trace(go.Bar(
                x=reasons.index.tolist(),
                y=reasons.values.tolist(),
                marker_color=[
                    reason_colors.get(r, "#94a3b8")
                    for r in reasons.index
                ],
                text=reasons.values.tolist(),
                textposition="outside",
                showlegend=False
            ), row=2, col=2)

    fig.update_layout(
        height=600,
        template="plotly_white",
        title=f"Pairs Trading — {name_a}/{name_b}",
        margin=dict(l=0, r=0, t=60, b=0)
    )

    fig.update_yaxes(title_text="Kapital ($)",  row=1, col=1)
    fig.update_yaxes(title_text="P&L ($)",      row=1, col=2)
    fig.update_yaxes(title_text="Anzahl",       row=2, col=1)
    fig.update_yaxes(title_text="Anzahl",       row=2, col=2)

    fig.show()

def compute_pairs_metrics(result: dict,
                          name:   str) -> dict:
    """Performance-Analyse für Pairs Trading."""
    equity  = result["equity"].dropna()
    trades  = result["trades"]
    capital = result["capital"]

    if len(equity) < 5:
        return {}

    returns   = equity.pct_change().dropna()
    years     = len(equity) / 252
    total_ret = (equity.iloc[-1] / equity.iloc[0] - 1) * 100
    cagr      = ((equity.iloc[-1] / equity.iloc[0])
                  ** (1/max(years, 0.1)) - 1) * 100

    sharpe = (returns.mean() / returns.std() *
              np.sqrt(252)) if returns.std() > 0 else 0

    rolling_max = equity.cummax()
    max_dd      = ((equity - rolling_max) /
                    rolling_max).min() * 100

    if not trades.empty and "pnl" in trades.columns:
        completed = trades.dropna(subset=["pnl"])
        wins      = completed[completed["pnl"] > 0]
        losses    = completed[completed["pnl"] <= 0]
        win_rate  = len(wins) / len(completed) * 100 \
                    if len(completed) > 0 else 0
        pf        = (
            wins["pnl"].sum() / abs(losses["pnl"].sum())
            if not losses.empty and
            losses["pnl"].sum() != 0 else 0
        )
        avg_dur   = completed["duration"].mean() \
                    if "duration" in completed.columns else 0
        conv_rate = (
            completed[completed["exit_reason"] == "Konvergenz"]
            .shape[0] / len(completed) * 100
            if "exit_reason" in completed.columns else 0
        )
    else:
        win_rate = pf = avg_dur = conv_rate = 0

    return {
        "Pair":              name,
        "Total Return (%)":  round(total_ret, 2),
        "CAGR (%)":          round(cagr, 2),
        "Sharpe":            round(sharpe, 2),
        "Max DD (%)":        round(max_dd, 2),
        "Win Rate (%)":      round(win_rate, 1),
        "Profit Factor":     round(pf, 2),
        "Avg Duration (d)":  round(avg_dur, 1),
        "Konvergenz Rate %": round(conv_rate, 1),
        "Trades":            len(trades),
    }


def print_pairs_tearsheet(metrics: dict) -> None:
    """Tearsheet im Terminal ausgeben."""
    print(f"\n{'='*48}")
    print(f"  PAIRS TRADING — {metrics.get('Pair', '')}")
    print(f"{'='*48}")
    for k, v in metrics.items():
        if k != "Pair":
            print(f"  {k:<24} {v:>15}")
    print(f"{'='*48}")

if __name__ == "__main__":

    print("Tag 20 — Pairs Trading & Cointegration")
    print("=" * 55)

    # --- Klassische Pairs ---
    CLASSIC_PAIRS = [
        ("KO",  "PEP",  "Coca-Cola / Pepsi"),
        ("GLD", "SLV",  "Gold / Silber"),
        ("XOM", "CVX",  "ExxonMobil / Chevron"),
        ("JPM", "BAC",  "JPMorgan / BofA"),
        ("MSFT","GOOGL","Microsoft / Alphabet"),
    ]

    # --- Stationaritäts-Tests ---
    print("\n1. ADF Tests — Einzelne Aktien")
    print("-" * 40)

    df_ko_pep = load_data(["KO", "PEP"], "5y")
    adf_test(df_ko_pep["KO"],           "KO (Preis)")
    adf_test(df_ko_pep["KO"].diff().dropna(), "KO (Returns)")
    adf_test(df_ko_pep["PEP"],          "PEP (Preis)")

    # --- Cointegration Tests ---
    print("\n2. Cointegration Tests — Klassische Pairs")
    print("-" * 40)

    coint_results = []
    for ticker_a, ticker_b, pair_name in CLASSIC_PAIRS:
        try:
            data = load_data([ticker_a, ticker_b], "5y")
            if ticker_a in data.columns and \
               ticker_b in data.columns:
                result = cointegration_test(
                    data[ticker_a],
                    data[ticker_b],
                    ticker_a, ticker_b
                )
                coint_results.append(result)
        except Exception as e:
            print(f"  {ticker_a}/{ticker_b}: Fehler — {e}")

    # --- Bestes Pair backtesten ---
    if coint_results:
        best = min(coint_results, key=lambda x: x["p_value"])
        name_a, name_b = best["pair"].split("/")

        print(f"\n3. Bestes Pair: {best['pair']}")
        print(f"   p-value:     {best['p_value']:.4f}")
        print(f"   Hedge Ratio: {best['hedge_ratio']:.4f}")
        print(f"   R²:          {best['r_squared']:.4f}")

        data     = load_data([name_a, name_b], "5y")
        series_a = data[name_a]
        series_b = data[name_b]

        spread_df = compute_spread(
            series_a, series_b,
            best["hedge_ratio"],
            best["intercept"]
        )

        hl = spread_df["half_life"].iloc[-1]
        print(f"   Half-Life:   {hl:.1f} Tage")
        print(f"   → Mean Reversion dauert ca. "
              f"{hl:.0f} Tage")

        # Pair Analysis Chart
        plot_pair_analysis(
            series_a, series_b,
            spread_df,
            name_a, name_b
        )

        # Backtest
        print(f"\n4. Backtest: {best['pair']}")
        result = backtest_pairs(
            series_a, series_b,
            hedge_ratio = best["hedge_ratio"],
            intercept   = best["intercept"],
            entry_z     = 2.0,
            exit_z      = 0.5,
            stop_z      = 3.5,
            capital     = 10_000
        )

        metrics = compute_pairs_metrics(
            result, best["pair"]
        )
        print_pairs_tearsheet(metrics)
        plot_pairs_backtest(result, name_a, name_b)

    # --- Alle klassischen Pairs backtesten ---
    print("\n5. Alle Pairs im Vergleich")
    print("-" * 40)

    all_metrics = []
    for ticker_a, ticker_b, pair_name in CLASSIC_PAIRS:
        try:
            data = load_data([ticker_a, ticker_b], "5y")
            if ticker_a not in data.columns or \
               ticker_b not in data.columns:
                continue

            ct = cointegration_test(
                data[ticker_a], data[ticker_b],
                ticker_a, ticker_b
            )

            r = backtest_pairs(
                data[ticker_a], data[ticker_b],
                hedge_ratio = ct["hedge_ratio"],
                intercept   = ct["intercept"],
                capital     = 10_000
            )

            m = compute_pairs_metrics(
                r, f"{ticker_a}/{ticker_b}"
            )
            if m:
                all_metrics.append(m)

        except Exception as e:
            print(f"  {ticker_a}/{ticker_b}: {e}")

    if all_metrics:
        summary = pd.DataFrame(all_metrics).sort_values(
            "Sharpe", ascending=False
        )
        cols = ["Pair", "Total Return (%)",
                "CAGR (%)", "Sharpe",
                "Win Rate (%)", "Trades"]
        print("\nRANKING:")
        print(summary[cols].to_string(index=False))

    # --- Pair Screener auf Tech-Sektor ---
    print("\n6. Pair Screener — Tech Sektor")
    print("-" * 40)

    tech = ["AAPL", "MSFT", "GOOGL", "META",
            "NVDA", "AMD",  "INTC",  "QCOM"]

    screen = screen_pairs(tech, period="3y")

    if not screen.empty:
        tradeable = screen[screen["tradeable"]]
        if not tradeable.empty:
            print(f"\nHandelbare Pairs:")
            cols = ["pair", "p_value",
                    "hedge_ratio", "r_squared",
                    "half_life_days"]
            print(tradeable[cols].head(5).to_string(index=False))

            # Bestes Pair aus Screener backtesten
            best_screen = tradeable.iloc[0]
            s_a, s_b    = best_screen["pair"].split("/")

            print(f"\nBacktest bestes Screener-Pair: "
                  f"{best_screen['pair']}")

            data_s = load_data([s_a, s_b], "3y")
            if s_a in data_s.columns and \
               s_b in data_s.columns:

                r_s = backtest_pairs(
                    data_s[s_a], data_s[s_b],
                    hedge_ratio = best_screen["hedge_ratio"],
                    intercept   = best_screen["intercept"],
                    capital     = 10_000
                )
                m_s = compute_pairs_metrics(
                    r_s, best_screen["pair"]
                )
                print_pairs_tearsheet(m_s)

    # Export
    if coint_results:
        pd.DataFrame(coint_results).to_csv(
            "day20_cointegration_results.csv", index=False
        )
        print("\nGespeichert: day20_cointegration_results.csv")

    if all_metrics:
        pd.DataFrame(all_metrics).to_csv(
            "day20_pairs_backtest.csv", index=False
        )
        print("Gespeichert: day20_pairs_backtest.csv")