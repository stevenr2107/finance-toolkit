"""
Day 18 - Options Greeks & Black scholes modell

Warum das wichtig ist:
    options volumen übersteigt aktienvolumen täglich
    instis hedgen damit jede große position
    retail verliert weil sie greeks nicht verstehem 

    Nach heute weißt du:
    - Was eine option wirklich wert ist (Black Scholes)
    - Wie sensibel der preis auf veränderungen reagiert (Greeks)
    - Wie du implied Volatility berechnest ( der Markt konsens)
    - wie du ein options-portfolio riskierst 

Mathematik level:
    mittel: jede zeile wird erklärt 
"""

import yfinance as yf 
import pandas as pd
import numpy as np
from scipy.stats import norm # Normalverteilung (norm.cdf(1.96) = 0.975 -> 95% Konfidenzintervall)
from scipy.optimize import brentq
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings("ignore")

# Eine Option ist eine Wette darauf, das ein Kurs einen bestimmten strike überschreitet.
# Black scholes berechnet die wahr5scheinlichkeit dass das passiert- und multipliziert mit gewinn
def black_scholes(S: float, # jetziger kurs
                  K: float, # Kaufpreis
                  T: float, # zeit bis expiration
                  r: float,
                  sigma: float,
                  option_type: str = "call") -> float:
    
    """
    Black scholes options pricing model.

    Parameter:
        S: Underlying asset price (aktueller kurs)
        K: Strike price (zb. 150$)
        T: Zeit bis expiration in jahren (30Tage = 30/365)
        r: risikoloser zinssatz (z.B 0.05 für 5%)
        sigma: Volatility des Underlying (0.3 für 30%)
        option_type: Option type (call or put)

    Intuition hinter der formel:
        eine call option ist wertlos wenn der kurs nie über den strike steugt.
        Black scholes berechnet due wahrscheinlichkeit dass das passiert -
        gewichtet mit dem erwarteten gewinn wenn es passiert.

        d1 = wie weit ist der kurs vom strike entfernt angepasst für vvolatility und zeit
        d2 = Wahrscheinlichkeit dass option im geld endet

        N(d1), N(d2) = kumulative Normalverteilung 
        Das ist der Kern : Optionspreise sind wahrscheinlichkeitsberechnungen 

    """
    if T <= 0:
        # Am verfallstag. nur intrinsicher Wert
        if option_type == "call":
            return max(S - K, 0)# entweder die option ist im geld
        else:
            return max(K - S, 0)# oder man verliert alles bis 0
        
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    # wie weit vom kurs entfernt  np.log(S / K)
    # r * T risikoloser zins - geld hat zeitwert
    # 0.5 * sigma² * T Korrektur: Volatilität schiebt den Erwartungswert
    # sigma * sqrt(T) Normierung — macht d1 zu einer Standardnormalzahl
    d2 = d1 - sigma * np.sqrt(T) # wsk das option am ende im geld ist 
    
    if option_type == "call":
        price = (S *norm.cdf(d1) - K* np.exp(-r * T) * norm.cdf(d2))
        # preis = was man bekommt - was man zahlt 
            #   = wsk s>k bei exp - barwert strike abgezinst auf heute
    else:
        price = (K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1))
        # man profitiert wen S < K
    return round(price, 4) # 4 da in cent. 

# vektorisieren von balck scholes
def black_scholes_batch(S: float,
                        K: np.ndarray, # array = [110, 155, 160]
                        T: float,
                        r: float,
                        sigma: float,
                        option_type: str = "call") -> np.ndarray:
    """
    Vektorisiertes Black-Scholes für mehrere Strikes gleichzeitig.
    Nützlich für Options Chains.
    """
    if T <= 0:
        if option_type =="call":
            return np.maximum(S - K, 0) # np.maximum da array
        else:
            return np.maximum(K - S, 0) 
    
    d1 = (np.log(S/K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 -sigma * np.sqrt(T)

    if option_type == "call":
        return (S * norm.cdf(d1) - K* np.exp(-r * T) * norm.cdf(d2))
    else: 
        return (K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1))
    
"""
Black-Scholes fragt: Wie viel ist es heute wert, das Recht zu haben, morgen zu einem festen Preis zu kaufen?
und beantwortet das mit Wahrscheinlichkeitsrechnung über die Normalverteilung. 
scipy.stats.norm.cdf ist das Werkzeug, das diese Wahrscheinlichkeiten liefert.
"""

# Black scholes gibt dir einen preis. Die greeks sagen dir wie fragil dieser preis ist 

def compute_greeks(
        S: float,
        K: float,
        T: float,
        r: float,
        sigma: float,
        option_type: str = "call")-> dict:
    """
    Alle fünf greeks - das risikoprofil einer option 

    DELTA (Δ): zeigt hebel an 

        Wie viel ändert sich der options preis
        wenn der kurs um 1$ steigt?
        Call delta: 0 bis +1
        put delta: -1 bis 0
        Delta = 0.5 -> Option ist "at the money"
        Delta = 0.9 -> Option verhält sich fast wie die aktie 

    GAMMA (Γ): Risiko des hebels 
    
    Der hebel der AKtie bewegt sich mit der aktie und das gamma misst dies  

    -> wie schell ändert sich delta wenn aktie um 1$ steigt? (Wie die 2. Ableitung)und die 1. ableitung von delta 
        wie schnell ändert sich delta?
        Hoch bei ATM Optionen kurz vor Expiration.
        Gamma-Risk: hohe Gamma = Delta ändert sich schnell 
        = gefährlich wenn man es nicht versteht 

    THETA (Θ): wieviel verliert die option pro tag allein durch zeit 

        Zeitwert-Verlust pro Tag. 
        Immer negativ für Käufer - Zeit ist dein Feind
        "theta - decay" - warum 90% der Retail-Optionskäufer verlieren 
        Eine option verliert täglich an Wert allein durch Zeitablauf.

    VEGA (Ω): wieviel teurer wird die option bei +1% volatilität? (IV Crush)

        Sensitivität gegenüber Volatilitätsänderungen
        Preis Änderung bei 1% mehr volatilität 
        vor earninngs: Vega explodiert (IV steigt). 
        Nach earnings: IV Cash - Vega kollabiert 

    RHO (ρ): Wie viel ändert sich der preis wenn risikolose Zins um 1% steigt -> Nur bei langen laufzeiten 
        Sensitivität gegenuber Zinsen
        Weniger wichtig für kurzfristige optionen 
        wichtig bei LEAPS (lange Laufzeiten)
    """
    if T <= 0: # option ist bereits abgelaufen
        return {g: 0.0 for g in ["delta", "gamma", "theta", "vega", "rho", "price"]}
    
    d1 = (np.log(S/K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    # Standard Formel PDF und CDF 
    n_d1 = norm.pdf(d1) # höhe glockenkurve an punkt d1
    N_d1 = norm.cdf(d1) # fläche links von d1
    N_d2 = norm.cdf(d2) # fläche links von d2
    N_nd1 = norm.cdf(-d1) # Fläche rechts von d1
    N_nd2 = norm.cdf(-d2)

    """
    CDF → norm.cdf(x) → Fläche unter der Kurve (Wahrscheinlichkeit)
    PDF → norm.pdf(x) → Höhe der Kurve an Punkt x (Dichte)


    PDF                        CDF
    n_d1 = Höhe hier          N_d1 = diese Fläche
          ↓                    ←————————————
    .    *                    1│         ___
    .   * *                   │        /
    .  *   *         →        │      /
    . *     *                 │    /
    .*_______*___             0│__/____________
              d1                          d1


    """

    # Preis 
    """
    → Du besitzt 1 Call mit Delta 0.5
    → Du bist "halb so exponiert" wie bei 1 Aktie
    → Um delta-neutral zu sein: Short 0.5 Aktien pro Option
    """
    if option_type == "call":
        price = S *N_d1 - K * np.exp(-r * T) * N_d2
        delta = N_d1 # zwischen 0 un +1
        rho = K * T * np.exp(-r * T) * N_d2 / 100
        # Delta misst wie viel der Preis steigt wenn der Kurs um 1$ steigt
        # 0.5 at the money -> 0.5 gewinn bei 1$ steigendem kurs
        # 0.1 deep out of the money - kaum reaktion 0.1 $ bei steigendem kurs
    else :
        price = K * np.exp(-r * T) * N_nd2 - S * N_nd1
        delta = -N_nd1 # zwischen -1 und 0 
        rho = -K * T * np.exp(-r * T) * N_nd2 / 100
    
    # Greeks die für Call und Put gleich sind 
    gamma = n_d1 / (S * sigma * np.sqrt(T)) # ableitung von delta 
    # kurz vor expiration wird np_sqrt klein 
    # -> Gamma explodiert 
    theta = (
        -S * n_d1 * sigma / (2 * np.sqrt(T))   # Zeitwert verfall durch Volatilität
        - r * K * np.exp(-r * T) *  # zeitwert durch risikolosen Zins 
        (N_d2 if option_type =="call" else N_nd2) )/ 365  
    """
    Theta-Verlust pro Tag:
    Tag 90: -$0.02
    Tag 30: -$0.05
    Tag 7:  -$0.15   ← beschleunigt sich
    Tag 1:  -$0.80   ← explodiert
    Deshalb ist der letzte Monat vor Expiration für Option-Käufer am gefährlichsten. 
    Theta Seller (Prämien verkaufen) nutzen genau das aus.
    """

    vega = S * n_d1 * np.sqrt(T) / 100 # wieviel ändert sich mein optionspreis wenn die vola um 1% steigt 
    # vegas = 0.15 -> option wird 0.15$ teurer wenn vola um 1% steigt 

    """
    Vor Earnings:  IV = 60%, Option kostet $5.00
    Nach Earnings: IV = 30%, Option kostet $2.50  ← IV Crush
    Aktie bewegt sich +2% → trotzdem Verlust
    """

    return {
        "price": round(price, 4),
        "delta": round(delta, 4),
        "gamma": round(gamma, 6),
        "theta": round(theta, 4),
        "vega":  round(vega, 4),
        "rho":   round(rho, 4),
        "d1":    round(d1, 4),
        "d2":    round(d2, 4),
    }

"""
Implied volatility dreht black scholes um 
Blackscholes: Eingabe s,k,t,r,sigma -> Preis # wir wollen den preis aber nicht wissen 

man will wissen, welche volatilität (sigma) der markt einpreist 
IV: Eingabe s,k,t,r,marktpreis -> sigma
"""
def implied_volatility(market_price: float, 
                       S: float,
                       K: float,
                       T: float,
                       r: float,
                       option_type: str = "call") -> float:
    
    """
    Implied Volatility — die wichtigste Zahl im Options-Markt.

    Was ist das?
        Black-Scholes gibt dir den fairen Preis bei
        gegebener Volatilität.
        IV dreht das um: gegeben den Marktpreis,
        welche Volatilität steckt darin?

    Warum das Gold wert ist:
        IV ist der Markt-Konsens über zukünftige Volatilität.
        IV > Historical Vol → Markt erwartet mehr Bewegung (teuer)
        IV < Historical Vol → Markt unterschätzt Risiko (günstig)

        VIX = IV-Index für S&P 500.
        Wenn VIX explodiert kaufen alle Angst-Puts.
        Wenn VIX niedrig ist schlafen alle — bis sie es nicht tun.

    Algorithmus:
        Brentq = numerische Nullstellensuche.
        Findet sigma so dass BS-Preis = Marktpreis.
        Effizient, stabil, professionell.
    """
    # Eine option kann nicht weniger wert sein als ihr intrinsischer wert -> würde sofort weggekauft werden
    intrinsic = max(S - K, 0) if option_type == "call" \
        else max(K - S, 0)
    
    if market_price <= intrinsic:
        return 0
    
    # wir suchen das sigma bei dem BS-Preis = Marktpreis also bspreis - marktpreis = 0
    
    def objective(sigma):
        return black_scholes(S, K, T, r, sigma, option_type) \
            - market_price
    
    try: 
        iv = brentq(objective, 1e-6, 10.0, xtol=1e-6) # brentq = nullstellensuche 
        return round(iv, 4)
    except ValueError: # wenn kein vorzeichenwechsel gefunden wird 
        return 0
    
def iv_surface(S: float,
               T_range: list,
               K_range: list,
               r: float,
               option_type: str = "call") -> pd.DataFrame:
    """
    Implied Volatility Surface — zeigt wie IV sich über
    Strike und Laufzeit verändert.

    In der bs-theorie ist sigma konstant - in der realität nicht  

    --- Gleiche Aktie, gleiche Expiration: ---
    Strike 140 (deep ITM):  IV = 35%
    Strike 150 (ATM):       IV = 25%   ← am niedrigsten unser tiefpunkt in der 3d fläche 
    Strike 160 (OTM):       IV = 30%

    es gibt einen "Volatility Smile" oder "Skew".
    Puts haben oft höhere IV als Calls (Put-Skew). da oft puts gekauft werden aus angst eines crashes 
    Das zeigt Angst vor Kursrückgängen.

    Hier simulieren wir einen realistischen Smile.
    """

    data =[]
    for T in T_range:
        for K in K_range:
            moneyness = K/S

            # Realistischer iv-Smile simulieren
            #ATM IV: 25%, nimmt zu bei extremen Strikes 
            base_iv = 0.25
            skew = 0.05 * (1-moneyness) 
            smile = 0.08 * (moneyness - 1)** 2 # symmetrisch um at the money -> je weiter weg desto höhere IV
            # Deep On the Money Positionen sind selten profitabel 
            term_adj = 0.03 * (1- T * 4) # kurzfristige Optionen haben höhere IV als langfristige 

            iv = max(base_iv + skew + smile + term_adj, 0.05)

            """
       IV
       │    *           *      ← hohe IV bei extremen Strikes
       │      *       *
       │        *   *
       │          *            ← ATM: niedrigste IV (Smile-Boden)
       │
       └──────────────────── Strike
         low   ATM    high
            """

            price = black_scholes(S, K, T, r, iv, option_type)
            greeks = compute_greeks(S, K, T, r, iv, option_type)

            data.append({
                "Strike":    K,
                "T (Tage)":  round(T * 365),
                "IV (%)":    round(iv * 100, 2),
                "Price":     round(price, 2),
                "Delta":     greeks["delta"],
                "Gamma":     greeks["gamma"],
                "Theta":     greeks["theta"],    
            })

    return pd.DataFrame(data)

"""
Marktpreis
    ↓
implied_volatility()        ← Brentq löst BS rückwärts
    ↓
IV für jeden Strike + T
    ↓
iv_surface()                ← sammelt alle IVs in 2D
    ↓
3D Volatility Surface       ← das was Trader wirklich anschauen

IV%
     │  ╲___________/   ← kurze Laufzeit: ausgeprägter Smile -> schwieriger auf otm zu kommen  - angst im markt 
     │    ╲_________/
     │      ╲_______/   ← lange Laufzeit: flacherer Smile - normal
     │
     └───────────────── Strike
          T wächst →
IV > Historical Vol Markt erwartet mehr Bewegung als in der Vergangenheit → Optionen teuer
IV < Historical Vol Markt unterschätzt Risiko → Optionen günstig (oder: stille vor dem Sturm)
"""


def options_pnl_profile(positions: list,
                        S_range: np.ndarray) -> pd.DataFrame:
    """
    P&L Profil einer Options-Position bei Expiration.

    positions = Liste von Dicts:
    [
        {"type": "call", "strike": 150,
         "premium": 5.0, "qty": 1},
        {"type": "put",  "strike": 140,
         "premium": 3.0, "qty": -1},   # short put
    ]

    qty > 0 = Long (gekauft)
    qty < 0 = Short (verkauft)

    Klassische Strategien:
        Long Call:              long call
        Covered Call:           long stock + short call
        Bull Call Spread:       long call(low K) + short call(high K)
        Straddle:               long call + long put (same K)
        Iron Condor:            short strangle + long strangle
    """
    total_pnl = np.zeros(len(S_range))

    for pos in positions:
        K       = pos["strike"]
        premium = pos["premium"]
        qty     = pos["qty"]
        typ     = pos["type"]

        if typ == "call":
            intrinsic = np.maximum(S_range - K, 0)
        elif typ == "put":
            intrinsic = np.maximum(K - S_range, 0)
        else:
            intrinsic = np.zeros(len(S_range))

        pnl = (intrinsic - premium) * qty * 100 # 1 contract = 100 shares 
        total_pnl += pnl

    return pd.DataFrame({
        "Kurs": S_range,
        "P&L ($)": total_pnl.round(2),
    })

def plot_greeks_profile(S:     float,
                        K:     float,
                        r:     float,
                        sigma: float,
                        option_type: str = "call") -> None:
    
    """
    4-Panel Greeks Profil:
    Zeigt wie jeder Greek sich mit dem Kurs verändert.
    Das ist was Options-Trader täglich brauchen.
    """
    S_range = np.linspace(S * 0.6, S * 1.4, 200) # linspace =  array mit gleichmäßigen num zwischen 90 und 210 -> 200 punkte
    # 0.6 -> -40% vom Kurs bis +40% vom Kurs (Aktuell 150)
    T_values = [30/365, 60/365, 90/365]
    colors   = ["#ef4444", "#f59e0b", "#2563eb"]
    labels   = ["30 Tage", "60 Tage", "90 Tage"]

    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=["Delta (Δ)", "Gamma (Γ)",
                        "Theta (Θ) — pro Tag",
                        "Vega (V) — pro 1% Vol"],
        vertical_spacing=0.12,
        horizontal_spacing=0.10
    )

    for T, color, label in zip(T_values, colors, labels):
        deltas = []
        gammas = []
        thetas = []
        vegas  = []

        for s in S_range:
            g = compute_greeks(s, K, T, r, sigma, option_type)
            deltas.append(g["delta"])
            gammas.append(g["gamma"])
            thetas.append(g["theta"])
            vegas.append(g["vega"])

        # Delta
        fig.add_trace(go.Scatter(
            x=S_range, y=deltas,
            name=label,
            line=dict(color=color, width=2),
            showlegend=True
        ), row=1, col=1)

        # Gamma
        fig.add_trace(go.Scatter(
            x=S_range, y=gammas,
            name=label,
            line=dict(color=color, width=2),
            showlegend=False
        ), row=1, col=2)

        # Theta
        fig.add_trace(go.Scatter(
            x=S_range, y=thetas,
            name=label,
            line=dict(color=color, width=2),
            showlegend=False
        ), row=2, col=1)

        # Vega
        fig.add_trace(go.Scatter(
            x=S_range, y=vegas,
            name=label,
            line=dict(color=color, width=2),
            showlegend=False
        ), row=2, col=2)

    # Strike Linie 
    for row, col in [(1,1), (1,2), (2,1), (2,2)]:
        fig.add_vline(
            x=K, line_dash="dot",
            line_color="#94a3b8",
            opacity=0.6,
            row=row, col=col
        )

    fig.update_layout(
        height=650,
        template="plotly_white",
        title=f"{option_type.upper()} Greeks Profil "
              f"(K=${K}, σ={sigma*100:.0f}%)",
        legend=dict(orientation="h", y=1.02),
        margin=dict(l=0, r=0, t=60, b=0)
    )

    fig.show()

def plot_pnl_profiles(S: float) -> None:
    """
    P&L Profile für 4 klassische Strategien.
    Jeder Options-Trader sollte diese auswendig kennen.
    """
    S_range = np.linspace(S * 0.7, S * 1.3, 300)

    strategies = { # man zahlt bei kauf noch eine prämie 
        "Long Call (K=+5%)": [
            {"type": "call", "strike": S*1.05,
             "premium": 3.0, "qty": 1}
        ],
        "Long Put (K=-5%)": [
            {"type": "put", "strike": S*0.95,
             "premium": 2.5, "qty": 1}
        ],
        "Bull Call Spread": [
            {"type": "call", "strike": S*0.98,
             "premium": 5.0, "qty": 1},
            {"type": "call", "strike": S*1.05,
             "premium": 2.0, "qty": -1}
        ],
        "Long Straddle (ATM)": [
            {"type": "call", "strike": S,
             "premium": 4.0, "qty": 1},
            {"type": "put",  "strike": S,
             "premium": 3.5, "qty": 1}
        ],
    }

    colors = ["#16a34a", "#ef4444", "#2563eb", "#f59e0b"]

    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=list(strategies.keys()),
        vertical_spacing=0.12,
        horizontal_spacing=0.10)
    
    positions = [(1,1), (1,2), (2,1), (2,2)]

    for (name, pos_list), color, (r,c) in zip(
        strategies.items(), colors, positions
    ):
        pnl_df = options_pnl_profile(pos_list, S_range)

        pnl_colors = [
            "#16a34a" if v >= 0 else "#ef4444"
            for v in pnl_df["P&L ($)"]
        ]

        fig.add_trace(go.Scatter(
            x=pnl_df["Kurs"],
            y=pnl_df["P&L ($)"],
            name=name,
            line=dict(color=color, width=2),
            fill="tozeroy",
            fillcolor=f"rgba{tuple(list(int(color.lstrip('#')[i:i+2], 16) for i in (0,2,4)) + [0.08])}",
        ), row=r, col=c)

        # Nulllinie
        fig.add_hline(
            y=0, line_dash="dot",
            line_color="#64748b",
            opacity=0.5,
            row=r, col=c
        )

        # Aktueller Kurs
        fig.add_vline(
            x=S, line_dash="dash",
            line_color="#94a3b8",
            opacity=0.5,
            row=r, col=c
        )

    fig.update_layout(
        height=650,
        template="plotly_white",
        title=f"Options P&L Profile bei Expiration (S=${S:.0f})",
        showlegend=False,
        margin=dict(l=0, r=0, t=60, b=0)
    )

    fig.show()

def plot_iv_surface(S: float,
                    r: float = 0.05) -> None:
    
    """
    3d implied volatility surface - das beeindruckendste bis jetzt 
    """
    T_range = [7/365, 14/365, 30/365, 60/365, 90/365, 180/365]
    K_range = np.linspace(S * 0.80, S * 1.20, 15)

    surface_df = iv_surface(S, T_range, K_range, r)

    T_labels = ["7d", "14d", "30d", "60d", "90d", "180d"]
    
    Z = []

    for T_days in [7,14, 30, 60, 90, 180]:
        row_data = surface_df[surface_df["T (Tage)"] == T_days]["IV (%)"].values
        Z.append(row_data.tolist())

    fig = go.Figure(data=[go.Surface(
        z=Z,
        x=K_range.tolist(),
        y=T_labels,
        colorscale=[
            [0.0, "#ad4ed6"],
            [0.25, "#3b82f6"],
            [0.5, "#93c5fd"],
            [0.75, "#fca5a5"],
            [1.0, "#dc2626"]
        ],
        opacity = 0.9,
    )])

    fig.update_layout(
        title=f"Implied Volatility Surface (S=${S:.0f})",
        scene=dict(
            xaxis_title="Strike ($)",
            yaxis_title="Laufzeit",
            zaxis_title="IV (%)",
            camera=dict(eye=dict(x=1.5, y=-1.5, z=0.8))
        ),
        height=600,
        template="plotly_white",
        margin=dict(l=0, r=0, t=40, b=0)
    )

    fig.show()

def build_options_chain(S:     float,
                        T:     float,
                        r:     float,
                        sigma: float) -> pd.DataFrame:
    """
    Synthetische Options-Chain — wie du sie auf jedem Broker siehst.
    Zeigt Calls und Puts für verschiedene Strikes.
    """
    strikes = np.arange(
        round(S * 0.85 / 5) * 5,
        round(S * 1.15 / 5) * 5 + 5,
        5
    )
    rows = []
    for K in strikes:
        call_g = compute_greeks(S, K, T, r, sigma, "call")
        put_g  = compute_greeks(S, K, T, r, sigma, "put")

        moneyness = "ATM" if abs(K-S) < S *0.02 \
                    else ("ITM" if K< S else "OTM")

        rows.append({
            "Strike":        K,
            "Moneyness":     moneyness,
            "Call Preis":    call_g["price"],
            "Call Delta":    call_g["delta"],
            "Call Gamma":    call_g["gamma"],
            "Call Theta":    call_g["theta"],
            "Put Preis":     put_g["price"],
            "Put Delta":     put_g["delta"],
            "Put Theta":     put_g["theta"],
            "Put-Call Par.": round(
                call_g["price"] - put_g["price"] - S + K * np.exp(-r * T), 4
            ), # Call - Put - S + K · e^(-rT) = 0 muss gelten sonst gibt es eine arbitrage möglichkeit 
            # zeigt atm
        })

    return pd.DataFrame(rows)

if __name__ == "__main__":

    # --- Parameter ---
    S     = 150.0    # Aktueller Kurs (z.B. AAPL ca. $150)
    K     = 155.0    # Strike Preis
    T     = 30/365   # 30 Tage bis Expiration
    r     = 0.05     # 5% risikoloser Zinssatz
    sigma = 0.30     # 30% Volatilität

    print("Tag 18 — Options Greeks & Black-Scholes")
    print("=" * 50)

    # --- Single Option Analyse ---
    call = compute_greeks(S, K, T, r, sigma, "call")
    put  = compute_greeks(S, K, T, r, sigma, "put")

    print(f"\nCall Option (S=${S}, K=${K}, "
          f"T=30d, σ={sigma*100:.0f}%):")
    for k, v in call.items():
        if k not in ["d1", "d2"]:
            print(f"  {k:<8} {v:>10.4f}")

    print(f"\nPut Option:")
    for k, v in put.items():
        if k not in ["d1", "d2"]:
            print(f"  {k:<8} {v:>10.4f}")

    # Put-Call Parität prüfen
    parity = call["price"] - put["price"] - S + K * np.exp(-r * T)
    print(f"\nPut-Call Parität Check: {parity:.6f}")
    print("→ Sollte ≈ 0 sein (Arbitrage-Bedingung)")

    # --- Implied Volatility ---
    print("\n--- Implied Volatility Berechnung ---")
    market_price = 6.50   # Angenommener Marktpreis
    iv = implied_volatility(market_price, S, K, T, r, "call")
    print(f"Marktpreis Call: ${market_price}")
    print(f"Implied Vol:     {iv*100:.2f}%")
    print(f"Eingabe-Vol:     {sigma*100:.2f}%")
    diff = (iv - sigma) * 100
    print(f"Differenz:       {diff:+.2f}%")
    if diff > 0:
        print("→ Markt preist mehr Volatilität ein als angenommen")
    else:
        print("→ Markt preist weniger Volatilität ein")

    # --- Options Chain ---
    print("\n--- Options Chain (30 Tage) ---")
    chain = build_options_chain(S, T, r, sigma)
    display_cols = ["Strike", "Moneyness",
                    "Call Preis", "Call Delta",
                    "Put Preis", "Put Delta"]
    print(chain[display_cols].to_string(index=False))

    # --- Theta Decay Analyse ---
    print("\n--- Theta Decay: ATM Call über Zeit ---")
    print(f"  {'Tage bis Exp':>14} {'Preis':>8} "
          f"{'Theta/Tag':>12} {'% Wert verloren':>16}")
    print("  " + "-"*52)

    for days in [90, 60, 45, 30, 21, 14, 7, 3, 1]:
        g = compute_greeks(S, S, days/365, r, sigma, "call")
        pct_lost = (1 - g["price"] / compute_greeks(
            S, S, 90/365, r, sigma, "call"
        )["price"]) * 100
        print(f"  {days:>14}d "
              f"  ${g['price']:>6.2f}"
              f"  ${g['theta']:>10.4f}"
              f"  {pct_lost:>14.1f}%")
        
    print("\n→ Theta Decay beschleunigt sich — "
          "die letzten 30 Tage verlierst du am schnellsten.")

    # --- Sensitivitäts-Analyse ---
    print("\n--- Vega Analyse: Preis bei verschiedenen Vols ---")
    print(f"  {'Volatilität':>12} "
          f"{'Call Preis':>12} {'Put Preis':>11}")
    print("  " + "-"*35)

    for vol in [0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.50]:
        c = black_scholes(S, K, T, r, vol, "call")
        p = black_scholes(S, K, T, r, vol, "put")
        marker = " ← aktuell" if vol == sigma else ""
        print(f"  {vol*100:>10.0f}%  "
              f"  ${c:>8.2f}  "
              f"  ${p:>8.2f}{marker}")

    # --- Charts ---
    plot_greeks_profile(S, K, r, sigma, "call")
    plot_pnl_profiles(S)
    plot_iv_surface(S, r)

    # --- Reale AAPL Daten ---
    print("\n--- Reale Parameter: AAPL ---")
    try:
        aapl    = yf.Ticker("AAPL")
        hist    = aapl.history(period="1y")
        S_real  = float(hist["Close"].iloc[-1])
        returns = hist["Close"].pct_change().dropna()
        vol_real = float(returns.std() * np.sqrt(252))

        print(f"AAPL Kurs:        ${S_real:.2f}")
        print(f"Hist. Volatilität: {vol_real*100:.1f}%")

        # ATM Call berechnen
        atm_call = compute_greeks(
            S_real, round(S_real/5)*5,
            30/365, 0.05, vol_real, "call"
        )
        print(f"\nATM Call (30d):")
        print(f"  Preis: ${atm_call['price']:.2f}")
        print(f"  Delta: {atm_call['delta']:.3f}")
        print(f"  Theta: ${atm_call['theta']:.3f} / Tag")
        print(f"  Vega:  ${atm_call['vega']:.3f} / 1% Vol")

        # Options Chain für AAPL
        print("\nAAPL Options Chain (30 Tage, Hist. Vol):")
        aapl_chain = build_options_chain(
            S_real, 30/365, 0.05, vol_real
        )
        print(aapl_chain[display_cols].to_string(index=False))

    except Exception as e:
        print(f"Fehler beim Laden: {e}")

    # Export
    chain.to_csv("day18_options_chain.csv", index=False)
    print("\nGespeichert: day18_options_chain.csv")


