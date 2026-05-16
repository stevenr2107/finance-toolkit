"""
Day 19 - Options Strategie Backtesting

Drei klassische Income Strategien:
    Covered Call: Aktie + Short Call + Prämie kassieren 
    Cash-Secured Put: Short Put -> Aktie günster kaufen 
    Iron Condor: Short Strangle + Long strangle -> Range strategie 

Warum das anders ist als aktien backtesting:
    Options verfallen  jeden monat neu 
    prämien - einnahmen sind der return
    volatilität ist das risiko und die chance
    tiiming ist wichtiger als bei aktien


Wann macht man was?:

"""

import yfinance as yf
import pandas as pd
import numpy as np
from scipy.stats import norm
from scipy.optimize import brentq
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from dataclasses import dataclass
from typing import Optional
import warnings
warnings.filterwarnings("ignore")

def load_data(ticker: str, period: str = "5y") -> pd.DataFrame:
    df = yf.download(ticker, period = period, auto_adjust=True, progress=False)
    df.columns = df.columns.get_level_values(0) # hier wird alles in einem Index gespeichert statt in mehreren Spalten
    return df.dropna()

def black_scholes(S: float, K: float, T: float, r:float, 
                  sigma:float, option_type: str = "call") -> float:
    """ BS Pricing - aus Tag 18"""
    if T <= 0:
        return max(S-K, 0 ) if option_type == "call" else max(K-S, 0)
    d1 = (np.log(S/K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    if option_type == "call":
        return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    return K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)

def historical_volatility(close: pd.Series,
                          window: int = 30) -> pd.Series:
    """Rollierende historische Volatilität - annualisiert."""
    return close.pct_change().rolling(window).std() * np.sqrt(252)



"""
Man kauft die Aktie und verkauft selber den Call
Problem: Wenn die aktie hoch geht, hat man zu seinem Strike quasi schon verkauft und macht keinen weiteren Gweinn 
3 Szenarien  

1. ich verkaufe den call für 1.50  2$ überm preis aber sie fällt um 2$ 
-> Kontrakt ist wertlos, ich verliere selber 2$, da die aktie gefallen ist, mache aber 1,50, da jemand den call gekauft hat
-> -0.50$

2. Die Aktie macht nichts (Seitwärts) 
-> Ich mache kein Gled an der Aktie, habe aber 1.50$ gewinn, da ich den call verkauft hab 

3. Die Aktie geht von 32$ auf 33$ 
-> Ich mache 1$ wegen der Aktie und 1.50$ wegen dem call
-> + 2.50$

4. Die Aktie steigt stark 32 -> 40$
-> Ich verkaufe sie jemand anderem für 34 
-> + 2$ stock und 1.50$ call aber die aktie hat ja eig 8 gemacht 
-> + 3.50$ bei möglichen 8$


Wichtig:

Expiration: 
eigentlich 2 Monate aber jetzt 6 monate 
-> Calls sind teurer, je länger sie gehen 

Beispiel: 2 monate kosten 1.50 und 6 monate 2.50 -> ich verkaufe einfach 3x 1.50 oder? 
Call optionen werden billiger, wenn der Kurs fällt -> 32->28 | 1.50->0.50$

Strike:
eigentlich 34$ aber jetzt 38$ 
-> Je weiter weg der strike desto niedriger der preis
-> Ich kann mehr geld mit der aktie selber machen, aber call jetzt 0.50$ statt 1.50$

Setup:
1. Kaufe die Aktie
2. Verkaufe die Call Optionen
- Man kauft keine müll aktie, da sie dann sinkt und man daran verliert 

3.1 Man bekommt instant das geld für die Kontrakte bsp. 5 Kontakte a 150$ da 100 shares = 750$
-> Man bekommt 750$ aufs konto 

3.1a Die Aktie geht von 32 auf 35 aber der strike ist bei 34 ->Man muss die Aktien für 34$ pro Share verkaufen 
(macht der broker automatisch)

3.1b Die aktie bleibt unter strike 34 auf 30 
-> Call wird automatisch eliminiert 

--- wie werden die Kontakte eingepreist? ---
- Greeks aus tag 18
- je länger sie gehen, -> teurer
- je näher der strike -> teurer
- Je volatiler die Aktie -> teurer

Man braucht eine bestimmte Anzahl an shares um Kontrakte schreiben zu können





Hedging ist eine starke Absicherung bei kleinen anstiegen 
"""
@dataclass
class OptionTrade:
    """Ein einzelner Options-Trade."""
    strategy: str
    entry_date:  pd.Timestamp # Tag an dem wir verkaufen ( unsere position eröffnen)
    expiry_date: pd.Timestamp
    stock_price: float
    strike:      float
    premium:     float # Wieviel wir pro Aktie kassieren 
    shares:      int        = 100 # Standardgröße : 100 Aktien = 1 option 
    exit_date:   Optional[pd.Timestamp] = None
    exit_stock:  Optional[float]        = None
    pnl:         Optional[float]        = None
    outcome:     Optional[str]          = None # assigned oder expired

def backtest_covered_call(df: pd.DataFrame,
                          delta_target: float = 0.3, # 30% angestrebt das er ins gewinn läuft 
                          dte: int = 30, # days to expiration option läuft 30 tage 
                          r: float = 0.05, # risikoloser zins ( bonds)
                          capital = 50_000) -> dict: # startkapital 
    """
    delta_ target = 0.3:
        call mit ca 30% delta -> ca. 30% wahrscheinlichkeit 
        dass er im geld landet -> klassischer sweet spot 
        """
    close = df["Close"].squeeze()
    vol = historical_volatility(close, 30) # Wird für die letzten 30 tage berechnet für black_scholes

    # Aktien kaufen mit gesamten Kapital 
    initial_price = float(close.iloc[dte])
    # Starten am tag dte weil wir 30 tage hist. Daten für volatility brauchen

    shares = int(capital / initial_price / 100) * 100
    # Wie viele aktien können wir kaufen?
    # Man kann keine 0.5 Optionen kaufen -> int(2.5) / 100 -> 2 nicht 2.5 wegen int 

    stock_cost = shares * initial_price
    # Ausgegebenes geld 

    cash = capital - stock_cost
    # Verbleibendes Cash nach aktienkauf / für prämien 

    equity_curve = [] # Liste: GEsamtwert (Cash + aktien)
    trades = [] # Ein eintrag pro monat 
    dates = []

    for i in range(dte, len(df) - dte, dte):
        # Start: i = 30
        # Ende: len(df) - 30 (puffer damit expiry nicht aus dem array durchläuft)
        # Schritte: 30 (jeden monat ein neuer trade )

        date = df.index[i] # Heutiges Datum 
        S = float(close.iloc[i]) # aktueller Aktienkurs
        sigma = float(vol.iloc[i]) if not np.isnan(vol.iloc[i]) else 0.25
        # Historische vola heute 
        # fallback 0.25 = 25% falls noch kein vol-wert berechnbar (anfang)
        T = dte/365
        # Time to expiration in Jahren ( für black scholes)
        # 30 tage / 365 = 0.082 Jahre

        # Strike finden der -delta_target hat 
        # Suche über strike range 
        best_strike = S
        best_delta_diff = 99 # Startwert: absichtlich riesig, wird überschrieben 


        """
        wir wollen einen call mir 30% delta aber wissen nicht bei welchem strike das ergibt 
        wir schauen in 0.005 schritten zwischen 1 und 1.2 x des spot 
        bsp: aktie = 200$ = ATM 
        k_mult 1.195 -> K = 239 $ (weit OTM out of the money )

        Strike   Delta    Abstand zu 0.30
        ──────────────────────────────────
        200$     0.50     0.20   ❌ zu nah dran (ATM)
        205$     0.42     0.12   ❌ immer noch zu hoch
        210$     0.35     0.05   fast...
        212$     0.31     0.01   sehr nah!
        213$     0.30     0.00   ✅ TREFFER → best_strike
        220$     0.20     0.10   ❌ schon zu weit weg
        """
        for k_mult in np.arange(1.00, 1.20, 0.005):
            # Suche über striike multiplikatoren von 1 x bis 1.2 x des spot 
            # (also ATM bis 20% OTM) in 0.5% schritten 
            K = S * k_mult # Kandidat strike 

            if T > 0 and sigma > 0:
                d1    = (np.log(S/K) + (r + 0.5*sigma**2)*T) / \
                        (sigma * np.sqrt(T))
                
                delta = norm.cdf(d1)
                # WSK das call im geld landet (delta = 1)
                if abs(delta - delta_target) < best_delta_diff:
                    # ist dieser strike näher am ziel-delta als vorheriger?
                    best_delta_diff = abs(delta - delta_target)
                    best_strike     = K # neuen besten merkken 

        K = round(best_strike/ 5) * 5 # auf nächsten 5$ runden  - so machens profis 

        premium = black_scholes(S, K, T, r, sigma, "call") # einnahmen für mich pro aktie 
        premium = max(premium, 0.01)# verhindetr devision by zero - min. 1 cent 

        # Prämie kassieren 
        premium_income = premium * shares # gesamte prämieneinnahmen
        cash += premium_income # prämie sofort ins cash konto 

        # Expiry prüfen 
        expiry_idx = min(i + dte, len(df) - 1) # verhindert out of bonds 

        expiry_date = df.index[expiry_idx] # datum verfallstag 
        S_expiry = float(close.iloc[expiry_idx]) # aktienkurs am verfallstag 

        if S_expiry > K:
            # Call wird ausgeübt - Aktien am verfallstag über strike
            outcome = "Assigned"
            stock_proceeds = K * shares # man bekommt k nicht s_expiry
            # man verpasst den kursanstieg 
            pnl_option = (K - S + premium) * shares
            # kassiert - verpasste upside 

            # Aktien neu kaufen 
            cash += stock_proceeds
            shares_new = int(cash / S_expiry / 100) * 100
            # wie viele aktien können wir zum neuen kurs kuafen?
            stock_cost = shares_new * S_expiry
            cash -= stock_cost
            shares = shares_new
            # re-investieren sofort -> beliben immer investiert 
        else: 
            # Call verfällt wertlos - Voller prämien-gewinn
            outcome = "Expired"
            pnl_option = premium_income # wir bekommen premium ohne abzüge 

        trades.append(OptionTrade(
            strategy = "Covered Call",
            entry_date = date,
            expiry_date = expiry_date,
            stock_price = S, # kurs beim schreiben des calls
            strike = K, # Gewählter strike 
            premium = round(premium, 2),
            shares = shares,
            exit_date = expiry_date,
            exit_stock = S_expiry,
            pnl = round(pnl_option,2),
            outcome = outcome # assigned oder expired 
        ))

        # Equity : Cash + Aktienwert
        stock_value = shares * S_expiry
        equity_curve.append(round(cash + stock_value, 2))
        # Gesamtwert = Cash inkl. Prämien + aktueller Marktwert der aktien 
        dates.append(expiry_date)
    
    equity = pd.Series(equity_curve, index = dates)
    # Zeitreihe des Portfoliowertes -> für charts und performancezahlen 
    trades_df = pd.DataFrame([t.__dict__ for t in trades])
    # Liste von optiontrade-objekten
    # __dict__ macht aus jedem objekt ein dict

    return {
        "strategy": "Covered Call",
        "equity": equity,
        "trades": trades_df,
        "capital": capital
    }

def backtest_cash_secured_put(df: pd.DataFrame,
                                delta_target: float = 0.30, # put mit ca 30% delta 
                                # -> Strike ca 5-8% unter aktuellem kurs 
                                 dte: int = 30, # Days to expiration
                                 r: float = 0.05, # risikoloser zinssatz 
                                 capital: float = 50_000) -> dict:
    """
    Cash secured puts: Puts verkaufen 
    Voraussetzung:
    100 Aktien für einen Kontrakt 


    Warum das interessant ist:
        besser als eine market order 

    Aktie liegt bei 4.89 

    - Man denkt die Aktie ist bei 4.89 ein guter Kauf 
    -> alles drunter ist ein noch besserer kauf 

    Szenarien: 
        wir gehen auf 1 monat expiry options
        4.50 strike put option kostet 0.32§

    1. Die aktie bleibt über 4.50 -> 7.1% roi
        wir kaufen die aktie nicht -> bekommen aber 0.32$ 
        Risk: Die aktie geht auf 4.50 und wir müssen kaufen 

    2. Die aktie fällt auf 4.40 
        wir müssen bei 4.50 kaufen und verlieren 0.10$ durch die aktie 
        wir bekommen aber 0.32$ prämie da wir das risiko eingegangen sind 

    3. Die aktie fällt auf 1$ 
        man muss bei 4.50$ kaufen und verliert 3.50$

    execution:
    trade options -> single order -> Ticker -> Sell to open -> wieviele contracts (100 Shares)
    -> Expiration -> Strike price -> put -> Limit price -> wieviel die put option kosten soll
    """

    close = df["Close"].squeeze()
    vol = historical_volatility(close, 30)
    # Vola wieder für black scholes

    cash         = capital # wir starten mit reinem cash , keine aktien 
    # Kern von "Cash-Secured"
    # Geld liegt bereit falls wir kaufen müssen
    shares       = 0 # erst 0 nur bei assigment kaufen 
    equity_curve = []
    trades       = []
    dates        = []

    for i in range(dte, len(df) - dte, dte): # monat für monat loop 
        date = df.index[i]
        S = float(close.iloc[i])
        sigma = float(vol.iloc[i]) if not np.isnan(vol.iloc[i]) else 0.25
        T = dte/365

        # strike finden
        best_strike = S 
        best_delta_diff = 99

        for k_mult in np.arange(0.80, 1.00, 0.005):
            # bei covered call 1-1.2 -> otm über SPOT
            # Cash-secured put sucht 0.8 bis 1x unter SPOT 
            K = S * k_mult
            # beispiel: S = 200$, k_mult = 0.92 -> K = 184$

            if T > 0 and sigma > 0:
                d1 = (np.log(S/K) + (r + 0.5*sigma**2)*T) / \
                        (sigma * np.sqrt(T))
                
                delta = abs(norm.cdf(d1) - 1)
                # Put delta ist NEGATIV -0.30
                # abs macht positiv damit vergleich funktioniert 

                if abs(delta - delta_target) < best_delta_diff:
                    best_delta_diff = abs(delta - delta_target)
                    best_strike = K

        K = round(best_strike / 5) * 5
        premium = black_scholes(S, K, T, r, sigma, "put")
        # Put prämie 
        premium = max(premium, 0.01)

        # Prämie kassieren 
        contracts = max(int(cash / (K * 100)), 1)
        # Wie viele Puts müssen wir mit unserem Cash absichern?
        # K * 100 = collateral pro kontrakt -> wir müssen 100 aktien zu k kaufen können
        # max (..., 1) -> mind. 1 Kontrakt auch wenn cash knapp


        contracts = min(contracts , 5) # max 5 kontrakte 
        premium_income = premium * contracts * 100 
        cash += premium_income

        # Collateral reservieren (K * 100 pro Kontrakt)
        collateral = K * 100 * contracts
        # Cash das blockiert ist falls assignment kommt.

        # Expiry 
        expiry_idx  = min(i + dte, len(df) - 1)
        expiry_date = df.index[expiry_idx]
        S_expiry    = float(close.iloc[expiry_idx])

        if S_expiry < K:
            # Put ausgeübt → Aktien kaufen zu Strike
            # Aktienkurs am verfallstag UNTER dem Strike 
            # -> Put käufer übt aus: er verkauft uns aktien zu K
            # -> wir müssen zu k kaufen 

            outcome    = "Assigned"
            cost       = K * 100 * contracts
            # was wir zahlen müssen 

            new_shares = 100 * contracts
            cash      -= cost # cash reduzieren 
            shares    += new_shares # ins portfolio 
            pnl        = (premium - (K - S_expiry)) * \
                          100 * contracts
            # Buchverlust: wir zahlen K , markt ist bei s_expiry 
            # Beispiel: premium=2$, K=185, S_expiry=175 → (2 - 10) * 200 = -1.600$
            # Aber: effektiver Einstandskurs = K - premium = 183$ statt 185$
        else: 
            # Put verfällt -> volle prämie behalten 
            outcome = "Expired"
            pnl = premium_income 

        trades.append(OptionTrade(
            strategy    = "Cash-Secured Put",
            entry_date  = date,
            expiry_date = expiry_date,
            stock_price = S,
            strike      = K,
            premium     = round(premium, 2),
            shares      = contracts * 100, # aktien die auf dem spiel stehen 
            exit_date   = expiry_date,
            exit_stock  = S_expiry,
            pnl         = round(pnl, 2),
            outcome     = outcome,
        ))

        # Equity 
        stock_value = shares * S_expiry
        equity_curve.append(round(cash + stock_value, 2))
        dates.append(expiry_date)
    
    equity = pd.Series(equity_curve, index=dates)
    trades_df = pd.DataFrame([t.__dict__ for t in trades])

    return {
        "strategy": "Cash-Secured Put",
        "equity":   equity,
        "trades":   trades_df,
        "capital":  capital,
    }


def backtest_iron_condor(df: pd.DataFrame,
                         width: float = 0.05, # short strikes 5% vom kurs entfernt 
                                            # short put 6% unter spot, short call 5% über spot                          
                         dte: int = 30,
                         r: float = 0.05,
                         capital = 50_000) -> dict:
    """
    Iron Condor Backtest

    Setup:
        Short OTM Put  (K1 = S × (1-width))
        Long  OTM Put  (K2 = K1 - wing)    ← Schutz
        Short OTM Call (K3 = S × (1+width))
        Long  OTM Call (K4 = K3 + wing)    ← Schutz

    man schaut sich eine seitwärtsphase an und fängt in der mitte an 
    wir hoffen das oben und unten nicht getroffen wird und so beide options auslaufen und wir unsere prämie machen 
    zudem wird ein call über dem sell call gekauft und ein put unter dem sell put gekauft.
    -> Absicherung bei breakout

    perfektes Setup:
    60 : put kaufen wir für 1$          -> -100$
    65 : put verkaufen wir für 2$       -> +200$
    75 : jetzt                          ->  200$ profit 
    85 : call verkaufen wir für 2$      -> +200$
    90 :  callkaufen wir für 1$         -> -100$

    max verlust: Begrenzt durch Long-Options 

    wann profitabel:
        niedrige vola 
        nach earnings wenn iv kollabiert
        seitwärtsmärkte 

    width = 0.05:
        short strike 5% vom kurs entfernt 
    wing = 2.5% 
        long strike 2.5% weiter draußen als short 
    """

    close  = df["Close"].squeeze()
    vol    = historical_volatility(close, 30)

    cash         = capital
    equity_curve = []
    trades       = []
    dates        = []
    # Loop macht jeden Monat 

    for i in range(dte, len(df) - dte, dte):
        date  = df.index[i]
        S     = float(close.iloc[i])
        sigma = float(vol.iloc[i]) if not np.isnan(vol.iloc[i]) \
                else 0.25
        T     = dte / 365
        wing  = width / 2
        # Wing = Abstand zwischen short und long strike ("Der flügel")
        # width = 5% -> wing = 2.5%

        # ---STRIKES DEFINIEREN -----------------------------------
        #
        # Visuell für S = 200$, width = 5%, wing = 2.5%:
        #
        #  K_long_put  K_short_put      S      K_short_call  K_long_call
        #     185$         190$        200$        210$          215$
        #      │            │           │            │             │
        #   Long Put    Short Put   (Kurs)      Short Call    Long Call
        #  (Schutz)    (verkauft)              (verkauft)    (Schutz)
        #      └────────────┴── Put Spread ────┴─────────────┘
        #                        Profit Zone
        #
        # Prämien kassieren (Short) minus Prämien zahlen (Long) = Netto-Einnahme


        K_short_put  = round(S * (1 - width) / 5) * 5
        # Short Put: 5% unter spot -> 200 * 0.95 = 190$ 
        # verkauft: prämie kassieren direkt 

        K_long_put   = round(S * (1 - width - wing) / 5) * 5
        # Long put: 7.5% unter Spot -> 200 * 0.925 = 185$ 
        # Gekauft -> Prämie zahlen, aber begrenzt den maximalen verlust nach unten 

        K_short_call = round(S * (1 + width) / 5) * 5
        # Short Call: 5% ober spot -> 200 * 1.05 = 210$ 
        # verkauft: prämie kassieren direkt

        K_long_call  = round(S * (1 + width + wing) / 5) * 5
        # Long call: 7.5% ober Spot -> 200 * 1.075 = 215$ 
        # gekauft -> Prämie zahlen, aber begrenzt den maximalen verlust nach oben

        if T <= 0 or sigma <= 0:
            continue 
        # Sicherheitscheck - falls fehlerhafte Daten, trade überspringen 

        # Prämien berechnen 
        p_short_put  = black_scholes(S, K_short_put,
                                     T, r, sigma, "put")
        p_long_put   = black_scholes(S, K_long_put,
                                     T, r, sigma, "put")
        p_short_call = black_scholes(S, K_short_call,
                                     T, r, sigma, "call")
        p_long_call  = black_scholes(S, K_long_call,
                                     T, r, sigma, "call")
        # Alle 4 Optionspreise einzeln berechnen
        # Short Put + Short Call: weiter OTM -> höhere Prämie 
        # Long Put + Long Call: weiter OTM -> niedrigere Prämie
        
        # Netto-Prämie (eingenommen - bezahlt)
        net_premium = (p_short_put  - p_long_put +
                       p_short_call - p_long_call)

        # p_short_put  - p_long_put  = Prämie der Put-Seite  (immer positiv)
        # p_short_call - p_long_call = Prämie der Call-Seite (immer positiv)
        # Beispiel: 2.50 - 1.20 + 2.50 - 1.20 = 2.60$ Netto pro Aktie
        
        contracts      = max(int(cash / 10_000), 1) # max 1 kontrakt 
        premium_income = net_premium * contracts * 100 # gesamteinnahme
        # bsp: 2.60$ * 3 Kontrakte * 100 = 780$ direkt 

        cash          += premium_income

        # Max Verlust pro Kontrakt
        max_loss_per = (K_short_put - K_long_put) * 100
        # Max verlust pro kontrakt = breite spread * 100
        # Worst case: Kurs crasht durch beide put strikes durch 

        # Expiry
        expiry_idx  = min(i + dte, len(df) - 1)
        expiry_date = df.index[expiry_idx]
        S_expiry    = float(close.iloc[expiry_idx])

        # P&L bei Expiry berechnen
        put_spread_loss  = max(K_short_put - S_expiry, 0) - \
                           max(K_long_put  - S_expiry, 0)
        
        # Verlust put seite am verfallstag:
        # max was man zahlt - max was man bekommt 
        # Beispiel A: S_expiry = 175$ (unter beiden Strikes)
        #   max(190-175, 0) - max(185-175, 0) = 15 - 10 = 5$ → Max-Verlust erreicht
        #
        # Beispiel B: S_expiry = 188$ (zwischen Long und Short Put)
        #   max(190-188, 0) - max(185-188, 0) = 2 - 0 = 2$ → Teilverlust
        #
        # Beispiel C: S_expiry = 200$ (in der Profit Zone)
        #   max(190-200, 0) - max(185-200, 0) = 0 - 0 = 0$ → kein Verlust


        call_spread_loss = max(S_expiry - K_short_call, 0) - \
                           max(S_expiry - K_long_call,  0)
        
        # Beispiel A: S_expiry = 225$ (über beiden Call-Strikes)
        #   max(225-210, 0) - max(225-215, 0) = 15 - 10 = 5$ → Max-Verlust
        #
        # Beispiel B: S_expiry = 212$ (zwischen Short und Long Call)
        #   max(212-210, 0) - max(212-215, 0) = 2 - 0 = 2$ → Teilverlust
        #
        # Beispiel C: S_expiry = 200$ (in der Profit Zone)
        #   max(200-210, 0) - max(200-215, 0) = 0 - 0 = 0$ → kein Verlust
        
        total_loss = (put_spread_loss + call_spread_loss) * \
                      contracts * 100
        # beide seiten addiert * Kontrakte * 100
        # Kann nie größer sein als max_loss_per × contracts (Long Options als Deckel)

        pnl        = premium_income - total_loss
        cash      -= total_loss
        # Verlust vom Cash abziehen (Prämie wurde bereits am Anfang gutgeschrieben)

        if S_expiry < K_short_put: # kurs unter short put strike -> Put seite im verlust 
            outcome = "Put Breached"
        elif S_expiry > K_short_call: # Kurs über short call strike -> Call seite im verlust
            outcome = "Call Breached"
        else:
            outcome = "Profitable" # Kurs in Range -> volle prämie 

        trades.append({
            # Hinweis: Iron Condor nutzt ein normales Dict statt OptionTrade-Dataclass
            # weil er 4 Strikes hat — OptionTrade hat nur einen Strike-Slot
            "entry_date":    date,
            "expiry_date":   expiry_date,
            "stock_entry":   S,
            "stock_expiry":  S_expiry,
            "K_short_put":   K_short_put,
            "K_short_call":  K_short_call,
            "net_premium":   round(net_premium, 2),
            "premium_income":round(premium_income, 2),
            "pnl":           round(pnl, 2),
            "outcome":       outcome,
        })

        equity_curve.append(round(cash, 2))
        dates.append(expiry_date)

    equity    = pd.Series(equity_curve, index=dates)
    trades_df = pd.DataFrame(trades)

    return {
        "strategy": "Iron Condor",
        "equity":   equity,
        "trades":   trades_df,
        "capital":  capital,
    }

def compute_metrics(result: dict) -> dict:
    """Performance-Metriken für Options-Strategien."""
    equity  = result["equity"].dropna()
    trades  = result["trades"]
    capital = result["capital"]

    if len(equity) < 2:
        return {}
    
    returns   = equity.pct_change().dropna()
    years     = (equity.index[-1] - equity.index[0]).days / 365.25   # monatliche Daten
    total_ret = (equity.iloc[-1] / equity.iloc[0] - 1) * 100
    cagr      = ((equity.iloc[-1] / equity.iloc[0]) ** (1 / max(years, 0.1)) - 1) * 100

    sharpe    = (returns.mean() / returns.std() *
                 np.sqrt(12)) if returns.std() > 0 else 0
    
    rolling_max = equity.cummax()
    max_dd      = ((equity - rolling_max) /
                    rolling_max).min() * 100
    
    if not trades.empty and "pnl" in trades.columns:
        completed    = trades.dropna(subset=["pnl"])
        wins         = completed[completed["pnl"] > 0]
        losses       = completed[completed["pnl"] <= 0]
        win_rate     = len(wins) / len(completed) * 100 \
                       if len(completed) > 0 else 0
        avg_premium  = completed.get("premium",
                       pd.Series([0])).mean()
        total_premium = completed["pnl"][
            completed["pnl"] > 0
        ].sum()
        profit_factor = (
            wins["pnl"].sum() / abs(losses["pnl"].sum())
            if not losses.empty and losses["pnl"].sum() != 0
            else 0
        )

        if "outcome" in completed.columns:
            outcomes = completed["outcome"].value_counts()
        else:
            outcomes = pd.Series()
    else:
        win_rate = profit_factor = 0
        total_premium = 0
        outcomes = pd.Series()

    return {
        "Strategie":         result["strategy"],
        "Total Return (%)":  round(total_ret, 2),
        "CAGR (%)":          round(cagr, 2),
        "Sharpe":            round(sharpe, 2),
        "Max DD (%)":        round(max_dd, 2),
        "Win Rate (%)":      round(win_rate, 1),
        "Profit Factor":     round(profit_factor, 2),
        "Total Prämien ($)": round(total_premium, 0),
        "Trades":            len(trades),
    }

def print_tearsheet(metrics: dict) -> None:
    """Tearsheet im Terminal."""
    print(f"\n{'='*45}")
    print(f"  {metrics.get('Strategie', 'Strategie')}")
    print(f"{'='*45}")
    for k, v in metrics.items():
        if k != "Strategie":
            print(f"  {k:<22} {v:>15}")
    print(f"{'='*45}")

def plot_strategy_comparison(results: list) -> None:
    """
    Vergleicht alle Options-Strategien plus Buy & Hold.
    """
    colors = {
        "Covered Call":      "#2563eb",
        "Cash-Secured Put":  "#16a34a",
        "Iron Condor":       "#f59e0b",
        "Buy & Hold":        "#94a3b8",
    }

    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=False,
        vertical_spacing=0.08,
        row_heights=[0.65, 0.35],
        subplot_titles=[
            "Options Strategien — Equity Vergleich",
            "Monatliche Returns"
        ]
    )

    for result in results:
        name   = result["strategy"]
        equity = result["equity"].dropna()
        color  = colors.get(name, "#8b5cf6")
        lw     = 1.5 if name == "Buy & Hold" else 2
        dash   = "dot" if name == "Buy & Hold" else "solid"

        # Normalisieren auf Startkapital
        norm_equity = equity / equity.iloc[0] * result["capital"]

        fig.add_trace(go.Scatter(
            x=norm_equity.index,
            y=norm_equity.round(2),
            name=name,
            line=dict(color=color, width=lw, dash=dash)
        ), row=1, col=1)

        # Monatliche Returns
        monthly = equity.pct_change().dropna() * 100
        fig.add_trace(go.Bar(
            x=monthly.index,
            y=monthly.round(2),
            name=f"{name} Ret",
            marker_color=color,
            opacity=0.5,
            showlegend=False
        ), row=2, col=1)

    fig.add_hline(
        y=0, line_dash="dot",
        line_color="#1e293b",
        opacity=0.3, row=2, col=1
    )

    fig.update_layout(
        height=650,
        template="plotly_white",
        hovermode="x unified",
        legend=dict(orientation="h", y=1.02),
        margin=dict(l=0, r=0, t=50, b=0)
    )

    fig.update_yaxes(title_text="Kapital ($)", row=1, col=1)
    fig.update_yaxes(title_text="Return (%)",  row=2, col=1)

    fig.show()

def plot_trade_outcomes(result: dict) -> None:
    """
    Analysiert Trade-Outcomes und P&L Verteilung.
    """
    trades = result["trades"]
    name   = result["strategy"]

    if trades.empty or "pnl" not in trades.columns:
        return

    completed = trades.dropna(subset=["pnl"])

    fig = make_subplots(
        rows=1, cols=3,
        subplot_titles=[
            "P&L pro Trade ($)",
            "Outcome Verteilung",
            "Kumulativer P&L"
        ],
        horizontal_spacing=0.10,
        specs=[[{"type": "xy"}, {"type": "pie"}, {"type": "xy"}]]
    #        col 1 normal    col 2 Pie-Chart   col 3 normal
    )

    # P&L Bars
    pnl_colors = [
        "#16a34a" if v > 0 else "#ef4444"
        for v in completed["pnl"]
    ]
    fig.add_trace(go.Bar(
        x=list(range(len(completed))),
        y=completed["pnl"].round(2),
        marker_color=pnl_colors,
        name="P&L",
        showlegend=False
    ), row=1, col=1)

    fig.add_trace(go.Scatter(
        x=[0, len(completed) - 1],
        y=[0, 0],
        mode="lines",
        line=dict(color="#1e293b", width=1),
        showlegend=False
    ), row=1, col=1)

    # Outcome Pie
    if "outcome" in completed.columns:
        outcome_counts = completed["outcome"].value_counts()
        outcome_colors_map = {
            "Expired":     "#16a34a",
            "Profitable":  "#16a34a",
            "Assigned":    "#f59e0b",
            "Put Breached": "#ef4444",
            "Call Breached": "#ef4444",
        }
        fig.add_trace(go.Pie(
            labels=outcome_counts.index.tolist(),
            values=outcome_counts.values.tolist(),
            hole=0.4,
            marker_colors=[
                outcome_colors_map.get(k, "#94a3b8")
                for k in outcome_counts.index
            ],
            showlegend=True
        ), row=1, col=2)

    # Kumulativer P&L
    cum_pnl = completed["pnl"].cumsum()
    fig.add_trace(go.Scatter(
        x=list(range(len(cum_pnl))),
        y=cum_pnl.round(2),
        name="Kum. P&L",
        line=dict(color="#2563eb", width=2),
        fill="tozeroy",
        fillcolor="rgba(37,99,235,0.08)",
        showlegend=False
    ), row=1, col=3)

    fig.add_trace(go.Scatter(
        x=[0, len(cum_pnl) - 1],
        y=[0, 0],
        mode="lines",
        line=dict(color="#94a3b8", dash="dot", width=1),
        showlegend=False
    ), row=1, col=3)

    fig.update_layout(
        height=400,
        template="plotly_white",
        title=f"{name} — Trade Analyse",
        margin=dict(l=0, r=0, t=50, b=0)
    )

    fig.show()

def plot_condor_payoff(S: float, width: float = 0.05) -> None:
    """
    Iron Condor Payoff Diagram bei verschiedenen Volatilitäten.
 
    FIX BUG 2: for-loop wird jetzt tatsächlich genutzt um 3 Traces
    (eine pro Volatilität) in den Chart zu zeichnen.
    Vorher: loop berechnete net_credit aber plottet nie, nur σ=25% wurde genutzt.
    """
    wing = width / 2
    K1   = round(S * (1 - width - wing) / 5) * 5  # Long Put
    K2   = round(S * (1 - width)        / 5) * 5  # Short Put
    K3   = round(S * (1 + width)        / 5) * 5  # Short Call
    K4   = round(S * (1 + width + wing) / 5) * 5  # Long Call
 
    S_range = np.linspace(S * 0.75, S * 1.25, 500)
 
    # Payoff bei Expiry (identisch für alle Volatilitäten)
    payoff = (
        np.maximum(K1 - S_range, 0) - np.maximum(K2 - S_range, 0) -  # ← K1 und K2 getauscht
        np.maximum(S_range - K3, 0) + np.maximum(S_range - K4, 0)
    ) * 100
 
    fig = go.Figure()
 
    vola_configs = [
        (0.20, "#16a34a", "σ=20% (niedrig)"),
        (0.25, "#2563eb", "σ=25% (mittel)"),
        (0.30, "#f59e0b", "σ=30% (hoch)"),
        (0.40, "#ef4444", "σ=40% (sehr hoch)"),
    ]
 
    # FIX BUG 2: jede Volatilität kriegt ihre eigene Linie im Chart
    for sigma, color, label in vola_configs:
        T = 30 / 365
        r = 0.05
        net_credit = (
            black_scholes(S, K2, T, r, sigma, "put")  -
            black_scholes(S, K1, T, r, sigma, "put")  +
            black_scholes(S, K3, T, r, sigma, "call") -
            black_scholes(S, K4, T, r, sigma, "call")
        ) * 100
        # net_credit variiert mit Volatilität:
        # höhere Vola -> höhere Prämien auf beiden Seiten -> mehr Netto-Einnahme
        # aber auch mehr Risiko dass Strike getroffen wird
 
        total_pnl = payoff + net_credit
 
        fig.add_trace(go.Scatter(
            x=S_range, y=total_pnl.round(2),
            mode="lines", name=f"{label} (Prämie: ${net_credit:.0f})",
            line=dict(color=color, width=2)
        ))
        # Strike-Linien
    for K, label in [(K1, f"Long Put ${K1}"), (K2, f"Short Put ${K2}"),
                     (K3, f"Short Call ${K3}"), (K4, f"Long Call ${K4}")]:
        fig.add_vline(
            x=K, line_dash="dot", line_color="#94a3b8", opacity=0.7,
            annotation_text=label, annotation_position="top"
        )

    fig.add_hline(y=0, line_color="#1e293b", line_width=1.5)

    fig.update_layout(
        title=f"Iron Condor Payoff bei Expiry — Volatilitätsvergleich (S=${S:.0f})",
        xaxis_title="Kurs bei Expiry ($)",
        yaxis_title="P&L ($)",
        template="plotly_white",
        height=500,
        legend=dict(orientation="h", y=1.02),
        margin=dict(l=0, r=0, t=60, b=0)
    )
    fig.show()
 
    

if __name__ == "__main__":

    TICKER  = "SPY"
    CAPITAL = 50_000

    print(f"Tag 19 — Options Strategie Backtest: {TICKER}")
    print("=" * 55)

    df = load_data(TICKER, "5y")
    print(f"Daten: {len(df)} Handelstage")

    # --- Covered Call ---
    print("\nBackteste Covered Call...")
    cc_result  = backtest_covered_call(df, capital=CAPITAL)
    cc_metrics = compute_metrics(cc_result)
    print_tearsheet(cc_metrics)
    plot_trade_outcomes(cc_result)

    # --- Cash-Secured Put ---
    print("\nBackteste Cash-Secured Put...")
    csp_result  = backtest_cash_secured_put(df, capital=CAPITAL)
    csp_metrics = compute_metrics(csp_result)
    print_tearsheet(csp_metrics)
    plot_trade_outcomes(csp_result)

    # --- Iron Condor ---
    print("\nBackteste Iron Condor...")
    ic_result  = backtest_iron_condor(df, capital=CAPITAL)
    ic_metrics = compute_metrics(ic_result)
    print_tearsheet(ic_metrics)

    # --- Buy & Hold Benchmark ---
    print("\nBerechne Buy & Hold Benchmark...")
    close      = df["Close"].squeeze()
    bah_equity = (close / close.iloc[0]) * CAPITAL
    bah_result = {
        "strategy": "Buy & Hold",
        "equity":   bah_equity.resample("ME").last(),
        "trades":   pd.DataFrame(),
        "capital":  CAPITAL,
    }

    # --- Vergleich ---
    all_results = [cc_result, csp_result,
                   ic_result, bah_result]

    print("\n" + "="*65)
    print("STRATEGY COMPARISON")
    print("="*65)

    all_metrics = [cc_metrics, csp_metrics,
                   ic_metrics,
                   compute_metrics(bah_result)]

    cols = ["Strategie", "Total Return (%)",
            "CAGR (%)", "Sharpe",
            "Max DD (%)", "Win Rate (%)"]

    summary = pd.DataFrame(all_metrics)
    if not summary.empty and "Strategie" in summary.columns:
        print(summary[
            [c for c in cols if c in summary.columns]
        ].to_string(index=False))

    plot_strategy_comparison(all_results)

    # --- Iron Condor Payoff Diagram ---
    current_price = float(df["Close"].iloc[-1])
    plot_condor_payoff(current_price, width=0.05)

    # --- Delta Sensitivität ---
    print("\n--- Covered Call: Delta Sensitivität ---")
    print(f"  {'Delta':>8} {'Win Rate':>10} "
          f"{'CAGR':>8} {'Sharpe':>8}")
    print("  " + "-"*38)

    for delta in [0.20, 0.25, 0.30, 0.35, 0.40]:
        r = backtest_covered_call(
            df, delta_target=delta,
            capital=CAPITAL
        )
        m = compute_metrics(r)
        if m:
            print(f"  {delta:>8.2f}"
                  f"  {m.get('Win Rate (%)', 0):>8.1f}%"
                  f"  {m.get('CAGR (%)', 0):>7.1f}%"
                  f"  {m.get('Sharpe', 0):>7.2f}")

    # Export
    if not cc_result["trades"].empty:
        cc_result["trades"].to_csv(
            "day19_covered_call_trades.csv", index=False
        )
    print("\nGespeichert: day19_covered_call_trades.csv")