# Was ist dcf überhaupt?
# die zukünftigen gewinne werden geschätzt und auf den heutigen Wert zurückgerechnet 
# Problem: jede Annahme kann falsch sein und heute lösen wir das Problem


"""
Day 22 — Monte Carlo DCF Simulation

Das Problem mit normalem DCF:
    Du gibst EINE Wachstumsrate ein → du bekommst EINEN Wert.
    Aber kein Analyst weiß genau was die Wachstumsrate ist.
    Ein einziger Wert täuscht Präzision vor die nicht existiert.

Monte Carlo Lösung:
    Statt einer Zahl: Wahrscheinlichkeitsverteilungen.
    Wachstum ist nicht 10% — es ist normalverteilt um 10%
    mit Standardabweichung von 3%.
    
    10.000 Simulationen → 10.000 verschiedene Unternehmenswerte.
    Ergebnis: "Mit 80% Wahrscheinlichkeit liegt der faire Wert
    zwischen $145 und $195."

    Das ist wie Profis wirklich denken.

Was du heute baust:
    1. Klassischer DCF als Basis
    2. Monte Carlo über alle unsicheren Parameter
    3. Sensitivitätsanalyse (Tornado Chart)
    4. Scenario Analysis (Bull / Base / Bear)
    5. Margin of Safety Berechnung
"""

import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy import stats
from dataclasses import dataclass, field
from typing import Optional
import warnings
warnings.filterwarnings("ignore")

@dataclass # ohne müsste man noch __init__ und __repr__ schrieben 
# -> Man kann dcfparams(fcf_mean, wacc_mean) benutzen 
class DCFParams:
    """
    Alle dcf parameter an einem ort 

    jeder parameter aht:
        mean: basisannahme (was man glaubt)
        std: Unsicherheit (wie sicher man ist)

    Kleine std -> man ist sehr sicher 
    Große std -> Große unsicherheit -> breite verteilung im Ergebnis 

    Das ist ehrlicher als ein einziger Wert 


    """

    # Free cash flow (aktuell, in mrd)
    fcf_mean: float = 100.0
    fcf_std: float = 10.0


    # Phase 1: Hohes Wachstum (Jahre 1-5)
    growth_1_mean:    float = 0.12      # 12% erwartet
    growth_1_std:     float = 0.04      # ±4% Unsicherheit

    """
    Wahrscheinlichkeit
       │     ┌───┐
       │   ┌─┘   └─┐
       │  ─┘       └─
       └─────────────────── Wachstum
         8%  12%  16%
         ←    μ    →
              ±2σ = ±8%
    """

    # Phase 2: Übergang (Jahre 6-10)
    growth_2_mean:    float = 0.07
    growth_2_std:     float = 0.03

    # Terminal Growth Rate (nach Jahr 10)
    terminal_mean:    float = 0.03
    terminal_std:     float = 0.01

    """
    Wachstum
   │
12%│────────┐
   │         │
7% │         └────────┐
   │                   │
3% │                   └──────────────────── (ewig)
   └─────────────────────────────────────── Zeit
    Jahr 1-5    Jahr 6-10    Jahr 10+
    Expansion   Übergang     Terminal

    Warum sinkt Wachstum über Zeit? Reversion to the Mean — kein Unternehmen wächst ewig mit 12%. 
    Irgendwann nähert es sich dem BIP-Wachstum (~3%).
    """

    # WACC — Weighted Average Cost of Capital
    wacc_mean:        float = 0.09      # 9% erwartet
    wacc_std:         float = 0.015     # ±1.5% Unsicherheit

    """
    WACC 8%  → hoher Barwert → Aktie erscheint günstig -> Mindestens 8% erwirtschaften um Kosten zu decken 
    WACC 10% → niedriger Barwert → Aktie erscheint teuer
    Wacc zeigt, wieviel das unternehmen erwirtschaften muss um Kapitalkosten zu decken 

    Unterschied von 2% → Bewertung kann sich um 30-40% ändern!
    Deshalb hat er auch eine std — niemand kennt den echten WACC exakt.
    """

    # Shares Outstanding (Milliarden)
    shares:           float = 1.0

    # Net Debt (Milliarden, positiv = Schulden)
    net_debt:         float = 0.0

    # Projektionsjahre
    years_1:          int   = 5
    years_2:          int   = 5

def get_real_params(ticker: str) -> DCFParams:
    """
    Lädt echte Fundamentaldaten von yfinance
    und baut daraus DCF Parameter.

    Nicht alle Felder sind immer verfügbar —
    Fallback-Werte wenn yfinance nichts zurückgibt.
    """
    try:
        stock = yf.Ticker(ticker)
        info  = stock.info # Dictionary mit hunderten Kennzahlen von yfinance

        # Free Cash Flow
        fcf   = info.get("freeCashflow", None) # gibt Nine zurück falls key nicht existiert 
        if fcf is None or fcf == 0:
            # Alternativ: Operating CF - Capex schätzen
            op_cf = info.get("operatingCashflow", 0)
            capex = info.get("capitalExpenditures", 0)
            fcf   = op_cf + capex   # capex ist negativ in yfinance -> 120B +- 25B
        fcf_b = (fcf or 1e9) / 1e9  # in Milliarden - wenn fcf = 0 nimm 1e9 als fallback -> 1

        # Shares Outstanding
        shares = (info.get("sharesOutstanding", 1e9)) / 1e9

        # Net Debt
        total_debt = info.get("totalDebt",  0) or 0
        cash       = info.get("totalCash",  0) or 0
        net_debt   = (total_debt - cash) / 1e9 # schulden - cash in Milliarden

        # Revenue Growth als Proxy für FCF Growth
        rev_growth = info.get("revenueGrowth", 0.10) or 0.10

        # Analyst Target als Sanity Check
        target = info.get("targetMeanPrice", None) # Nur zum vergleich analystenmeinung 

        print(f"\nFundamentaldaten {ticker}:")
        print(f"  FCF:           ${fcf_b:.2f}B")
        print(f"  Shares:        {shares:.2f}B")
        print(f"  Net Debt:      ${net_debt:.2f}B")
        print(f"  Rev. Growth:   {rev_growth*100:.1f}%")
        if target:
            print(f"  Analyst Target: ${target:.2f}")

            return DCFParams(
            fcf_mean      = max(fcf_b, 0.1),
            fcf_std       = max(fcf_b * 0.15, 0.1),
            growth_1_mean = min(max(rev_growth, 0.03), 0.40), # min. 3% - kein negatives wachstum als basis 
            growth_1_std  = 0.04,
            growth_2_mean = min(max(rev_growth * 0.6, 0.03), 0.20), # max 40% - kein hyperwachstum annehmen
            # rev_growth * 0,6 berechnet natürliche evrlangsamung 
            growth_2_std  = 0.03,
            terminal_mean = 0.03,
            terminal_std  = 0.008,
            wacc_mean     = 0.09,
            wacc_std      = 0.015,
            shares        = max(shares, 0.01),
            net_debt      = net_debt,
            )
        
    except Exception as e:
        print(f"Fehler beim Laden: {e} — nutze Default-Parameter")
        return DCFParams()
    

# Einzelner DCF , gibt einen aktienwert aus -> monte carlo läuft das 10000 mak durch 
def single_dcf(params:       DCFParams,
               fcf:          float,
               growth_1:     float,
               growth_2:     float,
               terminal:     float,
               wacc:         float) -> dict:
    """
    Berechnet einen einzelnen DCF-Wert mit gegebenen Parametern.

    Zwei-Phasen Modell:
        Phase 1 (Jahre 1-5):  Hohes Wachstum
        Phase 2 (Jahre 6-10): Rückgang zum Normalwachstum
        Terminal Value:       Ewige Rente ab Jahr 11

    Terminal Value Formel (Gordon Growth):
        TV = FCF_11 / (WACC - g)

    Discounting:
        Jeder zukünftige Cash Flow wird auf heute abdiskontiert.
        PV = CF_t / (1 + WACC)^t
        ### Ein Dollar in 10 Jahren ist heute weniger wert. ###
    """

    cash_flows = []
    current_fcf = fcf # startpunkt: heutiger fcf

    # Phase 1: Hohes Wachstum
    for year in range(1, params.years_1 + 1): # range 1,6 -> 1,2,3,4,5
        current_fcf *= (1 + growth_1)         # FCF wächst jedes Jahr mit zinseszins
        pv = current_fcf / (1 + wacc) ** year # # auf heute abdiskontieren
        # Je weiter in der Zukunft, desto stärker wird abgezinst — 
        # ein Dollar in Jahr 10 ist heute nur 42 Cent wert bei 9% WACC.
        cash_flows.append({
            "year":  year,
            "fcf":   round(current_fcf, 4),
            "pv":    round(pv, 4),
            "phase": "Phase 1"
        })

    # Phase 2: Übergang
    for year in range(params.years_1 + 1,
                      params.years_1 + params.years_2 + 1): # range 6 - 11
        current_fcf *= (1 + growth_2) # niedrigeres wachstum 
        # current läuft von davor weiter -> fcf wächst weiter nur langsamer 
        pv = current_fcf / (1 + wacc) ** year # gleiche diskontierung 
        cash_flows.append({
            "year":  year,
            "fcf":   round(current_fcf, 4),
            "pv":    round(pv, 4),
            "phase": "Phase 2"
        })

    # Terminal Value
    total_years  = params.years_1 + params.years_2 # 10
    terminal_fcf = current_fcf * (1 + terminal) # fcf in jahr 11

    if wacc <= terminal:
        # Mathematisch nicht definiert — Fallback da gerechnet wird wacc / terminal
        tv = current_fcf * 15   # vereinfachter Multiple falls fehler
    else:
        tv = terminal_fcf / (wacc - terminal) # das ist die formel ****
    
    pv_tv = tv / (1+wacc) ** total_years
    #       TV  /   1.09^10

    # Summen
    pv_cfs      = sum(cf["pv"] for cf in cash_flows) # summe aller pv cashflows
    enterprise  = pv_cfs + pv_tv # gesamtwert Unternehmen
    equity      = enterprise - params.net_debt # Wert für aktionäre 
    price       = equity / params.shares if params.shares > 0 else 0 # pro aktie 

    """
    
    PV Cash Flows:    542B
    PV Terminal:      901B
    ─────────────────────
    Enterprise Value: 1.443B
    - Net Debt:        -55B  (Apple hat mehr Cash als Schulden)
    ─────────────────────
    Equity Value:    1.498B
    / Shares:        15.5B Aktien
    ─────────────────────
    Intrinsic Price: $96.65 pro Aktie   
    """
    return {
        "cash_flows":       cash_flows,
        "pv_cash_flows":    round(pv_cfs, 4),
        "terminal_value":   round(tv, 4),
        "pv_terminal":      round(pv_tv, 4),
        "enterprise_value": round(enterprise, 4),
        "equity_value":     round(equity, 4),
        "intrinsic_price":  round(price, 4),
        "tv_pct":           round(pv_tv / enterprise * 100, 1)
                            if enterprise > 0 else 0,
    }
"""
wie viel prozent des unternehmenswertes kommt aus dem Terminal Value?
tv_pct = 60%  → normal, akzeptabel
tv_pct = 85%  → ⚠️ du bezahlst fast nur für weit entfernte Zukunft
tv_pct = 95%  → 🚨 extrem sensitiv auf WACC/Terminal Annahmen
-> je höher desto unsicherer bewertung, da annahmen über 50+ jahre basiert 
"""

def monte_carlo_dcf(params:       DCFParams,
                    n_sims:       int   = 10_000,
                    current_price: Optional[float] = None,
                    seed:         int   = 42) -> pd.DataFrame:
     # seed fixiert die zufallszahlen -> selben zufallszahlen 
     # seed macht es reproduzierbar 
    """
    Statt: "Die Aktie ist $245 wert"
    Dafür: "Mit 90% Wahrscheinlichkeit liegt der Wert zwischen $180 und $310"


    Monte Carlo Simulation über alle DCF Parameter.

    Für jede Simulation:
        1. Ziehe zufällige Werte aus Normalverteilungen
        2. Berechne DCF
        3. Speichere Ergebnis

    Daraus: Verteilung, Konfidenzintervalle, Wahrscheinlichkeiten.

    Korrelation zwischen Parametern:
        Hohes Wachstum → meist auch höherer WACC (Markt fordert mehr)
        Das berücksichtigen wir mit einer Korrelationsstruktur.
    """
    np.random.seed(seed)

    # Korrelationsmatrix der Parameter
    # [fcf, growth_1, growth_2, terminal, wacc]
    # Positiv: Wachstum korreliert mit sich selbst
    # Negativ: Wachstum korreliert negativ mit WACC (riskanter)
    """wird hier gegeben,um es ökonomisch konsistent zu machen """
    corr_matrix = np.array([
        [1.00,  0.20,  0.15,  0.05,  0.10],  # fcf
        [0.20,  1.00,  0.70,  0.30, -0.20],  # growth_1
        [0.15,  0.70,  1.00,  0.40, -0.15],  # growth_2
        [0.05,  0.30,  0.40,  1.00,  0.10],  # terminal
        [0.10, -0.20, -0.15,  0.10,  1.00],  # wacc
    ])
    """
    growth_1 / growth_2 Korrelation = 0.70  ← stark positiv
    "Ein Unternehmen das schnell wächst, wächst meist in beiden Phasen schnell" 

    growth_1 / wacc Korrelation = -0.20  ← leicht negativ
    "Hohes Wachstum = höheres Risiko = Markt fordert höheren WACC"
    """

    # Cholesky Decomposition für korrelierte Zufallszahlen
    L = np.linalg.cholesky(corr_matrix)

    # Unkorrellierte Zufallszahlen
    raw = np.random.standard_normal((n_sims, 5)) # 10.000 × 5 unkorrelierte Zahlen

    # Korrelierte Zufallszahlen
    correlated = raw @ L.T # Korrelationsstruktur reinbringen
    # Wenn Z∼N(0,I) dann gilt: LZ∼N(0,Σ)


    """
    [1.00  0.20  ...]        [1.00  0     ...]   [1.00  0.20  ...]
    [0.20  1.00  ...]   =    [0.20  0.98  ...]  ×  [0     0.98  ...]
    ...                       ...                   ...
             Σ          =          L           ×       L^T
    """

    # Parameter-Arrays aus Normalverteilungen -> standardisierungsformel umgekehrt
    means = np.array([
        params.fcf_mean,
        params.growth_1_mean,
        params.growth_2_mean,
        params.terminal_mean,
        params.wacc_mean,
    ])
    stds = np.array([
        params.fcf_std,
        params.growth_1_std,
        params.growth_2_std,
        params.terminal_std,
        params.wacc_std,
    ])

    # Simulierte Parameter
    sim_params = means + correlated * stds
    # X = μ + σ * Z

    # Constraints: Rahmenbedingungen
    sim_fcf      = np.maximum(sim_params[:, 0], 0.01) # setzt alles unter richtwert auf 0.01
    sim_growth_1 = np.clip(sim_params[:, 1], -0.10, 0.60) # 2 grenzen 
    sim_growth_2 = np.clip(sim_params[:, 2], -0.05, 0.40)
    sim_terminal = np.clip(sim_params[:, 3],  0.00, 0.05)
    sim_wacc     = np.clip(sim_params[:, 4],  0.04, 0.20)

    # Sicherstellen: terminal < wacc (sonst undefiniert)
    valid = sim_terminal < sim_wacc - 0.005
    sim_terminal = np.where( # np.where korrigiert die werte die falsch sind direkt im array
        valid, sim_terminal, # wenn gültig behalten 
        sim_wacc - 0.01 # wenn ungültig: wacc 1%
    )

    # Simulationen ausführen
    results = []
    for i in range(n_sims):
        try:
            result = single_dcf(
                params    = params,
                fcf       = sim_fcf[i],
                growth_1  = sim_growth_1[i],
                growth_2  = sim_growth_2[i],
                terminal  = sim_terminal[i],
                wacc      = sim_wacc[i],
            )

            results.append({
                "price":       result["intrinsic_price"],
                "ev":          result["enterprise_value"],
                "equity":      result["equity_value"],
                "pv_cfs":      result["pv_cash_flows"],
                "pv_tv":       result["pv_terminal"],
                "tv_pct":      result["tv_pct"],
                "fcf":         sim_fcf[i],
                "growth_1":    sim_growth_1[i] * 100,
                "growth_2":    sim_growth_2[i] * 100,
                "terminal":    sim_terminal[i] * 100,
                "wacc":        sim_wacc[i] * 100,
            })
        except Exception:
            pass

    df = pd.DataFrame(results)

    # Extremwerte entfernen (außerhalb 1%-99% Quantil) um nicht unnötig zu rechnen
    price_low  = df["price"].quantile(0.01)
    price_high = df["price"].quantile(0.99)
    df = df[
        (df["price"] >= price_low) &
        (df["price"] <= price_high)
    ].reset_index(drop=True)

    return df

def dcf_statistics(sim_df:        pd.DataFrame,
                   current_price: Optional[float] = None,
                   margin_of_safety: float = 0.25) -> dict:
    """
    Analysiert die Monte Carlo Ergebnisse.

    Margin of Safety (Graham):
        Kaufe nur wenn Kurs mindestens 25% unter
        dem fairen Wert liegt.
        Das ist Schutz gegen Fehler in deinen Annahmen.

        MOS = 25% → Kauflimit = fairer Wert × 0.75
    """
    prices = sim_df["price"]

    mean_price   = prices.mean()
    median_price = prices.median()
    std_price    = prices.std()

    # Konfidenzintervalle - .quantile gibt 2 werte zurück - den 10. und 90. perzentil
    ci_80_low,  ci_80_high  = prices.quantile([0.10, 0.90]) # 80% aller simulationen drin 
    ci_90_low,  ci_90_high  = prices.quantile([0.05, 0.95]) # 90% aller simulationen drin 
    ci_95_low,  ci_95_high  = prices.quantile([0.025, 0.975]) # 95% aller simulationen drin
    # lässt einschätzen wie wahrscheinlich 

    # Margin of Safety Kauflimit nur wenn der kurs mind. 25% unter schätzung wird gekauft 
    mos_price = median_price * (1 - margin_of_safety)
    # margin_of_safety = 0.25
    # median = $241
    # mos_price = 241 × 0.75 = $180.75  

    stats = {
        "mean":         round(mean_price, 2),
        "median":       round(median_price, 2),
        "std":          round(std_price, 2),
        "ci_80":        (round(ci_80_low, 2),  round(ci_80_high, 2)),
        "ci_90":        (round(ci_90_low, 2),  round(ci_90_high, 2)),
        "ci_95":        (round(ci_95_low, 2),  round(ci_95_high, 2)),
        "mos_price":    round(mos_price, 2),
        "p10":          round(prices.quantile(0.10), 2),
        "p25":          round(prices.quantile(0.25), 2),
        "p75":          round(prices.quantile(0.75), 2),
        "p90":          round(prices.quantile(0.90), 2),
        "tv_pct_mean":  round(sim_df["tv_pct"].mean(), 1),
        "n_sims":       len(sim_df),
    }

    # Upside/Downside wenn Kurs bekannt
    if current_price is not None:
        upside_pct = (median_price / current_price - 1) * 100
        buy_signal = current_price <= mos_price

        # Wahrscheinlichkeit dass Aktie unterbewertet ist
        prob_undervalued = (prices > current_price).mean() * 100 
        # gibt aus: [True, True, False, True, False, ...] und vergleicht 
        # bsp 67% True -> Aktie wahrscheinlich undervalued


        stats.update({
            "current_price":    round(current_price, 2),
            "upside_pct":       round(upside_pct, 1),
            "buy_signal":       buy_signal,
            "prob_undervalued": round(prob_undervalued, 1),
        })

    return stats


def scenario_analysis(params:        DCFParams,
                       current_price: Optional[float] = None) -> pd.DataFrame:
    """
    Bull / Base / Bear Scenario.

    Drei deterministische DCF Szenarien:
        Bear:  Pessimistische Annahmen
        Base:  Erwartete Annahmen (Mittelwert)
        Bull:  Optimistische Annahmen

    Das ist Standard in jedem Investment-Banking Modell.
    Monte Carlo zeigt die Verteilung dazwischen.
    """
    scenarios = {
        "Bear 🐻": { # schlechtesten 2.5% der wachstumsszenarien
            "fcf":      params.fcf_mean * 0.85, # 15% niedruger FCF
            "growth_1": params.growth_1_mean - 2 * params.growth_1_std, # -2σ -> 95% aller fälle
            "growth_2": params.growth_2_mean - 2 * params.growth_2_std, # +2σ
            "terminal": params.terminal_mean - params.terminal_std,
            "wacc":     params.wacc_mean     + 2 * params.wacc_std,
        },
        "Base 📊": {
            "fcf":      params.fcf_mean, # Mittelwert
            "growth_1": params.growth_1_mean,
            "growth_2": params.growth_2_mean,
            "terminal": params.terminal_mean,
            "wacc":     params.wacc_mean,
        },
        "Bull 🐂": { # besten 2.5% der wachstumsszenarien
            "fcf":      params.fcf_mean * 1.15, # 15% höherer FCF
            "growth_1": params.growth_1_mean + 2 * params.growth_1_std, # +2σ
            "growth_2": params.growth_2_mean + 2 * params.growth_2_std, # -2σ
            "terminal": params.terminal_mean + params.terminal_std,
            "wacc":     params.wacc_mean     - 2 * params.wacc_std,
        },
    }

    rows = []
    for name, s in scenarios.items():
        # Constraints
        s["wacc"]     = max(s["wacc"], 0.05)
        s["terminal"] = min(s["terminal"], s["wacc"] - 0.01)
        s["growth_1"] = max(s["growth_1"], -0.05)
        s["growth_2"] = max(s["growth_2"], -0.02)

        result = single_dcf(params, **s) # **s entpackt dictionary
        """
        s = {"fcf": 85.0, "growth_1": 0.04, "wacc": 0.12}

        single_dcf(params, **s)
        # ist identisch zu:
        single_dcf(params, fcf=85.0, growth_1=0.04, wacc=0.12)
        """
        price  = result["intrinsic_price"]
        """
        Szenario   FCF    Growth1   WACC    Preis   Upside
        Bear 🐻    85B    4.0%      12.0%   $142    -35%
        Base 📊   100B    12.0%     9.0%    $241     +9%
        Bull 🐂   115B    20.0%     6.0%    $398    +81%
        """

        row = {
            "Szenario":      name,
            "FCF ($B)":      round(s["fcf"], 2),
            "Growth 1 (%)":  round(s["growth_1"] * 100, 1),
            "Growth 2 (%)":  round(s["growth_2"] * 100, 1),
            "Terminal (%)":  round(s["terminal"] * 100, 1),
            "WACC (%)":      round(s["wacc"]     * 100, 1),
            "EV ($B)":       result["enterprise_value"],
            "Preis ($)":     round(price, 2),
        }

        if current_price:
            row["Upside (%)"] = round(
                (price / current_price - 1) * 100, 1
            )

        rows.append(row)

    return pd.DataFrame(rows)


def sensitivity_analysis(params:        DCFParams, 
                          current_price: Optional[float] = None) -> pd.DataFrame:
    """
    Sensitivitätsanalyse — welcher Parameter beeinflusst den Wert am stärksten?

    Methode: One-at-a-time (OAT)
        Variiere jeden Parameter ±1 Std.
        Halte alle anderen konstant.
        Miss wie stark sich der Preis ändert.

    Ergebnis: Tornado Chart — größter Einfluss oben.

    Das ist entscheidend für:
        1. Wo solltest du deine Research-Zeit investieren?
        2. Welche Annahmen musst du am genauesten machen?
    """
    # Base Case
    base = single_dcf(
        params,
        fcf      = params.fcf_mean,
        growth_1 = params.growth_1_mean,
        growth_2 = params.growth_2_mean,
        terminal = params.terminal_mean,
        wacc     = params.wacc_mean,
    )
    base_price = base["intrinsic_price"]

    param_configs = [ # erhöhe oder senke jeden parameter um 1 Std, aber halte anderen konstant
        ("FCF ($B)",      "fcf",
         params.fcf_mean,      params.fcf_std,
         False),
        ("Wachstum Ph.1", "growth_1",
         params.growth_1_mean, params.growth_1_std,
         False),
        ("Wachstum Ph.2", "growth_2",
         params.growth_2_mean, params.growth_2_std,
         False),
        ("Terminal Rate", "terminal",
         params.terminal_mean, params.terminal_std,
         False),
        ("WACC",          "wacc",
         params.wacc_mean,     params.wacc_std,
         True),   # True = negativer Effekt wenn steigt - WACC ist invers zu preis
    ]

    rows = []
    for label, param, mean, std, inverse in param_configs:
        # High Case: +1 Std - alle parameter am mittelwert festhalten
        kwargs_high = {
            "fcf":      params.fcf_mean,
            "growth_1": params.growth_1_mean,
            "growth_2": params.growth_2_mean,
            "terminal": params.terminal_mean,
            "wacc":     params.wacc_mean,
        }
        kwargs_low = kwargs_high.copy() # copy erstellen

        kwargs_high[param] = mean + std # selbsterklärend
        kwargs_low[param]  = mean - std

        # Constraints
        for kw in [kwargs_high, kwargs_low]:
            kw["wacc"]     = max(kw["wacc"], 0.05)
            kw["terminal"] = min(kw["terminal"],
                                  kw["wacc"] - 0.01)

        high_result = single_dcf(params, **kwargs_high)
        low_result  = single_dcf(params, **kwargs_low)

        high_price = high_result["intrinsic_price"]
        low_price  = low_result["intrinsic_price"]

        if inverse:
            high_price, low_price = low_price, high_price

        rows.append({
            "Parameter":   label,
            "Basis ($)":   round(base_price, 2),
            "Low ($)":     round(low_price, 2),
            "High ($)":    round(high_price, 2),
            "Range ($)":   round(high_price - low_price, 2),
            "Low Δ%":      round((low_price  / base_price - 1) * 100, 1),
            "High Δ%":     round((high_price / base_price - 1) * 100, 1),
        })

    df = pd.DataFrame(rows).sort_values( # gibt die range die nach unten und oben ist 
        "Range ($)", ascending=False
    ).reset_index(drop=True)

    return df

    """
    Parameter       Low($)  High($)  Range($)
    WACC             $198    $301     $103    ← größter Einfluss (oben)
    Wachstum Ph.1    $198    $289     $91
    Terminal Rate    $221    $265     $44
    FCF              $229    $253     $24     ← kleinster Einfluss (unten)
    """

def plot_monte_carlo_distribution(sim_df:        pd.DataFrame,
                                   stats:         dict,
                                   ticker:        str) -> None:
    """
    Verteilung der simulierten Preise.

    Zeigt:
        Histogramm mit Normalverteilungs-Overlay
        Konfidenzintervalle als farbige Zonen
        Aktueller Kurs als vertikale Linie
        Margin of Safety Kauflimit
    """
    prices = sim_df["price"]

    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=[
            f"{ticker} — Monte Carlo Preisverteilung "
            f"({stats['n_sims']:,} Simulationen)",
            "Parameter Verteilungen",
            "WACC vs. Intrinsic Price",
            "Growth Phase 1 vs. Intrinsic Price"
        ],
        vertical_spacing=0.12,
        horizontal_spacing=0.10
    )

    # --- Panel 1: Preisverteilung ---
    fig.add_trace(go.Histogram(
        x=prices,
        nbinsx=80,
        name="Simulierte Preise",
        marker_color="#3b82f6",
        opacity=0.7,
        showlegend=True
    ), row=1, col=1)

    # KDE Overlay
    x_range = np.linspace(prices.min(), prices.max(), 300)
    kde     = stats_module_kde(prices, x_range) # geglättete kurve über histogramm

    fig.add_trace(go.Scatter(
        x=x_range, y=kde,
        name="Verteilung",
        line=dict(color="#1e293b", width=2),
        yaxis="y2",
        showlegend=False
    ), row=1, col=1)

    # Konfidenzintervall Zonen
    ci_colors = [
        (stats["ci_95"], "rgba(239,68,68,0.10)",   "95% CI", "top right"),
        (stats["ci_90"], "rgba(245,158,11,0.15)",  "90% CI", "bottom right"),
        (stats["ci_80"], "rgba(37,99,235,0.20)",   "80% CI", "bottom left"),
    ]
    for (lo, hi), fill_color, ci_name, pos in ci_colors:
        fig.add_vrect(
            x0=lo, x1=hi,
            fillcolor=fill_color,
            layer="below", line_width=0,
            annotation_text=ci_name,
            annotation_position=pos,
            row=1, col=1
        )

        """
        |░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░|  95% CI (breit, hellrot)
            |▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒|      90% CI (mittel, orange)
                |████████████████|         80% CI (schmal, blau)
        $140      $180    $241    $310      $340
        """

    # Median Linie
    fig.add_vline(
        x=stats["median"],
        line_dash="solid",
        line_color="#16a34a",
        line_width=2,
        annotation_text=f"Median: ${stats['median']:.0f}",
        annotation_position="top right",
        row=1, col=1
    )

    # Aktueller Kurs
    if "current_price" in stats:
        cp = stats["current_price"]
        color = "#16a34a" if stats.get("buy_signal") \
                else "#ef4444" # Farbe abhängig vom buy signal 
        fig.add_vline(
            x=cp,
            line_dash="dash",
            line_color=color,
            line_width=2,
            annotation_text=f"Kurs: ${cp:.0f}",
            annotation_position="bottom left",
            row=1, col=1
        )

    # MOS Kauflimit
    fig.add_vline(
        x=stats["mos_price"],
        line_dash="dot",
        line_color="#f59e0b",
        line_width=1.5,
        annotation_text=f"MOS Limit: ${stats['mos_price']:.0f}",
        annotation_position="bottom right",
        row=1, col=1
    )

    # --- Panel 2: Parameter Verteilungen ---
    param_cols = {
        "WACC (%)":     "#ef4444",
        "growth_1":     "#16a34a",
        "terminal":     "#8b5cf6",
    }
    for col_name, color in param_cols.items():
        if col_name in sim_df.columns: # schutz falls eine spalte fehlt
            fig.add_trace(go.Histogram(
                x=sim_df[col_name],
                nbinsx=40,
                name=col_name,
                marker_color=color,
                opacity=0.6,
            ), row=1, col=2)

    # --- Panel 3: WACC vs Preis ---
    sample = sim_df.sample(min(1000, len(sim_df))) # nur 1000 punkte, da browser sonst langsam wird

    fig.add_trace(go.Scatter(
        x=sample["wacc"],
        y=sample["price"],
        mode="markers",
        name="WACC vs Preis",
        marker=dict(
            color=sample["price"],
            colorscale="RdYlGn", # red-yellow-green colorscale
            size=4, opacity=0.5,
            showscale=False
        ),
        showlegend=False
    ), row=2, col=1)

    # --- Panel 4: Growth 1 vs Preis ---
    fig.add_trace(go.Scatter(
        x=sample["growth_1"],
        y=sample["price"],
        mode="markers",
        name="Growth vs Preis",
        marker=dict(
            color=sample["price"],
            colorscale="RdYlGn",
            size=4, opacity=0.5,
            showscale=False
        ),
        showlegend=False
    ), row=2, col=2)

    fig.update_layout(
        height=700,
        template="plotly_white",
        legend=dict(orientation="h", y=-0.05 ,x =0.0),
        margin=dict(l=0, r=0, t=60, b=60)
    )

    fig.update_xaxes(title_text="Intrinsic Value ($)", row=1, col=1)
    fig.update_xaxes(title_text="Parameter Wert",      row=1, col=2)
    fig.update_xaxes(title_text="WACC (%)",            row=2, col=1)
    fig.update_xaxes(title_text="Growth Ph.1 (%)",     row=2, col=2)
    fig.update_yaxes(title_text="Häufigkeit",          row=1, col=1)
    fig.update_yaxes(title_text="Preis ($)",           row=2, col=1)
    fig.update_yaxes(title_text="Preis ($)",           row=2, col=2)

    fig.show()


def stats_module_kde(data: pd.Series,
                     x:    np.ndarray) -> np.ndarray:
    """Kernel Density Estimation für Overlay-Kurve."""
    kernel = stats.gaussian_kde(data)
    # Skalieren auf Histogram-Höhe
    bin_width = (data.max() - data.min()) / 80
    return kernel(x) * len(data) * bin_width 
# * len und * bin, da kernel werte von 0-1 gibt aber histogram bis zum preis geht 

# stats.gaussian_kde — legt um jeden der 10.000 Datenpunkte eine kleine Gauß-Glocke und summiert alle auf. 
# Das Ergebnis ist eine geglättete Dichtefunktion.


def plot_tornado_chart(sens_df: pd.DataFrame,
                       ticker:  str) -> None:
    """
    Tornado Chart — Sensitivitätsanalyse.

    Größter Balken = wichtigster Parameter.
    Zeigt sofort wo du deine Research-Zeit investieren solltest.
    """
    base_price = sens_df["Basis ($)"].iloc[0]

    fig = go.Figure()

    # Low Case Balken (links) - abweichung vom base case
    fig.add_trace(go.Bar(
        name="Pessimistisch (-1σ)",
        y=sens_df["Parameter"],
        x=sens_df["Low ($)"] - base_price, # negative werte gehen nach links
        orientation="h", # - base da wir die abweichungen anschauen
        marker_color="#ef4444",
        opacity=0.85,
        text=[f"${v:.0f}" for v in sens_df["Low ($)"]],
        textposition="outside",
    ))

    # High Case Balken (rechts)
    fig.add_trace(go.Bar(
        name="Optimistisch (+1σ)",
        y=sens_df["Parameter"],
        x=sens_df["High ($)"] - base_price,
        orientation="h",
        marker_color="#16a34a",
        opacity=0.85,
        text=[f"${v:.0f}" for v in sens_df["High ($)"]],
        textposition="outside",
    ))

    # Baseline
    fig.add_vline(
        x=0,
        line_color="#1e293b",
        line_width=2
    )

    fig.update_layout(
        barmode="overlay",
        title=f"{ticker} — Tornado Chart (Sensitivitätsanalyse)",
        xaxis_title="Abweichung vom Base Case ($)",
        yaxis_title="",
        template="plotly_white",
        height=400,
        legend=dict(orientation="h", y=1.05),
        margin=dict(l=0, r=80, t=60, b=0)
    )

    fig.show()


def plot_scenario_waterfall(scenario_df: pd.DataFrame,
                             ticker:     str) -> None:
    """
    Scenario Vergleich als Waterfall/Bar Chart.
    """
    colors = ["#ef4444", "#3b82f6", "#16a34a"]
    labels = scenario_df["Szenario"].tolist()
    prices = scenario_df["Preis ($)"].tolist()

    fig = go.Figure()

    # Haupt-Bars
    fig.add_trace(go.Bar(
        x=labels,
        y=prices,
        marker_color=colors[:len(labels)],
        text=[f"${p:.0f}" for p in prices],
        textposition="inside",
        textfont=dict(size=14, color="white"),
        width=0.5,
        showlegend=False
    ))

    # Aktueller Kurs als Linie
    if "current_price" in scenario_df.columns or \
       "Upside (%)" in scenario_df.columns:
        pass   # wird im Summary gezeigt

    # Annotations
    if "Upside (%)" in scenario_df.columns:
        for i, row in scenario_df.iterrows(): # iteriert durch jede row 
            upside = row.get("Upside (%)", 0)
            sign   = "+" if upside >= 0 else ""
            color  = "#16a34a" if upside >= 0 else "#ef4444"
            fig.add_annotation(
                x=row["Szenario"],
                y=row["Preis ($)"] + max(prices) * 0.06, # annotation 3% der max balkenhöhe
                text=f"{sign}{upside:.1f}%",
                showarrow=False,
                font=dict(color=color, size=12)
            )

    fig.update_layout(
        title=f"{ticker} — Szenario Analyse (Bull / Base / Bear)",
        yaxis=dict(
            title="Intrinsic Value ($)",
            range=[0, max(prices) * 1.20]  # ← 20% Platz nach oben
        ),
        template="plotly_white",
        height=420,
        margin=dict(l=0, r=0, t=60, b=0)
    )

    fig.show()


def plot_dcf_bridge(base_result: dict,
                    ticker:      str, current_price: float=None, shares: float=None) -> None:
    """
    Waterfall Chart: Wie kommt der Enterprise Value zustande?

    Zeigt Schritt für Schritt:
        Phase 1 Cash Flows → Phase 2 Cash Flows → Terminal Value
        → Enterprise Value → minus Net Debt → Equity Value
    """
    cfs     = base_result["cash_flows"]
    pv_cfs  = base_result["pv_cash_flows"]
    pv_tv   = base_result["pv_terminal"]
    ev      = base_result["enterprise_value"]

    # Phase Summen
    phase1_pv = sum(
        cf["pv"] for cf in cfs if cf["phase"] == "Phase 1"
    )
    phase2_pv = sum(
        cf["pv"] for cf in cfs if cf["phase"] == "Phase 2"
    )

    labels = [
        "Phase 1 PV",
        "Phase 2 PV",
        "Terminal Value PV",
        "Enterprise Value"
    ]
    values = [phase1_pv, phase2_pv, pv_tv, ev]
    colors = ["#3b82f6", "#8b5cf6", "#f59e0b", "#16a34a"]

    # Market Cap Säule nur wenn beide Werte vorhanden
    if current_price is not None and shares is not None:
        market_cap = current_price * shares   # Aktienkurs × Anzahl Aktien = Market Cap in B$
        labels.insert(0, "Market Cap (heute)")
        values.insert(0, market_cap)
        colors.insert(0, "#94a3b8")           # grau → Referenzwert, nicht DCF

    fig = go.Figure(go.Bar(
        x=labels,
        y=values,
        marker_color=colors,
        text=[f"${v:.1f}B" for v in values],
        textposition="outside",
        width=0.55
    ))

    fig.update_layout(
        title=f"{ticker} — DCF Wert-Brücke",
        yaxis_title="Wert ($B)",
        template="plotly_white",
        height=400,
        margin=dict(l=0, r=0, t=60, b=0)
    )

    fig.show()

"""
$B
1500│                                    ████ Enterprise Value
    │                           ████
1000│                  ████
    │         ████
 500│  ████
    └──────────────────────────────────
       Phase1  Phase2  Terminal  Total
        PV      PV      Value     EV

→ Du siehst sofort: Terminal Value dominiert (typisch 60-80%)
"""

def print_dcf_report(stats:       dict,
                     scenario_df: pd.DataFrame,
                     sens_df:     pd.DataFrame,
                     ticker:      str) -> None:
    """
    Vollständiger DCF Report im Terminal.
    Das ist was ein Analyst seinem Manager schickt.
    """
    print(f"\n{'='*58}")
    print(f"  DCF REPORT — {ticker}")
    print(f"{'='*58}")

    print(f"\n  MONTE CARLO ERGEBNISSE ({stats['n_sims']:,} Sims)")
    print(f"  {'Median Intrinsic Value:':<28} "
          f"${stats['median']:>8.2f}")
    print(f"  {'Mittelwert:':<28} "
          f"${stats['mean']:>8.2f}")
    print(f"  {'Standardabweichung:':<28} "
          f"${stats['std']:>8.2f}")
    print(f"  {'Terminal Value Anteil:':<28} "
          f"{stats['tv_pct_mean']:>7.1f}%")

    print(f"\n  KONFIDENZINTERVALLE")
    for ci_name, (lo, hi) in [
        ("80% CI", stats["ci_80"]),
        ("90% CI", stats["ci_90"]),
        ("95% CI", stats["ci_95"]),
    ]:
        print(f"  {ci_name:<10} ${lo:>7.0f} — ${hi:<7.0f}")

    print(f"\n  PERZENTILE")
    for pct, key in [(10,"p10"),(25,"p25"),(75,"p75"),(90,"p90")]:
        print(f"  P{pct:<4}      ${stats[key]:>8.2f}")

    print(f"\n  MARGIN OF SAFETY (25%)")
    print(f"  {'Kauflimit (MOS):':<28} ${stats['mos_price']:>8.2f}")

    if "current_price" in stats:
        cp     = stats["current_price"]
        upside = stats["upside_pct"]
        prob   = stats["prob_undervalued"]
        buy    = "✅ KAUFEN" if stats["buy_signal"] \
                 else "⚠ ABWARTEN"

        print(f"\n  BEWERTUNG")
        print(f"  {'Aktueller Kurs:':<28} ${cp:>8.2f}")
        print(f"  {'Upside/Downside:':<28} {upside:>+8.1f}%")
        print(f"  {'P(unterbewertet):':<28} {prob:>7.1f}%")
        print(f"  {'Signal:':<28} {buy:>10}")

    print(f"\n  SZENARIO ANALYSE")
    print(f"  {'Szenario':<12} {'Preis':>8}", end="")
    if "Upside (%)" in scenario_df.columns:
        print(f"  {'Upside':>8}", end="")
    print()
    print("  " + "-"*30)
    for _, row in scenario_df.iterrows():
        print(f"  {row['Szenario']:<12} "
              f"${row['Preis ($)']:>7.2f}", end="")
        if "Upside (%)" in row:
            sign = "+" if row["Upside (%)"] >= 0 else ""
            print(f"  {sign}{row['Upside (%)']:>6.1f}%", end="")
        print()

    print(f"\n  SENSITIVITÄT (Top Parameter)")
    print(f"  {'Parameter':<18} {'Range ($)':>10} "
          f"{'Low Δ%':>8} {'High Δ%':>9}")
    print("  " + "-"*46)
    for _, row in sens_df.head(5).iterrows():
        print(f"  {row['Parameter']:<18} "
              f"  ${row['Range ($)']:>7.0f}"
              f"  {row['Low Δ%']:>7.1f}%"
              f"  {row['High Δ%']:>8.1f}%")

    print(f"\n{'='*58}")


if __name__ == "__main__":

    print("Tag 22 — Monte Carlo DCF")
    print("=" * 55)

    # --- Ticker auswählen ---
    TICKER = "META"

    # --- Echte Parameter laden ---
    params = get_real_params(TICKER)

    # Aktuellen Kurs holen
    try:
        current_price = float(
            yf.Ticker(TICKER)
            .history(period="1d")["Close"]
            .iloc[-1]
        )
        print(f"\nAktueller Kurs: ${current_price:.2f}")
    except Exception:
        current_price = None
        print("Kurs nicht verfügbar")

    # --- Base Case DCF ---
    print("\n1. Base Case DCF")
    base_result = single_dcf(
        params,
        fcf      = params.fcf_mean,
        growth_1 = params.growth_1_mean,
        growth_2 = params.growth_2_mean,
        terminal = params.terminal_mean,
        wacc     = params.wacc_mean,
    )
    print(f"   PV Cash Flows:   ${base_result['pv_cash_flows']:.2f}B")
    print(f"   PV Terminal:     ${base_result['pv_terminal']:.2f}B")
    print(f"   Enterprise Value:${base_result['enterprise_value']:.2f}B")
    print(f"   Intrinsic Price: ${base_result['intrinsic_price']:.2f}")
    print(f"   Terminal Value %:{base_result['tv_pct']:.1f}%")

    plot_dcf_bridge(base_result, TICKER, current_price = current_price, shares=params.shares)

    # --- Szenario Analyse ---
    print("\n2. Szenario Analyse")
    scenario_df = scenario_analysis(params, current_price)
    print(scenario_df[[
        "Szenario", "Growth 1 (%)", "WACC (%)",
        "Preis ($)"
    ] + (["Upside (%)"] if current_price else [])
    ].to_string(index=False))
    plot_scenario_waterfall(scenario_df, TICKER)

    # --- Sensitivitätsanalyse ---
    print("\n3. Sensitivitätsanalyse")
    sens_df = sensitivity_analysis(params, current_price)
    print(sens_df[[
        "Parameter", "Range ($)", "Low Δ%", "High Δ%"
    ]].to_string(index=False))
    plot_tornado_chart(sens_df, TICKER)

    # --- Monte Carlo Simulation ---
    print("\n4. Monte Carlo Simulation (10.000 Runs)...")
    sim_df = monte_carlo_dcf(
        params,
        n_sims        = 10_000,
        current_price = current_price
    )

    # Statistiken
    dcf_stats = dcf_statistics(sim_df, current_price)

    # Report
    print_dcf_report(dcf_stats, scenario_df, sens_df, TICKER)

    # Charts
    plot_monte_carlo_distribution(sim_df, dcf_stats, TICKER)

    # --- Vergleich mehrerer Aktien ---
    print("\n5. Multi-Ticker DCF Vergleich")
    print("-" * 40)

    comparison_tickers = ["AAPL", "MSFT", "GOOGL", "AMZN"]
    comp_rows = []

    for t in comparison_tickers:
        try:
            p = get_real_params(t)

            cp = float(
                yf.Ticker(t)
                .history(period="1d")["Close"]
                .iloc[-1]
            )

            sim = monte_carlo_dcf(p, n_sims=2_000)
            st  = dcf_statistics(sim, cp)

            comp_rows.append({
                "Ticker":         t,
                "Kurs ($)":       cp,
                "Median IV ($)":  st["median"],
                "Upside (%)":     st.get("upside_pct", 0),
                "P(Unterbewertet)": st.get("prob_undervalued", 0),
                "80% CI Low":     st["ci_80"][0],
                "80% CI High":    st["ci_80"][1],
                "MOS Limit ($)":  st["mos_price"],
                "Signal":         "✅" if st.get("buy_signal")
                                   else "⚠",
            })
        except Exception as e:
            print(f"  {t}: Fehler — {e}")

    if comp_rows:
        comp_df = pd.DataFrame(comp_rows)
        print("\nDCF Vergleich:")
        print(comp_df[[
            "Ticker", "Kurs ($)", "Median IV ($)",
            "Upside (%)", "P(Unterbewertet)", "Signal"
        ]].to_string(index=False))

        comp_df.to_csv("day22_dcf_comparison.csv", index=False)
        print("\nGespeichert: day22_dcf_comparison.csv")

    # Export der Simulation
    sim_df.to_csv("day22_monte_carlo_sim.csv", index=False)
    print("Gespeichert: day22_monte_carlo_sim.csv")