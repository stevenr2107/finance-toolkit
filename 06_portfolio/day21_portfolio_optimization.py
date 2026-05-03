"""
# nicht die einzelne Aktie zählt sondern wie sie sich im zusammenspiel mit anderen verhält 
"""

"""
Jedes Portfolio hat zwei Kennzahlen:

    Erwartete Rendite:  μ_p = Σ wᵢ · μᵢ
    Portfoliorisiko:    σ_p = √(wᵀ · Σ · w)

    wᵢ  = Gewicht von Asset i im Portfolio (Summe = 1)
    μᵢ  = Erwartete Rendite von Asset i
    Σ   = Kovarianzmatrix aller Assets

    Die entscheidende Erkenntnis:
    Wenn zwei Assets NICHT perfekt korreliert sind (ρ < 1),
    ist das Portfoliorisiko KLEINER als der gewichtete
    Durchschnitt der Einzelrisiken.

    → Diversifikation reduziert Risiko ohne Renditeverlust.
    → Das nennt sich "Free Lunch" der Finanzwelt.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

WAS DIESES SCRIPT MACHT
━━━━━━━━━━━━━━━━━━━━━━━

1. KOVARIANZMATRIX
   Misst wie sich Assets gemeinsam bewegen.
   ρ = +1 → perfekt gleichläufig (keine Diversifikation)
   ρ =  0 → unabhängig (gute Diversifikation)
   ρ = -1 → perfekt gegenläufig (maximale Diversifikation)

2. MONTE CARLO SIMULATION
   Generiert tausende zufällige Portfolio-Gewichtungen.
   Jedes Portfolio bekommt Rendite + Risiko berechnet.
   Ergibt die charakteristische "Bullet Shape" der Frontier.

3. EFFICIENT FRONTIER
   Die obere Kante der Monte-Carlo-Wolke.
   Mathematisch optimiert via scipy.optimize.minimize.
   Jeder Punkt = Minimales Risiko für eine gegebene Rendite.

4. MAX SHARPE PORTFOLIO
   Sharpe Ratio = (Rendite - Risikofreier Zins) / Risiko
   Das Portfolio mit dem besten Risk/Return-Verhältnis.
   Tangentialpunkt zwischen Efficient Frontier und
   Capital Market Line (CML).

5. MIN VARIANCE PORTFOLIO
   Das Portfolio mit dem absolut geringsten Risiko.
   Linkster Punkt der Efficient Frontier.
   Relevant für sehr risikoaverse Anleger.

6. BLACK-LITTERMAN MODELL
   Erweiterung von Markowitz: eigene Marktmeinungen
   einbauen (z.B. "Ich glaube AAPL steigt um 10%").
   Kombiniert Markt-Gleichgewicht mit subjektiven Views.
   Wird von Goldman Sachs und großen Hedge Funds genutzt.

"""

import yfinance as yf
import pandas as pd 
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.optimize import minimize
from scipy.stats import norm
import warnings
warnings.filterwarnings("ignore")

def load_data(tickers: list,
              period:  str = "5y") -> pd.DataFrame:
    """Lädt Close-Preise für alle Ticker."""
    df = yf.download(
        tickers, period=period,
        auto_adjust=True, progress=False
    )["Close"]
    if len(tickers) == 1: # wenn es nur einen ticker gibt, werden keine spalten ausgegeben 
        df = df.to_frame(name=tickers[0])
    df.columns = df.columns.get_level_values(0)
    return df.dropna()

def compute_returns(prices: pd.DataFrame):
    """Tägliche prozentuale Returns"""
    return prices.pct_change().dropna()

"""
Kovarianzmatrix & Grundkennzahlen 
"""

def portfolio_stats(weights:      np.ndarray,
                    mean_returns: pd.Series, # tägliche durchschnittsreturns
                    cov_matrix:   pd.DataFrame,
                    risk_free:    float = 0.05) -> dict:
    
    """
    Berechnet Return, Volatilität und Sharpe für ein Portfolio.

    Formel:
        Portfolio Return     = Σ(wi × ri) × 252 weight * return * 252
        Portfolio Variance   = w^T × Σ × w × 252
        Portfolio Volatility = √Variance
        Sharpe Ratio         = (Return - rf) / Volatility

    Kovarianzmatrix Σ:
        Diagonal: Varianz jedes Assets
        Off-Diagonal: Kovarianz zwischen Assets
        Niedrige Kovarianz = gute Diversifikation

    weights: Array der Gewichtungen (Summe = 1)
    """

    weights = np.array(weights)

    # Annualisierter Portfolio-Return
    port_return = np.dot(weights, mean_returns) * 252 # skalarprodukt 
    # wir haben zwei vektoren und die werden elementweise multipliziert
    # und das ergebnis wird addiert weil weights und mean returns listen sind -> %

    """
    Bei der annualisierten portfolio volatilität kann man nicht einfach den durchschnitt
    der varianz nehmen, sondern muss schauen wie die assets korrelieren, da sie ggf. weniger 
    volatil gemeinsam sind 
    """
    # Annualisierte Portfolio-Volatilität
    port_variance = np.dot(#2. vektor weights * vektor -> Gesamtvarianz Portfolio
        weights.T,
        np.dot(cov_matrix * 252, weights) # 1. matrix * vektor -> Vektor
        # wieviel risiko trägt ein asset bei?
    )
    port_vol = np.sqrt(port_variance)

    # Sharpe Ratio -> Wieviel Rendite pro risikoeinheit 
    sharpe = (port_return - risk_free) / port_vol \
             if port_vol > 0 else 0

    return {
        "return":   port_return,
        "vol":      port_vol,
        "sharpe":   sharpe,
    }

def compute_covariance_matrix(returns:       pd.DataFrame,
                               method:        str = "sample") -> pd.DataFrame:
    """
    Kovarianzmatrix — Herzstück der Portfolio-Optimierung.

    Methoden:
        sample:     Standard OLS Schätzung
                    Gut bei vielen Datenpunkten
                    Instabil bei wenig Daten

        ledoit_wolf: Shrinkage Estimator
                    Robuster bei kleinen Stichproben
                    Reduziert Extremwerte (shrinks toward identity)
                    Das nutzen Profis in der Praxis

    Warum nicht immer sample?
        Mit 10 Assets und 252 Datenpunkten ist die
        sample Kovarianzmatrix oft schlecht konditioniert.
        Kleine Fehler in der Matrix → große Fehler in Gewichten.
        Ledoit-Wolf stabilisiert das.
        
        Kovarianzmatrix
        AAPL    MSFT    GOOG
AAPL  [ 0.040   0.012   0.008 ]   ← Diagonale: Varianz von AAPL
MSFT  [ 0.012   0.090   0.015 ]   ← Off-Diagonal: Kovarianz AAPL/MSFT
GOOG  [ 0.008   0.015   0.060 ]
    """
    if method == "sample": # instabil bei wenig daten 
        return returns.cov() # berechnet die klassische ols-Kovarianzmatrix

    elif method == "ledoit_wolf":
        # Ledoit-Wolf Shrinkage
        n, p      = returns.shape # n zeitpunkte, p assets
        sample    = returns.cov() # kovarianz
        mu        = sample.values.trace() / p # Durchschnittliche Varianz 

        # Shrinkage Intensität (vereinfacht)
        # Shrinkage zieht ausreißer stärker raus 
        delta     = 0.1 # shrinkage intensität (10%)
        shrunk    = (1 - delta) * sample.values + \
                     delta * mu * np.eye(p)
        """
        Ohne Shrinkage:  Kovarianz(AAPL, MSFT) = 0.85  ← evtl. Ausreißer
        Mit Shrinkage:   Kovarianz(AAPL, MSFT) = 0.77  ← stabiler
        """

        return pd.DataFrame(
            shrunk,
            index=sample.index,
            columns=sample.columns
        )

    return returns.cov()

def monte_carlo_portfolios(returns:       pd.DataFrame,
                            n_portfolios:  int   = 5_000,
                            risk_free:     float = 0.05) -> pd.DataFrame:
    """
    Generiert zufällige Portfolios via Monte Carlo.

    Warum Monte Carlo?
        Wir können nicht analytisch alle möglichen
        Gewichtungs-Kombinationen berechnen.
        Mit N Assets gibt es unendlich viele Möglichkeiten.

        Stattdessen: 5000 zufällige Portfolios generieren,
        jeden bewerten, visualisieren.
        Das ergibt die Form der Efficient Frontier.

    Dirichlet-Verteilung:
        np.random.dirichlet gibt Gewichte die
        automatisch auf 1 summieren und alle > 0 sind.
        Besser als random.random() dann normalisieren.
    """
    n_assets     = len(returns.columns) # anzahl aktien 
    mean_returns = returns.mean() # durchschnittl. Rendite 
    cov_matrix   = compute_covariance_matrix(returns, "ledoit_wolf")

    results = [] # durchschnittl. tägliche rendite der assets

    for _ in range(n_portfolios): # _ = dummy variable - man kann einfach ignorieren
        weights = np.random.dirichlet(np.ones(n_assets)) # verteilung nicht gleichmäßig
        # weights - gewichte der assets
        # alle punkte wo gewichte > 0 und summe = 1

        stats = portfolio_stats(
            weights, mean_returns, cov_matrix, risk_free
        )

        row = {
            "return":  stats["return"] * 100, # auf %
            "vol":     stats["vol"]    * 100, # auf %
            "sharpe":  stats["sharpe"], # Dezimal
        }

        # Gewichte hinzufügen
        for ticker, w in zip(returns.columns, weights):# zip verbindet 2 listen 
            row[f"w_{ticker}"] = round(w, 4) # Gewichte speichern 
        # zip → [("AAPL", 0.31), ("MSFT", 0.22), ("GOOG", 0.47)]

        results.append(row) # row dictionary erweitern 
        """
        row["w_AAPL"] = 0.31
        row["w_MSFT"] = 0.22
        row["w_GOOG"] = 0.47
        """

    return pd.DataFrame(results)

"""
Rendite
  │                    ·  · ·
  │               ·  ·· ·· ·          ← Efficient Frontier (obere Kante)
  │           · ·· ·· · ·
  │         ·· · · ·
  │        · ·
  └─────────────────────── Volatilität

    Jeder Punkt = ein zufälliges Portfolio
    Obere Kante = Efficient Frontier
"""


# Optimierung 
# monte carlo rät aus 5000 portfolios das beste, aber jetzt suchen wir es mathematisch 

def optimize_portfolio(returns: pd.DataFrame,
                       objective: str = "sharpe",
                       risk_free: float = 0.05,
                       constraints: dict = None ) -> dict:
    """
    Analytische Portfolio-Optimierung via scipy.

    Objectives:
        sharpe:   Max Sharpe Ratio → bestes Risk/Return
        min_vol:  Min Volatilität  → defensivstes Portfolio
        max_ret:  Max Return       → aggressivstes (= 100% beste Aktie)

    Constraints:
        weights_sum = 1       (vollständig investiert)
        weights >= 0          (Long-only, kein Short)
        Optional: max_weight  (max. X% pro Position)

    Algorithmus: SLSQP (Sequential Least Squares Programming)
        Schnell, stabil, gut für Constraints.
    """

    n_assets     = len(returns.columns)
    mean_returns = returns.mean()
    cov_matrix   = compute_covariance_matrix(returns, "ledoit_wolf")

    # Default Constraints 
    max_weight = constraints.get("max_weight", 1.0) \
                 if constraints else 1.0
    min_weight = constraints.get("min_weight", 0.0) \
                 if constraints else 0.0

    bounds = tuple(
        (min_weight, max_weight) for _ in range(n_assets)
    )
    # → ((0.0, 1.0), (0.0, 1.0), (0.0, 1.0), ...)
    # Jede Aktie: zwischen 0% und 100% Gewicht

    constraint_sum = {"type": "eq",
                      "fun": lambda w: np.sum(w) - 1}
    # "eq" = equality → Ergebnis muss = 0 sein
    # np.sum(w) - 1 = 0 → Gewichte summieren auf 1

    # Startgewichte: gleichgewichtet
    w0 = np.ones(n_assets) / n_assets # returned an array of ones divided by n_assets
    # Startpunkt: gleichgewichtet → [0.2, 0.2, 0.2, 0.2, 0.2]
    # Der Optimierer braucht einen Startpunkt

    # Objective Functions
    def neg_sharpe(w):
        s = portfolio_stats(w, mean_returns, cov_matrix, risk_free)
        return -s["sharpe"] # scipy minimiert immer daher drehen wir vorzeichen um -> später wird minimiert 

    def portfolio_vol(w):
        s = portfolio_stats(w, mean_returns, cov_matrix, risk_free)
        return s["vol"]

    def neg_return(w):
        s = portfolio_stats(w, mean_returns, cov_matrix, risk_free)
        return -s["return"]

    obj_map = { # sauberer als if else
        "sharpe":  neg_sharpe,
        "min_vol": portfolio_vol,
        "max_ret": neg_return,
    }

    result = minimize(
        obj_map.get(objective, neg_sharpe), # was minimieren?
        w0,                                 # startpunkt
        method      = "SLSQP",              # Algorithmus
        bounds      = bounds,               # Gewichte zwischen 0% und 100%
        constraints = [constraint_sum],     # Gewichte summieren auf 1
        options     = {"maxiter": 1000, "ftol": 1e-9} # maximale Anzahl an Iterationen
        # ftol=1e-9 → Konvergenzkriterium: stoppt wenn die Verbesserung kleiner als 0.000000001 ist.
    )

    """
    Start:    [0.2,  0.2,  0.2,  0.2,  0.2]   Sharpe = 0.65
    Iter 1:   [0.25, 0.18, 0.22, 0.19, 0.16]  Sharpe = 0.71
    Iter 2:   [0.31, 0.15, 0.24, 0.18, 0.12]  Sharpe = 0.78
    ...
    Iter 47:  [0.38, 0.12, 0.28, 0.14, 0.08]  Sharpe = 0.89 ✅ maximum
    """
    optimal_weights = result.x
    stats = portfolio_stats(
        optimal_weights, mean_returns, cov_matrix, risk_free
    )

    return {
        "weights":    pd.Series(
            optimal_weights,
            index=returns.columns
        ).round(4),
        "return":     round(stats["return"] * 100, 2),
        "vol":        round(stats["vol"]    * 100, 2),
        "sharpe":     round(stats["sharpe"],       4),
        "objective":  objective,
        "success":    result.success, # wichtig, da optimierer nicht immer konvergiert
        # konvergiert bedeutet: maximale Anzahl an Iterationen erreicht
    }

# Frontier curve gibt für jeden return level das portfolio mit dem kleinsten risiko an
# Punkt für Punkt 
def efficient_frontier_curve(returns: pd.DataFrame,
                             n_points: int = 50,
                             risk_free: float = 0.05) -> pd.DataFrame:
    """
    Berechnet die Efficient Frontier analytisch.

    Methode:
        Für jedes Target-Return-Level das Portfolio
        mit minimaler Volatilität finden.
        Das ergibt Punkte auf der Frontier.

    Die Frontier zeigt:
        Welches Risiko ist für jedes Return-Level minimal?
        Alles rechts davon ist ineffizient (mehr Risiko, gleicher Return).
        Alles links ist unerreichbar.
    """
    n_assets     = len(returns.columns)
    mean_returns = returns.mean()
    cov_matrix   = compute_covariance_matrix(returns, "ledoit_wolf")

    # Return-Range: von Min-Vol bis Max-Return
    min_vol_result = optimize_portfolio(returns, "min_vol", risk_free)
    max_sharpe_result = optimize_portfolio(returns, "sharpe", risk_free)

    target_returns = np.linspace( # 50 gleichmäßige Punkte zwischen Min-Vol und Max-Return
        min_vol_result["return"] / 100,
        max_sharpe_result["return"] / 100,
        n_points # 50
    )
    """
    Min Vol Return:  8%
    Max Return:     22%
    linspace → [0.08, 0.11, 0.14, 0.17, 0.20, 0.22]  ← 50 Punkte    
    """

    frontier = []

    for target in target_returns:
        constraints = [
            {"type": "eq", "fun": lambda w: np.sum(w) - 1},
            {"type": "eq", "fun": lambda w, t=target: np.dot(w, mean_returns) * 252 - t} # neuer constraint 
        ] # finde das prtfolio mit genau target reurn aber mit minimalen risiko 
        # t = target -> target return wird eingefroren , sonst benutzen alle den gleichen wert bei jeder schleife 

        bounds = tuple((0,1) for _ in range(n_assets)) # alle werte zwischen 0 und 1
        w0 = np.ones(n_assets) / n_assets # alle werte gleichmaessig verteilt

        result = minimize(
            lambda w: portfolio_stats(
                w, mean_returns, cov_matrix, risk_free
            )["vol"],
            w0,
            method      = "SLSQP",
            bounds      = bounds,
            constraints = constraints,
            options     = {"maxiter": 500}
        )
# if result.success — nur erfolgreiche Optimierungen kommen in die Frontier. 
# Fehlgeschlagene werden still übersprungen.
        if result.success:
            stats = portfolio_stats(
                result.x, mean_returns, cov_matrix, risk_free
            )
            frontier.append({
                "return": stats["return"] * 100,
                "vol":    stats["vol"]    * 100,
                "sharpe": stats["sharpe"],
            })
        
    return pd.DataFrame(frontier)

"""
Wie beide funktionen gemeinsam funktionieren 

optimize_portfolio("min_vol")  →  Linkster Punkt der Frontier
optimize_portfolio("max_ret")  →  Oberster Punkt der Frontier
         ↓
np.linspace(min, max, 50)      →  50 Return-Ziele dazwischen
         ↓
für jeden Target-Return:
    minimize(vol, constraint: return=target)
         ↓
50 Punkte auf der Frontier

         Rendite
           │                    ★ Max Sharpe
           │               ·  ·/· ·
           │           · ··/ · ·
           │         ·· ·/·
           │        · ·/
           │        ◆ ← Min Variance
           └──────────────────── Volatilität

Monte Carlo (·) = grobe Form - zeigt die wolke
Frontier (Linie) = mathematisch exakt
"""
def black_litterman(returns: pd.DataFrame,
                    views: dict,
                    tau: float = 0.05,
                    risk_free: float = 0.05) -> dict:
    
    """
    Das Problem das durch BL gelöst wird: Reines Markowitz hat einen praktischen fehler
Historische Renditen:
AAPL: +45% letztes Jahr
MSFT: +12% letztes Jahr

Markowitz sagt: "Steck fast alles in AAPL"
→ Ergebnis: 87% AAPL, 8% MSFT, 5% Rest


Statt historische Returns blind zu nutzen:
1. Starte mit Markt-Gleichgewicht als "neutraler Prior" ( market cap)
2. Mische deine eigenen Meinungen (Views) dazu (konfidenz)
3. Je höher deine Konfidenz → desto mehr dominiert deine View

→ Vernünftige, diversifizierte Gewichte als Ergebnis

    Konfidenz:
        Hoch → deine View dominiert
        Niedrig → Marktgleichgewicht dominiert
    """

    n_assets     = len(returns.columns)
    tickers      = returns.columns.tolist()
    mean_returns = returns.mean() * 252
    cov_matrix   = returns.cov() * 252

    # Market Equilibrium (gleichgewichtete Marktannahme)
    market_weights = np.ones(n_assets) / n_assets # gleichgewichtet
    risk_aversion  = 2.5 # standard risikoaversion 

    # implied equilibrium Returns 
    pi = risk_aversion * cov_matrix.values @ market_weights  # @ ist wie np.dot 
    # @ ist matrixmultiplikation 
    """
    Interpretation:
    "Wenn niemand besondere Meinungen hat, welche Rendite
    erwartet der Markt für jede Aktie?"

    AAPL: π = 0.082  → 8.2% Markterwartung
    MSFT: π = 0.091  → 9.1% Markterwartung
    """
    valid_views = {t: v for t, v in views.items() if t in tickers}

    if not valid_views:
        return {
            "weights": pd.Series(market_weights, index=tickers).round(4),
            "returns": pd.Series(pi, index=tickers).round(4),
        }

    # Views Matrix aufbauen
    P = np.zeros((len(valid_views), n_assets)) # 2 x 5 Matrix ( 2 Views, 5 Assets ) - Welche Aktie betrifft welche View 
    Q = np.zeros(len(valid_views))            # 2 x 1 Vektor ( die view- werte) - Renditeerwartung 

    for i, (ticker, view_return) in enumerate(valid_views.items()):
            idx = tickers.index(ticker) # Position der aktie
            P[i, idx] = 1               # 1 = diese aktie betrifft view i
            Q[i] = view_return          # meine erwartete rendite 

            # i=0, ticker="AAPL", view_return=0.15
            # i=1, ticker="MSFT", view_return=0.08

    if P.shape[0] == 0:
        # Keine Views -> Market Portfolio zurückgeben 
        return{
            "weights": pd.Series(
                market_weights, index = tickers
            ).round(4),
            "returns": pd.Series(pi, index=tickers).round(4),
            #P = [[1, 0, 0, 0, 0],   ← View 1 betrifft nur AAPL
            # [0, 1, 0, 0, 0]]       ← View 2 betrifft nur MSFT

            # Q = [0.15,             ← AAPL soll +15% machen
            # 0.08]                  ← MSFT soll +8% machen
        }
    # Unsicherheit der Views (omega) - wie sicher sind die views?
    omega = tau * P @ cov_matrix.values @ P.T
    """
    omega = [[0.002,  0   ],   ← Unsicherheit von View 1 (AAPL)
            [0,      0.003]]  ← Unsicherheit von View 2 (MSFT)
    """
    # tau = 0.05 → kleiner Wert → du bist ziemlich unsicher → Marktgleichgewicht dominiert
    # tau = 0.5  → großer Wert → du bist sehr sicher → deine View dominiert

    # Posterior Returns 
    tau_cov = tau * cov_matrix.values 
    A       = np.linalg.inv( # matrix invertieren mit mathematischer formel (bayes'sche statistik)
        np.linalg.inv(tau_cov) +
        P.T @ np.linalg.inv(omega) @ P
    )
    bl_returns = A @ (
        np.linalg.inv(tau_cov) @ pi +
        P.T @ np.linalg.inv(omega) @ Q
    )
    # gewichteter durchschnitt aus 
    # 1. markt prior π     (gewichtet mit Markt-Konfidenz)
    # 2. Deine Views Q     (gewichtet mit deiner Konfidenz)
    # Hohes tau → deine Views dominieren mehr
    # Niedriges tau → Markt-Prior dominiert mehr

    # Markt sagt AAPL: 8.2% gewichtung
    # Du sagst AAPL:  15.0% gewichtung
    # tau = 0.05 (wenig Konfidenz)

    # BL Ergebnis: ~9.1%  ← leicht über Markt, aber nicht 15%

    bl_returns_series = pd.Series(bl_returns, index=tickers)
    # Optimale Gewichte aus BL Returns
    bl_cov = cov_matrix + A # A = parameterschätzunsicherheit 

    # Optimierung mit BL Returns 
    def neg_sharpe_bl(w): # identisch zur normalen sharpe optimierung - aber jetzt mit bl ajustierten returns - weniger extreme gewichte 
        ret = np.dot(w, bl_returns) # BL Returns statt historische 
        vol = np.sqrt(w.T @ bl_cov.values @ w) # BL Kovarianz statt normale 
        return -(ret - risk_free) / vol if vol > 0 else 0
    
    constraints = [{"type": "eq",
                    "fun": lambda w: np.sum(w) - 1}] # summe der gewichte = 1
    bounds      = tuple((0, 1) for _ in range(n_assets))
    w0          = np.ones(n_assets) / n_assets

    result = minimize(
        neg_sharpe_bl, w0,
        method      = "SLSQP", # sequential least squares programming
        bounds      = bounds,
        constraints = constraints,
    )

    return {
        "weights":     pd.Series(
            result.x, index=tickers
        ).round(4),
        "bl_returns":  bl_returns_series.round(4),
        "prior_pi":    pd.Series(pi, index=tickers).round(4),
    }

"""
REINES MARKOWITZ:
Historische Returns → Optimierung → extreme Gewichte
[87%, 8%, 2%, 2%, 1%]  ← praktisch unbrauchbar

BLACK-LITTERMAN:
Markt-Prior π
     +                  → Posterior Returns → Optimierung → vernünftige Gewichte
Deine Views Q
[34%, 28%, 18%, 12%, 8%]  ← diversifiziert ✅


Der Output:
{
  "weights":    [0.34, 0.28, 0.18, 0.12, 0.08]   ← optimale Gewichte
  "bl_returns": [0.091, 0.103, ...]              ← adjustierte Renditeerwartungen
  "prior_pi":   [0.082, 0.091, ...]              ← was der Markt ohne Views impliziert
}
"""

# Jetzt wird die Volatilität noch mit reingerechnet -> hohe vola wird niedriger gewichtet
def risk_contribution(weights:    np.ndarray,
                      cov_matrix: pd.DataFrame,
                      tickers:    list) -> pd.DataFrame:
    """
    Risk Parity Analyse — wie viel Risiko trägt jede Position?

    Marginal Risk Contribution (MRC):
        Wie viel mehr Portfoliorisiko entsteht wenn
        ich eine Position um 1 Einheit erhöhe?

    Total Risk Contribution (TRC):
        Gewicht × MRC = tatsächlicher Risikobeitrag

    Risk Parity Portfolio:
        Jede Position trägt gleichviel zum Gesamtrisiko bei.
        Nicht gleichgewichtet in Dollar — gleichgewichtet in Risiko.
        Das verhindert dass eine volatile Aktie das
        Portfolio dominiert.

    Bridgewater (Ray Dalio) nutzt Risk Parity für
        den legendären All Weather Fund.
    """
    weights    = np.array(weights) # array erstellen
    cov        = cov_matrix.values # von dataframe zu array -> extrahiert die zahlen aus dem dataframe 
    port_vol   = np.sqrt(weights.T @ cov @ weights) # portfoliovolatilität ->wurzel aus dem skalarprodukt

    # Marginal Risk Contribution
    mrc = (cov @ weights) / port_vol # ableitung der portfoliovolatilität
    # wenn ich aapl um 1 einheit erhöhre wie stark erhöht sich das Gesamtrisiko

    # Total Risk Contribution
    trc = weights * mrc # die zusätzliche einheit an risk mit der erhöhung mal des gewichts -> tatsächlicher risk anstieg 

    # Prozentualer Anteil
    trc_pct = trc / trc.sum() * 100 # wieviel % des gesamtrisikos trägt jede aktie 

    return pd.DataFrame({
        "Ticker":     tickers,
        "Gewicht (%)": (weights * 100).round(2),
        "MRC":        mrc.round(6), # grenzrisiko pro einheit 
        "TRC ($)":    trc.round(6), # absoluter risikobeitrag
        "Risiko (%)": trc_pct.round(2),
    })


def risk_parity_weights(returns:    pd.DataFrame,
                         risk_free:  float = 0.05) -> dict:
    """
    Berechnet Risk Parity Gewichte via Optimierung.

    Ziel: alle TRC gleich → jede Aktie trägt gleichviel Risiko. - alle total risk contributions gleich

    Objective: Σ(TRC_i - TRC_j)² minimieren
    """
    n_assets   = len(returns.columns)
    cov_matrix = returns.cov() * 252 # risk parity stabiler daher * 252 - annualsiert
    tickers    = returns.columns.tolist()

    def risk_parity_objective(weights):
        weights  = np.array(weights)
        port_vol = np.sqrt(weights.T @ cov_matrix.values @ weights)
        mrc      = (cov_matrix.values @ weights) / port_vol
        trc      = weights * mrc
        # Minimiere Differenz zwischen allen TRCs
        target   = port_vol / n_assets # Das ist der ideale TRC wenn alle assets gleichviel risiko tragen sollen 
        return np.sum((trc - target) ** 2) # OLS - summe quadrierter abweichungen 

        """
        port_vol = 18%
        n_assets = 4

        target = 18% / 4 = 4.5% pro Asset
        → jede Aktie soll 4.5% zum Gesamtrisiko beitragen

        return np.sum - 
        Iteration 1:  trc = [0.123, 0.099, 0.043, 0.033]  →  Summe² = 0.0089
        Iteration 2:  trc = [0.098, 0.085, 0.058, 0.057]  →  Summe² = 0.0041
        ...
        Iteration N:  trc = [0.045, 0.045, 0.045, 0.045]  →  Summe² ≈ 0.0000 ✅
        """

    constraints = [{"type": "eq",
                    "fun": lambda w: np.sum(w) - 1}]
    bounds      = tuple((0.01, 1) for _ in range(n_assets)) # mindestens 1% in jeder position - sonst trc 0
    w0          = np.ones(n_assets) / n_assets

    result = minimize(
        risk_parity_objective, w0,
        method      = "SLSQP",
        bounds      = bounds,
        constraints = constraints,
        options     = {"maxiter": 1000}
    )

    weights = result.x
    stats   = portfolio_stats(
        weights, returns.mean(), cov_matrix / 252, risk_free # portfolio_stats annualisiert selber nochmal daher /252 
    )
    rc      = risk_contribution(weights, cov_matrix, tickers)

    return {
        "weights":  pd.Series(weights, index=tickers).round(4),
        "return":   round(stats["return"] * 100, 2),
        "vol":      round(stats["vol"]    * 100, 2),
        "sharpe":   round(stats["sharpe"],       4),
        "risk_contrib": rc,
    }
"""

Markowitz Max Sharpe:  [38%, 29%, 18%, 15%]  → optimale Rendite/Risiko
Risk Parity:           [12%, 16%, 34%, 38%]  → gleichmäßiges Risiko

Markowitz optimiert für RENDITE pro Risikoeinheit
Risk Parity optimiert für STABILITÄT unter allen Marktbedingungen
"""

def plot_efficient_frontier(mc_df:       pd.DataFrame,
                             frontier_df: pd.DataFrame,
                             max_sharpe:  dict,
                             min_vol:     dict,
                             risk_parity: dict,
                             tickers:     list) -> None:
    """
    Das ikonischste Chart der modernen Portfoliotheorie.

    Scatter: Monte Carlo Portfolios eingefärbt nach Sharpe
    Linie:   Analytische Efficient Frontier
    Punkte:  Max Sharpe, Min Vol, Risk Parity
    """
    fig = go.Figure()

    # Monte Carlo Scatter — eingefärbt nach Sharpe
    fig.add_trace(go.Scatter(
        x=mc_df["vol"],
        y=mc_df["return"],
        mode="markers",
        name="Monte Carlo Portfolios",
        marker=dict(
            color=mc_df["sharpe"],
            colorscale=[
                [0.0,  "#dc2626"],
                [0.3,  "#f59e0b"],
                [0.6,  "#3b82f6"],
                [1.0,  "#16a34a"]
            ],
            size=4,
            opacity=0.5,
            showscale=True,
            colorbar=dict(
                title="Sharpe Ratio",
                thickness=15,
                len=0.6
            )
        ),
        hovertemplate=(
            "Return: %{y:.2f}%<br>"
            "Volatilität: %{x:.2f}%<br>"
            "<extra></extra>"
        )
    ))

    # Efficient Frontier Linie
    if not frontier_df.empty:
        fig.add_trace(go.Scatter(
            x=frontier_df["vol"],
            y=frontier_df["return"],
            mode="lines",
            name="Efficient Frontier",
            line=dict(color="#1e293b", width=3)
        ))

    # Capital Market Line (CML)
    """alles unter CML ist schlecht und lohnt sich nicht"""
    # mischt unser risikoreiches Portfolio mit Staatsanleihen, die bei 5% liegen und erstellt eine neue möglichkeit
    rf_rate   = 0.05 * 100
    ms_ret    = max_sharpe["return"]
    ms_vol    = max_sharpe["vol"]
    cml_x     = np.linspace(0, ms_vol * 1.5, 100) # 100 gleichmäßige punkte auf der x achse mit 0% risiko
    cml_y     = rf_rate + (ms_ret - rf_rate) / ms_vol * cml_x

    fig.add_trace(go.Scatter(
        x=cml_x,
        y=cml_y,
        mode="lines",
        name="Capital Market Line",
        line=dict(color="#8b5cf6", width=1.5, dash="dash")
    ))

    # Schlüssel-Portfolios
    key_portfolios = [
        (max_sharpe,  "⭐ Max Sharpe",    "#16a34a", "star",         14),
        (min_vol,     "🛡 Min Volatilität", "#2563eb", "diamond",      12),
        (risk_parity, "⚖ Risk Parity",    "#f59e0b", "circle",        10),
    ]

    for port, name, color, symbol, size in key_portfolios:
        fig.add_trace(go.Scatter(
            x=[port["vol"]],
            y=[port["return"]],
            mode="markers+text",
            name=name,
            text=[name.split(" ", 1)[1]],
            textposition="top right",
            marker=dict(
                color=color, size=size,
                symbol=symbol,
                line=dict(width=2, color="white")
            )
        ))

    # Risk-free Rate
    fig.add_hline(
        y=rf_rate,
        line_dash="dot",
        line_color="#94a3b8",
        opacity=0.6,
        annotation_text=f"Risk-free Rate ({rf_rate:.1f}%)"
    )

    fig.update_layout(
        title="Efficient Frontier — Markowitz Portfolio Optimierung",
        xaxis_title="Volatilität / Risiko (%)",
        yaxis_title="Erwarteter Return (%)",
        template="plotly_white",
        height=600,
        legend=dict(orientation="v", x=1.15, y=0.5),
        margin=dict(l=0, r=120, t=50, b=0)
    )

    fig.show()


def plot_portfolio_weights(portfolios: dict) -> None:
    """
    Vergleicht Gewichtungen verschiedener Portfolios als Stacked Bar.
    """
    colors = [
        "#2563eb", "#16a34a", "#f59e0b",
        "#ef4444", "#8b5cf6", "#0891b2",
        "#f97316", "#84cc16", "#ec4899",
        "#14b8a6"
    ]

    fig = go.Figure()

    tickers = list(
        list(portfolios.values())[0]["weights"].index
    )

    for i, ticker in enumerate(tickers):
        weights_per_portfolio = [
            round(port["weights"].get(ticker, 0) * 100, 2)
            for port in portfolios.values()
        ]

        fig.add_trace(go.Bar(
            name=ticker,
            x=list(portfolios.keys()),
            y=weights_per_portfolio,
            marker_color=colors[i % len(colors)],
            text=[f"{w:.1f}%" for w in weights_per_portfolio],
            textposition="inside",
            textfont=dict(size=10, color="white")
        ))

    fig.update_layout(
        barmode="stack",
        title="Portfolio Gewichtungen — Vergleich",
        yaxis_title="Gewichtung (%)",
        template="plotly_white",
        height=450,
        legend=dict(orientation="h", y=-0.15),
        margin=dict(l=0, r=0, t=50, b=80)
    )

    fig.show()


def plot_risk_contribution(rc_df:  pd.DataFrame,
                            title: str) -> None:
    """
    Risk Contribution als Donut Chart + Gewichtungs-Vergleich.
    """
    colors = [
        "#2563eb", "#16a34a", "#f59e0b",
        "#ef4444", "#8b5cf6", "#0891b2",
        "#f97316", "#84cc16"
    ]

    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=[
            "Risiko-Beitrag (%)",
            "Gewicht vs. Risikobeitrag"
        ],
        specs=[[{"type": "pie"}, {"type": "bar"}]]
    )

    # Donut Chart
    fig.add_trace(go.Pie(
        labels=rc_df["Ticker"].tolist(),
        values=rc_df["Risiko (%)"].tolist(),
        hole=0.5,
        marker_colors=colors[:len(rc_df)],
        textinfo="label+percent",
        showlegend=False
    ), row=1, col=1)

    # Bar: Gewicht vs. Risikobeitrag
    x      = rc_df["Ticker"].tolist()
    weight = rc_df["Gewicht (%)"].tolist()
    risk   = rc_df["Risiko (%)"].tolist()

    fig.add_trace(go.Bar(
        name="Gewicht (%)",
        x=x, y=weight,
        marker_color="#2563eb",
        opacity=0.8
    ), row=1, col=2)

    fig.add_trace(go.Bar(
        name="Risikobeitrag (%)",
        x=x, y=risk,
        marker_color="#ef4444",
        opacity=0.8
    ), row=1, col=2)

    fig.update_layout(
        barmode="group",
        title=title,
        template="plotly_white",
        height=420,
        legend=dict(orientation="h", y=1.05),
        margin=dict(l=0, r=0, t=60, b=0)
    )

    fig.show()


def plot_correlation_heatmap(returns: pd.DataFrame) -> None:
    """
    Korrelationsmatrix als Heatmap — Basis für Diversifikation.
    """
    corr = returns.corr().round(3)

    fig = go.Figure(go.Heatmap(
        z=corr.values,
        x=corr.columns.tolist(),
        y=corr.index.tolist(),
        colorscale=[
            [0.0,  "#dc2626"],
            [0.25, "#fca5a5"],
            [0.5,  "#f9fafb"],
            [0.75, "#86efac"],
            [1.0,  "#16a34a"]
        ],
        zmid=0,
        text=corr.values.round(2),
        texttemplate="%{text}",
        textfont=dict(size=11),
        showscale=True
    ))

    fig.update_layout(
        title="Korrelationsmatrix — Diversifikations-Basis",
        template="plotly_white",
        height=500,
        margin=dict(l=0, r=0, t=50, b=0)
    )

    fig.show()

# Backtesting der strategie 

def backtest_portfolios(prices:      pd.DataFrame,
                         portfolios:  dict,
                         rebal_freq:  str   = "ME",
                         capital:     float = 10_000) -> pd.DataFrame:
    """
    Backtested verschiedene Portfolio-Strategien mit Rebalancing.
    Falls eine Position zu groß wird

    rebal_freq:
        ME → Monatlich rebalancieren
        QE → Quartalsweise
        YE → Jährlich

    Rebalancing-Kosten: 0.1% pro Trade
    """
    equity_curves = {}

    for name, port in portfolios.items():
        weights = port["weights"].values
        tickers = port["weights"].index.tolist()

        # Nur verfügbare Ticker
        avail_tickers = [
            t for t in tickers if t in prices.columns
        ]
        avail_weights = np.array([
            port["weights"][t] for t in avail_tickers
        ])
        avail_weights /= avail_weights.sum()

        price_subset = prices[avail_tickers].dropna()

        # Rebalancing-Daten
        rebal_dates  = price_subset.resample(rebal_freq).last().index

        equity       = [capital]
        dates        = [price_subset.index[0]]
        current_val  = capital
        shares       = np.zeros(len(avail_tickers))

        # Initial Buy
        init_prices = price_subset.iloc[0].values
        shares      = (avail_weights * capital) / init_prices

        for i in range(1, len(price_subset)):
            date    = price_subset.index[i]
            prices_ = price_subset.iloc[i].values

            # Portfolio-Wert
            current_val = np.dot(shares, prices_)

            # Rebalancing?
            if date in rebal_dates and i > 0:
                # Transaktionskosten 0.1%
                commission    = current_val * 0.001
                current_val  -= commission
                shares        = (avail_weights * current_val) \
                                / prices_

            equity.append(round(current_val, 2))
            dates.append(date)

        equity_curves[name] = pd.Series(equity, index=dates)

    return pd.DataFrame(equity_curves)


def plot_backtest_comparison(equity_df:  pd.DataFrame,
                              capital:    float) -> None:
    """Vergleicht Portfolio-Performance über Zeit."""
    colors = {
        "Max Sharpe":      "#16a34a",
        "Min Volatilität": "#2563eb",
        "Risk Parity":     "#f59e0b",
        "Equal Weight":    "#94a3b8",
    }

    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        row_heights=[0.65, 0.35],
        subplot_titles=[
            "Portfolio Performance — Backtest",
            "Drawdown (%)"
        ]
    )

    for col in equity_df.columns:
        equity = equity_df[col].dropna()
        color  = colors.get(col, "#8b5cf6")
        lw     = 2 if col != "Equal Weight" else 1.5
        dash   = "solid" if col != "Equal Weight" else "dot"

        fig.add_trace(go.Scatter(
            x=equity.index,
            y=equity.round(2),
            name=col,
            line=dict(color=color, width=lw, dash=dash)
        ), row=1, col=1)

        # Drawdown
        rolling_max = equity.cummax()
        dd = ((equity - rolling_max) / rolling_max * 100).round(2)

        fig.add_trace(go.Scatter(
            x=dd.index, y=dd,
            name=f"{col} DD",
            line=dict(color=color, width=1, dash=dash),
            showlegend=False
        ), row=2, col=1)

    fig.add_hline(
        y=capital,
        line_dash="dot",
        line_color="#94a3b8",
        opacity=0.5,
        row=1, col=1
    )

    fig.update_layout(
        height=650,
        template="plotly_white",
        hovermode="x unified",
        legend=dict(orientation="h", y=1.02),
        margin=dict(l=0, r=0, t=50, b=0)
    )

    fig.update_yaxes(title_text="Kapital ($)", row=1, col=1)
    fig.update_yaxes(title_text="DD (%)",      row=2, col=1)

    fig.show()

if __name__ == "__main__":

    # --- Universe ---
    TICKERS = [
        "AAPL", "MSFT", "GOOGL", "HIMS",
        "JPM",  "JNJ",  "XOM", "TSLA"
    ]
    CAPITAL   = 10_000
    RISK_FREE = 0.04

    print("Tag 21 — Portfolio Optimierung")
    print("=" * 55)

    # Daten laden
    prices  = load_data(TICKERS, "5y")
    returns = compute_returns(prices)

    print(f"Daten: {len(prices)} Tage, "
          f"{len(prices.columns)} Assets")

    # --- Kovarianzmatrix ---
    print("\n1. Kovarianzmatrix (Ledoit-Wolf)")
    cov_lw = compute_covariance_matrix(returns, "ledoit_wolf")
    print(f"   Kondition: {np.linalg.cond(cov_lw.values):.2f}")
    print("   (niedrig = stabil = gut)")

    # --- Korrelationsmatrix ---
    print("\n2. Korrelationsmatrix")
    corr = returns.corr()
    print(corr.round(2))
    plot_correlation_heatmap(returns)

    # --- Monte Carlo ---
    print("\n3. Monte Carlo — 5000 Portfolios generieren...")
    mc_df = monte_carlo_portfolios(
        returns, n_portfolios=5_000, risk_free=RISK_FREE
    )
    print(f"   Sharpe Range: "
          f"{mc_df['sharpe'].min():.2f} bis "
          f"{mc_df['sharpe'].max():.2f}")
    print(f"   Return Range: "
          f"{mc_df['return'].min():.1f}% bis "
          f"{mc_df['return'].max():.1f}%")

    # --- Optimierung ---
    print("\n4. Analytische Optimierung")

    # Max Sharpe
    print("\n   Max Sharpe Portfolio:")
    max_sharpe = optimize_portfolio(
        returns, "sharpe", RISK_FREE
    )
    print(f"   Return:     {max_sharpe['return']:.2f}%")
    print(f"   Volatilität:{max_sharpe['vol']:.2f}%")
    print(f"   Sharpe:     {max_sharpe['sharpe']:.3f}")
    print("\n   Top Gewichte:")
    top_w = max_sharpe["weights"].nlargest(5)
    for ticker, w in top_w.items():
        print(f"     {ticker:<8} {w*100:.1f}%")

    # Min Volatilität
    print("\n   Min Volatilität Portfolio:")
    min_vol = optimize_portfolio(
        returns, "min_vol", RISK_FREE
    )
    print(f"   Return:     {min_vol['return']:.2f}%")
    print(f"   Volatilität:{min_vol['vol']:.2f}%")
    print(f"   Sharpe:     {min_vol['sharpe']:.3f}")

    # Risk Parity
    print("\n   Risk Parity Portfolio:")
    rp = risk_parity_weights(returns, RISK_FREE)
    print(f"   Return:     {rp['return']:.2f}%")
    print(f"   Volatilität:{rp['vol']:.2f}%")
    print(f"   Sharpe:     {rp['sharpe']:.3f}")
    print("\n   Risikobeiträge:")
    print(rp["risk_contrib"][
        ["Ticker", "Gewicht (%)", "Risiko (%)"]
    ].to_string(index=False))

    # Equal Weight
    n        = len(returns.columns)
    eq_w     = pd.Series(
        np.ones(n) / n,
        index=returns.columns
    )
    eq_stats = portfolio_stats(
        eq_w.values, returns.mean(),
        cov_lw, RISK_FREE
    )
    equal_weight = {
        "weights": eq_w,
        "return":  round(eq_stats["return"] * 100, 2),
        "vol":     round(eq_stats["vol"]    * 100, 2),
        "sharpe":  round(eq_stats["sharpe"],       4),
    }

    # --- Efficient Frontier ---
    print("\n5. Efficient Frontier berechnen...")
    frontier_df = efficient_frontier_curve(
        returns, n_points=40, risk_free=RISK_FREE
    )

    # --- Hauptchart ---
    plot_efficient_frontier(
        mc_df, frontier_df,
        max_sharpe, min_vol,
        rp, TICKERS
    )

    # --- Gewichtungsvergleich ---
    portfolios_compare = {
        "Max Sharpe":      max_sharpe,
        "Min Volatilität": min_vol,
        "Risk Parity":     rp,
        "Equal Weight":    equal_weight,
    }
    plot_portfolio_weights(portfolios_compare)

    # --- Risk Contribution ---
    cov_ann = returns.cov() * 252
    rc_maxsharpe = risk_contribution(
        max_sharpe["weights"].values,
        cov_ann,
        returns.columns.tolist()
    )
    plot_risk_contribution(rc_maxsharpe,
                            "Risk Contribution — Max Sharpe")

    rc_rp = rp["risk_contrib"]
    plot_risk_contribution(rc_rp,
                            "Risk Contribution — Risk Parity")

    # --- Black-Litterman ---
    print("\n6. Black-Litterman Modell")
    views = {
        "GLD":  0.08,   # Gold +8% als Hedge
        "XOM":  0.05,   # Energy eher schwach
        "AAPL": 0.12,   # ← stattdessen einen Ticker der wirklich drin ist
    }
    print("   Views:")
    for t, v in views.items():
        print(f"     {t}: {v*100:+.0f}%")

    bl = black_litterman(returns, views, risk_free=RISK_FREE)
    print("\n   BL Posterior Returns:")
    print(bl["bl_returns"].sort_values(ascending=False))
    print("\n   BL Optimale Gewichte (Top 5):")
    print(bl["weights"].nlargest(5))

    # --- Backtest ---
    print("\n7. Portfolio Backtest (monatliches Rebalancing)")
    backtest_df = backtest_portfolios(
        prices, portfolios_compare,
        rebal_freq="ME",
        capital=CAPITAL
    )

    print("\n   Performance Summary:")
    for col in backtest_df.columns:
        eq     = backtest_df[col].dropna()
        ret    = (eq.iloc[-1] / eq.iloc[0] - 1) * 100
        rets   = eq.pct_change().dropna()
        sharpe = (rets.mean() / rets.std() *
                  np.sqrt(252)) if rets.std() > 0 else 0
        dd     = ((eq - eq.cummax()) /
                   eq.cummax()).min() * 100
        print(f"   {col:<20} "
              f"Return={ret:+.1f}%  "
              f"Sharpe={sharpe:.2f}  "
              f"MaxDD={dd:.1f}%")

    plot_backtest_comparison(backtest_df, CAPITAL)

    # --- Constraints Test ---
    print("\n8. Optimierung mit Constraints (max 20% pro Position)")
    max_sharpe_constrained = optimize_portfolio(
        returns, "sharpe", RISK_FREE,
        constraints={"max_weight": 0.20,
                     "min_weight": 0.02}
    )
    print(f"   Sharpe:     "
          f"{max_sharpe_constrained['sharpe']:.3f}")
    print(f"   Return:     "
          f"{max_sharpe_constrained['return']:.2f}%")
    print(f"   Volatilität:{max_sharpe_constrained['vol']:.2f}%")
    print("   Gewichte:")
    for t, w in max_sharpe_constrained["weights"].items():
        if w > 0.01:
            print(f"     {t:<8} {w*100:.1f}%")

    # Export
    mc_df.to_csv("day21_monte_carlo.csv", index=False)
    frontier_df.to_csv("day21_frontier.csv", index=False)
    print("\nGespeichert: day21_monte_carlo.csv, "
          "day21_frontier.csv")


                      