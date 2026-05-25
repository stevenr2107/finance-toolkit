"""
Day 30 — Professionelles Risk Management System

Die Wahrheit über Trading:
    Jeder Anfänger fokussiert auf Entries.
    Jeder Profi fokussiert auf Risk Management.

    Du kannst mit einer 40% Win Rate profitabel sein
    wenn deine Average Win 3x dein Average Loss ist.
    Du kannst mit einer 70% Win Rate pleite gehen
    wenn dein Average Loss 5x dein Average Win ist.

    Risk Management entscheidet — nicht Signale.

Was du heute baust:
    1. Value at Risk (VaR) — Wahrscheinlichkeit großer Verluste
    2. Expected Shortfall (CVaR) — Verlust im Worst Case
    3. Portfolio Heat — Risikokonzentration überwachen
    4. Drawdown Monitor — Max DD in Echtzeit
    5. Kelly Criterion — optimale Positionsgröße
    6. Correlation Monitor — Hedging-Qualität prüfen
    7. Stress Testing — was passiert bei Crash-Szenarien?
    8. Vollständiges Risk Dashboard
"""

import os
import time
import logging
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from typing import Optional, Dict, List
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy import stats
from scipy.stats import norm
import yfinance as yf
from dotenv import load_dotenv

load_dotenv()

API_KEY    = os.getenv("ALPACA_API_KEY",    "")
SECRET_KEY = os.getenv("ALPACA_SECRET_KEY", "")

logging.basicConfig(
    level  = logging.INFO,
    format = "%(asctime)s | %(levelname)s | %(message)s"
)
log = logging.getLogger("RiskManager")


def load_data(tickers: list,
              period:  str = "2y") -> pd.DataFrame:
    """Lädt historische Kursdaten."""
    df = yf.download(
        tickers, period=period,
        auto_adjust=True, progress=False
    )["Close"]
    if len(tickers) == 1:
        df = df.to_frame(name=tickers[0])
    df.columns = df.columns.get_level_values(0)
    return df.dropna()

# *** Value at Risk (VaR) & Expected Shortfall (CVaR) ***

def value_at_risk(returns:     pd.Series,
                  confidence:  float = 0.95, 
                  method:      str   = "historical",
                  horizon:     int   = 1) -> float:
    """
    Value at Risk — der Standard-Risikomaß.

    Aussage:
        VaR(95%, 1 Tag) = -2.3%
        → Mit 95% Wahrscheinlichkeit verlierst du
          nicht mehr als 2.3% an einem Tag.
        → Mit 5% Wahrscheinlichkeit verlierst du MEHR.

    Methoden:
        historical:  Empirische Verteilung der Returns
                     Keine Annahmen über Verteilung.
                     Gut wenn genug Daten vorhanden.

        parametric:  Annahme: Returns sind normalverteilt.
                     Schnell, aber Extremereignisse unterschätzt.
                     Tail Risk wird unterschätzt.

        monte_carlo: Simuliert 10.000 mögliche Returns.
                     Flexibel, kann nicht-normale Verteilungen.
                     Rechenintensiv.

    horizon:
        Skalierung für mehrere Tage.
        VaR(T Tage) = VaR(1 Tag) × √T
        (gilt nur für i.i.d. Returns — Vereinfachung)
    """
    clean = returns.dropna()

    if method == "historical":
        var = np.percentile(clean, (1 - confidence) * 100) # nimm das 5. perzentil

    elif method == "parametric":
        mu    = clean.mean() # durchschnitts return
        sigma = clean.std() # volatilität
        var   = norm.ppf(1 - confidence, mu, sigma) # quantil der normalverteilung
        #Inverse der Glockenkurve -> Retunrs normalverteilt und Extremereignisse unwahrscheinlich

    elif method == "monte_carlo":
        mu      = clean.mean()
        sigma   = clean.std()
        sims    = np.random.normal(mu, sigma, 100_000) # 100.000 zufällige Handelstage und davon 5. Perzentil
        var     = np.percentile(sims, (1 - confidence) * 100)

    else:
        raise ValueError(f"Unbekannte Methode: {method}")

    # Skalierung auf Horizont
    var_scaled = var * np.sqrt(horizon)
    # Square Root of Time -> Berechnet die Volatilität auf eine Periode 
    # Bsp: 1 Tag = 2% vola, 9 Tage = 2% * √9 = 6% vola (weil mehr Tage mehr Risiko)

    return round(float(var_scaled), 6)


def expected_shortfall(returns:    pd.Series,
                        confidence: float = 0.95) -> float:
    """
    Expected Shortfall (CVaR) — besser als VaR.

    *** VaR Problem: ***
        VaR sagt "nicht mehr als X mit 95% Wahrscheinlichkeit".
        Aber was passiert in den 5% schlimmsten Fällen?
        VaR schweigt darüber.

    CVaR (Conditional VaR):
        Durchschnittlicher Verlust in den schlimmsten (1-conf)% Fällen.
        CVaR(95%) = Erwarteter Verlust wenn VaR überschritten wird.
        

        CVaR ist immer schlechter als VaR.
        CVaR ist ehrlicher.
        Hedge Funds nutzen CVaR, nicht VaR.

    Beispiel:
        VaR(95%)  = -2.3%
        CVaR(95%) = -4.1%
        → Wenn es schlimm wird, verlierst du im Schnitt 4.1%.
    """
    # Formel: CVaR = 1/1-a * sum(Li)
    clean    = returns.dropna()
    var      = np.percentile(clean, (1 - confidence) * 100)
    tail     = clean[clean <= var]

    if len(tail) == 0:
        return var

    cvar = tail.mean()
    return round(float(cvar), 6)


def var_comparison(returns:    pd.Series,
                    confidence: float = 0.95,
                    capital:    float = 10_000) -> pd.DataFrame:
    """
    Vergleicht alle Value at Risk Methoden für ein Portfolio.

    Methode       VaR(%)   CVaR(%)   VaR($)   CVaR($)
    Historical    -2.300   -4.100   -230.00   -410.00
    Parametric    -2.150   -4.100   -215.00   -410.00
    Monte_Carlo   -2.280   -4.100   -228.00   -410.00
    """
    methods = ["historical", "parametric", "monte_carlo"]
    rows    = []

    for m in methods:
        var  = value_at_risk(returns, confidence, m)
        cvar = expected_shortfall(returns, confidence)

        rows.append({
            "Methode":          m.title(),
            "VaR (%)":          round(var * 100, 3),
            "CVaR (%)":         round(cvar * 100, 3),
            f"VaR ($, ${capital:,.0f})":  round(var * capital, 2),
            f"CVaR ($, ${capital:,.0f})": round(cvar * capital, 2),
        })

    return pd.DataFrame(rows)

# — stark korrelierte Aktien erhöhen das Risiko, negativ korrelierte reduzieren es.
def portfolio_var(weights:    np.ndarray,
                  returns_df: pd.DataFrame,
                  confidence: float = 0.95,
                  capital:    float = 10_000) -> dict:
    """
    Portfolio VaR — berücksichtigt Korrelationen.

    Wichtig:
        Portfolio VaR ≠ Summe der einzelnen VaRs.
        Korrelation reduziert das Portfolio-Risiko.
        Das ist der mathematische Beweis für Diversifikation.

    Parametric Portfolio VaR:
        σ_p = √(w^T × Σ × w)
        VaR_p = σ_p × z_alpha

    wobei:
        w  = Gewichtsvektor
        Σ  = Kovarianzmatrix der Returns
        z  = Quantil der Normalverteilung
    """
    # Kovarianzmatrix - wie stark bewegen sich die Aktien gemeinsam?
    """
            AAPL    MSFT    NVDA
    AAPL  [ 0.04   0.02   0.03 ]
    MSFT  [ 0.02   0.05   0.02 ]
    NVDA  [ 0.03   0.02   0.08 ] 0.08 = Volatilität von NVDA
    """
    returns  = returns_df.pct_change().dropna()
    cov      = returns.cov() * 252   # Annualisiert
    weights  = np.array(weights)

    # Portfolio Volatilität (täglich)
    port_var_annual = weights.T @ cov.values @ weights
    port_vol_daily  = np.sqrt(port_var_annual / 252)

    # VaR Z score -1.645 für 95% Konfidenz
    z     = norm.ppf(1 - confidence)
    var_1d = port_vol_daily * z   # wie weit kann der kurs in 1 tag fallen
    var_5d = var_1d * np.sqrt(5)

    # Historical Portfolio Returns
    port_returns = returns @ weights # multipliziert tages return mit gewicht der aktie 
    hist_var     = np.percentile(
        port_returns, (1 - confidence) * 100
    )
    hist_cvar    = port_returns[
        port_returns <= hist_var
    ].mean()

    # Marginale VaR Beiträge
    # Was passiert mit dem Portfolio wenn ich 1% mehr in AAPL investiere?
    marginal_var = {}
    for i, ticker in enumerate(returns_df.columns):
        w_bump    = weights.copy()
        w_bump[i] += 0.01   # +1% in dieser Aktie
        w_bump    /= w_bump.sum() # auf 100% normalisieren

        var_bump     = np.sqrt(
            w_bump.T @ cov.values @ w_bump / 252 # neue Portfolio Volatilität
        ) * z
        marginal_var[ticker] = round(
            float((var_bump - port_vol_daily * z) * capital), 2
        ) # differenz zwischen neuem VaR und altem VaR in Dollar
        # Ergebnis z.B.:
        # AAPL: +$12  → 1% mehr AAPL erhöht Risiko um $12
        # MSFT: +$8   → weniger riskant im Portfolio-Kontext
        # NVDA: +$25  → am riskantesten, höchste Korrelation

    return {
        "var_1d_pct":    round(float(var_1d) * 100, 3),
        "var_5d_pct":    round(float(var_5d) * 100, 3),
        "var_1d_dollar": round(float(var_1d) * capital, 2),
        "var_5d_dollar": round(float(var_5d) * capital, 2),
        "hist_var_pct":  round(float(hist_var) * 100, 3),
        "hist_cvar_pct": round(float(hist_cvar) * 100, 3),
        "port_vol_annual": round(
            float(np.sqrt(port_var_annual)) * 100, 2
        ),
        "marginal_var":  marginal_var,
    }

def compute_drawdown(equity_curve: pd.Series) -> pd.DataFrame:
    """
    Vollständige Drawdown Analyse.

    Drawdown:
        Verlust vom letzten Peak bis zum aktuellen Tief.
        Drawdown = (Current - Peak) / Peak

    Max Drawdown:
        Größter Peak-to-Trough Verlust.
        Das ist die Zahl die Institutionelle am meisten beachten.

    Drawdown Duration:
        Wie lange dauert ein Drawdown?
        Länge = Zeit von Peak bis zum Recovery auf neues High.

    Calmar Ratio:
        CAGR / |Max Drawdown|
        Mißt Return pro Drawdown-Einheit.
        > 1.0 ist gut. > 2.0 ist exzellent.
    """
    equity      = equity_curve.dropna()
    rolling_max = equity.cummax() # berechnet den bisherigen Höchststand der Equity-Kurve
    drawdown    = (equity - rolling_max) / rolling_max * 100 # berechnet den Drawdown in Prozent

    # Drawdown Perioden identifizieren
    in_dd      = drawdown < 0 # True wenn Portfolio unter Peak
    dd_starts  = []
    dd_ends    = []
    dd_depths  = []

    i = 0
    while i < len(in_dd):
        if in_dd.iloc[i]: # Drawdown beginnt
            start = i
            depth = drawdown.iloc[i]

            while i < len(in_dd) and in_dd.iloc[i]:
                depth = min(depth, drawdown.iloc[i]) # tiefstes Tief
                i    += 1

            end = i - 1
            dd_starts.append(equity.index[start]) # start drawdown
            dd_ends.append(equity.index[end]) # ende drawdown
            dd_depths.append(round(float(depth), 2)) # teifste drawdown ind er Zeit
        else:
            i += 1

    # Kennzahlen
    max_dd       = float(drawdown.min())
    current_dd   = float(drawdown.iloc[-1])
    avg_dd       = float(drawdown[drawdown < 0].mean()) \
                   if (drawdown < 0).any() else 0

    # Calmar Ratio
    returns      = equity.pct_change().dropna()
    years        = len(equity) / 252
    cagr         = ((equity.iloc[-1] / equity.iloc[0]) **
                    (1/max(years, 0.01)) - 1) * 100
    calmar       = cagr / abs(max_dd) if max_dd != 0 else 0

    # Underwater Perioden
    dd_periods = pd.DataFrame({
        "Start": dd_starts,
        "Ende":  dd_ends,
        "Tiefe (%)": dd_depths,
    }).sort_values("Tiefe (%)").head(5)

    return {
        "drawdown_series":  drawdown,
        "max_drawdown":     round(max_dd, 2),
        "current_drawdown": round(current_dd, 2),
        "avg_drawdown":     round(avg_dd, 2),
        "calmar_ratio":     round(calmar, 2),
        "cagr_pct":         round(cagr, 2),
        "top_drawdowns":    dd_periods,
        "n_dd_periods":     len(dd_starts),
    }


class DrawdownMonitor:
    """
    Echtzeit Drawdown Monitoring für den Trading Bot.

    Tracked:
        Intraday High Water Mark
        Aktueller Drawdown vom High Water Mark
        Alert wenn Drawdown Threshold überschritten
    """

    def __init__(self,
                 alert_threshold_pct: float = 3.0,
                 kill_threshold_pct:  float = 8.0):
        self.alert_threshold = alert_threshold_pct # Grenze ab der ein Alert ausgelöst wird
        self.kill_threshold  = kill_threshold_pct # Grenze ab der ein Kill Switch aktiviert wird
        self.high_water_mark = None # Höchste Portfolio Wert den es gab 
        self.peak_date       = None
        self._alerts         = []

    # Update wenn neues Hoch 
    def update(self,
               portfolio_value: float,
               date:            datetime = None) -> dict:
        """
        Update mit aktuellem Portfolio-Wert.
        Gibt Status zurück.
        """
        if date is None:
            date = datetime.now()

        # High Water Mark Update wenn neues Hoch
        if (self.high_water_mark is None or
                portfolio_value > self.high_water_mark):
            self.high_water_mark = portfolio_value
            self.peak_date       = date

        # Drawdown seit hoch
        dd_pct = (
            (portfolio_value - self.high_water_mark) /
            self.high_water_mark * 100
        )

        # Status
        if dd_pct < -self.kill_threshold:
            status = "KILL"
        elif dd_pct < -self.alert_threshold:
            status = "ALERT"
        else:
            status = "OK"

        # dd_pct:    0%    -3%         -8%
            #        │      │           │
        # Status:   OK → ALERT  ──► KILL
        #                             ↓
        #                close_all_positions()

        # Alert loggen
        if status in ["ALERT", "KILL"]:
            alert = {
                "timestamp":    date.isoformat(),
                "dd_pct":       round(dd_pct, 2),
                "portfolio":    portfolio_value,
                "hwm":          self.high_water_mark,
                "status":       status,
            }
            self._alerts.append(alert)
            log.warning(
                f"🚨 Drawdown Alert: {dd_pct:.2f}% "
                f"(HWM: ${self.high_water_mark:,.2f})"
            )

        return {
            "portfolio":     portfolio_value,
            "hwm":           self.high_water_mark,
            "dd_pct":        round(dd_pct, 2),
            "status":        status,
            "days_in_dd":    (date - self.peak_date).days
                              if self.peak_date else 0,
        }

    def get_alerts(self) -> pd.DataFrame:
        if not self._alerts:
            return pd.DataFrame()
        return pd.DataFrame(self._alerts)
    
# Welche positionen dominieren mein Risiko nicht Kapital 
# Position    Kapital-Gewicht    Risiko-Beitrag
#  TSLA           10%               25%  ← dominiert das Risiko
#   AAPL           10%               8%
#   JPM            10%               5%
class PortfolioHeat:
    """
    Portfolio Heat — Risikokonzentration visualisieren.

    Idee:
        Nicht nur Gewichtung (% Dollar) betrachten.
        Sondern: % des Gesamtrisikos pro Position.



        Das ist der Unterschied den Risk Parity löst.

    Heat Score:
        (Positions-Volatilität × Gewicht) / Portfolio-Volatilität
        Zeigt welche Positionen das Portfolio dominieren.
    """

    def __init__(self, positions: pd.DataFrame,
                 returns_data: pd.DataFrame):
        """
        positions:    DataFrame mit ticker, qty, market_val
        returns_data: Historische Returns aller Ticker
        """
        self.positions    = positions
        self.returns_data = returns_data

    def compute_heat(self) -> pd.DataFrame:
        """Berechnet Heat Score für alle Positionen."""
        if self.positions.empty:
            return pd.DataFrame()

        total_val = self.positions["market_val"].sum()
        rows      = []

        for _, pos in self.positions.iterrows():
            ticker = pos["ticker"]

            if ticker not in self.returns_data.columns:
                continue

            ret      = self.returns_data[ticker].pct_change().dropna()
            weight   = pos["market_val"] / total_val
            vol_ann  = ret.std() * np.sqrt(252) * 100 # annualisierte vola in %
            beta_spy = self._compute_beta(ticker)

            # Risk Contribution
            # TSLA: Gewicht 10% × Vola 45% = Risk Contrib 4.5%
            # JPM:  Gewicht 10% × Vola 18% = Risk Contrib 1.8%
            risk_contrib = weight * vol_ann

            # Heat Score: 0-10 Skala
            heat = min(
                risk_contrib / 5,  # 5% risk contribution = heat 1
                10.0
            )

            rows.append({
                "Ticker":        ticker,
                "Gewicht (%)":   round(weight * 100, 1),
                "Vola (%)":      round(float(vol_ann), 1),
                "Beta (SPY)":    round(float(beta_spy), 2),
                "Risk Contrib %": round(float(risk_contrib), 2),
                "Heat Score":    round(float(heat), 2),
                "Market Val ($)": round(float(pos["market_val"]), 2),
                "PnL ($)":       round(float(pos.get("pnl_abs", 0)), 2),
            })

        result = pd.DataFrame(rows)
        if not result.empty:
            result = result.sort_values(
                "Heat Score", ascending=False
            ).reset_index(drop=True)

        return result

        # Beta berechnen 
    def _compute_beta(self, ticker: str,
                      benchmark: str = "SPY") -> float:
        """Beta vs. SPY."""
        try:
            if benchmark not in self.returns_data.columns: # falls SPY nicht in Daten lad sie von YFinance runetr
                spy = yf.download(
                    benchmark, period="1y",
                    auto_adjust=True, progress=False
                )["Close"].pct_change().dropna()
            else:
                spy = self.returns_data[
                    benchmark
                ].pct_change().dropna()

            stock = self.returns_data[
                ticker
            ].pct_change().dropna()

            aligned = pd.concat(
                [stock, spy], axis=1 # beide in einen DataFrame packen und nach Datum ausrichten
            ).dropna() # nur Tage behalten an denen beide Daten haben

            if len(aligned) < 10:
                return 1.0

            cov  = aligned.cov().iloc[0, 1] # Kovarianz von Aktie und SPY
            var  = aligned.iloc[:, 1].var() # Varianz SPY
            return float(cov / var) if var > 0 else 1.0 # Beta = Cov(Aktie,SPY) / Var(SPY)
        # 0.5 halb so volatil wie der markt 
        # 1.0 genauso volatil wie der markt
        # 2.0 TSLA-Niveau - doppelt so volatil wie der markt

        except Exception:
            return 1.0

    def get_total_heat(self) -> float:
        """Gesamter Heat Score des Portfolios."""
        heat_df = self.compute_heat()
        if heat_df.empty:
            return 0.0
        return float(heat_df["Heat Score"].sum())

    def is_overheated(self,
                       threshold: float = 7.0) -> bool:
        """True wenn Portfolio zu konzentriert ist."""
        return self.get_total_heat() > threshold

# Kelly Criterion
# wieviel kapital sollte man in einen Trade stecken? 


def kelly_criterion(win_rate:   float,
                    avg_win:    float,
                    avg_loss:   float,
                    fraction:   float = 0.5) -> dict:
    """
    Kelly Criterion — mathematisch optimale Positionsgröße.

    Formel:
        f* = W - (1-W) / (W/L)
        f* = Win Rate - (Loss Rate / Odds)

    Warum Half-Kelly?
        Full Kelly ist theoretisch optimal für langfristiges Wachstum.
        Aber: hohe Varianz, emotionale Belastung.
        Kein professioneller Trader nutzt Full Kelly.
        Half Kelly = f*/2 → stabiler, weniger Drawdown.

    bei full kelly und odds von 2.0 also avg wind = 2 und avg loss = 1 
    mit einer win rate von 55% 
    wäre die opt. größe 32,5% 

    Wenn Kelly negativ → Strategie hat negative Erwartung → nicht traden.
    """
    if avg_loss == 0:
        return {"kelly": 0, "half_kelly": 0,
                "verdict": "Kein Verlust = Daten unvollständig"}

    loss_rate = 1 - win_rate
    odds      = avg_win / avg_loss

    kelly_full = win_rate - (loss_rate / odds)
    kelly_frac = kelly_full * fraction

    # Capped: max 25% des Portfolios
    kelly_capped = min(max(kelly_frac, 0), 0.25)

    if kelly_full < 0:
        verdict = "❌ Negative Erwartung — nicht traden"
        # verliert langfristig geld 
    elif kelly_full < 0.05:
        verdict = "⚠ Sehr kleiner Edge — vorsichtig"
    elif kelly_full < 0.15:
        verdict = "✅ Moderater Edge"
    else:
        verdict = "🟢 Starker Edge"

    return {
        "kelly_full":    round(kelly_full * 100, 2),
        "half_kelly":    round(kelly_capped * 100, 2),
        "win_rate":      round(win_rate * 100, 1),
        "avg_win":       round(avg_win * 100, 2),
        "avg_loss":      round(avg_loss * 100, 2),
        "odds":          round(odds, 3),
        "verdict":       verdict,
        "recommended_pct": round(kelly_capped * 100, 1),
    }


def dynamic_position_size(capital:      float,
                            price:        float,
                            stop_loss:    float,
                            risk_pct:     float = 0.02,
                            kelly_pct:    float = None,
                            volatility:   float = None) -> dict:
    """
    Kombinierter Position Sizing Ansatz.

    Methode 1: Fixed Risk (2% des Kapitals)
        Shares = (Capital × Risk%) / (Entry - Stop Loss)

    Methode 2: Kelly (wenn verfügbar)
        Shares = (Capital × Kelly%) / Price

    Methode 3: Volatility Adjusted
        Kleinere Position bei hoher Volatilität.
        Größere Position bei niedriger Volatilität.

    Empfehlung: Minimum der drei Methoden.
    """
    results = {}

    # Methode 1: Fixed Risk
    risk_dollar  = capital * risk_pct
    risk_per_shr = price - stop_loss
    if risk_per_shr > 0:
        shares_risk = int(risk_dollar / risk_per_shr)
    else:
        shares_risk = 0
    results["fixed_risk"] = shares_risk

    # Methode 2: Kelly
    if kelly_pct is not None:
        shares_kelly = int(
            (capital * kelly_pct / 100) / price
        )
        results["kelly"] = shares_kelly
    else:
        results["kelly"] = shares_risk

    # Methode 3: Volatility Adjusted
    if volatility is not None and volatility > 0:
        # Basis: 20% Jahresvolatilität = Normalgröße
        # 10% jahresvola = Doppelte Position
        vol_scalar   = 0.20 / max(volatility, 0.05)
        base_dollar  = capital * risk_pct * 3
        shares_vol   = int(base_dollar * vol_scalar / price)
        results["vol_adjusted"] = shares_vol
    else:
        results["vol_adjusted"] = shares_risk

    # Empfehlung: konservativste Methode
    recommended = min(
        results["fixed_risk"],
        results["kelly"],
        results["vol_adjusted"]
    )
    recommended = max(recommended, 0)

    return {
        "fixed_risk_shares":   results["fixed_risk"],
        "kelly_shares":        results["kelly"],
        "vol_adj_shares":      results["vol_adjusted"],
        "recommended_shares":  recommended,
        "recommended_dollar":  round(recommended * price, 2),
        "pct_of_capital":      round(
            recommended * price / capital * 100, 2
        ),
        "stop_loss":           stop_loss,
        "risk_dollar":         round(
            recommended * risk_per_shr, 2
        ) if risk_per_shr > 0 else 0,
    }

@dataclass
class StressScenario:
    """Ein Stress-Test Szenario."""
    name:        str
    description: str
    shocks:      Dict[str, float]   # Ticker → Schock in %


STRESS_SCENARIOS = [
    StressScenario(
        name        = "2020 COVID Crash",
        description = "S&P 500 -34% in 5 Wochen (Feb–Mär 2020)",
        shocks      = {
            "AAPL": -0.32, "MSFT": -0.28,
            "NVDA": -0.35, "SPY":  -0.34,
            "QQQ":  -0.29, "JPM":  -0.45,
            "GLD":  +0.08,
        }
    ),
    StressScenario(
        name        = "2022 Zinsanstieg",
        description = "Fed hebt Zinsen aggressiv an, Tech verliert 40%+",
        shocks      = {
            "AAPL": -0.28, "MSFT": -0.30,
            "NVDA": -0.65, "SPY":  -0.20,
            "QQQ":  -0.35, "JPM":  -0.15,
            "GLD":  -0.02,
        }
    ),
    StressScenario(
        name        = "2008 Finanzkrise",
        description = "Lehman Brothers, globale Bankenkrise",
        shocks      = {
            "AAPL": -0.55, "MSFT": -0.48,
            "NVDA": -0.70, "SPY":  -0.57,
            "QQQ":  -0.49, "JPM":  -0.68,
            "GLD":  +0.05,
        }
    ),
    StressScenario(
        name        = "Flash Crash",
        description = "Plötzlicher Intraday-Einbruch -10%",
        shocks      = {
            "AAPL": -0.10, "MSFT": -0.10,
            "NVDA": -0.12, "SPY":  -0.10,
            "QQQ":  -0.11, "JPM":  -0.09,
            "GLD":  -0.02,
        }
    ),
    StressScenario(
        name        = "Soft Landing",
        description = "Moderate Korrektur -15%, sanfte Erholung",
        shocks      = {
            "AAPL": -0.15, "MSFT": -0.14,
            "NVDA": -0.20, "SPY":  -0.15,
            "QQQ":  -0.17, "JPM":  -0.12,
            "GLD":  +0.05,
        }
    ),
]


def run_stress_tests(positions:  pd.DataFrame,
                      capital:    float,
                      scenarios:  List[StressScenario] = None) -> pd.DataFrame:
    """
    Stress Tests — was passiert bei Crash-Szenarien?

    Für jedes Szenario:
        1. Schock auf alle Positionen anwenden
        2. Portfolio-Verlust berechnen
        3. Margin Call Risiko prüfen
        4. Recovery-Zeit schätzen
    """
    if scenarios is None:
        scenarios = STRESS_SCENARIOS

    if positions.empty:
        print("Keine Positionen für Stress Test.")
        return pd.DataFrame()

    results = []

    for scenario in scenarios:
        portfolio_loss    = 0.0
        position_losses   = {}

        for _, pos in positions.iterrows():
            ticker    = pos["ticker"]
            mkt_val   = float(pos.get("market_val", 0))
            shock     = scenario.shocks.get(ticker, -0.15) # falls ticker nicht drin standard 15%
            pos_loss  = mkt_val * shock # Verlust in Dollar
            portfolio_loss   += pos_loss
            position_losses[ticker] = round(pos_loss, 2)

        # Cash unberührt
        cash           = capital - positions["market_val"].sum()
        total_after    = capital + portfolio_loss
        loss_pct       = portfolio_loss / capital * 100

        # Recovery Zeit (grobe Schätzung bei 10% CAGR)
        # wie lange braucht die aktie um zum anfangswert zu kommen 
        # -70% -> 13,5 jahre 
        if portfolio_loss < 0:
            recovery_years = (
                np.log(capital / total_after) /
                np.log(1.10)
            ) if total_after > 0 else 999
        else:
            recovery_years = 0

        results.append({
            "Szenario":          scenario.name,
            "Beschreibung":      scenario.description,
            "Portfolio Verlust": round(portfolio_loss, 2),
            "Verlust (%)":       round(float(loss_pct), 2),
            "Portfolio danach":  round(total_after, 2),
            "Recovery (Jahre)":  round(recovery_years, 1),
            **{f"PnL {k}": v # ** alle positions verluste als einzelne spalte 
               for k, v in position_losses.items()},
        })
        #"PnL AAPL":   -5500,
        #"PnL NVDA":   -7000,
        #"PnL GLD":    +500,

    result_df = pd.DataFrame(results).sort_values(
        "Verlust (%)"
    ).reset_index(drop=True)

    return result_df


def correlation_monitor(tickers:     list,
                          period:      str   = "6mo",
                          threshold:   float = 0.80) -> dict:
    """
    Überwacht Korrelationen zwischen Positionen.

    Problem:
        Wenn alle deine Positionen hoch korreliert sind
        ist deine Diversifikation illusorisch.
        Bei einem Crash verlierst du überall gleichzeitig.

    Warnung wenn:
        Durchschnittliche Korrelation > threshold
        → Positionen zu ähnlich → Diversifikation fehlt

    Lösung:
        Positionen aus verschiedenen Sektoren wählen.
        Gold (GLD) als Anti-Korrelations-Hedge.
        Anleihen (BND/TLT) als defensiver Puffer.
    """
    df      = load_data(tickers, period)
    returns = df.pct_change().dropna()
    corr    = returns.corr()

    # Durchschnittliche paarweise Korrelation
    mask         = np.triu(np.ones(corr.shape), k=1).astype(bool) # obere Dreiecksmatrix ohne Diagonale
    corr_values  = corr.values[mask] # da man ja alles quasi doppel hat 
    avg_corr     = float(np.mean(corr_values))
    max_corr     = float(np.max(corr_values))

    # Hochkorrelierte Paare identifizieren
    high_corr_pairs = []
    for i in range(len(tickers)):
        for j in range(i+1, len(tickers)):
            c = corr.iloc[i, j]
            if abs(c) > threshold:
                high_corr_pairs.append({
                    "Pair":        f"{tickers[i]}/{tickers[j]}",
                    "Korrelation": round(float(c), 3),
                    "Warnung":     "🔴 Zu hoch" if c > threshold
                                   else "✅ OK"
                })

    status = "🔴 Überkorreliert" if avg_corr > threshold \
             else ("⚠ Mäßig" if avg_corr > 0.6 else "✅ Gut")
    
    # 0,8 -> 80% Korrelation -> sehr hoch
    # 0,6 -> 60% Korrelation -> eher hoch
    # < 0,6 -> gute Diversifikation

    return {
        "corr_matrix":    corr,
        "avg_corr":       round(avg_corr, 3),
        "max_corr":       round(max_corr, 3),
        "status":         status,
        "high_corr_pairs": pd.DataFrame(high_corr_pairs),
        "n_positions":    len(tickers),
    }


def plot_var_analysis(returns:  pd.Series,
                       capital:  float,
                       ticker:   str) -> None:
    """
    VaR Analyse mit Verteilung und Konfidenzintervallen.
    """
    clean = returns.dropna() * 100   # in %

    var_95_h = value_at_risk(returns, 0.95, "historical") * 100
    var_99_h = value_at_risk(returns, 0.99, "historical") * 100
    cvar_95  = expected_shortfall(returns, 0.95) * 100

    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=[
            f"{ticker} — Return Verteilung + VaR",
            "VaR Methoden Vergleich"
        ],
        horizontal_spacing=0.12
    )

    # Histogramm
    fig.add_trace(go.Histogram(
        x=clean,
        nbinsx=80,
        name="Tages-Returns",
        marker_color="#3b82f6",
        opacity=0.70,
        showlegend=True
    ), row=1, col=1)

    # VaR Linien
    for val, color, label in [
        (var_95_h, "#f59e0b", "VaR 95%"),
        (var_99_h, "#ef4444", "VaR 99%"),
        (cvar_95,  "#dc2626", "CVaR 95%"),
    ]:
        fig.add_vline(
            x=val,
            line_dash="dash",
            line_color=color,
            line_width=2,
            annotation_text=f"{label}: {val:.2f}%",
            annotation_position="top left",
            row=1, col=1
        )

    # Tail Bereich einfärben
    tail_x = clean[clean <= var_95_h]
    if not tail_x.empty:
        fig.add_trace(go.Histogram(
            x=tail_x,
            nbinsx=20,
            name="Tail (schlechteste 5%)",
            marker_color="#ef4444",
            opacity=0.6,
        ), row=1, col=1)

    # VaR Vergleich Bars
    methods = ["Historical", "Parametric", "Monte Carlo"]
    var_values = [
        value_at_risk(returns, 0.95, m.lower()) * 100
        for m in ["historical", "parametric", "monte_carlo"]
    ]
    cvar_val = expected_shortfall(returns, 0.95) * 100

    fig.add_trace(go.Bar(
        x=methods,
        y=[abs(v) for v in var_values],
        name="VaR 95% (%)",
        marker_color="#3b82f6",
        text=[f"{abs(v):.3f}%" for v in var_values],
        textposition="outside",
    ), row=1, col=2)

    fig.add_hline(
        y=abs(cvar_val),
        line_dash="dash",
        line_color="#ef4444",
        annotation_text=f"CVaR: {abs(cvar_val):.3f}%",
        row=1, col=2
    )

    fig.update_layout(
        height=450,
        template="plotly_white",
        title=f"{ticker} — VaR & CVaR Analyse "
              f"(Kapital: ${capital:,.0f})",
        legend=dict(orientation="h", y=1.02),
        margin=dict(l=0, r=0, t=60, b=0)
    )

    fig.update_xaxes(title_text="Return (%)", row=1, col=1)
    fig.update_xaxes(title_text="Methode",    row=1, col=2)
    fig.update_yaxes(title_text="Häufigkeit", row=1, col=1)
    fig.update_yaxes(title_text="|VaR| (%)",  row=1, col=2)

    fig.show()


def plot_drawdown_analysis(equity_curve: pd.Series,
                            ticker:       str) -> None:
    """
    Vollständige Drawdown Visualisierung.
    """
    dd_result = compute_drawdown(equity_curve)
    drawdown  = dd_result["drawdown_series"]

    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        row_heights=[0.60, 0.40],
        subplot_titles=[
            f"{ticker} — Equity Curve",
            f"Drawdown (Max: {dd_result['max_drawdown']:.2f}%)"
        ]
    )

    # Equity Curve
    fig.add_trace(go.Scatter(
        x=equity_curve.index,
        y=equity_curve.round(2),
        name="Portfolio",
        line=dict(color="#2563eb", width=2),
        fill="tozeroy",
        fillcolor="rgba(37,99,235,0.06)"
    ), row=1, col=1)

    # Drawdown
    fig.add_trace(go.Scatter(
        x=drawdown.index,
        y=drawdown.round(2),
        name="Drawdown",
        line=dict(color="#ef4444", width=1.5),
        fill="tozeroy",
        fillcolor="rgba(239,68,68,0.12)"
    ), row=2, col=1)

    # Max DD Linie
    fig.add_hline(
        y=dd_result["max_drawdown"],
        line_dash="dash",
        line_color="#dc2626",
        line_width=1.5,
        annotation_text=(
            f"Max DD: {dd_result['max_drawdown']:.2f}%"
        ),
        row=2, col=1
    )

    # Alert Level
    fig.add_hline(
        y=-5.0,
        line_dash="dot",
        line_color="#f59e0b",
        opacity=0.6,
        annotation_text="Alert -5%",
        row=2, col=1
    )
    fig.add_hline(
        y=-10.0,
        line_dash="dot",
        line_color="#ef4444",
        opacity=0.6,
        annotation_text="Kill Switch -10%",
        row=2, col=1
    )

    # Metrics Annotation
    fig.add_annotation(
        x=0.01, y=0.97,
        xref="paper", yref="paper",
        text=(
            f"CAGR: {dd_result['cagr_pct']:.1f}% | "
            f"Max DD: {dd_result['max_drawdown']:.1f}% | "
            f"Calmar: {dd_result['calmar_ratio']:.2f}"
        ),
        showarrow=False,
        bgcolor="rgba(255,255,255,0.8)",
        bordercolor="#e2e8f0",
        font=dict(size=11)
    )

    fig.update_layout(
        height=580,
        template="plotly_white",
        hovermode="x unified",
        legend=dict(orientation="h", y=1.02),
        margin=dict(l=0, r=0, t=60, b=0)
    )

    fig.update_yaxes(title_text="Wert ($)",   row=1, col=1)
    fig.update_yaxes(title_text="DD (%)",     row=2, col=1)

    fig.show()


def plot_portfolio_heat(heat_df: pd.DataFrame) -> None:
    """Portfolio Heat Map — Risikokonzentration."""
    if heat_df.empty:
        print("Keine Positionen.")
        return

    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=[
            "Heat Score nach Ticker",
            "Gewicht vs. Risikobeitrag"
        ],
        horizontal_spacing=0.12
    )

    # Heat Bars
    heat_colors = []
    for h in heat_df["Heat Score"]:
        if h >= 6:
            heat_colors.append("#dc2626")
        elif h >= 4:
            heat_colors.append("#f59e0b")
        elif h >= 2:
            heat_colors.append("#3b82f6")
        else:
            heat_colors.append("#16a34a")

    fig.add_trace(go.Bar(
        x=heat_df["Ticker"],
        y=heat_df["Heat Score"],
        marker_color=heat_colors,
        text=[f"{h:.1f}" for h in heat_df["Heat Score"]],
        textposition="outside",
        name="Heat Score",
        showlegend=False
    ), row=1, col=1)

    # Warnschwelle
    fig.add_hline(
        y=6.0, line_dash="dash",
        line_color="#ef4444",
        annotation_text="⚠ Hoch",
        row=1, col=1
    )

    # Gewicht vs. Risiko
    fig.add_trace(go.Bar(
        x=heat_df["Ticker"],
        y=heat_df["Gewicht (%)"],
        name="Gewicht (%)",
        marker_color="#3b82f6",
        opacity=0.8,
    ), row=1, col=2)

    fig.add_trace(go.Bar(
        x=heat_df["Ticker"],
        y=heat_df["Risk Contrib %"],
        name="Risikobeitrag (%)",
        marker_color="#ef4444",
        opacity=0.8,
    ), row=1, col=2)

    fig.update_layout(
        barmode="group",
        height=420,
        template="plotly_white",
        title="Portfolio Heat — Risikokonzentration",
        legend=dict(orientation="h", y=1.02),
        margin=dict(l=0, r=0, t=60, b=0)
    )

    fig.update_yaxes(title_text="Heat Score", row=1, col=1,
                     range=[0, 10])
    fig.update_yaxes(title_text="%",          row=1, col=2)

    fig.show()


def plot_stress_test(stress_df:   pd.DataFrame,
                      capital:     float) -> None:
    """Stress Test Ergebnisse visualisieren."""
    if stress_df.empty:
        return

    colors = [
        "#ef4444" if v < -15
        else ("#f59e0b" if v < -8 else "#16a34a")
        for v in stress_df["Verlust (%)"]
    ]

    fig = go.Figure()

    fig.add_trace(go.Bar(
        x=stress_df["Szenario"],
        y=stress_df["Verlust (%)"],
        marker_color=colors,
        text=[f"{v:.1f}%" for v in stress_df["Verlust (%)"]],
        textposition="outside",
        name="Szenario Verlust",
    ))

    # Referenzlinien
    for level, color, label in [
        (-5,  "#f59e0b", "Alert -5%"),
        (-10, "#ef4444", "Kill -10%"),
        (-20, "#dc2626", "Katastrophal -20%"),
    ]:
        fig.add_hline(
            y=level,
            line_dash="dash",
            line_color=color,
            opacity=0.6,
            annotation_text=label
        )

    fig.update_layout(
        title=f"Stress Test Ergebnisse "
              f"(Portfolio: ${capital:,.0f})",
        yaxis_title="Portfolio Verlust (%)",
        template="plotly_white",
        height=450,
        margin=dict(l=0, r=0, t=60, b=0)
    )

    fig.show()


def plot_correlation_heatmap(corr_result: dict) -> None:
    """Korrelationsmatrix als Heatmap."""
    corr = corr_result["corr_matrix"]

    fig = go.Figure(go.Heatmap(
        z=corr.values.round(2),
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
        textfont=dict(size=12),
        showscale=True,
        zmin=-1, zmax=1
    ))

    avg    = corr_result["avg_corr"]
    status = corr_result["status"]

    fig.update_layout(
        title=(
            f"Korrelationsmatrix — "
            f"Avg: {avg:.2f} — {status}"
        ),
        template="plotly_white",
        height=480,
        margin=dict(l=0, r=0, t=60, b=0)
    )

    fig.show()

class RiskDashboard:
    """
    Zentrales Risk Management Dashboard.
    Kombiniert alle Risk-Metriken in einem Report.
    """

    def __init__(self,
                 tickers:  list,
                 capital:  float = 10_000,
                 period:   str   = "2y"):
        self.tickers  = tickers
        self.capital  = capital
        self.period   = period
        self.prices   = load_data(tickers, period)
        self.returns  = self.prices.pct_change().dropna()

    def run_full_analysis(self,
                           positions: pd.DataFrame = None,
                           weights:   list = None) -> dict:
        """
        Führt alle Risk-Analysen aus und gibt Report zurück.
        """
        results = {}

        # Default: Equal Weight
        if weights is None: # wenn keine gewichte übergeben werden dann gleichgewichtet
            n       = len(self.tickers)
            weights = [1/n] * n

        # 1. VaR pro Ticker
        print("\n📊 VaR Analyse...")
        var_results = {}
        for ticker in self.tickers:
            ret = self.returns[ticker]
            var_results[ticker] = {
                "var_95_hist": round(
                    value_at_risk(ret, 0.95, "historical")
                    * 100, 3
                ),
                "var_99_hist": round(
                    value_at_risk(ret, 0.99, "historical")
                    * 100, 3
                ),
                "cvar_95": round(
                    expected_shortfall(ret, 0.95) * 100, 3
                ),
            }
        results["var"] = var_results

        # 2. Portfolio VaR
        print("📊 Portfolio VaR...")
        port_var = portfolio_var(
            weights, self.prices, 0.95, self.capital
        )
        results["portfolio_var"] = port_var

        # 3. Drawdown (SPY als Proxy wenn keine echte Equity)
        print("📊 Drawdown Analyse...")
        equity_proxy = (
            (1 + (self.returns @ weights))
            .cumprod() * self.capital
        )
        dd_result          = compute_drawdown(equity_proxy)
        results["drawdown"] = dd_result

        # 4. Korrelation
        print("📊 Korrelations-Monitor...")
        corr_result          = correlation_monitor(
            self.tickers, self.period
        )
        results["correlation"] = corr_result

        # 5. Stress Tests
        print("📊 Stress Tests...")
        if positions is not None and not positions.empty:
            stress = run_stress_tests(
                positions, self.capital
            )
        else:
            # Synthetische Positionen aus Gewichten
            syn_pos = pd.DataFrame([{
                "ticker":     t,
                "market_val": self.capital * w,
                "pnl_abs":    0,
            } for t, w in zip(self.tickers, weights)])
            stress = run_stress_tests(syn_pos, self.capital)
        results["stress"] = stress

        # 6. Kelly für Portfolio
        print("📊 Kelly Criterion...")
        port_ret  = (self.returns @ weights)
        wins      = port_ret[port_ret > 0]
        losses    = port_ret[port_ret < 0]
        win_rate  = len(wins) / len(port_ret)
        avg_win   = float(wins.mean()) if len(wins) > 0 else 0
        avg_loss  = float(abs(losses.mean())) \
                    if len(losses) > 0 else 0.001
        kelly_res = kelly_criterion(win_rate, avg_win, avg_loss)
        results["kelly"] = kelly_res

        return results

    def print_report(self, results: dict) -> None:
        """Terminal Risk Report."""
        print("\n" + "="*58)
        print("  RISK MANAGEMENT REPORT")
        print(f"  {datetime.now().strftime('%d.%m.%Y %H:%M')}")
        print("="*58)

        # VaR Summary
        var_data = results.get("var", {})
        print("\n  VaR ÜBERSICHT (95% Konfidenz, 1 Tag)")
        print(f"  {'Ticker':<8} "
              f"{'VaR 95%':>8} "
              f"{'VaR 99%':>8} "
              f"{'CVaR 95%':>10}")
        print("  " + "-"*38)
        for ticker, v in var_data.items():
            print(
                f"  {ticker:<8}"
                f"  {v['var_95_hist']:>7.3f}%"
                f"  {v['var_99_hist']:>7.3f}%"
                f"  {v['cvar_95']:>9.3f}%"
            )

        # Portfolio VaR
        pv = results.get("portfolio_var", {})
        if pv:
            print(f"\n  PORTFOLIO VAR")
            print(f"  1-Tag VaR (95%):   "
                  f"{pv.get('var_1d_pct', 0):.3f}%  "
                  f"(${abs(pv.get('var_1d_dollar', 0)):,.2f})")
            print(f"  5-Tag VaR (95%):   "
                  f"{pv.get('var_5d_pct', 0):.3f}%  "
                  f"(${abs(pv.get('var_5d_dollar', 0)):,.2f})")
            print(f"  Portfolio Vola:    "
                  f"{pv.get('port_vol_annual', 0):.1f}%")

        # Drawdown
        dd = results.get("drawdown", {})
        if dd:
            print(f"\n  DRAWDOWN")
            print(f"  Max Drawdown:      "
                  f"{dd.get('max_drawdown', 0):.2f}%")
            print(f"  Aktueller DD:      "
                  f"{dd.get('current_drawdown', 0):.2f}%")
            print(f"  Calmar Ratio:      "
                  f"{dd.get('calmar_ratio', 0):.2f}")
            print(f"  CAGR:              "
                  f"{dd.get('cagr_pct', 0):.2f}%")

        # Korrelation
        corr = results.get("correlation", {})
        if corr:
            print(f"\n  KORRELATION")
            print(f"  Avg. Korrelation:  "
                  f"{corr.get('avg_corr', 0):.3f}")
            print(f"  Status:            "
                  f"{corr.get('status', '—')}")

        # Kelly
        kelly = results.get("kelly", {})
        if kelly:
            print(f"\n  KELLY CRITERION")
            print(f"  Win Rate:          "
                  f"{kelly.get('win_rate', 0):.1f}%")
            print(f"  Avg Win:           "
                  f"{kelly.get('avg_win', 0):.3f}%")
            print(f"  Avg Loss:          "
                  f"{kelly.get('avg_loss', 0):.3f}%")
            print(f"  Kelly (Half):      "
                  f"{kelly.get('half_kelly', 0):.1f}%")
            print(f"  Empfehlung:        "
                  f"{kelly.get('recommended_pct', 0):.1f}% "
                  f"pro Trade")
            print(f"  Verdict:           "
                  f"{kelly.get('verdict', '—')}")

        # Stress Test
        stress = results.get("stress", pd.DataFrame())
        if not stress.empty:
            print(f"\n  STRESS TESTS")
            print(f"  {'Szenario':<22} "
                  f"{'Verlust':>9} "
                  f"{'Verlust %':>10}")
            print("  " + "-"*43)
            for _, row in stress.head(5).iterrows():
                loss   = row["Portfolio Verlust"]
                pct    = row["Verlust (%)"]
                emoji  = ("🔴" if pct < -15
                           else ("🟡" if pct < -8 else "🟢"))
                print(
                    f"  {str(row['Szenario']):<22}"
                    f"  ${loss:>8,.0f}"
                    f"  {pct:>8.1f}% {emoji}"
                )

        print("\n" + "="*58)


if __name__ == "__main__":

    print("Tag 30 — Risk Management System")
    print("=" * 55)

    # --- Universe ---
    TICKERS  = ["SMCI", "PANW", "CORN", "DOC", "CRWV","UPST","RIVN","PEP","JD","HIMS","LIFT","LAC","ALB","SEDG","ENPH","PATH","TTD","MNDY","CRM","DOCU","ADBE"]
    CAPITAL  = 10_000
    PERIOD   = "2y"

    print(f"\nUniverse: {TICKERS}")
    print(f"Kapital:  ${CAPITAL:,}")

    # --- Daten ---
    prices  = load_data(TICKERS, PERIOD)
    returns = prices.pct_change().dropna()

    # --- 1. VaR Analyse für NVDA ---
    # print("\n1. VaR Analyse (NVDA)...")
    # nvda_ret = returns["NVDA"]

    # comparison = var_comparison(nvda_ret, 0.95, CAPITAL)
    # print(comparison.to_string(index=False))

    #plot_var_analysis(nvda_ret, CAPITAL, "NVDA")

    # --- 2. Portfolio VaR ---
    print("\n2. Portfolio VaR (Equal Weight)...")
    n       = len(TICKERS)
    weights = [1/n] * n

    port_var_result = portfolio_var(
        weights, prices, 0.95, CAPITAL
    )
    print(f"  1-Tag VaR (95%):    "
          f"{port_var_result['var_1d_pct']:.3f}%  "
          f"(${abs(port_var_result['var_1d_dollar']):,.2f})")
    print(f"  5-Tag VaR (95%):    "
          f"{port_var_result['var_5d_pct']:.3f}%")
    print(f"  Hist. VaR:          "
          f"{port_var_result['hist_var_pct']:.3f}%")
    print(f"  Hist. CVaR:         "
          f"{port_var_result['hist_cvar_pct']:.3f}%")
    print(f"  Portfolio Vola:     "
          f"{port_var_result['port_vol_annual']:.1f}%")

    print("\n  Marginale VaR Beiträge ($):")
    for t, mv in port_var_result["marginal_var"].items():
        print(f"    {t:<8} ${mv:+,.2f}")

    # --- 3. Drawdown Analyse ---
    print("\n3. Drawdown Analyse (Equal Weight Portfolio)...")
    equity_proxy = (
        (1 + (returns @ weights))
        .cumprod() * CAPITAL
    )
    dd = compute_drawdown(equity_proxy)

    print(f"  Max Drawdown:       {dd['max_drawdown']:.2f}%")
    print(f"  Aktueller DD:       {dd['current_drawdown']:.2f}%")
    print(f"  Calmar Ratio:       {dd['calmar_ratio']:.2f}")
    print(f"  Avg Drawdown:       {dd['avg_drawdown']:.2f}%")
    print(f"  DD Perioden:        {dd['n_dd_periods']}")
    print(f"\n  Top 5 Drawdowns:")
    print(dd["top_drawdowns"].to_string(index=False))

    plot_drawdown_analysis(equity_proxy, "Equal Weight Portfolio")

    # --- 4. Drawdown Monitor Demo ---
    print("\n4. Drawdown Monitor (Live Simulation)...")
    dd_monitor = DrawdownMonitor(
        alert_threshold_pct = 3.0,
        kill_threshold_pct  = 8.0
    )

    for i, (date, val) in enumerate(
        equity_proxy.items()
    ):
        status = dd_monitor.update(float(val), date)
        if i % 50 == 0:
            print(
                f"  {str(date)[:10]}: "
                f"${status['portfolio']:>8,.2f}  "
                f"DD: {status['dd_pct']:>+6.2f}%  "
                f"[{status['status']}]"
            )

    alerts = dd_monitor.get_alerts()
    if not alerts.empty:
        print(f"\n  {len(alerts)} Drawdown Alerts ausgelöst")

    # --- 5. Portfolio Heat ---
    print("\n5. Portfolio Heat Analyse...")

    # Synthetische Positionen
    positions_demo = pd.DataFrame([{
        "ticker":    t,
        "market_val": CAPITAL * w,
        "pnl_abs":    0,
        "qty":        10,
    } for t, w in zip(TICKERS, weights)])

    heat_analyzer = PortfolioHeat(
        positions_demo, prices
    )
    heat_df = heat_analyzer.compute_heat()

    print(heat_df[[
        "Ticker", "Gewicht (%)", "Vola (%)",
        "Beta (SPY)", "Risk Contrib %", "Heat Score"
    ]].to_string(index=False))

    total_heat = heat_analyzer.get_total_heat()
    overheated = heat_analyzer.is_overheated()
    print(f"\n  Gesamt Heat Score: {total_heat:.2f}")
    print(f"  Portfolio Status:  "
          f"{'🔴 Überkonzentriert' if overheated else '✅ Gut diversifiziert'}")

    plot_portfolio_heat(heat_df)

    # --- 6. Kelly Criterion ---
    print("\n6. Kelly Criterion...")
    port_ret  = (returns @ weights)
    wins_     = port_ret[port_ret > 0]
    losses_   = port_ret[port_ret < 0]
    win_rate_ = len(wins_) / len(port_ret)
    avg_win_  = float(wins_.mean()) if len(wins_) > 0 else 0
    avg_loss_ = float(abs(losses_.mean())) \
                if len(losses_) > 0 else 0.001

    kelly_res = kelly_criterion(win_rate_, avg_win_, avg_loss_)

    print(f"  Win Rate:          {kelly_res['win_rate']:.1f}%")
    print(f"  Avg Win:           {kelly_res['avg_win']:.3f}%")
    print(f"  Avg Loss:          {kelly_res['avg_loss']:.3f}%")
    print(f"  Odds:              {kelly_res['odds']:.2f}")
    print(f"  Full Kelly:        {kelly_res['kelly_full']:.2f}%")
    print(f"  Half Kelly:        {kelly_res['half_kelly']:.2f}%")
    print(f"  Empfehlung:        "
          f"{kelly_res['recommended_pct']:.1f}% pro Trade")
    print(f"  Verdict:           {kelly_res['verdict']}")

    # --- 7. Position Sizing ---
    print("\n7. Position Sizing Beispiel (AAPL)...")
    #aapl_price  = float(prices["AAPL"].iloc[-1])
    #stop_loss_p = round(aapl_price * 0.95, 2)
    #aapl_vol    = float(
    #    returns["AAPL"].std() * np.sqrt(252)
    #)

    #sizing = dynamic_position_size(
    #    capital    = CAPITAL,
    #    price      = aapl_price,
    #    stop_loss  = stop_loss_p,
    #    risk_pct   = 0.02,
    #    kelly_pct  = kelly_res["recommended_pct"],
    #    volatility = aapl_vol,
    #)

    #print(f"  AAPL Kurs:         ${aapl_price:.2f}")
    #print(f"  Stop Loss:         ${stop_loss_p:.2f}")
    #print(f"  Jahres-Vola:       {aapl_vol*100:.1f}%")
    #print(f"\n  Position Sizing:")
    #print(f"  Fixed Risk:        "
    #      f"{sizing['fixed_risk_shares']} Aktien")
    #print(f"  Kelly:             "
    #      f"{sizing['kelly_shares']} Aktien")
    #print(f"  Vol Adjusted:      "
    #      f"{sizing['vol_adj_shares']} Aktien")
    #print(f"  ✅ Empfohlen:      "
    #      f"{sizing['recommended_shares']} Aktien  "
    #      f"(${sizing['recommended_dollar']:,.2f}  "
    #      f"= {sizing['pct_of_capital']:.1f}% Portfolio)")
    #print(f"  Max Risiko:        "
    #      f"${sizing['risk_dollar']:,.2f}")

    # --- 8. Stress Tests ---
    print("\n8. Stress Tests...")
    stress_df = run_stress_tests(positions_demo, CAPITAL)
    print(stress_df[[
        "Szenario", "Verlust (%)",
        "Portfolio danach", "Recovery (Jahre)"
    ]].to_string(index=False))
    plot_stress_test(stress_df, CAPITAL)

    # --- 9. Korrelations-Monitor ---
    print("\n9. Korrelations-Monitor...")
    corr_result = correlation_monitor(TICKERS, PERIOD)
    print(f"  Avg Korrelation:   {corr_result['avg_corr']:.3f}")
    print(f"  Max Korrelation:   {corr_result['max_corr']:.3f}")
    print(f"  Status:            {corr_result['status']}")

    if not corr_result["high_corr_pairs"].empty:
        print(f"\n  Hochkorrelierte Paare:")
        print(corr_result["high_corr_pairs"].to_string(
            index=False
        ))

    plot_correlation_heatmap(corr_result)

    # --- 10. Vollständiges Dashboard ---
    print("\n10. Vollständiger Risk Report...")
    dashboard = RiskDashboard(TICKERS, CAPITAL, PERIOD)
    results   = dashboard.run_full_analysis(
        positions=positions_demo,
        weights=weights
    )
    dashboard.print_report(results)

    # --- Export ---
    stress_df.to_csv("day30_stress_tests.csv", index=False)
    heat_df.to_csv("day30_portfolio_heat.csv",  index=False)
    pd.DataFrame([kelly_res]).to_csv(
        "day30_kelly.csv", index=False
    )
    print("\nGespeichert: stress_tests, portfolio_heat, kelly")