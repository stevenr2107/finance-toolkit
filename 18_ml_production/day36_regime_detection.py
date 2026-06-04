"""
Day - 36 Marktregime Detection 

Warum Regime Detection?
   Jede Strategie funktioniert nur in bestimmten märkten 
   bsp: ma crossover in trned, mean reversion in range
   ML unterschiedliche Trefferquote je nach Regime 

   Bots die das Regime können, schalten direkt um 

Was wir heute bauen:
    1. Hiddwen Markov Model
    2. Technisches Regime basierend auf Indikatoren 
    3. Volatilitäts Regime - VIX und realized vol 
    4. Trend Regime - Momentum und MA-Struktur 
    5. Makro-Regime - Zinsen, Yield Curve
    6. Regime Aware Position Sizing 
    7. Bacltesting 

Die Haupt Regime:
    1. Bullenmarkt - Aufwärtsbewegung, hohe Korrelation, niedrige Volatilität -> Long 
    2. Bärenmarkt - Abwärtsbewegung, hohe Korrelation, hohe Volatilität  -> Kein Long 
    3. Seitwärtsmarkt - Keine klare Richtung, niedrige Korrelation, niedrige Volatilität -> Seletiv 
    4. Volatiler Markt - Unbeständig, hohe Korrelation, sehr hohe Volatilität -> Halbe größe
"""


import os
import json
import warnings
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from typing import Optional, Dict, List, Tuple
from enum import Enum
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import yfinance as yf
from scipy.stats import norm
from sklearn.preprocessing import StandardScaler
warnings.filterwarnings("ignore")


class Regime(Enum):
    """
    Vier Markt-Regime.
    Jedes hat andere optimale Strategien.
    """
    BULL_TREND  = "bull_trend"    # Klar aufwärts, niedrige Vol
    BEAR_TREND  = "bear_trend"    # Klar abwärts, hohe Vol
    HIGH_VOL    = "high_vol"      # Keine klare Richtung, hohe Vol
    SIDEWAYS    = "sideways"      # Seitwärts, niedrige Vol

    @property
    def position_scalar(self) -> float:
        """Wie viel des normalen Kapitals in diesem Regime."""
        scalars = {
            "bull_trend": 1.00,   # Volle Position
            "sideways":   0.50,   # Halbe Position
            "high_vol":   0.25,   # Viertel Position
            "bear_trend": 0.00,   # Kein Long
        }
        return scalars[self.value]

    @property # property macht es wie ein Attribut nutzbar, ohne () aufzurufen
    def color(self) -> str:
        colors = {
            "bull_trend": "#16a34a",
            "sideways":   "#f59e0b",
            "high_vol":   "#f97316",
            "bear_trend": "#ef4444",
        }
        return colors[self.value]

    @property
    def emoji(self) -> str:
        emojis = {
            "bull_trend": "🟢",
            "sideways":   "🟡",
            "high_vol":   "🟠",
            "bear_trend": "🔴",
        }
        return emojis[self.value]


def load_data(ticker: str, period: str = "5y") -> pd.DataFrame:
    df = yf.download(
        ticker, period=period,
        auto_adjust=True, progress=False
    )
    df.columns = df.columns.get_level_values(0)
    return df.dropna()


class TechnicalRegimeDetector:
    """
    Erkennt Regime basierend auf technischen Indikatoren.

    Logik:
        Bull Trend:  SMA50 > SMA200 + Preis > SMA50
                     + RSI 40-70 + ADX > 25
        Bear Trend:  SMA50 < SMA200 + Preis < SMA50
                     + RSI < 40
        High Vol:    ATR > 2x Durchschnitt ODER
                     Realized Vol > 30%
        Sideways:    SMA50 ≈ SMA200 (< 2% Abstand)
                     + ADX < 20

    ADX (Average Directional Index):
        Misst Trendstärke (nicht Richtung).
        ADX > 25 = starker Trend (egal ob bullish oder bearish)
        ADX < 20 = kein klarer Trend = Seitwärts

    Das Schöne: diese Regeln sind transparent.
    Du weißt GENAU warum ein Regime erkannt wurde.
    """

    def __init__(self,
                 adx_threshold:   float = 20.0,
                 vol_threshold:   float = 0.3,
                 sma_diff_pct:    float = 0.02):
        self.adx_threshold  = adx_threshold
        self.vol_threshold  = vol_threshold
        self.sma_diff_pct   = sma_diff_pct

    def _compute_adx(self,
                      df:     pd.DataFrame,
                      window: int = 14) -> pd.Series:
        """
        Average Directional Index — Trendstärke messen.

        Formel:
            +DM = High_today - High_yesterday (wenn positiv)
            -DM = Low_yesterday - Low_today   (wenn positiv)
            TR  = True Range
            +DI = EMA(+DM) / ATR × 100
            -DI = EMA(-DM) / ATR × 100
            DX  = |+DI - -DI| / (+DI + -DI) × 100
            ADX = EMA(DX, window)
        """
        high  = df["High"].squeeze()
        low   = df["Low"].squeeze()
        close = df["Close"].squeeze()

        # True Range
        prev_close = close.shift(1)
        tr = pd.concat([
            high - low,
            (high - prev_close).abs(),
            (low  - prev_close).abs()
        ], axis=1).max(axis=1)

        # Directional Movement
        plus_dm  = high.diff()
        minus_dm = -low.diff()
        plus_dm  = plus_dm.where(
            (plus_dm > minus_dm) & (plus_dm > 0), 0
        )
        minus_dm = minus_dm.where(
            (minus_dm > plus_dm) & (minus_dm > 0), 0
        )

        # Smoothed
        atr_s      = tr.ewm(span=window, adjust=False).mean()
        plus_di    = (
            plus_dm.ewm(span=window, adjust=False).mean() /
            atr_s * 100
        )
        minus_di   = (
            minus_dm.ewm(span=window, adjust=False).mean() /
            atr_s * 100
        )

        # DX und ADX
        dx = (
            (plus_di - minus_di).abs() /
            (plus_di + minus_di + 1e-8) * 100
        )
        adx = dx.ewm(span=window, adjust=False).mean()

        return adx.round(2)

    def detect(self, df: pd.DataFrame) -> pd.Series:
        """
        Berechnet Regime für jeden Tag.
        Returns: pd.Series mit Regime-Labels.
        """
        close = df["Close"].squeeze()
        high  = df["High"].squeeze()
        low   = df["Low"].squeeze()

        # Moving Averages
        sma50  = close.rolling(50).mean()
        sma200 = close.rolling(200).mean()

        # RSI
        delta    = close.diff()
        gain     = delta.clip(lower=0)
        loss     = -delta.clip(upper=0)
        avg_gain = gain.ewm(com=13, adjust=False).mean()
        avg_loss = loss.ewm(com=13, adjust=False).mean()
        rsi      = 100 - (100 / (1 + avg_gain / avg_loss))

        # ADX
        adx = self._compute_adx(df)

        # Realized Volatility (21-Tage annualisiert)
        realized_vol = (
            close.pct_change()
            .rolling(21).std() * np.sqrt(252)
        )

        # ATR ratio
        prev_close = close.shift(1)
        tr = pd.concat([
            high - low,
            (high - prev_close).abs(),
            (low  - prev_close).abs()
        ], axis=1).max(axis=1)
        atr     = tr.ewm(span=14, adjust=False).mean()
        atr_avg = atr.rolling(63).mean()
        atr_ratio = atr / (atr_avg + 1e-8)

        # SMA Spread (wie weit auseinander)
        sma_spread = (sma50 - sma200).abs() / sma200

        # Regime Labels
        regimes = pd.Series(
            Regime.SIDEWAYS, index=df.index
        )

        # Bull Trend
        bull_mask = (
            (sma50   > sma200) &
            (close   > sma50)  &
            (rsi     > 40)     &
            (rsi     < 75)     &
            (adx     > self.adx_threshold) &
            (realized_vol < self.vol_threshold)
        )
        regimes[bull_mask] = Regime.BULL_TREND

        # Bear Trend
        bear_mask = (
            (sma50   < sma200) &
            (close   < sma50)  &
            (rsi     < 55)     &
            (adx     > self.adx_threshold)
        )
        regimes[bear_mask] = Regime.BEAR_TREND

        # High Vol (überschreibt alles)
        high_vol_mask = (
            (realized_vol > 0.28) |
            (atr_ratio    > 2.0)
        )
        regimes[high_vol_mask] = Regime.HIGH_VOL

        # Sideways (wenn kein klarer Trend)
        sideways_mask = (
            (adx < 20) &
            (sma_spread < self.sma_diff_pct) &
            ~high_vol_mask
        )
        regimes[sideways_mask] = Regime.SIDEWAYS

        return regimes



class HMMRegimeDetector:
    """
    Hidden Markov Model für Regime Detection.

    Warum HMM?
        Technische Regeln sind explizit — du definierst die Grenzen.
        HMM lernt die Grenzen aus den Daten selbst.

        HMM Annahme:
            Der Markt befindet sich in einem "versteckten Zustand" (Regime).
            Wir sehen nur die Returns — nicht den Zustand direkt.
            HMM schätzt welcher Zustand am wahrscheinlichsten ist.

        Zwei-Zustand Modell:
            State 0: Low-Volatility (Bull/Sideways)
            State 1: High-Volatility (Bear/High-Risk)

        Viterbi-Algorithmus:
            Findet die wahrscheinlichste Sequenz von Zuständen.
            Das ist der Kern von HMM.

    Vereinfachte Implementierung ohne hmmlearn:
        Gaussian Mixture basierte Klassifikation.
        Ähnliches Konzept, einfachere Implementierung.



    Was HMM grundsätzlich macht: Wir sehen nur die Returns, nicht die Regime. HMM versucht zu lernen, welche Regime es gibt und wie wahrscheinlich es ist, dass wir uns in einem bestimmten Regime befinden, 
    basierend auf den beobachteten Returns. Es lernt auch, wie wahrscheinlich es ist, von einem Regime in ein anderes zu wechseln (Transition Matrix). Das ist besonders nützlich, weil es uns erlaubt, 
    die Dynamik des Marktes zu verstehen und vorherzusagen, welches Regime als nächstes kommen könnte.
    """

    def __init__(self, n_states: int = 2):
        self.n_states  = n_states
        self.params_   = None
        self.fitted_   = False

    def fit(self, returns: pd.Series) -> "HMMRegimeDetector":
        """
        Fittet das Modell auf historische Returns.

        Schritt 1: Clustering nach Return + Volatilität
        Schritt 2: Parameter (mu, sigma) pro Cluster schätzen
        Schritt 3: Transition-Matrix schätzen
        """
        r      = returns.dropna().values
        vol    = pd.Series(r).rolling(21).std().fillna(
            method="bfill"
        ).values
        # Bsp: wir haben 2.1% Return und 1.8% Vola an einem Tag, das sagt das wir wahrscheinlich in einem 
        # Bear market drin sind 

        # Features: Return + Volatilität
        X = np.column_stack([r, vol]) # 2D Array mit Returns und Volatilität als Features für das Clustering

        # K-Means findet zwei Gruppen in den Daten ohne zu wissen was bull oder bear ist 
        from sklearn.cluster import KMeans
        km = KMeans(
            n_clusters=self.n_states,
            random_state=42, n_init=10
        )
        labels = km.fit_predict(
            StandardScaler().fit_transform(X) # standardscaler (gleiche Skala)
        )

        # Sortiere States nach Volatilität
        # State 0 = niedrige Vol (Bull), State 1 = hohe Vol (Bear)
        vol_per_state = {
            s: float(np.mean(vol[labels == s]))
            for s in range(self.n_states)
        }
        sorted_states = sorted(
            vol_per_state, key=vol_per_state.get
        )
        state_map = {
            old: new for new, old in enumerate(sorted_states)
        }
        labels = np.array([state_map[l] for l in labels])

        # Parameter pro State
        self.params_ = {}
        for s in range(self.n_states):
            mask = labels == s
            self.params_[s] = {
                "mu":    float(np.mean(r[mask])), # durchschnittlicher Return 
                "sigma": float(np.std(r[mask]) + 1e-8),
                "mu_vol": float(np.mean(vol[mask])),
                "weight": float(mask.mean()), # wie oft kommt dieser State vor (relative Häufigkeit)
            }

        # Transition Matrix zählt wie oft State X zu State Y wechselt 
        #              → Bull   → Bear
        # Bull heute:   [ 0.95    0.05 ]  ← 95% bleibt Bull, 5% wechselt zu Bear
        # Bear heute:   [ 0.20    0.80 ]  ← 20% erholt sich, 80% bleibt Bear
        transitions = np.zeros((self.n_states, self.n_states))
        for i in range(len(labels) - 1):
            transitions[labels[i], labels[i+1]] += 1

        row_sums = transitions.sum(axis=1, keepdims=True)
        self.transition_ = transitions / (row_sums + 1e-8)
        self.fitted_      = True

        return self

    def predict_proba(self,
                       returns: pd.Series) -> pd.DataFrame:
        """
        Wahrscheinlichkeit für jeden State pro Tag.

        Return heute: -2.1%

        P(return | Bull): norm.pdf(-2.1%, mu=+0.04%, sigma=0.8%) = sehr klein
        P(return | Bear): norm.pdf(-2.1%, mu=-0.05%, sigma=1.8%) = größer

        → Wahrscheinlicher Bear State
        """
        if not self.fitted_:
            raise RuntimeError("Erst fit() aufrufen")

        r      = returns.values
        vol    = pd.Series(r).rolling(21).std().fillna(
            method="bfill"
        ).values

        probas = np.zeros((len(r), self.n_states)) 

        for s, params in self.params_.items():
            # Likelihood: P(return | state) × P(vol | state)
            p_ret  = norm.pdf(r,   params["mu"],    params["sigma"]) # wie wahrscheinlich ist es, diesen Return zu sehen, wenn wir in diesem State sind (z.B. Bull)
            p_vol  = norm.pdf(vol, params["mu_vol"], params["sigma"]) # wie wahrscheinlich ist es, diese Volatilität zu sehen, wenn wir in diesem State sind (z.B. Bull)
            probas[:, s] = p_ret * p_vol * params["weight"]

        # Normalisieren
        row_sums  = probas.sum(axis=1, keepdims=True)
        probas   /= (row_sums + 1e-8)# normalisieren, sodass sich alles auf 1 summiert 
        # [[0.8, 0.2],   # Tag 1: 80% Bull, 20% Bear

        df = pd.DataFrame(
            probas,
            index   = returns.index,
            columns = [f"state_{s}" for s in range(self.n_states)]
        )
        return df

    def predict(self, returns: pd.Series) -> pd.Series:
        """Wahrscheinlichster State pro Tag."""
        probas = self.predict_proba(returns)
        states = probas.idxmax(axis=1)
        return states

    def state_to_regime(self,
                         state: str) -> Regime:
        """Konvertiert HMM State → Regime Enum."""
        mapping = {
            "state_0": Regime.BULL_TREND,
            "state_1": Regime.BEAR_TREND,
        }
        return mapping.get(state, Regime.SIDEWAYS)

class VolatilityRegimeDetector:
    """
    Regime basierend auf Volatilitäts-Clustering.

    Volatilität hat ein bekanntes statistisches Phänomen:
    Sie clustert. Hohe Volatilität folgt auf hohe Volatilität.
    Niedrige folgt auf niedrige (Engle, 1982 — Nobelpreis 2003).

    GARCH-inspired Regime:
        Schätzt kurzfristige vs. langfristige Volatilität.
        Wenn kurzfristig >> langfristig → Stress-Regime
        Wenn kurzfristig ≈ langfristig  → Normales Regime
        Wenn kurzfristig << langfristig → Ruhiges Regime

    VIX Proxy (wenn keine Options-Daten):
        CBOE VIX misst Implied Volatility von SPX Options.
        VIX > 30: Angst/Stress
        VIX 15-30: Normal
        VIX < 15:  Complacency (oft zu ruhig vor Sturm)
    """

    def __init__(self,
                 short_window: int   = 5,
                 long_window:  int   = 63,
                 stress_ratio: float = 1.5,
                 calm_ratio:   float = 0.7):
        self.short_window = short_window
        self.long_window  = long_window
        self.stress_ratio = stress_ratio
        self.calm_ratio   = calm_ratio

    def detect(self, returns: pd.Series) -> pd.DataFrame:
        """
        Berechnet Volatilitäts-Regime für jeden Tag.
        auf 1 woche und 3 monate 
        """
        # Realized Volatilities
        short_vol = (
            returns.rolling(self.short_window)
            .std() * np.sqrt(252)
        )
        long_vol = (
            returns.rolling(self.long_window)
            .std() * np.sqrt(252)
        )

        # Vol Ratio
        vol_ratio = short_vol / (long_vol + 1e-8)

        # Vol Percentile (relativ zur eigenen Historie)
        # gibt an wo die aktuelle vola im vergleich zum letzten jahr steht 
        vol_pct = (
            short_vol.rolling(252)
            .rank(pct=True)
        )

        # GARCH-inspired Conditional Vol (vereinfacht)
        # EWM mit kurzem Decay als Proxy für GARCH
        cond_vol = (
            returns.pow(2) # quadrierte Returns 
            .ewm(span=22, adjust=False) # exp. gewichtet
            .mean()
            .apply(np.sqrt) * np.sqrt(252) # wurzel -> zurück zu vola
        )

        # Regime Labels
        regimes = pd.Series(
            "normal", index=returns.index # default = normal 
        )
        regimes[vol_ratio > self.stress_ratio] = "stress"
        regimes[vol_ratio < self.calm_ratio]   = "calm"
        regimes[vol_pct   > 0.90]              = "extreme_stress" # wichtig das als letztes, da sonst überschreibt 

        return pd.DataFrame({
            "short_vol":  short_vol.round(4),
            "long_vol":   long_vol.round(4),
            "vol_ratio":  vol_ratio.round(4),
            "vol_pct":    vol_pct.round(4),
            "cond_vol":   cond_vol.round(4),
            "vol_regime": regimes,
        })

    def get_vix_proxy(self,
                       prices: pd.DataFrame) -> pd.Series:
        """
        VIX Proxy aus SPY Options-implied Volatility.
        Vereinfacht: 30-Tage Realized Vol von SPY.
        In Produktion: CBOE VIX direkt laden.
        """
        try:
            vix = yf.download(
                "^VIX", period="5y",
                auto_adjust=True, progress=False
            )["Close"].squeeze()
            return vix
        except Exception:
            # Fallback: Realized Vol als VIX Proxy
            close = prices["Close"].squeeze()
            return (
                close.pct_change()
                .rolling(21).std() * np.sqrt(252) * 100
            )
        

class CombinedRegimeDetector:
    """
    Kombiniert alle Regime-Detektoren zu einem finalen Signal.

    Voting System:
        Jeder Detektor gibt eine Stimme ab.
        Finale Regime = gewichtete Mehrheit.

    Gewichtung:
        Technical:   0.40 (explizit, zuverlässig)
        HMM:         0.35 (statistisch, robust)
        Volatility:  0.25 (schnell, reaktiv)

    aus allen wird ein finales Signal gegeben 

    Regime Persistenz:
        Regime wechseln nicht täglich.
        Mindestens 3 Tage in einem Regime bevor Wechsel.
        Das verhindert unnötiges Hin- und Herschalten.
    """

    def __init__(self,
                 min_regime_days: int = 3):
        self.technical   = TechnicalRegimeDetector()
        self.hmm         = HMMRegimeDetector(n_states=2)
        self.volatility  = VolatilityRegimeDetector()
        self.min_days    = min_regime_days
        self._fitted     = False

    def fit(self, df: pd.DataFrame) -> "CombinedRegimeDetector":
        """Trainiert HMM auf historischen Daten."""
        returns     = df["Close"].squeeze().pct_change().dropna()
        self.hmm.fit(returns)
        self._fitted = True
        return self

    def detect_all(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Berechnet alle Regime-Signale und kombiniert sie.
        """
        close   = df["Close"].squeeze()
        returns = close.pct_change().dropna()

        # Technical Regime
        tech_regime = self.technical.detect(df)

        # HMM Regime
        if self._fitted:
            hmm_states = self.hmm.predict(returns)
            hmm_regime = hmm_states.map(
                self.hmm.state_to_regime
            )
        else:
            hmm_regime = pd.Series(
                Regime.SIDEWAYS, index=returns.index
            )

        # Vol Regime
        vol_df = self.volatility.detect(returns)

        # Voting System
        # Konvertiere zu numerischen Scores
        regime_scores = {
            Regime.BULL_TREND: 3,
            Regime.SIDEWAYS:   2,
            Regime.HIGH_VOL:   1,
            Regime.BEAR_TREND: 0,
        }

        # Align alle auf gemeinsamen Index
        common_idx = df.index

        tech_scores = tech_regime.reindex(
            common_idx
        ).map(regime_scores).fillna(2)

        hmm_scores  = hmm_regime.reindex(
            common_idx
        ).map(regime_scores).fillna(2)

        # Vol Score: stress → 1, extreme_stress → 0, normal → 2, calm → 3
        vol_map     = {
            "extreme_stress": 0,
            "stress":         1,
            "normal":         2,
            "calm":           3,
        }
        vol_scores  = vol_df["vol_regime"].reindex(
            common_idx
        ).map(vol_map).fillna(2)

        # Gewichtetes Voting
        combined_score = (
            tech_scores * 0.55 +
            hmm_scores  * 0.30 +
            vol_scores  * 0.15
        )

        # Score → Regime
        def score_to_regime(s: float) -> Regime:
            if s >= 2.3:
                return Regime.BULL_TREND
            elif s >= 1.6:
                return Regime.SIDEWAYS
            elif s >= 10.9:
                return Regime.HIGH_VOL
            else:
                return Regime.BEAR_TREND

        combined_regime = combined_score.apply(score_to_regime)

        # Regime Persistenz (min_days Filter)
        smoothed_regime = self._apply_persistence(
            combined_regime
        )
        # Ohne Persistence:    Bull Bear Bull Bull Bear Bull  ← zu nervös
        # Mit Persistence:     Bull Bull Bull Bull Bull Bull  ← stabil        

        # Alles zusammenbauen
        result = pd.DataFrame({
            "close":           close,
            "tech_regime":     tech_regime,
            "hmm_regime":      hmm_regime,
            "vol_regime":      vol_df["vol_regime"],
            "combined_score":  combined_score.round(3),
            "regime":          smoothed_regime,
            "position_scalar": smoothed_regime.map(
                lambda r: r.position_scalar
                if isinstance(r, Regime) else 0
            ),
            "short_vol":       vol_df["short_vol"],
            "long_vol":        vol_df["long_vol"],
        }, index=common_idx)

        return result.dropna()

    def _apply_persistence(self, regimes: pd.Series) -> pd.Series:
        smoothed          = regimes.copy()
        current           = regimes.iloc[0]
        candidate         = None
        candidate_streak  = 0

        for i in range(1, len(regimes)):
            if regimes.iloc[i] == current:
             # Altes Regime bestätigt → Kandidat zurücksetzen
                candidate        = None
                candidate_streak = 0
            else:
                # Neues Regime beobachtet
                if regimes.iloc[i] == candidate:
                    candidate_streak += 1
                else:
                    # Neuer Kandidat startet
                    candidate        = regimes.iloc[i]
                    candidate_streak = 1

                if candidate_streak >= self.min_days:
                    # Genug Bestätigung → Wechsel akzeptiert
                    current          = candidate
                    candidate        = None
                    candidate_streak = 0
                else:
                    # Noch nicht bestätigt → altes Regime halten
                    smoothed.iloc[i] = current

        return smoothed

    def get_current_regime(self,
                            df: pd.DataFrame) -> dict:
        """Gibt aktuelles Regime zurück."""
        result  = self.detect_all(df)
        latest  = result.iloc[-1]
        regime  = latest["regime"]

        if not isinstance(regime, Regime):
            regime = Regime.SIDEWAYS

        # Wie lange schon in diesem Regime?
        current_streak = 0
        for val in result["regime"][::-1]:
            if val == regime:
                current_streak += 1
            else:
                break

        return {
            "regime":          regime.value,
            "emoji":           regime.emoji,
            "position_scalar": regime.position_scalar,
            "days_in_regime":  current_streak,
            "combined_score":  float(latest["combined_score"]),
            "short_vol":       float(latest["short_vol"]),
            "long_vol":        float(latest["long_vol"]),
            "tech_regime":     str(latest["tech_regime"].value)
                               if isinstance(latest["tech_regime"], Regime)
                               else str(latest["tech_regime"]),
            "vol_regime":      str(latest["vol_regime"]),
            "timestamp":       datetime.now().isoformat(),
        }
    


def regime_filtered_backtest(df:           pd.DataFrame,
                               regime_df:   pd.DataFrame,
                               strategy:    str   = "sma",
                               capital:     float = 10_000,
                               commission:  float = 0.001) -> dict:
    """
    Backtesting mit Regime-Filter.

    Vergleicht:
        A) Strategie ohne Regime-Filter
        B) Strategie mit Regime-Filter
        C) Buy & Hold

    Regime-Filter Logik:
        Nur handeln wenn Regime = BULL_TREND
        In SIDEWAYS: halbe Position
        In HIGH_VOL und BEAR_TREND: kein Trade
    """
    close   = df["Close"].squeeze()
    returns = close.pct_change().fillna(0)

    # Strategy Signal (SMA Crossover als Default)
    sma20 = close.rolling(20).mean()
    sma50 = close.rolling(50).mean()
    raw_signal = (sma20 > sma50).astype(float).shift(1).fillna(0)

    # Position Scalar aus Regime
    scalar = regime_df["position_scalar"].reindex(
        df.index
    ).fillna(0.5)

    # Equity Curves
    # A: Ohne Filter
    strat_ret_unfiltered  = returns * raw_signal
    strat_ret_unfiltered -= raw_signal.diff().abs() * commission
    equity_unfiltered     = (
        (1 + strat_ret_unfiltered).cumprod() * capital
    )

    # B: Mit Regime-Filter
    filtered_signal       = raw_signal * scalar
    strat_ret_filtered    = returns * filtered_signal
    strat_ret_filtered   -= filtered_signal.diff().abs() * commission
    equity_filtered       = (
        (1 + strat_ret_filtered).cumprod() * capital
    )

    # C: Buy & Hold
    equity_bah = (1 + returns).cumprod() * capital

    # Metriken
    def calc_metrics(ret: pd.Series,
                      eq:  pd.Series) -> dict:
        ret_c   = ret.dropna()
        years   = len(ret_c) / 252
        total   = (eq.iloc[-1] / capital - 1) * 100
        cagr    = ((eq.iloc[-1] / capital)
                    ** (1/max(years, 0.01)) - 1) * 100
        sharpe  = (ret_c.mean() / ret_c.std() *
                   np.sqrt(252)) if ret_c.std() > 0 else 0
        roll_max = eq.cummax()
        max_dd   = ((eq - roll_max) / roll_max).min() * 100
        return {
            "total_return": round(total, 2),
            "cagr":         round(cagr, 2),
            "sharpe":       round(sharpe, 3),
            "max_dd":       round(max_dd, 2),
        }

    m_unfiltered = calc_metrics(
        strat_ret_unfiltered, equity_unfiltered
    )
    m_filtered   = calc_metrics(
        strat_ret_filtered,   equity_filtered
    )
    m_bah        = calc_metrics(returns, equity_bah)

    # Regime-Performance Breakdown
    regime_performance = {}
    for regime in Regime:
        mask = regime_df["regime"] == regime
        mask = mask.reindex(df.index).fillna(False)

        regime_ret   = strat_ret_filtered[mask]
        regime_days  = int(mask.sum())

        if regime_days > 0:
            regime_cagr = (
                (1 + regime_ret).prod()
                ** (252 / regime_days) - 1
            ) * 100
        else:
            regime_cagr = 0

        regime_performance[regime.value] = {
            "days":     regime_days,
            "pct_time": round(regime_days / len(df) * 100, 1),
            "cagr":     round(float(regime_cagr), 2),
            "scalar":   regime.position_scalar,
        }

    return {
        "equity_unfiltered":   equity_unfiltered,
        "equity_filtered":     equity_filtered,
        "equity_bah":          equity_bah,
        "metrics_unfiltered":  m_unfiltered,
        "metrics_filtered":    m_filtered,
        "metrics_bah":         m_bah,
        "regime_performance":  regime_performance,
        "regime_df":           regime_df,
    }



def plot_regime_timeline(result_df: pd.DataFrame,
                          ticker:    str) -> None:
    """
    Zeigt Kurs + Regime-Hintergrund über Zeit.
    Das ikonischste Chart für Regime Detection.
    """
    fig = make_subplots(
        rows=3, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.04,
        row_heights=[0.50, 0.25, 0.25],
        subplot_titles=[
            f"{ticker} — Kurs + Marktregime",
            "Combined Regime Score",
            "Realized Volatility"
        ]
    )

    # Kurs
    fig.add_trace(go.Scatter(
        x=result_df.index,
        y=result_df["close"].round(2),
        name="Kurs",
        line=dict(color="#1e293b", width=2),
    ), row=1, col=1)

    # Regime-Hintergrund
    colors_map = {
        Regime.BULL_TREND:  "rgba(22,163,74,0.12)",
        Regime.SIDEWAYS:    "rgba(245,158,11,0.12)",
        Regime.HIGH_VOL:    "rgba(249,115,22,0.15)",
        Regime.BEAR_TREND:  "rgba(239,68,68,0.15)",
    }

    prev_regime = None
    start_date  = None

    for date, row in result_df.iterrows():
        regime = row["regime"]
        if not isinstance(regime, Regime):
            continue

        if regime != prev_regime:
            if prev_regime is not None and start_date:
                fig.add_vrect(
                    x0=start_date, x1=date,
                    fillcolor=colors_map.get(
                        prev_regime,
                        "rgba(148,163,184,0.10)"
                    ),
                    layer="below", line_width=0,
                    row=1, col=1
                )
            start_date  = date
            prev_regime = regime

    # Letztes Regime
    if prev_regime and start_date:
        fig.add_vrect(
            x0=start_date,
            x1=result_df.index[-1],
            fillcolor=colors_map.get(
                prev_regime, "rgba(148,163,184,0.10)"
            ),
            layer="below", line_width=0,
            row=1, col=1
        )

    # Legende: eine Linie pro Regime
    for regime, color in [
        (Regime.BULL_TREND, "#16a34a"),
        (Regime.SIDEWAYS,   "#f59e0b"),
        (Regime.HIGH_VOL,   "#f97316"),
        (Regime.BEAR_TREND, "#ef4444"),
    ]:
        fig.add_trace(go.Scatter(
            x=[None], y=[None],
            mode="lines",
            name=f"{regime.emoji} {regime.value}",
            line=dict(color=color, width=8),
            opacity=0.5,
        ), row=1, col=1)

    # Combined Score
    if "combined_score" in result_df.columns:
        score_colors = result_df["combined_score"].apply(
            lambda s: (
                "#16a34a" if s >= 2.5
                else ("#f59e0b" if s >= 1.8
                      else ("#f97316" if s >= 1.0
                            else "#ef4444"))
            )
        )

        fig.add_trace(go.Scatter(
            x=result_df.index,
            y=result_df["combined_score"],
            name="Score",
            line=dict(color="#2563eb", width=1.5),
            showlegend=False
        ), row=2, col=1)

        for level, color, label in [
            (2.5, "#16a34a", "Bull 2.5"),
            (1.8, "#f59e0b", "Sideways 1.8"),
            (1.0, "#f97316", "HighVol 1.0"),
        ]:
            fig.add_hline(
                y=level, line_dash="dot",
                line_color=color, opacity=0.5,
                annotation_text=label,
                row=2, col=1
            )

    # Volatility
    if "short_vol" in result_df.columns:
        fig.add_trace(go.Scatter(
            x=result_df.index,
            y=result_df["short_vol"] * 100,
            name="Short Vol",
            line=dict(color="#ef4444", width=1.5),
            fill="tozeroy",
            fillcolor="rgba(239,68,68,0.08)",
            showlegend=False
        ), row=3, col=1)

        if "long_vol" in result_df.columns:
            fig.add_trace(go.Scatter(
                x=result_df.index,
                y=result_df["long_vol"] * 100,
                name="Long Vol",
                line=dict(color="#94a3b8", width=1.2,
                           dash="dot"),
                showlegend=False
            ), row=3, col=1)

        # 25% Vol Grenze
        fig.add_hline(
            y=25, line_dash="dash",
            line_color="#f97316", opacity=0.6,
            annotation_text="High Vol 25%",
            row=3, col=1
        )

    fig.update_layout(
        height=750,
        template="plotly_white",
        hovermode="x unified",
        legend=dict(orientation="h", y=1.02),
        margin=dict(l=0, r=0, t=60, b=0)
    )

    fig.update_yaxes(title_text="Kurs ($)",    row=1, col=1)
    fig.update_yaxes(title_text="Score",       row=2, col=1,
                     range=[0, 3.2])
    fig.update_yaxes(title_text="Vola (% pa)", row=3, col=1)

    fig.show()


def plot_regime_backtest(backtest: dict,
                          ticker:   str) -> None:
    """
    Vergleicht Strategy mit und ohne Regime-Filter.
    """
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=[
            "Equity Curves Vergleich",
            "Regime-Zeit-Verteilung",
            "CAGR nach Regime",
            "Kennzahlen Vergleich"
        ],
        vertical_spacing=0.14,
        horizontal_spacing=0.10,
        specs=[
        [{},                  {"type": "pie"}],  # Zeile 1: xy | pie
        [{"type": "xy"},      {"type": "xy"}],   # Zeile 2: xy | xy
    ]
    )

    # Equity Curves
    for equity, name, color, dash in [
        (backtest["equity_filtered"],
         "Mit Regime-Filter", "#16a34a", "solid"),
        (backtest["equity_unfiltered"],
         "Ohne Filter",       "#3b82f6", "dash"),
        (backtest["equity_bah"],
         "Buy & Hold",        "#94a3b8", "dot"),
    ]:
        fig.add_trace(go.Scatter(
            x=equity.index,
            y=equity.round(2),
            name=name,
            line=dict(color=color, width=2, dash=dash)
        ), row=1, col=1)

    # Regime Verteilung Pie
    rp          = backtest["regime_performance"]
    pie_labels  = list(rp.keys())
    pie_values  = [rp[r]["pct_time"] for r in pie_labels]
    pie_colors  = [
        Regime(r).color for r in pie_labels
    ]

    fig.add_trace(go.Pie(
        labels=[f"{Regime(r).emoji} {r}" for r in pie_labels],
        values=pie_values,
        hole=0.45,
        marker_colors=pie_colors,
        textinfo="label+percent",
        showlegend=False
    ), row=1, col=2)

    # CAGR nach Regime
    regime_cagrs = [rp[r]["cagr"] for r in pie_labels]
    r_colors     = [
        "#16a34a" if c > 0 else "#ef4444"
        for c in regime_cagrs
    ]
    fig.add_trace(go.Bar(
        x=[f"{Regime(r).emoji} {r}" for r in pie_labels],
        y=regime_cagrs,
        marker_color=r_colors,
        text=[f"{c:+.1f}%" for c in regime_cagrs],
        textposition="outside",
        showlegend=False
    ), row=2, col=1)

    fig.add_shape(
        type="line",
        x0=0, x1=1, xref="x2 domain",
        y0=0, y1=0, yref="y2",
        line=dict(color="#1e293b", width=1.5),
    )

    # Kennzahlen Vergleich
    strategies = [
        "Mit Filter", "Ohne Filter", "Buy & Hold"
    ]
    metric_keys = [
        "backtest_filtered", "backtest_unfiltered", "bah"
    ]
    metrics_data = {
        "Mit Filter":   backtest["metrics_filtered"],
        "Ohne Filter":  backtest["metrics_unfiltered"],
        "Buy & Hold":   backtest["metrics_bah"],
    }

    for metric, row_n, col_n in []:
        pass  # werden direkt unten geplottet

    sharpes = [
        metrics_data[s]["sharpe"] for s in strategies
    ]
    s_colors = [
        "#16a34a" if v == max(sharpes) else "#3b82f6"
        for v in sharpes
    ]
    fig.add_trace(go.Bar(
        x=strategies,
        y=sharpes,
        marker_color=s_colors,
        text=[f"{v:.3f}" for v in sharpes],
        textposition="outside",
        name="Sharpe",
        showlegend=False
    ), row=2, col=2)

    fig.update_layout(
        height=700,
        template="plotly_white",
        title=f"{ticker} — Regime-filtered Backtest",
        legend=dict(orientation="h", y=1.02),
        margin=dict(l=0, r=0, t=70, b=0)
    )

    fig.update_yaxes(title_text="Wert ($)",    row=1, col=1)
    fig.update_yaxes(title_text="CAGR (%)",    row=2, col=1)
    fig.update_yaxes(title_text="Sharpe",      row=2, col=2)

    fig.show()


def plot_hmm_states(returns:    pd.Series,
                    hmm:        HMMRegimeDetector,
                    ticker:     str) -> None:
    """
    HMM State-Verteilung und Return-Charakteristik.
    """
    if not hmm.fitted_:
        return

    probas = hmm.predict_proba(returns)

    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=[
            "HMM State Probabilities",
            "Return-Verteilung nach State"
        ],
        horizontal_spacing=0.10
    )

    state_colors = ["#16a34a", "#ef4444"]

    for i, col in enumerate(probas.columns):
        fig.add_trace(go.Scatter(
            x=probas.index,
            y=probas[col],
            name=f"P(State {i})",
            line=dict(
                color=state_colors[i], width=1.5
            ),
            fill="tozeroy",
            fillcolor=state_colors[i].replace(
                "#16a34a", "rgba(22,163,74,0.15)"
            ).replace(
                "#ef4444", "rgba(239,68,68,0.15)"
            ),
            stackgroup="states"
        ), row=1, col=1)

    # Return-Verteilung
    states = hmm.predict(returns)
    for i, color in enumerate(state_colors):
        mask    = states == f"state_{i}"
        r_state = returns[mask] * 100

        if not r_state.empty:
            label = (
                "Low-Vol (Bull)"
                if i == 0 else "High-Vol (Bear)"
            )
            fig.add_trace(go.Histogram(
                x=r_state,
                nbinsx=40,
                name=f"State {i}: {label}",
                marker_color=color,
                opacity=0.65,
            ), row=1, col=2)

            if hmm.params_:
                mu    = hmm.params_[i]["mu"] * 100
                sigma = hmm.params_[i]["sigma"] * 100
                fig.add_vline(
                    x=mu, line_dash="dash",
                    line_color=color,
                    annotation_text=f"μ={mu:.2f}%",
                    row=1, col=2
                )

    fig.update_layout(
        height=420,
        template="plotly_white",
        title=f"{ticker} — Hidden Markov Model States",
        barmode="overlay",
        legend=dict(orientation="h", y=1.02),
        margin=dict(l=0, r=0, t=60, b=0)
    )

    fig.update_yaxes(title_text="Wahrscheinlichkeit",
                     row=1, col=1, range=[0, 1])
    fig.update_xaxes(title_text="Return (%)", row=1, col=2)
    fig.update_yaxes(title_text="Häufigkeit", row=1, col=2)

    fig.show()


def regime_position_size(base_shares:    int,
                          regime:         Regime,
                          confidence:     float = 1.0,
                          days_in_regime: int   = 0) -> dict:
    """
    Passt Position Size basierend auf Regime an.

    Logik:
        Basis: Risk Manager gibt base_shares zurück.
        Regime Scalar: Bull=1.0, Sideways=0.5,
                       HighVol=0.25, Bear=0.0

    Confidence Bonus:
        Wenn Regime > 5 Tage stabil → +10% Bonus
        Frischer Regime-Wechsel → -10% Penalty

    Reasoning:
        In einem neuen Regime wissen wir noch nicht
        ob es anhält. Erst wenn es sich bestätigt
        erhöhen wir die Confidence.
    """
    scalar = regime.position_scalar

    # Regime-Stabilität
    if days_in_regime >= 10:
        stability_bonus = 1.10   # stabil → etwas mehr
    elif days_in_regime <= 2:
        stability_bonus = 0.90   # neu → etwas weniger
    else:
        stability_bonus = 1.00

    # Finaler Scalar
    final_scalar = scalar * stability_bonus * confidence
    final_scalar = round(min(max(final_scalar, 0), 1.0), 2)

    adjusted_shares = max(int(base_shares * final_scalar), 0)

    return {
        "base_shares":     base_shares,
        "regime":          regime.value,
        "regime_scalar":   scalar,
        "stability_bonus": stability_bonus,
        "confidence":      confidence,
        "final_scalar":    final_scalar,
        "adjusted_shares": adjusted_shares,
        "reduction_pct":   round(
            (1 - final_scalar) * 100, 1
        ),
    }


if __name__ == "__main__":

    print("Tag 36 — Regime Detection")
    print("=" * 55)

    TICKER  = "SPY"
    PERIOD  = "10y"
    CAPITAL = 10_000

    # --- Daten ---
    print(f"\n1. Daten laden: {TICKER} ({PERIOD})...")
    df      = load_data(TICKER, PERIOD)
    close   = df["Close"].squeeze()
    returns = close.pct_change().dropna()
    print(f"   {len(df)} Handelstage geladen")

    # --- Detector setup ---
    print("\n2. Regime Detector trainieren...")
    detector = CombinedRegimeDetector(min_regime_days=3)
    detector.fit(df)
    print("   HMM gefittet ✅")

    # --- Regime Detection ---
    print("\n3. Regime Detection...")
    regime_df = detector.detect_all(df)

    # Aktuelle Regime
    current = detector.get_current_regime(df)
    print(f"\n   AKTUELLES REGIME:")
    print(f"   Regime:          "
          f"{current['emoji']} {current['regime']}")
    print(f"   Position Scalar: "
          f"{current['position_scalar']:.2f}x")
    print(f"   Tage im Regime:  "
          f"{current['days_in_regime']}")
    print(f"   Score:           "
          f"{current['combined_score']:.3f}")
    print(f"   Short Vol:       "
          f"{current['short_vol']*100:.1f}%")
    print(f"   Vol-Regime:      "
          f"{current['vol_regime']}")

    # Regime-Verteilung
    print(f"\n   Regime-Verteilung (historisch):")
    regime_counts = regime_df["regime"].value_counts()
    for r, count in regime_counts.items():
        pct = count / len(regime_df) * 100
        if isinstance(r, Regime):
            print(
                f"   {r.emoji} {r.value:<12} "
                f"{count:>5} Tage  "
                f"({pct:.1f}%)"
            )

    # --- Timeline Chart ---
    print("\n4. Regime Timeline...")
    plot_regime_timeline(regime_df, TICKER)

    # --- HMM Analyse ---
    print("\n5. HMM State Analyse...")
    plot_hmm_states(returns, detector.hmm, TICKER)

    # HMM Parameter
    if detector.hmm.params_:
        print("\n   HMM Parameter:")
        for s, params in detector.hmm.params_.items():
            name = "Low-Vol" if s == 0 else "High-Vol"
            print(
                f"   State {s} ({name}):"
                f"  μ={params['mu']*100:.3f}%"
                f"  σ={params['sigma']*100:.3f}%"
                f"  Anteil={params['weight']*100:.1f}%"
            )

    # --- Regime Backtest ---
    print("\n6. Regime-filtered Backtest...")
    backtest = regime_filtered_backtest(
        df, regime_df, capital=CAPITAL
    )

    print(f"\n   {'Strategie':<20} "
          f"{'CAGR':>8} "
          f"{'Sharpe':>8} "
          f"{'Max DD':>8}")
    print("   " + "-"*46)

    for name, metrics in [
        ("Mit Regime-Filter",  backtest["metrics_filtered"]),
        ("Ohne Filter",        backtest["metrics_unfiltered"]),
        ("Buy & Hold",         backtest["metrics_bah"]),
    ]:
        print(
            f"   {name:<20}"
            f"  {metrics['cagr']:>+7.2f}%"
            f"  {metrics['sharpe']:>7.3f}"
            f"  {metrics['max_dd']:>7.2f}%"
        )

    # Regime Performance
    print(f"\n   Performance nach Regime:")
    rp = backtest["regime_performance"]
    for r_name, r_data in rp.items():
        emoji = Regime(r_name).emoji
        print(
            f"   {emoji} {r_name:<14}"
            f"  {r_data['days']:>5} Tage"
            f"  {r_data['pct_time']:>5.1f}%"
            f"  CAGR: {r_data['cagr']:>+6.1f}%"
            f"  Scalar: {r_data['scalar']:.2f}x"
        )

    plot_regime_backtest(backtest, TICKER)

    # --- Position Sizing Demo ---
    print("\n7. Regime-aware Position Sizing...")
    base_shares = 100

    for regime in Regime:
        sizing = regime_position_size(
            base_shares    = base_shares,
            regime         = regime,
            confidence     = 0.90,
            days_in_regime = 7,
        )
        print(
            f"   {regime.emoji} {regime.value:<14}"
            f"  {base_shares} → {sizing['adjusted_shares']} Aktien"
            f"  ({sizing['final_scalar']*100:.0f}%)"
        )

    # --- Multi-Ticker Regime ---
    print("\n8. Multi-Ticker Regime Scan...")
    scan_tickers = ["AAPL", "MSFT", "NVDA",
                    "JPM",  "GLD",  "SPY"]
    scan_results = []

    for t in scan_tickers:
        try:
            d  = load_data(t, "2y")
            detector_t = CombinedRegimeDetector()
            detector_t.fit(d)
            curr = detector_t.get_current_regime(d)

            scan_results.append({
                "Ticker":    t,
                "Regime":    f"{curr['emoji']} {curr['regime']}",
                "Scalar":    curr["position_scalar"],
                "Score":     curr["combined_score"],
                "Short Vol": f"{curr['short_vol']*100:.1f}%",
                "Tage":      curr["days_in_regime"],
            })
        except Exception as e:
            print(f"   {t}: Fehler — {e}")

    if scan_results:
        scan_df = pd.DataFrame(scan_results)
        print(scan_df.to_string(index=False))

    # --- Export ---
    regime_export = regime_df.copy()
    regime_export["regime_str"] = regime_export["regime"].apply(
        lambda r: r.value if isinstance(r, Regime) else str(r)
    )
    regime_export.drop(
        columns=["tech_regime", "hmm_regime", "regime"],
        errors="ignore"
    ).to_csv("day36_regimes.csv")

    with open("day36_current_regime.json", "w") as f:
        json.dump(current, f, indent=2)

    print("\n✅ Gespeichert: day36_regimes.csv, "
          "day36_current_regime.json")

    print("\n" + "="*55)
    print("INTEGRATION IN DEN BOT (Tag 29 + Tag 35):")
    print("="*55)
    print("""
  from day36_regime_detection import (
      CombinedRegimeDetector, regime_position_size
  )

  detector = CombinedRegimeDetector()
  detector.fit(df)

  # Im Bot-Loop:
  current_regime = detector.get_current_regime(df)
  regime         = Regime(current_regime["regime"])

  # Position Size anpassen:
  sizing = regime_position_size(
      base_shares    = risk_manager.calculate_shares(...),
      regime         = regime,
      days_in_regime = current_regime["days_in_regime"],
  )
  shares_to_buy = sizing["adjusted_shares"]
    """)