"""
Day 35 - ML Signal Pipeline 

Ziel:
    Das ML Modell aus Tag 23 (Random Forest / GBM)
    wird in den Live Trading Bot aus Tag 29 integriert 

    Statt SMA Crossover Signal 
    -> ML Modell gibt P(Aufstieg) zurück
    -> Bot kauft wenn P > 0.6
    -> Bot verkauft wenn P < 0.4

Architektur 
    Feature Store -> Indikatoren täglich berechnen 
    Signal Pipeline
    Signal Validator
    Bot Integration

Was anders ist als Tag 23:
    Kein Training während Live Trading 
    Modell wird täglich offline trainiert 
    Pipeline gibt nur predictions aus 
    Das nent sich Train/Serve Separation
"""


"""
***Jetzt werden Training und Serving getrennt ***
Das heißt wir trainieren den Bot offline mit neuen Daten, er berechnet 
und speichert 
-> Online wir das Modell geladen und die Prediction gemacht. Das passiert in jeder Loop 

Gründe weshalb wir es getrennt machen:
    Training dauert Minuten -> zu langsam für Live-Trading 
    Modell wir nie während Live-Trading verändert
    
    
Struktur heute:
    FeatureStore -> Indikatoren berechnen
    ModelTrainer -> Modell trainieren und validieren 
    SignalPipeline -> Online Prediction
    SignalValidator -> Plausibilitätscheck
    PipelineMonitor -> Performance des Modells tracken """


import os # os -> Datei- und Verzeichnisfunktionen
import json
import time
import joblib # Modell laden
import logging 
import warnings 
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from typing import Optional, Dict, List, Tuple
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import yfinance as yf

from sklearn.ensemble import (
    GradientBoostingClassifier,
    RandomForestClassifier
) # ML Modell
from sklearn.linear_model import LogisticRegression # zeigt welche Indikatoren wichtig sind
from sklearn.preprocessing import StandardScaler # Standardisierung
from sklearn.pipeline import Pipeline # Testdaten beeinflussen das lernen
from sklearn.metrics import (
    roc_auc_score, accuracy_score,
    classification_report
) # Evaluation
import warnings
warnings.filterwarnings("ignore")

logging.basicConfig(
    level  = logging.INFO,
    format = "%(asctime)s | %(levelname)s | %(message)s",
    handlers = [
        logging.FileHandler("ml_pipeline.log"), # Dateiausgabe
        logging.StreamHandler() # Konsolenausgabe
    ]
)
log = logging.getLogger("MLPipeline")

# Modell-Pfad
MODEL_DIR = "18_ml_production/models"
os.makedirs(MODEL_DIR, exist_ok=True) # Ordner erstellen falls nicht vorhanden

# *** Feature Store ***
@dataclass
class FeatureConfig:
    """
    Alle Features zentral definiert - 
    Training und Serving nutzen exakt dieselben 
    """
    forward_days:     int   = 5      # Vorhersage-Horizont
    threshold:        float = 0.005  # >0.5% = positiv
    seq_length:       int   = 60     # Lookback Window

    # Technische Indikatoren
    rsi_windows:      List[int] = field( # field -> erstellt Liste mit Standartwerten
        default_factory=lambda: [7, 14, 21] # jede Instanz bekommt eine neue Liste 
    )
    sma_windows:      List[int] = field(
        default_factory=lambda: [5, 10, 20, 50]
    )
    ret_windows:      List[int] = field(
        default_factory=lambda: [1, 2, 3, 5, 10, 21]
    )
    vol_windows:      List[int] = field(
        default_factory=lambda: [5, 10, 21]
    )

class FeatureStore:
    """
    Berechnet und cached Features für alle Ticker.

    Feature Store ist wichtig um - Selbe Berechnungen für Training und Serving zu bekommen 
        -> Kein copy paste zwischen zwei code stellen 
        - Ein Fehler hier zeiht sich durch den Code 
        Caching wird nicht bei jedem Signal neu berechnet 

    Feature Consistency:
        Training Feature ist nicht Serving Feature = Training/Serving Skew
        Das ist einer der häufigsten ML Prodution Bugs 
    """

    def __init__(self, config: FeatureConfig):
        self.config = config
        self._cache: Dict[str, pd.DataFrame] = {}
        self._cache_time: Dict[str, datetime] = {}
        self.cache_ttl_minutes = 60

    def _is_cache_valid(self, ticker: str) -> bool: # _ davor -> interner Gebrauch - nicht von außen aufrufen 
        """Prüft ob Cache noch frisch ist."""
        if ticker not in self._cache_time:
            return False
        age = (datetime.now() -
               self._cache_time[ticker]).seconds / 60
        return age < self.cache_ttl_minutes

    # Features werden 60 Minuten gespeichert 

    def _compute_rsi(self,
                      close: pd.Series,
                      window: int) -> pd.Series:
        """RSI Berechnung."""
        delta    = close.diff()
        gain     = delta.clip(lower=0)
        loss     = -delta.clip(upper=0)
        avg_gain = gain.ewm(com=window-1, adjust=False).mean()
        avg_loss = loss.ewm(com=window-1, adjust=False).mean()
        rs       = avg_gain / avg_loss
        return (100 - (100 / (1 + rs))) / 100  # 0-1
    
    def compute_features(self,
                         ticker:str,
                         df: pd.DataFrame) -> pd.DataFrame:
        """
        Vollständiges Feature Engineeriing 
        Signal von gestern -> Prediction für heute.
        -> kein look ahead bias 

        Feature-Kategorien:
            Returns:     Was ist passiert?
            Momentum:    In welche Richtung?
            Volatilität: Wie unsicher ist der Markt?
            Volumen:     Wer kauft/verkauft?
            Technisch:   Was sagen Indikatoren?
            Regime:      In welchem Markt sind wir?
        """

        data  = df.copy()
        close = data["Close"].squeeze()
        high  = data["High"].squeeze()
        low   = data["Low"].squeeze()
        vol   = data["Volume"].squeeze()
        open_ = data["Open"].squeeze() 

        features = pd.DataFrame(index=data.index)

        # ── Returns ──────────────────────────────────
        for w in self.config.ret_windows:
            features[f"ret_{w}d"] = close.pct_change(w)


        # ── SMAs & Distanz ──────────────────────────── also schauen wir wie weit der kurs vom SMA weg ist 
        for w in self.config.sma_windows:
            sma = close.rolling(w).mean()
            features[f"sma_{w}_dist"] = (close - sma) / sma

        # ── EMA ───────────────────────────────────────
        for span in [9, 21]:
            ema = close.ewm(span=span, adjust=False).mean()
            features[f"ema_{span}_dist"] = (close - ema) / ema

        # ── RSI ───────────────────────────────────────
        for w in self.config.rsi_windows:
            features[f"rsi_{w}"] = self._compute_rsi(close, w)

        # ── MACD ──────────────────────────────────────
        ema12    = close.ewm(span=12, adjust=False).mean()
        ema26    = close.ewm(span=26, adjust=False).mean()
        macd     = ema12 - ema26
        signal   = macd.ewm(span=9, adjust=False).mean()
        features["macd_hist"]   = (macd - signal) / close
        features["macd_signal"] = np.sign(macd - signal)

        # ── Bollinger Bands ───────────────────────────
        sma20 = close.rolling(20).mean()
        std20 = close.rolling(20).std()
        features["bb_pct_b"]   = (
            (close - (sma20 - 2*std20)) /
            (4 * std20 + 1e-8)
        )
        features["bb_width"]   = (4 * std20) / sma20

        # ── ATR (normalisiert) ────────────────────────
        # atr gibt an wie weit der kurs vom normalen handelstag weg ist
        # niedrig -> ruhige phase - hoch -> volatil
        prev_close = close.shift(1)
        tr = pd.concat([
            high - low,
            (high - prev_close).abs(),
            (low  - prev_close).abs()
        ], axis=1).max(axis=1)

        for w in [7, 14]:
            atr = tr.ewm(com=w-1, adjust=False).mean()
            features[f"atr_{w}_pct"] = atr / close

        # ── Volumen ───────────────────────────────────
        avg_vol = vol.rolling(20).mean()
        features["vol_ratio"]    = vol / (avg_vol + 1e-8)
        features["vol_ret_corr"] = (
            close.pct_change()
            .rolling(10)
            .corr(vol.pct_change())
        )

        obv = (np.sign(close.diff()) * vol).cumsum()
        obv_ma = obv.rolling(20).mean()
        features["obv_dist"] = (
            (obv - obv_ma) / (obv.rolling(20).std() + 1e-8)
        )

        # ── Preis-Patterns ────────────────────────────
        features["gap_open"]    = (
            (open_ - close.shift(1)) /
            close.shift(1)
        )
        features["candle_body"] = (
            (close - open_) / (high - low + 1e-8)
        )
        features["upper_wick"]  = (
            (high - np.maximum(close, open_)) /
            (high - low + 1e-8)
        )
        features["lower_wick"]  = (
            (np.minimum(close, open_) - low) /
            (high - low + 1e-8)
        )

        # ── Volatilität ───────────────────────────────
        for w in self.config.vol_windows:
            features[f"realized_vol_{w}"] = (
                close.pct_change()
                .rolling(w).std() * np.sqrt(252)
            )

        features["vol_ratio_5_21"] = (
            features["realized_vol_5"] /
            (features["realized_vol_21"] + 1e-8)
        )

        # ── 52-Wochen Position ────────────────────────
        features["pct_52w_high"] = close / close.rolling(252).max()
        features["pct_52w_low"]  = close / close.rolling(252).min()

        # ── Momentum Composite ────────────────────────
        features["momentum_composite"] = (
            features["ret_1d"]  * 0.10 +
            features["ret_5d"]  * 0.25 +
            features["ret_21d"] * 0.65
        )

        # ── Markt-Regime (SPY als Proxy) ──────────────
        try:
            spy = yf.download(
                "SPY", period="3y",
                auto_adjust=True, progress=False
            )["Close"].squeeze()
            spy_ret = spy.pct_change().reindex(close.index)
            features["spy_ret_5d"] = spy_ret.rolling(5).sum()
            features["spy_above_sma200"] = (
                (spy > spy.rolling(200).mean())
                .astype(int)
                .reindex(close.index)
            )
        except Exception:
            features["spy_ret_5d"]       = 0
            features["spy_above_sma200"] = 1

        # *** ── LAG ALLE FEATURES (Look-ahead Bias Prevention) ── ***
        features = features.shift(1)

        # ── Cleanup ───────────────────────────────────
        features = features.replace([np.inf, -np.inf], np.nan)

        log.info(
            f"{ticker}: {features.shape[1]} Features berechnet"
        )
        return features
    
    def get_features(self,
                      ticker: str,
                      period: str = "3y",
                      force_refresh: bool = False) -> pd.DataFrame:
        """
        Lädt Features — aus Cache wenn möglich.
        """
        if (not force_refresh and
                self._is_cache_valid(ticker)):
            log.info(f"{ticker}: Features aus Cache")
            return self._cache[ticker]

        df = yf.download(
            ticker, period=period,
            auto_adjust=True, progress=False
        )
        df.columns = df.columns.get_level_values(0)
        df = df.dropna()

        if df.empty:
            return pd.DataFrame()
        
        features = self.compute_features(ticker, df)

        self._cache[ticker]      = features
        self._cache_time[ticker] = datetime.now()

        return features
    
    def get_feature_names(self,
                           ticker: str = "AAPL") -> List[str]:
        """Gibt alle Feature-Namen zurück."""
        features = self.get_features(ticker)
        return [
            col for col in features.columns
            if col not in ["Open", "High", "Low",
                           "Close", "Volume"]
        ]
    
    # Model Trainer 

class ModelTrainer:
    """
    Trainiert ML-Modelle offline und speichert sie 

    Train/Serve Seperation:
        Der Code läuft nicht im Bot-Loop, sondern einmal täglich nachdem Markt zu ist 

    Walk-Forward Training:
        Letzte N Monate als test, um nicht zu overfitten

    Multi-Model:
        Mehrere Modelle trainieren und das beste deployen 
    """
    def __init__(self,
                 feature_store: FeatureStore,
                 config:        FeatureConfig,
                 model_dir:     str = MODEL_DIR):
        self.feature_store = feature_store
        self.config        = config
        self.model_dir     = model_dir

    def create_labels(self,
                       df:     pd.DataFrame,
                       ticker: str) -> pd.Series:
        """
        Labels: Steigt die Aktie in forward_days Tagen um mehr als threshold?
        1 = Ja , 0 = Nein
        """
        close         = yf.download(
            ticker, period="3y",
            auto_adjust=True, progress=False
        )["Close"].squeeze()

        future_return  = close.shift(-self.config.forward_days) / close - 1 
        # verschiebt die Preise 5 Tage in die Vergangenheit 
        labels         = (future_return > self.config.threshold).astype(int)
        # macht aus den Preisen True / False
        labels.name    = "label"
        return labels
    
    # Um die Daten nicht zu mischen, müssen wir sie chronologisch sortieren
    def time_series_splits(self,
                            n:          int,
                            train_pct:  float = 0.70,
                            n_splits:   int   = 4) -> List[Tuple]:
        """Walk-Forward CV Splits."""
        splits    = []
        step      = (n - int(n * train_pct)) // n_splits
        train_end = int(n * train_pct)
        # beispiel: 70% als training und 30% als test
        # Die restlichen 30% werden in n_splits aufgeteilt also 300 -> 75/Split 

        for i in range(n_splits):
            train_idx = list(range(0, train_end + i * step))
            test_start = train_end + i * step
            test_end   = min(test_start + step, n)
            test_idx   = list(range(test_start, test_end))
            if len(test_idx) > 10:
                splits.append((train_idx, test_idx)) # mind. 10 Test Tage - sonst Split zu klein 
            # Jeder Split rückt immer um diese 75 Tage vor 

        return splits
    
    def train_models(self,
                      ticker: str) -> dict:
        """
        Trainiert alle Modelle mit Walk-Forward CV.
        Gibt bestes Modell + Metriken zurück.
        """
        log.info(f"Starte Training für {ticker}...")

        # Features und Labels laden
        features = self.feature_store.get_features(
            ticker, force_refresh=True
        )
        labels   = self.create_labels(features, ticker)

        # Alignment
        feature_cols = [
            c for c in features.columns
            if c not in ["Open","High","Low",
                          "Close","Volume"]
        ]
        X = features[feature_cols].copy()
        y = labels.reindex(X.index)

        # NaN entfernen
        valid   = X.notna().all(axis=1) & y.notna() # True für Zeilen wo alle Features vorhanden sind 
        X, y    = X[valid], y[valid]

        # Letzte forward_days entfernen (kein Label)
        X = X.iloc[:-self.config.forward_days]
        y = y.iloc[:-self.config.forward_days]

        if len(X) < 200:
            log.warning(f"{ticker}: Nicht genug Daten ({len(X)})")
            return {}

        log.info(
            f"{ticker}: {len(X)} Samples, "
            f"{X.shape[1]} Features, "
            f"Label-Rate: {y.mean():.2%}"
        )

        # Modell-Kandidaten
        models = {
            "GBM": Pipeline([
                ("scaler", StandardScaler()),
                ("model",  GradientBoostingClassifier(
                    n_estimators  = 100,
                    max_depth     = 3,
                    learning_rate = 0.05,
                    subsample     = 0.8,
                    random_state  = 42
                ))
            ]),
            "RF": Pipeline([
                ("scaler", StandardScaler()),
                ("model",  RandomForestClassifier(
                    n_estimators     = 200,
                    max_depth        = 5,
                    min_samples_leaf = 50,
                    max_features     = "sqrt",
                    random_state     = 42,
                    n_jobs           = -1
                ))
            ]),
            "LR": Pipeline([
                ("scaler", StandardScaler()),
                ("model",  LogisticRegression(
                    C=0.1, max_iter=1000,
                    random_state=42
                ))
            ]),
        }

        # Walk-Forward CV
        splits = self.time_series_splits(len(X))
        cv_results = {name: [] for name in models}

        for fold, (train_idx, test_idx) in enumerate(splits):
            X_train = X.values[train_idx]
            y_train = y.values[train_idx]
            X_test  = X.values[test_idx]
            y_test  = y.values[test_idx]

            for name, model in models.items():
                model.fit(X_train, y_train)
                proba = model.predict_proba(X_test)[:, 1]
                try:
                    auc = roc_auc_score(y_test, proba)
                except Exception:
                    auc = 0.5
                cv_results[name].append(auc)

            log.info(
                f"  Fold {fold+1}/{len(splits)}: "
                f"GBM={cv_results['GBM'][-1]:.3f}  "
                f"RF={cv_results['RF'][-1]:.3f}  "
                f"LR={cv_results['LR'][-1]:.3f}"
            )

        # Bestes Modell
        avg_aucs = {
            name: np.mean(aucs)
            for name, aucs in cv_results.items()
        }
        best_name = max(avg_aucs, key=avg_aucs.get)
        best_auc  = avg_aucs[best_name]

        log.info(
            f"{ticker}: Bestes Modell: {best_name} "
            f"(AUC: {best_auc:.4f})"
        )

        # Finales Modell auf allen Daten trainieren
        # Außer letzten 10% (Out-of-Sample bleibt)
        final_train_size = int(len(X) * 0.90)
        X_final = X.values[:final_train_size]
        y_final = y.values[:final_train_size]
        X_oos   = X.values[final_train_size:]
        y_oos   = y.values[final_train_size:]

        final_model = models[best_name]
        final_model.fit(X_final, y_final)

        # OOS Validation
        oos_proba = final_model.predict_proba(X_oos)[:, 1]
        oos_pred  = (oos_proba > 0.5).astype(int)
        try:
            oos_auc = roc_auc_score(y_oos, oos_proba)
            oos_acc = accuracy_score(y_oos, oos_pred)
        except Exception:
            oos_auc = 0.5
            oos_acc = 0.5

        log.info(
            f"{ticker}: OOS AUC={oos_auc:.4f}  "
            f"ACC={oos_acc:.3f}"
        )

        # Feature Importance
        inner_model = final_model.named_steps["model"]
        if hasattr(inner_model, "feature_importances_"):
            importances = inner_model.feature_importances_
        elif hasattr(inner_model, "coef_"):
            importances = np.abs(inner_model.coef_[0])
        else:
            importances = np.zeros(X.shape[1])

        feature_importance = pd.Series(
            importances, index=feature_cols
        ).sort_values(ascending=False)

        # Modell speichern
        model_path = os.path.join(
            self.model_dir, f"{ticker}_model.joblib"
        )
        metadata_path = os.path.join(
            self.model_dir, f"{ticker}_metadata.json"
        )

        joblib.dump(final_model, model_path)

        metadata = {
            "ticker":          ticker,
            "model_type":      best_name,
            "trained_at":      datetime.now().isoformat(),
            "cv_auc":          round(best_auc, 4),
            "oos_auc":         round(oos_auc, 4),
            "oos_accuracy":    round(oos_acc, 4),
            "n_samples":       len(X),
            "n_features":      X.shape[1],
            "feature_cols":    feature_cols,
            "label_rate":      round(float(y.mean()), 4),
            "forward_days":    self.config.forward_days,
            "threshold":       self.config.threshold,
            "deployable":      oos_auc > 0.52,
        }

        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)

        log.info(
            f"{ticker}: Modell gespeichert → {model_path}"
        )
        log.info(
            f"{ticker}: Deployable: {metadata['deployable']}"
        )

        return {
            "model":              final_model,
            "metadata":           metadata,
            "cv_results":         cv_results,
            "feature_importance": feature_importance,
            "oos_proba":          oos_proba,
            "y_oos":              y_oos,
        }

    def train_universe(self,
                        tickers: List[str]) -> Dict[str, dict]:
        """Trainiert Modelle für alle Ticker im Universe."""
        results = {}
        for ticker in tickers:
            try:
                result = self.train_models(ticker)
                if result:
                    results[ticker] = result
            except Exception as e:
                log.error(f"Training Fehler {ticker}: {e}")
            time.sleep(1)

        # Summary
        deployable = [
            t for t, r in results.items()
            if r.get("metadata", {}).get("deployable", False)
        ]
        log.info(
            f"Training abgeschlossen: "
            f"{len(deployable)}/{len(tickers)} deployable"
        )

        return results


class SignalPipeline:
    """
    Online Signal Generation für den Trading Bot.

    Das ist der Serving-Teil — läuft im Bot-Loop.
    Muss schnell sein (< 1 Sekunde pro Ticker).

    Was es macht:
        1. Modell laden (gecached)
        2. Features für heute berechnen
        3. Prediction machen
        4. Signal zurückgeben

    Was es NICHT macht:
        Training (zu langsam)
        Parameter-Optimierung
        Modell-Updates (nur ModelTrainer macht das)
    """

    def __init__(self,
                 feature_store: FeatureStore,
                 model_dir:     str   = MODEL_DIR,
                 buy_threshold:  float = 0.60,
                 sell_threshold: float = 0.40):
        self.feature_store  = feature_store
        self.model_dir      = model_dir
        self.buy_threshold  = buy_threshold
        self.sell_threshold = sell_threshold

        # Model Cache — einmal laden, wiederverwenden
        self._model_cache:    Dict[str, Pipeline]  = {}
        self._metadata_cache: Dict[str, dict]      = {}
        self._load_time:      Dict[str, datetime]  = {}
        self.model_ttl_hours = 24  # Modell alle 24h neu laden

    def _load_model(self, ticker: str) -> Optional[Pipeline]:
        """Lädt Modell — aus Cache wenn möglich."""
        # Cache prüfen
        if ticker in self._load_time:
            age_h = (
                datetime.now() -
                self._load_time[ticker]
            ).seconds / 3600
            if age_h < self.model_ttl_hours:
                return self._model_cache.get(ticker)

        # Modell von Disk laden
        model_path = os.path.join(
            self.model_dir, f"{ticker}_model.joblib"
        )
        meta_path  = os.path.join(
            self.model_dir, f"{ticker}_metadata.json"
        )

        if not os.path.exists(model_path):
            log.warning(f"{ticker}: Kein Modell gefunden")
            return None

        try:
            model    = joblib.load(model_path)
            with open(meta_path) as f:
                metadata = json.load(f)

            self._model_cache[ticker]    = model
            self._metadata_cache[ticker] = metadata
            self._load_time[ticker]      = datetime.now()

            log.info(
                f"{ticker}: Modell geladen "
                f"(OOS AUC: {metadata.get('oos_auc', 0):.3f})"
            )
            return model

        except Exception as e:
            log.error(f"Modell-Ladefehler {ticker}: {e}")
            return None

    def predict(self, ticker: str) -> dict:
        """
        Hauptfunktion: gibt Signal für einen Ticker zurück.

        Returns:
            signal:      "long" | "flat" | "error"
            probability: P(Aufstieg) als float
            confidence:  "high" | "medium" | "low"
            metadata:    Model-Info
        """
        model = self._load_model(ticker)

        if model is None:
            return {
                "ticker":      ticker,
                "signal":      "flat",
                "probability": 0.5,
                "confidence":  "none",
                "reason":      "Kein Modell",
            }

        # Features für heute
        features = self.feature_store.get_features(ticker)

        if features.empty:
            return {
                "ticker":      ticker,
                "signal":      "flat",
                "probability": 0.5,
                "confidence":  "none",
                "reason":      "Keine Features",
            }

        # Feature-Spalten aus Metadata
        metadata     = self._metadata_cache.get(ticker, {})
        feature_cols = metadata.get(
            "feature_cols",
            [c for c in features.columns
             if c not in ["Open","High","Low",
                           "Close","Volume"]]
        )

        # Letzter Datenpunkt (heute)
        avail_cols = [
            c for c in feature_cols
            if c in features.columns
        ]
        latest = features[avail_cols].iloc[-1:].copy()
        latest = latest.replace([np.inf, -np.inf], np.nan)

        if latest.isna().any().any():
            n_nan = latest.isna().sum().sum()
            log.warning(
                f"{ticker}: {n_nan} NaN Features"
            )
            latest = latest.fillna(0)

        # Prediction
        try:
            proba = float(
                model.predict_proba(latest.values)[:, 1][0]
            )
        except Exception as e:
            log.error(f"Prediction Fehler {ticker}: {e}")
            return {
                "ticker":      ticker,
                "signal":      "flat",
                "probability": 0.5,
                "confidence":  "none",
                "reason":      f"Prediction Error: {e}",
            }

        # Signal
        if proba >= self.buy_threshold:
            signal     = "long"
            confidence = (
                "high"   if proba > 0.70
                else "medium"
            )
        elif proba <= self.sell_threshold:
            signal     = "flat"
            confidence = (
                "high"   if proba < 0.30
                else "medium"
            )
        else:
            signal     = "flat"
            confidence = "low"

        return {
            "ticker":      ticker,
            "signal":      signal,
            "probability": round(proba, 4),
            "confidence":  confidence,
            "buy_threshold":  self.buy_threshold,
            "sell_threshold": self.sell_threshold,
            "model_auc":   metadata.get("oos_auc", 0),
            "model_type":  metadata.get("model_type", ""),
            "trained_at":  metadata.get("trained_at", ""),
            "timestamp":   datetime.now().isoformat(),
            "reason":      "OK",
        }

    def predict_universe(self,
                          tickers: List[str]) -> Dict[str, dict]:
        """Predictions für alle Ticker."""
        signals = {}
        for ticker in tickers:
            signals[ticker] = self.predict(ticker)
            time.sleep(0.2)
        return signals


class SignalValidator:
    """
    Plausibilitäts-Check für ML Signale.

    Warum Validation?
        ML Modelle können falsch liegen.
        Manchmal aus offensichtlichen Gründen.
        Validator filtert die schlechtesten Fälle.

    Regeln:
        1. Kein Kauf wenn Modell-AUC zu niedrig (< 0.52)
        2. Kein Kauf wenn P(Aufstieg) zu niedrig (< threshold)
        3. Kein Kauf wenn Markt im Crash (SPY -5% in 5T)
        4. Kein Kauf wenn RSI extrem überkauft (> 80)
        5. Kein Kauf wenn IV zu hoch (Earnings in 2T)
        6. Kein Kauf wenn Konfidenz zu niedrig ("low")
    """

    def __init__(self,
                 min_model_auc:   float = 0.52,
                 max_rsi:         float = 78,
                 spy_crash_pct:   float = -0.05):
        self.min_model_auc  = min_model_auc
        self.max_rsi        = max_rsi
        self.spy_crash_pct  = spy_crash_pct

    def _get_current_rsi(self, ticker: str) -> float:
        """RSI des aktuellen Tages."""
        try:
            df    = yf.download(
                ticker, period="3mo",
                auto_adjust=True, progress=False
            )["Close"].squeeze()
            delta    = df.diff()
            gain     = delta.clip(lower=0)
            loss     = -delta.clip(upper=0)
            avg_gain = gain.ewm(com=13, adjust=False).mean()
            avg_loss = loss.ewm(com=13, adjust=False).mean()
            rs       = avg_gain / avg_loss
            rsi      = 100 - (100 / (1 + rs))
            return float(rsi.iloc[-1])
        except Exception:
            return 50.0

    def _get_spy_trend(self) -> float:
        """5-Tage Return von SPY."""
        try:
            spy = yf.download(
                "SPY", period="1mo",
                auto_adjust=True, progress=False
            )["Close"].squeeze()
            return float(spy.iloc[-1] / spy.iloc[-6] - 1)
        except Exception:
            return 0.0

    def validate(self,
                  signal: dict,
                  ticker: str) -> Tuple[bool, str]:
        """
        Validiert ein ML Signal.
        Returns: (is_valid, reason)
        """
        # 1. Kein Kauf-Signal
        if signal.get("signal") != "long":
            return False, "Signal nicht Long"

        # 2. Modell-Qualität
        model_auc = signal.get("model_auc", 0)
        if model_auc < self.min_model_auc:
            return False, (
                f"Modell AUC zu niedrig "
                f"({model_auc:.3f} < {self.min_model_auc})"
            )

        # 3. Konfidenz
        if signal.get("confidence") == "none":
            return False, "Kein Modell verfügbar"

        if signal.get("confidence") == "low":
            return False, "Konfidenz zu niedrig"

        # 4. RSI Check
        rsi = self._get_current_rsi(ticker)
        if rsi > self.max_rsi:
            return False, f"RSI überkauft ({rsi:.1f})"

        # 5. Markt-Crash Check
        spy_trend = self._get_spy_trend()
        if spy_trend < self.spy_crash_pct:
            return False, (
                f"Markt im Abschwung "
                f"(SPY {spy_trend:.1%} in 5T)"
            )

        return True, "OK"

    def validate_universe(self,
                           signals: Dict[str, dict]) -> Dict[str, dict]:
        """Validiert alle Signale."""
        validated = {}
        for ticker, signal in signals.items():
            is_valid, reason = self.validate(signal, ticker)
            signal["validated"]        = is_valid
            signal["validation_reason"] = reason
            validated[ticker]          = signal

        n_valid = sum(
            1 for s in validated.values() if s["validated"]
        )
        log.info(
            f"Validation: {n_valid}/{len(validated)} "
            f"Signale valide"
        )
        return validated

class PipelineMonitor:
    """
    Tracked Performance des ML-Modells über Zeit.

    Warum monitoring?
        Modelle driften — was gestern funktioniert
        funktioniert heute möglicherweise nicht mehr.
        Das nennt sich Concept Drift.

        Symptome:
            AUC sinkt über Wochen
            Prediction-Verteilung verschiebt sich
            Viele False Positives (Kaufen → Kurs fällt)

        Reaktion:
            Modell neu trainieren
            Features überprüfen
            Strategie anpassen
    """

    def __init__(self,
                 log_path: str = "ml_pipeline.log"):
        self.log_path  = log_path
        self._pred_log: List[dict] = []

    def log_prediction(self,
                        ticker:    str,
                        signal:    dict,
                        actual_return: Optional[float] = None) -> None:
        """Loggt eine Prediction für spätere Auswertung."""
        entry = {
            "timestamp":    datetime.now().isoformat(),
            "ticker":       ticker,
            "probability":  signal.get("probability", 0.5),
            "signal":       signal.get("signal", "flat"),
            "confidence":   signal.get("confidence", "low"),
            "model_auc":    signal.get("model_auc", 0),
            "validated":    signal.get("validated", False),
            "actual_return": actual_return,
        }
        self._pred_log.append(entry)

    def compute_live_metrics(self) -> dict:
        """
        Berechnet Live-Performance-Metriken.
        Nur Predictions mit bekanntem Outcome.
        """
        df = pd.DataFrame(self._pred_log)

        if df.empty or "actual_return" not in df.columns:
            return {}

        completed = df.dropna(subset=["actual_return"])

        if len(completed) < 10:
            return {"n_completed": len(completed)}

        # Binäre Labels
        y_true = (completed["actual_return"] > 0).astype(int)
        y_pred = (completed["probability"]   > 0.5).astype(int)

        try:
            auc = roc_auc_score(y_true, completed["probability"])
            acc = accuracy_score(y_true, y_pred)
        except Exception:
            auc = 0.5
            acc = 0.5

        # P&L der Signale
        long_signals = completed[completed["signal"] == "long"]
        if not long_signals.empty:
            avg_ret = float(
                long_signals["actual_return"].mean()
            ) * 100
        else:
            avg_ret = 0

        return {
            "n_predictions":  len(completed),
            "live_auc":       round(auc, 4),
            "live_accuracy":  round(acc, 4),
            "avg_return_long": round(avg_ret, 3),
            "n_long_signals": len(long_signals),
            "drift_warning":  auc < 0.51,
        }

    def get_log_df(self) -> pd.DataFrame:
        """Vollständiger Prediction Log als DataFrame."""
        return pd.DataFrame(self._pred_log)

    def save_log(self,
                  path: str = "ml_prediction_log.csv") -> None:
        """Speichert Log als CSV."""
        df = self.get_log_df()
        if not df.empty:
            df.to_csv(path, index=False)
            log.info(f"Prediction Log gespeichert: {path}")


def plot_training_results(train_results: Dict[str, dict]) -> None:
    """
    Zeigt Training-Ergebnisse für alle Ticker.
    """
    if not train_results:
        return

    tickers  = list(train_results.keys())
    cv_aucs  = [
        r["metadata"].get("cv_auc", 0)
        for r in train_results.values()
    ]
    oos_aucs = [
        r["metadata"].get("oos_auc", 0)
        for r in train_results.values()
    ]
    deployable = [
        r["metadata"].get("deployable", False)
        for r in train_results.values()
    ]

    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=[
            "CV AUC vs. OOS AUC",
            "Feature Importance (bestes Modell)"
        ],
        horizontal_spacing=0.12
    )

    # AUC Vergleich
    colors = [
        "#16a34a" if d else "#ef4444"
        for d in deployable
    ]

    fig.add_trace(go.Bar(
        x=tickers,
        y=cv_aucs,
        name="CV AUC",
        marker_color="#3b82f6",
        opacity=0.7,
    ), row=1, col=1)

    fig.add_trace(go.Bar(
        x=tickers,
        y=oos_aucs,
        name="OOS AUC",
        marker_color=colors,
        opacity=0.9,
    ), row=1, col=1)

    fig.add_hline(
        y=0.52, line_dash="dash",
        line_color="#ef4444",
        annotation_text="Deploy Grenze 0.52",
        row=1, col=1
    )
    fig.add_hline(
        y=0.50, line_dash="dot",
        line_color="#94a3b8",
        annotation_text="Random 0.50",
        row=1, col=1
    )

    # Feature Importance des ersten Tickers
    first_result = list(train_results.values())[0]
    fi           = first_result.get("feature_importance")

    if fi is not None and len(fi) > 0:
        top_fi      = fi.head(15)
        fi_colors   = [
            "#16a34a" if i < 5
            else ("#f59e0b" if i < 10
                  else "#ef4444")
            for i in range(len(top_fi))
        ]
        fig.add_trace(go.Bar(
            x=top_fi.values[::-1],
            y=top_fi.index[::-1],
            orientation="h",
            marker_color=fi_colors[::-1],
            name="Importance",
            showlegend=False
        ), row=1, col=2)

    fig.update_layout(
        height=480,
        template="plotly_white",
        title="ML Training Ergebnisse",
        barmode="group",
        legend=dict(orientation="h", y=1.02),
        margin=dict(l=0, r=0, t=60, b=0)
    )

    fig.update_yaxes(title_text="AUC",       row=1, col=1,
                     range=[0.45, 0.75])
    fig.update_xaxes(title_text="Importance", row=1, col=2)

    fig.show()


def plot_signal_dashboard(signals: Dict[str, dict],
                           validated: Dict[str, dict]) -> None:
    """
    Live Signal Dashboard für alle Ticker.
    """
    tickers   = list(signals.keys())
    probas    = [signals[t].get("probability", 0.5)
                 for t in tickers]
    signal_labels = [signals[t].get("signal", "flat")
                     for t in tickers]
    is_valid  = [validated[t].get("validated", False)
                 for t in tickers]

    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=[
            "P(Aufstieg) pro Ticker",
            "Signal + Validation Status"
        ],
        horizontal_spacing=0.12
    )

    # Probability Bars
    proba_colors = [
        "#16a34a" if p >= 0.60
        else ("#ef4444" if p <= 0.40 else "#f59e0b")
        for p in probas
    ]

    fig.add_trace(go.Bar(
        x=tickers,
        y=probas,
        marker_color=proba_colors,
        text=[f"{p:.2f}" for p in probas],
        textposition="outside",
        name="P(Aufstieg)",
        showlegend=False
    ), row=1, col=1)

    for level, color, label in [
        (0.60, "#16a34a", "Buy 0.60"),
        (0.50, "#94a3b8", "Neutral"),
        (0.40, "#ef4444", "Sell 0.40"),
    ]:
        fig.add_hline(
            y=level, line_dash="dot",
            line_color=color, opacity=0.6,
            annotation_text=label,
            row=1, col=1
        )

    # Status Bars
    status_colors = []
    status_vals   = []
    for t in tickers:
        sig = signals[t].get("signal", "flat")
        val = validated[t].get("validated", False)
        if sig == "long" and val:
            status_colors.append("#16a34a")
            status_vals.append(1)
        elif sig == "long" and not val:
            status_colors.append("#f59e0b")
            status_vals.append(0.5)
        else:
            status_colors.append("#94a3b8")
            status_vals.append(0)

    fig.add_trace(go.Bar(
        x=tickers,
        y=status_vals,
        marker_color=status_colors,
        text=[
            ("✅ LONG" if v == 1
             else ("⚠ BLOCKED" if v == 0.5
                   else "⬜ FLAT"))
            for v in status_vals
        ],
        textposition="inside",
        textfont=dict(color="white", size=11),
        name="Status",
        showlegend=False
    ), row=1, col=2)

    fig.update_layout(
        height=420,
        template="plotly_white",
        title=(
            f"ML Signal Dashboard — "
            f"{datetime.now().strftime('%d.%m.%Y %H:%M')}"
        ),
        margin=dict(l=0, r=0, t=60, b=0)
    )

    fig.update_yaxes(title_text="Wahrscheinlichkeit",
                     row=1, col=1, range=[0, 1.1])
    fig.update_yaxes(title_text="Status",
                     row=1, col=2, showticklabels=False)

    fig.show()


if __name__ == "__main__":

    print("Tag 35 — ML Signal Pipeline")
    print("=" * 55)

    UNIVERSE = ["AAPL", "MSFT", "NVDA",
                "GOOGL", "SPY"]

    config = FeatureConfig(
        forward_days = 5,
        threshold    = 0.005,
    )

    feature_store = FeatureStore(config)
    trainer       = ModelTrainer(feature_store, config)
    pipeline      = SignalPipeline(
        feature_store,
        buy_threshold  = 0.60,
        sell_threshold = 0.40
    )
    validator     = SignalValidator(
        min_model_auc = 0.52,
        max_rsi       = 78,
    )
    monitor       = PipelineMonitor()

    # --- TRAINING PHASE ---
    print("\n1. Training Phase (offline)...")
    print("   Trainiere Modelle für Universe...")
    train_results = trainer.train_universe(UNIVERSE)

    print(f"\n   Training Summary:")
    print(f"   {'Ticker':<8} {'Type':<6} "
          f"{'CV AUC':>8} {'OOS AUC':>9} {'Deploy':>8}")
    print("   " + "-"*42)

    for ticker, result in train_results.items():
        meta = result.get("metadata", {})
        dep  = "✅" if meta.get("deployable") else "❌"
        print(
            f"   {ticker:<8}"
            f"  {meta.get('model_type','?'):<6}"
            f"  {meta.get('cv_auc', 0):>8.4f}"
            f"  {meta.get('oos_auc', 0):>9.4f}"
            f"  {dep:>8}"
        )

    plot_training_results(train_results)

    # --- SERVING PHASE ---
    print("\n2. Serving Phase (online)...")
    print("   Generiere Signale für heute...")
    signals   = pipeline.predict_universe(UNIVERSE)

    print(f"\n   {'Ticker':<8} {'Signal':<6} "
          f"{'P(Up)':>7} {'Conf':<8} {'AUC':>6}")
    print("   " + "-"*40)

    for ticker, sig in signals.items():
        arrow = "🟢" if sig["signal"] == "long" else "⬜"
        print(
            f"   {ticker:<8}"
            f"  {arrow} {sig['signal']:<4}"
            f"  {sig['probability']:>7.4f}"
            f"  {sig['confidence']:<8}"
            f"  {sig.get('model_auc', 0):>5.3f}"
        )

    # --- VALIDATION ---
    print("\n3. Signal Validation...")
    validated = validator.validate_universe(signals)

    print(f"\n   {'Ticker':<8} {'Valid':>6} {'Grund':<30}")
    print("   " + "-"*46)

    for ticker, sig in validated.items():
        status = "✅" if sig["validated"] else "❌"
        reason = sig.get("validation_reason", "")[:28]
        print(
            f"   {ticker:<8}"
            f"  {status:>6}"
            f"  {reason:<30}"
        )

    # --- MONITORING ---
    print("\n4. Monitoring...")
    for ticker, sig in validated.items():
        monitor.log_prediction(ticker, sig)

    live_metrics = monitor.compute_live_metrics()
    if live_metrics:
        print(f"   Predictions geloggt: "
              f"{live_metrics.get('n_predictions', 0)}")
    else:
        print("   Noch keine completed predictions "
              "(braucht actual_returns)")

    # --- DASHBOARD ---
    print("\n5. Signal Dashboard...")
    plot_signal_dashboard(signals, validated)

    # --- INTEGRATION DEMO ---
    print("\n6. Bot Integration Demo...")
    print("   Wie der Bot die Signale nutzt:\n")

    for ticker, sig in validated.items():
        if sig["validated"] and sig["signal"] == "long":
            prob = sig["probability"]
            conf = sig["confidence"]
            print(
                f"   → KAUFEN {ticker}  "
                f"P={prob:.2f}  "
                f"Konfidenz={conf}"
            )
        else:
            reason = sig.get("validation_reason", "flat")
            print(
                f"   → SKIP   {ticker}  "
                f"({reason})"
            )

    # --- EXPORT ---
    monitor.save_log("day35_ml_predictions.csv")

    summary = {
        "trained":   datetime.now().isoformat(),
        "universe":  UNIVERSE,
        "models":    {
            t: {
                "type":      r["metadata"].get("model_type"),
                "oos_auc":   r["metadata"].get("oos_auc"),
                "deployable": r["metadata"].get("deployable"),
            }
            for t, r in train_results.items()
        },
        "signals": {
            t: {
                "signal":    s["signal"],
                "probability": s["probability"],
                "validated":  validated[t]["validated"],
            }
            for t, s in signals.items()
        }
    }

    with open("day35_pipeline_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print("\n✅ Gespeichert:")
    print("   day35_ml_predictions.csv")
    print("   day35_pipeline_summary.json")
    print(f"   {MODEL_DIR}/ — alle trainierten Modelle")

    # --- NÄCHSTER SCHRITT ---
    print("\n" + "="*55)
    print("NÄCHSTER SCHRITT: Bot Integration")
    print("="*55)
    print(
        "  In day29_trading_bot_v1.py:\n"
        "  Ersetze SignalEngine.compute_signal() durch:\n\n"
        "  from day35_signal_pipeline import SignalPipeline\n"
        "  pipeline = SignalPipeline(feature_store)\n"
        "  signal   = pipeline.predict(ticker)\n\n"
        "  Statt SMA Crossover → ML Probability als Signal."
    )