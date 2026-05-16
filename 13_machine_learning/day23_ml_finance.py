"""
Day 23 — Machine Learning für Finance

Der Unterschied zu klassischem Backtesting:
    Backtesting: Du definierst die Regel (RSI < 30 → kaufen) und entscheidest was wichtig ist
    ML:          Das Modell analysiert tausende Datenpunkte und findet die Regeln selbst
                 Kombiniert signale die man manuell nicht erkannt hätte

Klingt besser, aber ist schwieriger :
    1. Look-ahead Bias
        - Lösung: TimeSeriesSplit (nur vergangenheit -> Zukunft )

    2. Finanzdaten sind laut → Signal/Noise Ratio extrem niedrig
        - Aktienkurse sind extrem laut 
        - Wetter vorhersagen 70% accuracy möglich 
        - Aktienkurse vorhersagen > 55% sehr gut

    3. Regime Changes → was gestern funktioniert hat heute nicht
        - Was in einem Regime funktioniert, muss nicht im nächsten funktionieren 

    4. Overfitting ist massiv — Markt ist nicht statisch
        - Das modell passt sich perfekt an historische Daten an 
        -> In der Realität katastrophal

Was du heute baust:
    1. Feature Engineering  —> aus Kursdaten Signale machen (RSI,Returns, ATR...)
    2. Labels erstellen     —> was wollen wir vorhersagen?
    3. Time Series Cross-Validation —> korrekte Validation ohne look ahead
    4. Mehrere Modelle trainieren und vergleichen → Random Forest, Gradient Boosting, Logistic Regression
    5. Feature Importance   —> was lernt das Modell wirklich?
    6. Backtesting des ML-Signals → Funktioniert das auch in der Realität?

Ziel:
    Vorhersage: Steigt die Aktie in den nächsten 5 Tagen?
    *** Output: Wahrscheinlichkeit (nicht nur ja/nein) ***

WICHTIG:
    Dieses Script ist kein Handelssystem.
    Es ist ein Lernwerkzeug um zu verstehen wie ML auf Finanzdaten
    angewendet wird — und warum es so schwierig ist.
"""

# pip install scikit-learn

import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.ensemble import (
    RandomForestClassifier, # non-linear, robust, langsam
    GradientBoostingClassifier # non-linear, genaust, am langsamsten
)
""" RandomForest: 
# Idee: nicht ein Entscheidungsbaum sondern viele
# Jeder Baum sieht zufälligen Ausschnitt der Daten

Baum 1: "Wenn RSI < 40 UND Volumen hoch → kaufen"
Baum 2: "Wenn MACD positiv UND Momentum > 0 → kaufen"
Baum 3: "Wenn Preis über 50-SMA UND RSI steigt → kaufen"
...
500 Bäume → Mehrheitsvotum → finale Vorhersage

Vorteil: Overfitting-resistant, Feature Importance kostenlos
"""

"""Gradient
# Idee: Bäume lernen sequentiell voneinander
# Jeder neue Baum korrigiert Fehler des vorherigen

Baum 1:  erste Vorhersage (grob)
Baum 2:  lernt wo Baum 1 falsch lag
Baum 3:  lernt wo Baum 1+2 falsch lagen
...
→ Gradient = mathematische Richtung der Verbesserung

Vorteil: oft genauer als Random Forest
Nachteil: langsamer, sensitiver für Overfitting
"""
# Drei modelle, da man will das verschiedene voneinander lernen
from sklearn.linear_model import LogisticRegression # am schnellsten
# Berechnet wsk 0-1
# P(Aktie steigt) = σ(β₀ + β₁×RSI + β₂×MACD + ...)
# σ = Sigmoid Funktion → quetscht alles auf 0-1:
# σ(x) = 1 / (1 + e^(-x))
"""
σ(-5) = 0.007  → fast sicher NICHT steigen
σ(0)  = 0.500  → unentschieden
σ(+5) = 0.993  → fast sicher steigen
"""
from sklearn.preprocessing import StandardScaler 
# Standardisierung, da jeder indikator andere skalen hat
from sklearn.metrics import (
    accuracy_score, # richtige vorhersagen / alle vorhersagen
    classification_report, # Precision -> von allen signalen wieviele richtig?
    # Recall -> Von allen anstiegen, wieviele wurden gefunden? -> Harmonisches mittel aus beiden
    roc_auc_score, # AUC = Area Under teh curve 
    # Misst wie gut das Modell zwischen steigt/fällt trennt
    #AUC = 0.6 -> schwaches Signal aber für finance gut -> 0.5 = Coinflip
    # AUC = 0.7 gutes Modell
    roc_curve, # Gibt die Punkte für den ROC-Plot zurück
    confusion_matrix # zeigt wo das modell welche fehler macht 
)

"""confusion matrix 
TN = True Negative  (richtig: fällt vorhergesagt, fällt wirklich)
TP = True Positive  (richtig: steigt vorhergesagt, steigt wirklich)
FP = False Positive (falsch: kaufen → verliert Geld) -> Minimieren **
FN = False Negative (falsch: nicht kaufen → Gewinn verpasst)
"""
from sklearn.pipeline import Pipeline # Testdaten beeinflussen das lernen
# Nur auf Trainings-Daten fitten
import warnings
warnings.filterwarnings("ignore")


def load_data(ticker: str, period: str = "10y") -> pd.DataFrame:
    df = yf.download(ticker, period=period,
                     auto_adjust=True, progress=False)
    df.columns = df.columns.get_level_values(0)
    return df.dropna()
# 10 Jahre da ML mehr daten braucht 
"""
2015-2017: ruhiger Bull Market
2018:      Volatiler Einbruch
2020:      COVID Crash + Recovery
2021:      Meme Stock Euphorie
2022:      Zinserhöhungs-Bear Market
2023-2024: AI-getriebener Bull Market
"""

def compute_rsi(close: pd.Series, window: int = 14) -> pd.Series:
    delta    = close.diff()
    gain     = delta.clip(lower=0)
    loss     = -delta.clip(upper=0)
    avg_gain = gain.ewm(com=window - 1, adjust=False).mean()
    avg_loss = loss.ewm(com=window - 1, adjust=False).mean()
    rs       = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))


def create_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Feature Engineering — das Herzstück jedes ML-Modells.

    Was sind gute Features für Finanzdaten?
        1. Technische Indikatoren (RSI, MACD, BB)
        2. Preis-Returns über verschiedene Fenster
        3. Volatilitäts-Features
        4. Volumen-Features
        5. Cross-Asset Features (Markt-Kontext)

    WICHTIG: Alle Features müssen LAGGED sein.
        Wir verwenden nur Information die GESTERN verfügbar war.
        Sonst: Look-ahead Bias → Modell schummelt.

    Normalisierung:
        Rohe Preise funktionieren nicht als Features.
        AAPL bei $150 und MSFT bei $350 sind nicht vergleichbar.
        Returns und normalisierte Metriken schon.
    """
    data  = df.copy()
    close = data["Close"].squeeze() # serie aus spalten machen 
    high  = data["High"].squeeze()
    low   = data["Low"].squeeze()
    vol   = data["Volume"].squeeze()
    open_ = data["Open"].squeeze()

    # --- Returns über verschiedene Fenster kumulativ [1, 2, 3, 5, 10, 21]---
    # pct_change[5] = vergleich close 5 mit 0 -> Modell soll lernen welches am besten funktioneirt
    for window in [1, 2, 3, 5, 10, 21]:
        data[f"ret_{window}d"] = close.pct_change(window) 

    # --- Moving Average Features ---
    for window in [5, 10, 20, 50, 200]:
        sma = close.rolling(window).mean()
        data[f"sma_{window}_dist"] = (close - sma) / sma
        # Abstand vom MA in % — normalisiert und vergleichbar Meta und Hims nicht vergleichbar

    # --- EMA Features ---
    for span in [9, 21, 50]:
        ema = close.ewm(span=span, adjust=False).mean()
        data[f"ema_{span}_dist"] = (close - ema) / ema

    # --- RSI ---
    for window in [7, 14, 21]: # 3 vercshiedene RSI
        data[f"rsi_{window}"] = compute_rsi(close, window) / 100
        # Normalisiert auf 0-1 -> Standardscaler funktioniert besser wenn alles ähnlich ist

    # --- MACD ---
    ema12     = close.ewm(span=12, adjust=False).mean()
    ema26     = close.ewm(span=26, adjust=False).mean()
    macd_line = ema12 - ema26
    signal    = macd_line.ewm(span=9, adjust=False).mean()

    data["macd_hist"]    = (macd_line - signal) / close # stärke signal
    data["macd_signal"]  = np.sign(macd_line - signal) # nur +1, 0, -1 - richtung signal
    # np.sign gibt nur 3 werte zurück -> +1, 0, -1

    # --- Bollinger Bands ---
    for window in [20]:
        sma  = close.rolling(window).mean()
        std  = close.rolling(window).std()
        # bb_pct_b ueigt wo im Bollinger band der preis ist 
        data[f"bb_pct_b_{window}"] = (close - (sma - 2*std)) / \
                                      (4 * std + 1e-8) # + 1e-8 um division durch 0 zu vermeiden
        data[f"bb_width_{window}"] = (4 * std) / sma
        # wie breit ist der rand des Bollinger bands

    # --- Average True Range (normalisiert) ---
    prev_close = close.shift(1)
    tr = pd.concat([ # erweitert die Tagesrange um die Overnight Gaps 
        high - low,
        (high - prev_close).abs(), # Gap nach oben 
        (low  - prev_close).abs() # Gap nach unten
    ], axis=1).max(axis=1) # Maximum der drei 
    """
    high - low:              $7
    high - prev_close:       $5   (Gap nach oben)
    low  - prev_close:       $2   (Gap nach unten)

    True Range = max(7, 5, 2) = $7
    """

    for window in [7, 14]: # 7,14 um in 2 zeitfenstern zu schauen wie die Vola ist
        atr = tr.ewm(com=window-1, adjust=False).mean()
        data[f"atr_{window}_pct"] = atr / close # Auf prozent 

    # --- Volumen Features ---
    avg_vol = vol.rolling(20).mean()
    data["vol_ratio"]    = vol / (avg_vol + 1e-8) # heutiges Volumen hoch/niedrig zu Durchschnitt?
    data["vol_ret_corr"] = (
        close.pct_change()
        .rolling(10)
        .corr(vol.pct_change())
    )  # Korrelation zwischen Returns und Volumen über 10 Tage

    # OBV normalisiert ( On balance Volumen ) 
    obv = (np.sign(close.diff()) * vol).cumsum()
    obv_mean = obv.rolling(20).mean()
    data["obv_dist"] = (obv - obv_mean) / (obv.rolling(20).std() + 1e-8)

    # --- Preis-Patterns ---
    data["gap_open"]      = (open_ - close.shift(1)) / close.shift(1)
    data["candle_body"]   = (close - open_) / (high - low + 1e-8) 
    # wie groß ist die Candle im vergleich zur Gesamtrange
    # +1 = perfekt grüne kerze, 0 = doji (unentschlossen)
    data["upper_wick"]    = (high - np.maximum(close, open_)) / \
                             (high - low + 1e-8)# high - close 
    data["lower_wick"]    = (np.minimum(close, open_) - low) / \
                             (high - low + 1e-8)

    # --- Volatilität ---
    for window in [5, 10, 21]:
        data[f"realized_vol_{window}"] = (
            close.pct_change()
            .rolling(window).std() * np.sqrt(252)
        )

    # Volatility Ratio: kurzfristig vs. langfristig
    # >1 -> vola steigt, <1 -> vola sinkt
    data["vol_ratio_5_21"] = (
        data["realized_vol_5"] /
        (data["realized_vol_21"] + 1e-8)
    )

    # --- Momentum Composite ---
    # momentum verschieden gewichtet -> langfristiges Momentum hoch gewichtet 
    data["momentum_composite"] = (
        data["ret_1d"]  * 0.10 +
        data["ret_5d"]  * 0.25 +
        data["ret_21d"] * 0.65
    )

    # --- 52-Wochen Position ---
    data["pct_52w_high"] = close / close.rolling(252).max()
    data["pct_52w_low"]  = close / close.rolling(252).min()
    """
    pct_52w_high = 1.0  → Allzeithoch (52W) → starker Trend
    pct_52w_high = 0.8  → 20% unter 52W-Hoch → Schwäche
    pct_52w_low  = 1.0  → Allzeittief (52W) → Trendumkehr möglich?
    """

    return data


def create_labels(df:         pd.DataFrame,
                  forward_days: int   = 5,
                  threshold:    float = 0.0) -> pd.Series:
    """
    Label-Erstellung — was wollen wir vorhersagen?

    Binary Classification:
        1 → Kurs steigt in den nächsten forward_days Tagen
        0 → Kurs fällt oder bleibt gleich

    Threshold:
        0.0  → Jede positive Bewegung = 1
        0.01 → Nur wenn Kurs um mehr als 1% steigt = 1

    Warum forward_days = 5?
        1 Tag: zu viel Rauschen, zu niedrige Accuracy
        21 Tage: zu wenig Datenpunkte, regime shifts
        5 Tage: guter Kompromiss

    WICHTIG: Labels für die letzten forward_days fehlen.
    Diese Zeilen werden beim Training ausgeschlossen.
    """
    close          = df["Close"].squeeze()
    future_return  = close.shift(-forward_days) / close - 1 # ! -forward_days
    label          = (future_return > threshold).astype(int) # True = 1, False = 0

    """
    Tag 1: close = 100, shift(-5) = close[Tag 6] = 107
    → future_return = 107/100 - 1 = +7%
    → label = 1 (steigt)

    Tag 2: close = 102, shift(-5) = close[Tag 7] = 99
    → future_return = 99/102 - 1 = -3%
    → label = 0 (fällt)


    Tag 2516: shift(-5) → Tag 2521 → existiert nicht → NaN
    Tag 2517: NaN
    ...
    Tag 2520: NaN

    → diese Zeilen werden beim Training ausgeschlossen
    → kein Look-ahead Bias möglich ✅
    """
    return label


def time_series_cv_splits(n_samples:    int, # hnadelstage
                           n_splits:     int   = 5,
                           train_size:   float = 0.7) -> list: # 70% der Tage sind fürs traineiren
    # Man nimmt nur 70% der Daten und testet diese auf die 30% rest 
    """
    Time Series Cross-Validation — KEIN random split.

    Warum kein random split?
        Wenn Train: [Tag 1, 3, 5, 7] und Test: [Tag 2, 4, 6, 8]
        dann hat das Modell Zukunfts-Information im Training. wegen der 8
        Das ist Look-ahead Bias → Modell schummelt → Backtesting lügt.

    Purged Walk-Forward:
        Train: Tag 1 bis 700  → Test: Tag 701 bis 800
        Train: Tag 1 bis 800  → Test: Tag 801 bis 900
        Train: Tag 1 bis 900  → Test: Tag 901 bis 1000
        ...

    Expanding Window:
        Train wächst mit der Zeit — wie in der Realität.
        Je mehr Daten desto besser das Modell.

    Embargo:
        Optional: Lücke zwischen Train und Test.
        Verhindert Datenlecks durch überlappende Features.
    """
    splits    = []
    step_size = (n_samples - int(n_samples * train_size)) // n_splits 
    # Alle handelstage werden auf 5 Sessions aufgeteilt die seperat trained werden

    train_end = int(n_samples * train_size) # bei x. Tag hört man auf 

    for i in range(n_splits):
        train_idx = list(range(0, train_end + i * step_size))
        test_start = train_end + i * step_size
        test_end   = min(test_start + step_size, n_samples) # falls letzter block über datensatz herausgeht

        test_idx   = list(range(test_start, test_end))
        """
        i=0: Train [0→1750]    Test [1750→1900]
        i=1: Train [0→1900]    Test [1900→2050]  ← Train wächst!
        i=2: Train [0→2050]    Test [2050→2200]
        i=3: Train [0→2200]    Test [2200→2350]
        i=4: Train [0→2350]    Test [2350→2500]
        """

        if len(test_idx) > 10: # wenn ein split weniger 10 tage hat → ignoriere
            splits.append((train_idx, test_idx))

    return splits


def prepare_dataset(df:           pd.DataFrame,
                    forward_days: int   = 5,
                    threshold:    float = 0.0) -> tuple:
    """
    Erstellt finales Feature-Set für ML.

    Schritte:
        1. Features berechnen
        2. Labels erstellen
        3. Lag anwenden (alle Features um 1 Tag verzögert)
        4. NaN Zeilen entfernen
        5. Feature Matrix X und Label Vektor y trennen

    Feature Lag:
        WICHTIG: Alle Features werden um 1 Tag gelaggt.
        Heute verwenden wir gestrige Werte als Input.
        Das verhindert Look-ahead Bias automatisch.
    """
    # Features berechnen
    featured_df = create_features(df) # unsere 40 technischen features
    labels      = create_labels(df, forward_days, threshold) # 0 oder 1 pro tag

    # Feature Spalten auswählen
    feature_cols = [
        col for col in featured_df.columns
        if col not in ["Open", "High", "Low", "Close",
                       "Volume", "Dividends", "Stock Splits"]
    ] # alle wichtigen Daten
    """
    Close = $185.43   ← absolute Zahl, nicht vergleichbar
    ret_1d = +0.023   ← normalisierte Veränderung ✅

    Das Modell würde sonst lernen:
    "AAPL bei $185 → kaufen"
    Aber das ist kein generalisierbares Muster!
    """

    X = featured_df[feature_cols].copy()
    y = labels.copy()

    # Lag: Features von gestern → Vorhersage für morgen da wir nicht am abend traden
    X = X.shift(1)

    # NaN entfernen
    valid_idx = X.notna().all(axis=1) & y.notna()
    X = X[valid_idx]
    y = y[valid_idx]

    """
          RSI   MACD   SMA200   →  all() valid?
    Tag1  NaN   0.23   0.012       False ❌ (RSI noch nicht berechenbar)
    Tag2  0.45  0.31   NaN         False ❌ (SMA200 braucht 200 Tage)
    Tag200 0.52 0.28   0.034       True  ✅
    """

    # Unendliche Werte entfernen
    X = X.replace([np.inf, -np.inf], np.nan)
    # Wenn avg_vol = 0 (kein Handel über 20 Tage):
    # vol / 0.0000001 = 10.000.000  → nicht inf, aber extrem
    valid_idx2 = X.notna().all(axis=1)
    X = X[valid_idx2]
    y = y[valid_idx2]

    print(f"Dataset: {len(X)} Datenpunkte, "
          f"{X.shape[1]} Features")
    print(f"Label-Verteilung: "
          f"{y.mean()*100:.1f}% positiv")
    
    """
    y = [1, 0, 1, 1, 0, 1, 0, 0, 1, 1]
    y.mean() = 6/10 = 0.6 → "60% positiv"
    """

    return X, y, feature_cols

def get_models() -> dict:
    """
    Drei Modelle mit unterschiedlichen Stärken.

    Logistic Regression:
        Baseline — linear, interpretierbar.
        Wenn LR gut funktioniert ist das Signal stark.
        Wenn LR versagt aber RF gut → nicht-lineares Signal.

    Random Forest:
        Ensemble aus Decision Trees.
        Gut bei nicht-linearen Zusammenhängen.
        Feature Importance verfügbar.
        Weniger Overfitting als einzelner Tree.

    Gradient Boosting:
        Sequentielle Trees — jeder korrigiert den Fehler des letzten.
        Meist genauester Klassifizierer für tabellarische Daten.
        Anfälliger für Overfitting → weniger Trees als RF.
    """
    return {
        "Logistic Regression": Pipeline([
            ("scaler", StandardScaler()),
            ("model",  LogisticRegression(
                C=0.1, # regulierungsstärke - ich traue den daten nicht zu sehr 
                max_iter=1000,
                random_state=42
            ))
        ]),

        "Random Forest": Pipeline([
            ("scaler", StandardScaler()),
            ("model",  RandomForestClassifier(
                n_estimators  = 200,    # 200 Bäume 
                max_depth     = 5,       # max 5 Ebenen tief - sonst overfit - generalisiert bei 5 
                min_samples_leaf = 50,   # Jede Regel muss auf 50 Datenpunkten basieren (Blatt)
                max_features  = "sqrt",  # 40 Features gesamt -> Jeder Baum sieht zufällig 6 Features
                random_state  = 42,
                n_jobs        = -1      # Alle CPU Kerne nutzen, um Dauer zu reduzieren
                # 200 Bäume auf 8 Kerne -> 25 Bäume pro CPU
            ))
        ]),

        "Gradient Boosting": Pipeline([
            ("scaler", StandardScaler()),
            ("model",  GradientBoostingClassifier(
                n_estimators   = 100, # nur 100 statt 200 - mehr bäume -> overfitten 
                # Baum 1 → Fehler → Baum 2 korrigiert → Fehler → Baum 3...
                max_depth      = 3,   # noch flacher 
                learning_rate  = 0.05,    # langsam lernen
                # learning_rate = 0.05 → jeder Baum zählt nur 5% → langsam aber robust
                subsample      = 0.8,     # 80% der Daten pro Baum
                # Jeder Baum sieht zufällig 80% der Trainingsdaten -> gut gegen ausreißer
                random_state   = 42
            ))
        ]),
    }

    """
            Kleines C (0.1) → starke Regularisierung → Gewichte werden klein gehalten
            Großes C (10)   → schwache Regularisierung → Modell kann overfitten

            C=0.1 bedeutet:
            "Ich traue den Trainingsdaten nicht zu sehr —
            Finanzdaten sind laut, also vorsichtig sein
        
            Ohne Regularisierung:
            Feature "ret_1d" bekommt Gewicht +5.3
            → Modell verlässt sich blind auf 1-Tages-Return
            → funktioniert im Training, nicht im Test

            Mit C=0.1:
            Feature "ret_1d" bekommt maximal ~±0.5
            → kein Feature dominiert alleine
            → robusteres Modell
        """


def train_and_evaluate(X:          pd.DataFrame,
                        y:          pd.Series,
                        models:     dict,
                        n_splits:   int = 5) -> dict:
    """
    Trainiert alle Modelle mit Time Series Cross-Validation.

    Für jedes Modell und jeden CV-Split:
        1. Train auf historischen Daten
        2. Predict auf unseen Test-Daten
        3. Metrics berechnen

    Metrics:
        Accuracy:  % korrekte Vorhersagen
        AUC-ROC:   Wie gut trennt das Modell? (0.5=random, 1=perfekt)
        Precision: Von allen vorhergesagten Käufen — wie viele waren richtig?
        Recall:    Von allen echten Aufstiegen — wie viele hat es gefunden?

    Warum AUC-ROC wichtiger als Accuracy?
        Wenn 55% der Tage positiv sind kann ein Modell
        das immer "kaufen" sagt 55% Accuracy haben.
        AUC-ROC misst die Ranking-Qualität — robuster.
    """
    splits  = time_series_cv_splits(len(X), n_splits)
    results = {name: {
        "accuracy":    [], #pro Fold eine Zahle
        "auc_roc":     [], # ^
        "precision":   [], # ^ 
        "recall":      [], # ^
        "predictions": [], # Alle Vorhersagen gesammelt 
        "probabilities": [], # alle Wahrscheinlichkeiten
        "test_indices":  [], # Welche Datenpunkte wurden getestet?
    } for name in models} # gleich wie results = {} for name in...

    X_arr = X.values # pd.DataFrame -> np.ndarray
    y_arr = y.values # pd.Series -> np.ndarray

    print(f"\nTraining mit {n_splits}-Fold Time Series CV...")

    for fold, (train_idx, test_idx) in enumerate(splits): # Daten aufteilen - gibt wert und index gleichzeitig
        X_train = X_arr[train_idx]
        y_train = y_arr[train_idx]
        X_test  = X_arr[test_idx]
        y_test  = y_arr[test_idx]
        """
        splits = [(train1, test1), (train2, test2), ...]
        enumerate → (0, (train1, test1)), (1, (train2, test2)), ...
        fold=0, fold=1, ...  ← für den Print
        """

        for name, model in models.items():
            # Training
            model.fit(X_train, y_train)

            # Prediction
            y_pred  = model.predict(X_test)
            y_proba = model.predict_proba(X_test)[:, 1] # 2 Spalten zurück 
            """
            [[0.7, 0.3],   ← 70% Wahrscheinlichkeit fällt, 30% steigt
            [0.2, 0.8],   ← 20% fällt, 80% steigt
            [0.6, 0.4]]   ← 60% fällt, 40% steigt

            [:, 1]  → nur die zweite Spalte (Wahrscheinlichkeit steigt)
            → [0.3, 0.8, 0.4] -> probability score 
            """

            # Metrics
            acc     = accuracy_score(y_test, y_pred)
            try:
                auc = roc_auc_score(y_test, y_proba) # wirft fehler wenn alle 0 oder 1 sind
                # Passiert wenn zufällig extreme periode erwischt wird 
            except Exception:
                auc = 0.5

            report = classification_report(
                y_test, y_pred,
                output_dict=True, zero_division=0 # dict statt string 
            )
            prec   = report.get("1", {}).get("precision", 0) # wenn 1 nicht drin 0 -> verhindert keyerror 
            rec    = report.get("1", {}).get("recall", 0) 

            # append fügt Element hinzu [1, 2, [3, 4]]
            # extend -> [1, 2, 3, 4]
            results[name]["accuracy"].append(acc)
            results[name]["auc_roc"].append(auc)
            results[name]["precision"].append(prec)
            results[name]["recall"].append(rec)
            results[name]["predictions"].extend(y_pred.tolist())
            results[name]["probabilities"].extend(y_proba.tolist())
            results[name]["test_indices"].extend(test_idx)

        print(f"  Fold {fold+1}/{n_splits}: "
              f"Samples {len(train_idx)} Train / "
              f"{len(test_idx)} Test")

    # Summary
    summary = {}
    for name in models:
        r = results[name]
        summary[name] = {
            "Accuracy":   round(np.mean(r["accuracy"]),  4),
            "AUC-ROC":    round(np.mean(r["auc_roc"]),   4),
            "Precision":  round(np.mean(r["precision"]), 4),
            "Recall":     round(np.mean(r["recall"]),    4),
            "Acc Std":    round(np.std(r["accuracy"]),   4), # std der accuracy 
            # Um zu wissen welches Modell Akkurat ist, hilft die std stark um abweichungen zu verringern 
        }

    return results, summary

def get_feature_importance(model_pipeline: Pipeline,
                            feature_names:  list,
                            model_name:     str) -> pd.DataFrame:
    """
    Feature Importance — was hat das Modell gelernt?

    *** Modell hat 62% AUC aber man weiß nicht warum ***

    Das ist die wichtigste Analyse nach dem Training.
    Wenn das Modell Features wichtig findet die keinen
    wirtschaftlichen Sinn ergeben → Overfitting.
    Wenn es sinnvolle Features findet → echter Signal.

    Random Forest / GBM: feature_importances_
        Basiert auf wie oft und wie gut ein Feature
        die Daten aufteilt. Stabil und interpretierbar.

    Logistic Regression: coef_
        Linearer Koeffizient. Positiv → bullish Signal.
        Negativ → bearish Signal.
    """
    model = model_pipeline.named_steps["model"] # Modell aus pipeline extrahieren
    # named_steps -> hat 2 Spalten: Scaler und Modell und wir holen model
    # feature importance ist eine eigenschaft des modells 

    if hasattr(model, "feature_importances_"): # hasattr prüft ob ein Objekt ein Attribut hat
        # Gibt true oder false aus 
        importances = model.feature_importances_ # schaut wie oft ein Feature benutzt wird (RSI in x knoten)
    elif hasattr(model, "coef_"):
        importances = np.abs(model.coef_[0]) 
        """
        coef RSI      = +0.8   → RSI hoch → bullish  (wichtig)
        coef MACD     = -0.6   → MACD neg → bearish  (wichtig)
        coef ret_21d  = +0.1   → schwacher Effekt    (weniger wichtig)

        abs: [0.8, 0.6, 0.1] → Ranking: RSI > MACD > ret_21d
        """
    else:
        return pd.DataFrame()

    importance_df = pd.DataFrame({ # sortiertes dataframe ausgeben
        "Feature":    feature_names,
        "Importance": importances,
    }).sort_values("Importance", ascending=False) # größte importance oben

    return importance_df.reset_index(drop=True)

"""
✅ Sinnvolle Top-Features:
   RSI_14         → Überkauft/Überverkauft → klassisches Signal
   realized_vol_5 → Volatilität → bekannter Prediktor
   momentum_composite → Momentum-Effekt bekannt seit Jegadeesh 1993

❌ Verdächtige Top-Features:
   ret_2d         → zu spezifisch, warum genau 2 Tage?
   candle_body    → sehr rauschig
   gap_open       → sehr selten → wenig statistische Basis

Wenn verdächtige Features dominieren → Overfitting → Modell neu tunen
"""


def ml_backtest(X:        pd.DataFrame,
                y:        pd.Series,
                model:    Pipeline,
                results:  dict,
                df:       pd.DataFrame,
                name:     str,
                capital:  float = 10_000) -> dict:
    """
    Backtested das ML-Signal auf echten Preisdaten.

    Logic:
        Wenn Modell P(aufstieg) > 0.55 → Long
        Sonst → Cash

    Das ist anders als Backtesting.py:
        Wir nutzen die Out-of-Sample Predictions

        *** normal 20 tage backtest -> Signal ***
        *** jetzt: 4 Tage Training und dann Test ***
        *** Das modell predicted auf Daten, die es nie gesehen hat ***

    Threshold 0.55 statt 0.50:
        Wir kaufen nur wenn Modell relativ sicher ist.
        Reduziert Trades, erhöht Qualität pro Trade.
    """
    close = df["Close"].squeeze()

    # Out-of-Sample Predictions zusammensetzen
    indices = results[name]["test_indices"]
    proba   = results[name]["probabilities"]

    """
    Fold 1: indices [1750→1900], probas [0.52, 0.61, 0.48, ...]
    Fold 2: indices [1900→2050], probas [0.71, 0.43, 0.55, ...]
    ...
    → nach extend: indices [1750, 1751, ..., 2499]
                   proba   [0.52, 0.61, ..., 0.63]
    """

    if not indices:
        return {}

    # Sortieren nach Index da folds reihenfloge auseinander bringen kann
    sorted_pairs = sorted(zip(indices, proba)) # zip fügt liste zu tupel
    indices      = [p[0] for p in sorted_pairs] # erstes objekt in tupel
    proba        = [p[1] for p in sorted_pairs]

    # ***  Signal: buy wenn P > threshold *** hier kann man spielen 
    threshold  = 0.55 # 0.5 -> mehr transaktionskosten, 0.7 -> weniger trades
    signal_df  = pd.DataFrame({
        "date":  X.index[indices],
        "proba": proba,
        "signal": [1 if p > threshold else 0 for p in proba]
    }).set_index("date")

    # Returns berechnen
    price_aligned = close.loc[signal_df.index] # Nur der preis für tage mit predictions
    market_returns = price_aligned.pct_change().fillna(0)
    strategy_returns = market_returns * signal_df["signal"].shift(1).fillna(0) # look ahead bais schutz
    # und gibt an ob man investiert ist oder nicht 

    # Equity Curves
    equity_ml  = (1 + strategy_returns).cumprod() * capital
    equity_bah = (1 + market_returns).cumprod() * capital # vergleich 

    # Metriken
    years = len(equity_ml) / 252
    if years > 0:
        cagr_ml  = (equity_ml.iloc[-1]/capital)**(1/years) - 1
        cagr_bah = (equity_bah.iloc[-1]/capital)**(1/years) - 1
    else:
        cagr_ml = cagr_bah = 0 # compund annual growth rate

    """
    equity_ml startet bei 10.000$
    equity_ml endet   bei 14.200$
    Jahre = 750 / 252 = 2.98

    CAGR = (14.200/10.000)^(1/2.98) - 1 = 0.132 = 13.2% p.a.
    """

    sr_ml    = (strategy_returns.mean() /
                strategy_returns.std() * np.sqrt(252)) \
               if strategy_returns.std() > 0 else 0 # sharpe ratio 
    """
    Sharpe Ratio — kennst du aus den anderen Backtests. 
    Wichtig: bei invested_pct = 30% (70% in Cash) drückt das den Sharpe, 
    weil Cash-Tage 0% Return haben aber die Standardabweichung verringern.
    """

    trade_pct = signal_df["signal"].mean() * 100 # Wie oft ist das Modell investiert in pct

    return {
        "equity_ml":    equity_ml,
        "equity_bah":   equity_bah,
        "cagr_ml":      round(cagr_ml  * 100, 2),
        "cagr_bah":     round(cagr_bah * 100, 2),
        "sharpe_ml":    round(sr_ml, 3),
        "invested_pct": round(trade_pct, 1),
        "signal_df":    signal_df,
    }
# schlägt modell bah oder nicht ? Wenn ja -> echtes alpha 

def plot_model_comparison(summary: dict) -> None:
    """
    Vergleicht alle Modelle nach Accuracy, AUC-ROC, Precision, Recall.
    """
    metrics = ["Accuracy", "AUC-ROC", "Precision", "Recall"]
    colors  = ["#2563eb", "#16a34a", "#f59e0b"]
    models  = list(summary.keys())

    fig = make_subplots(
        rows=1, cols=4,
        subplot_titles=metrics,
        horizontal_spacing=0.08
    )

    for col_idx, metric in enumerate(metrics, 1):
        values = [summary[m][metric] for m in models]
        bar_colors = [
            "#16a34a" if v == max(values) else "#3b82f6"
            for v in values # AUC ROC Wert maximum wird grün gefärbt 
        ]

        fig.add_trace(go.Bar(
            x=models,
            y=values,
            marker_color=bar_colors,
            text=[f"{v:.3f}" for v in values],
            textposition="outside",
            showlegend=False
        ), row=1, col=col_idx)

        # Baseline: 0.5 
        # Random Guess Baseline 
        baseline = 0.5 if metric == "AUC-ROC" else None
        if baseline:
            fig.add_hline(
                y=baseline,
                line_dash="dot",
                line_color="#ef4444",
                opacity=0.6,
                row=1, col=col_idx
            )

    fig.update_layout(
        height=420,
        template="plotly_white",
        title="Model Comparison — Time Series Cross-Validation",
        margin=dict(l=0, r=0, t=60, b=0)
    )

    fig.update_yaxes(range=[0, 1])
    fig.show()


def plot_feature_importance(importance_df: pd.DataFrame,
                             model_name:    str,
                             top_n:         int = 20) -> None:
    """
    Top N Feature Importance als horizontaler Bar Chart.
    Hohes AUC -> Overfitting 
    Features 1-7:   grün  ← Top-Tier, wichtigste
    Features 8-13:  orange ← mittlere Bedeutung
    Features 14-20: rot   ← untere Bedeutung, fast irrelevant   
    """
    top = importance_df.head(top_n)

    colors = [
        "#16a34a" if i < top_n // 3
        else ("#f59e0b" if i < 2 * top_n // 3
              else "#ef4444")
        for i in range(len(top))
    ]

    fig = go.Figure(go.Bar(
        x=top["Importance"],
        y=top["Feature"],
        orientation="h",
        marker_color=colors[::-1], # zählt von hinten 
        opacity=0.85,
        text=[f"{v:.4f}" for v in top["Importance"]],
        textposition="outside",
    ))

    fig.update_layout(
        title=f"Feature Importance — {model_name} (Top {top_n})",
        xaxis_title="Importance",
        yaxis=dict(autorange="reversed"),
        template="plotly_white",
        height=max(400, top_n * 22),
        margin=dict(l=0, r=60, t=50, b=0)
    )

    fig.show()


def plot_roc_curves(X:       pd.DataFrame,
                    y:       pd.Series,
                    models:  dict,
                    results: dict) -> None:
    """
    ROC Kurven für alle Modelle.

    AUC-ROC Interpretation:
        0.50 = Random Guess (diagonal Linie)
        0.55 = Leichter Edge
        0.60 = Guter Edge
        0.70 = Sehr guter Edge (in Finance sehr selten)

    Für Finanzmodelle ist AUC > 0.55 bereits bedeutsam.

    Y-Achse: True Positive Rate  = Von allen echten Anstiegen gefunden
    X-Achse: False Positive Rate = Von allen echten Rückgängen falsch als Anstieg markiert

    Ideales Modell:   geht sofort nach oben links → AUC = 1.0
    Zufälliges Modell: Diagonale von (0,0) bis (1,1) → AUC = 0.5
    """
    fig = go.Figure()

    colors = {
        "Logistic Regression": "#3b82f6",
        "Random Forest":       "#16a34a",
        "Gradient Boosting":   "#f59e0b",
    }

    for name in models:
        r = results[name]
        if not r["probabilities"]:
            continue

        indices = r["test_indices"]
        proba   = r["probabilities"]
        sorted_ = sorted(zip(indices, proba))
        y_true  = y.values[[p[0] for p in sorted_]]
        y_score = [p[1] for p in sorted_]

        try:
            fpr, tpr, _ = roc_curve(y_true, y_score)
            auc         = roc_auc_score(y_true, y_score)
            color       = colors.get(name, "#8b5cf6")

            fig.add_trace(go.Scatter(
                x=fpr, y=tpr,
                name=f"{name} (AUC={auc:.3f})",
                line=dict(color=color, width=2)
            ))
        except Exception:
            pass

    # Diagonal (Random)
    fig.add_trace(go.Scatter(
        x=[0, 1], y=[0, 1],
        name="Random (AUC=0.50)",
        line=dict(color="#94a3b8", width=1.5, dash="dot")
    ))

    fig.update_layout(
        title="ROC Kurven — Out-of-Sample",
        xaxis_title="False Positive Rate",
        yaxis_title="True Positive Rate",
        template="plotly_white",
        height=500,
        legend=dict(orientation="h", y=1.02),
        margin=dict(l=0, r=0, t=60, b=0)
    )

    fig.show()


def plot_ml_backtest(backtest_result: dict,
                     model_name:      str,
                     ticker:          str) -> None:
    """
    ML Trading Signal Backtest vs. Buy & Hold.
    """
    if not backtest_result:
        return

    eq_ml  = backtest_result["equity_ml"]
    eq_bah = backtest_result["equity_bah"]

    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        row_heights=[0.65, 0.35],
        subplot_titles=[
            f"{ticker} — {model_name} Signal vs. Buy & Hold",
            "ML Signal (1=Long, 0=Cash)"
        ]
    )

    # Equity Curves
    fig.add_trace(go.Scatter(
        x=eq_ml.index,
        y=eq_ml.round(2),
        name=f"ML ({model_name})",
        line=dict(color="#2563eb", width=2)
    ), row=1, col=1)

    fig.add_trace(go.Scatter(
        x=eq_bah.index,
        y=eq_bah.round(2),
        name="Buy & Hold",
        line=dict(color="#94a3b8", width=1.5, dash="dot")
    ), row=1, col=1)

    # Signal
    signal_df = backtest_result["signal_df"]
    signal_colors = [
        "#16a34a" if s == 1 else "#ef4444"
        for s in signal_df["signal"]
    ]

    fig.add_trace(go.Bar(
        x=signal_df.index,
        y=signal_df["signal"],
        name="Signal",
        marker_color=signal_colors,
        opacity=0.7,
        showlegend=False
    ), row=2, col=1)

    fig.update_layout(
        height=600,
        template="plotly_white",
        hovermode="x unified",
        legend=dict(orientation="h", y=1.02),
        margin=dict(l=0, r=0, t=50, b=0)
    )

    fig.update_yaxes(title_text="Kapital ($)", row=1, col=1)
    fig.update_yaxes(title_text="Signal",      row=2, col=1,
                     range=[-0.1, 1.1])

    fig.show()


def plot_probability_distribution(results: dict,
                                   y:       pd.Series,
                                   best_model: str) -> None:
    """
    Verteilung der vorhergesagten Wahrscheinlichkeiten.
    Gut kalibriertes Modell: P=0.7 → 70% der Zeit korrekt?
    """
    r       = results[best_model]
    indices = r["test_indices"]
    proba   = r["probabilities"]

    sorted_ = sorted(zip(indices, proba))
    y_true  = y.values[[p[0] for p in sorted_]]
    y_score = np.array([p[1] for p in sorted_])

    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=[
            "P(aufstieg) Verteilung nach echter Klasse",
            "Calibration — Predicted vs. Actual"
        ],
        horizontal_spacing=0.12
    )

    # Verteilung nach Klasse
    for cls, color, label in [
        (1, "#16a34a", "Tatsächlich gestiegen"),
        (0, "#ef4444", "Tatsächlich gefallen")
    ]:
        mask = y_true == cls
        fig.add_trace(go.Histogram(
            x=y_score[mask],
            nbinsx=30,
            name=label,
            marker_color=color,
            opacity=0.6,
        ), row=1, col=1)

    # Calibration Plot
    n_bins    = 10
    bin_edges = np.linspace(0, 1, n_bins + 1)
    bin_means = []
    bin_fracs = []

    for i in range(n_bins):
        lo, hi = bin_edges[i], bin_edges[i+1]
        mask   = (y_score >= lo) & (y_score < hi)
        if mask.sum() > 5:
            bin_means.append(y_score[mask].mean())
            bin_fracs.append(y_true[mask].mean())

    fig.add_trace(go.Scatter(
        x=bin_means,
        y=bin_fracs,
        mode="markers+lines",
        name="Modell",
        line=dict(color="#2563eb", width=2),
        marker=dict(size=8)
    ), row=1, col=2)

    # Perfekte Kalibrierung
    fig.add_trace(go.Scatter(
        x=[0, 1], y=[0, 1],
        name="Perfekt kalibriert",
        line=dict(color="#94a3b8", width=1, dash="dot")
    ), row=1, col=2)

    fig.update_layout(
        height=420,
        template="plotly_white",
        title=f"Modell Analyse — {best_model}",
        legend=dict(orientation="h", y=1.05),
        margin=dict(l=0, r=0, t=60, b=0)
    )

    fig.update_xaxes(title_text="P(aufstieg)",       row=1, col=1)
    fig.update_xaxes(title_text="Predicted P",       row=1, col=2)
    fig.update_yaxes(title_text="Häufigkeit",        row=1, col=1)
    fig.update_yaxes(title_text="Actual Fraction",   row=1, col=2)

    fig.show()

"""
plot_model_comparison()
→ "Welches Modell gewinnt in welcher Metrik?"

plot_feature_importance()
→ "Was hat das Modell gelernt? Macht es Sinn?"

plot_roc_curves()
→ "Wie gut trennt es bei jedem Threshold?"

plot_ml_backtest()
→ "Hätte es Geld verdient?"

plot_probability_distribution()
→ "Kann man den Wahrscheinlichkeiten trauen?"

Zusammen → vollständige ML-Modell-Evaluation
wie sie in einem professionellen Quant-Team erwartet wird 🎯
"""



if __name__ == "__main__":

    TICKER   = "DOC"
    CAPITAL  = 10_000
    FWD_DAYS = 5

    print("Tag 23 — Machine Learning für Finance")
    print("=" * 55)

    # --- Daten laden ---
    df = load_data(TICKER, "10y")
    print(f"Daten: {len(df)} Handelstage ({TICKER})")

    # --- Feature Engineering ---
    print("\n1. Feature Engineering...")
    X, y, feature_cols = prepare_dataset(
        df,
        forward_days = FWD_DAYS,
        threshold    = 0.0
    )

    # --- Modelle ---
    models = get_models()

    # --- Training ---
    print("\n2. Time Series Cross-Validation Training...")
    results, summary = train_and_evaluate(
        X, y, models, n_splits=5
    )

    # --- Summary ---
    print("\n3. Model Performance Summary")
    print("=" * 55)
    print(f"  {'Modell':<25} {'Acc':>8} "
          f"{'AUC':>8} {'Prec':>8} {'Rec':>8}")
    print("  " + "-"*53)
    for name, metrics in summary.items():
        print(f"  {name:<25}"
              f"  {metrics['Accuracy']:>7.3f}"
              f"  {metrics['AUC-ROC']:>7.3f}"
              f"  {metrics['Precision']:>7.3f}"
              f"  {metrics['Recall']:>7.3f}")

    print(f"\n  Hinweis: AUC > 0.55 = echter Edge in Finance")
    print(f"  Baseline: {y.mean():.3f} "
          f"(Häufigkeit positiver Tage)")

    plot_model_comparison(summary)

    # --- ROC Kurven ---
    print("\n4. ROC Kurven plotten...")
    plot_roc_curves(X, y, models, results)

    # --- Bestes Modell ---
    best_model_name = max(
        summary, key=lambda m: summary[m]["AUC-ROC"]
    )
    print(f"\n5. Bestes Modell: {best_model_name}")
    print(f"   AUC-ROC: {summary[best_model_name]['AUC-ROC']:.4f}")

    # --- Feature Importance ---
    print("\n6. Feature Importance...")
    best_model = models[best_model_name]

    # Auf vollem Datensatz trainieren für Feature Importance
    X_arr = X.values
    y_arr = y.values
    train_size = int(len(X_arr) * 0.8)
    best_model.fit(X_arr[:train_size], y_arr[:train_size])

    importance_df = get_feature_importance(
        best_model, feature_cols, best_model_name
    )

    if not importance_df.empty:
        print("\n  Top 10 Features:")
        for _, row in importance_df.head(10).iterrows():
            print(f"    {row['Feature']:<30} "
                  f"{row['Importance']:.4f}")

        plot_feature_importance(
            importance_df, best_model_name, top_n=20
        )

    # --- Probability Distribution ---
    print("\n7. Wahrscheinlichkeitsverteilung analysieren...")
    plot_probability_distribution(
        results, y, best_model_name
    )

    # --- ML Backtest ---
    print("\n8. ML Signal Backtest...")
    backtest_result = ml_backtest(
        X, y, best_model,
        results, df,
        best_model_name, CAPITAL
    )

    if backtest_result:
        print(f"\n  ML Strategy ({best_model_name}):")
        print(f"  CAGR:           "
              f"{backtest_result['cagr_ml']:+.2f}%")
        print(f"  Buy & Hold CAGR:"
              f"{backtest_result['cagr_bah']:+.2f}%")
        print(f"  Sharpe (ML):    "
              f"{backtest_result['sharpe_ml']:.3f}")
        print(f"  Zeit investiert:"
              f"{backtest_result['invested_pct']:.1f}%")

        plot_ml_backtest(
            backtest_result, best_model_name, TICKER
        )

    # --- Multi-Ticker Test ---
    print("\n9. Multi-Ticker Out-of-Sample Test")
    print("-" * 40)

    test_tickers = ["QQQ", "AAPL", "MSFT", "JPM"]
    multi_results = []

    for t in test_tickers:
        try:
            d         = load_data(t, "5y")
            X_t, y_t, fc_t = prepare_dataset(
                d, forward_days=FWD_DAYS
            )

            # Gleiches Modell aber neu trainiert
            t_models  = get_models()
            _, t_sum  = train_and_evaluate(
                X_t, y_t, t_models, n_splits=3
            )

            best_t = max(
                t_sum, key=lambda m: t_sum[m]["AUC-ROC"]
            )

            multi_results.append({
                "Ticker":   t,
                "Modell":   best_t,
                "AUC-ROC":  t_sum[best_t]["AUC-ROC"],
                "Accuracy": t_sum[best_t]["Accuracy"],
            })
            print(f"  {t}: AUC={t_sum[best_t]['AUC-ROC']:.3f}, "
                  f"Acc={t_sum[best_t]['Accuracy']:.3f}")

        except Exception as e:
            print(f"  {t}: Fehler — {e}")

    if multi_results:
        multi_df = pd.DataFrame(multi_results)
        print(f"\n  Durchschnittliche AUC-ROC: "
              f"{multi_df['AUC-ROC'].mean():.3f}")

        # Export
        multi_df.to_csv("day23_ml_results.csv", index=False)
        importance_df.to_csv(
            "day23_feature_importance.csv", index=False
        )
        print("\nGespeichert: day23_ml_results.csv, "
              "day23_feature_importance.csv")

    print("\n" + "="*55)
    print("WICHTIGE ERKENNTNISSE:")
    print("="*55)
    print(f"  AUC > 0.55: echter Edge — in Finance sehr wertvoll")
    print(f"  AUC < 0.52: kein Signal — nicht handelbar")
    print(f"  Feature Importance zeigt was wirklich zählt")
    print(f"  Time Series CV verhindert Look-ahead Bias")






# verändern: 
# Threshold 