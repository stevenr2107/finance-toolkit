"""
Day 24 — LSTM Neural Network
Long Short Term Memory 

Recurrent neural network -> Input comes to Note -> Output comes back-> Repeat

Long Term dependency problem -> Desto mehr es lernt desto schwieriger zu behalten

LSTM Cell hat 3 Gates -> Forget Gate - Input Gate - Output Gate
Alle werden mit 1 oder 0 gefüllt und so kann man steuern, was man behält und was man vergisst
Bsp: Martin is buying apples - wir behalten marten und apples
neu: Jennifer is buying apples - wir vergessen marten und halten jennifer

*** -> wir schauen uns grundlegend an wie aht sich der satz verändert? ***

Warum LSTM für Zeitreihen?
    Normale Neural Networks vergessen.
    Sie sehen Input X → Output Y — fertig.
    Sie haben kein Gedächtnis zwischen Zeitschritten.

        Ideal für Sequenzen: Text, Audio, Zeitreihen.

    Das LSTM "liest" die letzten 60 Tage
    und versucht Tag 61 vorherzusagen.

Warum LSTM nicht magisch ist:
    Märkte sind teilweise effizient.
    LSTM findet Muster — aber ob sie morgen noch gelten ist unklar.
    Über-Engineering ohne Domain-Knowledge schlägt einfache Modelle.
    Trotzdem: LSTM ist State-of-the-Art für Sequenz-Probleme.

Was du heute baust:
    1. Daten vorbereiten — Sequenzen erstellen
    2. LSTM Architektur definieren
    3. Training mit Validation
    4. Vorhersage und Backtesting
    5. Multi-Step Forecast (30 Tage Horizont)
    6. Uncertainty Quantification (Monte Carlo Dropout)
"""

import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.preprocessing import MinMaxScaler # unterschied zu standardscaler:
# standardscaler: alle werte + / -
# minmaxscaler: alle werte zwischen 0 und 1
#-> Oben erklärt - arbeitet mit 3 Gates alle brauchen 0/1
from sklearn.metrics import mean_squared_error, mean_absolute_error # Regression stat Klassifikation
# Tag 23 - accuracy score und auc roc
# tag 24 - Konkreter Preis wird vorausgesagt
# mean squared - großer fehler -> starke Ausreißer 
# mean absolute error - robuster als mean squared  "druchschnittlich 5$ daneben"
import warnings
warnings.filterwarnings("ignore")

# TensorFlow / Keras crash vermeiden falls nicht vorhanden 
try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential, Model # sequential -> Schichten gestapelt
    # jede schicht hat einen input und einen output 
    # Model: Multiple Inputs und Outputs 
    # Für Monte Carlo Dropout 
    from tensorflow.keras.layers import (
        LSTM, Dense, Dropout, BatchNormalization,
        Input, Bidirectional, GRU
    )
    """
    Forget Gate:  Was soll vergessen werden?
              f = σ(W_f × [h_t-1, x_t] + b_f)

    Input Gate:   Was soll neu gespeichert werden?
              i = σ(W_i × [h_t-1, x_t] + b_i)

    Output Gate:  Was soll ausgegeben werden?
              o = σ(W_o × [h_t-1, x_t] + b_o)

    Cell State:   Das "Langzeitgedächtnis"
              C_t = f × C_t-1 + i × tanh(W_c × [h_t-1, x_t])    

    Das LSTM liest Tag 1:   "Kurs steigt, Volumen hoch" → merken
    Das LSTM liest Tag 30:  "Crash" → vielleicht alles vergessen
    Das LSTM liest Tag 60:  Gibt Vorhersage für Tag 61 aus
    """
    from tensorflow.keras.callbacks import (
        EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
    )
    from tensorflow.keras.optimizers import Adam
    print(f"TensorFlow Version: {tf.__version__}")
    TF_AVAILABLE = True
except ImportError:
    print("TensorFlow nicht installiert.")
    print("Installiere mit: pip install tensorflow")
    TF_AVAILABLE = False

# Reproduzierbarkeit
np.random.seed(42)
if TF_AVAILABLE:
    tf.random.set_seed(42)

def load_and_prepare(ticker:     str,
                     period:     str = "10y") -> pd.DataFrame:
    """
    Lädt Daten und berechnet zusätzliche Features für LSTM.

    Warum mehr als nur Close?
        LSTM kann aus mehreren parallelen Zeitreihen lernen.
        Returns, Volumen, technische Indikatoren geben Kontext und fließen ein .
        "Multivariate LSTM" → besser als univariates. -> 
        univariat sagt nur close_61 vorher aber keinen kontext
    """
    df = yf.download(ticker, period=period,
                     auto_adjust=True, progress=False)
    df.columns = df.columns.get_level_values(0)
    df = df.dropna()

    close  = df["Close"].squeeze()
    high   = df["High"].squeeze()
    low    = df["Low"].squeeze()
    volume = df["Volume"].squeeze()
    open_  = df["Open"].squeeze()

    # Returns
    df["ret_1d"]  = close.pct_change() # gestern
    df["ret_5d"]  = close.pct_change(5) # wochen
    df["ret_21d"] = close.pct_change(21) # monat
    # Lernt verschiedene Signale gleichzeitig 

    # Technische Indikatoren
    df["rsi"] = compute_rsi(close, 14) / 100

    ema12     = close.ewm(span=12, adjust=False).mean()
    ema26     = close.ewm(span=26, adjust=False).mean()
    df["macd_hist"] = (ema12 - ema26 - \
                       (ema12 - ema26).ewm(span=9).mean()) / close

    sma20         = close.rolling(20).mean()
    std20         = close.rolling(20).std()
    df["bb_pct_b"] = (close - (sma20 - 2*std20)) / \
                      (4 * std20 + 1e-8)

    # Normalisierte Preise
    df["hl_range"] = (high - low) / close # Tagesrange - vergleichbar
    df["gap"]      = (open_ - close.shift(1)) / close.shift(1)
    # gaps

    # Volumen
    avg_vol         = volume.rolling(20).mean()
    df["vol_ratio"] = volume / (avg_vol + 1e-8)

    # Volatilität
    df["realized_vol"] = close.pct_change().rolling(10).std() * \
                          np.sqrt(252) # 10 Tage volatilität zeigt ob Markt ruhig oder turbulent ist 

    return df.dropna()


def compute_rsi(close: pd.Series, window: int = 14) -> pd.Series:
    delta    = close.diff()
    gain     = delta.clip(lower=0)
    loss     = -delta.clip(upper=0)
    avg_gain = gain.ewm(com=window - 1, adjust=False).mean()
    avg_loss = loss.ewm(com=window - 1, adjust=False).mean()
    rs       = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))


def create_sequences(data:        np.ndarray,
                     seq_length:  int = 60,
                     target_col:  int = 0,
                     pred_steps:  int = 1) -> tuple:
    """
    Erstellt Sequenzen für LSTM Training.

    Input Shape:  (samples, timesteps, features)
    Output Shape: (samples, pred_steps)

    Beispiel mit seq_length=60:
        X[0] = Daten von Tag 0-59  → y[0] = Tag 60
        X[1] = Daten von Tag 1-60  → y[1] = Tag 61
        X[2] = Daten von Tag 2-61  → y[2] = Tag 62
        ...

    Slidet immer einen Tag weiter und lernt aus den vorherigen 60 Tagen
    """
    X, y = [], []

    for i in range(len(data) - seq_length - pred_steps + 1): # slidet bis zum letzten Tag -60
        X.append(data[i : i + seq_length]) # schaut ob es 60 tage sind 
        y.append(data[i + seq_length : i + seq_length + pred_steps,
                       target_col]) # Gibt nur close_61 aus nicht vol etc

    return np.array(X), np.array(y) # konverteirt in eine 3d liste ( zauberwürfel)


def prepare_lstm_data(df:           pd.DataFrame,
                      feature_cols: list,
                      seq_length:   int   = 60,
                      train_pct:    float = 0.70, # 70% train
                      val_pct:      float = 0.15, # 15% validation
                      pred_steps:   int   = 1) -> dict: # test an restlichen 15%
    """
    Vollständige Datenvorbereitung für LSTM.

    Schritte:
        1. Features auswählen
        2. Skalieren (MinMaxScaler → 0 bis 1)
        3. Sequenzen erstellen
        4. Train / Val / Test Split

    Warum MinMaxScaler und nicht StandardScaler?
        LSTM mit Sigmoid/Tanh Activation arbeitet
        am besten im Bereich [-1, 1] oder [0, 1].
        MinMaxScaler garantiert das.

    WICHTIG: Scaler nur auf Train-Daten fitten.
        Sonst sieht Test-Periode den Scaler — Look-ahead Bias.
    """
    data = df[feature_cols].values

    n_train = int(len(data) * train_pct)
    n_val   = int(len(data) * val_pct)

    train_data = data[:n_train]
    val_data   = data[n_train : n_train + n_val]
    test_data  = data[n_train + n_val:]

    # Scaler NUR auf Train-Daten fitten
    from sklearn.preprocessing import RobustScaler
    scaler = RobustScaler()
    scaler.fit(train_data)

    train_scaled = scaler.transform(train_data) 
    val_scaled   = scaler.transform(val_data)
    test_scaled  = scaler.transform(test_data)

    # alle bekommen eigene Sequenzen
    X_train, y_train = create_sequences( 
        train_scaled, seq_length, 0, pred_steps
    )
    X_val,   y_val   = create_sequences(
        val_scaled,   seq_length, 0, pred_steps
    )
    X_test,  y_test  = create_sequences(
        test_scaled,  seq_length, 0, pred_steps
    )

    """
    train_scaled: 1764 Zeilen → X_train: (1703, 60, 11)  ← 1764-60-1=1703
    val_scaled:    378 Zeilen → X_val:   (317,  60, 11)
    test_scaled:   378 Zeilen → X_test:  (317,  60, 11)
    """

    print(f"Shapes:")
    print(f"  X_train: {X_train.shape}")
    print(f"  X_val:   {X_val.shape}")
    print(f"  X_test:  {X_test.shape}")
    print(f"  Features: {len(feature_cols)}")

    return {
        "X_train": X_train, "y_train": y_train,
        "X_val":   X_val,   "y_val":   y_val,
        "X_test":  X_test,  "y_test":  y_test,
        "scaler":  scaler, # für inverse_transform
        "n_train": n_train, # für plot koordinaten
        "n_val":   n_val,
        "seq_length": seq_length,
        "feature_cols": feature_cols,
        "dates": df.index, # für x achse im plot 
        "raw_data": data, # für vergleich mit echten preisen
    }

"""
load_and_prepare("AAPL")
    → DataFrame mit 11 Features, 2520 Zeilen (10 Jahre)
         ↓
prepare_lstm_data(df, feature_cols, seq_length=60)
    │
    ├── Split: 70% Train / 15% Val / 15% Test
    │
    ├── scaler.fit(train_data)  ← NUR Training!
    │
    ├── create_sequences(train) → X_train: (1703, 60, 11)
    ├── create_sequences(val)   → X_val:   (317,  60, 11)
    └── create_sequences(test)  → X_test:  (317,  60, 11)
         ↓
LSTM Model:
    Input: (batch, 60, 11)   ← 60 Tage, 11 Features
    Output: (batch, 1)        ← Preis von morgen (skaliert)
         ↓
scaler.inverse_transform()
    → echte Dollar-Preise für Visualisierung und Backtest
"""


def build_lstm_model(seq_length:  int,
                     n_features:  int,
                     units:       list = [128, 64],
                     dropout:     float = 0.20,
                     pred_steps:  int   = 1,
                     learning_rate: float = 0.001) -> "tf.keras.Model":
    """
    LSTM Architektur — Schicht für Schicht erklärt. Es lernt selbst wann es vergisst und wasnn behält

    Forget Gate:
    "Ist der Crash von vor 6 Wochen noch relevant?"
    → Ja → behalten im Gedächtnis
    → Nein → vergessen

    Input Gate:
    "Ist das heutige Volumen-Signal wichtig genug?"
    → Ja → ins Gedächtnis aufnehmen
    → Nein → ignorieren

    Output Gate:
    "Was aus meinem Gedächtnis brauche ich JETZT für die Vorhersage?"
    → filtert was nach außen gegeben wird

    Cell State = das Notizbuch des Analysten
    → wird täglich mit den drei Gates aktualisiert
    → enthält das "Langzeitgedächtnis" der Sequenz 

    Input Layer:
        Shape: (seq_length, n_features)
        seq_length=60: letzten 60 Tage
        n_features=12: Anzahl der Features

    LSTM Layer 1 (128 Units):
        return_sequences=True → gibt Output für jeden Zeitschritt zurück
        Nötig wenn danach eine weitere LSTM-Schicht kommt.
        128 Units = 128 "Gedächtniszellen"

    Dropout (20%):
        Zufällig 20% der Neuronen werden pro Batch deaktiviert.
        Regularisierung → reduziert Overfitting.
        Standard in jedem tiefen Netz.
    """
    if not TF_AVAILABLE:
        raise ImportError("TensorFlow nicht verfügbar")

    model = Sequential([
        # Input
        Input(shape=(seq_length, n_features)), # (60 tage, 11 features)

        # LSTM Block 1 - Lernen
        LSTM(units[0],
             return_sequences=True, # nach jedem artikel eine zusammenfassung -> 60 Outputs
             kernel_regularizer=tf.keras.regularizers.l2(1e-4)),
        BatchNormalization(), # standardisierung nur mit 0,1 statt üblich LSTM -100 - 300
        Dropout(dropout), # Regulieren -> 20% der Neuronen werden deaktiviert
        # reduziert overfitting 

        # LSTM Block 2 Zusammenfassung der ganzen sequenzen 
        LSTM(units[1], 
             return_sequences=False, # nur eine Zusammenfassung nach 60 Inputs
             kernel_regularizer=tf.keras.regularizers.l2(1e-4)),
        BatchNormalization(),
        Dropout(dropout),

        # Dense Layers
        Dense(32, activation="relu"),
        Dropout(dropout / 2),

        # Output
        Dense(pred_steps, activation="linear")
    ])

    model.compile(
        optimizer=Adam(learning_rate=learning_rate,
                       clipnorm=1.0),   # Gradient Clipping
        loss="mse",
        metrics=["mae"]
    )

    return model


def build_bidirectional_lstm(seq_length:    int,
                              n_features:   int,
                              units:        int   = 64,
                              dropout:      float = 0.20,
                              pred_steps:   int   = 1) -> "tf.keras.Model":
    """
    Bidirectional LSTM — liest Sequenz vorwärts UND rückwärts.

    Warum Bidirectional?
        Normales LSTM: sieht nur Vergangenheit.
        Bidirectional: sieht auch "Zukunft" innerhalb der Sequenz.
        Geht auch von Tag 60 zurück zu tag 0 

    Für Forecasting: nur Training, nicht Inference.
    In der Praxis oft besser als unidirektionales LSTM.
    """
    if not TF_AVAILABLE:
        raise ImportError("TensorFlow nicht verfügbar")

    model = Sequential([
        Input(shape=(seq_length, n_features)),

        Bidirectional(LSTM(units,
                           return_sequences=True)),
        BatchNormalization(),
        Dropout(dropout),

        Bidirectional(LSTM(units // 2,
                           return_sequences=False)), # für die zweite schicht 
                           # man hätte dadurch das man vor und zurück geht die doppelte größe
                           # aber wir wollen nur 32 um es zu verdichten und die besten infos zu bekommen
        BatchNormalization(),
        Dropout(dropout),

        Dense(32, activation="relu"),
        Dense(pred_steps, activation="linear")
    ])

    model.compile(
        optimizer=Adam(learning_rate=0.001, clipnorm=1.0),
        loss="mse",
        metrics=["mae"]
    )

    return model


def build_gru_model(seq_length:  int,
                    n_features:  int,
                    units:       int   = 64,
                    dropout:     float = 0.20,
                    pred_steps:  int   = 1) -> "tf.keras.Model":
    """
    GRU — Gated Recurrent Unit.

    GRU vs LSTM:
        GRU: einfacher (2 Gates statt 3), weniger Parameter.
        LSTM: komplexer, besser für sehr lange Sequenzen.
        Für Finanzdaten (60-100 Tage): oft ähnlich gut.
        GRU: schneller zu trainieren.
        Reset und Update Gate 
        - Reset: wieviel vergangenheit soll einfließen 
        - update: Wieviel neu und alt
    """
    if not TF_AVAILABLE:
        raise ImportError("TensorFlow nicht verfügbar")

    model = Sequential([
        Input(shape=(seq_length, n_features)),

        GRU(units, return_sequences=True),
        BatchNormalization(),
        Dropout(dropout),

        GRU(units // 2, return_sequences=False),
        BatchNormalization(),
        Dropout(dropout),

        Dense(32, activation="relu"),
        Dense(pred_steps, activation="linear")
    ])

    model.compile(
        optimizer=Adam(learning_rate=0.001, clipnorm=1.0),
        loss="mse",
        metrics=["mae"]
    )

    return model

def train_model(model:       "tf.keras.Model",
                data:        dict,
                epochs:      int = 100, # anzahl an wiederholungen 
                batch_size:  int = 32,
                model_name:  str = "lstm") -> dict:
    """
    Training mit professionellen Callbacks.

    EarlyStopping:
        Stoppt wenn Validation Loss sich 15 Epochen
        nicht mehr verbessert.
        Verhindert Overfitting automatisch.
        restore_best_weights → bestes Modell wird gespeichert.

    ReduceLROnPlateau:
        Reduziert Learning Rate wenn kein Fortschritt.
        LR / 2 nach 7 Epochen ohne Verbesserung.
        Hilft aus lokalen Minima herauszukommen.

    Batch Size 32:
        Standard für Financial Time Series.
        Größer = stabiler Gradient aber weniger Updates.
        Kleiner = noisier aber öfter Updates.
    """

    """
    Epoch 10: val_loss = 0.0023  ← neuer Bestwert, speichern
    Epoch 11: val_loss = 0.0025  ← schlechter (1)
    Epoch 12: val_loss = 0.0027  ← schlechter (2)
    ...
    Epoch 25: val_loss = 0.0031  ← schlechter (15) → STOPP

    restore_best_weights=True:
    → springt automatisch zurück zu Epoch 10
    → das übermäßig trainierte Modell wird verworfen ✅
    """
    callbacks = [
        EarlyStopping(
            monitor              = "val_loss", # schaut auf validation loss
            patience             = 15, # wartet 15 epochen
            restore_best_weights = True, # springt zurück zum besten modell
            verbose              = 1
        ),
        ReduceLROnPlateau(
            monitor  = "val_loss", # fällt erst dann steigt er wieder
            factor   = 0.5, # halbe learning rate  # ↓↓↓↓↓↓↓↓↓↑↑↑↑↑↑ nimmt den ersten pfeil nach oben
            patience = 7, # nach 7 epochen ohne verbesserung 
            min_lr   = 1e-6, # schritte in der gelernt wird verhindert überspringen von wichtigem 
            verbose  = 1 # hält terminal sauber 
        ),

    ]
    """
    Epoch 1-20:  LR = 0.001   → schnelles Lernen
    Epoch 21-27: kein Fortschritt (patience=7)
    Epoch 28:    LR = 0.0005  ← halbiert
    Epoch 29-35: kein Fortschritt
    Epoch 36:    LR = 0.00025 ← wieder halbiert..."""

    print(f"\nTraining {model_name}...")
    print(f"  Epochs:     {epochs}") # anzahl an wiederholungen
    print(f"  Batch Size: {batch_size}")
    print(f"  Train:      {data['X_train'].shape[0]} Samples")
    print(f"  Val:        {data['X_val'].shape[0]} Samples") # Anzahl an "Übungsaufgaben"

    history = model.fit(
        data["X_train"], data["y_train"], # trainingsdaten
        validation_data = (data["X_val"], data["y_val"]),
        epochs          = epochs, # max 100 runden
        batch_size      = batch_size, # 32 sequenzen gleichzeitig da sonst zu viel speicher
        callbacks       = callbacks, # die zwei assistenten
        verbose         = 0, # kein output pro epoche
        shuffle         = False   # Zeitreihen: kein Shuffling!
        # shoffling zerstört die chronologische reihenfloge
    )

    # Best Epoch
    best_epoch = np.argmin(history.history["val_loss"]) + 1 # gibt kleinesten wert zurück
    best_val   = min(history.history["val_loss"])

    print(f"\n  Best Epoch:     {best_epoch}/{epochs}")
    print(f"  Best Val Loss:  {best_val:.6f}")

    return {
        "history":    history,
        "best_epoch": best_epoch,
        "best_val":   best_val,
        "model_name": model_name,
    }

"""
Epoch 1:   train_loss ↓   val_loss ↓   → Fortschritt
Epoch 2:   train_loss ↓   val_loss ↓   → Fortschritt
...
Epoch 15:  train_loss ↓   val_loss ↑   → Overfitting beginnt
...
Epoch 22:  ReduceLR greift  LR: 0.001→0.0005
Epoch 30:  EarlyStopping greift (15 Epochen ohne val_loss Verbesserung)
→ Modell springt zurück zu Epoch 15 (restore_best_weights)

Resultat:
best_epoch = 15
best_val   = 0.00312
"""

def predict_and_evaluate(model:  "tf.keras.Model",
                          data:   dict,
                          ticker: str) -> dict:
    """
    Vorhersage auf Test-Daten und Evaluation.

    Inverse Transform:
        Vorhersagen sind im skalierten Raum (0-1).
        Wir müssen zurück in den Original-Preis-Raum.
        Achtung: Scaler hat alle Features — wir brauchen
        nur die erste Spalte (Close Price).
    """
    scaler      = data["scaler"]
    n_features  = data["X_test"].shape[2]
    seq_length  = data["seq_length"]
    n_train     = data["n_train"]
    n_val       = data["n_val"]

    # Predictions
    y_pred_scaled = model.predict(
        data["X_test"], verbose=0, batch_size=64
    )
    y_true_scaled = data["y_test"]

    # Inverse Transform — nur Close (Spalte 0)
    """Ohne Trick:
    scaler.inverse_transform([0.73])  → Fehler: erwartet 11 Werte

    Mit Trick:
    scaler.inverse_transform([0.73, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    → [$185.4, 0, 0, ...]  → [:, 0] → $185.4 ✅
    """
    def inverse_transform_close(scaled: np.ndarray) -> np.ndarray:
        """Invert nur die Close-Preis Spalte."""
        dummy = np.zeros((len(scaled), n_features))
        dummy[:, 0] = scaled.flatten()
        return scaler.inverse_transform(dummy)[:, 0]

    y_pred = inverse_transform_close(y_pred_scaled)
    y_true = inverse_transform_close(y_true_scaled)

    # Metriken
    rmse = np.sqrt(mean_squared_error(y_true, y_pred)) # durchschnittlicher Fehler in Dollar, stärker von Ausreißern beeinflusst
    mae  = mean_absolute_error(y_true, y_pred) # durchschnittlicher Fehler in Dollar
    mape = np.mean(np.abs((y_true - y_pred) /
                           (y_true + 1e-8))) * 100 # durchschnittlicher Fehler in Prozent, robuster als RMSE

    # Richtungs-Genauigkeit
    # Wichiger als RMSE: prediziert das Modell die RICHTUNG?
    y_true_ret = np.diff(y_true) # echte tägliche veränderung 
    y_pred_ret = np.diff(y_pred) # vorhergesagte tägliche veränderung
    direction_acc = (np.sign(y_true_ret) ==
                     np.sign(y_pred_ret)).mean() * 100 # sign gibt richtung 
    # richtungs genauigkeit wichtig 

    # Datum-Index für Test-Periode
    test_start = n_train + n_val + seq_length
    dates      = data["dates"]
    test_dates = dates[test_start : test_start + len(y_pred)]

    print(f"\n  Evaluation auf Test-Daten ({len(y_true)} Punkte):")
    print(f"  RMSE:            ${rmse:.2f}")
    print(f"  MAE:             ${mae:.2f}")
    print(f"  MAPE:            {mape:.2f}%")
    print(f"  Richtungs-Acc:   {direction_acc:.1f}%")
    print(f"  Baseline (50%):  50.0%")

    return {
        "y_pred":        y_pred,
        "y_true":        y_true,
        "test_dates":    test_dates,
        "rmse":          round(rmse, 4),
        "mae":           round(mae, 4),
        "mape":          round(mape, 4),
        "direction_acc": round(direction_acc, 2),
    }


def multi_step_forecast(model:      "tf.keras.Model",
                         data:       dict,
                         n_days:     int = 30) -> dict:
    """
    Multi-Step Forecast — nächste N Tage vorhersagen.

    Recursive Strategy:
        Tag 1 vorhersagen → als Input für Tag 2 nutzen
        Tag 2 vorhersagen → als Input für Tag 3 nutzen
        ... -> 30. Tag am schwierigsten 

    Problem: Fehler akkumulieren sich.
    Je weiter in die Zukunft desto unzuverlässiger.
    Daher: Konfidenzintervall wird breiter mit der Zeit.

    Das ist ehrlicher als ein einzelner Wert.
    """
    scaler      = data["scaler"]
    n_features  = data["X_test"].shape[2]
    seq_length  = data["seq_length"]

    # Letzten verfügbaren Datenpunkt nehmen
    last_sequence = data["X_test"][-1].copy()

    forecasts = []

    for step in range(n_days):
        # Vorhersage
        input_seq  = last_sequence.reshape(1, seq_length, n_features)
        pred_scaled = model.predict( # einen tag voraus
            input_seq, verbose=0
        )[0, 0]

        forecasts.append(pred_scaled)

        # Sequence updaten: ältesten Tag raus, neuen rein
        new_row              = last_sequence[-1].copy()
        new_row[0]           = pred_scaled   # Close aktualisieren
        last_sequence        = np.roll(last_sequence, -1, axis=0)
        last_sequence[-1]    = new_row
        """
        Vorher:     [Tag1, Tag2, Tag3, ..., Tag60]
        np.roll(-1): [Tag2, Tag3, Tag4, ..., Tag60, Tag60]  ← Tag60 doppelt
        last[-1] = new_row:  [Tag2, Tag3, ..., Tag60, Tag61_predicted]
        """

    # Inverse Transform
    forecasts_arr = np.array(forecasts)
    dummy         = np.zeros((n_days, n_features))
    dummy[:, 0]   = forecasts_arr
    prices        = scaler.inverse_transform(dummy)[:, 0]

    # Unsicherheit: wächst mit der Zeit (sqrt)
    last_price = prices[0]
    volatility = 0.01   # ~1% tägl. Volatilität als Baseline
    uncertainty = last_price * volatility * np.sqrt(
        np.arange(1, n_days + 1)
    ) # wachsen mit der Zeit

    return {
        "prices":     prices,
        "upper":      prices + 1.96 * uncertainty,
        "lower":      prices - 1.96 * uncertainty,
        "n_days":     n_days,
    }


def monte_carlo_dropout_uncertainty(model:      "tf.keras.Model",
                                     data:       dict,
                                     n_samples:  int = 100) -> dict:
    """
    Monte Carlo Dropout — Uncertainty Quantification.

    Idee (Gal & Ghahramani, 2016):
        Dropout im Training verhindert Overfitting.
        Dropout im Inference mit N Passes → N verschiedene Vorhersagen.
        Deren Streuung = Modell-Unsicherheit.

    Das ist Bayesian Deep Learning — State of the Art
    für Uncertainty in Neural Networks.

    Wenn das Modell sicher ist:
        N Vorhersagen liegen nah beieinander → kleine Varianz.
    Wenn das Modell unsicher ist:
        N Vorhersagen streuen weit → große Varianz → Finger weg.
    """
    scaler     = data["scaler"]
    n_features = data["X_test"].shape[2]
    X_sample = data["X_test"][-50:]

    # Funktion die Dropout im Inference aktiviert
    X_tensor = tf.constant(X_sample, dtype=tf.float32)  
    @tf.function
    def predict_with_dropout(x):
        return model(x, training=True)   # training=True → Dropout aktiv

    # Letzten 50 Test-Punkte für Uncertainty
    X_sample = data["X_test"][-50:]
    all_preds = []

    for _ in range(n_samples):
        preds = predict_with_dropout(X_tensor)  # ← immer dasselbe Objekt
        preds = preds.numpy().flatten()
        all_preds.append(preds)

    all_preds = np.array(all_preds) # 100 runs x 50 samples

    # Statistiken
    mean_pred = all_preds.mean(axis=0) # mittelwert über 100 runs
    std_pred  = all_preds.std(axis=0) # streuung über 100 runs

    # Inverse Transform
    def inv(arr):
        dummy      = np.zeros((len(arr), n_features))
        dummy[:, 0] = arr
        return scaler.inverse_transform(dummy)[:, 0]

    return {
        "mean":   inv(mean_pred),
        "upper":  inv(mean_pred + 1.96 * std_pred),
        "lower":  inv(mean_pred - 1.96 * std_pred),
        "std":    std_pred,
    }

def plot_training_history(histories: dict) -> None:
    """
    Training Verlauf aller Modelle.
    Loss sinkt → Modell lernt. Wenn Val Loss steigt → Overfitting.
    """
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=["Training Loss (MSE)",
                        "Validation Loss (MSE)"],
        horizontal_spacing=0.10
    )

    colors = {
        "LSTM":             "#2563eb",
        "Bidirectional LSTM": "#16a34a",
        "GRU":              "#f59e0b",
    }

    for name, hist_result in histories.items():
        history = hist_result["history"]
        color   = colors.get(name, "#8b5cf6")
        epochs  = range(1, len(history.history["loss"]) + 1)

        fig.add_trace(go.Scatter(
            x=list(epochs),
            y=history.history["loss"],
            name=f"{name} Train",
            line=dict(color=color, width=2)
        ), row=1, col=1)

        fig.add_trace(go.Scatter(
            x=list(epochs),
            y=history.history["val_loss"],
            name=f"{name} Val",
            line=dict(color=color, width=1.5, dash="dash")
        ), row=1, col=2)

    fig.update_layout(
        height=400,
        template="plotly_white",
        title="LSTM Training Verlauf",
        legend=dict(orientation="h", y=1.05),
        margin=dict(l=0, r=0, t=60, b=0)
    )

    fig.update_yaxes(title_text="MSE Loss", row=1, col=1)
    fig.update_yaxes(title_text="MSE Loss", row=1, col=2)
    fig.update_xaxes(title_text="Epoche",   row=1, col=1)
    fig.update_xaxes(title_text="Epoche",   row=1, col=2)

    fig.show()


def plot_predictions(eval_result: dict,
                     ticker:      str,
                     model_name:  str,
                     df:          pd.DataFrame) -> None:
    """
    Vorhersage vs. echte Preise + Richtungs-Korrektheit.
    """
    y_pred    = eval_result["y_pred"]
    y_true    = eval_result["y_true"]
    dates     = eval_result["test_dates"]

    # Richtung korrekt?
    y_true_ret = np.diff(y_true)
    y_pred_ret = np.diff(y_pred)
    direction  = np.sign(y_true_ret) == np.sign(y_pred_ret)

    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        row_heights=[0.70, 0.30],
        subplot_titles=[
            f"{ticker} — {model_name}: Vorhersage vs. Realität",
            "Richtungs-Korrektheit (grün=richtig, rot=falsch)"
        ]
    )

    # Echter Preis
    fig.add_trace(go.Scatter(
        x=list(dates),
        y=y_true.round(2),
        name="Echter Preis",
        line=dict(color="#1e293b", width=2)
    ), row=1, col=1)

    # Vorhersage
    fig.add_trace(go.Scatter(
        x=list(dates),
        y=y_pred.round(2),
        name="LSTM Vorhersage",
        line=dict(color="#2563eb", width=1.5, dash="dash")
    ), row=1, col=1)

    # Error Band
    error = np.abs(y_pred - y_true)
    fig.add_trace(go.Scatter(
        x=list(dates) + list(dates[::-1]),
        y=list((y_pred + error).round(2)) +
          list((y_pred - error).round(2)[::-1]),
        fill="toself",
        fillcolor="rgba(37,99,235,0.08)",
        line=dict(width=0),
        name="Error Band",
        showlegend=True
    ), row=1, col=1)

    # Richtungs-Korrektheit
    dir_colors = [
        "#16a34a" if d else "#ef4444"
        for d in direction
    ]

    fig.add_trace(go.Bar(
        x=list(dates[1:]),
        y=[1] * len(direction),
        marker_color=dir_colors,
        opacity=0.8,
        name="Richtung",
        showlegend=False
    ), row=2, col=1)

    # Metrics Annotation
    metrics_text = (
        f"RMSE: ${eval_result['rmse']:.2f} | "
        f"MAE: ${eval_result['mae']:.2f} | "
        f"MAPE: {eval_result['mape']:.2f}% | "
        f"Dir. Acc: {eval_result['direction_acc']:.1f}%"
    )

    fig.add_annotation(
        x=0.5, y=1.02,
        xref="paper", yref="paper",
        text=metrics_text,
        showarrow=False,
        font=dict(size=11, color="#475569")
    )

    fig.update_layout(
        height=650,
        template="plotly_white",
        hovermode="x unified",
        legend=dict(orientation="h", y=1.06),
        margin=dict(l=0, r=0, t=70, b=0)
    )

    fig.update_yaxes(title_text="Preis ($)", row=1, col=1)
    fig.update_yaxes(title_text="",          row=2, col=1,
                     showticklabels=False)

    fig.show()


def plot_forecast(df:           pd.DataFrame,
                  forecast:     dict,
                  mc_dropout:   dict,
                  ticker:       str,
                  seq_length:   int = 60) -> None:
    """
    Future Forecast + Monte Carlo Dropout Uncertainty.
    """
    close = df["Close"].squeeze()

    # Letzte 120 Tage historisch + 30 Tage Forecast
    hist_dates  = close.index[-120:]
    hist_prices = close.values[-120:]

    last_date   = close.index[-1]
    future_dates = pd.bdate_range(
        start=last_date + pd.Timedelta(days=1),
        periods=forecast["n_days"]
    )

    fig = go.Figure()

    # Historisch
    fig.add_trace(go.Scatter(
        x=hist_dates,
        y=hist_prices.round(2),
        name="Historisch",
        line=dict(color="#1e293b", width=2)
    ))

    # Übergang
    fig.add_trace(go.Scatter(
        x=[hist_dates[-1], future_dates[0]],
        y=[float(hist_prices[-1]),
           float(forecast["prices"][0])],
        line=dict(color="#94a3b8", width=1, dash="dot"),
        showlegend=False
    ))

    # Forecast
    fig.add_trace(go.Scatter(
        x=future_dates,
        y=forecast["prices"].round(2),
        name="LSTM Forecast",
        line=dict(color="#2563eb", width=2.5)
    ))

    # 95% Konfidenzintervall
    fig.add_trace(go.Scatter(
        x=list(future_dates) + list(future_dates[::-1]),
        y=list(forecast["upper"].round(2)) +
          list(forecast["lower"].round(2)[::-1]),
        fill="toself",
        fillcolor="rgba(37,99,235,0.12)",
        line=dict(width=0),
        name="95% KI"
    ))

    # Monte Carlo Uncertainty (letzte 50 Test-Punkte)
    mc_dates = close.index[-50:]
    fig.add_trace(go.Scatter(
        x=mc_dates,
        y=mc_dropout["upper"].round(2),
        name="MC Upper",
        line=dict(color="#f59e0b", width=1, dash="dot"),
        showlegend=True
    ))
    fig.add_trace(go.Scatter(
        x=mc_dates,
        y=mc_dropout["lower"].round(2),
        name="MC Lower",
        line=dict(color="#f59e0b", width=1, dash="dot"),
        fill="tonexty",
        fillcolor="rgba(245,158,11,0.08)"
    ))

    # Heutiger Kurs
    fig.add_hline(
        y=float(close.iloc[-1]),
        line_dash="dot",
        line_color="#94a3b8",
        annotation_text=f"Heute: ${float(close.iloc[-1]):.2f}",
        opacity=0.6
    )

    fig.update_layout(
        title=f"{ticker} — LSTM Forecast (30 Tage) + MC Dropout Uncertainty",
        xaxis_title="Datum",
        yaxis_title="Preis ($)",
        template="plotly_white",
        height=550,
        legend=dict(orientation="h", y=1.02),
        margin=dict(l=0, r=0, t=60, b=0)
    )

    fig.show()


def plot_model_comparison_eval(eval_results: dict) -> None:
    """
    Vergleicht alle Modelle nach RMSE, MAE, Direction Accuracy.
    """
    metrics  = ["rmse", "mae", "direction_acc"]
    labels   = ["RMSE ($)", "MAE ($)", "Richtungs-Acc (%)"]
    colors   = ["#ef4444", "#f59e0b", "#16a34a"]
    models   = list(eval_results.keys())

    fig = make_subplots(
        rows=1, cols=3,
        subplot_titles=labels,
        horizontal_spacing=0.10
    )

    for col_idx, (metric, label, color) in enumerate(
        zip(metrics, labels, colors), 1
    ):
        values = [
            eval_results[m][metric] for m in models
        ]

        # Bester Wert: kleinstes RMSE/MAE, größtes Dir-Acc
        if metric == "direction_acc":
            best_val = max(values)
        else:
            best_val = min(values)

        bar_colors = [
            "#16a34a" if v == best_val else "#3b82f6"
            for v in values
        ]

        fig.add_trace(go.Bar(
            x=models,
            y=values,
            marker_color=bar_colors,
            text=[f"{v:.3f}" for v in values],
            textposition="outside",
            showlegend=False,
        ), row=1, col=col_idx)

        # Baseline für Direction Accuracy
        if metric == "direction_acc":
            fig.add_hline(
                y=50,
                line_dash="dot",
                line_color="#ef4444",
                opacity=0.6,
                annotation_text="Baseline 50%",
                row=1, col=col_idx
            )

    fig.update_layout(
        height=420,
        template="plotly_white",
        title="Model Comparison — Test Set Evaluation",
        margin=dict(l=0, r=0, t=60, b=0)
    )

    fig.show()


def lstm_trading_backtest(eval_result: dict,
                           df:          pd.DataFrame,
                           capital:     float = 10_000,
                           threshold:   float = 0.003) -> dict:
    """
    Backtested LSTM Signal auf echten Preisen.

    Signal:
        Wenn LSTM vorhergesagt hat: morgen steigt Preis > threshold%
        → Long heute (Kauf auf heutigem Close)
        → Exit morgen auf Close

    Threshold 0.3%:
        Nur kaufen wenn Modell klare Aufwärts-Bewegung sieht.
        Filtert kleine, unsichere Signale raus.
    """
    y_pred      = eval_result["y_pred"]
    y_true      = eval_result["y_true"]
    test_dates  = eval_result["test_dates"]

    # Returns die LSTM vorhergesagt hat
    pred_returns = np.diff(y_pred) / y_pred[:-1]
    true_returns = np.diff(y_true) / y_true[:-1]

    # Signal: Long wenn LSTM positiven Return erwartet
    signal = (pred_returns > threshold).astype(int)

    # Performance
    strat_returns  = true_returns * signal
    market_returns = true_returns

    equity_strat  = (1 + strat_returns).cumprod() * capital
    equity_market = (1 + market_returns).cumprod() * capital

    # Metriken
    years = len(equity_strat) / 252
    if years > 0 and equity_strat[0] > 0:
        cagr_strat  = (equity_strat[-1] /
                       capital) ** (1/years) - 1
        cagr_market = (equity_market[-1] /
                       capital) ** (1/years) - 1
    else:
        cagr_strat = cagr_market = 0

    sharpe = (strat_returns.mean() /
              strat_returns.std() *
              np.sqrt(252)) if strat_returns.std() > 0 else 0

    invested_pct = signal.mean() * 100

    rolling_max  = np.maximum.accumulate(equity_strat)
    max_dd       = ((equity_strat - rolling_max) /
                     rolling_max).min() * 100

    return {
        "equity_strat":  equity_strat,
        "equity_market": equity_market,
        "dates":         test_dates[1:],
        "signal":        signal,
        "cagr_strat":    round(cagr_strat  * 100, 2),
        "cagr_market":   round(cagr_market * 100, 2),
        "sharpe":        round(sharpe, 3),
        "invested_pct":  round(invested_pct, 1),
        "max_dd":        round(max_dd, 2),
    }

if __name__ == "__main__":

    if not TF_AVAILABLE:
        print("TensorFlow fehlt. Installieren mit:")
        print("  pip install tensorflow")
        exit()

    TICKER     = "HIMS"
    SEQ_LENGTH = 60
    CAPITAL    = 10_000

    print("Tag 24 — LSTM Neural Network")
    print("=" * 55)

    # --- Daten ---
    print(f"\n1. Daten laden: {TICKER}")
    df = load_and_prepare(TICKER, "10y")
    print(f"   {len(df)} Handelstage geladen")

    # Features definieren
    feature_cols = [
        "Close",
        "ret_1d", "ret_5d", "ret_21d",
        "rsi", "macd_hist", "bb_pct_b",
        "hl_range", "gap",
        "vol_ratio", "realized_vol"
    ]

    # Nur verfügbare Features
    feature_cols = [
        f for f in feature_cols if f in df.columns
    ]
    print(f"   Features: {feature_cols}")

    # --- Daten vorbereiten ---
    print("\n2. Datenvorbereitung...")
    data = prepare_lstm_data(
        df, feature_cols,
        seq_length = SEQ_LENGTH,
        train_pct  = 0.70,
        val_pct    = 0.15,
        pred_steps = 1
    )

    n_features = len(feature_cols)

    # --- Modelle bauen ---
    print("\n3. Modelle definieren...")
    models = {
        "LSTM": build_lstm_model(
            SEQ_LENGTH, n_features,
            units=[128, 64], dropout=0.20
        ),
        "Bidirectional LSTM": build_bidirectional_lstm(
            SEQ_LENGTH, n_features,
            units=64, dropout=0.20
        ),
        "GRU": build_gru_model(
            SEQ_LENGTH, n_features,
            units=64, dropout=0.20
        ),
    }

    # Summary des LSTM
    print("\n   LSTM Architektur:")
    models["LSTM"].summary()

    # --- Training ---
    print("\n4. Training...")
    histories   = {}
    eval_results = {}
    trained     = {}

    for name, model in models.items():
        print(f"\n  --- {name} ---")
        hist = train_model(
            model, data,
            epochs     = 100,
            batch_size = 32,
            model_name = name
        )
        histories[name] = hist

        # Evaluation
        print(f"\n  Evaluation {name}:")
        eval_result = predict_and_evaluate(
            model, data, TICKER
        )
        eval_results[name] = eval_result
        trained[name]      = model

    # Training History
    plot_training_history(histories)

    # --- Bestes Modell ---
    best_name = min(
        eval_results,
        key=lambda m: eval_results[m]["rmse"]
    )
    print(f"\n5. Bestes Modell: {best_name}")
    print(f"   RMSE:          ${eval_results[best_name]['rmse']:.2f}")
    print(f"   Direction Acc: {eval_results[best_name]['direction_acc']:.1f}%")

    # --- Plots ---
    print("\n6. Visualisierungen...")
    plot_predictions(
        eval_results[best_name], TICKER,
        best_name, df
    )
    plot_model_comparison_eval(eval_results)

    # --- Multi-Step Forecast ---
    print("\n7. 30-Tage Forecast...")
    forecast = multi_step_forecast(
        trained[best_name], data, n_days=30
    )

    print(f"   Letzter Preis:   ${df['Close'].iloc[-1]:.2f}")
    print(f"   Forecast Tag 5:  ${forecast['prices'][4]:.2f}")
    print(f"   Forecast Tag 15: ${forecast['prices'][14]:.2f}")
    print(f"   Forecast Tag 30: ${forecast['prices'][29]:.2f}")

    # --- Monte Carlo Dropout ---
    print("\n8. Monte Carlo Dropout Uncertainty...")
    mc_dropout = monte_carlo_dropout_uncertainty(
        trained[best_name], data, n_samples=100
    )
    print(f"   Durchschn. Unsicherheit: "
          f"±${np.mean(mc_dropout['std']) * 100:.2f}")

    plot_forecast(df, forecast, mc_dropout, TICKER, SEQ_LENGTH)

    # --- Backtest ---
    print("\n9. LSTM Trading Backtest...")
    backtest = lstm_trading_backtest(
        eval_results[best_name], df, CAPITAL
    )

    print(f"\n   LSTM Strategy:")
    print(f"   CAGR:           {backtest['cagr_strat']:+.2f}%")
    print(f"   Buy & Hold:     {backtest['cagr_market']:+.2f}%")
    print(f"   Sharpe:         {backtest['sharpe']:.3f}")
    print(f"   Max Drawdown:   {backtest['max_dd']:.2f}%")
    print(f"   Zeit investiert:{backtest['invested_pct']:.1f}%")

    # Backtest Chart
    fig_bt = go.Figure()
    fig_bt.add_trace(go.Scatter(
        x=backtest["dates"],
        y=backtest["equity_strat"].round(2),
        name=f"LSTM ({best_name})",
        line=dict(color="#2563eb", width=2)
    ))
    fig_bt.add_trace(go.Scatter(
        x=backtest["dates"],
        y=backtest["equity_market"].round(2),
        name="Buy & Hold",
        line=dict(color="#94a3b8", width=1.5, dash="dot")
    ))
    fig_bt.update_layout(
        title=f"{TICKER} — LSTM Trading Signal Backtest",
        yaxis_title="Kapital ($)",
        template="plotly_white",
        height=450,
        legend=dict(orientation="h", y=1.02),
        margin=dict(l=0, r=0, t=60, b=0)
    )
    fig_bt.show()

    # --- Multi-Ticker Test ---
    print("\n10. Multi-Ticker Generalisierung")
    print("-" * 40)

    test_tickers = ["QQQ", "AAPL"]

    for t in test_tickers:
        try:
            print(f"\n   {t}:")
            d = load_and_prepare(t, "5y")
            d_data = prepare_lstm_data(
                d, feature_cols,
                seq_length = SEQ_LENGTH,
                train_pct  = 0.70,
                val_pct    = 0.15
            )

            m = build_gru_model(
                SEQ_LENGTH, n_features,
                units=64, dropout=0.20
            )
            train_model(
                m, d_data,
                epochs=50, model_name=f"GRU-{t}"
            )
            ev = predict_and_evaluate(m, d_data, t)
            print(f"   RMSE: ${ev['rmse']:.2f}  "
                  f"Dir-Acc: {ev['direction_acc']:.1f}%")

        except Exception as e:
            print(f"   {t}: Fehler — {e}")

    print("\n" + "="*55)
    print("WICHTIGE ERKENNTNISSE:")
    print("="*55)
    print("  Direction Accuracy > 52% = echter Edge")
    print("  RMSE relativ zum Preis bewerten (MAPE %)")
    print("  MC Dropout zeigt wann das Modell unsicher ist")
    print("  Weiter Forecast Horizont = mehr Unsicherheit")