"""
Portfolio Tracker — Tranchen-basiert - sonst werden buy dates und kaufpreise verfälscht 
Vollständiger Portfolio-Tracker mit korrekter Equity Curve,
PnL, Benchmark-Vergleich und Risikokennzahlen.
 
Jede Position wird als einzelne Tranche geführt:
  shares > 0  → Kauf
  shares < 0  → Verkauf
 
Dadurch wird die Equity Curve täglich korrekt berechnet,
ohne rückwirkend falsche Stückzahlen zu verwenden.
"""
 
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings("ignore")

#--- Dein Portfolio definieren ---
# Format Ticker -> (Anzahl Aktien, Kaufpreis, Kaufdatum)

# =============================================================================
# PORTFOLIO — Jeder Kauf / Verkauf als eigene Tranche
# shares > 0 = Kauf | shares < 0 = Verkauf
# buy_price   = Transaktionspreis (EUR, da Scalable Capital)
# =============================================================================

PORTFOLIO = [
    # PYPL
    {"ticker": "PYPL", "shares":  +5,   "buy_price":  60.76, "buy_date": "2024-04-11"},
 
    # DOCU
    {"ticker": "DOCU", "shares":  +3,   "buy_price":  55.13, "buy_date": "2024-04-11"},
 
    # IWDA.AS  ← EUR-Ticker, PnL-Vergleich in USD nur annähernd
    {"ticker": "IWDA.AS", "shares": +10.000,    "buy_price":  90.61, "buy_date": "2024-04-11"},
 
    # ALB
    {"ticker": "ALB", "shares":  +3,   "buy_price": 119.76, "buy_date": "2024-04-11"},
 
    # HIMS
    {"ticker": "HIMS", "shares": +36,   "buy_price":  13.70, "buy_date": "2024-04-11"},
    {"ticker": "HIMS", "shares":  +7,   "buy_price":  11.71, "buy_date": "2024-04-24"},
    {"ticker": "HIMS", "shares":  +7,   "buy_price":  12.79, "buy_date": "2024-05-14"},
 
    # PYPL — Verkauf
    {"ticker": "PYPL", "shares":  -5,   "buy_price":  56.76, "buy_date": "2024-06-17"},
 
    # DOCU
    {"ticker": "DOCU", "shares":  +3,   "buy_price":  52.80, "buy_date": "2024-07-15"},
    {"ticker": "DOCU", "shares":  +8,   "buy_price":  52.80, "buy_date": "2024-07-15"},
 
    # CRM
    {"ticker": "CRM",  "shares":  +1,   "buy_price": 214.85, "buy_date": "2024-08-05"},
 
    # PAYC
    {"ticker": "PAYC", "shares":  +3,   "buy_price": 155.45, "buy_date": "2024-09-25"},
 
    # IWDA.AS
    {"ticker": "IWDA.AS", "shares": +1.007,    "buy_price":  99.24, "buy_date": "2024-10-14"},
 
    # HIMS — Verkauf
    {"ticker": "HIMS", "shares": -25,   "buy_price":  25.39, "buy_date": "2024-11-12"},
 
    # IWDA.AS
    {"ticker": "IWDA.AS", "shares": +0.973,    "buy_price": 102.70, "buy_date": "2024-11-13"},
 
    # ALB
    {"ticker": "ALB", "shares":  +2,   "buy_price":  95.43, "buy_date": "2024-11-15"},
 
    # BYDDY  ← HK-Aktie als ADR, Kurs in USD ≠ EUR-Kaufpreis
    {"ticker": "BYDDY", "shares":  +7,   "buy_price":  31.91, "buy_date": "2024-12-04"},
 
    # DOCU — Verkauf
    {"ticker": "DOCU", "shares":  -7,   "buy_price":  94.16, "buy_date": "2024-12-10"},
 
    # IWDA.AS
    {"ticker": "IWDA.AS", "shares": +0.942,    "buy_price": 106.05, "buy_date": "2024-12-13"},
    {"ticker": "IWDA.AS", "shares": +0.961,    "buy_price": 104.03, "buy_date": "2025-01-13"},
 
    # OKTA
    {"ticker": "OKTA", "shares":  +5,   "buy_price":  82.89, "buy_date": "2025-01-16"},
 
    # CRM
    {"ticker": "CRM",  "shares":  +1,   "buy_price": 313.35, "buy_date": "2025-01-16"},
 
    # LAC
    {"ticker": "LAC",  "shares": +171,  "buy_price":   2.91, "buy_date": "2025-02-11"},
 
    # IWDA.AS
    {"ticker": "IWDA.AS", "shares": +0.928,    "buy_price": 107.72, "buy_date": "2025-02-13"},
 
    # HIMS
    {"ticker": "HIMS", "shares":  +2,   "buy_price":  57.42, "buy_date": "2025-02-18"},
 
    # ALB
    {"ticker": "ALB", "shares":  +2,   "buy_price":  77.23, "buy_date": "2025-02-18"},
 
    # BYDDY
    {"ticker": "BYDDY", "shares":  +5,   "buy_price":  47.43, "buy_date": "2025-02-25"},
 
    # HIMS
    {"ticker": "HIMS", "shares":  +4,   "buy_price":  35.80, "buy_date": "2025-02-25"},
    {"ticker": "HIMS", "shares":  +3,   "buy_price":  33.98, "buy_date": "2025-03-11"},
 
    # IWDA.AS
    {"ticker": "IWDA.AS", "shares": +1.024,    "buy_price":  97.64, "buy_date": "2025-03-13"},
 
    # OKTA
    {"ticker": "OKTA", "shares":  +1,   "buy_price": 103.44, "buy_date": "2025-03-19"},
 
    # ALB
    {"ticker": "ALB", "shares":  +1,   "buy_price":  60.83, "buy_date": "2025-04-03"},
 
    # IWDA.AS
    {"ticker": "IWDA.AS", "shares": +1.106,    "buy_price":  90.39, "buy_date": "2025-04-14"},
 
    # ALB
    {"ticker": "ALB", "shares":  +3,   "buy_price":  55.54, "buy_date": "2025-05-12"},
 
    # IWDA.AS
    {"ticker": "IWDA.AS", "shares": +1.002,    "buy_price":  99.74, "buy_date": "2025-05-13"},
 
    # PLS.AX  ← AUD-Ticker
    {"ticker": "PLS.AX", "shares": +368,  "buy_price":   0.81, "buy_date": "2025-05-19"},
 
    # ALB
    {"ticker": "ALB", "shares":  +3,   "buy_price":  52.35, "buy_date": "2025-05-20"},
 
    # PLS.AX
    {"ticker": "PLS.AX", "shares":  +64,  "buy_price":   0.77, "buy_date": "2025-06-05"},
 
    # ALB
    {"ticker": "ALB", "shares":  +2,   "buy_price":  54.49, "buy_date": "2025-06-09"},
 
    # PLS.AX
    {"ticker": "PLS.AX", "shares": +129,  "buy_price":   0.77, "buy_date": "2025-06-10"},
 
    # IWDA.AS
    {"ticker": "IWDA.AS", "shares": +1.009,    "buy_price":  99.10, "buy_date": "2025-06-13"},
 
    # ALB
    {"ticker": "ALB", "shares":  +2,   "buy_price":  52.80, "buy_date": "2025-06-17"},
 
    # HIMS
    {"ticker": "HIMS", "shares":  +2,   "buy_price":  55.94, "buy_date": "2025-06-20"},
    {"ticker": "HIMS", "shares":  +5,   "buy_price":  37.79, "buy_date": "2025-06-24"},
    {"ticker": "HIMS", "shares":  +2,   "buy_price":  38.39, "buy_date": "2025-06-26"},
 
    # ALB
    {"ticker": "ALB", "shares":  +1,   "buy_price":  55.37, "buy_date": "2025-06-27"},
    {"ticker": "ALB", "shares":  +1,   "buy_price":  55.73, "buy_date": "2025-07-02"},
    {"ticker": "ALB", "shares":  +1,   "buy_price":  61.93, "buy_date": "2025-07-10"},
 
    # IWDA.AS
    {"ticker": "IWDA.AS", "shares": +0.988,    "buy_price": 101.18, "buy_date": "2025-07-14"},
 
    # ALB
    {"ticker": "ALB", "shares":  +1,   "buy_price":  62.70, "buy_date": "2025-07-14"},
 
    # PLS.AX
    {"ticker": "PLS.AX", "shares":  +98,  "buy_price":   1.01, "buy_date": "2025-07-21"},
    {"ticker": "PLS.AX", "shares":  +46,  "buy_price":   1.06, "buy_date": "2025-07-22"},
 
    # ALB
    {"ticker": "ALB", "shares":  +2,   "buy_price":  61.89, "buy_date": "2025-07-29"},
 
    # HIMS
    {"ticker": "HIMS", "shares":  +5,   "buy_price":  47.82, "buy_date": "2025-08-05"},
 
    # UNH
    {"ticker": "UNH",  "shares":  +2,   "buy_price": 224.30, "buy_date": "2025-08-12"},
 
    # IWDA.AS
    {"ticker": "IWDA.AS", "shares": +0.958, "buy_price": 104.36, "buy_date": "2025-08-13"},
 
    # TTD
    {"ticker": "TTD",  "shares":  +8,   "buy_price":  45.88, "buy_date": "2025-08-13"},
 
    # HIMS
    {"ticker": "HIMS", "shares":  +5,   "buy_price":  38.42, "buy_date": "2025-08-18"},
 
    # CRM
    {"ticker": "CRM",  "shares":  +2,   "buy_price": 211.10, "buy_date": "2025-08-20"},
 
    # UPST
    {"ticker": "UPST", "shares":  +5,   "buy_price":  61.09, "buy_date": "2025-08-25"},
 
    # MNDY
    {"ticker": "MNDY", "shares":  +3,   "buy_price": 153.00, "buy_date": "2025-08-27"},
 
    # ALB
    {"ticker": "ALB", "shares":  +3,   "buy_price":  64.05, "buy_date": "2025-09-09"},
 
    # SEDG
    {"ticker": "SEDG", "shares": +10,   "buy_price":  28.53, "buy_date": "2025-09-09"},
 
    # BYDDY
    {"ticker": "BYDDY", "shares": +17,   "buy_price":  11.54, "buy_date": "2025-09-09"},
 
    # JD
    {"ticker": "JD",   "shares": +10,   "buy_price":  28.20, "buy_date": "2025-09-09"},
 
    # TTD
    {"ticker": "TTD",  "shares": +13,   "buy_price":  41.95, "buy_date": "2025-09-10"},
 
    # UPST
    {"ticker": "UPST", "shares":  +2,   "buy_price":  54.98, "buy_date": "2025-09-10"},
 
    # NIO
    {"ticker": "NIO",  "shares": +61,   "buy_price":   4.91, "buy_date": "2025-09-10"},
 
    # IWDA.AS
    {"ticker": "IWDA.AS", "shares": +0.940,    "buy_price": 106.44, "buy_date": "2025-09-15"},
 
    # PAYC — Verkauf
    {"ticker": "PAYC", "shares":  -2,   "buy_price": 179.70, "buy_date": "2025-09-16"},
 
    # TTD
    {"ticker": "TTD",  "shares":  +5,   "buy_price":  38.85, "buy_date": "2025-09-22"},
    {"ticker": "TTD",  "shares": +10,   "buy_price":  37.35, "buy_date": "2025-09-22"},
 
    # UNH — Verkauf
    {"ticker": "UNH",  "shares":  -1,   "buy_price": 283.30, "buy_date": "2025-09-22"},
 
    # DOCU — Verkauf
    {"ticker": "DOCU", "shares":  -3,   "buy_price":  71.37, "buy_date": "2025-09-22"},
 
    # LAC — Verkauf
    {"ticker": "LAC",  "shares": -49,   "buy_price":   4.07, "buy_date": "2025-09-24"},
 
    # RIVN
    {"ticker": "RIVN", "shares": +22,   "buy_price":  13.28, "buy_date": "2025-09-24"},
 
    # UPST
    {"ticker": "UPST", "shares":  -7,   "buy_price":  49.70, "buy_date": "2025-09-25"},
    {"ticker": "UPST", "shares":  +5,   "buy_price":  45.38, "buy_date": "2025-09-29"},
 
    # SMCI
    {"ticker": "SMCI", "shares":  +7,   "buy_price":  40.38, "buy_date": "2025-09-30"},
 
    # PAYC
    {"ticker": "PAYC", "shares":  +1,   "buy_price": 174.95, "buy_date": "2025-10-01"},
 
    # MNDY
    {"ticker": "MNDY", "shares":  +1,   "buy_price": 162.85, "buy_date": "2025-10-01"},
 
    # DOCU
    {"ticker": "DOCU", "shares":  +2,   "buy_price":  60.01, "buy_date": "2025-10-01"},
 
    # JD
    {"ticker": "JD",   "shares":  +7,   "buy_price":  30.30, "buy_date": "2025-10-03"},
 
    # SOFI
    {"ticker": "SOFI", "shares":  +5,   "buy_price":  21.88, "buy_date": "2025-10-06"},
 
    # TTD — Verkauf
    {"ticker": "TTD",  "shares": -18,   "buy_price":  46.21, "buy_date": "2025-10-07"},
 
    # PLS.AX — Verkauf
    {"ticker": "PLS.AX", "shares": -204,  "buy_price":   1.47, "buy_date": "2025-10-07"},
 
    # SMCI — Verkauf
    {"ticker": "SMCI", "shares":  -3,   "buy_price":  48.06, "buy_date": "2025-10-07"},
 
    # UNH — Verkauf
    {"ticker": "UNH",  "shares":  -1,   "buy_price": 321.30, "buy_date": "2025-10-08"},
 
    # ALB — Verkauf
    {"ticker": "ALB", "shares": -15,   "buy_price":  78.01, "buy_date": "2025-10-08"},
 
    # TTD
    {"ticker": "TTD",  "shares":  +4,   "buy_price":  45.06, "buy_date": "2025-10-10"},
 
    # PAYC
    {"ticker": "PAYC", "shares":  +1,   "buy_price": 172.40, "buy_date": "2025-10-10"},
 
    # DOCU
    {"ticker": "DOCU", "shares":  +4,   "buy_price":  59.41, "buy_date": "2025-10-10"},
 
    # SMCI
    {"ticker": "SMCI", "shares":  +1,   "buy_price":  48.22, "buy_date": "2025-10-10"},
 
    # IWDA.AS
    {"ticker": "IWDA.AS", "shares": +0.922,    "buy_price": 108.45, "buy_date": "2025-10-13"},
 
    # ALB — Verkauf
    {"ticker": "ALB", "shares": -12,   "buy_price":  79.26, "buy_date": "2025-10-14"},
 
    # OKTA — Verkauf
    {"ticker": "OKTA", "shares":  -3,   "buy_price":  76.74, "buy_date": "2025-10-15"},
 
    # MP
    {"ticker": "MP",   "shares":  +3,   "buy_price":  80.00, "buy_date": "2025-10-15"},
 
    # MNDY
    {"ticker": "MNDY", "shares":  +1,   "buy_price": 151.75, "buy_date": "2025-10-16"},
 
    # SOFI
    {"ticker": "SOFI", "shares":  +4,   "buy_price":  22.70, "buy_date": "2025-10-16"},
 
    # RIVN
    {"ticker": "RIVN", "shares": +17,   "buy_price":  11.00, "buy_date": "2025-10-16"},
 
    # AI
    {"ticker": "AI",   "shares":  +6,   "buy_price":  16.48, "buy_date": "2025-10-16"},
 
    # NIO
    {"ticker": "NIO",  "shares": +64,   "buy_price":   5.29, "buy_date": "2025-10-16"},
 
    # ROK
    {"ticker": "ROK",  "shares":  +1,   "buy_price": 305.20, "buy_date": "2025-10-16"},
 
    # HIMS
    {"ticker": "HIMS", "shares":  +3,   "buy_price":  45.42, "buy_date": "2025-10-17"},
 
    # ALB
    {"ticker": "ALB", "shares": +15,   "buy_price":  77.00, "buy_date": "2025-10-17"},
 
    # PLS.AX
    {"ticker": "PLS.AX", "shares": +200,  "buy_price":   1.40, "buy_date": "2025-10-17"},
 
    # ENPH
    {"ticker": "ENPH", "shares":  +6,   "buy_price":  31.44, "buy_date": "2025-10-17"},
 
    # ORCL
    {"ticker": "ORCL", "shares":  +1,   "buy_price": 238.45, "buy_date": "2025-10-20"},
 
    # CRWV
    {"ticker": "CRWV", "shares":  +2,   "buy_price": 105.00, "buy_date": "2025-10-21"},
 
    # PATH
    {"ticker": "PATH", "shares":  +7,   "buy_price":  13.46, "buy_date": "2025-10-22"},
 
    # SMCI
    {"ticker": "SMCI", "shares":  +3,   "buy_price":  42.00, "buy_date": "2025-10-23"},
 
    # NIO — Verkauf
    {"ticker": "NIO",  "shares": -25,   "buy_price":   5.94, "buy_date": "2025-10-24"},
 
    # PLS.AX — Verkauf
    {"ticker": "PLS.AX", "shares": -225,  "buy_price":   1.78, "buy_date": "2025-10-24"},
    {"ticker": "PLS.AX", "shares": +111,  "buy_price":   1.70, "buy_date": "2025-10-27"},
    {"ticker": "PLS.AX", "shares": +111,  "buy_price":   1.75, "buy_date": "2025-10-27"},
 
    # MP
    {"ticker": "MP",   "shares":  +1,   "buy_price":  54.60, "buy_date": "2025-10-27"},
    {"ticker": "MP",   "shares":  +1,   "buy_price":  56.00, "buy_date": "2025-10-28"},
 
    # HIMS
    {"ticker": "HIMS", "shares":  +4,   "buy_price":  40.61, "buy_date": "2025-10-28"},
 
    # SNPS
    {"ticker": "SNPS", "shares":  +1,   "buy_price": 384.00, "buy_date": "2025-10-28"},
 
    # PAYC
    {"ticker": "PAYC", "shares":  +1,   "buy_price": 161.60, "buy_date": "2025-10-29"},
 
    # ORCL
    {"ticker": "ORCL", "shares":  +1,   "buy_price": 222.00, "buy_date": "2025-10-30"},
    {"ticker": "ORCL", "shares":  +1,   "buy_price": 227.00, "buy_date": "2025-10-30"},
    {"ticker": "ORCL", "shares":  +1,   "buy_price": 229.75, "buy_date": "2025-10-30"},
 
    # BYDDY
    {"ticker": "BYDDY", "shares":  +8,   "buy_price":  11.18, "buy_date": "2025-10-31"},
 
    # HIMS
    {"ticker": "HIMS", "shares":  +3,   "buy_price":  38.97, "buy_date": "2025-11-03"},
 
    # PATH
    {"ticker": "PATH", "shares": +10,   "buy_price":  13.38, "buy_date": "2025-11-03"},
 
    # BYDDY
    {"ticker": "BYDDY", "shares":  +5,   "buy_price":  10.96, "buy_date": "2025-11-03"},
 
    # CRWV
    {"ticker": "CRWV", "shares":  +1,   "buy_price": 111.50, "buy_date": "2025-11-03"},
 
    # SEDG
    {"ticker": "SEDG", "shares":  +2,   "buy_price":  28.88, "buy_date": "2025-11-03"},
 
    # PATH
    {"ticker": "PATH", "shares":  +7,   "buy_price":  13.01, "buy_date": "2025-11-04"},
 
    # UPST
    {"ticker": "UPST", "shares":  +2,   "buy_price":  34.56, "buy_date": "2025-11-05"},
 
    # SMCI
    {"ticker": "SMCI", "shares":  +3,   "buy_price":  37.58, "buy_date": "2025-11-05"},
 
    # NIO — Verkauf
    {"ticker": "NIO",  "shares": -31,   "buy_price":   5.97, "buy_date": "2025-11-07"},
 
    # ORCL
    {"ticker": "ORCL", "shares":  +1,   "buy_price": 207.00, "buy_date": "2025-11-07"},
 
    # UNH
    {"ticker": "UNH",  "shares":  +1,   "buy_price": 277.25, "buy_date": "2025-11-10"},
 
    # SEDG — Verkauf
    {"ticker": "SEDG", "shares":  -6,   "buy_price":  37.60, "buy_date": "2025-11-10"},
 
    # RIVN — Verkauf
    {"ticker": "RIVN", "shares": -25,   "buy_price":  13.32, "buy_date": "2025-11-10"},
 
    # PAYC
    {"ticker": "PAYC", "shares":  +1,   "buy_price": 142.05, "buy_date": "2025-11-10"},
 
    # PATH — Verkauf
    {"ticker": "PATH", "shares": -10,   "buy_price":  12.23, "buy_date": "2025-11-10"},
 
    # CRWV
    {"ticker": "CRWV", "shares":  +1,   "buy_price":  94.00, "buy_date": "2025-11-10"},
    {"ticker": "CRWV", "shares":  +1,   "buy_price":  94.40, "buy_date": "2025-11-10"},
 
    # UPST
    {"ticker": "UPST", "shares":  +6,   "buy_price":  34.36, "buy_date": "2025-11-10"},
 
    # PLS.AX — Verkauf
    {"ticker": "PLS.AX", "shares": -171,  "buy_price":   1.75, "buy_date": "2025-11-10"},
 
    # ALB — Verkauf
    {"ticker": "ALB", "shares":  -5,   "buy_price":  87.36, "buy_date": "2025-11-11"},
 
    # HIMS
    {"ticker": "HIMS", "shares":  +3,   "buy_price":  33.21, "buy_date": "2025-11-12"},
 
    # SMCI
    {"ticker": "SMCI", "shares":  +6,   "buy_price":  33.25, "buy_date": "2025-11-12"},
 
    # ALB — Verkauf
    {"ticker": "ALB", "shares":  -2,   "buy_price":  93.84, "buy_date": "2025-11-12"},
 
    # JD — Verkauf
    {"ticker": "JD",   "shares":  -3,   "buy_price":  27.10, "buy_date": "2025-11-12"},
 
    # NIO — Verkauf
    {"ticker": "NIO",  "shares": -18,   "buy_price":   5.55, "buy_date": "2025-11-12"},
 
    # OKTA
    {"ticker": "OKTA", "shares":  +1,   "buy_price":  74.52, "buy_date": "2025-11-12"},
 
    # AI
    {"ticker": "AI",   "shares":  +7,   "buy_price":  13.33, "buy_date": "2025-11-12"},
 
    # IWDA.AS
    {"ticker": "IWDA.AS", "shares": +0.894,    "buy_price": 111.82, "buy_date": "2025-11-13"},
 
    # PLS.AX — Verkauf
    {"ticker": "PLS.AX", "shares":  -47,  "buy_price":   2.09, "buy_date": "2025-11-13"},
    {"ticker": "PLS.AX", "shares":  -96,  "buy_price":   2.08, "buy_date": "2025-11-13"},
 
    # SMCI
    {"ticker": "SMCI", "shares":  +3,   "buy_price":  29.57, "buy_date": "2025-11-14"},
 
    # JD
    {"ticker": "JD",   "shares":  +1,   "buy_price":  26.00, "buy_date": "2025-11-14"},
 
    # UNH
    {"ticker": "UNH",  "shares":  +1,   "buy_price": 283.90, "buy_date": "2025-11-14"},
 
    # SEDG
    {"ticker": "SEDG", "shares":  +3,   "buy_price":  31.30, "buy_date": "2025-11-14"},
 
    # ENPH
    {"ticker": "ENPH", "shares":  +4,   "buy_price":  24.81, "buy_date": "2025-11-14"},
 
    # BYDDY — Verkauf
    {"ticker": "BYDDY", "shares":  -4,   "buy_price":  11.09, "buy_date": "2025-11-14"},
 
    # TTD
    {"ticker": "TTD",  "shares":  +2,   "buy_price":  36.79, "buy_date": "2025-11-14"},
 
    # MNDY
    {"ticker": "MNDY", "shares":  +1,   "buy_price": 138.40, "buy_date": "2025-11-14"},
 
    # UPST
    {"ticker": "UPST", "shares":  +6,   "buy_price":  32.24, "buy_date": "2025-11-14"},
 
    # PLS.AX — Verkauf
    {"ticker": "PLS.AX", "shares":  -97,  "buy_price":   2.05, "buy_date": "2025-11-14"},
 
    # ALB — Verkauf
    {"ticker": "ALB", "shares":  -2,   "buy_price": 107.02, "buy_date": "2025-11-17"},
    {"ticker": "ALB", "shares":  +2,   "buy_price":  97.75, "buy_date": "2025-11-18"},
 
    # UUUU
    {"ticker": "UUUU", "shares": +16,   "buy_price":  12.20, "buy_date": "2025-11-20"},
 
    # CRWV
    {"ticker": "CRWV", "shares":  +3,   "buy_price":  64.80, "buy_date": "2025-11-20"},
 
    # SMCI
    {"ticker": "SMCI", "shares":  +3,   "buy_price":  30.41, "buy_date": "2025-11-20"},
 
    # PLS.AX — Verkauf
    {"ticker": "PLS.AX", "shares":  -91,  "buy_price":   2.37, "buy_date": "2025-11-20"},
 
    # ORCL — Verkauf
    {"ticker": "ORCL", "shares":  -2,   "buy_price": 203.25, "buy_date": "2025-11-20"},
 
    # CRWV
    {"ticker": "CRWV", "shares":  +1,   "buy_price":  78.00, "buy_date": "2025-12-09"},
 
    # SMCI
    {"ticker": "SMCI", "shares":  +1,   "buy_price":  30.02, "buy_date": "2025-12-09"},
 
    # IWDA.AS
    {"ticker": "IWDA.AS", "shares": +0.900,    "buy_price": 111.11, "buy_date": "2025-12-15"},
 
    # ALB — Verkauf
    {"ticker": "ALB", "shares":  -3,   "buy_price": 125.00, "buy_date": "2025-12-19"},
 
    # TTD
    {"ticker": "TTD",  "shares":  +4,   "buy_price":  31.68, "buy_date": "2025-12-19"},
 
    # ORCL
    {"ticker": "ORCL", "shares":  +1,   "buy_price": 162.82, "buy_date": "2025-12-19"},
 
    # PLS.AX — Verkauf
    {"ticker": "PLS.AX", "shares":  -67,  "buy_price":   2.21, "buy_date": "2025-12-19"},
 
    # HIMS
    {"ticker": "HIMS", "shares":  +5,   "buy_price":  29.92, "buy_date": "2025-12-19"},
 
    # NIO
    {"ticker": "NIO",  "shares": +23,   "buy_price":   4.26, "buy_date": "2025-12-19"},
 
    # AI
    {"ticker": "AI",   "shares":  +4,   "buy_price":  11.84, "buy_date": "2025-12-19"},
 
    # RIVN
    {"ticker": "RIVN", "shares":  +5,   "buy_price":  16.20, "buy_date": "2026-01-02"},
 
    # PAYC
    {"ticker": "PAYC", "shares":  +1,   "buy_price": 136.45, "buy_date": "2026-01-12"},
 
    # SMCI
    {"ticker": "SMCI", "shares":  +5,   "buy_price":  23.93, "buy_date": "2026-01-13"},
 
    # IWDA.AS
    {"ticker": "IWDA.AS", "shares": +0.876,    "buy_price": 114.17, "buy_date": "2026-01-13"},
 
    # RIVN
    {"ticker": "RIVN", "shares":  +5,   "buy_price":  14.86, "buy_date": "2026-01-14"},
 
    # UPS
    {"ticker": "UPS",  "shares":  +3,   "buy_price":  87.88, "buy_date": "2026-01-27"},
 
    # UUUU — Verkauf
    {"ticker": "UUUU", "shares":  -8,   "buy_price":  18.48, "buy_date": "2026-01-30"},
 
    # RIVN
    {"ticker": "RIVN", "shares":  +8,   "buy_price":  12.40, "buy_date": "2026-01-30"},
 
    # BYDDY
    {"ticker": "BYDDY", "shares": +14,   "buy_price":   9.72, "buy_date": "2026-02-02"},
 
    # CRWV
    {"ticker": "CRWV", "shares":  +2,   "buy_price":  71.60, "buy_date": "2026-02-03"},
 
    # UUUU — Verkauf
    {"ticker": "UUUU", "shares":  -4,   "buy_price":  16.77, "buy_date": "2026-02-06"},
 
    # ORCL
    {"ticker": "ORCL", "shares":  +1,   "buy_price": 119.18, "buy_date": "2026-02-06"},
 
    # HIMS
    {"ticker": "HIMS", "shares": +12,   "buy_price":  15.70, "buy_date": "2026-02-09"},
 
    # MNDY
    {"ticker": "MNDY", "shares":  +4,   "buy_price":  65.74, "buy_date": "2026-02-10"},
 
    # HIMS
    {"ticker": "HIMS", "shares":  +7,   "buy_price":  13.71, "buy_date": "2026-02-13"},
 
    # IWDA.AS
    {"ticker": "IWDA.AS", "shares": +0.894,    "buy_price": 111.87, "buy_date": "2026-02-13"},
 
    # GME
    {"ticker": "GME",  "shares": +10,   "buy_price":  19.82, "buy_date": "2026-02-17"},
 
    # HIMS
    {"ticker": "HIMS", "shares":  +4,   "buy_price":  19.44, "buy_date": "2026-03-09"},
 
    # RIVN
    {"ticker": "RIVN", "shares":  +7,   "buy_price":  13.18, "buy_date": "2026-03-13"},
 
    # NIO
    {"ticker": "NIO",  "shares": +25,   "buy_price":   5.08, "buy_date": "2026-03-13"},
 
    # OKTA — Verkauf
    {"ticker": "OKTA", "shares":  -1,   "buy_price":  68.45, "buy_date": "2026-03-13"},
 
    # DOCU
    {"ticker": "DOCU", "shares":  +2,   "buy_price":  40.60, "buy_date": "2026-03-13"},
 
    # PATH
    {"ticker": "PATH", "shares":  +9,   "buy_price":  10.01, "buy_date": "2026-03-13"},
 
    # ENPH
    {"ticker": "ENPH", "shares":  +5,   "buy_price":  39.19, "buy_date": "2026-03-13"},
    {"ticker": "ENPH", "shares":  -5,   "buy_price":  39.03, "buy_date": "2026-03-13"},
    {"ticker": "ENPH", "shares":  +5,   "buy_price":  39.13, "buy_date": "2026-03-13"},
 
    # MP — Verkauf
    {"ticker": "MP",   "shares":  -2,   "buy_price":  50.80, "buy_date": "2026-03-13"},
 
    # ALB — Verkauf
    {"ticker": "ALB", "shares":  -2,   "buy_price": 139.20, "buy_date": "2026-03-13"},
 
    # PLS.AX — Verkauf
    {"ticker": "PLS.AX", "shares":  -34,  "buy_price":   2.93, "buy_date": "2026-03-13"},
 
    # UNH — Verkauf
    {"ticker": "UNH",  "shares":  -1,   "buy_price": 245.30, "buy_date": "2026-03-13"},
 
    # SOFI
    {"ticker": "SOFI", "shares":  +1,   "buy_price":  15.60, "buy_date": "2026-03-13"},
 
    # AI
    {"ticker": "AI",   "shares": +12,   "buy_price":   7.86, "buy_date": "2026-03-13"},
 
    # IWDA.AS — Verkauf
    {"ticker": "IWDA.AS", "shares": -9.000,    "buy_price": 110.86, "buy_date": "2026-03-19"},
 
    # IWDA.AS
    {"ticker": "IWDA.AS", "shares": +0.891,    "buy_price": 112.17, "buy_date": "2026-03-13"},
 
    # PYPL
    {"ticker": "PYPL", "shares":  +3,   "buy_price":  39.74, "buy_date": "2026-03-16"},
 
    # MOH
    {"ticker": "MOH",  "shares":  +2,   "buy_price": 131.65, "buy_date": "2026-03-16"},
 
    # LAC
    {"ticker": "LAC",  "shares": +76,   "buy_price":   3.47, "buy_date": "2026-03-19"},
 
    # OSCR
    {"ticker": "OSCR", "shares":  +9,   "buy_price":  11.84, "buy_date": "2026-03-19"},
 
    # OKTA
    {"ticker": "OKTA", "shares":  +2,   "buy_price":  68.21, "buy_date": "2026-03-19"},
 
    # DOC
    {"ticker": "DOC",  "shares": +19,   "buy_price":  15.30, "buy_date": "2026-03-19"},
 
    # CRM
    {"ticker": "CRM",  "shares":  +1,   "buy_price": 168.86, "buy_date": "2026-03-19"},
 
    # DOCU
    {"ticker": "DOCU", "shares":  +2,   "buy_price":  42.94, "buy_date": "2026-03-19"},
 
    # PATH
    {"ticker": "PATH", "shares": +22,   "buy_price":  10.93, "buy_date": "2026-03-19"},
 
    # MNDY
    {"ticker": "MNDY", "shares":  +1,   "buy_price":  64.98, "buy_date": "2026-03-19"},
    {"ticker": "MNDY", "shares":  +4,   "buy_price":  64.98, "buy_date": "2026-03-19"},
 
    # SEDG
    {"ticker": "SEDG", "shares":  +2,   "buy_price":  38.69, "buy_date": "2026-03-19"},
 
    # ENPH
    {"ticker": "ENPH", "shares":  +2,   "buy_price":  37.54, "buy_date": "2026-03-19"},
 
    # MP — Verkauf
    {"ticker": "MP",   "shares":  -1,   "buy_price":  49.20, "buy_date": "2026-03-19"},
 
    # PFE
    {"ticker": "PFE",  "shares":  +6,   "buy_price":  23.70, "buy_date": "2026-03-19"},
 
    # JD
    {"ticker": "JD",   "shares":  +4,   "buy_price":  24.55, "buy_date": "2026-03-19"},
 
    # RIVN
    {"ticker": "RIVN", "shares":  +7,   "buy_price":  13.54, "buy_date": "2026-03-19"},
 
    # UPST
    {"ticker": "UPST", "shares":  +4,   "buy_price":  22.72, "buy_date": "2026-03-19"},
 
    # UUUU — Verkauf
    {"ticker": "UUUU", "shares":  -4,   "buy_price":  16.14, "buy_date": "2026-03-25"},
 
    # JD
    {"ticker": "JD",   "shares":  +3,   "buy_price":  25.55, "buy_date": "2026-03-25"},
 
    # IWDA.AS
    {"ticker": "IWDA.AS", "shares": +0.889,    "buy_price": 112.43, "buy_date": "2026-04-13"},
]

# Benchmark
BENCHMARK = "SPY"

# Tickers mit Währungsabweichung (Kurs in yfinance ≠ EUR-Kaufpreis)
FOREIGN_CURRENCY = {"IWDA.AS", "PLS.AX", "BYDDY"}


# ---HILFSFUNKTION: Nächsten Handelstag finden---

def next_trading_day(date_str: str, index: pd.DatetimeIndex) -> pd.Timestamp | None:
    """Gibt den ersten Handelstag ab date_str zurück, der im Index liegt."""
    dt = pd.Timestamp(date_str)
    valid = index[index >= dt]
    return valid[0] if len(valid) > 0 else None


# --- Daten laden---

def load_portfolio_data(portfolio: list) -> tuple:
    # Lädt Kursdaten für alle positionen + Benchmark
    tickers = list({t["ticker"] for t in portfolio}) + [BENCHMARK]

    earliest = min(
        datetime.strptime(t["buy_date"], "%Y-%m-%d")
        for t in portfolio
    )
    start = (earliest - timedelta(days=7)).strftime("%Y-%m-%d")
    print(f"Lade Kursdaten ab {start} für {len(tickers) -1} Ticker...")

    raw = yf.download(tickers, start=start, auto_adjust=True, progress=False)

    # Close Preise extrahieren - robust für 1 oder n ticker
    if isinstance(raw.columns, pd.MultiIndex):
        prices = raw["Close"].copy()
    else:
        prices = raw[["Close"]].copy()
        prices.columns = [tickers[0]]
    
    # Sicherheitsstellen dass alle Ticker als spalte vorhanden sind
    prices.columns = prices.columns.get_level_values(0) \
        if isinstance(prices.columns, pd.MultiIndex) else prices.columns
    
    prices = prices.ffill()

    # Aktuelle info 
    info = {}
    for ticker in {t["ticker"] for t in portfolio}:
        try:
            i = yf.Ticker(ticker).info
            info[ticker] = {
                "name": i.get("shortName", ticker),
                "sector": i.get("sector", "-"),
                "pe": i.get("trailingPE", None),
                "market_cap": i.get("marketCap", None)
            }
        except Exception:
            info[ticker] = {"name":ticker, "sector": "-",
                            "pe": None, "market_cap": None}
    
    return prices, info

# --- Positionen berechnen ---

def calculate_positions(portfolio: list, prices: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregiert alle Tranchen pro Ticker
    Kostenbasisi: gewichteter Durchschnitt ( Average Cost method)
    """
    from collections import defaultdict  # falls etwas noch nicht existiert, stürzt nicht ab sondern erstellt es
 
    agg = defaultdict(lambda: { 
        "shares": 0.0, "total_cost": 0.0,
        "first_buy": None, "last_buy": None
    })

    for t in sorted(portfolio, key=lambda x: x["buy_date"]):
        ticker = t["ticker"]
        s= t["shares"]
        p = t["buy_price"]
        d = t["buy_date"]

        if s >0: #Kauf
            agg[ticker]["shares"] += s
            agg[ticker]["total_cost"] += s * p
            if agg[ticker]["first_buy"] is None:
                agg[ticker]["first_buy"] = d
            agg[ticker]["last_buy"] = d
        else: # Verkauf - Kostenbasis anteilig reduzieren 
            held = agg[ticker]["shares"]
            if held > 0:
                ratio = min(abs(s) / held, 1.0)  # max 1.0 = 100%
                agg[ticker]["total_cost"] -= ratio * agg[ticker]["total_cost"]
            agg[ticker]["shares"] += s # s ist negativ 
    
    rows = []
    for ticker, pos in agg.items():
        net_shares = pos["shares"]
        if net_shares < 0.001 or ticker not in prices.columns:
            continue

        price_series = prices[ticker].dropna() # bei leeren zeilen, wird weitergemacht
        if len(price_series) <2:
            continue

        current_price = float(price_series.iloc[-1])
        avg_cost = pos["total_cost"] / net_shares if net_shares > 0 else 0 

        cost_basis = net_shares * avg_cost
        current_value = net_shares *current_price
        pnl_abs = current_value - cost_basis
        pnl_pct = (pnl_abs / cost_basis * 100) if cost_basis != 0 else 0 

        today_return = float(
            (price_series.iloc[-1] / price_series.iloc[-2] - 1) * 100
        )

        first_buy = pos["first_buy"] or pos["last_buy"] or "2024-01-01"
        days_held = (datetime.now() - datetime.strptime(first_buy, "%Y-%m-%d")).days

        currency_flag = " ⚠️" if ticker in FOREIGN_CURRENCY else ""

        rows.append({
            "Ticker":        ticker + currency_flag,
            "Shares":        round(net_shares, 4),
            "Ø Kaufpreis":   round(avg_cost, 2),
            "Kurs":          round(current_price, 2),
            "Kaufwert":      round(cost_basis, 2),
            "Aktuell":       round(current_value, 2),
            "PnL ($)":       round(pnl_abs, 2),
            "PnL (%)":       round(pnl_pct, 2),
            "Heute (%)":     round(today_return, 2),
            "Tage gehalten": days_held,
        })

    df = pd.DataFrame(rows)
    if df.empty:
        return df
    
    # Währungs Tickers aus gesamtgewicht heraushalten
    usd_mask = ~df["Ticker"].str.contains("⚠️")
    total_usd = df.loc[usd_mask, "Aktuell"].sum()
    df["Gewicht (%)"] = df.apply(
        lambda r: round(r["Aktuell"] / total_usd * 100, 1)
        if not "⚠️" in r["Ticker"] else "—", axis=1
    )

    return df.sort_values("Aktuell", ascending=False).reset_index(drop=True)


# --- QUITY CURVE TRANCHEN BASIERT ---
def build_equity_curve(portfolio: list, prices: pd.DataFrame) -> pd.DataFrame:
    """
    Berechnet die tägliche Portfolio Entwicklung korrekt:
    für jeden tag wird die tatsächlich gehaltene stückzahl verwendet 
    nicht rückwrikend die aktuelle gesamtzahl
    """
    all_dates = prices.index
    tickers = {t["ticker"] for t in portfolio if t["ticker"] in prices.columns}

    portfolio_value = pd.Series(0.0, index=all_dates)

    for ticker in tickers:
        # Alle Tranchen für diesen ticker nach datum sortiert 
        tranches = sorted(
            [t for t in portfolio if t["ticker"] == ticker],
            key=lambda x: x["buy_date"]
        )

        # Tägliche Positionen aufbauen 
        position = pd.Series(0.0, index=all_dates)

        for tranche in tranches:
            td = next_trading_day(tranche["buy_date"], all_dates)
            if td is None:
                continue
            # Ab diesem tag die stückzahl (pos oder neg ) addieren 
            position.loc[td:] += tranche["shares"]

        # Tageswert = position * Kurs
        price_series = prices[ticker].reindex(all_dates).ffill()
        ticker_value = (position * price_series).fillna(0)
        portfolio_value = portfolio_value + ticker_value

    # Erste datum mit portfolio wert >0
    start = portfolio_value[portfolio_value> 0].index[0]
    portfolio_value = portfolio_value.loc[start:]

    # Benchmark auf gleiches Startkapital normalisieren 
    bench = prices[BENCHMARK].reindex(portfolio_value.index).ffill().dropna()
    bench_norm = (bench / bench.iloc[0]) * portfolio_value.iloc[0]

    result = pd.DataFrame({
        "Portfolio": portfolio_value,
        "Benchmark": bench_norm,
    }).dropna()

    return result


# --- Risikokennzahlen ---

def calculate_risk_metrics(equity_curve: pd.DataFrame) -> dict:
    """ Berechnet CAGR, Sharpe, Max Drawdown, Volatilität, Beta, Alpha """
    metrics = {}

    for col in ["Portfolio", "Benchmark"]:
        series = equity_curve[col].dropna()
        returns = series.pct_change().dropna()

        years = max(len(series) / 252, 0.01)
        total_ret = series.iloc[-1] / series.iloc[0]
        cagr_val = (total_ret ** (1/years) - 1) * 100

        sharpe = (returns.mean() / returns.std() * np.sqrt(252)) \
        if returns.std() != 0 else 0 

        rolling_max = series.cummax()
        max_dd = ((series - rolling_max) / rolling_max).min() * 100
        vol = returns.std() * np.sqrt(252) * 100
        win_rate = (returns> 0).mean() * 100

        metrics[col] = {
            "Total Return (%)": round((total_ret - 1) * 100, 2),
            "CAGR (%)":         round(cagr_val, 2),
            "Sharpe Ratio":     round(sharpe, 2),
            "Max Drawdown (%)": round(max_dd, 2),
            "Volatilität (%)":  round(vol, 2),
            "Win Rate (%)":     round(win_rate, 1),
        }
    
    # Beta & Alpha 
    port_ret = equity_curve["Portfolio"].pct_change().dropna()
    bench_ret = equity_curve["Benchmark"].pct_change().dropna()
    aligned = pd.concat([port_ret, bench_ret], axis=1).dropna() # zusammenfügen von port und bench
    aligned.columns= ["port", "bench"]

    var = aligned["bench"].var()
    cov = aligned.cov().iloc[0, 1] # misst die kovarianz 
    beta = cov/var if var != 0 else 1.0

    alpha = (
        metrics["Portfolio"]["CAGR (%)"] - 
        (0.05 + beta * (metrics["Benchmark"]["CAGR (%)"] - 0.05))
    )

    metrics["Portfolio"]["Beta"] = round(beta,2)
    metrics["Portfolio"]["Alpha (%)"] = round(alpha,2)
    metrics["Benchmark"]["Beta"] = 1.0
    metrics["Benchmark"]["Alpha (%)"] = 0.0

    return metrics

# --- Charts ---

def plot_portfolio_overview(positions: pd.DataFrame) -> None:
    """Allokations-Pie + PnL Bar Chart (nur USD Positionen)"""
    usd = positions[~positions["Ticker"].str.contains("⚠️")].copy()

    # --- Pie: kleine Positionen (<2%) zu "Andere" gruppieren ---
    threshold = 2.0
    main    = usd[usd["Gewicht (%)"] >= threshold].copy()
    others  = usd[usd["Gewicht (%)"] <  threshold].copy()

    if not others.empty:
        other_row = pd.DataFrame([{
            "Ticker":      "Andere",
            "Aktuell":     others["Aktuell"].sum(),
            "PnL ($)":     others["PnL ($)"].sum(),
            "Gewicht (%)": others["Gewicht (%)"].sum(),
        }])
        pie_df = pd.concat([main, other_row], ignore_index=True)
    else:
        pie_df = main

    # --- Layout: Pie links, Bar rechts, mehr Höhe ---
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=["Portfolio Allokation (USD)", "PnL pro Position ($)"],
        specs=[[{"type": "pie"}, {"type": "bar"}]],
        column_widths=[0.35, 0.65],
    )

    # Pie Chart
    fig.add_trace(go.Pie(
        labels=pie_df["Ticker"],
        values=pie_df["Aktuell"].round(2),
        hole=0.4,
        textinfo="label+percent",
        textposition="outside",
        marker=dict(colors=[
            "#2563eb","#16a34a","#f59e0b","#8b5cf6","#ef4444",
            "#0891b2","#d97706","#7c3aed","#059669","#dc2626",
            "#0284c7","#65a30d","#ea580c","#9333ea","#b91c1c",
            "#94a3b8",  # Andere → grau
        ])
    ), row=1, col=1)

    # Bar Chart — horizontal, nach PnL sortiert
    bar_df = usd.sort_values("PnL ($)", ascending=True)
    colors = ["#16a34a" if v > 0 else "#ef4444" for v in bar_df["PnL ($)"]]

    fig.add_trace(go.Bar(
        y=bar_df["Ticker"],          # ← Y-Achse = Ticker (horizontal)
        x=bar_df["PnL ($)"],         # ← X-Achse = Wert
        orientation="h",             # ← horizontal
        marker_color=colors,
        text=[f"${v:+.0f}" for v in bar_df["PnL ($)"]],
        textposition="outside",
        name="PnL",
        cliponaxis=False,
    ), row=1, col=2)

    fig.update_layout(
        height=700,                  # mehr Höhe für viele Positionen
        template="plotly_white",
        showlegend=False,
        margin=dict(l=20, r=120, t=50, b=20),  # rechts Platz für Labels
    )

    # X-Achse des Bar Charts: Nulllinie + etwas Padding
    max_abs = bar_df["PnL ($)"].abs().max() * 1.3
    fig.update_xaxes(range=[-max_abs, max_abs], row=1, col=2)
    fig.update_xaxes(title_text="PnL ($)", row=1, col=2)

    fig.show()


def plot_equity_vs_benchmark(equity_curve: pd.DataFrame) -> None:
    """Portfolio vs. Benchmark als prozentuale Returns — vergleichbar trotz Einzahlungen"""
    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True,
        vertical_spacing=0.05, row_heights=[0.65, 0.35],
        subplot_titles=["Portfolio vs. Benchmark (SPY) — kumulativer Return (%)", "Drawdown (%)"]
    )

    colors = {"Portfolio": "#2563eb", "Benchmark": "#94a3b8"}

    for col in ["Portfolio", "Benchmark"]:
        series = equity_curve[col]

        # Auf 0% normalisieren — beide starten bei 0
        pct = ((series / series.iloc[0]) - 1) * 100

        fig.add_trace(go.Scatter(
            x=pct.index, y=pct.round(2), name=col,
            line=dict(color=colors[col], width=2 if col == "Portfolio" else 1.5)
        ), row=1, col=1)

        # Drawdown bleibt in %
        rolling_max = series.cummax()
        dd = ((series - rolling_max) / rolling_max * 100).round(2)
        fig.add_trace(go.Scatter(
            x=dd.index, y=dd, name=f"{col} DD",
            line=dict(color=colors[col], width=1, dash="dot"),
            fill="tozeroy", opacity=0.4, showlegend=False
        ), row=2, col=1)

    # Nulllinie einzeichnen
    fig.add_hline(y=0, line_dash="dash", line_color="black",
                  line_width=0.8, row=1, col=1)

    fig.update_layout(
        height=600, template="plotly_white",
        hovermode="x unified",
        legend=dict(orientation="h", y=1.02),
        margin=dict(l=0, r=0, t=50, b=0)
    )
    fig.update_yaxes(title_text="Return (%)", row=1, col=1)
    fig.update_yaxes(title_text="Drawdown (%)", row=2, col=1)
    fig.show()


def plot_daily_returns_heatmap(equity_curve: pd.DataFrame) -> None:
    """ Monatliche Returns als Heatmap"""
    returns = equity_curve["Portfolio"].pct_change().dropna()
    monthly = returns.resample("ME").apply(lambda x: (1+x).prod() - 1)*100 
    # prozentualen veränderungen in "Month End" und anwenden auf produkt(1+x) - anfangs 1

    monthly_df = monthly.to_frame("Return")
    monthly_df["Jahr"] = monthly_df.index.year
    monthly_df["Monat"] = monthly_df.index.month

    pivot = monthly_df.pivot(index = "Jahr", columns="Monat", values="Return")
    pivot.columns = ["Jan", "Feb","Mär","Apr","Mai","Jun",
                     "Jul","Aug","Sep","Okt","Nov","Dez"]

    fig = go.Figure(go.Heatmap(
        z=pivot.values.round(1),
        x=pivot.columns.tolist(),
        y=pivot.index.tolist(),
        colorscale=[
            [0.0, "#dc2626"], [0.4, "#fca5a5"],
            [0.5, "#f9fafb"], [0.6, "#86efac"],
            [1.0, "#16a34a"] 
        ],
        text=[[f"{v:.1f}%" if not np.isnan(v) else "" for v in row] for row in pivot.values],
        texttemplate="%{text}",
        textfont=dict(size=11),
        showscale=True, zmid=0
    ))

    fig.update_layout(
        title="Monatliche Portfolio Returns(%)",
        template="plotly_white", height=300,
        margin=dict(l=0,r=0, t=40, b=0)
    )
    fig.show()

# --- Terminal Report ---
def print_daily_report(positions: pd.DataFrame, metrics: dict) -> None:
    """Tägliches Terminal Report """
    usd= positions[~positions["Ticker"].str.contains("⚠️")]

    total_value = usd["Aktuell"].sum()
    total_cost = usd["Kaufwert"].sum()
    total_pnl = usd["PnL ($)"].sum()
    total_pnl_pct = (total_pnl / total_cost * 100) if total_cost != 0 else 0
    today_pnl = (usd["Heute (%)"] * usd["Aktuell"] /100).sum()

    print("\n" + "=" * 58)
    print(f"  PORTFOLIO REPORT — {datetime.now().strftime('%d.%m.%Y %H:%M')}")
    print("=" * 58)
    print(f"  Gesamtwert (USD):   ${total_value:>10,.2f}")
    print(f"  Investiert (USD):   ${total_cost:>10,.2f}")
    print(f"  Gesamt PnL:         ${total_pnl:>+10,.2f}  ({total_pnl_pct:+.2f}%)")
    print(f"  Heute:              ${today_pnl:>+10,.2f}")
    print("  ⚠️  IWDA.AS / PLS.AX / BYDDY: EUR/AUD-Kurs, nicht in PnL")
    print("=" * 58)
 
    print("\n  POSITIONEN")
    print(f"  {'Ticker':<12} {'Kurs':>8} {'PnL ($)':>10} "
          f"{'PnL (%)':>8} {'Heute':>7} {'Gewicht':>8}")
    print("  " + "-" * 57)

    for _, row in positions.iterrows(): # iterrows -> zeile für zeile
        # es werden index und row rausgegeben wobei index = _ -> das wird nicht angezeigt
        ps = "+" if row["PnL ($)"] >= 0 else "" # gewinne mit + sonst -
        ts = "+" if row["Heute (%)"] >= 0 else "" # gewinne mit + sonst -
        w = f"{row['Gewicht (%)']:>6.1f}%" if row["Gewicht (%)"] != "—" else "   -"
        print(
            f"  {row['Ticker']:<12}"
            f"  ${row['Kurs']:>7.2f}"
            f"  {ps}${row['PnL ($)']:>8.2f}"
            f"  {ps}{row['PnL (%)']:>6.2f}%"
            f"  {ts}{row['Heute (%)']:>5.2f}%"
            f"  {w}"
        )

    print("\n  RISIKOKENNZAHLEN (USD-Portfolio vs SPY)")
    print(f"  {'Kennzahl':<22} {'Portfolio':>12} {'SPY':>12}")
    print("  " + "-" * 48)
 
    port  = metrics["Portfolio"]
    bench = metrics["Benchmark"]
    for key in ["Total Return (%)", "CAGR (%)", "Sharpe Ratio",
                "Max Drawdown (%)", "Volatilität (%)", "Beta", "Alpha (%)"]:
        bval = bench.get(key, "—")
        print(f"  {key:<22} {port[key]:>12}  {bval:>11}")
 
    print("=" * 58)
 
 
# --- Main ---
if __name__ == "__main__":
    prices,info = load_portfolio_data(PORTFOLIO)
    positions = calculate_positions(PORTFOLIO, prices)
    equity_curve = build_equity_curve(PORTFOLIO, prices)
    metrics = calculate_risk_metrics(equity_curve)

    print_daily_report(positions, metrics)
    plot_portfolio_overview(positions)
    plot_equity_vs_benchmark(equity_curve)
    plot_daily_returns_heatmap(equity_curve)

    timestamp = datetime.now().strftime("%Y%m%d")
    fname = f"portfolio_report_{timestamp}.csv"
    positions.to_csv(fname, index=False)
    print(f"\nReport gespeichert: {fname}")
