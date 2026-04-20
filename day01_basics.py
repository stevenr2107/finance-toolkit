"""
Day 01 Market Data Basics
Lädt historische Kursdaten via yfinance und beredhnet
grundlegende Kennzahlen: 52W High/Low, Volumen 
"""


import yfinance as yf
import pandas as pd

# Aktie definieren
ticker = "AAPL"
stock = yf.Ticker(ticker)

# Historische Daten laden: letzte 1 Jahr, täglich
df = stock.history(period="1y")

# Was haben wir?
print("=== Daten Überblick ===")
print(df.head())           # erste 5 Zeilen
print("\n=== Shape ===")
print(df.shape)            # wie viele Zeilen/Spalten
print("\n=== Spalten ===")
print(df.columns.tolist()) # welche Daten gibts

# Basics ausrechnen
latest_price = df["Close"].iloc[-1]
high_52w = df["Close"].max()
low_52w = df["Close"].min()
avg_volume = df["Volume"].mean()

print(f"\n=== {ticker} Quick Stats ===")
print(f"Aktueller Kurs:    ${latest_price:.2f}")
print(f"52-Wochen Hoch:    ${high_52w:.2f}")
print(f"52-Wochen Tief:    ${low_52w:.2f}")
print(f"Ø Volumen:         {avg_volume:,.0f}")
