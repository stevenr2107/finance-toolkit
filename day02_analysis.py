import yfinance as yf
import pandas as pd

# Die Magnificent 7
tickers = ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA"]

# Alle auf einmal laden - yfinance macht das in einem API Call
df = yf.download(tickers, period="1y", auto_adjust=True)["Close"]

print("=== Kursdaten geladen ===")
print(df.tail())        # letzte 5 Tage
print(f"\nShape: {df.shape}")  # (Tage, Aktien)


# Daily returns berechnen 

# Prozentuale Tagesveränderung  
returns = df.pct_change().dropna() # wir holen aus dem close  download des letzten jahres  das delta von gestern und heute 
# dropna nimmmt leere zeilen raus, am ersten tag gibt es nichts zum vergleichen 

print ("\n === Daily Returns (letzte 5 Tage ) ===") # printed einfach die letzten 5 tage 
print (returns.tail().round(4))

# annualisierte Volatilität ( Risiko )
volatility = returns.std() * (252** 0.5) # 252 Handelstage pro jahr
print ("\n=== Annualisierte Volatilität ===")
print(volatility.sort_values(ascending=False).round(4))

# Korrelationsmatrix

# wie stark korrelieren die aktien?
correlation = returns.corr()

print("\n=== Korrelationsmatrix ===")
print(correlation.round(2))

# Welche zwei Aktien korrelieren am stärksten? 
corr_pairs = (correlation
              .unstack() # Liste statt Tabelle für Computer
              .sort_values(ascending = False) # absteigend sortieren 
              .drop_duplicates() # duplikate weg 
              )
# korrelationsmatrixen sind ziemlich gut um zb öl mit anderen zu vergleichen 
# Top Paare ( ohne self correlation = 1.0)
print("\n=== Stärkste Korrelationen ===")
print(corr_pairs[corr_pairs < 1.0].head(5)) # alle haben ca 0,5 das heißt sie korrelieren eher

# Performance Vergleich
# wie weit sind sie von 52 wochen hoch/tief entfernt  
high_52w = df.max()
low_52w = df.min()
current = df.iloc[-1]

distance_from_high = ((current - high_52w) / high_52w*100).round(2)
distance_from_low = ((current - low_52w) / low_52w*100).round(2)

summary = pd.DataFrame({
    "Kurs":     current.round(2),
    "52W Hoch":     high_52w.round(2),
    "% vom Hoch":     distance_from_high,
    "52W Tief":     low_52w.round(2),
    "% vom Tief":     distance_from_low
 })

print("\n=== 52 Wochen Übersicht ===")
print(summary.sort_values("% vom Hoch"))


def period_return(df, days):
    return((df.iloc[-1] / df.iloc[-days]) -1*100).round(2) # verglichen wird der tag heute und -days also bei 5 5 tage in die vergangenheit 
# durch da wir vergliechen zwischen heute und damals 

performance = pd.DataFrame({
    "1 Monat": ((df.iloc[-1] / df.iloc[-21]) -1).round(4) * 100,
    "3 Monate": ((df.iloc[-1] / df.iloc[-63]) -1).round(4) * 100,
    "6 Monate": ((df.iloc[-1] / df.iloc[-126]) -1).round(4) * 100,
    "1 Jahr": ((df.iloc[-1] / df.iloc[0]) -1).round(4) * 100
})

print("\n=== Performance Vergleich (%) ===")
print(performance.sort_values("1 Jahr", ascending=False))



# Wiederverwendbare Funktionen erstellen - immer am ende der Datei 

# neuer Command  neue Liste   hol daten 1y.         später in Dataframe 
def load_data(tickers: list, period: str = "1y") -> pd.DataFrame: 
    return yf.download(tickers, period=period, auto_adjust=True)["Close"]
     # aus yf werden ticker periode gedownloadet./  preise sauber / nur der Schlusskurs

def get_returns(df: pd.DataFrame) -> pd.DataFrame:
    return df.pct_change().dropna()

def get_volatility(returns: pd.DataFrame) -> pd.Series:
    return (returns.std() * (252 ** 0.5)).sort_values(ascending=False)

def get_correlation(returns: pd.DataFrame) -> pd.DataFrame:
    return returns.corr().round(2)

def get_performance_table(df: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame({
    "1M": ((df.iloc[-1] / df.iloc[-21]) -1).round(4) * 100,
    "3M": ((df.iloc[-1] / df.iloc[-63]) -1).round(4) * 100,
    "6M": ((df.iloc[-1] / df.iloc[-126]) -1).round(4) * 100,
    "1Y": ((df.iloc[-1] / df.iloc[0]) -1).round(4) * 100
    })


# Main Block - So ruft man es auf 
if __name__ == "__main__": 
    # name = main heißt wenn genau hier auf play gedrückt wird, wird ausgeführt 
    # wenn man das programm wo anders einfügt, wird es erstmal nicht ausgeführt 
    tickers = ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA"]

    df     = load_data(tickers) # die preise runterladen 
    returns = get_returns(df) # täglichen gewinne ausrechnen


    print(get_volatility(returns))
    print(get_correlation(returns))
    print(get_performance_table(df))