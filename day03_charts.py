import yfinance as yf
import pandas as pd # Tabellen
import plotly.graph_objects as go # malt diagramme länger aber man kann alles verändern
import plotly.express as px # malt diagramme mit einem befehl schnell
from plotly.subplots import make_subplots # baut collagen also mehrere aktien nebeneinander
import matplotlib.pyplot as plt # kann alles zeichnen sieht aber manchmal altmodisch aus 
import seaborn as sns # Macht die alten Bilder hübsch mit riesigen Farbpaletten 


# Daten laden 
ticker = "AAPL"
df = yf.download(ticker, period = "1y", auto_adjust=True)
df.columns = df.columns.get_level_values(0) # Multiindex flatten 

# ---- Chart 1: Interaktiver Linienchart ----
fig = go.Figure() # weiße Leinwand 

fig.add_trace(go.Scatter( # nimm Stift und hinterlasse spur 
    x=df.index,
    y=df["Close"],
    mode="lines",
    name="Kurs",
    line=dict(color="#2563eb", width=1.5)
))

fig.update_layout(
    title=f"{ticker} - Kursverlauf 1 Jahr",
    xaxis_title="Datum",
    yaxis_title="Kurs (USD)",
    hovermode="x unified",          # Zeigt alle Werte beim hovern 
    xaxis_rangeslider_visible=True, # Zoom Slider unten
    template="plotly_white"
)

fig.show() # öffnet im Browser interaktiv 



# --- Chart 2 Candlestick ---
fig2 = go.Figure(data=[go.Candlestick(

    x=df.index,
    open=df["Open"],
    high=df["High"],
    low=df["Low"],
    close=df["Close"],
    increasing_line_color="#16a34a",
    decreasing_line_color="#dc2626",
    name=ticker
)])

# Nur letzten 90 Tage anzeigen - sonst zu eng 
fig2.update_xaxes(range=[df.index[-90], df.index[-1]])

fig2.update_layout(
    title=f"{ticker} - Candlestick (90 Tage)",
    xaxis_title="Datum",
    yaxis_title="Kurs (USD)",
    xaxis_rangeslider_visible=False,
    template="plotly_white"
)

fig2.show()



# --- Chart 3 Kurs oben Volumen unten --- 
# jetzt collage 
fig3 = make_subplots(
    rows=2, cols=1,
    shared_xaxes=True,
    vertical_spacing=0.05, # platz zwischen den beiden charts
    row_heights=[0.7,0.3] # Kurs hat mehr Platz 
) 

# Kurs 
fig3.add_trace(go.Scatter(
    x = df.index, y=df["Close"],
    name="Kurs",
    line=dict(color="#2563eb", width=1.5)

), row=1, col=1)

# Volumen - grün/rot je nach Kursrichtung 
colors = ["#16a34a" if df["Close"].iloc[i] >= df["Open"].iloc[i]
          else "#dc2626"
          for i in range(len(df))]

fig3.add_trace(go.Bar( # male
     x=df.index,
     y=df["Volume"],
     name="Volumen",
     marker_color=colors,
     opacity=0.7
), row=2, col=1)

fig3.update_layout(
    title=f"{ticker} - Kurs & Volumen ",
    template = "plotly_white",
    showlegend=False,
    hovermode="x unified"
)

fig3.update_yaxes(title_text="Kurs (USD)", row=1, col=1)
fig3.update_yaxes(title_text="Volumen", row=2, col=1)

fig3.show()


# --- Chart 4: Heatmap mit Seaborn (Matplotlib) --- 
tickers = ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA"]
prices = yf.download(tickers, period="1y", auto_adjust=True)["Close"]
returns = prices.pct_change().dropna()
corr = returns.corr()

fig4, ax = plt.subplots(figsize=(9, 7)) # beide sachen haben die gleiche größe fig rahmen ax ist papier

sns.heatmap(
    corr,           # Werte 
    annot=True,     # Zahlen in den Zellen
    fmt=".2f",      # 2 nachkommastellen
    cmap="RdYlGn",  # Farbpalette
    vmin=-1, vmax= 1, # wann welche farbe 
    linewidths=0.5,
    ax=ax,
    annot_kws={"size":11} # schriftgröße
)

ax.set_title("Korrelationsmatrix - Magnificent 7 (1 Jahr)",
             fontsize=14, pad=15)
plt.tight_layout()
plt.savefig("correlations_heatmap.png", dpi=150, bbox_inches="tight")
plt.show()

print("Heatmap gespeichert als correlation_heatmap.png")


# --- Chart 5: 1-Jahres Performance Vergleich ---
perf_1y = ((prices.iloc[-1] / prices.iloc[0]) - 1)*100

colors = ["#16a34a" if x > 0 else "#dc2626" for x in perf_1y]

fig5 = go.Figure(go.Bar(
    x=perf_1y.index,
    y=perf_1y.values.round(1),
    marker_color=colors,
    text=[f"{v:.1f}%" for v in perf_1y.values],
    textposition="outside"
))

fig5.update_layout(
    title="1 Jahres Performance - Magnificent 7 (%)",
    yaxis_title="Return (%)",
    template="plotly_white",
    showlegend=False
)

fig5.show()