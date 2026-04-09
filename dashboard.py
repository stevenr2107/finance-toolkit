import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Meine eigene library importieren
from indicators import sma, ema, rsi, macd

# --- Page config --- immer ganz oben vor allem anderen
st.set_page_config(
    page_title="Finance Dashboard",
    page_icon="📈",
    layout="wide"
)

st.title("📈 Finance Dashboard ")
st.caption("Built with Python - yfinance - Streamlit")

# --- Sidebar Inputs ---
with st.sidebar: # with bedeutet es wird nur auf die sidebar angewendet 
    st.header("Einstellungen")

    ticker = st.text_input(
        "Ticker Symbol",
        value="AAPL",
        help="z.B AAPL, MSFT, NVDA, BTC-USD"
    ).upper()

    period = st.selectbox(
        "Zeitraum",
        options=["3mo", "6mo", "1y", "2y", "5y"], # kürzel für yfinance
        index=2,
        format_func=lambda x: { # übersetzer für jedes x (3mo,...) mache 3 monate,...
            "3mo": "3 Monate",
            "6mo": "6 Monate",
            "1y":  "1 Jahr",
            "2y":  "2 Jahre",
            "5y":  "5 Jahre"
        }[x]
    )

    show_sma    = st.checkbox("SMA 20/50/200", value=True)
    show_ema    = st.checkbox("EMA 20", value=False)
    show_volume = st.checkbox("volumen anzeigen",value=True)

    st.divider()
    st.caption("Daten via yfinance - Kein Echtzeit-Feed")

# --- Daten laden ---
# @st.cache_data verhindert dass bei jeder Interaktion neu geladen wird
@st.cache_data(ttl=300)  # 5 Minuten Cache
def load_stock_data(ticker: str, period: str):
    df = yf.download(ticker, period=period, auto_adjust=True)
    df.columns = df.columns.get_level_values(0)
    info = yf.Ticker(ticker).info
    return df,info # df wird hier rausgegeben deshalb funktioniert close

#Laden mit Spinner
with st.spinner(f"Lade Daten fpr {ticker}..."): # dieser drehende kreis bei einer website wenn etwas lädt
    try:
        df,info = load_stock_data(ticker, period) # neues df
    except Exception as e:
        st.error(f"Ticker nicht gefunden: {e}")
        st.stop()

close = df["Close"].squeeze()


# --- KPI Cards ---
current_price  = close.iloc[-1]
prev_price = close.iloc[-2]
price_change = current_price - prev_price
pct_change = (price_change / prev_price) * 100

high_52w = close.max()
low_52w = close.min()
avg_vol = df["Volume"].mean()
current_rsi = rsi(close).iloc[-1]

col1, col2, col3, col4, col5 = st.columns(5)

col1.metric(
    label="Kurs",
    value = f"${current_price:.2f}",
    delta=f"{pct_change:.2f}"
)
col2.metric(
    label="52W Hoch",
    value = f"${high_52w:.2f}",
    delta=f"{((current_price/high_52w)-1)*100:.1f}% vom Hoch"
)
col3.metric(
    label="52W Tief",
    value = f"${low_52w:.2f}",
    delta=f"+{((current_price/low_52w)-1)*100:.1f}% vom Tief"
)
col4.metric(
    label="Ø Volumen",
    value = f"{avg_vol/1e6:.1f}M",
)
col5.metric(
    label="RSI (14)",
    value = f"{current_rsi:.1f}",
    delta="Überkauft" if current_rsi > 70 else ("Überverkauft" if current_rsi < 30 else "Neutral")
)


# --- Haupt Chart ---
st.subheader(f"{ticker} - Kursverlauf")

# Indikatoren berechnen 
df["SMA20"]  = sma(close, 20)
df["SMA50"]  = sma(close, 50)
df["SMA200"] = sma(close, 200)
df["EMA20"]  = ema(close, 20)


# Subplot setup
rows = 3 if show_volume else 2 
row_heights = [0.55, 0.22, 0.23] if show_volume else [0.7, 0.3]

fig = make_subplots(
    rows = rows, cols=1,
    shared_xaxes=True,
    vertical_spacing=0.03,
    row_heights=row_heights
)

# Kurs
fig.add_trace(go.Scatter(
    x=df.index, y=close,
    name = "Kurs",
    line=dict(color="#1e293b", width=1.5)
), row=1, col=1)

# Moving Averages
if show_sma:
    for window, color, dash in [
        (20, "#3b82f6", "dot"),
        (50, "#f59e0b", "solid"),
        (200, "#ef4444", "solid")
    ]:
        fig.add_trace(go.Scatter(
            x=df.index, y=df[f"SMA{window}"],
            name = f"SMA {window}",
            line=dict(color=color, width=1.2, dash=dash)
        ), row=1, col=1)

if show_ema:
    fig.add_trace(go.Scatter(
        x=df.index, y=df["EMA20"],
        name="EMA 20",
        line=dict(color="#8b5cf6", width=1.2, dash="dash")
    ),row=1, col=1)

# RSI Panel
df["RSI"] = rsi(close)

fig.add_trace(go.Scatter(
    x=df.index, y=df["RSI"],
    name="RSI",
    line=dict(color="#0891b2", width=1.5)
), row=2, col=1)

fig.add_hline(y=70, line_dash="dash", line_color="#ef4444",
              opacity=0.6, row=2, col=1)
fig.add_hline(y=30, line_dash="dash", line_color="#16a34a",
              opacity=0.6, row=2, col=1)

# Volumen Panel
if show_volume:
    vol_colors = [ # zeigt nur wann rot wann grün
        "#16a34a" if df["Close"].iloc[i] >= df["Open"].iloc[i]
        else "#ef4444"
        for i in range(len(df))
    ]
    fig.add_trace(go.Bar(
        x=df.index, y=df["Volume"],
        name="Volumen",
        marker_color=vol_colors,
        opacity=0.6
    ), row=3, col=1)

fig.update_layout(
    height=650,
    template="plotly_white",
    hovermode="x unified",
    legend=dict(orientation="h", y=1.02),
    margin=dict(l=0, r=0, t=30, b=0)
)

fig.update_yaxes(title_text="Kurs", row=1, col=1)
fig.update_yaxes(title_text="RSI",  row=2, col=1, range=[0, 100])
if show_volume:
    fig.update_yaxes(title_text="Vol", row=3, col=1)

st.plotly_chart(fig, use_container_width=True)

#--- macd expander ---
with st.expander("MACD anzeigen"):
    macd_df = macd(close)

    fig_macd = go.Figure()

    hist_colors = ["#16a34a" if v >= 0 else "#ef4444"
                   for v in macd_df["histogram"]]
    
    fig_macd.add_trace(go.Bar(
        x=df.index, y=macd_df["histogram"],
        name="Histogram", marker_color=hist_colors, opacity=0.6
    ))
    fig_macd.add_trace(go.Scatter(
        x=df.index, y=macd_df["macd"],
        name="MACD", line = dict(color="#2563eb", width=1.5)
    ))
    fig_macd.add_trace(go.Scatter(
        x=df.index, y=macd_df["signal"],
        name="Signal", line=dict(color="#f59e0b", width=1.5)
    ))

    fig_macd.update_layout(
        height=300,
        template="plotly_white",
        hovermode="x unified",
        margin=dict(l=0, r=0, t=10, b=0)
    )
    st.plotly_chart(fig_macd, use_container_width=True)

# --- Company info ---
with st.expander("Company Info"):
    col1, col2, col3, = st.columns(3)

    col1.metric("Sektor",   info.get("sector",     "-"))
    col2.metric("Marktcap",   f"${info.get("marketCap", 0)/1e9:.1f}B")
    col3.metric("P/E Ratio",   f"{info.get("trailingPE",   0):.1f}x")

    st.write(info.get("longBusinessSummary", "Keine Beschreibung vefügbar.")[:500] + "...")
    

# --- Rohdaten Download ---
st.divider()

col1, col2 = st.columns([3, 1])
col1.caption(f"Daten: {len(df)} Handelstage - Quelle Yahoo Finance ")

csv = df.round(2).to_csv().encode("utf-8")
col2.download_button(
    label="📥 CSV Download",
    data=csv,
    file_name= f"{ticker}_{period}.csv",
    mime="text/csv"
)
