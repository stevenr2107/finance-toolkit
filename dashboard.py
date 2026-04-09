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

st.markdown("""
    <style>
        .block-container { padding-top: 1rem; padding-bottom: 1rem; }
        [data-testid="metric-container] {
            background: #f8fafc;
            border: 1px solid #e2e8f0;
            border-radius: 8px;
            padding: 12px;
        }
        div[data-testid="stTabs"] button{
            font-size: 14px;
        }
    </style>
""", unsafe_allow_html=True)

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

tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Chart & Indikatoren",
    "🏢 Peer Comparison", 
    "💰 DCF Bewertung",
    "📋 Rohdaten"
])

with tab1:

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
with tab2:
    st.subheader("Peer Comparison")

    # Peer Gruppen vordefiniert - erweiterbar 
    peer_groups = {
        "AAPL": ["AAPL", "MSFT", "GOOGL", "META", "AMZN"],
        "NVDA": ["NVDA", "AMD", "INTC", "QCOM", "TSM"],
        "JPM":  ["JPM", "BAC", "GS", "MS", "WFC"],
        "TSLA": ["TSLA", "GM", "F", "RIVN", "NIO"],
    }

    # Automatisch passende Peers vorschlagen 
    default_peers = peer_groups.get(ticker, [ticker])
    default_str = ", ".join(default_peers)

    peer_input = st.text_input(
        "Ticker für Vergleich (kommagetrennt)",
        value=default_str
    ) # strip macht leerzeichen weg 
    peers = [p.strip().upper() for p in peer_input.split(",") if p.strip()]

    @st.cache_data(ttl=300)
    def load_peers(peers:list, period: str):
        import yfinance as yf
        prices=yf.download(peers,period=period, auto_adjust=True)["Close"] # nur close nehmen 
        if len(peers) == 1:
            prices = prices.to_frame(name=peers[0]) # zwingt bei einer aktie es in eine Tabelle zu drücken
        prices.columns = prices.columns.get_level_values(0)
        return prices
    
    with st.spinner("Lade Peer-Daten..."):
        peer_prices = load_peers(peers, period)

    #--- Performance vergleich---
    st.markdown("### Performance Vergleich")

    perf_data = {}
    for t in peers:
        if t not in peer_prices.columns:
            continue
        p = peer_prices[t].dropna()
        if len(p) < 21: # 21 handelstage pro monat 
            continue
        perf_data[t] = {
            "1M":  ((p.iloc[-1] / p.iloc[-21])  - 1) * 100, # vergleicht heute mit vor 1 monat 
            "3M":  ((p.iloc[-1] / p.iloc[min(63,  len(p)-1)]) - 1) * 100,
            "6M":  ((p.iloc[-1] / p.iloc[min(126, len(p)-1)]) - 1) * 100,
            "1Y":  ((p.iloc[-1] / p.iloc[0])    - 1) * 100,
            "Volatilität": p.pct_change().std() * (252**0.5) * 100,
            "RSI": rsi(p).iloc[-1]
        }

    perf_df = pd.DataFrame(perf_data).T.round(2)

    # Farb Coding 
    def color_returns(val):
        if isinstance(val, float): # nur auf zahlen angewendet 
            color = "#16a34a" if val > 0 else "#dc2626"
            return f"color: {color}; font-weight: 500" # gibt es als farbe in der font weight aus 
        return "" # gibt nichts zurück wenn keine zahl 
    # lies in der nächsten zeile weiter = backslash
    styled = perf_df.style\
        .map(color_returns, subset=["1M", "3M", "6M", "1Y"])\
        .format("{:.2f}%", subset=["1M", "3M", "6M", "1Y", "Volatilität"])\
        .format("{:.1f})", subset=["RSI"]) # applymap malt die zahlen an

    st.dataframe(styled, use_container_width=True)

    #--- normalisierter Kursverlauf ---
    st.markdown("#### Kursverlauf normalisiert (Startpunkt = 100)")

    import plotly.graph_objects as go

    fig_peer = go.Figure()
    colors_list = ["#2563eb", "#dc2626", "#16a34a",
                   "#f59e0b", "#8b5cf6", "#0891b2"]
    
    for i,t in enumerate(peers): # man bekommt ticker und die stelle wo sie ist
        if t not in peer_prices.columns:
            continue
        p = peer_prices[t].dropna()
        normalized = (p / p.iloc[0]) * 100 # alle starten bei 100

        fig_peer.add_trace(go.Scatter(
            x=p.index,
            y=normalized,
            name=t,
            line=dict(color=colors_list[i % len(colors_list)], width=2)
        ))
    
    fig_peer.add_hline(
        y=100, line_dash="dot",
        line_color="#94a3b8", opacity=0.7
    )

    fig_peer.update_layout(
        height=420,
        template="plotly_white",
        hovermode="x unified",
        yaxis_title= "Normalisiert (Start = 100)",
        legend=dict(orientation="h", y=1.02), # waagerecht machen dict weil man mehrere commands ausgibt 
        margin=dict(l=0, r=0, t=30, b=0)
    )

    st.plotly_chart(fig_peer, use_container_width=True)

with tab3:
    st.subheader("DCF Bewertung - Discounted Cash Flow")

    st.caption(
        "Vereinfachtes DCF-Modell. Zahlen manuell anpassen"
        "für tiefere Analyse."
    )

    # Echte Daten aus yfinace als Startwerte
    shares = info.get("sharesOutstanding", 1e9)
    market_cap = info.get("marketCap",      0)
    free_cf= info.get("freeCashflow", 0 )
    current_p = close.iloc[-1]

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### Eingaben")

        fcf = st.number_input(
            "Free Cash Flow (aktuell, $B)",
            value=round(free_cf /1e9, 2) if free_cf else 5.0,
            step=0.5,
            format="%.2f"
        )

        growth_1 = st.slider(
            "Wachstum Phase 1 (Jahre 1-5, %)",
            min_value=0, max_value=40, value=15
        )        
        growth_2 = st.slider(
            "Wachstum Phase 2 (Jahre 6-10, %)",
            min_value=0, max_value=25, value=8
        )        
        terminal_growth = st.slider(
            "Terminales Wachstum ( nach Jahr 10, %)",
            min_value=0, max_value=5, value=3
        )        
        discount_rate = st.slider(
            "Discount Rate / WACC °(%)",
            min_value=5, max_value=20, value=10
        )

        shares_out = st.number_input(
            "Ausstehende Aktien (Milliarden)",
            value=round(shares / 1e9, 2) if shares else 15.0,
            step=0.1,
            format="%.2f"
        )
    
    with col2:
        st.markdown("#### Ergebnis")

        # DCF Berechnung
        r = discount_rate / 100
        g1 = growth_1 / 100
        g2 = growth_2 / 100
        gt = terminal_growth / 100

        cash_flows = []
        pv_flows = []
        current_fcf = fcf * 1e9 # zurück in absolute Zahlen

        for year in range(1, 11):
            g =g1 if year <= 5 else g2
            current_fcf *= (1+g) # multiplizieren und direkt zuweisen 
            pv = current_fcf / ((1+r) ** year)
            cash_flows.append(current_fcf /1e9) # append ist am ende dranhängen
            pv_flows.append(pv / 1e9)

        # Terminal Value
        terminal_value = (current_fcf * (1 + gt)) / (r -gt)
        pv_terminal = terminal_value / ((1+r) ** 10)

        total_pv = sum(pv_flows) + pv_terminal / 1e9
        intrinsic_value = (total_pv*1e9) / (shares_out * 1e9)

        upside = ((intrinsic_value / current_p) - 1) * 100

        # Ausgabe 
        if upside > 0:
            verdict ="🟢 Unterbewertet"
            color   = "#16a34a"
        else:
            verdict = "🔴 Überbewertet" # verdict = urteil
            color   = "#dc2626"

        st.metric("Intrinsic Value", f"${intrinsic_value:.2f}")
        st.metric("Aktueller Kurs", f"${current_p:.2f}")
        st.metric("Upside/Downside", f"{upside:+.1f}%")
        st.metric("Terminal Value", f"${pv_terminal/1e9:.1f}B")
        st.metric("Summe PV Cash Flows", f"${sum(pv_flows):.1f}B")

        st.markdown(
            f"<h3 style='color:{color}'>{verdict}</h3>",
            unsafe_allow_html=True
        )

    # Cash Flow Wasserfall Chart
    st.markdown("#### Projizierte Cash Flows (nominal, $B)")

    years = [f"Jahr {i}" for i in range(1, 11)]

    fig_dcf = go.Figure()

    fig_dcf.add_trace(go.Bar(
        x=years,
        y=[round(cf, 2) for cf in cash_flows],
        name="Projitierter FCF",
        marker_color="#3b82f6",
        text=[f"${cf:.1f}B" for cf in cash_flows],
        textposition = "outside"
    ))

    fig_dcf.add_trace(go.Bar(
        x=years,
        y=[round(pv,2) for pv in pv_flows],
        name="Present Value",
        marker_color="#93c5fd",
        opacity=0.8
    ))

    fig_dcf.update_layout(
        height=350,
        template="plotly_white",
        barmode="group",
        yaxis_title="$B",
        legend=dict(orientation="h", y=1.02),
        margin=dict(l=0, r=0, t=30, b=0)
    )

    st.plotly_chart(fig_dcf, use_container_width=True)

with tab4:
    st.subheader("Rohdaten")

    # Indikatoren hinzufügen 
    export_df = df.copy().round(4)
    export_df["SMA20"] = sma(close, 20).round(2)
    export_df["SMA50"] = sma(close, 50).round(2)
    export_df["SMA200"] = sma(close, 200).round(2)
    export_df["EMA20"] = ema(close, 20).round(2)
    export_df["RSI14"] = rsi(close).round(2)
    export_df["Return"] = close.pct_change().round(4)

    # Farb coding für returns 
    def highlight_returns(val):
        if isinstance(val, float) and abs(val) < 1:
            color = "#dcfce7" if val > 0 else "#fee2e2"
            return f"background-color: {color}"
        return ""
    st.dataframe(
        export_df.style.map(
            highlight_returns, subset=["Return"]
        ),
        use_container_width=True,
        height=400
    )

    col1, col2 = st.columns([3, 1])
    col1.caption(f"{len(export_df)} Handelstage - inkl. berechnete Indikatoren")

    csv = export_df.to_csv().encode("utf-8")
    col2.download_button(
        label="📥 CSV herunterladen",
        data=csv,
        file_name=f"{ticker}_{period}_full.csv",
        mime="text/csv"
    )
        