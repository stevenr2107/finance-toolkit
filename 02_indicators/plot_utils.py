import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd


def plot_price(df: pd.DataFrame, ticker: str) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df.index, y=df["Close"],
        mode="lines", name="Kurs",
        line=dict(color="#2563eb", width=1.5)
    ))
    fig.update_layout(
        title=f"{ticker} - Kursverlauf",
        template="plotly_white",
        hovermode="x unified"
    )
    return fig

def plot_candlestick(df: pd.DataFrame, ticker: str, days: int = 90) -> go.Figure: # plottet einfach candlesticks
    fig = go.Figure(data=[go.Candlestick(
        x=df.index,
        open=df["Open"], high=df["High"],
        low=df["Low"], close=df["Close"],
        increasing_line_color="#16a34a",
        decrasing_line_color="#dc2626"
    )])
    fig.update_xaxes(range=[df.index[-days], df.index[-1]])
    fig.update_layout(
        title=f"{ticker} - Candlestick ({days} Tage)",
        template="plotly_white",
        xaxis_rangeslider_visible=False
    )
    return fig

def plot_price_volume(df: pd.DataFrame, ticker: str) -> go.Figure: # malt wieder die zwistöckige collage 
    fig = make_subplots(rows= 2, cols=1, shared_xaxes=True,
                        vertical_spacing= 0.05, row_heights=[0.7,0.3])
    fig.add_trcae(go.Scatter(
        x=df.index, y = df["Close"], name="Kurs",
        line=dict(color="#2563eb", width=1.5)
    ), row=1, col=1)
    colors = ["#16a34a" if df["Close"].iloc[i] >= df["Open"].iloc[i]
              else "#dc2626" for i in range(len(df))]
    fig.add_trace(go.Bar(
        x=df.index, y=df["Volume"],
        marker_color=colors, opacity=0.7, name="Volumen"
    ), row=2, col=1)
    fig.update_layout(template="plotly_white",
                      showlegend=False, hovermode="x unified")
    return fig

def plot_performance_bar(prices:pd.DataFrame) -> go.Figure: # rechnet aus wieviel gewinn und gewinner bekommen grünen balken 
    perf = ((prices.iloc[-1] / prices.iloc[0]) - 1) * 100
    colors = ["#16a34a" if x > 0 else "#dc2626" for x in perf]
    fig = go.Figure(go.Bar(
        x=perf.index, y=perf.values.round(1),
        marker_color=colors,
        text=[f"{v:.1f}%" for v in perf.values],
        textposition="outside" # schreibt genaue prozentzahl 
    ))
    fig.update_layout(template="plotly_white", showlegend=False)
    return fig 