"""
Day 10 — Portfolio Tracker
Vollständiger Portfolio-Tracker mit PnL, Benchmark-Vergleich,
Risikokennzahlen und täglichem Performance-Report.
"""

import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings("ignore")


# --- Dein Portfolio definieren ---
# Format: Ticker → (Anzahl Aktien, Kaufpreis, Kaufdatum)
PORTFOLIO = {
    "AAPL":  {"shares": 10,  "buy_price": 175.00, "buy_date": "2024-01-15"},
    "NVDA":  {"shares": 5,   "buy_price": 495.00, "buy_date": "2024-02-01"}, # funktioniert nicht, da man stock splits hat
    "MSFT":  {"shares": 8,   "buy_price": 375.00, "buy_date": "2024-01-20"},
    "GOOGL": {"shares": 12,  "buy_price": 140.00, "buy_date": "2024-03-01"},
    "META":  {"shares": 6,   "buy_price": 485.00, "buy_date": "2024-02-15"},
}

# Benchmark für Vergleich
BENCHMARK = "SPY"

def load_portfolio_data(portfolio: dict) -> tuple:
    """
    Lädt Kursdaten für alle Positionen + Benchmark.
    Gibt Preise und aktuelle Info zurück.
    """
    tickers = list(portfolio.keys()) + [BENCHMARK]

    # Frühestes Kaufdatum finden
    earliest = min(
        datetime.strptime(v["buy_date"], "%Y-%m-%d")
        for v in portfolio.values()
    )
    start = (earliest - timedelta(days=5)).strftime("%Y-%m-%d")

    print(f"Lade Daten ab {start}...")

    # Alle Kurse auf einmal
    prices = yf.download(
        tickers, start=start,
        auto_adjust=True, progress=False
    )["Close"]

    if len(tickers) == 1:
        prices = prices.to_frame(name=tickers[0])
    prices.columns = prices.columns.get_level_values(0)

    # Aktuelle Info für jede Position
    info = {}
    for ticker in portfolio:
        try:
            t = yf.Ticker(ticker)
            i = t.info
            info[ticker] = {
                "name":     i.get("shortName",   ticker),
                "sector":   i.get("sector",       "—"),
                "pe":       i.get("trailingPE",   None),
                "market_cap": i.get("marketCap",  None),
            }
        except Exception:
            info[ticker] = {"name": ticker, "sector": "—",
                            "pe": None, "market_cap": None}

    return prices, info

def calculate_positions(portfolio: dict,
                        prices: pd.DataFrame) -> pd.DataFrame:
    """
    Berechnet für jede Position:
    Kaufwert, aktueller Wert, PnL absolut + prozentual.
    """
    rows = []

    for ticker, pos in portfolio.items():
        if ticker not in prices.columns:
            continue

        shares     = pos["shares"]
        buy_price  = pos["buy_price"]
        buy_date   = pos["buy_date"]

        # Aktueller Kurs
        current_price = prices[ticker].dropna().iloc[-1]

        # Werte
        cost_basis    = shares * buy_price
        current_value = shares * current_price
        pnl_abs       = current_value - cost_basis
        pnl_pct       = (pnl_abs / cost_basis) * 100

        # Heutiger Return
        today_return  = (
            (prices[ticker].iloc[-1] / prices[ticker].iloc[-2]) - 1
        ) * 100

        # Tage gehalten
        buy_dt     = datetime.strptime(buy_date, "%Y-%m-%d")
        days_held  = (datetime.now() - buy_dt).days

        rows.append({
            "Ticker":        ticker,
            "Shares":        shares,
            "Kaufpreis":     round(buy_price, 2),
            "Kurs":          round(float(current_price), 2),
            "Kaufwert":      round(cost_basis, 2),
            "Aktuell":       round(float(current_value), 2),
            "PnL ($)":       round(float(pnl_abs), 2),
            "PnL (%)":       round(float(pnl_pct), 2),
            "Heute (%)":     round(float(today_return), 2),
            "Tage gehalten": days_held,
        })

    df = pd.DataFrame(rows)

    # Gewichtung im Portfolio
    total = df["Aktuell"].sum()
    df["Gewicht (%)"] = (df["Aktuell"] / total * 100).round(1)

    return df.sort_values("Aktuell", ascending=False).reset_index(drop=True)

def build_equity_curve(portfolio: dict,
                       prices: pd.DataFrame,
                       capital: float = None) -> pd.DataFrame:
    """
    Baut die tägliche Portfolio-Equity-Curve.
    Vergleicht mit Benchmark (SPY) normalisiert auf denselben Startkapital.
    """
    curves = {}

    for ticker, pos in portfolio.items():
        if ticker not in prices.columns:
            continue

        buy_date = pos["buy_date"]
        shares   = pos["shares"]

        # Nur ab Kaufdatum
        price_series = prices[ticker].loc[buy_date:].dropna()
        curves[ticker] = price_series * shares

    # Portfolio-Gesamtwert pro Tag
    portfolio_df = pd.DataFrame(curves).dropna(how="all").fillna(method="ffill")
    portfolio_value = portfolio_df.sum(axis=1)

    # Benchmark normalisieren auf Portfolio-Startwert
    bench = prices[BENCHMARK].loc[portfolio_value.index[0]:].dropna()
    bench_normalized = (bench / bench.iloc[0]) * portfolio_value.iloc[0]

    result = pd.DataFrame({
        "Portfolio":  portfolio_value,
        "Benchmark":  bench_normalized
    }).dropna()

    return result

def calculate_risk_metrics(equity_curve: pd.DataFrame) -> dict:
    """
    Berechnet alle wichtigen Risikokennzahlen für Portfolio und Benchmark.
    """
    metrics = {}

    for col in ["Portfolio", "Benchmark"]:
        series  = equity_curve[col].dropna()
        returns = series.pct_change().dropna()

        # CAGR
        years      = len(series) / 252
        total_ret  = series.iloc[-1] / series.iloc[0]
        cagr_val   = (total_ret ** (1 / max(years, 0.01)) - 1) * 100

        # Sharpe
        sharpe = (returns.mean() / returns.std()) * np.sqrt(252)

        # Max Drawdown
        rolling_max = series.cummax()
        drawdown    = (series - rolling_max) / rolling_max
        max_dd      = drawdown.min() * 100

        # Volatilität
        vol = returns.std() * np.sqrt(252) * 100

        # Win Rate
        win_rate = (returns > 0).mean() * 100

        # Beta vs SPY (nur für Portfolio)
        metrics[col] = {
            "Total Return (%)": round((total_ret - 1) * 100, 2),
            "CAGR (%)":         round(cagr_val, 2),
            "Sharpe Ratio":     round(sharpe, 2),
            "Max Drawdown (%)": round(max_dd, 2),
            "Volatilität (%)":  round(vol, 2),
            "Win Rate (%)":     round(win_rate, 1),
        }

    # Alpha: Portfolio Return - Benchmark Return
    port_ret  = equity_curve["Portfolio"].pct_change().dropna()
    bench_ret = equity_curve["Benchmark"].pct_change().dropna()

    aligned        = pd.concat([port_ret, bench_ret], axis=1).dropna()
    aligned.columns = ["port", "bench"]

    # Beta
    cov   = aligned.cov().iloc[0, 1]
    var   = aligned["bench"].var()
    beta  = cov / var if var != 0 else 1.0

    # Alpha (annualisiert)
    alpha = (
        metrics["Portfolio"]["CAGR (%)"] -
        (0.05 + beta * (metrics["Benchmark"]["CAGR (%)"] - 0.05))
    )

    metrics["Portfolio"]["Beta"]         = round(beta, 2)
    metrics["Portfolio"]["Alpha (%)"]    = round(alpha, 2)
    metrics["Benchmark"]["Beta"]         = 1.0
    metrics["Benchmark"]["Alpha (%)"]    = 0.0

    return metrics

def plot_portfolio_overview(positions: pd.DataFrame) -> None:
    """Allokations-Pie + PnL Bar Chart nebeneinander."""

    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=["Portfolio Allokation", "PnL pro Position"],
        specs=[[{"type": "pie"}, {"type": "bar"}]]
    )

    # Pie Chart
    fig.add_trace(go.Pie(
        labels=positions["Ticker"],
        values=positions["Aktuell"],
        hole=0.4,
        textinfo="label+percent",
        marker=dict(colors=[
            "#2563eb", "#16a34a", "#f59e0b",
            "#8b5cf6", "#ef4444", "#0891b2"
        ])
    ), row=1, col=1)

    # PnL Bar
    colors = ["#16a34a" if v > 0 else "#ef4444"
              for v in positions["PnL ($)"]]

    fig.add_trace(go.Bar(
        x=positions["Ticker"],
        y=positions["PnL ($)"],
        marker_color=colors,
        text=[f"${v:+.0f}" for v in positions["PnL ($)"]],
        textposition="outside",
        name="PnL"
    ), row=1, col=2)

    fig.update_layout(
        height=400,
        template="plotly_white",
        showlegend=False,
        margin=dict(l=0, r=0, t=40, b=0),
        annotations=[
            dict(
                x=0.98, y=0.98,          # Position: oben rechts
                xref="paper", yref="paper",
                xanchor="right", yanchor="top",
                text=(
                    "⚠️ <b>Warum Verlust bei NVDA?</b><br>"
                    "Der eingetragene Kaufpreis ($495) ist pre-split.<br>"
                    "yfinance liefert split-adjustierte Kurse (~$49.50).<br>"
                    "Fix: buy_price ÷ 10 eintragen <b>oder</b><br>"
                    "auto_adjust=False verwenden."
                ),
                showarrow=False,
                bgcolor="rgba(254, 243, 199, 0.95)",   # gelb
                bordercolor="#f59e0b",
                borderwidth=1.5,
                borderpad=8,
                font=dict(size=12, color="#1c1917"),
                align="left",
            )
        ]
    )

    fig.show()


def plot_equity_vs_benchmark(equity_curve: pd.DataFrame) -> None:
    """Portfolio vs. Benchmark + Drawdown."""

    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        row_heights=[0.65, 0.35],
        subplot_titles=["Portfolio vs. Benchmark", "Drawdown"]
    )

    colors = {"Portfolio": "#2563eb", "Benchmark": "#94a3b8"}

    for col in ["Portfolio", "Benchmark"]:
        series = equity_curve[col]
        lw     = 2 if col == "Portfolio" else 1.5

        fig.add_trace(go.Scatter(
            x=series.index, y=series.round(2),
            name=col,
            line=dict(color=colors[col], width=lw)
        ), row=1, col=1)

        # Drawdown
        rolling_max = series.cummax()
        dd = ((series - rolling_max) / rolling_max * 100).round(2)

        fig.add_trace(go.Scatter(
            x=dd.index, y=dd,
            name=f"{col} DD",
            line=dict(color=colors[col], width=1, dash="dot"),
            fill="tozeroy",
            opacity=0.4,
            showlegend=False
        ), row=2, col=1)

    fig.update_layout(
        height=600,
        template="plotly_white",
        hovermode="x unified",
        legend=dict(orientation="h", y=1.02),
        margin=dict(l=0, r=0, t=40, b=0)
    )

    fig.update_yaxes(title_text="Wert ($)",     row=1, col=1)
    fig.update_yaxes(title_text="Drawdown (%)", row=2, col=1)

    fig.show()


def plot_daily_returns_heatmap(equity_curve: pd.DataFrame) -> None:
    """
    Monatliche Returns als Heatmap — wie ein Hedge Fund Report.
    """
    returns = equity_curve["Portfolio"].pct_change().dropna()

    # Monatliche Returns aggregieren
    monthly = returns.resample("ME").apply(
        lambda x: (1 + x).prod() - 1
    ) * 100

    monthly_df            = monthly.to_frame("Return")
    monthly_df["Jahr"]    = monthly_df.index.year
    monthly_df["Monat"]   = monthly_df.index.month

    pivot = monthly_df.pivot(
        index="Jahr", columns="Monat", values="Return"
    )
    pivot.columns = ["Jan","Feb","Mär","Apr","Mai","Jun",
                     "Jul","Aug","Sep","Okt","Nov","Dez"]

    fig = go.Figure(go.Heatmap(
        z=pivot.values.round(1),
        x=pivot.columns.tolist(),
        y=pivot.index.tolist(),
        colorscale=[
            [0.0,  "#dc2626"],
            [0.4,  "#fca5a5"],
            [0.5,  "#f9fafb"],
            [0.6,  "#86efac"],
            [1.0,  "#16a34a"]
        ],
        text=[[f"{v:.1f}%" if not np.isnan(v) else ""
               for v in row]
              for row in pivot.values],
        texttemplate="%{text}",
        textfont=dict(size=11),
        showscale=True,
        zmid=0
    ))

    fig.update_layout(
        title="Monatliche Portfolio Returns (%)",
        template="plotly_white",
        height=300,
        margin=dict(l=0, r=0, t=40, b=0)
    )

    fig.show()

def print_daily_report(positions: pd.DataFrame,
                       metrics: dict) -> None:
    """
    Täglicher Terminal-Report — schneller Überblick.
    """
    total_value  = positions["Aktuell"].sum()
    total_cost   = positions["Kaufwert"].sum()
    total_pnl    = positions["PnL ($)"].sum()
    total_pnl_pct = (total_pnl / total_cost) * 100
    today_pnl    = (positions["Heute (%)"] *
                    positions["Aktuell"] / 100).sum()

    print("\n" + "="*55)
    print(f"  PORTFOLIO REPORT — {datetime.now().strftime('%d.%m.%Y %H:%M')}")
    print("="*55)
    print(f"  Gesamtwert:        ${total_value:>10,.2f}")
    print(f"  Investiert:        ${total_cost:>10,.2f}")
    print(f"  Gesamt PnL:        ${total_pnl:>+10,.2f}  ({total_pnl_pct:+.2f}%)")
    print(f"  Heute:             ${today_pnl:>+10,.2f}")
    print("="*55)

    print("\n  POSITIONEN")
    print(f"  {'Ticker':<8} {'Kurs':>8} {'PnL ($)':>10} "
          f"{'PnL (%)':>8} {'Heute':>7} {'Gewicht':>8}")
    print("  " + "-"*53)

    for _, row in positions.iterrows():
        pnl_sign  = "+" if row["PnL ($)"] >= 0 else ""
        today_sign = "+" if row["Heute (%)"] >= 0 else ""
        print(
            f"  {row['Ticker']:<8}"
            f"  ${row['Kurs']:>7.2f}"
            f"  {pnl_sign}${row['PnL ($)']:>8.2f}"
            f"  {pnl_sign}{row['PnL (%)']:>6.2f}%"
            f"  {today_sign}{row['Heute (%)']:>5.2f}%"
            f"  {row['Gewicht (%)']:>6.1f}%"
        )

    print("\n  RISIKOKENNZAHLEN")
    print(f"  {'Kennzahl':<22} {'Portfolio':>12} {'SPY':>12}")
    print("  " + "-"*46)

    port = metrics["Portfolio"]
    bench = metrics["Benchmark"]

    for key in ["Total Return (%)", "CAGR (%)", "Sharpe Ratio",
                "Max Drawdown (%)", "Volatilität (%)",
                "Beta", "Alpha (%)"]:
        print(f"  {key:<22} {port[key]:>12}  {bench.get(key, '—'):>11}")

    print("="*55)

if __name__ == "__main__":

    # --- Daten laden ---
    prices, info = load_portfolio_data(PORTFOLIO)

    # --- Positionen ---
    positions = calculate_positions(PORTFOLIO, prices)

    # --- Equity Curve ---
    equity_curve = build_equity_curve(PORTFOLIO, prices)

    # --- Risikokennzahlen ---
    metrics = calculate_risk_metrics(equity_curve)

    # --- Report ---
    print_daily_report(positions, metrics)

    # --- Charts ---
    plot_portfolio_overview(positions)
    plot_equity_vs_benchmark(equity_curve)
    plot_daily_returns_heatmap(equity_curve)

    # --- CSV Export ---
    timestamp = datetime.now().strftime("%Y%m%d")
    positions.to_csv(f"portfolio_report_{timestamp}.csv", index=False)
    print(f"\nReport gespeichert: portfolio_report_{timestamp}.csv")