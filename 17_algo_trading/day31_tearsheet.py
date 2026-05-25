"""
Day 31 professionelles Performance tearsheet 

Ein oder zwei seiten dokjument, das von hedgefunds an investoren gesendet wird 
Das generieren wir per Knopfdruck.

Was drauf ist:
- Equity Curve vs. Benchmark 
- Alle Kennzahlen: Sharpe, Calmar, Sortino, Max Drawdown
- Monatliche Returns Heatmap 
- Rolling Metrics (Sharpe, Vola)
- Drawdown Anayse
-Trade-Level STatistiken
-VaR und CVaR

Output:
1. Interaktives HTML Dashboard ( Plotly)
2. PDF Tearsheet (reportlab)
3. JSON Export aller Metriken 
"""

import os
import json
import warnings
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from typing import Optional, Dict, List
import pandas as pd 
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy import stats
from scipy.stats import norm
import yfinance as yf 
warnings.filterwarnings("ignore")

# PDF
try:
    from reportlab.lib.pagesizes import A4, landscape
    from reportlab.lib.styles import (
        getSampleStyleSheet, ParagraphStyle # samplestyle sind vordefinierte styles
        # paragraph sind eigene styles
    )
    from reportlab.lib import colors
    from reportlab.platypus import (
        SimpleDocTemplate, Paragraph, Spacer, # pdf, text block, spacer = space 
        Table, TableStyle, HRFlowable, Image # Tabellen, horizontale Linie, Bilder
    )
    from reportlab.lib.units import cm # einheiten
    from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT # Text Alignment
    REPORTLAB_AVAILABLE = True
except ImportError:
    REPORTLAB_AVAILABLE = False
    print("pip install reportlab für PDF Export")

def load_data(ticker: str, period: str = "5y") -> pd.DataFrame:
    df = yf.download(
        ticker, period=period,
        auto_adjust=True, progress=False
    )
    df.columns = df.columns.get_level_values(0)
    return df.dropna()


def compute_all_metrics(returns: pd.Series,
                        benchmark: pd.Series = None, # none das ist optional
                        risk_free: float = 0.05,
                        capital: float = 10_000) -> dict:
    
    """
    Vollständige Performance-Metriken Berechnung.

    Das ist das Herzstück des Tearsheets.
    Jede Kennzahl erklärt warum sie wichtig ist.
    """
    clean = returns.dropna()
    equity   = (1 + clean).cumprod() * capital

    years = len(clean) / 252
    total_ret = (equity.iloc[-1] / capital - 1) * 100

    # CAGR 
    cagr = (
        (equity.iloc[-1] / capital) ** (1/max(years,0.01)) -1
    ) * 100

    # Volatilität
    vol = clean.std() * np.sqrt(252) * 100

    # Sharpe Ratio
    excess = clean - risk_free / 252
    sharpe = (excess.mean() / clean.std()) * np.sqrt(252) if clean.std() > 0 else 0

    # Sortino Ratio
    # Wie sharpe nur mit downside vola 
    # bestrafz nur negative abweichungen 
    downside = clean[clean < 0].std() * np.sqrt(252)
    sortino = (clean.mean() * 252 - risk_free) / downside if downside > 0 else 0
    # eine strategie die oft +3% returns hat aber selten -1% würde 
    # die +3% returns bei sharpe als risiko werten. Sortino nicht

    # Calmar Ratio
    rolling_max = equity.cummax()
    drawdown    = (equity - rolling_max) / rolling_max * 100
    max_dd      = float(drawdown.min())
    calmar      = cagr / abs(max_dd) if max_dd != 0 else 0

    # Omega Ratio 
    # Summe Gewinne / Summe Verluste 
    # >1 mehr gewinne als Verluste 
    threshold = risk_free/252
    gains = clean[clean > threshold] - threshold 
    losses = threshold - clean[clean <= threshold]
    omega = gains.sum() / losses.sum() if losses.sum() > 0 else 999 
    # 999 ist ein Platzhalter für unendlich, wenn es keine Verluste gibt

    # Information Ratio (vs. Benchmark )
    # wie konsistent schlägt man die benchmark 
    ir = None 
    alpha = None 
    beta = None 
    if benchmark is not None:
        bench_clean = benchmark.dropna()
        aligned = pd.concat([clean, bench_clean], axis =1).dropna()
        aligned.columns = ["strat", "bench"]

        if len(aligned) > 10:
            active_ret = aligned["strat"] - aligned["bench"]
            ir         = (active_ret.mean() /
                          active_ret.std() *
                          np.sqrt(252)) \
                         if active_ret.std() > 0 else 0

            # Beta
            cov  = aligned.cov().iloc[0, 1]
            var  = aligned["bench"].var()
            beta = cov / var if var > 0 else 1.0

            # Jensen's Alpha
            # alpha = (R_p - R_f) - beta * (R_m - R_f)
            # -> Return der erzielt wird über das was das beta erklärt 
            # bsp: beta 1.2 und markt 10% -> 1.2 * 10% = 12% return erwartet
            # man macht aber 15 % -> alpha = 3% (überperformance)
            bench_ann = aligned["bench"].mean() * 252
            alpha     = (clean.mean() * 252 -
                         (risk_free +
                          beta * (bench_ann - risk_free)))

    # VaR / CVaR
    var_95  = float(np.percentile(clean, 5)) * 100 # 5% worst case
    var_99  = float(np.percentile(clean, 1)) * 100
    cvar_95 = float(
        clean[clean <= np.percentile(clean, 5)].mean() # durchschnitt der worst 5% cases
    ) * 100

    # Win/Loss Statistiken
    wins        = clean[clean > 0]
    losses_s    = clean[clean < 0]
    win_rate    = len(wins) / len(clean) * 100
    avg_win     = wins.mean() * 100 if len(wins) > 0 else 0
    avg_loss    = losses_s.mean() * 100 \
                  if len(losses_s) > 0 else 0
    profit_factor = (
        wins.sum() / abs(losses_s.sum())
        if losses_s.sum() != 0 else 999
    )

    # Best / Worst Day
    best_day  = float(clean.max()) * 100
    worst_day = float(clean.min()) * 100

    # Longest Drawdown Duration
    in_dd         = drawdown < 0
    max_dd_dur    = 0
    current_dur   = 0
    for val in in_dd:
        if val:
            current_dur += 1
            max_dd_dur   = max(max_dd_dur, current_dur)
        else:
            current_dur  = 0

    # Skewness & Kurtosis
    skew =float(stats.skew(clean))
    # skewness misst die asymmetrie der returns. Positive skewness bedeutet mehr extreme positive returns, 
    # negative skewness mehr extreme negative returns. 
    # Ein hoher negativer skew ist oft unerwünscht, da er auf häufige kleine Gewinne und seltene große Verluste hindeutet.
    kurt =float(stats.kurtosis(clean))
    # kurtosis misst die "Tailstärke" der NormalVerteilung. 
    # Ein hoher kurtosis-Wert deutet auf mehr extreme Werte (Gewinne oder Verluste) hin als bei einer Normalverteilung.

    return {
        # Returns
        "total_return_pct":  round(total_ret, 2),
        "cagr_pct":          round(cagr, 2),
        "volatility_pct":    round(vol, 2),

        # Risk-Adjusted
        "sharpe":            round(sharpe, 3),
        "sortino":           round(sortino, 3),
        "calmar":            round(calmar, 3),
        "omega":             round(min(omega, 999), 3),

        # Drawdown
        "max_drawdown_pct":  round(max_dd, 2),
        "max_dd_duration_d": max_dd_dur,
        "current_dd_pct":    round(float(drawdown.iloc[-1]), 2),

        # VaR
        "var_95_pct":        round(var_95, 3),
        "var_99_pct":        round(var_99, 3),
        "cvar_95_pct":       round(cvar_95, 3),

        # Trades
        "win_rate_pct":      round(win_rate, 1),
        "avg_win_pct":       round(avg_win, 3),
        "avg_loss_pct":      round(avg_loss, 3),
        "profit_factor":     round(profit_factor, 3),
        "best_day_pct":      round(best_day, 3),
        "worst_day_pct":     round(worst_day, 3),

        # Statistik
        "skewness":          round(skew, 3),
        "kurtosis":          round(kurt, 3),
        "n_trading_days":    len(clean),
        "years":             round(years, 2),

        # vs. Benchmark
        "information_ratio": round(ir, 3) if ir else None,
        "alpha_annual":      round(alpha * 100, 3) if alpha else None,
        "beta":              round(beta, 3) if beta else None,

        # Kapital
        "initial_capital":   capital,
        "final_value":       round(float(equity.iloc[-1]), 2),
        "profit_loss":       round(float(equity.iloc[-1]) - capital, 2),
    }

def compute_monthly_returns(returns: pd.Series) -> pd.DataFrame:
    """
    Monatliche Returns als Pivot-Tabelle.
    Basis für die Heatmap im Tearsheet.
    """
    monthly = returns.resample("ME").apply( # ME = Month End
        lambda x: (1 + x).prod() - 1 # wichtig nicht x.sum() -> returns müssen kumuliert werden, da sie sich multiplizieren
    ) * 100

    df           = monthly.to_frame("return")
    df["year"]   = df.index.year
    df["month"]  = df.index.month

    pivot = df.pivot( # dreht due pivot-tabelle sodass die jahre in den zeilen und die monate in den spalten sind
        index="year", columns="month", values="return"
    )
    pivot.columns = [
        "Jan","Feb","Mär","Apr","Mai","Jun",
        "Jul","Aug","Sep","Okt","Nov","Dez"
    ]

    # Jahres-Return
    pivot["Jahr"] = pivot.sum(axis=1).round(2)

    return pivot.round(2)


def compute_rolling_metrics(returns:   pd.Series,
                              window:   int   = 126,
                              risk_free: float = 0.05) -> pd.DataFrame:
    """
    Rolling Sharpe, Volatilität und Beta.

    Zeigt ob Performance konsistent ist
    oder von einzelnen Perioden abhängt.
    da sonst eine hohe Sharpe im jahr kkommen könnte obwohl 
    monat = 03
    """
    roll = pd.DataFrame(index=returns.index)

    # Rolling Sharpe
    roll["sharpe"] = (
        returns.rolling(window)
        .apply(lambda x: (
            (x.mean() - risk_free/252) /
            x.std() * np.sqrt(252)
        ) if x.std() > 0 else 0)
    ).round(3)

    # Rolling Volatilität
    roll["vol"] = (
        returns.rolling(window)
        .std() * np.sqrt(252) * 100
    ).round(3)

    # Rolling Return (annualisiert)
    roll["return"] = (
        returns.rolling(window)
        .apply(lambda x: (1 + x).prod() ** (252/window) - 1)
        * 100
    ).round(3)

    return roll.dropna()

def build_interactive_tearsheet(returns:     pd.Series,
                                  benchmark:   pd.Series  = None,
                                  strategy_name: str      = "Strategie",
                                  benchmark_name: str     = "SPY",
                                  capital:     float      = 10_000,
                                  risk_free:   float      = 0.05) -> go.Figure:
    """
    Vollständiges interaktives Tearsheet als Plotly Figure.

    Layout:
        Row 1: Equity Curve + Benchmark (groß)
        Row 2: Drawdown | Rolling Sharpe
        Row 3: Monatliche Returns Heatmap
        Row 4: Return Verteilung | Rolling Volatilität
    """
    equity  = (1 + returns).cumprod() * capital
    metrics = compute_all_metrics(
        returns, benchmark, risk_free, capital
    )
    monthly = compute_monthly_returns(returns)
    rolling = compute_rolling_metrics(returns, 126, risk_free)

    # Benchmark Equity
    if benchmark is not None:
        bench_aligned = benchmark.reindex(returns.index).dropna()
        bench_equity  = (1 + bench_aligned).cumprod() * capital
    else:
        bench_equity  = None

    # Drawdown
    rolling_max = equity.cummax()
    drawdown    = (equity - rolling_max) / rolling_max * 100

    fig = make_subplots(
        rows=4, cols=2,
        row_heights=[0.32, 0.18, 0.28, 0.22],
        vertical_spacing=0.06,
        horizontal_spacing=0.08,
        subplot_titles=[
            f"{strategy_name} — Equity Curve",
            "Drawdown (%)",
            "Rolling Sharpe (6M)",
            "Monatliche Returns (%)",
            "Monatliche Returns (%)",  # Heatmap span
            "Return Verteilung",
            "Rolling Volatilität (%)",
        ],
        specs=[
            [{"colspan": 2}, None],
            [{}, {}],
            [{"colspan": 2}, None],
            [{}, {}],
        ]
    )

    # ── Row 1: Equity Curve ──────────────────────────────
    fig.add_trace(go.Scatter(
        x=equity.index,
        y=equity.round(2),
        name=strategy_name,
        line=dict(color="#2563eb", width=2.5)
    ), row=1, col=1)

    if bench_equity is not None:
        fig.add_trace(go.Scatter(
            x=bench_equity.index,
            y=bench_equity.round(2),
            name=benchmark_name,
            line=dict(color="#94a3b8", width=1.5, dash="dot")
        ), row=1, col=1)

    fig.add_hline(
        y=capital, line_dash="dot",
        line_color="#e2e8f0", opacity=0.8,
        row=1, col=1
    )

    # Metrics Annotation oben rechts
    ann_text = (
        f"CAGR: {metrics['cagr_pct']:.1f}%  |  "
        f"Sharpe: {metrics['sharpe']:.2f}  |  "
        f"Max DD: {metrics['max_drawdown_pct']:.1f}%  |  "
        f"Calmar: {metrics['calmar']:.2f}"
    )
    fig.add_annotation(
        x=0.99, y=0.97,
        xref="paper", yref="paper",
        text=ann_text,
        showarrow=False,
        bgcolor="rgba(255,255,255,0.9)",
        bordercolor="#e2e8f0",
        borderwidth=1,
        font=dict(size=11, color="#1e293b")
    )

    # ── Row 2 Links: Drawdown ────────────────────────────
    fig.add_trace(go.Scatter(
        x=drawdown.index,
        y=drawdown.round(2),
        name="Drawdown",
        line=dict(color="#ef4444", width=1.2),
        fill="tozeroy",
        fillcolor="rgba(239,68,68,0.10)",
        showlegend=False
    ), row=2, col=1)

    for level, color, label in [
        (-5,  "#f59e0b", "-5%"),
        (-10, "#ef4444", "-10%"),
    ]:
        fig.add_hline(
            y=level, line_dash="dot",
            line_color=color, opacity=0.5,
            annotation_text=label,
            row=2, col=1
        )

    # ── Row 2 Rechts: Rolling Sharpe ────────────────────
    if not rolling.empty:
        sharpe_colors = [
            "#16a34a" if v > 0 else "#ef4444"
            for v in rolling["sharpe"]
        ]
        fig.add_trace(go.Scatter(
            x=rolling.index,
            y=rolling["sharpe"],
            name="Rolling Sharpe",
            line=dict(color="#2563eb", width=1.5),
            showlegend=False
        ), row=2, col=2)

        fig.add_hline(
            y=0, line_color="#94a3b8",
            line_width=1, row=2, col=2
        )
        fig.add_hline(
            y=1, line_dash="dot",
            line_color="#16a34a", opacity=0.5,
            annotation_text="Sharpe 1.0",
            row=2, col=2
        )

    # ── Row 3: Monatliche Returns Heatmap ───────────────
    month_cols = [
        "Jan","Feb","Mär","Apr","Mai","Jun",
        "Jul","Aug","Sep","Okt","Nov","Dez","Jahr"
    ]
    avail_cols = [
        c for c in month_cols if c in monthly.columns
    ]

    fig.add_trace(go.Heatmap(
        z=monthly[avail_cols].values,
        x=avail_cols,
        y=[str(y) for y in monthly.index],
        colorscale=[
            [0.0,  "#dc2626"],
            [0.35, "#fca5a5"],
            [0.5,  "#f9fafb"],
            [0.65, "#86efac"],
            [1.0,  "#16a34a"]
        ],
        text=[[
            f"{v:.1f}%" if not np.isnan(v) else ""
            for v in row
        ] for row in monthly[avail_cols].values],
        texttemplate="%{text}",
        textfont=dict(size=10),
        showscale=True,
        zmid=0,
        colorbar=dict(
            len=0.25, y=0.38,
            thickness=12
        )
    ), row=3, col=1)

    # ── Row 4 Links: Return Verteilung ──────────────────
    ret_pct = returns.dropna() * 100

    fig.add_trace(go.Histogram(
        x=ret_pct,
        nbinsx=60,
        name="Tages-Returns",
        marker_color="#3b82f6",
        opacity=0.7,
        showlegend=False
    ), row=4, col=1)

    # Normalverteilungs-Overlay
    x_range = np.linspace(
        float(ret_pct.min()),
        float(ret_pct.max()),
        200
    )
    mu      = float(ret_pct.mean())
    sigma   = float(ret_pct.std())
    norm_y  = (norm.pdf(x_range, mu, sigma) *
                len(ret_pct) * (ret_pct.max() - ret_pct.min()) / 60)

    fig.add_trace(go.Scatter(
        x=x_range, y=norm_y,
        name="Normalverteilung",
        line=dict(color="#ef4444", width=2),
        showlegend=False
    ), row=4, col=1)

    # VaR Linie
    fig.add_vline(
        x=metrics["var_95_pct"],
        line_dash="dash",
        line_color="#f59e0b",
        annotation_text=f"VaR 95%: {metrics['var_95_pct']:.2f}%",
        row=4, col=1
    )

    # ── Row 4 Rechts: Rolling Volatilität ───────────────
    if not rolling.empty:
        fig.add_trace(go.Scatter(
            x=rolling.index,
            y=rolling["vol"],
            name="Rolling Vol",
            line=dict(color="#f59e0b", width=1.5),
            fill="tozeroy",
            fillcolor="rgba(245,158,11,0.08)",
            showlegend=False
        ), row=4, col=2)

        avg_vol = float(rolling["vol"].mean())
        fig.add_hline(
            y=avg_vol,
            line_dash="dot",
            line_color="#94a3b8",
            annotation_text=f"Avg: {avg_vol:.1f}%",
            row=4, col=2
        )

    # ── Layout ───────────────────────────────────────────
    fig.update_layout(
        height=1050,
        template="plotly_white",
        title=dict(
            text=(
                f"<b>{strategy_name}</b> — Performance Tearsheet  |  "
                f"{returns.index[0].strftime('%d.%m.%Y')} – "
                f"{returns.index[-1].strftime('%d.%m.%Y')}"
            ),
            font=dict(size=16)
        ),
        hovermode="x unified",
        legend=dict(orientation="h", y=1.01),
        margin=dict(l=0, r=0, t=70, b=0)
    )

    fig.update_yaxes(title_text="Wert ($)",   row=1, col=1)
    fig.update_yaxes(title_text="DD (%)",     row=2, col=1)
    fig.update_yaxes(title_text="Sharpe",     row=2, col=2)
    fig.update_yaxes(title_text="Häufigkeit", row=4, col=1)
    fig.update_yaxes(title_text="Vol (%) pa", row=4, col=2)

    return fig

def generate_pdf_tearsheet(returns:       pd.Series,
                             benchmark:     pd.Series  = None,
                             strategy_name: str        = "Strategie",
                             benchmark_name: str       = "SPY",
                             capital:       float      = 10_000,
                             risk_free:     float      = 0.05,
                             output_path:   str        = None) -> str:
    """
    Professionelles PDF Tearsheet — 2 Seiten.

    Seite 1: Übersicht, Kennzahlen, Heatmap
    Seite 2: Risiko, Drawdown, Stress Tests
    """
    if not REPORTLAB_AVAILABLE:
        print("reportlab fehlt: pip install reportlab")
        return ""

    if output_path is None:
        ts          = datetime.now().strftime("%Y%m%d_%H%M")
        output_path = f"tearsheet_{strategy_name}_{ts}.pdf"

    metrics  = compute_all_metrics(
        returns, benchmark, risk_free, capital
    )
    monthly  = compute_monthly_returns(returns)
    equity   = (1 + returns).cumprod() * capital

    doc  = SimpleDocTemplate(
        output_path,
        pagesize     = A4,
        rightMargin  = 1.5*cm,
        leftMargin   = 1.5*cm,
        topMargin    = 1.5*cm,
        bottomMargin = 1.5*cm,
    )

    styles   = getSampleStyleSheet()
    elements = []

    # ── Stile ──────────────────────────────────────────
    title_style = ParagraphStyle(
        "TSTitle",
        parent    = styles["Title"],
        fontSize  = 20,
        textColor = colors.HexColor("#0f172a"),
        spaceAfter = 4,
    )
    sub_style = ParagraphStyle(
        "TSSub",
        parent    = styles["Normal"],
        fontSize  = 9,
        textColor = colors.HexColor("#64748b"),
        spaceAfter = 12,
    )
    h2_style = ParagraphStyle(
        "TSH2",
        parent     = styles["Heading2"],
        fontSize   = 11,
        textColor  = colors.HexColor("#1e40af"),
        spaceBefore = 12,
        spaceAfter  = 6,
    )
    body = ParagraphStyle(
        "TSBody",
        parent    = styles["Normal"],
        fontSize  = 9,
        textColor = colors.HexColor("#374151"),
    )

    # ── Header ─────────────────────────────────────────
    elements.append(Paragraph(
        f"📈 {strategy_name}", title_style
    ))
    elements.append(Paragraph(
        f"Performance Tearsheet  ·  "
        f"{returns.index[0].strftime('%d.%m.%Y')} – "
        f"{returns.index[-1].strftime('%d.%m.%Y')}  ·  "
        f"Erstellt: {datetime.now().strftime('%d.%m.%Y %H:%M')}",
        sub_style
    ))
    elements.append(HRFlowable(
        width="100%", thickness=2,
        color=colors.HexColor("#1e40af"),
        spaceAfter=10
    ))

    # ── KPI Summary Row ────────────────────────────────
    elements.append(Paragraph("Kennzahlen Übersicht", h2_style))

    def cell(label, value, good=None):
        """Hilfsfunktion für farbige Zellen."""
        if good is True:
            val_color = "#16a34a"
        elif good is False:
            val_color = "#dc2626"
        else:
            val_color = "#0f172a"
        return [
            Paragraph(label, ParagraphStyle(
                "lbl", parent=styles["Normal"],
                fontSize=8, textColor=colors.HexColor("#64748b")
            )),
            Paragraph(f"<b>{value}</b>", ParagraphStyle(
                "val", parent=styles["Normal"],
                fontSize=11,
                textColor=colors.HexColor(val_color)
            )),
        ]

    kpi_data = [
        [
            cell("Total Return",
                 f"{metrics['total_return_pct']:+.1f}%",
                 metrics['total_return_pct'] > 0),
            cell("CAGR",
                 f"{metrics['cagr_pct']:.1f}%",
                 metrics['cagr_pct'] > 8),
            cell("Volatilität",
                 f"{metrics['volatility_pct']:.1f}%"),
            cell("Sharpe Ratio",
                 f"{metrics['sharpe']:.2f}",
                 metrics['sharpe'] > 1),
        ],
        [
            cell("Max Drawdown",
                 f"{metrics['max_drawdown_pct']:.1f}%",
                 metrics['max_drawdown_pct'] > -15),
            cell("Calmar Ratio",
                 f"{metrics['calmar']:.2f}",
                 metrics['calmar'] > 1),
            cell("Sortino Ratio",
                 f"{metrics['sortino']:.2f}",
                 metrics['sortino'] > 1),
            cell("Win Rate",
                 f"{metrics['win_rate_pct']:.1f}%",
                 metrics['win_rate_pct'] > 50),
        ],
        [
            cell("VaR 95% (1T)",
                 f"{metrics['var_95_pct']:.2f}%"),
            cell("CVaR 95% (1T)",
                 f"{metrics['cvar_95_pct']:.2f}%"),
            cell("Profit Factor",
                 f"{metrics['profit_factor']:.2f}",
                 metrics['profit_factor'] > 1.3),
            cell("Omega Ratio",
                 f"{metrics['omega']:.2f}",
                 metrics['omega'] > 1),
        ],
    ]

    # Flachklopfen: jede Zeile hat 4 Spalten mit je 2 Elementen
    flat_data = []
    for row in kpi_data:
        flat_row = []
        for cell_pair in row:
            flat_row.extend(cell_pair)
        flat_data.append(flat_row)

    kpi_table = Table(
        flat_data,
        colWidths=[2.5*cm, 2.5*cm] * 4
    )
    kpi_table.setStyle(TableStyle([
        ("BACKGROUND",    (0,0), (-1,-1),
         colors.HexColor("#f8fafc")),
        ("ROWBACKGROUNDS",(0,0), (-1,-1),
         [colors.HexColor("#f8fafc"),
          colors.HexColor("#f1f5f9"),
          colors.HexColor("#f8fafc")]),
        ("BOX",           (0,0), (-1,-1), 0.5,
         colors.HexColor("#e2e8f0")),
        ("INNERGRID",     (0,0), (-1,-1), 0.3,
         colors.HexColor("#e2e8f0")),
        ("PADDING",       (0,0), (-1,-1), 8),
        ("VALIGN",        (0,0), (-1,-1), "MIDDLE"),
    ]))

    elements.append(kpi_table)
    elements.append(Spacer(1, 10))

    # ── Monatliche Returns Tabelle ─────────────────────
    elements.append(Paragraph(
        "Monatliche Returns (%)", h2_style
    ))

    month_cols = [
        "Jan","Feb","Mär","Apr","Mai","Jun",
        "Jul","Aug","Sep","Okt","Nov","Dez","Jahr"
    ]
    avail = [c for c in month_cols if c in monthly.columns]

    header = ["Jahr"] + avail
    m_data = [header]

    for year in monthly.index:
        row = [str(year)]
        for m in avail:
            val = monthly.loc[year, m]
            if pd.isna(val):
                row.append("—")
            else:
                row.append(f"{val:.1f}%")
        m_data.append(row)

    m_table = Table(
        m_data,
        colWidths=[1.5*cm] + [1.25*cm] * len(avail)
    )

    # Dynamisches Styling: grün = positiv, rot = negativ
    m_style = [
        ("BACKGROUND",  (0,0), (-1,0),
         colors.HexColor("#1e40af")),
        ("TEXTCOLOR",   (0,0), (-1,0), colors.white),
        ("FONTNAME",    (0,0), (-1,0), "Helvetica-Bold"),
        ("FONTSIZE",    (0,0), (-1,-1), 7.5),
        ("FONTNAME",    (0,1), (0,-1), "Helvetica-Bold"),
        ("GRID",        (0,0), (-1,-1), 0.3,
         colors.HexColor("#e2e8f0")),
        ("PADDING",     (0,0), (-1,-1), 4),
        ("ALIGN",       (1,1), (-1,-1), "CENTER"),
    ]

    for r_idx, row in enumerate(m_data[1:], start=1):
        for c_idx, val in enumerate(row[1:], start=1):
            if val == "—":
                continue
            try:
                num = float(val.replace("%", ""))
                if num > 0:
                    bg = colors.HexColor("#dcfce7")
                    fg = colors.HexColor("#166534")
                elif num < 0:
                    bg = colors.HexColor("#fee2e2")
                    fg = colors.HexColor("#991b1b")
                else:
                    bg = colors.white
                    fg = colors.HexColor("#374151")
                m_style.append(
                    ("BACKGROUND", (c_idx, r_idx),
                     (c_idx, r_idx), bg)
                )
                m_style.append(
                    ("TEXTCOLOR", (c_idx, r_idx),
                     (c_idx, r_idx), fg)
                )
            except ValueError:
                pass

    m_table.setStyle(TableStyle(m_style))
    elements.append(m_table)
    elements.append(Spacer(1, 10))

    # ── Detail Kennzahlen ──────────────────────────────
    elements.append(Paragraph("Detail Statistiken", h2_style))

    detail_data = [
        ["Kennzahl", "Wert", "Kennzahl", "Wert"],
        ["Startkapital",
         f"${metrics['initial_capital']:,.0f}",
         "Endwert",
         f"${metrics['final_value']:,.0f}"],
        ["Gewinn/Verlust",
         f"${metrics['profit_loss']:+,.0f}",
         "Handelstage",
         str(metrics['n_trading_days'])],
        ["Bester Tag",
         f"{metrics['best_day_pct']:+.2f}%",
         "Schlechtester Tag",
         f"{metrics['worst_day_pct']:+.2f}%"],
        ["Avg. Gewinn/Tag",
         f"{metrics['avg_win_pct']:+.3f}%",
         "Avg. Verlust/Tag",
         f"{metrics['avg_loss_pct']:+.3f}%"],
        ["Max DD Dauer",
         f"{metrics['max_dd_duration_d']} Tage",
         "Aktueller DD",
         f"{metrics['current_dd_pct']:.2f}%"],
        ["Schiefe (Skew)",
         f"{metrics['skewness']:.3f}",
         "Kurtosis",
         f"{metrics['kurtosis']:.3f}"],
    ]

    if metrics.get("beta"):
        detail_data += [
            ["Beta",
             f"{metrics['beta']:.3f}",
             "Alpha (pa.)",
             f"{metrics['alpha_annual']:+.2f}%"],
            ["Information Ratio",
             f"{metrics['information_ratio']:.3f}",
             "vs. Benchmark",
             benchmark_name],
        ]

    d_table = Table(
        detail_data,
        colWidths=[4.0*cm, 3.5*cm, 4.0*cm, 3.5*cm]
    )
    d_table.setStyle(TableStyle([
        ("BACKGROUND",    (0,0), (-1,0),
         colors.HexColor("#1e40af")),
        ("TEXTCOLOR",     (0,0), (-1,0), colors.white),
        ("FONTNAME",      (0,0), (-1,0), "Helvetica-Bold"),
        ("FONTNAME",      (0,1), (0,-1), "Helvetica-Bold"),
        ("FONTNAME",      (2,1), (2,-1), "Helvetica-Bold"),
        ("ROWBACKGROUNDS",(0,1), (-1,-1),
         [colors.HexColor("#f8fafc"),
          colors.HexColor("#ffffff")]),
        ("FONTSIZE",      (0,0), (-1,-1), 9),
        ("GRID",          (0,0), (-1,-1), 0.3,
         colors.HexColor("#e2e8f0")),
        ("PADDING",       (0,0), (-1,-1), 5),
    ]))

    elements.append(d_table)
    elements.append(Spacer(1, 16))

    # ── Footer ─────────────────────────────────────────
    elements.append(HRFlowable(
        width="100%", thickness=0.5,
        color=colors.HexColor("#e2e8f0"), spaceAfter=6
    ))
    elements.append(Paragraph(
        "⚠ Disclaimer: Dieses Tearsheet dient ausschließlich "
        "Informationszwecken. Vergangene Performance ist kein "
        "Indikator für zukünftige Ergebnisse. Keine Anlageberatung.",
        ParagraphStyle(
            "Disc",
            parent    = styles["Normal"],
            fontSize  = 7,
            textColor = colors.HexColor("#94a3b8"),
        )
    ))
    elements.append(Paragraph(
        f"Erstellt mit Python  ·  "
        f"{datetime.now().strftime('%d.%m.%Y')}  ·  "
        f"Kapital: ${capital:,.0f}  ·  "
        f"Risk-Free: {risk_free*100:.1f}%",
        ParagraphStyle(
            "Footer",
            parent    = styles["Normal"],
            fontSize  = 7,
            textColor = colors.HexColor("#94a3b8"),
            alignment = TA_CENTER,
        )
    ))

    doc.build(elements)
    print(f"✅ PDF gespeichert: {output_path}")
    return output_path


def compare_strategies(strategies: Dict[str, pd.Series],
                         benchmark:   pd.Series  = None,
                         capital:     float       = 10_000,
                         risk_free:   float       = 0.05) -> pd.DataFrame:
    """
    Vergleicht mehrere Strategien in einer Tabelle.

    Input:
        strategies = {
            "MA Crossover":    returns_series,
            "RSI Reversion":   returns_series,
            "Buy & Hold":      returns_series,
        }
    """
    rows = []

    for name, rets in strategies.items():
        m = compute_all_metrics(rets, benchmark, risk_free, capital)
        rows.append({
            "Strategie":        name,
            "Total Ret %":      m["total_return_pct"],
            "CAGR %":           m["cagr_pct"],
            "Vola %":           m["volatility_pct"],
            "Sharpe":           m["sharpe"],
            "Sortino":          m["sortino"],
            "Calmar":           m["calmar"],
            "Max DD %":         m["max_drawdown_pct"],
            "Win Rate %":       m["win_rate_pct"],
            "Profit Factor":    m["profit_factor"],
            "VaR 95%":          m["var_95_pct"],
            "CVaR 95%":         m["cvar_95_pct"],
            "Skewness":         m["skewness"],
            "Kurtosis":         m["kurtosis"],
        })

    df = pd.DataFrame(rows).set_index("Strategie")
    return df


def plot_strategy_comparison_tearsheet(
        strategies:    Dict[str, pd.Series],
        benchmark:     pd.Series  = None,
        strategy_name: str        = "Strategy Comparison",
        capital:       float      = 10_000) -> None:
    """
    Vergleichs-Tearsheet für mehrere Strategien.
    """
    colors_map = [
        "#2563eb", "#16a34a", "#f59e0b",
        "#ef4444", "#8b5cf6", "#0891b2"
    ]

    fig = make_subplots(
        rows=3, cols=2,
        row_heights=[0.4, 0.3, 0.3],
        vertical_spacing=0.08,
        horizontal_spacing=0.08,
        subplot_titles=[
            "Normalisierte Equity Curves",
            "Drawdown Vergleich",
            "Rolling Sharpe (6M)",
            "CAGR vs. Max Drawdown",
            "Monatliche Return Verteilung",
            "Kennzahlen Radar",
        ],
        specs=[
            [{"colspan": 2}, None],
            [{}, {}],
            [{}, {}],
        ]
    )

    # ── Row 1: Equity Curves ─────────────────────────
    for i, (name, rets) in enumerate(strategies.items()):
        equity = (1 + rets).cumprod() * capital
        color  = colors_map[i % len(colors_map)]
        lw     = 2 if i == 0 else 1.5

        fig.add_trace(go.Scatter(
            x=equity.index,
            y=equity.round(2),
            name=name,
            line=dict(color=color, width=lw)
        ), row=1, col=1)

    if benchmark is not None:
        b_eq = (1 + benchmark).cumprod() * capital
        fig.add_trace(go.Scatter(
            x=b_eq.index,
            y=b_eq.round(2),
            name="Benchmark",
            line=dict(color="#94a3b8", width=1.2, dash="dot")
        ), row=1, col=1)

    # ── Row 2 Links: Rolling Sharpe ──────────────────
    for i, (name, rets) in enumerate(strategies.items()):
        rolling = compute_rolling_metrics(rets, 126)
        color   = colors_map[i % len(colors_map)]

        if not rolling.empty:
            fig.add_trace(go.Scatter(
                x=rolling.index,
                y=rolling["sharpe"],
                name=f"{name} Sharpe",
                line=dict(color=color, width=1.2),
                showlegend=False
            ), row=2, col=1)

    fig.add_hline(
        y=0, line_color="#94a3b8",
        line_width=1, row=2, col=1
    )
    fig.add_hline(
        y=1, line_dash="dot",
        line_color="#16a34a", opacity=0.4,
        row=2, col=1
    )

    # ── Row 2 Rechts: CAGR vs Max DD Scatter ─────────
    comparison = compare_strategies(
        strategies, benchmark, capital
    )

    for i, (name, row) in enumerate(comparison.iterrows()):
        color = colors_map[i % len(colors_map)]
        fig.add_trace(go.Scatter(
            x=[abs(row["Max DD %"])],
            y=[row["CAGR %"]],
            mode="markers+text",
            name=name,
            text=[name],
            textposition="top right",
            marker=dict(
                color=color, size=12,
                line=dict(width=2, color="white")
            ),
            showlegend=False
        ), row=2, col=2)

    # ── Row 3 Links: Monthly Return Boxplot ──────────
    for i, (name, rets) in enumerate(strategies.items()):
        monthly = rets.resample("ME").apply(
            lambda x: (1 + x).prod() - 1
        ) * 100
        color   = colors_map[i % len(colors_map)]

        fig.add_trace(go.Box(
            y=monthly,
            name=name,
            marker_color=color,
            boxmean=True,
            showlegend=False
        ), row=3, col=1)

    # ── Row 3 Rechts: Radar Chart ─────────────────────
    metrics_radar = ["Sharpe", "Calmar", "Win Rate %",
                      "Profit Factor"]

    for i, (name, row) in enumerate(comparison.iterrows()):
        color = colors_map[i % len(colors_map)]
        vals  = [
            max(min(row["Sharpe"],          3),  0),
            max(min(row["Calmar"],          3),  0),
            max(min(row["Win Rate %"] / 100, 1), 0),
            max(min(row["Profit Factor"],    3),  0),
        ]
        # Polar Chart
        fig.add_trace(go.Scatterpolar(
            r=vals + [vals[0]],
            theta=metrics_radar + [metrics_radar[0]],
            name=name,
            line=dict(color=color, width=2),
            fill="toself",
            fillcolor=color.replace(
                "#", "rgba("
            ).replace("2563eb", "37,99,235,0.1)"),
            showlegend=False
        ) if False else go.Bar(  # Fallback: Bar Chart
            x=metrics_radar,
            y=vals,
            name=name,
            marker_color=color,
            opacity=0.7,
            showlegend=False
        ), row=3, col=2)

    fig.update_layout(
        barmode="group",
        height=850,
        template="plotly_white",
        title=strategy_name,
        hovermode="x unified",
        legend=dict(orientation="h", y=1.01),
        margin=dict(l=0, r=0, t=70, b=0)
    )

    fig.update_yaxes(title_text="Wert ($)",   row=1, col=1)
    fig.update_yaxes(title_text="Sharpe",     row=2, col=1)
    fig.update_xaxes(title_text="|Max DD| %", row=2, col=2)
    fig.update_yaxes(title_text="CAGR %",     row=2, col=2)
    fig.update_yaxes(title_text="Monthly %",  row=3, col=1)

    fig.show()


def live_portfolio_tearsheet(api_key:    str,
                               secret_key: str,
                               benchmark:  str = "SPY",
                               capital:    float = 10_000) -> None:
    """
    Generiert Tearsheet für echtes Alpaca Paper Portfolio.

    Lädt Trade History aus Alpaca und baut daraus
    eine echte Equity Curve.
    """
    try:
        from alpaca.trading.client import TradingClient
        from alpaca.trading.requests import GetPortfolioHistoryRequest

        client = TradingClient(
            api_key=api_key, secret_key=secret_key, paper=True
        )

        # Portfolio History laden
        try:
            history = client.get_portfolio_history(
                GetPortfolioHistoryRequest(
                    period="1M",
                    timeframe="1D",
                    extended_hours=False,
                )
            )

            timestamps = [
                datetime.fromtimestamp(t)
                for t in history.timestamp
            ]
            equity_vals = history.equity
            returns_vals = history.profit_loss_pct

            equity  = pd.Series(
                equity_vals,
                index=pd.DatetimeIndex(timestamps)
            )
            returns = pd.Series(
                returns_vals,
                index=pd.DatetimeIndex(timestamps)
            ).fillna(0)

            print(f"✅ Portfolio History geladen: "
                  f"{len(equity)} Datenpunkte")

        except Exception as e:
            print(f"Portfolio History Fehler: {e}")
            print("Nutze simulierte Daten für Demo...")
            # Fallback: simulierte Daten
            dates   = pd.bdate_range(
                end=datetime.now(), periods=60
            )
            returns = pd.Series(
                np.random.normal(0.001, 0.012, len(dates)),
                index=dates
            )

        # Benchmark laden
        bench_data = yf.download(
            benchmark, period="1mo",
            auto_adjust=True, progress=False
        )["Close"].pct_change().dropna()

        # Tearsheet generieren
        print("\nGeneriere Live Portfolio Tearsheet...")

        metrics = compute_all_metrics(
            returns, bench_data, 0.05, capital
        )

        print(f"\n{'='*50}")
        print(f"  LIVE PORTFOLIO — Paper Trading")
        print(f"{'='*50}")
        print(f"  CAGR:          {metrics['cagr_pct']:.2f}%")
        print(f"  Sharpe:        {metrics['sharpe']:.3f}")
        print(f"  Max DD:        {metrics['max_drawdown_pct']:.2f}%")
        print(f"  Calmar:        {metrics['calmar']:.3f}")
        print(f"  Win Rate:      {metrics['win_rate_pct']:.1f}%")
        print(f"{'='*50}")

        # PDF
        pdf_path = generate_pdf_tearsheet(
            returns,
            bench_data,
            strategy_name  = "Paper Portfolio",
            benchmark_name = benchmark,
            capital        = capital,
            output_path    = "live_portfolio_tearsheet.pdf"
        )

        return metrics

    except ImportError:
        print("alpaca-py nicht installiert")
        return {}
    


if __name__ == "__main__":

    print("Tag 31 — Performance Tearsheet")
    print("=" * 55)

    CAPITAL   = 10_000
    RISK_FREE = 0.05
    BENCHMARK = "SPY"

    # --- Benchmark laden ---
    print("\n1. Daten laden...")
    bench_df  = load_data(BENCHMARK, "5y")
    bench_ret = bench_df["Close"].pct_change().dropna()

    # --- Strategien simulieren ---
    # MA Crossover (20/50)
    def sma_strategy(prices: pd.Series,
                      fast:   int = 20,
                      slow:   int = 50) -> pd.Series:
        sma_f  = prices.rolling(fast).mean()
        sma_s  = prices.rolling(slow).mean()
        signal = (sma_f > sma_s).astype(int).shift(1)
        return (prices.pct_change() * signal).dropna()

    aapl_df  = load_data("AAPL", "5y")
    aapl_ret = aapl_df["Close"].pct_change().dropna()
    nvda_df  = load_data("NVDA", "5y")
    nvda_ret = nvda_df["Close"].pct_change().dropna()

    ma_ret   = sma_strategy(aapl_df["Close"].squeeze())
    rsi_ret  = bench_ret * 0.8 + np.random.normal(
        0.0002, 0.008, len(bench_ret)
    )   # Simuliert

    # --- Metriken ---
    print("\n2. Performance Metriken (AAPL Buy & Hold)...")
    metrics_aapl = compute_all_metrics(
        aapl_ret, bench_ret, RISK_FREE, CAPITAL
    )

    print(f"\n  Kennzahlen AAPL:")
    key_metrics = [
        ("Total Return",   f"{metrics_aapl['total_return_pct']:+.1f}%"),
        ("CAGR",           f"{metrics_aapl['cagr_pct']:.1f}%"),
        ("Sharpe",         f"{metrics_aapl['sharpe']:.3f}"),
        ("Sortino",        f"{metrics_aapl['sortino']:.3f}"),
        ("Calmar",         f"{metrics_aapl['calmar']:.3f}"),
        ("Omega",          f"{metrics_aapl['omega']:.3f}"),
        ("Max Drawdown",   f"{metrics_aapl['max_drawdown_pct']:.1f}%"),
        ("Win Rate",       f"{metrics_aapl['win_rate_pct']:.1f}%"),
        ("Profit Factor",  f"{metrics_aapl['profit_factor']:.3f}"),
        ("VaR 95% (1T)",   f"{metrics_aapl['var_95_pct']:.3f}%"),
        ("CVaR 95% (1T)",  f"{metrics_aapl['cvar_95_pct']:.3f}%"),
        ("Beta",           f"{metrics_aapl.get('beta', 0):.3f}"),
        ("Alpha (pa.)",    f"{metrics_aapl.get('alpha_annual', 0):.2f}%"),
        ("Skewness",       f"{metrics_aapl['skewness']:.3f}"),
        ("Kurtosis",       f"{metrics_aapl['kurtosis']:.3f}"),
    ]

    for k, v in key_metrics:
        print(f"  {k:<20} {v:>10}")

    # --- Monatliche Returns ---
    print("\n3. Monatliche Returns (AAPL)...")
    monthly = compute_monthly_returns(aapl_ret)
    print(monthly.tail(3).to_string())

    # --- Interaktives Tearsheet ---
    print("\n4. Interaktives Tearsheet generieren...")
    fig = build_interactive_tearsheet(
        returns        = aapl_ret,
        benchmark      = bench_ret,
        strategy_name  = "AAPL Buy & Hold",
        benchmark_name = "SPY",
        capital        = CAPITAL,
        risk_free      = RISK_FREE,
    )
    fig.show()

    # HTML Export
    html_path = "tearsheet_AAPL.html"
    fig.write_html(html_path)
    print(f"   ✅ HTML gespeichert: {html_path}")

    # --- PDF Tearsheet ---
    print("\n5. PDF Tearsheet generieren...")
    if REPORTLAB_AVAILABLE:
        pdf_path = generate_pdf_tearsheet(
            returns        = aapl_ret,
            benchmark      = bench_ret,
            strategy_name  = "AAPL Buy & Hold",
            benchmark_name = "SPY",
            capital        = CAPITAL,
            risk_free      = RISK_FREE,
            output_path    = "tearsheet_AAPL.pdf"
        )
    else:
        print("   reportlab fehlt — pip install reportlab")

    # --- Strategy Comparison ---
    print("\n6. Strategy Comparison...")
    strategies = {
        "AAPL Buy & Hold": aapl_ret,
        "NVDA Buy & Hold": nvda_ret,
        "MA Crossover":    ma_ret,
        "SPY (Benchmark)": bench_ret,
    }

    comparison = compare_strategies(
        strategies, bench_ret, CAPITAL, RISK_FREE
    )

    print("\n  Comparison Table:")
    display_cols = [
        "CAGR %", "Sharpe", "Calmar",
        "Max DD %", "Win Rate %", "Profit Factor"
    ]
    print(comparison[display_cols].to_string())

    # Beste Strategie nach Sharpe
    best = comparison["Sharpe"].idxmax()
    print(f"\n  Beste Strategie (Sharpe): {best}")
    print(
        f"  Sharpe: {comparison.loc[best, 'Sharpe']:.3f}  "
        f"CAGR: {comparison.loc[best, 'CAGR %']:.1f}%  "
        f"Max DD: {comparison.loc[best, 'Max DD %']:.1f}%"
    )

    # Comparison Chart
    plot_strategy_comparison_tearsheet(
        strategies     = strategies,
        benchmark      = bench_ret,
        strategy_name  = "Strategy Comparison — Tearsheet",
        capital        = CAPITAL,
    )

    # --- Rolling Metrics ---
    print("\n7. Rolling Metrics (AAPL, 6-Monats Fenster)...")
    rolling = compute_rolling_metrics(aapl_ret, 126)

    print(f"  Avg. Rolling Sharpe:   "
          f"{rolling['sharpe'].mean():.3f}")
    print(f"  Min. Rolling Sharpe:   "
          f"{rolling['sharpe'].min():.3f}")
    print(f"  Avg. Rolling Vol:      "
          f"{rolling['vol'].mean():.1f}%")
    print(f"  Max. Rolling Vol:      "
          f"{rolling['vol'].max():.1f}%")

    # --- Live Portfolio (wenn Alpaca verfügbar) ---
    API_KEY    = os.getenv("ALPACA_API_KEY", "")
    SECRET_KEY = os.getenv("ALPACA_SECRET_KEY", "")

    if API_KEY and SECRET_KEY:
        print("\n8. Live Portfolio Tearsheet...")
        live_portfolio_tearsheet(
            API_KEY, SECRET_KEY,
            benchmark=BENCHMARK,
            capital=CAPITAL
        )
    else:
        print("\n8. Live Portfolio: Kein API Key (.env fehlt)")

    # --- JSON Export ---
    print("\n9. JSON Export...")
    export = {
        "generated":       datetime.now().isoformat(),
        "strategy":        "AAPL Buy & Hold",
        "period":          "5y",
        "capital":         CAPITAL,
        "metrics":         metrics_aapl,
        "comparison":      comparison.to_dict(),
    }
    with open("tearsheet_metrics.json", "w") as f:
        json.dump(export, f, indent=2, default=str)
    print("   ✅ JSON gespeichert: tearsheet_metrics.json")

    # --- Commit Reminder ---
    print("\n" + "="*55)
    print("OUTPUTS ERSTELLT:")
    print("="*55)
    print("  tearsheet_AAPL.html      ← interaktiv, im Browser öffnen")
    print("  tearsheet_AAPL.pdf       ← professioneller PDF Report")
    print("  tearsheet_metrics.json   ← alle Metriken als JSON")