"""
Day 33 - Portfolio Performance Attribution

Was ist attribution?
    Return zerlegen in warum hat das Portfolio so performt?

Brinson-Hood-Beebower Modell (1986):
    Allocation Effect: Hast du richtig gewichtet?
    Selection Effect: Hast du die richtigen Aktien ausgewählt?
    Interaction Effect: Wechselwirkung zwischen Gewichtung und Auswahl

Beispiel:
    Portfolio 12%, Benchmark 10% -> 2% Alpha 
    Attribution sagt:
        + 1.5% kamen aus Sektorgewichtung 
        + 0.8% kamen aus Aktienauswahl
        - 0.3% Interaction Effekt 
    = 2% gesamt

-> Man weiß wo Glück war und wo man so weitermachen soll wie zuvor 

"""

import os
import warnings
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from typing import Optional, Dict, List, Tuple
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import yfinance as yf
warnings.filterwarnings("ignore")

def load_prices(tickers: list, period: str="2y") -> pd.DataFrame:
    df = yf.download(tickers, period=period, auto_adjust=True, progress=False)["Close"]
    # period = period damit man nicht 2 mal 2 jahre schreiben muss
    if len(tickers) == 1:
        df = df.to_frame(name=tickers[0])
    df.columns = df.columns.get_level_values(0)
    return df


# Standard Sektor-Mapping (S&P 500 GICS)
SECTOR_MAP = {
    # Technology
    "AAPL":  "Technology",
    "MSFT":  "Technology",
    "NVDA":  "Technology",
    "AMD":   "Technology",
    "INTC":  "Technology",
    "QCOM":  "Technology",
    "AVGO":  "Technology",
    "META":  "Communication",

    # Communication
    "GOOGL": "Communication",
    "NFLX":  "Communication",
    "DIS":   "Communication",
    "CMCSA": "Communication",

    # Consumer Discretionary
    "AMZN":  "Consumer Disc.",
    "TSLA":  "Consumer Disc.",
    "NKE":   "Consumer Disc.",
    "MCD":   "Consumer Disc.",

    # Healthcare
    "JNJ":   "Healthcare",
    "PFE":   "Healthcare",
    "ABBV":  "Healthcare",
    "UNH":   "Healthcare",

    # Financials
    "JPM":   "Financials",
    "BAC":   "Financials",
    "GS":    "Financials",
    "MS":    "Financials",
    "BRK-B": "Financials",

    # Energy
    "XOM":   "Energy",
    "CVX":   "Energy",
    "COP":   "Energy",

    # Consumer Staples
    "PG":    "Consumer Staples",
    "KO":    "Consumer Staples",
    "PEP":   "Consumer Staples",
    "WMT":   "Consumer Staples",

    # Industrials
    "CAT":   "Industrials",
    "BA":    "Industrials",
    "GE":    "Industrials",
    "HON":   "Industrials",

    # Materials
    "GLD":   "Commodities",
    "SLV":   "Commodities",

    # ETFs
    "SPY":   "Benchmark",
    "QQQ":   "Benchmark",
    "IWM":   "Benchmark",
    "BND":   "Fixed Income",
}


def get_sector(ticker: str) -> str:
    """Gibt Sektor für Ticker zurück."""
    # Erst in hartkodierter Map nachschauen
    if ticker in SECTOR_MAP:
        return SECTOR_MAP[ticker]

    # Dann yfinance fragen
    try:
        info   = yf.Ticker(ticker).info
        sector = info.get("sector", "Unknown")
        return sector or "Unknown"
    except Exception:
        return "Unknown"
    
@dataclass
class Position:
    """Eine Portfolio-Position."""
    ticker:     str
    weight:     float          # Gewicht im Portfolio (0-1)
    sector:     str = ""
    entry_date: str = ""

    def __post_init__(self):
        if not self.sector:
            self.sector = get_sector(self.ticker)


@dataclass
class Portfolio:
    """
    Ein Portfolio mit Positionen und Benchmark.

    Weights müssen auf 1 summieren.
    """
    name:      str
    positions: List[Position]
    benchmark: str = "SPY"

    def __post_init__(self):
        total = sum(p.weight for p in self.positions) # summiert automatisch falls nicht 1
        if abs(total - 1.0) > 0.01:
            # Normalisieren
            for p in self.positions:
                p.weight /= total

    @property
    def tickers(self) -> List[str]:
        return [p.ticker for p in self.positions]

    @property
    def weights(self) -> Dict[str, float]:
        return {p.ticker: p.weight for p in self.positions}

    @property
    def sectors(self) -> Dict[str, str]:
        return {p.ticker: p.sector for p in self.positions}

    def sector_weights(self) -> Dict[str, float]:
        """Gewichtung pro Sektor."""
        sw = {}
        for p in self.positions:
            sw[p.sector] = sw.get(p.sector, 0) + p.weight
        return sw
    
def compute_portfolio_returns(portfolio: Portfolio,
                               prices:    pd.DataFrame,
                               rebalance: str = "ME") -> dict:
    """
    Berechnet Portfolio Returns mit Rebalancing.

    rebalance:
        None → Einmaliger Kauf, kein Rebalancing
        "ME" → Monatliches Rebalancing
        "QE" → Quartalsweises Rebalancing

    Returns:
        portfolio_returns: Tägliche Portfolio Returns
        equity:            Equity Curve
        position_returns:  Returns pro Position
    """
    weights    = portfolio.weights
    tickers    = [t for t in portfolio.tickers
                  if t in prices.columns]

    price_sub  = prices[tickers].dropna()
    returns_df = price_sub.pct_change().dropna()

    if rebalance is None:
        # Einmaliger Kauf
        w_arr          = np.array([
            weights.get(t, 0) for t in tickers
        ])
        portfolio_ret  = (returns_df * w_arr).sum(axis=1)

    else:
        # Rebalancing
        rebal_dates    = returns_df.resample(rebalance).last().index
        portfolio_ret  = pd.Series(
            index=returns_df.index, dtype=float
        )
        current_weights = {t: weights.get(t, 0) for t in tickers}

        for i, date in enumerate(returns_df.index):
            w_arr = np.array([
                current_weights.get(t, 0) for t in tickers
            ])
            daily_ret = float(
                (returns_df.loc[date] * w_arr).sum()
            )
            portfolio_ret.loc[date] = daily_ret

            # Rebalancing?
            if date in rebal_dates and i > 0:
                current_weights = {
                    t: weights.get(t, 0) for t in tickers
                }

    equity             = (1 + portfolio_ret).cumprod() * 10_000
    position_returns   = returns_df.copy()

    return {
        "portfolio_returns":  portfolio_ret.dropna(),
        "equity":             equity.dropna(),
        "position_returns":   position_returns,
        "tickers":            tickers,
        "weights":            weights,
    }


def compute_benchmark_returns(benchmark: str,
                               prices:    pd.DataFrame) -> pd.Series:
    """Benchmark Returns."""
    if benchmark in prices.columns:
        return prices[benchmark].pct_change().dropna()

    try:
        bench  = yf.download(
            benchmark, period="3y",
            auto_adjust=True, progress=False
        )["Close"].squeeze()
        return bench.pct_change().dropna()
    except Exception:
        return pd.Series(dtype=float)
    

def brinson_attribution(portfolio:   Portfolio,
                          prices:      pd.DataFrame,
                          period:      str = "1y") -> dict:
    """
    Brinson-Hood-Beebower Performance Attribution.

    Kam die Outperformance durch die richtige Asset allokation, oder durch die auswahl der richtigen wertpapiere?
    Portfolio 15% 
    Markt 10% -> 5% Alpha 

    Das Modell zerlegt Alpha in:

    1. Allocation Effect (Gewichtungsentscheidung): Wurde mehr Geld in die richtigen Sektoren investiert?
        Hast du Sektoren mit guter Performance übergewichtet?
        AE_i = (w_p_i - w_b_i) × (R_b_i - R_b)

    2. Selection Effect (Titelauswahl): Wurden im Sektor die richtigen Aktien gewaehlt?
        Hast du besser als der Benchmark-Sektor performed?
        SE_i = w_b_i × (R_p_i - R_b_i)

    3. Interaction Effect: richtiger Sektor und gute Titel?
        Kombination — übergewichteter Sektor + bessere Selektion
        IE_i = (w_p_i - w_b_i) × (R_p_i - R_b_i)

    wobei:
        w_p_i = Portfolio-Gewicht Sektor i
        w_b_i = Benchmark-Gewicht Sektor i
        R_p_i = Portfolio-Return Sektor i
        R_b_i = Benchmark-Return Sektor i
        R_b   = Gesamter Benchmark-Return

    Quellen:
        Brinson, Hood, Beebower (1986):
        "Determinants of Portfolio Performance"
        Financial Analysts Journal
    """
    # Preise auf Periode trimmen
    cutoff = pd.Timestamp.now() - pd.DateOffset(years=1) # subtrahiert 1 jahr
    prices_period = prices[prices.index >= cutoff]

    # Benchmark Returns
    bench_ticker  = portfolio.benchmark
    if bench_ticker in prices_period.columns:
        bench_ret   = prices_period[bench_ticker].pct_change().dropna()
        bench_total = float((1 + bench_ret).prod() - 1) 
    else:
        bench_total = 0.0
        bench_ret   = pd.Series(dtype=float)

    # Sektor-Gewichte Portfolio
    port_sector_weights = portfolio.sector_weights()

    # Benchmark-Sektor-Gewichte (vereinfacht: equal weight)
    # In Produktion: echte SPY Sektor-Gewichte von SPDR ETFs
    benchmark_sectors = {
        "Technology":     0.29,
        "Communication":  0.09,
        "Consumer Disc.": 0.10,
        "Healthcare":     0.13,
        "Financials":     0.13,
        "Energy":         0.04,
        "Consumer Staples": 0.06,
        "Industrials":    0.08,
        "Commodities":    0.02,
        "Fixed Income":   0.04,
        "Unknown":        0.02,
    }

    # Sektor-Returns Portfolio
    sector_returns_port  = {}
    sector_returns_bench = {}

    for sector in set(portfolio.sectors.values()):
        # Portfolio-Tickers in diesem Sektor
        sector_tickers = [
            t for t, s in portfolio.sectors.items()
            if s == sector and t in prices_period.columns
        ]

        if not sector_tickers:
            continue

        # Gewichte innerhalb Sektor normalisieren
        total_sector_w = sum(
            portfolio.weights.get(t, 0)
            for t in sector_tickers
        )

        sector_ret = 0.0
        for t in sector_tickers:
            w   = portfolio.weights.get(t, 0) / max(
                total_sector_w, 1e-8
            )
            ret = float(
                (1 + prices_period[t].pct_change().dropna()).prod() - 1
            ) * 100
            sector_ret += w * ret / 100

        sector_returns_port[sector] = sector_ret

        # Benchmark-Return für Sektor (aus SPY approximiert)
        sector_returns_bench[sector] = bench_total * (
            benchmark_sectors.get(sector, 0.05) /
            sum(benchmark_sectors.values())
        )

    # Attribution Berechnung
    attribution = {}
    total_alloc = 0.0
    total_selec = 0.0
    total_inter = 0.0

    for sector in sector_returns_port:
        w_p   = port_sector_weights.get(sector, 0)
        w_b   = benchmark_sectors.get(sector, 0.05)
        r_p   = sector_returns_port.get(sector, 0)
        r_b_s = sector_returns_bench.get(sector, 0)
        r_b   = bench_total

        # Brinson Formel
        alloc = (w_p - w_b) * (r_b_s - r_b)
        selec = w_b * (r_p - r_b_s)
        inter = (w_p - w_b) * (r_p - r_b_s)

        attribution[sector] = {
            "port_weight":   round(w_p * 100, 2),
            "bench_weight":  round(w_b * 100, 2),
            "active_weight": round((w_p - w_b) * 100, 2),
            "port_return":   round(r_p, 2),
            "bench_return":  round(r_b_s, 2),
            "allocation":    round(alloc, 4),
            "selection":     round(selec, 4),
            "interaction":   round(inter, 4),
            "total_contrib": round(alloc + selec + inter, 4),
        }

        total_alloc += alloc
        total_selec += selec
        total_inter += inter

    # Portfolio Return
    port_data  = compute_portfolio_returns(portfolio, prices_period)
    port_total = float(
        (1 + port_data["portfolio_returns"]).prod() - 1
    ) * 100

    alpha = port_total - bench_total

    return {
        "attribution":      attribution,
        "total_allocation": round(total_alloc, 4),
        "total_selection":  round(total_selec, 4),
        "total_interaction": round(total_inter, 4),
        "total_alpha":      round(alpha, 4),
        "portfolio_return": round(port_total * 100, 2),
        "benchmark_return": round(bench_total, 2),
        "explained_alpha":  round(
            total_alloc + total_selec + total_inter, 4
        ),
    }

def factor_attribution(returns: pd.Series,
                        benchmark: pd.Series,
                        period: str = "2y") -> dict:
    """
    Faktor basierte Attribution (Fama-French inspired)

    normalerweise: ri = rf + beta(rm - rf)

    jetz: 
    ri = rf + beta(rm - rf) + beta(SMB) + beta(HML)
    beta(SMB) =  long small cap stocks - big cap stocks (Small minus big)
    beta(HML) = high book to market - low book to market (High minus low)
    book to market = market cap / book value
    book value = total assets - total liabilities

    Faktoren: 
        Market (Beta): Wie viel kommt aus dem Markt?
        Size: Market Cap
        Value: Value vs Growth Exposure
        Momentum: Trend Following exposure
        quality: Profitabel vs unprofitabel

    In echt:
        Echte Fama French faktoren von 
        mba.tuck.dartmouth.edu/pages/faculty/ken.french/data_library.html

    Jetzt:
        mit vereinfachten Yfinance Proxies

    Faktoren:
        Market: SPY Returns
        Size: IWM (Small Cap) - SPY (Large Cap)
        Value: IVE (S&P 500 Value) - IVW (S&P 500 Growth)
        Momentum: MTUM ETF Returns

    """

    # Faktor Daten laden
    factor_tickers = ["SPY", "IWM", "IVE", "IVW", "MTUM"]

    try: 
        factor_prices = yf.download(
            factor_tickers, period=period,
            auto_adjust=True, progress=False
        )["Close"]
        factor_prices.columns = factor_prices.columns.get_level_values(0)
        factor_rets    = factor_prices.pct_change().dropna()

        # Faktoren konstruieren
        factors = pd.DataFrame(index=factor_rets.index) # leere DataFrame erstellen
        factors["Market"]   = factor_rets.get("SPY", 0) # SPY Returns , 0 falls nicht vorhanden
        factors["Size"]     = (
            factor_rets.get("IWM", 0) -
            factor_rets.get("SPY", 0)
        )
        factors["Value"]    = (
            factor_rets.get("IVE", 0) -
            factor_rets.get("IVW", 0)
        )
        factors["Momentum"] = factor_rets.get("MTUM", 0)

    except Exception as e:
        print(f"Faktor-Daten Fehler: {e}")
        return {}
    
    # Align Returns mit Faktoren 
    aligned = pd.concat(
        [returns.rename("portfolio"), factors], # Faktoren hinzufügen
        axis = 1
    ).dropna()

    if len(aligned) < 30:
        return {"error": "nicht genug Daten"}

    # OLS Regression: Portfolio Returns ~ Faktoren
    from numpy.linalg import lstsq # linear least squares

    Y = aligned["portfolio"].values
    X = np.column_stack([
        np.ones(len(aligned)),  # np.ones erstellt eine 1-D array mit len(aligned) Elementen
        aligned["Market"].values,
        aligned["Size"].values,
        aligned["Value"].values,
        aligned["Momentum"].values,
    ])

    coeffs, residuals, rank, sv = lstsq(X, Y, rcond=None) # lstsq berechnet die Koeffizienten der Regression
    # residuals sind die Restwerte der Regression
    # rank ist die Rang der Matrix X
    # sv ist die Standardabweichung der Residuen

    # T-Statistiken
    n = len(Y)
    k = X.shape[1] # Anzahl der Koeffizienten
    y_hat = X @ coeffs
    resid = Y - y_hat # Residuen = Differenz zwischen Y und Y_hat
    # residuen sind die nicht erklärbaren Überbleibe 
    s2 = np.sum(resid**2) / (n - k) # Varianz der Residuen
    var_coef = s2 * np.linalg.inv(X.T @ X).diagonal() # Varianz der Koeffizienten
    # schaut wie stark die Koeffizienten schwanken
    t_stats = coeffs / np.sqrt(var_coef) # T-Statistiken

    #R2 berechnen 
    ss_res = np.sum(resid**2) # Summe der quadrierten Residuen
    ss_tot = np.sum((Y - Y.mean())**2) # Summe der quadrierten Differenzen zwischen Y und Y.mean()
    r2 = 1-ss_res / ss_tot
    # r2 = 1- quadrierte Residuen / quadrierte Differenzen zwischen Y und Y.mean()
    
    # R2 von 0.95  95% deines Returns durch Faktoren erklärt -> wenig echter alpha 
    # R2 von 0.7 -> Nur 70% erklärt 
    # R2 von 0.4 -> Nur 40% erklaert - sehr eigensinnige Strategie 

    # Indexfonds hat R2 von 0.99 -> fast alles durch Faktoren erklaert

    factor_names = ["Alpha", "Market", "Size",
                    "Value", "Momentum"]
    
    result = {
        "r_squared":   round(r2, 4),
        "factors":     {},
        "explained":   round(r2 * 100, 1),
        "unexplained": round((1 - r2) * 100, 1),
    }

    for name, coef, t_stat in zip(
        factor_names, coeffs, t_stats
    ):
        result["factors"][name] = {
            "coefficient": round(float(coef), 6),
            "t_stat":      round(float(t_stat), 3),
            "significant": abs(t_stat) > 2.0,
            "annualized":  round(float(coef) * 252 * 100, 4)
                           if name == "Alpha" else None,
        }


    return result


def rolling_factor_exposure(returns: pd.Series,
                            window: int=126) -> pd.DataFrame:
    """
    Rolling Beta und alpha 

    Zeigt wie sich das Marktrisiko über Zeit verändert 
    Ein gutes aktiv verwaltetes Portfolio hat:
        Stabiles Beta (kein markt timing Glück) also wenig schwnakung 
        Positives Alpha(positive Returns)
    """
    try:
        spy = yf.download(
            "SPY", period="3y",
            auto_adjust=True, progress=False
        )["Close"].squeeze().pct_change().dropna()

        aligned = pd.concat(
            [returns.rename("port"), spy.rename("spy")],
            axis=1
        ).dropna()

        betas   = []
        alphas  = []
        dates   = []

        for i in range(window, len(aligned)):
            window_data = aligned.iloc[i-window:i]
            r_p = window_data["port"].values
            r_b = window_data["spy"].values

            # OLS
            X       = np.column_stack([np.ones(window), r_b]) # stack funktioniert wie append
            coeffs  = np.linalg.lstsq(X, r_p, rcond=None)[0] # lstsq berechnet die Koeffizienten der Regression

            alphas.append(float(coeffs[0]) * 252 * 100)
            betas.append(float(coeffs[1]))
            dates.append(aligned.index[i])
            # Vereinfachte 2 faktor regression über 126 tage 

        # Beta      Alpha (annual)
        #Jan 2024:  1.05    +3.2%    ← normales Marktrisiko, positiver Alpha
        #Jun 2024:  1.45    -2.1%    ← Beta springt an, Alpha negativ
        #Dez 2024:  0.80    +8.4%    ← defensiver, starker Alpha

        # Gutes Portfolio -> stabiles Beta und positiver Alpha

        return pd.DataFrame({
            "alpha_annual": alphas,
            "beta":         betas,
        }, index=dates)

    except Exception as e:
        print(f"Rolling Factor Fehler: {e}")
        return pd.DataFrame()
    

def trade_level_attribution(trade_log: pd.DataFrame,
                            prices: pd.DataFrame) -> dict:
    """
    Schaut welche Trades das meiste geld verlieren 
    Skill / Timing? 

    Wie analysiert man das?
        Wenn NVDA gekauft wegen Signal und 20% anstieg -> Skill
        Wenn gesamter Sektor 20% gestiegen könnte es Luck gewesen sein.
    """
    if trade_log.empty:
        return {}

    required = ["ticker", "entry_price", "exit_price", "pnl"]
    if not all(c in trade_log.columns for c in required):
        return {}
    # Generator der prüft ob jede Pflichtspalte vorhanden ist 
    # -> Fehlende Column -> leeres Dict, kein Crash 

    completed = trade_log.dropna(subset=["pnl"])
    if completed.empty:
        return {}
        
    # Grundstatistiken
    wins      = completed[completed["pnl"] > 0]
    losses    = completed[completed["pnl"] <= 0]
    n         = len(completed)
    win_rate  = len(wins) / n * 100 if n > 0 else 0

    # Contribution Analysis
    total_pnl   = float(completed["pnl"].sum())
    pnl_per_trade = float(completed["pnl"].mean())

    # Top/Bottom Contributors
    top_wins = completed.nlargest(5, "pnl")[
        ["ticker", "pnl", "pnl_pct"]
    ] if "pnl_pct" in completed.columns else completed.nlargest(5, "pnl")

    top_losses = completed.nsmallest(5, "pnl")[
        ["ticker", "pnl", "pnl_pct"]
    ] if "pnl_pct" in completed.columns else completed.nsmallest(5, "pnl")

    # Ticker-Level Attribution
    ticker_attribution = completed.groupby("ticker").agg( # agg ist wie summe
        total_pnl = ("pnl", "sum"),
        n_trades  = ("pnl", "count"),
        win_rate  = ("pnl", lambda x: (x > 0).mean() * 100),
        avg_pnl   = ("pnl", "mean"),
    ).sort_values("total_pnl", ascending=False)

    #Ticker   total_pnl   n_trades   win_rate   avg_pnl
    #NVDA     +4,250      12         75.0%      +354
    #AAPL     +1,800      8          62.5%      +225
    #TSLA     -950        6          33.3%      -158

    # Holding Period Analyse 
    if "duration" in completed.columns:
        short_term = completed[completed["duration"] <= 5]
        medium_term = completed[
            (completed["duration"] > 5) & 
            (completed["duration"] <= 20)
        ]
        long_term = completed[completed["duration"] > 20]

        holding_analysis = {
            "short_term_avg":  round(
                float(short_term["pnl"].mean()), 2
            ) if not short_term.empty else 0,
            "medium_term_avg": round(
                float(medium_term["pnl"].mean()), 2
            ) if not medium_term.empty else 0,
            "long_term_avg":   round(
                float(long_term["pnl"].mean()), 2
            ) if not long_term.empty else 0,
        }
    else:
        holding_analysis = {}

    # Exit Reason Attribution
    if "exit_reason" in completed.columns:
        exit_attribution = completed.groupby(
            "exit_reason"
        )["pnl"].agg(["sum", "count", "mean"]).round(2)
    else:
        exit_attribution = pd.DataFrame()

    return {
        "summary": {
            "total_pnl":        round(total_pnl, 2),
            "n_trades":         n,
            "win_rate":         round(win_rate, 1),
            "avg_pnl_per_trade": round(pnl_per_trade, 2),
            "profit_factor":    round(
                wins["pnl"].sum() /
                abs(losses["pnl"].sum()), 3
            ) if not losses.empty and losses["pnl"].sum() != 0 else 0,
        },
        "top_wins":           top_wins,
        "top_losses":         top_losses,
        "ticker_attribution": ticker_attribution,
        "holding_analysis":   holding_analysis,
        "exit_attribution":   exit_attribution,
    }




# Visualisierung






def plot_brinson_attribution(attr_result: dict,
                              portfolio_name: str) -> None:
    """
    Brinson Attribution Waterfall + Sektor Breakdown.
    """
    attribution = attr_result["attribution"]

    sectors     = list(attribution.keys())
    alloc_vals  = [attribution[s]["allocation"]  for s in sectors]
    selec_vals  = [attribution[s]["selection"]   for s in sectors]
    inter_vals  = [attribution[s]["interaction"] for s in sectors]
    active_w    = [attribution[s]["active_weight"] for s in sectors]

    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=[
            "Allocation Effect nach Sektor (%)",
            "Selection Effect nach Sektor (%)",
            "Active Weights (Portfolio - Benchmark)",
            "Attribution Zusammenfassung"
        ],
        vertical_spacing=0.14,
        horizontal_spacing=0.10
    )

    # Allocation Effect
    a_colors = [
        "#16a34a" if v > 0 else "#ef4444"
        for v in alloc_vals
    ]
    fig.add_trace(go.Bar(
        x=sectors, y=alloc_vals,
        marker_color=a_colors,
        text=[f"{v:.3f}" for v in alloc_vals],
        textposition="outside",
        name="Allocation",
        showlegend=False
    ), row=1, col=1)

    fig.add_hline(
        y=0, line_color="#1e293b",
        line_width=1, row=1, col=1
    )

    # Selection Effect
    s_colors = [
        "#16a34a" if v > 0 else "#ef4444"
        for v in selec_vals
    ]
    fig.add_trace(go.Bar(
        x=sectors, y=selec_vals,
        marker_color=s_colors,
        text=[f"{v:.3f}" for v in selec_vals],
        textposition="outside",
        name="Selection",
        showlegend=False
    ), row=1, col=2)

    fig.add_hline(
        y=0, line_color="#1e293b",
        line_width=1, row=1, col=2
    )

    # Active Weights
    aw_colors = [
        "#16a34a" if v > 0 else "#ef4444"
        for v in active_w
    ]
    fig.add_trace(go.Bar(
        x=sectors, y=active_w,
        marker_color=aw_colors,
        text=[f"{v:+.1f}%" for v in active_w],
        textposition="outside",
        name="Active Weight",
        showlegend=False
    ), row=2, col=1)

    fig.add_hline(
        y=0, line_color="#1e293b",
        line_width=1, row=2, col=1
    )

    # Attribution Summary Waterfall
    components  = [
        "Benchmark",
        "Allocation",
        "Selection",
        "Interaction",
        "Portfolio"
    ]
    values      = [
        attr_result["benchmark_return"],
        attr_result["total_allocation"] * 100,
        attr_result["total_selection"]  * 100,
        attr_result["total_interaction"]* 100,
        0,  # End bar
    ]
    # Waterfall: Portfolio = Benchmark + Effekte
    port_ret = attr_result["portfolio_return"]

    bar_base = [0, attr_result["benchmark_return"], 0, 0, 0]

    running = attr_result["benchmark_return"]
    cum_values = [attr_result["benchmark_return"]]
    for v in values[1:-1]:
        running += v
        cum_values.append(running)
    cum_values.append(port_ret)

    waterfall_colors = [
        "#3b82f6",
        "#16a34a" if values[1] >= 0 else "#ef4444",
        "#16a34a" if values[2] >= 0 else "#ef4444",
        "#16a34a" if values[3] >= 0 else "#ef4444",
        "#2563eb",
    ]

    fig.add_trace(go.Bar(
        x=components,
        y=cum_values,
        marker_color=waterfall_colors,
        text=[f"{v:.2f}%" for v in cum_values],
        textposition="outside",
        name="Attribution",
        showlegend=False
    ), row=2, col=2)

    fig.update_layout(
        height=680,
        template="plotly_white",
        title=(
            f"{portfolio_name} — Brinson Attribution  |  "
            f"Portfolio: {attr_result['portfolio_return']:.2f}%  "
            f"vs  Benchmark: {attr_result['benchmark_return']:.2f}%  "
            f"= Alpha: {attr_result['total_alpha']:.2f}%"
        ),
        margin=dict(l=0, r=0, t=70, b=0)
    )

    fig.update_yaxes(title_text="Contribution (%)", row=1, col=1)
    fig.update_yaxes(title_text="Contribution (%)", row=1, col=2)
    fig.update_yaxes(title_text="Active Weight (%)", row=2, col=1)
    fig.update_yaxes(title_text="Return (%)",        row=2, col=2)

    fig.show()


def plot_factor_attribution(factor_result: dict,
                              rolling_df:    pd.DataFrame) -> None:
    """
    Faktor-Attribution + Rolling Beta/Alpha.
    """
    if not factor_result or "factors" not in factor_result:
        return

    factors = factor_result["factors"]

    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=[
            "Faktor-Koeffizienten",
            "T-Statistiken (|>2| = signifikant)",
            "Rolling Beta (6M)",
            "Rolling Alpha (6M, annualisiert %)"
        ],
        vertical_spacing=0.14,
        horizontal_spacing=0.10
    )

    names  = list(factors.keys())
    coeffs = [factors[n]["coefficient"] for n in names]
    tstats = [factors[n]["t_stat"]       for n in names]
    sigs   = [factors[n]["significant"]  for n in names]

    # Koeffizienten
    coef_colors = [
        "#16a34a" if c > 0 else "#ef4444"
        for c in coeffs
    ]
    fig.add_trace(go.Bar(
        x=names, y=coeffs,
        marker_color=coef_colors,
        text=[f"{c:.4f}" for c in coeffs],
        textposition="outside",
        showlegend=False
    ), row=1, col=1)

    fig.add_hline(
        y=0, line_color="#1e293b",
        line_width=1, row=1, col=1
    )

    # T-Statistiken
    t_colors = [
        "#16a34a" if abs(t) > 2 else "#94a3b8"
        for t in tstats
    ]
    fig.add_trace(go.Bar(
        x=names, y=[abs(t) for t in tstats],
        marker_color=t_colors,
        text=[f"{abs(t):.2f}" for t in tstats],
        textposition="outside",
        showlegend=False
    ), row=1, col=2)

    fig.add_hline(
        y=2.0, line_dash="dash",
        line_color="#ef4444",
        annotation_text="Signifikanz Grenze (2.0)",
        row=1, col=2
    )

    # Rolling Metrics
    if not rolling_df.empty:
        fig.add_trace(go.Scatter(
            x=rolling_df.index,
            y=rolling_df["beta"].round(3),
            name="Rolling Beta",
            line=dict(color="#2563eb", width=1.5),
            showlegend=False
        ), row=2, col=1)

        fig.add_hline(
            y=1.0, line_dash="dot",
            line_color="#94a3b8",
            annotation_text="Beta = 1.0",
            row=2, col=1
        )

        alpha_colors = [
            "#16a34a" if v > 0 else "#ef4444"
            for v in rolling_df["alpha_annual"]
        ]
        fig.add_trace(go.Scatter(
            x=rolling_df.index,
            y=rolling_df["alpha_annual"].round(3),
            name="Rolling Alpha",
            line=dict(color="#16a34a", width=1.5),
            fill="tozeroy",
            fillcolor="rgba(22,163,74,0.08)",
            showlegend=False
        ), row=2, col=2)

        fig.add_hline(
            y=0, line_color="#1e293b",
            line_width=1.5, row=2, col=2
        )

    r2  = factor_result.get("r_squared", 0)
    exp = factor_result.get("explained", 0)

    fig.add_annotation(
        x=0.5, y=1.04,
        xref="paper", yref="paper",
        text=(
            f"R² = {r2:.3f}  |  "
            f"{exp:.1f}% durch Faktoren erklärt  |  "
            f"{factor_result.get('unexplained', 0):.1f}% unerklärtes Alpha"
        ),
        showarrow=False,
        font=dict(size=11)
    )

    fig.update_layout(
        height=650,
        template="plotly_white",
        title="Faktor-Attribution (Fama-French Style)",
        margin=dict(l=0, r=0, t=70, b=0)
    )

    fig.update_yaxes(title_text="Koeffizient",  row=1, col=1)
    fig.update_yaxes(title_text="|T-Statistik|", row=1, col=2)
    fig.update_yaxes(title_text="Beta",          row=2, col=1)
    fig.update_yaxes(title_text="Alpha (%/pa.)", row=2, col=2)

    fig.show()


def plot_trade_attribution(trade_attr: dict) -> None:
    """
    Trade-Level Attribution Visualisierung.
    """
    if not trade_attr or "summary" not in trade_attr:
        print("Keine Trade-Daten.")
        return

    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=[
            "PnL nach Ticker",
            "Top Wins vs. Top Losses",
            "Exit Reason Attribution",
            "Win Rate nach Ticker"
        ],
        vertical_spacing=0.14,
        horizontal_spacing=0.10
    )

    ticker_attr = trade_attr.get(
        "ticker_attribution", pd.DataFrame()
    )

    # Ticker PnL
    if not ticker_attr.empty:
        t_colors = [
            "#16a34a" if v > 0 else "#ef4444"
            for v in ticker_attr["total_pnl"]
        ]
        fig.add_trace(go.Bar(
            x=ticker_attr.index,
            y=ticker_attr["total_pnl"].round(2),
            marker_color=t_colors,
            text=[f"${v:.0f}" for v in ticker_attr["total_pnl"]],
            textposition="outside",
            showlegend=False
        ), row=1, col=1)

        fig.add_hline(
            y=0, line_color="#1e293b",
            line_width=1, row=1, col=1
        )

    # Wins vs Losses
    top_wins   = trade_attr.get("top_wins", pd.DataFrame())
    top_losses = trade_attr.get("top_losses", pd.DataFrame())

    if not top_wins.empty:
        fig.add_trace(go.Bar(
            x=top_wins["ticker"] if "ticker" in top_wins.columns
              else top_wins.index,
            y=top_wins["pnl"].round(2),
            marker_color="#16a34a",
            name="Top Wins",
            text=[f"${v:.0f}" for v in top_wins["pnl"]],
            textposition="outside",
        ), row=1, col=2)

    if not top_losses.empty:
        fig.add_trace(go.Bar(
            x=top_losses["ticker"] if "ticker" in top_losses.columns
              else top_losses.index,
            y=top_losses["pnl"].round(2),
            marker_color="#ef4444",
            name="Top Losses",
            text=[f"${v:.0f}" for v in top_losses["pnl"]],
            textposition="outside",
        ), row=1, col=2)

    # Exit Reason
    exit_attr = trade_attr.get(
        "exit_attribution", pd.DataFrame()
    )
    if not exit_attr.empty and "sum" in exit_attr.columns:
        exit_colors = [
            "#16a34a" if v > 0 else "#ef4444"
            for v in exit_attr["sum"]
        ]
        fig.add_trace(go.Bar(
            x=exit_attr.index,
            y=exit_attr["sum"].round(2),
            marker_color=exit_colors,
            text=[f"${v:.0f}" for v in exit_attr["sum"]],
            textposition="outside",
            showlegend=False
        ), row=2, col=1)

    # Win Rate
    if not ticker_attr.empty:
        wr_colors = [
            "#16a34a" if v >= 50 else "#ef4444"
            for v in ticker_attr["win_rate"]
        ]
        fig.add_trace(go.Bar(
            x=ticker_attr.index,
            y=ticker_attr["win_rate"].round(1),
            marker_color=wr_colors,
            text=[f"{v:.0f}%" for v in ticker_attr["win_rate"]],
            textposition="outside",
            showlegend=False
        ), row=2, col=2)

        fig.add_hline(
            y=50, line_dash="dot",
            line_color="#94a3b8",
            annotation_text="50%",
            row=2, col=2
        )

    fig.update_layout(
        height=650,
        template="plotly_white",
        title="Trade-Level Performance Attribution",
        legend=dict(orientation="h", y=1.02),
        margin=dict(l=0, r=0, t=70, b=0)
    )

    fig.update_yaxes(title_text="PnL ($)",    row=1, col=1)
    fig.update_yaxes(title_text="PnL ($)",    row=1, col=2)
    fig.update_yaxes(title_text="PnL ($)",    row=2, col=1)
    fig.update_yaxes(title_text="Win Rate (%)", row=2, col=2)

    fig.show()


def plot_attribution_summary(attr_result: dict,
                              factor_result: dict,
                              portfolio_name: str) -> None:
    """
    Zusammenfassungs-Dashboard für alle Attribution Methoden.
    """
    fig = make_subplots(
        rows=1, cols=3,
        subplot_titles=[
            "Brinson Attribution",
            "Faktor Contribution",
            "Alpha Zerlegung"
        ],
        specs=[[{"type": "pie"},
                {"type": "bar"},
                {"type": "bar"}]],
        horizontal_spacing=0.10
    )

    # Brinson Pie
    brinson_vals = [
        abs(attr_result["total_allocation"]  * 100),
        abs(attr_result["total_selection"]   * 100),
        abs(attr_result["total_interaction"] * 100),
    ]
    brinson_labels = ["Allocation", "Selection", "Interaction"]
    brinson_colors = ["#2563eb", "#16a34a", "#f59e0b"]

    fig.add_trace(go.Pie(
        labels=brinson_labels,
        values=brinson_vals,
        hole=0.45,
        marker_colors=brinson_colors,
        textinfo="label+percent",
        showlegend=False
    ), row=1, col=1)

    # Faktor Bars
    if factor_result and "factors" in factor_result:
        fnames  = list(factor_result["factors"].keys())
        fcoeffs = [
            factor_result["factors"][n]["coefficient"] * 252 * 100
            if n == "Alpha"
            else factor_result["factors"][n]["coefficient"]
            for n in fnames
        ]
        f_colors = [
            "#16a34a" if c > 0 else "#ef4444"
            for c in fcoeffs
        ]
        fig.add_trace(go.Bar(
            x=fnames, y=fcoeffs,
            marker_color=f_colors,
            text=[f"{c:.3f}" for c in fcoeffs],
            textposition="outside",
            showlegend=False
        ), row=1, col=2)

        fig.add_shape(
            type="line",
            x0=0, x1=1,
            xref="x2 domain",   # x2 = zweiter subplot (Bar)
            y0=0, y1=0,
            yref="y2",
            line=dict(color="#1e293b", width=1)
        )

    # Alpha Zerlegung
    alpha_components = {
        "Allocation":  attr_result["total_allocation"]  * 100,
        "Selection":   attr_result["total_selection"]   * 100,
        "Interaction": attr_result["total_interaction"] * 100,
    }
    ac_colors = [
        "#16a34a" if v > 0 else "#ef4444"
        for v in alpha_components.values()
    ]
    fig.add_trace(go.Bar(
        x=list(alpha_components.keys()),
        y=list(alpha_components.values()),
        marker_color=ac_colors,
        text=[f"{v:+.3f}%" for v in alpha_components.values()],
        textposition="outside",
        showlegend=False
    ), row=1, col=3)

    total_alpha = attr_result["total_alpha"]
    color       = "#16a34a" if total_alpha > 0 else "#ef4444"

    fig.update_layout(
        height=450,
        template="plotly_white",
        title=(
            f"{portfolio_name} — Attribution Summary  |  "
            f"Total Alpha: "
            f"<span style='color:{color}'>"
            f"{total_alpha:+.2f}%</span>"
        ),
        margin=dict(l=0, r=0, t=70, b=0)
    )

    fig.show()


def print_attribution_report(attr_result:   dict,
                               factor_result: dict,
                               portfolio:     Portfolio) -> None:
    """Vollständiger Attribution Report im Terminal."""

    print(f"\n{'='*60}")
    print(f"  PERFORMANCE ATTRIBUTION REPORT")
    print(f"  Portfolio: {portfolio.name}")
    print(f"  {datetime.now().strftime('%d.%m.%Y %H:%M')}")
    print(f"{'='*60}")

    # Performance Übersicht
    print(f"\n  PERFORMANCE ÜBERSICHT")
    print(f"  {'Portfolio Return:':<28} "
          f"{attr_result['portfolio_return']:>+8.2f}%")
    print(f"  {'Benchmark Return:':<28} "
          f"{attr_result['benchmark_return']:>+8.2f}%")
    print(f"  {'Total Alpha:':<28} "
          f"{attr_result['total_alpha']:>+8.2f}%")

    # Brinson Zerlegung
    print(f"\n  BRINSON ATTRIBUTION (Alpha Zerlegung)")
    print(f"  {'Allocation Effect:':<28} "
          f"{attr_result['total_allocation']*100:>+8.4f}%")
    print(f"  {'Selection Effect:':<28} "
          f"{attr_result['total_selection']*100:>+8.4f}%")
    print(f"  {'Interaction Effect:':<28} "
          f"{attr_result['total_interaction']*100:>+8.4f}%")
    print(f"  {'Erklärtes Alpha:':<28} "
          f"{attr_result['explained_alpha']*100:>+8.4f}%")

    # Sektor Breakdown
    print(f"\n  SEKTOR ATTRIBUTION")
    print(f"  {'Sektor':<20} "
          f"{'Port%':>7} "
          f"{'Bench%':>7} "
          f"{'Alloc':>8} "
          f"{'Selec':>8} "
          f"{'Total':>8}")
    print("  " + "-"*58)

    for sector, vals in attr_result["attribution"].items():
        total = vals["total_contrib"]
        sign  = "+" if total >= 0 else ""
        print(
            f"  {sector:<20}"
            f"  {vals['port_weight']:>6.1f}%"
            f"  {vals['bench_weight']:>6.1f}%"
            f"  {vals['allocation']:>+7.4f}"
            f"  {vals['selection']:>+7.4f}"
            f"  {sign}{total:>6.4f}"
        )

    # Faktor Attribution
    if factor_result and "factors" in factor_result:
        print(f"\n  FAKTOR ATTRIBUTION")
        print(f"  R²: {factor_result.get('r_squared', 0):.3f}  |  "
              f"{factor_result.get('explained', 0):.1f}% erklärt")
        print(f"\n  {'Faktor':<12} "
              f"{'Koeff':>10} "
              f"{'T-Stat':>8} "
              f"{'Sig':>5}")
        print("  " + "-"*38)

        for name, fdata in factor_result["factors"].items():
            sig    = "✅" if fdata["significant"] else "  "
            coeff  = fdata["coefficient"]
            tstat  = fdata["t_stat"]
            annual = fdata.get("annualized")
            suffix = (f"  ({annual:.2f}% pa.)"
                      if annual else "")
            print(
                f"  {name:<12}"
                f"  {coeff:>9.5f}"
                f"  {tstat:>7.3f}"
                f"  {sig}{suffix}"
            )

    # Portfolio Gewichte
    print(f"\n  PORTFOLIO POSITIONEN")
    print(f"  {'Ticker':<8} {'Sektor':<20} {'Gewicht':>8}")
    print("  " + "-"*38)
    for pos in sorted(
        portfolio.positions,
        key=lambda x: x.weight, reverse=True
    ):
        print(
            f"  {pos.ticker:<8}"
            f"  {pos.sector:<20}"
            f"  {pos.weight*100:>7.1f}%"
        )

    print(f"\n{'='*60}")


if __name__ == "__main__":

    print("Tag 33 — Performance Attribution")
    print("=" * 55)

    # --- Portfolio definieren ---
    # Beispiel: Konzentriertes Tech-Portfolio
    tech_portfolio = Portfolio(
        name="Tech Growth Portfolio",
        positions=[
            Position("AAPL",  0.20),
            Position("MSFT",  0.20),
            Position("NVDA",  0.20),
            Position("GOOGL", 0.15),
            Position("META",  0.10),
            Position("AMZN",  0.10),
            Position("JPM",   0.05),   # Diversifikation
        ],
        benchmark="SPY"
    )

    # Diversifiziertes Portfolio zum Vergleich
    diversified_portfolio = Portfolio(
        name="Diversified Portfolio",
        positions=[
            Position("AAPL",  0.10),
            Position("MSFT",  0.10),
            Position("NVDA",  0.10),
            Position("JPM",   0.10),
            Position("JNJ",   0.10),
            Position("XOM",   0.10),
            Position("KO",    0.10),
            Position("GLD",   0.10),
            Position("BND",   0.10),
            Position("SPY",   0.10),
        ],
        benchmark="SPY"
    )

    # --- Daten laden ---
    print("\n1. Daten laden...")
    all_tickers = list(set(
        tech_portfolio.tickers +
        diversified_portfolio.tickers +
        ["SPY"]
    ))

    prices = load_prices(all_tickers, "2y")
    print(f"   {len(prices)} Handelstage, "
          f"{len(prices.columns)} Ticker")

    # --- Portfolio Returns ---
    print("\n2. Portfolio Returns berechnen...")
    tech_data = compute_portfolio_returns(
        tech_portfolio, prices
    )
    div_data  = compute_portfolio_returns(
        diversified_portfolio, prices
    )
    bench_ret = compute_benchmark_returns("SPY", prices)

    # Performance Summary
    for name, data in [
        ("Tech Portfolio",          tech_data),
        ("Diversified Portfolio",   div_data),
    ]:
        rets    = data["portfolio_returns"]
        equity  = data["equity"]
        total   = (equity.iloc[-1] / 10_000 - 1) * 100
        sharpe  = (rets.mean() / rets.std() *
                   np.sqrt(252)) if rets.std() > 0 else 0
        print(
            f"   {name:<28} "
            f"Return: {total:+.1f}%  "
            f"Sharpe: {sharpe:.2f}"
        )

    bench_total = (
        (1 + bench_ret).prod() - 1
    ) * 100
    print(f"   {'SPY Benchmark':<28} "
          f"Return: {bench_total:+.1f}%")

    # --- Brinson Attribution ---
    print("\n3. Brinson Attribution (Tech Portfolio)...")
    brinson_tech = brinson_attribution(
        tech_portfolio, prices, "1y"
    )

    print(
        f"\n   Portfolio:  {brinson_tech['portfolio_return']:+.2f}%"
    )
    print(
        f"   Benchmark:  {brinson_tech['benchmark_return']:+.2f}%"
    )
    print(
        f"   Alpha:      {brinson_tech['total_alpha']:+.2f}%"
    )
    print(
        f"   Allocation: {brinson_tech['total_allocation']*100:+.4f}%"
    )
    print(
        f"   Selection:  {brinson_tech['total_selection']*100:+.4f}%"
    )
    print(
        f"   Interaction:{brinson_tech['total_interaction']*100:+.4f}%"
    )

    plot_brinson_attribution(brinson_tech, "Tech Growth Portfolio")

    # --- Brinson Diversified ---
    print("\n4. Brinson Attribution (Diversified Portfolio)...")
    brinson_div = brinson_attribution(
        diversified_portfolio, prices, "1y"
    )
    print(
        f"   Alpha:      {brinson_div['total_alpha']:+.2f}%"
    )

    # --- Faktor Attribution ---
    print("\n5. Faktor Attribution...")
    tech_returns = tech_data["portfolio_returns"]

    factor_result = factor_attribution(
        tech_returns, bench_ret, "2y"
    )

    if factor_result and "factors" in factor_result:
        print(f"\n   R²: {factor_result['r_squared']:.3f}")
        print(f"   {factor_result['explained']:.1f}% "
              f"durch Faktoren erklärt")
        print(f"\n   {'Faktor':<12} {'Beta':>8} {'T-Stat':>8}")
        for name, fdata in factor_result["factors"].items():
            sig = "✅" if fdata["significant"] else "  "
            print(
                f"   {name:<12}"
                f"  {fdata['coefficient']:>7.4f}"
                f"  {fdata['t_stat']:>7.3f}"
                f"  {sig}"
            )

    # --- Rolling Factor Exposure ---
    print("\n6. Rolling Factor Exposure...")
    rolling_factors = rolling_factor_exposure(
        tech_returns, window=126
    )

    if not rolling_factors.empty:
        avg_beta  = rolling_factors["beta"].mean()
        avg_alpha = rolling_factors["alpha_annual"].mean()
        print(f"   Avg. Rolling Beta:  {avg_beta:.3f}")
        print(f"   Avg. Rolling Alpha: {avg_alpha:.2f}% pa.")

        pct_pos = (
            rolling_factors["alpha_annual"] > 0
        ).mean() * 100
        print(
            f"   Alpha positiv in:   {pct_pos:.1f}% der Perioden"
        )

    plot_factor_attribution(
        factor_result, rolling_factors
    )

    # --- Trade-Level Attribution (Demo) ---
    print("\n7. Trade-Level Attribution (Demo)...")

    # Lade echten Trade Log wenn vorhanden
    if os.path.exists("bot_v1_trades.csv"):
        trade_log = pd.read_csv("bot_v1_trades.csv")
        print(f"   {len(trade_log)} Trades aus bot_v1_trades.csv")
    else:
        # Synthetischer Trade Log für Demo
        np.random.seed(42)
        tickers_demo = [
            "AAPL", "MSFT", "NVDA", "AAPL",
            "GOOGL", "NVDA", "META", "AAPL"
        ]
        trade_log = pd.DataFrame({
            "ticker":      tickers_demo,
            "entry_price": np.random.uniform(100, 500, 8),
            "exit_price":  np.random.uniform(100, 550, 8),
            "shares":      np.random.randint(1, 20, 8),
            "duration":    np.random.randint(1, 30, 8),
            "exit_reason": np.random.choice(
                ["Take Profit", "Stop Loss",
                 "Momentum Exit", "MA Crossover"],
                8
            ),
        })
        trade_log["pnl"] = (
            (trade_log["exit_price"] -
             trade_log["entry_price"]) *
            trade_log["shares"]
        ).round(2)
        trade_log["pnl_pct"] = (
            (trade_log["exit_price"] /
             trade_log["entry_price"] - 1) * 100
        ).round(2)
        print("   Demo Trade Log erstellt")

    trade_attr = trade_level_attribution(trade_log, prices)

    if trade_attr:
        summary = trade_attr.get("summary", {})
        print(f"\n   Total PnL:    ${summary.get('total_pnl', 0):+,.2f}")
        print(f"   Win Rate:     {summary.get('win_rate', 0):.1f}%")
        print(f"   Profit Factor:{summary.get('profit_factor', 0):.2f}")

        print("\n   Ticker Attribution:")
        ta = trade_attr.get("ticker_attribution", pd.DataFrame())
        if not ta.empty:
            print(ta[[
                "total_pnl", "n_trades", "win_rate"
            ]].round(2).to_string())

        plot_trade_attribution(trade_attr)

    # --- Vollständiger Report ---
    print("\n8. Vollständiger Attribution Report...")
    print_attribution_report(
        brinson_tech, factor_result, tech_portfolio
    )

    # --- Portfolio Comparison ---
    print("\n9. Portfolio Vergleich (Tech vs. Diversified)...")
    print(f"\n  {'Kennzahl':<25} "
          f"{'Tech':>12} "
          f"{'Diversified':>14}")
    print("  " + "-"*52)

    for label, t_val, d_val in [
        ("Alpha (%)",
         brinson_tech["total_alpha"],
         brinson_div["total_alpha"]),
        ("Allocation Effect",
         brinson_tech["total_allocation"] * 100,
         brinson_div["total_allocation"] * 100),
        ("Selection Effect",
         brinson_tech["total_selection"] * 100,
         brinson_div["total_selection"] * 100),
    ]:
        print(
            f"  {label:<25}"
            f"  {t_val:>+10.4f}%"
            f"  {d_val:>+12.4f}%"
        )

    # Summary Chart
    plot_attribution_summary(
        brinson_tech, factor_result, "Tech Growth Portfolio"
    )

    # --- Export ---
    export = {
        "generated":        datetime.now().isoformat(),
        "tech_portfolio":   {
            "brinson":  {k: v for k, v in brinson_tech.items()
                         if k != "attribution"},
            "factors":  factor_result.get("factors", {}),
            "r_squared": factor_result.get("r_squared", 0),
        },
        "diversified":      {
            "brinson":  {k: v for k, v in brinson_div.items()
                         if k != "attribution"},
        },
    }

    import json
    with open("day33_attribution.json", "w") as f:
        json.dump(export, f, indent=2, default=str)

    trade_log.to_csv("day33_trade_log.csv", index=False)
    print("\n✅ Gespeichert: day33_attribution.json, "
          "day33_trade_log.csv")