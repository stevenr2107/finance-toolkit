"""
Day 28 — Alpaca API Setup & Live Paper Trading

Was wir bauen:
    - Alpaca Connection (Paper Trading)
    - Live Kurs-Stream via WebSocket
    - Erste Paper Order ausführen
    - Position Management
    - Order Book lesen

Warum Alpaca?
    - Kostenlos für Paper Trading
    - Commission-free für echtes Trading
    - Professionelle REST + WebSocket API
    - Perfekt für Algo Trading Bots

    # Installieren
pip install alpaca-trade-api
pip install alpaca-py
pip install alpaca-py websocket-client python-dotenv
"""

# TODO: Alpaca API Key aus Environment Variable laden
# TODO: Paper Trading Account verbinden
# TODO: Live Kurs für SPY streamen
# TODO: Market Order senden und prüfen
# TODO: Position lesen und schließen

"""
Day 28 — Alpaca API Setup & Live Paper Trading

Warum Paper Trading zuerst?
    Echtes Geld mit ungetesteten Bots zu riskieren
    ist einer der häufigsten und teuersten Fehler.
    Paper Trading = identische API, kein echtes Geld.
    Erst wenn der Bot 4 Wochen Paper Trading überlebt
    → echtes Kapital in Betracht ziehen.

Alpaca Basics:
    REST API  → Orders platzieren, Positionen lesen
    WebSocket → Live Kurse streamen, echtzeitfähig
    Paper URL → paper-api.alpaca.markets
    Live URL  → api.alpaca.markets (echtes Geld!)
"""

import os
import json
import time
import threading # mehrere Dinge gleichzeitig laufen lassen
from datetime import datetime, timedelta # Zeitberechnungen ( letzte 30 Tage)
from dataclasses import dataclass, field
from typing import Optional, Callable
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Alpaca
from alpaca.trading.client import TradingClient # kaufen/verkaufen
from alpaca.trading.requests import (
    MarketOrderRequest, # aktueller marktpreis
    LimitOrderRequest, # Limitorder
    StopOrderRequest, # Wird zu market order wenn stop price erreicht
    StopLimitOrderRequest, # kombination aus stop und limit
    GetOrdersRequest, # offene orders lesen
)
from alpaca.trading.enums import (
    OrderSide, TimeInForce, OrderStatus, # Buy/sell, wie lange gilt sie, status (filled...)
    AssetClass, PositionSide
)
from alpaca.data.historical import StockHistoricalDataClient # historische kursdaten
from alpaca.data.requests import (
    StockLatestQuoteRequest, # aktueller bid/ask preis
    StockBarsRequest, # historische kursdaten (open, high, low, close)
    StockLatestBarRequest, # aktueller bar (open, high, low, close)
)
from alpaca.data.timeframe import TimeFrame # Kerzenperiode
from alpaca.data.live import StockDataStream # Live Kurse in echtzeit

# Environment
from dotenv import load_dotenv
from pathlib import Path
load_dotenv(Path(__file__).parent / ".env")

API_KEY    = os.getenv("ALPACA_API_KEY",    "") # liest key, sonst leerer string
SECRET_KEY = os.getenv("ALPACA_SECRET_KEY", "")
BASE_URL   = os.getenv("ALPACA_BASE_URL",
                        "https://paper-api.alpaca.markets")

if not API_KEY or not SECRET_KEY:
    print("⚠ Kein API Key gefunden.")
    print("  1. Geh auf alpaca.markets")
    print("  2. Paper Trading → API Keys")
    print("  3. Kopiere Key + Secret in .env")
    print("  4. Skript neu starten")

@dataclass
class AlpacaConfig:
    """Konfiguration für den Alpaca Client."""
    api_key:    str
    secret_key: str
    paper:      bool  = True   # IMMER mit Paper starten
    base_url:   str   = "https://paper-api.alpaca.markets"


class AlpacaClient:
    """
    Wrapper um die Alpaca API.

    Warum ein Wrapper?
        Centralized Error Handling.
        Rate Limiting automatisch.
        Logging jedes API Calls.
        Einfacher zu testen und zu mocken.

    Paper vs. Live:
        paper=True  → paper-api.alpaca.markets (kein echtes Geld)
        paper=False → api.alpaca.markets       (ECHTES GELD!)

    Niemals paper=False ohne:
        1. 4 Wochen erfolgreicher Paper Trading
        2. Definiertes Max-Loss Limit
        3. Kill-Switch implementiert
    """

    def __init__(self, config: AlpacaConfig):
        self.config = config
        self._setup_clients()
        self._verify_connection()

    def _setup_clients(self):
        """Initialisiert Trading + Data Clients."""
        self.trading = TradingClient(
            api_key    = self.config.api_key, # order kaufen/verkaufen
            secret_key = self.config.secret_key,
            paper      = self.config.paper
        )
        self.data = StockHistoricalDataClient( # kursdaten holen
            api_key    = self.config.api_key,
            secret_key = self.config.secret_key,
        )
        print(f"✅ Alpaca Client initialisiert "
              f"({'PAPER' if self.config.paper else '⚠ LIVE'})")

    def _verify_connection(self):
        """Prüft ob API Verbindung funktioniert."""
        try:
            account = self.trading.get_account()
            self._account = account
            print(f"✅ Account verbunden")
            print(f"   Status:        {account.status}")
            print(f"   Portfolio:     "
                  f"${float(account.portfolio_value):,.2f}") # alpaca gibt zahlen als strings raus
            print(f"   Buying Power:  "
                  f"${float(account.buying_power):,.2f}")
            print(f"   Cash:          "
                  f"${float(account.cash):,.2f}")
        except Exception as e:
            print(f"❌ Verbindungsfehler: {e}")
            raise

    # ─── Account Info ─────────────────────────────────────
    def get_account(self) -> dict:
        """Gibt Account-Übersicht zurück."""
        acc = self.trading.get_account()
        return {
            "portfolio_value":  float(acc.portfolio_value),
            "buying_power":     float(acc.buying_power),
            "cash":             float(acc.cash),
            "equity":           float(acc.equity),
            "last_equity":      float(acc.last_equity),
            "pnl_today":        float(acc.equity) -
                                 float(acc.last_equity),
            "pnl_today_pct":    (float(acc.equity) /
                                  float(acc.last_equity) - 1) * 100, 
            "status":           str(acc.status),
            "pattern_day_trader": acc.pattern_day_trader,
        }

    # ─── Preise ───────────────────────────────────────────
    def get_latest_price(self, ticker: str) -> float:
        """Aktuellsten Preis für einen Ticker."""
        try:
            req    = StockLatestBarRequest(symbol_or_symbols=ticker)
            bar    = self.data.get_stock_latest_bar(req)
            return float(bar[ticker].close)
        except Exception as e:
            print(f"Preis Fehler {ticker}: {e}")
            return 0.0 # falls fehler

    def get_latest_prices(self, tickers: list) -> dict:
        """Preise für mehrere Ticker auf einmal."""
        # {"AAPL": 189.50, "TSLA": 245.30}
        try:
            req  = StockLatestBarRequest(
                symbol_or_symbols=tickers
            )
            bars = self.data.get_stock_latest_bar(req)
            return {
                t: float(bars[t].close)
                for t in tickers if t in bars
            }
        except Exception as e:
            print(f"Multi-Preis Fehler: {e}")
            return {}

    def get_bars(self,
                 ticker:    str,
                 timeframe: TimeFrame = TimeFrame.Day,
                 limit:     int       = 100) -> pd.DataFrame:
        """
        Historische Bars laden.

        TimeFrame.Minute  → 1-Minuten Bars
        TimeFrame.Hour    → 1-Stunden Bars
        TimeFrame.Day     → Tages-Bars
        """
        start = datetime.now() - timedelta(days=limit * 2)
        req   = StockBarsRequest(
            symbol_or_symbols = ticker,
            timeframe         = timeframe,
            start             = start,
            limit             = limit,
        )
        bars = self.data.get_stock_bars(req)
        df   = bars.df

        if isinstance(df.index, pd.MultiIndex):
            df = df.loc[ticker]

        df.index = pd.to_datetime(df.index)
        return df.sort_index()

    # ─── Positionen ───────────────────────────────────────
    def get_positions(self) -> pd.DataFrame:
        """Alle offenen Positionen."""
        positions = self.trading.get_all_positions()

        if not positions:
            return pd.DataFrame()

        rows = []
        for pos in positions:
            rows.append({
                "ticker":      pos.symbol,
                "qty":         float(pos.qty),
                "side":        str(pos.side),
                "avg_price":   float(pos.avg_entry_price),
                "current":     float(pos.current_price),
                "market_val":  float(pos.market_value),
                "pnl_abs":     float(pos.unrealized_pl),
                "pnl_pct":     float(pos.unrealized_plpc) * 100,
                "cost_basis":  float(pos.cost_basis),
            })

        return pd.DataFrame(rows)

    def get_position(self, ticker: str) -> Optional[dict]:
        """Position für einen Ticker."""
        try:
            pos = self.trading.get_open_position(ticker)
            return {
                "ticker":    pos.symbol,
                "qty":       float(pos.qty),
                "avg_price": float(pos.avg_entry_price),
                "current":   float(pos.current_price),
                "pnl_abs":   float(pos.unrealized_pl),
                "pnl_pct":   float(pos.unrealized_plpc) * 100,
            }
        except Exception:
            return None

    # ─── Orders ───────────────────────────────────────────
    def market_buy(self,
                   ticker: str,
                   qty:    float,
                   note:   str = "") -> dict:
        """
        Market Buy Order.

        qty: Anzahl Aktien (kann float sein für fractional)
        note: Optionaler Kommentar für Logging

        TimeInForce.DAY: Order verfällt am Tagesende
        TimeInForce.GTC: Good Till Cancelled
        """
        req = MarketOrderRequest(
            symbol       = ticker,
            qty          = qty,
            side         = OrderSide.BUY,
            time_in_force = TimeInForce.DAY,
        )
        order = self.trading.submit_order(req)
        result = {
            "order_id":  str(order.id),
            "ticker":    ticker,
            "qty":       qty,
            "side":      "BUY",
            "type":      "MARKET",
            "status":    str(order.status),
            "timestamp": datetime.now().isoformat(),
            "note":      note,
        }
        self._log_order(result)
        return result

    def market_sell(self,
                    ticker: str,
                    qty:    float,
                    note:   str = "") -> dict:
        """Market Sell Order."""
        req = MarketOrderRequest(
            symbol        = ticker,
            qty           = qty,
            side          = OrderSide.SELL,
            time_in_force = TimeInForce.DAY,
        )
        order = self.trading.submit_order(req)
        result = {
            "order_id":  str(order.id),
            "ticker":    ticker,
            "qty":       qty,
            "side":      "SELL",
            "type":      "MARKET",
            "status":    str(order.status),
            "timestamp": datetime.now().isoformat(),
            "note":      note,
        }
        self._log_order(result)
        return result

    def limit_buy(self,
                  ticker:      str,
                  qty:         float,
                  limit_price: float,
                  gtc:         bool = False) -> dict:
        """
        Limit Buy Order.

        limit_price: Maximalpreis den du zahlen willst.
        gtc: Good Till Cancelled — bleibt über mehrere Tage aktiv.

        Wann Limit statt Market?
            Bei illiquiden Aktien.
            Wenn du einen bestimmten Einstiegspunkt willst.
            Bei volatilen Märkten um Slippage zu vermeiden.
        """
        req = LimitOrderRequest(
            symbol        = ticker,
            qty           = qty,
            side          = OrderSide.BUY,
            time_in_force = TimeInForce.GTC if gtc
                            else TimeInForce.DAY,
            limit_price   = limit_price,
        )
        order = self.trading.submit_order(req)
        return {
            "order_id":    str(order.id),
            "ticker":      ticker,
            "qty":         qty,
            "side":        "BUY",
            "type":        "LIMIT",
            "limit_price": limit_price,
            "status":      str(order.status),
            "timestamp":   datetime.now().isoformat(),
        }

    def stop_loss_order(self,
                         ticker:      str,
                         qty:         float,
                         stop_price:  float) -> dict:
        """
      ***  Stop Loss Order — das wichtigste Risk Management Tool. ***

        Wenn Kurs unter stop_price fällt → automatischer Verkauf.
        Immer nach einer Long-Position setzen.

        WICHTIG: Stop Market (nicht Stop Limit) für garantierte Ausführung.
        Stop Limit kann bei starkem Fall nicht ausgeführt werden.
        """
        req = StopOrderRequest(
            symbol        = ticker,
            qty           = qty,
            side          = OrderSide.SELL,
            time_in_force = TimeInForce.GTC,
            stop_price    = stop_price,
        )
        order = self.trading.submit_order(req)
        return {
            "order_id":   str(order.id),
            "ticker":     ticker,
            "qty":        qty,
            "side":       "SELL",
            "type":       "STOP",
            "stop_price": stop_price,
            "status":     str(order.status),
            "timestamp":  datetime.now().isoformat(),
        }

    def close_position(self, ticker: str) -> dict:
        """
        Schließt eine komplette Position sofort. 
        Market Order für die gesamte Qty.
        """
        try:
            resp = self.trading.close_position(ticker)
            result = {
                "ticker":    ticker,
                "action":    "CLOSE",
                "status":    "submitted",
                "timestamp": datetime.now().isoformat(),
            }
            self._log_order(result)
            return result
        except Exception as e:
            return {"error": str(e), "ticker": ticker}

    def close_all_positions(self) -> list:
        """*** Schließt ALLE offenen Positionen. Kill-Switch. ***"""
        print("⚠ CLOSE ALL POSITIONS ausgeführt")
        try:
            self.trading.close_all_positions(cancel_orders=True)
            return [{"action": "CLOSE_ALL",
                     "timestamp": datetime.now().isoformat()}]
        except Exception as e:
            return [{"error": str(e)}]

    def cancel_all_orders(self) -> None:
        """Storniert alle offenen Orders."""
        self.trading.cancel_orders()
        print("✅ Alle Orders storniert")

    def get_orders(self,
                   status: str = "open",
                   limit:  int = 20) -> pd.DataFrame:
        """Gibt Orders zurück."""
        req    = GetOrdersRequest(
            status = status,
            limit  = limit,
        )
        orders = self.trading.get_orders(request_params=req)

        if not orders:
            return pd.DataFrame()

        rows = []
        for o in orders:
            rows.append({
                "order_id":   str(o.id)[:8],
                "ticker":     o.symbol,
                "qty":        float(o.qty or 0),
                "side":       str(o.side),
                "type":       str(o.type),
                "status":     str(o.status),
                "limit":      float(o.limit_price or 0),
                "stop":       float(o.stop_price or 0),
                "created":    str(o.created_at)[:19],
            })

        return pd.DataFrame(rows)

    #***  ─── Logging ────────────────────────────────────────── ***
    def _log_order(self, order: dict) -> None:
        """Loggt jede Order in CSV."""
        log_file = "trade_log.csv"
        log_df   = pd.DataFrame([order])

        if os.path.exists(log_file):
            existing = pd.read_csv(log_file)
            log_df   = pd.concat(
                [existing, log_df], ignore_index=True
            )

        log_df.to_csv(log_file, index=False)

    def load_trade_log(self) -> pd.DataFrame:
        """Lädt Trade Log."""
        if os.path.exists("trade_log.csv"):
            return pd.read_csv("trade_log.csv")
        return pd.DataFrame()
    
class LivePriceStream:
    """
    WebSocket Stream für Echtzeit-Kurse.

    Warum WebSocket statt REST Polling?
        REST: du fragst alle X Sekunden → Latenz + Rate Limits
        WebSocket: Alpaca pusht Updates sofort → echtzeitfähig

    Für Intraday-Bots ist das der Unterschied zwischen
    100ms und 5000ms Latenz.
    Für End-of-Day Bots ist REST völlig ausreichend.
    """

    def __init__(self,
                 api_key:    str,
                 secret_key: str):
        self.api_key    = api_key
        self.secret_key = secret_key
        self.prices     = {}
        self.callbacks  = []
        self._stream    = None
        self._thread    = None
        self._running   = False

    def add_callback(self, fn: Callable) -> None:
        """Fügt Callback für neue Preise hinzu."""
        self.callbacks.append(fn)
        # Das ist ein Event-System. 
        # Du registrierst Funktionen die automatisch aufgerufen werden wenn ein neuer Preis ankommt:

    async def _handle_bar(self, bar) -> None: # async: man ruft nicht selber auf, sondern wenn neuer preis reinkommt
        """Callback bei neuem Bar."""
        ticker = bar.symbol
        price  = float(bar.close)
        volume = float(bar.volume)

        self.prices[ticker] = {
            "price":  price,
            "volume": volume,
            "time":   bar.timestamp,
            "open":   float(bar.open),
            "high":   float(bar.high),
            "low":    float(bar.low),
        }

        # Alle Callbacks ausführen
        for fn in self.callbacks:
            try:
                fn(ticker, self.prices[ticker])
            except Exception as e:
                print(f"Callback Fehler: {e}")
            
            #Jede registrierte Funktion wird aufgerufen — 
            #try/except drum damit ein fehlerhafter Callback nicht den ganzen Stream zum Absturz bringt.

    def start(self, tickers: list) -> None:
        """
        Startet WebSocket Stream in separatem Thread.
        Non-blocking — dein Hauptprogramm läuft weiter.
        """
        self._stream  = StockDataStream(
            self.api_key, self.secret_key
        )
        self._running = True

        self._stream.subscribe_bars(
            self._handle_bar, *tickers
        )

        def run():
            self._stream.run()

        self._thread = threading.Thread( # würde ganzes program normal blockieren, deshalb in thread packen
            target=run, daemon=True # daemon : thread wird automatisch gestoppt wenn hauptprogramm endet
        )
        self._thread.start() # mit thread läuft es im hintergrund
        print(f"✅ Live Stream gestartet: {tickers}")

    def stop(self) -> None:
        """Stoppt den Stream sauber."""
        self._running = False
        if self._stream:
            self._stream.stop()
        print("Stream gestoppt")

    def get_price(self, ticker: str) -> Optional[float]:
        """Aktuellster Preis aus dem Stream."""
        data = self.prices.get(ticker) # gibt none statt error wenn nicht []
        return data["price"] if data else None
    
class PortfolioMonitor:
    """
    Echtzeit-Monitor für dein Paper Portfolio.

    Tracked:
        PnL intraday und kumulativ
        Position Sizes und Gewichtungen
        Risk Metrics: Max DD, Sharpe, Volatilität
        Kill-Switch: automatischer Stop bei X% Verlust
    """

    def __init__(self,
                 client:         AlpacaClient,
                 max_loss_pct:   float = 0.05,
                 alert_callback: Optional[Callable] = None):
        self.client         = client
        self.max_loss_pct   = max_loss_pct
        self.alert_callback = alert_callback

        # Portfolio History
        self.history    = []
        self.start_value = None

    def snapshot(self) -> dict:
        """Nimmt eine Portfolio-Momentaufnahme."""
        account   = self.client.get_account()
        positions = self.client.get_positions()

        portfolio_value = account["portfolio_value"]

        if self.start_value is None:
            self.start_value = portfolio_value

        total_pnl  = portfolio_value - self.start_value
        total_pct  = total_pnl / self.start_value * 100

        snapshot = {
            "timestamp":      datetime.now().isoformat(),
            "portfolio_value": portfolio_value,
            "cash":           account["cash"],
            "equity":         account["equity"],
            "pnl_today":      account["pnl_today"],
            "pnl_today_pct":  account["pnl_today_pct"],
            "total_pnl":      round(total_pnl, 2),
            "total_pct":      round(total_pct, 2),
            "n_positions":    len(positions) if not positions.empty
                              else 0,
        }

        self.history.append(snapshot)
        self._check_kill_switch(total_pct)
        return snapshot

    def _check_kill_switch(self, pnl_pct: float) -> None:
        """
        Kill-Switch — das wichtigste Risk Management Feature.

        Wenn Portfolio um mehr als max_loss_pct fällt:
            1. Alle Positionen schließen
            2. Alert senden
            3. Bot stoppen

        Das ist nicht optional.
        Jeder professionelle Algo-Trader hat einen Kill-Switch.
        Retail-Trader ohne Kill-Switch verlieren ihr Konto.
        """
        if pnl_pct < -self.max_loss_pct * 100:
            print(f"\n🚨 KILL SWITCH AKTIVIERT")
            print(f"   PnL: {pnl_pct:.2f}% (Limit: "
                  f"-{self.max_loss_pct*100:.0f}%)")
            print(f"   Schließe alle Positionen...")

            self.client.close_all_positions()
            self.client.cancel_all_orders()

            if self.alert_callback:
                self.alert_callback(
                    f"KILL SWITCH: Portfolio bei {pnl_pct:.2f}%"
                )

    def get_history_df(self) -> pd.DataFrame:
        """Portfolio History als DataFrame."""
        return pd.DataFrame(self.history)

    def print_summary(self) -> None:
        """Terminal Summary."""
        account   = self.client.get_account()
        positions = self.client.get_positions()

        print(f"\n{'='*50}")
        print(f"  PORTFOLIO SUMMARY — "
              f"{datetime.now().strftime('%H:%M:%S')}")
        print(f"{'='*50}")
        print(f"  Portfolio Wert: "
              f"${account['portfolio_value']:>10,.2f}")
        print(f"  Cash:           "
              f"${account['cash']:>10,.2f}")
        print(f"  Heute PnL:      "
              f"${account['pnl_today']:>+10,.2f} "
              f"({account['pnl_today_pct']:+.2f}%)")

        if not positions.empty:
            print(f"\n  POSITIONEN ({len(positions)})")
            print(f"  {'Ticker':<8} {'Qty':>6} "
                  f"{'Avg':>8} {'Kurs':>8} "
                  f"{'PnL':>10} {'PnL%':>7}")
            print("  " + "-"*52)
            for _, pos in positions.iterrows():
                pnl_sign = "+" if pos["pnl_abs"] >= 0 else ""
                print(f"  {pos['ticker']:<8}"
                      f"  {pos['qty']:>5.0f}"
                      f"  ${pos['avg_price']:>7.2f}"
                      f"  ${pos['current']:>7.2f}"
                      f"  {pnl_sign}${pos['pnl_abs']:>8.2f}"
                      f"  {pos['pnl_pct']:>+6.2f}%")
        else:
            print("\n  Keine offenen Positionen")

        print(f"{'='*50}")


def plot_portfolio_live(monitor:  PortfolioMonitor,
                         client:   AlpacaClient) -> None:
    """
    Live Portfolio Chart.
    Wird nicht als WebSocket upgedated —
    muss manuell neu aufgerufen werden.
    """
    history   = monitor.get_history_df()
    positions = client.get_positions()
    account   = client.get_account()

    if history.empty:
        print("Noch keine History.")
        return

    history["timestamp"] = pd.to_datetime(history["timestamp"])

    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=[
            "Portfolio Wert ($)",
            "PnL Heute (%)",
            "Position Gewichtungen",
            "Position PnL ($)",
        ],
        specs=[[{}, {}],
               [{"type": "pie"}, {}]],
        vertical_spacing=0.12,
        horizontal_spacing=0.10,
    )

    # Portfolio Wert
    fig.add_trace(go.Scatter(
        x=history["timestamp"],
        y=history["portfolio_value"].round(2),
        name="Portfolio",
        line=dict(color="#2563eb", width=2),
        fill="tozeroy",
        fillcolor="rgba(37,99,235,0.06)"
    ), row=1, col=1)

    # Startlinie
    if monitor.start_value:
        fig.add_hline(
            y=monitor.start_value,
            line_dash="dot",
            line_color="#94a3b8",
            opacity=0.6, row=1, col=1
        )

    # PnL %
    if "pnl_today_pct" in history.columns:
        pnl_colors = [
            "#16a34a" if v >= 0 else "#ef4444"
            for v in history["pnl_today_pct"]
        ]
        fig.add_trace(go.Bar(
            x=history["timestamp"],
            y=history["pnl_today_pct"].round(3),
            marker_color=pnl_colors,
            name="PnL %",
            showlegend=False
        ), row=1, col=2)

        fig.add_hline(
            y=0, line_color="#1e293b",
            line_width=1, row=1, col=2
        )

    # Position Pie
    if not positions.empty:
        total_val    = positions["market_val"].sum()
        cash         = account["cash"]
        labels       = list(positions["ticker"]) + ["Cash"]
        values       = (list(positions["market_val"]) +
                        [cash])
        colors_pie   = [
            "#2563eb","#16a34a","#f59e0b",
            "#ef4444","#8b5cf6","#0891b2",
            "#94a3b8"
        ]

        fig.add_trace(go.Pie(
            labels=labels,
            values=[round(v, 2) for v in values],
            hole=0.45,
            marker_colors=colors_pie[:len(labels)],
            textinfo="label+percent",
            showlegend=False
        ), row=2, col=1)

        # Position PnL Bars
        pos_pnl_colors = [
            "#16a34a" if v >= 0 else "#ef4444"
            for v in positions["pnl_abs"]
        ]
        fig.add_trace(go.Bar(
            x=positions["ticker"],
            y=positions["pnl_abs"].round(2),
            marker_color=pos_pnl_colors,
            text=[f"${v:+.0f}" for v in positions["pnl_abs"]],
            textposition="outside",
            name="Position PnL",
            showlegend=False
        ), row=2, col=2)

        fig.add_hline(
            y=0, line_color="#1e293b",
            line_width=1, row=2, col=2
        )

    fig.update_layout(
        height=650,
        template="plotly_white",
        title=f"Live Portfolio — "
              f"{datetime.now().strftime('%d.%m.%Y %H:%M')}",
        legend=dict(orientation="h", y=1.02),
        margin=dict(l=0, r=0, t=60, b=0)
    )

    fig.update_yaxes(title_text="Wert ($)",  row=1, col=1)
    fig.update_yaxes(title_text="PnL (%)",   row=1, col=2)
    fig.update_yaxes(title_text="PnL ($)",   row=2, col=2)

    fig.show()


def plot_trade_log(client: AlpacaClient) -> None:
    """Visualisiert Trade History."""
    log = client.load_trade_log()

    if log.empty:
        print("Noch keine Trades.")
        return

    log["timestamp"] = pd.to_datetime(log["timestamp"])

    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=[
            "Trades nach Ticker",
            "Buy vs. Sell Verteilung"
        ],
        horizontal_spacing=0.12
    )

    # Trades nach Ticker
    ticker_counts = log["ticker"].value_counts()
    fig.add_trace(go.Bar(
        x=ticker_counts.index.tolist(),
        y=ticker_counts.values.tolist(),
        marker_color="#3b82f6",
        text=ticker_counts.values.tolist(),
        textposition="outside",
        showlegend=False
    ), row=1, col=1)

    # Buy vs Sell
    if "side" in log.columns:
        side_counts = log["side"].value_counts()
        side_colors = {
            "BUY":  "#16a34a",
            "SELL": "#ef4444"
        }
        fig.add_trace(go.Pie(
            labels=side_counts.index.tolist(),
            values=side_counts.values.tolist(),
            hole=0.4,
            marker_colors=[
                side_colors.get(k, "#94a3b8")
                for k in side_counts.index
            ],
            textinfo="label+percent",
            showlegend=False
        ), row=1, col=2)

    fig.update_layout(
        height=400,
        template="plotly_white",
        title="Trade Log Analyse",
        margin=dict(l=0, r=0, t=50, b=0)
    )

    fig.show()

def is_market_open(client: AlpacaClient) -> bool:
    """
    Prüft ob der Markt offen ist.

    WICHTIG für jeden Trading Bot:
        Market Orders nur während Handelszeiten.
        Pre/After Market: andere Spreads, weniger Liquidität.
        Wochenende: kein Handel möglich.
    """
    try:
        clock = client.trading.get_clock()
        is_open = clock.is_open

        if not is_open:
            next_open = clock.next_open
            print(f"Markt geschlossen. "
                  f"Öffnet: {next_open}")
        else:
            next_close = clock.next_close
            time_left  = next_close - datetime.now(
                tz=next_close.tzinfo
            )
            hours = int(time_left.seconds / 3600)
            mins  = int((time_left.seconds % 3600) / 60)
            print(f"Markt offen. "
                  f"Schließt in {hours}h {mins}m")

        return is_open
    except Exception as e:
        print(f"Clock Fehler: {e}")
        return False


def get_market_calendar(client:     AlpacaClient,
                         start_date: str = None,
                         end_date:   str = None) -> pd.DataFrame:
    """
    Gibt Handelstage zurück.
    Nützlich um zu prüfen ob morgen ein Handelstag ist.
    """
    from alpaca.trading.requests import GetCalendarRequest
    from datetime import date

    if start_date is None:
        start_date = date.today().isoformat()
    if end_date is None:
        end_date = (date.today() +
                    timedelta(days=10)).isoformat()

    req      = GetCalendarRequest(
        start=start_date, end=end_date
    )
    calendar = client.trading.get_calendar(req)

    rows = []
    for day in calendar:
        rows.append({
            "date":  str(day.date),
            "open":  str(day.open),
            "close": str(day.close),
        })

    return pd.DataFrame(rows)


if __name__ == "__main__":

    print("Tag 28 — Alpaca API Setup")
    print("=" * 55)

    # --- Verbindung ---
    if not API_KEY or not SECRET_KEY:
        print("\n⚠ Keine API Keys. Bitte .env einrichten.")
        print("  Erstelle .env mit:")
        print("  ALPACA_API_KEY=dein_key")
        print("  ALPACA_SECRET_KEY=dein_secret")
        exit()

    config = AlpacaConfig(
        api_key    = API_KEY,
        secret_key = SECRET_KEY,
        paper      = True    # IMMER True für Anfänger
    )

    client  = AlpacaClient(config)
    monitor = PortfolioMonitor(
        client,
        max_loss_pct = 0.05   # Kill Switch bei -5%
    )

    # --- Account Info ---
    print("\n1. Account Overview")
    account = client.get_account()
    for k, v in account.items():
        if isinstance(v, float):
            print(f"   {k:<25} ${v:>10,.2f}"
                  if "pct" not in k
                  else f"   {k:<25} {v:>+9.2f}%")

    # --- Markt Status ---
    print("\n2. Markt Status")
    market_open = is_market_open(client)

    # --- Kalender ---
    print("\n3. Kommende Handelstage")
    cal = get_market_calendar(client)
    print(cal.head(5).to_string(index=False))

    # --- Aktuelle Preise ---
    print("\n4. Live Preise")
    watchlist = ["AAPL", "MSFT", "NVDA", "SPY", "QQQ"]
    prices    = client.get_latest_prices(watchlist)

    for ticker, price in prices.items():
        print(f"   {ticker:<8} ${price:>8.2f}")

    # --- Historische Bars ---
    print("\n5. Historische Daten (AAPL, letzte 10 Tage)")
    bars = client.get_bars("AAPL", TimeFrame.Day, limit=10)
    if not bars.empty:
        print(bars[["open", "high", "low",
                     "close", "volume"]].tail(5).round(2))

    # --- Positionen ---
    print("\n6. Aktuelle Positionen")
    positions = client.get_positions()
    if positions.empty:
        print("   Keine offenen Positionen")
    else:
        print(positions[[
            "ticker", "qty", "avg_price",
            "current", "pnl_abs", "pnl_pct"
        ]].to_string(index=False))

    # --- Orders ---
    print("\n7. Offene Orders")
    orders = client.get_orders("open")
    if orders.empty:
        print("   Keine offenen Orders")
    else:
        print(orders.to_string(index=False))

    # --- Erste Test-Order ---
    print("\n8. Erste Paper Order")
    if market_open:
        print("   Sende Market Buy: 1 Aktie AAPL...")
        order = client.market_buy(
            ticker = "AAPL",
            qty    = 1,
            note   = "Day 28 Test Order"
        )
        print(f"   Order ID:  {order['order_id']}")
        print(f"   Status:    {order['status']}")

        time.sleep(2)

        # Position prüfen
        pos = client.get_position("AAPL")
        if pos:
            print(f"\n   Position:")
            print(f"   AAPL: {pos['qty']} Aktien "
                  f"@ ${pos['avg_price']:.2f}")
            print(f"   PnL:  ${pos['pnl_abs']:+.2f} "
                  f"({pos['pnl_pct']:+.2f}%)")

        # Stop Loss setzen
        if pos:
            stop_price = round(
                pos["avg_price"] * 0.97, 2
            )   # 3% Stop Loss
            sl = client.stop_loss_order(
                "AAPL", 1, stop_price
            )
            print(f"\n   Stop Loss gesetzt: ${stop_price:.2f}")
            print(f"   (3% unter Einstieg)")

        # Monitor Snapshot
        print("\n   Portfolio Snapshot:")
        snap = monitor.snapshot()
        print(f"   Wert:     ${snap['portfolio_value']:,.2f}")
        print(f"   PnL/Tag:  ${snap['pnl_today']:+.2f}")

        # Position nach 5 Sekunden schließen
        print("\n   Warte 5 Sekunden...")
        time.sleep(5)

        print("   Schließe AAPL Position...")
        close_result = client.close_position("AAPL")
        print(f"   {close_result}")

    else:
        print("   Markt geschlossen — kein Order Test")
        print("   Führe Order Tests während Marktzeiten aus")
        print("   US Markt: Mo–Fr 15:30–22:00 Uhr MEZ")

    # --- Portfolio Chart ---
    print("\n9. Portfolio Visualisierung")
    monitor.snapshot()  # zweiter Snapshot für History
    plot_portfolio_live(monitor, client)

    # --- Trade Log ---
    print("\n10. Trade Log")
    log = client.load_trade_log()
    if not log.empty:
        print(log[["ticker", "side", "qty",
                    "status", "timestamp"]].to_string(
            index=False
        ))
        plot_trade_log(client)

    # --- Summary ---
    monitor.print_summary()

    print("\n" + "="*55)
    print("LIVE TRADING REGELN — NIEMALS VERGESSEN")
    print("="*55)
    print("  1. Immer mit Paper Trading beginnen")
    print("  2. Kill Switch immer aktiv (max -5% Portfolio)")
    print("  3. Niemals mehr als 2% pro Trade riskieren")
    print("  4. Stop Loss bei JEDER Long Position")
    print("  5. Kein echtes Geld ohne 4 Wochen Paper Daten")
    print("  6. API Keys NIEMALS in Git committen")
    print("  7. .env in .gitignore — immer")