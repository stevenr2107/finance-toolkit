"""
Day 29 — Trading Bot V1: Moving Average Crossover

Architektur:
    Signal Engine  → berechnet Indikatoren + Signale
    Risk Manager   → prüft ob Trade erlaubt ist
    Order Engine   → führt Orders aus
    Position Manager → verwaltet offene Positionen
    Logger         → loggt alles in CSV + Terminal

Warum MA Crossover als ersten Bot?
    Einfach zu verstehen — keine Black Box.
    Klare Entry/Exit Regeln.
    Bewährt auf vielen Märkten.
    Leicht zu debuggen wenn etwas schiefläuft.

    Komplexe Strategien kommen erst wenn du
    verstehst wie der Bot-Loop funktioniert.
    Erst gehen, dann laufen.

Signal:
    SMA_fast > SMA_slow → Long Signal
    SMA_fast < SMA_slow → Kein Signal (Cash)
    Täglich um 21:55 Uhr MEZ geprüft (vor Marktschluss)
"""

import os
import time
import logging
import json
import threading
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from typing import Optional, Dict, List
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from dotenv import load_dotenv

# Alpaca
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import (
    MarketOrderRequest, StopOrderRequest,
    GetOrdersRequest
)
from alpaca.trading.enums import (
    OrderSide, TimeInForce
)
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame

load_dotenv()

# ── Logging Setup ─────────────────────────────────────────────
logging.basicConfig(
    level   = logging.INFO,
    format  = "%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.FileHandler("bot_v1.log"),
        logging.StreamHandler()
    ]
)
log = logging.getLogger("TradingBot")

API_KEY    = os.getenv("ALPACA_API_KEY",    "")
SECRET_KEY = os.getenv("ALPACA_SECRET_KEY", "")

@dataclass
class BotConfig:
    """
    Alle Bot-Parameter an einem Ort.

    Änderungen hier → Änderungen überall.
    Niemals Magic Numbers im Code verteilen.
    """
    # Universe
    tickers:         List[str] = field(
        default_factory=lambda: ["AAPL", "MSFT", "NVDA"]
    )

    # Strategie
    sma_fast:        int   = 20    # Fast MA Periode
    sma_slow:        int   = 50    # Slow MA Periode

    # Risk Management
    capital_per_trade: float = 0.10  # 10% des Portfolios pro Trade
    max_positions:     int   = 3     # Maximal 3 gleichzeitige Positionen
    stop_loss_pct:     float = 0.05  # 5% Stop Loss
    take_profit_pct:   float = 0.15  # 15% Take Profit
    max_portfolio_loss: float = 0.10 # Kill Switch bei -10% Portfolio

    # Timing
    check_interval_sec: int  = 60   # Alle 60 Sekunden prüfen
    bars_needed:        int  = 60   # Mindest-Bars für MA

    # Paper Trading
    paper:              bool = True  # IMMER True bis du bereit bist

class SignalEngine:
    """
    Berechnet Trading Signale für alle Ticker.

    Separation of Concerns:
        Signal Engine weiß NICHTS von Orders.
        Sie gibt nur zurück: long / flat.
        Der Rest ist Sache der anderen Komponenten.

    Das macht Testing einfach:
        Du kannst Signal Engine isoliert testen
        ohne echte API Calls.
    """

    def __init__(self,
                 data_client: StockHistoricalDataClient,
                 config:      BotConfig):
        self.data   = data_client
        self.config = config

    def get_bars(self, ticker: str) -> pd.DataFrame:
        """Lädt aktuelle Bars für Indikator-Berechnung."""
        start = datetime.now() - timedelta(
            days=self.config.bars_needed * 2
        )
        req  = StockBarsRequest(
            symbol_or_symbols = ticker,
            timeframe         = TimeFrame.Day,
            start             = start,
            limit             = self.config.bars_needed + 10,
        )
        bars = self.data.get_stock_bars(req)
        df   = bars.df

        if isinstance(df.index, pd.MultiIndex): # schaut ob index einn multiindex ist 
            df = df.loc[ticker]

        df.index = pd.to_datetime(df.index)
        return df.sort_index()

    def compute_signal(self, ticker: str) -> dict:
        """
        Berechnet Signal für einen Ticker.

        Returns:
            signal:       "long" | "flat"
            sma_fast:     aktueller Fast MA Wert
            sma_slow:     aktueller Slow MA Wert
            current_price: letzter Schlusskurs
            cross:        "golden" | "death" | "none"
            strength:     0.0 bis 1.0 — wie stark ist das Signal?
        """
        try:
            df    = self.get_bars(ticker)
            close = df["close"]

            if len(close) < self.config.sma_slow: # schaut ob genug daten vorhanden sind 
                log.warning(f"{ticker}: Nicht genug Bars "
                             f"({len(close)} < "
                             f"{self.config.sma_slow})")
                return {"signal": "flat", "ticker": ticker,
                        "error": "insufficient_data"}

            sma_fast = close.rolling(
                self.config.sma_fast
            ).mean()
            sma_slow = close.rolling(
                self.config.sma_slow
            ).mean()

            current_price = float(close.iloc[-1])
            fast_now      = float(sma_fast.iloc[-1])
            slow_now      = float(sma_slow.iloc[-1])
            fast_prev     = float(sma_fast.iloc[-2])
            slow_prev     = float(sma_slow.iloc[-2])

            # Signal
            signal = "long" if fast_now > slow_now else "flat"

            # Crossover Detection
            if fast_prev <= slow_prev and fast_now > slow_now:
                cross = "golden"     # Frischer Bullish Crossover
            elif fast_prev >= slow_prev and fast_now < slow_now:
                cross = "death"      # Frischer Bearish Crossover
            else:
                cross = "none"

            # Signalstärke: Abstand der MAs als % des Kurses
            # schaut wie weit die mas auseinander liegen 
            strength = abs(fast_now - slow_now) / slow_now

            # RSI als zusätzlicher Filter
            delta    = close.diff()
            gain     = delta.clip(lower=0)
            loss     = -delta.clip(upper=0)
            avg_gain = gain.ewm(com=13, adjust=False).mean()
            avg_loss = loss.ewm(com=13, adjust=False).mean()
            rs       = avg_gain / avg_loss
            rsi      = float((100 - (100 / (1 + rs))).iloc[-1])

            return {
                "ticker":        ticker,
                "signal":        signal,
                "cross":         cross,
                "sma_fast":      round(fast_now, 2),
                "sma_slow":      round(slow_now, 2),
                "current_price": round(current_price, 2),
                "strength":      round(strength, 4),
                "rsi":           round(rsi, 1),
                "timestamp":     datetime.now().isoformat(),
                "bars_used":     len(close),
            }

        except Exception as e:
            log.error(f"Signal Fehler {ticker}: {e}")
            return {"signal": "flat", "ticker": ticker,
                    "error": str(e)}

    def get_all_signals(self) -> Dict[str, dict]:
        """Berechnet Signale für alle Ticker im Universe."""
        signals = {}
        for ticker in self.config.tickers:
            signals[ticker] = self.compute_signal(ticker)
            time.sleep(0.3)   # Rate Limiting
        return signals


class RiskManager:
    """
    Prüft jeden Trade bevor er ausgeführt wird.

    Regeln:
        1. Kein Trade wenn Portfolio-Loss > max_portfolio_loss
        2. Max positions_count gleichzeitige Positionen
        3. Position Size = capital_per_trade % des Portfolios
        4. Kein Kauf wenn RSI > 75 (überkauft)
        5. Kein Kauf wenn Spread zu groß (illiquide)

    Der Risk Manager ist der einzige Component
    der "Nein" sagen kann — und er wird gehört.
    """

    def __init__(self,
                 trading_client: TradingClient,
                 config:         BotConfig):
        self.trading  = trading_client
        self.config   = config
        self._blocked = False

    def get_portfolio_value(self) -> float:
        """Aktueller Portfolio-Wert."""
        acc = self.trading.get_account()
        return float(acc.portfolio_value)

    def get_buying_power(self) -> float:
        """Verfügbares Kapital."""
        acc = self.trading.get_account()
        return float(acc.buying_power)

    def get_position_count(self) -> int:
        """Anzahl offener Positionen."""
        positions = self.trading.get_all_positions()
        return len(positions)

    def is_already_invested(self, ticker: str) -> bool:
        """Prüft ob bereits eine Position offen ist."""
        try:
            self.trading.get_open_position(ticker)
            return True
        except Exception:
            return False

    def calculate_shares(self,
                          ticker: str,
                          price:  float) -> int:
        """
        Berechnet Anzahl kaufbarer Aktien.

        Formel:
            Max Dollar Amount = Portfolio Value × capital_per_trade
            Shares = Max Dollar Amount / Price
            Abgerundet auf ganze Aktien.
        """
        portfolio_value = self.get_portfolio_value()
        max_dollar      = portfolio_value * \
                          self.config.capital_per_trade
        shares          = int(max_dollar / price)
        return max(shares, 0)

    def check_daily_loss(self) -> bool:
        """
        Prüft ob täglicher Verlust-Limit erreicht ist.
        Gibt False zurück wenn Trading gestoppt werden soll.
        """
        try:
            acc         = self.trading.get_account()
            equity      = float(acc.equity)
            last_equity = float(acc.last_equity)
            daily_loss  = (equity / last_equity) - 1

            if daily_loss < -self.config.max_portfolio_loss:
                log.critical(
                    f"🚨 MAX PORTFOLIO LOSS ERREICHT: "
                    f"{daily_loss*100:.2f}%. "
                    f"Trading gestoppt."
                )
                self._blocked = True
                return False

            return True
        except Exception as e:
            log.error(f"Daily Loss Check Fehler: {e}")
            return True   # Im Zweifel: weitermachen

    def can_trade(self,
                   ticker: str,
                   signal: dict) -> tuple:
        """
        Haupt-Check: Darf dieser Trade ausgeführt werden?

        Returns: (bool, str) — (darf_traden, grund)
        """
        # Kill Switch
        if self._blocked:
            return False, "BOT BLOCKED — Max Loss erreicht"

        # Daily Loss Check
        if not self.check_daily_loss():
            return False, "Daily Loss Limit erreicht"

        # Bereits investiert?
        if self.is_already_invested(ticker):
            return False, f"Bereits in {ticker} investiert"

        # Max Positionen
        n_pos = self.get_position_count()
        if n_pos >= self.config.max_positions:
            return False, f"Max Positionen erreicht ({n_pos})"

        # Buying Power
        price  = signal.get("current_price", 0)
        shares = self.calculate_shares(ticker, price)
        if shares < 1:
            return False, "Nicht genug Buying Power"

        # RSI Filter: nicht kaufen wenn überkauft
        rsi = signal.get("rsi", 50)
        if rsi > 75:
            return False, f"RSI zu hoch ({rsi:.1f} > 75)"

        # Signalstärke
        strength = signal.get("strength", 0)
        if strength < 0.001:
            return False, f"Signal zu schwach ({strength:.4f})"

        return True, "OK"
    

class OrderEngine:
    """
    Führt Orders aus und setzt automatisch Stop Loss.

    Jeder Kauf bekommt sofort:
        1. Stop Loss Order (GTC)
        2. Take Profit Order (Limit GTC)

    Das ist OCO-ähnlich (One-Cancels-Other).
    In echter Implementierung: Bracket Orders.
    Alpaca bietet Bracket Orders direkt an.
    """

    def __init__(self,
                 trading_client: TradingClient,
                 config:         BotConfig):
        self.trading = trading_client
        self.config  = config

    def buy_with_stops(self,
                        ticker:    str,
                        shares:    int,
                        price:     float,
                        signal:    dict) -> dict:
        """
        Kauft Aktien und setzt sofort Stop Loss + Take Profit.

        Bracket Order Logik:
            Entry:       Market Buy
            Stop Loss:   price × (1 - stop_loss_pct)
            Take Profit: price × (1 + take_profit_pct)
        """
        stop_price   = round(
            price * (1 - self.config.stop_loss_pct), 2
        )
        profit_price = round(
            price * (1 + self.config.take_profit_pct), 2
        )

        # Bracket Order — Entry + Stop + Limit in einem
        from alpaca.trading.requests import (
            MarketOrderRequest, TakeProfitRequest,
            StopLossRequest
        )

        try:
            req = MarketOrderRequest(
                symbol        = ticker,
                qty           = shares,
                side          = OrderSide.BUY,
                time_in_force = TimeInForce.DAY,
                order_class   = "bracket",
                take_profit   = TakeProfitRequest(
                    limit_price=profit_price
                ),
                stop_loss     = StopLossRequest(
                    stop_price=stop_price
                ),
            )
            order = self.trading.submit_order(req)

            result = {
                "action":       "BUY",
                "ticker":       ticker,
                "shares":       shares,
                "entry_price":  price,
                "stop_loss":    stop_price,
                "take_profit":  profit_price,
                "order_id":     str(order.id),
                "status":       str(order.status),
                "signal":       signal.get("cross", "none"),
                "sma_fast":     signal.get("sma_fast", 0),
                "sma_slow":     signal.get("sma_slow", 0),
                "rsi":          signal.get("rsi", 0),
                "timestamp":    datetime.now().isoformat(),
            }

            log.info(
                f"BUY {ticker}: {shares} Aktien @ ${price:.2f} | "
                f"SL: ${stop_price:.2f} | "
                f"TP: ${profit_price:.2f}"
            )
            return result

        except Exception as e:
            log.error(f"Buy Fehler {ticker}: {e}")

            # Fallback: einfache Market Order
            try:
                req   = MarketOrderRequest(
                    symbol        = ticker,
                    qty           = shares,
                    side          = OrderSide.BUY,
                    time_in_force = TimeInForce.DAY,
                )
                order = self.trading.submit_order(req)

                # Manuell Stop Loss Order senden
                time.sleep(1)
                sl_req = StopOrderRequest(
                    symbol        = ticker,
                    qty           = shares,
                    side          = OrderSide.SELL,
                    time_in_force = TimeInForce.GTC,
                    stop_price    = stop_price,
                )
                self.trading.submit_order(sl_req)

                return {
                    "action":    "BUY",
                    "ticker":    ticker,
                    "shares":    shares,
                    "entry_price": price,
                    "stop_loss": stop_price,
                    "order_id":  str(order.id),
                    "status":    str(order.status),
                    "timestamp": datetime.now().isoformat(),
                    "note":      "Fallback ohne Bracket"
                }

            except Exception as e2:
                log.error(f"Fallback Buy Fehler: {e2}")
                return {"action": "ERROR", "error": str(e2)}

    def close_position(self,
                        ticker: str,
                        reason: str = "") -> dict:
        """Schließt eine Position sofort."""
        try:
            self.trading.close_position(ticker)
            # Alle zugehörigen Orders stornieren
            self.trading.cancel_orders()

            result = {
                "action":    "SELL",
                "ticker":    ticker,
                "reason":    reason,
                "timestamp": datetime.now().isoformat(),
            }
            log.info(f"SELL {ticker} — Grund: {reason}")
            return result

        except Exception as e:
            log.error(f"Close Fehler {ticker}: {e}")
            return {"action": "ERROR", "error": str(e)}
        
class PositionManager:
    """
    Verwaltet offene Positionen und trifft Exit-Entscheidungen.

    Checks bei jedem Loop:
        1. Stop Loss noch aktiv?
        2. Take Profit erreicht?
        3. MA Crossover Signal umgekehrt?
        4. Max Haltedauer überschritten?
    """

    def __init__(self,
                 trading_client: TradingClient,
                 order_engine:   OrderEngine,
                 config:         BotConfig):
        self.trading = trading_client
        self.orders  = order_engine
        self.config  = config
        self._trade_history: List[dict] = []

    def check_positions(self,
                         signals: Dict[str, dict]) -> List[dict]:
        """
        Prüft alle offenen Positionen gegen aktuelle Signale.
        Gibt Liste aller ausgeführten Exits zurück.
        """
        try:
            positions = self.trading.get_all_positions()
        except Exception as e:
            log.error(f"Positionen Fehler: {e}")
            return []

        exits = []

        for pos in positions:
            ticker  = pos.symbol
            pnl_pct = float(pos.unrealized_plpc) * 100
            signal  = signals.get(ticker, {})

            exit_reason = None

            # Signal hat sich umgekehrt → Exit
            if signal.get("signal") == "flat":
                exit_reason = f"Signal flat ({signal.get('cross', 'none')})"

            # Death Cross → sofort raus
            if signal.get("cross") == "death":
                exit_reason = "Death Cross Signal"

            # RSI überkauft bei bestehender Position → Gewinn mitnehmen
            rsi = signal.get("rsi", 50)
            if rsi > 78 and pnl_pct > 5:
                exit_reason = f"RSI überkauft ({rsi:.1f}) + Gewinn"

            if exit_reason:
                result = self.orders.close_position(
                    ticker, exit_reason
                )
                result["pnl_pct"] = pnl_pct
                exits.append(result)
                self._trade_history.append(result)

        return exits

    def get_trade_history(self) -> pd.DataFrame:
        """Trade History als DataFrame."""
        if not self._trade_history:
            return pd.DataFrame()
        return pd.DataFrame(self._trade_history)
    

class TradingBotV1:
    """
    Der vollständige Trading Bot.

    Loop-Logik (alle check_interval_sec):
        1. Markt offen?
        2. Kill Switch aktiv?
        3. Signale berechnen
        4. Positionen managen (Exits)
        5. Neue Entries prüfen
        6. Logging + Monitoring

    Threading:
        Bot läuft in eigenem Thread.
        Haupt-Thread kann weiter laufen.
        Sauberes Stoppen via stop() Methode.
    """

    def __init__(self, config: BotConfig):
        self.config = config
        self._running = False
        self._thread  = None
        self._loop_count = 0
        self._trade_log: List[dict] = []

        # Clients
        self.trading_client = TradingClient(
            api_key    = API_KEY,
            secret_key = SECRET_KEY,
            paper      = config.paper
        )
        self.data_client = StockHistoricalDataClient(
            api_key    = API_KEY,
            secret_key = SECRET_KEY,
        )

        # Komponenten
        self.signals   = SignalEngine(
            self.data_client, config
        )
        self.risk      = RiskManager(
            self.trading_client, config
        )
        self.orders    = OrderEngine(
            self.trading_client, config
        )
        self.positions = PositionManager(
            self.trading_client, self.orders, config
        )

        log.info("Trading Bot V1 initialisiert")
        log.info(f"Universe:    {config.tickers}")
        log.info(f"Strategie:   SMA {config.sma_fast}/{config.sma_slow}")
        log.info(f"Paper:       {config.paper}")

    def _is_market_open(self) -> bool:
        """Prüft ob Markt handelsbereit ist."""
        try:
            clock = self.trading_client.get_clock()
            return bool(clock.is_open)
        except Exception:
            return False

    def _run_loop(self) -> None:
        """Haupt-Loop des Bots."""
        while self._running:
            self._loop_count += 1
            loop_start = datetime.now()

            log.info(f"─── Loop #{self._loop_count} ───")

            try:
                # 1. Markt offen?
                if not self._is_market_open():
                    log.info("Markt geschlossen — warte...")
                    time.sleep(self.config.check_interval_sec)
                    continue

                # 2. Kill Switch
                if not self.risk.check_daily_loss():
                    log.critical("Kill Switch aktiv — Bot gestoppt")
                    self.stop()
                    break

                # 3. Alle Signale berechnen
                log.info("Berechne Signale...")
                all_signals = self.signals.get_all_signals()

                for ticker, sig in all_signals.items():
                    if "error" not in sig:
                        log.info(
                            f"  {ticker}: "
                            f"Signal={sig['signal'].upper()} | "
                            f"Cross={sig['cross']} | "
                            f"RSI={sig['rsi']:.1f} | "
                            f"Preis=${sig['current_price']:.2f}"
                        )

                # 4. Positionen managen
                exits = self.positions.check_positions(
                    all_signals
                )
                for exit_ in exits:
                    log.info(
                        f"EXIT: {exit_['ticker']} — "
                        f"{exit_.get('reason', '')}"
                    )
                    self._trade_log.append(exit_)

                # 5. Neue Entries
                for ticker, signal in all_signals.items():
                    if signal.get("signal") != "long":
                        continue
                    if signal.get("cross") != "golden":
                        # Nur bei frischem Golden Cross einsteigen
                        # Bestehender Trend = bereits drin oder zu spät
                        continue

                    # Risk Check
                    can_trade, reason = self.risk.can_trade(
                        ticker, signal
                    )

                    if not can_trade:
                        log.info(
                            f"SKIP {ticker}: {reason}"
                        )
                        continue

                    # Position Size
                    price  = signal["current_price"]
                    shares = self.risk.calculate_shares(
                        ticker, price
                    )

                    if shares < 1:
                        log.info(
                            f"SKIP {ticker}: "
                            f"0 Shares kalkuliert"
                        )
                        continue

                    # Order ausführen
                    log.info(
                        f"ENTRY: {ticker} "
                        f"{shares} Aktien @ ${price:.2f}"
                    )
                    result = self.orders.buy_with_stops(
                        ticker, shares, price, signal
                    )
                    self._trade_log.append(result)

                # 6. Loop Zeit messen
                elapsed = (datetime.now() -
                            loop_start).total_seconds()
                log.info(
                    f"Loop #{self._loop_count} fertig "
                    f"in {elapsed:.1f}s"
                )

                # Warten bis zum nächsten Loop
                wait = max(
                    self.config.check_interval_sec - elapsed,
                    1
                )
                time.sleep(wait)

            except Exception as e:
                log.error(f"Loop Fehler: {e}", exc_info=True)
                time.sleep(30)  # kurze Pause bei Fehler

    def start(self) -> None:
        """Startet den Bot in einem Thread."""
        if self._running:
            log.warning("Bot läuft bereits")
            return

        self._running = True
        self._thread  = threading.Thread(
            target=self._run_loop, daemon=True
        )
        self._thread.start()
        log.info("✅ Bot gestartet")

    def stop(self) -> None:
        """Stoppt den Bot sauber."""
        self._running = False
        log.info("Bot gestoppt")

    def get_trade_log(self) -> pd.DataFrame:
        """Alle Trades als DataFrame."""
        if not self._trade_log:
            return pd.DataFrame()
        return pd.DataFrame(self._trade_log)

    def get_status(self) -> dict:
        """Aktueller Bot-Status."""
        try:
            acc       = self.trading_client.get_account()
            positions = self.trading_client.get_all_positions()

            return {
                "running":       self._running,
                "loop_count":    self._loop_count,
                "portfolio":     float(acc.portfolio_value),
                "cash":          float(acc.cash),
                "n_positions":   len(positions),
                "n_trades":      len(self._trade_log),
                "paper":         self.config.paper,
                "timestamp":     datetime.now().isoformat(),
            }
        except Exception as e:
            return {"error": str(e)}

def plot_bot_performance(bot:     TradingBotV1,
                          client:  TradingClient) -> None:
    """
    Vollständiges Bot Performance Dashboard.
    """
    trade_log = bot.get_trade_log()
    status    = bot.get_status()

    try:
        acc       = client.get_account()
        positions = client.get_all_positions()
    except Exception:
        print("API nicht verfügbar")
        return

    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=[
            "Signal Overview",
            "Trade Log",
            "Aktuelle Positionen",
            "Bot Status"
        ],
        vertical_spacing=0.14,
        horizontal_spacing=0.10
    )

    # Panel 1: Signal Übersicht
    signals = bot.signals.get_all_signals()

    tickers_list = list(signals.keys())
    rsi_values   = [
        signals[t].get("rsi", 50) for t in tickers_list
    ]
    signal_colors = [
        "#16a34a" if signals[t].get("signal") == "long"
        else "#ef4444"
        for t in tickers_list
    ]

    fig.add_trace(go.Bar(
        x=tickers_list,
        y=rsi_values,
        marker_color=signal_colors,
        text=[
            f"{signals[t].get('signal', '?').upper()}"
            for t in tickers_list
        ],
        textposition="outside",
        name="Signal",
        showlegend=False
    ), row=1, col=1)

    fig.add_hline(
        y=70, line_dash="dash",
        line_color="#ef4444", opacity=0.5,
        row=1, col=1
    )
    fig.add_hline(
        y=30, line_dash="dash",
        line_color="#16a34a", opacity=0.5,
        row=1, col=1
    )

    # Panel 2: Trade Log
    if not trade_log.empty and "action" in trade_log.columns:
        action_counts = trade_log["action"].value_counts()
        action_colors = {
            "BUY":   "#16a34a",
            "SELL":  "#ef4444",
            "ERROR": "#94a3b8",
        }
        fig.add_trace(go.Bar(
            x=action_counts.index.tolist(),
            y=action_counts.values.tolist(),
            marker_color=[
                action_colors.get(k, "#3b82f6")
                for k in action_counts.index
            ],
            text=action_counts.values.tolist(),
            textposition="outside",
            showlegend=False
        ), row=1, col=2)
    else:
        fig.add_annotation(
            x=0.5, y=0.5,
            xref="x2 domain", yref="y2 domain",
            text="Noch keine Trades",
            showarrow=False, row=1, col=2
        )

    # Panel 3: Positionen
    if positions:
        pos_tickers = [p.symbol for p in positions]
        pos_pnl     = [
            float(p.unrealized_plpc) * 100
            for p in positions
        ]
        pos_colors  = [
            "#16a34a" if v >= 0 else "#ef4444"
            for v in pos_pnl
        ]

        fig.add_trace(go.Bar(
            x=pos_tickers,
            y=pos_pnl,
            marker_color=pos_colors,
            text=[f"{v:+.2f}%" for v in pos_pnl],
            textposition="outside",
            name="PnL %",
            showlegend=False
        ), row=2, col=1)

        fig.add_hline(
            y=0, line_color="#1e293b",
            line_width=1, row=2, col=1
        )
    else:
        fig.add_annotation(
            x=0.5, y=0.5,
            xref="x3 domain", yref="y3 domain",
            text="Keine Positionen",
            showarrow=False
        )

    # Panel 4: Status Metriken
    status_labels = [
        "Portfolio ($)",
        "Cash ($)",
        "Positionen",
        "Trades",
        "Loops"
    ]
    status_values = [
        status.get("portfolio", 0),
        status.get("cash", 0),
        status.get("n_positions", 0),
        status.get("n_trades", 0),
        status.get("loop_count", 0),
    ]

    fig.add_trace(go.Bar(
        x=status_labels,
        y=status_values,
        marker_color="#3b82f6",
        text=[
            f"${v:,.0f}" if i < 2 else str(int(v))
            for i, v in enumerate(status_values)
        ],
        textposition="outside",
        showlegend=False
    ), row=2, col=2)

    fig.update_layout(
        height=650,
        template="plotly_white",
        title=f"Trading Bot V1 — "
              f"{'🟢 LIVE' if status.get('running') else '🔴 STOPPED'} "
              f"({'PAPER' if config.paper else '⚠ LIVE'})",
        margin=dict(l=0, r=0, t=60, b=0)
    )

    fig.update_yaxes(title_text="RSI",     row=1, col=1,
                     range=[0, 100])
    fig.update_yaxes(title_text="Anzahl",  row=1, col=2)
    fig.update_yaxes(title_text="PnL (%)", row=2, col=1)
    fig.update_yaxes(title_text="Wert",    row=2, col=2)

    fig.show()


def plot_signals_detail(bot: TradingBotV1) -> None:
    """
    Detaillierter Signal Chart für alle Ticker.
    Kurs + SMA Fast + SMA Slow für jeden Ticker.
    """
    n        = len(bot.config.tickers)
    fig      = make_subplots(
        rows=n, cols=1,
        shared_xaxes=False,
        vertical_spacing=0.06,
        subplot_titles=bot.config.tickers
    )

    colors = {
        "long": "#16a34a",
        "flat": "#ef4444",
    }

    for row_idx, ticker in enumerate(
        bot.config.tickers, start=1
    ):
        try:
            df      = bot.signals.get_bars(ticker)
            close   = df["close"]
            sma_f   = close.rolling(bot.config.sma_fast).mean()
            sma_s   = close.rolling(bot.config.sma_slow).mean()
            signal  = bot.signals.compute_signal(ticker)
            sig_col = colors.get(
                signal.get("signal", "flat"), "#94a3b8"
            )

            # Kurs
            fig.add_trace(go.Scatter(
                x=df.index, y=close.round(2),
                name=ticker,
                line=dict(color="#1e293b", width=1.5),
                showlegend=False
            ), row=row_idx, col=1)

            # Fast SMA
            fig.add_trace(go.Scatter(
                x=df.index, y=sma_f.round(2),
                name=f"SMA{bot.config.sma_fast}",
                line=dict(color="#3b82f6", width=1.2),
                showlegend=row_idx == 1
            ), row=row_idx, col=1)

            # Slow SMA
            fig.add_trace(go.Scatter(
                x=df.index, y=sma_s.round(2),
                name=f"SMA{bot.config.sma_slow}",
                line=dict(color="#f59e0b", width=1.2),
                showlegend=row_idx == 1
            ), row=row_idx, col=1)

            # Signal Annotation
            fig.add_annotation(
                x=df.index[-1],
                y=float(close.iloc[-1]),
                text=f" {signal.get('signal','?').upper()}",
                showarrow=False,
                font=dict(color=sig_col, size=12),
                row=row_idx, col=1
            )

        except Exception as e:
            log.error(f"Plot Fehler {ticker}: {e}")

    fig.update_layout(
        height=280 * n,
        template="plotly_white",
        hovermode="x unified",
        title="Signal Detail — MA Crossover",
        legend=dict(orientation="h", y=1.01),
        margin=dict(l=0, r=60, t=60, b=0)
    )

    fig.show()

if __name__ == "__main__":

    print("Tag 29 — Trading Bot V1")
    print("=" * 55)

    if not API_KEY or not SECRET_KEY:
        print("⚠ Keine API Keys. .env prüfen.")
        exit()

    # --- Konfiguration ---
    config = BotConfig(
        tickers          = ["AAPL", "MSFT", "NVDA"],
        sma_fast         = 20,
        sma_slow         = 50,
        capital_per_trade  = 0.10,   # 10% pro Trade
        max_positions    = 3,
        stop_loss_pct    = 0.05,     # 5% Stop Loss
        take_profit_pct  = 0.15,     # 15% Take Profit
        max_portfolio_loss = 0.10,   # Kill Switch bei -10%
        check_interval_sec = 60,     # Jede Minute prüfen
        paper            = True      # IMMER True!
    )

    print(f"\nKonfiguration:")
    print(f"  Universe:    {config.tickers}")
    print(f"  Strategie:   SMA {config.sma_fast}/{config.sma_slow}")
    print(f"  Pro Trade:   {config.capital_per_trade*100:.0f}% Portfolio")
    print(f"  Stop Loss:   {config.stop_loss_pct*100:.0f}%")
    print(f"  Take Profit: {config.take_profit_pct*100:.0f}%")
    print(f"  Kill Switch: -{config.max_portfolio_loss*100:.0f}%")
    print(f"  Mode:        {'📄 PAPER' if config.paper else '⚠ LIVE'}")

    # --- Bot initialisieren ---
    bot = TradingBotV1(config)

    # --- Aktuelle Signale anzeigen ---
    print("\n1. Aktuelle Signale:")
    signals = bot.signals.get_all_signals()
    for ticker, sig in signals.items():
        if "error" in sig:
            print(f"  {ticker:<8} ⚠ {sig['error']}")
            continue
        arrow = "🟢" if sig["signal"] == "long" else "🔴"
        cross = f"[{sig['cross'].upper()}]" \
                if sig["cross"] != "none" else ""
        print(
            f"  {ticker:<8} {arrow} {sig['signal'].upper():<5} "
            f"{cross:<12} "
            f"RSI:{sig['rsi']:>5.1f} | "
            f"Preis:${sig['current_price']:>8.2f} | "
            f"SMA{config.sma_fast}:${sig['sma_fast']:>8.2f} | "
            f"SMA{config.sma_slow}:${sig['sma_slow']:>8.2f}"
        )

    # --- Signal Charts ---
    print("\n2. Signal Charts...")
    plot_signals_detail(bot)

    # --- Account Status ---
    print("\n3. Account Status:")
    status = bot.get_status()
    print(f"  Portfolio:   ${status.get('portfolio', 0):>10,.2f}")
    print(f"  Cash:        ${status.get('cash', 0):>10,.2f}")
    print(f"  Positionen:  {status.get('n_positions', 0)}")

    # --- Bot Modus wählen ---
    print("\n4. Bot Modus:")
    print("  A) Bot starten (läuft kontinuierlich)")
    print("  B) Einmaliger Signal Check (ohne Trading)")
    print("  C) Backtesting der Strategie")

    mode = input("\n  Wahl [A/B/C]: ").strip().upper()

    if mode == "A":
        print("\n🚀 Bot startet...")
        print("  Drücke Ctrl+C zum Stoppen")
        print(f"  Log: bot_v1.log")

        bot.start()

        try:
            while bot._running:
                # Alle 5 Minuten Status anzeigen
                time.sleep(300)
                s = bot.get_status()
                print(
                    f"\n  Status: "
                    f"Loops={s['loop_count']} | "
                    f"Trades={s['n_trades']} | "
                    f"Portfolio=${s['portfolio']:,.2f} | "
                    f"Positionen={s['n_positions']}"
                )

        except KeyboardInterrupt:
            print("\n\nStoppe Bot...")
            bot.stop()

            # Alle Positionen schließen?
            close_all = input(
                "Alle Positionen schließen? [j/n]: "
            ).strip().lower()
            if close_all == "j":
                bot.trading_client.close_all_positions(
                    cancel_orders=True
                )
                print("✅ Alle Positionen geschlossen")

    elif mode == "B":
        print("\n📊 Einmaliger Signal Check...")
        for ticker, sig in signals.items():
            if "error" not in sig:
                can, reason = bot.risk.can_trade(
                    ticker, sig
                )
                print(
                    f"  {ticker}: "
                    f"{'✅ Kaufbar' if can else '❌ Kein Trade'}"
                    f" — {reason}"
                )

    elif mode == "C":
        print("\n📈 Backtesting (letzte 2 Jahre)...")
        import yfinance as yf

        all_results = []
        for ticker in config.tickers:
            df = yf.download(
                ticker, period="2y",
                auto_adjust=True, progress=False
            )
            df.columns = df.columns.get_level_values(0)
            close = df["Close"].squeeze()

            sma_f = close.rolling(config.sma_fast).mean()
            sma_s = close.rolling(config.sma_slow).mean()
            sig   = (sma_f > sma_s).astype(int).shift(1)

            ret_m = close.pct_change()
            ret_s = ret_m * sig.fillna(0)

            capital = 10_000
            eq_s    = (1 + ret_s).cumprod() * capital
            eq_m    = (1 + ret_m).cumprod() * capital

            years   = len(close) / 252
            cagr_s  = (eq_s.iloc[-1]/capital)**(1/years)-1
            cagr_m  = (eq_m.iloc[-1]/capital)**(1/years)-1
            sharpe  = (
                ret_s.mean() /
                ret_s.std() * np.sqrt(252)
            ) if ret_s.std() > 0 else 0

            all_results.append({
                "Ticker":    ticker,
                "CAGR %":   round(cagr_s * 100, 2),
                "B&H CAGR %": round(cagr_m * 100, 2),
                "Sharpe":   round(sharpe, 2),
            })

            print(
                f"  {ticker}: "
                f"CAGR={cagr_s*100:+.1f}% | "
                f"B&H={cagr_m*100:+.1f}% | "
                f"Sharpe={sharpe:.2f}"
            )

        print("\n  Summary:")
        res_df = pd.DataFrame(all_results)
        print(res_df.to_string(index=False))

    # --- Performance Dashboard ---
    print("\n5. Performance Dashboard...")
    plot_bot_performance(bot, bot.trading_client)

    # --- Trade Log ---
    trade_log = bot.get_trade_log()
    if not trade_log.empty:
        trade_log.to_csv("bot_v1_trades.csv", index=False)
        print(f"\nGespeichert: bot_v1_trades.csv")
    else:
        print("\nNoch keine Trades (Bot noch nicht gelaufen)")

    print("\n" + "="*55)
    print("NÄCHSTE SCHRITTE")
    print("="*55)
    print("  1. Bot in Modus A starten und 1 Stunde laufen lassen")
    print("  2. bot_v1.log beobachten")
    print("  3. Alpaca Dashboard: paper.alpaca.markets")
    print("  4. Nach 24h: Trade Log analysieren")
    print("  5. Morgen: Risk Management System erweitern")