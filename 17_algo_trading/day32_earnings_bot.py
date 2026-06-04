"""
Day 32 - News & Earnings Kalender Bot 

warum das wichtig ist:
- earnings reports sind der größte vola treiber 


was wri heute bauen:
1. Earnings kalender scrapen ( yfinance + finviz)
2. Economic calendar  ( wichtige Makro Events )
3. Automatische Alerts via Terminal + Email
4. Pre earnings analyse ( historische moves)
5. IV Crush analyse ( options volatilität vor/ nach earnings)
6. Earnings Surprise Tracker
7. Vollständiger Event driven Alert bot 

- instis haben ganze teams, die earnings tracken 
"""

import os 
import json
import time 
import smtplib # verbindung zum email server
import warnings 
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from datetime import datetime, timedelta, date
from dataclasses import dataclass, field
from typing import Optional, Dict, List
import pandas as pd 
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import yfinance as yf
import requests
from dotenv import load_dotenv

load_dotenv()
warnings.filterwarnings("ignore")

# Email config
EMAIL_USER     = os.getenv("EMAIL_USER",     "")
EMAIL_PASSWORD = os.getenv("EMAIL_PASSWORD", "")
EMAIL_TO       = os.getenv("EMAIL_TO",       "") 

@dataclass
class EarningsEvent:
    """Ein einzelner Earnings Report."""
    ticker:          str
    company_name:    str
    report_date:     date
    time:            str        # "BMO" (Before Market Open) / "AMC" (After Market Close)
    eps_estimate:    Optional[float] = None # None als Default um Fehler zu vermeiden 
    revenue_estimate: Optional[float] = None
    eps_actual:      Optional[float] = None
    revenue_actual:  Optional[float] = None
    surprise_pct:    Optional[float] = None
    days_until:      int = 0


class EarningsCalendar:
    """
    Lädt und verwaltet Earnings Events.

    Quellen:
        1. yfinance — kostenlos, verzögert
        2. Finviz — kostenlos, keine API nötig
        3. SEC EDGAR — offizielle Quelle

    Für den Anfang: yfinance reicht vollständig aus.
    """

    def __init__(self, watchlist: List[str]):
        self.watchlist = watchlist
        self._cache: Dict[str, EarningsEvent] = {}

    def get_next_earnings(self, ticker: str) -> Optional[EarningsEvent]:
        """
        Lädt nächsten Earnings Report für einen Ticker.

        yfinance gibt calendar.earnings_dates zurück —
        ein DataFrame mit historischen und zukünftigen Dates.
        """
        try:
            stock = yf.Ticker(ticker)
            cal   = stock.calendar # dictionary

            if cal is None or "Earnings Date" not in cal:
                return None

            earnings_date = cal["Earnings Date"]

            # Kann eine Liste sein, da manchmal genauer tag nicht festgelegt ist
            if isinstance(earnings_date, list):
                if len(earnings_date) == 0:
                    return None
                earnings_date = earnings_date[0]

            if pd.isna(earnings_date):
                return None

            # Normalisieren
            if hasattr(earnings_date, "date"): # prüft ob date vorhanden 
                report_date = earnings_date.date()
            else: # wenn nicht such nach anderen Formaten
                report_date = pd.to_datetime(
                    earnings_date
                ).date()

            days_until = (
                report_date - date.today()
            ).days # days für tag delta

            info = stock.info

            return EarningsEvent(
                ticker       = ticker,
                company_name = info.get("shortName", ticker),
                report_date  = report_date,
                time         = "AMC",   # Default
                eps_estimate = cal.get("EPS Estimate"),
                revenue_estimate = cal.get("Revenue Estimate"),
                days_until   = days_until,
            )

        except Exception as e:
            return None

    def get_all_upcoming(self,
                          days_ahead: int = 30) -> List[EarningsEvent]:
        """
        Lädt alle kommenden Earnings für die Watchlist.
        Sortiert nach Datum.
        """
        events = []

        for ticker in self.watchlist:
            event = self.get_next_earnings(ticker)
            if event and 0 <= event.days_until <= days_ahead: # 0 schließt vergangene dates aus, days_ahead begrenzt die zukunft
                events.append(event)
            time.sleep(0.3)   # Rate Limiting

        return sorted(events, key=lambda x: x.days_until) # sortiert nach days_until aufsteigend 

    def get_historical_earnings(self,
                                  ticker:    str,
                                  n_quarters: int = 16) -> pd.DataFrame:
        """
        Lädt historische Earnings mit EPS Surprise.

        EPS Surprise:
            Aktual - Estimate = positiv → Beat → Kurs steigt oft
            Aktual - Estimate = negativ → Miss → Kurs fällt oft

        Aber: "Buy the rumor, sell the news" passiert auch.
        Deshalb: historische Analyse vor jedem Trade.
        """
        try:
            stock    = yf.Ticker(ticker)
            earnings = stock.earnings_history

            if earnings is None or earnings.empty:
                return pd.DataFrame()

            df = earnings.copy()
            df = df.tail(n_quarters) # nur die letzten n_quarters quarters

            # Surprise berechnen
            if "epsActual" in df.columns and \
               "epsEstimate" in df.columns:
                df["surprise_abs"] = (df["epsActual"] - df["epsEstimate"])

                df["surprise_pct"] = (df["surprise_abs"] / abs(df["epsEstimate"]) * 100
                ).round(2) # prozentuale surprise, gerundet auf 2 dezimalstellen

            return df

        except Exception as e:
            return pd.DataFrame()

    def print_calendar(self,
                        events: List[EarningsEvent]) -> None:
        """Gibt Earnings Kalender im Terminal aus."""
        print(f"\n{'='*65}")
        print(f"  EARNINGS KALENDER — "
              f"nächste {len(events)} Reports")
        print(f"{'='*65}")
        print(f"  {'Ticker':<8} {'Unternehmen':<25} "
              f"{'Datum':<12} {'Tage':>5} {'Zeit':<5}")
        print("  " + "-"*60)

        for e in events:
            urgency = ("🔴" if e.days_until <= 2
                       else ("🟡" if e.days_until <= 7
                             else "⚪"))
            print(
                f"  {e.ticker:<8}"
                f"  {e.company_name[:23]:<23}"
                f"  {str(e.report_date):<12}"
                f"  {e.days_until:>4}d"
                f"  {e.time:<4}"
                f"  {urgency}"
            )

            if e.eps_estimate is not None:
                print(
                    f"  {'':8}  EPS Estimate: "
                    f"${e.eps_estimate:.2f}"
                )

        print(f"{'='*65}")


class PreEarningsAnalysis:
    """
    Analysiert historisches Kursverhalten rund um Earnings.

    Beantwortet:
        Wie stark bewegt sich die Aktie typisch nach Earnings?
        In welche Richtung geht sie meist?
        Wie ist die IV vor/nach Earnings?

    Das ist der Unterschied zwischen
    "kaufen oder nicht kaufen" und "wieviel Risiko?"
    """

    def __init__(self, ticker: str):
        self.ticker = ticker
        self.stock  = yf.Ticker(ticker)

    def get_historical_moves(self,
                              n_quarters: int = 16) -> pd.DataFrame:
        """
        Berechnet Kursreaktion nach jedem historischen Earnings.

        Move = Schluss nach Earnings / Schluss vorher - 1
        """
        try:
            hist_earnings = self.stock.earnings_history

            if hist_earnings is None or hist_earnings.empty:
                return pd.DataFrame()

            price_data = yf.download(
                self.ticker, period="5y",
                auto_adjust=True, progress=False
            )["Close"].squeeze()
            price_data.index = pd.to_datetime(
                price_data.index
            ).normalize()

            moves = []
            for idx in hist_earnings.index[-n_quarters:]:
                try:
                    earn_date = pd.to_datetime(idx).normalize() # normalize entfernt uhrzeit

                    # Schlusskurs vor und nach Earnings
                    mask_before = price_data.index <= earn_date
                    mask_after  = price_data.index > earn_date

                    before_prices = price_data[mask_before] # alle Preise vor oder am earn_date
                    after_prices  = price_data[mask_after] # alle Preise nach earn_date

                    if before_prices.empty or after_prices.empty:
                        continue

                    price_before = float(before_prices.iloc[-1]) # Schlusskurs
                    price_after  = float(after_prices.iloc[0]) # Eröffnungskurs am nächsten Tag

                    move_pct = (price_after / price_before - 1) * 100

                    # EPS Surprise
                    row = hist_earnings.loc[idx]
                    surprise = None
                    if ("epsActual" in row and "epsEstimate" in row):
                        if (pd.notna(row["epsActual"]) and pd.notna(row["epsEstimate"]) and
                                row["epsEstimate"] != 0): # keine division durch null
                            surprise = (
                                (row["epsActual"] -
                                 row["epsEstimate"]) /
                                abs(row["epsEstimate"]) * 100
                            )

                    moves.append({
                        "date":         str(earn_date.date()),
                        "move_pct":     round(move_pct, 2),
                        "direction":    "Up" if move_pct > 0
                                        else "Down",
                        "abs_move":     round(abs(move_pct), 2),
                        "surprise_pct": round(surprise, 2)
                                        if surprise else None,
                        "eps_actual":   row.get("epsActual"),
                        "eps_estimate": row.get("epsEstimate"),
                    })

                except Exception:
                    continue

            return pd.DataFrame(moves)

        except Exception as e:
            return pd.DataFrame()

    def get_move_statistics(self) -> dict:
        """
        Statistiken über historische Earnings Moves.

        Gibt zurück:
            avg_move:    Durchschnittlicher absoluter Move
            up_rate:     % der Earnings bei denen Kurs gestiegen
            avg_up:      Avg. Move wenn positiv
            avg_down:    Avg. Move wenn negativ
            expected:    Erwarteter Move (weighted)
        """
        moves_df = self.get_historical_moves()

        if moves_df.empty:
            return {
                "avg_move":  0,
                "up_rate":   50,
                "avg_up":    0,
                "avg_down":  0,
                "expected":  0,
                "n_quarters": 0,
            }

        up    = moves_df[moves_df["move_pct"] > 0]
        down  = moves_df[moves_df["move_pct"] <= 0]
        n     = len(moves_df)

        avg_move  = float(moves_df["abs_move"].mean())
        up_rate   = len(up) / n * 100 if n > 0 else 50
        avg_up    = float(up["move_pct"].mean()) \
                    if not up.empty else 0
        avg_down  = float(down["move_pct"].mean()) \
                    if not down.empty else 0

        # Erwarteter Move
        expected  = (up_rate/100 * avg_up + (1 - up_rate/100) * avg_down)
        # erwartungswert aus statistik
        # E(Move) = P(Up) * Avg(Up) + P(Down) * Avg(Down) wie würfel 

        return {
            "avg_move":    round(avg_move, 2),
            "up_rate":     round(up_rate, 1),
            "avg_up":      round(avg_up, 2),
            "avg_down":    round(avg_down, 2),
            "expected":    round(expected, 2),
            "n_quarters":  n,
            "max_up":      round(float(moves_df["move_pct"].max()), 2),
            "max_down":    round(float(moves_df["move_pct"].min()), 2),
        }

    def get_pre_earnings_drift(self,
                                days_before: int = 10) -> pd.DataFrame:
        """
        Pre-Earnings Drift — steigt die Aktie vor Earnings?

        Viele Aktien haben einen bekannten Aufwärtsdrift
        in den 10 Tagen vor Earnings.
        Das nennt sich Pre-Earnings Announcement Drift (PEAD).
        Es ist akademisch dokumentiert und real nutzbar.
        """
        try:
            hist_earnings = self.stock.earnings_history
            if hist_earnings is None or hist_earnings.empty:
                return pd.DataFrame()

            price_data = yf.download(
                self.ticker, period="5y",
                auto_adjust=True, progress=False
            )["Close"].squeeze()
            price_data.index = pd.to_datetime(
                price_data.index
            ).normalize()

            drifts = []
            for idx in hist_earnings.index[-6:]:
                try:
                    earn_date = pd.to_datetime(idx).normalize()
                    start     = earn_date - pd.Timedelta(
                        days=days_before * 2 # 2 * days_before (um sicher zu sein, dass genug daten)
                    )

                    window = price_data.loc[start:earn_date]
                    window = window.iloc[-days_before:]

                    if len(window) < 3:
                        continue

                    drift = (
                        window.iloc[-1] / window.iloc[0] - 1
                    ) * 100

                    drifts.append({
                        "earnings_date": str(earn_date.date()),
                        f"drift_{days_before}d_pct":
                            round(float(drift), 2),
                    })

                except Exception:
                    continue

            return pd.DataFrame(drifts)

        except Exception:
            return pd.DataFrame()


class EconomicCalendar:
    """
    Trackt wichtige makroökonomische Events.

    Wichtigste Events für Trader:
        FOMC Meeting:        Fed Zinsentscheidung → größter Markt-Mover
        CPI Release:         Inflation → bestimmt Fed Kurs
        Jobs Report (NFP):   Non-Farm Payrolls → Risk On/Off
        GDP Release:         Wirtschaftswachstum
        PPI:                 Producer Price Index
        Retail Sales:        Konsumenten-Stärke

    Quellen:
        investing.com — beste Quelle, braucht Scraping
        fred.stlouisfed.org — kostenlose API
        US Treasury / Fed — offizielle Quellen

    Für diesen Bot: hartcodierter Kalender + manuelles Update.
    In Produktion: investing.com API oder econdb.com
    """

    # Bekannte, wiederkehrende Events (approximiert)
    RECURRING_EVENTS = {
        "FOMC Meeting": {
            "frequency": "8x jährlich",
            "impact":    "HOCH",
            "description": "Fed Zinsentscheidung",
            "typical_move_spy": "±0.5% bis ±2.0%",
            "next_month": None,  # Muss aktualisiert werden
        },
        "CPI Release": {
            "frequency": "Monatlich (2. Woche)",
            "impact":    "HOCH",
            "description": "Consumer Price Index",
            "typical_move_spy": "±0.3% bis ±1.5%",
        },
        "NFP Jobs Report": {
            "frequency": "Erster Freitag im Monat",
            "impact":    "HOCH",
            "description": "Non-Farm Payrolls",
            "typical_move_spy": "±0.3% bis ±1.0%",
        },
        "PPI Release": {
            "frequency": "Monatlich",
            "impact":    "MITTEL",
            "description": "Producer Price Index",
            "typical_move_spy": "±0.2% bis ±0.8%",
        },
        "Retail Sales": {
            "frequency": "Monatlich",
            "impact":    "MITTEL",
            "description": "US Einzelhandelsumsätze",
            "typical_move_spy": "±0.2% bis ±0.6%",
        },
        "GDP Release": {
            "frequency": "Quartalsweise",
            "impact":    "MITTEL",
            "description": "BIP-Wachstum USA",
            "typical_move_spy": "±0.3% bis ±1.0%",
        },
    }

    def get_this_weeks_events(self) -> pd.DataFrame:
        """
        Gibt Events der aktuellen Woche zurück.
        In Produktion: echte API. Hier: Approximation.
        """
        today    = date.today()
        monday   = today - timedelta(days=today.weekday())
        friday   = monday + timedelta(days=4)

        rows = []
        for name, info in self.RECURRING_EVENTS.items():
            rows.append({
                "Event":       name,
                "Impact":      info["impact"],
                "Beschreibung": info["description"],
                "Häufigkeit":  info["frequency"],
                "SPY Move":    info["typical_move_spy"],
            })

        return pd.DataFrame(rows)

    def get_fred_data(self,
                       series_id: str,
                       limit:      int = 24) -> pd.DataFrame:
        """
        Lädt Wirtschaftsdaten von FRED (kostenlos, kein Key).

        Nützliche Series IDs:
            CPIAUCSL   → CPI
            UNRATE     → Unemployment Rate
            FEDFUNDS   → Fed Funds Rate
            GDP        → Bruttoinlandsprodukt
            DGS10      → 10-Year Treasury Yield
            DEXUSEU    → EUR/USD Exchange Rate
        """
        try:
            url    = (
                f"https://fred.stlouisfed.org/graph/fredgraph.csv"
                f"?id={series_id}"
            )
            df = pd.read_csv(
                url,
                index_col   = 0,
                parse_dates  = True
            )
            df.columns = [series_id]
            df         = df.replace(".", np.nan).dropna() # fehlende werte werdfen von fred mit . beziffert
            # die . werden dann entfernt 
            df[series_id] = pd.to_numeric(df[series_id], errors="coerce") # sicherstellen dass die werte numerisch sind
            return df.tail(limit) # nur die letzten limit werte zurückgeben

        except Exception as e:
            print(f"FRED Fehler ({series_id}): {e}")
            return pd.DataFrame()

    def get_macro_dashboard(self) -> dict:
        """
        Lädt die wichtigsten Makro-Indikatoren.
        """
        print("Lade Makro-Daten von FRED...")

        series = {
            "CPI":          "CPIAUCSL",
            "Fed Funds":    "FEDFUNDS",
            "Unemployment": "UNRATE",
            "10Y Yield":    "DGS10",
            "2Y Yield":     "DGS2",
        }

        macro = {}
        for name, sid in series.items():
            df = self.get_fred_data(sid, limit=120) # limit
            if not df.empty:
                latest = float(df.iloc[-1, 0]) # aktuellster wert
                prev   = float(df.iloc[-2, 0]) if len(df) > 1 else latest # vorheriger wert, falls vorhanden
                change = latest - prev
                macro[name] = {
                    "current": round(latest, 3),
                    "change":  round(change, 3),
                    "data":    df,
                }
            time.sleep(0.5)

        # Yield Curve (2Y vs. 10Y)
        if "10Y Yield" in macro and "2Y Yield" in macro:
            spread = (
                macro["10Y Yield"]["current"] -
                macro["2Y Yield"]["current"]
            )
            macro["Yield Curve"] = {
                "current": round(spread, 3),
                "inverted": spread < 0,
            }
            # Inversion der Zinsstrukturkurve (Yield Curve) ist ein bekannter Indikator für mögliche Rezessionen. 
            # Wenn die 2-jährige Rendite höher ist als die 10-jährige
            # Normal:   10Y (4.5%) > 2Y (4.0%) → Spread +0.5% → Wirtschaft gesund
            # Invertiert: 10Y (4.0%) < 2Y (4.5%) → Spread -0.5% → Rezessionssignal

        return macro

class AlertSystem:
    """
    Sendet Alerts via Terminal, Email und optional Telegram.

    Alert Typen:
        EARNINGS_TOMORROW    → Earnings morgen
        EARNINGS_TODAY       → Earnings heute
        BIG_MOVE             → Aktie bewegt sich stark
        MACRO_EVENT          → Wichtiges Makro-Event heute
        IV_SPIKE             → Options Volatilität springt an
    """

    def __init__(self,
                 email_user:     str = "",
                 email_password: str = "",
                 email_to:       str = ""):
        self.email_user     = email_user
        self.email_password = email_password
        self.email_to       = email_to
        self._alert_log: List[dict] = []

    def _log_alert(self, alert: dict) -> None:
        """Loggt Alert in Memory und CSV."""
        self._alert_log.append(alert)

        log_file = "alert_log.csv"
        log_df   = pd.DataFrame([alert])

        if os.path.exists(log_file):
            existing = pd.read_csv(log_file)
            log_df   = pd.concat(
                [existing, log_df], ignore_index=True
            )
        log_df.to_csv(log_file, index=False)

    def send_terminal_alert(self,
                             alert_type: str,
                             message:    str,
                             urgency:    str = "INFO") -> None:
        """Terminal Alert mit Farb-Kodierung."""
        icons = {
            "CRITICAL": "🚨",
            "WARNING":  "⚠️ ",
            "INFO":     "ℹ️ ",
            "SUCCESS":  "✅",
        }
        icon = icons.get(urgency, "📢")

        timestamp = datetime.now().strftime("%H:%M:%S")
        print(f"\n{icon} [{timestamp}] {alert_type}")
        print(f"   {message}")

        alert = {
            "timestamp":  datetime.now().isoformat(),
            "type":       alert_type,
            "message":    message,
            "urgency":    urgency,
        }
        self._log_alert(alert)

    def send_email_alert(self,
                          subject:  str,
                          body:     str) -> bool:
        """
        Sendet Email Alert.

        Setup für Gmail:
            1. Gmail Account → Security → App Passwords
            2. App Password generieren
            3. In .env:
               EMAIL_USER=deine@gmail.com
               EMAIL_PASSWORD=app_password_hier
               EMAIL_TO=empfaenger@email.com
        """
        if not all([self.email_user,
                    self.email_password,
                    self.email_to]):
            return False

        try:
            msg              = MIMEMultipart()
            msg["From"]      = self.email_user
            msg["To"]        = self.email_to
            msg["Subject"]   = f"[Trading Bot] {subject}"

            msg.attach(MIMEText(body, "plain"))

            with smtplib.SMTP_SSL(
                "smtp.gmail.com", 465
            ) as server:
                server.login(
                    self.email_user,
                    self.email_password
                )
                server.send_message(msg)

            print(f"✅ Email gesendet: {subject}")
            return True

        except Exception as e:
            print(f"Email Fehler: {e}")
            return False

    def send_earnings_alert(self,
                             event: EarningsEvent) -> None:
        """Formatierter Earnings Alert."""
        urgency = (
            "CRITICAL" if event.days_until == 0
            else ("WARNING" if event.days_until <= 2
                  else "INFO")
        )

        if event.days_until == 0:
            timing = "HEUTE"
        elif event.days_until == 1:
            timing = "MORGEN"
        else:
            timing = f"in {event.days_until} Tagen"

        message = (
            f"{event.company_name} ({event.ticker}) "
            f"berichtet {timing} "
            f"({event.time})"
        )

        if event.eps_estimate:
            message += f"\n   EPS Estimate: ${event.eps_estimate:.2f}"

        self.send_terminal_alert(
            f"EARNINGS: {event.ticker}",
            message, urgency
        )

        # Email bei kritischen Events
        if urgency in ["CRITICAL", "WARNING"]:
            self.send_email_alert(
                subject = f"Earnings Alert: {event.ticker} {timing}",
                body    = (
                    f"Earnings Alert\n\n"
                    f"Ticker: {event.ticker}\n"
                    f"Unternehmen: {event.company_name}\n"
                    f"Datum: {event.report_date}\n"
                    f"Zeit: {event.time}\n"
                    f"Tage bis Earnings: {event.days_until}\n"
                    f"EPS Estimate: {event.eps_estimate}\n\n"
                    f"Vorbereitung empfohlen."
                )
            )

    def check_price_alerts(self,
                             watchlist:    List[str],
                             threshold:    float = 0.03) -> None:
        """
        Prüft ob Aktien sich stark bewegen.
        Alert wenn Tages-Move > threshold.
        """
        try:
            prices = yf.download(
                watchlist, period="2d",
                auto_adjust=True, progress=False
            )["Close"]

            if hasattr(prices.columns, "get_level_values"):
                prices.columns = prices.columns.get_level_values(0)

            if len(prices) < 2:
                return

            for ticker in watchlist:
                if ticker not in prices.columns:
                    continue

                today_price = float(prices[ticker].iloc[-1])
                prev_price  = float(prices[ticker].iloc[-2])

                if pd.isna(today_price) or pd.isna(prev_price):
                    continue

                move = (today_price / prev_price - 1)

                if abs(move) >= threshold:
                    direction = "📈" if move > 0 else "📉"
                    urgency   = (
                        "CRITICAL" if abs(move) > 0.05
                        else "WARNING"
                    )
                    self.send_terminal_alert(
                        f"BIG MOVE: {ticker}",
                        f"{direction} {move*100:+.2f}% heute "
                        f"(${today_price:.2f})",
                        urgency
                    )

        except Exception as e:
            print(f"Price Alert Fehler: {e}")

    def get_alert_log(self) -> pd.DataFrame:
        """Alle gesendeten Alerts."""
        if not self._alert_log:
            return pd.DataFrame()
        return pd.DataFrame(self._alert_log)


class EarningsSurpriseTracker:
    """
    Trackt EPS Surprises und Kursreaktionen.

    Earnings Surprise Effekte:
        Positive Surprise (Beat):
            Kurs steigt oft — aber nicht immer.
            "Buy the rumor, sell the news" ist real.

        Negative Surprise (Miss):
            Kurs fällt fast immer.
            Ausnahmen: Guidance überrascht positiv.

        In-Line (Meet):
            Oft enttäuschend wenn Markt mehr erwartet hat.

    Zusammenhang:
        Starke Überraschung (>10%) → stärkere Reaktion.
        Schwache Überraschung (<2%) → oft neutral.
    """

    def __init__(self, tickers: List[str]):
        self.tickers = tickers

    def analyze_surprise_correlation(self,
                                       ticker: str) -> dict:
        """
        Analysiert: Je größer die Surprise, desto größer der Move?
        """
        analyzer = PreEarningsAnalysis(ticker)
        moves_df = analyzer.get_historical_moves()
        hist_earn = yf.Ticker(ticker).earnings_history

        if moves_df.empty or hist_earn is None:
            return {}

        # Surprise vs. Move Korrelation
        merged = moves_df.dropna(subset=["surprise_pct"])

        if len(merged) < 3:
            return {"correlation": None, "n": 0}

        corr = float(merged["surprise_pct"].corr(merged["move_pct"]))
        # Pearson korrelation zwischen surprise_pct und move_pct
        # +0.8 -> großer beat = große bewegung 
        # 0 = kein Zusammenhang 
        # -0.2 sell the news 

        beat  = merged[merged["surprise_pct"] > 0]
        miss  = merged[merged["surprise_pct"] < 0]

        return {
            "correlation":      round(corr, 3),
            "n_quarters":       len(merged),
            "beat_count":       len(beat),
            "miss_count":       len(miss),
            "avg_move_on_beat": round(
                float(beat["move_pct"].mean()), 2
            ) if not beat.empty else 0,
            "avg_move_on_miss": round(
                float(miss["move_pct"].mean()), 2
            ) if not miss.empty else 0,
            "ticker":           ticker,
        }

    def get_sector_surprise_rate(self) -> pd.DataFrame:
        """
        Beat Rate für alle Ticker im Universe.
        Wer überrascht konsistent positiv?
        """
        rows = []

        for ticker in self.tickers:
            try:
                stock    = yf.Ticker(ticker)
                earnings = stock.earnings_history

                if earnings is None or earnings.empty:
                    continue

                if ("epsActual" not in earnings.columns or
                        "epsEstimate" not in earnings.columns):
                    continue

                valid = earnings.dropna(
                    subset=["epsActual", "epsEstimate"]
                )

                if valid.empty:
                    continue

                beats = (valid["epsActual"] > valid["epsEstimate"]).sum()
                # [True, True, False, True, True, False, True, True] (Beat = True)
                n     = len(valid)

                avg_surprise = float(
                    ((valid["epsActual"] - valid["epsEstimate"]) / abs(valid["epsEstimate"]) * 100
                    ).mean()
                )

                info = stock.info
                rows.append({
                    "Ticker":       ticker,
                    "Name":         info.get("shortName", ticker)[:20],
                    "Beat Rate %":  round(beats/n*100, 1),
                    "Avg Surprise": round(avg_surprise, 2),
                    "Quartale":     n,
                })

            except Exception:
                pass

            time.sleep(0.3)

        df = pd.DataFrame(rows)
        if not df.empty:
            df = df.sort_values(
                "Beat Rate %", ascending=False
            ).reset_index(drop=True)

        return df
    """
    Ticker   Name              Beat Rate%   Avg Surprise   Quartale
    NVDA     NVIDIA Corp       87.5%        +18.3%         8
    MSFT     Microsoft Corp    75.0%        +4.2%          8
    AAPL     Apple Inc         62.5%        +3.1%          8
    JPM      JPMorgan Chase    50.0%        +1.8%          8    
    """


def plot_earnings_calendar(events:     List[EarningsEvent],
                            ticker_data: Dict[str, dict]) -> None:
    """
    Visueller Earnings Kalender der nächsten 30 Tage.
    """
    if not events:
        print("Keine Events.")
        return

    fig = make_subplots(
        rows=1, cols=1,
        specs=[[{"type": "scatter"}]]
    )

    colors_urgency = {
        "today":    "#ef4444",
        "tomorrow": "#f59e0b",
        "week":     "#3b82f6",
        "later":    "#94a3b8",
    }

    for event in events:
        if event.days_until == 0:
            color = colors_urgency["today"]
            size  = 18
        elif event.days_until <= 1:
            color = colors_urgency["tomorrow"]
            size  = 15
        elif event.days_until <= 7:
            color = colors_urgency["week"]
            size  = 12
        else:
            color = colors_urgency["later"]
            size  = 9

        # Stats aus Analyse
        stats = ticker_data.get(event.ticker, {})
        hover = (
            f"<b>{event.ticker}</b><br>"
            f"{event.company_name}<br>"
            f"Datum: {event.report_date}<br>"
            f"Tage: {event.days_until}<br>"
            f"Zeit: {event.time}<br>"
        )
        if stats:
            hover += (
                f"Avg Move: ±{stats.get('avg_move', 0):.1f}%<br>"
                f"Beat Rate: {stats.get('up_rate', 0):.0f}%"
            )

        fig.add_trace(go.Scatter(
            x=[str(event.report_date)],
            y=[event.ticker],
            mode="markers+text",
            text=[event.ticker],
            textposition="middle right",
            marker=dict(
                size=size,
                color=color,
                line=dict(width=2, color="white")
            ),
            hovertemplate=hover +
                          "<extra></extra>",
            name=event.ticker,
            showlegend=False
        ))

    # Heute Linie
    fig.add_vline(
        x=str(date.today()),
        line_dash="dash",
        line_color="#1e293b",
        line_width=2,
        annotation_text="Heute"
    )

    fig.update_layout(
        title="Earnings Kalender — Nächste 30 Tage",
        xaxis_title="Datum",
        yaxis_title="Ticker",
        template="plotly_white",
        height=max(400, len(events) * 40 + 100),
        margin=dict(l=0, r=60, t=60, b=0)
    )

    fig.show()


def plot_historical_moves(ticker: str,
                           moves_df: pd.DataFrame,
                           stats:    dict) -> None:
    """
    Historische Earnings Moves als Bar Chart.
    """
    if moves_df.empty:
        print(f"Keine historischen Moves für {ticker}")
        return

    colors = [
        "#16a34a" if v > 0 else "#ef4444"
        for v in moves_df["move_pct"]
    ]

    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=[
            f"{ticker} — Historische Earnings Moves (%)",
            "Move Statistiken"
        ],
        horizontal_spacing=0.12
    )

    # Moves Bar Chart
    fig.add_trace(go.Bar(
        x=moves_df["date"],
        y=moves_df["move_pct"],
        marker_color=colors,
        text=[f"{v:+.1f}%" for v in moves_df["move_pct"]],
        textposition="outside",
        name="Move %",
        showlegend=False
    ), row=1, col=1)

    fig.add_hline(
        y=0, line_color="#1e293b",
        line_width=1.5, row=1, col=1
    )

    # Durchschnitt
    avg = stats.get("avg_move", 0)
    fig.add_hline(
        y=avg,
        line_dash="dot",
        line_color="#16a34a",
        annotation_text=f"+{avg:.1f}% avg",
        row=1, col=1
    )
    fig.add_hline(
        y=-avg,
        line_dash="dot",
        line_color="#ef4444",
        annotation_text=f"-{avg:.1f}% avg",
        row=1, col=1
    )

    # Statistiken Bar
    stat_labels = [
        "Avg Move", "Beat Rate", "Avg Up", "Avg Down"
    ]
    stat_values = [
        stats.get("avg_move",  0),
        stats.get("up_rate",   50),
        stats.get("avg_up",    0),
        abs(stats.get("avg_down", 0)),
    ]
    stat_colors = [
        "#3b82f6", "#16a34a", "#16a34a", "#ef4444"
    ]

    fig.add_trace(go.Bar(
        x=stat_labels,
        y=stat_values,
        marker_color=stat_colors,
        text=[
            f"{v:.1f}%" for v in stat_values
        ],
        textposition="outside",
        showlegend=False
    ), row=1, col=2)

    fig.update_layout(
        height=420,
        template="plotly_white",
        title=f"{ticker} — Earnings Analyse",
        margin=dict(l=0, r=0, t=60, b=0)
    )

    fig.update_yaxes(title_text="Move (%)", row=1, col=1)
    fig.update_yaxes(title_text="Wert (%)", row=1, col=2)

    fig.show()


def plot_macro_dashboard(macro: dict) -> None:
    """
    Makro-Indikatoren Dashboard.
    """
    indicators = {
        k: v for k, v in macro.items()
        if k != "Yield Curve" and "data" in v
    }

    if not indicators:
        print("Keine Makro-Daten.")
        return

    n   = len(indicators)
    fig = make_subplots(
        rows=2, cols=3,
        subplot_titles=list(indicators.keys())[:6],
        vertical_spacing=0.14,
        horizontal_spacing=0.10
    )

    positions = [
        (1,1),(1,2),(1,3),(2,1),(2,2),(2,3)
    ]

    for (name, data), (r, c) in zip(
        indicators.items(), positions
    ):
        df    = data["data"]
        color = (
            "#16a34a" if data["change"] >= 0
            else "#ef4444"
        )

        fig.add_trace(go.Scatter(
            x=df.index,
            y=df.iloc[:, 0],
            name=name,
            line=dict(color=color, width=2),
            fill="tozeroy",
            fillcolor=color.replace(
                "#16a34a", "rgba(22,163,74,0.08)"
            ).replace(
                "#ef4444", "rgba(239,68,68,0.08)"
            ),
            showlegend=False
        ), row=r, col=c)

        # Aktueller Wert
        current = data["current"]
        change  = data["change"]
        sign    = "+" if change >= 0 else ""
        ax_idx = (r - 1) * 3 + c
        xref = f"x{ax_idx} domain" if ax_idx > 1 else "x domain"
        yref = f"y{ax_idx} domain" if ax_idx > 1 else "y domain"
        fig.add_annotation(
            x=0.98, y=0.95,
            xref=xref,
            yref=yref,
            text=f"{current:.2f} ({sign}{change:.2f})",
            showarrow=False,
            font=dict(size=10, color=color),
            bgcolor="rgba(255,255,255,0.8)"
        )

    # Yield Curve
    if "Yield Curve" in macro:
        yc      = macro["Yield Curve"]
        spread  = yc["current"]
        inverted = yc["inverted"]
        status  = "🔴 Invertiert" if inverted else "✅ Normal"
        print(f"\n  Yield Curve (10Y-2Y): "
              f"{spread:+.3f}  {status}")
        if inverted:
            print("  ⚠ Invertierte Kurve historisch Rezessions-Signal")

    fig.update_layout(
        height=580,
        template="plotly_white",
        title="Makroökonomisches Dashboard — FRED Daten",
        margin=dict(l=0, r=0, t=60, b=0)
    )

    fig.show()


def plot_surprise_tracker(surprise_df: pd.DataFrame) -> None:
    """Beat Rate und Avg Surprise für alle Ticker."""
    if surprise_df.empty:
        return

    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=[
            "EPS Beat Rate (%)",
            "Avg. EPS Surprise (%)"
        ],
        horizontal_spacing=0.12
    )

    beat_colors = [
        "#16a34a" if v >= 70
        else ("#f59e0b" if v >= 50 else "#ef4444")
        for v in surprise_df["Beat Rate %"]
    ]

    fig.add_trace(go.Bar(
        x=surprise_df["Ticker"],
        y=surprise_df["Beat Rate %"],
        marker_color=beat_colors,
        text=[f"{v:.0f}%" for v in surprise_df["Beat Rate %"]],
        textposition="outside",
        showlegend=False
    ), row=1, col=1)

    fig.add_hline(
        y=70, line_dash="dot",
        line_color="#16a34a", opacity=0.5,
        annotation_text="70% Benchmark",
        row=1, col=1
    )

    surp_colors = [
        "#16a34a" if v > 0 else "#ef4444"
        for v in surprise_df["Avg Surprise"]
    ]

    fig.add_trace(go.Bar(
        x=surprise_df["Ticker"],
        y=surprise_df["Avg Surprise"],
        marker_color=surp_colors,
        text=[f"{v:+.1f}%" for v in surprise_df["Avg Surprise"]],
        textposition="outside",
        showlegend=False
    ), row=1, col=2)

    fig.add_hline(
        y=0, line_color="#1e293b",
        line_width=1, row=1, col=2
    )

    fig.update_layout(
        height=420,
        template="plotly_white",
        title="Earnings Surprise Tracker",
        margin=dict(l=0, r=0, t=60, b=0)
    )

    fig.update_yaxes(title_text="Beat Rate (%)",   row=1, col=1,
                     range=[0, 110])
    fig.update_yaxes(title_text="Avg Surprise (%)", row=1, col=2)

    fig.show()

class EarningsAlertBot:
    """
    Vollständiger Earnings & Event Alert Bot.

    Läuft entweder einmalig oder als kontinuierlicher Loop.
    """

    def __init__(self,
                 watchlist:    List[str],
                 alert_days:   List[int] = None):
        """
        watchlist:  Tickers die überwacht werden
        alert_days: Tage vor Earnings für Alert
                    [7, 2, 1, 0] = Alert 7T, 2T, 1T und am Tag
        """
        self.watchlist  = watchlist
        self.alert_days = alert_days or [7, 2, 1, 0] # wenn days none -> rechts

        self.calendar  = EarningsCalendar(watchlist)
        self.eco_cal   = EconomicCalendar()
        self.alerts    = AlertSystem(
            EMAIL_USER, EMAIL_PASSWORD, EMAIL_TO
        )
        self.surprise  = EarningsSurpriseTracker(watchlist)

    def run_daily_check(self) -> dict:
        """
        Täglicher Check — läuft einmal und gibt Report zurück.
        """
        print(f"\n{'='*55}")
        print(f"  EARNINGS BOT — "
              f"{datetime.now().strftime('%d.%m.%Y %H:%M')}")
        print(f"{'='*55}")

        results = {
            "timestamp":   datetime.now().isoformat(),
            "events":      [],
            "alerts_sent": 0,
            "ticker_analysis": {},
        }

        # 1. Earnings Events laden
        print("\n1. Earnings Events laden...")
        events = self.calendar.get_all_upcoming(days_ahead=30)
        results["events"] = [
            {
                "ticker":    e.ticker,
                "date":      str(e.report_date),
                "days_until": e.days_until,
            }
            for e in events
        ]
        self.calendar.print_calendar(events)

        # 2. Analyse für jeden Ticker
        print("\n2. Pre-Earnings Analyse...")
        ticker_data = {}

        for event in events[:10]:   # nächsten 10 Events analysieren
            analyzer = PreEarningsAnalysis(event.ticker)
            stats    = analyzer.get_move_statistics()
            ticker_data[event.ticker] = stats

            if stats.get("n_quarters", 0) > 0:
                print(
                    f"  {event.ticker:<8} "
                    f"Avg Move: ±{stats['avg_move']:.1f}% | "
                    f"Beat Rate: {stats['up_rate']:.0f}% | "
                    f"Quartile: {stats['n_quarters']}"
                )

        results["ticker_analysis"] = ticker_data

        # 3. Alerts senden
        print("\n3. Alerts prüfen...")
        for event in events:
            if event.days_until in self.alert_days: # nur an diesen tagen alert senden
                self.alerts.send_earnings_alert(event)
                results["alerts_sent"] += 1

        # 4. Price Alerts
        print("\n4. Price Alerts prüfen...")
        self.alerts.check_price_alerts(
            self.watchlist, threshold=0.03
        )

        # 5. Makro-Events
        print("\n5. Makro Kalender...")
        eco_events = self.eco_cal.get_this_weeks_events()
        print(eco_events[[
            "Event", "Impact", "SPY Move"
        ]].to_string(index=False))

        return results, events, ticker_data

    def run_continuous(self,
                        check_interval_hours: int = 4) -> None:
        """
        Kontinuierlicher Bot-Loop.
        Prüft alle X Stunden.
        """
        print(f"\n🤖 Earnings Bot gestartet")
        print(f"   Check-Intervall: alle {check_interval_hours}h")
        print(f"   Alert bei: {self.alert_days} Tage vor Earnings")
        print(f"   Drücke Ctrl+C zum Stoppen\n")

        while True:
            try:
                self.run_daily_check()

                next_check = datetime.now() + timedelta(
                    hours=check_interval_hours
                )
                print(f"\n   Nächster Check: "
                      f"{next_check.strftime('%d.%m. %H:%M')}")
                time.sleep(check_interval_hours * 3600)

            except KeyboardInterrupt:
                print("\n\nBot gestoppt.")
                break # gewolltes stoppen -> kein fehler
            except Exception as e:
                print(f"Fehler: {e}")
                time.sleep(300)   # ungewollter fehler 5min pause

if __name__ == "__main__":

    print("Tag 32 — Earnings & Event Alert Bot")
    print("=" * 55)

    # --- Watchlist ---
    WATCHLIST = [
        "AAPL", "MSFT", "NVDA", "GOOGL",
        "META", "AMZN", "TSLA", "JPM",
        "AMD",  "NFLX"
    ]

    print(f"Watchlist: {WATCHLIST}")

    # --- Bot initialisieren ---
    bot = EarningsAlertBot(
        watchlist  = WATCHLIST,
        alert_days = [7, 2, 1, 0]
    )

    # --- Täglicher Check ---
    results, events, ticker_data = bot.run_daily_check()

    # --- Visualisierungen ---

    # Earnings Kalender
    print("\n📊 Earnings Kalender Chart...")
    if events:
        plot_earnings_calendar(events, ticker_data)

    # Historische Moves — bestes Beispiel
    print("\n📊 Historische Earnings Moves...")
    focus_ticker = "MSFT"
    analyzer     = PreEarningsAnalysis(focus_ticker)
    moves_df     = analyzer.get_historical_moves()
    stats        = analyzer.get_move_statistics()

    if stats.get("n_quarters", 0) > 0:
        print(f"\n  {focus_ticker} Earnings Statistiken:")
        print(f"  Avg. Absoluter Move:  ±{stats['avg_move']:.1f}%")
        print(f"  Beat Rate:            {stats['up_rate']:.1f}%")
        print(f"  Avg. Move bei Beat:   {stats['avg_up']:+.1f}%")
        print(f"  Avg. Move bei Miss:   {stats['avg_down']:+.1f}%")
        print(f"  Bestes Quartal:       {stats['max_up']:+.1f}%")
        print(f"  Schlechtestes:        {stats['max_down']:+.1f}%")

    if not moves_df.empty:
        plot_historical_moves(focus_ticker, moves_df, stats)

    # Pre-Earnings Drift
    print(f"\n📊 Pre-Earnings Drift ({focus_ticker})...")
    drift_df = analyzer.get_pre_earnings_drift(days_before=10)
    if not drift_df.empty:
        avg_drift = drift_df.iloc[:, 1].mean()
        print(f"  Avg. 10-Tage Drift vor Earnings: "
              f"{avg_drift:+.2f}%")
        print(drift_df.to_string(index=False))

    # Surprise Tracker
    print(f"\n📊 Earnings Surprise Tracker...")
    surprise_df = bot.surprise.get_sector_surprise_rate()
    if not surprise_df.empty:
        print(surprise_df[[
            "Ticker", "Beat Rate %",
            "Avg Surprise", "Quartale"
        ]].to_string(index=False))
        plot_surprise_tracker(surprise_df)

    # Surprise-Move Korrelation
    print(f"\n📊 Surprise-Move Korrelation ({focus_ticker})...")
    corr_result = bot.surprise.analyze_surprise_correlation(
        focus_ticker
    )
    if corr_result.get("correlation"):
        print(
            f"  Korrelation Surprise/Move: "
            f"{corr_result['correlation']:.3f}"
        )
        print(
            f"  Avg. Move bei Beat:  "
            f"{corr_result['avg_move_on_beat']:+.2f}%"
        )
        print(
            f"  Avg. Move bei Miss:  "
            f"{corr_result['avg_move_on_miss']:+.2f}%"
        )

    # Makro Dashboard
    print(f"\n📊 Makro Dashboard...")
    eco_cal  = EconomicCalendar()
    macro    = eco_cal.get_macro_dashboard()

    if macro:
        print("\n  Aktuelle Makro-Indikatoren:")
        for name, data in macro.items():
            if "current" in data:
                change = data.get("change", 0)
                sign   = "+" if change >= 0 else ""
                print(
                    f"  {name:<15} "
                    f"{data['current']:.3f}  "
                    f"({sign}{change:.3f})"
                )
        plot_macro_dashboard(macro)

    # Alert Log
    print(f"\n📊 Alert Log...")
    alert_log = bot.alerts.get_alert_log()
    if not alert_log.empty:
        print(alert_log[[
            "timestamp", "type", "urgency"
        ]].to_string(index=False))
        alert_log.to_csv("day32_alerts.csv", index=False)
        print("✅ Gespeichert: day32_alerts.csv")

    # Export
    with open("day32_earnings_results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    print("✅ Gespeichert: day32_earnings_results.json")

    # Modus wählen
    print("\n" + "="*55)
    print("MODUS:")
    print("  A) Kontinuierlicher Bot (alle 4 Stunden)")
    print("  B) Einmalig (bereits ausgeführt)")
    mode = input("\n  Wahl [A/B]: ").strip().upper()

    if mode == "A":
        bot.run_continuous(check_interval_hours=4)