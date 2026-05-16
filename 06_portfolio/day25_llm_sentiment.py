"""
Day 25 — AI Sentiment Analyse mit LLMs

Warum LLMs besser sind als VADER/TextBlob:
    VADER: regelbasiert, kennt "not bad" nicht wirklich.
    TextBlob: einfaches NLP, versteht keinen Kontext.
    LLM:   versteht Nuancen, Ironie, Fachbegriffe.

    "Revenue missed estimates but management guided higher"
    VADER: neutral (zu komplex)
    GPT-4: bearish kurzfristig, bullish langfristig — erklärt warum

Was du heute baust:
    1. News Scraper — echte Finanznachrichten laden
    2. LLM Sentiment Analyse — strukturierter Output
    3. Earnings Call Analyse — CEO-Sprache verstehen
    4. Markt-Briefing Generator — tägliche AI-Zusammenfassung
    5. Multi-Ticker Sentiment Dashboard
    6. Sentiment → Trading Signal Backtesting

API Optionen (in Reihenfolge der Empfehlung):
    1. OpenAI GPT-4o-mini (günstig, gut)
    2. Anthropic Claude Haiku (sehr gut, günstig)
    3. Ollama lokal (kostenlos, braucht GPU)
    
Kosten Schätzung:
    GPT-4o-mini: ~$0.001 pro News-Artikel
    100 Artikel/Tag = ~$0.10/Tag = $3/Monat
    Sehr günstig für den Mehrwert.
"""

import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import requests
import json
import time
import os
from dataclasses import dataclass, field
from typing import Optional
import warnings
warnings.filterwarnings("ignore")

# Optional: OpenAI
try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    print("OpenAI nicht installiert: pip install openai")

    # Optional: Anthropic
try:
    import anthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False

# Immer verfügbar: VADER als Fallback
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from textblob import TextBlob

@dataclass
class LLMConfig:
    """
    Konfiguration für den LLM-Client.
    Unterstützt OpenAI, Anthropic und lokale Modelle.
    """
    provider:    str   = "openai"          # openai | anthropic | local
    model:       str   = "gpt-4o-mini"     # günstigstes GPT-4 Modell
    api_key:     str   = ""
    max_tokens:  int   = 500
    temperature: float = 0.1    # kreativität des modells            # niedrig = konsistent
    base_url:    str   = ""                # für lokale Modelle


class LLMClient:
    """
    Unified Client für verschiedene LLM Provider.
    Gleiche Interface — egal welcher Provider.
    """
    def __init__(self, config: LLMConfig): # ruft sofort init auf 
        self.config = config
        self._setup_client()

    def _setup_client(self): 
        """Initialisiert den richtigen Client."""
        if self.config.provider == "openai":
            if not OPENAI_AVAILABLE:
                raise ImportError("pip install openai")
            api_key = (self.config.api_key or
                       os.getenv("OPENAI_API_KEY", "")) # im terminal einmal eingeben
            if not api_key:
                raise ValueError(
                    "OPENAI_API_KEY nicht gesetzt. "
                    "Exportiere: export OPENAI_API_KEY=sk-..."
                )
            self.client = OpenAI(api_key=api_key)

        elif self.config.provider == "anthropic":
            if not ANTHROPIC_AVAILABLE:
                raise ImportError("pip install anthropic")
            api_key = (self.config.api_key or
                       os.getenv("ANTHROPIC_API_KEY", ""))
            self.client = anthropic.Anthropic(api_key=api_key)

        elif self.config.provider == "local":
            # Ollama oder anderer lokaler Server
            self.client = None
            self.base_url = (self.config.base_url or
                             "http://localhost:11434/api/generate")

    def complete(self, system: str, user: str) -> str:
        """
        Sendet eine Anfrage an den LLM.
        Gibt den Text zurück.
        """
        try:
            if self.config.provider == "openai":
                response = self.client.chat.completions.create(
                    model      = self.config.model,
                    messages   = [
                        {"role": "system", "content": system},
                        {"role": "user",   "content": user},
                    ],
                    max_tokens  = self.config.max_tokens,
                    temperature = self.config.temperature,
                )
                return response.choices[0].message.content

            elif self.config.provider == "anthropic":
                response = self.client.messages.create(
                    model      = self.config.model,
                    max_tokens = self.config.max_tokens,
                    system     = system,
                    messages   = [
                        {"role": "user", "content": user}
                    ],
                )
                return response.content[0].text

            elif self.config.provider == "local":
                response = requests.post(
                    self.base_url,
                    json={
                        "model":  self.config.model,
                        "prompt": f"{system}\n\n{user}",
                        "stream": False
                    },
                    timeout=30
                )
                return response.json().get("response", "")

        except Exception as e:
            print(f"LLM Fehler: {e}")
            return ""

    def complete_json(self, system: str, user: str) -> dict:
        """
        Wie complete() aber parsed JSON automatisch.
        Gibt leeres Dict bei Fehler zurück.
        """
        text = self.complete(system, user)
        try:
            # JSON aus dem Text extrahieren
            if "```json" in text:
                text = text.split("```json")[1].split("```")[0]
            elif "```" in text:
                text = text.split("```")[1].split("```")[0]
            return json.loads(text.strip())
        except json.JSONDecodeError:
            return {}

"""
llms antworten oft so:
Hier ist die Analyse:

```json
{
    "signal": "kaufen",
    "confidence": 0.75,
    "reasoning": "RSI überverkauft"
}
```

Dieser Code extrahiert nur den JSON-Teil heraus:

```python
text.split("```json")[1]  → '\n{"signal": "kaufen", ...}\n'
.split("```")[0]          → '\n{"signal": "kaufen", ...}\n'
.strip()                  → '{"signal": "kaufen", ...}'
json.loads(...)           → {"signal": "kaufen", "confidence": 0.75}
"""

def load_news_yfinance(ticker:      str,
                       max_articles: int = 20) -> list:
    """
    Lädt News via yfinance — kein API Key nötig.
    Gibt Liste von Dicts zurück.
    """
    try:
        stock = yf.Ticker(ticker)
        news  = stock.news or []

        articles = []
        for item in news[:max_articles]:
            ts = item.get("providerPublishTime", 0)
            dt = datetime.fromtimestamp(ts) if ts else datetime.now()

            articles.append({
                "title":     item.get("title", ""),
                "publisher": item.get("publisher", ""),
                "link":      item.get("link", ""),
                "date":      dt,
                "source":    "yfinance",
            })

        print(f"  {ticker}: {len(articles)} Artikel geladen")
        return articles

    except Exception as e:
        print(f"  {ticker}: Fehler — {e}")
        return []


def load_news_newsapi(ticker:   str,
                      api_key:  str,
                      days:     int = 7) -> list:
    """
    Lädt News via NewsAPI.org — 100 Requests/Tag kostenlos.
    API Key: newsapi.org/register

    Bessere Qualität als yfinance News.
    """
    if not api_key:
        return []

    # Ticker zu Firmenname
    try:
        info    = yf.Ticker(ticker).info
        company = info.get("shortName", ticker)
    except Exception:
        company = ticker

    url     = "https://newsapi.org/v2/everything"
    from_dt = (datetime.now() -
               timedelta(days=days)).strftime("%Y-%m-%d")

    params = {
        "q":        f"{company} OR {ticker} stock",
        "from":     from_dt,
        "sortBy":   "relevancy",
        "language": "en",
        "pageSize": 20,
        "apiKey":   api_key,
    }

    try:
        response = requests.get(url, params=params, timeout=10)
        data     = response.json()
        articles = []

        for item in data.get("articles", []):
            pub_date = item.get("publishedAt", "")
            try:
                dt = datetime.strptime(
                    pub_date, "%Y-%m-%dT%H:%M:%SZ"
                )
            except Exception:
                dt = datetime.now()

            articles.append({
                "title":       item.get("title", ""),
                "description": item.get("description", ""),
                "publisher":   item.get("source", {})
                                   .get("name", ""),
                "link":        item.get("url", ""),
                "date":        dt,
                "source":      "newsapi",
            })

        print(f"  {ticker}: {len(articles)} NewsAPI Artikel")
        return articles

    except Exception as e:
        print(f"  NewsAPI Fehler: {e}")
        return []
    

SENTIMENT_SYSTEM_PROMPT = """
Du bist ein erfahrener Finanzanalyst der News-Artikel analysiert.

Analysiere den gegebenen Finanznews-Artikel und gib einen
strukturierten JSON-Output zurück. Sei präzise und objektiv.

Antworte NUR mit validem JSON — kein Text davor oder danach.
Format:
{
  "sentiment": "bullish" | "bearish" | "neutral",
  "score": -1.0 bis +1.0,
  "confidence": 0.0 bis 1.0,
  "time_horizon": "short" | "medium" | "long",
  "key_factors": ["Faktor 1", "Faktor 2"],
  "risk_level": "low" | "medium" | "high",
  "summary": "Ein-Satz Zusammenfassung",
  "catalysts": ["Positiver Katalysator"],
  "risks": ["Risiko Faktor"]
}
"""


def analyze_article_llm(article: dict,
                          client:  LLMClient) -> dict:
    """
    Analysiert einen einzelnen Artikel mit dem LLM.

    Gibt strukturierten Sentiment-Output zurück.
    """
    title = article.get("title", "")
    desc  = article.get("description", title)
    pub   = article.get("publisher", "")
    date  = article.get("date", datetime.now())

    user_prompt = f"""
Analysiere diesen Finanz-Artikel:

Titel: {title}
Quelle: {pub}
Datum: {date.strftime('%Y-%m-%d') if hasattr(date, 'strftime') else str(date)}
Inhalt: {desc[:500]}

Gib einen strukturierten JSON-Output zurück.
"""

    result = client.complete_json(SENTIMENT_SYSTEM_PROMPT,
                                   user_prompt)

    # Fallback wenn LLM fehlschlägt
    if not result:
        result = vader_fallback(title)

    result["title"]     = title
    result["publisher"] = pub
    result["date"]      = date
    result["source"]    = "llm"

    return result


def vader_fallback(text: str) -> dict:
    """
    VADER als Fallback wenn kein LLM verfügbar.
    Gibt denselben Output-Format zurück.
    """
    analyzer = SentimentIntensityAnalyzer()
    scores   = analyzer.polarity_scores(text)
    compound = scores["compound"]

    if compound > 0.05:
        sentiment = "bullish"
    elif compound < -0.05:
        sentiment = "bearish"
    else:
        sentiment = "neutral"

    tb = TextBlob(text)

    return {
        "sentiment":    sentiment,
        "score":        round(compound, 3),
        "confidence":   round(abs(compound), 3),
        "time_horizon": "short",
        "key_factors":  [],
        "risk_level":   "medium",
        "summary":      text[:100],
        "catalysts":    [],
        "risks":        [],
        "source":       "vader_fallback",
        "subjectivity": round(tb.sentiment.subjectivity, 3),
    }


def analyze_all_articles(articles:   list,
                           client:     Optional[LLMClient],
                           batch_size: int   = 5,
                           delay:      float = 1.0) -> pd.DataFrame:
    """
    Analysiert alle Artikel — mit oder ohne LLM.

    batch_size: Pause nach jeder Batch (Rate Limiting).
    delay: Sekunden zwischen Batches.
    """
    results = []

    print(f"\nAnalysiere {len(articles)} Artikel...")

    for i, article in enumerate(articles):
        if client is not None:
            result = analyze_article_llm(article, client)
            if (i + 1) % batch_size == 0:
                time.sleep(delay)
        else:
            # VADER Fallback
            text   = (article.get("title", "") + " " +
                      article.get("description", ""))
            result = vader_fallback(text)
            result["title"]     = article.get("title", "")
            result["publisher"] = article.get("publisher", "")
            result["date"]      = article.get("date", datetime.now())

        results.append(result)

        if (i + 1) % 10 == 0:
            print(f"  {i+1}/{len(articles)} analysiert")

    return pd.DataFrame(results)

EARNINGS_SYSTEM_PROMPT = """
Du bist ein erfahrener Buy-Side Analyst der Earnings Calls analysiert.

Analysiere das gegebene Earnings-Transkript oder die Zusammenfassung
und identifiziere Management-Tonalität, Guidance-Qualität und
versteckte Signale.

Antworte NUR mit validem JSON:
{
  "overall_sentiment": "bullish" | "bearish" | "neutral",
  "management_confidence": 0.0 bis 1.0,
  "guidance_quality": "raised" | "maintained" | "lowered" | "withdrawn",
  "key_metrics_mentioned": ["Revenue", "Margins", ...],
  "positive_signals": ["Signal 1", ...],
  "negative_signals": ["Signal 1", ...],
  "language_analysis": {
    "hedging_words": ["möglicherweise", "könnte", ...],
    "confident_phrases": ["wir sind überzeugt", ...],
    "uncertainty_level": "low" | "medium" | "high"
  },
  "actionable_insight": "Was bedeutet das für den Kurs?",
  "score": -1.0 bis +1.0
}
"""


def analyze_earnings_call(transcript:  str,
                           ticker:      str,
                           client:      LLMClient,
                           quarter:     str = "Q1 2025") -> dict:
    """
    Analysiert Earnings Call Transkript mit LLM.

    In der Realität:
        Seeking Alpha Premium: vollständige Transkripte
        SEC EDGAR: 8-K Filings mit Earnings Releases
        earnings-whispers.com: Consensus und Surprises

    Hier: synthetisches Beispiel + echte API Calls.
    """
    user_prompt = f"""
Analysiere diesen Earnings Call für {ticker} ({quarter}):

{transcript[:2000]}

Identifiziere Management-Tonalität, Guidance und
versteckte Signale.
"""

    result = client.complete_json(
        EARNINGS_SYSTEM_PROMPT, user_prompt
    )

    if not result:
        return {
            "ticker":              ticker,
            "quarter":             quarter,
            "overall_sentiment":   "neutral",
            "management_confidence": 0.5,
            "score":               0.0,
            "error":               "LLM nicht verfügbar",
        }

    result["ticker"]  = ticker
    result["quarter"] = quarter
    return result


def get_synthetic_earnings(ticker: str) -> str:
    """
    Erstellt synthetisches Earnings-Transkript für Demo.
    In der Realität: echte Transkripte von Seeking Alpha laden.
    """
    templates = {
        "AAPL": """
CEO Tim Cook: "We're incredibly excited about our results this quarter.
iPhone revenue exceeded our expectations driven by strong demand for
iPhone 15 Pro models. Services revenue hit an all-time record of $23.9
billion, growing 16% year-over-year. We're seeing tremendous momentum
in our installed base which now exceeds 2.2 billion active devices.

Regarding guidance, we expect revenue to grow low to mid-single digits
year-over-year in the December quarter. We continue to invest heavily
in AI and we believe we're just scratching the surface of what's possible.

CFO Luca Maestri: Our gross margin was 46.2%, up 130 basis points
year-over-year. Operating cash flow was $26.8 billion. We returned
$29 billion to shareholders through dividends and buybacks.
We remain confident in our ability to generate significant value
for our shareholders going forward.
""",
        "DEFAULT": f"""
CEO: "We delivered solid results this quarter despite a challenging
macroeconomic environment. Revenue came in at the high end of our
guidance range. We're seeing strong customer demand and our pipeline
remains robust. However, we're mindful of macroeconomic headwinds
and are taking a prudent approach to our outlook.

CFO: Gross margins expanded sequentially driven by operational
efficiencies. We're maintaining our full-year guidance and remain
committed to returning capital to shareholders. We believe our
business model is well-positioned for sustainable long-term growth.
"""
    }
    return templates.get(ticker, templates["DEFAULT"])

BRIEFING_SYSTEM_PROMPT = """
Du bist ein Senior Analyst bei einem Top-Tier Hedge Fund.
Erstelle ein prägnantes, professionelles Markt-Briefing
auf Basis der gegebenen Daten.

Stil: Reuters/Bloomberg Stil. Faktenbasiert. Keine Meinungen
ohne Daten-Backing. Maximal 300 Wörter.

Antworte auf Deutsch. Kein JSON — normaler Fließtext.
"""


def generate_market_briefing(ticker:       str,
                              sentiment_df: pd.DataFrame,
                              price_data:   pd.DataFrame,
                              client:       LLMClient) -> str:
    """
    Generiert tägliches Markt-Briefing für einen Ticker.

    Kombiniert:
        - News Sentiment Zusammenfassung
        - Kurs-Performance
        - Technische Situation
    """
    close = price_data["Close"].squeeze()

    # Kurs-Statistiken
    current_price = float(close.iloc[-1])
    ret_1d  = (close.iloc[-1] / close.iloc[-2]  - 1) * 100
    ret_1w  = (close.iloc[-1] / close.iloc[-5]  - 1) * 100
    ret_1m  = (close.iloc[-1] / close.iloc[-21] - 1) * 100
    vol_30d = close.pct_change().rolling(30).std().iloc[-1] * \
              np.sqrt(252) * 100

    # Sentiment Summary
    if not sentiment_df.empty and "score" in sentiment_df.columns:
        avg_score   = sentiment_df["score"].mean()
        n_bullish   = (sentiment_df["sentiment"] == "bullish").sum()
        n_bearish   = (sentiment_df["sentiment"] == "bearish").sum()
        n_neutral   = (sentiment_df["sentiment"] == "neutral").sum()
        top_stories = sentiment_df.nlargest(3, "score")[
            "title"
        ].tolist()
    else:
        avg_score   = 0
        n_bullish   = n_bearish = n_neutral = 0
        top_stories = []

    user_prompt = f"""
Erstelle ein Markt-Briefing für {ticker} basierend auf:

KURS-DATEN:
- Aktueller Kurs: ${current_price:.2f}
- Performance 1T: {ret_1d:+.2f}%
- Performance 1W: {ret_1w:+.2f}%
- Performance 1M: {ret_1m:+.2f}%
- Annualisierte Volatilität: {vol_30d:.1f}%

NEWS SENTIMENT (letzte 24h):
- Durchschnittlicher Sentiment Score: {avg_score:+.3f}
- Bullish Artikel: {n_bullish}
- Bearish Artikel: {n_bearish}
- Neutral Artikel: {n_neutral}

TOP STORIES:
{chr(10).join(f"- {s}" for s in top_stories[:3])}

Erstelle ein professionelles, prägnantes Markt-Briefing.
"""

    briefing = client.complete(BRIEFING_SYSTEM_PROMPT, user_prompt)
    return briefing or "Briefing konnte nicht generiert werden."


def generate_trade_thesis(ticker:       str,
                           sentiment_df: pd.DataFrame,
                           price_data:   pd.DataFrame,
                           client:       LLMClient) -> dict:
    """
    Generiert eine vollständige Trade-These.

    Output: Strukturierter Investment Case
    mit Bull/Bear/Base Szenarien.
    """
    THESIS_SYSTEM = """
Du bist ein erfahrener Portfolio Manager.
Erstelle eine prägnante Trade-These.
Antworte NUR mit validem JSON.
Format:
{
  "recommendation": "BUY" | "HOLD" | "SELL",
  "conviction": "low" | "medium" | "high",
  "price_target_bull": 0.0,
  "price_target_base": 0.0,
  "price_target_bear": 0.0,
  "time_horizon": "1-3 Monate",
  "bull_case": "Bull Case in einem Satz",
  "base_case": "Base Case in einem Satz",
  "bear_case": "Bear Case in einem Satz",
  "key_risks": ["Risiko 1", "Risiko 2"],
  "catalysts": ["Katalysator 1", "Katalysator 2"],
  "stop_loss": "Bei welchem Preis aufgeben?"
}
"""

    close         = price_data["Close"].squeeze()
    current_price = float(close.iloc[-1])

    if not sentiment_df.empty and "score" in sentiment_df.columns:
        avg_score = sentiment_df["score"].mean()
        sentiment_summary = f"Avg Score: {avg_score:+.3f}"
    else:
        sentiment_summary = "Keine Sentiment-Daten"

    user_prompt = f"""
Erstelle eine Trade-These für {ticker}:
- Aktueller Kurs: ${current_price:.2f}
- Sentiment: {sentiment_summary}
- 1M Performance: {(close.iloc[-1]/close.iloc[-21]-1)*100:+.1f}%
- 52W Hoch: ${float(close.rolling(252).max().iloc[-1]):.2f}
- 52W Tief: ${float(close.rolling(252).min().iloc[-1]):.2f}
"""

    result = client.complete_json(THESIS_SYSTEM, user_prompt)
    if result:
        result["ticker"]        = ticker
        result["current_price"] = current_price
        result["timestamp"]     = datetime.now().isoformat()
    return result

def sentiment_to_signal(sentiment_df: pd.DataFrame,
                         price_data:   pd.DataFrame,
                         score_threshold: float = 0.2,
                         window:       int   = 3) -> pd.DataFrame:
    """
    Konvertiert LLM Sentiment zu Trading Signal.

    Logic:
        Aggregiere Sentiment über Rolling Window.
        Score > threshold  → Bullish Signal (Long)
        Score < -threshold → Bearish Signal (Cash/Short)
        Dazwischen         → Neutral (halten)

    Wichtig: Sentiment ist ein langsames Signal.
        Headlines heute → Kurs morgen.
        Rolling Window glättet tägliches Rauschen.
    """
    if sentiment_df.empty or "score" not in sentiment_df.columns:
        return pd.DataFrame()

    # Sentiment auf Tages-Basis aggregieren
    sentiment_df["date"] = pd.to_datetime(
        sentiment_df["date"]
    ).dt.normalize()

    daily_sentiment = sentiment_df.groupby("date").agg(
        avg_score     = ("score", "mean"),
        article_count = ("score", "count"),
        n_bullish     = ("sentiment",
                         lambda x: (x == "bullish").sum()),
        n_bearish     = ("sentiment",
                         lambda x: (x == "bearish").sum()),
    ).reset_index()

    daily_sentiment = daily_sentiment.set_index("date")

    # Rolling Window Sentiment
    daily_sentiment["roll_score"] = (
        daily_sentiment["avg_score"]
        .rolling(window, min_periods=1).mean()
    )

    # Signal
    daily_sentiment["signal"] = 0
    daily_sentiment.loc[
        daily_sentiment["roll_score"] > score_threshold, "signal"
    ] = 1
    daily_sentiment.loc[
        daily_sentiment["roll_score"] < -score_threshold, "signal"
    ] = -1

    # Mit Preisen mergen
    close = price_data["Close"].squeeze()
    close_df = pd.DataFrame({"close": close})
    close_df.index = pd.to_datetime(close_df.index).normalize()

    merged = daily_sentiment.join(
        close_df, how="inner"
    )

    return merged


def backtest_sentiment_signal(signal_df: pd.DataFrame,
                               capital:   float = 10_000) -> dict:
    """
    Backtested Sentiment Signal vs. Buy & Hold.
    """
    if signal_df.empty or "close" not in signal_df.columns:
        return {}

    returns = signal_df["close"].pct_change().fillna(0)
    signal  = signal_df["signal"].shift(1).fillna(0)

    strat_returns  = returns * signal.clip(lower=0)  # nur Long
    market_returns = returns

    equity_strat  = (1 + strat_returns).cumprod() * capital
    equity_market = (1 + market_returns).cumprod() * capital

    years = len(equity_strat) / 252
    if years > 0:
        cagr_strat  = (equity_strat.iloc[-1] /
                       capital) ** (1/years) - 1
        cagr_market = (equity_market.iloc[-1] /
                       capital) ** (1/years) - 1
    else:
        cagr_strat = cagr_market = 0

    sharpe = (strat_returns.mean() /
              strat_returns.std() *
              np.sqrt(252)) if strat_returns.std() > 0 else 0

    rolling_max = equity_strat.cummax()
    max_dd      = ((equity_strat - rolling_max) /
                    rolling_max).min() * 100

    return {
        "equity_strat":  equity_strat,
        "equity_market": equity_market,
        "cagr_strat":    round(cagr_strat * 100, 2),
        "cagr_market":   round(cagr_market * 100, 2),
        "sharpe":        round(sharpe, 3),
        "max_dd":        round(max_dd, 2),
        "n_signals":     int((signal > 0).sum()),
    }

def plot_sentiment_dashboard(sentiment_df: pd.DataFrame,
                              ticker:       str,
                              price_data:   pd.DataFrame) -> None:
    """
    Vollständiges Sentiment Dashboard.
    Panel 1: Sentiment Score Verteilung
    Panel 2: Kurs + Sentiment Overlay
    Panel 3: Bull/Bear/Neutral Pie
    Panel 4: Top Publisher Heatmap
    """
    if sentiment_df.empty:
        print("Keine Sentiment-Daten.")
        return

    close = price_data["Close"].squeeze()

    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=[
            "Sentiment Score Verteilung",
            f"{ticker} Kurs + Sentiment",
            "Sentiment Breakdown",
            "Score nach Publisher"
        ],
        specs=[
            [{"type": "histogram"}, {"secondary_y": True}],
            [{"type": "pie"},       {"type": "bar"}]
        ],
        vertical_spacing=0.12,
        horizontal_spacing=0.10
    )

    # Panel 1: Score Verteilung
    if "score" in sentiment_df.columns:
        colors_hist = [
            "#16a34a" if s > 0.05
            else ("#ef4444" if s < -0.05 else "#94a3b8")
            for s in sentiment_df["score"]
        ]

        fig.add_trace(go.Histogram(
            x=sentiment_df["score"],
            nbinsx=30,
            name="Sentiment Score",
            marker_color="#3b82f6",
            opacity=0.75,
        ), row=1, col=1)

        fig.add_vline(
            x=0, line_dash="dot",
            line_color="#1e293b",
            opacity=0.5, row=1, col=1
        )

    # Panel 2: Kurs + Sentiment
    fig.add_trace(go.Scatter(
        x=close.index[-60:],
        y=close.values[-60:].round(2),
        name="Kurs",
        line=dict(color="#1e293b", width=2)
    ), row=1, col=2)

    if "score" in sentiment_df.columns and \
       "date" in sentiment_df.columns:
        sent_dates  = pd.to_datetime(sentiment_df["date"])
        sent_scores = sentiment_df["score"]

        bar_colors = [
            "#16a34a" if s > 0.05
            else ("#ef4444" if s < -0.05 else "#94a3b8")
            for s in sent_scores
        ]

        fig.add_trace(go.Bar(
            x=sent_dates,
            y=sent_scores.round(3),
            name="Sentiment",
            marker_color=bar_colors,
            opacity=0.65,
        ), row=1, col=2, secondary_y=True)

    # Panel 3: Pie Chart
    if "sentiment" in sentiment_df.columns:
        sent_counts = sentiment_df["sentiment"].value_counts()
        pie_colors  = {
            "bullish": "#16a34a",
            "neutral": "#94a3b8",
            "bearish": "#ef4444",
        }
        fig.add_trace(go.Pie(
            labels=sent_counts.index.tolist(),
            values=sent_counts.values.tolist(),
            hole=0.45,
            marker_colors=[
                pie_colors.get(k, "#3b82f6")
                for k in sent_counts.index
            ],
            textinfo="label+percent",
            showlegend=False
        ), row=2, col=1)

    # Panel 4: Publisher Scores
    if "publisher" in sentiment_df.columns and \
       "score" in sentiment_df.columns:
        pub_scores = (
            sentiment_df.groupby("publisher")["score"]
            .mean()
            .sort_values(ascending=True)
            .tail(10)
        )

        pub_colors = [
            "#16a34a" if v > 0 else "#ef4444"
            for v in pub_scores.values
        ]

        fig.add_trace(go.Bar(
            x=pub_scores.values.round(3),
            y=pub_scores.index.tolist(),
            orientation="h",
            marker_color=pub_colors,
            name="Publisher Score",
            showlegend=False
        ), row=2, col=2)

    fig.update_layout(
        height=700,
        template="plotly_white",
        title=f"{ticker} — LLM Sentiment Dashboard",
        legend=dict(orientation="h", y=1.02),
        margin=dict(l=0, r=0, t=60, b=0)
    )

    fig.update_xaxes(title_text="Score",       row=1, col=1)
    fig.update_xaxes(title_text="Score",       row=2, col=2)
    fig.update_yaxes(title_text="Häufigkeit",  row=1, col=1)
    fig.update_yaxes(title_text="Kurs ($)",    row=1, col=2)

    fig.show()


def plot_multi_ticker_sentiment(results: dict) -> None:
    """
    Vergleicht Sentiment Score mehrerer Ticker.
    """
    if not results:
        return

    tickers = list(results.keys())
    scores  = [results[t].get("avg_score", 0) for t in tickers]
    counts  = [results[t].get("n_articles", 0) for t in tickers]

    bar_colors = [
        "#16a34a" if s > 0.05
        else ("#ef4444" if s < -0.05 else "#94a3b8")
        for s in scores
    ]

    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=[
            "Avg. Sentiment Score",
            "Anzahl Artikel"
        ],
        horizontal_spacing=0.12
    )

    fig.add_trace(go.Bar(
        x=tickers,
        y=[round(s, 3) for s in scores],
        marker_color=bar_colors,
        text=[f"{s:+.3f}" for s in scores],
        textposition="outside",
        showlegend=False
    ), row=1, col=1)

    fig.add_hline(
        y=0, line_dash="dot",
        line_color="#1e293b",
        opacity=0.4, row=1, col=1
    )

    fig.add_trace(go.Bar(
        x=tickers,
        y=counts,
        marker_color="#3b82f6",
        text=counts,
        textposition="outside",
        opacity=0.8,
        showlegend=False
    ), row=1, col=2)

    fig.update_layout(
        height=420,
        template="plotly_white",
        title="Multi-Ticker Sentiment Vergleich",
        margin=dict(l=0, r=0, t=60, b=0)
    )

    fig.update_yaxes(title_text="Score",   row=1, col=1)
    fig.update_yaxes(title_text="Artikel", row=1, col=2)

    fig.show()


def plot_trade_thesis(thesis: dict) -> None:
    """
    Visualisiert Trade-These mit Price Targets.
    """
    if not thesis or "current_price" not in thesis:
        return

    current    = thesis["current_price"]
    bull_tgt   = thesis.get("price_target_bull", current * 1.15)
    base_tgt   = thesis.get("price_target_base", current * 1.05)
    bear_tgt   = thesis.get("price_target_bear", current * 0.90)
    ticker     = thesis.get("ticker", "")
    rec        = thesis.get("recommendation", "HOLD")

    fig = go.Figure()

    # Szenarien als Gauge-ähnlicher Chart
    scenarios = [
        ("Bear Case", bear_tgt,  "#ef4444"),
        ("Aktuell",   current,   "#94a3b8"),
        ("Base Case", base_tgt,  "#3b82f6"),
        ("Bull Case", bull_tgt,  "#16a34a"),
    ]

    for name, price, color in scenarios:
        upside = (price / current - 1) * 100
        fig.add_trace(go.Bar(
            x=[name],
            y=[price],
            marker_color=color,
            text=[f"${price:.0f}<br>({upside:+.1f}%)"],
            textposition="outside",
            textfont=dict(size=12),
            width=0.5,
            showlegend=False
        ))

    # Aktueller Kurs als Linie
    fig.add_hline(
        y=current,
        line_dash="dash",
        line_color="#1e293b",
        line_width=2,
        annotation_text=f"Kurs: ${current:.2f}",
        annotation_position="right"
    )

    rec_color = {
        "BUY":  "#16a34a",
        "HOLD": "#f59e0b",
        "SELL": "#ef4444"
    }.get(rec, "#94a3b8")

    fig.update_layout(
        title=(
            f"{ticker} — Trade These: "
            f"<span style='color:{rec_color}'>{rec}</span>"
        ),
        yaxis_title="Preis ($)",
        template="plotly_white",
        height=450,
        margin=dict(l=0, r=80, t=60, b=0)
    )

    fig.show()

def print_sentiment_report(ticker:       str,
                            sentiment_df: pd.DataFrame,
                            briefing:     str,
                            thesis:       dict) -> None:
    """Terminal Report."""
    print(f"\n{'='*58}")
    print(f"  SENTIMENT REPORT — {ticker}")
    print(f"{'='*58}")

    if not sentiment_df.empty and \
       "score" in sentiment_df.columns:
        avg   = sentiment_df["score"].mean()
        std   = sentiment_df["score"].std()
        bull  = (sentiment_df["sentiment"] == "bullish").sum()
        bear  = (sentiment_df["sentiment"] == "bearish").sum()
        neut  = (sentiment_df["sentiment"] == "neutral").sum()
        total = len(sentiment_df)

        print(f"\n  STATISTIKEN ({total} Artikel)")
        print(f"  {'Avg Score:':<22} {avg:>+8.3f}")
        print(f"  {'Std Dev:':<22} {std:>8.3f}")
        print(f"  {'Bullish:':<22} {bull:>8} ({bull/total*100:.0f}%)")
        print(f"  {'Bearish:':<22} {bear:>8} ({bear/total*100:.0f}%)")
        print(f"  {'Neutral:':<22} {neut:>8} ({neut/total*100:.0f}%)")

        print(f"\n  TOP BULLISH ARTIKEL:")
        top_bull = sentiment_df.nlargest(3, "score")
        for _, row in top_bull.iterrows():
            title = str(row.get("title", ""))[:60]
            score = row.get("score", 0)
            print(f"    [{score:+.2f}] {title}")

        print(f"\n  TOP BEARISH ARTIKEL:")
        top_bear = sentiment_df.nsmallest(3, "score")
        for _, row in top_bear.iterrows():
            title = str(row.get("title", ""))[:60]
            score = row.get("score", 0)
            print(f"    [{score:+.2f}] {title}")

    if briefing:
        print(f"\n  MARKT-BRIEFING:")
        print("  " + "-"*54)
        for line in briefing.split("\n"):
            if line.strip():
                print(f"  {line}")

    if thesis:
        print(f"\n  TRADE THESE:")
        print(f"  {'Empfehlung:':<22} "
              f"{thesis.get('recommendation', '—')}")
        print(f"  {'Überzeugung:':<22} "
              f"{thesis.get('conviction', '—')}")
        print(f"  {'Bull Target:':<22} "
              f"${thesis.get('price_target_bull', 0):.2f}")
        print(f"  {'Base Target:':<22} "
              f"${thesis.get('price_target_base', 0):.2f}")
        print(f"  {'Bear Target:':<22} "
              f"${thesis.get('price_target_bear', 0):.2f}")

        risks = thesis.get("key_risks", [])
        if risks:
            print(f"\n  KEY RISKS:")
            for r in risks[:3]:
                print(f"    • {r}")

    print(f"\n{'='*58}")


if __name__ == "__main__":

    print("Tag 25 — LLM Sentiment Analyse")
    print("=" * 55)

    # --- LLM Setup ---
    # Option A: OpenAI (empfohlen)
    USE_LLM = False   # Auf True setzen wenn API Key vorhanden

    client = None
    if USE_LLM:
        try:
            config = LLMConfig(
                provider    = "openai",
                model       = "gpt-4o-mini",
                api_key     = os.getenv("OPENAI_API_KEY", ""),
                max_tokens  = 500,
                temperature = 0.1,
            )
            client = LLMClient(config)
            print("✅ OpenAI Client initialisiert")

        except Exception as e:
            print(f"⚠ LLM nicht verfügbar: {e}")
            print("  → Nutze VADER Fallback")
            client = None
    else:
        print("ℹ VADER Fallback Modus aktiv")
        print("  Setze USE_LLM=True und OPENAI_API_KEY für LLM")

    # --- Ticker definieren ---
    MAG7 = ["AAPL", "MSFT", "GOOGL",
            "AMZN", "NVDA", "META", "TSLA"]

    FOCUS_TICKER = "NVDA"

    # --- Daten laden ---
    print(f"\n1. News laden für {FOCUS_TICKER}...")
    articles = load_news_yfinance(FOCUS_TICKER, max_articles=15)

    price_data = yf.download(
        FOCUS_TICKER, period="3mo",
        auto_adjust=True, progress=False
    )
    price_data.columns = price_data.columns.get_level_values(0)

    # --- Sentiment Analyse ---
    print(f"\n2. Sentiment Analyse ({len(articles)} Artikel)...")
    sentiment_df = analyze_all_articles(
        articles, client, batch_size=5, delay=1.0
    )

    print(f"\n   Ergebnis:")
    if not sentiment_df.empty and \
       "sentiment" in sentiment_df.columns:
        print(sentiment_df[
            ["title", "sentiment", "score"]
        ].head(5).to_string(index=False))

    # --- Dashboard ---
    print(f"\n3. Sentiment Dashboard...")
    plot_sentiment_dashboard(
        sentiment_df, FOCUS_TICKER, price_data
    )

    # --- Markt Briefing ---
    print(f"\n4. Markt-Briefing generieren...")
    if client:
        briefing = generate_market_briefing(
            FOCUS_TICKER, sentiment_df,
            price_data, client
        )
    else:
        # Fallback Briefing ohne LLM
        close      = price_data["Close"].squeeze()
        avg_score  = sentiment_df["score"].mean() \
                     if not sentiment_df.empty else 0
        briefing   = (
            f"{FOCUS_TICKER} handelt bei "
            f"${float(close.iloc[-1]):.2f}. "
            f"Durchschnittlicher Sentiment Score: "
            f"{avg_score:+.3f}. "
            f"News-Analyse basiert auf VADER NLP."
        )

    print(f"\n   Briefing generiert ({len(briefing)} Zeichen)")

    # --- Trade These ---
    print(f"\n5. Trade These generieren...")
    if client:
        thesis = generate_trade_thesis(
            FOCUS_TICKER, sentiment_df,
            price_data, client
        )
    else:
        # Fallback
        close  = price_data["Close"].squeeze()
        cp     = float(close.iloc[-1])
        thesis = {
            "ticker":             FOCUS_TICKER,
            "recommendation":     "HOLD",
            "conviction":         "low",
            "price_target_bull":  round(cp * 1.15, 2),
            "price_target_base":  round(cp * 1.05, 2),
            "price_target_bear":  round(cp * 0.90, 2),
            "current_price":      cp,
            "key_risks":          ["Makrorisiken",
                                   "Competitive Pressure"],
            "catalysts":          ["Earnings Beat",
                                   "AI Tailwinds"],
        }

    if thesis:
        plot_trade_thesis(thesis)

    # --- Report ---
    print_sentiment_report(
        FOCUS_TICKER, sentiment_df, briefing, thesis
    )

    # --- Earnings Call Demo ---
    print(f"\n6. Earnings Call Analyse...")
    transcript = get_synthetic_earnings(FOCUS_TICKER)

    if client:
        earnings_result = analyze_earnings_call(
            transcript, FOCUS_TICKER, client
        )
        print(f"\n   Earnings Analyse:")
        print(f"   Sentiment:      "
              f"{earnings_result.get('overall_sentiment', '—')}")
        print(f"   Management Conf:{earnings_result.get('management_confidence', 0):.2f}")
        print(f"   Guidance:       "
              f"{earnings_result.get('guidance_quality', '—')}")
        print(f"   Score:          "
              f"{earnings_result.get('score', 0):+.3f}")
    else:
        print("   (LLM nicht verfügbar — "
              "nutze USE_LLM=True für echte Analyse)")

    # --- Multi-Ticker Vergleich ---
    print(f"\n7. Multi-Ticker Sentiment Vergleich...")
    multi_results = {}

    for t in MAG7[:4]:
        arts = load_news_yfinance(t, max_articles=8)
        sdf  = analyze_all_articles(
            arts, client, batch_size=5, delay=0.5
        )

        if not sdf.empty and "score" in sdf.columns:
            multi_results[t] = {
                "avg_score":   round(sdf["score"].mean(), 3),
                "n_articles":  len(sdf),
                "sentiment_df": sdf,
            }
        else:
            multi_results[t] = {
                "avg_score":  0.0,
                "n_articles": 0,
            }

        time.sleep(0.5)

    print("\n   Multi-Ticker Summary:")
    for t, r in multi_results.items():
        score = r["avg_score"]
        n     = r["n_articles"]
        label = ("🟢" if score > 0.05
                 else ("🔴" if score < -0.05 else "⚪"))
        print(f"   {t:<8} {label} {score:+.3f}  "
              f"({n} Artikel)")

    plot_multi_ticker_sentiment(multi_results)

    # --- Signal Backtest Demo ---
    print(f"\n8. Sentiment Signal Backtest...")
    if not sentiment_df.empty:
        signal_df = sentiment_to_signal(
            sentiment_df.copy(),
            price_data,
            score_threshold = 0.15,
            window          = 3
        )

        if not signal_df.empty:
            bt = backtest_sentiment_signal(signal_df, capital=10_000)
            if bt:
                print(f"   CAGR Sentiment: {bt['cagr_strat']:+.2f}%")
                print(f"   CAGR B&H:       {bt['cagr_market']:+.2f}%")
                print(f"   Sharpe:         {bt['sharpe']:.3f}")
                print(f"   Signals:        {bt['n_signals']}")

    # --- Export ---
    if not sentiment_df.empty:
        sentiment_df.to_csv(
            f"day25_sentiment_{FOCUS_TICKER}.csv",
            index=False
        )
        print(f"\nGespeichert: "
              f"day25_sentiment_{FOCUS_TICKER}.csv")

    if thesis:
        with open(f"day25_thesis_{FOCUS_TICKER}.json", "w") as f:
            json.dump(thesis, f, indent=2, default=str)
        print(f"Gespeichert: "
              f"day25_thesis_{FOCUS_TICKER}.json")