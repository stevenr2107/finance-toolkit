"""
Day 11 - News Sentiment Analyse
Kombiniert NLP (VADER, TextBlob) mit yfinance News
Korreliert Sentiment mit Kursperformance 
Zeigt wo Smart Money bs retail sentiment divergiert 
"""

import yfinance as yf
import pandas as pd 
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from textblob import TextBlob
from datetime import datetime, timedelta
import warnings 
warnings.filterwarnings("ignore")


def get_news(ticker: str, max_articles: int=50) -> pd.DataFrame:
    """
    Lädt News für einen Ricker via yfinance 
    kein api key nötig - direkt von yahoo finance 
    """
    stock = yf.Ticker(ticker)

    try:
        news=stock.news
    except Exception as e:
        print(f"Fehler beim Laden der News: {e}")
        return pd.DataFrame()
    
    if not news:
        print(f"Keine News für {ticker} gefunden.")
        return pd.DataFrame()
    
    rows = []
    for article in news[:max_articles]:
        # Neue Yahoo Struktur
        # daten stecken in content
        # Timestamp konvertieren 
        content = article.get("content") or {}
        # Datum ist jetzt direkt ein String (z.B. "2026-04-13T15:42:22Z")
        pub_date = content.get("pubDate")
        # direktesumwandeln des datums
        dt = pd.to_datetime(pub_date) if pub_date else None

        provider = content.get("provider") or {}
        click_url = content.get("clickThroughUrl") or {}

        rows.append({
            "datetime": dt,
            "title": content.get("title", ""),
            "publisher": provider.get("displayName", ""),
            # Link ist verschachtelt in clickThroughUrl -> url
            "link": click_url.get("url", ""),
            "type": content.get("contentType", ""),
        })
    
    # Ersetzt leere Strings durch NaN und wirft dann alles ohne Titel ODER Datum rigoros raus
    df = pd.DataFrame(rows).replace("", np.nan)
    # 2. Wir werfen jede Zeile raus, die keinen Titel ODER kein Datum hat
    df = df.dropna(subset=["title", "datetime"]) 
    # 3. Sortieren und Index aufräumen
    df = df.sort_values("datetime", ascending=False).reset_index(drop=True)

    print(f"{len(df)} News-Artikel für {ticker} geladen")
    return df

# --- VADER Sentiment ---
def analyze_vader(news_df: pd.DataFrame) ->pd.DataFrame:
    """
    VADER - speziell für Social Media und kurze Texte entwickelt.
    Versteht Großschreibung, Ausrufezeichen, Emojis

    Compound Score:
    > 0.05 -> positiv 
    < -0.05 -> Negativ
    dazwischen -> Neutral
    """
    analyzer = SentimentIntensityAnalyzer()
    df = news_df.copy()

    scores = df["title"].apply(
        lambda text: analyzer.polarity_scores(str(text))
    )

    df["vader_compound"] = scores.apply(lambda x: x["compound"])
    df["vader_pos"] = scores.apply(lambda x: x["pos"])
    df["vader_neg"] = scores.apply(lambda x: x["neg"])
    df["vader_neu"] = scores.apply(lambda x: x["neu"])

    # Label
    df["vader_label"] = df["vader_compound"].apply(
        lambda x: "Positiv" if x > 0.05
            else ("Negativ" if x < -0.05 else "Neutral")
    )

    return df

# --- Textblob sentiment ---

def analyze_textblob(news_df: pd.DataFrame) -> pd.DataFrame:
    """
    Textblob - klassisches nlp, versteht kontext besser als vader 

    polarity: -1.0 (sehr negativ) bis 1.0 (sehr positiv)
    Subjectivity: 0.0 (faktisch) bis 1.0 (meinungsbasiert)

    Kombination: hohe Subjektivität + positive Polarity = Hype 
    """
    df = news_df.copy()

    analysis = df["title"].apply(
        lambda text: TextBlob(str(text))
    )

    df["tb_polarity"] = analysis.apply(lambda x: x.sentiment.polarity)
    df["tb_subjectivity"] = analysis.apply(lambda x: x.sentiment.subjectivity)

    df["tb_label"] = df["tb_polarity"].apply(
        lambda x: "Positiv" if x > 0.05
            else ("Negativ" if x < -0.05 else "Neutral")
    )

    return df

# --- Kombinierter Sentiment Score ---
def compute_combined_score(news_df: pd.DataFrame) -> pd.DataFrame:
    """
    Kombiniert vader und textblob zu einem einzigen score 
    vader gewichtet stärker - besser für finanz headlines

    Bonus: erkennt Hype (hohe subjektivität trotz positivem score)
    """
    df = news_df.copy()

    # Gewichteter Score: 60% Vader, 40% Textblob
    df["combined_score"] = (
        df["vader_compound"] * 0.60 +
        df["tb_polarity"] * 0.40
    ).round(4)

    # Sentiment Stärke 
    df["strength"] = df["combined_score"].abs()

    # Hype Detektor: positiv aber sehr subjektiv
    df["is_hype"] = (
        (df["combined_score"] > 0.3) &
        (df["tb_subjectivity"] > 0.6)
    )

    # Finales Label
    df["sentiment"] = df["combined_score"].apply(
        lambda x: "Stark Positiv" if x > 0.3
            else ("Positiv" if x > 0.05 
            else ("Neutral" if x > -0.05
            else ("Negativ" if x > -0.3
            else "Stark Negativ")))

    )

    return df

# --- Sentiment über Zeit ---

def sentiment_over_time(news_df: pd.DataFrame,
                        ticker: str) -> pd.DataFrame:
    """
    Aggregiert sentiment pro tag
    erlaubt korrelation mit kurspeformance 
    """
    df = news_df.copy()
    df["date"] = pd.to_datetime(df["datetime"]).dt.date

    daily = df.groupby("date").agg(
        avg_score = ("combined_score", "mean"),
        article_count = ("combined_score", "count"),
        pos_count = ("sentiment",
                     lambda x:  (x == "Stark Positiv").sum() +
                                (x == "Positiv").sum()),
        neg_count = ("sentiment",
                     lambda x:  (x == "Stark Negativ").sum() +
                                (x == "Negativ").sum()),
        hype_count = ("is_hype", "sum"),
    ).reset_index()

    daily["date"] = pd.to_datetime(daily["date"])

    # Sentiment Ratio pos vs neg
    daily["sentiment_ratio"] = (
        daily["pos_count"] /
        (daily["pos_count"] + daily["neg_count"] + 1e-6)
    ).round(3)

    return daily.sort_values("date")

# --- Korrelation mit kursperformance ---

def correlate_with_price(sentiment_daily: pd.DataFrame,
                         ticker: str) -> pd.DataFrame:
    
    """
    Vergleicht sentiment mit nächsten tag return 

    die wichtige frage: sagt sentiment den kurs voraus?
    ( Spoiler: manchmal - aber nicht zuverlässig genug alleine)
    """
    # Kursdaten laden
    start = sentiment_daily["date"].min() - timedelta(days= 5)
    end = sentiment_daily["date"].max() + timedelta(days= 5)

    prices = yf.download(
        ticker,
        start=start.strftime("%Y-%m-%d"),
        end=end.strftime("%Y-%m-%d"),
        auto_adjust=True,
        progress=False
    )["Close"].squeeze()

    prices_df = pd.DataFrame({
        "date":     pd.to_datetime(prices.index),
        "close":    prices.values,
        "next_return": prices.pct_change().shift(-1).values * 100
    })

    # Mergen 
    merged = sentiment_daily.merge(
        prices_df, on="date", how="inner"
    )

    # Korrelation berechnen
    corr = merged[["avg_score", "next_return"]].corr().iloc[0, 1]
    print(f"\nKorrelation Sentiment -> Next-Day Return: {corr:.3f}")
    print("(> 0.3 wäre sehr gut | typisch: 0.05–0.20)")

    return merged

# --- Visualisierungen ---

def plot_sentiment_dashboard(merged: pd.DataFrame,
                             ticker: str) -> None:
    
    """
    3-Panel Chart:
    1. Kurs + Sentiment Score overlay
    2. Artikel-Volumen + Hype Detektor
    3. Sentiment Distribution
    """
    fig = make_subplots(
        rows=3, cols=1,
        shared_xaxes=False,
        vertical_spacing=0.08,
        subplot_titles=[
            f"{ticker} Kurs vs. Sentiment Score",
            "Artikel-Volumen & Hype Detektor",
            "Sentiment-Verteilung"
        ],
        row_heights=[0.45, 0.3, 0.25],
        specs= [[{"secondary_y": True}],
                [{"secondary_y": False}],
                [{"secondary_y": False}]]
    )

    # --- Panel 1 : Kurs + Sentiment ---
    # Kurs 
    fig.add_trace(go.Scatter(
        x=merged["date"],
        y=merged["close"].round(2),
        name="Kurs",
        line=dict(color="#1e293b", width=2)
    ), row=1, col=1, secondary_y=False)

    # Sentiment Score als Balken 
    sent_colors = [
        "#16a34a" if v > 0.05
        else ("#ef4444" if v < -0.05 else "#94a3b8")
        for v in merged["avg_score"]
    ]

    fig.add_trace(go.Bar(
        x=merged["date"],
        y=merged["avg_score"].round(3),
        name="Sentiment",
        marker_color=sent_colors,
        opacity=0.7
    ), row=1, col=1, secondary_y=True)

    fig.add_hline(y=0, line_dash="dot",
                  line_color="#94a3b8", opacity=0.5,
                  row=1, col=1)
    
    # --- Panel 2: Volumen + Hype---
    fig.add_trace(go.Bar(
        x=merged["date"],
        y=merged["article_count"],
        name="Artikel/Tag",
        marker_color="#3b82f6",
        opacity=0.7
    ), row=2, col=1)

    # Hype MArker 
    hype_days = merged[merged["hype_count"] > 0]
    if not hype_days.empty:
        fig.add_trace(go.Scatter(
            x=hype_days["date"],
            y=hype_days["article_count"],
            mode="markers",
            name="Hype-Signal",
            marker=dict(
                color="#f59e0b",
                size=14,
                symbol="star",
                line=dict(width=1, color="white")
            )
        ), row=2, col=1)

    # --- Panel 3: Distribution ---
    sentiment_counts = merged.groupby(
        merged["avg_score"].apply(
            lambda x: "Positiv" if x > 0.05
                else ("Negativ" if x < -0.05 else "Neutral")
        )
    ).size() # schaut wieviele artikel in den jeweiligen gruppen sind 

    dist_colors = {
        "Positiv": "#16a34a",
        "Neutral":  "#94a3b8",
        "Negativ": "#ef4444"
    }

    fig.add_trace(go.Bar(
        x=sentiment_counts.index,
        y=sentiment_counts.values,
        marker_color=[dist_colors.get(k, "#94a3b8")
                      for k in sentiment_counts.index],
        text=sentiment_counts.values,
        textposition="outside",
        showlegend=False
    ), row=3, col=1)

    fig.update_layout(
        height=750,
        template="plotly_white",
        legend=dict(orientation="h", y=1.02),
        margin=dict(l=0, r=0, t=40, b=0)
    )

    fig.update_yaxes(title_text="Kurs ($)", row=1, col=1,
                     secondary_y=False)
    fig.update_yaxes(title_text="Sentiment Score", row=1, col=1,
                     secondary_y=True)
    fig.update_yaxes(title_text="Artikel", row=2, col=1)
    fig.update_yaxes(title_text="Tage", row=3, col=1)

    fig.show()

def plot_top_headlines(news_df: pd.DataFrame,
                       top_n: int=10) -> None:
    """
    Zeigt die stärksten positiven und negativen Headlines.
    """
    df = news_df.copy()

    top_pos = df.nlargest(top_n // 2, "combined_score")[
        ["title", "combined_score", "publisher", "datetime"]
    ]
    top_neg = df.nsmallest(top_n // 2, "combined_score")[
        ["title", "combined_score", "publisher", "datetime"]
    ]

    print("\n" + "="*65)
    print("  TOP POSITIVE HEALDINES")
    print("="*65)
    for _, row in top_pos.iterrows():
        score = row["combined_score"]
        title = str(row["title"])[:70]
        print(f"  [{score:+.3f}]{title}")
        print(f"        {row['publisher']}  · "
              f"{row['datetime'].strftime('%d.%m.%Y') if row['datetime'] else '—'}")
        print()

    print("="*65)
    print("  TOP NEGATIVE HEADLINES")
    print("="*65)
    for _, row in top_neg.iterrows():
        score = row["combined_score"]
        title = str(row["title"])[:70]
        print(f"  [{score:+.3f}] {title}")
        print(f"           {row['publisher']} · "
              f"{row['datetime'].strftime('%d.%m.%Y') if row['datetime'] else '—'}")
        print()


#--- Multi- Ticker segment vergleich 

def compare_sentiment(tickers:list) -> pd.DataFrame:
    """
    Vergleicht sentiment score mehrer aktien gleichzeitig
    Nützlich für sektor analyse: wer hat gerade den besten / schlechtesten newsflow?
    """
    analyzer = SentimentIntensityAnalyzer()
    results = []

    for ticker in tickers:
        news = get_news(ticker, max_articles=30)

        if news.empty:
            continue
    
        scores = news["title"].apply(
            lambda t: analyzer.polarity_scores(str(t))["compound"]
        )

        results.append({
            "Ticker":         ticker,
            "Avg Sentiment":  round(scores.mean(), 3),
            "Positiv (%)":    round((scores > 0.05).mean() * 100, 1),
            "Negativ (%)":    round((scores < -0.05).mean() * 100, 1),
            "Artikel":        len(news),
            "Stärkstes Pos":  round(scores.max(), 3),
            "Stärkstes Neg":  round(scores.min(), 3),
        })

    df = pd.DataFrame(results).sort_values(
        "Avg Sentiment", ascending=False
    ).reset_index(drop=True)
    return df

def plot_sentiment_comparison(comparison_df: pd.DataFrame) -> None:
    """ Horizontaler Bar Chart - Sentiment Ranking """
    colors = [
        "#16a34a" if v> 0.05
        else ("#ef4444" if v < -0.05 else "#94a3b8")
        for v in comparison_df["Avg Sentiment"]
    ]

    fig = go.Figure(go.Bar(
        x=comparison_df["Avg Sentiment"],
        y=comparison_df["Ticker"],
        orientation = "h",
        marker_color=colors,
        text=[f"{v:+.3f}" for v in comparison_df["Avg Sentiment"]],
        textposition="outside"
    ))

    fig.add_vline(x=0, line_dash="solid", line_color="#1e293b", line_width=1.5)

    fig.update_layout(
        title="News Sentiment Vergleich",
        xaxis_title="Avg. Sentiment Score",
        template = "plotly_white",
        height = max(300, len(comparison_df) * 45),
        margin=dict(l=0,r=60,t=40,b=0)
    )

    fig.show()


if __name__ == "__main__":

    #--- Single Ticker deep dive---
    TICKER = "NVDA"
    print(f"\nSentiment Analyse: {TICKER}")
    print("="*50)

    #News laden
    news = get_news(TICKER, max_articles=50)

    if not news.empty:
        #Indikatoren berechnen
        news = analyze_vader(news)
        news = analyze_textblob(news)
        news = compute_combined_score(news)

        # Top Headlines
        plot_top_headlines(news, top_n=8)

        #Zeitreihe
        daily= sentiment_over_time(news, TICKER)

        # mit Kurs korrelieren 
        merged= correlate_with_price(daily, TICKER)

        # Sentiment Übersicht 
        print("\nSENTIMENT VERTEILUNG")
        print(news["sentiment"].value_counts().to_string())

        print(f"\nDurchschnittlicher Score: "
              f"{news['combined_score'].mean():+.3f}")
        print(f"Hype-Artikel erkannt:    "
              f"{news['is_hype'].sum()}")
        
        # Charts 
        plot_sentiment_dashboard(merged, TICKER)

    # Multi ticker verglecih 

    print("\n" + "="*50)
    print("SEKTOR SENTIMENT VERGLEICH")

    mag7 = ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA"]
    comparison = compare_sentiment(mag7)

    print(comparison.to_string(index = False))
    plot_sentiment_comparison(comparison)

    # Export
    if not news.empty:
        news.to_csv(f"sentiment_{TICKER}.csv", index=False)
        print(f"\nGespeichert: sentiment_{TICKER}.csv")
