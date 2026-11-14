"""
sentiment_agent.py - Fundamental & News Sentiment Agent (Phase 3, Node 2)

Triggered only when Technical Agent detects a high-conviction BUY or SELL setup.
Combines:
1. CryptoPanic API community votes + headline NLP (when CRYPTOPANIC_API_KEY is set).
2. NLTK VADER SentimentIntensityAnalyzer on headline text.
3. Public Bitcoin RSS feed fallback (CoinTelegraph / CoinDesk) when no API key is set.
"""

import os
import logging
import xml.etree.ElementTree as ET
from typing import List, Optional

import requests
from dotenv import load_dotenv

from agents.state import TradingState

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("SentimentAgent")

CRYPTOPANIC_API_KEY = os.getenv("CRYPTOPANIC_API_KEY", "")
ENABLE_RSS_FALLBACK = os.getenv("ENABLE_RSS_FALLBACK", "false").lower() == "true"

RSS_FEEDS = (
    "https://cointelegraph.com/rss/tag/bitcoin",
    "https://www.coindesk.com/arc/outboundfeeds/rss/",
)


def analyze_headlines_vader(headlines: List[str]) -> float:
    """
    Score a list of crypto news headlines using NLTK VADER compound polarity.
    Returns an average score clamped to [-1.0, 1.0].
    """
    clean_titles = [h.strip() for h in headlines if isinstance(h, str) and h.strip()]
    if not clean_titles:
        return 0.0
    try:
        from nltk.sentiment.vader import SentimentIntensityAnalyzer
        sia = SentimentIntensityAnalyzer()
        scores = [sia.polarity_scores(title)["compound"] for title in clean_titles]
        avg_score = sum(scores) / len(scores)
        return round(max(-1.0, min(1.0, avg_score)), 2)
    except Exception as e:
        logger.warning(f"VADER analysis unavailable: {e}")
        return 0.0


# ==========================================
# Pluggable Headline NLP Engine (FinBERT -> VADER fallback), 100% free & local
# ==========================================
SENTIMENT_ENGINE = os.getenv("SENTIMENT_ENGINE", "vader").strip().lower()
_FINBERT_PIPELINE = None
_FINBERT_LOAD_ATTEMPTED = False


def analyze_headlines_finbert(headlines: List[str]) -> Optional[float]:
    """
    Score headlines with the free local FinBERT model (ProsusAI/finbert via `transformers`).
    Returns None if transformers/torch are not installed so callers can fall back to VADER.
    Optional install: pip install transformers torch
    """
    global _FINBERT_PIPELINE, _FINBERT_LOAD_ATTEMPTED
    clean_titles = [h.strip() for h in headlines if isinstance(h, str) and h.strip()]
    if not clean_titles:
        return 0.0
    if not _FINBERT_LOAD_ATTEMPTED:
        _FINBERT_LOAD_ATTEMPTED = True
        try:
            from transformers import pipeline
            _FINBERT_PIPELINE = pipeline("text-classification", model="ProsusAI/finbert")
        except Exception as e:
            logger.info(f"FinBERT unavailable ({type(e).__name__}); using VADER instead.")
            _FINBERT_PIPELINE = None
    if _FINBERT_PIPELINE is None:
        return None
    try:
        results = _FINBERT_PIPELINE(clean_titles, truncation=True)
        signed = []
        for r in results:
            label = str(r.get("label", "")).lower()
            prob = float(r.get("score", 0.0))
            signed.append(prob if label == "positive" else -prob if label == "negative" else 0.0)
        return round(max(-1.0, min(1.0, sum(signed) / len(signed))), 2)
    except Exception as e:
        logger.warning(f"FinBERT inference failed: {e}")
        return None


def analyze_headlines(headlines: List[str]) -> float:
    """Dispatch to the configured NLP engine (SENTIMENT_ENGINE=finbert|vader)."""
    if SENTIMENT_ENGINE == "finbert":
        score = analyze_headlines_finbert(headlines)
        if score is not None:
            return score
    return analyze_headlines_vader(headlines)


def fetch_rss_headlines(max_items: int = 10) -> List[str]:
    """Fetch recent Bitcoin headlines from public RSS feeds as a free fallback."""
    headlines: List[str] = []
    headers = {"User-Agent": "NeuralPulseTrading/1.0"}
    for feed_url in RSS_FEEDS:
        try:
            resp = requests.get(feed_url, headers=headers, timeout=4)
            resp.raise_for_status()
            root = ET.fromstring(resp.content)
            for item in root.findall(".//item/title"):
                if item.text:
                    headlines.append(item.text.strip())
                if len(headlines) >= max_items:
                    return headlines
        except Exception:
            continue
    return headlines


def _describe_sentiment(score: float) -> str:
    if score > 0.2:
        return "Bullish Fundamental News"
    elif score < -0.2:
        return "Bearish Fundamental News"
    return "Neutral Market News"


def sentiment_analyst_agent(state: TradingState) -> TradingState:
    """
    Sentiment Agent — Node 2 of the LangGraph pipeline.
    Evaluates news sentiment (-1.0 to 1.0) to confirm technical setups.
    """
    # Check for API key
    if not CRYPTOPANIC_API_KEY or CRYPTOPANIC_API_KEY == "your_api_key_here":
        if ENABLE_RSS_FALLBACK:
            rss_titles = fetch_rss_headlines(max_items=10)
            if rss_titles:
                score = analyze_headlines(rss_titles)
                state["sentiment_score"] = score
                state["sentiment_summary"] = f"{_describe_sentiment(score)} (RSS+{SENTIMENT_ENGINE.upper()})"
                return state

        logger.info("CRYPTOPANIC_API_KEY not configured; defaulting to NEUTRAL sentiment (0.0).")
        state["sentiment_score"] = 0.0
        state["sentiment_summary"] = "Neutral (No API key configured)"
        return state

    url = (
        f"https://cryptopanic.com/api/free/v1/posts/"
        f"?auth_token={CRYPTOPANIC_API_KEY}"
        f"&filter=important&currencies=BTC"
    )

    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        news_data = response.json()

        total_bullish = 0
        total_bearish = 0
        titles: List[str] = []

        for post in news_data.get("results", [])[:10]:
            votes = post.get("votes", {})
            total_bullish += votes.get("positive", 0) + votes.get("important", 0)
            total_bearish += votes.get("negative", 0) + votes.get("toxic", 0)
            if post.get("title"):
                titles.append(str(post["title"]))

        has_votes = (total_bullish + total_bearish) > 0
        vote_score = (
            (total_bullish - total_bearish) / (total_bullish + total_bearish)
            if has_votes else 0.0
        )
        vader_score = analyze_headlines(titles) if titles else 0.0

        if has_votes and titles:
            score = (0.7 * vote_score) + (0.3 * vader_score)
        elif has_votes:
            score = vote_score
        elif titles:
            score = vader_score
        else:
            score = 0.0

        score = max(-1.0, min(1.0, score))
        state["sentiment_score"] = round(score, 2)
        state["sentiment_summary"] = _describe_sentiment(state["sentiment_score"])

        logger.info(
            f"Sentiment score: {state['sentiment_score']} "
            f"(Votes: +{total_bullish}/-{total_bearish}, Headlines: {len(titles)})"
        )

    except requests.exceptions.HTTPError as e:
        logger.error(f"CryptoPanic API error: {e}")
        state["sentiment_score"] = 0.0
        state["sentiment_summary"] = "Neutral (News API error)"
    except Exception as e:
        logger.error(f"Error fetching news: {e}")
        state["sentiment_score"] = 0.0
        state["sentiment_summary"] = "Neutral (News unavailable)"

    return state
