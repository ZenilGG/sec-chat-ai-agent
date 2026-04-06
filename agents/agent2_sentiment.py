"""
Agent 2: X (Twitter) Sentiment Scraper & Analyzer
===================================================
Takes the top 5 tickers from Agent 1, uses Apify to scrape tweets
about those tickers from the last 7 days, then performs sentiment
analysis on the collected tweets.
"""

import os
import time
import pandas as pd
from datetime import datetime, timedelta
from typing import Optional
from apify_client import ApifyClient
from dotenv import load_dotenv
from rich.console import Console
from rich.table import Table
from tenacity import retry, stop_after_attempt, wait_exponential

load_dotenv()
console = Console()


# ── Simple rule-based sentiment (no external ML needed) ─────────────────────

BULLISH_WORDS = {
    "buy", "bull", "bullish", "moon", "rocket", "breakout", "surge", "soar",
    "strong", "beat", "outperform", "upgrade", "long", "growth", "profit",
    "gain", "rally", "pumping", "up", "undervalued", "opportunity", "hold",
}

BEARISH_WORDS = {
    "sell", "bear", "bearish", "crash", "dump", "short", "overvalued", "risk",
    "loss", "decline", "drop", "fall", "weak", "downgrade", "avoid", "fraud",
    "warning", "red", "correction", "bubble", "down", "panic",
}


def simple_sentiment(text: str) -> tuple[str, float]:
    """
    Returns (label, score) where:
      label: 'bullish' | 'bearish' | 'neutral'
      score: float in [-1, 1]
    """
    words = set(text.lower().split())
    bull_hits = len(words & BULLISH_WORDS)
    bear_hits = len(words & BEARISH_WORDS)

    total = bull_hits + bear_hits
    if total == 0:
        return "neutral", 0.0

    score = (bull_hits - bear_hits) / total
    if score > 0.1:
        return "bullish", round(score, 3)
    elif score < -0.1:
        return "bearish", round(score, 3)
    return "neutral", round(score, 3)


# ── Main Agent Class ─────────────────────────────────────────────────────────

class XSentimentAgent:
    """
    Uses Apify's Twitter scraper to collect tweets for each ticker,
    then analyzes sentiment across all collected tweets.
    """

    APIFY_ACTOR_ID = "apidojo/tweet-scraper"   # Free Apify Twitter scraper

    def __init__(self):
        token = os.getenv("APIFY_API_TOKEN")
        if not token:
            raise ValueError("APIFY_API_TOKEN not set in .env")
        self.client = ApifyClient(token)
        self.lookback_days = 7

    def _build_search_queries(self, tickers: list[str]) -> list[str]:
        """Build search queries for each ticker (cashtag + keyword)."""
        queries = []
        for t in tickers:
            queries.append(f"${t} OR #{t} lang:en -is:retweet")
        return queries

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=2, min=4, max=30))
    def _scrape_tweets_for_ticker(self, ticker: str, max_tweets: int = 100) -> list[dict]:
        """
        Run Apify tweet scraper for a single ticker.
        Returns list of tweet dicts.
        """
        since_date = (datetime.now() - timedelta(days=self.lookback_days)).strftime("%Y-%m-%d")
        query = f"${ticker} OR #{ticker} lang:en -is:retweet since:{since_date}"

        console.log(f"[cyan]Scraping tweets for ${ticker}...")

        run_input = {
            "searchTerms": [query],
            "maxTweets": max_tweets,
            "addUserInfo": False,
            "startUrls": [],
        }

        try:
            run = self.client.actor(self.APIFY_ACTOR_ID).call(run_input=run_input)
            tweets = []
            for item in self.client.dataset(run["defaultDatasetId"]).iterate_items():
                tweets.append(item)
            console.log(f"[green]Collected {len(tweets)} tweets for ${ticker}")
            return tweets
        except Exception as e:
            console.log(f"[red]Apify error for {ticker}: {e}")
            return self._mock_tweets(ticker)   # fallback to mock

    def _mock_tweets(self, ticker: str) -> list[dict]:
        """Generate mock tweet data for demo/testing when Apify is unavailable."""
        console.log(f"[yellow]Using mock tweet data for ${ticker}")
        templates = [
            f"${ticker} looking very bullish today! Strong insider buying signal 🚀",
            f"Insiders buying ${ ticker} massively. This is a strong buy signal.",
            f"${ticker} breaking out! Moon incoming 🌙 #stocks",
            f"Be careful with ${ticker}, the rally might not last long.",
            f"${ticker} fundamentals look weak despite the insider activity.",
            f"Watching ${ticker} closely. Neutral for now.",
            f"${ticker} is a solid hold. Long term bullish on this one.",
            f"Selling my ${ticker} position. Too much risk.",
            f"${ticker} to the moon! This is the moment 🚀🚀",
            f"${ticker} seems overvalued at this point. Bear case is strong.",
        ]
        import random
        random.seed(42)
        return [
            {
                "full_text": t,
                "created_at": (datetime.now() - timedelta(days=random.randint(0, 7))).isoformat(),
                "public_metrics": {"like_count": random.randint(1, 500), "retweet_count": random.randint(0, 100)},
            }
            for t in templates
        ]

    def _analyze_tweet(self, tweet: dict) -> dict:
        """Extract text and analyze sentiment for one tweet."""
        text = tweet.get("full_text") or tweet.get("text") or ""
        label, score = simple_sentiment(text)
        return {
            "text": text[:280],
            "created_at": tweet.get("created_at", ""),
            "likes": tweet.get("public_metrics", {}).get("like_count", 0),
            "retweets": tweet.get("public_metrics", {}).get("retweet_count", 0),
            "sentiment": label,
            "sentiment_score": score,
        }

    def run(self, top5_df: pd.DataFrame) -> dict[str, pd.DataFrame]:
        """
        Main entry: scrape and analyze tweets for all 5 tickers.

        Args:
            top5_df: DataFrame from Agent 1 with 'ticker' column.

        Returns:
            Dict of {ticker: DataFrame of analyzed tweets}
        """
        console.rule("[bold cyan]Agent 2: X Sentiment Scraper")
        tickers = top5_df["ticker"].tolist()
        results = {}

        for ticker in tickers:
            raw_tweets = self._scrape_tweets_for_ticker(ticker)
            analyzed = [self._analyze_tweet(t) for t in raw_tweets]
            df = pd.DataFrame(analyzed)

            if not df.empty:
                df["ticker"] = ticker
                results[ticker] = df
                self._display_sentiment_summary(ticker, df)
            else:
                console.log(f"[yellow]No tweets found for ${ticker}")

        return results

    def get_combined_sentiment_summary(self, sentiment_data: dict[str, pd.DataFrame]) -> pd.DataFrame:
        """
        Combine per-ticker sentiment into a single summary DataFrame.
        Useful for RAG indexing in Agent 3.
        """
        rows = []
        for ticker, df in sentiment_data.items():
            if df.empty:
                continue
            counts = df["sentiment"].value_counts()
            total = len(df)
            avg_score = df["sentiment_score"].mean()
            rows.append({
                "ticker": ticker,
                "total_tweets": total,
                "bullish_pct": round(counts.get("bullish", 0) / total * 100, 1),
                "bearish_pct": round(counts.get("bearish", 0) / total * 100, 1),
                "neutral_pct": round(counts.get("neutral", 0) / total * 100, 1),
                "avg_sentiment_score": round(avg_score, 3),
                "overall_sentiment": "bullish" if avg_score > 0.05 else ("bearish" if avg_score < -0.05 else "neutral"),
            })
        return pd.DataFrame(rows)

    def _display_sentiment_summary(self, ticker: str, df: pd.DataFrame):
        """Pretty print sentiment summary for one ticker."""
        counts = df["sentiment"].value_counts()
        total = len(df)
        avg = df["sentiment_score"].mean()

        table = Table(title=f"${ticker} Sentiment ({total} tweets)", style="magenta")
        table.add_column("Sentiment", style="bold")
        table.add_column("Count", style="white")
        table.add_column("% Share", style="white")

        colors = {"bullish": "green", "bearish": "red", "neutral": "yellow"}
        for label in ["bullish", "neutral", "bearish"]:
            count = counts.get(label, 0)
            pct = f"{count/total*100:.1f}%"
            table.add_row(f"[{colors[label]}]{label}[/]", str(count), pct)

        console.print(table)
        color = "green" if avg > 0 else ("red" if avg < 0 else "yellow")
        console.print(f"  Average sentiment score: [{color}]{avg:+.3f}[/]\n")


# ── Standalone test ──────────────────────────────────────────────────────────
if __name__ == "__main__":
    # Mock top5 df for standalone testing
    top5 = pd.DataFrame({
        "ticker": ["NVDA", "MSFT", "TSLA", "AMZN", "META"],
        "entity_name": ["NVIDIA", "Microsoft", "Tesla", "Amazon", "Meta"],
        "total_value_usd": [45e6, 32e6, 28e6, 21e6, 15e6],
    })

    agent = XSentimentAgent()
    sentiment_data = agent.run(top5)
    summary = agent.get_combined_sentiment_summary(sentiment_data)
    print("\nSentiment Summary:")
    print(summary.to_string(index=False))
