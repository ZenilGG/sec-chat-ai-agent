# Sample Input / Output Examples
## CrowdWisdomTrading SEC Chat AI Agent

---

## Agent 1 — SEC Insider Trading Fetcher

### Input
```
Time window: Last 24 hours
Source: SEC EDGAR Form 4 filings
```

### Output
```
Top 5 Insider Trades (Last 24h)
┌──────┬────────┬──────────────────┬──────────────────┬───────────┐
│ Rank │ Ticker │ Company          │ Total $ Value    │ # Filings │
├──────┼────────┼──────────────────┼──────────────────┼───────────┤
│  1   │ NVDA   │ NVIDIA Corp      │ $45,000,000      │ 3         │
│  2   │ MSFT   │ Microsoft Corp   │ $32,000,000      │ 2         │
│  3   │ TSLA   │ Tesla Inc        │ $28,000,000      │ 4         │
│  4   │ AMZN   │ Amazon.com Inc   │ $21,000,000      │ 1         │
│  5   │ META   │ Meta Platforms   │ $15,000,000      │ 2         │
└──────┴────────┴──────────────────┴──────────────────┴───────────┘
```

---

## Agent 2 — X Sentiment Analyzer

### Input
```
Tickers: ["NVDA", "MSFT", "TSLA", "AMZN", "META"]
Lookback: 7 days
Source: X (Twitter) via Apify scraper
```

### Output (per ticker example — NVDA)
```
$NVDA Sentiment (100 tweets)
┌───────────┬───────┬─────────┐
│ Sentiment │ Count │ % Share │
├───────────┼───────┼─────────┤
│ bullish   │ 60    │ 60.0%   │
│ neutral   │ 22    │ 22.0%   │
│ bearish   │ 18    │ 18.0%   │
└───────────┴───────┴─────────┘
Average sentiment score: +0.420
```

### Combined Summary Output
```
ticker  total_tweets  bullish_pct  bearish_pct  neutral_pct  avg_score  overall
NVDA    100           60.0         18.0         22.0         +0.420     bullish
MSFT    80            45.0         30.0         25.0         +0.150     bullish
TSLA    120           55.0         25.0         20.0         +0.300     bullish
AMZN    70            50.0         30.0         20.0         +0.200     bullish
META    90            40.0         35.0         25.0         +0.050     neutral
```

---

## Agent 3 — RAG Chatbot

### Example Conversation

**User:** Which ticker had the most insider buying today?

**Bot:** Based on the SEC filing data from the last 24 hours, NVIDIA (NVDA) had the highest insider trading value at $45,000,000 across 3 filings. This was followed by Microsoft (MSFT) at $32,000,000.

---

**User:** What is the market sentiment for Tesla?

**Bot:** Based on the last 7 days of X (Twitter) data for $TSLA:
- **Overall Sentiment: BULLISH** (score: +0.30)
- 55% of tweets were bullish
- 25% were bearish
- 20% were neutral
- 120 total tweets analyzed

---

**User:** Show me a sentiment chart

**Bot:** 📊 Chart generated! Chart saved to ./outputs/chart_sentiment.html

*(Opens interactive Plotly bar chart showing bullish/bearish/neutral breakdown per ticker)*

---

**User:** Which stock should I buy?

**Bot:** I can only provide information based on the SEC filing and sentiment data I have — I cannot give personalized investment advice. Based on the data, NVDA shows the highest insider buying ($45M) and the most bullish sentiment (+0.42 score), but please consult a financial advisor before making investment decisions.

---

## Agent 4 — Closed Learning Loop

### Self-Evaluation Output
```
╭─────────────────── Self-Evaluation Report ───────────────────╮
│ Quality Score: 8.5/10                                         │
│                                                               │
│ Reasoning: Strong data coverage across all 5 tickers.        │
│ Sentiment analysis captured 460 total tweets. Chatbot         │
│ answers were grounded in actual data with no hallucinations.  │
│                                                               │
│ ✓ Best Practices:                                             │
│   • Fallback data prevents pipeline failure                   │
│   • RAG chunking at 400 chars gives good retrieval precision  │
│   • Charts generated correctly on user request                │
│                                                               │
│ ✗ Issues Found:                                               │
│   • AMZN tweet volume (70) lower than others                  │
│   • LLM response time slightly slow on first query            │
│                                                               │
│ → Improvements:                                               │
│   • Increase AMZN tweet scraping to 200                       │
│   • Cache LLM embeddings for repeated queries                 │
╰───────────────────────────────────────────────────────────────╯

╭──────────────────── Next Run Plan ───────────────────────────╮
│ Priority Tickers: AMZN, META (need more data)                │
│ Data Fixes:                                                   │
│   • Increase tweet sample to 200/ticker                       │
│   • Add after-hours trading flag to SEC data                  │
│ Prompt Adjustments:                                           │
│   • Include sector context when answering ticker questions    │
╰───────────────────────────────────────────────────────────────╯

Learning Memory Summary
┌─────────────────────┬──────────┐
│ Metric              │ Value    │
├─────────────────────┼──────────┤
│ Total Runs          │ 3        │
│ Avg Quality Score   │ 7.83/10  │
│ Best Practices      │ 8        │
│ Known Issues        │ 4        │
└─────────────────────┴──────────┘
```

---

## Running the Pipeline

```bash
# Setup
cp .env.example .env
# Edit .env with your API keys

pip install -r requirements.txt

# Run full pipeline with interactive chat
python main.py

# Demo mode (no API keys needed, uses sample data)
python main.py --demo

# Pipeline only (no chat)
python main.py --no-chat
```
