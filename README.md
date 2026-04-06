# CrowdWisdomTrading — SEC Chat AI Agent

A backend Python AI agent system for SEC insider trading analysis and sentiment-driven chat.

## Architecture

```
main.py
├── Agent 1: SEC EDGAR Form 4 Fetcher     → top 5 tickers by $ value (last 24h)
├── Agent 2: X Sentiment Scraper           → Apify tweets + sentiment (last 7d)
├── Agent 3: RAG Chatbot                   → ChromaDB + OpenRouter LLM + charts
└── Agent 4: Closed Learning Loop          → Hermes-style self-eval + memory
```

## Quick Start

```bash
# 1. Clone and enter project
cd sec_agent

# 2. Install dependencies
pip install -r requirements.txt

# 3. Configure environment
cp .env.example .env
# Add your OPENROUTER_API_KEY and APIFY_API_TOKEN to .env

# 4. Run (demo mode — no API keys needed)
python main.py --demo

# 5. Run full pipeline
python main.py
```

## Tech Stack

| Component | Library |
|---|---|
| Agent Framework | Hermes Agent (NousResearch) |
| LLM Provider | OpenRouter (free models) |
| Tweet Scraping | Apify (apidojo/tweet-scraper) |
| Vector Store | ChromaDB |
| Charts | Plotly |
| SEC Data | SEC EDGAR API (free, no key) |

## Project Structure

```
sec_agent/
├── main.py                      # Pipeline orchestrator
├── requirements.txt
├── .env.example
├── SAMPLE_IO.md                 # Sample inputs/outputs
├── agents/
│   ├── agent1_sec_fetcher.py    # SEC insider trading data
│   ├── agent2_sentiment.py      # X tweet scraping + sentiment
│   ├── agent3_chatbot.py        # RAG chatbot + chart generation
│   └── agent4_learning_loop.py  # Closed learning loop
├── data/
│   ├── chroma_db/               # Vector store (auto-created)
│   └── learning_memory.json     # Agent 4 memory (auto-created)
└── outputs/
    └── *.html                   # Generated Plotly charts
```

## Data Sources

- [SEC EDGAR](https://github.com/stefanoamorelli/sec-edgar-agentkit)
- [TradeSignal](https://github.com/skadri1601/TradeSignal)
- [Insider Trading Analyzer](https://github.com/wescules/insider-trading-analyzer)
- [SEC API Python](https://github.com/janlukasschroeder/sec-api-python)
