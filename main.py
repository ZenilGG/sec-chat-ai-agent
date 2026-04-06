"""
main.py — CrowdWisdomTrading SEC Chat AI Agent
===============================================
Orchestrates all 4 agents in sequence:

  Agent 1  →  Fetch SEC insider trades (last 24h)
  Agent 2  →  Scrape X tweets + sentiment (top 5 tickers, last 7d)
  Agent 3  →  Index into RAG + launch chatbot
  Agent 4  →  Closed learning loop (self-evaluate + improve)

Usage:
  python main.py                  # Full pipeline + interactive chat
  python main.py --no-chat        # Pipeline only (no interactive session)
  python main.py --demo           # Use sample data (no API keys needed)
"""

import argparse
import sys
import os
from pathlib import Path
from rich.console import Console
from rich.panel import Panel
from rich.rule import Rule

# Ensure project root is always in path regardless of where script is run from
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)  # also set working directory so relative paths (data/, outputs/) work

from agents.agent1_sec_fetcher import SECInsiderFetcher
from agents.agent2_sentiment import XSentimentAgent
from agents.agent3_chatbot import RAGIndex, ChatBot
from agents.agent4_learning_loop import ClosedLearningLoop

console = Console()


def print_banner():
    console.print(Panel.fit(
        "[bold cyan]CrowdWisdomTrading[/bold cyan]\n"
        "[white]SEC Chat AI Agent — Intern Assessment[/white]",
        border_style="cyan",
        padding=(1, 4),
    ))


def run_pipeline(demo_mode: bool = False, interactive: bool = True):
    """Run the full 4-agent pipeline."""
    print_banner()

    # ─────────────────────────────────────────────────────────────────────────
    # AGENT 1: SEC Insider Trading Data
    # ─────────────────────────────────────────────────────────────────────────
    console.print(Rule("[bold yellow]STEP 1/4 — SEC Data Fetch[/bold yellow]"))
    agent1 = SECInsiderFetcher()

    if demo_mode:
        console.log("[yellow]Demo mode: using sample SEC data")
        top5_df = agent1._fallback_sample_data()
    else:
        top5_df = agent1.fetch_top5_insider_trades()

    if top5_df.empty:
        console.print("[red]No SEC data available. Exiting.")
        return

    console.print(f"[green]✓ Agent 1 complete. Top tickers: {top5_df['ticker'].tolist()}")

    # ─────────────────────────────────────────────────────────────────────────
    # AGENT 2: X Sentiment Analysis
    # ─────────────────────────────────────────────────────────────────────────
    console.print(Rule("[bold yellow]STEP 2/4 — X Sentiment Analysis[/bold yellow]"))
    agent2 = XSentimentAgent()

    if demo_mode:
        # Use mock tweets in demo mode (skips Apify call)
        sentiment_data = {}
        for ticker in top5_df["ticker"].tolist():
            mock_tweets = agent2._mock_tweets(ticker)
            import pandas as pd
            analyzed = [agent2._analyze_tweet(t) for t in mock_tweets]
            df = pd.DataFrame(analyzed)
            df["ticker"] = ticker
            sentiment_data[ticker] = df
            agent2._display_sentiment_summary(ticker, df)
    else:
        sentiment_data = agent2.run(top5_df)

    sentiment_summary = agent2.get_combined_sentiment_summary(sentiment_data)
    console.print(f"[green]✓ Agent 2 complete. Analyzed {sum(len(v) for v in sentiment_data.values())} tweets.")

    # ─────────────────────────────────────────────────────────────────────────
    # AGENT 3: RAG Index + Chatbot
    # ─────────────────────────────────────────────────────────────────────────
    console.print(Rule("[bold yellow]STEP 3/4 — RAG Index & Chatbot[/bold yellow]"))
    rag = RAGIndex()
    rag.index_documents(top5_df, sentiment_data, sentiment_summary)
    bot = ChatBot(rag, sentiment_summary, top5_df)
    console.print("[green]✓ Agent 3 ready. RAG index built.")

    # Run a few sample Q&As to populate learning loop context
    sample_questions = [
        "Which ticker has the highest insider trading value?",
        "What is the overall market sentiment for the top tickers?",
        "Show me which companies had the most SEC filings today.",
    ]
    sample_qa = []
    console.print("\n[bold]Running sample Q&A for learning loop...[/bold]")
    for q in sample_questions:
        answer = bot.chat(q)
        sample_qa.append({"question": q, "answer": answer})
        console.print(f"  [cyan]Q:[/cyan] {q}")
        console.print(f"  [green]A:[/green] {answer[:150]}...\n")

    # ─────────────────────────────────────────────────────────────────────────
    # AGENT 4: Closed Learning Loop
    # ─────────────────────────────────────────────────────────────────────────
    console.print(Rule("[bold yellow]STEP 4/4 — Closed Learning Loop[/bold yellow]"))
    loop_agent = ClosedLearningLoop()
    loop_result = loop_agent.run_loop(top5_df, sentiment_summary, sample_qa)
    console.print("[green]✓ Agent 4 complete. Learning memory updated.")

    # ─────────────────────────────────────────────────────────────────────────
    # INTERACTIVE CHAT SESSION
    # ─────────────────────────────────────────────────────────────────────────
    if interactive:
        console.print(Rule("[bold green]INTERACTIVE CHAT SESSION[/bold green]"))
        bot.run_interactive()
    else:
        console.print(Panel(
            "[bold green]Pipeline completed successfully![/bold green]\n"
            f"Tickers analyzed: {', '.join(top5_df['ticker'].tolist())}\n"
            f"Tweets processed: {sum(len(v) for v in sentiment_data.values())}\n"
            f"RAG chunks: {rag.collection.count()}\n"
            f"Quality score: {loop_result['evaluation']['score']}/10",
            title="Summary",
            style="green",
        ))

    return {
        "top5_df": top5_df,
        "sentiment_data": sentiment_data,
        "sentiment_summary": sentiment_summary,
        "loop_result": loop_result,
        "bot": bot,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CrowdWisdomTrading SEC AI Agent")
    parser.add_argument("--no-chat", action="store_true", help="Skip interactive chat session")
    parser.add_argument("--demo", action="store_true", help="Use sample data (no API keys needed)")
    args = parser.parse_args()

    run_pipeline(
        demo_mode=args.demo,
        interactive=not args.no_chat,
    )