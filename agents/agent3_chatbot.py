"""
Agent 3: RAG Chatbot with Chart Generation
============================================
Indexes SEC + sentiment data into a ChromaDB vector store (RAG),
then exposes a chat interface that answers ONLY based on that data.
Can also generate charts on request using Plotly.
"""

import os
import json
import re
import hashlib
import textwrap
from pathlib import Path
from typing import Optional

import pandas as pd
import chromadb
from chromadb.utils import embedding_functions
from openai import OpenAI
from dotenv import load_dotenv
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel

load_dotenv()
console = Console()

# ── RAG Configuration ────────────────────────────────────────────────────────

CHROMA_PATH = os.getenv("CHROMA_DB_PATH", "./data/chroma_db")
COLLECTION_NAME = "sec_sentiment_data"
CHUNK_SIZE = 400          # characters per chunk
CHUNK_OVERLAP = 80        # overlap between chunks

# ── LLM Client (OpenRouter) ──────────────────────────────────────────────────

def get_llm_client() -> OpenAI:
    return OpenAI(
        api_key=os.getenv("OPENROUTER_API_KEY"),
        base_url=os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"),
    )

LLM_MODEL = os.getenv("LLM_MODEL", "mistralai/mistral-7b-instruct:free")


# ── Text Chunking ────────────────────────────────────────────────────────────

def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    """
    Split text into overlapping chunks.
    Strategy: sentence-aware splitting with character-level overlap.
    """
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    chunks, current = [], ""

    for sent in sentences:
        if len(current) + len(sent) <= chunk_size:
            current += " " + sent
        else:
            if current:
                chunks.append(current.strip())
            # start new chunk with overlap from previous
            overlap_text = current[-overlap:] if len(current) > overlap else current
            current = overlap_text + " " + sent

    if current.strip():
        chunks.append(current.strip())

    return chunks


# ── Data → Text Conversion ───────────────────────────────────────────────────

def sec_data_to_text(sec_df: pd.DataFrame) -> str:
    """Convert SEC insider trading DataFrame to readable text for indexing."""
    lines = ["SEC INSIDER TRADING REPORT - LAST 24 HOURS\n"]
    lines.append(f"Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M UTC')}\n")
    lines.append("Top 5 insider trades by total transaction value:\n")

    for _, row in sec_df.iterrows():
        lines.append(
            f"Ticker: {row['ticker']} | Company: {row['entity_name']} | "
            f"Total Value: ${row['total_value_usd']:,.0f} | "
            f"Filing Count: {row.get('filing_count', 'N/A')} | "
            f"Latest Filing: {row.get('latest_filing', 'N/A')}"
        )
    return "\n".join(lines)


def sentiment_data_to_text(ticker: str, tweet_df: pd.DataFrame, summary_row: Optional[pd.Series] = None) -> str:
    """Convert tweet sentiment DataFrame to readable text for indexing."""
    lines = [f"SENTIMENT ANALYSIS FOR ${ticker} - LAST 7 DAYS\n"]

    if summary_row is not None:
        lines.append(
            f"Overall Sentiment: {summary_row['overall_sentiment'].upper()} "
            f"(score: {summary_row['avg_sentiment_score']:+.3f})\n"
            f"Bullish: {summary_row['bullish_pct']}% | "
            f"Bearish: {summary_row['bearish_pct']}% | "
            f"Neutral: {summary_row['neutral_pct']}%\n"
            f"Total tweets analyzed: {summary_row['total_tweets']}\n"
        )

    lines.append("Sample tweets:\n")
    for _, row in tweet_df.head(20).iterrows():
        lines.append(
            f"[{row['sentiment'].upper()}] {row['text'][:200]} "
            f"(likes: {row.get('likes', 0)}, score: {row.get('sentiment_score', 0):+.2f})"
        )

    return "\n".join(lines)


# ── ChromaDB RAG ─────────────────────────────────────────────────────────────

class RAGIndex:
    """Manages ChromaDB vector store for SEC + sentiment data."""

    def __init__(self):
        Path(CHROMA_PATH).mkdir(parents=True, exist_ok=True)
        self.client = chromadb.PersistentClient(path=CHROMA_PATH)
        self.embed_fn = embedding_functions.DefaultEmbeddingFunction()
        self.collection = self.client.get_or_create_collection(
            name=COLLECTION_NAME,
            embedding_function=self.embed_fn,
            metadata={"hnsw:space": "cosine"},
        )

    def index_documents(
        self,
        sec_df: pd.DataFrame,
        sentiment_data: dict[str, pd.DataFrame],
        sentiment_summary: pd.DataFrame,
    ):
        """Index all SEC and sentiment data into ChromaDB."""
        console.log("[cyan]Indexing data into ChromaDB RAG store...")

        documents, ids, metadatas = [], [], []

        # ── Index SEC data ──
        sec_text = sec_data_to_text(sec_df)
        for i, chunk in enumerate(chunk_text(sec_text)):
            doc_id = f"sec_{hashlib.md5(chunk.encode()).hexdigest()[:8]}_{i}"
            documents.append(chunk)
            ids.append(doc_id)
            metadatas.append({"source": "SEC_EDGAR", "type": "insider_trading"})

        # ── Index sentiment data per ticker ──
        for ticker, tweet_df in sentiment_data.items():
            summary_row = None
            if not sentiment_summary.empty:
                match = sentiment_summary[sentiment_summary["ticker"] == ticker]
                if not match.empty:
                    summary_row = match.iloc[0]

            sent_text = sentiment_data_to_text(ticker, tweet_df, summary_row)
            for i, chunk in enumerate(chunk_text(sent_text)):
                doc_id = f"sentiment_{ticker}_{hashlib.md5(chunk.encode()).hexdigest()[:8]}_{i}"
                documents.append(chunk)
                ids.append(doc_id)
                metadatas.append({"source": "X_Twitter", "type": "sentiment", "ticker": ticker})

        # Upsert all at once
        if documents:
            self.collection.upsert(documents=documents, ids=ids, metadatas=metadatas)
            console.log(f"[green]Indexed {len(documents)} chunks into ChromaDB.")

    def query(self, question: str, n_results: int = 5) -> list[str]:
        """Retrieve top-k relevant chunks for a question."""
        results = self.collection.query(
            query_texts=[question],
            n_results=min(n_results, self.collection.count() or 1),
        )
        return results["documents"][0] if results["documents"] else []


# ── Chart Generation ─────────────────────────────────────────────────────────

def generate_chart(
    sentiment_summary: pd.DataFrame,
    sec_df: pd.DataFrame,
    chart_type: str = "sentiment",
    ticker: Optional[str] = None,
) -> str:
    """
    Generate a Plotly chart and save as HTML.

    chart_type options:
      'sentiment'   - bar chart of bullish/bearish/neutral % per ticker
      'value'       - bar chart of insider trade $ values
      'score'       - line/bar of average sentiment scores
    """
    try:
        import plotly.graph_objects as go
        import plotly.express as px
        from plotly.subplots import make_subplots
    except ImportError:
        return "Plotly not installed. Run: pip install plotly"

    output_dir = Path("./outputs")
    output_dir.mkdir(exist_ok=True)

    if chart_type == "sentiment" and not sentiment_summary.empty:
        fig = go.Figure()
        tickers = sentiment_summary["ticker"].tolist()

        fig.add_trace(go.Bar(name="Bullish %", x=tickers, y=sentiment_summary["bullish_pct"], marker_color="#00c853"))
        fig.add_trace(go.Bar(name="Neutral %", x=tickers, y=sentiment_summary["neutral_pct"], marker_color="#ffd600"))
        fig.add_trace(go.Bar(name="Bearish %", x=tickers, y=sentiment_summary["bearish_pct"], marker_color="#d50000"))

        fig.update_layout(
            barmode="stack",
            title="X (Twitter) Sentiment by Ticker — Last 7 Days",
            xaxis_title="Ticker",
            yaxis_title="Percentage (%)",
            template="plotly_dark",
            legend=dict(orientation="h", yanchor="bottom", y=1.02),
        )
        path = output_dir / "chart_sentiment.html"
        fig.write_html(str(path))
        return f"Chart saved to {path}"

    elif chart_type == "value" and not sec_df.empty:
        fig = px.bar(
            sec_df.sort_values("total_value_usd", ascending=True),
            x="total_value_usd",
            y="ticker",
            orientation="h",
            title="SEC Insider Trades — Total $ Value (Last 24h)",
            labels={"total_value_usd": "Total Value (USD)", "ticker": "Ticker"},
            color="total_value_usd",
            color_continuous_scale="Viridis",
            template="plotly_dark",
        )
        fig.update_xaxes(tickprefix="$", tickformat=",.0f")
        path = output_dir / "chart_insider_value.html"
        fig.write_html(str(path))
        return f"Chart saved to {path}"

    elif chart_type == "score" and not sentiment_summary.empty:
        colors = ["green" if s > 0 else "red" for s in sentiment_summary["avg_sentiment_score"]]
        fig = go.Figure(go.Bar(
            x=sentiment_summary["ticker"],
            y=sentiment_summary["avg_sentiment_score"],
            marker_color=colors,
            text=sentiment_summary["avg_sentiment_score"].round(3),
            textposition="outside",
        ))
        fig.update_layout(
            title="Average Sentiment Score by Ticker",
            xaxis_title="Ticker",
            yaxis_title="Sentiment Score (-1 bearish → +1 bullish)",
            template="plotly_dark",
            yaxis=dict(range=[-1.1, 1.1]),
        )
        path = output_dir / "chart_sentiment_score.html"
        fig.write_html(str(path))
        return f"Chart saved to {path}"

    return "No valid chart type or empty data."


# ── Chatbot ──────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are a financial AI assistant for CrowdWisdomTrading.
You ONLY answer questions based on the SEC insider trading data and X (Twitter) sentiment data provided in the context below.
Do NOT use any external knowledge or make up information.
If the context does not contain enough information to answer, say so clearly.
Be concise, factual, and helpful.

When the user asks for a chart or visualization, respond with: [CHART:<type>]
where <type> is one of: sentiment, value, score

Context:
{context}
"""


class ChatBot:
    """RAG-powered chatbot over SEC + sentiment data."""

    def __init__(self, rag_index: RAGIndex, sentiment_summary: pd.DataFrame, sec_df: pd.DataFrame):
        self.rag = rag_index
        self.sentiment_summary = sentiment_summary
        self.sec_df = sec_df
        self.llm = get_llm_client()
        self.history: list[dict] = []

    def _detect_chart_request(self, user_msg: str) -> Optional[str]:
        """Detect if user is asking for a chart."""
        msg_lower = user_msg.lower()
        if any(w in msg_lower for w in ["chart", "graph", "plot", "visual", "show me"]):
            if "sentiment" in msg_lower or "tweet" in msg_lower:
                return "sentiment"
            if "value" in msg_lower or "dollar" in msg_lower or "trade" in msg_lower:
                return "value"
            if "score" in msg_lower:
                return "score"
            return "sentiment"  # default chart
        return None

    def chat(self, user_message: str) -> str:
        """
        Process a user message and return a response.
        Handles both regular questions and chart requests.
        """
        # Check for chart request
        chart_type = self._detect_chart_request(user_message)
        if chart_type:
            result = generate_chart(self.sentiment_summary, self.sec_df, chart_type)
            return f"📊 Chart generated! {result}"

        # Retrieve relevant context via RAG
        context_chunks = self.rag.query(user_message, n_results=5)
        context = "\n\n---\n\n".join(context_chunks) if context_chunks else "No relevant data found."

        # Build prompt
        system = SYSTEM_PROMPT.format(context=context)
        self.history.append({"role": "user", "content": user_message})

        messages = [{"role": "system", "content": system}] + self.history[-10:]  # keep last 10 turns

        try:
            response = self.llm.chat.completions.create(
                model=LLM_MODEL,
                messages=messages,
                max_tokens=600,
                temperature=0.3,
            )
            reply = response.choices[0].message.content.strip()
        except Exception as e:
            reply = f"LLM error: {e}. (Check your OPENROUTER_API_KEY)"

        self.history.append({"role": "assistant", "content": reply})
        return reply

    def run_interactive(self):
        """Launch interactive terminal chat session."""
        console.rule("[bold green]Agent 3: SEC Chat Bot (RAG)")
        console.print(Panel(
            "[bold]Ask me anything about the SEC insider trades and market sentiment.\n"
            "Type [yellow]'chart'[/yellow] to see a sentiment chart, [yellow]'quit'[/yellow] to exit.[/bold]",
            title="CrowdWisdomTrading AI",
            style="green"
        ))

        while True:
            try:
                user_input = console.input("\n[bold cyan]You:[/bold cyan] ").strip()
            except (EOFError, KeyboardInterrupt):
                console.print("\n[yellow]Exiting chat...[/yellow]")
                break

            if not user_input:
                continue
            if user_input.lower() in ("quit", "exit", "bye"):
                console.print("[yellow]Goodbye![/yellow]")
                break

            with console.status("[bold green]Thinking..."):
                reply = self.chat(user_input)

            console.print(f"\n[bold green]Bot:[/bold green]")
            console.print(Markdown(reply))


# ── Standalone test ──────────────────────────────────────────────────────────
if __name__ == "__main__":
    # Mock data for standalone testing
    sec_df = pd.DataFrame({
        "ticker": ["NVDA", "MSFT", "TSLA", "AMZN", "META"],
        "entity_name": ["NVIDIA", "Microsoft", "Tesla", "Amazon", "Meta"],
        "total_value_usd": [45e6, 32e6, 28e6, 21e6, 15e6],
        "filing_count": [3, 2, 4, 1, 2],
        "latest_filing": ["2024-01-15"] * 5,
    })

    sentiment_data = {}  # Would come from Agent 2
    sentiment_summary = pd.DataFrame({
        "ticker": ["NVDA", "MSFT", "TSLA", "AMZN", "META"],
        "bullish_pct": [60, 45, 55, 50, 40],
        "bearish_pct": [20, 30, 25, 30, 35],
        "neutral_pct": [20, 25, 20, 20, 25],
        "avg_sentiment_score": [0.4, 0.15, 0.3, 0.2, 0.05],
        "overall_sentiment": ["bullish", "bullish", "bullish", "bullish", "neutral"],
        "total_tweets": [100, 80, 120, 70, 90],
    })

    rag = RAGIndex()
    rag.index_documents(sec_df, sentiment_data, sentiment_summary)

    bot = ChatBot(rag, sentiment_summary, sec_df)
    bot.run_interactive()
