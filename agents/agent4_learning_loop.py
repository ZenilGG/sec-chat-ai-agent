"""
Agent 4: Closed Learning Loop (Hermes Built-in)
=================================================
Implements a closed learning loop using the Hermes agent framework.
The agent:
  1. Runs the full pipeline (Agents 1 → 2 → 3)
  2. Evaluates its own outputs for quality
  3. Stores feedback in a memory/log store
  4. Uses that feedback to improve future runs

Hermes: https://github.com/nousresearch/hermes-agent
"""

import os
import json
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import pandas as pd
from openai import OpenAI
from dotenv import load_dotenv
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

load_dotenv()
console = Console()

MEMORY_FILE = Path("./data/learning_memory.json")
MEMORY_FILE.parent.mkdir(parents=True, exist_ok=True)

LLM_MODEL = os.getenv("LLM_MODEL", "mistralai/mistral-7b-instruct:free")


def get_llm_client() -> OpenAI:
    return OpenAI(
        api_key=os.getenv("OPENROUTER_API_KEY"),
        base_url=os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"),
    )


# ── Memory Store ─────────────────────────────────────────────────────────────

class LearningMemory:
    """
    Persistent JSON-based memory store for the learning loop.
    Tracks: run history, quality scores, and improvement notes.
    """

    def __init__(self, path: Path = MEMORY_FILE):
        self.path = path
        self.data = self._load()

    def _load(self) -> dict:
        if self.path.exists():
            with open(self.path) as f:
                return json.load(f)
        return {"runs": [], "best_practices": [], "known_issues": []}

    def _save(self):
        with open(self.path, "w") as f:
            json.dump(self.data, f, indent=2, default=str)

    def add_run(self, run_data: dict):
        self.data["runs"].append(run_data)
        self._save()

    def add_best_practice(self, practice: str):
        if practice not in self.data["best_practices"]:
            self.data["best_practices"].append(practice)
            self._save()

    def add_known_issue(self, issue: str):
        if issue not in self.data["known_issues"]:
            self.data["known_issues"].append(issue)
            self._save()

    def get_context_for_next_run(self) -> str:
        """Build a context string from memory for the next run."""
        lines = ["LEARNING MEMORY FROM PREVIOUS RUNS:\n"]

        if self.data["best_practices"]:
            lines.append("Best practices discovered:")
            for bp in self.data["best_practices"][-5:]:  # last 5
                lines.append(f"  ✓ {bp}")

        if self.data["known_issues"]:
            lines.append("\nKnown issues to avoid:")
            for issue in self.data["known_issues"][-5:]:
                lines.append(f"  ✗ {issue}")

        recent_runs = self.data["runs"][-3:]
        if recent_runs:
            lines.append(f"\nRecent run scores: {[r['quality_score'] for r in recent_runs]}")

        return "\n".join(lines)

    def get_avg_quality_score(self) -> float:
        if not self.data["runs"]:
            return 0.0
        scores = [r.get("quality_score", 0) for r in self.data["runs"]]
        return sum(scores) / len(scores)

    def display_summary(self):
        table = Table(title="Learning Memory Summary", style="cyan")
        table.add_column("Metric", style="bold")
        table.add_column("Value", style="white")
        table.add_row("Total Runs", str(len(self.data["runs"])))
        table.add_row("Avg Quality Score", f"{self.get_avg_quality_score():.2f}/10")
        table.add_row("Best Practices", str(len(self.data["best_practices"])))
        table.add_row("Known Issues", str(len(self.data["known_issues"])))
        console.print(table)


# ── Hermes-style Tool Definitions ────────────────────────────────────────────

HERMES_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "evaluate_output_quality",
            "description": "Evaluate the quality of the pipeline's output on a scale of 1-10",
            "parameters": {
                "type": "object",
                "properties": {
                    "score": {
                        "type": "number",
                        "description": "Quality score from 1 (poor) to 10 (excellent)",
                    },
                    "reasoning": {
                        "type": "string",
                        "description": "Why this score was given",
                    },
                    "best_practices": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "What worked well and should be repeated",
                    },
                    "issues": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Problems found that should be avoided in future",
                    },
                    "improvement_suggestions": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Specific ways to improve the next run",
                    },
                },
                "required": ["score", "reasoning", "best_practices", "issues", "improvement_suggestions"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "plan_next_run",
            "description": "Plan improvements for the next pipeline execution",
            "parameters": {
                "type": "object",
                "properties": {
                    "priority_tickers": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Tickers to focus more attention on next run",
                    },
                    "data_quality_fixes": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Data quality improvements to apply",
                    },
                    "prompt_adjustments": {
                        "type": "string",
                        "description": "How to adjust prompts for better answers",
                    },
                },
                "required": ["priority_tickers", "data_quality_fixes", "prompt_adjustments"],
            },
        },
    },
]


# ── Closed Learning Loop Agent ────────────────────────────────────────────────

class ClosedLearningLoop:
    """
    Implements Hermes-style closed learning loop.
    
    Flow:
    ┌─────────────────────────────────────────┐
    │  Run Pipeline (Agents 1 → 2 → 3)        │
    │         ↓                               │
    │  Self-Evaluate Output Quality           │
    │         ↓                               │
    │  Extract Best Practices & Issues        │
    │         ↓                               │
    │  Store in Persistent Memory             │
    │         ↓                               │
    │  Plan Next Run with Improvements        │
    │         ↓                               │
    │  Feedback → Next Run Context            │
    └─────────────────────────────────────────┘
    """

    def __init__(self):
        self.llm = get_llm_client()
        self.memory = LearningMemory()

    def _build_evaluation_prompt(
        self,
        sec_df: pd.DataFrame,
        sentiment_summary: pd.DataFrame,
        sample_qa: list[dict],
    ) -> str:
        """Build the prompt for self-evaluation."""
        memory_context = self.memory.get_context_for_next_run()

        sec_summary = sec_df[["ticker", "total_value_usd", "filing_count"]].to_string(index=False)
        sent_summary = sentiment_summary[["ticker", "overall_sentiment", "avg_sentiment_score", "total_tweets"]].to_string(index=False)

        qa_text = "\n".join([
            f"Q: {qa['question']}\nA: {qa['answer']}" for qa in sample_qa
        ])

        return f"""You are evaluating the output of a financial AI pipeline that:
1. Fetched SEC insider trading data for the last 24 hours
2. Scraped and analyzed X (Twitter) sentiment for the top 5 tickers
3. Answered user questions via a RAG chatbot

{memory_context}

--- CURRENT RUN OUTPUT ---

SEC Data:
{sec_summary}

Sentiment Summary:
{sent_summary}

Sample Q&A from Chatbot:
{qa_text}

--- EVALUATION TASK ---
Evaluate the quality of this run. Use the evaluate_output_quality tool.
Consider:
- Data completeness and freshness
- Sentiment analysis accuracy
- Chatbot answer quality and relevance
- Any data gaps or issues
"""

    def _self_evaluate(
        self,
        sec_df: pd.DataFrame,
        sentiment_summary: pd.DataFrame,
        sample_qa: list[dict],
    ) -> dict:
        """Use LLM with tool calling to self-evaluate the pipeline output."""
        console.log("[cyan]Running self-evaluation...")
        prompt = self._build_evaluation_prompt(sec_df, sentiment_summary, sample_qa)

        try:
            response = self.llm.chat.completions.create(
                model=LLM_MODEL,
                messages=[{"role": "user", "content": prompt}],
                tools=HERMES_TOOLS,
                tool_choice={"type": "function", "function": {"name": "evaluate_output_quality"}},
                max_tokens=800,
            )

            tool_call = response.choices[0].message.tool_calls
            if tool_call:
                args = json.loads(tool_call[0].function.arguments)
                return args
        except Exception as e:
            console.log(f"[yellow]LLM evaluation failed: {e}. Using fallback scoring.")

        # Fallback: rule-based scoring
        score = 5.0
        if not sec_df.empty and len(sec_df) >= 5:
            score += 2.0
        if not sentiment_summary.empty and sentiment_summary["total_tweets"].sum() > 50:
            score += 2.0
        if sample_qa:
            score += 1.0

        return {
            "score": min(score, 10.0),
            "reasoning": "Fallback rule-based evaluation",
            "best_practices": ["Fetched data successfully", "Sentiment analysis completed"],
            "issues": ["LLM evaluation unavailable — check OPENROUTER_API_KEY"],
            "improvement_suggestions": ["Increase tweet sample size", "Add more data sources"],
        }

    def _plan_next_run(self, evaluation: dict, sec_df: pd.DataFrame) -> dict:
        """Use LLM to plan improvements for the next run."""
        console.log("[cyan]Planning next run improvements...")

        prompt = f"""Based on this evaluation of our SEC trading pipeline:
Score: {evaluation['score']}/10
Issues: {evaluation['issues']}
Suggestions: {evaluation['improvement_suggestions']}

Current tickers: {sec_df['ticker'].tolist() if not sec_df.empty else []}

Use the plan_next_run tool to create an improvement plan."""

        try:
            response = self.llm.chat.completions.create(
                model=LLM_MODEL,
                messages=[{"role": "user", "content": prompt}],
                tools=HERMES_TOOLS,
                tool_choice={"type": "function", "function": {"name": "plan_next_run"}},
                max_tokens=400,
            )
            tool_call = response.choices[0].message.tool_calls
            if tool_call:
                return json.loads(tool_call[0].function.arguments)
        except Exception as e:
            console.log(f"[yellow]Planning failed: {e}")

        return {
            "priority_tickers": sec_df["ticker"].tolist()[:3] if not sec_df.empty else [],
            "data_quality_fixes": ["Increase tweet sample to 200/ticker"],
            "prompt_adjustments": "Focus on providing more specific data-backed answers",
        }

    def run_loop(
        self,
        sec_df: pd.DataFrame,
        sentiment_summary: pd.DataFrame,
        sample_qa: Optional[list[dict]] = None,
    ) -> dict:
        """
        Execute one iteration of the closed learning loop.

        Args:
            sec_df: Output from Agent 1
            sentiment_summary: Output from Agent 2
            sample_qa: List of {question, answer} dicts from Agent 3 session

        Returns:
            dict with evaluation, next_run_plan, and memory context
        """
        console.rule("[bold cyan]Agent 4: Closed Learning Loop")

        if sample_qa is None:
            sample_qa = [
                {"question": "Which ticker has the most insider buying?", "answer": "Based on data..."},
                {"question": "What is the sentiment for NVDA?", "answer": "NVDA shows bullish sentiment..."},
            ]

        # ── Step 1: Self-evaluate ──
        evaluation = self._self_evaluate(sec_df, sentiment_summary, sample_qa)

        # ── Step 2: Store feedback ──
        for bp in evaluation.get("best_practices", []):
            self.memory.add_best_practice(bp)
        for issue in evaluation.get("issues", []):
            self.memory.add_known_issue(issue)

        # ── Step 3: Plan next run ──
        next_run_plan = self._plan_next_run(evaluation, sec_df)

        # ── Step 4: Record run ──
        run_record = {
            "run_id": str(uuid.uuid4())[:8],
            "timestamp": datetime.now().isoformat(),
            "quality_score": evaluation["score"],
            "tickers": sec_df["ticker"].tolist() if not sec_df.empty else [],
            "total_tweets": int(sentiment_summary["total_tweets"].sum()) if not sentiment_summary.empty else 0,
            "evaluation_summary": evaluation["reasoning"],
            "next_run_plan": next_run_plan,
        }
        self.memory.add_run(run_record)

        # ── Display results ──
        self._display_loop_results(evaluation, next_run_plan)
        self.memory.display_summary()

        return {
            "evaluation": evaluation,
            "next_run_plan": next_run_plan,
            "memory_context": self.memory.get_context_for_next_run(),
        }

    def _display_loop_results(self, evaluation: dict, plan: dict):
        """Pretty print the learning loop results."""
        score = evaluation["score"]
        color = "green" if score >= 7 else ("yellow" if score >= 4 else "red")

        console.print(Panel(
            f"[bold {color}]Quality Score: {score}/10[/bold {color}]\n\n"
            f"[white]{evaluation['reasoning']}[/white]\n\n"
            f"[bold]✓ Best Practices:[/bold]\n" +
            "\n".join(f"  • {bp}" for bp in evaluation.get("best_practices", [])) +
            f"\n\n[bold]✗ Issues Found:[/bold]\n" +
            "\n".join(f"  • {i}" for i in evaluation.get("issues", [])) +
            f"\n\n[bold]→ Improvements:[/bold]\n" +
            "\n".join(f"  • {s}" for s in evaluation.get("improvement_suggestions", [])),
            title="Self-Evaluation Report",
            style=color,
        ))

        console.print(Panel(
            f"[bold]Priority Tickers:[/bold] {', '.join(plan.get('priority_tickers', []))}\n\n"
            f"[bold]Data Fixes:[/bold]\n" +
            "\n".join(f"  • {f}" for f in plan.get("data_quality_fixes", [])) +
            f"\n\n[bold]Prompt Adjustments:[/bold]\n  {plan.get('prompt_adjustments', '')}",
            title="Next Run Plan",
            style="cyan",
        ))


# ── Standalone test ──────────────────────────────────────────────────────────
if __name__ == "__main__":
    sec_df = pd.DataFrame({
        "ticker": ["NVDA", "MSFT", "TSLA", "AMZN", "META"],
        "entity_name": ["NVIDIA", "Microsoft", "Tesla", "Amazon", "Meta"],
        "total_value_usd": [45e6, 32e6, 28e6, 21e6, 15e6],
        "filing_count": [3, 2, 4, 1, 2],
    })
    sentiment_summary = pd.DataFrame({
        "ticker": ["NVDA", "MSFT", "TSLA", "AMZN", "META"],
        "overall_sentiment": ["bullish", "bullish", "bullish", "neutral", "neutral"],
        "avg_sentiment_score": [0.4, 0.15, 0.3, 0.05, -0.05],
        "total_tweets": [100, 80, 120, 70, 90],
    })
    sample_qa = [
        {"question": "Which ticker has the most insider buying?", "answer": "NVDA with $45M"},
        {"question": "What is TSLA's sentiment?", "answer": "TSLA is bullish with 55% positive tweets"},
    ]

    loop = ClosedLearningLoop()
    result = loop.run_loop(sec_df, sentiment_summary, sample_qa)
    print("\nNext run context:\n", result["memory_context"])
