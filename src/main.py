"""
CLI entry point for the Deep Research Agent.

Usage:
    python -m src.main "What are the economic impacts of climate change?"
"""

from __future__ import annotations

import argparse
import sys

from src.agents.graph import build_graph


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Deep Research Agent — powered by LangGraph",
    )
    parser.add_argument(
        "query",
        type=str,
        help="The research question to investigate.",
    )
    args = parser.parse_args()

    print(f"\n{'='*60}")
    print(f"  Deep Research Agent")
    print(f"{'='*60}")
    print(f"  Query: {args.query}\n")

    graph = build_graph()

    # Kick off the graph with initial state
    initial_state = {
        "query": args.query,
        "current_plan": [],
        "current_subtask_idx": 0,
        "research_findings": [],
        "worker_retries": 0,
    }

    final_state = graph.invoke(initial_state)

    # ── Pretty-print results ────────────────────────────────────────
    print(f"\n{'─'*60}")
    print("  PLAN")
    print(f"{'─'*60}")
    for st in final_state["current_plan"]:
        status = "done" if st.completed else "pending"
        print(f"  [{status}] {st.id}: {st.query}")

    print(f"\n{'─'*60}")
    print("  FINDINGS")
    print(f"{'─'*60}")
    for f in final_state["research_findings"]:
        marker = "approved" if f.approved else "rejected"
        print(f"\n  Sub-task {f.subtask_id} [{marker}]:")
        print(f"    {f.content}")

    print(f"\n{'='*60}\n")


if __name__ == "__main__":
    main()
