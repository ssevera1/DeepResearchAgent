"""
Deep Research Agent — LangGraph cyclic state graph.

Graph topology:
    START ──► Planner ──► Worker ──► Reviewer ──►─┐
                           ▲                      │
                           │  (answer insufficient │
                           │   & retries left)     │
                           └──────────────────────┘
                                                   │
                              (answer sufficient   │
                               OR retries exhausted│
                               & more sub-tasks)   │
                                                   ▼
                                           Worker (next sub-task)
                                               ...
                                                   │
                              (all sub-tasks done) │
                                                   ▼
                                                  END
"""

from __future__ import annotations

import json
import os
from typing import Literal

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph

from config.settings import LLM_MODEL, LLM_TEMPERATURE, MAX_PLAN_SUBTASKS, MAX_WORKER_RETRIES
from src.agents.state import AgentState, Finding, SubTask
from src.tools.search import mock_search


# ── LLM singleton ──────────────────────────────────────────────────────

def _get_llm() -> ChatOpenAI:
    return ChatOpenAI(model=LLM_MODEL, temperature=LLM_TEMPERATURE)


# ── Node: Planner ───────────────────────────────────────────────────────

def planner_node(state: AgentState) -> dict:
    """Break the user query into 3-5 concrete research sub-tasks."""
    llm = _get_llm()
    response = llm.invoke([
        SystemMessage(content=(
            "You are a research planner. Given a user query, decompose it into "
            f"3 to {MAX_PLAN_SUBTASKS} independent, concrete sub-tasks that "
            "together would fully answer the query.\n\n"
            "Respond ONLY with a JSON array of strings — each string is one sub-task.\n"
            "Example: [\"Find statistics on X\", \"Compare Y and Z\", \"Summarise expert opinions on W\"]"
        )),
        HumanMessage(content=state["query"]),
    ])

    try:
        raw = json.loads(response.content)
    except json.JSONDecodeError:
        # Fallback: treat the whole response as a single task
        raw = [response.content.strip()]

    subtasks = [
        SubTask(id=i, query=q)
        for i, q in enumerate(raw[:MAX_PLAN_SUBTASKS])
    ]

    return {
        "current_plan": subtasks,
        "current_subtask_idx": 0,
        "worker_retries": 0,
    }


# ── Node: Worker ────────────────────────────────────────────────────────

def worker_node(state: AgentState) -> dict:
    """Execute the current sub-task using the mock search tool."""
    idx = state["current_subtask_idx"]
    subtask = state["current_plan"][idx]

    results = mock_search(subtask.query)
    combined = "\n\n".join(
        f"**{r['title']}**\n{r['snippet']}" for r in results
    )

    llm = _get_llm()
    response = llm.invoke([
        SystemMessage(content=(
            "You are a research worker. You have been given search results for "
            "a specific sub-task. Synthesise the results into a concise finding "
            "(2-4 sentences) that directly answers the sub-task."
        )),
        HumanMessage(content=(
            f"Sub-task: {subtask.query}\n\n"
            f"Search results:\n{combined}"
        )),
    ])

    finding = Finding(
        subtask_id=subtask.id,
        content=response.content.strip(),
    )

    # research_findings uses operator.add reducer — wrap in a list
    return {"research_findings": [finding]}


# ── Node: Reviewer ──────────────────────────────────────────────────────

def reviewer_node(state: AgentState) -> dict:
    """Decide whether the latest finding adequately answers its sub-task."""
    idx = state["current_subtask_idx"]
    subtask = state["current_plan"][idx]
    latest_finding = state["research_findings"][-1]

    llm = _get_llm()
    response = llm.invoke([
        SystemMessage(content=(
            "You are a research reviewer. Evaluate whether the finding below "
            "adequately answers the sub-task. Respond with ONLY 'Yes' or 'No'."
        )),
        HumanMessage(content=(
            f"Sub-task: {subtask.query}\n\n"
            f"Finding: {latest_finding.content}"
        )),
    ])

    approved = response.content.strip().lower().startswith("yes")

    if approved:
        # Mark the finding and sub-task as approved / completed
        latest_finding.approved = True
        subtask.completed = True
        return {
            "current_subtask_idx": idx + 1,
            "worker_retries": 0,
        }

    # Not approved — bump retry counter
    return {"worker_retries": state["worker_retries"] + 1}


# ── Conditional edge after Reviewer ─────────────────────────────────────

def after_review(state: AgentState) -> Literal["worker", "__end__"]:
    """Route after Reviewer:
    - If the latest finding was NOT approved and retries remain → back to Worker
    - If approved and more sub-tasks remain → Worker (next sub-task)
    - If all sub-tasks done (or retries exhausted on last) → END
    """
    idx = state["current_subtask_idx"]
    plan = state["current_plan"]
    retries = state["worker_retries"]

    latest_finding = state["research_findings"][-1]

    if not latest_finding.approved and retries < MAX_WORKER_RETRIES:
        # Retry the same sub-task
        return "worker"

    if idx < len(plan):
        # Move to the next sub-task
        return "worker"

    # All done
    return END


# ── Graph assembly ──────────────────────────────────────────────────────

def build_graph() -> StateGraph:
    """Construct and compile the Deep Research Agent graph."""
    graph = StateGraph(AgentState)

    # Register nodes
    graph.add_node("planner", planner_node)
    graph.add_node("worker", worker_node)
    graph.add_node("reviewer", reviewer_node)

    # Edges
    graph.set_entry_point("planner")
    graph.add_edge("planner", "worker")
    graph.add_edge("worker", "reviewer")

    # Conditional: Reviewer decides what happens next
    graph.add_conditional_edges("reviewer", after_review)

    return graph.compile()
