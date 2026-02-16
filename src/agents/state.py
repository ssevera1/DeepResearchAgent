"""
State schema for the Deep Research Agent.

LangGraph state is a TypedDict annotated with reducer functions.
Pydantic models are used *inside* the state for structured data.
"""

from __future__ import annotations

import operator
from dataclasses import dataclass, field
from typing import Annotated, TypedDict

from pydantic import BaseModel, Field


# ── Pydantic models for structured sub-objects ──────────────────────────

class SubTask(BaseModel):
    """A single research sub-task produced by the Planner."""
    id: int
    query: str
    completed: bool = False


class Finding(BaseModel):
    """A single research finding produced by the Worker."""
    subtask_id: int
    content: str
    approved: bool = False


# ── LangGraph state (TypedDict + reducers) ──────────────────────────────

class AgentState(TypedDict):
    """Top-level state that flows through every node in the graph."""

    # The original user query
    query: str

    # Plan: list of sub-tasks (replaced wholesale by the Planner)
    current_plan: list[SubTask]

    # Index of the sub-task currently being worked on
    current_subtask_idx: int

    # Accumulated research findings (append-only via operator.add reducer)
    research_findings: Annotated[list[Finding], operator.add]

    # How many times the current sub-task has been retried by the Worker
    worker_retries: int
