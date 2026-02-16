"""
Tests for the Deep Research Agent graph.

These tests mock the LLM so they run without an API key and are deterministic.
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

from src.agents.graph import (
    after_review,
    build_graph,
    planner_node,
    reviewer_node,
    worker_node,
)
from src.agents.state import AgentState, Finding, SubTask


# ── Helpers ──────────────────────────────────────────────────────────────

def _make_state(**overrides) -> AgentState:
    """Build a minimal valid AgentState with sensible defaults."""
    defaults: AgentState = {
        "query": "Test query",
        "current_plan": [
            SubTask(id=0, query="Sub-task 0"),
            SubTask(id=1, query="Sub-task 1"),
        ],
        "current_subtask_idx": 0,
        "research_findings": [],
        "worker_retries": 0,
    }
    defaults.update(overrides)
    return defaults


def _mock_llm_response(content: str) -> MagicMock:
    """Create a mock LLM response with the given content."""
    mock_resp = MagicMock()
    mock_resp.content = content
    return mock_resp


# ── Planner tests ────────────────────────────────────────────────────────

@patch("src.agents.graph._get_llm")
def test_planner_parses_json_array(mock_get_llm):
    llm = MagicMock()
    llm.invoke.return_value = _mock_llm_response(
        json.dumps(["Research topic A", "Compare B vs C", "Summarise D"])
    )
    mock_get_llm.return_value = llm

    state = _make_state(current_plan=[])
    result = planner_node(state)

    assert len(result["current_plan"]) == 3
    assert result["current_plan"][0].query == "Research topic A"
    assert result["current_subtask_idx"] == 0


@patch("src.agents.graph._get_llm")
def test_planner_fallback_on_bad_json(mock_get_llm):
    llm = MagicMock()
    llm.invoke.return_value = _mock_llm_response("Not valid JSON at all")
    mock_get_llm.return_value = llm

    state = _make_state(current_plan=[])
    result = planner_node(state)

    assert len(result["current_plan"]) == 1
    assert "Not valid JSON" in result["current_plan"][0].query


# ── Worker tests ─────────────────────────────────────────────────────────

@patch("src.agents.graph.search", return_value=[
    {"title": "T", "snippet": "S", "url": "https://example.com"},
])
@patch("src.agents.graph._get_llm")
def test_worker_produces_finding(mock_get_llm, _mock_search):
    llm = MagicMock()
    llm.invoke.return_value = _mock_llm_response("Synthesised answer here.")
    mock_get_llm.return_value = llm

    state = _make_state()
    result = worker_node(state)

    assert len(result["research_findings"]) == 1
    finding = result["research_findings"][0]
    assert finding.subtask_id == 0
    assert finding.content == "Synthesised answer here."
    assert finding.approved is False


# ── Reviewer tests ───────────────────────────────────────────────────────

@patch("src.agents.graph._get_llm")
def test_reviewer_approves(mock_get_llm):
    llm = MagicMock()
    llm.invoke.return_value = _mock_llm_response("Yes")
    mock_get_llm.return_value = llm

    finding = Finding(subtask_id=0, content="Good answer")
    state = _make_state(research_findings=[finding])
    result = reviewer_node(state)

    assert finding.approved is True
    assert result["current_subtask_idx"] == 1
    assert result["worker_retries"] == 0


@patch("src.agents.graph._get_llm")
def test_reviewer_rejects(mock_get_llm):
    llm = MagicMock()
    llm.invoke.return_value = _mock_llm_response("No")
    mock_get_llm.return_value = llm

    finding = Finding(subtask_id=0, content="Bad answer")
    state = _make_state(research_findings=[finding])
    result = reviewer_node(state)

    assert finding.approved is False
    assert result["worker_retries"] == 1


# ── Routing tests ────────────────────────────────────────────────────────

def test_routing_retry_when_rejected_and_retries_left():
    finding = Finding(subtask_id=0, content="bad", approved=False)
    state = _make_state(
        research_findings=[finding],
        worker_retries=1,
        current_subtask_idx=0,
    )
    assert after_review(state) == "worker"


def test_routing_next_subtask_when_approved():
    finding = Finding(subtask_id=0, content="good", approved=True)
    state = _make_state(
        research_findings=[finding],
        worker_retries=0,
        current_subtask_idx=1,  # incremented by reviewer
    )
    # idx=1 < len(plan)=2 → move to next sub-task
    assert after_review(state) == "worker"


def test_routing_end_when_all_subtasks_done():
    finding = Finding(subtask_id=1, content="good", approved=True)
    state = _make_state(
        research_findings=[finding],
        worker_retries=0,
        current_subtask_idx=2,  # past the last sub-task
    )
    assert after_review(state) == "__end__"


def test_routing_end_when_retries_exhausted_on_last_subtask():
    finding = Finding(subtask_id=1, content="bad", approved=False)
    state = _make_state(
        research_findings=[finding],
        worker_retries=3,         # MAX_WORKER_RETRIES = 3
        current_subtask_idx=2,    # past end after forced advance
    )
    assert after_review(state) == "__end__"


# ── Graph compilation ────────────────────────────────────────────────────

def test_graph_compiles():
    """Smoke test: the graph compiles without error."""
    graph = build_graph()
    assert graph is not None
