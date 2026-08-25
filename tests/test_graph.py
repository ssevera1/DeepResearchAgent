"""
Tests for the Deep Research Agent graph.

These tests mock the LLM so they run without an API key and are deterministic.
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import httpx
import pytest
from tenacity import wait_none

from src.agents.graph import (
    _extract_synthesis,
    _invoke_llm_with_retry,
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


@patch("src.agents.graph._get_llm")
def test_planner_handles_dict_response(mock_get_llm):
    """JSON dict (not array) should be coerced to a single-item list."""
    llm = MagicMock()
    llm.invoke.return_value = _mock_llm_response(
        json.dumps({"task": "research something"})
    )
    mock_get_llm.return_value = llm

    state = _make_state(current_plan=[])
    result = planner_node(state)

    assert len(result["current_plan"]) == 1


@patch("src.agents.graph._get_llm")
def test_planner_filters_non_string_items(mock_get_llm):
    """Non-string items in JSON array should be coerced or filtered."""
    llm = MagicMock()
    llm.invoke.return_value = _mock_llm_response(
        json.dumps(["Valid task", None, {"nested": True}, "Another task"])
    )
    mock_get_llm.return_value = llm

    state = _make_state(current_plan=[])
    result = planner_node(state)

    queries = [st.query for st in result["current_plan"]]
    assert "Valid task" in queries
    assert "Another task" in queries
    # None and dict should be filtered out
    assert len(result["current_plan"]) == 2


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


@patch("src.agents.graph._get_llm")
def test_reviewer_advances_on_retries_exhausted(mock_get_llm):
    """When retries are exhausted, reviewer should advance to next subtask."""
    llm = MagicMock()
    llm.invoke.return_value = _mock_llm_response("No")
    mock_get_llm.return_value = llm

    finding = Finding(subtask_id=0, content="Bad answer")
    state = _make_state(
        research_findings=[finding],
        worker_retries=2,  # Will become 3 = MAX_WORKER_RETRIES
    )
    result = reviewer_node(state)

    assert result["current_subtask_idx"] == 1
    assert result["worker_retries"] == 0


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
    """After reviewer advances past last subtask on exhaustion, END."""
    finding = Finding(subtask_id=1, content="bad", approved=False)
    state = _make_state(
        research_findings=[finding],
        worker_retries=0,         # Reset by reviewer after exhaustion
        current_subtask_idx=2,    # Past end — advanced by reviewer
    )
    assert after_review(state) == "__end__"


def test_routing_advances_past_exhausted_retries_on_non_last_subtask():
    """After reviewer exhausts retries on subtask 0, advance to subtask 1."""
    finding = Finding(subtask_id=0, content="bad", approved=False)
    state = _make_state(
        research_findings=[finding],
        worker_retries=0,         # Reset by reviewer after exhaustion
        current_subtask_idx=1,    # Advanced by reviewer
    )
    # idx=1 < len(plan)=2 → should continue to next subtask
    assert after_review(state) == "worker"


def test_routing_empty_findings_returns_worker():
    """Empty research_findings should route back to worker."""
    state = _make_state(research_findings=[])
    assert after_review(state) == "worker"


# ── Edge case: planner with whitespace/empty LLM response ────────────────


@patch("src.agents.graph._get_llm")
def test_planner_whitespace_only_response(mock_get_llm):
    """LLM returning only whitespace should produce a valid fallback subtask."""
    llm = MagicMock()
    llm.invoke.return_value = _mock_llm_response("   \n\n   ")
    mock_get_llm.return_value = llm

    state = _make_state(current_plan=[])
    result = planner_node(state)

    assert len(result["current_plan"]) == 1
    assert result["current_plan"][0].query  # non-empty


@patch("src.agents.graph._get_llm")
def test_planner_empty_string_response(mock_get_llm):
    """LLM returning empty string should produce a valid fallback subtask."""
    llm = MagicMock()
    llm.invoke.return_value = _mock_llm_response("")
    mock_get_llm.return_value = llm

    state = _make_state(current_plan=[])
    result = planner_node(state)

    assert len(result["current_plan"]) >= 1
    assert result["current_plan"][0].query  # non-empty


@patch("src.agents.graph._get_llm")
def test_planner_whitespace_items_filtered(mock_get_llm):
    """JSON array with whitespace-only strings should filter them out."""
    llm = MagicMock()
    llm.invoke.return_value = _mock_llm_response(
        json.dumps(["Valid task", "   ", "", "Another task"])
    )
    mock_get_llm.return_value = llm

    state = _make_state(current_plan=[])
    result = planner_node(state)

    queries = [st.query for st in result["current_plan"]]
    assert "Valid task" in queries
    assert "Another task" in queries
    assert len(result["current_plan"]) == 2


# ── Edge case: worker with empty search results ─────────────────────────


@patch("src.agents.graph.search", return_value=[])
@patch("src.agents.graph._get_llm")
def test_worker_handles_empty_search_results(mock_get_llm, _mock_search):
    """Worker should not crash when search returns no results."""
    llm = MagicMock()
    llm.invoke.return_value = _mock_llm_response("No results found.")
    mock_get_llm.return_value = llm

    state = _make_state()
    result = worker_node(state)

    assert len(result["research_findings"]) == 1
    assert result["research_findings"][0].content == "No results found."


# ── Edge case: single subtask plan ───────────────────────────────────────


def test_routing_end_after_single_subtask_approved():
    """Plan with 1 subtask: approval should END immediately."""
    finding = Finding(subtask_id=0, content="good", approved=True)
    state = _make_state(
        current_plan=[SubTask(id=0, query="Only task")],
        research_findings=[finding],
        worker_retries=0,
        current_subtask_idx=1,  # incremented by reviewer
    )
    assert after_review(state) == "__end__"


# ── Graph compilation ────────────────────────────────────────────────────

def test_graph_compiles():
    """Smoke test: the graph compiles without error."""
    graph = build_graph()
    assert graph is not None


# ── LLM retry tests ──────────────────────────────────────────────────────

def _no_wait():
    """The retry wrapper with backoff stripped, so tests don't pay wall-clock time."""
    return _invoke_llm_with_retry.retry_with(wait=wait_none())


def test_llm_retry_recovers_from_transient_transport_error():
    llm = MagicMock()
    llm.invoke.side_effect = [
        httpx.ConnectError("connection refused"),
        _mock_llm_response("recovered"),
    ]

    result = _no_wait()(llm, [])

    assert result.content == "recovered"
    assert llm.invoke.call_count == 2


def test_llm_retry_stops_after_three_attempts_and_reraises():
    llm = MagicMock()
    llm.invoke.side_effect = httpx.ReadTimeout("too slow")

    with pytest.raises(httpx.ReadTimeout):
        _no_wait()(llm, [])

    assert llm.invoke.call_count == 3


def test_llm_retry_does_not_retry_permanent_errors():
    """A deterministic failure must fail fast rather than burn the backoff budget."""
    llm = MagicMock()
    llm.invoke.side_effect = ValueError("model 'nope' not found")

    with pytest.raises(ValueError):
        _no_wait()(llm, [])

    assert llm.invoke.call_count == 1
# ── Worker synthesis extraction ──────────────────────────────────────────


@pytest.mark.parametrize(
    "reply,expected",
    [
        ('{"synthesis": "a finding"}', "a finding"),
        ('```json\n{"synthesis": "a finding"}\n```', "a finding"),
        ('```\n{"synthesis": "a finding"}\n```', "a finding"),
        ('Sure! Here is the JSON:\n{"synthesis": "a finding"}', "a finding"),
        ('{"synthesis": "braces { } inside"}', "braces { } inside"),
    ],
)
def test_extract_synthesis_accepts_wrapped_json(reply, expected):
    assert _extract_synthesis(reply) == expected


@pytest.mark.parametrize(
    "reply",
    [
        '{"result": "wrong key"}',
        '{"synthesis": "   "}',
        '{"synthesis": ["not", "a", "string"]}',
        '{"synthesis": {"nested": true}}',
        '{"synthesis": null}',
        "Just prose, no JSON at all.",
        "",
    ],
)
def test_extract_synthesis_rejects_unusable_replies(reply):
    assert _extract_synthesis(reply) is None


@patch("src.agents.graph.search")
@patch("src.agents.graph._get_llm")
def test_worker_extracts_synthesis_from_fenced_json(mock_get_llm, mock_search):
    """A fenced reply must not leak the raw ```json blob into the finding."""
    mock_search.return_value = [{"title": "T", "snippet": "S", "url": "https://a.com"}]
    llm = MagicMock()
    llm.invoke.return_value = _mock_llm_response(
        '```json\n{"synthesis": "The clean finding."}\n```'
    )
    mock_get_llm.return_value = llm

    result = worker_node(_make_state())

    assert result["research_findings"][0].content == "The clean finding."


@patch("src.agents.graph.search")
@patch("src.agents.graph._get_llm")
def test_worker_falls_back_to_raw_text_when_key_missing(mock_get_llm, mock_search):
    mock_search.return_value = [{"title": "T", "snippet": "S", "url": "https://a.com"}]
    llm = MagicMock()
    llm.invoke.return_value = _mock_llm_response('{"result": "wrong key"}')
    mock_get_llm.return_value = llm

    result = worker_node(_make_state())

    assert result["research_findings"][0].content == '{"result": "wrong key"}'


@patch("src.agents.graph.search")
@patch("src.agents.graph._get_llm")
def test_worker_prompt_keeps_empty_result_example_intact(mock_get_llm, mock_search):
    """The worker's empty-result example sentence is prompt surface; pin it.

    It was introduced in #8 and must not drift as a side effect of unrelated
    changes to this module.
    """
    mock_search.return_value = [{"title": "T", "snippet": "S", "url": "https://a.com"}]
    llm = MagicMock()
    llm.invoke.return_value = _mock_llm_response('{"synthesis": "A finding."}')
    mock_get_llm.return_value = llm

    worker_node(_make_state())

    system_prompt = llm.invoke.call_args[0][0][0].content
    assert (
        "If the search results are empty or unavailable, respond with a JSON object like "
        '{"synthesis": "[Unable to find information for this query]"}.\n\n'
    ) in system_prompt
