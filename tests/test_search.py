"""Tests for the search tool — mock, Tavily, and dispatcher."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from src.tools.search import (
    _TAVILY_MAX_RETRIES,
    mock_search,
    search,
    tavily_search,
)


# ── mock_search tests (unchanged) ───────────────────────────────────────


def test_mock_search_returns_three_results():
    results = mock_search("test query")
    assert len(results) == 3


def test_mock_search_results_have_required_keys():
    results = mock_search("test query")
    for r in results:
        assert "title" in r
        assert "snippet" in r
        assert "url" in r


def test_mock_search_deterministic():
    a = mock_search("same query")
    b = mock_search("same query")
    assert a == b


def test_mock_search_varies_by_query():
    a = mock_search("query one")
    b = mock_search("query two")
    assert a != b


# ── tavily_search tests ─────────────────────────────────────────────────


@patch("tavily.TavilyClient")
def test_tavily_search_maps_fields(mock_client_cls):
    """Verify Tavily response fields are normalised to our contract."""
    mock_client = MagicMock()
    mock_client_cls.return_value = mock_client
    mock_client.search.return_value = {
        "results": [
            {"title": "T1", "content": "Snippet 1", "url": "https://a.com"},
            {"title": "T2", "content": "Snippet 2", "url": "https://b.com"},
        ]
    }

    with patch.dict("os.environ", {"TAVILY_API_KEY": "test-key"}):
        results = tavily_search("test query", max_results=2)

    assert len(results) == 2
    assert results[0] == {"title": "T1", "snippet": "Snippet 1", "url": "https://a.com"}
    assert results[1] == {"title": "T2", "snippet": "Snippet 2", "url": "https://b.com"}
    mock_client.search.assert_called_once_with(query="test query", max_results=2, timeout=10)


@patch("tavily.TavilyClient")
def test_tavily_search_passes_api_key(mock_client_cls):
    """Verify the Tavily client receives the API key from the environment."""
    mock_client = MagicMock()
    mock_client_cls.return_value = mock_client
    mock_client.search.return_value = {"results": []}

    with patch.dict("os.environ", {"TAVILY_API_KEY": "my-secret"}):
        tavily_search("q")

    mock_client_cls.assert_called_once_with(api_key="my-secret")


# ── search dispatcher tests ─────────────────────────────────────────────


@patch("src.tools.search.tavily_search")
def test_search_uses_tavily_when_key_set(mock_tavily):
    mock_tavily.return_value = [{"title": "T", "snippet": "S", "url": "U"}]
    with patch.dict("os.environ", {"TAVILY_API_KEY": "key"}):
        results = search("q")
    mock_tavily.assert_called_once_with("q")
    assert results == [{"title": "T", "snippet": "S", "url": "U"}]


@patch("src.tools.search.mock_search")
def test_search_falls_back_to_mock_without_key(mock_mock):
    mock_mock.return_value = [{"title": "M", "snippet": "S", "url": "U"}]
    with patch.dict("os.environ", {}, clear=True):
        results = search("q")
    mock_mock.assert_called_once_with("q")
    assert results == [{"title": "M", "snippet": "S", "url": "U"}]


# ── Security: missing API key ─────────────────────────────────────────


def test_tavily_search_raises_on_missing_api_key():
    """Missing TAVILY_API_KEY should raise RuntimeError, not KeyError."""
    with patch.dict("os.environ", {}, clear=True):
        with pytest.raises(RuntimeError, match="TAVILY_API_KEY"):
            tavily_search("q")


# ── Security: malformed Tavily response ────────────────────────────────


@patch("tavily.TavilyClient")
def test_tavily_search_handles_missing_results_key(mock_client_cls):
    """Response without 'results' key should return empty list."""
    mock_client = MagicMock()
    mock_client_cls.return_value = mock_client
    mock_client.search.return_value = {"error": "something went wrong"}

    with patch.dict("os.environ", {"TAVILY_API_KEY": "key"}):
        results = tavily_search("q")

    assert results == []


@patch("tavily.TavilyClient")
def test_tavily_search_skips_malformed_results(mock_client_cls):
    """Results missing required keys should be skipped, not crash."""
    mock_client = MagicMock()
    mock_client_cls.return_value = mock_client
    mock_client.search.return_value = {
        "results": [
            {"title": "Good", "content": "Snippet", "url": "https://a.com"},
            {"title": "Bad"},  # missing content and url
        ]
    }

    with patch.dict("os.environ", {"TAVILY_API_KEY": "key"}):
        results = tavily_search("q")

    assert len(results) == 1
    assert results[0]["title"] == "Good"


@patch("src.tools.search.time.sleep")
@patch("tavily.TavilyClient")
def test_tavily_search_returns_empty_on_api_error(mock_client_cls, _mock_sleep):
    """API exceptions should be caught, returning empty list."""
    mock_client = MagicMock()
    mock_client_cls.return_value = mock_client
    mock_client.search.side_effect = Exception("network error")

    with patch.dict("os.environ", {"TAVILY_API_KEY": "key"}):
        results = tavily_search("q")

    assert results == []


# ── Edge case: None values in Tavily results ───────────────────────────


@patch("tavily.TavilyClient")
def test_tavily_search_skips_results_with_none_values(mock_client_cls):
    """Results with None field values should be skipped."""
    mock_client = MagicMock()
    mock_client_cls.return_value = mock_client
    mock_client.search.return_value = {
        "results": [
            {"title": None, "content": "Snippet", "url": "https://a.com"},
            {"title": "Good", "content": "Snippet", "url": "https://b.com"},
            {"title": "Also bad", "content": "", "url": "https://c.com"},
        ]
    }

    with patch.dict("os.environ", {"TAVILY_API_KEY": "key"}):
        results = tavily_search("q")

    assert len(results) == 1
    assert results[0]["title"] == "Good"


# ── Retry classification ─────────────────────────────────────────────────


@patch("src.tools.search.time.sleep")
@patch("tavily.TavilyClient")
def test_tavily_search_retries_transient_failure_then_succeeds(
    mock_client_cls, mock_sleep
):
    mock_client = MagicMock()
    mock_client_cls.return_value = mock_client
    mock_client.search.side_effect = [
        ConnectionError("connection reset"),
        {"results": [{"title": "T", "content": "S", "url": "https://a.com"}]},
    ]

    with patch.dict("os.environ", {"TAVILY_API_KEY": "key"}):
        results = tavily_search("q")

    assert len(results) == 1
    assert mock_client.search.call_count == 2
    assert mock_sleep.call_count == 1


@patch("src.tools.search.time.sleep")
@patch("tavily.TavilyClient")
def test_tavily_search_exhausts_retries_on_persistent_transient_failure(
    mock_client_cls, mock_sleep
):
    mock_client = MagicMock()
    mock_client_cls.return_value = mock_client
    mock_client.search.side_effect = ConnectionError("connection reset")

    with patch.dict("os.environ", {"TAVILY_API_KEY": "key"}):
        results = tavily_search("q")

    assert results == []
    assert mock_client.search.call_count == _TAVILY_MAX_RETRIES
    # Slept between attempts, but not after the final one.
    assert mock_sleep.call_count == _TAVILY_MAX_RETRIES - 1


@pytest.mark.parametrize(
    "error_name",
    [
        "InvalidAPIKeyError",
        "MissingAPIKeyError",
        "UsageLimitExceededError",
        "BadRequestError",
        "ForbiddenError",
    ],
)
@patch("src.tools.search.time.sleep")
@patch("tavily.TavilyClient")
def test_tavily_search_does_not_retry_permanent_errors(
    mock_client_cls, mock_sleep, error_name
):
    """A bad key or exhausted quota must fail on the first call, not burn 3."""
    import inspect

    import tavily.errors

    error_cls = getattr(tavily.errors, error_name)
    # MissingAPIKeyError takes no message; the rest require one.
    takes_message = "message" in inspect.signature(error_cls.__init__).parameters

    mock_client = MagicMock()
    mock_client_cls.return_value = mock_client
    mock_client.search.side_effect = (
        error_cls("rejected") if takes_message else error_cls()
    )

    with patch.dict("os.environ", {"TAVILY_API_KEY": "key"}):
        results = tavily_search("q")

    assert results == []
    assert mock_client.search.call_count == 1
    assert mock_sleep.call_count == 0


@patch("src.tools.search.time.sleep")
@patch("tavily.TavilyClient")
def test_tavily_search_backoff_is_exponential(mock_client_cls, mock_sleep):
    mock_client = MagicMock()
    mock_client_cls.return_value = mock_client
    mock_client.search.side_effect = ConnectionError("boom")

    with patch.dict("os.environ", {"TAVILY_API_KEY": "key"}):
        tavily_search("q")

    delays = [call.args[0] for call in mock_sleep.call_args_list]
    assert len(delays) == 2
    assert 1.0 <= delays[0] < 1.5   # 1 * 2**0 + jitter
    assert 2.0 <= delays[1] < 2.5   # 1 * 2**1 + jitter
