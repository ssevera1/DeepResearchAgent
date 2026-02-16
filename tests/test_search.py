"""Tests for the mock search tool."""

from src.tools.search import mock_search


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
