"""
Search tool — dispatches to Tavily (if API key is set) or mock fallback.

The public entry point is ``search(query)``.
"""

from __future__ import annotations

import hashlib
import logging
import os

from config.settings import SEARCH_MAX_RESULTS

logger = logging.getLogger(__name__)


def mock_search(query: str) -> list[dict[str, str]]:
    """Return 3 dummy search results seeded from the query string."""
    seed = int(hashlib.sha256(query.encode()).hexdigest()[:8], 16)
    results = []
    for i in range(1, 4):
        results.append(
            {
                "title": f"Result {i} for: {query}",
                "snippet": (
                    f"This is a simulated finding #{seed + i} that addresses "
                    f"aspects of '{query}'. It contains relevant information "
                    f"that a real search engine would return."
                ),
                "url": f"https://example.com/result/{seed + i}",
            }
        )
    return results


def tavily_search(query: str, max_results: int = SEARCH_MAX_RESULTS) -> list[dict[str, str]]:
    """Real web search via the Tavily API."""
    from tavily import TavilyClient

    api_key = os.environ.get("TAVILY_API_KEY")
    if not api_key:
        raise RuntimeError(
            "TAVILY_API_KEY environment variable is not set. "
            "Please set it in your .env file or environment."
        )

    client = TavilyClient(api_key=api_key)
    try:
        response = client.search(query=query, max_results=max_results)
    except Exception:
        logger.exception("Tavily API request failed")
        return []

    results = []
    for r in response.get("results", []):
        try:
            title = r["title"]
            snippet = r["content"]
            url = r["url"]
            if not all(isinstance(v, str) and v.strip() for v in (title, snippet, url)):
                raise ValueError("empty or non-string field")
            results.append({"title": title, "snippet": snippet, "url": url})
        except (KeyError, ValueError, TypeError):
            logger.warning("Skipping malformed Tavily result: %s", r)
    return results


def search(query: str) -> list[dict[str, str]]:
    """Dispatch: Tavily if TAVILY_API_KEY is set, otherwise mock."""
    if os.environ.get("TAVILY_API_KEY"):
        return tavily_search(query)
    return mock_search(query)
