"""
Search tool — dispatches to Tavily (if API key is set) or mock fallback.

The public entry point is ``search(query)``.
"""

from __future__ import annotations

import hashlib
import logging
import os
import random
import time

from config.settings import SEARCH_MAX_RESULTS

logger = logging.getLogger(__name__)

# Attempts, not retries: the first call counts toward this budget.
_TAVILY_MAX_RETRIES = 3
_TAVILY_TIMEOUT_SECONDS = 10
_TAVILY_RETRY_DELAY_SECONDS = 1


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
    """Real web search via the Tavily API with retry and timeout logic."""
    from tavily import TavilyClient
    from tavily.errors import (
        BadRequestError,
        ForbiddenError,
        InvalidAPIKeyError,
        MissingAPIKeyError,
        UsageLimitExceededError,
    )

    api_key = os.environ.get("TAVILY_API_KEY")
    if not api_key:
        raise RuntimeError(
            "TAVILY_API_KEY environment variable is not set. "
            "Please set it in your .env file or environment."
        )

    client = TavilyClient(api_key=api_key)

    # Deterministic failures. A rejected key, an exhausted quota or a
    # malformed request cannot be fixed by trying again, so they must not
    # consume the retry budget or stall the agent loop on every search.
    permanent_errors = (
        InvalidAPIKeyError,
        MissingAPIKeyError,
        UsageLimitExceededError,
        BadRequestError,
        ForbiddenError,
    )

    for attempt in range(_TAVILY_MAX_RETRIES):
        try:
            response = client.search(
                query=query,
                max_results=max_results,
                timeout=_TAVILY_TIMEOUT_SECONDS,
            )
            break
        except permanent_errors:
            logger.exception("Tavily rejected the request; not retrying")
            return []
        except Exception:
            if attempt >= _TAVILY_MAX_RETRIES - 1:
                logger.exception(
                    "Tavily API request failed after %d attempts", _TAVILY_MAX_RETRIES
                )
                return []
            # Exponential backoff with jitter: three workers retrying in
            # lockstep is what keeps a struggling endpoint struggling.
            delay = _TAVILY_RETRY_DELAY_SECONDS * (2**attempt) + random.uniform(0, 0.5)
            logger.warning(
                "Tavily API request failed (attempt %d/%d), retrying in %.1fs",
                attempt + 1,
                _TAVILY_MAX_RETRIES,
                delay,
                exc_info=True,
            )
            time.sleep(delay)

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
