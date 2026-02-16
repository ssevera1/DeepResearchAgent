"""
Mock search tool — returns deterministic dummy results.

Swap this out for a real search API (Tavily, SerpAPI, etc.) later.
"""

from __future__ import annotations

import hashlib


def mock_search(query: str) -> list[dict[str, str]]:
    """Return 3 dummy search results seeded from the query string."""
    seed = int(hashlib.md5(query.encode()).hexdigest()[:8], 16)
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
