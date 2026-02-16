# ADR-004: Mock Search Tool as Default

| Field   | Value            |
|---------|------------------|
| Status  | Superseded by [ADR-005](005-tavily-search.md) |
| Date    | 2026-02-16       |
| Authors | —                |

## Context

The Worker node needs web search results to synthesise findings. Real search APIs (Tavily, SerpAPI, Google) require API keys and incur costs. During development and testing we need:

- Deterministic results for reproducible tests.
- Zero external dependencies for CI.
- A clear interface to swap in a real provider later.

## Decision

Ship a **`mock_search(query)`** function as the default search tool. It returns 3 deterministic dummy results seeded from the query hash.

```python
def mock_search(query: str) -> list[dict[str, str]]:
    seed = int(hashlib.md5(query.encode()).hexdigest()[:8], 16)
    ...
```

## Consequences

- Tests run without API keys — `pytest` works out of the box.
- The search interface is a simple `(str) -> list[dict]` contract — a real provider just needs to satisfy the same signature.
- Users must swap `mock_search` for a real implementation to get meaningful research results.
- The mock is clearly labelled in the module docstring to avoid confusion.
