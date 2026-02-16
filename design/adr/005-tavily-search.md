# ADR-005: Tavily Web Search Integration

| Field   | Value            |
|---------|------------------|
| Status  | Accepted         |
| Date    | 2026-02-16       |
| Authors | —                |

## Context

The Worker node relied on `mock_search()` (ADR-004), which returned deterministic dummy results. To produce meaningful research output the agent needs real web search results.

We evaluated two integration paths:

1. **`tavily-python`** — lightweight client, direct API access.
2. **`langchain-tavily`** — LangChain wrapper, heavier dependency tree.

## Decision

Use **`tavily-python`** directly. The search module (`src/tools/search.py`) exposes a `search()` dispatcher that:

- Calls `tavily_search()` when `TAVILY_API_KEY` is set in the environment.
- Falls back to `mock_search()` otherwise (preserving offline / CI behaviour).

The Tavily response is normalised to the existing `list[dict[str, str]]` contract (keys: `title`, `snippet`, `url`), so no downstream code changes are needed beyond the import.

## Consequences

- **Real search results** when `TAVILY_API_KEY` is configured.
- **Zero-config fallback**: tests and offline runs still work via mock — `pytest` needs no API key.
- **Minimal new dependency**: `tavily-python` is a thin HTTP client with no transitive bloat.
- **Lazy import**: `TavilyClient` is imported inside `tavily_search()` so the module loads even when the package is absent (e.g. in minimal test environments).
- Supersedes ADR-004 (mock search is retained but is no longer the primary path).
