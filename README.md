# Deep Research Agent

A cyclic deep-research agent built with [LangGraph](https://github.com/langchain-ai/langgraph). Given a complex question, it decomposes it into sub-tasks, researches each one, and self-reviews findings before accepting them.

## Architecture

```
START → Planner → Worker → Reviewer ──┐
                    ▲                  │  (retry if rejected, up to 3×)
                    └──────────────────┘
                                       ▼
                           Worker (next sub-task) … → END
```

| Node | Role |
|------|------|
| **Planner** | LLM decomposes the query into 3–5 independent sub-tasks |
| **Worker** | Searches for information and synthesises a concise finding |
| **Reviewer** | LLM judges whether the finding adequately answers the sub-task |

The Reviewer can loop a sub-task back to the Worker (up to `MAX_WORKER_RETRIES`) or advance to the next one. When all sub-tasks are complete the graph terminates and prints results.

## Quick Start

```bash
# 1. Clone
git clone https://github.com/ssevera1/DeepResearchAgent.git
cd DeepResearchAgent

# 2. Setup (creates .venv, installs deps)
./setup.sh

# 3. Activate
source .venv/bin/activate   # Linux/macOS
.venv\Scripts\activate      # Windows

# 4. Install Ollama and pull a model
# See https://ollama.com for install instructions
ollama pull llama3.2

# 5. Set your Tavily API key (for web search)
cp .env.example .env
# edit .env with your Tavily key

# 6. Run
python -m src.main "What are the economic impacts of climate change?"
```

## Testing

Tests mock the LLM — no API key required.

```bash
pytest
```

## Project Structure

```
src/
├── main.py              # CLI entry point
├── agents/
│   ├── graph.py         # LangGraph nodes, edges, build_graph()
│   └── state.py         # AgentState TypedDict + Pydantic models
└── tools/
    └── search.py        # Mock search tool (swap for Tavily/SerpAPI)
config/
    └── settings.py      # Tunables: model, temperature, retries
tests/
    ├── test_graph.py    # Node + routing tests
    └── test_search.py   # Search tool tests
design/
    ├── c4-context.md    # C4 Level 1 — System Context
    ├── c4-container.md  # C4 Level 2 — Containers
    ├── c4-component.md  # C4 Level 3 — Components
    └── adr/             # Architecture Decision Records
```

## Configuration

All tunables live in `config/settings.py`:

| Setting | Default | Description |
|---------|---------|-------------|
| `LLM_MODEL` | `llama3.2` | Any model available via `ollama list` |
| `LLM_TEMPERATURE` | `0.2` | LLM sampling temperature |
| `MAX_WORKER_RETRIES` | `3` | Max retries per sub-task |
| `MAX_PLAN_SUBTASKS` | `5` | Upper bound on planner output |

## Tech Stack

- **Python 3.11+**
- **LangGraph** — cyclic state-machine orchestration
- **LangChain** — LLM abstractions
- **Ollama** — llama3.2 locally (swappable to any Ollama model)
- **Pydantic v2** — structured state models
- **pytest** — testing with mocked LLMs

## License

MIT
