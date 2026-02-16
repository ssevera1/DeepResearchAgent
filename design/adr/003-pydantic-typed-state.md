# ADR-003: Pydantic Models Inside TypedDict State

| Field   | Value            |
|---------|------------------|
| Status  | Accepted         |
| Date    | 2026-02-16       |
| Authors | —                |

## Context

LangGraph state must be a `TypedDict` (for reducer annotations), but we also need structured, validated data objects for sub-tasks and findings.

Options:

| Option                         | Pros                              | Cons                                |
|--------------------------------|-----------------------------------|-------------------------------------|
| Plain dicts everywhere         | Zero overhead                     | No validation, typo-prone keys      |
| Dataclasses inside TypedDict   | Typed, lightweight                | No runtime validation               |
| **Pydantic inside TypedDict**  | Runtime validation, serialisation | Slight overhead                     |

## Decision

Use **Pydantic `BaseModel`** for `SubTask` and `Finding`, nested inside a `TypedDict` (`AgentState`) that carries LangGraph reducer annotations.

```python
class SubTask(BaseModel):
    id: int
    query: str
    completed: bool = False

class AgentState(TypedDict):
    current_plan: list[SubTask]
    research_findings: Annotated[list[Finding], operator.add]
    ...
```

## Consequences

- `operator.add` reducer on `research_findings` gives append-only semantics — nodes return `[finding]` and the framework accumulates.
- Pydantic validates data at construction time — catches schema drift early.
- Models serialise cleanly to JSON for logging and future checkpointing.
- Slight import cost (`pydantic>=2.0`), already a transitive dep of LangChain.
