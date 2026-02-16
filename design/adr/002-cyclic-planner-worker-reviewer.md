# ADR-002: Cyclic Planner → Worker → Reviewer Pattern

| Field   | Value            |
|---------|------------------|
| Status  | Accepted         |
| Date    | 2026-02-16       |
| Authors | —                |

## Context

A single LLM call cannot reliably perform deep research. We need a multi-step pattern that:

1. Decomposes a broad question into focused sub-tasks.
2. Researches each sub-task individually.
3. Quality-gates each finding before accepting it.
4. Retries when quality is insufficient.

## Decision

Adopt a **three-node cyclic graph**: Planner → Worker → Reviewer.

- **Planner** — runs once at the start; produces 3-5 `SubTask` objects.
- **Worker** — executes the current sub-task (search + LLM synthesis).
- **Reviewer** — LLM judge that approves or rejects the finding.
- **Conditional edge** (`after_review`) routes back to Worker on rejection (up to `MAX_WORKER_RETRIES`) or advances to the next sub-task.

```
START → Planner → Worker → Reviewer ──┐
                    ▲                  │ (retry)
                    └──────────────────┘
                                       │ (next / END)
                                       ▼
```

## Consequences

- Each sub-task gets independent quality review — reduces hallucination risk.
- The retry cap (`MAX_WORKER_RETRIES = 3`) prevents infinite loops.
- Worker and Reviewer are stateless pure functions — easy to unit-test with mocked LLMs.
- Adding more node types (e.g., Summariser, Citation Checker) later requires only adding nodes and edges.
