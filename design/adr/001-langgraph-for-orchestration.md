# ADR-001: Use LangGraph for Agent Orchestration

| Field   | Value            |
|---------|------------------|
| Status  | Accepted         |
| Date    | 2026-02-16       |
| Authors | —                |

## Context

We need an orchestration framework for a multi-step research agent that:

- Supports cyclic workflows (retry loops).
- Manages shared state across nodes.
- Integrates naturally with LangChain tool and LLM abstractions.

Alternatives considered:

| Option                | Pros                                        | Cons                                      |
|-----------------------|---------------------------------------------|-------------------------------------------|
| **LangGraph**         | First-class cycles, typed state, LangChain native | Relatively young ecosystem          |
| Raw LangChain agents  | Simple for linear chains                    | No native cycle support, manual state mgmt|
| Custom Python loops   | Full control                                | Boilerplate, no built-in checkpointing    |
| CrewAI / AutoGen      | High-level multi-agent patterns             | Heavier abstractions, harder to customise |

## Decision

Use **LangGraph** (`StateGraph`) as the orchestration backbone.

## Consequences

- The Planner → Worker → Reviewer cycle is expressed declaratively via `add_conditional_edges`.
- State is a typed `TypedDict` with reducer annotations — predictable and testable.
- Swapping in persistence (e.g. LangGraph checkpointing) is a one-line change.
- Team must learn LangGraph's state-reducer model.
