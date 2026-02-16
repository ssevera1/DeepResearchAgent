"""Minimal configuration — all tunables in one place."""

# LLM
LLM_MODEL = "gpt-4o-mini"          # swap to any langchain-supported model
LLM_TEMPERATURE = 0.2

# Agent behaviour
MAX_WORKER_RETRIES = 3              # max times Reviewer can loop a sub-task back to Worker
MAX_PLAN_SUBTASKS = 5               # upper bound for Planner output

# Search
SEARCH_MAX_RESULTS = 5              # max results returned by search tool
