"""Minimal configuration — all tunables in one place."""

from __future__ import annotations

import logging
import os
from urllib.parse import urlparse

# LLM (Ollama — local open-source models)
LLM_MODEL = "llama3.2"             # any model available via `ollama list`
LLM_TEMPERATURE = 0.2
OLLAMA_BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")

# Warn if Ollama is configured over HTTP to a non-localhost host
_parsed = urlparse(OLLAMA_BASE_URL)
if _parsed.scheme == "http" and _parsed.hostname not in ("localhost", "127.0.0.1", "::1"):
    logging.getLogger(__name__).warning(
        "OLLAMA_BASE_URL uses HTTP with a non-localhost host (%s). "
        "Use HTTPS in production to protect data in transit.",
        _parsed.hostname,
    )

# Agent behaviour
MAX_WORKER_RETRIES = 3              # max times Reviewer can loop a sub-task back to Worker
MAX_PLAN_SUBTASKS = 5               # upper bound for Planner output

# Input validation
MAX_QUERY_LENGTH = 5000             # max characters for a user query

# Search
SEARCH_MAX_RESULTS = 5              # max results returned by search tool

# ── Validate configuration ────────────────────────────────────────────
if not (0.0 <= LLM_TEMPERATURE <= 2.0):
    raise ValueError(f"LLM_TEMPERATURE must be between 0.0 and 2.0, got {LLM_TEMPERATURE}")
if MAX_WORKER_RETRIES < 1:
    raise ValueError(f"MAX_WORKER_RETRIES must be >= 1, got {MAX_WORKER_RETRIES}")
if MAX_PLAN_SUBTASKS < 3:
    raise ValueError(f"MAX_PLAN_SUBTASKS must be >= 3, got {MAX_PLAN_SUBTASKS}")
