#!/usr/bin/env bash
set -euo pipefail

echo "=== Deep Research Agent — Environment Setup ==="

# Prefer uv, fall back to pip
if command -v uv &>/dev/null; then
    echo "[*] Using uv"
    uv venv .venv
    echo "[*] Installing dependencies..."
    uv pip install -e ".[dev]"
else
    echo "[*] uv not found, falling back to python -m venv + pip"
    python -m venv .venv
    source .venv/bin/activate 2>/dev/null || .venv/Scripts/activate 2>/dev/null
    pip install --upgrade pip
    pip install -e ".[dev]"
fi

echo ""
echo "Done. Activate the environment with:"
echo "  source .venv/bin/activate   # Linux/macOS"
echo "  .venv\\Scripts\\activate      # Windows"
echo ""
echo "Then run:  python -m src.main \"Your research query\""
