#!/usr/bin/env bash
# One-command setup: creates .venv (Python 3.12), installs ask-why and its dependencies, installs the Chromium
# browser that Playwright drives, and creates .env from the template. Safe to run again.
set -euo pipefail
cd "$(dirname "$0")"

if command -v uv >/dev/null 2>&1; then
  echo "==> Creating .venv with uv (Python 3.12)"
  uv venv --python 3.12 --allow-existing .venv
  uv pip install --python .venv/bin/python -e ".[dev]"
else
  PY="$(command -v python3.12 || true)"
  if [ -z "$PY" ]; then
    echo "Python 3.12 not found. Install uv (https://docs.astral.sh/uv/getting-started/installation/), which downloads"
    echo "Python for you, or install Python 3.12, then run ./setup.sh again."
    exit 1
  fi
  echo "==> Creating .venv with $PY"
  "$PY" -m venv .venv
  .venv/bin/python -m pip install --quiet --upgrade pip
  .venv/bin/python -m pip install -e ".[dev]"
fi

echo "==> Installing the Chromium browser for Playwright"
.venv/bin/playwright install chromium

if [ ! -f .env ]; then
  cp .env.example .env
  echo "==> Created .env"
fi

cat <<'EOF'

Setup done. Three things left (see the README):
  1. Put your OpenRouter key in .env:            OPENROUTER_API_KEY=sk-or-...
  2. Activate the environment:                   source .venv/bin/activate
  3. Log in to Hugging Face (WorkArena access):  hf auth login
Then check everything with:                      python scripts/check_setup.py
EOF
