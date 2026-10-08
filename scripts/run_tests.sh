#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

echo "Running fast pytest suite (excludes @pytest.mark.slow)..."
python -m pytest tests/ -v -m "not slow"
echo "To run slow tests (real lm-eval/evalplus backends):"
echo "  python -m pytest tests/ -v -m slow"
