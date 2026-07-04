#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

# Example: run benchmarks against a real model from the Hugging Face Hub.
# Pass a config file as the first argument, or use the provided GPT-2 example.
CONFIG_FILE="${1:-actfold/configs/real_model_example.yaml}"

echo "Running real-model benchmark with config: $CONFIG_FILE"
echo "Note: the first run may download model weights from the Hugging Face Hub."
bash scripts/run_benchmarks.sh "$CONFIG_FILE"
