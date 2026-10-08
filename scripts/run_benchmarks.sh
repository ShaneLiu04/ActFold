#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

CONFIG_FILE="${1:-actfold/configs/default.yaml}"

echo "Running real-backend benchmarks with config: $CONFIG_FILE"
echo "Requires: pip install -r requirements-bench.txt"

python - <<PY
from pathlib import Path
import sys

from actfold.eval.benchmark_runner import BenchmarkRunner
from actfold.utils.config_manager import load_config

config_path = Path("$CONFIG_FILE")
if not config_path.exists():
    print(f"Config file not found: {config_path}", file=sys.stderr)
    sys.exit(1)

config = load_config(config_path)
runner = BenchmarkRunner(config)
results = runner.run(
    tasks=["gsm8k", "math", "ifeval", "humaneval_plus", "mbpp_plus"],
    num_samples=10,
    output_dir="results",
)
for task, metrics in results.items():
    print(f"\n=== {task} ===")
    for key, value in metrics.items():
        print(f"  {key}: {value}")
PY
