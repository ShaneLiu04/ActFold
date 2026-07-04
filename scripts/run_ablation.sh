#!/usr/bin/env bash
set -euo pipefail

SYNTHETIC=false
CONFIG=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --synthetic)
            SYNTHETIC=true
            shift
            ;;
        -*)
            echo "Unknown option: $1" >&2
            echo "Usage: $0 [--synthetic | path/to/config.yaml]" >&2
            exit 1
            ;;
        *)
            CONFIG="$1"
            shift
            ;;
    esac
done

cd "$(dirname "$0")/.."

if $SYNTHETIC; then
    echo "Running ablation studies with a tiny synthetic model..."
    python - <<'PY'
import torch
import torch.nn as nn

from actfold.core import ActivationCache, SimilarityGate

torch.manual_seed(42)
torch.backends.cudnn.deterministic = True
from actfold.core.folding_scheduler import FoldingScheduler
from actfold.eval.ablation_study import AblationStudy
from actfold.speculative import ActFoldVerificationEngine, DraftGenerator, FastDLLMAdapter


class TinyModel(nn.Module):
    def __init__(self, vocab_size: int, hidden_dim: int, num_layers: int) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_dim)
        self.layers = nn.ModuleList(
            nn.TransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=max(1, hidden_dim // 64),
                dim_feedforward=hidden_dim * 4,
                batch_first=True,
            )
            for _ in range(num_layers)
        )
        self.head = nn.Linear(hidden_dim, vocab_size)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        x = self.embedding(tokens)
        for layer in self.layers:
            x = layer(x)
        return self.head(x)


vocab_size = 1000
hidden_dim = 128
num_layers = 4
model = FastDLLMAdapter(
    TinyModel(vocab_size, hidden_dim, num_layers),
    num_layers=num_layers,
    hidden_dim=hidden_dim,
)
draft_generator = DraftGenerator(vocab_size=vocab_size, mode="copy_flip", flip_ratio=0.05)

study = AblationStudy(
    model=model,
    baseline=None,
    draft_generator=draft_generator,
    vocab_size=vocab_size,
    seq_len=16,
    device="cpu",
)

results = study.run_all(output_dir="results")

print("\n=== Threshold Sensitivity ===")
print(results["threshold_sensitivity"])

print("\n=== Layer-wise Folding ===")
print(results["layerwise_folding"])

print("\n=== Cache Size Impact ===")
print(results["cache_size_impact"])
PY
elif [[ -n "$CONFIG" ]]; then
    echo "Running ablation studies with config: $CONFIG"
    python - <<PY
from actfold.eval.ablation_study import AblationStudy
from actfold.eval.benchmark_runner import BenchmarkRunner
from actfold.speculative import DraftGenerator, SpiffyBaseline
from actfold.utils.config_manager import load_config

config = load_config("$CONFIG")
runner = BenchmarkRunner(config)
model = runner.model
draft_generator = DraftGenerator(
    vocab_size=model.vocab_size,
    mode="copy_flip",
    flip_ratio=0.05,
)
baseline = SpiffyBaseline(model, draft_generator)
study = AblationStudy(
    model=model,
    baseline=baseline,
    draft_generator=draft_generator,
    vocab_size=model.vocab_size,
    seq_len=config.seq_len,
    device=runner.device,
)

results = study.run_all(output_dir="results")

print("\n=== Threshold Sensitivity ===")
print(results["threshold_sensitivity"])

print("\n=== Layer-wise Folding ===")
print(results["layerwise_folding"])

print("\n=== Cache Size Impact ===")
print(results["cache_size_impact"])
PY
else
    echo "Usage: $0 [--synthetic | path/to/config.yaml]" >&2
    exit 1
fi
