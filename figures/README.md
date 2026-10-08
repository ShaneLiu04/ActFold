# ActFold Figures

This directory contains the figures used in the README and documentation.

## Provenance

All figures should be generated from real benchmark artifacts:

```bash
# Run benchmarks and ablations
python -m actfold.eval.benchmark_runner --config actfold/configs/real_model_example.yaml
bash scripts/run_ablation.sh actfold/configs/real_model_example.yaml

# Generate figures from the artifacts
python scripts/generate_figures.py --results-dir results/
```

The committed PNG files are **examples** produced by:

```bash
python scripts/generate_figures.py --demo
```

They use a tiny synthetic model and are intended only for documentation
illustration.  Replace them with figures generated from real model benchmarks
for publication.

## Files

- `fig1_similarity_heatmap.png` — Layer-token similarity between parent and child branches.
- `fig2_pareto_frontier.png` — Speedup vs. accuracy trade-off across thresholds.
- `fig3_tflops_reduction.png` — TFLOPs reduction per model.
- `fig4_ablation_table.png` — Ablation study results.
