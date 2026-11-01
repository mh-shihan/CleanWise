"""Create a compact Markdown experiment report from generated artifacts."""
from __future__ import annotations
import argparse, json
from pathlib import Path
import pandas as pd

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--artifacts", default="artifacts")
    p.add_argument("--output", default="reports/generated/experiment_report.md")
    args = p.parse_args()
    a = Path(args.artifacts)
    metrics = json.loads((a/"evaluation_metrics.json").read_text())
    baselines = json.loads((a/"baseline_metrics.json").read_text())
    pred = pd.read_csv(a/"heldout_predictions.csv")
    text = f"""# CleanWise Experiment Report

## Scope
This is an automatically generated starter-benchmark report. It is a pipeline validation,
not evidence of general performance across real-world tabular datasets.

## Dataset-level holdout
- Training datasets: {metrics['n_train_datasets']}
- Held-out datasets: {metrics['n_test_datasets']}
- Training experiment rows: {metrics['n_train_rows']}
- Held-out experiment rows: {metrics['n_test_rows']}

## Meta-model results
- MAE: {metrics['mae']:.6f}
- RMSE: {metrics['rmse']:.6f}
- R²: {metrics['r2']:.6f}
- Sign accuracy: {metrics['sign_accuracy']:.3f}
- Top-1 intervention accuracy: {metrics['top1_accuracy']:.3f}
- Mean recommendation regret: {metrics['mean_regret']:.6f}
- Mean selected delta macro-F1: {metrics['mean_selected_delta']:.6f}

## Baselines
"""
    for b in baselines:
        text += f"- **{b['strategy']}**: mean selected delta={b['mean_selected_delta']:.6f}, positive-selection rate={b['positive_selection_rate']:.3f}\n"
    text += """
## Interpretation
The exhaustive oracle is an upper bound because it uses measured outcomes for every candidate.
The heuristic is deployable but simple. CleanWise is useful only if it generalizes to unseen
datasets and improves the recommendation/cost trade-off over such baselines.

## Limitations
The starter benchmark contains only four built-in datasets. Expand the benchmark, profile the
observed corrupted training partition, use repeated dataset-grouped validation, add uncertainty,
and measure end-to-end computational cost before making publication-level claims.
"""
    out = Path(args.output); out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text)
    print(f"Wrote {out}")

if __name__ == "__main__":
    main()
