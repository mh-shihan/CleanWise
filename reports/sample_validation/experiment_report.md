# CleanWise Experiment Report

## Scope
This is an automatically generated starter-benchmark report. It is a pipeline validation,
not evidence of general performance across real-world tabular datasets.

## Dataset-level holdout
- Training datasets: 3
- Held-out datasets: 1
- Training experiment rows: 12
- Held-out experiment rows: 4

## Meta-model results
- MAE: 0.029891
- RMSE: 0.036480
- R²: -0.144529
- Sign accuracy: 0.250
- Top-1 intervention accuracy: 0.500
- Mean recommendation regret: 0.002181
- Mean selected delta macro-F1: 0.036413

## Baselines
- **problem_based_heuristic**: mean selected delta=0.038405, positive-selection rate=0.500
- **exhaustive_oracle_upper_bound**: mean selected delta=0.039339, positive-selection rate=0.625

## Interpretation
The exhaustive oracle is an upper bound because it uses measured outcomes for every candidate.
The heuristic is deployable but simple. CleanWise is useful only if it generalizes to unseen
datasets and improves the recommendation/cost trade-off over such baselines.

## Limitations
The starter benchmark contains only four built-in datasets. Expand the benchmark, profile the
observed corrupted training partition, use repeated dataset-grouped validation, add uncertainty,
and measure end-to-end computational cost before making publication-level claims.
