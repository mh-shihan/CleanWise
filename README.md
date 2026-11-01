# CleanWise

**ML-based prediction of the value of data-cleaning interventions**

CleanWise asks a concrete question: *Can observable dataset characteristics predict which
cleaning intervention is worth trying before exhaustively evaluating every candidate?*

This repository is a complete runnable **research prototype / portfolio project** for tabular
classification. It includes dataset profiling, controlled corruption, leakage-aware interventions,
a meta-learning recommender, dataset-grouped evaluation, heuristic and oracle baselines, tests,
a generated experiment report, and a Streamlit demo.

## Important research statement

The code is complete as a v1 prototype; the included four-dataset benchmark is intentionally a
**pipeline validation benchmark**, not sufficient evidence for a paper-level generalization claim.
Do not put invented performance numbers on a CV. Run the experiments and report measured results.

## Architecture

```text
Dataset
  │
  ├──> profiler ──> meta-features ──────────────────────────────┐
  │                                                            │
  └──> train/test split ─> controlled training corruption       │
                           │                                    │
                           ├─> baseline model                    │
                           └─> cleaning interventions ─> ΔF1     │
                                                                ▼
                                      meta-dataset ─> meta-model
                                                        │
                                                        ▼
                                            ranked interventions
```

## Supported v1 scope

- Classification only.
- Data issues: missingness, numeric outliers, synthetic label noise.
- Candidate interventions: median imputation, most-frequent imputation, robust scaling,
  standard scaling, and IQR capping.
- Downstream classifiers: logistic regression and random forest.
- Primary target: change in macro-F1.
- Meta-models: Ridge, Random Forest, Extra Trees.
- Recommendation metrics: MAE, RMSE, R², sign accuracy, top-1 accuracy, mean regret.
- Baselines: issue-based heuristic and exhaustive oracle upper bound.

`label_noise_filter` is deliberately **not** claimed as implemented. Injecting label noise is useful
for robustness experiments, but safely identifying/correcting noisy labels is a separate method.

## Install

```bash
pip install -e ".[dev]"
```

or:

```bash
pip install -r requirements.txt
export PYTHONPATH=src
```

## Fast end-to-end validation

```bash
pytest -q
python -m cleanwise.benchmark --quick --levels 0.05
python -m cleanwise.train --model ridge
python -m cleanwise.report
```

Then inspect:

```text
artifacts/evaluation_metrics.json
artifacts/baseline_metrics.json
artifacts/heldout_predictions.csv
reports/generated/experiment_report.md
```

## Larger starter run

```bash
python -m cleanwise.benchmark \
  --output data/processed/experiments.csv \
  --seed 42 \
  --repeats 2 \
  --levels 0.05 0.15 0.30

python -m cleanwise.train \
  --data data/processed/experiments.csv \
  --model random_forest

python -m cleanwise.report
```

## Streamlit demo

```bash
streamlit run src/cleanwise/app.py
```

The demo can profile an uploaded CSV. If a trained artifact exists, it can rank interventions
using the meta-model. Direct-comparison mode evaluates candidates on the uploaded dataset and is
explicitly labeled as an experiment, not a prediction.

## Kaggle

Upload the project ZIP as a Kaggle Dataset/Input, extract it, then:

```python
%cd /kaggle/working/cleanwise
!pip install -e .
!pytest -q
!python -m cleanwise.benchmark --quick --levels 0.05
!python -m cleanwise.train --model ridge
!python -m cleanwise.report
```

After that works, run the larger benchmark. Save `data/processed/experiments.csv`, `artifacts/`,
and `reports/generated/` as notebook outputs.

## Leakage controls

1. Train/test splitting happens before synthetic corruption.
2. Synthetic corruption is applied to the training partition, not the held-out test partition.
3. Imputers/scalers/clipping bounds are fit using training features only.
4. Meta-model evaluation holds out complete dataset identities.
5. Dataset identity itself is not a meta-model feature.

### Research-grade improvement still required

For publication experiments, meta-features should be computed from the **observed/corrupted
training partition** rather than from the original dataset object. The current starter benchmark
keeps feature generation simple so the entire pipeline is easy to understand and validate.

Intervention selection should also use validation data, with a final untouched test set reserved
for the selected pipeline's final performance estimate.

## Repository

```text
cleanwise/
├── configs/
├── data/
├── notebooks/
├── src/cleanwise/
│   ├── profiling.py
│   ├── interventions.py
│   ├── experiment.py
│   ├── benchmark.py
│   ├── meta_model.py
│   ├── train.py
│   ├── report.py
│   └── app.py
├── tests/
├── artifacts/
└── reports/
```

## Beginner mental model

Suppose a dataset has missing values. Normally you might try median imputation, mode imputation,
scaling, capping, and other options, train a classifier for every option, and compare them.

CleanWise learns from many previous experiments. Each experiment becomes one meta-learning row:

```text
dataset characteristics + problem + intervention + model -> observed Δ macro-F1
```

The meta-model then receives a new dataset's characteristics and estimates the ΔF1 for each
candidate intervention. The candidates can be ranked before an exhaustive search.

## What makes the evaluation meaningful?

Randomly splitting experiment rows would leak dataset-specific patterns: experiments from the same
dataset could appear in both training and test data. CleanWise instead groups by `dataset_id`.
A held-out dataset therefore behaves more like the real deployment question: *can this knowledge
transfer to a dataset the recommender has never seen?*

## Research expansion plan

For a serious conference submission:

1. Expand to 30–100+ independent OpenML/UCI datasets with documented licenses.
2. Compute all recommendation-time features from corrupted/observed training data only.
3. Use repeated GroupKFold or leave-one-dataset-out validation.
4. Add stronger heuristics and AutoML/pipeline-selection comparisons from related work.
5. Add multiple corruption mechanisms (MCAR/MAR-like missingness, structured label noise).
6. Add uncertainty calibration and an abstain option.
7. Measure wall-clock cost and number of evaluated interventions.
8. Add statistical confidence intervals and paired significance/effect-size analyses.
9. Evaluate naturally imperfect datasets in addition to synthetic corruptions.
10. Conduct ablation studies on meta-feature families.

## CV wording after you run and verify it

**CleanWise — Data-Cleaning Intervention Recommender**  
*Python, pandas, scikit-learn, Streamlit*

- Developed a reproducible meta-learning pipeline that predicts the downstream value of candidate
  data-cleaning interventions using dataset-level statistical and quality features.
- Implemented controlled data-quality corruptions, leakage-aware preprocessing, dataset-grouped
  holdout evaluation, heuristic baselines, and intervention-ranking metrics.
- Built an interactive Streamlit application for dataset profiling and cleaning-intervention
  recommendation.

Add numerical results only after your expanded benchmark produces defensible held-out results.

## License

MIT. Dataset licenses remain those of their original providers.
