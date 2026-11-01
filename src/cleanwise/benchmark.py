"""Generate reproducible CleanWise meta-datasets.

The starter benchmark deliberately uses scikit-learn built-in datasets so the full workflow can
run without internet access. Research claims require a substantially larger external benchmark.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import pandas as pd
from sklearn.datasets import load_breast_cancer, load_wine, load_iris, load_digits

from cleanwise.experiment import run_single_experiment, make_meta_features
from cleanwise.interventions import SUPPORTED_INTERVENTIONS

BUILTIN_DATASETS = {
    "breast_cancer": load_breast_cancer,
    "wine": load_wine,
    "iris": load_iris,
    "digits": load_digits,
}

def load_builtin_datasets():
    out = {}
    for name, loader in BUILTIN_DATASETS.items():
        data = loader(as_frame=True)
        out[name] = (data.data.copy(), pd.Series(data.target, name="target"))
    return out

def generate_benchmark(output_path="data/processed/experiments.csv", seed=42,
                       levels=(0.05, 0.15), repeats=1, quick=False):
    rows = []
    datasets = load_builtin_datasets()
    if quick:
        corruptions = ["missingness", "outliers"]
        interventions = ["median_imputation", "iqr_capping"]
        model_names = ["logistic_regression"]
        levels = tuple(levels[:1])
        repeats = 1
    else:
        corruptions = ["missingness", "outliers", "label_noise"]
        interventions = SUPPORTED_INTERVENTIONS
        model_names = ["logistic_regression", "random_forest"]

    for dataset_id, (X, y) in datasets.items():
        for corruption in corruptions:
            for level in levels:
                for intervention in interventions:
                    for model_name in model_names:
                        for repeat in range(repeats):
                            run_seed = seed + repeat
                            result = run_single_experiment(
                                X, y, dataset_id, corruption, level,
                                intervention, model_name, run_seed
                            )
                            # Deployment-available meta-features should describe observed data.
                            # Here we use the dataset itself for the starter benchmark. The research
                            # expansion should profile the corrupted training partition directly.
                            meta = make_meta_features(
                                X, y, corruption, level, intervention, model_name
                            )
                            rows.append({**meta, **result})
    frame = pd.DataFrame(rows)
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)
    print(f"Wrote {len(frame)} rows across {frame['dataset_id'].nunique()} datasets to {path}")
    return frame

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="data/processed/experiments.csv")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--levels", type=float, nargs="+", default=[0.05, 0.15])
    parser.add_argument("--quick", action="store_true",
                        help="Fast end-to-end smoke benchmark across all four starter datasets.")
    args = parser.parse_args()
    generate_benchmark(args.output, args.seed, tuple(args.levels), args.repeats, args.quick)

if __name__ == "__main__":
    main()
