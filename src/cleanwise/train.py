"""Train/evaluate the CleanWise meta-model and save reproducible artifacts."""
from __future__ import annotations
import argparse, json
from pathlib import Path
import pandas as pd
from cleanwise.meta_model import (
    grouped_evaluation, feature_columns, save_model,
    heuristic_baseline, exhaustive_oracle
)

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="data/processed/experiments.csv")
    p.add_argument("--model", default="random_forest",
                   choices=["ridge", "random_forest", "extra_trees"])
    p.add_argument("--output-dir", default="artifacts")
    args = p.parse_args()
    frame = pd.read_csv(args.data)
    required = {"dataset_id", "delta_macro_f1", "corruption", "intervention", "model_name"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")
    model, metrics, predictions = grouped_evaluation(frame, args.model)
    baselines = [heuristic_baseline(frame), exhaustive_oracle(frame)]
    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)
    save_model(model, feature_columns(frame), out / "cleanwise_meta_model.joblib")
    (out / "evaluation_metrics.json").write_text(json.dumps(metrics, indent=2))
    (out / "baseline_metrics.json").write_text(json.dumps(baselines, indent=2))
    predictions.to_csv(out / "heldout_predictions.csv", index=False)
    print(json.dumps({"meta_model": metrics, "baselines": baselines}, indent=2))

if __name__ == "__main__":
    main()
