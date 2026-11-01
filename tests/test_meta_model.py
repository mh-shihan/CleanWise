import pandas as pd
from cleanwise.meta_model import grouped_evaluation, heuristic_baseline, exhaustive_oracle

def fake_frame():
    rows=[]
    for d in ["a","b","c","d"]:
        for issue in ["missingness","outliers"]:
            for intervention, delta in [("median_imputation",0.01),("iqr_capping",0.02)]:
                rows.append({
                    "dataset_id":d, "corruption":issue, "corruption_level":0.1,
                    "intervention":intervention, "model_name":"logistic_regression", "seed":42,
                    "delta_macro_f1":delta if issue=="outliers" else delta/2,
                    "baseline_macro_f1":0.7, "intervention_macro_f1":0.71,
                    "baseline_balanced_accuracy":0.7, "intervention_balanced_accuracy":0.71,
                    "delta_balanced_accuracy":0.01, "baseline_runtime_seconds":0.1,
                    "intervention_runtime_seconds":0.1, "n_rows_log":5.0,
                    "n_columns_log":2.0, "missing_rate_mean":0.1,
                })
    return pd.DataFrame(rows)

def test_grouped_evaluation_runs():
    frame=fake_frame()
    model, metrics, pred=grouped_evaluation(frame, "ridge")
    assert metrics["n_test_datasets"] >= 1
    assert "top1_accuracy" in metrics
    assert len(pred) > 0

def test_baselines_run():
    frame=fake_frame()
    assert heuristic_baseline(frame)["strategy"] == "problem_based_heuristic"
    assert exhaustive_oracle(frame)["mean_selected_delta"] >= 0
