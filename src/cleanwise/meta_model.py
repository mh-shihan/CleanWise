"""Meta-learning, dataset-grouped evaluation, and recommendation baselines."""
from __future__ import annotations
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupShuffleSplit, GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

TARGET = "delta_macro_f1"
CATEGORICAL = ["corruption", "intervention", "model_name"]
DROP_COLUMNS = [
    "dataset_id", "seed", "baseline_macro_f1", "intervention_macro_f1",
    "delta_macro_f1", "baseline_balanced_accuracy", "intervention_balanced_accuracy",
    "delta_balanced_accuracy", "baseline_runtime_seconds", "intervention_runtime_seconds",
]

def feature_columns(frame):
    return [c for c in frame.columns if c not in DROP_COLUMNS]

def build_meta_pipeline(columns, model_name="random_forest"):
    cats = [c for c in CATEGORICAL if c in columns]
    nums = [c for c in columns if c not in cats]
    preprocess = ColumnTransformer([
        ("num", Pipeline([("imputer", SimpleImputer(strategy="median")),
                          ("scale", StandardScaler())]), nums),
        ("cat", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")),
                          ("onehot", OneHotEncoder(handle_unknown="ignore"))]), cats),
    ])
    if model_name == "ridge":
        model = Ridge(alpha=1.0)
    elif model_name == "random_forest":
        model = RandomForestRegressor(n_estimators=200, min_samples_leaf=2,
                                      random_state=42, n_jobs=-1)
    elif model_name == "extra_trees":
        model = ExtraTreesRegressor(n_estimators=200, min_samples_leaf=2,
                                    random_state=42, n_jobs=-1)
    else:
        raise ValueError(f"Unknown meta-model: {model_name}")
    return Pipeline([("preprocess", preprocess), ("model", model)])

def _ranking_metrics(test, predictions):
    work = test[["dataset_id", "corruption", "corruption_level", "model_name",
                 "intervention", TARGET]].copy()
    work["prediction"] = predictions
    group_cols = ["dataset_id", "corruption", "corruption_level", "model_name"]
    hits, regrets, selected_gains = [], [], []
    for _, group in work.groupby(group_cols, dropna=False):
        if len(group) < 2:
            continue
        predicted_best = group.loc[group["prediction"].idxmax()]
        oracle = group[TARGET].max()
        actual_selected = float(predicted_best[TARGET])
        hits.append(float(predicted_best["intervention"] ==
                          group.loc[group[TARGET].idxmax(), "intervention"]))
        regrets.append(float(oracle - actual_selected))
        selected_gains.append(actual_selected)
    return {
        "top1_accuracy": float(np.mean(hits)) if hits else np.nan,
        "mean_regret": float(np.mean(regrets)) if regrets else np.nan,
        "mean_selected_delta": float(np.mean(selected_gains)) if selected_gains else np.nan,
    }

def evaluate_predictions(test, predictions, model_name):
    y = test[TARGET].to_numpy()
    metrics = {
        "model": model_name,
        "mae": float(mean_absolute_error(y, predictions)),
        "rmse": float(np.sqrt(mean_squared_error(y, predictions))),
        "r2": float(r2_score(y, predictions)) if len(y) > 1 else np.nan,
        "sign_accuracy": float(np.mean(np.sign(predictions) == np.sign(y))),
    }
    metrics.update(_ranking_metrics(test, predictions))
    return metrics

def grouped_evaluation(frame, model_name="random_forest", test_size=0.25, seed=42):
    if frame["dataset_id"].nunique() < 4:
        raise ValueError("Need at least 4 distinct datasets for grouped evaluation.")
    columns = feature_columns(frame)
    groups = frame["dataset_id"].astype(str).to_numpy()
    train_idx, test_idx = next(GroupShuffleSplit(
        n_splits=1, test_size=test_size, random_state=seed
    ).split(frame, frame[TARGET], groups))
    train, test = frame.iloc[train_idx].copy(), frame.iloc[test_idx].copy()
    assert set(train.dataset_id).isdisjoint(set(test.dataset_id))
    model = build_meta_pipeline(columns, model_name)
    model.fit(train[columns], train[TARGET])
    pred = model.predict(test[columns])
    metrics = evaluate_predictions(test, pred, model_name)
    metrics.update({
        "n_train_datasets": int(train.dataset_id.nunique()),
        "n_test_datasets": int(test.dataset_id.nunique()),
        "n_train_rows": int(len(train)),
        "n_test_rows": int(len(test)),
    })
    output = test[["dataset_id", "corruption", "corruption_level",
                   "intervention", "model_name", TARGET]].copy()
    output["predicted_delta_macro_f1"] = pred
    return model, metrics, output

def heuristic_baseline(frame):
    """Issue-based fixed policy: impute missingness, cap outliers, otherwise median impute."""
    policy = {"missingness": "median_imputation", "outliers": "iqr_capping",
              "label_noise": "median_imputation"}
    chosen = []
    group_cols = ["dataset_id", "corruption", "corruption_level", "model_name", "seed"]
    for _, group in frame.groupby(group_cols, dropna=False):
        desired = policy.get(group.iloc[0]["corruption"], "median_imputation")
        candidates = group[group.intervention == desired]
        if candidates.empty:
            candidates = group.iloc[[0]]
        chosen.append(float(candidates.iloc[0][TARGET]))
    return {
        "strategy": "problem_based_heuristic",
        "mean_selected_delta": float(np.mean(chosen)) if chosen else np.nan,
        "positive_selection_rate": float(np.mean(np.array(chosen) > 0)) if chosen else np.nan,
    }

def exhaustive_oracle(frame):
    group_cols = ["dataset_id", "corruption", "corruption_level", "model_name", "seed"]
    best = frame.groupby(group_cols, dropna=False)[TARGET].max()
    return {"strategy": "exhaustive_oracle_upper_bound",
            "mean_selected_delta": float(best.mean()),
            "positive_selection_rate": float((best > 0).mean())}

def save_model(model, columns, path):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": model, "feature_columns": columns}, path)

def load_model(path):
    return joblib.load(path)

def predict_interventions(bundle, meta_features):
    model = bundle["model"] if isinstance(bundle, dict) else bundle
    cols = bundle.get("feature_columns") if isinstance(bundle, dict) else None
    X = meta_features.reindex(columns=cols) if cols else meta_features
    result = meta_features.copy()
    result["predicted_delta_macro_f1"] = model.predict(X)
    return result.sort_values("predicted_delta_macro_f1", ascending=False).reset_index(drop=True)
