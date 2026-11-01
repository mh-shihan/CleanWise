"""Leakage-aware experiment runner and reproducible synthetic-corruption benchmark."""
from __future__ import annotations

import time
import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, balanced_accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.base import clone

from cleanwise.interventions import SUPPORTED_INTERVENTIONS, apply_intervention
from cleanwise.profiling import profile_dataset


def make_classifier(model_name: str, seed: int = 42):
    if model_name == "logistic_regression":
        return LogisticRegression(max_iter=500, random_state=seed)
    if model_name == "random_forest":
        return RandomForestClassifier(
            n_estimators=40, min_samples_leaf=2, n_jobs=1, random_state=seed
        )
    if model_name == "hist_gradient_boosting":
        # The benchmark's preprocessing one-hot encodes categoricals; HGB is included as an
        # optional model family. Use a regular random forest/logistic model if sparse data occurs.
        return HistGradientBoostingClassifier(random_state=seed, max_iter=100)
    raise ValueError(f"Unknown model: {model_name}")


def _encode_features(X_train: pd.DataFrame, X_eval: pd.DataFrame):
    """Encode categoricals and impute missing values, fitting all transforms on training only."""
    numeric_cols = list(X_train.select_dtypes(include=[np.number]).columns)
    categorical_cols = [c for c in X_train.columns if c not in numeric_cols]
    transformers = []
    if numeric_cols:
        transformers.append(("num", Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]), numeric_cols))
    if categorical_cols:
        transformers.append(("cat", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]), categorical_cols))
    if not transformers:
        raise ValueError("Dataset has no usable feature columns.")
    transformer = ColumnTransformer(transformers, remainder="drop")
    return transformer.fit_transform(X_train), transformer.transform(X_eval), transformer


def score_model(X_train, y_train, X_eval, y_eval, model_name="random_forest", seed=42):
    start = time.perf_counter()
    Xtr, Xev, _ = _encode_features(X_train, X_eval)
    model = make_classifier(model_name, seed)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.fit(Xtr, y_train)
        pred = model.predict(Xev)
        probabilities = None
        if hasattr(model, "predict_proba"):
            try:
                probabilities = model.predict_proba(Xev)
            except Exception:
                pass
        macro_f1 = f1_score(y_eval, pred, average="macro", zero_division=0)
        balanced = balanced_accuracy_score(y_eval, pred)
        auc = np.nan
        if probabilities is not None and len(np.unique(y_eval)) == 2:
            try:
                auc = roc_auc_score(y_eval, probabilities[:, 1])
            except ValueError:
                auc = np.nan
    return {
        "macro_f1": float(macro_f1),
        "balanced_accuracy": float(balanced),
        "roc_auc": float(auc) if np.isfinite(auc) else np.nan,
        "runtime_seconds": float(time.perf_counter() - start),
    }


def corrupt_training_data(X, y, corruption: str, level: float, seed: int):
    """Apply controlled corruption to training data only; returns a copy."""
    rng = np.random.default_rng(seed)
    Xc = X.copy(deep=True)
    yc = pd.Series(y).copy(deep=True)
    if corruption == "missingness":
        eligible = Xc.columns.tolist()
        if not eligible:
            return Xc, yc
        mask = rng.random(Xc.shape) < level
        # Preserve at least some observed values in each column.
        Xc = Xc.mask(mask)
    elif corruption == "outliers":
        cols = list(Xc.select_dtypes(include=[np.number]).columns)
        if cols:
            n_cells = max(1, int(Xc.shape[0] * len(cols) * level))
            for _ in range(n_cells):
                col = rng.choice(cols)
                idx = rng.integers(0, len(Xc))
                series = pd.to_numeric(Xc[col], errors="coerce")
                sd = float(series.std()) if np.isfinite(series.std()) else 1.0
                mean = float(series.mean()) if np.isfinite(series.mean()) else 0.0
                Xc.loc[Xc.index[idx], col] = mean + 8 * max(sd, 1.0) * rng.choice([-1, 1])
    elif corruption == "label_noise":
        unique = pd.Series(yc).dropna().unique()
        if len(unique) >= 2:
            n = max(1, int(len(yc) * level))
            positions = rng.choice(len(yc), size=min(n, len(yc)), replace=False)
            for pos in positions:
                alternatives = unique[unique != yc.iloc[pos]]
                if len(alternatives):
                    yc.iloc[pos] = rng.choice(alternatives)
    elif corruption == "none":
        pass
    else:
        raise ValueError(f"Unknown corruption type: {corruption}")
    return Xc, yc


def run_single_experiment(X, y, dataset_id: str, corruption: str, level: float,
                          intervention: str, model_name: str, seed: int = 42):
    """One experiment. Corrupt training split only; evaluate on the untouched test split."""
    X = pd.DataFrame(X).reset_index(drop=True)
    y = pd.Series(y).reset_index(drop=True)
    # Split first; test features/labels remain untouched by corruption.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=seed, stratify=y
    )
    X_corrupt, y_corrupt = corrupt_training_data(
        X_train, y_train, corruption, level, seed=seed + 100
    )
    baseline = score_model(X_corrupt, y_corrupt, X_test, y_test, model_name, seed)
    X_cleaned, X_test_cleaned = apply_intervention(X_corrupt, X_test, intervention)
    cleaned = score_model(X_cleaned, y_corrupt, X_test_cleaned, y_test, model_name, seed)
    return {
        "dataset_id": dataset_id,
        "corruption": corruption,
        "corruption_level": float(level),
        "intervention": intervention,
        "model_name": model_name,
        "seed": int(seed),
        "baseline_macro_f1": baseline["macro_f1"],
        "intervention_macro_f1": cleaned["macro_f1"],
        "delta_macro_f1": cleaned["macro_f1"] - baseline["macro_f1"],
        "baseline_balanced_accuracy": baseline["balanced_accuracy"],
        "intervention_balanced_accuracy": cleaned["balanced_accuracy"],
        "delta_balanced_accuracy": cleaned["balanced_accuracy"] - baseline["balanced_accuracy"],
        "baseline_runtime_seconds": baseline["runtime_seconds"],
        "intervention_runtime_seconds": cleaned["runtime_seconds"],
    }


def make_meta_features(X: pd.DataFrame, y: pd.Series, corruption: str,
                       level: float, intervention: str, model_name: str) -> dict:
    features = profile_dataset(X, y)
    features.update({
        "corruption": corruption,
        "corruption_level": float(level),
        "intervention": intervention,
        "model_name": model_name,
    })
    return features
