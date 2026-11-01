"""Cleaning intervention definitions used by the benchmark and demo."""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import RobustScaler, StandardScaler


class IQRClipper(BaseEstimator, TransformerMixin):
    """Clip numeric columns to training-fold IQR bounds; leave categorical columns unchanged."""
    def __init__(self, factor: float = 1.5):
        self.factor = factor

    def fit(self, X, y=None):
        frame = X if isinstance(X, pd.DataFrame) else pd.DataFrame(X)
        self.columns_ = list(frame.columns)
        self.numeric_columns_ = list(frame.select_dtypes(include=[np.number]).columns)
        self.bounds_ = {}
        for col in self.numeric_columns_:
            series = pd.to_numeric(frame[col], errors="coerce")
            q1, q3 = series.quantile([0.25, 0.75])
            iqr = q3 - q1
            if not np.isfinite(iqr) or iqr == 0:
                self.bounds_[col] = (-np.inf, np.inf)
            else:
                self.bounds_[col] = (q1 - self.factor * iqr, q3 + self.factor * iqr)
        return self

    def transform(self, X):
        frame = X.copy() if isinstance(X, pd.DataFrame) else pd.DataFrame(X, columns=self.columns_)
        for col, (lo, hi) in self.bounds_.items():
            frame[col] = pd.to_numeric(frame[col], errors="coerce").clip(lo, hi)
        return frame


def apply_intervention(X_train: pd.DataFrame, X_eval: pd.DataFrame, intervention: str):
    """Fit preprocessing on training features only and transform train/evaluation features."""
    # Normalize Python None values to NumPy NaN so scikit-learn imputers handle both consistently.
    train = X_train.copy().replace({None: np.nan})
    evaluate = X_eval.copy().replace({None: np.nan})
    numeric_cols = list(train.select_dtypes(include=[np.number]).columns)
    categorical_cols = [c for c in train.columns if c not in numeric_cols]

    if intervention == "baseline":
        # Baseline is a leakage-safe minimal pipeline: median/mode imputation only.
        intervention = "median_imputation"
    if intervention == "row_removal":
        # Row removal is implemented by the experiment runner because X/y rows must stay aligned.
        raise ValueError("row_removal must be handled by the experiment runner.")
    if intervention == "label_noise_filter":
        # This needs a separate training-only label-cleaning algorithm and is not silently faked.
        raise ValueError(
            "label_noise_filter is not in the initial supported intervention set; "
            "it requires a separately validated label-cleaning method."
        )

    if intervention not in {
        "median_imputation", "most_frequent_imputation", "robust_scaling",
        "standard_scaling", "iqr_capping"
    }:
        raise ValueError(f"Unknown intervention: {intervention}")

    if intervention == "iqr_capping":
        clipper = IQRClipper().fit(train)
        train = clipper.transform(train)
        evaluate = clipper.transform(evaluate)

    if numeric_cols:
        strategy = "most_frequent" if intervention == "most_frequent_imputation" else "median"
        if intervention in {"median_imputation", "most_frequent_imputation"}:
            imputer = SimpleImputer(strategy=strategy)
            train.loc[:, numeric_cols] = imputer.fit_transform(train[numeric_cols])
            evaluate.loc[:, numeric_cols] = imputer.transform(evaluate[numeric_cols])
        elif intervention in {"robust_scaling", "standard_scaling"}:
            imputer = SimpleImputer(strategy="median")
            train.loc[:, numeric_cols] = imputer.fit_transform(train[numeric_cols])
            evaluate.loc[:, numeric_cols] = imputer.transform(evaluate[numeric_cols])
            scaler = RobustScaler() if intervention == "robust_scaling" else StandardScaler()
            train.loc[:, numeric_cols] = scaler.fit_transform(train[numeric_cols])
            evaluate.loc[:, numeric_cols] = scaler.transform(evaluate[numeric_cols])

    # Every intervention must return data without missing values. For IQR capping, imputation
    # is a separate necessary preprocessing step; the capping action itself only clips extremes.
    if categorical_cols:
        imputer = SimpleImputer(strategy="most_frequent")
        train.loc[:, categorical_cols] = imputer.fit_transform(train[categorical_cols])
        evaluate.loc[:, categorical_cols] = imputer.transform(evaluate[categorical_cols])

    remaining_numeric = [c for c in numeric_cols if train[c].isna().any()]
    if remaining_numeric:
        imputer = SimpleImputer(strategy="median")
        train.loc[:, remaining_numeric] = imputer.fit_transform(train[remaining_numeric])
        evaluate.loc[:, remaining_numeric] = imputer.transform(evaluate[remaining_numeric])

    return train, evaluate


SUPPORTED_INTERVENTIONS = [
    "median_imputation",
    "most_frequent_imputation",
    "robust_scaling",
    "standard_scaling",
    "iqr_capping",
]
