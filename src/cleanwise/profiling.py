"""Dataset profiling utilities. These features are calculated without using held-out labels."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder


def _safe_float(value: object, default: float = 0.0) -> float:
    try:
        result = float(value)
        return result if np.isfinite(result) else default
    except (TypeError, ValueError):
        return default


def profile_dataset(X: pd.DataFrame, y: pd.Series | None = None) -> dict[str, float]:
    """Return compact, model-independent meta-features for a tabular dataset."""
    if not isinstance(X, pd.DataFrame):
        X = pd.DataFrame(X)
    n_rows, n_cols = X.shape
    numeric = X.select_dtypes(include=[np.number])
    categorical = X.select_dtypes(exclude=[np.number])
    missing = X.isna()
    missing_rates = missing.mean(axis=0) if n_cols else pd.Series(dtype=float)
    duplicate_rate = float(X.duplicated().mean()) if n_rows else 0.0

    numeric_values = numeric.replace([np.inf, -np.inf], np.nan)
    if numeric_values.size:
        q1 = numeric_values.quantile(0.25)
        q3 = numeric_values.quantile(0.75)
        iqr = (q3 - q1).replace(0, np.nan)
        outlier_mask = ((numeric_values < (q1 - 1.5 * iqr)) |
                        (numeric_values > (q3 + 1.5 * iqr)))
        outlier_rate = _safe_float(outlier_mask.mean().mean())
        mean_skew = _safe_float(numeric_values.skew(numeric_only=True).abs().replace(
            [np.inf, -np.inf], np.nan).mean())
        mean_std = _safe_float(numeric_values.std(numeric_only=True).replace(
            [np.inf, -np.inf], np.nan).mean())
    else:
        outlier_rate, mean_skew, mean_std = 0.0, 0.0, 0.0

    feature = {
        "n_rows_log": float(np.log1p(n_rows)),
        "n_columns_log": float(np.log1p(n_cols)),
        "numeric_fraction": float(len(numeric.columns) / max(n_cols, 1)),
        "categorical_fraction": float(len(categorical.columns) / max(n_cols, 1)),
        "missing_rate_mean": _safe_float(missing_rates.mean()),
        "missing_rate_max": _safe_float(missing_rates.max()),
        "missing_rate_std": _safe_float(missing_rates.std()),
        "duplicate_rate": duplicate_rate,
        "numeric_outlier_rate": outlier_rate,
        "numeric_abs_skew_mean": mean_skew,
        "numeric_std_mean": mean_std,
        "mean_numeric_unique_fraction": _safe_float(
            numeric.nunique(dropna=True).div(max(n_rows, 1)).mean()
        ),
        "mean_categorical_cardinality_log": _safe_float(
            np.log1p(categorical.nunique(dropna=True)).mean()
        ),
    }

    if y is not None:
        y_clean = pd.Series(y).reset_index(drop=True)
        valid = y_clean.notna()
        y_clean = y_clean[valid]
        counts = y_clean.value_counts(normalize=True)
        feature["target_n_classes_log"] = float(np.log1p(y_clean.nunique()))
        feature["target_class_imbalance"] = (
            float(counts.max() - counts.min()) if len(counts) > 1 else 0.0
        )
        feature["target_min_class_fraction"] = _safe_float(counts.min())
    else:
        feature.update({
            "target_n_classes_log": 0.0,
            "target_class_imbalance": 0.0,
            "target_min_class_fraction": 0.0,
        })
    return feature


def infer_task(y: pd.Series) -> str:
    """Conservative target-type check for the first version (classification only)."""
    y = pd.Series(y).dropna()
    if y.empty:
        raise ValueError("Target column has no non-missing values.")
    if y.nunique() < 2:
        raise ValueError("Target must contain at least two classes.")
    return "classification"
