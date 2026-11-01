import pandas as pd
from cleanwise.experiment import corrupt_training_data, run_single_experiment
from sklearn.datasets import load_breast_cancer


def test_corruption_is_reproducible():
    X = pd.DataFrame({"a": range(20), "b": range(20, 40)})
    y = pd.Series([0, 1] * 10)
    Xa, ya = corrupt_training_data(X, y, "missingness", 0.2, 123)
    Xb, yb = corrupt_training_data(X, y, "missingness", 0.2, 123)
    pd.testing.assert_frame_equal(Xa, Xb)
    pd.testing.assert_series_equal(ya, yb)


def test_single_experiment_returns_delta():
    data = load_breast_cancer(as_frame=True)
    result = run_single_experiment(data.data, data.target, "breast_cancer",
                                   "missingness", 0.05, "median_imputation",
                                   "logistic_regression", seed=42)
    assert "delta_macro_f1" in result
    assert -1 <= result["delta_macro_f1"] <= 1
