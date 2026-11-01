import pandas as pd
from cleanwise.profiling import profile_dataset, infer_task


def test_profile_contains_basic_features():
    X = pd.DataFrame({"a": [1.0, None, 3.0], "b": ["x", "y", "y"]})
    y = pd.Series([0, 1, 1])
    result = profile_dataset(X, y)
    assert result["n_rows_log"] > 0
    assert result["missing_rate_mean"] > 0
    assert 0 <= result["numeric_fraction"] <= 1
    assert result["target_n_classes_log"] > 0


def test_infer_task_classification():
    assert infer_task(pd.Series(["a", "b", "a"])) == "classification"
