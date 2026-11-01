import numpy as np
import pandas as pd
import pytest
from cleanwise.interventions import apply_intervention


@pytest.mark.parametrize("action", [
    "median_imputation", "most_frequent_imputation", "robust_scaling",
    "standard_scaling", "iqr_capping"
])
def test_intervention_handles_missing_values(action):
    train = pd.DataFrame({"x": [1.0, np.nan, 3.0, 100.0], "cat": ["a", "b", None, "a"]})
    test = pd.DataFrame({"x": [np.nan, 4.0], "cat": [None, "c"]})
    Xt, Xv = apply_intervention(train, test, action)
    assert not Xt.isna().any().any()
    assert not Xv.isna().any().any()
    assert list(Xt.columns) == list(Xv.columns)
