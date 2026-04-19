import pandas as pd
import pytest

from tools.data_profiler import (
    infer_target_column,
    is_classification_target,
    dataset_fingerprint,
    profile_dataset,
)


def test_infer_target_column_prefers_common_name():
    df = pd.DataFrame({"x": [1, 2], "Target": [0, 1]})
    assert infer_target_column(df) == "Target"


def test_is_classification_target_for_object_series():
    s = pd.Series(["yes", "no", "yes"])
    assert is_classification_target(s) is True


def test_dataset_fingerprint_changes_when_columns_change():
    df1 = pd.DataFrame({"a": [1], "target": [0]})
    df2 = pd.DataFrame({"b": [1], "target": [0]})

    fp1 = dataset_fingerprint(df1, "target")
    fp2 = dataset_fingerprint(df2, "target")

    assert fp1.startswith("fp_")
    assert fp2.startswith("fp_")
    assert fp1 != fp2


def test_profile_dataset_returns_expected_keys(sample_df):
    profile = profile_dataset(sample_df, "target")

    assert profile["shape"] == {"rows": 6, "cols": 4}
    assert profile["target"] == "target"
    assert profile["is_classification"] is True
    assert "num1" in profile["feature_types"]["numeric"]
    assert "cat1" in profile["feature_types"]["categorical"]
    assert "outlier_ratio_by_col" in profile
    assert "skewness_by_col" in profile
    assert "imbalance_ratio" in profile


def test_profile_dataset_raises_when_target_missing(sample_df):
    with pytest.raises(ValueError, match="Target column"):
        profile_dataset(sample_df, "missing_target")