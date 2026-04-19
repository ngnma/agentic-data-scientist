import pandas as pd
import pytest

from tools.data_profiler import (
    dataset_fingerprint,
    infer_target_column,
    is_classification_target,
    profile_dataset,
)


def test_infer_target_column_prefers_common_name():
    df = pd.DataFrame({"x": [1, 2], "Survived": [0, 1]})
    assert infer_target_column(df) == "Survived"


def test_infer_target_column_uses_last_low_cardinality():
    df = pd.DataFrame({"a": [1, 2, 3], "b": [0, 1, 0]})
    assert infer_target_column(df) == "b"


def test_is_classification_target_true_for_object():
    s = pd.Series(["yes", "no", "yes"])
    assert is_classification_target(s) is True


def test_dataset_fingerprint_changes_with_columns():
    df1 = pd.DataFrame({"a": [1], "target": [0]})
    df2 = pd.DataFrame({"b": [1], "target": [0]})

    assert dataset_fingerprint(df1, "target") != dataset_fingerprint(df2, "target")


def test_profile_dataset_excludes_bool_from_numeric(bool_df):
    profile = profile_dataset(bool_df, "target")

    assert "flag" not in profile["feature_types"]["numeric"]
    assert "flag" in profile["feature_types"]["categorical"]
    assert profile["n_classes"] == 2


def test_profile_dataset_raises_when_target_missing(sample_df):
    with pytest.raises(ValueError, match="Target column"):
        profile_dataset(sample_df, "missing_target")