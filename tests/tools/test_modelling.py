import pandas as pd
import pytest
from sklearn.compose import ColumnTransformer

from tools.modelling import build_preprocessor, select_models, train_models


def test_build_preprocessor_returns_column_transformer(bool_df):
    profile = {
        "feature_types": {
            "numeric": ["num1"],
            "categorical": ["flag", "cat1"],
        }
    }
    internal_memory = {
        "onehot_encoding": ["flag", "cat1"],
        "scale": ["num1"],
    }

    preprocessor = build_preprocessor(profile, internal_memory)

    assert isinstance(preprocessor, ColumnTransformer)

    X = bool_df.drop(columns=["target"])
    transformed = preprocessor.fit_transform(X, bool_df["target"])
    assert transformed.shape[0] == len(bool_df)


def test_select_models_filters_unknown_model():
    internal_memory = {
        "candidates_name": ["LogisticRegression", "UnknownModel", "DummyMostFrequent"],
        "search_space": "simple",
    }

    candidates = select_models(internal_memory, seed=42)
    names = [name for name, _, _ in candidates]

    assert "LogisticRegression" in names
    assert "DummyMostFrequent" in names
    assert "UnknownModel" not in names


def test_train_models_raises_for_missing_target(sample_df, tmp_path):
    profile = {
        "feature_types": {
            "numeric": ["num1", "num2"],
            "categorical": ["cat1"],
        }
    }
    preprocessor = build_preprocessor(profile, {"onehot_encoding": ["cat1"], "scale": ["num1", "num2"]})

    with pytest.raises(ValueError, match="Target 'missing' not found"):
        train_models(
            df=sample_df,
            target="missing",
            preprocessor=preprocessor,
            candidates=[],
            seed=42,
            test_size=0.25,
            output_dir=str(tmp_path),
            internal_memory={},
            verbose=False,
        )


def test_train_models_happy_path(sample_df, tmp_path):
    profile = {
        "feature_types": {
            "numeric": ["num1", "num2"],
            "categorical": ["cat1"],
        }
    }
    preprocessor = build_preprocessor(profile, {"onehot_encoding": ["cat1"], "scale": ["num1", "num2"]})
    candidates = select_models(
        {"candidates_name": ["DummyMostFrequent"], "search_space": "normal"},
        seed=42,
    )

    result = train_models(
        df=sample_df,
        target="target",
        preprocessor=preprocessor,
        candidates=candidates,
        seed=42,
        test_size=0.25,
        output_dir=str(tmp_path),
        internal_memory={},
        verbose=False,
    )

    assert "best" in result
    assert "all_metrics" in result
    assert result["best"]["metrics"]["model"] == "DummyMostFrequent"