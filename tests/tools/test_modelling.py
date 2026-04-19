import pandas as pd
import pytest
from sklearn.compose import ColumnTransformer

from tools.modelling import build_preprocessor, select_models, train_models


def test_select_models_returns_only_known_models():
    internal_memory = {
        "candidates_name": ["LogisticRegression", "UnknownModel", "DummyMostFrequent"],
        "search_space": "simple",
    }

    candidates = select_models(internal_memory, seed=42)
    names = [name for name, _, _ in candidates]

    assert "LogisticRegression" in names
    assert "DummyMostFrequent" in names
    assert "UnknownModel" not in names


def test_build_preprocessor_returns_column_transformer():
    profile = {
        "feature_types": {
            "numeric": ["num1", "num2"],
            "categorical": ["cat1", "cat2"],
        }
    }
    internal_memory = {
        "drop_cols": ["cat2"],
        "onehot_encoding": ["cat1"],
        "scale": ["num1", "num2"],
    }

    preprocessor = build_preprocessor(profile, internal_memory)

    assert isinstance(preprocessor, ColumnTransformer)
    transformer_names = [name for name, _, _ in preprocessor.transformers]
    assert "num_scale" in transformer_names
    assert "cat_ohe" in transformer_names


def test_train_models_raises_for_missing_target(sample_df):
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
            test_size=0.3,
            output_dir=".",
            internal_memory={},
        )


def test_train_models_raises_for_single_class():
    df = pd.DataFrame(
        {
            "num1": [1, 2, 3, 4],
            "cat1": ["a", "b", "a", "b"],
            "target": [1, 1, 1, 1],
        }
    )
    profile = {
        "feature_types": {"numeric": ["num1"], "categorical": ["cat1"]}
    }
    preprocessor = build_preprocessor(profile, {"onehot_encoding": ["cat1"], "scale": ["num1"]})

    with pytest.raises(ValueError, match="at least 2 classes"):
        train_models(
            df=df,
            target="target",
            preprocessor=preprocessor,
            candidates=[],
            seed=42,
            test_size=0.25,
            output_dir=".",
            internal_memory={},
        )