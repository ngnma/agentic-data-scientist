import sys
from pathlib import Path

import pandas as pd
import pytest


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


@pytest.fixture
def sample_df():
    return pd.DataFrame(
        {
            "num1": [1, 2, 3, 4, 5, 6, 7, 8],
            "num2": [10, 11, 12, 13, 14, 15, 16, 17],
            "cat1": ["a", "a", "b", "b", "a", "b", "a", "b"],
            "target": [0, 0, 1, 1, 0, 1, 0, 1],
        }
    )


@pytest.fixture
def bool_df():
    return pd.DataFrame(
        {
            "num1": [1.0, 2.0, 3.0, 4.0],
            "flag": [True, False, True, False],
            "cat1": ["x", "y", "x", "y"],
            "target": [0, 1, 0, 1],
        }
    )


@pytest.fixture
def planner_profile():
    return {
        "shape": {"rows": 80, "cols": 120},
        "imbalance_ratio": 4.0,
        "missing_pct": {"a": 25.0, "b": 0.0},
        "feature_types": {
            "numeric": ["num1", "num2"],
            "categorical": ["cat1", "cat2"],
        },
        "n_unique_by_col": {"cat1": 3, "cat2": 60, "num1": 6, "num2": 6},
        "noise_ratio": 0.4,
        "skewness_by_col": {"num1": 1.2, "num2": 0.2},
    }