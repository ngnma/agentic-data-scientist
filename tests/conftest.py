import pandas as pd
import pytest

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


@pytest.fixture
def sample_df():
    return pd.DataFrame(
        {
            "num1": [1, 2, 3, 4, 100, 6],
            "num2": [10, 11, 12, 13, 14, 15],
            "cat1": ["a", "a", "b", "b", "c", "c"],
            "target": [0, 0, 1, 1, 1, 0],
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