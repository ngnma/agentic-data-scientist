from agents.planner import create_plan


def test_create_plan_adds_expected_steps(planner_profile):
    plan = create_plan(planner_profile, internal_memory={})

    assert "P3A1_select_additional_models" in plan
    assert "P3A_imb_class_weight" in plan
    assert "P3A4_regularization" in plan
    assert "P2A2_apply_categorical_encoding" in plan
    assert "P2A0_handle_missing_values" in plan
    assert "P3A_feature_selection" in plan
    assert "P2A5_handle_skewness" in plan


def test_create_plan_returns_sorted_unique_steps():
    profile = {
        "shape": {"rows": 50, "cols": 10},
        "imbalance_ratio": 5.0,
        "missing_pct": {"x": 30.0},
        "feature_types": {"numeric": [], "categorical": []},
        "n_unique_by_col": {},
        "noise_ratio": 0.5,
        "skewness_by_col": {},
    }

    plan = create_plan(profile, internal_memory={})

    assert plan == sorted(set(plan))