from agents.reflector import (
    apply_replan_strategy,
    baseline_comparison_successful,
    detect_data_quality_issues,
    per_class_analysis_successful,
    reflect,
    should_replan,
)


def test_apply_replan_strategy_returns_new_plan_and_actions():
    plan = ["P3B_select_models"]
    reflection = {
        "suggestions": [
            ["P4A_tune_hyperparameters", "P3A4_regularization"],
            ["P3A_SMOTE"],
        ]
    }

    new_plan, new_profile, actions = apply_replan_strategy(plan, {"shape": {"rows": 10, "cols": 2}}, reflection)

    assert "P4A_tune_hyperparameters" in new_plan
    assert "P3A_SMOTE" in new_plan
    assert actions == ["P4A_tune_hyperparameters", "P3A_SMOTE"]
    assert new_profile["shape"]["rows"] == 10


def test_baseline_comparison_successful_true():
    all_metrics = [
        {"model": "DummyMostFrequent", "balanced_accuracy": 0.50},
        {"model": "RandomForest", "balanced_accuracy": 0.70},
    ]
    best_metrics = {"model": "RandomForest", "balanced_accuracy": 0.70}

    assert baseline_comparison_successful(all_metrics, best_metrics) is True


def test_per_class_analysis_successful_false_with_imbalance():
    issues = []
    suggestions = []
    report = {
        "class_0": {"f1-score": 0.90, "precision": 0.95, "recall": 0.50},
        "class_1": {"f1-score": 0.50, "precision": 0.60, "recall": 0.90},
    }

    ok = per_class_analysis_successful(report, issues, suggestions)

    assert ok is False
    assert issues
    assert suggestions


def test_detect_data_quality_issues_adds_suggestions():
    issues = []
    suggestions = []
    profile = {
        "feature_types": {"categorical": ["id_col"]},
        "n_unique_by_col": {"id_col": 100},
        "shape": {"rows": 100, "cols": 2},
        "outlier_ratio_by_col": {"x": 0.10},
        "skewness_by_col": {"x": 1.2},
    }

    detect_data_quality_issues(issues, suggestions, profile, note="failed")

    assert len(issues) == 3
    assert any("P2A3_optimize_categorical_encoding" in s for s in suggestions)
    assert any("P2A4_handle_numerical_outliers" in s for s in suggestions)
    assert any("P2A6_optimize_skewness" in s for s in suggestions)


def test_should_replan_false_when_no_issues(capsys):
    result = should_replan({"issues": []})
    captured = capsys.readouterr()

    assert result is False
    assert "No issues detected" in captured.out


def test_should_replan_false_when_budget_exhausted(capsys):
    reflection = {
        "issues": ["overfitting"],
        "best_metrics": {"f1_macro": 0.4, "balanced_accuracy": 0.4},
        "problem_difficulty": "medium",
        "target_score": 0.7,
        "replan_recommended": True,
        "suggestions": [["P3A4_regularization"]],
        "confidence": 0.8,
        "resource_budget": {"remaining_replans": 0, "min_expected_gain": 0.01},
        "history": [],
        "failed_actions": [],
        "memory_matches_details": [],
    }

    result = should_replan(reflection)
    captured = capsys.readouterr()

    assert result is False
    assert "budget exhausted" in captured.out.lower()


def test_should_replan_true_when_clearly_under_target():
    reflection = {
        "issues": ["underfitting"],
        "best_metrics": {"f1_macro": 0.4, "balanced_accuracy": 0.45},
        "problem_difficulty": "medium",
        "target_score": 0.7,
        "replan_recommended": False,
        "suggestions": [["P3A5_increase_model_complexity"]],
        "confidence": 0.7,
        "resource_budget": {"remaining_replans": 2, "min_expected_gain": 0.01},
        "history": [],
        "failed_actions": [],
        "memory_matches_details": [],
    }

    assert should_replan(reflection) is True


def test_reflect_populates_should_replan_fields():
    dataset_profile = {
        "shape": {"rows": 100, "cols": 10},
        "imbalance_ratio": 1.0,
        "noise_ratio": 0.4,
        "missing_pct": {"a": 0.0},
        "feature_types": {"categorical": [], "numeric": ["x"]},
        "n_unique_by_col": {},
        "outlier_ratio_by_col": {"x": 0.0},
        "skewness_by_col": {"x": 1.2},
    }
    evaluation = {
        "best_metrics": {
            "model": "LogisticRegression",
            "balanced_accuracy": 0.45,
            "f1_macro": 0.40,
            "f1_train_macro": 0.60,
            "cv_f1_scores": [0.4, 0.41],
        },
        "all_metrics": [
            {"model": "LogisticRegression", "cv_f1_scores": [0.4, 0.41]},
            {"model": "RandomForest", "cv_f1_scores": [0.39, 0.40]},
        ],
        "classification_report": {},
    }

    result = reflect(dataset_profile, evaluation, reflection_memory={"history": [], "remaining_replans": 2})

    assert "confidence" in result
    assert "resource_budget" in result
    assert "history" in result
    assert "memory_matches_details" in result