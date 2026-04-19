from agents.reflector import (
    baseline_comparison_successful,
    per_class_analysis_successful,
    detect_overfitting,
    detect_underfitting,
    acceptable_performance,
    apply_replan_strategy,
)


def test_baseline_comparison_successful_when_best_beats_dummy():
    all_metrics = [
        {"model": "DummyMostFrequent", "balanced_accuracy": 0.50},
        {"model": "RandomForest", "balanced_accuracy": 0.70},
    ]
    best_metrics = {"model": "RandomForest", "balanced_accuracy": 0.70}

    assert baseline_comparison_successful(all_metrics, best_metrics) is True


def test_per_class_analysis_detects_imbalance_and_threshold_issue():
    issues = []
    suggestions = []
    report = {
        "class_0": {"f1-score": 0.90, "precision": 0.95, "recall": 0.50},
        "class_1": {"f1-score": 0.50, "precision": 0.60, "recall": 0.90},
    }

    ok = per_class_analysis_successful(report, issues, suggestions)

    assert ok is False
    assert len(issues) >= 1
    assert len(suggestions) >= 1


def test_detect_overfitting_adds_issue_and_suggestion():
    issues = []
    suggestions = []
    best_metrics = {"f1_train_macro": 0.95, "f1_macro": 0.70}

    assert detect_overfitting(best_metrics, issues, suggestions) is True
    assert "overfitting" in issues


def test_detect_underfitting_adds_issue_and_suggestion():
    issues = []
    suggestions = []
    best_metrics = {"f1_train_macro": 0.50, "f1_macro": 0.45}

    assert detect_underfitting(best_metrics, issues, suggestions) is True
    assert "underfitting" in issues


def test_acceptable_performance_false_when_f1_low():
    suggestions = []
    best_metrics = {"f1_macro": 0.55}

    assert acceptable_performance(best_metrics, suggestions) is False
    assert suggestions


def test_apply_replan_strategy_appends_sorted_unique_steps():
    plan = ["P3B_select_models"]
    reflection = {"suggestions": [["P4A_tune_hyperparameters"], ["P3A_SMOTE"]]}

    new_plan, new_profile = apply_replan_strategy(plan, {}, reflection)

    assert "P4A_tune_hyperparameters" in new_plan
    assert "P3A_SMOTE" in new_plan
    assert new_plan == sorted(new_plan)