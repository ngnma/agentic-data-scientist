"""
Reflector Agent - Students must extend this significantly

The reflector evaluates execution results, identifies issues, and suggests improvements.
Your task is to implement sophisticated analysis that goes beyond simple threshold checks.

TODO: Extend this module with:
1. Statistical significance testing between models
2. Per-class performance analysis
3. Root cause diagnosis (data quality, preprocessing, model issues)
4. Actionable, prioritized suggestions
5. Learning from past reflections (meta-learning)
"""

from typing import Any, Dict, List, Tuple, Optional
from scipy.stats import wilcoxon
import numpy as np

from agents.memory import get_relevant_reflections, prioritize_suggestions_from_memory


def reflect(
    dataset_profile: Dict[str, Any],
    evaluation: Dict[str, Any],
    reflection_memory: Optional[Dict[str, Any]]
) -> Dict[str, Any]:
    """
    Analyze results and generate reflection with issues and suggestions.
    
    This is a basic implementation. Students should extend this significantly.
    
    Args:
        dataset_profile: Dataset characteristics
        evaluation: Dictionary contains Best model's metrics and all models' metrics, classification report, etc.
    
    Returns:
        Dictionary with:
            - status: str ("ok" or "needs_attention")
            - best_model: str (model name)
            - issues: List[str] (identified problems)
            - suggestions: List[str] (improvement recommendations)
            - replan_recommended: bool (should we replan?)
    
    TODO for students:
    - Check for data quality issues
    - Prioritize suggestions by expected impact
    - Learn which suggestions work from memory
    """
    reflection_memory = reflection_memory or {}

    issues: List[str] = []
    suggestions: List[List[str]] = []

    f1_macro = float(evaluation.get("f1_macro", 0.0))
    best_metrics = evaluation.get("best_metrics", {})
    best_model = best_metrics.get("model")
    all_metrics = evaluation.get("all_metrics", [])
    classification_report = evaluation.get("classification_report", {})

    skip_data_quality = reflection_memory.get("skip_data_quality", False)
    skip_tuning = reflection_memory.get("skip_tuning", False)
    skip_handle_overfitting = reflection_memory.get(
        "skip_handle_overfitting",
        reflection_memory.get("skip_overfitting", False),
    )
    skip_handle_underfitting = reflection_memory.get(
        "skip_handle_underfitting",
        reflection_memory.get("skip_underfitting", False),
    )
    skip_per_class_analysis = reflection_memory.get("skip_per_class_analysis", False)

    # ------------------- Reflection logic starts here
    if significant_tests_succesfull(evaluation, suggestions, skip_data_quality):
        print(f"[Reflection] Statistical tests successful. Model {best_model} is significantly better than all others. -> Consider baseline comparison.")

        if baseline_comparison_successful(all_metrics, best_metrics, skip_data_quality):
            print("[Reflection] Baseline comparison successful. Best model significantly outperforms baseline. -> Consider deeper per-class analysis.")

            if per_class_analysis_successful(classification_report, issues, suggestions, skip_per_class_analysis):
                print("[Reflection] Per-class performance successful. -> Consider model optimization and tuning.")

                if not detect_overfitting(best_metrics, issues, suggestions, skip_handle_overfitting):
                    print("[Reflection] No overfitting detected. -> Check underfitting.")

                if not detect_underfitting(best_metrics, issues, suggestions, skip_handle_underfitting):
                    print("[Reflection] No underfitting detected. -> Check model performance")

                if acceptable_performance(best_metrics, suggestions, skip_tuning):
                    print("[Reflection] Model performance is acceptable. -> Finish.")
                else:
                    print("[Reflection] Model performance is not acceptable. -> not implemented.")
        else:
            print("[Reflection] Baseline comparison failed. Best model does not significantly outperform baseline. -> Consider data quality or feature issues.")
            detect_data_quality_issues(issues, suggestions, dataset_profile, note="Baseline comparison failed.")
    else:
        print("[Reflection] Statistical tests failed. No significant differences between models. -> Consider data quality or feature issues.")
        detect_data_quality_issues(issues, suggestions, dataset_profile, note="Statistical tests failed.")

    relevant_reflections = get_relevant_reflections(
        reflection_memory=reflection_memory,
        issues=issues,
        dataset_profile=dataset_profile,
        best_model=best_model,
        top_k=3,
    )
    suggestions = prioritize_suggestions_from_memory(suggestions, relevant_reflections)

    # ------------------- Reflection logic ends here

    status = "needs_attention" if issues else "ok"

    replan_recommended = bool(issues and f1_macro < 0.60)
    replan_recommended = True  # just for test TODO: CLEANUP

    print(f"[Reflection] Suggestions: {suggestions}")

    return {
        "status": status,
        "best_model": best_model,
        "issues": issues,
        "suggestions": suggestions,
        "replan_recommended": replan_recommended,
        "memory_matches": [item.get("run_id") for item in relevant_reflections],
    }


def should_replan(reflection: Dict[str, Any]) -> bool:
    """
    Decide whether to trigger replanning based on reflection.
    
    This is a simple policy. Students should implement more sophisticated logic.
    
    TODO for students:
    - Consider multiple factors (performance, confidence, resource budget)
    - Implement diminishing returns detection
    - Use memory to avoid repeating failed strategies
    - Set adaptive thresholds based on problem difficulty
    """
    return bool(reflection.get("replan_recommended", False))


def apply_replan_strategy(
    plan: List[str],
    dataset_profile: Dict[str, Any],
    reflection: Dict[str, Any],
) -> Tuple[List[str], Dict[str, Any], List[str]]:
    """
    Modify the plan and dataset profile based on reflection.

    Returns:
        Tuple of:
            - modified_plan
            - modified_profile
            - actions_applied (the exact actions selected from reflection)
    """

    new_plan = list(plan)
    new_profile = dict(dataset_profile)
    actions_applied: List[str] = []

    suggestions_list = reflection.get("suggestions", [])
    planned = set(new_plan)

    for suggestion_list in suggestions_list:
        for suggestion in suggestion_list:
            if suggestion not in planned:
                new_plan.append(suggestion)
                actions_applied.append(suggestion)
                planned.add(suggestion)
                break

    new_plan.sort()

    return new_plan, new_profile, actions_applied


# TODO: Add helper functions for reflection
# def prioritize_suggestions(...):
# def generate_explanation(...):


def baseline_comparison_successful(
        all_metrics: List[Dict[str, Any]],
        best_metrics: Dict[str, Any],
        should_skip: bool = False
    ) -> bool:
    """Check if best model significantly outperforms dummy baseline."""

    if should_skip:
        print("[Reflection] Skipping baseline comparison due to insufficient data quality improvement suggestions to improve model performance.")
        return True

    bal_acc = float(best_metrics.get("balanced_accuracy", 0.0))

    dummy = next((m for m in all_metrics if "Dummy" in m.get("model", "")), None)

    if dummy is not None:
        dummy_ba = float(dummy.get("balanced_accuracy", 0.0))
        improvement = bal_acc - dummy_ba
        if improvement > 0.05:
            return True
    return False


def significant_tests_succesfull(
    evaluation: Dict[str, Any],
    suggestions: List[List[str]],
    should_skip: bool = False
) -> bool:
    """
    Return best_model_name if it is significantly better than all other models
    using the Wilcoxon signed-rank test on cv_f1_scores.
    Otherwise return None.
    """
    best_model_name = evaluation.get("best_metrics", {}).get("model")
    all_metrics = evaluation.get("all_metrics", [])

    if should_skip:
        print("[Reflection] Skipping significant testing due to insufficient data quality improvement suggestions to improve model performance.")
        suggestions.append(["P3A3_choose_best_model"])
        return True

    if not all_metrics:
        return False

    if len(all_metrics) < 2:
        return True

    best_model_metrics = next(
        (
            m for m in all_metrics
            if m.get("model") == best_model_name
            and isinstance(m.get("cv_f1_scores"), list)
            and len(m.get("cv_f1_scores")) >= 2
        ),
        None,
    )

    if best_model_metrics is None:
        return False

    best_scores = np.asarray(best_model_metrics["cv_f1_scores"], dtype=float)

    for other_metrics in all_metrics:
        other_name = other_metrics.get("model")

        if other_name == best_model_name:
            continue

        other_scores_list = other_metrics.get("cv_f1_scores")
        if not isinstance(other_scores_list, list) or len(other_scores_list) < 2:
            return False

        other_scores = np.asarray(other_scores_list, dtype=float)

        if len(best_scores) != len(other_scores):
            return False

        if best_scores.mean() <= other_scores.mean():
            return False

        diffs = best_scores - other_scores

        if np.allclose(diffs, 0.0):
            return False

        try:
            _, p_value = wilcoxon(diffs, alternative="greater")
        except ValueError:
            return False

        if p_value >= 0.05:
            return False

    suggestions.append(["P3A3_choose_best_model"])
    return True


def per_class_analysis_successful(
        classification_report: Dict[str, Any],
        issues: List[str],
        suggestions: List[List[str]],
        should_skip: bool = False
    ) -> bool:
    """
    Analyze per-class performance and return True if it looks balanced.
    """

    if should_skip:
        print("[Reflection] Skipping per-class performance analysis due to insufficient improvement suggestions to improve model performance.")
        return True

    class_f1 = [v["f1-score"] for k, v in classification_report.items() if k.startswith("class_")]
    class_precision = [v["precision"] for k, v in classification_report.items() if k.startswith("class_")]
    class_recall = [v["recall"] for k, v in classification_report.items() if k.startswith("class_")]

    if not class_f1:
        return True

    change = False
    sug: List[str] = []

    if max(class_f1) - min(class_f1) > 0.20:
        issues.append("class imbalance")
        sug.extend(["P3A_imb_class_weight", "P3A_SMOTE"])
        print("[Reflection] Per-class performance fails due to class imbalance detected. -> Consider class_weight or SMOTE.")
        change = True

    if any(class_precision[i] > 0.85 and class_recall[i] < 0.60 for i in range(len(class_precision))):
        issues.append("High false negatives.")
        sug.extend(["P4A_Lower_decision_threshold"])
        print("[Reflection] Per-class performance fails due to low recall. -> Decrease decision threshold.")
        change = True

    elif any(class_recall[i] > 0.85 and class_precision[i] < 0.60 for i in range(len(class_precision))):
        issues.append("High false positives.")
        sug.extend(["P4A_Higher_decision_threshold"])
        print("[Reflection] Per-class performance fails due to low precision. -> Increase decision threshold.")
        change = True

    if change:
        sug.extend(["P8_skip_per_class_analysis"])
        suggestions.append(sug)
        return False
    else:
        return True


def detect_overfitting(
        best_metrics: Dict[str, Any],
        issues: List[str],
        suggestions: List[List[str]],
        should_skip: bool = False
    ) -> bool:
    """
    Detect overfitting based on train and test F1 scores.
    """
    train_f1 = best_metrics.get("f1_train_macro", 0.0)
    macro_f1 = best_metrics.get("f1_macro", 0.0)

    if should_skip:
        print("[Reflection] Skipping overfitting detection due to insufficient improvement suggestions to handle overfitting.")
        return False

    if train_f1 >= 0.7 and (train_f1 - macro_f1) >= 0.15:
        print("[Reflection] Overfitting detected. -> Consider regularization or simpler models.")
        issues.append("overfitting")
        suggestions.append(["P3A6_decrease_model_complexity", "P3A_feature_selection", "P3A2_simpler_models", "P8_skip_handle_overfirring"])
        return True

    return False


def detect_underfitting(
        best_metrics: Dict[str, Any],
        issues: List[str],
        suggestions: List[List[str]],
        should_skip: bool = False
    ) -> bool:
    """
    Detect underfitting based on train and test F1 scores.
    """
    if should_skip:
        print("[Reflection] Skipping underfitting detection due to insufficient improvement suggestions to handle underfitting.")
        return False

    train_f1 = best_metrics.get("f1_train_macro", 0.0)
    macro_f1 = best_metrics.get("f1_macro", 0.0)

    if train_f1 < 0.7 and macro_f1 < 0.7:
        print("[Reflection] Underfitting detected. -> Consider more complex models or removing regularization.")
        issues.append("underfitting")
        suggestions.append(["P3A5_increase_model_complexity", "P8_skip_handle_underfirring"])
        return True

    return False


def acceptable_performance(
        best_metrics: Dict[str, Any],
        suggestions: List[List[str]],
        should_skip: bool = False
    ) -> bool:
    """
    Check if model performance is acceptable based on balanced accuracy and F1 score.
    """
    if should_skip:
        print("[Reflection] Skipping performance acceptability check due to insufficient improvement suggestions to improve model performance.")
        return True

    f1_macro = float(best_metrics.get("f1_macro", 0.0))

    if f1_macro >= 0.70:
        return True
    else:
        print("[Reflection] Model performance is not acceptable. -> Consider hyperparameters tuning.")
        suggestions.append(["P4A_tune_hyperparameters", "P8_skip_tuning"])
        return False


def detect_data_quality_issues(issues: List[str], suggestions: List[List[str]], dataset_profile, note: str) -> None:
    """
    Placeholder for data quality issue detection logic.
    In a real implementation, this would analyze the dataset profile for issues like:
    - High missing value percentages
    - Extreme class imbalance
    - High cardinality categorical features
    - Outliers or noisy data
    - Feature importance patterns indicating irrelevant features
    """

    categorical_cols = dataset_profile.get("feature_types", {}).get("categorical", [])
    n_unique = dataset_profile.get("n_unique_by_col", {})
    rows = dataset_profile["shape"]["rows"]

    has_medium_cardinal_col = any(n_unique[c] > 15 for c in categorical_cols)
    has_constant_col = any(n_unique[c] == 1 for c in categorical_cols)
    has_id_col = any(n_unique[c] == rows for c in categorical_cols)

    if has_medium_cardinal_col or has_constant_col or has_id_col:
        issues.append("data_quality: categorical feature issues")
        suggestions.append(["P2A3_optimize_categorical_encoding", "P8_skip_data_quality_step"])
        print(f"[Reflection] {note}: Data quality issue detected: categorical feature issues. -> Consider optimizing encoding")

    outlier_ratio = dataset_profile.get("outlier_ratio_by_col", {})

    if any(value >= 0.05 for key, value in outlier_ratio.items()):
        issues.append("data_quality: numeric feature outliers")
        suggestions.append(["P2A4_handle_numerical_outliers", "P8_skip_data_quality_step"])
        print(f"[Reflection] {note}: Data quality issue detected: numeric feature outliers. -> Consider handling numeric outliers")

    max_skewness = max(dataset_profile.get("skewness_by_col", {}).values(), default=0)

    if max_skewness >= 1:
        issues.append("data_quality: skewed numeric features")
        suggestions.append(["P2A6_optimize_skewness", "P8_skip_data_quality_step"])
        print(f"[Reflection] {note}: Data quality issue detected: skewed numeric features. -> Consider optimizing skewness")