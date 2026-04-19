"""
Reflector Agent

The reflector evaluates execution results, identifies issues, and suggests improvements using sophisticated analysis that goes beyond simple threshold checks.
"""

from typing import Any, Dict, List, Tuple, Optional
from scipy.stats import wilcoxon
import numpy as np

from agents.memory import get_relevant_reflections, prioritize_suggestions_from_memory


def _estimate_problem_difficulty(
    dataset_profile: Dict[str, Any],
    best_metrics: Dict[str, Any],
) -> str:
    rows = dataset_profile.get("shape", {}).get("rows", 0) or 0
    cols = dataset_profile.get("shape", {}).get("cols", 0) or 0
    imbalance = float(dataset_profile.get("imbalance_ratio", 1.0) or 1.0)
    noise = float(dataset_profile.get("noise_ratio", 0.0) or 0.0)
    missing_pct = dataset_profile.get("missing_pct", {}) or {}
    missing_max = max(missing_pct.values(), default=0.0)
    f1_macro = float(best_metrics.get("f1_macro", 0.0) or 0.0)

    difficulty_score = 0
    if rows < 500:
        difficulty_score += 1
    if cols > 100:
        difficulty_score += 1
    if imbalance > 3:
        difficulty_score += 1
    if noise > 0.2:
        difficulty_score += 1
    if missing_max >= 20:
        difficulty_score += 1
    if f1_macro < 0.55:
        difficulty_score += 1

    if difficulty_score >= 4:
        return "hard"
    if difficulty_score >= 2:
        return "medium"
    return "easy"


def _estimate_confidence(
    issues: List[str],
    suggestions: List[List[str]],
    best_metrics: Dict[str, Any],
    relevant_reflections: List[Dict[str, Any]],
) -> float:
    score = 0.5

    if issues:
        score += 0.1
    if suggestions:
        score += 0.1

    f1_macro = float(best_metrics.get("f1_macro", 0.0) or 0.0)
    train_f1 = float(best_metrics.get("f1_train_macro", 0.0) or 0.0)
    gap = abs(train_f1 - f1_macro)

    if gap >= 0.15:
        score += 0.1

    if relevant_reflections:
        improved = sum(1 for item in relevant_reflections if item.get("outcome", {}).get("improved", False))
        score += min(0.2, 0.05 * improved)

    return max(0.1, min(0.95, score))


def _extract_failed_actions(relevant_reflections: List[Dict[str, Any]]) -> List[str]:
    failed: List[str] = []
    for item in relevant_reflections:
        outcome = item.get("outcome", {}) or {}
        if not outcome.get("improved", False):
            for action in item.get("actions_applied", []):
                if action not in failed:
                    failed.append(action)
    return failed


def reflect(
    dataset_profile: Dict[str, Any],
    evaluation: Dict[str, Any],
    reflection_memory: Optional[Dict[str, Any]]
) -> Dict[str, Any]:
    """
    Analyze results and generate reflection with issues and suggestions.
    """
    reflection_memory = reflection_memory or {}

    issues: List[str] = []
    suggestions: List[List[str]] = []

    best_metrics = evaluation.get("best_metrics", {}) or {}
    best_model = best_metrics.get("model")
    all_metrics = evaluation.get("all_metrics", []) or []
    classification_report = evaluation.get("classification_report", {}) or {}

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

    problem_difficulty = _estimate_problem_difficulty(dataset_profile, best_metrics)
    confidence = _estimate_confidence(issues, suggestions, best_metrics, relevant_reflections)
    failed_actions = _extract_failed_actions(relevant_reflections)

    target_by_difficulty = {
        "easy": 0.80,
        "medium": 0.70,
        "hard": 0.60,
    }
    target_score = target_by_difficulty[problem_difficulty]

    history = list(reflection_memory.get("history", []))
    remaining_replans = int(reflection_memory.get("remaining_replans", 1) or 0)
    min_expected_gain = float(reflection_memory.get("min_expected_gain", 0.01) or 0.01)

    f1_macro = float(best_metrics.get("f1_macro", 0.0) or 0.0)
    balanced_accuracy = float(best_metrics.get("balanced_accuracy", 0.0) or 0.0)
    current_score = max(f1_macro, balanced_accuracy)

    replan_recommended = bool(
        issues
        and suggestions
        and remaining_replans > 0
        and current_score < target_score
    )

    print(f"[Reflection] Suggestions: {suggestions}")

    return {
        "status": "needs_attention" if issues else "ok",
        "best_model": best_model,
        "issues": issues,
        "suggestions": suggestions,
        "replan_recommended": replan_recommended,

        # fields used by should_replan(...)
        "best_metrics": best_metrics,
        "confidence": confidence,
        "problem_difficulty": problem_difficulty,
        "target_score": target_score,
        "resource_budget": {
            "remaining_replans": remaining_replans,
            "min_expected_gain": min_expected_gain,
        },
        "history": history,
        "failed_actions": failed_actions,
        "memory_matches": [item.get("run_id") for item in relevant_reflections],
        "memory_matches_details": relevant_reflections,
    }


def should_replan(reflection: Dict[str, Any]) -> bool:
    """
    Decide whether to trigger replanning based on reflection.
    Adds debug prints explaining why replanning did NOT happen.
    """

    if not reflection:
        print("[Replan]  No reflection available → skip replanning")
        return False

    if not reflection.get("issues"):
        print("[Replan]  No issues detected → nothing to fix")
        return False

    best_metrics = reflection.get("best_metrics", {}) or {}
    f1_macro = float(best_metrics.get("f1_macro", 0.0) or 0.0)
    balanced_accuracy = float(best_metrics.get("balanced_accuracy", 0.0) or 0.0)
    score = max(f1_macro, balanced_accuracy)

    difficulty = str(reflection.get("problem_difficulty", "medium")).lower()
    target_by_difficulty = {
        "easy": 0.80,
        "medium": 0.70,
        "hard": 0.60,
    }
    target_score = float(reflection.get("target_score", target_by_difficulty.get(difficulty, 0.70)))

    recommended = bool(reflection.get("replan_recommended", False))

    suggestions = reflection.get("suggestions", []) or []
    confidence = float(reflection.get("confidence", 0.5) or 0.5)

    if not suggestions:
        print("[Replan]  No suggestions available → cannot improve")
        return False

    budget = reflection.get("resource_budget", {}) or {}
    remaining_replans = int(budget.get("remaining_replans", 1) or 0)
    min_expected_gain = float(budget.get("min_expected_gain", 0.01) or 0.01)

    if remaining_replans <= 0:
        print("[Replan]  Replan budget exhausted → stopping")
        return False

    # diminishing returns 
    history = reflection.get("history", []) or []
    recent_deltas = []

    for item in history[-3:]:
        outcome = item.get("outcome", {}) or {}
        delta = float(outcome.get("delta_f1_macro", 0.0) or 0.0)
        recent_deltas.append(delta)

    avg_recent_gain = sum(recent_deltas) / len(recent_deltas) if recent_deltas else None
    diminishing_returns = (
        avg_recent_gain is not None and avg_recent_gain < min_expected_gain
    )

    if diminishing_returns:
        print(f"[Replan] Diminishing returns detected → avg_gain={avg_recent_gain:.4f} < {min_expected_gain}")

    # memory check 
    failed_actions = set(reflection.get("failed_actions", []) or [])
    candidate_actions = []
    for group in suggestions:
        candidate_actions.extend(group)

    if candidate_actions and all(action in failed_actions for action in candidate_actions):
        print("[Replan]  All candidate actions previously failed → skipping")
        return False

    memory_matches = reflection.get("memory_matches_details", []) or []
    negative_memory_hits = 0
    positive_memory_hits = 0

    for item in memory_matches:
        outcome = item.get("outcome", {}) or {}
        if bool(outcome.get("improved", False)):
            positive_memory_hits += 1
        else:
            negative_memory_hits += 1

    memory_is_unfavorable = negative_memory_hits > positive_memory_hits and negative_memory_hits > 0

    if memory_is_unfavorable:
        print(f"[Replan] Memory unfavorable → {negative_memory_hits} failures vs {positive_memory_hits} successes")

    clearly_under_target = score < (target_score - 0.05)
    slightly_under_target = score < target_score

    # STOP condition
    if memory_is_unfavorable and diminishing_returns:
        print("[Replan]  Stop: bad memory + diminishing returns → no further replanning")
        return False

    # GO conditions 
    if clearly_under_target and confidence >= 0.4 and not diminishing_returns:
        print("[Replan] Replan: clearly under target + sufficient confidence")
        return True

    if recommended and slightly_under_target and r >= 0.6 and not memory_is_unfavorable:
        print("[Replan] Replan: reflector recommended + moderate confidence")
        return True

    if recommended and clearly_under_target and remaining_replans > 0 and not diminishing_returns:
        print("[Replan] Replan: recommended + clearly under target")
        return True

    # FINAL FALLBACK 
    print(
        "[Replan]  No condition met → skip replanning | "
        f"score={score:.3f}, target={target_score}, "
        f"confidence={confidence:.2f}, recommended={recommended}"
    )
    return False


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
            - actions_applied
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


def baseline_comparison_successful(
        all_metrics: List[Dict[str, Any]],
        best_metrics: Dict[str, Any],
        should_skip: bool = False
    ) -> bool:
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
    return True


def detect_overfitting(
        best_metrics: Dict[str, Any],
        issues: List[str],
        suggestions: List[List[str]],
        should_skip: bool = False
    ) -> bool:
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
    if should_skip:
        print("[Reflection] Skipping performance acceptability check due to insufficient improvement suggestions to improve model performance.")
        return True

    f1_macro = float(best_metrics.get("f1_macro", 0.0))

    if f1_macro >= 0.70:
        return True

    print("[Reflection] Model performance is not acceptable. -> Consider hyperparameters tuning.")
    suggestions.append(["P4A_tune_hyperparameters", "P8_skip_tuning"])
    return False


def detect_data_quality_issues(issues: List[str], suggestions: List[List[str]], dataset_profile, note: str) -> None:
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

    if any(value >= 0.05 for value in outlier_ratio.values()):
        issues.append("data_quality: numeric feature outliers")
        suggestions.append(["P2A4_handle_numerical_outliers", "P8_skip_data_quality_step"])
        print(f"[Reflection] {note}: Data quality issue detected: numeric feature outliers. -> Consider handling numeric outliers")

    max_skewness = max(dataset_profile.get("skewness_by_col", {}).values(), default=0)

    if max_skewness >= 1:
        issues.append("data_quality: skewed numeric features")
        suggestions.append(["P2A6_optimize_skewness", "P8_skip_data_quality_step"])
        print(f"[Reflection] {note}: Data quality issue detected: skewed numeric features. -> Consider optimizing skewness")