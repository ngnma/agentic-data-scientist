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


def reflect(
    dataset_profile: Dict[str, Any],
    evaluation: Dict[str, Any],
    all_metrics: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """
    Analyze results and generate reflection with issues and suggestions.
    
    This is a basic implementation. Students should extend this significantly.
    
    Args:
        dataset_profile: Dataset characteristics
        evaluation: Best model's metrics
        all_metrics: Metrics for all trained models
    
    Returns:
        Dictionary with:
            - status: str ("ok" or "needs_attention")
            - best_model: str (model name)
            - issues: List[str] (identified problems)
            - suggestions: List[str] (improvement recommendations)
            - replan_recommended: bool (should we replan?)
    
    TODO for students:
    - Implement statistical tests (paired t-tests, Wilcoxon tests)
    - Add per-class performance analysis
    - Detect overfitting vs underfitting
    - Analyze confusion matrix patterns
    - Check for data quality issues
    - Prioritize suggestions by expected impact
    - Learn which suggestions work from memory
    """
    
    best_model = evaluation.get("model")
    bal_acc = float(evaluation.get("balanced_accuracy", 0.0))
    f1_macro = float(evaluation.get("f1_macro", 0.0))
    f1_train_macro = float(evaluation.get("f1_train_macro", 0.0))
    imb = float(dataset_profile.get("imbalance_ratio") or 1.0)
    
    issues: List[str] = []
    suggestions: List[str] = []


    
    # Basic comparison with dummy baseline
    dummy = next((m for m in all_metrics if "Dummy" in m.get("model", "")), None)
    
    if dummy is not None:
        dummy_ba = float(dummy.get("balanced_accuracy", 0.0))
        improvement = bal_acc - dummy_ba
        
        # TODO: Make this more sophisticated
        # Consider: confidence intervals, effect sizes, etc.
        if improvement < 0.05:
            issues.append(
                f"Best model only {improvement:.3f} better than baseline. "
                "Weak signal or pipeline issues."
            )
            suggestions.append(
                "Check for target leakage, verify target quality, "
                "or improve feature engineering."
            )


    # ------------------- Reflection logic starts here
    best_model = evaluation.get("model")

    if significant_tests_succesfull(evaluation):
        print(f"[Reflection] Statistical tests indicate significant differences between models. Model {best_model} is significantly better than all others.")

        if baseline_comparison_successful(evaluation):
            print("[Reflection] Best model significantly outperforms baseline. Consider deeper per-class analysis.")

        #     if per_class_analysis_successful(evaluation):
        #         print("[Reflection] Per-class performance looks balanced. Consider model optimization and further tuning.")
        #         # TODO: Go to S3 (model optimization and tuning)
        #     else:
        #         print("[Reflection] Per-class analysis indicates specific class issues. Consider class-specific strategies.")
        #         # TODO: Go to S2 (per-class performance analysis) and Handle class-specific issues (e.g., class_weight, threshold tuning, SMOTE)

        else:
            print("[Reflection] Best model does not significantly outperform baseline. Consider data quality or feature issues.")
            # TODO: Go to S4 (data / feature issues)
    else:
        print("[Reflection] No significant differences detected between models. Consider data quality or feature issues.")
        # TODO: Go to S4 (data / feature issues)
    # ------------------- Reflection logic ends here




    
    # TODO: Add more sophisticated checks
    
    # Check F1 score
    # TODO: Make threshold adaptive based on problem difficulty
    if f1_macro < 0.60:
        issues.append("Macro F1 score is modest (<0.60).")
        suggestions.append(
            "Try different models, tune hyperparameters, "
            "or improve preprocessing."
        )
    
    # TODO: Add imbalance-specific analysis
    if imb >= 3.0:
        suggestions.append(
            "Imbalance detected: consider class_weight, "
            "threshold tuning, or SMOTE."
        )
    
    # TODO: Add checks for:
    # - Model diversity (are all models performing similarly?)
    # - Per-class performance (which classes are problematic?)
    # - Precision-recall tradeoff
    # - High-cardinality categorical features
    # - Feature importance patterns
    # - Learning curves (overfitting/underfitting)

    # Add overfitting check
    # if f1_train_macro - f1_macro > 0.10:
    if f1_train_macro - f1_macro > 0.01: # just for test TODO: CLEANUP
        issues.append("Potential overfitting detected (train F1 much higher than test).")
        suggestions.append(
            ["P3A_regularization", "P3A_feature_selection", "P3A_simpler_models"]
        )
    
    # Determine status
    status = "needs_attention" if issues else "ok"
    
    # Simple replanning trigger
    # TODO: Make this more sophisticated
    replan_recommended = bool(issues and f1_macro < 0.60)
    replan_recommended = True # just for test TODO: CLEANUP

    print(f"[Reflection] Suggestions: {suggestions}")
    
    return {
        "status": status,
        "best_model": best_model,
        "issues": issues,
        "suggestions": suggestions,
        "replan_recommended": replan_recommended,
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
) -> Tuple[List[str], Dict[str, Any]]:
    """
    Modify the plan and dataset profile based on reflection.
    
    This is a very basic implementation. Students should make this sophisticated.
    
    Args:
        plan: Current execution plan
        dataset_profile: Current dataset profile
        reflection: Reflection results
    
    Returns:
        Tuple of (modified_plan, modified_profile)
    
    TODO for students:
    - Implement specific strategies for specific issues
    - Add preprocessing steps based on identified problems
    - Modify model selection based on performance patterns
    - Adjust hyperparameters
    - Try ensemble methods
    - Implement different replan strategies (aggressive, conservative)
    """
    
    # Copy to avoid modifying originals
    new_plan = list(plan)
    new_profile = dict(dataset_profile)

    # add suggestions to the plan
    suggestions_list = reflection.get("suggestions", [])
    for suggestion_list in suggestions_list:
        for suggestion in suggestion_list:
            if suggestion not in(new_plan):
                new_plan.append(suggestion)
                break

    
    # Basic strategy: add a note
    # # TODO: Implement actual strategy changes
    # notes = list(new_profile.get("notes", []))
    # notes.append("Replan: adjusting strategy after reflection.")
    # new_profile["notes"] = notes
    
    # new_plan.append("replan_attempt")
    
    # TODO: Implement sophisticated replan strategies:
    # - If low performance: try ensemble methods
    # - If imbalance issues: add SMOTE or adjust thresholds
    # - If overfitting: add regularization
    # - If underfitting: increase model complexity
    # - If feature issues: add feature engineering steps

    # sort new_plan alfabetically (P1A, P1B, P2A, P2B, etc.) to ensure consistent execution order
    new_plan.sort()
    
    return new_plan, new_profile


# TODO: Add helper functions for reflection
# def compare_models_statistically(...):
# def analyze_per_class_performance(...):
# def detect_overfitting(...):
# def detect_data_quality_issues(...):
# def prioritize_suggestions(...):
# def generate_explanation(...):


from typing import List, Dict, Any, Optional
from scipy.stats import wilcoxon
import numpy as np


def baseline_comparison_successful(evaluation: Dict[str, Any]) -> bool:
    """Check if best model significantly outperforms dummy baseline."""

    all_metrics = evaluation.get("all_metrics", [])
    bal_acc = float(evaluation.get("balanced_accuracy", 0.0))

    dummy = next((m for m in all_metrics if "Dummy" in m.get("model", "")), None)
    
    if dummy is not None:
        dummy_ba = float(dummy.get("balanced_accuracy", 0.0))
        improvement = bal_acc - dummy_ba
        if improvement > 0.05:
            return True
    return False


def significant_tests_succesfull(
    evaluation: Dict[str, Any]
) -> bool:
    """
    Return best_model_name if it is significantly better than all other models
    using the Wilcoxon signed-rank test on cv_f1_scores.
    Otherwise return None.
    """
    best_model_name = evaluation.get("model")
    all_metrics = evaluation.get("all_metrics", [])

    if not all_metrics or len(all_metrics) < 2:
        return False

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

        # Wilcoxon requires paired samples with equal length
        if len(best_scores) != len(other_scores):
            return False

        # Best model must also have a higher mean score
        if best_scores.mean() <= other_scores.mean():
            return False

        diffs = best_scores - other_scores

        # No difference -> not significantly better
        if np.allclose(diffs, 0.0):
            return False

        try:
            # one-sided test: best model > other model
            _, p_value = wilcoxon(diffs, alternative="greater")
        except ValueError:
            return False

        if p_value >= 0.05:
            return False

    return True