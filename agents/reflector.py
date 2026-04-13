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
    - Check for data quality issues
    - Prioritize suggestions by expected impact
    - Learn which suggestions work from memory
    """
    
        
    issues: List[str] = []
    suggestions: List[str] = []

    f1_macro = float(evaluation.get("f1_macro", 0.0))
    best_metrics = evaluation.get('best_metrics')
    best_model = best_metrics.get("model")
    all_metrics = evaluation.get("all_metrics", [])
    classification_report = evaluation.get("classification_report", {})

    # ------------------- Reflection logic starts here


    if significant_tests_succesfull(evaluation):
        print(f"[Reflection] Statistical tests successful. Model {best_model} is significantly better than all others. -> Consider baseline comparison.")

        if baseline_comparison_successful(all_metrics, best_metrics):
            print("[Reflection] Baseline comparison successful. Best model significantly outperforms baseline. -> Consider deeper per-class analysis.")

            if per_class_analysis_successful(classification_report, issues, suggestions):
                print("[Reflection] Per-class performance successful. -> Consider model optimization and tuning.")
                
                if not detect_overfitting(best_metrics, issues, suggestions):
                    print("[Reflection] No overfitting detected. -> Check underfitting.")

                if not detect_underfitting(best_metrics, issues, suggestions):
                    print("[Reflection] No underfitting detected. -> Check model performance")
                
                if acceptable_performance(best_metrics):
                    print("[Reflection] Model performance is acceptable. -> Finish.") # exit the reflect
                else:
                    pass
            else:
                # Do nothing. per_class_analysis_successful already adds issues and suggestions if needed. Also the print statements in per_class_analysis_successful already print the reason for failure and suggestions.
                pass

        else:
            print("[Reflection] Baseline comparison failed. Best model does not significantly outperform baseline. -> Consider data quality or feature issues.")
            # TODO: Go to S4 (data / feature issues)
    else:
        print(f"[Reflection] Statistical tests failed. No significant differences between models. -> Consider data quality or feature issues.")
        # TODO: Go to S4 (data / feature issues)

    # ------------------- Reflection logic ends here

    

    
    # TODO: S4 - Data quality and feature issues
    # - High-cardinality categorical features
    # - Feature importance patterns


    
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

    # sort new_plan alfabetically (P1A, P1B, P2A, P2B, etc.) to ensure consistent execution order
    new_plan.sort()
    
    return new_plan, new_profile


# TODO: Add helper functions for reflection
# def detect_data_quality_issues(...):
# def prioritize_suggestions(...):
# def generate_explanation(...):



def baseline_comparison_successful(
        all_metrics: List[Dict[str, Any]], 
        best_metrics: Dict[str, Any]
    ) -> bool:
    """Check if best model significantly outperforms dummy baseline."""

    bal_acc = float(best_metrics.get("balanced_accuracy", 0.0))

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
    best_model_name = evaluation.get("best_metrics").get("model")
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

def per_class_analysis_successful(
        classification_report: Dict[str, Any], 
        issues: List[str], 
        suggestions: List[str]
    ) -> bool:
    """
    Analyze per-class performance and return True if it looks balanced.
    """

    # Extract per-class metrics
    class_f1 = [v["f1-score"] for k, v in classification_report.items() if k.startswith("class_")]
    class_precision = [v["precision"] for k, v in classification_report.items() if k.startswith("class_")]
    class_recall = [v["recall"] for k, v in classification_report.items() if k.startswith("class_")]

    # 1. Analyze class imbalance impact on performance
    if max(class_f1) - min(class_f1) > 0.20:
        issues.append("class imbalance")
        suggestions.append('P3A_class_weights','P3A_SMOTE') # maybe the model is changed after first planning step. Do both SMOTE and class_weight in same reflection cycle.
        print("[Reflection] Per-class performance fails due to class imbalance detected. -> Consider class_weight or SMOTE.")
        return False
    
    # 2. Analyze precision-recall tradeoff for each class
    change_treshold = False
    if any(class_precision[i] > 0.85 and class_recall[i] < 0.60 for i in range(len(class_precision))) and not change_treshold:
        issues.append("High false negatives.")
        suggestions.append("P4A_Lower_decision_threshold")
        print("[Reflection] Per-class performance fails due to low recall. -> Decrease decision threshold.")
        change_treshold = True

    if any(class_recall[i] > 0.85 and class_precision[i] < 0.60 for i in range(len(class_precision))) and not change_treshold:
        issues.append("High false positives.")
        suggestions.append("P4A_Higher_decision_threshold")
        print("[Reflection] Per-class performance fails due to low precision. -> Increase decision threshold.")
        change_treshold = True

    return False if change_treshold else True

def detect_overfitting(best_metrics: Dict[str, Any], issues: List[str], suggestions: List[str]) -> bool:
    """
    Detect overfitting based on train and test F1 scores.
    """
    train_f1 = best_metrics.get("f1_train_macro", 0.0)
    macro_f1 = best_metrics.get("f1_macro", 0.0)

    if train_f1 >= 0.7 and (train_f1 - macro_f1) >= 0.15:
        print("[Reflection] Overfitting detected. -> Consider regularization or simpler models.")
        issues.append("overfitting")
        suggestions.append(["P3A_decrease_model_complexity", "P3A_feature_selection", "P3A_simpler_models"])
        return True
    
    return False

def detect_underfitting(best_metrics: Dict[str, Any], issues: List[str], suggestions: List[str]) -> bool:
    """
    Detect underfitting based on train and test F1 scores.
    """
    train_f1 = best_metrics.get("f1_train_macro", 0.0)
    macro_f1 = best_metrics.get("f1_macro", 0.0)

    if train_f1 < 0.7 and macro_f1 < 0.7:
        print("[Reflection] Underfitting detected. -> Consider more complex models or removing regularization.")
        issues.append("underfitting")
        suggestions.append(["P3A_increase_model_complexity"])
        return True
    
    return False

def acceptable_performance(
        best_metrics: Dict[str, Any], 
        suggestions
    ) -> bool:
    """
    Check if model performance is acceptable based on balanced accuracy and F1 score.
    """
    f1_macro = float(best_metrics.get("f1_macro", 0.0))

    if f1_macro >= 0.70:
        return True
    else:
        print("[Reflection] Model performance is not acceptable. -> Consider hyperparameters tuning.")
        suggestions.append("P4A_tune_hyperparameters")
    return False

""" 
steps should be implemented:

P3A_decrease_model_complexity --> use simple search space
P3A_increase_model_complexity --> use complex search space
P4A_tune_hyperparameters--> use normal search space
P4A_Lower_decision_threshold --> treshold 0.5 -> 0.3
P4A_Higher_decision_threshold --> treshold 0.5 -> 0.7
P3A_SMOTE --> ?
P3A_class_weights --> use this logic
    In these models add "model__class_weight": ["balanced"] to search space to handle class imbalance:
        ['LogisticRegression', 'RandomForestClassifier', 'GradientBoostingClassifier', 'SVC_RBF', 'LinearSVM', 'DecisionTreeClassifier']
    These models have the class_weight parameter that can be set to "balanced" to handle class imbalance.
        GradientBoostingClassifier

"""


"""
1. SIMPLE Models (Overfitting Fix / Regularized)
SIMPLE_SEARCH_SPACE = {
    "LogisticRegression": {
        "model__C": [0.01, 0.1],
        "model__penalty": ["l2"],
        "model__class_weight": [None, "balanced"]
    },

    "RandomForest": {
        "model__n_estimators": [50, 100],
        "model__max_depth": [3, 5],
        "model__min_samples_leaf": [5, 10],
        "model__max_features": ["sqrt"]
    },

    "GradientBoosting": {
        "model__n_estimators": [50, 100],
        "model__learning_rate": [0.03, 0.05],
        "model__max_depth": [2, 3],
        "model__subsample": [0.6, 0.8]
    },

    "SVC_RBF": {
        "model__C": [0.01, 0.1],
        "model__gamma": ["scale", 0.001],
        "model__class_weight": [None, "balanced"]
    },

    "DecisionTree": {
        "model__max_depth": [2, 3, 5],
        "model__min_samples_leaf": [5, 10, 20],
        "model__criterion": ["gini"]
    },

    "LinearSVM": {
        "model__C": [0.01, 0.1],
        "model__class_weight": [None, "balanced"]
    }
}

2. COMPLEX (Underfitting Fix)
COMPLEX_SEARCH_SPACE = {
    "LogisticRegression": {
        "model__C": [10, 50, 100],
        "model__penalty": ["l2"],
        "model__class_weight": [None, "balanced"]
    },

    "RandomForest": {
        "model__n_estimators": [200, 300],
        "model__max_depth": [10, 20, None],
        "model__min_samples_leaf": [1, 2],
        "model__max_features": ["sqrt", None]
    },

    "GradientBoosting": {
        "model__n_estimators": [200, 300],
        "model__learning_rate": [0.1],
        "model__max_depth": [3, 5],
        "model__subsample": [1.0]
    },

    "SVC_RBF": {
        "model__C": [10, 50],
        "model__gamma": [0.1, 1],
        "model__class_weight": [None, "balanced"]
    },

    "DecisionTree": {
        "model__max_depth": [10, 20, None],
        "model__min_samples_leaf": [1, 2],
        "model__criterion": ["gini", "entropy"]
    },

    "LinearSVM": {
        "model__C": [10, 50, 100],
        "model__class_weight": [None, "balanced"]
    }
}

1. NORMAL (Balanced)
NORMAL_SEARCH_SPACE = {
    "LogisticRegression": {
        "model__C": [0.1, 1, 10],
        "model__penalty": ["l2"],
        "model__class_weight": [None, "balanced"]
    },

    "RandomForest": {
        "model__n_estimators": [100, 200],
        "model__max_depth": [None, 5, 10],
        "model__min_samples_leaf": [1, 5],
        "model__max_features": ["sqrt"]
    },

    "GradientBoosting": {
        "model__n_estimators": [100, 200],
        "model__learning_rate": [0.05, 0.1],
        "model__max_depth": [3, 5],
        "model__subsample": [0.8, 1.0]
    },

    "SVC_RBF": {
        "model__C": [0.1, 1, 10],
        "model__gamma": ["scale", 0.1, 0.01],
        "model__class_weight": [None, "balanced"]
    },

    "DecisionTree": {
        "model__max_depth": [None, 5, 10],
        "model__min_samples_leaf": [1, 5, 10],
        "model__criterion": ["gini", "entropy"]
    },

    "LinearSVM": {
        "model__C": [0.1, 1, 10],
        "model__class_weight": [None, "balanced"]
    }
}

"""