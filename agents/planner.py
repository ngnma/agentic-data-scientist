"""
Planner Agent - Students must extend this significantly

The planner analyzes dataset characteristics and generates an execution plan.
Your task is to implement sophisticated planning logic that adapts to different
dataset types, sizes, and characteristics.

TODO: Extend this module with:
1. Sophisticated planning logic based on dataset profiles
2. Different plan templates for different scenarios
3. Memory-guided planning (use past successful strategies)
4. Dependency management (task ordering)
5. Conditional planning (if X then Y else Z)
6. Fallback strategies for edge cases
"""

from typing import Any, Dict, List, Optional


def create_plan(
    dataset_profile: Dict[str, Any], 
    internal_memory: Dict[str, Any],
    memory_hint: Optional[Dict[str, Any]] = None
) -> List[str]:
    """
    Generate an execution plan based on dataset characteristics.
    
    This is a basic implementation. Students should extend this significantly.
    
    Args:
        dataset_profile: Dictionary containing dataset metadata including:
            - shape: {rows: int, cols: int}
            - feature_types: {numeric: List[str], categorical: List[str]}
            - imbalance_ratio: float (majority/minority class ratio)
            - missing_pct: Dict[str, float] (missing % per column)
            - is_classification: bool
            - notes: List[str] (warnings/observations)
        memory_hint: Optional dict with info from previous runs on similar datasets
    
    Returns:
        List of task names representing the execution plan
        
    Example:
        >>> profile = {"shape": {"rows": 5000}, "imbalance_ratio": 4.5}
        >>> plan = create_plan(profile)
        >>> print(plan)
        ['profile_dataset', 'consider_imbalance_strategy', 'train_models', ...]
    
    TODO for students:
    - Implement conditional logic based on dataset size
    - Add different strategies for imbalanced datasets
    - Handle high-cardinality categorical features
    - Use memory hints to prioritize successful models
    - Create plan templates for common scenarios
    - Add preprocessing steps based on data quality
    """
    
    # Basic plan structure (students should make this much more sophisticated)
    plan: List[str] = [
        # "P1B_profile_dataset",
        "P2B_build_preprocessor",
        "P3B_select_models",
        "P4B_train_models",
        "P5B_evaluate",
        "P6B_reflect",
        "P7B_write_report",
    ]

    # Extract key dataset characteristics
    rows = dataset_profile.get("shape", {}).get("rows", 0)
    cols = dataset_profile.get("shape", {}).get("cols", 0)
    imb = dataset_profile.get("imbalance_ratio") or 1.0

    # Add candidate models based on dataset size and characteristics
    internal_memory.setdefault('candidates', []).extend(['DummyMostFrequent', 'RandomForest', 'LogisticRegression'])

    # if rows <= 50000: # TODO: un-comment it. it is for testing purposes only
    #     internal_memory.get('candidates',[]).append('GradientBoosting')
    # if rows <= 20000 and cols <= 200:
    #     internal_memory.get('candidates',[]).append('SVC_RBF')

    # All logic for imbalance
    if imb >= 3.0:
        plan.append("P3A_imb_class_weight")
    
    # Add logic for small datasets
    if dataset_profile["shape"]["rows"] < 100:
        plan.append("P3A_regularization")
    
    # TODO: Add logic for high-cardinality categoricals
    # high_card_cats = [c for c in categorical_cols if n_unique[c] > 50]
    # if high_card_cats:
    #     plan.insert(..., "apply_target_encoding")
    
    # TODO: Use memory hints
    # if memory_hint and memory_hint.get("best_model"):
    #     plan.append(f"prioritize_model:{memory_hint['best_model']}")
    
    # TODO: Add logic based on missing data
    # max_missing = max(dataset_profile["missing_pct"].values())
    # if max_missing > 20:
    #     plan.insert(..., "handle_severe_missing_data")
    
    # Ensure consistent execution order (P1A, P1B, P2A, P2B, etc.)
    plan.sort()  
    return plan


# TODO: Add helper functions for planning
# def create_small_dataset_plan(...):
# def create_imbalanced_dataset_plan(...):
# def create_high_dimensional_plan(...):
# def select_preprocessing_strategy(...):
# def estimate_plan_cost(...):  # For cost-aware planning
