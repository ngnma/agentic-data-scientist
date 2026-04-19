"""
Planner Agent

The planner analyzes dataset characteristics and generates an execution plan.
Your task is to implement sophisticated planning logic that adapts to different
dataset types, sizes, and characteristics.

TODO: Extend this module with:
2. Different plan templates for different scenarios
3. Memory-guided planning (use past successful strategies)
4. Dependency management (task ordering)
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

    """
    
    # Basic plan structure (students should make this much more sophisticated)
    plan: List[str] = [
        # "P1B_profile_dataset",
        "P3A0_select_basic_models",
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
    max_missing = max(dataset_profile.get("missing_pct", {}).values(), default=0)
    categorical_cols = dataset_profile.get("feature_types", {}).get("categorical", [])
    n_unique = dataset_profile.get("n_unique_by_col", {})
    noise_ratio = dataset_profile.get("noise_ratio", 0.0)
    max_skewness = max(dataset_profile.get("skewness_by_col", {}).values(), default=0)

    # Add additional candidate models based on dataset size and characteristics
    if rows <= 50000 or cols <= 200:
        plan.append("P3A1_select_additional_models")

    # Imbalance Classes
    if imb >= 3.0:
        plan.append("P3A_imb_class_weight")
    
    # Small Datasets
    if rows < 100:
        plan.append("P3A4_regularization")
    
    # High Cardinality Categoricals
    high_card_cats = [c for c in categorical_cols if n_unique[c] > 50]
    if high_card_cats:
        plan.append("P2A2_apply_categorical_encoding")
    
    # Missing Values
    if max_missing > 20:
        plan.append("P2A0_handle_missing_values")

    # High Dimensionality
    if cols > 100 or (rows > 0 and cols / rows > 0.5):
        plan.append("P3A_feature_selection")

    # Noisy Data
    if noise_ratio> 0.3:
        plan.append("P3A4_regularization")
    
    # skewness (Normalization)
    if max_skewness >= 0.5:
        plan.append("P2A5_handle_skewness")


    # TODO: Use memory hints
    # if memory_hint and memory_hint.get("best_model"):
    #     plan.append(f"prioritize_model:{memory_hint['best_model']}")

    
    
    # remove plan dupplicates
    plan = list(set(plan))

    # Ensure consistent execution order (P1A, P1B, P2A, P2B, etc.)
    plan.sort()  
    return plan


# TODO: Add helper functions for planning
# def create_small_dataset_plan(...):
# def create_imbalanced_dataset_plan(...):
# def create_high_dimensional_plan(...):
# def select_preprocessing_strategy(...):
# def estimate_plan_cost(...):  # For cost-aware planning

