from typing import Any, Dict, List, Tuple
from imblearn.over_sampling import SMOTE

import os
import pandas as pd
import numpy as np

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, TargetEncoder

from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.base import clone
from sklearn.model_selection import (
    train_test_split,
    StratifiedKFold,
    KFold,
    cross_val_score,
    GridSearchCV
)
from sklearn.feature_selection import SelectKBest, f_classif


# def build_preprocessor(profile: Dict[str, Any], internal_memory: Dict[str, Any]) -> ColumnTransformer:
#     if "drop_cols" in internal_memory:
#         drop_cols = internal_memory["drop_cols"]
#         print(f"[Preprocessing] Dropping columns with high missing values: {drop_cols}")

#         num_cols = [c for c in profile["feature_types"]["numeric"] if c not in drop_cols]
#         cat_cols = [c for c in profile["feature_types"]["categorical"] if c not in drop_cols]
#     else:
#         num_cols = profile["feature_types"]["numeric"]
#         cat_cols = profile["feature_types"]["categorical"]

#     target_encoding_cols = internal_memory.get('target_encoding', [])

#     numeric_transformer = Pipeline(steps=[
#         ("imputer", SimpleImputer(strategy="median")),
#         ("scaler", StandardScaler(with_mean=True)),
#     ])

#     # scikit-learn renamed `sparse` -> `sparse_output` (v1.2+). Support both.
#     try:
#         ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
#     except TypeError:
#         ohe = OneHotEncoder(handle_unknown="ignore", sparse=False)

#     categorical_transformer = Pipeline(steps=[
#         ("imputer", SimpleImputer(strategy="most_frequent")),
#         ("onehot", ohe),
#     ])

#     return ColumnTransformer(
#         transformers=[
#             ("num", numeric_transformer, num_cols),
#             ("cat", categorical_transformer, cat_cols),
#         ],
#         remainder="drop",
#     )


def build_preprocessor(profile: Dict[str, Any], internal_memory: Dict[str, Any]) -> ColumnTransformer:

    drop_cols = internal_memory.get("drop_cols",[])
    num_cols = [c for c in profile["feature_types"]["numeric"] if c not in drop_cols]
    cat_cols = [c for c in profile["feature_types"]["categorical"] if c not in drop_cols]
    target_encoding_cols = [c for c in cat_cols if c in internal_memory.get("target_encoding", [])]
    onehot_encoded_cols = [c for c in cat_cols if c not in target_encoding_cols]

    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler(with_mean=True)),
    ])

    try:
        onehot_encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        onehot_encoder = OneHotEncoder(handle_unknown="ignore", sparse=False)

    onehot_enc_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", onehot_encoder),
    ])

    target_enc_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("target_encoder", TargetEncoder()),
    ])

    transformers = [
        ("num", numeric_transformer, num_cols),
    ]

    if onehot_encoded_cols:
        transformers.append(("cat_ohe", onehot_enc_transformer, onehot_encoded_cols))
        print(f"[Preprocessing] Applying OneHotEncoder to columns: {onehot_encoded_cols}")

    if target_encoding_cols:
        transformers.append(("cat_target", target_enc_transformer, target_encoding_cols))
        print(f"[Preprocessing] Applying TargetEncoder to columns: {target_encoding_cols}")


    return ColumnTransformer(
        transformers=transformers,
        remainder="drop",
    )

def select_models(internal_memory: Dict[str, Any], seed: int = 42) -> List[Tuple[str, Any, Dict[str, Any]]]:

    SEACRH_SPACE = {
        "with_reqularization": {
            "LogisticRegression": {
                "model__C": [0.1],
                "model__penalty": ["l2"],
                "model__solver": ["liblinear"]
            },

            "RandomForest": {
                "model__n_estimators": [100],
                "model__max_depth": [5],
                "model__min_samples_leaf": [10]
            },

            "GradientBoosting": {
                "model__n_estimators": [50],
                "model__learning_rate": [0.05],
                "model__max_depth": [2]
            },

            "SVC_RBF": {
                "model__C": [0.5],
                "model__gamma": ["scale"],
            },

            "DecisionTree": {
                "model__max_depth": [5]
            },

            "LinearSVM": {
                "model__C": [0.5],
            },

            "DummyMostFrequent": {}
        },
        "simple": {
            "LogisticRegression": {
                "model__C": [0.01, 0.1],
                "model__penalty": ["l2"],
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
            },

            "DecisionTree": {
                "model__max_depth": [2, 3, 5],
                "model__min_samples_leaf": [5, 10, 20],
                "model__criterion": ["gini"]
            },

            "LinearSVM": {
                "model__C": [0.01, 0.1],
            },

            "DummyMostFrequent": {}
        },
        "complex": {
            "LogisticRegression": {
                "model__C": [10, 50, 100],
                "model__penalty": ["l2"],
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
            },

            "DecisionTree": {
                "model__max_depth": [10, 20, None],
                "model__min_samples_leaf": [1, 2],
                "model__criterion": ["gini", "entropy"]
            },

            "LinearSVM": {
                "model__C": [10, 50, 100],
            },

            "DummyMostFrequent": {}
        },
        "normal": {
            "LogisticRegression": {
                "model__C": [0.1, 1, 10],
                "model__penalty": ["l2"],
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
            },

            "DecisionTree": {
                "model__max_depth": [None, 5, 10],
                "model__min_samples_leaf": [1, 5, 10],
                "model__criterion": ["gini", "entropy"]
            },

            "LinearSVM": {
                "model__C": [0.1, 1, 10],
            },

            "DummyMostFrequent": {}
        }
    }

    class_weight = internal_memory.get("class_weight", None) 
    MODEL_DICT = {
        "DummyMostFrequent": DummyClassifier(strategy="most_frequent"),
        "LogisticRegression": LogisticRegression(max_iter=2000, class_weight=class_weight),
        "RandomForest": RandomForestClassifier(random_state=seed, n_jobs=-1, class_weight=class_weight),
        "GradientBoosting": GradientBoostingClassifier(random_state=seed),
        "SVC_RBF": SVC(kernel="rbf", probability=True, class_weight=class_weight),
        "DecisionTree": DecisionTreeClassifier(class_weight=class_weight),
        "LinearSVM": SVC(kernel="linear", probability=True, class_weight=class_weight)
    }

    candidates: List[Tuple[str, Any]] = []

    for model_name in internal_memory.get("candidates", []):
        if model_name in MODEL_DICT:
            
            candidates.append((
                model_name,
                MODEL_DICT[model_name],
                SEACRH_SPACE.get(internal_memory.get("search_space", None), {}).get(model_name, {})
            ))
        else:
            print(f"[Modelling] Warning: Model '{model_name}' not recognized. Skipping.")

    return candidates


def feature_selection(k: int = 10) -> Any:
    return SelectKBest(score_func=f_classif, k=k)

def apply_smote(seed: int) -> Any:
    return SMOTE(random_state=seed)

def train_models(
    df: pd.DataFrame,
    target: str,
    preprocessor: ColumnTransformer,
    candidates: List[Tuple[str, Any, Dict[str, Any]]],
    seed: int,
    test_size: float,
    output_dir: str,
    internal_memory: Dict[str, Any],
    feature_selector: Any = None,
    smote: Any = None,
    verbose: bool = True,
) -> Dict[str, Any]:
    if target not in df.columns:
        raise ValueError(f"Target '{target}' not found.")
    
    X = df.drop(columns=[target]).copy()
    y = df[target].copy()

    # Drop missing target rows
    mask = ~y.isna()
    X = X.loc[mask]
    y = y.loc[mask]

    if y.nunique(dropna=True) < 2:
        raise ValueError("Target must contain at least 2 classes for classification.")

    # Stratify train/test split if possible
    stratify = y if y.value_counts().min() >= 2 else None

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=seed,
        stratify=stratify,
    )

    results: List[Dict[str, Any]] = []

    # Threshold logic is only valid for binary classification and only if configured
    decision_threshold = internal_memory.get("decision_threshold")

    # Use 5-fold CV when possible; otherwise fall back to the maximum valid number of folds
    min_class_count_train = y_train.value_counts().min()
    if y_train.nunique(dropna=True) > 1 and min_class_count_train >= 2:
        n_splits = min(5, int(min_class_count_train))
        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    else:
        n_splits = min(5, len(y_train))
        cv = KFold(n_splits=n_splits, shuffle=True, random_state=seed)

    for name, model, param_grid in candidates:

        pipe = ImbPipeline(steps=[
            ("preprocess", preprocessor),
            ("feature_selection", feature_selector if feature_selector is not None else "passthrough"),
            ("smote", smote if smote is not None else "passthrough"),
            ("model", model),
        ])

        if verbose:
            print(f"[Modelling] Training: {name}")

        grid = GridSearchCV(
            estimator=clone(pipe),
            param_grid=param_grid,
            cv=cv,
            scoring="f1_macro",
            n_jobs=None,
            refit=True,
        )

        print(f"[Modelling] Starting GridSearchCV for {name} with parameter: {param_grid}")

        grid.fit(X_train, y_train)
        pipe = grid.best_estimator_
        cv_scores = list(grid.cv_results_["mean_test_score"])

        # Default behavior
        y_pred = pipe.predict(X_test)
        y_train_pred = pipe.predict(X_train)

        # Apply custom decision threshold only when valid and available
        if decision_threshold is not None:
            if hasattr(pipe, "predict_proba"):
                try:
                    y_pred = (pipe.predict_proba(X_test)[:, 1] >= decision_threshold).astype(int)
                    y_train_pred = (pipe.predict_proba(X_train)[:, 1] >= decision_threshold).astype(int)
                    if verbose:
                        print(f"[Modelling] Applied decision threshold={decision_threshold} using predict_proba for {name}.")
                except Exception:
                    print(f"[Modelling] Model {name} does not support usable predict_proba thresholding. Continuing with default predict().")
            elif hasattr(pipe, "decision_function"):
                try:
                    y_pred = (pipe.decision_function(X_test) >= decision_threshold).astype(int)
                    y_train_pred = (pipe.decision_function(X_train) >= decision_threshold).astype(int)
                    if verbose:
                        print(f"[Modelling] Applied decision threshold={decision_threshold} using decision_function for {name}.")
                except Exception:
                    print(f"[Modelling] Model {name} does not support usable decision_function thresholding. Continuing with default predict().")
            else:
                print(f"[Modelling] Model {name} does not support custom decision threshold. Continuing with default predict().")

        metrics = {
            "model": name,
            "accuracy": float(accuracy_score(y_test, y_pred)),
            "balanced_accuracy": float(balanced_accuracy_score(y_test, y_pred)),
            "f1_macro": float(f1_score(y_test, y_pred, average="macro", zero_division=0)),
            "f1_train_macro": float(f1_score(y_train, y_train_pred, average="macro", zero_division=0)),
            "precision_macro": float(precision_score(y_test, y_pred, average="macro", zero_division=0)),
            "recall_macro": float(recall_score(y_test, y_pred, average="macro", zero_division=0)),
            "cv_f1_scores": [float(score) for score in cv_scores],
        }

        results.append({
            "name": name,
            "pipeline": pipe,
            "metrics": metrics,
            "X_test": X_test,
            "y_test": y_test,
            "y_pred": y_pred,
        })

    # Sort by balanced accuracy then macro F1
    results.sort(
        key=lambda r: (
            r["metrics"]["balanced_accuracy"],
            r["metrics"]["f1_macro"],
        ),
        reverse=True,
    )

    return {
        "results": results,
        "best": results[0],
        "all_metrics": [r["metrics"] for r in results]
    }
