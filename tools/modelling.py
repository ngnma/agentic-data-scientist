from typing import Any, Dict, List, Tuple

import os
import pandas as pd
import numpy as np

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split

from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)


def build_preprocessor(profile: Dict[str, Any]) -> ColumnTransformer:
    num_cols = profile["feature_types"]["numeric"]
    cat_cols = profile["feature_types"]["categorical"]

    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler(with_mean=True)),
    ])

    # scikit-learn renamed `sparse` -> `sparse_output` (v1.2+). Support both.
    try:
        ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        ohe = OneHotEncoder(handle_unknown="ignore", sparse=False)

    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", ohe),
    ])

    return ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, num_cols),
            ("cat", categorical_transformer, cat_cols),
        ],
        remainder="drop",
    )


def select_models(profile: Dict[str, Any], seed: int = 42) -> List[Tuple[str, Any]]:
    rows = profile["shape"]["rows"]
    cols = profile["shape"]["cols"]

    # ---- Handle imbalance strategy ----
    class_weight = "balanced" if profile['plan_notes'].get('imbalance_strategy') else None

    # Log the imbalance strategy decision
    need_imbalance_log = "Yes" if profile['plan_notes'].get('imbalance_strategy') else "No"
    print(f"[Modelling] Imbalance strategy applied: {need_imbalance_log}. Class weight set to '{class_weight}' for applicable models.")

    # ---- Regularization parameters ----
    need_reg =  profile['plan_notes'].get('regularization')

    # Logistic Regression
    LR_C = 0.1 if need_reg else 1.0  # default = 1.0

    # Random Forest
    RF_n_estimators = 100 if need_reg else 100  # default = 100
    RF_max_depth = 5 if need_reg else None     # default = None
    RF_min_samples_leaf = 10 if need_reg else 1  # default = 1

    # Gradient Boosting
    GB_n_estimators = 50 if need_reg else 100   # default = 100
    GB_learning_rate = 0.05 if need_reg else 0.1  # default = 0.1
    GB_max_depth = 2 if need_reg else 3         # default = 3

    # SVM (RBF)
    SVM_C = 0.5 if need_reg else 1.0  # default = 1.0
    SVM_gamma = "scale"  # default

    # Log the regularization decision
    reg_log = "Yes" if need_reg else "No"
    print(f"[Modelling] Regularization applied: {reg_log}. Parameters set accordingly for model selection.")


    candidates: List[Tuple[str, Any]] = [
        ("DummyMostFrequent", DummyClassifier(strategy="most_frequent")),

        ("LogisticRegression",
         LogisticRegression(
             max_iter=2000,
             C=LR_C,
             penalty="l2",
             solver="liblinear",
             class_weight=class_weight
         )),

        ("RandomForest",
         RandomForestClassifier(
             n_estimators=RF_n_estimators,
             max_depth=RF_max_depth,
             min_samples_leaf=RF_min_samples_leaf,
             random_state=seed,
             n_jobs=-1,
             class_weight=class_weight
         )),
    ]

    if rows <= 50000:
        candidates.append((
            "GradientBoosting",
            GradientBoostingClassifier(
                n_estimators=GB_n_estimators,
                learning_rate=GB_learning_rate,
                max_depth=GB_max_depth,
                random_state=seed
            )
        ))

    if rows <= 20000 and cols <= 200:
        candidates.append((
            "SVC_RBF",
            SVC(
                kernel="rbf",
                C=SVM_C,
                gamma=SVM_gamma,
                probability=True,
                class_weight=class_weight
            )
        ))

    return candidates


def train_models(
    df: pd.DataFrame,
    target: str,
    preprocessor: ColumnTransformer,
    candidates: List[Tuple[str, Any]],
    seed: int,
    test_size: float,
    output_dir: str,
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

    # Stratify if possible
    stratify = y if (y.nunique(dropna=True) > 1 and y.value_counts().min() >= 2) else None

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=seed, stratify=stratify
    )

    results: List[Dict[str, Any]] = []

    for name, model in candidates:
        if verbose:
            print(f"[Modelling] Training: {name}")

        pipe = Pipeline(steps=[
            ("preprocess", preprocessor),
            ("model", model),
        ])
        pipe.fit(X_train, y_train)

        y_pred = pipe.predict(X_test)

        metrics = {
            "model": name,
            "accuracy": float(accuracy_score(y_test, y_pred)),
            "balanced_accuracy": float(balanced_accuracy_score(y_test, y_pred)),
            "f1_macro": float(f1_score(y_test, y_pred, average="macro", zero_division=0)),
            "precision_macro": float(precision_score(y_test, y_pred, average="macro", zero_division=0)),
            "recall_macro": float(recall_score(y_test, y_pred, average="macro", zero_division=0)),
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
    results.sort(key=lambda r: (r["metrics"]["balanced_accuracy"], r["metrics"]["f1_macro"]), reverse=True)

    return {
        "results": results,
        "best": results[0],
        "all_metrics": [r["metrics"] for r in results],
    }
