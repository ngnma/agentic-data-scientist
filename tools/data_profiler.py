from typing import Any, Dict, Optional
import pandas as pd
from sklearn.feature_selection import f_classif
import numpy as np


def infer_target_column(df: pd.DataFrame) -> Optional[str]:
    """
    Heuristic target inference:
      - prefer common target-like column names
      - else last column if it has relatively low cardinality
    """
    candidates = ["target", "label", "class", "y", "outcome", "survived"]
    lower_map = {c.lower(): c for c in df.columns}
    for k in candidates:
        if k in lower_map:
            return lower_map[k]

    last = df.columns[-1]
    uniq = df[last].nunique(dropna=True)
    n = len(df)
    if n > 0 and (uniq <= 50 or (uniq / max(n, 1) < 0.05)):
        return last
    return None


def is_classification_target(series: pd.Series) -> bool:
    if series.dtype == "object" or str(series.dtype).startswith("category"):
        return True
    uniq = series.nunique(dropna=True)
    return uniq <= 50


def dataset_fingerprint(df: pd.DataFrame, target: str) -> str:
    cols = ",".join(df.columns.astype(str).tolist())
    shape = f"{df.shape[0]}x{df.shape[1]}"
    base = f"{shape}|{target}|{cols}"
    h = abs(hash(base)) % (10**12)
    return f"fp_{h}"


# def profile_dataset(df: pd.DataFrame, target: str) -> Dict[str, Any]:
#     if target not in df.columns:
#         raise ValueError(f"Target column '{target}' not found in dataset columns.")

#     y = df[target]
#     profile: Dict[str, Any] = {}

#     profile["shape"] = {"rows": int(df.shape[0]), "cols": int(df.shape[1])}
#     profile["columns"] = df.columns.astype(str).tolist()

#     missing = (df.isna().mean() * 100).round(2).to_dict()
#     profile["missing_pct"] = {str(k): float(v) for k, v in missing.items()}

#     profile["target"] = str(target)
#     profile["target_dtype"] = str(y.dtype)
#     profile["is_classification"] = bool(is_classification_target(y))

#     # Feature types
#     X = df.drop(columns=[target])
#     numeric_cols = X.select_dtypes(include=["number", "bool"]).columns.astype(str).tolist()
#     cat_cols = [c for c in X.columns.astype(str).tolist() if c not in numeric_cols]

#     profile["feature_types"] = {"numeric": numeric_cols, "categorical": cat_cols}
#     profile["n_unique_by_col"] = {str(c): int(df[c].nunique(dropna=True)) for c in df.columns.astype(str)}

#     # outlier_ratio_by_col using IQR method
#     outlier_ratio_by_col = {}
#     for col in numeric_cols:
#         s = pd.to_numeric(X[col], errors="coerce").dropna()

#         if s.empty:
#             outlier_ratio_by_col[str(col)] = 0.0
#             continue

#         q1 = s.quantile(0.25)
#         q3 = s.quantile(0.75)
#         iqr = q3 - q1

#         if iqr == 0:
#             outlier_ratio_by_col[str(col)] = 0.0
#             continue

#         lower = q1 - 1.5 * iqr
#         upper = q3 + 1.5 * iqr
#         ratio = ((s < lower) | (s > upper)).mean()

#         outlier_ratio_by_col[str(col)] = round(float(ratio), 4)

#     profile["outlier_ratio_by_col"] = outlier_ratio_by_col

#     # Class balance if classification
#     if profile["is_classification"]:
#         vc = y.value_counts(dropna=False)
#         profile["class_counts"] = {str(k): int(v) for k, v in vc.items()}
#         profile["n_classes"] = int(vc.nunique())
#         if len(vc) >= 2:
#             ratio = float(vc.max() / max(vc.min(), 1))
#         else:
#             ratio = 1.0
#         profile["imbalance_ratio"] = round(ratio, 3)
#     else:
#         profile["class_counts"] = None
#         profile["imbalance_ratio"] = None

#     # noise_ratio estimation using ANOVA F-test p-values for numeric features
#     aligned_numerical_df, aligned_y = df[numeric_cols].dropna().align(y, join='inner', axis=0)
#     F, p_value = f_classif(aligned_numerical_df, aligned_y)
#     noise_ratio = np.mean(p_value > 0.05)
#     profile["noise_ratio"] = round(float(noise_ratio), 3)

#     # skewness estimation for numeric features
#     skewness_by_col = {}
#     for col in numeric_cols:
#         skewness_by_col[str(col)] = abs(float(round(X[col].skew(),4)))
#     profile["skewness_by_col"] = skewness_by_col

#     return profile


def profile_dataset(df: pd.DataFrame, target: str) -> Dict[str, Any]:
    """
    Build a lightweight dataset profile used by the planner.

    Notes:
    - Boolean columns are excluded from continuous-statistics logic
      such as quantiles, IQR outlier detection, and skewness.
    """

    if target not in df.columns:
        raise ValueError(f"Target column '{target}' not found in dataframe.")

    X = df.drop(columns=[target])
    y = df[target]

    missing_pct = (df.isna().mean() * 100).round(2).to_dict()

    numeric_cols = [
        c for c in X.columns
        if pd.api.types.is_numeric_dtype(X[c]) and not pd.api.types.is_bool_dtype(X[c])
    ]
    categorical_cols = [c for c in X.columns if c not in numeric_cols]

    n_unique_by_col = {c: int(df[c].nunique(dropna=True)) for c in df.columns}

    outlier_ratio_by_col: Dict[str, float] = {}
    skewness_by_col: Dict[str, float] = {}

    for c in numeric_cols:
        s = X[c].dropna()

        if s.empty:
            outlier_ratio_by_col[c] = 0.0
            skewness_by_col[c] = 0.0
            continue

        q1 = s.quantile(0.25)
        q3 = s.quantile(0.75)
        iqr = q3 - q1

        if pd.isna(iqr) or iqr == 0:
            outlier_ratio_by_col[c] = 0.0
        else:
            lower = q1 - 1.5 * iqr
            upper = q3 + 1.5 * iqr
            outlier_ratio_by_col[c] = float(((s < lower) | (s > upper)).mean())

        skew_val = s.skew()
        skewness_by_col[c] = 0.0 if pd.isna(skew_val) else float(skew_val)

    is_ordinal = {}
    for c in categorical_cols:
        if pd.api.types.is_categorical_dtype(df[c]):
            is_ordinal[c] = bool(df[c].cat.ordered)
        else:
            is_ordinal[c] = False

    is_classification = is_classification_target(y)

    class_balance = None
    imbalance_ratio = None
    n_classes = None

    if is_classification:
        vc = y.value_counts(dropna=False)
        class_balance = {str(k): float(v / len(y)) for k, v in vc.items()}
        imbalance_ratio = float(vc.max() / vc.min()) if len(vc) > 1 and vc.min() > 0 else 1.0
        n_classes = int(len(vc))

    noise_ratio = float(np.mean(list(outlier_ratio_by_col.values()))) if outlier_ratio_by_col else 0.0

    profile = {
        "shape": {
            "rows": int(df.shape[0]),
            "cols": int(df.shape[1]),
        },
        "target": target,
        "is_classification": is_classification,
        "n_classes": n_classes,
        "class_balance": class_balance,
        "imbalance_ratio": imbalance_ratio,
        "missing_pct": missing_pct,
        "feature_types": {
            "numeric": numeric_cols,
            "categorical": categorical_cols,
        },
        "n_unique_by_col": n_unique_by_col,
        "outlier_ratio_by_col": outlier_ratio_by_col,
        "skewness_by_col": skewness_by_col,
        "noise_ratio": noise_ratio,
        "is_ordinal": is_ordinal,
    }

    return profile