"""
data_utils.py – Data loading, harmonisation, and preprocessing.

Implements the dataset configuration described in Section 3.3 of the paper.
Five publicly available cardiovascular datasets are harmonised to the common
13-feature UCI Heart Disease schema and split into per-client train/test
pairs that simulate the non-IID, institutionally siloed nature of real
healthcare data.

Datasets (Section 3.3.1 of the paper)
-------------------------------------
Client 0 – Framingham Heart Study            (framingham.csv)
Client 1 – Cleveland Clinic Foundation        (cleveland.csv)
Client 2 – Hungarian Institute of Cardiology  (hungarian.csv)
Client 3 – University Hospital Zurich, Switzerland (switzerland.csv)
Client 4 – Long Beach VA Medical Center       (long_beach_va.csv)

The four UCI sources (Cleveland, Hungarian, Switzerland, Long Beach VA)
natively share the 13-attribute schema. The Framingham study is harmonised
by mapping its corresponding variables; attributes absent from a given
source are treated as missing and imputed with site-specific medians
computed on the training partition only (Section 3.3.2 of the paper).

Preprocessing guarantees (Section 3.3.3 of the paper)
-----------------------------------------------------
1. The 80/20 stratified train/test split happens BEFORE federation.
2. Median imputation statistics are computed on the training partition
   only and then applied to the test partition (no train/test leakage).
3. StandardScaler is fitted on the training partition only.
4. All preprocessing is site-specific (no cross-site leakage).
"""

from __future__ import annotations

import os
import random
import warnings
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from flwr.common import Metrics
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore", category=UserWarning)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

FILENAMES = [
    "framingham.csv",
    "cleveland.csv",
    "hungarian.csv",
    "switzerland.csv",
    "long_beach_va.csv",
]

CLIENT_NAMES = [
    "Framingham",
    "Cleveland",
    "Hungarian",
    "Switzerland",
    "Long Beach VA",
]

# Standard 13-attribute UCI Heart Disease schema (paper Section 3.2)
FINAL_FEATURES = [
    "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
    "thalach", "exang", "oldpeak", "slope", "ca", "thal",
]
TARGET_COLUMN = "target"

# Column rename maps per dataset → common schema
COLUMN_MAPPINGS: List[Dict[str, str]] = [
    # Framingham: longitudinal study; maps the overlapping risk factors.
    # Attributes without a Framingham counterpart remain missing and are
    # median-imputed (documented in the paper, Section 3.3.2).
    {
        "male": "sex",
        "age": "age",
        "sysBP": "trestbps",
        "totChol": "chol",
        "diabetes": "fbs",
        "heartRate": "thalach",
        "TenYearCHD": "target",
    },
    # Cleveland (UCI processed schema)
    {"num": "target"},
    # Hungarian (UCI processed schema)
    {"num": "target"},
    # Switzerland (UCI processed schema)
    {"num": "target"},
    # Long Beach VA (UCI processed schema)
    {"num": "target"},
]


def set_global_seeds(seed: int = 42) -> None:
    """Fix all random sources for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load_and_preprocess_data(
    data_dir: str = "data",
    test_size: float = 0.2,
    random_state: int = 42,
) -> Tuple[
    Optional[List[Tuple[np.ndarray, np.ndarray]]],
    Optional[List[Tuple[np.ndarray, np.ndarray]]],
    Optional[List[str]],
]:
    """Load and harmonise all five cardiovascular datasets.

    Returns
    -------
    client_train_datasets : list of (X_train, y_train), one per client.
    client_test_datasets  : list of (X_test,  y_test),  one per client.
    client_names          : list of institution names (same order).

    All three values are ``None`` if loading fails for any dataset.
    """
    print("--- Loading and Preprocessing Datasets for 5 Clients ---")

    client_train_datasets: List[Tuple[np.ndarray, np.ndarray]] = []
    client_test_datasets: List[Tuple[np.ndarray, np.ndarray]] = []

    for i, filename in enumerate(FILENAMES):
        filepath = os.path.join(data_dir, filename)
        try:
            df = pd.read_csv(filepath)
            df.columns = df.columns.str.strip()
            df.rename(columns=COLUMN_MAPPINGS[i], inplace=True)

            if TARGET_COLUMN not in df.columns:
                raise ValueError(
                    f"Target column '{TARGET_COLUMN}' not found in {filename} "
                    f"after mapping. Found columns: {list(df.columns)}"
                )

            # Ensure every schema feature exists (missing ones → NaN)
            for col in FINAL_FEATURES:
                if col not in df.columns:
                    df[col] = np.nan

            df = df[FINAL_FEATURES + [TARGET_COLUMN]]

            # Coerce to numeric ('?' in UCI files becomes NaN)
            for col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

            # Drop rows with missing target, binarise (any positive class → 1)
            df = df.dropna(subset=[TARGET_COLUMN])
            df[TARGET_COLUMN] = (df[TARGET_COLUMN] > 0).astype(int)

            X = df[FINAL_FEATURES]
            y = df[TARGET_COLUMN].values

            # 1) Stratified 80/20 split BEFORE any statistic is computed
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size,
                random_state=random_state, stratify=y,
            )

            # 2) Median imputation: statistics from the TRAINING split only
            train_medians = X_train.median(numeric_only=True)
            X_train = X_train.fillna(train_medians).fillna(0.0)
            X_test = X_test.fillna(train_medians).fillna(0.0)

            # 3) Standardisation: fitted on the TRAINING split only
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train.values)
            X_test_scaled = scaler.transform(X_test.values)

            client_train_datasets.append((X_train_scaled, y_train))
            client_test_datasets.append((X_test_scaled, y_test))

            print(
                f"  Client {i} ({CLIENT_NAMES[i]}): {len(df)} samples — "
                f"train={len(y_train)}, test={len(y_test)}, "
                f"pos_rate={y.mean():.1%}"
            )

        except FileNotFoundError:
            print(
                f"ERROR: '{filepath}' not found. "
                "See data/README.md for download instructions."
            )
            return None, None, None
        except Exception as exc:
            print(f"ERROR processing {filename}: {exc}")
            return None, None, None

    return client_train_datasets, client_test_datasets, CLIENT_NAMES


def build_global_test_set(
    client_test_data: List[Tuple[np.ndarray, np.ndarray]],
) -> Tuple[np.ndarray, np.ndarray]:
    """Composite global test set: concatenation of all client test splits.

    Implements the 'Global Test Set Evaluation' described in Section 3.3.3:
    the 20% test partitions of all institutions are aggregated into a single
    held-out evaluation set never exposed to training.
    """
    X = np.concatenate([X_te for X_te, _ in client_test_data], axis=0)
    y = np.concatenate([y_te for _, y_te in client_test_data], axis=0)
    return X, y


def aggregate_metrics_fn(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    """Weighted average of per-client evaluation metrics (Flower hook)."""
    if not metrics:
        return {}
    valid = [(n, m) for n, m in metrics if n > 0 and m]
    if not valid:
        return {}
    total = sum(n for n, _ in valid)
    if total == 0:
        return {}
    keys = ["accuracy", "precision", "recall", "f1_score", "auc"]
    return {
        k: sum(n * m[k] for n, m in valid if k in m) / total
        for k in keys
        if any(k in m for _, m in valid)
    }
