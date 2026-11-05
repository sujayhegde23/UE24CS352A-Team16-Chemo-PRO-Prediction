"""Model factories and data helpers shared by both experiments."""
from __future__ import annotations

from pathlib import Path
from typing import Any
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

RANDOM_STATE = 42
ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
OUTPUT_DIR = ROOT / "outputs"


def load_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load the supplied KNN, binned, and binary outcome matrices."""
    required = [DATA_DIR / "chemo_X_knn_933.csv", DATA_DIR / "chemo_X_binned_1367.csv", DATA_DIR / "chemo_targets.csv"]
    if not all(path.exists() for path in required):
        raise FileNotFoundError("Data CSVs are missing. Run: python src/generate_data.py")
    return tuple(pd.read_csv(path) for path in required)  # type: ignore[return-value]


def _logistic(C: float = 0.1, penalty: str = "l2") -> LogisticRegression:
    return LogisticRegression(max_iter=1000, solver="liblinear", C=C, penalty=penalty, random_state=RANDOM_STATE)


def _svm(C: float = 1.0, kernel: str = "rbf") -> SVC:
    return SVC(probability=True, C=C, kernel=kernel, random_state=RANDOM_STATE)


def _forest(min_samples_split: int = 2, max_features: str | None = "sqrt") -> RandomForestClassifier:
    return RandomForestClassifier(n_estimators=200, min_samples_split=min_samples_split, max_features=max_features, random_state=RANDOM_STATE, n_jobs=1)


def _lightgbm(learning_rate: float = 0.05, num_leaves: int = 15) -> LGBMClassifier:
    return LGBMClassifier(verbose=-1, random_state=RANDOM_STATE, n_estimators=150, learning_rate=learning_rate, num_leaves=num_leaves, n_jobs=1)


def make_pipeline(estimator: Any) -> Pipeline:
    """Scaling is fitted only on training folds, preventing test-set leakage."""
    return Pipeline([("scaler", StandardScaler()), ("model", estimator)])


def model_search_spaces() -> dict[str, tuple[Pipeline, dict[str, list[Any]]]]:
    """The intentionally small grids keep the full 20-evaluation run practical."""
    return {
        "Logistic Regression": (make_pipeline(_logistic()), {"model__penalty": ["l1", "l2"], "model__C": [0.01, 0.1, 1.0]}),
        "SVM": (make_pipeline(_svm()), {"model__C": [0.1, 1.0, 10.0], "model__kernel": ["linear", "rbf"]}),
        "Random Forest": (make_pipeline(_forest()), {"model__min_samples_split": [2, 10], "model__max_features": ["sqrt", None]}),
        "LightGBM": (make_pipeline(_lightgbm()), {"model__learning_rate": [0.05, 0.1], "model__num_leaves": [15, 31]}),
    }


def make_ensemble() -> Pipeline:
    """Soft-voting version of the four tuned model families.

    Settings are the compact grid's stable defaults; Experiment 1 uses its
    per-task CV winners as the final voting members.
    """
    voting = VotingClassifier(
        estimators=[("lr", _logistic()), ("svm", _svm()), ("rf", _forest()), ("lgbm", _lightgbm())],
        voting="soft", n_jobs=1,
    )
    return make_pipeline(voting)


def ensemble_from_best(best_estimators: dict[str, Pipeline]) -> Pipeline:
    """Build a voting pipeline from the four GridSearchCV winners."""
    members = []
    labels = {"Logistic Regression": "lr", "SVM": "svm", "Random Forest": "rf", "LightGBM": "lgbm"}
    for name, label in labels.items():
        members.append((label, best_estimators[name].named_steps["model"]))
    return make_pipeline(VotingClassifier(estimators=members, voting="soft", n_jobs=1))
