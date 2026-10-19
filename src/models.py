"""
models.py — Model definitions and hyperparameter grids for the CS 229 replication.

Defines 5 core models:
    1. Regularized Logistic Regression (L1/L2)
    2. Support Vector Machine (linear/RBF)
    3. Random Forest
    4. Gradient Boosted Trees (LightGBM)
    5. Ensemble Voting Classifier (soft voting over 1–4)

All models are wrapped in sklearn Pipelines with StandardScaler
to prevent data leakage during cross-validation.
"""

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from lightgbm import LGBMClassifier


# ---------------------------------------------------------------------------
#  Individual model factories  (return Pipeline + param_grid for GridSearchCV)
# ---------------------------------------------------------------------------

def get_logistic_regression():
    """Regularized Logistic Regression with L1/L2 penalty search."""
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=1000, solver="liblinear", random_state=42)),
    ])
    param_grid = {
        "clf__penalty": ["l1", "l2"],
        "clf__C": [0.01, 0.1, 1.0],
    }
    return pipe, param_grid


def get_svm():
    """Support Vector Machine with linear and RBF kernel search."""
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", SVC(probability=True, random_state=42)),
    ])
    param_grid = {
        "clf__C": [0.1, 1.0, 10.0],
        "clf__kernel": ["linear", "rbf"],
    }
    return pipe, param_grid


def get_random_forest():
    """Random Forest with tunable split and feature selection."""
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", RandomForestClassifier(n_estimators=200, random_state=42)),
    ])
    param_grid = {
        "clf__min_samples_split": [2, 10],
        "clf__max_features": ["sqrt", None],
    }
    return pipe, param_grid


def get_lgbm():
    """Gradient Boosted Trees via LightGBM."""
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LGBMClassifier(verbose=-1, random_state=42)),
    ])
    param_grid = {
        "clf__learning_rate": [0.05, 0.1],
        "clf__num_leaves": [15, 31],
    }
    return pipe, param_grid


def get_ensemble(lr_pipe, svm_pipe, rf_pipe, lgbm_pipe):
    """
    Soft-Voting Ensemble combining the 4 tuned base models.

    Parameters
    ----------
    lr_pipe, svm_pipe, rf_pipe, lgbm_pipe : fitted Pipeline objects
        Best estimators obtained after GridSearchCV.

    Returns
    -------
    VotingClassifier (already wraps the fitted estimators).
    """
    ensemble = VotingClassifier(
        estimators=[
            ("lr", lr_pipe),
            ("svm", svm_pipe),
            ("rf", rf_pipe),
            ("lgbm", lgbm_pipe),
        ],
        voting="soft",
    )
    return ensemble


# ---------------------------------------------------------------------------
#  Convenience registry
# ---------------------------------------------------------------------------

MODEL_REGISTRY = {
    "Logistic Regression": get_logistic_regression,
    "SVM": get_svm,
    "Random Forest": get_random_forest,
    "LightGBM": get_lgbm,
    # Ensemble is built separately after tuning the 4 base models
}
