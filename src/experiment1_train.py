"""
experiment1_train.py — Experiment 1: Full Model Comparison (Table 1)

Trains 5 models × 2 feature sets × 2 outcomes = 20 evaluations.
Uses 80/20 stratified train/test split and 5-fold GridSearchCV (roc_auc).
Saves outputs/table1_auroc_results.csv matching Table 1 of the paper.
"""

import os
import sys
import time
import warnings
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.metrics import roc_auc_score

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from src.models import MODEL_REGISTRY, get_ensemble

warnings.filterwarnings("ignore")

# ── Paths ─────────────────────────────────────────────────────────────────────
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── Load data ─────────────────────────────────────────────────────────────────
print("=" * 72)
print("  EXPERIMENT 1 — Full Model Comparison (Table 1)")
print("=" * 72)

X_knn = pd.read_csv(os.path.join(DATA_DIR, "chemo_X_knn_933.csv"))
X_binned = pd.read_csv(os.path.join(DATA_DIR, "chemo_X_binned_1367.csv"))
targets = pd.read_csv(os.path.join(DATA_DIR, "chemo_targets.csv"))

FEATURE_SETS = {"KNN": X_knn, "Binned": X_binned}
OUTCOMES = {"y_drop": targets["y_drop"], "y_threshold": targets["y_threshold"]}

print(f"  KNN feature matrix   : {X_knn.shape}")
print(f"  Binned feature matrix: {X_binned.shape}")
print(f"  y_drop  prevalence   : {targets['y_drop'].mean():.1%}")
print(f"  y_threshold prevalence: {targets['y_threshold'].mean():.1%}\n")


# ── Helper: train & evaluate one model on one (X, y) task ─────────────────────
def train_and_evaluate(pipe, param_grid, X_train, X_test, y_train, y_test):
    """Run GridSearchCV, refit best model, return test AUROC."""
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    grid = GridSearchCV(
        pipe,
        param_grid,
        scoring="roc_auc",
        cv=cv,
        n_jobs=-1,
        refit=True,
    )
    grid.fit(X_train, y_train)
    y_prob = grid.best_estimator_.predict_proba(X_test)[:, 1]
    auroc = roc_auc_score(y_test, y_prob)
    return auroc, grid.best_estimator_


# ── Main experiment loop ──────────────────────────────────────────────────────
results = {}  # {model_name: {task_key: auroc}}
best_base_models = {}  # {(fs_name, outcome_name): {model_name: fitted_pipeline}}

total = len(MODEL_REGISTRY) * len(FEATURE_SETS) * len(OUTCOMES)
counter = 0

for fs_name, X in FEATURE_SETS.items():
    for out_name, y in OUTCOMES.items():
        task_key = f"{fs_name}_{out_name}"

        # 80/20 stratified split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.20, random_state=42, stratify=y
        )

        fitted_for_ensemble = {}

        for model_name, factory_fn in MODEL_REGISTRY.items():
            counter += 1
            pipe, param_grid = factory_fn()
            t0 = time.time()

            auroc, best_est = train_and_evaluate(
                pipe, param_grid, X_train, X_test, y_train, y_test
            )

            elapsed = time.time() - t0
            results.setdefault(model_name, {})[task_key] = auroc
            fitted_for_ensemble[model_name] = best_est

            print(
                f"  [{counter:2d}/{total}] {model_name:<22s} | {task_key:<20s} "
                f"| AUROC = {auroc:.4f}  ({elapsed:5.1f}s)"
            )

        # ── Build Ensemble from the 4 tuned models ──────────────────────────
        counter += 1  # count the ensemble evaluation
        total_with_ens = len(MODEL_REGISTRY) * len(FEATURE_SETS) * len(OUTCOMES) + len(FEATURE_SETS) * len(OUTCOMES)

        ensemble = get_ensemble(
            fitted_for_ensemble["Logistic Regression"],
            fitted_for_ensemble["SVM"],
            fitted_for_ensemble["Random Forest"],
            fitted_for_ensemble["LightGBM"],
        )
        t0 = time.time()
        ensemble.fit(X_train, y_train)
        y_prob_ens = ensemble.predict_proba(X_test)[:, 1]
        auroc_ens = roc_auc_score(y_test, y_prob_ens)
        elapsed = time.time() - t0

        results.setdefault("Ensemble Voting", {})[task_key] = auroc_ens
        print(
            f"  [  /{total_with_ens}] {'Ensemble Voting':<22s} | {task_key:<20s} "
            f"| AUROC = {auroc_ens:.4f}  ({elapsed:5.1f}s)"
        )
        print()

# ── Build Table 1 DataFrame ──────────────────────────────────────────────────
MODEL_ORDER = [
    "Logistic Regression",
    "SVM",
    "Random Forest",
    "LightGBM",
    "Ensemble Voting",
]
TASK_ORDER = ["KNN_y_drop", "KNN_y_threshold", "Binned_y_drop", "Binned_y_threshold"]

rows = []
for m in MODEL_ORDER:
    row = {"Model": m}
    vals = []
    for t in TASK_ORDER:
        v = results[m][t]
        row[t] = round(v, 4)
        vals.append(v)
    row["Mean"] = round(np.mean(vals), 4)
    row["SD"] = round(np.std(vals, ddof=1), 4)
    rows.append(row)

# Outcome-level mean and SD (across models for each task)
outcome_mean = {"Model": "Outcome Mean"}
outcome_sd = {"Model": "Outcome SD"}
for t in TASK_ORDER:
    col_vals = [results[m][t] for m in MODEL_ORDER]
    outcome_mean[t] = round(np.mean(col_vals), 4)
    outcome_sd[t] = round(np.std(col_vals, ddof=1), 4)
outcome_mean["Mean"] = ""
outcome_mean["SD"] = ""
outcome_sd["Mean"] = ""
outcome_sd["SD"] = ""
rows.extend([outcome_mean, outcome_sd])

table1 = pd.DataFrame(rows)
table1.to_csv(os.path.join(OUTPUT_DIR, "table1_auroc_results.csv"), index=False)

print("\n" + "=" * 72)
print("  TABLE 1 — AUROC Results")
print("=" * 72)
print(table1.to_string(index=False))
print(f"\n  [OK] Saved to outputs/table1_auroc_results.csv")
print("=" * 72)
