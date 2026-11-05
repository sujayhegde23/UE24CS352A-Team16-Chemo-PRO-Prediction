"""Experiment 1: cross-validated AUROC comparison for four task matrices."""
from __future__ import annotations

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split

from src.models import OUTPUT_DIR, RANDOM_STATE, ensemble_from_best, load_data, model_search_spaces


def main() -> None:
    OUTPUT_DIR.mkdir(exist_ok=True)
    X_knn, X_binned, targets = load_data()
    matrices = {"KNN": X_knn, "Binned": X_binned}
    outcomes = ["y_drop", "y_threshold"]
    scores: dict[str, dict[str, float]] = {name: {} for name in [*model_search_spaces(), "Ensemble Voting"]}
    saved_models = {}
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    for matrix_name, X in matrices.items():
        for outcome in outcomes:
            y = targets[outcome]
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, stratify=y, random_state=RANDOM_STATE)
            winners = {}
            task = f"{matrix_name}_{outcome}"
            print(f"\nRunning {task}...")
            for model_name, (pipeline, grid) in model_search_spaces().items():
                search = GridSearchCV(pipeline, grid, scoring="roc_auc", cv=cv, n_jobs=-1, refit=True)
                search.fit(X_train, y_train)
                winners[model_name] = search.best_estimator_
                scores[model_name][task] = roc_auc_score(y_test, search.predict_proba(X_test)[:, 1])
                print(f"  {model_name}: {scores[model_name][task]:.3f} (CV={search.best_score_:.3f})")
            ensemble = ensemble_from_best(winners)
            ensemble.fit(X_train, y_train)
            scores["Ensemble Voting"][task] = roc_auc_score(y_test, ensemble.predict_proba(X_test)[:, 1])
            saved_models[task] = ensemble
            print(f"  Ensemble Voting: {scores['Ensemble Voting'][task]:.3f}")
    columns = ["KNN_y_drop", "KNN_y_threshold", "Binned_y_drop", "Binned_y_threshold"]
    table = pd.DataFrame.from_dict(scores, orient="index").reindex(columns=columns)
    table["Mean"] = table.mean(axis=1)
    table["SD"] = table[columns].std(axis=1, ddof=1)
    table.loc["Outcome Mean"] = table.mean(axis=0)
    table.loc["Outcome SD"] = table.std(axis=0, ddof=1)
    table.index.name = "Model"
    table.round(4).to_csv(OUTPUT_DIR / "table1_auroc_results.csv")
    joblib.dump(saved_models, OUTPUT_DIR / "ensemble_models.joblib")
    print(f"\nSaved {OUTPUT_DIR / 'table1_auroc_results.csv'} and trained ensemble models.")


if __name__ == "__main__":
    main()
