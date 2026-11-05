"""Experiment 2: ensemble ROC stability and feature-group ablation analysis."""
from __future__ import annotations

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import auc, roc_auc_score, roc_curve
from sklearn.model_selection import StratifiedShuffleSplit, train_test_split
from src.models import OUTPUT_DIR, RANDOM_STATE, load_data, make_ensemble


def split_data(X, y):
    return train_test_split(X, y, test_size=0.20, stratify=y, random_state=RANDOM_STATE)


def bootstrap_ci(y_true, probabilities, n_resamples=200):
    """Percentile 95% AUROC CI from 200 test-set bootstrap resamples."""
    rng, y, values = np.random.default_rng(RANDOM_STATE), np.asarray(y_true), []
    for _ in range(n_resamples):
        idx = rng.integers(0, len(y), len(y))
        if len(np.unique(y[idx])) == 2:
            values.append(roc_auc_score(y[idx], probabilities[idx]))
    return tuple(np.quantile(values, [0.025, 0.975]).round(4))


def create_roc_figure(X, targets):
    """Plot five 80%-of-test resamples with mean and +/- 1 SD ROC curves."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
    mean_fpr = np.linspace(0, 1, 200)
    for ax, outcome, title in zip(axes, ["y_drop", "y_threshold"], ["Drop Outcome", "Threshold Outcome"]):
        X_train, X_test, y_train, y_test = split_data(X, targets[outcome])
        model = make_ensemble().fit(X_train, y_train)
        splitter, tprs, aucs = StratifiedShuffleSplit(n_splits=5, train_size=.8, random_state=RANDOM_STATE), [], []
        for fold, (subset, _) in enumerate(splitter.split(X_test, y_test), 1):
            fpr, tpr, _ = roc_curve(y_test.iloc[subset], model.predict_proba(X_test.iloc[subset])[:, 1])
            interp = np.interp(mean_fpr, fpr, tpr); interp[0] = 0
            tprs.append(interp); aucs.append(auc(fpr, tpr))
            ax.plot(fpr, tpr, color="#91b8de", alpha=.45, lw=1, label="Test sub-fold" if fold == 1 else None)
        avg, sd = np.mean(tprs, axis=0), np.std(tprs, axis=0); avg[-1] = 1
        ax.plot(mean_fpr, avg, color="#083b7a", lw=2.5, label=f"Mean ROC (AUC = {np.mean(aucs):.3f})")
        ax.fill_between(mean_fpr, np.maximum(avg-sd, 0), np.minimum(avg+sd, 1), color="#6fa8dc", alpha=.30, label="± 1 SD")
        ax.plot([0, 1], [0, 1], "r--", lw=1.5, label="Chance")
        ax.set(title=title, xlabel="False Positive Rate", ylabel="True Positive Rate", xlim=(0, 1), ylim=(0, 1.03))
        ax.grid(alpha=.2); ax.legend(loc="lower right", fontsize=8)
    fig.suptitle("Ensemble Voting ROC Stability — KNN Laboratory Features", fontweight="bold")
    fig.savefig(OUTPUT_DIR / "figure1_roc_curves.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def ablate(X, prefixes=(), only_promis=False):
    if only_promis:
        return X.loc[:, [c for c in X if c.startswith("PROMIS_")]]
    return X.loc[:, [c for c in X if not any(c.startswith(prefix) for prefix in prefixes)]]


def ablation_study(X, y):
    studies = [("Full model (none removed)", (), False), ("Without diagnoses", ("Diag_",), False),
        ("Without demographics", ("Demo_",), False), ("Without utilization", ("Util_",), False),
        ("Without prescriptions", ("Med_",), False), ("Without vitals", ("Vital_",), False),
        ("Without procedures", ("Proc_",), False), ("Without demographics + baseline scores", ("Demo_", "PROMIS_"), False),
        ("Baseline PROMIS scores only", (), True)]
    rows = []
    for label, prefixes, only_promis in studies:
        current = ablate(X, prefixes, only_promis)
        X_train, X_test, y_train, y_test = split_data(current, y)
        probabilities = make_ensemble().fit(X_train, y_train).predict_proba(X_test)[:, 1]
        score = roc_auc_score(y_test, probabilities); low, high = bootstrap_ci(y_test, probabilities)
        rows.append({"Ablated feature set": label, "Features retained": current.shape[1], "AUROC": score, "CI 95% lower": low, "CI 95% upper": high})
        print(f"{label}: {score:.3f} ({low:.3f}-{high:.3f})")
    return pd.DataFrame(rows)


def create_ablation_plot(table):
    plot = table.sort_values("AUROC"); y = np.arange(len(plot))
    errors = np.vstack([plot["AUROC"]-plot["CI 95% lower"], plot["CI 95% upper"]-plot["AUROC"]])
    fig, ax = plt.subplots(figsize=(10, 6), constrained_layout=True)
    colors = ["#083b7a" if "Full" in label else "#5b9bd5" for label in plot["Ablated feature set"]]
    ax.barh(y, plot["AUROC"], xerr=errors, color=colors, capsize=4)
    ax.set_yticks(y, plot["Ablated feature set"])
    ax.set(xlabel="Test AUROC (95% bootstrap CI)", title="Feature Ablation — Ensemble / KNN / Drop", xlim=(max(0, plot["CI 95% lower"].min()-.08), 1))
    ax.grid(axis="x", alpha=.25)
    fig.savefig(OUTPUT_DIR / "ablation_plot.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def main():
    OUTPUT_DIR.mkdir(exist_ok=True)
    X_knn, _, targets = load_data()
    create_roc_figure(X_knn, targets)
    table = ablation_study(X_knn, targets["y_drop"])
    table.round(4).to_csv(OUTPUT_DIR / "table2_ablation_results.csv", index=False)
    create_ablation_plot(table)
    print(f"Saved Experiment 2 outputs in {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
