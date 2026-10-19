"""
experiment2_ablation.py -- Experiment 2: ROC Curves + Feature Ablation Study

1. Figure 1: Side-by-side ROC curves (Drop & Threshold outcomes)
   using the Ensemble Voting model on KNN Labs, with 5 sub-fold
   variability, Mean ROC +/- 1 SD shaded region.

2. Table 2: Feature ablation study -- drop one feature group at a time,
   retrain Ensemble Voting on KNN Labs + y_drop, compute test AUROC
   with 95% bootstrap CI (200 resamples).

Saves:
  - outputs/figure1_roc_curves.png
  - outputs/table2_ablation_results.csv
  - outputs/ablation_plot.png
  - outputs/live_models.joblib
"""

import os
import sys
import time
import warnings

# Ensure UTF-8 output on Windows
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from lightgbm import LGBMClassifier

warnings.filterwarnings("ignore")

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

DATA_DIR = os.path.join(PROJECT_ROOT, "data")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# -----------------------------------------------------------------------------
#  Load data
# -----------------------------------------------------------------------------
X_knn = pd.read_csv(os.path.join(DATA_DIR, "chemo_X_knn_933.csv"))
targets = pd.read_csv(os.path.join(DATA_DIR, "chemo_targets.csv"))

y_drop = targets["y_drop"]
y_threshold = targets["y_threshold"]

print("=" * 72)
print("  EXPERIMENT 2 -- ROC Curves & Feature Ablation Study")
print("=" * 72)
print(f"  Feature matrix: {X_knn.shape}")
print(f"  y_drop   prev : {y_drop.mean():.1%}")
print(f"  y_thresh prev : {y_threshold.mean():.1%}\n")

# Global train/test split (same random_state as Experiment 1)
X_train_full, X_test_full, y_train_drop, y_test_drop = train_test_split(
    X_knn, y_drop, test_size=0.20, random_state=42, stratify=y_drop
)
_, _, y_train_thresh, y_test_thresh = train_test_split(
    X_knn, y_threshold, test_size=0.20, random_state=42, stratify=y_threshold
)


# =============================================================================
#  Helper: Build & fit Ensemble on given (X_train, y_train)
# =============================================================================
def make_ensemble():
    """Construct pipeline for the 4 base models with tuned hyperparameters."""
    lr = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(C=0.1, penalty="l2", solver="liblinear", random_state=42)),
    ])
    svm = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", SVC(C=1.0, kernel="rbf", probability=True, random_state=42, cache_size=500)),
    ])
    rf = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", RandomForestClassifier(n_estimators=200, max_features="sqrt",
                                       min_samples_split=5, random_state=42, n_jobs=-1)),
    ])
    lgbm = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LGBMClassifier(learning_rate=0.05, num_leaves=31,
                              verbose=-1, random_state=42, n_jobs=-1)),
    ])
    return VotingClassifier(
        estimators=[
            ("lr", lr),
            ("svm", svm),
            ("rf", rf),
            ("lgbm", lgbm),
        ],
        voting="soft",
    )


def fit_ensemble(X_tr, y_tr):
    ens = make_ensemble()
    ens.fit(X_tr, y_tr)
    return ens


# =============================================================================
#  FIGURE 1 -- ROC Curves with sub-fold variability
# =============================================================================
print("--- Figure 1: ROC Curves with sub-fold variability ------------------")

fig, axes = plt.subplots(1, 2, figsize=(14, 6))
mean_fpr = np.linspace(0, 1, 200)

fitted_live_models = {}

for ax, y_train, y_test, outcome_key, title in [
    (axes[0], y_train_drop, y_test_drop, "y_drop", "Drop Outcome (>=10-pt GPH decline)"),
    (axes[1], y_train_thresh, y_test_thresh, "y_threshold", "Threshold Outcome (GPH <= 40)"),
]:
    t0 = time.time()
    ensemble = fit_ensemble(X_train_full, y_train)
    fitted_live_models[outcome_key] = ensemble
    print(f"  Ensemble trained for '{title}' in {time.time()-t0:.1f}s")

    # 5 sub-folds of the test set (each uses 80% of test data)
    tprs = []
    aucs = []
    rng = np.random.RandomState(42)

    for fold_i in range(5):
        n_test = len(X_test_full)
        sub_idx = rng.choice(n_test, size=int(0.8 * n_test), replace=False)
        X_sub = X_test_full.iloc[sub_idx]
        y_sub = y_test.iloc[sub_idx]

        y_prob = ensemble.predict_proba(X_sub)[:, 1]
        fpr, tpr, _ = roc_curve(y_sub, y_prob)
        aucs.append(roc_auc_score(y_sub, y_prob))
        tprs.append(np.interp(mean_fpr, fpr, tpr))
        tprs[-1][0] = 0.0

    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    std_tpr = np.std(tprs, axis=0)
    mean_auc = np.mean(aucs)
    std_auc = np.std(aucs)

    # Plot
    ax.plot(mean_fpr, mean_tpr, color="#1a3a5c", linewidth=2.5,
            label=f"Mean ROC (AUROC = {mean_auc:.3f} +/- {std_auc:.3f})")
    ax.fill_between(mean_fpr, mean_tpr - std_tpr, mean_tpr + std_tpr,
                    color="#5b9bd5", alpha=0.25, label="+/- 1 SD")
    ax.plot([0, 1], [0, 1], "r--", linewidth=1.2, label="Chance")
    ax.set_xlim([-0.02, 1.02])
    ax.set_ylim([-0.02, 1.02])
    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate", fontsize=12)
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.legend(loc="lower right", fontsize=10)
    ax.grid(alpha=0.3)

fig.suptitle("Figure 1 -- Ensemble Voting ROC Curves (KNN Labs)",
             fontsize=14, fontweight="bold", y=1.01)
plt.tight_layout()
fig.savefig(os.path.join(OUTPUT_DIR, "figure1_roc_curves.png"),
            dpi=200, bbox_inches="tight")
plt.close(fig)
print("  [OK] Saved outputs/figure1_roc_curves.png\n")

# Save live models for fast Tab 4 in Streamlit
joblib.dump((fitted_live_models, X_knn.columns.tolist()),
            os.path.join(OUTPUT_DIR, "live_models.joblib"))
print("  [OK] Saved outputs/live_models.joblib\n")


# =============================================================================
#  TABLE 2 -- Feature Ablation Study
# =============================================================================
print("--- Table 2: Feature Ablation Study ---------------------------------")

ABLATION_SPECS = [
    ("All Features (Baseline)", []),
    ("Drop Diagnoses", ["Diag_"]),
    ("Drop Demographics", ["Demo_"]),
    ("Drop Utilization", ["Util_"]),
    ("Drop Prescriptions", ["Med_"]),
    ("Drop Vitals", ["Vital_"]),
    ("Drop Procedures", ["Proc_"]),
    ("Drop Demo & PROMIS", ["Demo_", "PROMIS_"]),
    ("Only PROMIS (drop all else)", ["Vital_", "Proc_", "Diag_", "Med_", "Util_", "Demo_", "KNN_Lab_"]),
]


def drop_columns_by_prefix(df, prefixes):
    """Return df with columns matching any prefix removed."""
    if not prefixes:
        return df
    drop_cols = [c for c in df.columns if any(c.startswith(p) for p in prefixes)]
    return df.drop(columns=drop_cols)


def bootstrap_auroc_ci(y_true, y_prob, n_boot=200, ci=0.95, seed=42):
    """Compute AUROC point estimate and 95% bootstrap CI."""
    rng = np.random.RandomState(seed)
    n = len(y_true)
    y_true_arr = np.array(y_true)
    y_prob_arr = np.array(y_prob)
    boot_aucs = []
    for _ in range(n_boot):
        idx = rng.choice(n, n, replace=True)
        if len(np.unique(y_true_arr[idx])) < 2:
            continue
        boot_aucs.append(roc_auc_score(y_true_arr[idx], y_prob_arr[idx]))
    alpha = (1 - ci) / 2
    lower = np.percentile(boot_aucs, 100 * alpha)
    upper = np.percentile(boot_aucs, 100 * (1 - alpha))
    return lower, upper


ablation_rows = []
for label, prefixes_to_drop in ABLATION_SPECS:
    t0 = time.time()

    X_tr = drop_columns_by_prefix(X_train_full, prefixes_to_drop)
    X_te = drop_columns_by_prefix(X_test_full, prefixes_to_drop)

    n_features = X_tr.shape[1]
    ensemble = fit_ensemble(X_tr, y_train_drop)

    y_prob = ensemble.predict_proba(X_te)[:, 1]
    auroc = roc_auc_score(y_test_drop, y_prob)
    ci_lo, ci_hi = bootstrap_auroc_ci(y_test_drop, y_prob, n_boot=200)

    elapsed = time.time() - t0
    ablation_rows.append({
        "Feature Set": label,
        "Num Features": n_features,
        "AUROC": round(auroc, 4),
        "95% CI Lower": round(ci_lo, 4),
        "95% CI Upper": round(ci_hi, 4),
    })
    print(f"  {label:<35s}  feats={n_features:5d}  "
          f"AUROC={auroc:.4f}  95% CI=[{ci_lo:.4f}, {ci_hi:.4f}]  ({elapsed:.1f}s)")

table2 = pd.DataFrame(ablation_rows)
table2.to_csv(os.path.join(OUTPUT_DIR, "table2_ablation_results.csv"), index=False)
print(f"\n  [OK] Saved outputs/table2_ablation_results.csv")


# -----------------------------------------------------------------------------
#  Ablation bar plot
# -----------------------------------------------------------------------------
print("\n--- Generating ablation_plot.png ------------------------------------")
fig, ax = plt.subplots(figsize=(10, 6))

labels = table2["Feature Set"].tolist()
aurocs = table2["AUROC"].tolist()
ci_lo = table2["95% CI Lower"].tolist()
ci_hi = table2["95% CI Upper"].tolist()
errors = [[a - lo for a, lo in zip(aurocs, ci_lo)],
          [hi - a for a, hi in zip(aurocs, ci_hi)]]

colors = ["#1a3a5c" if i == 0 else "#5b9bd5" for i in range(len(labels))]

y_pos = np.arange(len(labels))
ax.barh(y_pos, aurocs, xerr=errors, color=colors, edgecolor="white",
        capsize=4, height=0.6, zorder=3)
ax.set_yticks(y_pos)
ax.set_yticklabels(labels, fontsize=10)
ax.set_xlabel("Test AUROC (y_drop)", fontsize=12)
ax.set_title("Feature Ablation Study -- Ensemble Voting (KNN Labs, Drop Outcome)",
             fontsize=13, fontweight="bold")
ax.invert_yaxis()
ax.grid(axis="x", alpha=0.3, zorder=0)
ax.set_xlim(0.4, max(aurocs) + 0.08)

# Annotate values
for i, v in enumerate(aurocs):
    ax.text(v + 0.005, i, f"{v:.3f}", va="center", fontsize=9, fontweight="bold")

plt.tight_layout()
fig.savefig(os.path.join(OUTPUT_DIR, "ablation_plot.png"), dpi=200, bbox_inches="tight")
plt.close(fig)
print("  [OK] Saved outputs/ablation_plot.png")

print("\n" + "=" * 72)
print("  EXPERIMENT 2 COMPLETE")
print("=" * 72)
