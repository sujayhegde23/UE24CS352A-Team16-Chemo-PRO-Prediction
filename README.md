# 🧬 Predicting a Decline in Patient-Reported Outcomes for Cancer Patients on Chemotherapy

**Course:** UE24CS352A — Machine Learning Mini-Project  
**Paper Replicated:** *Predicting a Decline in Patient Reported Outcomes for Cancer Patients on Chemotherapy* — Nicolai Ostberg & Dylan Peterson (Stanford CS 229)

---

## 👥 Team

| Name | SRN |
|------|-----|
| Simran Joshi | PES1UG24CS455 |
| Sujay Hegde | PES1UG24CS478 |

---

## 📁 Project Structure

```
├── data/
│   ├── chemo_X_knn_933.csv       # 1252 × 932  (KNN-imputed labs)
│   ├── chemo_X_binned_1367.csv   # 1252 × 1366 (Tertile-binned labs)
│   └── chemo_targets.csv         # 1252 × 2    (y_drop, y_threshold)
├── src/
│   ├── generate_data.py          # Generates the 3 CSV datasets
│   ├── models.py                 # 5 ML model definitions & grids
│   ├── experiment1_train.py      # Experiment 1: Table 1 (5 models × 4 tasks)
│   └── experiment2_ablation.py   # Experiment 2: ROC curves + Ablation
├── outputs/
│   ├── table1_auroc_results.csv  # Table 1 — Model Comparison
│   ├── table2_ablation_results.csv # Table 2 — Feature Ablation
│   ├── figure1_roc_curves.png    # Figure 1 — ROC with ±1 SD
│   └── ablation_plot.png         # Forest/Bar plot for Ablation
├── app.py                        # Streamlit interactive dashboard
├── requirements.txt
└── README.md
```

---

## 🔬 Overview

This project replicates the Stanford CS 229 paper's methodology for predicting clinical deterioration in cancer patients undergoing chemotherapy, measured through PROMIS Global Physical Health (GPH) scores.

### Two Outcomes

| Outcome | Definition |
|---------|------------|
| **y_drop** | ≥ 10-point decline in GPH score from baseline |
| **y_threshold** | On-chemotherapy GPH score ≤ 40 |

### Two Feature Engineering Schemes

| Scheme | Description | Total Features |
|--------|-------------|----------------|
| **KNN Labs** | Missing lab values imputed via KNNImputer (k=5) | 932 |
| **Binned Labs** | Labs split into tertiles + explicit `_Missing` indicator | 1,366 |

### Five Models

1. **Regularized Logistic Regression** (L1/L2 penalty)
2. **Support Vector Machine** (Linear / RBF kernel)
3. **Random Forest** (200 trees)
4. **LightGBM** (Gradient Boosted Trees)
5. **Ensemble Voting** (Soft-voting over models 1–4)

---

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Generate Data (if CSVs not present)

```bash
python src/generate_data.py
```

### 3. Run Experiment 1 — Model Comparison (Table 1)

```bash
python src/experiment1_train.py
```

### 4. Run Experiment 2 — ROC Curves & Ablation (Table 2 + Figures)

```bash
python src/experiment2_ablation.py
```

### 5. Launch Interactive Dashboard

```bash
# Using standard streamlit CLI:
streamlit run app.py

# Or if streamlit is not in your system PATH (recommended on Windows):
python -m streamlit run app.py
```

Open `http://localhost:8501` in your browser.

---

## 📊 Key Results

### Experiment 1 — Table 1
Comparison of AUROC scores across 5 models × 2 feature sets × 2 outcomes. The Ensemble Voting classifier consistently achieves the highest mean AUROC.

### Experiment 2 — Feature Ablation
Systematic removal of feature groups reveals that **baseline PROMIS scores** are the single most predictive feature category. Removing procedures or diagnoses has moderate impact, while dropping vitals or utilization features alone has minimal effect.

---

## 🖥️ Streamlit Dashboard

The interactive dashboard (`app.py`) provides four views:

1. **Dataset & Features** — Cohort statistics, feature group breakdowns, KNN vs Binned comparison
2. **Experiment 1 Leaderboard** — Table 1 with visual model comparison
3. **Experiment 2 ROC & Ablation** — Figure 1 ROC curves and ablation analysis
4. **Live Patient Risk Predictor** — Real-time risk scoring with adjustable patient parameters

---

## ⚖️ License

Academic use only — course project for UE24CS352A.
