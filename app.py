"""
app.py — Interactive Streamlit Dashboard for Faculty Demo
Course: UE24CS352A — Machine Learning Mini-Project
Paper:  Predicting a Decline in Patient Reported Outcomes for Cancer Patients
        on Chemotherapy (Ostberg & Peterson, Stanford CS 229)
"""

import os
import sys
import warnings
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

DATA_DIR = os.path.join(PROJECT_ROOT, "data")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "outputs")

# ═══════════════════════════════════════════════════════════════════════════════
#  PAGE CONFIG
# ═══════════════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="CS229 Chemo PRO Predictor",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS for a premium look ─────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
    
    .main-header {
        background: linear-gradient(135deg, #0f2027, #203a43, #2c5364);
        padding: 2rem 2.5rem;
        border-radius: 16px;
        margin-bottom: 2rem;
        box-shadow: 0 8px 32px rgba(0,0,0,0.18);
    }
    .main-header h1 {
        color: #ffffff;
        font-size: 1.85rem;
        font-weight: 700;
        margin-bottom: 0.3rem;
    }
    .main-header p {
        color: #a8d8ea;
        font-size: 0.95rem;
        margin: 0;
    }
    
    .metric-card {
        background: linear-gradient(135deg, #1a1a2e, #16213e);
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 12px;
        padding: 1.4rem;
        text-align: center;
        box-shadow: 0 4px 15px rgba(0,0,0,0.12);
        transition: transform 0.2s ease;
    }
    .metric-card:hover { transform: translateY(-3px); }
    .metric-card .value {
        font-size: 2rem;
        font-weight: 700;
        background: linear-gradient(90deg, #5b9bd5, #42e695);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .metric-card .label {
        font-size: 0.8rem;
        color: #8b98a5;
        text-transform: uppercase;
        letter-spacing: 1px;
        margin-top: 0.3rem;
    }
    
    .risk-gauge {
        border-radius: 12px;
        padding: 1.8rem;
        text-align: center;
        box-shadow: 0 4px 20px rgba(0,0,0,0.15);
    }
    .risk-low { background: linear-gradient(135deg, #0f3443, #34e89e22); border-left: 4px solid #34e89e; }
    .risk-med { background: linear-gradient(135deg, #3a2d00, #f7971e22); border-left: 4px solid #f7971e; }
    .risk-high { background: linear-gradient(135deg, #3a0d0d, #ff416c22); border-left: 4px solid #ff416c; }
    
    div[data-testid="stTabs"] button {
        font-weight: 600;
        font-size: 0.95rem;
    }
</style>
""", unsafe_allow_html=True)

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="main-header">
    <h1>🧬 Predicting Decline in Patient-Reported Outcomes</h1>
    <p>Replication of Ostberg &amp; Peterson (Stanford CS 229) &nbsp;•&nbsp;
       UE24CS352A Machine Learning Mini-Project &nbsp;•&nbsp;
       Simran Joshi (PES1UG24CS455) &amp; Sujay Hegde (PES1UG24CS478)</p>
</div>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
#  TABS
# ═══════════════════════════════════════════════════════════════════════════════
tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Dataset & Features",
    "🏆 Experiment 1 — Leaderboard",
    "📈 Experiment 2 — ROC & Ablation",
    "🩺 Live Patient Risk Predictor",
])


# ═══════════════════════════════════════════════════════════════════════════════
#  TAB 1: Dataset & Featurization Explorer
# ═══════════════════════════════════════════════════════════════════════════════
with tab1:
    st.subheader("Cohort & Feature Overview")

    # Load feature matrices for stats
    @st.cache_data
    def load_knn():
        return pd.read_csv(os.path.join(DATA_DIR, "chemo_X_knn_933.csv"))

    @st.cache_data
    def load_binned():
        return pd.read_csv(os.path.join(DATA_DIR, "chemo_X_binned_1367.csv"))

    @st.cache_data
    def load_targets():
        return pd.read_csv(os.path.join(DATA_DIR, "chemo_targets.csv"))

    X_knn_df = load_knn()
    X_bin_df = load_binned()
    tgt_df = load_targets()

    # Metric cards
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown('<div class="metric-card"><div class="value">1,252</div>'
                    '<div class="label">Patients in Cohort</div></div>',
                    unsafe_allow_html=True)
    with c2:
        st.markdown(f'<div class="metric-card"><div class="value">{X_knn_df.shape[1]}</div>'
                    '<div class="label">KNN Lab Features</div></div>',
                    unsafe_allow_html=True)
    with c3:
        st.markdown(f'<div class="metric-card"><div class="value">{X_bin_df.shape[1]}</div>'
                    '<div class="label">Binned Lab Features</div></div>',
                    unsafe_allow_html=True)
    with c4:
        pct_drop = tgt_df["y_drop"].mean() * 100
        st.markdown(f'<div class="metric-card"><div class="value">{pct_drop:.1f}%</div>'
                    '<div class="label">y_drop Prevalence</div></div>',
                    unsafe_allow_html=True)

    st.markdown("---")

    # Feature group breakdown
    FEATURE_GROUPS = {
        "Vital_": "Vitals",
        "Proc_": "Procedures",
        "Diag_": "Diagnoses",
        "Med_": "Prescriptions",
        "Util_": "Utilization",
        "Demo_": "Demographics",
        "PROMIS_": "Baseline PROMIS",
        "KNN_Lab_": "KNN Lab Values",
    }

    st.markdown("#### Clinical Feature Groups (KNN Labs Matrix)")
    group_data = []
    for prefix, label in FEATURE_GROUPS.items():
        cols = [c for c in X_knn_df.columns if c.startswith(prefix)]
        group_data.append({
            "Group": label,
            "Prefix": f"`{prefix}`",
            "Count": len(cols),
            "Example Columns": ", ".join(cols[:3]) + ("…" if len(cols) > 3 else ""),
        })
    st.dataframe(pd.DataFrame(group_data), use_container_width=True, hide_index=True)

    # KNN vs Binned comparison
    st.markdown("#### KNN Imputation vs. Tertile Binning")
    col_a, col_b = st.columns(2)
    with col_a:
        st.info("**KNN Labs (k=5 Imputer)**\n\n"
                "- 148 continuous lab features\n"
                "- Missing values imputed using 5 nearest neighbours\n"
                f"- Total features: **{X_knn_df.shape[1]}**")
    with col_b:
        st.info("**Binned Labs (Tertiles + Missing)**\n\n"
                "- Each lab split into 3 tertile bins + explicit `_Missing` indicator\n"
                "- Produces 4 binary columns per lab\n"
                f"- Total features: **{X_bin_df.shape[1]}**")

    # Target distributions
    st.markdown("#### Outcome Distributions")
    fig_tgt, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 3.5))
    for ax, col, title, colour in [
        (ax1, "y_drop", "≥10-pt GPH Drop", "#5b9bd5"),
        (ax2, "y_threshold", "On-Chemo GPH ≤ 40", "#e8746a"),
    ]:
        counts = tgt_df[col].value_counts().sort_index()
        ax.bar(["Negative (0)", "Positive (1)"], counts.values,
               color=[colour + "55", colour], edgecolor="white", linewidth=1.2)
        ax.set_title(title, fontweight="bold")
        ax.set_ylabel("Patients")
        for i, v in enumerate(counts.values):
            ax.text(i, v + 8, str(v), ha="center", fontweight="bold", fontsize=10)
    plt.tight_layout()
    st.pyplot(fig_tgt)
    plt.close(fig_tgt)


# ═══════════════════════════════════════════════════════════════════════════════
#  TAB 2: Experiment 1 — Model Leaderboard
# ═══════════════════════════════════════════════════════════════════════════════
with tab2:
    st.subheader("Experiment 1 — AUROC Model Comparison (Table 1)")

    table1_path = os.path.join(OUTPUT_DIR, "table1_auroc_results.csv")
    if os.path.exists(table1_path):
        table1 = pd.read_csv(table1_path)

        # Highlight the Ensemble Voting row
        st.markdown("> **Ensemble Voting** combines Logistic Regression, SVM, "
                    "Random Forest, and LightGBM via soft-voting probability averaging.")

        def highlight_ensemble(row):
            if row["Model"] == "Ensemble Voting":
                return ["background-color: #1a3a5c33; font-weight: bold"] * len(row)
            return [""] * len(row)

        st.dataframe(
            table1.style.apply(highlight_ensemble, axis=1).format(precision=4),
            use_container_width=True,
            hide_index=True,
        )

        # Bar chart of mean AUROC per model (exclude summary rows)
        model_rows = table1[~table1["Model"].isin(["Outcome Mean", "Outcome SD"])].copy()
        model_rows["Mean"] = pd.to_numeric(model_rows["Mean"], errors="coerce")

        fig_bar, ax_bar = plt.subplots(figsize=(8, 4))
        colors = ["#5b9bd5"] * 4 + ["#1a3a5c"]
        ax_bar.barh(model_rows["Model"], model_rows["Mean"], color=colors,
                    edgecolor="white", height=0.5)
        ax_bar.set_xlabel("Mean AUROC (across 4 tasks)", fontsize=11)
        ax_bar.set_title("Model Comparison — Mean AUROC", fontweight="bold", fontsize=13)
        ax_bar.invert_yaxis()
        for i, v in enumerate(model_rows["Mean"]):
            ax_bar.text(v + 0.002, i, f"{v:.4f}", va="center", fontsize=10, fontweight="bold")
        ax_bar.grid(axis="x", alpha=0.3)
        plt.tight_layout()
        st.pyplot(fig_bar)
        plt.close(fig_bar)
    else:
        st.warning("⚠️ `outputs/table1_auroc_results.csv` not found. "
                   "Run `python src/experiment1_train.py` first.")


# ═══════════════════════════════════════════════════════════════════════════════
#  TAB 3: Experiment 2 — ROC Curves & Ablation
# ═══════════════════════════════════════════════════════════════════════════════
with tab3:
    st.subheader("Experiment 2 — ROC Curves & Feature Ablation Study")

    col_roc, col_abl = st.columns([1.1, 0.9])

    with col_roc:
        st.markdown("#### Figure 1 — Ensemble Voting ROC Curves")
        roc_path = os.path.join(OUTPUT_DIR, "figure1_roc_curves.png")
        if os.path.exists(roc_path):
            st.image(roc_path, use_container_width=True)
        else:
            st.warning("Run `python src/experiment2_ablation.py` to generate.")

    with col_abl:
        st.markdown("#### Ablation Forest Plot")
        abl_plot_path = os.path.join(OUTPUT_DIR, "ablation_plot.png")
        if os.path.exists(abl_plot_path):
            st.image(abl_plot_path, use_container_width=True)
        else:
            st.warning("Run `python src/experiment2_ablation.py` to generate.")

    st.markdown("---")
    st.markdown("#### Table 2 — Feature Ablation Results")
    table2_path = os.path.join(OUTPUT_DIR, "table2_ablation_results.csv")
    if os.path.exists(table2_path):
        table2 = pd.read_csv(table2_path)

        def colour_auroc(val):
            try:
                v = float(val)
                if v >= 0.70:
                    return "color: #34e89e; font-weight: bold"
                elif v >= 0.60:
                    return "color: #f7971e"
                else:
                    return "color: #ff416c"
            except (ValueError, TypeError):
                return ""

        st.dataframe(
            table2.style.map(colour_auroc, subset=["AUROC"]).format(precision=4),
            use_container_width=True,
            hide_index=True,
        )
    else:
        st.warning("Run `python src/experiment2_ablation.py` to generate.")


# ═══════════════════════════════════════════════════════════════════════════════
#  TAB 4: Live Patient Risk Predictor
# ═══════════════════════════════════════════════════════════════════════════════
with tab4:
    st.subheader("🩺 Live Patient Risk Predictor")
    st.markdown("Adjust the patient's pre-chemotherapy clinical values below. "
                "The Ensemble Voting model predicts risk for both outcomes in real time.")

    # ── Load or build ensemble for live prediction ───────────────────────────
    @st.cache_resource
    def load_live_model():
        """Load pre-trained models from disk if available, else fit lightweight ensemble."""
        import joblib
        joblib_path = os.path.join(OUTPUT_DIR, "live_models.joblib")
        if os.path.exists(joblib_path):
            return joblib.load(joblib_path)

        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler
        from sklearn.linear_model import LogisticRegression
        from sklearn.svm import SVC
        from sklearn.ensemble import RandomForestClassifier, VotingClassifier
        from lightgbm import LGBMClassifier

        X = pd.read_csv(os.path.join(DATA_DIR, "chemo_X_knn_933.csv"))
        tgt = pd.read_csv(os.path.join(DATA_DIR, "chemo_targets.csv"))
        feature_cols = X.columns.tolist()

        models = {}
        for outcome in ["y_drop", "y_threshold"]:
            y = tgt[outcome]
            lr = Pipeline([("scaler", StandardScaler()),
                           ("clf", LogisticRegression(C=0.1, penalty="l2", solver="liblinear", random_state=42))])
            svm = Pipeline([("scaler", StandardScaler()),
                            ("clf", SVC(C=1.0, kernel="rbf", probability=True, random_state=42))])
            rf = Pipeline([("scaler", StandardScaler()),
                           ("clf", RandomForestClassifier(n_estimators=100, max_features="sqrt", random_state=42, n_jobs=-1))])
            lgbm = Pipeline([("scaler", StandardScaler()),
                             ("clf", LGBMClassifier(learning_rate=0.05, num_leaves=31, verbose=-1, random_state=42, n_jobs=-1))])
            ens = VotingClassifier(estimators=[("lr", lr), ("svm", svm), ("rf", rf), ("lgbm", lgbm)], voting="soft")
            ens.fit(X, y)
            models[outcome] = ens

        return models, feature_cols

    with st.spinner("Loading trained models..."):
        live_models, feature_cols = load_live_model()

    st.markdown("---")

    # ── Patient input sliders ─────────────────────────────────────────────────
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("##### 📋 Baseline PROMIS Scores")
        baseline_gph = st.slider("Baseline GPH (Global Physical Health)", 20.0, 80.0, 50.0, 0.5)
        baseline_gmh = st.slider("Baseline GMH (Global Mental Health)", 20.0, 80.0, 50.0, 0.5)
        fatigue = st.slider("Fatigue Score", 20.0, 80.0, 50.0, 0.5)
        pain = st.slider("Pain Score", 20.0, 80.0, 50.0, 0.5)
        depression = st.slider("Depression Score", 20.0, 80.0, 50.0, 0.5)
        physical_func = st.slider("Physical Function Score", 20.0, 80.0, 50.0, 0.5)

    with col2:
        st.markdown("##### 🏥 Utilization & Vitals")
        ed_visits = st.slider("ED Visits (prior 6 months)", 0, 10, 1)
        hospitalizations = st.slider("Hospitalizations", 0, 8, 0)
        avg_los = st.slider("Average Length of Stay (days)", 0.0, 20.0, 3.5, 0.5)
        vital_1 = st.slider("Vital Sign Index 1 (z-score)", -3.0, 3.0, 0.0, 0.1)
        vital_2 = st.slider("Vital Sign Index 2 (z-score)", -3.0, 3.0, 0.0, 0.1)
        age = st.slider("Patient Age", 21, 90, 62)

    # ── Build feature vector ──────────────────────────────────────────────────
    patient_vec = np.zeros(len(feature_cols))
    patient_df = pd.DataFrame([patient_vec], columns=feature_cols)

    # Map slider values to actual feature columns
    promis_map = {
        "PROMIS_Baseline_GPH": baseline_gph,
        "PROMIS_Baseline_GMH": baseline_gmh,
        "PROMIS_Fatigue": fatigue,
        "PROMIS_Pain": pain,
        "PROMIS_Depression": depression,
        "PROMIS_Physical_Func": physical_func,
    }
    util_map = {
        "Util_ED_Visits": ed_visits,
        "Util_Hosp_Count": hospitalizations,
        "Util_Avg_LOS": avg_los,
    }
    vital_map = {
        "Vital_1": vital_1,
        "Vital_2": vital_2,
    }
    demo_map = {"Demo_1": age}

    for col, val in {**promis_map, **util_map, **vital_map, **demo_map}.items():
        if col in patient_df.columns:
            patient_df[col] = val

    # ── Predict & display ─────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("### Predicted Risk")
    res_col1, res_col2 = st.columns(2)

    for res_col, outcome, label, emoji in [
        (res_col1, "y_drop", "≥10-Point GPH Drop", "📉"),
        (res_col2, "y_threshold", "On-Chemo GPH ≤ 40", "⚠️"),
    ]:
        prob = live_models[outcome].predict_proba(patient_df)[:, 1][0]
        risk_class = "risk-low" if prob < 0.35 else ("risk-med" if prob < 0.65 else "risk-high")
        risk_label = "Low" if prob < 0.35 else ("Moderate" if prob < 0.65 else "High")
        with res_col:
            st.markdown(
                f'<div class="risk-gauge {risk_class}">'
                f'<div style="font-size:2.2rem; font-weight:700;">{emoji} {prob:.1%}</div>'
                f'<div style="font-size:0.95rem; color:#8b98a5; margin-top:0.3rem;">'
                f'{label}</div>'
                f'<div style="font-size:0.85rem; margin-top:0.5rem; font-weight:600;">'
                f'Risk Level: {risk_label}</div></div>',
                unsafe_allow_html=True,
            )

    st.markdown("---")
    st.caption("⚙️ Model: Ensemble Voting (Logistic Regression + SVM + Random Forest + LightGBM) "
               "trained on KNN-imputed feature matrix. "
               "This is a research prototype — not for clinical use.")


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🧬 Project Info")
    st.markdown("""
    **Course:** UE24CS352A  
    **Paper:** *Predicting a Decline in PROs for Cancer Patients on Chemotherapy*  
    **Authors:** Ostberg & Peterson (Stanford CS 229)  
    
    ---
    **Team:**  
    - Simran Joshi (PES1UG24CS455)  
    - Sujay Hegde (PES1UG24CS478)  
    
    ---
    **Models:**  
    1. Logistic Regression (L1/L2)  
    2. SVM (Linear/RBF)  
    3. Random Forest  
    4. LightGBM  
    5. Ensemble Voting  
    """)
    st.markdown("---")
    st.markdown("📄 [View on GitHub](#) &nbsp; | &nbsp; 📝 [Paper PDF](#)")
