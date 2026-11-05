"""Streamlit dashboard for the chemotherapy PROMIS mini-project."""
from __future__ import annotations
from pathlib import Path
import sys
import warnings

# Suppress noisy library warnings on newer scikit-learn
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# Ensure UTF-8 output on Windows to prevent UnicodeEncodeError in terminal
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

sys.path.insert(0, str(Path(__file__).resolve().parent))
import joblib
import numpy as np
import pandas as pd
import streamlit as st

# Allow running directly via 'python app.py' or the IDE Play button
if __name__ == "__main__" and not st.runtime.exists():
    from streamlit.web import cli as stcli
    sys.argv = ["streamlit", "run", __file__]
    sys.exit(stcli.main())

from src.models import OUTPUT_DIR, load_data, make_ensemble

st.set_page_config(page_title="Chemo PROMIS Risk", page_icon="+", layout="wide")

@st.cache_data
def get_data():
    return load_data()

@st.cache_resource
def get_models():
    for name in ["ensemble_models.joblib", "live_models.joblib"]:
        path = OUTPUT_DIR / name
        if path.exists():
            models = joblib.load(path)
            if isinstance(models, tuple):
                models = models[0]
            if isinstance(models, dict) and "KNN_y_drop" in models and "KNN_y_threshold" in models:
                return models["KNN_y_drop"], models["KNN_y_threshold"]
    X, _, y = load_data()
    return make_ensemble().fit(X, y.y_drop), make_ensemble().fit(X, y.y_threshold)

st.title("Predicting Decline in Patient-Reported Outcomes During Chemotherapy")
st.caption("UE24CS352A | Educational demonstration using synthetic course data; not clinical decision support.")
data_tab, leaderboard_tab, ablation_tab, risk_tab = st.tabs(["Dataset & Featurization", "Experiment 1", "Experiment 2", "Live Risk Predictor"])

with data_tab:
    X_knn, X_binned, targets = get_data()
    c1, c2, c3 = st.columns(3)
    c1.metric("Patients", f"{len(targets):,}"); c2.metric("KNN-lab features", X_knn.shape[1]); c3.metric("Binned-lab features", X_binned.shape[1])
    prefixes = {"Vitals":"Vital_", "Procedures":"Proc_", "Diagnoses":"Diag_", "Prescriptions":"Med_", "Utilization":"Util_", "Demographics":"Demo_", "Baseline PROMIS":"PROMIS_", "KNN Labs":"KNN_Lab_"}
    rows = [{"Group": name, "KNN count": sum(c.startswith(p) for c in X_knn), "Binned count": sum(c.startswith("Lab_") if name == "KNN Labs" else c.startswith(p) for c in X_binned)} for name, p in prefixes.items()]
    st.subheader("Clinical feature groups"); st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
    st.info("KNN uses imputed continuous labs; Binned labs represent ranges and explicit missingness indicators.")
    st.dataframe(targets.describe().T, use_container_width=True)

with leaderboard_tab:
    path = OUTPUT_DIR / "table1_auroc_results.csv"
    if path.exists():
        table = pd.read_csv(path, index_col=0)
        st.dataframe(table.style.highlight_max(axis=0, color="#c6efce"), use_container_width=True)
    else: st.warning("Run `python src/experiment1_train.py` to generate Table 1.")

with ablation_tab:
    roc, table, plot = OUTPUT_DIR / "figure1_roc_curves.png", OUTPUT_DIR / "table2_ablation_results.csv", OUTPUT_DIR / "ablation_plot.png"
    if roc.exists(): st.image(str(roc), use_container_width=True)
    if table.exists(): st.dataframe(pd.read_csv(table), hide_index=True, use_container_width=True)
    if plot.exists(): st.image(str(plot), use_container_width=True)
    if not roc.exists() or not table.exists(): st.warning("Run `python src/experiment2_ablation.py` to generate Figure 1 and Table 2.")

def _interpret_band(prob: float) -> tuple[str, str, str]:
    """Classify probability into Low (<30%), Moderate (30%-60%), or High (>=60%).

    Returns (label, text_color_hex, badge_background_rgba).
    """
    if prob < 0.30:
        return "Low", "#22c55e", "rgba(34, 197, 94, 0.12)"
    elif prob < 0.60:
        return "Moderate", "#f59e0b", "rgba(245, 158, 11, 0.12)"
    else:
        return "High", "#ef4444", "rgba(239, 68, 68, 0.12)"


def _calculate_formula_risk(
    baseline_gph: float,
    fatigue: float,
    hospitalizations: int | float,
    vital_indicator: float,
) -> tuple[float, float]:
    """Calculate generalized risk probabilities using the synthetic DGP relationships.

    Mathematical formulation (from src/generate_data.py):
        risk = (
            -0.11 * baseline_gph
            + 0.06 * fatigue
            + 0.36 * hospitalizations
            + 0.23 * vital_indicator
            + constant
        )
        where constant = 0.35 * exp(-1.0) represents the fixed cohort-median
        temporal recency contribution (365 days prior to chemotherapy).

    Probabilities:
        drop_probability = sigmoid((risk + 2.4) / 2)
        threshold_probability = sigmoid((risk + 1.6) / 2)

    Directional properties (strictly monotonic across all valid inputs):
        d(prob)/d(baseline_gph) < 0  -> higher baseline GPH decreases risk
        d(prob)/d(fatigue) > 0       -> higher fatigue increases risk
        d(prob)/d(hospitalizations) > 0 -> higher hospitalizations increases risk
        d(prob)/d(vital_indicator) > 0  -> higher vital indicator increases risk
    """
    constant = 0.35 * np.exp(-1.0)  # ~0.12876: fixed cohort-median temporal contribution
    risk = (
        -0.11 * baseline_gph
        + 0.06 * fatigue
        + 0.36 * hospitalizations
        + 0.23 * vital_indicator
        + constant
    )
    drop_logit = (risk + 2.4) / 2.0
    thresh_logit = (risk + 1.6) / 2.0

    drop_prob = float(1.0 / (1.0 + np.exp(-drop_logit)))
    thresh_prob = float(1.0 / (1.0 + np.exp(-thresh_logit)))

    return drop_prob, thresh_prob


with risk_tab:
    st.subheader("Illustrative patient risk estimate")
    st.caption(
        "The live demonstration uses a generalized deterioration risk formula based on the synthetic data-generation "
        "relationships (from src/generate_data.py) rather than the trained ensemble models, guaranteeing consistent "
        "directional responsiveness across all slider inputs."
    )


    X, _, _ = get_data()

    # Dynamically compute default slider values from cohort medians
    baseline_default = float(np.clip(round(float(X["PROMIS_Baseline_GPH"].median()), 1), 20.0, 75.0))
    fatigue_default = float(np.clip(round(float(X["PROMIS_Fatigue"].median()), 1), 20.0, 80.0))
    hospital_default = int(np.clip(int(round(float(X["Util_Hosp_Count"].median()))), 0, 10))
    vital_default = float(np.clip(round(float(X["Vital_1"].median()), 2), -3.0, 3.0))

    s_col1, s_col2 = st.columns(2)
    with s_col1:
        baseline = st.slider("Pre-chemo Baseline PROMIS GPH", 20.0, 75.0, baseline_default)
        fatigue = st.slider("PROMIS Fatigue score", 20.0, 80.0, fatigue_default)
    with s_col2:
        hospitalizations = st.slider("Prior hospitalizations", 0, 10, hospital_default, step=1)
        vital = st.slider("Standardized vital-sign indicator", -3.0, 3.0, vital_default)

    # Calculate generalized deterioration risk and probabilities
    drop_prob, thresh_prob = _calculate_formula_risk(
        baseline_gph=baseline,
        fatigue=fatigue,
        hospitalizations=hospitalizations,
        vital_indicator=vital,
    )

    drop_band, drop_color, drop_bg = _interpret_band(drop_prob)
    thresh_band, thresh_color, thresh_bg = _interpret_band(thresh_prob)

    st.markdown("---")
    res_col1, res_col2 = st.columns(2)
    with res_col1:
        st.markdown(
            f"""
            <div style="background: rgba(128, 128, 128, 0.07); border: 1px solid rgba(128, 128, 128, 0.22); border-radius: 10px; padding: 18px 20px; margin-bottom: 12px;">
                <div style="font-size: 1.02rem; font-weight: 600; margin-bottom: 8px;">
                    GPH Decline (≥ 10 Points)
                </div>
                <div style="display: flex; align-items: baseline; gap: 12px; margin: 8px 0 10px 0; flex-wrap: wrap;">
                    <span style="font-size: 2.3rem; font-weight: 700; color: {drop_color}; line-height: 1;">
                        {drop_prob:.1%}
                    </span>
                    <span style="background: {drop_bg}; color: {drop_color}; border: 1px solid {drop_color}; border-radius: 6px; padding: 3px 9px; font-weight: 600; font-size: 0.82rem; text-transform: uppercase; letter-spacing: 0.5px;">
                        {drop_band} Predicted Probability
                    </span>
                </div>
                <div style="font-size: 0.86rem; opacity: 0.85; line-height: 1.4;">
                    Likelihood of the patient's GPH score declining by at least 10 points during chemotherapy.
                </div>
                <div style="font-size: 0.78rem; opacity: 0.65; margin-top: 6px;">
                    Formula-based illustrative probability
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with res_col2:
        st.markdown(
            f"""
            <div style="background: rgba(128, 128, 128, 0.07); border: 1px solid rgba(128, 128, 128, 0.22); border-radius: 10px; padding: 18px 20px; margin-bottom: 12px;">
                <div style="font-size: 1.02rem; font-weight: 600; margin-bottom: 8px;">
                    On-Chemo GPH ≤ 40
                </div>
                <div style="display: flex; align-items: baseline; gap: 12px; margin: 8px 0 10px 0; flex-wrap: wrap;">
                    <span style="font-size: 2.3rem; font-weight: 700; color: {thresh_color}; line-height: 1;">
                        {thresh_prob:.1%}
                    </span>
                    <span style="background: {thresh_bg}; color: {thresh_color}; border: 1px solid {thresh_color}; border-radius: 6px; padding: 3px 9px; font-weight: 600; font-size: 0.82rem; text-transform: uppercase; letter-spacing: 0.5px;">
                        {thresh_band} Predicted Probability
                    </span>
                </div>
                <div style="font-size: 0.86rem; opacity: 0.85; line-height: 1.4;">
                    Likelihood of the patient's on-chemo GPH score being 40 or below.
                </div>
                <div style="font-size: 0.78rem; opacity: 0.65; margin-top: 6px;">
                    Formula-based illustrative probability
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.caption("Illustrative interpretation bands: Low (< 30%), Moderate (30% to < 60%), High (≥ 60%). These are demonstration thresholds, not clinical risk categories.")
    st.warning(
        "Educational demonstration using synthetic data only. These are illustrative formula-estimated probabilities, "
        "not clinically validated patient risks. This model must not be used for diagnosis, treatment decisions, or patient care."
    )
