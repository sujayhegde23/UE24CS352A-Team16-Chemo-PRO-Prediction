import numpy as np
import pandas as pd
from sklearn.impute import KNNImputer

# Set random seed for reproducibility across team members
np.random.seed(42)
N = 1252  # Exact patient cohort size from Ostberg & Peterson (CS 229)

print("Step 1: Simulating multi-visit longitudinal vitals & labs (6 months pre-chemo)...")
# Simulate 3 pre-chemo clinic visits per patient within 180 days (6 months) prior to chemo
# Paper formula: w_i = exp(-t / 365), where t is days before chemo start
days_prior = np.random.randint(1, 181, size=(N, 3))  # 3 visits between 1 and 180 days prior
weights = np.exp(-days_prior / 365.0)                # Shape: (N, 3)
weights_norm = weights / weights.sum(axis=1, keepdims=True)

# 1. Vitals (7 features time-weighted across visits)
raw_vitals_visits = np.random.normal(0, 1, size=(N, 3, 7))
time_weighted_vitals = np.einsum('ij,ijk->ik', weights_norm, raw_vitals_visits)
df_vitals = pd.DataFrame(time_weighted_vitals, columns=[f'Vital_{i}' for i in range(1, 8)])

# 2. Raw Labs (148 features time-weighted across visits, with 35% clinical missingness)
raw_labs_visits = np.random.normal(0, 1, size=(N, 3, 148))
time_weighted_labs = np.einsum('ij,ijk->ik', weights_norm, raw_labs_visits)
missing_mask = np.random.rand(N, 148) < 0.35
time_weighted_labs[missing_mask] = np.nan
df_raw_labs = pd.DataFrame(time_weighted_labs, columns=[f'Lab_{i}' for i in range(1, 149)])

print("Step 2: Generating Baseline PROMIS, Demographics, Utilization, and Clinical Histories...")
# 3. Baseline PROMIS Scores (6 features: population mean = 50, SD = 10)
promis_cols = ['PROMIS_Baseline_GPH', 'PROMIS_Baseline_GMH', 'PROMIS_Fatigue', 
               'PROMIS_Pain', 'PROMIS_Depression', 'PROMIS_Physical_Func']
df_promis = pd.DataFrame(
    np.random.normal(loc=50, scale=10, size=(N, 6)).clip(20, 80), 
    columns=promis_cols
)

# 4. Healthcare Utilization (5 features in 6 months prior to chemo)
util_cols = ['Util_ED_Visits', 'Util_ED_Hosp_Prop', 'Util_Hosp_Count', 'Util_Avg_LOS', 'Util_Psych_Visits']
df_util = pd.DataFrame({
    'Util_ED_Visits': np.random.poisson(1.2, size=N),
    'Util_ED_Hosp_Prop': np.random.uniform(0, 1, size=N),
    'Util_Hosp_Count': np.random.poisson(0.8, size=N),
    'Util_Avg_LOS': np.random.exponential(3.5, size=N),
    'Util_Psych_Visits': np.random.poisson(0.4, size=N)
})

# 5. Demographics (88 features: Age + one-hot categorical demographics)
df_demo = pd.DataFrame(np.random.binomial(1, 0.2, size=(N, 88)), columns=[f'Demo_{i}' for i in range(1, 89)])
df_demo['Demo_1'] = np.random.normal(62, 12, size=N).clip(21, 90)  # Patient Age

# 6. Diagnoses (266 binary features, present in >= 2.5% of cohort)
df_diag = pd.DataFrame(np.random.binomial(1, 0.08, size=(N, 266)), columns=[f'Diag_{i}' for i in range(1, 267)])

# 7. Procedures (291 time-weighted features)
df_proc = pd.DataFrame(
    np.random.exponential(0.3, size=(N, 291)) * np.random.binomial(1, 0.08, size=(N, 291)),
    columns=[f'Proc_{i}' for i in range(1, 292)]
)

# 8. Medications (121 binary medication class features + summary counts)
df_meds = pd.DataFrame(np.random.binomial(1, 0.12, size=(N, 121)), columns=[f'Med_{i}' for i in range(1, 122)])

print("Step 3: Applying KNN Imputation vs. Tertile Binning on Lab Values...")
# SCHEME A: KNN Imputed Labs (148 features)
knn_imputer = KNNImputer(n_neighbors=5)
df_knn_labs = pd.DataFrame(
    knn_imputer.fit_transform(df_raw_labs), 
    columns=[f'KNN_Lab_{i}' for i in range(1, 149)]
)

# SCHEME B: Binned Labs (Tertiles + Missingness Indicator = 582 features)
binned_frames = []
for col in df_raw_labs.columns:
    s = df_raw_labs[col]
    tertiles = pd.qcut(s, q=3, labels=['T1', 'T2', 'T3'])
    dummies = pd.get_dummies(tertiles, prefix=col, dtype=int)
    dummies[f'{col}_Missing'] = s.isna().astype(int)
    binned_frames.append(dummies)
df_binned_labs = pd.concat(binned_frames, axis=1).iloc[:, :582]

print("Step 4: Generating On-Chemotherapy Outcomes (y_drop & y_threshold)...")
# Latent clinical deterioration signal (weighted heavily on Baseline PROMIS + Procedures + Vitals)
latent_decline = (
    -0.55 * ((df_promis['PROMIS_Baseline_GPH'] - 50) / 10.0)
    - 0.20 * ((df_promis['PROMIS_Fatigue'] - 50) / 10.0)
    - 0.15 * df_proc.iloc[:, :15].sum(axis=1)
    - 0.12 * df_vitals['Vital_1']
    - 0.10 * df_knn_labs.iloc[:, :10].mean(axis=1)
    + np.random.normal(0, 0.9, size=N)
)

# Simulate on-chemo GPH score
on_chemo_gph = (df_promis['PROMIS_Baseline_GPH'] - 6.5 + 5.5 * latent_decline).clip(15, 80)

# Apply exact target formulas from Page 2 of the PDF
y_drop = ((on_chemo_gph - df_promis['PROMIS_Baseline_GPH']) <= -10).astype(int)
y_threshold = (on_chemo_gph <= 40).astype(int)
df_targets = pd.DataFrame({'y_drop': y_drop, 'y_threshold': y_threshold})

# Combine into Final Feature Matrices
base_features = pd.concat([df_vitals, df_proc, df_diag, df_meds, df_util, df_demo, df_promis], axis=1)
X_knn_933 = pd.concat([base_features, df_knn_labs], axis=1)
X_binned_1367 = pd.concat([base_features, df_binned_labs], axis=1)

# Export to CSV files
import os
data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
os.makedirs(data_dir, exist_ok=True)
X_knn_933.to_csv(os.path.join(data_dir, "chemo_X_knn_933.csv"), index=False)
X_binned_1367.to_csv(os.path.join(data_dir, "chemo_X_binned_1367.csv"), index=False)
df_targets.to_csv(os.path.join(data_dir, "chemo_targets.csv"), index=False)

print("\n--- GENERATION COMPLETE ---")
print(f"1. KNN Feature Matrix saved:    'chemo_X_knn_933.csv'    -> Shape: {X_knn_933.shape}")
print(f"2. Binned Feature Matrix saved: 'chemo_X_binned_1367.csv' -> Shape: {X_binned_1367.shape}")
print(f"3. Target Labels saved:         'chemo_targets.csv'      -> Shape: {df_targets.shape}")
print(f"   - Drop Outcome (y_drop = 1):           {y_drop.mean():.1%} positive class")
print(f"   - Threshold Outcome (y_threshold = 1): {y_threshold.mean():.1%} positive class")