"""Create a deterministic, synthetic stand-in for the course project data.

The script is only needed when the supplied CSVs are absent.  It deliberately
does not represent real patients and should never be used for clinical care.
"""
from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.impute import KNNImputer

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
RNG = np.random.default_rng(42)


def _columns(prefix: str, n: int) -> list[str]:
    return [f"{prefix}{i}" for i in range(1, n + 1)]


def _binary_block(n_rows: int, names: list[str], probability: float = 0.15) -> pd.DataFrame:
    return pd.DataFrame(RNG.binomial(1, probability, (n_rows, len(names))), columns=names)


def main() -> None:
    DATA.mkdir(exist_ok=True)
    n = 1252
    # Recency is included in the data-generation process using the paper's
    # temporal weighting: w_i = exp(-t_i / 365), with t_i in days pre-chemo.
    days_before_chemo = RNG.uniform(0, 730, size=n)
    temporal_weight = np.exp(-days_before_chemo / 365)

    vital = pd.DataFrame(RNG.normal(0, 1, (n, 7)), columns=_columns("Vital_", 7))
    proc = _binary_block(n, _columns("Proc_", 291), 0.10)
    diag = _binary_block(n, _columns("Diag_", 266), 0.12)
    med = _binary_block(n, _columns("Med_", 121), 0.14)
    util = pd.DataFrame({
        "Util_ED_Visits": RNG.poisson(0.7, n),
        "Util_ED_Hosp_Prop": RNG.uniform(0, 1, n),
        "Util_Hosp_Count": RNG.poisson(0.35, n),
        "Util_Avg_LOS": RNG.gamma(1.5, 1.5, n),
        "Util_Psych_Visits": RNG.poisson(0.15, n),
    })
    demo = _binary_block(n, _columns("Demo_", 88), 0.25)
    promis = pd.DataFrame({
        "PROMIS_Baseline_GPH": RNG.normal(48, 9, n),
        "PROMIS_Baseline_GMH": RNG.normal(49, 9, n),
        "PROMIS_Fatigue": RNG.normal(50, 10, n),
        "PROMIS_Pain": RNG.normal(48, 10, n),
        "PROMIS_Depression": RNG.normal(47, 10, n),
        "PROMIS_Physical_Func": RNG.normal(48, 9, n),
    })
    labs_raw = RNG.normal(0, 1, (n, 148))
    labs_raw[RNG.random(labs_raw.shape) < 0.08] = np.nan
    knn_labs = pd.DataFrame(KNNImputer(n_neighbors=5).fit_transform(labs_raw), columns=_columns("KNN_Lab_", 148))
    binned_parts = []
    for j in range(148):
        raw = labs_raw[:, j]
        observed = raw[~np.isnan(raw)]
        lo, hi = np.quantile(observed, [1 / 3, 2 / 3])
        binned_parts.extend([
            (raw <= lo).astype(int), ((raw > lo) & (raw <= hi)).astype(int),
            (raw > hi).astype(int), np.isnan(raw).astype(int),
        ])
    binned_labs = pd.DataFrame(np.column_stack(binned_parts), columns=_columns("Lab_", 592))
    # 592 encodings are reduced to 582 to preserve the documented feature count.
    binned_labs = binned_labs.iloc[:, :582]
    common = pd.concat([vital, proc, diag, med, util, demo, promis], axis=1)
    X_knn = pd.concat([common, knn_labs], axis=1)
    X_binned = pd.concat([common, binned_labs], axis=1)
    risk = (-0.11 * promis["PROMIS_Baseline_GPH"] + 0.06 * promis["PROMIS_Fatigue"]
            + 0.36 * util["Util_Hosp_Count"] + 0.23 * vital["Vital_1"]
            + 0.35 * temporal_weight + RNG.normal(0, 1.1, n))
    drop_probability = 1 / (1 + np.exp(-((risk + 2.4) / 2)))
    threshold_probability = 1 / (1 + np.exp(-((risk + 1.6) / 2)))
    targets = pd.DataFrame({
        "y_drop": RNG.binomial(1, drop_probability),
        "y_threshold": RNG.binomial(1, threshold_probability),
    })
    X_knn.to_csv(DATA / "chemo_X_knn_933.csv", index=False)
    X_binned.to_csv(DATA / "chemo_X_binned_1367.csv", index=False)
    targets.to_csv(DATA / "chemo_targets.csv", index=False)
    print(f"Created synthetic data in {DATA} ({n} patients; {X_knn.shape[1]} / {X_binned.shape[1]} features).")


if __name__ == "__main__":
    main()
