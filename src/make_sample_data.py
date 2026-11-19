"""Generate a SYNTHETIC dataset with the same columns as the Framingham heart-disease CSV.

For smoke-testing the pipeline only - the values are random and carry no medical meaning.
Usage:  python src/make_sample_data.py [--rows 4000]
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def make(rows: int, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    age = rng.integers(32, 70, rows)
    smoker = rng.integers(0, 2, rows)
    df = pd.DataFrame({
        "Gender": rng.choice(["Male", "Female"], rows),
        "age": age,
        "education": rng.integers(1, 5, rows).astype(float),
        "currentSmoker": smoker,
        "cigsPerDay": smoker * rng.integers(1, 40, rows),
        "BPMeds": rng.choice([0.0, 1.0], rows, p=[0.97, 0.03]),
        "prevalentStroke": rng.choice(["no", "yes"], rows, p=[0.994, 0.006]),
        "prevalentHyp": rng.choice([0, 1], rows, p=[0.69, 0.31]),
        "diabetes": rng.choice([0, 1], rows, p=[0.97, 0.03]),
        "totChol": rng.normal(236, 44, rows).round(),
        "sysBP": rng.normal(132, 22, rows).round(1),
        "diaBP": rng.normal(83, 12, rows).round(1),
        "BMI": rng.normal(25.8, 4, rows).round(2),
        "heartRate": rng.normal(76, 12, rows).round(),
        "glucose": rng.normal(82, 24, rows).round(),
    })
    z = -5.3 + 0.045 * age + 0.012 * (df["sysBP"] - 130) + 0.012 * df["cigsPerDay"] + 0.5 * df["diabetes"]
    p = 1 / (1 + np.exp(-z))
    df["Heart_ stroke"] = np.where(rng.random(rows) < p, "yes", "no")
    # sprinkle missing values like the real dataset
    for col in ["education", "cigsPerDay", "BPMeds", "totChol", "BMI", "glucose"]:
        df.loc[rng.random(rows) < 0.02, col] = np.nan
    return df


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", type=int, default=4000)
    ap.add_argument("--out", type=Path, default=Path(__file__).resolve().parent.parent / "data" / "sample_synthetic.csv")
    a = ap.parse_args()
    a.out.parent.mkdir(parents=True, exist_ok=True)
    make(a.rows).to_csv(a.out, index=False)
    print(f"Wrote {a.rows} synthetic rows to {a.out}")
