"""Heart disease (stroke) risk prediction with classical machine-learning models.

Trains and compares six classifiers (Logistic Regression, Decision Tree,
Random Forest, SVM, Naive Bayes, k-NN) on a tabular health dataset
(Framingham-style columns), reports several metrics and saves plots,
a results table and the best model to ``outputs/``.

Usage
-----
    python src/heart_disease_prediction.py --data data/heart_disease.csv
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")  # headless-safe: figures are saved, not shown
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

RANDOM_STATE = 42
TARGET = "Heart_stroke"
CONTINUOUS_FEATURES = [
    "age", "cigsPerDay", "totChol", "sysBP", "diaBP", "BMI", "heartRate", "glucose",
]
PROJECT_ROOT = Path(__file__).resolve().parent.parent


# --------------------------------------------------------------------------- #
# Data loading & preprocessing
# --------------------------------------------------------------------------- #
def load_data(path: Path) -> pd.DataFrame:
    """Read the CSV and normalise column names (e.g. ``'Heart_ stroke'`` -> ``'Heart_stroke'``)."""
    if not path.exists():
        sys.exit(
            f"Dataset not found: {path}\n"
            "Download the Framingham heart-disease CSV (see data/README.md) or generate a "
            "synthetic demo file with:  python src/make_sample_data.py"
        )
    df = pd.read_csv(path)
    df.columns = [re.sub(r"\s+", "", c) for c in df.columns]
    if TARGET not in df.columns:
        sys.exit(f"Target column '{TARGET}' not found. Columns present: {list(df.columns)}")
    print(f"[INFO] Loaded {df.shape[0]} rows x {df.shape[1]} columns from {path}")
    return df


def encode_categoricals(df: pd.DataFrame) -> pd.DataFrame:
    """Turn text columns (Male/Female, yes/no, ...) into numeric codes."""
    df = df.copy()
    binary_maps = {"yes": 1, "no": 0, "male": 1, "female": 0, "true": 1, "false": 0}
    for col in df.select_dtypes(include=["object", "category", "string"]).columns:
        lowered = df[col].astype(str).str.strip().str.lower()
        if set(lowered.dropna().unique()) <= set(binary_maps):
            df[col] = lowered.map(binary_maps)
        else:
            df[col] = pd.Categorical(df[col]).codes
    return df


def prepare_data(df: pd.DataFrame, test_size: float):
    """Encode, split (stratified) and return X/y train & test sets."""
    df = encode_categoricals(df)
    X = df.drop(columns=[TARGET])
    y = df[TARGET].astype(int)
    print(f"[INFO] Features ({X.shape[1]}): {list(X.columns)}")
    print(f"[INFO] Positive-class rate: {y.mean():.1%}  (class imbalance -> accuracy alone is misleading)")
    return train_test_split(
        X, y, test_size=test_size, random_state=RANDOM_STATE, shuffle=True, stratify=y
    )


# --------------------------------------------------------------------------- #
# Models
# --------------------------------------------------------------------------- #
def build_models() -> dict[str, Pipeline]:
    """Each model sits in a pipeline: median imputation -> min-max scaling -> classifier."""
    classifiers = {
        "Logistic Regression": LogisticRegression(
            C=12, max_iter=1000, class_weight="balanced", random_state=RANDOM_STATE),
        "Decision Tree": DecisionTreeClassifier(
            max_depth=6, class_weight="balanced", random_state=RANDOM_STATE),
        "Random Forest": RandomForestClassifier(
            n_estimators=300, class_weight="balanced", n_jobs=-1, random_state=RANDOM_STATE),
        "SVM": SVC(class_weight="balanced", random_state=RANDOM_STATE),
        "Naive Bayes": GaussianNB(),
        "k-NN": KNeighborsClassifier(n_neighbors=5),
    }
    return {
        name: Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", MinMaxScaler()),
            ("clf", clf),
        ])
        for name, clf in classifiers.items()
    }


def evaluate(model: Pipeline, X_test, y_test) -> dict[str, float]:
    """Accuracy, precision, recall, F1 and ROC-AUC on the held-out test set."""
    pred = model.predict(X_test)
    # ROC-AUC needs a continuous score: probabilities where available (SVC exposes a margin instead)
    score = (model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba")
             else model.decision_function(X_test))
    return {
        "Accuracy": accuracy_score(y_test, pred),
        "Precision": precision_score(y_test, pred, zero_division=0),
        "Recall": recall_score(y_test, pred, zero_division=0),
        "F1": f1_score(y_test, pred, zero_division=0),
        "ROC-AUC": roc_auc_score(y_test, score),
    }


# --------------------------------------------------------------------------- #
# Plots
# --------------------------------------------------------------------------- #
def plot_feature_distributions(df: pd.DataFrame, out_dir: Path) -> None:
    cols = [c for c in CONTINUOUS_FEATURES if c in df.columns]
    fig, axes = plt.subplots(2, 4, figsize=(16, 7))
    for ax, col in zip(axes.ravel(), cols):
        sns.histplot(df[col].dropna(), bins=30, ax=ax, color="steelblue")
        ax.set_title(col)
    for ax in axes.ravel()[len(cols):]:
        ax.axis("off")
    fig.suptitle("Distribution of continuous features", fontweight="bold")
    fig.tight_layout()
    fig.savefig(out_dir / "feature_distributions.png", dpi=130)
    plt.close(fig)


def plot_confusion_matrices(models: dict[str, Pipeline], X_test, y_test, out_dir: Path) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    for ax, (name, model) in zip(axes.ravel(), models.items()):
        ConfusionMatrixDisplay.from_estimator(model, X_test, y_test, ax=ax, colorbar=False)
        ax.set_title(name)
    fig.tight_layout()
    fig.savefig(out_dir / "confusion_matrices.png", dpi=130)
    plt.close(fig)


def plot_comparison(results: pd.DataFrame, out_dir: Path) -> None:
    ax = results.set_index("Model")[["Accuracy", "Recall", "F1", "ROC-AUC"]].plot.bar(
        figsize=(11, 5), rot=20, width=0.8)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Score")
    ax.set_title("Model comparison (test set)")
    ax.legend(loc="lower right", ncol=4)
    plt.tight_layout()
    plt.savefig(out_dir / "model_comparison.png", dpi=130)
    plt.close()


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data", type=Path, default=PROJECT_ROOT / "data" / "heart_disease.csv")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "outputs")
    parser.add_argument("--test-size", type=float, default=0.2)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    df = load_data(args.data)
    plot_feature_distributions(df, args.output_dir)
    X_train, X_test, y_train, y_test = prepare_data(df, args.test_size)

    models, rows = build_models(), []
    for name, model in models.items():
        model.fit(X_train, y_train)
        metrics = evaluate(model, X_test, y_test)
        rows.append({"Model": name, **metrics})
        print(f"[RESULT] {name:<20} " + "  ".join(f"{k}={v:.3f}" for k, v in metrics.items()))

    results = pd.DataFrame(rows).sort_values("ROC-AUC", ascending=False).reset_index(drop=True)
    results.round(4).to_csv(args.output_dir / "results.csv", index=False)
    print("\n" + results.round(3).to_string(index=False))

    plot_confusion_matrices(models, X_test, y_test, args.output_dir)
    plot_comparison(results, args.output_dir)

    best_name = results.loc[0, "Model"]
    joblib.dump(models[best_name], args.output_dir / "best_model.joblib")
    print(f"\n[INFO] Best model by ROC-AUC: {best_name} (saved to {args.output_dir / 'best_model.joblib'})")


if __name__ == "__main__":
    main()
