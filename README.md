# ❤️ Heart Disease Risk Prediction

![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-orange)
![License](https://img.shields.io/badge/license-MIT-lightgrey)

A machine-learning pipeline that compares **six classifiers** — Logistic Regression, Decision Tree, Random Forest, SVM, Naive Bayes and k-NN — for predicting heart disease / stroke risk from patient health records (Framingham-style tabular data: age, blood pressure, cholesterol, BMI, smoking, diabetes, …).

> ⚠️ **Educational project only.** Not a medical device and not intended for clinical decisions.

## What the pipeline does

1. Loads the CSV and normalises column names (`Heart_ stroke` → `Heart_stroke`).
2. Encodes categorical columns (Male/Female, yes/no).
3. Makes a **stratified** 80/20 train/test split (the positive class is a minority).
4. Trains each model inside a pipeline: median imputation → Min-Max scaling → classifier (class weights are balanced where supported).
5. Evaluates with **Accuracy, Precision, Recall, F1 and ROC-AUC** — accuracy alone is misleading on imbalanced medical data.
6. Saves a results table, feature distributions, confusion matrices, a comparison chart and the best model (ranked by ROC-AUC).

## Project structure

```
heart-disease-prediction/
├── data/
│   ├── README.md                 # where to get the dataset
│   └── sample_synthetic.csv      # synthetic demo data (random, NOT real patients)
├── src/
│   ├── heart_disease_prediction.py
│   └── make_sample_data.py       # regenerates the synthetic demo file
├── requirements.txt
└── LICENSE
```

## Getting started

```bash
git clone https://github.com/<your-username>/heart-disease-prediction.git
cd heart-disease-prediction
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

**With the real dataset** — download the Framingham heart-disease CSV (details in [`data/README.md`](data/README.md)), save it as `data/heart_disease.csv`, then:

```bash
python src/heart_disease_prediction.py
```

**Quick demo without downloading anything** (synthetic data, results are meaningless medically):

```bash
python src/heart_disease_prediction.py --data data/sample_synthetic.csv
```

Options: `--data PATH`, `--output-dir DIR`, `--test-size 0.2`.
Outputs (in `outputs/`): `results.csv`, `feature_distributions.png`, `confusion_matrices.png`, `model_comparison.png`, `best_model.joblib`.

## Expected data format

A CSV with these columns (the last one is the target):
`Gender, age, education, currentSmoker, cigsPerDay, BPMeds, prevalentStroke, prevalentHyp, diabetes, totChol, sysBP, diaBP, BMI, heartRate, glucose, Heart_ stroke`

## Notes

- Results depend on the dataset; the earlier hard-coded claim "Logistic Regression is best" was removed — the best model is now selected from the measured metrics.
- Future work: hyper-parameter tuning, cross-validation, SMOTE/threshold tuning for recall, SHAP explanations.

## License

[MIT](LICENSE)
