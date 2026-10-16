# Heart Disease Prediction

End-to-end **binary classification** project: exploratory analysis of a clinical dataset and a **logistic regression** model that predicts whether a patient has heart disease.

> **Disclaimer:** educational project only, not medical advice. Always consult a qualified health professional.

## Problem

Given 13 clinical attributes of a patient (age, sex, chest-pain type, resting blood pressure, cholesterol, max heart rate, ST depression, etc.), predict the presence of heart disease (`target`: 1 = diagnosed, 0 = not diagnosed).

## What the script does

1. **Data quality checks**: schema, null values and descriptive statistics.
2. **EDA**: class balance (~54% positive / ~46% negative), sex distribution, correlation heatmap.
3. **Modelling**: stratified 75/25 train/test split, `LogisticRegression(max_iter=1000)`.
4. **Evaluation**: accuracy on training and test sets.
5. **Inference**: predicts the class of an example patient.

## Results

Running `heart_disease_prediction.py` as-is (stratified split, `random_state=1`) prints:

| Split | Accuracy |
|-------|----------|
| Train | 0.863 |
| Test  | 0.842 |

The dataset is small (303 rows), so these numbers vary with the split; they are a baseline, not a clinical-grade result.

## Tech stack

Python · pandas · NumPy · scikit-learn · seaborn · matplotlib

## Getting started

```bash
git clone https://github.com/raimondoDiSangro/heart-disease-prediction.git
cd heart-disease-prediction
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python heart_disease_prediction.py
```

Plots open in interactive windows. Close each one to continue.

## Dataset

`data/heart.csv` contains 303 patients and 14 columns (UCI Cleveland heart disease data, Kaggle version).

| Feature | Description |
|---|---|
| `age` | age in years |
| `sex` | 1 = male, 0 = female |
| `cp` | chest pain type (0-3) |
| `trestbps` | resting blood pressure (mm Hg) |
| `chol` | serum cholesterol (mg/dl) |
| `fbs` | fasting blood sugar > 120 mg/dl (1 = true) |
| `restecg` | resting ECG results |
| `thalach` | maximum heart rate achieved |
| `exang` | exercise-induced angina (1 = yes) |
| `oldpeak` | ST depression induced by exercise relative to rest |
| `slope` | slope of the peak exercise ST segment |
| `ca` | number of major vessels (0-3) coloured by fluoroscopy |
| `thal` | thalassemia |
| `target` | 1 = heart disease, 0 = no heart disease |

## Project structure

```
heart-disease-prediction/
├── heart_disease_prediction.py   # EDA + training + evaluation + example prediction
├── data/heart.csv
├── requirements.txt
└── LICENSE
```

## Possible next steps

Feature scaling and pipelines, cross-validation, comparison with tree-based models, and precision/recall/ROC-AUC reporting (more informative than accuracy for medical screening).

## License

[MIT](LICENSE)
