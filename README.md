# Titanic Survival Prediction

Predicting passenger survival on the Titanic — with a focus on *why* the model thinks what it thinks, not just whether it's right.

[![Python](https://img.shields.io/badge/Python-3.13-blue)]()
[![scikit-learn](https://img.shields.io/badge/scikit--learn-RandomForest-orange)]()
[![SHAP](https://img.shields.io/badge/Explainability-SHAP-green)]()

---

## Why This Project Is Different

Most beginner Titanic notebooks stop at "Sex and Pclass predict survival, here's 80% accuracy." This project goes further:

- **Engineered features that matter**: extracted passenger titles from names, derived cabin decks from mostly-missing data, and built family-size groupings — and then *proved* they mattered using feature importance.
- **Compared 7 model families** with cross-validation before committing to one, instead of jumping straight to a single algorithm.
- **Explained individual predictions with SHAP**, not just global feature importance — so any single passenger's prediction can be traced back to specific, quantified reasons.
- **Reported the model's weaknesses honestly** — including where it struggles (recall on survivors is lower than recall on non-survivors) rather than only showcasing the headline accuracy number.

---

## Results Summary

| Metric | Score |
|---|---|
| Validation Accuracy | 79.3% |
| ROC-AUC | 0.838 |
| Recall (Did Not Survive) | 86% |
| Recall (Survived) | 68% |

The model is notably better at identifying passengers who did **not** survive than those who did — a pattern worth interrogating rather than hiding. See the [Findings](#key-findings) section below.

---

## Project Structure

```
titanic-survival-prediction/
│
├── data/
│   ├── train.csv
│   ├── test.csv
│   └── gender_submission.csv
│
├── notebooks/
│   └── titanic_survival_tess.ipynb
│
├── outputs/
│   ├── 01_survival_overview.png
│   ├── 02_survival_by_sex.png
│   ├── 03_survival_by_class.png
│   ├── 04_survival_by_age.png
│   ├── 05_family_embarkation.png
│   ├── 06_fare_correlation.png
│   ├── 07_model_comparison.png
│   ├── 08_confusion_roc.png
│   ├── 09_feature_importance.png
│   ├── 10_shap_summary.png
│   ├── 11_shap_waterfall.png
│   └── submission.csv
│
└── README.md
```

---

## Methodology

### 1. Exploratory Data Analysis
Investigated survival patterns across sex, class, age, family size, fare, and embarkation port before touching a single model. Confirmed the well-known "women and children first" pattern, but also surfaced less obvious findings — like the survival advantage of small families (2–4 members) over both solo travelers and large families (7+).

### 2. Feature Engineering
Rather than relying on the raw columns, several new features were derived:

- **Title** — extracted from passenger names (Mr, Mrs, Miss, Master, Rare) as a richer proxy for age and social standing than age alone.
- **FamilySize / IsAlone / FamilyGroup** — combined `SibSp` and `Parch` into a single, more predictive signal.
- **Deck** — extracted from the (77% missing) Cabin column. Even the "Unknown" category carried predictive signal, likely correlating with lower-class passengers who had no recorded cabin.
- **AgeGroup / FareBin** — binned continuous variables into life-stage and price-tier categories.

Missing ages were imputed using the **median age per title group** rather than a single global median — a `Master` and a `Mrs` have very different typical ages, and treating them identically would have introduced noise.

### 3. Modelling
Seven model families were compared via 5-fold cross-validation (Logistic Regression, Decision Tree, Random Forest, Gradient Boosting, SVM, KNN, XGBoost) before selecting the strongest performer for hyperparameter tuning via `GridSearchCV`.

### 4. Evaluation
Went beyond accuracy: confusion matrix, ROC-AUC, and per-class recall were used to understand *what kind* of mistakes the model makes — critical in a context where false negatives (predicting death for an actual survivor) and false positives carry different real-world weight.

### 5. Explainability (SHAP)
Used SHAP's TreeExplainer to generate:
- A **global summary plot** showing how each feature pushes predictions toward survival or death across the whole dataset.
- A **waterfall plot** for an individual passenger, showing exactly which features pushed their prediction up or down, by how much.

---

## Key Findings

1. **Sex was the single strongest predictor** (importance ≈ 0.25), consistent with the historical "women and children first" protocol.
2. **Engineered features outperformed raw ones.** `Title` ranked above raw `Pclass` and `FamilySize` in importance — direct evidence that the feature engineering step added real signal, not just complexity.
3. **Deck mattered despite 77% missingness.** Even an "Unknown" cabin category carried predictive weight, likely as an indirect marker of lower-class passengers.
4. **The model under-recalls survivors.** It correctly identifies 86% of passengers who died, but only 68% of those who survived. This asymmetry suggests the model leans on strong "death" signals (male, 3rd class, no cabin) more confidently than it captures the more varied paths to survival.
5. **Binning isn't always better.** `FareBin` underperformed raw `Fare` in feature importance — a useful reminder that simplifying a feature can sometimes cost predictive power, and that choice should be validated, not assumed.

---

## Tech Stack

`Python` · `pandas` · `numpy` · `scikit-learn` · `XGBoost` · `SHAP` · `matplotlib` · `seaborn`

---

## Author

**Tess Kamau**
Data Analyst | Aspiring Data Scientist
[GitHub](https://github.com/Tee-ai63) · [LinkedIn](https://linkedin.com/in/tesskamau) · [Medium](https://medium.com/@wanjirutee4)
