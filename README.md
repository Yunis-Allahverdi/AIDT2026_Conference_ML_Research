-# AIDT2026 — Fairness in ML Classifiers for Employment Prediction

Experimental codebase accompanying our AIDT 2026 conference paper on **gender bias and bias-mitigation techniques** in machine learning classifiers, evaluated on an employment-prediction task derived from Stack Overflow developer survey data.

This repository contains the full set of experiments: baseline models, pre-processing / in-processing / post-processing mitigation methods, and an additional sampling-ratio sensitivity study.

---

## Overview

We train six classifiers — **Logistic Regression, Random Forest, XGBoost, Linear SVM, KNN, and Gaussian Naive Bayes** — to predict employment status, then measure how fair their predictions are across gender groups (Female = unprivileged, Male = privileged).

For each model and each mitigation strategy, we compute four standard group-fairness metrics, plus a composite score we define as the **Fairness Impact Score (FIS)**:

| Metric | Formula |
|---|---|
| **Statistical Parity Difference (SPD)** | P(Ŷ=1 \| unpriv) − P(Ŷ=1 \| priv) |
| **Disparate Impact (DI)** | P(Ŷ=1 \| unpriv) / P(Ŷ=1 \| priv) |
| **Equal Opportunity (EOD)** | TPR_Female − TPR_Male |
| **Equalized Odds (AAOD)** | ½ · [(TPR_F − TPR_M) + (FPR_F − FPR_M)] |
| **FIS (ours)** | ½ · mean(\|SPD\|, \|1−DI\|) + ½ · mean(\|EOD\|, \|AAOD\|) |

Lower FIS = fairer model.

---

## Repository Structure

```
.
├── fairness_AI_final_project.ipynb       # Main experiment: all mitigation methods
├── fairness_down_oversampling_test.ipynb # Sensitivity study: gender ratio in training data
├── processed_data.csv                    # Cleaned dataset (must be present in repo root)
└── README.md
```

### `fairness_AI_final_project.ipynb`
The primary experiment notebook. It runs the six classifiers under five conditions:

1. **Baseline** — original data, no mitigation
2. **Pre-processing: Fairness Through Blindness** — drop the `Gender` attribute before training
3. **Pre-processing: Reweighing** — balanced sample weights computed over (gender × label) groups
4. **In-processing: Exponentiated Gradient** — `fairlearn`'s reduction approach with a Demographic Parity constraint
5. **Post-processing: Threshold Optimizer** — `fairlearn`'s post-hoc threshold adjustment

Each condition reports per-group confusion matrices, the four fairness metrics, and the composite FIS, with bar-chart visualizations comparing all six models.

### `fairness_down_oversampling_test.ipynb`
A supporting experiment that asks: **how does the gender ratio in the training set affect model fairness?** We construct five resampled datasets — 30/70, 40/60, 50/50, 60/40, and 70/30 (Female/Male) — and re-run the full fairness pipeline on each, holding the total dataset size constant.

---

## Requirements

```bash
pip install pandas numpy scikit-learn xgboost fairlearn seaborn matplotlib
```

Tested with Python 3.10+.

---

## How to Reproduce

1. Place `processed_data.csv` in the repository root.
2. Open either notebook in Jupyter / VS Code / Colab.
3. Run all cells top-to-bottom — each section is self-contained and prints its own metrics + chart.

The dataset is expected to contain (at minimum) the columns: `Gender` (0 = Female, 1 = Male), `Employed` (target), and the numeric feature columns used for training. String columns `EdLevel`, `Country`, and `Skills` are dropped at the start of each notebook.

---

## License

MIT
