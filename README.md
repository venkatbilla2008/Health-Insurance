# 🏆 Health Insurance Claim Prediction

> **Goal:** Predict the probability that a customer will file a health insurance claim (`target = 1`).

This repository documents the iterative development of three model versions for the **Hackathon** health insurance claim prediction challenge. Each version introduced new techniques and improvements, culminating in **v3** — the highest-performing pipeline.

---

## 📁 Repository Structure

| File | Description |
|---|---|
| `train_model.py` | **v1** — Baseline LightGBM + XGBoost ensemble |
| `train_model_v2.py` | **v2** — Improved feature engineering + isotonic calibration |
| `train_model_v3.py` | **v3** ★ — Competition-grade 3-model blend with OOF target encoding |
| `health_insurance_v1.ipynb` | Notebook version of v1 pipeline |
| `health_insurance_v2.ipynb` | Notebook version of v2 pipeline |
| `health_insurance_v3.ipynb` | Notebook version of v3 pipeline (PR-AUC Maximizer) |
| `submission.csv` | v1 submission file |
| `submission_v2.csv` | v2 submission file |
| `submission_v3.csv` | v3 submission file (best) |
| `analyze_submissions.py` | Script to compare v2 vs v3 submission quality |
| `training_data.csv` | Training dataset |
| `test_data_hackathon.csv` | Test dataset (predictions required) |

---

## 📊 Version Comparison Table

| Aspect | **v1** (`train_model.py`) | **v2** (`train_model_v2.py`) | **v3** (`train_model_v3.py`) ★ |
|---|---|---|---|
| **Primary Metric** | ROC-AUC | ROC-AUC | **PR-AUC** (competition metric) |
| **Models Used** | LightGBM + XGBoost | LightGBM + XGBoost | 3× HistGradientBoosting (A, B, C) |
| **Ensemble Method** | 2-model blend + Meta-LR | 2-model optimal blend | **3-model grid-search blend on PR-AUC** |
| **CV Strategy** | 5-fold Stratified K-Fold | 5-fold Stratified K-Fold | 3-fold Stratified K-Fold (speed optimized) |
| **Feature Engineering** | Row-level stats (sum, mean, std, max, min, NaN count, zero count) | v1 features + missing indicators, ratio features, skew, binary sum, row median | v2 features + **interaction terms**, **log transforms**, **squared features**, **NaN pattern aggregation**, **binary mean** |
| **Target Encoding** | ❌ None | ❌ None | ✅ **OOF Target Encoding** (12 features, 20 quantile bins, smoothing=30) |
| **Missing Value Handling** | Median imputation (SimpleImputer) | Median imputation + 4 missing indicator flags | **Native NaN handling** (HistGBT) + **11 missing indicator flags** |
| **Missing Indicator Cols** | 0 | 4 (`feature_8`, `39`, `45`, `38`) | **11** (added `feature_28`, `12`, `31`, `34`, `35`, `15`, `42`) |
| **Calibration** | Meta-LR stacking | **Isotonic Regression** | **Rank-based normalization** |
| **Hyperparameters (LR)** | 0.03 | 0.02 | 0.06 / 0.08 / 0.12 (per model) |
| **Estimators / Iterations** | 2000 | 3000 | 100–150 (early stopping) |
| **Regularization** | L1=0.1, L2=0.1 | L1=0.5, L2=1.0 | L2 = 0.5 / 1.5 / 3.0 (per model) |
| **Class Imbalance** | `scale_pos_weight` | `scale_pos_weight` | **`class_weight` dict** |
| **Ratio Features** | ❌ None | 4 ratio features | **5 ratio features** + 2 interaction products |
| **Log / Polynomial Features** | ❌ None | ❌ None | ✅ `log1p(feature_25)`, `log1p(feature_45)`, `feature_24²` |
| **Dependencies** | LightGBM, XGBoost, sklearn | LightGBM, XGBoost, sklearn | **sklearn only** (no LightGBM/XGBoost needed) |
| **Total Features** | ~57 | ~73 | **~95+** (including target-encoded features) |
| **Submission File** | `submission.csv` | `submission_v2.csv` | `submission_v3.csv` |

---

## 🚀 Why v3 Achieved the Best Results

### 1. 🎯 Optimizing the Right Metric — PR-AUC Instead of ROC-AUC

The single most impactful change. The competition metric was **PR-AUC (Precision-Recall AUC)**, not ROC-AUC. v1 and v2 optimized ROC-AUC throughout — from model training to blending. v3 switched **every stage** to optimize PR-AUC directly:
- Model evaluation uses `average_precision_score`
- Blend weight grid-search maximizes OOF PR-AUC
- Rank normalization preserves PR-AUC ordering

> **Why it matters:** ROC-AUC can be misleadingly high on imbalanced datasets. PR-AUC focuses on how well the model identifies the **positive class (claims)**, which is exactly what the competition rewards.

### 2. 🔄 OOF Target Encoding (Leak-Free)

v3 introduced **Out-of-Fold (OOF) Target Encoding** on 12 key numeric features — a technique that lets the model learn the relationship between feature bins and the target, without data leakage:

- Features are quantile-binned into 20 bins
- For each CV fold, encoding is computed only on the training portion
- Smoothing (α=30) prevents overfitting on rare bins
- Test encoding uses the full training set

**Features encoded:** `feature_24`, `25`, `29`, `38`, `45`, `2`, `7`, `10`, `40`, `47`, `48`, `33`

### 3. 🧩 3-Model Diversity Ensemble

Instead of relying on two different algorithms (LightGBM + XGBoost), v3 uses **three diverse configurations of the same algorithm** (HistGradientBoosting), each with different:

| Model | Learning Rate | Max Leaf Nodes | L2 Regularization | Min Samples/Leaf | Seed |
|---|---|---|---|---|---|
| **A** (main) | 0.08 | 63 | 0.5 | 30 | 42 |
| **B** (diversity) | 0.12 | 31 | 3.0 | 50 | 43 |
| **C** (heavy reg.) | 0.06 | 47 | 1.5 | 40 | 44 |

The optimal blend weights are found via grid-search over OOF PR-AUC (step=0.05), ensuring maximum diversity benefit.

### 4. 📐 Richer Feature Engineering

v3 expanded the feature space significantly:

| Category | v1 | v2 | v3 |
|---|---|---|---|
| Row-level stats | 7 | 9 | 7 |
| Missing indicators | 0 | 4 | **11** |
| Ratio features | 0 | 4 | **5** |
| Interaction products | 0 | 0 | **2** (`fe_25×29`, `fe_7×2`) |
| Log transforms | 0 | 0 | **2** (`log1p(f25)`, `log1p(f45)`) |
| Polynomial features | 0 | 0 | **1** (`f24²`) |
| Binary aggregates | 0 | 1 | **2** (sum + mean) |
| NaN pattern features | 0 | 0 | **1** (`miss_pattern_sum`) |
| Target-encoded features | 0 | 0 | **12** |
| **Total engineered** | **7** | **~18** | **~43** |

### 5. 📊 Rank-Based Normalization

Instead of isotonic regression calibration (which in v2 was found to **crush max probabilities to ~0.33** — destroying PR-AUC), v3 uses rank-based normalization:

- Maps predictions to their quantile positions in the OOF distribution
- Preserves the **relative ordering** of predictions perfectly
- Test predictions are interpolated into the same scale
- Only applied if it improves PR-AUC; otherwise raw blend is used

### 6. ⚡ Native NaN Handling (No Imputation)

v1 and v2 used `SimpleImputer(strategy='median')` which replaces NaN with a single value — losing the **informational signal** that a value is missing. v3's `HistGradientBoostingClassifier` handles NaN **natively**, learning optimal split directions for missing values at each tree node.

### 7. 🪶 Zero External Dependencies

v3 runs purely on **scikit-learn + numpy + pandas** — no LightGBM or XGBoost installation required. This makes it:
- More portable across environments (e.g., Google Colab, Kaggle kernels)
- Faster to set up for reproducibility
- Less prone to version conflicts

---

## 🧬 Evolution Timeline

```
v1 (Baseline)
├── LightGBM + XGBoost, ROC-AUC, basic row stats
├── 2-model blend + Meta-LR stacking
└── Simple median imputation
        │
        ▼
v2 (Improved)
├── + Missing indicator flags (4 features)
├── + Ratio & interaction features
├── + Isotonic Regression calibration
├── + Row skew, median, binary sum
├── + Tuned hyperparameters (3000 trees, lr=0.02)
└── Still optimizing ROC-AUC ⚠️
        │
        ▼
v3 (Competition-Grade) ★
├── ☑ Switched to PR-AUC everywhere
├── ☑ OOF Target Encoding (12 features)
├── ☑ 3× HistGBT diverse ensemble
├── ☑ Expanded to 11 missing indicators
├── ☑ Log, polynomial, interaction features
├── ☑ Rank normalization (not isotonic)
├── ☑ Native NaN handling (no imputation)
└── ☑ Zero external dependencies
```

---

## 🔬 Key Techniques Summary

| Technique | Impact | Used In |
|---|---|---|
| PR-AUC optimization | 🔴 **Critical** — aligns with competition metric | v3 only |
| OOF Target Encoding | 🔴 **High** — captures non-linear feature-target relationships | v3 only |
| 3-model diverse ensemble | 🟠 **High** — reduces variance through diversity | v3 only |
| Missing indicator flags | 🟡 **Medium** — preserves missingness signal | v2, v3 |
| Rank normalization | 🟡 **Medium** — preserves PR-AUC unlike isotonic | v3 only |
| Ratio / interaction features | 🟡 **Medium** — captures feature relationships | v2, v3 |
| Native NaN handling | 🟢 **Moderate** — better than median imputation | v3 only |
| Log / polynomial transforms | 🟢 **Moderate** — handles skewed distributions | v3 only |
| Isotonic calibration | ⚠️ **Harmful** for PR-AUC (crushes probabilities) | v2 only |

---

## 🏃 How to Run

```bash
# v1
python train_model.py

# v2
python train_model_v2.py

# v3 (recommended — best results)
python train_model_v3.py
```

> **Note:** v1 and v2 require `lightgbm` and `xgboost` packages. v3 only requires `scikit-learn`, `numpy`, and `pandas`.

---

## 📝 Competition Details

- **Event:** Hackathon
- **Task:** Binary classification — predict probability of health insurance claim
- **Metric:** PR-AUC (Precision-Recall Area Under Curve)
- **Dataset:** ~500K+ training samples, anonymized features (`feature_0` through `feature_50`)
- **Class Balance:** Imbalanced dataset (low positive rate)

---

*Built iteratively across three versions, each learning from the limitations of the previous one. v3 represents the final, competition-optimized pipeline.*
