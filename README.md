# 🏆 Health Insurance Claim Prediction — Hackathon

> **Goal:** Predict the probability that a customer will file a health insurance claim (`target = 1`).

This repository documents the iterative development of four model versions for the **Data & AI Day 2026 Hackathon** health insurance claim prediction challenge. Each version introduced new techniques and improvements, culminating in **v4** — the highest-performing pipeline.

---

## 📁 Repository Structure

| File | Description |
|---|---|
| `train_model.py` | **v1** — Baseline LightGBM + XGBoost ensemble |
| `train_model_v2.py` | **v2** — Improved feature engineering + isotonic calibration |
| `train_model_v3.py` | **v3** — Competition-grade 4-model blend with OOF target encoding |
| `train_model_v4.py` | **v4** ★ — v3 + SMOTE oversampling + feature_12 interactions |
| `train_model_v4_zerve.py` | **v4 Zerve** ★ — Checkpoint-resume edition for 700s timeout environments |
| `health_insurance_v1.ipynb` | Notebook version of v1 pipeline |
| `health_insurance_v2.ipynb` | Notebook version of v2 pipeline |
| `health_insurance_v3.ipynb` | Notebook version of v3 pipeline (PR-AUC Maximizer) |
| `health_insurance_v4_SMOTE.ipynb` | Notebook version of v4 pipeline (SMOTE edition) |
| `submission.csv` | v1 submission file |
| `submission_v2.csv` | v2 submission file |
| `submission_v3.csv` | v3 submission file |
| `submission_v4.csv` | v4 submission file (best) |
| `training_data.csv` | Training dataset |
| `test_data_hackathon.csv` | Test dataset (predictions required) |

---

## 📊 Version Comparison Table

| Aspect | **v1** | **v2** | **v3** | **v4** ★ |
|---|---|---|---|---|
| **Primary Metric** | ROC-AUC | ROC-AUC | PR-AUC | **PR-AUC** |
| **Models Used** | LGB + XGB | LGB + XGB | LGB GBDT + LGB DART + XGB + CatBoost | **LGB GBDT + LGB DART + XGB + CatBoost** |
| **Ensemble Method** | 2-model blend + Meta-LR | 2-model optimal blend | 4-model grid-search blend on PR-AUC | **4-model grid-search blend on PR-AUC** |
| **CV Strategy** | 5-fold | 5-fold | 10-fold | **10-fold** |
| **Imbalance Handling** | `scale_pos_weight` | `scale_pos_weight` | `scale_pos_weight` | **SMOTE (10%) + `scale_pos_weight`** |
| **SMOTE** | ❌ | ❌ | ❌ | ✅ **Inside every CV fold (train only)** |
| **OOF Target Encoding** | ❌ | ❌ | ✅ 14 features | ✅ **14 features** |
| **Feature Engineering** | Row stats (7) | v1 + indicators + ratios (~18) | v2 + log/poly/interactions (~95+) | **v3 + feature_12 interactions (6 new)** |
| **feature_12 interactions** | ❌ | ❌ | ❌ | ✅ **×24, ×16, ×22, ×31, ÷31, ²** |
| **Optuna Tuning** | ❌ | ❌ | ✅ 60 trials | ✅ **80 trials (SMOTE in inner CV)** |
| **Calibration** | Meta-LR stacking | Isotonic Regression ⚠️ | Rank normalisation | **Rank normalisation** |
| **Pseudo-labeling** | ❌ | ❌ | ✅ | ✅ |
| **Missing Indicators** | 0 | 4 | 11 | **11** |
| **Total Features** | ~57 | ~73 | ~115 | **~121** |
| **Dependencies** | LGB, XGB, sklearn | LGB, XGB, sklearn | LGB, XGB, CatBoost, sklearn | **LGB, XGB, CatBoost, imbalanced-learn, sklearn** |
| **Submission File** | `submission.csv` | `submission_v2.csv` | `submission_v3.csv` | `submission_v4.csv` |

---

## 🚀 Why v4 Achieved the Best Results

### 1. 🧬 SMOTE — The Missing Piece for PR-AUC

The single most impactful addition in v4. The training set has only **3.64% positive rate** — 1 claim per 27 non-claims. v3 addressed this at the loss level (`scale_pos_weight = 26.44`), but the models still only *saw* 1 positive example per 26 negatives in every batch.

v4 adds **SMOTE (Synthetic Minority Oversampling Technique)** inside every CV fold:

```python
# Applied to training fold only — validation ALWAYS stays clean
X_tr_sm, y_tr_sm = smote.fit_resample(X_full[tr_idx], y[tr_idx])
m.fit(X_tr_sm, y_tr_sm,
      eval_set=[(X_full[val_idx], y[val_idx])],  # ← untouched
      ...)
```

| | Before SMOTE | After SMOTE (10%) |
|---|---|---|
| Positives per fold | ~15,600 | ~49,200 |
| Minority ratio | 3.64% | 10% |
| Ratio | 1 : 26 | 1 : 9 |

> **Why this fixes the PR curve cliff-drop:** v3 validation showed precision collapsing from 0.20 to 0.04 by recall=0.10, then flatlining at the random baseline. This is the classic symptom of insufficient minority class exposure — the model hasn't seen enough positive examples to learn the decision boundary well. SMOTE creates synthetic positive rows by interpolating between existing positive examples in feature space, directly addressing this.

**SMOTE is applied at 5 levels:**
- Inside Optuna's 3-fold inner CV (hyperparams tuned on balanced data)
- Inside LGB GBDT 10-fold CV
- Inside LGB DART 10-fold CV
- Inside XGBoost 10-fold CV
- Inside CatBoost 10-fold CV (using imputed matrix for SMOTE, NaN-native for validation)
- Inside pseudo-labeling re-train loop

### 2. 🔗 feature_12 Interaction Features

Permutation importance analysis from v3 revealed `feature_12` was the **#1 most important feature by a factor of 1.7×** over the next feature (`feature_24`). v4 explicitly engineers 6 cross-products to expose the interaction signal:

| New Feature | Formula | Rationale |
|---|---|---|
| `fe_12_x_24` | `feature_12 × feature_24` | Top-2 features crossed |
| `fe_12_x_16` | `feature_12 × feature_16` | Top-2 binary × continuous |
| `fe_12_x_22` | `feature_12 × feature_22` | Skewed binary interaction |
| `fe_12_x_31` | `feature_12 × feature_31` | Ordinal pair interaction |
| `fe_12_div_31` | `feature_12 ÷ (feature_31 + 1)` | Ratio of dominant features |
| `fe_12_sq` | `feature_12²` | Non-linear self-interaction |

### 3. 🔍 Optuna Tuned on SMOTE-Balanced Data (80 trials)

v3's Optuna tuned hyperparameters on the raw imbalanced distribution. v4 applies SMOTE inside the Optuna inner CV too, so the tuned hyperparameters (`num_leaves`, `min_child_samples`, `learning_rate`, etc.) are optimised for the same balanced training distribution used at full training time. Trial count also increased from 60 → 80.

---

## 🔬 Imbalance Handling — Full Audit

The 3.64% positive rate is the central challenge of this dataset. v4 addresses it at every level:

| Level | Technique | Where | Effect |
|---|---|---|---|
| **Data level** | SMOTE (10% ratio) | Inside every CV fold | 3× more positive examples seen per fold |
| **Loss level** | `scale_pos_weight = 26.44` | All 4 models | Each positive penalised 26× in gradient |
| **Metric level** | PR-AUC as eval metric | All 4 models + Optuna | Optimises minority class recall directly |
| **Split level** | `StratifiedKFold` | All CV splits | Every fold preserves 3.64% ratio |
| **Feature level** | OOF target encoding | Feature engineering | Encodes claim rate signal per feature bin |
| **Post-training** | Conservative pseudo-labels | After blending | High-confidence test samples added back |

> **Key insight:** `scale_pos_weight` adjusts the gradient — the model still *sees* 1:26 in every batch. SMOTE changes the training data itself so the model sees 1:9. They address different aspects of imbalance and are **complementary**, not redundant.

---

## 🏗️ v4 Zerve Edition — Checkpoint-Resume Architecture

For environments with execution time limits (e.g. workspaces with a 700s timeout), `train_model_v4_zerve.py` implements a checkpoint-per-model system:

- Each model saves OOF arrays + test predictions to `.npz` files immediately after training
- Feature engineering result cached to `features.npz`
- On timeout or restart, completed models are loaded from disk — no re-training
- Each section runs independently and safely under 700s

| Section | Content | Est. time |
|---|---|---|
| Section 1 | Feature eng + encoding + imputation → `features.npz` | ~90s |
| Section 2 | LGB GBDT 5-fold + SMOTE → `lgb.npz` | ~250s |
| Section 3 | XGBoost 5-fold + SMOTE → `xgb.npz` | ~280s |
| Section 4 | CatBoost 5-fold + SMOTE → `cat.npz` | ~300s |
| Section 5 | 3-model blend + rank norm + submission | ~10s |

**Differences from full v4 (to fit time budget):**
- N_FOLDS: 10 → 5
- Optuna: removed (best params baked in)
- LGB DART: removed
- Pseudo-labeling: removed
- n_estimators: 5000 → 2500, early stopping: 200 → 100
- Expected quality cost: ~0.003–0.005 PR-AUC vs full v4

---

## 🧬 Evolution Timeline

```
v1 (Baseline)
├── LightGBM + XGBoost, ROC-AUC, basic row stats
├── 2-model blend + Meta-LR stacking
└── Simple median imputation, scale_pos_weight only
        │
        ▼
v2 (Improved)
├── + Missing indicator flags (4 features)
├── + Ratio & interaction features
├── + Isotonic Regression calibration ⚠️
├── + Row skew, median, binary sum
├── + Tuned hyperparameters (3000 trees, lr=0.02)
└── Still optimizing ROC-AUC ⚠️
        │
        ▼
v3 (PR-AUC Maximizer)
├── ☑ Switched to PR-AUC everywhere (all 4 models + Optuna + blend)
├── ☑ OOF Target Encoding (14 features, Bayesian smoothing)
├── ☑ 4-model ensemble: LGB GBDT + LGB DART + XGBoost + CatBoost
├── ☑ Expanded to 11 missing indicators
├── ☑ Log, polynomial, interaction features (~115 total)
├── ☑ Rank normalisation (replaced harmful isotonic calibration)
├── ☑ Optuna 60-trial hyperparameter search
├── ☑ Conservative pseudo-labeling (auto-reverts if no gain)
└── ☑ 10-fold stratified CV
        │
        ▼
v4 (SMOTE + PR-AUC) ★  ← BEST
├── ☑ All v3 improvements carried forward
├── ☑ SMOTE (10% ratio) inside EVERY CV fold — train only, val clean
├── ☑ SMOTE inside Optuna inner CV (hyperparams tuned on balanced data)
├── ☑ feature_12 interaction features (×6 new, was #1 importance)
├── ☑ Optuna 80 trials (up from 60)
├── ☑ min_child_weight lowered for XGBoost (SMOTE data is denser)
├── ☑ Pseudo-labeling also uses SMOTE in re-train loop
└── ☑ Zerve checkpoint edition for timeout-constrained environments
```

---

## 🔬 Key Techniques Summary

| Technique | Impact | Used In |
|---|---|---|
| PR-AUC optimization | 🔴 **Critical** — aligns with competition metric | v3, v4 |
| SMOTE oversampling | 🔴 **Critical** — fixes PR curve cliff-drop | **v4 only** |
| OOF Target Encoding | 🔴 **High** — captures non-linear feature-target relationships | v3, v4 |
| feature_12 interactions | 🟠 **High** — exploits #1 importance feature | **v4 only** |
| 4-model diverse ensemble | 🟠 **High** — reduces variance through algorithm diversity | v3, v4 |
| Missing indicator flags | 🟡 **Medium** — preserves missingness signal | v2, v3, v4 |
| Rank normalisation | 🟡 **Medium** — preserves PR-AUC unlike isotonic | v3, v4 |
| Optuna hyperparameter search | 🟡 **Medium** — Bayesian tuning on correct metric | v3, v4 |
| Ratio / interaction features | 🟡 **Medium** — captures feature relationships | v2, v3, v4 |
| Pseudo-labeling | 🟢 **Moderate** — semi-supervised gain, auto-reverts | v3, v4 |
| Log / polynomial transforms | 🟢 **Moderate** — handles skewed distributions | v3, v4 |
| Isotonic calibration | ⚠️ **Harmful** for PR-AUC — crushed max prob to 0.33 | v2 only |

---

## 📈 Submission Comparison

| Version | Key Addition | Score Shape | Notes |
|---|---|---|---|
| v1 | Baseline | Mean=0.45, Max=0.89 | ROC-AUC optimised, uncalibrated |
| v2 | Isotonic calibration | Mean=0.036, Max=0.33 | Calibration crushed probability range → worst PR-AUC |
| v3 | PR-AUC + rank norm | Mean=0.504, Max=1.00 | Full score range restored, rank-normalised |
| v4 ★ | + SMOTE + feature_12 | Mean=0.505, Max=1.00 | Same range, better minority boundary learning |

> **Note on mean probability:** Both v3 and v4 show mean ≈ 0.50 rather than the true positive rate of 3.64%. This is expected — rank normalisation maps scores to their percentile positions, so the median score becomes 0.50 by construction. This does **not** affect PR-AUC, which is a purely rank-based metric.

---

## 🏃 How to Run

```bash
# v1
python train_model.py

# v2
python train_model_v2.py

# v3
python train_model_v3.py

# v4 (recommended — best results, requires GPU for full speed)
python train_model_v4.py

# v4 Zerve edition (for timeout-constrained environments)
# Run each section separately — each is under 700s
python train_model_v4_zerve.py
```

**Install dependencies:**

```bash
# v1, v2
pip install lightgbm xgboost scikit-learn pandas numpy

# v3
pip install lightgbm xgboost catboost optuna scikit-learn pandas numpy

# v4 (full)
pip install lightgbm xgboost catboost optuna imbalanced-learn scikit-learn pandas numpy

# v4 Zerve edition
pip install lightgbm xgboost catboost imbalanced-learn scikit-learn pandas numpy
```

> **Note:** v4 uses `device='cuda'` for XGBoost and `task_type='GPU'` for CatBoost. Change these to `'cpu'` / `'CPU'` if running without a GPU. The Zerve edition already defaults to CPU.

---

## 📝 Competition Details

- **Event:** Data & AI Day 2026 Hackathon
- **Task:** Binary classification — predict probability of health insurance claim
- **Metric:** PR-AUC (Precision-Recall Area Under Curve)
- **Dataset:** ~476K training rows, ~119K test rows, 50 anonymised numeric features
- **Class Balance:** 3.64% positive rate (1 claim per 27 non-claims)
- **scale_pos_weight:** 26.44

---

*Built iteratively across four versions, each learning from the limitations of the previous one. v4 represents the final, competition-optimized pipeline.*
