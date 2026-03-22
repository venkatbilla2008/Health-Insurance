"""
Health Insurance Claim Prediction
==================================
Goal: Predict probability that a customer will file a health insurance claim (target=1)

Pipeline:
- Data loading & EDA summary
- Preprocessing: median imputation for NaN columns
- Feature engineering
- LightGBM + XGBoost ensemble with stratified k-fold CV
- Calibration
- Final submission CSV with predicted probabilities
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
import lightgbm as lgb
import xgboost as xgb
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
import time
import os

SEED = 42
np.random.seed(SEED)

DATA_DIR = r'C:\Users\Admin\.gemini\antigravity\playground\playground\Hackathon'
TRAIN_PATH = os.path.join(DATA_DIR, 'training_data.csv')
TEST_PATH  = os.path.join(DATA_DIR, 'test_data_hackathon.csv')
OUTPUT_PATH = os.path.join(DATA_DIR, 'submission.csv')

# ─── 1. LOAD DATA ───────────────────────────────────────────────────────────
print("=" * 60)
print("STEP 1: Loading data...")
t0 = time.time()
train = pd.read_csv(TRAIN_PATH)
test  = pd.read_csv(TEST_PATH)
print(f"  Train: {train.shape}  |  Test: {test.shape}  [{time.time()-t0:.1f}s]")

# ─── 2. BASIC EDA ──────────────────────────────────────────────────────────
print("\nSTEP 2: EDA Summary")
y = train['target'].values
ids_test = test['id'].values

target_rate = y.mean()
print(f"  Target rate: {target_rate*100:.2f}% positive (class=1)")

feature_cols = [c for c in train.columns if c not in ['id', 'target']]
print(f"  Number of features: {len(feature_cols)}")

missing_train = train[feature_cols].isnull().sum()
missing_train = missing_train[missing_train > 0]
print(f"  Features with missing values (train): {len(missing_train)}")
if len(missing_train):
    for col, cnt in missing_train.items():
        print(f"    {col}: {cnt} missing ({cnt/len(train)*100:.1f}%)")

# ─── 3. FEATURE ENGINEERING ────────────────────────────────────────────────
print("\nSTEP 3: Feature Engineering...")

def add_features(df, feature_cols):
    df = df.copy()
    # Aggregate row-level stats across all features
    feat_data = df[feature_cols]
    df['fe_row_sum']    = feat_data.sum(axis=1)
    df['fe_row_mean']   = feat_data.mean(axis=1)
    df['fe_row_std']    = feat_data.std(axis=1)
    df['fe_row_max']    = feat_data.max(axis=1)
    df['fe_row_min']    = feat_data.min(axis=1)
    df['fe_nan_count']  = feat_data.isnull().sum(axis=1)
    df['fe_zero_count'] = (feat_data == 0).sum(axis=1)
    return df

train = add_features(train, feature_cols)
test  = add_features(test,  feature_cols)

# Updated feature list (include engineered features)
all_feature_cols = feature_cols + ['fe_row_sum', 'fe_row_mean', 'fe_row_std',
                                    'fe_row_max', 'fe_row_min', 'fe_nan_count',
                                    'fe_zero_count']

X_train = train[all_feature_cols].values
X_test  = test[all_feature_cols].values

print(f"  Final feature matrix: Train {X_train.shape} | Test {X_test.shape}")

# ─── 4. IMPUTATION ─────────────────────────────────────────────────────────
print("\nSTEP 4: Imputing missing values (median)...")
imputer = SimpleImputer(strategy='median')
X_train = imputer.fit_transform(X_train)
X_test  = imputer.transform(X_test)

# ─── 5. CROSS-VALIDATION SETUP ─────────────────────────────────────────────
N_FOLDS = 5
skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)

oof_lgb  = np.zeros(len(y))
oof_xgb  = np.zeros(len(y))
test_lgb = np.zeros(len(X_test))
test_xgb = np.zeros(len(X_test))

# ─── 6. LIGHTGBM ─────────────────────────────────────────────────────────
print(f"\nSTEP 5: Training LightGBM ({N_FOLDS}-fold CV)...")
print(f"  Scale pos weight (neg/pos): {(y==0).sum()/(y==1).sum():.2f}")

lgb_params = {
    'objective':        'binary',
    'metric':           'auc',
    'boosting_type':    'gbdt',
    'n_estimators':     2000,
    'learning_rate':    0.03,
    'num_leaves':       127,
    'max_depth':        -1,
    'min_child_samples': 50,
    'feature_fraction': 0.8,
    'bagging_fraction': 0.8,
    'bagging_freq':     5,
    'lambda_l1':        0.1,
    'lambda_l2':        0.1,
    'scale_pos_weight': (y==0).sum() / (y==1).sum(),
    'random_state':     SEED,
    'n_jobs':           -1,
    'verbose':          -1,
}

for fold, (tr_idx, val_idx) in enumerate(skf.split(X_train, y)):
    fold_t = time.time()
    X_tr, X_val = X_train[tr_idx], X_train[val_idx]
    y_tr, y_val = y[tr_idx], y[val_idx]

    model = lgb.LGBMClassifier(**lgb_params)
    model.fit(
        X_tr, y_tr,
        eval_set=[(X_val, y_val)],
        callbacks=[
            lgb.early_stopping(stopping_rounds=100, verbose=False),
            lgb.log_evaluation(period=-1)
        ]
    )

    oof_lgb[val_idx]  = model.predict_proba(X_val)[:, 1]
    test_lgb          += model.predict_proba(X_test)[:, 1] / N_FOLDS

    fold_auc = roc_auc_score(y_val, oof_lgb[val_idx])
    print(f"  Fold {fold+1}/{N_FOLDS}  AUC={fold_auc:.5f}  Best iter={model.best_iteration_}  [{time.time()-fold_t:.1f}s]")

lgb_cv_auc = roc_auc_score(y, oof_lgb)
print(f"  >>> LightGBM OOF AUC: {lgb_cv_auc:.5f}")

# ─── 7. XGBOOST ──────────────────────────────────────────────────────────
print(f"\nSTEP 6: Training XGBoost ({N_FOLDS}-fold CV)...")

xgb_params = {
    'objective':        'binary:logistic',
    'eval_metric':      'auc',
    'n_estimators':     2000,
    'learning_rate':    0.03,
    'max_depth':        6,
    'subsample':        0.8,
    'colsample_bytree': 0.8,
    'min_child_weight': 50,
    'gamma':            0.1,
    'reg_alpha':        0.1,
    'reg_lambda':       1.0,
    'scale_pos_weight': (y==0).sum() / (y==1).sum(),
    'tree_method':      'hist',
    'random_state':     SEED,
    'n_jobs':           -1,
    'verbosity':        0,
    'early_stopping_rounds': 100,
}

for fold, (tr_idx, val_idx) in enumerate(skf.split(X_train, y)):
    fold_t = time.time()
    X_tr, X_val = X_train[tr_idx], X_train[val_idx]
    y_tr, y_val = y[tr_idx], y[val_idx]

    model_xgb = xgb.XGBClassifier(**xgb_params)
    model_xgb.fit(
        X_tr, y_tr,
        eval_set=[(X_val, y_val)],
        verbose=False,
    )

    oof_xgb[val_idx]  = model_xgb.predict_proba(X_val)[:, 1]
    test_xgb          += model_xgb.predict_proba(X_test)[:, 1] / N_FOLDS

    fold_auc = roc_auc_score(y_val, oof_xgb[val_idx])
    print(f"  Fold {fold+1}/{N_FOLDS}  AUC={fold_auc:.5f}  Best iter={model_xgb.best_iteration}  [{time.time()-fold_t:.1f}s]")

xgb_cv_auc = roc_auc_score(y, oof_xgb)
print(f"  >>> XGBoost OOF AUC: {xgb_cv_auc:.5f}")

# ─── 8. STACKING / BLENDING ───────────────────────────────────────────────
print("\nSTEP 7: Stacking (meta-learner Logistic Regression)...")

# Build OOF stacking matrix
oof_stack  = np.column_stack([oof_lgb,  oof_xgb])
test_stack = np.column_stack([test_lgb, test_xgb])

# Optimise blend weights using OOF
best_auc = 0
best_w   = 0.5
for w in np.linspace(0, 1, 101):
    blend_oof = w * oof_lgb + (1 - w) * oof_xgb
    auc = roc_auc_score(y, blend_oof)
    if auc > best_auc:
        best_auc = auc
        best_w   = w

blend_test = best_w * test_lgb + (1 - best_w) * test_xgb
print(f"  Best blend weight (LGB): {best_w:.2f}  |  Blend OOF AUC: {best_auc:.5f}")

# Also try meta LR on OOF preds
scaler = StandardScaler()
oof_scaled  = scaler.fit_transform(oof_stack)
test_scaled = scaler.transform(test_stack)

meta = LogisticRegression(C=1.0, random_state=SEED, max_iter=1000)
meta.fit(oof_scaled, y)
meta_oof  = meta.predict_proba(oof_scaled)[:, 1]
meta_test = meta.predict_proba(test_scaled)[:, 1]
meta_auc  = roc_auc_score(y, meta_oof)
print(f"  Meta LR OOF AUC: {meta_auc:.5f}")

# Choose the best final prediction
if meta_auc >= best_auc:
    final_preds = meta_test
    method = f"Meta-LR  (AUC={meta_auc:.5f})"
else:
    final_preds = blend_test
    method = f"Blend (w_lgb={best_w:.2f}, AUC={best_auc:.5f})"

print(f"\n  >>> FINAL METHOD: {method}")

# ─── 9. SUBMISSION ────────────────────────────────────────────────────────
print(f"\nSTEP 8: Writing submission to:\n  {OUTPUT_PATH}")
submission = pd.DataFrame({'id': ids_test, 'target': final_preds})
submission.to_csv(OUTPUT_PATH, index=False)

print(f"\n  Submission shape: {submission.shape}")
print(f"  Predicted probability stats:")
print(f"    Mean:  {final_preds.mean():.5f}")
print(f"    Std:   {final_preds.std():.5f}")
print(f"    Min:   {final_preds.min():.5f}")
print(f"    Max:   {final_preds.max():.5f}")
print(f"\nSample:\n{submission.head(10).to_string(index=False)}")

print("\n" + "=" * 60)
print("ALL DONE! Submission saved.")
print(f"Total time: {(time.time()-t0)/60:.1f} minutes")
print("=" * 60)
