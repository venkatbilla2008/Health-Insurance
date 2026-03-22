"""
Health Insurance Claim Prediction  v2 (Improved)
===================================================
Key improvements over v1:
  1. Missing-value indicator features (creates binary flags for NaN columns)
  2. Interaction features for high-cardinality numeric pairs
  3. Tuned LightGBM hyperparameters (more trees, better regularization)
  4. Two-stage: LightGBM (main) + XGBoost (diversity) + blend optimization
  5. Isotonic Regression calibration of final probabilities
  6. Submission columns: id, target  (probabilities, not classes)
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.impute import SimpleImputer
from sklearn.calibration import calibration_curve
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import lightgbm as lgb
import xgboost as xgb
import time, os

SEED = 42
np.random.seed(SEED)

DATA_DIR    = r'C:\Users\Admin\.gemini\antigravity\playground\playground\Hackathon'
TRAIN_PATH  = os.path.join(DATA_DIR, 'training_data.csv')
TEST_PATH   = os.path.join(DATA_DIR, 'test_data_hackathon.csv')
OUTPUT_PATH = os.path.join(DATA_DIR, 'submission_v2.csv')

#  1. LOAD 
print("=" * 65)
print("STEP 1  Loading data")
t_start = time.time()
train = pd.read_csv(TRAIN_PATH)
test  = pd.read_csv(TEST_PATH)
print(f"  Train {train.shape}  |  Test {test.shape}  [{time.time()-t_start:.1f}s]")

y        = train['target'].values
ids_test = test['id'].values
feature_cols = [c for c in train.columns if c not in ['id', 'target']]

pos_rate    = y.mean()
scale_pw    = (y==0).sum() / (y==1).sum()
print(f"  Positive rate: {pos_rate*100:.2f}%  |  scale_pos_weight: {scale_pw:.2f}")

#  2. FEATURE ENGINEERING 
print("\nSTEP 2  Feature Engineering")

# Columns with significant missingness  create indicator flags
HIGH_MISS_COLS = ['feature_8', 'feature_39', 'feature_45', 'feature_38']

def engineer(df, feature_cols, high_miss_cols):
    df = df.copy()

    # (a) Missing-value indicator flags
    for col in high_miss_cols:
        df[f'{col}_miss'] = df[col].isnull().astype(np.int8)

    # (b) Row-level aggregate statistics (computed BEFORE imputation)
    feat_data = df[feature_cols]
    df['fe_nan_count']  = feat_data.isnull().sum(axis=1).astype(np.float32)
    df['fe_zero_count'] = (feat_data.fillna(0) == 0).sum(axis=1).astype(np.float32)
    df['fe_row_sum']    = feat_data.sum(axis=1).astype(np.float32)
    df['fe_row_mean']   = feat_data.mean(axis=1).astype(np.float32)
    df['fe_row_std']    = feat_data.std(axis=1).astype(np.float32)
    df['fe_row_max']    = feat_data.max(axis=1).astype(np.float32)
    df['fe_row_min']    = feat_data.min(axis=1).astype(np.float32)
    df['fe_row_median'] = feat_data.median(axis=1).astype(np.float32)

    # (c) Skew & kurtosis (useful for anomaly signal)
    df['fe_row_skew']   = feat_data.skew(axis=1).astype(np.float32)

    # (d) Count of binary columns = 1 (many features are 0/1 binary)
    binary_candidates = [c for c in feature_cols
                         if df[c].dropna().isin([0, 1]).all()]
    if binary_candidates:
        df['fe_binary_sum'] = df[binary_candidates].sum(axis=1).astype(np.float32)

    # (e) Ratio features (avoid div-by-zero)
    df['fe_ratio_25_7']  = df['feature_25'] / (df['feature_7']  + 1)
    df['fe_ratio_13_2']  = df['feature_13'] / (df['feature_2']  + 1)
    df['fe_ratio_40_47'] = df['feature_40'] / (df['feature_47'] + 1)
    df['fe_ratio_50_26'] = df['feature_50'] / (df['feature_26'] + 1)

    return df

train = engineer(train, feature_cols, HIGH_MISS_COLS)
test  = engineer(test,  feature_cols, HIGH_MISS_COLS)

# All columns except id and target are features
all_features = [c for c in train.columns if c not in ['id', 'target']]
print(f"  Total features after engineering: {len(all_features)}")

X_full = train[all_features].values.astype(np.float32)
X_test = test[all_features].values.astype(np.float32)

#  3. IMPUTE 
print("\nSTEP 3  Imputing (median strategy)")
imputer = SimpleImputer(strategy='median')
X_full  = imputer.fit_transform(X_full)
X_test  = imputer.transform(X_test)

#  4. CV SETUP 
N_FOLDS  = 5
skf      = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
oof_lgb  = np.zeros(len(y), dtype=np.float64)
oof_xgb  = np.zeros(len(y), dtype=np.float64)
test_lgb = np.zeros(len(X_test), dtype=np.float64)
test_xgb = np.zeros(len(X_test), dtype=np.float64)

#  5. LIGHTGBM 
print(f"\nSTEP 4  LightGBM ({N_FOLDS}-fold CV)")

LGB_PARAMS = {
    'objective':         'binary',
    'metric':            'auc',
    'boosting_type':     'gbdt',
    'n_estimators':      3000,
    'learning_rate':     0.02,
    'num_leaves':        255,
    'max_depth':         -1,
    'min_child_samples': 30,
    'feature_fraction':  0.7,
    'bagging_fraction':  0.7,
    'bagging_freq':      5,
    'lambda_l1':         0.5,
    'lambda_l2':         1.0,
    'min_split_gain':    0.01,
    'scale_pos_weight':  scale_pw,
    'random_state':      SEED,
    'n_jobs':            -1,
    'verbose':           -1,
}

lgb_fold_aucs = []
for fold, (tr_idx, val_idx) in enumerate(skf.split(X_full, y)):
    t0 = time.time()
    X_tr, X_val = X_full[tr_idx], X_full[val_idx]
    y_tr, y_val = y[tr_idx],      y[val_idx]

    m = lgb.LGBMClassifier(**LGB_PARAMS)
    m.fit(
        X_tr, y_tr,
        eval_set=[(X_val, y_val)],
        callbacks=[
            lgb.early_stopping(150, verbose=False),
            lgb.log_evaluation(-1),
        ]
    )
    oof_lgb[val_idx] = m.predict_proba(X_val)[:, 1]
    test_lgb        += m.predict_proba(X_test)[:, 1] / N_FOLDS

    fa = roc_auc_score(y_val, oof_lgb[val_idx])
    lgb_fold_aucs.append(fa)
    print(f"  Fold {fold+1}/{N_FOLDS}  AUC={fa:.5f}  best_iter={m.best_iteration_}  [{time.time()-t0:.1f}s]")

lgb_oof_auc = roc_auc_score(y, oof_lgb)
print(f"   LightGBM OOF AUC: {lgb_oof_auc:.5f}  (std={np.std(lgb_fold_aucs):.5f})")


#  6. XGBOOST 
print(f"\nSTEP 5  XGBoost ({N_FOLDS}-fold CV)")

XGB_PARAMS = {
    'objective':          'binary:logistic',
    'eval_metric':        'auc',
    'n_estimators':       3000,
    'learning_rate':      0.02,
    'max_depth':          7,
    'subsample':          0.7,
    'colsample_bytree':   0.7,
    'min_child_weight':   30,
    'gamma':              0.1,
    'reg_alpha':          0.5,
    'reg_lambda':         1.0,
    'scale_pos_weight':   scale_pw,
    'tree_method':        'hist',
    'random_state':       SEED,
    'n_jobs':             -1,
    'verbosity':          0,
    'early_stopping_rounds': 150,
}

xgb_fold_aucs = []
for fold, (tr_idx, val_idx) in enumerate(skf.split(X_full, y)):
    t0 = time.time()
    X_tr, X_val = X_full[tr_idx], X_full[val_idx]
    y_tr, y_val = y[tr_idx],      y[val_idx]

    m = xgb.XGBClassifier(**XGB_PARAMS)
    m.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)

    oof_xgb[val_idx] = m.predict_proba(X_val)[:, 1]
    test_xgb        += m.predict_proba(X_test)[:, 1] / N_FOLDS

    fa = roc_auc_score(y_val, oof_xgb[val_idx])
    xgb_fold_aucs.append(fa)
    print(f"  Fold {fold+1}/{N_FOLDS}  AUC={fa:.5f}  best_iter={m.best_iteration}  [{time.time()-t0:.1f}s]")

xgb_oof_auc = roc_auc_score(y, oof_xgb)
print(f"   XGBoost OOF AUC: {xgb_oof_auc:.5f}  (std={np.std(xgb_fold_aucs):.5f})")


#  7. OPTIMAL BLEND 
print("\nSTEP 6  Optimising blend weights on OOF")
best_auc = 0
best_w   = 0.5
for w in np.linspace(0, 1, 201):
    a = roc_auc_score(y, w * oof_lgb + (1 - w) * oof_xgb)
    if a > best_auc:
        best_auc = a
        best_w   = w

print(f"  LGB weight={best_w:.3f}  XGB weight={(1-best_w):.3f}  |  Blend OOF AUC: {best_auc:.5f}")

oof_blend  = best_w * oof_lgb  + (1 - best_w) * oof_xgb
test_blend = best_w * test_lgb + (1 - best_w) * test_xgb


#  8. ISOTONIC CALIBRATION 
print("\nSTEP 7  Isotonic Regression Calibration")
# Fit calibrator on OOF predictions
iso = IsotonicRegression(out_of_bounds='clip')
iso.fit(oof_blend, y)
test_calibrated = iso.predict(test_blend)
cal_oof         = iso.predict(oof_blend)
cal_auc         = roc_auc_score(y, cal_oof)
print(f"  Calibrated OOF AUC: {cal_auc:.5f}")
print(f"  Pre-calibration mean: {test_blend.mean():.5f}  |  Post-calibration mean: {test_calibrated.mean():.5f}")
print(f"  True positive rate: {y.mean():.5f}")


#  9. SUBMISSION 
print(f"\nSTEP 8  Saving submission  {OUTPUT_PATH}")
submission = pd.DataFrame({'id': ids_test, 'target': test_calibrated})
submission.to_csv(OUTPUT_PATH, index=False)

print(f"  Shape: {submission.shape}")
print(f"  Probability stats: mean={test_calibrated.mean():.5f}  std={test_calibrated.std():.5f}"
      f"  min={test_calibrated.min():.5f}  max={test_calibrated.max():.5f}")

print(f"\n  Sample predictions:")
print(submission.head(10).to_string(index=False))

print("\n" + "=" * 65)
print(f"  RESULTS SUMMARY")
print(f"  LightGBM OOF AUC : {lgb_oof_auc:.5f}")
print(f"  XGBoost  OOF AUC : {xgb_oof_auc:.5f}")
print(f"  Blend    OOF AUC : {best_auc:.5f}   (LGB w={best_w:.2f})")
print(f"  Calibrated AUC   : {cal_auc:.5f}")
print(f"  Total time       : {(time.time()-t_start)/60:.1f} min")
print("=" * 65)
print("  submission_v2.csv saved successfully!")
