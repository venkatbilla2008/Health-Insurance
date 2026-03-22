"""
Health Insurance v3 FINAL VERSION — PR-AUC

Note: sklearn/numpy/pandas (compatible with environment).

FIX applied after challenges faced in previous version v2:

  - Replaced slow GradientBoostingClassifier (Model C) with another fast
    HistGradientBoostingClassifier (Model C) — speedup on large data
  - Reduced max_iter (A: 300→150, B: 200→100, C: 150)
  - Replaced scipy.stats.rankdata with numpy equivalent (no scipy dependency)
  - Reduced max_bins to 127 for Models A and B for faster training
  - 3-fold CV retained for reasonable PR-AUC estimation

"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.impute import SimpleImputer
from sklearn.ensemble import HistGradientBoostingClassifier
import time, os, gc

SEED    = 42
N_FOLDS = 3    # 3-fold for speed
np.random.seed(SEED)

TRAIN_PATH  = 'training_data.csv'
TEST_PATH   = 'test_data_hackathon.csv'
OUTPUT_PATH = 'submission_v3.csv'

t_start = time.time()


# STEP 1  LOAD

print("=" * 70)
print("STEP 1  Loading data")
train_df = pd.read_csv(TRAIN_PATH)
test_df  = pd.read_csv(TEST_PATH)
print(f"  Train {train_df.shape}  |  Test {test_df.shape}")

y        = train_df['target'].values
ids_test = test_df['id'].values
pos_rate = y.mean()
scale_pw = (y == 0).sum() / (y == 1).sum()
cw = {0: 1.0, 1: float(scale_pw)}
print(f"  Positive rate: {pos_rate*100:.2f}%  |  scale_pos_weight: {scale_pw:.2f}")

feature_cols = [c for c in train_df.columns if c not in ['id', 'target']]


# STEP 2  FEATURE ENGINEERING 

print("\nSTEP 2  Feature Engineering")

HIGH_MISS_COLS = ['feature_8', 'feature_39', 'feature_45', 'feature_38',
                  'feature_28', 'feature_12', 'feature_31', 'feature_34',
                  'feature_35', 'feature_15', 'feature_42']

BINARY_FEATS = ['feature_4','feature_5','feature_6','feature_8','feature_11',
                'feature_14','feature_15','feature_16','feature_18','feature_19',
                'feature_20','feature_21','feature_22','feature_27','feature_28',
                'feature_30','feature_32','feature_39','feature_41','feature_42',
                'feature_44','feature_46','feature_49']


def engineer(df, feat_cols, bin_feats):
    df = df.copy()

    # Missing indicator flags
    for col in HIGH_MISS_COLS:
        if col in df.columns:
            df[f'{col}_miss'] = df[col].isnull().astype(np.int8)

    feat_data = df[feat_cols]

    # Row statistics (vectorized)
    df['fe_nan_count']  = feat_data.isnull().sum(axis=1).astype(np.float32)
    df['fe_zero_count'] = (feat_data.fillna(0) == 0).sum(axis=1).astype(np.float32)
    df['fe_row_sum']    = feat_data.sum(axis=1).astype(np.float32)
    df['fe_row_mean']   = feat_data.mean(axis=1).astype(np.float32)
    df['fe_row_std']    = feat_data.std(axis=1).astype(np.float32)
    df['fe_row_max']    = feat_data.max(axis=1).astype(np.float32)
    df['fe_row_min']    = feat_data.min(axis=1).astype(np.float32)

    # Binary feature aggregates
    b_present = [c for c in bin_feats if c in df.columns]
    b_data    = df[b_present].fillna(0)
    df['fe_binary_sum']  = b_data.sum(axis=1).astype(np.float32)
    df['fe_binary_mean'] = b_data.mean(axis=1).astype(np.float32)

    # Key ratio and log features
    df['fe_ratio_25_7']  = df['feature_25'] / (df['feature_7']  + 1)
    df['fe_ratio_13_2']  = df['feature_13'] / (df['feature_2']  + 1)
    df['fe_ratio_40_47'] = df['feature_40'] / (df['feature_47'] + 1)
    df['fe_ratio_24_25'] = df['feature_24'] / (df['feature_25'] + 1)
    df['fe_ratio_45_38'] = df['feature_45'].fillna(0) / (df['feature_38'].fillna(0) + 1)
    df['fe_25_x_29']     = df['feature_25'] * df['feature_29']
    df['fe_7_x_2']       = df['feature_7']  * df['feature_2']
    df['fe_24_sq']       = df['feature_24'] ** 2
    df['fe_25_log']      = np.log1p(df['feature_25'].clip(lower=0))
    df['fe_45_log']      = np.log1p(df['feature_45'].fillna(0).clip(lower=0))

    # NaN pattern
    miss_cols = [f'{c}_miss' for c in HIGH_MISS_COLS if f'{c}_miss' in df.columns]
    if miss_cols:
        df['fe_miss_pattern_sum'] = df[miss_cols].sum(axis=1).astype(np.float32)

    return df


train_df = engineer(train_df, feature_cols, BINARY_FEATS)
test_df  = engineer(test_df,  feature_cols, BINARY_FEATS)

all_features = [c for c in train_df.columns if c not in ['id', 'target']]
print(f"  Total features: {len(all_features)}")



# STEP 3  OOF TARGET ENCODING

print("\nSTEP 3  OOF Target Encoding")

TE_COLS = ['feature_24', 'feature_25', 'feature_29', 'feature_38',
           'feature_45', 'feature_2',  'feature_7',  'feature_10',
           'feature_40', 'feature_47', 'feature_48', 'feature_33']
SMOOTHING = 30
N_BINS    = 20


def oof_target_encode(tr_df, te_df, cols, target, n_splits=3, smooth=30, n_bins=20, seed=42):
    tr_enc = tr_df.copy()
    te_enc = te_df.copy()
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    gm  = target.mean()

    for col in cols:
        if col not in tr_df.columns:
            continue
        new_col      = f'te_{col}'
        combined     = pd.concat([tr_df[col], te_df[col]], axis=0)
        bin_edges    = combined.quantile(np.linspace(0, 1, n_bins + 1)).unique()
        bin_edges[0] = -np.inf
        bin_edges[-1]= np.inf
        tr_binned = pd.cut(tr_df[col], bins=bin_edges, labels=False, duplicates='drop')
        te_binned = pd.cut(te_df[col], bins=bin_edges, labels=False, duplicates='drop')

        oof_enc = np.full(len(tr_df), gm, dtype=np.float64)
        for tr_idx, val_idx in skf.split(tr_df, target):
            fd      = pd.DataFrame({'bin': tr_binned.iloc[tr_idx], 'y': target[tr_idx]})
            stats   = fd.groupby('bin')['y'].agg(['sum', 'count'])
            enc_map = (stats['sum'] + smooth * gm) / (stats['count'] + smooth)
            oof_enc[val_idx] = tr_binned.iloc[val_idx].map(enc_map).fillna(gm).values
        tr_enc[new_col] = oof_enc.astype(np.float32)

        full_fd   = pd.DataFrame({'bin': tr_binned, 'y': target})
        stats_all = full_fd.groupby('bin')['y'].agg(['sum', 'count'])
        enc_all   = (stats_all['sum'] + smooth * gm) / (stats_all['count'] + smooth)
        te_enc[new_col] = te_binned.map(enc_all).fillna(gm).values.astype(np.float32)

    return tr_enc, te_enc


train_df, test_df = oof_target_encode(train_df, test_df, TE_COLS, y,
                                      n_splits=3, smooth=SMOOTHING, n_bins=N_BINS, seed=SEED)
all_features = [c for c in train_df.columns if c not in ['id', 'target']]
print(f"  Features after target encoding: {len(all_features)}")



# STEP 4  PREPARE ARRAYS (HistGBT handles NaN natively — no imputation needed)

print("\nSTEP 4  Preparing feature arrays")

X_raw  = train_df[all_features].values.astype(np.float32)
Xt_raw = test_df[all_features].values.astype(np.float32)

print(f"  Train matrix: {X_raw.shape}  |  Test matrix: {Xt_raw.shape}")
print(f"  NaN in train: {np.isnan(X_raw).sum()}")



# STEP 5  CV SETUP

skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)

oof_a = np.zeros(len(y), dtype=np.float64)
oof_b = np.zeros(len(y), dtype=np.float64)
oof_c = np.zeros(len(y), dtype=np.float64)

test_a = np.zeros(Xt_raw.shape[0], dtype=np.float64)
test_b = np.zeros(Xt_raw.shape[0], dtype=np.float64)
test_c = np.zeros(Xt_raw.shape[0], dtype=np.float64)



# STEP 6  MODEL A: HistGBT (main — lower lr, more leaves)

print(f"\nSTEP 6  HistGBT Model A ({N_FOLDS}-fold CV)")

HGBT_A = {
    'loss':               'log_loss',
    'learning_rate':      0.08,
    'max_iter':           150,         # reduced from 300 → faster
    'max_leaf_nodes':     63,
    'max_depth':          None,
    'min_samples_leaf':   30,
    'l2_regularization':  0.5,
    'max_bins':           127,         # reduced from 255 → faster
    'class_weight':       cw,
    'early_stopping':     True,
    'validation_fraction': 0.1,
    'n_iter_no_change':   20,
    'tol':                1e-4,
    'random_state':       SEED,
    'verbose':            0,
}

fold_ap_a = []
for fold, (tr_idx, val_idx) in enumerate(skf.split(X_raw, y)):
    t0 = time.time()
    m  = HistGradientBoostingClassifier(**HGBT_A)
    m.fit(X_raw[tr_idx], y[tr_idx])
    oof_a[val_idx] = m.predict_proba(X_raw[val_idx])[:, 1]
    test_a        += m.predict_proba(Xt_raw)[:, 1] / N_FOLDS
    ap = average_precision_score(y[val_idx], oof_a[val_idx])
    fold_ap_a.append(ap)
    print(f"  Fold {fold+1}/{N_FOLDS}  PR-AUC={ap:.5f}  iter={m.n_iter_}  [{time.time()-t0:.1f}s]")
    del m; gc.collect()

ap_a = average_precision_score(y, oof_a)
print(f"  ★ HistGBT-A OOF PR-AUC: {ap_a:.5f}  ROC-AUC: {roc_auc_score(y, oof_a):.5f}")



# STEP 7  MODEL B: HistGBT (higher lr, fewer leaves — diversity)

print(f"\nSTEP 7  HistGBT Model B ({N_FOLDS}-fold CV)")

HGBT_B = {
    'loss':               'log_loss',
    'learning_rate':      0.12,
    'max_iter':           100,         # reduced from 200 → faster
    'max_leaf_nodes':     31,
    'max_depth':          None,
    'min_samples_leaf':   50,
    'l2_regularization':  3.0,
    'max_bins':           127,
    'class_weight':       cw,
    'early_stopping':     True,
    'validation_fraction': 0.1,
    'n_iter_no_change':   15,
    'tol':                1e-4,
    'random_state':       SEED + 1,
    'verbose':            0,
}

fold_ap_b = []
for fold, (tr_idx, val_idx) in enumerate(skf.split(X_raw, y)):
    t0 = time.time()
    m  = HistGradientBoostingClassifier(**HGBT_B)
    m.fit(X_raw[tr_idx], y[tr_idx])
    oof_b[val_idx] = m.predict_proba(X_raw[val_idx])[:, 1]
    test_b        += m.predict_proba(Xt_raw)[:, 1] / N_FOLDS
    ap = average_precision_score(y[val_idx], oof_b[val_idx])
    fold_ap_b.append(ap)
    print(f"  Fold {fold+1}/{N_FOLDS}  PR-AUC={ap:.5f}  iter={m.n_iter_}  [{time.time()-t0:.1f}s]")
    del m; gc.collect()

ap_b = average_precision_score(y, oof_b)
print(f"  ★ HistGBT-B OOF PR-AUC: {ap_b:.5f}")



# STEP 8 - MODEL C: HistGBT (heavier regularization + different seed)
# REPLACES the slow GradientBoostingClassifier (10-100x speedup)

print(f"\nSTEP 8  HistGBT Model C ({N_FOLDS}-fold CV)")

HGBT_C = {
    'loss':               'log_loss',
    'learning_rate':      0.06,
    'max_iter':           150,
    'max_leaf_nodes':     47,
    'max_depth':          None,
    'min_samples_leaf':   40,
    'l2_regularization':  1.5,
    'max_bins':           127,
    'class_weight':       cw,
    'early_stopping':     True,
    'validation_fraction': 0.1,
    'n_iter_no_change':   20,
    'tol':                1e-4,
    'random_state':       SEED + 2,
    'verbose':            0,
}

fold_ap_c = []
for fold, (tr_idx, val_idx) in enumerate(skf.split(X_raw, y)):
    t0 = time.time()
    m  = HistGradientBoostingClassifier(**HGBT_C)
    m.fit(X_raw[tr_idx], y[tr_idx])
    oof_c[val_idx] = m.predict_proba(X_raw[val_idx])[:, 1]
    test_c        += m.predict_proba(Xt_raw)[:, 1] / N_FOLDS
    ap = average_precision_score(y[val_idx], oof_c[val_idx])
    fold_ap_c.append(ap)
    print(f"  Fold {fold+1}/{N_FOLDS}  PR-AUC={ap:.5f}  iter={m.n_iter_}  [{time.time()-t0:.1f}s]")
    del m; gc.collect()

ap_c = average_precision_score(y, oof_c)
print(f"  ★ HistGBT-C OOF PR-AUC: {ap_c:.5f}")


# STEP 9  OPTIMAL 3-MODEL BLEND (grid search on OOF PR-AUC)

print("\nSTEP 9  Optimal 3-Model Blend (PR-AUC)")

best_blend_ap = 0
best_w        = (0.34, 0.33, 0.33)
step          = 0.05

for wa in np.arange(0, 1 + step, step):
    for wb in np.arange(0, 1 - wa + step, step):
        wc = 1 - wa - wb
        if wc < -1e-9:
            continue
        wc = max(wc, 0.0)
        blend = wa * oof_a + wb * oof_b + wc * oof_c
        ap    = average_precision_score(y, blend)
        if ap > best_blend_ap:
            best_blend_ap = ap
            best_w        = (wa, wb, wc)

wa, wb, wc_w = best_w
print(f"  Best weights → A={wa:.2f}  B={wb:.2f}  C={wc_w:.2f}")
print(f"  Blend OOF PR-AUC: {best_blend_ap:.5f}")

oof_blend  = wa * oof_a  + wb * oof_b  + wc_w * oof_c
test_blend = wa * test_a + wb * test_b + wc_w * test_c


# STEP 10  RANK-BASED NORMALIZATION (numpy-only, no scipy dependency)

print("\nSTEP 10  Rank Normalization")

def rank_normalize_np(train_preds, test_preds):
    """Rank-normalize test preds using OOF distribution (numpy only)."""
    n_train = len(train_preds)
    # argsort twice = rankdata (average method approximation)
    order     = np.argsort(train_preds)
    ranks     = np.empty(n_train, dtype=np.float64)
    ranks[order] = (np.arange(n_train) + 1.0) / (n_train + 1)
    sorted_oof   = np.sort(train_preds)
    sorted_ranks = np.sort(ranks)
    return ranks, np.interp(test_preds, sorted_oof, sorted_ranks)

oof_norm, test_norm = rank_normalize_np(oof_blend, test_blend)

blend_ap = average_precision_score(y, oof_blend)
norm_ap  = average_precision_score(y, oof_norm)
print(f"  Raw blend OOF PR-AUC:         {blend_ap:.5f}")
print(f"  Rank-normalized OOF PR-AUC:   {norm_ap:.5f}")

if norm_ap >= blend_ap:
    final_preds = test_norm
    print("  ✓ Using rank-normalized predictions")
else:
    final_preds = test_blend
    print("  ✓ Using raw blend predictions (rank norm didn't help)")


# STEP 11  SAVE SUBMISSION

print(f"\nSTEP 11  Saving submission → {OUTPUT_PATH}")

_output_dir = os.path.dirname(OUTPUT_PATH)
if _output_dir:
    os.makedirs(_output_dir, exist_ok=True)

submission = pd.DataFrame({'id': ids_test, 'target': final_preds})
submission.to_csv(OUTPUT_PATH, index=False)

print(f"  Shape: {submission.shape}")
print(f"  Probability stats:")
print(f"    Mean: {final_preds.mean():.5f}  (true rate: {y.mean():.5f})")
print(f"    Std : {final_preds.std():.5f}")
print(f"    Min : {final_preds.min():.6f}")
print(f"    Max : {final_preds.max():.5f}")
print(f"    p99 : {np.percentile(final_preds, 99):.5f}")
print(f"    >0.5: {(final_preds>0.5).sum():,}")

print("\n" + "=" * 70)
print("  FINAL RESULTS SUMMARY")
print("=" * 70)
for name, score in [
    ('HistGBT-A (lr=0.08)',     ap_a),
    ('HistGBT-B (lr=0.12)',     ap_b),
    ('HistGBT-C (lr=0.06)',     ap_c),
    ('3-Model Blend',            best_blend_ap),
    ('Final (after rank norm)',  max(norm_ap, blend_ap)),
]:
    print(f"  {name:<28} OOF PR-AUC: {score:.5f}")
print(f"\n  Total time: {(time.time()-t_start)/60:.1f} min")
print("=" * 70)
print("submission_v3.csv SAVED!")
