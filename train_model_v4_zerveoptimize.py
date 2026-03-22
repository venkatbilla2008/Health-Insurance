"""
Health Insurance Claim Prediction — v4 ZERVE EDITION
=====================================================
Optimised for Zerve AI workspace (700s timeout per execution).

CHECKPOINT-RESUME ARCHITECTURE:
  Each model saves its OOF + test predictions to disk immediately.
  On timeout/restart, completed models are loaded — no re-training.
  Run each section independently; re-run only the one that timed out.

CHANGES vs full v4:
  - N_FOLDS  : 10 → 5     (halves training time)
  - Optuna   : removed    (best params baked in from prior v4 run)
  - LGB DART : removed    (saves ~20% runtime, tiny quality loss)
  - Pseudo   : removed    (saves another full LGB pass)
  - n_estimators : 5000 → 2500
  - early_stopping : 200 → 100

QUALITY vs full v4:
  Expected PR-AUC loss: ~0.003-0.005  (still >> v3 which had no SMOTE)

ESTIMATED RUNTIME PER SECTION (Zerve CPU):
  Section 1 - Setup + features + encoding : ~90s
  Section 2 - LGB GBDT  (5-fold + SMOTE)  : ~250s  <-- run separately
  Section 3 - XGBoost   (5-fold + SMOTE)  : ~280s  <-- run separately
  Section 4 - CatBoost  (5-fold + SMOTE)  : ~300s  <-- run separately
  Section 5 - Blend + save                : ~10s
  Each section is well under 700s.
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.impute import SimpleImputer
from imblearn.over_sampling import SMOTE
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostClassifier
from scipy.stats import rankdata
import time, os, gc

SEED        = 42
N_FOLDS     = 5
SMOTE_RATIO = 0.10
np.random.seed(SEED)

# ── UPDATE THESE PATHS ──────────────────────────────────────────────────────
DATA_DIR       = '/path/to/your/data'
TRAIN_PATH     = os.path.join(DATA_DIR, 'training_data.csv')
TEST_PATH      = os.path.join(DATA_DIR, 'test_data_hackathon.csv')
OUTPUT_PATH    = os.path.join(DATA_DIR, 'submission_v4_zerve.csv')
CHECKPOINT_DIR = os.path.join(DATA_DIR, 'checkpoints')
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

t_start = time.time()


# ═══════════════════════════════════════════════════════════════════════════
# CHECKPOINT HELPERS
# ═══════════════════════════════════════════════════════════════════════════
def save_ckpt(name, oof, test_p, ap):
    path = os.path.join(CHECKPOINT_DIR, f'{name}.npz')
    np.savez_compressed(path, oof=oof, test_p=test_p, ap=np.array([ap]))
    print(f"  [SAVED] {name}.npz  PR-AUC={ap:.5f}")


def load_ckpt(name):
    path = os.path.join(CHECKPOINT_DIR, f'{name}.npz')
    if os.path.exists(path):
        d  = np.load(path)
        ap = float(d['ap'][0])
        print(f"  [LOADED] {name}.npz  PR-AUC={ap:.5f}  (skipping re-train)")
        return d['oof'], d['test_p'], ap
    return None, None, None


def T():
    return f"[{time.time()-t_start:.0f}s]"


# ═══════════════════════════════════════════════════════════════════════════
# SECTION 1  SETUP — run once, reuse across restarts via features.npz
# ═══════════════════════════════════════════════════════════════════════════
FEAT_CACHE = os.path.join(CHECKPOINT_DIR, 'features.npz')

if os.path.exists(FEAT_CACHE):
    print(f"[LOADED] features.npz — skipping feature engineering")
    d        = np.load(FEAT_CACHE, allow_pickle=True)
    X_full   = d['X_full']
    X_test   = d['X_test']
    X_full_cat = d['X_full_cat']
    X_test_cat = d['X_test_cat']
    y        = d['y']
    ids_test = d['ids_test']
    scale_pw = float(d['scale_pw'])
    pos_rate = float(d['pos_rate'])
    n_feats  = X_full.shape[1]
    print(f"  X_full {X_full.shape}  X_test {X_test.shape}")

else:
    print(f"SECTION 1  Feature Engineering  {T()}")

    train = pd.read_csv(TRAIN_PATH)
    test  = pd.read_csv(TEST_PATH)
    print(f"  Train {train.shape}  Test {test.shape}")

    y        = train['target'].values
    ids_test = test['id'].values
    scale_pw = (y == 0).sum() / (y == 1).sum()
    pos_rate = y.mean()
    print(f"  Positive rate: {pos_rate*100:.2f}%  scale_pos_weight: {scale_pw:.2f}")

    feature_cols = [c for c in train.columns if c not in ['id', 'target']]

    HIGH_MISS = ['feature_8','feature_39','feature_45','feature_38',
                 'feature_28','feature_12','feature_31','feature_34',
                 'feature_35','feature_15','feature_42']
    BINARY    = ['feature_4','feature_5','feature_6','feature_8','feature_11',
                 'feature_14','feature_15','feature_16','feature_18','feature_19',
                 'feature_20','feature_21','feature_22','feature_27','feature_28',
                 'feature_30','feature_32','feature_39','feature_41','feature_42',
                 'feature_44','feature_46','feature_49']

    def engineer(df, fcols, bcols):
        df = df.copy()
        for col in HIGH_MISS:
            if col in df.columns:
                df[f'{col}_miss'] = df[col].isnull().astype(np.int8)
        ft = df[fcols]
        df['fe_nan_count']  = ft.isnull().sum(axis=1).astype(np.float32)
        df['fe_zero_count'] = (ft.fillna(0)==0).sum(axis=1).astype(np.float32)
        df['fe_row_sum']    = ft.sum(axis=1).astype(np.float32)
        df['fe_row_mean']   = ft.mean(axis=1).astype(np.float32)
        df['fe_row_std']    = ft.std(axis=1).astype(np.float32)
        df['fe_row_max']    = ft.max(axis=1).astype(np.float32)
        df['fe_row_min']    = ft.min(axis=1).astype(np.float32)
        df['fe_row_median'] = ft.median(axis=1).astype(np.float32)
        df['fe_row_skew']   = ft.skew(axis=1).astype(np.float32)
        df['fe_row_kurt']   = ft.kurt(axis=1).astype(np.float32)
        b  = [c for c in bcols if c in df.columns]
        bd = df[b].fillna(0)
        df['fe_binary_sum']  = bd.sum(axis=1).astype(np.float32)
        df['fe_binary_mean'] = bd.mean(axis=1).astype(np.float32)
        for i, c1 in enumerate(b[:8]):
            for c2 in b[i+1:9]:
                df[f'fe_{c1}_{c2}'] = (df[c1].fillna(0)*df[c2].fillna(0)).astype(np.int8)
        df['fe_ratio_25_7']  = df['feature_25']/(df['feature_7']+1)
        df['fe_ratio_13_2']  = df['feature_13']/(df['feature_2']+1)
        df['fe_ratio_40_47'] = df['feature_40']/(df['feature_47']+1)
        df['fe_ratio_50_26'] = df['feature_50']/(df['feature_26']+1)
        df['fe_ratio_24_25'] = df['feature_24']/(df['feature_25']+1)
        df['fe_ratio_45_38'] = df['feature_45'].fillna(0)/(df['feature_38'].fillna(0)+1)
        df['fe_ratio_29_7']  = df['feature_29']/(df['feature_7']+1)
        df['fe_25_x_29']     = df['feature_25']*df['feature_29']
        df['fe_7_x_2']       = df['feature_7']*df['feature_2']
        df['fe_24_sq']       = df['feature_24']**2
        df['fe_25_log']      = np.log1p(df['feature_25'].clip(lower=0))
        df['fe_45_log']      = np.log1p(df['feature_45'].fillna(0).clip(lower=0))
        df['fe_38_log']      = np.log1p(df['feature_38'].fillna(0).clip(lower=0))
        df['fe_12_x_24']     = df['feature_12'].fillna(0)*df['feature_24']
        df['fe_12_x_16']     = df['feature_12'].fillna(0)*df['feature_16'].fillna(0)
        df['fe_12_x_22']     = df['feature_12'].fillna(0)*df['feature_22'].fillna(0)
        df['fe_12_x_31']     = df['feature_12'].fillna(0)*df['feature_31'].fillna(0)
        df['fe_12_div_31']   = df['feature_12'].fillna(0)/(df['feature_31'].fillna(0)+1)
        df['fe_12_sq']       = df['feature_12'].fillna(0)**2
        mc = [f'{c}_miss' for c in HIGH_MISS if f'{c}_miss' in df.columns]
        if mc:
            df['fe_miss_pattern_sum'] = df[mc].sum(axis=1).astype(np.float32)
        return df

    train = engineer(train, feature_cols, BINARY)
    test  = engineer(test,  feature_cols, BINARY)
    print(f"  Feature engineering done  {T()}")

    # OOF target encoding
    TE_COLS = ['feature_24','feature_25','feature_29','feature_38',
               'feature_45','feature_2', 'feature_7', 'feature_10',
               'feature_40','feature_47','feature_48','feature_33',
               'feature_36','feature_37']

    def oof_te(train_df, test_df, cols, target, n_splits=5, smooth=30, n_bins=20):
        tr = train_df.copy(); te = test_df.copy()
        skf_i = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=SEED)
        gm = target.mean()
        for col in cols:
            if col not in train_df.columns: continue
            comb = pd.concat([train_df[col], test_df[col]])
            edg  = comb.quantile(np.linspace(0,1,n_bins+1)).unique()
            edg[0]=-np.inf; edg[-1]=np.inf
            trb = pd.cut(train_df[col], bins=edg, labels=False, duplicates='drop')
            teb = pd.cut(test_df[col],  bins=edg, labels=False, duplicates='drop')
            oe  = np.full(len(train_df), gm)
            for ti,vi in skf_i.split(train_df, target):
                fd = pd.DataFrame({'b':trb.iloc[ti],'y':target[ti]})
                st = fd.groupby('b')['y'].agg(['sum','count'])
                em = (st['sum']+smooth*gm)/(st['count']+smooth)
                oe[vi] = trb.iloc[vi].map(em).fillna(gm).values
            tr[f'te_{col}'] = oe.astype(np.float32)
            fa = pd.DataFrame({'b':trb,'y':target})
            sa = fa.groupby('b')['y'].agg(['sum','count'])
            ea = (sa['sum']+smooth*gm)/(sa['count']+smooth)
            te[f'te_{col}'] = teb.map(ea).fillna(gm).values.astype(np.float32)
        return tr, te

    train, test = oof_te(train, test, TE_COLS, y)
    all_feats   = [c for c in train.columns if c not in ['id','target']]
    print(f"  Total features: {len(all_feats)}  {T()}")

    # Imputation
    Xr = train[all_feats].values.astype(np.float32)
    Xt = test[all_feats].values.astype(np.float32)
    X_full_cat = Xr.copy()
    X_test_cat = Xt.copy()
    imp    = SimpleImputer(strategy='median')
    X_full = imp.fit_transform(Xr)
    X_test = imp.transform(Xt)
    print(f"  Imputed. NaN remaining: {np.isnan(X_full).sum()}  {T()}")

    # Cache features to disk
    np.savez_compressed(FEAT_CACHE,
                        X_full=X_full, X_test=X_test,
                        X_full_cat=X_full_cat, X_test_cat=X_test_cat,
                        y=y, ids_test=ids_test,
                        scale_pw=np.array([scale_pw]),
                        pos_rate=np.array([pos_rate]))
    print(f"  [SAVED] features.npz  {T()}")
    del train, test; gc.collect()

# SMOTE + CV shared setup
smote = SMOTE(sampling_strategy=SMOTE_RATIO, k_neighbors=5, random_state=SEED)
skf   = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)


# ═══════════════════════════════════════════════════════════════════════════
# SECTION 2  LIGHTGBM GBDT
# ═══════════════════════════════════════════════════════════════════════════
print(f"\nSECTION 2  LightGBM GBDT  {T()}")
oof_lgb, test_lgb, lgb_ap = load_ckpt('lgb')

if oof_lgb is None:
    LGB_P = {
        'objective':'binary', 'metric':'average_precision',
        'boosting_type':'gbdt', 'n_estimators':2500,
        'learning_rate':0.025, 'num_leaves':255, 'max_depth':8,
        'min_child_samples':30, 'feature_fraction':0.70,
        'bagging_fraction':0.75, 'bagging_freq':5,
        'lambda_l1':0.5, 'lambda_l2':1.0, 'min_split_gain':0.01,
        'scale_pos_weight':scale_pw, 'random_state':SEED,
        'n_jobs':-1, 'verbose':-1,
    }
    oof_lgb  = np.zeros(len(y));      test_lgb = np.zeros(len(X_test))
    for fold,(tr,val) in enumerate(skf.split(X_full, y)):
        t0 = time.time()
        Xs,ys = smote.fit_resample(X_full[tr], y[tr])
        m = lgb.LGBMClassifier(**LGB_P)
        m.fit(Xs, ys, eval_set=[(X_full[val],y[val])],
              callbacks=[lgb.early_stopping(100,verbose=False), lgb.log_evaluation(-1)])
        oof_lgb[val]  = m.predict_proba(X_full[val])[:,1]
        test_lgb     += m.predict_proba(X_test)[:,1] / N_FOLDS
        ap = average_precision_score(y[val], oof_lgb[val])
        print(f"  Fold {fold+1}/{N_FOLDS}  PR-AUC={ap:.5f}  iter={m.best_iteration_}  [{time.time()-t0:.0f}s]  {T()}")
        del m,Xs,ys; gc.collect()
    lgb_ap = average_precision_score(y, oof_lgb)
    print(f"  ★ LGB OOF PR-AUC: {lgb_ap:.5f}  ROC-AUC: {roc_auc_score(y,oof_lgb):.5f}")
    save_ckpt('lgb', oof_lgb, test_lgb, lgb_ap)


# ═══════════════════════════════════════════════════════════════════════════
# SECTION 3  XGBOOST
# ═══════════════════════════════════════════════════════════════════════════
print(f"\nSECTION 3  XGBoost  {T()}")
oof_xgb, test_xgb, xgb_ap = load_ckpt('xgb')

if oof_xgb is None:
    XGB_P = {
        'objective':'binary:logistic', 'eval_metric':'aucpr',
        'n_estimators':2500, 'learning_rate':0.025, 'max_depth':7,
        'subsample':0.70, 'colsample_bytree':0.70, 'min_child_weight':10,
        'gamma':0.1, 'reg_alpha':0.5, 'reg_lambda':1.0,
        'scale_pos_weight':scale_pw, 'tree_method':'hist', 'device':'cpu',
        'random_state':SEED, 'n_jobs':-1, 'verbosity':0,
        'early_stopping_rounds':100,
    }
    oof_xgb  = np.zeros(len(y));      test_xgb = np.zeros(len(X_test))
    for fold,(tr,val) in enumerate(skf.split(X_full, y)):
        t0 = time.time()
        Xs,ys = smote.fit_resample(X_full[tr], y[tr])
        m = xgb.XGBClassifier(**XGB_P)
        m.fit(Xs,ys, eval_set=[(X_full[val],y[val])], verbose=False)
        oof_xgb[val]  = m.predict_proba(X_full[val])[:,1]
        test_xgb     += m.predict_proba(X_test)[:,1] / N_FOLDS
        ap = average_precision_score(y[val], oof_xgb[val])
        print(f"  Fold {fold+1}/{N_FOLDS}  PR-AUC={ap:.5f}  iter={m.best_iteration}  [{time.time()-t0:.0f}s]  {T()}")
        del m,Xs,ys; gc.collect()
    xgb_ap = average_precision_score(y, oof_xgb)
    print(f"  ★ XGB OOF PR-AUC: {xgb_ap:.5f}")
    save_ckpt('xgb', oof_xgb, test_xgb, xgb_ap)


# ═══════════════════════════════════════════════════════════════════════════
# SECTION 4  CATBOOST
# ═══════════════════════════════════════════════════════════════════════════
print(f"\nSECTION 4  CatBoost  {T()}")
oof_cat, test_cat, cat_ap = load_ckpt('cat')

if oof_cat is None:
    CAT_P = {
        'iterations':2500, 'learning_rate':0.025, 'depth':8,
        'l2_leaf_reg':3.0, 'bagging_temperature':0.5, 'random_strength':1.0,
        'scale_pos_weight':scale_pw, 'eval_metric':'PRAUC',
        'loss_function':'Logloss', 'random_seed':SEED,
        'task_type':'CPU', 'verbose':False,
        'early_stopping_rounds':100, 'use_best_model':True,
    }
    oof_cat  = np.zeros(len(y));      test_cat = np.zeros(len(X_test))
    for fold,(tr,val) in enumerate(skf.split(X_full, y)):
        t0 = time.time()
        Xs,ys = smote.fit_resample(X_full[tr], y[tr])
        m = CatBoostClassifier(**CAT_P)
        m.fit(Xs,ys, eval_set=(X_full_cat[val],y[val]), verbose=False)
        oof_cat[val]  = m.predict_proba(X_full_cat[val])[:,1]
        test_cat     += m.predict_proba(X_test_cat)[:,1] / N_FOLDS
        ap = average_precision_score(y[val], oof_cat[val])
        print(f"  Fold {fold+1}/{N_FOLDS}  PR-AUC={ap:.5f}  iter={m.best_iteration_}  [{time.time()-t0:.0f}s]  {T()}")
        del m,Xs,ys; gc.collect()
    cat_ap = average_precision_score(y, oof_cat)
    print(f"  ★ CAT OOF PR-AUC: {cat_ap:.5f}")
    save_ckpt('cat', oof_cat, test_cat, cat_ap)


# ═══════════════════════════════════════════════════════════════════════════
# SECTION 5  BLEND + RANK NORM + SUBMISSION
# ═══════════════════════════════════════════════════════════════════════════
print(f"\nSECTION 5  Blend + Submission  {T()}")

best_ap = 0; best_w = (0.33,0.33,0.34); step = 0.05
for wl in np.arange(0, 1+step, step):
    for wx in np.arange(0, 1-wl+step, step):
        wc = max(1-wl-wx, 0.0)
        if wc < -1e-9: continue
        ap = average_precision_score(y, wl*oof_lgb + wx*oof_xgb + wc*oof_cat)
        if ap > best_ap: best_ap=ap; best_w=(wl,wx,wc)

wl,wx,wc   = best_w
oof_blend  = wl*oof_lgb  + wx*oof_xgb  + wc*oof_cat
test_blend = wl*test_lgb + wx*test_xgb + wc*test_cat
print(f"  LGB={wl:.2f}  XGB={wx:.2f}  CAT={wc:.2f}  Blend PR-AUC={best_ap:.5f}")

n         = len(oof_blend)
oof_ranks = rankdata(oof_blend) / (n+1)
test_norm = np.interp(test_blend, np.sort(oof_blend), np.sort(oof_ranks))
norm_ap   = average_precision_score(y, oof_ranks)
final     = test_norm if norm_ap >= best_ap else test_blend
print(f"  Rank-norm PR-AUC: {norm_ap:.5f}  → using {'rank-norm' if norm_ap>=best_ap else 'raw blend'}")

pd.DataFrame({'id': ids_test, 'target': final}).to_csv(OUTPUT_PATH, index=False)

print("\n" + "="*65)
print("  FINAL RESULTS")
print("="*65)
print(f"  LGB GBDT  : {lgb_ap:.5f}")
print(f"  XGBoost   : {xgb_ap:.5f}")
print(f"  CatBoost  : {cat_ap:.5f}")
print(f"  Blend     : {best_ap:.5f}")
print(f"  Final     : {max(best_ap, norm_ap):.5f}")
print(f"  Runtime   : {(time.time()-t_start)/60:.1f} min")
print(f"  Output    : {OUTPUT_PATH}")
print("="*65)
print("  submission_v4_zerve.csv SAVED!")
