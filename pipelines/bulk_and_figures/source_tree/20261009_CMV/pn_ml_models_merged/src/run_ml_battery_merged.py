#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Merged-cohort (Home 126 + External 163 = 289 known-N patients) pN prediction battery.
Run: 2026-10-09. Analyst: OmicOS immune_repertoire_analyst_pro.
Feature arms:
  CLUSTER3      - frozen TCR-BERT/SCEPTR Leiden clusters tcrbert:11, sceptr:5, tcrbert:7
                  (log((reads+1)/(total+1)), read scale - the exact pooled289 selection scale)
  REPO          - merged-cohort 262-equivalent MiXCR repertoire blocks (label-free rebuild:
                  scalars/Lr/AA/PC/kmer4 + Vr/Jr from atlas clone_detail, qc_pass rows)
  CONCAT        - CLUSTER3 + REPO
  *_cov         - each + depth(2) + cohort + batch dummies + age + sex + T_ordinal
Baselines: depth-only / cohort+batch-only / depth+cohort+batch (KEY) / T-stage / age+sex.
MAIN scheme: StratifiedKFold(5, rs=42) on label x batchgroup (B/C/D/EXT) strata;
CONTRAST: plain label-stratified. Nested: inner 3-fold GridSearchCV (AUC), all in-fold.
Uniform pipeline: median impute -> var filter -> Spearman top-50 (only >60 feats, train-fold only)
-> standardize -> classifier. 12 methods; seed 42.
LEAKAGE DISCIPLINE: the 3 clusters were SELECTED by nominal P<0.05 on pooled289 (Home+External
labels) -> merged-cohort numbers carry selection optimism; frozen-definition transfer arms
(train Home -> test External, and reverse) are reported alongside as the honest evaluation.
"""
from release_runtime import private_literal as _private_literal, release_path as _release_path
import os, json, time, warnings, pickle
import numpy as np, pandas as pd
from scipy.stats import spearmanr, t as tdist, ttest_rel
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import StandardScaler
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import roc_auc_score, average_precision_score
from xgboost import XGBClassifier

warnings.filterwarnings('ignore')
RS = 42
OUT = _release_path('path_6fe8b0cdaf15a381')
rng_global = np.random.RandomState(RS)

design = pd.read_csv(f'{OUT}/src/merged_design.csv', index_col=0)
y = design['y'].astype(int).values
cohort = design['cohort_external'].values
strata_main = design['stratum'].values
feat_sets = json.load(open(f'{OUT}/src/feature_sets.json'))
X_all = pd.read_csv(f'{OUT}/src/X_merged_all.csv', index_col=0).loc[design.index]
assert X_all.index.equals(design.index)

DEPTH2 = ['log1p_reads_z', 'log1p_unique_z']
BATCHC = feat_sets['_batchcohorts']          # cohort_external + home_batch_* + technical_batch_*
CLIN   = ['age', 'sex_m', 'T_ordinal']
COV    = DEPTH2 + BATCHC + CLIN

SETS = {
    'CLUSTER3'  : feat_sets['CLUSTER3'],
    'REPO'      : feat_sets['REPO'],
    'CONCAT'    : feat_sets['CLUSTER3'] + feat_sets['REPO'],
    'CLUSTER3cov': feat_sets['CLUSTER3'] + COV,
    'REPOcov'   : feat_sets['REPO'] + COV,
    'CONCATcov' : feat_sets['CLUSTER3'] + feat_sets['REPO'] + COV,
}
BASE_SETS = {
    'BASE_depth'        : DEPTH2,
    'BASE_cohortbatch'  : BATCHC,
    'BASE_depthcohbatch': DEPTH2 + BATCHC,   # KEY baseline
    'BASE_Tstage'       : ['T_ordinal'],
    'BASE_agesex'       : ['age', 'sex_m'],
}

class SpearmanTopK(BaseEstimator, TransformerMixin):
    def __init__(self, k=50): self.k = k
    def fit(self, X, y):
        X = np.asarray(X); y = np.asarray(y)
        if X.shape[1] <= 60:
            self.idx_ = np.arange(X.shape[1]); return self
        rs = np.array([abs(spearmanr(X[:, j], y).statistic) if np.std(X[:, j]) > 0 else 0.0
                       for j in range(X.shape[1])])
        self.idx_ = np.argsort(-rs)[:min(self.k, X.shape[1])]
        return self
    def transform(self, X): return np.asarray(X)[:, self.idx_]

def make_pipe(clf):
    return Pipeline([('imp', SimpleImputer(strategy='median')),
                     ('var', VarianceThreshold(1e-12)),
                     ('scr', SpearmanTopK(k=50)),
                     ('scl', StandardScaler()),
                     ('clf', clf)])

MODELS = {
 'LASSO_logistic': (LogisticRegression(penalty='l1', solver='liblinear', max_iter=5000, random_state=RS),
                    {'clf__C': [0.01, 0.03, 0.1, 0.3, 1, 3]}),
 'Ridge_logistic': (LogisticRegression(penalty='l2', solver='liblinear', max_iter=5000, random_state=RS),
                    {'clf__C': [0.01, 0.03, 0.1, 0.3, 1, 3]}),
 'ElasticNet_logistic': (LogisticRegression(penalty='elasticnet', solver='saga', max_iter=8000, random_state=RS),
                    {'clf__C': [0.03, 0.1, 0.3, 1], 'clf__l1_ratio': [0.2, 0.5, 0.8]}),
 'Logistic_unpenalized': (LogisticRegression(penalty=None, solver='lbfgs', max_iter=8000, random_state=RS), {}),
 'RandomForest': (RandomForestClassifier(n_estimators=500, random_state=RS, n_jobs=2),
                    {'clf__max_depth': [None, 4, 8], 'clf__min_samples_leaf': [2, 5]}),
 'HistGradBoost': (HistGradientBoostingClassifier(max_leaf_nodes=15, min_samples_leaf=10,
                    l2_regularization=1.0, random_state=RS),
                    {'clf__max_iter': [100, 300], 'clf__learning_rate': [0.05, 0.1]}),
 'XGBoost': (XGBClassifier(n_jobs=2, random_state=RS, eval_metric='logloss',
                           subsample=0.8, colsample_bytree=0.8, verbosity=0),
             {'clf__n_estimators': [100, 300], 'clf__max_depth': [2, 3], 'clf__learning_rate': [0.05, 0.1]}),
 'SVM_linear': (SVC(kernel='linear', probability=False, random_state=RS), {'clf__C': [0.03, 0.3, 3, 30]}),
 'SVM_RBF': (SVC(kernel='rbf', probability=False, random_state=RS),
             {'clf__C': [0.3, 1, 3, 10], 'clf__gamma': ['scale', 0.01]}),
 'kNN': (KNeighborsClassifier(), {'clf__n_neighbors': [5, 9, 15, 25], 'clf__weights': ['uniform', 'distance']}),
 'GaussianNB': (GaussianNB(), {'clf__var_smoothing': [1e-9, 1e-7, 1e-5]}),
 'MLP': (MLPClassifier(early_stopping=True, max_iter=3000, random_state=RS),
         {'clf__alpha': [1e-3, 1e-2, 1e-1], 'clf__hidden_layer_sizes': [(32, 16), (64, 32)]}),
}
BASE_CLF = LogisticRegression(penalty='l2', solver='liblinear', max_iter=5000, random_state=RS)

SCHEMES = {'stratLabelBatchCohort': strata_main, 'stratLabel': y}
NJ = min(8, os.cpu_count() or 4)

def score_of(pipe, Xte):
    return pipe.predict_proba(Xte)[:, 1] if hasattr(pipe, 'predict_proba') else pipe.decision_function(Xte)

def pooled_boot_ci(yy, oof, n=2000):
    bs = []
    for _ in range(n):
        idx = rng_global.randint(0, len(yy), len(yy))
        if len(np.unique(yy[idx])) < 2: continue
        bs.append(roc_auc_score(yy[idx], oof[idx]))
    lo, hi = np.percentile(bs, [2.5, 97.5])
    return lo, hi, np.array(bs)

def run_arm(name, feats, clf, grid, split_labels):
    X = X_all[feats].values
    outer = StratifiedKFold(5, shuffle=True, random_state=RS)
    inner = StratifiedKFold(3, shuffle=True, random_state=RS)
    folds_auc, folds_ap, oof, bps = [], [], np.full(len(y), np.nan), []
    t0 = time.time()
    for tr, te in outer.split(X, split_labels):
        gs = GridSearchCV(make_pipe(clf), grid or {}, cv=inner, scoring='roc_auc', n_jobs=NJ, refit=True)
        gs.fit(X[tr], y[tr])
        oof[te] = score_of(gs.best_estimator_, X[te])
        folds_auc.append(roc_auc_score(y[te], oof[te]))
        folds_ap.append(average_precision_score(y[te], oof[te]))
        bps.append(dict(gs.best_params_))
    auc = np.array(folds_auc); ap = np.array(folds_ap)
    pooled_auc = roc_auc_score(y, oof); pooled_ap = average_precision_score(y, oof)
    lo, hi, _ = pooled_boot_ci(y, oof)
    mean, sd = auc.mean(), auc.std(ddof=1); tcrit = tdist.ppf(0.975, len(auc) - 1)
    rec = dict(arm=name, auc_fold_mean=mean, auc_fold_sd=sd,
               auc_fold_ci_lo=mean - tcrit * sd / np.sqrt(len(auc)),
               auc_fold_ci_hi=mean + tcrit * sd / np.sqrt(len(auc)),
               auc_pooled_oof=pooled_auc, auc_pooled_ci95_lo=lo, auc_pooled_ci95_hi=hi,
               prauc_pooled=pooled_ap,
               fold_aucs=';'.join(f'{a:.4f}' for a in auc),
               best_params=' || '.join(json.dumps({k: str(v) for k, v in b.items()}) for b in bps),
               seconds=round(time.time() - t0, 1))
    return rec, oof, bps

def holm(ps):
    ps = np.asarray(ps); order = np.argsort(ps); m = len(ps)
    q = np.empty(m); running = 0.0
    for rank, oi in enumerate(order):
        val = (m - rank) * ps[oi]; running = max(running, val); q[oi] = min(1.0, running)
    return q

# ---------------- 1) nested-CV battery ----------------
rows, oofs, raw_bps = [], {}, {}
jobs = []
for sk in SCHEMES:
    for setname, feats in SETS.items():
        for mname, (clf, grid) in MODELS.items():
            jobs.append((f'{mname}__{setname}__{sk}', feats, clf, grid, sk))
    for bname, feats in BASE_SETS.items():
        jobs.append((f'{bname}__logL2__{sk}', feats, BASE_CLF, {'clf__C': [0.03, 0.3, 3]}, sk))
print(f'{len(jobs)} nested arms total', flush=True)
t00 = time.time()
for i, (nm, feats, clf, grid, sk) in enumerate(jobs):
    rec, oof, bps = run_arm(nm, feats, clf, grid, SCHEMES[sk])
    rows.append(rec); oofs[nm] = oof; raw_bps[nm] = bps
    pd.DataFrame(rows).to_csv(f'{OUT}/results/_partial_rows.csv', index=False)
    print(f'[{i+1}/{len(jobs)}] {nm}: pooled={rec["auc_pooled_oof"]:.3f} fold {rec["auc_fold_mean"]:.3f}±{rec["auc_fold_sd"]:.3f} ({rec["seconds"]}s)', flush=True)
res = pd.DataFrame(rows).set_index('arm')
res.reset_index().to_csv(f'{OUT}/results/model_performance_full.csv', index=False)
pd.DataFrame(oofs, index=X_all.index).to_csv(f'{OUT}/results/oof_predictions_all_arms.csv')

# ---------------- 2) contrasts vs KEY baseline + CLUSTER3 vs REPO ----------------
def paired_boot(mo, bo, yy=y, n=5000):
    d = []
    for _ in range(n):
        idx = rng_global.randint(0, len(yy), len(yy))
        if len(np.unique(yy[idx])) < 2: continue
        d.append(roc_auc_score(yy[idx], mo[idx]) - roc_auc_score(yy[idx], bo[idx]))
    return np.array(d)

contrast_rows = []
for sk in SCHEMES:
    base = f'BASE_depthcohbatch__logL2__{sk}'
    bo = oofs[base]; base_auc = res.loc[base, 'auc_pooled_oof']
    base_folds = np.array([float(x) for x in res.loc[base, 'fold_aucs'].split(';')])
    fam = [a for a in res.index if a.endswith(f'__{sk}') and a.split('__')[0] in MODELS]
    ps = []
    for a in fam:
        mo = oofs[a]; diffs = paired_boot(mo, bo)
        m_folds = np.array([float(x) for x in res.loc[a, 'fold_aucs'].split(';')])
        tt = ttest_rel(m_folds, base_folds)
        contrast_rows.append(dict(arm=a, scheme=sk, contrast='vs_BASE_depthcohbatch',
            delta_auc_pooled=res.loc[a, 'auc_pooled_oof'] - base_auc, delta_boot_mean=diffs.mean(),
            delta_ci95_lo=np.percentile(diffs, 2.5), delta_ci95_hi=np.percentile(diffs, 97.5),
            p_paired_boot=(1 + np.sum(diffs <= 0)) / (1 + len(diffs)), p_fold_ttest=tt.pvalue))
        ps.append(contrast_rows[-1]['p_paired_boot'])
    for rr, qq in zip([c for c in contrast_rows if c['scheme'] == sk and c['contrast'] == 'vs_BASE_depthcohbatch'], holm(ps)):
        rr['p_holm'] = qq
# key question: CLUSTER3 vs REPO (and cov variants), per model
for sk in SCHEMES:
    for a_, b_, tag in [('CLUSTER3', 'REPO', 'CLUSTER3_vs_REPO'), ('CLUSTER3cov', 'REPOcov', 'CLUSTER3cov_vs_REPOcov')]:
        ps = []
        for mname in MODELS:
            a = f'{mname}__{a_}__{sk}'; b = f'{mname}__{b_}__{sk}'
            diffs = paired_boot(oofs[a], oofs[b])
            contrast_rows.append(dict(arm=f'{mname}__{a_}', scheme=sk, contrast=tag,
                delta_auc_pooled=res.loc[a, 'auc_pooled_oof'] - res.loc[b, 'auc_pooled_oof'],
                delta_boot_mean=diffs.mean(),
                delta_ci95_lo=np.percentile(diffs, 2.5), delta_ci95_hi=np.percentile(diffs, 97.5),
                p_paired_boot=(1 + np.sum(diffs <= 0)) / (1 + len(diffs))))
            ps.append(contrast_rows[-1]['p_paired_boot'])
        for rr, qq in zip([c for c in contrast_rows if c['scheme'] == sk and c['contrast'] == tag], holm(ps)):
            rr['p_holm'] = qq
con = pd.DataFrame(contrast_rows)
con.to_csv(f'{OUT}/results/contrasts.csv', index=False)

# ---------------- 3) frozen-definition transfer (leakage-honest) ----------------
home_m = cohort == 0; ext_m = cohort == 1
transfer_rows = []
tsets = dict(SETS); tsets['BASE_depthcohbatch'] = BASE_SETS['BASE_depthcohbatch']
for setname, feats in tsets.items():
    for mname, (clf, grid) in ([('logL2', (BASE_CLF, {'clf__C': [0.03, 0.3, 3]}))] if setname.startswith('BASE')
                               else list(MODELS.items())):
        Xf = X_all[feats].values
        for train_m, test_m, tname in [(home_m, ext_m, 'Home2Ext'), (ext_m, home_m, 'Ext2Home')]:
            inner = StratifiedKFold(3, shuffle=True, random_state=RS)
            gs = GridSearchCV(make_pipe(clf), grid or {}, cv=inner, scoring='roc_auc', n_jobs=NJ, refit=True)
            gs.fit(Xf[train_m], y[train_m])
            sc = score_of(gs.best_estimator_, Xf[test_m])
            auc_t = roc_auc_score(y[test_m], sc); ap_t = average_precision_score(y[test_m], sc)
            lo, hi, _ = pooled_boot_ci(y[test_m], sc, n=2000)
            transfer_rows.append(dict(arm=f'{mname}__{setname}', direction=tname, auc_test=auc_t,
                auc_ci95_lo=lo, auc_ci95_hi=hi, prauc=ap_t, best_params=json.dumps({k: str(v) for k, v in gs.best_params_.items()})))
            print(f'TRANSFER {mname}__{setname} {tname}: AUC={auc_t:.3f} ({lo:.3f}-{hi:.3f})', flush=True)
tr = pd.DataFrame(transfer_rows); tr.to_csv(f'{OUT}/results/transfer_frozen_definition.csv', index=False)

json.dump(dict(n_nested_arms=len(jobs), n_samples=int(len(y)), n_pos=int(y.sum()),
               n_home=int(home_m.sum()), n_ext=int(ext_m.sum()), seed=RS, n_jobs=NJ,
               runtime_min=round((time.time() - t00) / 60, 1)),
          open(f'{OUT}/results/run_meta.json', 'w'), indent=2)
print('BATTERY DONE in', round((time.time() - t00) / 60, 1), 'min', flush=True)
