#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
One-arm repair for pn_ml_models_external: the continuation patch used a wrong
degeneracy threshold (<2 varying features) and skipped BASE_Tstage, which is a
valid single-feature arm (T_ordinal varies in External; first attempt computed
it fine before crashing on BASE_agesex). Re-run exactly that arm with the same
deterministic protocol (seed 42, same pipeline, same split labels), splice the
row into model_performance_full.csv and the OOF column into
oof_predictions_all_arms.csv, and log the repair in run_meta.json.
"""
from release_runtime import private_literal as _private_literal, release_path as _release_path
import json, warnings
import numpy as np, pandas as pd
from scipy.stats import t as tdist
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import StandardScaler
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score

warnings.filterwarnings('ignore')
RS = 42
MERGED = _release_path('path_6fe8b0cdaf15a381')
OUT = _release_path('path_a450e0588c0571ed')
rng = np.random.RandomState(RS)

design = pd.read_csv(f'{MERGED}/src/merged_design.csv', index_col=0)
design = design[design['cohort_external'] == 1]
y = design['y'].astype(int).values
strata = design['stratum'].values
X_all = pd.read_csv(f'{MERGED}/src/X_merged_all.csv', index_col=0).loc[design.index]

class SpearmanTopK(BaseEstimator, TransformerMixin):
    def __init__(self, k=50): self.k = k
    def fit(self, X, y):
        X = np.asarray(X); y = np.asarray(y)
        if X.shape[1] <= 60:
            self.idx_ = np.arange(X.shape[1]); return self
        from scipy.stats import spearmanr
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

feats = ['T_ordinal']
name = 'BASE_Tstage__logL2__stratLabelBatchCohort'
X = X_all[feats].values
clf = LogisticRegression(penalty='l2', solver='liblinear', max_iter=5000, random_state=RS)
grid = {'clf__C': [0.03, 0.3, 3]}
outer = StratifiedKFold(5, shuffle=True, random_state=RS)
inner = StratifiedKFold(3, shuffle=True, random_state=RS)
folds_auc, folds_ap, oof, bps = [], [], np.full(len(y), np.nan), []
for tr, te in outer.split(X, strata):
    gs = GridSearchCV(make_pipe(clf), grid, cv=inner, scoring='roc_auc', n_jobs=8, refit=True)
    gs.fit(X[tr], y[tr])
    oof[te] = gs.predict_proba(X[te])[:, 1]
    folds_auc.append(roc_auc_score(y[te], oof[te]))
    folds_ap.append(average_precision_score(y[te], oof[te]))
    bps.append(dict(gs.best_params_))
auc = np.array(folds_auc); ap = np.array(folds_ap)
bs = []
for _ in range(2000):
    idx = rng.randint(0, len(y), len(y))
    if len(np.unique(y[idx])) < 2: continue
    bs.append(roc_auc_score(y[idx], oof[idx]))
lo, hi = np.percentile(bs, [2.5, 97.5])
mean, sd = auc.mean(), auc.std(ddof=1); tcrit = tdist.ppf(0.975, len(auc) - 1)
row = dict(arm=name, auc_fold_mean=mean, auc_fold_sd=sd,
           auc_fold_ci_lo=mean - tcrit * sd / np.sqrt(len(auc)),
           auc_fold_ci_hi=mean + tcrit * sd / np.sqrt(len(auc)),
           auc_pooled_oof=roc_auc_score(y, oof), auc_pooled_ci95_lo=lo, auc_pooled_ci95_hi=hi,
           prauc_pooled=average_precision_score(y, oof),
           fold_aucs=';'.join(f'{a:.4f}' for a in auc),
           best_params=' || '.join(json.dumps({k: str(v) for k, v in b.items()}) for b in bps),
           seconds=0.0)
print('repaired arm:', {k: (round(v,4) if isinstance(v,float) else v) for k,v in row.items() if k in
      ('arm','auc_pooled_oof','auc_pooled_ci95_lo','auc_pooled_ci95_hi','auc_fold_mean','prauc_pooled')})

perf = pd.read_csv(f'{OUT}/results/model_performance_full.csv')
assert not (perf.arm == name).any() or perf.loc[perf.arm == name, 'auc_pooled_oof'].isna().all()
perf = perf[perf.arm != name]
perf = pd.concat([perf, pd.DataFrame([row])], ignore_index=True)
perf.to_csv(f'{OUT}/results/model_performance_full.csv', index=False)

oof_df = pd.read_csv(f'{OUT}/results/oof_predictions_all_arms.csv', index_col=0)
oof_df[name] = pd.Series(oof, index=X_all.index)
oof_df.to_csv(f'{OUT}/results/oof_predictions_all_arms.csv')

try:
    meta = json.load(open(f'{OUT}/results/run_meta.json'))
except FileNotFoundError:
    meta = {}
meta['repair_BASE_Tstage'] = ('rerun as single deterministic arm after wrong <2-varying threshold '
                              'skipped it; protocol identical (seed 42); row spliced into '
                              'model_performance_full.csv and oof_predictions_all_arms.csv')
meta['degenerate_arms_final'] = ['BASE_agesex__logL2__stratLabelBatchCohort']
json.dump(meta, open(f'{OUT}/results/run_meta.json', 'w'), indent=2)
print('splice done')
