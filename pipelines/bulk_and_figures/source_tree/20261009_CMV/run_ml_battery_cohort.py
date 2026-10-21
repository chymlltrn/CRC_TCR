#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Cohort-specific pN prediction battery — continuation of pn_ml_models_merged (2026-10-09).
Scenarios: Home-only (126 patients) and External-only (163 labelable of 202).
Protocol is IDENTICAL to run_ml_battery_merged.py (same 12 methods, grids, pipeline,
seed 42, nested 5x3 CV, paired bootstrap 5000 + Holm) — only the cohort subset changes,
so that the 3-scenario x A/B comparison is fair and frozen.

Usage: run_ml_battery_cohort.py {home|external}

Scheme notes:
  home    : stratLabelBatchCohort = label x batchgroup(B/C/D) [6 strata, min 1_B=5];
            stratLabel = plain label stratification (contrast scheme, as in merged run).
  external: stratLabelBatchCohort = design stratum (label x EXT -> degenerates to label;
            label x pool not feasible, min cell 3 < 5). Pool structure retained as
            in-model covariates via *_cov arms (external_technical_batch_pool* dummies).
            stratLabel duplicate scheme therefore NOT rerun.

LEAKAGE DISCIPLINE (unchanged from merged run): the 3 clusters were SELECTED by nominal
P<0.05 on pooled289 labels -> Home-only and External-only numbers ALSO carry selection
optimism (both cohorts' labels were used in selection). Reported as optimistic bounds.
"""
from release_runtime import private_literal as _private_literal, release_path as _release_path
import os, sys, json, time, warnings
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
COHORT = sys.argv[1]
assert COHORT in ('home', 'external')
MERGED = _release_path('path_6fe8b0cdaf15a381')
OUT = f"{_release_path('path_5bd48f922d960395')}{COHORT}"
os.makedirs(f'{OUT}/results', exist_ok=True)
os.makedirs(f'{OUT}/src', exist_ok=True)
rng_global = np.random.RandomState(RS)

design_full = pd.read_csv(f'{MERGED}/src/merged_design.csv', index_col=0)
mask = (design_full['cohort_external'] == (0 if COHORT == 'home' else 1)).values
design = design_full[mask]
y = design['y'].astype(int).values
strata_main = design['stratum'].values
feat_sets = json.load(open(f'{MERGED}/src/feature_sets.json'))
X_all = pd.read_csv(f'{MERGED}/src/X_merged_all.csv', index_col=0).loc[design.index]
assert X_all.index.equals(design.index)
design.to_csv(f'{OUT}/src/design_{COHORT}.csv')

DEPTH2 = ['log1p_reads_z', 'log1p_unique_z']
BATCHC = feat_sets['_batchcohorts']
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
    'BASE_depthcohbatch': DEPTH2 + BATCHC,   # KEY baseline (within-cohort: depth + own batch dummies)
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

if COHORT == 'home':
    SCHEMES = {'stratLabelBatchCohort': strata_main, 'stratLabel': y}
else:
    SCHEMES = {'stratLabelBatchCohort': strata_main}  # == label strat; label x pool infeasible (min cell 3)
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

def n_varying(feats):
    sub = X_all[feats]
    return int(sum(sub[c].nunique(dropna=False) > 1 for c in sub.columns))

# ---------------- 1) nested-CV battery ----------------
rows, oofs = [], {}
jobs = []
for sk in SCHEMES:
    for setname, feats in SETS.items():
        for mname, (clf, grid) in MODELS.items():
            jobs.append((f'{mname}__{setname}__{sk}', feats, clf, grid, sk))
    for bname, feats in BASE_SETS.items():
        jobs.append((f'{bname}__logL2__{sk}', feats, BASE_CLF, {'clf__C': [0.03, 0.3, 3]}, sk))
print(f'{COHORT}: {len(jobs)} nested arms, n={len(y)} pos={int(y.sum())}', flush=True)
t00 = time.time()
degenerate_arms = []
for i, (nm, feats, clf, grid, sk) in enumerate(jobs):
    if n_varying(feats) < 2:   # cohort-constant covariates (e.g. External age/sex fully imputed)
        degenerate_arms.append(nm)
        rows.append(dict(arm=nm, auc_fold_mean=np.nan, auc_fold_sd=np.nan, auc_fold_ci_lo=np.nan,
                         auc_fold_ci_hi=np.nan, auc_pooled_oof=np.nan, auc_pooled_ci95_lo=np.nan,
                         auc_pooled_ci95_hi=np.nan, prauc_pooled=np.nan, fold_aucs='degenerate',
                         best_params='skipped: <2 varying features within cohort subset',
                         seconds=0.0))
        pd.DataFrame(rows).to_csv(f'{OUT}/results/_partial_rows.csv', index=False)
        print(f'[{i+1}/{len(jobs)}] {nm}: SKIPPED (degenerate, <2 varying features)', flush=True)
        continue
    rec, oof, bps = run_arm(nm, feats, clf, grid, SCHEMES[sk])

    rows.append(rec); oofs[nm] = oof
    pd.DataFrame(rows).to_csv(f'{OUT}/results/_partial_rows.csv', index=False)
    print(f'[{i+1}/{len(jobs)}] {nm}: pooled={rec["auc_pooled_oof"]:.3f} fold {rec["auc_fold_mean"]:.3f}+-{rec["auc_fold_sd"]:.3f} ({rec["seconds"]}s)', flush=True)
res = pd.DataFrame(rows).set_index('arm')
res.reset_index().to_csv(f'{OUT}/results/model_performance_full.csv', index=False)
pd.DataFrame(oofs, index=X_all.index).to_csv(f'{OUT}/results/oof_predictions_all_arms.csv')

# ---------------- 2) contrasts ----------------
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
    if base not in oofs:
        print(f'SKIP vs-baseline contrasts ({sk}): base arm degenerate', flush=True)
        continue
    bo = oofs[base]; base_auc = res.loc[base, 'auc_pooled_oof']
    base_folds = np.array([float(x) for x in res.loc[base, 'fold_aucs'].split(';')])
    fam = [a for a in res.index if a.endswith(f'__{sk}') and a.split('__')[0] in MODELS
           and 'degenerate' not in str(res.loc[a, 'fold_aucs'])]
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
for sk in SCHEMES:
    for a_, b_, tag in [('CLUSTER3', 'REPO', 'CLUSTER3_vs_REPO'), ('CLUSTER3cov', 'REPOcov', 'CLUSTER3cov_vs_REPOcov')]:
        ps = []
        for mname in MODELS:
            a = f'{mname}__{a_}__{sk}'; b = f'{mname}__{b_}__{sk}'
            if a not in oofs or b not in oofs:
                continue
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

json.dump(dict(run=f'pn_ml_models_{COHORT}', cohort=COHORT, n_nested_arms=len(jobs),
               n_samples=int(len(y)), n_pos=int(y.sum()), seed=RS, n_jobs=NJ,
               schemes=list(SCHEMES.keys()),
               degenerate_arms=degenerate_arms,
               scheme_note=('label x EXT degenerates to label; label x pool infeasible (min cell 3<5); pool kept as in-model covariates'
                            if COHORT == 'external' else 'main = label x batchgroup(B/C/D); contrast = label'),
               runtime_min=round((time.time() - t00) / 60, 1)),
          open(f'{OUT}/results/run_meta.json', 'w'), indent=2)
print(f'BATTERY {COHORT} DONE in', round((time.time() - t00) / 60, 1), 'min', flush=True)
