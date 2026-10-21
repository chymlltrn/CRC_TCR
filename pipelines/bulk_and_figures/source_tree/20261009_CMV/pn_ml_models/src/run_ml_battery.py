#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
pN status prediction from MiXCR TRB repertoire features — 12-model nested-CV battery.
Run: 2026-10-09. Analyst: OmicOS immune_repertoire_analyst_pro.
MAIN scheme: StratifiedKFold(5, rs=42) on label x batch strata; CONTRAST: plain label-stratified.
Uniform pipeline per model: median impute -> var filter -> Spearman top-50 screening (only if >60 feats,
fit on train only) -> standardize -> classifier; hyperparams tuned in INNER 3-fold CV (AUC).
Lasso-Cox infeasible (no survival outcome in cohort) -> LASSO-logistic represents the Lasso family.
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
OUT = _release_path('path_5fa1e30032a0d742')
rng_global = np.random.RandomState(RS)

X_tcr = pd.read_csv(f'{OUT}/src/X_tcr_only_262.csv', index_col=0)
cov   = pd.read_csv(f'{OUT}/src/X_covariates.csv', index_col=0)
lab   = pd.read_csv(f'{OUT}/src/labels_strata.csv', index_col=0)
cov['T_ordinal'] = cov['T_ordinal'].replace(9999, np.nan)
y = lab['y'].values.astype(int)
strata_main = lab['stratum'].values

DEPTH3 = ['depth_total_reads','depth_total_seq_reads','depth_aligned_reads']
BATCH2 = ['batch_B','batch_C']
SETS = {
    'TCR_only'      : list(X_tcr.columns),
    'TCR_depthbatch': list(X_tcr.columns) + DEPTH3 + BATCH2,
    'TCR_fullcov'   : list(X_tcr.columns) + DEPTH3 + BATCH2 + ['age','sex_m','T_ordinal'],
}
BASE_SETS = {
    'BASE_depth'      : DEPTH3,
    'BASE_batch'      : BATCH2,
    'BASE_depthbatch' : DEPTH3 + BATCH2,
    'BASE_Tstage'     : ['T_ordinal'],
    'BASE_agesex'     : ['age','sex_m'],
}
X_all = pd.concat([X_tcr, cov], axis=1).loc[lab.index]

class SpearmanTopK(BaseEstimator, TransformerMixin):
    """Keep top-K |Spearman rho| features vs y; fitted on TRAIN fold only."""
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

SCHEMES = {'stratLabelBatch': strata_main, 'stratLabel': y}
NJ = min(8, os.cpu_count() or 4)

def score_of(pipe, Xte):
    return pipe.predict_proba(Xte)[:, 1] if hasattr(pipe, 'predict_proba') else pipe.decision_function(Xte)

def run_arm(name, feats, clf, grid, scheme_key, split_labels):
    X = X_all[feats].values
    outer = StratifiedKFold(5, shuffle=True, random_state=RS)
    inner = StratifiedKFold(3, shuffle=True, random_state=RS)
    folds_auc, folds_ap, oof, bps, fitted = [], [], np.full(len(y), np.nan), [], []
    t0 = time.time()
    for tr, te in outer.split(X, split_labels):
        gs = GridSearchCV(make_pipe(clf), grid or {}, cv=inner, scoring='roc_auc', n_jobs=NJ, refit=True)
        gs.fit(X[tr], y[tr])
        oof[te] = score_of(gs.best_estimator_, X[te])
        folds_auc.append(roc_auc_score(y[te], oof[te]))
        folds_ap.append(average_precision_score(y[te], oof[te]))
        bps.append(dict(gs.best_params_))
        fitted.append(gs.best_estimator_)
    auc = np.array(folds_auc); ap = np.array(folds_ap)
    pooled_auc = roc_auc_score(y, oof); pooled_ap = average_precision_score(y, oof)
    bs = []
    for _ in range(2000):
        idx = rng_global.randint(0, len(y), len(y))
        if len(np.unique(y[idx])) < 2: continue
        bs.append(roc_auc_score(y[idx], oof[idx]))
    lo, hi = np.percentile(bs, [2.5, 97.5])
    mean, sd = auc.mean(), auc.std(ddof=1); tcrit = tdist.ppf(0.975, len(auc) - 1)
    rec = dict(arm=name, scheme=scheme_key, auc_fold_mean=mean, auc_fold_sd=sd,
               auc_fold_ci_lo=mean - tcrit * sd / np.sqrt(len(auc)),
               auc_fold_ci_hi=mean + tcrit * sd / np.sqrt(len(auc)),
               auc_pooled_oof=pooled_auc, auc_pooled_ci95_lo=lo, auc_pooled_ci95_hi=hi,
               prauc_pooled=pooled_ap,
               fold_aucs=';'.join(f'{a:.4f}' for a in auc),
               best_params=' || '.join(json.dumps({k: str(v) for k, v in b.items()}) for b in bps),
               seconds=round(time.time() - t0, 1))
    return rec, oof, fitted, bps

rows, oofs, pipes_store, raw_bps = [], {}, {}, {}
t00 = time.time()
jobs = []
for sk in SCHEMES:
    for setname, feats in SETS.items():
        for mname, (clf, grid) in MODELS.items():
            jobs.append((f'{mname}__{setname}__{sk}', feats, clf, grid, sk))
    for bname, feats in BASE_SETS.items():
        jobs.append((f'{bname}__logL2__{sk}', feats, BASE_CLF, {'clf__C': [0.03, 0.3, 3]}, sk))
print(f'{len(jobs)} arms total', flush=True)
for i, (nm, feats, clf, grid, sk) in enumerate(jobs):
    rec, oof, fitted, bps = run_arm(nm, feats, clf, grid, sk, SCHEMES[sk])
    rows.append(rec); oofs[nm] = oof; pipes_store[nm] = fitted; raw_bps[nm] = bps
    pd.DataFrame(rows).to_csv(f'{OUT}/results/_partial_rows.csv', index=False)
    print(f'[{i+1}/{len(jobs)}] {nm}: pooled={rec["auc_pooled_oof"]:.3f} '
          f'fold {rec["auc_fold_mean"]:.3f}±{rec["auc_fold_sd"]:.3f} ({rec["seconds"]}s)', flush=True)

res = pd.DataFrame(rows).set_index('arm')

def holm(ps):
    ps = np.asarray(ps); order = np.argsort(ps); m = len(ps)
    q = np.empty(m); running = 0.0
    for rank, oi in enumerate(order):
        val = (m - rank) * ps[oi]
        running = max(running, val)
        q[oi] = min(1.0, running)
    return q

contrast_rows = []
for sk in SCHEMES:
    base = f'BASE_depthbatch__logL2__{sk}'
    bo = oofs[base]; base_auc = res.loc[base, 'auc_pooled_oof']
    base_folds = np.array([float(x) for x in res.loc[base, 'fold_aucs'].split(';')])
    fam = [a for a in res.index if a.endswith(f'__{sk}') and a.split('__')[0] in MODELS]
    ps = []
    for a in fam:
        mo = oofs[a]
        diffs = []
        for _ in range(5000):
            idx = rng_global.randint(0, len(y), len(y))
            if len(np.unique(y[idx])) < 2: continue
            diffs.append(roc_auc_score(y[idx], mo[idx]) - roc_auc_score(y[idx], bo[idx]))
        diffs = np.array(diffs)
        m_folds = np.array([float(x) for x in res.loc[a, 'fold_aucs'].split(';')])
        tt = ttest_rel(m_folds, base_folds)
        contrast_rows.append(dict(arm=a, scheme=sk,
            delta_auc_pooled=res.loc[a, 'auc_pooled_oof'] - base_auc,
            delta_boot_mean=diffs.mean(),
            delta_ci95_lo=np.percentile(diffs, 2.5), delta_ci95_hi=np.percentile(diffs, 97.5),
            p_paired_boot=(1 + np.sum(diffs <= 0)) / (1 + len(diffs)),
            p_fold_ttest=tt.pvalue))
        ps.append(contrast_rows[-1]['p_paired_boot'])
    for rr, qq in zip([c for c in contrast_rows if c['scheme'] == sk], holm(ps)):
        rr['p_holm'] = qq
con = pd.DataFrame(contrast_rows)
res.reset_index().to_csv(f'{OUT}/results/model_performance_full.csv', index=False)
con.to_csv(f'{OUT}/results/contrasts_vs_depthbatch.csv', index=False)
pd.DataFrame(oofs, index=X_all.index).to_csv(f'{OUT}/results/oof_predictions_all_arms.csv')
with open(f'{OUT}/artifacts/fitted_pipelines.pkl', 'wb') as f: pickle.dump(pipes_store, f)
print('battery done in', round((time.time() - t00) / 60, 1), 'min', flush=True)

# ---- flagship (best TCR arm on MAIN scheme) + permutation null ----
main = res[res.scheme == 'stratLabelBatch']
flag_arm = main[main.index.str.split('__').str[0].isin(MODELS.keys())].sort_values('auc_pooled_oof').index[-1]
print('FLAGSHIP:', flag_arm, flush=True)

def modal_params(arm):
    keys = raw_bps[arm][0].keys()
    return {k: pd.Series([b[k] for b in raw_bps[arm]]).mode()[0] for k in keys}

def pooled_auc_fixed(feats, clf, fixed, split_labels, yy=None, n_perm=200, do_perm=True):
    X = X_all[feats].values
    outer = StratifiedKFold(5, shuffle=True, random_state=RS)
    def cv(yv):
        o = np.full(len(yv), np.nan)
        for tr, te in outer.split(X, split_labels):
            p = make_pipe(clf); p.set_params(**fixed); p.fit(X[tr], yv[tr])
            o[te] = score_of(p, X[te])
        return roc_auc_score(yv, o)
    obs = cv(y if yy is None else yy)
    if not do_perm: return obs, None
    null = np.array([cv(rng_global.permutation(y)) for _ in range(n_perm)])
    return obs, null

mkey, skey, skk = flag_arm.split('__')
obs_f, null_f = pooled_auc_fixed(SETS[skey], MODELS[mkey][0], modal_params(flag_arm), strata_main)
obs_b, null_b = pooled_auc_fixed(BASE_SETS['BASE_depthbatch'], BASE_CLF, {'clf__C': 0.3}, strata_main)
perm = pd.DataFrame({'arm': [flag_arm, 'BASE_depthbatch__logL2__stratLabelBatch'],
    'observed_pooled_auc': [obs_f, obs_b], 'null_mean': [null_f.mean(), null_b.mean()],
    'null_p95': [np.percentile(null_f, 95), np.percentile(null_b, 95)],
    'perm_p': [(1 + np.sum(null_f >= obs_f)) / 201, (1 + np.sum(null_b >= obs_b)) / 201]})
perm.to_csv(f'{OUT}/results/permutation_null_flagship.csv', index=False)
print(perm.to_string(), flush=True)

# ---- LASSO coefficient aggregation (main scheme, per-fold refit at modal C) ----
coef_rows = []
for setname in ['TCR_only', 'TCR_depthbatch', 'TCR_fullcov']:
    armn = f'LASSO_logistic__{setname}__stratLabelBatch'
    X = X_all[SETS[setname]].values
    cmod = pd.Series([b['clf__C'] for b in raw_bps[armn]]).mode()[0]
    outer = StratifiedKFold(5, shuffle=True, random_state=RS)
    for fold, (tr, te) in enumerate(outer.split(X, strata_main)):
        p = make_pipe(LogisticRegression(penalty='l1', solver='liblinear', max_iter=5000, random_state=RS))
        p.set_params(clf__C=cmod); p.fit(X[tr], y[tr])
        kept = p.named_steps['var'].get_support()
        sel_names = np.asarray(SETS[setname])[kept][p.named_steps['scr'].idx_]
        for n, c in zip(sel_names, p.named_steps['clf'].coef_[0]):
            coef_rows.append(dict(set=setname, fold=fold, feature=n, coef=c, nonzero=int(abs(c) > 1e-8)))
coefdf = pd.DataFrame(coef_rows)
agg = (coefdf.groupby(['set', 'feature'])
       .agg(coef_mean=('coef', 'mean'), coef_sd=('coef', 'std'), sel_freq=('nonzero', 'mean'),
            n_folds=('fold', 'nunique')).reset_index())
agg['abs_coef'] = agg.coef_mean.abs()
agg.sort_values(['set', 'sel_freq', 'abs_coef'], ascending=[True, False, False]
                ).to_csv(f'{OUT}/results/lasso_coefficients_by_fold.csv', index=False)
json.dump(dict(flagship=flag_arm, n_samples=int(len(y)), n_pos=int(y.sum()), seed=RS,
               n_jobs=NJ, runtime_min=round((time.time() - t00) / 60, 1)),
          open(f'{OUT}/results/run_meta.json', 'w'), indent=2)
print('ALL DONE', flush=True)
