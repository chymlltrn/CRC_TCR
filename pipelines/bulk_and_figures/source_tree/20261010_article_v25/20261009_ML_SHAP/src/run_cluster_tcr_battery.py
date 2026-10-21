#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
B2 = CLUSTERTCR pN battery v2 — per-TCR features of the 3 frozen P<0.05 clusters
(tcrbert:11, sceptr:5, tcrbert:7), 294,532 features, 289 patients, absent->0 encoding
(present: -log((reads+1)/(total+1)), see build_cluster_tcr_features.py).

Protocol FROZEN from run_ml_battery_merged.py / run_ml_battery_cohort.py:
  same 12 methods, same grids, nested 5x3 CV (StratifiedKFold(5/3, shuffle, rs=42)),
  pipeline = median impute -> VarianceThreshold(1e-12) -> Spearman |rho| top-50
  (fit on the SAME data the frozen pipeline fit it on: inner folds -> inner-train,
  final refit -> outer-train) -> StandardScaler -> classifier; same schemes per scenario.

Compute-necessitated, semantics-preserving re-implementations (documented):
  - SimpleImputer(median) skipped: X has NO NaN by construction (asserted once) -> identity.
  - VarianceThreshold(1e-12) -> vectorized mask X.var(0) > 1e-12 (identical semantics).
  - scipy.spearmanr per feature -> scipy.stats.rankdata(axis=0)+Pearson-on-ranks
    (verified exactly equal: |drho|<=8.3e-17, identical top-50 order incl. ties).
  - The preprocessing (var masks + top-50 selection) is MODEL-INDEPENDENT, so it is
    computed once per (scheme, arm) and reused by all 12 models / grid combos — identical
    to what the frozen GridSearchCV recomputed deterministically at each fit.
  - Grid combos iterated in ParameterGrid (sorted-key) order; tie-break = first max,
    same as GridSearchCV.best_params_.

Contrasts vs the EXISTING frozen battery OOF (same folds, same seed, same patients):
REPO / REPOcov / CLUSTER3 / BASE arms; paired bootstrap 5000 + Holm across 12-model family.

Usage: run_cluster_tcr_battery.py {home|external|merged}
"""
from release_runtime import private_literal as _private_literal, release_path as _release_path
import os, sys, json, time, warnings, itertools
import numpy as np
import pandas as pd
from scipy.stats import rankdata, t as tdist
from joblib import Parallel, delayed
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import roc_auc_score, average_precision_score
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")
RS = 42
SCEN = sys.argv[1]
assert SCEN in ("home", "external", "merged")
MERGED = _release_path('path_6fe8b0cdaf15a381')
PRIOR = f"{_release_path('path_5bd48f922d960395')}{SCEN}" if SCEN != "merged" else MERGED
OUT = _release_path('path_d4455a6432ab939c')
RESD = f"{OUT}/results/{SCEN}"
os.makedirs(RESD, exist_ok=True)
rng_global = np.random.RandomState(RS)

design_full = pd.read_csv(f"{MERGED}/src/merged_design.csv", index_col=0)
design = design_full if SCEN == "merged" else design_full[design_full["cohort_external"] == (0 if SCEN == "home" else 1)]
y = design["y"].astype(int).values
strata_main = design["stratum"].values
feat_sets = json.load(open(f"{MERGED}/src/feature_sets.json"))
COV = ["log1p_reads_z", "log1p_unique_z"] + feat_sets["_batchcohorts"] + ["age", "sex_m", "T_ordinal"]

print("loading X_clusterTCR parquet ...", flush=True)
t0 = time.time()
XB2 = pd.read_parquet(f"{OUT}/src/X_clusterTCR_289.parquet").loc[design.index]
Xcov = pd.read_csv(f"{MERGED}/src/X_merged_all.csv", index_col=0, usecols=[design.index.name] + COV).loc[design.index]
X_all = pd.concat([XB2, Xcov], axis=1)
assert X_all.index.equals(design.index)
B2FEATS = list(XB2.columns)
assert not np.isnan(X_all[B2FEATS].values).any(), "B2 block must be NaN-free"
del XB2
SETS = {"CLUSTERTCR": B2FEATS, "CLUSTERTCRcov": B2FEATS + COV}
print("loaded", X_all.shape, round(time.time() - t0, 1), "s", flush=True)

MODELS = {
 "LASSO_logistic": (LogisticRegression(penalty="l1", solver="liblinear", max_iter=5000, random_state=RS),
                    {"clf__C": [0.01, 0.03, 0.1, 0.3, 1, 3]}),
 "Ridge_logistic": (LogisticRegression(penalty="l2", solver="liblinear", max_iter=5000, random_state=RS),
                    {"clf__C": [0.01, 0.03, 0.1, 0.3, 1, 3]}),
 "ElasticNet_logistic": (LogisticRegression(penalty="elasticnet", solver="saga", max_iter=8000, random_state=RS),
                    {"clf__C": [0.03, 0.1, 0.3, 1], "clf__l1_ratio": [0.2, 0.5, 0.8]}),
 "Logistic_unpenalized": (LogisticRegression(penalty=None, solver="lbfgs", max_iter=8000, random_state=RS), {}),
 "RandomForest": (RandomForestClassifier(n_estimators=500, random_state=RS, n_jobs=2),
                    {"clf__max_depth": [None, 4, 8], "clf__min_samples_leaf": [2, 5]}),
 "HistGradBoost": (HistGradientBoostingClassifier(max_leaf_nodes=15, min_samples_leaf=10,
                    l2_regularization=1.0, random_state=RS),
                    {"clf__max_iter": [100, 300], "clf__learning_rate": [0.05, 0.1]}),
 "XGBoost": (XGBClassifier(n_jobs=2, random_state=RS, eval_metric="logloss",
                           subsample=0.8, colsample_bytree=0.8, verbosity=0),
             {"clf__n_estimators": [100, 300], "clf__max_depth": [2, 3], "clf__learning_rate": [0.05, 0.1]}),
 "SVM_linear": (SVC(kernel="linear", probability=False, random_state=RS), {"clf__C": [0.03, 0.3, 3, 30]}),
 "SVM_RBF": (SVC(kernel="rbf", probability=False, random_state=RS),
             {"clf__C": [0.3, 1, 3, 10], "clf__gamma": ["scale", 0.01]}),
 "kNN": (KNeighborsClassifier(), {"clf__n_neighbors": [5, 9, 15, 25], "clf__weights": ["uniform", "distance"]}),
 "GaussianNB": (GaussianNB(), {"clf__var_smoothing": [1e-9, 1e-7, 1e-5]}),
 "MLP": (MLPClassifier(early_stopping=True, max_iter=3000, random_state=RS),
         {"clf__alpha": [1e-3, 1e-2, 1e-1], "clf__hidden_layer_sizes": [(32, 16), (64, 32)]}),
}
SCHEMES = {"stratLabelBatchCohort": strata_main}
if SCEN != "external":
    SCHEMES["stratLabel"] = y

def spearman_abs(X, yv):
    Xr = rankdata(X, axis=0)
    Xc = Xr - Xr.mean(axis=0)
    sd = Xr.std(axis=0, ddof=1)
    yc = yv - yv.mean()
    den = (X.shape[0] - 1) * sd * yv.std(ddof=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        rs = np.where((sd > 0) & (yv.std(ddof=1) > 0), (Xc.T @ yc) / np.where(den == 0, np.nan, den), 0.0)
    return np.abs(np.nan_to_num(rs, nan=0.0))

def score_of(est, Z):
    return est.predict_proba(Z)[:, 1] if hasattr(est, "predict_proba") else est.decision_function(Z)

def build_prep(feats, split_labels):
    """Per (scheme, arm): all fold preprocessing, model-independent."""
    X = X_all[feats].values.astype(np.float64)
    names = np.array(feats, dtype=object)
    outer = StratifiedKFold(5, shuffle=True, random_state=RS)
    inner = StratifiedKFold(3, shuffle=True, random_state=RS)
    folds = []
    for tr, te in outer.split(X, split_labels):
        fd = {"tr": tr, "te": te}
        # inner folds: impute(train-fold median) + var + scr fitted on inner-train (frozen semantics)
        inners = []
        Xtr = X[tr]
        for itr, ival in inner.split(Xtr, y[tr]):
            Xi = Xtr[itr]
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                med = np.nanmedian(Xi, axis=0)  # train-fold medians (all-NaN col -> NaN, stays NaN, dropped by var)
            Xi = np.where(np.isnan(Xi), med, Xi)
            m = Xi.var(axis=0) > 1e-12
            Xiv = Xi[:, m]
            idx = np.argsort(-spearman_abs(Xiv, y[tr][itr]))[:50]
            sel = np.flatnonzero(m)[idx]
            Zitr = Xiv[:, idx]
            Xiva = np.where(np.isnan(Xtr[ival]), med, Xtr[ival])
            Ziva = Xiva[:, sel]
            inners.append((Zitr, y[tr][itr], Ziva, y[tr][ival]))
        fd["inners"] = inners
        # final-refit: impute + var + scr on outer-train
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            medo = np.nanmedian(Xtr, axis=0)
        Xtro = np.where(np.isnan(Xtr), medo, Xtr)
        mo = Xtro.var(axis=0) > 1e-12
        Xov = Xtro[:, mo]
        ido = np.argsort(-spearman_abs(Xov, y[tr]))[:50]
        selo = np.flatnonzero(mo)[ido]
        fd["Ztr"] = Xov[:, ido]
        fd["ytr"] = y[tr]
        fd["Xte_sel"] = np.where(np.isnan(X[te]), medo, X[te])[:, selo]
        fd["yte"] = y[te]
        fd["names_full"] = [str(v) for v in names[selo]]
        folds.append(fd)
    return folds

def clone_with(clf, params):
    from sklearn.base import clone
    c = clone(clf)
    c.set_params(**{k.replace("clf__", ""): v for k, v in params.items()})
    return c

def make_small(clf):
    # frozen make_pipe tail: scaler -> clf (imputer=identity [no NaN, asserted], post-scr
    # VarianceThreshold=no-op [all top-50 features have train-fold variance>0])
    return Pipeline([("scl", StandardScaler()), ("clf", clf)])

def run_model_on_prep(mname, clf, grid, folds):
    keys = sorted(grid.keys()) if grid else []
    combos = [dict(zip(keys, v)) for v in itertools.product(*[grid[k] for k in keys])] if grid else [{}]
    oof = np.full(len(y), np.nan)
    folds_auc, folds_ap, bps, sel_names = [], [], [], []
    for fd in folds:
        scores = np.zeros(len(combos))
        for Zitr, yitr, Ziva, yiva in fd["inners"]:
            for gi, gp in enumerate(combos):
                p = make_small(clone_with(clf, gp))
                p.fit(Zitr, yitr)
                scores[gi] += roc_auc_score(yiva, score_of(p, Ziva))
        best = combos[int(np.argmax(scores))]
        p = make_small(clone_with(clf, best))
        p.fit(fd["Ztr"], fd["ytr"])
        oof[fd["te"]] = score_of(p, fd["Xte_sel"])
        folds_auc.append(roc_auc_score(fd["yte"], oof[fd["te"]]))
        folds_ap.append(average_precision_score(fd["yte"], oof[fd["te"]]))
        bps.append(dict(best))
        sel_names.append(fd["names_full"])
    return mname, oof, folds_auc, folds_ap, bps, sel_names

def holm(ps):
    ps = np.asarray(ps); order = np.argsort(ps); m = len(ps)
    q = np.empty(m); running = 0.0
    for r, oi in enumerate(order):
        v = (m - r) * ps[oi]; running = max(running, v); q[oi] = min(1.0, running)
    return q

def pooled_ci(oof, n=2000):
    bs = []
    for _ in range(n):
        i = rng_global.randint(0, len(y), len(y))
        if len(np.unique(y[i])) < 2:
            continue
        bs.append(roc_auc_score(y[i], oof[i]))
    lo, hi = np.percentile(bs, [2.5, 97.5])
    return lo, hi

def paired_boot(mo, bo, n=5000):
    d = []
    for _ in range(n):
        i = rng_global.randint(0, len(y), len(y))
        if len(np.unique(y[i])) < 2:
            continue
        d.append(roc_auc_score(y[i], mo[i]) - roc_auc_score(y[i], bo[i]))
    return np.array(d)

# ---------- 1) preprocessing (parallel over blocks) ----------
blocks = [(sk, sn) for sk in SCHEMES for sn in SETS]
t00 = time.time()
print(f"preprocessing {len(blocks)} blocks (scheme x arm) ...", flush=True)
preps = Parallel(n_jobs=len(blocks))(
    delayed(build_prep)(SETS[sn], SCHEMES[sk]) for sk, sn in blocks)
prep = {(sk, sn): p for (sk, sn), p in zip(blocks, preps)}
print("prep done", round((time.time() - t00) / 60, 1), "min", flush=True)

# ---------- 2) battery (models in parallel per block) ----------
rows, oofs, sel_log = [], {}, {}
for sk, sn in blocks:
    folds = prep[(sk, sn)]
    results = Parallel(n_jobs=6)(
        delayed(run_model_on_prep)(mn, clf, grid, folds) for mn, (clf, grid) in MODELS.items())
    for mn, oof, fa, fp, bps, seln in results:
        arm = f"{mn}__{sn}__{sk}"
        auc = np.array(fa); ap = np.array(fp)
        pooled = roc_auc_score(y, oof); pooled_ap = average_precision_score(y, oof)
        lo, hi = pooled_ci(oof)
        mean, sd = auc.mean(), auc.std(ddof=1)
        tc = tdist.ppf(0.975, len(auc) - 1)
        rows.append(dict(arm=arm, auc_fold_mean=mean, auc_fold_sd=sd,
                         auc_fold_ci_lo=mean - tc * sd / np.sqrt(len(auc)),
                         auc_fold_ci_hi=mean + tc * sd / np.sqrt(len(auc)),
                         auc_pooled_oof=pooled, auc_pooled_ci95_lo=lo, auc_pooled_ci95_hi=hi,
                         prauc_pooled=pooled_ap,
                         fold_aucs=";".join(f"{a:.4f}" for a in auc),
                         best_params=" || ".join(json.dumps({k: str(v) for k, v in b.items()}) for b in bps)))
        oofs[arm] = oof
        sel_log[arm] = [{"fold": k, "features": seln[k],
                         "best_params": {kk: str(vv) for kk, vv in bps[k].items()}} for k in range(5)]
        print(f"  {arm}: pooled={pooled:.3f}", flush=True)
    pd.DataFrame(rows).to_csv(f"{RESD}/_partial_rows.csv", index=False)
res = pd.DataFrame(rows).set_index("arm")
res.reset_index().to_csv(f"{RESD}/perf_clusterTCR.csv", index=False)
pd.DataFrame(oofs, index=design.index).to_csv(f"{RESD}/oof_clusterTCR.csv")
json.dump(sel_log, open(f"{RESD}/fold_selection_clusterTCR.json", "w"))

# ---------- 3) contrasts vs prior frozen battery ----------
prior_oof = pd.read_csv(f"{PRIOR}/results/oof_predictions_all_arms.csv", index_col=0).loc[design.index]
prior_perf = pd.read_csv(f"{PRIOR}/results/model_performance_full.csv").set_index("arm")
contrast_rows = []
FAMS = [("CLUSTERTCR", "REPO", "CLUSTERTCR_vs_REPO"),
        ("CLUSTERTCRcov", "REPOcov", "CLUSTERTCRcov_vs_REPOcov"),
        ("CLUSTERTCR", "CLUSTER3", "CLUSTERTCR_vs_CLUSTER3"),
        ("CLUSTERTCRcov", "BASE_Tstage", "CLUSTERTCRcov_vs_BASE_Tstage"),
        ("CLUSTERTCR", "BASE_depthcohbatch", "CLUSTERTCR_vs_BASE_depthcohbatch")]
for sk in SCHEMES:
    for a_, b_, tag in FAMS:
        ps = []
        for mname in MODELS:
            a = f"{mname}__{a_}__{sk}"
            b = f"{b_}__logL2__{sk}" if b_.startswith("BASE") else f"{mname}__{b_}__{sk}"
            if a not in oofs or b not in prior_oof.columns:
                continue
            diffs = paired_boot(oofs[a], prior_oof[b].values)
            contrast_rows.append(dict(arm=f"{mname}__{a_}", scheme=sk, contrast=tag,
                delta_auc_pooled=res.loc[a, "auc_pooled_oof"] - prior_perf.loc[b, "auc_pooled_oof"],
                delta_boot_mean=diffs.mean(),
                delta_ci95_lo=np.percentile(diffs, 2.5), delta_ci95_hi=np.percentile(diffs, 97.5),
                p_paired_boot=(1 + np.sum(diffs <= 0)) / (1 + len(diffs))))
            ps.append(contrast_rows[-1]["p_paired_boot"])
        for rr, qq in zip([c for c in contrast_rows if c["scheme"] == sk and c["contrast"] == tag], holm(ps)):
            rr["p_holm"] = qq
pd.DataFrame(contrast_rows).to_csv(f"{RESD}/contrasts_clusterTCR.csv", index=False)
json.dump(dict(scenario=SCEN, n_nested_arms=len(rows), n_samples=int(len(y)), n_pos=int(y.sum()),
               seed=RS, n_features_B2=len(B2FEATS), runtime_min=round((time.time() - t00) / 60, 1)),
          open(f"{RESD}/run_meta.json", "w"), indent=2)
print(f"BATTERY {SCEN} DONE in {(time.time()-t00)/60:.1f} min", flush=True)
