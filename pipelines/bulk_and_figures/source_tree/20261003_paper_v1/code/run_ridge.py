#!/usr/bin/env python3
"""Patient-level, train-only nested ridge evaluation for the frozen paper plan.

The principal prediction is the mean of three independently nested OOF
probabilities per patient. Bootstrap multiplicities are expanded rows, while
the original patient_key remains the grouping identity in BOTH CV levels.
No external labels or clinical fields enter Home model selection.
"""
import os
for _name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_name, "1")

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path
import sys
import time
import traceback
import warnings

import numpy as np
import pandas as pd
import joblib
from scipy.special import logit
import statsmodels.api as sm
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score
from sklearn.model_selection import GridSearchCV, StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

SEEDS = (17, 29, 43)
CS = (0.01, 0.1, 1.0, 10.0)
SUMMARY = ("normalized_shannon", "top10_fraction", "weighted_mean_cdr3_length")
TECH = ("log1p_reads", "log1p_unique_aa")
PROHIBITED = {"n_binary", "n_raw", "N", "T", "M", "stage", "tnm", "recorded_t", "pt", "ypn", "pn"}


def json_write(path, obj):
    Path(path).write_text(json.dumps(obj, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")


def finite_dict(obj):
    if isinstance(obj, dict):
        return {str(k): finite_dict(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [finite_dict(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (float, np.floating)):
        return float(obj) if np.isfinite(obj) else None
    return obj


def feature_blocks(frame):
    """Only the explicitly frozen feature names are admitted to the models."""
    vj = sorted(c for c in frame if c.startswith(("trbv_clr__", "trbj_clr__")))
    required = ["patient_key", "cohort", "n_binary", "age_years", "sex", "batch", *TECH, *SUMMARY]
    absent = [c for c in required if c not in frame]
    if absent:
        raise ValueError(f"Missing required feature columns: {absent}")
    if not vj:
        raise ValueError("No frozen V/J CLR features were supplied")
    tcr = [*SUMMARY, *vj]
    blocks = {
        "M0": {"numeric": ["age_years"], "categorical": ["sex"]},
        "M1": {"numeric": ["age_years", *TECH], "categorical": ["sex", "batch"]},
        "M2": {"numeric": tcr, "categorical": []},
        "M3": {"numeric": ["age_years", *TECH, *tcr], "categorical": ["sex", "batch"]},
        "M1_portable": {"numeric": list(TECH), "categorical": []},
        "M3_portable": {"numeric": [*TECH, *tcr], "categorical": []},
        "M1_lobo": {"numeric": ["age_years", *TECH], "categorical": ["sex"]},
        "M3_lobo": {"numeric": ["age_years", *TECH, *tcr], "categorical": ["sex"]},
    }
    for b in blocks.values():
        assert not set(b["numeric"] + b["categorical"]) & PROHIBITED
    return blocks


def make_pipeline(numeric, categorical, sceptr=None):
    transforms = []
    if numeric:
        transforms.append(("numeric", Pipeline([
            ("impute", SimpleImputer(strategy="median", keep_empty_features=True)),
            ("scale", StandardScaler()),
        ]), numeric))
    if categorical:
        transforms.append(("categorical", Pipeline([
            ("impute", SimpleImputer(strategy="constant", fill_value="Missing", keep_empty_features=True)),
            ("encode", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]), categorical))
    if sceptr:
        transforms.append(("sceptr", Pipeline([
            ("impute", SimpleImputer(strategy="median", keep_empty_features=True)),
            ("scale", StandardScaler()),
            ("pca", PCA(n_components=32, svd_solver="full")),
        ]), sceptr))
    return Pipeline([
        ("preprocess", ColumnTransformer(transforms, remainder="drop", sparse_threshold=0)),
        ("ridge", LogisticRegression(penalty="l2", solver="lbfgs", max_iter=2000, tol=1e-8)),
    ])


def splits(frame, folds, seed, return_seed=False):
    y = frame.n_binary.to_numpy(dtype=int)
    groups = frame.patient_key.astype(str).to_numpy()
    for retry in range(5):
        partition_seed = seed+7919*retry
        cv = StratifiedGroupKFold(n_splits=folds, shuffle=True, random_state=partition_seed)
        out = list(cv.split(frame,y,groups))
        seen = np.zeros(len(frame),dtype=int); feasible = True
        for train,test in out:
            assert not set(groups[train])&set(groups[test]), "Patient leakage in grouped CV"
            feasible &= len(np.unique(y[train]))==2 and len(np.unique(y[test]))==2
            seen[test] += 1
        assert np.all(seen==1), "Every row must have exactly one held-out prediction"
        if feasible:
            return (out,partition_seed) if return_seed else out
    raise ValueError("CV fold class feasibility failed in five deterministic partitions; no model-performance selection")


def tuned_fit(train, block, seed=17, inner_folds=3):
    inner = splits(train, inner_folds, seed)
    search = GridSearchCV(
        make_pipeline(block["numeric"], block["categorical"], block.get("sceptr")),
        {"ridge__C": list(CS)}, scoring="neg_log_loss", cv=inner,
        n_jobs=1, refit=True, error_score="raise", return_train_score=False,
    )
    search.fit(train, train.n_binary.astype(int))
    if np.max(search.best_estimator_.named_steps["ridge"].n_iter_) >= 2000:
        raise RuntimeError("Selected ridge fit did not converge within the fixed iteration limit")
    return search


def nested_oof(frame, blocks, models=("M0", "M1", "M2", "M3"), seeds=SEEDS,
               outer_folds=5, inner_folds=3):
    """Return long OOF predictions and explicit group/fold provenance."""
    frame = frame.reset_index(drop=True).copy()
    preds, audit = [], []
    for seed in seeds:
        outer, outer_partition_seed = splits(frame,outer_folds,seed,return_seed=True)
        for fold, (train_idx, test_idx) in enumerate(outer):
            train, test = frame.iloc[train_idx], frame.iloc[test_idx]
            inner_requested_seed = seed+1000+fold
            inner,inner_partition_seed = splits(train,inner_folds,inner_requested_seed,return_seed=True)
            for inner_fold, (itr, ite) in enumerate(inner):
                audit.append({
                    "level": "inner", "seed": seed, "outer_fold": fold, "inner_fold": inner_fold,
                    "requested_partition_seed":inner_requested_seed,"actual_partition_seed":inner_partition_seed,
                    "partition_retry_count":(inner_partition_seed-inner_requested_seed)//7919,
                    "partition_retry_reason":"missing_outcome_class_before_fit" if inner_partition_seed!=inner_requested_seed else "none",
                    "train_keys": "|".join(sorted(set(train.iloc[itr].patient_key.astype(str)))),
                    "test_keys": "|".join(sorted(set(train.iloc[ite].patient_key.astype(str)))),
                    "patient_overlap": 0, "train_rows": len(itr), "test_rows": len(ite),
                })
            audit.append({
                "level": "outer", "seed": seed, "outer_fold": fold, "inner_fold": -1,
                "requested_partition_seed":seed,"actual_partition_seed":outer_partition_seed,
                "partition_retry_count":(outer_partition_seed-seed)//7919,
                "partition_retry_reason":"missing_outcome_class_before_fit" if outer_partition_seed!=seed else "none",
                "train_keys": "|".join(sorted(set(train.patient_key.astype(str)))),
                "test_keys": "|".join(sorted(set(test.patient_key.astype(str)))),
                "patient_overlap": 0, "train_rows": len(train), "test_rows": len(test),
            })
            for model in models:
                fit = tuned_fit(train, blocks[model], seed + 1000 + fold, inner_folds)
                prob = fit.predict_proba(test)[:, 1]
                for row_idx, p in zip(test_idx, prob):
                    row = frame.iloc[row_idx]
                    preds.append({
                        "row_id": int(row_idx), "patient_key": str(row.patient_key),
                        "n_binary": int(row.n_binary), "batch": str(row.batch),
                        "seed": seed, "fold": fold, "model": model,
                        "probability": float(p), "selected_C": float(fit.best_params_["ridge__C"]),
                        "inner_log_loss": float(-fit.best_score_),
                    })
    return pd.DataFrame(preds), pd.DataFrame(audit)


def ensemble_predictions(oof):
    return oof.groupby(["row_id", "patient_key", "n_binary", "batch", "model"], as_index=False).probability.mean()


def metrics(y, prob):
    y, prob = np.asarray(y, dtype=int), np.asarray(prob, dtype=float)
    p = np.clip(prob, 1e-8, 1 - 1e-8)
    result = {
        "n": len(y), "positive": int(y.sum()), "AUROC": roc_auc_score(y, p),
        "AUPRC": average_precision_score(y, p), "log_loss": log_loss(y, p, labels=[0, 1]),
        "Brier": brier_score_loss(y, p), "calibration_intercept": np.nan,
        "calibration_slope": np.nan, "calibration_status": "not_estimable",
    }
    try:
        if np.ptp(logit(p)) < 1e-10:
            result["calibration_status"] = "constant_prediction"
            return result
        with warnings.catch_warnings(record=True) as records:
            warnings.simplefilter("always")
            cal = sm.GLM(y, sm.add_constant(logit(p)), family=sm.families.Binomial()).fit()
        separated = any("separation" in str(w.message).lower() for w in records)
        if cal.converged and np.isfinite(cal.params).all() and not separated:
            result["calibration_intercept"], result["calibration_slope"] = cal.params
            result["calibration_status"] = "ok"
        else:
            result["calibration_status"] = "separation_or_nonconvergence"
    except Exception as exc:
        result["calibration_status"] = f"fit_failed:{type(exc).__name__}"
    return result


def paired_metrics(y, base, new):
    b, n = metrics(y, base), metrics(y, new)
    return {"delta_log_loss": b["log_loss"] - n["log_loss"],
            "delta_AUROC": n["AUROC"] - b["AUROC"],
            "delta_AUPRC": n["AUPRC"] - b["AUPRC"],
            "delta_Brier": b["Brier"] - n["Brier"]}


def paired_array(ensemble, base="M1", new="M3"):
    a = ensemble[ensemble.model == base].set_index("row_id")
    b = ensemble[ensemble.model == new].set_index("row_id")
    assert a.index.is_unique and b.index.is_unique
    b = b.loc[a.index]
    assert (a.patient_key == b.patient_key).all() and (a.n_binary == b.n_binary).all()
    return a.n_binary.to_numpy(int), a.probability.to_numpy(), b.probability.to_numpy()


def conditional_ci(y, base, new, repetitions=2000, seed=62031):
    rng = np.random.default_rng(seed)
    idx0, idx1 = np.flatnonzero(np.asarray(y) == 0), np.flatnonzero(np.asarray(y) == 1)
    values = []
    for _ in range(repetitions):
        ix = np.r_[rng.choice(idx0, len(idx0), replace=True), rng.choice(idx1, len(idx1), replace=True)]
        yy, bb, nn = np.asarray(y)[ix], np.asarray(base)[ix], np.asarray(new)[ix]
        # Calibration is not needed in this loop and can be unstable in small resamples.
        values.append({
            "delta_log_loss": log_loss(yy, bb) - log_loss(yy, nn),
            "delta_AUROC": roc_auc_score(yy, nn) - roc_auc_score(yy, bb),
            "delta_AUPRC": average_precision_score(yy, nn) - average_precision_score(yy, bb),
            "delta_Brier": brier_score_loss(yy, bb) - brier_score_loss(yy, nn),
        })
    vals = pd.DataFrame(values)
    return [{"metric": col, "ci_low": vals[col].quantile(.025), "ci_high": vals[col].quantile(.975),
             "ci_type": "conditional_fixed_OOF_patient_stratified_percentile", "repetitions": repetitions}
            for col in vals]


def conditional_metrics_ci(y, prob, cohort, model, prediction, repetitions=2000, seed=62031):
    y, prob = np.asarray(y,dtype=int), np.asarray(prob,dtype=float)
    point = metrics(y,prob); rng = np.random.default_rng(seed)
    indices = [np.flatnonzero(y==outcome) for outcome in (0,1)]
    resampled = []
    for _ in range(repetitions):
        ix = np.concatenate([rng.choice(group,len(group),replace=True) for group in indices])
        resampled.append(metrics(y[ix],prob[ix]))
    values = pd.DataFrame(resampled)
    rows = []
    for metric in ("AUROC","AUPRC","log_loss","Brier","calibration_intercept","calibration_slope"):
        valid = pd.to_numeric(values[metric],errors="coerce").dropna()
        reason = "" if len(valid)==repetitions else "calibration separation, constant prediction, or failed fit in some resamples"
        rows.append({"cohort":cohort,"model":model,"prediction":prediction,"metric":metric,
                     "estimate":point[metric], "ci_low":valid.quantile(.025) if len(valid) else np.nan,
                     "ci_high":valid.quantile(.975) if len(valid) else np.nan,
                     "requested_repetitions":repetitions,"valid_repetitions":len(valid),"n":len(y),
                     "ci_type":"conditional_fixed_prediction_patient_stratified_percentile",
                     "status":"ok" if len(valid) else "not_estimable",
                     "NA_reason":reason or (point["calibration_status"] if not np.isfinite(point[metric]) else "")})
    return rows


def bootstrap_sample(frame, seed):
    rng = np.random.default_rng(seed)
    chosen = []
    for outcome in (0, 1):
        positions = np.flatnonzero(frame.n_binary.to_numpy(dtype=int) == outcome)
        chosen.extend(rng.choice(positions, len(positions), replace=True))
    rng.shuffle(chosen)
    out = frame.iloc[chosen].reset_index(drop=True).copy()
    # Each duplicate draw has weight one; multiplicity is its bootstrap frequency.
    # Original identity is NEVER replaced by a draw index for CV grouping.
    assert out.groupby("patient_key").n_binary.nunique().max() == 1
    return out


def bootstrap_worker(frame, blocks, repetition, seeds, outer_folds, inner_folds, audit_output=None):
    start = time.time()
    try:
        sample = bootstrap_sample(frame, 100000 + repetition)
        oof, audit = nested_oof(sample, blocks, ("M1", "M3"), seeds, outer_folds, inner_folds)
        if audit_output is not None:
            audit.to_csv(audit_output,sep="\t",index=False)
        y, b, n = paired_array(ensemble_predictions(oof))
        return {"repetition": repetition, "status": "success", "n_rows": len(sample),
                "n_unique_patients": sample.patient_key.nunique(),
                "patient_overlap": int(audit.patient_overlap.max()),
                "max_partition_retry_count": int(audit.partition_retry_count.max()),
                "elapsed_seconds": time.time() - start, **paired_metrics(y, b, n)}
    except Exception as exc:
        return {"repetition": repetition, "status": "failure", "error": repr(exc),
                "traceback": traceback.format_exc(), "elapsed_seconds": time.time() - start}


def negative_worker(frame, blocks, repetition, seeds, outer_folds, inner_folds):
    shuffled = frame.copy()
    rng = np.random.default_rng(300000 + repetition)
    # Shuffle within acquisition batch: preserve its prevalence while destroying
    # patient-level feature/outcome coupling. This is a diagnostic, not a main p value.
    for _, ix in shuffled.groupby("batch").groups.items():
        shuffled.loc[ix, "n_binary"] = rng.permutation(shuffled.loc[ix, "n_binary"].to_numpy())
    try:
        pred, audit = nested_oof(shuffled, blocks, ("M1", "M3"), seeds, outer_folds, inner_folds)
        y, b, n = paired_array(ensemble_predictions(pred))
        return {"repetition": repetition, "status": "success", **paired_metrics(y, b, n),
                "M3_AUROC": roc_auc_score(y, n), "patient_overlap": int(audit.patient_overlap.max())}
    except Exception as exc:
        return {"repetition": repetition, "status": "failure", "error": repr(exc)}


def portable_evaluation(home, external, blocks, output=None):
    rows, fits = [], []
    for model in ("M1_portable", "M3_portable"):
        fit = tuned_fit(home, blocks[model], 17)
        if output is not None:
            joblib.dump(fit.best_estimator_, Path(output)/f"frozen_{model}.joblib")
        p = fit.predict_proba(external)[:, 1]
        for key, y, prob in zip(external.patient_key, external.n_binary, p):
            rows.append({"patient_key": key, "n_binary": int(y), "model": model, "probability": prob})
        fits.append({"model": model, "selected_C": fit.best_params_["ridge__C"],
                     "home_inner_log_loss": -fit.best_score_, **metrics(external.n_binary, p)})
    return pd.DataFrame(rows), pd.DataFrame(fits)


def lobo_evaluation(home, blocks):
    predictions, summary = [], []
    for batch in sorted(home.batch.astype(str).unique()):
        test, train = home[home.batch == batch], home[home.batch != batch]
        for model in ("M1_lobo", "M3_lobo"):
            # Acquisition batch is deliberately omitted: its unseen offset cannot
            # be estimated in a held-out batch. Both comparator models use the same
            # clinical/technical columns and patient partition.
            fit = tuned_fit(train, blocks[model], 17)
            prob = fit.predict_proba(test)[:, 1]
            summary.append({"held_out_batch": batch, "model": model,
                            "selected_C": fit.best_params_["ridge__C"], **metrics(test.n_binary, prob)})
            for key, y, p in zip(test.patient_key, test.n_binary, prob):
                predictions.append({"patient_key": key, "n_binary": int(y), "batch": batch,
                                    "model": model, "probability": p})
    return pd.DataFrame(predictions), pd.DataFrame(summary)


def load_sceptr(frame, mean_path, index_path):
    means = np.load(mean_path, mmap_mode="r")
    index = pd.read_csv(index_path, sep="\t")
    if "row_index" not in index:
        index["row_index"] = np.arange(len(index))
    assert index.patient_key.is_unique and means.shape[0] == len(index)
    lookup = index.set_index("patient_key").row_index
    if not set(frame.patient_key) <= set(lookup.index):
        raise ValueError("Missing SCEPTR means for a modeled patient")
    columns = [f"sceptr_{i:03d}" for i in range(means.shape[1])]
    arr = np.asarray(means[lookup.loc[frame.patient_key].to_numpy(int)], dtype=float)
    assert np.isfinite(arr).all()
    frame = frame.copy()
    frame[columns] = arr
    return frame, columns


def pooled_home_frame(home, path, blocks):
    pooled = pd.read_csv(path, sep="\t")
    assert pooled.patient_key.is_unique
    pooled = pooled.set_index("patient_key").loc[home.patient_key].reset_index()
    assert np.array_equal(pooled.n_binary.to_numpy(), home.n_binary.to_numpy()), "Pooling must not relabel patients"
    columns = sorted(set(blocks["M1"]["numeric"]+blocks["M3"]["numeric"]))
    changed = ~np.isclose(home[columns].to_numpy(float), pooled[columns].to_numpy(float), rtol=1e-10, atol=1e-12, equal_nan=True).all(axis=1)
    assert int(changed.sum()) <= 1, "Pooling sensitivity may replace only the duplicate-patient representation"
    # Retain the primary patient's metadata and order; replace only numeric repertoire/technical fields.
    result = home.copy()
    result[columns] = pooled[columns].to_numpy()
    return result, home.loc[changed,"patient_key"].astype(str).tolist()


def run_sceptr_sensitivity(home, blocks, out, mean_path, index_path):
    json_write(out/"sceptr_status.json",{"state":"running","n":len(home)})
    embedded, cols = load_sceptr(home,mean_path,index_path)
    blocks = dict(blocks)
    blocks["M1_sceptr"] = {**blocks["M1"],"sceptr":cols}
    soof,saudit = nested_oof(embedded,blocks,("M1_sceptr",))
    soof.to_csv(out/"sceptr_oof_predictions.tsv",sep="\t",index=False)
    saudit.to_csv(out/"sceptr_fold_audit.tsv",sep="\t",index=False)
    sens = ensemble_predictions(soof)
    pd.DataFrame([{**metrics(sens.n_binary,sens.probability),"model":"M1_sceptr"}]).to_csv(out/"sceptr_metrics.tsv",sep="\t",index=False)
    main = pd.read_csv(out/"oof_patient_ensemble.tsv",sep="\t")
    combo = pd.concat([main[main.model=="M1"],sens],ignore_index=True)
    sy,sb,sn = paired_array(combo,"M1","M1_sceptr")
    json_write(out/"sceptr_paired_metrics.json",finite_dict(paired_metrics(sy,sb,sn)))
    pd.DataFrame(conditional_ci(sy,sb,sn)).to_csv(out/"sceptr_conditional_paired_ci.tsv",sep="\t",index=False)
    existing = pd.read_csv(out/"metrics_ci.tsv",sep="\t")
    existing = existing[existing.model!="M1_sceptr"]
    extra = pd.DataFrame(conditional_metrics_ci(sens.n_binary,sens.probability,"Home","M1_sceptr","three_seed_mean"))
    pd.concat([existing,extra],ignore_index=True).to_csv(out/"metrics_ci.tsv",sep="\t",index=False)
    json_write(out/"sceptr_status.json",{
        "state":"complete","n":len(home),"patient_overlap":int(saudit.patient_overlap.max()),
        "input_dimensions":len(cols),"fold_PCA_dimensions":32,"seeds":SEEDS,
        "blocks":blocks["M1_sceptr"],"mean_sha256":hashlib.sha256(Path(mean_path).read_bytes()).hexdigest(),
        "index_sha256":hashlib.sha256(Path(index_path).read_bytes()).hexdigest(),
        "bootstrap":"conditional fixed OOF only; main fullpipeline bootstrap not repeated"})


def repair_bootstrap_failures(home, blocks, out):
    table = pd.read_csv(out/"full_pipeline_bootstrap.tsv",sep="\t")
    initial_failures = table[table.status!="success"].copy()
    receipt_path = out/"bootstrap_initial_failures.tsv"
    if len(initial_failures):
        previous = pd.read_csv(receipt_path,sep="\t") if receipt_path.exists() else pd.DataFrame()
        pd.concat([previous,initial_failures],ignore_index=True).drop_duplicates("repetition").to_csv(receipt_path,sep="\t",index=False)
        repaired = []
        for repetition in initial_failures.repetition.astype(int):
            value = bootstrap_worker(home,blocks,repetition,SEEDS,5,3,out/f"bootstrap_repair_{repetition}_fold_audit.tsv")
            value["initial_attempt_failed"] = True
            repaired.append(value)
            print(f"Bootstrap class-feasibility repair {repetition}: {value['status']}",flush=True)
        table = pd.concat([table[table.status=="success"],pd.DataFrame(repaired)],ignore_index=True).sort_values("repetition")
        table.to_csv(out/"full_pipeline_bootstrap.tsv",sep="\t",index=False)
    success = table[table.status=="success"]
    rows = [{"metric":metric,"ci_low":success[metric].quantile(.025),"ci_high":success[metric].quantile(.975),
             "repetitions":len(success),"ci_type":"full_pipeline_patient_stratified_percentile","baseline":"M1","new":"M3"}
            for metric in ("delta_log_loss","delta_AUROC","delta_AUPRC","delta_Brier")]
    pd.DataFrame(rows).to_csv(out/"full_pipeline_paired_ci.tsv",sep="\t",index=False)
    status = json.loads((out/"status.json").read_text())
    preserved_failures = pd.read_csv(receipt_path,sep="\t") if receipt_path.exists() else pd.DataFrame()
    repaired_count = int(table.get("initial_attempt_failed",pd.Series(False,index=table.index)).fillna(False).astype(bool).sum())
    status.update({"state":"complete" if len(success)==len(table) else "completed_with_bootstrap_failures",
                   "bootstrap_success":len(success),"bootstrap_failed":len(table)-len(success),
                   "bootstrap_initial_failures":len(preserved_failures),
                   "bootstrap_class_feasibility_repaired":repaired_count})
    if (out/"sceptr_status.json").exists() and json.loads((out/"sceptr_status.json").read_text()).get("state")=="complete":
        status["sceptr_status"]="complete"
    json_write(out/"status.json",status)
    spec = json.loads((out/"model_spec.json").read_text())
    spec["partition_feasibility"] = "At either CV level, require both outcome classes in every train/test fold; only if infeasible retry seed+7919*j for j=1..4 before any fitting. Initial failures remain recorded; no performance-based partition selection."
    json_write(out/"model_spec.json",spec)
    print(json.dumps(status),flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--bootstrap", type=int, default=200)
    parser.add_argument("--jobs", type=int, default=10)
    parser.add_argument("--negative-controls", type=int, default=20)
    parser.add_argument("--sceptr-means")
    parser.add_argument("--sceptr-index")
    parser.add_argument("--pooled-features")
    parser.add_argument("--sceptr-only",action="store_true",help="Run delayed sensitivity against existing frozen main OOF without rerunning its bootstrap")
    parser.add_argument("--repair-bootstrap-failures",action="store_true",help="Repair recorded failed draws under deterministic class-feasibility rule, keeping initial failures")
    args = parser.parse_args()
    out = Path(args.output); out.mkdir(parents=True, exist_ok=True)
    start = time.time()
    status = {"state": "running", "started_unix": start, "bootstrap_requested": args.bootstrap}
    if not args.sceptr_only and not args.repair_bootstrap_failures:
        json_write(out / "status.json", status)
    data = pd.read_csv(args.features, sep="\t")
    if not data.patient_key.is_unique:
        raise ValueError("Primary input must contain one row per patient")
    blocks = feature_blocks(data)
    known = data[data.n_binary.isin([0, 1])].copy()
    home = known[known.cohort.astype(str).str.lower() == "home"].reset_index(drop=True)
    external = known[known.cohort.astype(str).str.lower() == "external"].reset_index(drop=True)
    assert (len(home), int(home.n_binary.sum())) == (126, 56), "Frozen Home cohort changed"
    assert (len(external), int(external.n_binary.sum())) == (163, 54), "Frozen External cohort changed"
    for column in ("sex", "batch"):
        home[column] = home[column].astype("object").where(home[column].notna(), "Missing").astype(str)
        external[column] = external[column].astype("object").where(external[column].notna(), "Missing").astype(str)
    if args.sceptr_only or args.repair_bootstrap_failures:
        previous = json.loads((out/"model_spec.json").read_text())
        assert previous["input_sha256"]==hashlib.sha256(Path(args.features).read_bytes()).hexdigest(), "Main feature freeze changed"
        if args.repair_bootstrap_failures:
            repair_bootstrap_failures(home,blocks,out)
            return
        if not args.sceptr_means or not args.sceptr_index:
            raise ValueError("Both mean and index inputs are required")
        run_sceptr_sensitivity(home,blocks,out,args.sceptr_means,args.sceptr_index)
        main_status = json.loads((out/"status.json").read_text())
        if main_status.get("state")!="running":
            main_status["sceptr_status"]="complete"
            json_write(out/"status.json",main_status)
        print("Delayed SCEPTR sensitivity complete; primary models/bootstrap untouched",flush=True)
        return
    spec = {"seeds": SEEDS, "outer_folds": 5, "inner_folds": 3, "C_grid": CS,
            "models": blocks, "primary_comparison": "M3 versus M1",
            "delta_direction": "log_loss(M1)-log_loss(M3); positive is improvement",
            "primary_prediction": "mean of three patient-held-out OOF probabilities",
            "bootstrap": "200 outcome-stratified patient draws; duplicate identity grouped in both CV levels; refit all steps",
            "partition_feasibility": "At either CV level, require both outcome classes in every train/test fold; only if infeasible retry seed+7919*j for j=1..4 before any model fitting. No performance-based partition selection.",
            "conditional_ci": "fixed OOF probabilities resampled by patients; does not include pipeline refitting",
            "portable_note": "historically exposed external cohort recheck; no external selection or clinical imputation",
            "lobo_note": "batch omitted in both comparators; no held-out batch offset estimable",
            "input_sha256": hashlib.sha256(Path(args.features).read_bytes()).hexdigest(),
            "home_n": len(home), "external_n": len(external)}
    json_write(out / "model_spec.json", spec)
    print("Main nested OOF", flush=True)
    oof, audit = nested_oof(home, blocks)
    oof.to_csv(out / "oof_predictions.tsv", sep="\t", index=False)
    audit.to_csv(out / "fold_audit.tsv", sep="\t", index=False)
    ensemble = ensemble_predictions(oof)
    ensemble.to_csv(out / "oof_patient_ensemble.tsv", sep="\t", index=False)
    metric_rows = []
    for (seed, model), part in oof.groupby(["seed", "model"]):
        metric_rows.append({"prediction": f"seed_{seed}", "model": model, **metrics(part.n_binary, part.probability)})
    for model, part in ensemble.groupby("model"):
        metric_rows.append({"prediction": "three_seed_mean", "model": model, **metrics(part.n_binary, part.probability)})
    pd.DataFrame(metric_rows).to_csv(out / "metrics.tsv", sep="\t", index=False)
    metric_ci = []
    for model,part in ensemble.groupby("model"):
        metric_ci.extend(conditional_metrics_ci(part.n_binary,part.probability,"Home",model,"three_seed_mean"))
    pd.DataFrame(metric_ci).to_csv(out/"metrics_ci.tsv",sep="\t",index=False)
    comparisons, ci = [], []
    for base, new in (("M1", "M3"), ("M0", "M3")):
        y, b, n = paired_array(ensemble, base, new)
        comparisons.append({"baseline": base, "new": new, "prediction": "three_seed_mean", **paired_metrics(y, b, n)})
        ci.extend({"baseline": base, "new": new, **row} for row in conditional_ci(y, b, n))
        for seed in SEEDS:
            yy, bb, nn = paired_array(oof[oof.seed == seed], base, new)
            comparisons.append({"baseline": base, "new": new, "prediction": f"seed_{seed}", **paired_metrics(yy, bb, nn)})
    pd.DataFrame(comparisons).to_csv(out / "paired_comparisons.tsv", sep="\t", index=False)
    pd.DataFrame(ci).to_csv(out / "conditional_paired_ci.tsv", sep="\t", index=False)
    if args.pooled_features:
        print("Duplicate-library pooling sensitivity (not primary)", flush=True)
        pooled_home, changed = pooled_home_frame(home, args.pooled_features, blocks)
        poof, paudit = nested_oof(pooled_home, blocks, ("M1", "M3"))
        poof.to_csv(out/"pooled_duplicate_oof_predictions.tsv", sep="\t", index=False)
        paudit.to_csv(out/"pooled_duplicate_fold_audit.tsv", sep="\t", index=False)
        pens = ensemble_predictions(poof)
        pd.DataFrame([{"model": m, **metrics(part.n_binary,part.probability)} for m,part in pens.groupby("model")]).to_csv(out/"pooled_duplicate_metrics.tsv", sep="\t", index=False)
        py, pb, pn = paired_array(pens)
        json_write(out/"pooled_duplicate_paired_metrics.json", finite_dict(paired_metrics(py,pb,pn)))
        status["pooled_duplicate_sensitivity"] = {"changed_patient_keys": changed, "n": len(pooled_home), "bootstrap": "not_run_sensitivity_only"}
    print("Portable and leave-one-batch evaluations", flush=True)
    port_pred, port_metrics = portable_evaluation(home, external, blocks, out)
    port_pred.to_csv(out / "external_predictions.tsv", sep="\t", index=False)
    port_metrics.to_csv(out / "external_metrics.tsv", sep="\t", index=False)
    for model,part in port_pred.groupby("model"):
        metric_ci.extend(conditional_metrics_ci(part.n_binary,part.probability,"External",model,"frozen_Home_fit"))
    pd.DataFrame(metric_ci).to_csv(out/"metrics_ci.tsv",sep="\t",index=False)
    p1 = port_pred[port_pred.model == "M1_portable"].set_index("patient_key")
    p3 = port_pred[port_pred.model == "M3_portable"].set_index("patient_key").loc[p1.index]
    pd.DataFrame(conditional_ci(p1.n_binary.to_numpy(int), p1.probability.to_numpy(), p3.probability.to_numpy())).to_csv(out / "external_conditional_paired_ci.tsv", sep="\t", index=False)
    json_write(out / "external_paired_metrics.json", finite_dict(paired_metrics(p1.n_binary, p1.probability, p3.probability)))
    lobo_pred, lobo_metrics = lobo_evaluation(home, blocks)
    lobo_pred.to_csv(out / "lobo_predictions.tsv", sep="\t", index=False)
    lobo_metrics.to_csv(out / "lobo_metrics.tsv", sep="\t", index=False)
    lobo_pairs = []
    for batch, part in lobo_pred.groupby("batch"):
        b = part[part.model == "M1_lobo"].set_index("patient_key")
        n = part[part.model == "M3_lobo"].set_index("patient_key").loc[b.index]
        lobo_pairs.append({"held_out_batch": batch, **paired_metrics(b.n_binary, b.probability, n.probability)})
    pd.DataFrame(lobo_pairs).to_csv(out / "lobo_paired_metrics.tsv", sep="\t", index=False)
    if args.sceptr_means and args.sceptr_index:
        print("SCEPTR full-repertoire sensitivity", flush=True)
        run_sceptr_sensitivity(home,blocks,out,args.sceptr_means,args.sceptr_index)
    else:
        status["sceptr_status"] = "not_run_missing_input"
    bootstrap_results = []
    print(f"Full-pipeline bootstrap {args.bootstrap} repeats, {args.jobs} jobs", flush=True)
    with ProcessPoolExecutor(max_workers=args.jobs) as pool:
        futures = [pool.submit(bootstrap_worker, home, blocks, r, SEEDS, 5, 3) for r in range(args.bootstrap)]
        for completed, future in enumerate(as_completed(futures), 1):
            bootstrap_results.append(future.result())
            pd.DataFrame(sorted(bootstrap_results, key=lambda x: x["repetition"])).to_csv(out / "full_pipeline_bootstrap.tsv", sep="\t", index=False)
            status.update({"bootstrap_finished": completed, "bootstrap_success": sum(r["status"] == "success" for r in bootstrap_results)})
            json_write(out / "status.json", status)
            if completed % 10 == 0:
                print(f"Bootstrap {completed}/{args.bootstrap} elapsed={time.time()-start:.0f}s", flush=True)
    successes = pd.DataFrame([r for r in bootstrap_results if r["status"] == "success"])
    boot_ci = []
    for metric in ("delta_log_loss", "delta_AUROC", "delta_AUPRC", "delta_Brier"):
        if len(successes):
            boot_ci.append({"metric": metric, "ci_low": successes[metric].quantile(.025),
                            "ci_high": successes[metric].quantile(.975), "repetitions": len(successes),
                            "ci_type": "full_pipeline_patient_stratified_percentile", "baseline": "M1", "new": "M3"})
    pd.DataFrame(boot_ci).to_csv(out / "full_pipeline_paired_ci.tsv", sep="\t", index=False)
    print("Within-batch shuffled-label diagnostics", flush=True)
    controls = []
    with ProcessPoolExecutor(max_workers=args.jobs) as pool:
        futures = [pool.submit(negative_worker, home, blocks, r, SEEDS, 5, 3) for r in range(args.negative_controls)]
        for future in as_completed(futures):
            controls.append(future.result())
    pd.DataFrame(sorted(controls, key=lambda x: x["repetition"])).to_csv(out / "negative_controls.tsv", sep="\t", index=False)
    status.update({"state": "complete" if len(successes) == args.bootstrap else "completed_with_bootstrap_failures",
                   "elapsed_seconds": time.time()-start, "bootstrap_success": len(successes),
                   "bootstrap_failed": args.bootstrap-len(successes), "negative_controls_success": sum(r["status"] == "success" for r in controls),
                   "main_models": ["M0", "M1", "M2", "M3"], "n_patient_overlap": int(audit.patient_overlap.max()),
                   "sceptr_status": "complete" if args.sceptr_means and args.sceptr_index else "not_run_missing_input"})
    json_write(out / "status.json", finite_dict(status))
    print(json.dumps(finite_dict(status), ensure_ascii=False), flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        if "--output" in sys.argv:
            output = Path(sys.argv[sys.argv.index("--output")+1]); output.mkdir(parents=True,exist_ok=True)
            path = output/("sceptr_status.json" if "--sceptr-only" in sys.argv else "status.json")
            failure = json.loads(path.read_text()) if path.exists() else {}
            failure.update({"state":"failed","error":repr(exc),"traceback":traceback.format_exc()})
            json_write(path,failure)
        raise
