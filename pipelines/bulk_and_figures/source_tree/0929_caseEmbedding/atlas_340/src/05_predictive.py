#!/usr/bin/env python3
"""Nested, patient-level Home prediction using frozen fold-local TRB groups.

Each outer test patient is projected into a dictionary formed from that outer
training set without its labels. Inner tuning reuses this frozen outer feature
space; inner validation patients participated in label-blind dictionary building.
"""
from __future__ import annotations
from release_runtime import private_literal as _private_literal, release_path as _release_path

import argparse
import csv
import hashlib
import json
import os
import resource
import time
import warnings
from pathlib import Path

os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score, brier_score_loss, log_loss, roc_auc_score,
)
import statsmodels.api as sm
from statsmodels.tools.sm_exceptions import PerfectSeparationWarning


ROOT = Path(_release_path('path_99d8c1c6fde2a6b9'))
OUT = ROOT / "prediction"
FOLDS = ROOT / "folds/folds.tsv"
MANIFEST = ROOT / "config/cohort_samples.tsv"
MODEL_RANDOM_STATE = 314159
C_GRID = (0.01, 0.1, 1.0, 10.0)
BOOTSTRAP_N = 5000
CLIP = 1e-6


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def id_hash(ids: set[str]) -> str:
    return hashlib.sha256(("\n".join(sorted(ids)) + "\n").encode()).hexdigest()


def json_write(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    tmp.replace(path)


def tsv_write(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(tmp, sep="\t", index=False, float_format="%.12g")
    tmp.replace(path)


def folds_and_clinical() -> tuple[pd.DataFrame, pd.DataFrame]:
    folds = pd.read_csv(FOLDS, sep="\t", dtype={"patient_key": str})
    needed = {"patient_key", "recorded_n_binary", "outer_fold", "partition", "inner_fold"}
    if not needed <= set(folds) or len(folds) != 630:
        raise ValueError("Patient fold file differs from locked 5 by 3 layout")
    if folds.patient_key.nunique() != 126 or folds.groupby("patient_key").size().ne(5).any():
        raise ValueError("Patient fold multiplicity invalid")
    if folds.groupby("patient_key").recorded_n_binary.nunique().ne(1).any():
        raise ValueError("Patient label differs between folds")
    if folds.recorded_n_binary.value_counts().to_dict() != {0: 350, 1: 280}:
        raise ValueError("Fold labels do not match 70 N0 and 56 N+ patients")
    for outer in range(5):
        part = folds.loc[folds.outer_fold.eq(outer)]
        if len(part) != 126 or part.patient_key.nunique() != 126:
            raise ValueError("Outer fold does not cover each patient exactly once")
        test = part.loc[part.partition.eq("test")]
        train = part.loc[part.partition.eq("train")]
        if not 24 <= len(test) <= 27 or len(train) + len(test) != 126:
            raise ValueError("Outer train/test size invalid")
        if not test.inner_fold.isna().all() or set(train.inner_fold.dropna()) != {0, 1, 2}:
            raise ValueError("Inner fold assignments missing or heldout patient has inner fold")
        if len(set(train.patient_key) & set(test.patient_key)):
            raise ValueError("Patient appears in train and test")
    with MANIFEST.open(newline="") as handle:
        header = handle.readline()
        home_lines = [line for line in handle if line.startswith("home\t")]
    fields = ("patient_key", "recorded_n_binary", "age_years", "sex",
              "neoadjuvant", "home_batch")
    rows = [{key: row[key] for key in fields}
            for row in csv.DictReader([header] + home_lines, delimiter="\t")]
    clinical = pd.DataFrame(rows)
    for key, group in clinical.groupby("patient_key"):
        if len(group.drop_duplicates()) != 1:
            raise ValueError("Same-patient Home libraries disagree on clinical fields: " + key)
    clinical = clinical.drop_duplicates("patient_key")
    known = set(folds.patient_key)
    clinical = clinical.loc[clinical.patient_key.isin(known)].set_index("patient_key")
    if set(clinical.index) != known or len(clinical) != 126:
        raise ValueError("Clinical patient IDs do not match folds")
    fold_y = folds.drop_duplicates("patient_key").set_index("patient_key").recorded_n_binary
    if not (clinical.recorded_n_binary.astype(int).sort_index() == fold_y.sort_index()).all():
        raise ValueError("Fold label and manifest recorded N disagree")
    clinical["age_years"] = pd.to_numeric(clinical.age_years, errors="raise")
    if clinical.age_years.isna().any() or not clinical.sex.isin(("m", "f")).all() or (
        not clinical.neoadjuvant.isin(("yes", "no")).all()
    ) or not clinical.home_batch.isin(("H2498B", "H2498C", "H2498D")).all():
        raise ValueError("Home baseline covariate unavailable")
    clinical["y"] = clinical.recorded_n_binary.astype(int)
    return folds, clinical


def scope_train_ids(folds: pd.DataFrame, scope: str) -> set[str]:
    if not scope.startswith("outer_") or scope.split("_")[1] not in {
        "0", "1", "2", "3", "4"
    }:
        raise ValueError("Only frozen outer_0..outer_4 dictionaries are supported")
    outer = int(scope.split("_")[1])
    rows = folds.loc[folds.outer_fold.eq(outer) & folds.partition.eq("train")]
    return set(rows.patient_key)


def load_scope(
    folds: pd.DataFrame, scope: str, selected_ids: set[str],
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    directory = ROOT / "groups" / scope
    receipt_path = directory / "burden_receipt.json"
    burden_path = directory / "patient_group_burden.parquet"
    membership_path = directory / "group_membership.parquet"
    receipt = json.loads(receipt_path.read_text())
    hashes = {
        "receipt": sha256(receipt_path),
        "burden": sha256(burden_path),
        "membership": sha256(membership_path),
    }
    trained = scope_train_ids(folds, scope)
    if receipt.get("status") != "complete" or receipt.get("scope") != scope or (
        receipt.get("target_cohort") != "home"
    ) or receipt.get("train_patient_sha256") != id_hash(trained) or (
        receipt.get("folds_sha256") != sha256(FOLDS)
    ) or receipt.get("patient_group_burden_sha256") != hashes["burden"] or (
        receipt.get("group_membership_sha256") != hashes["membership"]
    ) or receipt.get("n_patients") != 137:
        raise ValueError("Fold-local frozen dictionary or burden provenance invalid: " + scope)
    required = ["patient_key", "group_id", "presence", "total_reads", "total_unique_aa"]
    burden = pd.read_parquet(burden_path, columns=required)
    if len(burden) != receipt.get("n_rows") or burden.group_id.nunique() != (
        receipt.get("n_groups")
    ) or burden[["patient_key", "group_id"]].duplicated().any():
        raise ValueError("Fold burden grid malformed: " + scope)
    if not selected_ids <= set(burden.patient_key.astype(str)):
        raise ValueError("Fold burden omits requested projected patients: " + scope)
    burden = burden.loc[burden.patient_key.isin(selected_ids)]
    expected = len(selected_ids) * int(receipt["n_groups"])
    if len(burden) != expected:
        raise ValueError("Fold burden lacks zero-carrier rows: " + scope)
    totals = burden[["patient_key", "total_reads", "total_unique_aa"]].drop_duplicates()
    if totals.patient_key.duplicated().any() or totals[["total_reads", "total_unique_aa"]].isna().any().any() or (
        totals[["total_reads", "total_unique_aa"]].le(0).any().any()
    ):
        raise ValueError("Patient repertoire totals inconsistent: " + scope)
    groups = sorted(burden.group_id.astype(str).unique())
    matrix = burden.pivot(index="patient_key", columns="group_id", values="presence")
    matrix = matrix.reindex(index=sorted(selected_ids), columns=groups)
    if matrix.isna().any().any() or not matrix.isin((0, 1)).all().all():
        raise ValueError("Invalid group carrier matrix: " + scope)
    return matrix.astype(float), totals.set_index("patient_key"), hashes


def baseline_raw(clinical: pd.DataFrame, totals: pd.DataFrame, ids: list[str]) -> np.ndarray:
    c = clinical.loc[ids]
    t = totals.loc[ids]
    return np.column_stack((
        c.age_years.to_numpy(dtype=float),
        (c.sex == "m").to_numpy(dtype=float),
        np.log1p(t.total_reads.to_numpy(dtype=float)),
        np.log1p(t.total_unique_aa.to_numpy(dtype=float)),
        (c.home_batch == "H2498C").to_numpy(dtype=float),
        (c.home_batch == "H2498D").to_numpy(dtype=float),
        (c.neoadjuvant == "yes").to_numpy(dtype=float),
    ))


def transform_baseline(train: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    a = train.copy()
    b = target.copy()
    for col in (0, 2, 3):
        mean = a[:, col].mean()
        sd = a[:, col].std()
        if not np.isfinite(sd) or sd == 0:
            sd = 1.0
        a[:, col] = (a[:, col] - mean) / sd
        b[:, col] = (b[:, col] - mean) / sd
    if not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError("Nonfinite clinical/technical model feature")
    return a, b


def fit_predict(
    kind: str, C: float, y_train: np.ndarray, b_train: np.ndarray,
    b_target: np.ndarray, g_train: np.ndarray, g_target: np.ndarray,
) -> tuple[np.ndarray, int]:
    if kind not in ("baseline", "groups"):
        raise ValueError("Unknown prediction model kind: " + kind)
    x_train, x_target = transform_baseline(b_train, b_target)
    if kind == "groups":
        x_train = sparse.hstack((sparse.csr_matrix(x_train), sparse.csr_matrix(g_train)),
                                format="csr")
        x_target = sparse.hstack((sparse.csr_matrix(x_target), sparse.csr_matrix(g_target)),
                                 format="csr")
    model = LogisticRegression(
        C=C, penalty="l1", solver="liblinear", max_iter=2000,
        tol=1e-5, random_state=MODEL_RANDOM_STATE,
    )
    model.fit(x_train, y_train)
    if model.n_iter_[0] >= 2000:
        raise ValueError("Penalized logistic did not converge")
    p = model.predict_proba(x_target)[:, 1]
    if not np.isfinite(p).all():
        raise ValueError("Nonfinite model predictions")
    nonzero = int(np.count_nonzero(model.coef_[0][7:])) if kind == "groups" else 0
    return np.clip(p, CLIP, 1-CLIP), nonzero


def patient_loss(y: np.ndarray, p: np.ndarray) -> np.ndarray:
    p = np.clip(np.asarray(p, dtype=float), CLIP, 1-CLIP)
    return -(y*np.log(p) + (1-y)*np.log1p(-p))


def select_c(scores: pd.DataFrame) -> float:
    stats = scores.groupby("C", sort=True).log_loss.agg(
        ["mean", "std", "count"]
    ).reset_index()
    stats["se"] = stats["std"].fillna(0) / np.sqrt(stats["count"])
    best = stats.loc[stats["mean"].idxmin()]
    eligible = stats.loc[stats["mean"].le(best["mean"] + best["se"] + 1e-12)]
    return float(eligible.C.min())


def inner_tune(
    folds: pd.DataFrame, clinical: pd.DataFrame, outer: int,
    matrix: pd.DataFrame, totals: pd.DataFrame,
) -> tuple[dict[str, float], list[dict]]:
    """Tune only logistic C inside one label-blind frozen outer dictionary."""
    records = []
    fold_rows = folds.loc[folds.outer_fold.eq(outer) &
                          folds.partition.eq("train")]
    for inner in range(3):
        train_ids = sorted(fold_rows.loc[fold_rows.inner_fold.ne(inner), "patient_key"])
        val_ids = sorted(fold_rows.loc[fold_rows.inner_fold.eq(inner), "patient_key"])
        if not train_ids or not val_ids or not set(train_ids+val_ids) <= set(matrix.index):
            raise ValueError("Inner split lacks fold-local projected group features")
        y_train = clinical.loc[train_ids].y.to_numpy()
        y_val = clinical.loc[val_ids].y.to_numpy()
        b_train = baseline_raw(clinical, totals, train_ids)
        b_val = baseline_raw(clinical, totals, val_ids)
        g_train = matrix.loc[train_ids].to_numpy(dtype=float)
        g_val = matrix.loc[val_ids].to_numpy(dtype=float)
        for kind in ("baseline", "groups"):
            for C in C_GRID:
                p, nonzero = fit_predict(
                    kind, C, y_train, b_train, b_val, g_train, g_val
                )
                records.append({
                    "outer_fold": outer, "inner_fold": inner, "model": kind,
                    "C": C, "n_train": len(train_ids), "n_valid": len(val_ids),
                    "dictionary_scope": f"outer_{outer}",
                    "n_group_features": g_train.shape[1], "n_nonzero_groups": nonzero,
                    "log_loss": float(patient_loss(y_val, p).mean()),
                })
    table = pd.DataFrame(records)
    chosen = {kind: select_c(table.loc[table.model.eq(kind)])
              for kind in ("baseline", "groups")}
    return chosen, records


def calibration(y: np.ndarray, p: np.ndarray) -> tuple[float | None, float | None]:
    logit = np.log(np.clip(p, CLIP, 1-CLIP) / (1-np.clip(p, CLIP, 1-CLIP)))
    if np.unique(y).size < 2 or np.std(logit) == 0:
        return None, None
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", PerfectSeparationWarning)
            fit = sm.GLM(y, sm.add_constant(logit), family=sm.families.Binomial()).fit(
                maxiter=100, tol=1e-8, disp=0
            )
        if fit.converged and np.isfinite(fit.params).all():
            return float(fit.params[0]), float(fit.params[1])
    except Exception:
        pass
    return None, None


def metrics(y: np.ndarray, p: np.ndarray) -> dict:
    intercept, slope = calibration(y, p)
    return {
        "log_loss": float(log_loss(y, p, labels=[0, 1])),
        "brier": float(brier_score_loss(y, p)),
        "auroc": float(roc_auc_score(y, p)) if np.unique(y).size == 2 else None,
        "ap": float(average_precision_score(y, p)) if np.unique(y).size == 2 else None,
        "calibration_intercept": intercept,
        "calibration_slope": slope,
    }


def paired_bootstrap(delta: np.ndarray, n: int = BOOTSTRAP_N) -> dict:
    rng = np.random.default_rng(20260930)
    ix = rng.integers(0, len(delta), size=(n, len(delta)), endpoint=False)
    samples = delta[ix].mean(axis=1)
    lo, hi = np.quantile(samples, (0.025, 0.975))
    return {
        "mean": float(delta.mean()),
        "ci_low": float(lo),
        "ci_high": float(hi),
        "n_bootstrap": n,
        "interval": "patient-paired nonparametric percentile, conditional on fixed folds",
    }


def run() -> None:
    started = time.monotonic()
    OUT.mkdir(parents=True, exist_ok=True)
    folds, clinical = folds_and_clinical()
    oof_rows = []
    tune_rows = []
    fold_rows = []
    source_hashes = {}
    for outer in range(5):
        part = folds.loc[folds.outer_fold.eq(outer)]
        train_ids = sorted(part.loc[part.partition.eq("train"), "patient_key"])
        test_ids = sorted(part.loc[part.partition.eq("test"), "patient_key"])
        scope = f"outer_{outer}"
        matrix, totals, hashes = load_scope(
            folds, scope, set(train_ids) | set(test_ids)
        )
        source_hashes[scope] = hashes
        chosen, scores = inner_tune(folds, clinical, outer, matrix, totals)
        tune_rows.extend(scores)
        y_train = clinical.loc[train_ids].y.to_numpy()
        y_test = clinical.loc[test_ids].y.to_numpy()
        b_train = baseline_raw(clinical, totals, train_ids)
        b_test = baseline_raw(clinical, totals, test_ids)
        g_train = matrix.loc[train_ids].to_numpy(dtype=float)
        g_test = matrix.loc[test_ids].to_numpy(dtype=float)
        p0, _ = fit_predict(
            "baseline", chosen["baseline"], y_train, b_train, b_test, g_train, g_test
        )
        p1, nonzero = fit_predict(
            "groups", chosen["groups"], y_train, b_train, b_test, g_train, g_test
        )
        loss0, loss1 = patient_loss(y_test, p0), patient_loss(y_test, p1)
        base_metrics = metrics(y_test, p0)
        group_metrics = metrics(y_test, p1)
        fold_rows.append({
            "outer_fold": outer, "n_test": len(test_ids),
            "baseline_C": chosen["baseline"], "groups_C": chosen["groups"],
            "n_group_features": int(g_train.shape[1]), "n_nonzero_groups": nonzero,
            "paired_log_loss_improvement": float((loss0-loss1).mean()),
            **{"baseline_" + k: v for k, v in base_metrics.items()},
            **{"groups_" + k: v for k, v in group_metrics.items()},
        })
        for i,key in enumerate(test_ids):
            oof_rows.append({
                "patient_key": key, "outer_fold": outer,
                "recorded_n_binary": int(y_test[i]),
                "baseline_probability": float(p0[i]),
                "groups_probability": float(p1[i]),
                "baseline_patient_log_loss": float(loss0[i]),
                "groups_patient_log_loss": float(loss1[i]),
                "paired_log_loss_improvement": float(loss0[i]-loss1[i]),
                "baseline_C": chosen["baseline"], "groups_C": chosen["groups"],
                "n_fold_group_features": int(g_train.shape[1]),
                "n_fold_nonzero_group_features": nonzero,
            })
        print(json.dumps({"outer_fold": outer, "n_test": len(test_ids),
                          "paired_log_loss_improvement": float((loss0-loss1).mean()),
                          "n_groups": g_train.shape[1], "nonzero": nonzero}),
              flush=True)
    oof = pd.DataFrame(oof_rows).sort_values("patient_key").reset_index(drop=True)
    if len(oof) != 126 or oof.patient_key.duplicated().any() or (
        set(oof.patient_key) != set(clinical.index)
    ):
        raise ValueError("Each Home patient must have exactly one outer-fold prediction")
    fold_summary = pd.DataFrame(fold_rows)
    tuning = pd.DataFrame(tune_rows)
    y = oof.recorded_n_binary.to_numpy()
    p0 = oof.baseline_probability.to_numpy()
    p1 = oof.groups_probability.to_numpy()
    paired = paired_bootstrap(oof.paired_log_loss_improvement.to_numpy())
    favorable_folds = int(fold_summary.paired_log_loss_improvement.gt(0).sum())
    summary = {
        "status": "complete",
        "endpoint": "recorded N+; pN/ypN unverified",
        "population": "126 N-known Home patients; 70 N0 and 56 N+",
        "baseline": metrics(y, p0),
        "all_group_carrier_model": metrics(y, p1),
        "paired_baseline_minus_groups_log_loss": paired,
        "favorable_outer_folds": favorable_folds,
        "n_outer_folds": 5, "n_inner_folds": 3,
        "meets_predictive_contribution_gate": bool(
            paired["ci_low"] > 0 and favorable_folds >= 4
        ),
        "gate_rule": "patient-paired log-loss improvement CI lower bound >0 and at least 4 of 5 favorable outer folds",
        "model_spec": {
            "baseline": "L1 logistic on age, sex, log1p reads, log1p unique AA, Home batch, neoadjuvant",
            "groups": "L1 logistic on same covariates plus all fold-local group carrier indicators",
            "C_grid": list(C_GRID), "selection": "3-inner-fold log loss in frozen outer feature space; smallest C within one SE of minimum",
            "intercept": True, "class_weight": None,
            "clinical_scaling": "continuous columns standardized on current training patients only",
            "group_scaling": "binary presence kept as 0/1",
            "inner_dictionary_boundary": "inner validation patients contributed unlabeled sequence data to their outer-training dictionary; inner labels were not used",
            "model_seed": MODEL_RANDOM_STATE,
        },
        "interpretation_limit": "External and group-deletion results are separate gates; this is fixed-fold exploratory OOF evidence",
    }
    oof_path = OUT / "05_patient_oof.tsv"
    fold_path = OUT / "05_outer_metrics.tsv"
    tune_path = OUT / "05_inner_tuning.tsv"
    summary_path = OUT / "05_summary.json"
    tsv_write(oof_path, oof)
    tsv_write(fold_path, fold_summary)
    tsv_write(tune_path, tuning)
    json_write(summary_path, summary)
    receipt = {
        "status": "complete",
        "n_oof_patients": len(oof),
        "n_outer_folds": 5, "n_inner_folds": 3,
        "n_inner_model_fits": len(tuning),
        "source_sha256": {
            "folds": sha256(FOLDS), "manifest": sha256(MANIFEST),
            "analysis_spec": sha256(ROOT / "config/analysis_spec.json"),
            "scope_files": source_hashes,
        },
        "output_sha256": {
            "oof": sha256(oof_path), "outer_metrics": sha256(fold_path),
            "inner_tuning": sha256(tune_path), "summary": sha256(summary_path),
        },
        "outer_heldout_labels_used_for_tuning": False,
        "fold_local_group_dictionary_verified_by_train_patient_hash": True,
        "n_outer_dictionaries": len(source_hashes),
        "inner_reuses_frozen_outer_dictionary": True,
        "runtime_seconds": round(time.monotonic()-started, 3),
        "maxrss_gib": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024**2), 3),
        "cpu_threads_max_configured": 1,
    }
    json_write(OUT / "05_receipt.json", receipt)
    print(json.dumps({"status": "complete", "summary": summary_path.as_posix(),
                      "gate": summary["meets_predictive_contribution_gate"]}), flush=True)


def self_test() -> None:
    rng = np.random.default_rng(41)
    n = 120
    baseline = np.column_stack((
        rng.normal(60, 8, n), rng.integers(0, 2, n),
        rng.normal(12, .5, n), rng.normal(8, .5, n),
        rng.integers(0, 2, n), rng.integers(0, 2, n),
        rng.integers(0, 2, n),
    ))
    signal = rng.integers(0, 2, n)
    y = np.bitwise_xor(signal, (rng.random(n) < .1).astype(int))
    empty_groups = np.zeros((n, 3))
    train, test = slice(0, 90), slice(90, n)
    for c in C_GRID:
        p0, _ = fit_predict("baseline", c, y[train], baseline[train],
                            baseline[test], empty_groups[train], empty_groups[test])
        p1, nonzero = fit_predict("groups", c, y[train], baseline[train],
                                  baseline[test], empty_groups[train], empty_groups[test])
        if nonzero or not np.allclose(p0, p1, atol=1e-4, rtol=1e-4):
            raise AssertionError("Zero group features changed matched-model predictions")
    groups = np.column_stack((signal, np.zeros(n), rng.integers(0, 2, n)))
    p0, _ = fit_predict("baseline", 1.0, y[train], baseline[train],
                        baseline[test], groups[train], groups[test])
    p1, nonzero = fit_predict("groups", 1.0, y[train], baseline[train],
                              baseline[test], groups[train], groups[test])
    if nonzero < 1 or patient_loss(y[test], p1).mean() >= patient_loss(y[test], p0).mean():
        raise AssertionError("Added informative group did not improve held-out log loss")
    print(json.dumps({"status": "PASS_self_test", "checks": [
        "zero-group matched-model predictions", "informative-group held-out gain"]}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test()
    else:
        run()
