#!/usr/bin/env python3
"""Train-only L2 clinical/technical baseline sensitivity for representation screen."""
from __future__ import annotations
from release_runtime import private_literal as _private_literal, release_path as _release_path
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import time

for key in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[key] = "1"

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss

ROOT = Path(_release_path('path_99d8c1c6fde2a6b9'))
FROZEN = ROOT / "src/05_predictive.py"
TOTALS = ROOT / "representation_comparison/home_sample/patient_totals.tsv"
OUT = ROOT / "representation_comparison/baseline"
OOF = OUT / "baseline_l2_oof.tsv"
TUNE = OUT / "baseline_l2_inner_tuning.tsv"
RECEIPT = OUT / "l2_receipt.json"
C_GRID = (0.01, 0.1, 1.0, 10.0)


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def save_tsv(path, frame):
    tmp = path.with_suffix(path.suffix + ".partial")
    frame.to_csv(tmp, sep="\t", index=False, float_format="%.12g")
    os.replace(tmp, path)


def save_json(path, value):
    tmp = path.with_suffix(path.suffix + ".partial")
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    os.replace(tmp, path)


def predict_l2(frozen, clinical, totals, train, test, c):
    xtr = frozen.baseline_raw(clinical, totals, train)
    xte = frozen.baseline_raw(clinical, totals, test)
    xtr, xte = frozen.transform_baseline(xtr, xte)
    ytr = clinical.loc[train].y.to_numpy(dtype=int)
    fit = LogisticRegression(penalty="l2", C=c, solver="lbfgs", max_iter=2000,
                             tol=1e-6, random_state=314159)
    fit.fit(xtr, ytr)
    if fit.n_iter_[0] >= 2000:
        raise RuntimeError("L2 clinical baseline failed convergence")
    return np.clip(fit.predict_proba(xte)[:, 1], 1e-6, 1 - 1e-6)


def main():
    start = time.monotonic()
    OUT.mkdir(parents=True, exist_ok=True)
    spec = importlib.util.spec_from_file_location("frozen_atlas_predictive", FROZEN)
    frozen = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(frozen)
    folds, clinical = frozen.folds_and_clinical()
    totals = pd.read_csv(TOTALS, sep="\t", dtype={"patient_key": str}).set_index("patient_key")
    source = {"folds": sha(ROOT / "folds/folds.tsv"),
              "manifest": sha(ROOT / "config/cohort_samples.tsv"),
              "totals": sha(TOTALS), "frozen_covariate_code": sha(FROZEN),
              "this_script": sha(Path(__file__))}
    if RECEIPT.exists():
        old = json.loads(RECEIPT.read_text())
        if (old.get("status") == "complete" and old.get("source_sha256") == source
                and old.get("oof_sha256") == sha(OOF)
                and old.get("tuning_sha256") == sha(TUNE)):
            print(json.dumps(old, sort_keys=True))
            return
        raise RuntimeError("L2 baseline receipt mismatch")
    oof = []
    tune = []
    for outer in range(5):
        rows = folds.loc[folds.outer_fold.eq(outer)]
        train = sorted(rows.loc[rows.partition.eq("train"), "patient_key"])
        test = sorted(rows.loc[rows.partition.eq("test"), "patient_key"])
        for inner in range(3):
            inner_train = sorted(rows.loc[rows.partition.eq("train") & rows.inner_fold.ne(inner), "patient_key"])
            val = sorted(rows.loc[rows.partition.eq("train") & rows.inner_fold.eq(inner), "patient_key"])
            yv = clinical.loc[val].y.to_numpy(dtype=int)
            for c in C_GRID:
                p = predict_l2(frozen, clinical, totals, inner_train, val, c)
                tune.append({"outer_fold": outer, "inner_fold": inner, "C": c,
                             "log_loss": float(log_loss(yv, p, labels=[0, 1]))})
        chosen = frozen.select_c(pd.DataFrame([r for r in tune if r["outer_fold"] == outer]))
        p = predict_l2(frozen, clinical, totals, train, test, chosen)
        for i, key in enumerate(test):
            oof.append({"patient_key": key, "outer_fold": outer,
                        "recorded_n_binary": int(clinical.loc[key, "y"]),
                        "baseline_l2_probability": float(p[i]), "baseline_l2_C": float(chosen)})
    frame = pd.DataFrame(oof).sort_values("patient_key").reset_index(drop=True)
    if len(frame) != 126 or frame.patient_key.duplicated().any() or set(frame.patient_key) != set(clinical.index):
        raise RuntimeError("L2 baseline OOF linkage invalid")
    save_tsv(OOF, frame)
    save_tsv(TUNE, pd.DataFrame(tune))
    receipt = {"status": "complete", "purpose": "pre-model clinical/technical baseline sensitivity",
               "endpoint": "recorded N+; pN/ypN unverified",
               "covariates": ["age_years", "sex", "log1p_total_reads", "log1p_total_unique_aa",
                              "home_batch_C", "home_batch_D", "neoadjuvant"],
               "classifier": "L2 logistic; same train-only scaling and one-SE inner C selection as frozen L1 baseline",
               "C_grid": list(C_GRID), "external_labels_read": False,
               "n_patients": len(frame), "n_outer_folds": 5, "n_inner_folds": 3,
               "source_sha256": source, "oof_sha256": sha(OOF),
               "tuning_sha256": sha(TUNE),
               "metrics": frozen.metrics(frame.recorded_n_binary.to_numpy(dtype=int),
                                         frame.baseline_l2_probability.to_numpy(dtype=float)),
               "seconds": round(time.monotonic() - start, 3)}
    save_json(RECEIPT, receipt)
    print(json.dumps(receipt, sort_keys=True))


if __name__ == "__main__":
    main()
