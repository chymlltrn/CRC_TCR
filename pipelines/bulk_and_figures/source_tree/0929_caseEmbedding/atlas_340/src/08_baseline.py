#!/usr/bin/env python3
"""Frozen-fold clinical/technical baseline shared by all representation arms."""
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

ROOT = Path(_release_path('path_99d8c1c6fde2a6b9'))
SOURCE = ROOT / "src/05_predictive.py"
TOTALS = ROOT / "representation_comparison/home_sample/patient_totals.tsv"
OUT = ROOT / "representation_comparison/baseline"
OOF = OUT / "baseline_oof.tsv"
TUNE = OUT / "baseline_inner_tuning.tsv"
RECEIPT = OUT / "receipt.json"


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write_tsv(path: Path, frame: pd.DataFrame) -> None:
    tmp = path.with_suffix(path.suffix + ".partial")
    frame.to_csv(tmp, sep="\t", index=False, float_format="%.12g")
    os.replace(tmp, path)


def write_json(path: Path, data: dict) -> None:
    tmp = path.with_suffix(path.suffix + ".partial")
    tmp.write_text(json.dumps(data, indent=2, sort_keys=True, allow_nan=False) + "\n")
    os.replace(tmp, path)


def main() -> None:
    start = time.monotonic()
    OUT.mkdir(parents=True, exist_ok=True)
    spec = importlib.util.spec_from_file_location("atlas_frozen_predictive", SOURCE)
    frozen = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(frozen)
    folds, clinical = frozen.folds_and_clinical()
    totals = pd.read_csv(TOTALS, sep="\t", dtype={"patient_key": str}).set_index("patient_key")
    totals = totals.rename(columns={"total_reads": "total_reads", "total_unique_aa": "total_unique_aa"})
    if set(clinical.index) - set(totals.index):
        raise ValueError("Full repertoire totals missing known Home patients")
    if (totals[["total_reads", "total_unique_aa"]] <= 0).any().any():
        raise ValueError("Invalid full repertoire totals")
    if RECEIPT.exists():
        r = json.loads(RECEIPT.read_text())
        inputs = {
            "folds": sha(ROOT / "folds/folds.tsv"),
            "manifest": sha(ROOT / "config/cohort_samples.tsv"),
            "patient_totals": sha(TOTALS),
            "frozen_baseline_code": sha(SOURCE),
            "this_script": sha(Path(__file__)),
        }
        if r.get("status") == "complete" and r.get("source_sha256") == inputs and r.get("oof_sha256") == sha(OOF) and r.get("tuning_sha256") == sha(TUNE):
            print(json.dumps(r, sort_keys=True))
            return
        raise RuntimeError("Existing baseline receipt fails input/output hash validation")
    oof = []
    tuning = []
    for outer in range(5):
        rows = folds.loc[folds.outer_fold.eq(outer)]
        train = sorted(rows.loc[rows.partition.eq("train"), "patient_key"])
        test = sorted(rows.loc[rows.partition.eq("test"), "patient_key"])
        for inner in range(3):
            inner_train = sorted(rows.loc[rows.partition.eq("train") & rows.inner_fold.ne(inner), "patient_key"])
            val = sorted(rows.loc[rows.partition.eq("train") & rows.inner_fold.eq(inner), "patient_key"])
            a = frozen.baseline_raw(clinical, totals, inner_train)
            b = frozen.baseline_raw(clinical, totals, val)
            ytr = clinical.loc[inner_train].y.to_numpy(dtype=int)
            yv = clinical.loc[val].y.to_numpy(dtype=int)
            dummy_tr = np.zeros((len(inner_train), 0))
            dummy_val = np.zeros((len(val), 0))
            for c in frozen.C_GRID:
                pred, _ = frozen.fit_predict("baseline", c, ytr, a, b, dummy_tr, dummy_val)
                tuning.append({"outer_fold": outer, "inner_fold": inner, "C": float(c),
                               "log_loss": float(frozen.patient_loss(yv, pred).mean()),
                               "n_train": len(inner_train), "n_val": len(val)})
        chosen = frozen.select_c(pd.DataFrame([r for r in tuning if r["outer_fold"] == outer]))
        ytr = clinical.loc[train].y.to_numpy(dtype=int)
        yte = clinical.loc[test].y.to_numpy(dtype=int)
        xtr = frozen.baseline_raw(clinical, totals, train)
        xte = frozen.baseline_raw(clinical, totals, test)
        pred, _ = frozen.fit_predict("baseline", chosen, ytr, xtr, xte,
                                     np.zeros((len(train), 0)), np.zeros((len(test), 0)))
        for i, key in enumerate(test):
            oof.append({"patient_key": key, "outer_fold": outer, "recorded_n_binary": int(yte[i]),
                        "baseline_probability": float(pred[i]), "baseline_C": float(chosen),
                        "total_reads": int(totals.loc[key, "total_reads"]),
                        "total_unique_aa": int(totals.loc[key, "total_unique_aa"]),
                        "neoadjuvant": str(clinical.loc[key, "neoadjuvant"]),
                        "home_batch": str(clinical.loc[key, "home_batch"])})
    frame = pd.DataFrame(oof).sort_values("patient_key").reset_index(drop=True)
    if len(frame) != 126 or frame.patient_key.duplicated().any() or set(frame.patient_key) != set(clinical.index):
        raise RuntimeError("Baseline is not one OOF probability per N-known Home patient")
    write_tsv(OOF, frame)
    write_tsv(TUNE, pd.DataFrame(tuning))
    source = {
        "folds": sha(ROOT / "folds/folds.tsv"), "manifest": sha(ROOT / "config/cohort_samples.tsv"),
        "patient_totals": sha(TOTALS), "frozen_baseline_code": sha(SOURCE),
        "this_script": sha(Path(__file__)),
    }
    r = {"status": "complete", "endpoint": "recorded N+; pN/ypN unverified",
         "n_oof_patients": len(frame), "n_outer_folds": 5, "n_inner_folds": 3,
         "n_inner_model_fits": len(tuning), "source_sha256": source,
         "oof_sha256": sha(OOF), "tuning_sha256": sha(TUNE),
         "baseline_metrics": frozen.metrics(frame.recorded_n_binary.to_numpy(dtype=int),
                                            frame.baseline_probability.to_numpy(dtype=float)),
         "model": "Frozen 05_predictive.py L1 logistic covariates, C grid, standardization and one-SE selection",
         "fold_train_only": True, "external_labels_read": False,
         "seconds": round(time.monotonic() - start, 3)}
    write_json(RECEIPT, r)
    print(json.dumps(r, sort_keys=True))


if __name__ == "__main__":
    main()
