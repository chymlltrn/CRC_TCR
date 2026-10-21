#!/usr/bin/env python3
"""Outcome-blind patient-balanced PCA-vector batch diagnostics."""
from __future__ import annotations
from release_runtime import private_literal as _private_literal, release_path as _release_path

import hashlib
import json
import os
from pathlib import Path

for name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[name] = "1"

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(_release_path('path_99d8c1c6fde2a6b9'))
MODELS = ("tcrbert_dualtask_aa_mean", "sceptr_cdr3_only")
SEED = 20260930


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def classify(x: np.ndarray, y: np.ndarray) -> dict:
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(C=1.0, max_iter=1000, solver="liblinear"),
    )
    pred = cross_val_predict(model, x, y, cv=cv, method="predict_proba")[:, 1]
    return {
        "oof_auc": float(roc_auc_score(y, pred)),
        "oof_balanced_accuracy": float(balanced_accuracy_score(y, pred >= 0.5)),
    }


def main() -> None:
    scope = ROOT / "atlas" / "joint"
    draw_path = scope / "pca_patient_balanced_draws.parquet"
    draw = pd.read_parquet(draw_path)
    manifest_path = ROOT / "config" / "cohort_samples.tsv"
    manifest = pd.read_csv(
        manifest_path, sep="\t", dtype=str, keep_default_na=False,
        usecols=["cohort", "sample_id", "patient_key", "home_batch"],
    )
    info = manifest.groupby("patient_key", as_index=False).agg(
        cohort=("cohort", "first"), home_batch=("home_batch", "first"),
        libraries=("sample_id", "size"),
    )
    assert len(manifest) == 340 and len(info) == 339
    assert len(draw) == 339 * draw.groupby("patient_key").size().iloc[0]
    assert draw.groupby("patient_key").size().nunique() == 1
    assert draw.patient_key.nunique() == 339
    table = info.copy()
    receipts = {}
    for model_name in MODELS:
        model_dir = scope / model_name
        projection_path = model_dir / "pca_scores_fp16.npy"
        projection = json.loads((model_dir / "projection_receipt.json").read_text())
        fit = json.loads((model_dir / "pca_fit_receipt.json").read_text())
        assert projection["status"] == fit["status"] == "complete"
        assert fit["draw_sha256"] == sha(draw_path)
        assert projection["scores_sha256"] == sha(projection_path)
        scores = np.load(projection_path, mmap_mode="r")
        assert len(scores) == projection["n_aa"] and scores.shape[1] == 64
        selected = draw[["patient_key", "seq_idx"]].copy()
        selected["row"] = np.arange(len(selected))
        selected = selected.sort_values(["patient_key", "row"])
        records = []
        for patient_key, part in selected.groupby("patient_key", sort=False):
            x = np.asarray(scores[part.seq_idx.to_numpy(dtype=np.int64)], dtype=np.float32)
            if not np.isfinite(x).all():
                raise ValueError("nonfinite projected vector")
            center = x.mean(axis=0)
            spread = float(np.mean(np.sum((x - center) ** 2, axis=1)))
            records.append((patient_key, spread, *center.tolist()))
        cols = ["patient_key", model_name + "_within_patient_mse"] + [
            model_name + "_pc_" + str(i + 1) for i in range(64)
        ]
        one = pd.DataFrame.from_records(records, columns=cols)
        table = table.merge(one, on="patient_key", validate="one_to_one")
        features = one.drop(columns="patient_key").to_numpy(dtype=float)
        info_aligned = info.set_index("patient_key").loc[one.patient_key]
        y = info_aligned.cohort.eq("external").to_numpy(dtype=int)
        cohort_result = classify(features, y)
        home = info_aligned.cohort.eq("home").to_numpy()
        home_batch = info_aligned.home_batch.to_numpy()[home]
        home_features = features[home]
        home_batch_result = {}
        for batch in sorted(set(home_batch)):
            yy = (home_batch == batch).astype(int)
            home_batch_result[batch] = classify(home_features, yy)
        receipts[model_name] = {
            "cohort_from_balanced_pca_vectors": cohort_result,
            "home_batch_one_vs_rest_from_balanced_pca_vectors": home_batch_result,
            "projection_receipt_sha256": sha(model_dir / "projection_receipt.json"),
            "fit_receipt_sha256": sha(model_dir / "pca_fit_receipt.json"),
            "draw_sha256": fit["draw_sha256"],
        }
    out = ROOT / "atlas" / "patient_vector_summaries.parquet"
    table.to_parquet(out, index=False, compression="zstd")
    result = {
        "status": "complete", "label_blind": True,
        "patients": 339, "home_patients": 137, "external_patients": 202,
        "samples_per_patient": int(draw.groupby("patient_key").size().iloc[0]),
        "basis": "joint 64D PCA fitted on equal per-patient draws",
        "interpretation": "Residual cohort or Home-batch predictability; no batch-removal claim",
        "manifest_sha256": sha(manifest_path),
        "patient_vector_summaries_sha256": sha(out),
        "models": receipts,
        "script_sha256": sha(Path(__file__)),
    }
    receipt_path = ROOT / "atlas" / "vector_batch_receipt.json"
    temp = receipt_path.with_suffix(".json.tmp")
    temp.write_text(json.dumps(result, sort_keys=True, indent=2) + "\n")
    os.replace(temp, receipt_path)
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
