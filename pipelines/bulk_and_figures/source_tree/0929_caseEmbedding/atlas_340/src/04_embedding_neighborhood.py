#!/usr/bin/env python3
"""Label-blind BERT/SCEPTR PCA64 neighborhood agreement on balanced Home AA."""
from __future__ import annotations
from release_runtime import private_literal as _private_literal, release_path as _release_path
import argparse
import hashlib
import json
import os
import time
from pathlib import Path

os.environ["OPENBLAS_NUM_THREADS"] = "4"
os.environ["OMP_NUM_THREADS"] = "4"
os.environ["MKL_NUM_THREADS"] = "4"

import numpy as np
import pandas as pd

ROOT = Path(_release_path('path_99d8c1c6fde2a6b9'))
MODELS = ("tcrbert_dualtask_aa_mean", "sceptr_cdr3_only")
SEED = 20260929
PER_PATIENT = 32
TOP_K = 50


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def neighbors(scores: np.ndarray, k: int) -> np.ndarray:
    x = scores.astype(np.float32, copy=True)
    if not np.isfinite(x).all() or x.ndim != 2 or x.shape[1] != 64:
        raise ValueError("PCA64 sample is missing or nonfinite")
    norms = np.linalg.norm(x, axis=1)
    if not np.all(norms > 0):
        raise ValueError("Zero-length PCA64 score")
    x /= norms[:, None]
    sim = x @ x.T
    np.fill_diagonal(sim, -np.inf)
    ix = np.argpartition(sim, -k, axis=1)[:, -k:]
    return ix


def run() -> dict:
    began = time.monotonic()
    draw_path = ROOT / "atlas/home/pca_patient_balanced_draws.parquet"
    draws = pd.read_parquet(draw_path, columns=["patient_key", "seq_idx"])
    if draws.patient_key.nunique() != 137 or draws.seq_idx.isna().any():
        raise ValueError("Home balanced draw has wrong patient/AA coverage")
    rng = np.random.default_rng(SEED)
    parts = []
    for patient, part in draws.groupby("patient_key", sort=True):
        if len(part) < PER_PATIENT:
            raise ValueError(f"Too few balanced AA for {patient}")
        pick = rng.choice(part.seq_idx.to_numpy(dtype=np.int64),
                          size=PER_PATIENT, replace=False)
        parts.append(pd.DataFrame({"patient_key": patient, "seq_idx": pick}))
    selected = pd.concat(parts, ignore_index=True).drop_duplicates("seq_idx")
    selected = selected.sort_values(["seq_idx", "patient_key"]).reset_index(drop=True)
    ids = selected.seq_idx.to_numpy(dtype=np.int64)
    if len(ids) <= TOP_K or np.any(ids < 0):
        raise ValueError("Sample too small for top-k exact comparison")
    receipt_hashes = {}
    idx = {}
    for model in MODELS:
        folder = ROOT / "atlas/home" / model
        receipt_path = folder / "projection_receipt.json"
        receipt = json.loads(receipt_path.read_text())
        if (receipt.get("status") != "complete" or
            receipt.get("n_aa") != 28380380 or
            receipt.get("missing") != 0 or receipt.get("nonfinite") != 0):
            raise ValueError(f"Incomplete {model} projection receipt")
        matrix = np.load(folder / "pca_scores_fp16.npy", mmap_mode="r")
        if matrix.shape != (28380380, 64) or int(ids.max()) >= len(matrix):
            raise ValueError(f"Projection shape/index mismatch: {model}")
        idx[model] = neighbors(np.asarray(matrix[ids], dtype=np.float32), TOP_K)
        receipt_hashes[model] = sha(receipt_path)
    first, second = (idx[name] for name in MODELS)
    hits = np.fromiter((len(set(a) & set(b)) for a, b in zip(first, second)),
                       count=len(ids), dtype=np.int64)
    selected["top50_common"] = hits
    selected["top50_recall_across_models"] = hits / TOP_K
    selected["top50_jaccard"] = hits / (2 * TOP_K - hits)
    out = ROOT / "atlas/embedding_neighborhood"
    out.mkdir(parents=True, exist_ok=True)
    detail = out / "home_balanced_pca64_top50.tsv"
    selected.to_csv(detail, sep="\t", index=False)
    report = {
        "status": "complete", "scope": "Home label-blind descriptive",
        "representation": "model-specific patient-balanced Home PCA64",
        "comparison": "exact cosine top-50 among the same balanced AA subset",
        "seed": SEED, "per_patient_draw": PER_PATIENT,
        "n_home_patients": int(draws.patient_key.nunique()),
        "n_sampled_unique_aa": len(ids), "top_k": TOP_K,
        "median_overlap_fraction": float(np.median(hits / TOP_K)),
        "mean_overlap_fraction": float(np.mean(hits / TOP_K)),
        "overlap_fraction_q05_q95": [float(x) for x in np.quantile(hits / TOP_K, [.05, .95])],
        "mean_jaccard": float(np.mean(hits / (2 * TOP_K - hits))),
        "random_independent_top50_expected_overlap_fraction": TOP_K / (len(ids)-1),
        "balanced_draw_sha256": sha(draw_path),
        "pca_projection_receipt_sha256": receipt_hashes,
        "detail_sha256": sha(detail),
        "elapsed_seconds": time.monotonic() - began,
        "interpretation": "Descriptive representation agreement only; neither grouping nor candidate selection uses this statistic",
    }
    path = out / "receipt.json"
    path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", action="store_true")
    args = parser.parse_args()
    if not args.run:
        parser.error("Pass --run")
    print(json.dumps(run(), sort_keys=True))
