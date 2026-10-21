#!/usr/bin/env python3
"""Patient-level nested folds and leakage-free, patient-balanced PCA.

The AA embedding bank is a phenotype-blind reusable cache. PCA fitting and
dimension selection use only patients in the corresponding training partition.
Nothing in this module reads the outcome to fit a representation.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import resource
import time

# Set before importing numerical libraries in standalone use.
for _name in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_name] = str(min(16, int(os.environ.get(_name, "16"))))

import numpy as np
import pandas as pd
import pyarrow.dataset as ds
from sklearn.decomposition import PCA
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import NearestNeighbors

METRIC_SPEC = {
    "version": "pca_geometry_v1", "dimensions": [16, 32, 64],
    "raw_neighbor_k": 50, "distance": "euclidean",
    "reference": "exact_brute_force_raw_bank", "min_neighbor_overlap": 0.80,
    "reconstruction_rule": "smallest_dimension_with_mean_mse_within_one_se_of_best",
    "probe_target_rows": 2048,
    "probe_source_policy": "unfitted_training_patient_aa_else_fit_aa_fallback",
    "training_only": True,
}


def _seed(*parts: object) -> int:
    digest = hashlib.sha256("|".join(map(str, parts)).encode()).digest()
    return int.from_bytes(digest[:4], "little")


def _atomic_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    tmp.replace(path)


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def _patient_hash(patients: list[str]) -> str:
    return hashlib.sha256(("\n".join(sorted(patients)) + "\n").encode()).hexdigest()


def _manifest_fingerprints(manifest: Path) -> tuple[str, str]:
    """Hash identities and Home split labels without accessing external labels."""
    identities: list[list[str]] = []
    home_labels: list[list[str]] = []
    with manifest.open(newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        if not {"cohort", "sample_id", "patient_key", "recorded_n_binary"} <= set(reader.fieldnames or []):
            raise ValueError("Manifest lacks required identity or Home split columns")
        for row in reader:
            cohort = row["cohort"].lower()
            identities.append([cohort, row["sample_id"], row["patient_key"]])
            if cohort == "home":
                home_labels.append([row["patient_key"], row["recorded_n_binary"]])
    identity = pd.DataFrame(identities, columns=["cohort", "sample_id", "patient_key"])
    identity = identity.sort_values(["cohort", "sample_id", "patient_key"])
    identity_sha = hashlib.sha256(identity.to_csv(sep="\t", index=False,
                                                  lineterminator="\n").encode("utf-8")).hexdigest()
    def digest(rows: list[list[str]]) -> str:
        return hashlib.sha256(json.dumps(sorted(rows), ensure_ascii=False,
                                         separators=(",", ":")).encode("utf-8")).hexdigest()
    return identity_sha, digest(home_labels)


def make_folds(manifest: Path, out: Path, seed: int = 20260929) -> pd.DataFrame:
    if seed != 20260929:
        raise ValueError("The prospective patient split seed is locked to 20260929")
    # Branch on cohort before accessing the endpoint field. External N is
    # never selected, decoded into a dataframe, or used for split decisions.
    with manifest.open(newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        required = {"cohort", "patient_key", "recorded_n_binary"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"manifest missing columns: {sorted(missing)}")
        home = pd.DataFrame([
            (row["patient_key"], row["recorded_n_binary"])
            for row in reader if row["cohort"].lower() == "home"
        ], columns=["patient_key", "recorded_n_binary"])
    home = home[home["patient_key"].ne("")]
    if home.empty:
        raise ValueError("No Home patients with nonempty patient_key")
    conflicts = home.groupby("patient_key")["recorded_n_binary"].nunique()
    if (conflicts != 1).any():
        raise ValueError(f"Conflicting N labels for patients: {conflicts[conflicts != 1].index.tolist()[:8]}")
    home = home.drop_duplicates("patient_key").sort_values("patient_key").reset_index(drop=True)
    if not home["recorded_n_binary"].isin(["0", "1"]).all():
        raise ValueError("Home recorded_n_binary must contain only 0 or 1")
    y = home["recorded_n_binary"].astype(int).to_numpy()
    counts = np.bincount(y, minlength=2)
    if counts.min() < 5:
        raise ValueError(f"Need >=5 independent Home patients in each class for 5 outer folds: {counts.tolist()}")
    rows = []
    outer = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    for outer_fold, (train_idx, test_idx) in enumerate(outer.split(home, y)):
        inner_assignment: dict[int, int] = {}
        inner_y = y[train_idx]
        if np.bincount(inner_y, minlength=2).min() < 3:
            raise ValueError(f"Outer fold {outer_fold} has <3 train patients in a class")
        inner = StratifiedKFold(n_splits=3, shuffle=True, random_state=_seed(seed, "inner", outer_fold))
        for inner_fold, (_, val_pos) in enumerate(inner.split(train_idx, inner_y)):
            for pos in val_pos:
                inner_assignment[int(train_idx[pos])] = inner_fold
        for i in train_idx:
            rows.append((home.at[i, "patient_key"], int(y[i]), outer_fold, "train", inner_assignment[int(i)]))
        for i in test_idx:
            rows.append((home.at[i, "patient_key"], int(y[i]), outer_fold, "test", pd.NA))
    folds = pd.DataFrame(rows, columns=["patient_key", "recorded_n_binary", "outer_fold", "partition", "inner_fold"])
    folds = folds.sort_values(["outer_fold", "partition", "patient_key"]).reset_index(drop=True)
    out.parent.mkdir(parents=True, exist_ok=True)
    folds.to_csv(out, sep="\t", index=False)
    stage1_path = manifest.parent.parent / "audit/01_ingest_summary.json"
    identity_sha, home_label_sha = _manifest_fingerprints(manifest)
    stage1 = json.loads(stage1_path.read_text()) if stage1_path.exists() else {}
    _atomic_json(out.with_suffix(".json"), {
        "seed": seed,
        "manifest": str(manifest.resolve()),
        "manifest_identity_sha256": identity_sha,
        "home_split_label_sha256": home_label_sha,
        "stage1_source_contract_sha256": stage1.get("source_contract_sha256"),
        "stage1_input_sha256_manifest_sha256": stage1.get("input_sha256_manifest_sha256"),
        "stage1_union_sha256": stage1.get("union_sha256"),
        "folds_sha256": _sha256(out),
        "n_home_patients": int(len(home)),
        "class_counts": {"0": int(counts[0]), "1": int(counts[1])},
        "outer_folds": 5, "inner_folds": 3,
        "rows": int(len(folds)),
        "patient_level": True,
        "external_in_folds": False,
    })
    return folds


def _verify_folds_receipt(root: Path) -> dict:
    folds_path = root / "folds/folds.tsv"
    manifest_path = root / "config/cohort_samples.tsv"
    summary_path = root / "audit/01_ingest_summary.json"
    receipt = json.loads(folds_path.with_suffix(".json").read_text())
    summary = json.loads(summary_path.read_text())
    identity_sha, home_label_sha = _manifest_fingerprints(manifest_path)
    if receipt.get("folds_sha256") != _sha256(folds_path) or\
            receipt.get("manifest_identity_sha256") != identity_sha or\
            receipt.get("home_split_label_sha256") != home_label_sha or\
            summary.get("status") != "PASS" or\
            receipt.get("stage1_source_contract_sha256") != summary.get("source_contract_sha256") or\
            receipt.get("stage1_input_sha256_manifest_sha256") != summary.get("input_sha256_manifest_sha256") or\
            receipt.get("stage1_union_sha256") != summary.get("union_sha256") or\
            summary.get("input_sha256_manifest_sha256") != _sha256(root / "audit/01_input_sha256.tsv") or\
            receipt.get("external_in_folds") is not False or\
            receipt.get("n_home_patients") != 114:
        raise ValueError("Fold receipt or stage1 patient identity source changed")
    return receipt


def _patient_aa_source_fingerprint(root: Path) -> str:
    paths = sorted((root / "data/patient_aa").glob("*.parquet"))
    if len(paths) != 279:
        raise ValueError("Patient AA source file count differs from Stage1 279 libraries")
    records = [(p.name, _sha256(p)) for p in paths]
    return hashlib.sha256(json.dumps(records, separators=(",", ":")).encode()).hexdigest()


def train_patients(folds: pd.DataFrame, outer_fold: int, inner_fold: int | None) -> list[str]:
    frame = folds.loc[folds["outer_fold"].astype(int).eq(outer_fold)]
    if frame.empty:
        raise ValueError(f"Missing outer fold {outer_fold}")
    train = frame.loc[frame["partition"].eq("train")].copy()
    if inner_fold is not None:
        train = train.loc[~train["inner_fold"].astype(int).eq(inner_fold)]
    return sorted(train["patient_key"].astype(str).tolist())


def _patient_sequences(patient_dir: Path, patients: set[str], aa_to_idx: dict[str, int]) -> dict[str, np.ndarray]:
    result: dict[str, set[int]] = {p: set() for p in patients}
    dataset = ds.dataset(str(patient_dir), format="parquet")
    missing = {"patient_key", "aa", "cohort"} - set(dataset.schema.names)
    if missing:
        raise ValueError(f"patient_aa files missing columns: {sorted(missing)}")
    scanner = dataset.scanner(columns=["patient_key", "aa"], filter=(ds.field("cohort") == "home"), batch_size=131072)
    for batch in scanner.to_batches():
        keys = batch.column(0).to_pylist()
        aas = batch.column(1).to_pylist()
        for patient, aa in zip(keys, aas):
            if patient in result:
                idx = aa_to_idx.get(aa)
                if idx is not None:
                    result[patient].add(idx)
    missing_patients = [p for p, seqs in result.items() if not seqs]
    if missing_patients:
        raise ValueError(f"Train patients have no QC-passing indexed AA: {missing_patients[:8]}")
    return {p: np.fromiter(sorted(seqs), dtype=np.int64) for p, seqs in result.items()}


def balanced_draws(per_patient: dict[str, np.ndarray], k: int, seed: int, scope: str) -> pd.DataFrame:
    if k <= 0 or k > 4096:
        raise ValueError("k must be 1..4096")
    rows = []
    for patient in sorted(per_patient):
        idx = per_patient[patient]
        rng = np.random.default_rng(_seed(seed, scope, patient))
        ceiling = 4096
        first = rng.choice(idx, size=min(ceiling, len(idx)), replace=False)
        if len(first) < ceiling:
            first = np.concatenate([first, rng.choice(idx, size=ceiling - len(first), replace=True)])
        draw = first[:k]
        rows.extend((patient, int(seq_idx), i, len(idx) < ceiling) for i, seq_idx in enumerate(draw))
    return pd.DataFrame(rows, columns=["patient_key", "seq_idx", "draw_index", "replacement_pool_lt_4096"])


def _metric_rows(per_patient: dict[str, np.ndarray], draws: pd.DataFrame, bank: np.ndarray,
                 seed: int, scope: str, max_points: int) -> tuple[np.ndarray, str]:
    used = set(draws["seq_idx"].astype(int).tolist())
    rng = np.random.default_rng(_seed(seed, scope, "metric"))
    # Prefer sequences not selected for fitting. Sampling is label-blind.
    candidates = []
    per_limit = max(1, max_points // max(1, len(per_patient)))
    for patient in sorted(per_patient):
        available = np.fromiter((int(i) for i in per_patient[patient] if int(i) not in used), dtype=np.int64)
        if len(available):
            candidates.extend(rng.choice(available, size=min(per_limit, len(available)), replace=False))
    if len(candidates) >= 128:
        chosen = np.asarray(candidates[:max_points], dtype=np.int64)
        source = "train_patient_unfitted_aa"
    else:
        unique_fit = np.unique(draws["seq_idx"].to_numpy(dtype=np.int64))
        chosen = rng.choice(unique_fit, size=min(max_points, len(unique_fit)), replace=False)
        source = "fit_aa_fallback"
    return np.asarray(bank[chosen], dtype=np.float32), source


def selection_metrics(pca: PCA, probe: np.ndarray, dims: tuple[int, ...],
                      neighbors: int, min_overlap: float) -> dict:
    """Exact top-k preservation plus the prespecified one-SE reconstruction rule."""
    if len(probe) < neighbors + 2:
        return {
            "selected_dim": None, "abstain": True, "reason": "insufficient_probe_rows",
            "thresholds": {"neighbor_overlap": min_overlap,
                           "reconstruction": "mean per-row MSE within one SE of best"},
            "by_dim": {},
        }
    k = min(neighbors, len(probe) - 1)
    raw_idx = NearestNeighbors(n_neighbors=k + 1, metric="euclidean",
                               algorithm="brute").fit(probe).kneighbors(
                                   probe, return_distance=False)
    raw_sets = [set(row[row != i][:k]) for i, row in enumerate(raw_idx)]
    transformed = pca.transform(probe)
    centered = probe - pca.mean_
    by_dim = {}
    for dim in dims:
        z = transformed[:, :dim]
        idx = NearestNeighbors(n_neighbors=k + 1, metric="euclidean",
                               algorithm="brute").fit(z).kneighbors(
                                   z, return_distance=False)
        overlap = float(np.mean([
            len(raw_sets[i] & set(row[row != i][:k])) / k for i, row in enumerate(idx)]))
        reconstruct = z @ pca.components_[:dim, :]
        per_row_mse = np.mean(np.square(centered - reconstruct, dtype=np.float64), axis=1)
        mean_mse = float(per_row_mse.mean())
        se_mse = float(per_row_mse.std(ddof=1) / np.sqrt(len(per_row_mse)))
        by_dim[str(dim)] = {
            "top50_neighbor_overlap": overlap,
            "reconstruction_mse": mean_mse,
            "reconstruction_mse_se": se_mse,
        }
    best_dim = min(dims, key=lambda d: by_dim[str(d)]["reconstruction_mse"])
    best = by_dim[str(best_dim)]
    one_se_cutoff = best["reconstruction_mse"] + best["reconstruction_mse_se"]
    selected = None
    for dim in dims:
        item = by_dim[str(dim)]
        item["within_one_se_of_best"] = bool(item["reconstruction_mse"] <= one_se_cutoff)
        if selected is None and item["top50_neighbor_overlap"] >= min_overlap and item["within_one_se_of_best"]:
            selected = dim
    return {
        "selected_dim": selected,
        "abstain": selected is None,
        "reason": None if selected is not None else "no_dimension_met_prespecified_thresholds",
        "thresholds": {
            "neighbor_overlap": min_overlap,
            "reconstruction": "mean per-row MSE within one SE of best dimension",
            "best_dim": best_dim, "one_se_mse_cutoff": one_se_cutoff,
        },
        "neighbor_k": k,
        "probe_rows": int(len(probe)),
        "neighbor_reference": "exact brute-force Euclidean top-50 in raw embedding bank",
        "by_dim": by_dim,
    }


def _validate_selection_contract(selection: dict) -> None:
    """Reject a changed metric or a hand-edited dimension decision."""
    if selection.get("metric_spec") != METRIC_SPEC or selection.get("neighbor_k") != 50 or\
            int(selection.get("probe_rows", 0)) < 52 or\
            selection.get("probe_source") not in ("train_patient_unfitted_aa", "fit_aa_fallback"):
        raise ValueError("PCA dimension selection metric specification changed")
    by_dim = selection.get("by_dim", {})
    if set(by_dim) != {"16", "32", "64"}:
        raise ValueError("PCA dimension selection is missing a dimension")
    best_dim = min((16, 32, 64), key=lambda dim: float(by_dim[str(dim)]["reconstruction_mse"]))
    cutoff = float(by_dim[str(best_dim)]["reconstruction_mse"]) +\
        float(by_dim[str(best_dim)]["reconstruction_mse_se"])
    thresholds = selection.get("thresholds", {})
    if thresholds.get("neighbor_overlap") != 0.80 or thresholds.get("best_dim") != best_dim or\
            not np.isclose(float(thresholds.get("one_se_mse_cutoff", -1)), cutoff, rtol=1e-8, atol=1e-10):
        raise ValueError("PCA one-SE threshold was changed")
    expected = next((dim for dim in (16, 32, 64)
                     if float(by_dim[str(dim)]["top50_neighbor_overlap"]) >= 0.80 and
                     float(by_dim[str(dim)]["reconstruction_mse"]) <= cutoff), None)
    if selection.get("selected_dim") != expected or selection.get("abstain") != (expected is None):
        raise ValueError("PCA selected dimension differs from locked metric decision")


def fit_pca(root: Path, model: str, outer_fold: int, inner_fold: int | None = None,
            k: int = 4096, n_components: int = 64, seed: int = 20260929,
            min_overlap: float = 0.80, metric_rows: int = 2048,
            project_all: bool = False, draw_replicate: int = 0) -> Path:
    started = time.monotonic()
    if not 0 <= outer_fold < 5 or (inner_fold is not None and not 0 <= inner_fold < 3):
        raise ValueError("outer_fold must be 0..4 and inner_fold 0..2")
    if n_components != 64:
        raise ValueError("Fit 64 components once; 16 and 32 are prefix projections")
    if min_overlap != 0.80 or metric_rows != 2048 or seed != 20260929:
        raise ValueError("PCA metric specification is locked: top50 >=0.80, 2048 target probe rows, seed 20260929")
    if not 0 <= draw_replicate <= 2 or (k == 4096 and draw_replicate != 0):
        raise ValueError("Use replicate 0 for K=4096 and replicates 0..2 for K=2048")
    folds_file = root / "folds/folds.tsv"
    folds_receipt = _verify_folds_receipt(root)
    source_fingerprint = _patient_aa_source_fingerprint(root)
    folds = pd.read_csv(folds_file, sep="\t", dtype={"patient_key": str})
    patients = train_patients(folds, outer_fold, inner_fold)
    scope = f"outer_{outer_fold}" + (f"_inner_{inner_fold}" if inner_fold is not None else "")
    union_path = root / "data/aa_union.parquet"
    union_sha = _sha256(union_path)
    union = pd.read_parquet(union_path, columns=["seq_idx", "aa"])
    seq_idx = union["seq_idx"].to_numpy(dtype=np.int64)
    if not np.array_equal(np.sort(seq_idx), np.arange(len(union), dtype=np.int64)):
        raise ValueError("AA union seq_idx must be a contiguous zero-based bank index")
    aa_to_idx = dict(zip(union["aa"].astype(str), seq_idx))
    del union
    bank_path = root / "embeddings" / model / "bank_fp16.npy"
    bank_meta_path = bank_path.parent / "meta.json"
    bank_meta = json.loads(bank_meta_path.read_text())
    if bank_meta.get("status") != "complete" or not bank_meta.get("bank_sha256"):
        raise ValueError("Embedding bank metadata is incomplete or lacks SHA256")
    if bank_meta.get("input_sha256") != union_sha:
        raise ValueError("AA union content hash differs from embedding bank input")
    bank = np.load(bank_path, mmap_mode="r")
    if (list(bank.shape) != list(bank_meta.get("bank_shape", [])) or
            bank_meta.get("bank_dtype") != "float16" or
            int(bank_meta.get("n_total", -1)) != len(aa_to_idx)):
        raise ValueError("Embedding bank disagrees with completion metadata")
    if bank_meta.get("bank_size_bytes") is not None and bank_path.stat().st_size != bank_meta["bank_size_bytes"]:
        raise ValueError("Embedding bank file size changed after completion")
    if bank_meta.get("bank_mtime_ns") is not None and bank_path.stat().st_mtime_ns != bank_meta["bank_mtime_ns"]:
        raise ValueError("Embedding bank mtime changed after completion")
    if bank.ndim != 2 or bank.shape[0] != len(aa_to_idx):
        raise ValueError(f"Embedding bank shape {bank.shape} incompatible with union {len(aa_to_idx)}")
    per_patient = _patient_sequences(root / "data/patient_aa", set(patients), aa_to_idx)
    draw_scope = scope if k == 4096 else f"{scope}|k2048|rep{draw_replicate}"
    draws = balanced_draws(per_patient, k=k, seed=seed, scope=draw_scope)
    indices = draws["seq_idx"].to_numpy(dtype=np.int64)
    X = np.asarray(bank[indices], dtype=np.float32)
    if not np.isfinite(X).all():
        raise ValueError("Nonfinite embeddings in PCA training draw")
    if X.shape[0] <= n_components:
        raise ValueError("Too few training embedding rows for PCA")
    pca = PCA(n_components=n_components, svd_solver="randomized", random_state=_seed(seed, scope, model))
    pca.fit(X)
    del X
    probe, metric_source = _metric_rows(per_patient, draws, bank, seed, draw_scope, metric_rows)
    metrics = selection_metrics(pca, probe, (16, 32, 64), 50, min_overlap)
    metrics["probe_source"] = metric_source
    metrics["metric_spec"] = METRIC_SPEC
    if metrics.get("reason") != "insufficient_probe_rows":
        _validate_selection_contract(metrics)
    if source_fingerprint != _patient_aa_source_fingerprint(root) or\
            folds_receipt["folds_sha256"] != _sha256(folds_file):
        raise ValueError("Patient AA source or fold table changed during PCA fit")
    output = root / "pca" / model
    if k != 4096:
        output = output / f"stability_k{k}" / f"rep_{draw_replicate}"
    output = output / f"outer_{outer_fold}"
    if inner_fold is not None:
        output = output / f"inner_{inner_fold}"
    output.mkdir(parents=True, exist_ok=True)
    np.savez(output / "pca.npz", mean=pca.mean_.astype(np.float32),
             components=pca.components_.astype(np.float32),
             explained_variance_ratio=pca.explained_variance_ratio_.astype(np.float32),
             explained_variance=pca.explained_variance_.astype(np.float32),
             singular_values=pca.singular_values_.astype(np.float32))
    draws.to_parquet(output / "draws.parquet", index=False)
    _atomic_json(output / "metadata.json", {
        "model": model, "outer_fold": outer_fold, "inner_fold": inner_fold,
        "k_per_patient": k, "draw_replicate": draw_replicate,
        "components": 64, "seed": seed,
        "fit_patient_ids": patients,
        "train_patient_sha256": _patient_hash(patients),
        "fit_patient_count": len(patients),
        "patient_unique_aa_counts": {p: int(len(per_patient[p])) for p in patients},
        "undersized_patients": {p: int(len(per_patient[p])) for p in patients if len(per_patient[p]) < 4096},
        "sampling_strategy": "without replacement first; then replacement to fixed equal draw count when unique AA < 4096",
        "replacement_draw_rows": int(sum(max(0, k - len(per_patient[p])) for p in patients)),
        "draw_rows": int(len(draws)), "draws_parquet": "draws.parquet",
        "bank_path": str(bank_path.resolve()),
        "bank_sha256_from_meta": bank_meta["bank_sha256"],
        "bank_input_sha256": bank_meta.get("input_sha256"),
        "aa_union_sha256": union_sha,
        "bank_model_fingerprint": bank_meta.get("model_fingerprint"),
        "folds_sha256": _sha256(folds_file),
        "folds_receipt_sha256": _sha256(folds_file.with_suffix(".json")),
        "manifest_identity_sha256": folds_receipt["manifest_identity_sha256"],
        "home_split_label_sha256": folds_receipt["home_split_label_sha256"],
        "stage1_source_contract_sha256": folds_receipt["stage1_source_contract_sha256"],
        "stage1_input_sha256_manifest_sha256": folds_receipt["stage1_input_sha256_manifest_sha256"],
        "stage1_union_sha256": folds_receipt["stage1_union_sha256"],
        "patient_aa_source_fingerprint": source_fingerprint,
        "elapsed_seconds_through_dimension_selection": time.monotonic() - started,
        "rss_ru_maxrss_kib_through_dimension_selection": resource.getrusage(
            resource.RUSAGE_SELF).ru_maxrss,
        "label_usage": "Only stratified split metadata; no outcome used by PCA fit or dimension selection",
    })
    _atomic_json(output / "selection.json", metrics)
    if project_all:
        out = np.lib.format.open_memmap(output / "projection64.npy", mode="w+",
                                        dtype=np.float32, shape=(bank.shape[0], 64))
        for start in range(0, len(bank), 32768):
            end = min(start + 32768, len(bank))
            out[start:end] = pca.transform(np.asarray(bank[start:end], dtype=np.float32))
        out.flush()
        del out
    return output


def summarize_2048_stability(root: Path, model: str, outer_fold: int,
                             min_subspace_overlap: float = 0.90,
                             inner_fold: int | None = None) -> Path:
    """Compare three independent K=2048 draws with the K=4096 primary fit."""
    if not 0 <= outer_fold < 5 or (inner_fold is not None and not 0 <= inner_fold < 3):
        raise ValueError("outer_fold must be 0..4 and inner_fold 0..2")
    if min_subspace_overlap != 0.90:
        raise ValueError("K=2048 subspace stability threshold is locked at 0.90")
    folds_receipt = _verify_folds_receipt(root)
    source_fingerprint = _patient_aa_source_fingerprint(root)
    primary = root / "pca" / model / f"outer_{outer_fold}"
    folders = [root / "pca" / model / "stability_k2048" / f"rep_{r}" /
               f"outer_{outer_fold}" for r in range(3)]
    if inner_fold is not None:
        primary = primary / f"inner_{inner_fold}"
        folders = [folder / f"inner_{inner_fold}" for folder in folders]
    p_meta = json.loads((primary / "metadata.json").read_text())
    folds_path = root / "folds/folds.tsv"
    current_folds_sha = _sha256(folds_path)
    expected_patients = train_patients(pd.read_csv(folds_path, sep="\t",
                                                   dtype={"patient_key": str}),
                                       outer_fold, inner_fold)
    if p_meta["fit_patient_ids"] != expected_patients or\
            p_meta["folds_sha256"] != current_folds_sha or\
            p_meta["outer_fold"] != outer_fold or p_meta["inner_fold"] != inner_fold or\
            p_meta.get("model") != model or p_meta.get("k_per_patient") != 4096 or\
            p_meta.get("draw_replicate") != 0 or\
            p_meta.get("train_patient_sha256") != _patient_hash(expected_patients) or\
            p_meta.get("folds_receipt_sha256") != _sha256(folds_path.with_suffix(".json")) or\
            p_meta.get("manifest_identity_sha256") != folds_receipt["manifest_identity_sha256"] or\
            p_meta.get("home_split_label_sha256") != folds_receipt["home_split_label_sha256"] or\
            p_meta.get("stage1_source_contract_sha256") != folds_receipt["stage1_source_contract_sha256"] or\
            p_meta.get("stage1_input_sha256_manifest_sha256") != folds_receipt["stage1_input_sha256_manifest_sha256"] or\
            p_meta.get("stage1_union_sha256") != folds_receipt["stage1_union_sha256"] or\
            p_meta.get("patient_aa_source_fingerprint") != source_fingerprint:
        raise ValueError("PCA primary fit scope or fold table changed")
    p_sel = json.loads((primary / "selection.json").read_text())
    _validate_selection_contract(p_sel)
    p_comp = np.load(primary / "pca.npz")["components"]
    if p_comp.shape[0] != 64:
        raise ValueError("PCA primary components are not the locked 64D basis")
    primary_hashes = {name: _sha256(primary / name) for name in
                      ("pca.npz", "selection.json", "metadata.json")}
    scores = {}
    replicate_hashes = {}
    selected = []
    for repeat, folder in enumerate(folders):
        meta = json.loads((folder / "metadata.json").read_text())
        sel = json.loads((folder / "selection.json").read_text())
        _validate_selection_contract(sel)
        if meta["fit_patient_ids"] != p_meta["fit_patient_ids"] or\
                meta["folds_sha256"] != current_folds_sha or\
                meta["outer_fold"] != outer_fold or meta["inner_fold"] != inner_fold or\
                meta["bank_sha256_from_meta"] != p_meta["bank_sha256_from_meta"] or\
                meta.get("model") != model or meta.get("k_per_patient") != 2048 or\
                meta.get("draw_replicate") != repeat or\
                meta.get("train_patient_sha256") != p_meta["train_patient_sha256"] or\
                meta.get("folds_receipt_sha256") != p_meta["folds_receipt_sha256"] or\
                meta.get("manifest_identity_sha256") != p_meta["manifest_identity_sha256"] or\
                meta.get("home_split_label_sha256") != p_meta["home_split_label_sha256"] or\
                meta.get("stage1_source_contract_sha256") != p_meta["stage1_source_contract_sha256"] or\
                meta.get("stage1_input_sha256_manifest_sha256") != p_meta["stage1_input_sha256_manifest_sha256"] or\
                meta.get("stage1_union_sha256") != p_meta["stage1_union_sha256"] or\
                meta.get("patient_aa_source_fingerprint") != source_fingerprint:
            raise ValueError(f"K=2048 repeat {repeat} has different train patients or embedding bank")
        comp = np.load(folder / "pca.npz")["components"]
        if comp.shape != p_comp.shape:
            raise ValueError(f"K=2048 repeat {repeat} has a different component shape")
        replicate_hashes[str(repeat)] = {name: _sha256(folder / name) for name in
                                         ("pca.npz", "selection.json", "metadata.json")}
        overlap = {}
        for dim in (16, 32, 64):
            singular = np.linalg.svd(p_comp[:dim] @ comp[:dim].T,
                                     compute_uv=False)
            overlap[str(dim)] = float(np.sum(singular**2) / dim)
        scores[str(repeat)] = {
            "subspace_overlap_vs_k4096": overlap,
            "selected_dim": sel["selected_dim"],
            "abstain": sel["abstain"],
            "top50_neighbor_overlap_by_dim": {
                dim: sel["by_dim"][dim]["top50_neighbor_overlap"] for dim in ("16", "32", "64")},
        }
        selected.append(sel["selected_dim"])
    chosen_key = str(p_sel["selected_dim"]) if p_sel["selected_dim"] is not None else None
    passed = bool(
        not p_sel["abstain"] and
        all(dim == p_sel["selected_dim"] for dim in selected) and
        all(scores[str(r)]["subspace_overlap_vs_k4096"][chosen_key] >= min_subspace_overlap
            for r in range(3))
    )
    path = primary / "stability.json"
    _atomic_json(path, {
        "model": model, "outer_fold": outer_fold, "inner_fold": inner_fold,
        "primary_k": 4096, "repeat_k": 2048, "repeats": 3,
        "fit_patient_ids": p_meta["fit_patient_ids"],
        "train_patient_sha256": _patient_hash(p_meta["fit_patient_ids"]),
        "folds_sha256": current_folds_sha,
        "bank_sha256_from_meta": p_meta["bank_sha256_from_meta"],
        "metric_spec": METRIC_SPEC,
        "manifest_identity_sha256": p_meta["manifest_identity_sha256"],
        "home_split_label_sha256": p_meta["home_split_label_sha256"],
        "stage1_source_contract_sha256": p_meta["stage1_source_contract_sha256"],
        "stage1_input_sha256_manifest_sha256": p_meta["stage1_input_sha256_manifest_sha256"],
        "stage1_union_sha256": p_meta["stage1_union_sha256"],
        "patient_aa_source_fingerprint": source_fingerprint,
        "folds_receipt_sha256": p_meta["folds_receipt_sha256"],
        "primary_artifact_sha256s": primary_hashes,
        "replicate_artifact_sha256s": replicate_hashes,
        "primary_selected_dim": p_sel["selected_dim"],
        "min_subspace_overlap_at_selected_dim": min_subspace_overlap,
        "selected_dim_for_stability": p_sel["selected_dim"],
        "all_replicates_match_primary_dim": all(dim == p_sel["selected_dim"] for dim in selected),
        "stability_pass": passed,
        "replicate_metrics": scores,
        "meaning": "unlabeled PCA geometry stability only; no pN association or predictive validation",
    })
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    sub = parser.add_subparsers(dest="command", required=True)
    f = sub.add_parser("folds")
    f.add_argument("--seed", type=int, default=20260929)
    p = sub.add_parser("fit-pca")
    p.add_argument("--model", choices=["tcrbert_dualtask_aa_mean", "tcrbert_mlm", "sceptr_cdr3_only"], required=True)
    p.add_argument("--outer-fold", type=int, required=True)
    p.add_argument("--inner-fold", type=int)
    p.add_argument("--k", type=int, choices=[2048, 4096], default=4096)
    p.add_argument("--seed", type=int, default=20260929)
    p.add_argument("--min-neighbor-overlap", type=float, default=0.80)
    p.add_argument("--draw-replicate", type=int, choices=[0, 1, 2], default=0)
    p.add_argument("--metric-rows", type=int, default=2048)
    p.add_argument("--project-all", action="store_true")
    q = sub.add_parser("stability")
    q.add_argument("--model", choices=["tcrbert_dualtask_aa_mean", "tcrbert_mlm", "sceptr_cdr3_only"], required=True)
    q.add_argument("--outer-fold", type=int, required=True)
    q.add_argument("--inner-fold", type=int)
    q.add_argument("--min-subspace-overlap", type=float, default=0.90)
    args = parser.parse_args()
    root = args.root.resolve()
    if args.command == "folds":
        result = make_folds(root / "config/cohort_samples.tsv", root / "folds/folds.tsv", args.seed)
        print(json.dumps({"fold_rows": len(result), "patients": result.patient_key.nunique()}))
    elif args.command == "fit-pca":
        result = fit_pca(root, args.model, args.outer_fold, args.inner_fold, args.k, seed=args.seed,
                         min_overlap=args.min_neighbor_overlap,
                         metric_rows=args.metric_rows, project_all=args.project_all,
                         draw_replicate=args.draw_replicate)
        print(json.dumps({"pca_output": str(result)}))
    else:
        result = summarize_2048_stability(root, args.model, args.outer_fold,
                                          args.min_subspace_overlap, args.inner_fold)
        print(json.dumps({"stability_output": str(result)}))


if __name__ == "__main__":
    main()
