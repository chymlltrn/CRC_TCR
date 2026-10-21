#!/usr/bin/env python3
"""Patient-level nested MIL on full-repertoire TCR embeddings.

All data and outputs are rooted at 0929_caseEmbedding. The command refuses to
train until complete patient index, embedding bank, and fold-local PCA inputs
exist. No external-cohort labels enter model selection.
"""
from __future__ import annotations
from release_runtime import private_literal as _private_literal, release_path as _release_path

import argparse
import fcntl
from contextlib import contextmanager
import hashlib
import json
import math
import os
import sys
import time
from datetime import datetime, timezone
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import expit
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score
from sklearn.model_selection import StratifiedShuffleSplit
import torch
import psutil
from threadpoolctl import threadpool_limits
from torch import nn

ROOT = Path(_release_path('path_1304218e86d2e596'))
MODELS = ("tcrbert_dualtask_aa_mean", "sceptr_cdr3_only")
BAGS = (2048, 4096)
SEEDS = (17, 29, 43)
LOCKED_MAX_EPOCHS = 40
LOCKED_PATIENCE = 6
LOCKED_GPU_GIB = 12.0
EARLY_STOP_FRACTION = 0.2
PCA_METRIC_SPEC = {
    "version": "pca_geometry_v1",
    "dimensions": [16, 32, 64],
    "raw_neighbor_k": 50,
    "distance": "euclidean",
    "reference": "exact_brute_force_raw_bank",
    "min_neighbor_overlap": 0.80,
    "reconstruction_rule": "smallest_dimension_with_mean_mse_within_one_se_of_best",
    "probe_target_rows": 2048,
    "probe_source_policy": "unfitted_training_patient_aa_else_fit_aa_fallback",
    "training_only": True,
}


def set_threads(n: int = 8) -> None:
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[name] = str(n)
    threadpool_limits(limits=n)
    torch.set_num_threads(n)
    torch.set_num_interop_threads(min(4, n))



def check_rss(limit_gib: float = 128.0) -> float:
    rss = psutil.Process().memory_info().rss / 1024**3
    if rss > limit_gib:
        raise RuntimeError(f"CPU RSS {rss:.2f} GiB > policy {limit_gib:.2f} GiB")
    return rss


@contextmanager
def gpu_slot(root: Path, device: str, max_gpu_gib: float):
    if device != "cuda":
        yield
        return
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    lock_path = root / ".gpu.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(lock_path, os.O_CREAT | os.O_RDWR, 0o664)
    try:
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError("MIL GPU slot already held") from exc
        total = torch.cuda.get_device_properties(0).total_memory
        fraction = min(0.75, max_gpu_gib * 1024**3 / total)
        torch.cuda.set_per_process_memory_fraction(fraction, 0)
        yield
    finally:
        fcntl.flock(fd, fcntl.LOCK_UN)
        os.close(fd)


def atomic_json(path: Path, obj: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w") as fh:
        json.dump(obj, fh, indent=2, sort_keys=True, default=str)
    tmp.replace(path)


def stable_seed(*parts: object) -> int:
    raw = "|".join(map(str, parts)).encode()
    return int.from_bytes(hashlib.sha256(raw).digest()[:8], "little") % (2**32 - 1)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def execution_config(device: str, cpu_threads: int, max_gpu_gib: float) -> dict:
    """The scientific and runtime settings bound to every resumed MIL fit."""
    if (device not in ("cpu", "cuda") or not isinstance(cpu_threads, int) or
            not 1 <= cpu_threads <= 16 or not isinstance(max_gpu_gib, (int, float)) or
            not math.isfinite(max_gpu_gib) or not 0 < max_gpu_gib <= LOCKED_GPU_GIB):
        raise ValueError("MIL device, CPU threads or 12 GiB GPU limit invalid")
    return {"device": device, "cpu_threads": cpu_threads,
            "max_gpu_gib": float(max_gpu_gib), "max_epochs": LOCKED_MAX_EPOCHS,
            "patience": LOCKED_PATIENCE, "patient_batch": 2,
            "early_stop_mc_bags": 2, "early_stop_fraction": EARLY_STOP_FRACTION,
            "heldout_mc_bags": 8, "heldout_mc_escalation_bags": 32,
            "heldout_mc_sd_threshold": 0.05,
            "optimizer": "AdamW", "learning_rate": 3e-4,
            "weight_decay": 1e-3, "bag_draw": "draw-v1"}


def validate_execution_config(config: dict) -> None:
    if not isinstance(config, dict):
        raise ValueError("MIL execution settings absent")
    expected = execution_config(config.get("device"), config.get("cpu_threads"),
                                config.get("max_gpu_gib"))
    if config != expected:
        raise ValueError("MIL execution settings differ from locked contract")


def early_stop_split(train: list[str], labels: dict[str, int], outer: int,
                     inner: int, seed: int) -> tuple[list[str], list[str]]:
    """Choose the epoch within the inner training patients only."""
    if not train or inner not in range(3) or len(train) != len(set(train)):
        raise ValueError("Invalid inner training patients for early stopping")
    patients = np.array(sorted(train), dtype=str)
    y = np.array([labels[k] for k in patients], dtype=int)
    n_stop = math.ceil(len(patients) * EARLY_STOP_FRACTION)
    if (set(y) != {0, 1} or min(np.bincount(y)) < 2 or
            n_stop < 2 or len(patients) - n_stop < 2):
        raise ValueError("Early stopping requires both labels in both partitions")
    splitter = StratifiedShuffleSplit(n_splits=1, test_size=EARLY_STOP_FRACTION,
                                      random_state=stable_seed("early-stop-v1", outer, inner, seed))
    fit_idx, stop_idx = next(splitter.split(patients, y))
    fit, stop = patients[fit_idx].tolist(), patients[stop_idx].tolist()
    if (set(fit) & set(stop) or set(fit + stop) != set(train) or
            {labels[k] for k in fit} != {0, 1} or
            {labels[k] for k in stop} != {0, 1}):
        raise ValueError("Early stopping split did not partition inner training patients")
    return fit, stop



LOCKED_MIL_RULES_SHA256 = "16749330e863c9a95f83ca1b1b4fc1678eee9fb15c7203b41cefb9534d904655"


def load_decision_rules(root: Path) -> tuple[dict, str]:
    path = root / "config/mil_decision_rules.json"
    digest = sha256(path)
    if digest != LOCKED_MIL_RULES_SHA256:
        raise RuntimeError(f"MIL decision rules changed after preregistration: {path}")
    rules = json.loads(path.read_text())
    if rules.get("version") != 1 or not rules.get("frozen_before_outcome_run"):
        raise ValueError("MIL decision rules invalid")
    return rules, digest


def file_stamp(path: Path) -> dict:
    st = path.stat()
    return {"path": str(path.resolve()), "bytes": st.st_size, "mtime_ns": st.st_mtime_ns}


def run_fingerprint(root: Path, outer: int, inner: int | None, model: str, bag: int,
                    seed: int, bank: np.ndarray, train: list[str], evaluation: list[str],
                    labels: dict[str, int], execution: dict,
                    fixed_epochs: int | None = None) -> str:
    validate_execution_config(execution)
    bank_path = Path(bank.filename)
    pca_dir = root / "pca" / model / f"outer_{outer}"
    if inner is not None:
        pca_dir /= f"inner_{inner}"
    bank_meta = root / "embeddings" / model / "meta.json"
    payload = {
        "version": 5, "outer": outer, "inner": inner, "model": model, "bag": bag, "seed": seed,
        "code_sha256": sha256(Path(__file__)), "execution_config": execution,
        "fixed_epochs": fixed_epochs,
        "decision_rules_sha256": load_decision_rules(root)[1],
        "train": sorted((k, labels[k]) for k in train),
        "eval": sorted((k, labels[k]) for k in evaluation),
        "manifest_identity_sha": manifest_identity_sha256(root),
        "folds_sha": sha256(root / "folds/folds.tsv"),
        "patient_index_receipt_sha": sha256(root / "mil/patient_index/receipt.json"),
        "bank_integrity_receipt_sha": sha256(root / "mil/bank_integrity.json"),
        "pca_sha": sha256(pca_dir / "pca.npz"),
        "pca_selection_sha": sha256(pca_dir / "selection.json"),
        "scope_pca_stability_sha": sha256(pca_dir / "stability.json"),
        "pca_metadata_sha": sha256(pca_dir / "metadata.json"),
        "bank_stamp": file_stamp(bank_path),
        "raw_bank_meta_sha": sha256(bank_meta) if bank_meta.exists() else None,
    }
    if inner is not None:
        fit, stop = early_stop_split(train, labels, outer, inner, seed)
        payload["early_stop_fit_sha256"] = train_patient_hash(fit)
        payload["early_stop_validation_sha256"] = train_patient_hash(stop)
    else:
        if fixed_epochs is None or not 1 <= fixed_epochs <= LOCKED_MAX_EPOCHS:
            raise ValueError("Final fit needs a frozen epoch count from inner training")
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def valid_saved_prediction(path: Path, expected_keys: list[str], labels: dict[str, int],
                           model: str, bag: int, seed: int, outer: int,
                           inner: int | None) -> pd.DataFrame | None:
    try:
        frame = pd.read_csv(path, sep="\t", dtype={"patient_key": str})
        if len(frame) != len(expected_keys) or set(frame.patient_key) != set(expected_keys) or frame.patient_key.duplicated().any():
            return None
        if not pd.to_numeric(frame.label, errors="coerce").eq(frame.patient_key.map(labels)).all():
            return None
        if not frame.model.eq(model).all() or not frame.bag_size.eq(bag).all() or not frame.seed.eq(seed).all():
            return None
        if not frame.outer_fold.eq(outer).all():
            return None
        if inner is not None and not frame.inner_fold.eq(inner).all():
            return None
        if not np.isfinite(frame.prediction).all() or not frame.prediction.between(0, 1).all():
            return None
        return frame
    except (OSError, ValueError, KeyError):
        return None


def read_manifest(root: Path = ROOT) -> pd.DataFrame:
    """Read identity only; external recorded N is reserved for frozen scoring."""
    p = root / "config/cohort_samples.tsv"
    if not p.exists():
        raise FileNotFoundError(f"Missing sample manifest: {p}")
    fields = ["sample_id", "patient_key", "cohort"]
    df = pd.read_csv(p, sep="\t", usecols=fields, dtype=str)
    if df[fields].isna().any().any() or df.sample_id.duplicated().any():
        raise ValueError("Invalid or repeated sample identity in manifest")
    if df.groupby("patient_key").cohort.nunique().gt(1).any():
        raise ValueError("Patient key reused across Home and external cohorts")
    if not set(df.cohort).issubset({"home", "external"}) or "home" not in set(df.cohort):
        raise ValueError("Unexpected manifest cohort")
    if df.loc[df.cohort.eq("home"), "patient_key"].nunique() != 114:
        raise ValueError(_private_literal('private_ea693679178bbe32'))
    return df


def manifest_identity_sha256(root: Path) -> str:
    """Fingerprint the permitted identity columns, excluding all N labels."""
    identity = read_manifest(root)[["cohort", "sample_id", "patient_key"]]
    canonical = identity.sort_values(["cohort", "sample_id", "patient_key"]).to_csv(
        sep="\t", index=False, lineterminator="\n")
    return hashlib.sha256(canonical.encode()).hexdigest()


def read_home_labels(root: Path, manifest: pd.DataFrame) -> dict[str, int]:
    """Home-only audit labels, with no external N column read."""
    path = root / "config/home_clinical_covariates.tsv"
    df = pd.read_csv(path, sep="\t", usecols=["patient_key", "recorded_n_binary"],
                     dtype={"patient_key": str})
    home = set(manifest.loc[manifest.cohort.eq("home"), "patient_key"])
    if df.patient_key.duplicated().any() or set(df.patient_key) != home:
        raise ValueError("Home clinical labels do not match Home manifest identities")
    if not df.recorded_n_binary.isin([0, 1]).all():
        raise ValueError("Home clinical labels must be binary")
    labels = dict(zip(df.patient_key, df.recorded_n_binary.astype(int)))
    counts = pd.Series(labels).value_counts().to_dict()
    if counts != {0: 66, 1: 48}:
        raise ValueError(f"{_private_literal('private_758302c106d4f094')}{counts}")
    return labels


def read_folds(root: Path, manifest: pd.DataFrame) -> pd.DataFrame:
    p = root / "folds/folds.tsv"
    if not p.exists():
        raise FileNotFoundError(f"Missing fold assignment: {p}")
    df = pd.read_csv(p, sep="\t", dtype={"patient_key": str, "partition": str})
    req = {"patient_key", "recorded_n_binary", "outer_fold", "partition", "inner_fold"}
    if not req.issubset(df):
        raise ValueError(f"Folds missing columns: {sorted(req - set(df))}")
    home_labels = pd.Series(read_home_labels(root, manifest))
    expected = set(home_labels.index)
    if set(df.patient_key) != expected:
        raise ValueError("Fold patient keys do not equal Home manifest patients")
    if df.duplicated(["outer_fold", "patient_key"]).any():
        raise ValueError("Repeated patient within an outer fold")
    outers = sorted(df.outer_fold.unique().tolist())
    if outers != list(range(5)):
        raise ValueError(f"Expected outer folds 0..4, got {outers}")
    for outer, block in df.groupby("outer_fold"):
        if set(block.patient_key) != expected:
            raise ValueError(f"Outer {outer} does not cover all Home patients")
        if not block.partition.isin(["train", "test"]).all():
            raise ValueError(f"Invalid partition in outer {outer}")
        if not block.loc[block.partition.eq("train"), "inner_fold"].isin([0, 1, 2]).all():
            raise ValueError(f"Train patients need inner fold 0..2 in outer {outer}")
        if block.loc[block.partition.eq("test"), "inner_fold"].notna().any():
            raise ValueError(f"Outer test patient has inner_fold in outer {outer}")
        if set(block.loc[block.partition.eq("train"), "inner_fold"].astype(int)) != {0, 1, 2}:
            raise ValueError(f"Missing inner fold in outer {outer}")
        if block.partition.eq("test").sum() == 0:
            raise ValueError(f"Outer {outer} has no held-out patients")
    test_counts = df.loc[df.partition.eq("test")].groupby("patient_key").size().reindex(home_labels.index, fill_value=0)
    if not test_counts.eq(1).all():
        raise ValueError("Each Home patient must be outer test exactly once")
    labels = pd.to_numeric(df.set_index("patient_key").recorded_n_binary, errors="coerce")
    if not labels.eq(labels.index.map(home_labels)).all():
        raise ValueError("Fold labels disagree with manifest")
    return df


def patient_index_output_manifest(index_dir: Path) -> dict:
    """Hash the complete materialized patient index, including its row map."""
    table_path = index_dir / "index.tsv"
    table = pd.read_csv(table_path, sep="\t", dtype={"patient_key": str, "file": str})
    required = {"patient_key", "file", "unique_aa", "read_count"}
    if not required.issubset(table) or table.patient_key.isna().any() or table.file.isna().any():
        raise ValueError("Patient index table is incomplete")
    if table.patient_key.duplicated().any() or table.file.duplicated().any():
        raise ValueError("Patient index has repeated patient or file")
    if table.unique_aa.le(0).any() or table.read_count.le(0).any():
        raise ValueError("Patient index has nonpositive totals")
    files = []
    for name in sorted(table.file.tolist()):
        if Path(name).name != name or not name.endswith(".npz"):
            raise ValueError(f"Unsafe patient index file name: {name}")
        path = index_dir / name
        files.append({"file": name, "sha256": sha256(path), "bytes": path.stat().st_size})
    return {"index_tsv_sha256": sha256(table_path), "index_files": files}


def prepare_patient_index(root: Path = ROOT, include_external: bool = True) -> dict:
    """Map AA to global seq_idx; sum clone counts over samples within patient."""
    manifest = read_manifest(root)
    union_path = root / "data/aa_union.parquet"
    union = pd.read_parquet(union_path, columns=["seq_idx", "aa"])
    if union.empty or union.aa.isna().any() or union.aa.duplicated().any():
        raise ValueError("Invalid AA union")
    indices = union.seq_idx.to_numpy(dtype=np.int64)
    if not np.array_equal(indices, np.arange(len(indices), dtype=np.int64)):
        raise ValueError("AA union seq_idx must be contiguous in row order")
    aa_to_idx = dict(zip(union.aa.tolist(), indices.tolist()))
    del union
    paths = sorted((root / "data/patient_aa").rglob("*.parquet"))
    if not paths:
        raise FileNotFoundError("No patient_aa parquet parts")
    index_dir = root / "mil/patient_index"
    index_dir.mkdir(parents=True, exist_ok=True)
    atomic_json(index_dir / "receipt.json", {"status": "building",
                "source_manifest_identity_sha256": manifest_identity_sha256(root),
                "source_aa_union_sha256": sha256(union_path)})
    source_parts = []
    input_read_sum = 0
    valid_samples = set(manifest.sample_id.astype(str))
    sample_cohort = manifest.set_index("sample_id").cohort.to_dict()
    sample_patient = manifest.set_index("sample_id").patient_key.to_dict()
    accumulator: dict[str, dict[int, int]] = defaultdict(dict)
    seen_samples: set[str] = set()
    for path in paths:
        check_rss()
        source_parts.append({"relative_path": str(path.relative_to(root)),
                             "sha256": sha256(path), **file_stamp(path)})
        frame = pd.read_parquet(path, columns=["cohort", "sample_id", "patient_key", "aa", "read_count"])
        for (sample, patient), g in frame.groupby(["sample_id", "patient_key"], sort=False):
            sample = str(sample)
            patient = str(patient)
            if sample not in valid_samples:
                raise ValueError(f"Unlisted sample {sample} in {path}")
            if patient != str(sample_patient[sample]) or not g.cohort.eq(sample_cohort[sample]).all():
                raise ValueError(f"Sample metadata mismatch in {path}: {sample}")
            if not include_external and sample_cohort[sample] != "home":
                continue
            if g.read_count.isna().any() or g.read_count.le(0).any():
                raise ValueError(f"Nonpositive read count in {path}")
            seen_samples.add(sample)
            input_read_sum += int(g.read_count.sum())
            bucket = accumulator[patient]
            by_aa = g.groupby("aa", sort=False).read_count.sum()
            for aa, count in by_aa.items():
                idx = aa_to_idx.get(aa)
                if idx is None:
                    raise ValueError(f"AA missing from union: {aa}")
                bucket[idx] = bucket.get(idx, 0) + int(count)
    required = valid_samples if include_external else set(
        manifest.loc[manifest.cohort.eq("home"), "sample_id"].astype(str))
    if seen_samples != required:
        raise ValueError(f"Patient AA sample coverage mismatch: missing={len(required - seen_samples)}, "
                         f"unexpected={len(seen_samples - required)}")
    summary = []
    for patient, bucket in sorted(accumulator.items()):
        idx = np.fromiter(bucket.keys(), dtype=np.int64)
        count = np.fromiter(bucket.values(), dtype=np.int64)
        order = np.argsort(idx)
        idx, count = idx[order], count[order]
        if len(idx) == 0 or np.any(count <= 0):
            raise ValueError(f"Invalid patient index for {patient}")
        digest = hashlib.sha256(patient.encode()).hexdigest()[:24]
        np.savez_compressed(index_dir / f"{digest}.npz", seq_idx=idx, read_count=count)
        summary.append({"patient_key": patient, "file": f"{digest}.npz", "unique_aa": len(idx), "read_count": int(count.sum())})
    if sum(row["read_count"] for row in summary) != input_read_sum:
        raise ValueError("Patient index read-count conservation failed")
    pd.DataFrame(summary).to_csv(index_dir / "index.tsv", sep="\t", index=False)
    receipt = {"status": "complete", "patients": len(summary), "aa_union_rows": len(aa_to_idx),
               "source_aa_union_sha256": sha256(union_path),
               "source_manifest_identity_sha256": manifest_identity_sha256(root),
               "source_parts": source_parts, "source_part_count": len(source_parts),
               "home_samples_seen": manifest.loc[manifest.cohort.eq("home"), "sample_id"].nunique(),
               "total_samples_seen": len(seen_samples),
               "include_external": bool(include_external),
               "input_read_count_sum": input_read_sum,
               "patient_index_read_count_sum": sum(row["read_count"] for row in summary)}
    receipt.update(patient_index_output_manifest(index_dir))
    atomic_json(index_dir / "receipt.json", receipt)
    return receipt


class PatientIndex:
    def __init__(self, root: Path, verify_sources: bool = True):
        index_dir = root / "mil/patient_index"
        receipt = index_dir / "receipt.json"
        if not receipt.exists():
            raise RuntimeError("Complete patient index required; run prepare first")
        provenance = json.loads(receipt.read_text())
        if provenance.get("status") != "complete":
            raise RuntimeError("Complete patient index required; run prepare first")
        output_manifest = patient_index_output_manifest(index_dir)
        if (provenance.get("index_tsv_sha256") != output_manifest["index_tsv_sha256"] or
                provenance.get("index_files") != output_manifest["index_files"]):
            raise ValueError("Patient index output files or index.tsv changed after sealing")
        if verify_sources:
            if provenance.get("source_manifest_identity_sha256") != manifest_identity_sha256(root):
                raise ValueError("Patient index manifest source changed")
            if provenance.get("source_aa_union_sha256") != sha256(root / "data/aa_union.parquet"):
                raise ValueError("Patient index AA union source changed")
            parts = sorted((root / "data/patient_aa").rglob("*.parquet"))
            saved = provenance.get("source_parts", [])
            if len(parts) != provenance.get("source_part_count") or len(parts) != len(saved):
                raise ValueError("Patient index source part count changed")
            for path, record in zip(parts, saved):
                if (str(path.relative_to(root)) != record.get("relative_path") or
                    file_stamp(path) != {k: record.get(k) for k in ("path", "bytes", "mtime_ns")} or
                    sha256(path) != record.get("sha256")):
                    raise ValueError(f"Patient index source part changed: {path}")
        table = pd.read_csv(index_dir / "index.tsv", sep="\t", dtype={"patient_key": str, "file": str})
        if len(table) != provenance.get("patients", len(table)):
            raise ValueError("Patient index patient count differs from receipt")
        if int(table.read_count.sum()) != provenance.get("patient_index_read_count_sum", int(table.read_count.sum())):
            raise ValueError("Patient index read count differs from receipt")
        self.files = {r.patient_key: index_dir / r.file for r in table.itertuples()}
        self.stats = table.set_index("patient_key")
        self.cache: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    def get(self, key: str) -> tuple[np.ndarray, np.ndarray]:
        if key not in self.cache:
            if key not in self.files:
                raise KeyError(f"Missing patient index: {key}")
            with np.load(self.files[key]) as z:
                idx = z["seq_idx"].astype(np.int64)
                cnt = z["read_count"].astype(np.int64)
            expected = self.stats.loc[key]
            if (idx.ndim != 1 or cnt.ndim != 1 or len(idx) != len(cnt) or
                len(idx) != int(expected.unique_aa) or
                not np.all(idx[1:] > idx[:-1]) or np.any(idx < 0) or
                np.any(cnt <= 0) or int(cnt.sum()) != int(expected.read_count)):
                raise ValueError(f"Patient index data mismatch for {key}")
            self.cache[key] = (idx, cnt)
        return self.cache[key]


def sceptr_warning_ids(root: Path, bank_rows: int) -> tuple[np.ndarray, str]:
    path = root / "embeddings/sceptr_cdr3_only/nonstandard_junction_seq_idx.npy"
    if not path.exists():
        raise FileNotFoundError(f"SCEPTR input-domain warning mask required: {path}")
    ids = np.load(path, mmap_mode="r")
    if ids.ndim != 1 or ids.dtype.kind not in "iu" or len(ids) == 0:
        raise ValueError("Invalid SCEPTR nonstandard junction index")
    if ids[0] < 0 or ids[-1] >= bank_rows or np.any(ids[1:] <= ids[:-1]):
        raise ValueError("SCEPTR nonstandard junction index must be sorted, unique and in bank range")
    return ids, sha256(path)


def warning_membership(seq_idx: np.ndarray, warning_ids: np.ndarray) -> np.ndarray:
    pos = np.searchsorted(warning_ids, seq_idx)
    valid = pos < len(warning_ids)
    out = np.zeros(len(seq_idx), dtype=bool)
    out[valid] = warning_ids[pos[valid]] == seq_idx[valid]
    return out


def sceptr_warning_features(index: PatientIndex, keys: list[str],
                            warning_ids: np.ndarray) -> pd.DataFrame:
    rows = []
    for key in keys:
        seq_idx, read_count = index.get(key)
        flagged = warning_membership(seq_idx, warning_ids)
        rows.append({"patient_key": key,
                     "sceptr_nonstandard_aa": int(flagged.sum()),
                     "sceptr_nonstandard_aa_fraction": float(flagged.mean()),
                     "sceptr_nonstandard_reads": int(read_count[flagged].sum()),
                     "sceptr_nonstandard_read_fraction": float(read_count[flagged].sum() / read_count.sum()),
                     "sceptr_total_unique_aa": len(seq_idx),
                     "sceptr_total_reads": int(read_count.sum())})
    return pd.DataFrame(rows)


def paired_draw(n: int, bag_size: int, patient: str, epoch: int, draw: int, scope: str) -> np.ndarray:
    """Draw prefix from a 4096 permutation, paired across encoders and bag sizes."""
    if n <= bag_size:
        return np.arange(n, dtype=np.int64)
    rng = np.random.default_rng(stable_seed("draw-v1", patient, epoch, draw, scope))
    return rng.choice(n, size=min(max(BAGS), n), replace=False)[:bag_size]


@dataclass(frozen=True)
class Projection:
    path: Path | None
    dim: int
    raw_fallback: bool
    source_sha256: str | None
    stability_sha256: str
    fallback_reason: str | None


def _bounded_metric(value: object, label: str, lower: float = 0.0,
                    upper: float | None = None) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"PCA {label} must be numeric")
    number = float(value)
    if not math.isfinite(number) or number < lower or (upper is not None and number > upper):
        raise ValueError(f"PCA {label} outside expected range")
    return number


def pca_source_contract(root: Path, manifest: pd.DataFrame,
                        bank_meta: dict) -> dict[str, str]:
    """Tie PCA receipts to identity-only folds and the sealed patient AA sources."""
    folds_receipt_path = root / "folds/folds.json"
    folds_receipt = json.loads(folds_receipt_path.read_text())
    stage1 = json.loads((root / "audit/01_ingest_summary.json").read_text())
    index_receipt = json.loads((root / "mil/patient_index/receipt.json").read_text())
    manifest_sha = manifest_identity_sha256(root)
    home_labels = read_home_labels(root, manifest)
    home_rows = sorted([[str(row.patient_key), str(home_labels[str(row.patient_key)])]
                        for row in manifest.loc[manifest.cohort.eq("home")].itertuples()])
    home_sha = hashlib.sha256(json.dumps(
        home_rows, ensure_ascii=False, separators=(",", ":")).encode("utf-8")).hexdigest()
    if (folds_receipt.get("folds_sha256") != sha256(root / "folds/folds.tsv") or
        folds_receipt.get("manifest_identity_sha256") != manifest_sha or
        folds_receipt.get("home_split_label_sha256") != home_sha or
        folds_receipt.get("external_in_folds") is not False or
        folds_receipt.get("n_home_patients") != 114 or
        stage1.get("status") != "PASS"):
        raise ValueError("PCA fold or Home-only source receipt mismatch")
    source_fields = ("stage1_source_contract_sha256",
                     "stage1_input_sha256_manifest_sha256", "stage1_union_sha256")
    source_keys = ("source_contract_sha256", "input_sha256_manifest_sha256",
                   "union_sha256")
    for receipt_field, stage1_field in zip(source_fields, source_keys):
        if not folds_receipt.get(receipt_field) or (
                folds_receipt[receipt_field] != stage1.get(stage1_field)):
            raise ValueError(f"PCA Stage1 {receipt_field} changed")
    if (index_receipt.get("status") != "complete" or
        index_receipt.get("source_manifest_identity_sha256") != manifest_sha or
        index_receipt.get("source_aa_union_sha256") != folds_receipt["stage1_union_sha256"] or
        bank_meta.get("input_sha256") != folds_receipt["stage1_union_sha256"]):
        raise ValueError("PCA patient index, AA union, or bank source mismatch")
    parts = index_receipt.get("source_parts")
    if not isinstance(parts, list) or len(parts) != 279 or index_receipt.get("source_part_count") != 279:
        raise ValueError("PCA patient AA source receipt must cover 279 libraries")
    source_hashes = []
    for record in parts:
        rel = Path(record.get("relative_path", ""))
        if rel.parts[:2] != ("data", "patient_aa") or len(rel.parts) != 3 or (
                not rel.name.endswith(".parquet")):
            raise ValueError("PCA patient AA source path outside flat Stage1 directory")
        path = root / rel
        if file_stamp(path) != {k: record.get(k) for k in ("path", "bytes", "mtime_ns")}:
            raise ValueError(f"PCA patient AA source stamp changed: {path}")
        digest = record.get("sha256")
        if not isinstance(digest, str) or len(digest) != 64:
            raise ValueError(f"PCA patient AA source hash missing: {path}")
        source_hashes.append((rel.name, digest))
    if len({name for name, _ in source_hashes}) != 279:
        raise ValueError("PCA patient AA source file name duplicated")
    source_fingerprint = hashlib.sha256(json.dumps(
        sorted(source_hashes), separators=(",", ":")).encode()).hexdigest()
    return {
        "manifest_identity_sha256": manifest_sha,
        "home_split_label_sha256": home_sha,
        "stage1_source_contract_sha256": folds_receipt["stage1_source_contract_sha256"],
        "stage1_input_sha256_manifest_sha256": folds_receipt[
            "stage1_input_sha256_manifest_sha256"],
        "stage1_union_sha256": folds_receipt["stage1_union_sha256"],
        "patient_aa_source_fingerprint": source_fingerprint,
        "folds_receipt_sha256": sha256(folds_receipt_path),
    }


def validate_pca_selection(selection: dict, label: str) -> int | None:
    """Recompute the fixed dimension rule from the saved unlabeled diagnostics."""
    if selection.get("metric_spec") != PCA_METRIC_SPEC:
        raise ValueError(f"PCA immutable metric_spec mismatch: {label}")
    if (selection.get("neighbor_k") != 50 or
        selection.get("neighbor_reference") !=
            "exact brute-force Euclidean top-50 in raw embedding bank" or
        selection.get("probe_source") not in
            ("train_patient_unfitted_aa", "fit_aa_fallback") or
        not isinstance(selection.get("probe_rows"), int) or
        not 52 <= selection["probe_rows"] <= 2048):
        raise ValueError(f"PCA top50/probe specification mismatch: {label}")
    thresholds = selection.get("thresholds")
    if not isinstance(thresholds, dict) or thresholds.get("neighbor_overlap") != 0.80 or (
        thresholds.get("reconstruction") !=
            "mean per-row MSE within one SE of best dimension"):
        raise ValueError(f"PCA locked selection thresholds mismatch: {label}")
    by_dim = selection.get("by_dim")
    if not isinstance(by_dim, dict) or set(by_dim) != {"16", "32", "64"}:
        raise ValueError(f"PCA by_dim must contain exactly 16/32/64: {label}")
    mse = {}
    se = {}
    overlap = {}
    for dim in (16, 32, 64):
        item = by_dim[str(dim)]
        if not isinstance(item, dict):
            raise ValueError(f"PCA dimension metric missing: {label}")
        mse[dim] = _bounded_metric(item.get("reconstruction_mse"), f"{label} MSE {dim}")
        se[dim] = _bounded_metric(item.get("reconstruction_mse_se"), f"{label} MSE SE {dim}")
        overlap[dim] = _bounded_metric(item.get("top50_neighbor_overlap"),
                                       f"{label} top50 overlap {dim}", upper=1.0)
    best_dim = min((16, 32, 64), key=lambda dim: mse[dim])
    cutoff = mse[best_dim] + se[best_dim]
    if thresholds.get("best_dim") != best_dim or not math.isclose(
            _bounded_metric(thresholds.get("one_se_mse_cutoff"), f"{label} one-SE cutoff"),
            cutoff, rel_tol=1e-10, abs_tol=1e-12):
        raise ValueError(f"PCA one-SE cutoff mismatch: {label}")
    for dim in (16, 32, 64):
        if by_dim[str(dim)].get("within_one_se_of_best") is not (mse[dim] <= cutoff):
            raise ValueError(f"PCA one-SE flag mismatch at {dim}: {label}")
    eligible = [dim for dim in (16, 32, 64)
                if overlap[dim] >= 0.80 and mse[dim] <= cutoff]
    chosen = eligible[0] if eligible else None
    if (selection.get("selected_dim") != chosen or
        selection.get("abstain") is not (chosen is None) or
        selection.get("reason") !=
            (None if chosen is not None else "no_dimension_met_prespecified_thresholds")):
        raise ValueError(f"PCA selected_dim does not follow locked metrics: {label}")
    return chosen


def validate_pca_stability(root: Path, model: str, outer: int, inner: int | None,
                           base: Path, selection: dict, metadata: dict,
                           stability: dict, primary_components: np.ndarray) -> None:
    """Verify that the three K2048 receipts support the K4096 stability decision."""
    artifacts = ("pca.npz", "selection.json", "metadata.json")
    if stability.get("metric_spec") != PCA_METRIC_SPEC:
        raise ValueError(f"PCA stability immutable metric_spec mismatch: {base}")
    if (stability.get("min_subspace_overlap_at_selected_dim") != 0.90 or
        stability.get("selected_dim_for_stability") != selection["selected_dim"]):
        raise ValueError(f"PCA locked subspace threshold or dimension mismatch: {base}")
    repeated_hashes = stability.get("replicate_artifact_sha256s")
    repeated_metrics = stability.get("replicate_metrics")
    if (not isinstance(repeated_hashes, dict) or set(repeated_hashes) != {"0", "1", "2"} or
        not isinstance(repeated_metrics, dict) or set(repeated_metrics) != {"0", "1", "2"}):
        raise ValueError(f"PCA K2048 replicate receipt incomplete: {base}")
    chosen = selection["selected_dim"]
    repeated_dims = []
    repeated_overlaps = []
    for repeat in range(3):
        key = str(repeat)
        folder = (root / "pca" / model / "stability_k2048" /
                  f"rep_{repeat}" / f"outer_{outer}")
        if inner is not None:
            folder /= f"inner_{inner}"
        expected_hashes = {name: sha256(folder / name) for name in artifacts}
        if repeated_hashes[key] != expected_hashes:
            raise ValueError(f"PCA K2048 repeat {repeat} artifact hash mismatch: {folder}")
        repeat_meta = json.loads((folder / "metadata.json").read_text())
        repeat_sel = json.loads((folder / "selection.json").read_text())
        for name in ("model", "outer_fold", "inner_fold", "fit_patient_ids",
                     "train_patient_sha256", "folds_sha256", "bank_sha256_from_meta",
                     "manifest_identity_sha256", "home_split_label_sha256",
                     "stage1_source_contract_sha256", "stage1_input_sha256_manifest_sha256",
                     "stage1_union_sha256", "patient_aa_source_fingerprint",
                     "folds_receipt_sha256", "aa_union_sha256", "bank_input_sha256",
                     "bank_model_fingerprint"):
            if repeat_meta.get(name) != metadata.get(name):
                raise ValueError(f"PCA K2048 repeat {repeat} {name} mismatch: {folder}")
        if (repeat_meta.get("fit_patient_count") != metadata.get("fit_patient_count") or
            repeat_meta.get("k_per_patient") != 2048 or
            repeat_meta.get("draw_replicate") != repeat or
            repeat_meta.get("components") != 64):
            raise ValueError(f"PCA K2048 repeat {repeat} sampling mismatch: {folder}")
        repeat_dim = validate_pca_selection(repeat_sel, str(folder / "selection.json"))
        repeated_dims.append(repeat_dim)
        metric = repeated_metrics[key]
        if (metric.get("selected_dim") != repeat_dim or
            metric.get("abstain") is not (repeat_dim is None)):
            raise ValueError(f"PCA K2048 repeat {repeat} selected dimension mismatch: {folder}")
        reported_neighbors = metric.get("top50_neighbor_overlap_by_dim")
        if not isinstance(reported_neighbors, dict) or set(reported_neighbors) != {"16", "32", "64"}:
            raise ValueError(f"PCA K2048 repeat {repeat} top50 metrics missing: {folder}")
        for dim in (16, 32, 64):
            expected = repeat_sel["by_dim"][str(dim)]["top50_neighbor_overlap"]
            if not math.isclose(_bounded_metric(reported_neighbors[str(dim)],
                                               f"repeat {repeat} top50 {dim}", upper=1.0),
                                expected, rel_tol=1e-12, abs_tol=1e-12):
                raise ValueError(f"PCA K2048 repeat {repeat} top50 mismatch: {folder}")
        with np.load(folder / "pca.npz") as z:
            components = z["components"]
        if components.shape != primary_components.shape:
            raise ValueError(f"PCA K2048 repeat {repeat} component shape mismatch: {folder}")
        reported_subspaces = metric.get("subspace_overlap_vs_k4096")
        if not isinstance(reported_subspaces, dict) or set(reported_subspaces) != {"16", "32", "64"}:
            raise ValueError(f"PCA K2048 repeat {repeat} subspace metrics missing: {folder}")
        for dim in (16, 32, 64):
            singular = np.linalg.svd(primary_components[:dim] @ components[:dim].T,
                                     compute_uv=False)
            actual = float(np.sum(singular ** 2) / dim)
            reported = _bounded_metric(reported_subspaces[str(dim)],
                                       f"repeat {repeat} subspace {dim}", upper=1.0001)
            if not math.isclose(actual, reported, rel_tol=1e-5, abs_tol=1e-5):
                raise ValueError(f"PCA K2048 repeat {repeat} subspace mismatch: {folder}")
        repeated_overlaps.append(reported_subspaces)
    all_match = all(dim == chosen for dim in repeated_dims)
    passed = bool(chosen is not None and all_match and
                  all(overlaps[str(chosen)] >= 0.90 for overlaps in repeated_overlaps))
    if (stability.get("all_replicates_match_primary_dim") is not all_match or
        stability.get("stability_pass") is not passed):
        raise ValueError(f"PCA K2048 stability_pass does not follow locked metrics: {base}")


def pca_spec(root: Path, model: str, outer: int, inner: int | None, bank_dim: int) -> Projection:
    """Accept PCA only when every scope-local provenance field matches this split."""
    base = root / "pca" / model / f"outer_{outer}"
    if inner is not None:
        base /= f"inner_{inner}"
    sel_path, pca_path = base / "selection.json", base / "pca.npz"
    metadata_path, stability_path = base / "metadata.json", base / "stability.json"
    for path in (sel_path, pca_path, metadata_path, stability_path):
        if not path.exists():
            raise FileNotFoundError(f"Fold-local PCA artifact required: {path}")
    manifest = read_manifest(root)
    folds = read_folds(root, manifest)
    train, _ = split_keys(folds, outer, inner)
    expected_train = sorted(train)
    train_sha = train_patient_hash(train)
    folds_sha = sha256(root / "folds/folds.tsv")
    bank_meta_path = root / "embeddings" / model / "meta.json"
    bank_meta = json.loads(bank_meta_path.read_text())
    if (bank_meta.get("status") != "complete" or
        bank_meta.get("bank_shape", [None, None])[1] != bank_dim or
        not bank_meta.get("bank_sha256")):
        raise ValueError(f"Complete embedding bank metadata required for PCA: {bank_meta_path}")
    selection = json.loads(sel_path.read_text())
    metadata = json.loads(metadata_path.read_text())
    stability = json.loads(stability_path.read_text())
    common = {"model": model, "outer_fold": outer, "inner_fold": inner,
              "fit_patient_ids": expected_train,
              "train_patient_sha256": train_sha,
              "folds_sha256": folds_sha,
              "bank_sha256_from_meta": bank_meta["bank_sha256"]}
    common.update(pca_source_contract(root, manifest, bank_meta))
    for key, value in common.items():
        if metadata.get(key) != value or stability.get(key) != value:
            raise ValueError(f"PCA {key} scope/provenance mismatch: {base}")
    if (metadata.get("aa_union_sha256") != bank_meta.get("input_sha256") or
        metadata.get("bank_input_sha256") != bank_meta.get("input_sha256") or
        metadata.get("bank_model_fingerprint") != bank_meta.get("model_fingerprint") or
        metadata.get("fit_patient_count") != len(expected_train) or
        metadata.get("components") != 64 or
        metadata.get("k_per_patient") != 4096 or metadata.get("draw_replicate") != 0 or
        stability.get("primary_k") != 4096 or stability.get("repeat_k") != 2048 or
        stability.get("repeats") != 3 or
        stability.get("primary_selected_dim") != selection.get("selected_dim")):
        raise ValueError(f"PCA selection/stability metadata mismatch: {base}")
    expected_primary_hashes = {name: sha256(base / name)
                               for name in ("pca.npz", "selection.json", "metadata.json")}
    if stability.get("primary_artifact_sha256s") != expected_primary_hashes:
        raise ValueError(f"PCA primary artifact hash mismatch: {base}")
    chosen = validate_pca_selection(selection, str(sel_path))
    with np.load(pca_path) as z:
        components, mean = z["components"], z["mean"]
    if (components.ndim != 2 or components.shape[0] != 64 or
        components.shape[1] != bank_dim or mean.shape != (bank_dim,) or
        not np.isfinite(components).all() or not np.isfinite(mean).all()):
        raise ValueError(f"PCA component/bank mismatch: {pca_path}")
    validate_pca_stability(root, model, outer, inner, base, selection, metadata,
                           stability, components)
    stability_sha = sha256(stability_path)
    if not stability.get("stability_pass", False):
        return Projection(None, bank_dim, True, None, stability_sha, "K2048_repeat_stability_failed")
    if bool(selection.get("abstain", False)):
        return Projection(None, bank_dim, True, None, stability_sha, "PCA_selection_abstained")
    dim = int(chosen)
    if dim not in (16, 32, 64):
        raise ValueError(f"Invalid selected PCA dim in {sel_path}")
    return Projection(pca_path, dim, False, sha256(pca_path), stability_sha, None)


def verify_projection_file(path: Path, rows: int, dim: int,
                           expected_sha256: str | None = None,
                           chunk: int = 131072) -> tuple[np.ndarray, str]:
    """Check every projected vector and bind a complete cache to its file bytes."""
    arr = np.load(path, mmap_mode="r", allow_pickle=False)
    if arr.shape != (rows, dim) or arr.dtype != np.float16:
        raise ValueError(f"Projection cache shape/dtype mismatch: {path}")
    for begin in range(0, rows, chunk):
        if not np.isfinite(arr[begin:begin + chunk]).all():
            raise ValueError(f"Nonfinite projection cache values: {path}")
    digest = sha256(path)
    if expected_sha256 is not None and digest != expected_sha256:
        raise ValueError(f"Projection cache SHA-256 mismatch: {path}")
    return arr, digest


def project_bank(root: Path, model: str, outer: int, inner: int | None,
                 raw_bank: np.ndarray, spec: Projection, device: str, chunk: int = 32768) -> np.ndarray:
    if spec.raw_fallback:
        return raw_bank
    name = f"outer_{outer}" if inner is None else f"outer_{outer}_inner_{inner}"
    base = root / "mil/projections" / model
    base.mkdir(parents=True, exist_ok=True)
    out_path = base / f"{name}_d{spec.dim}.npy"
    meta_path = base / f"{name}_d{spec.dim}.json"
    raw_path = root / "embeddings" / model / "bank_fp16.npy"
    raw_meta = root / "embeddings" / model / "meta.json"
    metadata = {"bank_rows": int(raw_bank.shape[0]), "bank_dim": int(raw_bank.shape[1]),
                "pca_sha256": spec.source_sha256,
                "scope_pca_stability_sha256": spec.stability_sha256,
                "pca_metadata_sha256": sha256(root / "pca" / model / f"outer_{outer}" /
                                               (f"inner_{inner}" if inner is not None else "") / "metadata.json"),
                "dim": spec.dim, "raw_bank_stamp": file_stamp(raw_path),
                "raw_bank_meta_sha256": sha256(raw_meta) if raw_meta.exists() else None}
    if out_path.exists() and meta_path.exists():
        saved = json.loads(meta_path.read_text())
        if all(saved.get(k) == v for k, v in metadata.items()):
            if (saved.get("status") == "complete" and
                    saved.get("projection_dtype") == "float16" and
                    saved.get("nonfinite_values") == 0 and
                    isinstance(saved.get("projection_sha256"), str) and
                    saved.get("projection_size_bytes") == out_path.stat().st_size):
                arr, _ = verify_projection_file(out_path, len(raw_bank), spec.dim,
                                                saved["projection_sha256"])
                return arr
            if saved.get("status") == "complete":
                raise ValueError(f"Projection cache receipt lacks integrity fields: {meta_path}")
    with np.load(spec.path) as z:
        mean = z["mean"].astype(np.float32)
        comp = z["components"][:spec.dim].astype(np.float32).T.copy()
    projection_start = time.monotonic()
    tmp = out_path.with_name(out_path.stem + ".tmp.npy")
    out = np.lib.format.open_memmap(tmp, mode="w+", dtype=np.float16, shape=(len(raw_bank), spec.dim))
    if device == "cuda":
        torch.cuda.empty_cache()
        m = torch.as_tensor(mean, dtype=torch.float32, device="cuda")
        c = torch.as_tensor(comp, dtype=torch.float32, device="cuda")
        for start in range(0, len(raw_bank), chunk):
            stop = min(start + chunk, len(raw_bank))
            x = torch.from_numpy(np.array(raw_bank[start:stop], dtype=np.float32)).to("cuda")
            y = (x - m) @ c
            out[start:stop] = y.cpu().numpy().astype(np.float16)
            del x, y
    else:
        for start in range(0, len(raw_bank), chunk):
            stop = min(start + chunk, len(raw_bank))
            x = np.asarray(raw_bank[start:stop], dtype=np.float32)
            out[start:stop] = ((x - mean) @ comp).astype(np.float16)
    out.flush()
    del out
    _, projection_sha = verify_projection_file(tmp, len(raw_bank), spec.dim)
    projection_bytes = tmp.stat().st_size
    tmp.replace(out_path)
    atomic_json(meta_path, {**metadata, "status": "complete",
                            "projection_sha256": projection_sha,
                            "projection_size_bytes": projection_bytes,
                            "projection_dtype": "float16", "nonfinite_values": 0,
                            "projection_seconds": round(time.monotonic() - projection_start, 3),
                            "peak_observed_cpu_rss_gib": round(check_rss(), 3),
                            "device": device})
    return np.load(out_path, mmap_mode="r", allow_pickle=False)


def load_bank(root: Path, model: str) -> np.ndarray:
    p = root / "embeddings" / model / "bank_fp16.npy"
    if not p.exists():
        raise FileNotFoundError(f"Missing embedding bank {p}")
    bank = np.load(p, mmap_mode="r")
    if bank.ndim != 2 or bank.dtype != np.float16:
        raise ValueError(f"Invalid bank shape/type: {p}, {bank.shape}, {bank.dtype}")
    meta_path = p.with_name("meta.json")
    if not meta_path.exists():
        raise RuntimeError(f"Embedding completion metadata absent: {meta_path}")
    meta = json.loads(meta_path.read_text())
    if meta.get("status") != "complete":
        raise RuntimeError(f"Embedding not complete: {meta_path}")
    if meta.get("bank_shape") != list(bank.shape) or meta.get("bank_dtype") != "float16":
        raise ValueError(f"Embedding bank metadata shape/dtype mismatch: {model}")
    if int(meta.get("n_total", -1)) != len(bank):
        raise ValueError(f"Embedding bank metadata count mismatch: {model}")
    stat = p.stat()
    if "bank_size_bytes" in meta and int(meta["bank_size_bytes"]) != stat.st_size:
        raise ValueError(f"Embedding bank metadata size mismatch: {model}")
    if "bank_mtime_ns" in meta and int(meta["bank_mtime_ns"]) != stat.st_mtime_ns:
        raise ValueError(f"Embedding bank metadata mtime mismatch: {model}")
    verification_path = p.with_name("verification.json")
    if not verification_path.exists():
        raise RuntimeError(f"Embedding independent verification absent: {verification_path}")
    verified = json.loads(verification_path.read_text())
    expected = {"status": "PASS", "model": model, "n_total": len(bank),
                "dimension": bank.shape[1], "input_sha256": meta.get("input_sha256"),
                "model_fingerprint": meta.get("model_fingerprint"),
                "bank_sha256_verified": meta.get("bank_sha256"),
                "nonfinite_values": 0, "zero_vectors": 0}
    if any(verified.get(key) != value for key, value in expected.items()):
        raise ValueError(f"Embedding independent verification mismatch: {model}")
    union_path = root / "data/aa_union.parquet"
    import pyarrow.parquet as pq
    if len(bank) != pq.ParquetFile(union_path).metadata.num_rows:
        raise ValueError(f"Bank rows do not match AA union: {model}")
    return bank


def verify_primary_bank_bytes(root: Path) -> dict:
    """Hash both complete banks during preflight and retain a fit-reusable receipt."""
    path = root / "mil/bank_integrity.json"
    expected = {}
    for model in MODELS:
        bank = load_bank(root, model)
        bank_path = root / "embeddings" / model / "bank_fp16.npy"
        meta_path = bank_path.with_name("meta.json")
        meta = json.loads(meta_path.read_text())
        expected[model] = {"bank_stamp": file_stamp(bank_path),
                           "bank_meta_sha256": sha256(meta_path),
                           "bank_sha256": meta["bank_sha256"],
                           "shape": list(bank.shape)}
    for model, record in expected.items():
        bank_path = root / "embeddings" / model / "bank_fp16.npy"
        if sha256(bank_path) != record["bank_sha256"]:
            raise ValueError(f"Current embedding bank SHA-256 differs from sealed metadata: {model}")
    if path.exists():
        saved = json.loads(path.read_text())
        if saved.get("status") == "complete" and saved.get("models") == expected:
            return saved
    result = {"status": "complete", "checked_utc": datetime.now(timezone.utc).isoformat(),
              "method": "current complete .npy bytes compared with sealed embedding metadata",
              "models": expected}
    atomic_json(path, result)
    return result


def require_primary_bank_receipt(root: Path) -> dict:
    path = root / "mil/bank_integrity.json"
    if not path.exists():
        raise RuntimeError("Current-byte bank integrity receipt absent; run preflight before formal MIL")
    saved = json.loads(path.read_text())
    if saved.get("status") != "complete" or set(saved.get("models", {})) != set(MODELS):
        raise ValueError("Primary bank integrity receipt incomplete")
    for model, record in saved["models"].items():
        bank_path = root / "embeddings" / model / "bank_fp16.npy"
        meta_path = bank_path.with_name("meta.json")
        meta = json.loads(meta_path.read_text())
        if (record.get("bank_stamp") != file_stamp(bank_path) or
                record.get("bank_meta_sha256") != sha256(meta_path) or
                record.get("bank_sha256") != meta.get("bank_sha256")):
            raise ValueError(f"Primary bank changed after current-byte verification: {model}")
    return saved


def metrics(y: np.ndarray, p: np.ndarray) -> dict:
    y, p = np.asarray(y, dtype=int), np.clip(np.asarray(p, dtype=float), 1e-6, 1 - 1e-6)
    result = {"n": int(len(y)), "log_loss": float(log_loss(y, p, labels=[0, 1])),
              "brier": float(brier_score_loss(y, p))}
    result["auroc"] = float(roc_auc_score(y, p)) if len(np.unique(y)) == 2 else None
    result["ap"] = float(average_precision_score(y, p)) if len(np.unique(y)) == 2 else None
    if len(y) >= 20 and len(np.unique(y)) == 2:
        logits = np.log(p / (1 - p)).reshape(-1, 1)
        cal = LogisticRegression(C=1e6, solver="lbfgs", max_iter=1000).fit(logits, y)
        result.update(calibration_intercept=float(cal.intercept_[0]), calibration_slope=float(cal.coef_[0, 0]))
    else:
        result.update(calibration_intercept=None, calibration_slope=None)
    return result


class GatedMIL(nn.Module):
    def __init__(self, dim: int, hidden: int = 64, attn: int = 32, dropout: float = 0.2):
        super().__init__()
        self.encoder = nn.Sequential(nn.LayerNorm(dim + 1), nn.Linear(dim + 1, hidden),
                                     nn.ReLU(), nn.Dropout(dropout))
        self.attn_tanh = nn.Linear(hidden, attn)
        self.attn_gate = nn.Linear(hidden, attn)
        self.attn_out = nn.Linear(attn, 1)
        self.head = nn.Sequential(nn.LayerNorm(hidden), nn.Linear(hidden, 1))

    def forward(self, x: torch.Tensor, count_log: torch.Tensor, mask: torch.Tensor,
                return_attention: bool = False) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        h = self.encoder(torch.cat([x, count_log.unsqueeze(-1)], dim=-1))
        score = self.attn_out(torch.tanh(self.attn_tanh(h)) * torch.sigmoid(self.attn_gate(h))).squeeze(-1)
        score = score.masked_fill(~mask, -1e9)
        weight = torch.softmax(score, dim=1)
        pooled = (weight.unsqueeze(-1) * h).sum(dim=1)
        logit = self.head(pooled).squeeze(-1)
        return (logit, weight) if return_attention else logit


def bag_tensors(keys: list[str], index: PatientIndex, bank: np.ndarray,
                bag_size: int, epoch: int, draw: int, scope: str, device: str,
                exclude: dict[str, set[int]] | None = None) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    dim = bank.shape[1]
    x = np.zeros((len(keys), bag_size, dim), dtype=np.float32)
    count = np.zeros((len(keys), bag_size), dtype=np.float32)
    mask = np.zeros((len(keys), bag_size), dtype=np.bool_)
    for row, key in enumerate(keys):
        seq_idx, reads = index.get(key)
        if exclude and key in exclude:
            keep = ~np.isin(seq_idx, np.fromiter(exclude[key], dtype=np.int64))
            seq_idx, reads = seq_idx[keep], reads[keep]
        if len(seq_idx) == 0:
            raise ValueError(f"No AA remain in patient {key}")
        pick = paired_draw(len(seq_idx), bag_size, key, epoch, draw, scope)
        n = len(pick)
        x[row, :n] = np.asarray(bank[seq_idx[pick]], dtype=np.float32)
        count[row, :n] = np.log1p(reads[pick]).astype(np.float32)
        mask[row, :n] = True
    return (torch.from_numpy(x).to(device), torch.from_numpy(count).to(device),
            torch.from_numpy(mask).to(device))


@torch.no_grad()
def predict_mc(model: GatedMIL, keys: list[str], index: PatientIndex, bank: np.ndarray,
               bag_size: int, device: str, scope: str, draws: int = 8,
               exclude: dict[str, set[int]] | None = None) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    preds = np.empty((len(keys), draws), dtype=np.float64)
    for d in range(draws):
        for i, key in enumerate(keys):
            x, count, mask = bag_tensors([key], index, bank, bag_size, -1, d, scope, device, exclude)
            preds[i, d] = torch.sigmoid(model(x, count, mask)).item()
    return preds.mean(axis=1), preds.std(axis=1, ddof=1) if draws > 1 else np.zeros(len(keys))



def predict_adaptive(model: GatedMIL, keys: list[str], index: PatientIndex, bank: np.ndarray,
                     bag_size: int, device: str, scope: str,
                     exclude: dict[str, set[int]] | None = None) -> tuple[np.ndarray, np.ndarray, int, bool]:
    pred, sd = predict_mc(model, keys, index, bank, bag_size, device, scope, draws=8, exclude=exclude)
    draws = 8
    if np.any(sd > 0.05):
        pred, sd = predict_mc(model, keys, index, bank, bag_size, device, scope, draws=32, exclude=exclude)
        draws = 32
    return pred, sd, draws, bool(np.any(sd > 0.05))


@torch.no_grad()
def sceptr_warning_attention_mc(model: GatedMIL, keys: list[str], index: PatientIndex,
                                bank: np.ndarray, bag_size: int, device: str,
                                warning_ids: np.ndarray, scope: str, draws: int = 8) -> pd.DataFrame:
    """Descriptive heldout attention mass on SCEPTR warning AA over fixed bags."""
    model.eval()
    rows = []
    for key in keys:
        seq_idx, _ = index.get(key)
        masses = []
        for draw in range(draws):
            chosen = paired_draw(len(seq_idx), bag_size, key, -1, draw, scope)
            x, count, mask = bag_tensors([key], index, bank, bag_size, -1, draw,
                                          scope, device)
            _, weights = model(x, count, mask, return_attention=True)
            flagged = warning_membership(seq_idx[chosen], warning_ids)
            masses.append(float(np.sum(weights[0, :len(chosen)].cpu().numpy() * flagged)))
        rows.append({"patient_key": key, "warning_attention_mass": float(np.mean(masses)),
                     "warning_attention_mass_sd": float(np.std(masses, ddof=1)) if draws > 1 else 0.0,
                     "warning_attention_draws": draws})
    return pd.DataFrame(rows)


def warning_error_association(frame: pd.DataFrame, model_column: str) -> dict:
    """Spearman descriptions only; no threshold search or inference claim."""
    from scipy.stats import spearmanr
    result = {}
    for model_name, sub in frame.groupby(model_column):
        sub = sub.copy()
        sub["patient_logloss"] = patient_logloss(sub.label, sub.prediction)
        fields = ("sceptr_nonstandard_aa_fraction", "sceptr_nonstandard_read_fraction")
        correlations = {}
        for field in fields:
            rho = spearmanr(sub[field], sub.patient_logloss).statistic if len(sub) >= 5 else np.nan
            correlations[f"{field}_vs_patient_logloss_spearman_rho"] = (
                float(rho) if np.isfinite(rho) else None)
        if model_name == "sceptr_cdr3_only" and "warning_attention_mass" in sub:
            att = sub.warning_attention_mass
            if att.notna().sum() >= 5:
                for field in (*fields, "patient_logloss"):
                    good = att.notna() & sub[field].notna()
                    rho = spearmanr(att[good], sub.loc[good, field]).statistic
                    correlations[f"warning_attention_mass_vs_{field}_spearman_rho"] = (
                        float(rho) if np.isfinite(rho) else None)
        result[str(model_name)] = {"n_patients": len(sub),
                                   "median_warning_aa_fraction": float(sub.sceptr_nonstandard_aa_fraction.median()),
                                   "median_warning_read_fraction": float(sub.sceptr_nonstandard_read_fraction.median()),
                                   "correlations": correlations}
    return {"status": "descriptive_only", "by_model": result,
            "warning_definition": "SCEPTR nonstandard junction index; all raw AA were encoded",
            "interpretation": "Association with error or attention is descriptive and does not change model or bag selection."}


def train_mil(train_keys: list[str], labels: dict[str, int], index: PatientIndex, bank: np.ndarray,
              bag_size: int, seed: int, device: str, scope: str, val_keys: list[str] | None = None,
              max_epochs: int = 40, patience: int = 6, fixed_epochs: int | None = None,
              patient_batch: int = 2, mc_val: int = 2, exclude: dict[str, set[int]] | None = None,
              max_gpu_gib: float = 12.0,
              shuffle_scope: str | None = None) -> tuple[GatedMIL, dict]:
    if not train_keys or (val_keys is not None and not val_keys):
        raise ValueError("Empty train/validation split")
    torch.manual_seed(seed)
    np.random.seed(seed)
    if device == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but unavailable")
        torch.cuda.reset_peak_memory_stats()
    model = GatedMIL(bank.shape[1]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-3)
    crit = nn.BCEWithLogitsLoss()
    best_loss, best_epoch, stale, best_state = math.inf, 0, 0, None
    total_epochs = fixed_epochs if fixed_epochs is not None else max_epochs
    effective_shuffle_scope = shuffle_scope if shuffle_scope is not None else scope
    start = time.monotonic()
    for epoch in range(total_epochs):
        check_rss()
        model.train()
        rng = np.random.default_rng(stable_seed("patient-shuffle-v1", seed,
                                                effective_shuffle_scope, epoch))
        order = rng.permutation(train_keys).tolist()
        for begin in range(0, len(order), patient_batch):
            keys = order[begin:begin + patient_batch]
            x, count, mask = bag_tensors(keys, index, bank, bag_size, epoch, 0,
                                         scope="train", device=device, exclude=exclude)
            target = torch.tensor([labels[k] for k in keys], dtype=torch.float32, device=device)
            optimizer.zero_grad(set_to_none=True)
            loss = crit(model(x, count, mask), target)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        if device == "cuda":
            peak = torch.cuda.max_memory_allocated() / (1024 ** 3)
            if peak > max_gpu_gib:
                raise RuntimeError(f"GPU allocated {peak:.2f} GiB > policy {max_gpu_gib:.2f} GiB")
        if val_keys is not None:
            pred, _ = predict_mc(model, val_keys, index, bank, bag_size, device,
                                 scope=f"val-{scope}", draws=mc_val, exclude=exclude)
            y = np.array([labels[k] for k in val_keys], dtype=int)
            value = log_loss(y, np.clip(pred, 1e-6, 1 - 1e-6), labels=[0, 1])
            if value < best_loss - 1e-4:
                best_loss, best_epoch, stale = value, epoch + 1, 0
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            else:
                stale += 1
                if stale >= patience:
                    break
    if best_state is not None:
        model.load_state_dict(best_state)
    receipt = {"epochs_ran": epoch + 1, "best_epoch": best_epoch or epoch + 1,
               "val_log_loss": None if val_keys is None else float(best_loss),
               "seconds": round(time.monotonic() - start, 3),
               "peak_gpu_gib": round(torch.cuda.max_memory_allocated() / 1024**3, 4) if device == "cuda" else 0.0,
               "bag_size": bag_size, "seed": seed, "dim": int(bank.shape[1]),
               "shuffle_scope": effective_shuffle_scope,
               "n_train": len(train_keys), "n_val": 0 if val_keys is None else len(val_keys)}
    return model, receipt


def prior_baseline(train_keys: list[str], test_keys: list[str], labels: dict[str, int]) -> np.ndarray:
    prevalence = (sum(labels[k] for k in train_keys) + 0.5) / (len(train_keys) + 1)
    return np.full(len(test_keys), prevalence, dtype=np.float64)


def technical_baseline(train_keys: list[str], test_keys: list[str], labels: dict[str, int],
                       index: PatientIndex) -> np.ndarray:
    def make_x(keys: list[str]) -> np.ndarray:
        return np.array([[math.log1p(float(index.stats.loc[k, "unique_aa"])),
                          math.log1p(float(index.stats.loc[k, "read_count"]))] for k in keys])
    y = np.array([labels[k] for k in train_keys], dtype=int)
    if len(np.unique(y)) < 2:
        return prior_baseline(train_keys, test_keys, labels)
    xtr, xte = make_x(train_keys), make_x(test_keys)
    mu, sd = xtr.mean(axis=0), xtr.std(axis=0)
    sd[sd == 0] = 1
    clf = LogisticRegression(C=1.0, max_iter=1000, solver="lbfgs", random_state=0)
    clf.fit((xtr - mu) / sd, y)
    return clf.predict_proba((xte - mu) / sd)[:, 1]




def read_home_covariates(root: Path, home_keys: set[str]) -> pd.DataFrame:
    path = root / "config/home_clinical_covariates.tsv"
    if not path.exists():
        raise FileNotFoundError(f"Verified Home clinical covariates required: {path}")
    cov = pd.read_csv(path, sep="\t", dtype={"patient_key": str, "sex": str, "library_batch": str})
    required = {"patient_key", "age_years", "sex", "library_batch"}
    if not required.issubset(cov):
        raise ValueError(f"Clinical covariate columns absent: {sorted(required - set(cov))}")
    if cov.patient_key.duplicated().any() or set(cov.patient_key) != home_keys:
        raise ValueError("Clinical covariate keys must exactly equal Home patients")
    cov["age_years"] = pd.to_numeric(cov.age_years, errors="coerce")
    if not np.isfinite(cov.age_years).all() or cov.age_years.le(0).any():
        raise ValueError("Missing or invalid age in Home clinical covariates")
    for col in ("sex", "library_batch"):
        if cov[col].isna().any() or cov[col].astype(str).str.strip().eq("").any():
            raise ValueError(f"Missing {col} in Home clinical covariates")
    return cov.set_index("patient_key")


def clinical_baselines(root: Path, train_keys: list[str], test_keys: list[str],
                       labels: dict[str, int], index: PatientIndex) -> tuple[np.ndarray, np.ndarray, dict]:
    from sklearn.compose import ColumnTransformer
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder, StandardScaler
    cov = read_home_covariates(root, set(labels))
    order = train_keys + test_keys
    x = cov.loc[order, ["age_years", "sex", "library_batch"]].copy()
    x["log_unique_aa"] = [math.log1p(float(index.stats.loc[k, "unique_aa"])) for k in order]
    x["log_reads"] = [math.log1p(float(index.stats.loc[k, "read_count"])) for k in order]
    y = np.array([labels[k] for k in train_keys], dtype=int)
    def fit_predict(numeric: list[str], categorical: list[str]) -> np.ndarray:
        transformer = ColumnTransformer([
            ("numeric", Pipeline([("impute", SimpleImputer(strategy="median")),
                                   ("scale", StandardScaler())]), numeric),
            ("categorical", Pipeline([("impute", SimpleImputer(strategy="most_frequent")),
                                       ("onehot", OneHotEncoder(handle_unknown="ignore"))]), categorical),
        ])
        pipe = Pipeline([("preprocess", transformer),
                         ("logistic", LogisticRegression(C=1.0, max_iter=1000, solver="lbfgs", random_state=0))])
        pipe.fit(x.iloc[:len(train_keys)], y)
        return pipe.predict_proba(x.iloc[len(train_keys):])[:, 1]
    clinical = fit_predict(["age_years"], ["sex"])
    combined = fit_predict(["age_years", "log_unique_aa", "log_reads"], ["sex", "library_batch"])
    receipt = {"status": "complete", "source_sha256": sha256(root / "config/home_clinical_covariates.tsv"),
               "train_patient_sha256": train_patient_hash(train_keys),
               "test_patient_sha256": train_patient_hash(test_keys),
               "folds_sha256": sha256(root / "folds/folds.tsv"),
               "patient_index_receipt_sha256": sha256(root / "mil/patient_index/receipt.json"),
               "clinical_features": ["age_years", "sex"],
               "clinical_technical_batch_features": ["age_years", "sex", "log_unique_aa", "log_reads", "library_batch"],
               "transform_fit": "outer train only, impute/scale/one-hot",
               "warning": "Library batch and documented sample material are confounded; no material covariate is included."}
    return clinical, combined, receipt



def validate_group_fold(root: Path, outer: int, train_keys: list[str]) -> dict:
    """Refuse fold groups unless frozen projection and all burden files agree."""
    base = root / "groups" / f"outer_{outer}"
    receipt_path = base / "burden_receipt.json"
    if not receipt_path.exists():
        raise FileNotFoundError(f"Complete frozen fold group burden required: {base}")
    receipt = json.loads(receipt_path.read_text())
    expected = train_patient_hash(train_keys)
    home = set(read_manifest(root).loc[lambda x: x.cohort.eq("home"), "patient_key"].astype(str))
    if (receipt.get("train_patient_sha256") != expected or
        receipt.get("status") != "complete" or
        receipt.get("scope") != f"outer_{outer}" or
        receipt.get("target_cohort") != "home" or
        set(receipt.get("target_patient_ids", [])) != home or
        receipt.get("target_patient_sha256") != train_patient_hash(list(home)) or
        receipt.get("train_patient_count") != len(train_keys) or
        receipt.get("heldout_patient_count") != len(home) - len(train_keys) or
        receipt.get("folds_sha256") != sha256(root / "folds/folds.tsv")):
        raise ValueError(f"Fold group-burden receipt mismatch: {receipt_path}")
    inputs = {
        "patient_group_burden_sha256": base / "patient_group_burden.parquet",
        "group_membership_sha256": base / "group_membership.parquet",
        "patient_group_aa_safe_sha256": base / "patient_group_aa_safe.parquet",
        "description_sha256": base / "description.json",
        "integrated_projection_receipt_sha256": base / "integrated_projection_receipt.json",
    }
    for field, path in inputs.items():
        if not path.exists() or receipt.get(field) != sha256(path):
            raise ValueError(f"Fold group-burden {field} mismatch: {path}")
    integrated = json.loads(inputs["integrated_projection_receipt_sha256"].read_text())
    if (integrated.get("status") != "complete_home_scope" or
        integrated.get("projection_population") != "home" or
        integrated.get("scope") != f"outer_{outer}" or
        integrated.get("train_patient_sha256") != expected or
        integrated.get("node_description_sha256") != receipt["description_sha256"] or
        integrated.get("folds_sha256") != receipt["folds_sha256"] or
        integrated.get("membership_sha256") != receipt["group_membership_sha256"] or
        receipt.get("membership_sha256") != receipt["group_membership_sha256"]):
        raise ValueError(f"Fold integrated projection receipt mismatch: {base}")
    return receipt


def group_burden_baseline(root: Path, outer: int, train_keys: list[str], test_keys: list[str],
                          labels: dict[str, int], index: PatientIndex, omit_group: str | None = None) -> tuple[np.ndarray | None, dict]:
    """Exact V-aware fold-dictionary burden baseline, fixed L2 penalty.

    Membership and eligibility are label blind. Only the outer training patients
    determine which groups have at least ten carriers. The penalty is fixed so
    no inner-validation dictionary is needed for this baseline.
    """
    path = root / "groups" / f"outer_{outer}" / "patient_group_burden.parquet"
    if not path.exists():
        return None, {"status": "blocked", "reason": f"Missing {path}"}
    from scipy import sparse
    df = pd.read_parquet(path, columns=["patient_key", "group_id", "presence"])
    df["patient_key"] = df.patient_key.astype(str)
    df["group_id"] = df.group_id.astype(str)
    if df.duplicated(["patient_key", "group_id"]).any():
        raise ValueError("Duplicate patient-group burden row")
    if not df.presence.isin([0, 1, False, True]).all():
        raise ValueError("Invalid group presence")
    train_set = set(train_keys)
    selected = df.loc[df.patient_key.isin(train_set) & df.presence.astype(bool)]
    eligible = selected.groupby("group_id").patient_key.nunique()
    groups = sorted(g for g in eligible.loc[eligible.ge(10)].index if g != omit_group)
    if not groups:
        return None, {"status": "unavailable", "reason_code": "no_eligible_group_ge10_outer_train",
                      "reason": "No groups with >=10 outer-train carriers",
                      "eligible_groups": 0, "source_sha256": sha256(path)}
    patient_order = train_keys + test_keys
    row_lookup = {p: i for i, p in enumerate(patient_order)}
    col_lookup = {g: i for i, g in enumerate(groups)}
    active = df.loc[df.group_id.isin(col_lookup) & df.patient_key.isin(row_lookup)
                    & df.presence.astype(bool), ["patient_key", "group_id"]]
    rr = active.patient_key.map(row_lookup).to_numpy(dtype=int)
    cc = active.group_id.map(col_lookup).to_numpy(dtype=int)
    burden = sparse.csr_matrix((np.ones(len(active), dtype=np.float32), (rr, cc)),
                               shape=(len(patient_order), len(groups)))
    technical = np.array([[math.log1p(float(index.stats.loc[p, "unique_aa"])),
                           math.log1p(float(index.stats.loc[p, "read_count"]))] for p in patient_order], dtype=np.float32)
    mu = technical[:len(train_keys)].mean(axis=0)
    sd = technical[:len(train_keys)].std(axis=0)
    sd[sd == 0] = 1
    tech = sparse.csr_matrix((technical - mu) / sd)
    x = sparse.hstack([tech, burden], format="csr")
    y = np.array([labels[k] for k in train_keys], dtype=int)
    if len(np.unique(y)) != 2:
        return None, {"status": "blocked", "reason": "One-class outer training set"}
    clf = LogisticRegression(C=0.1, penalty="l2", solver="liblinear", max_iter=1000, random_state=0)
    clf.fit(x[:len(train_keys)], y)
    pred = clf.predict_proba(x[len(train_keys):])[:, 1]
    return pred, {"status": "complete", "eligible_groups": len(groups),
                  "total_fold_groups": int(df.group_id.nunique()),
                  "features": "technical log depth plus exact V-aware group presence",
                  "penalty": "fixed L2 C=0.1", "omitted_group": omit_group, "source_sha256": sha256(path)}



def group_baseline_available(prediction: np.ndarray | None, receipt: dict,
                             expected_source_sha256: str, n_test: int) -> bool:
    """Allow only a valid fold dictionary with no eligible group to lack predictions."""
    if receipt.get("source_sha256") != expected_source_sha256:
        raise ValueError("Exact V-aware group baseline source SHA-256 mismatch")
    if prediction is None:
        if (receipt.get("status") == "unavailable" and
                receipt.get("reason_code") == "no_eligible_group_ge10_outer_train" and
                receipt.get("eligible_groups") == 0):
            return False
        raise RuntimeError(f"Exact V-aware group baseline failed: {receipt}")
    values = np.asarray(prediction, dtype=float)
    if (receipt.get("status") != "complete" or values.shape != (n_test,) or
            not np.isfinite(values).all() or np.any((values < 0) | (values > 1))):
        raise ValueError("Exact V-aware group baseline predictions invalid")
    return True


def one_se_choice(inner_rows: pd.DataFrame, labels: dict[str, int]) -> tuple[str, int, pd.DataFrame]:
    """Compare patient-level mean seed prediction; one SE measured over patients."""
    summaries = []
    losses = {}
    for (model, bag), frame in inner_rows.groupby(["model", "bag_size"]):
        by_patient = frame.groupby("patient_key", as_index=False).prediction.mean()
        y = np.array([labels[k] for k in by_patient.patient_key])
        p = np.clip(by_patient.prediction.to_numpy(), 1e-6, 1 - 1e-6)
        patient_loss = -(y * np.log(p) + (1 - y) * np.log(1 - p))
        key = (str(model), int(bag))
        losses[key] = patient_loss
        summaries.append({"model": model, "bag_size": int(bag), "log_loss": float(patient_loss.mean()),
                          "se": float(patient_loss.std(ddof=1) / math.sqrt(len(patient_loss))),
                          "patients": len(patient_loss), **{f"metric_{k}": v for k, v in metrics(y, p).items()}})
    table = pd.DataFrame(summaries)
    compared = [set(group.patient_key) for _, group in inner_rows.groupby(["model", "bag_size"])]
    if (table.empty or table.patients.nunique() != 1 or
        any(keys != compared[0] for keys in compared) or
        any(inner_rows.loc[(inner_rows.model.eq(row.model) &
                            inner_rows.bag_size.eq(row.bag_size))]
            .groupby("patient_key").seed.nunique().ne(len(SEEDS)).any()
            for row in table.itertuples())):
        raise ValueError("Incomplete inner model comparisons")
    best = table.loc[table.log_loss.idxmin()]
    allowed = table.loc[table.log_loss.le(best.log_loss + best.se + 1e-12)].copy()
    allowed["model_order"] = allowed.model.map({m: i for i, m in enumerate(MODELS)})
    winner = allowed.sort_values(["bag_size", "model_order", "log_loss"]).iloc[0]
    return str(winner.model), int(winner.bag_size), table.drop(columns=[c for c in ["model_order"] if c in table])


def encoder_one_se_choices(inner_rows: pd.DataFrame, labels: dict[str, int]) -> dict[str, int]:
    """Freeze each encoder's bag using only its own inner held-out patients."""
    choices = {}
    for model in MODELS:
        scoped = inner_rows.loc[inner_rows.model.eq(model)]
        if set(scoped.bag_size) != set(BAGS):
            raise ValueError(f"Incomplete inner bag comparison for {model}")
        winner, bag, _ = one_se_choice(scoped, labels)
        if winner != model:
            raise ValueError(f"Inner encoder comparison changed model: {model}")
        choices[model] = bag
    if set(inner_rows.model) != set(MODELS):
        raise ValueError("Incomplete inner encoder comparison")
    return choices


def split_keys(folds: pd.DataFrame, outer: int, inner: int | None = None) -> tuple[list[str], list[str]]:
    block = folds.loc[folds.outer_fold.eq(outer)]
    if inner is None:
        train = block.loc[block.partition.eq("train"), "patient_key"].tolist()
        test = block.loc[block.partition.eq("test"), "patient_key"].tolist()
    else:
        if inner not in (0, 1, 2):
            raise ValueError("Inner fold must be 0..2")
        train = block.loc[block.partition.eq("train") & block.inner_fold.ne(inner), "patient_key"].tolist()
        test = block.loc[block.partition.eq("train") & block.inner_fold.eq(inner), "patient_key"].tolist()
    if set(train) & set(test):
        raise ValueError("Train/evaluation patient leakage")
    return train, test


def model_paths(root: Path, outer: int, model: str, bag: int, seed: int,
                inner: int | None) -> tuple[Path, Path]:
    base = root / "mil" / "cv" / f"outer_{outer}" / ("final" if inner is None else f"inner_{inner}")
    base = base / model / f"bag_{bag}" / f"seed_{seed}"
    return base / "prediction.tsv", base / "receipt.json"


def run_inner(root: Path, folds: pd.DataFrame, labels: dict[str, int], index: PatientIndex,
              outer: int, model_name: str, bag: int, seed: int, inner: int,
              bank: np.ndarray, pca: Projection, device: str, max_epochs: int, patience: int,
              max_gpu_gib: float, resume: bool, execution: dict) -> pd.DataFrame:
    tr, val = split_keys(folds, outer, inner)
    fit, stop = early_stop_split(tr, labels, outer, inner, seed)
    pred_path, receipt_path = model_paths(root, outer, model_name, bag, seed, inner)
    fingerprint = run_fingerprint(root, outer, inner, model_name, bag, seed, bank,
                                  tr, val, labels, execution)
    if resume and pred_path.exists() and receipt_path.exists():
        receipt = json.loads(receipt_path.read_text())
        saved = valid_saved_prediction(pred_path, val, labels, model_name, bag, seed, outer, inner)
        if (receipt.get("status") == "complete" and
                receipt.get("fingerprint") == fingerprint and
                receipt.get("prediction_sha256") == sha256(pred_path) and saved is not None):
            return saved
    stopper, stop_receipt = train_mil(
        fit, labels, index, bank, bag, seed, device,
        scope=f"outer{outer}-inner{inner}-earlystop-seed{seed}",
        val_keys=stop, max_epochs=max_epochs, patience=patience,
        max_gpu_gib=max_gpu_gib)
    selected_epochs = int(stop_receipt["best_epoch"])
    del stopper
    if device == "cuda":
        torch.cuda.empty_cache()
    model, receipt = train_mil(
        tr, labels, index, bank, bag, seed, device,
        scope=f"outer{outer}-inner{inner}-seed{seed}",
        fixed_epochs=selected_epochs, max_gpu_gib=max_gpu_gib)
    infer_start = time.monotonic()
    pred, sd, mc_draws, unresolved = predict_adaptive(
        model, val, index, bank, bag, device, scope=f"inner-outer{outer}-fold{inner}")
    inference_seconds = time.monotonic() - infer_start
    frame = pd.DataFrame({"patient_key": val, "label": [labels[k] for k in val],
                          "prediction": pred, "mc_sd": sd, "outer_fold": outer,
                          "inner_fold": inner, "model": model_name,
                          "bag_size": bag, "seed": seed})
    pred_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(pred_path, sep="\t", index=False)
    atomic_json(receipt_path, {**receipt, "status": "complete", "fingerprint": fingerprint,
                               "code_sha256": sha256(Path(__file__)),
                               "execution_config": execution,
                               "bank_path": str(Path(bank.filename).resolve()),
                               "prediction_sha256": sha256(pred_path),
                               "early_stop_fit_patient_sha256": train_patient_hash(fit),
                               "early_stop_validation_patient_sha256": train_patient_hash(stop),
                               "early_stop_receipt": stop_receipt,
                               "fixed_epochs": selected_epochs,
                               "decision_rules_sha256": load_decision_rules(root)[1],
                               "pca_raw_fallback": pca.raw_fallback,
                               "pca_fallback_reason": pca.fallback_reason,
                               "scope_pca_stability_sha256": pca.stability_sha256,
                               "validation_metrics": metrics(frame.label, frame.prediction),
                               "mc_draws": mc_draws, "mc_unresolved_sd_gt_0_05": unresolved,
                               "inference_seconds": round(inference_seconds, 3)})
    del model
    if device == "cuda":
        torch.cuda.empty_cache()
    return frame



def candidate_capture_diagnostic(root: Path, outer: int, index: PatientIndex,
                                 train_keys: list[str], patient_keys: list[str]) -> dict:
    """Exact hypergeometric probability of at least one safe candidate AA in a bag.

    These outer-train nominations are diagnostic only and never feed model or
    bag selection. Ambiguous V/J projections are excluded from AA capture.
    """
    from scipy.stats import hypergeom
    nomination_path = root / "groups/fold_nominations.tsv"
    safe_path = root / "groups" / f"outer_{outer}" / "patient_group_aa_safe.parquet"
    out = root / "mil/cv" / f"outer_{outer}"
    out.mkdir(parents=True, exist_ok=True)
    if not nomination_path.exists() or not safe_path.exists():
        result = {"status": "unavailable", "reason": "Train nominations or patient-safe AA map absent"}
        atomic_json(out / "candidate_capture.json", result)
        return result
    nominees = pd.read_csv(nomination_path, sep="\t", dtype={"group_id": str})
    required = {"group_id", "outer_fold", "scope", "train_patient_sha256"}
    if not required.issubset(nominees):
        raise ValueError(f"Candidate nominations missing columns: {sorted(required - set(nominees))}")
    nominees = nominees.loc[nominees.outer_fold.eq(outer)].copy()
    if not nominees.train_patient_sha256.eq(train_patient_hash(train_keys)).all() or not nominees.scope.eq(f"outer_{outer}").all():
        raise ValueError("Candidate capture nominations are not outer-train scoped")
    nominees = nominees.group_id.drop_duplicates().tolist()
    if not nominees:
        result = {"status": "unavailable", "reason": "No train-nominated groups"}
        atomic_json(out / "candidate_capture.json", result)
        return result
    safe = pd.read_parquet(safe_path, columns=["patient_key", "group_id", "seq_idx", "projection_exact"])
    if not safe.projection_exact.eq(True).all():
        raise ValueError("Candidate capture map contains ambiguous AA")
    safe["patient_key"] = safe.patient_key.astype(str)
    safe["group_id"] = safe.group_id.astype(str)
    counts = (safe.loc[safe.group_id.isin(nominees) & safe.patient_key.isin(patient_keys)]
              .groupby(["group_id", "patient_key"]).seq_idx.nunique())
    rows = []
    for group_id in nominees:
        for patient in patient_keys:
            n = int(index.stats.loc[patient, "unique_aa"])
            m = int(counts.get((group_id, patient), 0))
            for bag in BAGS:
                k = min(bag, n)
                prob = float(hypergeom.sf(0, n, m, k)) if m else 0.0
                rows.append({"outer_fold": outer, "group_id": group_id, "patient_key": patient,
                             "bag_size": bag, "patient_unique_aa": n, "safe_group_aa": m,
                             "capture_probability": prob, "expected_group_aa_in_bag": k * m / n})
    table = pd.DataFrame(rows)
    table.to_csv(out / "candidate_capture.tsv", sep="\t", index=False)
    summary = (table.loc[table.safe_group_aa.gt(0)]
               .groupby(["group_id", "bag_size"], as_index=False)
               .agg(carriers=("patient_key", "nunique"),
                    mean_capture_probability=("capture_probability", "mean"),
                    median_capture_probability=("capture_probability", "median")))
    summary.to_csv(out / "candidate_capture_summary.tsv", sep="\t", index=False)
    result = {"status": "complete", "groups": len(nominees), "patients": len(patient_keys),
              "method": "exact hypergeometric uniform-AA bag capture; safe V/J projection only; diagnostic not used for selection",
              "table": str(out / "candidate_capture.tsv")}
    atomic_json(out / "candidate_capture.json", result)
    return result


def validate_pilot_cost(root: Path) -> dict:
    """Require four three-epoch trials and complete-pool runtime evidence."""
    trials = []
    for model_name in MODELS:
        for bag in BAGS:
            path = root / "mil/pilot" / model_name / f"bag_{bag}" / "outer_0_inner_0.json"
            if not path.exists():
                raise FileNotFoundError(f"Four model x bag cost pilots required before MIL: {path}")
            trial = json.loads(path.read_text())
            pca_dir = root / "pca" / model_name / "outer_0"
            required = {
                "status": "complete_pilot_only", "outer": 0, "inner": 0,
                "model": model_name, "bag_size": bag, "device": "cuda",
                "source_code_sha256": sha256(Path(__file__)),
                "decision_rules_sha256": load_decision_rules(root)[1],
                "embedding_meta_sha256": sha256(root / "embeddings" / model_name / "meta.json"),
                "pca_sha256": sha256(pca_dir / "inner_0/pca.npz"),
                "scope_pca_stability_sha256": sha256(pca_dir / "inner_0/stability.json"),
                "pca_metadata_sha256": sha256(pca_dir / "inner_0/metadata.json"),
                "patient_index_receipt_sha256": sha256(root / "mil/patient_index/receipt.json"),
                "folds_sha256": sha256(root / "folds/folds.tsv"),
            }
            if any(trial.get(key) != value for key, value in required.items()):
                raise ValueError(f"Cost pilot provenance mismatch; rerun pilot: {path}")
            fit = trial.get("fit", {})
            if fit.get("epochs_ran") != 3 or fit.get("peak_gpu_gib", 0) > 12.0:
                raise ValueError(f"Cost pilot incomplete or above GPU policy: {path}")
            trials.append({"model": model_name, "bag_size": bag,
                           "fit_seconds_per_epoch": float(fit["seconds"]) / 3,
                           "projection_rows_per_second": trial["projection_rows_per_second"]})
    benchmark_path = root / "mil/pilot/full_pool_cost_benchmark.json"
    if not benchmark_path.exists():
        raise FileNotFoundError("Complete patient-pool cost benchmark required before formal MIL")
    benchmark = json.loads(benchmark_path.read_text())
    required_benchmark = {
        "status": "cost_only_complete_full_pool",
        "source_code_sha256": sha256(Path(__file__)),
        "patient_index_receipt_sha256": sha256(root / "mil/patient_index/receipt.json"),
        "bank_integrity_receipt_sha256": sha256(root / "mil/bank_integrity.json"),
    }
    cases = benchmark.get("cases", [])
    extrapolation = benchmark.get("extrapolation", {})
    if (any(benchmark.get(k) != v for k, v in required_benchmark.items()) or
            extrapolation.get("total_fit_count_upper") != 555 or
            extrapolation.get("outer_final_fits") != 45 or
            extrapolation.get("inner_early_stop_fits") != 180 or
            extrapolation.get("inner_full_train_refits") != 180 or
            extrapolation.get("max_gated_ablation_fits") != 150 or
            len(cases) != len(MODELS) * len(BAGS) or
            {(c.get("model"), c.get("bag_size")) for c in cases} !=
                {(model, bag) for model in MODELS for bag in BAGS} or
            any(c.get("fit_patients") != 12 or c.get("inference_draws_per_patient") != 8 or
                c.get("training_target") != "synthetic alternation independent of recorded N labels" or
                c.get("fit_peak_gpu_allocated_gib", 13) > LOCKED_GPU_GIB or
                c.get("inference_peak_gpu_allocated_gib", 13) > LOCKED_GPU_GIB or
                c.get("bank_sha256_from_meta") != json.loads((root / "embeddings" / c["model"] /
                                                                  "meta.json").read_text())["bank_sha256"]
                for c in cases)):
        raise ValueError("Complete patient-pool cost benchmark provenance mismatch")
    return {"status": "complete_cost_pilots", "trials": trials,
            "full_pool_benchmark_sha256": sha256(benchmark_path),
            "full_pool_extrapolation": benchmark["extrapolation"],
            "worst_case_fit_count": 555,
            "inner_early_stop_fit_count": 180, "inner_full_train_refit_count": 180,
            "outer_final_fit_count": 45,
            "gated_max_ablation_fit_count": 150,
            "max_epochs_per_inner_fit": 40,
            "warning": "At most three unique model/bag final pairs per outer; full-bank PCA projection and adaptive 32-bag inference add cost."}


def preflight_outer(root: Path, outer: int) -> dict:
    """Validate one outer fold's complete dependencies without starting GPU work."""
    if outer not in range(5):
        raise ValueError("Outer fold must be 0..4")
    require_primary_bank_receipt(root)
    cost = validate_pilot_cost(root)
    manifest = read_manifest(root)
    folds = read_folds(root, manifest)
    labels = read_home_labels(root, manifest)
    index = PatientIndex(root)
    train, test = split_keys(folds, outer)
    if not set(train + test).issubset(index.files):
        raise ValueError(f"Outer {outer} patient index incomplete")
    read_home_covariates(root, set(labels))
    load_decision_rules(root)
    validate_group_fold(root, outer, train)
    banks = {}
    for model_name in MODELS:
        bank = load_bank(root, model_name)
        banks[model_name] = list(bank.shape)
        if model_name == "sceptr_cdr3_only":
            sceptr_warning_ids(root, len(bank))
        pca_spec(root, model_name, outer, None, bank.shape[1])
        for inner in range(3):
            pca_spec(root, model_name, outer, inner, bank.shape[1])
    return {"status": "ready_outer_only", "outer_fold": outer,
            "train_patients": len(train), "heldout_patients": len(test),
            "banks": banks, "cost_pilot_gate": cost,
            "bank_integrity_receipt_sha256": sha256(root / "mil/bank_integrity.json")}


def run_outer(root: Path, outer: int, device: str, max_epochs: int, patience: int,
              max_gpu_gib: float, resume: bool, cpu_threads: int = 8) -> dict:
    if max_epochs != LOCKED_MAX_EPOCHS or patience != LOCKED_PATIENCE:
        raise ValueError("Formal MIL must use locked 40 epochs and patience 6")
    execution = execution_config(device, cpu_threads, max_gpu_gib)
    scoped_preflight = preflight_outer(root, outer)
    cost_pilot = scoped_preflight["cost_pilot_gate"]
    manifest = read_manifest(root)
    folds = read_folds(root, manifest)
    labels = read_home_labels(root, manifest)
    index = PatientIndex(root)
    train, test = split_keys(folds, outer)
    if not set(train + test).issubset(index.files):
        raise ValueError("Patient index missing Home patients")
    covariates = read_home_covariates(root, set(labels))
    _, rules_sha = load_decision_rules(root)
    fold_group_receipt = validate_group_fold(root, outer, train)
    early_group_pred, early_group_receipt = group_burden_baseline(
        root, outer, train, test, labels, index)
    group_available = group_baseline_available(
        early_group_pred, early_group_receipt,
        fold_group_receipt["patient_group_burden_sha256"], len(test))
    early_group_receipt = {**early_group_receipt, "outer_fold": outer,
                           "train_patient_sha256": train_patient_hash(train),
                           "test_patient_count": len(test)}
    group_receipt_path = root / "mil/cv" / f"outer_{outer}" / "group_burden_baseline.json"
    atomic_json(group_receipt_path, early_group_receipt)
    sceptr_bank = load_bank(root, "sceptr_cdr3_only")
    warning_ids, warning_sha = sceptr_warning_ids(root, len(sceptr_bank))
    warning_features = sceptr_warning_features(index, train + test, warning_ids)
    del sceptr_bank
    batch_composition = {}
    for part, keys in (("train", train), ("test", test)):
        subset = covariates.loc[keys, ["library_batch"]].copy()
        subset["label"] = [labels[k] for k in keys]
        batch_composition[part] = {f"{batch}|N{label}": int(count)
                                   for (batch, label), count in
                                   subset.groupby(["library_batch", "label"]).size().items()}
    inner_frames = []
    epoch_record: dict[tuple[str, int, int], list[int]] = defaultdict(list)
    for model_name in MODELS:
        raw = load_bank(root, model_name)
        for inner in range(3):
            spec = pca_spec(root, model_name, outer, inner, raw.shape[1])
            bank = project_bank(root, model_name, outer, inner, raw, spec, device)
            for bag in BAGS:
                for seed in SEEDS:
                    frame = run_inner(root, folds, labels, index, outer, model_name, bag,
                                      seed, inner, bank, spec, device, max_epochs, patience,
                                      max_gpu_gib, resume, execution)
                    inner_frames.append(frame)
                    _, receipt_path = model_paths(root, outer, model_name, bag, seed, inner)
                    epoch_record[(model_name, bag, seed)].append(int(json.loads(receipt_path.read_text())["best_epoch"]))
            del bank
    inner_rows = pd.concat(inner_frames, ignore_index=True)
    selected_model, selected_bag, comparison = one_se_choice(inner_rows, labels)
    summary_dir = root / "mil" / "cv" / f"outer_{outer}"
    summary_dir.mkdir(parents=True, exist_ok=True)
    inner_rows.to_csv(summary_dir / "inner_predictions.tsv", sep="\t", index=False)
    warning_features.to_csv(summary_dir / "sceptr_nonstandard_patient_features.tsv", sep="\t", index=False)
    sceptr_inner = (inner_rows.loc[inner_rows.model.eq("sceptr_cdr3_only")]
                    .groupby(["bag_size", "patient_key", "label"], as_index=False)
                    .prediction.mean())
    sceptr_inner["model_bag"] = sceptr_inner.bag_size.map(lambda x: f"sceptr_cdr3_only_bag_{x}")
    sceptr_inner = sceptr_inner.merge(warning_features, on="patient_key", validate="many_to_one")
    atomic_json(summary_dir / "sceptr_inner_domain_shift.json",
                {**warning_error_association(sceptr_inner, "model_bag"),
                 "warning_mask_sha256": warning_sha,
                 "data_scope": "repeated inner validation patients; descriptive only"})
    cost_rows = []
    for (candidate_model, candidate_bag), _ in inner_rows.groupby(["model", "bag_size"]):
        receipts = [json.loads(model_paths(root, outer, candidate_model, int(candidate_bag), seed, inner)[1].read_text())
                    for inner in range(3) for seed in SEEDS]
        cost_rows.append({"model": candidate_model, "bag_size": int(candidate_bag),
                          "train_seconds": sum(r["seconds"] for r in receipts),
                          "inference_seconds": sum(r.get("inference_seconds", 0) for r in receipts),
                          "mc_escalated_fits": sum(r.get("mc_draws", 8) > 8 for r in receipts),
                          "mc_unresolved_fits": sum(bool(r.get("mc_unresolved_sd_gt_0_05", False)) for r in receipts)})
    comparison = comparison.merge(pd.DataFrame(cost_rows), on=["model", "bag_size"], validate="one_to_one")
    comparison.to_csv(summary_dir / "inner_comparison.tsv", sep="\t", index=False)
    capture_receipt = candidate_capture_diagnostic(root, outer, index, train, train + test)
    encoder_choices = encoder_one_se_choices(inner_rows, labels)
    candidate_pairs = list(dict.fromkeys(
        [(selected_model, selected_bag)] +
        [(model, encoder_choices[model]) for model in MODELS]))
    final_by_candidate = {}
    spec_by_candidate = {}
    for candidate_model, candidate_bag in candidate_pairs:
        raw = load_bank(root, candidate_model)
        spec = pca_spec(root, candidate_model, outer, None, raw.shape[1])
        bank = project_bank(root, candidate_model, outer, None, raw, spec, device)
        final_frames = []
        for seed in SEEDS:
            pred_path, receipt_path = model_paths(root, outer, candidate_model, candidate_bag, seed, None)
            epochs = max(1, int(np.median(epoch_record[(candidate_model, candidate_bag, seed)])))
            epoch_sources = {str(inner): sha256(model_paths(
                root, outer, candidate_model, candidate_bag, seed, inner)[1]) for inner in range(3)}
            fingerprint = run_fingerprint(root, outer, None, candidate_model, candidate_bag,
                                          seed, bank, train, test, labels, execution,
                                          fixed_epochs=epochs)
            if resume and pred_path.exists() and receipt_path.exists():
                saved_receipt = json.loads(receipt_path.read_text())
                saved = valid_saved_prediction(pred_path, test, labels, candidate_model, candidate_bag, seed, outer, None)
                warning_columns = {"warning_attention_mass", "warning_attention_mass_sd", "warning_attention_draws"}
                if (saved_receipt.get("status") == "complete" and
                    saved_receipt.get("fingerprint") == fingerprint and
                    saved_receipt.get("prediction_sha256") == sha256(pred_path) and
                    saved_receipt.get("fixed_epochs") == epochs and
                    saved_receipt.get("epoch_source_receipt_sha256s") == epoch_sources and
                    saved_receipt.get("sceptr_warning_mask_sha256") == warning_sha and
                    saved is not None and warning_columns.issubset(saved.columns)):
                    final_frames.append(saved)
                    continue
            model, receipt = train_mil(train, labels, index, bank, candidate_bag, seed, device,
                                      scope=f"outer{outer}-final-seed{seed}",
                                      shuffle_scope=f"outer{outer}-final-seed{seed}",
                                      fixed_epochs=epochs, max_gpu_gib=max_gpu_gib)
            infer_start = time.monotonic()
            pred, sd, mc_draws, unresolved = predict_adaptive(
                model, test, index, bank, candidate_bag, device, scope=f"outer{outer}-test")
            inference_seconds = time.monotonic() - infer_start
            frame = pd.DataFrame({"patient_key": test, "label": [labels[k] for k in test],
                                  "prediction": pred, "mc_sd": sd, "outer_fold": outer,
                                  "model": candidate_model, "bag_size": candidate_bag, "seed": seed})
            if candidate_model == "sceptr_cdr3_only":
                attention_start = time.monotonic()
                attention = sceptr_warning_attention_mc(
                    model, test, index, bank, candidate_bag, device, warning_ids,
                    scope=f"outer{outer}-test", draws=8)
                frame = frame.merge(attention, on="patient_key", validate="one_to_one")
                attention_seconds = time.monotonic() - attention_start
            else:
                frame["warning_attention_mass"] = np.nan
                frame["warning_attention_mass_sd"] = np.nan
                frame["warning_attention_draws"] = 0
                attention_seconds = 0.0
            pred_path.parent.mkdir(parents=True, exist_ok=True)
            frame.to_csv(pred_path, sep="\t", index=False)
            atomic_json(receipt_path, {**receipt, "status": "complete", "fingerprint": fingerprint,
                                       "code_sha256": sha256(Path(__file__)),
                                       "execution_config": execution,
                                       "bank_path": str(Path(bank.filename).resolve()),
                                       "prediction_sha256": sha256(pred_path),
                                       "fixed_epochs": epochs,
                                       "epoch_source_receipt_sha256s": epoch_sources,
                                       "decision_rules_sha256": rules_sha,
                                       "mc_draws": mc_draws, "mc_unresolved_sd_gt_0_05": unresolved,
                                       "inference_seconds": round(inference_seconds, 3),
                                       "warning_attention_seconds": round(attention_seconds, 3),
                                       "sceptr_warning_mask_sha256": warning_sha,
                                       "pca_raw_fallback": spec.raw_fallback,
                                       "pca_fallback_reason": spec.fallback_reason,
                                       "scope_pca_stability_sha256": spec.stability_sha256,
                                       "pca_sha256": spec.source_sha256,
                                       "heldout_metrics": metrics(frame.label, frame.prediction)})
            final_frames.append(frame)
            del model
            if device == "cuda":
                torch.cuda.empty_cache()
        final_by_candidate[(candidate_model, candidate_bag)] = pd.concat(
            final_frames, ignore_index=True)
        spec_by_candidate[(candidate_model, candidate_bag)] = spec
        del raw, bank
    spec = spec_by_candidate[(selected_model, selected_bag)]
    seed_rows = final_by_candidate[(selected_model, selected_bag)]
    patient_pred = seed_rows.groupby(["patient_key", "label", "outer_fold"], as_index=False).agg(
        prediction=("prediction", "mean"), seed_sd=("prediction", "std"),
        mc_sd=("mc_sd", "mean"),
        warning_attention_mass=("warning_attention_mass", "mean"),
        warning_attention_mass_sd=("warning_attention_mass_sd", "mean"))
    patient_pred = patient_pred.merge(warning_features, on="patient_key", validate="one_to_one")
    patient_pred["selected_model"] = selected_model
    patient_pred["selected_bag"] = selected_bag
    patient_pred["prior"] = prior_baseline(train, patient_pred.patient_key.tolist(), labels)
    patient_pred["technical"] = technical_baseline(train, patient_pred.patient_key.tolist(), labels, index)
    clinical_pred, combined_pred, clinical_receipt = clinical_baselines(
        root, train, patient_pred.patient_key.tolist(), labels, index)
    patient_pred["clinical"] = clinical_pred
    patient_pred["clinical_technical_batch"] = combined_pred
    atomic_json(summary_dir / "clinical_baselines.json", clinical_receipt)
    patient_pred["group_burden"] = (
        pd.Series(early_group_pred, index=test).reindex(patient_pred.patient_key).to_numpy()
        if group_available else np.nan)
    outer_oof_path = summary_dir / "outer_oof.tsv"
    patient_pred.to_csv(outer_oof_path, sep="\t", index=False)
    encoder_frames = []
    for model_name in MODELS:
        bag = encoder_choices[model_name]
        block = final_by_candidate[(model_name, bag)].groupby(
            ["patient_key", "label", "outer_fold"], as_index=False).agg(
                prediction=("prediction", "mean"), seed_sd=("prediction", "std"),
                mc_sd=("mc_sd", "mean"),
                warning_attention_mass=("warning_attention_mass", "mean"),
                warning_attention_mass_sd=("warning_attention_mass_sd", "mean"))
        block["model"] = model_name
        block["bag_size"] = bag
        encoder_frames.append(block)
    encoder_oof = pd.concat(encoder_frames, ignore_index=True).sort_values(
        ["model", "patient_key"]).reset_index(drop=True)
    encoder_oof_path = summary_dir / "encoder_oof.tsv"
    encoder_oof.to_csv(encoder_oof_path, sep="\t", index=False)
    selection = {"status": "complete", "outer_fold": outer, "selected_model": selected_model,
                 "selected_bag": selected_bag, "pca_raw_fallback": spec.raw_fallback,
                 "pca_fallback_reason": spec.fallback_reason,
                 "scope_pca_stability_sha256": spec.stability_sha256,
                 "decision_rules_sha256": rules_sha,
                 "code_sha256": sha256(Path(__file__)),
                 "bank_integrity_receipt_sha256": sha256(root / "mil/bank_integrity.json"),
                 "outer_oof_sha256": sha256(outer_oof_path),
                 "encoder_oof_sha256": sha256(encoder_oof_path),
                 "encoder_choices": encoder_choices,
                 "encoder_choice_rule": "within each encoder: inner patient mean seed predictions; best log loss + one SE; smallest bag",
                 "encoder_final_unique_model_bag_pairs": len(candidate_pairs),
                 "encoder_metrics": {model: metrics(block.label, block.prediction)
                                     for model, block in encoder_oof.groupby("model")},
                 "inner_predictions_sha256": sha256(summary_dir / "inner_predictions.tsv"),
                 "inner_comparison_sha256": sha256(summary_dir / "inner_comparison.tsv"),
                 "clinical_baselines_sha256": sha256(summary_dir / "clinical_baselines.json"),
                 "group_burden_baseline_sha256": sha256(group_receipt_path),
                 "train_patients": len(train), "test_patients": len(test),
                 "selected_metrics": metrics(patient_pred.label, patient_pred.prediction),
                 "prior_metrics": metrics(patient_pred.label, patient_pred.prior),
                 "technical_metrics": metrics(patient_pred.label, patient_pred.technical),
                 "clinical_metrics": metrics(patient_pred.label, patient_pred.clinical),
                 "clinical_technical_batch_metrics": metrics(patient_pred.label, patient_pred.clinical_technical_batch),
                 "group_burden_metrics": (
                     metrics(patient_pred.label, patient_pred.group_burden) if group_available else None),
                 "group_burden_status": early_group_receipt,
                 "selection_rule": "inner patient mean seed predictions; best log loss + one SE; cheapest bag then model order",
                 "candidate_capture_diagnostic": capture_receipt,
                 "cost_pilot_gate": cost_pilot,
                 "sceptr_warning_mask_sha256": warning_sha,
                 "sceptr_warning_diagnostic": warning_error_association(patient_pred, "selected_model"),
                 "batch_label_composition": batch_composition}
    atomic_json(summary_dir / "selection.json", selection)
    return selection



def documented_blood_sensitivity(root: Path, frame: pd.DataFrame) -> dict:
    lock = root / "config/documented_blood31.tsv"
    if not lock.exists():
        return {"status": "blocked", "reason": "documented-blood patient lock absent"}
    selected = pd.read_csv(lock, sep="\t", dtype={"patient_key": str})
    if len(selected) != 31 or selected.patient_key.duplicated().any():
        raise ValueError("Documented-blood lock must contain 31 distinct patients")
    keys = set(selected.patient_key)
    if not keys.issubset(set(frame.patient_key)):
        raise ValueError("Documented-blood patient absent from OOF")
    sub = frame.loc[frame.patient_key.isin(keys)].copy()
    if sub.label.value_counts().to_dict() != {1: 18, 0: 13}:
        raise ValueError("Documented-blood labels drift")
    y = sub.label.to_numpy(dtype=int)
    p = np.clip(sub.prediction.to_numpy(dtype=float), 1e-6, 1 - 1e-6)
    b = np.clip(sub.technical.to_numpy(dtype=float), 1e-6, 1 - 1e-6)
    ll = lambda z: -(y * np.log(z) + (1 - y) * np.log(1 - z))
    gain = ll(b) - ll(p)
    rng = np.random.default_rng(20260929)
    boot = gain[rng.integers(0, len(gain), size=(5000, len(gain)))].mean(axis=1)
    result = {"status": "complete", "n": len(sub), "n0": 13, "nplus": 18,
              "mil": metrics(y, p), "technical": metrics(y, b),
              "mean_logloss_gain_vs_technical": float(gain.mean()),
              "bootstrap_95ci": [float(x) for x in np.quantile(boot, [0.025, 0.975])],
              "material_scope": _private_literal('private_f25535167b9356d7'),
              "method": _private_literal('private_8e2cf532e8264aec')}
    out = root / "mil/oof"
    out.mkdir(parents=True, exist_ok=True)
    sub.to_csv(out / "documented_blood31_oof.tsv", sep="\t", index=False)
    atomic_json(out / "documented_blood31_summary.json", result)
    return result


def validate_outer_output_provenance(root: Path, outer: int, selection: dict,
                                     folds: pd.DataFrame, labels: dict[str, int]) -> None:
    """Bind stored OOF values, comparators and final fits to current inputs."""
    require_primary_bank_receipt(root)
    base = root / "mil/cv" / f"outer_{outer}"
    files = {
        "outer_oof_sha256": base / "outer_oof.tsv",
        "encoder_oof_sha256": base / "encoder_oof.tsv",
        "inner_predictions_sha256": base / "inner_predictions.tsv",
        "inner_comparison_sha256": base / "inner_comparison.tsv",
        "clinical_baselines_sha256": base / "clinical_baselines.json",
        "group_burden_baseline_sha256": base / "group_burden_baseline.json",
    }
    if (selection.get("code_sha256") != sha256(Path(__file__)) or
            selection.get("decision_rules_sha256") != load_decision_rules(root)[1] or
            selection.get("bank_integrity_receipt_sha256") != sha256(root / "mil/bank_integrity.json") or
            any(selection.get(field) != sha256(path) for field, path in files.items())):
        raise ValueError(f"OOF outer {outer} output or source receipt SHA-256 mismatch")
    clinical_receipt = json.loads(files["clinical_baselines_sha256"].read_text())
    train, test = split_keys(folds, outer)
    if (clinical_receipt.get("source_sha256") != sha256(root / "config/home_clinical_covariates.tsv") or
            clinical_receipt.get("train_patient_sha256") != train_patient_hash(train) or
            clinical_receipt.get("test_patient_sha256") != train_patient_hash(test) or
            clinical_receipt.get("folds_sha256") != sha256(root / "folds/folds.tsv") or
            clinical_receipt.get("patient_index_receipt_sha256") != sha256(
                root / "mil/patient_index/receipt.json")):
        raise ValueError(f"OOF outer {outer} clinical comparator source/split SHA-256 mismatch")
    inner_rows = pd.read_csv(base / "inner_predictions.tsv", sep="\t",
                             dtype={"patient_key": str})
    expected_model, expected_bag, _ = one_se_choice(inner_rows, labels)
    encoder_choices = encoder_one_se_choices(inner_rows, labels)
    if (selection.get("selected_model") != expected_model or
            selection.get("selected_bag") != expected_bag or
            selection.get("encoder_choices") != encoder_choices or
            selection.get("encoder_final_unique_model_bag_pairs") != len(set(
                [(expected_model, expected_bag)] +
                [(model, bag) for model, bag in encoder_choices.items()]))):
        raise ValueError(f"OOF outer {outer} inner-only model/bag choices mismatch")
    final_pairs = set([(expected_model, expected_bag)] +
                      [(model, bag) for model, bag in encoder_choices.items()])
    checked_inner_rows = 0
    for model in MODELS:
        for bag in BAGS:
            for seed in SEEDS:
                inner_receipts = [model_paths(root, outer, model, bag, seed, inner)[1]
                                  for inner in range(3)]
                configs = []
                for inner, path in enumerate(inner_receipts):
                    inner_receipt = json.loads(path.read_text())
                    inner_train, inner_val = split_keys(folds, outer, inner)
                    inner_fit, inner_stop = early_stop_split(inner_train, labels, outer, inner, seed)
                    inner_pred = model_paths(root, outer, model, bag, seed, inner)[0]
                    saved = valid_saved_prediction(inner_pred, inner_val, labels,
                                                   model, bag, seed, outer, inner)
                    block = inner_rows.loc[
                        inner_rows.model.eq(model) & inner_rows.bag_size.eq(bag) &
                        inner_rows.seed.eq(seed) & inner_rows.inner_fold.eq(inner)]
                    if (saved is None or set(saved.columns) != set(block.columns) or
                            not saved.sort_values("patient_key").reset_index(drop=True).equals(
                                block[saved.columns].sort_values("patient_key").reset_index(drop=True))):
                        raise ValueError(f"OOF outer {outer} inner {inner} seed {seed} rows differ from fit output")
                    checked_inner_rows += len(saved)
                    if (inner_receipt.get("status") != "complete" or
                            inner_receipt.get("code_sha256") != sha256(Path(__file__)) or
                            inner_receipt.get("prediction_sha256") != sha256(inner_pred) or
                            inner_receipt.get("early_stop_fit_patient_sha256") != train_patient_hash(inner_fit) or
                            inner_receipt.get("early_stop_validation_patient_sha256") != train_patient_hash(inner_stop) or
                            inner_receipt.get("early_stop_receipt", {}).get("n_train") != len(inner_fit) or
                            inner_receipt.get("early_stop_receipt", {}).get("n_val") != len(inner_stop) or
                            inner_receipt.get("early_stop_receipt", {}).get("best_epoch") != inner_receipt.get("fixed_epochs") or
                            inner_receipt.get("fixed_epochs") != inner_receipt.get("best_epoch") or
                            inner_receipt.get("epochs_ran") != inner_receipt.get("best_epoch")):
                        raise ValueError(f"OOF outer {outer} inner {inner} seed {seed} source mismatch")
                    inner_config = inner_receipt.get("execution_config")
                    validate_execution_config(inner_config)
                    configs.append(inner_config)
                    inner_bank = Path(inner_receipt["bank_path"])
                    expected_inner_bank = (root / "embeddings" / model / "bank_fp16.npy" if
                                           inner_receipt.get("pca_raw_fallback") else
                                           root / "mil/projections" / model /
                                           f"outer_{outer}_inner_{inner}_d{inner_receipt.get('dim')}.npy")
                    if inner_bank != expected_inner_bank.resolve():
                        raise ValueError(f"OOF outer {outer} inner {inner} seed {seed} bank path mismatch")
                    inner_bank_array = np.load(expected_inner_bank, mmap_mode="r", allow_pickle=False)
                    current_inner = run_fingerprint(root, outer, inner, model, bag, seed,
                                                    inner_bank_array, inner_train, inner_val,
                                                    labels, inner_config)
                    if inner_receipt.get("fingerprint") != current_inner:
                        raise ValueError(f"OOF outer {outer} inner {inner} seed {seed} fingerprint mismatch")
                if len({json.dumps(c, sort_keys=True) for c in configs}) != 1:
                    raise ValueError(f"OOF outer {outer} inner execution settings differ")
                if (model, bag) not in final_pairs:
                    continue
                pred_path, receipt_path = model_paths(root, outer, model, bag, seed, None)
                receipt = json.loads(receipt_path.read_text())
                config = receipt.get("execution_config")
                validate_execution_config(config)
                if any(inner_config != config for inner_config in configs):
                    raise ValueError(f"OOF outer {outer} inner/final settings mismatch")
                epochs = receipt.get("fixed_epochs")
                expected_sources = {str(inner): sha256(path)
                                    for inner, path in enumerate(inner_receipts)}
                expected_epochs = int(np.median([
                    json.loads(path.read_text())["best_epoch"] for path in inner_receipts]))
                raw = bool(receipt.get("pca_raw_fallback"))
                expected_bank = (root / "embeddings" / model / "bank_fp16.npy" if raw else
                                 root / "mil/projections" / model /
                                 f"outer_{outer}_d{receipt.get('dim')}.npy")
                if (receipt.get("status") != "complete" or
                        receipt.get("code_sha256") != sha256(Path(__file__)) or
                        receipt.get("prediction_sha256") != sha256(pred_path) or
                        receipt.get("bank_path") != str(expected_bank.resolve()) or
                        receipt.get("epoch_source_receipt_sha256s") != expected_sources or
                        epochs != expected_epochs or receipt.get("epochs_ran") != epochs):
                    raise ValueError(f"OOF outer {outer} {model} seed {seed} fit or prediction provenance mismatch")
                bank = np.load(expected_bank, mmap_mode="r", allow_pickle=False)
                spec = pca_spec(root, model, outer, None,
                                np.load(root / "embeddings" / model / "bank_fp16.npy",
                                        mmap_mode="r", allow_pickle=False).shape[1])
                if (raw != spec.raw_fallback or receipt.get("pca_sha256") != spec.source_sha256 or
                        receipt.get("scope_pca_stability_sha256") != spec.stability_sha256):
                    raise ValueError(f"OOF outer {outer} {model} final PCA scope mismatch")
                current = run_fingerprint(root, outer, None, model, bag, seed, bank,
                                          train, test, labels, config, fixed_epochs=epochs)
                if receipt.get("fingerprint") != current:
                    raise ValueError(f"OOF outer {outer} {model} seed {seed} current fingerprint mismatch")
    if checked_inner_rows != len(inner_rows):
        raise ValueError(f"OOF outer {outer} inner comparison has extra rows")
    primary_receipt = json.loads(model_paths(root, outer, expected_model,
                                             expected_bag, SEEDS[0], None)[1].read_text())
    if (selection.get("pca_raw_fallback") != primary_receipt.get("pca_raw_fallback") or
            selection.get("scope_pca_stability_sha256") !=
                primary_receipt.get("scope_pca_stability_sha256")):
        raise ValueError(f"OOF outer {outer} primary PCA receipt mismatch")


def validate_pooled_oof_provenance(root: Path, frame: pd.DataFrame) -> None:
    summary_path = root / "mil/oof/summary.json"
    if not summary_path.exists():
        raise ValueError("Pooled OOF summary receipt absent")
    summary = json.loads(summary_path.read_text())
    pooled_path = root / _private_literal('private_4b09056568ab6f20')
    if (summary.get("status") != "complete" or
            summary.get(_private_literal('private_365a4bb69941e1fd')) != sha256(pooled_path)):
        raise ValueError("Pooled OOF SHA-256 differs from summary receipt")
    outer = pd.concat([pd.read_csv(root / "mil/cv" / f"outer_{o}" / "outer_oof.tsv",
                                   sep="\t", dtype={"patient_key": str})
                       for o in range(5)], ignore_index=True)
    if not frame.equals(outer):
        raise ValueError("Pooled OOF values differ from verified outer-fold outputs")
    encoder_path = root / _private_literal('private_ee6d9683e4c22c9b')
    comparison_path = root / "mil/oof/encoder_comparison.json"
    if (summary.get(_private_literal('private_2d712dab280e0a2d')) != sha256(encoder_path) or
            summary.get("encoder_comparison_sha256") != sha256(comparison_path)):
        raise ValueError("Pooled encoder OOF or comparison SHA-256 differs from summary receipt")
    encoder = pd.read_csv(encoder_path, sep="\t", dtype={"patient_key": str})
    outer_encoder = pd.concat([
        pd.read_csv(root / "mil/cv" / f"outer_{o}" / "encoder_oof.tsv",
                    sep="\t", dtype={"patient_key": str}) for o in range(5)],
        ignore_index=True)
    if not encoder.equals(outer_encoder):
        raise ValueError("Pooled encoder OOF values differ from verified outer-fold outputs")
    comparison = json.loads(comparison_path.read_text())
    if (comparison.get("status") != "complete_descriptive_secondary" or
            comparison.get(_private_literal('private_2d712dab280e0a2d')) != sha256(encoder_path) or
            comparison.get(_private_literal('private_3d9311639f5bcb17')) != sha256(pooled_path) or
            comparison.get("code_sha256") != sha256(Path(__file__)) or
            comparison.get("outer_selection_sha256s") != {
                str(o): sha256(root / "mil/cv" / f"outer_{o}" / "selection.json")
                for o in range(5)}):
        raise ValueError("Encoder comparison source provenance mismatch")


def validate_home_oof(root: Path, frame: pd.DataFrame) -> dict[int, np.ndarray]:
    """Verify exact Home held-out identities and the three final seed predictions."""
    required = {"patient_key", "label", "outer_fold", "selected_model",
                "selected_bag", "prediction"}
    if not required.issubset(frame):
        raise ValueError(f"OOF table missing columns: {sorted(required - set(frame))}")
    manifest = read_manifest(root)
    labels = read_home_labels(root, manifest)
    folds = read_folds(root, manifest)
    expected_test = folds.loc[folds.partition.eq("test"),
                              ["patient_key", "outer_fold"]].set_index("patient_key")
    if (len(frame) != len(labels) or frame.patient_key.duplicated().any() or
            set(frame.patient_key.astype(str)) != set(labels) or
            not pd.to_numeric(frame.label, errors="coerce").eq(frame.patient_key.map(labels)).all() or
            not pd.to_numeric(frame.outer_fold, errors="coerce").eq(
                frame.patient_key.map(expected_test.outer_fold)).all()):
        raise ValueError("OOF Home patient keys, labels, or held-out folds differ from frozen split")
    probabilities = pd.to_numeric(frame.prediction, errors="coerce").to_numpy(dtype=float)
    if not np.isfinite(probabilities).all() or np.any((probabilities < 0) | (probabilities > 1)):
        raise ValueError("OOF patient prediction is invalid")
    seed_predictions = {seed: np.empty(len(frame), dtype=float) for seed in SEEDS}
    for outer in range(5):
        block = frame.loc[frame.outer_fold.eq(outer)]
        expected_keys = expected_test.loc[expected_test.outer_fold.eq(outer)].index.tolist()
        if set(block.patient_key) != set(expected_keys):
            raise ValueError(f"OOF outer {outer} patient membership differs from frozen split")
        if block.selected_model.nunique() != 1 or block.selected_bag.nunique() != 1:
            raise ValueError(f"OOF outer {outer} model/bag selection is not unique")
        model = str(block.selected_model.iloc[0])
        bag_value = pd.to_numeric(block.selected_bag, errors="coerce")
        bag = int(bag_value.iloc[0]) if pd.notna(bag_value.iloc[0]) else -1
        if not bag_value.eq(bag).all() or model not in MODELS or bag not in BAGS:
            raise ValueError(f"OOF outer {outer} model/bag is outside frozen comparison")
        selection_path = root / "mil/cv" / f"outer_{outer}" / "selection.json"
        selection = json.loads(selection_path.read_text())
        if (selection.get("status") != "complete" or
                selection.get("outer_fold") != outer or
                selection.get("selected_model") != model or
                selection.get("selected_bag") != bag):
            raise ValueError(f"OOF outer {outer} selection receipt mismatch")
        validate_outer_output_provenance(root, outer, selection, folds, labels)
        final = []
        for seed in SEEDS:
            pred_path, receipt_path = model_paths(root, outer, model, bag, seed, None)
            receipt = json.loads(receipt_path.read_text())
            if receipt.get("status") != "complete":
                raise ValueError(f"OOF outer {outer} seed {seed} fit incomplete")
            saved = valid_saved_prediction(pred_path, expected_keys, labels,
                                           model, bag, seed, outer, None)
            if saved is None:
                raise ValueError(f"OOF outer {outer} seed {seed} prediction mismatch")
            pred = saved.set_index("patient_key").loc[block.patient_key, "prediction"].to_numpy(dtype=float)
            seed_predictions[seed][np.flatnonzero(frame.outer_fold.to_numpy() == outer)] = pred
            final.append(pred)
        mean = np.mean(final, axis=0)
        if not np.allclose(mean, block.prediction.to_numpy(dtype=float),
                           rtol=1e-10, atol=1e-12):
            raise ValueError(f"OOF outer {outer} prediction differs from three seed mean")
    return seed_predictions


def validate_encoder_oof(root: Path, frame: pd.DataFrame,
                         primary_frame: pd.DataFrame | None = None) -> None:
    """Require paired held-out patients and current three-seed means per encoder."""
    required = {"patient_key", "label", "outer_fold", "model", "bag_size",
                "prediction", "seed_sd", "mc_sd", "warning_attention_mass",
                "warning_attention_mass_sd"}
    if not required.issubset(frame):
        raise ValueError(f"Encoder OOF missing columns: {sorted(required - set(frame))}")
    manifest = read_manifest(root)
    labels = read_home_labels(root, manifest)
    folds = read_folds(root, manifest)
    if (len(frame) != len(labels) * len(MODELS) or
            frame.duplicated(["patient_key", "model"]).any() or
            set(frame.model) != set(MODELS)):
        raise ValueError("Encoder OOF must contain each Home patient once per encoder")
    if primary_frame is not None:
        primary_keys = primary_frame[["patient_key", "label", "outer_fold"]].sort_values(
            "patient_key").reset_index(drop=True)
    for outer in range(5):
        _, test = split_keys(folds, outer)
        selection = json.loads((root / "mil/cv" / f"outer_{outer}" / "selection.json").read_text())
        choices = selection.get("encoder_choices")
        if not isinstance(choices, dict) or set(choices) != set(MODELS):
            raise ValueError(f"Encoder OOF outer {outer} inner choices absent")
        for model in MODELS:
            bag = choices[model]
            if bag not in BAGS:
                raise ValueError(f"Encoder OOF outer {outer} bag outside locked choices")
            block = frame.loc[frame.outer_fold.eq(outer) & frame.model.eq(model)].copy()
            if (len(block) != len(test) or set(block.patient_key) != set(test) or
                    not pd.to_numeric(block.label, errors="coerce").eq(
                        block.patient_key.map(labels)).all() or
                    not pd.to_numeric(block.bag_size, errors="coerce").eq(bag).all()):
                raise ValueError(f"Encoder OOF outer {outer} {model} patient/label/bag mismatch")
            seed_frames = []
            for seed in SEEDS:
                pred_path, receipt_path = model_paths(root, outer, model, bag, seed, None)
                receipt = json.loads(receipt_path.read_text())
                saved = valid_saved_prediction(pred_path, test, labels, model, bag,
                                               seed, outer, None)
                if (saved is None or receipt.get("status") != "complete" or
                        receipt.get("prediction_sha256") != sha256(pred_path)):
                    raise ValueError(f"Encoder OOF outer {outer} {model} seed {seed} source mismatch")
                seed_frames.append(saved)
            expected = pd.concat(seed_frames).groupby(
                ["patient_key", "label", "outer_fold"], as_index=False).agg(
                    prediction=("prediction", "mean"), seed_sd=("prediction", "std"),
                    mc_sd=("mc_sd", "mean"),
                    warning_attention_mass=("warning_attention_mass", "mean"),
                    warning_attention_mass_sd=("warning_attention_mass_sd", "mean"))
            expected["model"] = model
            expected["bag_size"] = bag
            actual = block[expected.columns].sort_values("patient_key").reset_index(drop=True)
            expected = expected.sort_values("patient_key").reset_index(drop=True)
            try:
                pd.testing.assert_frame_equal(actual, expected, check_dtype=False,
                                              check_exact=False, rtol=1e-10, atol=1e-12)
            except AssertionError as exc:
                raise ValueError(f"Encoder OOF outer {outer} {model} differs from three seed means") from exc
    if primary_frame is not None:
        for model in MODELS:
            side_keys = frame.loc[frame.model.eq(model),
                                  ["patient_key", "label", "outer_fold"]].sort_values(
                                      "patient_key").reset_index(drop=True)
            if not side_keys.equals(primary_keys):
                raise ValueError(f"Encoder OOF {model} patients differ from primary OOF")


def group_baseline_oof_status(root: Path, frame: pd.DataFrame) -> dict:
    """Require each fold's group prediction or its audited no-eligible-group reason."""
    if "group_burden" not in frame:
        raise ValueError("OOF group-burden column missing")
    unavailable = []
    folds = read_folds(root, read_manifest(root))
    for outer in range(5):
        train, test = split_keys(folds, outer)
        block = frame.loc[frame.outer_fold.eq(outer)]
        path = root / "mil/cv" / f"outer_{outer}" / "group_burden_baseline.json"
        receipt = json.loads(path.read_text())
        fold_receipt = validate_group_fold(root, outer, train)
        if (receipt.get("source_sha256") != fold_receipt["patient_group_burden_sha256"] or
                receipt.get("outer_fold") != outer or
                receipt.get("train_patient_sha256") != train_patient_hash(train) or
                receipt.get("test_patient_count") != len(test) or
                set(block.patient_key) != set(test)):
            raise ValueError(f"OOF outer {outer} group baseline source receipt mismatch")
        values = pd.to_numeric(block.group_burden, errors="coerce").to_numpy(dtype=float)
        if receipt.get("status") == "complete":
            if not np.isfinite(values).all() or np.any((values < 0) | (values > 1)):
                raise ValueError(f"OOF outer {outer} complete group baseline has invalid values")
        elif (receipt.get("status") == "unavailable" and
              receipt.get("reason_code") == "no_eligible_group_ge10_outer_train" and
              receipt.get("eligible_groups") == 0):
            if block.group_burden.notna().any():
                raise ValueError(f"OOF outer {outer} unavailable group baseline has predictions")
            unavailable.append(outer)
        else:
            raise ValueError(f"OOF outer {outer} group baseline status invalid")
    return {"status": "complete" if not unavailable else "unavailable",
            "unavailable_outer_folds": unavailable,
            "reason": None if not unavailable else "No group reached ten outer-train carriers in these folds"}


def summarize_all(root: Path) -> dict:
    paths = [root / "mil/cv" / f"outer_{o}" / "outer_oof.tsv" for o in range(5)]
    if not all(p.exists() for p in paths):
        raise RuntimeError("All five outer predictions required")
    frame = pd.concat([pd.read_csv(p, sep="\t", dtype={"patient_key": str}) for p in paths], ignore_index=True)
    validate_home_oof(root, frame)
    encoder_paths = [root / "mil/cv" / f"outer_{o}" / "encoder_oof.tsv" for o in range(5)]
    if not all(p.exists() for p in encoder_paths):
        raise RuntimeError("Both encoder predictions required in all five outer folds")
    encoder_frame = pd.concat([pd.read_csv(p, sep="\t", dtype={"patient_key": str})
                               for p in encoder_paths], ignore_index=True)
    validate_encoder_oof(root, encoder_frame, frame)
    group_status = group_baseline_oof_status(root, frame)
    out = root / "mil/oof"
    out.mkdir(parents=True, exist_ok=True)
    pooled_path = out / _private_literal('private_62deb35004db4a1b')
    frame.to_csv(pooled_path, sep="\t", index=False)
    encoder_path = out / _private_literal('private_2e439bd6a0651a3d')
    encoder_frame.to_csv(encoder_path, sep="\t", index=False)
    by_model = {model: metrics(block.label, block.prediction)
                for model, block in encoder_frame.groupby("model")}
    paired = encoder_frame.pivot(index="patient_key", columns="model", values="prediction")
    paired_y = frame.set_index("patient_key").loc[paired.index, "label"].to_numpy(dtype=int)
    paired_difference = (patient_logloss(paired_y, paired["sceptr_cdr3_only"]) -
                         patient_logloss(paired_y, paired["tcrbert_dualtask_aa_mean"]))
    encoder_comparison = {
        "status": "complete_descriptive_secondary", "comparison_scope": _private_literal('private_6c34db28dbb3b20b'),
        "difference_direction": "SCEPTR minus TCR-BERT patient log loss; negative favors SCEPTR",
        _private_literal('private_2d712dab280e0a2d'): sha256(encoder_path),
        _private_literal('private_3d9311639f5bcb17'): sha256(pooled_path),
        "code_sha256": sha256(Path(__file__)),
        "outer_selection_sha256s": {str(o): sha256(root / "mil/cv" / f"outer_{o}" / "selection.json")
                                    for o in range(5)},
        "model_metrics": by_model,
        "paired_patient_logloss_sceptr_minus_tcrbert": paired_effect(paired_difference),
        "per_outer_fold": {str(o): {model: metrics(block.label, block.prediction)
                                    for model, block in encoder_frame.loc[
                                        encoder_frame.outer_fold.eq(o)].groupby("model")}
                           for o in range(5)},
    }
    encoder_comparison_path = out / "encoder_comparison.json"
    atomic_json(encoder_comparison_path, encoder_comparison)
    result = {"status": "complete", "n_patients": 114,
              _private_literal('private_365a4bb69941e1fd'): sha256(pooled_path),
              _private_literal('private_2d712dab280e0a2d'): sha256(encoder_path),
              "encoder_comparison_sha256": sha256(encoder_comparison_path),
              "encoder_model_metrics": by_model,
              "code_sha256": sha256(Path(__file__)),
              "mil": metrics(frame.label, frame.prediction),
              "prior": metrics(frame.label, frame.prior),
              "technical": metrics(frame.label, frame.technical),
              "clinical": metrics(frame.label, frame.clinical),
              "clinical_technical_batch": metrics(frame.label, frame.clinical_technical_batch),
              "group_burden": (
                  metrics(frame.label, frame.group_burden)
                  if group_status["status"] == "complete" else None),
              "group_burden_status": group_status,
              "folds": [json.loads((root / "mil/cv" / f"outer_{o}" / "selection.json").read_text()) for o in range(5)],
              "endpoint": "recorded N; pN/ypN prefix unverified",
              "material_scope": "31 Home documented peripheral blood; 83 Home material unverified",
              "candidate_status": (
                  "not assessed; requires fold-aligned group family, external direction and gated ablation"
                  if group_status["status"] == "complete"
                  else "blocked; exact V-aware group baseline unavailable"),
              "sceptr_domain_shift_diagnostic": warning_error_association(frame, "selected_model"),
              "documented_blood31": documented_blood_sensitivity(root, frame)}
    atomic_json(out / "summary.json", result)
    return result


def preflight(root: Path) -> dict:
    manifest = read_manifest(root)
    folds = read_folds(root, manifest)
    index = PatientIndex(root)
    home = set(manifest.loc[manifest.cohort.eq("home"), "patient_key"].astype(str))
    if not home.issubset(index.files):
        raise ValueError("Missing indexed Home patient")
    read_home_covariates(root, home)
    load_decision_rules(root)
    bank_integrity = verify_primary_bank_bytes(root)
    cost_pilots = validate_pilot_cost(root)
    for outer in range(5):
        train, _ = split_keys(folds, outer)
        validate_group_fold(root, outer, train)
    dims = {}
    for model in MODELS:
        bank = load_bank(root, model)
        dims[model] = list(bank.shape)
        if model == "sceptr_cdr3_only":
            sceptr_warning_ids(root, len(bank))
        for outer in range(5):
            pca_spec(root, model, outer, None, bank.shape[1])
            for inner in range(3):
                pca_spec(root, model, outer, inner, bank.shape[1])
    return {"status": "ready", "home_patients": len(home), "fold_rows": len(folds),
            "banks": dims, "cost_pilots": cost_pilots,
            "bank_integrity_receipt_sha256": sha256(root / "mil/bank_integrity.json"),
            "bank_integrity_checked_utc": bank_integrity["checked_utc"]}




def train_patient_hash(keys: list[str]) -> str:
    return hashlib.sha256(("\n".join(sorted(keys)) + "\n").encode()).hexdigest()


def patient_logloss(y: np.ndarray, p: np.ndarray) -> np.ndarray:
    p = np.clip(np.asarray(p, dtype=float), 1e-6, 1 - 1e-6)
    y = np.asarray(y, dtype=int)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))


def paired_effect(values: np.ndarray, seed: int = 20260929) -> dict:
    values = np.asarray(values, dtype=float)
    if not np.isfinite(values).all() or len(values) == 0:
        raise ValueError("Invalid paired effects")
    rng = np.random.default_rng(seed)
    draws = values[rng.integers(0, len(values), size=(5000, len(values)))].mean(axis=1)
    signs = rng.choice([-1, 1], size=(20000, len(values)))
    null = (signs * values).mean(axis=1)
    p_one_sided = (1 + np.count_nonzero(null >= values.mean())) / (len(null) + 1)
    return {"mean": float(values.mean()),
            "bootstrap_95ci": [float(x) for x in np.quantile(draws, [0.025, 0.975])],
            "signflip_one_sided_p": float(p_one_sided)}


def bh(p_values: list[float]) -> list[float]:
    p = np.asarray(p_values, dtype=float)
    if np.any((p < 0) | (p > 1) | ~np.isfinite(p)):
        raise ValueError("Invalid p values")
    order = np.argsort(p)
    adjusted = np.minimum.accumulate((p[order] * len(p) / np.arange(1, len(p) + 1))[::-1])[::-1]
    out = np.empty(len(p))
    out[order] = np.minimum(adjusted, 1)
    return out.tolist()


def ablation_gate(root: Path) -> dict:
    """Require stable pooled OOF gain over the exact V-aware group baseline."""
    p = root / _private_literal('private_4b09056568ab6f20')
    if not p.exists():
        raise RuntimeError(_private_literal('private_4f9c3ed23d76a99d'))
    rules, rules_sha = load_decision_rules(root)
    frame = pd.read_csv(p, sep="\t", dtype={"patient_key": str})
    validate_pooled_oof_provenance(root, frame)
    seed_predictions = validate_home_oof(root, frame)
    group_status = group_baseline_oof_status(root, frame)
    if group_status["status"] != "complete":
        return {"status": "blocked", "decision_rules_sha256": rules_sha,
                "oof_sha256": sha256(p), "group_burden_status": group_status,
                "reason": "Exact V-aware group-burden OOF baseline unavailable; cluster contribution cannot be evaluated"}
    y = frame.label.to_numpy(dtype=int)
    loss_mil = patient_logloss(y, frame.prediction)
    loss_group = patient_logloss(y, frame.group_burden)
    loss_technical = patient_logloss(y, frame.technical)
    loss_combined = patient_logloss(y, frame.clinical_technical_batch)
    effect = paired_effect(loss_group - loss_mil)
    combined_effect = paired_effect(loss_combined - loss_mil)
    technical_effect = paired_effect(loss_technical - loss_mil)
    fold_gain = {int(fold): float((patient_logloss(g.label, g.group_burden) -
                                   patient_logloss(g.label, g.prediction)).mean())
                 for fold, g in frame.groupby("outer_fold")}
    combined_fold_gain = {int(fold): float((patient_logloss(g.label, g.clinical_technical_batch) -
                                            patient_logloss(g.label, g.prediction)).mean())
                          for fold, g in frame.groupby("outer_fold")}
    technical_fold_gain = {int(fold): float((patient_logloss(g.label, g.technical) -
                                             patient_logloss(g.label, g.prediction)).mean())
                           for fold, g in frame.groupby("outer_fold")}
    model_rule = rules["model_gate"]
    comparators = set(model_rule["comparators"])
    if comparators != {"group_burden", "clinical_technical_batch"}:
        raise ValueError("Unsupported locked MIL comparator set")
    seed_gains = {}
    for seed, pred in seed_predictions.items():
        seed_loss = patient_logloss(y, pred)
        seed_gains[str(seed)] = {
            comparator: float((patient_logloss(y, frame[comparator]) - seed_loss).mean())
            for comparator in sorted(comparators)}
    min_seed = int(model_rule["positive_seeds_min"])
    min_fold = int(model_rule["positive_outer_folds_min"])
    lower = float(model_rule["pooled_patient_bootstrap_lower_gt"])
    passed = (effect["bootstrap_95ci"][0] > lower and
              combined_effect["bootstrap_95ci"][0] > lower and
              sum(v > 0 for v in fold_gain.values()) >= min_fold and
              sum(v > 0 for v in combined_fold_gain.values()) >= min_fold and
              all(sum(seed_gains[str(seed)][comparator] > 0 for seed in SEEDS) >= min_seed
                  for comparator in comparators))
    return {"status": "pass" if passed else "blocked",
            "criterion": "locked pooled patient bootstrap, >=4/5 outer folds and >=2/3 seed directions for both comparators",
            "decision_rules_sha256": rules_sha,
            "oof_sha256": sha256(p),
            "seed_gain_by_comparator": seed_gains,
            "paired_gain_vs_group_burden": effect, "fold_gain_vs_group_burden": fold_gain,
            "paired_gain_vs_clinical_technical_batch": combined_effect,
            "fold_gain_vs_clinical_technical_batch": combined_fold_gain,
            "paired_gain_vs_technical_sensitivity": technical_effect,
            "fold_gain_vs_technical_sensitivity": technical_fold_gain}


def matched_random_exclude(index: PatientIndex, target: dict[str, set[int]],
                           patient_keys: list[str], seed: int,
                           return_diagnostics: bool = False) -> dict[str, set[int]] | tuple[dict[str, set[int]], dict]:
    """Match within-patient deletion count and nearest log10 read-count bins."""
    out: dict[str, set[int]] = {}
    diagnostics = {}
    for key in patient_keys:
        wanted = target.get(key, set())
        if not wanted:
            continue
        seq, reads = index.get(key)
        is_target = np.isin(seq, np.fromiter(wanted, dtype=np.int64))
        if int(is_target.sum()) != len(wanted):
            raise ValueError(f"Target group AA absent from patient index: {key}")
        bins = np.floor(np.log10(reads)).astype(int)
        rng = np.random.default_rng(stable_seed("matched-v1", seed, key))
        chosen = []
        available = ~is_target
        for b in np.unique(bins[is_target]):
            need = int(np.count_nonzero(is_target & bins.__eq__(b)))
            candidates = np.flatnonzero(available & bins.__eq__(b))
            same_count = min(len(candidates), need)
            if same_count:
                take = rng.choice(candidates, size=same_count, replace=False)
                chosen.extend(seq[take].tolist())
                available[take] = False
            remaining = need - same_count
            if remaining:
                alternatives = np.flatnonzero(available)
                if len(alternatives) < remaining:
                    raise ValueError(f"Insufficient matched AA controls for patient {key}")
                distances = np.abs(bins[alternatives] - b)
                for distance in np.unique(distances):
                    candidates = alternatives[distances == distance]
                    take_n = min(len(candidates), remaining)
                    if take_n:
                        take = rng.choice(candidates, size=take_n, replace=False)
                        chosen.extend(seq[take].tolist())
                        available[take] = False
                        remaining -= take_n
                    if remaining == 0:
                        break
        out[key] = set(chosen)
        if len(out[key]) != len(wanted):
            raise AssertionError("Random removal count mismatch")
        target_bins = np.sort(bins[is_target])
        chosen_bins = np.sort(bins[np.isin(seq, np.fromiter(out[key], dtype=np.int64))])
        gaps = np.abs(target_bins - chosen_bins)
        diagnostics[key] = {"target_aa": len(wanted), "exact_bin_matches": int(np.count_nonzero(gaps == 0)),
                            "total_absolute_log10_bin_gap": int(gaps.sum())}
    return (out, diagnostics) if return_diagnostics else out


def validate_ablation_predictions(frame: pd.DataFrame, test: list[str],
                                  labels: dict[str, int], group_id: str) -> None:
    """Require all three seeds and both treatments for every held-out patient."""
    treatments = {"group_removed", "matched_random"}
    required = {"patient_key", "label", "prediction", "seed", "treatment", "group_id"}
    if not required.issubset(frame):
        raise ValueError(f"Ablation predictions missing {sorted(required - set(frame))}")
    if (len(frame) != len(test) * len(SEEDS) * len(treatments) or
        frame.duplicated(["patient_key", "seed", "treatment"]).any() or
        set(frame.patient_key.astype(str)) != set(test) or
        set(frame.seed.astype(int)) != set(SEEDS) or
        set(frame.treatment.astype(str)) != treatments or
        not frame.group_id.astype(str).eq(group_id).all() or
        not frame.label.astype(int).eq(frame.patient_key.map(labels)).all() or
        not np.isfinite(frame.prediction.astype(float)).all() or
        not frame.prediction.astype(float).between(0, 1).all()):
        raise ValueError(f"Incomplete or invalid three-seed ablation predictions for {group_id}")
    counts = frame.groupby(["patient_key", "seed"]).treatment.nunique()
    if not counts.eq(2).all() or len(counts) != len(test) * len(SEEDS):
        raise ValueError(f"Missing paired ablation treatment for {group_id}")


def run_ablation(root: Path, outer: int, device: str, max_gpu_gib: float,
                 resume: bool, cpu_threads: int = 8) -> dict:
    execution = execution_config(device, cpu_threads, max_gpu_gib)
    require_primary_bank_receipt(root)
    gate = ablation_gate(root)
    atomic_json(root / "mil/ablation/gate.json", gate)
    if gate["status"] != "pass":
        return gate
    rules, rules_sha = load_decision_rules(root)
    manifest = read_manifest(root)
    folds = read_folds(root, manifest)
    train, test = split_keys(folds, outer)
    labels = read_home_labels(root, manifest)
    nominations_path = root / "groups/fold_nominations.tsv"
    if not nominations_path.exists():
        raise FileNotFoundError(nominations_path)
    nominees = pd.read_csv(nominations_path, sep="\t", dtype={"group_id": str})
    required = {"group_id", "outer_fold", "rank", "scope", "train_subset", "train_patient_sha256"}
    if not required.issubset(nominees):
        raise ValueError(f"Missing nominations columns: {sorted(required - set(nominees))}")
    nominees = nominees.loc[nominees.outer_fold.eq(outer)].copy()
    if len(nominees) > 5 or nominees.group_id.duplicated().any():
        raise ValueError("At most five unique train-only groups per outer fold")
    expected_hash = train_patient_hash(train)
    if (not nominees.train_patient_sha256.eq(expected_hash).all() or
        not nominees.scope.eq(f"outer_{outer}").all() or
        not nominees.train_subset.eq("all_recorded_home_exploratory").all() or
        not nominees["rank"].isin(range(1, 6)).all()):
        raise ValueError("Nominations training-patient or scope provenance mismatch")
    validate_group_fold(root, outer, train)
    base = root / "groups" / f"outer_{outer}"
    for receipt_name in ("description.json", "burden_receipt.json"):
        rp = base / receipt_name
        if not rp.exists() or json.loads(rp.read_text()).get("train_patient_sha256") != expected_hash:
            raise ValueError(f"Fold group dictionary provenance mismatch: {rp}")
    safe_path = base / "patient_group_aa_safe.parquet"
    burden_path = base / "patient_group_burden.parquet"
    safe_sha, burden_sha = sha256(safe_path), sha256(burden_path)
    safe = pd.read_parquet(safe_path, columns=["patient_key", "group_id", "seq_idx", "projection_exact"])
    burden = pd.read_parquet(burden_path, columns=["patient_key", "group_id", "group_unique_aa", "presence"])
    safe["patient_key"] = safe.patient_key.astype(str)
    safe["group_id"] = safe.group_id.astype(str)
    burden["patient_key"] = burden.patient_key.astype(str)
    burden["group_id"] = burden.group_id.astype(str)
    if not safe.projection_exact.eq(True).all():
        raise ValueError("AA-only deletion table contains ambiguous V/J projections")
    if safe.duplicated(["patient_key", "group_id", "seq_idx"]).any():
        raise ValueError("Duplicate patient-group-AA safe row")
    selection_path = root / "mil/cv" / f"outer_{outer}" / "selection.json"
    selection = json.loads(selection_path.read_text())
    model_name, bag = selection["selected_model"], int(selection["selected_bag"])
    index = PatientIndex(root)
    raw = load_bank(root, model_name)
    spec = pca_spec(root, model_name, outer, None, raw.shape[1])
    bank = project_bank(root, model_name, outer, None, raw, spec, device)
    full_seed = []
    for seed in SEEDS:
        pred_path, receipt_path = model_paths(root, outer, model_name, bag, seed, None)
        if json.loads(receipt_path.read_text()).get("status") != "complete":
            raise ValueError(f"Full-seed fit is incomplete: {receipt_path}")
        frame = valid_saved_prediction(pred_path, test, labels, model_name, bag, seed, outer, None)
        if frame is None:
            raise ValueError(f"Full-seed held-out prediction invalid: {pred_path}")
        full_seed.append(frame)
    full = pd.concat(full_seed).set_index(["seed", "patient_key"])
    results = []
    paired_rows = []
    for nominee in nominees.itertuples():
        group_id = nominee.group_id
        group_safe = safe.loc[safe.group_id.eq(group_id) & safe.patient_key.isin(train + test)]
        target = {str(patient): set(g.seq_idx.astype(int))
                  for patient, g in group_safe.groupby("patient_key")}
        if not target:
            results.append({"group_id": group_id, "status": "skipped", "reason": "No safe AA removals"})
            continue
        total = burden.loc[burden.group_id.eq(group_id) & burden.patient_key.isin(train + test)]
        total_unique = int(total.group_unique_aa.sum())
        safe_unique = sum(len(v) for v in target.values())
        total_test_unique = int(total.loc[total.patient_key.isin(test), "group_unique_aa"].sum())
        safe_test_unique = sum(len(target.get(key, set())) for key in test)
        for key, seq_ids in target.items():
            patient_seq, _ = index.get(key)
            if not set(seq_ids).issubset(set(patient_seq.tolist())):
                raise ValueError(f"Safe deletion AA absent from patient index: {key}/{group_id}")
        if safe_unique > total_unique or safe_test_unique > total_test_unique:
            raise ValueError("Safe AA count exceeds full group AA burden")
        no_control = [key for key, seq_ids in target.items()
                      if 2 * len(seq_ids) > int(index.stats.loc[key, "unique_aa"])]
        if no_control:
            results.append({"group_id": group_id, "outer_fold": outer, "status": "skipped",
                            "reason": "Insufficient non-group AA for size-matched random deletion",
                            "affected_patients": len(no_control),
                            "safe_unique_aa": safe_unique,
                            "full_group_unique_aa": total_unique})
            continue
        group_dir = root / "mil/ablation" / f"outer_{outer}" / hashlib.sha256(group_id.encode()).hexdigest()[:16]
        group_dir.mkdir(parents=True, exist_ok=True)
        pred_rows = []
        control_matching_by_seed = {}
        for seed in SEEDS:
            receipt = json.loads(model_paths(root, outer, model_name, bag, seed, None)[1].read_text())
            epochs = int(receipt["epochs_ran"])
            random_exclude, matching = matched_random_exclude(
                index, target, train + test, stable_seed(group_id, outer, seed),
                return_diagnostics=True)
            matched_count = sum(x["target_aa"] for x in matching.values())
            control_matching_by_seed[str(seed)] = {
                "target_aa": matched_count,
                "exact_log10_bin_match_fraction": sum(x["exact_bin_matches"] for x in matching.values()) / matched_count,
                "mean_absolute_log10_bin_gap": sum(x["total_absolute_log10_bin_gap"] for x in matching.values()) / matched_count}
            for treatment, excludes in (("group_removed", target), ("matched_random", random_exclude)):
                pred_path = group_dir / f"seed_{seed}_{treatment}.tsv"
                ab_receipt_path = group_dir / f"seed_{seed}_{treatment}.json"
                removal_hash = hashlib.sha256(json.dumps(
                    {k: sorted(v) for k, v in sorted(excludes.items())}, sort_keys=True).encode()).hexdigest()
                fingerprint = hashlib.sha256(json.dumps({
                    "code_sha256": sha256(Path(__file__)),
                    "execution_config": execution,
                    "full_model_fingerprint": receipt["fingerprint"], "group_id": group_id,
                    "group_safe_sha256": safe_sha, "burden_sha256": burden_sha,
                    "treatment": treatment, "removal_hash": removal_hash, "seed": seed,
                    "test_patients": sorted(test)}, sort_keys=True).encode()).hexdigest()
                if resume and pred_path.exists() and ab_receipt_path.exists():
                    saved_receipt = json.loads(ab_receipt_path.read_text())
                    saved = pd.read_csv(pred_path, sep="\t", dtype={"patient_key": str})
                    if (saved_receipt.get("status") == "complete" and
                        saved_receipt.get("fingerprint") == fingerprint and
                        saved_receipt.get("prediction_sha256") == sha256(pred_path) and
                        len(saved) == len(test) and set(saved.patient_key) == set(test) and
                        not saved.patient_key.duplicated().any() and
                        saved.label.astype(int).eq(saved.patient_key.map(labels)).all() and
                        saved.seed.eq(seed).all() and saved.treatment.eq(treatment).all() and
                        saved.group_id.eq(group_id).all() and
                        np.isfinite(saved.prediction).all() and
                        saved.prediction.between(0, 1).all()):
                        pred_rows.append(saved)
                        continue
                trained, ab_receipt = train_mil(
                    train, labels, index, bank, bag, seed, device,
                    scope=f"outer{outer}-ablate-{group_id}-seed{seed}-{treatment}",
                    shuffle_scope=f"outer{outer}-final-seed{seed}",
                    fixed_epochs=epochs, exclude=excludes, max_gpu_gib=max_gpu_gib)
                infer_start = time.monotonic()
                pred, sd, mc_draws, unresolved = predict_adaptive(
                    trained, test, index, bank, bag, device,
                    scope=f"outer{outer}-test", exclude=excludes)
                inference_seconds = time.monotonic() - infer_start
                frame = pd.DataFrame({"patient_key": test, "label": [labels[k] for k in test],
                                      "prediction": pred, "mc_sd": sd, "seed": seed,
                                      "treatment": treatment, "group_id": group_id})
                frame.to_csv(pred_path, sep="\t", index=False)
                atomic_json(ab_receipt_path,
                            {**ab_receipt, "status": "complete", "fingerprint": fingerprint,
                             "code_sha256": sha256(Path(__file__)),
                             "execution_config": execution,
                             "prediction_sha256": sha256(pred_path),
                             "removed_patient_aa": sum(map(len, excludes.values())),
                             "mc_draws": mc_draws, "mc_unresolved_sd_gt_0_05": unresolved,
                             "inference_seconds": round(inference_seconds, 3)})
                pred_rows.append(frame)
                del trained
                if device == "cuda":
                    torch.cuda.empty_cache()
        predictions = pd.concat(pred_rows, ignore_index=True)
        validate_ablation_predictions(predictions, test, labels, group_id)
        predictions.to_csv(group_dir / "all_predictions.tsv", sep="\t", index=False)
        seed_wide = predictions.pivot(index=["seed", "patient_key", "label"],
                                      columns="treatment", values="prediction")
        group_paired = []
        for (seed, patient, label), row in seed_wide.iterrows():
            full_prediction = float(full.loc[(seed, patient), "prediction"])
            removed = float(row["group_removed"])
            random_removed = float(row["matched_random"])
            losses = patient_logloss(np.array([label, label, label]),
                                     np.array([full_prediction, removed, random_removed]))
            group_paired.append({
                "outer_fold": outer, "group_id": group_id, "patient_key": patient,
                "label": int(label), "seed": int(seed), "treatment": "group_removed_vs_full",
                "model": model_name, "bag_size": bag, "full_prediction": full_prediction,
                "deleted_prediction": removed, "random_deleted_prediction": random_removed,
                "full_logloss": float(losses[0]), "deleted_logloss": float(losses[1]),
                "random_deleted_logloss": float(losses[2]),
                "delta_logloss_vs_full": float(losses[1] - losses[0]),
                "random_delta_logloss_vs_full": float(losses[2] - losses[0]),
                "delta_vs_random": float(losses[1] - losses[2]),
            })
        group_paired_frame = pd.DataFrame(group_paired)
        group_paired_path = group_dir / "paired_patient_seed_logloss.tsv"
        group_paired_frame.to_csv(group_paired_path, sep="\t", index=False)
        paired_rows.extend(group_paired)
        per_patient = predictions.pivot_table(index="patient_key", columns="treatment", values="prediction", aggfunc="mean")
        ys = np.array([labels[k] for k in per_patient.index], dtype=int)
        full_pred = full.reset_index().groupby("patient_key").prediction.mean().reindex(per_patient.index).to_numpy()
        delta_group = patient_logloss(ys, per_patient.group_removed) - patient_logloss(ys, full_pred)
        delta_random = patient_logloss(ys, per_patient.matched_random) - patient_logloss(ys, full_pred)
        group_effect = paired_effect(delta_group)
        vs_random = paired_effect(delta_group - delta_random)
        exact_drop, exact_receipt = group_burden_baseline(
            root, outer, train, per_patient.index.tolist(), labels, index, omit_group=group_id)
        original_group = pd.read_csv(root / "mil/cv" / f"outer_{outer}" / "outer_oof.tsv",
                                     sep="\t", dtype={"patient_key": str}).set_index("patient_key")
        group_baseline_full = original_group.loc[per_patient.index, "group_burden"].to_numpy()
        exact_effect = (paired_effect(patient_logloss(ys, exact_drop) -
                                      patient_logloss(ys, group_baseline_full))
                        if exact_drop is not None else None)
        per_seed_effect = {
            str(seed): float(group_paired_frame.loc[
                group_paired_frame.seed.eq(seed), "delta_logloss_vs_full"].mean())
            for seed in SEEDS}
        summary = {"group_id": group_id, "outer_fold": outer, "status": "complete",
                   "rank": int(nominee.rank), "model": model_name, "bag_size": bag,
                   "decision_rules_sha256": rules_sha,
                   "per_seed_mean_delta_logloss": per_seed_effect,
                   "random_control_matching_by_seed": control_matching_by_seed,
                   "test_patients": len(test),
                   "test_carriers_safe": sum(k in target for k in test),
                   "test_carriers_exact_vj": int(total.loc[
                       total.patient_key.isin(test) & total.presence.astype(bool), "patient_key"].nunique()),
                   "safe_unique_aa": safe_unique, "full_group_unique_aa": total_unique,
                   "safe_test_unique_aa": safe_test_unique,
                   "full_group_test_unique_aa": total_test_unique,
                   "safe_deletion_coverage_fraction": (
                       safe_test_unique / total_test_unique if total_test_unique else 0.0),
                   "safe_deletion_coverage_fraction_all_home": (
                       safe_unique / total_unique if total_unique else 0.0),
                   "ambiguous_or_unprojectable_aa": total_unique - safe_unique,
                   "ambiguous_or_unprojectable_test_aa": total_test_unique - safe_test_unique,
                   "oof_sha256": gate["oof_sha256"],
                   "fold_selection_sha256": sha256(selection_path),
                   "paired_patient_seed_logloss_sha256": sha256(group_paired_path),
                   "leave_group_out_delta_logloss": group_effect,
                   "matched_random_delta_logloss": paired_effect(delta_random),
                   "group_minus_random_delta_logloss": vs_random,
                   "exact_vj_group_burden_leave_group_out": exact_effect,
                   "exact_vj_group_burden_receipt": exact_receipt,
                   "deletion_interpretation": "Conservative AA-only diagnostic; exact clone V/J attribution is evaluated by group burden"}
        atomic_json(group_dir / "summary.json", summary)
        results.append(summary)
    paired_path = root / "mil/ablation" / f"outer_{outer}" / "paired_patient_seed_logloss.tsv"
    paired_path.parent.mkdir(parents=True, exist_ok=True)
    paired_columns = ["outer_fold", "group_id", "patient_key", "label", "seed", "treatment",
                      "model", "bag_size", "full_prediction", "deleted_prediction",
                      "random_deleted_prediction", "full_logloss", "deleted_logloss",
                      "random_deleted_logloss", "delta_logloss_vs_full",
                      "random_delta_logloss_vs_full", "delta_vs_random"]
    pd.DataFrame(paired_rows, columns=paired_columns).to_csv(paired_path, sep="\t", index=False)
    complete = [r for r in results if r["status"] == "complete"]
    if complete:
        q = bh([r["leave_group_out_delta_logloss"]["signflip_one_sided_p"] for r in complete])
        ab_rule = rules["single_group_ablation"]
        for r, value in zip(complete, q):
            r["within_fold_bh_q"] = value
            r["passes_prespecified_fold_ablation_rule"] = bool(
                r["leave_group_out_delta_logloss"]["mean"] > ab_rule["primary_mean_gt"] and
                r["group_minus_random_delta_logloss"]["mean"] > ab_rule["matched_random_mean_gt"] and
                sum(x > 0 for x in r["per_seed_mean_delta_logloss"].values()) >=
                    ab_rule["positive_seed_effects_min"])
            group_dir = root / "mil/ablation" / f"outer_{outer}" / hashlib.sha256(
                r["group_id"].encode()).hexdigest()[:16]
            atomic_json(group_dir / "summary.json", r)
    outcome = {"status": "complete", "outer_fold": outer, "gate": gate,
               "decision_rules_sha256": rules_sha,
               "oof_sha256": gate["oof_sha256"],
               "fold_selection_sha256": sha256(selection_path),
               "paired_patient_seed_logloss": str(paired_path),
               "paired_patient_seed_logloss_sha256": sha256(paired_path),
               "scope": "Fold-specific group IDs; no cross-fold candidate confirmation without alignment",
               "groups": results}
    atomic_json(root / "mil/ablation" / f"outer_{outer}" / "summary.json", outcome)
    return outcome



class PilotIndex:
    def __init__(self, base: PatientIndex, keys: list[str], pool: int = 8192):
        self.data = {}
        for key in keys:
            seq, reads = base.get(key)
            if len(seq) > pool:
                pick = np.random.default_rng(stable_seed("pilot-pool-v1", key)).choice(
                    len(seq), size=pool, replace=False)
                self.data[key] = (seq[pick], reads[pick])
            else:
                self.data[key] = (seq, reads)
    def get(self, key: str) -> tuple[np.ndarray, np.ndarray]:
        return self.data[key]


class PilotBank:
    def __init__(self, seq_idx: np.ndarray, vectors: np.ndarray, total: int):
        self.lookup = {int(idx): i for i, idx in enumerate(seq_idx)}
        self.vectors = vectors
        self.shape = (total, vectors.shape[1])
    def __getitem__(self, rows: np.ndarray) -> np.ndarray:
        indices = np.asarray(rows).reshape(-1)
        pos = [self.lookup[int(i)] for i in indices]
        return self.vectors[pos]


def run_pilot(root: Path, outer: int, inner: int, model_name: str, bag: int,
              device: str, max_gpu_gib: float) -> dict:
    """Three-epoch cost trial, not a validation result or complete-repertoire fit."""
    manifest = read_manifest(root)
    folds = read_folds(root, manifest)
    tr, val = split_keys(folds, outer, inner)
    labels = read_home_labels(root, manifest)
    base_index = PatientIndex(root)
    pilot_index = PilotIndex(base_index, tr + val, pool=8192)
    raw = load_bank(root, model_name)
    spec = pca_spec(root, model_name, outer, inner, raw.shape[1])
    selected = np.unique(np.concatenate([pilot_index.get(k)[0] for k in tr + val]))
    projection_start = time.monotonic()
    if spec.raw_fallback:
        vectors = np.asarray(raw[selected], dtype=np.float16)
    else:
        with np.load(spec.path) as z:
            mean = z["mean"].astype(np.float32)
            comp = z["components"][:spec.dim].astype(np.float32).T.copy()
        vectors = np.empty((len(selected), spec.dim), dtype=np.float16)
        if device == "cuda":
            m = torch.as_tensor(mean, dtype=torch.float32, device="cuda")
            c = torch.as_tensor(comp, dtype=torch.float32, device="cuda")
        for start in range(0, len(selected), 32768):
            stop = min(start + 32768, len(selected))
            x = np.asarray(raw[selected[start:stop]], dtype=np.float32)
            if device == "cuda":
                vectors[start:stop] = ((torch.from_numpy(x).to("cuda") - m) @ c).cpu().numpy().astype(np.float16)
            else:
                vectors[start:stop] = ((x - mean) @ comp).astype(np.float16)
    projected_seconds = time.monotonic() - projection_start
    bank = PilotBank(selected, vectors, len(raw))
    model, fit = train_mil(tr, labels, pilot_index, bank, bag, SEEDS[0], device,
                           scope=f"PILOT-outer{outer}-inner{inner}-{model_name}-{bag}",
                           val_keys=val, max_epochs=3, patience=3,
                           max_gpu_gib=max_gpu_gib)
    del model
    result = {"status": "complete_pilot_only", "outer": outer, "inner": inner,
              "model": model_name, "bag_size": bag, "device": device,
              "source_code_sha256": sha256(Path(__file__)),
              "decision_rules_sha256": load_decision_rules(root)[1],
              "restricted_pool_per_patient": 8192,
              "pilot_unique_global_aa": len(selected), "projection_dim": int(bank.shape[1]),
              "pca_raw_fallback": spec.raw_fallback,
              "pca_fallback_reason": spec.fallback_reason,
              "scope_pca_stability_sha256": spec.stability_sha256,
              "embedding_meta_sha256": sha256(root / "embeddings" / model_name / "meta.json"),
              "pca_sha256": sha256(root / "pca" / model_name / f"outer_{outer}" / f"inner_{inner}" / "pca.npz"),
              "pca_metadata_sha256": sha256(root / "pca" / model_name / f"outer_{outer}" / f"inner_{inner}" / "metadata.json"),
              "patient_index_receipt_sha256": sha256(root / "mil/patient_index/receipt.json"),
              "folds_sha256": sha256(root / "folds/folds.tsv"),
              "projection_rows_per_second": round(len(selected) / max(projected_seconds, 1e-9), 1),
              "projection_seconds": round(projected_seconds, 3),
              "fit": fit, "extrapolation_warning": "Patient pools restricted to 8192 for cost only; not a scientific fit or full-bank projection."}
    out = root / "mil/pilot" / model_name / f"bag_{bag}" / f"outer_{outer}_inner_{inner}.json"
    atomic_json(out, result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["prepare", "verify-banks", "preflight", "preflight-outer", "pilot", "run", "summarize", "ablate"])
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--outer", type=int, choices=range(5))
    parser.add_argument("--inner", type=int, choices=range(3))
    parser.add_argument("--model", choices=MODELS)
    parser.add_argument("--bag", type=int, choices=BAGS)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--cpu-threads", type=int, default=8)
    parser.add_argument("--max-epochs", type=int, default=40)
    parser.add_argument("--patience", type=int, default=6)
    parser.add_argument("--max-gpu-gib", type=float, default=12.0)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.cpu_threads < 1 or args.cpu_threads > 16:
        raise ValueError("CPU threads must be 1..16")
    if not math.isfinite(args.max_gpu_gib) or not 0 < args.max_gpu_gib <= LOCKED_GPU_GIB:
        raise ValueError("GPU memory cap must be >0 and <=12 GiB")
    if args.command in ("run", "ablate") and (
            args.max_epochs != LOCKED_MAX_EPOCHS or args.patience != LOCKED_PATIENCE):
        raise ValueError("Formal MIL uses locked 40 epochs and patience 6")
    set_threads(args.cpu_threads)
    root = args.root.resolve()
    if args.command == "prepare":
        result = prepare_patient_index(root)
    elif args.command == "verify-banks":
        result = verify_primary_bank_bytes(root)
    elif args.command == "preflight":
        result = preflight(root)
    elif args.command == "preflight-outer":
        if args.outer is None:
            raise ValueError("--outer required for preflight-outer")
        result = preflight_outer(root, args.outer)
    elif args.command == "pilot":
        if any(value is None for value in (args.outer, args.inner, args.model, args.bag)):
            raise ValueError("pilot requires --outer --inner --model --bag")
        with gpu_slot(root, args.device, args.max_gpu_gib):
            result = run_pilot(root, args.outer, args.inner, args.model, args.bag,
                               args.device, args.max_gpu_gib)
    elif args.command == "run":
        if args.outer is None:
            raise ValueError("--outer required for run")
        with gpu_slot(root, args.device, args.max_gpu_gib):
            result = run_outer(root, args.outer, args.device, args.max_epochs,
                               args.patience, args.max_gpu_gib, args.resume, args.cpu_threads)
    elif args.command == "summarize":
        result = summarize_all(root)
    else:
        if args.outer is None:
            raise ValueError("--outer required for ablate")
        with gpu_slot(root, args.device, args.max_gpu_gib):
            result = run_ablation(root, args.outer, args.device, args.max_gpu_gib,
                                  args.resume, args.cpu_threads)
    print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    main()
