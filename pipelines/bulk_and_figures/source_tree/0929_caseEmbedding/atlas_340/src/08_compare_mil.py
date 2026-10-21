#!/usr/bin/env python3
"""Matched Home CDR3-only representation screen with fold-local projections.

This is a sampled representation screen. It does not alter the frozen atlas,
group dictionary, association tests, or the separate full-repertoire baseline.
"""
from __future__ import annotations
from release_runtime import private_literal as _private_literal, release_path as _release_path

import argparse
from contextlib import contextmanager
import fcntl
import hashlib
import json
import os
from pathlib import Path
import resource
import time

for _name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
              "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_name, "8")
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import pandas as pd
import psutil
from scipy import sparse
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.metrics import log_loss
from sklearn.model_selection import StratifiedShuffleSplit
from threadpoolctl import threadpool_limits
import torch
from torch import nn

ROOT = Path(_release_path('path_99d8c1c6fde2a6b9'))
OUT = ROOT / "representation_comparison/mil"
SAMPLE = ROOT / "representation_comparison/home_sample"
FOLDS = ROOT / "folds/folds.tsv"
FOLDS_RECEIPT = ROOT / "folds/folds.json"
DRAW = SAMPLE / "patient_aa_draw.parquet"
UNIQUE = SAMPLE / "unique_aa.parquet"
SAMPLE_RECEIPT = SAMPLE / "sample_receipt.json"
PARENT_GPU_LOCK = ROOT.parent / ".gpu.lock"
MODELS = (
    "tcrbert_dualtask_aa_mean", "sceptr_cdr3_only", "cdr3vec_small",
    "kmer3_cosine_svd",
)
DIMS = {"tcrbert_dualtask_aa_mean": 768, "sceptr_cdr3_only": 64,
        "cdr3vec_small": 128}
SEEDS = (17, 29, 43)
BAG = 2048
POOL = 4096
DIM = 64
MAX_EPOCHS = 40
PATIENCE = 6
GPU_LIMIT_GIB = 14.0
RSS_LIMIT_GIB = 512.0
N_MC = 8
N_MC_ESCALATED = 32
MC_SD_THRESHOLD = 0.05
SOURCE_CODE = Path(__file__)


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def hash_object(obj: object) -> str:
    raw = json.dumps(obj, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(raw.encode()).hexdigest()


def id_hash(keys: list[str]) -> str:
    return hashlib.sha256(("\n".join(sorted(keys)) + "\n").encode()).hexdigest()


def stable_seed(*parts: object) -> int:
    raw = "|".join(map(str, parts)).encode()
    return int.from_bytes(hashlib.sha256(raw).digest()[:8], "little") % (2**32 - 1)


def atomic_json(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".partial")
    tmp.write_text(json.dumps(obj, indent=2, sort_keys=True, allow_nan=False) + "\n")
    os.replace(tmp, path)


def atomic_tsv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".partial")
    frame.to_csv(tmp, sep="\t", index=False, float_format="%.12g")
    os.replace(tmp, path)


def rss_gib() -> float:
    return psutil.Process().memory_info().rss / 1024**3


def check_rss() -> None:
    if rss_gib() > RSS_LIMIT_GIB:
        raise RuntimeError("RSS exceeded 512 GiB")


@contextmanager
def file_lock(path: Path, message: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(path, os.O_CREAT | os.O_RDWR, 0o664)
    try:
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError(message) from exc
        yield
    finally:
        fcntl.flock(fd, fcntl.LOCK_UN)
        os.close(fd)


def setup_threads(n: int) -> None:
    if not 1 <= n <= 48:
        raise ValueError("CPU threads must be between 1 and 48")
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                 "NUMEXPR_NUM_THREADS"):
        os.environ[name] = str(n)
    threadpool_limits(limits=n)
    torch.set_num_threads(n)
    torch.set_num_interop_threads(min(n, 4))
    # The process itself is bounded. The main atlas serial runner retains its
    # separate, earlier 16-thread/128-GiB policy.
    requested = int(RSS_LIMIT_GIB * 1024**3)
    old_soft, old_hard = resource.getrlimit(resource.RLIMIT_AS)
    new_soft = requested if old_soft < 0 else min(old_soft, requested)
    if old_hard >= 0:
        new_soft = min(new_soft, old_hard)
    resource.setrlimit(resource.RLIMIT_AS, (new_soft, old_hard))


class SharedData:
    def __init__(self) -> None:
        receipt = json.loads(SAMPLE_RECEIPT.read_text())
        folds_receipt = json.loads(FOLDS_RECEIPT.read_text())
        if receipt.get("status") != "complete" or receipt.get("n_aa_per_patient") != POOL:
            raise ValueError("Shared label-blind Home draw is not complete")
        if receipt.get("draw_sha256") != sha(DRAW) or receipt.get("unique_sha256") != sha(UNIQUE):
            raise ValueError("Shared draw hashes differ from receipt")
        if folds_receipt.get("status") != "complete" or folds_receipt.get("folds_sha256") != sha(FOLDS):
            raise ValueError("Frozen patient folds changed")
        if receipt.get("manifest_sha256") != folds_receipt.get("manifest_sha256"):
            raise ValueError("Draw and frozen folds use different manifests")
        self.source_hashes = {
            "sample_receipt": sha(SAMPLE_RECEIPT), "draw": sha(DRAW),
            "unique": sha(UNIQUE), "folds": sha(FOLDS),
            "folds_receipt": sha(FOLDS_RECEIPT), "script": sha(SOURCE_CODE),
        }
        self.folds = pd.read_csv(FOLDS, sep="\t", dtype={"patient_key": str})
        expected = {"patient_key", "recorded_n_binary", "outer_fold", "partition", "inner_fold"}
        if len(self.folds) != 630 or not expected <= set(self.folds):
            raise ValueError("Frozen fold layout differs")
        labels = self.folds.drop_duplicates("patient_key").set_index("patient_key").recorded_n_binary
        self.labels = {str(k): int(v) for k, v in labels.items()}
        if len(self.labels) != 126 or list(np.bincount(list(self.labels.values()))) != [70, 56]:
            raise ValueError("Home label count differs")
        unique = pd.read_parquet(UNIQUE, columns=["aa", "seq_idx"])
        if len(unique) != 534376 or unique.aa.duplicated().any() or unique.seq_idx.duplicated().any():
            raise ValueError("Sampled global AA index differs")
        self.aa = unique.aa.astype(str).tolist()
        self.seq_idx = unique.seq_idx.to_numpy(dtype=np.int64)
        if not np.all(np.diff(self.seq_idx) > 0):
            raise ValueError("Selected global AA index must be increasing")
        draw = pd.read_parquet(DRAW, columns=["patient_key", "aa", "seq_idx", "read_count"])
        if len(draw) != 137 * POOL or draw[["patient_key", "aa"]].duplicated().any():
            raise ValueError("Shared patient draw changed")
        positions = np.searchsorted(self.seq_idx, draw.seq_idx.to_numpy(dtype=np.int64))
        if (positions >= len(self.seq_idx)).any() or not np.array_equal(self.seq_idx[positions], draw.seq_idx):
            raise ValueError("Patient AA absent from aligned vector index")
        if not np.array_equal(np.asarray(self.aa, dtype=object)[positions], draw.aa.to_numpy(dtype=object)):
            raise ValueError("Patient AA mapping disagrees")
        self.patients: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        for key, sub in draw.assign(bank_row=positions).groupby("patient_key", sort=False):
            ix = sub.bank_row.to_numpy(dtype=np.int64)
            reads = sub.read_count.to_numpy(dtype=np.int64)
            if len(ix) != POOL or len(np.unique(ix)) != POOL or np.any(reads <= 0):
                raise ValueError("Patient bag pool invalid: " + str(key))
            self.patients[str(key)] = ix, reads
        if len(self.patients) != 137 or not set(self.labels) <= set(self.patients):
            raise ValueError("Patient linkage invalid")
        for outer in range(5):
            block = self.folds.loc[self.folds.outer_fold.eq(outer)]
            if len(block) != 126 or block.patient_key.nunique() != 126:
                raise ValueError("Outer fold omits patient")
            train = set(block.loc[block.partition.eq("train"), "patient_key"])
            test = set(block.loc[block.partition.eq("test"), "patient_key"])
            if train & test or train | test != set(self.labels) or not 24 <= len(test) <= 27:
                raise ValueError("Outer split invalid")
            if set(block.loc[block.partition.eq("train"), "inner_fold"].astype(int)) != {0, 1, 2}:
                raise ValueError("Inner split invalid")
        self.kmer = None

    def split(self, outer: int, inner: int | None) -> tuple[list[str], list[str]]:
        block = self.folds.loc[self.folds.outer_fold.eq(outer)]
        train = block.loc[block.partition.eq("train")]
        if inner is None:
            a = train.patient_key.tolist()
            b = block.loc[block.partition.eq("test"), "patient_key"].tolist()
        else:
            a = train.loc[train.inner_fold.ne(inner), "patient_key"].tolist()
            b = train.loc[train.inner_fold.eq(inner), "patient_key"].tolist()
        if not a or not b or set(a) & set(b):
            raise ValueError("Train/evaluation patient overlap")
        return sorted(a), sorted(b)

    def train_rows(self, keys: list[str]) -> np.ndarray:
        return np.concatenate([self.patients[k][0] for k in sorted(keys)])

    def load_bank(self, model: str) -> tuple[np.ndarray | None, str]:
        if model == "kmer3_cosine_svd":
            if self.kmer is None:
                vectorizer = HashingVectorizer(
                    analyzer="char", ngram_range=(3, 3), n_features=4096,
                    alternate_sign=False, norm="l2", lowercase=False,
                    dtype=np.float32,
                )
                self.kmer = vectorizer.transform(self.aa).tocsr()
                if self.kmer.shape != (len(self.aa), 4096) or self.kmer.nnz == 0:
                    raise ValueError("K-mer matrix invalid")
            return None, hash_object({
                "unique_sha256": self.source_hashes["unique"],
                "features": 4096, "ngram_range": [3, 3],
                "norm": "l2", "alternate_sign": False, "lowercase": False,
                "dtype": "float32", "method": "hashed_char3_cosine",
            })
        path = ROOT / "representation_comparison/vectors" / model / "bank_fp16.npy"
        if not path.exists():
            raise FileNotFoundError("Aligned vector bank missing: " + str(path))
        bank = np.load(path, mmap_mode="r", allow_pickle=False)
        if bank.shape != (len(self.aa), DIMS[model]) or bank.dtype != np.float16:
            raise ValueError("Aligned bank shape/dtype differs: " + str(path))
        for start in range(0, len(bank), 100000):
            if not np.isfinite(bank[start:start + 100000]).all():
                raise ValueError("Nonfinite aligned bank vector")
        return bank, sha(path)


class GatedMIL(nn.Module):
    """Same 64/32 gated attention network as 0929_caseEmbedding/src/05_mil.py."""
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(nn.LayerNorm(DIM + 1), nn.Linear(DIM + 1, 64),
                                     nn.ReLU(), nn.Dropout(0.2))
        self.attn_tanh = nn.Linear(64, 32)
        self.attn_gate = nn.Linear(64, 32)
        self.attn_out = nn.Linear(32, 1)
        self.head = nn.Sequential(nn.LayerNorm(64), nn.Linear(64, 1))

    def forward(self, x: torch.Tensor, count_log: torch.Tensor,
                mask: torch.Tensor) -> torch.Tensor:
        h = self.encoder(torch.cat([x, count_log.unsqueeze(-1)], dim=-1))
        score = self.attn_out(torch.tanh(self.attn_tanh(h)) *
                              torch.sigmoid(self.attn_gate(h))).squeeze(-1)
        score = score.masked_fill(~mask, -1e9)
        weight = torch.softmax(score, dim=1)
        return self.head((weight.unsqueeze(-1) * h).sum(dim=1)).squeeze(-1)


def projected_dir(model: str, outer: int, inner: int | None) -> Path:
    return OUT / "projections" / model / f"outer_{outer}" / (
        "final" if inner is None else f"inner_{inner}")


def projected_bank(data: SharedData, model: str, outer: int, inner: int | None,
                   raw: np.ndarray | None, raw_hash: str) -> tuple[np.ndarray, str]:
    train, _ = data.split(outer, inner)
    directory = projected_dir(model, outer, inner)
    bank_path = directory / "projected_fp16.npy"
    meta_path = directory / "metadata.json"
    method = "train_only_hashed_char3_l2_4096_truncated_svd64" if model == (
        "kmer3_cosine_svd") else "train_only_patient_balanced_pca64"
    contract = {
        "model": model, "outer_fold": outer, "inner_fold": inner,
        "train_patient_hash": id_hash(train), "n_train_patients": len(train),
        "n_rows_per_train_patient": POOL, "n_fit_rows": len(train) * POOL,
        "raw_bank_sha256": raw_hash, "folds_sha256": data.source_hashes["folds"],
        "draw_sha256": data.source_hashes["draw"],
        "unique_sha256": data.source_hashes["unique"],
        "script_sha256": data.source_hashes["script"], "method": method,
        "dim": DIM,
    }
    if meta_path.exists():
        meta = json.loads(meta_path.read_text())
        if (meta.get("status") != "complete" or
            {k: meta.get(k) for k in contract} != contract or
            not bank_path.is_file() or meta.get("projection_sha256") != sha(bank_path)):
            raise RuntimeError("Existing fold projection receipt invalid: " + str(directory))
        output = np.load(bank_path, mmap_mode="r", allow_pickle=False)
        if output.shape != (len(data.aa), DIM) or output.dtype != np.float16:
            raise RuntimeError("Projected bank dimensions invalid")
        return output, meta["projection_sha256"]
    directory.mkdir(parents=True, exist_ok=True)
    start_time = time.monotonic()
    fit_rows = data.train_rows(train)
    if len(fit_rows) != len(train) * POOL:
        raise ValueError("Patient-balanced PCA/SVD rows differ")
    rng = stable_seed("projection-v1", outer, inner if inner is not None else "final")
    if model == "kmer3_cosine_svd":
        assert data.kmer is not None
        fit_x = data.kmer[fit_rows]
        reducer = TruncatedSVD(n_components=DIM, n_iter=3, random_state=rng)
        reducer.fit(fit_x)
        components = reducer.components_.astype(np.float32)
        mean = np.zeros(4096, dtype=np.float32)
        variance = reducer.explained_variance_ratio_.astype(np.float32)
    else:
        assert raw is not None
        fit_x = np.asarray(raw[fit_rows], dtype=np.float32)
        reducer = PCA(n_components=DIM, svd_solver="randomized", iterated_power=3,
                      random_state=rng)
        reducer.fit(fit_x)
        components = reducer.components_.astype(np.float32)
        mean = reducer.mean_.astype(np.float32)
        variance = reducer.explained_variance_ratio_.astype(np.float32)
    del fit_x
    check_rss()
    basis_path = directory / "basis.npz"
    basis_tmp = directory / "basis.partial.npz"
    np.savez_compressed(basis_tmp, components=components, mean=mean,
                        explained_variance_ratio=variance)
    os.replace(basis_tmp, basis_path)
    tmp = directory / "projected.partial.npy"
    output = np.lib.format.open_memmap(tmp, mode="w+", dtype=np.float16,
                                        shape=(len(data.aa), DIM))
    for start in range(0, len(data.aa), 50000):
        stop = min(start + 50000, len(data.aa))
        if model == "kmer3_cosine_svd":
            values = np.asarray(data.kmer[start:stop] @ components.T, dtype=np.float32)
        else:
            values = (np.asarray(raw[start:stop], dtype=np.float32) - mean) @ components.T
        if values.shape != (stop - start, DIM) or not np.isfinite(values).all():
            raise ValueError("Fold projection produced nonfinite rows")
        output[start:stop] = values.astype(np.float16)
        if not np.isfinite(output[start:stop]).all():
            raise ValueError("FP16 fold projection produced nonfinite rows")
        check_rss()
    output.flush()
    del output
    os.replace(tmp, bank_path)
    bank_hash = sha(bank_path)
    meta = {**contract, "status": "complete", "basis_sha256": sha(basis_path),
            "projection_sha256": bank_hash,
            "total_aa": len(data.aa), "seconds": round(time.monotonic()-start_time, 3),
            "peak_rss_gib": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss /
                                  (1024**2), 4)}
    atomic_json(meta_path, meta)
    return np.load(bank_path, mmap_mode="r", allow_pickle=False), bank_hash


def bag_positions(n: int, patient: str, epoch: int, draw: int, scope: str) -> np.ndarray:
    if n <= BAG:
        return np.arange(n, dtype=np.int64)
    rng = np.random.default_rng(stable_seed("draw-v1", patient, epoch, draw, scope))
    return rng.choice(n, size=n, replace=False)[:BAG]


def make_bag(keys: list[str], data: SharedData, bank: np.ndarray, epoch: int,
             draw: int, scope: str, device: str) -> tuple[torch.Tensor, ...]:
    x = np.zeros((len(keys), BAG, DIM), dtype=np.float32)
    counts = np.zeros((len(keys), BAG), dtype=np.float32)
    mask = np.zeros((len(keys), BAG), dtype=np.bool_)
    for row, key in enumerate(keys):
        positions, reads = data.patients[key]
        pick = bag_positions(len(positions), key, epoch, draw, scope)
        n = len(pick)
        x[row, :n] = np.asarray(bank[positions[pick]], dtype=np.float32)
        counts[row, :n] = np.log1p(reads[pick]).astype(np.float32)
        mask[row, :n] = True
    return (torch.from_numpy(x).to(device), torch.from_numpy(counts).to(device),
            torch.from_numpy(mask).to(device))


@torch.no_grad()
def predict_mc(model: GatedMIL, keys: list[str], data: SharedData, bank: np.ndarray,
               device: str, scope: str, n_draws: int) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    predictions = np.empty((len(keys), n_draws), dtype=np.float64)
    for draw in range(n_draws):
        for i, key in enumerate(keys):
            x, count, mask = make_bag([key], data, bank, -1, draw, scope, device)
            predictions[i, draw] = torch.sigmoid(model(x, count, mask)).item()
    return predictions.mean(1), predictions.std(1, ddof=1)


def predict_adaptive(model: GatedMIL, keys: list[str], data: SharedData,
                     bank: np.ndarray, device: str, scope: str) -> tuple[np.ndarray, np.ndarray, int]:
    pred, sd = predict_mc(model, keys, data, bank, device, scope, N_MC)
    n = N_MC
    if np.any(sd > MC_SD_THRESHOLD):
        pred, sd = predict_mc(model, keys, data, bank, device, scope, N_MC_ESCALATED)
        n = N_MC_ESCALATED
    return pred, sd, n


def early_split(train: list[str], labels: dict[str, int], outer: int,
                inner: int, seed: int) -> tuple[list[str], list[str]]:
    y = np.array([labels[k] for k in train], dtype=int)
    splitter = StratifiedShuffleSplit(
        n_splits=1, test_size=0.2,
        random_state=stable_seed("early-stop-v1", outer, inner, seed))
    fit_pos, stop_pos = next(splitter.split(np.asarray(train), y))
    fit = [train[i] for i in fit_pos]
    stop = [train[i] for i in stop_pos]
    if set(fit) & set(stop) or set(fit + stop) != set(train):
        raise ValueError("Early stopping split invalid")
    return fit, stop


def train_model(train_ids: list[str], stop_ids: list[str] | None, data: SharedData,
                bank: np.ndarray, seed: int, outer: int, inner: int | None,
                device: str, fixed_epochs: int | None) -> tuple[GatedMIL, dict]:
    torch.manual_seed(seed)
    np.random.seed(seed)
    if device == "cuda":
        torch.cuda.manual_seed_all(seed)
        torch.cuda.reset_peak_memory_stats()
    model = GatedMIL().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-3)
    loss_fn = nn.BCEWithLogitsLoss()
    best_loss, best_epoch, best_state, stale = float("inf"), 0, None, 0
    n_epochs = fixed_epochs if fixed_epochs is not None else MAX_EPOCHS
    start_time = time.monotonic()
    for epoch in range(n_epochs):
        check_rss()
        model.train()
        order = np.random.default_rng(stable_seed(
            "patient-shuffle-v1", seed, outer, inner if inner is not None else "final",
            epoch)).permutation(train_ids).tolist()
        for begin in range(0, len(order), 2):
            keys = order[begin:begin+2]
            x, counts, mask = make_bag(keys, data, bank, epoch, 0, "train", device)
            target = torch.tensor([data.labels[k] for k in keys], dtype=torch.float32,
                                  device=device)
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(model(x, counts, mask), target)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        if device == "cuda" and torch.cuda.max_memory_reserved() / 1024**3 > GPU_LIMIT_GIB:
            raise RuntimeError("MIL peak reserved GPU memory exceeded 14 GiB")
        if stop_ids is not None:
            p, _ = predict_mc(model, stop_ids, data, bank, device,
                              f"early-outer{outer}-inner{inner}-seed{seed}", 2)
            y = np.array([data.labels[k] for k in stop_ids], dtype=int)
            value = log_loss(y, np.clip(p, 1e-6, 1-1e-6), labels=[0, 1])
            if value < best_loss - 1e-4:
                best_loss, best_epoch, stale = value, epoch + 1, 0
                best_state = {k: v.detach().cpu().clone()
                              for k, v in model.state_dict().items()}
            else:
                stale += 1
                if stale >= PATIENCE:
                    break
    if best_state is not None:
        model.load_state_dict(best_state)
    receipt = {
        "epochs_ran": epoch + 1, "best_epoch": best_epoch or epoch + 1,
        "stop_log_loss": None if stop_ids is None else float(best_loss),
        "fit_epochs": n_epochs, "n_fit_patients": len(train_ids),
        "n_stop_patients": 0 if stop_ids is None else len(stop_ids),
        "seconds": round(time.monotonic()-start_time, 3),
        "peak_gpu_allocated_gib": round(torch.cuda.max_memory_allocated()/1024**3, 4)
        if device == "cuda" else 0.0,
        "peak_gpu_reserved_gib": round(torch.cuda.max_memory_reserved()/1024**3, 4)
        if device == "cuda" else 0.0,
        "peak_rss_gib": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss /
                              (1024**2), 4),
    }
    return model, receipt


def fit_dir(model: str, outer: int, inner: int | None, seed: int) -> Path:
    stage = "final" if inner is None else f"inner_{inner}"
    return OUT / model / f"outer_{outer}" / stage / f"seed_{seed}"


def saved_fit(directory: Path, contract: dict, expected: list[str],
              labels: dict[str, int]) -> dict | None:
    receipt_path = directory / "receipt.json"
    if not receipt_path.exists():
        return None
    receipt = json.loads(receipt_path.read_text())
    pred_path = directory / "prediction.tsv"
    model_path = directory / "model.pt"
    if (receipt.get("status") != "complete" or receipt.get("contract") != contract or
        not pred_path.exists() or not model_path.exists() or
        receipt.get("prediction_sha256") != sha(pred_path) or
        receipt.get("model_sha256") != sha(model_path)):
        raise RuntimeError("Existing fit receipt invalid: " + str(directory))
    p = pd.read_csv(pred_path, sep="\t", dtype={"patient_key": str})
    if (len(p) != len(expected) or set(p.patient_key) != set(expected) or
        p.patient_key.duplicated().any() or
        not np.isfinite(p.probability).all() or
        not p.probability.between(0, 1).all() or
        any(int(row.recorded_n_binary) != labels[row.patient_key]
            for row in p.itertuples())):
        raise RuntimeError("Existing fit predictions invalid: " + str(directory))
    return receipt


def one_fit(data: SharedData, model: str, outer: int, inner: int | None, seed: int,
            raw: np.ndarray | None, raw_hash: str, device: str) -> dict:
    train, eval_ids = data.split(outer, inner)
    directory = fit_dir(model, outer, inner, seed)
    bank, projection_hash = projected_bank(data, model, outer, inner, raw, raw_hash)
    prior = []
    if inner is None:
        for fold in range(3):
            prior_path = fit_dir(model, outer, fold, seed) / "receipt.json"
            if not prior_path.exists():
                raise RuntimeError("Final fit requires all three inner fits: " + str(prior_path))
            prior.append(json.loads(prior_path.read_text()))
        if any(x.get("status") != "complete" for x in prior):
            raise RuntimeError("Incomplete inner fit")
        for fold, prior_receipt in enumerate(prior):
            prior_contract = prior_receipt.get("contract", {})
            prior_prediction = fit_dir(model, outer, fold, seed) / "prediction.tsv"
            if (prior_contract.get("model") != model or
                prior_contract.get("outer_fold") != outer or
                prior_contract.get("inner_fold") != fold or
                prior_contract.get("seed") != seed or
                prior_contract.get("raw_bank_sha256") != raw_hash or
                prior_contract.get("script_sha256") != data.source_hashes["script"] or
                prior_contract.get("folds_sha256") != data.source_hashes["folds"] or
                prior_contract.get("draw_sha256") != data.source_hashes["draw"] or
                not prior_prediction.is_file() or
                prior_receipt.get("prediction_sha256") != sha(prior_prediction)):
                raise RuntimeError("Inner fit provenance invalid before final training")
        fixed_epochs = int(np.median([x["training"]["best_epoch"] for x in prior]))
        if not 1 <= fixed_epochs <= MAX_EPOCHS:
            raise ValueError("Inner-selected final epoch count invalid")
        inner_hashes = [sha(fit_dir(model, outer, fold, seed) / "receipt.json")
                        for fold in range(3)]
        fit_ids, stop_ids = train, None
        draw_scope = f"test-outer{outer}-seed{seed}"
    else:
        fixed_epochs = None
        inner_hashes = []
        fit_ids, stop_ids = early_split(train, data.labels, outer, inner, seed)
        draw_scope = f"inner-outer{outer}-inner{inner}-seed{seed}"
    contract = {
        "model": model, "outer_fold": outer, "inner_fold": inner, "seed": seed,
        "bag_size": BAG, "pool_per_patient": POOL, "projection_sha256": projection_hash,
        "raw_bank_sha256": raw_hash, "script_sha256": data.source_hashes["script"],
        "folds_sha256": data.source_hashes["folds"],
        "draw_sha256": data.source_hashes["draw"],
        "train_patient_hash": id_hash(train), "fit_patient_hash": id_hash(fit_ids),
        "stop_patient_hash": None if stop_ids is None else id_hash(stop_ids),
        "eval_patient_hash": id_hash(eval_ids), "draw_scope": draw_scope,
        "device": device, "fixed_epochs": fixed_epochs,
        "inner_receipt_sha256": inner_hashes,
        "optimizer": "AdamW lr=3e-4 weight_decay=1e-3",
        "network": "GatedMIL dim64 hidden64 attention32 dropout0.2",
        "max_epochs": MAX_EPOCHS, "patience": PATIENCE, "patient_batch": 2,
        "mc_initial": N_MC, "mc_escalation": N_MC_ESCALATED,
        "mc_sd_threshold": MC_SD_THRESHOLD,
    }
    previous = saved_fit(directory, contract, eval_ids, data.labels)
    if previous is not None:
        return previous
    directory.mkdir(parents=True, exist_ok=True)
    with file_lock(PARENT_GPU_LOCK, "Parent GPU slot is occupied"):
        if device == "cuda":
            if not torch.cuda.is_available():
                raise RuntimeError("CUDA requested but unavailable")
            total = torch.cuda.get_device_properties(0).total_memory
            torch.cuda.set_per_process_memory_fraction(
                min(1.0, GPU_LIMIT_GIB * 1024**3 / total), 0)
        model_fit, training = train_model(
            fit_ids, stop_ids, data, bank, seed, outer, inner, device, fixed_epochs)
        p, sd, draws = predict_adaptive(
            model_fit, eval_ids, data, bank, device, draw_scope)
        peak_reserved = (torch.cuda.max_memory_reserved() / 1024**3
                         if device == "cuda" else 0.0)
        if peak_reserved > GPU_LIMIT_GIB:
            raise RuntimeError("Inference exceeded 14 GiB GPU budget")
        model_tmp = directory / "model.partial.pt"
        torch.save(model_fit.state_dict(), model_tmp)
        os.replace(model_tmp, directory / "model.pt")
        del model_fit
        if device == "cuda":
            torch.cuda.empty_cache()
    frame = pd.DataFrame({
        "patient_key": eval_ids,
        "recorded_n_binary": [data.labels[k] for k in eval_ids],
        "outer_fold": outer, "inner_fold": inner,
        "model": model, "seed": seed, "bag_size": BAG,
        "probability": p, "prediction_sd": sd, "mc_bags": draws,
        "draw_scope": draw_scope, "best_epoch": training["best_epoch"],
        "fit_epochs": training["fit_epochs"],
    })
    if not np.isfinite(p).all() or not np.isfinite(sd).all():
        raise ValueError("Nonfinite heldout probabilities")
    pred_path = directory / "prediction.tsv"
    atomic_tsv(pred_path, frame)
    receipt = {
        "status": "complete", "contract": contract, "training": training,
        "n_predictions": len(frame), "prediction_sha256": sha(pred_path),
        "model_sha256": sha(directory / "model.pt"),
        "projection_metadata_sha256": sha(
            projected_dir(model, outer, inner) / "metadata.json"),
        "peak_gpu_reserved_including_inference_gib": round(peak_reserved, 4),
        "mc_bags": draws, "n_predictions_sd_gt_0_05": int(np.sum(sd > 0.05)),
    }
    atomic_json(directory / "receipt.json", receipt)
    print(json.dumps({
        "model": model, "outer": outer, "inner": inner, "seed": seed,
        "status": "complete", "n_eval": len(frame), "epochs": training["best_epoch"],
        "gpu_peak_gib": receipt["peak_gpu_reserved_including_inference_gib"],
        "seconds": training["seconds"],
    }), flush=True)
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=("all",) + MODELS, default="all")
    parser.add_argument("--outer", type=int, choices=range(5))
    parser.add_argument("--seed", type=int, choices=SEEDS)
    parser.add_argument("--inner", type=int, choices=range(3))
    parser.add_argument("--phase", choices=("project", "inner", "final", "all"),
                        default="all")
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--threads", type=int, default=8)
    args = parser.parse_args()
    if args.inner is not None and args.phase != "inner":
        parser.error("--inner requires --phase inner")
    setup_threads(args.threads)
    data = SharedData()
    models = MODELS if args.model == "all" else (args.model,)
    outers = range(5) if args.outer is None else (args.outer,)
    seeds = SEEDS if args.seed is None else (args.seed,)
    with file_lock(OUT / ".comparison.lock", "Another representation screen is running"):
        for model in models:
            raw, raw_hash = data.load_bank(model)
            print(json.dumps({"model": model, "raw_sha256": raw_hash,
                              "n_unique_aa": len(data.aa)}), flush=True)
            for outer in outers:
                if args.phase == "project":
                    scopes = (None, 0, 1, 2)
                    for inner in scopes:
                        projected_bank(data, model, outer, inner, raw, raw_hash)
                    continue
                if args.phase in ("inner", "all"):
                    inner_scopes = range(3) if args.inner is None else (args.inner,)
                    for inner in inner_scopes:
                        for seed in seeds:
                            one_fit(data, model, outer, inner, seed, raw, raw_hash,
                                    args.device)
                if args.phase in ("final", "all"):
                    for seed in seeds:
                        one_fit(data, model, outer, None, seed, raw, raw_hash,
                                args.device)


if __name__ == "__main__":
    main()
