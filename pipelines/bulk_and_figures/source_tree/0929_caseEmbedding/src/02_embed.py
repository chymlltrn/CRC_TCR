#!/usr/bin/env python3
"""Frozen, CDR3-beta-only embedding with atomic shards and a published mmap bank.

Input: data/aa_union.parquet with seq_idx (int64, 0..N-1) and aa (canonical AA).
Default invocation is a 50,000-sequence pilot. A full run requires --full.
Finished outputs: embeddings/<model>/bank_fp16.npy, indexed directly by seq_idx.
"""
from __future__ import annotations
from release_runtime import private_literal as _private_literal, release_path as _release_path

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import logging
import os
import re
import sys
import time
from importlib import metadata
from pathlib import Path

ROOT = Path(_release_path('path_1304218e86d2e596'))
DEFAULT_BERT = Path(_release_path('path_a521916029b565d4'))
OLD_META = Path(_release_path('path_2c546708477b822a'))
MODELS = ("tcrbert_mlm", "sceptr_cdr3_only")
CANONICAL = re.compile(r"^[ACDEFGHIKLMNPQRSTVWY]+$")


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def json_hash(value: dict) -> str:
    data = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(data).hexdigest()


def atomic_json(path: Path, value: dict) -> None:
    temp = path.with_name(path.name + f".partial.{os.getpid()}")
    try:
        with temp.open("w") as stream:
            json.dump(value, stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temp, path)
    finally:
        temp.unlink(missing_ok=True)


def version(name: str) -> str | None:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return None


def model_spec(name: str, bert_path: Path, precision: str) -> dict:
    import torch

    base = {
        "model": name,
        "input_fields": ["seq_idx", "aa"],
        "bank_dtype": "float16",
        "script_sha256": sha256(Path(__file__)),
        "torch": torch.__version__,
        "numpy": version("numpy"),
        "pandas": version("pandas"),
        "pyarrow": version("pyarrow"),
    }
    if name == "tcrbert_mlm":
        require(bert_path.is_dir(), f"TCR-BERT model missing: {bert_path}")
        files = ("config.json", "pytorch_model.bin", "vocab.txt",
                 "tokenizer_config.json", "special_tokens_map.json")
        require(all((bert_path / f).is_file() for f in files),
                f"TCR-BERT checkpoint incomplete: {bert_path}")
        cfg = json.loads((bert_path / "config.json").read_text())
        require(cfg.get("model_type") == "bert", "TCR-BERT model_type is not bert")
        require(cfg["max_position_embeddings"] >= 64,
                "TCR-BERT position limit below expected 64")
        base.update({
            "model_source": "wukevin/tcr-bert-mlm-only local mirror",
            "model_path": str(bert_path.resolve()),
            "model_files_sha256": {f: sha256(bert_path / f) for f in files},
            "hidden_dim": int(cfg["hidden_size"]),
            "max_model_tokens": int(cfg["max_position_embeddings"]),
            "tokenization": "uppercase CDR3 amino acids separated by spaces; BertTokenizerFast; special tokens",
            "pooling": "attention-mask mean of last hidden state, including CLS and SEP; no pooler_output",
            "precision": precision,
            "transformers": version("transformers"),
        })
    elif name == "sceptr_cdr3_only":
        import sceptr
        save = Path(sceptr.__file__).parent / "_model_saves" / "SCEPTR_CDR3_only"
        files = ("config.json", "state_dict.pt")
        require(all((save / f).is_file() for f in files),
                f"SCEPTR cdr3_only state missing: {save}")
        base.update({
            "model_source": "sceptr.variant.cdr3_only, official PyPI package",
            "model_path": str(save.resolve()),
            "model_files_sha256": {f: sha256(save / f) for f in files},
            "hidden_dim": 64,
            "tokenization": "sceptr 1.2.0 CDR3B-only DataFrame via package API",
            "pooling": "package-trained CLS sequence representation",
            "precision": "float32 inference",
            "sceptr": version("sceptr"),
            "libtcrlm": version("libtcrlm"),
            "tidytcells": version("tidytcells"),
        })
        require(base["sceptr"] == "1.2.0", "Unexpected SCEPTR version")
    else:
        raise ValueError(name)
    return base


def old_cache_assessment(spec: dict) -> dict:
    """The historical cache is diagnostic only unless all identity evidence exists."""
    required = {
        "weight_sha256", "tokenizer_sha256", "pooling_definition",
        "transformers_version", "torch_version", "input_sha256",
        "sample_vector_validation",
    }
    if not OLD_META.is_file():
        return {"reuse_allowed": False, "reason": "historical metadata absent"}
    old = json.loads(OLD_META.read_text())
    missing = sorted(required - old.keys())
    return {
        "reuse_allowed": False,
        "reason": "historical cache lacks exact provenance and row alignment; fresh recompute required",
        "missing_provenance_fields": missing,
        "historical_rows": old.get("n_total"),
        "historical_pooling_claim": old.get("token_pooling"),
        "current_weight_sha256": spec["model_files_sha256"].get("pytorch_model.bin"),
    }


def input_info(path: Path) -> tuple[int, str, tuple[int, int]]:
    import pyarrow as pa
    import pyarrow.parquet as pq

    require(path.is_file(), f"Missing union input: {path}")
    pf = pq.ParquetFile(path)
    names = pf.schema_arrow.names
    require("seq_idx" in names and "aa" in names,
            f"Union schema needs seq_idx and aa; got {names}")
    require(pf.schema_arrow.field("seq_idx").type == pa.int64(),
            "seq_idx must be int64")
    require(pf.schema_arrow.field("aa").type in (pa.string(), pa.large_string()),
            "aa must be a string column")
    n = pf.metadata.num_rows
    require(n > 0, "Union is empty")
    stat = path.stat()
    return n, sha256(path), (stat.st_size, stat.st_mtime_ns)


def iter_shards(path: Path, shard_rows: int, row_limit: int):
    import pyarrow.parquet as pq

    pf = pq.ParquetFile(path)
    pending_ids: list[int] = []
    pending_aa: list[str] = []
    expected = 0
    for batch in pf.iter_batches(batch_size=shard_rows, columns=["seq_idx", "aa"]):
        ids = batch.column("seq_idx").to_pylist()
        aas = batch.column("aa").to_pylist()
        take = min(len(ids), row_limit - expected)
        if take <= 0:
            break
        for idx, aa in zip(ids[:take], aas[:take]):
            require(idx == expected, f"Noncontiguous or missing seq_idx at {expected}: {idx}")
            require(isinstance(aa, str) and CANONICAL.fullmatch(aa) is not None,
                    f"Invalid CDR3 amino-acid sequence at seq_idx={expected}")
            require(len(aa) <= 62,
                    f"CDR3 length {len(aa)} at seq_idx={expected} exceeds TCR-BERT's 62-residue input")
            pending_ids.append(idx)
            pending_aa.append(aa)
            expected += 1
            if len(pending_aa) == shard_rows:
                start = pending_ids[0]
                yield start, expected, pending_aa
                pending_ids, pending_aa = [], []
        if expected >= row_limit:
            break
    require(expected == row_limit,
            f"Union ended at {expected}, expected {row_limit} rows")
    if pending_aa:
        yield pending_ids[0], expected, pending_aa


def shard_name(start: int, end: int) -> str:
    return f"{start:012d}_{end:012d}"


def verify_shard(outdir: Path, start: int, end: int, dimension: int,
                 input_sha: str, model_sha: str) -> bool:
    import numpy as np

    stem = shard_name(start, end)
    arr_path = outdir / "shards" / f"{stem}.npy"
    receipt_path = outdir / "shards" / f"{stem}.json"
    if not (arr_path.is_file() and receipt_path.is_file()):
        return False
    try:
        receipt = json.loads(receipt_path.read_text())
        expected = {"start": start, "end": end, "rows": end - start,
                    "dimension": dimension, "dtype": "float16",
                    "input_sha256": input_sha, "model_fingerprint": model_sha,
                    "nonfinite": 0}
        if any(receipt.get(k) != v for k, v in expected.items()):
            return False
        if not isinstance(receipt.get("nonstandard_junction_warnings"), int):
            return False
        if receipt["nonstandard_junction_warnings"] < 0:
            return False
        if sha256(arr_path) != receipt.get("shard_sha256"):
            return False
        arr = np.load(arr_path, mmap_mode="r", allow_pickle=False)
        return arr.shape == (end - start, dimension) and arr.dtype == np.float16
    except (OSError, ValueError, KeyError):
        return False


def write_shard(outdir: Path, start: int, end: int, vectors,
                input_sha: str, model_sha: str, peak_mib: float,
                nonstandard_warnings: int = 0) -> None:
    import numpy as np

    require(vectors.shape[0] == end - start,
            "Backend returned wrong number of vectors")
    require(np.isfinite(vectors).all(), "Backend returned NaN or Inf")
    fp16 = np.asarray(vectors, dtype=np.float16)
    require(np.isfinite(fp16).all(), "float16 conversion produced NaN or Inf")
    stem = shard_name(start, end)
    arr_path = outdir / "shards" / f"{stem}.npy"
    receipt_path = outdir / "shards" / f"{stem}.json"
    temp = arr_path.with_name(arr_path.name + f".partial.{os.getpid()}")
    try:
        with temp.open("wb") as stream:
            np.save(stream, fp16, allow_pickle=False)
            stream.flush()
            os.fsync(stream.fileno())
        digest = sha256(temp)
        os.replace(temp, arr_path)
    finally:
        temp.unlink(missing_ok=True)
    atomic_json(receipt_path, {
        "start": start, "end": end, "rows": end - start,
        "dimension": int(fp16.shape[1]), "dtype": "float16",
        "input_sha256": input_sha, "model_fingerprint": model_sha,
        "shard_sha256": digest, "nonfinite": 0, "gpu_peak_reserved_mib": peak_mib,
        "nonstandard_junction_warnings": nonstandard_warnings,
    })


class TCRBert:
    def __init__(self, model_path: Path, device: str, precision: str,
                 batch_size: int, peak_limit_mib: int):
        import torch
        from transformers import BertModel, BertTokenizerFast

        self.torch = torch
        self.device = device
        self.precision = precision
        self.batch_size = batch_size
        self.peak_limit_mib = peak_limit_mib
        self.tokenizer = BertTokenizerFast.from_pretrained(
            str(model_path), do_lower_case=False, local_files_only=True)
        self.model = BertModel.from_pretrained(
            str(model_path), add_pooling_layer=False, local_files_only=True)
        self.model.eval().requires_grad_(False).to(device)
        if precision == "fp16":
            require(device == "cuda", "fp16 TCR-BERT requires CUDA")
            self.model.half()

    def embed(self, sequences: list[str]):
        import numpy as np

        torch = self.torch
        result = []
        offset = 0
        while offset < len(sequences):
            size = min(self.batch_size, len(sequences) - offset)
            try:
                encoded = self.tokenizer(
                    [" ".join(s) for s in sequences[offset:offset + size]],
                    return_tensors="pt", padding=True, truncation=False)
                require(encoded["input_ids"].shape[1] <= 64,
                        "Tokenizer output exceeds 64 tokens")
                encoded = {k: v.to(self.device) for k, v in encoded.items()}
                with torch.inference_mode():
                    h = self.model(**encoded).last_hidden_state
                    mask = encoded["attention_mask"].unsqueeze(-1).to(h.dtype)
                    pooled = (h * mask).sum(1) / mask.sum(1).clamp_min(1)
                result.append(pooled.float().cpu().numpy())
                offset += size
                if self.device == "cuda":
                    torch.cuda.synchronize()
                    peak = torch.cuda.max_memory_reserved() / (1024 ** 2)
                    require(peak <= self.peak_limit_mib,
                            f"GPU reserved memory {peak:.0f} MiB exceeds policy")
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                if size <= 1:
                    raise
                self.batch_size = max(1, size // 2)
                print(f"CUDA OOM: reducing TCR-BERT batch to {self.batch_size}",
                      flush=True)
        return np.concatenate(result, axis=0)



class NonstandardJunctionCounter(logging.Filter):
    """Count package junction warnings without writing millions of log lines."""

    def __init__(self):
        super().__init__()
        self.count = 0
        self.examples: list[str] = []

    def filter(self, record: logging.LogRecord) -> bool:
        message = record.getMessage()
        if ("doesn't look like a standardised junction" in message
                and record.levelno == logging.WARNING):
            self.count += 1
            if len(self.examples) < 20:
                self.examples.append(message.split(" doesn't look")[0])
            return False
        return True



class SceptrCdr3:
    def __init__(self, device: str, batch_size: int, peak_limit_mib: int):
        import torch
        import sceptr
        from sceptr import variant

        if device == "cpu":
            sceptr.disable_hardware_acceleration()
        self.torch = torch
        self.device = device
        self.batch_size = batch_size
        self.peak_limit_mib = peak_limit_mib
        self.model = variant.cdr3_only()
        self.warning_counter = NonstandardJunctionCounter()
        logging.getLogger("libtcrlm.schema.tcr").addFilter(self.warning_counter)
        if device == "cuda":
            self.model.enable_hardware_acceleration()
        else:
            self.model.disable_hardware_acceleration()
        self.model.set_batch_size(batch_size)

    def embed(self, sequences: list[str]):
        import numpy as np
        import pandas as pd

        result = []
        offset = 0
        while offset < len(sequences):
            size = min(self.batch_size, len(sequences) - offset)
            try:
                frame = pd.DataFrame({"CDR3B": sequences[offset:offset + size]})
                vectors = self.model.calc_vector_representations(frame)
                require(vectors.shape == (size, 64),
                        f"SCEPTR returned unexpected shape {vectors.shape}")
                result.append(np.asarray(vectors, dtype=np.float32))
                offset += size
                if self.device == "cuda":
                    self.torch.cuda.synchronize()
                    peak = self.torch.cuda.max_memory_reserved() / (1024 ** 2)
                    require(peak <= self.peak_limit_mib,
                            f"GPU reserved memory {peak:.0f} MiB exceeds policy")
            except self.torch.cuda.OutOfMemoryError:
                self.torch.cuda.empty_cache()
                if size <= 1:
                    raise
                self.batch_size = max(1, size // 2)
                self.model.set_batch_size(self.batch_size)
                print(f"CUDA OOM: reducing SCEPTR batch to {self.batch_size}",
                      flush=True)
        return np.concatenate(result, axis=0)


def publish_bank(outdir: Path, n: int, dimension: int, shard_rows: int,
                 input_sha: str, model_sha: str) -> str:
    import numpy as np

    temp = outdir / f"bank_fp16.npy.partial.{os.getpid()}"
    final = outdir / "bank_fp16.npy"
    try:
        bank = np.lib.format.open_memmap(
            temp, mode="w+", dtype=np.float16, shape=(n, dimension))
        for start in range(0, n, shard_rows):
            end = min(start + shard_rows, n)
            require(verify_shard(outdir, start, end, dimension, input_sha, model_sha),
                    f"Shard missing or invalid at {start}:{end}")
            arr = np.load(
                outdir / "shards" / f"{shard_name(start, end)}.npy",
                mmap_mode="r", allow_pickle=False)
            require(np.isfinite(arr).all(), f"Nonfinite shard at {start}:{end}")
            bank[start:end] = arr
        bank.flush()
        del bank
        with temp.open("rb") as stream:
            os.fsync(stream.fileno())
        bank_sha = sha256(temp)
        os.replace(temp, final)
        return bank_sha
    finally:
        temp.unlink(missing_ok=True)


def preflight_gpu(device: str) -> None:
    if device != "cuda":
        return
    import torch

    require(torch.cuda.is_available(), "CUDA requested but unavailable")
    require(torch.cuda.device_count() == 1,
            "Expected exactly one visible GPU for single-process inference")
    free, total = torch.cuda.mem_get_info()
    require(free >= 13 * 1024 ** 3,
            f"GPU free memory too low for solo run: {free / 1024 ** 2:.0f} MiB")
    print(f"GPU: {torch.cuda.get_device_name(0)}; free {free / 1024**2:.0f} MiB; "
          f"total {total / 1024**2:.0f} MiB", flush=True)




@contextmanager
def gpu_lock(path: Path, model: str, mode: str):
    """Kernel flock is authoritative; any owner text left after a crash is stale."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            lock.seek(0)
            owner = lock.read(2048).strip()
            raise RuntimeError(
                f"GPU slot busy at {path}; recorded owner={owner or 'unknown'}"
            ) from exc
        lock.seek(0)
        stale = lock.read(2048).strip()
        if stale:
            print(f"Replacing stale GPU lock owner: {stale}", flush=True)
        owner = {
            "pid": os.getpid(), "hostname": os.uname().nodename,
            "acquired_utc": datetime.now(timezone.utc).isoformat(),
            "model": model, "mode": mode,
        }
        lock.seek(0)
        lock.truncate()
        json.dump(owner, lock, sort_keys=True)
        lock.flush()
        os.fsync(lock.fileno())
        try:
            yield
        finally:
            lock.seek(0)
            lock.truncate()
            lock.flush()
            os.fsync(lock.fileno())
            fcntl.flock(lock, fcntl.LOCK_UN)



def stratified_pilot(path: Path, n: int, count: int, seed: int):
    """Sample globally across the union, with coverage of rare CDR3 lengths."""
    import numpy as np
    import pyarrow as pa
    import pyarrow.compute as pc
    import pyarrow.parquet as pq

    table = pq.read_table(path, columns=["seq_idx", "aa"])
    require(table.num_rows == n, "Union row count changed during pilot sampling")
    lengths = pc.utf8_length(table["aa"]).to_numpy(zero_copy_only=False)
    edges = np.asarray([10, 15, 20, 25, 30, 62], dtype=np.int32)
    groups = np.searchsorted(edges, lengths, side="left")
    populations = np.bincount(groups, minlength=7)
    require(populations[6] == 0,
            f"{populations[6]} QC CDR3 sequences exceed TCR-BERT's 62-residue context")
    target = min(count, n)
    rng = np.random.default_rng(seed)
    quotas = rng.multivariate_hypergeometric(populations[:6], target)
    active = int(np.count_nonzero(populations[:6]))
    floor = min(100, target // max(1, active))
    floors = np.minimum(populations[:6], floor)
    for i in range(6):
        while quotas[i] < floors[i]:
            donor = int(np.argmax(quotas - floors))
            require(quotas[donor] > floors[donor], "Cannot satisfy pilot length strata")
            quotas[donor] -= 1
            quotas[i] += 1
    chosen = [
        rng.choice(np.flatnonzero(groups == i), size=int(quotas[i]), replace=False)
        for i in range(6) if quotas[i] > 0
    ]
    indices = np.sort(np.concatenate(chosen).astype(np.int64))
    require(len(indices) == target and len(np.unique(indices)) == target,
            "Pilot sample has missing or duplicate rows")
    selected = table.take(pa.array(indices))
    ids = selected["seq_idx"].to_numpy(zero_copy_only=False)
    require(np.array_equal(ids, indices),
            "Union seq_idx differs from physical row position in pilot")
    sequences = selected["aa"].to_pylist()
    require(all(isinstance(aa, str) and CANONICAL.fullmatch(aa) is not None
                for aa in sequences), "Pilot sampled noncanonical CDR3")
    stats = {
        "edges": edges.tolist(),
        "bin_labels": ["1-10", "11-15", "16-20", "21-25", "26-30", "31-62"],
        "population_by_length_bin": populations[:6].tolist(),
        "sample_by_length_bin": quotas.tolist(),
        "global_min_length": int(lengths.min()),
        "global_max_length": int(lengths.max()),
        "seed": seed,
    }
    return indices, sequences, stats


def atomic_npy(path: Path, array) -> str:
    import numpy as np

    temp = path.with_name(path.name + f".partial.{os.getpid()}")
    try:
        with temp.open("wb") as stream:
            np.save(stream, array, allow_pickle=False)
            stream.flush()
            os.fsync(stream.fileno())
        digest = sha256(temp)
        os.replace(temp, path)
        return digest
    finally:
        temp.unlink(missing_ok=True)


def make_backend(name: str, bert_path: Path, device: str, precision: str,
                 batch_size: int, peak_limit: int):
    if name == "tcrbert_mlm":
        return TCRBert(bert_path, device, precision, batch_size, peak_limit)
    return SceptrCdr3(device, batch_size, peak_limit)


def precision_comparison(name: str, bert_path: Path, sequences: list[str],
                         gpu_vectors, peak_limit: int, seed: int) -> dict:
    import numpy as np

    rng = np.random.default_rng(seed + 1)
    positions = np.sort(rng.choice(
        len(sequences), size=min(64, len(sequences)), replace=False))
    cpu = make_backend(name, bert_path, "cpu", "fp32", 32, peak_limit)
    reference = cpu.embed([sequences[i] for i in positions])
    del cpu
    comparison = np.asarray(gpu_vectors[positions], dtype=np.float32)
    require(np.isfinite(reference).all() and np.isfinite(comparison).all(),
            "Precision comparison has nonfinite vectors")
    dot = np.einsum("ij,ij->i", comparison, reference)
    norm = np.linalg.norm(comparison, axis=1) * np.linalg.norm(reference, axis=1)
    require(np.all(norm > 0), "Precision comparison has zero-norm vector")
    cosine = dot / norm
    relative = np.linalg.norm(comparison - reference, axis=1) / np.linalg.norm(
        reference, axis=1)
    return {
        "rows": int(len(positions)),
        "cpu_precision": "fp32",
        "gpu_precision": "fp16 model" if name == "tcrbert_mlm" else "fp32 model",
        "cosine_min": float(cosine.min()),
        "cosine_median": float(np.median(cosine)),
        "relative_l2_max": float(relative.max()),
        "relative_l2_median": float(np.median(relative)),
    }


def run_pilot(args, outdir: Path, n: int, input_sha: str,
              spec: dict, model_sha: str, peak_limit: int) -> None:
    import numpy as np

    pilot_dir = outdir / "pilot"
    pilot_dir.mkdir(exist_ok=True)
    receipt = pilot_dir / "meta.json"
    indices_path = pilot_dir / "seq_idx.npy"
    vectors_path = pilot_dir / "vectors_fp16.npy"
    if receipt.is_file():
        previous = json.loads(receipt.read_text())
        if (previous.get("status") == "complete"
                and previous.get("input_sha256") == input_sha
                and previous.get("model_fingerprint") == model_sha
                and previous.get("requested_rows") == args.pilot_rows
                and indices_path.is_file() and vectors_path.is_file()
                and sha256(indices_path) == previous.get("seq_idx_sha256")
                and sha256(vectors_path) == previous.get("vectors_sha256")):
            print(f"Verified existing pilot: {vectors_path}", flush=True)
            return
    started = time.time()
    indices, sequences, strata = stratified_pilot(
        args.input, n, args.pilot_rows, seed=20260929)
    print(f"Stratified pilot selected {len(indices)} of {n} rows; "
          f"length range {strata['global_min_length']}-"
          f"{strata['global_max_length']}", flush=True)
    preflight_gpu(args.device)
    backend = make_backend(args.model, args.bert_model_path, args.device,
                           args.precision, args.batch_size, peak_limit)
    if args.device == "cuda":
        backend.torch.cuda.reset_peak_memory_stats()
    encode_started = time.time()
    vectors = backend.embed(sequences)
    encode_sec = time.time() - encode_started
    nonstandard_warnings = (
        backend.warning_counter.count if args.model == "sceptr_cdr3_only" else 0)
    nonstandard_examples = (
        backend.warning_counter.examples if args.model == "sceptr_cdr3_only" else [])
    require(vectors.shape == (len(indices), spec["hidden_dim"]),
            f"Pilot backend returned wrong shape: {vectors.shape}")
    require(np.isfinite(vectors).all(), "Pilot produced NaN or Inf")
    peak = (backend.torch.cuda.max_memory_reserved() / 1024 ** 2
            if args.device == "cuda" else 0.0)
    comparison = (precision_comparison(
        args.model, args.bert_model_path, sequences, vectors, peak_limit, 20260929)
        if args.device == "cuda" else None)
    fp16 = np.asarray(vectors, dtype=np.float16)
    require(np.isfinite(fp16).all(), "Pilot fp16 storage produced NaN or Inf")
    ids_sha = atomic_npy(indices_path, indices)
    vec_sha = atomic_npy(vectors_path, fp16)
    meta = {
        "status": "complete", "requested_rows": args.pilot_rows,
        "selected_rows": len(indices), "n_total": n,
        "input_path": str(args.input.resolve()), "input_sha256": input_sha,
        "model_fingerprint": model_sha, "model_spec": spec,
        "selection": strata, "seq_idx_sha256": ids_sha,
        "vectors_sha256": vec_sha, "shape": list(fp16.shape),
        "dtype": "float16", "nonfinite": 0,
        "nonstandard_junction_warnings": nonstandard_warnings,
        "nonstandard_junction_examples": nonstandard_examples,
        "encode_sec": round(encode_sec, 2),
        "seq_per_sec": round(len(indices) / encode_sec, 2),
        "elapsed_sec": round(time.time() - started, 2),
        "initial_batch_size": args.batch_size,
        "final_batch_size": backend.batch_size,
        "tokenizer_workers": args.tokenizer_workers,
        "gpu_peak_reserved_mib": round(peak, 1),
        "gpu_peak_limit_mib": peak_limit,
        "precision_comparison": comparison,
    }
    atomic_json(receipt, meta)
    print(f"pilot complete: {len(indices)} rows, {meta['seq_per_sec']:.1f} seq/s, "
          f"peak {peak:.0f} MiB, nonstandard junction warnings="
          f"{nonstandard_warnings}, CPU agreement={comparison}", flush=True)



def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, choices=MODELS)
    parser.add_argument("--input", type=Path, default=ROOT / "data" / "aa_union.parquet")
    parser.add_argument("--bert-model-path", type=Path, default=DEFAULT_BERT)
    parser.add_argument("--shard-rows", type=int, default=50_000)
    parser.add_argument("--pilot-rows", type=int, default=50_000)
    parser.add_argument("--full", action="store_true",
                        help="Explicitly encode all union rows and publish bank")
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--precision", choices=("fp16", "fp32"), default="fp16")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--tokenizer-workers", type=int, default=4)
    args = parser.parse_args()

    policy = json.loads((ROOT / "config" / "resource_policy.json").read_text())
    require(policy["embedding_shard_rows_min"] <= args.shard_rows
            <= policy["embedding_shard_rows_max"], "shard_rows violates policy")
    require(1 <= args.tokenizer_workers
            <= policy["embedding_tokenizer_workers_max"],
            "tokenizer_workers violates policy")
    require(args.batch_size >= 1, "batch_size must be positive")
    require(args.pilot_rows > 0, "pilot_rows must be positive")
    require(args.model != "sceptr_cdr3_only" or args.precision == "fp16",
            "SCEPTR output storage is fp16; leave default precision")
    os.environ["RAYON_NUM_THREADS"] = str(args.tokenizer_workers)
    os.environ["TOKENIZERS_PARALLELISM"] = "true"
    os.environ["OMP_NUM_THREADS"] = str(min(args.tokenizer_workers, 8))

    outdir = ROOT / "embeddings" / args.model
    (outdir / "shards").mkdir(parents=True, exist_ok=True)
    lock_path = ROOT / ".gpu.lock"
    with gpu_lock(lock_path, args.model, "full" if args.full else "pilot"):
        n, input_sha, input_stat = input_info(args.input)
        row_limit = n
        spec = model_spec(args.model, args.bert_model_path, args.precision)
        model_sha = json_hash(spec)
        peak_limit = int(policy["gpu_peak_mib_target"])
        if args.model == "tcrbert_mlm":
            atomic_json(outdir / "old_cache_assessment.json",
                        old_cache_assessment(spec))
        if not args.full:
            run_pilot(args, outdir, n, input_sha, spec, model_sha, peak_limit)
            return
        meta_path = outdir / "meta.json"
        if meta_path.exists():
            old = json.loads(meta_path.read_text())
            require(old.get("input_sha256") == input_sha
                    and old.get("model_fingerprint") == model_sha
                    and old.get("shard_rows") == args.shard_rows,
                    f"Existing embedding provenance differs: {meta_path}")
            require(old.get("n_total") == n, "Union row count changed")
            if old.get("status") == "complete":
                bank_path = outdir / "bank_fp16.npy"
                require(bank_path.is_file(), "Complete metadata without bank file")
                require(sha256(bank_path) == old.get("bank_sha256"),
                        "Published bank checksum differs from complete metadata")
                print(f"Already complete: {bank_path}")
                return
        started = time.time()
        meta = {
            "status": "running", "n_total": n, "row_limit": row_limit,
            "input_path": str(args.input.resolve()), "input_sha256": input_sha,
            "input_stat": list(input_stat), "model_fingerprint": model_sha,
            "model_spec": spec, "shard_rows": args.shard_rows,
            "bank_shape": [n, spec["hidden_dim"]],
            "bank_dtype": "float16", "gpu_peak_limit_mib": peak_limit,
            "initial_batch_size": args.batch_size,
            "tokenizer_workers": args.tokenizer_workers,
        }
        atomic_json(meta_path, meta)
        backend = None
        done = 0
        skipped = 0
        nonstandard_total = 0
        for start, end, sequences in iter_shards(
                args.input, args.shard_rows, row_limit):
            if verify_shard(outdir, start, end, spec["hidden_dim"],
                            input_sha, model_sha):
                skipped += 1
                done = end
                receipt = json.loads((outdir / "shards" /
                    f"{shard_name(start, end)}.json").read_text())
                nonstandard_total += receipt["nonstandard_junction_warnings"]
                print(f"verified existing shard {start}:{end}", flush=True)
                continue
            if backend is None:
                preflight_gpu(args.device)
                backend = make_backend(args.model, args.bert_model_path,
                                       args.device, args.precision,
                                       args.batch_size, peak_limit)
                if args.device == "cuda":
                    backend.torch.cuda.reset_peak_memory_stats()
            warnings_before = (
                backend.warning_counter.count
                if args.model == "sceptr_cdr3_only" else 0)
            vectors = backend.embed(sequences)
            warning_delta = (
                backend.warning_counter.count - warnings_before
                if args.model == "sceptr_cdr3_only" else 0)
            nonstandard_total += warning_delta
            require(vectors.shape == (end - start, spec["hidden_dim"]),
                    f"Wrong backend shape: {vectors.shape}")
            peak = (backend.torch.cuda.max_memory_reserved() / 1024 ** 2
                    if args.device == "cuda" else 0.0)
            write_shard(outdir, start, end, vectors, input_sha, model_sha,
                        round(peak, 1), warning_delta)
            done = end
            print(f"completed shard {start}:{end}; peak_reserved={peak:.0f} MiB; "
                  f"elapsed={time.time()-started:.1f}s", flush=True)
        require((args.input.stat().st_size, args.input.stat().st_mtime_ns)
                == input_stat, "Union input changed during embedding")
        meta.update({
            "status": "shards_complete",
            "n_done": done, "shards_verified_on_resume": skipped,
            "nonstandard_junction_warnings_total": nonstandard_total,
            "elapsed_sec": round(time.time() - started, 2),
            "final_batch_size": backend.batch_size if backend else args.batch_size,
            "gpu_peak_reserved_mib": round(
                backend.torch.cuda.max_memory_reserved() / 1024 ** 2, 1)
                if backend and args.device == "cuda" else 0.0,
        })
        atomic_json(meta_path, meta)
        digest = publish_bank(outdir, n, spec["hidden_dim"], args.shard_rows,
                              input_sha, model_sha)
        bank_stat = (outdir / "bank_fp16.npy").stat()
        meta.update({
            "status": "complete", "bank_sha256": digest,
            "bank_size_bytes": bank_stat.st_size,
            "bank_mtime_ns": bank_stat.st_mtime_ns,
        })
        atomic_json(meta_path, meta)
        print(f"published {outdir / 'bank_fp16.npy'}", flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"EMBEDDING FAILED: {type(exc).__name__}: {exc}", file=sys.stderr)
        raise

