#!/usr/bin/env python3
"""Label-blind 50k full-union CDR3vec_small cost and integrity pilot."""
from __future__ import annotations
from release_runtime import private_literal as _private_literal, release_path as _release_path
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import resource
import sys
import time

for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[name] = "4"

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

ROOT = Path(_release_path('path_99d8c1c6fde2a6b9'))
OUT = ROOT / "representation_comparison" / "pilot"
AA = ROOT / "data" / "aa_union.parquet"
SOURCE = ROOT / "third_party" / "TCR2vec"
SITE = ROOT / "third_party" / "site_packages"
CHECKPOINT = ROOT / "third_party" / "CDR3vec_small" / "CDR3vec_small_128"
PILOT = OUT / "pilot_50k.parquet"
SELECT = OUT / "selection.json"
BANK = OUT / "cdr3vec_small_fp16.npy"
RECEIPT = OUT / "cdr3vec_small_receipt.json"
LIMIT_MIB = 14336


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def save_json(path: Path, data: dict) -> None:
    tmp = path.with_suffix(path.suffix + ".partial")
    tmp.write_text(json.dumps(data, indent=2, sort_keys=True, allow_nan=False) + "\n")
    os.replace(tmp, path)


def rss_gib() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024 ** 2)


def prepare() -> dict:
    OUT.mkdir(parents=True, exist_ok=True)
    source_sha = sha(AA)
    if SELECT.exists():
        receipt = json.loads(SELECT.read_text())
        if (receipt.get("status") != "complete" or
            receipt.get("source_sha256") != source_sha or
            receipt.get("pilot_sha256") != sha(PILOT)):
            raise RuntimeError("Existing 50k selection has changed")
        return receipt
    table = pq.read_table(AA, columns=["seq_idx", "aa"])
    n = len(table)
    ids = table["seq_idx"].to_numpy(zero_copy_only=False)
    if n != 28380380 or not np.array_equal(ids, np.arange(n)):
        raise RuntimeError("Full-union index changed")
    lengths = pc.utf8_length(table["aa"]).to_numpy(zero_copy_only=False)
    edges = [(0, 11), (12, 13), (14, 15), (16, 17), (18, 100)]
    selected = []
    counts = {}
    for lo, hi in edges:
        candidates = np.flatnonzero((lengths >= lo) & (lengths <= hi))
        if len(candidates) < 10000:
            raise RuntimeError("Length stratum too small: %s-%s" % (lo, hi))
        rng = np.random.default_rng(20260930 + lo * 100 + hi)
        pick = np.sort(rng.choice(candidates, 10000, replace=False))
        selected.append(pick)
        counts["%s-%s" % (lo, hi)] = {"population": int(len(candidates)), "sampled": 10000}
    selected = np.sort(np.concatenate(selected)).astype(np.int64)
    if len(selected) != 50000 or len(np.unique(selected)) != 50000:
        raise RuntimeError("Pilot selection duplicate")
    take = table.take(pa.array(selected))
    lens = pa.array(lengths[selected].astype(np.int16))
    out = pa.table({"seq_idx": take["seq_idx"], "aa": take["aa"], "length": lens})
    tmp = PILOT.with_suffix(".parquet.partial")
    pq.write_table(out, tmp, compression="zstd")
    os.replace(tmp, PILOT)
    receipt = {"status": "complete", "label_blind": True,
               "source_path": str(AA), "source_sha256": source_sha,
               "pilot_sha256": sha(PILOT), "n_full_aa": n, "n_pilot": len(selected),
               "seed_rule": "20260930 + low_length*100 + high_length",
               "length_strata": counts, "rss_peak_gib": round(rss_gib(), 3),
               "script_sha256": sha(Path(__file__))}
    if receipt["rss_peak_gib"] > 512:
        raise RuntimeError("Pilot selection exceeded 512 GiB")
    save_json(SELECT, receipt)
    return receipt


def model_and_tokenizer():
    sys.path.insert(0, str(SITE))
    sys.path.insert(0, str(SOURCE))
    import torch
    from tape import TAPETokenizer
    from tcr2vec.model import TCR2vec
    model = TCR2vec(str(CHECKPOINT)).eval().cuda()
    tokenizer = TAPETokenizer(vocab="iupac")
    return torch, model, tokenizer


def encode_batch(torch, model, tokenizer, seqs):
    toks = [tokenizer.encode(s) for s in seqs]
    width = max(map(len, toks))
    ids = np.zeros((len(toks), width), dtype=np.int64)
    mask = np.zeros((len(toks), width), dtype=np.int64)
    for i, tok in enumerate(toks):
        ids[i, :len(tok)] = tok
        mask[i, :len(tok)] = 1
    with torch.inference_mode():
        v = model({"input_ids": torch.from_numpy(ids).cuda(),
                   "input_mask": torch.from_numpy(mask).cuda()})
    return np.asarray(v, dtype=np.float32)


def run(batch: int) -> dict:
    selection = prepare()
    if RECEIPT.exists():
        old = json.loads(RECEIPT.read_text())
        if old.get("status") == "complete" and old.get("bank_sha256") == sha(BANK):
            return old
        raise RuntimeError("Existing pilot receipt invalid")
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable")
    if batch < 1 or batch > 1024:
        raise ValueError("Batch outside pilot range")
    for p in (SOURCE / "tcr2vec/model.py", CHECKPOINT / "config.json",
              CHECKPOINT / "pytorch_model.bin"):
        if not p.is_file():
            raise FileNotFoundError(p)
    lock = ROOT.parent / ".gpu.lock"
    lock.parent.mkdir(parents=True, exist_ok=True)
    with lock.open("a+") as fd:
        fcntl.flock(fd.fileno(), fcntl.LOCK_EX)
        free, total = torch.cuda.mem_get_info()
        if free < 2 * 1024**3:
            raise RuntimeError("Less than 2 GiB GPU free before pilot")
        torch.cuda.set_per_process_memory_fraction(min(0.875, 14 * 1024**3 / total))
        torch.cuda.reset_peak_memory_stats()
        t0 = time.monotonic()
        torch, model, tokenizer = model_and_tokenizer()
        table = pq.read_table(PILOT, columns=["seq_idx", "aa"])
        aa = table["aa"].to_pylist()
        if len(aa) != 50000:
            raise RuntimeError("Pilot length mismatch")
        tmp = BANK.with_suffix(".npy.partial")
        out = np.lib.format.open_memmap(tmp, mode="w+", dtype=np.float16,
                                        shape=(len(aa), 128))
        effective = batch
        pos = 0
        failures = []
        while pos < len(aa):
            stop = min(pos + effective, len(aa))
            try:
                v = encode_batch(torch, model, tokenizer, aa[pos:stop])
                if v.shape != (stop-pos, 128) or not np.isfinite(v).all():
                    raise ValueError("Nonfinite/shape-invalid vector at %s" % pos)
                out[pos:stop] = v.astype(np.float16)
                pos = stop
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                if effective == 1:
                    failures.append(pos)
                    break
                effective = max(1, effective // 2)
        out.flush()
        del out
        if failures or pos != len(aa):
            raise RuntimeError("Pilot failed keys: %s, completed %s" % (failures, pos))
        check = np.load(tmp, mmap_mode="r", allow_pickle=False)
        if check.shape != (50000, 128) or not np.isfinite(check).all():
            raise RuntimeError("Stored pilot bank invalid")
        repeated = encode_batch(torch, model, tokenizer, aa[:64])
        first = np.asarray(check[:64], dtype=np.float32)
        rel = np.linalg.norm(first-repeated, axis=1) / np.maximum(np.linalg.norm(repeated, axis=1), 1e-8)
        if float(rel.max()) > 0.01:
            raise RuntimeError("Repeat or fp16 agreement failed")
        del check
        os.replace(tmp, BANK)
        seconds = time.monotonic()-t0
        receipt = {
            "status": "complete", "n_pilot": len(aa), "n_failed": 0,
            "source_selection_sha256": sha(SELECT),
            "pilot_sha256": selection["pilot_sha256"],
            "bank_sha256": sha(BANK),
            "source_repo_commit": "beca917e0fdaa173e44e0f49b1fecec1b6707c14",
            "source_archive_sha256": sha(ROOT / "third_party/TCR2vec_main.zip"),
            "model_source_sha256": sha(SOURCE / "tcr2vec/model.py"),
            "checkpoint_sha256": sha(CHECKPOINT / "pytorch_model.bin"),
            "config_sha256": sha(CHECKPOINT / "config.json"),
            "input": "full productive standard-AA CDR3beta",
            "tokenizer": "TAPE IUPAC", "pooling": "TCR2vec author mask-aware mean",
            "dimension": 128, "batch_initial": batch, "batch_effective": effective,
            "gpu_total_mib": int(total / 1024**2),
            "gpu_free_before_mib": int(free / 1024**2),
            "gpu_peak_allocated_mib": round(torch.cuda.max_memory_allocated() / 1024**2, 2),
            "gpu_peak_reserved_mib": round(torch.cuda.max_memory_reserved() / 1024**2, 2),
            "cpu_rss_peak_gib": round(rss_gib(), 3),
            "seconds": round(seconds, 3), "sequences_per_second": round(len(aa)/seconds, 2),
            "repeat_relative_l2_max": round(float(rel.max()), 8),
            "script_sha256": sha(Path(__file__)),
        }
        if (receipt["gpu_peak_reserved_mib"] > LIMIT_MIB or
            receipt["cpu_rss_peak_gib"] > 512):
            raise RuntimeError("Pilot exceeded resource cap")
        save_json(RECEIPT, receipt)
        return receipt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("prepare", "run"))
    parser.add_argument("--batch", type=int, default=128)
    args = parser.parse_args()
    result = prepare() if args.mode == "prepare" else run(args.batch)
    print(json.dumps(result, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
