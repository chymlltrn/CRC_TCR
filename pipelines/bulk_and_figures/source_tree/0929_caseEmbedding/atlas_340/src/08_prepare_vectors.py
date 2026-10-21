#!/usr/bin/env python3
"""Align frozen full banks and encode a label-blind Home comparison draw.

All bank rows follow home_sample/unique_aa.parquet exactly. This script never
reads outcome labels and does not alter the frozen full-repertoire banks.
"""
from __future__ import annotations
from release_runtime import private_literal as _private_literal, release_path as _release_path

import argparse
import fcntl
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import resource
import time
from datetime import datetime, timezone

for _name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
              "NUMEXPR_NUM_THREADS"):
    os.environ[_name] = "4"

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

ROOT = Path(_release_path('path_99d8c1c6fde2a6b9'))
SAMPLE = ROOT / "representation_comparison/home_sample/unique_aa.parquet"
SAMPLE_RECEIPT = ROOT / "representation_comparison/home_sample/sample_receipt.json"
INDEX = ROOT / "data/atlas_aa_index.parquet"
MANIFEST = ROOT / "embeddings/bank_manifest.json"
VERIFICATION = ROOT / "embeddings/verification.json"
OUT = ROOT / "representation_comparison/vectors"
MAPPING = OUT / "sample_bank_mapping.parquet"
MAPPING_RECEIPT = OUT / "mapping_receipt.json"
PILOT_SCRIPT = ROOT / "src/08_cdr3vec_pilot.py"
PILOT_RECEIPT = ROOT / "representation_comparison/pilot/cdr3vec_small_receipt.json"
CHECKPOINT = ROOT / "third_party/CDR3vec_small/CDR3vec_small_128"
SOURCE = ROOT / "third_party/TCR2vec"
GPU_LIMIT_MIB = 14336
CHUNK_SIZE = 50000
MODELS = ("tcrbert_dualtask_aa_mean", "sceptr_cdr3_only")


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def save_json(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".partial")
    tmp.write_text(json.dumps(obj, indent=2, sort_keys=True, allow_nan=False) + "\n")
    os.replace(tmp, path)


def peak_rss_gib() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024 ** 2)


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def sample_info():
    if not SAMPLE.is_file() or not SAMPLE_RECEIPT.is_file():
        raise RuntimeError("Completed Home draw is required")
    receipt = json.loads(SAMPLE_RECEIPT.read_text())
    sample_sha = sha(SAMPLE)
    if receipt.get("unique_sha256") not in (None, sample_sha):
        raise RuntimeError("Home draw changed after receipt")
    table = pq.read_table(SAMPLE, columns=["aa", "seq_idx"])
    aa = table["aa"].to_pylist()
    ids = table["seq_idx"].to_numpy(zero_copy_only=False)
    if len(aa) != 534376 or len(np.unique(ids)) != len(ids):
        raise RuntimeError("Home draw AA count or index uniqueness changed")
    if any(not s or set(s) - set("ACDEFGHIKLMNPQRSTVWY") for s in aa):
        raise RuntimeError("Home draw contains invalid CDR3 AA")
    return table, sample_sha


def load_mapping():
    OUT.mkdir(parents=True, exist_ok=True)
    sample, sample_sha = sample_info()
    manifest = json.loads(MANIFEST.read_text())
    if manifest.get("status") != "complete" or manifest.get("n_atlas") != 28380380:
        raise RuntimeError("Full atlas manifest incomplete")
    if MAPPING_RECEIPT.exists():
        receipt = json.loads(MAPPING_RECEIPT.read_text())
        if (receipt.get("status") == "complete"
                and receipt.get("sample_sha256") == sample_sha
                and receipt.get("index_sha256") == manifest["index_sha256"]
                and MAPPING.exists() and sha(MAPPING) == receipt.get("mapping_sha256")):
            mapping = pq.read_table(MAPPING)
            if mapping.num_rows != sample.num_rows or not pc.all(pc.equal(
                    mapping["aa"], sample["aa"])).as_py():
                raise RuntimeError("Existing mapping does not match Home sample")
            return mapping, sample_sha, receipt
        raise RuntimeError("Existing mapping receipt is inconsistent")
    index_sha = sha(INDEX)
    if index_sha != manifest["index_sha256"]:
        raise RuntimeError("Frozen global index hash changed")
    full = pq.read_table(INDEX, columns=["seq_idx", "aa", "bank_source", "bank_idx"])
    ids = sample["seq_idx"].to_numpy(zero_copy_only=False)
    if ids.min() < 0 or ids.max() >= full.num_rows:
        raise RuntimeError("Sample global seq_idx outside atlas union")
    picked = full.take(pa.array(ids, type=pa.int64()))
    if not np.array_equal(picked["seq_idx"].to_numpy(zero_copy_only=False), ids):
        raise RuntimeError("Global seq_idx is not a physical row index")
    if not pc.all(pc.equal(picked["aa"], sample["aa"])).as_py():
        raise RuntimeError("Global index AA differs from sampled AA")
    sources = picked["bank_source"].to_pylist()
    if set(sources) != {"old", "new"}:
        raise RuntimeError("Unexpected source classes in sample")
    bank_ids = picked["bank_idx"].to_numpy(zero_copy_only=False)
    if np.any(bank_ids < 0):
        raise RuntimeError("Negative bank index")
    mapping = pa.table({
        "aa": sample["aa"],
        "seq_idx": sample["seq_idx"],
        "bank_source": picked["bank_source"],
        "bank_idx": picked["bank_idx"],
    })
    tmp = MAPPING.with_suffix(".parquet.partial")
    pq.write_table(mapping, tmp, compression="zstd")
    os.replace(tmp, MAPPING)
    receipt = {
        "status": "complete", "created_utc": now_utc(), "label_blind": True,
        "sample_sha256": sample_sha, "sample_receipt_sha256": sha(SAMPLE_RECEIPT),
        "index_sha256": index_sha, "manifest_sha256": sha(MANIFEST),
        "mapping_sha256": sha(MAPPING), "n_rows": mapping.num_rows,
        "n_old": sources.count("old"), "n_new": sources.count("new"),
        "row_order": "home_sample/unique_aa.parquet",
        "cpu_peak_rss_gib": round(peak_rss_gib(), 3),
        "script_sha256": sha(Path(__file__)),
    }
    if receipt["cpu_peak_rss_gib"] > 512:
        raise RuntimeError("512 GiB CPU RSS cap exceeded")
    save_json(MAPPING_RECEIPT, receipt)
    return mapping, sample_sha, receipt


def align_model(name: str, mapping, sample_sha: str, mapping_receipt: dict):
    model_manifest = json.loads(MANIFEST.read_text())["model_banks"][name]
    verified = json.loads(VERIFICATION.read_text())
    if verified.get("status") != "PASS":
        raise RuntimeError("Full-bank verification is not PASS")
    dim = model_manifest["dimension"]
    old_path = Path(model_manifest["old_bank_path"])
    new_path = Path(model_manifest["new_bank_path"])
    expected_old = verified["models"][name]["old_bank_sha256"]
    expected_new = verified["models"][name]["new_bank_sha256"]
    target_dir = OUT / name
    target_dir.mkdir(parents=True, exist_ok=True)
    target = target_dir / "bank_fp16.npy"
    receipt_path = target_dir / "receipt.json"
    if receipt_path.exists():
        receipt = json.loads(receipt_path.read_text())
        if (receipt.get("status") == "complete"
                and receipt.get("sample_sha256") == sample_sha
                and receipt.get("mapping_sha256") == mapping_receipt["mapping_sha256"]
                and receipt.get("old_bank_sha256") == expected_old
                and receipt.get("new_bank_sha256") == expected_new
                and target.exists() and sha(target) == receipt.get("bank_sha256")):
            bank = np.load(target, mmap_mode="r", allow_pickle=False)
            if bank.shape != (mapping.num_rows, dim):
                raise RuntimeError("Completed aligned bank shape changed")
            return receipt
        raise RuntimeError(f"Existing {name} receipt is inconsistent")
    t0 = time.monotonic()
    for path, expected in ((old_path, expected_old), (new_path, expected_new)):
        if sha(path) != expected:
            raise RuntimeError(f"Frozen {name} source bank hash changed: {path}")
    old = np.load(old_path, mmap_mode="r", allow_pickle=False)
    new = np.load(new_path, mmap_mode="r", allow_pickle=False)
    if (old.dtype != np.float16 or new.dtype != np.float16
            or old.shape != (23974109, dim) or new.shape != (4499952, dim)):
        raise RuntimeError(f"Frozen {name} source bank shape/dtype changed")
    sources = np.asarray(mapping["bank_source"].to_pylist(), dtype="U3")
    bank_idx = mapping["bank_idx"].to_numpy(zero_copy_only=False)
    if np.any(bank_idx[sources == "old"] >= len(old)) or np.any(
            bank_idx[sources == "new"] >= len(new)):
        raise RuntimeError(f"{name} mapping points beyond source bank")
    tmp = target.with_suffix(".npy.partial")
    out = np.lib.format.open_memmap(tmp, mode="w+", dtype=np.float16,
                                    shape=(mapping.num_rows, dim))
    nonfinite = 0
    for start in range(0, len(out), CHUNK_SIZE):
        stop = min(start + CHUNK_SIZE, len(out))
        block = np.empty((stop - start, dim), dtype=np.float16)
        src = sources[start:stop]
        idx = bank_idx[start:stop]
        old_mask = src == "old"
        new_mask = src == "new"
        block[old_mask] = old[idx[old_mask]]
        block[new_mask] = new[idx[new_mask]]
        nonfinite += int(block.size - np.isfinite(block).sum())
        out[start:stop] = block
    out.flush()
    del out
    if nonfinite:
        raise RuntimeError(f"{name} has {nonfinite} nonfinite values")
    aligned = np.load(tmp, mmap_mode="r", allow_pickle=False)
    if aligned.shape != (mapping.num_rows, dim) or aligned.dtype != np.float16:
        raise RuntimeError("Aligned bank shape/dtype invalid")
    check_rows = np.linspace(0, len(aligned) - 1, 257, dtype=np.int64)
    for row in check_rows:
        source = old if sources[row] == "old" else new
        if not np.array_equal(aligned[row], source[bank_idx[row]]):
            raise RuntimeError(f"Alignment mismatch at row {row}")
    del aligned
    os.replace(tmp, target)
    receipt = {
        "status": "complete", "completed_utc": now_utc(), "label_blind": True,
        "model": name, "dimension": dim, "dtype": "float16",
        "n_rows": mapping.num_rows, "n_failed": 0, "n_nonfinite": nonfinite,
        "n_old": int(np.sum(sources == "old")), "n_new": int(np.sum(sources == "new")),
        "row_order": "home_sample/unique_aa.parquet",
        "sample_sha256": sample_sha,
        "mapping_sha256": mapping_receipt["mapping_sha256"],
        "old_bank_path": str(old_path), "old_bank_sha256": expected_old,
        "new_bank_path": str(new_path), "new_bank_sha256": expected_new,
        "full_bank_verification_sha256": sha(VERIFICATION),
        "bank_sha256": sha(target), "exact_row_checks": len(check_rows),
        "elapsed_sec": round(time.monotonic() - t0, 3),
        "cpu_peak_rss_gib": round(peak_rss_gib(), 3),
        "cpu_threads": 4, "script_sha256": sha(Path(__file__)),
    }
    if receipt["cpu_peak_rss_gib"] > 512:
        raise RuntimeError("512 GiB CPU RSS cap exceeded")
    save_json(receipt_path, receipt)
    return receipt


def pilot_module():
    spec = importlib.util.spec_from_file_location("cdr3vec_pilot_frozen", PILOT_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError("CDR3vec pilot import failed")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def cdr3vec(batch: int):
    sample, sample_sha = sample_info()
    aa = sample["aa"].to_pylist()
    pilot_receipt = json.loads(PILOT_RECEIPT.read_text())
    if pilot_receipt.get("status") != "complete":
        raise RuntimeError("CDR3vec 50k pilot has not passed")
    source_hashes = {
        "checkpoint_sha256": sha(CHECKPOINT / "pytorch_model.bin"),
        "config_sha256": sha(CHECKPOINT / "config.json"),
        "model_source_sha256": sha(SOURCE / "tcr2vec/model.py"),
        "pilot_script_sha256": sha(PILOT_SCRIPT),
    }
    for field in ("checkpoint_sha256", "config_sha256", "model_source_sha256"):
        if pilot_receipt.get(field) != source_hashes[field]:
            raise RuntimeError(f"CDR3vec {field} differs from technical pilot")
    model_dir = OUT / "cdr3vec_small"
    chunks_dir = model_dir / "chunks"
    chunks_dir.mkdir(parents=True, exist_ok=True)
    target = model_dir / "bank_fp16.npy"
    final_receipt_path = model_dir / "receipt.json"
    if final_receipt_path.exists():
        receipt = json.loads(final_receipt_path.read_text())
        if (receipt.get("status") == "complete"
                and receipt.get("sample_sha256") == sample_sha
                and all(receipt.get(k) == v for k, v in source_hashes.items())
                and target.exists() and sha(target) == receipt.get("bank_sha256")):
            bank = np.load(target, mmap_mode="r", allow_pickle=False)
            if bank.shape != (len(aa), 128):
                raise RuntimeError("Completed CDR3vec bank shape changed")
            return receipt
        raise RuntimeError("Existing CDR3vec receipt is inconsistent")
    if batch < 1 or batch > 1024:
        raise ValueError("Batch must be 1..1024")
    module = pilot_module()
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable")
    lock = ROOT.parent / ".gpu.lock"
    with lock.open("a+") as fd:
        fcntl.flock(fd.fileno(), fcntl.LOCK_EX)
        free, total = torch.cuda.mem_get_info()
        if free < 2 * 1024**3:
            raise RuntimeError("Less than 2 GiB GPU free")
        torch.cuda.set_per_process_memory_fraction(
            min(0.875, GPU_LIMIT_MIB * 1024**2 / total))
        torch.cuda.reset_peak_memory_stats()
        start_all = time.monotonic()
        model = tokenizer = None
        chunk_receipts = []
        for start in range(0, len(aa), CHUNK_SIZE):
            stop = min(start + CHUNK_SIZE, len(aa))
            num = start // CHUNK_SIZE
            chunk = chunks_dir / f"chunk_{num:04d}.npy"
            receipt_path = chunks_dir / f"chunk_{num:04d}.json"
            if receipt_path.exists():
                cr = json.loads(receipt_path.read_text())
                if (cr.get("status") != "complete"
                        or cr.get("start") != start or cr.get("stop") != stop
                        or cr.get("sample_sha256") != sample_sha
                        or any(cr.get(k) != v for k, v in source_hashes.items())
                        or not chunk.exists() or sha(chunk) != cr.get("chunk_sha256")):
                    raise RuntimeError(f"Existing chunk {num} receipt inconsistent")
                arr = np.load(chunk, mmap_mode="r", allow_pickle=False)
                if arr.shape != (stop-start, 128) or arr.dtype != np.float16:
                    raise RuntimeError(f"Existing chunk {num} shape invalid")
                chunk_receipts.append(cr)
                continue
            if model is None:
                _, model, tokenizer = module.model_and_tokenizer()
            t0 = time.monotonic()
            tmp = chunk.with_suffix(".npy.partial")
            out = np.lib.format.open_memmap(
                tmp, mode="w+", dtype=np.float16, shape=(stop-start, 128))
            pos = start
            effective = batch
            while pos < stop:
                end = min(pos + effective, stop)
                try:
                    vectors = module.encode_batch(
                        torch, model, tokenizer, aa[pos:end])
                    if vectors.shape != (end-pos, 128) or not np.isfinite(vectors).all():
                        raise ValueError(f"Invalid CDR3vec vector at row {pos}")
                    out[pos-start:end-start] = vectors.astype(np.float16)
                    pos = end
                except torch.cuda.OutOfMemoryError:
                    torch.cuda.empty_cache()
                    if effective == 1:
                        raise RuntimeError(f"CDR3vec OOM at row {pos}")
                    effective = max(1, effective // 2)
            out.flush()
            del out
            saved = np.load(tmp, mmap_mode="r", allow_pickle=False)
            if not np.isfinite(saved).all():
                raise RuntimeError(f"CDR3vec chunk {num} nonfinite after fp16 cast")
            del saved
            os.replace(tmp, chunk)
            peak_mib = torch.cuda.max_memory_reserved() / 1024**2
            if peak_mib > GPU_LIMIT_MIB or peak_rss_gib() > 512:
                raise RuntimeError("CDR3vec resource cap exceeded")
            cr = {
                "status": "complete", "completed_utc": now_utc(),
                "start": start, "stop": stop, "n_rows": stop-start,
                "sample_sha256": sample_sha, **source_hashes,
                "chunk_sha256": sha(chunk),
                "batch_initial": batch, "batch_effective": effective,
                "gpu_peak_reserved_mib": round(peak_mib, 2),
                "gpu_peak_allocated_mib": round(
                    torch.cuda.max_memory_allocated()/1024**2, 2),
                "cpu_peak_rss_gib": round(peak_rss_gib(), 3),
                "seconds": round(time.monotonic()-t0, 3),
                "script_sha256": sha(Path(__file__)),
            }
            save_json(receipt_path, cr)
            chunk_receipts.append(cr)
            print(f"CDR3vec chunk {num}: {stop}/{len(aa)} complete", flush=True)
        if model is not None:
            del model
            torch.cuda.empty_cache()
        tmp = target.with_suffix(".npy.partial")
        final = np.lib.format.open_memmap(tmp, mode="w+", dtype=np.float16,
                                          shape=(len(aa), 128))
        for cr in chunk_receipts:
            start, stop = cr["start"], cr["stop"]
            part = np.load(chunks_dir / f"chunk_{start // CHUNK_SIZE:04d}.npy",
                           mmap_mode="r", allow_pickle=False)
            final[start:stop] = part
        final.flush()
        del final
        verified = np.load(tmp, mmap_mode="r", allow_pickle=False)
        nonfinite = 0
        for start in range(0, len(aa), CHUNK_SIZE):
            nonfinite += int((~np.isfinite(verified[start:start+CHUNK_SIZE])).sum())
        if nonfinite:
            raise RuntimeError(f"CDR3vec final bank has {nonfinite} nonfinite")
        del verified
        os.replace(tmp, target)
        receipt = {
            "status": "complete", "completed_utc": now_utc(), "label_blind": True,
            "model": "cdr3vec_small", "dimension": 128, "dtype": "float16",
            "n_rows": len(aa), "n_failed": 0, "n_nonfinite": 0,
            "n_chunks": len(chunk_receipts), "chunk_size": CHUNK_SIZE,
            "row_order": "home_sample/unique_aa.parquet",
            "sample_sha256": sample_sha, **source_hashes,
            "pilot_receipt_sha256": sha(PILOT_RECEIPT),
            "bank_sha256": sha(target),
            "gpu_total_mib": round(total/1024**2),
            "gpu_free_before_mib": round(free/1024**2),
            "gpu_peak_reserved_mib": round(
                torch.cuda.max_memory_reserved()/1024**2, 2),
            "gpu_peak_allocated_mib": round(
                torch.cuda.max_memory_allocated()/1024**2, 2),
            "cpu_peak_rss_gib": round(peak_rss_gib(), 3),
            "cpu_threads": 4,
            "elapsed_sec": round(time.monotonic()-start_all, 3),
            "script_sha256": sha(Path(__file__)),
        }
        if receipt["gpu_peak_reserved_mib"] > GPU_LIMIT_MIB or receipt["cpu_peak_rss_gib"] > 512:
            raise RuntimeError("CDR3vec final resource cap exceeded")
        save_json(final_receipt_path, receipt)
        return receipt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("align", "cdr3vec", "all"))
    parser.add_argument("--batch", type=int, default=128)
    args = parser.parse_args()
    if args.mode in ("align", "all"):
        mapping, sample_sha, mapping_receipt = load_mapping()
        for model in MODELS:
            receipt = align_model(model, mapping, sample_sha, mapping_receipt)
            print(json.dumps({"model": model, "status": receipt["status"],
                              "n_rows": receipt["n_rows"],
                              "bank_sha256": receipt["bank_sha256"]}), flush=True)
    if args.mode in ("cdr3vec", "all"):
        receipt = cdr3vec(args.batch)
        print(json.dumps({"model": "cdr3vec_small", "status": receipt["status"],
                          "n_rows": receipt["n_rows"],
                          "bank_sha256": receipt["bank_sha256"]}), flush=True)


if __name__ == "__main__":
    main()
