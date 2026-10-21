#!/usr/bin/env python3
"""Independent full-bank finite, coverage, checksum and pilot-row verification."""
from release_runtime import private_literal as _private_literal, release_path as _release_path
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile

import numpy as np

ROOT = Path(_release_path('path_1304218e86d2e596'))


def digest(path):
    h = hashlib.sha256()
    with open(path, "rb") as stream:
        while chunk := stream.read(8 * 1024 * 1024):
            h.update(chunk)
    return h.hexdigest()


def check(condition, message):
    if not condition:
        raise RuntimeError(message)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=("tcrbert_mlm", "sceptr_cdr3_only", "tcrbert_dualtask_aa_mean"), required=True)
    args = parser.parse_args()
    out = ROOT / "embeddings" / args.model
    meta = json.loads((out / "meta.json").read_text())
    check(meta["status"] == "complete", "bank metadata is not complete")
    n, dim = meta["bank_shape"]
    check(n == meta["n_total"] == 23_974_109, "unexpected union row count")
    check(meta["bank_dtype"] == "float16", "unexpected bank dtype")
    bank_path = out / "bank_fp16.npy"
    stat = bank_path.stat()
    check(stat.st_size == meta["bank_size_bytes"], "bank size mismatch")
    check(stat.st_mtime_ns == meta["bank_mtime_ns"], "bank mtime mismatch")
    bank = np.load(bank_path, mmap_mode="r", allow_pickle=False)
    check(bank.shape == (n, dim) and bank.dtype == np.float16, "bank shape or dtype mismatch")
    receipts = sorted((out / "shards").glob("*.json"))
    expected = (n + meta["shard_rows"] - 1) // meta["shard_rows"]
    check(len(receipts) == expected, "shard receipt count mismatch")
    for start in range(0, n, meta["shard_rows"]):
        end = min(start + meta["shard_rows"], n)
        name = f"{start:012d}_{end:012d}"
        receipt = out / "shards" / (name + ".json")
        shard = out / "shards" / (name + ".npy")
        check(receipt.is_file() and shard.is_file(), f"missing shard {name}")
        rec = json.loads(receipt.read_text())
        check(rec["start"] == start and rec["end"] == end, f"bad shard coordinates {name}")
        check(rec["input_sha256"] == meta["input_sha256"], f"input hash mismatch {name}")
        check(rec["model_fingerprint"] == meta["model_fingerprint"], f"model hash mismatch {name}")
    nonfinite = 0
    zero_rows = 0
    for start in range(0, n, 100_000):
        block = bank[start:min(start + 100_000, n)]
        nonfinite += int(np.count_nonzero(~np.isfinite(block)))
        zero_rows += int(np.count_nonzero(~np.any(block != 0, axis=1)))
    check(nonfinite == 0 and zero_rows == 0, f"nonfinite={nonfinite}, zero_rows={zero_rows}")
    actual_sha = digest(bank_path)
    check(actual_sha == meta["bank_sha256"], "full bank SHA mismatch")
    pilot_meta = json.loads((out / "pilot" / "meta.json").read_text())
    check(pilot_meta["input_sha256"] == meta["input_sha256"], "pilot input differs")
    check(pilot_meta["model_fingerprint"] == meta["model_fingerprint"], "pilot model differs")
    ids = np.load(out / "pilot" / "seq_idx.npy", mmap_mode="r", allow_pickle=False)
    vec = np.load(out / "pilot" / "vectors_fp16.npy", mmap_mode="r", allow_pickle=False)
    positions = np.linspace(0, len(ids) - 1, 128, dtype=np.int64)
    left = bank[np.asarray(ids[positions], dtype=np.int64)].astype(np.float32)
    right = vec[positions].astype(np.float32)
    den = np.linalg.norm(left, axis=1) * np.linalg.norm(right, axis=1)
    check(np.all(den > 0), "zero pilot sample vector")
    cosine = np.einsum("ij,ij->i", left, right) / den
    check(float(cosine.min()) > 0.999, "pilot and bank rows disagree")
    result = {
        "status": "PASS", "model": args.model, "n_total": n, "dimension": dim,
        "shards": len(receipts), "nonfinite_values": nonfinite,
        "zero_vectors": zero_rows, "bank_sha256_verified": actual_sha,
        "pilot_rows_compared": len(positions), "pilot_bank_min_cosine": float(cosine.min()),
        "input_sha256": meta["input_sha256"],
        "model_fingerprint": meta["model_fingerprint"],
    }
    if args.model == "sceptr_cdr3_only":
        compat = json.loads((out / "input_compatibility.json").read_text())
        check(compat["status"] == "PASS" and compat["input_sha256"] == meta["input_sha256"],
              "SCEPTR input audit mismatch")
        check(compat["unparseable_count"] == 0, "SCEPTR unparseable inputs")
        check(meta["nonstandard_junction_warnings_total"] == compat["nonstandard_count"],
              "SCEPTR warning count mismatch")
        result["nonstandard_junction_warnings"] = compat["nonstandard_count"]
        result["nonstandard_junction_fraction"] = compat["nonstandard_count"] / n
        result["raw_aa_preserved"] = True
        result["unparseable_count"] = 0
    temp = out / f"verification.json.partial.{os.getpid()}"
    with temp.open("w") as stream:
        json.dump(result, stream, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temp, out / "verification.json")
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
