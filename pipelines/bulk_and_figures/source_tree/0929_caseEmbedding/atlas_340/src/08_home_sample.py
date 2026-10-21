#!/usr/bin/env python3
"""Label-blind Home 4096-AA-per-patient draw shared by all representation arms."""
from __future__ import annotations
from release_runtime import private_literal as _private_literal, release_path as _release_path
import csv
import hashlib
import json
import os
from pathlib import Path
import resource
import sys
import time

for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[name] = "8"

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

ROOT = Path(_release_path('path_99d8c1c6fde2a6b9'))
OUT = ROOT / "representation_comparison" / "home_sample"
MANIFEST = ROOT / "config/cohort_samples.tsv"
SOURCE = ROOT / "data/patient_aa"
UNION = ROOT / "data/aa_union.parquet"
DRAW = OUT / "patient_aa_draw.parquet"
UNIQUE = OUT / "unique_aa.parquet"
TOTALS = OUT / "patient_totals.tsv"
RECEIPT = OUT / "sample_receipt.json"
N_DRAW = 4096


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def seed(patient):
    return int.from_bytes(hashlib.sha256((_private_literal('private_4243dc7e9f914613') + patient).encode()).digest()[:8], "little")


def write_json(path, d):
    tmp = path.with_suffix(path.suffix + ".partial")
    tmp.write_text(json.dumps(d, indent=2, sort_keys=True, allow_nan=False) + "\n")
    os.replace(tmp, path)


def main():
    start = time.monotonic()
    OUT.mkdir(parents=True, exist_ok=True)
    if RECEIPT.exists():
        d = json.loads(RECEIPT.read_text())
        if (d.get("status") == "complete" and d.get("manifest_sha256") == sha(MANIFEST)
            and d.get("union_sha256") == sha(UNION) and
            d.get("draw_sha256") == sha(DRAW) and d.get("unique_sha256") == sha(UNIQUE)):
            print(json.dumps(d, sort_keys=True))
            return
        raise RuntimeError("Existing sample receipt invalid")
    with MANIFEST.open(newline="") as f:
        rows = list(csv.DictReader(f, delimiter="\t"))
    home = [r for r in rows if r["cohort"] == "home"]
    if len(home) != 138:
        raise RuntimeError("Home library manifest count changed")
    by_patient = {}
    for r in home:
        by_patient.setdefault(r["patient_key"], []).append(r["sample_id"])
    if len(by_patient) != 137:
        raise RuntimeError("Home patient count changed")
    records = []
    totals = []
    sources = {}
    for patient, samples in sorted(by_patient.items()):
        combined = {}
        for sample in samples:
            path = SOURCE / ("home__" + sample + ".parquet")
            if not path.is_file():
                raise FileNotFoundError(path)
            table = pq.read_table(path, columns=["patient_key", "aa", "read_count"])
            if set(table["patient_key"].to_pylist()) != {patient}:
                raise RuntimeError("Patient linkage differs in %s" % path)
            aa = table["aa"].to_pylist()
            reads = table["read_count"].to_numpy(zero_copy_only=False)
            if len(aa) != len(set(aa)) or len(aa) != len(reads) or (reads <= 0).any():
                raise RuntimeError("Invalid sample AA/read_count in %s" % path)
            for a, count in zip(aa, reads):
                combined[a] = combined.get(a, 0) + int(count)
            sources[sample] = sha(path)
        items = sorted(combined.items())
        if len(items) < N_DRAW:
            raise RuntimeError("Patient has fewer than 4096 eligible AA: %s" % patient)
        rng = np.random.default_rng(seed(patient))
        pick = np.sort(rng.choice(len(items), N_DRAW, replace=False))
        chosen = [items[i] for i in pick]
        all_reads = sum(combined.values())
        sampled_reads = sum(v for _, v in chosen)
        totals.append((patient, len(items), all_reads, N_DRAW, sampled_reads,
                       N_DRAW/len(items), sampled_reads/all_reads))
        records.extend((patient, a, count) for a, count in chosen)
    if len(records) != 137 * N_DRAW:
        raise RuntimeError("Incorrect patient draw size")
    aas = pa.array([r[1] for r in records], type=pa.string())
    unique_aa = sorted(set(aas.to_pylist()))
    union = pq.read_table(UNION, columns=["seq_idx", "aa"])
    hit = pc.index_in(pa.array(unique_aa), value_set=union["aa"]).to_numpy(zero_copy_only=False)
    if len(hit) != len(unique_aa) or (hit < 0).any() or (hit >= len(union)).any():
        raise RuntimeError("Draw AA absent from full union")
    if union["aa"].take(pa.array(hit)).to_pylist() != unique_aa:
        raise RuntimeError("Full union AA index mismatch")
    mapping = dict(zip(unique_aa, hit))
    draw_table = pa.table({"patient_key": pa.array([r[0] for r in records]),
                           "aa": aas,
                           "seq_idx": pa.array([mapping[r[1]] for r in records], type=pa.int64()),
                           "read_count": pa.array([r[2] for r in records], type=pa.int64())})
    unique_table = pa.table({"aa": pa.array(unique_aa), "seq_idx": pa.array(hit, type=pa.int64())})
    for path, table in ((DRAW, draw_table), (UNIQUE, unique_table)):
        tmp = path.with_suffix(path.suffix + ".partial")
        pq.write_table(table, tmp, compression="zstd")
        os.replace(tmp, path)
    with TOTALS.with_suffix(".tsv.partial").open("w", newline="") as f:
        w = csv.writer(f, delimiter="\t")
        w.writerow(("patient_key", "total_unique_aa", "total_reads",
                    "draw_unique_aa", "draw_reads", "aa_coverage", "read_coverage"))
        w.writerows(totals)
    os.replace(TOTALS.with_suffix(".tsv.partial"), TOTALS)
    d = {"status": "complete", "label_blind": True, "n_home_libraries": 138,
         "n_home_patients": 137, "n_aa_per_patient": N_DRAW,
         "n_patient_aa_rows": len(records), "n_unique_aa": len(unique_aa),
         "draw_sha256": sha(DRAW), "unique_sha256": sha(UNIQUE),
         "totals_sha256": sha(TOTALS), "union_sha256": sha(UNION),
         "manifest_sha256": sha(MANIFEST), "source_file_sha256": sources,
         "method": "uniform without replacement over each patient's complete unique AA; same AA and read count for every representation",
         "selection_seed": 20260930, "labels_read": False,
         "seconds": round(time.monotonic()-start, 3),
         "peak_rss_gib": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/(1024**2), 3),
         "script_sha256": sha(Path(__file__))}
    if d["peak_rss_gib"] > 512:
        raise RuntimeError("Sample selection exceeds 512 GiB")
    write_json(RECEIPT, d)
    print(json.dumps({k:v for k,v in d.items() if k != "source_file_sha256"}, sort_keys=True))


if __name__ == "__main__":
    main()
