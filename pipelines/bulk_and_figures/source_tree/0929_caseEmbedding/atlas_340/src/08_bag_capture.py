#!/usr/bin/env python3
"""Exact heldout 2,048-bag repertoire/group capture over frozen screen draws."""
from __future__ import annotations
from release_runtime import private_literal as _private_literal, release_path as _release_path
import hashlib
import json
import os
from pathlib import Path
import resource
import time

for name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[name] = "8"

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

ROOT = Path(_release_path('path_99d8c1c6fde2a6b9'))
BASE = ROOT / "representation_comparison"
DRAW = BASE / "home_sample/patient_aa_draw.parquet"
TOTALS = BASE / "home_sample/patient_totals.tsv"
FOLDS = ROOT / "folds/folds.tsv"
GROUP_AA = ROOT / "groups/full_home/patient_group_aa_safe.parquet"
BURDEN = ROOT / "groups/full_home/patient_group_burden.parquet"
OUT = BASE / "coverage"
BAG_OUT = OUT / "heldout_bag_capture.tsv"
SUMMARY_OUT = OUT / "heldout_bag_capture_summary.tsv"
RECEIPT = OUT / "bag_receipt.json"
SEEDS = (17, 29, 43)
DRAWS = 8
BAG = 2048


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def stable_seed(*parts):
    value = "|".join(map(str, parts)).encode()
    return int.from_bytes(hashlib.sha256(value).digest()[:8], "little") % (2**32 - 1)


def write_tsv(path, frame):
    tmp = path.with_suffix(path.suffix + ".partial")
    frame.to_csv(tmp, sep="\t", index=False, float_format="%.12g")
    os.replace(tmp, path)


def main():
    started = time.monotonic()
    OUT.mkdir(parents=True, exist_ok=True)
    source = {name: sha(path) for name, path in {
        "draw": DRAW, "totals": TOTALS, "folds": FOLDS, "group_aa": GROUP_AA,
        "burden": BURDEN, "script": Path(__file__)}.items()}
    if RECEIPT.exists():
        old = json.loads(RECEIPT.read_text())
        if (old.get("status") == "complete" and old.get("source_sha256") == source
                and old.get("bag_output_sha256") == sha(BAG_OUT)
                and old.get("summary_output_sha256") == sha(SUMMARY_OUT)):
            print(json.dumps(old, sort_keys=True))
            return
        raise RuntimeError("Bag capture receipt mismatch")
    draw = pq.read_table(DRAW, columns=["patient_key", "seq_idx", "read_count"]).to_pandas()
    folds = pd.read_csv(FOLDS, sep="\t", dtype={"patient_key": str})
    heldout = folds.loc[folds.partition.eq("test"), ["patient_key", "outer_fold"]]
    if len(heldout) != 126 or heldout.patient_key.duplicated().any():
        raise ValueError("Frozen patient heldout map invalid")
    known = set(heldout.patient_key)
    draw = draw.loc[draw.patient_key.isin(known)]
    if len(draw) != 126 * 4096 or draw.duplicated(["patient_key", "seq_idx"]).any():
        raise ValueError("Known patient draw invalid")
    full_carriers = pq.read_table(BURDEN, columns=["patient_key", "group_id", "presence"]).to_pandas()
    full_carriers = full_carriers.loc[
        full_carriers.patient_key.isin(known) & full_carriers.presence.eq(1),
        ["patient_key", "group_id"]].drop_duplicates()
    full_by_patient = full_carriers.groupby("patient_key").group_id.nunique().to_dict()
    member = {}
    source_rows = 0
    for batch in pq.ParquetFile(GROUP_AA).iter_batches(
            batch_size=750_000, columns=["patient_key", "seq_idx", "group_id"]):
        group = batch.to_pandas()
        group = group.loc[group.patient_key.isin(known)]
        matched = group.merge(draw[["patient_key", "seq_idx"]], on=["patient_key", "seq_idx"],
                              how="inner", validate="many_to_one")
        source_rows += len(matched)
        for patient, seq, gid in matched.itertuples(index=False, name=None):
            member.setdefault((patient, int(seq)), set()).add(gid)
    totals = pd.read_csv(TOTALS, sep="\t", dtype={"patient_key": str}).set_index("patient_key")
    outer_map = heldout.set_index("patient_key").outer_fold.to_dict()
    rows = []
    for patient, data in draw.groupby("patient_key", sort=False):
        seq = data.seq_idx.to_numpy(dtype=np.int64)
        reads = data.read_count.to_numpy(dtype=np.int64)
        if len(seq) != 4096 or patient not in outer_map:
            raise ValueError("Patient bag draw invalid")
        groups_full = int(full_by_patient.get(patient, 0))
        total_unique = int(totals.loc[patient, "total_unique_aa"])
        total_reads = int(totals.loc[patient, "total_reads"])
        for seed in SEEDS:
            scope = f"test-outer{outer_map[patient]}-seed{seed}"
            for draw_id in range(DRAWS):
                rng = np.random.default_rng(stable_seed("draw-v1", patient, -1, draw_id, scope))
                pick = rng.choice(4096, size=4096, replace=False)[:BAG]
                seen = set()
                for pos in pick:
                    seen.update(member.get((patient, int(seq[pos])), ()))
                rows.append({
                    "patient_key": patient, "outer_fold": int(outer_map[patient]),
                    "seed": seed, "draw_index": draw_id,
                    "n_unique_aa_in_bag": BAG,
                    "full_unique_aa": total_unique,
                    "unique_aa_coverage": BAG / total_unique,
                    "reads_in_bag": int(reads[pick].sum()),
                    "full_reads": total_reads,
                    "read_coverage": float(reads[pick].sum() / total_reads),
                    "n_full_carried_groups": groups_full,
                    "n_captured_groups": len(seen),
                    "group_capture_fraction": (len(seen) / groups_full if groups_full else np.nan),
                })
    frame = pd.DataFrame(rows)
    if len(frame) != 126 * len(SEEDS) * DRAWS or frame.duplicated(
            ["patient_key", "seed", "draw_index"]).any():
        raise ValueError("Exact heldout bag capture table incomplete")
    summary = frame.groupby(["patient_key", "outer_fold", "seed"], as_index=False).agg(
        unique_aa_coverage=("unique_aa_coverage", "mean"),
        mean_read_coverage=("read_coverage", "mean"),
        sd_read_coverage=("read_coverage", "std"),
        mean_group_capture_fraction=("group_capture_fraction", "mean"),
        sd_group_capture_fraction=("group_capture_fraction", "std"),
        n_full_carried_groups=("n_full_carried_groups", "first"))
    write_tsv(BAG_OUT, frame)
    write_tsv(SUMMARY_OUT, summary)
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024**2
    if rss > 512:
        raise RuntimeError("Bag capture exceeded 512 GiB")
    receipt = {"status": "complete", "label_blind_for_groups": True,
               "n_patients": 126, "n_seeds": len(SEEDS), "n_draws_per_seed": DRAWS,
               "bag_size": BAG, "n_bag_rows": len(frame),
               "n_group_member_rows_in_shared_pool": source_rows,
               "median_bag_unique_aa_coverage": float(summary.unique_aa_coverage.median()),
               "median_bag_read_coverage": float(summary.mean_read_coverage.median()),
               "median_bag_group_capture_fraction": float(summary.mean_group_capture_fraction.median()),
               "scope": "same heldout draw seeds and scope as 08_compare_mil.py; full-Home groups used descriptively only",
               "source_sha256": source, "bag_output_sha256": sha(BAG_OUT),
               "summary_output_sha256": sha(SUMMARY_OUT),
               "peak_rss_gib": round(rss, 3),
               "seconds": round(time.monotonic() - started, 3)}
    tmp = RECEIPT.with_suffix(".json.partial")
    tmp.write_text(json.dumps(receipt, indent=2, sort_keys=True, allow_nan=False) + "\n")
    os.replace(tmp, RECEIPT)
    print(json.dumps(receipt, sort_keys=True))


if __name__ == "__main__":
    main()
