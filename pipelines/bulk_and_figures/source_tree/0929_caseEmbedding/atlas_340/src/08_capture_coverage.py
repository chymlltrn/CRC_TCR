#!/usr/bin/env python3
"""Label-blind coverage of frozen full-Home groups by the representation screen draw."""
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
DRAW = ROOT / "representation_comparison/home_sample/patient_aa_draw.parquet"
TOTALS = ROOT / "representation_comparison/home_sample/patient_totals.tsv"
GROUP_AA = ROOT / "groups/full_home/patient_group_aa_safe.parquet"
BURDEN = ROOT / "groups/full_home/patient_group_burden.parquet"
OUT = ROOT / "representation_comparison/coverage"
PATIENT_OUT = OUT / "patient_group_capture.tsv"
GROUP_OUT = OUT / "group_capture.tsv"
RECEIPT = OUT / "receipt.json"


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write_tsv(path, frame):
    tmp = path.with_suffix(path.suffix + ".partial")
    frame.to_csv(tmp, sep="\t", index=False, float_format="%.12g")
    os.replace(tmp, path)


def main():
    start = time.monotonic()
    OUT.mkdir(parents=True, exist_ok=True)
    source = {"draw": sha(DRAW), "patient_totals": sha(TOTALS),
              "full_home_group_aa": sha(GROUP_AA), "full_home_burden": sha(BURDEN),
              "script": sha(Path(__file__))}
    if RECEIPT.exists():
        old = json.loads(RECEIPT.read_text())
        if old.get("status") == "complete" and old.get("source_sha256") == source and (
            old.get("patient_output_sha256") == sha(PATIENT_OUT)
        ) and old.get("group_output_sha256") == sha(GROUP_OUT):
            print(json.dumps(old, sort_keys=True))
            return
        raise RuntimeError("Coverage receipt invalid")
    draw = pq.read_table(DRAW, columns=["patient_key", "seq_idx"]).to_pandas()
    if len(draw) != 137 * 4096 or draw.duplicated(["patient_key", "seq_idx"]).any():
        raise ValueError("Screen draw patient-AA rows incorrect")
    burden = pq.read_table(BURDEN, columns=["patient_key", "group_id", "presence"]).to_pandas()
    carrier = burden.loc[burden.presence.eq(1), ["patient_key", "group_id"]].drop_duplicates()
    if carrier.duplicated(["patient_key", "group_id"]).any():
        raise ValueError("Full Home carrier table malformed")
    captured = []
    part = pq.ParquetFile(GROUP_AA)
    for batch in part.iter_batches(batch_size=750_000, columns=["patient_key", "seq_idx", "group_id"]):
        groups = batch.to_pandas()
        matched = groups.merge(draw, on=["patient_key", "seq_idx"], how="inner",
                               validate="many_to_one")
        if len(matched):
            captured.append(matched[["patient_key", "group_id"]].drop_duplicates())
    seen = pd.concat(captured, ignore_index=True).drop_duplicates() if captured else (
        carrier.iloc[0:0].copy())
    if len(seen.merge(carrier, on=["patient_key", "group_id"], how="left",
                      indicator=True).query("_merge != 'both'")):
        raise ValueError("Captured group carrier absent from full burden")
    counts = carrier.groupby("patient_key").size().rename("n_full_carried_groups")
    hits = seen.groupby("patient_key").size().rename("n_draw_captured_groups")
    totals = pd.read_csv(TOTALS, sep="\t", dtype={"patient_key": str}).set_index("patient_key")
    patients = totals.join(counts, how="left").join(hits, how="left").fillna({
        "n_full_carried_groups": 0, "n_draw_captured_groups": 0})
    for key in ("n_full_carried_groups", "n_draw_captured_groups"):
        patients[key] = patients[key].astype(int)
    patients["group_capture_fraction"] = np.where(
        patients.n_full_carried_groups.gt(0),
        patients.n_draw_captured_groups / patients.n_full_carried_groups, np.nan)
    patient_frame = patients.reset_index()
    group_count = carrier.groupby("group_id").size().rename("n_full_carrier_patients")
    group_hit = seen.groupby("group_id").size().rename("n_draw_captured_patients")
    groups = pd.concat((group_count, group_hit), axis=1).fillna(0).astype(int)
    groups["patient_capture_fraction"] = (groups.n_draw_captured_patients /
                                          groups.n_full_carrier_patients)
    group_frame = groups.reset_index()
    write_tsv(PATIENT_OUT, patient_frame)
    write_tsv(GROUP_OUT, group_frame)
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024**2
    if rss > 512:
        raise RuntimeError("Coverage job exceeded 512 GiB RSS")
    receipt = {"status": "complete", "label_blind": True, "purpose": "sample capture of full-Home descriptive group carriers only; not predictive fold features",
               "n_patients": len(patient_frame), "n_groups_with_any_carrier": len(group_frame),
               "n_full_carrier_pairs": len(carrier), "n_captured_carrier_pairs": len(seen),
               "median_aa_coverage": float(patient_frame.aa_coverage.median()),
               "median_read_coverage": float(patient_frame.read_coverage.median()),
               "median_group_capture_fraction": float(patient_frame.group_capture_fraction.median()),
               "source_sha256": source,
               "patient_output_sha256": sha(PATIENT_OUT),
               "group_output_sha256": sha(GROUP_OUT),
               "seconds": round(time.monotonic() - start, 3),
               "peak_rss_gib": round(rss, 3)}
    tmp = RECEIPT.with_suffix(".json.partial")
    tmp.write_text(json.dumps(receipt, indent=2, sort_keys=True, allow_nan=False) + "\n")
    os.replace(tmp, RECEIPT)
    print(json.dumps(receipt, sort_keys=True))


if __name__ == "__main__":
    main()
