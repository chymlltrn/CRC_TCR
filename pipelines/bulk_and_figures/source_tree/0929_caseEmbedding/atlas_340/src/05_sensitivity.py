#!/usr/bin/env python3
"""Prespecified Home candidate robustness checks at common read depth.

The random rarefaction is keyed only by group and repeat, never by N. A failed
or unresolved sensitivity check cannot promote a group to a final candidate.
"""
from __future__ import annotations
from release_runtime import private_literal as _private_literal, release_path as _release_path

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(_release_path('path_99d8c1c6fde2a6b9'))
OUT = ROOT / "results"
COLUMNS = [
    "method", "group_id", "equal_depth_min_Nplus_minus_N0",
    "equal_depth_min_carriers", "equal_depth_all_three_positive",
    "batch_evaluable", "batch_positive", "batch_reversal",
    "untreated_beta_positive", "sensitivity_stable",
]
REPEAT_COLUMNS = ["method", "group_id", "repeat", "n_carriers",
                  "Nplus_carriers", "N0_carriers", "Nplus_minus_N0"]
SEED = 20260930


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path: Path, value: dict) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n")
    tmp.replace(path)


def write_tsv(path: Path, frame: pd.DataFrame, columns: list[str]) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    frame.reindex(columns=columns).to_csv(tmp, sep="\t", index=False,
                                           float_format="%.12g")
    tmp.replace(path)


def labels() -> pd.DataFrame:
    path = ROOT / "config/cohort_samples.tsv"
    with path.open(newline="") as f:
        rows = list(csv.DictReader(f, delimiter="\t"))
    frame = pd.DataFrame(rows)
    frame = frame.loc[frame.cohort.eq("home"),
                      ["patient_key", "recorded_n_binary", "neoadjuvant", "home_batch"]]
    if frame.groupby("patient_key").apply(lambda x: len(x.drop_duplicates()) != 1).any():
        raise ValueError("Clinical fields differ within same-patient Home libraries")
    frame = frame.drop_duplicates("patient_key").set_index("patient_key")
    frame = frame.loc[frame.recorded_n_binary.isin(("0", "1"))]
    if len(frame) != 126 or frame.recorded_n_binary.value_counts().to_dict() != {"0": 70, "1": 56}:
        raise ValueError("Unexpected N-known Home patient population")
    return frame


def run() -> dict:
    OUT.mkdir(exist_ok=True)
    inputs = {
        "candidate": ROOT / "groups/frozen_home_candidates.tsv",
        "candidate_receipt": ROOT / "groups/frozen_home_candidates.json",
        "association": ROOT / "groups/full_home/association_recorded_home.tsv",
        "burden": ROOT / "groups/full_home/patient_group_burden.parquet",
        "burden_receipt": ROOT / "groups/full_home/burden_receipt.json",
        "batch_report": ROOT / "atlas/batch_report.json",
        "manifest": ROOT / "config/cohort_samples.tsv",
        "folds": ROOT / "folds/folds.tsv",
    }
    hashes = {name: sha(path) for name, path in inputs.items()}
    candidate_receipt = json.loads(inputs["candidate_receipt"].read_text())
    burden_receipt = json.loads(inputs["burden_receipt"].read_text())
    batch = json.loads(inputs["batch_report"].read_text())
    if candidate_receipt.get("status") != "complete" or candidate_receipt.get("candidate_sha256") != hashes["candidate"] or (
        candidate_receipt.get("source_association_sha256") != hashes["association"]
    ):
        raise ValueError("Home candidate freeze receipt is incomplete or stale")
    if burden_receipt.get("status") != "complete" or burden_receipt.get("scope") != "full_home" or (
        burden_receipt.get("patient_group_burden_sha256") != hashes["burden"]
    ):
        raise ValueError("Home burden receipt is incomplete or stale")
    if batch.get("status") != "complete" or batch.get("outcome_blind") is not True:
        raise ValueError("Common-depth batch diagnostic is incomplete")
    depth = int(batch["equal_depth_reads"])
    candidates = pd.read_csv(inputs["candidate"], sep="\t", dtype={"group_id": str})
    association = pd.read_csv(inputs["association"], sep="\t", dtype={"group_id": str})
    if len(candidates) != candidate_receipt.get("n_candidates") or (
        candidates[["method", "group_id"]].duplicated().any()
    ):
        raise ValueError("Frozen candidate list malformed")
    burden = pd.read_parquet(inputs["burden"],
                             columns=["patient_key", "group_id", "presence",
                                      "group_reads", "total_reads"])
    burden["group_id"] = burden.group_id.astype(str)
    clinical = labels()
    if set(clinical.index) - set(burden.patient_key):
        raise ValueError("N-known Home patient missing from burden")
    if depth < 1 or (burden.total_reads < depth).any():
        raise ValueError("Common depth exceeds a Home patient's available reads")
    result_rows, repeat_rows = [], []
    for cand in candidates.itertuples(index=False):
        key = str(cand.group_id)
        sub = burden.loc[burden.group_id.eq(key)].set_index("patient_key").loc[clinical.index]
        if len(sub) != len(clinical) or sub.index.duplicated().any():
            raise ValueError("Candidate burden patient grid incomplete: " + key)
        expected = association.loc[association.group_id.eq(key) &
                                   association.method.eq(cand.method)]
        if len(expected) != 1 or not bool(expected.iloc[0].home_association_candidate):
            raise ValueError("Candidate absent from Home association: " + key)
        y = clinical.recorded_n_binary.to_numpy()
        counts = sub.group_reads.to_numpy(dtype=np.int64)
        totals = sub.total_reads.to_numpy(dtype=np.int64)
        if (counts < 0).any() or (counts > totals).any() or (
            (counts > 0) != sub.presence.to_numpy(dtype=bool)
        ).any():
            raise ValueError("Group read and presence counts disagree: " + key)
        diffs, carriers = [], []
        for repeat in range(3):
            digest = hashlib.sha256(f"{SEED}|{cand.method}|{key}|{repeat}".encode()).digest()
            rng = np.random.default_rng(int.from_bytes(digest[:8], "little"))
            present = rng.hypergeometric(counts, totals-counts, depth).astype(bool)
            plus, zero = int(present[y == "1"].sum()), int(present[y == "0"].sum())
            delta = plus / 56 - zero / 70
            diffs.append(delta)
            carriers.append(plus+zero)
            repeat_rows.append({"method": cand.method, "group_id": key,
                                "repeat": repeat, "n_carriers": plus+zero,
                                "Nplus_carriers": plus, "N0_carriers": zero,
                                "Nplus_minus_N0": delta})
        batch_evaluable = batch_positive = batch_reversal = 0
        raw_presence = sub.presence.to_numpy(dtype=bool)
        for home_batch in ("H2498B", "H2498C", "H2498D"):
            in_batch = clinical.home_batch.to_numpy() == home_batch
            plus = in_batch & (y == "1")
            zero = in_batch & (y == "0")
            if plus.sum() >= 5 and zero.sum() >= 5:
                batch_evaluable += 1
                delta = raw_presence[plus].mean() - raw_presence[zero].mean()
                batch_positive += int(delta > 0)
                batch_reversal += int(delta < 0)
        row = expected.iloc[0]
        untreated_ok = row.untreated_status == "ok" and np.isfinite(row.untreated_beta) and row.untreated_beta > 0
        depth_ok = min(carriers) >= 5 and all(delta > 0 for delta in diffs)
        batch_ok = batch_evaluable >= 2 and batch_positive >= 2 and batch_reversal == 0
        result_rows.append({
            "method": cand.method, "group_id": key,
            "equal_depth_min_Nplus_minus_N0": min(diffs),
            "equal_depth_min_carriers": min(carriers),
            "equal_depth_all_three_positive": bool(depth_ok),
            "batch_evaluable": batch_evaluable,
            "batch_positive": batch_positive,
            "batch_reversal": batch_reversal,
            "untreated_beta_positive": bool(untreated_ok),
            "sensitivity_stable": bool(depth_ok and batch_ok and untreated_ok),
        })
    table = pd.DataFrame(result_rows, columns=COLUMNS)
    repeats = pd.DataFrame(repeat_rows, columns=REPEAT_COLUMNS)
    out_path = OUT / "05_candidate_sensitivity.tsv"
    repeats_path = OUT / "05_equal_depth_repeats.tsv"
    write_tsv(out_path, table, COLUMNS)
    write_tsv(repeats_path, repeats, REPEAT_COLUMNS)
    receipt = {
        "status": "complete", "endpoint": "recorded N+",
        "n_candidates": len(table), "n_stable": int(table.sensitivity_stable.sum()) if len(table) else 0,
        "common_depth_reads": depth, "repeat_seed": SEED, "n_repeats": 3,
        "source_sha256": hashes,
        "candidate_sensitivity_sha256": sha(out_path),
        "equal_depth_repeats_sha256": sha(repeats_path),
        "rule": "three label-blind read-level hypergeometric subsamples each retain >=5 carriers and positive N+ minus N0 prevalence; >=2 Home batches each have >=5 patients/class and positive direction with no evaluated reversal; untreated adjusted beta positive with converged fit",
        "interpretation": "Sensitivity gate for candidate designation, not a new discovery test",
    }
    write_json(OUT / "05_sensitivity_receipt.json", receipt)
    return receipt


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", action="store_true")
    parser.parse_args()
    print(json.dumps(run(), ensure_ascii=False))
