#!/usr/bin/env python3
"""Fold-scoped tcrdist3 seed graph and exhaustive frozen projection.

All graph seeds come from Home training patients; held-out Home and external
AA/V/J tuples are projected onto the frozen graph without reading labels.
A full-scope receipt requires an exact-neighbor ANN recall pilot and an
evaluated assignment or explicit unassigned status for every eligible tuple.
"""
from __future__ import annotations
from release_runtime import private_literal as _private_literal, release_path as _release_path

import argparse
import csv
import hashlib
import json
import os
import resource
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

for _e in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_e] = "1"

import numpy as np
import pandas as pd
import psutil
from annoy import AnnoyIndex
from numba import njit
from pwseqdist.matrices import seqs2mat, parasail_aa_alphabet
from pwseqdist.metrics import nb_tcrdist
from tcrdist.repertoire import TCRrep
from tcrdist.repertoire_db import RefGeneSet
import tcrdist

KEY = ["aa", "v_gene", "j_gene"]
MAX_RSS = 128 * 1024**3
REFERENCE_DB = "combo_xcr_2024-03-05.tsv"
PRIMARY_ANN_MODEL = "tcrbert_dualtask_aa_mean"
RADIUS_GRID = (16, 20, 24)
PROJECTION_COST_FACTOR = 1.25  # bounded pilot omits full-output serialization
ANN_QUERY_WORKERS = 8  # 2k-query benchmark: 1/4/8 threads gave identical candidates
AA_CODE = {aa: code for code, aa in enumerate(parasail_aa_alphabet)}


def known_human_v() -> set[str]:
    db = RefGeneSet(db_file=REFERENCE_DB)
    return {g for g in db.all_genes["human"] if g.startswith("TRBV")}


def reference_hash() -> str:
    return sha(Path(tcrdist.__file__).parent / "db" / REFERENCE_DB)


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def receipt(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, sort_keys=True, ensure_ascii=False) + "\n")
    tmp.replace(path)


class GateStop(RuntimeError):
    """A declared scientific or resource gate stopped this method."""


def gate(ok: bool, why: str) -> None:
    if not ok:
        raise GateStop(why)


def check(ok: bool, why: str) -> None:
    if not ok:
        raise ValueError(why)


def memcheck() -> int:
    rss = psutil.Process().memory_info().rss
    if rss > MAX_RSS:
        raise MemoryError("tcrdist projection exceeded 128 GiB RSS")
    return rss


def manifest_fingerprints(manifest_path: Path) -> tuple[str, str]:
    """Canonical sample identities and Home split labels; never use external N."""
    identities: list[list[str]] = []
    home_labels: list[list[str]] = []
    with manifest_path.open(newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        required = {"cohort", "sample_id", "patient_key", "recorded_n_binary"}
        check(required <= set(reader.fieldnames or []),
              "cohort manifest missing identity or Home split columns")
        for row in reader:
            cohort = row["cohort"].lower()
            identities.append([cohort, row["sample_id"], row["patient_key"]])
            if cohort == "home":
                home_labels.append([row["patient_key"], row["recorded_n_binary"]])
    identity = pd.DataFrame(identities,
                            columns=["cohort", "sample_id", "patient_key"])
    identity = identity.sort_values(["cohort", "sample_id", "patient_key"])
    identity_sha = hashlib.sha256(identity.to_csv(
        sep="\t", index=False, lineterminator="\n").encode("utf-8")).hexdigest()
    label_sha = hashlib.sha256(json.dumps(
        sorted(home_labels), ensure_ascii=False,
        separators=(",", ":")).encode("utf-8")).hexdigest()
    return identity_sha, label_sha


def scope_ids(root: Path, scope: str) -> list[str]:
    f = pd.read_csv(root / "folds/folds.tsv", sep="\t",
                    usecols=["patient_key", "outer_fold", "partition", "inner_fold"],
                    dtype={"patient_key": str})
    if scope == "full_home":
        ids = set(f.patient_key)
    elif scope.startswith("outer_") and scope[6:].isdigit():
        k = int(scope[6:])
        check(k in range(5), "outer fold out of range")
        ids = set(f.loc[f.outer_fold.eq(k) & f.partition.eq("train"), "patient_key"])
    elif scope.startswith("inner_"):
        parts = scope.split("_")
        check(len(parts) == 3 and parts[1].isdigit() and parts[2].isdigit(),
              "bad inner scope")
        outer, inner = map(int, parts[1:])
        check(outer in range(5) and inner in range(3), "inner fold out of range")
        ids = set(f.loc[f.outer_fold.eq(outer) & f.partition.eq("train") &
                        f.inner_fold.ne(inner), "patient_key"])
    else:
        raise ValueError(f"bad scope {scope}")
    check(bool(ids), "no training patients")
    return sorted(ids)

def ids_hash(ids: list[str]) -> str:
    return hashlib.sha256(("\n".join(sorted(ids)) + "\n").encode()).hexdigest()


def prepare(root: Path) -> Path:
    """Unique global AA/V/J projection universe, label blind."""
    manifest_path = root / "config/cohort_samples.tsv"
    union_path = root / "data/aa_union.parquet"
    identity_sha, home_label_sha = manifest_fingerprints(manifest_path)
    m = pd.read_csv(manifest_path, sep="\t", dtype=str,
                    usecols=["cohort", "sample_id", "patient_key"])
    check(len(m) == 279 and set(m.cohort) == {"home", "external"}, "279-sample manifest not ready")
    check(union_path.exists(), "AA union not ready")
    out = root / "groups/tcrdist_all_nodes.parquet"
    rpath = root / "groups/tcrdist_all_nodes_receipt.json"
    malformed_path = root / "groups/tcrdist_malformed_vj_clone_rows.parquet"
    coverage_path = root / "groups/tcrdist_patient_vj_coverage.parquet"
    if out.exists() and rpath.exists():
        old = json.loads(rpath.read_text())
        if (old.get("status") == "complete" and old.get("schema_version") == 3 and
            malformed_path.exists() and coverage_path.exists() and
            old.get("patient_vj_coverage_sha256") == sha(coverage_path) and
            old.get("malformed_vj_rows_sha256") == sha(malformed_path) and
            old.get("aa_union_sha256") == sha(union_path) and
            old.get("manifest_identity_sha256") == identity_sha and
            old.get("home_split_label_sha256") == home_label_sha and
            old.get("output_sha256") == sha(out)):
            return out
    import pyarrow as pa
    import pyarrow.parquet as pq
    malformed_path.parent.mkdir(parents=True, exist_ok=True)
    malformed_tmp = malformed_path.with_suffix(".parquet.tmp")
    malformed_schema = pa.schema([
        pa.field("cohort", pa.string()), pa.field("sample_id", pa.string()),
        pa.field("patient_key", pa.string()), pa.field("clone_id", pa.int64()),
        pa.field("aa", pa.string()), pa.field("v_gene", pa.string()),
        pa.field("j_gene", pa.string()), pa.field("read_count", pa.int64()),
    ])
    malformed_writer = pq.ParquetWriter(str(malformed_tmp), malformed_schema,
                                        compression="zstd")
    seen: dict[tuple[str, str, str], int] = {}
    known = known_human_v()
    patient_coverage: dict[tuple[str, str], dict[str, int]] = {}
    qc_rows = bad_vj = qc_reads = unscorable_reads = unscorable_rows = 0
    malformed_vj_rows = malformed_vj_reads = 0
    unscorable_home_patients: set[str] = set()
    unscorable_external_patients: set[str] = set()
    malformed_home_patients: set[str] = set()
    malformed_external_patients: set[str] = set()
    t0 = time.monotonic()
    for n, row in enumerate(m.itertuples(index=False), 1):
        path = root / "data/clone_detail" / f"{row.cohort}__{row.sample_id}.parquet"
        check(path.exists(), f"missing {path}")
        d = pd.read_parquet(path, columns=["cohort", "sample_id", "patient_key", "clone_id"] + KEY + ["qc_pass", "read_count"])
        d = d.loc[d.qc_pass == True]
        qc_rows += len(d)
        qc_reads += int(d.read_count.sum())
        eligible_vj = d.v_gene.astype(str).str.startswith("TRBV") & d.j_gene.astype(str).str.startswith("TRBJ")
        malformed = ~eligible_vj
        malformed_vj_rows += int(malformed.sum())
        malformed_vj_reads += int(d.loc[malformed, "read_count"].sum())
        if malformed.any():
            target = malformed_home_patients if row.cohort == "home" else malformed_external_patients
            target.update(d.loc[malformed, "patient_key"].astype(str))
            malformed_columns = ["cohort", "sample_id", "patient_key", "clone_id"] + KEY + ["read_count"]
            malformed_writer.write_table(pa.Table.from_pandas(
                d.loc[malformed, malformed_columns], schema=malformed_schema,
                preserve_index=False))
        unsupported = eligible_vj & ~d.v_gene.astype(str).map(normal_gene).isin(known)
        unscorable_rows += int(unsupported.sum())
        unscorable_reads += int(d.loc[unsupported, "read_count"].sum())
        if unsupported.any():
            target = unscorable_home_patients if row.cohort == "home" else unscorable_external_patients
            target.update(d.loc[unsupported, "patient_key"].astype(str))
        # Patient-level source QC lets each fold gate ignore held-out burden.
        for patient_key, frame in d.groupby("patient_key", sort=False):
            idx = frame.index
            key = (str(row.cohort), str(patient_key))
            totals = patient_coverage.setdefault(key, {
                "qc_clone_rows": 0, "qc_reads": 0,
                "unsupported_v_clone_rows": 0, "unsupported_v_reads": 0,
                "malformed_vj_clone_rows": 0, "malformed_vj_reads": 0})
            totals["qc_clone_rows"] += len(frame)
            totals["qc_reads"] += int(frame.read_count.sum())
            totals["unsupported_v_clone_rows"] += int(unsupported.loc[idx].sum())
            totals["unsupported_v_reads"] += int(frame.loc[unsupported.loc[idx], "read_count"].sum())
            totals["malformed_vj_clone_rows"] += int(malformed.loc[idx].sum())
            totals["malformed_vj_reads"] += int(frame.loc[malformed.loc[idx], "read_count"].sum())
        d = d[KEY].drop_duplicates(KEY)
        bit = 1 if row.cohort == "home" else 2
        for aa, v, j in d.itertuples(index=False, name=None):
            if not isinstance(v, str) or not isinstance(j, str) or not v.startswith("TRBV") or not j.startswith("TRBJ"):
                bad_vj += 1
                continue
            key = (aa, v, j)
            seen[key] = seen.get(key, 0) | bit
        if n % 10 == 0:
            print(json.dumps({"prepared_samples": n, "nodes": len(seen), "rss_gib": round(memcheck()/1024**3, 2)}), flush=True)
    malformed_writer.close()
    check((identity_sha, home_label_sha) == manifest_fingerprints(manifest_path),
          "cohort identities or Home split labels changed during node preparation")
    malformed_tmp.replace(malformed_path)
    coverage_rows = [
        {"cohort": cohort, "patient_key": patient_key, **totals}
        for (cohort, patient_key), totals in sorted(patient_coverage.items())]
    coverage = pd.DataFrame.from_records(coverage_rows)
    check(bool(len(coverage)), "no patient-level V/J coverage")
    check(int(coverage.qc_clone_rows.sum()) == qc_rows and
          int(coverage.qc_reads.sum()) == qc_reads and
          int(coverage.unsupported_v_clone_rows.sum()) == unscorable_rows and
          int(coverage.unsupported_v_reads.sum()) == unscorable_reads and
          int(coverage.malformed_vj_clone_rows.sum()) == malformed_vj_rows and
          int(coverage.malformed_vj_reads.sum()) == malformed_vj_reads,
          "patient-level V/J QC totals do not conserve source rows/reads")
    coverage_tmp = coverage_path.with_suffix(".parquet.tmp")
    coverage.to_parquet(coverage_tmp, index=False, compression="zstd")
    coverage_tmp.replace(coverage_path)
    keys = sorted(seen)
    nodes = pd.DataFrame.from_records(keys, columns=KEY)
    nodes["cohort_mask"] = np.fromiter((seen[k] for k in keys), dtype=np.uint8)
    del seen, keys
    aa = pd.read_parquet(union_path, columns=["aa", "seq_idx"])
    check(not aa.aa.duplicated().any(), "AA union duplicated")
    nodes = nodes.merge(aa, on="aa", how="left", validate="many_to_one")
    check(nodes.seq_idx.notna().all(), "projection nodes missing from AA union")
    nodes.seq_idx = nodes.seq_idx.astype(np.int64)
    check(not nodes.duplicated(KEY).any(), "projection nodes duplicated")
    tmp = out.with_suffix(".parquet.tmp")
    nodes.to_parquet(tmp, index=False, compression="zstd")
    tmp.replace(out)
    receipt(rpath, {"status": "complete", "schema_version": 3, "label_blind": True, "samples": len(m),
                    "patient_vj_coverage_sha256": sha(coverage_path),
                    "qc_clone_rows": qc_rows, "qc_total_reads": qc_reads,
                    "malformed_vj_clone_rows": malformed_vj_rows,
                    "malformed_vj_rows_sha256": sha(malformed_path),
                    "malformed_vj_reads": malformed_vj_reads,
                    "malformed_vj_read_fraction": malformed_vj_reads/qc_reads if qc_reads else None,
                    "malformed_vj_patient_carriers_home": len(malformed_home_patients),
                    "malformed_vj_patient_carriers_external": len(malformed_external_patients),
                    "unscorable_v_clone_rows": unscorable_rows,
                    "unscorable_v_reads": unscorable_reads,
                    "unscorable_v_read_fraction": unscorable_reads/qc_reads if qc_reads else None,
                    "unscorable_v_patient_carriers_home": len(unscorable_home_patients),
                    "unscorable_v_patient_carriers_external": len(unscorable_external_patients),
                    "unusable_vj_distinct_per_sample_rows": bad_vj,
                    "source_node_universe": "all globally distinct QC AA x TRBV x TRBJ tuples; missing or malformed V/J retained only in ingest records",
                    "unscorable_read_fraction_denominator": "all QC readCount, including malformed V/J",
                    "tcrdist_reference_db_sha256": reference_hash(),
                    "eligible_global_nodes": len(nodes),
                    "home_observed_nodes": int(((nodes.cohort_mask & 1) != 0).sum()),
                    "external_observed_nodes": int(((nodes.cohort_mask & 2) != 0).sum()),
                    "aa_union_sha256": sha(union_path),
                    "manifest_identity_sha256": identity_sha,
                    "home_split_label_sha256": home_label_sha,
                    "raw_manifest_sha256_at_prepare_audit_only": sha(manifest_path),
                    "output_sha256": sha(out), "seconds": time.monotonic()-t0,
                    "peak_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss})
    return out


def reference_coverage(root: Path) -> Path:
    """List every QC AA/V/J node outside the frozen human V reference."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    node_path = root / "groups/tcrdist_all_nodes.parquet"
    prep_path = root / "groups/tcrdist_all_nodes_receipt.json"
    out = root / "groups/tcrdist_unscorable_v_nodes.parquet"
    report_path = root / "groups/tcrdist_v_reference_coverage.json"
    check(node_path.exists() and prep_path.exists(), "global nodes absent")
    prep = json.loads(prep_path.read_text())
    identity_sha, home_label_sha = manifest_fingerprints(
        root / "config/cohort_samples.tsv")
    check(prep.get("status") == "complete" and
          prep.get("output_sha256") == sha(node_path) and
          prep.get("manifest_identity_sha256") == identity_sha and
          prep.get("home_split_label_sha256") == home_label_sha,
          "global node source or canonical manifest changed")
    source_sha = sha(node_path)
    if out.exists() and report_path.exists():
        old = json.loads(report_path.read_text())
        if (old.get("status") == "complete" and
            old.get("all_nodes_sha256") == source_sha and
            old.get("unscorable_v_nodes_sha256") == sha(out) and
            old.get("manifest_identity_sha256") == identity_sha and
            old.get("home_split_label_sha256") == home_label_sha):
            return report_path
    source = pq.ParquetFile(node_path)
    known = known_human_v()
    tmp = out.with_suffix(".parquet.tmp")
    writer = pq.ParquetWriter(str(tmp), source.schema_arrow, compression="zstd")
    total = unscorable = home_total = home_bad = external_total = external_bad = 0
    gene_counts = Counter()
    t0 = time.monotonic()
    try:
        for batch in source.iter_batches(batch_size=250000):
            frame = batch.to_pandas()
            bad = ~frame.v_gene.astype(str).map(normal_gene).isin(known)
            mask = frame.cohort_mask.to_numpy(dtype=np.uint8)
            total += len(frame)
            unscorable += int(bad.sum())
            home_total += int(((mask & 1) != 0).sum())
            external_total += int(((mask & 2) != 0).sum())
            home_bad += int((bad.to_numpy() & ((mask & 1) != 0)).sum())
            external_bad += int((bad.to_numpy() & ((mask & 2) != 0)).sum())
            if bad.any():
                gene_counts.update(frame.loc[bad, "v_gene"].astype(str))
                writer.write_table(pa.Table.from_pandas(
                    frame.loc[bad], schema=source.schema_arrow,
                    preserve_index=False))
            memcheck()
    finally:
        writer.close()
    check(total == int(prep["eligible_global_nodes"]) and
          total == source.metadata.num_rows,
          "global V coverage scan did not conserve node count")
    tmp.replace(out)
    receipt(report_path, {
        "status": "complete", "method": "tcrdist3", "label_blind": True,
        "source_nodes_total": total,
        "scorable_nodes_total": total - unscorable,
        "n_unscorable_v_nodes": unscorable,
        "unscorable_v_node_fraction": unscorable / total,
        "home_observed_nodes": home_total,
        "home_unscorable_v_nodes": home_bad,
        "home_unscorable_v_node_fraction": home_bad / home_total,
        "external_observed_nodes": external_total,
        "external_unscorable_v_nodes": external_bad,
        "external_unscorable_v_node_fraction": external_bad / external_total,
        "top_unscorable_v_genes": dict(gene_counts.most_common(20)),
        "unscorable_v_clone_rows": prep["unscorable_v_clone_rows"],
        "unscorable_v_reads": prep["unscorable_v_reads"],
        "unscorable_v_read_fraction": prep["unscorable_v_read_fraction"],
        "unscorable_v_patient_carriers_home": prep["unscorable_v_patient_carriers_home"],
        "unscorable_v_patient_carriers_external": prep["unscorable_v_patient_carriers_external"],
        "all_nodes_sha256": source_sha,
        "unscorable_v_nodes_sha256": sha(out),
        "manifest_identity_sha256": identity_sha,
        "home_split_label_sha256": home_label_sha,
        "script_sha256": sha(Path(__file__)),
        "seconds": time.monotonic()-t0,
        "peak_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss})
    return report_path


def training_scope_quality(root: Path, scope: str, ids: list[str],
                           known: set[str]) -> dict:
    """Exclusion gate from this dictionary's Home training patients only.

    The frozen projection universe includes held-out and external tuples, but
    neither may decide whether a training dictionary passes a scientific gate.
    """
    import pyarrow.parquet as pq

    out = root / "groups" / scope
    description_path = out / "description.json"
    clone_path = out / "clone_nodes.parquet"
    coverage_path = root / "groups/tcrdist_patient_vj_coverage.parquet"
    prep_path = root / "groups/tcrdist_all_nodes_receipt.json"
    check(description_path.exists() and clone_path.exists(),
          f"training clone_nodes/description absent for {scope}")
    check(coverage_path.exists() and prep_path.exists(),
          "patient-level source V/J coverage absent; rerun tcrdist prepare")
    desc = json.loads(description_path.read_text())
    prep = json.loads(prep_path.read_text())
    identity_sha, home_label_sha = manifest_fingerprints(
        root / "config/cohort_samples.tsv")
    check(desc.get("scope") == scope and
          desc.get("manifest_identity_sha256") == identity_sha and
          desc.get("home_split_label_sha256") == home_label_sha and
          desc.get("train_patient_sha256") == ids_hash(ids) and
          set(map(str, desc.get("patient_ids", []))) == set(ids) and
          desc.get("folds_sha256") == sha(root / "folds/folds.tsv") and
          desc.get("clone_nodes_sha256") == sha(clone_path),
          "training clone_nodes scope/patient/folds/source hash mismatch")
    check(prep.get("schema_version") == 3 and
          prep.get("status") == "complete" and
          prep.get("manifest_identity_sha256") == identity_sha and
          prep.get("home_split_label_sha256") == home_label_sha and
          prep.get("patient_vj_coverage_sha256") == sha(coverage_path),
          "patient-level V/J source receipt mismatch")
    n_nodes = n_eligible = n_bad_vj = n_unscorable = 0
    total_node_reads = 0
    source = pq.ParquetFile(clone_path)
    for batch in source.iter_batches(batch_size=100000,
                                     columns=["v_gene", "j_gene", "total_reads"]):
        frame = batch.to_pandas()
        v = frame.v_gene.astype(str)
        j = frame.j_gene.astype(str)
        eligible = v.str.startswith("TRBV") & j.str.startswith("TRBJ")
        unsupported = eligible & ~v.map(normal_gene).isin(known)
        n_nodes += len(frame)
        n_eligible += int(eligible.sum())
        n_bad_vj += int((~eligible).sum())
        n_unscorable += int(unsupported.sum())
        total_node_reads += int(frame.total_reads.sum())
        memcheck()
    check(n_nodes == int(desc["n_unique_aa_vj"]), "training clone node count mismatch")
    cov = pd.read_parquet(coverage_path)
    cov = cov.loc[cov.cohort.eq("home") &
                  cov.patient_key.astype(str).isin(ids)].copy()
    check(len(cov) == len(ids) and set(cov.patient_key.astype(str)) == set(ids) and
          not cov.patient_key.duplicated().any(),
          "training patient V/J coverage incomplete or duplicated")
    totals = {col: int(cov[col].sum()) for col in (
        "qc_clone_rows", "qc_reads", "unsupported_v_clone_rows",
        "unsupported_v_reads", "malformed_vj_clone_rows",
        "malformed_vj_reads")}
    check(total_node_reads == totals["qc_reads"],
          "training QC reads differ between clone_nodes and source V/J coverage")
    check(n_eligible > 0 and totals["qc_reads"] > 0,
          "no eligible training V/J nodes or QC reads")
    return {
        "population": "Home patients used to discover this exact scope dictionary",
        "train_patient_sha256": ids_hash(ids),
        "manifest_identity_sha256": identity_sha,
        "home_split_label_sha256": home_label_sha,
        "training_patients": len(ids),
        "training_clone_nodes_total": n_nodes,
        "training_eligible_vj_nodes": n_eligible,
        "training_malformed_vj_nodes": n_bad_vj,
        "training_unscorable_v_nodes": n_unscorable,
        "training_unscorable_v_node_fraction": n_unscorable/n_eligible,
        "training_excluded_vj_read_fraction": (
            totals["unsupported_v_reads"] + totals["malformed_vj_reads"]
        )/totals["qc_reads"],
        "training_source_qc": totals,
        "clone_nodes_sha256": sha(clone_path),
        "patient_vj_coverage_sha256": sha(coverage_path),
    }


def balanced_seeds(root: Path, scope: str, per_patient: int, seed: int) -> tuple[pd.DataFrame, list[str]]:
    ids = scope_ids(root, scope)
    check(per_patient >= 16, "per_patient must be >=16")
    path = root / "groups" / scope / "patient_nodes.parquet"
    check(path.exists(), f"prepare-nodes missing for {scope}")
    desc = json.loads((root / "groups" / scope / "description.json").read_text())
    identity_sha, home_label_sha = manifest_fingerprints(
        root / "config/cohort_samples.tsv")
    check(desc.get("scope") == scope and
          desc.get("train_patient_sha256") == ids_hash(ids) and
          desc.get("patient_nodes_sha256") == sha(path) and
          desc.get("folds_sha256") == sha(root / "folds/folds.tsv") and
          desc.get("manifest_identity_sha256") == identity_sha and
          desc.get("home_split_label_sha256") == home_label_sha,
          "training patient set or patient_nodes/folds source hash mismatch")
    d = pd.read_parquet(path, columns=["patient_key"] + KEY + ["seq_idx", "read_count"])
    d.patient_key = d.patient_key.astype(str)
    check(set(d.patient_key) == set(ids), "patient_nodes scope mismatch")
    known = known_human_v()
    d = d.loc[d.v_gene.astype(str).map(normal_gene).isin(known) &
              d.j_gene.astype(str).str.startswith("TRBJ")].drop_duplicates(["patient_key"]+KEY)
    check(set(d.patient_key) == set(ids),
          f"{len(set(ids)-set(d.patient_key))} training patients lack reference-scorable V/J nodes")
    pieces = []
    for pid, frame in d.groupby("patient_key", sort=True):
        key = int.from_bytes(hashlib.sha256(f"{seed}:{pid}".encode()).digest()[:8], "little")
        rng = np.random.default_rng(key)
        ix = rng.choice(len(frame), size=min(per_patient, len(frame)), replace=False)
        pieces.append(frame.iloc[np.sort(ix)])
    sampled = pd.concat(pieces, ignore_index=True)
    seeds = sampled.groupby(KEY+["seq_idx"], as_index=False, sort=True).agg(
        seed_patient_count=("patient_key", "nunique"), seed_total_reads=("read_count", "sum"))
    check(not seeds.duplicated(KEY).any(), "duplicate seed tuple")
    return seeds, ids


def normal_gene(g: str) -> str:
    return g if "*" in g else g + "*01"


def v_penalties(genes: list[str]) -> tuple[np.ndarray, dict[str, int]]:
    """tcrdist3 human germline CDR1/2/2.5 reference, default beta weights."""
    genes = sorted({normal_gene(g) for g in genes})
    query = pd.DataFrame({"cdr3_b_aa": [_private_literal('private_8b5efd300a8432fa')]*len(genes),
                          "v_b_gene": [normal_gene(x) for x in genes],
                          "j_b_gene": ["TRBJ2-5*01"]*len(genes),
                          "count": [1]*len(genes)})
    rep = TCRrep(cell_df=query, organism="human", chains=["beta"],
                 compute_distances=False, cpus=1)
    loops = {str(r.v_b_gene): (str(r.cdr1_b_aa), str(r.cdr2_b_aa), str(r.pmhc_b_aa))
             for r in rep.clone_df.itertuples(index=False)}
    check(all(normal_gene(x) in loops for x in genes), "unrecognized TRBV allele in tcrdist3 reference")
    ids = {x: i for i, x in enumerate(genes)}
    vcost = np.zeros((len(genes), len(genes)), dtype=np.int16)
    encoded = {g: [seqs2mat([s])[0][0, :len(s)] for s in loops[normal_gene(g)]] for g in genes}
    for i, a in enumerate(genes):
        for j in range(i+1, len(genes)):
            b = genes[j]
            v = sum(int(nb_tcrdist(x, y, dist_weight=1, gap_penalty=4,
                                    ntrim=0, ctrim=0, fixed_gappos=True))
                    for x, y in zip(encoded[a], encoded[b]))
            vcost[i,j] = vcost[j,i] = v
    return vcost, ids


@njit(cache=False)
def selected(qmat, qlen, qv, smat, slen, sv, vcost, qi, si):
    out = np.empty(len(qi), dtype=np.int32)
    for p in range(len(qi)):
        i, j = qi[p], si[p]
        d = nb_tcrdist(qmat[i, :qlen[i]], smat[j, :slen[j]],
                       dist_weight=1, gap_penalty=4, ntrim=3, ctrim=2,
                       fixed_gappos=False)
        out[p] = 3*d + vcost[qv[i], sv[j]]
    return out


def encode(d: pd.DataFrame, ids: dict[str, int]):
    sequences = d.aa.tolist()
    lengths = np.asarray([len(x) for x in sequences], dtype=np.int32)
    mat = np.full((len(sequences), int(lengths.max(initial=0))), -1, dtype=np.int8)
    for i, sequence in enumerate(sequences):
        try:
            mat[i, :len(sequence)] = [AA_CODE[aa] for aa in sequence]
        except KeyError as exc:
            raise ValueError(f"nonstandard amino acid in QC sequence: {exc}") from exc
    v = d.v_gene.astype(str).map(normal_gene).map(ids)
    check(v.notna().all(), "unknown V code")
    return mat, lengths, v.to_numpy(dtype=np.int16)


def index_seeds(root: Path, scope: str, model: str, seeds: pd.DataFrame,
                seed_sha: str, trees: int, rng_seed: int):
    bank_path = root / "embeddings" / model / "bank_fp16.npy"
    meta_path = root / "embeddings" / model / "meta.json"
    check(bank_path.exists() and meta_path.exists(), "completed embedding bank missing")
    meta = json.loads(meta_path.read_text())
    check(meta.get("status") == "complete", "embedding bank has not passed completion gate")
    bank_stat = bank_path.stat()
    check(meta.get("bank_size_bytes") == bank_stat.st_size and
          meta.get("bank_mtime_ns") == bank_stat.st_mtime_ns,
          "embedding bank differs from completed metadata")
    verified_sha = None
    if model == PRIMARY_ANN_MODEL:
        verification_path = root / "embeddings" / model / "verification.json"
        check(verification_path.exists(), "primary embedding independent verification absent")
        verification = json.loads(verification_path.read_text())
        check(verification.get("status") == "PASS" and
              verification.get("bank_sha256_verified") == meta.get("bank_sha256") and
              verification.get("input_sha256") == meta.get("input_sha256") and
              verification.get("n_total") == meta.get("n_total") and
              verification.get("dimension") == meta.get("bank_shape", [None, None])[1],
              "primary embedding independent verification does not match bank metadata")
        verified_sha = sha(verification_path)
    bank = np.load(bank_path, mmap_mode="r")
    check(bank.ndim == 2 and seeds.seq_idx.max() < len(bank) and
          list(bank.shape) == meta.get("bank_shape"),
          "embedding bank shape/coverage invalid")
    aa_to_nodes: dict[int, list[int]] = defaultdict(list)
    for n, idx in enumerate(seeds.seq_idx.astype(int)):
        aa_to_nodes[idx].append(n)
    unique = sorted(aa_to_nodes)
    index_path = root / "groups" / scope / "tcrdist_annoy.ann"
    meta_out = root / "groups" / scope / "tcrdist_annoy.json"
    expected = {"seed_sha256": seed_sha, "bank_meta_sha256": sha(meta_path),
                "bank_size": bank_stat.st_size, "bank_sha256": meta.get("bank_sha256"),
                "verification_sha256": verified_sha, "trees": trees,
                "random_seed": rng_seed, "n_items": len(unique),
                "dimension": bank.shape[1]}
    index = AnnoyIndex(bank.shape[1], "angular")
    if index_path.exists() and meta_out.exists():
        old = json.loads(meta_out.read_text())
        if all(old.get(k) == v for k,v in expected.items()) and old.get("sha256") == sha(index_path):
            check(index.load(str(index_path)), "failed to load frozen Annoy index")
            return index, {i: aa_to_nodes[x] for i,x in enumerate(unique)}, aa_to_nodes, bank, old["sha256"]
    index.set_seed(rng_seed)
    for local, seq_idx in enumerate(unique):
        vector = np.asarray(bank[seq_idx], dtype=np.float32)
        check(np.isfinite(vector).all(), f"nonfinite embedding row {seq_idx}")
        index.add_item(local, vector.tolist())
    index.build(trees, n_jobs=min(16, os.cpu_count() or 1))
    tmp = index_path.with_suffix(".ann.tmp")
    check(index.save(str(tmp)), "Annoy index save failed")
    tmp.replace(index_path)
    index_sha = sha(index_path)
    receipt(meta_out, expected | {"sha256": index_sha, "status": "complete"})
    return index, {i: aa_to_nodes[x] for i,x in enumerate(unique)}, aa_to_nodes, bank, index_sha


def candidates(index, remap, aa_to_nodes, bank, seq_idx: int, k: int, search_k: int) -> list[int]:
    vector = np.asarray(bank[seq_idx], dtype=np.float32)
    check(np.isfinite(vector).all(), f"nonfinite embedding row {seq_idx}")
    items = index.get_nns_by_vector(vector.tolist(), min(k, index.get_n_items()),
                                    search_k=search_k)
    found = {node for item in items for node in remap[item]}
    found.update(aa_to_nodes.get(seq_idx, []))
    return sorted(found)


def candidate_batch(index, remap, aa_to_nodes, bank, seq_indices,
                    k: int, search_k: int) -> list[list[int]]:
    """Ordered, bounded read-only Annoy queries over one frozen seed index."""
    ids = [int(idx) for idx in seq_indices]
    if len(ids) < 8:
        return [candidates(index, remap, aa_to_nodes, bank, idx, k, search_k)
                for idx in ids]
    def one(idx: int) -> list[int]:
        return candidates(index, remap, aa_to_nodes, bank, idx, k, search_k)
    with ThreadPoolExecutor(max_workers=min(ANN_QUERY_WORKERS, len(ids))) as pool:
        return list(pool.map(one, ids))


def exact_pilot(seeds, index, remap, aa_to_nodes, bank, vcost, vids,
                radius: int, ann_k: int, search_k: int, n: int, seed: int) -> dict:
    check(100 <= n <= 2000 and len(seeds) >= n, "pilot requires 100..2000 sampled seed nodes")
    strata = seeds.v_gene.astype(str) + ":" + seeds.aa.str.len().astype(str)
    rng = np.random.default_rng(seed)
    picks = []
    top = list(strata.value_counts().head(20).index)
    for group in top:
        ix = np.flatnonzero(strata.to_numpy() == group)
        picks.extend(rng.choice(ix, size=min(len(ix), max(1, n//len(top))), replace=False))
    picks = list(dict.fromkeys(map(int, picks)))
    if len(picks) < n:
        remaining = np.setdiff1d(np.arange(len(seeds)), np.array(picks))
        picks.extend(map(int, rng.choice(remaining, size=n-len(picks), replace=False)))
    sample = seeds.iloc[picks[:n]].reset_index(drop=True)
    qm, ql, qv = encode(sample, vids)
    qi = np.repeat(np.arange(n, dtype=np.int64), n)
    si = np.tile(np.arange(n, dtype=np.int64), n)
    t0 = time.monotonic()
    exact = selected(qm, ql, qv, qm, ql, qv, vcost, qi, si).reshape(n,n)
    check(np.array_equal(exact, exact.T) and np.all(np.diag(exact)==0), "exact distance symmetry failed")
    hit = total = novel_hit = novel_total = 0
    per_query_hit = np.zeros(n, dtype=np.int32)
    per_query_total = np.zeros(n, dtype=np.int32)
    per_query_novel_hit = np.zeros(n, dtype=np.int32)
    per_query_novel_total = np.zeros(n, dtype=np.int32)
    all_keys = [tuple(x) for x in seeds[KEY].itertuples(index=False, name=None)]
    pilot_keys = [tuple(x) for x in sample[KEY].itertuples(index=False, name=None)]
    pilot_candidates = candidate_batch(
        index, remap, aa_to_nodes, bank, sample.seq_idx, ann_k, search_k)
    for i, row in sample.iterrows():
        truth = set(np.flatnonzero(exact[i] <= radius)) - {i}
        cand = pilot_candidates[i]
        cand_keys = {all_keys[j] for j in cand}
        per_query_hit[i] = sum(pilot_keys[j] in cand_keys for j in truth)
        per_query_total[i] = len(truth)
        hit += int(per_query_hit[i])
        total += len(truth)
        novel = {j for j in truth if sample.aa.iloc[j] != row.aa}
        per_query_novel_hit[i] = sum(pilot_keys[j] in cand_keys for j in novel)
        per_query_novel_total[i] = len(novel)
        novel_hit += int(per_query_novel_hit[i])
        novel_total += len(novel)
    from scipy.sparse import csr_matrix
    from scipy.sparse.csgraph import connected_components
    adj = exact <= radius
    np.fill_diagonal(adj, False)
    nc, comp = connected_components(csr_matrix(adj), directed=False)
    sizes = np.bincount(comp, minlength=nc)
    lengths = sample.aa.str.len().to_numpy()
    strata_masks = {
        "length_le_12": lengths <= 12,
        "length_13_15": (lengths >= 13) & (lengths <= 15),
        "length_ge_16": lengths >= 16,
    }
    v_array = sample.v_gene.to_numpy()
    for gene in sorted(set(v_array)):
        strata_masks[f"v_{gene}"] = v_array == gene
    strata = {}
    weak_strata = []
    for name, mask in strata_masks.items():
        t = int(per_query_novel_total[mask].sum())
        h = int(per_query_novel_hit[mask].sum())
        strata[name] = {"query_nodes": int(mask.sum()), "nonidentical_aa_directed_neighbors": t,
                        "hits": h, "recall": h/t if t else None}
        if t >= 100 and h/t < .95:
            weak_strata.append(name)
    passed = novel_total >= 50 and novel_hit/novel_total >= .95 and not weak_strata
    return {"status": "pass" if passed else "fail",
            "well_powered_strata_below_recall_target": weak_strata,
            "stratum_minimum_directed_neighbors": 100,
            "recall_by_query_stratum": strata,
            "pilot_n": n, "exact_directed_neighbors": total,
            "neighbor_recall": hit/total if total else None,
            "nonidentical_aa_directed_neighbors": novel_total,
            "nonidentical_aa_neighbor_recall": novel_hit/novel_total if novel_total else None,
            "minimum_nonidentical_aa_neighbors": 50,
            "required_recall": 0.95, "radius": radius, "ann_k": ann_k,
            "search_k": search_k, "exact_pairs": n*n,
            "exact_truth_universe": "stratified training seed subset x same subset; ANN queries use the complete frozen training seed index",
            "complete_training_seed_nodes": len(seeds),
            "giant_component_fraction": float(sizes.max()/n),
            "unassigned_fraction": float((adj.sum(axis=1)==0).mean()),
            "seconds": time.monotonic()-t0,
            "peak_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}



def exact_v_candidates(query_v: np.ndarray, seed_v: np.ndarray,
                       vcost: np.ndarray, radius: int) -> list[np.ndarray]:
    """Exhaustive admissible candidates for default beta tcrdist3.

    The nonnegative CDR3 term means total distance >= V-loop penalty. No
    within-radius neighbor can be discarded when V penalty exceeds radius.
    """
    check(radius >= 0 and bool(np.all(vcost >= 0)), "nonnegative V penalty required")
    lookup = [np.flatnonzero(vcost[v, seed_v] <= radius).astype(np.int64)
              for v in range(len(vcost))]
    return [lookup[int(v)] for v in query_v]


def exact_v_lower_bound_diagnostic(root: Path, scope: str, seed: int,
                                   sample_n: int = 2000,
                                   query_per_cohort: int = 5000) -> Path:
    """Label-blind exhaustive small truth and real-node exact cost pilot."""
    import inspect
    import pyarrow.parquet as pq
    from pwseqdist.matrices import tcr_nb_distance_matrix
    from pwseqdist import metrics as pw_metrics

    check(scope == "full_home", "project-level diagnostic requires full_home")
    check(500 <= sample_n <= 2000 and 1000 <= query_per_cohort <= 10000,
          "diagnostic sample outside bounded range")
    out = root / "groups" / scope
    path = out / "tcrdist_exact_v_lower_bound_diagnostic.json"
    seed_path = out / "tcrdist_seed_nodes.parquet"
    pilot_path = out / "tcrdist_primary_cost_pilot.json"
    desc_path = out / "description.json"
    all_path = root / "groups/tcrdist_all_nodes.parquet"
    seed_receipt = json.loads(pilot_path.read_text())
    desc = json.loads(desc_path.read_text())
    check(seed_receipt.get("status") == "TECHNICAL_ONLY" and
          seed_receipt.get("ann_model") == PRIMARY_ANN_MODEL and
          seed_receipt.get("seed_sha256") == sha(seed_path) and
          seed_receipt.get("train_patient_sha256") == desc.get("train_patient_sha256"),
          "stale training seed or primary bank technical receipt")
    seeds = pd.read_parquet(seed_path)
    check(not seeds.duplicated(KEY).any(), "seed tuple duplicate")
    known = known_human_v()
    vcost, vids = v_penalties(sorted(known))
    mat = np.asarray(tcr_nb_distance_matrix)
    check(mat.min() >= 0 and mat.shape[0] == mat.shape[1] and
          bool(np.all(vcost >= 0)), "distance terms cannot prove V lower bound")
    rep = TCRrep(cell_df=pd.DataFrame({
        "cdr3_b_aa": [_private_literal('private_8b5efd300a8432fa')], "v_b_gene": ["TRBV19*01"],
        "j_b_gene": ["TRBJ2-5*01"], "count": [1]}),
        organism="human", chains=["beta"], compute_distances=False, cpus=1)
    check(rep.weights_b == {"cdr3_b_aa": 3, "pmhc_b_aa": 1,
                            "cdr2_b_aa": 1, "cdr1_b_aa": 1} and
          rep.kargs_b["cdr3_b_aa"]["gap_penalty"] == 4 and
          rep.kargs_b["cdr3_b_aa"]["ntrim"] == 3 and
          rep.kargs_b["cdr3_b_aa"]["ctrim"] == 2 and
          rep.kargs_b["cdr3_b_aa"]["fixed_gappos"] is False,
          "tcrdist3 installed beta defaults changed")
    source = Path(inspect.getsourcefile(pw_metrics.nb_tcrdist.py_func))
    repertoire_source = Path(inspect.getsourcefile(TCRrep))
    rng = np.random.default_rng(seed)
    strata = seeds.v_gene.astype(str) + ":" + seeds.aa.str.len().astype(str)
    picks = []
    top = list(strata.value_counts().head(20).index)
    for group in top:
        ix = np.flatnonzero(strata.to_numpy() == group)
        picks.extend(map(int, rng.choice(ix, size=min(len(ix), max(1, sample_n//len(top))),
                                         replace=False)))
    picks = list(dict.fromkeys(picks))
    if len(picks) < sample_n:
        remaining = np.setdiff1d(np.arange(len(seeds)), np.array(picks))
        picks.extend(map(int, rng.choice(remaining, size=sample_n-len(picks),
                                          replace=False)))
    small = seeds.iloc[picks[:sample_n]].reset_index(drop=True)
    qm, ql, qv = encode(small, vids)
    qi = np.repeat(np.arange(sample_n, dtype=np.int64), sample_n)
    si = np.tile(np.arange(sample_n, dtype=np.int64), sample_n)
    t0 = time.monotonic()
    truth = selected(qm, ql, qv, qm, ql, qv, vcost, qi, si).reshape(sample_n, sample_n)
    check(bool(np.all(truth >= 0)) and np.array_equal(truth, truth.T) and
          bool(np.all(np.diag(truth) == 0)), "exact truth invalid")
    exact_seconds = time.monotonic() - t0
    ref_small = small.drop_duplicates(["aa", "v_gene"]).head(32).reset_index(drop=True)
    ref_in = pd.DataFrame({
        "cdr3_b_aa": ref_small.aa,
        "v_b_gene": ref_small.v_gene.map(normal_gene),
        "j_b_gene": ["TRBJ2-5*01"] * len(ref_small),
        "count": [1] * len(ref_small)})
    r = TCRrep(cell_df=ref_in, organism="human", chains=["beta"],
               compute_distances=True, cpus=1)
    rm, rl, rv = encode(ref_small, vids)
    nr = len(ref_small)
    own_ref = selected(rm, rl, rv, rm, rl, rv, vcost,
                       np.repeat(np.arange(nr, dtype=np.int64), nr),
                       np.tile(np.arange(nr, dtype=np.int64), nr)).reshape(nr, nr)
    expected_keys = [tuple(x) for x in ref_in[["cdr3_b_aa", "v_b_gene"]]
                     .itertuples(index=False, name=None)]
    observed_keys = [tuple(x) for x in r.clone_df[["cdr3_b_aa", "v_b_gene"]]
                     .itertuples(index=False, name=None)]
    order = [expected_keys.index(x) for x in observed_keys]
    check(np.array_equal(own_ref[np.ix_(order, order)], np.asarray(r.pw_beta)),
          "selected differs from installed tcrdist3 TCRrep on real nodes")
    by_radius = {}
    lengths = small.aa.str.len().to_numpy()
    stratum_masks = {"all": np.ones(sample_n, dtype=bool),
                     "length_le_12": lengths <= 12,
                     "length_13_15": (lengths >= 13) & (lengths <= 15),
                     "length_ge_16": lengths >= 16}
    for gene in sorted(set(small.v_gene)):
        stratum_masks[f"v_{gene}"] = small.v_gene.to_numpy() == gene
    for radius in RADIUS_GRID:
        truth_mask = truth <= radius
        np.fill_diagonal(truth_mask, False)
        allowed = vcost[qv[:, None], qv[None, :]] <= radius
        missed = truth_mask & ~allowed
        strata_out = {}
        for name, mask in stratum_masks.items():
            denominator = int(truth_mask[mask].sum())
            missing = int(missed[mask].sum())
            strata_out[name] = {"query_nodes": int(mask.sum()),
                                "directed_exact_neighbors": denominator,
                                "missed_by_v_bound": missing,
                                "recall": 1.0 if denominator else None}
        by_radius[str(radius)] = {
            "status": "PASS" if not bool(missed.any()) else "FAIL",
            "total_directed_exact_neighbors": int(truth_mask.sum()),
            "missed_by_v_bound": int(missed.sum()),
            "recall": 1.0 if truth_mask.any() and not missed.any() else None,
            "candidate_pair_fraction": float(allowed.mean()),
            "strata": strata_out}
    check(all(x["status"] == "PASS" for x in by_radius.values()),
          "V-bound exact recall failed")
    # The global union is read only. No holdout or external node enters seeds,
    # tcrdist reference, threshold selection, or graph fitting.
    pf = pq.ParquetFile(all_path)
    pools: dict[str, list[pd.DataFrame]] = {"home": [], "external": []}
    scan_start = time.monotonic()
    for batch in pf.iter_batches(batch_size=250000,
                                 columns=KEY + ["seq_idx", "cohort_mask"]):
        frame = batch.to_pandas()
        for cohort, bit in (("home", 1), ("external", 2)):
            part = frame.loc[(frame.cohort_mask.to_numpy(dtype=np.uint8) & bit) > 0]
            if len(part):
                chosen = rng.choice(len(part), size=min(100, len(part)), replace=False)
                pools[cohort].append(part.iloc[chosen])
        memcheck()
    scan_seconds = time.monotonic() - scan_start
    sm, sl, sv = encode(seeds, vids)
    query_cost = {}
    for cohort in ("home", "external"):
        sample = pd.concat(pools[cohort], ignore_index=True).drop_duplicates(KEY)
        sample = sample.loc[sample.v_gene.astype(str).map(normal_gene).isin(known)]
        check(len(sample) >= query_per_cohort, f"too few sampled {cohort} nodes")
        sample = sample.iloc[np.sort(rng.choice(len(sample), query_per_cohort,
                                                replace=False))].reset_index(drop=True)
        qm, ql, qv = encode(sample, vids)
        chunk_seconds = []
        candidate_counts = []
        within = {str(r): 0 for r in RADIUS_GRID}
        exact_pairs = 0
        start = time.monotonic()
        for first in range(0, len(sample), 1000):
            last = min(first+1000, len(sample))
            cand = exact_v_candidates(qv[first:last], sv, vcost, RADIUS_GRID[-1])
            counts = np.fromiter((len(x) for x in cand), dtype=np.int64,
                                 count=len(cand))
            qidx = np.repeat(np.arange(last-first, dtype=np.int64), counts)
            sidx = np.concatenate(cand) if counts.sum() else np.empty(0, dtype=np.int64)
            t1 = time.monotonic()
            distances = selected(qm[first:last], ql[first:last], qv[first:last],
                                 sm, sl, sv, vcost, qidx, sidx)
            offset = 0
            for n in counts:
                n = int(n)
                if n:
                    dmin = int(distances[offset:offset+n].min())
                    for radius in RADIUS_GRID:
                        within[str(radius)] += dmin <= radius
                offset += n
            chunk_seconds.append(time.monotonic()-t1)
            exact_pairs += len(sidx)
            candidate_counts.extend(map(int, counts))
            memcheck()
        seconds = time.monotonic()-start
        rate = len(sample)/seconds
        arr = np.asarray(candidate_counts)
        bands = {
            "length_le_12": sample.aa.str.len().to_numpy() <= 12,
            "length_13_15": sample.aa.str.len().to_numpy().astype(int).clip(13,15) == sample.aa.str.len().to_numpy(),
            "length_ge_16": sample.aa.str.len().to_numpy() >= 16}
        query_cost[cohort] = {
            "n_query_nodes": len(sample), "seconds": seconds,
            "nodes_per_second": rate, "exact_pairs": exact_pairs,
            "pairs_per_second": exact_pairs/sum(chunk_seconds),
            "mean_candidates_per_query": float(arr.mean()),
            "p95_candidates_per_query": float(np.quantile(arr, .95)),
            "max_candidates_per_query": int(arr.max()),
            "zero_candidate_queries": int((arr == 0).sum()),
            "within_radius_query_fraction": {r:n/len(sample) for r,n in within.items()},
            "length_strata": {name: {"query_nodes": int(mask.sum()),
                                     "mean_candidate_count": float(arr[mask].mean())
                                     if mask.any() else None}
                              for name, mask in bands.items()},
        }
    conservative_rate = min(x["nodes_per_second"] for x in query_cost.values())
    all_receipt = json.loads((root/"groups/tcrdist_all_nodes_receipt.json").read_text())
    n_global = int(all_receipt["eligible_global_nodes"])
    conservative_factor = 1.5
    six_hours = 6 * conservative_factor * n_global / conservative_rate / 3600
    data = {
        "status": "PASS_TECHNICAL_LOWER_BOUND"
                  if six_hours <= 96 else "STOP_RESOURCE",
        "scope": scope, "method": "tcrdist3",
        "candidate_strategy": "exact_V_cost_pruning",
        "training_only_dictionary": True,
        "global_nodes_role": "read-only cost/projection universe; no graph or threshold fitting",
        "cannot_authorize_formal_projection": True,
        "reference_equivalence": {"real_nodes": nr,
                                  "real_node_pairs": nr*nr,
                                  "distance_equal_to_tcrdist3_TCRrep": True,
                                  "tcrdist3_default_beta_weights": rep.weights_b,
                                  "nonnegative_matrix_min": int(mat.min()),
                                  "nonnegative_vcost_min": int(vcost.min()),
                                  "cdr3_gap_penalty": 4,
                                  "cdr3_multiplier": 3,
                                  "v_loop_multipliers": [1,1,1],
                                  "pwseqdist_metrics_source_sha256": sha(source),
                                  "tcrdist_repertoire_source_sha256": sha(repertoire_source)},
        "radius_grid": list(RADIUS_GRID),
        "exhaustive_truth_sample_n": sample_n,
        "exhaustive_truth_pairs": sample_n*sample_n,
        "exhaustive_truth_seconds": exact_seconds,
        "exhaustive_recall_by_radius": by_radius,
        "cost_sample_scan_seconds": scan_seconds,
        "real_node_cost_by_cohort": query_cost,
        "exact_global_nodes": n_global,
        "six_scope_hours_with_1p5_output_factor": six_hours,
        "21_scope_hours_with_1p5_output_factor": 21*conservative_factor*n_global/conservative_rate/3600,
        "required_six_scope_hours_max": 96,
        "seed_sha256": sha(seed_path),
        "train_patient_sha256": desc["train_patient_sha256"],
        "node_description_sha256": sha(desc_path),
        "global_node_file_sha256": sha(all_path),
        "script_sha256": sha(Path(__file__)),
        "peak_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "remaining_gates": "full training seed radius-grid graph and giant fraction <=0.10; six-scope project gate; exhaustive full projection/checkpoint validation",
    }
    receipt(path, data)
    return path

def components(n: int, edges: list[tuple[int,int]]):
    parent = np.arange(n, dtype=np.int64)
    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = int(parent[x])
        return x
    for i,j in edges:
        a,b=find(i),find(j)
        if a!=b:
            parent[max(a,b)] = min(a,b)
    roots=np.array([find(i) for i in range(n)],dtype=np.int64)
    _, inv, counts=np.unique(roots,return_inverse=True,return_counts=True)
    return inv,counts




def frozen_seed_graph_grid(seeds: pd.DataFrame, index, remap, aa_to_nodes,
                           bank, vcost, vids, radii: tuple[int, ...],
                           ann_k: int, search_k: int, chunk_size: int):
    """One training-only ANN/exact pass, then fixed-radius component summaries."""
    check(bool(radii) and tuple(sorted(set(radii))) == radii,
          "radius grid must be nonempty, unique and increasing")
    sm, sl, sv = encode(seeds, vids)
    edge_distance: dict[tuple[int, int], int] = {}
    maximum = radii[-1]
    for start in range(0, len(seeds), chunk_size):
        end = min(start + chunk_size, len(seeds))
        cand = candidate_batch(index, remap, aa_to_nodes, bank,
                               seeds.seq_idx.iloc[start:end], ann_k, search_k)
        qi = np.repeat(np.arange(end-start, dtype=np.int64), [len(x) for x in cand])
        si = np.asarray([j for row in cand for j in row], dtype=np.int64)
        qm, ql, qv = encode(seeds.iloc[start:end], vids)
        dist = selected(qm, ql, qv, sm, sl, sv, vcost, qi, si)
        for local, other, distance in zip(qi, si, dist):
            node = start + int(local)
            other = int(other)
            distance = int(distance)
            if node == other or distance > maximum:
                continue
            key = (min(node, other), max(node, other))
            if distance < edge_distance.get(key, maximum + 1):
                edge_distance[key] = distance
        memcheck()
    graphs = {}
    for radius in radii:
        edges = (pair for pair, distance in edge_distance.items()
                 if distance <= radius)
        comp, sizes = components(len(seeds), edges)
        grouped = sizes[comp] >= 2
        nonsingle = sizes[sizes >= 2]
        graphs[str(radius)] = {
            "comp": comp, "sizes": sizes,
            "metrics": {
                "radius": radius,
                "edges": sum(distance <= radius for distance in edge_distance.values()),
                "components": len(sizes),
                "largest_component_size": int(sizes.max()),
                "giant_component_fraction": float(sizes.max()/len(seeds)),
                "singleton_seed_fraction": float((~grouped).mean()),
                "grouped_seed_fraction": float(grouped.mean()),
                "groups_ge_2": int(len(nonsingle)),
                "groups_ge_10": int((sizes >= 10).sum()),
                "nonsingle_group_size_p50": float(np.median(nonsingle)) if len(nonsingle) else None,
                "nonsingle_group_size_p95": float(np.quantile(nonsingle, .95)) if len(nonsingle) else None,
                "acceptance_population": "Home training seed nodes only",
            },
        }
    return graphs



def frozen_seed_graph_exact_v_grid(seeds: pd.DataFrame, vcost: np.ndarray,
                                   vids: dict[str, int],
                                   radii: tuple[int, ...] = RADIUS_GRID,
                                   chunk_size: int = 1000) -> dict:
    """Exact train-only seed graph; V-bound prunes impossible radius edges."""
    check(bool(radii) and tuple(sorted(set(radii))) == radii,
          "radius grid must be nonempty, unique and increasing")
    check(1 <= chunk_size <= 10000, "exact graph chunk outside safe bounds")
    sm, sl, sv = encode(seeds, vids)
    heads, tails, weights = [], [], []
    maximum = radii[-1]
    total_candidates = 0
    t0 = time.monotonic()
    for start in range(0, len(seeds), chunk_size):
        end = min(start + chunk_size, len(seeds))
        cand = exact_v_candidates(sv[start:end], sv, vcost, maximum)
        counts = np.fromiter((len(x) for x in cand), dtype=np.int64,
                             count=len(cand))
        qi = np.repeat(np.arange(end-start, dtype=np.int64), counts)
        si = np.concatenate(cand) if counts.sum() else np.empty(0, dtype=np.int64)
        dist = selected(sm[start:end], sl[start:end], sv[start:end],
                        sm, sl, sv, vcost, qi, si)
        total_candidates += len(si)
        source = start + qi
        keep = (dist <= maximum) & (si > source)
        heads.append(source[keep].astype(np.int32))
        tails.append(si[keep].astype(np.int32))
        weights.append(dist[keep].astype(np.int16))
        memcheck()
    ei = np.concatenate(heads) if heads else np.empty(0, dtype=np.int32)
    ej = np.concatenate(tails) if tails else np.empty(0, dtype=np.int32)
    ed = np.concatenate(weights) if weights else np.empty(0, dtype=np.int16)
    check(len(ei) == len(ej) == len(ed) and bool(np.all(ej > ei)),
          "exact seed graph edges invalid")
    graphs = {}
    for radius in radii:
        chosen = ed <= radius
        edges = zip(ei[chosen].tolist(), ej[chosen].tolist())
        comp, sizes = components(len(seeds), edges)
        grouped = sizes[comp] >= 2
        nonsingle = sizes[sizes >= 2]
        graphs[str(radius)] = {
            "comp": comp, "sizes": sizes,
            "metrics": {
                "radius": radius, "edges": int(chosen.sum()),
                "components": len(sizes),
                "largest_component_size": int(sizes.max()),
                "giant_component_fraction": float(sizes.max()/len(seeds)),
                "singleton_seed_fraction": float((~grouped).mean()),
                "grouped_seed_fraction": float(grouped.mean()),
                "groups_ge_2": int(len(nonsingle)),
                "groups_ge_10": int((sizes >= 10).sum()),
                "nonsingle_group_size_p50": float(np.median(nonsingle))
                if len(nonsingle) else None,
                "nonsingle_group_size_p95": float(np.quantile(nonsingle, .95))
                if len(nonsingle) else None,
                "acceptance_population": "Home training seed nodes only",
                "candidate_strategy": "exact_V_cost_pruning",
            }}
    return {"graphs": graphs, "candidate_pairs": total_candidates,
            "exact_undirected_edges_max_radius": len(ei),
            "seconds": time.monotonic()-t0}


def exact_v_graph_diagnostic(root: Path, scope: str, seed: int,
                             chunk_size: int = 1000,
                             per_patient: int = 512) -> Path:
    """Complete training-seed radius grid; no labels or global projection fit."""
    out = root / "groups" / scope
    out.mkdir(parents=True, exist_ok=True)
    path = out / "tcrdist_exact_v_graph_preflight.json"
    proof_path = root / "groups/full_home/tcrdist_exact_v_lower_bound_diagnostic.json"
    seed_path = out / "tcrdist_seed_nodes.parquet"
    desc_path = out / "description.json"
    report = {"status": "running", "scope": scope, "method": "tcrdist3",
              "candidate_strategy": "exact_V_cost_pruning",
              "radius_grid": list(RADIUS_GRID),
              "max_giant_component_fraction": 0.10,
              "dictionary_fit_population": "Home training patients in this scope only",
              "global_nodes_role": "read-only cost/projection universe, never graph fitting",
              "script_sha256": sha(Path(__file__))}
    receipt(path, report)
    t0 = time.monotonic()
    try:
        proof = json.loads(proof_path.read_text())
        gate(proof.get("status") == "PASS_TECHNICAL_LOWER_BOUND" and
             all(proof["exhaustive_recall_by_radius"][str(r)]["status"] == "PASS"
                 for r in RADIUS_GRID),
             "exact V-cost lower-bound proof has not passed")
        desc = json.loads(desc_path.read_text())
        check(desc.get("scope") == scope and
              desc.get("patient_nodes_sha256") ==
              sha(out / "patient_nodes.parquet") and
              desc.get("clone_nodes_sha256") ==
              sha(out / "clone_nodes.parquet"),
              "training-scope nodes changed after description")
        ids = scope_ids(root, scope)
        check(desc.get("train_patient_sha256") == ids_hash(ids),
              "training patient hash differs from node description")
        if not seed_path.exists():
            prepared, prepared_ids = balanced_seeds(root, scope, per_patient, seed)
            check(prepared_ids == ids, "prepared seed patient set differs")
            tmp_seed = seed_path.with_suffix(".parquet.tmp")
            prepared.to_parquet(tmp_seed, index=False, compression="zstd")
            tmp_seed.replace(seed_path)
        if scope == "full_home":
            check(proof.get("seed_sha256") == sha(seed_path) and
                  proof.get("train_patient_sha256") == ids_hash(ids),
                  "full_home seed differs from lower-bound proof")
        seeds = pd.read_parquet(seed_path)
        check(len(seeds) > 0 and not seeds.duplicated(KEY).any(),
              "training seed file invalid")
        known = known_human_v()
        check(seeds.v_gene.astype(str).map(normal_gene).isin(known).all(),
              "seed contains unscorable V")
        vcost, vids = v_penalties(sorted(known))
        graph = frozen_seed_graph_exact_v_grid(
            seeds, vcost, vids, RADIUS_GRID, chunk_size)
        metrics = {r:x["metrics"] for r,x in graph["graphs"].items()}
        valid = [r for r in RADIUS_GRID
                 if metrics[str(r)]["giant_component_fraction"] <= .10]
        report.update({
            "status": "PASS_TRAIN_GRAPH" if valid else "STOP_GIANT_COMPONENT",
            "train_patient_sha256": ids_hash(ids),
            "training_patients": len(ids),
            "node_description_sha256": sha(desc_path),
            "folds_sha256": sha(root / "folds/folds.tsv"),
            "seed_sha256": sha(seed_path), "seed_nodes": len(seeds),
            "source_node_count": desc.get("n_unique_aa_vj"),
            "training_seed_graph_grid": metrics,
            "selected_radius": max(valid) if valid else None,
            "radius_selection_rule": "largest prespecified 16/20/24 radius with train-only giant <=0.10 and exact V-bound recall 1.0",
            "candidate_pairs": graph["candidate_pairs"],
            "exact_undirected_edges_at_max_radius": graph["exact_undirected_edges_max_radius"],
            "graph_seconds": graph["seconds"],
            "proof_receipt_sha256": sha(proof_path),
            "reference_db_sha256": reference_hash(),
            "cannot_authorize_formal_projection": True,
            "remaining_gates": "project-level six-scope cost, scope-specific preflight and resumable full projection",
        })
        if valid:
            selected_graph = graph["graphs"][str(max(valid))]
            graph_path = out / "tcrdist_exact_v_selected_seed_graph.npz"
            graph_tmp = graph_path.with_suffix(".npz.tmp")
            with graph_tmp.open("wb") as handle:
                np.savez_compressed(handle, comp=selected_graph["comp"],
                                    sizes=selected_graph["sizes"])
            graph_tmp.replace(graph_path)
            report["selected_seed_graph_sha256"] = sha(graph_path)
            report["selected_seed_graph_path"] = str(graph_path)
    except GateStop as exc:
        report.update({"status": "STOP_PROOF", "stop_reason": str(exc)})
    except Exception as exc:
        report.update({"status": "failed", "failure": f"{type(exc).__name__}: {exc}"})
        raise
    finally:
        report["seconds"] = time.monotonic()-t0
        report["peak_rss_kib"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        receipt(path, report)
    return path

def select_training_radius(exact_recall: dict, graph_grid: dict,
                           radii: tuple[int, ...] = RADIUS_GRID) -> int:
    """Largest prespecified radius passing train-only recall and giant gates."""
    valid = [radius for radius in radii
             if exact_recall[str(radius)]["status"] == "pass" and
             graph_grid[str(radius)]["giant_component_fraction"] <= .10]
    gate(bool(valid), "no radius in training-only 16/20/24 grid passes exact recall and giant-component gates")
    return max(valid)


def consolidate_membership(all_nodes: pd.DataFrame, parts: Path, out: Path,
                           scope: str, chunk_size: int,
                           membership_name: str = "tcrdist_group_membership.parquet") -> dict:
    """Stream verified ordered projection parts into the assigned-only table."""
    import pyarrow as pa
    import pyarrow.parquet as pq
    from collections import Counter

    member_path = out / membership_name
    tmp = member_path.with_suffix(".parquet.tmp")
    writer = None
    group_counts = Counter()
    cohort_counts = {
        "home": {"observed": 0, "assigned": 0, "ambiguous": 0, "unassigned": 0},
        "external": {"observed": 0, "assigned": 0, "ambiguous": 0, "unassigned": 0},
    }
    evaluated = assigned = ambiguous = 0
    for start in range(0, len(all_nodes), chunk_size):
        end = min(start + chunk_size, len(all_nodes))
        part = parts / f"part_{start//chunk_size:06d}.parquet"
        check(part.exists(), f"missing projection part {part}")
        frame = pd.read_parquet(part)
        expected_keys = all_nodes.iloc[start:end][KEY].reset_index(drop=True)
        check(len(frame) == end-start and
              frame[KEY].reset_index(drop=True).equals(expected_keys),
              f"projection part order or keys differ: {part}")
        statuses = frame.projection_status
        check(statuses.isin(["assigned", "ambiguous", "unassigned"]).all(),
              f"invalid projection status in {part}")
        select = statuses.eq("assigned")
        for flag, cohort in ((1, "home"), (2, "external")):
            observed = (frame.cohort_mask.astype(np.uint8) & flag) != 0
            cohort_counts[cohort]["observed"] += int(observed.sum())
            for name in ("assigned", "ambiguous", "unassigned"):
                cohort_counts[cohort][name] += int((observed & statuses.eq(name)).sum())
        check(frame.loc[select, "group_id"].astype(str).str.len().gt(0).all(),
              f"assigned node missing group ID in {part}")
        check(frame.loc[~select, "group_id"].astype(str).str.len().eq(0).all(),
              f"unassigned node carries group ID in {part}")
        member = frame.loc[select,
                           ["scope", "method", "group_id"]+KEY+
                           ["seq_idx", "cohort_mask", "nearest_tcrdist"]]
        check(member.scope.eq(scope).all() and member.method.eq("tcrdist3").all(),
              f"projection method or scope differs in {part}")
        evaluated += len(frame)
        assigned += len(member)
        ambiguous += int(statuses.eq("ambiguous").sum())
        if len(member):
            table = pa.Table.from_pandas(member, preserve_index=False)
            if writer is None:
                writer = pq.ParquetWriter(str(tmp), table.schema, compression="zstd")
            writer.write_table(table)
            group_counts.update(member.group_id)
        memcheck()
    if writer is not None:
        writer.close()
    else:
        empty = pd.DataFrame({
            "scope": pd.Series(dtype="str"), "method": pd.Series(dtype="str"),
            "group_id": pd.Series(dtype="str"), "aa": pd.Series(dtype="str"),
            "v_gene": pd.Series(dtype="str"), "j_gene": pd.Series(dtype="str"),
            "seq_idx": pd.Series(dtype="int64"),
            "cohort_mask": pd.Series(dtype="uint8"),
            "nearest_tcrdist": pd.Series(dtype="int32"),
        })
        empty.to_parquet(tmp, index=False, compression="zstd")
    check(evaluated == len(all_nodes), "membership consolidation incomplete")
    tmp.replace(member_path)
    return {"evaluated_nodes": evaluated, "assigned_nodes": assigned,
            "ambiguous_nodes": ambiguous,
            "group_count": len(group_counts),
            "max_group_fraction": max(group_counts.values(), default=0)/evaluated,
            "cohort_projection_diagnostics": cohort_counts,
            "group_membership_sha256": sha(member_path)}


def run(root: Path, scope: str, model: str, per_patient: int, radius: int,
        ann_k: int, search_k: int, trees: int, pilot_n: int, chunk_size: int,
        seed: int, max_hours: float) -> Path:
    check(1 <= radius <= 100 and ann_k >= 16 and 100 <= chunk_size <= 100000,
          "invalid radius, ann_k or chunk_size")
    out = root / "groups" / scope
    out.mkdir(parents=True, exist_ok=True)
    rpath = out / "tcrdist_projection_receipt.json"
    state = {"scope":scope,"method":"tcrdist3","script_sha256":sha(Path(__file__)),"status":"initializing",
             "label_blind":True,"projection_complete":False,
             "global_nodes_role":"read-only global AA/V/J projection universe; never used to fit seed/index/radius",
             "dictionary_fit_population":"only Home training patients in this scope",
             "locked_gates":{"max_rss_gib":128,"max_cpu_threads":16,
                             "max_one_scope_projection_hours":max_hours,
                             "max_giant_component_fraction":0.10,
                             "max_unscorable_v_node_fraction":0.05,
                             "max_excluded_vj_read_fraction":0.05,
                             "min_nonidentical_aa_recall":0.95,
                             "min_nonidentical_aa_pairs":50,
                             "min_stratum_pairs_for_recall_gate":100,
                             "min_well_powered_stratum_recall":0.95}}
    receipt(rpath,state)
    t0=time.monotonic()
    try:
        gate_path=root/"groups/tcrdist_project_cost_gate.json"
        gate(gate_path.exists(),"project-level tcrdist cost gate missing")
        project=json.loads(gate_path.read_text())
        identity_sha, home_label_sha = manifest_fingerprints(
            root / "config/cohort_samples.tsv")
        gate(project.get("status")=="PASS" and scope in project.get("authorized_scopes",[]) and
             project.get("script_sha256")==sha(Path(__file__)) and
             project.get("ann_model")==model and
             project.get("manifest_identity_sha256")==identity_sha and
             project.get("home_split_label_sha256")==home_label_sha,
             "project-level tcrdist cost gate not valid for this scope/script")
        state["project_cost_gate_sha256"]=sha(gate_path)
        all_path=prepare(root)
        gate(project.get("all_nodes_sha256")==sha(all_path),
             "global node universe changed after project cost gate")
        all_nodes_full=pd.read_parquet(all_path)
        known=known_human_v()
        scorable=all_nodes_full.v_gene.astype(str).map(normal_gene).isin(known)
        unscorable=all_nodes_full.loc[~scorable].copy()
        unscorable["reason"]="V_absent_from_tcrdist3_human_reference"
        unscorable_path=out/"tcrdist_unscorable_v_nodes.parquet"
        unscorable.to_parquet(unscorable_path,index=False,compression="zstd")
        all_nodes=all_nodes_full.loc[scorable].reset_index(drop=True)
        check(len(all_nodes)>0,"no reference-scorable nodes")
        source_receipt=json.loads((root/"groups/tcrdist_all_nodes_receipt.json").read_text())
        seeds, ids=balanced_seeds(root,scope,per_patient,seed)
        train_quality=training_scope_quality(root,scope,ids,known)
        state.update({"train_patient_sha256":ids_hash(ids),
                      "training_patients":len(ids),
                      "node_description_sha256":sha(out/"description.json"),
                      "folds_sha256":sha(root/"folds/folds.tsv"),
                      "manifest_identity_sha256":identity_sha,
                      "home_split_label_sha256":home_label_sha,
                      "training_scope_quality":train_quality,
                      "scientific_gate_population":"Home training patients of this scope only",
                      "heldout_external_projection_metrics_role":"diagnostic_only"})
        receipt(rpath,state)
        gate(train_quality["training_unscorable_v_node_fraction"]<=0.05 and
             train_quality["training_excluded_vj_read_fraction"]<=0.05,
             "training-scope V/J exclusion exceeds 5% node or read gate: "
             f"nodes={train_quality['training_unscorable_v_node_fraction']:.4f}, "
             f"excluded_reads={train_quality['training_excluded_vj_read_fraction']:.4f}")
        seed_path=out/"tcrdist_seed_nodes.parquet"
        if not seed_path.exists() or not pd.read_parquet(seed_path).equals(seeds):
            tmp_seed=seed_path.with_suffix(".parquet.tmp")
            seeds.to_parquet(tmp_seed,index=False,compression="zstd")
            tmp_seed.replace(seed_path)
        vcost, vids=v_penalties(sorted(known))
        seed_sha=sha(seed_path); all_sha=sha(all_path)
        index,remap,aa_to_nodes,bank,index_sha=index_seeds(root,scope,model,seeds,seed_sha,trees,seed)
        pre_path=out/"tcrdist_preflight_receipt.json"
        gate(pre_path.exists(), "matching training-scope radius-grid preflight missing")
        pre=json.loads(pre_path.read_text())
        pre_params=pre.get("params",{})
        gate(pre.get("status")=="PASS_PRELIMINARY" and
             pre.get("scientific_dictionary_status")=="PASS" and
             pre.get("resource_status")=="PASS_ESTIMATE" and
             pre.get("scope")==scope and
             pre.get("script_sha256")==sha(Path(__file__)) and
             pre.get("train_patient_sha256")==ids_hash(ids) and
             pre.get("manifest_identity_sha256")==identity_sha and
             pre.get("home_split_label_sha256")==home_label_sha and
             pre.get("seed_sha256")==seed_sha and
             pre.get("annoy_index_sha256")==index_sha and
             pre_params.get("model")==model and
             pre_params.get("per_patient")==per_patient and
             pre_params.get("search_k")==search_k and
             pre_params.get("trees")==trees and
             pre_params.get("pilot_n")==pilot_n and
             pre_params.get("chunk_size")==chunk_size and
             pre_params.get("seed")==seed and
             pre_params.get("radius_grid")==list(RADIUS_GRID) and
             pre.get("selected_ann_k")==ann_k==512,
             "training-scope primary-bank preflight does not match this frozen run")
        requested_radius=radius
        radius=int(pre["selected_radius"])
        gate(radius in RADIUS_GRID and
             select_training_radius(pre["exact_recall_by_radius_k512"],
                                    pre["seed_graph_grid_k512"])==radius,
             "preflight selected radius is not justified by training-only grid")
        graph_path=out/"tcrdist_selected_seed_graph.npz"
        gate(graph_path.exists() and
             pre.get("selected_seed_graph_sha256")==sha(graph_path),
             "selected training seed graph cache missing or changed")
        with np.load(graph_path) as cached:
            comp=np.asarray(cached["comp"],dtype=np.int64)
            sizes=np.asarray(cached["sizes"],dtype=np.int64)
        check(len(comp)==len(seeds) and sizes.sum()==len(seeds) and
              len(sizes)>0 and comp.min()>=0 and comp.max()<len(sizes) and
              np.array_equal(np.bincount(comp,minlength=len(sizes)),sizes),
              "selected training seed graph cache invalid")
        graph_metrics=pre["seed_graph_grid_k512"][str(radius)]
        check(int(sizes.max())==graph_metrics["largest_component_size"],
              "training graph cache differs from preflight metrics")
        sm,sl,sv=encode(seeds,vids)
        pilot_path=out/"tcrdist_projection_pilot.json"
        pilot=dict(pre["exact_recall_by_radius_k512"][str(radius)])
        pilot.update({"scope":scope,"model":model,
                      "source_preflight_sha256":sha(pre_path),
                      "script_sha256":sha(Path(__file__)),"seed_sha256":seed_sha,
                      "annoy_index_sha256":index_sha,"radius":radius,
                      "ann_k":ann_k,"search_k":search_k,
                      "pilot_n":pilot_n,"random_seed":seed})
        if not pilot_path.exists() or json.loads(pilot_path.read_text())!=pilot:
            receipt(pilot_path,pilot)
        pilot_sha=sha(pilot_path)
        excluded_frac=len(unscorable)/len(all_nodes_full)
        read_frac=float(source_receipt["unscorable_v_read_fraction"])
        state.update({"train_patient_sha256":ids_hash(ids),"training_patients":len(ids),
                      "seed_nodes":len(seeds),"source_nodes_total":len(all_nodes_full),
                      "scorable_nodes_total":len(all_nodes),
                      "n_unscorable_v_nodes":len(unscorable),
                      "unscorable_v_nodes_sha256":sha(unscorable_path),
                      "unscorable_v_node_fraction":excluded_frac,
                      "unscorable_v_read_fraction":read_frac,
                      "global_projection_exclusions_role":"diagnostic_only; excluded from scientific dictionary PASS gate",
                      "unscorable_v_patient_carriers_home":source_receipt["unscorable_v_patient_carriers_home"],
                      "unscorable_v_patient_carriers_external":source_receipt["unscorable_v_patient_carriers_external"],
                      "malformed_vj_clone_rows":source_receipt["malformed_vj_clone_rows"],
                      "malformed_vj_rows_sha256":source_receipt["malformed_vj_rows_sha256"],
                      "malformed_vj_reads":source_receipt["malformed_vj_reads"],
                      "malformed_vj_read_fraction":source_receipt["malformed_vj_read_fraction"],
                      "malformed_vj_patient_carriers_home":source_receipt["malformed_vj_patient_carriers_home"],
                      "malformed_vj_patient_carriers_external":source_receipt["malformed_vj_patient_carriers_external"],
                      "tcrdist_reference_db_sha256":reference_hash(),
                      "seed_sha256":seed_sha,"all_nodes_sha256":all_sha,"annoy_index_sha256":index_sha,
                      "pilot_sha256":pilot_sha,"pilot":pilot,
                      "params":{"model":model,"per_patient":per_patient,"radius":radius,
                                "ann_k":ann_k,"search_k":search_k,"trees":trees,
                                "pilot_n":pilot_n,"chunk_size":chunk_size,"seed":seed},
                      "distance_definition":"tcrdist3 human beta default V CDR1/2/2.5 plus 3x CDR3; J is node annotation"})
        gate(pilot["status"]=="pass","training-seed ANN exact-neighbor recall pilot failed")
        state["seed_graph"]=graph_metrics
        state["preflight_receipt_sha256"]=sha(pre_path)
        state["selected_seed_graph_sha256"]=sha(graph_path)
        state["radius_selection"]="largest train-only passing radius in 16/20/24 grid"
        state["requested_radius_argument_ignored_for_formal_run"]=requested_radius
        gate(graph_metrics["giant_component_fraction"]<=0.10,
             f"giant seed component {graph_metrics['giant_component_fraction']:.3f} exceeds 0.10")
        names={}
        for c in np.flatnonzero(sizes>=2):
            members=seeds.loc[comp==c,KEY]
            first=min("|".join(map(str,r)) for r in members.itertuples(index=False,name=None))
            names[int(c)]="TD_"+hashlib.sha256(first.encode()).hexdigest()[:16]
        sg=np.array([names.get(int(c),"") for c in comp],dtype=object)
        parts=out/"tcrdist_projection_parts"
        parts.mkdir(exist_ok=True)
        projected=assigned=ambiguous=0
        for start in range(0,len(all_nodes),chunk_size):
            end=min(start+chunk_size,len(all_nodes)); shard=start//chunk_size
            part=parts/f"part_{shard:06d}.parquet"
            pr=parts/f"part_{shard:06d}.json"
            expected={"start":start,"end":end,"seed_sha256":seed_sha,
                      "all_nodes_sha256":all_sha,"annoy_index_sha256":index_sha,
                      "pilot_sha256":pilot_sha}
            if part.exists() and pr.exists():
                old=json.loads(pr.read_text())
                if all(old.get(k)==v for k,v in expected.items()) and old.get("sha256")==sha(part):
                    projected+=end-start;assigned+=int(old["assigned"]);ambiguous+=int(old["ambiguous"])
                    continue
            chunk=all_nodes.iloc[start:end].reset_index(drop=True)
            tc=time.monotonic()
            cand=candidate_batch(index,remap,aa_to_nodes,bank,
                                 chunk.seq_idx,ann_k,search_k)
            qi=np.repeat(np.arange(len(chunk),dtype=np.int64),[len(x) for x in cand])
            si=np.asarray([j for row in cand for j in row],dtype=np.int64)
            qm,ql,qv=encode(chunk,vids)
            dist=selected(qm,ql,qv,sm,sl,sv,vcost,qi,si)
            groups=np.array([""]*len(chunk),dtype=object)
            status=np.array(["unassigned"]*len(chunk),dtype=object)
            nearest=np.full(len(chunk),-1,dtype=np.int32)
            offset=0
            for local, options in enumerate(cand):
                d=dist[offset:offset+len(options)]
                offset+=len(options)
                if not len(d): continue
                md=int(d.min());nearest[local]=md
                if md>radius: continue
                best={sg[options[k]] for k in np.flatnonzero(d==md)}
                if len(best)==1 and "" not in best:
                    groups[local]=next(iter(best));status[local]="assigned"
                elif len(best)>1:
                    status[local]="ambiguous"
            chunk["scope"]=scope;chunk["method"]="tcrdist3"
            chunk["group_id"]=groups;chunk["projection_status"]=status
            chunk["nearest_tcrdist"]=nearest
            tmp=part.with_suffix(".parquet.tmp")
            chunk.to_parquet(tmp,index=False,compression="zstd");tmp.replace(part)
            rec=expected|{"sha256":sha(part),"evaluated":len(chunk),
                          "assigned":int((status=="assigned").sum()),
                          "ambiguous":int((status=="ambiguous").sum()),
                          "seconds":time.monotonic()-tc,"rss_bytes":memcheck()}
            receipt(pr,rec)
            projected+=len(chunk);assigned+=rec["assigned"];ambiguous+=rec["ambiguous"]
            if start==0:
                estimate=rec["seconds"]*len(all_nodes)/len(chunk)/3600
                state["estimated_projection_hours_from_first_chunk"]=estimate
                receipt(rpath,state)
                gate(estimate<=max_hours,f"estimated {estimate:.1f} hours exceeds {max_hours}h cap")
            print(json.dumps({"scope":scope,"projected":projected,"of":len(all_nodes),
                              "assigned":assigned}),flush=True)
        check(projected==len(all_nodes),"incomplete projection")
        summary=consolidate_membership(all_nodes,parts,out,scope,chunk_size)
        check(summary["evaluated_nodes"]==projected and summary["assigned_nodes"]==assigned and
              summary["ambiguous_nodes"]==ambiguous,"part receipt and membership counts disagree")
        state.update({"status":"complete_full_scope","projection_complete":True,
                      "evaluated_nodes":projected,"scorable_nodes_projected_fraction":projected/len(all_nodes),"assigned_nodes":assigned,
                      "unassigned_nodes":projected-assigned-ambiguous,
                      "ambiguous_nodes":ambiguous,"assigned_fraction":assigned/projected,
                      "home_observed_nodes":int(((all_nodes.cohort_mask&1)!=0).sum()),
                      "external_observed_nodes":int(((all_nodes.cohort_mask&2)!=0).sum()),
                      "unscorable_v_nodes_home":int(((unscorable.cohort_mask&1)!=0).sum()),
                      "unscorable_v_nodes_external":int(((unscorable.cohort_mask&2)!=0).sum()),
                      "cohort_projection_diagnostics":summary["cohort_projection_diagnostics"],
                      "cohort_projection_diagnostics_role":"descriptive_only; no dictionary PASS decision",
                      "group_membership_sha256":summary["group_membership_sha256"],
                      "membership_sha256":summary["group_membership_sha256"],
                      "group_count":summary["group_count"],
                      "max_group_fraction":summary["max_group_fraction"],
                      "seconds":time.monotonic()-t0,
                      "peak_rss_kib":resource.getrusage(resource.RUSAGE_SELF).ru_maxrss})
        receipt(rpath,state)
    except GateStop as exc:
        state.update({"status":"STOP","projection_complete":False,
                      "reason":str(exc),"stop_reason":str(exc),
                      "seconds":time.monotonic()-t0,
                      "peak_rss_kib":resource.getrusage(resource.RUSAGE_SELF).ru_maxrss})
        receipt(rpath,state)
        return rpath
    except Exception as exc:
        state.update({"status":"failed","projection_complete":False,
                      "reason":f"{type(exc).__name__}: {exc}",
                      "failure":f"{type(exc).__name__}: {exc}",
                      "seconds":time.monotonic()-t0,
                      "peak_rss_kib":resource.getrusage(resource.RUSAGE_SELF).ru_maxrss})
        receipt(rpath,state)
        raise
    return rpath





def project_gate_exact_v(root: Path, max_project_hours: float = 96) -> Path:
    """Six-scope cost gate for Home-first exact V-pruned projection."""
    out = root / "groups/tcrdist_exact_v_project_cost_gate.json"
    graph_path = root / "groups/full_home/tcrdist_exact_v_graph_preflight.json"
    proof_path = root / "groups/full_home/tcrdist_exact_v_lower_bound_diagnostic.json"
    global_path = root / "groups/tcrdist_all_nodes_receipt.json"
    identity_sha, home_label_sha = manifest_fingerprints(
        root / "config/cohort_samples.tsv")
    report = {"status": "STOP", "method": "tcrdist3",
              "candidate_strategy": "exact_V_cost_pruning",
              "primary_group_scope_count": 6,
              "optional_inner_sensitivity_scope_count": 15,
              "authorized_scopes": ["full_home"] +
                                   [f"outer_{k}" for k in range(5)],
              "radius": 24, "max_project_hours": max_project_hours,
              "manifest_identity_sha256": identity_sha,
              "home_split_label_sha256": home_label_sha,
              "external_projection_locked_until_candidates_frozen": True,
              "script_sha256": sha(Path(__file__))}
    try:
        graph = json.loads(graph_path.read_text())
        proof = json.loads(proof_path.read_text())
        global_receipt = json.loads(global_path.read_text())
        gate(graph.get("status") == "PASS_TRAIN_GRAPH" and
             graph.get("selected_radius") == 24 and
             graph["training_seed_graph_grid"]["24"]["giant_component_fraction"] <= .10,
             "full_home r24 exact training graph failed")
        gate(proof.get("status") == "PASS_TECHNICAL_LOWER_BOUND" and
             all(proof["exhaustive_recall_by_radius"][str(r)]["status"] == "PASS"
                 and proof["exhaustive_recall_by_radius"][str(r)]["missed_by_v_bound"] == 0
                 for r in RADIUS_GRID),
             "exact lower-bound validation failed")
        gate(graph.get("seed_sha256") == proof.get("seed_sha256") and
             graph.get("train_patient_sha256") == proof.get("train_patient_sha256"),
             "proof and graph have different Home training seeds")
        gate(global_receipt.get("status") == "complete" and
             global_receipt.get("schema_version") == 3 and
             proof.get("global_node_file_sha256") == global_receipt.get("output_sha256"),
             "global node universe changed")
        gate(global_receipt.get("manifest_identity_sha256") == identity_sha and
             global_receipt.get("home_split_label_sha256") == home_label_sha,
             "global node source identity changed")
        total_hours = float(proof["six_scope_hours_with_1p5_output_factor"])
        one_hours = total_hours / 6
        gate(total_hours <= max_project_hours and one_hours <= 24,
             "six-scope or one-scope cost exceeds declared limit")
        report.update({
            "status": "PASS_HOME_FIRST",
            "estimated_six_scope_all_node_hours_with_1p5_factor": total_hours,
            "estimated_one_scope_all_node_hours_with_1p5_factor": one_hours,
            "estimated_21_scope_all_node_hours_with_1p5_factor":
                proof["21_scope_hours_with_1p5_output_factor"],
            "global_node_count": global_receipt["eligible_global_nodes"],
            "graph_receipt_sha256": sha(graph_path),
            "proof_receipt_sha256": sha(proof_path),
            "global_nodes_receipt_sha256": sha(global_path),
            "seed_sha256": graph["seed_sha256"],
            "selected_seed_graph_sha256": graph["selected_seed_graph_sha256"],
            "train_patient_sha256": graph["train_patient_sha256"],
            "cost_query_rate_basis": "minimum measured Home/external 5000-node exact projection kernel rate",
            "cost_output_factor": 1.5,
            "projection_population_now": "Home observed nodes only",
            "external_projection_status": "locked until frozen_home_candidates receipt complete",
        })
    except GateStop as exc:
        report["stop_reason"] = str(exc)
    receipt(out, report)
    return out


def _first_part_exact_recall(chunk: pd.DataFrame, seeds: pd.DataFrame,
                             cand: list[np.ndarray], dist: np.ndarray,
                             vcost: np.ndarray, vids: dict[str, int],
                             radius: int, n: int = 128) -> dict:
    """Exhaustive all-seed check on a stratified portion of the first part."""
    total = min(n, len(chunk))
    lengths = chunk.aa.str.len().to_numpy()
    categories = [(lengths <= 12), ((lengths >= 13) & (lengths <= 15)),
                  (lengths >= 16)]
    chosen = []
    for mask in categories:
        ix = np.flatnonzero(mask)
        chosen.extend(map(int, ix[:min(len(ix), max(1, total//3))]))
    chosen = list(dict.fromkeys(chosen))
    if len(chosen) < total:
        chosen.extend(i for i in range(len(chunk)) if i not in set(chosen))
    chosen = chosen[:total]
    q = chunk.iloc[chosen].reset_index(drop=True)
    qm, ql, qv = encode(q, vids)
    sm, sl, sv = encode(seeds, vids)
    qi = np.repeat(np.arange(total, dtype=np.int64), len(seeds))
    si = np.tile(np.arange(len(seeds), dtype=np.int64), total)
    exact = selected(qm, ql, qv, sm, sl, sv, vcost, qi, si).reshape(total, len(seeds))
    candidate_offsets = np.cumsum([0]+[len(x) for x in cand])
    hit = truth = 0
    by_length = {}
    for local, source in enumerate(chosen):
        true = np.flatnonzero(exact[local] <= radius)
        options = cand[source]
        subset = dist[candidate_offsets[source]:candidate_offsets[source+1]]
        found = np.isin(true, options, assume_unique=True)
        truth += len(true)
        hit += int(found.sum())
        if len(true):
            check(bool(found.all()) and int(subset.min()) == int(exact[local].min()),
                  "first-part exact candidate recall or nearest distance failed")
        length = len(q.aa.iloc[local])
        name = "length_le_12" if length <= 12 else ("length_13_15" if length <= 15 else "length_ge_16")
        item = by_length.setdefault(name, {"query_nodes": 0, "directed_neighbors": 0,
                                           "hits": 0})
        item["query_nodes"] += 1
        item["directed_neighbors"] += len(true)
        item["hits"] += int(found.sum())
    check(truth > 0 and hit == truth, "first-part exhaustive recall failed")
    return {"status": "PASS", "exact_query_nodes": total,
            "exact_seed_nodes_per_query": len(seeds),
            "exact_pairs": total*len(seeds),
            "directed_radius_neighbors": truth,
            "radius_neighbor_hits": hit,
            "radius_neighbor_recall": hit/truth,
            "length_strata": by_length,
            "population": "Home source nodes only against all frozen Home training seeds"}




def seed_group_ids(seeds: pd.DataFrame, comp: np.ndarray,
                   sizes: np.ndarray) -> np.ndarray:
    """Content-stable IDs for nonsingleton frozen seed components."""
    check(len(comp) == len(seeds) and sizes.sum() == len(seeds),
          "seed graph does not match frozen seed table")
    names = {}
    for c in np.flatnonzero(sizes >= 2):
        members = seeds.loc[comp == c, KEY]
        first = min("|".join(map(str,r))
                    for r in members.itertuples(index=False,name=None))
        names[int(c)] = "TD_"+hashlib.sha256(first.encode()).hexdigest()[:16]
    return np.array([names.get(int(c), "") for c in comp], dtype=object)

def home_projected_burden_audit(root: Path, scope: str,
                                 membership_path: Path,
                                 known: set[str]) -> dict:
    """Training-only maximum projected family by distinct nodes and reads."""
    out = root / "groups" / scope
    members = pd.read_parquet(membership_path, columns=KEY+["group_id"])
    check(not members.duplicated(KEY).any() and
          members.group_id.astype(str).str.len().gt(0).all(),
          "projected membership has duplicate or empty group")
    nodes = pd.read_parquet(out / "clone_nodes.parquet", columns=KEY)
    node_good = nodes.v_gene.astype(str).map(normal_gene).isin(known)
    scorable_nodes = nodes.loc[node_good, KEY]
    scored = scorable_nodes.merge(members, on=KEY, how="inner",
                                  validate="one_to_one")
    node_counts = scored.group_id.value_counts()
    largest_node_count = int(node_counts.max()) if len(node_counts) else 0
    largest_node_group = str(node_counts.idxmax()) if len(node_counts) else None
    node_fraction = largest_node_count/len(scorable_nodes)
    del nodes, scored, scorable_nodes
    memcheck()
    patient = pd.read_parquet(out / "patient_nodes.parquet",
                              columns=KEY+["read_count"])
    patient_good = patient.v_gene.astype(str).map(normal_gene).isin(known)
    total_scorable_reads = int(patient.loc[patient_good,"read_count"].sum())
    check(total_scorable_reads > 0, "no training-scope scorable patient reads")
    scored_patient = patient.loc[patient_good, KEY+["read_count"]].merge(
        members, on=KEY, how="inner", validate="many_to_one")
    reads = scored_patient.groupby("group_id",sort=False).read_count.sum()
    largest_read_count = int(reads.max()) if len(reads) else 0
    largest_read_group = str(reads.idxmax()) if len(reads) else None
    read_fraction = largest_read_count/total_scorable_reads
    result = {
        "population": "Home training patients in scope only",
        "scorable_train_unique_nodes": int(node_good.sum()),
        "total_scorable_train_reads": total_scorable_reads,
        "largest_group_train_node_count": largest_node_count,
        "largest_group_train_node_id": largest_node_group,
        "largest_group_train_node_fraction": node_fraction,
        "largest_group_train_read_count": largest_read_count,
        "largest_group_train_read_id": largest_read_group,
        "largest_group_train_read_fraction": read_fraction,
        "required_max_fraction": 0.10,
        "membership_sha256": sha(membership_path),
        "clone_nodes_sha256": sha(out/"clone_nodes.parquet"),
        "patient_nodes_sha256": sha(out/"patient_nodes.parquet"),
        "peak_rss_bytes": memcheck(),
    }
    return result

def run_exact_v_home(root: Path, scope: str, per_patient: int, seed: int,
                     chunk_size: int = 10000, max_hours: float = 24,
                     stop_after_first_part: bool = False) -> Path:
    """Home-only frozen r24 projection, SHA-checkpointed in ordered parts."""
    check(1000 <= chunk_size <= 10000 and scope in
          (["full_home"]+[f"outer_{i}" for i in range(5)]),
          "Home exact run scope or chunk outside locked bounds")
    out = root / "groups" / scope
    out.mkdir(parents=True, exist_ok=True)
    rpath = out / "tcrdist_home_projection_receipt.json"
    graph_path = out / "tcrdist_exact_v_graph_preflight.json"
    gate_path = root / "groups/tcrdist_exact_v_project_cost_gate.json"
    all_path = root / "groups/tcrdist_all_nodes.parquet"
    identity_sha, home_label_sha = manifest_fingerprints(
        root / "config/cohort_samples.tsv")
    state = {"status": "initializing", "method": "tcrdist3", "scope": scope,
             "projection_population": "home",
             "candidate_strategy": "exact_V_cost_pruning",
             "radius": 24, "projection_complete": False,
             "external_group_projection_performed": False,
             "external_label_read": False,
             "dictionary_fit_population": "Home training patients of this scope only",
             "scientific_gate_population": "Home training seed graph only",
             "global_nodes_role": "read-only source index, filtered to Home for this run",
             "manifest_identity_sha256": identity_sha,
             "home_split_label_sha256": home_label_sha,
             "script_sha256": sha(Path(__file__))}
    receipt(rpath, state)
    t0 = time.monotonic()
    try:
        project = json.loads(gate_path.read_text())
        graph = json.loads(graph_path.read_text())
        gate(project.get("status") == "PASS_HOME_FIRST" and
             project.get("script_sha256") == sha(Path(__file__)) and
             scope in project["authorized_scopes"],
             "exact V project cost gate does not authorize this run")
        gate(graph.get("status") == "PASS_TRAIN_GRAPH" and
             graph.get("selected_radius") == 24 and
             graph["training_seed_graph_grid"]["24"]["giant_component_fraction"] <= .10,
             "scope training graph at frozen r24 failed")
        desc_path = out / "description.json"
        desc = json.loads(desc_path.read_text())
        gate(graph.get("node_description_sha256") == sha(desc_path) and
             graph.get("selected_seed_graph_sha256") ==
             sha(out / "tcrdist_exact_v_selected_seed_graph.npz"),
             "scope graph or training node description changed")
        check(desc.get("manifest_identity_sha256") == identity_sha and
              desc.get("home_split_label_sha256") == home_label_sha,
              "scope nodes do not match canonical sample identity/Home split")
        seeds, ids = balanced_seeds(root, scope, per_patient, seed)
        seed_path = out / "tcrdist_seed_nodes.parquet"
        check(pd.read_parquet(seed_path).equals(seeds) and
              graph.get("seed_sha256") == sha(seed_path) and
              graph.get("train_patient_sha256") == ids_hash(ids),
              "frozen graph seed or patient set changed")
        known = known_human_v()
        quality = training_scope_quality(root, scope, ids, known)
        gate(quality["training_unscorable_v_node_fraction"] <= .05 and
             quality["training_excluded_vj_read_fraction"] <= .05,
             "training V/J exclusion exceeds predeclared 5% gate")
        all_receipt = json.loads((root/"groups/tcrdist_all_nodes_receipt.json").read_text())
        gate(all_receipt.get("status") == "complete" and
             all_receipt.get("output_sha256") == sha(all_path),
             "global source universe changed")
        full = pd.read_parquet(all_path)
        home = full.loc[(full.cohort_mask.to_numpy(dtype=np.uint8) & 1) > 0].reset_index(drop=True)
        scorable = home.v_gene.astype(str).map(normal_gene).isin(known)
        excluded = home.loc[~scorable].copy()
        excluded["reason"] = "V_absent_from_tcrdist3_human_reference"
        excluded_path = out / "tcrdist_home_unscorable_v_nodes.parquet"
        tmp_ex = excluded_path.with_suffix(".parquet.tmp")
        excluded.to_parquet(tmp_ex, index=False, compression="zstd")
        tmp_ex.replace(excluded_path)
        nodes = home.loc[scorable].reset_index(drop=True)
        check(len(nodes) > 0 and len(nodes)+len(excluded)==len(home),
              "Home node coverage conservation failed")
        vcost, vids = v_penalties(sorted(known))
        sm, sl, sv = encode(seeds, vids)
        with np.load(out / "tcrdist_exact_v_selected_seed_graph.npz") as cached:
            comp = np.asarray(cached["comp"], dtype=np.int64)
            sizes = np.asarray(cached["sizes"], dtype=np.int64)
        check(len(comp)==len(seeds) and sizes.sum()==len(seeds) and
              int(sizes.max()) ==
              graph["training_seed_graph_grid"]["24"]["largest_component_size"],
              "frozen seed graph cache invalid")
        sg = seed_group_ids(seeds, comp, sizes)
        # J is retained in node identity and output; default beta distance has
        # V-loop and CDR3 terms only.
        all_sha = sha(all_path)
        graph_sha = sha(out / "tcrdist_exact_v_selected_seed_graph.npz")
        seed_sha = sha(seed_path)
        state.update({
            "project_cost_gate_sha256": sha(gate_path),
            "graph_preflight_sha256": sha(graph_path),
            "node_description_sha256": sha(desc_path),
            "folds_sha256": sha(root/"folds/folds.tsv"),
            "train_patient_sha256": ids_hash(ids),
            "training_patients": len(ids),
            "training_scope_quality": quality,
            "seed_nodes": len(seeds), "seed_sha256": seed_sha,
            "selected_seed_graph_sha256": graph_sha,
            "seed_graph": graph["training_seed_graph_grid"]["24"],
            "source_nodes_total": len(home),
            "scorable_nodes_total": len(nodes),
            "n_unscorable_v_nodes": len(excluded),
            "unscorable_v_nodes_sha256": sha(excluded_path),
            "unscorable_v_node_fraction": len(excluded)/len(home),
            "all_nodes_sha256": all_sha,
            "tcrdist_reference_db_sha256": reference_hash(),
            "params": {"per_patient":per_patient,"seed":seed,
                       "radius":24,"chunk_size":chunk_size,
                       "candidate_strategy":"exact_V_cost_pruning"},
            "membership_population": _private_literal('private_d1b7091295d8d1d2'),
            "distance_definition": "tcrdist3 human beta default: 3x CDR3 plus V CDR1/2/2.5; J annotation retained",
        })
        receipt(rpath, state)
        parts = out / "tcrdist_home_projection_parts"
        parts.mkdir(exist_ok=True)
        projected = assigned = ambiguous = 0
        first_validation = None
        for start in range(0, len(nodes), chunk_size):
            end = min(start+chunk_size, len(nodes))
            shard = start//chunk_size
            part = parts / f"part_{shard:06d}.parquet"
            pr = parts / f"part_{shard:06d}.json"
            expected = {
                "start":start,"end":end,"seed_sha256":seed_sha,
                "all_nodes_sha256":all_sha,"selected_seed_graph_sha256":graph_sha,
                "radius":24,"candidate_strategy":"exact_V_cost_pruning",
                "script_sha256":sha(Path(__file__)),
            }
            if part.exists() and pr.exists():
                old = json.loads(pr.read_text())
                if all(old.get(k)==v for k,v in expected.items()) and old.get("sha256")==sha(part):
                    projected += end-start
                    assigned += int(old["assigned"])
                    ambiguous += int(old["ambiguous"])
                    if start == 0:
                        gate(old.get("first_part_exact_recall",{}).get("status")=="PASS",
                             "resumed first part lacks exact recall validation")
                        first_validation = old["first_part_exact_recall"]
                    if stop_after_first_part:
                        break
                    continue
            chunk = nodes.iloc[start:end].reset_index(drop=True)
            tc = time.monotonic()
            qm, ql, qv = encode(chunk, vids)
            cand = exact_v_candidates(qv, sv, vcost, 24)
            counts = np.fromiter((len(x) for x in cand), dtype=np.int64,
                                 count=len(cand))
            qi = np.repeat(np.arange(len(chunk),dtype=np.int64),counts)
            si = np.concatenate(cand) if counts.sum() else np.empty(0,dtype=np.int64)
            distances = selected(qm,ql,qv,sm,sl,sv,vcost,qi,si)
            first_extra = {}
            if start == 0:
                first_validation = _first_part_exact_recall(
                    chunk,seeds,cand,distances,vcost,vids,24)
                first_extra["first_part_exact_recall"] = first_validation
            groups = np.array([""]*len(chunk),dtype=object)
            statuses = np.array(["unassigned"]*len(chunk),dtype=object)
            nearest = np.full(len(chunk),-1,dtype=np.int32)
            offset = 0
            for local, options in enumerate(cand):
                d = distances[offset:offset+len(options)]
                offset += len(options)
                if not len(d):
                    continue
                md = int(d.min())
                nearest[local] = md
                if md > 24:
                    continue
                best = {sg[options[k]] for k in np.flatnonzero(d == md)}
                if len(best) == 1 and "" not in best:
                    groups[local] = next(iter(best))
                    statuses[local] = "assigned"
                elif len(best) > 1:
                    statuses[local] = "ambiguous"
            chunk["scope"] = scope
            chunk["method"] = "tcrdist3"
            chunk["group_id"] = groups
            chunk["projection_status"] = statuses
            chunk["nearest_tcrdist"] = nearest
            tmp = part.with_suffix(".parquet.tmp")
            chunk.to_parquet(tmp,index=False,compression="zstd")
            tmp.replace(part)
            rec = expected | {
                "sha256":sha(part),"evaluated":len(chunk),
                "assigned":int((statuses=="assigned").sum()),
                "ambiguous":int((statuses=="ambiguous").sum()),
                "candidate_pairs":len(si),"seconds":time.monotonic()-tc,
                "rss_bytes":memcheck(),
            } | first_extra
            receipt(pr,rec)
            projected += len(chunk)
            assigned += rec["assigned"]
            ambiguous += rec["ambiguous"]
            if start == 0:
                one_est = rec["seconds"]*len(nodes)/len(chunk)/3600
                six_est = 6*rec["seconds"]*len(home)/len(chunk)/3600
                state["first_part_hours_estimate_home_one_scope"] = one_est
                state["first_part_hours_estimate_home_six_scopes"] = six_est
                state["first_part_exact_recall"] = first_validation
                state["first_part_sha256"] = rec["sha256"]
                receipt(rpath,state)
                gate(one_est <= max_hours and six_est <= 96,
                     "first Home part extrapolation exceeds one/six-scope cap")
            print(json.dumps({"scope":scope,"projection_population":"home",
                              "projected":projected,"of":len(nodes),
                              "assigned":assigned}),flush=True)
            if stop_after_first_part:
                break
        if stop_after_first_part:
            state.update({"status":"PAUSED_FIRST_PART","projection_complete":False,
                          "evaluated_nodes_so_far":projected,
                          "first_part_exact_recall":first_validation,
                          "seconds":time.monotonic()-t0,
                          "peak_rss_kib":resource.getrusage(resource.RUSAGE_SELF).ru_maxrss})
            receipt(rpath,state)
            return rpath
        check(projected == len(nodes),"Home projection incomplete")
        summary = consolidate_membership(
            nodes,parts,out,scope,chunk_size,
            "tcrdist_home_group_membership.parquet")
        check(summary["evaluated_nodes"]==projected and
              summary["assigned_nodes"]==assigned and
              summary["ambiguous_nodes"]==ambiguous,
              "Home part and membership counts disagree")
        burden = home_projected_burden_audit(
            root, scope, out / "tcrdist_home_group_membership.parquet",
            known)
        state["projected_group_burden"] = burden
        state["projected_group_gate_population"] = "Home training patients of this scope only"
        gate(burden["largest_group_train_node_fraction"] <= .10 and
             burden["largest_group_train_read_fraction"] <= .10,
             "projected giant group exceeds 10% of training-scope scorable nodes or reads")
        state.update({
            "status":"complete_home_scope","projection_complete":True,
            "evaluated_nodes":projected,
            "scorable_nodes_projected_fraction":projected/len(nodes),
            "assigned_nodes":assigned,"ambiguous_nodes":ambiguous,
            "unassigned_nodes":projected-assigned-ambiguous,
            "assigned_fraction":assigned/projected,
            "unassigned_fraction":(projected-assigned-ambiguous)/projected,
            "multi_group_ambiguity_fraction":ambiguous/projected,
            "group_membership_sha256":summary["group_membership_sha256"],
            "membership_sha256":summary["group_membership_sha256"],
            "group_count":summary["group_count"],
            "max_group_fraction":summary["max_group_fraction"],
            "cohort_projection_diagnostics":summary["cohort_projection_diagnostics"],
            "cohort_projection_diagnostics_role":"Home scoring only; external not projected",
            "external_projected_nodes":0,
            "first_part_exact_recall":first_validation,
            "seconds":time.monotonic()-t0,
            "peak_rss_kib":resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        })
        receipt(rpath,state)
    except GateStop as exc:
        state.update({"status":"STOP_SCIENTIFIC","projection_complete":False,
                      "reason":str(exc),"stop_reason":str(exc),
                      "seconds":time.monotonic()-t0,
                      "peak_rss_kib":resource.getrusage(resource.RUSAGE_SELF).ru_maxrss})
        receipt(rpath,state)
    except Exception as exc:
        state.update({"status":"failed","projection_complete":False,
                      "failure":f"{type(exc).__name__}: {exc}",
                      "seconds":time.monotonic()-t0,
                      "peak_rss_kib":resource.getrusage(resource.RUSAGE_SELF).ru_maxrss})
        receipt(rpath,state)
        raise
    return rpath

def frozen_external_contract(root: Path, home_receipt: dict) -> tuple[set[str], dict]:
    """Validate the locked Home candidates without opening external outcomes."""
    frozen = root / "groups/frozen_home_candidates.tsv"
    frozen_receipt = root / "groups/frozen_home_candidates.json"
    integrated = root / "groups/full_home/group_membership.parquet"
    home = root / "groups/full_home/tcrdist_home_group_membership.parquet"
    gate(all(p.exists() for p in (frozen, frozen_receipt, integrated, home)),
         "frozen Home candidate or membership artifact missing")
    lock = json.loads(frozen_receipt.read_text())
    gate(lock.get("status") == "complete" and
         lock.get("candidate_sha256") == sha(frozen) and
         lock.get("membership_sha256") == sha(integrated),
         "frozen candidate receipt does not match Home artifacts")
    gate(home_receipt.get("status") == "complete_home_scope" and
         home_receipt.get("projection_population") == "home" and
         home_receipt.get("membership_sha256") == sha(home),
         "Home-only tcrdist membership has changed")
    candidates = pd.read_csv(frozen, sep="\t", dtype=str,
                             usecols=["method", "group_id"])
    selected = set(candidates.loc[candidates.method.eq("tcrdist3"),
                                  "group_id"].dropna())
    available = set(pd.read_parquet(home, columns=["group_id"]).group_id)
    gate(selected <= available, "candidate group absent from frozen Home dictionary")
    return selected, {
        "frozen_candidates_sha256": sha(frozen),
        "frozen_candidate_receipt_sha256": sha(frozen_receipt),
        "home_membership_sha256": sha(integrated),
        "home_tcrdist_membership_sha256": sha(home),
        "group_ids_sha256": ids_hash(sorted(selected)),
        "frozen_tcrdist_group_count": len(selected),
        "source_association_sha256": lock.get("source_association_sha256"),
    }


def external_identity_ids(root: Path) -> list[str]:
    """Load only cohort and patient identity columns, never external N."""
    manifest = pd.read_csv(root / "config/cohort_samples.tsv", sep="\t",
                           usecols=["cohort", "patient_key"], dtype=str)
    ids = sorted(set(manifest.loc[manifest.cohort.str.lower().eq("external"),
                                  "patient_key"]))
    check(len(ids) == 165, "expected 165 external patient identities")
    return ids


def run_exact_v_external_frozen(root: Path, chunk_size: int = 10000,
                                max_hours: float = 24,
                                stop_after_first_part: bool = False) -> Path:
    """Use the immutable Home seed graph to project all external QC nodes.

    Every scorable external node is evaluated. Membership outputs retain only
    predeclared Home candidate groups, including hits from novel external AA.
    """
    check(1000 <= chunk_size <= 10000, "external chunk outside safe bounds")
    out = root / "groups/full_home"
    rpath = out / "tcrdist_external_frozen_projection_receipt.json"
    state = {"status":"initializing", "scope":"full_home", "method":"tcrdist3",
             "projection_population":"external", "projection_complete":False,
             "external_label_read":False, "radius":24,
             "candidate_strategy":"exact_V_cost_pruning",
             "dictionary_fit_population":_private_literal('private_8db940e772b243a5'),
             "script_sha256":sha(Path(__file__))}
    receipt(rpath, state)
    t0 = time.monotonic()
    try:
        home_receipt_path = out / "tcrdist_home_projection_receipt.json"
        gate(home_receipt_path.exists(), "Home-only projection receipt missing")
        home_receipt = json.loads(home_receipt_path.read_text())
        selected_groups, hashes = frozen_external_contract(root, home_receipt)
        ext_ids = external_identity_ids(root)
        identity_sha, label_sha = manifest_fingerprints(root / "config/cohort_samples.tsv")
        gate(home_receipt.get("manifest_identity_sha256") == identity_sha and
             home_receipt.get("home_split_label_sha256") == label_sha,
             "source identity or Home split changed after candidate freeze")
        graph_path = out / "tcrdist_exact_v_graph_preflight.json"
        graph = json.loads(graph_path.read_text())
        seed_path = out / "tcrdist_seed_nodes.parquet"
        cache_path = out / "tcrdist_exact_v_selected_seed_graph.npz"
        gate(graph.get("status") == "PASS_TRAIN_GRAPH" and
             graph.get("selected_radius") == 24 and
             graph.get("seed_sha256") == sha(seed_path) and
             graph.get("selected_seed_graph_sha256") == sha(cache_path) and
             home_receipt.get("selected_seed_graph_sha256") == sha(cache_path),
             "frozen Home r24 graph changed")
        seeds = pd.read_parquet(seed_path)
        check(not seeds.duplicated(KEY).any(), "frozen seeds have duplicate nodes")
        with np.load(cache_path) as cached:
            comp = np.asarray(cached["comp"], dtype=np.int64)
            sizes = np.asarray(cached["sizes"], dtype=np.int64)
        seed_groups = seed_group_ids(seeds, comp, sizes)
        known = known_human_v()
        vcost, vids = v_penalties(sorted(known))
        sm, sl, sv = encode(seeds, vids)
        source_path = root / "groups/tcrdist_all_nodes.parquet"
        source_receipt = json.loads((root / "groups/tcrdist_all_nodes_receipt.json").read_text())
        source_sha = sha(source_path)
        gate(source_receipt.get("status") == "complete" and
             source_receipt.get("output_sha256") == source_sha and
             home_receipt.get("all_nodes_sha256") == source_sha,
             "global node source changed")
        full = pd.read_parquet(source_path)
        external = full.loc[(full.cohort_mask.to_numpy(dtype=np.uint8) & 2) > 0].reset_index(drop=True)
        good = external.v_gene.astype(str).map(normal_gene).isin(known)
        excluded = external.loc[~good].copy()
        excluded["reason"] = "V_absent_from_tcrdist3_human_reference"
        excluded_path = out / "tcrdist_external_frozen_unscorable_v_nodes.parquet"
        excluded_tmp = excluded_path.with_suffix(".parquet.tmp")
        excluded.to_parquet(excluded_tmp, index=False, compression="zstd")
        excluded_tmp.replace(excluded_path)
        nodes = external.loc[good].reset_index(drop=True)
        check(len(nodes) > 0 and len(nodes)+len(excluded)==len(external),
              "external node count conservation failed")
        state.update(hashes)
        state.update({
            "train_patient_sha256":graph["train_patient_sha256"],
            "external_identity_sha256":ids_hash(ext_ids),
            "external_patient_count":len(ext_ids),
            "manifest_identity_sha256":identity_sha,
            "home_split_label_sha256":label_sha,
            "home_projection_receipt_sha256":sha(home_receipt_path),
            "seed_sha256":sha(seed_path),
            "selected_seed_graph_sha256":sha(cache_path),
            "all_nodes_sha256":source_sha,
            "source_nodes_total":len(external),
            "scorable_nodes_total":len(nodes),
            "n_unscorable_v_nodes":len(excluded),
            "unscorable_v_nodes_sha256":sha(excluded_path),
            "unscorable_v_node_fraction":len(excluded)/len(external),
            "distance_definition":"tcrdist3 beta default 3x CDR3 plus V CDR1/2/2.5; J annotation retained",
            "frozen_rule":"nearest exact seed at r24, unique nonsingleton component; output only frozen candidate IDs",
            "params":{"radius":24,"chunk_size":chunk_size,
                      "candidate_strategy":"exact_V_cost_pruning"},
        })
        receipt(rpath, state)
        parts = out / "tcrdist_external_frozen_projection_parts"
        parts.mkdir(exist_ok=True)
        projected = assigned = ambiguous = 0
        first_validation = None
        for start in range(0, len(nodes), chunk_size):
            end = min(start+chunk_size, len(nodes))
            part = parts / f"part_{start//chunk_size:06d}.parquet"
            pr = parts / f"part_{start//chunk_size:06d}.json"
            expected = {
                "start":start,"end":end,"seed_sha256":state["seed_sha256"],
                "all_nodes_sha256":source_sha,
                "selected_seed_graph_sha256":state["selected_seed_graph_sha256"],
                "frozen_candidates_sha256":state["frozen_candidates_sha256"],
                "home_membership_sha256":state["home_membership_sha256"],
                "group_ids_sha256":state["group_ids_sha256"],
                "radius":24,"candidate_strategy":"exact_V_cost_pruning",
                "script_sha256":sha(Path(__file__)),
            }
            if part.exists() and pr.exists():
                old = json.loads(pr.read_text())
                if all(old.get(k)==v for k,v in expected.items()) and old.get("sha256")==sha(part):
                    projected += end-start
                    assigned += int(old["assigned"])
                    ambiguous += int(old["ambiguous"])
                    if start == 0:
                        gate(old.get("first_part_exact_recall",{}).get("status")=="PASS",
                             "resumed external first part lacks exhaustive check")
                        first_validation = old["first_part_exact_recall"]
                    if stop_after_first_part:
                        break
                    continue
            chunk = nodes.iloc[start:end].reset_index(drop=True)
            tc = time.monotonic()
            qm, ql, qv = encode(chunk, vids)
            cand = exact_v_candidates(qv, sv, vcost, 24)
            counts = np.fromiter((len(x) for x in cand), dtype=np.int64, count=len(cand))
            qi = np.repeat(np.arange(len(chunk), dtype=np.int64), counts)
            si = np.concatenate(cand) if counts.sum() else np.empty(0,dtype=np.int64)
            distances = selected(qm,ql,qv,sm,sl,sv,vcost,qi,si)
            extra = {}
            if start == 0:
                first_validation = _first_part_exact_recall(
                    chunk,seeds,cand,distances,vcost,vids,24)
                first_validation["population"] = "external nodes against all frozen Home seeds"
                extra["first_part_exact_recall"] = first_validation
            groups = np.array([""]*len(chunk), dtype=object)
            statuses = np.array(["unassigned"]*len(chunk), dtype=object)
            nearest = np.full(len(chunk), -1, dtype=np.int32)
            offset = 0
            for local, options in enumerate(cand):
                d = distances[offset:offset+len(options)]
                offset += len(options)
                if not len(d):
                    continue
                md = int(d.min())
                nearest[local] = md
                if md > 24:
                    continue
                best = {seed_groups[options[k]] for k in np.flatnonzero(d==md)}
                if len(best)==1 and "" not in best:
                    matched = next(iter(best))
                    if matched in selected_groups:
                        groups[local] = matched
                        statuses[local] = "assigned"
                elif len(best)>1:
                    statuses[local] = "ambiguous"
            chunk["scope"] = "full_home"
            chunk["method"] = "tcrdist3"
            chunk["group_id"] = groups
            chunk["projection_status"] = statuses
            chunk["nearest_tcrdist"] = nearest
            tmp = part.with_suffix(".parquet.tmp")
            chunk.to_parquet(tmp,index=False,compression="zstd")
            tmp.replace(part)
            rec = expected | {
                "sha256":sha(part),"evaluated":len(chunk),
                "assigned":int((statuses=="assigned").sum()),
                "ambiguous":int((statuses=="ambiguous").sum()),
                "candidate_pairs":len(si),"seconds":time.monotonic()-tc,
                "rss_bytes":memcheck(),
            } | extra
            receipt(pr,rec)
            projected += len(chunk)
            assigned += rec["assigned"]
            ambiguous += rec["ambiguous"]
            if start == 0:
                estimate = rec["seconds"]*len(nodes)/len(chunk)/3600
                state.update({"first_part_hours_estimate_external":estimate,
                              "first_part_exact_recall":first_validation,
                              "first_part_sha256":rec["sha256"]})
                receipt(rpath,state)
                gate(estimate <= max_hours, "external first part extrapolation exceeds cap")
            print(json.dumps({"scope":"full_home","projection_population":"external",
                              "projected":projected,"of":len(nodes),
                              "assigned":assigned}),flush=True)
            if stop_after_first_part:
                break
        if stop_after_first_part:
            state.update({"status":"PAUSED_FIRST_PART","projection_complete":False,
                          "evaluated_nodes_so_far":projected,
                          "first_part_exact_recall":first_validation,
                          "seconds":time.monotonic()-t0,
                          "peak_rss_kib":resource.getrusage(resource.RUSAGE_SELF).ru_maxrss})
            receipt(rpath,state)
            return rpath
        check(projected == len(nodes), "external projection incomplete")
        summary = consolidate_membership(
            nodes,parts,out,"full_home",chunk_size,
            "tcrdist_external_frozen_membership.parquet")
        check(summary["evaluated_nodes"]==projected and
              summary["assigned_nodes"]==assigned and
              summary["ambiguous_nodes"]==ambiguous,
              "external part/membership counts disagree")
        state.update({
            "status":"complete_external_frozen_scope","projection_complete":True,
            "evaluated_nodes":projected,
            "scorable_nodes_projected_fraction":projected/len(nodes),
            "assigned_nodes":assigned,"ambiguous_nodes":ambiguous,
            "unassigned_nodes":projected-assigned-ambiguous,
            "assigned_fraction":assigned/projected,
            "unassigned_fraction":(projected-assigned-ambiguous)/projected,
            "multi_group_ambiguity_fraction":ambiguous/projected,
            "membership_sha256":summary["group_membership_sha256"],
            "group_count":summary["group_count"],
            "cohort_projection_diagnostics":summary["cohort_projection_diagnostics"],
            "first_part_exact_recall":first_validation,
            "seconds":time.monotonic()-t0,
            "peak_rss_kib":resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        })
        receipt(rpath,state)
    except GateStop as exc:
        state.update({"status":"STOP_FROZEN_RULE","projection_complete":False,
                      "reason":str(exc),"stop_reason":str(exc),
                      "seconds":time.monotonic()-t0,
                      "peak_rss_kib":resource.getrusage(resource.RUSAGE_SELF).ru_maxrss})
        receipt(rpath,state)
    except Exception as exc:
        state.update({"status":"failed","projection_complete":False,
                      "failure":f"{type(exc).__name__}: {exc}",
                      "seconds":time.monotonic()-t0,
                      "peak_rss_kib":resource.getrusage(resource.RUSAGE_SELF).ru_maxrss})
        receipt(rpath,state)
        raise
    return rpath


def preflight(root: Path, scope: str, model: str, per_patient: int,
              radius: int, search_k: int, trees: int, pilot_n: int,
              chunk_size: int, seed: int, max_hours: float) -> Path:
    """Training-only radius-grid and project-cost preflight before projection."""
    import pyarrow.parquet as pq
    check(radius == RADIUS_GRID[-1],
          "formal preflight uses the fixed 16/20/24 radius grid")

    out = root / "groups" / scope
    out.mkdir(parents=True, exist_ok=True)
    rpath = out / "tcrdist_preflight_receipt.json"
    identity_sha, home_label_sha = manifest_fingerprints(
        root / "config/cohort_samples.tsv")
    report = {"scope": scope, "method": "tcrdist3", "status": "running",
              "label_blind": True, "script_sha256": sha(Path(__file__)),
              "global_nodes_role": "read-only global AA/V/J projection universe; never used to fit seed/index/radius",
              "dictionary_fit_population": "only Home training patients in this scope",
              "ann_query_workers": ANN_QUERY_WORKERS,
              "manifest_identity_sha256": identity_sha,
              "home_split_label_sha256": home_label_sha,
              "scientific_dictionary_status": "pending",
              "scientific_gate_population": "Home training patients of this scope only",
              "resource_gate_population": "global scorable Home plus external projection workload",
              "locked_gates": {"max_rss_gib": 128, "max_cpu_threads": 16,
                               "max_one_scope_projection_hours": max_hours,
                               "max_giant_component_fraction": 0.10,
                               "max_unscorable_v_node_fraction": 0.05,
                               "max_excluded_vj_read_fraction": 0.05,
                               "min_nonidentical_aa_recall": 0.95,
                               "min_nonidentical_aa_pairs": 50,
                               "min_stratum_pairs_for_recall_gate": 100,
                               "min_well_powered_stratum_recall": 0.95},
              "params": {"model": model, "per_patient": per_patient,
                         "radius_grid": list(RADIUS_GRID),
                         "radius_selection_rule": "largest radius with train-only exact recall PASS and giant <=0.10",
                         "ann_k": 512, "search_k": search_k,
                         "trees": trees, "pilot_n": pilot_n,
                         "chunk_size": chunk_size, "seed": seed}}
    receipt(rpath, report)
    t0 = time.monotonic()
    try:
        seeds, ids = balanced_seeds(root, scope, per_patient, seed)
        report["train_patient_sha256"] = ids_hash(ids)
        report["training_patients"] = len(ids)
        report["node_description_sha256"] = sha(out / "description.json")
        report["folds_sha256"] = sha(root / "folds/folds.tsv")
        report["seed_nodes"] = len(seeds)
        seed_path = out / "tcrdist_seed_nodes.parquet"
        if not seed_path.exists() or not pd.read_parquet(seed_path).equals(seeds):
            tmp = seed_path.with_suffix(".parquet.tmp")
            seeds.to_parquet(tmp, index=False, compression="zstd")
            tmp.replace(seed_path)
        seed_sha = sha(seed_path)
        report["seed_sha256"] = seed_sha
        known = known_human_v()
        train_quality = training_scope_quality(root, scope, ids, known)
        report["training_scope_quality"] = train_quality
        report["scientific_gate_population"] = "Home training patients of this scope only"
        report["heldout_external_projection_metrics_role"] = "diagnostic_only"
        receipt(rpath, report)
        gate(train_quality["training_unscorable_v_node_fraction"] <= .05 and
             train_quality["training_excluded_vj_read_fraction"] <= .05,
             "training-scope V/J exclusion exceeds 5% node or read gate")
        vcost, vids = v_penalties(sorted(known))
        index, remap, aa_to_nodes, bank, index_sha = index_seeds(
            root, scope, model, seeds, seed_sha, trees, seed)
        report["annoy_index_sha256"] = index_sha
        report["exact_recall_by_radius_k512"] = {}
        for candidate_radius in RADIUS_GRID:
            report["exact_recall_by_radius_k512"][str(candidate_radius)] = exact_pilot(
                seeds, index, remap, aa_to_nodes, bank, vcost, vids,
                candidate_radius, 512, search_k, pilot_n, seed)
            receipt(rpath, report)
        graph_start = time.monotonic()
        graph_grid = frozen_seed_graph_grid(
            seeds, index, remap, aa_to_nodes, bank, vcost, vids,
            RADIUS_GRID, 512, search_k, chunk_size)
        report["seed_graph_grid_k512"] = {
            key: graph["metrics"] for key, graph in graph_grid.items()}
        report["seed_graph_grid_seconds"] = time.monotonic()-graph_start
        receipt(rpath, report)
        selected_radius = select_training_radius(
            report["exact_recall_by_radius_k512"],
            report["seed_graph_grid_k512"])
        report["selected_radius"] = selected_radius
        report["selected_ann_k"] = 512
        report["k256_diagnostic_at_selected_radius"] = exact_pilot(
            seeds, index, remap, aa_to_nodes, bank, vcost, vids,
            selected_radius, 256, search_k, pilot_n, seed)
        selected_graph = graph_grid[str(selected_radius)]
        graph_path = out / "tcrdist_selected_seed_graph.npz"
        graph_tmp = graph_path.with_suffix(".npz.tmp")
        with graph_tmp.open("wb") as handle:
            np.savez_compressed(handle, comp=selected_graph["comp"],
                                sizes=selected_graph["sizes"])
        graph_tmp.replace(graph_path)
        report["selected_seed_graph_sha256"] = sha(graph_path)
        report["selected_seed_graph_path"] = str(graph_path)
        report["scientific_dictionary_status"] = "PASS"
        receipt(rpath, report)
        sm, sl, sv = encode(seeds, vids)
        train_path = root / "groups" / scope / "clone_nodes.parquet"
        pf = pq.ParquetFile(train_path)
        rng = np.random.default_rng(seed + 99)
        samples = []
        for batch in pf.iter_batches(batch_size=100000,
                                     columns=KEY + ["seq_idx"]):
            frame = batch.to_pandas()
            good = frame.loc[frame.v_gene.astype(str).map(normal_gene).isin(known) &
                             frame.j_gene.astype(str).str.startswith("TRBJ"),
                             KEY + ["seq_idx"]]
            if len(good):
                pick = rng.choice(len(good), size=min(100, len(good)), replace=False)
                samples.append(good.iloc[pick])
            memcheck()
        report["training_clone_nodes"] = train_quality["training_clone_nodes_total"]
        report["training_unscorable_v_nodes"] = train_quality["training_unscorable_v_nodes"]
        report["training_unscorable_v_node_fraction"] = train_quality["training_unscorable_v_node_fraction"]
        report["training_excluded_vj_read_fraction"] = train_quality["training_excluded_vj_read_fraction"]
        check(bool(samples), "no training nodes sampled for cost pilot")
        q = pd.concat(samples, ignore_index=True).drop_duplicates(KEY)
        if len(q) > 2000:
            q = q.iloc[np.sort(rng.choice(len(q), 2000, replace=False))]
        q = q.reset_index(drop=True)
        check(len(q) >= 500, f"only {len(q)} training nodes sampled for projection cost pilot")
        qm, ql, qv = encode(q, vids)
        selected(qm[:1], ql[:1], qv[:1], sm[:1], sl[:1], sv[:1],
                 vcost, np.array([0], dtype=np.int64), np.array([0], dtype=np.int64))
        tc = time.monotonic()
        cand = candidate_batch(index, remap, aa_to_nodes, bank,
                               q.seq_idx, 512, search_k)
        qi = np.repeat(np.arange(len(q), dtype=np.int64), [len(x) for x in cand])
        si = np.asarray([j for row in cand for j in row], dtype=np.int64)
        distances = selected(qm, ql, qv, sm, sl, sv, vcost, qi, si)
        projection_seconds = time.monotonic()-tc
        in_radius = []
        offset = 0
        for options in cand:
            d = distances[offset:offset+len(options)]
            in_radius.append(bool(len(d) and d.min() <= selected_radius))
            offset += len(options)
        rate = len(q)/projection_seconds
        all_receipt_path = root / "groups/tcrdist_all_nodes_receipt.json"
        if all_receipt_path.exists():
            all_receipt = json.loads(all_receipt_path.read_text())
            check(all_receipt.get("status") == "complete" and
                  all_receipt.get("schema_version") == 3,
                  "global node receipt is stale or incomplete")
            projected_upper = int(all_receipt["eligible_global_nodes"])
            count_basis = "global unique AA x V x J"
        else:
            manifest = pd.read_csv(root / "config/cohort_samples.tsv", sep="\t",
                                   dtype=str,
                                   usecols=["cohort", "sample_id", "patient_key"])
            projected_upper = sum(pq.ParquetFile(
                root / "data/clone_detail" / f"{row.cohort}__{row.sample_id}.parquet"
            ).metadata.num_rows for row in manifest.itertuples(index=False))
            count_basis = "all clone rows, conservative upper bound"
        report["projection_cost_pilot"] = {
            "query_nodes": len(q), "evaluated_fraction": 1.0,
            "within_selected_training_radius_fraction": float(np.mean(in_radius)),
            "selected_training_radius": selected_radius,
            "seconds": projection_seconds, "nodes_per_second": rate,
            "projection_cost_factor": PROJECTION_COST_FACTOR,
            "estimated_one_scope_hours_with_cost_factor": PROJECTION_COST_FACTOR*projected_upper/rate/3600,
            "estimated_six_scope_hours_with_cost_factor": 6*PROJECTION_COST_FACTOR*projected_upper/rate/3600,
            "estimated_21_scope_hours_with_cost_factor": 21*PROJECTION_COST_FACTOR*projected_upper/rate/3600,
            "node_count_basis": count_basis, "node_count": projected_upper,
            "peak_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}
        gate(PROJECTION_COST_FACTOR*projected_upper/rate/3600 <= max_hours,
             f"one-scope cost-adjusted projection estimate exceeds {max_hours}h")
        report["status"] = "PASS_PRELIMINARY"
        report["resource_status"] = "PASS_ESTIMATE"
        report["limitation"] = ("Training nodes and complete frozen seed index assessed; "
                                "held-out/external coverage is diagnostic, and full projection remains pending.")
    except GateStop as exc:
        if report["scientific_dictionary_status"] == "PASS":
            report["status"] = "STOP_RESOURCE"
            report["resource_status"] = "STOP"
        else:
            report["status"] = "STOP_SCIENTIFIC"
            report["scientific_dictionary_status"] = "STOP"
        report["stop_reason"] = str(exc)
    except Exception as exc:
        report["status"] = "failed"
        report["failure"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        report["seconds"] = time.monotonic()-t0
        report["peak_rss_kib"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        receipt(rpath, report)
    return rpath



def technical_cost_pilot(root: Path, scope: str, model: str,
                         per_patient: int, search_k: int, trees: int,
                         chunk_size: int, pilot_n: int, seed: int) -> Path:
    """Training-seed technical throughput/recall; cannot authorize a dictionary."""
    import pyarrow.parquet as pq

    check(model in (PRIMARY_ANN_MODEL, "tcrbert_mlm"), "unsupported technical bank")
    out = root / "groups" / scope
    out.mkdir(parents=True, exist_ok=True)
    suffix = "primary" if model == PRIMARY_ANN_MODEL else "legacy"
    path = out / f"tcrdist_{suffix}_cost_pilot.json"
    report = {
        "status": "running", "method": "tcrdist3", "scope": scope,
        "ann_model": model,
        "purpose": "training-only technical resource and exact-recall forecast",
        "seed_training_only": True,
        "global_nodes_role": "read-only projection-count universe; never enters seed/index fitting",
        "ann_query_workers": ANN_QUERY_WORKERS,
        "cannot_lock_radius_or_ann_k": True,
        "cannot_authorize_formal_projection": True,
        "script_sha256": sha(Path(__file__)),
        "manifest_identity_sha256": manifest_fingerprints(
            root / "config/cohort_samples.tsv")[0],
        "params": {"model": model, "per_patient": per_patient, "ann_k": 512,
                   "search_k": search_k, "trees": trees,
                   "chunk_size": chunk_size, "pilot_n": pilot_n, "seed": seed},
    }
    receipt(path, report)
    t0 = time.monotonic()
    try:
        seeds, ids = balanced_seeds(root, scope, per_patient, seed)
        seed_path = out / "tcrdist_seed_nodes.parquet"
        if not seed_path.exists() or not pd.read_parquet(seed_path).equals(seeds):
            tmp = seed_path.with_suffix(".parquet.tmp")
            seeds.to_parquet(tmp, index=False, compression="zstd")
            tmp.replace(seed_path)
        seed_sha = sha(seed_path)
        known = known_human_v()
        vcost, vids = v_penalties(sorted(known))
        index, remap, aa_to_nodes, bank, index_sha = index_seeds(
            root, scope, model, seeds, seed_sha, trees, seed)
        report.update({
            "train_patient_sha256": ids_hash(ids),
            "training_patient_hash_verified_against_description": True,
            "bank_meta_sha256": sha(root / "embeddings" / model / "meta.json"),
            "bank_verification_sha256": (
                sha(root / "embeddings" / model / "verification.json")
                if model == PRIMARY_ANN_MODEL else None),
            "seed_nodes": len(seeds),
            "seed_distinct_aa": index.get_n_items(),
            "seed_sha256": seed_sha,
            "annoy_index_sha256": index_sha,
        })
        rng = np.random.default_rng(seed + 101)
        samples = []
        source = pq.ParquetFile(out / "clone_nodes.parquet")
        for batch in source.iter_batches(batch_size=100000,
                                         columns=KEY + ["seq_idx"]):
            frame = batch.to_pandas()
            good = frame.loc[frame.v_gene.astype(str).map(normal_gene).isin(known) &
                             frame.j_gene.astype(str).str.startswith("TRBJ"),
                             KEY + ["seq_idx"]]
            if len(good):
                pick = rng.choice(len(good), size=min(100, len(good)), replace=False)
                samples.append(good.iloc[pick])
            memcheck()
        check(bool(samples), "no scorable training nodes for technical throughput")
        q = pd.concat(samples, ignore_index=True).drop_duplicates(KEY)
        if len(q) > 2000:
            q = q.iloc[np.sort(rng.choice(len(q), 2000, replace=False))]
        q = q.reset_index(drop=True)
        check(len(q) >= 500, "fewer than 500 training cost-pilot nodes")
        sm, sl, sv = encode(seeds, vids)
        qm, ql, qv = encode(q, vids)
        selected(qm[:1], ql[:1], qv[:1], sm[:1], sl[:1], sv[:1],
                 vcost, np.array([0], dtype=np.int64),
                 np.array([0], dtype=np.int64))
        tc = time.monotonic()
        cand = candidate_batch(index, remap, aa_to_nodes, bank,
                               q.seq_idx, 512, search_k)
        qi = np.repeat(np.arange(len(q), dtype=np.int64),
                       [len(x) for x in cand])
        si = np.asarray([j for row in cand for j in row], dtype=np.int64)
        distances = selected(qm, ql, qv, sm, sl, sv, vcost, qi, si)
        check(len(distances) == len(si), "technical cost pilot distance count mismatch")
        elapsed = time.monotonic()-tc
        rate = len(q)/elapsed
        all_receipt_path = root / "groups/tcrdist_all_nodes_receipt.json"
        if all_receipt_path.exists():
            all_receipt = json.loads(all_receipt_path.read_text())
            check(all_receipt.get("status") == "complete" and
                  all_receipt.get("schema_version") == 3,
                  "global node receipt is stale or incomplete")
            node_count = int(all_receipt["eligible_global_nodes"])
            basis = "exact global eligible AA x V x J nodes"
        else:
            manifest = pd.read_csv(root / "config/cohort_samples.tsv",
                                   sep="\t", dtype=str,
                                   usecols=["cohort", "sample_id", "patient_key"])
            node_count = sum(pq.ParquetFile(
                root / "data/clone_detail" /
                f"{row.cohort}__{row.sample_id}.parquet"
            ).metadata.num_rows for row in manifest.itertuples(index=False))
            basis = "all source clone rows; upper bound on unique eligible nodes"
        report.update({
            "status": "TECHNICAL_ONLY",
            "sampled_training_query_nodes": len(q),
            "candidate_pairs_refined_exactly": len(si),
            "projection_kernel_seconds": elapsed,
            "projection_kernel_nodes_per_second": rate,
            "node_count_basis": basis,
            "node_count": node_count,
            "projection_cost_factor": PROJECTION_COST_FACTOR,
            "six_scope_projection_hours_at_measured_rate": 6*node_count/rate/3600,
            "21_scope_projection_hours_at_measured_rate": 21*node_count/rate/3600,
            "six_scope_hours_with_output_cost_factor": 6*PROJECTION_COST_FACTOR*node_count/rate/3600,
            "21_scope_hours_with_output_cost_factor": 21*PROJECTION_COST_FACTOR*node_count/rate/3600,
            "limitation": "Training-only bounded throughput and small exact-neighbor diagnostics cannot authorize full-scope projection. A formal training seed graph, radius selection and six-scope project cost gate remain required.",
        })
        report["exact_recall_by_radius_k512"] = {
            str(r): exact_pilot(seeds, index, remap, aa_to_nodes, bank,
                                vcost, vids, r, 512, search_k, pilot_n, seed)
            for r in RADIUS_GRID
        }
    except Exception as exc:
        report["status"] = "failed"
        report["reason"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        report["seconds"] = time.monotonic()-t0
        report["peak_rss_kib"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        receipt(path, report)
    return path


def project_gate(root: Path, project_scopes: int, max_project_hours: float) -> Path:
    """Block full projection until a whole-project upper bound is accepted."""
    out = root / "groups/tcrdist_project_cost_gate.json"
    pre_path = root / "groups/full_home/tcrdist_preflight_receipt.json"
    all_path = root / "groups/tcrdist_all_nodes_receipt.json"
    identity_sha, home_label_sha = manifest_fingerprints(
        root / "config/cohort_samples.tsv")
    report = {"status": "STOP", "method": "tcrdist3",
              "manifest_identity_sha256": identity_sha,
              "home_split_label_sha256": home_label_sha,
              "project_scopes": project_scopes,
              "primary_group_scope_count": 6,
              "optional_inner_sensitivity_scope_count": 15,
              "max_project_hours": max_project_hours,
              "authorized_scopes": (["full_home"] + [f"outer_{k}" for k in range(5)] +
                                    ([f"inner_{o}_{i}" for o in range(5) for i in range(3)]
                                     if project_scopes == 21 else [])),
              "script_sha256": sha(Path(__file__)),
              "label_blind": True}
    try:
        gate(project_scopes in (6, 21), "project scope count must be 6 or 21")
        gate(pre_path.exists() and all_path.exists(),
             "full_home preflight and exact global node receipt required")
        pre = json.loads(pre_path.read_text())
        all_receipt = json.loads(all_path.read_text())
        gate(pre["status"] == "PASS_PRELIMINARY" and
             pre.get("scientific_dictionary_status") == "PASS" and
             pre.get("resource_status") == "PASS_ESTIMATE" and
             pre.get("manifest_identity_sha256") == identity_sha and
             pre.get("home_split_label_sha256") == home_label_sha and
             all_receipt.get("manifest_identity_sha256") == identity_sha and
             all_receipt.get("home_split_label_sha256") == home_label_sha,
             "full_home scientific and resource preflight have not both passed")
        gate(pre.get("params", {}).get("model") == PRIMARY_ANN_MODEL,
             "primary dual-task AA-mean bank preflight has not passed")
        gate(pre.get("params", {}).get("radius_grid") == list(RADIUS_GRID) and
             pre.get("selected_ann_k") == 512 and
             pre.get("selected_radius") in RADIUS_GRID and
             bool(pre.get("selected_seed_graph_sha256")),
             "primary-bank train-only radius-grid selection is incomplete")
        gate(all_receipt["status"] == "complete" and all_receipt.get("schema_version") == 3,
             "global node preparation or patient-level V/J coverage not complete")
        gate(pre["script_sha256"] == report["script_sha256"],
             "preflight script version differs from current implementation")
        node_count = int(all_receipt["eligible_global_nodes"])
        rate = float(pre["projection_cost_pilot"]["nodes_per_second"])
        one_scope = node_count/rate/3600
        preflight_hours = float(pre["seconds"])/3600
        selected_hours = project_scopes*(PROJECTION_COST_FACTOR*one_scope+preflight_hours)
        report.update({"ann_model": PRIMARY_ANN_MODEL,
                       "full_home_training_selected_radius": pre["selected_radius"],
                       "radius_selection_population": "full_home descriptive only; each fold needs its own training-only preflight and radius",
                       "exact_global_nodes": node_count,
                       "all_nodes_sha256": all_receipt["output_sha256"],
                       "home_preflight_sha256": sha(pre_path),
                       "full_home_measured_nodes_per_second": rate,
                       "estimated_one_scope_hours": one_scope,
                       "estimated_six_scope_hours": one_scope*6,
                       "estimated_21_scope_hours": one_scope*21,
                       "full_home_preflight_hours": preflight_hours,
                       "projection_cost_factor": PROJECTION_COST_FACTOR,
                       "estimated_six_scope_hours_including_preflight": 6*(PROJECTION_COST_FACTOR*one_scope+preflight_hours),
                       "estimated_21_scope_hours_including_preflight": 21*(PROJECTION_COST_FACTOR*one_scope+preflight_hours),
                       "estimated_selected_project_hours": selected_hours,
                       "estimate_basis": "full_home complete seed-index preflight time plus 1.25x exact global tuple projection kernel cost for output overhead, multiplied by scope count; first full chunk rechecks actual rate"})
        gate(selected_hours <= max_project_hours,
             f"{project_scopes}-scope preflight plus projection estimate {selected_hours:.1f}h exceeds {max_project_hours}h cap")
        report["status"] = "PASS"
        report["limitation"] = ("Every authorized scope still requires its own exact recall, "
                                 "giant-component, training-scope V/J coverage and per-scope wall gate. "
                                "Six-scope analysis reuses outer dictionaries in inner validation "
                                "and is therefore unlabeled transductive at that level.")
    except GateStop as exc:
        report["stop_reason"] = str(exc)
    receipt(out, report)
    return out


def pilot_shard(root: Path, samples: list[str], shard_start: int, shard_end: int,
                radius: int, ann_k: int, search_k: int, trees: int,
                seed: int, per_library: int = 600) -> Path:
    """Bounded, label-blind exact recall test across selected Home libraries.

    This checks a small Annoy index built only from sampled nodes. A formal
    scope must repeat exact_pilot against its complete frozen seed index.
    """
    import pyarrow.parquet as pq
    from scipy.sparse import csr_matrix
    from scipy.sparse.csgraph import connected_components

    check(2 <= len(samples) <= 4 and all(x.startswith("H") for x in samples),
          "pilot needs 2..4 Home sample IDs")
    check(shard_start % 50000 == 0 and shard_end == shard_start + 50000 and
          shard_start // 1000000 == (shard_end - 1) // 1000000,
          "pilot must use one complete 50k shard in a single union row group")
    cohort = pd.read_csv(root / "config/cohort_samples.tsv", sep="\t",
                         dtype=str, usecols=["cohort", "sample_id", "patient_key"])
    home_samples = set(cohort.loc[cohort.cohort == "home", "sample_id"])
    check(set(samples) <= home_samples, "non-Home or unknown pilot sample")
    union_path = root / "data/aa_union.parquet"
    union = pq.ParquetFile(union_path)
    row_group = shard_start // 1000000
    aa = union.read_row_group(row_group, columns=["aa", "seq_idx"]).to_pandas()
    aa = aa.loc[(aa.seq_idx >= shard_start) & (aa.seq_idx < shard_end)]
    check(len(aa) == 50000 and aa.seq_idx.is_unique, "AA union range not complete")
    bank_path = root / "embeddings/tcrbert_mlm/shards" / f"{shard_start:012d}_{shard_end:012d}.npy"
    check(bank_path.exists(), f"completed BERT shard missing: {bank_path}")
    bank = np.load(bank_path, mmap_mode="r")
    check(bank.shape[0] == 50000, "BERT shard row count mismatch")
    known = known_human_v()
    rng = np.random.default_rng(seed)
    pieces, source_counts, paths = [], {}, {}
    excluded_nodes = excluded_reads = scoped_reads = 0
    for sample in samples:
        path = root / "data/clone_detail" / f"home__{sample}.parquet"
        check(path.exists(), f"clone source missing: {path}")
        paths[sample] = sha(path)
        d = pd.read_parquet(path, columns=KEY + ["qc_pass", "read_count"])
        d = d.loc[d.qc_pass == True].merge(aa, on="aa", how="inner")
        scoped_reads += int(d.read_count.sum())
        d = d.loc[d.v_gene.astype(str).str.startswith("TRBV") &
                  d.j_gene.astype(str).str.startswith("TRBJ")]
        bad = ~d.v_gene.astype(str).map(normal_gene).isin(known)
        excluded_nodes += len(d.loc[bad, KEY].drop_duplicates())
        excluded_reads += int(d.loc[bad, "read_count"].sum())
        d = d.loc[~bad, KEY + ["seq_idx"]].drop_duplicates(KEY).copy()
        source_counts[sample] = len(d)
        if len(d) > per_library:
            d = d.iloc[np.sort(rng.choice(len(d), per_library, replace=False))]
        d["source_library"] = sample
        pieces.append(d)
    nodes = pd.concat(pieces, ignore_index=True).drop_duplicates(KEY).reset_index(drop=True)
    check(len(nodes) >= 1100, f"pilot has only {len(nodes)} distinct scorable nodes")
    pick = rng.permutation(len(nodes))[:min(1400, len(nodes))]
    nodes = nodes.iloc[pick].reset_index(drop=True)
    nseed = min(1000, int(len(nodes) * .72))
    nquery = min(400, len(nodes) - nseed)
    seeds = nodes.iloc[:nseed].copy().reset_index(drop=True)
    query = nodes.iloc[nseed:nseed+nquery].copy().reset_index(drop=True)
    for frame in (seeds, query):
        frame["seq_idx"] -= shard_start
    vcost, vids = v_penalties(list(set(nodes.v_gene)))
    sm, sl, sv = encode(seeds, vids)
    qm, ql, qv = encode(query, vids)
    by_aa: dict[int, list[int]] = defaultdict(list)
    for i, idx in enumerate(seeds.seq_idx):
        by_aa[int(idx)].append(i)
    unique = sorted(by_aa)
    index = AnnoyIndex(bank.shape[1], "angular")
    index.set_seed(seed)
    for i, idx in enumerate(unique):
        v = np.asarray(bank[idx], dtype=np.float32)
        check(np.isfinite(v).all(), f"nonfinite bank row {idx}")
        index.add_item(i, v.tolist())
    index.build(trees, n_jobs=min(4, os.cpu_count() or 1))
    remap = {i: by_aa[x] for i, x in enumerate(unique)}
    if len(query) and len(seeds):
        selected(qm[:1], ql[:1], qv[:1], sm[:1], sl[:1], sv[:1],
                 vcost, np.array([0], dtype=np.int64), np.array([0], dtype=np.int64))
    qi = np.repeat(np.arange(nquery, dtype=np.int64), nseed)
    si = np.tile(np.arange(nseed, dtype=np.int64), nquery)
    exact_start = time.monotonic()
    exact = selected(qm, ql, qv, sm, sl, sv, vcost, qi, si).reshape(nquery, nseed)
    exact_seconds = time.monotonic() - exact_start
    nonident = query.aa.to_numpy()[:, None] != seeds.aa.to_numpy()[None, :]
    truth = (exact <= radius) & nonident
    denom = int(truth.sum())
    recalls = {}
    candidates_by_k = {}
    for k in sorted(set([128, ann_k, 512])):
        t0 = time.monotonic()
        cand = [candidates(index, remap, by_aa, bank, int(idx), k, search_k)
                for idx in query.seq_idx]
        elapsed = time.monotonic() - t0
        hits = np.array([int(truth[i, c].sum()) for i, c in enumerate(cand)])
        totals = truth.sum(axis=1).astype(int)
        recalls[str(k)] = {
            "nonidentical_aa_directed_neighbors": denom,
            "hits": int(hits.sum()),
            "recall": float(hits.sum()/denom) if denom else None,
            "candidate_seconds": elapsed,
            "query_nodes_per_second": nquery/elapsed if elapsed else None,
        }
        if k == ann_k:
            candidates_by_k = cand
            strata = {}
            qlen = query.aa.str.len().to_numpy()
            strata_masks = {
                "length_le_12": qlen <= 12,
                "length_13_15": (qlen >= 13) & (qlen <= 15),
                "length_ge_16": qlen >= 16,
            }
            for v in query.v_gene.value_counts().head(8).index:
                strata_masks[f"v_{v}"] = query.v_gene.to_numpy() == v
            for name, mask in strata_masks.items():
                num = int(totals[mask].sum())
                strata[name] = {"query_nodes": int(mask.sum()),
                                "directed_neighbors": num,
                                "recall": float(hits[mask].sum()/num) if num else None}
    tc = time.monotonic()
    counts = np.array([len(c) for c in candidates_by_k], dtype=np.int32)
    qi2 = np.repeat(np.arange(nquery, dtype=np.int64), counts)
    si2 = np.array([j for row in candidates_by_k for j in row], dtype=np.int64)
    ann_dist = selected(qm, ql, qv, sm, sl, sv, vcost, qi2, si2)
    best = np.full(nquery, -1, dtype=np.int32)
    offset = 0
    for i, n in enumerate(counts):
        if n: best[i] = int(ann_dist[offset:offset+n].min())
        offset += n
    projection_seconds = time.monotonic() - tc + recalls[str(ann_k)]["candidate_seconds"]
    gstart = time.monotonic()
    qg = np.repeat(np.arange(nseed, dtype=np.int64), nseed)
    sg = np.tile(np.arange(nseed, dtype=np.int64), nseed)
    seed_exact = selected(sm, sl, sv, sm, sl, sv, vcost, qg, sg).reshape(nseed, nseed)
    graph = seed_exact <= radius
    np.fill_diagonal(graph, False)
    ncomp, comps = connected_components(csr_matrix(graph), directed=False)
    sizes = np.bincount(comps, minlength=ncomp)
    exact_grouped_query = 0
    for i in range(nquery):
        eligible = np.flatnonzero(exact[i] <= radius)
        if len(eligible) and any(sizes[comps[j]] >= 2 for j in eligible):
            exact_grouped_query += 1
    graph_seconds = time.monotonic() - gstart
    status = "pass" if denom >= 50 and recalls[str(ann_k)]["recall"] is not None and recalls[str(ann_k)]["recall"] >= .95 else "fail"
    out = root / "groups/pilot" / f"tcrdist_{'_'.join(samples)}_{shard_start:012d}.json"
    receipt(out, {
        "status": status, "method": "tcrdist3", "scope": "bounded_label_blind_pilot",
        "samples": samples, "shard_start": shard_start, "shard_end": shard_end,
        "script_sha256": sha(Path(__file__)), "clone_detail_sha256": paths,
        "bert_shard_sha256": sha(bank_path), "reference_db_sha256": reference_hash(),
        "params": {"radius": radius, "ann_k": ann_k, "search_k": search_k,
                   "trees": trees, "seed": seed, "per_library": per_library},
        "scoped_node_counts_per_library": source_counts,
        "sampled_nodes": len(nodes), "sampled_libraries": nodes.source_library.value_counts().to_dict(),
        "seed_nodes": nseed, "query_nodes": nquery,
        "seed_distinct_aa": len(unique),
        "unscorable_v_nodes_in_selected_shard_libraries": excluded_nodes,
        "unscorable_v_reads_in_selected_shard_libraries": excluded_reads,
        "all_qc_reads_in_selected_shard_libraries": scoped_reads,
        "unscorable_v_read_fraction": excluded_reads/scoped_reads if scoped_reads else None,
        "recall_by_k": recalls, "recall_by_query_stratum_at_selected_k": strata,
        "exact_query_seed_seconds": exact_seconds, "seed_exact_graph_seconds": graph_seconds,
        "seed_exact_graph_giant_fraction": float(sizes.max()/nseed),
        "seed_exact_graph_singleton_fraction": float((sizes[comps]==1).mean()),
        "exact_seed_micrograph_query_grouped_fraction": exact_grouped_query/nquery,
        "query_evaluated_fraction": 1.0,
        "query_within_radius_ann_fraction": float(((best >= 0) & (best <= radius)).mean()),
        "query_unassigned_ann_fraction": float(((best < 0) | (best > radius)).mean()),
        "query_projection_seconds": projection_seconds,
        "query_projection_nodes_per_second": nquery/projection_seconds if projection_seconds else None,
        "estimated_global_aa_lower_bound_hours": 23974109/(nquery/projection_seconds)/3600 if projection_seconds else None,
        "peak_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "limitation": "Legacy tcrbert_mlm small sampled shard is a technical diagnostic only; it cannot lock primary dual-task AA-mean ANN k/radius. Full scope requires fresh exact recall against each complete frozen training seed index. Grouped fraction is an exact pilot micrograph diagnostic, not full biological coverage.",
    })
    return out

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("command",choices=["prepare","reference-coverage","run","pilot-shard","preflight","project-gate","technical-cost-only","exact-v-diagnostic","exact-v-graph","project-gate-exact-v","run-exact-home","run-exact-external-frozen"])
    p.add_argument("--root",type=Path,default=Path(_release_path('path_1304218e86d2e596')))
    p.add_argument("--scope",default="full_home")
    p.add_argument("--model",default=PRIMARY_ANN_MODEL)
    p.add_argument("--seed-per-patient",type=int,default=512)
    p.add_argument("--radius",type=int,default=24)
    p.add_argument("--ann-k",type=int,default=512)
    p.add_argument("--search-k",type=int,default=8192)
    p.add_argument("--ann-trees",type=int,default=32)
    p.add_argument("--pilot-n",type=int,default=2000)
    p.add_argument("--chunk-size",type=int,default=10000)
    p.add_argument("--seed",type=int,default=20260929)
    p.add_argument("--max-hours",type=float,default=24)
    p.add_argument("--project-scopes",type=int,default=6)
    p.add_argument("--max-project-hours",type=float,default=96)
    p.add_argument("--pilot-samples",default="")
    p.add_argument("--shard-start",type=int,default=0)
    p.add_argument("--per-library",type=int,default=600)
    p.add_argument("--stop-after-first-part",action="store_true")
    a=p.parse_args()
    if a.command=="prepare": print(prepare(a.root))
    elif a.command=="reference-coverage": print(reference_coverage(a.root))
    elif a.command=="project-gate":
        print(project_gate(a.root,a.project_scopes,a.max_project_hours))
    elif a.command=="preflight":
        print(preflight(a.root,a.scope,a.model,a.seed_per_patient,a.radius,
                        a.search_k,a.ann_trees,a.pilot_n,a.chunk_size,
                        a.seed,a.max_hours))
    elif a.command=="project-gate-exact-v":
        print(project_gate_exact_v(a.root,a.max_project_hours))
    elif a.command=="run-exact-home":
        print(run_exact_v_home(a.root,a.scope,a.seed_per_patient,a.seed,
                               a.chunk_size,a.max_hours,a.stop_after_first_part))
    elif a.command=="run-exact-external-frozen":
        print(run_exact_v_external_frozen(a.root,a.chunk_size,a.max_hours,
                                          a.stop_after_first_part))
    elif a.command=="exact-v-graph":
        print(exact_v_graph_diagnostic(a.root,a.scope,a.seed,
                                       min(a.chunk_size,1000),
                                       a.seed_per_patient))
    elif a.command=="exact-v-diagnostic":
        print(exact_v_lower_bound_diagnostic(a.root,a.scope,a.seed,a.pilot_n))
    elif a.command=="technical-cost-only":
        print(technical_cost_pilot(a.root,a.scope,a.model,a.seed_per_patient,
                                   a.search_k,a.ann_trees,a.chunk_size,
                                   a.pilot_n,a.seed))
    elif a.command=="pilot-shard":
        print(pilot_shard(a.root,[x for x in a.pilot_samples.split(",") if x],
                          a.shard_start,a.shard_start+50000,a.radius,a.ann_k,
                          a.search_k,a.ann_trees,a.seed,a.per_library))
    else: print(run(a.root,a.scope,a.model,a.seed_per_patient,a.radius,
                    a.ann_k,a.search_k,a.ann_trees,a.pilot_n,a.chunk_size,
                    a.seed,a.max_hours))


if __name__=="__main__": main()
