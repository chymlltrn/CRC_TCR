#!/usr/bin/env python3
"""Fold-scoped TCR family inputs, exact membership projection, and patient tests.

A full-Home dictionary is descriptive only. Fold-local dictionaries are built
from outer-training patients and never from held-out patients. V/J-aware family
membership remains at (AA,V,J); AA-only ablation has an explicit safety gate.
"""
from __future__ import annotations
from release_runtime import private_literal as _private_literal, release_path as _release_path

import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import resource
import subprocess
import time

for _name in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_name] = str(min(16, int(os.environ.get(_name, "16"))))

import numpy as np
import pandas as pd
import pyarrow.dataset as ds
from scipy.stats import fisher_exact, mannwhitneyu
from statsmodels.stats.multitest import multipletests

LEGACY_GLIPH_ROOT = Path(_release_path('path_2c83fb1164fd9d90'))
KEY = ["aa", "v_gene", "j_gene"]
RAW_COLS = ["cohort", "patient_key", "aa", "v_gene", "j_gene", "read_count", "qc_pass"]


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def _json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False, sort_keys=True) + "\n")
    tmp.replace(path)


def _patient_hash(patients: set[str]) -> str:
    return hashlib.sha256(("\n".join(sorted(patients)) + "\n").encode()).hexdigest()


def _manifest_fingerprints(root: Path) -> tuple[str, str]:
    """Canonical identities and Home-only labels; external labels are unread."""
    identities: list[list[str]] = []
    home_labels: list[list[str]] = []
    with (root / "config/cohort_samples.tsv").open(newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        if not {"cohort", "sample_id", "patient_key", "recorded_n_binary"} <= set(reader.fieldnames or []):
            raise ValueError("Cohort manifest lacks identity or Home split columns")
        for row in reader:
            cohort = row["cohort"].lower()
            identities.append([cohort, row["sample_id"], row["patient_key"]])
            if cohort == "home":
                home_labels.append([row["patient_key"], row["recorded_n_binary"]])
    identity = pd.DataFrame(identities, columns=["cohort", "sample_id", "patient_key"])
    identity_sha = hashlib.sha256(identity.sort_values(
        ["cohort", "sample_id", "patient_key"]).to_csv(
            sep="\t", index=False, lineterminator="\n").encode("utf-8")).hexdigest()
    label_sha = hashlib.sha256(json.dumps(sorted(home_labels), ensure_ascii=False,
                                          separators=(",", ":")).encode("utf-8")).hexdigest()
    return identity_sha, label_sha


def _manifest_identity_sha256(root: Path) -> str:
    return _manifest_fingerprints(root)[0]


def _home_split_label_sha256(root: Path) -> str:
    return _manifest_fingerprints(root)[1]


def _stem(value: object) -> str:
    return str(value).split("*", 1)[0].strip()


def _scope_patients(root: Path, scope: str) -> set[str]:
    folds = pd.read_csv(root / "folds/folds.tsv", sep="\t", dtype={"patient_key": str})
    if scope == "full_home":
        return set(folds["patient_key"])
    if scope.startswith("outer_") and scope[6:].isdigit():
        outer = int(scope[6:])
        if outer not in range(5):
            raise ValueError("outer fold must be 0..4")
        part = folds.loc[folds.outer_fold.eq(outer) & folds.partition.eq("train"), "patient_key"]
    elif scope.startswith("inner_"):
        tokens = scope.split("_")
        if len(tokens) != 3 or not all(token.isdigit() for token in tokens[1:]):
            raise ValueError("inner scope must be inner_<outer 0..4>_<inner 0..2>")
        outer, inner = map(int, tokens[1:])
        if outer not in range(5) or inner not in range(3):
            raise ValueError("inner scope fold index out of range")
        part = folds.loc[folds.outer_fold.eq(outer) & folds.partition.eq("train") &
                         folds.inner_fold.ne(inner), "patient_key"]
    else:
        raise ValueError("scope must be full_home, outer_0..outer_4 or inner_<outer>_<inner>")
    if part.empty:
        raise ValueError(f"No training patients for {scope}")
    return set(part)


def _load_patient_nodes(root: Path, selected: set[str], cohort: str = "home") -> pd.DataFrame:
    manifest_path = root / "config/cohort_samples.tsv"
    manifest = pd.read_csv(manifest_path, sep="\t", dtype=str,
                           usecols=["cohort", "sample_id", "patient_key"])
    relevant = manifest.loc[manifest.cohort.eq(cohort) & manifest.patient_key.isin(selected)]
    if set(relevant.patient_key) != selected:
        raise ValueError(f"{cohort} manifest lacks selected patient identities")
    paths = [root / "data/clone_detail" / f"{row.cohort}__{row.sample_id}.parquet"
             for row in relevant.itertuples(index=False)]
    if not paths or not all(path.is_file() for path in paths):
        raise FileNotFoundError(f"Selected {cohort} clone detail files are incomplete")
    dataset = ds.dataset([str(path) for path in paths], format="parquet")
    missing = set(RAW_COLS) - set(dataset.schema.names)
    if missing:
        raise ValueError(f"clone_detail missing columns: {sorted(missing)}")
    condition = (ds.field("cohort") == cohort) & (ds.field("qc_pass") == True)
    parts = []
    scanner = dataset.scanner(columns=RAW_COLS, filter=condition, batch_size=131072)
    for batch in scanner.to_batches():
        frame = batch.to_pandas()
        frame = frame.loc[frame.patient_key.astype(str).isin(selected), ["patient_key"] + KEY + ["read_count"]]
        if frame.empty:
            continue
        frame = frame.dropna(subset=["aa"])
        for col in KEY:
            frame[col] = frame[col].fillna("").astype(str)
        frame["read_count"] = pd.to_numeric(frame["read_count"], errors="coerce").fillna(0)
        frame = frame.loc[frame.read_count.gt(0)]
        if not frame.empty:
            parts.append(frame.groupby(["patient_key"] + KEY, as_index=False, sort=False).read_count.sum())
    if not parts:
        raise ValueError(f"No QC-passing {cohort} clone rows for selected patients")
    return pd.concat(parts, ignore_index=True).groupby(["patient_key"] + KEY, as_index=False, sort=False).read_count.sum()


def prepare_nodes(root: Path, scope: str) -> Path:
    """Build full QC AA/V/J nodes with bounded per-sample and hash-bucket memory."""
    import tempfile
    import pyarrow as pa
    import pyarrow.parquet as pq

    selected = _scope_patients(root, scope)
    out = root / "groups" / scope
    out.mkdir(parents=True, exist_ok=True)
    manifest_path = root / "config/cohort_samples.tsv"
    manifest_identity_sha256, home_split_label_sha256 = _manifest_fingerprints(root)
    manifest = pd.read_csv(manifest_path, sep="\t", dtype=str,
                           usecols=["cohort", "sample_id", "patient_key"])
    relevant = manifest.loc[manifest.cohort.eq("home") & manifest.patient_key.isin(selected)]
    if set(relevant.patient_key) != selected or relevant.patient_key.duplicated().any():
        raise ValueError("Node preparation requires one complete Home library per selected patient")
    files = [(row.patient_key,
              root / "data/clone_detail" / f"home__{row.sample_id}.parquet")
             for row in relevant.itertuples(index=False)]
    if not all(path.is_file() for _, path in files):
        raise FileNotFoundError("Selected Home clone-detail files incomplete")
    union_file = pq.ParquetFile(root / "data/aa_union.parquet")
    aa_to_idx: dict[str, int] = {}
    for batch in union_file.iter_batches(batch_size=131072, columns=["aa", "seq_idx"]):
        aa_to_idx.update(zip(batch.column("aa").to_pylist(),
                             batch.column("seq_idx").to_pylist()))
    if len(aa_to_idx) != union_file.metadata.num_rows:
        raise ValueError("AA union contains duplicate AA")
    seen_aa = np.zeros(len(aa_to_idx), dtype=np.bool_)
    patient_schema = pa.schema([
        ("patient_key", pa.string()), ("aa", pa.string()),
        ("v_gene", pa.string()), ("j_gene", pa.string()),
        ("read_count", pa.int64()), ("seq_idx", pa.int64())])
    clone_schema = pa.schema([
        ("aa", pa.string()), ("v_gene", pa.string()),
        ("j_gene", pa.string()), ("seq_idx", pa.int64()),
        ("n_patients", pa.int64()), ("total_reads", pa.int64())])
    patient_tmp = out / "patient_nodes.parquet.tmp"
    clone_tmp = out / "clone_nodes.parquet.tmp"
    patient_rows = total_reads = clone_rows = v_missing = j_missing = 0
    started = time.monotonic()
    with tempfile.TemporaryDirectory(prefix="node_buckets_", dir=out) as bucket_dir:
        bucket_paths = [Path(bucket_dir) / f"{i:02}.parquet" for i in range(64)]
        bucket_writers: dict[int, pq.ParquetWriter] = {}
        with pq.ParquetWriter(patient_tmp, patient_schema, compression="zstd") as patient_writer:
            try:
                for patient, path in files:
                    frame = pd.read_parquet(path, columns=RAW_COLS)
                    if not frame.cohort.eq("home").all() or not frame.patient_key.eq(patient).all():
                        raise ValueError(f"Clone file identity differs from manifest: {path.name}")
                    frame = frame.loc[frame.cohort.eq("home") & frame.patient_key.eq(patient) &
                                      frame.qc_pass.eq(True), ["patient_key"] + KEY + ["read_count"]]
                    frame = frame.dropna(subset=["aa"])
                    for col in ["patient_key"] + KEY:
                        frame[col] = frame[col].fillna("").astype(str)
                    frame["read_count"] = pd.to_numeric(frame.read_count, errors="coerce").fillna(0).astype(np.int64)
                    frame = frame.loc[frame.read_count.gt(0)]
                    if frame.empty:
                        raise ValueError(f"No QC nodes for {patient}")
                    frame = frame.groupby(["patient_key"] + KEY, as_index=False,
                                          sort=False).read_count.sum()
                    frame["seq_idx"] = frame.aa.map(aa_to_idx)
                    missing = int(frame.seq_idx.isna().sum())
                    if missing:
                        raise ValueError(f"{missing} QC nodes absent from AA union for {patient}")
                    frame["seq_idx"] = frame.seq_idx.astype(np.int64)
                    patient_rows += len(frame)
                    total_reads += int(frame.read_count.sum())
                    table = pa.Table.from_pandas(frame, schema=patient_schema, preserve_index=False)
                    patient_writer.write_table(table)
                    bucket_ids = pd.util.hash_pandas_object(frame[KEY], index=False).to_numpy() % 64
                    for bucket_id in np.unique(bucket_ids):
                        subset = frame.iloc[np.flatnonzero(bucket_ids == bucket_id)]
                        writer = bucket_writers.get(int(bucket_id))
                        if writer is None:
                            writer = pq.ParquetWriter(bucket_paths[int(bucket_id)], patient_schema,
                                                      compression="zstd")
                            bucket_writers[int(bucket_id)] = writer
                        writer.write_table(pa.Table.from_pandas(
                            subset, schema=patient_schema, preserve_index=False))
            finally:
                for writer in bucket_writers.values():
                    writer.close()
        with pq.ParquetWriter(clone_tmp, clone_schema, compression="zstd") as clone_writer:
            for path in bucket_paths:
                if not path.exists():
                    continue
                frame = pd.read_parquet(path)
                nodes = frame.groupby(KEY + ["seq_idx"], as_index=False, sort=False).agg(
                    n_patients=("patient_key", "nunique"), total_reads=("read_count", "sum"))
                clone_rows += len(nodes)
                v_missing += int(nodes.v_gene.eq("").sum())
                j_missing += int(nodes.j_gene.eq("").sum())
                seen_aa[nodes.seq_idx.to_numpy(dtype=np.int64)] = True
                clone_writer.write_table(pa.Table.from_pandas(
                    nodes, schema=clone_schema, preserve_index=False))
    patient_tmp.replace(out / "patient_nodes.parquet")
    clone_tmp.replace(out / "clone_nodes.parquet")
    if (manifest_identity_sha256, home_split_label_sha256) != _manifest_fingerprints(root):
        raise ValueError("Cohort manifest changed during node preparation")
    _json(out / "description.json", {
        "scope": scope,
        "phenotype_blind": True,
        "role": ("descriptive_only" if scope == "full_home" else
                 "inner_train_dictionary_input" if scope.startswith("inner_") else
                 "outer_train_dictionary_input"),
        "patient_ids": sorted(selected),
        "train_patient_sha256": _patient_hash(selected),
        "folds_sha256": _sha256(root / "folds/folds.tsv"),
        "manifest_identity_sha256": manifest_identity_sha256,
        "home_split_label_sha256": home_split_label_sha256,
        "patient_nodes_sha256": _sha256(out / "patient_nodes.parquet"),
        "clone_nodes_sha256": _sha256(out / "clone_nodes.parquet"),
        "n_patients": len(selected),
        "n_unique_aa": int(seen_aa.sum()),
        "n_unique_aa_vj": int(clone_rows),
        "n_patient_aa_vj_rows": int(patient_rows),
        "total_reads": int(total_reads),
        "missing_union_rows": 0,
        "v_missing_nodes": v_missing,
        "j_missing_nodes": j_missing,
        "preparation": "per-patient grouping and 64 hash buckets, exact AA/V/J global aggregation",
        "elapsed_seconds": time.monotonic() - started,
        "rss_ru_maxrss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
    })
    return out


def derive_train_nodes_from_full_home(root: Path, scope: str) -> Path:
    """Filter exact full-Home patient nodes to this train set, then reaggregate.

    This reuses phenotype-blind per-patient clone counts only. Held-out
    patients are excluded before any fold-level AA/V/J aggregation.
    """
    import tempfile
    import pyarrow as pa
    import pyarrow.parquet as pq

    if scope == "full_home" or not (scope.startswith("outer_") or scope.startswith("inner_")):
        raise ValueError("Derivation requires an outer or inner training scope")
    source_desc = _validate_scope_nodes(root, "full_home")
    selected = _scope_patients(root, scope)
    folds_sha = _sha256(root / "folds/folds.tsv")
    if not selected < set(source_desc["patient_ids"]):
        raise ValueError("Fold training set must be a proper subset of full Home")
    source_dir = root / "groups/full_home"
    source_path = source_dir / "patient_nodes.parquet"
    source_desc_sha = _sha256(source_dir / "description.json")
    source_nodes_sha = source_desc["patient_nodes_sha256"]
    out = root / "groups" / scope
    out.mkdir(parents=True, exist_ok=True)
    patient_tmp = out / "patient_nodes.parquet.tmp"
    clone_tmp = out / "clone_nodes.parquet.tmp"
    patient_schema = pq.read_schema(source_path)
    clone_schema = pq.read_schema(source_dir / "clone_nodes.parquet")
    n_union = pq.ParquetFile(root / "data/aa_union.parquet").metadata.num_rows
    seen_aa = np.zeros(n_union, dtype=np.bool_)
    patient_rows = total_reads = clone_rows = v_missing = j_missing = 0
    seen_patients: set[str] = set()
    started = time.monotonic()
    source = ds.dataset(str(source_path), format="parquet")
    scanner = source.scanner(
        columns=["patient_key"] + KEY + ["read_count", "seq_idx"],
        filter=ds.field("patient_key").isin(sorted(selected)),
        batch_size=131072)
    with tempfile.TemporaryDirectory(prefix="train_node_buckets_", dir=out) as bucket_dir:
        bucket_paths = [Path(bucket_dir) / f"{i:02}.parquet" for i in range(64)]
        bucket_writers: dict[int, pq.ParquetWriter] = {}
        with pq.ParquetWriter(patient_tmp, patient_schema, compression="zstd") as patient_writer:
            try:
                for batch in scanner.to_batches():
                    frame = batch.to_pandas()
                    if frame.empty:
                        continue
                    if not frame.patient_key.isin(selected).all():
                        raise ValueError("Held-out patient entered fold node stream")
                    if not frame.read_count.gt(0).all():
                        raise ValueError("Full-Home patient node has nonpositive read count")
                    seen_patients.update(frame.patient_key.astype(str))
                    patient_rows += len(frame)
                    total_reads += int(frame.read_count.sum())
                    patient_writer.write_table(pa.Table.from_pandas(
                        frame, schema=patient_schema, preserve_index=False))
                    bucket_ids = pd.util.hash_pandas_object(frame[KEY], index=False).to_numpy() % 64
                    for bucket_id in np.unique(bucket_ids):
                        subset = frame.iloc[np.flatnonzero(bucket_ids == bucket_id)]
                        writer = bucket_writers.get(int(bucket_id))
                        if writer is None:
                            writer = pq.ParquetWriter(bucket_paths[int(bucket_id)], patient_schema,
                                                      compression="zstd")
                            bucket_writers[int(bucket_id)] = writer
                        writer.write_table(pa.Table.from_pandas(
                            subset, schema=patient_schema, preserve_index=False))
            finally:
                for writer in bucket_writers.values():
                    writer.close()
        if seen_patients != selected:
            raise ValueError("Fold node stream did not include every training patient")
        with pq.ParquetWriter(clone_tmp, clone_schema, compression="zstd") as clone_writer:
            for path in bucket_paths:
                if not path.exists():
                    continue
                frame = pd.read_parquet(path)
                nodes = frame.groupby(KEY + ["seq_idx"], as_index=False, sort=False).agg(
                    n_patients=("patient_key", "nunique"), total_reads=("read_count", "sum"))
                clone_rows += len(nodes)
                v_missing += int(nodes.v_gene.eq("").sum())
                j_missing += int(nodes.j_gene.eq("").sum())
                indices = nodes.seq_idx.to_numpy(dtype=np.int64)
                if len(indices) and (indices.min() < 0 or indices.max() >= n_union):
                    raise ValueError("Fold node has invalid global seq_idx")
                seen_aa[indices] = True
                clone_writer.write_table(pa.Table.from_pandas(
                    nodes, schema=clone_schema, preserve_index=False))
    if (_sha256(source_path) != source_nodes_sha
            or _sha256(source_dir / "description.json") != source_desc_sha
            or _sha256(root / "folds/folds.tsv") != folds_sha
            or _scope_patients(root, scope) != selected
            or (_manifest_identity_sha256(root), _home_split_label_sha256(root)) != (
                source_desc["manifest_identity_sha256"], source_desc["home_split_label_sha256"])):
        raise ValueError("Full-Home node source changed during fold derivation")
    patient_tmp.replace(out / "patient_nodes.parquet")
    clone_tmp.replace(out / "clone_nodes.parquet")
    _json(out / "description.json", {
        "scope": scope, "phenotype_blind": True,
        "role": ("inner_train_dictionary_input" if scope.startswith("inner_")
                 else "outer_train_dictionary_input"),
        "patient_ids": sorted(selected), "train_patient_sha256": _patient_hash(selected),
        "folds_sha256": folds_sha,
        "manifest_identity_sha256": source_desc["manifest_identity_sha256"],
        "home_split_label_sha256": source_desc["home_split_label_sha256"],
        "patient_nodes_sha256": _sha256(out / "patient_nodes.parquet"),
        "clone_nodes_sha256": _sha256(out / "clone_nodes.parquet"),
        "source_full_home_patient_nodes_sha256": source_nodes_sha,
        "source_full_home_description_sha256": source_desc_sha,
        "n_patients": len(selected), "n_unique_aa": int(seen_aa.sum()),
        "n_unique_aa_vj": int(clone_rows), "n_patient_aa_vj_rows": int(patient_rows),
        "total_reads": int(total_reads), "missing_union_rows": 0,
        "v_missing_nodes": int(v_missing), "j_missing_nodes": int(j_missing),
        "preparation": "filter full-Home exact per-patient nodes to train patients before 64-bucket AA/V/J aggregation",
        "elapsed_seconds": time.monotonic() - started,
        "rss_ru_maxrss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
    })
    return out


def _validate_scope_nodes(root: Path, scope: str) -> dict:
    """Reject stale fold or node files before defining a group dictionary."""
    out = root / "groups" / scope
    path = out / "description.json"
    desc = json.loads(path.read_text())
    if desc.get("scope") != scope:
        raise ValueError("Scope node description mismatch")
    if scope.startswith("pilot_home_"):
        selected = set(desc.get("patient_ids", []))
        if not desc.get("pilot_only") or not selected or\
                desc.get("source_manifest_identity_sha256") != _manifest_identity_sha256(root) or\
                desc.get("home_split_label_sha256") != _home_split_label_sha256(root):
            raise ValueError("Pilot node source changed")
    else:
        selected = _scope_patients(root, scope)
        if desc.get("folds_sha256") != _sha256(root / "folds/folds.tsv") or\
                desc.get("manifest_identity_sha256") != _manifest_identity_sha256(root) or\
                desc.get("home_split_label_sha256") != _home_split_label_sha256(root) or\
                set(desc.get("patient_ids", [])) != selected:
            raise ValueError("Fold table changed after node preparation")
    if desc.get("train_patient_sha256") != _patient_hash(selected) or\
            desc.get("patient_nodes_sha256") != _sha256(out / "patient_nodes.parquet") or\
            desc.get("clone_nodes_sha256") != _sha256(out / "clone_nodes.parquet"):
        raise ValueError("Node input or training patients changed")
    if "source_full_home_patient_nodes_sha256" in desc:
        source_desc = _validate_scope_nodes(root, "full_home")
        if (desc.get("source_full_home_patient_nodes_sha256") != source_desc["patient_nodes_sha256"]
                or desc.get("source_full_home_description_sha256") != _sha256(
                    root / "groups/full_home/description.json")):
            raise ValueError("Fold node derivation source changed")
    return desc


def rebase_node_manifest_receipt(root: Path, scope: str) -> Path:
    """Upgrade a completed pre-lock node receipt without rescanning clone rows.

    The old raw-manifest SHA is accepted only to prove the manifest was
    unchanged between the original scan and this one-time receipt upgrade.
    The raw SHA is retained in an audit file, outside dictionary state.
    """
    out = root / "groups" / scope
    desc_path = out / "description.json"
    desc = json.loads(desc_path.read_text())
    if "manifest_identity_sha256" in desc:
        _validate_scope_nodes(root, scope)
        return desc_path
    if scope != "full_home" or desc.get("scope") != scope or\
            desc.get("manifest_sha256") != _sha256(root / "config/cohort_samples.tsv") or\
            desc.get("folds_sha256") != _sha256(root / "folds/folds.tsv") or\
            set(desc.get("patient_ids", [])) != _scope_patients(root, scope) or\
            desc.get("train_patient_sha256") != _patient_hash(_scope_patients(root, scope)) or\
            desc.get("patient_nodes_sha256") != _sha256(out / "patient_nodes.parquet") or\
            desc.get("clone_nodes_sha256") != _sha256(out / "clone_nodes.parquet"):
        raise ValueError("Cannot safely rebase stale or incomplete node receipt")
    manifest = pd.read_csv(root / "config/cohort_samples.tsv", sep="\t", dtype=str,
                           usecols=["cohort", "sample_id", "patient_key"])
    relevant = manifest.loc[manifest.cohort.eq("home") &
                            manifest.patient_key.isin(_scope_patients(root, scope))]
    if len(relevant) != len(desc["patient_ids"]) or relevant.patient_key.duplicated().any():
        raise ValueError("Home node source identity changed during receipt rebase")
    source_stats = []
    oldest_output = min((out / "patient_nodes.parquet").stat().st_mtime_ns,
                        (out / "clone_nodes.parquet").stat().st_mtime_ns)
    for row in relevant.itertuples(index=False):
        source = root / "data/clone_detail" / f"home__{row.sample_id}.parquet"
        info = source.stat()
        if info.st_mtime_ns > oldest_output:
            raise ValueError("Clone detail source changed after node output was written")
        source_stats.append([source.name, info.st_size, info.st_mtime_ns])
    old_description_sha = _sha256(desc_path)
    old_raw_sha = desc.pop("manifest_sha256")
    identity_sha, home_label_sha = _manifest_fingerprints(root)
    desc["manifest_identity_sha256"] = identity_sha
    desc["home_split_label_sha256"] = home_label_sha
    _json(desc_path, desc)
    _json(root / "audit/04_full_home_node_manifest_rebase.json", {
        "status": "PASS", "scope": scope, "old_description_sha256": old_description_sha,
        "new_description_sha256": _sha256(desc_path),
        "old_raw_manifest_sha256_audit_only": old_raw_sha,
        "manifest_identity_sha256": identity_sha,
        "home_split_label_sha256": home_label_sha,
        "source_clone_file_stats_sha256": hashlib.sha256(json.dumps(
            sorted(source_stats), separators=(",", ":")).encode()).hexdigest(),
        "n_source_clone_files": len(source_stats),
        "node_files_unchanged": True,
    })
    _validate_scope_nodes(root, scope)
    return desc_path


def prepare_gliph2(root: Path, scope: str, max_per_patient: int, seed: int = 20260929,
                   simulation_depth: int = 1000) -> Path:
    if max_per_patient < 0:
        raise ValueError("max_per_patient must be >=0")
    if max_per_patient == 0:
        gate = root / "groups" / "gliph2_full_feasibility.json"
        if not gate.exists() or not json.loads(gate.read_text()).get("pass_full_run", False):
            raise ValueError("Full GLIPH2 requires an explicit passed feasibility gate")
    out = root / "groups" / scope
    node_desc = _validate_scope_nodes(root, scope)
    nodes = pd.read_parquet(out / "patient_nodes.parquet")
    nodes = nodes.loc[nodes.v_gene.astype(str).str.match(r"^TRBV[0-9]+(?:-[0-9]+)?(?:\*[0-9]+)?$") &
                      nodes.j_gene.astype(str).str.match(r"^TRBJ[0-9]+(?:-[0-9]+)?(?:\*[0-9]+)?$")].copy()
    if nodes.empty:
        raise ValueError("No V/J-assigned nodes for GLIPH2")
    if max_per_patient:
        sampled = []
        for patient, group in nodes.groupby("patient_key", sort=True):
            n = min(max_per_patient, len(group))
            sampled.append(group.sample(n=n, random_state=int.from_bytes(
                hashlib.sha256(f"{seed}|{scope}|{patient}".encode()).digest()[:4], "little")))
        nodes = pd.concat(sampled, ignore_index=True)
    gliph = pd.DataFrame({
        "#CDR3b": nodes.aa.astype(str),
        "TRBV": nodes.v_gene.map(_stem),
        "TRBJ": nodes.j_gene.map(_stem),
        "CDR3c": nodes.aa.astype(str),
        "CDR3p": "NA",
        "subject:condition": nodes.patient_key.astype(str) + ":HomeTrain",
        "Frequency": 1,
    })
    pilot = out / ("gliph2_full" if max_per_patient == 0 else f"gliph2_pilot_{max_per_patient}")
    pilot.mkdir(parents=True, exist_ok=True)
    gliph.to_csv(pilot / "input.tsv", sep="\t", index=False)
    refs = {
        "refer_file": LEGACY_GLIPH_ROOT / "reference_human_v2_CD48/ref_CD48_v2.0.reconstructed.txt",
        "v_usage_freq_file": LEGACY_GLIPH_ROOT / "reference_human_v2_CD48/ref_V_CD48_v2.0.reconstructed.txt",
        "cdr3_length_freq_file": LEGACY_GLIPH_ROOT / "reference_human_v2_CD48/ref_L_CD48_v2.0.reconstructed.txt",
    }
    for path in refs.values():
        if not path.exists():
            raise FileNotFoundError(path)
    (pilot / "hla_unavailable_empty.tsv").touch()
    lines = [
        "out_prefix=gliph2_result",
        f"hla_file={pilot / 'hla_unavailable_empty.tsv'}",
        f"cdr3_file={pilot / 'input.tsv'}",
    ]
    lines += [f"{key}={value}" for key, value in refs.items()]
    lines += [
        "local_min_pvalue=0.001", "p_depth=1000", "global_convergence_cutoff=1",
        f"simulation_depth={simulation_depth}", "kmer_min_depth=3",
        "local_min_OVE=10", "algorithm=GLIPH2", "all_aa_interchangeable=1",
    ]
    (pilot / "params.txt").write_text("\n".join(lines) + "\n")
    _json(pilot / "input_receipt.json", {
        "scope": scope, "max_per_patient": max_per_patient, "seed": seed,
        "train_patient_sha256": node_desc["train_patient_sha256"],
        "node_description_sha256": _sha256(out / "description.json"),
        "folds_sha256": node_desc.get("folds_sha256"),
        "manifest_identity_sha256": node_desc.get("manifest_identity_sha256", node_desc.get("source_manifest_identity_sha256")),
        "home_split_label_sha256": node_desc["home_split_label_sha256"],
        "n_input_rows": int(len(gliph)), "n_input_patients": int(nodes.patient_key.nunique()),
        "input_sha256": _sha256(pilot / "input.tsv"),
        "refs": {key: {"path": str(path), "sha256": _sha256(path)} for key, path in refs.items()},
        "simulation_depth": simulation_depth, "hla_unavailable": True,
        "not_full_coverage": max_per_patient != 0,
    })
    return pilot


def prepare_gliph2_resource_input(root: Path, max_patients: int = 12,
                                  max_per_patient: int = 100,
                                  seed: int = 20260929) -> Path:
    """Bounded label-blind binary/projection smoke input; never a discovery scope."""
    if not 3 <= max_patients <= 24 or not 20 <= max_per_patient <= 800:
        raise ValueError("GLIPH2 resource pilot limits: 3..24 patients, 20..800 nodes each")
    import pyarrow.parquet as pq
    manifest_path = root / "config/cohort_samples.tsv"
    manifest = pd.read_csv(manifest_path, sep="\t", dtype=str,
                           usecols=["cohort", "sample_id", "patient_key"])
    home = manifest.loc[manifest.cohort.eq("home")]
    if home.patient_key.duplicated().any():
        raise ValueError("Pilot expects one Home library per patient")
    home = home.assign(order=home.patient_key.map(
        lambda value: hashlib.sha256(f"{seed}|{value}".encode()).hexdigest()))
    selected = home.sort_values("order").head(max_patients)
    if len(selected) != max_patients:
        raise ValueError("Too few Home patients for requested resource pilot")
    scope = f"pilot_home_{max_patients}_{max_per_patient}"
    out = root / "groups" / scope
    out.mkdir(parents=True, exist_ok=True)
    parts = []
    source_files = []
    for row in selected.itertuples(index=False):
        path = root / "data/clone_detail" / f"home__{row.sample_id}.parquet"
        if not path.is_file() or pq.ParquetFile(path).metadata.num_rows > 1_000_000:
            raise ValueError(f"Pilot source absent or >1m raw clone rows: {path}")
        frame = pd.read_parquet(path, columns=RAW_COLS)
        if not frame.cohort.eq("home").all() or not frame.patient_key.eq(row.patient_key).all():
            raise ValueError(f"Pilot source identity mismatch: {path.name}")
        frame = frame.loc[frame.qc_pass.eq(True) &
                          frame.v_gene.astype(str).str.startswith("TRBV") &
                          frame.j_gene.astype(str).str.startswith("TRBJ"),
                          ["patient_key"] + KEY + ["read_count"]]
        frame = frame.groupby(["patient_key"] + KEY, as_index=False, sort=False).read_count.sum()
        if len(frame) < 20:
            raise ValueError(f"Too few QC V/J nodes for pilot patient {row.patient_key}")
        rng = int.from_bytes(hashlib.sha256(f"{seed}|{row.patient_key}".encode()).digest()[:4], "little")
        parts.append(frame.sample(n=min(max_per_patient, len(frame)), random_state=rng))
        source_files.append({"path": str(path), "bytes": path.stat().st_size,
                             "mtime_ns": path.stat().st_mtime_ns})
    nodes = pd.concat(parts, ignore_index=True)
    local_ids = {aa: i for i, aa in enumerate(sorted(nodes.aa.unique()))}
    nodes["seq_idx"] = nodes.aa.map(local_ids).astype(np.int64)
    nodes.to_parquet(out / "patient_nodes.parquet", index=False)
    clone_nodes = nodes.groupby(KEY + ["seq_idx"], as_index=False).agg(
        n_patients=("patient_key", "nunique"), total_reads=("read_count", "sum"))
    clone_nodes.to_parquet(out / "clone_nodes.parquet", index=False)
    _json(out / "description.json", {
        "scope": scope, "pilot_only": True, "never_association_or_integration": True,
        "train_patient_sha256": _patient_hash(set(selected.patient_key)),
        "patient_nodes_sha256": _sha256(out / "patient_nodes.parquet"),
        "clone_nodes_sha256": _sha256(out / "clone_nodes.parquet"),
        "patient_ids": sorted(selected.patient_key.tolist()),
        "local_seq_idx_not_embedding_bank": True,
        "source_manifest_identity_sha256": _manifest_identity_sha256(root),
        "home_split_label_sha256": _home_split_label_sha256(root),
        "source_file_stats": source_files,
        "n_input_patient_nodes": len(nodes), "n_unique_aa_vj": len(clone_nodes),
        "sampling": "deterministic patient hash; at most 1m raw rows per selected library; fixed equal pilot nodes per patient",
        "labels_read": False,
    })
    return prepare_gliph2(root, scope, max_per_patient, seed=seed, simulation_depth=1000)


def run_gliph2(pilot: Path, binary: Path = LEGACY_GLIPH_ROOT / "irtools.centos",
               max_seconds: int = 3600) -> dict:
    if not binary.is_file():
        raise FileNotFoundError(binary)
    if max_seconds < 1 or max_seconds > 86400:
        raise ValueError("max_seconds must be 1..86400")
    input_receipt_path = pilot / "input_receipt.json"
    input_receipt = json.loads(input_receipt_path.read_text())
    if input_receipt.get("input_sha256") != _sha256(pilot / "input.tsv"):
        raise ValueError("GLIPH2 input changed after receipt")
    node_desc_path = pilot.parent / "description.json"
    if input_receipt.get("node_description_sha256") != _sha256(node_desc_path):
        raise ValueError("GLIPH2 node description changed after input")
    def limit() -> None:
        resource.setrlimit(resource.RLIMIT_AS, (128 * 1024**3, 128 * 1024**3))
    started = time.monotonic()
    with (pilot / "stdout.log").open("w") as stdout, (pilot / "stderr.log").open("w") as stderr:
        try:
            result = subprocess.run([str(binary), "-c", "params.txt"], cwd=pilot,
                                    stdout=stdout, stderr=stderr, timeout=max_seconds,
                                    preexec_fn=limit, check=False)
            exit_code = result.returncode
        except subprocess.TimeoutExpired:
            exit_code = 124
    receipt = {
        "binary": str(binary), "binary_sha256": _sha256(binary),
        "pilot": str(pilot), "params_sha256": _sha256(pilot / "params.txt"),
        "input_receipt_sha256": _sha256(input_receipt_path),
        "train_patient_sha256": input_receipt["train_patient_sha256"],
        "node_description_sha256": input_receipt["node_description_sha256"],
        "elapsed_seconds": time.monotonic() - started,
        "exit_code": exit_code, "max_seconds": max_seconds,
        "rss_ru_maxrss_kib": resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss,
        "full_coverage_claim": False,
    }
    cluster_path = pilot / "gliph2_result_cluster.csv"
    receipt["cluster_csv_sha256"] = _sha256(cluster_path) if exit_code == 0 and cluster_path.is_file() else None
    _json(pilot / "run_receipt.json", receipt)
    if exit_code:
        raise RuntimeError(f"GLIPH2 failed with exit {exit_code}; see {pilot}/stderr.log")
    return receipt


def import_gliph2(root: Path, scope: str, cluster_csv: Path) -> Path:
    out = root / "groups" / scope
    desc = _validate_scope_nodes(root, scope)
    cluster_csv = cluster_csv.resolve()
    if not cluster_csv.is_relative_to(out.resolve()) or not cluster_csv.is_file():
        raise ValueError("GLIPH2 cluster CSV must be a completed run in this scope")
    input_receipt_path = cluster_csv.parent / "input_receipt.json"
    run_receipt_path = cluster_csv.parent / "run_receipt.json"
    if not input_receipt_path.exists() or not run_receipt_path.exists():
        raise ValueError("GLIPH2 source lacks input and successful run receipts")
    input_receipt = json.loads(input_receipt_path.read_text())
    run_receipt = json.loads(run_receipt_path.read_text())
    if input_receipt.get("scope") != scope or run_receipt.get("exit_code") != 0 or\
            input_receipt.get("input_sha256") != _sha256(cluster_csv.parent / "input.tsv") or\
            run_receipt.get("params_sha256") != _sha256(cluster_csv.parent / "params.txt") or\
            input_receipt.get("train_patient_sha256") != desc["train_patient_sha256"] or\
            input_receipt.get("node_description_sha256") != _sha256(out / "description.json") or\
            run_receipt.get("train_patient_sha256") != desc["train_patient_sha256"] or\
            run_receipt.get("input_receipt_sha256") != _sha256(input_receipt_path) or\
            run_receipt.get("cluster_csv_sha256") != _sha256(cluster_csv):
        raise ValueError("GLIPH2 source scope/input/run receipt mismatch")
    nodes = pd.read_parquet(out / "clone_nodes.parquet")
    source = pd.read_csv(cluster_csv)
    source.columns = source.columns.str.strip()
    required = {"index", "TcRb", "V", "J"}
    if required - set(source.columns):
        raise ValueError(f"GLIPH2 CSV missing {sorted(required - set(source.columns))}")
    source = source.dropna(subset=["index", "TcRb"]).copy()
    source["aa"] = source.TcRb.astype(str)
    source["v_stem"] = source.V.map(_stem)
    source["j_stem"] = source.J.map(_stem)
    source["group_id"] = scope + ":gliph2:" + source["index"].astype(str)
    metadata_cols = [c for c in ["group_id", "pattern", "type", "final_score", "number_subject"] if c in source.columns]
    if "group_id" in metadata_cols:
        source[metadata_cols].drop_duplicates("group_id").to_csv(
            out / "group_metadata.tsv", sep="\t", index=False)
    source = source[["group_id", "aa", "v_stem", "j_stem"]].drop_duplicates()
    nodes["v_stem"] = nodes.v_gene.map(_stem)
    nodes["j_stem"] = nodes.j_gene.map(_stem)
    matched = nodes.merge(source, on=["aa", "v_stem", "j_stem"], how="inner", validate="many_to_many")
    if matched.empty:
        raise ValueError("No GLIPH2 members matched scope training nodes")
    membership = matched[["group_id", "aa", "v_gene", "j_gene", "seq_idx"]].drop_duplicates()
    membership.insert(0, "method", "gliph2")
    membership.insert(0, "scope", scope)
    seed_path = out / "gliph2_seed_membership.parquet"
    membership.to_parquet(seed_path, index=False)
    _json(out / "membership_receipt.json", {
        "status": "pilot_seed_members_only",
        "scope": scope, "method": "gliph2", "source": str(cluster_csv),
        "source_sha256": _sha256(cluster_csv),
        "train_patient_sha256": desc["train_patient_sha256"],
        "node_description_sha256": _sha256(out / "description.json"),
        "folds_sha256": desc.get("folds_sha256"),
        "input_receipt_sha256": _sha256(input_receipt_path),
        "run_receipt_sha256": _sha256(run_receipt_path),
        "seed_membership_sha256": _sha256(seed_path),
        "group_metadata_sha256": _sha256(out / "group_metadata.tsv") if (out / "group_metadata.tsv").exists() else None,
        "input_group_rows": int(len(source)), "matched_group_rows": int(len(matched)),
        "group_count": int(membership.group_id.nunique()),
        "member_aa_vj": int(len(membership)),
        "matched_unique_node_fraction": float(membership[KEY].drop_duplicates().shape[0] / len(nodes)),
        "unassigned_node_fraction": float(1 - membership[KEY].drop_duplicates().shape[0] / len(nodes)),
        "largest_group_node_fraction": float(membership.groupby("group_id").size().max() / len(nodes)),
        "pilot_run_receipt": str(cluster_csv.parent / "run_receipt.json") if (cluster_csv.parent / "run_receipt.json").exists() else None,
        "fold_local": scope != "full_home",
        "full_home_use": "description_and_home_association_only" if scope == "full_home" else None,
    })
    return seed_path


def _aa_safety(out: Path, nodes: pd.DataFrame, membership: pd.DataFrame) -> None:
    variants = nodes.groupby("aa").size().rename("all_variants")
    in_group = membership.groupby(["group_id", "aa"]).size().rename("member_variants").reset_index()
    in_group = in_group.merge(variants.reset_index(), on="aa", how="left")
    safe = in_group.loc[in_group.member_variants.eq(in_group.all_variants), ["group_id", "aa"]]
    safe = safe.merge(nodes[["aa", "seq_idx"]].drop_duplicates(), on="aa", how="left").drop_duplicates()
    safe["scope_only_validation"] = True
    safe.to_parquet(out / "group_membership_aa_safe.parquet", index=False)

def project_gliph2_motifs(root: Path, scope: str,
                          max_scan_rows: int | None = None) -> Path:
    """Apply eligible single literal GLIPH2 seed motifs to every QC clone row.

    This is a derived classifier from GLIPH2 seed groups, not GLIPH2 scoring
    of newly projected clones. Only the seed group's literal motif and its
    observed V/J-stem pairs define a projectable family.
    """
    import pyarrow as pa
    import pyarrow.compute as pc
    from datetime import datetime, timezone
    pa.set_cpu_count(min(16, os.cpu_count() or 1))
    soft, hard = resource.getrlimit(resource.RLIMIT_AS)
    limit_bytes = 128 * 1024**3
    if soft == resource.RLIM_INFINITY or soft > limit_bytes:
        resource.setrlimit(resource.RLIMIT_AS, (min(limit_bytes, hard) if hard != resource.RLIM_INFINITY else limit_bytes, hard))
    out = root / "groups" / scope
    node_desc = _validate_scope_nodes(root, scope)
    train_patients = set(node_desc["patient_ids"])
    seed_path = out / "gliph2_seed_membership.parquet"
    metadata_path = out / "group_metadata.tsv"
    if not seed_path.exists() or not metadata_path.exists():
        raise FileNotFoundError("Import a verified same-scope GLIPH2 run first")
    seed_receipt_path = out / "membership_receipt.json"
    if not seed_receipt_path.exists():
        raise ValueError("GLIPH2 seed lacks verified import receipt")
    seed_receipt = json.loads(seed_receipt_path.read_text())
    source_path = Path(seed_receipt.get("source", ""))
    if seed_receipt.get("scope") != scope or\
            seed_receipt.get("status") != "pilot_seed_members_only" or\
            seed_receipt.get("seed_membership_sha256") != _sha256(seed_path) or\
            seed_receipt.get("train_patient_sha256") != node_desc["train_patient_sha256"] or\
            seed_receipt.get("node_description_sha256") != _sha256(out / "description.json") or\
            seed_receipt.get("group_metadata_sha256") != _sha256(metadata_path) or\
            not source_path.is_file() or\
            seed_receipt.get("source_sha256") != _sha256(source_path):
        raise ValueError("GLIPH2 seed import provenance invalid")
    seed = pd.read_parquet(seed_path)
    if not seed.scope.eq(scope).all() or not seed.method.eq("gliph2").all():
        raise ValueError("GLIPH2 seed method/scope mismatch")
    meta = pd.read_csv(metadata_path, sep="\t")
    required = {"group_id", "pattern", "type", "final_score", "number_subject"}
    if required - set(meta):
        raise ValueError("GLIPH2 metadata lacks motif qualification columns")
    meta = meta.drop_duplicates("group_id").copy()
    meta["pattern"] = meta.pattern.fillna("").astype(str)
    meta["type"] = meta.type.fillna("").astype(str)
    meta["final_score"] = pd.to_numeric(meta.final_score, errors="coerce")
    meta["number_subject"] = pd.to_numeric(meta.number_subject, errors="coerce")
    eligible = meta.loc[meta.pattern.str.fullmatch(r"[ACDEFGHIKLMNPQRSTVWY]{4,8}") &
                        meta.type.eq("motif-" + meta.pattern) &
                        meta.final_score.le(0.05) & meta.number_subject.ge(3)].copy()
    # One motif => one family, so projection cannot double-label a sequence
    # with duplicate IDs for the same seed rule.
    duplicated = set(eligible.loc[eligible.pattern.duplicated(False), "pattern"])
    eligible = eligible.loc[~eligible.pattern.isin(duplicated)]
    eligible = eligible.loc[eligible.group_id.isin(seed.group_id)]
    if eligible.empty:
        _json(out / "gliph2_projection_receipt.json", {
            "status": "STOP_no_projectable_simple_motifs", "scope": scope,
            "method": "gliph2", "n_seed_groups": int(len(meta)),
            "n_duplicate_motifs_excluded": len(duplicated),
            "reason": "No simple literal motif satisfies the prespecified projection rule",
        })
        raise ValueError("No unique eligible simple GLIPH2 motif groups")
    seed_by_group = seed.loc[seed.group_id.isin(eligible.group_id)].merge(
        eligible[["group_id", "pattern"]], on="group_id", validate="many_to_one")
    invalid_seed = seed_by_group.loc[~seed_by_group.apply(
        lambda r: r.pattern in r.aa, axis=1), "group_id"].unique()
    if len(invalid_seed):
        eligible = eligible.loc[~eligible.group_id.isin(invalid_seed)]
    if eligible.empty:
        raise ValueError("All simple motifs failed seed-member rule validation")
    seed_by_group = seed_by_group.loc[seed_by_group.group_id.isin(eligible.group_id)]
    allowed_pairs = {
        group_id: set(zip(part.v_gene.map(_stem), part.j_gene.map(_stem)))
        for group_id, part in seed_by_group.groupby("group_id")
    }
    if max_scan_rows is not None and max_scan_rows < 1:
        raise ValueError("Pilot max_scan_rows must be positive")
    manifest = pd.read_csv(root / "config/cohort_samples.tsv", sep="\t", dtype=str,
                           usecols=["cohort", "sample_id"])
    expected_files = {f"{r.cohort}__{r.sample_id}.parquet" for r in manifest.itertuples(index=False)}
    clone_dir = root / "data/clone_detail"
    actual_files = {p.name for p in clone_dir.glob("*.parquet")}
    if expected_files != actual_files:
        raise ValueError(f"Clone-detail scope incomplete: missing={len(expected_files-actual_files)}, "
                         f"unexpected={len(actual_files-expected_files)}")
    home_files = sorted(clone_dir.glob("home__*.parquet"))
    if len(home_files) != sum(manifest.cohort.eq("home")):
        raise ValueError("Home clone-detail file count differs from identity manifest")
    # A formal Home dictionary never reads external clone content. The mixed
    # label-blind cost pilot below remains a technical diagnostic only.
    fingerprint_files = home_files if max_scan_rows is None else sorted(clone_dir.glob("*.parquet"))
    source_stats = [(p.name, p.stat().st_size, p.stat().st_mtime_ns)
                    for p in fingerprint_files]
    source_fingerprint = hashlib.sha256(json.dumps(source_stats).encode()).hexdigest()
    pilot_sampling = None
    if max_scan_rows is None or max_scan_rows < 1000:
        dataset = ds.dataset([str(p) for p in home_files], format="parquet")
        scanner = dataset.scanner(columns=["cohort", "patient_key", "aa", "v_gene", "j_gene"],
                                  filter=ds.field("qc_pass") == True,
                                  batch_size=131072)
        batches = (batch.to_pandas() for batch in scanner.to_batches())
    else:
        # A cost/coverage pilot must not take only the first MiXCR library or
        # its first, often abundance-sorted, rows. Sample QC rows from
        # deterministically selected Home and external files without labels.
        selected_files = []
        for cohort in ("home", "external"):
            choices = sorted((p for p in clone_dir.glob(f"{cohort}__*.parquet")),
                             key=lambda p: hashlib.sha256(p.name.encode()).hexdigest())
            selected_files.extend(choices[:min(24, len(choices))])
        if not selected_files:
            raise ValueError("No clone files for stratified projection pilot")
        per_file = max(1, math.ceil(max_scan_rows / len(selected_files)))
        def pilot_batches():
            remaining = max_scan_rows
            for path in selected_files:
                if remaining <= 0:
                    break
                frame = pd.read_parquet(path, columns=["cohort", "patient_key", "aa", "v_gene", "j_gene", "qc_pass"])
                frame = frame.loc[frame.qc_pass.eq(True),
                                  ["cohort", "patient_key", "aa", "v_gene", "j_gene"]]
                n = min(len(frame), per_file, remaining)
                if n == 0:
                    continue
                rng = int.from_bytes(hashlib.sha256(
                    f"20260929|projection_pilot|{path.name}".encode()).digest()[:4], "little")
                sampled = frame.sample(n=n, random_state=rng)
                remaining -= n
                yield sampled
        batches = pilot_batches()
        pilot_sampling = {"scheme": "deterministic_unlabeled_QC_rows_across_files",
                          "n_selected_files": len(selected_files),
                          "n_home_files": sum(p.name.startswith("home__") for p in selected_files),
                          "n_external_files": sum(p.name.startswith("external__") for p in selected_files),
                          "per_file_target": per_file,
                          "selected_file_names_sha256": hashlib.sha256(
                              "\n".join(p.name for p in selected_files).encode()).hexdigest()}
    motifs = list(eligible[["group_id", "pattern"]].itertuples(index=False, name=None))
    started = time.monotonic()
    scanned = valid = matched_clone_rows = train_valid = train_matched_clone_rows = 0
    group_clone_hits = {group_id: 0 for group_id, _ in motifs}
    group_train_hits = {group_id: 0 for group_id, _ in motifs}
    hit_parts = []
    train_hit_parts = []
    for frame in batches:
        if max_scan_rows is not None and scanned >= max_scan_rows:
            break
        if max_scan_rows is not None:
            frame = frame.head(max_scan_rows - scanned)
        scanned += len(frame)
        frame = frame.loc[frame.v_gene.astype(str).str.startswith("TRBV") &
                          frame.j_gene.astype(str).str.startswith("TRBJ")].copy()
        valid += len(frame)
        if frame.empty:
            continue
        train_row = frame.cohort.eq("home").to_numpy() &\
                    frame.patient_key.isin(train_patients).to_numpy()
        train_valid += int(train_row.sum())
        aa_arrow = pa.array(frame.aa.astype(str))
        vj_pairs = pd.MultiIndex.from_arrays([frame.v_gene.map(_stem), frame.j_gene.map(_stem)])
        any_hit = np.zeros(len(frame), dtype=bool)
        any_train_hit = np.zeros(len(frame), dtype=bool)
        for group_id, motif in motifs:
            motif_mask = pc.match_substring(aa_arrow, motif).to_numpy(zero_copy_only=False)
            mask = motif_mask & vj_pairs.isin(allowed_pairs[group_id])
            if mask.any():
                group_clone_hits[group_id] += int(mask.sum())
                any_hit |= mask
                train_mask = mask & train_row
                group_train_hits[group_id] += int(train_mask.sum())
                any_train_hit |= train_mask
                part = frame.loc[mask, ["aa", "v_gene", "j_gene"]].drop_duplicates()
                part.insert(0, "group_id", group_id)
                hit_parts.append(part)
                if train_mask.any():
                    train_part = frame.loc[train_mask, KEY].drop_duplicates()
                    train_part.insert(0, "group_id", group_id)
                    train_hit_parts.append(train_part)
        matched_clone_rows += int(any_hit.sum())
        train_matched_clone_rows += int(any_train_hit.sum())
    status = "partial_pilot" if max_scan_rows is not None else "complete_home_scope"
    receipt_path = out / "gliph2_projection_receipt.json"
    if not hit_parts:
        _json(receipt_path, {"status": "STOP_no_motif_matches", "scope": scope,
                             "method": "gliph2", "scanned_qc_clone_rows": scanned,
                             "reason": "No eligible motif matched QC clone rows"})
        raise ValueError("No full-scope motif matches")
    hits = pd.concat(hit_parts, ignore_index=True).drop_duplicates()
    group_counts = hits.groupby(KEY).group_id.nunique()
    n_ambiguous = int(group_counts.gt(1).sum())
    ambiguity_fraction = n_ambiguous / max(1, len(group_counts))
    train_hits = (pd.concat(train_hit_parts, ignore_index=True).drop_duplicates()
                  if train_hit_parts else pd.DataFrame(columns=["group_id"] + KEY))
    train_counts = train_hits.groupby(KEY).group_id.nunique()
    n_train_ambiguous = int(train_counts.gt(1).sum())
    train_ambiguity_fraction = n_train_ambiguous / max(1, len(train_counts))
    if max_scan_rows is None:
        union = pd.read_parquet(root / "data/aa_union.parquet", columns=["seq_idx", "aa"])
        if union.aa.duplicated().any():
            raise ValueError("AA union has duplicate sequence keys")
        hits = hits.merge(union, on="aa", how="left", validate="many_to_one")
        if hits.seq_idx.isna().any():
            raise ValueError("Motif-matched QC clone absent from AA union")
        hits["seq_idx"] = hits.seq_idx.astype(np.int64)
    else:
        # A bounded cost pilot never creates bank-index-valid membership.
        hits["seq_idx"] = np.int64(-1)
    hits.insert(0, "method", "gliph2")
    hits.insert(0, "scope", scope)
    path = out / ("gliph2_group_membership_pilot.parquet" if max_scan_rows is not None else
                  "gliph2_group_membership.parquet")
    hits.to_parquet(path, index=False)
    giant = max(group_clone_hits.values()) / max(1, valid)
    train_giant = max(group_train_hits.values()) / max(1, train_valid)
    report = {
        "status": status, "scope": scope, "method": "gliph2",
        "projection_population": "home" if max_scan_rows is None else "mixed_unlabeled_technical_pilot",
        "external_projected_clone_rows": 0 if max_scan_rows is None else None,
        "method_label": "GLIPH2 seed-derived motif projection",
        "family_definition": "Unique single literal seed motif (length 4-8, seed subjects >=3, GLIPH2 seed final_score <=0.05) plus seed-observed TRBV/TRBJ stem pair; derived classifier applied to all QC clone rows; global/wildcard/compound GLIPH2 groups excluded",
        "not_exact_gliph2_cluster_membership": True,
        "train_patient_sha256": node_desc["train_patient_sha256"],
        "node_description_sha256": _sha256(out / "description.json"),
        "folds_sha256": node_desc.get("folds_sha256"),
        "manifest_identity_sha256": node_desc.get("manifest_identity_sha256", node_desc.get("source_manifest_identity_sha256")),
        "home_split_label_sha256": node_desc["home_split_label_sha256"],
        "source_clone_files": len(fingerprint_files),
        "source_clone_stats_sha256": source_fingerprint,
        "n_seed_groups": int(len(meta)), "n_eligible_motifs": len(motifs),
        "n_duplicate_motifs_excluded": len(duplicated),
        "n_invalid_seed_groups_excluded": len(invalid_seed),
        "n_nonprojectable_seed_groups": int(len(meta) - len(motifs)),
        "n_allowed_seed_vj_pairs": int(sum(len(v) for v in allowed_pairs.values())),
        "scanned_qc_clone_rows": scanned, "valid_vj_clone_rows_scanned": valid,
        "matched_valid_clone_rows": matched_clone_rows,
        "train_valid_vj_clone_rows_scanned": train_valid,
        "train_matched_valid_clone_rows": train_matched_clone_rows,
        "unassigned_valid_clone_row_fraction": (1 - matched_clone_rows / valid) if valid else None,
        "n_exact_aa_vj_memberships": int(len(hits)),
        "n_unique_aa_assigned": int(hits.aa.nunique()),
        "n_ambiguous_aa_vj_nodes": n_ambiguous,
        "ambiguous_aa_vj_fraction_of_assigned": ambiguity_fraction,
        "train_ambiguous_aa_vj_fraction_of_assigned": train_ambiguity_fraction,
        "largest_group_fraction_of_valid_clone_rows": float(giant),
        "train_largest_group_fraction_of_valid_clone_rows": float(train_giant),
        "acceptance_population": "dictionary-fit Home patients only; held-out/external are diagnostics",
        "frozen_specificity_gates": {"largest_group_max_valid_clone_fraction": 0.05,
                                     "ambiguous_max_assigned_aa_vj_fraction": 0.05},
        "elapsed_seconds": time.monotonic() - started,
        "rss_ru_maxrss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "membership_sha256": _sha256(path),
        "seq_idx_global_bank_valid": max_scan_rows is None,
        "pilot_sampling": pilot_sampling,
        "seed_membership_sha256": _sha256(seed_path),
        "seed_import_receipt_sha256": _sha256(seed_receipt_path),
        "group_metadata_sha256": _sha256(metadata_path),
        "completed_utc": datetime.now(timezone.utc).isoformat(),
        "labels_read": False,
    }
    if status == "complete_home_scope" and train_matched_clone_rows == 0:
        report["status"] = "STOP_no_training_motif_matches"
        report["reason"] = "No eligible motif matched dictionary-fit Home clone rows"
    if status == "complete_home_scope" and train_giant > 0.05:
        report["status"] = "STOP_giant_motif_family"
        report["reason"] = "Largest seed-derived motif family matches >5% of training QC clone rows"
    if status == "complete_home_scope" and train_ambiguity_fraction > 0.05:
        report["status"] = "STOP_ambiguous_motif_projection"
        report["reason"] = ">5% of training assigned AA/V/J nodes map to multiple seed-derived motif families"
    _json(receipt_path if max_scan_rows is None else
          out / "gliph2_projection_pilot_receipt.json", report)
    return path


def _check_projection_complete(out: Path, membership: pd.DataFrame) -> None:
    """Never permit seed-only families into patient outcome association or MIL."""
    if membership.empty:
        raise ValueError("No projected groups")
    methods = set(membership.method.astype(str))
    receipts = {"gliph2": out / "gliph2_projection_receipt.json",
                "tcrdist3": out / "tcrdist_home_projection_receipt.json"}
    for method in methods:
        receipt_path = receipts.get(method)
        if receipt_path is None or not receipt_path.exists():
            raise ValueError(f"STOP_partial_projection: missing {method} projection receipt")
        receipt = json.loads(receipt_path.read_text())
        if receipt.get("status") != "complete_home_scope" or receipt.get("projection_population") != "home":
            raise ValueError(f"STOP_partial_projection: {method} is {receipt.get('status')}")
        source_path = out / ("gliph2_group_membership.parquet" if method == "gliph2" else
                             "tcrdist_home_group_membership.parquet")
        if receipt.get("membership_sha256") != _sha256(source_path):
            raise ValueError(f"STOP_partial_projection: {method} membership hash mismatch")
        if receipt.get("scope") != out.name:
            raise ValueError(f"STOP_partial_projection: {method} scope mismatch")
        if method == "tcrdist3":
            total = receipt.get("source_nodes_total")
            scorable = receipt.get("scorable_nodes_total")
            evaluated = receipt.get("evaluated_nodes")
            unscorable = receipt.get("n_unscorable_v_nodes")
            if not all(isinstance(n, int) and n >= 0 for n in
                       (total, scorable, evaluated, unscorable)) or\
                    total != scorable + unscorable or evaluated != scorable or\
                    receipt.get("scorable_nodes_projected_fraction") != 1.0 or\
                    (unscorable > 0 and not receipt.get("unscorable_v_nodes_sha256")):
                raise ValueError("STOP_partial_projection: tcrdist scorable-node coverage incomplete")


def integrate_projections(root: Path, scope: str) -> Path:
    out = root / "groups" / scope
    node_desc = _validate_scope_nodes(root, scope)
    parts = []
    statuses = {}
    coverage_gap = []
    method_source_sha256s = {}
    method_receipt_sha256s = {}
    specs = (("gliph2", "gliph2_group_membership.parquet", "gliph2_projection_receipt.json"),
             ("tcrdist3", "tcrdist_home_group_membership.parquet", "tcrdist_home_projection_receipt.json"))
    for method, filename, receipt_name in specs:
        path = out / filename
        receipt_path = out / receipt_name
        if not receipt_path.exists():
            raise ValueError(f"STOP_unattempted_method: {method} has no projection receipt")
        receipt = json.loads(receipt_path.read_text())
        method_receipt_sha256s[method] = _sha256(receipt_path)
        status = str(receipt.get("status", ""))
        statuses[method] = status
        if status == "complete_home_scope":
            if receipt.get("scope") != scope or receipt.get("method") != method or\
                    receipt.get("train_patient_sha256") != node_desc["train_patient_sha256"] or\
                    receipt.get("node_description_sha256") != _sha256(out / "description.json") or\
                    receipt.get("folds_sha256") != node_desc.get("folds_sha256"):
                raise ValueError(f"{method} projection is not bound to current training scope")
            if not path.exists():
                raise ValueError(f"{method} PASS receipt lacks full membership")
            frame = pd.read_parquet(path)
            if not frame.method.eq(method).all() or not frame.scope.eq(scope).all():
                raise ValueError(f"Method/scope mismatch in {path}")
            _check_projection_complete(out, frame)
            method_source_sha256s[method] = _sha256(path)
            parts.append(frame[["scope", "method", "group_id", "aa", "v_gene", "j_gene", "seq_idx"]])
        elif status.startswith("STOP") or status.lower() in {"failed", "resource_failed", "inconclusive", "infeasible"}:
            if receipt.get("scope") != scope or receipt.get("method") != method or not receipt.get("reason"):
                raise ValueError(f"{method} STOP receipt lacks scope, method or reason")
            coverage_gap.append(method)
        else:
            raise ValueError(f"{method} projection status is neither complete nor explicit STOP: {status}")
    if not parts:
        raise ValueError("STOP_partial_projection: no completed family projection")
    membership = pd.concat(parts, ignore_index=True).drop_duplicates()
    membership["projection_type"] = membership.method.map({
        "gliph2": "gliph2_seed_motif_projection",
        "tcrdist3": "tcrdist3_train_seed_full_projection"})
    if membership.group_id.nunique() != membership[["group_id", "method"]].drop_duplicates().shape[0]:
        raise ValueError("Group IDs collide across methods")
    path = out / "group_membership.parquet"
    membership.to_parquet(path, index=False)
    _json(out / "integrated_projection_receipt.json", {
        "status": "complete_home_scope", "scope": scope,
        "projection_population": "home",
        "methods": sorted(set(membership.method)),
        "method_statuses": statuses, "coverage_gap_methods": coverage_gap,
        "method_source_sha256s": method_source_sha256s,
        "method_receipt_sha256s": method_receipt_sha256s,
        "node_description_sha256": _sha256(out / "description.json"),
        "folds_sha256": node_desc.get("folds_sha256"),
        "manifest_identity_sha256": node_desc.get("manifest_identity_sha256", node_desc.get("source_manifest_identity_sha256")),
        "home_split_label_sha256": node_desc["home_split_label_sha256"],
        "n_groups": int(membership.group_id.nunique()),
        "n_exact_aa_vj_memberships": int(len(membership)),
        "membership_sha256": _sha256(path),
        "train_patient_sha256": node_desc["train_patient_sha256"],
    })
    return path


def _validate_integrated_scope(root: Path, scope: str) -> pd.DataFrame:
    out = root / "groups" / scope
    node_desc = _validate_scope_nodes(root, scope)
    path = out / "group_membership.parquet"
    receipt_path = out / "integrated_projection_receipt.json"
    if not receipt_path.exists():
        raise ValueError("STOP_partial_projection: integrated receipt absent")
    receipt = json.loads(receipt_path.read_text())
    if receipt.get("status") != "complete_home_scope" or receipt.get("projection_population") != "home" or receipt.get("scope") != scope or\
            not set(receipt.get("methods", [])).issubset({"gliph2", "tcrdist3"}) or not receipt.get("methods") or\
            receipt.get("membership_sha256") != _sha256(path) or\
            receipt.get("train_patient_sha256") != node_desc["train_patient_sha256"] or\
            receipt.get("node_description_sha256") != _sha256(out / "description.json") or\
            receipt.get("folds_sha256") != node_desc.get("folds_sha256") or\
            receipt.get("manifest_identity_sha256") != node_desc.get("manifest_identity_sha256", node_desc.get("source_manifest_identity_sha256")) or\
            receipt.get("home_split_label_sha256") != node_desc["home_split_label_sha256"]:
        raise ValueError("STOP_partial_projection: integrated membership receipt invalid")
    statuses = receipt.get("method_statuses", {})
    if set(statuses) != {"gliph2", "tcrdist3"}:
        raise ValueError("Integrated method attempt statuses incomplete")
    receipt_hashes = receipt.get("method_receipt_sha256s", {})
    source_hashes = receipt.get("method_source_sha256s", {})
    if set(receipt_hashes) != set(statuses) or set(source_hashes) != set(receipt["methods"]):
        raise ValueError("Integrated source/receipt hash inventory incomplete")
    for method, status in statuses.items():
        method_receipt = out / ("gliph2_projection_receipt.json" if method == "gliph2" else
                                "tcrdist_home_projection_receipt.json")
        method_data = json.loads(method_receipt.read_text())
        if receipt_hashes[method] != _sha256(method_receipt) or\
                method_data.get("status") != status or\
                method_data.get("scope") != scope or method_data.get("method") != method:
            raise ValueError("Method projection receipt changed after integration")
        if method in receipt["methods"] and status != "complete_home_scope":
            raise ValueError("Included method lacks complete status")
        if method in receipt["methods"]:
            if method_data.get("train_patient_sha256") != node_desc["train_patient_sha256"] or\
                    method_data.get("node_description_sha256") != _sha256(out / "description.json") or\
                    method_data.get("folds_sha256") != node_desc.get("folds_sha256"):
                raise ValueError("Included method projection source chain changed")
            if method == "gliph2" and (method_data.get("group_metadata_sha256") !=
                    _sha256(out / "group_metadata.tsv") or
                    method_data.get("seed_membership_sha256") !=
                    _sha256(out / "gliph2_seed_membership.parquet") or
                    method_data.get("seed_import_receipt_sha256") !=
                    _sha256(out / "membership_receipt.json")):
                raise ValueError("GLIPH2 seed motif definition changed after projection")
            method_source = out / ("gliph2_group_membership.parquet" if method == "gliph2" else
                                   "tcrdist_home_group_membership.parquet")
            if source_hashes[method] != _sha256(method_source):
                raise ValueError("Method membership changed after integration")
        if method not in receipt["methods"] and (not method_data.get("reason") or
                not (str(status).startswith("STOP") or
                     str(status).lower() in {"failed", "resource_failed", "inconclusive", "infeasible"})):
            raise ValueError("Excluded method lacks explicit failed receipt")
    membership = pd.read_parquet(path)
    if set(membership.method) != set(receipt["methods"]):
        raise ValueError("Integrated method list differs from membership")
    _check_projection_complete(out, membership)
    return membership


def _validate_frozen_file(root: Path, frozen_candidates: Path) -> pd.DataFrame:
    expected = (root / "groups/frozen_home_candidates.tsv").resolve()
    if frozen_candidates.resolve() != expected:
        raise ValueError("External projection requires the canonical frozen Home candidate file")
    receipt_path = expected.with_suffix(".json")
    if not receipt_path.exists():
        raise ValueError("Frozen Home candidate receipt missing")
    receipt = json.loads(receipt_path.read_text())
    source = root / "groups/full_home/association_all_recorded_home_exploratory.tsv"
    source_table = _validate_association_receipt(root, "full_home", "all_recorded_home_exploratory")
    _validate_association_receipt(root, "full_home", "verified_blood")
    blood_source = root / "groups/full_home/association_verified_blood.tsv"
    if receipt.get("status") != "complete" or receipt.get("candidate_sha256") != _sha256(expected) or\
            receipt.get("q_strictly_less_than") != 0.05 or receipt.get("max_frozen_candidates") is not None or\
            receipt.get("direction_rule") != "Home adjusted OR>1 and Home patient-presence delta>0; external patient-presence delta>0 with >=5 carriers" or\
            receipt.get("source_subset") != "all_recorded_home_exploratory" or\
            receipt.get("candidate_truncation") is not False or\
            receipt.get("candidate_policy") != "all_eligible_home_primary_groups" or\
            receipt.get("candidate_sha256") != receipt.get("eligible_home_candidates_sha256") or\
            receipt.get("n_candidates") != receipt.get("n_eligible_home_candidates") or\
            Path(receipt.get("source_association", "")).resolve() != source.resolve() or\
            receipt.get("membership_sha256") != _sha256(root / "groups/full_home/group_membership.parquet") or\
            receipt.get("source_association_sha256") != _sha256(source) or\
            receipt.get("source_association_receipt_sha256") != _sha256(source.with_suffix(".json")) or\
            receipt.get("source_eligible_groups_sha256") != _sha256(root / "groups/full_home/eligible_groups_all_recorded_home_exploratory.tsv") or\
            receipt.get("home_burden_sha256") != _sha256(root / "groups/full_home/patient_group_burden.parquet") or\
            receipt.get("home_burden_receipt_sha256") != _sha256(root / "groups/full_home/burden_receipt.json") or\
            receipt.get("manifest_identity_sha256") != _manifest_identity_sha256(root) or\
            receipt.get("home_split_label_sha256") != _home_split_label_sha256(root) or\
            receipt.get("clinical_covariates_sha256") != _sha256(root / "config/home_clinical_covariates.tsv") or\
            receipt.get("documented_blood_sensitivity_sha256") != _sha256(blood_source) or\
            receipt.get("documented_blood_sensitivity_receipt_sha256") != _sha256(blood_source.with_suffix(".json")):
        raise ValueError("Frozen Home candidate receipt/hash/threshold invalid")
    frozen = pd.read_csv(expected, sep="\t", dtype=str)
    ranked = source_table.loc[source_table.adjusted_model_status.eq("ok") &
                              source_table.adjusted_presence_q_pooled.lt(0.05) &
                              source_table.adjusted_presence_or.gt(1) &
                              source_table.presence_delta.gt(0)].sort_values(
                                  ["adjusted_presence_q_pooled", "group_id"])
    if frozen.group_id.tolist() != ranked.group_id.astype(str).tolist() or\
            "home_presence_delta" not in frozen or\
            not np.allclose(frozen.home_presence_delta.astype(float), ranked.presence_delta.to_numpy(), rtol=1e-9, atol=1e-12) or\
            len(frozen) != receipt.get("n_candidates"):
        raise ValueError("Frozen Home groups differ from primary association ranking")
    return frozen


def patient_burden(root: Path, scope: str, target_patients: set[str] | None = None,
                   cohort: str = "home", group_ids: set[str] | None = None,
                   projection_membership_path: Path | None = None) -> pd.DataFrame:
    out = root / "groups" / scope
    dictionary = _validate_integrated_scope(root, scope)
    if group_ids is not None:
        unknown = group_ids - set(dictionary.group_id)
        if unknown:
            raise ValueError(f"Requested groups not in frozen dictionary: {sorted(unknown)[:4]}")
    if cohort == "external":
        canonical = out / "external_frozen_membership.parquet"
        if target_patients is None or group_ids is None or projection_membership_path is None or\
                projection_membership_path.resolve() != canonical.resolve():
            raise ValueError("External burden requires frozen-rule external membership and patient IDs")
        membership = pd.read_parquet(canonical)
        required = {"scope", "method", "group_id", "aa", "v_gene", "j_gene", "seq_idx"}
        if required - set(membership) or not set(membership.group_id).issubset(group_ids) or\
                not membership.scope.eq(scope).all():
            raise ValueError("External frozen membership schema/group scope invalid")
    else:
        if projection_membership_path is not None:
            raise ValueError("Home burden cannot use an external membership override")
        membership = dictionary
    if group_ids is not None:
        membership = membership.loc[membership.group_id.isin(group_ids)]
    # Exact AA/V/J projection. For fold-local scopes, evaluate all Home patients,
    # including held-out patients, against the already frozen train dictionary.
    if target_patients is None:
        target_patients = _scope_patients(root, "full_home")
    patient_nodes = _load_patient_nodes(root, target_patients, cohort=cohort)
    totals = patient_nodes.groupby("patient_key").agg(
        total_reads=("read_count", "sum"), total_unique_aa=("aa", "nunique")).reset_index()
    hits = patient_nodes.merge(membership[["group_id", "seq_idx"] + KEY], on=KEY, how="inner")
    all_variants = patient_nodes.groupby(["patient_key", "aa"], as_index=False).size().rename(
        columns={"size": "all_vj_variants"})
    group_variants = hits.groupby(["patient_key", "group_id", "aa", "seq_idx"], as_index=False).size().rename(
        columns={"size": "member_vj_variants"})
    safe = group_variants.merge(all_variants, on=["patient_key", "aa"], how="left")
    safe = safe.loc[safe.member_vj_variants.eq(safe.all_vj_variants),
                    ["patient_key", "group_id", "seq_idx", "aa"]].drop_duplicates()
    safe.insert(0, "scope", scope)
    safe["projection_exact"] = True
    burden = hits.groupby(["patient_key", "group_id"], as_index=False).agg(
        group_reads=("read_count", "sum"), group_unique_aa=("aa", "nunique"),
        group_unique_aa_vj=("aa", "size"))
    patients = pd.DataFrame({"patient_key": sorted(target_patients)})
    # Qualification is blind to N and uses only dictionary-training patients.
    # It bounds the dense matrix without selecting families using outer holdout.
    train_patients = _scope_patients(root, scope)
    if cohort == "home":
        group_hits_train = burden.loc[burden.patient_key.isin(train_patients)].groupby("group_id").patient_key.nunique()
        floor = 10 if scope == "full_home" else 3
        qualified = set(group_hits_train.loc[group_hits_train.ge(floor)].index)
    else:
        home_burden, _ = _validate_burden_receipt(root, scope, cohort="home")
        qualified = set(home_burden.group_id.astype(str))
    if group_ids is not None:
        qualified &= group_ids
    burden = burden.loc[burden.group_id.isin(qualified)]
    safe = safe.loc[safe.group_id.isin(qualified)]
    safe_path = out / ("patient_group_aa_safe.parquet" if cohort == "home" else
                       f"patient_group_aa_safe_{cohort}.parquet")
    safe.to_parquet(safe_path, index=False)
    groups = pd.DataFrame({"group_id": sorted(qualified)})
    if len(patients) * len(groups) > 5_000_000:
        raise ValueError("Dense patient x eligible group matrix exceeds 5M rows; use sparse design")
    full = patients.merge(groups, how="cross").merge(burden, on=["patient_key", "group_id"], how="left")
    for col in ("group_reads", "group_unique_aa", "group_unique_aa_vj"):
        full[col] = full[col].fillna(0)
    full = full.merge(totals, on="patient_key", how="left", validate="many_to_one")
    full["presence"] = full.group_unique_aa.gt(0).astype(int)
    full["unique_aa_fraction"] = full.group_unique_aa / full.total_unique_aa
    full["read_fraction"] = full.group_reads / full.total_reads
    full.insert(0, "scope", scope)
    _assert_complete_burden_grid(full, target_patients, qualified, scope)
    burden_path = out / ("patient_group_burden.parquet" if cohort == "home" else
                         f"patient_group_burden_{cohort}.parquet")
    full.to_parquet(burden_path, index=False)
    _json(out / ("burden_receipt.json" if cohort == "home" else f"burden_receipt_{cohort}.json"), {
        "status": "complete", "scope": scope, "target_cohort": cohort,
        "target_patient_ids": sorted(target_patients),
        "target_patient_sha256": _patient_hash(target_patients),
        "train_patient_count": len(train_patients),
        "heldout_patient_count": len(target_patients - train_patients) if cohort == "home" else None,
        "n_patients": len(patients), "n_groups": len(groups),
        "qualified_group_ids_sha256": _patient_hash(qualified),
        "all_dictionary_groups": int(dictionary.group_id.nunique()),
        "external_frozen_membership_sha256": _sha256(out / "external_frozen_membership.parquet") if cohort == "external" else None,
        "external_frozen_membership_receipt_sha256": _sha256(out / "external_frozen_membership_receipt.json") if cohort == "external" else None,
        "label_blind_min_train_patients_present": 10 if scope == "full_home" else 3,
        "membership_sha256": _sha256(out / "group_membership.parquet"),
        "group_membership_sha256": _sha256(out / "group_membership.parquet"),
        "patient_group_burden_sha256": _sha256(burden_path),
        "patient_group_aa_safe_sha256": _sha256(safe_path),
        "description_sha256": _sha256(out / "description.json"),
        "integrated_projection_receipt_sha256": _sha256(out / "integrated_projection_receipt.json"),
        "folds_sha256": _sha256(root / "folds/folds.tsv"),
        "manifest_identity_sha256": _manifest_identity_sha256(root),
        "home_split_label_sha256": _home_split_label_sha256(root),
        "n_rows": len(full), "projection": "exact_aa_v_gene_j_gene",
        "safe_aa_deletion_rows": int(len(safe)),
        "safe_aa_deletion_interpretation": "per-patient all observed V/J variants match frozen family",
        "train_patient_sha256": _patient_hash(_scope_patients(root, scope)),
        "read_count_interpretation": "MiXCR reads, not cells",
    })
    return full


def _assert_complete_burden_grid(frame: pd.DataFrame, patients: set[str],
                                 groups: set[str], scope: str) -> None:
    """A frozen patient/group design includes explicit zero-carrier rows."""
    if not patients or not groups or len(frame) != len(patients) * len(groups) or\
            frame[["patient_key", "group_id"]].duplicated().any() or\
            set(frame.patient_key.astype(str)) != patients or\
            set(frame.group_id.astype(str)) != groups or\
            not frame.scope.astype(str).eq(scope).all():
        raise ValueError("Patient x frozen-group burden grid is incomplete or duplicated")


def _validate_burden_receipt(root: Path, scope: str,
                             cohort: str = "home") -> tuple[pd.DataFrame, dict]:
    """Bind group burden, exact membership and patient identities to sources."""
    out = root / "groups" / scope
    _validate_integrated_scope(root, scope)
    suffix = "" if cohort == "home" else f"_{cohort}"
    receipt_path = out / f"burden_receipt{suffix}.json"
    burden_path = out / f"patient_group_burden{suffix}.parquet"
    safe_path = out / f"patient_group_aa_safe{suffix}.parquet"
    receipt = json.loads(receipt_path.read_text())
    manifest_path = root / "config/cohort_samples.tsv"
    manifest = pd.read_csv(manifest_path, sep="\t", dtype=str,
                           usecols=["cohort", "patient_key"])
    patients = set(manifest.loc[manifest.cohort.eq(cohort), "patient_key"].dropna())
    patients.discard("")
    if receipt.get("status") != "complete" or receipt.get("scope") != scope or\
            receipt.get("target_cohort") != cohort or\
            receipt.get("target_patient_sha256") != _patient_hash(patients) or\
            set(receipt.get("target_patient_ids", [])) != patients or\
            receipt.get("manifest_identity_sha256") != _manifest_identity_sha256(root) or\
            receipt.get("home_split_label_sha256") != _home_split_label_sha256(root) or\
            receipt.get("train_patient_sha256") != _patient_hash(_scope_patients(root, scope)) or\
            receipt.get("folds_sha256") != _sha256(root / "folds/folds.tsv") or\
            receipt.get("patient_group_burden_sha256") != _sha256(burden_path) or\
            receipt.get("patient_group_aa_safe_sha256") != _sha256(safe_path) or\
            receipt.get("group_membership_sha256") != _sha256(out / "group_membership.parquet") or\
            receipt.get("description_sha256") != _sha256(out / "description.json") or\
            receipt.get("integrated_projection_receipt_sha256") != _sha256(out / "integrated_projection_receipt.json") or\
            (cohort == "external" and (
                receipt.get("external_frozen_membership_sha256") != _sha256(out / "external_frozen_membership.parquet") or
                receipt.get("external_frozen_membership_receipt_sha256") != _sha256(out / "external_frozen_membership_receipt.json"))):
        raise ValueError("Burden receipt does not match completed frozen-rule projection")
    frame = pd.read_parquet(burden_path)
    groups = set(frame.group_id.astype(str))
    if len(groups) != receipt.get("n_groups") or len(frame) != receipt.get("n_rows") or\
            receipt.get("qualified_group_ids_sha256") != _patient_hash(groups):
        raise ValueError("Burden row/group counts differ from receipt")
    _assert_complete_burden_grid(frame, patients, groups, scope)
    return frame, receipt


def _bh(p: pd.Series) -> np.ndarray:
    values = p.to_numpy(dtype=float)
    # Failed models count as p=1 in the declared family, never disappear from BH.
    values = np.where(np.isfinite(values), values, 1.0)
    return multipletests(values, method="fdr_bh")[1]


def _adjusted_presence(part: pd.DataFrame) -> tuple[float, float, str, int, int]:
    import warnings
    import statsmodels.api as sm
    from statsmodels.tools.sm_exceptions import PerfectSeparationWarning

    y = part.recorded_n_binary.to_numpy(dtype=float)
    x = pd.DataFrame({
        "presence": part.presence.to_numpy(dtype=float),
        "log_reads": np.log1p(part.total_reads.to_numpy(dtype=float)),
        "log_unique_aa": np.log1p(part.total_unique_aa.to_numpy(dtype=float)),
    }, index=part.index)
    for col in ("log_reads", "log_unique_aa"):
        sd = float(x[col].std())
        x[col] = (x[col] - float(x[col].mean())) / sd if sd > 0 else 0.0
    age = pd.to_numeric(part.age_years, errors="coerce")
    if age.isna().any():
        return np.nan, np.nan, "missing_age", 0, 0
    age_sd = float(age.std())
    x["age_years"] = (age - float(age.mean())) / age_sd if age_sd > 0 else 0.0
    if age_sd == 0:
        x = x.drop(columns=["age_years"])
    sex = pd.get_dummies(part.sex.astype(str), prefix="sex", drop_first=True, dtype=float)
    batch = pd.get_dummies(part.source_batch.astype(str), prefix="batch", drop_first=True, dtype=float)
    x = pd.concat([x, sex], axis=1)
    x = pd.concat([x, batch], axis=1)
    x = sm.add_constant(x, has_constant="add")
    rank, n_parameters = int(np.linalg.matrix_rank(x.to_numpy())), int(x.shape[1])
    if rank != n_parameters:
        return np.nan, np.nan, "rank_deficient", rank, n_parameters
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", PerfectSeparationWarning)
            fit = sm.GLM(y, x, family=sm.families.Binomial()).fit(maxiter=100, disp=0)
        beta = float(fit.params["presence"])
        se = float(fit.bse["presence"])
        p = float(fit.pvalues["presence"])
        if not fit.converged or not np.isfinite([beta, se, p]).all() or abs(beta) > 10 or se > 10:
            return np.nan, np.nan, "separation_or_unstable", rank, n_parameters
        return float(np.exp(beta)), p, "ok", rank, n_parameters
    except Exception as exc:
        return np.nan, np.nan, f"{type(exc).__name__}", rank, n_parameters


def association_table(burden: pd.DataFrame, labels: pd.DataFrame,
                      min_present: int = 3, min_absent: int = 3) -> pd.DataFrame:
    labels = labels[["patient_key", "recorded_n_binary", "source_batch", "age_years", "sex"]].drop_duplicates("patient_key")
    frame = burden.merge(labels, on="patient_key", how="inner", validate="many_to_one")
    frame["recorded_n_binary"] = pd.to_numeric(frame.recorded_n_binary, errors="coerce")
    frame = frame.loc[frame.recorded_n_binary.isin([0, 1])].copy()
    groups = []
    for group_id, part in frame.groupby("group_id", sort=True):
        n_patients = part.patient_key.nunique()
        if n_patients != labels.patient_key.nunique():
            raise ValueError(f"Incomplete patient burden for group {group_id}")
        present = int(part.presence.sum())
        if present < min_present or n_patients - present < min_absent:
            continue
        y = part.recorded_n_binary.to_numpy(dtype=int)
        pres = part.presence.to_numpy(dtype=int)
        table = np.array([
            [int(np.sum((y == 1) & (pres == 1))), int(np.sum((y == 1) & (pres == 0)))],
            [int(np.sum((y == 0) & (pres == 1))), int(np.sum((y == 0) & (pres == 0)))],
        ])
        odds, p_presence = fisher_exact(table, alternative="two-sided")
        med_pos = float(np.median(part.loc[y == 1, "read_fraction"]))
        med_neg = float(np.median(part.loc[y == 0, "read_fraction"]))
        try:
            p_read = float(mannwhitneyu(part.loc[y == 1, "read_fraction"],
                                        part.loc[y == 0, "read_fraction"],
                                        alternative="two-sided").pvalue)
        except ValueError:
            p_read = 1.0
        n_informative_batches = sum(
            batch_part.recorded_n_binary.nunique() == 2 and batch_part.presence.nunique() == 2
            for _, batch_part in part.groupby("source_batch"))
        adj_or, adj_p, adj_status, model_rank, model_n_parameters = _adjusted_presence(part)
        groups.append({
            "group_id": group_id, "n_patients": n_patients, "n_present": present,
            "positive_present": int(table[0, 0]), "negative_present": int(table[1, 0]),
            "presence_delta": float(table[0, 0] / table[0].sum() - table[1, 0] / table[1].sum()),
            "n_informative_batches": n_informative_batches,
            "model_rank": model_rank, "model_n_parameters": model_n_parameters,
            "presence_or_unadjusted": float(odds), "presence_p_unadjusted": float(p_presence),
            "adjusted_presence_or": adj_or, "adjusted_presence_p": adj_p,
            "adjusted_model_status": adj_status,
            "read_fraction_median_positive": med_pos,
            "read_fraction_median_negative": med_neg,
            "read_fraction_delta_median": med_pos - med_neg,
            "read_fraction_p_unadjusted": p_read,
        })
    columns = [
        "group_id", "n_patients", "n_present", "positive_present", "negative_present", "presence_delta",
        "n_informative_batches", "model_rank", "model_n_parameters",
        "presence_or_unadjusted", "presence_p_unadjusted",
        "adjusted_presence_or", "adjusted_presence_p", "adjusted_model_status",
        "read_fraction_median_positive", "read_fraction_median_negative",
        "read_fraction_delta_median", "read_fraction_p_unadjusted",
        "adjusted_presence_q_pooled", "presence_q_pooled_unadjusted",
        "read_fraction_q_pooled_unadjusted",
    ]
    result = pd.DataFrame(groups)
    if result.empty:
        return pd.DataFrame(columns=columns)
    # The single primary BH family is adjusted presence across all eligible
    # groups and methods. Fisher and read-fraction each have separate
    # prespecified sensitivity families; failures count as p=1.
    result["adjusted_presence_q_pooled"] = _bh(result.adjusted_presence_p)
    result["presence_q_pooled_unadjusted"] = _bh(result.presence_p_unadjusted)
    result["read_fraction_q_pooled_unadjusted"] = _bh(result.read_fraction_p_unadjusted)
    return result[columns].sort_values(
        ["adjusted_presence_q_pooled", "adjusted_presence_p", "group_id"]).reset_index(drop=True)


def _subset_patient_ids(root: Path, scope: str, subset: str) -> set[str]:
    eligible = _scope_patients(root, scope)
    if subset == "all_recorded_home_exploratory":
        return eligible
    if subset != "verified_blood":
        raise ValueError("Unknown analysis subset")
    doc = pd.read_csv(root / "config/documented_blood31.tsv", sep="\t", dtype=str,
                      keep_default_na=False, usecols=["patient_key", "documented_blood"])
    return eligible & set(doc.loc[doc.documented_blood.str.lower().eq("yes"), "patient_key"])


def _manifest_labels(root: Path, patients: set[str], subset: str) -> tuple[pd.DataFrame, dict]:
    manifest = pd.read_csv(root / "config/cohort_samples.tsv", sep="\t", dtype=str,
                           keep_default_na=False,
                           usecols=["cohort", "patient_key", "recorded_n_binary"])
    home = manifest.loc[manifest.cohort.str.lower().eq("home") &
                        manifest.patient_key.isin(patients)].copy()
    if home.patient_key.duplicated().any():
        repeated = home.groupby("patient_key").recorded_n_binary.nunique()
        if repeated.gt(1).any():
            raise ValueError("Conflicting Home labels")
    home = home.drop_duplicates("patient_key")
    cov_path = root / "config/home_clinical_covariates.tsv"
    cov = pd.read_csv(cov_path, sep="\t", dtype=str, keep_default_na=False,
                      usecols=["patient_key", "age_years", "sex", "library_batch"])
    if cov.patient_key.duplicated().any():
        raise ValueError("Duplicate clinical covariate patient keys")
    labels = home[["patient_key", "recorded_n_binary"]].merge(
        cov, on="patient_key", how="left", validate="one_to_one")
    labels = labels.rename(columns={"library_batch": "source_batch"})
    if labels[["age_years", "sex", "source_batch"]].isna().any().any() or\
            labels[["age_years", "sex", "source_batch"]].eq("").any().any():
        raise ValueError("Missing age, sex or batch for Home patient")
    if not labels.recorded_n_binary.isin(["0", "1"]).all():
        raise ValueError("Subset contains missing/nonbinary recorded N")
    if set(labels.patient_key) != patients:
        raise ValueError("Subset patients missing from manifest")
    return labels, {
        "subset": subset, "n_patients": int(labels.patient_key.nunique()),
        "n_positive": int(labels.recorded_n_binary.eq("1").sum()),
        "n_negative": int(labels.recorded_n_binary.eq("0").sum()),
        "material_interpretation": "report-explicit peripheral blood" if subset == "verified_blood" else
            "recorded Home N only; 83 source-material labels unverified unless provenance resolves",
        "source_batch_counts": labels.source_batch.value_counts().to_dict(),
        "clinical_covariates_sha256": _sha256(cov_path),
        "covariates": ["log_total_reads", "log_unique_aa", "age_years", "sex", "library_batch"],
    }


def _preoutcome_analysis_rules() -> dict:
    """Prospectively fixed Home test family and candidate decision contract."""
    return {
        "primary_population": _private_literal('private_090825d49c8c80bb'),
        "documented_blood_sensitivity_population": _private_literal('private_45a15b111d17e62a'),
        "method_family": ["gliph2_seed_motif_projection", "tcrdist3_train_seed_full_projection"],
        "completed_methods_only": True,
        "failed_method_policy": "explicit method-specific technical STOP receipt; never include seed-only membership",
        "home_projection_order": "Home-only complete projection, label-blind group eligibility, Home associations and candidate freeze, external frozen-rule projection, then external N score",
        "gliph2_seed_max_per_training_patient": 2048,
        "gliph2_family_definition": "official GLIPH2 training-seed motifs plus a derived literal-motif/TRBV/TRBJ projection; no official full-repertoire GLIPH2 cluster claim",
        "tcrdist_candidate_strategy": "exact_V_cost_pruning",
        "tcrdist_radius_grid": [16, 20, 24],
        "tcrdist_full_home_selected_radius": 24,
        "tcrdist_acceptance": "exact lower-bound recall 1.0; training graph giant <=0.10; projected giant nodes and reads <=0.10; all reference-scorable Home nodes evaluated",
        "primary_eligibility": "at least 10 of 114 Home patients carry and at least 10 do not carry, label-blind before Home N read",
        "primary_min_present": 10,
        "primary_min_absent": 10,
        "outer_train_min_present_and_absent": 3,
        "primary_test": "patient-level group presence GLM Binomial adjusted for log total reads, log unique AA, age, sex and library batch; rank/separation/unstable fits unavailable with p=1",
        "primary_multiple_testing": "one pooled BH across every eligible tested group from every completed method; failed adjusted models remain in family with p=1",
        "fisher_presence_and_mwu_read_fraction": "separate sensitivity BH families, never substituted for primary adjusted q",
        "home_direction": "adjusted OR>1 and patient-presence delta (N+ carrier rate minus N0 carrier rate)>0",
        "home_candidate_rule": _private_literal('private_402ba73402bf3595'),
        "fold_nomination_rule": "outer-train adjusted OR>1 and patient-presence delta>0, at most five per outer fold; group dictionaries fit without outer-heldout patients",
        "external_direction_rule": "frozen group presence delta>0 with at least five external proxy carriers; no parameter or family selection from external N",
        "external_identity_limit": "165 BioSample proxies, independence unverified; directional transport only",
        "sensitivity_plan": {
            _private_literal('private_b6b1b2517b10d5e5'): "required association and receipt before candidate freeze; non-significance descriptive",
            "unique_aa_fraction": "04c independent sensitivity; not primary q",
            "read_fraction": "04c independent sensitivity; not primary q",
            "equal_read_depth": "04c exact hypergeometric marginal per group and patient; not a joint repertoire draw",
            "vj_length_composition": "04c V/J and length background adjustment",
            "batch_strata_cmh": "04c batch-specific carrier rates and nominal CMH when >=2 informative batches; sparse/unestimable reported",
            "pgen": "04c OLGA post-selection unconditioned AA characterization if available, else explicit NA receipt; never a primary test or selection gate",
            "sceptr_nonstandard_junction": "post-candidate applicability sensitivity; primary QC/bank retain all AA",
        },
        "final_claim": "Home adjusted association plus frozen external direction, heldout OOF gain, exact safe family ablation and cross-fold alignment; no antigen specificity or causality implied",
    }


def lock_preoutcome_analysis(root: Path) -> Path:
    """Seal rules while no Home group association table has been written."""
    from datetime import datetime, timezone
    old_tables = list((root / "groups").glob("*/association_*.tsv"))
    if old_tables:
        raise ValueError("Cannot create or revise preoutcome lock after group associations exist")
    gliph_lock = root / "audit/04_gliph_seed_depth_lock.json"
    gliph = json.loads(gliph_lock.read_text())
    if gliph.get("status") != "frozen_before_outcomes" or gliph.get("decision") != "gliph2_seed_max_per_train_patient=2048":
        raise ValueError("GLIPH2 technical seed decision must be frozen first")
    tcr_cost = root / "groups/tcrdist_exact_v_project_cost_gate.json"
    tcr_proof = root / "groups/full_home/tcrdist_exact_v_lower_bound_diagnostic.json"
    tcr_graph = root / "groups/full_home/tcrdist_exact_v_graph_preflight.json"
    cost = json.loads(tcr_cost.read_text())
    proof = json.loads(tcr_proof.read_text())
    graph = json.loads(tcr_graph.read_text())
    if (cost.get("status") != "PASS_HOME_FIRST" or cost.get("radius") != 24 or
            cost.get("candidate_strategy") != "exact_V_cost_pruning" or
            cost.get("external_projection_locked_until_candidates_frozen") is not True or
            proof.get("status") != "PASS_TECHNICAL_LOWER_BOUND" or
            graph.get("status") != "PASS_TRAIN_GRAPH" or graph.get("selected_radius") != 24):
        raise ValueError("tcrdist exact-V r24 technical gates are not prospectively fixed")
    tcr_home = root / "groups/full_home/tcrdist_home_projection_receipt.json"
    tcr_status = json.loads(tcr_home.read_text()).get("status") if tcr_home.exists() else "pending"
    lock = root / "audit/04_preoutcome_analysis_lock.json"
    _json(lock, {
        "status": "frozen_before_association",
        "frozen_utc": datetime.now(timezone.utc).isoformat(),
        "association_tables_at_freeze": 0,
        "external_n_read_for_this_lock": False,
        "rules": _preoutcome_analysis_rules(),
        "tcrdist_home_status_at_freeze": tcr_status,
        "association_start_gate": "Home-only integrated projection complete or explicit technical STOP per method; Home burden and label-blind eligibility complete; no seed-only membership",
        "source_sha256": {
            "src/04_groups_stats.py": _sha256(Path(__file__)),
            "src/04b_tcrdist_projection.py": _sha256(root / "src/04b_tcrdist_projection.py"),
            "src/04c_group_sensitivities.py": _sha256(root / "src/04c_group_sensitivities.py"),
        },
        "technical_receipt_sha256": {
            "gliph2_seed_depth_lock": _sha256(gliph_lock),
            "gliph2_home_projection": _sha256(root / "groups/full_home/gliph2_projection_receipt.json"),
            "tcrdist_exact_v_project_cost_gate": _sha256(tcr_cost),
            "tcrdist_exact_v_lower_bound_diagnostic": _sha256(tcr_proof),
            "tcrdist_exact_v_graph_preflight": _sha256(tcr_graph),
        },
        "manifest_identity_sha256": _manifest_identity_sha256(root),
        "home_split_label_sha256": _home_split_label_sha256(root),
        "folds_sha256": _sha256(root / "folds/folds.tsv"),
        "clinical_covariates_sha256": _sha256(root / "config/home_clinical_covariates.tsv"),
        "documented_blood31_sha256": _sha256(root / "config/documented_blood31.tsv"),
        "group_scope_names": ["full_home"] + [f"outer_{i}" for i in range(5)],
    })
    _validate_preoutcome_analysis_lock(root)
    return lock


def _validate_preoutcome_analysis_lock(root: Path) -> dict:
    path = root / "audit/04_preoutcome_analysis_lock.json"
    lock = json.loads(path.read_text())
    paths = {
        "gliph2_seed_depth_lock": root / "audit/04_gliph_seed_depth_lock.json",
        "gliph2_home_projection": root / "groups/full_home/gliph2_projection_receipt.json",
        "tcrdist_exact_v_project_cost_gate": root / "groups/tcrdist_exact_v_project_cost_gate.json",
        "tcrdist_exact_v_lower_bound_diagnostic": root / "groups/full_home/tcrdist_exact_v_lower_bound_diagnostic.json",
        "tcrdist_exact_v_graph_preflight": root / "groups/full_home/tcrdist_exact_v_graph_preflight.json",
    }
    expected_sources = {name: _sha256(root / name) for name in (
        "src/04_groups_stats.py", "src/04b_tcrdist_projection.py", "src/04c_group_sensitivities.py")}
    if (lock.get("status") != "frozen_before_association" or
            lock.get("association_tables_at_freeze") != 0 or
            lock.get("external_n_read_for_this_lock") is not False or
            lock.get("rules") != _preoutcome_analysis_rules() or
            lock.get("source_sha256") != expected_sources or
            lock.get("technical_receipt_sha256") != {name: _sha256(file) for name, file in paths.items()} or
            lock.get("manifest_identity_sha256") != _manifest_identity_sha256(root) or
            lock.get("home_split_label_sha256") != _home_split_label_sha256(root) or
            lock.get("folds_sha256") != _sha256(root / "folds/folds.tsv") or
            lock.get("clinical_covariates_sha256") != _sha256(root / "config/home_clinical_covariates.tsv") or
            lock.get("documented_blood31_sha256") != _sha256(root / "config/documented_blood31.tsv") or
            lock.get("group_scope_names") != ["full_home"] + [f"outer_{i}" for i in range(5)]):
        raise ValueError("Prospective preoutcome analysis lock or source hash changed")
    return lock


def associate(root: Path, scope: str, subset: str, min_present: int | None = None,
              min_absent: int | None = None) -> Path:
    from datetime import datetime, timezone
    primary_full = scope == "full_home" and subset == "all_recorded_home_exploratory"
    locked_floor = 10 if primary_full else 3
    if min_present is None:
        min_present = locked_floor
    if min_absent is None:
        min_absent = locked_floor
    if min_present != locked_floor or min_absent != locked_floor:
        raise ValueError(f"Association for {scope}/{subset} locks exactly {locked_floor} present and absent patients")
    _validate_preoutcome_analysis_lock(root)
    out = root / "groups" / scope
    _validate_integrated_scope(root, scope)
    burden, burden_receipt = _validate_burden_receipt(root, scope, cohort="home")
    burden_receipt_path = out / "burden_receipt.json"
    eligible_patients = _subset_patient_ids(root, scope, subset)
    if not eligible_patients:
        raise ValueError("No eligible patients in analysis subset")
    selected = burden.loc[burden.patient_key.isin(eligible_patients)]
    counts = selected.groupby("group_id").presence.agg(["sum", "count"]).reset_index()
    if counts.empty or not counts["count"].eq(len(eligible_patients)).all():
        raise ValueError("Incomplete patient x group burden grid")
    eligible_groups = counts.loc[
        counts["sum"].ge(min_present) & (counts["count"] - counts["sum"]).ge(min_absent),
        ["group_id", "sum", "count"]].rename(columns={"sum": "n_present", "count": "n_patients"})
    eligible_path = out / f"eligible_groups_{subset}.tsv"
    eligible_groups.to_csv(eligible_path, sep="\t", index=False)
    eligibility_frozen_utc = datetime.now(timezone.utc).isoformat()
    # The outcome column is only read after the label-blind eligibility file
    # has been written and hashed.
    labels, context = _manifest_labels(root, eligible_patients, subset)
    selected = selected.loc[selected.group_id.isin(eligible_groups.group_id)]
    result = association_table(selected, labels, min_present, min_absent)
    context["role"] = "fold_train_nomination_only" if scope != "full_home" else "descriptive_home_association"
    path = out / f"association_{subset}.tsv"
    result.to_csv(path, sep="\t", index=False)
    _json(path.with_suffix(".json"), {
        **context, "scope": scope, "groups_tested": int(len(result)),
        "association_sha256": _sha256(path),
        "eligible_groups_sha256": _sha256(eligible_path),
        "burden_sha256": burden_receipt["patient_group_burden_sha256"],
        "burden_receipt_sha256": _sha256(burden_receipt_path),
        "integrated_projection_receipt_sha256": _sha256(out / "integrated_projection_receipt.json"),
        "manifest_identity_sha256": _manifest_identity_sha256(root),
        "home_split_label_sha256": _home_split_label_sha256(root),
        "folds_sha256": _sha256(root / "folds/folds.tsv"),
        "subset_patient_sha256": _patient_hash(eligible_patients),
        "documented_blood_sha256": _sha256(root / "config/documented_blood31.tsv") if subset == "verified_blood" else None,
        "eligibility_frozen_utc": eligibility_frozen_utc,
        "test_family": "primary pooled BH over adjusted presence for all eligible groups and methods; Fisher presence and Mann-Whitney read fraction are separate sensitivity families; failed adjusted models p=1 in primary family",
        "primary_test": "GLM Binomial presence adjusted for log total reads, log unique AA, age, sex, and library batch; unavailable on separation/instability",
        "direction_scale": "patient group presence prevalence delta: N+ carriers/N+ patients minus N0 carriers/N0 patients",
        "min_present": min_present, "min_absent": min_absent,
        "no_causal_claim": True,
    })
    return path


def _validate_association_receipt(root: Path, scope: str, subset: str) -> pd.DataFrame:
    out = root / "groups" / scope
    path = out / f"association_{subset}.tsv"
    receipt_path = path.with_suffix(".json")
    receipt = json.loads(receipt_path.read_text())
    burden, burden_receipt = _validate_burden_receipt(root, scope, cohort="home")
    eligible_path = out / f"eligible_groups_{subset}.tsv"
    patients = _subset_patient_ids(root, scope, subset)
    expected_min = 10 if scope == "full_home" and subset == "all_recorded_home_exploratory" else 3
    if receipt.get("scope") != scope or receipt.get("subset") != subset or\
            receipt.get("association_sha256") != _sha256(path) or\
            receipt.get("eligible_groups_sha256") != _sha256(eligible_path) or\
            receipt.get("burden_sha256") != burden_receipt["patient_group_burden_sha256"] or\
            receipt.get("burden_receipt_sha256") != _sha256(out / "burden_receipt.json") or\
            receipt.get("integrated_projection_receipt_sha256") != _sha256(out / "integrated_projection_receipt.json") or\
            receipt.get("manifest_identity_sha256") != _manifest_identity_sha256(root) or\
            receipt.get("home_split_label_sha256") != _home_split_label_sha256(root) or\
            receipt.get("clinical_covariates_sha256") != _sha256(root / "config/home_clinical_covariates.tsv") or\
            receipt.get("folds_sha256") != _sha256(root / "folds/folds.tsv") or\
            receipt.get("subset_patient_sha256") != _patient_hash(patients) or\
            receipt.get("direction_scale") != "patient group presence prevalence delta: N+ carriers/N+ patients minus N0 carriers/N0 patients" or\
            receipt.get("min_present") != expected_min or receipt.get("min_absent") != expected_min:
        raise ValueError("Association source/eligibility/threshold receipt invalid")
    if subset == "verified_blood" and receipt.get("documented_blood_sha256") !=\
            _sha256(root / "config/documented_blood31.tsv"):
        raise ValueError("Documented-blood subset changed after association")
    if scope == "full_home" and subset == "all_recorded_home_exploratory" and\
            (receipt.get("n_patients"), receipt.get("n_positive"), receipt.get("n_negative")) != (114, 48, 66):
        raise ValueError(_private_literal('private_6638b3b7026d19e5'))
    eligible = pd.read_csv(eligible_path, sep="\t")
    table = pd.read_csv(path, sep="\t")
    if table.group_id.duplicated().any() or set(table.group_id) != set(eligible.group_id) or\
            len(table) != receipt.get("groups_tested"):
        raise ValueError("Association table differs from label-blind test family")
    selected = burden.loc[burden.patient_key.isin(patients)]
    counts = selected.groupby("group_id").presence.agg(["sum", "count"])
    recomputed = set(counts.index[counts["sum"].ge(expected_min) &
                                  (counts["count"] - counts["sum"]).ge(expected_min)])
    if recomputed != set(eligible.group_id):
        raise ValueError("Eligible group universe changed after association")
    if len(table) and not np.allclose(_bh(table.adjusted_presence_p),
                                      table.adjusted_presence_q_pooled, equal_nan=True):
        raise ValueError("Primary pooled BH q-values do not match prespecified test family")
    if len(table) and ("presence_delta" not in table or
            not np.allclose(table.presence_delta,
                            table.positive_present / receipt["n_positive"] -
                            table.negative_present / receipt["n_negative"], rtol=1e-9, atol=1e-12)):
        raise ValueError("Association patient-presence direction differs from locked class counts")
    return table


def nominate(root: Path, subset: str = "all_recorded_home_exploratory", top_n: int = 5) -> Path:
    if not 1 <= top_n <= 5:
        raise ValueError("top_n must be 1..5")
    rows = []
    for outer in range(5):
        scope = f"outer_{outer}"
        path = root / "groups" / scope / f"association_{subset}.tsv"
        if not path.exists():
            raise FileNotFoundError(path)
        table = _validate_association_receipt(root, scope, subset)
        table = table.loc[table.adjusted_presence_or.gt(1) & table.presence_delta.gt(0) &
                          table.adjusted_model_status.eq("ok")].copy()
        table = table.sort_values(["adjusted_presence_q_pooled", "adjusted_presence_p", "group_id"]).head(top_n)
        for rank, (_, row) in enumerate(table.iterrows(), 1):
            rows.append({
                "group_id": row.group_id, "outer_fold": outer, "rank": rank,
                "train_patient_sha256": _patient_hash(_scope_patients(root, scope)),
                "scope": scope, "train_subset": subset,
                "adjusted_presence_q_pooled_train": row.adjusted_presence_q_pooled,
                "adjusted_presence_or_train": row.adjusted_presence_or,
                "presence_delta_train": row.presence_delta,
                "read_fraction_delta_train": row.read_fraction_delta_median,
            })
    result = pd.DataFrame(rows, columns=[
        "group_id", "outer_fold", "rank", "scope", "train_subset", "train_patient_sha256",
        "adjusted_presence_q_pooled_train", "adjusted_presence_or_train", "presence_delta_train",
        "read_fraction_delta_train"])
    output = root / "groups/fold_nominations.tsv"
    output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output, sep="\t", index=False)
    _json(output.with_suffix(".json"), {
        "status": "complete", "subset": subset, "max_per_outer_fold": top_n,
        "direction_rule": "train adjusted OR>1 and train patient-presence delta>0",
        "nominations_sha256": _sha256(output),
        "train_patient_sha256_by_outer": {str(outer): _patient_hash(_scope_patients(root, f"outer_{outer}"))
                                          for outer in range(5)},
        "association_sha256_by_outer": {str(outer): _sha256(root / "groups" / f"outer_{outer}" /
                                                          f"association_{subset}.tsv")
                                        for outer in range(5)},
        "folds_sha256": _sha256(root / "folds/folds.tsv"),
    })
    return output


def tcrdist_pilot(root: Path, scope: str, model: str, pilot_n: int = 800,
                   radius: int = 50, candidate_k: int = 128,
                   seed: int = 20260929) -> Path:
    """Exact tcrdist3 pilot with AA-bank candidate retrieval recall audit.

    This never authorizes a full 20M-node graph by itself. A lack of >=50
    exact-neighbor pairs is inconclusive and fails the full-run gate.
    """
    if not 100 <= pilot_n <= 2000:
        raise ValueError("pilot_n must be 100..2000; this is an exact small-scale pilot")
    if not 1 <= candidate_k < pilot_n:
        raise ValueError("candidate_k must be 1..pilot_n-1")
    from tcrdist.repertoire import TCRrep
    from pynndescent import NNDescent

    out = root / "groups" / scope / f"tcrdist_pilot_n{pilot_n}"
    out.mkdir(parents=True, exist_ok=True)
    nodes = pd.read_parquet(root / "groups" / scope / "clone_nodes.parquet")
    valid = nodes.v_gene.astype(str).str.match(r"^TRBV[0-9]+(?:-[0-9]+)?(?:\*[0-9]+)?$") &\
            nodes.j_gene.astype(str).str.match(r"^TRBJ[0-9]+(?:-[0-9]+)?(?:\*[0-9]+)?$")
    nodes = nodes.loc[valid].copy()
    if len(nodes) < pilot_n:
        raise ValueError(f"Only {len(nodes)} valid V/J nodes, fewer than pilot_n")
    # Deterministic patient-blind stratification over V and CDR3 length raises
    # the chance that the exact pilot contains true local neighbors.
    nodes["stratum"] = nodes.v_gene.map(_stem) + ":" + nodes.aa.str.len().astype(str)
    rng = np.random.default_rng(seed)
    strata = []
    for _, frame in nodes.groupby("stratum", sort=True):
        if len(frame) >= 20:
            strata.append(frame)
    if not strata:
        strata = [nodes]
    strata = sorted(strata, key=len, reverse=True)[:min(20, len(strata))]
    quota = max(1, pilot_n // len(strata))
    pieces = []
    for frame in strata:
        take = min(quota, len(frame))
        pieces.append(frame.iloc[rng.choice(len(frame), size=take, replace=False)])
    sample = pd.concat(pieces).drop_duplicates(KEY)
    if len(sample) < pilot_n:
        remaining = nodes.loc[~nodes.index.isin(sample.index)]
        add = remaining.iloc[rng.choice(len(remaining), size=pilot_n-len(sample), replace=False)]
        sample = pd.concat([sample, add])
    sample = sample.head(pilot_n).reset_index(drop=True)
    sample.to_parquet(out / "pilot_nodes.parquet", index=False)
    tcr = pd.DataFrame({
        "cdr3_b_aa": sample.aa.astype(str),
        "v_b_gene": sample.v_gene.astype(str).map(lambda x: x if "*" in x else x + "*01"),
        "j_b_gene": sample.j_gene.astype(str).map(lambda x: x if "*" in x else x + "*01"),
        "count": 1,
        "subject": "HomeTrain",
    })
    started = time.monotonic()
    try:
        rep = TCRrep(cell_df=tcr, organism="human", chains=["beta"],
                     compute_distances=True, cpus=min(16, os.cpu_count() or 1))
        exact = np.asarray(rep.pw_beta)
        clone_df = rep.clone_df
        if exact.shape != (pilot_n, pilot_n):
            raise ValueError(f"tcrdist deduplicated/dropped receptors: {exact.shape}")
        tcr_keys = list(zip(tcr.cdr3_b_aa, tcr.v_b_gene, tcr.j_b_gene))
        key_to_seq = dict(zip(tcr_keys, sample.seq_idx.astype(int)))
        ordered_seq = np.array([
            key_to_seq[(r.cdr3_b_aa, r.v_b_gene, r.j_b_gene)]
            for r in clone_df.itertuples(index=False)], dtype=np.int64)
        bank = np.load(root / "embeddings" / model / "bank_fp16.npy", mmap_mode="r")
        vectors = np.asarray(bank[ordered_seq], dtype=np.float32)
        if not np.isfinite(vectors).all():
            raise ValueError("Nonfinite embedding rows in tcrdist pilot")
        ann_started = time.monotonic()
        ann = NNDescent(vectors, n_neighbors=min(candidate_k + 1, pilot_n),
                        metric="euclidean", random_state=seed,
                        n_jobs=min(16, os.cpu_count() or 1))
        indices, _ = ann.query(vectors, k=min(candidate_k + 1, pilot_n))
        candidate_sets = [set(int(j) for j in row if j != i) for i, row in enumerate(indices)]
        exact_neighbors = [set(int(j) for j in np.flatnonzero(exact[i] <= radius) if j != i)
                           for i in range(pilot_n)]
        directed_pairs = sum(len(s) for s in exact_neighbors)
        found = sum(len(s & candidate_sets[i]) for i, s in enumerate(exact_neighbors))
        recall = found / directed_pairs if directed_pairs else None
        giant_components = _component_sizes(exact <= radius)
        giant_fraction = max(giant_components) / pilot_n if giant_components else 0.0
        ungrouped_fraction = sum(not s for s in exact_neighbors) / pilot_n
        pass_recall = directed_pairs >= 50 and recall is not None and recall >= 0.95
        report = {
            "scope": scope, "method": "tcrdist3_beta_exact_pilot",
            "model": model, "pilot_n": pilot_n, "radius": radius,
            "candidate_k": candidate_k, "exact_directed_neighbor_pairs": directed_pairs,
            "candidate_neighbor_recall": recall,
            "minimum_required_recall": 0.95,
            "minimum_directed_pairs": 50,
            "pass_recall": pass_recall,
            "giant_component_fraction": giant_fraction,
            "ungrouped_fraction": ungrouped_fraction,
            "exact_seconds": ann_started - started,
            "ann_seconds": time.monotonic() - ann_started,
            "rss_ru_maxrss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            "gene_allele_01_imputation_fraction": float(
                np.mean(~sample.v_gene.astype(str).str.contains(r"\*") |
                        ~sample.j_gene.astype(str).str.contains(r"\*"))),
            "full_union_nodes": int(len(nodes)),
            "full_coverage_claim": False,
            "pass_full_run": False,
            "reason_full_gate": "20M-scale index/projection resource pilot still required"
                                if pass_recall else "candidate recall/positive-pair gate failed",
        }
    except Exception as exc:
        report = {
            "scope": scope, "method": "tcrdist3_beta_exact_pilot",
            "model": model, "pilot_n": pilot_n, "radius": radius,
            "status": "failed", "error": f"{type(exc).__name__}: {exc}",
            "elapsed_seconds": time.monotonic() - started,
            "rss_ru_maxrss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            "pass_recall": False, "pass_full_run": False,
            "full_coverage_claim": False,
        }
        _json(out / "receipt.json", report)
        raise
    _json(out / "receipt.json", report)
    return out / "receipt.json"


def _component_sizes(adjacency: np.ndarray) -> list[int]:
    from scipy.sparse import csr_matrix
    from scipy.sparse.csgraph import connected_components
    n, labels = connected_components(csr_matrix(adjacency), directed=False)
    return np.bincount(labels, minlength=n).astype(int).tolist()


def project_gliph2_external_frozen(root: Path, frozen_candidates: Path) -> Path:
    """Apply frozen seed-derived literal motif/VJ rules to every external QC clone.

    External recorded N is never read. The Home dictionary remains immutable.
    This is a derived classifier, not official full GLIPH2 clustering.
    """
    import pyarrow as pa
    import pyarrow.compute as pc
    from datetime import datetime, timezone
    out = root / "groups/full_home"
    integrated = _validate_integrated_scope(root, "full_home")
    frozen = _validate_frozen_file(root, frozen_candidates)
    selected = set(frozen.loc[frozen.method.eq("gliph2"), "group_id"].astype(str))
    if not selected:
        raise ValueError("Frozen candidates contain no GLIPH2 family")
    if not selected.issubset(set(integrated.loc[integrated.method.eq("gliph2"), "group_id"])):
        raise ValueError("Frozen GLIPH2 families missing from Home dictionary")
    home_receipt_path = out / "gliph2_projection_receipt.json"
    home_receipt = json.loads(home_receipt_path.read_text())
    meta_path = out / "group_metadata.tsv"
    seed_path = out / "gliph2_seed_membership.parquet"
    if home_receipt.get("status") != "complete_home_scope" or\
            home_receipt.get("projection_population") != "home" or\
            home_receipt.get("group_metadata_sha256") != _sha256(meta_path) or\
            home_receipt.get("seed_membership_sha256") != _sha256(seed_path):
        raise ValueError("Home GLIPH2 seed rule has changed")
    meta = pd.read_csv(meta_path, sep="\t").drop_duplicates("group_id")
    meta = meta.loc[meta.group_id.isin(selected)].copy()
    if set(meta.group_id) != selected:
        raise ValueError("Frozen motif metadata missing group")
    meta["pattern"] = meta.pattern.fillna("").astype(str)
    meta["type"] = meta.type.fillna("").astype(str)
    meta["final_score"] = pd.to_numeric(meta.final_score, errors="coerce")
    meta["number_subject"] = pd.to_numeric(meta.number_subject, errors="coerce")
    valid = (meta.pattern.str.fullmatch(r"[ACDEFGHIKLMNPQRSTVWY]{4,8}") &
             meta.type.eq("motif-" + meta.pattern) &
             meta.final_score.le(0.05) & meta.number_subject.ge(3))
    if not valid.all() or meta.pattern.duplicated().any():
        raise ValueError("Frozen GLIPH2 motifs are not unique qualified literals")
    seed = pd.read_parquet(seed_path).loc[lambda x: x.group_id.isin(selected)]
    if set(seed.group_id) != selected:
        raise ValueError("Frozen GLIPH2 seed members incomplete")
    seed = seed.merge(meta[["group_id", "pattern"]], on="group_id", validate="many_to_one")
    if (~seed.apply(lambda r: r.pattern in r.aa, axis=1)).any():
        raise ValueError("Frozen motif absent from its seed member")
    pairs = {gid: set(zip(part.v_gene.map(_stem), part.j_gene.map(_stem)))
             for gid, part in seed.groupby("group_id")}
    rules = list(meta[["group_id", "pattern"]].itertuples(index=False, name=None))
    manifest = pd.read_csv(root / "config/cohort_samples.tsv", sep="\t", dtype=str,
                           usecols=["cohort", "sample_id", "patient_key"])
    external = manifest.loc[manifest.cohort.eq("external")]
    external_patients = set(external.patient_key.dropna()) - {""}
    if external.empty or len(external) != len(external_patients):
        raise ValueError("External proxy identities are missing or duplicated")
    paths = [root / "data/clone_detail" / f"external__{r.sample_id}.parquet"
             for r in external.itertuples(index=False)]
    if not all(q.is_file() for q in paths):
        raise FileNotFoundError("External clone-detail source incomplete")
    scanner = ds.dataset([str(q) for q in paths], format="parquet").scanner(
        columns=["cohort", "patient_key", "aa", "v_gene", "j_gene"],
        filter=ds.field("qc_pass") == True, batch_size=131072)
    scanned = valid_vj = matched_rows = 0
    parts = []
    started = time.monotonic()
    for batch in scanner.to_batches():
        frame = batch.to_pandas()
        scanned += len(frame)
        if not frame.cohort.eq("external").all() or\
                not set(frame.patient_key).issubset(external_patients):
            raise ValueError("External clone row identity mismatch")
        frame = frame.loc[frame.v_gene.astype(str).str.startswith("TRBV") &
                          frame.j_gene.astype(str).str.startswith("TRBJ")]
        valid_vj += len(frame)
        if frame.empty:
            continue
        aa = pa.array(frame.aa.astype(str))
        vj = pd.MultiIndex.from_arrays([frame.v_gene.map(_stem), frame.j_gene.map(_stem)])
        any_hit = np.zeros(len(frame), dtype=bool)
        for gid, motif in rules:
            mask = pc.match_substring(aa, motif).to_numpy(zero_copy_only=False) & vj.isin(pairs[gid])
            if mask.any():
                any_hit |= mask
                part = frame.loc[mask, KEY].drop_duplicates().copy()
                part.insert(0, "group_id", gid)
                parts.append(part)
        matched_rows += int(any_hit.sum())
    hits = (pd.concat(parts, ignore_index=True).drop_duplicates()
            if parts else pd.DataFrame(columns=["group_id"] + KEY))
    if not hits.empty:
        union = pd.read_parquet(root / "data/aa_union.parquet", columns=["aa", "seq_idx"])
        if union.aa.duplicated().any():
            raise ValueError("Global AA index duplicated")
        hits = hits.merge(union, on="aa", how="left", validate="many_to_one")
        if hits.seq_idx.isna().any():
            raise ValueError("External QC motif hit missing global AA index")
        hits["seq_idx"] = hits.seq_idx.astype(np.int64)
    else:
        hits["seq_idx"] = pd.Series(dtype="int64")
    hits.insert(0, "method", "gliph2")
    hits.insert(0, "scope", "full_home")
    path = out / "gliph2_external_frozen_membership.parquet"
    tmp = path.with_suffix(".parquet.tmp")
    hits[["scope", "method", "group_id"] + KEY + ["seq_idx"]].to_parquet(tmp, index=False)
    tmp.replace(path)
    receipt_path = out / "gliph2_external_frozen_projection_receipt.json"
    _json(receipt_path, {
        "status": "complete_external_frozen_scope", "scope": "full_home", "method": "gliph2",
        "projection_population": "external", "external_label_read": False,
        "projection_type": "gliph2_seed_motif_projection",
        "family_definition": "Derived single literal GLIPH2 seed motif plus observed TRBV/TRBJ stem pair; not official full GLIPH2 clustering",
        "frozen_candidates_sha256": _sha256(frozen_candidates),
        "frozen_candidate_receipt_sha256": _sha256(frozen_candidates.with_suffix(".json")),
        "home_membership_sha256": _sha256(out / "group_membership.parquet"),
        "home_gliph2_projection_receipt_sha256": _sha256(home_receipt_path),
        "train_patient_sha256": _patient_hash(_scope_patients(root, "full_home")),
        "external_identity_sha256": _patient_hash(external_patients),
        "manifest_identity_sha256": _manifest_identity_sha256(root),
        "group_ids_sha256": _patient_hash(selected),
        "membership_sha256": _sha256(path),
        "source_clone_file_count": len(paths),
        "source_clone_stats_sha256": hashlib.sha256(json.dumps([
            (q.name, q.stat().st_size, q.stat().st_mtime_ns) for q in paths]).encode()).hexdigest(),
        "scanned_qc_clone_rows": scanned, "valid_vj_clone_rows": valid_vj,
        "matched_valid_clone_rows": matched_rows,
        "n_exact_aa_vj_memberships": len(hits),
        "n_unique_aa_assigned": int(hits.aa.nunique()),
        "n_frozen_groups": len(selected), "n_groups_with_external_hits": int(hits.group_id.nunique()),
        "elapsed_seconds": time.monotonic() - started,
        "rss_ru_maxrss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "completed_utc": datetime.now(timezone.utc).isoformat(),
    })
    return receipt_path


def _validate_external_frozen_methods(root: Path, frozen: pd.DataFrame,
                                      frozen_candidates: Path) -> tuple[pd.DataFrame, dict]:
    out = root / "groups/full_home"
    home_sha = _sha256(out / "group_membership.parquet")
    manifest = pd.read_csv(root / "config/cohort_samples.tsv", sep="\t", dtype=str,
                           usecols=["cohort", "patient_key"])
    patients = set(manifest.loc[manifest.cohort.eq("external"), "patient_key"].dropna()) - {""}
    if not patients or len(patients) != int(manifest.cohort.eq("external").sum()):
        raise ValueError("External proxy identities are missing or duplicated")
    parts = []
    per_method = {}
    for method, candidate_part in frozen.groupby("method"):
        if method not in {"gliph2", "tcrdist3"}:
            raise ValueError("Unknown frozen method")
        prefix = "gliph2" if method == "gliph2" else "tcrdist"
        source = out / f"{prefix}_external_frozen_membership.parquet"
        receipt_path = out / f"{prefix}_external_frozen_projection_receipt.json"
        if not source.is_file() or not receipt_path.is_file():
            raise ValueError(f"External frozen {method} projection absent")
        receipt = json.loads(receipt_path.read_text())
        candidate_ids = set(candidate_part.group_id.astype(str))
        if (receipt.get("status") != "complete_external_frozen_scope" or
                receipt.get("projection_population") != "external" or
                receipt.get("external_label_read") is not False or
                receipt.get("scope") != "full_home" or receipt.get("method") != method or
                receipt.get("frozen_candidates_sha256") != _sha256(frozen_candidates) or
                receipt.get("frozen_candidate_receipt_sha256") != _sha256(frozen_candidates.with_suffix(".json")) or
                receipt.get("manifest_identity_sha256") != _manifest_identity_sha256(root) or
                receipt.get("home_membership_sha256") != home_sha or
                receipt.get("train_patient_sha256") != _patient_hash(_scope_patients(root, "full_home")) or
                receipt.get("external_identity_sha256") != _patient_hash(patients) or
                receipt.get("group_ids_sha256") != _patient_hash(candidate_ids) or
                receipt.get("membership_sha256") != _sha256(source)):
            raise ValueError(f"External frozen {method} receipt invalid")
        if method == "gliph2":
            identity = pd.read_csv(root / "config/cohort_samples.tsv", sep="\t", dtype=str,
                                   usecols=["cohort", "sample_id"])
            ext_paths = [root / "data/clone_detail" / f"external__{row.sample_id}.parquet"
                         for row in identity.loc[identity.cohort.eq("external")].itertuples(index=False)]
            source_stats = hashlib.sha256(json.dumps([
                (q.name, q.stat().st_size, q.stat().st_mtime_ns) for q in ext_paths]).encode()).hexdigest()
            if (receipt.get("home_gliph2_projection_receipt_sha256") !=
                    _sha256(out / "gliph2_projection_receipt.json") or
                    receipt.get("source_clone_file_count") != len(ext_paths) or
                    receipt.get("source_clone_stats_sha256") != source_stats):
                raise ValueError("Frozen external GLIPH2 source changed")
        part = pd.read_parquet(source)
        required = {"scope", "method", "group_id", "aa", "v_gene", "j_gene", "seq_idx"}
        if (required - set(part) or not part.scope.eq("full_home").all() or
                not part.method.eq(method).all() or
                not set(part.group_id).issubset(candidate_ids) or
                part[["group_id"] + KEY].duplicated().any()):
            raise ValueError(f"External frozen {method} membership invalid")
        parts.append(part[["scope", "method", "group_id"] + KEY + ["seq_idx"]])
        per_method[method] = {
            "membership_sha256": _sha256(source),
            "receipt_sha256": _sha256(receipt_path),
            "group_ids_sha256": _patient_hash(candidate_ids)}
    combined = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(
        columns=["scope", "method", "group_id"] + KEY + ["seq_idx"])
    return combined, per_method


def external_project(root: Path, frozen_candidates: Path) -> Path:
    """Merge complete frozen-rule external method projections before N read."""
    from datetime import datetime, timezone
    out = root / "groups/full_home"
    home_membership = out / "group_membership.parquet"
    _validate_integrated_scope(root, "full_home")
    home_burden, home_receipt = _validate_burden_receipt(root, "full_home", cohort="home")
    frozen = _validate_frozen_file(root, frozen_candidates)
    group_ids = set(frozen.group_id.astype(str))
    if not group_ids or frozen.group_id.duplicated().any() or\
            not group_ids.issubset(set(home_burden.group_id.astype(str))):
        raise ValueError("Frozen candidates not present in verified Home burden")
    manifest = pd.read_csv(root / "config/cohort_samples.tsv", sep="\t", dtype=str,
                           usecols=["cohort", "patient_key"])
    patients = set(manifest.loc[manifest.cohort.eq("external"), "patient_key"].dropna()) - {""}
    if not patients or len(patients) != int(manifest.cohort.eq("external").sum()):
        raise ValueError("External proxy identities are missing or duplicated")
    combined, per_method = _validate_external_frozen_methods(root, frozen, frozen_candidates)
    if combined[["group_id"] + KEY].duplicated().any():
        raise ValueError("External frozen memberships collide")
    external_path = out / "external_frozen_membership.parquet"
    tmp = external_path.with_suffix(".parquet.tmp")
    combined.to_parquet(tmp, index=False)
    tmp.replace(external_path)
    external_receipt_path = out / "external_frozen_membership_receipt.json"
    _json(external_receipt_path, {
        "status": "complete_external_frozen_scope",
        "scope": "full_home", "projection_population": "external",
        "external_label_read": False,
        "home_membership_sha256": _sha256(home_membership),
        "frozen_candidates_sha256": _sha256(frozen_candidates),
        "frozen_candidate_receipt_sha256": _sha256(frozen_candidates.with_suffix(".json")),
        "external_identity_sha256": _patient_hash(patients),
        "group_ids_sha256": _patient_hash(group_ids),
        "methods": per_method,
        "membership_sha256": _sha256(external_path),
        "n_membership_rows": len(combined),
        "n_frozen_groups": len(group_ids),
        "completed_utc": datetime.now(timezone.utc).isoformat(),
    })
    started = datetime.now(timezone.utc).isoformat()
    projection_receipt_path = out / "external_projection_receipt.json"
    _json(projection_receipt_path, {
        "status": "started", "started_utc": started,
        "external_n_read": False, "membership_sha256": _sha256(home_membership),
        "external_frozen_membership_sha256": _sha256(external_path),
        "external_frozen_membership_receipt_sha256": _sha256(external_receipt_path),
        "frozen_candidates_sha256": _sha256(frozen_candidates),
    })
    burden = patient_burden(root, "full_home", patients, cohort="external",
                            group_ids=group_ids,
                            projection_membership_path=external_path)
    _assert_complete_burden_grid(burden, patients, group_ids, "full_home")
    _validate_burden_receipt(root, "full_home", cohort="external")
    burden_path = out / "patient_group_burden_external.parquet"
    _json(projection_receipt_path, {
        "status": "complete", "started_utc": started,
        "completed_utc": datetime.now(timezone.utc).isoformat(),
        "external_n_read": False,
        "projection": "complete frozen-rule AA/V/J projection over all external QC clones",
        "membership_sha256": _sha256(home_membership),
        "external_frozen_membership_sha256": _sha256(external_path),
        "external_frozen_membership_receipt_sha256": _sha256(external_receipt_path),
        "method_extension_receipts": per_method,
        "frozen_candidates_sha256": _sha256(frozen_candidates),
        "frozen_candidate_receipt_sha256": _sha256(frozen_candidates.with_suffix(".json")),
        "home_burden_sha256": home_receipt["patient_group_burden_sha256"],
        "home_burden_receipt_sha256": _sha256(out / "burden_receipt.json"),
        "home_qualified_group_ids_sha256": home_receipt["qualified_group_ids_sha256"],
        "frozen_group_ids_sha256": _patient_hash(group_ids),
        "manifest_identity_sha256": _manifest_identity_sha256(root),
        "home_split_label_sha256": _home_split_label_sha256(root),
        "target_patient_sha256": _patient_hash(patients),
        "target_proxy_count": len(patients),
        "burden_sha256": _sha256(burden_path),
        "external_burden_receipt_sha256": _sha256(out / "burden_receipt_external.json"),
        "burden_rows": len(burden),
        "external_identity": "BioSample_proxy_unverified",
    })
    return projection_receipt_path


def external_score(root: Path, frozen_candidates: Path) -> Path:
    """Read external recorded N only after label-blind projection receipt exists."""
    from datetime import datetime, timezone
    out = root / "groups/full_home"
    receipt_path = out / "external_projection_receipt.json"
    receipt = json.loads(receipt_path.read_text())
    if receipt.get("status") != "complete" or receipt.get("external_n_read") is not False:
        raise ValueError("External projection has not completed without N labels")
    frozen = _validate_frozen_file(root, frozen_candidates)[["group_id"]]
    home_burden, home_burden_receipt = _validate_burden_receipt(root, "full_home", cohort="home")
    manifest_path = root / "config/cohort_samples.tsv"
    manifest_unlabeled = pd.read_csv(manifest_path, sep="\t", dtype=str,
                                     usecols=["cohort", "patient_key"])
    patients = set(manifest_unlabeled.loc[manifest_unlabeled.cohort.eq("external"), "patient_key"].dropna())
    patients.discard("")
    if receipt["frozen_candidates_sha256"] != _sha256(frozen_candidates) or\
            receipt.get("frozen_candidate_receipt_sha256") != _sha256(frozen_candidates.with_suffix(".json")) or\
            receipt.get("membership_sha256") != _sha256(out / "group_membership.parquet") or\
            receipt.get("home_burden_sha256") != home_burden_receipt["patient_group_burden_sha256"] or\
            receipt.get("home_burden_receipt_sha256") != _sha256(out / "burden_receipt.json") or\
            receipt.get("home_qualified_group_ids_sha256") != _patient_hash(set(home_burden.group_id)) or\
            receipt.get("frozen_group_ids_sha256") != _patient_hash(set(frozen.group_id)) or\
            receipt.get("manifest_identity_sha256") != _manifest_identity_sha256(root) or\
            receipt.get("home_split_label_sha256") != _home_split_label_sha256(root) or\
            receipt.get("target_patient_sha256") != _patient_hash(patients):
        raise ValueError("Frozen candidate file changed after projection")
    burden_path = out / "patient_group_burden_external.parquet"
    external_path = out / "external_frozen_membership.parquet"
    external_receipt_path = out / "external_frozen_membership_receipt.json"
    external_receipt = json.loads(external_receipt_path.read_text())
    projected, per_method = _validate_external_frozen_methods(root, _validate_frozen_file(root, frozen_candidates),
                                                               frozen_candidates)
    if (receipt.get("external_frozen_membership_sha256") != _sha256(external_path) or
            receipt.get("external_frozen_membership_receipt_sha256") != _sha256(external_receipt_path) or
            external_receipt.get("status") != "complete_external_frozen_scope" or
            external_receipt.get("external_label_read") is not False or
            external_receipt.get("membership_sha256") != _sha256(external_path) or
            external_receipt.get("home_membership_sha256") != receipt["membership_sha256"] or
            external_receipt.get("frozen_candidates_sha256") != _sha256(frozen_candidates) or
            external_receipt.get("external_identity_sha256") != _patient_hash(patients) or
            external_receipt.get("group_ids_sha256") != _patient_hash(set(frozen.group_id)) or
            external_receipt.get("methods") != per_method or
            len(projected) != len(pd.read_parquet(external_path))):
        raise ValueError("Frozen-rule external projection source changed before scoring")
    if receipt["burden_sha256"] != _sha256(burden_path):
        raise ValueError("External burden changed after projection")
    if receipt.get("external_burden_receipt_sha256") != _sha256(out / "burden_receipt_external.json"):
        raise ValueError("External burden receipt changed after label-blind projection")
    burden, _ = _validate_burden_receipt(root, "full_home", cohort="external")
    _assert_complete_burden_grid(burden, patients, set(frozen.group_id), "full_home")
    manifest = pd.read_csv(manifest_path, sep="\t", dtype=str,
                           usecols=["cohort", "patient_key", "recorded_n_binary", "identity_basis"])
    labels = manifest.loc[manifest.cohort.str.lower().eq("external"),
                          ["patient_key", "recorded_n_binary", "identity_basis"]].drop_duplicates()
    if not labels.identity_basis.eq("BioSample_proxy_unverified").all():
        raise ValueError("External identity basis changed; review required")
    if labels.patient_key.duplicated().any():
        raise ValueError("Conflicting external proxy labels")
    if set(labels.patient_key) != patients:
        raise ValueError("External labels do not match projected proxy identities")
    if not labels.recorded_n_binary.isin(["0", "1"]).all():
        raise ValueError("External recorded N missing or nonbinary")
    frame = burden.merge(labels[["patient_key", "recorded_n_binary"]],
                         on="patient_key", how="inner", validate="many_to_one")
    scored_labels = (frame[["patient_key", "recorded_n_binary"]]
                     .drop_duplicates()
                     .sort_values(["patient_key", "recorded_n_binary"]))
    if (len(scored_labels) != len(patients)
            or len(scored_labels) != receipt.get("target_proxy_count")
            or scored_labels.patient_key.duplicated().any()):
        raise ValueError("Scored external labels do not match projected proxy identities")
    ingest_summary_path = root / "audit/01_ingest_summary.json"
    if ingest_summary_path.exists():
        expected_external = json.loads(ingest_summary_path.read_text()).get("external_samples")
        if expected_external is not None and len(scored_labels) != expected_external:
            raise ValueError("Scored external proxy count differs from Stage1 cohort")
    external_label_sha256 = hashlib.sha256(json.dumps(
        scored_labels.values.tolist(), ensure_ascii=False,
        separators=(",", ":")).encode("utf-8")).hexdigest()
    result = []
    for group_id, part in frame.groupby("group_id", sort=True):
        pos = part.loc[part.recorded_n_binary.eq("1")]
        neg = part.loc[part.recorded_n_binary.eq("0")]
        if pos.empty or neg.empty:
            raise ValueError("External directional scoring requires both recorded N classes")
        delta = float(pos.presence.mean() - neg.presence.mean())
        read_delta = float(pos.read_fraction.median() - neg.read_fraction.median())
        result.append({
            "group_id": group_id, "n_positive_proxy": len(pos), "n_negative_proxy": len(neg),
            "presence_positive": float(pos.presence.mean()),
            "presence_negative": float(neg.presence.mean()),
            "presence_delta": delta,
            "read_fraction_median_delta": read_delta,
            "n_carriers_external": int(part.presence.sum()),
            "evaluable": bool(part.presence.sum() >= 5),
            "direction_positive": bool(part.presence.sum() >= 5 and delta > 0),
        })
    path = out / "external_direction.tsv"
    pd.DataFrame(result, columns=[
        "group_id", "n_positive_proxy", "n_negative_proxy",
        "presence_positive", "presence_negative", "presence_delta",
        "read_fraction_median_delta", "n_carriers_external", "evaluable", "direction_positive"]).to_csv(path, sep="\t", index=False)
    _json(path.with_suffix(".json"), {
        "scored_utc": datetime.now(timezone.utc).isoformat(),
        "projection_completed_utc": receipt["completed_utc"],
        "projection_receipt_sha256": _sha256(receipt_path),
        "frozen_candidates_sha256": receipt["frozen_candidates_sha256"],
        "external_direction_tsv_sha256": _sha256(path),
        "external_label_sha256": external_label_sha256,
        "external_label_canonicalization": "sorted unique [patient_key,recorded_n_binary] pairs; compact UTF-8 JSON",
        "n_external_scored_patients": int(len(scored_labels)),
        "n_positive_proxy": int(scored_labels.recorded_n_binary.eq("1").sum()),
        "n_negative_proxy": int(scored_labels.recorded_n_binary.eq("0").sum()),
        "external_identity": "BioSample_proxy_unverified",
        "inference": "directional transport only; no confirmatory independent-patient p-value",
        "n_frozen_groups": int(len(frozen)),
        "n_scored_groups": int(len(result)),
    })
    return path

def freeze_home_candidates(root: Path, subset: str = "all_recorded_home_exploratory",
                           q_max: float = 0.05, max_candidates: int | None = None) -> Path:
    """Freeze Home-only exact-V/J candidates before external projection."""
    from datetime import datetime, timezone
    if subset != "all_recorded_home_exploratory" or q_max != 0.05 or max_candidates is not None:
        raise ValueError(_private_literal('private_24376e81ba4a238e'))
    _validate_preoutcome_analysis_lock(root)
    scope_dir = root / "groups/full_home"
    _validate_integrated_scope(root, "full_home")
    source = scope_dir / f"association_{subset}.tsv"
    table = _validate_association_receipt(root, "full_home", subset)
    required = {"group_id", "adjusted_presence_q_pooled", "adjusted_presence_or", "presence_delta",
                "adjusted_model_status", "read_fraction_delta_median"}
    if required - set(table):
        raise ValueError("Association lacks frozen candidate fields")
    table = table.loc[table.adjusted_model_status.eq("ok") &
                      table.adjusted_presence_q_pooled.lt(q_max) &
                      table.adjusted_presence_or.gt(1) & table.presence_delta.gt(0)].copy()
    # The 31 documented-blood patients are a sensitivity analysis. Retain
    # its estimate/status without using it as a candidate selection gate.
    sensitivity = scope_dir / "association_verified_blood.tsv"
    if not sensitivity.exists():
        raise ValueError(_private_literal('private_45524c611f0209ae'))
    blood = _validate_association_receipt(root, "full_home", "verified_blood")
    if subset != "verified_blood":
        blood = blood[["group_id", "adjusted_presence_or", "read_fraction_delta_median",
                       "adjusted_model_status"]].rename(columns={
                           "adjusted_presence_or": "blood_adjusted_presence_or",
                           "read_fraction_delta_median": "blood_read_fraction_delta_median",
                           "adjusted_model_status": "blood_model_status"})
        table = table.merge(blood, on="group_id", how="left", validate="one_to_one")
    membership_path = scope_dir / "group_membership.parquet"
    membership = pd.read_parquet(membership_path,
                                 columns=["group_id", "method", "projection_type"]).drop_duplicates()
    table = table.merge(membership, on="group_id", how="inner", validate="one_to_one")
    table = table.sort_values(["adjusted_presence_q_pooled", "group_id"])
    table = table.rename(columns={"adjusted_presence_q_pooled": "home_q_pooled",
                                  "adjusted_presence_or": "home_adjusted_presence_or",
                                  "presence_delta": "home_presence_delta",
                                  "read_fraction_delta_median": "home_read_fraction_delta_median"})
    table["family_definition"] = table.projection_type.map({
        "gliph2_seed_motif_projection":
            "Single literal GLIPH2 seed motif plus seed-observed TRBV/TRBJ pair, projected by derived classifier; not official full GLIPH2 clustering",
        "tcrdist3_train_seed_full_projection":
            "Training-seed tcrdist3 family with exact-distance projection over all reference-scorable V/J nodes"})
    columns = ["group_id", "method", "projection_type", "family_definition", "home_q_pooled", "home_adjusted_presence_or", "home_presence_delta",
               "home_read_fraction_delta_median"]
    if "blood_model_status" in table:
        columns += ["blood_adjusted_presence_or", "blood_read_fraction_delta_median",
                    "blood_model_status"]
    eligible_output = root / "groups/eligible_home_candidates.tsv"
    table[columns].to_csv(eligible_output, sep="\t", index=False)
    output = root / "groups/frozen_home_candidates.tsv"
    output.write_bytes(eligible_output.read_bytes())
    _json(output.with_suffix(".json"), {
        "status": "complete" if len(table) else "STOP_no_home_candidates",
        "frozen_utc": datetime.now(timezone.utc).isoformat(),
        "source_association": str(source), "source_association_sha256": _sha256(source),
        "source_association_receipt_sha256": _sha256(source.with_suffix(".json")),
        "source_eligible_groups_sha256": _sha256(scope_dir / f"eligible_groups_{subset}.tsv"),
        "home_burden_sha256": _sha256(scope_dir / "patient_group_burden.parquet"),
        "home_burden_receipt_sha256": _sha256(scope_dir / "burden_receipt.json"),
        "manifest_identity_sha256": _manifest_identity_sha256(root),
        "home_split_label_sha256": _home_split_label_sha256(root),
        "clinical_covariates_sha256": _sha256(root / "config/home_clinical_covariates.tsv"),
        "documented_blood_sensitivity_sha256": _sha256(sensitivity),
        "documented_blood_sensitivity_receipt_sha256": _sha256(sensitivity.with_suffix(".json")),
        "membership_sha256": _sha256(membership_path),
        "candidate_sha256": _sha256(output), "n_candidates": len(table),
        "eligible_home_candidates_sha256": _sha256(eligible_output),
        "n_eligible_home_candidates": len(pd.read_csv(eligible_output, sep="\t")),
        "q_strictly_less_than": q_max, "max_frozen_candidates": None,
        "candidate_policy": "all_eligible_home_primary_groups",
        "direction_rule": "Home adjusted OR>1 and Home patient-presence delta>0; external patient-presence delta>0 with >=5 carriers",
        "candidate_truncation": False,
        "source_subset": subset, "external_n_read": False,
        "meaning": "Home-only discovery nominations; not final pN+ family claims",
    })
    return output


def _training_member_aa(root: Path, scope: str, membership: pd.DataFrame,
                        group_ids: set[str]) -> pd.DataFrame:
    """Limit alignment evidence to AA/V/J observed in dictionary-fit patients."""
    selected = membership.loc[membership.group_id.isin(group_ids),
                              ["group_id", "method"] + KEY].drop_duplicates()
    train_nodes = pd.read_parquet(root / "groups" / scope / "clone_nodes.parquet", columns=KEY)
    train_nodes = train_nodes.drop_duplicates()
    return selected.merge(train_nodes, on=KEY, how="inner", validate="many_to_many")


def align_fold_nominations(root: Path, min_jaccard: float = 0.50,
                           min_shared_aa: int = 2, min_folds: int = 3) -> Path:
    """Compare fold nominations by exact AA overlap and method, without holdout N."""
    if not 0 < min_jaccard <= 1 or min_shared_aa < 2 or not 2 <= min_folds <= 5:
        raise ValueError("Invalid predeclared alignment thresholds")
    nomination_path = root / "groups/fold_nominations.tsv"
    nomination = pd.read_csv(nomination_path, sep="\t", dtype={"group_id": str})
    if nomination.empty:
        raise ValueError("No fold nominations to align")
    if nomination.group_id.duplicated().any():
        raise ValueError("Duplicate fold nomination group IDs")
    by_group: dict[str, dict] = {}
    membership_hashes = {}
    for fold in range(5):
        scope = f"outer_{fold}"
        subset = nomination.loc[nomination.outer_fold.eq(fold)]
        if subset.empty:
            continue
        if not subset.train_patient_sha256.eq(_patient_hash(_scope_patients(root, scope))).all():
            raise ValueError(f"Fold {fold} nomination train hash mismatch")
        path = root / "groups" / scope / "group_membership.parquet"
        membership_hashes[scope] = _sha256(path)
        membership = _validate_integrated_scope(root, scope)
        membership = _training_member_aa(root, scope, membership, set(subset.group_id))
        motif_path = root / "groups" / scope / "group_metadata.tsv"
        motif_map = {}
        if motif_path.exists():
            motifs = pd.read_csv(motif_path, sep="\t")
            if {"group_id", "pattern", "type"}.issubset(motifs.columns):
                motif_map = dict(zip(motifs.group_id.astype(str),
                                     motifs.pattern.fillna("").astype(str)))
        for group_id, frame in membership.groupby("group_id", sort=True):
            method = set(frame.method)
            if len(method) != 1:
                raise ValueError(f"Mixed family method for {group_id}")
            by_group[group_id] = {"fold": fold, "method": next(iter(method)),
                                  "aa": set(frame.aa.astype(str)),
                                  "motif": motif_map.get(group_id, ""),
                                  "rank": int(subset.loc[subset.group_id.eq(group_id), "rank"].iloc[0])}
    if set(nomination.group_id) != set(by_group):
        raise ValueError("Some nominations lack fold membership")
    ids = sorted(by_group)
    pairs = []
    compatible: dict[tuple[str, str], tuple[int, float]] = {}
    for i, left in enumerate(ids):
        a = by_group[left]
        for right in ids[i+1:]:
            b = by_group[right]
            if a["fold"] == b["fold"]:
                continue
            shared = len(a["aa"] & b["aa"])
            union = len(a["aa"] | b["aa"])
            jaccard = shared / union if union else 0.0
            same_method = a["method"] == b["method"]
            same_motif = bool(a["motif"] and a["motif"] == b["motif"])
            aligned = (same_method and shared >= min_shared_aa and jaccard >= min_jaccard and
                       (a["method"] != "gliph2" or same_motif))
            pairs.append({"group_a": left, "group_b": right,
                          "method_a": a["method"], "method_b": b["method"],
                          "shared_aa": shared, "aa_jaccard": jaccard,
                          "same_method": same_method, "same_motif": same_motif,
                          "aligned": aligned})
            if aligned:
                compatible[(left, right)] = (shared, jaccard)
                compatible[(right, left)] = (shared, jaccard)
    pairs_path = root / "groups/cross_fold_pairs.tsv"
    pd.DataFrame(pairs, columns=["group_a", "group_b", "method_a", "method_b",
                                "shared_aa", "aa_jaccard", "same_method", "same_motif", "aligned"]).to_csv(
                                    pairs_path, sep="\t", index=False)
    # Anchor matching prevents transitive chaining of unrelated families.
    # Require every chosen group to pass against the anchor; at most one per fold.
    unused = set(ids)
    aligned_rows = []
    for anchor in sorted(ids, key=lambda x: (by_group[x]["rank"], x)):
        if anchor not in unused:
            continue
        choices = [anchor]
        for fold in range(5):
            if fold == by_group[anchor]["fold"]:
                continue
            candidates = [x for x in unused if by_group[x]["fold"] == fold and
                          (anchor, x) in compatible]
            if candidates:
                choices.append(max(candidates, key=lambda x: (
                    compatible[(anchor, x)][1], -by_group[x]["rank"], x)))
        if len(choices) < min_folds:
            continue
        canonical = "cf_" + hashlib.sha256("\n".join(sorted(choices)).encode()).hexdigest()[:12]
        for item in choices:
            unused.remove(item)
            shared, jac = (len(by_group[item]["aa"]), 1.0) if item == anchor else compatible[(anchor, item)]
            aligned_rows.append({"canonical_id": canonical, "outer_fold": by_group[item]["fold"],
                                 "group_id": item, "method": by_group[item]["method"],
                                 "anchor_group_id": anchor, "shared_aa_to_anchor": shared,
                                 "aa_jaccard_to_anchor": jac,
                                 "n_aligned_folds": len(choices),
                                 "rank_in_fold": by_group[item]["rank"],
                                 "motif": by_group[item]["motif"]})
    output = root / "groups/cross_fold_alignment.tsv"
    pd.DataFrame(aligned_rows, columns=["canonical_id", "outer_fold", "group_id", "method",
                                        "anchor_group_id", "shared_aa_to_anchor",
                                        "aa_jaccard_to_anchor", "n_aligned_folds",
                                        "rank_in_fold", "motif"]).to_csv(output, sep="\t", index=False)
    _json(output.with_suffix(".json"), {
        "status": "PASS_alignment" if aligned_rows else "STOP_candidate",
        "thresholds": {"min_aa_jaccard": min_jaccard, "min_shared_aa": min_shared_aa,
                       "same_method_required": True, "gliph2_exact_motif_required": True,
                       "min_distinct_outer_folds": min_folds,
                       "anchor_matching": True},
        "outcome_access": "training-only nominations; no outer holdout or external N",
        "n_nominated": len(ids), "n_canonical_families": len(set(r["canonical_id"] for r in aligned_rows)),
        "n_aligned_rows": len(aligned_rows),
        "nomination_sha256": _sha256(nomination_path),
        "membership_sha256_by_scope": membership_hashes,
        "pairs_sha256": _sha256(pairs_path),
        "alignment_sha256": _sha256(output),
    })
    return output

def align_full_home_to_folds(root: Path, min_jaccard: float = 0.50,
                             min_shared_aa: int = 2, min_matching_folds: int = 3) -> Path:
    """Map frozen full-Home families to fold canonical families without N labels."""
    if (min_jaccard, min_shared_aa, min_matching_folds) != (0.50, 2, 3):
        raise ValueError("Frozen Home-to-fold alignment thresholds cannot change")
    frozen_path = root / "groups/frozen_home_candidates.tsv"
    frozen = _validate_frozen_file(root, frozen_path)
    alignment_path = root / "groups/cross_fold_alignment.tsv"
    aligned = pd.read_csv(alignment_path, sep="\t", dtype={"group_id": str})
    if aligned.empty:
        raise ValueError("STOP_candidate: no cross-fold canonical family")
    full = _training_member_aa(root, "full_home",
                               _validate_integrated_scope(root, "full_home"),
                               set(frozen.group_id))
    full_method = full.drop_duplicates("group_id").set_index("group_id").method.to_dict()
    full_sets_by_fold = {}
    fold_sets = {}
    fold_method = {}
    reference_patient_hashes = {}
    reference_patient_counts = {}
    reference_node_hashes = {}
    for scope in sorted(set(f"outer_{int(i)}" for i in aligned.outer_fold)):
        fold = int(scope.removeprefix("outer_"))
        m = _validate_integrated_scope(root, scope)
        train_patients = _scope_patients(root, scope)
        train_hash = _patient_hash(train_patients)
        node_path = root / "groups" / scope / "clone_nodes.parquet"
        node_desc = json.loads((root / "groups" / scope / "description.json").read_text())
        if (node_desc.get("train_patient_sha256") != train_hash or
                set(node_desc.get("patient_ids", [])) != train_patients):
            raise ValueError(f"{scope} alignment reference is not its outer training patients")
        reference_patient_hashes[str(fold)] = train_hash
        reference_patient_counts[str(fold)] = len(train_patients)
        reference_node_hashes[str(fold)] = _sha256(node_path)
        train_keys = pd.read_parquet(node_path, columns=KEY).drop_duplicates()
        full_train = full.merge(train_keys, on=KEY, how="inner", validate="many_to_many")
        full_sets_by_fold[fold] = full_train.groupby("group_id").aa.agg(
            lambda x: set(x.astype(str))).to_dict()
        fold_ids = set(aligned.loc[aligned.outer_fold.eq(fold), "group_id"])
        selected = m.loc[m.group_id.isin(fold_ids),
                         ["group_id", "method"] + KEY].drop_duplicates()
        selected = selected.merge(train_keys, on=KEY, how="inner", validate="many_to_many")
        fold_sets.update(selected.groupby("group_id").aa.agg(lambda x: set(x.astype(str))).to_dict())
        fold_method.update(selected.drop_duplicates("group_id").set_index("group_id").method.to_dict())
    motif_map = {}
    for scope in ["full_home"] + [f"outer_{i}" for i in range(5)]:
        path = root / "groups" / scope / "group_metadata.tsv"
        if path.exists():
            tab = pd.read_csv(path, sep="\t")
            if {"group_id", "pattern"}.issubset(tab):
                motif_map.update(dict(zip(tab.group_id.astype(str), tab.pattern.fillna("").astype(str))))
    rows = []
    member_match_rows = []
    for full_id in frozen.group_id:
        candidates = []
        for canonical, part in aligned.groupby("canonical_id", sort=True):
            method = str(part.method.iloc[0])
            if full_method.get(full_id) != method:
                continue
            comparisons = []
            for member_row in part.itertuples(index=False):
                member = member_row.group_id
                full_aa = full_sets_by_fold[int(member_row.outer_fold)].get(full_id, set())
                fold_aa = fold_sets.get(member, set())
                shared = len(full_aa & fold_aa)
                union = len(full_aa | fold_aa)
                jac = shared / union if union else 0.0
                motif_equal = bool(motif_map.get(full_id) and
                                   motif_map.get(full_id) == motif_map.get(member))
                passes = shared >= min_shared_aa and jac >= min_jaccard and\
                    (method != "gliph2" or motif_equal)
                member_match_rows.append({
                    "full_home_group_id": full_id, "canonical_id": canonical,
                    "outer_fold": int(member_row.outer_fold), "group_id": member,
                    "match_pass": bool(passes), "shared_aa": shared,
                    "jaccard": jac, "motif_match": motif_equal if method == "gliph2" else None,
                })
                comparisons.append((member, shared, jac, motif_equal, passes))
            passing = [c for c in comparisons if c[4]]
            if len(passing) >= min_matching_folds:
                candidates.append((canonical, method, len(passing),
                                   float(np.median([c[2] for c in passing])),
                                   float(min(c[2] for c in passing)),
                                   all(c[3] for c in passing) if method == "gliph2" else None))
        candidates.sort(key=lambda x: (-x[2], -x[3], x[0]))
        ambiguous = len(candidates) > 1 and candidates[0][2] == candidates[1][2] and\
            abs(candidates[0][3] - candidates[1][3]) < 0.05
        if candidates and not ambiguous:
            canonical, method, match_n, median_j, min_j, motif = candidates[0]
            rows.append({"full_home_group_id": full_id, "canonical_id": canonical,
                         "method": method, "n_aligned_folds": int(aligned.loc[
                             aligned.canonical_id.eq(canonical), "outer_fold"].nunique()),
                         "n_matching_folds": match_n,
                         "member_jaccard_median": median_j,
                         "member_jaccard_min": min_j,
                         "motif_evidence_exact": motif,
                         "alignment_pass": True, "reason": "matched"})
        else:
            rows.append({"full_home_group_id": full_id, "canonical_id": "",
                         "method": full_method.get(full_id, ""),
                         "n_aligned_folds": 0, "n_matching_folds": 0,
                         "member_jaccard_median": np.nan,
                         "member_jaccard_min": np.nan,
                         "motif_evidence_exact": None,
                         "alignment_pass": False,
                         "reason": "ambiguous" if ambiguous else "no_reliable_cross_fold_match"})
    matched_canonical = [r["canonical_id"] for r in rows if r["alignment_pass"]]
    collisions = {key for key in matched_canonical if matched_canonical.count(key) > 1}
    for row in rows:
        if row["canonical_id"] in collisions:
            row["alignment_pass"] = False
            row["reason"] = "canonical_collision_multiple_full_home"
    member_output = root / "groups/full_home_fold_member_matches.tsv"
    pd.DataFrame(member_match_rows, columns=["full_home_group_id", "canonical_id",
                                            "outer_fold", "group_id", "match_pass",
                                            "shared_aa", "jaccard", "motif_match"]).to_csv(
                                                member_output, sep="\t", index=False)
    output = root / "groups/full_home_to_fold_alignment.tsv"
    pd.DataFrame(rows, columns=["full_home_group_id", "canonical_id", "method",
                                "n_aligned_folds", "n_matching_folds",
                                "member_jaccard_median", "member_jaccard_min",
                                "motif_evidence_exact", "alignment_pass", "reason"]).to_csv(
                                    output, sep="\t", index=False)
    _json(output.with_suffix(".json"), {
        "status": "PASS_alignment" if rows and all(r["alignment_pass"] for r in rows) else
                  "STOP_candidate",
        "thresholds": {"min_aa_jaccard": min_jaccard,
                       "min_shared_aa": min_shared_aa,
                       "min_matching_folds": min_matching_folds,
                       "gliph2_exact_motif_required": True,
                       "ambiguous_top2_median_jaccard_delta_lt": 0.05},
        "frozen_candidates_sha256": _sha256(frozen_path),
        "source_association_sha256": json.loads(frozen_path.with_suffix(".json").read_text()).get(
            "source_association_sha256"),
        "cross_fold_alignment_sha256": _sha256(alignment_path),
        "full_home_fold_member_matches_sha256": _sha256(member_output),
        "output_sha256": _sha256(output),
        "n_frozen": len(frozen),
        "n_matched": sum(r["alignment_pass"] for r in rows),
        "label_usage": "No outer holdout or external N read during alignment",
        "alignment_membership_source": "Each group intersected with its dictionary-fit patient clone_nodes AA/V/J",
        "matching_reference_population": "outer_fold_training_patient_exact_aa_vj",
        "matching_reference_folds_sha256": _sha256(root / "folds/folds.tsv"),
        "matching_reference_train_patient_sha256_by_outer_fold": reference_patient_hashes,
        "matching_reference_train_patient_count_by_outer_fold": reference_patient_counts,
        "matching_reference_clone_nodes_sha256_by_outer_fold": reference_node_hashes,
    })
    return output

def resource_pilot(root: Path, scope: str, max_clone_rows: int = 1_000_000,
                   max_union_rows: int = 1_000_000) -> Path:
    """Bounded label-blind cost probe before large group preparation."""
    import pyarrow.parquet as pq
    if not 10_000 <= max_clone_rows <= 2_000_000 or not 10_000 <= max_union_rows <= 2_000_000:
        raise ValueError("Pilot rows must be 10k..2m")
    selected = _scope_patients(root, scope)
    clone_dir = root / "data/clone_detail"
    files = sorted(clone_dir.glob("*.parquet"))
    if not files:
        raise FileNotFoundError(clone_dir)
    total_file_rows = sum(pq.ParquetFile(path).metadata.num_rows for path in files)
    manifest = pd.read_csv(root / "config/cohort_samples.tsv", sep="\t", dtype=str,
                           usecols=["cohort", "sample_id", "patient_key"])
    relevant = manifest.loc[manifest.cohort.eq("home") & manifest.patient_key.isin(selected)]
    if set(relevant.patient_key) != selected:
        raise ValueError("Selected Home patients missing from manifest")
    selected_files = [clone_dir / f"home__{sample}.parquet" for sample in relevant.sample_id]
    if not all(path.is_file() for path in selected_files):
        raise FileNotFoundError("Selected Home clone-detail files incomplete")
    selected_source_rows = sum(pq.ParquetFile(path).metadata.num_rows for path in selected_files)
    def current_rss_kib() -> int | None:
        status = Path("/proc/self/status")
        if not status.exists():
            return None
        for line in status.read_text().splitlines():
            if line.startswith("VmRSS:"):
                return int(line.split()[1])
        return None
    before = current_rss_kib()
    started = time.monotonic()
    scanner = ds.dataset([str(path) for path in selected_files], format="parquet").scanner(
        columns=RAW_COLS, filter=(ds.field("cohort") == "home") &
        (ds.field("qc_pass") == True), batch_size=131072)
    scanned = selected_rows = 0
    unique = set()
    parts = []
    for batch in scanner.to_batches():
        if scanned >= max_clone_rows:
            break
        frame = batch.to_pandas().head(max_clone_rows-scanned)
        scanned += len(frame)
        frame = frame.loc[frame.patient_key.astype(str).isin(selected)]
        frame = frame.loc[frame.v_gene.astype(str).str.startswith("TRBV") &
                          frame.j_gene.astype(str).str.startswith("TRBJ")]
        selected_rows += len(frame)
        if not frame.empty:
            unique.update(zip(frame.aa, frame.v_gene, frame.j_gene))
            parts.append(frame.groupby(["patient_key"] + KEY, as_index=False,
                                       sort=False).read_count.sum())
    clone_seconds = time.monotonic() - started
    clone_rss = current_rss_kib()
    clone_peak_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    union_path = root / "data/aa_union.parquet"
    parquet = pq.ParquetFile(union_path)
    union_total = parquet.metadata.num_rows
    mapping = {}
    union_started = time.monotonic()
    for batch in parquet.iter_batches(batch_size=131072, columns=["seq_idx", "aa"]):
        remaining = max_union_rows - len(mapping)
        if remaining <= 0:
            break
        ids = batch.column("seq_idx").to_pylist()[:remaining]
        aas = batch.column("aa").to_pylist()[:remaining]
        mapping.update(zip(aas, ids))
    union_seconds = time.monotonic() - union_started
    union_rss = current_rss_kib()
    union_peak_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    union_increment = max(0, union_rss-clone_rss) if union_rss is not None and clone_rss is not None else None
    path = root / f"pilot/04_{scope}_clone_resource_pilot.json"
    _json(path, {
        "status": "bounded_pilot_complete", "scope": scope,
        "train_patient_sha256": _patient_hash(selected),
        "max_clone_rows": max_clone_rows, "scanned_qc_home_clone_rows": scanned,
        "selected_valid_vj_clone_rows": selected_rows,
        "unique_aa_vj_in_pilot": len(unique),
        "aggregated_batch_rows": sum(len(x) for x in parts),
        "clone_seconds": clone_seconds,
        "clone_rows_per_second": scanned / clone_seconds if clone_seconds else None,
        "clone_rss_current_kib": clone_rss, "clone_rss_peak_kib": clone_peak_rss,
        "clone_source_all_rows_metadata": total_file_rows,
        "clone_source_selected_rows_metadata": selected_source_rows,
        "clone_source_selected_file_count": len(selected_files),
        "clone_source_selected_compressed_bytes": sum(path.stat().st_size for path in selected_files),
        "max_union_rows": max_union_rows, "union_rows_loaded": len(mapping),
        "union_total_rows": union_total, "union_seconds": union_seconds,
        "union_dict_rss_increment_kib": union_increment,
        "union_dict_projected_increment_gib": (union_increment / 1024**2) *
                                             union_total / max(1, len(mapping)) if union_increment is not None else None,
        "rss_current_before_kib": before, "rss_current_after_kib": union_rss,
        "rss_peak_kib": union_peak_rss,
        "interpretation": "Read/parse cost only; not GLIPH2 or tcrdist all-node feasibility. Do not authorize full run from this receipt alone.",
        "labels_read": False,
    })
    return path


def _cmd() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("resource-pilot")
    p.add_argument("--scope", required=True)
    p.add_argument("--max-clone-rows", type=int, default=1_000_000)
    p.add_argument("--max-union-rows", type=int, default=1_000_000)
    p = sub.add_parser("prepare-nodes")
    p.add_argument("--scope", required=True)
    p = sub.add_parser("derive-train-nodes")
    p.add_argument("--scope", required=True)
    p = sub.add_parser("rebase-node-manifest-receipt")
    p.add_argument("--scope", required=True)
    p = sub.add_parser("prepare-gliph2")
    p.add_argument("--scope", required=True)
    p.add_argument("--max-per-patient", type=int, default=800)
    p.add_argument("--simulation-depth", type=int, default=1000)
    p = sub.add_parser("gliph2-resource-input")
    p.add_argument("--max-patients", type=int, default=12)
    p.add_argument("--max-per-patient", type=int, default=100)
    p = sub.add_parser("run-gliph2")
    p.add_argument("--pilot", type=Path, required=True)
    p.add_argument("--binary", type=Path, default=LEGACY_GLIPH_ROOT / "irtools.centos")
    p.add_argument("--max-seconds", type=int, default=3600)
    p = sub.add_parser("import-gliph2")
    p.add_argument("--scope", required=True)
    p.add_argument("--csv", type=Path, required=True)
    p = sub.add_parser("project-gliph2-motifs")
    p.add_argument("--scope", required=True)
    p.add_argument("--max-scan-rows", type=int)
    p = sub.add_parser("integrate-projections")
    p.add_argument("--scope", required=True)
    p = sub.add_parser("burden")
    p.add_argument("--scope", required=True)
    p = sub.add_parser("lock-preoutcome")
    p = sub.add_parser("associate")
    p.add_argument("--scope", required=True)
    p.add_argument("--subset", choices=["verified_blood", "all_recorded_home_exploratory"], required=True)
    p.add_argument("--min-present", type=int)
    p.add_argument("--min-absent", type=int)
    p = sub.add_parser("nominate")
    p.add_argument("--subset", choices=["verified_blood", "all_recorded_home_exploratory"], default="all_recorded_home_exploratory")
    p.add_argument("--top-n", type=int, default=5)
    p = sub.add_parser("freeze-home-candidates")
    p.add_argument("--subset", choices=["verified_blood", "all_recorded_home_exploratory"],
                   default="all_recorded_home_exploratory")
    p.add_argument("--q-max", type=float, default=0.05)
    p.add_argument("--max-candidates", type=int, default=None,
                   help="No full-Home cap is permitted; this legacy flag is rejected if set")
    p = sub.add_parser("align-folds")
    p.add_argument("--min-jaccard", type=float, default=0.50)
    p.add_argument("--min-shared-aa", type=int, default=2)
    p.add_argument("--min-folds", type=int, default=3)
    p = sub.add_parser("align-full-home")
    p = sub.add_parser("project-gliph2-external-frozen")
    p.add_argument("--frozen-candidates", type=Path, required=True)
    p = sub.add_parser("external-project")
    p.add_argument("--frozen-candidates", type=Path, required=True)
    p = sub.add_parser("external-score")
    p.add_argument("--frozen-candidates", type=Path, required=True)
    p = sub.add_parser("tcrdist-pilot")
    p.add_argument("--scope", required=True)
    p.add_argument("--model", choices=["tcrbert_mlm", "sceptr_cdr3_only"], required=True)
    p.add_argument("--pilot-n", type=int, default=800)
    p.add_argument("--radius", type=int, default=50)
    p.add_argument("--candidate-k", type=int, default=128)
    args = parser.parse_args()
    root = args.root.resolve()
    if args.command == "resource-pilot":
        print(resource_pilot(root, args.scope, args.max_clone_rows, args.max_union_rows))
    elif args.command == "prepare-nodes":
        print(prepare_nodes(root, args.scope))
    elif args.command == "derive-train-nodes":
        print(derive_train_nodes_from_full_home(root, args.scope))
    elif args.command == "rebase-node-manifest-receipt":
        print(rebase_node_manifest_receipt(root, args.scope))
    elif args.command == "prepare-gliph2":
        print(prepare_gliph2(root, args.scope, args.max_per_patient, simulation_depth=args.simulation_depth))
    elif args.command == "gliph2-resource-input":
        print(prepare_gliph2_resource_input(root, args.max_patients, args.max_per_patient))
    elif args.command == "run-gliph2":
        pilot = args.pilot.resolve()
        if not pilot.is_relative_to(root / "groups"):
            raise ValueError("GLIPH2 output must remain under target/groups")
        print(json.dumps(run_gliph2(pilot, args.binary, args.max_seconds), indent=2))
    elif args.command == "import-gliph2":
        print(import_gliph2(root, args.scope, args.csv))
    elif args.command == "project-gliph2-motifs":
        try:
            print(project_gliph2_motifs(root, args.scope, args.max_scan_rows))
        except Exception as exc:
            _json(root / "groups" / args.scope / "gliph2_projection_receipt.json", {
                "status": "STOP_projection_failed", "scope": args.scope, "method": "gliph2",
                "reason": f"{type(exc).__name__}: {exc}",
                "full_coverage_claim": False,
            })
            raise
    elif args.command == "integrate-projections":
        print(integrate_projections(root, args.scope))
    elif args.command == "burden":
        table = patient_burden(root, args.scope)
        print(json.dumps({"rows": len(table), "groups": table.group_id.nunique()}))
    elif args.command == "lock-preoutcome":
        print(lock_preoutcome_analysis(root))
    elif args.command == "associate":
        print(associate(root, args.scope, args.subset, args.min_present, args.min_absent))
    elif args.command == "nominate":
        print(nominate(root, args.subset, args.top_n))
    elif args.command == "freeze-home-candidates":
        print(freeze_home_candidates(root, args.subset, args.q_max, args.max_candidates))
    elif args.command == "align-folds":
        print(align_fold_nominations(root, args.min_jaccard, args.min_shared_aa, args.min_folds))
    elif args.command == "align-full-home":
        print(align_full_home_to_folds(root))
    elif args.command == "project-gliph2-external-frozen":
        print(project_gliph2_external_frozen(root, args.frozen_candidates.resolve()))
    elif args.command == "external-project":
        print(external_project(root, args.frozen_candidates))
    elif args.command == "external-score":
        print(external_score(root, args.frozen_candidates))
    elif args.command == "tcrdist-pilot":
        print(tcrdist_pilot(root, args.scope, args.model, args.pilot_n,
                            args.radius, args.candidate_k))


if __name__ == "__main__":
    _cmd()
