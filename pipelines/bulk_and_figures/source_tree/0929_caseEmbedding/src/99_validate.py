#!/usr/bin/env python3
"""Independent, bounded-memory audit of the recorded-N TCR analysis artifacts.

Default validation reads manifests, receipts, Parquet footers, and NumPy headers.
--deep streams the sequence and vector payloads. Raw source rehashing is separate
because the 558 immutable MiXCR inputs are large. Every receipt states its depth;
a quick PASS must not be interpreted as full content validation.
"""
from __future__ import annotations
from release_runtime import private_literal as _private_literal, release_path as _release_path

import argparse
import csv
import hashlib
import io
import json
import os
import re
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

DEFAULT_ROOT = Path(_release_path('path_1304218e86d2e596'))
HEX64 = re.compile(r"^[0-9a-f]{64}$")
MODELS = {"tcrbert_dualtask_aa_mean": 768, "sceptr_cdr3_only": 64, "tcrbert_mlm": 768}
STANDARD_AA = frozenset("ACDEFGHIKLMNPQRSTVWY")


class AuditError(RuntimeError):
    pass


def need(condition: bool, message: str) -> None:
    if not condition:
        raise AuditError(message)


def read_json(path: Path) -> dict:
    need(path.is_file(), f"Missing {path}")
    return json.loads(path.read_text())


def read_tsv(path: Path) -> list[dict]:
    need(path.is_file(), f"Missing {path}")
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream, delimiter="\t"))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def valid_digest(value: object) -> bool:
    return isinstance(value, str) and HEX64.fullmatch(value) is not None


def file_stat_matches(path: Path, recorded: dict) -> bool:
    stat = path.stat()
    return stat.st_size == int(recorded["bytes"]) and stat.st_mtime_ns == int(recorded["mtime_ns"])


def parquet_rows_and_columns(path: Path) -> tuple[int, set[str]]:
    need(path.is_file(), f"Missing {path}")
    parquet = pq.ParquetFile(path)
    return parquet.metadata.num_rows, set(parquet.schema_arrow.names)


def patient_hash(keys: set[str]) -> str:
    return hashlib.sha256(("\n".join(sorted(keys)) + "\n").encode()).hexdigest()


def manifest_scope_hashes(rows: list[dict]) -> tuple[str, str]:
    """Recompute fold source hashes without reading external outcome values."""
    identities = sorted((row["cohort"].lower(), row["sample_id"], row["patient_key"])
                        for row in rows)
    stream = io.StringIO()
    writer = csv.writer(stream, delimiter="\t", lineterminator="\n")
    writer.writerow(("cohort", "sample_id", "patient_key"))
    writer.writerows(identities)
    identity_hash = hashlib.sha256(stream.getvalue().encode("utf-8")).hexdigest()
    home_labels = sorted((row["patient_key"], row["recorded_n_binary"])
                         for row in rows if row["cohort"].lower() == "home")
    home_hash = hashlib.sha256(
        json.dumps(home_labels, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return identity_hash, home_hash


def fold_sets(root: Path) -> tuple[dict[int, dict[str, dict]], set[str]]:
    manifest = read_tsv(root / "config/cohort_samples.tsv")
    home = {row["patient_key"]: row["recorded_n_binary"] for row in manifest if row["cohort"] == "home"}
    external = {row["patient_key"] for row in manifest if row["cohort"] == "external"}
    need(len(home) == sum(row["cohort"] == "home" for row in manifest), "Home patient key is not unique")
    rows = read_tsv(root / "folds/folds.tsv")
    result: dict[int, dict[str, dict]] = defaultdict(dict)
    for row in rows:
        need(row["patient_key"] in home and row["patient_key"] not in external,
             "External or unknown identity occurs in Home folds")
        need(row["recorded_n_binary"] == home[row["patient_key"]], "Fold label differs from manifest")
        fold = int(row["outer_fold"])
        need(row["patient_key"] not in result[fold], f"Repeated patient in outer fold {fold}")
        result[fold][row["patient_key"]] = row
    return dict(result), set(home)


def validate_stage1(root: Path, deep: bool = False, rehash_raw: bool = False,
                    rehash_outputs: bool = False,
                    expected_home: int = 114, expected_external: int = 165) -> dict:
    started = time.time()
    need(not rehash_raw or deep, "--rehash-raw requires --deep")
    need(not rehash_outputs or deep, "--rehash-outputs requires --deep")
    summary = read_json(root / "audit/01_ingest_summary.json")
    need(summary["status"] == "PASS", "Stage1 producer receipt is not PASS")
    manifest_path = root / "config/cohort_samples.tsv"
    input_path = root / "audit/01_input_sha256.tsv"
    need(sha256(manifest_path) == summary["cohort_samples_sha256"], "Cohort manifest SHA drift")
    need(sha256(input_path) == summary["input_sha256_manifest_sha256"], "Source hash manifest SHA drift")
    manifest = read_tsv(manifest_path)
    keys = [(row["cohort"], row["sample_id"]) for row in manifest]
    need(len(keys) == len(set(keys)), "Duplicate cohort/sample key")
    need(Counter(row["cohort"] for row in manifest) ==
         {"home": expected_home, "external": expected_external}, "Cohort sizes differ")
    home_labels = Counter(row["recorded_n_binary"] for row in manifest if row["cohort"] == "home")
    external_labels = Counter(row["recorded_n_binary"] for row in manifest if row["cohort"] == "external")
    if expected_home == 114:
        need(home_labels == {"0": 66, "1": 48}, "Home recorded-N class counts differ")
    if expected_external == 165:
        need(external_labels == {"0": 110, "1": 55}, "External recorded-N class counts differ")
    need(len({row["patient_key"] for row in manifest}) == len(manifest),
         "Repeated patient/proxy identity in source manifest")
    source_rows = read_tsv(input_path)
    expected_source_keys = {(c, s, typ) for c, s in keys for typ in ("clone_tsv", "clns")}
    observed_source_keys = {(r["cohort"], r["sample_id"], r["artifact_type"]) for r in source_rows}
    need(len(source_rows) == 2 * len(keys) and observed_source_keys == expected_source_keys,
         "Source manifest lacks exact TSV/CLNS pair for each sample")
    source_by_key = {(r["cohort"], r["sample_id"], r["artifact_type"]): r for r in source_rows}
    totals = Counter()
    raw_hashes_checked = 0
    output_hashes_checked = 0
    for row in manifest:
        cohort, sample = row["cohort"], row["sample_id"]
        receipt = read_json(root / "audit/sample_receipts" / f"{cohort}__{sample}.json")
        need(receipt.get("status") == "PASS", f"{cohort}/{sample}: receipt not PASS")
        for field in ("cohort", "sample_id", "patient_key", "recorded_n_binary"):
            expected = int(row[field]) if field == "recorded_n_binary" else row[field]
            need(receipt[field] == expected, f"{cohort}/{sample}: {field} mismatch")
        need(receipt.get("six_fields_equal_by_clone_id") is True,
             f"{cohort}/{sample}: MiXCR cloneId check did not pass")
        for typ, prefix, manifest_field in (
            ("clone_tsv", "source_tsv", "clone_tsv_path"),
            ("clns", "source_clns", "clns_path"),
        ):
            record = source_by_key[(cohort, sample, typ)]
            source = Path(row[manifest_field])
            need(source.is_file() and source == Path(record["source_path"]) ==
                 Path(receipt[prefix + "_path"]), f"{cohort}/{sample}: source path mismatch")
            need(valid_digest(record["sha256"]) and
                 record["sha256"] == receipt[prefix + "_sha256"],
                 f"{cohort}/{sample}: source SHA receipt mismatch")
            need(int(record["bytes"]) == int(receipt[prefix + "_stat"]["bytes"]) and
                 int(record["mtime_ns"]) == int(receipt[prefix + "_stat"]["mtime_ns"]),
                 f"{cohort}/{sample}: source stat receipt mismatch")
            need(str(source.resolve()) == record["resolved_path"],
                 f"{cohort}/{sample}: source symlink target changed")
            need(file_stat_matches(source, receipt[prefix + "_stat"]),
                 f"{cohort}/{sample}: source file changed after receipt")
            if rehash_raw:
                need(sha256(source) == record["sha256"], f"{cohort}/{sample}: source content SHA differs")
                raw_hashes_checked += 1
        clone_path = root / "data/clone_detail" / f"{cohort}__{sample}.parquet"
        aa_path = root / "data/patient_aa" / f"{cohort}__{sample}.parquet"
        need(clone_path == Path(receipt["clone_detail_path"]) and
             aa_path == Path(receipt["patient_aa_path"]), f"{cohort}/{sample}: derived path mismatch")
        clone_rows, clone_columns = parquet_rows_and_columns(clone_path)
        aa_rows, aa_columns = parquet_rows_and_columns(aa_path)
        need(clone_rows == int(receipt["source_clone_rows"]) and
             aa_rows == int(receipt["unique_aa"]), f"{cohort}/{sample}: Parquet row count mismatch")
        need({"clone_id", "aa", "read_count", "productive", "qc_pass", "v_gene", "j_gene"} <= clone_columns,
             f"{cohort}/{sample}: clone detail columns missing")
        need({"cohort", "sample_id", "patient_key", "aa", "read_count"} <= aa_columns,
             f"{cohort}/{sample}: patient-AA columns missing")
        need(int(receipt["source_clone_rows"]) ==
             int(receipt["productive_clone_rows"]) + int(receipt["nonproductive_clone_rows"]),
             f"{cohort}/{sample}: clone productivity count not conserved")
        need(0 < int(receipt["qc_clone_rows"]) <= int(receipt["productive_clone_rows"]) and
             0 < int(receipt["unique_aa"]) <= int(receipt["qc_clone_rows"]),
             f"{cohort}/{sample}: QC clone/AA count impossible")
        need(int(receipt["unique_aa"]) == int(row["expected_unique_aa"]) and
             int(receipt["qc_read_count"]) == int(row["expected_read_count"]),
             f"{cohort}/{sample}: QC count differs from locked ledger")
        for field in ("source_clone_rows", "qc_clone_rows", "unique_aa", "qc_read_count"):
            totals[field] += int(receipt[field])
        if rehash_outputs:
            need(sha256(clone_path) == receipt["clone_detail_sha256"] and
                 sha256(aa_path) == receipt["patient_aa_sha256"],
                 f"{cohort}/{sample}: derived file SHA mismatch")
            output_hashes_checked += 2
        if deep:
            import pandas as pd
            aa = pd.read_parquet(aa_path)
            need(len(aa) == aa_rows and aa["aa"].is_unique and aa["aa"].is_monotonic_increasing,
                 f"{cohort}/{sample}: patient AA not sorted unique")
            need(aa["cohort"].eq(cohort).all() and aa["sample_id"].eq(sample).all() and
                 aa["patient_key"].eq(row["patient_key"]).all(),
                 f"{cohort}/{sample}: patient AA identity mismatch")
            need(aa["read_count"].gt(0).all() and int(aa["read_count"].sum()) ==
                 int(receipt["qc_read_count"]), f"{cohort}/{sample}: AA read mass mismatch")
            detail = pq.ParquetFile(clone_path)
            qc_rows = 0
            qc_reads = 0
            for batch in detail.iter_batches(batch_size=200_000,
                                             columns=["qc_pass", "read_count"]):
                flags = batch.column("qc_pass").to_numpy(zero_copy_only=False).astype(bool)
                counts = batch.column("read_count").to_numpy(zero_copy_only=False)
                qc_rows += int(flags.sum())
                qc_reads += int(counts[flags].sum())
            need(qc_rows == int(receipt["qc_clone_rows"]) and
                 qc_reads == int(receipt["qc_read_count"]),
                 f"{cohort}/{sample}: clone QC count/read mass mismatch")
    need(totals["source_clone_rows"] >= totals["qc_clone_rows"] and
         totals["qc_clone_rows"] >= totals["unique_aa"], "Aggregate clone count impossible")
    need(totals["unique_aa"] == int(summary["sum_per_sample_unique_aa"]) and
         totals["qc_clone_rows"] == int(summary["total_qc_clones"]) and
         totals["qc_read_count"] == int(summary["total_qc_assigned_reads"]),
         "Stage1 aggregate receipts differ from summary")
    union_path = root / "data/aa_union.parquet"
    union_rows, union_columns = parquet_rows_and_columns(union_path)
    need(union_columns == {"seq_idx", "aa"} and
         union_rows == int(summary["union_unique_aa"]), "AA union schema/row count mismatch")
    union_content_sha_checked = False
    if deep:
        need(sha256(union_path) == summary["union_sha256"], "AA union content SHA differs")
        union_content_sha_checked = True
        expected_idx = 0
        previous_aa = ""
        for batch in pq.ParquetFile(union_path).iter_batches(batch_size=200_000,
                                                              columns=["seq_idx", "aa"]):
            idx = batch.column("seq_idx").to_numpy(zero_copy_only=False)
            need(idx.dtype == np.int64 and
                 np.array_equal(idx, np.arange(expected_idx, expected_idx + len(idx), dtype=np.int64)),
                 f"Union seq_idx noncontiguous at {expected_idx}")
            strings = batch.column("aa").to_pylist()
            need(all(isinstance(s, str) and s and set(s) <= STANDARD_AA for s in strings),
                 f"Noncanonical or empty AA near {expected_idx}")
            need(previous_aa < strings[0] and
                 all(a < b for a, b in zip(strings, strings[1:])),
                 f"Union AA not strictly sorted near {expected_idx}")
            previous_aa = strings[-1]
            expected_idx += len(strings)
        need(expected_idx == union_rows, "Union scan row count mismatch")
    return {
        "status": "PASS", "stage": "stage1", "mode": "deep" if deep else "quick",
        "samples": len(keys), "source_manifest_rows": len(source_rows),
        "source_stat_current": True,
        "source_hash_receipt_agreement": True,
        "source_content_hashes_recomputed": raw_hashes_checked,
        "output_content_hashes_recomputed": output_hashes_checked,
        "union_content_sha_and_order_checked": union_content_sha_checked,
        "union_unique_aa": union_rows,
        "total_qc_clones": totals["qc_clone_rows"],
        "total_qc_reads": totals["qc_read_count"],
        "scope_note": "Quick checks use existing content SHA receipts plus current file stats; they do not rehash raw or derived payloads.",
        "elapsed_s": round(time.time() - started, 2),
    }


def validate_folds(root: Path, expected_outer: int = 5, expected_inner: int = 3) -> dict:
    started = time.time()
    fold_path = root / "folds/folds.tsv"
    if not fold_path.exists():
        return {"status": "PENDING", "stage": "folds", "reason": "folds.tsv absent"}
    manifest = read_tsv(root / "config/cohort_samples.tsv")
    home = {row["patient_key"]: row["recorded_n_binary"] for row in manifest if row["cohort"] == "home"}
    external = {row["patient_key"] for row in manifest if row["cohort"] == "external"}
    folds, keys = fold_sets(root)
    need(keys == set(home), "Fold and manifest Home identities differ")
    need(set(folds) == set(range(expected_outer)), "Outer fold IDs are not contiguous 0..K-1")
    all_test = []
    for outer, rows in sorted(folds.items()):
        need(set(rows) == keys, f"Outer {outer} omits Home identities")
        train = {key for key, row in rows.items() if row["partition"] == "train"}
        test = {key for key, row in rows.items() if row["partition"] == "test"}
        need(train and test and train.isdisjoint(test) and train | test == keys,
             f"Outer {outer}: train/test invalid")
        need(not (set(rows) & external), f"Outer {outer}: external leakage")
        need(all(rows[key]["inner_fold"] == "" for key in test),
             f"Outer {outer}: test patient has inner assignment")
        inner = {key: int(rows[key]["inner_fold"]) for key in train}
        need(set(inner.values()) == set(range(expected_inner)),
             f"Outer {outer}: inner folds missing or invalid")
        for inner_fold in range(expected_inner):
            validation = {key for key, value in inner.items() if value == inner_fold}
            fit = train - validation
            need(validation and fit and validation.isdisjoint(fit),
                 f"Outer {outer}/inner {inner_fold}: invalid split")
        need({home[key] for key in test} == {"0", "1"},
             f"Outer {outer}: test split lacks a class")
        all_test.extend(test)
    need(len(all_test) == len(keys) and set(all_test) == keys,
         "Outer held-out patients are not exactly one per patient")
    meta_path = root / "folds/folds.json"
    if meta_path.is_file():
        meta = read_json(meta_path)
        identity_hash, home_hash = manifest_scope_hashes(manifest)
        source = read_json(root / "audit/01_ingest_summary.json")
        need(meta.get("manifest_identity_sha256") == identity_hash and
             meta.get("home_split_label_sha256") == home_hash and
             meta.get("folds_sha256") == sha256(fold_path) and
             meta.get("stage1_source_contract_sha256") == source.get("source_contract_sha256") and
             meta.get("stage1_input_sha256_manifest_sha256") == source.get("input_sha256_manifest_sha256") and
             meta.get("stage1_union_sha256") == source.get("union_sha256") and
             meta.get("external_in_folds") is False,
             "Fold receipt refers to different source identities, labels, or rows")
        need(meta["n_home_patients"] == len(keys) and
             meta["rows"] == len(keys) * expected_outer and
             meta["outer_folds"] == expected_outer and
             meta["inner_folds"] == expected_inner,
             "Fold receipt counts differ")
    return {"status": "PASS", "stage": "folds", "home_patients": len(keys),
            "outer_folds": expected_outer, "inner_folds": expected_inner,
            "external_in_folds": False, "each_patient_once_heldout": True,
            "elapsed_s": round(time.time() - started, 2)}


def validate_embedding(root: Path, model: str, deep: bool = False) -> dict:
    started = time.time()
    need(model in MODELS, f"Unknown model {model}")
    out = root / "embeddings" / model
    meta_path = out / "meta.json"
    if not meta_path.is_file():
        return {"status": "PENDING", "stage": "embedding", "model": model,
                "reason": "No full-bank metadata; pilot alone is insufficient"}
    meta = read_json(meta_path)
    need(meta["input_sha256"] == read_json(root / "audit/01_ingest_summary.json")["union_sha256"],
         f"{model}: input union SHA mismatch")
    n, dim = int(meta["n_total"]), MODELS[model]
    need(meta["model_spec"]["hidden_dim"] == dim and
         meta["bank_shape"] == [n, dim] and meta["bank_dtype"] == "float16",
         f"{model}: embedding schema mismatch")
    need(valid_digest(meta["model_fingerprint"]), f"{model}: invalid model fingerprint")
    shard_rows = int(meta["shard_rows"])
    need(50_000 <= shard_rows <= 100_000, f"{model}: shard size outside locked range")
    receipts = []
    for receipt_path in sorted((out / "shards").glob("*.json")):
        rec = read_json(receipt_path)
        start, end = int(rec["start"]), int(rec["end"])
        name = f"{start:012d}_{end:012d}"
        need(receipt_path.stem == name and 0 <= start < end <= n,
             f"{model}: bad shard name/range {receipt_path.name}")
        need(rec["rows"] == end - start and rec["dimension"] == dim and
             rec["dtype"] == "float16" and rec["nonfinite"] == 0 and
             rec["input_sha256"] == meta["input_sha256"] and
             rec["model_fingerprint"] == meta["model_fingerprint"] and
             valid_digest(rec["shard_sha256"]),
             f"{model}: bad shard receipt {name}")
        shard_path = receipt_path.with_suffix(".npy")
        need(shard_path.is_file(), f"{model}: missing shard {name}")
        vector = np.load(shard_path, mmap_mode="r", allow_pickle=False)
        need(vector.shape == (end - start, dim) and vector.dtype == np.float16,
             f"{model}: shard shape/dtype mismatch {name}")
        receipts.append((start, end, shard_path, rec))
    receipts.sort()
    coverage = 0
    for start, end, _, _ in receipts:
        need(start == coverage, f"{model}: shard gap/overlap at {coverage}:{start}")
        coverage = end
    if meta.get("status") == "complete":
        need(coverage == n, f"{model}: metadata complete but shards cover only {coverage}/{n}")
        need(int(meta.get("n_done", -1)) == n and valid_digest(meta.get("bank_sha256")),
             f"{model}: completion receipt incomplete")
        bank_path = out / "bank_fp16.npy"
        need(bank_path.is_file(), f"{model}: bank absent")
        bank = np.load(bank_path, mmap_mode="r", allow_pickle=False)
        need(bank.shape == (n, dim) and bank.dtype == np.float16,
             f"{model}: bank shape/dtype mismatch")
        stat = bank_path.stat()
        need(stat.st_size == meta["bank_size_bytes"] and stat.st_mtime_ns == meta["bank_mtime_ns"],
             f"{model}: bank changed after completion")
        if deep:
            for start, end, shard_path, rec in receipts:
                need(sha256(shard_path) == rec["shard_sha256"],
                     f"{model}: shard SHA mismatch at {start}:{end}")
                shard = np.load(shard_path, mmap_mode="r", allow_pickle=False)
                for offset in range(0, end-start, 16_384):
                    part = np.asarray(shard[offset:offset+16_384])
                    need(np.isfinite(part).all(), f"{model}: nonfinite shard at {start+offset}")
                    need(np.array_equal(part, bank[start+offset:start+offset+len(part)]),
                         f"{model}: bank/shard difference at {start+offset}")
            need(sha256(bank_path) == meta["bank_sha256"], f"{model}: bank SHA mismatch")
        return {"status": "PASS", "stage": "embedding", "model": model,
                "mode": "deep" if deep else "quick", "rows": n, "dimension": dim,
                "shards": len(receipts), "coverage": coverage,
                "content_hash_and_finite_checked": deep,
                "elapsed_s": round(time.time()-started, 2)}
    need(meta.get("status") in {"running", "shards_complete"},
         f"{model}: unexpected metadata status")
    return {"status": "PENDING", "stage": "embedding", "model": model,
            "metadata_status": meta["status"], "shards": len(receipts),
            "coverage": coverage, "rows": n,
            "elapsed_s": round(time.time()-started, 2)}


def validate_completed_group_projection(root: Path, scope: str, deep: bool = False) -> dict:
    """Only completed Home-scope projections may enter integrated membership."""
    folder = root / "groups" / scope
    combined = folder / "group_membership.parquet"
    if not combined.exists():
        return {"status": "PENDING", "scope": scope, "reason": "No merged group membership"}
    merged = read_json(folder / "integrated_projection_receipt.json")
    all_methods = {"gliph2", "tcrdist3"}
    methods_list = merged.get("methods", [])
    need(isinstance(methods_list, list) and len(methods_list) == len(set(methods_list)),
         f"{scope}: duplicated or malformed method list")
    completed = set(methods_list)
    absent = all_methods - completed
    need(merged.get("status") == "complete_home_scope" and
         merged.get("projection_population") == "home" and
         merged.get("scope") == scope and
         completed and completed <= all_methods,
         f"{scope}: merged membership lacks a completed Home-scope method")
    need(set(merged.get("coverage_gap_methods", [])) == absent,
         f"{scope}: incomplete method coverage not disclosed")
    statuses = merged.get("method_statuses")
    need(isinstance(statuses, dict) and set(statuses) == all_methods,
         f"{scope}: per-method status map incomplete")
    need(valid_digest(merged.get("membership_sha256")),
         f"{scope}: merged membership SHA receipt missing")
    folds, home = fold_sets(root)
    expected = home if scope == "full_home" else {
        key for key, row in folds[int(scope.split("_")[1])].items()
        if row["partition"] == "train"
    }
    need(merged.get("train_patient_sha256") == patient_hash(expected),
         f"{scope}: merged group training identities differ")
    method_files = {
        "gliph2": ("gliph2_projection_receipt.json", "gliph2_group_membership.parquet"),
        "tcrdist3": ("tcrdist_home_projection_receipt.json", "tcrdist_home_group_membership.parquet"),
    }
    method_rows = {}
    gaps = {}
    for method, (receipt_name, parquet_name) in method_files.items():
        detail = read_json(folder / receipt_name)
        status = detail.get("status")
        need(detail.get("method") == method and detail.get("scope") == scope and
             statuses[method] == status,
             f"{scope}/{method}: per-method status/scope disagrees with integrated receipt")
        if method in absent:
            need(isinstance(status, str) and
                 (status.startswith("STOP") or status.lower() in {"failed", "resource_failed", "inconclusive", "infeasible"}) and
                 isinstance(detail.get("reason"), str) and detail["reason"].strip(),
                 f"{scope}/{method}: absent method lacks explicit STOP/failure reason")
            gaps[method] = {"status": status, "reason": detail["reason"]}
            continue
        need(status == "complete_home_scope" and detail.get("projection_population") == "home",
             f"{scope}/{method}: non-Home or pilot-only membership entered integrated table")
        need(valid_digest(detail.get("membership_sha256")),
             f"{scope}/{method}: missing membership SHA")
        path = folder / parquet_name
        rows, columns = parquet_rows_and_columns(path)
        need({"group_id", "aa"} <= columns and rows > 0,
             f"{scope}/{method}: empty or malformed group membership")
        if deep:
            need(sha256(path) == detail["membership_sha256"],
                 f"{scope}/{method}: membership SHA differs")
        method_rows[method] = rows
    merged_rows, merged_columns = parquet_rows_and_columns(combined)
    need({"method", "group_id", "aa"} <= merged_columns and merged_rows > 0,
         f"{scope}: malformed merged membership")
    if deep:
        need(sha256(combined) == merged["membership_sha256"],
             f"{scope}: merged membership SHA differs")
        import pyarrow.compute as pc
        methods_observed = set()
        for batch in pq.ParquetFile(combined).iter_batches(batch_size=200_000,
                                                            columns=["method"]):
            methods_observed.update(pc.unique(batch.column("method")).to_pylist())
        need(methods_observed == completed,
             f"{scope}: merged rows contain pilot-only or missing method")
    return {"status": "PASS", "scope": scope,
            "merged_rows": merged_rows, "completed_methods": sorted(completed),
            "coverage_gaps": gaps, "method_rows": method_rows,
            "content_sha_checked": deep}


def validate_pca_and_groups(root: Path, deep: bool = False) -> dict:
    started = time.time()
    pca_root = root / "pca"
    group_root = root / "groups"
    pca_meta = sorted(pca_root.glob("*/**/metadata.json")) if pca_root.exists() else []
    group_meta = sorted(group_root.glob("outer_*/description.json")) if group_root.exists() else []
    projected_memberships = (
        sorted(group_root.glob("outer_*/group_membership.parquet")) +
        [group_root / "full_home/group_membership.parquet"]
        if group_root.exists() else []
    )
    projected_memberships = [p for p in projected_memberships if p.is_file()]
    if not pca_meta and not group_meta and not projected_memberships:
        return {"status": "PENDING", "stage": "pca_groups",
                "reason": "No PCA fit or group dictionary metadata"}
    folds, _ = fold_sets(root)
    folds_sha = sha256(root / "folds/folds.tsv")
    union_sha = read_json(root / "audit/01_ingest_summary.json")["union_sha256"]
    pca_checked = 0
    group_checked = 0
    projections_checked = 0
    for meta_path in pca_meta:
        meta = read_json(meta_path)
        outer = int(meta["outer_fold"])
        inner = meta["inner_fold"]
        rows = folds[outer]
        expected = {key for key, row in rows.items() if row["partition"] == "train"}
        if inner is not None:
            expected = {key for key in expected if int(rows[key]["inner_fold"]) != int(inner)}
        need(set(meta["fit_patient_ids"]) == expected and
             meta["fit_patient_count"] == len(expected),
             f"{meta_path}: PCA fit identity leakage")
        need(meta["folds_sha256"] == folds_sha and
             meta["bank_input_sha256"] == union_sha,
             f"{meta_path}: PCA input/fold SHA mismatch")
        k = int(meta["k_per_patient"])
        need(k in (2048, 4096) and int(meta["draw_rows"]) == k * len(expected),
             f"{meta_path}: unbalanced PCA draw count")
        draws_path = meta_path.parent / "draws.parquet"
        draw_rows, draw_cols = parquet_rows_and_columns(draws_path)
        need(draw_rows == int(meta["draw_rows"]) and {"patient_key", "seq_idx"} <= draw_cols,
             f"{meta_path}: PCA draws metadata mismatch")
        if deep:
            import pandas as pd
            draws = pd.read_parquet(draws_path, columns=["patient_key", "seq_idx"])
            need(set(draws.patient_key) == expected and
                 draws.groupby("patient_key").size().eq(k).all(),
                 f"{meta_path}: PCA draw identity/balance mismatch")
        pca_checked += 1
    for path in group_meta:
        meta = read_json(path)
        scope = path.parent.name
        outer = int(scope.split("_")[1])
        expected = {key for key, row in folds[outer].items() if row["partition"] == "train"}
        need(meta["scope"] == scope and set(meta["patient_ids"]) == expected and
             meta["train_patient_sha256"] == patient_hash(expected) and
             meta["role"] == "outer_train_dictionary_input" and
             meta.get("phenotype_blind") is True,
             f"{path}: group dictionary patient leakage")
        group_checked += 1
    if group_root.exists():
        for folder in sorted(group_root.glob("outer_*")) + [group_root / "full_home"]:
            if (folder / "group_membership.parquet").exists():
                validate_completed_group_projection(root, folder.name, deep)
                projections_checked += 1
    expected_pca = {
        pca_root / model / f"outer_{outer}" / sub / "metadata.json"
        for model in ("tcrbert_dualtask_aa_mean", "sceptr_cdr3_only")
        for outer in folds
        for sub in ("", "inner_0", "inner_1", "inner_2")
    }
    expected_group_scopes = {"full_home"} | {f"outer_{outer}" for outer in folds}
    completed_group_scopes = {path.parent.name for path in projected_memberships}
    missing_pca = sorted(str(path.relative_to(root)) for path in expected_pca - set(pca_meta))
    missing_group_scopes = sorted(expected_group_scopes - completed_group_scopes)
    status = "PASS" if not missing_pca and not missing_group_scopes else "PENDING"
    return {"status": status, "stage": "pca_groups",
            "mode": "deep" if deep else "quick",
            "pca_fit_receipts_checked": pca_checked,
            "outer_group_dictionaries_checked": group_checked,
            "completed_group_projections_checked": projections_checked,
            "missing_primary_pca_scopes": missing_pca,
            "missing_group_projection_scopes": missing_group_scopes,
            "elapsed_s": round(time.time()-started, 2)}


def validate_external(root: Path, deep: bool = False) -> dict:
    started = time.time()
    out = root / "groups/full_home"
    projection_path = out / "external_projection_receipt.json"
    if not projection_path.exists():
        return {"status": "PENDING", "stage": "external",
                "reason": "No label-blind external projection receipt"}
    receipt = read_json(projection_path)
    validate_completed_group_projection(root, "full_home", deep)
    frozen = root / "groups/frozen_home_candidates.tsv"
    need(frozen.is_file() and sha256(frozen) == receipt["frozen_candidates_sha256"],
         "Frozen candidate list differs from external projection")
    score = out / "external_direction.tsv"
    if receipt.get("status") == "started" and receipt.get("external_n_read") is False:
        need(not score.exists(), "External labels scored before projection completed")
        return {"status": "PENDING", "stage": "external",
                "reason": "Label-blind projection is still running"}
    need(receipt["status"] == "complete" and receipt["external_n_read"] is False,
         "External projection not completed label-blind")
    need(receipt["external_identity"] == "BioSample_proxy_unverified",
         "External identity basis differs")
    membership = out / "group_membership.parquet"
    burden = out / "patient_group_burden_external.parquet"
    need(sha256(membership) == receipt["membership_sha256"] and
         sha256(burden) == receipt["burden_sha256"],
         "External projection input/output hash mismatch")
    manifest = read_tsv(root / "config/cohort_samples.tsv")
    ext_keys = {row["patient_key"] for row in manifest if row["cohort"] == "external"}
    need(receipt["target_proxy_count"] == len(ext_keys),
         "External projection identity count differs")
    if deep:
        import pandas as pd
        observed = pd.read_parquet(burden, columns=["patient_key"])
        need(set(observed.patient_key).issubset(ext_keys),
             "External burden contains unknown or Home identity")
    if score.exists():
        score_meta = read_json(score.with_suffix(".json"))
        need(score_meta["projection_receipt_sha256"] == sha256(projection_path) and
             score_meta["frozen_candidates_sha256"] == receipt["frozen_candidates_sha256"],
             "External scoring did not use frozen label-blind projection")
        need(score_meta["projection_completed_utc"] == receipt["completed_utc"],
             "External score/projection timestamps differ")
    return {"status": "PASS", "stage": "external",
            "mode": "deep" if deep else "quick",
            "projection_completed_label_blind": True,
            "external_scoring_present": score.exists(),
            "elapsed_s": round(time.time()-started, 2)}


def write_receipt(root: Path, stage: str, outcome: dict) -> Path:
    target = root / "audit" / f"99_validate_{stage}.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = {"schema": "crc_tcr_independent_validation_v1", **outcome}
    temporary = target.with_name(target.name + f".partial.{os.getpid()}")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2) + "\n")
    os.replace(temporary, target)
    return target


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--stage", choices=("stage1", "folds", "embedding", "pca_groups", "external", "all"),
                        nargs="+", default=["all"])
    parser.add_argument("--model", choices=tuple(MODELS), nargs="+",
                        default=list(MODELS))
    parser.add_argument("--deep", action="store_true",
                        help="Scan Parquet payloads and vector content; avoid during live GPU embedding")
    parser.add_argument("--rehash-raw", action="store_true",
                        help="Recompute all 558 raw TSV/CLNS SHA256; requires --deep")
    parser.add_argument("--rehash-outputs", action="store_true",
                        help="Recompute all sample-derived Parquet SHA256; requires --deep")
    args = parser.parse_args()
    args.root = args.root.resolve()
    stages = ["stage1", "folds", "embedding", "pca_groups", "external"] if "all" in args.stage else args.stage
    outcomes = []
    for stage in stages:
        items = args.model if stage == "embedding" else [None]
        for model in items:
            name = stage + (f"_{model}" if model else "")
            try:
                if stage == "stage1":
                    result = validate_stage1(args.root, args.deep, args.rehash_raw, args.rehash_outputs)
                elif stage == "folds":
                    result = validate_folds(args.root)
                elif stage == "embedding":
                    result = validate_embedding(args.root, model, args.deep)
                elif stage == "pca_groups":
                    result = validate_pca_and_groups(args.root, args.deep)
                else:
                    result = validate_external(args.root, args.deep)
            except Exception as exc:
                result = {"status": "FAIL", "stage": stage, "model": model,
                          "error": f"{type(exc).__name__}: {exc}"}
            path = write_receipt(args.root, name, result)
            outcomes.append(result)
            print(f"{name}: {result['status']} -> {path}", flush=True)
            if result["status"] == "FAIL":
                print(result["error"], file=sys.stderr)
    return 1 if any(x["status"] == "FAIL" for x in outcomes) else (
        2 if any(x["status"] == "PENDING" for x in outcomes) else 0)


if __name__ == "__main__":
    raise SystemExit(main())
