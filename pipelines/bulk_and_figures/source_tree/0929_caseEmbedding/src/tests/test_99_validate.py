"""Small synthetic contracts for the independent audit CLI."""
from __future__ import annotations
from release_runtime import private_literal as _private_literal, release_path as _release_path
import csv
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(_release_path('path_1304218e86d2e596'))
SPEC = importlib.util.spec_from_file_location("audit99", ROOT / "src/99_validate.py")
audit = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(audit)


def tsv(path: Path, fields: list[str], rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def js(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


class AuditTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="audit99_", dir=ROOT / "tmp")
        self.root = Path(self.temp.name)

    def tearDown(self):
        self.temp.cleanup()

    def make_folds(self):
        rows = [
            dict(cohort="home", sample_id=f"S{i}", patient_key=f"H{i}",
                 recorded_n_binary=str(i % 2))
            for i in range(4)
        ]
        rows += [dict(cohort="external", sample_id="E0", patient_key="E0",
                      recorded_n_binary="1")]
        tsv(self.root / "config/cohort_samples.tsv", list(rows[0]), rows)
        folds = []
        for outer in range(2):
            test = {f"H{2*outer}", f"H{2*outer+1}"}
            train = sorted({f"H{i}" for i in range(4)} - test)
            for row in rows[:4]:
                key = row["patient_key"]
                folds.append(dict(patient_key=key,
                    recorded_n_binary=row["recorded_n_binary"], outer_fold=outer,
                    partition="test" if key in test else "train",
                    inner_fold="" if key in test else str(train.index(key))))
        tsv(self.root / "folds/folds.tsv", list(folds[0]), folds)
        return rows

    def test_folds_patient_boundaries(self):
        self.make_folds()
        result = audit.validate_folds(self.root, expected_outer=2, expected_inner=2)
        self.assertEqual(result["status"], "PASS")
        path = self.root / "folds/folds.tsv"
        content = path.read_text().replace("H0\t0\t0\ttest", "E0\t0\t0\ttest")
        path.write_text(content)
        with self.assertRaises(audit.AuditError):
            audit.validate_folds(self.root, expected_outer=2, expected_inner=2)

    def test_folds_receipt_uses_identity_and_home_label_hashes(self):
        rows = self.make_folds()
        identity_hash, home_hash = audit.manifest_scope_hashes(rows)
        js(self.root / "audit/01_ingest_summary.json", {
            "source_contract_sha256": "a" * 64,
            "input_sha256_manifest_sha256": "b" * 64,
            "union_sha256": "c" * 64,
        })
        meta = {
            "manifest_identity_sha256": identity_hash,
            "home_split_label_sha256": home_hash,
            "folds_sha256": audit.sha256(self.root / "folds/folds.tsv"),
            "stage1_source_contract_sha256": "a" * 64,
            "stage1_input_sha256_manifest_sha256": "b" * 64,
            "stage1_union_sha256": "c" * 64,
            "external_in_folds": False,
            "n_home_patients": 4, "rows": 8,
            "outer_folds": 2, "inner_folds": 2,
        }
        js(self.root / "folds/folds.json", meta)
        self.assertEqual(audit.validate_folds(
            self.root, expected_outer=2, expected_inner=2)["status"], "PASS")
        rows[-1]["recorded_n_binary"] = "0"
        tsv(self.root / "config/cohort_samples.tsv", list(rows[0]), rows)
        self.assertEqual(audit.validate_folds(
            self.root, expected_outer=2, expected_inner=2)["status"], "PASS")
        meta["home_split_label_sha256"] = "0" * 64
        js(self.root / "folds/folds.json", meta)
        with self.assertRaises(audit.AuditError):
            audit.validate_folds(self.root, expected_outer=2, expected_inner=2)

    def test_home_scope_projection_gate(self):
        self.make_folds()
        folder = self.root / "groups/full_home"
        folder.mkdir(parents=True)
        merged = pd.DataFrame({"method": ["gliph2", "tcrdist3"],
                               "group_id": ["G1", "T1"], "aa": ["CASSF", "CASST"]})
        merged_path = folder / "group_membership.parquet"
        merged.to_parquet(merged_path, index=False)
        with self.assertRaises(audit.AuditError):
            audit.validate_completed_group_projection(self.root, "full_home")
        for method, stem in (("gliph2", "gliph2"), ("tcrdist3", "tcrdist_home")):
            path = folder / f"{stem}_group_membership.parquet"
            merged.loc[merged.method.eq(method)].to_parquet(path, index=False)
            js(folder / f"{stem}_projection_receipt.json", {
                "status": "complete_home_scope", "method": method, "scope": "full_home",
                "projection_population": "home", "membership_sha256": audit.sha256(path)})
        js(folder / "integrated_projection_receipt.json", {
            "status": "complete_home_scope", "projection_population": "home",
            "scope": "full_home", "methods": ["gliph2", "tcrdist3"],
            "method_statuses": {"gliph2": "complete_home_scope",
                                "tcrdist3": "complete_home_scope"},
            "coverage_gap_methods": [],
            "membership_sha256": audit.sha256(merged_path),
            "train_patient_sha256": audit.patient_hash({f"H{i}" for i in range(4)})})
        self.assertEqual(audit.validate_completed_group_projection(self.root, "full_home",
                         deep=True)["status"], "PASS")
        receipt_path = folder / "tcrdist_home_projection_receipt.json"
        data = json.loads(receipt_path.read_text())
        data["status"] = "pilot_only"
        js(receipt_path, data)
        with self.assertRaises(audit.AuditError):
            audit.validate_completed_group_projection(self.root, "full_home")
        js(receipt_path, {"status": "STOP_resource", "method": "tcrdist3",
                          "scope": "full_home", "reason": "Exact full projection exceeded RSS gate"})
        merged.iloc[:1].to_parquet(merged_path, index=False)
        js(folder / "integrated_projection_receipt.json", {
            "status": "complete_home_scope", "projection_population": "home",
            "scope": "full_home", "methods": ["gliph2"],
            "method_statuses": {"gliph2": "complete_home_scope",
                                "tcrdist3": "STOP_resource"},
            "coverage_gap_methods": ["tcrdist3"],
            "membership_sha256": audit.sha256(merged_path),
            "train_patient_sha256": audit.patient_hash({f"H{i}" for i in range(4)})})
        self.assertEqual(
            audit.validate_completed_group_projection(self.root, "full_home",
                                                      deep=True)["completed_methods"],
            ["gliph2"])
        for alternate_status in ("resource_failed", "inconclusive", "failed", "infeasible"):
            detail = json.loads(receipt_path.read_text())
            detail["status"] = alternate_status
            js(receipt_path, detail)
            integrated_path = folder / "integrated_projection_receipt.json"
            integrated = json.loads(integrated_path.read_text())
            integrated["method_statuses"]["tcrdist3"] = alternate_status
            js(integrated_path, integrated)
            result = audit.validate_completed_group_projection(
                self.root, "full_home", deep=True)
            self.assertEqual(result["completed_methods"], ["gliph2"])
            self.assertEqual(result["coverage_gaps"]["tcrdist3"]["status"],
                             alternate_status)
        frozen = self.root / "groups/frozen_home_candidates.tsv"
        tsv(frozen, ["group_id"], [{"group_id": "G1"}])
        burden = folder / "patient_group_burden_external.parquet"
        pd.DataFrame({"patient_key": ["E0"], "group_id": ["G1"],
                      "presence": [1]}).to_parquet(burden, index=False)
        projection = folder / "external_projection_receipt.json"
        js(projection, {
            "status": "complete", "external_n_read": False,
            "frozen_candidates_sha256": audit.sha256(frozen),
            "membership_sha256": audit.sha256(merged_path),
            "burden_sha256": audit.sha256(burden),
            "external_identity": "BioSample_proxy_unverified",
            "target_proxy_count": 1,
            "completed_utc": "2026-09-29T00:00:00+00:00"})
        self.assertEqual(audit.validate_external(self.root, deep=True)["status"], "PASS")
        frozen.write_text("group_id\nchanged\n")
        with self.assertRaises(audit.AuditError):
            audit.validate_external(self.root)

    def test_pca_groups_stays_pending_until_projection_scopes_complete(self):
        self.make_folds()
        js(self.root / "audit/01_ingest_summary.json", {"union_sha256": "a" * 64})
        train = {"H2", "H3"}
        js(self.root / "groups/outer_0/description.json", {
            "scope": "outer_0", "patient_ids": sorted(train),
            "train_patient_sha256": audit.patient_hash(train),
            "role": "outer_train_dictionary_input", "phenotype_blind": True})
        result = audit.validate_pca_and_groups(self.root)
        self.assertEqual(result["status"], "PENDING")
        self.assertIn("full_home", result["missing_group_projection_scopes"])
        self.assertTrue(result["missing_primary_pca_scopes"])

    def test_embedding_bank_coverage_and_payload(self):
        model = "sceptr_cdr3_only"
        self.assertEqual(audit.validate_embedding(self.root, model)["status"], "PENDING")
        out = self.root / "embeddings" / model
        (out / "shards").mkdir(parents=True)
        union_hash = "a" * 64
        js(self.root / "audit/01_ingest_summary.json", {"union_sha256": union_hash})
        vectors = np.ones((3, 64), dtype=np.float16)
        shard = out / "shards/000000000000_000000000003.npy"
        np.save(shard, vectors)
        js(shard.with_suffix(".json"), {
            "start": 0, "end": 3, "rows": 3, "dimension": 64, "dtype": "float16",
            "nonfinite": 0, "input_sha256": union_hash, "model_fingerprint": "b"*64,
            "shard_sha256": audit.sha256(shard)})
        bank = out / "bank_fp16.npy"
        np.save(bank, vectors)
        stat = bank.stat()
        js(out / "meta.json", {
            "status": "complete", "input_sha256": union_hash,
            "model_fingerprint": "b"*64, "model_spec": {"hidden_dim": 64},
            "n_total": 3, "n_done": 3, "bank_shape": [3, 64],
            "bank_dtype": "float16", "shard_rows": 50000,
            "bank_size_bytes": stat.st_size, "bank_mtime_ns": stat.st_mtime_ns,
            "bank_sha256": audit.sha256(bank)})
        self.assertEqual(audit.validate_embedding(self.root, model, deep=True)["status"], "PASS")
        altered = np.load(bank)
        altered[0,0] = 2
        np.save(bank, altered)
        meta = json.loads((out / "meta.json").read_text())
        meta["bank_size_bytes"] = bank.stat().st_size
        meta["bank_mtime_ns"] = bank.stat().st_mtime_ns
        js(out / "meta.json", meta)
        with self.assertRaises(audit.AuditError):
            audit.validate_embedding(self.root, model, deep=True)

    def test_stage1_clone_read_conservation(self):
        records = [
            dict(cohort="home", sample_id="S0", patient_key="H0",
                 recorded_n_binary="0", expected_unique_aa="1", expected_read_count="5"),
            dict(cohort="external", sample_id="E0", patient_key="E0",
                 recorded_n_binary="1", expected_unique_aa="1", expected_read_count="7"),
        ]
        for rec in records:
            stem = rec["cohort"] + "__" + rec["sample_id"]
            raw_tsv = self.root / "raw" / (stem + ".tsv")
            raw_clns = self.root / "raw" / (stem + ".clns")
            raw_tsv.parent.mkdir(parents=True, exist_ok=True)
            raw_tsv.write_text("source\n")
            raw_clns.write_text("source\n")
            rec["clone_tsv_path"] = str(raw_tsv)
            rec["clns_path"] = str(raw_clns)
            count = int(rec["expected_read_count"])
            clone = pd.DataFrame({
                "clone_id": [0], "aa": ["CASSF"], "read_count": [count],
                "productive": [True], "qc_pass": [True],
                "v_gene": ["TRBV1"], "j_gene": ["TRBJ1"]})
            aa = pd.DataFrame({
                "cohort": [rec["cohort"]], "sample_id": [rec["sample_id"]],
                "patient_key": [rec["patient_key"]], "aa": ["CASSF"],
                "read_count": [count]})
            clone_path = self.root / "data/clone_detail" / (stem + ".parquet")
            aa_path = self.root / "data/patient_aa" / (stem + ".parquet")
            clone_path.parent.mkdir(parents=True, exist_ok=True)
            aa_path.parent.mkdir(parents=True, exist_ok=True)
            clone.to_parquet(clone_path, index=False)
            aa.to_parquet(aa_path, index=False)
            js(self.root / "audit/sample_receipts" / (stem + ".json"), {
                "status": "PASS", "cohort": rec["cohort"],
                "sample_id": rec["sample_id"], "patient_key": rec["patient_key"],
                "recorded_n_binary": int(rec["recorded_n_binary"]),
                "six_fields_equal_by_clone_id": True,
                "source_tsv_path": str(raw_tsv),
                "source_tsv_stat": {"bytes": raw_tsv.stat().st_size,
                                   "mtime_ns": raw_tsv.stat().st_mtime_ns},
                "source_tsv_sha256": audit.sha256(raw_tsv),
                "source_clns_path": str(raw_clns),
                "source_clns_stat": {"bytes": raw_clns.stat().st_size,
                                    "mtime_ns": raw_clns.stat().st_mtime_ns},
                "source_clns_sha256": audit.sha256(raw_clns),
                "clone_detail_path": str(clone_path),
                "clone_detail_sha256": audit.sha256(clone_path),
                "patient_aa_path": str(aa_path),
                "patient_aa_sha256": audit.sha256(aa_path),
                "source_clone_rows": 1, "productive_clone_rows": 1,
                "nonproductive_clone_rows": 0, "qc_clone_rows": 1,
                "unique_aa": 1, "qc_read_count": count})
        tsv(self.root / "config/cohort_samples.tsv", list(records[0]), records)
        source_rows = []
        for rec in records:
            for typ, field in (("clone_tsv", "clone_tsv_path"), ("clns", "clns_path")):
                source = Path(rec[field])
                source_rows.append({
                    "cohort": rec["cohort"], "sample_id": rec["sample_id"],
                    "artifact_type": typ, "source_path": str(source),
                    "resolved_path": str(source.resolve()),
                    "bytes": source.stat().st_size,
                    "mtime_ns": source.stat().st_mtime_ns,
                    "sha256": audit.sha256(source)})
        tsv(self.root / "audit/01_input_sha256.tsv", list(source_rows[0]), source_rows)
        union = self.root / "data/aa_union.parquet"
        pd.DataFrame({"seq_idx": np.array([0], dtype=np.int64),
                      "aa": ["CASSF"]}).to_parquet(union, index=False)
        js(self.root / "audit/01_ingest_summary.json", {
            "status": "PASS",
            "cohort_samples_sha256": audit.sha256(self.root / "config/cohort_samples.tsv"),
            "input_sha256_manifest_sha256": audit.sha256(self.root / "audit/01_input_sha256.tsv"),
            "union_unique_aa": 1, "union_sha256": audit.sha256(union),
            "sum_per_sample_unique_aa": 2, "total_qc_clones": 2,
            "total_qc_assigned_reads": 12})
        self.assertEqual(audit.validate_stage1(self.root, deep=True,
                         expected_home=1, expected_external=1)["status"], "PASS")
        receipt = self.root / "audit/sample_receipts/home__S0.json"
        altered = json.loads(receipt.read_text())
        altered["qc_read_count"] = 4
        js(receipt, altered)
        with self.assertRaises(audit.AuditError):
            audit.validate_stage1(self.root, expected_home=1, expected_external=1)


if __name__ == "__main__":
    unittest.main()
