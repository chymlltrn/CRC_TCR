from release_runtime import private_literal as _private_literal, release_path as _release_path
import importlib.util
import json
import hashlib
import shutil
import subprocess
import sys
from types import SimpleNamespace
from unittest import mock
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

MODULE = Path(__file__).resolve().parents[1] / "99_final_acceptance.py"
spec = importlib.util.spec_from_file_location("independent_final_acceptance", MODULE)
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)


class FinalAcceptanceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rules = json.loads((MODULE.parents[1] / "config/final_candidate_rules.json").read_text())

    def test_unsealed_run_is_pending_without_reading_outcomes(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "final").mkdir()
            (root / "final/decision.json").write_text('{"status":"COMPLETE_candidates","candidate_claim":true}')
            receipt = audit.run(root)
            self.assertEqual(receipt["status"], "PENDING")
            self.assertTrue((root / "audit/99_final_acceptance.json").exists())

    def test_full_family_bh_counts_unaligned_and_failed_models_as_one(self):
        adjusted = audit.bh([0.01, 0.04, np.nan, 1.0, 1.0])
        self.assertAlmostEqual(adjusted[0], 0.05)
        self.assertAlmostEqual(adjusted[1], 0.1)
        self.assertEqual(adjusted[-1], 1.0)
        self.assertEqual(adjusted[2], 1.0)

    def test_independent_candidate_predicate_checks_each_locked_gate(self):
        home = {"n_present": 12, "model_status": "ok", "q": 0.01,
                "or": 2.0, "delta": 0.1}
        external = {"carriers": 6, "evaluable": True, "positive": True, "delta": 0.05}
        alignment = {"pass": True, "matching_folds": 4, "method_match": True}
        ablation = {"mean_delta_logloss": 0.05, "patient_bootstrap_ci95": [0.01, 0.08],
                    "mean_delta_vs_matched_random": 0.02, "positive_fold_effects": 4,
                    "tested_outer_folds": 4, "positive_seed_effects": 3}
        safe = {"pooled": 8, "each_fold": True, "min_coverage": 0.9,
                "min_fraction": 0.9}
        result = audit.candidate_predicate(home, external, alignment, ablation, safe,
                                           0.01, self.rules, True)
        self.assertTrue(result["final_candidate"])
        self.assertFalse(audit.candidate_predicate(home, external, alignment, ablation,
                                                   safe, 0.05, self.rules, True)["final_candidate"])
        self.assertFalse(audit.candidate_predicate(home, external, alignment, ablation,
                                                   safe, 0.01, self.rules, False)["final_candidate"])
        bad_external = dict(external, carriers=4)
        self.assertFalse(audit.candidate_predicate(home, bad_external, alignment, ablation,
                                                   safe, 0.01, self.rules, True)["final_candidate"])

    def test_tampered_final_flag_is_rejected_after_source_recomputation(self):
        expected = {"home_pass": True, "external_pass": True,
                    "alignment_pass": True, "ablation_pass": False,
                    "final_candidate": False}
        with self.assertRaisesRegex(ValueError, "final_candidate"):
            audit.verify_reported_gates("g0", {**expected, "final_candidate": True}, expected)
        audit.verify_reported_gates("g0", expected, expected)

    def test_ablation_effect_collapses_three_seeds_and_rejects_missing_seed(self):
        rows = []
        for patient, label, full, deleted, random in [
            ("a", 1, 0.8, 0.6, 0.78),
            ("b", 0, 0.2, 0.4, 0.22),
            ("c", 1, 0.7, 0.5, 0.69)]:
            for seed in audit.SEEDS:
                rows.append({"patient_key": patient, "outer_fold": 0, "label": label,
                             "seed": seed, "full_prediction": full,
                             "deleted_prediction": deleted,
                             "random_deleted_prediction": random})
        frame = pd.DataFrame(rows)
        result = audit.effect(frame, {"random_seed": 11, "patient_bootstrap_draws": 100,
                                      "patient_signflip_draws": 100})
        self.assertEqual(result["n_unique_test_patients"], 3)
        self.assertGreater(result["mean_delta_logloss"], 0)
        self.assertGreater(result["mean_delta_vs_matched_random"], 0)
        with self.assertRaisesRegex(ValueError, "seed"):
            audit.effect(frame.iloc[:-1], {"random_seed": 11, "patient_bootstrap_draws": 10,
                                            "patient_signflip_draws": 10})

    def test_one_se_uses_all_inner_patients_and_favors_2048(self):
        rows = []
        for patient in range(12):
            label = patient % 2
            true_probability = 0.55 + 0.03 * (patient % 5)
            for model in audit.MODELS:
                for bag in audit.BAGS:
                    p = true_probability + (0.005 if bag == 4096 else 0.0)
                    if model == "sceptr_cdr3_only":
                        p -= 0.01
                    pred = p if label else 1 - p
                    for seed in audit.SEEDS:
                        rows.append({"patient_key": str(patient), "label": label,
                                     "model": model, "bag_size": bag, "seed": seed,
                                     "prediction": pred})
        frame = pd.DataFrame(rows)
        labels = {str(i): i % 2 for i in range(12)}
        self.assertEqual(audit.chosen(frame, labels), (audit.MODELS[0], 2048))
        self.assertEqual(audit.chosen(frame, labels, "sceptr_cdr3_only"),
                         ("sceptr_cdr3_only", 2048))

    def _home_fixture(self, root: Path, p: list[float], frozen_groups: list[str]):
        base = root / "groups/full_home"
        base.mkdir(parents=True)
        ids = [f"g{i}" for i in range(len(p))]
        source = pd.DataFrame({
            "group_id": ids, "adjusted_presence_p": p,
            "adjusted_presence_q_pooled": audit.bh(p),
            "adjusted_presence_or": [2.0] * len(p),
            "presence_delta": [0.1] * len(p),
            "n_present": [12] * len(p),
            "adjusted_model_status": ["ok"] * len(p)})
        source_path = base / "association_all_recorded_home_exploratory.tsv"
        source.to_csv(source_path, sep="\t", index=False)
        source_path.with_suffix(".json").write_text(json.dumps({
            "association_sha256": audit.sha(source_path), "n_patients": 114,
            "n_positive": 48, "n_negative": 66, "groups_tested": len(source),
            "min_present": 10, "min_absent": 10}))
        family = pd.DataFrame({
            "group_id": frozen_groups, "method": ["gliph2"] * len(frozen_groups),
            "projection_type": ["gliph2_seed_motif_projection"] * len(frozen_groups),
            "home_q_pooled": [float(source.set_index("group_id").loc[g, "adjusted_presence_q_pooled"])
                              for g in frozen_groups],
            "home_adjusted_presence_or": [2.0] * len(frozen_groups),
            "home_presence_delta": [0.1] * len(frozen_groups)})
        frozen_path = root / "groups/frozen_home_candidates.tsv"
        eligible_path = root / "groups/eligible_home_candidates.tsv"
        family.to_csv(frozen_path, sep="\t", index=False)
        eligible_path.write_bytes(frozen_path.read_bytes())
        frozen_path.with_suffix(".json").write_text(json.dumps({
            "candidate_sha256": audit.sha(frozen_path),
            "eligible_home_candidates_sha256": audit.sha(eligible_path),
            "external_n_read": False,
            "source_association_sha256": audit.sha(source_path),
            "source_association_receipt_sha256": audit.sha(source_path.with_suffix(".json")),
            "candidate_truncation": False, "max_frozen_candidates": None,
            "n_candidates": len(family)}))

    def test_empty_frozen_family_is_verified_against_all_home_tests(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._home_fixture(root, [1.0, 1.0], [])
            frozen, source = audit.verify_home_family(root, self.rules)
            self.assertEqual(len(frozen), 0)
            self.assertEqual(len(source), 2)

    def test_truncated_home_family_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._home_fixture(root, [0.001, 0.002], ["g0"])
            with self.assertRaisesRegex(ValueError, "incomplete"):
                audit.verify_home_family(root, self.rules)

    def test_outer_patient_test_exclusivity(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            path = root / "folds/folds.tsv"
            path.parent.mkdir()
            rows = []
            for outer in audit.OUTER:
                for i in range(114):
                    test = i % 5 == outer
                    rows.append({"patient_key": f"p{i:03d}", "recorded_n_binary": int(i < 48),
                                 "outer_fold": outer, "partition": "test" if test else "train",
                                 "inner_fold": np.nan if test else i % 3})
            frame = pd.DataFrame(rows)
            frame.to_csv(path, sep="\t", index=False)
            _, labels, scope = audit.folds_and_labels(root)
            self.assertEqual(len(labels), 114)
            self.assertEqual(sum(len(v["test"]) for v in scope.values()), 114)
            mask = frame.outer_fold.eq(1) & frame.patient_key.eq("p000")
            frame.loc[mask, "partition"] = "test"
            frame.loc[mask, "inner_fold"] = np.nan
            frame.to_csv(path, sep="\t", index=False)
            with self.assertRaisesRegex(ValueError, "multiple outer tests"):
                audit.folds_and_labels(root)

    def _sealed_empty_home_fixture(self, root: Path):
        project = MODULE.parents[1]
        for name in ("audit/06_final_rule_freeze_v7.json",
                     "audit/06_oof_ablation_provenance_revision_20260929.json",
                     "config/final_candidate_rules.json",
                     "config/mil_decision_rules.json",
                     "config/documented_blood31.tsv",
                     "src/05_mil.py", "src/06_decision.py",
                     "folds/folds.tsv"):
            dest = root / name
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(project / name, dest)
        self._home_fixture(root, [1.0, 1.0], [])
        base = root / "groups/full_home"
        (base / "patient_group_burden.parquet").write_bytes(b"fixture-burden")
        (base / "integrated_projection_receipt.json").write_text("{}")
        (base / "burden_receipt.json").write_text(json.dumps({
            "patient_group_burden_sha256": audit.sha(base / "patient_group_burden.parquet")}))
        blood = base / "association_verified_blood.tsv"
        frame = pd.DataFrame({"group_id": ["g0", "g1"], "adjusted_presence_p": [1., 1.],
                              "adjusted_presence_q_pooled": [1., 1.]})
        frame.to_csv(blood, sep="\t", index=False)
        eligible = base / "eligible_groups_verified_blood.tsv"
        frame[["group_id"]].to_csv(eligible, sep="\t", index=False)
        lock = root / "config/documented_blood31.tsv"
        blood.write_text(blood.read_text())
        blood.with_suffix(".json").write_text(json.dumps({
            "scope": "full_home", "subset": "verified_blood",
            "n_patients": 31, "n_positive": 18, "n_negative": 13,
            "groups_tested": 2, "association_sha256": audit.sha(blood),
            "eligible_groups_sha256": audit.sha(eligible),
            "documented_blood_sha256": audit.sha(lock),
            "folds_sha256": audit.sha(root / "folds/folds.tsv"),
            "subset_patient_sha256": audit.hash_keys(
                set(pd.read_csv(lock, sep="\t").patient_key)),
            "burden_sha256": audit.sha(base / "patient_group_burden.parquet"),
            "burden_receipt_sha256": audit.sha(base / "burden_receipt.json"),
            "integrated_projection_receipt_sha256":
                audit.sha(base / "integrated_projection_receipt.json")}))
        (root / "final").mkdir()
        (root / "final/decision.json").write_text(json.dumps({
            "status": "STOP_no_Home_association", "candidate_claim": False,
            "endpoint": "recorded N+ only"}))

    def test_empty_home_stop_is_validated_but_full_acceptance_waits_for_mil(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._sealed_empty_home_fixture(root)
            receipt = audit.run(root)
            self.assertEqual(receipt["status"], "PENDING")
            self.assertEqual(receipt["validated_early_decision"], "STOP_no_Home_association")
            self.assertTrue(any(_private_literal('private_62deb35004db4a1b') in path
                                for path in receipt["missing_or_unfinished"]))
            (root / "groups/full_home/association_verified_blood.json").unlink()
            self.assertEqual(audit.run(root)["status"], "FAIL")

    def test_cli_pending_exit_is_nonzero_and_receipt_is_retained(self):
        with tempfile.TemporaryDirectory() as tmp:
            command = [sys.executable, str(MODULE), "--root", tmp]
            process = subprocess.run(command, capture_output=True, text=True)
            self.assertEqual(process.returncode, 2)
            self.assertEqual(json.loads(process.stdout)["status"], "PENDING")
            self.assertEqual(json.loads((Path(tmp) / "audit/99_final_acceptance.json").read_text())["status"],
                             "PENDING")

    def test_candidate_card_export_is_bound_to_source_and_attachments(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            base = root / "groups/full_home"
            base.mkdir(parents=True)
            gid = "g0"
            data = pd.DataFrame({"group_id": [gid], "aa": ["CASS"],
                                 "v_gene": ["TRBV1"], "j_gene": ["TRBJ1"],
                                 "seq_idx": [7],
                                 "projection_type": ["gliph2_seed_motif_projection"]})
            source = base / "group_membership.parquet"
            data.to_parquet(source, index=False)
            folder = root / "final/candidates" / hashlib.sha256(gid.encode()).hexdigest()[:16]
            folder.mkdir(parents=True)
            members = folder / "members.parquet"
            data.to_parquet(members, index=False)
            first = folder / "first_100_members.tsv"
            data[["aa", "v_gene", "j_gene"]].to_csv(first, sep="\t", index=False)
            row = {"group_id": gid, "canonical_id": "c0", "method": "gliph2",
                   "projection_type": "gliph2_seed_motif_projection",
                   "family_definition": "test", "home_carriers": 12,
                   "home_q_pooled": 0.01, "home_presence_delta": 0.1,
                   "external_carriers": 6, "external_presence_delta": 0.02,
                   "ablation": {"mean_delta_logloss": 0.04},
                   "safe_deletion": {"test_carriers_safe_pooled": 8}}
            card = {**row, "endpoint": "recorded N+ exploratory",
                    "n_exact_aa_vj_members": 1, "members_path": str(members),
                    "members_sha256": audit.sha(members),
                    "full_home_membership_sha256": audit.sha(source),
                    "first_100_members_semantics": "lexical scan order, not importance ranking"}
            card_path = folder / "card.json"
            card_path.write_text(json.dumps(card))
            decision = {"candidate_cards": [str(card_path)]}
            audit.verify_candidate_cards(root, decision, [row])
            card_path.unlink()
            with self.assertRaisesRegex(ValueError, "attachment missing"):
                audit.verify_candidate_cards(root, decision, [row])
            card_path.write_text(json.dumps(card))
            data.loc[0, "aa"] = "CASR"
            data.to_parquet(members, index=False)
            with self.assertRaisesRegex(ValueError, "source SHA invalid"):
                audit.verify_candidate_cards(root, decision, [row])

    def test_external_frozen_extension_deletion_breaks_projection_chain(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            base = root / "groups/full_home"
            base.mkdir(parents=True)
            frozen_path = root / "groups/frozen_home_candidates.tsv"
            frozen = pd.DataFrame({"group_id": ["g0"], "method": ["gliph2"]})
            frozen.to_csv(frozen_path, sep="\t", index=False)
            frozen_path.with_suffix(".json").write_text("{}")
            home = base / "group_membership.parquet"
            pd.DataFrame({"group_id": ["g0"]}).to_parquet(home, index=False)
            manifest = root / "config/cohort_samples.tsv"
            manifest.parent.mkdir()
            pd.DataFrame({"cohort": ["external"] * 165,
                          "patient_key": [f"e{i:03d}" for i in range(165)]}).to_csv(
                              manifest, sep="\t", index=False)
            identity = audit.hash_keys(set(pd.read_csv(manifest, sep="\t").patient_key))
            ext = base / "gliph2_external_frozen_membership.parquet"
            pd.DataFrame({"group_id": ["g0"]}).to_parquet(ext, index=False)
            ext_receipt = base / "gliph2_external_frozen_projection_receipt.json"
            projection = {"manifest_identity_sha256": "manifest-fixture",
                          "target_patient_sha256": identity,
                          "frozen_group_ids_sha256": audit.hash_keys({"g0"})}
            ext_receipt.write_text(json.dumps({
                "status": "complete_external_frozen_scope", "scope": "full_home",
                "projection_population": "external", "external_label_read": False,
                "method": "gliph2", "membership_sha256": audit.sha(ext),
                "frozen_candidates_sha256": audit.sha(frozen_path),
                "frozen_candidate_receipt_sha256": audit.sha(frozen_path.with_suffix(".json")),
                "home_membership_sha256": audit.sha(home),
                "external_identity_sha256": identity,
                "group_ids_sha256": audit.hash_keys({"g0"}),
                "manifest_identity_sha256": "manifest-fixture"}))
            methods = {"gliph2": {"membership_sha256": audit.sha(ext),
                                  "receipt_sha256": audit.sha(ext_receipt),
                                  "group_ids_sha256": audit.hash_keys({"g0"})}}
            combined = base / "external_frozen_membership.parquet"
            shutil.copyfile(ext, combined)
            combined_receipt = base / "external_frozen_membership_receipt.json"
            combined_receipt.write_text(json.dumps({
                "status": "complete_external_frozen_scope", "scope": "full_home",
                "projection_population": "external", "external_label_read": False,
                "home_membership_sha256": audit.sha(home),
                "frozen_candidates_sha256": audit.sha(frozen_path),
                "frozen_candidate_receipt_sha256": audit.sha(frozen_path.with_suffix(".json")),
                "external_identity_sha256": identity,
                "group_ids_sha256": audit.hash_keys({"g0"}), "methods": methods,
                "membership_sha256": audit.sha(combined),
                "n_membership_rows": 1, "n_frozen_groups": 1}))
            projection.update({
                "external_frozen_membership_sha256": audit.sha(combined),
                "external_frozen_membership_receipt_sha256": audit.sha(combined_receipt),
                "method_extension_receipts": methods})
            audit.verify_external_frozen_sources(root, frozen, projection)
            ext.unlink()
            with self.assertRaisesRegex(ValueError, "extension attachment missing"):
                audit.verify_external_frozen_sources(root, frozen, projection)

    def test_projection_cache_current_byte_or_pca_mutation_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            model = audit.MODELS[0]
            raw = root / "embeddings" / model / "bank_fp16.npy"
            raw.parent.mkdir(parents=True)
            np.save(raw, np.zeros((2, 4), dtype=np.float16))
            raw_meta = raw.with_name("meta.json")
            raw_meta.write_text(json.dumps({"bank_shape": [2, 4]}))
            pca = root / "pca" / model / "outer_0"
            pca.mkdir(parents=True)
            for name in ("pca.npz", "metadata.json", "stability.json"):
                (pca / name).write_text("{}")
            projected = root / "mil/projections" / model / "outer_0_d2.npy"
            projected.parent.mkdir(parents=True)
            np.save(projected, np.ones((2, 2), dtype=np.float16))
            stat = raw.stat()
            receipt = projected.with_suffix(".json")
            data = {
                "status": "complete", "projection_sha256": audit.sha(projected),
                "projection_size_bytes": projected.stat().st_size,
                "projection_dtype": "float16", "nonfinite_values": 0, "dim": 2,
                "pca_sha256": audit.sha(pca / "pca.npz"),
                "scope_pca_stability_sha256": audit.sha(pca / "stability.json"),
                "pca_metadata_sha256": audit.sha(pca / "metadata.json"),
                "raw_bank_meta_sha256": audit.sha(raw_meta),
                "raw_bank_stamp": {"path": str(raw.resolve()), "bytes": stat.st_size,
                                   "mtime_ns": stat.st_mtime_ns}}
            receipt.write_text(json.dumps(data))
            audit.verify_projection_cache(root, model, 0, None, 2)
            np.save(projected, np.zeros((2, 2), dtype=np.float16))
            with self.assertRaisesRegex(ValueError, "current bytes/PCA source mismatch"):
                audit.verify_projection_cache(root, model, 0, None, 2)
            np.save(projected, np.ones((2, 2), dtype=np.float16))
            (pca / "pca.npz").write_text("changed")
            with self.assertRaisesRegex(ValueError, "current bytes/PCA source mismatch"):
                audit.verify_projection_cache(root, model, 0, None, 2)

    def test_sensitivity_attachment_deletion_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            required = root / "groups/full_home/sensitivity_pgen_receipt.json"
            required.parent.mkdir(parents=True)
            required.write_text("{}")
            stub = SimpleNamespace(sensitivity_required_paths=lambda root: [required],
                                   validate_sensitivity_chain=lambda root, frozen: {"pgen": "ok"})
            with mock.patch.object(audit, "frozen_source_module", return_value=stub):
                audit.verify_sensitivity_chain(
                    root, root / "groups/frozen_home_candidates.tsv",
                    {"status": "COMPLETE_no_candidate", "sensitivities": {"pgen": "ok"}},
                    {"stage06_code_sha256": "fixture"})
                required.unlink()
                with self.assertRaisesRegex(ValueError, "attachment missing"):
                    audit.verify_sensitivity_chain(
                        root, root / "groups/frozen_home_candidates.tsv",
                        {"status": "COMPLETE_no_candidate", "sensitivities": {"pgen": "ok"}},
                        {"stage06_code_sha256": "fixture"})



if __name__ == "__main__":
    unittest.main()
