from release_runtime import private_literal as _private_literal, release_path as _release_path
import importlib.util
import json
import pathlib
import tempfile
import unittest
import hashlib

import numpy as np
from unittest import mock

import pandas as pd

MODULE = pathlib.Path(__file__).resolve().parents[1] / "06_decision.py"
spec = importlib.util.spec_from_file_location("final_decision", MODULE)
decision = importlib.util.module_from_spec(spec)
spec.loader.exec_module(decision)


class DecisionTests(unittest.TestCase):
    def frame(self):
        rows = []
        for patient, y, full, deleted, random in [
            ("a", 1, 0.8, 0.6, 0.78),
            ("b", 0, 0.2, 0.4, 0.22),
            ("c", 1, 0.7, 0.5, 0.69),
        ]:
            for seed in (11, 23, 47):
                rows.append(dict(patient_key=patient, outer_fold=0, label=y,
                                 seed=seed, full_prediction=full,
                                 deleted_prediction=deleted,
                                 random_deleted_prediction=random))
        return pd.DataFrame(rows)

    def test_collapses_three_seeds_to_three_patients(self):
        out = decision.patient_effect(self.frame(), 1, 100, 100)
        self.assertEqual(out["n_unique_test_patients"], 3)
        self.assertEqual(out["positive_seed_effects"], 3)
        self.assertGreater(out["mean_delta_logloss"], 0)
        self.assertGreater(out["mean_delta_vs_matched_random"], 0)

    def test_reject_duplicate_patient_seed(self):
        frame = self.frame()
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            decision.patient_effect(pd.concat([frame, frame.iloc[:1]]), 1, 100, 100)

    def test_bh_all_frozen_family_pvalues(self):
        q = decision.bh([0.01, 0.04, 1.0, 1.0, 1.0])
        self.assertAlmostEqual(q[0], 0.05)
        self.assertAlmostEqual(q[1], 0.1)
        self.assertEqual(q[-1], 1.0)

    def test_empty_home_set_requires_frozen_source_receipt(self):
        with tempfile.TemporaryDirectory() as temp:
            root = pathlib.Path(temp)
            rules = root / "config/final_candidate_rules.json"
            rules.parent.mkdir(parents=True)
            rules.write_text('{"frozen_before_outcome_run":true,"max_families":null,"candidate_scope":"all_home_primary_eligible"}')
            freeze = root / decision.RULE_FREEZE
            freeze.parent.mkdir(parents=True)
            freeze.write_text(json.dumps({
                "rules_sha256": decision.sha256(rules),
                "all_outcome_outputs_absent": True}))
            frozen = root / "groups/frozen_home_candidates.tsv"
            frozen.parent.mkdir(parents=True)
            pd.DataFrame(columns=["group_id", "method", "projection_type",
                                  "family_definition", "home_q_pooled",
                                  "home_adjusted_presence_or", "home_presence_delta"]).to_csv(
                                      frozen, sep="\t", index=False)
            with self.assertRaises(FileNotFoundError):
                decision.run(root)

    def test_empty_home_freeze_writes_final_negative_report_after_home31(self):
        with tempfile.TemporaryDirectory() as temp:
            root = pathlib.Path(temp)
            rules = root / "config/final_candidate_rules.json"
            rules.parent.mkdir(parents=True)
            rules.write_text(json.dumps({
                "frozen_before_outcome_run": True,
                "max_families": None,
                "candidate_scope": "all_home_primary_eligible"}))
            freeze = root / decision.RULE_FREEZE
            freeze.parent.mkdir(parents=True)
            freeze.write_text(json.dumps({
                "rules_sha256": decision.sha256(rules),
                "all_outcome_outputs_absent": True}))
            frozen = root / "groups/frozen_home_candidates.tsv"
            frozen.parent.mkdir(parents=True)
            pd.DataFrame(columns=[
                "group_id", "method", "projection_type",
                "family_definition", "home_q_pooled",
                "home_adjusted_presence_or", "home_presence_delta"
            ]).to_csv(frozen, sep="\t", index=False)
            blood = root / "groups/full_home/association_verified_blood.tsv"
            blood.parent.mkdir(parents=True)
            blood.write_text("group_id\tadjusted_presence_q_pooled\n")
            blood.with_suffix(".json").write_text('{"n_patients":31}')
            stats = mock.Mock()
            with mock.patch.object(decision, "verify_receipts"),\
                 mock.patch.object(decision, "validate_code_revision", return_value="sealed"),\
                 mock.patch.object(decision, "_group_stats_module", return_value=stats):
                result = decision.run(root)
            self.assertEqual(result["status"], "STOP_no_Home_association")
            self.assertIs(result["candidate_claim"], False)
            self.assertTrue((root / "final/decision.json").exists())
            self.assertTrue((root / "final/report.md").exists())
            self.assertFalse((root / "groups/full_home/external_direction.tsv").exists())
            stats._validate_association_receipt.assert_called_once_with(
                root, "full_home", "verified_blood")

    def test_nonempty_home_freeze_uses_stage04_complete_validator(self):
        with tempfile.TemporaryDirectory() as temp:
            root = pathlib.Path(temp)
            frozen = root / "groups/frozen_home_candidates.tsv"
            frozen.parent.mkdir(parents=True)
            pd.DataFrame([{"group_id": "g1"}]).to_csv(frozen, sep="\t", index=False)
            eligible = root / "groups/eligible_home_candidates.tsv"
            pd.DataFrame([{"group_id": "g1"}]).to_csv(eligible, sep="\t", index=False)
            frozen.with_suffix(".json").write_text(json.dumps({
                "eligible_home_candidates_sha256": decision.sha256(eligible),
                "n_eligible_home_candidates": 1, "n_candidates": 1,
                "max_frozen_candidates": None, "candidate_truncation": False}))
            stats = mock.Mock()
            with mock.patch.object(decision, "_group_stats_module", return_value=stats):
                decision._validate_home_chain(root, frozen)
            stats._validate_frozen_file.assert_called_once_with(root, frozen)

    def test_all_six_eligible_home_groups_must_be_frozen(self):
        with tempfile.TemporaryDirectory() as temp:
            root = pathlib.Path(temp)
            groups = root / "groups"
            groups.mkdir()
            rows = [{"group_id": f"g{i}"} for i in range(6)]
            eligible = groups / "eligible_home_candidates.tsv"
            frozen = groups / "frozen_home_candidates.tsv"
            pd.DataFrame(rows).to_csv(eligible, sep="\t", index=False)
            pd.DataFrame(rows).to_csv(frozen, sep="\t", index=False)
            receipt = {
                "eligible_home_candidates_sha256": decision.sha256(eligible),
                "n_eligible_home_candidates": 6, "n_candidates": 6,
                "max_frozen_candidates": None, "candidate_truncation": False}
            frozen.with_suffix(".json").write_text(json.dumps(receipt))
            with mock.patch.object(decision, "_group_stats_module"):
                decision._validate_home_chain(root, frozen)
                pd.DataFrame(rows[:5]).to_csv(frozen, sep="\t", index=False)
                with self.assertRaisesRegex(ValueError, "truncated family set"):
                    decision._validate_home_chain(root, frozen)

    def test_home_only_gate_does_not_read_external_n(self):
        with tempfile.TemporaryDirectory() as temp:
            root = pathlib.Path(temp)
            frozen = root / "groups/frozen_home_candidates.tsv"
            frozen.parent.mkdir(parents=True)
            frozen.write_text("group_id\n")
            membership = root / "groups/full_home/group_membership.parquet"
            membership.parent.mkdir(parents=True)
            membership.write_bytes(b"synthetic membership")
            frozen.with_suffix(".json").write_text(json.dumps({
                "candidate_sha256": decision.sha256(frozen),
                "external_n_read": False,
                "source_subset": "all_recorded_home_exploratory",
                "membership_sha256": decision.sha256(membership)}))
            with mock.patch.object(decision, "_validate_home_chain"),\
                 mock.patch.object(decision, "_external_labels",
                                   side_effect=AssertionError("external N read")):
                decision.verify_receipts(root, frozen, require_external=False)

    def test_home_and_external_use_same_presence_difference_scale(self):
        rule = {"minimum_carriers": 10,
                "adjusted_presence_q_strictly_below": 0.05,
                "adjusted_presence_or_above": 1,
                "presence_delta_above": 0}
        candidate = {"home_q_pooled": 0.01,
                     "home_adjusted_presence_or": 2.0,
                     "home_presence_delta": 0.15}
        row = pd.Series({"n_present": 12,
                         "adjusted_model_status": "ok",
                         "adjusted_presence_q_pooled": 0.01,
                         "adjusted_presence_or": 2.0,
                         "presence_delta": 0.15})
        self.assertTrue(decision.home_primary_pass(row, candidate, rule))
        row["presence_delta"] = -0.15
        candidate["home_presence_delta"] = -0.15
        self.assertFalse(decision.home_primary_pass(row, candidate, rule))
        row["presence_delta"] = 0.15
        self.assertFalse(decision.home_primary_pass(row, candidate, rule))

    def test_card_rejects_projection_type_drift(self):
        with tempfile.TemporaryDirectory() as temp:
            root = pathlib.Path(temp)
            source = root / "groups/full_home/group_membership.parquet"
            source.parent.mkdir(parents=True)
            pd.DataFrame([{"group_id": "g1", "aa": _private_literal('private_8b5efd300a8432fa'),
                           "v_gene": "TRBV7-2*01", "j_gene": "TRBJ2-5*01",
                           "seq_idx": 4,
                           "projection_type": "gliph2_seed_motif_projection"}]).to_parquet(source)
            with self.assertRaisesRegex(ValueError, "projection type differs"):
                decision.write_candidate_cards(root, [{
                    "group_id": "g1", "canonical_id": "cf1", "method": "tcrdist3",
                    "projection_type": "tcrdist3_train_seed_full_projection",
                    "family_definition": "training seed and full projection",
                    "home_carriers": 12, "home_q_pooled": 0.01,
                    "external_carriers": 6, "external_presence_delta": 0.2,
                    "ablation": {}, "safe_deletion": {}}])

    def test_incomplete_upstream_yields_pending(self):
        with tempfile.TemporaryDirectory() as temp:
            root = pathlib.Path(temp)
            (root / "config").mkdir()
            rules_path = root / "config/final_candidate_rules.json"
            rules_path.write_text('{"frozen_before_outcome_run":true,"max_families":null,"candidate_scope":"all_home_primary_eligible"}')
            (root / "audit").mkdir()
            (root / decision.RULE_FREEZE).write_text(
                '{"rules_sha256":"' + decision.sha256(rules_path) +
                '","all_outcome_outputs_absent":true}')
            out = decision.run(root)
            self.assertEqual(out["status"], "PENDING")
            self.assertFalse(out["candidate_claim"])
            self.assertTrue((root / "final/report.md").exists())


class AblationProvenanceTests(unittest.TestCase):
    def test_stage06_logloss_matches_stage05_clipping_at_extremes(self):
        y = np.array([1, 0, 1, 0])
        probabilities = np.array([0.0, 1.0, 1e-8, 1 - 1e-8])
        expected = -(y * np.log(np.clip(probabilities, 1e-6, 1 - 1e-6)) +
                     (1 - y) * np.log1p(-np.clip(probabilities, 1e-6, 1 - 1e-6)))
        np.testing.assert_allclose(decision.logloss(y, probabilities), expected, rtol=0, atol=1e-12)

    def test_stale_mil_gate_rejected_after_oof_chain_recomputation(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            oof = root / _private_literal('private_4b09056568ab6f20')
            oof.parent.mkdir(parents=True)
            oof.write_text("patient_key\n")
            (root / "mil/oof/summary.json").write_text(json.dumps({
                "code_sha256": decision.EXPECTED_MIL_CODE_SHA256}))
            current = {"status": "blocked", "oof_sha256": decision.sha256(oof),
                       "reason": "No stable gain"}
            fake = mock.Mock()
            fake.ablation_gate.return_value = current
            with mock.patch.object(decision, "_mil_module", return_value=fake):
                decision.validate_mil_oof_chain(root, current)
                stale = {**current, "status": "pass"}
                with self.assertRaisesRegex(ValueError, "differs from current complete pooled OOF"):
                    decision.validate_mil_oof_chain(root, stale)
            (root / "mil/oof/summary.json").write_text(json.dumps({"code_sha256": "stale"}))
            with mock.patch.object(decision, "_mil_module", return_value=fake):
                with self.assertRaisesRegex(ValueError, "Stage 05 source SHA"):
                    decision.validate_mil_oof_chain(root, current)

    def test_paired_full_prediction_mutation_rejected_even_if_summary_sha_resigned(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            def table(rel, rows):
                path = root / rel
                path.parent.mkdir(parents=True, exist_ok=True)
                pd.DataFrame(rows).to_csv(path, sep="\t", index=False)
                return path
            def receipt(path):
                path.with_suffix(".json").write_text(json.dumps({
                    "status": "complete", "code_sha256": decision.EXPECTED_MIL_CODE_SHA256,
                    "prediction_sha256": decision.sha256(path)}))
            table("folds/folds.tsv", [{"patient_key": "a", "outer_fold": 0, "partition": "test"},
                                      {"patient_key": "b", "outer_fold": 0, "partition": "test"}])
            model, bag, gid = "sceptr_cdr3_only", 2048, "g1"
            group_dir = root / "mil/ablation/outer_0" / hashlib.sha256(gid.encode()).hexdigest()[:16]
            rows = []
            for seed in (11, 23, 47):
                base = root / f"mil/cv/outer_0/final/{model}/bag_{bag}/seed_{seed}"
                full = table(str(base.relative_to(root) / "prediction.tsv"), [
                    {"patient_key": p, "label": y, "prediction": f, "model": model,
                     "bag_size": bag, "seed": seed, "outer_fold": 0}
                    for p, y, f in (("a", 1, 0.8), ("b", 0, 0.2))])
                (base / "receipt.json").write_text(json.dumps({
                    "status": "complete", "code_sha256": decision.EXPECTED_MIL_CODE_SHA256,
                    "prediction_sha256": decision.sha256(full)}))
                for treatment, values in (("group_removed", (0.6, 0.4)),
                                          ("matched_random", (0.79, 0.21))):
                    path = table(str((group_dir / f"seed_{seed}_{treatment}.tsv").relative_to(root)), [
                        {"patient_key": p, "label": y, "prediction": values[i],
                         "seed": seed, "treatment": treatment, "group_id": gid}
                        for i, (p, y) in enumerate((("a", 1), ("b", 0)))])
                    receipt(path)
                for p, y, f, d, r in (("a", 1, .8, .6, .79), ("b", 0, .2, .4, .21)):
                    lf, ld, lr = decision.logloss(np.array([y,y,y]), np.array([f,d,r]))
                    rows.append({"outer_fold": 0, "group_id": gid, "patient_key": p,
                        "label": y, "seed": seed, "model": model, "bag_size": bag,
                        "full_prediction": f, "deleted_prediction": d,
                        "random_deleted_prediction": r, "full_logloss": lf,
                        "deleted_logloss": ld, "random_deleted_logloss": lr,
                        "delta_logloss_vs_full": ld-lf,
                        "random_delta_logloss_vs_full": lr-lf,
                        "delta_vs_random": ld-lr})
            paired = table("mil/ablation/outer_0/paired_patient_seed_logloss.tsv", rows)
            summary = {"groups": [{"group_id": gid, "status": "complete"}],
                       "paired_patient_seed_logloss_sha256": decision.sha256(paired)}
            class FakeMil:
                SEEDS = (11, 23, 47)
                patient_logloss = staticmethod(decision.logloss)
                @staticmethod
                def model_paths(root, fold, model, bag, seed, inner):
                    base = root / f"mil/cv/outer_{fold}/final/{model}/bag_{bag}/seed_{seed}"
                    return base / "prediction.tsv", base / "receipt.json"
                @staticmethod
                def valid_saved_prediction(path, expected_keys, labels, model, bag, seed, outer, inner):
                    value = pd.read_csv(path, sep="\t", dtype={"patient_key": str})
                    return value if set(value.patient_key) == set(expected_keys) else None
            with mock.patch.object(decision, "_mil_module", return_value=FakeMil):
                decision.validate_ablation_source_predictions(root, 0, pd.read_csv(paired, sep="\t"),
                    summary, {"selected_model": model, "selected_bag": bag})
                tampered = pd.read_csv(paired, sep="\t")
                tampered.loc[0, "full_prediction"] = 0.9
                tampered.to_csv(paired, sep="\t", index=False)
                summary["paired_patient_seed_logloss_sha256"] = decision.sha256(paired)
                self.assertEqual(summary["paired_patient_seed_logloss_sha256"], decision.sha256(paired))
                with self.assertRaisesRegex(ValueError, "full_prediction differs from source prediction"):
                    decision.validate_ablation_source_predictions(root, 0, tampered,
                        summary, {"selected_model": model, "selected_bag": bag})


class SensitivityGateTests(unittest.TestCase):
    def test_code_revision_binds_v7_to_prior_freezes_and_stage05(self):
        with tempfile.TemporaryDirectory() as temp:
            root = pathlib.Path(temp)
            audit = root / "audit"
            audit.mkdir()
            presence = {"groups/frozen_home_candidates.tsv": False}
            cfg = "c" * 64
            v5f, v5r = root / decision.V5_RULE_FREEZE, root / decision.V5_CODE_REVISION
            v5f.write_text(json.dumps({"rules_sha256": cfg, "code_sha256": "b" * 64,
                                      "output_presence_at_freeze": presence}))
            v5r.write_text(json.dumps({"status": "locked_preoutcome",
                "change_type": "remove full-Home top-five truncation before first association",
                "new_code_sha256": "b" * 64, "rules_sha256": cfg,
                "rule_freeze_sha256": decision.sha256(v5f)}))
            v6f, v6r = root / decision.V6_RULE_FREEZE, root / decision.V6_CODE_REVISION
            v6f.write_text(json.dumps({"rules_sha256": cfg, "code_sha256": "d" * 64,
                                       "previous_rule_freeze_sha256": decision.sha256(v5f),
                                       "output_presence_at_freeze": presence}))
            v6r.write_text(json.dumps({"status": "locked_preoutcome",
                "change_type": "align external projection receipt literal with frozen Stage 04 contract",
                "old_code_sha256": "b" * 64, "new_code_sha256": "d" * 64,
                "prior_code_revision_sha256": decision.sha256(v5r),
                "prior_rule_freeze_sha256": decision.sha256(v5f),
                "rule_freeze_sha256": decision.sha256(v6f)}))
            freeze = root / decision.RULE_FREEZE
            code_sha = decision.sha256(pathlib.Path(decision.__file__))
            mil_sha = decision.EXPECTED_MIL_CODE_SHA256
            freeze.write_text(json.dumps({"rules_sha256": cfg, "code_sha256": code_sha,
                "stage05_code_sha256": mil_sha,
                "previous_rule_freeze_sha256": decision.sha256(v6f),
                "all_outcome_outputs_absent": True,
                "output_presence_at_freeze": presence}))
            revision = root / decision.CODE_REVISION
            value = {"status": "locked_preoutcome",
                "change_type": "bind pooled OOF and source predictions before final family decision",
                "old_code_sha256": "d" * 64, "new_code_sha256": code_sha,
                "stage05_code_sha256": mil_sha,
                "prior_code_revision_sha256": decision.sha256(v6r),
                "prior_rule_freeze_sha256": decision.sha256(v6f),
                "rule_freeze_sha256": decision.sha256(freeze),
                "rules_sha256": cfg, "decision_thresholds_changed": False,
                "historical_freezes_modified": False,
                "output_presence_at_revision": presence}
            revision.write_text(json.dumps(value))
            self.assertEqual(decision.validate_code_revision(root, cfg), decision.sha256(revision))
            value["stage05_code_sha256"] = "e" * 64
            revision.write_text(json.dumps(value))
            with self.assertRaisesRegex(ValueError, "v7 code revision"):
                decision.validate_code_revision(root, cfg)


    def fixture(self, root):
        base = root / "groups/full_home"
        base.mkdir(parents=True)

        def write(path, content=b"fixture"):
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(content)

        def dump(path, data):
            write(path, (json.dumps(data) + "\n").encode())

        frozen = root / "groups/frozen_home_candidates.tsv"
        write(frozen, b"group_id\ng1\n")
        dump(frozen.with_suffix(".json"), {"candidate_sha256": decision.sha256(frozen)})
        for name in ("patient_group_burden.parquet", "group_membership.parquet",
                     "patient_nodes.parquet", "integrated_projection_receipt.json",
                     "burden_receipt.json", "association_all_recorded_home_exploratory.json"):
            write(base / name)
        write(base / "association_verified_blood.tsv", b"group_id\ng1\n")
        dump(base / "association_verified_blood.json", {"n_patients": 31})
        source = pathlib.Path(decision.__file__).with_name("04c_group_sensitivities.py")
        inputs = {
            "burden": base / "patient_group_burden.parquet",
            "membership": base / "group_membership.parquet",
            "nodes": base / "patient_nodes.parquet",
            "integrated_receipt": base / "integrated_projection_receipt.json",
            "burden_receipt": base / "burden_receipt.json",
            "frozen": frozen,
            "frozen_receipt": frozen.with_suffix(".json"),
            "primary_source": pathlib.Path(decision.__file__).with_name("04_groups_stats.py"),
        }
        for subset, n_patients in decision.SENSITIVITY_SUBSETS.items():
            eligible = base / f"eligible_groups_{subset}.tsv"
            write(eligible, b"group_id\ng1\n")
            association_receipt = base / f"association_{subset}.json"
            if not association_receipt.exists():
                dump(association_receipt, {"synthetic": True})
            plan_path = base / f"sensitivity_plan_{subset}.json"
            plan = {"status": "locked_label_blind", "version": 3,
                    "scope": "full_home", "subset": subset,
                    "source_sha256": decision.sha256(source),
                    "input_sha256": {name: decision.sha256(path) for name, path in
                                     (inputs | {"eligible": eligible,
                                                "association_receipt": association_receipt}).items()},
                    "n_patients": n_patients, "n_groups": 1,
                    "tests": sorted(decision.SENSITIVITY_METRICS),
                    "no_recorded_n_used_for_sensitivity_parameter_selection": True,
                    "rarefaction": "independent group-wise draws are not a joint repertoire subsample",
                    "rarefaction_depth_reads": 100, "repeats": 5}
            dump(plan_path, plan)
            dump(base / f"sensitivity_pilot_{subset}.json",
                 {"status": "PASS", "scope": "full_home", "subset": subset,
                  "plan_sha256": decision.sha256(plan_path)})
            patient = base / f"sensitivity_patient_burden_{subset}.parquet"
            tests = base / f"sensitivity_tests_{subset}.tsv"
            strata = base / f"sensitivity_batch_strata_{subset}.tsv"
            cmh = base / f"sensitivity_batch_cmh_{subset}.tsv"
            write(patient)
            pd.DataFrame([{"group_id": "g1", "metric": metric, "p": 0.8}
                          for metric in sorted(decision.SENSITIVITY_METRICS)]).to_csv(
                              tests, sep="\t", index=False)
            write(strata, b"group_id\tlibrary_batch\ng1\tb1\n")
            write(cmh, b"group_id\tcmh_status\ng1\tnot_estimable\n")
            dump(base / f"sensitivity_receipt_{subset}.json", {
                "status": "complete", "scope": "full_home", "subset": subset,
                "plan_sha256": decision.sha256(plan_path),
                "patient_table_sha256": decision.sha256(patient),
                "tests_sha256": decision.sha256(tests),
                "batch_strata_sha256": decision.sha256(strata),
                "batch_cmh_sha256": decision.sha256(cmh),
                "n_batch_strata_rows": 1, "n_batch_cmh_rows": 1,
                "n_patients": n_patients, "n_groups": 1,
                "n_patient_group_rows": n_patients, "n_tests": 4,
                "rarefaction_depth_reads": 100, "rarefaction_repeats": 5,
                "rarefaction_draw_coupling":
                    "independent exact group marginal draws; not a joint repertoire subsample",
                "cmh_p_is_nominal_only": True,
                "cmh_in_primary_pooled_BH_family": False,
                "pgen_in_primary_test_family": False,
                "n_documented_blood_included": 31,
                "n_material_unverified_included": 0 if subset == "verified_blood" else 83,
            })
        pgen_path = base / "sensitivity_pgen_receipt.json"
        dump(pgen_path, {
            "status": "STOP_Pgen_model_unavailable_ImportError",
            "scope": "full_home",
            "candidate_sha256": decision.sha256(frozen),
            "candidate_receipt_sha256": decision.sha256(frozen.with_suffix(".json")),
            "membership_sha256": decision.sha256(inputs["membership"]),
            "home_association_receipt_sha256": decision.sha256(
                base / "association_all_recorded_home_exploratory.json"),
            "burden_receipt_sha256": decision.sha256(inputs["burden_receipt"]),
            "integrated_projection_receipt_sha256": decision.sha256(inputs["integrated_receipt"]),
            "source_sha256": decision.sha256(source),
            "Pgen_used_for_primary_q": False,
            "Pgen_used_for_candidate_selection": False,
            "pgen_definition": "unconditioned CDR3 AA; not V/J-adjusted",
            "pgen_method": "Post-selection OLGA human TRB unconditioned CDR3 AA Pgen for frozen candidates; label-blind length-matched Home AA background is descriptive; no V/J-conditioned model",
            "unavailable_reason": "OLGA model files unavailable",
            "no_pgen_values_fabricated": True,
        })
        return frozen

    def test_valid_unavailable_pgen_receipt_keeps_sensitivity_descriptive(self):
        with tempfile.TemporaryDirectory() as temp:
            root = pathlib.Path(temp)
            frozen = self.fixture(root)
            stats = mock.Mock()
            with mock.patch.object(decision, "_group_stats_module", return_value=stats):
                summary = decision.validate_sensitivity_chain(root, frozen)
            stats._validate_association_receipt.assert_called_once_with(
                root, "full_home", "verified_blood")
            self.assertEqual(summary["subsets"]["verified_blood"]["n_patients"], 31)
            self.assertEqual(summary["pgen"]["status"], "STOP_Pgen_model_unavailable_ImportError")

    def test_stale_scope_or_missing_metric_cannot_complete(self):
        with tempfile.TemporaryDirectory() as temp:
            root = pathlib.Path(temp)
            frozen = self.fixture(root)
            receipt = root / "groups/full_home/sensitivity_receipt_verified_blood.json"
            value = json.loads(receipt.read_text())
            value["scope"] = "external"
            receipt.write_text(json.dumps(value))
            with mock.patch.object(decision, "_group_stats_module"):
                with self.assertRaisesRegex(ValueError, "Sensitivity result receipt invalid"):
                    decision.validate_sensitivity_chain(root, frozen)
            value["scope"] = "full_home"
            receipt.write_text(json.dumps(value))
            tests = root / "groups/full_home/sensitivity_tests_verified_blood.tsv"
            frame = pd.read_csv(tests, sep="\t")
            frame.loc[frame.metric.eq("rare_capture"), "metric"] = "read_fraction"
            frame.to_csv(tests, sep="\t", index=False)
            value["tests_sha256"] = decision.sha256(tests)
            receipt.write_text(json.dumps(value))
            with mock.patch.object(decision, "_group_stats_module"):
                with self.assertRaisesRegex(ValueError, "Sensitivity tests incomplete"):
                    decision.validate_sensitivity_chain(root, frozen)

    def test_pgen_unavailability_requires_reason_and_current_source(self):
        with tempfile.TemporaryDirectory() as temp:
            root = pathlib.Path(temp)
            frozen = self.fixture(root)
            pgen = root / "groups/full_home/sensitivity_pgen_receipt.json"
            value = json.loads(pgen.read_text())
            value.pop("unavailable_reason")
            pgen.write_text(json.dumps(value))
            with mock.patch.object(decision, "_group_stats_module"):
                with self.assertRaisesRegex(ValueError, "explicit unavailable reason"):
                    decision.validate_sensitivity_chain(root, frozen)
            value["unavailable_reason"] = "missing model"
            value["source_sha256"] = "stale"
            pgen.write_text(json.dumps(value))
            with mock.patch.object(decision, "_group_stats_module"):
                with self.assertRaisesRegex(ValueError, "Pgen post-selection provenance"):
                    decision.validate_sensitivity_chain(root, frozen)

    def test_completed_pgen_requires_hashed_outputs(self):
        with tempfile.TemporaryDirectory() as temp:
            root = pathlib.Path(temp)
            frozen = self.fixture(root)
            base = root / "groups/full_home"
            pgen_path = base / "sensitivity_pgen_receipt.json"
            pgen = json.loads(pgen_path.read_text())
            pgen["status"] = "complete"
            pgen.pop("unavailable_reason")
            for field, name in (("sequence_sha256", "sensitivity_pgen_sequences.parquet"),
                                ("background_sha256", "sensitivity_pgen_background.parquet"),
                                ("group_sha256", "sensitivity_pgen_groups.tsv"),
                                ("background_source_sha256", "clone_nodes.parquet")):
                path = base / name
                path.write_bytes(name.encode())
                pgen[field] = decision.sha256(path)
            pgen_path.write_text(json.dumps(pgen))
            with mock.patch.object(decision, "_group_stats_module"):
                self.assertEqual(decision.validate_sensitivity_chain(root, frozen)["pgen"]["status"],
                                 "complete")
            (base / "sensitivity_pgen_groups.tsv").write_bytes(b"changed")
            with mock.patch.object(decision, "_group_stats_module"):
                with self.assertRaisesRegex(ValueError, "Pgen output/source hash invalid"):
                    decision.validate_sensitivity_chain(root, frozen)

    def test_missing_batch_receipt_keeps_final_decision_pending(self):
        with tempfile.TemporaryDirectory() as temp:
            root = pathlib.Path(temp)
            self.fixture(root)
            missing = root / "groups/full_home/sensitivity_batch_cmh_verified_blood.tsv"
            missing.unlink()
            rules = root / "config/final_candidate_rules.json"
            rules.parent.mkdir(parents=True)
            rules.write_text('{"frozen_before_outcome_run":true,"max_families":null,"candidate_scope":"all_home_primary_eligible"}')
            freeze = root / decision.RULE_FREEZE
            freeze.parent.mkdir(parents=True)
            freeze.write_text(json.dumps({"rules_sha256": decision.sha256(rules),
                                          "all_outcome_outputs_absent": True}))
            frozen = root / "groups/frozen_home_candidates.tsv"
            pd.DataFrame([{"group_id": "g1", "method": "tcrdist3",
                           "projection_type": "tcrdist3_train_seed_full_projection",
                           "family_definition": "family", "home_q_pooled": 0.01,
                           "home_adjusted_presence_or": 2.0,
                           "home_presence_delta": 0.1}]).to_csv(frozen, sep="\t", index=False)
            with mock.patch.object(decision, "verify_receipts"),\
                 mock.patch.object(decision, "validate_code_revision"):
                result = decision.run(root)
            self.assertEqual(result["status"], "PENDING")
            self.assertFalse(result["candidate_claim"])
            self.assertIn(str(missing), result["missing"])


class SyntheticIntegrationTests(unittest.TestCase):
    def test_complete_candidate_uses_unique_outer_test_patients(self):
        import hashlib

        def dump(path, obj):
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(obj) + "\n")

        def sha(path):
            return hashlib.sha256(path.read_bytes()).hexdigest()

        def table(path, rows):
            path.parent.mkdir(parents=True, exist_ok=True)
            pd.DataFrame(rows).to_csv(path, sep="\t", index=False)

        with tempfile.TemporaryDirectory() as temp:
            root = pathlib.Path(temp)
            rules = {
                "frozen_before_outcome_run": True, "max_families": None,
                "candidate_scope": "all_home_primary_eligible",
                "home": {"minimum_carriers": 10, "adjusted_presence_q_strictly_below": 0.05,
                         "adjusted_presence_or_above": 1, "presence_delta_above": 0},
                "external": {"minimum_carriers": 5},
                "alignment": {"minimum_matching_outer_folds": 3},
                "ablation": {"random_seed": 1, "patient_bootstrap_draws": 100,
                             "patient_signflip_draws": 1000,
                             "minimum_positive_fold_fraction": 0.8,
                             "minimum_positive_seed_effects": 2,
                             "primary_mean_logloss_delta_above": 0,
                             "primary_patient_bootstrap_lower_above": 0,
                             "matched_random_mean_difference_above": 0,
                             "pooled_one_sided_signflip_bh_q_strictly_below": 0.05,
                             "minimum_safe_test_carriers_pooled": 5,
                             "minimum_safe_test_carriers_per_matched_fold": 1,
                             "minimum_safe_deletion_coverage_fraction": 0.8,
                             "minimum_safe_test_carrier_fraction_each_matched_fold": 0.8},
                "final_interpretation": "recorded N+ candidate only"}
            rules_path = root / "config/final_candidate_rules.json"
            dump(rules_path, rules)
            dump(root / decision.RULE_FREEZE,
                 {"rules_sha256": sha(rules_path), "all_outcome_outputs_absent": True})
            frozen = root / "groups/frozen_home_candidates.tsv"
            table(frozen, [
                {"group_id": gid, "method": "tcrdist3",
                 "projection_type": "tcrdist3_train_seed_full_projection",
                 "family_definition": "training seed and full projection",
                 "home_q_pooled": q, "home_adjusted_presence_or": 3.0,
                 "home_presence_delta": 0.2,
                 "home_read_fraction_delta_median": 0.002}
                for gid, q in (("g1", 0.001), ("g2", 0.002))])
            table(root / "groups/full_home/association_all_recorded_home_exploratory.tsv",
                  [{"group_id": gid, "n_present": 20,
                    "adjusted_presence_q_pooled": q,
                    "adjusted_presence_or": 3.0, "presence_delta": 0.2,
                    "adjusted_model_status": "ok"}
                   for gid, q in (("g1", 0.001), ("g2", 0.002))])
            source_association = root / "groups/full_home/association_all_recorded_home_exploratory.tsv"
            membership = root / "groups/full_home/group_membership.parquet"
            pd.DataFrame([
                {"group_id": gid, "aa": aa,
                 "projection_type": "tcrdist3_train_seed_full_projection",
                 "v_gene": "TRBV7-2*01", "j_gene": "TRBJ2-5*01",
                 "seq_idx": seq_idx}
                for gid, aa, seq_idx in (
                    ("g1", _private_literal('private_8b5efd300a8432fa'), 4), ("g2", _private_literal('private_3d91b232aa13ed28'), 5))
            ]).to_parquet(membership, index=False)
            dump(frozen.with_suffix(".json"), {
                "candidate_sha256": sha(frozen), "external_n_read": False,
                "source_subset": "all_recorded_home_exploratory",
                "source_association_sha256": sha(source_association),
                "membership_sha256": sha(membership)})
            manifest = root / "config/cohort_samples.tsv"
            table(manifest, [{"cohort": "home", "sample_id": _private_literal('private_fb1ebcd36c9eb1af'), "patient_key": _private_literal('private_fb1ebcd36c9eb1af'),
                              "recorded_n_binary": "0", "identity_basis": "Home_record"}] +
                  [{"cohort": "external", "sample_id": f"ext-{i:03d}",
                    "patient_key": f"ext-{i:03d}", "recorded_n_binary": str(i % 2),
                    "identity_basis": "BioSample_proxy_unverified"} for i in range(165)])
            stats = decision._group_stats_module()
            identity_sha, home_label_sha = stats._manifest_fingerprints(root)
            patients, label_sha, n_positive, n_negative = decision._external_labels(root)
            cross = root / "groups/cross_fold_alignment.tsv"
            table(cross, [{"canonical_id": "cf1", "outer_fold": f, "group_id": f"fg{f}",
                           "method": "tcrdist3"} for f in range(4)])
            alignment = root / "groups/full_home_to_fold_alignment.tsv"
            table(alignment, [
                {"full_home_group_id": "g1", "canonical_id": "cf1",
                 "alignment_pass": True, "n_matching_folds": 3,
                 "method": "tcrdist3"},
                {"full_home_group_id": "g2", "canonical_id": "",
                 "alignment_pass": False, "n_matching_folds": 0,
                 "method": "tcrdist3"}])
            match = root / "groups/full_home_fold_member_matches.tsv"
            table(match, [{"full_home_group_id": "g1", "canonical_id": "cf1",
                           "outer_fold": f, "group_id": f"fg{f}", "match_pass": f < 3}
                          for f in range(4)])
            dump(alignment.with_suffix(".json"), {
                "frozen_candidates_sha256": sha(frozen),
                "cross_fold_alignment_sha256": sha(cross),
                "full_home_fold_member_matches_sha256": sha(match),
                "source_association_sha256": sha(source_association),
                "output_sha256": sha(alignment)})
            burden = root / "groups/full_home/patient_group_burden_external.parquet"
            burden.write_bytes(b"synthetic burden hash fixture")
            home_burden = root / "groups/full_home/patient_group_burden.parquet"
            home_burden.write_bytes(b"synthetic Home burden hash fixture")
            home_burden_receipt = root / "groups/full_home/burden_receipt.json"
            dump(home_burden_receipt, {"qualified_group_ids_sha256": stats._patient_hash({"g1", "g2"})})
            external_burden_receipt = root / "groups/full_home/burden_receipt_external.json"
            dump(external_burden_receipt, {"synthetic": True})
            projection = root / "groups/full_home/external_projection_receipt.json"
            dump(projection, {"status": "complete", "external_n_read": False,
                              "frozen_candidates_sha256": sha(frozen),
                              "frozen_candidate_receipt_sha256": sha(frozen.with_suffix(".json")),
                              "burden_sha256": sha(burden),
                              "membership_sha256": sha(membership),
                              "home_burden_sha256": sha(home_burden),
                              "home_burden_receipt_sha256": sha(home_burden_receipt),
                              "home_qualified_group_ids_sha256": stats._patient_hash({"g1", "g2"}),
                              "frozen_group_ids_sha256": stats._patient_hash({"g1", "g2"}),
                              "manifest_identity_sha256": identity_sha,
                              "home_split_label_sha256": home_label_sha,
                              "target_patient_sha256": stats._patient_hash(patients),
                              "target_proxy_count": 165, "burden_rows": 330,
                              "external_burden_receipt_sha256": sha(external_burden_receipt),
                              "external_identity": "BioSample_proxy_unverified",
                              "projection": "complete frozen-rule AA/V/J projection over all external QC clones",
                              "completed_utc": "2026-09-29T00:00:00+00:00"})
            direction = root / "groups/full_home/external_direction.tsv"
            table(direction, [
                {"group_id": "g1", "n_carriers_external": 8,
                 "evaluable": True, "direction_positive": True,
                 "presence_delta": 0.1},
                {"group_id": "g2", "n_carriers_external": 7,
                 "evaluable": True, "direction_positive": False,
                 "presence_delta": -0.1}])
            dump(direction.with_suffix(".json"), {
                "projection_receipt_sha256": sha(projection),
                "frozen_candidates_sha256": sha(frozen),
                "projection_completed_utc": "2026-09-29T00:00:00+00:00",
                "external_direction_tsv_sha256": sha(direction),
                "external_label_sha256": label_sha,
                "external_label_canonicalization":
                    "sorted unique [patient_key,recorded_n_binary] pairs; compact UTF-8 JSON",
                "n_external_scored_patients": 165,
                "n_positive_proxy": n_positive, "n_negative_proxy": n_negative,
                "external_identity": "BioSample_proxy_unverified",
                "n_frozen_groups": 2, "n_scored_groups": 2})
            mil_rules = root / "config/mil_decision_rules.json"
            dump(mil_rules, {"version": 1})
            oof = root / _private_literal('private_4b09056568ab6f20')
            table(oof, [{"patient_key": "synthetic", "prediction": 0.5}])
            dump(root / "mil/ablation/gate.json", {
                "status": "pass", "decision_rules_sha256": sha(mil_rules),
                "oof_sha256": sha(oof)})
            fold_rows = []
            for i in range(114):
                f = i % 5
                y = i % 2
                fold_rows.append({"patient_key": f"p{i:03d}",
                                  "outer_fold": f, "partition": "test",
                                  "recorded_n_binary": y})
            table(root / "folds/folds.tsv", fold_rows)
            for f in range(3):
                rows = []
                for patient in [r for r in fold_rows if r["outer_fold"] == f]:
                    y = patient["recorded_n_binary"]
                    for seed in (11, 23, 47):
                        rows.append({"outer_fold": f, "group_id": f"fg{f}",
                                     "patient_key": patient["patient_key"],
                                     "label": y, "seed": seed,
                                     "full_prediction": 0.8 if y else 0.2,
                                     "deleted_prediction": 0.6 if y else 0.4,
                                     "random_deleted_prediction": 0.79 if y else 0.21})
                paired = root / "mil/ablation" / f"outer_{f}" / "paired_patient_seed_logloss.tsv"
                table(paired, rows)
                selection = root / "mil/cv" / f"outer_{f}" / "selection.json"
                dump(selection, {"synthetic": True})
                dump(paired.parent / "summary.json", {
                    "status": "complete", "oof_sha256": sha(oof),
                    "fold_selection_sha256": sha(selection),
                    "paired_patient_seed_logloss_sha256": sha(paired),
                    "decision_rules_sha256": sha(mil_rules),
                    "groups": [{"group_id": f"fg{f}", "status": "complete",
                                "test_carriers_safe": 8, "test_carriers_exact_vj": 8,
                                "safe_deletion_coverage_fraction": 1.0}]})
            # Full Home source-chain validation is Stage 04's contract; this
            # compact integration fixture isolates Stage 06's decision gates.
            home_validator = mock.patch.object(decision, "_validate_home_chain")
            alignment_validator = mock.patch.object(decision, "validate_alignment_training_reference")
            code_revision_validator = mock.patch.object(decision, "validate_code_revision",
                return_value="synthetic-revision-sha")
            self.addCleanup(home_validator.stop)
            self.addCleanup(alignment_validator.stop)
            self.addCleanup(code_revision_validator.stop)
            home_validator.start()
            alignment_validator.start()
            revision_check = code_revision_validator.start()
            oof_validator = mock.patch.object(decision, "validate_mil_oof_chain")
            ablation_validator = mock.patch.object(decision, "validate_ablation_source_predictions")
            self.addCleanup(oof_validator.stop)
            self.addCleanup(ablation_validator.stop)
            oof_validator.start()
            ablation_validator.start()
            sensitivity_paths = mock.patch.object(decision, "sensitivity_required_paths", return_value=[])
            sensitivity_validator = mock.patch.object(decision, "validate_sensitivity_chain",
                return_value={"pgen": {"status": "complete"}, "subsets": {}})
            self.addCleanup(sensitivity_paths.stop)
            self.addCleanup(sensitivity_validator.stop)
            sensitivity_paths.start()
            sensitivity_validator.start()
            result = decision.run(root)
            revision_check.assert_called_with(root, sha(rules_path))
            self.assertEqual(result["status"], "COMPLETE_candidates")
            self.assertEqual(len(result["candidate_cards"]), 1)
            self.assertTrue(pathlib.Path(result["candidate_cards"][0]).exists())
            self.assertTrue(result["families"][0]["final_candidate"])
            self.assertEqual(result["families"][0]["ablation"]["n_unique_test_patients"], 69)
            self.assertEqual(result["ablation_bh_family_count"], 2)
            self.assertEqual(len(result["families"]), 2)
            self.assertFalse(result["families"][1]["alignment_pass"])
            self.assertEqual(result["families"][1]["pooled_ablation_p"], 1.0)
            self.assertEqual(result["families"][1]["pooled_ablation_q"], 1.0)
            self.assertAlmostEqual(
                result["families"][0]["pooled_ablation_q"],
                2 * result["families"][0]["pooled_ablation_p"])

            original_projection = projection.read_bytes()
            projection_receipt = decision.json_read(projection)
            projection_receipt["projection"] = "exact_aa_v_gene_j_gene_only"
            dump(projection, projection_receipt)
            with self.assertRaisesRegex(ValueError, "not projected label-blind"):
                decision.run(root)
            projection.write_bytes(original_projection)

            projection_receipt = decision.json_read(projection)
            projection_receipt["external_n_read"] = True
            dump(projection, projection_receipt)
            with self.assertRaisesRegex(ValueError, "not projected label-blind"):
                decision.run(root)
            projection.write_bytes(original_projection)

            gate_path = root / "mil/ablation/gate.json"
            original_gate = gate_path.read_bytes()
            gate_receipt = decision.json_read(gate_path)
            gate_receipt["oof_sha256"] = "wrong"
            dump(gate_path, gate_receipt)
            with self.assertRaisesRegex(ValueError, "MIL gate rule/OOF provenance mismatch"):
                decision.run(root)
            gate_path.write_bytes(original_gate)

            gate_receipt["oof_sha256"] = sha(oof)
            gate_receipt["status"] = "blocked"
            dump(gate_path, gate_receipt)
            blocked = decision.run(root)
            self.assertEqual(blocked["status"], "STOP_no_stable_model_gain")
            self.assertFalse(blocked["candidate_claim"])
            gate_path.write_bytes(original_gate)

            paired0 = root / "mil/ablation/outer_0/paired_patient_seed_logloss.tsv"
            original_paired0 = paired0.read_bytes()
            paired0.write_bytes(original_paired0 + b"\n")
            with self.assertRaisesRegex(ValueError, "Fold 0 ablation provenance mismatch"):
                decision.run(root)
            paired0.write_bytes(original_paired0)

            original_direction = direction.read_bytes()
            direction.write_bytes(original_direction.replace(b"0.1", b"0.9"))
            with self.assertRaisesRegex(ValueError, "External scoring provenance mismatch"):
                decision.run(root)
            direction.write_bytes(original_direction)
            current_manifest = pd.read_csv(manifest, sep="\t", dtype=str)
            current_manifest.loc[current_manifest.patient_key.eq("ext-000"),
                                 "recorded_n_binary"] = "1"
            current_manifest.to_csv(manifest, sep="\t", index=False)
            with self.assertRaisesRegex(ValueError, "External scoring provenance mismatch"):
                decision.run(root)
            current_manifest.loc[current_manifest.patient_key.eq("ext-000"),
                                 "recorded_n_binary"] = "0"
            current_manifest.to_csv(manifest, sep="\t", index=False)

            with mock.patch.object(decision, "write_candidate_cards",
                                   side_effect=RuntimeError("card export failed")):
                with self.assertRaisesRegex(RuntimeError, "card export failed"):
                    decision.run(root)
            pending = decision.json_read(root / "final/decision.json")
            self.assertEqual(pending["status"], "PENDING_card_export")
            self.assertFalse(pending["candidate_claim"])
            summary_path = root / "mil/ablation/outer_0/summary.json"
            summary = decision.json_read(summary_path)
            summary["groups"][0]["test_carriers_safe"] = 1
            dump(summary_path, summary)
            low_coverage = decision.run(root)
            self.assertEqual(low_coverage["status"], "COMPLETE_no_candidate")
            self.assertFalse(low_coverage["families"][0]["ablation_pass"])
            summary["groups"][0]["test_carriers_safe"] = 8
            summary["groups"][0]["status"] = "skipped"
            summary["groups"][0]["reason"] = "No safe AA removals"
            paired0 = root / "mil/ablation/outer_0/paired_patient_seed_logloss.tsv"
            pd.read_csv(paired0, sep="\t").iloc[0:0].to_csv(
                paired0, sep="\t", index=False)
            summary["paired_patient_seed_logloss_sha256"] = sha(paired0)
            dump(summary_path, summary)
            skipped = decision.run(root)
            self.assertEqual(skipped["status"], "COMPLETE_no_candidate")
            self.assertEqual(skipped["families"][0]["reason"], "ablation_unassessable")
            self.assertEqual(skipped["families"][0]["pooled_ablation_p"], 1.0)
            pd.DataFrame([{"group_id": "g1", "aa": "DIFFERENT",
                           "projection_type": "tcrdist3_train_seed_full_projection",
                           "v_gene": "TRBV7-2*01", "j_gene": "TRBJ2-5*01",
                           "seq_idx": 4}]).to_parquet(membership, index=False)
            with self.assertRaisesRegex(ValueError, "membership changed"):
                decision.run(root)


if __name__ == "__main__":
    unittest.main()
