from release_runtime import private_literal as _private_literal, release_path as _release_path
import importlib.util
from math import comb
from pathlib import Path
from types import SimpleNamespace
import tempfile

import pytest

import numpy as np
import pandas as pd
from scipy.stats import chi2

SOURCE = Path(__file__).resolve().parents[1] / "04c_group_sensitivities.py"
SPEC = importlib.util.spec_from_file_location("group_sens", SOURCE)
SENS = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(SENS)


def test_capture_probability_matches_exact_hypergeometric():
    total = np.array([10, 10, 10, 5], dtype=np.int64)
    group = np.array([0, 2, 10, 1], dtype=np.int64)
    got = SENS.capture_probability(total, group, depth=3)
    expected = np.array([0.0, 1-comb(8, 3)/comb(10, 3), 1.0, 1-comb(4, 3)/comb(5, 3)])
    assert np.allclose(got, expected, rtol=1e-12, atol=1e-12)


def test_rarefaction_is_deterministic_and_keeps_patient_unit():
    burden = pd.DataFrame({
        "group_id": ["g1", "g1", "g2", "g2"],
        "patient_key": ["p1", "p2", "p1", "p2"],
        "total_reads": [10, 10, 10, 10],
        "group_reads": [2, 0, 5, 1],
    })
    left = SENS.rarefy(burden, 3, 1234, 10, "full_home", "all_recorded_home_exploratory")
    right = SENS.rarefy(burden, 3, 1234, 10, "full_home", "all_recorded_home_exploratory")
    pd.testing.assert_frame_equal(left, right)
    assert len(left) == 4
    assert left.loc[(left.group_id == "g1") & (left.patient_key == "p2"), "rare_presence_mean"].iat[0] == 0
    assert left.rare_read_fraction_mean.between(0, 1).all()


@pytest.mark.parametrize("receipt", [
    {"status": "complete_full_scope", "projection_population": "home", "scope": "full_home"},
    {"status": "complete_home_scope", "projection_population": "external", "scope": "full_home"},
    {"status": "complete_home_scope", "projection_population": "home", "scope": "outer_0"},
    {"status": "complete_home_scope", "scope": "full_home"},
])
def test_integrated_receipt_rejects_stale_or_non_home_scope(tmp_path, receipt):
    group = tmp_path / "groups/full_home"
    group.mkdir(parents=True)
    SENS.save_json(group / "integrated_projection_receipt.json", receipt)
    with pytest.raises(RuntimeError, match="STOP_incomplete_dictionary"):
        SENS._receipt(tmp_path, "full_home")


def test_locked_plan_pilot_run_and_missing_input_stop(monkeypatch):
    # All temporary data live inside the requested analysis directory.
    with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent) as folder:
        root = Path(folder)
        cfg = root / "config"
        group = root / "groups/full_home"
        cfg.mkdir()
        group.mkdir(parents=True)
        SENS.save_json(group / "integrated_projection_receipt.json", {
            "status": "complete_home_scope", "projection_population": "home", "scope": "full_home"})
        frozen = root / "groups/frozen_home_candidates.tsv"
        frozen.write_text("group_id\ng1\n")
        SENS.save_json(frozen.with_suffix(".json"), {"candidate_sha256": SENS.sha256(frozen)})
        for name in ("burden_receipt.json", "association_all_recorded_home_exploratory.json"):
            SENS.save_json(group / name, {})
        # This test isolates 04c computations; mutation tests below exercise
        # the real Stage 04 frozen-candidate validator.
        monkeypatch.setattr(SENS, "_groups_stats", lambda: SimpleNamespace(
            _validate_frozen_file=lambda *args: None))
        patient_rows, node_rows, labels, clinical = [], [], [], []
        membership = pd.DataFrame({
            "scope": ["full_home", "full_home"],
            "method": ["gliph2", "tcrdist"],
            "group_id": ["g1", "g2"],
            "aa": [_private_literal('private_1a4bf3d44eaceb1d'), _private_literal('private_2cb2e66bf01b6e4d')],
            "v_gene": ["TRBV1", "TRBV2"],
            "j_gene": ["TRBJ1-1", "TRBJ2-1"],
            "seq_idx": [0, 1],
        })
        membership.to_parquet(group / "group_membership.parquet", index=False)
        for i in range(40):
            key = f"p{i:02}"
            y = i % 2
            labels.append({"cohort": "home", "patient_key": key, "recorded_n_binary": y})
            clinical.append({"patient_key": key, "age_years": 50+i, "sex": "f" if i % 3 else "m",
                             "library_batch": "A" if i < 10 else "B"})
            has1 = i % 3 != 0
            has2 = i % 4 != 0
            node_rows.append({"patient_key": key, "aa": _private_literal('private_1a4bf3d44eaceb1d') if has1 else _private_literal('private_eee7123d7df78359'),
                              "v_gene": "TRBV1" if has1 else "TRBV3", "j_gene": "TRBJ1-1"})
            node_rows.append({"patient_key": key, "aa": _private_literal('private_2cb2e66bf01b6e4d') if has2 else _private_literal('private_32619d760ab0d399'),
                              "v_gene": "TRBV2" if has2 else "TRBV4", "j_gene": "TRBJ2-1"})
            for gid, present in (("g1", has1), ("g2", has2)):
                patient_rows.append({
                    "scope": "full_home", "patient_key": key, "group_id": gid,
                    "group_reads": 30 if present else 0, "group_unique_aa": int(present),
                    "group_unique_aa_vj": int(present), "total_reads": 100,
                    "total_unique_aa": 2, "presence": int(present),
                    "unique_aa_fraction": 0.5 if present else 0,
                    "read_fraction": 0.3 if present else 0,
                })
        pd.DataFrame(patient_rows).to_parquet(group / "patient_group_burden.parquet", index=False)
        pd.DataFrame(node_rows).to_parquet(group / "patient_nodes.parquet", index=False)
        pd.DataFrame(labels).to_csv(cfg / "cohort_samples.tsv", sep="\t", index=False)
        pd.DataFrame(clinical).to_csv(cfg / "home_clinical_covariates.tsv", sep="\t", index=False)
        pd.DataFrame({"patient_key": [r["patient_key"] for r in labels],
                      "documented_blood": ["yes"] * 40}).to_csv(
                          cfg / "documented_blood31.tsv", sep="\t", index=False)
        pd.DataFrame({"group_id": ["g1", "g2"], "n_present": [26, 30], "n_patients": [40, 40]}).to_csv(
            group / "eligible_groups_all_recorded_home_exploratory.tsv", sep="\t", index=False)
        plan = SENS.plan(root, "full_home", "all_recorded_home_exploratory")
        assert SENS.json.loads(plan.read_text())["home_n_read_only_for_upstream_provenance_validation"] is True
        assert SENS.json.loads(plan.read_text())["no_recorded_n_used_for_sensitivity_parameter_selection"] is True
        assert SENS.json.loads(plan.read_text())["rarefaction_depth_reads"] == 100
        assert SENS.json.loads(plan.read_text())["batch_sensitivity"]["multiple_testing"].startswith(
            "not in primary pooled BH family")
        pilot = SENS.pilot(root, "full_home", "all_recorded_home_exploratory")
        assert SENS.json.loads(pilot.read_text())["status"] == "PASS"
        result = SENS.run(root, "full_home", "all_recorded_home_exploratory")
        test_table = pd.read_csv(result, sep="\t")
        assert len(test_table) == 8
        assert set(test_table.metric) == {name for name, _ in SENS.TESTS}
        assert test_table.q_sensitivity_pooled.between(0, 1).all()
        receipt = SENS.json.loads((group / "sensitivity_receipt_all_recorded_home_exploratory.json").read_text())
        assert receipt["unit"] == "patient"
        assert receipt["n_tests"] == 8
        strata_path = group / "sensitivity_batch_strata_all_recorded_home_exploratory.tsv"
        cmh_path = group / "sensitivity_batch_cmh_all_recorded_home_exploratory.tsv"
        strata = pd.read_csv(strata_path, sep="\t")
        cmh = pd.read_csv(cmh_path, sep="\t")
        assert len(strata) == 4 and len(cmh) == 2
        assert set(strata.library_batch) == {"A", "B"}
        assert "q_sensitivity_pooled" not in cmh
        assert receipt["batch_strata_sha256"] == SENS.sha256(strata_path)
        assert receipt["batch_cmh_sha256"] == SENS.sha256(cmh_path)
        assert receipt["cmh_in_primary_pooled_BH_family"] is False
        assert receipt["cmh_used_for_candidate_selection"] is False
        assert receipt["cmh_is_not_the_main_adjusted_model"] is True
        frozen_receipt = frozen.with_suffix(".json")
        original_receipt = frozen_receipt.read_text()
        frozen_receipt.write_text(original_receipt + "\n")
        with pytest.raises(RuntimeError, match="STOP_input_changed_after_sensitivity_plan: frozen_receipt"):
            SENS.pilot(root, "full_home", "all_recorded_home_exploratory")
        frozen_receipt.write_text(original_receipt)
        (group / "integrated_projection_receipt.json").unlink()
        try:
            SENS.plan(root, "full_home", "all_recorded_home_exploratory")
        except RuntimeError as exc:
            assert "STOP_missing_complete_dictionary" in str(exc)
        else:
            raise AssertionError("Missing integrated projection must stop")


def test_length_matched_pgen_background_is_label_blind_and_deterministic():
    with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent) as folder:
        path = Path(folder) / "clone_nodes.parquet"
        pd.DataFrame({"aa": [_private_literal('private_1a4bf3d44eaceb1d'), _private_literal('private_1a4bf3d44eaceb1d'), _private_literal('private_2cb2e66bf01b6e4d'),
                             _private_literal('private_32619d760ab0d399'), _private_literal('private_39bbb48b2ee40966'), _private_literal('private_badbf6e7b446feb9')]}).to_parquet(
                                 path, index=False)
        first = SENS._length_matched_background(path, {_private_literal('private_1a4bf3d44eaceb1d')}, {10, 11, 14}, per_length=1)
        second = SENS._length_matched_background(path, {_private_literal('private_1a4bf3d44eaceb1d')}, {10, 11, 14}, per_length=1)
        assert first == second
        assert first == [_private_literal('private_2cb2e66bf01b6e4d'), _private_literal('private_32619d760ab0d399')]


def test_pgen_unavailable_receipt_keeps_post_selection_provenance(tmp_path, monkeypatch):
    root = tmp_path
    group = root / "groups/full_home"
    group.mkdir(parents=True)
    frozen = root / "groups/frozen_home_candidates.tsv"
    frozen.write_text("group_id\ng1\n")
    SENS.save_json(frozen.with_suffix(".json"), {"candidate_sha256": SENS.sha256(frozen)})
    pd.DataFrame({"group_id": ["g1"], "aa": [_private_literal('private_32619d760ab0d399')]}).to_parquet(
        group / "group_membership.parquet", index=False)
    for name in ("association_all_recorded_home_exploratory.json", "burden_receipt.json",
                 "integrated_projection_receipt.json"):
        SENS.save_json(group / name, {"synthetic": True})
    monkeypatch.setattr(SENS, "_validate_upstream", lambda *args: None)

    def unavailable_model():
        raise ImportError("OLGA model files absent")

    monkeypatch.setattr(SENS, "_olga_model", unavailable_model)
    receipt_path = SENS.pgen_candidates(root)
    receipt = SENS.json.loads(receipt_path.read_text())
    assert receipt["status"] == "STOP_Pgen_model_unavailable_ImportError"
    assert "OLGA model files absent" in receipt["unavailable_reason"]
    assert receipt["candidate_sha256"] == SENS.sha256(frozen)
    assert receipt["source_sha256"] == SENS.sha256(SOURCE)
    assert receipt["scope"] == "full_home"
    assert receipt["Pgen_used_for_primary_q"] is False
    assert receipt["Pgen_used_for_candidate_selection"] is False
    assert "no V/J-conditioned model" in receipt["pgen_method"]
    assert not (group / "sensitivity_pgen_groups.tsv").exists()


def _frozen_chain_fixture(root, monkeypatch):
    """Minimal hash-linked freeze; retain Stage 04's real frozen-file validator."""
    group = root / "groups/full_home"
    cfg = root / "config"
    group.mkdir(parents=True)
    cfg.mkdir()
    SENS.save_json(group / "integrated_projection_receipt.json", {
        "status": "complete_home_scope", "projection_population": "home", "scope": "full_home"})
    (group / "group_membership.parquet").write_bytes(b"membership")
    (group / "patient_group_burden.parquet").write_bytes(b"burden")
    SENS.save_json(group / "burden_receipt.json", {"status": "complete"})
    (cfg / "home_clinical_covariates.tsv").write_text("patient_key\n")
    source = group / "association_all_recorded_home_exploratory.tsv"
    association = pd.DataFrame([{
        "group_id": "g1", "adjusted_model_status": "ok",
        "adjusted_presence_q_pooled": 0.01, "adjusted_presence_or": 2.0,
        "presence_delta": 0.1,
    }])
    association.to_csv(source, sep="\t", index=False)
    SENS.save_json(source.with_suffix(".json"), {"status": "complete"})
    blood_source = group / "association_verified_blood.tsv"
    association.to_csv(blood_source, sep="\t", index=False)
    SENS.save_json(blood_source.with_suffix(".json"), {"status": "complete"})
    eligible = group / "eligible_groups_all_recorded_home_exploratory.tsv"
    eligible.write_text("group_id\ng1\n")
    frozen = root / "groups/frozen_home_candidates.tsv"
    pd.DataFrame([{"group_id": "g1", "home_presence_delta": 0.1}]).to_csv(
        frozen, sep="\t", index=False)
    home_eligible = root / "groups/eligible_home_candidates.tsv"
    home_eligible.write_bytes(frozen.read_bytes())
    SENS.save_json(frozen.with_suffix(".json"), {
        "status": "complete",
        "candidate_sha256": SENS.sha256(frozen),
        "q_strictly_less_than": 0.05, "max_frozen_candidates": None,
        "candidate_truncation": False, "candidate_policy": "all_eligible_home_primary_groups",
        "eligible_home_candidates_sha256": SENS.sha256(home_eligible),
        "n_eligible_home_candidates": 1,
        "direction_rule": "Home adjusted OR>1 and Home patient-presence delta>0; external patient-presence delta>0 with >=5 carriers",
        "source_subset": "all_recorded_home_exploratory",
        "source_association": str(source),
        "source_association_sha256": SENS.sha256(source),
        "source_association_receipt_sha256": SENS.sha256(source.with_suffix(".json")),
        "source_eligible_groups_sha256": SENS.sha256(eligible),
        "home_burden_sha256": SENS.sha256(group / "patient_group_burden.parquet"),
        "home_burden_receipt_sha256": SENS.sha256(group / "burden_receipt.json"),
        "membership_sha256": SENS.sha256(group / "group_membership.parquet"),
        "manifest_identity_sha256": "identity",
        "home_split_label_sha256": "home-labels",
        "clinical_covariates_sha256": SENS.sha256(cfg / "home_clinical_covariates.tsv"),
        "documented_blood_sensitivity_sha256": SENS.sha256(blood_source),
        "documented_blood_sensitivity_receipt_sha256": SENS.sha256(blood_source.with_suffix(".json")),
        "n_candidates": 1,
    })
    primary = SENS._groups_stats()
    monkeypatch.setattr(primary, "_manifest_identity_sha256", lambda _: "identity")
    monkeypatch.setattr(primary, "_home_split_label_sha256", lambda _: "home-labels")
    return primary, association, source, frozen


def test_plan_rejects_mutated_home_association_receipt(monkeypatch):
    with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent) as folder:
        root = Path(folder)
        primary, association, source, frozen = _frozen_chain_fixture(root, monkeypatch)
        checked = []

        def checked_association(_, scope, subset):
            checked.append((scope, subset))
            return association

        monkeypatch.setattr(primary, "_validate_association_receipt", checked_association)
        SENS._validate_upstream(root, "full_home", "all_recorded_home_exploratory")
        assert checked == [("full_home", "all_recorded_home_exploratory"),
                           ("full_home", "verified_blood")]
        source.with_suffix(".json").write_text('{"status":"changed"}')
        with pytest.raises(RuntimeError, match="STOP_invalid_Home_discovery_chain"):
            SENS.plan(root, "full_home", "all_recorded_home_exploratory")
        assert not (root / "groups/full_home/sensitivity_plan_all_recorded_home_exploratory.json").exists()


def test_pgen_rejects_mutated_frozen_candidate_tsv(monkeypatch):
    with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent) as folder:
        root = Path(folder)
        primary, association, source, frozen = _frozen_chain_fixture(root, monkeypatch)
        monkeypatch.setattr(primary, "_validate_association_receipt", lambda *args: association)
        SENS._validate_upstream(root, "full_home", "all_recorded_home_exploratory")
        frozen.write_text(frozen.read_text() + "g2\t0.2\n")
        with pytest.raises(RuntimeError, match="STOP_frozen_Home_candidate_hash_changed"):
            SENS.pgen_candidates(root)
        assert not (root / "groups/full_home/sensitivity_pgen_receipt.json").exists()


def test_blood_sensitivity_requires_its_own_association_receipt(monkeypatch):
    with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent) as folder:
        root = Path(folder)
        primary, association, source, frozen = _frozen_chain_fixture(root, monkeypatch)

        def checked_association(_, scope, subset):
            if subset == "verified_blood":
                raise ValueError("Documented-blood association receipt invalid")
            return association

        monkeypatch.setattr(primary, "_validate_association_receipt", checked_association)
        with pytest.raises(RuntimeError, match="STOP_invalid_Home_discovery_chain"):
            SENS.plan(root, "full_home", "verified_blood")
        assert not (root / "groups/full_home/sensitivity_plan_verified_blood.json").exists()


def _batch_case(tables):
    rows = []
    for batch, counts in tables.items():
        for y, carrier, n in ((1, 1, counts[0][0]), (1, 0, counts[0][1]),
                              (0, 1, counts[1][0]), (0, 0, counts[1][1])):
            for _ in range(n):
                rows.append({"group_id": "g1", "patient_key": f"p{len(rows):03}",
                             "library_batch": batch, "recorded_n_binary": y,
                             "presence": carrier})
    return pd.DataFrame(rows)


def test_batch_cmh_matches_manual_mantel_haenszel_and_reports_carrier_rates():
    tables = {"A": [[8, 2], [4, 6]], "B": [[4, 1], [2, 3]]}
    strata, cmh = SENS.batch_presence_sensitivity(_batch_case(tables))
    assert len(strata) == 2 and len(cmh) == 1
    assert np.allclose(strata.positive_carrier_rate, [0.8, 0.8])
    assert np.allclose(strata.negative_carrier_rate, [0.4, 0.4])
    assert np.allclose(strata.presence_delta, [0.4, 0.4])
    row = cmh.iloc[0]
    assert np.isclose(row.cmh_common_or, 6.0)
    diff = variance = 0.0
    for cell in tables.values():
        a, b = cell[0]
        c, d = cell[1]
        n = a+b+c+d
        diff += a - (a+b)*(a+c)/n
        variance += (a+b)*(c+d)*(a+c)*(b+d)/(n*n*(n-1))
    assert np.isclose(row.cmh_p_nominal, chi2.sf(diff*diff/variance, 1))
    assert row.cmh_status == "ok_exploratory"
    assert not row.direction_discordant
    assert not any("q" in column for column in cmh.columns)


def test_batch_cmh_flags_sparse_b_and_unestimable_batch():
    sparse = {"A": [[8, 2], [4, 6]], "B": [[1, 0], [2, 7]]}
    strata, cmh = SENS.batch_presence_sensitivity(_batch_case(sparse))
    b = strata.set_index("library_batch").loc["B"]
    assert b.n_positive == 1
    assert b.sparse_class_lt5 and b.zero_cell
    assert np.isnan(b.within_batch_or)
    assert cmh.iloc[0].cmh_status == "sparse_or_zero_cell_exploratory"
    assert np.isfinite(cmh.iloc[0].cmh_common_or)
    assert np.isfinite(cmh.iloc[0].cmh_p_nominal)

    unestimable = {"A": [[8, 2], [4, 6]], "B": [[0, 0], [3, 7]]}
    strata, cmh = SENS.batch_presence_sensitivity(_batch_case(unestimable))
    assert strata.set_index("library_batch").loc["B", "status"] == "single_outcome_class"
    assert np.isnan(cmh.iloc[0].cmh_common_or)
    assert np.isnan(cmh.iloc[0].cmh_p_nominal)
    assert cmh.iloc[0].cmh_status == "not_estimable_fewer_than_two_informative_batches"


def test_batch_sensitivity_reports_direction_discordance_and_rejects_duplicates():
    tables = {"A": [[8, 2], [4, 6]], "B": [[1, 4], [4, 1]]}
    frame = _batch_case(tables)
    strata, cmh = SENS.batch_presence_sensitivity(frame)
    assert cmh.iloc[0].direction_discordant
    assert np.isclose(cmh.iloc[0].batch_presence_delta_range, 1.0)
    with pytest.raises(ValueError, match="Invalid patient-level"):
        SENS.batch_presence_sensitivity(pd.concat([frame, frame.iloc[[0]]], ignore_index=True))
