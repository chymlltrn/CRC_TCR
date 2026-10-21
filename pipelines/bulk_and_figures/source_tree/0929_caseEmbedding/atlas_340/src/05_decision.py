#!/usr/bin/env python3
"""Fail-closed recorded-N+ contribution decision from frozen patient-level evidence.

The final table is deliberately empty unless Home pooled-q association, paired
outer-fold prediction, frozen External direction, and fold-local deletion
retraining all agree. This script only combines evidence; it never searches
for groups or retunes a model using outer test labels.
"""
from __future__ import annotations
from release_runtime import private_literal as _private_literal, release_path as _release_path

import argparse
import hashlib
import json
import math
import os
import tempfile
from pathlib import Path

os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import numpy as np
import pandas as pd


ROOT = Path(_release_path('path_99d8c1c6fde2a6b9'))
FINAL_COLUMNS = (
    "method", "group_id", "home_q", "contribution_pass",
    "external_same_direction", "oof_gain_ci_lower_gt_zero",
    "outer_folds_favorable_ge4", "ablation_stable", "sensitivity_stable",
    "home_adjusted_p", "home_adjusted_beta", "home_presence_difference",
    "external_n_carriers_known", "external_presence_delta",
    "oof_paired_log_loss_gain", "oof_paired_gain_ci_low",
    "oof_favorable_outer_folds", "ablation_n_nominated_folds",
    "ablation_n_favorable_folds", "ablation_deletion_loss_ci_low",
    "ablation_deletion_vs_random_ci_low",
)
AUDIT_COLUMNS = FINAL_COLUMNS + ("reason",)
SOURCE_PATHS = {
    "source_home_candidates_sha256": "groups/frozen_home_candidates.tsv",
    "source_oof_receipt_sha256": "prediction/05_receipt.json",
    "source_external_receipt_sha256": "groups/full_home/external_direction.json",
    "source_ablation_receipt_sha256": "results/05_ablation_receipt.json",
    "source_sensitivity_receipt_sha256": "results/05_sensitivity_receipt.json",
}


class GateError(ValueError):
    """Missing, inconsistent, or adverse evidence prevents a final candidate."""


def require(condition: bool, message: str) -> None:
    if not bool(condition):
        raise GateError(message)


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def load_json(path: Path) -> dict:
    value = json.loads(path.read_text())
    require(isinstance(value, dict), f"JSON object required: {path}")
    return value


def table(path: Path, needed: set[str]) -> pd.DataFrame:
    value = pd.read_csv(path, sep="\t", dtype={"group_id": str, "method": str,
                                               "patient_key": str})
    require(needed <= set(value.columns), f"Missing columns in {path}: {needed-set(value.columns)}")
    return value


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode="w", dir=path.parent, suffix=".tmp",
                                     delete=False) as f:
        json.dump(value, f, sort_keys=True, indent=2, allow_nan=False)
        f.write("\n")
        name = f.name
    os.replace(name, path)


def write_tsv(path: Path, value: pd.DataFrame, columns: tuple[str, ...]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode="w", dir=path.parent, suffix=".tmp",
                                     delete=False) as f:
        value.reindex(columns=list(columns)).to_csv(f, sep="\t", index=False,
                                                    float_format="%.12g")
        name = f.name
    os.replace(name, path)


def exact_bool(value: object) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, str):
        if value.lower() in ("true", "1", "yes"):
            return True
        if value.lower() in ("false", "0", "no"):
            return False
    if value in (0, 1):
        return bool(value)
    raise GateError(f"Invalid boolean gate value: {value!r}")


def unique_key(frame: pd.DataFrame, name: str) -> None:
    require(not frame[["method", "group_id"]].isna().any().any(),
            f"Null group key in {name}")
    require(not frame[["method", "group_id"]].duplicated().any(),
            f"Duplicate method/group key in {name}")


def checked_number(value: object, name: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise GateError(f"Invalid {name}: {value!r}") from exc
    require(math.isfinite(result), f"Nonfinite {name}")
    return result


def bh(p: np.ndarray) -> np.ndarray:
    p = np.asarray(p, dtype=float)
    require(np.isfinite(p).all() and ((p >= 0) & (p <= 1)).all(), "Invalid pooled P values")
    order = np.argsort(p, kind="stable")
    q = np.empty(len(p), dtype=float)
    q[order] = np.minimum.accumulate((p[order] * len(p) /
                                     np.arange(1, len(p)+1))[::-1])[::-1].clip(0, 1)
    return q


def verify_home(root: Path) -> pd.DataFrame:
    base = root / "groups"
    full = base / "full_home"
    cand_path = base / "frozen_home_candidates.tsv"
    assoc_path = full / "association_recorded_home.tsv"
    lock = root / "audit/05_association_universe_lock.json"
    receipt_path = root / "audit/05_association_receipt.json"
    cand_receipt_path = base / "frozen_home_candidates.json"
    receipt = load_json(receipt_path)
    cr = load_json(cand_receipt_path)
    require(receipt.get("status") == cr.get("status") == "complete",
            "Home association or candidate receipt incomplete")
    checks = (
        (receipt, "association_tsv_sha256", assoc_path),
        (receipt, "candidate_tsv_sha256", cand_path),
        (receipt, "candidate_receipt_sha256", cand_receipt_path),
        (receipt, "eligibility_lock_sha256", lock),
        (cr, "candidate_sha256", cand_path),
        (cr, "source_association_sha256", assoc_path),
        (cr, "eligibility_lock_sha256", lock),
        (cr, "membership_sha256", full / "group_membership.parquet"),
    )
    for obj, field, path in checks:
        require(path.is_file() and obj.get(field) == sha(path),
                f"Home source hash mismatch: {field}")
    assoc = table(assoc_path, {"method", "group_id", "adjusted_p", "adjusted_beta",
                               "adjusted_status", "adjusted_q_bh_pooled",
                               "prevalence_difference", "home_association_candidate"})
    candidates = table(cand_path, set(assoc.columns))
    unique_key(assoc, "association")
    unique_key(candidates, "candidate")
    require(len(assoc) == receipt.get("n_tested_pooled") and
            len(candidates) == cr.get("n_candidates") == receipt.get("n_home_association_candidates"),
            "Home tested/candidate counts differ from receipts")
    fit_ok = int(assoc.adjusted_status.eq("ok").sum())
    fit_failed = len(assoc) - fit_ok
    require(receipt.get("n_primary_fit_ok") == fit_ok and
            receipt.get("n_primary_fit_p1") == fit_failed,
            "Home adjusted-fit status counts differ from receipt")
    require(assoc.empty or fit_ok > 0,
            "All eligible Home adjusted fits failed; no negative association conclusion")
    q = bh(assoc.adjusted_p.to_numpy(dtype=float))
    require(np.allclose(q, assoc.adjusted_q_bh_pooled.to_numpy(dtype=float),
                        atol=1e-10, rtol=1e-8), "Home pooled BH q does not recalculate")
    expected = (assoc.adjusted_status.eq("ok") & assoc.adjusted_q_bh_pooled.lt(.05) &
                assoc.adjusted_beta.gt(0) & assoc.prevalence_difference.gt(0))
    require((assoc.home_association_candidate.map(exact_bool).to_numpy() ==
             expected.to_numpy()).all(), "Home candidate flag inconsistent with locked q/direction")
    require(set(zip(candidates.method, candidates.group_id)) ==
            set(zip(assoc.loc[expected, "method"], assoc.loc[expected, "group_id"])),
            "Frozen Home candidate set differs from all positive pooled-q groups")
    expected_rows = assoc.loc[expected].sort_values(["method", "group_id"]).reset_index(drop=True)
    observed_rows = candidates.sort_values(["method", "group_id"]).reset_index(drop=True)
    require(expected_rows.empty or expected_rows.equals(observed_rows),
            "Frozen Home candidate values differ from association rows")
    require(receipt.get("n_home_known") == 126 and
            receipt.get("n_recorded_N0") == 70 and receipt.get("n_recorded_Nplus") == 56,
            "Home recorded-N population differs from locked patient counts")
    return candidates


def paired_ci(delta: np.ndarray) -> tuple[float, float, float]:
    require(delta.ndim == 1 and len(delta) == 126, "Expected 126 patient-paired gains")
    return bootstrap_ci(delta)


def bootstrap_ci(delta: np.ndarray) -> tuple[float, float, float]:
    require(delta.ndim == 1 and len(delta) > 0 and np.isfinite(delta).all(),
            "Invalid patient-paired effect vector")
    rng = np.random.default_rng(20260930)
    ix = rng.integers(0, len(delta), size=(5000, len(delta)), endpoint=False)
    low, high = np.quantile(delta[ix].mean(axis=1), (.025, .975))
    return float(delta.mean()), float(low), float(high)


def verify_prediction(root: Path) -> dict:
    folder = root / "prediction"
    receipt = load_json(folder / "05_receipt.json")
    summary = load_json(folder / "05_summary.json")
    require(receipt.get("status") == summary.get("status") == "complete",
            "Prediction receipt or summary incomplete")
    require(receipt.get("n_oof_patients") == 126 and receipt.get("n_outer_folds") == 5 and
            summary.get("n_outer_folds") == 5, "Prediction patient/fold count mismatch")
    outs = {"oof": "05_patient_oof.tsv", "outer_metrics": "05_outer_metrics.tsv",
            "inner_tuning": "05_inner_tuning.tsv", "summary": "05_summary.json"}
    for key, name in outs.items():
        require(receipt.get("output_sha256", {}).get(key) == sha(folder / name),
                f"Prediction hash mismatch: {key}")
    sources = receipt.get("source_sha256", {})
    require(isinstance(sources, dict), "Prediction source-hash receipt malformed")
    for key, path in {
        "folds": root / "folds/folds.tsv",
        "manifest": root / "config/cohort_samples.tsv",
        "analysis_spec": root / "config/analysis_spec.json",
    }.items():
        require(path.is_file() and sources.get(key) == sha(path),
                f"Prediction {key} source changed")
    scope_files = sources.get("scope_files")
    require(isinstance(scope_files, dict) and
            set(scope_files) == {f"outer_{i}" for i in range(5)},
            "Prediction fold-local source-hash set incomplete")
    for scope, hashes in scope_files.items():
        require(isinstance(hashes, dict) and set(hashes) ==
                {"receipt", "burden", "membership"},
                f"Prediction source-hash fields incomplete: {scope}")
        for key, name in {
            "receipt": "burden_receipt.json",
            "burden": "patient_group_burden.parquet",
            "membership": "group_membership.parquet",
        }.items():
            path = root / "groups" / scope / name
            require(path.is_file() and hashes[key] == sha(path),
                    f"Prediction fold-local {scope}/{key} source changed")
    oof = table(folder / "05_patient_oof.tsv",
                {"patient_key", "outer_fold", "recorded_n_binary", "baseline_probability",
                 "groups_probability", "paired_log_loss_improvement"})
    folds = table(root / "folds/folds.tsv",
                  {"patient_key", "outer_fold", "recorded_n_binary", "partition"})
    expected = folds.loc[folds.partition.eq("test"),
                         ["patient_key", "outer_fold", "recorded_n_binary"]]
    require(len(oof) == len(expected) == 126 and not oof.patient_key.duplicated().any() and
            oof[["patient_key", "outer_fold", "recorded_n_binary"]].sort_values("patient_key").reset_index(drop=True).equals(
                expected.sort_values("patient_key").reset_index(drop=True)),
            "OOF rows do not match held-out patient assignment")
    y = oof.recorded_n_binary.to_numpy(dtype=int)
    p0 = oof.baseline_probability.to_numpy(dtype=float)
    p1 = oof.groups_probability.to_numpy(dtype=float)
    require(np.isin(y, [0, 1]).all() and np.isfinite(p0).all() and np.isfinite(p1).all() and
            ((p0 > 0) & (p0 < 1)).all() and ((p1 > 0) & (p1 < 1)).all(),
            "Invalid OOF labels or probabilities")
    delta = (-(y*np.log(p0) + (1-y)*np.log1p(-p0)) +
             (y*np.log(p1) + (1-y)*np.log1p(-p1)))
    require(np.allclose(delta, oof.paired_log_loss_improvement, atol=1e-9, rtol=1e-8),
            "OOF patient paired log loss does not recalculate")
    mean, low, high = paired_ci(delta)
    obj = summary.get("paired_baseline_minus_groups_log_loss", {})
    require(all(np.isclose(value, checked_number(obj.get(key), key), atol=1e-8,
                           rtol=1e-6) for key, value in
                (("mean", mean), ("ci_low", low), ("ci_high", high))),
            "Prediction paired CI does not recalculate")
    per_fold = oof.groupby("outer_fold").paired_log_loss_improvement.mean()
    require(set(per_fold.index) == set(range(5)), "Missing OOF fold")
    favorable = int(per_fold.gt(0).sum())
    require(favorable == summary.get("favorable_outer_folds") and
            exact_bool(summary.get("meets_predictive_contribution_gate")) ==
            (low > 0 and favorable >= 4), "Prediction summary gate inconsistent")
    outer = table(folder / "05_outer_metrics.tsv",
                  {"outer_fold", "paired_log_loss_improvement"})
    require(len(outer) == 5 and set(outer.outer_fold) == set(range(5)) and
            all(np.isclose(per_fold.loc[int(r.outer_fold)], r.paired_log_loss_improvement,
                           atol=1e-9, rtol=1e-8) for r in outer.itertuples()),
            "OOF fold gains differ from fold metrics")
    return {"gain_mean": mean, "gain_ci_low": low, "gain_ci_high": high,
            "favorable_folds": favorable, "pass": low > 0 and favorable >= 4}


def verify_external(root: Path, candidates: pd.DataFrame) -> pd.DataFrame:
    full = root / "groups/full_home"
    projection_path = full / "external_projection_receipt.json"
    direction_path = full / "external_direction.tsv"
    direction_receipt_path = full / "external_direction.json"
    projection = load_json(projection_path)
    direction_receipt = load_json(direction_receipt_path)
    frozen = root / "groups/frozen_home_candidates.tsv"
    require(projection.get("status") == "complete" and
            projection.get("external_n_read") is False,
            "External label-blind projection incomplete or contaminated")
    require(projection.get("frozen_candidates_sha256") == sha(frozen) and
            projection.get("frozen_candidate_receipt_sha256") ==
            sha(root / "groups/frozen_home_candidates.json") and
            projection.get("source_association_sha256") ==
            sha(full / "association_recorded_home.tsv") and
            projection.get("home_membership_sha256") ==
            sha(full / "group_membership.parquet") and
            projection.get("home_burden_sha256") ==
            sha(full / "patient_group_burden.parquet") and
            projection.get("home_burden_receipt_sha256") ==
            sha(full / "burden_receipt.json"),
            "External projection frozen source hash mismatch")
    require(direction_receipt.get("status") == "complete" and
            direction_receipt.get("projection_receipt_sha256") == sha(projection_path) and
            direction_receipt.get("frozen_candidates_sha256") == sha(frozen) and
            direction_receipt.get("direction_tsv_sha256") == sha(direction_path) and
            direction_receipt.get("n_external_N_known") == 163,
            "External score receipt/source mismatch")
    direction = table(direction_path,
                      {"method", "group_id", "n_carriers_external_known",
                       "prevalence_Nplus", "prevalence_N0", "presence_delta",
                       "evaluable", "external_same_direction"})
    unique_key(direction, "external direction")
    require(len(direction) == direction_receipt.get("n_scored_groups") == len(candidates) and
            set(zip(direction.method, direction.group_id)) ==
            set(zip(candidates.method, candidates.group_id)),
            "External direction does not cover exactly frozen Home candidates")
    for row in direction.itertuples(index=False):
        n = int(row.n_carriers_external_known)
        diff = checked_number(row.presence_delta, "External presence difference")
        require(n >= 0 and np.isclose(diff,
                                     checked_number(row.prevalence_Nplus, "External N+ prevalence") -
                                     checked_number(row.prevalence_N0, "External N0 prevalence"),
                                     atol=1e-10, rtol=1e-8),
                "External presence direction inconsistent")
        require(exact_bool(row.evaluable) == (n >= 5) and
                exact_bool(row.external_same_direction) == (n >= 5 and diff > 0),
                "External direction gate inconsistent")
    return direction


def verify_sensitivity(root: Path, candidates: pd.DataFrame) -> pd.DataFrame:
    folder = root / "results"
    receipt = load_json(folder / "05_sensitivity_receipt.json")
    output = folder / "05_candidate_sensitivity.tsv"
    repeats = folder / "05_equal_depth_repeats.tsv"
    require(receipt.get("status") == "complete" and
            receipt.get("n_candidates") == len(candidates) and
            receipt.get("n_repeats") == 3 and
            int(receipt.get("common_depth_reads", 0)) > 0 and
            receipt.get("candidate_sensitivity_sha256") == sha(output) and
            receipt.get("equal_depth_repeats_sha256") == sha(repeats),
            "Candidate sensitivity receipt or output hashes invalid")
    sources = {
        "candidate": "groups/frozen_home_candidates.tsv",
        "candidate_receipt": "groups/frozen_home_candidates.json",
        "association": "groups/full_home/association_recorded_home.tsv",
        "burden": "groups/full_home/patient_group_burden.parquet",
        "burden_receipt": "groups/full_home/burden_receipt.json",
        "batch_report": "atlas/batch_report.json",
        "manifest": "config/cohort_samples.tsv",
        "folds": "folds/folds.tsv",
    }
    require(all(receipt.get("source_sha256", {}).get(key) == sha(root / path)
                for key, path in sources.items()),
            "Candidate sensitivity sources changed after computation")
    sensitivity = table(output, {"method", "group_id", "sensitivity_stable",
                                 "equal_depth_min_Nplus_minus_N0",
                                 "equal_depth_min_carriers", "equal_depth_all_three_positive",
                                 "batch_evaluable", "batch_positive", "batch_reversal",
                                 "untreated_beta_positive"})
    repeats_table = table(repeats, {"method", "group_id", "repeat", "n_carriers",
                                    "Nplus_carriers", "N0_carriers", "Nplus_minus_N0"})
    unique_key(sensitivity, "sensitivity")
    require(len(sensitivity) == len(candidates) and
            set(zip(sensitivity.method, sensitivity.group_id)) ==
            set(zip(candidates.method, candidates.group_id)),
            "Sensitivity does not cover exactly the Home q candidates")
    stable = []
    for row in sensitivity.itertuples(index=False):
        rep = repeats_table.loc[(repeats_table.method == row.method) &
                                (repeats_table.group_id == row.group_id)]
        require(len(rep) == 3 and set(rep.repeat.astype(int)) == {0, 1, 2} and
                not rep.repeat.duplicated().any(),
                "Candidate does not have three equal-depth replicates")
        plus = rep.Nplus_carriers.to_numpy(dtype=int)
        zero = rep.N0_carriers.to_numpy(dtype=int)
        deltas = plus / 56 - zero / 70
        require(((plus >= 0) & (plus <= 56)).all() and
                ((zero >= 0) & (zero <= 70)).all() and
                np.array_equal(plus + zero, rep.n_carriers.to_numpy(dtype=int)) and
                np.allclose(deltas, rep.Nplus_minus_N0.to_numpy(dtype=float),
                            atol=1e-10, rtol=1e-8) and
                np.isclose(deltas.min(), row.equal_depth_min_Nplus_minus_N0,
                           atol=1e-10, rtol=1e-8) and
                min(plus + zero) == int(row.equal_depth_min_carriers),
                "Equal-depth replicate counts differ from summary")
        depth_ok = (int(row.equal_depth_min_carriers) >= 5 and
                    checked_number(row.equal_depth_min_Nplus_minus_N0,
                                   "equal-depth direction") > 0 and
                    (deltas > 0).all())
        require(exact_bool(row.equal_depth_all_three_positive) == depth_ok,
                "Equal-depth direction flag inconsistent with repeat counts")
        batch_ok = (int(row.batch_evaluable) >= 2 and
                    int(row.batch_positive) >= 2 and int(row.batch_reversal) == 0)
        value = depth_ok and batch_ok and exact_bool(row.untreated_beta_positive)
        require(exact_bool(row.sensitivity_stable) == value,
                "Sensitivity stable flag inconsistent with locked rule")
        stable.append(value)
    require(sum(stable) == int(receipt.get("n_stable", -1)),
            "Sensitivity stable count differs from receipt")
    require(len(repeats_table) == 3 * len(sensitivity),
            "Equal-depth repeat table has unexpected extra rows")
    return sensitivity


def verify_ablation(root: Path, candidates: pd.DataFrame) -> pd.DataFrame:
    folder = root / "results"
    receipt = load_json(folder / "05_ablation_receipt.json")
    result_path = folder / "05_ablation_summary.tsv"
    require(receipt.get("status") == "complete", "Deletion-retraining receipt incomplete")
    require(receipt.get("summary_tsv_sha256") == sha(result_path) and
            receipt.get("source_home_candidates_sha256") ==
            sha(root / "groups/frozen_home_candidates.tsv") and
            receipt.get("source_home_candidate_receipt_sha256") ==
            sha(root / "groups/frozen_home_candidates.json") and
            receipt.get("source_oof_receipt_sha256") ==
            sha(root / "prediction/05_receipt.json") and
            receipt.get("source_oof_sha256") ==
            sha(root / "prediction/05_patient_oof.tsv") and
            receipt.get("source_prediction_summary_sha256") ==
            sha(root / "prediction/05_summary.json") and
            receipt.get("source_folds_sha256") == sha(root / "folds/folds.tsv") and
            receipt.get("source_manifest_sha256") ==
            sha(root / "config/cohort_samples.tsv") and
            receipt.get("fold_tsv_sha256") ==
            sha(folder / "05_ablation_fold_tests.tsv") and
            receipt.get("patient_deltas_sha256") ==
            sha(folder / "05_ablation_patient_deltas.parquet"),
            "Deletion-retraining source/summary hash mismatch")
    require(receipt.get("n_outer_folds") == 5 and
            0 <= int(receipt.get("max_nominations_per_outer_fold", -1)) <= 5 and
            receipt.get("heldout_labels_used_for_nomination") is False and
            receipt.get("external_labels_used") is False,
            "Deletion-retraining fold nomination boundary invalid")
    result = table(result_path,
                   {"method", "full_home_group_id", "ablation_stable",
                    "n_nominated_folds", "n_favorable_folds",
                    "deletion_loss_ci_low", "deletion_vs_random_ci_low"})
    result = result.rename(columns={"full_home_group_id": "group_id"})
    unique_key(result, "deletion-retraining result")
    require(set(zip(result.method, result.group_id)) <=
            set(zip(candidates.method, candidates.group_id)),
            "Ablation result contains group outside frozen Home candidates")
    require(len(result) == receipt.get("n_home_q_candidates") == len(candidates),
            "Ablation result count differs from receipt")
    fold = table(folder / "05_ablation_fold_tests.tsv",
                 {"outer_fold", "fold_group_id", "method", "full_home_group_id",
                  "status"})
    numeric_fold = {"jaccard_train_nodes", "train_fisher_p_one_sided",
                    "deletion_loss_mean"}
    require(fold.empty or numeric_fold <= set(fold),
            "Tested ablation fold rows lack matching/effect columns")
    for column in numeric_fold - set(fold):
        fold[column] = pd.Series(dtype=float)
    require(set(fold.outer_fold.dropna().astype(int)) <= set(range(5)) and
            fold.groupby("outer_fold").size().le(5).all() and
            (int(fold.groupby("outer_fold").size().max()) if len(fold) else 0) ==
            receipt.get("max_nominations_per_outer_fold") and
            int(fold.status.eq("complete").sum()) == receipt.get("n_ablation_tested"),
            "Ablation fold nominations or tested count inconsistent")
    patients = pd.read_parquet(folder / "05_ablation_patient_deltas.parquet")
    required_patient = {"patient_key", "outer_fold", "fold_group_id",
                        "full_home_group_id", "method", "deletion_loss_increase",
                        "deletion_vs_random"}
    require(required_patient <= set(patients) and
            not patients[["patient_key", "outer_fold", "fold_group_id"]].duplicated().any(),
            "Ablation paired patient rows malformed")
    for row in result.itertuples(index=False):
        folds = int(row.n_nominated_folds)
        favorable = int(row.n_favorable_folds)
        loss_low = float(row.deletion_loss_ci_low)
        random_low = float(row.deletion_vs_random_ci_low)
        require(0 <= favorable <= folds <= 5, "Invalid deletion fold counts")
        selected = fold.loc[(fold.method == row.method) &
                            (fold.full_home_group_id == row.group_id) &
                            fold.status.eq("complete")].sort_values(
                                ["outer_fold", "jaccard_train_nodes",
                                 "train_fisher_p_one_sided"],
                                ascending=[True, False, True]).drop_duplicates("outer_fold")
        require(folds == selected.outer_fold.nunique() and
                favorable == int(selected.deletion_loss_mean.gt(0).sum()),
                "Ablation fold summary differs from matched fold tests")
        if folds:
            selected_keys = selected[["outer_fold", "fold_group_id"]]
            sub = patients.merge(selected_keys, on=["outer_fold", "fold_group_id"],
                                 how="inner", validate="many_to_one")
            require(len(sub) == int(row.n_heldout_patients) and
                    set(sub.full_home_group_id) == {row.group_id} and
                    set(sub.method) == {row.method},
                    "Ablation patient deltas do not match fold nominees")
            dmean, dlow, _ = bootstrap_ci(sub.deletion_loss_increase.to_numpy(dtype=float))
            vmean, vlow, _ = bootstrap_ci(sub.deletion_vs_random.to_numpy(dtype=float))
            require(np.isclose(dmean, float(row.deletion_loss_mean), atol=1e-9, rtol=1e-8) and
                    np.isclose(dlow, loss_low, atol=1e-9, rtol=1e-8) and
                    np.isclose(vmean, float(row.deletion_vs_random_mean), atol=1e-9, rtol=1e-8) and
                    np.isclose(vlow, random_low, atol=1e-9, rtol=1e-8),
                    "Ablation patient-paired effect or interval does not recalculate")
        else:
            require(int(row.n_heldout_patients) == 0,
                    "Unnominated group unexpectedly has ablation patient rows")
        if folds > 0:
            require(np.isfinite(loss_low) and np.isfinite(random_low),
                    "A tested deletion group lacks finite paired intervals")
        stable = (folds >= 4 and favorable >= 4 and np.isfinite(loss_low) and
                  np.isfinite(random_low) and loss_low > 0 and random_low > 0)
        require(exact_bool(row.ablation_stable) == stable,
                "Ablation stable flag inconsistent with fold/paired CI evidence")
    return result



def verify_ablation_stop(root: Path, expected_status: str) -> None:
    folder = root / "results"
    receipt = load_json(folder / "05_ablation_receipt.json")
    require(receipt.get("status") == expected_status and
            receipt.get("n_outer_folds") == 5 and
            receipt.get("max_nominations_per_outer_fold") == 0 and
            receipt.get("n_ablation_tested") == 0 and
            receipt.get("heldout_labels_used_for_nomination") is False and
            receipt.get("external_labels_used") is False,
            "Negative deletion-retraining receipt is incomplete")
    sources = {
        "source_home_candidates_sha256": "groups/frozen_home_candidates.tsv",
        "source_home_candidate_receipt_sha256": "groups/frozen_home_candidates.json",
        "source_oof_receipt_sha256": "prediction/05_receipt.json",
        "source_oof_sha256": "prediction/05_patient_oof.tsv",
        "source_prediction_summary_sha256": "prediction/05_summary.json",
        "source_folds_sha256": "folds/folds.tsv",
        "source_manifest_sha256": "config/cohort_samples.tsv",
        "summary_tsv_sha256": "results/05_ablation_summary.tsv",
        "fold_tsv_sha256": "results/05_ablation_fold_tests.tsv",
    }
    require(all(receipt.get(field) == sha(root / path)
                for field, path in sources.items()),
            "Negative deletion-retraining source or output hash mismatch")
    summary = table(folder / "05_ablation_summary.tsv",
                    {"method", "full_home_group_id", "ablation_stable"})
    fold = table(folder / "05_ablation_fold_tests.tsv",
                 {"outer_fold", "fold_group_id", "method", "full_home_group_id"})
    require(summary.empty and fold.empty,
            "Stopped deletion retraining has nonempty results")


def decide_rows(home: pd.DataFrame, prediction: dict, external: pd.DataFrame,
                ablation: pd.DataFrame,
                sensitivity: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    ext = external.set_index(["method", "group_id"])
    abl = ablation.set_index(["method", "group_id"])
    sens = sensitivity.set_index(["method", "group_id"])
    rows = []
    for r in home.itertuples(index=False):
        key = (str(r.method), str(r.group_id))
        e = ext.loc[key]
        a = abl.loc[key] if key in abl.index else None
        s = sens.loc[key]
        ext_ok = exact_bool(e.external_same_direction)
        abl_ok = a is not None and exact_bool(a.ablation_stable)
        sens_ok = exact_bool(s.sensitivity_stable)
        all_ok = bool(prediction["pass"] and ext_ok and abl_ok and sens_ok)
        reason = "PASS" if all_ok else (
            "OOF_no_stable_increment" if not prediction["pass"] else
            "External_no_same_direction" if not ext_ok else
            "Sensitivity_unstable" if not sens_ok else
            "Ablation_unavailable_or_unstable")
        rows.append({
            "method": key[0], "group_id": key[1],
            "home_q": float(r.adjusted_q_bh_pooled),
            "contribution_pass": all_ok,
            "external_same_direction": ext_ok,
            "oof_gain_ci_lower_gt_zero": bool(prediction["gain_ci_low"] > 0),
            "outer_folds_favorable_ge4": bool(prediction["favorable_folds"] >= 4),
            "ablation_stable": abl_ok,
            "sensitivity_stable": sens_ok,
            "home_adjusted_p": float(r.adjusted_p),
            "home_adjusted_beta": float(r.adjusted_beta),
            "home_presence_difference": float(r.prevalence_difference),
            "external_n_carriers_known": int(e.n_carriers_external_known),
            "external_presence_delta": float(e.presence_delta),
            "oof_paired_log_loss_gain": prediction["gain_mean"],
            "oof_paired_gain_ci_low": prediction["gain_ci_low"],
            "oof_favorable_outer_folds": prediction["favorable_folds"],
            "ablation_n_nominated_folds": int(a.n_nominated_folds) if a is not None else 0,
            "ablation_n_favorable_folds": int(a.n_favorable_folds) if a is not None else 0,
            "ablation_deletion_loss_ci_low": float(a.deletion_loss_ci_low) if a is not None else None,
            "ablation_deletion_vs_random_ci_low": float(a.deletion_vs_random_ci_low) if a is not None else None,
            "reason": reason,
        })
    audit = pd.DataFrame(rows, columns=AUDIT_COLUMNS)
    passed = audit.loc[audit.contribution_pass].copy()
    return audit, passed


def run(root: Path = ROOT) -> dict:
    root = root.resolve()
    out = root / "results"
    out.mkdir(parents=True, exist_ok=True)
    final_path = out / "05_final_candidates.tsv"
    audit_path = out / "05_final_candidate_audit.tsv"
    gate_path = out / "05_final_candidate_gate.json"
    audit = pd.DataFrame(columns=AUDIT_COLUMNS)
    passed = pd.DataFrame(columns=FINAL_COLUMNS)
    stage = "home"
    reason = ""
    evidence_complete = False
    try:
        home = verify_home(root)
        stage = "prediction"
        prediction = verify_prediction(root)
        stage = "external"
        external = verify_external(root, home)
        stage = "sensitivity"
        sensitivity = verify_sensitivity(root, home)
        stage = "ablation"
        if home.empty:
            verify_ablation_stop(root, "STOP_no_q_candidates")
            reason = "No Home group meets pooled q<0.05 and positive N+ direction"
        elif not prediction["pass"]:
            verify_ablation_stop(root, "STOP_no_oof_gain")
            reason = "All-group OOF model lacks positive paired CI or 4 favorable folds"
        else:
            ablation = verify_ablation(root, home)
            stage = "decision"
            audit, passed = decide_rows(home, prediction, external, ablation, sensitivity)
            if passed.empty:
                reason = ("No q-qualified Home group passed External direction, "
                          "sensitivity, and stable deletion retraining")
        evidence_complete = True
    except Exception as exc:
        reason = f"{stage}: {type(exc).__name__}: {exc}"
    write_tsv(audit_path, audit, AUDIT_COLUMNS)
    write_tsv(final_path, passed, FINAL_COLUMNS)
    status = "PASS" if len(passed) else "STOP" if evidence_complete else "BLOCKED"
    source_hashes = {key: sha(root / path) if (root / path).is_file() else None
                     for key, path in SOURCE_PATHS.items()}
    result = {
        "status": status, "endpoint": "recorded N+; pN/ypN unverified",
        "evidence_complete": evidence_complete,
        "n_candidates": len(passed), "reason": reason or "All evidence gates passed",
        "stopped_at": stage if status != "PASS" else None,
        "candidate_tsv_sha256": sha(final_path),
        "candidate_audit_tsv_sha256": sha(audit_path),
        "n_home_q_candidates_audited": len(audit),
        **source_hashes,
        "source_paths": SOURCE_PATHS,
        "rules": {
            "home": "pooled adjusted BH q<0.05, adjusted beta>0, N+ presence delta>0",
            "prediction": "126-patient paired OOF log loss gain CI lower>0 and >=4/5 favorable outer folds",
            "external": "frozen projection before N scoring; >=5 known carriers and N+ presence delta>0",
            "ablation": "<=5 groups nominated per outer training fold; >=4 nominated and favorable folds; paired deletion and matched-random contrast CI lower>0",
            "sensitivity": "Three common-depth repeats, Home batch directions, and untreated adjusted direction stable",
            "negative_evidence": "STOP only after complete verified evidence; BLOCKED means technical or provenance failure",
        },
        "interpretation": "Exploratory recorded-N+ association and predictive contribution; neither antigen binding nor pathologic pN+ established",
        "inference_limit": "Fold-local ablation nominees are mapped to full-Home q candidates selected using all Home labels; candidate-specific deletion intervals are post-selection evidence, not independent validation.",
    }
    write_json(gate_path, result)
    return result


def self_test() -> None:
    assert np.allclose(bh(np.array([.01, .03, .4])), [.03, .045, .4])
    assert exact_bool("true") and not exact_bool("false")
    home = pd.DataFrame([{"method": "tcrdist3", "group_id": "g1", "adjusted_q_bh_pooled": .01,
                          "adjusted_p": .001, "adjusted_beta": .8,
                          "prevalence_difference": .2}])
    ext = pd.DataFrame([{"method": "tcrdist3", "group_id": "g1",
                         "external_same_direction": True,
                         "n_carriers_external_known": 7, "presence_delta": .05}])
    abl = pd.DataFrame([{"method": "tcrdist3", "group_id": "g1",
                         "ablation_stable": True, "n_nominated_folds": 5,
                         "n_favorable_folds": 5, "deletion_loss_ci_low": .02,
                         "deletion_vs_random_ci_low": .01}])
    sens = pd.DataFrame([{"method": "tcrdist3", "group_id": "g1",
                          "sensitivity_stable": True}])
    pred = {"pass": True, "gain_mean": .1, "gain_ci_low": .03,
            "favorable_folds": 5}
    audit, passed = decide_rows(home, pred, ext, abl, sens)
    assert len(audit) == len(passed) == 1 and passed.iloc[0].contribution_pass
    pred["pass"] = False
    assert decide_rows(home, pred, ext, abl, sens)[1].empty
    pred["pass"] = True
    abl.loc[0, "ablation_stable"] = False
    assert decide_rows(home, pred, ext, abl, sens)[1].empty
    with tempfile.TemporaryDirectory() as name:
        root = Path(name)
        result = run(root)
        assert result["status"] == "BLOCKED" and not result["evidence_complete"] and result["n_candidates"] == 0
        assert list(pd.read_csv(root / "results/05_final_candidates.tsv", sep="\t").columns) == list(FINAL_COLUMNS)
        assert result["candidate_tsv_sha256"] == sha(root / "results/05_final_candidates.tsv")
    with tempfile.TemporaryDirectory() as name:
        root = Path(name)
        (root / "groups/full_home").mkdir(parents=True)
        (root / "audit").mkdir()
        (root / "prediction").mkdir()
        (root / "folds").mkdir()
        (root / "config").mkdir()
        (root / "results").mkdir()
        member = root / "groups/full_home/group_membership.parquet"
        member.write_bytes(b"synthetic membership\n")
        burden = root / "groups/full_home/patient_group_burden.parquet"
        burden.write_bytes(b"synthetic burden\n")
        burden_receipt = root / "groups/full_home/burden_receipt.json"
        write_json(burden_receipt, {"status": "complete"})
        manifest = root / "config/cohort_samples.tsv"
        manifest.write_text("cohort\n")
        spec = root / "config/analysis_spec.json"
        write_json(spec, {"fixture": True})
        scope_files = {}
        for outer in range(5):
            scope = f"outer_{outer}"
            directory = root / "groups" / scope
            directory.mkdir()
            files = {
                "receipt": directory / "burden_receipt.json",
                "burden": directory / "patient_group_burden.parquet",
                "membership": directory / "group_membership.parquet",
            }
            for key, path in files.items():
                path.write_bytes(f"{scope}/{key}\n".encode())
            scope_files[scope] = {key: sha(path) for key, path in files.items()}
        lock = root / "audit/05_association_universe_lock.json"
        write_json(lock, {"status": "frozen_before_home_N_values"})
        assoc_path = root / "groups/full_home/association_recorded_home.tsv"
        cand_path = root / "groups/frozen_home_candidates.tsv"
        assoc = pd.DataFrame([{
            "method": "tcrdist3", "group_id": "g1", "adjusted_p": .001,
            "adjusted_beta": .8, "adjusted_status": "ok",
            "adjusted_q_bh_pooled": .001, "prevalence_difference": .2,
            "home_association_candidate": True,
        }])
        write_tsv(assoc_path, assoc, tuple(assoc.columns))
        write_tsv(cand_path, assoc, tuple(assoc.columns))
        cr_path = root / "groups/frozen_home_candidates.json"
        write_json(cr_path, {
            "status": "complete", "candidate_sha256": sha(cand_path),
            "source_association_sha256": sha(assoc_path),
            "eligibility_lock_sha256": sha(lock), "membership_sha256": sha(member),
            "n_candidates": 1,
        })
        write_json(root / "audit/05_association_receipt.json", {
            "status": "complete", "association_tsv_sha256": sha(assoc_path),
            "candidate_tsv_sha256": sha(cand_path),
            "candidate_receipt_sha256": sha(cr_path),
            "eligibility_lock_sha256": sha(lock), "n_tested_pooled": 1,
            "n_home_association_candidates": 1, "n_primary_fit_ok": 1,
            "n_primary_fit_p1": 0, "n_home_known": 126,
            "n_recorded_N0": 70, "n_recorded_Nplus": 56,
        })
        patients = [f"P{i:03d}" for i in range(126)]
        labels = {key: int(i >= 70) for i, key in enumerate(patients)}
        fold_rows = []
        oof_rows = []
        for outer in range(5):
            for i, key in enumerate(patients):
                fold_rows.append({"patient_key": key, "outer_fold": outer,
                                  "recorded_n_binary": labels[key],
                                  "partition": "test" if i % 5 == outer else "train"})
                if i % 5 == outer:
                    p1 = .8 if labels[key] else .2
                    loss0 = -math.log(.5)
                    loss1 = -math.log(.8)
                    oof_rows.append({"patient_key": key, "outer_fold": outer,
                                     "recorded_n_binary": labels[key],
                                     "baseline_probability": .5,
                                     "groups_probability": p1,
                                     "paired_log_loss_improvement": loss0-loss1})
        folds_path = root / "folds/folds.tsv"
        write_tsv(folds_path, pd.DataFrame(fold_rows), tuple(fold_rows[0]))
        oof_path = root / "prediction/05_patient_oof.tsv"
        oof = pd.DataFrame(oof_rows)
        write_tsv(oof_path, oof, tuple(oof.columns))
        outer_path = root / "prediction/05_outer_metrics.tsv"
        outer = oof.groupby("outer_fold", as_index=False).paired_log_loss_improvement.mean()
        write_tsv(outer_path, outer, tuple(outer.columns))
        inner_path = root / "prediction/05_inner_tuning.tsv"
        write_tsv(inner_path, pd.DataFrame(columns=["outer_fold", "inner_fold", "C"]),
                  ("outer_fold", "inner_fold", "C"))
        ci_mean, ci_low, ci_high = paired_ci(oof.paired_log_loss_improvement.to_numpy())
        summary_path = root / "prediction/05_summary.json"
        write_json(summary_path, {
            "status": "complete", "n_outer_folds": 5, "favorable_outer_folds": 5,
            "meets_predictive_contribution_gate": True,
            "paired_baseline_minus_groups_log_loss": {
                "mean": ci_mean, "ci_low": ci_low, "ci_high": ci_high,
            },
        })
        oof_receipt_path = root / "prediction/05_receipt.json"
        write_json(oof_receipt_path, {
            "status": "complete", "n_oof_patients": 126, "n_outer_folds": 5,
            "source_sha256": {
                "folds": sha(folds_path), "manifest": sha(manifest),
                "analysis_spec": sha(spec), "scope_files": scope_files,
            },
            "output_sha256": {"oof": sha(oof_path), "outer_metrics": sha(outer_path),
                              "inner_tuning": sha(inner_path),
                              "summary": sha(summary_path)},
        })
        projection_path = root / "groups/full_home/external_projection_receipt.json"
        write_json(projection_path, {
            "status": "complete", "external_n_read": False,
            "frozen_candidates_sha256": sha(cand_path),
            "frozen_candidate_receipt_sha256": sha(cr_path),
            "source_association_sha256": sha(assoc_path),
            "home_membership_sha256": sha(member),
            "home_burden_sha256": sha(burden),
            "home_burden_receipt_sha256": sha(burden_receipt),
        })
        direction_path = root / "groups/full_home/external_direction.tsv"
        direction = pd.DataFrame([{
            "method": "tcrdist3", "group_id": "g1",
            "n_carriers_external_known": 7,
            "prevalence_Nplus": .12, "prevalence_N0": .04,
            "presence_delta": .08, "evaluable": True,
            "external_same_direction": True,
        }])
        write_tsv(direction_path, direction, tuple(direction.columns))
        write_json(root / "groups/full_home/external_direction.json", {
            "status": "complete", "projection_receipt_sha256": sha(projection_path),
            "frozen_candidates_sha256": sha(cand_path),
            "direction_tsv_sha256": sha(direction_path),
            "n_external_N_known": 163, "n_scored_groups": 1,
        })
        (root / "atlas").mkdir()
        batch_report = root / "atlas/batch_report.json"
        write_json(batch_report, {"status": "complete", "equal_depth_reads": 208985})
        sensitivity_path = root / "results/05_candidate_sensitivity.tsv"
        sensitivity = pd.DataFrame([{
            "method": "tcrdist3", "group_id": "g1",
            "equal_depth_min_Nplus_minus_N0": 6/56-2/70,
            "equal_depth_min_carriers": 8,
            "equal_depth_all_three_positive": True,
            "batch_evaluable": 3, "batch_positive": 3, "batch_reversal": 0,
            "untreated_beta_positive": True, "sensitivity_stable": True,
        }])
        write_tsv(sensitivity_path, sensitivity, tuple(sensitivity.columns))
        repeat_path = root / "results/05_equal_depth_repeats.tsv"
        write_tsv(repeat_path, pd.DataFrame([{"method": "tcrdist3", "group_id": "g1",
                                             "repeat": i, "n_carriers": 8,
                                             "Nplus_carriers": 6, "N0_carriers": 2,
                                             "Nplus_minus_N0": 6/56-2/70}
                                            for i in range(3)]),
                  ("method", "group_id", "repeat", "n_carriers", "Nplus_carriers",
                   "N0_carriers", "Nplus_minus_N0"))
        write_json(root / "results/05_sensitivity_receipt.json", {
            "status": "complete", "n_candidates": 1, "n_stable": 1,
            "n_repeats": 3, "common_depth_reads": 208985,
            "candidate_sensitivity_sha256": sha(sensitivity_path),
            "equal_depth_repeats_sha256": sha(repeat_path),
            "source_sha256": {
                "candidate": sha(cand_path), "candidate_receipt": sha(cr_path),
                "association": sha(assoc_path), "burden": sha(burden),
                "burden_receipt": sha(burden_receipt),
                "batch_report": sha(batch_report), "manifest": sha(manifest),
                "folds": sha(folds_path),
            },
        })
        ablation_path = root / "results/05_ablation_summary.tsv"
        abl = pd.DataFrame([{
            "method": "tcrdist3", "full_home_group_id": "g1",
            "ablation_stable": True, "n_nominated_folds": 5,
            "n_favorable_folds": 5, "n_heldout_patients": 126,
            "deletion_loss_mean": .1, "deletion_loss_ci_low": .1,
            "deletion_vs_random_mean": .09,
            "deletion_vs_random_ci_low": .09,
        }])
        write_tsv(ablation_path, abl, tuple(abl.columns))
        fold_test_path = root / "results/05_ablation_fold_tests.tsv"
        fold_test = pd.DataFrame([{"outer_fold": i, "method": "tcrdist3",
                                   "fold_group_id": f"fold_g{i}",
                                   "full_home_group_id": "g1", "status": "complete",
                                   "deletion_loss_mean": .1,
                                   "jaccard_train_nodes": .9,
                                   "train_fisher_p_one_sided": .01}
                                  for i in range(5)])
        write_tsv(fold_test_path, fold_test, tuple(fold_test.columns))
        patient_path = root / "results/05_ablation_patient_deltas.parquet"
        pd.DataFrame([{"patient_key": r["patient_key"],
                       "outer_fold": r["outer_fold"],
                       "fold_group_id": f"fold_g{r['outer_fold']}",
                       "full_home_group_id": "g1", "method": "tcrdist3",
                       "deletion_loss_increase": .1,
                       "random_deletion_loss_increase": .01,
                       "deletion_vs_random": .09} for r in oof_rows]).to_parquet(patient_path)
        write_json(root / "results/05_ablation_receipt.json", {
            "status": "complete", "summary_tsv_sha256": sha(ablation_path),
            "source_home_candidates_sha256": sha(cand_path),
            "source_home_candidate_receipt_sha256": sha(cr_path),
            "source_oof_receipt_sha256": sha(oof_receipt_path),
            "source_oof_sha256": sha(oof_path),
            "source_prediction_summary_sha256": sha(summary_path),
            "source_folds_sha256": sha(folds_path),
            "source_manifest_sha256": sha(manifest),
            "fold_tsv_sha256": sha(fold_test_path),
            "patient_deltas_sha256": sha(patient_path),
            "n_outer_folds": 5, "max_nominations_per_outer_fold": 1,
            "heldout_labels_used_for_nomination": False,
            "external_labels_used": False, "n_home_q_candidates": 1,
            "n_ablation_tested": 5,
        })
        result = run(root)
        assert result["status"] == "PASS", result["reason"]
        assert result["n_candidates"] == 1
        assert verify_prediction(root)["pass"]
        manifest.write_text("cohort\nchanged\n")
        stale = run(root)
        assert stale["status"] == "BLOCKED" and "prediction" in stale["reason"].lower()
        manifest.write_text("cohort\n")
        assert len(verify_external(root, verify_home(root))) == 1
        assert len(verify_ablation(root, verify_home(root))) == 1
        direction_path.write_text(direction_path.read_text().replace("0.08", "-0.08"))
        stopped = run(root)
        assert stopped["status"] == "BLOCKED" and not stopped["evidence_complete"] and stopped["n_candidates"] == 0
        assert "external" in stopped["reason"].lower()
        # A complete negative result needs the empty External, sensitivity,
        # and ablation receipts; a missing one is BLOCKED, not STOP.
        assoc.loc[0, "adjusted_p"] = .5
        assoc.loc[0, "adjusted_q_bh_pooled"] = .5
        assoc.loc[0, "home_association_candidate"] = False
        write_tsv(assoc_path, assoc, tuple(assoc.columns))
        write_tsv(cand_path, assoc.iloc[:0], tuple(assoc.columns))
        cr = load_json(cr_path)
        cr.update(candidate_sha256=sha(cand_path),
                  source_association_sha256=sha(assoc_path), n_candidates=0)
        write_json(cr_path, cr)
        ar_path = root / "audit/05_association_receipt.json"
        ar = load_json(ar_path)
        ar.update(association_tsv_sha256=sha(assoc_path),
                  candidate_tsv_sha256=sha(cand_path),
                  candidate_receipt_sha256=sha(cr_path),
                  n_home_association_candidates=0)
        write_json(ar_path, ar)
        projection = load_json(projection_path)
        projection.update(frozen_candidates_sha256=sha(cand_path),
                          frozen_candidate_receipt_sha256=sha(cr_path),
                          source_association_sha256=sha(assoc_path))
        write_json(projection_path, projection)
        write_tsv(direction_path, direction.iloc[:0], tuple(direction.columns))
        dr_path = root / "groups/full_home/external_direction.json"
        dr = load_json(dr_path)
        dr.update(projection_receipt_sha256=sha(projection_path),
                  frozen_candidates_sha256=sha(cand_path),
                  direction_tsv_sha256=sha(direction_path),
                  n_scored_groups=0)
        write_json(dr_path, dr)
        write_tsv(sensitivity_path, sensitivity.iloc[:0],
                  tuple(sensitivity.columns))
        write_tsv(repeat_path, pd.DataFrame(columns=[
            "method", "group_id", "repeat", "n_carriers",
            "Nplus_carriers", "N0_carriers", "Nplus_minus_N0"]),
                  ("method", "group_id", "repeat", "n_carriers",
                   "Nplus_carriers", "N0_carriers", "Nplus_minus_N0"))
        sr_path = root / "results/05_sensitivity_receipt.json"
        sr = load_json(sr_path)
        sr.update(n_candidates=0, n_stable=0,
                  candidate_sensitivity_sha256=sha(sensitivity_path),
                  equal_depth_repeats_sha256=sha(repeat_path))
        sr["source_sha256"].update(candidate=sha(cand_path),
                                   candidate_receipt=sha(cr_path),
                                   association=sha(assoc_path))
        write_json(sr_path, sr)
        write_tsv(ablation_path, abl.iloc[:0], tuple(abl.columns))
        write_tsv(fold_test_path, fold_test.iloc[:0], tuple(fold_test.columns))
        abr_path = root / "results/05_ablation_receipt.json"
        abr = load_json(abr_path)
        abr.update(status="STOP_no_q_candidates",
                   source_home_candidates_sha256=sha(cand_path),
                   source_home_candidate_receipt_sha256=sha(cr_path),
                   summary_tsv_sha256=sha(ablation_path),
                   fold_tsv_sha256=sha(fold_test_path),
                   max_nominations_per_outer_fold=0, n_ablation_tested=0)
        write_json(abr_path, abr)
        negative = run(root)
        assert negative["status"] == "STOP" and negative["evidence_complete"], negative
        assert negative["n_candidates"] == 0
        assoc.loc[0, "adjusted_status"] = "fit_error_LinAlgError"
        assoc.loc[0, "adjusted_p"] = 1.0
        assoc.loc[0, "adjusted_q_bh_pooled"] = 1.0
        write_tsv(assoc_path, assoc, tuple(assoc.columns))
        ar = load_json(ar_path)
        ar.update(association_tsv_sha256=sha(assoc_path), n_primary_fit_ok=0,
                  n_primary_fit_p1=1)
        write_json(ar_path, ar)
        cr = load_json(cr_path)
        cr.update(source_association_sha256=sha(assoc_path))
        write_json(cr_path, cr)
        ar = load_json(ar_path)
        ar.update(candidate_receipt_sha256=sha(cr_path))
        write_json(ar_path, ar)
        all_failed = run(root)
        assert all_failed["status"] == "BLOCKED" and not all_failed["evidence_complete"]
        assert "all eligible home adjusted fits failed" in all_failed["reason"].lower()
        dr_path.unlink()
        incomplete = run(root)
        assert incomplete["status"] == "BLOCKED" and not incomplete["evidence_complete"]
    print(json.dumps({"status": "PASS_self_test", "checks": [
        "BH", "strict_boolean", "all_gate_positive", "predictive_stop",
        "ablation_stop", "missing_evidence_header_only_blocked",
        "full_126_patient_positive_gate", "external_hash_tamper_blocked", "complete_zero_q_stop",
        "zero_q_missing_external_blocked"]}))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test()
    else:
        print(json.dumps(run(args.root), ensure_ascii=False))


if __name__ == "__main__":
    main()
