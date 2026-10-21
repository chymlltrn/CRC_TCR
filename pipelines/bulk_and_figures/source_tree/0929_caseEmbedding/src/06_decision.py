#!/usr/bin/env python3
"""Frozen, patient-level decision for recorded-N+ TRB candidate families.

This module cannot promote a group from attention, a 2D plot, a seed graph, or
a per-fold p-value. It reads only completed upstream receipts and held-out
patient predictions. No thresholds are chosen from these predictions.
"""
from __future__ import annotations
from release_runtime import private_literal as _private_literal, release_path as _release_path

import argparse
import hashlib
import importlib.util
import json
import math
import sys
from functools import cache
from pathlib import Path

import numpy as np
import pandas as pd

DEFAULT_ROOT = Path(_release_path('path_1304218e86d2e596'))
V5_CODE_REVISION = "audit/06_fullhome_all_candidates_revision_20260929.json"
V5_RULE_FREEZE = "audit/06_final_rule_freeze_v5.json"
V6_CODE_REVISION = "audit/06_external_projection_contract_revision_20260929.json"
V6_RULE_FREEZE = "audit/06_final_rule_freeze_v6.json"
CODE_REVISION = "audit/06_oof_ablation_provenance_revision_20260929.json"
PRIOR_CODE_REVISION = V6_CODE_REVISION
PRIOR_RULE_FREEZE = V6_RULE_FREEZE
RULE_FREEZE = "audit/06_final_rule_freeze_v7.json"
EXPECTED_MIL_CODE_SHA256 = "9f1589d803ee0b7c014acdc4a83a0da1d3d7a351d74686c242834c3aab94dec6"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def json_read(path: Path) -> dict:
    return json.loads(path.read_text())


def atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n")
    tmp.replace(path)


def tsv(path: Path, columns: set[str]) -> pd.DataFrame:
    value = pd.read_csv(path, sep="\t", dtype={"group_id": str, "patient_key": str})
    missing = columns - set(value.columns)
    if missing:
        raise ValueError(f"{path}: missing columns {sorted(missing)}")
    return value


def boolean(value: object) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    return str(value).strip().lower() in {"true", "1", "yes"}


def logloss(y: np.ndarray, p: np.ndarray) -> np.ndarray:
    prob = np.clip(np.asarray(p, dtype=float), 1e-6, 1 - 1e-6)
    label = np.asarray(y, dtype=int)
    return -(label * np.log(prob) + (1 - label) * np.log1p(-prob))


def bh(p: list[float]) -> list[float]:
    if not p:
        return []
    values = np.asarray(p, dtype=float)
    if not np.isfinite(values).all() or ((values < 0) | (values > 1)).any():
        raise ValueError("Invalid pooled p-values")
    order = np.argsort(values)
    ranked = values[order] * len(values) / np.arange(1, len(values) + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    out = np.empty(len(values), dtype=float)
    out[order] = np.minimum(ranked, 1)
    return out.tolist()


def patient_effect(frame: pd.DataFrame, seed: int, boot_draws: int,
                   flip_draws: int) -> dict:
    """Collapse seeds to one prediction per test patient before inference."""
    keys = ["patient_key", "outer_fold"]
    required = {"patient_key", "outer_fold", "label", "seed",
                "full_prediction", "deleted_prediction", "random_deleted_prediction"}
    if required - set(frame):
        raise ValueError(f"Missing ablation columns {sorted(required - set(frame))}")
    if frame.duplicated(keys + ["seed"]).any():
        raise ValueError("Duplicate patient/fold/seed ablation rows")
    if frame.groupby("patient_key").outer_fold.nunique().gt(1).any():
        raise ValueError("Patient appears in multiple outer test folds")
    if frame.groupby("patient_key").label.nunique().gt(1).any():
        raise ValueError("Conflicting patient labels")
    seed_set = set(frame.seed)
    if len(seed_set) != 3 or frame.groupby("patient_key").seed.nunique().ne(3).any():
        raise ValueError("Every tested patient must have all three training seeds")
    numeric = ["full_prediction", "deleted_prediction", "random_deleted_prediction"]
    if not np.isfinite(frame[numeric].to_numpy(dtype=float)).all():
        raise ValueError("Nonfinite ablation predictions")
    if not frame[numeric].apply(lambda s: s.between(0, 1)).all().all():
        raise ValueError("Ablation prediction outside [0,1]")
    collapsed = frame.groupby("patient_key", as_index=False).agg(
        outer_fold=("outer_fold", "first"),
        label=("label", "first"),
        full_prediction=("full_prediction", "mean"),
        deleted_prediction=("deleted_prediction", "mean"),
        random_deleted_prediction=("random_deleted_prediction", "mean"))
    y = collapsed.label.to_numpy(dtype=int)
    if not np.isin(y, [0, 1]).all():
        raise ValueError("Nonbinary ablation outcome")
    baseline = logloss(y, collapsed.full_prediction)
    delta = logloss(y, collapsed.deleted_prediction) - baseline
    random_delta = logloss(y, collapsed.random_deleted_prediction) - baseline
    vs_random = delta - random_delta
    rng = np.random.default_rng(seed)
    n = len(delta)
    if n < 1:
        raise ValueError("No held-out patients")
    draws = rng.integers(0, n, size=(boot_draws, n))
    means = delta[draws].mean(axis=1)
    random_means = vs_random[draws].mean(axis=1)
    signs = rng.choice(np.array([-1, 1], dtype=np.int8),
                       size=(flip_draws, n), replace=True)
    observed = float(delta.mean())
    p_one_sided = (1 + int(np.sum((signs * delta).mean(axis=1) >= observed - 1e-15))) / (flip_draws + 1)
    seed_means = {}
    for training_seed, part in frame.groupby("seed"):
        yy = part.label.to_numpy(dtype=int)
        effect = logloss(yy, part.deleted_prediction) - logloss(yy, part.full_prediction)
        seed_means[str(training_seed)] = float(effect.mean())
    fold_means = {}
    for fold in sorted(collapsed.outer_fold.unique()):
        mask = collapsed.outer_fold.eq(fold).to_numpy()
        fold_means[str(int(fold))] = float(delta[mask].mean())
    return {
        "n_unique_test_patients": n,
        "n_positive_test_patients": int(y.sum()),
        "tested_outer_folds": len(fold_means),
        "mean_delta_logloss": observed,
        "patient_bootstrap_ci95": [float(np.quantile(means, 0.025)),
                                   float(np.quantile(means, 0.975))],
        "mean_delta_vs_matched_random": float(vs_random.mean()),
        "random_specificity_ci95": [float(np.quantile(random_means, 0.025)),
                                    float(np.quantile(random_means, 0.975))],
        "pooled_patient_signflip_one_sided_p": float(p_one_sided),
        "positive_fold_effects": sum(v > 0 for v in fold_means.values()),
        "positive_seed_effects": sum(v > 0 for v in seed_means.values()),
        "fold_mean_deltas": fold_means,
        "seed_mean_deltas": seed_means,
    }


def stop_report(root: Path, status: str, reason: str, missing: list[str] | None = None) -> dict:
    result = {
        "status": status, "reason": reason, "missing": missing or [],
        "endpoint": "recorded N+ only",
        "material_scope": _private_literal('private_8c7df61198dfc6dc'),
        "external_scope": "165 recorded-N BioSample proxies, previously examined; directional only",
        "candidate_claim": False,
    }
    dest = root / "final"
    atomic_json(dest / "decision.json", result)
    (dest / "report.md").write_text(
        f"# Recorded N+ TRB candidate decision\n\nStatus: {status}\n\nReason: {reason}{_private_literal('private_807d867103c7d064')}")
    return result


@cache
def _group_stats_module():
    """Use Stage 04's source validators instead of a second receipt contract."""
    path = Path(__file__).with_name("04_groups_stats.py")
    spec = importlib.util.spec_from_file_location("group_stats_provenance", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load group provenance validator: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _validate_home_chain(root: Path, frozen_path: Path) -> None:
    """Validate the complete Home chain without using external N labels."""
    stats = _group_stats_module()
    receipt = json_read(frozen_path.with_suffix(".json"))
    frozen = pd.read_csv(frozen_path, sep="\t", dtype=str)
    if len(frozen):
        stats._validate_frozen_file(root, frozen_path)
    else:
        # Stage 04's validator requires status=complete; an empty freeze has
        # status=STOP_no_home_candidates and must still prove the source chain.
        source = root / "groups/full_home/association_all_recorded_home_exploratory.tsv"
        table = stats._validate_association_receipt(
            root, "full_home", "all_recorded_home_exploratory")
        identity_sha, home_label_sha = stats._manifest_fingerprints(root)
        if (receipt.get("status") != "STOP_no_home_candidates" or
            receipt.get("n_candidates") != 0 or
            receipt.get("candidate_sha256") != sha256(frozen_path) or
            receipt.get("q_strictly_less_than") != 0.05 or
            receipt.get("max_frozen_candidates") is not None or
            receipt.get("direction_rule") !=
                "Home adjusted OR>1 and Home patient-presence delta>0; external patient-presence delta>0 with >=5 carriers" or
            receipt.get("source_subset") != "all_recorded_home_exploratory" or
            Path(receipt.get("source_association", "")).resolve() != source.resolve() or
            receipt.get("source_association_sha256") != sha256(source) or
            receipt.get("source_association_receipt_sha256") != sha256(source.with_suffix(".json")) or
            receipt.get("source_eligible_groups_sha256") != sha256(
                root / "groups/full_home/eligible_groups_all_recorded_home_exploratory.tsv") or
            receipt.get("membership_sha256") != sha256(root / "groups/full_home/group_membership.parquet") or
            receipt.get("home_burden_sha256") != sha256(root / "groups/full_home/patient_group_burden.parquet") or
            receipt.get("home_burden_receipt_sha256") != sha256(root / "groups/full_home/burden_receipt.json") or
            receipt.get("manifest_identity_sha256") != identity_sha or
            receipt.get("home_split_label_sha256") != home_label_sha or
            receipt.get("clinical_covariates_sha256") != sha256(root / "config/home_clinical_covariates.tsv") or
            table.loc[table.adjusted_model_status.eq("ok") &
                      table.adjusted_presence_q_pooled.lt(0.05) &
                      table.adjusted_presence_or.gt(1) &
                      table.presence_delta.gt(0)].shape[0]):
            raise ValueError("Empty frozen Home candidate receipt/source chain invalid")
    eligible = root / "groups/eligible_home_candidates.tsv"
    eligible_table = pd.read_csv(eligible, sep="\t", dtype=str)
    if (receipt.get("eligible_home_candidates_sha256") != sha256(eligible) or
        receipt.get("n_eligible_home_candidates") != len(eligible_table) or
        receipt.get("n_candidates") != len(eligible_table) or
        receipt.get("max_frozen_candidates") is not None or
        receipt.get("candidate_truncation") is not False or
        len(frozen) != len(eligible_table) or
        sha256(frozen_path) != sha256(eligible)):
        raise ValueError("Eligible Home candidate freeze receipt mismatch or truncated family set")


SENSITIVITY_SUBSETS = {"all_recorded_home_exploratory": 114, "verified_blood": 31}
SENSITIVITY_METRICS = {"unique_fraction", "read_fraction", "rare_capture",
                       "vj_length_adjusted_presence"}


def sensitivity_required_paths(root: Path) -> list[Path]:
    base = root / "groups/full_home"
    paths = [base / "association_verified_blood.tsv",
             base / "association_verified_blood.json"]
    for subset in SENSITIVITY_SUBSETS:
        for stem, suffix in (("plan", "json"), ("pilot", "json"),
                             ("patient_burden", "parquet"), ("tests", "tsv"),
                             ("batch_strata", "tsv"), ("batch_cmh", "tsv"),
                             ("receipt", "json")):
            paths.append(base / f"sensitivity_{stem}_{subset}.{suffix}")
    pgen_receipt = base / "sensitivity_pgen_receipt.json"
    paths.append(pgen_receipt)
    if pgen_receipt.exists() and json_read(pgen_receipt).get("status") == "complete":
        paths.extend(base / f"sensitivity_pgen_{name}.{suffix}" for name, suffix in
                     (("sequences", "parquet"), ("background", "parquet"), ("groups", "tsv")))
    return paths


def validate_code_revision(root: Path, cfg_sha: str) -> str:
    """Bind v7 to immutable v5/v6 history and the current Stage 05 source."""
    v5rp, v5fp = root / V5_CODE_REVISION, root / V5_RULE_FREEZE
    v6rp, v6fp = root / V6_CODE_REVISION, root / V6_RULE_FREEZE
    rp, fp = root / CODE_REVISION, root / RULE_FREEZE
    v5r, v5f = json_read(v5rp), json_read(v5fp)
    v6r, v6f = json_read(v6rp), json_read(v6fp)
    revision, freeze = json_read(rp), json_read(fp)
    old_code = v6f.get("code_sha256")
    current_code = sha256(Path(__file__).resolve())
    mil_code = sha256(Path(__file__).with_name("05_mil.py"))
    prior_presence = v6f.get("output_presence_at_freeze")
    if (v5r.get("status") != "locked_preoutcome" or
        v5r.get("change_type") != "remove full-Home top-five truncation before first association" or
        v5r.get("new_code_sha256") != v5f.get("code_sha256") or
        v5r.get("rule_freeze_sha256") != sha256(v5fp) or
        v5r.get("rules_sha256") != cfg_sha or
        v5f.get("rules_sha256") != cfg_sha or
        v6r.get("status") != "locked_preoutcome" or
        v6r.get("change_type") != "align external projection receipt literal with frozen Stage 04 contract" or
        v6r.get("old_code_sha256") != v5r.get("new_code_sha256") or
        v6r.get("new_code_sha256") != old_code or
        v6r.get("prior_code_revision_sha256") != sha256(v5rp) or
        v6r.get("prior_rule_freeze_sha256") != sha256(v5fp) or
        v6r.get("rule_freeze_sha256") != sha256(v6fp) or
        v6f.get("previous_rule_freeze_sha256") != sha256(v5fp) or
        v6f.get("rules_sha256") != cfg_sha or
        revision.get("status") != "locked_preoutcome" or
        revision.get("change_type") != "bind pooled OOF and source predictions before final family decision" or
        revision.get("old_code_sha256") != old_code or
        revision.get("new_code_sha256") != current_code or
        revision.get("stage05_code_sha256") != EXPECTED_MIL_CODE_SHA256 or
        revision.get("stage05_code_sha256") != mil_code or
        revision.get("prior_code_revision_sha256") != sha256(v6rp) or
        revision.get("prior_rule_freeze_sha256") != sha256(v6fp) or
        revision.get("rule_freeze_sha256") != sha256(fp) or
        revision.get("rules_sha256") != cfg_sha or
        revision.get("decision_thresholds_changed") is not False or
        revision.get("historical_freezes_modified") is not False or
        revision.get("output_presence_at_revision") != prior_presence or
        not isinstance(prior_presence, dict) or any(prior_presence.values()) or
        freeze.get("rules_sha256") != cfg_sha or
        freeze.get("code_sha256") != current_code or
        freeze.get("stage05_code_sha256") != mil_code or
        freeze.get("previous_rule_freeze_sha256") != sha256(v6fp) or
        freeze.get("all_outcome_outputs_absent") is not True or
        freeze.get("output_presence_at_freeze") != prior_presence):
        raise ValueError("Stage 06 v7 code revision/frozen rules provenance invalid")
    return sha256(rp)


@cache
def _mil_module():
    """Load Stage 05 read-only validators after verifying its frozen SHA."""
    path = Path(__file__).with_name("05_mil.py")
    if sha256(path) != EXPECTED_MIL_CODE_SHA256:
        raise ValueError("Stage 05 source changed after prospective v7 rule freeze")
    spec = importlib.util.spec_from_file_location("stage05_final_provenance", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load Stage 05 provenance validator: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def validate_mil_oof_chain(root: Path, gate: dict) -> None:
    """Recompute the locked MIL gate from all original held-out predictions."""
    mil = _mil_module()
    oof_path = root / _private_literal('private_4b09056568ab6f20')
    summary = json_read(root / "mil/oof/summary.json")
    if summary.get("code_sha256") != EXPECTED_MIL_CODE_SHA256:
        raise ValueError("Pooled MIL OOF Stage 05 source SHA mismatch")
    actual = mil.ablation_gate(root)
    if actual != gate:
        raise ValueError("MIL gate differs from current complete pooled OOF and comparators")
    if gate.get("status") not in {"pass", "blocked"} or gate.get("oof_sha256") != sha256(oof_path):
        raise ValueError("MIL gate status or OOF SHA invalid")


def validate_ablation_source_predictions(root: Path, fold: int, paired: pd.DataFrame,
                                         summary: dict, selection: dict) -> None:
    """Compare paired values with the original held-out and both retrained files."""
    mil = _mil_module()
    complete = {str(g["group_id"]) for g in summary.get("groups", [])
                if g.get("status") == "complete"}
    if set(paired.group_id.astype(str)) != complete:
        raise ValueError(f"Fold {fold} completed groups differ from paired ablation rows")
    if paired.empty:
        return
    needed = {"patient_key", "label", "seed", "group_id", "outer_fold", "model",
              "bag_size", "full_prediction", "deleted_prediction",
              "random_deleted_prediction", "full_logloss", "deleted_logloss",
              "random_deleted_logloss", "delta_logloss_vs_full",
              "random_delta_logloss_vs_full", "delta_vs_random"}
    if needed - set(paired):
        raise ValueError(f"Fold {fold} paired ablation fields missing: {sorted(needed - set(paired))}")
    model, bag = str(selection["selected_model"]), int(selection["selected_bag"])
    if (not paired.outer_fold.eq(fold).all() or not paired.model.eq(model).all() or
        not pd.to_numeric(paired.bag_size, errors="coerce").eq(bag).all() or
        paired.duplicated(["group_id", "patient_key", "seed"]).any()):
        raise ValueError(f"Fold {fold} paired ablation identity/model mismatch")
    fold_table = pd.read_csv(root / "folds/folds.tsv", sep="\t", dtype={"patient_key": str})
    expected_test = set(fold_table.loc[fold_table.partition.eq("test") &
                                       fold_table.outer_fold.eq(fold), "patient_key"])
    for group_id in sorted(complete):
        group = paired.loc[paired.group_id.astype(str).eq(group_id)]
        if len(group) != len(expected_test) * len(mil.SEEDS) or set(group.patient_key) != expected_test:
            raise ValueError(f"Fold {fold} group {group_id} paired held-out rows incomplete")
        if group.groupby("patient_key").label.nunique().ne(1).any():
            raise ValueError(f"Fold {fold} group {group_id} has conflicting labels")
        group_labels = dict(zip(group.patient_key.astype(str), group.label.astype(int)))
        group_dir = root / "mil/ablation" / f"outer_{fold}" / hashlib.sha256(group_id.encode()).hexdigest()[:16]
        for seed in mil.SEEDS:
            block = group.loc[group.seed.eq(seed)].set_index("patient_key")
            if set(block.index) != expected_test or len(block) != len(expected_test):
                raise ValueError(f"Fold {fold} group {group_id} seed {seed} patient set mismatch")
            original, original_receipt = mil.model_paths(root, fold, model, bag, seed, None)
            fit = json_read(original_receipt)
            if (fit.get("status") != "complete" or
                fit.get("code_sha256") != EXPECTED_MIL_CODE_SHA256 or
                fit.get("prediction_sha256") != sha256(original)):
                raise ValueError(f"Fold {fold} original full prediction receipt invalid")
            full = mil.valid_saved_prediction(original, sorted(expected_test), group_labels,
                                              model, bag, seed, fold, None)
            if full is None:
                raise ValueError(f"Fold {fold} original held-out prediction invalid")
            full = full.set_index("patient_key").loc[block.index]
            sources = {"full_prediction": full.prediction.to_numpy(dtype=float)}
            for treatment, field in (("group_removed", "deleted_prediction"),
                                     ("matched_random", "random_deleted_prediction")):
                path = group_dir / f"seed_{seed}_{treatment}.tsv"
                receipt = json_read(path.with_suffix(".json"))
                if (receipt.get("status") != "complete" or
                    receipt.get("code_sha256") != EXPECTED_MIL_CODE_SHA256 or
                    receipt.get("prediction_sha256") != sha256(path)):
                    raise ValueError(f"Fold {fold} group {group_id} {treatment} receipt invalid")
                prediction = pd.read_csv(path, sep="\t", dtype={"patient_key": str})
                if (len(prediction) != len(expected_test) or
                    prediction.patient_key.duplicated().any() or
                    set(prediction.patient_key) != expected_test or
                    not prediction.group_id.astype(str).eq(group_id).all() or
                    not prediction.treatment.eq(treatment).all() or
                    not prediction.seed.eq(seed).all() or
                    not prediction.label.astype(int).eq(prediction.patient_key.map(group_labels)).all() or
                    not np.isfinite(prediction.prediction).all() or
                    not prediction.prediction.between(0, 1).all()):
                    raise ValueError(f"Fold {fold} group {group_id} {treatment} patient/treatment mismatch")
                prediction = prediction.set_index("patient_key").loc[block.index]
                sources[field] = prediction.prediction.to_numpy(dtype=float)
            for field, source in sources.items():
                values = block[field].to_numpy(dtype=float)
                if not np.isfinite(values).all() or not np.allclose(values, source, rtol=0, atol=1e-12):
                    raise ValueError(f"Fold {fold} group {group_id} {field} differs from source prediction")
            labels = block.label.to_numpy(dtype=int)
            losses = {name: mil.patient_logloss(labels, source) for name, source in sources.items()}
            expected = {
                "full_logloss": losses["full_prediction"],
                "deleted_logloss": losses["deleted_prediction"],
                "random_deleted_logloss": losses["random_deleted_prediction"],
                "delta_logloss_vs_full": losses["deleted_prediction"] - losses["full_prediction"],
                "random_delta_logloss_vs_full": losses["random_deleted_prediction"] - losses["full_prediction"],
                "delta_vs_random": losses["deleted_prediction"] - losses["random_deleted_prediction"],
            }
            for field, expected_values in expected.items():
                if not np.allclose(block[field].to_numpy(dtype=float), expected_values,
                                   rtol=0, atol=1e-10):
                    raise ValueError(f"Fold {fold} group {group_id} {field} differs from Stage 05 source values")


def validate_sensitivity_chain(root: Path, frozen_path: Path) -> dict:
    """Require completed sensitivity provenance, without using its p/q as a selection gate."""
    base = root / "groups/full_home"
    blood = base / "association_verified_blood.tsv"
    _group_stats_module()._validate_association_receipt(root, "full_home", "verified_blood")
    blood_receipt = json_read(blood.with_suffix(".json"))
    if blood_receipt.get("n_patients") != 31:
        raise ValueError(_private_literal('private_1c8263f2d0aba9b5'))
    source = Path(__file__).with_name("04c_group_sensitivities.py")
    input_paths = {
        "burden": base / "patient_group_burden.parquet",
        "membership": base / "group_membership.parquet",
        "nodes": base / "patient_nodes.parquet",
        "integrated_receipt": base / "integrated_projection_receipt.json",
        "burden_receipt": base / "burden_receipt.json",
        "frozen": frozen_path,
        "frozen_receipt": frozen_path.with_suffix(".json"),
        "primary_source": Path(__file__).with_name("04_groups_stats.py"),
    }
    summary = {_private_literal('private_1f896173eac6fdce'): sha256(blood),
               _private_literal('private_cbade51666a36565'): sha256(blood.with_suffix(".json")),
               "subsets": {}}
    for subset, n_patients in SENSITIVITY_SUBSETS.items():
        paths = {stem: base / f"sensitivity_{stem}_{subset}.{suffix}"
                 for stem, suffix in (("plan", "json"), ("pilot", "json"),
                                      ("patient_burden", "parquet"), ("tests", "tsv"),
                                      ("batch_strata", "tsv"), ("batch_cmh", "tsv"),
                                      ("receipt", "json"))}
        plan, pilot, receipt = (json_read(paths[name]) for name in ("plan", "pilot", "receipt"))
        expected_inputs = input_paths | {
            "eligible": base / f"eligible_groups_{subset}.tsv",
            "association_receipt": base / f"association_{subset}.json",
        }
        if (plan.get("status") != "locked_label_blind" or plan.get("version") != 3 or
            plan.get("scope") != "full_home" or plan.get("subset") != subset or
            plan.get("source_sha256") != sha256(source) or
            plan.get("input_sha256") != {key: sha256(path) for key, path in expected_inputs.items()} or
            plan.get("n_patients") != n_patients or
            set(plan.get("tests", [])) != SENSITIVITY_METRICS or
            plan.get("no_recorded_n_used_for_sensitivity_parameter_selection") is not True or
            "independent group-wise draws are not a joint repertoire subsample" not in plan.get("rarefaction", "")):
            raise ValueError(f"Sensitivity plan source/scope/test family invalid: {subset}")
        if (pilot.get("status") != "PASS" or pilot.get("scope") != "full_home" or
            pilot.get("subset") != subset or pilot.get("plan_sha256") != sha256(paths["plan"])):
            raise ValueError(f"Sensitivity pilot receipt invalid: {subset}")
        if (receipt.get("status") != "complete" or receipt.get("scope") != "full_home" or
            receipt.get("subset") != subset or receipt.get("plan_sha256") != sha256(paths["plan"]) or
            receipt.get("patient_table_sha256") != sha256(paths["patient_burden"]) or
            receipt.get("tests_sha256") != sha256(paths["tests"]) or
            receipt.get("batch_strata_sha256") != sha256(paths["batch_strata"]) or
            receipt.get("batch_cmh_sha256") != sha256(paths["batch_cmh"]) or
            receipt.get("n_patients") != n_patients or
            receipt.get("n_groups") != plan.get("n_groups") or
            receipt.get("n_patient_group_rows") != n_patients * plan.get("n_groups", -1) or
            receipt.get("n_tests") != 4 * plan.get("n_groups", -1) or
            receipt.get("rarefaction_depth_reads") != plan.get("rarefaction_depth_reads") or
            receipt.get("rarefaction_repeats") != plan.get("repeats") or
            receipt.get("rarefaction_draw_coupling") !=
                "independent exact group marginal draws; not a joint repertoire subsample" or
            receipt.get("cmh_p_is_nominal_only") is not True or
            receipt.get("cmh_in_primary_pooled_BH_family") is not False or
            receipt.get("pgen_in_primary_test_family") is not False or
            receipt.get("n_documented_blood_included") != 31 or
            receipt.get("n_material_unverified_included") != (0 if subset == "verified_blood" else 83)):
            raise ValueError(f"Sensitivity result receipt invalid: {subset}")
        tests = tsv(paths["tests"], {"group_id", "metric"})[["group_id", "metric"]]
        eligible = tsv(expected_inputs["eligible"], {"group_id"})
        if (tests.duplicated().any() or
            set(tests.group_id) != set(eligible.group_id) or
            any(metrics != SENSITIVITY_METRICS for metrics in
                tests.groupby("group_id").metric.agg(set)) or
            len(eligible) != plan["n_groups"]):
            raise ValueError(f"Sensitivity tests incomplete: {subset}")
        batch_strata = tsv(paths["batch_strata"], {"group_id", "library_batch"})
        batch = tsv(paths["batch_cmh"], {"group_id"})
        if (batch_strata.duplicated(["group_id", "library_batch"]).any() or
            set(batch_strata.group_id) != set(eligible.group_id) or
            len(batch_strata) != receipt.get("n_batch_strata_rows") or
            batch.group_id.duplicated().any() or
            set(batch.group_id) != set(eligible.group_id) or
            len(batch) != receipt.get("n_batch_cmh_rows")):
            raise ValueError(f"Sensitivity batch outputs incomplete: {subset}")
        summary["subsets"][subset] = {"receipt_sha256": sha256(paths["receipt"]),
                                      "n_patients": n_patients,
                                      "n_groups": len(eligible)}
    pgen_path = base / "sensitivity_pgen_receipt.json"
    pgen = json_read(pgen_path)
    if (pgen.get("scope") != "full_home" or
        pgen.get("candidate_sha256") != sha256(frozen_path) or
        pgen.get("candidate_receipt_sha256") != sha256(frozen_path.with_suffix(".json")) or
        pgen.get("membership_sha256") != sha256(input_paths["membership"]) or
        pgen.get("home_association_receipt_sha256") != sha256(
            base / "association_all_recorded_home_exploratory.json") or
        pgen.get("burden_receipt_sha256") != sha256(input_paths["burden_receipt"]) or
        pgen.get("integrated_projection_receipt_sha256") != sha256(input_paths["integrated_receipt"]) or
        pgen.get("source_sha256") != sha256(source) or
        pgen.get("Pgen_used_for_primary_q") is not False or
        pgen.get("Pgen_used_for_candidate_selection") is not False or
        pgen.get("pgen_definition") != "unconditioned CDR3 AA; not V/J-adjusted" or
        pgen.get("pgen_method") !=
            "Post-selection OLGA human TRB unconditioned CDR3 AA Pgen for frozen candidates; label-blind length-matched Home AA background is descriptive; no V/J-conditioned model"):
        raise ValueError("Pgen post-selection provenance/scope invalid")
    if pgen.get("status") == "complete":
        for field, path in (("sequence_sha256", base / "sensitivity_pgen_sequences.parquet"),
                            ("background_sha256", base / "sensitivity_pgen_background.parquet"),
                            ("group_sha256", base / "sensitivity_pgen_groups.tsv"),
                            ("background_source_sha256", base / "clone_nodes.parquet")):
            if pgen.get(field) != sha256(path):
                raise ValueError(f"Pgen output/source hash invalid: {field}")
    elif not (str(pgen.get("status", "")).startswith("STOP_Pgen_") and
              isinstance(pgen.get("unavailable_reason"), str) and
              pgen["unavailable_reason"].strip() and
              (pgen.get("no_pgen_values_fabricated") is True or
               pgen.get("no_partial_pgen_published") is True)):
        raise ValueError("Pgen requires completed run or explicit unavailable reason")
    summary["pgen"] = {"status": pgen["status"],
                       "receipt_sha256": sha256(pgen_path),
                       "unavailable_reason": pgen.get("unavailable_reason")}
    return summary


def _external_labels(root: Path) -> tuple[set[str], str, int, int]:
    """Fingerprint the current 165 proxy N labels using Stage 04's format."""
    manifest = pd.read_csv(root / "config/cohort_samples.tsv", sep="\t", dtype=str,
                           keep_default_na=False,
                           usecols=["cohort", "patient_key", "recorded_n_binary", "identity_basis"])
    labels = manifest.loc[manifest.cohort.str.lower().eq("external"),
                          ["patient_key", "recorded_n_binary", "identity_basis"]]
    if (len(labels) != 165 or labels.patient_key.eq("").any() or
        labels.patient_key.duplicated().any() or
        not labels.identity_basis.eq("BioSample_proxy_unverified").all() or
        not labels.recorded_n_binary.isin(["0", "1"]).all()):
        raise ValueError("External labels must be 165 unique BioSample proxies with binary recorded N")
    pairs = labels[["patient_key", "recorded_n_binary"]].values.tolist()
    canonical = sorted(pairs)
    label_sha = hashlib.sha256(json.dumps(canonical, ensure_ascii=False,
                                         separators=(",", ":")).encode("utf-8")).hexdigest()
    return set(labels.patient_key), label_sha, sum(row[1] == "1" for row in pairs), sum(row[1] == "0" for row in pairs)


def verify_receipts(root: Path, frozen_path: Path, require_external: bool = True) -> None:
    frozen_receipt = json_read(frozen_path.with_suffix(".json"))
    if frozen_receipt.get("candidate_sha256") != sha256(frozen_path):
        raise ValueError("Frozen Home candidate hash mismatch")
    if frozen_receipt.get("external_n_read") is not False:
        raise ValueError("Frozen Home candidate selection accessed external N")
    if frozen_receipt.get("source_subset") != "all_recorded_home_exploratory":
        raise ValueError("Home candidate source subset changed")
    _validate_home_chain(root, frozen_path)
    membership = root / "groups/full_home/group_membership.parquet"
    membership_hash = sha256(membership)
    if frozen_receipt.get("membership_sha256") != membership_hash:
        raise ValueError("Home group membership changed after candidate freeze")
    if not require_external:
        return
    stats = _group_stats_module()
    patients, label_sha, n_positive, n_negative = _external_labels(root)
    if not n_positive or not n_negative:
        raise ValueError("External directional scoring requires both recorded N classes")
    patient_sha = stats._patient_hash(patients)
    identity_sha, home_label_sha = stats._manifest_fingerprints(root)
    frozen_groups = set(pd.read_csv(frozen_path, sep="\t", dtype={"group_id": str}).group_id)
    home_burden = root / "groups/full_home/patient_group_burden.parquet"
    home_burden_receipt = root / "groups/full_home/burden_receipt.json"
    home_burden_data = json_read(home_burden_receipt)
    ext_projection = root / "groups/full_home/external_projection_receipt.json"
    ext = json_read(ext_projection)
    if (ext.get("status") != "complete" or ext.get("external_n_read") is not False or
        ext.get("frozen_candidates_sha256") != sha256(frozen_path) or
        ext.get("frozen_candidate_receipt_sha256") != sha256(frozen_path.with_suffix(".json")) or
        ext.get("membership_sha256") != membership_hash or
        ext.get("home_burden_sha256") != sha256(home_burden) or
        ext.get("home_burden_receipt_sha256") != sha256(home_burden_receipt) or
        ext.get("home_qualified_group_ids_sha256") != home_burden_data.get("qualified_group_ids_sha256") or
        ext.get("frozen_group_ids_sha256") != stats._patient_hash(frozen_groups) or
        ext.get("manifest_identity_sha256") != identity_sha or
        ext.get("home_split_label_sha256") != home_label_sha or
        ext.get("target_patient_sha256") != patient_sha or
        ext.get("target_proxy_count") != 165 or
        ext.get("external_identity") != "BioSample_proxy_unverified" or
        ext.get("projection") !=
            "complete frozen-rule AA/V/J projection over all external QC clones"):
        raise ValueError("External was not projected label-blind against frozen group membership")
    burden = root / "groups/full_home/patient_group_burden_external.parquet"
    if (ext.get("burden_sha256") != sha256(burden) or
        ext.get("external_burden_receipt_sha256") != sha256(
            root / "groups/full_home/burden_receipt_external.json") or
        ext.get("burden_rows") != 165 * len(frozen_groups)):
        raise ValueError("External burden changed after label-blind projection")
    score_receipt = json_read(root / "groups/full_home/external_direction.json")
    direction = root / "groups/full_home/external_direction.tsv"
    direction_table = tsv(direction, {"group_id"})
    if (score_receipt.get("projection_receipt_sha256") != sha256(ext_projection) or
        score_receipt.get("frozen_candidates_sha256") != sha256(frozen_path) or
        score_receipt.get("projection_completed_utc") != ext.get("completed_utc") or
        score_receipt.get("external_direction_tsv_sha256") != sha256(direction) or
        score_receipt.get("external_label_sha256") != label_sha or
        score_receipt.get("external_label_canonicalization") !=
            "sorted unique [patient_key,recorded_n_binary] pairs; compact UTF-8 JSON" or
        score_receipt.get("n_external_scored_patients") != 165 or
        score_receipt.get("n_positive_proxy") != n_positive or
        score_receipt.get("n_negative_proxy") != n_negative or
        score_receipt.get("external_identity") != "BioSample_proxy_unverified" or
        score_receipt.get("n_frozen_groups") != len(frozen_groups) or
        score_receipt.get("n_scored_groups") != len(direction_table) or
        direction_table.group_id.duplicated().any() or
        set(direction_table.group_id) != frozen_groups):
        raise ValueError("External scoring provenance mismatch")


def validate_alignment_training_reference(root: Path, receipt: dict,
                                          cross: pd.DataFrame) -> None:
    """Reject alignment made with held-out AA or a stale outer-training population."""
    expected_folds = {str(int(fold)) for fold in cross.outer_fold}
    if (not expected_folds or
        not expected_folds <= {str(i) for i in range(5)} or
        receipt.get("matching_reference_population") !=
            "outer_fold_training_patient_exact_aa_vj" or
        receipt.get("matching_reference_folds_sha256") !=
            sha256(root / "folds/folds.tsv")):
        raise ValueError("Alignment reference is not frozen outer training patients")
    patient_hashes = receipt.get("matching_reference_train_patient_sha256_by_outer_fold")
    patient_counts = receipt.get("matching_reference_train_patient_count_by_outer_fold")
    node_hashes = receipt.get("matching_reference_clone_nodes_sha256_by_outer_fold")
    if (not isinstance(patient_hashes, dict) or set(patient_hashes) != expected_folds or
        not isinstance(patient_counts, dict) or set(patient_counts) != expected_folds or
        not isinstance(node_hashes, dict) or set(node_hashes) != expected_folds):
        raise ValueError("Alignment training reference fold inventory mismatch")
    folds = tsv(root / "folds/folds.tsv", {"outer_fold", "partition", "patient_key"})
    patient_hash = _group_stats_module()._patient_hash
    for fold in sorted(expected_folds):
        part = folds.loc[folds.outer_fold.eq(int(fold))]
        if (part.empty or part.patient_key.duplicated().any() or
            not part.partition.isin(["train", "test"]).all()):
            raise ValueError(f"Outer fold {fold} patient partition invalid")
        train = set(part.loc[part.partition.eq("train"), "patient_key"].astype(str))
        test = set(part.loc[part.partition.eq("test"), "patient_key"].astype(str))
        if not train or not test or train & test:
            raise ValueError(f"Outer fold {fold} training/holdout split invalid")
        scope = root / "groups" / f"outer_{fold}"
        node_desc = json_read(scope / "description.json")
        expected_hash = patient_hash(train)
        if (patient_hashes[fold] != expected_hash or
            patient_counts[fold] != len(train) or
            node_desc.get("train_patient_sha256") != expected_hash or
            set(node_desc.get("patient_ids", [])) != train or
            node_hashes[fold] != sha256(scope / "clone_nodes.parquet") or
            node_desc.get("clone_nodes_sha256") != node_hashes[fold]):
            raise ValueError(f"Outer fold {fold} alignment training reference mismatch")


def home_primary_pass(source: pd.Series, candidate: dict, rule: dict) -> bool:
    """Require adjusted association and matching positive presence scale."""
    return bool(
        int(source.n_present) >= rule["minimum_carriers"] and
        source.adjusted_model_status == "ok" and
        float(source.adjusted_presence_q_pooled) < rule["adjusted_presence_q_strictly_below"] and
        float(source.adjusted_presence_or) > rule["adjusted_presence_or_above"] and
        float(source.presence_delta) > rule["presence_delta_above"] and
        np.isclose(float(source.adjusted_presence_q_pooled), candidate["home_q_pooled"]) and
        np.isclose(float(source.adjusted_presence_or), candidate["home_adjusted_presence_or"]) and
        np.isclose(float(source.presence_delta), candidate["home_presence_delta"]))


def write_candidate_cards(root: Path, candidates: list[dict]) -> list[str]:
    """Export all exact AA/V/J members of each passing Home family in bounded batches."""
    import pyarrow as pa
    import pyarrow.dataset as ds
    import pyarrow.parquet as pq

    source = root / "groups/full_home/group_membership.parquet"
    dataset = ds.dataset(str(source), format="parquet")
    required = {"group_id", "aa", "v_gene", "j_gene", "seq_idx", "projection_type"}
    if required - set(dataset.schema.names):
        raise ValueError("Candidate group membership lacks exact sequence fields")
    cards = []
    source_sha = sha256(source)
    for candidate in candidates:
        gid = str(candidate["group_id"])
        stem = hashlib.sha256(gid.encode()).hexdigest()[:16]
        output = root / "final/candidates" / stem
        output.mkdir(parents=True, exist_ok=True)
        member_path = output / "members.parquet"
        tmp = member_path.with_suffix(".parquet.tmp")
        writer = None
        n = 0
        first = []
        scanner = dataset.scanner(columns=sorted(required),
                                  filter=ds.field("group_id") == gid,
                                  batch_size=65536)
        try:
            for batch in scanner.to_batches():
                table = pa.Table.from_batches([batch])
                observed_types = set(table.column("projection_type").to_pylist())
                if observed_types != {candidate["projection_type"]}:
                    raise ValueError(f"Candidate {gid} projection type differs from frozen family")
                if writer is None:
                    writer = pq.ParquetWriter(str(tmp), table.schema, compression="zstd")
                writer.write_table(table)
                n += table.num_rows
                if len(first) < 100:
                    first.extend(table.select(["aa", "v_gene", "j_gene"])
                                 .to_pandas().head(100 - len(first)).to_dict("records"))
        finally:
            if writer is not None:
                writer.close()
        if n == 0:
            raise ValueError(f"Passing candidate {gid} has no exact AA/V/J members")
        tmp.replace(member_path)
        pd.DataFrame(first).to_csv(output / "first_100_members.tsv", sep="\t", index=False)
        card = {
            "group_id": gid, "canonical_id": candidate["canonical_id"],
            "method": candidate["method"],
            "projection_type": candidate["projection_type"],
            "family_definition": candidate["family_definition"],
            "endpoint": "recorded N+ exploratory",
            "n_exact_aa_vj_members": n,
            "members_path": str(member_path),
            "members_sha256": sha256(member_path),
            "full_home_membership_sha256": source_sha,
            "home_carriers": candidate["home_carriers"],
            "home_q_pooled": candidate["home_q_pooled"],
            "home_presence_delta": candidate["home_presence_delta"],
            "external_carriers": candidate["external_carriers"],
            "external_presence_delta": candidate["external_presence_delta"],
            "ablation": candidate["ablation"],
            "safe_deletion": candidate.get("safe_deletion"),
            "first_100_members_semantics": "lexical scan order, not importance ranking",
            "limitations": [
                "83/114 Home sample materials unverified",
                "pathologic pN prefix and sampling/treatment timing unverified",
                "external BioSample proxies previously examined",
                "post-selection OOF significance exploratory",
                "antigen specificity and mechanism untested",
            ],
        }
        atomic_json(output / "card.json", card)
        cards.append(str(output / "card.json"))
    return cards


def run(root: Path) -> dict:
    cfg_path = root / "config/final_candidate_rules.json"
    cfg = json_read(cfg_path)
    if not cfg.get("frozen_before_outcome_run"):
        raise ValueError("Final decision rules must be frozen first")
    cfg_sha = sha256(cfg_path)
    freeze = json_read(root / RULE_FREEZE)
    if (freeze.get("rules_sha256") != cfg_sha or
        freeze.get("all_outcome_outputs_absent") is not True):
        raise ValueError("Prospective final decision rule freeze does not match")
    frozen_path = root / "groups/frozen_home_candidates.tsv"
    if not frozen_path.exists():
        return stop_report(root, "PENDING", "Home candidate set is not frozen", [str(frozen_path)])
    frozen = tsv(frozen_path, {"group_id", "method", "projection_type",
                               "family_definition", "home_q_pooled",
                               "home_adjusted_presence_or", "home_presence_delta"})
    expected_projection = {
        "gliph2": "gliph2_seed_motif_projection",
        "tcrdist3": "tcrdist3_train_seed_full_projection",
    }
    if frozen.family_definition.isna().any() or frozen.family_definition.astype(str).str.strip().eq("").any():
        raise ValueError("Frozen candidate family definition is missing")
    if any(expected_projection.get(str(r.method)) != str(r.projection_type)
           for r in frozen.itertuples()):
        raise ValueError("Frozen candidate method/projection type mismatch")
    if (cfg.get("candidate_scope") != "all_home_primary_eligible" or
        cfg.get("max_families") is not None or
        frozen.group_id.duplicated().any()):
        raise ValueError("Frozen family scope or identity invalid")
    verify_receipts(root, frozen_path, require_external=False)
    code_revision_sha = validate_code_revision(root, cfg_sha)
    if frozen.empty:
        blood_paths = sensitivity_required_paths(root)[:2]
        missing = [str(path) for path in blood_paths if not path.exists()]
        if missing:
            return stop_report(root, "PENDING", _private_literal('private_791f19752975e687'), missing)
        _group_stats_module()._validate_association_receipt(root, "full_home", "verified_blood")
        return stop_report(root, "STOP_no_Home_association",
                           "No Home candidate met the pooled patient-level primary test")
    need = [
        root / "groups/full_home/association_all_recorded_home_exploratory.tsv",
        root / "groups/full_home_to_fold_alignment.tsv",
        root / "groups/cross_fold_alignment.tsv",
        root / "groups/full_home_fold_member_matches.tsv",
        root / "groups/full_home/external_projection_receipt.json",
        root / "groups/full_home/patient_group_burden_external.parquet",
        root / "groups/full_home/external_direction.tsv",
        root / "groups/full_home/external_direction.json",
        root / "mil/ablation/gate.json",
        root / "folds/folds.tsv",
    ]
    missing = [str(p) for p in need + sensitivity_required_paths(root) if not p.exists()]
    if missing:
        return stop_report(root, "PENDING", "Upstream projection, sensitivity or MIL is incomplete", missing)
    verify_receipts(root, frozen_path)
    sensitivities = validate_sensitivity_chain(root, frozen_path)
    home = tsv(need[0], {"group_id", "n_present", "adjusted_presence_q_pooled",
                         "adjusted_presence_or", "presence_delta", "adjusted_model_status"})
    home = home.set_index("group_id")
    align = tsv(need[1], {"full_home_group_id", "canonical_id", "alignment_pass",
                          "n_matching_folds", "method"})
    cross = tsv(need[2], {"canonical_id", "outer_fold", "group_id", "method"})
    ext = tsv(need[6], {"group_id", "n_carriers_external", "evaluable",
                        "direction_positive", "presence_delta"})
    ext = ext.set_index("group_id")
    if align.full_home_group_id.duplicated().any() or cross.duplicated(["outer_fold", "group_id"]).any():
        raise ValueError("Ambiguous family alignment")
    if ext.index.duplicated().any():
        raise ValueError("Duplicate external group score")
    matched = tsv(need[3], {"full_home_group_id", "canonical_id", "outer_fold",
                            "group_id", "match_pass"})
    if matched.duplicated(["full_home_group_id", "outer_fold", "group_id"]).any():
        raise ValueError("Duplicate Home-to-fold member match")
    alignment_receipt = json_read(need[1].with_suffix(".json"))
    if (alignment_receipt.get("frozen_candidates_sha256") != sha256(frozen_path) or
        alignment_receipt.get("cross_fold_alignment_sha256") != sha256(need[2]) or
        alignment_receipt.get("full_home_fold_member_matches_sha256") != sha256(need[3]) or
        alignment_receipt.get("source_association_sha256") != sha256(need[0]) or
        alignment_receipt.get("output_sha256") != sha256(need[1])):
        raise ValueError("Full-Home-to-fold alignment receipt mismatch")
    validate_alignment_training_reference(root, alignment_receipt, cross)
    aligned_pass = align.loc[align.alignment_pass.map(boolean)]
    if aligned_pass.canonical_id.duplicated().any():
        raise ValueError("Two Home groups map to the same fold family")
    gate = json_read(root / "mil/ablation/gate.json")
    mil_rules = root / "config/mil_decision_rules.json"
    oof_path = root / _private_literal('private_4b09056568ab6f20')
    if (gate.get("decision_rules_sha256") != sha256(mil_rules) or
        gate.get("oof_sha256") != sha256(oof_path)):
        raise ValueError("MIL gate rule/OOF provenance mismatch")
    validate_mil_oof_chain(root, gate)
    validate_mil_oof_chain(root, gate)
    if gate.get("status") != "pass":
        return stop_report(root, "STOP_no_stable_model_gain",
                           "OOF MIL did not beat both locked comparators stably")
    folds = tsv(root / "folds/folds.tsv", {"outer_fold", "partition", "patient_key",
                                          "recorded_n_binary"})
    test = folds.loc[folds.partition.eq("test")]
    if test.patient_key.duplicated().any() or len(test) != 114:
        raise ValueError(_private_literal('private_48915bb53f9dce12'))
    test_lookup = test.set_index("patient_key")[["outer_fold", "recorded_n_binary"]]
    rows: list[dict] = []
    pooled_p: list[float] = []
    all_ablation: dict[int, pd.DataFrame] = {}
    all_ablation_summaries: dict[int, dict] = {}
    for fold in range(5):
        path = root / "mil/ablation" / f"outer_{fold}" / "paired_patient_seed_logloss.tsv"
        if path.exists():
            summary_path = root / "mil/ablation" / f"outer_{fold}" / "summary.json"
            if not summary_path.exists():
                raise ValueError(f"Fold {fold} ablation summary missing")
            summary = json_read(summary_path)
            selection = root / "mil/cv" / f"outer_{fold}" / "selection.json"
            if (summary.get("status") != "complete" or
                summary.get("paired_patient_seed_logloss_sha256") != sha256(path) or
                summary.get("oof_sha256") != sha256(oof_path) or
                summary.get("fold_selection_sha256") != sha256(selection) or
                summary.get("decision_rules_sha256") != sha256(mil_rules)):
                raise ValueError(f"Fold {fold} ablation provenance mismatch")
            all_ablation_summaries[fold] = summary
            all_ablation[fold] = tsv(path, {"outer_fold", "group_id", "patient_key", "label",
                                            "seed", "full_prediction", "deleted_prediction",
                                            "random_deleted_prediction"})
            validate_ablation_source_predictions(root, fold, all_ablation[fold], summary,
                                                 json_read(selection))
            validate_ablation_source_predictions(root, fold, all_ablation[fold], summary,
                                                 json_read(selection))
    for candidate in frozen.itertuples():
        gid = str(candidate.group_id)
        result: dict = {"group_id": gid, "method": str(candidate.method),
                        "projection_type": str(candidate.projection_type),
                        "family_definition": str(candidate.family_definition),
                        "home_q_pooled": float(candidate.home_q_pooled),
                        "home_adjusted_presence_or": float(candidate.home_adjusted_presence_or),
                        "home_presence_delta": float(candidate.home_presence_delta),
                        "home_read_fraction_delta_median": float(
                            getattr(candidate, "home_read_fraction_delta_median", np.nan)),
                        "rules_sha256": cfg_sha}
        if gid not in home.index:
            raise ValueError(f"Frozen group {gid} absent from Home complete test family")
        h = home.loc[gid]
        result["home_carriers"] = int(h.n_present)
        result["home_pass"] = home_primary_pass(h, result, cfg["home"])
        e = ext.loc[gid] if gid in ext.index else None
        result["external_carriers"] = int(e.n_carriers_external) if e is not None else 0
        result["external_presence_delta"] = float(e.presence_delta) if e is not None else math.nan
        result["external_pass"] = bool(
            e is not None and result["external_carriers"] >= cfg["external"]["minimum_carriers"] and
            boolean(e.evaluable) and boolean(e.direction_positive) and
            float(e.presence_delta) > 0)
        pair = align.loc[align.full_home_group_id.astype(str).eq(gid)]
        if len(pair) != 1:
            raise ValueError(f"Expected one Home-to-fold alignment row for {gid}")
        a = pair.iloc[0]
        result["canonical_id"] = "" if pd.isna(a.canonical_id) else str(a.canonical_id)
        result["alignment_pass"] = bool(boolean(a.alignment_pass) and
                                        int(a.n_matching_folds) >= cfg["alignment"]["minimum_matching_outer_folds"] and
                                        str(a.method) == str(candidate.method))
        result["n_matching_folds"] = int(a.n_matching_folds)
        result["ablation_pass"] = False
        result["ablation"] = None
        result["ablation_unassessable"] = None
        result["pooled_ablation_p"] = 1.0
        if result["alignment_pass"]:
            members = cross.loc[cross.canonical_id.eq(result["canonical_id"])]
            matches = matched.loc[
                matched.full_home_group_id.astype(str).eq(gid) &
                matched.canonical_id.eq(result["canonical_id"]) &
                matched.match_pass.map(boolean)]
            members = members.merge(matches[["outer_fold", "group_id"]],
                                    on=["outer_fold", "group_id"], how="inner",
                                    validate="one_to_one")
            if len(members) != int(a.n_matching_folds):
                raise ValueError("Matched-fold count differs from frozen Home alignment")
            if members.outer_fold.duplicated().any():
                raise ValueError("Two fold groups assigned to one canonical family")
            if members.outer_fold.nunique() < cfg["alignment"]["minimum_matching_outer_folds"]:
                raise ValueError("Alignment receipt and member table disagree")
            pieces = []
            safe_details = []
            skipped_members = []
            for member in members.itertuples():
                fold = int(member.outer_fold)
                if fold not in all_ablation:
                    return stop_report(root, "PENDING", "A nominated fold ablation is unfinished",
                                       [str(root / "mil/ablation" / f"outer_{fold}" /
                                            "paired_patient_seed_logloss.tsv")])
                p = all_ablation[fold]
                p = p.loc[p.group_id.astype(str).eq(str(member.group_id))].copy()
                groups = [g for g in all_ablation_summaries[fold].get("groups", [])
                          if str(g.get("group_id")) == str(member.group_id)]
                if len(groups) != 1:
                    raise ValueError("Missing unique group-specific ablation receipt")
                group_status = groups[0].get("status")
                if group_status == "skipped":
                    if not p.empty:
                        raise ValueError("Skipped group unexpectedly has held-out ablation rows")
                    skipped_members.append({
                        "outer_fold": fold, "group_id": str(member.group_id),
                        "reason": str(groups[0].get("reason", "unspecified"))})
                    continue
                if group_status != "complete":
                    raise ValueError("Unknown group-specific ablation status")
                if p.empty:
                    raise ValueError("Completed group lacks held-out retraining output")
                if not p.outer_fold.eq(fold).all():
                    raise ValueError("Ablation fold label mismatch")
                safe_details.append({
                    "outer_fold": fold, "group_id": str(member.group_id),
                    "test_carriers_safe": int(groups[0]["test_carriers_safe"]),
                    "test_carriers_exact_vj": int(groups[0]["test_carriers_exact_vj"]),
                    "safe_deletion_coverage_fraction":
                        float(groups[0]["safe_deletion_coverage_fraction"]),
                    "safe_test_carrier_fraction": (
                        int(groups[0]["test_carriers_safe"]) /
                        int(groups[0]["test_carriers_exact_vj"])
                        if int(groups[0]["test_carriers_exact_vj"]) > 0 else 0.0)})
                expected = set(test_lookup.loc[test_lookup.outer_fold.eq(fold)].index)
                if set(p.patient_key) != expected:
                    raise ValueError("Ablation held-out patient set is incomplete or leaked")
                joined = p.join(test_lookup, on="patient_key", rsuffix="_locked")
                if not joined.outer_fold_locked.eq(fold).all() or not joined.label.astype(int).eq(
                        joined.recorded_n_binary.astype(int)).all():
                    raise ValueError("Ablation label or outer-fold provenance mismatch")
                pieces.append(p)
            if skipped_members:
                result["ablation_unassessable"] = {
                    "status": "skipped", "members": skipped_members}
                rows.append(result)
                pooled_p.append(1.0)
                continue
            ablation = pd.concat(pieces, ignore_index=True)
            result["safe_deletion"] = {
                "matched_fold_details": safe_details,
                "test_carriers_safe_pooled": sum(x["test_carriers_safe"] for x in safe_details),
                "minimum_fold_coverage_fraction": min(
                    x["safe_deletion_coverage_fraction"] for x in safe_details),
                "minimum_fold_safe_test_carrier_fraction": min(
                    x["safe_test_carrier_fraction"] for x in safe_details),
                "each_matched_fold_has_safe_carrier": all(
                    x["test_carriers_safe"] >= cfg["ablation"][
                        "minimum_safe_test_carriers_per_matched_fold"]
                    for x in safe_details),
            }
            effect = patient_effect(
                ablation, cfg["ablation"]["random_seed"],
                cfg["ablation"]["patient_bootstrap_draws"],
                cfg["ablation"]["patient_signflip_draws"])
            result["ablation"] = effect
            result["pooled_ablation_p"] = effect["pooled_patient_signflip_one_sided_p"]
        rows.append(result)
        pooled_p.append(float(result["pooled_ablation_p"]))
    q = bh(pooled_p)
    for r, adjusted in zip(rows, q):
        r["pooled_ablation_q"] = adjusted
        e = r["ablation"]
        if e is not None:
            needed_folds = math.ceil(cfg["ablation"]["minimum_positive_fold_fraction"] *
                                     e["tested_outer_folds"])
            r["ablation_pass"] = bool(
                e["mean_delta_logloss"] > cfg["ablation"]["primary_mean_logloss_delta_above"] and
                e["patient_bootstrap_ci95"][0] >
                    cfg["ablation"]["primary_patient_bootstrap_lower_above"] and
                e["mean_delta_vs_matched_random"] >
                    cfg["ablation"]["matched_random_mean_difference_above"] and
                e["positive_fold_effects"] >= needed_folds and
                e["positive_seed_effects"] >= cfg["ablation"]["minimum_positive_seed_effects"] and
                r["safe_deletion"]["test_carriers_safe_pooled"] >= cfg["ablation"][
                    "minimum_safe_test_carriers_pooled"] and
                r["safe_deletion"]["each_matched_fold_has_safe_carrier"] and
                r["safe_deletion"]["minimum_fold_coverage_fraction"] >= cfg["ablation"][
                    "minimum_safe_deletion_coverage_fraction"] and
                r["safe_deletion"]["minimum_fold_safe_test_carrier_fraction"] >= cfg[
                    "ablation"]["minimum_safe_test_carrier_fraction_each_matched_fold"] and
                adjusted < cfg["ablation"]["pooled_one_sided_signflip_bh_q_strictly_below"])
        r["final_candidate"] = bool(r["home_pass"] and r["external_pass"] and
                                     r["alignment_pass"] and r["ablation_pass"])
        r["reason"] = "all_locked_gates_pass" if r["final_candidate"] else ",".join(
            ("ablation_unassessable" if name == "ablation" and
              r.get("ablation_unassessable") else name)
            for name in ("home", "external", "alignment", "ablation")
            if not r[name + "_pass"])
    result = {
        "status": "COMPLETE_candidates" if any(r["final_candidate"] for r in rows)
                  else "COMPLETE_no_candidate",
        "endpoint": "recorded N+ exploratory",
        "candidate_claim": any(r["final_candidate"] for r in rows),
        "material_scope": "Only 31/114 explicitly peripheral blood; 83 unverified",
        "external_scope": "165 BioSample proxies previously examined; directional only",
        "rules_sha256": cfg_sha,
        "code_revision_sha256": code_revision_sha,
        "frozen_candidates_sha256": sha256(frozen_path),
        "home_primary_association_sha256": sha256(need[0]),
        "external_projection_receipt_sha256": sha256(need[4]),
        "external_direction_sha256": sha256(need[6]),
        "home_fold_member_matches_sha256": sha256(need[3]),
        "cross_fold_alignment_sha256": sha256(need[2]),
        "home_to_fold_alignment_sha256": sha256(need[1]),
        "model_gate_sha256": sha256(root / "mil/ablation/gate.json"),
        "sensitivities": sensitivities,
        "families": rows,
        "frozen_family_scope": _private_literal('private_ed9ac51b285950ae'),
        "ablation_bh_family_count": len(rows),
        "interpretation": cfg["final_interpretation"],
        "inference_boundary": "post-selection, conditional-on-fitted-folds exploratory p/q and patient bootstrap CI; not an independent confirmatory test",
    }
    out = root / "final"
    passed = [r for r in rows if r["final_candidate"]]
    if passed:
        stop_report(root, "PENDING_card_export", "Final candidate cards are being written")
        result["candidate_cards"] = write_candidate_cards(root, passed)
    atomic_json(out / "decision.json", result)
    flat = [{k: v for k, v in r.items() if k != "ablation"} |
            ({f"ablation_{k}": v for k, v in r["ablation"].items()
              if isinstance(v, (int, float))} if r["ablation"] else {})
            for r in rows]
    pd.DataFrame(flat).to_csv(out / "candidate_gates.tsv", sep="\t", index=False)
    lines = [
        "# Recorded N+ TRB candidate decision", "",
        f"Status: {result['status']}. Final candidate families: {len(passed)} / {len(rows)}.",
        "", _private_literal('private_d0a5b9757ecf6eca'),
        "Each outer training fold still nominated at most five families. Ablation BH covers every frozen Home family.",
        "Primary Home evidence uses pooled BH across all eligible groups and methods.",
        "The deletion test pools unique outer-test patients after averaging the three seed predictions.",
        "Its p, q and patient bootstrap CI are post-selection exploratory and conditional on the fitted folds.",
        "Per-fold significance is diagnostic only. External direction was scored after frozen label-blind projection.",
        _private_literal('private_5f4dbd867601bf8a'),
        "These sensitivity results are descriptive and do not change the primary candidate threshold.",
        f"Post-selection Pgen status: {sensitivities['pgen']['status']}; unconditioned CDR3 AA only, not V/J-adjusted.",
        *([f"Pgen unavailable reason: {sensitivities['pgen']['unavailable_reason']}."]
          if sensitivities["pgen"].get("unavailable_reason") else []),
        "", "Only 31/114 Home records explicitly document peripheral blood; material of 83 is unverified.",
        "Pathologic pN/ypN prefix, collection time and treatment sequence are unverified.",
        "The 165 external BioSample proxies were examined previously and are directional exploratory evidence.",
        "Antigen recognition and mechanisms require further clinical and functional evidence.", "",
        "| Full Home group | Projection | Fold family | Home q | External carriers | Pooled ablation q | Decision |",
        "| --- | --- | --- | ---: | ---: | ---: | --- |",
    ]
    for r in rows:
        lines.append(f"| {r['group_id']} | {r['projection_type']} | {r['canonical_id']} | {r['home_q_pooled']:.4g} | "
                     f"{r['external_carriers']} | {r['pooled_ablation_q']:.4g} | "
                     f"{'candidate' if r['final_candidate'] else r['reason']} |")
    (out / "report.md").write_text("\n".join(lines) + "\n")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    args = parser.parse_args()
    result = run(args.root)
    print(json.dumps({"status": result["status"],
                      "candidate_claim": result["candidate_claim"],
                      "n_families": len(result.get("families", []))}, sort_keys=True))


if __name__ == "__main__":
    main()
