#!/usr/bin/env python3
"""Independent, read-only acceptance audit of the recorded-N+ analysis.

Only audit/99_final_acceptance.json is written. A sealed but unfinished
upstream run is PENDING; inconsistent completed inputs are FAIL.
"""
from __future__ import annotations
from release_runtime import private_literal as _private_literal, release_path as _release_path

import argparse
import hashlib
import importlib.util
import sys
import json
import math
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(_release_path('path_1304218e86d2e596'))
MODELS = ("tcrbert_dualtask_aa_mean", "sceptr_cdr3_only")
BAGS = (2048, 4096)
SEEDS = (17, 29, 43)
OUTER = range(5)
INNER = range(3)
CLIP = 1e-6


class Pending(Exception):
    def __init__(self, paths: list[Path] | str, context: dict | None = None):
        self.paths = [str(p) for p in paths] if isinstance(paths, list) else [paths]
        self.context = context or {}
        super().__init__("Upstream artifacts incomplete")


def require(paths: list[Path]) -> None:
    absent = [path for path in paths if not path.exists()]
    if absent:
        raise Pending(absent)


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read(path: Path) -> dict:
    return json.loads(path.read_text())


def table(path: Path, required: set[str]) -> pd.DataFrame:
    frame = pd.read_csv(path, sep="\t", dtype={"patient_key": str, "group_id": str,
                                                "full_home_group_id": str, "canonical_id": str})
    missing = required - set(frame)
    if missing:
        raise ValueError(f"{path}: missing columns {sorted(missing)}")
    return frame


def check(ok: bool, message: str) -> None:
    if not ok:
        raise ValueError(message)


def truth(value: object) -> bool:
    return str(value).strip().lower() in {"true", "1", "yes"}


def close(a: object, b: object, tol: float = 1e-9) -> bool:
    try:
        return math.isclose(float(a), float(b), rel_tol=tol, abs_tol=tol)
    except (TypeError, ValueError):
        return False


def hash_keys(keys: set[str]) -> str:
    return hashlib.sha256(("\n".join(sorted(keys)) + "\n").encode()).hexdigest()


def bh(values: list[float]) -> list[float]:
    if not values:
        return []
    p = np.asarray(values, dtype=float)
    p = np.where(np.isfinite(p), p, 1.0)
    check(bool(((p >= 0) & (p <= 1)).all()), "Invalid p-value family")
    order = np.argsort(p)
    ranked = p[order] * len(p) / np.arange(1, len(p) + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    result = np.empty(len(p))
    result[order] = np.minimum(ranked, 1)
    return result.tolist()


def loss(labels: np.ndarray, values: np.ndarray) -> np.ndarray:
    p = np.clip(np.asarray(values, dtype=float), CLIP, 1 - CLIP)
    y = np.asarray(labels, dtype=int)
    check(bool(np.isin(y, [0, 1]).all() and np.isfinite(p).all()), "Invalid prediction/label")
    return -(y * np.log(p) + (1 - y) * np.log1p(-p))


def effect(frame: pd.DataFrame, cfg: dict) -> dict:
    """Collapse three seed predictions before patient-level uncertainty."""
    check(not frame.duplicated(["patient_key", "seed"]).any(), "Duplicate ablation patient/seed")
    check(set(frame.seed) == set(SEEDS) and frame.groupby("patient_key").seed.nunique().eq(3).all(),
          "Incomplete ablation seed set")
    check(frame.groupby("patient_key").label.nunique().eq(1).all() and
          frame.groupby("patient_key").outer_fold.nunique().eq(1).all(),
          "Conflicting ablation labels or outer folds")
    block = frame.groupby("patient_key", as_index=False).agg(
        outer_fold=("outer_fold", "first"), label=("label", "first"),
        full_prediction=("full_prediction", "mean"),
        deleted_prediction=("deleted_prediction", "mean"),
        random_deleted_prediction=("random_deleted_prediction", "mean"))
    y = block.label.to_numpy(dtype=int)
    delta = loss(y, block.deleted_prediction) - loss(y, block.full_prediction)
    versus_random = delta - (loss(y, block.random_deleted_prediction) - loss(y, block.full_prediction))
    rng = np.random.default_rng(int(cfg["random_seed"]))
    n = len(block)
    draws = rng.integers(0, n, size=(int(cfg["patient_bootstrap_draws"]), n))
    signs = rng.choice(np.array([-1, 1], dtype=np.int8),
                       size=(int(cfg["patient_signflip_draws"]), n), replace=True)
    p = (1 + int(np.sum((signs * delta).mean(axis=1) >= delta.mean() - 1e-15))) / (len(signs) + 1)
    per_fold = {int(fold): float(delta[block.outer_fold.eq(fold)].mean())
                for fold in sorted(block.outer_fold.unique())}
    per_seed = {}
    for seed, part in frame.groupby("seed"):
        yy = part.label.to_numpy(dtype=int)
        per_seed[int(seed)] = float((loss(yy, part.deleted_prediction) -
                                    loss(yy, part.full_prediction)).mean())
    return {
        "n_unique_test_patients": n,
        "tested_outer_folds": len(per_fold),
        "mean_delta_logloss": float(delta.mean()),
        "patient_bootstrap_ci95": [float(v) for v in np.quantile(delta[draws].mean(axis=1), [0.025, 0.975])],
        "mean_delta_vs_matched_random": float(versus_random.mean()),
        "pooled_patient_signflip_one_sided_p": p,
        "positive_fold_effects": sum(v > 0 for v in per_fold.values()),
        "positive_seed_effects": sum(v > 0 for v in per_seed.values()),
    }


def candidate_predicate(home: dict, external: dict, alignment: dict,
                        ablation: dict | None, safe: dict | None, q: float,
                        cfg: dict, model_pass: bool) -> dict:
    """Recompute every gate from frozen source measurements, never final flags."""
    h, e, a, ar = (cfg[k] for k in ("home", "external", "alignment", "ablation"))
    home_pass = (home["n_present"] >= h["minimum_carriers"] and home["model_status"] == "ok" and
                 home["q"] < h["adjusted_presence_q_strictly_below"] and
                 home["or"] > h["adjusted_presence_or_above"] and
                 home["delta"] > h["presence_delta_above"])
    external_pass = (external["carriers"] >= e["minimum_carriers"] and
                     external["evaluable"] and external["positive"] and external["delta"] > 0)
    alignment_pass = (alignment["pass"] and
                      alignment["matching_folds"] >= a["minimum_matching_outer_folds"] and
                      alignment["method_match"])
    ablation_pass = False
    if ablation is not None and safe is not None:
        ablation_pass = (
            ablation["mean_delta_logloss"] > ar["primary_mean_logloss_delta_above"] and
            ablation["patient_bootstrap_ci95"][0] > ar["primary_patient_bootstrap_lower_above"] and
            ablation["mean_delta_vs_matched_random"] > ar["matched_random_mean_difference_above"] and
            ablation["positive_fold_effects"] >= math.ceil(
                ar["minimum_positive_fold_fraction"] * ablation["tested_outer_folds"]) and
            ablation["positive_seed_effects"] >= ar["minimum_positive_seed_effects"] and
            safe["pooled"] >= ar["minimum_safe_test_carriers_pooled"] and
            safe["each_fold"] and
            safe["min_coverage"] >= ar["minimum_safe_deletion_coverage_fraction"] and
            safe["min_fraction"] >= ar["minimum_safe_test_carrier_fraction_each_matched_fold"] and
            q < ar["pooled_one_sided_signflip_bh_q_strictly_below"])
    return {"home_pass": bool(home_pass), "external_pass": bool(external_pass),
            "alignment_pass": bool(alignment_pass), "ablation_pass": bool(ablation_pass),
            "final_candidate": bool(model_pass and home_pass and external_pass and
                                    alignment_pass and ablation_pass)}


def verify_reported_gates(group_id: str, reported: dict, expected: dict) -> None:
    """Reject a promoted or suppressed family flag after source recomputation."""
    for name, value in expected.items():
        check(truth(reported.get(name)) == value,
              f"Final {group_id} {name} not supported by frozen sources")


def check_seal(root: Path) -> dict:
    paths = [root / "audit/06_final_rule_freeze_v7.json",
             root / "audit/06_oof_ablation_provenance_revision_20260929.json",
             root / "config/final_candidate_rules.json",
             root / "config/mil_decision_rules.json",
             root / "src/05_mil.py", root / "src/06_decision.py"]
    require(paths)
    freeze = read(paths[0])
    revision = read(paths[1])
    check(freeze.get("status") == "locked_preoutcome" and
          freeze.get("all_outcome_outputs_absent") is True and
          freeze.get("rules_sha256") == sha(paths[2]) and
          freeze.get("code_sha256") == sha(paths[5]) and
          freeze.get("stage05_code_sha256") == sha(paths[4]) and
          revision.get("rule_freeze_sha256") == sha(paths[0]) and
          revision.get("stage05_code_sha256") == sha(paths[4]) and
          revision.get("new_code_sha256") == sha(paths[5]),
          "Prospective Stage05/06 v7 seal invalid")
    rules = read(paths[2])
    check(rules.get("frozen_before_outcome_run") is True and
          rules.get("candidate_scope") == "all_home_primary_eligible" and
          rules.get("max_families") is None, "Final rule scope altered")
    return {"freeze_sha256": sha(paths[0]), "stage05_code_sha256": sha(paths[4]),
            "stage06_code_sha256": sha(paths[5]), "rules_sha256": sha(paths[2]),
            "mil_rules_sha256": sha(paths[3])}


def frozen_source_module(root: Path, name: str, expected_sha256: str):
    """Run only the frozen upstream provenance checks, with current file hashes."""
    path = root / "src" / name
    check(sha(path) == expected_sha256, f"Frozen source changed: {name}")
    key = "acceptance_readonly_" + name.replace(".", "_")
    spec = importlib.util.spec_from_file_location(key, path)
    check(spec is not None and spec.loader is not None, f"Cannot load frozen {name}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[key] = module
    spec.loader.exec_module(module)
    return module


def verify_home31(root: Path, labels: dict[str, int]) -> dict:
    """Check the documented-blood association even for the legal empty Home STOP."""
    base = root / "groups/full_home"
    blood = base / "association_verified_blood.tsv"
    receipt_path = blood.with_suffix(".json")
    lock = root / "config/documented_blood31.tsv"
    eligible = base / "eligible_groups_verified_blood.tsv"
    burden = base / "patient_group_burden.parquet"
    burden_receipt = base / "burden_receipt.json"
    integrated = base / "integrated_projection_receipt.json"
    paths = [blood, receipt_path, lock, eligible, burden, burden_receipt, integrated]
    check(all(p.is_file() for p in paths), _private_literal('private_06c11f45f4495fc7'))
    locked = table(lock, {"patient_key", "recorded_n_binary"})
    keys = set(locked.patient_key)
    check(len(locked) == 31 and len(keys) == 31 and keys <= set(labels) and
          locked.recorded_n_binary.astype(int).eq(locked.patient_key.map(labels)).all() and
          sum(labels[k] for k in keys) == 18,
          _private_literal('private_a07ef4748839bf23'))
    source = table(blood, {"group_id", "adjusted_presence_p", "adjusted_presence_q_pooled"})
    tests = table(eligible, {"group_id"})
    rec = read(receipt_path)
    burden_rec = read(burden_receipt)
    check(rec.get("scope") == "full_home" and rec.get("subset") == "verified_blood" and
          rec.get("n_patients") == 31 and rec.get("n_positive") == 18 and
          rec.get("n_negative") == 13 and rec.get("groups_tested") == len(source) and
          rec.get("association_sha256") == sha(blood) and
          rec.get("eligible_groups_sha256") == sha(eligible) and
          rec.get("documented_blood_sha256") == sha(lock) and
          rec.get("folds_sha256") == sha(root / "folds/folds.tsv") and
          rec.get("subset_patient_sha256") == hash_keys(keys) and
          rec.get("burden_sha256") == sha(burden) and
          rec.get("burden_receipt_sha256") == sha(burden_receipt) and
          rec.get("integrated_projection_receipt_sha256") == sha(integrated) and
          burden_rec.get("patient_group_burden_sha256") == sha(burden) and
          not source.group_id.duplicated().any() and
          set(source.group_id) == set(tests.group_id) and
          np.allclose(bh(source.adjusted_presence_p.astype(float).tolist()),
                      source.adjusted_presence_q_pooled.astype(float), rtol=1e-9, atol=1e-9),
          _private_literal('private_b25b4a34a419b5b8'))
    return {"association_sha256": sha(blood), "receipt_sha256": sha(receipt_path),
            "n_patients": 31, "groups_tested": len(source)}


def verify_sensitivity_chain(root: Path, frozen_path: Path, decision: dict,
                             seal: dict) -> dict:
    """Validate all 04c/Pgen attachments and bind them to the final decision."""
    stage06 = frozen_source_module(root, "06_decision.py", seal["stage06_code_sha256"])
    required = stage06.sensitivity_required_paths(root)
    check(all(p.is_file() for p in required),
          "Completed sensitivity or Pgen attachment missing")
    result = stage06.validate_sensitivity_chain(root, frozen_path)
    if decision.get("status") in {"COMPLETE_candidates", "COMPLETE_no_candidate"}:
        check(decision.get("sensitivities") == result,
              _private_literal('private_f80e3ed550dc3d82'))
    return result


def folds_and_labels(root: Path) -> tuple[pd.DataFrame, dict[str, int], dict[int, dict[str, set[str]]]]:
    folds = table(root / "folds/folds.tsv",
                  {"patient_key", "recorded_n_binary", "outer_fold", "partition", "inner_fold"})
    check(set(folds.outer_fold) == set(OUTER) and len(folds) == 570, _private_literal('private_78c270c6d51b3729'))
    check(folds.groupby("outer_fold").patient_key.nunique().eq(114).all() and
          not folds.duplicated(["outer_fold", "patient_key"]).any(), "Duplicate/missing fold patient")
    labels = {}
    for key, block in folds.groupby("patient_key"):
        check(block.recorded_n_binary.nunique() == 1 and len(block) == 5, "Patient fold/label drift")
        labels[str(key)] = int(block.recorded_n_binary.iloc[0])
    check(len(labels) == 114 and sum(labels.values()) == 48, _private_literal('private_b02314bdf83279b9'))
    scope = {}
    all_test = []
    for outer in OUTER:
        block = folds.loc[folds.outer_fold.eq(outer)]
        train = set(block.loc[block.partition.eq("train"), "patient_key"])
        test = set(block.loc[block.partition.eq("test"), "patient_key"])
        check(train | test == set(labels) and not train & test and len(train) > len(test),
              f"Outer {outer} split invalid")
        check(block.loc[block.partition.eq("train"), "inner_fold"].isin(list(INNER)).all(),
              f"Outer {outer} inner assignments incomplete")
        check(block.loc[block.partition.eq("test"), "inner_fold"].isna().all(),
              f"Outer {outer} test entered inner folds")
        scope[outer] = {"train": train, "test": test}
        all_test.extend(test)
    check(len(all_test) == 114 and len(set(all_test)) == 114, "Patient has multiple outer tests")
    return folds, labels, scope


def prediction(root: Path, outer: int, model: str, bag: int, seed: int,
               inner: int | None, expected: set[str], labels: dict[str, int],
               code_sha: str) -> pd.DataFrame:
    base = root / "mil/cv" / f"outer_{outer}" / (
        "final" if inner is None else f"inner_{inner}") / model / f"bag_{bag}" / f"seed_{seed}"
    path, receipt_path = base / "prediction.tsv", base / "receipt.json"
    require([path, receipt_path])
    receipt = read(receipt_path)
    check(receipt.get("status") == "complete" and receipt.get("prediction_sha256") == sha(path) and
          receipt.get("code_sha256") == code_sha and receipt.get("bag_size") == bag and
          receipt.get("seed") == seed, f"Unsealed MIL prediction: {path}")
    frame = table(path, {"patient_key", "label", "prediction", "model", "bag_size", "seed", "outer_fold"})
    check(len(frame) == len(expected) and set(frame.patient_key) == expected and
          not frame.patient_key.duplicated().any() and
          frame.label.astype(int).eq(frame.patient_key.map(labels)).all() and
          frame.model.eq(model).all() and frame.bag_size.eq(bag).all() and
          frame.seed.eq(seed).all() and frame.outer_fold.eq(outer).all(),
          f"MIL prediction patient/label/setting mismatch: {path}")
    if inner is not None:
        check("inner_fold" in frame and frame.inner_fold.eq(inner).all(),
              f"Inner prediction fold mismatch: {path}")
    p = frame.prediction.to_numpy(dtype=float)
    check(bool(np.isfinite(p).all() and ((p >= 0) & (p <= 1)).all()),
          f"MIL prediction nonfinite/out-of-range: {path}")
    return frame


def chosen(inner: pd.DataFrame, labels: dict[str, int],
           model_filter: str | None = None) -> tuple[str, int]:
    data = inner if model_filter is None else inner.loc[inner.model.eq(model_filter)]
    rows = []
    for (model, bag), block in data.groupby(["model", "bag_size"]):
        by = block.groupby("patient_key", as_index=False).prediction.mean()
        check(by.patient_key.nunique() == len(set(inner.patient_key)),
              "Inner comparison patient count differs by setting")
        yy = by.patient_key.map(labels).to_numpy(dtype=int)
        values = loss(yy, by.prediction.to_numpy(dtype=float))
        rows.append((str(model), int(bag), float(values.mean()),
                     float(values.std(ddof=1) / math.sqrt(len(values)))))
    check(len(rows) == (2 if model_filter else 4), "Missing 2048/4096 encoder comparison")
    best = min(rows, key=lambda row: row[2])
    allowed = [row for row in rows if row[2] <= best[2] + best[3] + 1e-12]
    return min(allowed, key=lambda row: (row[1], MODELS.index(row[0]), row[2]))[:2]


def verify_mil(root: Path, seal: dict, scope: dict[int, dict[str, set[str]]],
               labels: dict[str, int]) -> tuple[pd.DataFrame, dict]:
    pooled_path = root / _private_literal('private_4b09056568ab6f20')
    encoder_pooled_path = root / _private_literal('private_ee6d9683e4c22c9b')
    encoder_comparison_path = root / "mil/oof/encoder_comparison.json"
    summary_path = root / "mil/oof/summary.json"
    gate_path = root / "mil/ablation/gate.json"
    require([pooled_path, encoder_pooled_path, encoder_comparison_path,
             summary_path, gate_path] +
            [root / "mil/cv" / f"outer_{o}" / "selection.json" for o in OUTER])
    pooled = table(pooled_path, {"patient_key", "label", "outer_fold", "prediction",
                                 "selected_model", "selected_bag", "group_burden",
                                 "clinical_technical_batch"})
    summary, gate = read(summary_path), read(gate_path)
    check(summary.get("status") == "complete" and summary.get("n_patients") == 114 and
          summary.get(_private_literal('private_365a4bb69941e1fd')) == sha(pooled_path) and
          summary.get(_private_literal('private_2d712dab280e0a2d')) == sha(encoder_pooled_path) and
          summary.get("encoder_comparison_sha256") == sha(encoder_comparison_path) and
          summary.get("code_sha256") == seal["stage05_code_sha256"] and
          gate.get("oof_sha256") == sha(pooled_path) and
          gate.get("decision_rules_sha256") == seal["mil_rules_sha256"],
          "Pooled MIL provenance mismatch")
    check(len(pooled) == 114 and not pooled.patient_key.duplicated().any() and
          set(pooled.patient_key) == set(labels), _private_literal('private_770daacf7e739424'))
    check(pooled.label.astype(int).eq(pooled.patient_key.map(labels)).all(),
          "OOF label mismatch")
    encoder_pooled = table(encoder_pooled_path,
                           {"patient_key", "label", "outer_fold", "model",
                            "bag_size", "prediction"})
    check(len(encoder_pooled) == 114 * len(MODELS) and
          not encoder_pooled.duplicated(["patient_key", "model"]).any() and
          set(encoder_pooled.patient_key) == set(labels) and
          set(encoder_pooled.model) == set(MODELS),
          "Pooled encoder OOF incomplete")
    inner_fits = final_fits = 0
    selected_seed_frames = {seed: [] for seed in SEEDS}
    folds_hash = sha(root / "folds/folds.tsv")
    def pca_scope(model_name: str, outer_index: int, inner_index: int | None,
                  patients: set[str]) -> None:
        path = root / "pca" / model_name / f"outer_{outer_index}"
        if inner_index is not None:
            path = path / f"inner_{inner_index}"
        path = path / "metadata.json"
        require([path])
        metadata = read(path)
        check(metadata.get("model") == model_name and
              metadata.get("outer_fold") == outer_index and
              metadata.get("inner_fold") == inner_index and
              metadata.get("folds_sha256") == folds_hash and
              metadata.get("k_per_patient") == 4096 and
              metadata.get("fit_patient_count") == len(patients) and
              set(metadata.get("fit_patient_ids", [])) == patients,
              f"PCA fit used held-out or unequal-sampling population: {path}")
    folds_source = table(root / "folds/folds.tsv",
                         {"patient_key", "outer_fold", "partition", "inner_fold"})
    for outer in OUTER:
        train, test = scope[outer]["train"], scope[outer]["test"]
        base = root / "mil/cv" / f"outer_{outer}"
        selection_path, inner_path, outer_path = (
            base / "selection.json", base / "inner_predictions.tsv", base / "outer_oof.tsv")
        require([inner_path, outer_path, base / "inner_comparison.tsv", base / "encoder_oof.tsv"])
        selection = read(selection_path)
        check(selection.get("status") == "complete" and selection.get("outer_fold") == outer and
              selection.get("code_sha256") == seal["stage05_code_sha256"] and
              selection.get("inner_predictions_sha256") == sha(inner_path) and
              selection.get("inner_comparison_sha256") == sha(base / "inner_comparison.tsv") and
              selection.get("outer_oof_sha256") == sha(outer_path) and
              selection.get("encoder_oof_sha256") == sha(base / "encoder_oof.tsv"),
              f"Outer {outer} selection/source mismatch")
        source = table(inner_path, {"patient_key", "label", "prediction", "model",
                                    "bag_size", "seed", "outer_fold", "inner_fold"})
        group_desc = root / "groups" / f"outer_{outer}" / "description.json"
        require([group_desc])
        desc = read(group_desc)
        check(set(desc.get("patient_ids", [])) == train and
              desc.get("folds_sha256") == folds_hash,
              f"Outer {outer} group dictionary includes held-out patient")
        for model_name in MODELS:
            pca_scope(model_name, outer, None, train)
        expected_frames = []
        for inner in INNER:
            val = set(folds_source.loc[folds_source.outer_fold.eq(outer) &
                                       folds_source.partition.eq("train") &
                                       folds_source.inner_fold.eq(inner), "patient_key"])
            check(val and val <= train, f"Outer {outer} inner {inner} val leakage")
            inner_train = train - val
            for model in MODELS:
                pca_scope(model, outer, inner, inner_train)
                for bag in BAGS:
                    for seed in SEEDS:
                        expected_frames.append(prediction(root, outer, model, bag, seed, inner,
                                                          val, labels, seal["stage05_code_sha256"]))
                        inner_fits += 1
        concatenated = pd.concat(expected_frames, ignore_index=True)
        fields = ["patient_key", "label", "prediction", "model", "bag_size", "seed", "outer_fold", "inner_fold"]
        order = ["model", "inner_fold", "bag_size", "seed", "patient_key"]
        check(len(source) == len(concatenated) and
              source[fields].sort_values(order).reset_index(drop=True).equals(
                  concatenated[fields].sort_values(order).reset_index(drop=True)),
              f"Outer {outer} inner predictions do not match fit files")
        model, bag = chosen(source, labels)
        check(selection.get("selected_model") == model and selection.get("selected_bag") == bag,
              f"Outer {outer} choice differs from patient-level one-SE comparison")
        encoder = selection.get("encoder_choices")
        check(isinstance(encoder, dict) and set(encoder) == set(MODELS) and
              all(encoder[m] == chosen(source, labels, m)[1] for m in MODELS),
              f"Outer {outer} encoder 2048/4096 choice changed")
        comparisons = table(base / "inner_comparison.tsv", {"model", "bag_size", "log_loss", "se", "patients"})
        check(len(comparisons) == 4 and not comparisons.duplicated(["model", "bag_size"]).any() and
              set(zip(comparisons.model, comparisons.bag_size)) ==
              {(m, b) for m in MODELS for b in BAGS},
              f"Outer {outer} comparison does not contain both bags for each model")
        for row in comparisons.itertuples():
            block = source.loc[source.model.eq(row.model) & source.bag_size.eq(row.bag_size)]
            by = block.groupby("patient_key", as_index=False).prediction.mean()
            yy = by.patient_key.map(labels).to_numpy(dtype=int)
            values = loss(yy, by.prediction.to_numpy(dtype=float))
            check(close(row.log_loss, values.mean()) and
                  close(row.se, values.std(ddof=1) / math.sqrt(len(values))) and
                  int(row.patients) == len(train), "Inner comparison metrics disagree with predictions")
        fold = pooled.loc[pooled.outer_fold.eq(outer)]
        out = table(outer_path, {"patient_key", "prediction", "selected_model", "selected_bag"})
        check(len(fold) == len(test) and set(fold.patient_key) == test and
              out.sort_values("patient_key").reset_index(drop=True).equals(
                  fold.sort_values("patient_key").reset_index(drop=True)),
              f"Outer {outer} pooled OOF source/heldout patient mismatch")
        final_pairs = {(model, bag)} | {(m, int(encoder[m])) for m in MODELS}
        final_frames = {}
        for candidate_model, candidate_bag in final_pairs:
            for seed in SEEDS:
                final_frames[(candidate_model, candidate_bag, seed)] = prediction(
                    root, outer, candidate_model, candidate_bag, seed, None, test,
                    labels, seal["stage05_code_sha256"])
                final_fits += 1
        encoder_rows = table(base / "encoder_oof.tsv",
                             {"patient_key", "label", "outer_fold", "model",
                              "bag_size", "prediction"})
        check(len(encoder_rows) == len(test) * len(MODELS) and
              encoder_rows[["patient_key", "model"]].duplicated().sum() == 0 and
              set(encoder_rows.patient_key) == test and
              encoder_rows.label.astype(int).eq(encoder_rows.patient_key.map(labels)).all() and
              encoder_rows.outer_fold.eq(outer).all(),
              f"Outer {outer} encoder OOF patient source invalid")
        pooled_encoder_fold = encoder_pooled.loc[encoder_pooled.outer_fold.eq(outer)]
        encoder_fields = ["patient_key", "label", "outer_fold", "model", "bag_size", "prediction"]
        check(encoder_rows[encoder_fields].sort_values(["model", "patient_key"]).reset_index(drop=True).equals(
                  pooled_encoder_fold[encoder_fields].sort_values(["model", "patient_key"]).reset_index(drop=True)),
              f"Outer {outer} encoder OOF differs from pooled source")
        for encoder_model in MODELS:
            encoder_bag = int(encoder[encoder_model])
            block = encoder_rows.loc[encoder_rows.model.eq(encoder_model)].set_index("patient_key")
            seed_concat = pd.concat(
                [final_frames[(encoder_model, encoder_bag, seed)] for seed in SEEDS])
            encoder_mean = seed_concat.groupby("patient_key").prediction.mean()
            check(block.bag_size.eq(encoder_bag).all() and
                  np.allclose(block.loc[encoder_mean.index, "prediction"],
                              encoder_mean, rtol=1e-10, atol=1e-12),
                  f"Outer {outer} {encoder_model} OOF is not three-seed mean")
        for seed in SEEDS:
            rec = read(root / "mil/cv" / f"outer_{outer}" / "final" / model /
                       f"bag_{bag}" / f"seed_{seed}" / "receipt.json")
            check(rec.get("n_train") == len(train) and rec.get("n_val") == 0 and
                  rec.get("shuffle_scope") == f"outer{outer}-final-seed{seed}",
                  f"Outer {outer} final model train scope drift")
        for seed in SEEDS:
            selected_seed_frames[seed].append(
                final_frames[(model, bag, seed)][["patient_key", "prediction"]])
        means = pd.concat([final_frames[(model, bag, seed)] for seed in SEEDS])
        means = means.groupby("patient_key").prediction.mean()
        check(np.allclose(fold.set_index("patient_key").loc[means.index, "prediction"],
                            means, atol=1e-12, rtol=1e-10),
              f"Outer {outer} OOF differs from three full final fits")
    check(inner_fits == 180 and final_fits >= 30, "Incomplete 5x3x2x2x3 inner/final fits")
    rules = read(root / "config/mil_decision_rules.json")["model_gate"]
    y = pooled.label.to_numpy(dtype=int)
    model = pooled.prediction.to_numpy(dtype=float)
    comparator_results = {}
    unavailable = False
    seed_predictions = {}
    for seed in SEEDS:
        frame = pd.concat(selected_seed_frames[seed], ignore_index=True)
        check(len(frame) == 114 and not frame.patient_key.duplicated().any() and
              set(frame.patient_key) == set(labels), "Selected per-seed OOF incomplete")
        seed_predictions[seed] = frame.set_index("patient_key").loc[
            pooled.patient_key, "prediction"].to_numpy(dtype=float)
    for field in ("group_burden", "clinical_technical_batch"):
        comparison = pooled[field].to_numpy(dtype=float)
        if not np.isfinite(comparison).all():
            unavailable = True
            continue
        delta = loss(y, comparison) - loss(y, model)
        rng = np.random.default_rng(20260929)
        draws = rng.integers(0, len(y), size=(5000, len(y)))
        ci_low = float(np.quantile(delta[draws].mean(axis=1), 0.025))
        folds_positive = sum(
            float((loss(block.label, block[field]) - loss(block.label, block.prediction)).mean()) > 0
            for _, block in pooled.groupby("outer_fold"))
        seed_gains = {seed: float((loss(y, comparison) -
                                  loss(y, seed_predictions[seed])).mean())
                      for seed in SEEDS}
        comparator_results[field] = {
            "ci_low": ci_low, "positive_folds": folds_positive,
            "positive_seeds": sum(v > 0 for v in seed_gains.values())}
        if gate.get("status") == "pass":
            gate_ci = gate.get("paired_gain_vs_" + field, {}).get("bootstrap_95ci", [None])[0]
            check(close(gate_ci, ci_low, 1e-8) and
                  all(close(gate.get("seed_gain_by_comparator", {}).get(str(seed), {}).get(field),
                            seed_gains[seed], 1e-8) for seed in SEEDS),
                  f"MIL gate {field} effect differs from original held-out predictions")
    check(gate.get("status") in {"pass", "blocked"}, "MIL gate status invalid")
    pass_expected = not unavailable and all(
        v["ci_low"] > rules["pooled_patient_bootstrap_lower_gt"] and
        v["positive_folds"] >= rules["positive_outer_folds_min"] and
        v["positive_seeds"] >= rules["positive_seeds_min"]
        for v in comparator_results.values())
    check((gate["status"] == "pass") == pass_expected,
          "MIL gate pass/blocked status differs from independently recomputed full OOF")
    return pooled, {"status": gate["status"], "inner_fits": inner_fits,
                    "final_fits": final_fits, "comparators": comparator_results}


def verify_projection_cache(root: Path, model: str, outer: int,
                            inner: int | None, dim: int) -> str:
    """Hash current projected-bank bytes against its PCA/source receipt."""
    name = f"outer_{outer}" if inner is None else f"outer_{outer}_inner_{inner}"
    base = root / "mil/projections" / model / f"{name}_d{dim}"
    cache, receipt_path = base.with_suffix(".npy"), base.with_suffix(".json")
    pca_base = root / "pca" / model / f"outer_{outer}"
    if inner is not None:
        pca_base /= f"inner_{inner}"
    raw = root / "embeddings" / model / "bank_fp16.npy"
    raw_meta = raw.with_name("meta.json")
    paths = [cache, receipt_path, pca_base / "pca.npz",
             pca_base / "metadata.json", pca_base / "stability.json",
             raw, raw_meta]
    check(all(path.is_file() for path in paths),
          f"Completed {model} projection/PCA attachment missing: {name}")
    receipt = read(receipt_path)
    meta = read(raw_meta)
    stat = raw.stat()
    expected_stamp = {"path": str(raw.resolve()), "bytes": stat.st_size,
                      "mtime_ns": stat.st_mtime_ns}
    bank = np.load(cache, mmap_mode="r", allow_pickle=False)
    expected_shape = (int(meta["bank_shape"][0]), dim)
    check(receipt.get("status") == "complete" and
          receipt.get("projection_sha256") == sha(cache) and
          receipt.get("projection_size_bytes") == cache.stat().st_size and
          receipt.get("projection_dtype") == "float16" and
          receipt.get("nonfinite_values") == 0 and
          receipt.get("dim") == dim and
          receipt.get("pca_sha256") == sha(pca_base / "pca.npz") and
          receipt.get("scope_pca_stability_sha256") == sha(pca_base / "stability.json") and
          receipt.get("pca_metadata_sha256") == sha(pca_base / "metadata.json") and
          receipt.get("raw_bank_meta_sha256") == sha(raw_meta) and
          receipt.get("raw_bank_stamp") == expected_stamp and
          tuple(bank.shape) == expected_shape and bank.dtype == np.float16,
          f"Projected bank current bytes/PCA source mismatch: {cache}")
    return sha(cache)


def verify_mil_source_chain(root: Path, seal: dict,
                            scope: dict[int, dict[str, set[str]]],
                            labels: dict[str, int]) -> dict:
    """Re-run frozen fit fingerprints and independently hash banks/PCA caches."""
    stage05 = frozen_source_module(root, "05_mil.py", seal["stage05_code_sha256"])
    integrity_path = root / "mil/bank_integrity.json"
    check(integrity_path.is_file(), "Completed MIL bank integrity receipt missing")
    integrity = read(integrity_path)
    check(integrity.get("status") == "complete" and
          set(integrity.get("models", {})) == set(MODELS),
          "MIL bank integrity receipt incomplete")
    raw_hashes = {}
    for model in MODELS:
        raw = root / "embeddings" / model / "bank_fp16.npy"
        meta_path = raw.with_name("meta.json")
        check(raw.is_file() and meta_path.is_file(),
              f"Completed embedding bank attachment missing: {model}")
        meta = read(meta_path)
        record = integrity["models"][model]
        digest = sha(raw)
        check(meta.get("status") == "complete" and
              digest == meta.get("bank_sha256") == record.get("bank_sha256") and
              record.get("bank_meta_sha256") == sha(meta_path),
              f"Embedding bank current bytes differ from MIL source seal: {model}")
        raw_hashes[model] = digest
    manifest = stage05.read_manifest(root)
    folds = stage05.read_folds(root, manifest)
    stage_labels = stage05.read_home_labels(root, manifest)
    check(stage_labels == labels, _private_literal('private_d78517ec0a6bc370'))
    projection_hashes = {}
    for outer in OUTER:
        for model in MODELS:
            for inner in (None, *INNER):
                pca_base = root / "pca" / model / f"outer_{outer}"
                if inner is not None:
                    pca_base /= f"inner_{inner}"
                required = [pca_base / name for name in
                            ("pca.npz", "selection.json", "metadata.json", "stability.json")]
                check(all(path.is_file() for path in required),
                      f"Completed {model} PCA source attachment missing: outer {outer}, inner {inner}")
                metadata, stability, choice = (read(pca_base / name) for name in
                                               ("metadata.json", "stability.json", "selection.json"))
                fit = scope[outer]["train"]
                if inner is not None:
                    fold_rows = folds.loc[folds.outer_fold.eq(outer) &
                                          folds.partition.eq("train") &
                                          folds.inner_fold.eq(inner)]
                    fit = fit - set(fold_rows.patient_key.astype(str))
                expected_primary = {name: sha(pca_base / name)
                                    for name in ("pca.npz", "selection.json", "metadata.json")}
                check(stability.get("primary_artifact_sha256s") == expected_primary and
                      metadata.get("model") == model and
                      metadata.get("outer_fold") == outer and
                      metadata.get("inner_fold") == inner and
                      set(metadata.get("fit_patient_ids", [])) == fit and
                      metadata.get("train_patient_sha256") == hash_keys(fit) and
                      metadata.get("folds_sha256") == sha(root / "folds/folds.tsv") and
                      metadata.get("bank_sha256_from_meta") == raw_hashes[model] and
                      stability.get("primary_selected_dim") == choice.get("selected_dim"),
                      f"PCA current bytes or train scope mismatch: {pca_base}")
                pca = stage05.pca_spec(root, model, outer, inner,
                                       int(read(root / "embeddings" / model / "meta.json")["bank_shape"][1]))
                if not pca.raw_fallback:
                    key = f"{model}/outer_{outer}" + ("" if inner is None else f"/inner_{inner}")
                    projection_hashes[key] = verify_projection_cache(
                        root, model, outer, inner, int(pca.dim))
        selection = read(root / "mil/cv" / f"outer_{outer}" / "selection.json")
        stage05.validate_outer_output_provenance(root, outer, selection,
                                                  folds, stage_labels)
    return {"raw_bank_sha256": raw_hashes, "projected_bank_sha256": projection_hashes}


def verify_home_family(root: Path, cfg: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    frozen_path = root / "groups/frozen_home_candidates.tsv"
    eligible_path = root / "groups/eligible_home_candidates.tsv"
    source_path = root / "groups/full_home/association_all_recorded_home_exploratory.tsv"
    require([frozen_path, frozen_path.with_suffix(".json"), eligible_path,
             source_path, source_path.with_suffix(".json")])
    frozen_receipt = read(frozen_path.with_suffix(".json"))
    source_receipt = read(source_path.with_suffix(".json"))
    check(frozen_receipt.get("candidate_sha256") == sha(frozen_path) and
          frozen_receipt.get("eligible_home_candidates_sha256") == sha(eligible_path) and
          frozen_receipt.get("external_n_read") is False and
          frozen_receipt.get("source_association_sha256") == sha(source_path) and
          frozen_receipt.get("source_association_receipt_sha256") == sha(source_path.with_suffix(".json")) and
          frozen_receipt.get("candidate_truncation") is False and
          frozen_receipt.get("max_frozen_candidates") is None,
          "Home frozen candidate source/label-blind receipt mismatch")
    check(sha(frozen_path) == sha(eligible_path), "Frozen Home family set differs from full eligible file")
    source = table(source_path, {"group_id", "adjusted_presence_p", "adjusted_presence_q_pooled",
                                 "adjusted_presence_or", "presence_delta", "n_present",
                                 "adjusted_model_status"})
    frozen = table(frozen_path, {"group_id", "method", "projection_type",
                                 "home_q_pooled", "home_adjusted_presence_or",
                                 "home_presence_delta"})
    check(not source.group_id.duplicated().any() and not frozen.group_id.duplicated().any(),
          "Duplicate Home group test")
    check(source_receipt.get("association_sha256") == sha(source_path) and
          source_receipt.get("n_patients") == 114 and
          source_receipt.get("n_positive") == 48 and
          source_receipt.get("n_negative") == 66 and
          source_receipt.get("groups_tested") == len(source) and
          source_receipt.get("min_present") == 10 and
          source_receipt.get("min_absent") == 10,
          "Home association receipt/test population mismatch")
    q = bh(source.adjusted_presence_p.astype(float).tolist())
    check(np.allclose(q, source.adjusted_presence_q_pooled.astype(float),
                      rtol=1e-9, atol=1e-9), "Home pooled BH family differs from complete test table")
    h = cfg["home"]
    selected = source.loc[source.adjusted_model_status.eq("ok") &
                          source.n_present.ge(h["minimum_carriers"]) &
                          source.adjusted_presence_q_pooled.lt(h["adjusted_presence_q_strictly_below"]) &
                          source.adjusted_presence_or.gt(h["adjusted_presence_or_above"]) &
                          source.presence_delta.gt(h["presence_delta_above"])]
    check(set(selected.group_id) == set(frozen.group_id) and
          frozen_receipt.get("n_candidates") == len(frozen),
          "Frozen Home candidates are incomplete or include ineligible groups")
    if not frozen.empty:
        linked = frozen.merge(source[["group_id", "adjusted_presence_q_pooled",
                                      "adjusted_presence_or", "presence_delta"]],
                              on="group_id", validate="one_to_one")
        for f, s in (("home_q_pooled", "adjusted_presence_q_pooled"),
                     ("home_adjusted_presence_or", "adjusted_presence_or"),
                     ("home_presence_delta", "presence_delta")):
            check(np.allclose(linked[f].astype(float), linked[s].astype(float),
                              rtol=1e-9, atol=1e-9), f"Frozen Home {f} differs from source")
    return frozen, source


def verify_external_frozen_sources(root: Path, frozen: pd.DataFrame,
                                   projection: dict) -> dict:
    """Bind each label-blind method extension to the scored combined projection."""
    import pyarrow.parquet as pq

    base = root / "groups/full_home"
    frozen_path = root / "groups/frozen_home_candidates.tsv"
    home_membership = base / "group_membership.parquet"
    combined = base / "external_frozen_membership.parquet"
    combined_receipt = base / "external_frozen_membership_receipt.json"
    check(combined.is_file() and combined_receipt.is_file(),
          "Completed external frozen membership attachment missing")
    manifest = table(root / "config/cohort_samples.tsv", {"cohort", "patient_key"})
    patients = set(manifest.loc[manifest.cohort.eq("external"), "patient_key"])
    check(len(patients) == 165, "External frozen projection identity set invalid")
    identity_hash = hash_keys(patients)
    group_ids = set(frozen.group_id.astype(str))
    method_records = {}
    for method, part in frozen.groupby("method"):
        check(method in {"gliph2", "tcrdist3"}, "Unknown external frozen method")
        prefix = "gliph2" if method == "gliph2" else "tcrdist"
        path = base / f"{prefix}_external_frozen_membership.parquet"
        receipt_path = base / f"{prefix}_external_frozen_projection_receipt.json"
        check(path.is_file() and receipt_path.is_file(),
              f"Completed {method} external extension attachment missing")
        rec = read(receipt_path)
        method_groups = set(part.group_id.astype(str))
        check(rec.get("status") == "complete_external_frozen_scope" and
              rec.get("scope") == "full_home" and
              rec.get("projection_population") == "external" and
              rec.get("external_label_read") is False and
              rec.get("method") == method and
              rec.get("membership_sha256") == sha(path) and
              rec.get("frozen_candidates_sha256") == sha(frozen_path) and
              rec.get("frozen_candidate_receipt_sha256") == sha(frozen_path.with_suffix(".json")) and
              rec.get("home_membership_sha256") == sha(home_membership) and
              rec.get("external_identity_sha256") == identity_hash and
              rec.get("group_ids_sha256") == hash_keys(method_groups) and
              rec.get("manifest_identity_sha256") == projection.get("manifest_identity_sha256"),
              f"External frozen {method} extension source/label seal invalid")
        method_records[method] = {
            "membership_sha256": sha(path), "receipt_sha256": sha(receipt_path),
            "group_ids_sha256": hash_keys(method_groups)}
    combined_rec = read(combined_receipt)
    check(combined_rec.get("status") == "complete_external_frozen_scope" and
          combined_rec.get("scope") == "full_home" and
          combined_rec.get("projection_population") == "external" and
          combined_rec.get("external_label_read") is False and
          combined_rec.get("home_membership_sha256") == sha(home_membership) and
          combined_rec.get("frozen_candidates_sha256") == sha(frozen_path) and
          combined_rec.get("frozen_candidate_receipt_sha256") == sha(frozen_path.with_suffix(".json")) and
          combined_rec.get("external_identity_sha256") == identity_hash and
          combined_rec.get("group_ids_sha256") == hash_keys(group_ids) and
          combined_rec.get("methods") == method_records and
          combined_rec.get("membership_sha256") == sha(combined) and
          combined_rec.get("n_membership_rows") == pq.ParquetFile(combined).metadata.num_rows and
          combined_rec.get("n_frozen_groups") == len(group_ids) and
          projection.get("external_frozen_membership_sha256") == sha(combined) and
          projection.get("external_frozen_membership_receipt_sha256") == sha(combined_receipt) and
          projection.get("method_extension_receipts") == method_records and
          projection.get("target_patient_sha256") == identity_hash and
          projection.get("frozen_group_ids_sha256") == hash_keys(group_ids),
          "External combined frozen membership source/label seal invalid")
    return {"combined_sha256": sha(combined),
            "receipt_sha256": sha(combined_receipt),
            "method_extension_receipts": method_records}


def verify_external(root: Path, frozen: pd.DataFrame) -> pd.DataFrame:
    base = root / "groups/full_home"
    projection_path, burden_path = base / "external_projection_receipt.json", base / "patient_group_burden_external.parquet"
    score_path, direction_path = base / "external_direction.json", base / "external_direction.tsv"
    require([projection_path, burden_path, score_path, direction_path,
             base / "burden_receipt_external.json"])
    projection, score = read(projection_path), read(score_path)
    verify_external_frozen_sources(root, frozen, projection)
    frozen_path = root / "groups/frozen_home_candidates.tsv"
    home_membership = base / "group_membership.parquet"
    home_burden = base / "patient_group_burden.parquet"
    home_burden_receipt = base / "burden_receipt.json"
    require([home_membership, home_burden, home_burden_receipt])
    check(projection.get("status") == "complete" and projection.get("external_n_read") is False and
          projection.get("frozen_candidates_sha256") == sha(frozen_path) and
          projection.get("frozen_candidate_receipt_sha256") == sha(frozen_path.with_suffix(".json")) and
          projection.get("membership_sha256") == sha(home_membership) and
          projection.get("home_burden_sha256") == sha(home_burden) and
          projection.get("home_burden_receipt_sha256") == sha(home_burden_receipt) and
          projection.get("burden_sha256") == sha(burden_path) and
          projection.get("external_burden_receipt_sha256") == sha(base / "burden_receipt_external.json") and
          projection.get("target_proxy_count") == 165 and
          score.get("projection_receipt_sha256") == sha(projection_path) and
          score.get("projection_completed_utc") == projection.get("completed_utc") and
          score.get("external_direction_tsv_sha256") == sha(direction_path) and
          score.get("frozen_candidates_sha256") == sha(frozen_path),
          "External freeze -> label-blind projection -> scoring chain invalid")
    check(datetime.fromisoformat(projection["completed_utc"]) <=
          datetime.fromisoformat(score["scored_utc"]), "External N joined before projection completed")
    manifest = table(root / "config/cohort_samples.tsv",
                     {"cohort", "sample_id", "patient_key", "recorded_n_binary", "identity_basis"})
    identities = manifest[["cohort", "sample_id", "patient_key"]].copy()
    identities["cohort"] = identities.cohort.str.lower()
    identity_sha = hashlib.sha256(identities.sort_values(
        ["cohort", "sample_id", "patient_key"]).to_csv(
            sep="\t", index=False, lineterminator="\n").encode()).hexdigest()
    check(projection.get("manifest_identity_sha256") == identity_sha,
          "External label-blind manifest identity changed after projection")
    labels = manifest.loc[manifest.cohort.eq("external"),
                          ["patient_key", "recorded_n_binary", "identity_basis"]].drop_duplicates()
    check(len(labels) == 165 and not labels.patient_key.duplicated().any() and
          labels.identity_basis.eq("BioSample_proxy_unverified").all() and
          labels.recorded_n_binary.isin([0, 1, "0", "1"]).all(),
          "External 165 proxy identity/recorded-N drift")
    canonical = sorted(labels[["patient_key", "recorded_n_binary"]].astype(str).values.tolist())
    label_sha = hashlib.sha256(json.dumps(canonical, ensure_ascii=False,
                                          separators=(",", ":")).encode()).hexdigest()
    check(score.get("external_label_sha256") == label_sha and
          score.get("n_external_scored_patients") == 165,
          "External score label hash/count mismatch")
    burden = pd.read_parquet(burden_path, columns=["patient_key", "group_id", "presence"])
    burden["patient_key"] = burden.patient_key.astype(str)
    burden["group_id"] = burden.group_id.astype(str)
    check(len(burden) == 165 * len(frozen) and
          not burden.duplicated(["patient_key", "group_id"]).any() and
          set(burden.patient_key) == set(labels.patient_key) and
          set(burden.group_id) == set(frozen.group_id),
          "External projection burden grid incomplete")
    joined = burden.merge(labels[["patient_key", "recorded_n_binary"]],
                          on="patient_key", validate="many_to_one")
    joined["recorded_n_binary"] = joined.recorded_n_binary.astype(int)
    direction = table(direction_path, {"group_id", "n_carriers_external", "presence_delta",
                                       "evaluable", "direction_positive"})
    check(len(direction) == len(frozen) and not direction.group_id.duplicated().any() and
          set(direction.group_id) == set(frozen.group_id),
          "External direction test set differs from frozen Home families")
    for group, part in joined.groupby("group_id"):
        score_row = direction.set_index("group_id").loc[group]
        carriers = int(part.presence.astype(bool).sum())
        delta = (float(part.loc[part.recorded_n_binary.eq(1), "presence"].mean()) -
                 float(part.loc[part.recorded_n_binary.eq(0), "presence"].mean()))
        check(carriers == int(score_row.n_carriers_external) and
              close(delta, score_row.presence_delta) and
              truth(score_row.evaluable) == (carriers >= 5) and
              truth(score_row.direction_positive) == (carriers >= 5 and delta > 0),
              f"External directional score differs from projected burden: {group}")
    return direction


def verify_alignment(root: Path, frozen: pd.DataFrame,
                     scope: dict[int, dict[str, set[str]]]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    base = root / "groups"
    ap, cp, mp = (base / "full_home_to_fold_alignment.tsv",
                  base / "cross_fold_alignment.tsv",
                  base / "full_home_fold_member_matches.tsv")
    require([ap, ap.with_suffix(".json"), cp, mp, base / "fold_nominations.tsv"])
    align = table(ap, {"full_home_group_id", "canonical_id", "alignment_pass",
                       "n_matching_folds", "method"})
    cross = table(cp, {"canonical_id", "outer_fold", "group_id", "method"})
    matched = table(mp, {"full_home_group_id", "canonical_id", "outer_fold",
                         "group_id", "match_pass"})
    receipt = read(ap.with_suffix(".json"))
    check(receipt.get("frozen_candidates_sha256") == sha(root / "groups/frozen_home_candidates.tsv") and
          receipt.get("cross_fold_alignment_sha256") == sha(cp) and
          receipt.get("full_home_fold_member_matches_sha256") == sha(mp) and
          receipt.get("output_sha256") == sha(ap) and
          receipt.get("matching_reference_population") == "outer_fold_training_patient_exact_aa_vj" and
          receipt.get("matching_reference_folds_sha256") == sha(root / "folds/folds.tsv"),
          "Fold alignment source/training reference invalid")
    check(len(align) == len(frozen) and not align.full_home_group_id.duplicated().any() and
          set(align.full_home_group_id) == set(frozen.group_id) and
          not cross.duplicated(["outer_fold", "group_id"]).any(),
          "Ambiguous or incomplete Home/fold alignment")
    nominees = table(base / "fold_nominations.tsv",
                     {"outer_fold", "group_id", "scope", "train_patient_sha256"})
    check(not nominees.duplicated(["outer_fold", "group_id"]).any() and
          nominees.groupby("outer_fold").size().le(5).all(), "More than five fold nominees")
    hashes = receipt.get("matching_reference_train_patient_sha256_by_outer_fold", {})
    for outer in OUTER:
        n = nominees.loc[nominees.outer_fold.eq(outer)]
        check(n.scope.eq(f"outer_{outer}").all() and
              n.train_patient_sha256.eq(hash_keys(scope[outer]["train"])).all(),
              f"Outer {outer} nominee used test patients")
        if str(outer) in hashes:
            desc = read(base / f"outer_{outer}" / "description.json")
            check(hashes[str(outer)] == hash_keys(scope[outer]["train"]) and
                  set(desc.get("patient_ids", [])) == scope[outer]["train"],
                  f"Outer {outer} alignment reference leaked test patient")
    return align, cross, matched


def verify_ablation_sources(root: Path, outer: int, group: str,
                            expected: set[str], labels: dict[str, int],
                            model: str, bag: int, code_sha: str) -> pd.DataFrame:
    group_dir = root / "mil/ablation" / f"outer_{outer}" / hashlib.sha256(group.encode()).hexdigest()[:16]
    frames = []
    for seed in SEEDS:
        original = prediction(root, outer, model, bag, seed, None, expected, labels, code_sha)
        original_receipt = read(root / "mil/cv" / f"outer_{outer}" / "final" / model /
                                f"bag_{bag}" / f"seed_{seed}" / "receipt.json")
        source = [original.rename(columns={"prediction": "full_prediction"})]
        treatment_fingerprints = []
        removal_counts = []
        for treatment in ("group_removed", "matched_random"):
            path = group_dir / f"seed_{seed}_{treatment}.tsv"
            receipt_path = path.with_suffix(".json")
            require([path, receipt_path])
            receipt = read(receipt_path)
            check(receipt.get("status") == "complete" and
                  receipt.get("code_sha256") == code_sha and
                  receipt.get("prediction_sha256") == sha(path) and
                  receipt.get("bag_size") == bag and receipt.get("seed") == seed and
                  receipt.get("n_train") == 114 - len(expected) and
                  receipt.get("n_val") == 0 and
                  receipt.get("epochs_ran") == original_receipt.get("fixed_epochs"),
                  f"Ablation not a complete matched retrain: {path}")
            check(receipt.get("shuffle_scope") == f"outer{outer}-final-seed{seed}" and
                  float(receipt.get("seconds", 0)) > 0 and
                  receipt.get("fingerprint") != original_receipt.get("fingerprint"),
                  f"Ablation train scope/fingerprint invalid: {path}")
            treatment_fingerprints.append(receipt.get("fingerprint"))
            removal_counts.append(receipt.get("removed_patient_aa"))
            frame = table(path, {"patient_key", "label", "prediction", "seed", "treatment", "group_id"})
            check(len(frame) == len(expected) and set(frame.patient_key) == expected and
                  not frame.patient_key.duplicated().any() and
                  frame.label.astype(int).eq(frame.patient_key.map(labels)).all() and
                  frame.seed.eq(seed).all() and frame.treatment.eq(treatment).all() and
                  frame.group_id.eq(group).all(),
                  f"Ablation treatment patient/setting mismatch: {path}")
            pp = frame.prediction.to_numpy(dtype=float)
            check(bool(np.isfinite(pp).all() and ((pp >= 0) & (pp <= 1)).all()),
                  f"Ablation prediction invalid: {path}")
            source.append(frame.rename(columns={"prediction": treatment}))
        check(treatment_fingerprints[0] != treatment_fingerprints[1] and
              removal_counts[0] == removal_counts[1] and
              int(removal_counts[0]) > 0,
              "Ablation and matched random treatment fingerprint/count not distinct/matched")
        merged = source[0][["patient_key", "label", "seed", "full_prediction"]]
        for frame, column in zip(source[1:], ("group_removed", "matched_random")):
            merged = merged.merge(frame[["patient_key", column]], on="patient_key",
                                  validate="one_to_one")
        merged["outer_fold"] = outer
        merged["group_id"] = group
        frames.append(merged.rename(columns={"group_removed": "deleted_prediction",
                                               "matched_random": "random_deleted_prediction"}))
    return pd.concat(frames, ignore_index=True)


def verify_candidate_cards(root: Path, decision: dict,
                           passed: list[dict]) -> dict[str, dict[str, str]]:
    """Compare exported candidate cards and exact members to source rows."""
    import pyarrow.dataset as ds
    import pyarrow.parquet as pq

    cards = decision.get("candidate_cards", [])
    check(isinstance(cards, list) and len(cards) == len(passed),
          "Final candidate card list is incomplete")
    source = root / "groups/full_home/group_membership.parquet"
    check(source.is_file(), "Final candidate membership source missing")
    dataset = ds.dataset(str(source), format="parquet")
    columns = sorted({"group_id", "aa", "v_gene", "j_gene", "seq_idx", "projection_type"})
    check(set(columns) <= set(dataset.schema.names),
          "Candidate source membership lacks exact fields")
    source_hash = sha(source)
    result = {}
    for row, listed in zip(passed, cards):
        gid = str(row["group_id"])
        stem = hashlib.sha256(gid.encode()).hexdigest()[:16]
        folder = root / "final/candidates" / stem
        card_path, members_path = folder / "card.json", folder / "members.parquet"
        first_path = folder / "first_100_members.tsv"
        check(str(listed) == str(card_path) and
              all(path.is_file() for path in (card_path, members_path, first_path)),
              f"Final candidate {gid} card or member attachment missing")
        card = read(card_path)
        expected = {"group_id": gid, "canonical_id": row["canonical_id"],
                    "method": row["method"], "projection_type": row["projection_type"],
                    "family_definition": row["family_definition"],
                    "home_carriers": row["home_carriers"],
                    "home_q_pooled": row["home_q_pooled"],
                    "home_presence_delta": row["home_presence_delta"],
                    "external_carriers": row["external_carriers"],
                    "external_presence_delta": row["external_presence_delta"],
                    "ablation": row["ablation"],
                    "safe_deletion": row.get("safe_deletion")}
        check(all(card.get(key) == value for key, value in expected.items()) and
              card.get("endpoint") == "recorded N+ exploratory" and
              card.get("members_path") == str(members_path) and
              card.get("members_sha256") == sha(members_path) and
              card.get("full_home_membership_sha256") == source_hash and
              card.get("first_100_members_semantics") ==
                  "lexical scan order, not importance ranking",
              f"Final candidate {gid} card values or source SHA invalid")
        member_file = pq.ParquetFile(members_path)
        check(set(columns) <= set(member_file.schema_arrow.names),
              f"Final candidate {gid} exact members lack fields")
        source_batches = iter(dataset.scanner(columns=columns,
                              filter=ds.field("group_id") == gid,
                              batch_size=65536).to_batches())
        member_batches = iter(member_file.iter_batches(batch_size=65536,
                                                        columns=columns))
        left = right = None
        left_pos = right_pos = 0
        n_rows = 0
        first = []
        while True:
            if left is None or left_pos == left.num_rows:
                left = next(source_batches, None)
                left_pos = 0
            if right is None or right_pos == right.num_rows:
                right = next(member_batches, None)
                right_pos = 0
            if left is None or right is None:
                check(left is None and right is None,
                      f"Final candidate {gid} source/export member count differs")
                break
            take = min(left.num_rows - left_pos, right.num_rows - right_pos)
            source_slice = left.slice(left_pos, take)
            member_slice = right.slice(right_pos, take)
            check(source_slice.equals(member_slice),
                  f"Final candidate {gid} exported member rows differ from source")
            if len(first) < 100:
                first.extend(member_slice.select(["aa", "v_gene", "j_gene"])
                             .to_pandas().head(100 - len(first)).fillna("")
                             .astype(str).to_dict("records"))
            n_rows += take
            left_pos += take
            right_pos += take
        check(n_rows > 0 and n_rows == member_file.metadata.num_rows and
              card.get("n_exact_aa_vj_members") == n_rows,
              f"Final candidate {gid} member count invalid")
        first_reported = pd.read_csv(first_path, sep="\t", dtype=str,
                                     keep_default_na=False).to_dict("records")
        check(first_reported == first,
              f"Final candidate {gid} first-100 member attachment differs")
        result[gid] = {"card_sha256": sha(card_path),
                       "members_sha256": sha(members_path),
                       "first_100_sha256": sha(first_path)}
    return result


def verify_complete_families(root: Path, cfg: dict, frozen: pd.DataFrame,
                             source: pd.DataFrame, direction: pd.DataFrame,
                             align: pd.DataFrame, cross: pd.DataFrame,
                             matched: pd.DataFrame, scope: dict,
                             labels: dict[str, int], decision: dict,
                             seal: dict, model_pass: bool) -> dict:
    reported = decision.get("families")
    check(isinstance(reported, list) and len(reported) == len(frozen),
          "Final result omits frozen Home family")
    reported_by_id = {str(row["group_id"]): row for row in reported}
    check(set(reported_by_id) == set(frozen.group_id), "Final family IDs differ from freeze")
    home = source.set_index("group_id")
    external = direction.set_index("group_id")
    aligned = align.set_index("full_home_group_id")
    p_values, computed = [], []
    for item in frozen.itertuples():
        gid = str(item.group_id)
        h, e, a = home.loc[gid], external.loc[gid], aligned.loc[gid]
        row = reported_by_id[gid]
        home_fields = {"n_present": int(h.n_present), "model_status": str(h.adjusted_model_status),
                       "q": float(h.adjusted_presence_q_pooled),
                       "or": float(h.adjusted_presence_or), "delta": float(h.presence_delta)}
        ext_fields = {"carriers": int(e.n_carriers_external), "evaluable": truth(e.evaluable),
                      "positive": truth(e.direction_positive), "delta": float(e.presence_delta)}
        align_fields = {"pass": truth(a.alignment_pass),
                        "matching_folds": int(a.n_matching_folds),
                        "method_match": str(a.method) == str(item.method)}
        check(close(row.get("home_q_pooled"), home_fields["q"]) and
              close(row.get("external_presence_delta"), ext_fields["delta"]) and
              row.get("external_carriers") == ext_fields["carriers"] and
              row.get("n_matching_folds") == align_fields["matching_folds"] and
              row.get("canonical_id") == ("" if pd.isna(a.canonical_id) else str(a.canonical_id)),
              f"Final family source fields changed: {gid}")
        effect_data, safe_data = None, None
        if align_fields["pass"] and align_fields["matching_folds"] >= cfg["alignment"]["minimum_matching_outer_folds"] and align_fields["method_match"]:
            members = cross.loc[cross.canonical_id.eq(str(a.canonical_id))].merge(
                matched.loc[matched.full_home_group_id.eq(gid) &
                            matched.canonical_id.eq(str(a.canonical_id)) &
                            matched.match_pass.map(truth), ["outer_fold", "group_id"]],
                on=["outer_fold", "group_id"], how="inner", validate="one_to_one")
            check(len(members) == align_fields["matching_folds"] and
                  not members.outer_fold.duplicated().any(), f"Family fold membership invalid: {gid}")
            pieces, details, skipped = [], [], False
            for member in members.itertuples():
                outer, group = int(member.outer_fold), str(member.group_id)
                base = root / "mil/ablation" / f"outer_{outer}"
                paths = [base / "summary.json", base / "paired_patient_seed_logloss.tsv",
                         root / "mil/cv" / f"outer_{outer}" / "selection.json"]
                require(paths)
                fold_summary = read(paths[0])
                check(fold_summary.get("status") == "complete" and
                      fold_summary.get("paired_patient_seed_logloss_sha256") == sha(paths[1]) and
                      fold_summary.get("fold_selection_sha256") == sha(paths[2]) and
                      fold_summary.get("oof_sha256") == sha(root / _private_literal('private_4b09056568ab6f20')),
                      f"Fold {outer} ablation summary source mismatch")
                records = [r for r in fold_summary.get("groups", [])
                           if str(r.get("group_id")) == group]
                check(len(records) == 1, f"Fold {outer} group {group} ablation receipt absent")
                if records[0].get("status") == "skipped":
                    skipped = True
                    continue
                check(records[0].get("status") == "complete", "Unknown ablation status")
                selection = read(paths[2])
                raw = verify_ablation_sources(
                    root, outer, group, scope[outer]["test"], labels,
                    selection["selected_model"], int(selection["selected_bag"]),
                    seal["stage05_code_sha256"])
                paired = table(paths[1], {"group_id", "patient_key", "seed",
                                          "full_prediction", "deleted_prediction",
                                          "random_deleted_prediction"})
                paired = paired.loc[paired.group_id.eq(group)].copy()
                check(len(paired) == len(raw) and
                      not paired.duplicated(["patient_key", "seed"]).any(),
                      f"Fold {outer} paired ablation incomplete")
                merged = paired.merge(raw, on=["group_id", "patient_key", "seed"],
                                      suffixes=("_reported", "_source"), validate="one_to_one")
                check(len(merged) == len(raw), f"Fold {outer} paired/source rows differ")
                for field in ("full_prediction", "deleted_prediction", "random_deleted_prediction"):
                    check(np.allclose(merged[f"{field}_reported"].astype(float),
                                      merged[f"{field}_source"].astype(float),
                                      atol=1e-12, rtol=0),
                          f"Fold {outer} {group} paired prediction changed from source")
                pieces.append(raw)
                rec = records[0]
                total = int(rec["test_carriers_exact_vj"])
                safe_count = int(rec["test_carriers_safe"])
                details.append((safe_count, safe_count / total if total else 0.0,
                                float(rec["safe_deletion_coverage_fraction"])))
            if not skipped:
                check(bool(pieces), f"Aligned family {gid} has no retrain results")
                pooled = pd.concat(pieces, ignore_index=True)
                effect_data = effect(pooled, cfg["ablation"])
                safe_data = {
                    "pooled": sum(v[0] for v in details),
                    "each_fold": all(v[0] >= cfg["ablation"]["minimum_safe_test_carriers_per_matched_fold"]
                                     for v in details),
                    "min_fraction": min(v[1] for v in details),
                    "min_coverage": min(v[2] for v in details),
                }
                for field in ("mean_delta_logloss", "mean_delta_vs_matched_random",
                              "pooled_patient_signflip_one_sided_p"):
                    check(close((row.get("ablation") or {}).get(field), effect_data[field], 1e-8),
                          f"Final family {gid} {field} differs from source retrain predictions")
                check((row.get("safe_deletion") or {}).get("test_carriers_safe_pooled") == safe_data["pooled"],
                      f"Final family {gid} safe carrier count mismatch")
        p_values.append(effect_data["pooled_patient_signflip_one_sided_p"] if effect_data else 1.0)
        computed.append((gid, home_fields, ext_fields, align_fields, effect_data, safe_data))
    adjusted = bh(p_values)
    flags = {}
    for (gid, h, e, a, eff, safe), q in zip(computed, adjusted):
        expected = candidate_predicate(h, e, a, eff, safe, q, cfg, model_pass)
        row = reported_by_id[gid]
        verify_reported_gates(gid, row, expected)
        check(close(row.get("pooled_ablation_q"), q), f"Final {gid} pooled BH q mismatch")
        flags[gid] = expected["final_candidate"]
    check(decision.get("ablation_bh_family_count") == len(frozen) and
          truth(decision.get("candidate_claim")) == any(flags.values()) and
          decision.get("status") == ("COMPLETE_candidates" if any(flags.values())
                                    else "COMPLETE_no_candidate"),
          "Final claim/status does not equal independently recomputed whole-family predicate")
    passed = [reported_by_id[gid] for gid in (str(row["group_id"]) for row in reported)
              if flags[gid]]
    cards = verify_candidate_cards(root, decision, passed)
    return {"n_frozen_families": len(frozen), "n_candidates": sum(flags.values()),
            "candidate_group_ids": sorted(g for g, flag in flags.items() if flag),
            "candidate_cards_sha256": cards}


def validate(root: Path) -> dict:
    seal = check_seal(root)
    home_paths = [root / "folds/folds.tsv", root / "groups/frozen_home_candidates.tsv",
                  root / "groups/full_home/association_all_recorded_home_exploratory.tsv",
                  root / "final/decision.json"]
    require(home_paths)
    cfg = read(root / "config/final_candidate_rules.json")
    folds, labels, scope = folds_and_labels(root)
    frozen, source = verify_home_family(root, cfg)
    decision = read(root / "final/decision.json")
    check(str(decision.get("endpoint", "")).lower().startswith("recorded n+") and
          isinstance(decision.get("candidate_claim"), bool),
          "Final endpoint or claim field invalid")
    status = decision.get("status")
    if status in {"PENDING", "PENDING_card_export", None}:
        raise Pending("Final decision is still pending")
    home31 = verify_home31(root, labels)
    if frozen.empty:
        check(status == "STOP_no_Home_association" and
              decision.get("candidate_claim") is False,
              "Empty Home test family must terminate without candidate")
        check(not decision.get("candidate_cards") and not decision.get("families"),
              "Empty Home STOP cannot publish candidate cards or families")
    mil_paths = [root / _private_literal('private_4b09056568ab6f20'),
                 root / "mil/oof/summary.json", root / "mil/ablation/gate.json"]
    mil_paths += [root / "mil/cv" / f"outer_{o}" / "selection.json" for o in OUTER]
    missing_mil = [path for path in mil_paths if not path.exists()]
    if missing_mil:
        context = ({"validated_early_decision": "STOP_no_Home_association",
                    _private_literal('private_1f896173eac6fdce'): home31["association_sha256"]}
                   if frozen.empty else {})
        raise Pending(missing_mil, context)
    pooled, mil = verify_mil(root, seal, scope, labels)
    source_chain = verify_mil_source_chain(root, seal, scope, labels)
    if frozen.empty:
        family = {"n_frozen_families": 0, "n_candidates": 0,
                  "candidate_group_ids": [], "external_not_run_reason": "no Home candidate"}
    else:
        sensitivity = verify_sensitivity_chain(
            root, root / "groups/frozen_home_candidates.tsv", decision, seal)
        direction = verify_external(root, frozen)
        align, cross, matched = verify_alignment(root, frozen, scope)
        if mil["status"] != "pass":
            check(status == "STOP_no_stable_model_gain" and
                  truth(decision.get("candidate_claim")) is False,
                  "Blocked MIL gate cannot yield final candidate")
            family = {"n_frozen_families": len(frozen), "n_candidates": 0,
                      "candidate_group_ids": [], "ablation_not_run_reason": "model gate blocked"}
        else:
            check(status in {"COMPLETE_candidates", "COMPLETE_no_candidate"},
                  "Complete upstream chain lacks final whole-family decision")
            expected_hashes = {
                "rules_sha256": seal["rules_sha256"],
                "frozen_candidates_sha256": sha(root / "groups/frozen_home_candidates.tsv"),
                "home_primary_association_sha256": sha(root / "groups/full_home/association_all_recorded_home_exploratory.tsv"),
                "external_projection_receipt_sha256": sha(root / "groups/full_home/external_projection_receipt.json"),
                "external_direction_sha256": sha(root / "groups/full_home/external_direction.tsv"),
                "home_to_fold_alignment_sha256": sha(root / "groups/full_home_to_fold_alignment.tsv"),
                "cross_fold_alignment_sha256": sha(root / "groups/cross_fold_alignment.tsv"),
                "model_gate_sha256": sha(root / "mil/ablation/gate.json"),
                "home_fold_member_matches_sha256": sha(root / "groups/full_home_fold_member_matches.tsv"),
                "code_revision_sha256": sha(root / "audit/06_oof_ablation_provenance_revision_20260929.json"),
            }
            check(all(decision.get(field) == value for field, value in expected_hashes.items()),
                  "Final decision source hashes changed")
            family = verify_complete_families(root, cfg, frozen, source, direction, align,
                                              cross, matched, scope, labels, decision, seal, True)
    source_paths = {
        "folds": root / "folds/folds.tsv",
        "home_association": root / "groups/full_home/association_all_recorded_home_exploratory.tsv",
        _private_literal('private_9aff820ad794c50c'): root / "groups/full_home/association_verified_blood.tsv",
        _private_literal('private_b41ab16bdc6eaa45'): root / "groups/full_home/association_verified_blood.json",
        "frozen_home_candidates": root / "groups/frozen_home_candidates.tsv",
        "pooled_oof": root / _private_literal('private_4b09056568ab6f20'),
        "pooled_encoder_oof": root / _private_literal('private_ee6d9683e4c22c9b'),
        "model_gate": root / "mil/ablation/gate.json",
        "decision": root / "final/decision.json",
        "external_projection": root / "groups/full_home/external_projection_receipt.json",
        "external_frozen_membership": root / "groups/full_home/external_frozen_membership.parquet",
        "external_frozen_membership_receipt": root / "groups/full_home/external_frozen_membership_receipt.json",
        "pgen_receipt": root / "groups/full_home/sensitivity_pgen_receipt.json",
        "external_direction": root / "groups/full_home/external_direction.tsv",
        "cross_fold_alignment": root / "groups/cross_fold_alignment.tsv",
        "home_to_fold_alignment": root / "groups/full_home_to_fold_alignment.tsv",
    }
    source_hashes = {name: sha(path) for name, path in source_paths.items() if path.exists()}
    if not frozen.empty:
        source_hashes["sensitivities"] = sensitivity
    source_hashes["fold_selections"] = {
        str(outer): sha(root / "mil/cv" / f"outer_{outer}" / "selection.json") for outer in OUTER}
    source_hashes["mil_source_chain"] = source_chain
    source_hashes["ablation_summaries"] = {
        str(outer): sha(root / "mil/ablation" / f"outer_{outer}" / "summary.json")
        for outer in OUTER
        if (root / "mil/ablation" / f"outer_{outer}" / "summary.json").exists()}
    return {"status": "PASS", "endpoint": "recorded N+ exploratory",
            "seal": seal, "source_sha256": source_hashes,
            "counts": {"home_patients": 114, "outer_folds": 5, "inner_folds": 3,
                       **{k: mil[k] for k in ("inner_fits", "final_fits")}, **family},
            "mil_gate": mil["status"], "final_status": status,
            "interpretation_limit": "Recorded N+ association/predictive contribution only; 83 Home material unverified; external BioSample proxies previously examined"}


def run(root: Path) -> dict:
    dest = root / "audit/99_final_acceptance.json"
    try:
        receipt = validate(root)
    except Pending as exc:
        receipt = {"status": "PENDING", "missing_or_unfinished": exc.paths,
                   "n_missing_or_unfinished": len(exc.paths), **exc.context}
    except (ValueError, KeyError, TypeError, OSError, ImportError, RuntimeError,
            AttributeError, IndexError, pd.errors.ParserError) as exc:
        receipt = {"status": "FAIL", "error": f"{type(exc).__name__}: {exc}"}
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(receipt, indent=2, sort_keys=True, ensure_ascii=False) + "\n")
    tmp.replace(dest)
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=ROOT)
    args = parser.parse_args()
    receipt = run(args.root)
    print(json.dumps(receipt, ensure_ascii=False, sort_keys=True))
    raise SystemExit({"PASS": 0, "FAIL": 1, "PENDING": 2}[receipt["status"]])


if __name__ == "__main__":
    main()
