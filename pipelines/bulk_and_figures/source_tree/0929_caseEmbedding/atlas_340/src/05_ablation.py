#!/usr/bin/env python3
"""Train-fold-nominated group deletion with matched random deletion.

This analysis is conditional on a positive all-group held-out gain. Each outer
training set nominates at most five N+-enriched groups without test labels.
Every nominated feature and three prevalence/method matched controls are
deleted and the model is retuned and refitted on the outer training set.
Full-Home q candidates are aligned after nomination and prediction.
"""
from __future__ import annotations
from release_runtime import private_literal as _private_literal, release_path as _release_path

import hashlib
import importlib.util
import json
import os
import time
from pathlib import Path

for name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[name] = "1"

import numpy as np
import pandas as pd
from scipy.stats import fisher_exact

ROOT = Path(_release_path('path_99d8c1c6fde2a6b9'))
RESULTS = ROOT / "results"
KEY = ["aa", "v_gene", "j_gene"]
RANDOM_SEEDS = (20260930, 20261001, 20261002)
MAX_NOMINATIONS = 5


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, sort_keys=True, allow_nan=False) + "\n")
    os.replace(tmp, path)


def write_tsv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(tmp, sep="\t", index=False, float_format="%.12g")
    os.replace(tmp, path)


def pred_module():
    path = ROOT / "src" / "05_predictive.py"
    spec = importlib.util.spec_from_file_location("atlas340_predictive", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def verified_gate() -> tuple[pd.DataFrame, pd.DataFrame, dict, dict]:
    cand_path = ROOT / "groups" / "frozen_home_candidates.tsv"
    cand_receipt_path = ROOT / "groups" / "frozen_home_candidates.json"
    assoc_receipt_path = ROOT / "audit" / "05_association_receipt.json"
    oof_path = ROOT / "prediction" / "05_patient_oof.tsv"
    tune_path = ROOT / "prediction" / "05_inner_tuning.tsv"
    summary_path = ROOT / "prediction" / "05_summary.json"
    oof_receipt_path = ROOT / "prediction" / "05_receipt.json"
    ca = json.loads(cand_receipt_path.read_text())
    ar = json.loads(assoc_receipt_path.read_text())
    pr = json.loads(oof_receipt_path.read_text())
    ps = json.loads(summary_path.read_text())
    if ca.get("status") != "complete" or ca.get("candidate_sha256") != sha(cand_path):
        raise ValueError("Home candidate freeze invalid")
    if ar.get("status") != "complete" or ar.get("candidate_tsv_sha256") != sha(cand_path):
        raise ValueError("Home association receipt invalid")
    if pr.get("status") != "complete" or pr.get("output_sha256", {}).get("oof") != sha(oof_path):
        raise ValueError("Patient OOF receipt invalid")
    if pr.get("output_sha256", {}).get("inner_tuning") != sha(tune_path) or (
        pr.get("output_sha256", {}).get("summary") != sha(summary_path)
    ):
        raise ValueError("Prediction tuning or summary hash invalid")
    candidates = pd.read_csv(cand_path, sep="\t", dtype={"group_id": str})
    required = {"method", "group_id", "adjusted_q_bh_pooled", "adjusted_beta",
                "prevalence_difference", "home_association_candidate"}
    if not required <= set(candidates):
        raise ValueError("Home candidate columns invalid")
    if candidates[["method", "group_id"]].duplicated().any() or (
        not candidates.adjusted_q_bh_pooled.lt(0.05).all()
    ) or (not candidates.adjusted_beta.gt(0).all()) or (
        not candidates.prevalence_difference.gt(0).all()
    ) or (not candidates.home_association_candidate.eq(True).all()):
        raise ValueError("Home candidates violate frozen q and direction rule")
    oof = pd.read_csv(oof_path, sep="\t", dtype={"patient_key": str})
    if len(oof) != 126 or oof.patient_key.duplicated().any():
        raise ValueError("OOF patient coverage invalid")
    return candidates, oof, ps, {
        "source_home_candidates_sha256": sha(cand_path),
        "source_home_candidate_receipt_sha256": sha(cand_receipt_path),
        "source_oof_receipt_sha256": sha(oof_receipt_path),
        "source_oof_sha256": sha(oof_path),
        "source_prediction_summary_sha256": sha(summary_path),
        "source_folds_sha256": sha(ROOT / "folds" / "folds.tsv"),
        "source_manifest_sha256": sha(ROOT / "config" / "cohort_samples.tsv"),
    }


def method_map(scope: str, groups: list[str]) -> dict[str, str]:
    path = ROOT / "groups" / scope / "group_membership.parquet"
    d = pd.read_parquet(path, columns=["group_id", "method"],
                        filters=[("group_id", "in", groups)])
    d = d.drop_duplicates()
    if d.group_id.duplicated().any() or set(d.group_id) != set(groups):
        raise ValueError("Fold group-to-method map invalid")
    return d.set_index("group_id").method.astype(str).to_dict()


def nominate(matrix: pd.DataFrame, y: np.ndarray, train_ids: list[str],
             methods: dict[str, str]) -> pd.DataFrame:
    g = matrix.loc[train_ids]
    plus = y == 1
    minus = y == 0
    np1 = int(plus.sum())
    n0 = int(minus.sum())
    records = []
    for group_id in g.columns:
        x = g[group_id].to_numpy(dtype=int)
        a = int(x[plus].sum())
        b = int(x[minus].sum())
        if a + b < 5 or a + b > len(x) - 5:
            continue
        difference = a / np1 - b / n0
        if difference <= 0:
            continue
        p = fisher_exact([[a, np1 - a], [b, n0 - b]], alternative="greater").pvalue
        if p < 0.05:
            records.append({
                "fold_group_id": group_id,
                "method": methods[group_id],
                "train_carriers": a + b,
                "train_nplus_carriers": a,
                "train_n0_carriers": b,
                "train_prevalence_difference": difference,
                "train_fisher_p_one_sided": float(p),
            })
    if not records:
        return pd.DataFrame(columns=["fold_group_id", "method", "train_carriers",
                                     "train_nplus_carriers", "train_n0_carriers",
                                     "train_prevalence_difference",
                                     "train_fisher_p_one_sided"])
    return pd.DataFrame(records).sort_values(
        ["train_fisher_p_one_sided", "fold_group_id"]
    ).head(MAX_NOMINATIONS).reset_index(drop=True)


def tuning_after_drop(pred, folds: pd.DataFrame, clinical: pd.DataFrame,
                      outer: int, matrix: pd.DataFrame, totals: pd.DataFrame,
                      drop_group: str) -> float:
    chosen = matrix.drop(columns=[drop_group])
    if chosen.shape[1] == 0:
        raise ValueError("Deletion leaves no group model features")
    part = folds.loc[folds.outer_fold.eq(outer) & folds.partition.eq("train")]
    rows = []
    for inner in range(3):
        ids0 = sorted(part.loc[part.inner_fold.ne(inner), "patient_key"])
        ids1 = sorted(part.loc[part.inner_fold.eq(inner), "patient_key"])
        y0 = clinical.loc[ids0].y.to_numpy(dtype=int)
        y1 = clinical.loc[ids1].y.to_numpy(dtype=int)
        b0 = pred.baseline_raw(clinical, totals, ids0)
        b1 = pred.baseline_raw(clinical, totals, ids1)
        x0 = chosen.loc[ids0].to_numpy(dtype=float)
        x1 = chosen.loc[ids1].to_numpy(dtype=float)
        for c in pred.C_GRID:
            p, _ = pred.fit_predict("groups", c, y0, b0, b1, x0, x1)
            rows.append({"C": c, "log_loss": float(pred.patient_loss(y1, p).mean())})
    return pred.select_c(pd.DataFrame(rows))


def predict_after_drop(pred, folds: pd.DataFrame, clinical: pd.DataFrame,
                       outer: int, matrix: pd.DataFrame, totals: pd.DataFrame,
                       drop_group: str, train_ids: list[str],
                       test_ids: list[str]) -> tuple[np.ndarray, float]:
    c = tuning_after_drop(pred, folds, clinical, outer, matrix, totals, drop_group)
    selected = matrix.drop(columns=[drop_group])
    p, _ = pred.fit_predict(
        "groups", c, clinical.loc[train_ids].y.to_numpy(dtype=int),
        pred.baseline_raw(clinical, totals, train_ids),
        pred.baseline_raw(clinical, totals, test_ids),
        selected.loc[train_ids].to_numpy(dtype=float),
        selected.loc[test_ids].to_numpy(dtype=float),
    )
    return p, c


def control_groups(group_id: str, method: str, nominees: set[str],
                   methods: dict[str, str], counts: pd.Series,
                   outer: int, n_train: int) -> list[str]:
    pool = [g for g in counts.index if g != group_id and g not in nominees and
            methods[g] == method and 3 <= counts[g] <= n_train - 3]
    if len(pool) < 3:
        return []
    target = int(counts[group_id])
    pool = sorted(pool, key=lambda g: (abs(int(counts[g]) - target), g))
    # Randomize among the nearest prevalence-matched controls; deterministic seeds.
    near = pool[:max(30, min(100, len(pool)))]
    chosen = []
    for seed in RANDOM_SEEDS:
        rng = np.random.default_rng(seed + outer + int.from_bytes(
            hashlib.sha256(group_id.encode()).digest()[:4], "little"))
        available = [g for g in near if g not in chosen]
        if not available:
            break
        chosen.append(str(rng.choice(available)))
    return chosen if len(chosen) == 3 else []


def members(scope: str, group_ids: list[str]) -> pd.DataFrame:
    if not group_ids:
        return pd.DataFrame(columns=["group_id", "method"] + KEY)
    path = ROOT / "groups" / scope / "group_membership.parquet"
    d = pd.read_parquet(path, columns=["group_id", "method"] + KEY,
                        filters=[("group_id", "in", group_ids)])
    if set(d.group_id) != set(group_ids):
        raise ValueError("Selected group membership incomplete in " + scope)
    return d.drop_duplicates(["group_id"] + KEY)


def align(scope: str, nominations: pd.DataFrame, candidates: pd.DataFrame) -> dict:
    if nominations.empty or candidates.empty:
        return {}
    train_keys = pd.read_parquet(ROOT / "groups" / scope / "clone_nodes.parquet",
                                 columns=KEY).drop_duplicates(KEY)
    fold = members(scope, nominations.fold_group_id.astype(str).tolist())
    full = members("full_home", candidates.group_id.astype(str).tolist())
    fold = fold.merge(train_keys, on=KEY, how="inner")
    full = full.merge(train_keys, on=KEY, how="inner")
    fsets = {gid: set(map(tuple, part[KEY].itertuples(index=False, name=None)))
             for gid, part in fold.groupby("group_id")}
    hsets = {gid: set(map(tuple, part[KEY].itertuples(index=False, name=None)))
             for gid, part in full.groupby("group_id")}
    cand_method = candidates.set_index("group_id").method.to_dict()
    result = {}
    for row in nominations.itertuples(index=False):
        source = fsets.get(row.fold_group_id, set())
        hits = []
        for group_id, target in hsets.items():
            if cand_method[group_id] != row.method:
                continue
            common = len(source & target)
            union = len(source) + len(target) - common
            score = common / union if union else 0
            if common >= 2 and score >= 0.50:
                hits.append((score, common, group_id))
        if hits:
            score, common, gid = max(hits, key=lambda x: (x[0], x[1], x[2]))
            result[row.fold_group_id] = {
                "full_home_group_id": gid, "jaccard_train_nodes": score,
                "shared_train_nodes": common,
            }
    return result


def empty_outputs(status: str, hashes: dict, reason: str) -> None:
    summary = pd.DataFrame(columns=[
        "method", "full_home_group_id", "ablation_stable", "n_nominated_folds",
        "n_favorable_folds", "deletion_loss_mean", "deletion_loss_ci_low",
        "deletion_vs_random_mean", "deletion_vs_random_ci_low",
    ])
    fold = pd.DataFrame(columns=[
        "outer_fold", "fold_group_id", "method", "full_home_group_id",
        "train_fisher_p_one_sided", "jaccard_train_nodes",
        "deletion_loss_mean", "random_deletion_loss_mean",
    ])
    RESULTS.mkdir(parents=True, exist_ok=True)
    summary_path = RESULTS / "05_ablation_summary.tsv"
    fold_path = RESULTS / "05_ablation_fold_tests.tsv"
    write_tsv(summary_path, summary)
    write_tsv(fold_path, fold)
    write_json(RESULTS / "05_ablation_receipt.json", {
        "status": status, "reason": reason,
        **hashes,
        "summary_tsv_sha256": sha(summary_path),
        "fold_tsv_sha256": sha(fold_path),
        "n_outer_folds": 5,
        "max_nominations_per_outer_fold": 0,
        "heldout_labels_used_for_nomination": False,
        "external_labels_used": False,
        "n_ablation_tested": 0,
    })


def run() -> None:
    t0 = time.monotonic()
    candidates, oof, summary, hashes = verified_gate()
    if candidates.empty:
        empty_outputs("STOP_no_q_candidates", hashes, "No positive q<0.05 Home group")
        return
    if not summary.get("meets_predictive_contribution_gate", False):
        empty_outputs("STOP_no_oof_gain", hashes, "All-group patient OOF gain gate failed")
        return
    pred = pred_module()
    folds, clinical = pred.folds_and_clinical()
    oof = oof.set_index("patient_key")
    fold_tests = []
    per_patient = []
    sources = {}
    max_nominated = 0
    for outer in range(5):
        scope = "outer_" + str(outer)
        part = folds.loc[folds.outer_fold.eq(outer)]
        train_ids = sorted(part.loc[part.partition.eq("train"), "patient_key"])
        test_ids = sorted(part.loc[part.partition.eq("test"), "patient_key"])
        matrix, totals, scope_hashes = pred.load_scope(
            folds, scope, set(train_ids) | set(test_ids))
        sources[scope] = scope_hashes
        methods = method_map(scope, list(matrix.columns))
        ytrain = clinical.loc[train_ids].y.to_numpy(dtype=int)
        nominees = nominate(matrix, ytrain, train_ids, methods)
        max_nominated = max(max_nominated, len(nominees))
        if nominees.empty:
            continue
        chosen_c = float(oof.loc[test_ids].groups_C.iloc[0])
        if oof.loc[test_ids].groups_C.nunique() != 1:
            raise ValueError("OOF selected C varies inside outer fold")
        p_refit, _ = pred.fit_predict(
            "groups", chosen_c, ytrain,
            pred.baseline_raw(clinical, totals, train_ids),
            pred.baseline_raw(clinical, totals, test_ids),
            matrix.loc[train_ids].to_numpy(dtype=float),
            matrix.loc[test_ids].to_numpy(dtype=float))
        if not np.allclose(p_refit, oof.loc[test_ids].groups_probability.to_numpy(),
                           rtol=1e-5, atol=1e-5):
            raise ValueError("Unablated fold refit does not reproduce OOF predictions")
        ytest = clinical.loc[test_ids].y.to_numpy(dtype=int)
        loss_full = pred.patient_loss(ytest, p_refit)
        alignments = align(scope, nominees, candidates)
        counts = matrix.loc[train_ids].sum(axis=0).astype(int)
        excluded = set(nominees.fold_group_id.astype(str))
        for row in nominees.itertuples(index=False):
            controls = control_groups(row.fold_group_id, row.method, excluded,
                                      methods, counts, outer, len(train_ids))
            if len(controls) != 3:
                fold_tests.append({
                    "outer_fold": outer, "fold_group_id": row.fold_group_id,
                    "method": row.method,
                    "full_home_group_id": alignments.get(row.fold_group_id, {}).get(
                        "full_home_group_id", ""),
                    "train_fisher_p_one_sided": row.train_fisher_p_one_sided,
                    "jaccard_train_nodes": alignments.get(row.fold_group_id, {}).get(
                        "jaccard_train_nodes", 0),
                    "shared_train_nodes": alignments.get(row.fold_group_id, {}).get(
                        "shared_train_nodes", 0),
                    "deletion_loss_mean": np.nan,
                    "random_deletion_loss_mean": np.nan,
                    "status": "STOP_no_three_matched_controls",
                })
                continue
            p_deleted, c_deleted = predict_after_drop(
                pred, folds, clinical, outer, matrix, totals, row.fold_group_id,
                train_ids, test_ids)
            delta = pred.patient_loss(ytest, p_deleted) - loss_full
            random_deltas = []
            for control in controls:
                p_random, _ = predict_after_drop(
                    pred, folds, clinical, outer, matrix, totals, control,
                    train_ids, test_ids)
                random_deltas.append(pred.patient_loss(ytest, p_random) - loss_full)
            random_mean = np.mean(random_deltas, axis=0)
            match = alignments.get(row.fold_group_id, {})
            gid = match.get("full_home_group_id", "")
            fold_tests.append({
                "outer_fold": outer, "fold_group_id": row.fold_group_id,
                "method": row.method, "full_home_group_id": gid,
                "train_fisher_p_one_sided": row.train_fisher_p_one_sided,
                "train_carriers": row.train_carriers,
                "jaccard_train_nodes": match.get("jaccard_train_nodes", 0),
                "shared_train_nodes": match.get("shared_train_nodes", 0),
                "deleted_C": c_deleted, "matched_random_group_ids": "|".join(controls),
                "deletion_loss_mean": float(delta.mean()),
                "random_deletion_loss_mean": float(random_mean.mean()),
                "deletion_vs_random_mean": float((delta - random_mean).mean()),
                "status": "complete",
            })
            for i, patient in enumerate(test_ids):
                per_patient.append({
                    "patient_key": patient, "outer_fold": outer,
                    "fold_group_id": row.fold_group_id,
                    "full_home_group_id": gid,
                    "method": row.method,
                    "jaccard_train_nodes": match.get("jaccard_train_nodes", 0),
                    "deletion_loss_increase": float(delta[i]),
                    "random_deletion_loss_increase": float(random_mean[i]),
                    "deletion_vs_random": float(delta[i] - random_mean[i]),
                })
        print(json.dumps({"outer_fold": outer, "nominated": len(nominees),
                          "mapped_to_home_q": len(alignments)}), flush=True)
    fold = pd.DataFrame(fold_tests)
    patient = pd.DataFrame(per_patient, columns=[
        "patient_key", "outer_fold", "fold_group_id",
        "full_home_group_id", "method", "jaccard_train_nodes",
        "deletion_loss_increase", "random_deletion_loss_increase",
        "deletion_vs_random",
    ])
    result = []
    for row in candidates.itertuples(index=False):
        subfold = fold.loc[
            fold.full_home_group_id.eq(row.group_id) &
            fold.method.eq(row.method) & fold.status.eq("complete")
        ] if not fold.empty else pd.DataFrame()
        if not subfold.empty:
            subfold = subfold.sort_values(
                ["outer_fold", "jaccard_train_nodes", "train_fisher_p_one_sided"],
                ascending=[True, False, True]).drop_duplicates("outer_fold")
            chosen = set(zip(subfold.outer_fold, subfold.fold_group_id))
            subpatient = patient.loc[
                patient.apply(lambda x: (x.outer_fold, x.fold_group_id) in chosen, axis=1)
            ]
            d = subpatient.deletion_loss_increase.to_numpy(dtype=float)
            v = subpatient.deletion_vs_random.to_numpy(dtype=float)
            deletion = pred.paired_bootstrap(d)
            relative = pred.paired_bootstrap(v)
            nominated = int(subfold.outer_fold.nunique())
            favorable = int(subfold.deletion_loss_mean.gt(0).sum())
            stable = bool(
                nominated >= 4 and favorable >= 4 and
                deletion["ci_low"] > 0 and relative["ci_low"] > 0
            )
            result.append({
                "method": row.method, "full_home_group_id": row.group_id,
                "ablation_stable": stable,
                "n_nominated_folds": nominated,
                "n_favorable_folds": favorable,
                "n_heldout_patients": len(subpatient),
                "deletion_loss_mean": deletion["mean"],
                "deletion_loss_ci_low": deletion["ci_low"],
                "deletion_loss_ci_high": deletion["ci_high"],
                "deletion_vs_random_mean": relative["mean"],
                "deletion_vs_random_ci_low": relative["ci_low"],
                "deletion_vs_random_ci_high": relative["ci_high"],
            })
        else:
            result.append({
                "method": row.method, "full_home_group_id": row.group_id,
                "ablation_stable": False, "n_nominated_folds": 0,
                "n_favorable_folds": 0, "n_heldout_patients": 0,
                "deletion_loss_mean": np.nan, "deletion_loss_ci_low": np.nan,
                "deletion_loss_ci_high": np.nan,
                "deletion_vs_random_mean": np.nan,
                "deletion_vs_random_ci_low": np.nan,
                "deletion_vs_random_ci_high": np.nan,
            })
    RESULTS.mkdir(parents=True, exist_ok=True)
    summary_path = RESULTS / "05_ablation_summary.tsv"
    fold_path = RESULTS / "05_ablation_fold_tests.tsv"
    patient_path = RESULTS / "05_ablation_patient_deltas.parquet"
    write_tsv(summary_path, pd.DataFrame(result))
    write_tsv(fold_path, fold if not fold.empty else pd.DataFrame(columns=[
        "outer_fold", "fold_group_id", "method", "full_home_group_id", "status"]))
    patient.to_parquet(patient_path, index=False)
    write_json(RESULTS / "05_ablation_receipt.json", {
        "status": "complete",
        **hashes,
        "n_outer_folds": 5, "max_nominations_per_outer_fold": max_nominated,
        "heldout_labels_used_for_nomination": False,
        "external_labels_used": False,
        "n_ablation_tested": int(fold.status.eq("complete").sum()) if not fold.empty else 0,
        "n_home_q_candidates": len(candidates),
        "n_ablation_stable": int(pd.DataFrame(result).ablation_stable.sum()),
        "alignment": "training-only AA/V/J member Jaccard >=0.50 and >=2 shared tuples",
        "nomination": "outer training one-sided Fisher P<0.05; positive N+ prevalence; at most five per fold",
        "deletion": "reselect L1 C by three inner folds, then retrain from scratch",
        "matched_random": "three same-method nearest training carrier count features, deterministic seeds",
        "stable_rule": ">=4 nominated/favorable outer folds, patient paired deletion loss CI low>0, deletion-minus-random CI low>0",
        "random_seeds": list(RANDOM_SEEDS),
        "scope_hashes": sources,
        "summary_tsv_sha256": sha(summary_path),
        "fold_tsv_sha256": sha(fold_path),
        "patient_deltas_sha256": sha(patient_path),
        "runtime_seconds": round(time.monotonic() - t0, 3),
        "script_sha256": sha(Path(__file__)),
    })


if __name__ == "__main__":
    run()
