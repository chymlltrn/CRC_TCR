#!/usr/bin/env python3
"""Evaluate the post hoc bag-size sensitivity experiment without outcome-tuned selection.

This module reads independent, size-matched retraining results. It never fits a
MIL model, changes a frozen Stage05 file, or joins external-cohort labels.
"""
from __future__ import annotations
from release_runtime import private_literal as _private_literal, release_path as _release_path

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import hypergeom
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

ROOT = Path(_release_path('path_1304218e86d2e596'))


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2, sort_keys=True, default=str) + "\n")
    tmp.replace(path)


def config_valid(config: dict) -> None:
    if config.get("version") != 2 or config.get("analysis_status") != "post_hoc_sensitivity_after_2048_4096_results_seen":
        raise ValueError("Expected explicit post hoc bag-curve v2 configuration")
    for field, expected in (("models", 2), ("bag_sizes", 5), ("outer_folds", 5),
                            ("inner_folds", 3), ("seeds", 3)):
        values = config.get(field)
        if not isinstance(values, list) or len(values) != expected or len(set(values)) != expected:
            raise ValueError(f"Incomplete or duplicated configuration field {field}")
    if config["bag_sizes"] != sorted(config["bag_sizes"]) or config["bag_sizes"] != [1024, 2048, 4096, 8192, 16384]:
        raise ValueError("Bag curve must contain the five requested sparse scales")
    if config["outer_folds"] != list(range(5)) or config["inner_folds"] != list(range(3)):
        raise ValueError("The frozen patient folds must remain 5 outer x 3 inner")
    if config.get("selection", {}).get("scope") != "inner_train_only_per_outer_fold":
        raise ValueError("Bag selection must use only inner training folds")
    if config.get("curve", {}).get("same_size_training_and_inference") is not True:
        raise ValueError("Every bag size requires same-size training and inference")
    gate = config.get("mil_value_gate", {})
    if (set(gate.get("comparators", [])) != {"clinical_technical_batch", "group_burden"} or
            int(gate.get("paired_patient_bootstrap_draws", 0)) < 1000 or
            not gate.get("proceed_only_if_all_required_conditions_pass")):
        raise ValueError("Patient-level model-value gate is incomplete")


def metric_set(y: np.ndarray, p: np.ndarray) -> dict:
    y = np.asarray(y, dtype=int)
    p = np.clip(np.asarray(p, dtype=float), 1e-6, 1 - 1e-6)
    if len(y) != len(p) or len(y) == 0 or not np.isfinite(p).all():
        raise ValueError("Invalid predictions")
    loss = -(y * np.log(p) + (1 - y) * np.log(1 - p))
    result = {"n": int(len(y)), "log_loss": float(loss.mean()),
              "brier": float(brier_score_loss(y, p)),
              "auroc": float(roc_auc_score(y, p)) if len(np.unique(y)) == 2 else None,
              "ap": float(average_precision_score(y, p)) if len(np.unique(y)) == 2 else None}
    if len(y) >= 20 and len(np.unique(y)) == 2:
        logits = np.log(p / (1 - p)).reshape(-1, 1)
        fit = LogisticRegression(C=1e6, solver="lbfgs", max_iter=1000).fit(logits, y)
        result["calibration_intercept"] = float(fit.intercept_[0])
        result["calibration_slope"] = float(fit.coef_[0, 0])
    else:
        result["calibration_intercept"] = None
        result["calibration_slope"] = None
    return result


def losses(y: np.ndarray, p: np.ndarray) -> np.ndarray:
    p = np.clip(np.asarray(p, dtype=float), 1e-6, 1 - 1e-6)
    y = np.asarray(y, dtype=int)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))


def paired_patient_gain(y: np.ndarray, model_p: np.ndarray, baseline_p: np.ndarray,
                        draws: int, seed: int) -> dict:
    difference = losses(y, baseline_p) - losses(y, model_p)
    if len(difference) == 0 or not np.isfinite(difference).all():
        raise ValueError("Nonfinite paired patient log-loss gain")
    rng = np.random.default_rng(seed)
    boot = np.empty(draws, dtype=float)
    for start in range(0, draws, 256):
        n = min(256, draws - start)
        idx = rng.integers(0, len(difference), size=(n, len(difference)))
        boot[start:start+n] = difference[idx].mean(axis=1)
    return {"mean": float(difference.mean()),
            "bootstrap_95ci": [float(v) for v in np.quantile(boot, [0.025, 0.975])],
            "bootstrap_draws": int(draws), "patients": int(len(difference)),
            "sign_convention": "baseline patient log loss minus MIL patient log loss"}


def fit_path(run_dir: Path, outer: int, inner: int | None, model: str, bag: int, seed: int) -> Path:
    scope = f"inner_{inner}" if inner is not None else "final"
    return run_dir / "fits" / f"outer_{outer}" / scope / model / f"bag_{bag}" / f"seed_{seed}"


def expected_keys(folds: pd.DataFrame, outer: int, inner: int | None) -> set[str]:
    block = folds.loc[folds.outer_fold.eq(outer)]
    if inner is None:
        return set(block.loc[block.partition.eq("test"), "patient_key"])
    return set(block.loc[block.partition.eq("train") & block.inner_fold.eq(inner), "patient_key"])


def read_fit(base: Path, outer: int, inner: int | None, model: str, bag: int, seed: int,
             expected: set[str], label_map: dict[str, int], unique_map: dict[str, int],
             require_coverage: bool, config_sha: str, folds_sha: str, runner_sha: str,
             max_gpu_gib: float, max_cpu_rss_gib: float) -> tuple[pd.DataFrame, dict, dict]:
    pred_path, receipt_path = base / "prediction.tsv", base / "receipt.json"
    receipt = json.loads(receipt_path.read_text())
    if receipt.get("status") != "complete" or receipt.get("prediction_sha256") != sha256(pred_path):
        raise ValueError(f"Incomplete or altered fit: {base}")
    hashes = receipt.get("source_hashes", {})
    if (receipt.get("draw_policy") != "draw-v2-fixed-prefix-32768" or
            not isinstance(receipt.get("fingerprint"), str) or len(receipt["fingerprint"]) != 64 or
            hashes.get("config") != config_sha or hashes.get("folds") != folds_sha or
            hashes.get("runner") != runner_sha or
            not isinstance(hashes.get("projection_meta"), str)):
        raise ValueError(f"Fit provenance or draw policy mismatch: {base}")
    fit_seconds = float(receipt.get("seconds", math.nan))
    infer_seconds = float(receipt.get("inference_seconds", math.nan))
    stop = receipt.get("early_stop_receipt")
    stop_seconds = float(stop.get("seconds", math.nan)) if isinstance(stop, dict) else 0.0
    gpu_gib = float(receipt.get("peak_gpu_gib", math.nan))
    cpu_gib = float(receipt.get("peak_cpu_rss_gib", math.nan))
    if (not all(math.isfinite(x) and x >= 0 for x in
                (fit_seconds, infer_seconds, stop_seconds, gpu_gib, cpu_gib)) or
            gpu_gib > max_gpu_gib or cpu_gib > max_cpu_rss_gib):
        raise ValueError(f"Fit resource receipt missing, nonfinite, or exceeds cap: {base}")
    for key, value in (("model", model), ("bag_size", bag), ("seed", seed), ("outer_fold", outer)):
        if key in receipt and receipt[key] != value:
            raise ValueError(f"Fit receipt {key} mismatch: {base}")
    frame = pd.read_csv(pred_path, sep="\t", dtype={"patient_key": str})
    required = {"patient_key", "label", "prediction", "mc_sd", "outer_fold",
                "model", "bag_size", "seed"}
    if require_coverage:
        required |= {"mc_draws", "coverage_fraction", "coverage_read_fraction", "n_unique_aa"}
    if not required.issubset(frame):
        raise ValueError(f"Fit prediction fields missing at {base}: {sorted(required-set(frame))}")
    if (len(frame) != len(expected) or set(frame.patient_key) != expected or
            frame.patient_key.duplicated().any()):
        raise ValueError(f"Patient scope mismatch: {base}")
    if inner is not None and ("inner_fold" not in frame or not frame.inner_fold.eq(inner).all()):
        raise ValueError(f"Inner fold mismatch: {base}")
    for key, value in (("outer_fold", outer), ("model", model), ("bag_size", bag), ("seed", seed)):
        if not frame[key].eq(value).all():
            raise ValueError(f"Prediction {key} mismatch: {base}")
    if not pd.to_numeric(frame.label, errors="coerce").eq(frame.patient_key.map(label_map)).all():
        raise ValueError(f"Patient labels mismatch: {base}")
    for column in ("prediction", "mc_sd"):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    if (not np.isfinite(frame[["prediction", "mc_sd"]]).all().all() or
            not frame.prediction.between(0, 1).all() or
            frame.mc_sd.lt(0).any()):
        raise ValueError(f"Invalid prediction or MC SD: {base}")
    if require_coverage:
        for column in ("coverage_fraction", "coverage_read_fraction", "n_unique_aa", "mc_draws"):
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
        if (not np.isfinite(frame[["coverage_fraction", "coverage_read_fraction", "n_unique_aa", "mc_draws"]]).all().all() or
                not frame.coverage_fraction.between(0, 1).all() or
                not frame.coverage_read_fraction.between(0, 1).all() or
                frame.mc_draws.lt(8).any() or frame.n_unique_aa.le(0).any() or
                not frame.n_unique_aa.eq(frame.patient_key.map(unique_map)).all()):
            raise ValueError(f"Invalid repertoire coverage or MC draws: {base}")
        expected_unique_coverage = np.minimum(1, bag / frame.n_unique_aa.to_numpy(dtype=float))
        if not np.allclose(frame.coverage_fraction.to_numpy(dtype=float), expected_unique_coverage, atol=1e-9):
            raise ValueError(f"Unique-AA coverage does not equal bag/patient pool: {base}")
    else:
        for column in ("mc_draws", "coverage_fraction", "coverage_read_fraction", "n_unique_aa"):
            if column not in frame:
                frame[column] = np.nan
    stamp = {"prediction": str(pred_path), "prediction_sha256": sha256(pred_path),
             "receipt": str(receipt_path), "receipt_sha256": sha256(receipt_path)}
    return frame, receipt, stamp


def select_inner(inner: pd.DataFrame, config: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    selected = []
    models = config["models"]
    bags = config["bag_sizes"]
    for outer in config["outer_folds"]:
        scoped = inner.loc[inner.outer_fold.eq(outer)]
        for model in models:
            for bag in bags:
                sub = scoped.loc[scoped.model.eq(model) & scoped.bag_size.eq(bag)]
                patients = (sub.groupby("patient_key", as_index=False)
                            .agg(label=("label", "first"), prediction=("prediction", "mean"),
                                 seeds=("seed", "nunique")))
                if patients.empty or not patients.seeds.eq(len(config["seeds"])).all():
                    raise ValueError(f"Incomplete inner pairing: outer {outer}, {model}, {bag}")
                patient_loss = losses(patients.label, patients.prediction)
                rows.append({"outer_fold": outer, "model": model, "bag_size": bag,
                             "inner_patients": len(patients), "inner_log_loss": float(patient_loss.mean()),
                             "inner_patient_se": float(patient_loss.std(ddof=1) / math.sqrt(len(patient_loss))),
                             "inner_brier": metric_set(patients.label, patients.prediction)["brier"]})
        table = pd.DataFrame([row for row in rows if row["outer_fold"] == outer])
        if len(table) != len(models) * len(bags) or table.inner_patients.nunique() != 1:
            raise ValueError(f"Incomplete inner curve in outer fold {outer}")
        best = table.loc[table.inner_log_loss.idxmin()]
        within_one_se = table.loc[table.inner_log_loss.le(best.inner_log_loss + best.inner_patient_se + 1e-12)].copy()
        within_one_se["model_order"] = within_one_se.model.map({model: i for i, model in enumerate(models)})
        chosen = within_one_se.sort_values(["bag_size", "model_order", "inner_log_loss"]).iloc[0]
        selected.append({"outer_fold": outer, "model": chosen.model, "bag_size": int(chosen.bag_size),
                         "inner_log_loss": float(chosen.inner_log_loss),
                         "best_inner_log_loss": float(best.inner_log_loss),
                         "best_inner_patient_se": float(best.inner_patient_se),
                         "eligible_within_one_se": len(within_one_se),
                         "selection_scope": "inner_train_only"})
    return pd.DataFrame(rows), pd.DataFrame(selected)


def seed_mean_oof(final: pd.DataFrame, config: dict) -> pd.DataFrame:
    group_cols = ["patient_key", "label", "outer_fold", "model", "bag_size"]
    result = final.groupby(group_cols, as_index=False).agg(
        prediction=("prediction", "mean"), seed_sd=("prediction", "std"),
        mc_sd=("mc_sd", "mean"), mc_draws=("mc_draws", "mean"),
        coverage_fraction=("coverage_fraction", "mean"),
        coverage_read_fraction=("coverage_read_fraction", "mean"),
        n_unique_aa=("n_unique_aa", "first"), seeds=("seed", "nunique"))
    if not result.seeds.eq(len(config["seeds"])).all():
        raise ValueError("Incomplete final seed pairing")
    result["seed_sd"] = result.seed_sd.fillna(0)
    return result


def compare_to_baselines(data: pd.DataFrame, final_seed: pd.DataFrame,
                         config: dict) -> dict:
    rule = config["mil_value_gate"]
    y = data.label.to_numpy(dtype=int)
    pred = data.prediction.to_numpy(dtype=float)
    if len(data) != 114 or data.patient_key.nunique() != 114:
        raise ValueError(_private_literal('private_719c705267afb202'))
    out = {}
    for name in rule["comparators"]:
        baseline = data[name].to_numpy(dtype=float)
        if not np.isfinite(baseline).all() or np.any((baseline < 0) | (baseline > 1)):
            raise ValueError(f"Incomplete baseline {name}")
        paired = paired_patient_gain(y, pred, baseline,
                                     int(rule["paired_patient_bootstrap_draws"]),
                                     int(rule["paired_patient_bootstrap_seed"]))
        folds = {}
        for outer, block in data.groupby("outer_fold"):
            folds[str(int(outer))] = float((losses(block.label, block[name]) -
                                            losses(block.label, block.prediction)).mean())
        seed_gain = {}
        for seed in config["seeds"]:
            sub = final_seed.loc[final_seed.seed.eq(seed)].merge(
                data[["patient_key", name]], on="patient_key", validate="one_to_one")
            if len(sub) != 114:
                raise ValueError(f"Incomplete final seed OOF: {seed}")
            seed_gain[str(seed)] = float((losses(sub.label, sub[name]) -
                                           losses(sub.label, sub.prediction)).mean())
        out[name] = {"paired_logloss_gain": paired,
                     "baseline_metrics": metric_set(y, baseline),
                     "fold_gain": folds, "seed_gain": seed_gain,
                     "positive_outer_folds": sum(v > 0 for v in folds.values()),
                     "positive_seeds": sum(v > 0 for v in seed_gain.values())}
    return out


def evaluate_utility_gate(metrics: dict, comparisons: dict, config: dict) -> dict:
    'Pure, preconfigured continuation decision on inner-selected [PRIVATE_SUBJECT] OOF.'
    rule = config["mil_value_gate"]
    checks = {}
    for comparator in rule["comparators"]:
        item = comparisons[comparator]
        paired = item["paired_logloss_gain"]
        checks[f"{comparator}_paired_ci_lower_gt_zero"] = bool(
            paired["bootstrap_95ci"][0] > float(rule["paired_logloss_gain_ci_lower_gt"]))
        checks[f"{comparator}_positive_outer_folds"] = bool(
            item["positive_outer_folds"] >= int(rule["min_positive_outer_folds"]))
        checks[f"{comparator}_positive_seeds"] = bool(
            item["positive_seeds"] >= int(rule["min_positive_seeds"]))
    baseline = comparisons["clinical_technical_batch"]["baseline_metrics"]
    if rule.get("brier_must_not_worsen_vs_clinical_technical_batch"):
        checks["brier_no_worse"] = bool(metrics["brier"] <= baseline["brier"])
    if rule.get("auroc_must_not_worsen_vs_clinical_technical_batch"):
        checks["auroc_no_worse"] = bool(metrics["auroc"] is not None and metrics["auroc"] >= baseline["auroc"])
    if rule.get("ap_must_not_worsen_vs_clinical_technical_batch"):
        checks["ap_no_worse"] = bool(metrics["ap"] is not None and metrics["ap"] >= baseline["ap"])
    if rule.get("calibration_and_sampling_report_required"):
        checks["calibration_reported"] = bool(
            metrics.get("calibration_intercept") is not None and
            metrics.get("calibration_slope") is not None and
            math.isfinite(metrics["calibration_intercept"]) and
            math.isfinite(metrics["calibration_slope"]))
        checks["sampling_reported"] = bool(
            all(key in metrics and metrics[key] is not None and math.isfinite(float(metrics[key]))
                for key in ("mean_mc_sd", "mean_seed_sd", "mean_coverage_fraction",
                            "mean_coverage_read_fraction")))
    passed = bool(checks and all(checks.values()))
    return {"status": "pass" if passed else "blocked", "passes": passed,
            "checks": checks, "failed_checks": [key for key, value in checks.items() if not value],
            "scope": _private_literal('private_7de383ac6b93256a'),
            "nominal_p_continuation_allowed": bool(passed and
                config["nominal_p_continuation"]["only_after_mil_value_gate"])}


def analytic_candidate_capture(root: Path, bags: list[int], outer_folds: list[int],
                               folds: pd.DataFrame, unique_map: dict[str, int],
                               final: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Expected safe-group capture under uniform, independent AA bag draws.

    Single-bag probability is exact hypergeometric. K-bag at-least-one and
    expected union coverage assume independent draws; neither is an observed
    hit or a representation fed jointly into the MIL model.
    """
    nomination_path = root / "groups/fold_nominations.tsv"
    if not nomination_path.exists():
        return pd.DataFrame(), {"status": "unavailable", "reason": "Fold nominations missing"}
    nominations = pd.read_csv(nomination_path, sep="\t", dtype={"group_id": str})
    if not {"outer_fold", "group_id"}.issubset(nominations):
        return pd.DataFrame(), {"status": "unavailable", "reason": "Nomination columns missing"}
    rows = []
    sources = {"fold_nominations_sha256": sha256(nomination_path)}
    for outer in outer_folds:
        safe_path = root / "groups" / f"outer_{outer}" / "patient_group_aa_safe.parquet"
        if not safe_path.exists():
            return pd.DataFrame(), {"status": "unavailable", "reason": f"Safe AA map absent: outer {outer}"}
        sources[f"outer_{outer}_safe_map_sha256"] = sha256(safe_path)
        nominated = set(nominations.loc[nominations.outer_fold.eq(outer), "group_id"].astype(str))
        safe = pd.read_parquet(safe_path, columns=["patient_key", "group_id", "seq_idx", "projection_exact"])
        if not safe.projection_exact.eq(True).all():
            raise ValueError(f"Ambiguous candidate AA in safe map outer {outer}")
        safe["patient_key"] = safe.patient_key.astype(str)
        safe["group_id"] = safe.group_id.astype(str)
        test = expected_keys(folds, outer, None)
        counts = (safe.loc[safe.group_id.isin(nominated) & safe.patient_key.isin(test)]
                  .groupby(["group_id", "patient_key"]).seq_idx.nunique())
        oof = final.loc[final.outer_fold.eq(outer)]
        for group_id in sorted(nominated):
            for patient in sorted(test):
                n = int(unique_map[patient])
                m = int(counts.get((group_id, patient), 0))
                if m > n:
                    raise ValueError("Candidate count exceeds patient AA pool")
                for bag in bags:
                    k = min(n, bag)
                    single = float(hypergeom.sf(0, n, m, k)) if m else 0.0
                    matched = oof.loc[oof.patient_key.eq(patient) & oof.bag_size.eq(bag)]
                    for row in matched.itertuples():
                        draws = int(row.mc_draws)
                        if draws < 1:
                            raise ValueError("No inference bags for candidate capture estimate")
                        rows.append({"outer_fold": outer, "group_id": group_id,
                                     "patient_key": patient, "model": row.model,
                                     "seed": int(row.seed), "bag_size": bag,
                                     "patient_unique_aa": n, "safe_group_aa": m,
                                     "mc_draws": draws,
                                     "single_bag_exact_capture_probability": single,
                                     "mc_bag_any_capture_probability": float(-math.expm1(draws * math.log1p(-single))) if single < 1 else 1.0,
                                     "single_bag_unique_aa_coverage": k/n,
                                     "mc_bag_expected_union_unique_aa_coverage": float(-math.expm1(draws * math.log1p(-k/n))) if k < n else 1.0,
                                     "expected_group_aa_in_single_bag": float(k*m/n),
                                     "method": "analytic_uniform_AA_independent_bags_not_observed"})
    result = pd.DataFrame(rows)
    return result, {"status": "complete" if rows else "unavailable",
                    "scope": "outer-heldout patients, outer-training nominations, exact V/J-safe AA",
                    "method": "single-bag exact hypergeometric; K-bag at-least-once 1-(1-p)^K under independent uniform draws; neither is observed hit; union coverage is not one MIL input",
                    "sources": sources, "rows": len(rows)}


def continuation_scale_check(inner: pd.DataFrame, config: dict) -> dict:
    """Cost expansion advice from inner data only; never choose scale from outer OOF."""
    rule = config["expand_32768_rule"]
    candidates = {}
    for model in config["models"]:
        model_result = {}
        both = True
        for low, high in ((4096, 8192), (8192, 16384)):
            fold_gain = {}
            seed_gain = {}
            brier = {}
            for outer in config["outer_folds"]:
                sub = inner.loc[inner.outer_fold.eq(outer) & inner.model.eq(model)]
                low_patient = sub.loc[sub.bag_size.eq(low)].groupby("patient_key", as_index=False).agg(
                    label=("label", "first"), prediction=("prediction", "mean"))
                high_patient = sub.loc[sub.bag_size.eq(high)].groupby("patient_key", as_index=False).agg(
                    label=("label", "first"), prediction=("prediction", "mean"))
                merged = low_patient.merge(high_patient, on=["patient_key", "label"], suffixes=("_low", "_high"), validate="one_to_one")
                if len(merged) != len(low_patient) or len(merged) != len(high_patient):
                    raise ValueError("Inner scale comparison patient mismatch")
                fold_gain[str(outer)] = float((losses(merged.label, merged.prediction_low) -
                                               losses(merged.label, merged.prediction_high)).mean())
                brier[str(outer)] = {"low": metric_set(merged.label, merged.prediction_low)["brier"],
                                      "high": metric_set(merged.label, merged.prediction_high)["brier"]}
            for seed in config["seeds"]:
                sub = inner.loc[inner.model.eq(model) & inner.seed.eq(seed)]
                low_rows = sub.loc[sub.bag_size.eq(low)]
                high_rows = sub.loc[sub.bag_size.eq(high)]
                merged = low_rows.merge(high_rows, on=["patient_key", "label", "outer_fold", "inner_fold", "model", "seed"],
                                        suffixes=("_low", "_high"), validate="one_to_one")
                if len(merged) != len(low_rows) or len(merged) != len(high_rows):
                    raise ValueError("Inner scale seed comparison patient mismatch")
                seed_gain[str(seed)] = float((losses(merged.label, merged.prediction_low) -
                                              losses(merged.label, merged.prediction_high)).mean())
            criterion = (sum(v > 0 for v in fold_gain.values()) >= int(rule["min_improving_outer_folds_each_step"]) and
                         sum(v > 0 for v in seed_gain.values()) >= int(rule["min_improving_seeds_each_step"]))
            if (low, high) == (8192, 16384) and rule.get("brier_not_worse_than_8192"):
                criterion = criterion and np.mean([v["high"] for v in brier.values()]) <= np.mean([v["low"] for v in brier.values()])
            both = both and criterion
            model_result[f"{low}_to_{high}"] = {"fold_gain": fold_gain, "seed_gain": seed_gain,
                                                  "fold_brier": brier, "criterion_pass": bool(criterion)}
        model_result["both_steps_pass"] = bool(both)
        candidates[model] = model_result
    return {"status": "pilot_32768_first" if any(x["both_steps_pass"] for x in candidates.values())
            else "stop_platform_or_unstable_gain", "by_model": candidates,
            "data_scope": "inner validation only; no outer outcome feedback"}


def evaluate(root: Path, config_path: Path, run_dir: Path, out_dir: Path) -> dict:
    config = json.loads(config_path.read_text())
    config_valid(config)
    folds_path = root / "folds/folds.tsv"
    baseline_path = root / _private_literal('private_4b09056568ab6f20')
    index_path = root / "mil/patient_index/index.tsv"
    folds = pd.read_csv(folds_path, sep="\t", dtype={"patient_key": str})
    baseline = pd.read_csv(baseline_path, sep="\t", dtype={"patient_key": str})
    index = pd.read_csv(index_path, sep="\t", dtype={"patient_key": str})
    if baseline.patient_key.nunique() != 114 or len(baseline) != 114 or baseline.patient_key.duplicated().any():
        raise ValueError(_private_literal('private_74e85e2b21501e17'))
    if not {"patient_key", "label", "outer_fold", "clinical_technical_batch", "group_burden"}.issubset(baseline):
        raise ValueError("Baseline OOF missing clinical or exact-group comparator")
    label_map = dict(zip(baseline.patient_key, baseline.label.astype(int)))
    unique_map = dict(zip(index.patient_key, index.unique_aa.astype(int)))
    if not set(label_map).issubset(unique_map):
        raise ValueError("Patient index missing Home patients")
    expected_outer = folds.loc[folds.partition.eq("test"), ["patient_key", "outer_fold"]]
    if (len(expected_outer) != 114 or expected_outer.patient_key.duplicated().any() or
            set(expected_outer.patient_key) != set(label_map) or
            not baseline.set_index("patient_key").outer_fold.eq(expected_outer.set_index("patient_key").outer_fold).all()):
        raise ValueError("Frozen Home fold assignment differs from baseline OOF")
    if not pd.to_numeric(folds.recorded_n_binary, errors="coerce").eq(folds.patient_key.map(label_map)).all():
        raise ValueError("Frozen fold labels differ from baseline OOF")
    missing = []
    for outer in config["outer_folds"]:
        for model in config["models"]:
            for bag in config["bag_sizes"]:
                for seed in config["seeds"]:
                    for inner in [*config["inner_folds"], None]:
                        base = fit_path(run_dir, outer, inner, model, bag, seed)
                        if not (base / "prediction.tsv").exists() or not (base / "receipt.json").exists():
                            missing.append(str(base))
    if missing:
        raise FileNotFoundError(f"Bag curve incomplete: {len(missing)} fits missing; first: {missing[:5]}")
    inner_frames, final_frames, stamps, receipts = [], [], [], []
    for outer in config["outer_folds"]:
        for model in config["models"]:
            for bag in config["bag_sizes"]:
                for seed in config["seeds"]:
                    for inner in [*config["inner_folds"], None]:
                        base = fit_path(run_dir, outer, inner, model, bag, seed)
                        frame, receipt, stamp = read_fit(base, outer, inner, model, bag, seed,
                                                         expected_keys(folds, outer, inner),
                                                         label_map, unique_map, inner is None,
                                                         sha256(config_path), sha256(folds_path),
                                                         sha256(root / "src/05_bag_curve_v2.py"),
                                                         float(config["runtime"]["max_gpu_allocated_gib"]),
                                                         float(config["runtime"]["max_cpu_rss_gib"]))
                        (final_frames if inner is None else inner_frames).append(frame)
                        stamps.append(stamp)
                        receipts.append({"outer_fold": outer, "inner_fold": inner,
                                         "model": model, "bag_size": bag, "seed": seed,
                                         "train_seconds": float(receipt.get("seconds", 0)) + float((receipt.get("early_stop_receipt") or {}).get("seconds", 0)),
                                         "inference_seconds": float(receipt.get("inference_seconds", 0)),
                                         "peak_gpu_gib": float(receipt.get("peak_gpu_gib", math.nan)),
                                         "receipt_sha256": stamp["receipt_sha256"]})
    inner = pd.concat(inner_frames, ignore_index=True)
    final = pd.concat(final_frames, ignore_index=True)
    inner_comparison, selection = select_inner(inner, config)
    oof = seed_mean_oof(final, config)
    if (len(oof) != 114*len(config["models"])*len(config["bag_sizes"]) or
            oof.groupby(["model", "bag_size"]).patient_key.nunique().ne(114).any()):
        raise ValueError("Incomplete paired final OOF curve")
    selected = oof.merge(selection[["outer_fold", "model", "bag_size"]],
                         on=["outer_fold", "model", "bag_size"], validate="many_to_one")
    if len(selected) != 114 or selected.patient_key.nunique() != 114:
        raise ValueError(_private_literal('private_0ea01f8debbf02b4'))
    baseline_columns = ["patient_key", "clinical_technical_batch", "group_burden"]
    selected = selected.merge(baseline[baseline_columns], on="patient_key", validate="one_to_one")
    selected_seed = final.merge(selection[["outer_fold", "model", "bag_size"]],
                                on=["outer_fold", "model", "bag_size"], validate="many_to_one")
    if len(selected_seed) != 114*len(config["seeds"]):
        raise ValueError("Selected seed OOF incomplete")
    selected_metrics = metric_set(selected.label, selected.prediction)
    selected_metrics.update({"mean_mc_sd": float(selected.mc_sd.mean()),
                             "max_mc_sd": float(selected.mc_sd.max()),
                             "mean_seed_sd": float(selected.seed_sd.mean()),
                             "mean_coverage_fraction": float(selected.coverage_fraction.mean()),
                             "mean_coverage_read_fraction": float(selected.coverage_read_fraction.mean())})
    comparisons = compare_to_baselines(selected, selected_seed, config)
    gate = evaluate_utility_gate(selected_metrics, comparisons, config)
    resources = pd.DataFrame(receipts)
    if not np.isfinite(resources[["train_seconds", "inference_seconds", "peak_gpu_gib"]]).all().all():
        raise ValueError("Incomplete GPU cost receipt")
    costs = resources.groupby(["model", "bag_size"], as_index=False).agg(
        total_train_seconds=("train_seconds", "sum"),
        total_inference_seconds=("inference_seconds", "sum"),
        peak_gpu_gib=("peak_gpu_gib", "max"), fits=("receipt_sha256", "count"))
    capture, capture_receipt = analytic_candidate_capture(root, config["bag_sizes"],
                                                           config["outer_folds"], folds, unique_map, final)
    capture_summary = {}
    selected_capture = {"status": "unavailable", "reason": "Train-nominated safe-AA capture unavailable"}
    if not capture.empty:
        for (model, bag), block in capture.loc[capture.safe_group_aa.gt(0)].groupby(["model", "bag_size"]):
            capture_summary[(model, int(bag))] = {
                "carrier_group_seed_pairs": len(block),
                "single_bag_probability": float(block.single_bag_exact_capture_probability.mean()),
                "mc_bag_any_probability": float(block.mc_bag_any_capture_probability.mean()),
                "mc_bag_expected_union_unique_coverage": float(block.mc_bag_expected_union_unique_aa_coverage.mean())}
        chosen_capture = capture.merge(selection[["outer_fold", "model", "bag_size"]],
                                       on=["outer_fold", "model", "bag_size"], validate="many_to_one")
        carriers = chosen_capture.loc[chosen_capture.safe_group_aa.gt(0)]
        selected_capture = {"status": "complete", "carrier_group_seed_pairs": int(len(carriers)),
                            "single_bag_probability_mean": float(carriers.single_bag_exact_capture_probability.mean()) if len(carriers) else None,
                            "mc_bag_any_probability_mean": float(carriers.mc_bag_any_capture_probability.mean()) if len(carriers) else None,
                            "interpretation": "Uniform AA analytic expectation, never observed candidate capture"}
    curves = []
    for (model, bag), block in oof.groupby(["model", "bag_size"]):
        merged = block.merge(baseline[baseline_columns], on="patient_key", validate="one_to_one")
        values = metric_set(merged.label, merged.prediction)
        values.update({"model": model, "bag_size": int(bag),
                       "mean_mc_sd": float(merged.mc_sd.mean()),
                       "mean_seed_sd": float(merged.seed_sd.mean()),
                       "mean_coverage_fraction": float(merged.coverage_fraction.mean()),
                       "mean_coverage_read_fraction": float(merged.coverage_read_fraction.mean()),
                       "candidate_capture_single_bag_probability_carriers": capture_summary.get((model, int(bag)), {}).get("single_bag_probability"),
                       "candidate_capture_mc_any_probability_carriers": capture_summary.get((model, int(bag)), {}).get("mc_bag_any_probability"),
                       "mc_expected_union_unique_coverage_carriers": capture_summary.get((model, int(bag)), {}).get("mc_bag_expected_union_unique_coverage"),
                       "candidate_capture_method": capture_receipt.get("method", "unavailable")})
        for name in config["mil_value_gate"]["comparators"]:
            effect = paired_patient_gain(merged.label, merged.prediction, merged[name],
                                         int(config["mil_value_gate"]["paired_patient_bootstrap_draws"]),
                                         int(config["mil_value_gate"]["paired_patient_bootstrap_seed"]))
            values[f"paired_gain_vs_{name}"] = effect["mean"]
            values[f"paired_gain_vs_{name}_ci_lower"] = effect["bootstrap_95ci"][0]
            values[f"paired_gain_vs_{name}_ci_upper"] = effect["bootstrap_95ci"][1]
            values[f"positive_outer_folds_vs_{name}"] = sum(
                (losses(g.label, g[name]) - losses(g.label, g.prediction)).mean() > 0
                for _, g in merged.groupby("outer_fold"))
            raw = final.loc[final.model.eq(model) & final.bag_size.eq(bag)].merge(
                baseline[["patient_key", name]], on="patient_key", validate="many_to_one")
            values[f"positive_seeds_vs_{name}"] = sum(
                (losses(g.label, g[name]) - losses(g.label, g.prediction)).mean() > 0
                for _, g in raw.groupby("seed"))
        cost = costs.loc[costs.model.eq(model) & costs.bag_size.eq(bag)].iloc[0]
        values.update({"gpu_train_seconds": float(cost.total_train_seconds),
                       "gpu_inference_seconds": float(cost.total_inference_seconds),
                       "gpu_total_seconds": float(cost.total_train_seconds + cost.total_inference_seconds),
                       "peak_gpu_gib": float(cost.peak_gpu_gib), "fit_count": int(cost.fits)})
        curves.append(values)
    expansion = continuation_scale_check(inner, config)
    if gate["status"] != "pass" and expansion["status"] == "pilot_32768_first":
        expansion["status"] = "stop_no_stable_gain_over_baselines"
        expansion["reason"] = "Inner scale trends do not override the patient-level utility gate"
    out_dir.mkdir(parents=True, exist_ok=True)
    outputs = {"curve_patient_seed.tsv": final, "curve_oof_by_model_bag.tsv": oof,
               "inner_comparison.tsv": inner_comparison, "fold_selection.tsv": selection,
               "selected_oof.tsv": selected, "curve_metrics.tsv": pd.DataFrame(curves)}
    if not capture.empty:
        outputs["candidate_capture_analytic.tsv"] = capture
    for name, table in outputs.items():
        table.to_csv(out_dir / name, sep="\t", index=False)
    chosen_costs = resources.merge(selection[["outer_fold", "model", "bag_size"]],
                                   on=["outer_fold", "model", "bag_size"], validate="many_to_one")
    selected_cost = {"all_inner_search_gpu_seconds": float(resources.loc[resources.inner_fold.notna(),
                                                ["train_seconds", "inference_seconds"]].to_numpy().sum()),
                     "chosen_outer_final_gpu_seconds": float(chosen_costs.loc[chosen_costs.inner_fold.isna(),
                                                ["train_seconds", "inference_seconds"]].to_numpy().sum()),
                     "peak_gpu_gib_all_fits": float(resources.peak_gpu_gib.max()),
                     "interpretation": "GPU process time from fit receipts; includes all five-size inner search and selected final fits"}
    write_json(out_dir / "gate.json", {**gate, "selected_metrics": selected_metrics,
                                      "comparisons": comparisons, "fold_selection": selection.to_dict("records"),
                                      "candidate_capture": capture_receipt,
                                      "selected_candidate_capture": selected_capture,
                                      "selected_gpu_cost": selected_cost,
                                      "expansion_32768": expansion})
    write_json(out_dir / "input_manifest.json", {"fit_files": stamps,
                                                   "config_sha256": sha256(config_path),
                                                   "folds_sha256": sha256(folds_path),
                                                   "baseline_oof_sha256": sha256(baseline_path),
                                                   "patient_index_sha256": sha256(index_path)})
    receipt = {"status": "complete", "analysis_status": config["analysis_status"],
               "source_code_sha256": sha256(Path(__file__)),
               "input_manifest_sha256": sha256(out_dir / "input_manifest.json"),
               "config_sha256": sha256(config_path),
               "inner_fits": len(inner_frames), "outer_fits": len(final_frames),
               "oof_patients": len(selected), "curve_model_bag_points": len(curves),
               "gate_status": gate["status"],
               "outputs": {name: sha256(out_dir / name) for name in outputs},
               "gate_sha256": sha256(out_dir / "gate.json")}
    write_json(out_dir / "receipt.json", receipt)
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--out-dir", type=Path)
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    root = args.root.resolve()
    config_path = args.config or root / "config/bag_curve_v2.json"
    run_dir = args.run_dir or root / "mil/bag_curve_v2"
    out_dir = args.out_dir or run_dir / "evaluation"
    config = json.loads(config_path.read_text())
    config_valid(config)
    if args.check_only:
        total = len(config["outer_folds"])*len(config["models"])*len(config["bag_sizes"])*len(config["seeds"])*(len(config["inner_folds"])+1)
        present = sum((fit_path(run_dir, outer, inner, model, bag, seed) / "receipt.json").exists()
                      for outer in config["outer_folds"] for model in config["models"]
                      for bag in config["bag_sizes"] for seed in config["seeds"]
                      for inner in [*config["inner_folds"], None])
        print(json.dumps({"expected_fits": total, "present_receipts": present,
                          "missing_receipts": total-present, "selection_scope": config["selection"]["scope"]}, indent=2))
        return
    print(json.dumps(evaluate(root, config_path, run_dir, out_dir), indent=2))


if __name__ == "__main__":
    main()
