#!/usr/bin/env python3
"""Patient-level OOF evidence and locked screen gate for four CDR3 representations."""
from __future__ import annotations
from release_runtime import private_literal as _private_literal, release_path as _release_path
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import time

for name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[name] = "1"

import numpy as np
import pandas as pd

ROOT = Path(_release_path('path_99d8c1c6fde2a6b9'))
BASE = ROOT / "representation_comparison"
OUT = BASE / "report"
FROZEN = ROOT / "src/05_predictive.py"
MIL_CODE = ROOT / "src/08_compare_mil.py"
MODELS = ("tcrbert_dualtask_aa_mean", "sceptr_cdr3_only", "cdr3vec_small", "kmer3_cosine_svd")
SEEDS = (17, 29, 43)
OOF = OUT / "model_patient_oof.tsv"
SEED_METRICS = OUT / "seed_fold_metrics.tsv"
PAIRS = OUT / "pairwise_comparisons.tsv"
SUMMARY = OUT / "summary.json"
RECEIPT = OUT / "receipt.json"


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def save_tsv(path, frame):
    tmp = path.with_suffix(path.suffix + ".partial")
    frame.to_csv(tmp, sep="\t", index=False, float_format="%.12g")
    os.replace(tmp, path)


def save_json(path, value):
    tmp = path.with_suffix(path.suffix + ".partial")
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    os.replace(tmp, path)


def loss(y, p):
    y = np.asarray(y, dtype=float)
    p = np.clip(np.asarray(p, dtype=float), 1e-6, 1 - 1e-6)
    return -(y * np.log(p) + (1 - y) * np.log1p(-p))


def paired(delta, seed=20260930):
    delta = np.asarray(delta, dtype=float)
    if len(delta) < 5 or not np.isfinite(delta).all():
        raise ValueError("Invalid patient paired differences")
    rng = np.random.default_rng(seed)
    ix = rng.integers(0, len(delta), size=(5000, len(delta)))
    means = delta[ix].mean(axis=1)
    lo, hi = np.quantile(means, (0.025, 0.975))
    return {"n_patients": int(len(delta)), "mean": float(delta.mean()),
            "ci_low": float(lo), "ci_high": float(hi),
            "method": "patient-paired nonparametric percentile; fixed folds; 5000 draws"}


def get_frozen():
    spec = importlib.util.spec_from_file_location("frozen_atlas_predictive_summary", FROZEN)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def get_mil():
    spec = importlib.util.spec_from_file_location("representation_comparison_mil", MIL_CODE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if (tuple(module.MODELS) != MODELS or tuple(module.SEEDS) != SEEDS or
            (module.BAG, module.POOL, module.DIM, module.MAX_EPOCHS,
             module.PATIENCE, module.N_MC, getattr(module, _private_literal('private_ad724481966f3a4d')),
             module.MC_SD_THRESHOLD) != (2048, 4096, 64, 40, 6, 8, 32, 0.05)):
        raise ValueError("MIL comparison settings differ from locked screen")
    return module


def verify_baseline_receipts(data):
    totals = BASE / "home_sample/patient_totals.tsv"
    common = {"folds": data.source_hashes["folds"],
              "manifest": sha(ROOT / "config/cohort_samples.tsv")}
    for name, script, oof, tuning, extra in (
        ("L1", "08_baseline.py", "baseline_oof.tsv", "baseline_inner_tuning.tsv",
         {"patient_totals": sha(totals), "frozen_baseline_code": sha(FROZEN)}),
        ("L2", "08_baseline_l2.py", "baseline_l2_oof.tsv", "baseline_l2_inner_tuning.tsv",
         {"totals": sha(totals), "frozen_covariate_code": sha(FROZEN)}),
    ):
        receipt_path = BASE / "baseline" / ("receipt.json" if name == "L1" else "l2_receipt.json")
        receipt = json.loads(receipt_path.read_text())
        expected = {**common, **extra, "this_script": sha(ROOT / "src" / script)}
        if (receipt.get("status") != "complete" or
                receipt.get("source_sha256") != expected or
                receipt.get("oof_sha256") != sha(BASE / "baseline" / oof) or
                receipt.get("tuning_sha256") != sha(BASE / "baseline" / tuning)):
            raise ValueError(name + " baseline receipt/source mismatch")


def verified_raw_hash(data, mil, model):
    if model == "kmer3_cosine_svd":
        return mil.hash_object({
            "unique_sha256": data.source_hashes["unique"],
            "features": 4096, "ngram_range": [3, 3],
            "norm": "l2", "alternate_sign": False, "lowercase": False,
            "dtype": "float32", "method": "hashed_char3_cosine",
        })
    directory = BASE / "vectors" / model
    bank_path = directory / "bank_fp16.npy"
    receipt = json.loads((directory / "receipt.json").read_text())
    bank_hash = sha(bank_path)
    bank = np.load(bank_path, mmap_mode="r", allow_pickle=False)
    if (receipt.get("status") != "complete" or receipt.get("model") != model or
            receipt.get("label_blind") is not True or
            receipt.get("sample_sha256") != data.source_hashes["unique"] or
            receipt.get("row_order") != "home_sample/unique_aa.parquet" or
            receipt.get("bank_sha256") != bank_hash or
            receipt.get("dimension") != mil.DIMS[model] or
            receipt.get("dtype") != "float16" or
            receipt.get("n_rows") != len(data.aa) or
            receipt.get("n_failed") != 0 or receipt.get("n_nonfinite") != 0 or
            bank.shape != (len(data.aa), mil.DIMS[model]) or bank.dtype != np.float16):
        raise ValueError("Aligned vector bank receipt invalid: " + model)
    if "mapping_sha256" in receipt:
        mapping = json.loads((BASE / "vectors/mapping_receipt.json").read_text())
        if (mapping.get("status") != "complete" or
                mapping.get("sample_sha256") != data.source_hashes["unique"] or
                mapping.get("mapping_sha256") != receipt["mapping_sha256"] or
                mapping.get("mapping_sha256") != sha(BASE / "vectors/sample_bank_mapping.parquet")):
            raise ValueError("Aligned vector mapping receipt invalid: " + model)
    return bank_hash


def verify_model_contracts(data, mil, model):
    """Validate the complete fit chain before using any outcome probabilities."""
    raw_hash = verified_raw_hash(data, mil, model)
    receipts = []
    projection_seconds = 0.0
    for outer in range(5):
        for inner in (0, 1, 2, None):
            train, evaluation = data.split(outer, inner)
            scope = "final" if inner is None else f"inner_{inner}"
            directory = BASE / "mil/projections" / model / f"outer_{outer}" / scope
            meta_path = directory / "metadata.json"
            meta = json.loads(meta_path.read_text())
            method = ("train_only_hashed_char3_l2_4096_truncated_svd64"
                      if model == "kmer3_cosine_svd" else "train_only_patient_balanced_pca64")
            expected_meta = {
                "status": "complete", "model": model, "outer_fold": outer,
                "inner_fold": inner, "train_patient_hash": mil.id_hash(train),
                "n_train_patients": len(train), "n_rows_per_train_patient": 4096,
                "n_fit_rows": len(train) * 4096, "raw_bank_sha256": raw_hash,
                "folds_sha256": data.source_hashes["folds"],
                "draw_sha256": data.source_hashes["draw"],
                "unique_sha256": data.source_hashes["unique"],
                "script_sha256": data.source_hashes["script"],
                "method": method, "dim": 64, "total_aa": len(data.aa),
            }
            if any(meta.get(key) != value for key, value in expected_meta.items()):
                raise ValueError("Fold projection contract invalid: " + str(meta_path))
            if (meta.get("basis_sha256") != sha(directory / "basis.npz") or
                    meta.get("projection_sha256") != sha(directory / "projected_fp16.npy")):
                raise ValueError("Fold projection data hash invalid: " + str(directory))
            meta_hash = sha(meta_path)
            projection_seconds += float(meta["seconds"])
            for seed in SEEDS:
                fit_dir = BASE / "mil" / model / f"outer_{outer}" / scope / f"seed_{seed}"
                fit_path = fit_dir / "receipt.json"
                fit = json.loads(fit_path.read_text())
                if inner is None:
                    prior_paths = [BASE / "mil" / model / f"outer_{outer}" /
                                   f"inner_{fold}" / f"seed_{seed}" / "receipt.json"
                                   for fold in range(3)]
                    prior = [json.loads(path.read_text()) for path in prior_paths]
                    fixed_epochs = int(np.median([r["training"]["best_epoch"] for r in prior]))
                    inner_hashes = [sha(path) for path in prior_paths]
                    fit_ids, stop_ids = train, None
                    draw_scope = f"test-outer{outer}-seed{seed}"
                else:
                    fixed_epochs, inner_hashes = None, []
                    fit_ids, stop_ids = mil.early_split(train, data.labels, outer, inner, seed)
                    draw_scope = f"inner-outer{outer}-inner{inner}-seed{seed}"
                expected_contract = {
                    "model": model, "outer_fold": outer, "inner_fold": inner, "seed": seed,
                    "bag_size": 2048, "pool_per_patient": 4096,
                    "projection_sha256": meta["projection_sha256"],
                    "raw_bank_sha256": raw_hash,
                    "script_sha256": data.source_hashes["script"],
                    "folds_sha256": data.source_hashes["folds"],
                    "draw_sha256": data.source_hashes["draw"],
                    "train_patient_hash": mil.id_hash(train),
                    "fit_patient_hash": mil.id_hash(fit_ids),
                    "stop_patient_hash": None if stop_ids is None else mil.id_hash(stop_ids),
                    "eval_patient_hash": mil.id_hash(evaluation),
                    "draw_scope": draw_scope, "device": "cuda", "fixed_epochs": fixed_epochs,
                    "inner_receipt_sha256": inner_hashes,
                    "optimizer": "AdamW lr=3e-4 weight_decay=1e-3",
                    "network": "GatedMIL dim64 hidden64 attention32 dropout0.2",
                    "max_epochs": 40, "patience": 6, "patient_batch": 2,
                    "mc_initial": 8, "mc_escalation": 32, "mc_sd_threshold": 0.05,
                }
                prediction_path = fit_dir / "prediction.tsv"
                training = fit.get("training", {})
                if (fit.get("status") != "complete" or fit.get("contract") != expected_contract or
                        fit.get("projection_metadata_sha256") != meta_hash or
                        fit.get("model_sha256") != sha(fit_dir / "model.pt") or
                        fit.get("prediction_sha256") != sha(prediction_path) or
                        fit.get("n_predictions") != len(evaluation) or
                        fit.get("mc_bags") not in (8, 32) or
                        training.get("n_fit_patients") != len(fit_ids) or
                        training.get("n_stop_patients") != (0 if stop_ids is None else len(stop_ids)) or
                        training.get("fit_epochs") != (40 if fixed_epochs is None else fixed_epochs) or
                        not 1 <= training.get("best_epoch", 0) <= training.get("epochs_ran", 0) <=
                        (40 if fixed_epochs is None else fixed_epochs) or
                        float(training.get("peak_rss_gib", float("inf"))) > 512 or
                        float(fit.get("peak_gpu_reserved_including_inference_gib", float("inf"))) > 14):
                    raise ValueError("MIL fit provenance invalid: " + str(fit_path))
                prediction = pd.read_csv(prediction_path, sep="\t", dtype={"patient_key": str})
                expected_inner = (prediction.inner_fold.isna().all() if inner is None
                                  else prediction.inner_fold.eq(inner).all())
                if (len(prediction) != len(evaluation) or prediction.patient_key.duplicated().any() or
                        set(prediction.patient_key) != set(evaluation) or not expected_inner or
                        not prediction.outer_fold.eq(outer).all() or
                        not prediction.model.eq(model).all() or not prediction.seed.eq(seed).all() or
                        not prediction.bag_size.eq(2048).all() or
                        not prediction.draw_scope.eq(draw_scope).all() or
                        not prediction.mc_bags.eq(fit.get("mc_bags")).all() or
                        not prediction.probability.between(0, 1, inclusive="neither").all() or
                        not np.isfinite(prediction[["probability", "prediction_sd"]].to_numpy(dtype=float)).all() or
                        prediction.prediction_sd.lt(0).any() or
                        any(int(row.recorded_n_binary) != data.labels[row.patient_key]
                            for row in prediction.itertuples())):
                    raise ValueError("MIL prediction identity invalid: " + str(prediction_path))
                receipts.append(fit)
    if len(receipts) != 60:
        raise ValueError("Expected 60 verified inner/final fits: " + model)
    return receipts, projection_seconds, raw_hash


def equal_depth_receipt_present(model, data):
    path = BASE / "sensitivity/equal_depth" / model / "receipt.json"
    if not path.is_file():
        return False
    receipt = json.loads(path.read_text())
    source = receipt.get("source_sha256", {})
    if (receipt.get("status") != "complete" or receipt.get("model") != model or
            source.get("folds") != data.source_hashes["folds"] or
            source.get("draw") != data.source_hashes["draw"]):
        raise ValueError("Equal-depth receipt does not match frozen screen: " + str(path))
    return True


def main():
    start = time.monotonic()
    OUT.mkdir(parents=True, exist_ok=True)
    frozen = get_frozen()
    mil = get_mil()
    data = mil.SharedData()
    verify_baseline_receipts(data)
    baseline = pd.read_csv(BASE / "baseline/baseline_oof.tsv", sep="\t", dtype={"patient_key": str})
    l2 = pd.read_csv(BASE / "baseline/baseline_l2_oof.tsv", sep="\t", dtype={"patient_key": str})
    if len(baseline) != 126 or baseline.patient_key.duplicated().any() or len(l2) != 126:
        raise ValueError("Clinical baseline OOF incomplete")
    baseline = baseline.merge(l2[["patient_key", "outer_fold", "recorded_n_binary",
                                  "baseline_l2_probability"]],
                              on=["patient_key", "outer_fold", "recorded_n_binary"],
                              validate="one_to_one")
    folds = pd.read_csv(ROOT / "folds/folds.tsv", sep="\t", dtype={"patient_key": str})
    expected = folds.loc[folds.partition.eq("test"),
                         ["patient_key", "outer_fold", "recorded_n_binary"]].sort_values("patient_key")
    if len(expected) != 126 or expected.patient_key.duplicated().any() or not (
        baseline[["patient_key", "outer_fold", "recorded_n_binary"]].sort_values("patient_key").reset_index(drop=True)
        .equals(expected.reset_index(drop=True))
    ):
        raise ValueError("Clinical baseline differs from frozen heldout patients/labels")
    source_predictions = {}
    seed_rows = []
    model_rows = []
    combined = []
    verified_model_sources = {}
    for model in MODELS:
        fit_receipts, projection_seconds, raw_hash = verify_model_contracts(data, mil, model)
        verified_model_sources[model] = {"raw_bank_sha256": raw_hash,
                                         "n_verified_fits": len(fit_receipts)}
        by_seed = []
        for seed in SEEDS:
            parts = []
            for outer in range(5):
                path = BASE / "mil" / model / f"outer_{outer}" / "final" / f"seed_{seed}" / "prediction.tsv"
                receipt_path = path.parent / "receipt.json"
                if not path.is_file() or not receipt_path.is_file():
                    raise FileNotFoundError("Comparison fit incomplete: " + str(path))
                receipt = json.loads(receipt_path.read_text())
                if receipt.get("status") != "complete" or receipt.get("prediction_sha256") != sha(path):
                    raise ValueError("Comparison fit receipt invalid: " + str(path))
                p = pd.read_csv(path, sep="\t", dtype={"patient_key": str})
                needed = {"patient_key", "recorded_n_binary", "outer_fold", "model", "seed",
                          "probability", "prediction_sd", "mc_bags", "bag_size"}
                if not needed <= set(p):
                    raise ValueError("Comparison prediction columns missing: " + str(path))
                if (not p.model.eq(model).all() or not p.seed.eq(seed).all()
                        or not p.outer_fold.eq(outer).all() or not p.bag_size.eq(2048).all()
                        or not p.mc_bags.isin((8, 32)).all() or
                        not p.probability.between(0, 1, inclusive="neither").all() or
                        not np.isfinite(p[["probability", "prediction_sd"]].to_numpy(dtype=float)).all()):
                    raise ValueError("Comparison prediction values/identity invalid: " + str(path))
                parts.append(p)
                source_predictions[str(path.relative_to(ROOT))] = {
                    "prediction_sha256": sha(path), "receipt_sha256": sha(receipt_path)}
            one_seed = pd.concat(parts, ignore_index=True).sort_values("patient_key")
            check = one_seed[["patient_key", "outer_fold", "recorded_n_binary"]].reset_index(drop=True)
            if len(one_seed) != 126 or one_seed.patient_key.duplicated().any() or not check.equals(expected.reset_index(drop=True)):
                raise ValueError(f"{model} seed {seed}: patient fold/label mismatch")
            merged = baseline.merge(one_seed[["patient_key", "probability", "prediction_sd",
                                             "mc_bags"]], on="patient_key", validate="one_to_one")
            y = merged.recorded_n_binary.to_numpy(dtype=int)
            model_p = merged.probability.to_numpy(dtype=float)
            base_p = merged.baseline_probability.to_numpy(dtype=float)
            l2_p = merged.baseline_l2_probability.to_numpy(dtype=float)
            delta_l1 = loss(y, base_p) - loss(y, model_p)
            delta_l2 = loss(y, l2_p) - loss(y, model_p)
            fold_delta = merged.assign(delta_l2=delta_l2).groupby("outer_fold").delta_l2.mean()
            seed_rows.append({"model": model, "seed": seed, "n_patients": 126,
                              "paired_l1_mean": float(delta_l1.mean()),
                              "paired_l2_mean": float(delta_l2.mean()),
                              "favorable_l2_outer_folds": int((fold_delta > 0).sum()),
                              "mean_inference_sd": float(merged.prediction_sd.mean()),
                              "max_inference_sd": float(merged.prediction_sd.max()),
                              "n_mc_escalated": int(merged.mc_bags.eq(32).sum()),
                              "log_loss": float(loss(y, model_p).mean())})
            for fold, value in fold_delta.items():
                seed_rows.append({"model": model, "seed": seed, "outer_fold": int(fold),
                                  "fold_paired_l2_mean": float(value),
                                  "record_type": "outer_fold"})
            by_seed.append(one_seed[["patient_key", "probability", "prediction_sd", "mc_bags"]]
                           .rename(columns={"probability": f"p_{seed}",
                                            "prediction_sd": f"sd_{seed}",
                                            "mc_bags": f"mc_{seed}"}))
        pred = by_seed[0]
        for other in by_seed[1:]:
            pred = pred.merge(other, on="patient_key", validate="one_to_one")
        pred["model_probability"] = pred[[f"p_{s}" for s in SEEDS]].mean(axis=1)
        pred["mean_inference_sd"] = pred[[f"sd_{s}" for s in SEEDS]].mean(axis=1)
        pred = baseline.merge(pred, on="patient_key", validate="one_to_one")
        pred["model"] = model
        pred["paired_l1_log_loss_improvement"] = (loss(pred.recorded_n_binary, pred.baseline_probability)
                                                   - loss(pred.recorded_n_binary, pred.model_probability))
        pred["paired_l2_log_loss_improvement"] = (loss(pred.recorded_n_binary, pred.baseline_l2_probability)
                                                   - loss(pred.recorded_n_binary, pred.model_probability))
        combined.append(pred)
        gpu_train_seconds = sum(float(r["training"]["seconds"]) for r in fit_receipts)
        peak_gpu_gib = max(float(r["peak_gpu_reserved_including_inference_gib"]) for r in fit_receipts)
        y = pred.recorded_n_binary.to_numpy(dtype=int)
        mp = pred.model_probability.to_numpy(dtype=float)
        metric = frozen.metrics(y, mp)
        fold_mean = pred.groupby("outer_fold").paired_l2_log_loss_improvement.mean()
        untreated = pred.loc[pred.neoadjuvant.eq("no")]
        untreated_delta = untreated.paired_l2_log_loss_improvement.to_numpy(dtype=float)
        model_rows.append({
            "model": model, "n_patients": 126, "metrics": metric,
            "paired_l1": paired(pred.paired_l1_log_loss_improvement),
            "paired_l2": paired(pred.paired_l2_log_loss_improvement),
            "favorable_l2_outer_folds": int((fold_mean > 0).sum()),
            "l2_outer_fold_means": {str(int(k)): float(v) for k, v in fold_mean.items()},
            "untreated_n": int(len(untreated)),
            "untreated_paired_l2": paired(untreated_delta),
            "mean_inference_sd": float(pred.mean_inference_sd.mean()),
            "mean_seed_prediction_sd": float(pred[[f"p_{s}" for s in SEEDS]].std(axis=1, ddof=1).mean()),
            "n_inner_and_final_fits": len(fit_receipts),
            "total_training_seconds_all_fits": gpu_train_seconds,
            "total_fold_projection_seconds": projection_seconds,
            "peak_gpu_reserved_gib": peak_gpu_gib,
            "n_seed_fold_favorable_at_least_4": int(sum(
                row.get("favorable_l2_outer_folds", 0) >= 4
                for row in seed_rows if row.get("model") == model and "outer_fold" not in row)),
            "all_seed_l2_mean_positive": bool(all(
                row.get("paired_l2_mean", -1) > 0
                for row in seed_rows if row.get("model") == model and "outer_fold" not in row)),
        })
    all_oof = pd.concat(combined, ignore_index=True)
    if len(all_oof) != 126 * len(MODELS):
        raise ValueError("Combined OOF row count incorrect")
    metrics_by_model = {row["model"]: row for row in model_rows}
    y = baseline.recorded_n_binary.to_numpy(dtype=int)
    base_metrics = {"frozen_l1": frozen.metrics(y, baseline.baseline_probability),
                    "predefined_l2_sensitivity": frozen.metrics(y, baseline.baseline_l2_probability)}
    l2_brier = base_metrics["predefined_l2_sensitivity"]["brier"]
    best_old_loss = min(metrics_by_model[m]["metrics"]["log_loss"]
                        for m in ("tcrbert_dualtask_aa_mean", "sceptr_cdr3_only"))
    decisions = {}
    for model, row in metrics_by_model.items():
        slope = row["metrics"].get("calibration_slope")
        preliminary = bool(
            row["paired_l1"]["ci_low"] > 0
            and row["paired_l2"]["ci_low"] > 0
            and row["favorable_l2_outer_folds"] >= 4
            and row["all_seed_l2_mean_positive"]
            and row["n_seed_fold_favorable_at_least_4"] >= 2
            and row["metrics"]["brier"] <= l2_brier + 0.01
            and slope is not None and slope > 0
            and row["untreated_paired_l2"]["mean"] > 0
        )
        distinct_value = bool(row["metrics"]["log_loss"] <= best_old_loss + 0.01) if model == "cdr3vec_small" else None
        equal_depth_present = (equal_depth_receipt_present(model, data)
                               if model == "cdr3vec_small" else False)
        if preliminary and model == "cdr3vec_small" and distinct_value:
            status = "PENDING_EQUAL_DEPTH_SENSITIVITY" if not equal_depth_present else "EQUAL_DEPTH_REVIEW_REQUIRED"
        elif model == "cdr3vec_small":
            status = "DO_NOT_FULL_ENCODE_FROM_THIS_SCREEN"
        else:
            status = "PRELIMINARY_POSITIVE" if preliminary else "NO_STABLE_GAIN"
        decisions[model] = {"preliminary_gate": preliminary,
                            "within_0p01_log_loss_of_best_existing_arm": distinct_value,
                            "equal_depth_receipt_present": equal_depth_present,
                            "equal_depth_direction_verified": False,
                            "status": status}
    ordered = sorted(MODELS, key=lambda m: metrics_by_model[m]["metrics"]["log_loss"])
    pair_rows = []
    for a in MODELS:
        pa = all_oof.loc[all_oof.model.eq(a)].sort_values("patient_key")
        for b in MODELS:
            if a >= b:
                continue
            pb = all_oof.loc[all_oof.model.eq(b)].sort_values("patient_key")
            if pa.patient_key.tolist() != pb.patient_key.tolist():
                raise ValueError("Model patient alignment mismatch")
            d = loss(pa.recorded_n_binary, pb.model_probability) - loss(pa.recorded_n_binary, pa.model_probability)
            pair_rows.append({"model_a": a, "model_b": b, **paired(d)})
    save_tsv(OOF, all_oof)
    save_tsv(SEED_METRICS, pd.DataFrame(seed_rows))
    save_tsv(PAIRS, pd.DataFrame(pair_rows))
    coverage_path = BASE / "home_sample/patient_totals.tsv"
    coverage = pd.read_csv(coverage_path, sep="\t")
    bag_receipt_path = BASE / "coverage/bag_receipt.json"
    bag_capture = json.loads(bag_receipt_path.read_text())
    if bag_capture.get("status") != "complete" or bag_capture.get("source_sha256", {}).get("draw") != sha(BASE / "home_sample/patient_aa_draw.parquet"):
        raise ValueError("Heldout bag coverage receipt invalid")
    capture_receipt_path = BASE / "coverage/receipt.json"
    capture = json.loads(capture_receipt_path.read_text())
    if capture.get("status") != "complete" or capture.get("label_blind") is not True or capture.get("source_sha256", {}).get("patient_totals") != sha(coverage_path):
        raise ValueError("Label-blind full-Home group capture receipt invalid")
    summary = {
        "status": "complete_screen", "endpoint": "recorded N+; pN/ypN unverified",
        "population": "126 N-known Home patients; 70 N0, 56 N+",
        "screen_pool": "4096 label-blind unique AA per Home patient, 2048 train/inference MIL bag",
        "median_full_repertoire_aa_coverage": float(coverage.aa_coverage.median()),
        "median_full_repertoire_read_coverage": float(coverage.read_coverage.median()),
        "median_full_home_group_capture_fraction": float(capture["median_group_capture_fraction"]),
        "group_capture_receipt_sha256": sha(capture_receipt_path),
        "median_heldout_bag_aa_coverage": float(bag_capture["median_bag_unique_aa_coverage"]),
        "median_heldout_bag_read_coverage": float(bag_capture["median_bag_read_coverage"]),
        "median_heldout_bag_group_capture_fraction": float(bag_capture["median_bag_group_capture_fraction"]),
        "heldout_bag_capture_receipt_sha256": sha(bag_receipt_path),
        "null_interpretation": "Sparse sampled screen cannot establish no full-repertoire value",
        "baseline": base_metrics, "models": metrics_by_model, "decisions": decisions,
        "ranking_by_oof_log_loss": ordered,
        "third_full_bank_allowed_now": False,
        "third_full_bank_gate": "CDR3vec requires both baseline paired 95% CI lower bounds >0, >=4/5 favorable outer folds, consistent seeds, no material Brier loss, positive calibration slope, untreated and actual equal-depth direction, and no >0.01 log-loss deficit to best existing arm",
        "external_used_for_selection": False,
        "verified_model_sources": verified_model_sources,
        "source_prediction_hashes": source_predictions,
    }
    save_json(SUMMARY, summary)
    receipt = {"status": "complete", "source_sha256": {
        "this_script": sha(Path(__file__)), "fit_script": sha(MIL_CODE),
        "folds": sha(ROOT / "folds/folds.tsv"),
        "frozen_metrics_code": sha(FROZEN),
        "shared_draw": data.source_hashes["draw"],
        "shared_unique_aa": data.source_hashes["unique"],
        "baseline_l1": sha(BASE / "baseline/baseline_oof.tsv"),
        "baseline_l2": sha(BASE / "baseline/baseline_l2_oof.tsv"),
        "draw_totals": sha(coverage_path), "group_capture_receipt": sha(capture_receipt_path),
        "bag_capture_receipt": sha(bag_receipt_path)},
        "n_models": len(MODELS), "n_seeds": len(SEEDS), "n_oof_patients_per_model": 126,
        "output_sha256": {"oof": sha(OOF), "seed_fold": sha(SEED_METRICS),
                          "pairwise": sha(PAIRS), "summary": sha(SUMMARY)},
        "seconds": round(time.monotonic() - start, 3)}
    save_json(RECEIPT, receipt)
    print(json.dumps({"status": summary["status"], "decisions": decisions,
                      "ranking": ordered, "seconds": receipt["seconds"]}, sort_keys=True))


if __name__ == "__main__":
    main()
