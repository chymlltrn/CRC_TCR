#!/usr/bin/env python3
"""Frozen-fold, held-out RandomForest SHAP repair; no feature selection or tuning."""
from release_runtime import private_literal as _private_literal, release_path as _release_path
import os
for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[key] = "2"
import hashlib
import importlib.metadata
import json
from pathlib import Path
import shutil
import sys
import time

import joblib
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import shap
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

SOURCE = Path(_release_path('path_d4455a6432ab939c'))
DESIGN = Path(_release_path('path_ba45c484b7dabb43'))
OUT = Path(_release_path('path_6f3f3692efa62d86'))
ARM = "RandomForest__CLUSTERTCR__stratLabelBatchCohort"
OOF_ATOL = 1e-12
ADDITIVITY_ATOL = 1e-10
START = time.time()


def log(message):
    print(time.strftime("%Y-%m-%d %H:%M:%S"), message, flush=True)


def dump_json(path, obj):
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + "\n")


def sha256(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(4 << 20), b""):
            h.update(block)
    return h.hexdigest()


def feature_parts(feature):
    prefix, aa = feature.split("::")
    model = "tcrbert" if prefix.startswith("tcrbert") else "sceptr"
    return model, int(prefix[len(model):]), aa


def main():
    for child in ("scripts", "tables", "models", "audit", "source_snapshots", "logs"):
        (OUT / child).mkdir(parents=True, exist_ok=True)
    paths = {
        "feature_matrix": SOURCE / "src/X_clusterTCR_289.parquet",
        "fold_selection": SOURCE / "results/merged/fold_selection_clusterTCR.json",
        "oof_predictions": SOURCE / "results/merged/oof_clusterTCR.csv",
        "design": DESIGN,
        "performance_reference": SOURCE / "results/merged/perf_clusterTCR.csv",
        "original_battery_code": SOURCE / "src/run_cluster_tcr_battery.py",
        "original_shap_code": SOURCE / "src/run_shap_clustertcr.py",
        "original_feature_builder": SOURCE / "src/build_cluster_tcr_features.py",
        "feature_build_statistics": SOURCE / "src/clusterTCR_build_stats.json",
    }
    provenance = {}
    for name, path in paths.items():
        st = path.stat()
        provenance[name] = dict(path=str(path), bytes=st.st_size, mtime_ns=st.st_mtime_ns,
                                sha256=sha256(path))
        if name != "feature_matrix":
            shutil.copy2(path, OUT / "source_snapshots" / path.name)
    dump_json(OUT / "audit/input_provenance.json", provenance)
    versions = {name: importlib.metadata.version(name) for name in
                ("numpy", "scipy", "pandas", "scikit-learn", "shap", "pyarrow", "joblib")}
    versions["python"] = sys.version
    versions["python_executable"] = sys.executable
    dump_json(OUT / "audit/environment.json", versions)
    design = pd.read_csv(DESIGN, index_col=0, float_precision="round_trip")
    original = pd.read_csv(paths["oof_predictions"], index_col=0, float_precision="round_trip")
    folds = json.loads(paths["fold_selection"].read_text())[ARM]
    assert len(design) == 289 and design.index.is_unique
    assert original.index.is_unique and original.index.equals(design.index)
    assert len(folds) == 5 and [f["fold"] for f in folds] == list(range(5))
    assert int(design.y.sum()) == 110 and set(design.y) == {0, 1}
    all_names = pq.read_schema(paths["feature_matrix"]).names
    original_positions = {name: position for position, name in enumerate(all_names)}
    # Stable union is only a storage order. Each model uses its frozen fold order below.
    union = list(dict.fromkeys(name for f in folds for name in f["features"]))
    assert len(union) == 188 and all(name in original_positions for name in union)
    log(f"Reading {len(union)} selected-union columns from original parquet")
    X = pd.read_parquet(paths["feature_matrix"], columns=union).loc[design.index]
    assert X.index.equals(design.index) and X.shape == (289, 188)
    assert np.isfinite(X.to_numpy()).all() and (X.to_numpy() >= 0).all()
    y = design.y.to_numpy(dtype=int)
    splits = list(StratifiedKFold(5, shuffle=True, random_state=42).split(X, design.stratum))
    prediction = np.full(len(X), np.nan)
    fold_id = np.full(len(X), -1, dtype=int)
    fold_models, fold_records, order_rows = [], [], []
    for k, ((tr, te), fd) in enumerate(zip(splits, folds)):
        names = fd["features"]
        assert len(names) == len(set(names)) == 50
        # IMPORTANT: no sorting by original column positions; these names define model order.
        Ztr = X.iloc[tr].loc[:, names].to_numpy(dtype=np.float64)
        Zte = X.iloc[te].loc[:, names].to_numpy(dtype=np.float64)
        assert (Ztr.var(axis=0) > 1e-12).all()  # frozen post-selection variance filter is identity
        params = {name.replace("clf__", ""): None if value == "None" else json.loads(value)
                  for name, value in fd["best_params"].items()}
        pipe = Pipeline([("scl", StandardScaler()),
                         ("clf", RandomForestClassifier(n_estimators=500, random_state=42,
                                                        n_jobs=2, **params))])
        pipe.fit(Ztr, y[tr])
        assert pipe["clf"].classes_.tolist() == [0, 1]
        prediction[te] = pipe.predict_proba(Zte)[:, 1]
        fold_id[te] = k
        joblib.dump(dict(pipeline=pipe, features=names, fold=k, best_params=params,
                         train_patients=design.index[tr].tolist(),
                         test_patients=design.index[te].tolist()),
                    OUT / "models" / f"fold_{k}_randomforest.joblib", compress=3)
        fold_models.append((pipe, names, tr, te))
        sorted_names = sorted(names, key=original_positions.__getitem__)
        fold_records.append(dict(fold=k, n_train=len(tr), n_test=len(te),
                                 features=names, best_params=params,
                                 full_classifier_parameters=pipe["clf"].get_params(),
                                 sorted_vs_frozen_position_mismatches=sum(a != b for a, b in zip(names, sorted_names)),
                                 test_patients=design.index[te].tolist()))
        for j, name in enumerate(names):
            model, cluster, aa = feature_parts(name)
            order_rows.append(dict(fold=k, input_position_0based=j, feature=name, model=model,
                                   cluster=cluster, aa=aa, original_column_position_0based=original_positions[name],
                                   scaler_mean=pipe["scl"].mean_[j], scaler_scale=pipe["scl"].scale_[j],
                                   train_variance=pipe["scl"].var_[j]))
        log(f"Refit fold {k}: n_train={len(tr)}, n_test={len(te)}, parameters={params}")
    assert np.isfinite(prediction).all() and (fold_id >= 0).all()
    diff = prediction - original[ARM].to_numpy()
    prediction_table = design.copy()
    prediction_table["fold"] = fold_id
    prediction_table["original_oof_probability"] = original[ARM]
    prediction_table["refit_oof_probability"] = prediction
    prediction_table["oof_difference"] = diff
    prediction_table.to_csv(OUT / "tables/patient_oof_reproduction.tsv", sep="\t")
    pd.DataFrame(order_rows).to_csv(OUT / "tables/fold_feature_order.tsv", sep="\t", index=False)
    dump_json(OUT / "audit/frozen_fold_models.json", fold_records)
    gate = dict(arm=ARM, n_patients=len(X), n_positive=int(y.sum()), n_folds=5,
                source_oof_auc=float(roc_auc_score(y, original[ARM])),
                refit_oof_auc=float(roc_auc_score(y, prediction)),
                absolute_tolerance=OOF_ATOL, relative_tolerance=0,
                max_absolute_difference=float(np.abs(diff).max()),
                n_exceeding_tolerance=int((np.abs(diff) > OOF_ATOL).sum()),
                by_fold=[dict(fold=k, n_test=int((fold_id == k).sum()),
                              max_absolute_difference=float(np.abs(diff[fold_id == k]).max())) for k in range(5)])
    gate["status"] = "PASS" if gate["n_exceeding_tolerance"] == 0 else "FAIL"
    dump_json(OUT / "audit/oof_reproduction.json", gate)
    log(f"OOF reproduction {gate['status']}; max |difference|={gate['max_absolute_difference']:.3g}")
    if gate["status"] != "PASS":
        (OUT / "README.md").write_text("# OOF reproduction failed\n\nNo SHAP was computed. See audit/oof_reproduction.json and tables/patient_oof_reproduction.tsv. Frozen feature order, parameters and refitted models are saved for diagnosis. Original files remain untouched.\n")
        raise RuntimeError("OOF reproduction failed: stop before SHAP explanations")

    # No SHAP calculation is allowed until the all-patient prediction gate passes.
    phi = pd.DataFrame(0.0, index=design.index, columns=union)
    selected = pd.DataFrame(False, index=design.index, columns=union)
    standardized = pd.DataFrame(np.nan, index=design.index, columns=union)
    base = np.full(len(X), np.nan)
    long_parts, fold_importance, shap_audit = [], [], []
    for k, (pipe, names, tr, te) in enumerate(fold_models):
        Zte = pipe["scl"].transform(X.iloc[te].loc[:, names].to_numpy(dtype=np.float64))
        explainer = shap.TreeExplainer(pipe["clf"], data=None, model_output="raw",
                                       feature_perturbation="tree_path_dependent", feature_names=names)
        values = explainer.shap_values(Zte, check_additivity=True, approximate=False)
        values = values[1] if isinstance(values, list) else np.asarray(values)[:, :, 1]
        expected = float(np.asarray(explainer.expected_value)[1])
        assert values.shape == (len(te), len(names)) and np.isfinite(values).all()
        errors = expected + values.sum(axis=1) - prediction[te]
        assert np.max(np.abs(errors)) <= ADDITIVITY_ATOL
        keys = design.index[te]
        phi.loc[keys, names] = values
        selected.loc[keys, names] = True
        standardized.loc[keys, names] = Zte
        base[te] = expected
        encoded = X.loc[keys, names].to_numpy()
        long_parts.append(pd.DataFrame(dict(
            patient_key=np.repeat(keys, len(names)), fold=k,
            input_position_0based=np.tile(np.arange(len(names)), len(te)),
            feature=np.tile(names, len(te)), y=np.repeat(y[te], len(names)),
            encoded_value=encoded.ravel(), standardized_value=Zte.ravel(),
            encoded_nonzero=(encoded > 0).ravel(),
            shap_probability=values.ravel(), expected_probability=expected,
            refit_oof_probability=np.repeat(prediction[te], len(names)))))
        for j, name in enumerate(names):
            fold_importance.append(dict(fold=k, feature=name, n_test=len(te), selected=True,
                                        mean_abs_shap_probability=float(np.abs(values[:, j]).mean()),
                                        mean_signed_shap_probability=float(values[:, j].mean())))
        shap_audit.append(dict(fold=k, n_test=len(te), expected_probability=expected,
                               max_absolute_additivity_error=float(np.max(np.abs(errors))),
                               output_unit="P(y=1) probability", model_output_argument="raw",
                               feature_perturbation="tree_path_dependent", background="outer-training tree path counts",
                               explained_patients="outer-test only", approximate=False))
        log(f"SHAP fold {k}: base={expected:.9f}, max additivity error={np.max(np.abs(errors)):.3g}")
    assert selected.sum(axis=1).eq(50).all() and np.isfinite(base).all()
    reconstruction = base + phi.sum(axis=1).to_numpy()
    prediction_table["expected_probability"] = base
    prediction_table["sum_shap_probability"] = phi.sum(axis=1)
    prediction_table["reconstructed_probability"] = reconstruction
    prediction_table["additivity_error"] = reconstruction - prediction
    prediction_table.to_csv(OUT / "tables/patient_predictions_and_additivity.tsv", sep="\t")
    for name, frame in (("heldout_shap_probability", phi),
                        ("heldout_feature_values_encoded", X),
                        ("heldout_feature_selected_mask", selected.astype(np.uint8)),
                        ("heldout_features_standardized", standardized)):
        frame.to_csv(OUT / "tables" / f"{name}.tsv.gz", sep="\t", compression="gzip")
    long = pd.concat(long_parts, ignore_index=True)
    assert len(long) == 289 * 50 and not long.duplicated(["patient_key", "feature"]).any()
    long.to_csv(OUT / "tables/heldout_shap_long.tsv.gz", sep="\t", index=False, compression="gzip")
    per_fold = pd.DataFrame(fold_importance).set_index(["fold", "feature"])
    per_fold = per_fold.reindex(pd.MultiIndex.from_product([range(5), union], names=["fold", "feature"]))
    per_fold["selected"] = per_fold["selected"].fillna(False).astype(bool)
    for column in ("mean_abs_shap_probability", "mean_signed_shap_probability"):
        per_fold[column] = per_fold[column].fillna(0.0)
    per_fold["n_test"] = [int((fold_id == k).sum()) for k, _ in per_fold.index]
    per_fold.to_csv(OUT / "tables/feature_importance_by_fold.tsv", sep="\t")
    records = []
    for name in union:
        rows = per_fold.xs(name, level="feature")
        model, cluster, aa = feature_parts(name)
        values = phi[name].to_numpy()
        mask = selected[name].to_numpy()
        records.append(dict(feature=name, model=model, cluster=cluster, aa=aa,
                            n_folds_selected=int(rows.selected.sum()), selection_frequency=float(rows.selected.mean()),
                            n_heldout_patients_selected=int(mask.sum()),
                            patient_weighted_mean_abs_shap=float(np.abs(values).mean()),
                            patient_weighted_mean_signed_shap=float(values.mean()),
                            fold_equal_mean_abs_shap=float(rows.mean_abs_shap_probability.mean()),
                            fold_equal_sd_abs_shap=float(rows.mean_abs_shap_probability.std(ddof=1)),
                            fold_equal_mean_signed_shap=float(rows.mean_signed_shap_probability.mean()),
                            selected_only_mean_abs_shap=float(np.abs(values[mask]).mean()),
                            n_encoded_nonzero_all_patients=int((X[name] > 0).sum())))
    importance = pd.DataFrame(records).sort_values(["patient_weighted_mean_abs_shap", "feature"], ascending=[False, True])
    importance.insert(0, "rank_patient_weighted", np.arange(1, len(importance) + 1))
    importance.to_csv(OUT / "tables/feature_importance.tsv", sep="\t", index=False)
    dump_json(OUT / "audit/shap_additivity.json", dict(status="PASS", unit="probability", class_label=1,
              absolute_tolerance=ADDITIVITY_ATOL,
              max_absolute_error=float(np.abs(reconstruction - prediction).max()), folds=shap_audit))
    unchanged = {name: sha256(path) == provenance[name]["sha256"] for name, path in paths.items()}
    assert all(unchanged.values())
    dump_json(OUT / "audit/source_unchanged.json", unchanged)
    summary = dict(status="COMPLETE", arm=ARM, n_patients=289, n_positive=110, n_negative=179,
                   n_features_union=188, features_per_fold=50, n_selected_patient_feature_rows=len(long),
                   oof_reproduction=gate, shap_by_fold=shap_audit,
                   max_absolute_additivity_error=float(np.abs(reconstruction - prediction).max()),
                   original_sources_unchanged=all(unchanged.values()), runtime_seconds=round(time.time() - START, 3),
                   limitation="Three source clusters were preselected using pooled289 labels outside CV; exploratory, non-nested cluster screening remains.")
    dump_json(OUT / "audit/completion.json", summary)
    log(f"COMPLETE in {summary['runtime_seconds']} seconds")


if __name__ == "__main__":
    main()
