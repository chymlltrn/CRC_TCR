#!/usr/bin/env python3
"""Read back the saved repair package and check label/value/model alignment."""
from release_runtime import private_literal as _private_literal, release_path as _release_path
import os
for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[key] = "2"
import hashlib
import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd

OUT = Path(_release_path('path_6f3f3692efa62d86'))
TABLES = OUT / "tables"


def read(name, index_col=0):
    return pd.read_csv(TABLES / name, sep="\t", index_col=index_col, float_precision="round_trip")


phi = read("heldout_shap_probability.tsv.gz")
encoded = read("heldout_feature_values_encoded.tsv.gz")
mask = read("heldout_feature_selected_mask.tsv.gz").astype(bool)
scaled = read("heldout_features_standardized.tsv.gz")
pred = read("patient_predictions_and_additivity.tsv")
long = read("heldout_shap_long.tsv.gz", index_col=None)
importance = read("feature_importance.tsv", index_col="feature")
order = read("fold_feature_order.tsv", index_col=None)
fold_importance = read("feature_importance_by_fold.tsv", index_col=None)
source_folds = json.loads((OUT / "source_snapshots/fold_selection_clusterTCR.json").read_text())[
    "RandomForest__CLUSTERTCR__stratLabelBatchCohort"]
assert phi.shape == (289, 188) and phi.index.is_unique and phi.columns.is_unique
for frame in (encoded, mask, scaled):
    assert frame.index.equals(phi.index) and frame.columns.equals(phi.columns)
assert pred.index.equals(phi.index)
assert np.isfinite(phi.to_numpy()).all() and np.isfinite(encoded.to_numpy()).all()
assert mask.sum(axis=1).eq(50).all()
assert (phi.to_numpy()[~mask.to_numpy()] == 0).all()
assert np.array_equal(scaled.notna().to_numpy(), mask.to_numpy())
assert len(long) == 14450 and not long.duplicated(["patient_key", "feature"]).any()
max_model_difference = 0.0
max_scaled_difference = 0.0
mismatches = []
for k in range(5):
    bundle = joblib.load(OUT / "models" / f"fold_{k}_randomforest.joblib")
    names = bundle["features"]
    assert names == source_folds[k]["features"]
    od = order[order.fold == k].sort_values("input_position_0based")
    assert od.feature.tolist() == names and od.input_position_0based.tolist() == list(range(50))
    keys = bundle["test_patients"]
    assert set(bundle["train_patients"]).isdisjoint(keys)
    assert set(bundle["train_patients"]) | set(keys) == set(phi.index)
    assert set(keys) == set(pred.index[pred.fold == k])
    assert mask.loc[keys, names].all().all()
    z = bundle["pipeline"]["scl"].transform(encoded.loc[keys, names].to_numpy())
    max_scaled_difference = max(max_scaled_difference,
                                float(np.max(np.abs(z - scaled.loc[keys, names].to_numpy()))))
    p = bundle["pipeline"]["clf"].predict_proba(z)[:, 1]
    max_model_difference = max(max_model_difference,
                              float(np.max(np.abs(p - pred.loc[keys, "refit_oof_probability"].to_numpy()))))
    subset = long[long.fold == k]
    for patient, rows in subset.groupby("patient_key", sort=False):
        rows = rows.sort_values("input_position_0based")
        assert rows.feature.tolist() == names
        np.testing.assert_array_equal(rows.shap_probability.to_numpy(), phi.loc[patient, names].to_numpy())
        np.testing.assert_array_equal(rows.encoded_value.to_numpy(), encoded.loc[patient, names].to_numpy())
    sorted_names = od.sort_values("original_column_position_0based").feature.tolist()
    mismatches.append(sum(a != b for a, b in zip(names, sorted_names)))
assert max_model_difference < 1e-12 and max_scaled_difference < 1e-12
reconstructed = pred.expected_probability.to_numpy() + phi.sum(axis=1).to_numpy()
max_additivity_error = float(np.max(np.abs(reconstructed - pred.refit_oof_probability.to_numpy())))
max_oof_error = float(np.max(np.abs(pred.original_oof_probability - pred.refit_oof_probability)))
assert max_additivity_error < 1e-10 and max_oof_error < 1e-12
recalculated = phi.abs().mean(axis=0)
np.testing.assert_allclose(recalculated.to_numpy(),
                           importance.loc[phi.columns, "patient_weighted_mean_abs_shap"].to_numpy(),
                           atol=1e-15, rtol=0)
assert importance.rank_patient_weighted.tolist() == list(range(1, 189))
assert importance.patient_weighted_mean_abs_shap.is_monotonic_decreasing
assert fold_importance.shape[0] == 5 * 188
for name in phi.columns:
    rows = fold_importance[fold_importance.feature == name]
    np.testing.assert_allclose(rows.mean_abs_shap_probability.mean(),
                               importance.loc[name, "fold_equal_mean_abs_shap"], atol=1e-15, rtol=0)
    assert int(rows.selected.sum()) == int(importance.loc[name, "n_folds_selected"])
    assert int(mask[name].sum()) == int(importance.loc[name, "n_heldout_patients_selected"])
checks = dict(status="PASS", n_patients=289, n_features_union=188, selected_features_per_patient=50,
              long_rows=14450, n_unique_aa=int(importance.aa.nunique()),
              source_sorted_vs_frozen_feature_position_mismatches_by_fold=mismatches,
              max_model_reloaded_prediction_difference=max_model_difference,
              max_reloaded_scaled_input_difference=max_scaled_difference,
              max_oof_probability_error=max_oof_error, max_additivity_error=max_additivity_error,
              checks=["exact frozen feature order in all five models",
                      "disjoint train/test patients and complete fold coverage",
                      "all serialized matrices share exact patient/feature axes",
                      "structural zeros distinguishable by selection mask",
                      "long-table values and names match wide matrices exactly",
                      "reloaded scalers and models reproduce saved inputs and probabilities",
                      "patient-weighted and fold-equal importance recomputed from saved SHAP",
                      "original OOF reproduction and probability additivity remain within tolerances"])
(OUT / "audit/readback_checks.json").write_text(json.dumps(checks, indent=2) + "\n")
print(json.dumps(checks, indent=2))
