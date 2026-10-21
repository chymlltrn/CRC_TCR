#!/usr/bin/env python3
"""Exact full-Leiden AA-to-patient mapping and frozen patient-level associations.

The graph is transductive across Home and External. These associations are
exploratory within a shared atlas; this program does not estimate unbiased
out-of-fold prediction or certify antigen binding / predictive contribution.
"""
from __future__ import annotations
from release_runtime import private_literal as _private_literal, release_path as _release_path
import os
for _name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_name] = "8"
import argparse
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import resource
import time
import warnings
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import psutil
from scipy import sparse
from scipy.stats import chi2, fisher_exact
import statsmodels.api as sm
from statsmodels.tools.sm_exceptions import PerfectSeparationWarning
pa.set_cpu_count(8)
pa.set_io_thread_count(2)
ROOT = Path(_release_path('path_cdf25e571cdc1f16'))
ATLAS = ROOT.parent
STATS = ROOT / "stats"
CONFIG = ROOT / "config"
DESIGN = CONFIG / "statistical_design.json"
EXPECTED_VERTICES = 28_380_380

def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()

def dump(path, obj, immutable=False):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if immutable and path.exists():
        if json.loads(path.read_text()) != obj:
            raise ValueError(f"Frozen artifact differs: {path}")
        return
    tmp = path.with_name(path.name + ".part")
    tmp.write_text(json.dumps(obj, sort_keys=True, indent=2, allow_nan=False) + "\n")
    tmp.replace(path)

def save_pq(path, frame):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".part")
    frame.to_parquet(tmp, index=False)
    tmp.replace(path)

def save_tsv(path, frame):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".part")
    frame.to_csv(tmp, sep="\t", index=False)
    tmp.replace(path)

def save_npz(path, matrix):
    path = Path(path)
    tmp = path.with_name(path.stem + ".part.npz")
    sparse.save_npz(tmp, matrix, compressed=True)
    tmp.replace(path)

def stamp():
    return datetime.now(timezone.utc).isoformat()

def memcheck():
    rss = psutil.Process().memory_info().rss
    if rss > 128 * (1 << 30):
        raise MemoryError("Statistical task exceeded 128 GiB RSS sublimit")
    return rss

def resource_info():
    return {"peak_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            "cpu_threads": 8, "rss_sublimit_gib": 128}

def make_design():
    return {
        "version": 2,
        "status": "frozen_before_new_leiden_N_association",
        "endpoint": "recorded N+; pN / ypN not established",
        "analysis_unit": "patient",
        "expected_vertices": EXPECTED_VERTICES,
        "expected_libraries": 340,
        "expected_patients": 339,
        "expected_home_patients": 137,
        "expected_external_patients": 202,
        "expected_home_known": 126,
        "expected_external_known": 163,
        "embedding_graph_scope": "all unique AA across Home and External",
        "interpretation": "exploratory associations in a transductive joint atlas",
        "not_estimated": ["independent External validation",
                          "leakage-free out-of-fold predictive contribution",
                          "antigen binding"],
        "primary_scale": "carrier (at least one complete eligible AA)",
        "primary_eligibility": {"home_known_carriers_min": 10,
                                "home_known_noncarriers_min": 5},
        "abundance_eligibility": {"home_known_carriers_min": 10},
        "sensitivity_scales": {
            "unique_fraction_log_per_sd": "z(log((group_unique_AA + 1) / (patient_unique_AA + 1)))",
            "read_fraction_log_per_sd": "z(log((group_reads + 1) / (patient_reads + 1)))",
            "home_untreated_carrier": "same frozen carrier group universe, no new carrier threshold"
        },
        "home_adjustment": ["age_years_z", "sex_male",
                            "log1p_total_reads_z", "log1p_total_unique_AA_z",
                            "home_batch_sorted_reference", "neoadjuvant_yes"],
        "external_adjustment": ["log1p_total_reads_z", "log1p_total_unique_AA_z",
                                "technical_batch_sorted_reference"],
        "external_unavailable_covariates": ["age", "sex", "neoadjuvant"],
        "test": "one-df likelihood-ratio adjusted binomial GLM",
        "effect_ci": "Wald 95% interval on odds ratio; sensitivity OR is per feature SD",
        "unstable_fit_rule": "separation/nonfinite/nonconvergence/abs(beta)>10/se>10 => p=1, effect NA",
        "external_effects": "all fixed Home-eligible groups; >=5 carriers and >=5 noncarriers flagged evaluable",
        "multiple_testing": {
            "q_bh_all_predefined_tests": "all eligible Home+External carrier, unique/read abundance, Home untreated carrier tests together; failed fits p=1",
            "q_bh_primary_cohorts": "all eligible Home+External carrier tests together",
            "q_bh_scale_cohort": "all eligible groups within each cohort and scale",
            "decision_q": "q_bh_all_predefined_tests",
            "fisher_unadjusted": "descriptive only, not a candidate selection test"
        },
        "denominators": "all productive standard-AA readCount>0 AA, including singleton or saturated clusters",
        "same_patient_libraries": "sum per-AA reads, union unique AA before cluster counts",
        "saturated_abundance_guard": "g=T yields log((g+1)/(T+1))=0 regardless of depth",
        "graph_lock_requirement": "PASS frozen graph receipt, complete seq_idx labels and matching file hashes",
        "label_access_order": "build counts -> freeze group universe using pre-existing Home known-ID list -> load recorded N values",
        "covariate_rank_policy": "fixed nuisance columns; remove label-blind constant/dependent nuisance columns in declared order and report them",
        "all_cluster_report": "each cluster retained with primary/abundance eligibility and exclusion reason"
    }

def identity():
    use = ["cohort", "sample_id", "patient_key", "age_years", "sex",
           "neoadjuvant", "home_batch", "technical_batch"]
    df = pd.read_csv(ATLAS / "config/cohort_samples.tsv", sep="\t", dtype=str,
                     usecols=use, keep_default_na=False)
    if len(df) != 340 or df[["cohort", "sample_id"]].duplicated().any():
        raise ValueError("Library manifest differs from 340-library lock")
    if not set(df.cohort) == {"home", "external"}:
        raise ValueError("Unexpected cohort")
    for key, part in df.groupby("patient_key", sort=True):
        if len(part[["cohort", "age_years", "sex", "neoadjuvant", "home_batch",
                     "technical_batch"]].drop_duplicates()) != 1:
            raise ValueError("Patient identity/covariate disagreement: " + key)
    patients = df.drop_duplicates("patient_key").sort_values("patient_key").reset_index(drop=True)
    if len(patients) != 339 or patients.cohort.value_counts().to_dict() != {"external": 202, "home": 137}:
        raise ValueError("Patient lock differs from 137 Home + 202 External")
    return df, patients

def verified_labels(labels_path, receipt_path, graph_design):
    receipt = json.loads(Path(receipt_path).read_text())
    if receipt.get("status") != "PASS" or receipt.get("frozen") is not True:
        raise ValueError("Formal full graph receipt is not PASS and frozen")
    if receipt.get("n_vertices") != EXPECTED_VERTICES:
        raise ValueError("Full graph vertex count differs")
    if receipt.get("design_sha256") != sha(graph_design):
        raise ValueError("Graph design hash mismatch")
    if receipt.get("labels_sha256") != sha(labels_path):
        raise ValueError("Cluster label hash mismatch")
    labels = np.load(labels_path, mmap_mode="r")
    if labels.shape != (EXPECTED_VERTICES,) or labels.dtype.kind not in "iu":
        raise ValueError("Invalid full label shape/dtype")
    if labels.min() < 0:
        raise ValueError("Unassigned vertices are forbidden")
    n_clusters = int(labels.max()) + 1
    counts = np.bincount(labels, minlength=n_clusters)
    if np.any(counts == 0) or receipt.get("n_clusters") != n_clusters:
        raise ValueError("Labels are not consecutive populated cluster IDs")
    if int(counts.sum()) != EXPECTED_VERTICES:
        raise ValueError("Vertex conservation failure")
    return labels, counts, receipt

def merge_patient_frames(frames):
    """Collapse repeated same-AA entries across a patient's libraries exactly."""
    frame = pd.concat(frames, ignore_index=True)
    if frame.aa.isna().any() or frame.read_count.isna().any() or (frame.read_count <= 0).any():
        raise ValueError("Invalid QC-filtered AA/read_count")
    before = int(frame.read_count.sum())
    merged = frame.groupby("aa", sort=False, as_index=False).read_count.sum()
    if int(merged.read_count.sum()) != before or merged.aa.duplicated().any():
        raise ValueError("Same-patient read/AA conservation failure")
    return merged

def exact_indices(aa_values, lookup):
    ids = np.fromiter((lookup.get(aa, -1) for aa in aa_values), dtype=np.int64,
                      count=len(aa_values))
    if np.any(ids < 0):
        bad = [aa_values[i] for i in np.flatnonzero(ids < 0)[:5]]
        raise ValueError("AA missing from atlas: " + repr(bad))
    return ids

def aggregate_clusters(ids, counts):
    order = np.argsort(ids, kind="stable")
    ordered = ids[order]
    starts = np.r_[0, np.flatnonzero(ordered[1:] != ordered[:-1]) + 1]
    cluster_ids = ordered[starts]
    unique_counts = np.diff(np.r_[starts, len(ids)]).astype(np.int64)
    read_counts = np.add.reduceat(np.asarray(counts, dtype=np.int64)[order], starts)
    return cluster_ids, unique_counts, read_counts

def build(args):
    began = time.monotonic()
    dump(DESIGN, make_design(), immutable=True)
    labels, vertex_counts, graph_receipt = verified_labels(args.labels, args.labels_receipt, args.graph_design)
    n_clusters = len(vertex_counts)
    existing_mapping = STATS / "patient_mapping_receipt.json"
    if existing_mapping.exists():
        previous = json.loads(existing_mapping.read_text())
        if (previous.get("status") == "PASS" and previous.get("labels_sha256") == graph_receipt["labels_sha256"]
                and previous.get("graph_receipt_sha256") == sha(args.labels_receipt)
                and previous.get("graph_design_sha256") == sha(args.graph_design)):
            read_verified_mapping()
            print(json.dumps({"stage": "patient_mapping", "status": "resumed_verified_PASS"}), flush=True)
            return
        raise ValueError("Existing formal mapping does not match graph; retain old artifacts and diagnose")
    libraries, patients = identity()
    union = ATLAS / "data/aa_union.parquet"
    bank = json.loads((ATLAS / "embeddings/bank_manifest.json").read_text())
    # The atlas AA union, not old bank order, is the seq_idx interface.
    union_sha = sha(union)
    if union_sha != bank["atlas_union_sha256"] or bank["n_atlas"] != EXPECTED_VERTICES:
        raise ValueError("Atlas AA union differs from verified embedding-bank manifest")
    if Path(bank["atlas_union_path"]) != union:
        raise ValueError("Unexpected AA union source path")
    if graph_receipt.get("atlas_union_sha256") != union_sha:
        raise ValueError("Graph AA ordering hash differs from patient-mapping AA source")
    lookup = {}
    seen_position = 0
    for b in pq.ParquetFile(union).iter_batches(batch_size=262144, columns=["seq_idx", "aa"]):
        idx = b.column(0).to_numpy()
        if not np.array_equal(idx, np.arange(seen_position, seen_position + len(idx))):
            raise ValueError("Atlas seq_idx is not contiguous in union file order")
        lookup.update(zip(b.column(1).to_pylist(), map(int, idx)))
        seen_position += len(idx)
        memcheck()
        if seen_position % (8 * 262144) == 0:
            print(json.dumps({"stage": "global_AA_lookup", "seq_idx_complete": seen_position,
                              "seq_idx_total": EXPECTED_VERTICES,
                              "elapsed_seconds": time.monotonic() - began}), flush=True)
    if seen_position != EXPECTED_VERTICES or len(lookup) != EXPECTED_VERTICES:
        raise ValueError("Atlas unique AA / seq_idx count mismatch")
    rows, cols, uc, rc = [], [], [], []
    home_seen = np.zeros(EXPECTED_VERTICES, dtype=bool)
    external_seen = np.zeros(EXPECTED_VERTICES, dtype=bool)
    library_rows, total_rows, source_hashes = [], [], {}
    label_sha = graph_receipt["labels_sha256"]
    cache_dir = STATS / "patient_nodes"
    for row_idx, patient in patients.iterrows():
        part = libraries.loc[libraries.patient_key.eq(patient.patient_key)]
        frames = []
        patient_source_hashes = {}
        for library in part.itertuples(index=False):
            path = ATLAS / "data/patient_aa" / f"{library.cohort}__{library.sample_id}.parquet"
            patient_source_hashes[str(path)] = source_hashes[str(path)] = sha(path)
            raw = pd.read_parquet(path)
            if not (raw.cohort.eq(library.cohort).all() and raw.sample_id.eq(library.sample_id).all()
                    and raw.patient_key.eq(library.patient_key).all()):
                raise ValueError("Library parquet identity differs: " + str(path))
            if raw.aa.duplicated().any():
                raise ValueError("Library AA table is not deduplicated")
            library_rows.append({"cohort": library.cohort, "sample_id": library.sample_id,
                                 "patient_key": library.patient_key, "unique_aa": len(raw),
                                 "total_reads": int(raw.read_count.sum()), "source_sha256": source_hashes[str(path)]})
            frames.append(raw[["aa", "read_count"]])
        merged = merge_patient_frames(frames)
        seq_idx = exact_indices(merged.aa.tolist(), lookup)
        cluster = np.asarray(labels[seq_idx], dtype=np.int64)
        cid, uu, rr = aggregate_clusters(cluster, merged.read_count.to_numpy(np.int64))
        if int(uu.sum()) != len(merged) or int(rr.sum()) != int(merged.read_count.sum()):
            raise ValueError("Patient-cluster count conservation failure")
        (home_seen if patient.cohort == "home" else external_seen)[seq_idx] = True
        rows.append(np.full(len(cid), row_idx, dtype=np.int32))
        cols.append(cid)
        uc.append(uu)
        rc.append(rr)
        total_rows.append({"patient_key": patient.patient_key, "cohort": patient.cohort,
                           "library_count": len(part), "total_unique_aa": len(merged),
                           "total_reads": int(merged.read_count.sum()),
                           "mapped_unique_aa": int(uu.sum()), "mapped_reads": int(rr.sum()),
                           "unique_aa_coverage": 1.0, "read_coverage": 1.0})
        cache = cache_dir / f"{patient.patient_key}.parquet"
        save_pq(cache, pd.DataFrame({"seq_idx": seq_idx, "cluster_id": cluster.astype(np.int32),
                                    "read_count": merged.read_count.to_numpy(np.int64)}))
        dump(cache.with_suffix(".receipt.json"),
             {"status": "PASS", "patient_key": patient.patient_key, "cohort": patient.cohort,
              "n_unique_aa": len(merged), "total_reads": int(merged.read_count.sum()),
              "library_source_sha256": patient_source_hashes, "labels_sha256": label_sha,
              "nodes_sha256": sha(cache)})
        memcheck()
        print(json.dumps({"stage": "patient_mapping", "patient_complete": row_idx + 1,
                          "patient_total": len(patients), "rss_bytes": psutil.Process().memory_info().rss}), flush=True)
    del lookup
    if not np.all(home_seen | external_seen):
        raise ValueError("Full atlas vertices not conserved across patient mapping")
    rowvec, colvec = np.concatenate(rows), np.concatenate(cols)
    unique = sparse.csr_matrix((np.concatenate(uc), (rowvec, colvec)), shape=(len(patients), n_clusters), dtype=np.int64)
    reads = sparse.csr_matrix((np.concatenate(rc), (rowvec, colvec)), shape=unique.shape, dtype=np.int64)
    totals = pd.DataFrame(total_rows)
    if not np.array_equal(np.asarray(unique.sum(axis=1)).ravel(), totals.total_unique_aa.to_numpy()):
        raise ValueError("Unique-AA CSR denominator mismatch")
    if not np.array_equal(np.asarray(reads.sum(axis=1)).ravel(), totals.total_reads.to_numpy()):
        raise ValueError("Read CSR denominator mismatch")
    save_npz(STATS / "patient_cluster_unique_counts.npz", unique)
    save_npz(STATS / "patient_cluster_read_counts.npz", reads)
    save_pq(STATS / "patient_totals.parquet", totals)
    save_tsv(STATS / "patient_totals.tsv", totals)
    save_pq(STATS / "patient_identity.parquet", patients)
    save_tsv(STATS / "library_mapping_qc.tsv", pd.DataFrame(library_rows))
    present = unique.copy()
    present.data[:] = 1
    home_mask = patients.cohort.eq("home").to_numpy()
    external_mask = ~home_mask
    cluster_info = pd.DataFrame({
        "cluster_id": np.arange(n_clusters), "global_unique_aa": vertex_counts,
        "singleton": vertex_counts == 1,
        "home_distinct_aa": np.bincount(labels[home_seen], minlength=n_clusters),
        "external_distinct_aa": np.bincount(labels[external_seen], minlength=n_clusters),
        "shared_home_external_distinct_aa": np.bincount(labels[home_seen & external_seen], minlength=n_clusters),
        "home_patient_carriers": np.asarray(present[home_mask].sum(axis=0)).ravel(),
        "external_patient_carriers": np.asarray(present[external_mask].sum(axis=0)).ravel(),
        "home_patient_unique_aa_sum": np.asarray(unique[home_mask].sum(axis=0)).ravel(),
        "external_patient_unique_aa_sum": np.asarray(unique[external_mask].sum(axis=0)).ravel(),
        "home_reads_sum": np.asarray(reads[home_mask].sum(axis=0)).ravel(),
        "external_reads_sum": np.asarray(reads[external_mask].sum(axis=0)).ravel()})
    cluster_info["cohort_exclusive"] = np.select(
        [cluster_info.external_distinct_aa.eq(0), cluster_info.home_distinct_aa.eq(0)],
        ["home_only", "external_only"], default="both")
    for cohort, column in [("home", "home_batch"), ("external", "technical_batch")]:
        for batch in sorted(patients.loc[patients.cohort.eq(cohort), column].unique()):
            mask = patients.cohort.eq(cohort) & patients[column].eq(batch)
            cluster_info[f"{cohort}_{batch}_patient_carriers"] = np.asarray(present[mask.to_numpy()].sum(axis=0)).ravel()
    save_pq(STATS / "cluster_descriptive.parquet", cluster_info)
    outputs = ["patient_cluster_unique_counts.npz", "patient_cluster_read_counts.npz",
               "patient_totals.parquet", "patient_identity.parquet", "cluster_descriptive.parquet",
               "library_mapping_qc.tsv"]
    dump(STATS / "patient_mapping_receipt.json",
         {"status": "PASS", "n_vertices": EXPECTED_VERTICES, "n_clusters": n_clusters,
          "n_libraries": len(libraries), "n_patients": len(patients),
          "n_home": int(home_mask.sum()), "n_external": int(external_mask.sum()),
          "AA_mapped_missing": 0, "unmapped_vertices": 0, "read_conservation": True,
          "unique_patient_union_conservation": True, "N_values_accessed": False,
          "source_manifest_sha256": sha(ATLAS / "config/cohort_samples.tsv"),
          "atlas_aa_union_sha256": union_sha, "bank_manifest_sha256": sha(ATLAS / "embeddings/bank_manifest.json"),
          "labels_sha256": label_sha, "labels_path": str(args.labels.resolve()),
          "graph_receipt_path": str(args.labels_receipt.resolve()),
          "graph_design_path": str(args.graph_design.resolve()),
          "graph_receipt_sha256": sha(args.labels_receipt),
          "graph_design_sha256": sha(args.graph_design), "statistical_design_sha256": sha(DESIGN),
          "library_source_sha256": source_hashes,
          "output_sha256": {f: sha(STATS / f) for f in outputs},
          "elapsed_seconds": time.monotonic() - began,
          "patient_AA_rows": int(totals.total_unique_aa.sum()),
          "patient_AA_rows_per_second": float(totals.total_unique_aa.sum() / (time.monotonic() - began)),
          "completed_utc": stamp(), **resource_info()})

def read_verified_mapping():
    receipt = json.loads((STATS / "patient_mapping_receipt.json").read_text())
    if receipt.get("status") != "PASS" or receipt.get("N_values_accessed") is not False:
        raise ValueError("Mapping has not passed without new N-value access")
    if receipt["statistical_design_sha256"] != sha(DESIGN):
        raise ValueError("Statistical design changed")
    if receipt["source_manifest_sha256"] != sha(ATLAS / "config/cohort_samples.tsv"):
        raise ValueError("Clinical/source manifest changed")
    for filename, digest in receipt["output_sha256"].items():
        if sha(STATS / filename) != digest:
            raise ValueError("Frozen mapping file changed: " + filename)
    for field, digest_field in [("labels_path", "labels_sha256"),
                                 ("graph_receipt_path", "graph_receipt_sha256"),
                                 ("graph_design_path", "graph_design_sha256")]:
        if sha(receipt[field]) != receipt[digest_field]:
            raise ValueError("Formal graph source changed after mapping: " + field)
    unique = sparse.load_npz(STATS / "patient_cluster_unique_counts.npz")
    reads = sparse.load_npz(STATS / "patient_cluster_read_counts.npz")
    identity_df = pd.read_parquet(STATS / "patient_identity.parquet")
    totals = pd.read_parquet(STATS / "patient_totals.parquet")
    info = pd.read_parquet(STATS / "cluster_descriptive.parquet")
    if unique.shape != reads.shape or unique.shape != (339, receipt["n_clusters"]):
        raise ValueError("Patient matrix dimensions differ")
    if identity_df.patient_key.tolist() != totals.patient_key.tolist():
        raise ValueError("Patient CSR row-order mismatch")
    return unique, reads, identity_df, totals, info, receipt

def eligibility(carrier_counts, n_known):
    carrier = np.asarray(carrier_counts, int)
    noncarrier = n_known - carrier
    primary = (carrier >= 10) & (noncarrier >= 5)
    abundance = carrier >= 10
    reason = np.where(carrier < 10, "fewer_than_10_Home_known_carriers",
                      np.where(noncarrier < 5, "fewer_than_5_Home_known_noncarriers_or_saturated", "eligible"))
    return primary, abundance, reason

def freeze():
    dump(DESIGN, make_design(), immutable=True)
    unique, reads, patients, totals, info, mapping = read_verified_mapping()
    folds = ATLAS / "folds/folds.tsv"
    # No outcome column is loaded: only membership in the previously fixed known-ID set.
    known_ids = set(pd.read_csv(folds, sep="\t", dtype=str, usecols=["patient_key"]).patient_key)
    home = patients.cohort.eq("home") & patients.patient_key.isin(known_ids)
    if len(known_ids) != 126 or int(home.sum()) != 126:
        raise ValueError("Pre-existing Home known-ID lock changed")
    present = unique[home.to_numpy()].copy()
    present.data[:] = 1
    n_carriers = np.asarray(present.sum(axis=0)).ravel().astype(int)
    primary, abundance, reason = eligibility(n_carriers, 126)
    universe = info.copy()
    universe["home_known_carriers"] = n_carriers
    universe["home_known_noncarriers"] = 126 - n_carriers
    universe["primary_carrier_eligible"] = primary
    universe["abundance_sensitivity_eligible"] = abundance
    universe["primary_exclusion_reason"] = reason
    target = STATS / "frozen_cluster_universe.parquet"
    if target.exists():
        old = pd.read_parquet(target)
        pd.testing.assert_frame_equal(old, universe)
    else:
        save_pq(target, universe)
    hypotheses = []
    for cohort in ("home", "external"):
        for scale, selected in [("carrier", primary), ("unique_fraction_log_per_sd", abundance),
                                ("read_fraction_log_per_sd", abundance)]:
            hypotheses.extend({"cohort": cohort, "scale": scale, "cluster_id": int(cid)}
                              for cid in np.flatnonzero(selected))
    hypotheses.extend({"cohort": "home_untreated", "scale": "carrier", "cluster_id": int(cid)}
                      for cid in np.flatnonzero(primary))
    hypothesis_file = STATS / "frozen_hypotheses.parquet"
    hdf = pd.DataFrame(hypotheses, columns=["cohort", "scale", "cluster_id"])
    if hypothesis_file.exists():
        pd.testing.assert_frame_equal(pd.read_parquet(hypothesis_file), hdf)
    else:
        save_pq(hypothesis_file, hdf)
    lock = {"status": "frozen_before_new_leiden_N_values",
            "analysis_unit": "patient", "N_values_accessed": False,
            "n_clusters": len(info), "n_primary_carrier_clusters": int(primary.sum()),
            "n_abundance_sensitivity_clusters": int(abundance.sum()),
            "n_home_known_ids": 126, "home_known_ids_sha256": hashlib.sha256("\n".join(sorted(known_ids)).encode()).hexdigest(),
            "previous_fold_file_sha256": sha(folds),
            "cluster_universe_sha256": sha(target), "hypotheses_sha256": sha(hypothesis_file),
            "n_hypotheses": len(hdf), "statistical_design_sha256": sha(DESIGN),
            "mapping_receipt_sha256": sha(STATS / "patient_mapping_receipt.json"),
            "joint_graph_interpretation": "same-atlas cohort recheck; not independent External validation"}
    dump(STATS / "hypothesis_universe_lock.json", lock, immutable=True)
    print(json.dumps(lock), flush=True)

def standardized(values):
    v = np.asarray(values, float)
    if not np.isfinite(v).all():
        raise ValueError("Nonfinite feature")
    sd = float(v.std(ddof=0))
    return (v - v.mean()) / sd if sd > 0 else np.zeros_like(v)

def nuisance_design(patients, totals, cohort):
    data = patients.reset_index(drop=True)
    totals = totals.reset_index(drop=True)
    columns = [("intercept", np.ones(len(data)))]
    if cohort in ("home", "home_untreated"):
        age = pd.to_numeric(data.age_years, errors="raise").to_numpy(float)
        if not data.sex.isin(["m", "f"]).all() or not data.neoadjuvant.isin(["yes", "no"]).all():
            raise ValueError("Home covariates absent or invalid")
        columns += [("age_years_z", standardized(age)),
                    ("sex_male", data.sex.eq("m").to_numpy(float))]
    for col in ("total_reads", "total_unique_aa"):
        columns.append(("log1p_" + col + "_z", standardized(np.log1p(totals[col].to_numpy(float)))))
    batch_col = "home_batch" if cohort in ("home", "home_untreated") else "technical_batch"
    levels = sorted(data[batch_col].unique())
    if any(level == "" for level in levels):
        raise ValueError("Missing batch covariate")
    columns += [(batch_col + "_" + level, data[batch_col].eq(level).to_numpy(float)) for level in levels[1:]]
    if cohort == "home":
        columns.append(("neoadjuvant_yes", data.neoadjuvant.eq("yes").to_numpy(float)))
    kept, dropped, x = [], [], np.empty((len(data), 0))
    for name, col in columns:
        candidate = np.column_stack((x, col))
        if np.linalg.matrix_rank(candidate) > x.shape[1]:
            kept.append(name)
            x = candidate
        else:
            dropped.append(name)
    return x, kept, dropped, levels[0]

def glm(y, x):
    with warnings.catch_warnings():
        warnings.simplefilter("error", PerfectSeparationWarning)
        warnings.simplefilter("ignore", RuntimeWarning)
        fit = sm.GLM(y, x, family=sm.families.Binomial()).fit(maxiter=200, tol=1e-8, disp=0)
    if not fit.converged or not np.isfinite(fit.params).all() or not np.isfinite(fit.bse).all() or not np.isfinite(fit.llf):
        raise ValueError("Nonfinite or nonconverged GLM")
    return fit

def effect(y, x, null_fit, feature):
    out = {"p": 1.0, "beta": np.nan, "se": np.nan, "odds_ratio": np.nan,
           "ci_lower": np.nan, "ci_upper": np.nan, "status": "unstable"}
    f = np.asarray(feature, float)
    if not np.isfinite(f).all() or np.std(f) < 1e-12:
        out["status"] = "constant_feature"
        return out
    xx = np.column_stack((x, f))
    if np.linalg.matrix_rank(xx) != xx.shape[1]:
        out["status"] = "rank_deficient"
        return out
    if null_fit is None:
        out["status"] = "baseline_fit_failed"
        return out
    try:
        model = glm(y, xx)
        beta, se = float(model.params[-1]), float(model.bse[-1])
        lr = 2 * (float(model.llf) - float(null_fit.llf))
        if abs(beta) > 10 or se > 10 or lr < -1e-6 or not np.isfinite(lr):
            out["status"] = "extreme_or_invalid_fit"
            return out
        out.update(p=float(chi2.sf(max(lr, 0), 1)), beta=beta, se=se,
                   odds_ratio=float(np.exp(beta)), ci_lower=float(np.exp(beta - 1.96 * se)),
                   ci_upper=float(np.exp(beta + 1.96 * se)), status="ok")
    except Exception as exc:
        out["status"] = "fit_error_" + type(exc).__name__
    return out

def bh(values):
    p = np.asarray(values, float)
    if not np.isfinite(p).all() or np.any((p < 0) | (p > 1)):
        raise ValueError("Invalid frozen-family P values")
    if not len(p):
        return np.array([], float)
    order = np.argsort(p, kind="stable")
    q = np.empty_like(p)
    q[order] = np.minimum.accumulate((p[order] * len(p) / np.arange(1, len(p) + 1))[::-1])[::-1].clip(0, 1)
    return q

def score():
    began = time.monotonic()
    lock_path = STATS / "hypothesis_universe_lock.json"
    lock = json.loads(lock_path.read_text())
    if lock.get("status") != "frozen_before_new_leiden_N_values":
        raise ValueError("Hypothesis universe has not been frozen")
    for filename, key in [("frozen_cluster_universe.parquet", "cluster_universe_sha256"),
                           ("frozen_hypotheses.parquet", "hypotheses_sha256"),
                           ("patient_mapping_receipt.json", "mapping_receipt_sha256")]:
        if sha(STATS / filename) != lock[key]:
            raise ValueError("Frozen hypothesis artifact changed")
    if lock["statistical_design_sha256"] != sha(DESIGN):
        raise ValueError("Frozen statistical design changed")
    unique, reads, patients, totals, info, mapping = read_verified_mapping()
    universe = pd.read_parquet(STATS / "frozen_cluster_universe.parquet")
    hypotheses = pd.read_parquet(STATS / "frozen_hypotheses.parquet")
    # First access to outcome VALUES in this new analysis occurs after the lock.
    labels = pd.read_csv(ATLAS / "config/cohort_samples.tsv", sep="\t", dtype=str,
                         usecols=["patient_key", "recorded_n_binary"], keep_default_na=False)
    for key, part in labels.groupby("patient_key", sort=False):
        if part.recorded_n_binary.nunique(dropna=False) != 1:
            raise ValueError("Same-patient N labels disagree: " + key)
    labels = labels.drop_duplicates("patient_key").set_index("patient_key").recorded_n_binary
    labels = labels.loc[patients.patient_key].reset_index(drop=True)
    known = labels.isin(["0", "1"]).to_numpy()
    cohorts, out, designs = {}, [], {}
    for cohort in ("home", "external", "home_untreated"):
        selected = known & patients.cohort.eq("home" if cohort == "home_untreated" else cohort).to_numpy()
        if cohort == "home_untreated":
            selected &= patients.neoadjuvant.eq("no").to_numpy()
        pos = np.flatnonzero(selected)
        y = labels.iloc[pos].eq("1").to_numpy(int)
        if cohort == "home" and (len(pos), int(y.sum())) != (126, 56):
            raise ValueError("Home known label composition differs")
        if cohort == "external" and (len(pos), int(y.sum())) != (163, 54):
            raise ValueError("External known label composition differs")
        x, kept, dropped, reference = nuisance_design(patients.iloc[pos], totals.iloc[pos], cohort)
        try:
            null = glm(y, x) if len(np.unique(y)) == 2 else None
            baseline_status = "ok" if null is not None else "one_class_only"
        except Exception as exc:
            null, baseline_status = None, "fit_error_" + type(exc).__name__
        missing_counts = {"age_years": int(patients.iloc[pos].age_years.eq("").sum()),
                          "sex": int((~patients.iloc[pos].sex.isin(["m", "f"])).sum()),
                          "neoadjuvant": int((~patients.iloc[pos].neoadjuvant.isin(["yes", "no"])).sum())}
        designs[cohort] = {"complete_case_patients_dropped": 0,
                           "clinical_covariate_missing_counts": missing_counts,
                           "adjustment_scope": "reduced_technical_only" if cohort == "external" else "clinical_and_technical",
                           "n_patients": len(pos), "n_Nplus": int(y.sum()), "n_N0": int((1-y).sum()),
                           "columns_kept": kept, "columns_dropped_label_blind": dropped,
                           "batch_reference": reference, "baseline_status": baseline_status}
        cov_df = pd.DataFrame(x, columns=kept)
        cov_df.insert(0, "patient_key", patients.iloc[pos].patient_key.to_numpy())
        save_tsv(STATS / f"{cohort}_covariate_design.tsv", cov_df)
        cohorts[cohort] = (pos, y, x, null)
    for h in hypotheses.itertuples(index=False):
        pos, y, x, null = cohorts[h.cohort]
        u = unique[pos, h.cluster_id].toarray().ravel().astype(float)
        r = reads[pos, h.cluster_id].toarray().ravel().astype(float)
        carrier = u > 0
        if h.scale == "carrier":
            z = carrier.astype(float)
        elif h.scale == "unique_fraction_log_per_sd":
            z = standardized(np.log((u + 1) / (totals.iloc[pos].total_unique_aa.to_numpy(float) + 1)))
        elif h.scale == "read_fraction_log_per_sd":
            z = standardized(np.log((r + 1) / (totals.iloc[pos].total_reads.to_numpy(float) + 1)))
        else:
            raise ValueError("Unknown frozen scale")
        fit = effect(y, x, null, z)
        a, b = int(carrier[y == 1].sum()), int(carrier[y == 0].sum())
        plus, zero = int(y.sum()), int((1-y).sum())
        fp = float(fisher_exact([[a, plus-a], [b, zero-b]])[1]) if plus and zero else 1.0
        out.append({"cohort": h.cohort, "scale": h.scale, "cluster_id": h.cluster_id,
                    "n_patients": len(pos), "n_Nplus": plus, "n_N0": zero,
                    "carriers_Nplus": a, "carriers_N0": b, "carriers_total": a+b,
                    "noncarriers_total": len(pos)-a-b,
                    "prevalence_Nplus": a/plus if plus else np.nan,
                    "prevalence_N0": b/zero if zero else np.nan,
                    "prevalence_difference": a/plus-b/zero if plus and zero else np.nan,
                    "external_evaluable_5_carrier_5_noncarrier": a+b >= 5 and len(pos)-a-b >= 5,
                    "fisher_p_unadjusted_descriptive": fp,
                    "unique_fraction_mean_Nplus": float(np.mean(u[y == 1] / totals.iloc[pos].total_unique_aa.to_numpy(float)[y == 1])) if plus else np.nan,
                    "unique_fraction_mean_N0": float(np.mean(u[y == 0] / totals.iloc[pos].total_unique_aa.to_numpy(float)[y == 0])) if zero else np.nan,
                    "read_fraction_mean_Nplus": float(np.mean(r[y == 1] / totals.iloc[pos].total_reads.to_numpy(float)[y == 1])) if plus else np.nan,
                    "read_fraction_mean_N0": float(np.mean(r[y == 0] / totals.iloc[pos].total_reads.to_numpy(float)[y == 0])) if zero else np.nan,
                    **fit})
        if len(out) % 500 == 0:
            print(json.dumps({"stage": "association", "completed": len(out),
                              "total": len(hypotheses)}), flush=True)
            memcheck()
    effects = pd.DataFrame(out, columns=["cohort", "scale", "cluster_id", "n_patients", "n_Nplus", "n_N0",
                          "carriers_Nplus", "carriers_N0", "carriers_total", "noncarriers_total",
                          "prevalence_Nplus", "prevalence_N0", "prevalence_difference",
                          "external_evaluable_5_carrier_5_noncarrier", "fisher_p_unadjusted_descriptive",
                          "unique_fraction_mean_Nplus", "unique_fraction_mean_N0",
                          "read_fraction_mean_Nplus", "read_fraction_mean_N0",
                          "p", "beta", "se", "odds_ratio", "ci_lower", "ci_upper", "status"])
    if len(effects) != lock["n_hypotheses"]:
        raise ValueError("Frozen tested hypothesis count mismatch")
    effects["q_bh_all_predefined_tests"] = bh(effects.p.to_numpy())
    effects["q_bh_primary_cohorts"] = np.nan
    primary = effects.scale.eq("carrier") & effects.cohort.isin(["home", "external"])
    effects.loc[primary, "q_bh_primary_cohorts"] = bh(effects.loc[primary, "p"].to_numpy())
    effects["q_bh_scale_cohort"] = np.nan
    for (cohort, scale), sub in effects.groupby(["cohort", "scale"], sort=False):
        effects.loc[sub.index, "q_bh_scale_cohort"] = bh(sub.p.to_numpy())
    effects["nominal_p_lt_0_05"] = effects.p.lt(.05) & effects.status.eq("ok")
    effects["exploratory_association_q_lt_0_05"] = effects.q_bh_all_predefined_tests.lt(.05) & effects.status.eq("ok")
    effects["validated_Nplus_predictive_contribution"] = False
    save_pq(STATS / "associations_all_predefined_tests.parquet", effects)
    save_tsv(STATS / "associations_all_predefined_tests.tsv", effects)
    summary = universe.copy()
    for cohort in ("home", "external", "home_untreated"):
        for scale in ("carrier", "unique_fraction_log_per_sd", "read_fraction_log_per_sd"):
            selected = effects.cohort.eq(cohort) & effects.scale.eq(scale)
            if not selected.any():
                continue
            sub = effects.loc[selected].drop(columns=["cohort", "scale"])
            prefix = cohort + "_" + scale + "_"
            sub = sub.rename(columns={col: prefix+col for col in sub.columns if col != "cluster_id"})
            summary = summary.merge(sub, on="cluster_id", how="left", validate="one_to_one")
    for scale in ("carrier", "unique_fraction_log_per_sd", "read_fraction_log_per_sd"):
        h, e = f"home_{scale}_beta", f"external_{scale}_beta"
        if h in summary and e in summary:
            summary[f"{scale}_same_adjusted_direction"] = np.isfinite(summary[h]) & np.isfinite(summary[e]) & (summary[h]*summary[e] > 0)
    summary["interpretation"] = "exploratory_joint_atlas_association_not_validated_predictive_contribution"
    save_pq(STATS / "all_cluster_effects_summary.parquet", summary)
    save_tsv(STATS / "all_cluster_effects_summary.tsv", summary)
    dump(STATS / "association_receipt.json",
         {"status": "PASS", "endpoint": "recorded N+", "analysis_unit": "patient",
          "joint_atlas_external_interpretation": "same-atlas exploratory cohort recheck; not independent validation",
          "n_clusters": len(universe), "n_hypotheses": len(effects),
          "n_primary_carrier_clusters": lock["n_primary_carrier_clusters"],
          "n_abundance_sensitivity_clusters": lock["n_abundance_sensitivity_clusters"],
          "n_primary_ineligible_saturated_or_near_saturated": int(((universe.home_known_carriers >= 10) & (universe.home_known_noncarriers < 5)).sum()),
          "n_nominal_p_lt_0_05": int(effects.nominal_p_lt_0_05.sum()),
          "n_association_global_q_lt_0_05": int(effects.exploratory_association_q_lt_0_05.sum()),
          "validated_Nplus_predictive_contribution_not_assessed": True,
          "cohort_designs": designs, "hypothesis_lock_sha256": sha(lock_path),
          "statistical_design_sha256": sha(DESIGN),
          "output_sha256": {name: sha(STATS / name) for name in
                           ["associations_all_predefined_tests.parquet", "associations_all_predefined_tests.tsv",
                            "all_cluster_effects_summary.parquet", "all_cluster_effects_summary.tsv"]},
          "elapsed_seconds": time.monotonic() - began,
          "hypotheses_per_second": len(effects) / (time.monotonic() - began),
          "completed_utc": stamp(), **resource_info()})

def selftest():
    # Shared AA across two libraries is one clone type but both read counts contribute.
    f1 = pd.DataFrame({"aa": ["CASSA", "CASSB"], "read_count": [3, 5]})
    f2 = pd.DataFrame({"aa": ["CASSA", "CASSC"], "read_count": [7, 11]})
    merged = merge_patient_frames([f1, f2]).set_index("aa")
    assert len(merged) == 3 and int(merged.read_count.sum()) == 26
    assert int(merged.loc["CASSA", "read_count"]) == 10
    # Non-lexical embedding-bank order must not change atlas seq_idx labels.
    lookup = {"CASSC": 0, "CASSA": 1, "CASSB": 2}
    labels = np.array([2, 0, 0])
    idx = exact_indices(["CASSA", "CASSB", "CASSC"], lookup)
    cid, u, r = aggregate_clusters(labels[idx], np.array([10, 5, 11]))
    assert cid.tolist() == [0, 2] and u.tolist() == [2, 1] and r.tolist() == [15, 11]
    assert int(u.sum()) == 3 and int(r.sum()) == 26
    try:
        exact_indices(["ABSENT"], lookup)
        raise AssertionError("Missing AA was silently mapped")
    except ValueError:
        pass
    primary, abundance, reason = eligibility(np.array([0, 9, 10, 121, 122, 126]), 126)
    assert primary.tolist() == [False, False, True, True, False, False]
    assert abundance.tolist() == [False, False, True, True, True, True]
    # Saturated carrier is unidentifiable, and remains p=1 (not an apparent effect).
    y = np.array([0, 0, 1, 1, 0, 1])
    x = np.ones((6, 1))
    fit = effect(y, x, glm(y, x), np.ones(6))
    assert fit["status"] == "constant_feature" and fit["p"] == 1
    # A cluster covering the whole repertoire cannot become variable from pseudocounts.
    depth = np.array([1, 10, 100, 1000, 10000, 100000], dtype=float)
    saturated = standardized(np.log((depth + 1) / (depth + 1)))
    saturated_fit = effect(y, x, glm(y, x), saturated)
    assert np.array_equal(saturated, np.zeros(6))
    assert saturated_fit["status"] == "constant_feature" and saturated_fit["p"] == 1
    # BH includes failed predeclared tests at p=1, rather than discarding them.
    assert np.allclose(bh([.01, .04, 1]), [.03, .06, 1])
    return {"status": "PASS", "checks": ["same_patient_AA_union_and_reads",
            "exact_seq_idx_label_mapping", "missing_AA_rejected", "fixed_carrier_eligibility",
            "saturated_carrier_nonestimable", "saturated_abundance_not_depth_artifact",
            "failed_tests_retained_in_BH"]}

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["build", "freeze", "score", "run", "selftest"])
    parser.add_argument("--labels", type=Path, default=ROOT / "clusters/primary_labels.npy")
    parser.add_argument("--labels-receipt", type=Path, default=ROOT / "clusters/primary_receipt.json")
    parser.add_argument("--graph-design", type=Path, default=ROOT / "config/graph_design.json")
    args = parser.parse_args()
    if args.command == "selftest":
        result = selftest()
        dump(ROOT / "audit/patient_cluster_stats_selftest.json", result)
        print(json.dumps(result))
        return
    if args.command in ("build", "run"):
        build(args)
    if args.command in ("freeze", "run"):
        freeze()
    if args.command in ("score", "run"):
        score()

if __name__ == "__main__":
    main()
