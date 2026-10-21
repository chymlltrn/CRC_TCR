#!/usr/bin/env python3
"""Post-run acceptance and actual full-atlas Leiden counts.

Polling uses only the standard library. Numerical/dataframe modules are loaded
after runner COMPLETE, with four compute threads. Synthetic smoke counts never
appear in the full-atlas report.
"""
from __future__ import annotations
from release_runtime import private_literal as _private_literal, release_path as _release_path
import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import tempfile
import time

BASE = Path(_release_path('path_cdf25e571cdc1f16'))
N_VERTICES = 28_380_380
PRIOR_AUDITED_STATS_SHA = "ff9f1a01dfbfb87d89ad6373d68a184fcbd750c062ea40ada83cd1c6bbb3c381"
REQUIRED_STAGES = ("geometry50k", "pilot1M", "index_full", "knn_full",
                   "graph_full", "leiden_primary", "patient_stats", "leiden_scale_counts")
NUMERICALS = None

class AcceptanceError(RuntimeError):
    pass

def need(condition, message):
    if not condition:
        raise AcceptanceError(message)

def sha(path):
    path = Path(path)
    need(path.is_file(), "Missing file: " + str(path))
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()

def valid_sha(value):
    return isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value)

def load_json(path):
    path = Path(path)
    need(path.is_file(), "Missing receipt: " + str(path))
    try:
        value = json.loads(path.read_text())
    except (OSError, ValueError) as exc:
        raise AcceptanceError("Unreadable receipt: " + str(path)) from exc
    need(isinstance(value, dict), "Receipt is not an object: " + str(path))
    return value

def passing(path, expected_source=None):
    obj = load_json(path)
    need(obj.get("status") == "PASS", "Receipt is not PASS: " + str(path))
    if expected_source is not None:
        need(obj.get("source_code_sha256") == expected_source,
             "Receipt does not bind audited graph source: " + str(path))
    return obj

def atomic_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".part")
    tmp.write_text(json.dumps(obj, sort_keys=True, indent=2, allow_nan=False) + "\n")
    tmp.replace(path)

def now():
    return datetime.now(timezone.utc).isoformat()

def numerical_modules():
    global NUMERICALS
    if NUMERICALS is None:
        for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                    "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
            os.environ[var] = "4"
        if hasattr(os, "sched_getaffinity"):
            current = sorted(os.sched_getaffinity(0))
            os.sched_setaffinity(0, current[:4])
        import numpy as np
        import pandas as pd
        import pyarrow as pa
        pa.set_cpu_count(4)
        pa.set_io_thread_count(2)
        NUMERICALS = (np, pd)
    return NUMERICALS

def wait_for_complete(wait):
    """Do not write or print while runtime state is unchanged."""
    path = BASE / "audit/runner_state.json"
    while True:
        state = load_json(path)
        status = state.get("status")
        if status == "COMPLETE":
            stages = {row["stage"]: row for row in state.get("stages", [])}
            for name in REQUIRED_STAGES:
                need(name in stages, "Runner omits required stage: " + name)
                need(stages[name].get("status") in ("PASS", "PASS_REUSED"),
                     "Runner stage is not successful: " + name)
                if "exit_code" in stages[name]:
                    need(stages[name]["exit_code"] == 0, "Nonzero runner exit: " + name)
            return state
        if status in ("FAILED", "FAIL", "STOPPED", "CANCELLED"):
            raise AcceptanceError("Runner " + str(status) + ": " + str(state.get("error", "")))
        need(status in ("RUNNING", "PENDING"), "Unknown runner status: " + str(status))
        if not wait:
            raise AcceptanceError("Runner is still " + str(status) + "; final full counts are pending")
        time.sleep(20)

def audited_sources(base):
    audit_path = base / "audit/independent_review.json"
    audit = load_json(audit_path)
    for name in ("graph", "stats"):
        need(str(audit.get("source_review", {}).get(name, "")).startswith("PASS"),
             "Independent source review does not pass: " + name)
    graph_expected = audit.get("graph_source_sha256")
    stats_expected = audit.get("stats_source_sha256", PRIOR_AUDITED_STATS_SHA)
    need(valid_sha(graph_expected) and valid_sha(stats_expected), "Missing audited source hashes")
    graph_actual = sha(base / "src/full_leiden_graph.py")
    stats_actual = sha(base / "src/patient_cluster_stats.py")
    need(graph_actual == graph_expected, "Audited graph source hash mismatch")
    need(stats_actual == stats_expected, "Audited statistics source hash mismatch")
    return audit, {"graph_source_sha256": graph_actual, "stats_source_sha256": stats_actual,
                   "stats_hash_basis": "audit/independent_review.json.stats_source_sha256"
                       if "stats_source_sha256" in audit else
                       "prior audited final statistics source SHA supplied by root; never a mismatch bypass",
                   "independent_review_sha256": sha(audit_path)}

def graph_gate(base, cfg, graph_source):
    atlas = base.parent
    design_sha = sha(base / "config/graph_design.json")
    need(cfg.get("frozen") is True and cfg.get("freeze_before_any_phenotype_read") is True,
         "Graph design is not formally frozen before outcomes")
    need(cfg.get("n_vertices") == N_VERTICES and cfg.get("dimension") == 64 and cfg.get("k") == 30,
         "Graph scope/geometry differs from approved full analysis")
    need(cfg.get("parameters_selected_using_N") is False and cfg.get("phenotype_tables_read") == [],
         "Graph parameters use phenotype information")
    need(cfg.get("primary_resolution") == 1 and cfg.get("primary_seed") == 17
         and cfg.get("repeat_seed") == 29
         and set(cfg.get("full_sensitivity_resolutions", [])) == {.5, 2},
         "Frozen resolution/seed schedule differs")
    need(cfg.get("joint_graph_is_exploratory") is True
         and cfg.get("external_independent_validation") is False,
         "Joint-graph interpretation boundary changed")
    frozen = passing(base / "audit/design_freeze.json", graph_source)
    need(frozen.get("frozen") is True and frozen.get("design_sha256") == design_sha,
         "Freeze receipt is not bound to current design")
    need(frozen.get("resolution_selection_by_N") is False
         and frozen.get("phenotype_tables_read") == [],
         "Freeze receipt permits phenotype selection")
    bootstrap = passing(base / "audit/bootstrap_inputs.json")
    expected = {cfg["atlas_union_path"]: cfg["atlas_union_sha256"],
                cfg["atlas_index_path"]: cfg["atlas_index_sha256"],
                cfg["pca_scores_path"]: cfg["pca_scores_sha256"]}
    bootstrap_paths = {row["path"]: row for row in bootstrap["sources"].values()}
    for path, digest in expected.items():
        need(path in bootstrap_paths and bootstrap_paths[path]["sha256"] == digest,
             "Bootstrap source binding differs: " + path)
        need(Path(path).is_file() and Path(path).stat().st_size == bootstrap_paths[path]["size"],
             "Bootstrap input is missing or its size changed: " + path)
    projection = load_json(cfg["projection_receipt_path"])
    need(projection.get("status") == "complete" and projection.get("n_aa") == N_VERTICES
         and projection.get("missing") == 0 and projection.get("nonfinite") == 0,
         "Full PCA projection is not complete")
    need(projection.get("scores_sha256") == cfg["pca_scores_sha256"]
         and projection.get("input_index_sha256") == cfg["atlas_index_sha256"],
         "PCA projection/index source mismatch")
    index = passing(base / "graph/index_receipt.json", graph_source)
    need(index.get("n_vertices") == N_VERTICES and index.get("dimension") == cfg["dimension"]
         and index.get("source_sha256") == cfg["pca_scores_sha256"],
         "Full ANN index source/size mismatch")
    need(valid_sha(index.get("index_sha256")), "ANN index lacks checksum")
    recall = passing(base / "audit/fullbank_ann_recall.json", graph_source)
    need(recall.get("exact_database_rows") == N_VERTICES
         and recall.get("source_sha256") == cfg["pca_scores_sha256"]
         and recall.get("index_sha256") == index["index_sha256"]
         and recall.get("k") == cfg["k"]
         and recall.get("selected_nprobe") == cfg["selected_nprobe"],
         "Full-bank ANN recall source/selection mismatch")
    need(recall.get("recall_target") == cfg["recall_target"] and cfg["recall_target"] >= .95,
         "ANN recall target was weakened")
    selected = [r for r in recall["results"] if r["nprobe"] == cfg["selected_nprobe"]]
    need(len(selected) == 1 and selected[0]["strict_id_recall_mean"] >= cfg["recall_target"],
         "Selected ANN setting does not pass full-bank strict-ID recall")
    for kind in ("indices", "distances"):
        need(sha(recall["exact_" + kind + "_path"]) == recall["exact_" + kind + "_sha256"],
             "Exact recall array hash changed")
        need(sha(selected[0]["approx_" + kind + "_path"]) == selected[0]["approx_" + kind + "_sha256"],
             "Selected ANN recall array hash changed")
    query_path = base / "pilot/global_recall_queries.parquet"
    need(sha(query_path) == recall["query_sha256"], "Recall query identity changed")
    np, pd = numerical_modules()
    exact_i = np.load(recall["exact_indices_path"], allow_pickle=False)
    approx_i = np.load(selected[0]["approx_indices_path"], allow_pickle=False)
    need(exact_i.shape == approx_i.shape == (cfg["recall_queries"], cfg["k"]),
         "Exact/ANN recall matrices have wrong shape")
    independently_recalled = float(np.mean([len(set(a) & set(b)) / cfg["k"]
                                           for a, b in zip(exact_i, approx_i)]))
    need(math.isclose(independently_recalled, selected[0]["strict_id_recall_mean"], abs_tol=1e-12)
         and independently_recalled >= cfg["recall_target"], "ANN recall recomputation mismatch")
    knn = passing(base / "graph/knn_receipt.json", graph_source)
    need(knn.get("n_vertices") == N_VERTICES and knn.get("completed_rows") == N_VERTICES
         and knn.get("failures") == 0 and knn.get("k") == cfg["k"]
         and knn.get("nprobe") == cfg["selected_nprobe"]
         and knn.get("design_sha256") == design_sha
         and knn.get("index_sha256") == index["index_sha256"],
         "Full kNN receipt binding/coverage failed")
    cursor = 0
    shard_hashes = {}
    for path in knn["shards_receipts"]:
        row = passing(path, graph_source)
        need(row.get("start") == cursor and row.get("end", 0) > cursor
             and row["end"] <= N_VERTICES and row.get("n_rows") == row["end"] - cursor,
             "kNN shard coverage has a gap/overlap")
        need(row.get("source_sha256") == cfg["pca_scores_sha256"]
             and row.get("design_sha256") == design_sha
             and row.get("index_sha256") == index["index_sha256"]
             and row.get("nprobe") == cfg["selected_nprobe"] and row.get("k") == cfg["k"],
             "kNN shard source binding mismatch")
        for kind in ("indices", "distances"):
            need(valid_sha(row.get(kind + "_sha256")) and Path(row[kind + "_path"]).is_file(),
                 "Missing kNN shard data/checksum")
        shard_hashes[str(path)] = sha(path)
        cursor = row["end"]
    need(cursor == N_VERTICES and len(shard_hashes) == knn["shards"],
         "kNN receipts do not cover full atlas")
    graph = passing(base / "graph/graph_receipt.json", graph_source)
    need(graph.get("n_vertices") == N_VERTICES
         and graph.get("n_directed_neighbors") == N_VERTICES * cfg["k"]
         and graph.get("design_sha256") == design_sha
         and graph.get("all_vertices_retained") is True and graph.get("no_self_loops") is True
         and graph.get("symmetric_rule") == cfg["symmetric_rule"],
         "Full graph receipt binding/coverage failed")
    for kind in ("edges", "weights", "edgelist"):
        need(valid_sha(graph.get(kind + "_sha256")) and Path(graph[kind + "_path"]).is_file(),
             "Graph data/checksum missing: " + kind)
    return graph, {"design_sha256": design_sha, "strict_id_recall_mean": independently_recalled,
                   "selected_nprobe": cfg["selected_nprobe"],
                   "large_input_hash_basis": "audited full graph construction and Leiden consumer receipts; large vector/index/edge datasets are not reread",
                   "knn_shard_receipt_sha256": shard_hashes}

def geometry_gate(base, cfg):
    atlas = base.parent
    geom = passing(base / "pilot/geometry50k_receipt.json")
    need(geom.get("n_pool") == 50000 and geom.get("no_N_labels_accessed") is True,
         "Geometry diagnostic scope/outcome access differs")
    need(geom.get("source_code_sha256") == sha(base / "src/geometry_diagnostic.py"),
         "Geometry diagnostic source hash mismatch")
    bank_path, verification_path = atlas / "embeddings/bank_manifest.json", atlas / "embeddings/verification.json"
    bank, verification = load_json(bank_path), passing(verification_path)
    need(geom.get("source_manifest_sha256") == sha(bank_path)
         and geom.get("verification_receipt_sha256") == sha(verification_path),
         "Geometry diagnostic lacks current frozen-bank manifest bindings")
    model = bank["model_banks"]["tcrbert_dualtask_aa_mean"]
    verify_model = verification["models"]["tcrbert_dualtask_aa_mean"]
    sceptr_projection = load_json(atlas / "atlas/joint/sceptr_cdr3_only/projection_receipt.json")
    need(sceptr_projection.get("status") == "complete" and sceptr_projection.get("n_aa") == N_VERTICES
         and sceptr_projection.get("missing") == 0 and sceptr_projection.get("nonfinite") == 0,
         "SCEPTR diagnostic input projection incomplete")
    expected = {cfg["atlas_index_path"]: cfg["atlas_index_sha256"],
                cfg["pca_scores_path"]: cfg["pca_scores_sha256"],
                str(atlas / "atlas/joint/sceptr_cdr3_only/pca_scores_fp16.npy"): sceptr_projection["scores_sha256"],
                model["old_bank_path"]: verify_model["old_bank_sha256"],
                model["new_bank_path"]: verify_model["new_bank_sha256"]}
    inputs = geom.get("source_inputs", {})
    required_paths = set(expected) | {cfg["balanced_draws_path"]}
    need(set(inputs) == required_paths, "Geometry diagnostic source manifest lacks an input or contains unexpected inputs")
    for path, data in inputs.items():
        actual = Path(path).stat()
        need(actual.st_size == data.get("size") and actual.st_mtime_ns == data.get("mtime_ns"),
             "Geometry input stat changed since diagnostic: " + path)
        need(valid_sha(data.get("sha256")) and data.get("hash_basis"),
             "Geometry input hash/basis missing")
        if path in expected:
            need(data["sha256"] == expected[path], "Geometry source hash differs from frozen upstream input: " + path)
        if path in (model["old_bank_path"], model["new_bank_path"]):
            need(data["hash_basis"].startswith("prior_frozen_full_verification_receipt"),
                 "Full bank geometry hash lacks declared prior full verification basis")
        else:
            need(data["hash_basis"] == "fresh_complete_file_read_sha256",
                 "Geometry PCA/index/draw hash must derive from a fresh whole-file read")
    need(sha(base / "pilot/geometry50k_seq_idx.npy") == geom["indices_sha256"],
         "Geometry sampled sequence IDs changed")
    need(sha(base / "pilot/geometry50k_raw768.npy") == geom.get("raw_sample_sha256"),
         "Geometry raw sample hash missing/changed")
    # Draws are small and lack a separate upstream checksum; bind freshly here.
    need(sha(cfg["balanced_draws_path"]) == inputs[cfg["balanced_draws_path"]]["sha256"],
         "Geometry balanced draws changed")
    return {"status": "PASS", "receipt_sha256": sha(base / "pilot/geometry50k_receipt.json"),
            "source_inputs": inputs, "source_manifest_sha256": geom["source_manifest_sha256"],
            "verification_receipt_sha256": geom["verification_receipt_sha256"],
            "raw_sample_sha256": geom["raw_sample_sha256"],
            "metrics": geom["metrics"], "scope": geom["scope"],
            "not_fullbank_clustering": True}

def cluster_paths(base, resolution, seed):
    if resolution == 1 and seed == 17:
        return base / "clusters/primary_labels.npy", base / "clusters/primary_receipt.json"
    stem = "resolution_" + str(float(resolution)).replace(".", "p") + "_seed_" + str(seed)
    return base / "clusters" / (stem + "_labels.npy"), base / "clusters" / (stem + "_receipt.json")

def inspect_labels(label_path, receipt_path, n_vertices, binding, resolution, seed):
    np, pd = numerical_modules()
    receipt = passing(receipt_path, binding["graph_source_sha256"])
    need(receipt.get("n_vertices") == n_vertices and receipt.get("frozen") is True
         and receipt.get("all_vertices_assigned") is True,
         "Leiden receipt lacks full vertex assignment")
    for key, expected in [("design_sha256", binding["design_sha256"]),
                           ("source_sha256", binding["pca_scores_sha256"]),
                           ("atlas_union_sha256", binding["atlas_union_sha256"]),
                           ("graph_edges_sha256", binding["graph_edges_sha256"])]:
        need(receipt.get(key) == expected, "Leiden receipt source binding failed: " + key)
    for key, expected in binding.get("leiden_parameters", {}).items():
        need(receipt.get(key) == expected, "Leiden frozen algorithm parameter differs: " + key)
    need(receipt.get("joint_all_cohort_graph") is True
         and receipt.get("external_is_independent_validation") is False,
         "Leiden joint/cohort interpretation changed")
    need(receipt.get("resolution") == resolution and receipt.get("seed") == seed,
         "Leiden resolution/seed receipt mismatch")
    need(receipt.get("primary_statistical_partition") is (resolution == 1 and seed == 17),
         "Leiden primary/sensitivity role mismatch")
    need(Path(receipt.get("labels_path", "")).resolve() == Path(label_path).resolve(),
         "Leiden label receipt path differs")
    need(sha(label_path) == receipt.get("labels_sha256"), "Leiden label hash mismatch")
    labels = np.load(label_path, mmap_mode="r", allow_pickle=False)
    need(labels.shape == (n_vertices,) and np.issubdtype(labels.dtype, np.integer),
         "Leiden labels are sampled or wrong dtype")
    need(int(labels.min()) >= 0 and int(labels.max()) < n_vertices,
         "Leiden labels contain missing/invalid vertices")
    bins = np.bincount(labels)
    need(np.all(bins > 0) and int(bins.sum()) == n_vertices,
         "Leiden populated cluster/vertex conservation failed")
    need(len(bins) == receipt.get("n_clusters") and int(bins.max()) == receipt.get("largest_cluster")
         and int((bins == 1).sum()) == receipt.get("singleton_clusters"),
         "Actual label bins disagree with Leiden receipt counts")
    quantiles = {str(q): float(np.quantile(bins, q)) for q in (0, .25, .5, .75, .9, .99, 1)}
    need(receipt.get("cluster_size_quantiles") == quantiles, "Actual cluster quantiles differ")
    need(math.isclose(receipt.get("largest_cluster_fraction", -1), float(bins.max()/n_vertices),
                      abs_tol=1e-15), "Actual largest cluster fraction differs")
    return {"resolution": float(resolution), "seed": seed, "n_vertices": n_vertices,
            "n_clusters": len(bins), "singleton_clusters": int((bins == 1).sum()),
            "largest_cluster": int(bins.max()), "largest_cluster_fraction": float(bins.max()/n_vertices),
            "cluster_size_quantiles": quantiles, "labels_path": str(label_path),
            "labels_sha256": receipt["labels_sha256"], "receipt_sha256": sha(receipt_path),
            "primary_statistical_partition": resolution == 1 and seed == 17,
            "source_code_sha256": receipt["source_code_sha256"],
            "n_iterations": receipt.get("n_iterations"), "leiden_seconds": receipt.get("leiden_seconds"),
            "peak_rss_gib": receipt.get("peak_rss_gib")}, bins

def cluster_runs(base, n_vertices, binding):
    rows, bins_by_role = [], {}
    for resolution, seed in ((.5, 17), (1., 17), (2., 17), (1., 29)):
        lp, rp = cluster_paths(base, resolution, seed)
        row, bins = inspect_labels(lp, rp, n_vertices, binding, resolution, seed)
        rows.append(row)
        bins_by_role[(resolution, seed)] = bins
    stability = passing(base / "clusters/primary_seed_stability.json", binding["graph_source_sha256"])
    need(stability.get("n_vertices") == n_vertices and stability.get("seeds") == [17, 29],
         "Primary seed stability is not full-scope 17/29")
    by_role = {(r["resolution"], r["seed"]): r for r in rows}
    need(stability.get("primary_labels_sha256") == by_role[(1., 17)]["labels_sha256"]
         and stability.get("repeat_labels_sha256") == by_role[(1., 29)]["labels_sha256"],
         "Seed stability labels hash binding differs")
    ari = stability.get("adjusted_rand_index")
    need(isinstance(ari, (int, float)) and math.isfinite(ari) and -1 <= ari <= 1,
         "Seed stability ARI invalid")
    return rows, bins_by_role[(1., 17)], {"status": "PASS", "seeds": [17, 29],
              "n_vertices": n_vertices, "adjusted_rand_index": float(ari),
              "primary_labels_sha256": stability["primary_labels_sha256"],
              "repeat_labels_sha256": stability["repeat_labels_sha256"],
              "receipt_sha256": sha(base / "clusters/primary_seed_stability.json")}

def bh(values):
    np, pd = numerical_modules()
    p = np.asarray(values, float)
    if len(p) == 0:
        return np.array([], float)
    order = np.argsort(p, kind="stable")
    q = np.empty(len(p), float)
    q[order] = np.minimum.accumulate((p[order]*len(p)/np.arange(1, len(p)+1))[::-1])[::-1].clip(0, 1)
    return q

def statistical_gate(base, primary_bins, audited, sources):
    np, pd = numerical_modules()
    stat_design_sha = sha(base / "config/statistical_design.json")
    need(stat_design_sha == audited.get("statistical_design_sha256",
                                       audited.get("stats_evidence", {}).get("statistical_design_sha256")),
         "Statistical design differs from independent reviewed design")
    mapping_path, assoc_path = base / "stats/patient_mapping_receipt.json", base / "stats/association_receipt.json"
    mapping, assoc = passing(mapping_path), passing(assoc_path)
    need((mapping.get("n_vertices"), mapping.get("n_clusters"), mapping.get("n_libraries"),
          mapping.get("n_patients"), mapping.get("n_home"), mapping.get("n_external")) ==
         (N_VERTICES, len(primary_bins), 340, 339, 137, 202), "Patient mapping scope/count mismatch")
    need(mapping.get("AA_mapped_missing") == 0 and mapping.get("unmapped_vertices") == 0
         and mapping.get("read_conservation") is True and mapping.get("unique_patient_union_conservation") is True
         and mapping.get("N_values_accessed") is False, "Patient mapping conservation/unlabelled gate failed")
    need(mapping.get("statistical_design_sha256") == stat_design_sha,
         "Mapping statistics design binding differs")
    for key, target in [("labels_sha256", base / "clusters/primary_labels.npy"),
                         ("graph_receipt_sha256", base / "clusters/primary_receipt.json"),
                         ("graph_design_sha256", base / "config/graph_design.json")]:
        need(mapping.get(key) == sha(target), "Mapping formal graph binding mismatch: " + key)
    bootstrap = load_json(base / "audit/bootstrap_inputs.json")
    clinical_source = base.parent / "config/cohort_samples.tsv"
    need(mapping.get("source_manifest_sha256") == sha(clinical_source)
         == bootstrap["sources"]["config/cohort_samples.tsv"]["sha256"],
         "Mapping clinical/source manifest changed")
    need(len(mapping.get("library_source_sha256", {})) == 340, "Library provenance set incomplete")
    need(set(mapping.get("output_sha256", {})) >= {"patient_cluster_unique_counts.npz",
         "patient_cluster_read_counts.npz", "patient_totals.parquet", "patient_identity.parquet",
         "cluster_descriptive.parquet", "library_mapping_qc.tsv"}, "Mapping output hash manifest incomplete")
    need(set(assoc.get("output_sha256", {})) >= {"associations_all_predefined_tests.parquet",
         "associations_all_predefined_tests.tsv", "all_cluster_effects_summary.parquet",
         "all_cluster_effects_summary.tsv"}, "Association output hash manifest incomplete")
    for filename, digest in mapping.get("output_sha256", {}).items():
        need(sha(base / "stats" / filename) == digest, "Mapping output checksum changed: " + filename)
    for filename, digest in assoc.get("output_sha256", {}).items():
        need(sha(base / "stats" / filename) == digest, "Association output checksum changed: " + filename)
    totals = pd.read_parquet(base / "stats/patient_totals.parquet")
    need(len(totals) == 339 and not totals.patient_key.duplicated().any(), "Patient totals row identity failed")
    need(totals.cohort.value_counts().to_dict() == {"external": 202, "home": 137},
         "Patient totals cohort count differs")
    need(np.array_equal(totals.total_unique_aa, totals.mapped_unique_aa)
         and np.array_equal(totals.total_reads, totals.mapped_reads)
         and totals.unique_aa_coverage.eq(1).all() and totals.read_coverage.eq(1).all(),
         "Patient-level AA/read denominator conservation failed")
    lock_path = base / "stats/hypothesis_universe_lock.json"
    lock = load_json(lock_path)
    need(lock.get("status") == "frozen_before_new_leiden_N_values"
         and lock.get("N_values_accessed") is False and lock.get("n_home_known_ids") == 126,
         "Hypothesis universe was not frozen before N")
    need(lock.get("mapping_receipt_sha256") == sha(mapping_path)
         and lock.get("statistical_design_sha256") == stat_design_sha
         and assoc.get("hypothesis_lock_sha256") == sha(lock_path),
         "Hypothesis/statistics receipt binding mismatch")
    for filename, key in [("frozen_cluster_universe.parquet", "cluster_universe_sha256"),
                           ("frozen_hypotheses.parquet", "hypotheses_sha256")]:
        need(sha(base / "stats" / filename) == lock.get(key), "Frozen hypothesis file changed")
    universe = pd.read_parquet(base / "stats/frozen_cluster_universe.parquet")
    need(len(universe) == len(primary_bins)
         and np.array_equal(universe.cluster_id.to_numpy(), np.arange(len(primary_bins)))
         and np.array_equal(universe.global_unique_aa.to_numpy(), primary_bins),
         "Patient group universe does not match actual primary label bins")
    carrier = universe.home_known_carriers.to_numpy(int)
    noncarrier = universe.home_known_noncarriers.to_numpy(int)
    need(np.all((carrier >= 0) & (carrier <= 126)) and np.array_equal(noncarrier, 126-carrier),
         "Home carrier/noncarrier totals invalid")
    primary, abundance = (carrier >= 10) & (noncarrier >= 5), carrier >= 10
    need(np.array_equal(universe.primary_carrier_eligible, primary)
         and np.array_equal(universe.abundance_sensitivity_eligible, abundance),
         "Frozen eligibility rule changed")
    p_count, a_count = int(primary.sum()), int(abundance.sum())
    saturated = int(((carrier >= 10) & (noncarrier < 5)).sum())
    fully_saturated = int((carrier == 126).sum())
    expected_tests = 3*p_count + 4*a_count
    need(lock.get("n_primary_carrier_clusters") == p_count
         and lock.get("n_abundance_sensitivity_clusters") == a_count
         and lock.get("n_hypotheses") == expected_tests
         and assoc.get("n_primary_carrier_clusters") == p_count
         and assoc.get("n_abundance_sensitivity_clusters") == a_count
         and assoc.get("n_primary_ineligible_saturated_or_near_saturated") == saturated
         and assoc.get("n_hypotheses") == expected_tests
         and assoc.get("n_clusters") == len(primary_bins),
         "Actual eligibility/test counts disagree with receipts")
    for cohort, counts in [("home", (126, 56, 70)), ("external", (163, 54, 109))]:
        d = assoc["cohort_designs"][cohort]
        need((d.get("n_patients"), d.get("n_Nplus"), d.get("n_N0")) == counts,
             "Known patient endpoint counts differ: " + cohort)
        need(d.get("complete_case_patients_dropped") == 0, "Unreported complete-case loss")
    need(assoc["cohort_designs"]["external"].get("adjustment_scope") == "reduced_technical_only",
         "External adjustment scope is not explicitly reduced")
    effects = pd.read_parquet(base / "stats/associations_all_predefined_tests.parquet")
    hypotheses = pd.read_parquet(base / "stats/frozen_hypotheses.parquet")
    key = ["cohort", "scale", "cluster_id"]
    need(len(effects) == expected_tests and len(hypotheses) == expected_tests
         and not effects[key].duplicated().any()
         and effects[key].equals(hypotheses[key]), "Actual tested hypothesis universe differs")
    values = effects.p.to_numpy(float)
    need(np.isfinite(values).all() and np.all((values >= 0) & (values <= 1)),
         "Invalid/missing tested P values")
    need(np.allclose(effects.q_bh_all_predefined_tests.to_numpy(float), bh(values), rtol=0, atol=1e-12),
         "Global BH family omits or changes frozen tests")
    failed = ~effects.status.eq("ok")
    need(effects.loc[failed, "p"].eq(1).all(), "Failed fits not retained at p=1")
    need(not bool(effects.validated_Nplus_predictive_contribution.fillna(True).astype(bool).any()),
         "Association output incorrectly claims validated contribution")
    metric_rows = []
    for (cohort, scale), group in effects.groupby(["cohort", "scale"], sort=True):
        need(np.allclose(group.q_bh_scale_cohort.to_numpy(float), bh(group.p.to_numpy(float)),
                         rtol=0, atol=1e-12), "Scale/cohort BH mismatch")
        ok = group.status.eq("ok")
        nominal = ok & group.p.lt(.05)
        global_sig = ok & group.q_bh_all_predefined_tests.lt(.05)
        family_sig = ok & group.q_bh_scale_cohort.lt(.05)
        metric_rows.append({"cohort": cohort, "scale": scale, "tested_clusters": len(group),
                            "fit_ok": int(ok.sum()), "failed_p1": int((~ok).sum()),
                            "nominal_P_lt_0_05": int(nominal.sum()),
                            "nominal_positive": int((nominal & group.beta.gt(0)).sum()),
                            "nominal_negative": int((nominal & group.beta.lt(0)).sum()),
                            "global_q_lt_0_05": int(global_sig.sum()),
                            "global_q_positive": int((global_sig & group.beta.gt(0)).sum()),
                            "scale_cohort_q_lt_0_05": int(family_sig.sum()),
                            "min_P": float(group.p.min()), "min_global_q": float(group.q_bh_all_predefined_tests.min()),
                            "External_evaluable_5_carrier_5_noncarrier": int(group.external_evaluable_5_carrier_5_noncarrier.sum()) if cohort == "external" else None})
    primary_rows = effects.scale.eq("carrier") & effects.cohort.isin(["home", "external"])
    need(np.allclose(effects.loc[primary_rows, "q_bh_primary_cohorts"].to_numpy(float),
                     bh(effects.loc[primary_rows, "p"].to_numpy(float)), rtol=0, atol=1e-12),
         "Primary pooled Home/External BH family mismatch")
    return {"status": "PASS", "n_libraries": 340, "n_patients": 339, "n_home": 137, "n_external": 202,
            "n_home_known": 126, "n_external_known": 163,
            "n_primary_carrier_clusters": p_count, "n_abundance_sensitivity_clusters": a_count,
            "n_primary_ineligible_saturated_or_near_saturated": saturated,
            "n_fully_saturated_Home_known_carrier_clusters": fully_saturated,
            "n_rare_Home_known_carrier_clusters": int((carrier < 10).sum()),
            "n_hypotheses": expected_tests, "all_patient_AA_read_coverage": 1.0,
            "patient_union_AA_rows": int(totals.total_unique_aa.sum()),
            "patient_total_reads": int(totals.total_reads.sum()),
            "cohort_designs": assoc["cohort_designs"], "association_summary": metric_rows,
            "mapping_receipt_sha256": sha(mapping_path), "association_receipt_sha256": sha(assoc_path),
            "hypothesis_lock_sha256": sha(lock_path), "statistical_design_sha256": stat_design_sha,
            "validated_predictive_contribution": "not assessed by this geometric association analysis"}


def carrier_eligibility_summary(carriers, n_known):
    """Fixed, phenotype-blind eligibility; no N-group values are used."""
    np, pd = numerical_modules()
    carriers = np.asarray(carriers)
    need(np.issubdtype(carriers.dtype, np.integer)
         and np.all((carriers >= 0) & (carriers <= n_known)), "Invalid patient carrier vector")
    primary = (carriers >= 10) & (n_known-carriers >= 5)
    abundance = carriers >= 10
    return {"home_known_patient_count": int(n_known),
            "n_primary_carrier_eligible": int(primary.sum()),
            "n_abundance_sensitivity_eligible": int(abundance.sum()),
            "n_carrier_saturated_or_near_saturated": int(((carriers >= 10) & (n_known-carriers < 5)).sum()),
            "n_fully_saturated_carrier_clusters": int((carriers == n_known).sum()),
            "n_rare_carrier_clusters": int((carriers < 10).sum()),
            "n_absent_from_Home_known_clusters": int((carriers == 0).sum()),
            "home_known_carrier_distribution": {
                str(i): int(v) for i, v in enumerate(np.bincount(carriers, minlength=n_known+1))},
            "carrier_counts_sha256": hashlib.sha256(carriers.astype("<i4").tobytes()).hexdigest(),
            "eligibility_role": "descriptive fixed-rule testability; associations only for primary resolution1 seed17"}

def partition_testability(base, rows, primary_bins, statistics, n_vertices=N_VERTICES, n_known=126):
    """Use hashed per-patient seq_idx caches to assess all frozen partitions.

    Only prior known-patient ID membership is read. No N value, phenotype
    subgroup or association output enters these partition eligibility counts.
    """
    np, pd = numerical_modules()
    mapping_path = base / "stats/patient_mapping_receipt.json"
    lock_path = base / "stats/hypothesis_universe_lock.json"
    mapping = passing(mapping_path)
    lock = load_json(lock_path)
    need(mapping.get("N_values_accessed") is False
         and sha(mapping_path) == statistics["mapping_receipt_sha256"]
         and sha(lock_path) == statistics["hypothesis_lock_sha256"]
         and lock.get("mapping_receipt_sha256") == sha(mapping_path)
         and lock.get("statistical_design_sha256") == statistics["statistical_design_sha256"],
         "Partition testability cache/statistics receipt binding differs")
    need(lock.get("status") == "frozen_before_new_leiden_N_values"
         and lock.get("N_values_accessed") is False and lock.get("n_home_known_ids") == n_known,
         "Partition testability known-patient membership is not frozen")
    folds_path = base.parent / "folds/folds.tsv"
    need(sha(folds_path) == lock["previous_fold_file_sha256"], "Known-ID fold source changed")
    known = pd.read_csv(folds_path, sep="\t", dtype=str, usecols=["patient_key"]).patient_key
    need(not known.isna().any(), "Missing known-patient ID")
    known_ids = sorted(set(known))
    ids_sha = hashlib.sha256("\n".join(known_ids).encode()).hexdigest()
    need(len(known_ids) == n_known and ids_sha == lock["home_known_ids_sha256"],
         "Known-patient ID membership differs from frozen universe")
    clinical_path = base.parent / "config/cohort_samples.tsv"
    need(sha(clinical_path) == mapping["source_manifest_sha256"], "Patient library manifest changed")
    libraries = pd.read_csv(clinical_path, sep="\t", dtype=str,
                            usecols=["cohort", "sample_id", "patient_key"])
    need(not libraries.isna().any().any()
         and not libraries[["cohort", "sample_id"]].duplicated().any(),
         "Patient library identity manifest invalid")
    for filename in ("patient_identity.parquet", "patient_totals.parquet"):
        need(sha(base / "stats" / filename) == mapping.get("output_sha256", {}).get(filename),
             "Patient identity/totals mapping output hash changed: " + filename)
    identity = pd.read_parquet(base / "stats/patient_identity.parquet",
                               columns=["patient_key", "cohort"])
    need(not identity.patient_key.duplicated().any()
         and set(known_ids).issubset(set(identity.loc[identity.cohort.eq("home"), "patient_key"])),
         "Known Home IDs disagree with patient identity")
    totals = pd.read_parquet(base / "stats/patient_totals.parquet")
    need(not totals.patient_key.duplicated().any(), "Duplicate patient totals")
    totals = totals.set_index("patient_key")
    universe_path = base / "stats/frozen_cluster_universe.parquet"
    need(sha(universe_path) == lock["cluster_universe_sha256"], "Frozen primary universe changed")
    universe = pd.read_parquet(universe_path, columns=["cluster_id", "global_unique_aa",
        "home_known_carriers", "primary_carrier_eligible", "abundance_sensitivity_eligible"])
    need(np.array_equal(universe.cluster_id.to_numpy(), np.arange(len(primary_bins)))
         and np.array_equal(universe.global_unique_aa.to_numpy(), primary_bins),
         "Partition testability primary cluster universe differs")
    labels_by_partition, carriers_by_partition = {}, {}
    rows_by_partition = {(row["resolution"], row["seed"]): row for row in rows}
    need(set(rows_by_partition) == {(.5,17), (1.,17), (2.,17), (1.,29)},
         "Partition testability requires all frozen scales and repeat")
    for key, row in rows_by_partition.items():
        label_path = Path(row["labels_path"])
        need(sha(label_path) == row["labels_sha256"], "Partition labels changed before patient projection")
        labels = np.load(label_path, mmap_mode="r", allow_pickle=False)
        need(labels.shape == (n_vertices,), "Partition patient projection label scope differs")
        labels_by_partition[key] = labels
        carriers_by_partition[key] = np.zeros(row["n_clusters"], dtype=np.int32)
    primary_sha = rows_by_partition[(1.,17)]["labels_sha256"]
    node_provenance, source_rows = {}, 0
    for patient_key in known_ids:
        node_path = base / "stats/patient_nodes" / (patient_key + ".parquet")
        receipt_path = node_path.with_suffix(".receipt.json")
        receipt = passing(receipt_path)
        node_sha = sha(node_path)
        need(receipt.get("patient_key") == patient_key and receipt.get("cohort") == "home"
             and receipt.get("labels_sha256") == primary_sha
             and receipt.get("nodes_sha256") == node_sha,
             "Patient node cache identity/label/hash mismatch: " + patient_key)
        part = libraries.loc[libraries.patient_key.eq(patient_key)]
        need(len(part) > 0 and part.cohort.eq("home").all(), "Patient library membership differs")
        expected_sources = {}
        for library in part.itertuples(index=False):
            path = str(base.parent / "data/patient_aa" /
                       (library.cohort + "__" + library.sample_id + ".parquet"))
            need(path in mapping["library_source_sha256"], "Patient library missing from mapping hash manifest")
            expected_sources[path] = mapping["library_source_sha256"][path]
        need(receipt.get("library_source_sha256") == expected_sources
             and all(valid_sha(v) for v in expected_sources.values()),
             "Patient node cache source library hashes differ: " + patient_key)
        nodes = pd.read_parquet(node_path, columns=["seq_idx", "cluster_id", "read_count"])
        seq_idx, cached_clusters, reads = (nodes[col].to_numpy() for col in
                                           ["seq_idx", "cluster_id", "read_count"])
        need(np.issubdtype(seq_idx.dtype, np.integer) and len(seq_idx) > 0
             and int(seq_idx.min()) >= 0 and int(seq_idx.max()) < n_vertices
             and len(np.unique(seq_idx)) == len(seq_idx),
             "Patient node cache seq_idx is missing, duplicated or out of range: " + patient_key)
        need(np.issubdtype(reads.dtype, np.integer) and np.all(reads > 0)
             and len(seq_idx) == receipt["n_unique_aa"] == int(totals.loc[patient_key, "total_unique_aa"])
             and int(reads.sum()) == receipt["total_reads"] == int(totals.loc[patient_key, "total_reads"]),
             "Patient node cache AA/read denominator conservation failed: " + patient_key)
        need(np.array_equal(cached_clusters, labels_by_partition[(1.,17)][seq_idx]),
             "Patient node cache primary seq_idx-to-label mapping differs: " + patient_key)
        for key, labels in labels_by_partition.items():
            cluster_ids = np.unique(labels[seq_idx])
            need(int(cluster_ids.min()) >= 0 and int(cluster_ids.max()) < len(carriers_by_partition[key]),
                 "Projected cluster ID outside complete partition")
            # Multiple AA or merged libraries in one patient contribute one carrier.
            carriers_by_partition[key][cluster_ids] += 1
        node_provenance[patient_key] = {"nodes_sha256": node_sha,
            "receipt_sha256": sha(receipt_path), "n_unique_aa": len(seq_idx),
            "library_source_sha256": expected_sources}
        source_rows += len(seq_idx)
    primary_carriers = carriers_by_partition[(1.,17)]
    need(np.array_equal(primary_carriers, universe.home_known_carriers.to_numpy()),
         "Reprojected primary patient carrier vector differs from frozen test universe")
    primary_summary = carrier_eligibility_summary(primary_carriers, n_known)
    need(primary_summary["n_primary_carrier_eligible"] == statistics["n_primary_carrier_clusters"]
         == lock["n_primary_carrier_clusters"]
         and primary_summary["n_abundance_sensitivity_eligible"] ==
         statistics["n_abundance_sensitivity_clusters"] == lock["n_abundance_sensitivity_clusters"]
         and np.array_equal(universe.primary_carrier_eligible,
                            (primary_carriers >= 10) & (n_known-primary_carriers >= 5))
         and np.array_equal(universe.abundance_sensitivity_eligible, primary_carriers >= 10),
         "Reprojected primary eligibility differs from frozen primary tests")
    for key, row in rows_by_partition.items():
        row["patient_testability"] = carrier_eligibility_summary(carriers_by_partition[key], n_known)
    return {"status": "PASS", "phenotype_values_accessed": False,
            "n_Home_known_patient_ids": n_known, "home_known_ids_sha256": ids_sha,
            "previous_fold_file_sha256": sha(folds_path),
            "mapping_receipt_sha256": sha(mapping_path), "hypothesis_lock_sha256": sha(lock_path),
            "statistical_design_sha256": statistics["statistical_design_sha256"],
            "clinical_identity_manifest_sha256": sha(clinical_path),
            "primary_carrier_vector_matches_frozen_universe": True,
            "known_patient_union_AA_rows_checked": source_rows,
            "patient_nodes": node_provenance,
            "nonprimary_associations_run": False}

def make_report(result):
    lines = ["# 全量 Leiden 实际群数与患者级关联验收", "",
             "完成时间：" + result["completed_utc"], "",
             "**验收 PASS。全部 28,380,380 条 unique CDR3β AA 均为图顶点并有正式标签。**", "",
             "以下为实际全图结果，不能由子集试测群数外推。主统计分区固定为 resolution=1、seed=17；0.5/2仅作预设尺度敏感性，未据 N 结果选分辨率。", "",
             "| 分辨率 | 种子 | cluster 数 | 主 carrier 可检验 | 丰度可检验 | 携带近/全饱和 | 全饱和 | 稀少/无携带 | singleton 数 | 最大群 AA | 最大群占比 |",
             "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for row in result["partitions"]:
        t = row["patient_testability"]
        lines.append("|"+ "|".join([str(row["resolution"]), str(row["seed"]), f'{row["n_clusters"]:,}',
                                   f'{t["n_primary_carrier_eligible"]:,}', f'{t["n_abundance_sensitivity_eligible"]:,}',
                                   f'{t["n_carrier_saturated_or_near_saturated"]:,}',
                                   f'{t["n_fully_saturated_carrier_clusters"]:,}',
                                   f'{t["n_rare_carrier_clusters"]:,}',
                                   f'{row["singleton_clusters"]:,}', f'{row["largest_cluster"]:,}',
                                   f'{100*row["largest_cluster_fraction"]:.3f}%']) + "|")
    stat = result["patient_statistics"]
    lines += ["", "各分区的可检验性仅用同一冻结 Home 126 人 ID 清单及总携带情况计算，不读取 N0/N+ 数值：carrier 资格为至少 10 名携带者且至少 5 名非携带者；丰度资格为至少 10 名携带者。近/全饱和指非携带者少于 5 人，稀少/无携带指携带者少于 10 人。只有主分区执行预设关联检验；其他尺度及重复种子的这些数目是描述性可检验群数，不能当作新增 N 检验结果。",
              "", "主分区两算法随机种子（17/29）全图 ARI："
              + f'{result["primary_seed_stability"]["adjusted_rand_index"]:.6f}' + "。", "",
              "## 患者回联与检验对象", "",
              f'340 个文库回联 339 名患者：Home 137、External 202；N 已知分别为 126（56 N+ / 70 N0）和 163（54 N+ / 109 N0）。全部患者 AA 与读数覆盖为 100%，同患者文库先合并 AA 再相加读数。',
              "", f'主 carrier 合格群：{stat["n_primary_carrier_clusters"]:,}；丰度敏感性合格群：{stat["n_abundance_sensitivity_clusters"]:,}。',
              f'主 carrier 因接近/完全饱和排除：{stat["n_primary_ineligible_saturated_or_near_saturated"]:,}，其中全部 126 名 Home N 已知患者均携带的群为 {stat["n_fully_saturated_Home_known_carrier_clusters"]:,}。少于 10 名 Home N 已知携带者的群：{stat["n_rare_Home_known_carrier_clusters"]:,}。',
              "", "## Home 与 External 分别估计的探索性效应", "",
              "Home 调整年龄、性别、测序深度、组库大小、Home 批次及新辅助状态。External 缺少年龄、性别及治疗信息，仅做技术项调整。",
              "",
              "| 队列 | 尺度 | 受检群 | 稳定拟合 | P<0.05 | 其中正向/负向 | 合并全部检验 q<0.05 | 单队列单尺度 q<0.05 |",
              "|---|---|---:|---:|---:|---:|---:|---:|"]
    for row in stat["association_summary"]:
        lines.append("|"+ "|".join([row["cohort"], row["scale"], str(row["tested_clusters"]),
                                   str(row["fit_ok"]), str(row["nominal_P_lt_0_05"]),
                                   str(row["nominal_positive"])+"/"+str(row["nominal_negative"]),
                                   str(row["global_q_lt_0_05"]), str(row["scale_cohort_q_lt_0_05"])]) + "|")
    lines += ["", "合并 BH 包含所有预设 Home/External 主携带检验、两种丰度敏感性及 Home 未接受新辅助治疗的携带检验；失败拟合保留为 P=1。单队列单尺度 q 另列，不能替代预设合并 q。",
              "", "完整结果：[全部预设检验](../stats/associations_all_predefined_tests.tsv)；[所有 cluster 含不合格群](../stats/all_cluster_effects_summary.tsv)；[正式计数与来源哈希](full_cluster_counts_by_resolution.json)。", "",
              "## 解释范围", "",
              "本图包含 Home 与 External 的全部 AA，因此 External 是同一联合图谱内的队列复核。关联或名义 P<0.05 不能称为已经验证的 N+ 预测贡献。几何 Leiden 群不等于抗原特异性群；本报告没有据这些结果确认抗原结合。",
              "", "计数验收使用每个分辨率的完整标签、正式回执及来源哈希；大型向量/边数据的哈希沿用已审计构建及消费回执，不重复读取原始组库。"]
    return "\n".join(lines) + "\n"

def finalize(state):
    began = time.monotonic()
    numerical_modules()
    audited, source_info = audited_sources(BASE)
    cfg = load_json(BASE / "config/graph_design.json")
    graph, graph_info = graph_gate(BASE, cfg, source_info["graph_source_sha256"])
    geometry = geometry_gate(BASE, cfg)
    binding = {**source_info, "design_sha256": graph_info["design_sha256"],
               "pca_scores_sha256": cfg["pca_scores_sha256"],
               "atlas_union_sha256": cfg["atlas_union_sha256"],
               "graph_edges_sha256": graph["edges_sha256"],
               "leiden_parameters": {"n_iterations": cfg["full_leiden_iterations"],
                                     "beta": cfg["leiden_beta"],
                                     "objective": "igraph_native_weighted_modularity"}}
    rows, primary_bins, stability = cluster_runs(BASE, N_VERTICES, binding)
    stats = statistical_gate(BASE, primary_bins, audited, source_info)
    testability = partition_testability(BASE, rows, primary_bins, stats)
    # Source files and design must still match after all checks.
    _, sources_again = audited_sources(BASE)
    need(all(sources_again[key] == source_info[key] for key in
             ("graph_source_sha256", "stats_source_sha256"))
         and sha(BASE / "config/statistical_design.json") == stats["statistical_design_sha256"]
         and sha(BASE / "config/graph_design.json") == graph_info["design_sha256"],
         "Sources/design changed during postprocessing")
    result = {"status": "PASS", "scope": "actual full-atlas Leiden counts and exploratory patient associations",
              "completed_utc": now(), "n_vertices": N_VERTICES, "partitions": rows,
              "primary_seed_stability": stability, "patient_statistics": stats,
              "partition_patient_testability": testability,
              "geometry_diagnostic": geometry, "graph_acceptance": graph_info,
              "source_provenance": {**sources_again, "postprocessor_source_sha256": sha(Path(__file__)),
                                    "runner_state_sha256": sha(BASE / "audit/runner_state.json"),
                                    "bootstrap_inputs_sha256": sha(BASE / "audit/bootstrap_inputs.json"),
                                    "atlas_union_sha256": cfg["atlas_union_sha256"],
                                    "atlas_index_sha256": cfg["atlas_index_sha256"],
                                    "pca_scores_sha256": cfg["pca_scores_sha256"],
                                    "graph_edges_sha256": graph["edges_sha256"],
                                    "graph_weights_sha256": graph["weights_sha256"],
                                    "graph_edgelist_sha256": graph["edgelist_sha256"]},
              "cpu_compute_threads": 4, "gpu_used": False,
              "elapsed_seconds": time.monotonic() - began,
              "not_estimated": ["independent external validation", "validated N+ predictive contribution", "antigen binding"]}
    atomic_json(BASE / "report/full_cluster_counts_by_resolution.json", result)
    target = BASE / "report/FINAL_COUNTS.md"
    tmp = target.with_name(target.name + ".part")
    tmp.write_text(make_report(result))
    tmp.replace(target)
    print(json.dumps({"status": "PASS", "n_vertices": N_VERTICES,
                      "counts": [{"resolution": r["resolution"], "seed": r["seed"],
                                  "n_clusters": r["n_clusters"]} for r in rows],
                      "report": str(target)}), flush=True)

def smoke():
    np, pd = numerical_modules()
    checks = []
    audit_dir = BASE / "audit"
    audit_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".finalize_counts_smoke_", dir=audit_dir) as td:
        base = Path(td) / "branch"
        (base / "clusters").mkdir(parents=True)
        binding = {"graph_source_sha256": "a"*64, "design_sha256": "b"*64,
                   "pca_scores_sha256": "c"*64, "atlas_union_sha256": "d"*64,
                   "graph_edges_sha256": "e"*64}
        arrays = {(.5, 17): np.array([0]*4 + [1]*4 + [2]*4, dtype=np.int32),
                  (1., 17): np.array([0]*3 + [1]*3 + [2]*3 + [3]*3, dtype=np.int32),
                  (2., 17): np.array([0,0,1,1,2,2,3,3,4,4,5,6], dtype=np.int32),
                  (1., 29): np.array([3]*3 + [2]*3 + [1]*3 + [0]*3, dtype=np.int32)}
        for (resolution, seed), labels in arrays.items():
            lp, rp = cluster_paths(base, resolution, seed)
            np.save(lp, labels)
            bins = np.bincount(labels)
            atomic_json(rp, {"status": "PASS", "source_code_sha256": binding["graph_source_sha256"],
                            "design_sha256": binding["design_sha256"], "source_sha256": binding["pca_scores_sha256"],
                            "atlas_union_sha256": binding["atlas_union_sha256"],
                            "graph_edges_sha256": binding["graph_edges_sha256"], "frozen": True,
                            "all_vertices_assigned": True, "joint_all_cohort_graph": True,
                            "external_is_independent_validation": False,
                            "resolution": resolution, "seed": seed,
                            "n_vertices": 12, "n_clusters": len(bins), "largest_cluster": int(bins.max()),
                            "largest_cluster_fraction": float(bins.max()/12), "singleton_clusters": int((bins==1).sum()),
                            "cluster_size_quantiles": {str(q):float(np.quantile(bins,q)) for q in (0,.25,.5,.75,.9,.99,1)},
                            "labels_path": str(lp), "labels_sha256": sha(lp),
                            "primary_statistical_partition": resolution == 1 and seed == 17})
        p, _ = cluster_paths(base, 1., 17)
        r, _ = cluster_paths(base, 1., 29)
        atomic_json(base / "clusters/primary_seed_stability.json",
                    {"status":"PASS","source_code_sha256":binding["graph_source_sha256"],"n_vertices":12,
                     "seeds":[17,29],"primary_labels_sha256":sha(p),"repeat_labels_sha256":sha(r),
                     "adjusted_rand_index":1.0})
        rows, bins, st = cluster_runs(base, 12, binding)
        need([v["n_clusters"] for v in rows] == [3,4,7,4] and rows[2]["singleton_clusters"] == 2,
             "Synthetic actual-bin counts failed")
        checks.append("actual_full_fixture_bins_and_singletons_all_three_resolutions_plus_repeat")
        lp, rp = cluster_paths(base, 2., 17)
        old = load_json(rp)
        damaged = dict(old, labels_sha256="f"*64)
        atomic_json(rp, damaged)
        try:
            cluster_runs(base, 12, binding)
            raise AssertionError("Corrupt label checksum accepted")
        except AcceptanceError:
            checks.append("corrupt_label_hash_rejected")
        atomic_json(rp, old)
        rp.rename(rp.with_suffix(".missing"))
        try:
            cluster_runs(base, 12, binding)
            raise AssertionError("Missing scale receipt accepted")
        except AcceptanceError:
            checks.append("missing_resolution_receipt_rejected")
        rp.with_suffix(".missing").rename(rp)
        damaged = dict(old, graph_edges_sha256="f"*64)
        atomic_json(rp, damaged)
        try:
            cluster_runs(base, 12, binding)
            raise AssertionError("Wrong graph source accepted")
        except AcceptanceError:
            checks.append("sensitivity_graph_source_binding_mismatch_rejected")
        atomic_json(rp, old)
        st_path = base / "clusters/primary_seed_stability.json"
        st_bad = load_json(st_path);st_bad["repeat_labels_sha256"] = "f"*64
        atomic_json(st_path, st_bad)
        try:
            cluster_runs(base, 12, binding)
            raise AssertionError("Wrong stability label binding accepted")
        except AcceptanceError:
            checks.append("repeat_seed_stability_label_hash_mismatch_rejected")

        # Twenty synthetic patients share multiple AA in the same cluster;
        # finer partitions must change testable group counts, not patient weights.
        stats_dir = base / "stats"
        (stats_dir / "patient_nodes").mkdir(parents=True)
        (base.parent / "folds").mkdir()
        (base.parent / "config").mkdir()
        patient_keys = ["home_fixture_" + str(i) for i in range(20)]
        pd.DataFrame({"patient_key": patient_keys, "outcome_never_loaded": ["DO_NOT_READ"]*20}).to_csv(
            base.parent / "folds/folds.tsv", sep="\t", index=False)
        libraries = pd.DataFrame({"cohort": ["home"]*20,
            "sample_id": ["fixture_" + str(i) for i in range(20)], "patient_key": patient_keys})
        libraries.to_csv(base.parent / "config/cohort_samples.tsv", sep="\t", index=False)
        pd.DataFrame({"cohort": ["home"]*20, "patient_key": patient_keys}).to_parquet(
            stats_dir / "patient_identity.parquet", index=False)
        library_hashes, total_rows = {}, []
        for i, patient_key in enumerate(patient_keys):
            seq_idx = np.array([0,1] + ([3,4] if i < 12 else []) + ([6] if i < 5 else []),
                               dtype=np.int64)
            node = pd.DataFrame({"seq_idx": seq_idx, "cluster_id": arrays[(1.,17)][seq_idx],
                                 "read_count": np.arange(1, len(seq_idx)+1, dtype=np.int64)})
            node_path = stats_dir / "patient_nodes" / (patient_key + ".parquet")
            node.to_parquet(node_path, index=False)
            library_path = str(base.parent / "data/patient_aa" / ("home__fixture_" + str(i) + ".parquet"))
            library_hashes[library_path] = hashlib.sha256(library_path.encode()).hexdigest()
            atomic_json(node_path.with_suffix(".receipt.json"),
                {"status": "PASS", "patient_key": patient_key, "cohort": "home",
                 "n_unique_aa": len(node), "total_reads": int(node.read_count.sum()),
                 "library_source_sha256": {library_path: library_hashes[library_path]},
                 "labels_sha256": sha(p), "nodes_sha256": sha(node_path)})
            total_rows.append({"patient_key": patient_key, "total_unique_aa": len(node),
                               "total_reads": int(node.read_count.sum())})
        pd.DataFrame(total_rows).to_parquet(stats_dir / "patient_totals.parquet", index=False)
        carrier = np.array([20,12,5,0], dtype=np.int32)
        universe = pd.DataFrame({"cluster_id": np.arange(4), "global_unique_aa": bins,
            "home_known_carriers": carrier, "primary_carrier_eligible": (carrier >=10) & (20-carrier >=5),
            "abundance_sensitivity_eligible": carrier>=10})
        universe_path = stats_dir / "frozen_cluster_universe.parquet"
        universe.to_parquet(universe_path, index=False)
        mapping_path = stats_dir / "patient_mapping_receipt.json"
        atomic_json(mapping_path, {"status":"PASS","N_values_accessed":False,
            "source_manifest_sha256":sha(base.parent / "config/cohort_samples.tsv"),
            "library_source_sha256":library_hashes,
            "output_sha256":{fn:sha(stats_dir/fn) for fn in ["patient_identity.parquet","patient_totals.parquet"]}})
        lock_path = stats_dir / "hypothesis_universe_lock.json"
        lock = {"status":"frozen_before_new_leiden_N_values","N_values_accessed":False,
            "n_home_known_ids":20,"home_known_ids_sha256":hashlib.sha256("\n".join(sorted(patient_keys)).encode()).hexdigest(),
            "previous_fold_file_sha256":sha(base.parent / "folds/folds.tsv"),
            "mapping_receipt_sha256":sha(mapping_path),"statistical_design_sha256":"f"*64,
            "cluster_universe_sha256":sha(universe_path),
            "n_primary_carrier_clusters":1,"n_abundance_sensitivity_clusters":2}
        atomic_json(lock_path,lock)
        stats = {"mapping_receipt_sha256":sha(mapping_path), "hypothesis_lock_sha256":sha(lock_path),
            "statistical_design_sha256":"f"*64,"n_primary_carrier_clusters":1,
            "n_abundance_sensitivity_clusters":2}
        proof = partition_testability(base, rows, bins, stats, n_vertices=12, n_known=20)
        need(proof["phenotype_values_accessed"] is False
             and [v["patient_testability"]["n_primary_carrier_eligible"] for v in rows] == [1,1,2,1]
             and rows[1]["patient_testability"]["home_known_carrier_distribution"]["20"] == 1
             and rows[1]["patient_testability"]["n_absent_from_Home_known_clusters"] == 1
             and rows[1]["patient_testability"]["n_abundance_sensitivity_eligible"] == 2,
             "Partition testability/saturation/patient-unit counts failed")
        checks.append("all_partitions_label_blind_patient_testability_multiple_AA_one_carrier_and_saturated_rare_absent")

        report_fixture = {"completed_utc":"synthetic-rendering-test-not-full-data","partitions":rows,
            "primary_seed_stability":{"adjusted_rand_index":1},
            "patient_statistics":{"n_primary_carrier_clusters":1,
                "n_abundance_sensitivity_clusters":2,
                "n_primary_ineligible_saturated_or_near_saturated":1,
                "n_fully_saturated_Home_known_carrier_clusters":1,
                "n_rare_Home_known_carrier_clusters":2,"association_summary":[]}}
        rendered = make_report(report_fixture)
        first_table = [line for line in rendered.splitlines() if line.startswith("|")][:6]
        need(len(first_table) == 6 and all(len(line.strip("|").split("|")) == 11 for line in first_table)
             and "主 carrier 可检验" in first_table[0]
             and "singleton 数" in first_table[0],
             "Rendered table header/row columns differ")
        checks.append("report_partition_table_header_and_all_rows_have_11_matching_columns")

        bad_node = stats_dir / "patient_nodes" / (patient_keys[0] + ".parquet")
        bad_receipt = bad_node.with_suffix(".receipt.json")
        saved_node_receipt = load_json(bad_receipt)
        atomic_json(bad_receipt, dict(saved_node_receipt,nodes_sha256="f"*64))
        try:
            partition_testability(base, rows, bins, stats, n_vertices=12, n_known=20)
            raise AssertionError("Wrong node cache checksum accepted")
        except AcceptanceError:
            checks.append("patient_node_cache_hash_mismatch_rejected")
        atomic_json(bad_receipt,saved_node_receipt)
        changed_universe = universe.copy()
        changed_universe.loc[1,"home_known_carriers"] = 11
        changed_universe.to_parquet(universe_path,index=False)
        changed_lock = dict(lock,cluster_universe_sha256=sha(universe_path))
        atomic_json(lock_path,changed_lock)
        changed_stats = dict(stats,hypothesis_lock_sha256=sha(lock_path))
        try:
            partition_testability(base, rows, bins, changed_stats, n_vertices=12, n_known=20)
            raise AssertionError("Wrong primary per-cluster carrier vector accepted")
        except AcceptanceError:
            checks.append("primary_carrier_vector_disagrees_with_frozen_universe_rejected")
        universe.to_parquet(universe_path,index=False)
        atomic_json(lock_path,lock)
        pd.DataFrame({"patient_key":patient_keys[:-1]+["another_patient"]}).to_csv(
            base.parent / "folds/folds.tsv",sep="\t",index=False)
        try:
            partition_testability(base, rows, bins, stats, n_vertices=12, n_known=20)
            raise AssertionError("Changed known-ID membership accepted")
        except AcceptanceError:
            checks.append("changed_frozen_Home_known_ID_source_rejected")

        need(np.allclose(bh([.01,.04,1]), [.03,.06,1]), "Frozen-family BH smoke failed")
        checks.append(_private_literal('private_4c5af2f5f9ad6109'))
    result = {"status": "PASS", "scope": "isolated synthetic acceptance tests only; not actual fullbank counts",
              "not_fullbank_cluster_count": True, "checks": checks,
              "postprocessor_source_sha256": sha(Path(__file__)), "time_utc": now()}
    atomic_json(BASE / "audit/finalize_counts_smoke.json", result)
    print(json.dumps(result), flush=True)

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wait", action="store_true", help="Silently poll the serial runner every 20 seconds")
    parser.add_argument("--selftest", action="store_true", help="Run isolated synthetic acceptance tests; no full result report")
    args = parser.parse_args()
    try:
        if args.selftest:
            smoke()
        else:
            state = wait_for_complete(args.wait)
            finalize(state)
    except (AcceptanceError, OSError, KeyError, ValueError) as exc:
        print("Final full-Leiden acceptance failed: " + str(exc), file=sys.stderr)
        raise SystemExit(2)

if __name__ == "__main__":
    main()
