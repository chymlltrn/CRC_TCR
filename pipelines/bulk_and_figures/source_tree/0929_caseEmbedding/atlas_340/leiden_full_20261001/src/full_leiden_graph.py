#!/usr/bin/env python3
"""All-vertex, label-blind IVF-kNN / native igraph Leiden pipeline.

No phenotype table is read. Every global AA seq_idx is retained as a vertex.
Sparse approximate neighbours, not an all-pairs distance matrix, define the graph.
"""
from release_runtime import private_literal as _private_literal, release_path as _release_path
import argparse
import gc
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import random
import resource
import signal
import sys
import threading
import time
import traceback

# Set before importing BLAS / FAISS. Monitoring threads do not perform computation.
_threads = int(os.environ.get("ATLAS_CPU_THREADS", "48"))
for _key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
             "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "BLIS_NUM_THREADS"):
    os.environ[_key] = str(_threads)
os.environ["OMP_WAIT_POLICY"] = "PASSIVE"
import numpy as np
import psutil
import pyarrow as pa
import pyarrow.parquet as pq
import pyarrow.csv as pacsv
from scipy import sparse
from sklearn.metrics import adjusted_rand_score

BASE = Path(_release_path('path_99d8c1c6fde2a6b9'))
ROOT = BASE / "leiden_full_20261001"
STOP = threading.Event()
PROCESS = psutil.Process()
STAGE_STARTED = time.time()
PEAK_RSS = 0
RESOURCE_FILE = None
CURRENT_STAGE = "initialization"


def now():
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as handle:
        for data in iter(lambda: handle.read(16 << 20), b""):
            h.update(data)
    return h.hexdigest()


def atomic_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, sort_keys=True, allow_nan=False) + "\n")
    os.replace(tmp, path)


def atomic_npy(path, array):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "wb") as handle:
        np.save(handle, array, allow_pickle=False)
    os.replace(tmp, path)


def emit(event, **kwargs):
    obj = {"time": now(), "stage": CURRENT_STAGE, "event": event, **kwargs}
    print(json.dumps(obj, sort_keys=True, allow_nan=False), flush=True)
    atomic_json(ROOT / "logs" / "progress.json", obj)


def receipt(path, status="PASS", **kwargs):
    global PEAK_RSS
    PEAK_RSS = max(PEAK_RSS, PROCESS.memory_info().rss, resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024)
    obj = {"status": status, "time": now(), "stage": CURRENT_STAGE,
           "label_blind": True, "phenotype_tables_read": [],
           "elapsed_s": round(time.time() - STAGE_STARTED, 3),
           "peak_rss_gib": round(PEAK_RSS / (1 << 30), 4),
           "cpu_compute_threads": _threads, "gpu_used": False,
           "source_code_sha256": CODE_SHA256, **kwargs}
    atomic_json(path, obj)
    return obj


def signal_handler(signum, frame):
    STOP.set()
    emit("stop_requested", signal=signum)


def checkpoint_stop():
    if STOP.is_set():
        raise InterruptedError("Stop requested; completed checkpoints are preserved")


def resource_monitor(max_rss_gib):
    global PEAK_RSS
    limit = min(max_rss_gib * (1 << 30), 512000000000)
    while not STOP.wait(5):
        try:
            rss = PROCESS.memory_info().rss
            PEAK_RSS = max(PEAK_RSS, rss)
            if RESOURCE_FILE is not None:
                with open(RESOURCE_FILE, "a") as handle:
                    handle.write(json.dumps({"time": now(), "stage": CURRENT_STAGE,
                        "rss_gib": round(rss / (1 << 30), 4),
                        "peak_rss_gib": round(PEAK_RSS / (1 << 30), 4),
                        "cpu_percent": PROCESS.cpu_percent(),
                        "threads": PROCESS.num_threads()}) + "\n")
            if rss > limit:
                atomic_json(ROOT / "logs" / "resource_limit_failure.json", {
                    "status": "FAIL", "rss_gib": rss / (1 << 30),
                    "limit_gib": max_rss_gib, "time": now(),
                    "stage": CURRENT_STAGE,
                    "reason": "RSS ceiling exceeded; hard exit preserves prior checkpoints"})
                # Native C calls may defer Python signal handlers. Exit immediately
                # on a real RSS violation rather than exceeding the approved ceiling.
                os._exit(137)
        except psutil.Error:
            pass


def begin(stage):
    global CURRENT_STAGE, STAGE_STARTED
    CURRENT_STAGE, STAGE_STARTED = stage, time.time()
    emit("start")


def load_config():
    path = ROOT / "config" / "graph_design.json"
    cfg = json.loads(path.read_text())
    return cfg


def source_scores(cfg):
    a = np.load(cfg["pca_scores_path"], mmap_mode="r", allow_pickle=False)
    expected = (cfg["n_vertices"], cfg["dimension"])
    if a.shape != expected or a.dtype != np.float16:
        raise ValueError(f"Score shape/dtype mismatch: {a.shape} {a.dtype}, expected {expected} float16")
    return a


def source_validation(cfg):
    path = ROOT / "audit" / "source_validation.json"
    if path.exists():
        obj = json.loads(path.read_text())
        if (obj["status"] == "PASS" and obj["pca_scores_sha256"] == cfg["pca_scores_sha256"]
                and obj.get("atlas_union_sha256") == cfg["atlas_union_sha256"]
                and obj.get("bank_mapping_sha256") == cfg["atlas_index_sha256"]):
            a = source_scores(cfg)
            return obj
    a = source_scores(cfg)
    actual = sha256(cfg["pca_scores_path"])
    if actual != cfg["pca_scores_sha256"]:
        raise ValueError(f"PCA input hash mismatch: {actual}")
    projection = json.loads(Path(cfg["projection_receipt_path"]).read_text())
    if projection.get("n_aa", projection.get("rows", cfg["n_vertices"])) != cfg["n_vertices"]:
        raise ValueError("Projection receipt row count mismatch")
    for key in ["atlas_index", "atlas_union"]:
        if sha256(cfg[key + "_path"]) != cfg[key + "_sha256"]:
            raise ValueError(f"{key} source hash mismatch")
    return receipt(path, n_vertices=len(a), shape=list(a.shape), dtype=str(a.dtype),
        pca_scores_sha256=actual, pca_scope="joint_label_blind_patient_balanced",
        existing_projection_receipt=projection,
        bank_mapping_path=cfg["atlas_index_path"],
        bank_mapping_sha256=cfg["atlas_index_sha256"],
        atlas_union_path=cfg["atlas_union_path"], atlas_union_sha256=cfg["atlas_union_sha256"])


def sample_with_forced(n, target, forced, seed):
    forced = np.unique(np.asarray(forced, dtype=np.int64))
    if target > n or target < 1 or np.any((forced < 0) | (forced >= n)):
        raise ValueError("Invalid sample target or forced global IDs")
    rng = np.random.default_rng(seed)
    if len(forced) >= target:
        # For a small smoke/pilot, retain a deterministic random forced subset.
        return np.sort(rng.choice(forced, size=target, replace=False))
    random_candidates = rng.choice(n, size=min(n, max(target * 2, target + 10000)),
                                   replace=False)
    extra = random_candidates[~np.isin(random_candidates, forced, assume_unique=False)]
    if len(extra) < target - len(forced):
        raise RuntimeError("Insufficient unique random sample candidates")
    return np.sort(np.concatenate([forced, extra[:target - len(forced)]]))


def balanced_ids(cfg):
    table = pq.read_table(cfg["balanced_draws_path"],
                         columns=["patient_key", "cohort", "seq_idx"])
    ids = table["seq_idx"].combine_chunks().to_numpy(zero_copy_only=False)
    return np.unique(ids), {
        "input_rows": table.num_rows, "unique_aa_rows": len(np.unique(ids)),
        "draws_sha256": sha256(cfg["balanced_draws_path"]),
        "input_columns": ["patient_key", "cohort", "seq_idx"],
        "phenotype_columns_read": []}


def query_metadata(cfg, ids):
    """Fetch only needed row-group slices; no cohort/outcome metadata is loaded."""
    ids = np.asarray(ids, dtype=np.int64)
    pf = pq.ParquetFile(cfg["atlas_index_path"])
    out = {"seq_idx": [], "aa_length": [], "bank_source": []}
    offset = 0
    for rg in range(pf.num_row_groups):
        count = pf.metadata.row_group(rg).num_rows
        wanted = ids[(ids >= offset) & (ids < offset + count)]
        if len(wanted):
            t = pf.read_row_group(rg, columns=["seq_idx", "aa", "bank_source"])
            t = t.take(pa.array(wanted - offset, type=pa.int64()))
            actual = t["seq_idx"].to_numpy()
            if not np.array_equal(actual, wanted):
                raise ValueError("Global AA parquet row order does not match seq_idx")
            out["seq_idx"].extend(actual.tolist())
            out["aa_length"].extend([len(x) for x in t["aa"].to_pylist()])
            out["bank_source"].extend(t["bank_source"].to_pylist())
        offset += count
    if len(out["seq_idx"]) != len(ids):
        raise ValueError("Query metadata has missing AA keys")
    return pa.table(out)


def audit_query_ids(cfg):
    p = ROOT / "pilot" / "global_recall_queries.parquet"
    if p.exists():
        t = pq.read_table(p)
        if t.num_rows != cfg["recall_queries"]:
            raise ValueError("Existing recall query count differs from frozen specification")
        return t["seq_idx"].to_numpy(), t
    rng = np.random.default_rng(cfg["audit_seed"])
    candidates = np.sort(rng.choice(cfg["n_vertices"],
                                   size=min(50000, cfg["n_vertices"]), replace=False))
    t = query_metadata(cfg, candidates)
    lengths = t["aa_length"].to_numpy()
    sources = np.array(t["bank_source"].to_pylist())
    bounds = np.array(cfg["recall_query_length_boundaries"])
    strata = np.searchsorted(bounds, lengths, side="left")
    keys = [(int(s), str(src)) for s, src in zip(strata, sources)]
    groups = {}
    for pos, key in enumerate(keys):
        groups.setdefault(key, []).append(pos)
    total = cfg["recall_queries"]
    # Equal allocation ensures long/short and old/new AA are explicitly audited.
    selected = []
    per = total // len(groups)
    for key in sorted(groups):
        g = np.array(groups[key], dtype=np.int64)
        selected.extend(rng.choice(g, size=min(per, len(g)), replace=False).tolist())
    remaining = np.setdiff1d(np.arange(len(candidates)), np.array(selected),
                            assume_unique=False)
    if len(selected) < total:
        selected.extend(rng.choice(remaining, size=total - len(selected),
                                   replace=False).tolist())
    selected = np.asarray(selected, dtype=np.int64)
    t = t.take(pa.array(selected, type=pa.int64()))
    t = t.append_column("length_stratum", pa.array(strata[selected]))
    order = np.argsort(t["seq_idx"].to_numpy())
    t = t.take(pa.array(order, type=pa.int64()))
    tmp = p.with_name(p.name + ".tmp")
    pq.write_table(t, tmp)
    os.replace(tmp, p)
    return t["seq_idx"].to_numpy(), t


def import_faiss():
    import faiss
    faiss.omp_set_num_threads(_threads)
    return faiss


def create_ivf(d, nlist, cfg, coarse="hnsw"):
    f = import_faiss()
    if coarse == "hnsw":
        index = f.index_factory(d, f"IVF{nlist}_HNSW32,Flat", f.METRIC_L2)
        quantizer = f.downcast_index(index.quantizer)
        quantizer.hnsw.efConstruction = cfg["coarse_ef_construction"]
        quantizer.hnsw.efSearch = cfg["coarse_min_ef_search"]
        # factory's quantizer_trains_alone=2 otherwise trains via a Flat
        # assigner. Explicit HNSW assignment accelerates centroid training.
        assigner = f.IndexHNSWFlat(d, cfg["coarse_hnsw_m"])
        assigner.hnsw.efConstruction = cfg["coarse_ef_construction"]
        assigner.hnsw.efSearch = cfg["training_ef_search"]
        index.clustering_index = assigner
    elif coarse == "flat":
        index = f.IndexIVFFlat(f.IndexFlatL2(d), d, nlist, f.METRIC_L2)
        assigner = None
    else:
        raise ValueError("Unsupported coarse index")
    index.cp.niter = cfg["ivf_training_iterations"]
    index.cp.seed = cfg["training_seed"]
    index.cp.nredo = 1
    # assigner must be kept alive for the lifetime of the native training call.
    return index, assigner


def set_probe(index, nprobe, cfg):
    f = import_faiss()
    index.nprobe = min(int(nprobe), int(index.nlist))
    q = f.downcast_index(index.quantizer)
    if hasattr(q, "hnsw"):
        q.hnsw.efSearch = max(cfg["coarse_min_ef_search"], 2 * int(index.nprobe))


def float_rows(scores, ids_or_slice):
    x = np.ascontiguousarray(scores[ids_or_slice], dtype=np.float32)
    if not np.isfinite(x).all():
        raise ValueError("Missing/non-finite PCA coordinates")
    return x


def clean_knn(distances, ids, self_ids, k, n_vertices):
    if distances.shape != ids.shape or distances.ndim != 2:
        raise ValueError("kNN shape mismatch")
    valid = (ids >= 0) & (ids < n_vertices) & (ids != np.asarray(self_ids)[:, None])
    ranks = np.cumsum(valid, axis=1)
    take = valid & (ranks <= k)
    if not np.all(take.sum(axis=1) == k):
        bad = np.where(take.sum(axis=1) != k)[0][:10].tolist()
        raise ValueError(f"Insufficient valid non-self neighbours: query rows {bad}")
    out_i = ids[take].reshape(len(ids), k).astype(np.int32)
    out_d = distances[take].reshape(len(ids), k).astype(np.float32)
    if not np.isfinite(out_d).all():
        raise ValueError("Non-finite neighbour distances")
    if out_d.min() < -1e-2:
        raise ValueError("Materially negative squared L2 distances")
    np.maximum(out_d, 0, out=out_d)  # Float32 cancellation at zero distances.
    if np.any(np.diff(np.sort(out_i, axis=1), axis=1) == 0):
        raise ValueError("Duplicate neighbour IDs within query row")
    if np.any(out_i == np.asarray(self_ids)[:, None]):
        raise ValueError("Self neighbour survived removal")
    return out_d, out_i


def search_clean(index, x, self_ids, k, n_vertices, batch=2048):
    out_i = np.empty((len(x), k), dtype=np.int32)
    out_d = np.empty((len(x), k), dtype=np.float32)
    for start in range(0, len(x), batch):
        checkpoint_stop()
        end = min(len(x), start + batch)
        d, i = index.search(x[start:end], k + 1)
        out_d[start:end], out_i[start:end] = clean_knn(
            d, i, self_ids[start:end], k, n_vertices)
    return out_d, out_i


def recall_metrics(true_d, true_i, approx_d, approx_i, metadata=None):
    strict = np.array([len(set(a.tolist()).intersection(b.tolist())) / len(a)
                       for a, b in zip(true_i, approx_i)])
    tol = np.maximum(1e-6, 1e-5 * true_d[:, -1])
    tie_aware = np.mean(approx_d <= true_d[:, -1, None] + tol[:, None], axis=1)
    obj = {"strict_id_recall_mean": float(strict.mean()),
           "strict_id_recall_standard_error": float(strict.std(ddof=1) / math.sqrt(len(strict))),
           "strict_id_recall_min": float(strict.min()),
           "tie_aware_distance_recall_mean": float(tie_aware.mean()),
           "exact_zero_kth_distance_queries": int(np.sum(true_d[:, -1] == 0)),
           "queries": len(true_i)}
    if metadata is not None:
        strata = metadata["length_stratum"].to_numpy()
        sources = np.array(metadata["bank_source"].to_pylist())
        obj["strata"] = []
        for s, src in sorted(set(zip(strata.tolist(), sources.tolist()))):
            mask = (strata == s) & (sources == src)
            obj["strata"].append({"length_stratum": int(s), "bank_source": src,
                "queries": int(mask.sum()), "strict_recall": float(strict[mask].mean()),
                "tie_aware_recall": float(tie_aware[mask].mean())})
    return obj


def ensure_train_ids(cfg):
    path = ROOT / "graph" / "ivf_training_seq_idx.npy"
    meta_path = path.with_suffix(".receipt.json")
    if path.exists() and meta_path.exists():
        meta = json.loads(meta_path.read_text())
        if meta["status"] != "PASS" or sha256(path) != meta["sha256"]:
            raise ValueError("Invalid persisted IVF training sample")
        return np.load(path)
    forced, summary = balanced_ids(cfg)
    ids = sample_with_forced(cfg["n_vertices"], cfg["training_rows"], forced,
                             cfg["training_seed"])
    atomic_npy(path, ids)
    receipt(meta_path, n_rows=len(ids), sha256=sha256(path),
            balanced_draws=summary, sampling="all_unique_balanced_draws_plus_fixed_uniform_unique_AA",
            patient_balanced_2M=False, sampling_seed=cfg["training_seed"])
    return ids


def index_stage(cfg):
    begin("index")
    source_validation(cfg)
    f = import_faiss()
    final = ROOT / "graph" / "full.index"
    final_receipt = ROOT / "graph" / "index_receipt.json"
    if final.exists() and final_receipt.exists():
        obj = json.loads(final_receipt.read_text())
        if obj["status"] == "PASS" and obj["n_vertices"] == cfg["n_vertices"]:
            if sha256(final) != obj["index_sha256"]:
                raise ValueError("Full index hash mismatch")
            emit("already_complete", n_vertices=cfg["n_vertices"])
            return
    scores = source_scores(cfg)
    trained = ROOT / "graph" / "trained.index"
    working = ROOT / "graph" / "working.index"
    if working.exists():
        index = f.read_index(str(working))
        emit("resume_index", completed_rows=int(index.ntotal))
    else:
        if trained.exists():
            index = f.read_index(str(trained))
        else:
            train_ids = ensure_train_ids(cfg)
            train = float_rows(scores, train_ids)
            index, assigner = create_ivf(cfg["dimension"], cfg["nlist"], cfg)
            t = time.time()
            emit("train_begin", training_rows=len(train), nlist=cfg["nlist"],
                 coarse="HNSW32", explicit_hnsw_training_assigner=True)
            index.train(train)
            elapsed = time.time() - t
            tmp = trained.with_name("trained.index.tmp")
            f.write_index(index, str(tmp))
            os.replace(tmp, trained)
            receipt(ROOT / "graph" / "training_receipt.json",
                training_rows=len(train), training_seconds=elapsed,
                training_rows_per_s=len(train) / elapsed,
                index_sha256=sha256(trained), nlist=cfg["nlist"],
                training_assigner="IndexHNSWFlat", training_ef_search=cfg["training_ef_search"])
            del train, assigner
            gc.collect()
    if index.d != cfg["dimension"] or index.nlist != cfg["nlist"] or index.ntotal > len(scores):
        raise ValueError("Resumed IVF index dimensions/count mismatch")
    q = f.downcast_index(index.quantizer)
    if hasattr(q, "hnsw"):
        q.hnsw.efSearch = cfg["coarse_min_ef_search"]
    last_checkpoint = int(index.ntotal)
    t = time.time()
    start_total = int(index.ntotal)
    for start in range(int(index.ntotal), len(scores), cfg["index_add_chunk"]):
        end = min(len(scores), start + cfg["index_add_chunk"])
        x = float_rows(scores, slice(start, end))
        index.add(x)
        del x
        if end - last_checkpoint >= cfg["index_checkpoint_rows"] or end == len(scores) or STOP.is_set():
            tmp = working.with_name("working.index.tmp")
            f.write_index(index, str(tmp))
            os.replace(tmp, working)
            last_checkpoint = end
            atomic_json(ROOT / "graph" / "index_progress.json", {
                "status": "RUNNING", "completed_rows": end, "total_rows": len(scores),
                "source_sha256": cfg["pca_scores_sha256"], "time": now()})
        rate = (end - start_total) / max(time.time() - t, 1e-9)
        emit("index_add", completed_rows=end, total_rows=len(scores), rows_per_s=rate,
             eta_s=(len(scores) - end) / max(rate, 1e-9))
        checkpoint_stop()
    if index.ntotal != len(scores):
        raise ValueError("Full index row count mismatch")
    os.replace(working, final)
    receipt(final_receipt, n_vertices=int(index.ntotal), dimension=index.d,
        index_sha256=sha256(final), source_sha256=cfg["pca_scores_sha256"],
        faiss_version=f.__version__, index_factory=f"IVF{cfg['nlist']}_HNSW32,Flat",
        full_vectors_float32=True, pq_compression=False)


def full_recall(cfg, index):
    path = ROOT / "audit" / "fullbank_ann_recall.json"
    f = import_faiss()
    scores = source_scores(cfg)
    query_ids, metadata = audit_query_ids(cfg)
    query_hash = sha256(ROOT / "pilot" / "global_recall_queries.parquet")
    index_hash = json.loads((ROOT / "graph" / "index_receipt.json").read_text())["index_sha256"]
    queries = float_rows(scores, query_ids)
    if path.exists():
        obj = json.loads(path.read_text())
        matching = (obj["status"] == "PASS"
            and obj["source_sha256"] == cfg["pca_scores_sha256"]
            and obj.get("index_sha256") == index_hash
            and obj.get("query_sha256") == query_hash
            and obj.get("k") == cfg["k"]
            and obj.get("recall_target") == cfg["recall_target"]
            and obj.get("nprobe_candidates") == cfg["nprobe_candidates"])
        if matching:
            for row in obj["results"]:
                for kind in ["indices", "distances"]:
                    if sha256(row["approx_" + kind + "_path"]) != row["approx_" + kind + "_sha256"]:
                        raise ValueError("Persisted ANN recall arrays hash mismatch")
            for kind in ["indices", "distances"]:
                if sha256(obj["exact_" + kind + "_path"]) != obj["exact_" + kind + "_sha256"]:
                    raise ValueError("Persisted exact recall arrays hash mismatch")
            return int(obj["selected_nprobe"])
    true_i_path = ROOT / "pilot" / "exact_fullbank_neighbors.npy"
    true_d_path = ROOT / "pilot" / "exact_fullbank_distances.npy"
    truth_receipt = ROOT / "pilot" / "exact_fullbank_receipt.json"
    if true_i_path.exists() and true_d_path.exists() and truth_receipt.exists():
        tr = json.loads(truth_receipt.read_text())
        if (tr["status"] != "PASS" or tr["groundtruth_n_vertices"] != cfg["n_vertices"]
                or tr["source_sha256"] != cfg["pca_scores_sha256"]
                or tr["query_sha256"] != query_hash
                or tr["exact_indices_sha256"] != sha256(true_i_path)
                or tr["exact_distances_sha256"] != sha256(true_d_path)):
            raise ValueError("Persisted exact truth source/query/file mismatch")
        true_i, true_d = np.load(true_i_path), np.load(true_d_path)
    else:
        emit("exact_groundtruth_build", queries=len(queries), database_rows=len(scores))
        t = time.time()
        exact = f.IndexFlatL2(cfg["dimension"])
        for start in range(0, len(scores), cfg["index_add_chunk"]):
            checkpoint_stop()
            exact.add(float_rows(scores, slice(start, min(len(scores), start + cfg["index_add_chunk"]))))
        build_s = time.time() - t
        t = time.time()
        true_d, true_i = search_clean(exact, queries, query_ids, cfg["k"], len(scores),
                                      batch=cfg["exact_query_batch"])
        search_s = time.time() - t
        atomic_npy(true_i_path, true_i)
        atomic_npy(true_d_path, true_d)
        receipt(truth_receipt, source_sha256=cfg["pca_scores_sha256"],
            query_sha256=sha256(ROOT / "pilot" / "global_recall_queries.parquet"),
            groundtruth_database="entire_28380380_PCA64_bank",
            groundtruth_n_vertices=len(scores), queries=len(queries),
            exact_build_s=build_s, exact_search_s=search_s,
            exact_indices_sha256=sha256(true_i_path), exact_distances_sha256=sha256(true_d_path))
        del exact
        gc.collect()
    results = []
    for nprobe in cfg["nprobe_candidates"]:
        set_probe(index, nprobe, cfg)
        t = time.time()
        d, i = search_clean(index, queries, query_ids, cfg["k"], len(scores))
        elapsed = time.time() - t
        metrics = recall_metrics(true_d, true_i, d, i, metadata)
        ai = ROOT / "pilot" / f"fullbank_approx_nprobe_{int(index.nprobe)}_indices.npy"
        ad = ROOT / "pilot" / f"fullbank_approx_nprobe_{int(index.nprobe)}_distances.npy"
        atomic_npy(ai, i)
        atomic_npy(ad, d)
        results.append({"approx_indices_path": str(ai), "approx_distances_path": str(ad),
                        "approx_indices_sha256": sha256(ai), "approx_distances_sha256": sha256(ad),"nprobe": int(index.nprobe),
                        "coarse_ef_search": max(cfg["coarse_min_ef_search"], 2 * int(index.nprobe)),
                        "query_seconds": elapsed, "queries_per_s": len(queries) / elapsed,
                        "estimated_all_query_seconds": len(scores) * elapsed / len(queries),
                        **metrics})
        emit("fullbank_recall", **results[-1])
        checkpoint_stop()
    eligible = [r for r in results if r["strict_id_recall_mean"] >= cfg["recall_target"]]
    if not eligible:
        receipt(path, status="FAIL", source_sha256=cfg["pca_scores_sha256"],
                recall_target=cfg["recall_target"], results=results,
                reason="No tested nprobe met strict-ID fullbank recall gate; downstream graph blocked")
        raise RuntimeError("Full-bank strict-ID top-k recall is below 0.95")
    selected = min(eligible, key=lambda r: r["nprobe"])
    receipt(path, source_sha256=cfg["pca_scores_sha256"],
            index_sha256=index_hash, query_sha256=query_hash, k=cfg["k"],
            nprobe_candidates=cfg["nprobe_candidates"],
            exact_indices_path=str(true_i_path), exact_distances_path=str(true_d_path),
            exact_indices_sha256=sha256(true_i_path), exact_distances_sha256=sha256(true_d_path),
            exact_database_rows=len(scores), recall_target=cfg["recall_target"],
            selected_nprobe=selected["nprobe"], results=results,
            tie_aware_diagnostic_only=True)
    return selected["nprobe"]


def freeze_design(cfg, nprobe):
    path = ROOT / "config" / "graph_design.json"
    if cfg["frozen"]:
        if cfg["selected_nprobe"] != nprobe:
            raise ValueError("Attempt to alter a frozen nprobe")
        return cfg
    cfg["selected_nprobe"] = int(nprobe)
    cfg["frozen"] = True
    cfg["frozen_at"] = now()
    cfg["freeze_basis"] = "unlabelled_cost_pilot_and_entire_bank_strict_ID_ANN_recall"
    cfg["freeze_before_any_phenotype_read"] = True
    atomic_json(path, cfg)
    receipt(ROOT / "audit" / "design_freeze.json", frozen=True,
        design_sha256=sha256(path), selected_nprobe=nprobe,
        phenotype_tables_read=[], resolution_selection_by_N=False)
    emit("design_frozen", nprobe=nprobe, primary_resolution=cfg["primary_resolution"],
         full_leiden_iterations=cfg["full_leiden_iterations"])
    return cfg


def knn_stage(cfg):
    begin("knn")
    source_validation(cfg)
    index_receipt = json.loads((ROOT / "graph" / "index_receipt.json").read_text())
    if (index_receipt["status"] != "PASS"
            or index_receipt["n_vertices"] != cfg["n_vertices"]
            or index_receipt["source_sha256"] != cfg["pca_scores_sha256"]):
        raise ValueError("Full index not complete or source mismatch")
    if sha256(ROOT / "graph" / "full.index") != index_receipt["index_sha256"]:
        raise ValueError("Full index hash mismatch before ANN audit")
    f = import_faiss()
    index = f.read_index(str(ROOT / "graph" / "full.index"))
    if index.ntotal != cfg["n_vertices"]:
        raise ValueError("kNN database incomplete")
    nprobe = full_recall(cfg, index)
    cfg = freeze_design(cfg, nprobe)
    set_probe(index, nprobe, cfg)
    scores = source_scores(cfg)
    folder = ROOT / "graph" / "knn"
    folder.mkdir(parents=True, exist_ok=True)
    results = []
    t = time.time()
    new_rows = 0
    for start in range(0, len(scores), cfg["knn_shard_rows"]):
        end = min(len(scores), start + cfg["knn_shard_rows"])
        stem = f"part_{start:09d}_{end:09d}"
        ip, dp, rp = folder / (stem + "_indices.npy"), folder / (stem + "_distances.npy"), folder / (stem + ".json")
        if rp.exists():
            r = json.loads(rp.read_text())
            if r["status"] != "PASS" or r["source_sha256"] != cfg["pca_scores_sha256"] or r["nprobe"] != nprobe or r.get("index_sha256") != index_receipt["index_sha256"] or r.get("design_sha256") != sha256(ROOT / "config" / "graph_design.json"):
                raise ValueError(f"Invalid kNN shard receipt: {rp}")
            if sha256(ip) != r["indices_sha256"] or sha256(dp) != r["distances_sha256"]:
                raise ValueError(f"kNN shard hash mismatch: {rp}")
            results.append(r)
            continue
        x = float_rows(scores, slice(start, end))
        ids = np.arange(start, end, dtype=np.int64)
        ts = time.time()
        d, i = search_clean(index, x, ids, cfg["k"], len(scores),
                            batch=cfg["knn_query_batch"])
        atomic_npy(ip, i)
        atomic_npy(dp, d)
        elapsed = time.time() - ts
        r = receipt(rp, start=start, end=end, n_rows=end - start, k=cfg["k"],
            nprobe=nprobe, source_sha256=cfg["pca_scores_sha256"],
            index_sha256=index_receipt["index_sha256"],
            design_sha256=sha256(ROOT / "config" / "graph_design.json"),
            indices_path=str(ip), distances_path=str(dp),
            indices_sha256=sha256(ip), distances_sha256=sha256(dp),
            query_seconds=elapsed, rows_per_s=(end - start) / elapsed,
            self_neighbors=0, duplicate_neighbor_ids=0, invalid_neighbor_ids=0,
            nonfinite_distances=0, zero_distances=int(np.sum(d == 0)))
        results.append(r)
        new_rows += end - start
        rate = new_rows / max(time.time() - t, 1e-9)
        emit("knn_shard_complete", start=start, end=end, total_rows=len(scores),
            completed_rows=sum(rr["n_rows"] for rr in results), rows_per_s=rate,
            eta_s=(len(scores) - end) / max(rate, 1e-9))
        del x, d, i
        checkpoint_stop()
    if sum(r["n_rows"] for r in results) != len(scores):
        raise ValueError("kNN shard count does not conserve vertices")
    receipt(ROOT / "graph" / "knn_receipt.json", n_vertices=len(scores), k=cfg["k"],
        shards=len(results), completed_rows=len(scores), failures=0, nprobe=nprobe,
        index_sha256=index_receipt["index_sha256"],
        design_sha256=sha256(ROOT / "config" / "graph_design.json"),
        shards_receipts=[str(folder / f"part_{r['start']:09d}_{r['end']:09d}.json") for r in results])


def symmetric_graph(neighbors, distances, cfg, dest):
    """Build weighted undirected union with explicit N x N shape.

    d2 is squared Euclidean distance. sigma_i=sqrt(d2_i,k), so the
    denominator sigma_i*sigma_j is sqrt(d2_i,k*d2_j,k).
    """
    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    n, k = neighbors.shape
    if distances.shape != neighbors.shape:
        raise ValueError("Neighbour and distance arrays have incompatible shapes")
    sigma2 = np.asarray(distances[:, -1], dtype=np.float32)
    positive = sigma2[sigma2 > 0]
    floor = max(float(np.median(positive)) * cfg["local_scale_epsilon_multiplier"],
                np.finfo(np.float32).tiny) if len(positive) else 1e-12
    sigma2 = np.maximum(sigma2, floor)
    weights = np.lib.format.open_memmap(dest / "direct_weights.npy", mode="w+",
                                       dtype=np.float32, shape=(n, k))
    inds = np.lib.format.open_memmap(dest / "direct_indices.npy", mode="w+",
                                    dtype=np.int32, shape=(n, k))
    for start in range(0, n, cfg["knn_shard_rows"]):
        checkpoint_stop()
        end = min(n, start + cfg["knn_shard_rows"])
        nn = np.asarray(neighbors[start:end])
        dd = np.asarray(distances[start:end])
        if np.any((nn < 0) | (nn >= n)) or not np.isfinite(dd).all():
            raise ValueError("Illegal graph neighbour/distance values")
        denominator = np.sqrt(sigma2[start:end, None] * sigma2[nn])
        ratio = dd / denominator
        w = np.exp(-np.minimum(ratio, cfg["local_weight_ratio_clip"])).astype(np.float32)
        if not np.isfinite(w).all() or np.any(w <= 0):
            raise ValueError("Non-finite/non-positive graph weight")
        inds[start:end], weights[start:end] = nn, w
    inds.flush()
    weights.flush()
    pointer_dtype = np.int64 if n * k > np.iinfo(np.int32).max else np.int32
    indptr = np.arange(0, n * k + 1, k, dtype=pointer_dtype)
    directed = sparse.csr_matrix((weights.reshape(-1), inds.reshape(-1), indptr),
                                  shape=(n, n), copy=False)
    directed.sort_indices()
    directed.sum_duplicates()
    symmetric = directed.maximum(directed.transpose())
    del directed
    gc.collect()
    upper = sparse.triu(symmetric, k=1, format="csr")
    n_isolates = int(np.sum(np.diff(symmetric.indptr) == 0))
    del symmetric
    gc.collect()
    n_edges = int(upper.nnz)
    edges_path, weights_path = dest / "edges.npy", dest / "weights.npy"
    edges = np.lib.format.open_memmap(edges_path, mode="w+", dtype=np.int64,
                                     shape=(n_edges, 2))
    ew = np.lib.format.open_memmap(weights_path, mode="w+", dtype=np.float32,
                                  shape=(n_edges,))
    for start in range(0, n, cfg["knn_shard_rows"]):
        checkpoint_stop()
        end = min(n, start + cfg["knn_shard_rows"])
        lo, hi = int(upper.indptr[start]), int(upper.indptr[end])
        edges[lo:hi, 0] = np.repeat(np.arange(start, end, dtype=np.int64),
                                    np.diff(upper.indptr[start:end + 1]))
        edges[lo:hi, 1] = upper.indices[lo:hi]
        ew[lo:hi] = upper.data[lo:hi]
    edges.flush()
    ew.flush()
    if np.any(edges[:, 0] >= edges[:, 1]) or not np.isfinite(ew).all():
        raise ValueError("Undirected edge invariants violated")
    del upper, weights, inds, edges, ew
    gc.collect()
    edgelist_path = dest / "edges.edgelist"
    tmp_text = edgelist_path.with_name(edgelist_path.name + ".tmp")
    schema = pa.schema([("from", pa.int64()), ("to", pa.int64())])
    options = pacsv.WriteOptions(include_header=False, delimiter=" ", quoting_style="none")
    pairs = np.load(edges_path, mmap_mode="r")
    with pa.OSFile(str(tmp_text), "wb") as sink:
        with pacsv.CSVWriter(sink, schema=schema, write_options=options) as writer:
            for start in range(0, n_edges, 1000000):
                checkpoint_stop()
                sub = pairs[start:min(n_edges, start + 1000000)]
                writer.write_table(pa.table({"from": sub[:, 0], "to": sub[:, 1]}, schema=schema))
    os.replace(tmp_text, edgelist_path)
    del pairs
    return {"n_vertices": n, "n_directed_neighbors": n * k,
            "n_undirected_edges": n_edges, "n_isolated_vertices": n_isolates,
            "local_scale_floor_d2": floor, "edges_path": str(edges_path),
            "weights_path": str(weights_path), "edgelist_path": str(edgelist_path),
            "edgelist_sha256": sha256(edgelist_path), "edges_sha256": sha256(edges_path),
            "weights_sha256": sha256(weights_path)}


def graph_stage(cfg):
    begin("graph")
    if not cfg["frozen"]:
        raise ValueError("Graph design must be frozen before full graph construction")
    ready = ROOT / "graph" / "graph_receipt.json"
    if ready.exists():
        r = json.loads(ready.read_text())
        if r["status"] == "PASS" and r["design_sha256"] == sha256(ROOT / "config" / "graph_design.json"):
            for kind in ("edges", "weights", "edgelist"):
                if sha256(r[kind + "_path"]) != r[kind + "_sha256"]:
                    raise ValueError("Existing graph file hash mismatch")
            emit("already_complete", n_edges=r["n_undirected_edges"])
            return
    r = json.loads((ROOT / "graph" / "knn_receipt.json").read_text())
    if (r["status"] != "PASS" or r["completed_rows"] != cfg["n_vertices"]
            or r["design_sha256"] != sha256(ROOT / "config" / "graph_design.json")):
        raise ValueError("Full kNN not complete or design mismatch")
    n, k = cfg["n_vertices"], cfg["k"]
    ip, dp = ROOT / "graph" / "all_neighbors.npy", ROOT / "graph" / "all_distances.npy"
    ii = np.lib.format.open_memmap(ip, mode="w+", dtype=np.int32, shape=(n, k))
    dd = np.lib.format.open_memmap(dp, mode="w+", dtype=np.float32, shape=(n, k))
    cursor = 0
    for rp in r["shards_receipts"]:
        sr = json.loads(Path(rp).read_text())
        start, end = sr["start"], sr["end"]
        if (sr["status"] != "PASS" or start != cursor or end <= start
                or end > n or sr["n_rows"] != end - start
                or sr["source_sha256"] != cfg["pca_scores_sha256"]
                or sr["nprobe"] != cfg["selected_nprobe"] or sr["k"] != k
                or sr.get("design_sha256") != r["design_sha256"]
                or sr.get("index_sha256") != r["index_sha256"]):
            raise ValueError("kNN shard continuity/design/source invariant failed")
        if (sha256(sr["indices_path"]) != sr["indices_sha256"]
                or sha256(sr["distances_path"]) != sr["distances_sha256"]):
            raise ValueError("kNN shard hash mismatch during graph assembly")
        si = np.load(sr["indices_path"], mmap_mode="r")
        sd = np.load(sr["distances_path"], mmap_mode="r")
        if si.shape != (end - start, k) or sd.shape != si.shape:
            raise ValueError("kNN shard array shape mismatch")
        ii[start:end], dd[start:end] = si, sd
        cursor = end
    if cursor != n:
        raise ValueError("kNN shards do not cover all vertices exactly once")
    ii.flush()
    dd.flush()
    info = symmetric_graph(ii, dd, cfg, ROOT / "graph")
    del ii, dd
    gc.collect()
    receipt(ready, **info, design_sha256=sha256(ROOT / "config" / "graph_design.json"),
            symmetric_rule="union_max", all_vertices_retained=True, no_self_loops=True)


def run_leiden(edges, weights, n, resolution, seed, iterations, beta, edgelist_path):
    import igraph as ig
    ig.set_random_number_generator(random.Random(int(seed)))
    t = time.time()
    # Installed igraph is abi3: ndarray->memoryview can internally call
    # tolist(). Use the official native C edgelist reader instead of an
    # E-sized Python edge list. Arrow's C++ writer preserves row order.
    graph = ig.Graph.Read_Edgelist(str(edgelist_path), directed=False)
    if graph.vcount() < n:
        graph.add_vertices(n - graph.vcount())
    build_seconds = time.time() - t
    if graph.vcount() != n or graph.ecount() != len(weights):
        raise ValueError("igraph vertex/edge count mismatch")
    positions = np.unique(np.concatenate([np.arange(min(100, len(edges))),
                                           np.arange(max(0, len(edges) - 100), len(edges))]))
    for pos in positions:
        if tuple(graph.es[int(pos)].tuple) != tuple(edges[int(pos)]):
            raise ValueError("Native graph reader changed edge order; weights misaligned")
    t = time.time()
    partition = graph.community_leiden(objective_function="modularity",
        weights=weights, resolution=float(resolution), beta=float(beta),
        n_iterations=int(iterations))
    elapsed = time.time() - t
    labels = np.asarray(partition.membership, dtype=np.int32)
    if labels.shape != (n,) or np.any(labels < 0):
        raise ValueError("Leiden left unassigned vertices")
    # Relabel deterministically by smallest seq_idx in each group. This does not
    # change graph partition; cluster IDs remain audit-friendly across reruns.
    unique, first = np.unique(labels, return_index=True)
    ordered = unique[np.argsort(first)]
    mapper = np.empty(int(labels.max()) + 1, dtype=np.int32)
    mapper[ordered] = np.arange(len(ordered), dtype=np.int32)
    labels = mapper[labels]
    sizes = np.bincount(labels)
    info = {"n_vertices": n, "n_clusters": int(len(sizes)),
            "largest_cluster": int(sizes.max()),
            "largest_cluster_fraction": float(sizes.max() / n),
            "singleton_clusters": int(np.sum(sizes == 1)),
            "cluster_size_quantiles": {str(q): float(np.quantile(sizes, q))
                                      for q in [0, .25, .5, .75, .9, .99, 1]},
            "resolution": float(resolution), "seed": int(seed),
            "n_iterations": int(iterations), "beta": float(beta),
            "objective": "igraph_native_weighted_modularity",
            "RBConfiguration_equivalent_normalized_objective": True,
            "igraph_version": ig.__version__,
            "graph_constructor": "native_C_Read_Edgelist",
            "graph_build_seconds": build_seconds, "leiden_seconds": elapsed,
            "quality": float(partition.quality)}
    del partition, graph
    gc.collect()
    return labels, info


def primary_paths(resolution, seed, cfg):
    is_primary = math.isclose(resolution, cfg["primary_resolution"]) and seed == cfg["primary_seed"]
    if is_primary:
        return ROOT / "clusters" / "primary_labels.npy", ROOT / "clusters" / "primary_receipt.json"
    token = str(resolution).replace(".", "p")
    stem = f"resolution_{token}_seed_{seed}"
    return ROOT / "clusters" / (stem + "_labels.npy"), ROOT / "clusters" / (stem + "_receipt.json")


def leiden_stage(cfg, resolutions, repeat_primary):
    begin("leiden")
    if not cfg["frozen"]:
        raise ValueError("Leiden design not frozen")
    design_hash = sha256(ROOT / "config" / "graph_design.json")
    r = json.loads((ROOT / "graph" / "graph_receipt.json").read_text())
    if r["status"] != "PASS" or r["design_sha256"] != design_hash:
        raise ValueError("Graph receipt incomplete or design changed")
    for kind in ["edges", "weights", "edgelist"]:
        if sha256(r[kind + "_path"]) != r[kind + "_sha256"]:
            raise ValueError(f"Leiden input {kind} hash mismatch")
    edges = np.load(r["edges_path"], mmap_mode="r", allow_pickle=False)
    weights = np.load(r["weights_path"], mmap_mode="r", allow_pickle=False)
    if edges.shape != (r["n_undirected_edges"], 2) or len(weights) != len(edges):
        raise ValueError("Graph arrays have wrong shape")
    jobs = [(res, cfg["primary_seed"]) for res in resolutions]
    if repeat_primary:
        jobs.append((cfg["primary_resolution"], cfg["repeat_seed"]))
    outputs = []
    for resolution, seed in jobs:
        lp, rp = primary_paths(resolution, seed, cfg)
        if rp.exists():
            obj = json.loads(rp.read_text())
            if obj["status"] == "PASS" and obj["design_sha256"] == design_hash and sha256(lp) == obj["labels_sha256"]:
                outputs.append(obj)
                emit("already_complete", resolution=resolution, seed=seed, n_clusters=obj["n_clusters"])
                continue
            raise ValueError("Existing Leiden result does not match frozen design")
        emit("leiden_begin", resolution=resolution, seed=seed, n_vertices=cfg["n_vertices"],
             n_edges=len(edges), n_iterations=cfg["full_leiden_iterations"])
        labels, info = run_leiden(edges, weights, cfg["n_vertices"], resolution, seed,
                                   cfg["full_leiden_iterations"], cfg["leiden_beta"], r["edgelist_path"])
        atomic_npy(lp, labels)
        obj = receipt(rp, **info, labels_path=str(lp), labels_sha256=sha256(lp),
            design_sha256=design_hash, frozen=True, source_sha256=cfg["pca_scores_sha256"],
            atlas_union_path=cfg["atlas_union_path"], atlas_union_sha256=cfg["atlas_union_sha256"],
            graph_edges_sha256=r["edges_sha256"], primary_statistical_partition=
                (math.isclose(resolution, cfg["primary_resolution"]) and seed == cfg["primary_seed"]),
            joint_all_cohort_graph=True, external_is_independent_validation=False,
            all_vertices_assigned=True)
        outputs.append(obj)
        emit("leiden_complete", **info)
        del labels
        checkpoint_stop()
    main, _ = primary_paths(cfg["primary_resolution"], cfg["primary_seed"], cfg)
    repeat, _ = primary_paths(cfg["primary_resolution"], cfg["repeat_seed"], cfg)
    if main.exists() and repeat.exists():
        a, b = np.load(main, mmap_mode="r"), np.load(repeat, mmap_mode="r")
        ari = float(adjusted_rand_score(a, b))
        receipt(ROOT / "clusters" / "primary_seed_stability.json",
            adjusted_rand_index=ari, n_vertices=len(a),
            seeds=[cfg["primary_seed"], cfg["repeat_seed"]],
            primary_labels_sha256=sha256(main), repeat_labels_sha256=sha256(repeat))
    receipt(ROOT / "clusters" / "leiden_runs_receipt.json", runs=outputs,
            design_sha256=design_hash, frozen=True)


def pilot_stage(cfg, rows=None):
    begin("pilot")
    source_validation(cfg)
    rows = int(rows or cfg["pilot_rows"])
    if rows < 1000 or rows > cfg["n_vertices"]:
        raise ValueError("Pilot must contain between 1000 and all global AA")
    pdir = ROOT / "pilot" / f"n_{rows}"
    pdir.mkdir(parents=True, exist_ok=True)
    ready = pdir / "pilot_receipt.json"
    if ready.exists():
        obj = json.loads(ready.read_text())
        if obj["status"] == "PASS" and obj["source_sha256"] == cfg["pca_scores_sha256"]:
            emit("already_complete", pilot_rows=rows)
            return obj
    f = import_faiss()
    forced, balance = balanced_ids(cfg)
    ids = sample_with_forced(cfg["n_vertices"], rows, forced, cfg["pilot_seed"])
    atomic_npy(pdir / "global_seq_idx.npy", ids)
    scores = source_scores(cfg)
    x = float_rows(scores, ids)
    rng = np.random.default_rng(cfg["pilot_seed"])
    train_pos = np.sort(rng.choice(rows, size=min(rows, cfg["pilot_training_rows"]), replace=False))
    train = x[train_pos]
    nlist = min(cfg["pilot_nlist"], max(16, len(train) // 39))
    qids = np.sort(rng.choice(rows, size=min(cfg["recall_queries"], rows), replace=False))
    exact = f.IndexFlatL2(cfg["dimension"])
    exact.add(x)
    true_d, true_i = search_clean(exact, x[qids], qids, cfg["k"], rows,
                                  batch=cfg["exact_query_batch"])
    del exact
    gc.collect()
    # Same fixed training/query data for flat/HNSW coarse cost comparison.
    benchmarks = []
    chosen = None
    for coarse in ["flat", "hnsw"]:
        index, assigner = create_ivf(cfg["dimension"], nlist, cfg, coarse=coarse)
        t = time.time()
        index.train(train)
        train_s = time.time() - t
        t = time.time()
        index.add(x)
        add_s = time.time() - t
        records = []
        for probe in cfg["nprobe_candidates"]:
            set_probe(index, min(probe, nlist), cfg)
            t = time.time()
            d, i = search_clean(index, x[qids], qids, cfg["k"], rows)
            seconds = time.time() - t
            records.append({"nprobe": int(index.nprobe), "query_seconds": seconds,
                            "queries_per_s": len(qids) / seconds,
                            **recall_metrics(true_d, true_i, d, i)})
            emit("pilot_ann", coarse=coarse, nlist=nlist, **records[-1])
        passing = [r for r in records if r["strict_id_recall_mean"] >= cfg["recall_target"]]
        benchmarks.append({"coarse": coarse, "training_seconds": train_s,
                           "add_seconds": add_s, "nlist": nlist,
                           "training_rows": len(train), "searches": records})
        if coarse == "hnsw":
            if not passing:
                receipt(ready, status="FAIL", reason="Pilot ANN recall gate failed",
                        source_sha256=cfg["pca_scores_sha256"], benchmarks=benchmarks)
                raise RuntimeError("Pilot HNSW-coarse IVF recall failed")
            chosen = min(passing, key=lambda r: r["nprobe"])
            set_probe(index, chosen["nprobe"], cfg)
            atomic_json(pdir / "ann_benchmarks.json", benchmarks)
            t = time.time()
            all_d, all_i = search_clean(index, x, np.arange(rows), cfg["k"], rows,
                                         batch=cfg["knn_query_batch"])
            knn_seconds = time.time() - t
            atomic_npy(pdir / "neighbors.npy", all_i)
            atomic_npy(pdir / "distances.npy", all_d)
        del index, assigner
        gc.collect()
    t = time.time()
    graph_info = symmetric_graph(all_i, all_d, cfg, pdir / "graph")
    graph_seconds = time.time() - t
    del x, train, all_i, all_d
    gc.collect()
    edges, weights = np.load(graph_info["edges_path"], mmap_mode="r"), np.load(graph_info["weights_path"], mmap_mode="r")
    runs = []
    label_paths = {}
    for resolution in cfg["pilot_resolutions"]:
        for iterations in cfg["pilot_iterations"]:
            for seed in cfg["pilot_seeds"]:
                token = str(resolution).replace(".", "p")
                lp = pdir / f"labels_r{token}_iter{iterations}_seed{seed}.npy"
                rp = lp.with_suffix(".receipt.json")
                if lp.exists() and rp.exists():
                    run = json.loads(rp.read_text())
                    if (run["status"] != "PASS" or sha256(lp) != run["labels_sha256"]
                            or run.get("source_sha256") != cfg["pca_scores_sha256"]
                            or run.get("global_seq_idx_sha256") != sha256(pdir / "global_seq_idx.npy")
                            or run.get("graph_edges_sha256") != graph_info["edges_sha256"]
                            or run.get("graph_weights_sha256") != graph_info["weights_sha256"]):
                        raise ValueError("Pilot labels checkpoint failed verification")
                else:
                    labels, info = run_leiden(edges, weights, rows, resolution, seed, iterations, cfg["leiden_beta"], graph_info["edgelist_path"])
                    atomic_npy(lp, labels)
                    run = receipt(rp, **info, labels_sha256=sha256(lp), labels_path=str(lp),
                                  pilot_only=True, fullbank_cluster_count=False,
                                  source_sha256=cfg["pca_scores_sha256"],
                                  global_seq_idx_sha256=sha256(pdir / "global_seq_idx.npy"),
                                  graph_edges_sha256=graph_info["edges_sha256"],
                                  graph_weights_sha256=graph_info["weights_sha256"])
                    del labels
                label_paths[(resolution, iterations, seed)] = lp
                runs.append(run)
                emit("pilot_leiden", resolution=resolution, iterations=iterations, seed=seed, n_clusters=run["n_clusters"],
                     leiden_seconds=run["leiden_seconds"])
                checkpoint_stop()
    stability = []
    for resolution in cfg["pilot_resolutions"]:
        for iterations in cfg["pilot_iterations"]:
            a, b = [np.load(label_paths[(resolution, iterations, s)], mmap_mode="r") for s in cfg["pilot_seeds"]]
            stability.append({"resolution": resolution, "n_iterations": iterations,
                              "comparison": "seed17_vs_seed29",
                              "ARI": float(adjusted_rand_score(a, b))})
        for seed in cfg["pilot_seeds"]:
            a = np.load(label_paths[(resolution, 2, seed)], mmap_mode="r")
            b = np.load(label_paths[(resolution, 4, seed)], mmap_mode="r")
            stability.append({"resolution": resolution, "seed": seed,
                              "comparison": "iterations2_vs_iterations4",
                              "ARI": float(adjusted_rand_score(a, b))})
    primary_iter_stability = [s["ARI"] for s in stability
                             if s["comparison"] == "iterations2_vs_iterations4"
                             and s["resolution"] == cfg["primary_resolution"]]
    recommendation = 2 if min(primary_iter_stability) >= cfg["iteration_stability_target"] else 4
    if not cfg["frozen"]:
        cfg["full_leiden_iterations"] = recommendation
        cfg["iteration_choice_basis"] = "pilot_resolution1_both_seeds_iteration2_vs4_ARI_min_ge_0.98_choose2_else4"
        atomic_json(ROOT / "config" / "graph_design.json", cfg)
    result = receipt(ready, source_sha256=cfg["pca_scores_sha256"],
        pilot_rows=rows, pilot_only=True, fullbank_cluster_count=False,
        balanced_draws=balance, forced_balanced_unique_aa=min(len(forced), rows),
        actual_sampling="balanced_unique_draws_plus_uniform_unique_AA",
        benchmark_training_rows=len(train_pos), benchmark_nlist=nlist,
        ann_benchmarks=benchmarks, selected_pilot_nprobe=chosen["nprobe"],
        knn_seconds=knn_seconds, graph_seconds=graph_seconds, graph=graph_info,
        leiden_runs=runs, stability=stability, recommended_full_iterations=recommendation,
        fullbank_ann_recall_still_required=True)
    emit("pilot_complete", pilot_rows=rows, recommended_full_iterations=recommendation)
    return result


def smoke_stage(cfg):
    begin("smoke")
    f = import_faiss()
    rng = np.random.default_rng(44)
    centers = rng.normal(size=(5, cfg["dimension"])).astype(np.float32) * 5
    x = centers[np.arange(5000) % 5] + rng.normal(scale=.15, size=(5000, cfg["dimension"])).astype(np.float32)
    x[1] = x[0]  # Valid distinct AA vertices can share zero-distance embeddings.
    index = f.IndexFlatL2(cfg["dimension"])
    index.add(x)
    d, i = search_clean(index, x, np.arange(len(x)), cfg["k"], len(x), batch=512)
    assert i.shape == (5000, cfg["k"])
    assert not np.any(i == np.arange(5000)[:, None])
    assert np.all(np.isfinite(d))
    assert np.array_equal(x[0], x[1])
    # Analytically zero distances can have positive BLAS cancellation. Force
    # the known identical-vector pair to exactly zero to test kernel safety.
    d[0, np.where(i[0] == 1)[0][0]] = 0.0
    d[1, np.where(i[1] == 0)[0][0]] = 0.0
    ivf_cfg = dict(cfg, ivf_training_iterations=3, training_ef_search=64)
    ivf, training_assigner = create_ivf(cfg["dimension"], 64, ivf_cfg)
    ivf.train(np.ascontiguousarray(x[:2000]))
    del training_assigner
    gc.collect()
    ivf.add(x)
    set_probe(ivf, 16, cfg)
    ivf_d, ivf_i = search_clean(ivf, x[:100], np.arange(100), cfg["k"], len(x), batch=100)
    smoke_index = ROOT / "pilot" / "smoke" / "ivf_resume.index"
    smoke_index.parent.mkdir(parents=True, exist_ok=True)
    f.write_index(ivf, str(smoke_index))
    restored = f.read_index(str(smoke_index))
    set_probe(restored, 16, cfg)
    restored_d, restored_i = search_clean(restored, x[:100], np.arange(100), cfg["k"], len(x), batch=100)
    assert restored.ntotal == len(x)
    assert np.array_equal(ivf_i, restored_i) and np.array_equal(ivf_d, restored_d)
    assert hasattr(f.downcast_index(restored.quantizer), "hnsw")
    # Illegal IDs and duplicate neighbours must fail, not silently pass.
    failures = {}
    for case in ["illegal_ids", "duplicate_ids"]:
        ids = np.tile(np.arange(cfg["k"] + 1), (1, 1)).astype(np.int64)
        distances = np.zeros_like(ids, dtype=np.float32)
        self_ids = np.array([100])
        if case == "illegal_ids":
            ids[:] = -1
        else:
            ids[:] = 2
        try:
            clean_knn(distances, ids, self_ids, cfg["k"], len(x))
            failures[case] = False
        except ValueError:
            failures[case] = True
    assert all(failures.values())
    info = symmetric_graph(i, d, cfg, ROOT / "pilot" / "smoke" / "graph")
    edges = np.load(info["edges_path"], mmap_mode="r")
    weights = np.load(info["weights_path"], mmap_mode="r")
    labels, stats = run_leiden(edges, weights, len(x) + 1, 1, 17, 2, cfg["leiden_beta"], info["edgelist_path"])
    assert len(labels) == 5001
    assert np.sum(labels == labels[-1]) == 1  # explicit isolated vertex retained
    result = receipt(ROOT / "pilot" / "smoke" / "smoke_receipt.json",
        zero_distance_neighbour_preserved=bool(np.any(d == 0)),
        ivf_hnsw_training_assignment_tested=True, ivf_write_read_resume_identical=True,
        native_C_edgelist_reader_tested=True,
        illegal_ids_rejected=failures["illegal_ids"],
        duplicate_ids_rejected=failures["duplicate_ids"],
        isolated_vertex_retained=True, synthetic_n_vertices=5001,
        synthetic_n_clusters=stats["n_clusters"],
        no_real_N_labels_read=True, faiss_version=f.__version__)
    emit("smoke_complete", **result)


def main():
    global RESOURCE_FILE, _threads
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=["smoke", "pilot", "index", "knn", "graph", "leiden", "all"], required=True)
    parser.add_argument("--threads", type=int, default=_threads)
    parser.add_argument("--max-rss-gib", type=int, default=512)
    parser.add_argument("--pilot-rows", type=int)
    parser.add_argument("--resolutions", default="1")
    parser.add_argument("--repeat-primary", action="store_true")
    args = parser.parse_args()
    if args.threads < 1 or args.threads > 48 or args.max_rss_gib > 512:
        raise ValueError("Approved CPU/RSS ceiling is 48 computation threads / 512 GiB")
    _threads = args.threads
    from threadpoolctl import threadpool_limits
    _blas_limit = threadpool_limits(limits=_threads)
    for folder in ["config", "pilot", "logs", "graph", "clusters", "audit"]:
        (ROOT / folder).mkdir(parents=True, exist_ok=True)
    RESOURCE_FILE = ROOT / "logs" / f"resources_{args.stage}_{int(time.time())}.jsonl"
    # Address-space ceiling is more conservative than RSS; catches native OOM
    # allocations before they can consume the approved memory ceiling.
    approved_bytes = min(args.max_rss_gib * (1 << 30), 512000000000)
    resource.setrlimit(resource.RLIMIT_AS, (approved_bytes, approved_bytes))
    for sig in [signal.SIGTERM, signal.SIGINT]:
        signal.signal(sig, signal_handler)
    threading.Thread(target=resource_monitor, args=(args.max_rss_gib,), daemon=True).start()
    cfg = load_config()
    try:
        if args.stage == "smoke":
            smoke_stage(cfg)
        elif args.stage == "pilot":
            pilot_stage(cfg, args.pilot_rows)
        elif args.stage == "index":
            index_stage(cfg)
        elif args.stage == "knn":
            knn_stage(cfg)
        elif args.stage == "graph":
            graph_stage(cfg)
        elif args.stage == "leiden":
            leiden_stage(cfg, [float(x) for x in args.resolutions.split(",")], args.repeat_primary)
        elif args.stage == "all":
            pilot_stage(cfg, args.pilot_rows)
            cfg = load_config()
            index_stage(cfg)
            knn_stage(cfg)
            cfg = load_config()
            graph_stage(cfg)
            leiden_stage(cfg, [float(x) for x in args.resolutions.split(",")], args.repeat_primary)
    except BaseException as exc:
        receipt(ROOT / "logs" / f"failure_{args.stage}_{int(time.time())}.json",
            status="STOPPED" if isinstance(exc, InterruptedError) else "FAIL",
            error_type=type(exc).__name__, error=str(exc), traceback=traceback.format_exc(),
            completed_checkpoints_preserved=True)
        raise
    finally:
        STOP.set()


CODE_SHA256 = sha256(__file__)

if __name__ == "__main__":
    main()

