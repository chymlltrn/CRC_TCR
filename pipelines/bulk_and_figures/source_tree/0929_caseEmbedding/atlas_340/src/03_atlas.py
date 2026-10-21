#!/usr/bin/env python3
"""Label-blind patient-balanced TRB atlas and within-cohort depth diagnostics."""
from __future__ import annotations
from release_runtime import private_literal as _private_literal, release_path as _release_path
import argparse
import hashlib
import json
import os
import resource
import time
from pathlib import Path

for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[name] = "8"

import numpy as np
import pandas as pd
from scipy.stats import kruskal
from sklearn.decomposition import PCA

ROOT = Path(_release_path('path_99d8c1c6fde2a6b9'))
OLD = ROOT.parent
MODELS = {"tcrbert_dualtask_aa_mean": 768, "sceptr_cdr3_only": 64}
SEED = 20260930
MAX_RSS = 128 * 1024**3

def require(ok, msg):
    if not ok:
        raise RuntimeError(msg)

def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()

def atomic_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    t = path.with_suffix(path.suffix + ".partial")
    t.write_text(json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n")
    os.replace(t, path)

def rss():
    import psutil
    value = psutil.Process().memory_info().rss
    require(value < MAX_RSS, "RSS over 128 GiB")
    return value

def manifest():
    path = ROOT / "config/cohort_samples.tsv"
    need = ["cohort", "sample_id", "patient_key", "home_batch"]
    d = pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False, usecols=need)
    require(len(d) == 340 and d.sample_id.nunique() == 340, "340-library manifest not ready")
    require(d.groupby("patient_key").cohort.nunique().eq(1).all(), "cross-cohort patient collision")
    return d

def patient_files(m, patient):
    rows = m.loc[m.patient_key.eq(patient)]
    return [ROOT / "data/patient_aa" / f"{r.cohort}__{r.sample_id}.parquet"
            for r in rows.itertuples(index=False)]

def patient_counts(m, patient):
    paths = patient_files(m, patient)
    require(all(p.is_file() for p in paths), f"AA files missing for {patient}")
    parts = [pd.read_parquet(p, columns=["aa", "read_count"]) for p in paths]
    d = pd.concat(parts, ignore_index=True).groupby("aa", as_index=False, sort=True).read_count.sum()
    require(d.aa.is_unique and d.read_count.gt(0).all(), f"patient AA invalid {patient}")
    return d

def diagnose(repeats=3, pilot_patients=None):
    m = manifest()
    patients = sorted(m.patient_key.unique())
    if pilot_patients is not None:
        patients = patients[:pilot_patients]
    summaries = []
    by_patient = {}
    gene_usage = []
    t0 = time.monotonic()
    for i, patient in enumerate(patients):
        frame = patient_counts(m, patient)
        by_patient[patient] = frame.read_count.to_numpy(dtype=np.int64)
        sub = m.loc[m.patient_key.eq(patient)]
        cohort = sub.cohort.iloc[0]
        batch = sub.home_batch.iloc[0] if cohort == "home" else sub.sample_id.iloc[0].split("_", 1)[0]
        lengths = frame.aa.str.len().to_numpy(dtype=np.int32)
        summaries.append(dict(patient_key=patient, cohort=cohort, batch=batch,
                              libraries=len(sub), unique_aa=len(frame),
                              total_reads=int(frame.read_count.sum()),
                              median_aa_length=float(np.median(lengths)),
                              reads_weighted_mean_aa_length=float(np.average(lengths, weights=frame.read_count))))
        for r in sub.itertuples(index=False):
            cp = ROOT / "data/clone_detail" / f"{r.cohort}__{r.sample_id}.parquet"
            clones = pd.read_parquet(cp, columns=["v_gene", "j_gene", "read_count", "qc_pass"])
            clones = clones.loc[clones.qc_pass.eq(True)]
            for kind, col in (("V", "v_gene"), ("J", "j_gene")):
                totals = clones.groupby(col, as_index=False).read_count.sum()
                gene_usage.extend((patient, cohort, batch, kind, str(g), int(n))
                                  for g, n in totals.itertuples(index=False, name=None) if str(g))
        if (i + 1) % 25 == 0:
            print(json.dumps({"stage": "diagnose", "patients": i + 1, "rss_gib": round(rss()/1024**3, 2)}), flush=True)
    summary = pd.DataFrame(summaries)
    common_depth = int(summary.total_reads.min())
    require(common_depth > 0, "zero common depth")
    rare = []
    for patient, counts in by_patient.items():
        total = int(counts.sum())
        rr = np.random.default_rng(int.from_bytes(hashlib.sha256(f"{SEED}:{patient}".encode()).digest()[:8], "little"))
        for rep in range(repeats):
            draw = rr.multivariate_hypergeometric(counts, common_depth) if total > common_depth else counts
            positive = draw[draw > 0].astype(np.float64)
            p = positive / common_depth
            rare.append(dict(patient_key=patient, repeat=rep, equal_reads=common_depth,
                             unique_aa_after=int(len(positive)),
                             shannon_after=float(-(p * np.log(p)).sum())))
        rss()
    rare = pd.DataFrame(rare).merge(summary[["patient_key", "cohort", "batch"]], on="patient_key", validate="many_to_one")
    out = ROOT / "atlas" / ("pilot" if pilot_patients is not None else "")
    out.mkdir(parents=True, exist_ok=True)
    gene = pd.DataFrame(gene_usage, columns=["patient_key", "cohort", "batch", "gene_type", "gene", "reads"])
    gene = gene.groupby(["patient_key", "cohort", "batch", "gene_type", "gene"], as_index=False).reads.sum()
    gene["fraction"] = gene.reads / gene.groupby(["patient_key", "gene_type"]).reads.transform("sum")
    gene.to_parquet(out / "patient_vj_usage.parquet", index=False)
    summary.to_parquet(out / "patient_technical_summary.parquet", index=False)
    rare.to_parquet(out / "equal_depth_repeats.parquet", index=False)
    tests = []
    merged = summary.merge(rare.groupby("patient_key", as_index=False).unique_aa_after.mean(),
                           on="patient_key", validate="one_to_one")
    for cohort, subset in merged.groupby("cohort"):
        for metric in ("total_reads", "unique_aa", "median_aa_length", "unique_aa_after"):
            groups = [g[metric].to_numpy(dtype=float) for _, g in subset.groupby("batch") if len(g) >= 2]
            stat, p = kruskal(*groups) if len(groups) >= 2 else (np.nan, np.nan)
            tests.append(dict(cohort=cohort, metric=metric, test="Kruskal within cohort by source batch",
                              statistic=stat, p=p, eligible_batches=len(groups)))
    for (cohort, kind), part in gene.groupby(["cohort", "gene_type"]):
        top = part.groupby("gene").reads.sum().nlargest(5).index
        ids = summary.loc[summary.cohort.eq(cohort), ["patient_key", "batch"]]
        for symbol in top:
            per = ids.merge(part.loc[part.gene.eq(symbol), ["patient_key", "fraction"]],
                            on="patient_key", how="left").fillna({"fraction": 0})
            groups = [g.fraction.to_numpy(dtype=float) for _, g in per.groupby("batch") if len(g) >= 2]
            stat, p = kruskal(*groups) if len(groups) >= 2 else (np.nan, np.nan)
            tests.append(dict(cohort=cohort, metric=f"{kind}:{symbol}:read_fraction",
                              test="Kruskal within cohort by source batch",
                              statistic=stat, p=p, eligible_batches=len(groups)))
    pd.DataFrame(tests).to_csv(out / "batch_diagnostic_tests.tsv", sep="\t", index=False)
    atomic_json(out / "diagnostic_receipt.json", dict(status="pilot" if pilot_patients is not None else "complete", label_blind=True,
        samples=int(m.loc[m.patient_key.isin(patients)].shape[0]), patients=len(patients), home_patients=int(summary.cohort.eq("home").sum()),
        external_patients=int(summary.cohort.eq("external").sum()), common_depth_reads=common_depth,
        common_depth_definition="minimum patient-level MiXCR readCount among QC-passing atlas patients",
        repeats=repeats, patient_summary_sha256=sha(out / "patient_technical_summary.parquet"),
        rarefaction_sha256=sha(out / "equal_depth_repeats.parquet"),
        vj_usage_sha256=sha(out / "patient_vj_usage.parquet"),
        manifest_sha256=sha(ROOT / "config/cohort_samples.tsv"),
        elapsed_s=round(time.monotonic()-t0, 1), peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss))
    return out / "diagnostic_receipt.json"

def get_index():
    path = ROOT / "data/atlas_aa_index.parquet"
    d = pd.read_parquet(path)
    need = {"seq_idx", "aa", "bank_source", "bank_idx"}
    require(need <= set(d), f"AA index columns missing {need-set(d)}")
    require(np.array_equal(d.seq_idx.to_numpy(dtype=np.int64), np.arange(len(d), dtype=np.int64)),
            "atlas seq_idx noncontiguous")
    require(d.aa.is_unique and d.bank_source.isin(["old", "new"]).all(), "AA index duplicates/source invalid")
    return d

def banks(model):
    require(model in MODELS, "unknown encoder")
    a = np.load(OLD / "embeddings" / model / "bank_fp16.npy", mmap_mode="r")
    b = np.load(ROOT / "embeddings" / model / "bank_new_fp16.npy", mmap_mode="r")
    require(a.ndim == b.ndim == 2 and a.shape[1] == b.shape[1] == MODELS[model], "bank dimensions invalid")
    return a, b

def gather(index, old, new):
    out = np.empty((len(index), old.shape[1]), dtype=np.float32)
    source = index.bank_source.to_numpy()
    loc = index.bank_idx.to_numpy(dtype=np.int64)
    mask = source == "old"
    require((loc[mask] >= 0).all() and (loc[mask] < len(old)).all() and
            (loc[~mask] >= 0).all() and (loc[~mask] < len(new)).all(), "bank row out of range")
    if mask.any():
        out[mask] = old[loc[mask]]
    if (~mask).any():
        out[~mask] = new[loc[~mask]]
    require(np.isfinite(out).all(), "nonfinite bank vectors")
    return out

def draws(scope, k=None):
    require(scope in ("joint", "home"), "scope invalid")
    m = manifest()
    patients = sorted(m.patient_key.unique()) if scope == "joint" else sorted(m.loc[m.cohort.eq("home"), "patient_key"].unique())
    # A few external libraries have fewer than 4,096 unique CDR3 beta sequences.
    # Keep all patients in the descriptive atlas by lowering the common draw size.
    import pyarrow.parquet as pq
    available = [sum(pq.ParquetFile(p).metadata.num_rows for p in patient_files(m, patient))
                 for patient in patients]
    k = min(4096, min(available)) if k is None else k
    require(k > 0 and all(n >= k for n in available), "invalid common PCA draw size")
    index = get_index()
    aa_idx = pd.Index(index.aa)
    records = []
    for patient in patients:
        counts = patient_counts(m, patient)
        require(len(counts) >= k, f"{patient}: fewer than {k} AA; lock a new common draw size before PCA")
        seed = int.from_bytes(hashlib.sha256(f"{SEED}:{scope}:{patient}".encode()).digest()[:8], "little")
        rng = np.random.default_rng(seed)
        chosen = counts.iloc[np.sort(rng.choice(len(counts), size=k, replace=False))]
        idx = aa_idx.get_indexer(chosen.aa)
        require((idx >= 0).all(), f"{patient}: AA absent from index")
        cohort = m.loc[m.patient_key.eq(patient), "cohort"].iloc[0]
        records.extend(zip([patient]*k, [cohort]*k, idx.tolist()))
        if len(records) % (k * 30) == 0:
            rss()
    d = pd.DataFrame.from_records(records, columns=["patient_key", "cohort", "seq_idx"])
    out = ROOT / "atlas" / scope
    out.mkdir(parents=True, exist_ok=True)
    d.to_parquet(out / "pca_patient_balanced_draws.parquet", index=False)
    return d

def fit_pca(scope, model):
    t0 = time.monotonic()
    draw = draws(scope)
    index = get_index()
    old, new = banks(model)
    rows = index.iloc[draw.seq_idx.to_numpy(dtype=np.int64)]
    x = gather(rows, old, new)
    pca = PCA(n_components=64, svd_solver="randomized", random_state=SEED, iterated_power=3)
    pca.fit(x)
    out = ROOT / "atlas" / scope / model
    out.mkdir(parents=True, exist_ok=True)
    np.savez(out / "pca.npz", mean=pca.mean_, components=pca.components_,
             explained_variance=pca.explained_variance_, explained_variance_ratio=pca.explained_variance_ratio_)
    atomic_json(out / "pca_fit_receipt.json", dict(status="complete", label_blind=True,
        scope=scope, model=model, patients=int(draw.patient_key.nunique()),
        draws_per_patient=int(draw.groupby("patient_key").size().iloc[0]), fit_rows=len(draw), dimension=64,
        explained_variance_sum=float(pca.explained_variance_ratio_.sum()),
        input_index_sha256=sha(ROOT / "data/atlas_aa_index.parquet"),
        draw_sha256=sha(ROOT / "atlas" / scope / "pca_patient_balanced_draws.parquet"),
        pca_sha256=sha(out / "pca.npz"), elapsed_s=round(time.monotonic()-t0, 1),
        peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss))
    return out / "pca_fit_receipt.json"

def project(scope, model, chunk_rows=50000):
    t0 = time.monotonic()
    out = ROOT / "atlas" / scope / model
    fit = json.loads((out / "pca_fit_receipt.json").read_text())
    require(fit["status"] == "complete" and fit["input_index_sha256"] == sha(ROOT / "data/atlas_aa_index.parquet"), "stale PCA fit")
    with np.load(out / "pca.npz") as z:
        mean, components = z["mean"], z["components"]
    old, new = banks(model)
    index = get_index()
    final = out / "pca_scores_fp16.npy"
    temp = out / "pca_scores_fp16.partial.npy"
    scores = np.lib.format.open_memmap(temp, mode="w+", dtype=np.float16, shape=(len(index), 64))
    for start in range(0, len(index), chunk_rows):
        end = min(len(index), start + chunk_rows)
        x = gather(index.iloc[start:end], old, new)
        y = (x - mean) @ components.T
        require(np.isfinite(y).all(), f"nonfinite PCA projection {start}:{end}")
        scores[start:end] = y.astype(np.float16)
        if start % (chunk_rows * 100) == 0:
            scores.flush()
            print(json.dumps({"scope": scope, "model": model, "projected": end, "total": len(index),
                              "rss_gib": round(rss()/1024**3, 2)}), flush=True)
    scores.flush()
    del scores
    os.replace(temp, final)
    atomic_json(out / "projection_receipt.json", dict(status="complete", scope=scope, model=model,
        input_index_sha256=fit["input_index_sha256"], pca_sha256=fit["pca_sha256"],
        n_aa=len(index), missing=0, nonfinite=0, scores_sha256=sha(final),
        elapsed_s=round(time.monotonic()-t0, 1),
        peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss))
    return out / "projection_receipt.json"

def display_umap(scope, model, per_patient=100):
    import umap
    out = ROOT / "atlas" / scope / model
    require((out / "projection_receipt.json").is_file(), "PCA projection incomplete")
    d = pd.read_parquet(ROOT / "atlas" / scope / "pca_patient_balanced_draws.parquet")
    selected = d.groupby("patient_key", group_keys=False).sample(n=per_patient, random_state=SEED)
    scores = np.load(out / "pca_scores_fp16.npy", mmap_mode="r")
    x = np.asarray(scores[selected.seq_idx.to_numpy(dtype=np.int64)], dtype=np.float32)
    xy = umap.UMAP(n_components=2, n_neighbors=30, min_dist=.3, random_state=SEED).fit_transform(x)
    selected = selected.copy()
    selected["umap_1"], selected["umap_2"] = xy[:, 0], xy[:, 1]
    selected.to_parquet(out / "umap_balanced_display.parquet", index=False)
    atomic_json(out / "umap_receipt.json", dict(status="descriptive_only", sampled_rows=len(selected),
        per_patient=per_patient, label_blind=True, use_for_group_definition=False,
        data_sha256=sha(out / "umap_balanced_display.parquet")))
    return out / "umap_receipt.json"

def main():
    p = argparse.ArgumentParser()
    p.add_argument("command", choices=["diagnose", "fit", "project", "umap"])
    p.add_argument("--scope", choices=["joint", "home"])
    p.add_argument("--model", choices=list(MODELS))
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--pilot-patients", type=int)
    a = p.parse_args()
    if a.command == "diagnose":
        print(diagnose(a.repeats, a.pilot_patients))
    else:
        require(a.scope and a.model, "scope/model required")
        print({"fit": fit_pca, "project": project, "umap": display_umap}[a.command](a.scope, a.model))
if __name__ == "__main__":
    main()
