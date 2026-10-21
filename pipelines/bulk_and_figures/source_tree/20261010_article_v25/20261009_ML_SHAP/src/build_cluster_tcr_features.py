#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Build B2 = CLUSTERTCR feature matrix (per-TCR features for the 3 frozen P<0.05 clusters).

Frozen cluster definition (pooled_N_20261003/design_freeze.json, all_248_pooled_tests.tsv):
  tcrbert:11  (P=0.0142)  leiden res=1 seed=17, 32 clusters
  sceptr:5    (P=0.0182)  leiden res=1 seed=17, 30 clusters
  tcrbert:7   (P=0.0377)

Feature value per patient p, member TCR t:  log((reads_p,t + 1)/(patient_total_assigned_reads_p + 1))
  - the EXACT pooled289 selection scale (read_fraction_log), same as the CLUSTER3 arm.
  - absent TCR -> log(1/(total+1)) (defined value, no NaN) — same convention as CLUSTER3.

Label-free publicity prefilter (to keep the frozen pipeline computationally feasible):
  keep (model, cluster) member TCR observed with read_count>0 in >= MIN_PATIENTS of the
  289 modeling patients. No label is used. Downstream pipeline (impute->var->Spearman top50
  ->scale->clf) unchanged from run_ml_battery_merged.py.

Conservation check: per patient, sum of reads over ALL member TCRs of each group must equal
  n_bias_20261002/stats/{model}_patient_cluster_abundance.tsv cluster_assigned_reads.
"""
from release_runtime import private_literal as _private_literal, release_path as _release_path
import os, json, hashlib, time
for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[k] = "8"
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

ATLAS = _release_path('path_99d8c1c6fde2a6b9')
MERGED = _release_path('path_6fe8b0cdaf15a381')
OUT = _release_path('path_d4455a6432ab939c')
MIN_PATIENTS = int(os.environ.get("MIN_PATIENTS", "2"))
GROUPS = [("tcrbert", 11), ("sceptr", 5), ("tcrbert", 7)]  # frozen, ordered by pooled P

def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()

t0 = time.time()
# ---- 1. labels + receipts (frozen) ----
lab = {}
for model, nc, d in [("tcrbert", 32, "leiden_full_20261001"), ("sceptr", 30, "leiden_full_sceptr_20261002")]:
    rec = json.load(open(f"{ATLAS}/{d}/clusters/primary_receipt.json"))
    assert rec["status"] == "PASS" and rec["n_clusters"] == nc and rec["resolution"] == 1.0 and rec["seed"] == 17
    lp = f"{ATLAS}/{d}/clusters/primary_labels.npy"
    assert sha(lp) == rec["labels_sha256"], f"labels drift {model}"
    lab[model] = np.load(lp)
    assert lab[model].shape == (28380380,) and lab[model].max() == nc - 1
union_p = f"{ATLAS}{_release_path('path_0395fac586379350')}"
assert sha(union_p) == "66d7cf4e812b5eaa0d151ebab88b652ce0d0ef594bf423a17fa35053d8d66052"
print("labels+union verified", round(time.time() - t0, 1), "s", flush=True)

# ---- 2. member AA sets (scan union once) ----
member_masks = {}
for model, cid in GROUPS:
    member_masks[(model, cid)] = lab[model] == cid
member_aa = {g: set() for g in GROUPS}
union_size = 0
for batch in pq.ParquetFile(union_p).iter_batches(batch_size=1 << 18, columns=["seq_idx", "aa"]):
    d = batch.to_pandas()
    union_size += len(d)
    seq = d["seq_idx"].to_numpy()
    aa = d["aa"].astype(str).to_numpy()
    for g in GROUPS:
        m = member_masks[g][seq]
        if m.any():
            member_aa[g].update(aa[m])
print("union rows", union_size, "| member AA per group:",
      {f"{m}c{c}": len(s) for (m, c), s in member_aa.items()},
      round(time.time() - t0, 1), "s", flush=True)

# ---- 3. modeling patients + libraries ----
design = pd.read_csv(f"{MERGED}/src/merged_design.csv", index_col=0)
keys = list(design.index)
assert len(keys) == 289
libs = pd.read_csv(f"{ATLAS}/config/cohort_samples.tsv", sep="\t", dtype=str, keep_default_na=False)
libs = libs[libs.patient_key.isin(keys)]
assert libs.patient_key.nunique() == 289

# ---- 4. per-patient member reads ----
rows = {}          # patient -> {group: {aa: reads}}
totals = {}        # patient -> total assigned reads
lib_check = {}
for i, key in enumerate(keys):
    part = libs[libs.patient_key == key]
    frames = []
    for _, r in part.iterrows():
        p = f"{ATLAS}{_release_path('path_266633c118edee4b')}{r.cohort}__{r.sample_id}.parquet"
        raw = pd.read_parquet(p, columns=["aa", "read_count"])
        assert raw.aa.duplicated().sum() == 0
        frames.append(raw)
    m = pd.concat(frames).groupby("aa", sort=False)["read_count"].sum()
    assert int(m.sum()) == int(pd.concat(frames)["read_count"].sum())
    totals[key] = int(m.sum())
    sub = {}
    for g in GROUPS:
        s = m[m.index.isin(member_aa[g])]
        if len(s):
            sub[g] = s.to_dict()
    rows[key] = sub
    lib_check[key] = len(part)
    if (i + 1) % 50 == 0:
        print(f"  patients {i+1}/289", round(time.time() - t0, 1), "s", flush=True)

# ---- 5. publicity filter ----
from collections import Counter
cnt = {g: Counter() for g in GROUPS}
for key in keys:
    for g, d in rows[key].items():
        for aa in d:
            cnt[g][aa] += 1
kept = {g: sorted([a for a, c in cnt[g].items() if c >= MIN_PATIENTS]) for g in GROUPS}
D = sum(len(v) for v in kept.values())
print("kept features per group:", {f"{m}c{c}": len(v) for (m, c), v in kept.items()}, "| D =", D, flush=True)

# ---- 6. dense matrix ----
# Encoding (B2-specific fix, documented): present TCR -> -log((reads+1)/(total+1))
#   (negative of the frozen pooled selection scale; larger = more abundant);
# absent TCR -> 0.0. Rationale: filling absent with row-varying -log(total+1) creates
# thousands of IDENTICAL depth-proxies among fold-wise-absent columns, which would flood
# the Spearman top-50; absent->0 makes fold-wise-absent columns all-zero so the frozen
# VarianceThreshold removes them (standard sparse convention).
feat_names, feat_group, feat_aa = [], [], []
for (m, c) in GROUPS:
    for a in kept[(m, c)]:
        feat_names.append(f"{m}{c}::{a}")
        feat_group.append(f"{m}:{c}")
        feat_aa.append(a)
X = np.zeros((289, D))
for i, key in enumerate(keys):
    tot = totals[key]
    rd = rows[key]
    for g, d in rd.items():
        colmap = {a: j for j, a in enumerate(kept[g])}
        for a, r in d.items():
            j = colmap.get(a)
            if j is not None:
                gi = GROUPS.index(g)
                col = sum(len(kept[GROUPS[k]]) for k in range(gi)) + j
                X[i, col] = -np.log((r + 1.0) / (tot + 1.0))
Xdf = pd.DataFrame(X, index=pd.Index(keys, name="patient_key"), columns=feat_names)
os.makedirs(f"{OUT}/src", exist_ok=True)
Xdf.to_csv(f"{OUT}/src/X_clusterTCR_289.csv.gz", index=True, compression="gzip")
Xdf.to_parquet(f"{OUT}/src/X_clusterTCR_289.parquet", index=True)
feat_meta = pd.DataFrame({"feature": feat_names, "group": feat_group, "aa": feat_aa})
feat_meta.to_csv(f"{OUT}/src/clusterTCR_feature_meta.csv", index=False)

# ---- 7. conservation check vs n_bias cluster abundance ----
ok = True
for (m, c) in GROUPS:
    ab = pd.read_csv(f"{ATLAS}/leiden_full_20261001/n_bias_20261002/stats/{m}_patient_cluster_abundance.tsv",
                     sep="\t", keep_default_na=False)
    ab = ab[(ab.cluster_id == c) & ab.patient_key.isin(keys)]
    ref = ab.set_index("patient_key")["cluster_assigned_reads"].astype(int).reindex(keys)
    mine = pd.Series(0, index=pd.Index(keys))
    for key in keys:
        mine[key] = int(sum(rows[key].get((m, c), {}).values()))
    bad = (ref != mine).sum()
    ok &= (bad == 0)
    print(f"conservation {m}:{c} mismatches={bad} (n={int(ref.sum())} reads)", flush=True)
assert ok, "conservation FAILED"

stats = {
    "built_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    "union_sha256": "66d7cf4e812b5eaa0d151ebab88b652ce0d0ef594bf423a17fa35053d8d66052",
    "tcrbert_labels_sha256": "d2dd7f98d335fba4ba73a268c5ad0af987bd996b2947df0712c5721260186399",
    "sceptr_labels_sha256": "a5552c835709065d9392f1b219a9e9a2f3f6c2dc085ed06e023d942dddb3e6d",
    "groups": [f"{m}:{c}" for m, c in GROUPS],
    "member_aa_total": {f"{m}c{c}": len(member_aa[(m, c)]) for m, c in GROUPS},
    "min_patients_filter": MIN_PATIENTS,
    "kept_features": {f"{m}c{c}": len(kept[(m, c)]) for m, c in GROUPS},
    "n_features_total": D,
    "n_patients": 289,
    "conservation": "PASS",
    "scale": "present: -log((reads+1)/(patient_total_assigned_reads+1)); absent: 0",
    "seconds": round(time.time() - t0, 1),
}
json.dump(stats, open(f"{OUT}/src/clusterTCR_build_stats.json", "w"), indent=2)
print(json.dumps(stats, indent=2))
print("BUILD DONE", round(time.time() - t0, 1), "s", flush=True)
