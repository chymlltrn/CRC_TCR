#!/usr/bin/env python3
"""Frozen Home family projection and recorded-N external direction check.

Commands are separate by design: project-gliph2 and integrate never parse
external N; score can only parse it after a complete, hashed projection receipt.
The tcrdist3 projection is run by 04b_tcrdist_projection.py under its own cost
gate and is consumed here only after its frozen-source receipt passes.
"""
from __future__ import annotations
from release_runtime import private_literal as _private_literal, release_path as _release_path

import argparse
import csv
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import resource
import time

for name in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[name] = "1"

import numpy as np
import pandas as pd
import pyarrow.dataset as ds
import pyarrow.parquet as pq

ROOT = Path(_release_path('path_99d8c1c6fde2a6b9'))
GROUPS = ROOT / "groups"
SCOPE = GROUPS / "full_home"
FROZEN = GROUPS / "frozen_home_candidates.tsv"
FROZEN_RECEIPT = GROUPS / "frozen_home_candidates.json"
ASSOC = SCOPE / "association_recorded_home.tsv"
ASSOC_RECEIPT = ROOT / "audit/05_association_receipt.json"
LOCK = ROOT / "audit/05_association_universe_lock.json"
HOME_MEMBER = SCOPE / "group_membership.parquet"
HOME_BURDEN = SCOPE / "patient_group_burden.parquet"
HOME_BURDEN_RECEIPT = SCOPE / "burden_receipt.json"
EXT_MEMBER = SCOPE / "external_frozen_membership.parquet"
EXT_MEMBER_RECEIPT = SCOPE / "external_frozen_membership_receipt.json"
EXT_BURDEN = SCOPE / "patient_group_burden_external.parquet"
EXT_BURDEN_RECEIPT = SCOPE / "burden_receipt_external.json"
PROJECTION_RECEIPT = SCOPE / "external_projection_receipt.json"
DIRECTION = SCOPE / "external_direction.tsv"
DIRECTION_RECEIPT = SCOPE / "external_direction.json"
KEY = ["aa", "v_gene", "j_gene"]
MEMBER_COLS = ["scope", "method", "group_id", *KEY, "seq_idx"]
DIRECTION_COLS = [
    "method", "group_id", "home_adjusted_p", "home_adjusted_q_bh_pooled",
    "n_recorded_Nplus", "n_recorded_N0", "n_carriers_external_known",
    "n_carriers_external_all", "n_carriers_Nplus", "n_carriers_N0",
    "prevalence_Nplus", "prevalence_N0", "presence_delta",
    "read_fraction_median_delta", "evaluable", "external_same_direction",
]


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def ids_hash(values: set[str]) -> str:
    return hashlib.sha256(("\n".join(sorted(values)) + "\n").encode()).hexdigest()


def save_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, sort_keys=True, indent=2, ensure_ascii=False) + "\n")
    tmp.replace(path)


def save_parquet(path: Path, frame: pd.DataFrame) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    frame.to_parquet(tmp, index=False, compression="zstd")
    tmp.replace(path)


def save_tsv(path: Path, frame: pd.DataFrame) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(tmp, sep="\t", index=False, float_format="%.12g")
    tmp.replace(path)


def _identity_manifest() -> pd.DataFrame:
    # Deliberately restrict columns: no external outcome may enter projection.
    m = pd.read_csv(ROOT / "config/cohort_samples.tsv", sep="\t", dtype=str,
                    usecols=["cohort", "sample_id", "patient_key"])
    if len(m) != 340 or set(m.cohort) != {"home", "external"} or (
        m[["cohort", "sample_id", "patient_key"]].isna().any().any()
    ) or m[["cohort", "sample_id"]].duplicated().any():
        raise ValueError("Atlas manifest identity scope is incomplete")
    external = m.loc[m.cohort.eq("external")]
    if len(external) != 202 or external.patient_key.nunique() != 202:
        raise ValueError("Expected 202 single-library External proxy patients")
    return m


def _identity_sha(m: pd.DataFrame) -> str:
    canonical = m[["cohort", "sample_id", "patient_key"]].sort_values(
        ["cohort", "sample_id", "patient_key"])
    return hashlib.sha256(canonical.to_csv(
        sep="\t", index=False, lineterminator="\n").encode()).hexdigest()


def frozen_contract() -> tuple[pd.DataFrame, pd.DataFrame, dict, pd.DataFrame]:
    m = _identity_manifest()
    paths = [FROZEN, FROZEN_RECEIPT, ASSOC, ASSOC_RECEIPT, LOCK,
             HOME_MEMBER, HOME_BURDEN, HOME_BURDEN_RECEIPT]
    if not all(path.is_file() for path in paths):
        raise FileNotFoundError("Home association/frozen family contract incomplete")
    lock = json.loads(FROZEN_RECEIPT.read_text())
    association = json.loads(ASSOC_RECEIPT.read_text())
    universe = json.loads(LOCK.read_text())
    burden_receipt = json.loads(HOME_BURDEN_RECEIPT.read_text())
    if lock.get("status") != "complete" or (
        lock.get("candidate_sha256") != sha(FROZEN)
    ) or lock.get("membership_sha256") != sha(HOME_MEMBER) or (
        lock.get("source_association_sha256") != sha(ASSOC)
    ) or lock.get("eligibility_lock_sha256") != sha(LOCK) or (
        association.get("status") != "complete"
    ) or association.get("external_N_values_parsed") is not False or (
        association.get("association_tsv_sha256") != sha(ASSOC)
    ) or association.get("candidate_tsv_sha256") != sha(FROZEN) or (
        association.get("candidate_receipt_sha256") != sha(FROZEN_RECEIPT)
    ) or universe.get("status") != "frozen_before_home_N_values" or (
        universe.get("input_sha256", {}).get("manifest") != sha(ROOT / "config/cohort_samples.tsv")
    ) or burden_receipt.get("status") != "complete" or (
        burden_receipt.get("patient_group_burden_sha256") != sha(HOME_BURDEN)
    ) or burden_receipt.get("group_membership_sha256") != sha(HOME_MEMBER):
        raise ValueError("Frozen Home association source or hash changed")
    frozen = pd.read_csv(FROZEN, sep="\t")
    required = {"method", "group_id", "adjusted_status", "adjusted_q_bh_pooled",
                "adjusted_beta", "prevalence_difference", "adjusted_p"}
    if required - set(frozen) or frozen[["method", "group_id"]].duplicated().any() or (
        frozen.group_id.duplicated().any()
    ):
        raise ValueError("Frozen candidate schema or group identity invalid")
    if len(frozen) != lock.get("n_candidates") or not set(frozen.method) <= {
        "gliph2", "tcrdist3"
    }:
        raise ValueError("Frozen candidate count or method changed")
    if len(frozen) and not (
        frozen.adjusted_status.eq("ok").all() and
        frozen.adjusted_q_bh_pooled.lt(.05).all() and
        frozen.adjusted_beta.gt(0).all() and
        frozen.prevalence_difference.gt(0).all()
    ):
        raise ValueError("Frozen candidate fails prespecified Home association gate")
    assoc = pd.read_csv(ASSOC, sep="\t")
    selected = assoc.loc[assoc.home_association_candidate.eq(True),
                         ["method", "group_id"]]
    if set(map(tuple, selected.to_numpy())) != set(
        map(tuple, frozen[["method", "group_id"]].to_numpy())
    ):
        raise ValueError("Frozen groups differ from source association selection")
    home_groups = set(pd.read_parquet(HOME_BURDEN, columns=["group_id"]).group_id)
    if not set(frozen.group_id) <= home_groups:
        raise ValueError("Candidate absent from Home qualified burden")
    return m, frozen, lock, assoc


def _source_clone_paths(m: pd.DataFrame) -> list[Path]:
    ext = m.loc[m.cohort.eq("external")]
    paths = [ROOT / "data/clone_detail" / f"external__{sid}.parquet"
             for sid in ext.sample_id]
    if not all(path.is_file() for path in paths):
        raise FileNotFoundError("External clone-detail file missing")
    return paths


def _stem(value: object) -> str:
    return str(value).split("*", 1)[0].strip()


def project_gliph2() -> Path:
    """Project selected literal motif/VJ rules to all External QC clone rows."""
    m, frozen, _, _ = frozen_contract()
    selected = set(frozen.loc[frozen.method.eq("gliph2"), "group_id"].astype(str))
    if not selected:
        raise ValueError("No frozen GLIPH2 group; run integrate directly")
    started = time.monotonic()
    home_receipt_path = SCOPE / "gliph2_projection_receipt.json"
    meta_path = SCOPE / "group_metadata.tsv"
    seed_path = SCOPE / "gliph2_seed_membership.parquet"
    receipt = json.loads(home_receipt_path.read_text())
    if receipt.get("status") != "complete_home_scope" or (
        receipt.get("projection_population") != "home"
    ) or receipt.get("group_metadata_sha256") != sha(meta_path) or (
        receipt.get("seed_membership_sha256") != sha(seed_path)
    ):
        raise ValueError("Home GLIPH2 projection rule changed")
    meta = pd.read_csv(meta_path, sep="\t").drop_duplicates("group_id")
    meta = meta.loc[meta.group_id.isin(selected)].copy()
    if set(meta.group_id.astype(str)) != selected:
        raise ValueError("Frozen GLIPH2 group lacks motif metadata")
    meta["pattern"] = meta.pattern.fillna("").astype(str)
    meta["type"] = meta.type.fillna("").astype(str)
    valid = (
        meta.pattern.str.fullmatch(r"[ACDEFGHIKLMNPQRSTVWY]{4,8}") &
        meta.type.eq("motif-" + meta.pattern) &
        pd.to_numeric(meta.final_score, errors="coerce").le(.05) &
        pd.to_numeric(meta.number_subject, errors="coerce").ge(3)
    )
    if not valid.all() or meta.pattern.duplicated().any():
        raise ValueError("Frozen GLIPH2 motif is ambiguous or unqualified")
    seed = pd.read_parquet(seed_path)
    seed = seed.loc[seed.group_id.isin(selected)].copy()
    if set(seed.group_id.astype(str)) != selected:
        raise ValueError("Frozen GLIPH2 seed group incomplete")
    seed = seed.merge(meta[["group_id", "pattern"]], on="group_id", validate="many_to_one")
    if not seed.apply(lambda row: row.pattern in row.aa, axis=1).all():
        raise ValueError("Frozen motif absent from seed member")
    pairs = {gid: set(zip(part.v_gene.map(_stem), part.j_gene.map(_stem)))
             for gid, part in seed.groupby("group_id")}
    rules = list(meta[["group_id", "pattern"]].itertuples(index=False, name=None))
    paths = _source_clone_paths(m)
    source_stats_before = [[p.name, p.stat().st_size, p.stat().st_mtime_ns] for p in paths]
    external_ids = set(m.loc[m.cohort.eq("external"), "patient_key"])
    scanner = ds.dataset([str(path) for path in paths], format="parquet").scanner(
        columns=["cohort", "patient_key", *KEY],
        filter=ds.field("qc_pass") == True, batch_size=131072)
    hits: set[tuple[str, str, str, str]] = set()
    scanned = valid_vj = 0
    for batch in scanner.to_batches():
        frame = batch.to_pandas()
        scanned += len(frame)
        if not frame.cohort.eq("external").all() or not set(frame.patient_key) <= external_ids:
            raise ValueError("External clone identity mismatch")
        frame = frame.loc[
            frame.v_gene.astype(str).str.match(r"^TRBV[0-9]+(?:-[0-9]+)?(?:\*[0-9]+)?$") &
            frame.j_gene.astype(str).str.match(r"^TRBJ[0-9]+(?:-[0-9]+)?(?:\*[0-9]+)?$")
        ]
        valid_vj += len(frame)
        if frame.empty:
            continue
        vj = pd.MultiIndex.from_arrays([frame.v_gene.map(_stem), frame.j_gene.map(_stem)])
        for gid, motif in rules:
            mask = frame.aa.astype(str).str.contains(motif, regex=False).to_numpy() & vj.isin(pairs[gid])
            hits.update((gid, str(aa), str(v), str(j)) for aa, v, j in
                        frame.loc[mask, KEY].itertuples(index=False, name=None))
    aa_needed = {item[1] for item in hits}
    aa_index: dict[str, int] = {}
    if aa_needed:
        union = pq.ParquetFile(ROOT / "data/aa_union.parquet")
        for batch in union.iter_batches(batch_size=262144, columns=["aa", "seq_idx"]):
            part = batch.to_pandas()
            found = part.loc[part.aa.isin(aa_needed), ["aa", "seq_idx"]]
            aa_index.update(zip(found.aa, found.seq_idx))
        if set(aa_index) != aa_needed:
            raise ValueError("Matched External AA absent from global AA index")
    member = pd.DataFrame([
        ("full_home", "gliph2", gid, aa, v, j, aa_index[aa])
        for gid, aa, v, j in sorted(hits)
    ], columns=MEMBER_COLS)
    if member.empty:
        member = pd.DataFrame({name: pd.Series(dtype=("int64" if name == "seq_idx" else "str"))
                               for name in MEMBER_COLS})
    path = SCOPE / "gliph2_external_frozen_membership.parquet"
    save_parquet(path, member)
    stats = [[p.name, p.stat().st_size, p.stat().st_mtime_ns] for p in paths]
    if stats != source_stats_before:
        raise ValueError("External clone source changed during GLIPH2 projection")
    out = SCOPE / "gliph2_external_frozen_projection_receipt.json"
    save_json(out, {
        "status": "complete_external_frozen_scope", "scope": "full_home",
        "method": "gliph2", "projection_population": "external",
        "external_label_read": False,
        "projection_type": "gliph2_seed_motif_projection",
        "frozen_candidates_sha256": sha(FROZEN),
        "frozen_candidate_receipt_sha256": sha(FROZEN_RECEIPT),
        "home_membership_sha256": sha(HOME_MEMBER),
        "home_gliph2_projection_receipt_sha256": sha(home_receipt_path),
        "external_identity_sha256": ids_hash(external_ids),
        "manifest_identity_sha256": _identity_sha(m),
        "group_ids_sha256": ids_hash(selected),
        "membership_sha256": sha(path),
        "source_clone_file_count": len(paths),
        "source_clone_stats_sha256": hashlib.sha256(json.dumps(
            stats, separators=(",", ":")).encode()).hexdigest(),
        "scanned_qc_clone_rows": scanned, "valid_vj_clone_rows": valid_vj,
        "n_exact_aa_vj_memberships": len(member),
        "n_frozen_groups": len(selected), "n_groups_with_external_hits": member.group_id.nunique(),
        "elapsed_seconds": time.monotonic() - started,
        "peak_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
    })
    return out


def _load_method(m: pd.DataFrame, method: str, selected: set[str]) -> tuple[pd.DataFrame, dict]:
    prefix = "gliph2" if method == "gliph2" else "tcrdist"
    path = SCOPE / f"{prefix}_external_frozen_membership.parquet"
    receipt_path = SCOPE / f"{prefix}_external_frozen_projection_receipt.json"
    if not path.is_file() or not receipt_path.is_file():
        raise FileNotFoundError(f"Frozen external {method} projection incomplete")
    receipt = json.loads(receipt_path.read_text())
    external_ids = set(m.loc[m.cohort.eq("external"), "patient_key"])
    if receipt.get("status") != "complete_external_frozen_scope" or (
        receipt.get("method") != method
    ) or receipt.get("scope") != "full_home" or (
        receipt.get("projection_population") != "external"
    ) or receipt.get("external_label_read") is not False or (
        receipt.get("frozen_candidates_sha256") != sha(FROZEN)
    ) or receipt.get("frozen_candidate_receipt_sha256") != sha(FROZEN_RECEIPT) or (
        receipt.get("home_membership_sha256") != sha(HOME_MEMBER)
    ) or receipt.get("membership_sha256") != sha(path) or (
        receipt.get("external_identity_sha256") != ids_hash(external_ids)
    ) or receipt.get("group_ids_sha256") != ids_hash(selected) or (
        receipt.get("manifest_identity_sha256") != _identity_sha(m)
    ):
        raise ValueError(f"Frozen external {method} receipt/hash invalid")
    if method == "gliph2":
        source = SCOPE / "gliph2_projection_receipt.json"
        paths = _source_clone_paths(m)
        stats = [[p.name, p.stat().st_size, p.stat().st_mtime_ns] for p in paths]
        if receipt.get("home_gliph2_projection_receipt_sha256") != sha(source) or (
            receipt.get("source_clone_file_count") != len(paths)
        ) or receipt.get("source_clone_stats_sha256") != hashlib.sha256(json.dumps(
            stats, separators=(",", ":")).encode()).hexdigest():
            raise ValueError("External GLIPH2 clone source changed")
    else:
        source = SCOPE / "tcrdist_home_projection_receipt.json"
        home = json.loads(source.read_text())
        if home.get("status") != "complete_home_scope" or (
            receipt.get("home_projection_receipt_sha256") != sha(source)
        ) or receipt.get("home_tcrdist_membership_sha256") != sha(
            SCOPE / "tcrdist_home_group_membership.parquet"
        ):
            raise ValueError("External tcrdist3 Home rule changed")
    member = pd.read_parquet(path)
    if set(MEMBER_COLS) - set(member) or (
        not member.scope.eq("full_home").all()
    ) or not member.method.eq(method).all() or (
        not set(member.group_id) <= selected
    ) or member[["group_id", *KEY]].duplicated().any():
        raise ValueError(f"External {method} membership invalid")
    return member[MEMBER_COLS], {"receipt_sha256": sha(receipt_path),
                                 "membership_sha256": sha(path)}


def _load_group_module():
    path = ROOT / "src/04_groups_stats.py"
    spec = importlib.util.spec_from_file_location("atlas_groups_stats", path)
    if spec is None or spec.loader is None:
        raise ImportError("Cannot load Atlas group burden implementation")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def integrate() -> Path:
    """Finalize label-blind membership and 202-patient burden before N access."""
    m, frozen, _, _ = frozen_contract()
    external_ids = set(m.loc[m.cohort.eq("external"), "patient_key"])
    selected = set(frozen.group_id.astype(str))
    parts = []
    sources = {}
    for method, part in frozen.groupby("method"):
        table, source = _load_method(m, method, set(part.group_id.astype(str)))
        parts.append(table)
        sources[method] = source
    combined = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame({
        name: pd.Series(dtype=("int64" if name == "seq_idx" else "str"))
        for name in MEMBER_COLS
    })
    if combined[["group_id", *KEY]].duplicated().any():
        raise ValueError("External method membership collision")
    save_parquet(EXT_MEMBER, combined)
    save_json(EXT_MEMBER_RECEIPT, {
        "status": "complete_external_frozen_scope", "scope": "full_home",
        "projection_population": "external", "external_label_read": False,
        "frozen_candidates_sha256": sha(FROZEN),
        "frozen_candidate_receipt_sha256": sha(FROZEN_RECEIPT),
        "home_membership_sha256": sha(HOME_MEMBER),
        "external_identity_sha256": ids_hash(external_ids),
        "group_ids_sha256": ids_hash(selected),
        "methods": sources, "membership_sha256": sha(EXT_MEMBER),
        "n_membership_rows": len(combined), "n_frozen_groups": len(selected),
    })
    if selected:
        module = _load_group_module()
        burden = module.patient_burden(
            ROOT, "full_home", external_ids, cohort="external",
            group_ids=selected, projection_membership_path=EXT_MEMBER)
        module._assert_complete_burden_grid(burden, external_ids, selected, "full_home")
        module._validate_burden_receipt(ROOT, "full_home", cohort="external")
    else:
        empty = pd.DataFrame({name: pd.Series(dtype=("int64" if name in {
            "group_reads", "group_unique_aa", "group_unique_aa_vj", "total_reads",
            "total_unique_aa", "presence"
        } else "float64" if name in {"unique_aa_fraction", "read_fraction"}
            else "str")) for name in [
            "scope", "patient_key", "group_id", "group_reads", "group_unique_aa",
            "group_unique_aa_vj", "total_reads", "total_unique_aa", "presence",
            "unique_aa_fraction", "read_fraction",
        ]})
        save_parquet(EXT_BURDEN, empty)
        save_json(EXT_BURDEN_RECEIPT, {
            "status": "complete_empty_frozen_universe",
            "target_cohort": "external", "n_patients": 202, "n_groups": 0,
            "patient_group_burden_sha256": sha(EXT_BURDEN),
            "external_frozen_membership_sha256": sha(EXT_MEMBER),
            "external_frozen_membership_receipt_sha256": sha(EXT_MEMBER_RECEIPT),
            "external_label_read": False,
        })
    save_json(PROJECTION_RECEIPT, {
        "status": "complete", "scope": "full_home",
        "external_n_read": False, "projection_population": "all_202_external_proxy_patients",
        "projection_rule": "frozen Home q<0.05 motif/tcrdist rules; exact AA/V/J join for patient burden",
        "frozen_candidates_sha256": sha(FROZEN),
        "frozen_candidate_receipt_sha256": sha(FROZEN_RECEIPT),
        "source_association_sha256": sha(ASSOC),
        "home_membership_sha256": sha(HOME_MEMBER),
        "home_burden_sha256": sha(HOME_BURDEN),
        "home_burden_receipt_sha256": sha(HOME_BURDEN_RECEIPT),
        "manifest_identity_sha256": _identity_sha(m),
        "external_identity_sha256": ids_hash(external_ids),
        "external_identity_count": len(external_ids),
        "external_frozen_membership_sha256": sha(EXT_MEMBER),
        "external_frozen_membership_receipt_sha256": sha(EXT_MEMBER_RECEIPT),
        "external_burden_sha256": sha(EXT_BURDEN),
        "external_burden_receipt_sha256": sha(EXT_BURDEN_RECEIPT),
        "method_sources": sources,
        "n_frozen_groups": len(selected),
        "n_burden_rows": len(selected) * len(external_ids),
        "empty_candidate_universe": len(selected) == 0,
    })
    return PROJECTION_RECEIPT


def _check_projection_before_labels() -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    m, frozen, _, _ = frozen_contract()
    if not PROJECTION_RECEIPT.is_file():
        raise FileNotFoundError("External label-blind projection receipt missing")
    receipt = json.loads(PROJECTION_RECEIPT.read_text())
    external_ids = set(m.loc[m.cohort.eq("external"), "patient_key"])
    expected = {
        "status": "complete", "external_n_read": False,
        "frozen_candidates_sha256": sha(FROZEN),
        "frozen_candidate_receipt_sha256": sha(FROZEN_RECEIPT),
        "source_association_sha256": sha(ASSOC),
        "home_membership_sha256": sha(HOME_MEMBER),
        "home_burden_sha256": sha(HOME_BURDEN),
        "home_burden_receipt_sha256": sha(HOME_BURDEN_RECEIPT),
        "manifest_identity_sha256": _identity_sha(m),
        "external_identity_sha256": ids_hash(external_ids),
        "external_identity_count": len(external_ids),
        "external_frozen_membership_sha256": sha(EXT_MEMBER),
        "external_frozen_membership_receipt_sha256": sha(EXT_MEMBER_RECEIPT),
        "external_burden_sha256": sha(EXT_BURDEN),
        "external_burden_receipt_sha256": sha(EXT_BURDEN_RECEIPT),
        "n_frozen_groups": len(frozen),
        "n_burden_rows": len(frozen) * len(external_ids),
    }
    if any(receipt.get(key) != value for key, value in expected.items()):
        raise ValueError("External frozen projection changed before outcome read")
    membership_receipt = json.loads(EXT_MEMBER_RECEIPT.read_text())
    burden_receipt = json.loads(EXT_BURDEN_RECEIPT.read_text())
    if membership_receipt.get("status") != "complete_external_frozen_scope" or (
        membership_receipt.get("external_label_read") is not False
    ) or membership_receipt.get("membership_sha256") != sha(EXT_MEMBER) or (
        burden_receipt.get("patient_group_burden_sha256") != sha(EXT_BURDEN)
    ):
        raise ValueError("External member or burden receipt incomplete")
    for method, part in frozen.groupby("method"):
        _, source = _load_method(m, method, set(part.group_id.astype(str)))
        if receipt.get("method_sources", {}).get(method) != source:
            raise ValueError("External method projection source changed")
    burden = pd.read_parquet(EXT_BURDEN)
    groups = set(frozen.group_id.astype(str))
    if len(burden) != len(groups) * len(external_ids) or (
        not burden.empty and (
            burden[["patient_key", "group_id"]].duplicated().any() or
            set(burden.patient_key) != external_ids or
            set(burden.group_id) != groups
        )
    ):
        raise ValueError("External burden grid incomplete")
    if len(groups):
        module = _load_group_module()
        module._validate_burden_receipt(ROOT, "full_home", cohort="external")
    return m, frozen, receipt


def score() -> Path:
    """Read External N only after projection is complete and source-verified."""
    m, frozen, projection = _check_projection_before_labels()
    # The first read of external N in this program occurs strictly here.
    labels = pd.read_csv(ROOT / "config/cohort_samples.tsv", sep="\t", dtype=str,
                         usecols=["cohort", "patient_key", "recorded_n_binary"])
    labels = labels.loc[labels.cohort.eq("external"),
                         ["patient_key", "recorded_n_binary"]]
    if len(labels) != 202 or labels.patient_key.duplicated().any() or (
        set(labels.patient_key) != set(m.loc[m.cohort.eq("external"), "patient_key"])
    ):
        raise ValueError("External recorded-N identity mismatch")
    y = labels.recorded_n_binary
    if not y.fillna("").isin(["", "0", "1"]).all():
        raise ValueError("External recorded N nonbinary")
    known = labels.loc[y.isin(["0", "1"])].copy()
    if len(known) != 163 or known.recorded_n_binary.nunique() != 2:
        raise ValueError("Expected 163 External N-known patients with both classes")
    burden = pd.read_parquet(EXT_BURDEN)
    group_map = frozen.set_index("group_id")
    scored = burden.merge(known, on="patient_key", how="inner", validate="many_to_one")
    rows = []
    for gid, part in scored.groupby("group_id", sort=True):
        pos = part.loc[part.recorded_n_binary.eq("1")]
        neg = part.loc[part.recorded_n_binary.eq("0")]
        if len(pos) != int(known.recorded_n_binary.eq("1").sum()) or (
            len(neg) != int(known.recorded_n_binary.eq("0").sum())
        ):
            raise ValueError("Scored group missing an N-known patient")
        all_group = burden.loc[burden.group_id.eq(gid)]
        carriers_known = int(part.presence.sum())
        delta = float(pos.presence.mean() - neg.presence.mean())
        row = group_map.loc[gid]
        rows.append({
            "method": row.method, "group_id": gid,
            "home_adjusted_p": row.adjusted_p,
            "home_adjusted_q_bh_pooled": row.adjusted_q_bh_pooled,
            "n_recorded_Nplus": len(pos), "n_recorded_N0": len(neg),
            "n_carriers_external_known": carriers_known,
            "n_carriers_external_all": int(all_group.presence.sum()),
            "n_carriers_Nplus": int(pos.presence.sum()),
            "n_carriers_N0": int(neg.presence.sum()),
            "prevalence_Nplus": float(pos.presence.mean()),
            "prevalence_N0": float(neg.presence.mean()),
            "presence_delta": delta,
            "read_fraction_median_delta": float(
                pos.read_fraction.median() - neg.read_fraction.median()),
            "evaluable": carriers_known >= 5,
            "external_same_direction": carriers_known >= 5 and delta > 0,
        })
    direction = pd.DataFrame(rows, columns=DIRECTION_COLS)
    if set(direction.group_id) != set(frozen.group_id):
        raise ValueError("External direction missing frozen group")
    save_tsv(DIRECTION, direction)
    label_canonical = known.sort_values("patient_key").values.tolist()
    label_sha = hashlib.sha256(json.dumps(label_canonical, ensure_ascii=False,
                                          separators=(",", ":")).encode()).hexdigest()
    save_json(DIRECTION_RECEIPT, {
        "status": "complete", "endpoint": "recorded N+; pN/ypN unverified",
        "projection_receipt_sha256": sha(PROJECTION_RECEIPT),
        "frozen_candidates_sha256": sha(FROZEN),
        "direction_tsv_sha256": sha(DIRECTION),
        "external_label_sha256": label_sha,
        "n_external_all": 202, "n_external_N_known": len(known),
        "n_recorded_Nplus": int(known.recorded_n_binary.eq("1").sum()),
        "n_recorded_N0": int(known.recorded_n_binary.eq("0").sum()),
        "n_frozen_groups": len(frozen), "n_scored_groups": len(direction),
        "n_evaluable_groups": int(direction.evaluable.sum()) if len(direction) else 0,
        "n_same_direction": int(direction.external_same_direction.sum()) if len(direction) else 0,
        "inference": "directional external recheck; historical cohort and BioSample proxy identity",
    })
    return DIRECTION


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["check-frozen", "project-gliph2",
                                           "integrate", "score"])
    args = parser.parse_args()
    soft, hard = resource.getrlimit(resource.RLIMIT_AS)
    cap = 128 * 1024**3
    if soft == resource.RLIM_INFINITY or soft > cap:
        resource.setrlimit(resource.RLIMIT_AS, (min(cap, hard) if hard != resource.RLIM_INFINITY else cap, hard))
    if args.command == "check-frozen":
        _, frozen, _, _ = frozen_contract()
        print(json.dumps({"status": "PASS", "n_frozen_groups": len(frozen),
                          "candidate_sha256": sha(FROZEN)}))
    elif args.command == "project-gliph2":
        print(project_gliph2())
    elif args.command == "integrate":
        print(integrate())
    else:
        print(score())


if __name__ == "__main__":
    main()
