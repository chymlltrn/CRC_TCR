#!/usr/bin/env python3
"""Prespecified patient-level sensitivity analyses for exact TRB families.

Inputs are frozen group memberships and dense patient burdens from
04_groups_stats.py. No group or depth threshold is selected using recorded N.
Full-Home results remain exploratory while source material is unverified. MiXCR readCount measures reads, not cells.

OLGA human_TRB Pgen is optional post-selection characterization of frozen
Home candidates, never a primary association or evidence of antigen binding.
See https://github.com/statbiophys/OLGA/blob/master/README.md
"""
from __future__ import annotations
from release_runtime import private_literal as _private_literal, release_path as _release_path

import argparse
import hashlib
import importlib.metadata
import importlib.util
import json
import math
import os
from functools import lru_cache
from pathlib import Path
import resource
import time
import warnings

for _v in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_v] = str(min(8, int(os.environ.get(_v, "8"))))

import numpy as np
import pandas as pd
from scipy.special import gammaln
from statsmodels.stats.contingency_tables import StratifiedTable
from statsmodels.stats.multitest import multipletests

SUBSETS = ("all_recorded_home_exploratory", "verified_blood")
TESTS = (
    ("unique_fraction", "unique_aa_fraction"),
    ("read_fraction", "read_fraction"),
    ("rare_capture", "capture_probability"),
    ("vj_length_adjusted_presence", "presence"),
)
PLAN_VERSION = 3


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def save_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n")
    tmp.replace(path)


def _receipt(root: Path, scope: str) -> dict:
    path = root / "groups" / scope / "integrated_projection_receipt.json"
    if not path.exists():
        raise RuntimeError(f"STOP_missing_complete_dictionary: {path}")
    value = json.loads(path.read_text())
    if (value.get("status") != "complete_home_scope" or
            value.get("projection_population") != "home" or
            value.get("scope") != scope):
        raise RuntimeError(
            f"STOP_incomplete_dictionary: {path}: "
            f"status={value.get('status')}, population={value.get('projection_population')}, "
            f"scope={value.get('scope')}")
    return value



@lru_cache(maxsize=1)
def _groups_stats():
    """Load the canonical Stage 04 validators from its numeric filename."""
    source = Path(__file__).with_name("04_groups_stats.py")
    spec = importlib.util.spec_from_file_location("stage04_groups_stats", source)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"STOP_missing_stage04_validator: {source}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _validate_upstream(root: Path, scope: str, subset: str) -> None:
    """Require the completed Home discovery chain before 04c consumes it."""
    if scope != "full_home":
        raise ValueError("04c is scoped to descriptive full_home only")
    if subset not in SUBSETS:
        raise ValueError(f"Unknown subset {subset}")
    _receipt(root, scope)
    frozen = root / "groups/frozen_home_candidates.tsv"
    frozen_receipt = frozen.with_suffix(".json")
    if not frozen.is_file() or not frozen_receipt.is_file():
        raise RuntimeError("STOP_missing_frozen_Home_candidate_chain")
    receipt = json.loads(frozen_receipt.read_text())
    if receipt.get("candidate_sha256") != sha256(frozen):
        raise RuntimeError("STOP_frozen_Home_candidate_hash_changed")
    try:
        primary = _groups_stats()
        # This checks integration, patient burden, the [PRIVATE_SUBJECT] association
        # family and pooled BH values, and the canonical frozen ranking.
        primary._validate_frozen_file(root, frozen)
        if subset == "verified_blood":
            primary._validate_association_receipt(root, scope, subset)
    except (OSError, KeyError, ValueError, RuntimeError) as exc:
        raise RuntimeError(f"STOP_invalid_Home_discovery_chain: {exc}") from exc


def _paths(root: Path, scope: str, subset: str) -> dict[str, Path]:
    if subset not in SUBSETS:
        raise ValueError(f"Unknown subset {subset}")
    base = root / "groups" / scope
    return {
        "burden": base / "patient_group_burden.parquet",
        "membership": base / "group_membership.parquet",
        "nodes": base / "patient_nodes.parquet",
        "eligible": base / f"eligible_groups_{subset}.tsv",
        "integrated_receipt": base / "integrated_projection_receipt.json",
        "burden_receipt": base / "burden_receipt.json",
        "association_receipt": base / f"association_{subset}.json",
        "frozen": root / "groups/frozen_home_candidates.tsv",
        "frozen_receipt": root / "groups/frozen_home_candidates.json",
        "primary_source": Path(__file__).with_name("04_groups_stats.py"),
        "plan": base / f"sensitivity_plan_{subset}.json",
        "patient": base / f"sensitivity_patient_burden_{subset}.parquet",
        "tests": base / f"sensitivity_tests_{subset}.tsv",
        "batch_strata": base / f"sensitivity_batch_strata_{subset}.tsv",
        "batch_cmh": base / f"sensitivity_batch_cmh_{subset}.tsv",
        "receipt": base / f"sensitivity_receipt_{subset}.json",
        "pilot": base / f"sensitivity_pilot_{subset}.json",
    }


def _subset_keys(root: Path, subset: str) -> set[str]:
    manifest = pd.read_csv(root / "config/cohort_samples.tsv", sep="\t", dtype=str,
                           keep_default_na=False,
                           usecols=["cohort", "patient_key"])
    home = manifest.loc[manifest.cohort.eq("home")]
    keys = set(home.patient_key)
    if subset == "verified_blood":
        documented = pd.read_csv(root / "config/documented_blood31.tsv", sep="\t", dtype=str,
                                 keep_default_na=False,
                                 usecols=["patient_key", "documented_blood"])
        keys &= set(documented.loc[documented.documented_blood.str.lower().eq("yes"), "patient_key"])
    return keys


def _inputs(root: Path, scope: str, subset: str) -> tuple[dict[str, Path], pd.DataFrame, pd.DataFrame]:
    _validate_upstream(root, scope, subset)
    paths = _paths(root, scope, subset)
    for name in ("burden", "membership", "nodes", "eligible"):
        if not paths[name].is_file():
            raise RuntimeError(f"STOP_missing_input_{name}: {paths[name]}")
    burden = pd.read_parquet(paths["burden"])
    eligible = pd.read_csv(paths["eligible"], sep="\t", dtype={"group_id": str})
    if eligible.group_id.duplicated().any():
        raise ValueError("Duplicate eligible group ID")
    allowed = _subset_keys(root, subset)
    burden = burden.loc[burden.patient_key.astype(str).isin(allowed) &
                        burden.group_id.astype(str).isin(set(eligible.group_id))].copy()
    if eligible.empty:
        raise RuntimeError("STOP_no_eligible_groups")
    n_patients = len(allowed)
    counts = burden.groupby("group_id").patient_key.nunique()
    if len(counts) != len(eligible) or not counts.eq(n_patients).all():
        raise ValueError("Incomplete eligible patient x group burden matrix")
    if {"n_present", "n_patients"} - set(eligible):
        raise ValueError("Eligibility table lacks patient counts")
    actual_present = burden.groupby("group_id").presence.sum()
    stated = eligible.set_index("group_id")
    if not stated.n_patients.eq(n_patients).all() or not stated.n_present.eq(actual_present).all():
        raise ValueError("Frozen eligibility counts disagree with burden")
    min_carriers = 10 if subset == "all_recorded_home_exploratory" else 3
    if not ((stated.n_present >= min_carriers) &
            ((stated.n_patients - stated.n_present) >= min_carriers)).all():
        raise ValueError("Frozen eligible family fails predeclared carrier threshold")
    totals = burden.groupby("patient_key")[["total_reads", "total_unique_aa"]].nunique()
    if not totals.eq(1).all().all() or len(totals) != n_patients:
        raise ValueError("Missing/inconsistent patient totals")
    if burden[["group_reads", "group_unique_aa", "total_reads", "total_unique_aa"]].isna().any().any():
        raise ValueError("Burden has missing counts")
    if (burden.group_reads < 0).any() or (burden.group_reads > burden.total_reads).any():
        raise ValueError("Group reads outside patient total")
    for field in ("group_reads", "total_reads"):
        arr = burden[field].to_numpy(dtype=float)
        if not np.all(np.isfinite(arr)) or not np.all(arr == np.rint(arr)) or np.max(arr) >= 2**53:
            raise ValueError(f"{field} must be finite exactly representable integer counts")
    return paths, burden, eligible


def choose_depth(totals: pd.DataFrame, fixed_depth: int | None = None) -> int:
    """Depth is fixed from patient read totals only; all patients retained."""
    values = pd.to_numeric(totals.total_reads, errors="raise").to_numpy(dtype=float)
    if len(values) == 0 or not np.isfinite(values).all() or np.min(values) <= 0:
        raise ValueError("Invalid patient read totals")
    floor = int(np.min(values))
    if fixed_depth is None:
        return floor
    if fixed_depth < 1 or fixed_depth > floor:
        raise ValueError("Fixed depth must be positive and <= every target patient total")
    return fixed_depth


def plan(root: Path, scope: str, subset: str, seed: int = 20260929,
         repeats: int = 5, fixed_depth: int | None = None,
         max_rows: int = 5_000_000) -> Path:
    if repeats < 2 or repeats > 50 or max_rows < 1:
        raise ValueError("repeats must be 2..50 and max_rows > 0")
    paths, burden, eligible = _inputs(root, scope, subset)
    totals = burden[["patient_key", "total_reads"]].drop_duplicates()
    depth = choose_depth(totals, fixed_depth)
    rows = len(burden)
    import pyarrow.parquet as pq
    nodes_rows = pq.ParquetFile(paths["nodes"]).metadata.num_rows
    if rows > max_rows:
        raise RuntimeError(f"STOP_patient_group_grid_exceeds_{max_rows}: {rows}")
    estimated_rss_gib = (rows * 640 + nodes_rows * 160) / 1024**3
    if estimated_rss_gib > 128:
        raise RuntimeError(f"STOP_estimated_RSS_exceeds_128GiB: {estimated_rss_gib:.1f}")
    payload = {
        "status": "locked_label_blind", "version": PLAN_VERSION,
        "scope": scope, "subset": subset, "seed": seed, "repeats": repeats,
        "rarefaction_depth_reads": depth,
        "depth_rule": "minimum QC read total across all included patients; outcome-blind",
        "depth_fraction_of_median_total": float(depth / totals.total_reads.median()),
        "n_patients": len(totals), "n_groups": len(eligible),
        "n_patient_group_rows": rows, "n_patient_variant_rows": nodes_rows,
        "estimated_peak_rss_gib": estimated_rss_gib,
        "cpu_threads_limit": 8, "rss_soft_cap_gib": 128,
        "rarefaction": "exact hypergeometric marginal per patient and group; independent group-wise draws are not a joint repertoire subsample",
        "tests": [name for name, _ in TESTS],
        "batch_sensitivity": {
            "unit": "patient", "exposure": "binary family presence",
            "strata": "library_batch",
            "adjustment": "library_batch stratification only; no age, sex, or depth adjustment",
            "outputs": "within-batch N+ and N0 carrier rates/deltas; CMH common OR and nominal two-sided p",
            "cmh_rule": "at least two informative batches; no continuity correction or zero-cell shift",
            "sparse_rule": "any batch with fewer than five N+ or N0 patients is flagged; finite CMH estimates remain descriptive only",
            "heterogeneity": "batch delta direction and range only; no formal homogeneity claim",
            "multiple_testing": "not in primary pooled BH family; no q or candidate-selection use",
        },
        "pgen": "separate OLGA candidate-characterization command; no primary q contribution",
        "input_sha256": {name: sha256(paths[name]) for name in (
            "burden", "membership", "nodes", "eligible", "integrated_receipt",
            "burden_receipt", "association_receipt", "frozen", "frozen_receipt",
            "primary_source")},
        "source_sha256": sha256(Path(__file__).resolve()),
        "home_n_read_only_for_upstream_provenance_validation": True,
        "no_recorded_n_used_for_sensitivity_parameter_selection": True,
    }
    save_json(paths["plan"], payload)
    return paths["plan"]


def capture_probability(total: np.ndarray, group: np.ndarray, depth: int) -> np.ndarray:
    """Exact P(at least one family read) for depth without replacement."""
    r = np.asarray(total, dtype=np.int64)
    g = np.asarray(group, dtype=np.int64)
    if r.shape != g.shape or np.any(r < depth) or np.any(g < 0) or np.any(g > r):
        raise ValueError("Invalid hypergeometric counts")
    out = np.zeros(len(r), dtype=float)
    impossible_zero = depth > r - g
    out[impossible_zero] = 1.0
    possible = ~impossible_zero
    rr = r[possible].astype(float)
    gg = g[possible].astype(float)
    # log choose(R-G,d) - log choose(R,d)
    lp0 = (gammaln(rr-gg+1) - gammaln(depth+1) - gammaln(rr-gg-depth+1)
           - gammaln(rr+1) + gammaln(depth+1) + gammaln(rr-depth+1))
    out[possible] = -np.expm1(np.minimum(lp0, 0.0))
    out[g == 0] = 0.0
    return np.clip(out, 0.0, 1.0)


def _seed(seed: int, scope: str, subset: str, group_id: str, rep: int) -> int:
    data = f"{seed}|{scope}|{subset}|{group_id}|{rep}".encode()
    return int.from_bytes(hashlib.sha256(data).digest()[:8], "little")


def rarefy(burden: pd.DataFrame, depth: int, seed: int, repeats: int,
           scope: str, subset: str) -> pd.DataFrame:
    """One patient contributes one mean per group, regardless of replicates."""
    out = burden.copy().sort_values(["group_id", "patient_key"]).reset_index(drop=True)
    out["capture_probability"] = capture_probability(
        out.total_reads.to_numpy(dtype=np.int64), out.group_reads.to_numpy(dtype=np.int64), depth)
    out["rare_presence_mean"] = 0.0
    out["rare_read_fraction_mean"] = 0.0
    for group_id, indices in out.groupby("group_id", sort=True).indices.items():
        pos = np.asarray(indices, dtype=np.int64)
        g = out.loc[pos, "group_reads"].to_numpy(dtype=np.int64)
        r = out.loc[pos, "total_reads"].to_numpy(dtype=np.int64)
        presence = np.zeros(len(pos), dtype=np.float64)
        reads = np.zeros(len(pos), dtype=np.float64)
        for rep in range(repeats):
            rng = np.random.default_rng(_seed(seed, scope, subset, str(group_id), rep))
            draw = rng.hypergeometric(g, r-g, depth).astype(np.int64)
            presence += draw > 0
            reads += draw
        out.loc[pos, "rare_presence_mean"] = presence / repeats
        out.loc[pos, "rare_read_fraction_mean"] = reads / (repeats * depth)
    return out


def _background(root: Path, scope: str, eligible: pd.DataFrame,
                target_patients: set[str], timing: dict | None = None) -> pd.DataFrame:
    """Outcome-blind patient repertoire composition for each family signature."""
    entered = time.monotonic()
    paths = _paths(root, scope, "all_recorded_home_exploratory")
    members = pd.read_parquet(paths["membership"],
                              columns=["group_id", "aa", "v_gene", "j_gene"])
    members = members.loc[members.group_id.isin(set(eligible.group_id))].copy()
    if members.empty or set(members.group_id) != set(eligible.group_id):
        raise ValueError("Eligible family missing from membership")
    members["length"] = members.aa.astype(str).str.len()
    signature = members.groupby("group_id").agg(
        v_values=("v_gene", lambda x: frozenset(x.astype(str))),
        j_values=("j_gene", lambda x: frozenset(x.astype(str))),
        lengths=("length", lambda x: frozenset(x.astype(int))),
        member_aa=("aa", "nunique"),
        member_aa_vj=("aa", "size")).reset_index()
    nodes = pd.read_parquet(paths["nodes"],
                            columns=["patient_key", "aa", "v_gene", "j_gene"])
    nodes = nodes.loc[nodes.patient_key.astype(str).isin(target_patients)].copy()
    nodes["length"] = nodes.aa.astype(str).str.len()
    hist = {}
    for feature in ("v_gene", "j_gene", "length"):
        hist[feature] = nodes.groupby(["patient_key", feature], observed=True).size()
    totals = nodes.groupby("patient_key").size()
    if set(totals.index) != target_patients:
        raise ValueError("Patient nodes missing for target subset")
    loop_start = time.monotonic()
    rows = []
    for rec in signature.itertuples(index=False):
        for patient in sorted(target_patients):
            n = int(totals.loc[patient])
            def fraction(field: str, values: frozenset) -> float:
                h = hist[field]
                return sum(int(h.get((patient, value), 0)) for value in values) / n
            rows.append((patient, rec.group_id,
                         fraction("v_gene", rec.v_values),
                         fraction("j_gene", rec.j_values),
                         fraction("length", rec.lengths)))
    if timing is not None:
        timing["one_time_seconds"] = loop_start-entered
        timing["per_group_loop_seconds"] = time.monotonic()-loop_start
    return pd.DataFrame(rows, columns=["patient_key", "group_id",
                                       "v_background_fraction", "j_background_fraction",
                                       "length_background_fraction"])


def _labels(root: Path, subset: str, patients: set[str]) -> pd.DataFrame:
    manifest = pd.read_csv(root / "config/cohort_samples.tsv", sep="\t", dtype=str,
                           usecols=["cohort", "patient_key", "recorded_n_binary"])
    home = manifest.loc[manifest.cohort.eq("home") & manifest.patient_key.isin(patients)]
    home = home.drop_duplicates("patient_key")
    cov = pd.read_csv(root / "config/home_clinical_covariates.tsv", sep="\t",
                      dtype=str, usecols=["patient_key", "age_years", "sex", "library_batch"])
    if cov.patient_key.duplicated().any():
        raise ValueError("Duplicate clinical patient")
    frame = home.merge(cov, on="patient_key", how="left", validate="one_to_one")
    if set(frame.patient_key) != patients or not frame.recorded_n_binary.isin(["0", "1"]).all():
        raise ValueError("Incomplete Home labels")
    if frame[["age_years", "sex", "library_batch"]].isna().any().any():
        raise ValueError("Missing clinical covariates")
    frame["recorded_n_binary"] = frame.recorded_n_binary.astype(int)
    frame["age_years"] = pd.to_numeric(frame.age_years, errors="raise")
    return frame


def _fit_patient_glm(part: pd.DataFrame, exposure: str,
                     composition: bool) -> tuple[float, float, str]:
    import statsmodels.api as sm
    from statsmodels.tools.sm_exceptions import PerfectSeparationWarning
    y = part.recorded_n_binary.to_numpy(dtype=float)
    values = part[exposure].to_numpy(dtype=float)
    if not np.isfinite(values).all():
        return math.nan, math.nan, "nonfinite_exposure"
    if exposure in ("unique_aa_fraction", "read_fraction"):
        values = np.log1p(values * 1e6)
    x = pd.DataFrame({"exposure": values,
                      "log_reads": np.log1p(part.total_reads.to_numpy(dtype=float)),
                      "log_unique_aa": np.log1p(part.total_unique_aa.to_numpy(dtype=float)),
                      "age_years": part.age_years.to_numpy(dtype=float)}, index=part.index)
    if composition:
        for col in ("v_background_fraction", "j_background_fraction",
                    "length_background_fraction"):
            x[col] = part[col].to_numpy(dtype=float)
    for col in list(x):
        sd = float(x[col].std())
        if not np.isfinite(sd) or sd <= 0:
            if col == "exposure":
                return math.nan, math.nan, "constant_exposure"
            x = x.drop(columns=[col])
        else:
            x[col] = (x[col] - x[col].mean()) / sd
    dummies = pd.get_dummies(part[["sex", "library_batch"]].astype(str),
                             drop_first=True, dtype=float)
    x = pd.concat([x, dummies], axis=1)
    x = sm.add_constant(x, has_constant="add")
    if np.linalg.matrix_rank(x.to_numpy()) != x.shape[1]:
        return math.nan, math.nan, "rank_deficient"
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", PerfectSeparationWarning)
            fitted = sm.GLM(y, x, family=sm.families.Binomial()).fit(maxiter=100, disp=0)
        beta = float(fitted.params["exposure"])
        se = float(fitted.bse["exposure"])
        p = float(fitted.pvalues["exposure"])
        if not fitted.converged or not np.isfinite([beta, se, p]).all() or abs(beta) > 10 or se > 10:
            return math.nan, math.nan, "separation_or_unstable"
        return float(np.exp(beta)), p, "ok"
    except Exception as exc:
        return math.nan, math.nan, type(exc).__name__


def patient_tests(frame: pd.DataFrame) -> pd.DataFrame:
    results = []
    for group_id, part in frame.groupby("group_id", sort=True):
        y = part.recorded_n_binary.to_numpy(dtype=int)
        for name, exposure in TESTS:
            odds, p, status = _fit_patient_glm(part, exposure,
                                                composition=name == "vj_length_adjusted_presence")
            results.append({
                "group_id": group_id, "metric": name,
                "n_patients": len(part),
                "n_positive": int(y.sum()), "n_negative": int((y == 0).sum()),
                "or_per_1sd": odds, "p": p, "status": status,
                "median_positive": float(np.median(part.loc[y == 1, exposure])),
                "median_negative": float(np.median(part.loc[y == 0, exposure])),
            })
    output = pd.DataFrame(results)
    p = np.where(np.isfinite(output.p.to_numpy(dtype=float)),
                 output.p.to_numpy(dtype=float), 1.0)
    output["q_sensitivity_pooled"] = multipletests(p, method="fdr_bh")[1]
    return output



def batch_presence_sensitivity(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Descriptive batch-stratified carrier rates and nominal CMH sensitivity."""
    needed = {"group_id", "patient_key", "library_batch", "recorded_n_binary", "presence"}
    if needed - set(frame) or frame.empty:
        raise ValueError("Batch sensitivity lacks complete patient/group fields")
    if frame[["group_id", "patient_key"]].duplicated().any() or\
            frame[["group_id", "patient_key", "library_batch"]].isna().any().any() or\
            frame.library_batch.astype(str).str.strip().eq("").any() or\
            not frame.recorded_n_binary.isin([0, 1]).all() or\
            not frame.presence.isin([0, 1]).all():
        raise ValueError("Invalid patient-level batch sensitivity input")
    strata_rows, cmh_rows = [], []
    for group_id, group in frame.groupby("group_id", sort=True):
        tables = []
        deltas = []
        sparse_batches = zero_cell_batches = 0
        for batch, part in group.groupby("library_batch", sort=True):
            y = part.recorded_n_binary.to_numpy(dtype=int)
            x = part.presence.to_numpy(dtype=int)
            a = int(np.sum((y == 1) & (x == 1)))
            b = int(np.sum((y == 1) & (x == 0)))
            c = int(np.sum((y == 0) & (x == 1)))
            d = int(np.sum((y == 0) & (x == 0)))
            n_positive, n_negative = a+b, c+d
            positive_rate = a/n_positive if n_positive else math.nan
            negative_rate = c/n_negative if n_negative else math.nan
            delta = positive_rate-negative_rate if n_positive and n_negative else math.nan
            informative = bool(n_positive and n_negative and a+c and b+d)
            sparse = bool(min(n_positive, n_negative) < 5)
            zero_cell = bool(min(a, b, c, d) == 0)
            sparse_batches += sparse
            zero_cell_batches += zero_cell
            if informative:
                tables.append(np.array([[a, b], [c, d]], dtype=np.int64))
            if math.isfinite(delta):
                deltas.append(delta)
            if not n_positive or not n_negative:
                status = "single_outcome_class"
            elif not a+c or not b+d:
                status = "no_carrier_variation"
            elif zero_cell:
                status = "zero_cell_or_unestimable"
            elif sparse:
                status = "sparse_class_lt5"
            else:
                status = "ok"
            strata_rows.append({
                "group_id": str(group_id), "library_batch": str(batch),
                "n_patients": len(part), "n_positive": n_positive,
                "n_negative": n_negative, "positive_carriers": a,
                "positive_absent": b, "negative_carriers": c,
                "negative_absent": d,
                "positive_carrier_rate": positive_rate,
                "negative_carrier_rate": negative_rate,
                "presence_delta": delta,
                "within_batch_or": a*d/(b*c) if not zero_cell else math.nan,
                "cmh_informative": informative,
                "sparse_class_lt5": sparse, "zero_cell": zero_cell,
                "status": status,
            })
        signs = np.sign(deltas)
        n_positive_delta = int(np.sum(signs > 0))
        n_negative_delta = int(np.sum(signs < 0))
        n_zero_delta = int(np.sum(signs == 0))
        result = {
            "group_id": str(group_id), "n_batches": group.library_batch.nunique(),
            "n_informative_batches": len(tables),
            "n_excluded_batches": group.library_batch.nunique()-len(tables),
            "n_sparse_class_lt5_batches": sparse_batches,
            "n_zero_cell_batches": zero_cell_batches,
            "n_positive_delta_batches": n_positive_delta,
            "n_negative_delta_batches": n_negative_delta,
            "n_zero_delta_batches": n_zero_delta,
            "direction_discordant": bool(n_positive_delta and n_negative_delta),
            "batch_presence_delta_range": max(deltas)-min(deltas) if len(deltas) >= 2 else math.nan,
            "cmh_common_or": math.nan, "cmh_p_nominal": math.nan,
            "cmh_status": "not_estimable_fewer_than_two_informative_batches",
        }
        if len(tables) >= 2:
            try:
                with warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter("always", RuntimeWarning)
                    cmh = StratifiedTable(tables, shift_zeros=False)
                    odds = float(cmh.oddsratio_pooled)
                    p_value = float(cmh.test_null_odds(correction=False).pvalue)
                if not math.isfinite(odds) or odds <= 0 or not math.isfinite(p_value) or\
                        not 0 <= p_value <= 1:
                    result["cmh_status"] = "not_estimable_nonfinite_common_or_or_p"
                else:
                    result["cmh_common_or"] = odds
                    result["cmh_p_nominal"] = p_value
                    result["cmh_status"] = (
                        "sparse_or_zero_cell_exploratory" if sparse_batches or zero_cell_batches else
                        "ok_exploratory")
                    if any(issubclass(w.category, RuntimeWarning) for w in caught):
                        result["cmh_status"] = "numerical_warning_exploratory"
            except (FloatingPointError, ValueError, ZeroDivisionError) as exc:
                result["cmh_status"] = f"not_estimable_{type(exc).__name__}"
        cmh_rows.append(result)
    return pd.DataFrame(strata_rows), pd.DataFrame(cmh_rows)


def pilot(root: Path, scope: str, subset: str,
          max_estimated_hours: float = 4.0) -> Path:
    """Outcome-blind empirical cost check on small, middle, large families."""
    start = time.monotonic()
    paths, burden, eligible = _inputs(root, scope, subset)
    if not paths["plan"].exists():
        raise RuntimeError("STOP_missing_label_blind_sensitivity_plan")
    frozen = json.loads(paths["plan"].read_text())
    if frozen.get("source_sha256") != sha256(Path(__file__).resolve()):
        raise RuntimeError("STOP_sensitivity_source_changed_after_plan")
    for name, old_hash in frozen["input_sha256"].items():
        if sha256(paths[name]) != old_hash:
            raise RuntimeError(f"STOP_input_changed_after_sensitivity_plan: {name}")
    members = pd.read_parquet(paths["membership"], columns=["group_id"])
    sizes = members.loc[members.group_id.isin(set(eligible.group_id))].groupby("group_id").size()
    if len(sizes) != len(eligible):
        raise ValueError("Missing eligible group in membership")
    ordered = list(sizes.sort_values().index)
    choices = sorted(set((ordered[0], ordered[len(ordered)//2], ordered[-1])))
    subset_eligible = eligible.loc[eligible.group_id.isin(choices)]
    part = burden.loc[burden.group_id.isin(choices)]
    rare_start = time.monotonic()
    sampled = rarefy(part, int(frozen["rarefaction_depth_reads"]), int(frozen["seed"]),
                      int(frozen["repeats"]), scope, subset)
    rare_seconds = time.monotonic()-rare_start
    bg_timing = {}
    backgrounds = _background(root, scope, subset_eligible, set(part.patient_key), bg_timing)
    # Benchmark the same four GLMs with synthetic alternating outcomes; the
    # real recorded-N column is deliberately not accessed during this gate.
    fake = sampled.merge(backgrounds, on=["patient_key", "group_id"],
                         validate="one_to_one")
    cov = pd.read_csv(root / "config/home_clinical_covariates.tsv", sep="\t", dtype=str,
                      usecols=["patient_key", "age_years", "sex", "library_batch"])
    fake = fake.merge(cov, on="patient_key", how="left", validate="many_to_one")
    fake["age_years"] = pd.to_numeric(fake.age_years, errors="raise")
    synthetic_y = {k: i % 2 for i, k in enumerate(sorted(set(fake.patient_key)))}
    fake["recorded_n_binary"] = fake.patient_key.map(synthetic_y)
    glm_start = time.monotonic()
    patient_tests(fake)
    batch_presence_sensitivity(fake)
    glm_seconds = time.monotonic()-glm_start
    elapsed = time.monotonic()-start
    variable_seconds = rare_seconds + bg_timing["per_group_loop_seconds"] + glm_seconds
    projected = (elapsed + variable_seconds * (len(eligible)/len(choices)-1)) / 3600
    peak_gib = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024**2
    status = "PASS" if projected <= max_estimated_hours and peak_gib < 128 else "STOP_resource_cost"
    save_json(paths["pilot"], {
        "status": status, "scope": scope, "subset": subset,
        "plan_sha256": sha256(paths["plan"]),
        "selected_group_ids": choices, "selected_group_sizes": {k: int(sizes.loc[k]) for k in choices},
        "pilot_seconds": elapsed, "pilot_rarefaction_seconds": rare_seconds,
        "pilot_background_one_time_seconds": bg_timing["one_time_seconds"],
        "pilot_background_group_loop_seconds": bg_timing["per_group_loop_seconds"],
        "projected_hours": projected,
        "projection_rule": "one-time pilot cost plus observed per-group rarefaction, composition, and synthetic-label GLM cost times remaining groups",
        "max_estimated_hours": max_estimated_hours,
        "peak_rss_gib": peak_gib, "peak_rss_cap_gib": 128,
        "home_n_read_only_for_upstream_provenance_validation": True,
        "no_recorded_n_used_for_pilot_parameter_selection": True,
    })
    return paths["pilot"]


def run(root: Path, scope: str, subset: str) -> Path:
    start = time.monotonic()
    paths, burden, eligible = _inputs(root, scope, subset)
    if not paths["plan"].is_file():
        raise RuntimeError("STOP_missing_label_blind_sensitivity_plan")
    if not paths["pilot"].is_file():
        raise RuntimeError("STOP_missing_sensitivity_resource_pilot")
    pilot_receipt = json.loads(paths["pilot"].read_text())
    if pilot_receipt.get("status") != "PASS" or pilot_receipt.get("plan_sha256") != sha256(paths["plan"]):
        raise RuntimeError("STOP_failed_or_stale_sensitivity_resource_pilot")
    resource.setrlimit(resource.RLIMIT_AS, (128 * 1024**3, 128 * 1024**3))
    frozen = json.loads(paths["plan"].read_text())
    if frozen.get("source_sha256") != sha256(Path(__file__).resolve()):
        raise RuntimeError("STOP_sensitivity_source_changed_after_plan")
    if frozen.get("status") != "locked_label_blind" or frozen.get("version") != PLAN_VERSION:
        raise ValueError("Sensitivity plan is not locked")
    for name, old_hash in frozen["input_sha256"].items():
        if sha256(paths[name]) != old_hash:
            raise RuntimeError(f"STOP_input_changed_after_sensitivity_plan: {name}")
    if len(burden) != frozen["n_patient_group_rows"] or len(eligible) != frozen["n_groups"]:
        raise ValueError("Sensitivity plan dimensions changed")
    target = set(burden.patient_key.astype(str))
    sampled = rarefy(burden, int(frozen["rarefaction_depth_reads"]), int(frozen["seed"]),
                      int(frozen["repeats"]), scope, subset)
    background = _background(root, scope, eligible, target)
    sampled = sampled.merge(background, on=["patient_key", "group_id"],
                            how="left", validate="one_to_one")
    if sampled[["v_background_fraction", "j_background_fraction",
                "length_background_fraction"]].isna().any().any():
        raise ValueError("Missing composition background")
    labels = _labels(root, subset, target)
    scored = sampled.merge(labels, on="patient_key", how="left", validate="many_to_one")
    tests = patient_tests(scored)
    batch_strata, batch_cmh = batch_presence_sensitivity(scored)
    # Write staged complete tables only after every group has been tested.
    pt_tmp = paths["patient"].with_suffix(".tmp.parquet")
    ts_tmp = paths["tests"].with_suffix(".tmp.tsv")
    strata_tmp = paths["batch_strata"].with_suffix(".tmp.tsv")
    cmh_tmp = paths["batch_cmh"].with_suffix(".tmp.tsv")
    sampled.to_parquet(pt_tmp, index=False)
    tests.to_csv(ts_tmp, sep="\t", index=False)
    batch_strata.to_csv(strata_tmp, sep="\t", index=False)
    batch_cmh.to_csv(cmh_tmp, sep="\t", index=False)
    pt_tmp.replace(paths["patient"])
    ts_tmp.replace(paths["tests"])
    strata_tmp.replace(paths["batch_strata"])
    cmh_tmp.replace(paths["batch_cmh"])
    save_json(paths["receipt"], {
        "status": "complete", "scope": scope, "subset": subset,
        "plan_sha256": sha256(paths["plan"]),
        "patient_table_sha256": sha256(paths["patient"]),
        "tests_sha256": sha256(paths["tests"]),
        "batch_strata_sha256": sha256(paths["batch_strata"]),
        "batch_cmh_sha256": sha256(paths["batch_cmh"]),
        "n_batch_strata_rows": len(batch_strata),
        "n_batch_cmh_rows": len(batch_cmh),
        "n_cmh_estimable": int(batch_cmh.cmh_common_or.notna().sum()),
        "n_cmh_sparse_or_zero_cell": int(batch_cmh.cmh_status.eq(
            "sparse_or_zero_cell_exploratory").sum()),
        "cmh_p_is_nominal_only": True,
        "cmh_in_primary_pooled_BH_family": False,
        "cmh_used_for_candidate_selection": False,
        "cmh_is_not_the_main_adjusted_model": True,
        "batch_heterogeneity_interpretation": "descriptive carrier-rate delta direction/range; sparse batch and asymptotic CMH p cannot confirm homogeneity",
        "n_patients": len(target), "n_groups": len(eligible),
        "n_patient_group_rows": len(sampled),
        "n_tests": len(tests), "n_failed_models": int(tests.status.ne("ok").sum()),
        "failed_models_counted_in_BH_as_p1": True,
        "pgen_in_primary_test_family": False,
        "unit": "patient", "read_count_unit": "MiXCR reads, not cells",
        "rarefaction_depth_reads": frozen["rarefaction_depth_reads"],
        "rarefaction_repeats": frozen["repeats"],
        "rarefaction_draw_coupling": "independent exact group marginal draws; not a joint repertoire subsample",
        "material_interpretation": "report-explicit peripheral blood" if subset == "verified_blood" else
            "Home recorded N exploratory; source material remains partly unverified",
        "n_documented_blood_included": len(target & _subset_keys(root, "verified_blood")),
        "n_material_unverified_included": len(target - _subset_keys(root, "verified_blood")),
        "elapsed_seconds": time.monotonic()-start,
        "peak_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
    })
    return paths["tests"]


def _olga_model():
    import olga
    import olga.load_model as load_model
    import olga.generation_probability as pgen
    from olga.performance.fast_pgen import FastPgen
    folder = Path(olga.__file__).resolve().parent / "default_models/human_T_beta"
    names = ("model_params.txt", "model_marginals.txt",
             "V_gene_CDR3_anchors.csv", "J_gene_CDR3_anchors.csv")
    files = {name: folder / name for name in names}
    if not all(p.is_file() for p in files.values()):
        raise RuntimeError("STOP_OLGA_human_TRB_model_files_unavailable")
    genomic = load_model.GenomicDataVDJ()
    genomic.load_igor_genomic_data(str(files["model_params.txt"]),
                                   str(files["V_gene_CDR3_anchors.csv"]),
                                   str(files["J_gene_CDR3_anchors.csv"]))
    generative = load_model.GenerativeModelVDJ()
    generative.load_and_process_igor_model(str(files["model_marginals.txt"]))
    model = pgen.GenerationProbabilityVDJ(generative, genomic)
    FastPgen(model)
    return model, {name: sha256(path) for name, path in files.items()}


def _length_matched_background(clone_nodes: Path, excluded: set[str],
                               lengths: set[int], per_length: int = 100) -> list[str]:
    """Lowest stable-hash Home AA per CDR3 length; no phenotype is read."""
    import heapq
    import pyarrow.parquet as pq
    if per_length < 1:
        raise ValueError("per_length must be positive")
    heaps: dict[int, list[tuple[int, str]]] = {length: [] for length in lengths}
    selected: dict[int, set[str]] = {length: set() for length in lengths}
    reader = pq.ParquetFile(clone_nodes)
    if "aa" not in reader.schema.names:
        raise ValueError("Home clone_nodes lacks AA")
    for batch in reader.iter_batches(columns=["aa"], batch_size=131072):
        for seq in batch.column(0).to_pylist():
            if not seq or seq in excluded:
                continue
            length = len(seq)
            if length not in heaps or seq in selected[length]:
                continue
            if not seq.startswith("C") or not seq.endswith(("F", "W")):
                continue
            key = int.from_bytes(hashlib.blake2b(seq.encode(), digest_size=8).digest(), "little")
            heap = heaps[length]
            if len(heap) < per_length:
                heapq.heappush(heap, (-key, seq))
                selected[length].add(seq)
            elif key < -heap[0][0]:
                _, removed = heapq.heapreplace(heap, (-key, seq))
                selected[length].remove(removed)
                selected[length].add(seq)
    return sorted(seq for values in selected.values() for seq in values)


def pgen_candidates(root: Path, scope: str = "full_home", max_sequences: int = 50_000,
                    max_estimated_hours: float = 2.0) -> Path:
    """Characterize every AA in frozen Home candidates with the installed OLGA model."""
    start = time.monotonic()
    out = root / "groups" / scope
    candidates_path = root / "groups/frozen_home_candidates.tsv"
    membership_path = out / "group_membership.parquet"
    receipt_path = out / "sensitivity_pgen_receipt.json"
    if not candidates_path.is_file():
        raise RuntimeError("STOP_no_frozen_Home_candidates_for_Pgen")
    if not membership_path.is_file():
        raise RuntimeError("STOP_no_full_Home_membership_for_Pgen")
    _validate_upstream(root, scope, "all_recorded_home_exploratory")
    provenance = {
        "scope": scope,
        "candidate_sha256": sha256(candidates_path),
        "candidate_receipt_sha256": sha256(candidates_path.with_suffix(".json")),
        "membership_sha256": sha256(membership_path),
        "home_association_receipt_sha256": sha256(
            out / "association_all_recorded_home_exploratory.json"),
        "burden_receipt_sha256": sha256(out / "burden_receipt.json"),
        "integrated_projection_receipt_sha256": sha256(
            out / "integrated_projection_receipt.json"),
        "source_sha256": sha256(Path(__file__).resolve()),
        "Pgen_used_for_primary_q": False,
        "Pgen_used_for_candidate_selection": False,
        "pgen_definition": "unconditioned CDR3 AA; not V/J-adjusted",
        "pgen_method": "Post-selection OLGA human TRB unconditioned CDR3 AA Pgen for frozen candidates; label-blind length-matched Home AA background is descriptive; no V/J-conditioned model",
    }

    def unavailable(status: str, reason: str, **details: object) -> Path:
        save_json(receipt_path, {**provenance, "status": status,
                                 "unavailable_reason": reason,
                                 "no_pgen_values_fabricated": True, **details})
        return receipt_path

    frozen = pd.read_csv(candidates_path, sep="\t", dtype={"group_id": str})
    if "group_id" not in frozen or frozen.group_id.duplicated().any():
        raise ValueError("Invalid frozen candidate list")
    if frozen.empty:
        return unavailable("STOP_no_frozen_candidates", "No frozen Home candidates")
    member = pd.read_parquet(membership_path, columns=["group_id", "aa"])
    member = member.loc[member.group_id.isin(set(frozen.group_id))].drop_duplicates()
    if set(member.group_id) != set(frozen.group_id):
        raise ValueError("Frozen candidate membership incomplete")
    aa = sorted(set(member.aa.astype(str)))
    if len(aa) > max_sequences:
        return unavailable("STOP_Pgen_sequence_cap",
                           f"{len(aa)} unique candidate AA exceeds cap {max_sequences}",
                           n_unique_candidate_aa=len(aa), max_sequences=max_sequences)
    try:
        model, hashes = _olga_model()
    except Exception as exc:
        return unavailable(f"STOP_Pgen_model_unavailable_{type(exc).__name__}",
                           f"OLGA model unavailable: {exc}", detail=str(exc))
    valid = [x for x in aa if x.startswith("C") and x.endswith(("F", "W"))]
    clone_nodes_path = out / "clone_nodes.parquet"
    if not clone_nodes_path.is_file():
        raise RuntimeError("STOP_no_Home_clone_nodes_for_Pgen_background")
    background_aa = _length_matched_background(clone_nodes_path, set(aa),
                                               {len(x) for x in valid})
    # Warm JIT on a reference human TRB CDR3 before timing the sample.
    by_length = sorted(valid, key=lambda x: (len(x), x))
    pilot = [by_length[i] for i in np.linspace(0, len(by_length)-1,
             min(25, len(by_length)), dtype=int)] if by_length else []
    try:
        reference_pgen = float(model.compute_aa_CDR3_pgen(_private_literal('private_32619d760ab0d399'), print_warnings=False))
        if not math.isclose(reference_pgen, 5.26507446955e-08, rel_tol=1e-6):
            raise ValueError("OLGA reference Pgen mismatch")
    except Exception as exc:
        return unavailable(f"STOP_Pgen_reference_{type(exc).__name__}",
                           f"OLGA reference check failed: {exc}", detail=str(exc))
    pilot_start = time.monotonic()
    try:
        for seq in pilot:
            model.compute_aa_CDR3_pgen(seq, print_warnings=False)
    except Exception as exc:
        return unavailable(f"STOP_Pgen_pilot_{type(exc).__name__}",
                           f"OLGA timing pilot failed: {exc}", detail=str(exc))
    pilot_seconds = time.monotonic()-pilot_start
    projected_hours = pilot_seconds / max(1, len(pilot)) * (len(valid)+len(background_aa)) / 3600
    if projected_hours > max_estimated_hours:
        return unavailable("STOP_Pgen_projected_time",
                           f"Projected {projected_hours:.3f} hours exceeds cap {max_estimated_hours:.3f}",
                           pilot_n=len(pilot), pilot_seconds=pilot_seconds,
                           projected_hours=projected_hours,
                           max_estimated_hours=max_estimated_hours)
    records = []
    for seq in aa:
        anchor = seq.startswith("C") and seq.endswith(("F", "W"))
        if not anchor:
            records.append((seq, math.nan, False, "STOP_anchor_definition"))
            continue
        try:
            value = float(model.compute_aa_CDR3_pgen(seq, print_warnings=False))
            status = "ok" if math.isfinite(value) and value > 0 else "zero_or_nonfinite"
            records.append((seq, value, True, status))
        except Exception as exc:
            records.append((seq, math.nan, True, f"error_{type(exc).__name__}"))
    sequence = pd.DataFrame(records, columns=["aa", "pgen_aa_unconditioned",
                                               "anchor_valid", "pgen_status"])
    bg_records = []
    for seq in background_aa:
        try:
            value = float(model.compute_aa_CDR3_pgen(seq, print_warnings=False))
            status = "ok" if math.isfinite(value) and value > 0 else "zero_or_nonfinite"
            bg_records.append((seq, len(seq), value, status))
        except Exception as exc:
            bg_records.append((seq, len(seq), math.nan, f"error_{type(exc).__name__}"))
    background = pd.DataFrame(bg_records, columns=["aa", "length",
                                                     "pgen_aa_unconditioned", "pgen_status"])
    if background.pgen_status.astype(str).str.startswith("error_").any():
        n_errors = int(background.pgen_status.astype(str).str.startswith("error_").sum())
        return unavailable("STOP_Pgen_background_errors",
                           f"OLGA failed for {n_errors} background AA", n_errors=n_errors,
                           no_partial_pgen_published=True)
    matched = {}
    for length, rows in background.loc[background.pgen_status.eq("ok")].groupby("length"):
        matched[int(length)] = np.sort(np.log10(rows.pgen_aa_unconditioned.to_numpy(dtype=float)))
    percentiles = []
    for row in sequence.itertuples(index=False):
        reference = matched.get(len(row.aa))
        if row.pgen_status != "ok" or reference is None or len(reference) == 0:
            percentiles.append(math.nan)
        else:
            percentiles.append(float(np.searchsorted(reference, math.log10(row.pgen_aa_unconditioned),
                                                     side="right") / len(reference)))
    sequence["length_matched_pgen_percentile"] = percentiles
    if sequence.pgen_status.astype(str).str.startswith("error_").any():
        n_errors = int(sequence.pgen_status.astype(str).str.startswith("error_").sum())
        return unavailable("STOP_Pgen_sequence_errors",
                           f"OLGA failed for {n_errors} candidate AA", n_errors=n_errors,
                           no_partial_pgen_published=True)
    joined = member.merge(sequence, on="aa", how="left", validate="many_to_one")
    groups = []
    for group_id, part in joined.groupby("group_id", sort=True):
        positive = part.loc[part.pgen_status.eq("ok"), "pgen_aa_unconditioned"].to_numpy(dtype=float)
        groups.append({
            "group_id": group_id, "n_unique_member_aa": len(part),
            "n_anchor_valid": int(part.anchor_valid.sum()),
            "n_pgen_positive": len(positive),
            "n_pgen_zero": int(part.pgen_status.eq("zero_or_nonfinite").sum()),
            "median_log10_pgen": float(np.median(np.log10(positive))) if len(positive) else math.nan,
            "n_length_matched_background_available": int(part.length_matched_pgen_percentile.notna().sum()),
            "median_length_matched_pgen_percentile": float(part.length_matched_pgen_percentile.median()) if
                part.length_matched_pgen_percentile.notna().any() else math.nan,
            "pgen_interpretation": "unconditioned AA generation only; not antigen specificity",
        })
    seq_path = out / "sensitivity_pgen_sequences.parquet"
    bg_path = out / "sensitivity_pgen_background.parquet"
    group_path = out / "sensitivity_pgen_groups.tsv"
    seq_tmp = seq_path.with_suffix(".tmp.parquet")
    bg_tmp = bg_path.with_suffix(".tmp.parquet")
    group_tmp = group_path.with_suffix(".tmp.tsv")
    sequence.to_parquet(seq_tmp, index=False)
    background.to_parquet(bg_tmp, index=False)
    pd.DataFrame(groups).to_csv(group_tmp, sep="\t", index=False)
    seq_tmp.replace(seq_path)
    bg_tmp.replace(bg_path)
    group_tmp.replace(group_path)
    save_json(receipt_path, {
        **provenance,
        "status": "complete", "model": "OLGA installed human_T_beta",
        "olga_version": importlib.metadata.version("olga"),
        "model_file_sha256": hashes,
        "model_source": "https://github.com/statbiophys/OLGA",
        "candidate_sha256": sha256(candidates_path),
        "candidate_receipt_sha256": sha256(candidates_path.with_suffix(".json")),
        "home_association_receipt_sha256": sha256(
            out / "association_all_recorded_home_exploratory.json"),
        "burden_receipt_sha256": sha256(out / "burden_receipt.json"),
        "integrated_projection_receipt_sha256": sha256(
            out / "integrated_projection_receipt.json"),
        "membership_sha256": sha256(membership_path),
        "sequence_sha256": sha256(seq_path), "background_sha256": sha256(bg_path),
        "background_source_sha256": sha256(clone_nodes_path),
        "group_sha256": sha256(group_path),
        "n_unique_candidate_aa": len(aa), "n_anchor_valid": len(valid),
        "n_length_matched_Home_background_aa": len(background_aa),
        "background_per_length_cap": 100,
        "background_selection": "lowest fixed blake2b hash among label-blind Home clone nodes at same length",
        "n_pgen_positive": int(sequence.pgen_status.eq("ok").sum()),
        "n_pgen_zero": int(sequence.pgen_status.eq("zero_or_nonfinite").sum()),
        "n_anchor_stop": int(sequence.pgen_status.eq("STOP_anchor_definition").sum()),
        "pilot_n": len(pilot), "pilot_seconds": pilot_seconds,
        "projected_hours": projected_hours,
        "elapsed_seconds": time.monotonic()-start,
        "peak_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "Pgen_used_for_primary_q": False,
        "Pgen_used_for_candidate_selection": False,
        "length_matched_percentile_is_descriptive": True,
        "interpretation": "CDR3 definition and default model may be mismatched for non C...F/W sequences; no forced Pgen",
    })
    return group_path


def cli() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("plan")
    p.add_argument("--scope", default="full_home")
    p.add_argument("--subset", choices=SUBSETS, default=SUBSETS[0])
    p.add_argument("--seed", type=int, default=20260929)
    p.add_argument("--repeats", type=int, default=5)
    p.add_argument("--fixed-depth", type=int)
    p.add_argument("--max-rows", type=int, default=5_000_000)
    p = sub.add_parser("pilot")
    p.add_argument("--scope", default="full_home")
    p.add_argument("--subset", choices=SUBSETS, default=SUBSETS[0])
    p.add_argument("--max-estimated-hours", type=float, default=4.0)
    p = sub.add_parser("run")
    p.add_argument("--scope", default="full_home")
    p.add_argument("--subset", choices=SUBSETS, default=SUBSETS[0])
    p = sub.add_parser("pgen")
    p.add_argument("--scope", default="full_home")
    p.add_argument("--max-sequences", type=int, default=50_000)
    p.add_argument("--max-estimated-hours", type=float, default=2.0)
    args = parser.parse_args()
    root = args.root.resolve()
    if args.command == "plan":
        print(plan(root, args.scope, args.subset, args.seed, args.repeats,
                   args.fixed_depth, args.max_rows))
    elif args.command == "pilot":
        print(pilot(root, args.scope, args.subset, args.max_estimated_hours))
    elif args.command == "run":
        print(run(root, args.scope, args.subset))
    else:
        print(pgen_candidates(root, args.scope, args.max_sequences,
                              args.max_estimated_hours))


if __name__ == "__main__":
    cli()
