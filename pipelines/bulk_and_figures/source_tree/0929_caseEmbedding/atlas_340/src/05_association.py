#!/usr/bin/env python3
"""Frozen Home TRB-family universe and recorded-N patient-level association.

The freeze step uses Home identities and known-label membership only. N values
are parsed only after the eligible family universe and its source hashes are
written. External N values are never parsed by this program.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import warnings
from pathlib import Path

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import numpy as np
import pandas as pd
from scipy.stats import chi2, fisher_exact
import statsmodels.api as sm
from statsmodels.tools.sm_exceptions import PerfectSeparationWarning


HERE = Path(__file__).resolve().parent.parent
SCOPE = HERE / "groups" / "full_home"
LOCK = HERE / "audit" / "05_association_universe_lock.json"
ELIGIBLE = SCOPE / "eligible_groups_recorded_home.tsv"
RESULT = SCOPE / "association_recorded_home.tsv"
NOMINAL = SCOPE / "exploratory_nominal_p_groups.tsv"
CANDIDATES = HERE / "groups" / "frozen_home_candidates.tsv"
CANDIDATE_RECEIPT = HERE / "groups" / "frozen_home_candidates.json"
RECEIPT = HERE / "audit" / "05_association_receipt.json"
MIN_CARRIERS = 10


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, sort_keys=True, indent=2) + "\n")
    tmp.replace(path)


def atomic_tsv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(tmp, sep="\t", index=False, float_format="%.12g")
    tmp.replace(path)


def home_ids_from_manifest(path: Path) -> set[str]:
    out: set[str] = set()
    with path.open(newline="") as handle:
        header = handle.readline().rstrip("\r\n").split("\t")
        assert header[:3] == ["cohort", "sample_id", "patient_key"]
        for line in handle:
            if line.startswith("home\t"):
                fields = line.split("\t", 3)
                if len(fields) < 3 or not fields[2]:
                    raise ValueError("Home manifest row lacks patient key")
                out.add(fields[2])
    return out


def known_home_ids_from_folds(path: Path) -> set[str]:
    out: set[str] = set()
    with path.open(newline="") as handle:
        assert handle.readline().startswith("patient_key\t")
        for line in handle:
            key = line.split("\t", 1)[0]
            if not key:
                raise ValueError("Blank fold patient key")
            out.add(key)
    return out


def input_paths() -> dict[str, Path]:
    return {
        "analysis_spec": HERE / "config" / "analysis_spec.json",
        "manifest": HERE / "config" / "cohort_samples.tsv",
        "folds": HERE / "folds" / "folds.tsv",
        "membership": SCOPE / "group_membership.parquet",
        "burden": SCOPE / "patient_group_burden.parquet",
        "burden_receipt": SCOPE / "burden_receipt.json",
    }


def read_verified_burden() -> tuple[pd.DataFrame, dict, dict[str, str], set[str], set[str]]:
    paths = input_paths()
    hashes = {name: sha256(path) for name, path in paths.items()}
    spec = json.loads(paths["analysis_spec"].read_text())["association"]
    if spec["label_blind_eligibility"] != (
        "at least 10 N-known Home carriers and at least 10 noncarriers"
    ):
        raise ValueError("Association eligibility specification changed")
    receipt = json.loads(paths["burden_receipt"].read_text())
    burden = pd.read_parquet(paths["burden"])
    member = pd.read_parquet(paths["membership"], columns=["method", "group_id"])
    if receipt.get("status") != "complete" or receipt.get("scope") != "full_home" or (
        receipt.get("target_cohort") != "home"
    ):
        raise ValueError("Full Home burden receipt is incomplete")
    if receipt.get("patient_group_burden_sha256") != hashes["burden"] or (
        receipt.get("group_membership_sha256") != hashes["membership"]
    ):
        raise ValueError("Burden or membership hash differs from receipt")
    home_ids = home_ids_from_manifest(paths["manifest"])
    known_ids = known_home_ids_from_folds(paths["folds"])
    if len(home_ids) != 137 or len(known_ids) != 126 or not known_ids <= home_ids:
        raise ValueError("Home patient identity count/set differs from locked cohort")
    required = {
        "patient_key", "group_id", "group_reads", "group_unique_aa",
        "total_reads", "total_unique_aa", "presence",
        "unique_aa_fraction", "read_fraction",
    }
    if not required <= set(burden) or burden[["patient_key", "group_id"]].duplicated().any():
        raise ValueError("Burden columns or patient/group uniqueness invalid")
    groups = set(burden.group_id.astype(str))
    if len(burden) != len(home_ids) * len(groups) or (
        set(burden.patient_key.astype(str)) != home_ids
    ) or len(groups) != receipt.get("n_groups") or len(burden) != receipt.get("n_rows"):
        raise ValueError("Full Home burden grid is incomplete")
    if len(member) == 0 or member[["method", "group_id"]].drop_duplicates().groupby(
        "group_id"
    ).method.nunique().gt(1).any():
        raise ValueError("Membership method mapping is ambiguous")
    method_map = member.drop_duplicates(["group_id", "method"]).set_index("group_id")["method"]
    if not groups <= set(method_map.index.astype(str)):
        raise ValueError("Burden groups missing from membership")
    burden["group_id"] = burden.group_id.astype(str)
    burden["patient_key"] = burden.patient_key.astype(str)
    return burden, method_map.to_dict(), hashes, home_ids, known_ids


def expected_lock_payload(
    burden: pd.DataFrame, methods: dict[str, str], hashes: dict[str, str],
    home_ids: set[str], known_ids: set[str],
) -> tuple[dict, pd.DataFrame]:
    known = burden.loc[burden.patient_key.isin(known_ids)]
    counts = known.groupby("group_id", sort=True).presence.sum().astype(int)
    eligible = counts.loc[counts.ge(MIN_CARRIERS) & counts.le(len(known_ids) - MIN_CARRIERS)]
    table = pd.DataFrame({
        "method": [methods[g] for g in eligible.index],
        "group_id": eligible.index.astype(str),
        "carriers_known_home": eligible.to_numpy(),
        "noncarriers_known_home": len(known_ids) - eligible.to_numpy(),
    }).sort_values(["method", "group_id"]).reset_index(drop=True)
    payload = {
        "status": "frozen_before_home_N_values",
        "scope": "full_home",
        "eligibility_rule": ">=10 carriers and >=10 noncarriers among 126 N-known Home patients",
        "n_home_patients": len(home_ids),
        "n_known_home_patients": len(known_ids),
        "n_groups_burden": int(burden.group_id.nunique()),
        "n_eligible_groups": len(table),
        "eligible_by_method": table.method.value_counts().sort_index().to_dict(),
        "input_sha256": hashes,
        "known_home_ids_sha256": hashlib.sha256(
            ("\n".join(sorted(known_ids)) + "\n").encode()
        ).hexdigest(),
        "external_N_values_parsed": False,
    }
    return payload, table


def freeze_universe() -> tuple[pd.DataFrame, pd.DataFrame, set[str]]:
    burden, methods, hashes, home_ids, known_ids = read_verified_burden()
    payload, table = expected_lock_payload(burden, methods, hashes, home_ids, known_ids)
    if LOCK.exists() or ELIGIBLE.exists():
        if not LOCK.exists() or not ELIGIBLE.exists():
            raise ValueError("Partial prior universe lock; inspect before retrying")
        old = json.loads(LOCK.read_text())
        if old != {**payload, "eligible_tsv_sha256": sha256(ELIGIBLE)}:
            raise ValueError("Existing universe lock differs from current sources")
        observed = pd.read_csv(ELIGIBLE, sep="\t")
        if not observed.equals(table):
            raise ValueError("Existing eligible universe differs from current source")
    else:
        if RESULT.exists() or RECEIPT.exists():
            raise ValueError("Association output already exists without universe lock")
        atomic_tsv(ELIGIBLE, table)
        payload["eligible_tsv_sha256"] = sha256(ELIGIBLE)
        atomic_json(LOCK, payload)
    return burden, table, known_ids


def read_home_clinical(known_ids: set[str]) -> pd.DataFrame:
    path = HERE / "config" / "cohort_samples.tsv"
    with path.open(newline="") as handle:
        header = handle.readline()
        home_lines = [line for line in handle if line.startswith("home\t")]
    reader = csv.DictReader([header] + home_lines, delimiter="\t")
    rows = []
    columns = (
        "patient_key", "recorded_n_binary", "age_years", "sex",
        "neoadjuvant", "home_batch",
    )
    for row in reader:
        rows.append({key: row[key] for key in columns})
    clinical = pd.DataFrame(rows)
    for key, sub in clinical.groupby("patient_key", sort=False):
        if len(sub.drop_duplicates()) != 1:
            raise ValueError("Clinical covariates disagree across same-patient libraries: " + key)
    clinical = clinical.drop_duplicates("patient_key")
    clinical = clinical.loc[clinical.patient_key.isin(known_ids)].copy()
    if len(clinical) != len(known_ids) or set(clinical.patient_key) != known_ids:
        raise ValueError("Home clinical data do not match frozen N-known IDs")
    if not clinical.recorded_n_binary.isin(["0", "1"]).all():
        raise ValueError("Known Home patient has invalid N label")
    clinical["y"] = clinical.recorded_n_binary.astype(int)
    if clinical.y.value_counts().to_dict() != {0: 70, 1: 56}:
        raise ValueError("Home recorded N class counts differ from manifest")
    clinical["age_years"] = pd.to_numeric(clinical.age_years, errors="raise")
    if clinical.age_years.isna().any() or not clinical.sex.isin(["m", "f"]).all() or (
        not clinical.neoadjuvant.isin(["yes", "no"]).all()
    ) or not clinical.home_batch.isin(["H2498B", "H2498C", "H2498D"]).all():
        raise ValueError("Missing or unexpected Home adjustment covariate")
    return clinical.set_index("patient_key").sort_index()


def bh(p_values: np.ndarray) -> np.ndarray:
    p = np.asarray(p_values, dtype=float)
    if not np.isfinite(p).all() or ((p < 0) | (p > 1)).any():
        raise ValueError("Invalid P values for BH")
    order = np.argsort(p, kind="stable")
    m = len(p)
    q = np.empty(m, dtype=float)
    q[order] = np.minimum.accumulate(
        (p[order] * m / np.arange(1, m + 1))[::-1]
    )[::-1].clip(0, 1)
    return q


def standardize(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    if not np.isfinite(x).all():
        raise ValueError("Nonfinite design feature")
    sd = x.std()
    return (x - x.mean()) / sd if sd > 0 else np.zeros_like(x)


def design(clinical: pd.DataFrame, totals: pd.DataFrame, include_neo: bool) -> np.ndarray:
    aligned = totals.loc[clinical.index]
    x = [
        np.ones(len(clinical)),
        standardize(clinical.age_years.to_numpy()),
        (clinical.sex == "m").to_numpy(dtype=float),
        standardize(np.log1p(aligned.total_reads.to_numpy(dtype=float))),
        standardize(np.log1p(aligned.total_unique_aa.to_numpy(dtype=float))),
        (clinical.home_batch == "H2498C").to_numpy(dtype=float),
        (clinical.home_batch == "H2498D").to_numpy(dtype=float),
    ]
    if include_neo:
        x.append((clinical.neoadjuvant == "yes").to_numpy(dtype=float))
    matrix = np.column_stack(x)
    if np.linalg.matrix_rank(matrix) < matrix.shape[1]:
        raise ValueError("Baseline design is rank deficient")
    return matrix


def fit_glm(y: np.ndarray, x: np.ndarray):
    with warnings.catch_warnings():
        warnings.simplefilter("error", PerfectSeparationWarning)
        warnings.simplefilter("ignore", RuntimeWarning)
        fit = sm.GLM(y, x, family=sm.families.Binomial()).fit(
            maxiter=100, tol=1e-8, disp=0
        )
    if not fit.converged or not np.isfinite(fit.params).all() or (
        not np.isfinite(fit.bse).all()
    ) or not np.isfinite(fit.llf):
        raise ValueError("GLM failed to converge to finite fit")
    return fit


def fit_effect(
    y: np.ndarray, x0: np.ndarray, null_fit, group_feature: np.ndarray,
) -> dict:
    feature = np.asarray(group_feature, dtype=float)
    out = {"p": 1.0, "beta": np.nan, "odds_ratio": np.nan, "status": "unstable"}
    if not np.isfinite(feature).all() or np.allclose(feature, feature[0]):
        out["status"] = "constant_feature"
        return out
    x1 = np.column_stack((x0, feature))
    if np.linalg.matrix_rank(x1) < x1.shape[1]:
        out["status"] = "rank_deficient"
        return out
    try:
        fit = fit_glm(y, x1)
        beta, se = float(fit.params[-1]), float(fit.bse[-1])
        lr = 2 * (float(fit.llf) - float(null_fit.llf))
        if abs(beta) > 10 or se > 10 or lr < -1e-6 or not np.isfinite(lr):
            out["status"] = "extreme_or_invalid_fit"
            return out
        out.update(p=float(chi2.sf(max(lr, 0), 1)), beta=beta,
                   odds_ratio=float(np.exp(beta)), status="ok")
    except (ValueError, FloatingPointError, OverflowError, np.linalg.LinAlgError):
        pass
    return out


def conservative_fit_effect(y, x0, null_fit, feature) -> dict:
    try:
        return fit_effect(y, x0, null_fit, feature)
    except Exception as exc:
        return {
            "p": 1.0, "beta": np.nan, "odds_ratio": np.nan,
            "status": "fit_error_" + type(exc).__name__,
        }


def totals_from_burden(burden: pd.DataFrame, ids: list[str]) -> pd.DataFrame:
    totals = burden[["patient_key", "total_reads", "total_unique_aa"]].drop_duplicates()
    if totals.patient_key.duplicated().any():
        raise ValueError("Inconsistent patient total depth across groups")
    totals = totals.set_index("patient_key").loc[ids]
    if totals.isna().any().any() or totals.le(0).any().any():
        raise ValueError("Nonpositive or missing patient repertoire totals")
    return totals


def associations(burden: pd.DataFrame, eligible: pd.DataFrame, known_ids: set[str]) -> None:
    if json.loads(LOCK.read_text()).get("status") != "frozen_before_home_N_values":
        raise ValueError("Universe lock missing")
    clinical = read_home_clinical(known_ids)
    ids = list(clinical.index)
    totals = totals_from_burden(burden, ids)
    y = clinical.y.to_numpy()
    x0 = design(clinical, totals, include_neo=True)
    null = fit_glm(y, x0)
    no_neo_ids = list(clinical.index[clinical.neoadjuvant.eq("no")])
    untreated = clinical.loc[no_neo_ids]
    x_untreated = design(untreated, totals, include_neo=False)
    null_untreated = fit_glm(untreated.y.to_numpy(), x_untreated)
    subset = burden.loc[
        burden.patient_key.isin(known_ids) & burden.group_id.isin(set(eligible.group_id)),
        ["patient_key", "group_id", "presence", "group_reads", "group_unique_aa",
         "unique_aa_fraction", "read_fraction"],
    ]
    if len(subset) != len(ids) * len(eligible):
        raise ValueError("Eligible patient by group grid incomplete")
    out = []
    for row in eligible.itertuples(index=False):
        g = subset.loc[subset.group_id.eq(row.group_id)].set_index("patient_key").loc[ids]
        presence = g.presence.to_numpy(dtype=float)
        primary = conservative_fit_effect(y, x0, null, presence)
        a = int(presence[y == 1].sum())
        b = int(presence[y == 0].sum())
        p_fisher = float(fisher_exact([[a, int((y == 1).sum()) - a],
                                       [b, int((y == 0).sum()) - b]])[1])
        unique_fraction = standardize(np.log1p(1e6 * g.unique_aa_fraction.to_numpy(dtype=float)))
        read_fraction = standardize(np.log1p(1e6 * g.read_fraction.to_numpy(dtype=float)))
        unique_fit = conservative_fit_effect(y, x0, null, unique_fraction)
        read_fit = conservative_fit_effect(y, x0, null, read_fraction)
        pos = np.asarray([ids.index(k) for k in no_neo_ids], dtype=int)
        untreated_fit = conservative_fit_effect(
            untreated.y.to_numpy(), x_untreated, null_untreated, presence[pos]
        )
        batch_delta = {}
        for batch in ("H2498B", "H2498C", "H2498D"):
            mask = clinical.home_batch.to_numpy() == batch
            plus = mask & (y == 1)
            zero = mask & (y == 0)
            batch_delta[batch] = float(presence[plus].mean() - presence[zero].mean()) if (
                plus.any() and zero.any()
            ) else np.nan
        out.append({
            "method": row.method, "group_id": row.group_id,
            "n_home_known": len(ids), "n_recorded_Nplus": int(y.sum()),
            "n_recorded_N0": int((1-y).sum()),
            "carriers_Nplus": a, "carriers_N0": b,
            "prevalence_Nplus": a / int(y.sum()),
            "prevalence_N0": b / int((1-y).sum()),
            "prevalence_difference": a / int(y.sum()) - b / int((1-y).sum()),
            "adjusted_p": primary["p"], "adjusted_beta": primary["beta"],
            "adjusted_odds_ratio": primary["odds_ratio"], "adjusted_status": primary["status"],
            "fisher_p_unadjusted": p_fisher,
            "untreated_p": untreated_fit["p"],
            "untreated_beta": untreated_fit["beta"],
            "untreated_status": untreated_fit["status"],
            "unique_fraction_p": unique_fit["p"],
            "unique_fraction_beta_per_sd": unique_fit["beta"],
            "unique_fraction_status": unique_fit["status"],
            "read_fraction_p": read_fit["p"],
            "read_fraction_beta_per_sd": read_fit["beta"],
            "read_fraction_status": read_fit["status"],
            "batch_B_delta": batch_delta["H2498B"],
            "batch_C_delta": batch_delta["H2498C"],
            "batch_D_delta": batch_delta["H2498D"],
        })
    table = pd.DataFrame(out) if out else pd.DataFrame(columns=[
        "method", "group_id", "n_home_known", "n_recorded_Nplus",
        "n_recorded_N0", "carriers_Nplus", "carriers_N0",
        "prevalence_Nplus", "prevalence_N0", "prevalence_difference",
        "adjusted_p", "adjusted_beta", "adjusted_odds_ratio",
        "adjusted_status", "fisher_p_unadjusted", "untreated_p",
        "untreated_beta", "untreated_status", "unique_fraction_p",
        "unique_fraction_beta_per_sd", "unique_fraction_status",
        "read_fraction_p", "read_fraction_beta_per_sd", "read_fraction_status",
        "batch_B_delta", "batch_C_delta", "batch_D_delta",
    ])
    table["adjusted_q_bh_pooled"] = bh(table.adjusted_p.to_numpy())
    for p_col, q_col in (
        ("untreated_p", "untreated_q_bh"),
        ("unique_fraction_p", "unique_fraction_q_bh"),
        ("read_fraction_p", "read_fraction_q_bh"),
    ):
        table[q_col] = bh(table[p_col].to_numpy())
    table["exploratory_nominal_p_lt_0_05"] = table.adjusted_p.lt(0.05)
    table["home_association_candidate"] = (
        table.adjusted_status.eq("ok") &
        table.adjusted_q_bh_pooled.lt(0.05) &
        table.adjusted_beta.gt(0) &
        table.prevalence_difference.gt(0)
    )
    table = table.sort_values(["adjusted_p", "method", "group_id"]).reset_index(drop=True)
    candidates = table.loc[table.home_association_candidate].copy()
    atomic_tsv(RESULT, table)
    atomic_tsv(NOMINAL, table.loc[table.exploratory_nominal_p_lt_0_05].copy())
    atomic_tsv(CANDIDATES, candidates)
    atomic_json(CANDIDATE_RECEIPT, {
        "status": "complete",
        "scope": "full_home",
        "candidate_sha256": sha256(CANDIDATES),
        "membership_sha256": sha256(SCOPE / "group_membership.parquet"),
        "source_association_sha256": sha256(RESULT),
        "eligibility_lock_sha256": sha256(LOCK),
        "n_candidates": len(candidates),
        "interpretation": "Home adjusted association candidates; final contribution gate not assessed",
    })
    receipt = {
        "status": "complete",
        "endpoint": "recorded N+; pN/ypN unverified",
        "scope": "full_home",
        "analysis_unit": "patient",
        "n_home_known": len(ids), "n_recorded_N0": int((1-y).sum()),
        "n_recorded_Nplus": int(y.sum()),
        "n_untreated": len(untreated),
        "n_tested_pooled": len(table),
        "n_primary_fit_ok": int(table.adjusted_status.eq("ok").sum()),
        "n_primary_fit_p1": int(table.adjusted_status.ne("ok").sum()),
        "n_nominal_exploratory": int(table.exploratory_nominal_p_lt_0_05.sum()),
        "n_home_association_candidates": len(candidates),
        "adjustment": ["age_years", "sex", "log1p(total_reads)",
                       "log1p(total_unique_aa)", "home_batch", "neoadjuvant"],
        "test": "1-df likelihood ratio against shared adjusted Binomial GLM",
        "multiple_testing": "pooled BH across all eligible methods/groups; failed fits p=1",
        "sensitivity_p_interpretation": "separate BH families; no q-based final candidacy from sensitivities",
        "eligibility_lock_sha256": sha256(LOCK),
        "eligible_tsv_sha256": sha256(ELIGIBLE),
        "association_tsv_sha256": sha256(RESULT),
        "nominal_tsv_sha256": sha256(NOMINAL),
        "candidate_tsv_sha256": sha256(CANDIDATES),
        "candidate_receipt_sha256": sha256(CANDIDATE_RECEIPT),
        "input_sha256": json.loads(LOCK.read_text())["input_sha256"],
        "external_N_values_parsed": False,
        "final_contribution_gate_not_assessed": True,
    }
    atomic_json(RECEIPT, receipt)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("freeze", "associate", "run"), default="run")
    args = parser.parse_args()
    burden, eligible, known_ids = freeze_universe()
    if args.mode == "freeze":
        print(json.dumps({"frozen_groups": len(eligible), "lock": str(LOCK)}))
        return
    if RESULT.exists() and RECEIPT.exists():
        receipt = json.loads(RECEIPT.read_text())
        if receipt.get("status") == "complete" and (
            receipt.get("eligibility_lock_sha256") == sha256(LOCK)
        ) and receipt.get("association_tsv_sha256") == sha256(RESULT) and (
            CANDIDATES.exists() and CANDIDATE_RECEIPT.exists()
        ) and receipt.get("candidate_tsv_sha256") == sha256(CANDIDATES) and (
            receipt.get("candidate_receipt_sha256") == sha256(CANDIDATE_RECEIPT)
        ):
            print(json.dumps({"status": "already_complete", "n_tested": receipt["n_tested_pooled"]}))
            return
    associations(burden, eligible, known_ids)
    print(json.dumps({"status": "complete", "n_tested": len(eligible),
                      "receipt": str(RECEIPT)}))


if __name__ == "__main__":
    main()
