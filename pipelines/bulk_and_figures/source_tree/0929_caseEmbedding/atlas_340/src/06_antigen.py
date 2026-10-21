#!/usr/bin/env python3
"""Annotate frozen, contribution-qualified Home TRB groups with local antigen leads.

Only a PASS final candidate gate can enter the analysis. A full CDR3beta AA
match is a receptor annotation, never proof that this patient's T cell binds
the listed epitope. Similarity/motif leads are deliberately kept separate.
"""
from __future__ import annotations
from release_runtime import private_literal as _private_literal, release_path as _release_path

import argparse
from collections import defaultdict
import csv
import hashlib
import io
import json
import math
import os
from pathlib import Path
import random
import re
import tempfile
import zipfile

for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_var] = "1"

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.dataset as ds
import pyarrow.parquet as pq
from scipy.stats import fisher_exact


ROOT = Path(_release_path('path_99d8c1c6fde2a6b9'))
REFERENCES = Path(_release_path('path_b431b0be7d9625c5'))
OUT = ROOT / "antigen"
GATE_JSON = ROOT / "results/05_final_candidate_gate.json"
GATE_TSV = ROOT / "results/05_final_candidates.tsv"
MEMBERSHIP = ROOT / "groups/full_home/group_membership.parquet"
NODES = ROOT / "groups/tcrdist_all_nodes.parquet"
VALID_AA = re.compile(r"^[ACDEFGHIKLMNPQRSTVWY]+$")
TRUE_FIELDS = ("contribution_pass", "external_same_direction",
               "oof_gain_ci_lower_gt_zero", "outer_folds_favorable_ge4",
               "ablation_stable")
CLASSES = ("viral", "bacterial", "fungal", "parasitic", "tumor_associated",
           "autoimmune_or_self", "allergen", "pathogen_unspecified", "human_or_self")
EVIDENCE_COLS = ("method", "group_id", "member_aa", "member_v", "member_j",
                 "database", "source_file", "source_row", "reference_id", "pmid",
                 "ref_cdr3beta", "ref_v", "ref_j", "vj_check", "evidence_tier",
                 "epitope", "antigen", "source_organism", "antigen_class",
                 "class_rule", "assay_method", "mhc", "database_score",
                 "home_patient_count", "external_patient_count")
ENRICH_COLS = ("method", "group_id", "antigen_class", "n_member_aa_total",
               "n_member_aa_matched", "n_background_aa", "member_hit_aa",
               "background_hit_aa", "odds_ratio_haldane", "or_ci95_low",
               "or_ci95_high", "p_enrichment", "q_bh", "test_status",
               "matched_fraction", "background_per_member", "match_rule")
SUMMARY_COLS = ("method", "group_id", "n_member_aa", "n_member_aa_ambiguous_vj",
                "n_exact_match_aa", "n_vj_concordant_match_aa", "n_reference_records",
                "n_home_patients_with_exact_hit", "n_external_patients_with_exact_hit",
                "top_antigen_class_by_distinct_aa")


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + f".partial.{os.getpid()}")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    os.replace(temp, path)


def write_tsv(path: Path, rows: list[dict], columns: tuple[str, ...]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + f".partial.{os.getpid()}")
    with temp.open("w", newline="", encoding="utf8") as f:
        writer = csv.DictWriter(f, fieldnames=columns, delimiter="\t", extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temp, path)


def clean(value: object) -> str:
    text = "" if value is None else str(value).strip()
    return "" if text.upper() in ("NA", "N/A", "NONE", "NAN", "NULL") else text


def gene(value: object) -> str:
    return clean(value).split("*", 1)[0].upper()


def full_beta(value: object) -> str:
    aa = clean(value).upper()
    return aa if len(aa) >= 5 and aa.startswith("C") and aa.endswith(("F", "W")) and VALID_AA.fullmatch(aa) else ""


def vj_check(member_v: str, member_j: str, ref_v: str, ref_j: str) -> str:
    v, j = gene(ref_v), gene(ref_j)
    if v and v != gene(member_v) or j and j != gene(member_j):
        return "discordant"
    if v and j:
        return "concordant"
    return "partly_unreported"


def category(database: str, source: str, mcpas_category: str = "") -> tuple[str, str]:
    source = clean(source).lower()
    mcpas_category = clean(mcpas_category).lower()
    if database == "McPAS" and mcpas_category:
        mapped = {"cancer": "tumor_associated", "autoimmune": "autoimmune_or_self",
                  "allergy": "allergen"}
        if mcpas_category in mapped:
            return mapped[mcpas_category], "McPAS Category=" + mcpas_category
    terms = (
        ("viral", ("virus", "viral", "influenza", "hiv", "hcv", "hbv", "sars", "cmv", "ebv", "covid")),
        ("bacterial", ("bacter", "mycobacter", "escherich", "salmonella", "streptoc", "staphyloc", "listeria")),
        ("fungal", ("fung", "aspergill", "candida", "yeast")),
        ("parasitic", ("parasite", "plasmodium", "toxoplasm", "leishmania")),
    )
    for label, needles in terms:
        if any(x in source for x in needles):
            return label, "source_organism_keyword"
    if database == "McPAS" and mcpas_category == "pathogens":
        return "pathogen_unspecified", "McPAS Category=pathogens"
    if any(x in source for x in ("homo sapiens", "human", "homosapiens")):
        return "human_or_self", "human_source_organism"
    return "unclassified", "no_supported_class_rule"


def reference_records(refdir: Path, target: set[str]):
    """Stream source records; retain exact, complete human TRB AA only."""
    path = refdir / "vdjdb-2026-06-03.zip"
    with zipfile.ZipFile(path) as z:
        member = "vdjdb-2026-06-03/vdjdb_full.txt"
        with io.TextIOWrapper(z.open(member), encoding="utf8", errors="replace") as f:
            for row_no, r in enumerate(csv.DictReader(f, delimiter="\t"), 2):
                aa = full_beta(r.get("cdr3.beta"))
                if aa not in target or clean(r.get("species")).lower() not in ("homosapiens", "homo sapiens"):
                    continue
                cl, rule = category("VDJdb", r.get("antigen.species", ""))
                ident = clean(r.get("reference.id"))
                yield dict(database="VDJdb", source_file=f"{path}!{member}", source_row=row_no,
                           reference_id=ident, pmid=ident if re.fullmatch(r"PMID:\d+", ident) else "",
                           aa=aa, ref_v=clean(r.get("v.beta")), ref_j=clean(r.get("j.beta")),
                           epitope=clean(r.get("antigen.epitope")), antigen=clean(r.get("antigen.gene")),
                           source_organism=clean(r.get("antigen.species")), antigen_class=cl,
                           class_rule=rule, assay_method="; ".join(filter(None, (
                               clean(r.get("method.identification")), clean(r.get("method.verification")),
                               clean(r.get("method.sequencing"))))), mhc=clean(r.get("mhc.a")),
                           database_score=clean(r.get("vdjdb.score")))

    path = refdir / "receptor_full_v3.zip"
    with zipfile.ZipFile(path) as z:
        member = "tcr_full_v3.csv"
        with io.TextIOWrapper(z.open(member), encoding="utf8", errors="replace", newline="") as f:
            reader = csv.reader(f)
            next(reader)  # group header
            header = next(reader)
            expected = {0: "Group IRI", 4: "IEDB IRI", 6: "Name", 9: "Type",
                        12: "Type", 23: "CDR3 Curated", 25: "Junction Calculated",
                        43: "Type", 54: "CDR3 Curated", 56: "Junction Calculated"}
            if len(header) != 74 or any(header[k] != value for k, value in expected.items()):
                raise ValueError("IEDB receptor_full_v3 schema changed")
            for row_no, r in enumerate(reader, 3):
                if len(r) != 74:
                    raise ValueError(f"IEDB record width changed at row {row_no}")
                for start in (12, 43):
                    if clean(r[start]).lower() != "beta" or clean(r[start + 1]) != "http://purl.obolibrary.org/obo/NCBITaxon_9606":
                        continue
                    aa = full_beta(r[start + 13])
                    if not aa:
                        # A curated CDR3 is accepted only when already complete as deposited.
                        aa = full_beta(r[start + 11]) or full_beta(r[start + 12])
                    if aa not in target:
                        continue
                    cl, rule = category("IEDB", r[8])
                    yield dict(database="IEDB", source_file=f"{path}!{member}", source_row=row_no,
                               reference_id=clean(r[4]), pmid="", aa=aa,
                               ref_v=clean(r[start + 3]) or clean(r[start + 4]),
                               ref_j=clean(r[start + 7]) or clean(r[start + 8]),
                               epitope=clean(r[6]), antigen=clean(r[7]),
                               source_organism=clean(r[8]), antigen_class=cl,
                               class_rule=rule, assay_method=clean(r[9]), mhc=clean(r[11]),
                               database_score="")

    path = refdir / "McPAS-TCR.csv"
    with path.open(newline="", encoding="utf-8-sig", errors="replace") as f:
        for row_no, r in enumerate(csv.DictReader(f), 2):
            aa = full_beta(r.get("CDR3.beta.aa"))
            if aa not in target or clean(r.get("Species")).lower() != "human":
                continue
            cl, rule = category("McPAS", r.get("Pathology", ""), r.get("Category", ""))
            pmid = clean(r.get("PubMed.ID"))
            if pmid and pmid.isdigit():
                pmid = "PMID:" + pmid
            yield dict(database="McPAS", source_file=str(path), source_row=row_no,
                       reference_id=pmid, pmid=pmid, aa=aa,
                       ref_v=clean(r.get("TRBV")), ref_j=clean(r.get("TRBJ")),
                       epitope=clean(r.get("Epitope.peptide")), antigen=clean(r.get("Antigen.protein")),
                       source_organism=clean(r.get("Pathology")), antigen_class=cl,
                       class_rule=rule, assay_method=clean(r.get("Antigen.identification.method")),
                       mhc=clean(r.get("MHC")), database_score="")


def gate(root: Path) -> tuple[str, list[dict], dict]:
    gp, tp = root / "results/05_final_candidate_gate.json", root / "results/05_final_candidates.tsv"
    if not gp.exists() and not tp.exists():
        return "STOP_gate_pending", [], {"reason": "Final contribution gate has not been produced"}
    if not gp.exists():
        raise ValueError("Candidate TSV exists without final gate JSON")
    meta = json.loads(gp.read_text())
    if meta.get("status") != "PASS":
        if tp.exists() and meta.get("candidate_tsv_sha256") and meta["candidate_tsv_sha256"] != sha(tp):
            raise ValueError("Stopped final gate candidate TSV hash mismatch")
        return "STOP_gate_not_pass", [], {"reason": f"Final gate status={meta.get('status')}",
                                          "gate_sha256": sha(gp),
                                          "candidate_tsv_sha256": sha(tp) if tp.exists() else None}
    if not tp.exists():
        raise ValueError("PASS final gate lacks candidate TSV")
    if meta.get("candidate_tsv_sha256") != sha(tp):
        raise ValueError("Final gate candidate TSV hash mismatch")
    with tp.open(newline="", encoding="utf8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        needed = {"method", "group_id", "home_q"} | set(TRUE_FIELDS)
        if not needed <= set(reader.fieldnames or []):
            raise ValueError("Final candidate TSV misses required gate fields")
        rows = list(reader)
    if len(rows) != int(meta.get("n_candidates", -1)):
        raise ValueError("Final gate candidate count mismatch")
    if not rows:
        return "NO_QUALIFIED_CANDIDATES", [], {"reason": "Final PASS gate has zero candidates",
                                                 "gate_sha256": sha(gp), "candidate_tsv_sha256": sha(tp)}
    for r in rows:
        if not r["method"] or not r["group_id"] or not (0 <= float(r["home_q"]) < .05):
            raise ValueError("Final gate contains invalid or q-ineligible candidate")
        if any(clean(r[k]).lower() not in ("true", "1", "yes") for k in TRUE_FIELDS):
            raise ValueError("Final gate contains candidate failing a contribution requirement")
    if len({r["group_id"] for r in rows}) != len(rows):
        raise ValueError("Duplicate final candidate group")
    expected = {"source_home_candidates_sha256": root / "groups/frozen_home_candidates.tsv",
                "source_oof_receipt_sha256": root / "prediction/05_receipt.json",
                "source_external_receipt_sha256": root / "groups/full_home/external_direction.json",
                "source_ablation_receipt_sha256": root / "results/05_ablation_receipt.json",
                "source_sensitivity_receipt_sha256": root / "results/05_sensitivity_receipt.json"}
    overrides = meta.get("source_paths", {})
    for field, default_path in expected.items():
        provided = meta.get(field)
        if not isinstance(provided, str) or not re.fullmatch(r"[0-9a-f]{64}", provided):
            raise ValueError(f"Final gate lacks valid {field}")
        source = Path(overrides.get(field, str(default_path)))
        if not source.is_absolute():
            source = root / source
        if not source.resolve().is_relative_to(root.resolve()):
            raise ValueError(f"Final gate source escapes atlas directory: {field}")
        if not source.is_file() or sha(source) != provided:
            raise ValueError(f"Final gate upstream hash mismatch: {field}: {source}")
    return "PASS", rows, {"gate_sha256": sha(gp), "candidate_tsv_sha256": sha(tp),
                          "source_hashes": {k: meta[k] for k in expected}}


def members(root: Path, candidates: list[dict]) -> tuple[pd.DataFrame, dict]:
    path = root / "groups/full_home/group_membership.parquet"
    receipt = root / "groups/full_home/integrated_projection_receipt.json"
    if not path.exists() or not receipt.exists():
        raise ValueError("Frozen Home membership or receipt missing")
    sealed = json.loads(receipt.read_text())
    if sealed.get("status") != "complete_home_scope" or sealed.get("membership_sha256") != sha(path):
        raise ValueError("Frozen Home membership receipt/hash invalid")
    gids = [c["group_id"] for c in candidates]
    table = ds.dataset(path, format="parquet").to_table(
        columns=["method", "group_id", "aa", "v_gene", "j_gene"],
        filter=ds.field("group_id").isin(gids))
    frame = table.to_pandas().drop_duplicates()
    if set(frame.group_id) != set(gids) or not frame.method.eq(frame.group_id.map(
        {c["group_id"]: c["method"] for c in candidates})).all():
        raise ValueError("Final candidates do not map to frozen Home groups/methods")
    if frame[["aa", "v_gene", "j_gene"]].isna().any().any():
        raise ValueError("Frozen candidate membership contains null key")
    return frame, {"membership_sha256": sha(path), "membership_receipt_sha256": sha(receipt)}


def select_background(nodes: Path, member: pd.DataFrame, ratio: int = 5) -> tuple[pd.DataFrame, int]:
    """Patient-label-blind reservoir per exact V/J/AA-length stratum in Home."""
    unique = member.sort_values(["aa", "v_gene", "j_gene"]).drop_duplicates("aa")
    counts = unique.assign(length=unique.aa.str.len()).groupby(["v_gene", "j_gene", "length"]).size()
    caps = {k: ratio * int(n) for k, n in counts.items()}
    selected: dict[tuple, list[tuple]] = {k: [] for k in caps}
    observed = defaultdict(int)
    rng = random.Random(20260930)
    excluded = set(member.aa)
    parquet = pq.ParquetFile(nodes)
    required = {"aa", "v_gene", "j_gene", "cohort_mask"}
    if not required <= set(parquet.schema_arrow.names):
        raise ValueError("Home node universe has wrong schema")
    for batch in parquet.iter_batches(batch_size=100_000, columns=list(required)):
        frame = batch.to_pandas()
        frame = frame.loc[frame.cohort_mask.astype(int).mod(2).eq(1)]
        if frame.empty:
            continue
        frame = frame.loc[~frame.aa.isin(excluded)].copy()
        if frame.empty:
            continue
        frame["length"] = frame.aa.str.len()
        frame["stratum"] = list(zip(frame.v_gene, frame.j_gene, frame.length))
        frame = frame.loc[frame.stratum.isin(caps)]
        for aa, v, j, key in frame[["aa", "v_gene", "j_gene", "stratum"]].itertuples(index=False, name=None):
            observed[key] += 1
            bucket = selected[key]
            candidate = (aa, v, j)
            if len(bucket) < caps[key]:
                bucket.append(candidate)
            else:
                i = rng.randrange(observed[key])
                if i < caps[key]:
                    bucket[i] = candidate
    bg = pd.DataFrame([v for bucket in selected.values() for v in bucket],
                      columns=["aa", "v_gene", "j_gene"]).drop_duplicates("aa")
    return bg, sum(bool(selected[k]) for k in caps)


def patient_counts(root: Path, member: pd.DataFrame, hit_aa: set[str]) -> dict[tuple, set[str]]:
    wanted = set(zip(member.aa, member.v_gene, member.j_gene))
    target = pa.array(sorted(hit_aa))
    counts: dict[tuple, set[str]] = defaultdict(set)
    if not hit_aa:
        return counts
    for path in sorted((root / "data/clone_detail").glob("*.parquet")):
        pf = pq.ParquetFile(path)
        for batch in pf.iter_batches(batch_size=100_000,
                                     columns=["aa", "v_gene", "j_gene", "patient_key", "cohort", "qc_pass"]):
            table = pa.Table.from_batches([batch])
            selected = table.filter(pc.and_(table["qc_pass"], pc.is_in(table["aa"], value_set=target)))
            for row in selected.to_pylist():
                key = (row["aa"], row["v_gene"], row["j_gene"])
                if key in wanted:
                    counts[(row["cohort"],) + key].add(row["patient_key"])
    return counts


def bh(pvals: list[float]) -> list[float]:
    if not pvals:
        return []
    p = np.asarray(pvals, dtype=float)
    if not np.isfinite(p).all() or ((p < 0) | (p > 1)).any():
        raise ValueError("Invalid enrichment P value")
    order = np.argsort(p, kind="stable")
    out = np.empty(len(p))
    out[order] = np.minimum.accumulate((p[order] * len(p) / np.arange(1, len(p)+1))[::-1])[::-1].clip(0, 1)
    return out.tolist()


def run(root: Path = ROOT, refdir: Path = REFERENCES) -> dict:
    out = root / "antigen"
    state, candidates, provenance = gate(root)
    if state != "PASS":
        for name, cols in (("evidence_exact.tsv", EVIDENCE_COLS), ("enrichment.tsv", ENRICH_COLS),
                           ("group_summary.tsv", SUMMARY_COLS), ("evidence_similarity.tsv", EVIDENCE_COLS)):
            write_tsv(out / name, [], cols)
        receipt = {"status": state, **provenance, "n_groups": 0, "n_exact_records": 0,
                   "interpretation": "No antigen category or binding claim is made without the frozen contribution gate."}
        write_json(out / "receipt.json", receipt)
        return receipt
    member, source = members(root, candidates)
    if not (root / "groups/tcrdist_all_nodes.parquet").exists():
        raise ValueError("Label-blind Home node universe missing")
    refpaths = [refdir / x for x in ("vdjdb-2026-06-03.zip", "receptor_full_v3.zip", "McPAS-TCR.csv")]
    refhash = {p.name: sha(p) for p in refpaths}
    by_aa: dict[str, list[dict]] = defaultdict(list)
    backgrounds: dict[str, pd.DataFrame] = {}
    matched_strata: dict[str, int] = {}
    target = set(member.aa)
    for c in candidates:
        g = member.loc[member.group_id.eq(c["group_id"])]
        bg, n_strata = select_background(root / "groups/tcrdist_all_nodes.parquet", g)
        backgrounds[c["group_id"]] = bg
        matched_strata[c["group_id"]] = n_strata
        target.update(bg.aa)
    for record in reference_records(refdir, target):
        by_aa[record["aa"]].append(record)
    matched_aa = set(member.aa) & set(by_aa)
    patient_map = patient_counts(root, member, matched_aa)
    evidence: list[dict] = []
    summary: list[dict] = []
    enrich: list[dict] = []
    tested: list[int] = []
    pvals: list[float] = []
    for c in candidates:
        gid, method = c["group_id"], c["method"]
        g = member.loc[member.group_id.eq(gid)].sort_values(["aa", "v_gene", "j_gene"])
        uniq = g.drop_duplicates("aa")
        bg = backgrounds[gid]
        hits = defaultdict(set)
        concordant = set()
        home_people, external_people = set(), set()
        for row in g.itertuples(index=False):
            for ref in by_aa.get(row.aa, []):
                check = vj_check(row.v_gene, row.j_gene, ref["ref_v"], ref["ref_j"])
                home = patient_map.get(("home", row.aa, row.v_gene, row.j_gene), set())
                external = patient_map.get(("external", row.aa, row.v_gene, row.j_gene), set())
                if check != "discordant":
                    home_people.update(home); external_people.update(external)
                entry = dict(method=method, group_id=gid, member_aa=row.aa,
                             member_v=row.v_gene, member_j=row.j_gene,
                             ref_cdr3beta=ref["aa"], vj_check=check,
                             evidence_tier="full_AA_VJ_concordant" if check == "concordant" else
                             "full_AA_VJ_unreported" if check == "partly_unreported" else "full_AA_VJ_discordant",
                             home_patient_count=len(home), external_patient_count=len(external))
                entry.update({k: v for k, v in ref.items() if k != "aa"})
                evidence.append(entry)
                if check == "concordant":
                    concordant.add(row.aa)
                if check != "discordant" and ref["antigen_class"] in CLASSES:
                    hits[row.aa].add(ref["antigen_class"])
        summary.append(dict(method=method, group_id=gid, n_member_aa=len(uniq),
                            n_member_aa_ambiguous_vj=int(g.aa.value_counts().gt(1).sum()),
                            n_exact_match_aa=len(set(g.aa) & set(by_aa)),
                            n_vj_concordant_match_aa=len(concordant),
                            n_reference_records=sum(1 for e in evidence if e["group_id"] == gid),
                            n_home_patients_with_exact_hit=len(home_people),
                            n_external_patients_with_exact_hit=len(external_people),
                            top_antigen_class_by_distinct_aa=(max(CLASSES, key=lambda label: (
                                sum(label in h for h in hits.values()), -CLASSES.index(label)))
                                if hits else "")))
        # Fisher contrasts only AA from strata where a same V/J/length Home background exists.
        strata_bg = set(zip(bg.v_gene, bg.j_gene, bg.aa.str.len()))
        test_member = uniq.loc[[((v, j, len(aa)) in strata_bg)
                                for aa, v, j in uniq[["aa", "v_gene", "j_gene"]].itertuples(index=False, name=None)]]
        def labels(frame: pd.DataFrame, cl: str) -> int:
            return sum(any(ref["antigen_class"] == cl and
                           vj_check(v, j, ref["ref_v"], ref["ref_j"]) != "discordant"
                           for ref in by_aa.get(aa, []))
                       for aa, v, j in frame[["aa", "v_gene", "j_gene"]].itertuples(index=False, name=None))
        for cl in CLASSES:
            a, b = labels(test_member, cl), labels(bg, cl)
            n1, n0 = len(test_member), len(bg)
            status = "tested" if n1 and n0 >= n1 else "insufficient_exact_stratum_background"
            row = dict(method=method, group_id=gid, antigen_class=cl,
                       n_member_aa_total=len(uniq), n_member_aa_matched=n1,
                       n_background_aa=n0, member_hit_aa=a, background_hit_aa=b,
                       matched_fraction=n1 / len(uniq) if len(uniq) else 0,
                       background_per_member=n0 / n1 if n1 else 0,
                       match_rule="Home only; exact V/J stem and AA length; 5:1 reservoir; distinct AA",
                       test_status=status)
            if status == "tested":
                _, p = fisher_exact([[a, n1-a], [b, n0-b]], alternative="greater")
                aa, bb, cc, dd = a+.5, n1-a+.5, b+.5, n0-b+.5
                log_or = math.log(aa*dd/(bb*cc))
                se = math.sqrt(1/aa+1/bb+1/cc+1/dd)
                row.update(odds_ratio_haldane=math.exp(log_or),
                           or_ci95_low=math.exp(log_or-1.96*se),
                           or_ci95_high=math.exp(log_or+1.96*se), p_enrichment=p)
                tested.append(len(enrich)); pvals.append(p)
            enrich.append(row)
    qvals = bh(pvals)
    for i, q in zip(tested, qvals):
        enrich[i]["q_bh"] = q
    write_tsv(out / "evidence_exact.tsv", evidence, EVIDENCE_COLS)
    write_tsv(out / "enrichment.tsv", enrich, ENRICH_COLS)
    write_tsv(out / "group_summary.tsv", summary, SUMMARY_COLS)
    write_tsv(out / "evidence_similarity.tsv", [], EVIDENCE_COLS)
    receipt = dict(status="complete", gate=provenance, sources=source,
                   reference_sha256=refhash, node_universe_sha256=sha(root / "groups/tcrdist_all_nodes.parquet"),
                   n_groups=len(candidates), n_exact_records=len(evidence),
                   n_exact_distinct_member_aa=len(matched_aa), n_tests=len(pvals),
                   matched_strata=matched_strata,
                   outputs_sha256={name: sha(out / name) for name in
                                   ("evidence_exact.tsv", "enrichment.tsv", "group_summary.tsv", "evidence_similarity.tsv")},
                   reference_match_rule="Human TCR beta, full deposited AA exact match; C...F/W anchoring; no trimming or motif imputation",
                   limitations=["Exact public beta AA match is an antigen-response clue, not binding in this patient",
                                "No paired alpha chain, patient HLA, or functional assay",
                                "IEDB receptor export does not directly provide a verified PMID",
                                "Source-record duplicates do not multiply AA counts in enrichment",
                                "Similarity/motif evidence was not computed and is excluded from enrichment"])
    write_json(out / "receipt.json", receipt)
    return receipt


def self_test() -> None:
    assert full_beta(_private_literal('private_afe396ab2995c4e1')) == _private_literal('private_afe396ab2995c4e1')
    assert not full_beta(_private_literal('private_fca63371fa6f464f'))
    assert vj_check("TRBV19", "TRBJ2-1", "TRBV19*01", "TRBJ2-1*01") == "concordant"
    assert vj_check("TRBV19", "TRBJ2-1", "TRBV6-5*01", "") == "discordant"
    assert np.allclose(bh([.01, .03, .4]), [0.03, 0.045, 0.4])
    assert category("McPAS", "", "Cancer")[0] == "tumor_associated"
    with tempfile.TemporaryDirectory() as name:
        root = Path(name)
        status, rows, _ = gate(root)
        assert status == "STOP_gate_pending" and not rows
        (root / "results").mkdir()
        write_tsv(root / "results/05_final_candidates.tsv", [],
                  ("method", "group_id", "home_q") + TRUE_FIELDS)
        write_json(root / "results/05_final_candidate_gate.json",
                   {"status": "PASS", "n_candidates": 0,
                    "candidate_tsv_sha256": sha(root / "results/05_final_candidates.tsv")})
        status, rows, _ = gate(root)
        assert status == "NO_QUALIFIED_CANDIDATES" and not rows
        sources = {}
        for field in ("source_home_candidates_sha256", "source_oof_receipt_sha256",
                      "source_external_receipt_sha256", "source_ablation_receipt_sha256",
                      "source_sensitivity_receipt_sha256"):
            path = root / (field + ".txt")
            path.write_text(field + "\n")
            sources[field] = path
        valid = {"method": "tcrdist3", "group_id": "test_group", "home_q": "0.01",
                 **{field: "true" for field in TRUE_FIELDS}}
        write_tsv(root / "results/05_final_candidates.tsv", [valid],
                  ("method", "group_id", "home_q") + TRUE_FIELDS)
        write_json(root / "results/05_final_candidate_gate.json",
                   {"status": "PASS", "n_candidates": 1,
                    "candidate_tsv_sha256": sha(root / "results/05_final_candidates.tsv"),
                    "source_paths": {key: str(path) for key, path in sources.items()},
                    **{key: sha(path) for key, path in sources.items()}})
        status, rows, _ = gate(root)
        assert status == "PASS" and rows[0]["group_id"] == "test_group"
        ablation_source = sources["source_ablation_receipt_sha256"]
        ablation_source.write_text("changed after gate\n")
        try:
            gate(root)
        except ValueError as exc:
            assert "source_ablation_receipt_sha256" in str(exc)
        else:
            raise AssertionError("Changed ablation receipt passed the final gate")
        ablation_source.write_text("source_ablation_receipt_sha256\n")
        sensitivity_source = sources["source_sensitivity_receipt_sha256"]
        sensitivity_source.write_text("changed after gate\n")
        try:
            gate(root)
        except ValueError as exc:
            assert "source_sensitivity_receipt_sha256" in str(exc)
        else:
            raise AssertionError("Changed sensitivity receipt passed the final gate")
        sensitivity_source.write_text("source_sensitivity_receipt_sha256\n")
        valid["ablation_stable"] = "false"
        write_tsv(root / "results/05_final_candidates.tsv", [valid],
                  ("method", "group_id", "home_q") + TRUE_FIELDS)
        changed = json.loads((root / "results/05_final_candidate_gate.json").read_text())
        changed["candidate_tsv_sha256"] = sha(root / "results/05_final_candidates.tsv")
        write_json(root / "results/05_final_candidate_gate.json", changed)
        try:
            gate(root)
        except ValueError as exc:
            assert "failing a contribution requirement" in str(exc)
        else:
            raise AssertionError("A failing candidate passed the final gate")
        node_path = root / "nodes.parquet"
        pq.write_table(pa.table({"aa": [_private_literal('private_25632297a2ea2fd3'), _private_literal('private_b61a2a99e39b79fb'), _private_literal('private_9bf3cc429e0e5a7d')],
                                 "v_gene": ["TRBV19"] * 3,
                                 "j_gene": ["TRBJ2-1"] * 3,
                                 "cohort_mask": [1, 1, 2]}), node_path)
        sample, matched = select_background(node_path, pd.DataFrame([
            {"aa": _private_literal('private_25632297a2ea2fd3'), "v_gene": "TRBV19", "j_gene": "TRBJ2-1"}]))
        assert matched == 1 and sample.aa.tolist() == [_private_literal('private_b61a2a99e39b79fb')]
    print("06_antigen synthetic gate/AA/VJ/class/BH checks PASS", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test()
    else:
        print(json.dumps(run(), indent=2, ensure_ascii=False), flush=True)
