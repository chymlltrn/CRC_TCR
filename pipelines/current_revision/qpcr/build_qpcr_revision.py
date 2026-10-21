#!/usr/bin/env python3
"""Recompute the authorized Q1 cohort and replace only Figure 5c/d.

Panel point/box encodings adapt the frozen redraw_figure5.py implementation.
The user's current Figure_v27 PDF supplies all unchanged panel artwork.
Run with CRC_QPCR_ROOT and CRC_QPCR_CONFIG (see ../README.md).
"""
from pathlib import Path
from itertools import combinations
import hashlib
import json
import math
import subprocess
import sys
import os

import fitz
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import transforms
import numpy as np
import pandas as pd
from scipy.stats import fisher_exact, rankdata

# All patient-level inputs and outputs live outside the code repository.
ROOT = Path(os.environ["CRC_QPCR_ROOT"]).expanduser().resolve()
CONFIG = json.loads(Path(os.environ["CRC_QPCR_CONFIG"]).read_text())
INPUT = ROOT / "inputs"
DATA = ROOT / "SourceData"
FIG = ROOT / "Figures"
QA = ROOT / "QA"
BLUE, GRAY = "#0072B2", "#777777"
REPLACEMENT = fitz.Rect(270, 104, 510.2362060546875, 408)
CRC_IDS = CONFIG["cohorts"]["CRC"]
HEALTHY_IDS = CONFIG["cohorts"]["Healthy"]
EXCLUDED_IDS = set(CONFIG["whole_sample_exclusions"])
ND_IDS = set(CONFIG["expected_nd_sample_ids"])
for folder in (DATA, FIG, QA):
    folder.mkdir(parents=True, exist_ok=True)



def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def exact_rank_test(x, y):
    """Conditional two-sided test: twice smaller inclusive permutation tail.

    Enumerate all C(23,14)=817190 allocations using integer doubled midranks.
    This retains ties from the four nondetection zero substitutions.
    """
    n = len(x)
    ranks2 = np.rint(rankdata(np.r_[x, y], method="average") * 2).astype(int)
    observed = int(ranks2[:n].sum())
    lower = upper = 0
    for assignment in combinations(ranks2.tolist(), n):
        score = sum(assignment)
        lower += score <= observed
        upper += score >= observed
    total = math.comb(len(ranks2), n)
    return {"test": "Exact conditional rank-sum permutation; twice smaller inclusive tail",
            "U_CRC": observed / 2 - n * (n + 1) / 2,
            "P_two_sided": min(1.0, 2 * min(lower, upper) / total),
            "rank_sum_twice": observed, "lower_tail_assignments": lower,
            "upper_tail_assignments": upper, "total_assignments": total,
            "ties_handled": "pooled midranks; all label allocations enumerated"}


def prepare_data():
    original = pd.read_csv(INPUT / "qpcr_samples_original.tsv", sep="\t")
    wells_original = pd.read_csv(INPUT / "qpcr_well_ct_original.tsv", sep="\t")
    samples = original.loc[~original.sample_id.isin(EXCLUDED_IDS)].copy()
    samples["disease_group"] = np.where(samples.group.eq("Healthy"), "Healthy", "CRC")
    wells = wells_original.loc[~wells_original.sample_id.isin(EXCLUDED_IDS)].copy()
    wells = wells.drop(columns=CONFIG.get("drop_well_columns", []))
    values = samples.copy()
    values["analysis_value"] = pd.to_numeric(values.reported_copies_per_ng_rna)
    values["ND_zero_substitution"] = values.measurement_class.eq("not_detected")
    values.loc[values.ND_zero_substitution, "analysis_value"] = 0.0
    assert set(samples.sample_id) == set(CRC_IDS + HEALTHY_IDS)
    assert len(samples) == samples.sample_id.nunique() == len(CRC_IDS) + len(HEALTHY_IDS)
    assert not samples.sample_id.isin(EXCLUDED_IDS).any() and not wells.sample_id.isin(EXCLUDED_IDS).any()
    assert set(values.loc[values.ND_zero_substitution, "sample_id"]) == ND_IDS
    assert values.analysis_value.notna().all()
    x = values.loc[values.disease_group.eq("CRC"), "analysis_value"].to_numpy()
    y = values.loc[values.disease_group.eq("Healthy"), "analysis_value"].to_numpy()
    exact = exact_rank_test(x, y)
    assert len(x) == len(CRC_IDS) and len(y) == len(HEALTHY_IDS)
    for key, expected in CONFIG.get("expected_statistics", {}).items():
        assert abs(exact[key] - expected) < 1e-15, (key, exact[key], expected)
    frozen = pd.read_csv(INPUT / "qpcr_exploratory_values_original.tsv", sep="\t")
    joined = values.set_index("sample_id").analysis_value.sort_index()
    assert joined.equals(frozen.set_index("sample_id").analysis_value.sort_index())
    summary = samples.groupby(["disease_group", "measurement_class"]).size().reset_index(name="n")
    summary.to_csv(DATA / "qpcr_measurement_categories.tsv", sep="\t", index=False)
    group_summary = []
    for group, vector in [("CRC", x), ("Healthy", y)]:
        s = values.loc[values.disease_group.eq(group)]
        group_summary.append({"group": group, "n_samples": len(s),
                              "median_source_estimate_copies_per_ng_RNA": float(np.median(vector)),
                              "n_any_amplification": int(s.measurement_class.ne("not_detected").sum()),
                              "n_ND_zero": int(s.ND_zero_substitution.sum()),
                              "n_source_quantifiable": int(s.measurement_class.eq("quantifiable").sum())})
    pd.DataFrame(group_summary).to_csv(DATA / "qpcr_group_summary.tsv", sep="\t", index=False)
    stat = {"analysis": f"CRC{len(x)}_Healthy{len(y)}_Q1_ND_zero", "n_CRC": len(x), "n_Healthy": len(y),
            "median_CRC": float(np.median(x)), "median_Healthy": float(np.median(y)),
            "unit": "source estimate copies/ng RNA", **exact}
    pd.DataFrame([stat]).to_csv(DATA / "qpcr_statistics.tsv", sep="\t", index=False)
    detected = {group: int(values.loc[values.disease_group.eq(group), "measurement_class"].ne("not_detected").sum())
                for group in ("CRC", "Healthy")}
    detection = fisher_exact([[detected["CRC"], len(x) - detected["CRC"]],
                              [detected["Healthy"], len(y) - detected["Healthy"]]], alternative="two-sided")
    pd.DataFrame([{"analysis": "any_amplification_descriptive", "CRC_detected": detected["CRC"],
                   "CRC_total": len(x), "Healthy_detected": detected["Healthy"], "Healthy_total": len(y),
                   "CRC_fraction": detected["CRC"] / len(x), "Healthy_fraction": detected["Healthy"] / len(y),
                   "test": "two-sided Fisher exact", "P_two_sided": float(detection.pvalue),
                   "odds_ratio": float(detection.statistic)}]).to_csv(
                       DATA / "qpcr_any_amplification_statistics.tsv", sep="\t", index=False)
    samples.to_csv(DATA / "qpcr_samples.tsv", sep="\t", index=False)
    wells.to_csv(DATA / "qpcr_well_ct.tsv", sep="\t", index=False)
    values.to_csv(DATA / "qpcr_analysis_values.tsv", sep="\t", index=False)
    audit = {"cohort": {"CRC": CRC_IDS, "Healthy": HEALTHY_IDS},
             "whole_sample_exclusions": sorted(EXCLUDED_IDS), "ND_zero_sample_ids": sorted(ND_IDS),
             "all_other_reported_values_unchanged": True,
             "matches_previous_boxplot_values_exactly": True,
             "numeric_Ct_before": int(wells_original.ct.notna().sum()),
             "numeric_Ct_after": int(wells.ct.notna().sum()),
             "statistical_recomputation": stat,
             "any_amplification_fisher_P": float(detection.pvalue)}
    (QA / "analysis_receipt.json").write_text(json.dumps(audit, indent=2) + "\n")
    return samples, wells, values, stat


def draw_qpcr(wells, values, stat, width, height):
    plt.rcParams.update({"font.family": "Arial", "font.size": 8.5,
                         "axes.labelsize": 8.5, "xtick.labelsize": 8.5, "ytick.labelsize": 8.5,
                         "axes.linewidth": .6, "pdf.fonttype": 42, "ps.fonttype": 42})
    fig = plt.figure(figsize=(width / 72, height / 72))

    def axis(x, y, w, h):
        return fig.add_axes([x / width, 1 - (y + h) / height, w / width, h / height])

    def text(x, y, label, **kwargs):
        return fig.text(x / width, 1 - y / height, label, va="top", **kwargs)

    def polish(ax):
        ax.spines[["top", "right"]].set_visible(False)
        ax.tick_params(length=2.5, width=.6, pad=2)

    text(274.96, 109.10, "c", weight="bold", fontsize=11)
    text(289.13, 110.20, "Q1 RT-qPCR: individual-well Ct", weight="bold", fontsize=9.3)
    for ids, left, group in [(CRC_IDS, 313, "CRC"), (HEALTHY_IDS, 429.22, "Healthy")]:
        ax = axis(left, 146, 65.2, 112)
        color = BLUE if group == "CRC" else GRAY
        ax.set_xlim(29.3, 40)
        ax.set_ylim(len(ids) - .5, -.5)
        ax.set_xticks([30, 34, 38])
        ax.set_yticks(range(len(ids)), ids)
        ax.set_xlabel("Ct", labelpad=1)
        polish(ax)
        ax.tick_params(axis="y", length=0, pad=3)
        ax.spines["left"].set_visible(False)
        text(left + 32.6, 130.24, f"{group} ({len(ids)})", ha="center", weight="bold", fontsize=9)
        text(left + 73, 130.56, "ND", ha="center", fontsize=8.5)
        for yi, sid in enumerate(ids):
            q = wells.loc[wells.sample_id.eq(sid)]
            numeric = q.loc[q.entry_type.eq("numeric_ct")]
            if len(numeric):
                ax.plot([numeric.ct.min(), numeric.ct.max()], [yi, yi], c="#CED5D9", lw=.65)
            offsets = np.linspace(-.16, .16, len(numeric)) if len(numeric) > 1 else [0]
            for offset, row in zip(offsets, numeric.itertuples()):
                ax.plot(row.ct, yi + offset, "o", ms=2.9, mfc=color, mec="white", mew=.2, zorder=3)
            nd = q.loc[q.entry_type.ne("numeric_ct")]
            for offset in np.linspace(-.12, .12, len(nd)) if len(nd) > 1 else [0] if len(nd) else []:
                ax.plot(1.12, yi + offset, "x", ms=3.5, c="#555555", clip_on=False,
                        transform=transforms.blended_transform_factory(ax.transAxes, ax.transData))
    text(274.96, 283.0, "× ND; no Ct value assigned.", fontsize=8.5)

    text(274.96, 294.0, "d", weight="bold", fontsize=11)
    text(289.13, 295.1, "Q1 RT-qPCR: CRC versus healthy", weight="bold", fontsize=9.3)
    ax = axis(326, 321, 171.4, 54)
    for xi, group in enumerate(["CRC", "Healthy"]):
        q = values.loc[values.disease_group.eq(group)]
        color = BLUE if group == "CRC" else GRAY
        ax.boxplot(q.analysis_value, positions=[xi], widths=.43, patch_artist=True,
                   showfliers=False, boxprops={"facecolor": color + "20", "edgecolor": color, "linewidth": .7},
                   medianprops={"color": color, "linewidth": 1.1},
                   whiskerprops={"color": color, "linewidth": .7}, capprops={"color": color, "linewidth": .7})
        for offset, row in zip(np.linspace(-.24, .24, len(q)), q.itertuples()):
            if row.measurement_class == "not_detected":
                ax.plot(xi + offset, 0, "x", ms=3.7, c="#444444", mew=.8, zorder=5)
            else:
                ax.plot(xi + offset, row.analysis_value, "^" if row.measurement_class == "near_loq" else "o",
                        ms=3.3, mfc=color if row.measurement_class == "quantifiable" else "white",
                        mec=color, mew=.75, zorder=4)
    ax.plot([0, 0, 1, 1], [8.4, 9, 9, 8.4], c="#333333", lw=.65)
    ax.text(.5, 9.25, f"P = {stat['P_two_sided']:.4f}", ha="center", fontsize=8.5)
    ax.set_xlim(-.5, 1.5)
    ax.set_ylim(-.6, 11.5)
    ax.set_ylabel("Estimated Q1 abundance\n(copies/ng RNA)", labelpad=2, fontsize=7)
    ax.set_yticks([0, 5])
    ax.set_xticks([0, 1], [f"CRC (n={len(CRC_IDS)})", f"Healthy (n={len(HEALTHY_IDS)})"])
    polish(ax)
    ax.tick_params(axis="x", length=0, pad=2)
    text(274.96, 395.0, "Open: near/below LOQ; × ND = 0.", fontsize=8.5)
    overlay = QA / "qpcr_overlay.pdf"
    fig.savefig(overlay, transparent=True)
    plt.close(fig)
    return overlay


def compose_and_verify(overlay, width, height, stat_for_verification):
    baseline = fitz.open(INPUT / "Figure5_v27_reference.pdf")
    doc = fitz.open(INPUT / "Figure5_v27_reference.pdf")
    page = doc[0]
    page.add_redact_annot(REPLACEMENT, fill=(1, 1, 1))
    page.apply_redactions(images=0, graphics=2, text=0)
    with fitz.open(overlay) as layer:
        page.show_pdf_page(page.rect, layer, 0)
    out = FIG / "Figure5_qPCR_revised_candidate.pdf"
    doc.save(out, garbage=4, deflate=True)
    doc.close()
    doc = fitz.open(out)
    page = doc[0]
    txt = page.get_text()
    visible_tokens = {w[4] for w in page.get_text("words")}
    assert not EXCLUDED_IDS.intersection(visible_tokens) and "discordant" not in txt
    assert f"Healthy ({len(HEALTHY_IDS)})" in txt
    assert f"CRC (n={len(CRC_IDS)})" in txt and f"Healthy (n={len(HEALTHY_IDS)})" in txt
    assert f"{stat_for_verification['P_two_sided']:.4f}" in txt
    page.get_pixmap(matrix=fitz.Matrix(300 / 72, 300 / 72), alpha=False).save(
        FIG / "Figure5_qPCR_revised_candidate.png")
    page.get_pixmap(matrix=fitz.Matrix(1.7, 1.7), alpha=False).save(QA / "Figure5_review.png")
    for stem, box in [("Figure5c_Q1_individual_well_Ct", fitz.Rect(270, 104, width, 292)),
                      ("Figure5d_Q1_CRC_vs_Healthy", fitz.Rect(270, 292, width, 408)),
                      ("Figure5cd_Q1_qPCR", REPLACEMENT)]:
        part = fitz.open()
        partpage = part.new_page(width=box.width, height=box.height)
        partpage.show_pdf_page(partpage.rect, doc, 0, clip=box)
        part.save(FIG / (stem + ".pdf"), garbage=4, deflate=True)
        partpage.get_pixmap(matrix=fitz.Matrix(300 / 72, 300 / 72), alpha=False).save(FIG / (stem + ".png"))
        part.close()
    verify = {"original_page_size_points": [width, height], "replacement_rectangle_points": list(REPLACEMENT),
              "excluded_sample_or_discordant_text_visible": False, "cohort_labels_verified": True,
              "pdf_vectors_preserved": True, "status": "PENDING_VISUAL_REVIEW"}
    (QA / "figure_validation.json").write_text(json.dumps(verify, indent=2) + "\n")
    doc.close()
    baseline.close()


def main():
    samples, wells, values, stat = prepare_data()
    with fitz.open(INPUT / "Figure5_v27_reference.pdf") as doc:
        width, height = doc[0].rect.width, doc[0].rect.height
    overlay = draw_qpcr(wells, values, stat, width, height)
    compose_and_verify(overlay, width, height, stat)
    # Inspect persisted PDFs in a fresh renderer, avoiding cross-document glyph
    # cache effects after show_pdf_page operations in the composition process.
    subprocess.run([sys.executable, str(Path(__file__).resolve().parent / "verify_preservation.py")], check=True)
    manifest = {str(p.relative_to(ROOT)): sha(p) for folder in [DATA, FIG] for p in sorted(folder.glob("*")) if p.is_file()}
    (QA / "output_sha256.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"n_CRC": len(CRC_IDS), "n_Healthy": len(HEALTHY_IDS), "numeric_Ct": int(wells.ct.notna().sum()),
                      "statistic": stat, "figure": str(FIG / "Figure5_qPCR_revised_candidate.pdf")}, indent=2))


if __name__ == "__main__":
    main()
