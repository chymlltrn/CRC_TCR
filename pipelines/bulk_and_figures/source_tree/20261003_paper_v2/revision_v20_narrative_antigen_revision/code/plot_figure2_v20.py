#!/usr/bin/env python3
"""Reorder frozen sequence-cluster evidence and patient prediction; no new fits.

The v19 D/F clinical estimates share rows in D, while fixed-size reference
cells encode their original count in color. Source estimates, bootstrap limits,
archived UMAP coordinates and all inference populations remain unchanged.
"""
from __future__ import annotations
from release_runtime import private_literal as _private_literal, release_path as _release_path

import datetime
import hashlib
import json
import shutil
from pathlib import Path

import fitz
from lxml import etree
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.text import Text
from matplotlib.colors import Normalize
import numpy as np
import pandas as pd

ROOT = Path(_release_path('path_052a3da416b1c4ab'))
BASE = ROOT.parent / "revision_v19_visual_reference_revision"
OUT = ROOT / "analysis/figure2_v20"
SOURCES = OUT / "sources"
WIDTH, HEIGHT = 180, 240
BLUE, PURPLE, ORANGE = "#0072B2", "#7655A3", "#D55E00"
GRAY, LIGHT_GRAY = "#65717B", "#BBC3CA"
ENCODER_COLOR = {"tcrbert": BLUE, "sceptr": ORANGE}
CLUSTER_COLOR = {("tcrbert", 11): BLUE, ("tcrbert", 7): PURPLE, ("sceptr", 5): ORANGE}
MODEL_COLOR = {"Baseline": GRAY, "TCR-BERT": BLUE, "SCEPTR": ORANGE,
               "CDR3vec": "#009E73", "3-mer SVD": PURPLE}
CLUSTERS = [("tcrbert", 11), ("tcrbert", 7), ("sceptr", 5)]
INPUTS = {
    "A_TCR_BERT_archived_umap.tsv": "analysis/figure2_v19/sources/A_TCR_BERT_archived_umap.tsv",
    "A_sampling_receipt.json": "analysis/figure2_v19/sources/A_sampling_receipt.json",
    "B_SCEPTR_archived_umap.tsv": "analysis/figure2_v19/sources/B_SCEPTR_archived_umap.tsv",
    "B_sampling_receipt.json": "analysis/figure2_v19/sources/B_sampling_receipt.json",
    "C_cluster_associations.tsv": "analysis/figure2_v19/sources/C_cluster_associations.tsv",
    "D_selected_estimates.tsv": "analysis/figure2_v19/sources/D_selected_estimates.tsv",
    "D_cohort_medians.tsv": "analysis/figure2_v19/sources/F_cohort_medians.tsv",
    "E_reference_matches.tsv": "analysis/figure2_v19/sources/E_reference_matches.tsv",
    "E_cluster_coverage.tsv": "manuscript/SourceData/cluster_exact_coverage_reviewed.tsv",
    "E_cluster_antigen_profile.tsv": "manuscript/SourceData/cluster_antigen_profile.tsv",
    "F_MIL_AUROC.tsv": "analysis/figure2_v19/sources/G_MIL_AUROC.tsv",
    "G_MIL_bag_curve.tsv": "analysis/figure2_v19/sources/H_MIL_bag_curve.tsv",
    "C_complete_248_tests.tsv": "manuscript/SourceData/all_248_pooled_tests.tsv",
}
PANEL_MAP = {
    "A": {"v19_panels": ["A"], "meaning": "Descriptive archived TCR-BERT UMAP; selected clusters11/7"},
    "B": {"v19_panels": ["B"], "meaning": "Descriptive archived SCEPTR UMAP; selected cluster5"},
    "C": {"v19_panels": ["C"], "meaning": "All62 plotted read-fraction cluster tests; full248-test negative boundary"},
    "D": {"v19_panels": ["D", "F"], "meaning": "Three selected adjusted ORs with95%CI and cohort-specific nodal group medians"},
    "E": {"v19_panels": ["E"], "meaning": "Fixed-size matrix of24 strictV/J antigen-reference sequence counts"},
    "F": {"v19_panels": ["G"], "meaning": "Actual126-patient MIL discrimination and fixed-prediction confidence limits"},
    "G": {"v19_panels": ["H"], "meaning": "Separate114-patient bag-size comparison; all paired gain intervals include zero"},
}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(name, data):
    (OUT / name).write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n")


def read(name):
    return pd.read_csv(SOURCES / name, sep="\t")


def ax_at(fig, x, y, w, h):
    ax = fig.add_axes([x / WIDTH, y / HEIGHT, w / WIDTH, h / HEIGHT])
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(width=.6, length=2, pad=2)
    return ax


def heading(fig, letter, text, x, y):
    fig.text(x / WIDTH, y / HEIGHT, letter, size=11, weight="bold")
    fig.text((x + 5) / WIDTH, y / HEIGHT, text, size=9.5, weight="bold")


def copy_and_verify_sources():
    OUT.mkdir(parents=True, exist_ok=True); SOURCES.mkdir(exist_ok=True)
    copies = []
    for target, relative in INPUTS.items():
        src, dst = BASE / relative, SOURCES / target
        shutil.copyfile(src, dst)
        assert sha(src) == sha(dst)
        copies.append({"baseline_version": "v19", "source": relative,
                       "copy": str(dst.relative_to(ROOT)), "sha256": sha(dst), "byte_identical": True})
    a, b = read("A_TCR_BERT_archived_umap.tsv"), read("B_SCEPTR_archived_umap.tsv")
    for key, data, name in [("A", a, "A_TCR_BERT_archived_umap.tsv"), ("B", b, "B_SCEPTR_archived_umap.tsv")]:
        receipt = json.loads((SOURCES / (key + "_sampling_receipt.json")).read_text())
        assert len(data) == 8475 and data.patient_key.nunique() == 339 and data.seq_idx.nunique() == 8468
        assert data.groupby("patient_key").size().eq(25).all() and np.isfinite(data[["umap_1", "umap_2"]]).all().all()
        assert sha(SOURCES / name) == receipt["sample_sha256"]
        assert receipt["label_blind_sampling"] and not receipt["new_umap_fit"] and not receipt["new_clustering"]
    assert a[["source_display_row", "patient_key", "seq_idx"]].equals(b[["source_display_row", "patient_key", "seq_idx"]])
    screen, selected, medians = read("C_cluster_associations.tsv"), read("D_selected_estimates.tsv"), read("D_cohort_medians.tsv")
    assert len(screen) == 62 and screen.n_patients.eq(289).all() and screen.N0.eq(179).all() and screen.Nplus.eq(110).all()
    assert len(selected) == 3 and list(zip(selected.model, selected.cluster_id)) == CLUSTERS
    assert selected.q_BH_all_248.gt(.05).all() and len(medians) == 6
    full = read("C_complete_248_tests.tsv")
    assert len(full) == 248 and full.q_BH_all_248.gt(.05).all()
    minimum_q = float(full.q_BH_all_248.min())
    assert abs(minimum_q - .2304980263305431) < 1e-10
    # All clinical medians must derive from their corresponding frozen OR-source row.
    clinical = []
    for row in selected.itertuples():
        output = {"model": row.model, "cluster_id": row.cluster_id, "odds_ratio": row.odds_ratio,
                  "ci_lower": row.ci_lower, "ci_upper": row.ci_upper, "model_P": row.p,
                  "q_BH_all_248": row.q_BH_all_248, "OR_patients": row.n_patients}
        for prefix, cohort in [("home", "Study"), ("external", "Comparison")]:
            median = medians[(medians.model == row.model) & (medians.cluster_id == row.cluster_id) & (medians.cohort == cohort)].iloc[0]
            for group, raw in [("pN0", prefix + "_median_fraction_N0"), ("pNplus", prefix + "_median_fraction_Nplus")]:
                value = getattr(median, group + "_percent")
                assert np.isclose(value, 100 * getattr(row, raw), atol=1e-12)
                output[cohort + "_" + group + "_read_percent"] = value
        clinical.append(output)
    pd.DataFrame(clinical).to_csv(OUT / "D_shared_clinical_rows.tsv", sep="\t", index=False)
    counts, coverage, profile = read("E_reference_matches.tsv"), read("E_cluster_coverage.tsv"), read("E_cluster_antigen_profile.tsv").fillna("")
    assert len(counts) == 24
    for row in counts.itertuples():
        match = profile[(profile.model == row.model) & (profile.cluster_id == row.cluster_id) &
                        (profile.canonical_organism == row.organism) & (profile.canonical_antigen == row.antigen)]
        assert len(match) == 1 and match.fully_VJ_concordant_AA.iloc[0] == row.strict_VJ_reference_AA
    denominator = coverage[["model", "cluster_id", "total_unique_AA", "exact_specificity_fully_VJ_concordant_AA"]]
    display = counts.merge(denominator, on=["model", "cluster_id"], validate="many_to_one")
    display["antigen_annotation_coverage_percent"] = 100 * display.strict_VJ_reference_AA / display.total_unique_AA
    display["cell_size"] = "fixed"
    display["color_value"] = display.strict_VJ_reference_AA
    display.to_csv(OUT / "E_reference_count_and_coverage.tsv", sep="\t", index=False)
    correlation = float(np.corrcoef(display.strict_VJ_reference_AA, display.antigen_annotation_coverage_percent)[0, 1])
    write("reference_size_encoding_decision.json", {"status": "SOURCE_VERIFIED_DECISION", "selected_encoding": "fixed-size cells; original reference counts in continuous color and text",
          "coverage_denominator_available": True, "coverage_unit": "distinct antigen-supported CDR3betaAA / all distinct CDR3betaAA in the same full cluster",
          "cluster_totals": denominator.to_dict("records"), "not_denominators": [205, 439, 359],
          "all24_numerators_match_original_antigen_profile": True,
          "coverage_percent_range": [float(display.antigen_annotation_coverage_percent.min()), float(display.antigen_annotation_coverage_percent.max())],
          "count_coverage_Pearson_r": correlation,
          "reason": "Coverage and count correlate r=.9953; additional size nearly repeats color and reintroduces tiny unreadable cells. Keep complete legitimate coverage values in SourceData.",
          "category_overlap": "Antigen categories may overlap; cell counts must not be summed into a unique total.",
          "inference_boundary": "Sparse reference annotations do not assign common functional specificity to a cluster."})
    auc, curve = read("F_MIL_AUROC.tsv"), read("G_MIL_bag_curve.tsv")
    assert len(auc) == 5 and auc.n.eq(126).all() and auc.N0.eq(70).all() and auc.Nplus.eq(56).all()
    assert len(curve) == 10 and curve.n.eq(114).all()
    assert sorted(curve.bag_size.unique()) == [1024, 2048, 4096, 8192, 16384]
    assert curve.paired_gain_vs_clinical_technical_batch_ci_lower.lt(0).all() and curve.paired_gain_vs_clinical_technical_batch_ci_upper.gt(0).all()
    write("source_manifest.json", {"status": "PASS", "copied_frozen_v19_inputs": copies,
          "minimum_q_complete248": minimum_q, "no_new_model_training": True,
          "no_refit_or_new_endpoint_test": True, "no_new_embedding_or_clustering": True,
          "clinical_units": {"UMAP_patients": 339, "association_patients": 289, "actual_MIL": 126, "bag_size_MIL": 114},
          "UMAP_occurrences": 8475, "UMAP_distinct_AA": 8468})


def umap(fig, letter, title, x, name, highlights):
    heading(fig, letter, title, x - 14, 237)
    ax = ax_at(fig, x, 190, 43, 44)
    ax.set_box_aspect(68 / 67)
    data = read(name); selected = [item[0] for item in highlights]
    other = data[~data.cluster_id.isin(selected)]
    ax.scatter(other.umap_1, other.umap_2, s=1.5, c=LIGHT_GRAY, alpha=.48, lw=0)
    handles = []
    for cluster, color, label in highlights:
        subset = data[data.cluster_id.eq(cluster)]
        ax.scatter(subset.umap_1, subset.umap_2, s=4, c=color, alpha=.85, lw=0)
        handles.append(Line2D([], [], marker="o", ls="", ms=3, color=color, label=label))
    handles.append(Line2D([], [], marker="o", ls="", ms=3, color=LIGHT_GRAY, label="Other"))
    ax.set(xlabel="UMAP 1", ylabel="UMAP 2")
    if letter == "A":
        ax.set_xticks([0, 5, 10]); ax.set_yticks([0, 5, 10, 15])
    else:
        ax.set_xticks([-2, 0, 5, 10]); ax.set_yticks([0, 2, 4, 6, 8])
    ax.legend(handles=handles, loc="center left", bbox_to_anchor=(1.02, .55),
              frameon=False, handletextpad=.25, borderpad=.1, labelspacing=.5)
    return ax


def build():
    copy_and_verify_sources()
    plt.rcParams.update({"font.family": "Arial", "font.size": 8, "axes.labelsize": 8,
        "axes.titlesize": 8.5, "xtick.labelsize": 8, "ytick.labelsize": 8,
        "legend.fontsize": 8, "svg.fonttype": "path", "pdf.fonttype": 42,
        "axes.linewidth": .6, "savefig.facecolor": "white"})
    fig = plt.figure(figsize=(WIDTH / 25.4, HEIGHT / 25.4), dpi=300)
    umap(fig, "A", "TCR-BERT sequence map", 16, "A_TCR_BERT_archived_umap.tsv",
         [(11, BLUE, "BERT 11"), (7, PURPLE, "BERT 7")])
    umap(fig, "B", "SCEPTR sequence map", 106, "B_SCEPTR_archived_umap.tsv", [(5, ORANGE, "SCEPTR 5")])

    heading(fig, "C", "Cluster-level pN associations", 2, 174)
    ax = ax_at(fig, 17, 145, 146, 24)
    screen, selected = read("C_cluster_associations.tsv"), read("D_selected_estimates.tsv")
    for model, marker, label in [("tcrbert", "o", "TCR-BERT"), ("sceptr", "s", "SCEPTR")]:
        data = screen[screen.model.eq(model)]
        ax.scatter(data.beta, -np.log10(data.p), s=12, marker=marker, facecolor="white", edgecolor=ENCODER_COLOR[model], lw=.8, label=label)
        for key in CLUSTERS:
            if key[0] != model:
                continue
            row = selected[(selected.model == key[0]) & (selected.cluster_id == key[1])].iloc[0]
            ax.scatter(row.beta, -np.log10(row.p), s=22, marker=marker, color=CLUSTER_COLOR[key], zorder=4)
    ax.axhline(-np.log10(.05), color="#BAC4CD", lw=.6, ls="--")
    ax.axvline(0, color="#BAC4CD", lw=.6, ls="--")
    ax.set(xlabel="Adjusted log OR", ylabel="−log10(P)", xlim=(-.78, .72), ylim=(-.02, 2.8))
    ax.set_yticks([0, 1, 2])
    ax.legend(loc="upper right", frameon=False, handletextpad=.3, borderpad=.1)

    heading(fig, "D", "Selected clinical profiles", 2, 130)
    fig.text(51 / WIDTH, 126 / HEIGHT, "Adjusted OR", ha="center", size=8.5)
    fig.text(85 / WIDTH, 126 / HEIGHT, "Model P", ha="center", size=8.5)
    fig.text(122 / WIDTH, 126 / HEIGHT, "Study", ha="center", size=8.5)
    fig.text(161 / WIDTH, 126 / HEIGHT, "Comparison", ha="center", size=8.5)
    ax = ax_at(fig, 31, 101, 40, 22)
    labels = []
    for index, row in selected.iterrows():
        labels.append(("BERT " if row.model == "tcrbert" else "SCEPTR ") + str(row.cluster_id))
        ax.errorbar(row.odds_ratio, index,
                    xerr=[[row.odds_ratio - row.ci_lower], [row.ci_upper - row.odds_ratio]],
                    fmt="o", ms=4, color=CLUSTER_COLOR[row.model, row.cluster_id], lw=1.2, capsize=0)
        ax.text(1.35, index, f"{row.p:.3f}", transform=ax.get_yaxis_transform(), ha="center", va="center", size=8)
    ax.axvline(1, color="#BAC4CD", lw=.6, ls="--")
    ax.set(yticks=range(3), yticklabels=labels, ylim=(2.5, -.6), xlim=(.85, 1.95), xlabel="OR per 1 SD")
    ax.set_xticks([1, 1.5])
    medians = read("D_cohort_medians.tsv")
    for cohort, x in [("Study", 107), ("Comparison", 146)]:
        ax = ax_at(fig, x, 101, 31, 22)
        for index, key in enumerate(CLUSTERS):
            row = medians[(medians.model == key[0]) & (medians.cluster_id == key[1]) & (medians.cohort == cohort)].iloc[0]
            ax.plot([row.pN0_percent, row.pNplus_percent], [index, index], color="#ABB4BD", lw=1)
            ax.plot(row.pN0_percent, index, "o", color=GRAY, ms=3.7)
            ax.plot(row.pNplus_percent, index, "s", color=ORANGE, ms=3.7)
        ax.set(ylim=(2.5, -.6), xlim=(1.1, 2.05), yticks=[], xticks=[1.2, 1.6, 2.0])
        ax.spines["left"].set_visible(False)
    fig.text(142 / WIDTH, 94 / HEIGHT, "Median repertoire reads (%)", ha="center", size=8)
    fig.legend(handles=[Line2D([], [], marker="o", color=GRAY, ls="", label="pN0"),
                        Line2D([], [], marker="s", color=ORANGE, ls="", label="pN+")],
               loc="center", bbox_to_anchor=(143 / WIDTH, 132 / HEIGHT), frameon=False, ncol=2,
               handletextpad=.3, columnspacing=1, borderpad=0)

    heading(fig, "E", "Antigen-reference context", 2, 86)
    counts = read("E_reference_matches.tsv")
    pairs = list(dict.fromkeys(zip(counts.organism, counts.antigen)))
    matrix = np.array([[int(counts[(counts.organism == pair[0]) & (counts.antigen == pair[1]) &
                                (counts.model == key[0]) & (counts.cluster_id == key[1])].strict_VJ_reference_AA.iloc[0])
                        for key in CLUSTERS] for pair in pairs])
    ax = ax_at(fig, 73, 55, 65, 24)
    norm = Normalize(vmin=0, vmax=65)
    cmap = matplotlib.colormaps["YlGnBu"]
    image = ax.imshow(matrix, cmap=cmap, norm=norm, aspect="auto", interpolation="none")
    labels = [pair[0] + " · " + pair[1] for pair in pairs]
    ax.set(xticks=range(3), xticklabels=["B11", "B7", "S5"], yticks=range(8), yticklabels=labels)
    ax.xaxis.tick_top(); ax.tick_params(length=0, pad=3)
    ax.set_xticks(np.arange(-.5, 3, 1), minor=True); ax.set_yticks(np.arange(-.5, 8, 1), minor=True)
    ax.grid(which="minor", color="white", lw=.8); ax.tick_params(which="minor", length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    for y in range(8):
        for x in range(3):
            rgba = cmap(norm(matrix[y, x])); luminance = .2126 * rgba[0] + .7152 * rgba[1] + .0722 * rgba[2]
            ax.text(x, y, str(matrix[y, x]), ha="center", va="center", color="white" if luminance < .48 else "#202B33", size=8)
    bar = fig.add_axes([149 / WIDTH, 55 / HEIGHT, 2.5 / WIDTH, 24 / HEIGHT])
    colorbar = fig.colorbar(image, cax=bar, orientation="vertical", ticks=[0, 20, 40, 60])
    colorbar.outline.set_linewidth(.5); colorbar.ax.tick_params(length=2, width=.5, pad=1)
    colorbar.ax.yaxis.set_ticks_position("left")
    colorbar.ax.yaxis.set_label_position("right")
    fig.text(163 / WIDTH, 67 / HEIGHT, "Reference TCRβ sequences\n(strict V/J)",
             size=8, rotation=270, ha="center", va="center", multialignment="center")

    fig.add_artist(Line2D([2 / WIDTH, 178 / WIDTH], [47 / HEIGHT, 47 / HEIGHT], color="#D6DDE2", lw=.55))
    heading(fig, "F", "Patient-level discrimination", 2, 42)
    ax = ax_at(fig, 17, 11, 67, 23)
    auc = read("F_MIL_AUROC.tsv")
    for index, row in auc.iterrows():
        ax.errorbar(index, row.auroc, yerr=[[row.auroc - row.ci_low], [row.ci_high - row.auroc]],
                    fmt="o", color=MODEL_COLOR[row.label], ms=3.7, lw=1, capsize=1.5)
    ax.axhline(.5, color="#BAC4CD", lw=.6, ls="--")
    ticklabels = auc.label.replace({"TCR-BERT": "TCR-\nBERT", "3-mer SVD": "3-mer\nSVD"})
    ax.set(xticks=range(5), xticklabels=ticklabels, ylabel="OOF AUROC", ylim=(.35, .73), xlim=(-.4, 4.4))
    ax.set_yticks([.4, .5, .6, .7])

    heading(fig, "G", "Bag-size sensitivity", 94, 42)
    ax = ax_at(fig, 110, 11, 64, 23)
    curve = read("G_MIL_bag_curve.tsv")
    for model, color, marker, label in [("tcrbert_dualtask_aa_mean", BLUE, "o", "TCR-BERT"),
                                         ("sceptr_cdr3_only", ORANGE, "s", "SCEPTR")]:
        data = curve[curve.model.eq(model)].sort_values("bag_size")
        ax.fill_between(data.bag_size, data.paired_gain_vs_clinical_technical_batch_ci_lower,
                        data.paired_gain_vs_clinical_technical_batch_ci_upper, color=color, alpha=.10, lw=0)
        ax.plot(data.bag_size, data.paired_gain_vs_clinical_technical_batch, color=color, marker=marker, ms=3, lw=1, label=label)
    ax.axhline(0, color="#BAC4CD", lw=.6, ls="--")
    ax.set_xscale("log", base=2)
    ax.set_xticks([1024, 2048, 4096, 8192, 16384], ["1k", "2k", "4k", "8k", "16k"])
    ax.set(xlabel="Sequences per bag", ylabel="Δ log loss", ylim=(-.14, .085))
    ax.set_yticks([-.1, 0, .05])
    ax.legend(loc="lower center", bbox_to_anchor=(.5, 1.02), ncol=2,
              frameon=False, handletextpad=.3, borderpad=.1, labelspacing=.2, columnspacing=.8)

    fig.canvas.draw(); renderer = fig.canvas.get_renderer(); bounds = []
    for text in fig.findobj(Text):
        if not text.get_visible() or not text.get_text():
            continue
        box = text.get_window_extent(renderer)
        bounds.append({"text": text.get_text(), "fontsize_pt": text.get_fontsize(), "bbox_pixels": list(box.bounds),
                       "inside_canvas": bool(box.x0 >= -.2 and box.y0 >= -.2 and box.x1 <= fig.bbox.width + .2 and box.y1 <= fig.bbox.height + .2)})
    base = ROOT / "figures/main/Figure2"
    base.parent.mkdir(parents=True, exist_ok=True)
    for ext in ["svg", "pdf", "png"]:
        fig.savefig(base.with_suffix("." + ext), dpi=300,
                    metadata={"Creator": "CRC-TCR v20 source-preserving Figure2"} if ext == "pdf" else None)
    plt.close(fig)
    tree = etree.parse(str(base.with_suffix(".svg"))); svg = tree.getroot()
    svg.set("width", f"{WIDTH}mm"); svg.set("height", f"{HEIGHT}mm")
    tree.write(str(base.with_suffix(".svg")), encoding="utf-8", xml_declaration=True)
    pdf = fitz.open(base.with_suffix(".pdf")); assert len(pdf) == 1
    assert np.allclose([pdf[0].rect.width, pdf[0].rect.height], [WIDTH / 25.4 * 72, HEIGHT / 25.4 * 72], atol=.01)
    pdf[0].get_pixmap(matrix=fitz.Matrix(300 / 72, 300 / 72), alpha=False).save(OUT / "native_pdf_review.png")
    pdf.close()
    write("panel_mapping.json", {"status": "ACCEPTED_STORY_REORDER", "baseline_version": "v19", "main_figure": 2,
          "dimensions_mm": [WIDTH, HEIGHT], "panels": PANEL_MAP,
          "old_to_new": {"A": "A", "B": "B", "C": "C", "D": "D", "E": "E", "F": "D", "G": "F", "H": "G"},
          "blocks": {"A-E": "sequence-cluster evidence, with descriptive maps and exploratory clinical/reference context",
                     "F-G": "patient-level prediction and bag-size sensitivity; distinct126/114 inference populations"}})
    write("figure_build_receipt.json", {"status": "GENERATED_AWAITING_VISUAL_REVIEW", "version": "v20",
          "prepared_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(), "dimensions_mm": [WIDTH, HEIGHT],
          "panels": list(PANEL_MAP), "minimum_text_pt": min(x["fontsize_pt"] for x in bounds),
          "all_text_inside_canvas": all(x["inside_canvas"] for x in bounds), "text_bounds": bounds,
          "plotting_code_sha256": sha(__file__), "original_estimates_and_coordinates_unchanged": True,
          "no_new_training_refit_or_endpoint_tests": True, "reference_cell_size": "fixed",
          "artifacts": {ext: {"path": str(base.with_suffix("." + ext).relative_to(ROOT)), "sha256": sha(base.with_suffix("." + ext))} for ext in ["svg", "pdf", "png"]},
          "native_pdf_render": {"path": str((OUT / "native_pdf_review.png").relative_to(ROOT)), "sha256": sha(OUT / "native_pdf_review.png")}})
    print(json.dumps({"Figure2": "drawn", "panels": 7, "frozen_source_checks": "PASS", "minimum_font_pt": min(x["fontsize_pt"] for x in bounds), "all_text_inside_canvas": all(x["inside_canvas"] for x in bounds)}))


if __name__ == "__main__":
    build()
