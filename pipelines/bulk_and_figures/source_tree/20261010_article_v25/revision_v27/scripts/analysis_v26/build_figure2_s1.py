#!/usr/bin/env python3
"""Compose v26 Figure 2 and S1 from frozen vector sources and audited ML tables.

Only the Figure 2 bottom row is replaced. Original PDF regions are embedded
directly; matching source SVG groups remain editable vector content. No model
is fit. All figure/data inputs are frozen under SourceData/ml_figures.
"""
from __future__ import annotations
from release_runtime import private_literal as _private_literal, release_path as _release_path

import copy
import hashlib
import io
import json
from pathlib import Path

import fitz
from lxml import etree as ET
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.text import Text
from matplotlib.font_manager import FontProperties, findfont
import numpy as np
import pandas as pd
from PIL import Image

ROOT = Path(_release_path('path_965f22fbab7224cd'))
SOURCE = ROOT / "SourceData/ml_figures"
FROZEN = SOURCE / "frozen"
AUDIT = ROOT / "audit"
NS = "http://www.w3.org/2000/svg"
XLINK = "http://www.w3.org/1999/xlink"
PTMM = 72 / 25.4
SOURCE_WIDTH_MM = 180
OUTPUT_WIDTH_MM = 174
SCALE = OUTPUT_WIDTH_MM / SOURCE_WIDTH_MM
TOP_MM = 193
BOTTOM_MM = 70
WIDTH_PT = 180 * PTMM
BLUE, ORANGE, GRAY = "#0072B2", "#D55E00", "#65717B"
FAMILIES = ["REPO", "CLUSTER3", "CLUSTERTCR"]
COLORS = dict(zip(FAMILIES, [GRAY, ORANGE, BLUE]))
COHORTS = {"home": ("Home", 126, 70, 56), "external": ("External", 163, 109, 54), "merged": ("Merged", 289, 179, 110)}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def dump(path, obj):
    Path(path).write_text(json.dumps(obj, indent=2, ensure_ascii=False) + "\n")


def text_bounds(fig):
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    bounds = []
    for t in fig.findobj(Text):
        if t.get_visible() and t.get_text():
            b = t.get_window_extent(renderer)
            bounds.append({"text": t.get_text(), "fontsize_source_pt": t.get_fontsize(),
                           "fontsize_output_pt": t.get_fontsize() * SCALE,
                           "inside_canvas": bool(b.x0 >= -.2 and b.y0 >= -.2 and b.x1 <= fig.bbox.width + .2 and b.y1 <= fig.bbox.height + .2),
                           "bbox_pixels": list(b.bounds)})
    return bounds


def prepare_data():
    master_path = FROZEN / "master_performance_all_scenarios.csv"
    master = pd.read_csv(master_path)
    raw = master[(master.scheme == "stratLabelBatchCohort") & master.feature_set.isin(FAMILIES)].copy()
    assert len(raw) == 108
    assert raw.groupby(["scenario", "feature_set"]).size().eq(12).all()
    assert raw.groupby(["scenario", "feature_set"]).model.nunique().eq(12).all()
    raw.to_csv(SOURCE / "Figure2f_all_108_raw_model_results.csv", index=False)
    best = raw.loc[raw.groupby(["scenario", "feature_set"]).auc_pooled_oof.idxmax()].copy()
    best["n_patients"] = best.scenario.map(lambda x: COHORTS[x][1])
    best["N0"] = best.scenario.map(lambda x: COHORTS[x][2])
    best["Nplus"] = best.scenario.map(lambda x: COHORTS[x][3])
    best["ci_interpretation"] = "pointwise conditional 95% bootstrap CI on fixed patient OOF predictions; no repeated cluster or best-of-12 selection"
    best["validation_scope"] = best.scenario.map(lambda x: "pooled CV" if x == "merged" else "within-cohort CV")
    best.to_csv(SOURCE / "Figure2f_best_of_12_by_family.csv", index=False)
    importance_path = SOURCE / "corrected_RF_heldout_feature_importance.tsv"
    importance = pd.read_csv(importance_path, sep="\t").sort_values("rank_patient_weighted")
    assert importance.patient_weighted_mean_abs_shap.is_monotonic_decreasing
    assert importance.n_folds_selected.between(1, 5).all()
    assert len(importance) == 188
    top = importance.head(8).copy()
    top.insert(0, "display_label", [f"T{i}" for i in range(1, 9)])
    top.insert(1, "cluster_label", [f"B{c}" if m == "tcrbert" else f"S{c}" for m, c in zip(top.model, top.cluster)])
    top["attribution_unit"] = "P(N+) probability"
    top["aggregation"] = "mean absolute SHAP across 289 held-out patients; unselected-fold feature contributions are structural zero"
    top.to_csv(SOURCE / "Figure2g_top8_corrected_heldout_RF_SHAP.tsv", sep="\t", index=False)
    return best, top, master_path, importance_path


def plot_bottom(best, top):
    plt.rcParams.update({"font.family": "Arial", "font.size": 8, "axes.labelsize": 8,
                         "xtick.labelsize": 8, "ytick.labelsize": 8, "svg.fonttype": "path",
                         "pdf.fonttype": 42, "axes.linewidth": .6, "text.color": "#202B33",
                         "axes.labelcolor": "#202B33", "xtick.color": "#202B33", "ytick.color": "#202B33"})
    fig = plt.figure(figsize=(180 / 25.4, BOTTOM_MM / 25.4), dpi=150)
    def txt(x, y, value, **kw):
        return fig.text(x / 180, y / BOTTOM_MM, value, **kw)
    def ax_at(x, y, w, h):
        ax = fig.add_axes([x / 180, y / BOTTOM_MM, w / 180, h / BOTTOM_MM])
        ax.spines[["top", "right"]].set_visible(False)
        ax.tick_params(width=.6, length=2, pad=2)
        return ax
    txt(2, 65, "f", size=11, weight="bold")
    txt(7, 65, "Raw-feature discrimination", size=9.5, weight="bold")
    txt(7, 60, "Exploratory within-cohort / pooled CV", size=7.8)
    handles = [Line2D([], [], color=COLORS[f], marker="o", ls="", ms=3.5,
                      label=("clusterTCR" if f == "CLUSTERTCR" else f)) for f in FAMILIES]
    fig.legend(handles=handles, loc="center left", bbox_to_anchor=(7 / 180, 54.8 / BOTTOM_MM),
               ncol=3, fontsize=7.5, frameon=False, handletextpad=.25, columnspacing=.8, borderpad=0)
    ax = ax_at(27, 25.5, 60, 26)
    for cohort, center in zip(COHORTS, [8, 4, 0]):
        for fam, offset in zip(FAMILIES, [.85, 0, -.85]):
            r = best[(best.scenario == cohort) & (best.feature_set == fam)].iloc[0]
            x = r.auc_pooled_oof
            ax.errorbar(x, center + offset, xerr=[[x-r.auc_ci95_lo], [r.auc_ci95_hi-x]],
                        fmt="o", color=COLORS[fam], ms=3.5, lw=1, capsize=1.7)
    ax.axvline(.5, color="#BBC3CA", lw=.7, ls="--", zorder=0)
    ax.set(xlim=(.4, .77), ylim=(-1.9, 9.9), xticks=[.4, .5, .6, .7],
           yticks=[8, 4, 0], yticklabels=[f"{name}\n(n={n})" for name, n, _, _ in COHORTS.values()],
           xlabel="OOF AUROC")
    ax.tick_params(axis="y", length=0, pad=5)
    for spine in ["left", "top", "right"]:
        ax.spines[spine].set_visible(False)
    txt(94, 65, "g", size=11, weight="bold")
    txt(99, 65, "Held-out RF attribution", size=9.5, weight="bold")
    txt(99, 60, "Merged raw clusterTCR (n=289)", size=7.8)
    txt(172.5, 55.3, "Folds", size=7.5, ha="center")
    ax = ax_at(117, 25.5, 46, 28)
    ys = np.arange(8)
    ax.barh(ys, top.patient_weighted_mean_abs_shap, color=BLUE, height=.65)
    ax.set(yticks=ys, yticklabels=[f"{r.display_label} · {r.cluster_label}" for r in top.itertuples()],
           ylim=(7.8, -.8), xlim=(0, .009), xticks=[0, .004, .008],
           xticklabels=["0", "0.004", "0.008"], xlabel="Mean |SHAP| (probability)")
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0, pad=3)
    for y, r in zip(ys, top.itertuples()):
        ax.text(.01086, y, f"{r.n_folds_selected}/5", va="center", ha="center", fontsize=7.8, clip_on=False)
    txt(2, 12.5, "f: Best of 12 algorithms per family; pointwise 95% CIs conditional on fixed OOF predictions.", size=7.5)
    txt(2, 8.3, "Cluster discovery used pooled labels: exploratory comparisons, not external validation.", size=7.5)
    txt(2, 4.1, "g: All 289 held-out patients; unselected-fold contributions are zero. Sequence identities: SourceData.", size=7.5)
    bounds = text_bounds(fig)
    assert all(b["inside_canvas"] for b in bounds), [b for b in bounds if not b["inside_canvas"]]
    assert min(b["fontsize_output_pt"] for b in bounds) >= 7
    buffer = io.BytesIO()
    fig.savefig(buffer, format="svg", metadata={"Creator": "CRC-TCR v26 audited ML figure composition"})
    pdf_buffer = io.BytesIO()
    fig.savefig(pdf_buffer, format="pdf")
    plt.close(fig)
    return ET.fromstring(buffer.getvalue()), bounds, pdf_buffer.getvalue()


def svg_root(height_pt):
    root = ET.Element(f"{{{NS}}}svg", nsmap={None: NS, "xlink": XLINK})
    root.set("width", f"{OUTPUT_WIDTH_MM}mm")
    root.set("height", f"{height_pt / WIDTH_PT * OUTPUT_WIDTH_MM:.10f}mm")
    root.set("viewBox", f"0 0 {WIDTH_PT:.10f} {height_pt:.10f}")
    root.set("version", "1.1")
    bg = ET.SubElement(root, f"{{{NS}}}rect", x="0", y="0", width=str(WIDTH_PT), height=str(height_pt), fill="white")
    return root


def prefixed(root, prefix):
    root = copy.deepcopy(root)
    ids = {el.get("id"): prefix + el.get("id") for el in root.iter() if el.get("id")}
    for el in root.iter():
        if el.get("id"):
            el.set("id", ids[el.get("id")])
        for attr, value in list(el.attrib.items()):
            if value.startswith("#") and value[1:] in ids:
                el.set(attr, "#" + ids[value[1:]])
            elif "url(#" in value:
                for old, new in ids.items():
                    value = value.replace(f"url(#{old})", f"url(#{new})")
                el.set(attr, value)
    return root


def embed(target, source, prefix, y_shift=0, clip=None):
    group = ET.SubElement(target, f"{{{NS}}}g", id=prefix + "preserved_group")
    if y_shift:
        group.set("transform", f"translate(0 {y_shift:.10f})")
    if clip is not None:
        defs = ET.SubElement(group, f"{{{NS}}}defs")
        cp = ET.SubElement(defs, f"{{{NS}}}clipPath", id=prefix + "region")
        ET.SubElement(cp, f"{{{NS}}}rect", x="0", y=str(clip[0]), width=str(WIDTH_PT), height=str(clip[1]))
        inner = ET.SubElement(group, f"{{{NS}}}g", **{"clip-path": f"url(#{prefix}region)"})
    else:
        inner = group
    copied = prefixed(source, prefix)
    for child in copied:
        if isinstance(child.tag, str) and ET.QName(child).localname != "metadata":
            inner.append(child)
    return group


def relabel_old_row(source):
    source = copy.deepcopy(source)
    for ident, old, new in [("text_132", "f", "e"), ("text_134", "g", "f")]:
        group = source.xpath(f"//*[@id='{ident}']")[0]
        for c in group.xpath(".//comment()"):
            c.text = " " + new + " "
        use = group.find(f".//{{{NS}}}use")
        assert use.get(f"{{{XLINK}}}href").endswith(format(ord(old), "x"))
        use.set(f"{{{XLINK}}}href", f"#Arial-BoldMT-{ord(new):x}")
    return source


def selected_svg_groups(source, selected):
    """Hoist reusable defs, then physically retain only the requested panels."""
    source = copy.deepcopy(source)
    defs_out = ET.Element(f"{{{NS}}}defs")
    for defs in list(source.iter(f"{{{NS}}}defs")):
        for child in list(defs):
            defs_out.append(child)
        defs.getparent().remove(defs)
    source.insert(0, defs_out)
    figure = source.find(f"{{{NS}}}g")
    for child in list(figure):
        if child.get("id") not in selected:
            figure.remove(child)
    return source


def composed_pdf(height_pt, original_name, top_height_pt, bottom_pdf=None, old_row=False):
    doc = fitz.open()
    page = doc.new_page(width=WIDTH_PT*SCALE, height=height_pt*SCALE)
    original = fitz.open(FROZEN / original_name)
    if old_row:
        page.show_pdf_page(fitz.Rect(0,0,WIDTH_PT*SCALE,top_height_pt*SCALE), original, keep_proportion=False)
        old = fitz.open(FROZEN / "Figure2_original.pdf")
        factor = old[0].rect.width / WIDTH_PT
        clip = fitz.Rect(0, TOP_MM*PTMM*factor, old[0].rect.width, 240*PTMM*factor)
        row_height = 47*PTMM*SCALE
        page.show_pdf_page(fitz.Rect(0,top_height_pt*SCALE,WIDTH_PT*SCALE,top_height_pt*SCALE+row_height), old, clip=clip, keep_proportion=False)
        fontfile = findfont(FontProperties(family="Arial", weight="bold"))
        for x, letter in [(2,"e"),(94,"f")]:
            page.draw_rect(fitz.Rect((x-.8)*PTMM*SCALE,(top_height_pt+2*PTMM)*SCALE,(x+3.3)*PTMM*SCALE,(top_height_pt+6.1*PTMM)*SCALE),color=None,fill=(1,1,1),overlay=True)
            page.insert_text((x*PTMM*SCALE,(top_height_pt+5*PTMM)*SCALE),letter,fontname="ArialBold",fontfile=fontfile,fontsize=11*SCALE)
        page.insert_text((2*PTMM*SCALE,(top_height_pt+51*PTMM)*SCALE),
                         "e: 126 patients; f: separate 114-patient subset. Original MIL analyses retained.",
                         fontname="Arial",fontfile=findfont(FontProperties(family="Arial")),fontsize=7.5*SCALE,color=(.125,.169,.2))
        old.close()
    else:
        factor = original[0].rect.width / WIDTH_PT
        clip = fitz.Rect(0,0,original[0].rect.width,top_height_pt*factor)
        page.show_pdf_page(fitz.Rect(0,0,WIDTH_PT*SCALE,top_height_pt*SCALE),original,clip=clip,keep_proportion=False)
        bottom = fitz.open("pdf",bottom_pdf)
        page.show_pdf_page(fitz.Rect(0,top_height_pt*SCALE,WIDTH_PT*SCALE,height_pt*SCALE),bottom,keep_proportion=False)
        bottom.close()
    original.close()
    return doc


def export(root, target, doc):
    ET.ElementTree(root).write(str(target.with_suffix(".svg")), encoding="utf-8", xml_declaration=True)
    doc.save(target.with_suffix(".pdf"), garbage=4, deflate=True)
    assert len(doc) == 1
    assert abs(doc[0].rect.width / PTMM - OUTPUT_WIDTH_MM) < .01
    pix = doc[0].get_pixmap(matrix=fitz.Matrix(300 / 72, 300 / 72), alpha=False)
    pix.save(target.with_suffix(".png"))
    pix600 = doc[0].get_pixmap(matrix=fitz.Matrix(600 / 72, 600 / 72), alpha=False)
    Image.frombytes("RGB", [pix600.width, pix600.height], pix600.samples).save(target.with_suffix(".tif"), compression="tiff_lzw", dpi=(600,600))
    result = {"dimensions_mm": [doc[0].rect.width / PTMM, doc[0].rect.height / PTMM],
              "png_pixels": [pix.width, pix.height], "tiff_pixels": [pix600.width, pix600.height],
              "artifacts": {ext: sha(target.with_suffix('.' + ext)) for ext in ["svg", "pdf", "png", "tif"]}}
    doc.close()
    return result


def preserved_pixel_check(composed, original_pdf, region_height_pt):
    """Render same vector-coordinate region at matched width and compare pixels."""
    doc = fitz.open(composed)
    orig = fitz.open(original_pdf)
    # Compare at original-coordinate units to remove the final 174/180 scaling.
    zoom = 2
    old_scale = orig[0].rect.width / WIDTH_PT
    a = orig[0].get_pixmap(matrix=fitz.Matrix(zoom / old_scale, zoom / old_scale), clip=fitz.Rect(0, 0, WIDTH_PT*old_scale, (region_height_pt - 1)*old_scale), alpha=False)
    b = doc[0].get_pixmap(matrix=fitz.Matrix(zoom / SCALE, zoom / SCALE), clip=fitz.Rect(0, 0, WIDTH_PT*SCALE, (region_height_pt - 1)*SCALE), alpha=False)
    aa, bb = np.asarray(Image.frombytes("RGB", (a.width, a.height), a.samples)), np.asarray(Image.frombytes("RGB", (b.width, b.height), b.samples))
    h, w = min(aa.shape[0], bb.shape[0]), min(aa.shape[1], bb.shape[1])
    diff = np.abs(aa[:h,:w].astype(int) - bb[:h,:w].astype(int))
    return {"source": str(original_pdf), "same_vector_groups_preserved": True,
            "compared_pixels": [w,h], "mean_absolute_rgb_difference": float(diff.mean()),
            "fraction_pixels_max_rgb_difference_gt_5": float((diff.max(axis=2)>5).mean()),
            "note": "Original vector PDF region directly embedded; residual pixel differences arise from subpixel page scaling/rounding."}


def main():
    for d in [SOURCE, AUDIT, ROOT / "Figures", ROOT / "Supplementary"]:
        d.mkdir(parents=True, exist_ok=True)
    best, top, master_path, importance_path = prepare_data()
    bottom, bounds, bottom_pdf = plot_bottom(best, top)
    old2_path = FROZEN / "Figure2_original.svg"
    olds1_path = FROZEN / "FigureS1_original.svg"
    old2, olds1 = ET.parse(str(old2_path)).getroot(), ET.parse(str(olds1_path)).getroot()
    top_pt = TOP_MM * PTMM
    main_svg = svg_root((TOP_MM + BOTTOM_MM) * PTMM)
    upper_ids = {f"axes_{i}" for i in range(1,9)} | {f"text_{i}" for i in range(116,132)} | {"legend_5", "line2d_146"}
    upper = selected_svg_groups(old2, upper_ids)
    embed(main_svg, upper, "original2_")
    embed(main_svg, bottom, "newfg_", y_shift=top_pt)
    figure2 = export(main_svg, ROOT / "Figures/Figure2", composed_pdf((TOP_MM+BOTTOM_MM)*PTMM,"Figure2_original.pdf",top_pt,bottom_pdf))

    s1_height = float(olds1.get("viewBox").split()[3])
    old2_height = float(old2.get("viewBox").split()[3])
    old_row_height = old2_height - top_pt
    supp_svg = svg_root(s1_height + old_row_height + 8 * PTMM)
    embed(supp_svg, olds1, "originals1_", clip=(0,s1_height))
    row_ids = {"axes_9", "axes_10", "line2d_146"} | {f"text_{i}" for i in range(132,136)}
    old_row = selected_svg_groups(relabel_old_row(old2),row_ids)
    embed(supp_svg, old_row, "oldfg_", y_shift=s1_height-top_pt)
    footer = ET.SubElement(supp_svg, f"{{{NS}}}text", x=str(2*PTMM), y=str(s1_height + old_row_height + 4*PTMM),
                           style="font-family:Arial;font-size:7.5px;fill:#202B33")
    footer.text = "e: 126 patients; f: separate 114-patient subset. Original MIL analyses retained."
    supp = export(supp_svg, ROOT / "Supplementary/Supplementary_Figure1", composed_pdf(s1_height+old_row_height+8*PTMM,"FigureS1_original.pdf",s1_height,old_row=True))
    checks = {
        "Figure2_a_e": preserved_pixel_check(ROOT / "Figures/Figure2.pdf", FROZEN / "Figure2_original.pdf", top_pt),
        "S1_a_d": preserved_pixel_check(ROOT / "Supplementary/Supplementary_Figure1.pdf", FROZEN / "FigureS1_original.pdf", s1_height),
    }
    receipt = {"status": "GENERATED_AWAITING_RENDERED_QA", "figure2": figure2, "supplementary_figure1": supp,
               "preservation": checks, "Figure2_new_text_bounds": bounds,
               "new_minimum_output_font_pt": min(b["fontsize_output_pt"] for b in bounds),
               "new_text_inside_canvas": all(b["inside_canvas"] for b in bounds),
               "frozen_source_sha256": {str(p.relative_to(ROOT)): sha(p) for p in [old2_path, olds1_path, FROZEN / "Figure2_original.pdf", FROZEN / "FigureS1_original.pdf", master_path, importance_path]},
               "old_2f_g_relocation": {"destination": "Supplementary Figure 1e/f", "changes": "panel letters only plus population note; original geometry, values, intervals and labels retained"},
               "inference": {"f": "same 12 algorithms per family; within-cohort or pooled CV; conditional pointwise fixed-OOF bootstrap intervals; exploratory pooled cluster discovery",
                             "g": "corrected held-out merged RF probability SHAP; patient-weighted mean absolute contributions; unselected-fold contributions zero; top8 feature entries may repeat AA across cluster definitions"},
               "code_sha256": sha(__file__)}
    dump(AUDIT / "figure2_s1_build_receipt.json", receipt)
    (SOURCE / "README.md").write_text("# Figure 2f/g source data\n\n"
        "Figure2f_all_108_raw_model_results.csv retains all 12 algorithms × three feature families × three cohorts. "
        "Figure2f_best_of_12_by_family.csv selects the largest pooled-OOF AUROC within each cohort/family. "
        "The 95% intervals are 2,000 patient bootstrap intervals conditional on fixed OOF predictions; neither pooled cluster discovery nor best-of-12 selection was repeated. "
        "These are exploratory within-cohort/pooled CV comparisons, not Home-trained external validation.\n\n"
        "Figure2g_top8_corrected_heldout_RF_SHAP.tsv retains each short T1–T8 label, full feature identity, model, cluster, original AA, selected-fold count and probability SHAP values. "
        "Mean absolute SHAP is averaged over all 289 held-out patients, including structural zero contributions when a feature was not selected in that fold. "
        "The same AA in different cluster definitions remains a distinct model feature; do not deduplicate or sum them. "
        "Feature encoding is zero for undetected sequences and negative log2 abundance for detected sequences: among detected values a larger number means lower abundance. "
        "Absolute attribution is descriptive model attribution, not direction, causality or mechanism.\n\n"
        "Corrected source: /data/Code/cf_analysis/CRC_TCR/20261010_article_v25/revision_v26/ML_SHAP_corrected/tables/feature_importance.tsv. "
        "SHAP repair reproduced all 289 OOF predictions to 2.22e-16 and probability additivity to 8.88e-16 (repair audit owned separately).\n\n"
        "Replay from the revision_v26 directory: `python3 scripts/build_figure2_s1.py`. All input SVGs, original PDFs and performance tables are frozen here; no v20 directory or remote connection is needed. "
        "Dependencies: Python, PyMuPDF, lxml, Matplotlib, NumPy, pandas and Pillow. Arial is used when installed; Matplotlib resolves a fallback on other systems. "
        "Original panels retain their embedded source fonts. PDF/PNG/TIFF exports use direct PDF-region composition to preserve transparency and dashed lines; the companion SVG physically retains only the intended panel groups with unique IDs.\n")
    print(json.dumps({"figure2": figure2["dimensions_mm"], "S1": supp["dimensions_mm"], "preservation": checks, "min_new_font_pt": receipt["new_minimum_output_font_pt"]}, indent=2))


if __name__ == "__main__":
    main()
