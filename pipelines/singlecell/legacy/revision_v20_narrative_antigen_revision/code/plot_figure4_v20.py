#!/usr/bin/env python3
"""Replot v20 Figure 4 from frozen v19 patient-level sources.

No endpoint tests, interval estimation, patient selection, or significance
thresholds are changed. KDEs in Figure 4E only display the supplied patient
distributions over their observed ranges. All mean/interval/test estimates are
read from the accepted tables. Source copies make the plotting lane portable.
"""
from __future__ import annotations
from release_config import release_path, private_value

import argparse
import datetime
import hashlib
import json
import shutil
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from matplotlib.lines import Line2D
from matplotlib.text import Text
import fitz
import numpy as np
import pandas as pd
from lxml import etree
from scipy.stats import gaussian_kde

ROOT = release_path('render_v20_root')
BASE = ROOT.parent / "revision_v19_visual_reference_revision"
OUT = ROOT / "analysis/figure4_v20"
SOURCE = OUT / "source_baseline"
WIDTH = 180
PROGRAMS = ["cytotoxic", "exhaustion", "naive_memory", "interferon", "residency", "proliferation"]
LABELS = ["Cytotoxicity", "Exhaustion", "Naive/memory", "Interferon", "Residency", "Proliferation"]
COLORS = {"Tumor": "#D55E00", "Normal": "#0072B2"}
INPUTS = {
    "fractions": "analysis/figure34/source_baseline/fractions.tsv",
    "fractions_patients": "analysis/figure34/source_baseline/fractions_patients.tsv",
    "source_program": "analysis/figure34/source_baseline/source_program.tsv",
    "source_program_patients": "analysis/figure34/source_baseline/source_program_patients.tsv",
    "viral": "analysis/figure34/source_baseline/viral.tsv",
    "viral_patients": "analysis/figure34/source_baseline/viral_patients.tsv",
    "abundance_program": "analysis/figure34/source_baseline/abundance_program.tsv",
    "abundance_state": "analysis/figure34/source_baseline/abundance_state.tsv",
    "abundance_state_patients": "analysis/figure34/source_baseline/abundance_state_patients.tsv",
    "clinical_matching": "analysis/figure34/source_baseline/clinical_matching.tsv",
}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def prepare():
    OUT.mkdir(parents=True, exist_ok=True)
    SOURCE.mkdir(parents=True, exist_ok=True)
    (ROOT / "figures/main").mkdir(parents=True, exist_ok=True)
    manifest = []
    for key, rel in INPUTS.items():
        original = BASE / rel
        copied = SOURCE / f"{key}.tsv"
        if not copied.exists():
            shutil.copyfile(original, copied)
        if original.exists():
            assert sha(original) == sha(copied), key
        manifest.append({"key": key, "v19_source": str(original),
                         "portable_path": str(copied.relative_to(ROOT)),
                         "sha256": sha(copied), "bytes": copied.stat().st_size})
    (OUT / "input_source_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


def read(key):
    return pd.read_csv(SOURCE / f"{key}.tsv", sep="\t")


def table(frame, name):
    frame.to_csv(OUT / name, sep="\t", index=False, na_rep="NA")


def axis(fig, height, x, y, w, h):
    ax = fig.add_axes([x / WIDTH, y / height, w / WIDTH, h / height])
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(width=.6, length=2, pad=2)
    return ax


def heading(fig, height, letter, title, x, y):
    fig.text(x / WIDTH, y / height, letter, fontsize=11, fontweight="bold", va="baseline")
    fig.text((x + 5) / WIDTH, y / height, title, fontsize=9.5, fontweight="bold", va="baseline")


def mark(ax, x, y, color, q=None, shape="D", size=3.5):
    ax.plot(x, y, marker=shape, markersize=size,
            markerfacecolor=color if q is None or q < .05 else "white",
            markeredgecolor=color, markeredgewidth=.9, linestyle="none", zorder=5)


def pstring(p):
    return "P<.001" if p < .001 else f"P={p:.3f}".replace("0.", ".")


def forest(ax, rows, patients=None, paired=False, effect="mean_difference",
           low="CI_low", high="CI_high", pcol="P", qcol="q", color=None, p_labels=False):
    for k, program in enumerate(PROGRAMS):
        for _, r in rows[rows.program.eq(program)].iterrows():
            tissue = r.get("tissue", "Tumor")
            dy = (-.18 if tissue == "Tumor" else .18) if paired else 0
            col = color or COLORS[tissue]
            if patients is not None:
                pp = patients[patients.program.eq(program)]
                if "tissue" in pp:
                    pp = pp[pp.tissue.eq(tissue)]
                pp = pp.sort_values("patient_id")
                assert len(pp) == int(r.n)
                assert np.isclose(pp.difference.mean(), r[effect], atol=1e-11)
                ax.scatter(pp.difference, k + dy + np.linspace(-.12, .12, len(pp)),
                           s=5, color=col, alpha=.22, linewidths=0, zorder=2)
            ax.plot([r[low], r[high]], [k + dy, k + dy], color=col, lw=.95, zorder=4)
            mark(ax, r[effect], k + dy, col, r[qcol],
                 shape=("o" if tissue == "Tumor" else "s") if paired else "D")
            if p_labels:
                ax.text(1.025, k + dy, pstring(r[pcol]), transform=ax.get_yaxis_transform(),
                        fontsize=8, va="center")
    ax.axvline(0, color="#A7A7A7", lw=.6, zorder=1)
    ax.set_yticks(range(6), LABELS)
    ax.set_ylim(5.55, -.55)
    ax.grid(axis="x", color="#E4E6E8", lw=.4, zorder=0)


def tissue_legend(fig, height, y):
    fig.legend(handles=[Line2D([], [], color=COLORS["Tumor"], marker="o", ms=3,
                               lw=1, label="Tumor"),
                        Line2D([], [], color=COLORS["Normal"], marker="s", ms=3,
                               lw=1, label="Adjacent")],
               loc="center left", bbox_to_anchor=(158 / WIDTH, y / height), ncol=1,
               frameon=False, handlelength=1.0, handletextpad=.35,
               labelspacing=.6, borderaxespad=0)


def export(fig, ordinal, height, mapping):
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    bounds = []
    for t in fig.findobj(Text):
        if not t.get_visible() or not t.get_text():
            continue
        b = t.get_window_extent(renderer)
        bounds.append({"text": t.get_text(), "fontsize_pt": t.get_fontsize(),
                       "bbox_pixels": list(b.bounds),
                       "inside_canvas": bool(b.x0 >= -.2 and b.y0 >= -.2 and
                                              b.x1 <= fig.bbox.width + .2 and
                                              b.y1 <= fig.bbox.height + .2)})
    base = ROOT / "figures/main" / f"Figure{ordinal}"
    for ext in ("svg", "pdf", "png"):
        fig.savefig(base.with_suffix("." + ext), dpi=300,
                    metadata={"Creator": "CRC-TCR v20 frozen-source Figure 4 replot"} if ext == "pdf" else None)
    plt.close(fig)
    tree = etree.parse(str(base.with_suffix(".svg")))
    svg = tree.getroot(); svg.set("width", f"{WIDTH}mm"); svg.set("height", f"{height}mm")
    tree.write(str(base.with_suffix(".svg")), encoding="utf-8", xml_declaration=True)
    doc = fitz.open(base.with_suffix(".pdf"))
    assert len(doc) == 1
    assert np.allclose([doc[0].rect.width, doc[0].rect.height],
                       [WIDTH / 25.4 * 72, height / 25.4 * 72], atol=.01)
    # Native PDF raster is deliberately separate from Matplotlib's PNG export.
    doc[0].get_pixmap(matrix=fitz.Matrix(300 / 72, 300 / 72), alpha=False).save(
        OUT / f"figure{ordinal}_native_pdf_render.png")
    doc.close()
    receipt = {"status": "GENERATED_AWAITING_VISUAL_REVIEW", "figure": ordinal,
               "prepared_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
               "dimensions_mm": [WIDTH, height], "minimum_text_pt": min(t["fontsize_pt"] for t in bounds),
               "all_text_inside_canvas": all(t["inside_canvas"] for t in bounds),
               "text_bounds": bounds, "svg_root": dict(svg.attrib),
               "plotting_code_sha256": sha(__file__), "panel_mapping": mapping,
               "no_new_endpoint_tests": True, "original_means_intervals_P_q_preserved": True,
               "artifacts": {ext: {"path": str(base.with_suffix("." + ext).relative_to(ROOT)),
                                   "sha256": sha(base.with_suffix("." + ext))}
                             for ext in ("svg", "pdf", "png")},
               "native_pdf_render": {"path": str((OUT / f"figure{ordinal}_native_pdf_render.png").relative_to(ROOT)),
                                     "sha256": sha(OUT / f"figure{ordinal}_native_pdf_render.png")}}
    (OUT / f"figure{ordinal}_build_receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    return {k: v for k, v in receipt.items() if k not in ("text_bounds", "svg_root")}


def figure4():
    height = 232
    fig = plt.figure(figsize=(WIDTH / 25.4, height / 25.4))
    a, ap = read("fractions"), read("fractions_patients")
    assert len(a) == 16 and a.n.eq(18).all()
    heading(fig, height, "A", "Blood-source detection", 2, 228)
    categories = ["both", "own_only", "bulk_only", "neither"]
    catlabels = ["Both", "Autologous only", "Independent only", "Neither"]
    for metric, xpos, title in (("cell_fraction", 40, "Cells"),
                                ("clone_fraction", 113, "Clonotypes")):
        ax = axis(fig, height, xpos, 199, 40, 23)
        ax.set_title(title, fontsize=8, pad=3)
        for _, row in a[a.metric.eq(metric)].iterrows():
            y = categories.index(row.category) + (-.15 if row.tissue == "Tumor" else .15)
            vals = ap[ap.tissue.eq(row.tissue) & ap.category.eq(row.category)][metric]
            assert len(vals) == row.n and np.isclose(vals.mean(), row.mean_fraction, atol=1e-12)
            ax.plot(100 * np.array([row.CI_low, row.CI_high]), [y, y], color=COLORS[row.tissue], lw=.9)
            mark(ax, 100 * row.mean_fraction, y, COLORS[row.tissue],
                 shape="o" if row.tissue == "Tumor" else "s", size=3.5)
        ax.set_yticks(range(4), catlabels); ax.set_ylim(3.5, -.5)
        ax.set_xlim(0, 80); ax.set_xticks([0, 20, 40, 60, 80])
        ax.set_xlabel("Fraction (%)")
        ax.grid(axis="x", color="#E4E6E8", lw=.4)
    tissue_legend(fig, height, 211)
    table(a, "Figure4A_source_fractions.tsv"); table(ap, "Figure4A_patients.tsv")

    b, bp = read("source_program"), read("source_program_patients")
    heading(fig, height, "B", "Source-associated programs", 2, 186)
    ax = axis(fig, height, 30, 150, 50, 29)
    forest(ax, b, bp, paired=True)
    ax.set_xlim(-.5, 1.1); ax.set_xticks([-.5, 0, .5, 1])
    ax.set_xlabel("Autologous − independent")
    table(b, "Figure4B_source_programs.tsv"); table(bp, "Figure4B_patients.tsv")

    c, cp = read("viral"), read("viral_patients")
    heading(fig, height, "C", "Viral-reference matches", 93, 186)
    ax = axis(fig, height, 107, 150, 48, 29)
    for tissue, xx in (("Tumor", [0, 1]), ("Normal", [2.4, 3.4])):
        group = cp[cp.tissue.eq(tissue)]; row = c[c.tissue.eq(tissue)].iloc[0]
        assert len(group) == row.n_patients
        assert np.isclose(100 * group.own_fraction.mean(), row.mean_own_pct, atol=1e-12)
        assert np.isclose(100 * group.bulk_fraction.mean(), row.mean_bulk_pct, atol=1e-12)
        col = COLORS[tissue]
        for _, patient in group.iterrows():
            ax.plot(xx, 100 * np.array([patient.own_fraction, patient.bulk_fraction]),
                    color=col, lw=.45, alpha=.4,
                    marker="o" if tissue == "Tumor" else "s", ms=1.9)
        for x, y in zip(xx, [row.mean_own_pct, row.mean_bulk_pct]):
            mark(ax, x, y, col, shape="D", size=4)
        label = "q<.001" if row.q < .001 else f"q={row.q:.3f}".replace("0.", ".")
        ax.text(np.mean(xx), 8.4, label, ha="center", color=col, fontsize=8)
    ax.set_xlim(-.3, 3.7); ax.set_ylim(0, 9); ax.set_yticks([0, 2, 4, 6, 8])
    ax.set_xticks([0, 1, 2.4, 3.4], ["Auto", "Ind", "Auto", "Ind"])
    ax.set_ylabel("Viral matches (%)")
    table(c, "Figure4C_viral_reference_tests.tsv"); table(cp, "Figure4C_patient_pairs.tsv")

    d = read("abundance_program")
    assert len(d) == 12 and d.family_size.eq(12).all()
    heading(fig, height, "D", "Abundance-associated programs", 2, 139)
    ax = axis(fig, height, 30, 103, 50, 29)
    forest(ax, d, paired=True, effect="effect")
    ax.set_xlim(-.26, .16); ax.set_xticks([-.2, 0, .1])
    ax.set_xlabel("High − low program score")
    sig = d[d.tissue.eq("Tumor") & d.program.eq("exhaustion")].iloc[0]
    ax.text(-.24, .65, f"q={sig.q:.3f}".replace("0.", "."), fontsize=8, color=COLORS["Tumor"])
    table(d, "Figure4D_abundance_programs.tsv")

    e, ep = read("abundance_state"), read("abundance_state_patients")
    assert len(e) == 12 and e.n_three_bin_patients.eq(18).all()
    heading(fig, height, "E", "Abundance-associated states", 93, 139)
    long, kde_rows = [], []
    for state, title, xpos in (("c03_CD4_Tn_NR4A2", "NR4A2+ naive\nCD4", 104),
                                ("c20_CD8_Tem_GZMK", "GZMK+ effector-\nmemory CD8", 136)):
        ax = axis(fig, height, xpos, 105, 24, 25)
        for tissue in ("Tumor", "Normal"):
            patient = ep[ep.endpoint.eq(state) & ep.Tissue.eq(tissue)].sort_values("patient_id")
            assert len(patient) == 18
            hh = e[e.endpoint.eq(state) & e.Tissue.eq(tissue)].set_index("abundance_bin")
            for k, binname in enumerate(("low", "middle", "high")):
                values = patient[binname + "_percentage"].to_numpy()
                mean, lo, hi = hh.loc[binname, ["mean_percentage", "CI_low", "CI_high"]]
                assert np.isclose(values.mean(), mean, atol=1e-11)
                direction = -1 if tissue == "Tumor" else 1
                center = k + direction * .035
                grid = np.linspace(values.min(), values.max(), 160)
                density = gaussian_kde(values, bw_method="scott")(grid)
                width = .38 * density / density.max()
                ax.fill_betweenx(grid, center, center + direction * width,
                                 color=COLORS[tissue], alpha=.18, linewidth=0)
                # All dots are actual per-patient proportions; horizontal offsets
                # merely separate their marks and never change vertical values.
                jitter = .07 + np.linspace(0, .25, len(values))
                ax.scatter(center + direction * jitter, values, color=COLORS[tissue],
                           s=4, alpha=.5, lw=0, zorder=3)
                mpos = k + direction * .10
                ax.errorbar(mpos, mean, yerr=[[mean - lo], [hi - mean]],
                            color=COLORS[tissue], marker="o" if tissue == "Tumor" else "s",
                            ms=3, lw=1.0, capsize=1.7, zorder=5)
                for pid, value in zip(patient.patient_id, values):
                    long.append({"endpoint": state, "Tissue": tissue, "patient_id": pid,
                                 "abundance_bin": binname, "state_proportion_percent": value})
                for gy, gd, gw in zip(grid, density, width):
                    kde_rows.append({"endpoint": state, "Tissue": tissue, "abundance_bin": binname,
                                     "state_proportion_percent": gy, "kernel_density": gd,
                                     "normalized_half_width": gw, "bandwidth_rule": "Scott",
                                     "observed_min": values.min(), "observed_max": values.max()})
        ax.set_xticks(range(3), ["Low", "Mid", "High"])
        ax.set_xlim(-.52, 2.52); ax.set_ylim(0, 45); ax.set_yticks([0, 20, 40])
        ax.set_title(title, fontsize=8, pad=3, linespacing=1.05)
        if state.startswith("c03"):
            ax.set_ylabel("State proportion (%)")
    fig.text(132 / WIDTH, 93 / height, "Peripheral-blood bulk TCR abundance", fontsize=8, ha="center")
    table(e, "Figure4E_state_means_CI_tests.tsv"); table(ep, "Figure4E_patient_three_bin_roster.tsv")
    table(pd.DataFrame(long), "Figure4E_patient_distribution.tsv")
    table(pd.DataFrame(kde_rows), "Figure4E_display_kernel_density.tsv")

    f = read("clinical_matching")
    f = f[f.definition.eq("AA_VJ")].copy()
    assert len(f) == 30 and f.family_size.eq(30).all() and f.q.ge(.05).all()
    catalogs = [("GSE236581_Tumor", "Tumor scTCR", 19),
                ("GSE236581_Adjacent", "Adjacent scTCR", 19),
                ("Le2017_primary_tumor", "Primary-tumor bulk", 3)]
    metrics = [("matched_unique_keys", "Matched sequence\ncount"),
               ("matched_identity_fraction", "Distinct-sequence\nfraction"),
               ("matched_read_fraction", "Read fraction")]
    selected, colors = [], np.empty((6, 3))
    for k, (catalog, label, ref_n) in enumerate(catalogs):
        for j, (cohort, cohort_label, model_n) in enumerate(
                [("home", "Study", 126), ("external", "Comparison", 163)]):
            for col, (metric, title) in enumerate(metrics):
                row = f[f.catalog.eq(catalog) & f.metric.eq(metric) & f.cohort.eq(cohort)].iloc[0].copy()
                assert row.n == model_n and row.CI_low > 0 and row.CI_high >= row.CI_low
                row["display_row"] = 2 * k + j
                row["display_column"] = col
                row["reference_catalog_n"] = ref_n
                row["blood_model_n"] = model_n
                row["log2_OR_color"] = np.log2(row.OR_per_SD)
                row["display_OR_CI"] = f"{row.OR_per_SD:.2f} [{row.CI_low:.2f}, {row.CI_high:.2f}]"
                row["CI_includes_null_1"] = bool(row.CI_low <= 1 <= row.CI_high)
                row["corrected_association"] = bool(row.q < .05)
                selected.append(row)
                colors[2 * k + j, col] = row.log2_OR_color
    heading(fig, height, "F", "Adjusted pN+ versus pN0 associations", 2, 85)
    ax = axis(fig, height, 74, 27, 102, 43)
    im = ax.imshow(colors, cmap="RdBu_r", norm=TwoSlopeNorm(vmin=-2, vcenter=0, vmax=2),
                   aspect="auto", interpolation="none")
    ax.set_axis_off()
    for col in range(4):
        ax.axvline(col - .5, color="white", lw=1)
    for row in range(7):
        ax.axhline(row - .5, color="white", lw=1.8 if row in (2, 4) else .7)
    for _, r in pd.DataFrame(selected).iterrows():
        ax.text(r.display_column, r.display_row,
                f"{r.OR_per_SD:.2f}\n[{r.CI_low:.2f}, {r.CI_high:.2f}]",
                ha="center", va="center", fontsize=8.5, linespacing=1.05,
                color="white" if abs(r.log2_OR_color) > 1.2 else "#202020")
    for col, (_, title) in enumerate(metrics):
        fig.text((74 + (col + .5) * 34) / WIDTH, 75 / height, title,
                 ha="center", va="center", fontsize=8, linespacing=1.05)
    fig.text(2 / WIDTH, 75 / height, "Reference catalog\n(participants)",
             fontsize=8, va="center", linespacing=1.05)
    fig.text(39 / WIDTH, 75 / height, "Blood cohort\n(model n)",
             fontsize=8, va="center", linespacing=1.05)
    for k, (_, label, ref_n) in enumerate(catalogs):
        center_y = 70 - (2 * k + 1) * (43 / 6)
        fig.text(2 / WIDTH, center_y / height, f"{label}\n(n = {ref_n})",
                 fontsize=8, va="center", linespacing=1.05)
        for j, (cohort_label, model_n) in enumerate([("Study", 126), ("Comparison", 163)]):
            y = 70 - (2 * k + j + .5) * (43 / 6)
            fig.text(39 / WIDTH, y / height, f"{cohort_label} ({model_n})", fontsize=8, va="center")
    fig.text(2 / WIDTH, 21 / height, "Cell: OR [95% CI] per 1-SD increase", fontsize=8)
    fig.text(2 / WIDTH, 15 / height,
             f"No corrected association (30 tests; all q ≥ {f.q.min():.3f})", fontsize=8)
    fig.text(2 / WIDTH, 9 / height, "CI crosses 1: compatible with no nominal effect", fontsize=8)
    cax = fig.add_axes([121 / WIDTH, 13 / height, 54 / WIDTH, 3 / height])
    cb = fig.colorbar(im, cax=cax, orientation="horizontal", ticks=[-2, -1, 0, 1, 2])
    cb.ax.set_xticklabels(["0.25", "0.5", "1", "2", "4"])
    cb.ax.tick_params(labelsize=8, width=.4, length=2, pad=1)
    cb.outline.set_linewidth(.4)
    cb.ax.set_title("OR color (log2 scale; null = 1)", fontsize=8, pad=3)
    table(pd.DataFrame(selected), "Figure4F_effect_matrix.tsv")
    table(f, "Figure4F_complete_30_test_family.tsv")
    mapping = {"A": "v19 Figure4A, scientific quantities unchanged",
               "B": "v19 Figure4B, scientific quantities unchanged",
               "C": "v19 Figure4C, scientific quantities unchanged",
               "D": "v19 Figure4D, scientific quantities unchanged",
               "E": "v19 Figure4E, scientific quantities unchanged",
               "F": "v19 Figure4F/G/H, all18 estimates in annotated6x3 effect matrix"}
    return export(fig, 4, height, mapping)


def build():
    prepare()
    plt.rcParams.update({"font.family": "Arial", "font.size": 8,
                         "axes.labelsize": 8, "axes.titlesize": 9,
                         "xtick.labelsize": 8, "ytick.labelsize": 8,
                         "legend.fontsize": 8, "svg.fonttype": "none", "pdf.fonttype": 42,
                         "axes.linewidth": .6, "savefig.facecolor": "white"})
    receipt = figure4()
    (OUT / "build_receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, default=BASE)
    args = parser.parse_args(); BASE = args.baseline
    build()
