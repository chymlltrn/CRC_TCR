#!/usr/bin/env python3
"""Replot v18 Figures 3/4 from frozen v17 patient-level sources.

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
from matplotlib.lines import Line2D
from matplotlib.text import Text
import fitz
import numpy as np
import pandas as pd
from lxml import etree
from scipy.stats import gaussian_kde

ROOT = release_path('render_v20_root')
BASE = ROOT.parent / "revision_v17_figure_qpcr_revision"
OUT = ROOT / "analysis/figure34"
SOURCE = OUT / "source_baseline"
WIDTH = 180
PROGRAMS = ["cytotoxic", "exhaustion", "naive_memory", "interferon", "residency", "proliferation"]
LABELS = ["Cytotoxicity", "Exhaustion", "Naive/memory", "Interferon", "Residency", "Proliferation"]
COLORS = {"Tumor": "#D55E00", "Normal": "#0072B2"}
INPUTS = {
    "state_composition": "analysis/figure4/A_state_composition_plot_source.tsv",
    "expansion": "analysis/figure4/B_expansion_program_plot_source.tsv",
    "expansion_patients": "analysis/figure4/B_patient_points.tsv",
    "interaction": "analysis/figure4/C_expansion_interaction_plot_source.tsv",
    "interaction_patients": "analysis/figure4/C_patient_points.tsv",
    "sharing": "manuscript/SourceData/sharing/statistics.tsv",
    "sharing_patients": "manuscript/SourceData/sharing/patient_contrasts.tsv",
    "ownblood": "manuscript/SourceData/tissue/ownblood_program_tests.tsv",
    "ownblood_patients": "manuscript/SourceData/tissue/ownblood_program_patient_contrasts.tsv",
    "fractions": "analysis/figure4/D_source_fraction_plot_source.tsv",
    "fractions_patients": "analysis/figure4/D_patient_source_fractions.tsv",
    "source_program": "analysis/figure4/E_source_program_plot_source.tsv",
    "source_program_patients": "analysis/figure4/E_patient_points.tsv",
    "viral": "analysis/figure4/F_viral_reference_test_source.tsv",
    "viral_patients": "analysis/figure4/F_patient_pairs.tsv",
    "abundance_program": "analysis/figure4/G_abundance_program_plot_source.tsv",
    "abundance_state": "analysis/figure4/H_abundance_state_plot_source.tsv",
    "abundance_state_patients": "analysis/figure4/H_patient_bin_roster.tsv",
    "clinical_matching": "manuscript/SourceData/matching/bulk_reference_N_adjusted_tests.tsv",
}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def prepare():
    OUT.mkdir(parents=True, exist_ok=True)
    SOURCE.mkdir(parents=True, exist_ok=True)
    (ROOT / "figures").mkdir(exist_ok=True)
    manifest = []
    for key, rel in INPUTS.items():
        original = BASE / rel
        copied = SOURCE / f"{key}.tsv"
        if not copied.exists():
            shutil.copyfile(original, copied)
        if original.exists():
            assert sha(original) == sha(copied), key
        manifest.append({"key": key, "v17_source": str(original),
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
               loc="lower right", bbox_to_anchor=(.985, y / height), ncol=2,
               frameon=False, handlelength=1.3, handletextpad=.4,
               columnspacing=1.0, borderaxespad=0)


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
    base = ROOT / "figures" / f"new_figure{ordinal}"
    for ext in ("svg", "pdf", "png"):
        fig.savefig(base.with_suffix("." + ext), dpi=300,
                    metadata={"Creator": "CRC-TCR v18 frozen-source figure replot"} if ext == "pdf" else None)
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


def figure3():
    height = 220
    fig = plt.figure(figsize=(WIDTH / 25.4, height / 25.4))
    a = read("state_composition").sort_values("subcelltype").reset_index(drop=True)
    assert len(a) == 32 and a.n.eq(19).all() and a.family_size.eq(32).all()
    heading(fig, height, "A", "Tissue states", 2, 216)
    ax = axis(fig, height, 47, 109, 30, 101)
    for k, r in a.iterrows():
        ax.plot([r.ci_low_pp, r.ci_high_pp], [k, k], color="#454545", lw=.8)
        mark(ax, r.mean_difference_pp, k, "#454545", r.q_bh, size=3.2)
    ax.axvline(0, color="#A7A7A7", lw=.6)
    ax.set_yticks(range(32), a.subcelltype.str.replace("_", " ", regex=False))
    ax.set_ylim(31.6, -.6); ax.set_xlim(-18, 12); ax.set_xticks([-15, 0, 10])
    ax.set_xlabel("Tumor − adjacent (pp)")
    ax.grid(axis="x", color="#E4E6E8", lw=.4)
    table(a, "Figure3A_state_composition.tsv")

    b, bp = read("expansion"), read("expansion_patients")
    heading(fig, height, "B", "Local expansion", 87, 216)
    ax = axis(fig, height, 115, 175, 47, 31)
    forest(ax, b, bp, low="ci_low", high="ci_high", pcol="p_value", qcol="q_bh", p_labels=True)
    ax.set_xlim(-.75, 1); ax.set_xticks([-.5, 0, .5, 1])
    ax.set_xlabel("Expanded − singleton")
    table(b, "Figure3B_expansion.tsv"); table(bp, "Figure3B_patients.tsv")

    c, cp = read("interaction"), read("interaction_patients")
    heading(fig, height, "C", "Tissue contrast", 87, 160)
    ax = axis(fig, height, 115, 119, 47, 31)
    forest(ax, c, cp, color="#8064A7", p_labels=True)
    ax.set_xlim(-.5, .8); ax.set_xticks([-.5, 0, .5])
    ax.set_xlabel("Tumor − adjacent difference")
    table(c, "Figure3C_interaction.tsv"); table(cp, "Figure3C_patients.tsv")

    d = read("sharing"); d = d[d.analysis_id.eq("primary_NT_VJ_500")].copy()
    dp = read("sharing_patients"); dp = dp[dp.analysis_id.eq("primary_NT_VJ_500")].copy()
    heading(fig, height, "D", "Excess autologous sharing", 2, 98)
    ax = axis(fig, height, 73, 61, 88, 30)
    labels, plotted = [], []
    for metric in ("cell_fraction", "clone_fraction"):
        unit = "Cells" if metric == "cell_fraction" else "Clonotypes"
        for tissue, label in (("Tumor", "Tumor"), ("Normal", "Adjacent"),
                              ("Tumor_minus_Normal", "Tumor − adjacent")):
            k = len(labels)
            labels.append(f"{unit}: {label}")
            r = d[(d.metric == metric) & d.contrast.eq(tissue + "_self_excess")].iloc[0]
            if tissue == "Tumor_minus_Normal":
                v = dp[dp.metric.eq(metric)].pivot(index="patient_id", columns="Tissue", values="self_excess")
                values = v.Tumor - v.Normal
            else:
                values = dp[dp.metric.eq(metric) & dp.Tissue.eq(tissue)].set_index("patient_id").self_excess
            values = values.sort_index()
            assert len(values) == r.n_patients == 18
            assert np.isclose(values.mean(), r.mean_effect, atol=1e-12)
            ax.scatter(100 * values, k + np.linspace(-.12, .12, len(values)),
                       color="#454545", s=5, alpha=.3, lw=0)
            ax.plot(100 * np.array([r.ci_low, r.ci_high]), [k, k], color="#454545", lw=1.0)
            mark(ax, 100 * r.mean_effect, k, "#454545")
            ax.text(1.025, k, pstring(r.P), fontsize=8, va="center", transform=ax.get_yaxis_transform())
            for patient, value in values.items():
                plotted.append({"patient_id": patient, "metric": metric, "contrast": r.contrast,
                                "effect_pp": 100 * value})
    ax.set_yticks(range(6), labels); ax.set_ylim(5.55, -.55)
    ax.set_xlim(-2.2, 21.5); ax.set_xticks([0, 5, 10, 15, 20])
    ax.axvline(0, color="#A7A7A7", lw=.6)
    ax.grid(axis="x", color="#E4E6E8", lw=.4)
    ax.set_xlabel("Autologous − other-donor blood sharing (pp)")
    table(d, "Figure3D_autologous_sharing.tsv")
    table(pd.DataFrame(plotted), "Figure3D_patient_points.tsv")

    own = read("ownblood")
    own = own[own.threshold.eq(1) & own.weighting.eq("clone_equal")].copy()
    op = read("ownblood_patients")
    op = op[op.threshold.eq(1) & op.weighting.eq("clone_equal") & op.eligible].copy()
    for letter, tissue, x, title in (("E", "Tumor", 30, "Tumor blood detection"),
                                    ("F", "Normal", 122, "Adjacent blood detection")):
        heading(fig, height, letter, title, 2 if tissue == "Tumor" else 92, 47)
        ax = axis(fig, height, x, 11, 39, 29)
        rows, patients = own[own.tissue.eq(tissue)], op[op.tissue.eq(tissue)]
        forest(ax, rows, patients, p_labels=True)
        ax.set_xlim(-.4, 1.1); ax.set_xticks([0, .5, 1])
        ax.set_xlabel("Detected − not detected")
        table(rows, f"Figure3{letter}_ownblood_programs.tsv")
        table(patients, f"Figure3{letter}_patients.tsv")
    mapping = {"A": "v17 Figure4A", "B": "v17 Figure4B", "C": "v17 Figure4C",
               "D": "v17 Figure3D", "E": "v17 Figure3E", "F": "v17 Figure3F"}
    return export(fig, 3, height, mapping)


def figure4():
    height = 240
    fig = plt.figure(figsize=(WIDTH / 25.4, height / 25.4))
    a, ap = read("fractions"), read("fractions_patients")
    assert len(a) == 16 and a.n.eq(18).all()
    heading(fig, height, "A", "Blood-source detection", 2, 235)
    categories = ["both", "own_only", "bulk_only", "neither"]
    catlabels = ["Both", "Autologous only", "Independent only", "Neither"]
    for metric, xpos, title in (("cell_fraction", 43, "Cells"),
                                ("clone_fraction", 130, "Clonotypes")):
        ax = axis(fig, height, xpos, 203, 44, 24)
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
    tissue_legend(fig, height, 233)
    table(a, "Figure4A_source_fractions.tsv"); table(ap, "Figure4A_patients.tsv")

    b, bp = read("source_program"), read("source_program_patients")
    heading(fig, height, "B", "Source-associated programs", 2, 189)
    ax = axis(fig, height, 30, 150, 55, 32)
    forest(ax, b, bp, paired=True)
    ax.set_xlim(-.5, 1.1); ax.set_xticks([-.5, 0, .5, 1])
    ax.set_xlabel("Autologous − independent")
    table(b, "Figure4B_source_programs.tsv"); table(bp, "Figure4B_patients.tsv")

    c, cp = read("viral"), read("viral_patients")
    heading(fig, height, "C", "Viral-reference matches", 96, 189)
    ax = axis(fig, height, 109, 150, 66, 32)
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
    heading(fig, height, "D", "Abundance-associated programs", 2, 136)
    ax = axis(fig, height, 30, 96, 55, 33)
    forest(ax, d, paired=True, effect="effect")
    ax.set_xlim(-.26, .16); ax.set_xticks([-.2, 0, .1])
    ax.set_xlabel("High − low program score")
    sig = d[d.tissue.eq("Tumor") & d.program.eq("exhaustion")].iloc[0]
    ax.text(-.24, .65, f"q={sig.q:.3f}".replace("0.", "."), fontsize=8, color=COLORS["Tumor"])
    table(d, "Figure4D_abundance_programs.tsv")

    e, ep = read("abundance_state"), read("abundance_state_patients")
    assert len(e) == 12 and e.n_three_bin_patients.eq(18).all()
    heading(fig, height, "E", "Abundance-associated states", 96, 136)
    long, kde_rows = [], []
    for state, title, xpos in (("c03_CD4_Tn_NR4A2", "NR4A2+ naive\nCD4", 111),
                                ("c20_CD8_Tem_GZMK", "GZMK+ effector-\nmemory CD8", 151)):
        ax = axis(fig, height, xpos, 98, 27, 26)
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
    fig.text(138 / WIDTH, 85 / height, "Peripheral-blood bulk TCR abundance", fontsize=8, ha="center")
    table(e, "Figure4E_state_means_CI_tests.tsv"); table(ep, "Figure4E_patient_three_bin_roster.tsv")
    table(pd.DataFrame(long), "Figure4E_patient_distribution.tsv")
    table(pd.DataFrame(kde_rows), "Figure4E_display_kernel_density.tsv")

    f = read("clinical_matching")
    f = f[f.definition.eq("AA_VJ")].copy()
    catalogs = [("GSE236581_Tumor", "Tumor scTCR"),
                ("GSE236581_Adjacent", "Adjacent scTCR"),
                ("Le2017_primary_tumor", "Primary tumor bulk")]
    metrics = [("matched_unique_keys", "Matched sequence\ncount"),
               ("matched_identity_fraction", "Fraction of\nsequences"),
               ("matched_read_fraction", "Fraction of\nreads")]
    selected = []
    for letter, (metric, title), xpos in zip("FGH", metrics, (53, 96, 139)):
        heading(fig, height, letter, "", 2 if letter == "F" else xpos - 4, 77)
        ax = axis(fig, height, xpos, 28, 35, 38)
        ax.set_title(title, fontsize=8, pad=4, linespacing=1.05)
        for k, (catalog, label) in enumerate(catalogs):
            for cohort, col, shape, dy in (("home", "#0072B2", "o", -.13),
                                           ("external", "#D55E00", "s", .13)):
                row = f[f.catalog.eq(catalog) & f.metric.eq(metric) & f.cohort.eq(cohort)].iloc[0]
                assert row.n == (126 if cohort == "home" else 163)
                ax.plot([row.CI_low, row.CI_high], [k + dy, k + dy], color=col, lw=1.0)
                mark(ax, row.OR_per_SD, k + dy, col, q=1, shape=shape, size=3.6)
                selected.append(row)
        ax.axvline(1, color="#A7A7A7", lw=.6, ls=(0, (4, 3)))
        ax.set_xscale("log"); ax.set_xlim(.08, 5); ax.set_xticks([.1, .3, 1, 3], ["0.1", "0.3", "1", "3"])
        ax.minorticks_off(); ax.set_ylim(2.6, -.6)
        ax.set_yticks(range(3), [label for _, label in catalogs] if letter == "F" else [])
    fig.text(.5, 15 / height, "Adjusted OR for pN+ versus pN0 (per 1-SD increase)", fontsize=8, ha="center")
    fig.legend(handles=[Line2D([], [], color="#0072B2", marker="o", mfc="white", ms=3,
                               lw=1, label="Study cohort"),
                        Line2D([], [], color="#D55E00", marker="s", mfc="white", ms=3,
                               lw=1, label="Comparison cohort")],
               loc="lower center", bbox_to_anchor=(.62, 2 / height), ncol=2,
               frameon=False, handlelength=1.5, handletextpad=.5,
               columnspacing=1.1, borderaxespad=0)
    table(pd.DataFrame(selected), "Figure4FGH_clinical_matching_model_estimates.tsv")
    table(f, "Figure4FGH_complete_30_test_family.tsv")
    mapping = {"A": "v17 Figure4D", "B": "v17 Figure4E", "C": "v17 Figure4F",
               "D": "v17 Figure4G", "E": "v17 Figure4H", "F": "v17 Figure3A",
               "G": "v17 Figure3B", "H": "v17 Figure3C"}
    return export(fig, 4, height, mapping)


def build():
    prepare()
    plt.rcParams.update({"font.family": "Arial", "font.size": 8,
                         "axes.labelsize": 8, "axes.titlesize": 9,
                         "xtick.labelsize": 8, "ytick.labelsize": 8,
                         "legend.fontsize": 8, "svg.fonttype": "none", "pdf.fonttype": 42,
                         "axes.linewidth": .6, "savefig.facecolor": "white"})
    receipts = [figure3(), figure4()]
    (OUT / "build_receipt.json").write_text(json.dumps(receipts, indent=2) + "\n")
    print(json.dumps(receipts, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, default=BASE)
    args = parser.parse_args(); BASE = args.baseline
    build()
