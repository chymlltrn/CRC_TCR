#!/usr/bin/env python3
"""Frozen-source Figure 4 redraw: graphical pN association panel, unchanged a-e.

Reproduce from the figure-specific source directory with no access to the
original machine. This renderer performs no new regression or significance
selection. ORs remain regression effects; no flow or Sankey encoding is used.
"""
from __future__ import annotations
from release_config import release_path, private_value
import argparse
import datetime
import hashlib
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.text import Text
from matplotlib.ticker import NullLocator
import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde
from PIL import Image

WIDTH = 174
LAYOUT_WIDTH = 180
HEIGHT_MM = 232
TOP_SHIFT = HEIGHT_MM - 232
PROGRAMS = ["cytotoxic", "exhaustion", "naive_memory", "interferon", "residency", "proliferation"]
LABELS = ["Cytotoxicity", "Exhaustion", "Naive/memory", "Interferon", "Residency", "Proliferation"]
COLORS = {"Tumor": "#D55E00", "Normal": "#0072B2"}
INPUTS = {
    "fractions": "Figure4A_source_fractions.tsv",
    "fractions_patients": "Figure4A_patients.tsv",
    "source_program": "Figure4B_source_programs.tsv",
    "source_program_patients": "Figure4B_patients.tsv",
    "viral": "Figure4C_viral_reference_tests.tsv",
    "viral_patients": "Figure4C_patient_pairs.tsv",
    "abundance_program": "Figure4D_abundance_programs.tsv",
    "abundance_state": "Figure4E_state_means_CI_tests.tsv",
    "abundance_state_patients": "Figure4E_patient_three_bin_roster.tsv",
    "clinical_matching": "Figure4F_complete_30_test_family.tsv",
    "selected_estimates": "Figure4F_effect_matrix.tsv",
}
DATA_ROOT = OUTPUT_DIR = QA_DIR = None
TABLES = {}

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def read(key):
    return pd.read_csv(DATA_ROOT / INPUTS[key], sep="\t")

def table(frame, name):
    # QA tables stay out of the frozen source directory.
    frame.to_csv(QA_DIR / name, sep="\t", index=False, na_rep="NA")
    TABLES[name] = {"rows": len(frame), "sha256": sha(QA_DIR / name)}

def axis(fig, height, x, y, w, h):
    ax = fig.add_axes([x / LAYOUT_WIDTH, (y + TOP_SHIFT) / height, w / LAYOUT_WIDTH, h / height])
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(width=.6, length=2, pad=2)
    return ax


def heading(fig, height, letter, title, x, y):
    fig.text(x / LAYOUT_WIDTH, (y + TOP_SHIFT) / height, letter.lower(), fontsize=11, fontweight="bold", va="baseline")
    fig.text((x + 5) / LAYOUT_WIDTH, (y + TOP_SHIFT) / height, title, fontsize=9.5, fontweight="bold", va="baseline")


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
               loc="center left", bbox_to_anchor=(158 / LAYOUT_WIDTH, (y + TOP_SHIFT) / height), ncol=1,
               frameon=False, handlelength=1.0, handletextpad=.35,
               labelspacing=.6, borderaxespad=0)


def export(fig, height, mapping, comparison):
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
    assert all(r["inside_canvas"] for r in bounds), [r for r in bounds if not r["inside_canvas"]]
    assert min(r["fontsize_pt"] for r in bounds) >= 8
    base = OUTPUT_DIR / "Figure4"
    for ext in ("pdf", "png"):
        fig.savefig(base.with_suffix("." + ext), dpi=300,
                    metadata={"Creator": "CRC-TCR 20261008 Figure4 frozen-source redraw"} if ext == "pdf" else None)
    fig.savefig(base.with_suffix(".tif"), dpi=600, pil_kwargs={"compression": "tiff_lzw"})
    with Image.open(base.with_suffix(".tif")) as rgba:
        rgb = rgba.convert("RGB")
    rgb.save(base.with_suffix(".tif"), compression="tiff_lzw", dpi=(600, 600))
    with Image.open(base.with_suffix(".png")) as img:
        # The complete lower panel (including metric labels and inference note).
        bottom_panel_height = int(round(88 / height * img.height))
        img.crop((0, img.height - bottom_panel_height, img.width, img.height)).save(QA_DIR / "Figure4f_preview.png")
    with Image.open(base.with_suffix(".tif")) as img:
        tiff_info = {"pixels": list(img.size), "dpi": [float(x) for x in img.info.get("dpi", ())],
                     "mode": img.mode, "compression": img.info.get("compression")}
    plt.close(fig)
    receipt = {
        "status": "GENERATED_AWAITING_VISUAL_REVIEW", "figure": 4,
        "prepared_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "dimensions_mm": [WIDTH, height], "minimum_text_pt": min(r["fontsize_pt"] for r in bounds),
        "all_text_inside_canvas": all(r["inside_canvas"] for r in bounds),
        "text_bounds": bounds, "panel_mapping": mapping,
        "no_new_endpoint_tests": True, "original_means_intervals_P_q_preserved": True,
        "panel_f_estimates_verified": len(comparison), "complete_BH_family": 30,
        "panel_f_exact_equality_columns": ["OR_per_SD", "CI_low", "CI_high", "P", "q", "n", "family_size"],
        "input_sha256": {key: {"file": filename, "sha256": sha(DATA_ROOT / filename)}
                           for key, filename in INPUTS.items()},
        "plotting_code_sha256": sha(__file__), "tiff": tiff_info, "qa_tables": TABLES,
        "artifacts": {ext: {"file": base.with_suffix("." + ext).name,
                            "sha256": sha(base.with_suffix("." + ext))}
                      for ext in ("pdf", "png", "tif")},
        "standalone_preview": {"file": "Figure4f_preview.png", "sha256": sha(QA_DIR / "Figure4f_preview.png")}
    }
    (QA_DIR / "Figure4_redraw_receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    return receipt

def figure4():
    height = HEIGHT_MM
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
    fig.text(132 / LAYOUT_WIDTH, (93 + TOP_SHIFT) / height, "Peripheral-blood bulk TCR abundance", fontsize=8, ha="center")
    table(e, "Figure4E_state_means_CI_tests.tsv"); table(ep, "Figure4E_patient_three_bin_roster.tsv")
    table(pd.DataFrame(long), "Figure4E_patient_distribution.tsv")
    table(pd.DataFrame(kde_rows), "Figure4E_display_kernel_density.tsv")

    f = read("clinical_matching")
    f = f[f.definition.eq("AA_VJ")].copy()
    assert len(f) == 30 and f.family_size.eq(30).all() and f.q.ge(.05).all()
    catalogs = [("GSE236581_Tumor", "Tumor\nscTCR", 19, "#D55E00", "o"),
                ("GSE236581_Adjacent", "Adjacent\nscTCR", 19, "#0072B2", "s"),
                ("Le2017_primary_tumor", "Primary\nbulk", 3, "#66737D", "D")]
    metrics = [("matched_unique_keys", "Identity count",
                "Matched AA/V/J\nidentities"),
               ("matched_identity_fraction", "Identity fraction",
                "Matched / all\nAA/V/J identities"),
               ("matched_read_fraction", "Read fraction",
                "Matched / all\nQC reads")]
    supplied = read("selected_estimates")
    key_cols = ["definition", "cohort", "catalog", "metric"]
    measure_cols = ["OR_per_SD", "CI_low", "CI_high", "P", "q", "n", "family_size"]
    selected = f[f.catalog.isin([r[0] for r in catalogs])].copy()
    assert len(selected) == 18 and len(supplied) == 18
    comparison = selected.merge(supplied, on=key_cols, validate="one_to_one", suffixes=("_render", "_frozen"))
    for col in measure_cols:
        np.testing.assert_array_equal(comparison[col + "_render"], comparison[col + "_frozen"])
    # Six sparse vertical effect glyph plots replace the annotated heatmap.
    # All intervals use the same log OR scale, with a visible null of 1.
    heading(fig, height, "f", "Blood–tissue matching and pN status", 2, 85)
    positions = [37, 84, 131]
    width = 39
    for j, (_, title, definition) in enumerate(metrics):
        center = (positions[j] + width / 2) / LAYOUT_WIDTH
        fig.text(center, 76 / height, title, ha="center", fontsize=8.6,
                 fontweight="bold", va="center")
        fig.text(center, 68 / height, definition, ha="center", fontsize=8,
                 va="center", linespacing=1.05)
    for cohort, label, model_n, y in [("home", "Study", 126, 42),
                                      ("external", "Comparison", 163, 18)]:
        fig.text(16 / LAYOUT_WIDTH, (y + 9.5) / height,
                 f"{label}\n(n = {model_n})", fontsize=8.5, fontweight="bold",
                 ha="center", va="center", linespacing=1.15)
        for j, (metric, _, _) in enumerate(metrics):
            # These coordinates belong to the new panel and are not shifted.
            ax = fig.add_axes([positions[j] / LAYOUT_WIDTH, y / height,
                               width / LAYOUT_WIDTH, 19 / height])
            ax.spines[["top", "right"]].set_visible(False)
            ax.spines[["left", "bottom"]].set_color("#84919B")
            ax.tick_params(width=.6, length=2, pad=2, labelsize=8)
            ax.set_yscale("log")
            ax.set_ylim(.0625, 4.5)
            ax.set_xlim(-.6, 2.6)
            ax.set_yticks([.1, 1, 4], ["0.1", "1", "4"])
            ax.yaxis.set_minor_locator(NullLocator())
            ax.axhline(1, color="#98A3AC", lw=.8, ls=(0, (3, 2)), zorder=1)
            ax.grid(axis="y", color="#EDF0F2", lw=.45, zorder=0)
            for k, (catalog, _, ref_n, color, marker) in enumerate(catalogs):
                row = selected[selected.cohort.eq(cohort) & selected.catalog.eq(catalog)
                               & selected.metric.eq(metric)].iloc[0]
                assert int(row.n) == model_n and 0 < row.CI_low < row.OR_per_SD < row.CI_high
                # Hollow symbols for every result: no BH-corrected association.
                ax.errorbar(k, row.OR_per_SD,
                            yerr=[[row.OR_per_SD - row.CI_low], [row.CI_high - row.OR_per_SD]],
                            fmt=marker, color=color, mfc="white", mec=color,
                            mew=1.0, ms=4.1, lw=1.2, capsize=2.8, capthick=.8,
                            zorder=4)
            ax.set_xticks(range(3), ["Tumor\nscTCR", "Adj.\nscTCR", "Tumor\nbulk"])
            if cohort == "home":
                ax.tick_params(axis="x", labelbottom=False)
            if j:
                ax.tick_params(axis="y", labelleft=False)
    # The actual OR metric carries the pN contrast, not just the panel title.
    fig.text(3.8 / LAYOUT_WIDTH, 42 / height,
             "Adjusted OR for pN+ vs pN0", rotation=90,
             va="center", ha="center", fontsize=8.5)
    fig.text(37 / LAYOUT_WIDTH, 6.5 / height,
             "OR per 1-SD increase in the transformed metric; whiskers: 95% CI", fontsize=8)
    fig.text(37 / LAYOUT_WIDTH, 2 / height,
             f"30 tests: all q ≥ {f.q.min():.3f}. Tumor/adjacent scTCR n=19; primary-tumor bulk n=3", fontsize=8)
    table(selected, "Figure4F_displayed_18_estimates.tsv")
    mapping = {"a": "Frozen v20 Figure4A; quantities and geometry unchanged",
               "b": "Frozen v20 Figure4B; quantities and geometry unchanged",
               "c": "Frozen v20 Figure4C; quantities and geometry unchanged",
               "d": "Frozen v20 Figure4D; quantities and geometry unchanged",
               "e": "Frozen v20 Figure4E; quantities and geometry unchanged",
               "f": "18 unchanged OR/95% CI; six graphical facets; complete 30-test BH family retained"}
    return export(fig, height, mapping, comparison)

def main():
    global DATA_ROOT, OUTPUT_DIR, QA_DIR
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--qa-dir", type=Path, required=True)
    args = parser.parse_args()
    DATA_ROOT, OUTPUT_DIR, QA_DIR = args.data_root.resolve(), args.output_dir.resolve(), args.qa_dir.resolve()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    QA_DIR.mkdir(parents=True, exist_ok=True)
    for file in INPUTS.values():
        if not (DATA_ROOT / file).is_file():
            raise FileNotFoundError(DATA_ROOT / file)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8,
                         "axes.labelsize": 8, "axes.titlesize": 9,
                         "xtick.labelsize": 8, "ytick.labelsize": 8,
                         "legend.fontsize": 8, "svg.fonttype": "none", "pdf.fonttype": 42,
                         "axes.linewidth": .6, "savefig.facecolor": "white"})
    receipt = figure4()
    print(json.dumps({k: v for k, v in receipt.items() if k not in ("text_bounds", "input_sha256", "qa_tables")}, indent=2))

if __name__ == "__main__":
    main()
