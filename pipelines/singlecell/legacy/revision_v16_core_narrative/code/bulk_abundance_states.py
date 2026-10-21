#!/usr/bin/env python3
"""Source-bound independent-bulk abundance bins versus patient-level tissue states."""
from __future__ import annotations
from release_config import release_path, private_value
import os
for key in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(key, "2")
import argparse
import hashlib
import json
import platform
from pathlib import Path
import warnings
import numpy as np
import pandas as pd
from scipy.stats import friedmanchisquare, wilcoxon
from statsmodels.stats.multitest import multipletests

ROOT = release_path('abundance_root')
BASE = ROOT.parent
OUT = ROOT / "analysis"
SEED = 20261006
BOOT = 10000
PROGRAMS = ["cytotoxic", "exhaustion", "naive_memory", "interferon", "residency", "proliferation"]
EXPOSURES = ["combined_mean", "study_mean", "external_mean", "positive_carrier_mean"]
BINS = ["unmatched", "low", "middle", "high"]
WEIGHTS = ["cell_equal", "clone_equal"]
SUPPORT_SHA = "3ed5bc4318c36ee9ff5317fe6fc2d19f7633ce12211e406cae389e1b48ff15a3"
REMOTE_SUPPORT = str(release_path('bulk_abundance_states_source_28'))

def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

def write(name, frame):
    frame.to_csv(OUT / name, sep="\t", index=False, na_rep="NA")

def random(token):
    return np.random.default_rng((SEED + int(hashlib.sha256(token.encode()).hexdigest()[:8], 16)) % 2**32)

def json_write(name, obj):
    (OUT / name).write_text(json.dumps(obj, indent=2, allow_nan=False) + "\n")

def sources():
    return {
        "single_cell_cache": BASE / "revision_v6/cohort/baseline_cells.parquet",
        "selected_roster": BASE / "revision_v5/bulk_sc_match/ingest/selected_library_roster.tsv",
        "bulk_patient_denominators": BASE / "revision_v5/bulk_sc_match/ingest/patient_coverage.tsv",
        "complete_EOF_receipt": BASE / "revision_v5/bulk_sc_match/ingest/ingest_receipt.json",
        "bulk_support": OUT / "inputs/bulk_support.parquet",
        "original_ingest_code": BASE / "revision_v5/code/bulk_sc_match_ingest.py",
        "protocol": OUT / "PROTOCOL.md",
        "analysis_code": Path(__file__).resolve(),
    }

def inputs():
    paths = sources()
    assert sha(paths["bulk_support"]) == SUPPORT_SHA
    receipt = json.loads(paths["complete_EOF_receipt"].read_text())
    assert receipt["full_EOF"] and receipt["independent_blood_patients"] == 339
    roster = pd.read_csv(paths["selected_roster"], sep="\t")
    assert len(roster) == 339 and roster.patient_key.is_unique
    assert roster.groupby("cohort").size().to_dict() == {"external": 202, "home": 137}
    den = pd.read_csv(paths["bulk_patient_denominators"], sep="\t")
    den = den[den.definition.eq("AA_VJ") & den.read_threshold.eq(2)].copy()
    assert len(den) == 339 and den.patient_key.is_unique
    assert set(den.patient_key) == set(roster.patient_key) and den.qc_reads.gt(0).all()
    columns = ["cell_name", "patient_id", "Tissue", "SubCellType", "beta_clone", "key_AA_VJ", "is_primary_cell",
               "is_cd8", "positive_expression_qc", "match_AA_VJ_union_ge2"] + ["program_" + p for p in PROGRAMS]
    cells = pd.read_parquet(paths["single_cell_cache"], columns=columns)
    cells = cells[cells.is_primary_cell & cells.Tissue.isin(["Tumor", "Normal"])].copy()
    assert len(cells) == 53886 and cells.cell_name.is_unique and cells.patient_id.nunique() == 19
    assert cells.groupby("Tissue").size().to_dict() == {"Normal": 24865, "Tumor": 29021}
    assert cells.key_AA_VJ.notna().all() and cells.beta_clone.notna().all()
    assert cells.SubCellType.nunique() == 32
    assert cells.groupby("beta_clone").key_AA_VJ.nunique().max() == 1
    support = pd.read_parquet(paths["bulk_support"], filters=[("definition", "=", "AA_VJ")],
                              columns=["patient_key", "cohort", "sample_id", "definition", "match_key", "read_sum"])
    assert not support.duplicated(["patient_key", "match_key"]).any()
    assert set(support.patient_key) == set(roster.patient_key)
    support = support[support.match_key.isin(set(cells.key_AA_VJ))].copy()
    support = support.merge(den[["patient_key", "qc_reads"]], on="patient_key", validate="many_to_one")
    assert support.read_sum.le(support.qc_reads).all()
    return cells, support, den

def exposure_scores(cells, support):
    keys = pd.Index(sorted(cells.key_AA_VJ.unique()), name="match_key")
    scores = pd.DataFrame(index=keys)
    detected = support[support.read_sum.ge(2)].copy()
    detected["normalized_ppm"] = 1e6 * detected.read_sum / detected.qc_reads
    for label, cohort, denominator in [("combined_mean", None, 339), ("study_mean", "home", 137),
                                        ("external_mean", "external", 202)]:
        x = detected if cohort is None else detected[detected.cohort.eq(cohort)]
        scores[label] = x.groupby("match_key").normalized_ppm.sum().reindex(keys, fill_value=0) / denominator
        scores[label + "_carriers"] = x.groupby("match_key").patient_key.nunique().reindex(keys, fill_value=0)
    scores["positive_carrier_mean"] = detected.groupby("match_key").normalized_ppm.mean().reindex(keys, fill_value=0)
    expected = set(cells.loc[cells.match_AA_VJ_union_ge2, "key_AA_VJ"])
    assert set(scores.index[scores.combined_mean.gt(0)]) == expected
    assert len(keys) == 33915 and len(expected) == 9764
    cuts = []
    for label in EXPOSURES:
        positive = scores.loc[scores[label].gt(0), label]
        lower, upper = positive.quantile([1 / 3, 2 / 3]).to_numpy()
        assert 0 < lower < upper
        scores[label + "_bin"] = np.select([scores[label].eq(0), scores[label].le(lower), scores[label].le(upper)],
                                             ["unmatched", "low", "middle"], default="high")
        cuts.append(dict(exposure=label, lower_ppm=float(lower), upper_ppm=float(upper),
                         matched_identities=len(positive), unmatched_identities=len(keys) - len(positive),
                         low_identities=int(scores[label + "_bin"].eq("low").sum()),
                         middle_identities=int(scores[label + "_bin"].eq("middle").sum()),
                         high_identities=int(scores[label + "_bin"].eq("high").sum())))
    return scores.reset_index(), pd.DataFrame(cuts)

def membership(cells, scores):
    keep = ["match_key"] + [p + "_bin" for p in EXPOSURES]
    return cells.merge(scores[keep], left_on="key_AA_VJ", right_on="match_key", validate="many_to_one")

def eligibility(cells):
    rows = []
    for exposure in EXPOSURES:
        for (pid, tissue, abin), group in cells.groupby(["patient_id", "Tissue", exposure + "_bin"], observed=True):
            cd8 = group[group.is_cd8 & group.positive_expression_qc]
            row = dict(exposure=exposure, patient_id=pid, Tissue=tissue, abundance_bin=abin,
                       n_T_cells=len(group), n_T_clonotypes=group.beta_clone.nunique(),
                       n_CD8_cells=len(cd8), n_CD8_clonotypes=cd8.beta_clone.nunique())
            row["state_eligible"] = row["n_T_cells"] >= 20 and row["n_T_clonotypes"] >= 5
            row["program_eligible"] = row["n_CD8_cells"] >= 20 and row["n_CD8_clonotypes"] >= 5
            rows.append(row)
    frame = pd.DataFrame(rows)
    full = pd.MultiIndex.from_product([EXPOSURES, sorted(cells.patient_id.unique()), ["Tumor", "Normal"], BINS],
                                      names=["exposure", "patient_id", "Tissue", "abundance_bin"])
    frame = frame.set_index(list(full.names)).reindex(full).reset_index()
    counts = ["n_T_cells", "n_T_clonotypes", "n_CD8_cells", "n_CD8_clonotypes"]
    frame[counts] = frame[counts].fillna(0).astype(int)
    frame[["state_eligible", "program_eligible"]] = frame[["state_eligible", "program_eligible"]].fillna(False).astype(bool)
    return frame

def preflight():
    OUT.mkdir(parents=True, exist_ok=True)
    cells, support, den = inputs()
    scores, cutpoints = exposure_scores(cells, support)
    elig = eligibility(membership(cells, scores))
    write("identity_abundance_scores.tsv", scores)
    write("abundance_cutpoints.tsv", cutpoints)
    write("patient_bin_eligibility.tsv", elig)
    write("bulk_patient_denominators.tsv", den)
    manifest = pd.DataFrame([dict(role=role, path=str(path), bytes=path.stat().st_size, sha256=sha(path),
                                  original_remote_path=REMOTE_SUPPORT if role == "bulk_support" else "")
                             for role, path in sources().items()])
    write("input_source_manifest.tsv", manifest)
    derived = ["identity_abundance_scores.tsv", "abundance_cutpoints.tsv", "patient_bin_eligibility.tsv", "bulk_patient_denominators.tsv"]
    json_write("pretest_freeze.json", dict(stage="before_endpoint_tests", seed=SEED, bootstrap=BOOT,
               independent_bulk_patients=339, single_cell_patients=19, tissue_T_cells=53886,
               tissue_AA_VJ_identities=33915, bulk_matched_identities_ge2=9764,
               protocol_sha256=sha(OUT / "PROTOCOL.md"), code_sha256=sha(__file__),
               input_hashes={role: sha(path) for role, path in sources().items()},
               derived_hashes={name: sha(OUT / name) for name in derived},
               eligibility_rule="20 cells and 5 NT/V/J clonotypes in each bin; 8 patients for inference"))
    print(cutpoints.to_string(index=False), flush=True)
    print(elig.groupby(["exposure", "Tissue", "abundance_bin"])[["state_eligible", "program_eligible"]].sum().to_string(), flush=True)

def validate_freeze():
    frozen = json.loads((OUT / "pretest_freeze.json").read_text())
    for role, path in sources().items():
        assert sha(path) == frozen["input_hashes"][role], ("changed input", role)
    for name, expected in frozen["derived_hashes"].items():
        assert sha(OUT / name) == expected, ("changed pretest derivative", name)
    return frozen

def endpoint_summaries(cells, elig):
    states = sorted(cells.SubCellType.unique())
    rows = []
    for exposure in EXPOSURES:
        for (pid, tissue, abin), group in cells.groupby(["patient_id", "Tissue", exposure + "_bin"], observed=True):
            cd8 = group[group.is_cd8 & group.positive_expression_qc]
            for weighting in WEIGHTS:
                if weighting == "cell_equal":
                    fractions = group.SubCellType.value_counts().reindex(states, fill_value=0) / len(group)
                    means = cd8[["program_" + p for p in PROGRAMS]].mean()
                else:
                    state_by_clone = pd.crosstab(group.beta_clone, group.SubCellType, normalize="index")
                    fractions = state_by_clone.reindex(columns=states, fill_value=0).mean()
                    means = cd8.groupby("beta_clone")[["program_" + p for p in PROGRAMS]].mean().mean()
                assert np.isclose(fractions.sum(), 1)
                for state in states:
                    rows.append(dict(exposure=exposure, weighting=weighting, patient_id=pid, Tissue=tissue,
                                     abundance_bin=abin, endpoint_type="state", endpoint=state, value=float(fractions[state])))
                for program in PROGRAMS:
                    rows.append(dict(exposure=exposure, weighting=weighting, patient_id=pid, Tissue=tissue,
                                     abundance_bin=abin, endpoint_type="program", endpoint=program,
                                     value=float(means["program_" + program])))
    data = pd.DataFrame(rows).merge(elig, on=["exposure", "patient_id", "Tissue", "abundance_bin"], validate="many_to_one")
    data["eligible"] = np.where(data.endpoint_type.eq("state"), data.state_eligible, data.program_eligible)
    assert not data.duplicated(["exposure", "weighting", "patient_id", "Tissue", "abundance_bin", "endpoint_type", "endpoint"]).any()
    return data

def run_tests(data):
    rows = []
    for keys, group in data.groupby(["exposure", "weighting", "endpoint_type", "Tissue", "endpoint"], sort=True):
        exposure, weighting, etype, tissue, endpoint = keys
        z = group[group.eligible].pivot(index="patient_id", columns="abundance_bin", values="value").reindex(columns=BINS)
        pair = z[["low", "high"]].dropna()
        triple = z[["low", "middle", "high"]].dropna()
        size = 64 if etype == "state" else 12
        scale = 100 if etype == "state" else 1
        for test, n in [("high_minus_low", len(pair)), ("friedman_three_bins", len(triple))]:
            row = dict(exposure=exposure, weighting=weighting, endpoint_type=etype, Tissue=tissue, endpoint=endpoint,
                       test=test, n=n, family="|".join([exposure, weighting, etype, test]), family_size=size,
                       effect_unit="percentage_points" if etype == "state" else "program_score",
                       effect=np.nan, CI_low=np.nan, CI_high=np.nan, statistic=np.nan, P=np.nan,
                       low_mean=np.nan, middle_mean=np.nan, high_mean=np.nan, status="descriptive_n_lt_8")
            if test == "high_minus_low" and n:
                delta = (pair.high - pair.low).to_numpy() * scale
                row.update(effect=float(delta.mean()), low_mean=float(pair.low.mean()) * scale,
                           high_mean=float(pair.high.mean()) * scale)
                if n >= 8:
                    boot = delta[random("|".join(map(str, keys)) + test).integers(n, size=(BOOT, n))].mean(axis=1)
                    lo, hi = np.quantile(boot, [.025, .975])
                    if np.allclose(delta, 0, rtol=0, atol=1e-12):
                        stat, p = 0., 1.
                    else:
                        with warnings.catch_warnings():
                            warnings.simplefilter("ignore", UserWarning)
                            result = wilcoxon(np.round(delta, 12), zero_method="wilcox", correction=False, alternative="two-sided", method="approx")
                        stat, p = float(result.statistic), float(result.pvalue)
                    row.update(CI_low=float(lo), CI_high=float(hi), statistic=stat, P=p, status="inferential")
            elif test == "friedman_three_bins" and n:
                x = triple.to_numpy() * scale
                row.update(low_mean=float(x[:, 0].mean()), middle_mean=float(x[:, 1].mean()), high_mean=float(x[:, 2].mean()))
                if n >= 8:
                    if np.allclose(x, x[:, [0]], rtol=0, atol=1e-12):
                        stat, p = 0., 1.
                    else:
                        result = friedmanchisquare(*[x[:, i] for i in range(3)])
                        stat, p = float(result.statistic), float(result.pvalue)
                    row.update(statistic=stat, P=p, status="inferential")
            rows.append(row)
    tests = pd.DataFrame(rows)
    tests["q"] = np.nan
    for family, group in tests.groupby("family"):
        assert len(group) == int(group.family_size.iloc[0]), (family, len(group))
        q = multipletests(group.P.fillna(1), method="fdr_bh")[1]
        tests.loc[group.index, "q"] = np.where(group.P.notna(), q, np.nan)
    return tests

def analysis():
    frozen = validate_freeze()
    cells, support, den = inputs()
    scores = pd.read_csv(OUT / "identity_abundance_scores.tsv", sep="\t")
    elig = pd.read_csv(OUT / "patient_bin_eligibility.tsv", sep="\t")
    data = endpoint_summaries(membership(cells, scores), elig)
    tests = run_tests(data)
    write("patient_state_program_bin_summaries.tsv", data)
    write("complete_patient_tests.tsv", tests)
    summary = data[data.eligible].groupby(["exposure", "weighting", "endpoint_type", "Tissue", "abundance_bin", "endpoint"]).agg(
        n_patients=("patient_id", "nunique"), mean=("value", "mean"), median=("value", "median")).reset_index()
    write("group_endpoint_plot_source.tsv", summary)
    coverage = elig.copy()
    totals = coverage.groupby(["exposure", "patient_id", "Tissue"])["n_T_cells"].transform("sum")
    coverage["cell_proportion"] = coverage.n_T_cells / totals
    write("coverage_plot_source.tsv", coverage)
    json_write("analysis_receipt.json", dict(status="PASS", input_freeze_sha256=sha(OUT / "pretest_freeze.json"),
               seed=SEED, patient_units=19, bulk_reference_patients=339, tests=len(tests), families=tests.family.nunique(),
               nonestimable=int(tests.P.isna().sum()), nominal=int(tests.P.lt(.05).sum()), corrected=int(tests.q.lt(.05).sum()),
               python=platform.python_version(), script_sha256=sha(__file__)))
    print(tests.groupby(["exposure", "weighting", "endpoint_type", "test"]).agg(
        n_min=("n", "min"), n_max=("n", "max"), nominal=("P", lambda x: int(x.lt(.05).sum())),
        corrected=("q", lambda x: int(x.lt(.05).sum()))).to_string(), flush=True)

def figure():
    validate_freeze()
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize
    plt.rcParams.update({"font.family": "Arial", "font.size": 8, "axes.spines.top": False, "axes.spines.right": False,
                         "svg.fonttype": "none", "pdf.fonttype": 42, "axes.linewidth": .6})
    summary = pd.read_csv(OUT / "group_endpoint_plot_source.tsv", sep="\t")
    summary = summary[summary.exposure.eq("combined_mean") & summary.weighting.eq("cell_equal")]
    tests = pd.read_csv(OUT / "complete_patient_tests.tsv", sep="\t")
    tests = tests[tests.exposure.eq("combined_mean") & tests.weighting.eq("cell_equal")]
    coverage = pd.read_csv(OUT / "coverage_plot_source.tsv", sep="\t")
    coverage = coverage[coverage.exposure.eq("combined_mean")]
    cuts = pd.read_csv(OUT / "abundance_cutpoints.tsv", sep="\t").set_index("exposure").loc["combined_mean"]
    fig = plt.figure(figsize=(8.1, 10.4))
    gs = fig.add_gridspec(3, 2, height_ratios=[1.05, 3.5, 1.8], hspace=.36, wspace=.62)
    ax = fig.add_subplot(gs[0, :])
    colors = {"unmatched": "#CECECE", "low": "#B6CDE0", "middle": "#719BB8", "high": "#224D6A"}
    means = coverage.groupby(["Tissue", "abundance_bin"]).cell_proportion.mean().unstack().reindex(index=["Tumor", "Normal"], columns=BINS)
    left = np.zeros(2)
    for bin_name in BINS:
        values = means[bin_name].to_numpy() * 100
        ax.barh([1, 0], values, left=left, height=.45, color=colors[bin_name], label=bin_name.title(), edgecolor="white", linewidth=.4)
        for y, value, offset in zip([1, 0], values, left):
            if value > 6:
                ax.text(offset + value / 2, y, f"{value:.1f}%", ha="center", va="center", fontsize=8,
                        color="white" if bin_name == "high" else "#222222")
        left += values
    ax.set_yticks([1, 0], ["Tumor", "Adjacent normal"])
    ax.set_xlim(0, 100)
    ax.set_xlabel("Mean percentage of qualifying tissue T cells (19 patients per tissue)")
    ax.set_title("A  Independent bulk-reference abundance groups", loc="left", fontweight="bold")
    ax.legend(ncol=4, loc="upper center", bbox_to_anchor=(.55, 1.03), frameon=False, fontsize=7)
    ax.text(.0, 1.45, "339 unrelated bulk-blood patients; AA/V/J identity; equal-patient mean reads per million QC reads",
            transform=ax.transAxes, fontsize=8)
    ax.text(.0, 1.24, f"Positive-key tertiles: low ≤ {cuts.lower_ppm:.3g}; middle ≤ {cuts.upper_ppm:.3g}; high > {cuts.upper_ppm:.3g} ppm",
            transform=ax.transAxes, fontsize=7.5)
    for col, tissue in enumerate(["Tumor", "Normal"]):
        ax = fig.add_subplot(gs[1, col])
        z = summary[summary.endpoint_type.eq("state") & summary.Tissue.eq(tissue)]
        states = sorted(z.endpoint.unique())
        matrix = z.pivot(index="endpoint", columns="abundance_bin", values="mean").reindex(index=states, columns=BINS) * 100
        labels = [x.replace("_", " ") for x in states]
        # Cell colors are absolute percentages, shared across both tissues.
        im = ax.imshow(matrix.to_numpy(), cmap="YlGnBu", norm=Normalize(0, 40), aspect="auto", interpolation="nearest")
        ax.set_xticks(range(4), ["Unmatched", "Low", "Middle", "High"], fontsize=7)
        ax.set_yticks(range(32), labels, fontsize=6.3)
        ax.set_title(("B  Tumor" if tissue == "Tumor" else "C  Adjacent normal") + " cell-state proportions", loc="left", fontweight="bold", fontsize=8)
        ax.tick_params(length=0)
        for spine in ax.spines.values():
            spine.set_visible(False)
        st = tests[tests.endpoint_type.eq("state") & tests.test.eq("high_minus_low") & tests.Tissue.eq(tissue)].set_index("endpoint")
        for i, state in enumerate(states):
            if state in st.index and st.loc[state, "q"] < .05:
                ax.text(3, i, "*", ha="center", va="center", fontsize=10, color="#B73442")
        ax.text(0, -.1, "* high vs low, q < 0.05 across 64 state tests", transform=ax.transAxes, fontsize=6.4)
    color_ax = fig.add_axes([.89, .435, .018, .15])
    cb = fig.colorbar(im, cax=color_ax)
    cb.set_label("Mean state proportion (%)", fontsize=7)
    cb.ax.tick_params(labelsize=7)
    ax = fig.add_subplot(gs[2, :])
    labels = {"cytotoxic": "Cytotoxicity", "exhaustion": "Exhaustion-associated", "naive_memory": "Naive/memory",
              "interferon": "Interferon response", "residency": "Tissue residency", "proliferation": "Proliferation"}
    tumor_color, normal_color = "#B14F5B", "#317D88"
    for tissue, color, offset in [("Tumor", tumor_color, -.13), ("Normal", normal_color, .13)]:
        z = tests[tests.endpoint_type.eq("program") & tests.test.eq("high_minus_low") & tests.Tissue.eq(tissue)].set_index("endpoint")
        for i, program in enumerate(PROGRAMS):
            r = z.loc[program]
            y = 5 - i + offset
            if pd.notna(r.CI_low):
                ax.plot([r.CI_low, r.CI_high], [y, y], color=color, linewidth=1.2)
            ax.scatter([r.effect], [y], marker="o" if tissue == "Tumor" else "s", s=28,
                       facecolor=color if pd.notna(r.q) and r.q < .05 else "white", edgecolor=color, linewidth=1, zorder=3)
            ax.text(1.005, y, f"n={int(r.n)}; q={r.q:.3g}" if pd.notna(r.q) else f"n={int(r.n)}; descriptive",
                    transform=ax.get_yaxis_transform(), fontsize=6.8, color=color, va="center")
    ax.axvline(0, color="#777777", linewidth=.6)
    ax.set_yticks(range(6), [labels[p] for p in PROGRAMS[::-1]])
    ax.set_xlabel("High-minus-low CD8 program score (mean paired difference, 95% patient-bootstrap CI)", fontsize=7.5)
    ax.set_title("D  Patient-level program contrasts", loc="left", fontweight="bold")
    ax.scatter([], [], color=tumor_color, marker="o", label="Tumor")
    ax.scatter([], [], color=normal_color, marker="s", label="Adjacent normal")
    ax.legend(loc="lower left", bbox_to_anchor=(0, -0.55), ncol=2, frameon=False, fontsize=7)
    ax.text(.38, -.37, "Filled: q < 0.05 across 12 program tests; open: q ≥ 0.05", transform=ax.transAxes, fontsize=6.7)
    fig.subplots_adjust(left=.26, right=.84, top=.9, bottom=.1)
    target = OUT / "figures"
    target.mkdir(exist_ok=True)
    for suffix in ["png", "pdf", "svg"]:
        fig.savefig(target / ("Supplementary_Figure15_bulk_abundance_states." + suffix), dpi=300, facecolor="white")
    plt.close(fig)
    write("Figure15_state_plot_source.tsv", summary[summary.endpoint_type.eq("state")])
    write("Figure15_program_plot_source.tsv", tests[tests.endpoint_type.eq("program") & tests.test.eq("high_minus_low")])
    write("Figure15_coverage_plot_source.tsv", coverage)
    print("Figure15 PNG/PDF/SVG and plot-source tables saved.", flush=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["preflight", "analysis", "figure"])
    args = parser.parse_args()
    {"preflight": preflight, "analysis": analysis, "figure": figure}[args.stage]()
