#!/usr/bin/env python3
"""Render the complete post hoc MIL bag-size sensitivity as a patient-level figure."""
from __future__ import annotations
from release_runtime import private_literal as _private_literal, release_path as _release_path

import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(_release_path('path_1304218e86d2e596'))
OUT = ROOT / "mil/bag_curve_v2/evaluation"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    receipt_path = OUT / "receipt.json"
    if not receipt_path.exists() or json.loads(receipt_path.read_text()).get("status") != "complete":
        raise RuntimeError("A complete five-size evaluation is required")
    table_path = OUT / "curve_metrics.tsv"
    gate_path = OUT / "gate.json"
    d = pd.read_csv(table_path, sep="\t")
    gate = json.loads(gate_path.read_text())
    if len(d) != 10 or d.groupby("model").bag_size.nunique().ne(5).any():
        raise ValueError("Expected both models at all five bag sizes")
    baseline = gate["comparisons"]["clinical_technical_batch"]["baseline_metrics"]
    models = ["tcrbert_dualtask_aa_mean", "sceptr_cdr3_only"]
    if set(d.model) != set(models):
        raise ValueError("Unexpected encoder names")
    colors = {models[0]: "#2166ac", models[1]: "#b2182b"}
    labels = {models[0]: "TCR-BERT", models[1]: "SCEPTR"}
    fig, axes = plt.subplots(2, 3, figsize=(14.8, 8.6), layout="constrained")
    sizes = [1024, 2048, 4096, 8192, 16384]
    for model in models:
        block = d.loc[d.model.eq(model)].sort_values("bag_size")
        x = block.bag_size.to_numpy()
        c = colors[model]
        lab = labels[model]
        axes[0, 0].plot(x, block.paired_gain_vs_clinical_technical_batch, "o-", color=c, label=lab)
        axes[0, 0].fill_between(x, block.paired_gain_vs_clinical_technical_batch_ci_lower,
                                block.paired_gain_vs_clinical_technical_batch_ci_upper,
                                color=c, alpha=.14)
        axes[0, 1].plot(x, block.brier, "o-", color=c, label=lab)
        axes[0, 2].plot(x, block.auroc, "o-", color=c, label=f"{lab} AUROC")
        axes[0, 2].plot(x, block.ap, "s--", color=c, label=f"{lab} AP")
        axes[1, 0].plot(x, block.mean_coverage_fraction, "o-", color=c, label=f"{lab} unique AA")
        axes[1, 0].plot(x, block.candidate_capture_single_bag_probability_carriers,
                        "s--", color=c, label=f"{lab} group capture estimate")
        axes[1, 1].plot(x, block.mean_mc_sd, "o-", color=c, label=lab)
        axes[1, 2].plot(x, block.gpu_total_seconds / 3600, "o-", color=c, label=lab)
    axes[0, 0].axhline(0, color="#555555", ls=":", lw=1)
    axes[0, 1].axhline(baseline["brier"], color="#555555", ls=":", lw=1, label="Clinical/technical baseline")
    axes[0, 2].axhline(baseline["auroc"], color="#555555", ls=":", lw=1, label="Baseline AUROC")
    axes[0, 2].axhline(baseline["ap"], color="#888888", ls="--", lw=1, label="Baseline AP")
    titles = ["Paired log-loss gain vs baseline (95% CI)", "Brier score (lower is better)",
              "Discrimination", "Single-bag input coverage", "Repeat-inference prediction SD",
              "GPU process hours, 60 fits per point"]
    ylabels = ["Baseline loss − MIL loss", "Brier", "AUROC / AP", "Fraction", "Mean MC SD", "Hours"]
    for ax, title, ylabel in zip(axes.flat, titles, ylabels):
        ax.set_xscale("log", base=2)
        ax.set_xticks(sizes, [f"{v:,}" for v in sizes], rotation=30)
        ax.set_xlabel("Train and inference bag size")
        ax.set_ylabel(ylabel)
        ax.set_title(title, fontsize=11)
        ax.grid(alpha=.2)
    axes[0, 0].legend(fontsize=8)
    axes[0, 1].legend(fontsize=8)
    axes[0, 2].legend(fontsize=7, ncol=2)
    axes[1, 0].legend(fontsize=7)
    axes[1, 1].legend(fontsize=8)
    axes[1, 2].legend(fontsize=8)
    fig.suptitle(f"{_private_literal('private_f38a7a588fc0c10d')}{gate['status']}",
                 fontsize=14)
    png = OUT / "bag_curve_summary.png"
    svg = OUT / "bag_curve_summary.svg"
    fig.savefig(png, dpi=180)
    fig.savefig(svg)
    plt.close(fig)
    report = {"status": "complete", "table_sha256": sha256(table_path),
              "gate_sha256": sha256(gate_path), "png_sha256": sha256(png),
              "svg_sha256": sha256(svg),
              "scope": "descriptive held-out curves; selection and gate use inner-selected OOF"}
    (OUT / "bag_curve_plot_receipt.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": "complete", "png": str(png), "svg": str(svg)}))


if __name__ == "__main__":
    main()
