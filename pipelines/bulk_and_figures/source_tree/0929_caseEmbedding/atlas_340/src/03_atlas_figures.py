#!/usr/bin/env python3
"""Descriptive atlas figures; UMAP coordinates never define groups."""
from __future__ import annotations
from release_runtime import private_literal as _private_literal, release_path as _release_path

import hashlib
import json
import os
from pathlib import Path

os.environ["MPLBACKEND"] = "Agg"
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(_release_path('path_99d8c1c6fde2a6b9'))
MODELS = ("tcrbert_dualtask_aa_mean", "sceptr_cdr3_only")
COLORS = {"home": "#247ba0", "external": "#d95f02",
          "H2498B": "#247ba0", "H2498C": "#35a37d", "H2498D": "#9467bd"}


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main() -> None:
    out = ROOT / "figures"
    out.mkdir(parents=True, exist_ok=True)
    manifest = pd.read_csv(ROOT / "config/cohort_samples.tsv", sep="\t",
                           dtype=str, keep_default_na=False,
                           usecols=["cohort", "patient_key", "home_batch"])
    info = manifest.drop_duplicates("patient_key")
    fig, axes = plt.subplots(2, 2, figsize=(12, 9), constrained_layout=True)
    input_hashes = {}
    for row, model in enumerate(MODELS):
        for col, scope in enumerate(("joint", "home")):
            path = ROOT / "atlas" / scope / model / "umap_balanced_display.parquet"
            receipt = json.loads((path.parent / "umap_receipt.json").read_text())
            if receipt.get("status") != "descriptive_only" or (
                receipt.get("data_sha256") != sha(path)
            ):
                raise ValueError("UMAP display input receipt invalid: " + str(path))
            d = pd.read_parquet(path).merge(
                info, on="patient_key", how="left", validate="many_to_one",
                suffixes=("", "_manifest"))
            ax = axes[row, col]
            field = "cohort" if scope == "joint" else "home_batch"
            labels = ("home", "external") if scope == "joint" else (
                "H2498B", "H2498C", "H2498D")
            for label in labels:
                part = d.loc[d[field].eq(label)]
                ax.scatter(part.umap_1, part.umap_2, s=3, alpha=.17,
                           color=COLORS[label], rasterized=True,
                           linewidths=0, label=label)
            ax.set_title(("Joint descriptive" if scope == "joint" else "Home discovery") +
                         " • " + ("TCR-BERT" if row == 0 else "SCEPTR"))
            ax.set_xlabel("UMAP 1")
            ax.set_ylabel("UMAP 2")
            ax.legend(markerscale=4, fontsize=8, frameon=False, loc="best")
            ax.text(.01, .01, "Patient-balanced display; no cluster boundary",
                    fontsize=7, transform=ax.transAxes, ha="left", va="bottom")
            input_hashes[scope + "/" + model] = sha(path)
    png = out / "balanced_embedding_atlas.png"
    pdf = out / "balanced_embedding_atlas.pdf"
    fig.savefig(png, dpi=180, facecolor="white")
    fig.savefig(pdf, facecolor="white")
    plt.close(fig)

    summary_path = ROOT / "atlas" / "patient_technical_summary.parquet"
    rare_path = ROOT / "atlas" / "equal_depth_repeats.parquet"
    technical = pd.read_parquet(summary_path)
    rare = pd.read_parquet(rare_path)
    rare_mean = rare.groupby("patient_key", as_index=False).unique_aa_after.mean()
    technical = technical.merge(rare_mean, on="patient_key", validate="one_to_one")
    fig, ax = plt.subplots(1, 3, figsize=(12, 3.8), constrained_layout=True)
    features = [
        ("total_reads", "QC reads", True),
        ("unique_aa", "Unique CDR3β AA", True),
        ("unique_aa_after", "Unique AA at common depth", False),
    ]
    rng = np.random.default_rng(20260930)
    for a, (key, name, log) in zip(ax, features):
        data = [technical.loc[technical.cohort.eq(cohort), key].to_numpy()
                for cohort in ("home", "external")]
        box = a.boxplot(data, positions=(1, 2), widths=.5, patch_artist=True,
                        showfliers=False)
        for patch, color in zip(box["boxes"], (COLORS["home"], COLORS["external"])):
            patch.set(facecolor=color, alpha=.4, edgecolor=color)
        for i, arr in enumerate(data, 1):
            a.scatter(i + rng.uniform(-.16, .16, len(arr)), arr, s=7, alpha=.25,
                      color=COLORS["home" if i == 1 else "external"])
        a.set_xticks([1, 2], ["Home", "External"])
        a.set_ylabel(name)
        if log:
            a.set_yscale("log")
        a.grid(axis="y", alpha=.16)
    fig.suptitle("Patient-level technical and equal-depth distributions", fontsize=11)
    depth_png = out / "technical_depth_diagnostics.png"
    depth_pdf = out / "technical_depth_diagnostics.pdf"
    fig.savefig(depth_png, dpi=180, facecolor="white")
    fig.savefig(depth_pdf, facecolor="white")
    plt.close(fig)
    report = {
        "status": "complete", "label_blind": True,
        "interpretation": "descriptive; neither UMAP nor cohort geometry defines TCR groups",
        "balanced_umap_input_sha256": input_hashes,
        "technical_summary_sha256": sha(summary_path),
        "equal_depth_sha256": sha(rare_path),
        "outputs_sha256": {
            png.name: sha(png), pdf.name: sha(pdf),
            depth_png.name: sha(depth_png), depth_pdf.name: sha(depth_pdf),
        },
        "script_sha256": sha(Path(__file__)),
    }
    path = out / "figures_receipt.json"
    path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
