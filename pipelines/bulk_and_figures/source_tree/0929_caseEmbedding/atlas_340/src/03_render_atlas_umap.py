#!/usr/bin/env python3
"""Render label-blind, patient-balanced UMAP displays; never use geometry to define groups."""
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

ROOT = Path(_release_path('path_99d8c1c6fde2a6b9'))
MODELS = (
    ("tcrbert_dualtask_aa_mean", "TCR-BERT"),
    ("sceptr_cdr3_only", "SCEPTR"),
)
COLORS = {"home": "#147D92", "external": "#CE7045"}


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(4 << 20), b""):
            h.update(block)
    return h.hexdigest()


def main() -> None:
    out = ROOT / "atlas" / "figures"
    out.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.65), constrained_layout=True)
    inputs = {}
    counts = {}
    for ax, (model, title) in zip(axes, MODELS):
        path = ROOT / "atlas" / "joint" / model / "umap_balanced_display.parquet"
        df = pd.read_parquet(path, columns=["patient_key", "cohort", "seq_idx", "umap_1", "umap_2"])
        assert set(df.cohort.unique()) == set(COLORS)
        assert df[["umap_1", "umap_2"]].notna().all().all()
        per_patient = df.groupby("patient_key", observed=True).size()
        assert per_patient.nunique() == 1
        assert not df.duplicated(["patient_key", "seq_idx"]).any()
        inputs[model] = digest(path)
        counts[model] = {
            "points": int(len(df)),
            "patients": int(per_patient.size),
            "points_per_patient": int(per_patient.iloc[0]),
            "home_points": int((df.cohort == "home").sum()),
            "external_points": int((df.cohort == "external").sum()),
        }
        rng = np.random.default_rng(20260930)
        perm = rng.permutation(len(df))
        df = df.iloc[perm]
        for cohort, label in (("external", "External"), ("home", "Home")):
            sub = df.loc[df.cohort == cohort]
            ax.scatter(
                sub.umap_1, sub.umap_2, s=2.2, alpha=0.23, c=COLORS[cohort],
                linewidths=0, rasterized=True, label=label,
            )
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.set_xlabel("UMAP 1")
        ax.set_ylabel("UMAP 2")
        ax.spines[["top", "right"]].set_visible(False)
        ax.legend(frameon=False, markerscale=3.5, loc="best", fontsize=9)
    fig.suptitle("Patient-balanced, label-blind CDR3β representation", fontsize=13)
    fig.text(
        0.5, -0.015,
        "100 unique AA per patient; abundance excluded from coordinates. "
        "UMAP is descriptive and does not define TCR groups.",
        ha="center", va="top", fontsize=8.3,
    )
    png = out / "joint_patient_balanced_umap_cohort.png"
    pdf = out / "joint_patient_balanced_umap_cohort.pdf"
    fig.savefig(png, dpi=240, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)
    receipt = {
        "status": "complete",
        "purpose": "descriptive label-blind display; not a grouping or association input",
        "input_sha256": inputs,
        "counts": counts,
        "png_sha256": digest(png),
        "pdf_sha256": digest(pdf),
    }
    (out / "render_receipt.json").write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    print(json.dumps(receipt, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
