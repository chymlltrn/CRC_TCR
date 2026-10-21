#!/usr/bin/env python3
"""Freeze patient-level Home outer and inner splits after 340-library QC."""
from __future__ import annotations
from release_runtime import private_literal as _private_literal, release_path as _release_path
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

ROOT = Path(_release_path('path_99d8c1c6fde2a6b9'))
SEED = 20260930

def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()

def main() -> None:
    qc_path = ROOT / "audit/01_ingest_summary.json"
    qc = json.loads(qc_path.read_text())
    if qc.get("status") != "complete":
        raise RuntimeError("Stage1 is not complete")
    manifest_path = ROOT / "config/cohort_samples.tsv"
    m = pd.read_csv(manifest_path, sep="\t", dtype=str, keep_default_na=False,
                    usecols=["cohort", "sample_id", "patient_key", "recorded_n_binary"])
    if len(m) != 340 or m.sample_id.nunique() != 340:
        raise RuntimeError("manifest identity mismatch")
    home = m.loc[m.cohort.eq("home"), ["patient_key", "recorded_n_binary"]]
    if len(home) != 138:
        raise RuntimeError("Home library count mismatch")
    conflicts = home.groupby("patient_key").recorded_n_binary.nunique()
    if not conflicts.eq(1).all():
        raise RuntimeError("same patient has conflicting N labels")
    home = home.loc[home.recorded_n_binary.isin(["0", "1"])].drop_duplicates("patient_key")
    home = home.sort_values("patient_key").reset_index(drop=True)
    y = home.recorded_n_binary.astype(int).to_numpy()
    counts = np.bincount(y, minlength=2)
    if len(home) != 126 or (counts < 5).any():
        raise RuntimeError(f"unexpected Home N-known patients: {len(home)}, classes {counts.tolist()}")
    rows = []
    outer = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    for fold, (tr, te) in enumerate(outer.split(home, y)):
        train_y = y[tr]
        if (np.bincount(train_y, minlength=2) < 3).any():
            raise RuntimeError("inner class count insufficient")
        inner_assign = {}
        seed = int.from_bytes(hashlib.sha256(f"{SEED}:inner:{fold}".encode()).digest()[:4], "little")
        inner = StratifiedKFold(n_splits=3, shuffle=True, random_state=seed)
        for inner_fold, (_, val_pos) in enumerate(inner.split(tr, train_y)):
            for pos in val_pos:
                inner_assign[int(tr[pos])] = inner_fold
        for i in tr:
            rows.append((home.at[i, "patient_key"], int(y[i]), fold, "train", inner_assign[int(i)]))
        for i in te:
            rows.append((home.at[i, "patient_key"], int(y[i]), fold, "test", pd.NA))
    out = ROOT / "folds/folds.tsv"
    out.parent.mkdir(parents=True, exist_ok=True)
    f = pd.DataFrame(rows, columns=["patient_key", "recorded_n_binary", "outer_fold", "partition", "inner_fold"])
    f = f.sort_values(["outer_fold", "partition", "patient_key"]).reset_index(drop=True)
    if f.groupby(["outer_fold", "patient_key"]).size().ne(1).any():
        raise RuntimeError("duplicate patient/fold")
    if not f.groupby("patient_key").outer_fold.nunique().eq(5).all():
        raise RuntimeError("patient missing from an outer fold")
    f.to_csv(out, sep="\t", index=False)
    receipt = dict(status="complete", patient_level=True, external_in_folds=False,
                   n_home_patients=len(home), class_counts={"0": int(counts[0]), "1": int(counts[1])},
                   outer_folds=5, inner_folds=3, rows=len(f), seed=SEED,
                   manifest_sha256=sha(manifest_path), stage1_sha256=sha(qc_path),
                   folds_sha256=sha(out))
    (ROOT / "folds/folds.json").write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    print(json.dumps(receipt, sort_keys=True))

if __name__ == "__main__":
    main()
