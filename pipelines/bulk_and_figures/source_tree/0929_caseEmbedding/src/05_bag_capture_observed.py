#!/usr/bin/env python3
"""Audit observed held-out bag capture of training-nominated, V/J-safe TCR groups."""
from __future__ import annotations
from release_runtime import private_literal as _private_literal, release_path as _release_path

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(_release_path('path_1304218e86d2e596'))
RUN = ROOT / "mil/bag_curve_v2"
OUT = RUN / "evaluation"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def stable_seed(*parts: object) -> int:
    raw = "|".join(map(str, parts)).encode()
    return int.from_bytes(hashlib.sha256(raw).digest()[:8], "little") % (2**32 - 1)


def draw_order(n: int, patient: str, draw: int, scope: str, seed: int) -> np.ndarray:
    rng = np.random.default_rng(stable_seed("draw-v2-fixed-prefix", patient, -1, draw, scope, seed))
    return rng.choice(n, size=min(32768, n), replace=False)


def main() -> None:
    receipt_path = OUT / "receipt.json"
    if not receipt_path.exists() or json.loads(receipt_path.read_text()).get("status") != "complete":
        raise RuntimeError("Complete, validated five-size curve evaluation required")
    cfg_path = ROOT / "config/bag_curve_v2.json"
    cfg = json.loads(cfg_path.read_text())
    if cfg["draw_policy"]["name"] != "draw-v2-fixed-prefix":
        raise ValueError("Unexpected draw policy")
    preds_path = OUT / "curve_patient_seed.tsv"
    preds = pd.read_csv(preds_path, sep="\t", dtype={"patient_key": str})
    folds_path = ROOT / "folds/folds.tsv"
    folds = pd.read_csv(folds_path, sep="\t", dtype={"patient_key": str})
    index_path = ROOT / "mil/patient_index/index.tsv"
    index = pd.read_csv(index_path, sep="\t", dtype={"patient_key": str, "file": str}).set_index("patient_key")
    noms_path = ROOT / "groups/fold_nominations.tsv"
    noms = pd.read_csv(noms_path, sep="\t", dtype={"group_id": str})
    expected = 114 * len(cfg["models"]) * len(cfg["bag_sizes"]) * len(cfg["seeds"])
    if len(preds) != expected or preds[["outer_fold", "patient_key", "model", "bag_size", "seed"]].duplicated().any():
        raise ValueError("Incomplete or duplicated outer prediction curve")
    if set(preds.bag_size) != set(cfg["bag_sizes"]) or set(preds.model) != set(cfg["models"]) or set(preds.seed) != set(cfg["seeds"]):
        raise ValueError("Model, bag, or seed axes changed")
    if not pd.to_numeric(preds.mc_draws, errors="coerce").isin([8, 32]).all():
        raise ValueError("Unknown inference draw count")
    output = []
    source_hashes = {"config": sha256(cfg_path), "folds": sha256(folds_path),
                     "index": sha256(index_path), "nominations": sha256(noms_path),
                     "predictions": sha256(preds_path),
                     "runner": sha256(ROOT / "src/05_bag_curve_v2.py")}
    for outer in cfg["outer_folds"]:
        heldout = set(folds.loc[(folds.outer_fold == outer) & (folds.partition == "test"), "patient_key"])
        nominated = sorted(set(noms.loc[noms.outer_fold == outer, "group_id"]))
        if not heldout or len(nominated) != 5:
            raise ValueError(f"Unexpected held-out patients or nominations at outer {outer}")
        safe_path = ROOT / f"groups/outer_{outer}/patient_group_aa_safe.parquet"
        source_hashes[f"safe_outer_{outer}"] = sha256(safe_path)
        safe = pd.read_parquet(safe_path, columns=["patient_key", "group_id", "seq_idx", "projection_exact"])
        safe = safe.loc[safe.patient_key.isin(heldout) & safe.group_id.isin(nominated)]
        if not safe.projection_exact.eq(True).all():
            raise ValueError(f"Ambiguous V/J projection in outer {outer}")
        safe_groups = {(str(p), str(g)): np.unique(block.seq_idx.to_numpy(dtype=np.int64))
                       for (p, g), block in safe.groupby(["patient_key", "group_id"])}
        scoped = preds.loc[preds.outer_fold == outer]
        for patient in sorted(heldout):
            patient_rows = scoped.loc[scoped.patient_key == patient]
            if len(patient_rows) != len(cfg["models"]) * len(cfg["bag_sizes"]) * len(cfg["seeds"]):
                raise ValueError(f"Missing curve rows for {patient} in outer {outer}")
            with np.load(ROOT / "mil/patient_index" / str(index.loc[patient, "file"])) as bank:
                seq_idx = bank["seq_idx"].astype(np.int64)
            n = len(seq_idx)
            if n != int(index.loc[patient, "unique_aa"]) or np.any(seq_idx[1:] <= seq_idx[:-1]):
                raise ValueError(f"Patient index mismatch: {patient}")
            positions = {}
            for group in nominated:
                gids = safe_groups.get((patient, group), np.empty(0, dtype=np.int64))
                pos = np.searchsorted(seq_idx, gids)
                if len(gids) and (np.any(pos >= n) or not np.array_equal(seq_idx[pos], gids)):
                    raise ValueError(f"Safe group AA not found in patient index: {patient}, {group}")
                positions[group] = pos
            for seed in cfg["seeds"]:
                rows = patient_rows.loc[patient_rows.seed == seed]
                max_draws = int(rows.mc_draws.max())
                first_hit = {group: np.full(max_draws, 32768, dtype=np.int32) for group in nominated}
                scope = f"curve-outer{outer}-test"
                for draw in range(max_draws):
                    order = draw_order(n, patient, draw, scope, int(seed))
                    if len(np.unique(order)) != len(order):
                        raise ValueError("Repeated AA in inferred bag")
                    reverse = np.full(n, 32768, dtype=np.int32)
                    reverse[order] = np.arange(len(order), dtype=np.int32)
                    for group in nominated:
                        if len(positions[group]):
                            first_hit[group][draw] = int(reverse[positions[group]].min())
                for pred in rows.itertuples():
                    bag = int(pred.bag_size)
                    k = int(pred.mc_draws)
                    for group in nominated:
                        hits = first_hit[group][:k] < min(bag, n)
                        output.append({
                            "outer_fold": outer, "patient_key": patient, "group_id": group,
                            "model": str(pred.model), "seed": int(seed), "bag_size": bag,
                            "patient_unique_aa": n, "safe_group_aa": len(positions[group]),
                            "mc_draws": k, "observed_single_bag_hit_fraction": float(hits.mean()),
                            "observed_any_mc_bag_hit": bool(hits.any()),
                            "observed_hit_bags": int(hits.sum()),
                            "method": "reconstructed_actual_fixed_seed_inference_draws"
                        })
    frame = pd.DataFrame(output)
    expected_rows = 114 * 5 * len(cfg["models"]) * len(cfg["bag_sizes"]) * len(cfg["seeds"])
    if len(frame) != expected_rows:
        raise ValueError(f"Unexpected candidate capture rows: {len(frame)}")
    frame = frame.sort_values(["outer_fold", "patient_key", "group_id", "model", "bag_size", "seed"])
    output_path = OUT / "candidate_capture_observed.tsv"
    frame.to_csv(output_path, sep="\t", index=False)
    carrier = frame.loc[frame.safe_group_aa > 0]
    summary = (carrier.groupby(["model", "bag_size"], as_index=False)
               .agg(carrier_group_seed_pairs=("patient_key", "size"),
                    observed_single_bag_hit_fraction=("observed_single_bag_hit_fraction", "mean"),
                    observed_any_mc_bag_hit_fraction=("observed_any_mc_bag_hit", "mean"),
                    mean_mc_draws=("mc_draws", "mean")))
    summary_path = OUT / "candidate_capture_observed_summary.tsv"
    summary.to_csv(summary_path, sep="\t", index=False)
    receipt = {"status": "complete", "scope": "outer held-out patients and outer-training nominations",
               "rows": len(frame), "carrier_rows": len(carrier),
               "source_hashes": source_hashes,
               "output_sha256": sha256(output_path), "summary_sha256": sha256(summary_path),
               "interpretation": "Observed capture by the exact seeded inference bags; capture is input coverage, not model contribution"}
    receipt_out = OUT / "candidate_capture_observed_receipt.json"
    receipt_out.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": "complete", "rows": len(frame), "carrier_rows": len(carrier)}))


if __name__ == "__main__":
    main()
