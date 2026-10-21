#!/usr/bin/env python3
"""Attach independently validated SCEPTR input applicability to final metadata."""
from release_runtime import private_literal as _private_literal, release_path as _release_path
import json
import os
from pathlib import Path
import tempfile

import numpy as np

ROOT = Path(_release_path('path_1304218e86d2e596'))
OUT = ROOT / "embeddings" / "sceptr_cdr3_only"


def require(ok, message):
    if not ok:
        raise RuntimeError(message)


def main():
    meta_path = OUT / "meta.json"
    meta = json.loads(meta_path.read_text())
    audit = json.loads((OUT / "input_compatibility.json").read_text())
    verified = json.loads((OUT / "verification.json").read_text())
    require(meta["status"] == "complete" and verified["status"] == "PASS",
            "SCEPTR bank is not independently complete")
    require(audit["status"] == "PASS"
            and audit["input_sha256"] == meta["input_sha256"]
            and verified["input_sha256"] == meta["input_sha256"]
            and audit["rows"] == meta["n_total"], "SCEPTR audit does not match bank")
    require(audit["nonstandard_count"] ==
            meta["nonstandard_junction_warnings_total"], "Warning totals disagree")
    require(audit["unparseable_count"] == 0, "Unparseable sequence present")
    ids = np.load(OUT / "nonstandard_junction_seq_idx.npy",
                  mmap_mode="r", allow_pickle=False)
    require(len(ids) == audit["nonstandard_count"]
            and (len(ids) == 0 or (ids[0] >= 0
            and ids[-1] < meta["n_total"]
            and bool(np.all(ids[1:] > ids[:-1])))),
            "Nonstandard junction index mask is malformed")
    diagnostic = {
        "audit_receipt": "embeddings/sceptr_cdr3_only/input_compatibility.json",
        "nonstandard_seq_idx_mask": "embeddings/sceptr_cdr3_only/nonstandard_junction_seq_idx.npy",
        "nonstandard_count": int(audit["nonstandard_count"]),
        "nonstandard_fraction": audit["nonstandard_count"] / meta["n_total"],
        "unparseable_count": 0,
        "raw_aa_to_model_input": "identity: original CDR3B AA string retained",
        "sequence_modification_count": 0,
        "sequence_modification_collision_count": 0,
        "evidence": "libtcrlm schema validates a tidytcells standardized candidate, emits warning when different, then stores original junction_b_sequence; Cdr3Tokeniser tokenizes that original string one residue at a time",
        "library_versions": {
            "sceptr": audit["sceptr_version"],
            "tidytcells": audit["tidytcells_version"],
        },
    }
    if "input_compatibility" in meta:
        require(meta["input_compatibility"] == diagnostic,
                "Existing applicability annotation differs")
    meta["input_compatibility"] = diagnostic
    temp = meta_path.with_name(meta_path.name + f".partial.{os.getpid()}")
    with temp.open("w") as stream:
        json.dump(meta, stream, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temp, meta_path)
    print(json.dumps(diagnostic, sort_keys=True))


if __name__ == "__main__":
    main()
