#!/usr/bin/env python3
"""Record a provenance-bound tcrdist omission after a failed prespecified gate."""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read(path: Path) -> dict | None:
    return json.loads(path.read_text()) if path.is_file() else None


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--scope", required=True)
    a = p.parse_args()
    root = a.root.resolve()
    if a.scope not in {"full_home", *(f"outer_{i}" for i in range(5))}:
        raise ValueError("Only full Home and five outer scopes are allowed")
    out = root / "groups" / a.scope
    desc_path = out / "description.json"
    desc = read(desc_path)
    if not desc or desc.get("scope") != a.scope or not (out / "patient_nodes.parquet").is_file():
        raise ValueError("Training scope lacks completed node description")
    for name, field in (("patient_nodes.parquet", "patient_nodes_sha256"), ("clone_nodes.parquet", "clone_nodes_sha256")):
        if desc.get(field) != sha(out / name):
            raise ValueError("Scope node hashes differ from description")
    proof_path = root / "groups/full_home/tcrdist_exact_v_lower_bound_diagnostic.json"
    graph_path = root / "groups/full_home/tcrdist_exact_v_graph_preflight.json"
    cost_path = root / "groups/tcrdist_exact_v_project_cost_gate.json"
    proof, graph, cost = (read(x) for x in (proof_path, graph_path, cost_path))
    evidence = []
    if proof and proof.get("status") == "STOP_RESOURCE":
        evidence.append(("STOP_RESOURCE", "Full Home exact-V six-scope projected cost exceeds 96 hours", proof_path))
    if graph and graph.get("status") in {"STOP_GIANT_COMPONENT", "STOP_PROOF"}:
        evidence.append(("STOP_SCIENTIFIC", "Prespecified full Home radius-24 graph or exact-distance proof failed", graph_path))
    if cost and cost.get("status") == "STOP" and cost.get("stop_reason"):
        reason = cost["stop_reason"]
        kind = "STOP_RESOURCE" if "cost" in reason.lower() or "96" in reason else "STOP_SCIENTIFIC"
        evidence.append((kind, reason, cost_path))
    if not evidence:
        raise ValueError("No completed explicit tcrdist gate failure; cannot record an omission")
    status, reason, primary = evidence[0]
    if (out / "tcrdist_home_group_membership.parquet").exists():
        raise ValueError("A tcrdist membership exists; audit its formal projection receipt")
    receipt = {
        "status": status, "scope": a.scope, "method": "tcrdist3", "reason": reason,
        "projection_complete": False, "projection_population": "home", "label_blind": True,
        "dictionary_fit_population": "Home training patients of this scope only",
        "radius_prespecified": 24, "seed_per_patient_prespecified": 512,
        "node_description_sha256": sha(desc_path),
        "train_patient_sha256": desc["train_patient_sha256"],
        "folds_sha256": desc["folds_sha256"],
        "manifest_identity_sha256": desc["manifest_identity_sha256"],
        "home_split_label_sha256": desc["home_split_label_sha256"],
        "primary_gate_receipt_sha256": sha(primary),
        "proof_sha256": sha(proof_path) if proof else None,
        "graph_preflight_sha256": sha(graph_path) if graph else None,
        "project_cost_gate_sha256": sha(cost_path) if cost else None,
        "interpretation": "tcrdist3 unavailable under the frozen gate; no tcrdist3 group tested",
        "script_sha256": sha(Path(__file__)),
    }
    dest = out / "tcrdist_home_projection_receipt.json"
    if dest.exists():
        if read(dest) != receipt:
            raise ValueError("Existing tcrdist projection receipt differs; do not overwrite it")
    else:
        tmp = dest.with_suffix(dest.suffix + ".tmp")
        tmp.write_text(json.dumps(receipt, indent=2, sort_keys=True, allow_nan=False) + "\n")
        os.replace(tmp, dest)
    print(json.dumps({"scope": a.scope, "status": status, "reason": reason, "receipt": str(dest)}))

if __name__ == "__main__":
    main()
