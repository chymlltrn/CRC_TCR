#!/usr/bin/env python3
"""Resume the frozen atlas analysis after both technical pilots.

Nothing runs without --execute. One child process runs at a time with a 128 GiB
address-space limit and at most 16 configured CPU threads. This is an
orchestrator, not a way to relax a method's scientific or resource gates.
"""
from __future__ import annotations
from release_runtime import private_literal as _private_literal, release_path as _release_path

import argparse
import csv
import fcntl
import hashlib
import json
import os
import resource
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(_release_path('path_99d8c1c6fde2a6b9'))
SCOPES = ("full_home", *(f"outer_{i}" for i in range(5)))
SOURCE_FILES = (
    "src/04_groups_stats.py", "src/04b_tcrdist_projection.py",
    "src/04_record_tcrdist_stop.py", "src/05_association.py",
    "src/05_predictive.py", "src/05_external.py", "src/05_sensitivity.py",
    "src/05_ablation.py",
    "src/05_decision.py", "src/06_antigen.py",
    "config/analysis_spec.json", "config/cohort_samples.tsv", "folds/folds.tsv",
    "atlas/batch_report.json",
)
GI = 1024 ** 3


class Stop(RuntimeError):
    """A missing or inconsistent receipt prevents further analysis."""


def need(ok: bool, message: str) -> None:
    if not ok:
        raise Stop(message)


def sha(path: Path) -> str:
    need(path.is_file(), f"Missing file: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read(path: Path) -> dict:
    need(path.is_file(), f"Missing receipt: {path}")
    value = json.loads(path.read_text())
    need(isinstance(value, dict), f"Receipt is not a JSON object: {path}")
    return value


def write_once(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        need(read(path) == value, f"Frozen receipt changed: {path}")
        return
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    os.replace(tmp, path)


def validate_spec(root: Path) -> None:
    spec = read(root / "config/analysis_spec.json")
    groups = spec.get("groups", {})
    need(groups.get("primary_radius") == 24 and
         groups.get("tcrdist_seed_max_per_training_patient") == 512 and
         groups.get("gliph2_seed_max_per_training_patient") == 2048 and
         groups.get("technical_gates", {}).get("estimated_six_scope_runtime_hours_max") == 96,
         "Frozen radius, seed sizes or 96-hour resource cap changed")
    need(spec.get("association", {}).get("candidate_gate") ==
         "adjusted q<0.05 and positive Home direction",
         "Home candidate multiplicity rule changed")


def validate_home_nodes(root: Path) -> dict:
    directory = root / "groups/full_home"
    desc = read(directory / "description.json")
    need(desc.get("scope") == "full_home" and desc.get("n_patients") == 137 and
         bool(desc.get("train_patient_sha256")) and
         desc.get("folds_sha256") == sha(root / "folds/folds.tsv"),
         "Full Home node description is incomplete or changed")
    for name, field in (("patient_nodes.parquet", "patient_nodes_sha256"),
                        ("clone_nodes.parquet", "clone_nodes_sha256")):
        need(desc.get(field) == sha(directory / name),
             f"Full Home {name} hash differs from description")
    return desc


def validate_pilots(root: Path) -> tuple[dict, dict, dict]:
    """Require both pilots to be complete or to have an explicit resource stop."""
    validate_spec(root)
    batch = read(root / "atlas/batch_report.json")
    need(batch.get("status") == "complete" and
         batch.get("outcome_blind") is True and
         isinstance(batch.get("equal_depth_reads"), int) and
         batch["equal_depth_reads"] > 0,
         "Label-blind batch/common-depth report is missing or incomplete")
    desc = validate_home_nodes(root)
    group = root / "groups"
    proof_path = group / "full_home/tcrdist_exact_v_lower_bound_diagnostic.json"
    proof = read(proof_path)
    need(proof.get("status") in {"PASS_TECHNICAL_LOWER_BOUND", "STOP_RESOURCE"} and
         proof.get("scope") == "full_home" and proof.get("method") == "tcrdist3" and
         proof.get("candidate_strategy") == "exact_V_cost_pruning" and
         proof.get("script_sha256") == sha(root / "src/04b_tcrdist_projection.py") and
         proof.get("radius_grid") == [16, 20, 24] and
         proof.get("train_patient_sha256") == desc["train_patient_sha256"] and
         proof.get("node_description_sha256") == sha(group / "full_home/description.json") and
         proof.get("seed_sha256") == sha(group / "full_home/tcrdist_seed_nodes.parquet") and
         proof.get("global_node_file_sha256") == sha(group / "tcrdist_all_nodes.parquet") and
         proof.get("reference_equivalence", {}).get("distance_equal_to_tcrdist3_TCRrep") is True and
         all(proof.get("exhaustive_recall_by_radius", {}).get(str(r), {}).get("status") == "PASS" and
             proof["exhaustive_recall_by_radius"][str(r)].get("missed_by_v_bound") == 0
             for r in (16, 20, 24)),
         "Exact-V technical pilot is missing, incomplete or inconsistent")
    hours = float(proof.get("six_scope_hours_with_1p5_output_factor", -1))
    need(hours >= 0 and ((proof["status"] == "PASS_TECHNICAL_LOWER_BOUND") ==
                          (hours <= 96)), "Exact-V pilot status and resource estimate disagree")
    all_nodes = read(group / "tcrdist_all_nodes_receipt.json")
    need(all_nodes.get("status") == "complete" and all_nodes.get("schema_version") == 3 and
         all_nodes.get("output_sha256") == sha(group / "tcrdist_all_nodes.parquet"),
         "Global AA/V/J node receipt is not complete")

    pilot = group / "full_home/gliph2_pilot_256"
    inp_path, run_path = pilot / "input_receipt.json", pilot / "run_receipt.json"
    inp, run = read(inp_path), read(run_path)
    need(inp.get("scope") == "full_home" and inp.get("max_per_patient") == 256 and
         inp.get("seed") == 20260929 and inp.get("simulation_depth") == 1000 and
         inp.get("n_input_patients") == 137 and inp.get("not_full_coverage") is True and
         inp.get("train_patient_sha256") == desc["train_patient_sha256"] and
         inp.get("node_description_sha256") == sha(group / "full_home/description.json") and
         inp.get("input_sha256") == sha(pilot / "input.tsv") and
         run.get("input_receipt_sha256") == sha(inp_path) and
         run.get("params_sha256") == sha(pilot / "params.txt") and
         run.get("binary_sha256") == sha(Path(run.get("binary", ""))) and
         run.get("train_patient_sha256") == desc["train_patient_sha256"] and
         run.get("node_description_sha256") == inp["node_description_sha256"] and
         run.get("max_seconds") == 3600 and run.get("exit_code") in {0, 124},
         "GLIPH2 256 technical pilot source or run receipt is incomplete")
    elapsed = float(run.get("elapsed_seconds", -1))
    need(elapsed > 0, "GLIPH2 256 pilot has no measured elapsed time")
    if run["exit_code"] == 0:
        cluster = pilot / "gliph2_result_cluster.csv"
        need(run.get("cluster_csv_sha256") == sha(cluster) and
             cluster.stat().st_mtime_ns >= inp_path.stat().st_mtime_ns and
             cluster.stat().st_mtime_ns <= run_path.stat().st_mtime_ns,
             "GLIPH2 256 result is missing, stale or changed")
    else:
        need(run.get("cluster_csv_sha256") is None,
             "Timed-out GLIPH2 pilot unexpectedly claims a cluster result")
    return proof, inp, run


def gliph_cost_gate(root: Path, inp: dict, run: dict) -> dict:
    elapsed = float(run["elapsed_seconds"])
    # A label-blind, conservative quadratic scaling from 256 to 2048 seeds.
    six_hours = 6 * elapsed * (2048 / 256) ** 2 * 1.5 / 3600
    passed = run["exit_code"] == 0 and six_hours <= 96 and six_hours / 6 <= 24
    value = {
        "status": "PASS_RESOURCE_ESTIMATE" if passed else "STOP_RESOURCE",
        "method": "gliph2", "pilot_seed_per_patient": 256,
        "formal_seed_per_patient": 2048, "n_scopes": 6,
        "scaling": "quadratic input-size extrapolation with 1.5 output factor",
        "estimated_six_scope_hours": six_hours,
        "pilot_timeout": run["exit_code"] == 124,
        "reason": None if passed else "Technical 256-seed runtime predicts >96h for six 2048-seed scopes, or the pilot timed out",
        "pilot_input_receipt_sha256": sha(root / "groups/full_home/gliph2_pilot_256/input_receipt.json"),
        "pilot_run_receipt_sha256": sha(root / "groups/full_home/gliph2_pilot_256/run_receipt.json"),
    }
    need(inp["max_per_patient"] == 256, "Unexpected GLIPH2 pilot depth")
    return value


class Runner:
    def __init__(self, root: Path):
        self.root = root.resolve()
        self.run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + f"_{os.getpid()}"
        self.step = 0
        self.logs = self.root / "logs" / ("07_after_pilots_" + self.run_id)
        self.logs.mkdir(parents=True, exist_ok=False)
        self.event_path = self.root / "logs/07_after_pilots.jsonl"
        self.home_N_association_started = False

    def event(self, **data: object) -> None:
        record = {"utc": datetime.now(timezone.utc).isoformat(),
                  "run_id": self.run_id, **data}
        with self.event_path.open("a") as handle:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
        print(json.dumps(record, ensure_ascii=False), flush=True)

    def cmd(self, label: str, script: str, *args: str,
            timeout: int = 86400, allow_failure: bool = False) -> int:
        self.verify_sources()
        self.step += 1
        name = f"{self.step:03d}_{label.replace('/', '_')}.log"
        path = self.logs / name
        argv = [sys.executable, str(self.root / "src" / script), *args]
        env = os.environ.copy()
        env.update({"PYTHONDONTWRITEBYTECODE": "1", "OPENBLAS_NUM_THREADS": "1",
                    "OMP_NUM_THREADS": "16", "MKL_NUM_THREADS": "1",
                    "NUMEXPR_NUM_THREADS": "1", "NUMBA_NUM_THREADS": "16",
                    "ARROW_NUM_THREADS": "16"})

        def limit() -> None:
            soft, hard = resource.getrlimit(resource.RLIMIT_AS)
            cap = 128 * GI
            resource.setrlimit(resource.RLIMIT_AS,
                               (min(cap, hard) if hard != resource.RLIM_INFINITY else cap, hard))

        self.event(step=label, state="start", argv=argv, output=str(path))
        started = time.monotonic()
        with path.open("w") as output:
            try:
                result = subprocess.run(argv, cwd=self.root, env=env, stdout=output,
                                        stderr=subprocess.STDOUT, timeout=timeout,
                                        preexec_fn=limit, check=False)
                code = result.returncode
            except subprocess.TimeoutExpired:
                code = 124
        self.event(step=label, state="end", exit_code=code,
                   elapsed_seconds=round(time.monotonic() - started, 3),
                   output_sha256=sha(path))
        self.verify_sources()
        if code and not allow_failure:
            raise Stop(f"Step {label} exited {code}; inspect {path}")
        return code

    def verify_sources(self) -> None:
        lock = read(self.root / "audit/07_orchestrator_source_lock.json")
        need(lock.get("status") == "frozen_preassociation_v2_with_prior_label_hash_access" and
             lock.get("runner_sha256") == sha(Path(__file__)),
             "Orchestrator source lock or running script changed")
        expected = set(SOURCE_FILES) | {
            "groups/full_home/tcrdist_exact_v_lower_bound_diagnostic.json",
            "groups/full_home/gliph2_pilot_256/input_receipt.json",
            "groups/full_home/gliph2_pilot_256/run_receipt.json",
        }
        frozen = lock.get("source_sha256", {})
        need(isinstance(frozen, dict) and set(frozen) == expected,
             "Orchestrator source lock is incomplete")
        for name, frozen_sha in frozen.items():
            need(sha(self.root / name) == frozen_sha,
                 f"Frozen source changed during orchestration: {name}")

    def freeze_sources(self) -> None:
        files = {name: sha(self.root / name) for name in SOURCE_FILES}
        files.update({
            "groups/full_home/tcrdist_exact_v_lower_bound_diagnostic.json":
                sha(self.root / "groups/full_home/tcrdist_exact_v_lower_bound_diagnostic.json"),
            "groups/full_home/gliph2_pilot_256/input_receipt.json":
                sha(self.root / "groups/full_home/gliph2_pilot_256/input_receipt.json"),
            "groups/full_home/gliph2_pilot_256/run_receipt.json":
                sha(self.root / "groups/full_home/gliph2_pilot_256/run_receipt.json"),
        })
        write_once(self.root / "audit/07_orchestrator_source_lock.json", {
            "status": "frozen_preassociation_v2_with_prior_label_hash_access",
            "source_sha256": files, "runner_sha256": sha(Path(__file__)),
            "policy": "one serial child, <=16 threads, RLIMIT_AS 128GiB; no outcome-guided Stage4 selection or retuning",
        })

    def nodes(self, scope: str) -> None:
        path = self.root / "groups" / scope / "description.json"
        if not path.exists():
            need(scope != "full_home", "Full Home source nodes must predate both pilots")
            self.cmd(scope + "_nodes", "04_groups_stats.py", "--root", str(self.root),
                     "derive-train-nodes", "--scope", scope, timeout=21600)
        desc = read(path)
        need(desc.get("scope") == scope and
             desc.get("folds_sha256") == sha(self.root / "folds/folds.tsv") and
             desc.get("train_patient_sha256") and
             desc.get("patient_nodes_sha256") == sha(path.parent / "patient_nodes.parquet") and
             desc.get("clone_nodes_sha256") == sha(path.parent / "clone_nodes.parquet"),
             f"{scope} node description/source mismatch")
        self.event(step=scope + "_nodes", state="verified")

    def stop_gliph(self, scope: str, reason: str, evidence: Path) -> None:
        directory = self.root / "groups" / scope
        path = directory / "gliph2_projection_receipt.json"
        need(not (directory / "gliph2_group_membership.parquet").exists(),
             "Cannot omit GLIPH2 while formal membership exists")
        desc = read(directory / "description.json")
        write_once(path, {
            "status": "STOP_RESOURCE", "scope": scope, "method": "gliph2",
            "reason": reason, "projection_complete": False,
            "label_blind": True, "formal_seed_per_patient": 2048,
            "node_description_sha256": sha(directory / "description.json"),
            "train_patient_sha256": desc["train_patient_sha256"],
            "evidence_path": str(evidence.relative_to(self.root)),
            "evidence_sha256": sha(evidence),
        })
        self.event(step=scope + "_gliph2", state="STOP_RESOURCE", reason=reason)

    def check_method(self, scope: str, method: str) -> str | None:
        directory = self.root / "groups" / scope
        prefix = "gliph2" if method == "gliph2" else "tcrdist_home"
        receipt_path = directory / (prefix + "_projection_receipt.json")
        if not receipt_path.exists():
            return None
        item = read(receipt_path)
        status = str(item.get("status", ""))
        need(item.get("method") == method and item.get("scope") == scope,
             f"{scope} {method} method/scope receipt mismatch")
        if status == "complete_home_scope":
            member = directory / ("gliph2_group_membership.parquet" if method == "gliph2" else
                                  "tcrdist_home_group_membership.parquet")
            need(item.get("membership_sha256") == sha(member) and
                 item.get("train_patient_sha256") ==
                 read(directory / "description.json")["train_patient_sha256"],
                 f"{scope} {method} completed membership changed")
            if method == "tcrdist3":
                need(item.get("projection_complete") is True and
                     item.get("evaluated_nodes") == item.get("scorable_nodes_total") and
                     item.get("scorable_nodes_projected_fraction") == 1.0,
                     f"{scope} tcrdist coverage incomplete")
            return status
        allowed_stops = ({"STOP_RESOURCE", "STOP_SCIENTIFIC"} if method == "tcrdist3" else
                         {"STOP_RESOURCE", "STOP_no_projectable_simple_motifs",
                          "STOP_no_motif_matches", "STOP_no_training_motif_matches",
                          "STOP_giant_motif_family", "STOP_ambiguous_motif_projection"})
        if status in allowed_stops:
            need(bool(item.get("reason")), f"{scope} {method} STOP lacks reason")
            return status
        if method == "tcrdist3" and status in {"initializing", "PAUSED_FIRST_PART"}:
            return status
        raise Stop(f"{scope} {method} has nonfinal receipt status {status}")

    def gliph(self, scope: str, gate: dict) -> None:
        status = self.check_method(scope, "gliph2")
        if status:
            need(not (gate["status"] == "STOP_RESOURCE" and status == "complete_home_scope"),
                 f"{scope} GLIPH2 formal result conflicts with failed technical cost gate")
            self.event(step=scope + "_gliph2", state="reuse", status=status)
            return
        directory = self.root / "groups" / scope
        if gate["status"] == "STOP_RESOURCE":
            self.stop_gliph(scope, gate["reason"], self.root / "audit/07_gliph2_cost_gate.json")
            return
        pilot = directory / "gliph2_pilot_2048"
        inp_path = pilot / "input_receipt.json"
        if not inp_path.exists():
            need(not pilot.exists() or not any(pilot.iterdir()),
                 f"{scope} 2048 input directory has unreceipted contents")
            self.cmd(scope + "_gliph_input", "04_groups_stats.py", "--root", str(self.root),
                     "prepare-gliph2", "--scope", scope, "--max-per-patient", "2048",
                     "--simulation-depth", "1000", timeout=21600)
        inp = read(inp_path)
        need(inp.get("scope") == scope and inp.get("max_per_patient") == 2048 and
             inp.get("seed") == 20260929 and inp.get("simulation_depth") == 1000 and
             inp.get("node_description_sha256") == sha(directory / "description.json") and
             inp.get("input_sha256") == sha(pilot / "input.tsv"),
             f"{scope} formal GLIPH2 input receipt invalid")
        run_path = pilot / "run_receipt.json"
        cluster = pilot / "gliph2_result_cluster.csv"
        if not run_path.exists():
            need(not cluster.exists(), f"{scope} GLIPH2 cluster CSV has no run receipt")
            self.cmd(scope + "_gliph_run", "04_groups_stats.py", "--root", str(self.root),
                     "run-gliph2", "--pilot", str(pilot), "--max-seconds", "86400",
                     timeout=86700, allow_failure=True)
        run = read(run_path)
        need(run.get("input_receipt_sha256") == sha(inp_path) and
             run.get("params_sha256") == sha(pilot / "params.txt") and
             run.get("binary_sha256") == sha(Path(run.get("binary", ""))) and
             run.get("node_description_sha256") == sha(directory / "description.json") and
             run.get("max_seconds") == 86400 and
             run.get("exit_code") in {0, 124},
             f"{scope} GLIPH2 formal run failed without a resource-stop receipt")
        if run["exit_code"] == 124:
            self.stop_gliph(scope, "Prespecified 2048-seed GLIPH2 run reached the 24-hour timeout", run_path)
            return
        need(run.get("cluster_csv_sha256") == sha(cluster) and
             cluster.stat().st_mtime_ns >= inp_path.stat().st_mtime_ns,
             f"{scope} GLIPH2 formal result absent, stale or changed")
        import_path = directory / "membership_receipt.json"
        if not import_path.exists():
            self.cmd(scope + "_gliph_import", "04_groups_stats.py", "--root", str(self.root),
                     "import-gliph2", "--scope", scope, "--csv", str(cluster), timeout=21600)
        imported = read(import_path)
        need(imported.get("status") == "pilot_seed_members_only" and
             imported.get("source_sha256") == sha(cluster) and
             imported.get("seed_membership_sha256") == sha(directory / "gliph2_seed_membership.parquet"),
             f"{scope} GLIPH2 seed import invalid")
        self.cmd(scope + "_gliph_project", "04_groups_stats.py", "--root", str(self.root),
                 "project-gliph2-motifs", "--scope", scope, timeout=86400,
                 allow_failure=True)
        need(self.check_method(scope, "gliph2") is not None,
             f"{scope} GLIPH2 full projection left no receipt")

    def tcr(self, scope: str, project_pass: bool) -> None:
        status = self.check_method(scope, "tcrdist3")
        if status and status not in {"initializing", "PAUSED_FIRST_PART"}:
            need(project_pass or status != "complete_home_scope",
                 f"{scope} completed tcrdist conflicts with failed technical/project gate")
            self.event(step=scope + "_tcrdist", state="reuse", status=status)
            return
        if not project_pass:
            need(status is None, f"{scope} has partial tcrdist run despite failed cost gate")
            self.cmd(scope + "_tcr_stop", "04_record_tcrdist_stop.py",
                     "--root", str(self.root), "--scope", scope, timeout=3600)
            need(str(self.check_method(scope, "tcrdist3")).startswith("STOP"),
                 f"{scope} failed tcrdist gate lacks explicit STOP receipt")
            return
        graph_path = self.root / "groups" / scope / "tcrdist_exact_v_graph_preflight.json"
        if not graph_path.exists():
            self.cmd(scope + "_tcr_graph", "04b_tcrdist_projection.py", "exact-v-graph",
                     "--root", str(self.root), "--scope", scope,
                     "--seed-per-patient", "512", "--seed", "20260929",
                     "--chunk-size", "1000", timeout=86400)
        graph = read(graph_path)
        need(graph.get("status") in {"PASS_TRAIN_GRAPH", "STOP_GIANT_COMPONENT", "STOP_PROOF"} and
             graph.get("scope") == scope and graph.get("candidate_strategy") == "exact_V_cost_pruning" and
             graph.get("script_sha256") == sha(self.root / "src/04b_tcrdist_projection.py") and
             (graph.get("status") == "STOP_PROOF" and bool(graph.get("stop_reason")) or
              graph.get("node_description_sha256") ==
              sha(self.root / "groups" / scope / "description.json")),
             f"{scope} exact-V graph receipt is inconclusive")
        if graph["status"] != "PASS_TRAIN_GRAPH":
            # The project gate only knows the full-Home graph. An outer graph
            # can independently fail the prespecified training-only radius
            # gate; in that case no full projection command is authorized.
            need(scope != "full_home" and status is None,
                 f"{scope} failed graph has a partial or conflicting projection receipt")
            directory = self.root / "groups" / scope
            need(not (directory / "tcrdist_home_group_membership.parquet").exists(),
                 f"{scope} failed graph already has tcrdist membership")
            desc = read(directory / "description.json")
            reason = ("Training-only seed graph has no radius in 16/20/24 below "
                      "the prespecified giant-component limit" if
                      graph["status"] == "STOP_GIANT_COMPONENT" else
                      str(graph["stop_reason"]))
            write_once(directory / "tcrdist_home_projection_receipt.json", {
                "status": "STOP_SCIENTIFIC", "scope": scope, "method": "tcrdist3",
                "reason": reason, "projection_complete": False,
                "projection_population": "home", "label_blind": True,
                "radius_prespecified": 24, "seed_per_patient_prespecified": 512,
                "node_description_sha256": sha(directory / "description.json"),
                "train_patient_sha256": desc["train_patient_sha256"],
                "folds_sha256": desc["folds_sha256"],
                "graph_preflight_sha256": sha(graph_path),
                "proof_sha256": sha(self.root /
                    "groups/full_home/tcrdist_exact_v_lower_bound_diagnostic.json"),
                "project_cost_gate_sha256": sha(self.root /
                    "groups/tcrdist_exact_v_project_cost_gate.json"),
                "script_sha256": sha(Path(__file__)),
            })
            need(self.check_method(scope, "tcrdist3") == "STOP_SCIENTIFIC",
                 f"{scope} exact-V graph failure lacks explicit STOP receipt")
            self.event(step=scope + "_tcrdist", state="STOP_SCIENTIFIC",
                       reason=reason)
            return
        self.cmd(scope + "_tcr_project", "04b_tcrdist_projection.py", "run-exact-home",
                 "--root", str(self.root), "--scope", scope,
                 "--seed-per-patient", "512", "--seed", "20260929",
                 "--radius", "24", "--chunk-size", "10000",
                 "--max-hours", "24", timeout=86700)
        final = self.check_method(scope, "tcrdist3")
        need(final == "complete_home_scope" or str(final).startswith("STOP"),
             f"{scope} exact-V full projection incomplete")

    def integrate_burden(self, scope: str) -> None:
        statuses = [self.check_method(scope, method) for method in ("gliph2", "tcrdist3")]
        need(any(status == "complete_home_scope" for status in statuses),
             f"{scope}: neither group method completed; N-values remain unread")
        path = self.root / "groups" / scope / "integrated_projection_receipt.json"
        if not path.exists():
            self.cmd(scope + "_integrate", "04_groups_stats.py", "--root", str(self.root),
                     "integrate-projections", "--scope", scope, timeout=21600)
        item = read(path)
        need(item.get("status") == "complete_home_scope" and item.get("scope") == scope and
             item.get("membership_sha256") == sha(path.parent / "group_membership.parquet") and
             set(item.get("method_statuses", {})) == {"gliph2", "tcrdist3"} and
             all(item.get("method_receipt_sha256s", {}).get(m) == sha(path.parent /
                 (("gliph2" if m == "gliph2" else "tcrdist_home") + "_projection_receipt.json"))
                 for m in ("gliph2", "tcrdist3")),
             f"{scope} integrated membership receipt invalid")
        burden_path = path.parent / "burden_receipt.json"
        if not burden_path.exists():
            self.cmd(scope + "_burden", "04_groups_stats.py", "--root", str(self.root),
                     "burden", "--scope", scope, timeout=21600)
        burden = read(burden_path)
        need(burden.get("status") == "complete" and burden.get("scope") == scope and
             burden.get("target_cohort") == "home" and burden.get("n_patients") == 137 and
             burden.get("integrated_projection_receipt_sha256") == sha(path) and
             burden.get("group_membership_sha256") == sha(path.parent / "group_membership.parquet") and
             burden.get("patient_group_burden_sha256") == sha(path.parent / "patient_group_burden.parquet") and
             burden.get("n_groups", 0) > 0 and burden.get("n_rows") == 137 * burden["n_groups"],
             f"{scope} patient burden receipt invalid")
        self.event(step=scope + "_burden", state="verified", n_groups=burden["n_groups"])

    def simple_receipt(self, label: str, script: str, receipt: str,
                       statuses: set[str], *args: str, timeout: int = 86400) -> dict:
        path = self.root / receipt
        if not path.exists():
            self.cmd(label, script, *args, timeout=timeout)
        item = read(path)
        need(item.get("status") in statuses, f"{label} has nonfinal status: {item.get('status')}")
        self.event(step=label, state="verified", status=item["status"])
        return item

    def downstream(self) -> None:
        # Stage4 read Home N only to hash provenance; no group choice used it.
        # Freeze the tested universe before any outcome-guided association.
        self.verify_sources()
        self.simple_receipt("association_freeze", "05_association.py",
                            "audit/05_association_universe_lock.json",
                            {"frozen_before_home_N_values"}, "--mode", "freeze")
        self.home_N_association_started = True
        association = self.simple_receipt("association", "05_association.py",
            "audit/05_association_receipt.json", {"complete"}, "--mode", "associate")
        need(association.get("association_tsv_sha256") ==
             sha(self.root / "groups/full_home/association_recorded_home.tsv") and
             association.get("candidate_tsv_sha256") ==
             sha(self.root / "groups/frozen_home_candidates.tsv"),
             "Home association source hashes invalid")
        sensitivity = self.simple_receipt("sensitivity", "05_sensitivity.py",
            "results/05_sensitivity_receipt.json", {"complete"}, "--run")
        sensitivity_sources = {
            "candidate": "groups/frozen_home_candidates.tsv",
            "candidate_receipt": "groups/frozen_home_candidates.json",
            "association": "groups/full_home/association_recorded_home.tsv",
            "burden": "groups/full_home/patient_group_burden.parquet",
            "burden_receipt": "groups/full_home/burden_receipt.json",
            "batch_report": "atlas/batch_report.json",
            "manifest": "config/cohort_samples.tsv", "folds": "folds/folds.tsv",
        }
        need(sensitivity.get("source_sha256") == {
                 key: sha(self.root / name) for key, name in sensitivity_sources.items()
             } and
             sensitivity.get("candidate_sensitivity_sha256") ==
             sha(self.root / "results/05_candidate_sensitivity.tsv") and
             sensitivity.get("equal_depth_repeats_sha256") ==
             sha(self.root / "results/05_equal_depth_repeats.tsv"),
             "Home sensitivity receipt has changed sources or outputs")
        prediction = self.simple_receipt("predictive", "05_predictive.py",
            "prediction/05_receipt.json", {"complete"}, "--run")
        need(prediction.get("n_oof_patients") == 126 and
             prediction.get("n_outer_dictionaries") == 5 and
             prediction.get("output_sha256", {}).get("oof") ==
             sha(self.root / "prediction/05_patient_oof.tsv"),
             "Patient-level OOF receipt incomplete")

        self.cmd("external_check_frozen", "05_external.py", "check-frozen", timeout=3600)
        frozen = self.root / "groups/frozen_home_candidates.tsv"
        with frozen.open(newline="") as handle:
            rows = list(csv.DictReader(handle, delimiter="\t"))
        methods = {r["method"] for r in rows}
        need(methods <= {"gliph2", "tcrdist3"}, "Frozen candidate has unknown method")
        ext = self.root / "groups/full_home"
        if "gliph2" in methods:
            self.simple_receipt("external_gliph2_projection", "05_external.py",
                "groups/full_home/gliph2_external_frozen_projection_receipt.json",
                {"complete_external_frozen_scope"}, "project-gliph2", timeout=86400)
        if "tcrdist3" in methods:
            self.simple_receipt("external_tcrdist_projection", "04b_tcrdist_projection.py",
                "groups/full_home/tcrdist_external_frozen_projection_receipt.json",
                {"complete_external_frozen_scope"}, "run-exact-external-frozen",
                "--root", str(self.root), "--chunk-size", "10000", "--max-hours", "24",
                timeout=86700)
        projection = self.simple_receipt("external_integrate", "05_external.py",
            "groups/full_home/external_projection_receipt.json", {"complete"},
            "integrate", timeout=21600)
        need(projection.get("external_n_read") is False and
             projection.get("frozen_candidates_sha256") == sha(frozen) and
             projection.get("home_membership_sha256") == sha(ext / "group_membership.parquet") and
             projection.get("external_frozen_membership_sha256") ==
             sha(ext / "external_frozen_membership.parquet") and
             projection.get("external_frozen_membership_receipt_sha256") ==
             sha(ext / "external_frozen_membership_receipt.json") and
             projection.get("external_burden_sha256") ==
             sha(ext / "patient_group_burden_external.parquet") and
             projection.get("external_burden_receipt_sha256") ==
             sha(ext / "burden_receipt_external.json") and
             projection.get("n_frozen_groups") == len(rows) and
             projection.get("n_burden_rows") == 202 * len(rows) and
             projection.get("empty_candidate_universe") is (len(rows) == 0),
             "External projection was not frozen before N scoring")
        external_member = read(ext / "external_frozen_membership_receipt.json")
        external_burden = read(ext / "burden_receipt_external.json")
        need(external_member.get("status") == "complete_external_frozen_scope" and
             external_member.get("external_label_read") is False and
             external_member.get("frozen_candidates_sha256") == sha(frozen) and
             external_member.get("membership_sha256") ==
             sha(ext / "external_frozen_membership.parquet") and
             external_member.get("n_frozen_groups") == len(rows) and
             set(external_member.get("methods", {})) == methods and
             external_burden.get("n_patients") == 202 and
             external_burden.get("n_groups") == len(rows) and
             external_burden.get("patient_group_burden_sha256") ==
             sha(ext / "patient_group_burden_external.parquet"),
             "External frozen membership/burden receipt incomplete")
        if not rows:
            need(external_member.get("n_membership_rows") == 0 and
                 external_burden.get("status") == "complete_empty_frozen_universe" and
                 external_burden.get("external_label_read") is False,
                 "Zero-q Home universe lacks an explicit empty External projection")
        else:
            need(external_burden.get("status") == "complete" and
                 external_burden.get("target_cohort") == "external",
                 "Nonempty External burden receipt incomplete")
        direction = self.simple_receipt("external_score", "05_external.py",
            "groups/full_home/external_direction.json", {"complete"},
            "score", timeout=21600)
        need(direction.get("projection_receipt_sha256") == sha(ext / "external_projection_receipt.json") and
             direction.get("direction_tsv_sha256") == sha(ext / "external_direction.tsv") and
             direction.get("n_frozen_groups") == len(rows) and
             direction.get("n_scored_groups") == len(rows) and
             direction.get("n_external_N_known") == 163,
             "External direction receipt or source changed")
        ablation = self.simple_receipt("ablation", "05_ablation.py",
            "results/05_ablation_receipt.json",
            {"complete", "STOP_no_q_candidates", "STOP_no_oof_gain"})
        need(ablation.get("source_home_candidates_sha256") == sha(frozen) and
             ablation.get("source_oof_receipt_sha256") ==
             sha(self.root / "prediction/05_receipt.json") and
             ablation.get("summary_tsv_sha256") ==
             sha(self.root / "results/05_ablation_summary.tsv") and
             ablation.get("fold_tsv_sha256") ==
             sha(self.root / "results/05_ablation_fold_tests.tsv") and
             ablation.get("max_nominations_per_outer_fold", 6) <= 5 and
             ablation.get("heldout_labels_used_for_nomination") is False and
             ablation.get("external_labels_used") is False,
             "Ablation receipt source mismatch")
        decision = self.simple_receipt("decision", "05_decision.py",
            "results/05_final_candidate_gate.json", {"PASS", "STOP", "BLOCKED"},
            "--root", str(self.root))
        if decision["status"] == "BLOCKED":
            raise Stop("Final contribution decision BLOCKED by incomplete upstream evidence: " +
                       str(decision.get("reason", "unspecified")))
        need(decision["status"] == "PASS" or decision.get("evidence_complete") is True,
             "STOP decision lacks a complete negative-evidence receipt")
        need(decision.get("candidate_tsv_sha256") ==
             sha(self.root / "results/05_final_candidates.tsv") and
             decision.get("source_home_candidates_sha256") == sha(frozen) and
             decision.get("source_oof_receipt_sha256") ==
             sha(self.root / "prediction/05_receipt.json") and
             decision.get("source_external_receipt_sha256") ==
             sha(self.root / "groups/full_home/external_direction.json") and
             decision.get("source_ablation_receipt_sha256") ==
             sha(self.root / "results/05_ablation_receipt.json") and
             decision.get("source_sensitivity_receipt_sha256") ==
             sha(self.root / "results/05_sensitivity_receipt.json"),
             "Final candidate table differs from gate receipt")
        if decision["status"] == "PASS" and decision.get("n_candidates", 0) > 0:
            antigen = self.simple_receipt("antigen", "06_antigen.py",
                "antigen/receipt.json", {"complete"})
            need(antigen.get("n_groups") == decision["n_candidates"],
                 "Antigen output group count differs from final gate")
        else:
            antigen = self.simple_receipt("antigen_empty", "06_antigen.py",
                "antigen/receipt.json", {"STOP_gate_not_pass"})
            need(antigen.get("n_groups") == 0 and
                 antigen.get("gate_sha256") ==
                 sha(self.root / "results/05_final_candidate_gate.json"),
                 "Stopped final gate did not yield a provenance-bound empty antigen result")
        self.event(step="all", state="complete", final_gate=decision["status"],
                   n_final_candidates=decision.get("n_candidates", 0))

    def execute(self) -> None:
        proof, inp, run = validate_pilots(self.root)
        self.freeze_sources()
        gliph_gate = gliph_cost_gate(self.root, inp, run)
        write_once(self.root / "audit/07_gliph2_cost_gate.json", gliph_gate)
        self.event(step="technical_pilots", state="verified", tcrdist=proof["status"],
                   gliph2=gliph_gate["status"])
        if proof["status"] == "STOP_RESOURCE" and gliph_gate["status"] == "STOP_RESOURCE":
            raise Stop("Both prespecified grouping methods stopped on technical resource gates; no N values read")
        project_pass = False
        if proof["status"] == "PASS_TECHNICAL_LOWER_BOUND":
            path = self.root / "groups/full_home/tcrdist_exact_v_graph_preflight.json"
            if not path.exists():
                self.cmd("full_home_tcr_graph", "04b_tcrdist_projection.py", "exact-v-graph",
                         "--root", str(self.root), "--scope", "full_home",
                         "--seed-per-patient", "512", "--seed", "20260929",
                         "--chunk-size", "1000", timeout=86400)
            graph = read(path)
            need(graph.get("status") in {"PASS_TRAIN_GRAPH", "STOP_GIANT_COMPONENT", "STOP_PROOF"} and
                 graph.get("scope") == "full_home", "Full Home exact-V graph result inconclusive")
            cost_path = self.root / "groups/tcrdist_exact_v_project_cost_gate.json"
            if not cost_path.exists():
                self.cmd("tcrdist_six_scope_cost_gate", "04b_tcrdist_projection.py",
                         "project-gate-exact-v", "--root", str(self.root),
                         "--max-project-hours", "96", timeout=3600)
            cost = read(cost_path)
            need(cost.get("status") in {"PASS_HOME_FIRST", "STOP"} and
                 cost.get("radius") == 24 and
                 cost.get("max_project_hours") == 96 and
                 (cost["status"] != "PASS_HOME_FIRST" or (
                     cost.get("proof_receipt_sha256") == sha(self.root /
                         "groups/full_home/tcrdist_exact_v_lower_bound_diagnostic.json") and
                     cost.get("graph_receipt_sha256") == sha(path) and
                     cost.get("global_nodes_receipt_sha256") == sha(self.root /
                         "groups/tcrdist_all_nodes_receipt.json"))) and
                 (cost["status"] != "STOP" or bool(cost.get("stop_reason"))) and
                 cost.get("script_sha256") == sha(self.root / "src/04b_tcrdist_projection.py"),
                 "Exact-V six-scope project gate inconclusive")
            project_pass = cost["status"] == "PASS_HOME_FIRST"
        for scope in SCOPES:
            self.nodes(scope)
            self.gliph(scope, gliph_gate)
            self.tcr(scope, project_pass)
            self.integrate_burden(scope)
        self.verify_sources()
        self.downstream()


def self_test() -> None:
    with tempfile.TemporaryDirectory() as temp:
        root = Path(temp)
        pilot = root / "groups/full_home/gliph2_pilot_256"
        pilot.mkdir(parents=True)
        (pilot / "input_receipt.json").write_text("{}")
        (pilot / "run_receipt.json").write_text("{}")
        pass_run = {"exit_code": 0, "elapsed_seconds": 120.0}
        stop_run = {"exit_code": 124, "elapsed_seconds": 3600.0}
        assert gliph_cost_gate(root, {"max_per_patient": 256}, pass_run)["status"] == "PASS_RESOURCE_ESTIMATE"
        assert gliph_cost_gate(root, {"max_per_patient": 256}, stop_run)["status"] == "STOP_RESOURCE"
        with tempfile.TemporaryDirectory() as other:
            target = Path(other) / "lock.json"
            write_once(target, {"status": "PASS"})
            write_once(target, {"status": "PASS"})
            try:
                write_once(target, {"status": "STOP"})
            except Stop:
                pass
            else:
                raise AssertionError("Changed frozen receipt accepted")
        try:
            validate_pilots(root)
        except Stop:
            pass
        else:
            raise AssertionError("Missing technical pilots accepted")
        script = root / "src/04b_tcrdist_projection.py"
        script.parent.mkdir()
        script.write_text("# synthetic source\n")
        outer = root / "groups/outer_0"
        outer.mkdir()
        desc = outer / "description.json"
        desc.write_text(json.dumps({"train_patient_sha256": "synthetic",
                                    "folds_sha256": "synthetic"}))
        graph = outer / "tcrdist_exact_v_graph_preflight.json"
        graph.write_text(json.dumps({
            "status": "STOP_GIANT_COMPONENT", "scope": "outer_0",
            "candidate_strategy": "exact_V_cost_pruning",
            "script_sha256": sha(script), "node_description_sha256": sha(desc),
        }))
        (root / "groups/full_home/tcrdist_exact_v_lower_bound_diagnostic.json").write_text("{}")
        (root / "groups/tcrdist_exact_v_project_cost_gate.json").write_text("{}")
        runner = Runner(root)
        runner.tcr("outer_0", True)
        assert runner.check_method("outer_0", "tcrdist3") == "STOP_SCIENTIFIC"
        assert runner.step == 0, "Failed outer graph launched a projection command"
    print("07 orchestration self-test PASS (resource gate, frozen receipt, missing pilot, outer graph STOP)")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true",
                        help="Start serial analysis only after both technical pilots pass validation")
    parser.add_argument("--self-test", action="store_true",
                        help="Use temporary synthetic fixtures; never run the real analysis")
    args = parser.parse_args()
    need(args.execute != args.self_test, "Choose exactly one of --execute or --self-test")
    if args.self_test:
        self_test()
        return
    root = ROOT
    lock_path = root / "audit/07_runner.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("w") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise Stop("Another 07 orchestration process holds the lock") from exc
        runner = Runner(root)
        try:
            runner.execute()
        except Exception as exc:
            runner.event(step="all", state="STOP_ORCHESTRATION",
                         reason=f"{type(exc).__name__}: {exc}",
                         home_N_association_started=runner.home_N_association_started)
            raise


if __name__ == "__main__":
    main()
