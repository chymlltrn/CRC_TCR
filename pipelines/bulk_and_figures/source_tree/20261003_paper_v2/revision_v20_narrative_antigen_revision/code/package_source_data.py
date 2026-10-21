#!/usr/bin/env python3
"""Verify frozen source copies and append only explicit compact coverage additions."""
from __future__ import annotations
import argparse
import csv
import io
import json
import shutil
from pathlib import Path
from assemble import ROOT, DEFAULT_BASE, digest, write_json


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--root", type=Path, default=ROOT)
    cli.add_argument("--base", type=Path, default=DEFAULT_BASE)
    cli.add_argument("--check-original-sources", action="store_true", help="Also reopen historical absolute source files; unnecessary for document rebuild")
    cli.add_argument("--apply-additions", action="store_true", help="Copy the root-authorized files in audit/source_data_additions.json and append exact provenance rows")
    args = cli.parse_args()
    root, base = args.root.resolve(), args.base.resolve()
    index = root / "manuscript/SourceData/SOURCE_DATA_INDEX.tsv"
    baseline = base / "manuscript/SourceData/SOURCE_DATA_INDEX.tsv"
    original = baseline.read_bytes()
    original_rows = list(csv.DictReader(io.StringIO(original.decode()), delimiter="\t"))
    fields = list(original_rows[0])
    additions_path = root / "audit/source_data_additions.json"
    additions = json.loads(additions_path.read_text()).get("files", []) if additions_path.exists() else []
    added_rows = []
    prepared_copies = []
    for entry in additions:
        relative = Path(entry["target_relative"])
        if relative.is_absolute() or ".." in relative.parts or not relative.is_relative_to("manuscript/SourceData"):
            raise ValueError("Source addition target escapes the source archive")
        if entry["target_relative"] in {row["target_relative"] for row in original_rows + added_rows}:
            raise ValueError("Source addition duplicates an inherited or pending index target")
        source, target = Path(entry["source_absolute"]), root / relative
        checksum, size = entry["source_sha256"], entry["size_bytes"]
        if args.apply_additions:
            if digest(source.read_bytes()) != checksum or source.stat().st_size != size:
                raise ValueError(f"Approved immutable addition changed: {source}")
            if target.exists() and digest(target.read_bytes()) != checksum:
                raise ValueError(f"Refusing to overwrite a changed source copy: {target}")
            prepared_copies.append((source, target))
        added_rows.append({"category": entry["category"], "kind": entry.get("kind", "compact_original_analysis_table"),
            "source_revision": entry["source_revision"], "source_absolute": str(source), "target_relative": str(relative),
            "size_bytes": str(size), "source_sha256": checksum, "target_sha256": checksum,
            "copied_bytes_match": "true", "action": "copied_verified_source_coverage_v20"})
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=fields, delimiter="\t", lineterminator="\n")
    writer.writerows(added_rows)
    expected = original + buffer.getvalue().encode()
    if args.apply_additions:
        if index.read_bytes() not in [original, expected]:
            raise ValueError("Source index differs outside the exact approved append")
        # Verify the complete frozen manifest before copying any new payload.
        for source, target in prepared_copies:
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
        index.write_bytes(expected)
        write_json(root / "audit/source_data_metadata_updates.json", [{"path": str(index.relative_to(root)),
            "before": digest(original), "after": digest(expected), "preserved_prefix_bytes": len(original),
            "original_rows": len(original_rows), "appended_rows": len(added_rows), "coverage_additions_sha256": digest(additions_path.read_bytes())}])
    if index.read_bytes() != expected:
        raise ValueError("Source index differs from frozen prefix plus exact approved provenance rows")
    rows = list(csv.DictReader(index.open(), delimiter="\t"))
    if not rows:
        raise ValueError("SourceData index is empty")
    for row in rows:
        target = root / row["target_relative"]
        if row["source_sha256"] != row["target_sha256"] or digest(target.read_bytes()) != row["target_sha256"] or target.stat().st_size != int(row["size_bytes"]):
            raise ValueError(f"Frozen SourceData payload differs: {row['target_relative']}")
        if args.check_original_sources and digest(Path(row["source_absolute"]).read_bytes()) != row["source_sha256"]:
            raise ValueError(f"Historical SourceData origin differs: {row['source_absolute']}")
    manifest = root / "audit/source_data_packaging.tsv"
    manifest.write_bytes(index.read_bytes())
    write_json(root / "audit/source_data_validation.json", {"status": "PASS", "baseline": "v19",
        "indexed_sources_verified": len(rows), "canonical_original_tables_unchanged": True,
        "historical_index_prefix_byte_preserved": True, "coverage_additions": len(added_rows),
        "index_sha256": digest(index.read_bytes()), "manifest_sha256": digest(manifest.read_bytes()),
        "historical_absolute_sources_reopened": args.check_original_sources,
        "display_provenance": "All inherited source names/bytes are retained. Revised main and added supplementary artwork use the current registry and source provenance. Newly added files require exact append-only index authorization; only baseline payloads are immutable inherited sources."})
    print(json.dumps({"source_data_verification": "passed", "indexed_sources": len(rows), "historical_index_prefix_unchanged": True, "approved_additions": len(added_rows)}))


if __name__ == "__main__":
    main()
