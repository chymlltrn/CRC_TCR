#!/usr/bin/env python3
"""Inventory compact current plot inputs without approving or copying payloads."""
from __future__ import annotations
from release_runtime import private_literal as _private_literal, release_path as _release_path
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

ROOT = Path(_release_path('path_052a3da416b1c4ab'))
FOLDERS = ('figure2_v20', 'figure4_v20', 'figure5_v20', 'supplement_v20')
METADATA = (
    'analysis/figure2_v20/source_manifest.json',
    'analysis/figure2_v20/plot_source_index.json',
    'analysis/figure2_v20/panel_mapping.json',
    'analysis/figure2_v20/reference_size_encoding_decision.json',
    'analysis/figure2_v20/sources/A_sampling_receipt.json',
    'analysis/figure2_v20/sources/B_sampling_receipt.json',
    'analysis/figure4_v20/input_source_manifest.json',
    'analysis/figure5_v20/inputs_lineage.json',
    'analysis/figure5_v20/qpcr_statistical_scope.md',
)


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument('--root', type=Path, default=ROOT)
    args = cli.parse_args()
    root = args.root.resolve()
    selected = {path for folder in FOLDERS for path in (root / 'analysis' / folder).rglob('*.tsv')
                if 'source_baseline' not in path.parts}
    selected.update(root / relative for relative in METADATA if (root / relative).is_file())
    files = []
    for path in sorted(selected):
        relative = path.relative_to(root)
        folder = relative.parts[1]
        checksum = hashlib.sha256(path.read_bytes()).hexdigest()
        target = Path('manuscript/SourceData/v20/plot_sources') / Path(*relative.parts[1:])
        files.append({'category': folder, 'kind': 'current_plot_table' if path.suffix == '.tsv' else 'plot_source_provenance',
            'source_revision': 'v20', 'source_absolute': str(path), 'source_sha256': checksum,
            'target_relative': str(target), 'size_bytes': path.stat().st_size})
    freezes = [root / 'analysis/figure2_v20/final_freeze.json', root / 'analysis/figure4_v20/final_freeze.json',
               root / 'analysis/figure5_v20/final_freeze.json'] + [root / f'analysis/supplement_v20/S{i}_final_freeze.json' for i in range(1, 7)]
    receipt = {'status': 'CANDIDATE_PENDING_COMPLETE_OWNER_FREEZES_AND_ROOT_APPROVAL', 'version': 'v20',
        'created_utc': datetime.now(timezone.utc).isoformat(), 'original_index_prefix_rows': 727,
        'file_count': len(files), 'total_bytes': sum(item['size_bytes'] for item in files),
        'files': files, 'owner_freezes': {str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
                                        for path in freezes if path.is_file()},
        'pending_owner_freezes': [str(path.relative_to(root)) for path in freezes if not path.is_file()],
        'archived_antigen_rankings': 'All original per-method/context top10 and complete ranking payloads remain in the inherited727 index. The former 90-row S16 display is archived data, not a current displayed figure.',
        'index_or_payloads_mutated': False}
    destination = root / 'audit/source_data_additions_candidate_v20.json'
    destination.write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({key: receipt[key] for key in ('status', 'file_count', 'total_bytes', 'pending_owner_freezes')}))


if __name__ == '__main__':
    main()
