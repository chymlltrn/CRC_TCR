"""Version and explicit current figure inventory for portable v20 builds."""
from __future__ import annotations
from release_runtime import private_literal as _private_literal, release_path as _release_path
import json
from pathlib import Path

ROOT = Path(_release_path('path_052a3da416b1c4ab'))
VERSION = 'v20'
BASE_VERSION = 'v19'
BASE_REVISION = 'revision_v19_visual_reference_revision'

def load_figure_registry(root: Path = ROOT) -> dict:
    registry = json.loads((root / 'audit/figure_registry.json').read_text())
    if registry.get('version') != VERSION or set(registry.get('documents', {})) != {'manuscript', 'supplement'}:
        raise ValueError('Current versioned main/supplement figure registry is required')
    for kind, entries in registry['documents'].items():
        if not entries or [item.get('ordinal') for item in entries] != list(range(1, len(entries) + 1)):
            raise ValueError(f'{kind}: registry ordinals must be unique, sequential and complete')
        for item in entries:
            if item.get('mode') not in {'preserved', 'registered'}: raise ValueError('Unknown figure inventory mode')
            if item['mode'] == 'preserved' and (not isinstance(item.get('baseline_ordinal'), int) or item['baseline_ordinal'] < 1):
                raise ValueError('Preserved artwork requires its positive baseline ordinal')
    policy = registry.get('policy', {}).get('immutable_entries')
    if not isinstance(policy, dict) or set(policy) != set(registry['documents']):
        raise ValueError('Root registry must specify immutable entries for both documents')
    for kind, entries in registry['documents'].items():
        protected = [item for item in entries if item['mode'] == 'preserved']
        if protected != policy[kind]:
            raise ValueError(f'{kind}: immutable entry policy differs from preserved artwork')
        baseline_ordinals = [item['baseline_ordinal'] for item in protected]
        if len(set(baseline_ordinals)) != len(baseline_ordinals):
            raise ValueError(f'{kind}: one preserved baseline figure cannot be duplicated')
    return registry

def expected_figure_counts(root: Path = ROOT) -> dict[str, int]:
    return {kind: len(entries) for kind, entries in load_figure_registry(root)['documents'].items()}

def expected_registrations(root: Path = ROOT) -> dict[str, set[int]]:
    return {kind: {item['ordinal'] for item in entries if item['mode'] == 'registered'}
        for kind, entries in load_figure_registry(root)['documents'].items()}
