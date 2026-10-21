"""Preserve accepted v19 viewports and register the explicit current inventory."""
from __future__ import annotations
import hashlib, json, re
from pathlib import Path
from revision_config import ROOT, VERSION, load_figure_registry, expected_registrations

def sha(data): return hashlib.sha256(data if isinstance(data, bytes) else data.encode()).hexdigest()
def clean_svg(path): return re.sub(r'<\?xml[^>]+>|<!DOCTYPE[^>]+>', '', path.read_text()).strip()
def display_asset_mapping(relative): return relative
def approved_edits(root, base): return {}

def protected_asset(relative, root=ROOT):
    if relative.startswith('manuscript/SourceData/'):
        fixture=root/'reproducibility/frozen_v19_baseline/baseline_fixture_manifest.json'
        manifest=root/'audit/v19_baseline_manifest.json'
        if fixture.is_file(): return relative in json.loads(fixture.read_text())['asset_sha256']
        if manifest.is_file(): return relative in json.loads(manifest.read_text())['files']
        return False
    registry = load_figure_registry(root)['documents']
    name = Path(relative).name
    supp = re.match(r'FigureS(\d+)(?=[_.])', name)
    main = re.match(r'Figure(\d+)(?=[_.])', name)
    if not relative.startswith(('figures/', 'manuscript/Figure_PDF/', 'manuscript/Figure_PNG/')): return False
    if supp: return any(item['mode'] == 'preserved' and item['ordinal'] == int(supp[1]) for item in registry['supplement'])
    if main: return any(item['mode'] == 'preserved' and item['ordinal'] == int(main[1]) for item in registry['manuscript'])
    return relative.startswith('figures/supplement/')

def registered_figures(root, require=True):
    path = root / 'audit/figure_registration.json'
    expected = expected_registrations(root)
    if not path.is_file():
        if require and any(expected.values()): raise ValueError('New artwork registration is required before production')
        return {kind: {} for kind in expected}
    ledger = json.loads(path.read_text())
    if ledger.get('version') != VERSION or ledger.get('status') not in {'REGISTERED_HASH_BOUND', 'PARTIALLY_REGISTERED_HASH_BOUND'}:
        raise ValueError('Invalid current figure registration ledger')
    if ledger.get('registry_sha256') != sha((root / 'audit/figure_registry.json').read_bytes()):
        raise ValueError('Figure registry changed; rerun registration to bind the final inventory')
    records = ledger.get('documents', {})
    for kind, ordinals in expected.items():
        found = {int(key) for key in records.get(kind, {})}
        if not found.issubset(ordinals) or (require and found != ordinals):
            raise ValueError(f'{kind}: finalized registrations do not match the current inventory')
        for key, item in records.get(kind, {}).items():
            if item['display_ordinal'] != int(key): raise ValueError('Figure ordinal mismatch')
            for relative, checksum in item['source_and_output_sha256'].items():
                if not (root / relative).is_file() or sha((root / relative).read_bytes()) != checksum:
                    raise ValueError('Registered drawing/source changed: ' + relative)
    return {kind: records.get(kind, {}) for kind in expected}

def registered_main(root, require=True): return registered_figures(root, require)['manuscript']

def validate_display_mapping(root, base, blueprints):
    registry = load_figure_registry(root)
    registered = registered_figures(root)
    return {'status': 'VERIFIED', 'figure_registry_sha256': sha((root / 'audit/figure_registry.json').read_bytes()),
        'documents': registry['documents'], 'source_data_names_unchanged': True,
        'registration_sha256': sha((root / 'audit/figure_registration.json').read_bytes()),
        'registered_artifact_sha256': {kind: {key: item['source_and_output_sha256'] for key, item in records.items()} for kind, records in registered.items()}}

def current_figure_sections(root, base, kind, baseline_html, section_re, figure_re):
    sections = section_re.findall(baseline_html)
    entries = load_figure_registry(root)['documents'][kind]
    registered = registered_figures(root)[kind]
    folder, prefix = ('main', 'Figure') if kind == 'manuscript' else ('supplement', 'FigureS')
    result = []
    for entry in entries:
        ordinal = entry['ordinal']
        if entry['mode'] == 'preserved':
            original = entry['baseline_ordinal']
            if original > len(sections): raise ValueError('Preserved ordinal is outside the accepted baseline')
            relative = f'figures/{folder}/{prefix}{original}.svg'
            if sha((base / relative).read_bytes()) != sha((root / relative).read_bytes()):
                raise ValueError('Protected original SVG changed: ' + relative)
            section = re.sub(r'\sdata-baseline-ordinal="[^"]*"', '', sections[original - 1], count=1)
            section = re.sub(r'\A<section\b', f'<section data-baseline-ordinal="{original}"', section, count=1)
        else:
            item = registered[str(ordinal)]
            svg = clean_svg(root / f'figures/{folder}/{prefix}{ordinal}.svg')
            classes = item.get('section_class', 'figure-unit tall new-figure')
            section = (f'<section class="{classes}" id="{kind}-figure-{ordinal}" data-registration-ordinal="{ordinal}">'
                f'<h2>{"Figure" if kind == "manuscript" else "Supplementary Figure"} {ordinal}</h2>'
                f'<figure>{svg}</figure><div class="figure-legend"></div></section>')
        result.append(section)
    return result
