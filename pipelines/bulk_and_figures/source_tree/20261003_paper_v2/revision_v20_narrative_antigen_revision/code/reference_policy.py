"""Authorize exact bibliographic corrections while protecting all other rows."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
from revision_config import VERSION

def reference_row_sha256(row: dict) -> str:
    return hashlib.sha256(json.dumps(row, ensure_ascii=False, sort_keys=True,
        separators=(',', ':')).encode('utf-8')).hexdigest()

def validate_reference_rows(root: Path, baseline: dict, current: dict) -> dict:
    missing = set(baseline) - set(current)
    if missing: raise ValueError('Frozen reference removed: ' + str(sorted(missing)))
    changed = {key for key, row in baseline.items() if current[key] != row}
    path = root / f'audit/reference_corrections_{VERSION}.json'
    authorization = json.loads(path.read_text()) if path.is_file() else {}
    corrections = authorization.get('corrections', [])
    keys = [item.get('id') for item in corrections]
    if len(keys) != len(set(keys)) or set(keys) != changed:
        raise ValueError('Reference changes differ from the exact root-authorized correction set')
    if changed and (authorization.get('version') != VERSION or authorization.get('status') != 'ROOT_AUTHORIZED'
            or authorization.get('row_sha256_encoding') != 'canonical_json_utf8_sorted_keys'):
        raise ValueError('Current root bibliographic authorization is required')
    records = []
    for item in corrections:
        key = item['id']
        if item.get('before_row_sha256') != reference_row_sha256(baseline[key]) or item.get('after_row_sha256') != reference_row_sha256(current[key]):
            raise ValueError('Authorized bibliographic row hash differs: ' + key)
        if baseline[key].get('id') != current[key].get('id') or set(baseline[key]) != set(current[key]):
            raise ValueError('Bibliographic correction cannot change key/schema: ' + key)
        fields = {field for field in current[key] if current[key][field] != baseline[key][field]}
        if not fields.issubset({'formatted_reference', 'url', 'doi'}):
            raise ValueError('Unauthorized reference field: ' + key)
        audit = Path(item['primary_source_audit'])
        if audit.is_absolute() or '..' in audit.parts or not (root/audit).is_file():
            raise ValueError('Primary reference audit must be a packaged relative file')
        if hashlib.sha256((root/audit).read_bytes()).hexdigest() != item.get('primary_source_audit_sha256'):
            raise ValueError('Bibliographic primary-source audit changed: ' + key)
        records.append({'id': key, 'changed_fields': sorted(fields), **item})
    return {'unmodified_baseline_rows_byte_equivalent': True, 'baseline_row_count': len(baseline),
        'root_authorized_corrections': records, 'frozen_reference_rows_unchanged': not changed,
        'correction_authorization': {'path': str(path.relative_to(root)),
            'sha256': hashlib.sha256(path.read_bytes()).hexdigest()} if changed else None}
