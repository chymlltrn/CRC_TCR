"""Check a supplied frozen signature delivery without certifying a new study.

Unlike the historical delivery-seal script, this verifier does not infer visual
approval, independent review, or pre-outcome timing from a rerun.
"""
from pathlib import Path
import os, json, hashlib
import pandas as pd

root = Path(os.environ['CRC_SIGNATURE_ROOT']).expanduser().resolve()
def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()

freeze = json.loads((root / 'freeze.json').read_text())
assert all(sha(root / name) == value for name, value in freeze['files'].items())
assert sha(root / 'protocol_frozen.json') == freeze['protocol_sha256']
m = pd.read_csv(root / 'frozen_signature_markers.tsv', sep='\t')
assert len(m) == 120 and m.groupby('state').size().eq(30).all()
assert not m.duplicated(['state', 'gene']).any() and m.selection_pass.all()
assert (m.logFC > .5).all() and (m.FDR < .05).all()
assert (m.donor_direction_fraction >= .7).all()
assert (m.pct_target_donor_equal >= .1).all() and (m.pct_gap_donor_equal >= .05).all()
assert m.exclusion_reason.isna().all()
assert (m.min_leave_one_donor_out_mean_difference > 0).all()
gmt = {}
for line in (root / 'frozen_signatures.gmt').read_text().splitlines():
    fields = line.split('\t')
    gmt[fields[0]] = fields[2:]
for name, group in m.groupby('signature_id', sort=False):
    assert gmt[name] == group.sort_values('rank').gene.tolist()
print(json.dumps({'status': 'PASS', 'gmt_sha256': sha(root / 'frozen_signatures.gmt'),
                  'n_signatures': len(gmt), 'marker_rows': len(m),
                  'freeze_timestamp_from_input': freeze['frozen_utc'],
                  'scope': 'hashes and frozen marker invariants only; no new visual or independent-review attestation'}, indent=2))
