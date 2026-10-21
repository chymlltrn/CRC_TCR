"""Convert the official, frozen TCGA-CDR workbook to its unmodified data table.

Input and output are external patient-level data and must not be committed.
Source: https://gdc.cancer.gov/about-data/publications/PanCan-Clinical-2018
"""
import argparse
import csv
import hashlib
from pathlib import Path
import openpyxl

p = argparse.ArgumentParser()
p.add_argument('workbook', type=Path)
p.add_argument('output_tsv', type=Path)
a = p.parse_args()
expected = 'ea594c0fbb6731477c7ac511fab449ca9c38b0d42d269591ed9f5c4090e75a5a'
assert hashlib.sha256(a.workbook.read_bytes()).hexdigest() == expected, 'Unexpected CDR workbook version'
w = openpyxl.load_workbook(a.workbook, read_only=True, data_only=True)
rows = w['TCGA-CDR'].iter_rows(values_only=True)
header = list(next(rows)); header[0] = 'source_row_id'
records = list(rows)
idx = header.index('bcr_patient_barcode')
assert len(records) == 11160 and len({r[idx] for r in records}) == len(records)
a.output_tsv.parent.mkdir(parents=True, exist_ok=True)
with a.output_tsv.open('w', newline='') as f:
    writer = csv.writer(f, delimiter='\t')
    writer.writerow(header); writer.writerows(records)
w.close()
print(f'Wrote {len(records)} records; redaction retained for endpoint-specific exclusions')
