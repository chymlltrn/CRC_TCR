#!/usr/bin/env python3
"""Build the isolated v29 package: five main and three retained supplements."""
from release_runtime import private_literal as _private_literal, release_path as _release_path
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json
import shutil
import fitz
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(_release_path('path_70fe814f385a17f4'))
WORK = ROOT.parents[1]
PREVIOUS = WORK / '20261018_v27_revision/figure_package/delivery'
ORIGINAL = WORK / '20261018_v27_audit/source/20261018_article_v27/Figure_v27'

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def render_array(page):
    pix = page.get_pixmap(dpi=72, alpha=False)
    return np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, 3)

def main():
    for directory in ['Main_Figures', 'Supplementary_Figures', 'Previews']:
        (ROOT / directory).mkdir(parents=True, exist_ok=True)
    entries = []
    for number in range(1, 6):
        entries.append({'label': f'Figure {number}', 'kind': 'main',
                        'source': PREVIOUS / f'Main_Figures/Figure_{number}.pdf',
                        'output': ROOT / f'Main_Figures/Figure_{number}.pdf',
                        'previous_label': f'v28 main Figure {number}',
                        'change': 'none; retain the revised v28 main figure'})
    descriptions = {
        1: 'Global bulk repertoire, nodal associations and model sensitivity',
        3: 'Blood–tissue program sensitivity and paired longitudinal sharing',
        5: 'Q1/Q2 depth, nodal, batch-held-out and external candidate sensitivity'}
    for new, old in enumerate([1, 3, 5], 1):
        entries.append({'label': f'Figure S{new}', 'kind': 'supplementary',
                        'source': ORIGINAL / f'All_Figures_S{old}_v26.pdf',
                        'output': ROOT / f'Supplementary_Figures/Figure_S{new}.pdf',
                        'previous_label': f'original v27 Figure S{old}',
                        'change': 'numbering only; original PDF content unchanged',
                        'description': descriptions[old]})
    review = fitz.open()
    records = []
    toc = []
    previews = []
    for page_number, entry in enumerate(entries, 1):
        source_hash = sha(entry['source'])
        shutil.copyfile(entry['source'], entry['output'])
        assert sha(entry['output']) == source_hash
        doc = fitz.open(entry['output'])
        assert len(doc) == 1
        text = doc[0].get_text()
        assert not any(term in text for term in ['TCGA', _private_literal('private_6e848d2b6b25c025'), 'TOP30', 'Signature definition'])
        review.insert_pdf(doc)
        toc.append([1, entry['label'], page_number])
        preview = ROOT / 'Previews' / (entry['output'].stem + '.png')
        doc[0].get_pixmap(dpi=150, alpha=False).save(preview)
        previews.append((entry['label'], preview))
        records.append(dict(page=page_number, label=entry['label'], kind=entry['kind'],
                            previous_label=entry['previous_label'], change=entry['change'],
                            description=entry.get('description', ''),
                            source=str(entry['source']), source_sha256=source_hash,
                            output=str(entry['output'].relative_to(ROOT)),
                            output_sha256=sha(entry['output']),
                            preview=str(preview.relative_to(ROOT)), pages=1))
        doc.close()
    review.set_toc(toc)
    review.set_metadata({'title': 'v29 figure review: Figures 1–5 and S1–S3',
                         'subject': 'Eight pages; no new signature or TCGA figures',
                         'creator': 'Isolated figure packaging, 2026-10-18'})
    combined = ROOT / 'Figure_order_review.pdf'
    review.save(combined, garbage=4, deflate=True)
    review.close()
    check = fitz.open(combined)
    assert len(check) == 8
    for i, entry in enumerate(entries):
        source = fitz.open(entry['output'])
        np.testing.assert_array_equal(render_array(source[0]), render_array(check[i]))
        source.close()
    assert check.get_toc() == toc
    check.close()
    # A visual navigation preview, separate from all scientific figure contents.
    canvas = Image.new('RGB', (1800, 1500), 'white')
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.truetype('/System/Library/Fonts/Supplemental/Arial Bold.ttf', 25)
    small = ImageFont.truetype('/System/Library/Fonts/Supplemental/Arial.ttf', 17)
    for index, (label, path) in enumerate(previews):
        x, y = (index % 4) * 450, (index // 4) * 750
        draw.text((x + 15, y + 14), label, font=font, fill='#172433')
        if index >= 5:
            draw.text((x + 15, y + 48), f'Original v27 S{[1,3,5][index-5]}', font=small, fill='#526272')
        image = Image.open(path).convert('RGB')
        image.thumbnail((425, 660), Image.Resampling.LANCZOS)
        canvas.paste(image, (x + (450-image.width)//2, y + 78 + (660-image.height)//2))
    canvas.save(ROOT / 'Figure_order_contact_sheet.png')
    manifest = {'created_utc': datetime.now(timezone.utc).isoformat(),
                'scope': 'v29 five main plus three supplementary figures',
                'main_count': 5, 'supplementary_count': 3, 'review_pages': 8,
                'supplementary_mapping': {'original_v27_S1': 'v29_S1',
                                          'original_v27_S3': 'v29_S2',
                                          'original_v27_S5': 'v29_S3'},
                'excluded': ['original v27 S2', 'original v27 S4', 'original v27 S6',
                             'v28 new S5 signature definition', 'v28 new S6 TCGA forests'],
                'tcga_and_new_signature_figures_included': False,
                'individual_pdfs_byte_identical_to_selected_sources': True,
                'combined_pages_pixel_identical_to_individuals_at_72dpi': True,
                'main_figure_order_changed': False, 'original_and_v28_modified': False,
                'review_file': combined.name, 'review_sha256': sha(combined),
                'figures': records}
    (ROOT / 'figure_manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    lines = ['sha256  file']
    for p in sorted(ROOT.rglob('*')):
        if p.is_file() and p.suffix in ['.pdf', '.png']:
            lines.append(f'{sha(p)}  {p.relative_to(ROOT)}')
    (ROOT / 'SHA256SUMS.txt').write_text('\n'.join(lines) + '\n')
    print(json.dumps({k:v for k,v in manifest.items() if k!='figures'}, indent=2))

if __name__ == '__main__':
    main()
