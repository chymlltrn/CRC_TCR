from release_runtime import private_literal as _private_literal, release_path as _release_path
from pathlib import Path
import hashlib
import json
import shutil
import fitz

out = Path(_release_path('path_cbbb7df30be5e226'))
source = Path(_release_path('path_adb74c94d7c28564'))
png = out / 'Figure1A_schematic_v3_20261012.png'
pdf = out / 'Figure1A_schematic_v3_20261012.pdf'
shutil.copy2(source, png)
pix = fitz.Pixmap(str(png))
width = 180 / 25.4 * 72
height = width * pix.height / pix.width
doc = fitz.open()
page = doc.new_page(width=width, height=height)
page.insert_image(page.rect, filename=str(png))
doc.set_metadata({'title': 'Figure 1A: blood TCR, local tumor immunity and nodal-status analyses', 'subject': 'Schematic; unchanged generated raster embedded in PDF'})
doc.save(pdf, deflate=True)
doc.close()
with fitz.open(pdf) as check:
    assert len(check) == 1
    images = check[0].get_images(full=True)
    assert len(images) == 1
    embedded = fitz.Pixmap(check, images[0][0])
    assert (embedded.width, embedded.height) == (pix.width, pix.height)
    assert embedded.samples == pix.samples
receipt = {
    'generation_mode': 'built-in imagegen; reference-image edits',
    'source_image': str(source),
    'dimensions_px': [pix.width, pix.height],
    'pdf_page_mm': [180, 180 * pix.height / pix.width],
    'pdf_embedded_pixels_match_png': True,
    'editable_vector': False,
    'files': {p.name: {'bytes': p.stat().st_size, 'sha256': hashlib.sha256(p.read_bytes()).hexdigest()} for p in [png, pdf]},
}
(out / 'artifact_receipt.json').write_text(json.dumps(receipt, indent=2) + '\n')
manifest = {str(p.relative_to(out)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(out.rglob('*')) if p.is_file() and p.name != 'SHA256_manifest.json'}
(out / 'SHA256_manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
print(json.dumps(receipt, indent=2))
print('Manifest files:', len(manifest))
