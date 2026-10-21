#!/usr/bin/env python3
"""Verify the saved Figure 5 artwork independently of the composition process."""
from pathlib import Path
import os
import json
import math
import fitz
import numpy as np

root = Path(os.environ["CRC_QPCR_ROOT"]).expanduser().resolve()
original = fitz.open(root / "inputs/Figure5_v27_reference.pdf")
updated = fitz.open(root / "Figures/Figure5_qPCR_revised_candidate.pdf")
box = fitz.Rect(270, 104, 510.2362060546875, 408)
a = original[0].get_pixmap(matrix=fitz.Matrix(2, 2), alpha=False)
b = updated[0].get_pixmap(matrix=fitz.Matrix(2, 2), alpha=False)
aa = np.frombuffer(a.samples, np.uint8).reshape(a.height, a.width, a.n)
bb = np.frombuffer(b.samples, np.uint8).reshape(b.height, b.width, b.n)
delta = np.max(np.abs(aa.astype(int) - bb.astype(int)), axis=2)
outside = np.ones(delta.shape, bool)
outside[math.floor(box.y0 * 2) - 2:math.ceil(box.y1 * 2) + 2,
        math.floor(box.x0 * 2) - 2:math.ceil(box.x1 * 2) + 2] = False
max_delta = int(delta[outside].max())
assert max_delta <= 2, f"Non-qPCR artwork changed beyond rounding tolerance: {max_delta}"

def outside_words(page):
    return sorted(w[4] for w in page.get_text("words") if not fitz.Rect(w[:4]).intersects(box))

assert outside_words(original[0]) == outside_words(updated[0])
receipt = {"status": "PASS", "render_scale": 2,
           "outside_region_changed_pixels": int(np.sum((delta > 0) & outside)),
           "outside_region_pixels_exceeding_tolerance": int(np.sum((delta > 2) & outside)),
           "outside_region_max_channel_delta": max_delta,
           "channel_tolerance": 2, "tolerance_reason": "Subpixel vector-coordinate rounding during PDF redaction",
           "outside_region_word_content_unchanged": True,
           "panels_preserved": ["a", "b", "e", "f"]}
(root / "QA/unchanged_panels_verification.json").write_text(json.dumps(receipt, indent=2) + "\n")
print(json.dumps(receipt))
