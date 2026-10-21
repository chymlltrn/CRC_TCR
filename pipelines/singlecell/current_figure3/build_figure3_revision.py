#!/usr/bin/env python3
"""Relabel/redraw panel 3f from frozen statistics; preserve other PDF panels."""
from release_config import release_path, private_value
from pathlib import Path
import importlib.util
import hashlib
import json
import shutil
import re
import fitz
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import false_discovery_control

ROOT = release_path('figure3_output_root')
WORK = ROOT.parents[1]
SOURCE = release_path('figure3_source_pdf')
DATA = Path(str(release_path('build_figure3_revision_source_20')))
ORIGINAL_RENDERER = Path(__file__).with_name('original_redraw_figure4.py')
RECT = fitz.Rect(48, 558, 262, 700)

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def freeze_state_selection():
    source = DATA / 'Shared/bulk_abundance/complete_patient_tests.tsv'
    tests = pd.read_csv(source, sep='\t')
    states = tests[(tests.exposure == 'combined_mean') &
                   (tests.weighting == 'cell_equal') &
                   (tests.endpoint_type == 'state')].copy()
    assert len(states) == 128
    for name, group in states.groupby('test'):
        assert len(group) == 64 and group.family_size.eq(64).all()
        assert group.endpoint.nunique() == 32
        np.testing.assert_allclose(false_discovery_control(group.P, method='bh'), group.q,
                                   rtol=0, atol=1e-12)
        group.to_csv(ROOT / f'bulk_state_complete_BH64_{name}.tsv', sep='\t', index=False)
    primary = states[(states.test == 'high_minus_low') & (states.q < .05)].copy()
    assert set(primary.endpoint) == {'c03_CD4_Tn_NR4A2', 'c20_CD8_Tem_GZMK'}
    assert primary.Tissue.eq('Tumor').all() and primary.n.eq(19).all()
    secondary = states[(states.test == 'friedman_three_bins') & (states.q < .05)].copy()
    primary.to_csv(ROOT / 'bulk_state_selected_primary.tsv', sep='\t', index=False)
    secondary.to_csv(ROOT / 'bulk_state_selected_secondary.tsv', sep='\t', index=False)
    rule = {
        'locked_before_TCGA_analysis': True,
        'selection_date': '2026-10-18',
        'source': str(source), 'source_sha256': sha(source),
        'exposure': 'combined_mean', 'weighting': 'cell_equal',
        'endpoint_type': 'state',
        'primary_rule': 'high_minus_low; BH q < 0.05 within all 32 states x 2 tissues (64 tests)',
        'primary_states': sorted(primary.endpoint.unique()),
        'primary_tissue': 'Tumor',
        'secondary_rule': 'friedman_three_bins; separate BH q < 0.05 family of 64 tests; exploratory extension only',
        'secondary_states': sorted(secondary.endpoint.unique()),
        'do_not_pool_families_or_select_by_TCGA_results': True,
        'interpretation': 'States associated with independent blood bulk TCR abundance; not a direct gene-expression high/low contrast or a continuous correlation test.',
        'c27': 'Not selected by this primary bulk-abundance state test. Its independent biological rationale must remain separate.'
    }
    (ROOT / 'bulk_state_selection_lock.json').write_text(json.dumps(rule, indent=2) + '\n')

def redraw():
    spec = importlib.util.spec_from_file_location('frozen_figure_renderer', ORIGINAL_RENDERER)
    renderer = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(renderer)
    source_table = DATA / 'Figure3/Figure4D_abundance_programs.tsv'
    rows = pd.read_csv(source_table, sep='\t')
    assert len(rows) == 12 and rows.family_size.eq(12).all()
    assert rows[rows.tissue == 'Tumor'].n.eq(13).all()
    assert rows[rows.tissue == 'Normal'].n.eq(14).all()
    rows.to_csv(ROOT / 'Figure3f_frozen_statistics.tsv', sep='\t', index=False)
    shutil.copyfile(ORIGINAL_RENDERER, ROOT / 'original_redraw_figure4.py')
    plt.rcParams.update({'font.family': 'Arial', 'font.size': 8,
                         'axes.labelsize': 8, 'xtick.labelsize': 8,
                         'ytick.labelsize': 8, 'axes.linewidth': .6,
                         'pdf.fonttype': 42, 'svg.fonttype': 'none',
                         'savefig.facecolor': 'white'})
    fig = plt.figure(figsize=(RECT.width / 72, RECT.height / 72))
    # These coordinates reproduce the original edited PDF's forest region.
    ax = fig.add_axes([(112.6753 - RECT.x0) / RECT.width,
                       (RECT.y1 - 658.3488) / RECT.height,
                       (249.6833 - 112.6753) / RECT.width,
                       (658.3488 - 576.1438) / RECT.height])
    ax.spines[['top', 'right']].set_visible(False)
    ax.tick_params(width=.6, length=2, pad=2)
    renderer.forest(ax, rows, paired=True, effect='effect')
    ax.set_xlim(-.26, .16)
    ax.set_xticks([-.2, 0, .1])
    ax.set_xlabel('Program score difference\n(high − low bulk TCR abundance)', fontsize=7.6, labelpad=3)
    sig = rows[(rows.tissue == 'Tumor') & (rows.program == 'exhaustion')].iloc[0]
    ax.text(-.24, .65, f'q={sig.q:.3f}'.replace('0.', '.'), fontsize=8,
            color=renderer.COLORS['Tumor'])
    fig.text(9 / RECT.width, 1 - 3 / RECT.height, 'Independent blood bulk TCR',
             fontsize=8.5, fontweight='bold', va='top')
    for ext in ('pdf', 'svg', 'png'):
        fig.savefig(ROOT / f'Figure3f_bulk_TCR_abundance.{ext}', dpi=300)
    plt.close(fig)
    source_doc = fitz.open(SOURCE)
    before = source_doc[0].get_pixmap(matrix=fitz.Matrix(2, 2), alpha=False)
    panel = fitz.open(ROOT / 'Figure3f_bulk_TCR_abundance.pdf')
    page = source_doc[0]
    # General redaction rewrites unrelated transparency groups in this Illustrator
    # PDF. Remove only target text-show operators while retaining font/color state,
    # then cover the old target graphics with the opaque replacement panel.
    removed_text_blocks = 0
    for xref in page.get_contents():
        stream = source_doc.xref_stream(xref)
        def remove_target_text(match):
            nonlocal removed_text_blocks
            block = match.group()
            matrices = re.findall(rb'([-\d.]+)\s+([-\d.]+)\s+([-\d.]+)\s+([-\d.]+)\s+([-\d.]+)\s+([-\d.]+)\s+Tm', block)
            if not matrices:
                return block
            xs = [float(m[4]) for m in matrices]
            ys = [page.rect.height - float(m[5]) for m in matrices]
            if all(RECT.x0 < x < RECT.x1 and RECT.y0 < y < RECT.y1 for x, y in zip(xs, ys)):
                removed_text_blocks += 1
                return re.sub(rb'(?:<[^>]*>|\((?:\\.|[^\\)])*\))\s*Tj', b'', block)
            return block
        updated = re.sub(rb'BT\s.*?ET', remove_target_text, stream, flags=re.S)
        source_doc.update_stream(xref, updated)
    assert removed_text_blocks == 10, removed_text_blocks
    page.draw_rect(RECT, color=None, fill=(1, 1, 1), overlay=True)
    page.show_pdf_page(RECT, panel, 0, keep_proportion=True)
    output = ROOT / 'All_Figures_3_v28_candidate.pdf'
    source_doc.save(output, garbage=4, deflate=True)
    source_doc.close()
    revised = fitz.open(output)
    after = revised[0].get_pixmap(matrix=fitz.Matrix(2, 2), alpha=False)
    a = np.frombuffer(before.samples, dtype=np.uint8).reshape(before.height, before.width, 3)
    b = np.frombuffer(after.samples, dtype=np.uint8).reshape(after.height, after.width, 3)
    changed = np.any(a != b, axis=2)
    outside = np.ones(changed.shape, dtype=bool)
    x0, y0, x1, y1 = [int(v * 2) for v in RECT]
    outside[y0-2:y1+2, x0-2:x1+2] = False
    unchanged = int(np.count_nonzero(changed & outside))
    assert unchanged == 0, f'{unchanged} pixels changed outside target panel'
    text = revised[0].get_text()
    assert 'High − low program score' not in text
    assert 'Independent blood bulk TCR' in text
    assert 'bulk TCR abundance' in text
    revised[0].get_pixmap(dpi=300, alpha=False).save(ROOT / 'All_Figures_3_v28_candidate.png')
    revised[0].get_pixmap(dpi=150, alpha=False).save(ROOT / 'All_Figures_3_v28_preview.png')
    original = fitz.open(SOURCE)
    original[0].get_pixmap(dpi=150, alpha=False).save(ROOT / 'All_Figures_3_v27_reference.png')
    original[0].get_pixmap(dpi=300, clip=RECT, alpha=False).save(ROOT / 'Figure3f_original_reference.png')
    receipt = {'source_pdf': str(SOURCE), 'source_pdf_sha256': sha(SOURCE),
               'source_statistics': str(source_table), 'source_statistics_sha256': sha(source_table),
               'source_renderer': str(ORIGINAL_RENDERER), 'source_renderer_sha256': sha(ORIGINAL_RENDERER),
               'reused_function': 'forest', 'statistics_rows': len(rows),
               'new_analysis_performed': False, 'replacement_rect_points': list(RECT),
               'target_text_blocks_removed': removed_text_blocks,
               'outside_panel_changed_pixels_at_144dpi': unchanged,
               'old_ambiguous_xlabel_removed': True,
               'other_panels_and_order_preserved': True,
               'output_pdf_sha256': sha(output)}
    (ROOT / 'figure3_revision_receipt.json').write_text(json.dumps(receipt, indent=2) + '\n')

if __name__ == '__main__':
    ROOT.mkdir(parents=True, exist_ok=True)
    freeze_state_selection()
    redraw()
