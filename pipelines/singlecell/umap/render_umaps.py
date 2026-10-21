#!/usr/bin/env python
"""Render the recomputed scTCR-covered RNA embedding without changing bulk evidence.

Coordinates are joined 1:1 by cell_name. Every cell, matching key, library
count, abundance and author subtype is retained. Only new output files are
written; the accepted source CSVs are read-only. All panels use identical
full-population coordinate limits with equal aspect. Scatter layers are
rasterized within PDF/SVG; text and axes remain editable vectors.
"""
from release_config import release_path, private_value
from pathlib import Path
import argparse
import hashlib
import json
import math
import platform

import numpy as np
import pandas as pd
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import FuncNorm, ListedColormap, Normalize
from matplotlib.patches import Rectangle
from scipy.spatial import cKDTree

BASE = Path(str(release_path('render_umaps_source_26')))
OUT = BASE / 'recomputed_scTCR_20261009'
SUBTYPE = BASE / 'GSE236581_scRNA_scTCR_Tcells_subtypes_UMAP.csv'
COORD = OUT / 'analysis' / 'coordinates.tsv.gz'
SPECS = [
    ('home', 'Home 137', 'GSE236581_Home137', 78745, 137),
    ('external', 'External 202', 'GSE236581_External202', 72653, 202),
    ('union', 'Home + External 339', 'GSE236581_Home137_External202', 98215, 339),
]
BG_DARK = '#0B0D19'
FG_DARK = '#EDF0F6'
BULK_SUFFIX = '_bulkCPM_AAVJ_ge2_singleBeta_UMAP'
FIGURES = OUT / 'figures'
DATA = OUT / 'plot_data'
DPI = 300
MANIFEST = {'panels': [], 'checks': {}, 'sources': {}}
CLUSTER = 'leiden_r0_8'

mpl.rcParams.update({
    'font.family': 'sans-serif', 'font.sans-serif': ['Arial', 'DejaVu Sans'],
    'pdf.fonttype': 42, 'ps.fonttype': 42, 'svg.fonttype': 'none',
    'font.size': 9, 'figure.dpi': 100, 'savefig.dpi': DPI,
    'axes.linewidth': .7, 'axes.grid': False,
})


def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def record_source(path):
    path = Path(path)
    MANIFEST['sources'][str(path)] = {'sha256': sha(path), 'bytes': path.stat().st_size}


def join_coordinates(source, coords):
    assert source.cell_name.is_unique and coords.cell_name.is_unique
    result = source.rename(columns={'UMAP1': 'original_UMAP1', 'UMAP2': 'original_UMAP2'}).merge(
        coords, how='left', on='cell_name', sort=False, validate='one_to_one', indicator=True)
    assert len(result) == len(source)
    assert result['_merge'].eq('both').all()
    assert result.cell_name.equals(source.cell_name)
    assert np.isfinite(result[['UMAP1', 'UMAP2']]).all().all()
    result = result.drop(columns='_merge')
    immutable = [c for c in source.columns if c not in ('UMAP1', 'UMAP2')]
    pd.testing.assert_frame_equal(result[immutable], source[immutable], check_dtype=True, check_exact=True)
    coord_check = coords.set_index('cell_name').loc[result.cell_name, ['UMAP1', 'UMAP2']].to_numpy()
    np.testing.assert_array_equal(result[['UMAP1', 'UMAP2']].to_numpy(), coord_check)
    result['coordinate_basis'] = 'RNA re-embedding of all 369063 scTCR-covered T cells; see analysis/parameters.json'
    result['coordinate_source'] = str(COORD)
    result['author_subtype_definition'] = 'Original publication SubCellType retained; not reassigned from new Leiden clusters'
    return result


def limits(coords):
    xy = coords[['UMAP1', 'UMAP2']].to_numpy()
    lo = xy.min(axis=0)
    hi = xy.max(axis=0)
    pad = .018 * (hi - lo)
    return (float(lo[0] - pad[0]), float(hi[0] + pad[0])), (float(lo[1] - pad[1]), float(hi[1] + pad[1]))


def axes_format(ax, xlim, ylim, dark=False, arrows=True):
    fg = FG_DARK if dark else '#303846'
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_aspect('equal', adjustable='box')
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel('')
    ax.set_ylabel('')
    for spine in ax.spines.values():
        spine.set_visible(False)
    if arrows:
        arrow_kw = {'arrowstyle': '-|>', 'lw': .9, 'color': fg,
                    'mutation_scale': 9, 'shrinkA': 0, 'shrinkB': 0}
        ax.annotate('', xy=(.09, -.031), xytext=(-.031, -.031),
                    xycoords='axes fraction', arrowprops=arrow_kw, annotation_clip=False)
        ax.annotate('', xy=(-.031, .09), xytext=(-.031, -.031),
                    xycoords='axes fraction', arrowprops=arrow_kw, annotation_clip=False)
        ax.text(.035, -.066, 'UMAP 1', transform=ax.transAxes, ha='center', va='top', fontsize=8.3, color=fg)
        ax.text(-.065, .035, 'UMAP 2', transform=ax.transAxes, ha='center', va='center',
                rotation=90, fontsize=8.3, color=fg)


def dense_anchors(frame, group_col, short_label=None):
    """Find a real observed cell in a locally dense region per group.

    Labels are explanatory annotations only. This density calculation never
    changes coordinates, cell inclusion, abundance, or scatter colors.
    """
    rows = []
    for group, part in frame.groupby(group_col, sort=True, observed=True):
        xy = part[['UMAP1', 'UMAP2']].to_numpy()
        candidates = np.linspace(0, len(xy) - 1, min(len(xy), 2500)).astype(int)
        tree = cKDTree(xy)
        distances, _ = tree.query(xy[candidates], k=min(32, len(xy)))
        scores = distances.mean(axis=1) if distances.ndim > 1 else distances
        idx = candidates[int(np.argmin(scores))]
        rows.append({'group': str(group), 'label': short_label(str(group)) if short_label else str(group),
                     'count': len(part), 'anchor_cell_name': part.cell_name.iloc[idx],
                     'anchor_UMAP1': xy[idx, 0], 'anchor_UMAP2': xy[idx, 1]})
    return pd.DataFrame(rows)


def label_groups(ax, anchors, dark=False, fontsize=8.5):
    """Choose short-label boxes near real dense points, with no box overlap."""
    fig = ax.figure
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    fg, bg = (FG_DARK, BG_DARK) if dark else ('#263141', '#FFFFFF')
    bounds = ax.get_window_extent(renderer)
    occupied = []
    placed = []
    # Larger populations are placed first; every group is still labelled.
    for row in anchors.sort_values(['count', 'group'], ascending=[False, True]).to_dict('records'):
        original = ax.transData.transform((row['anchor_UMAP1'], row['anchor_UMAP2']))
        probe = ax.text(0, 0, row['label'], fontsize=fontsize, fontweight='bold', ha='center', va='center')
        extent = probe.get_window_extent(renderer)
        w, h = extent.width + 8, extent.height + 6
        probe.remove()
        candidates = [(0, 0)]
        for radius in (13, 22, 32, 44, 58, 74, 95, 120):
            for angle in np.linspace(0, 2 * np.pi, 16, endpoint=False):
                candidates.append((radius * np.cos(angle), radius * np.sin(angle)))
        best = None
        fixed = 'label_UMAP1' in row and 'label_UMAP2' in row
        if fixed:
            fixed_pos = ax.transData.transform((row['label_UMAP1'], row['label_UMAP2']))
            candidates = [(fixed_pos[0] - original[0], fixed_pos[1] - original[1])]
        for dx, dy in candidates:
            cx, cy = original + [dx, dy]
            cx = np.clip(cx, bounds.x0 + w / 2 + 1, bounds.x1 - w / 2 - 1)
            cy = np.clip(cy, bounds.y0 + h / 2 + 1, bounds.y1 - h / 2 - 1)
            box = mpl.transforms.Bbox.from_bounds(cx - w / 2, cy - h / 2, w, h)
            overlap = sum(box.overlaps(b) for b in occupied)
            distance = np.linalg.norm([cx - original[0], cy - original[1]])
            score = overlap * 100000 + distance
            if best is None or score < best[0]:
                best = (score, box, (cx, cy), overlap)
            if overlap == 0 and distance < .01:
                break
        _, box, pos, overlap = best
        assert overlap == 0, 'Short label placement collision: ' + row['label']
        occupied.append(box)
        point = (np.array([row['label_UMAP1'], row['label_UMAP2']]) if fixed
                 else ax.transData.inverted().transform(pos))
        displacement = np.linalg.norm(np.array(pos) - original)
        annotation = ax.annotate(row['label'], xy=(row['anchor_UMAP1'], row['anchor_UMAP2']),
            xytext=point, ha='center', va='center', fontsize=fontsize, fontweight='bold',
            color=fg, zorder=8, annotation_clip=False,
            bbox={'boxstyle': 'round,pad=0.18', 'facecolor': bg, 'edgecolor': 'none', 'alpha': .87},
            arrowprops=({'arrowstyle': '-', 'lw': .6, 'color': fg, 'alpha': .65,
                         'shrinkA': 3, 'shrinkB': 1} if displacement > 7 else None))
        placed.append({**row, 'label_UMAP1': point[0], 'label_UMAP2': point[1],
                       'label_displacement_pixels_at_100dpi': displacement,
                       'label_collision_count': overlap})
    return pd.DataFrame(placed).sort_values('group')


def assert_plot(ax, points, xy, xlim, ylim, vals=None):
    np.testing.assert_array_equal(np.asarray(points.get_offsets()), xy)
    if vals is not None:
        np.testing.assert_array_equal(np.asarray(points.get_array()), vals)
    np.testing.assert_array_equal(ax.get_xlim(), xlim)
    np.testing.assert_array_equal(ax.get_ylim(), ylim)
    assert ((xy[:, 0] >= xlim[0]) & (xy[:, 0] <= xlim[1]) &
            (xy[:, 1] >= ylim[0]) & (xy[:, 1] <= ylim[1])).all()
    ax.figure.canvas.draw()
    p = ax.transData.transform([[0, 0], [1, 0], [0, 1]])
    assert abs(np.linalg.norm(p[1] - p[0]) - np.linalg.norm(p[2] - p[0])) < 1e-8


def save(fig, name, source_input, plot_data, cells, **extra):
    outputs = {ext: str(FIGURES / (name + '.' + ext)) for ext in ('png', 'pdf', 'svg')}
    for ext, path in outputs.items():
        fig.savefig(path, dpi=DPI, facecolor=fig.get_facecolor(), transparent=False)
    manifest = {'panel': name, 'source_input': str(source_input), 'coordinates_source': str(COORD),
                'plot_data': str(plot_data), 'outputs': outputs, 'cells': int(cells),
                'source_csv': str(source_input), 'png': outputs['png'], 'pdf': outputs['pdf'], 'svg': outputs['svg'],
                'cell_count': int(cells), 'coordinate_mismatches': 0, 'point_filtering': False, 'coordinate_clipping': False,
                'equal_aspect': True, **extra}
    MANIFEST['panels'].append(manifest)
    print('RENDERED', name, outputs['png'], flush=True)
    plt.close(fig)


def categorical_legend(fig, names, counts, colors, left=.69, right=.97, top=.86, bottom=.155,
                       label_title='Author subtypes', prefix=None):
    fig.text(left, top + .036, label_title, fontsize=10.3, fontweight='bold', color='#263141')
    fig.text(right, top + .036, 'Cells', fontsize=9, color='#647082', ha='right')
    dy = min(.022, (top - bottom) / max(len(names) - 1, 1))
    for i, name in enumerate(names):
        y = top - i * dy
        fig.add_artist(Rectangle((left, y - .005), .008, .010, transform=fig.transFigure,
                                facecolor=colors[name], edgecolor='#7C8490', linewidth=.25))
        display = name.replace('_', ' ') if prefix is None else prefix(name)
        fig.text(left + .016, y, display, fontsize=8.6, va='center', color='#263141')
        fig.text(right, y, f'{counts[name]:,}', fontsize=8.3, va='center', ha='right', color='#647082')


def render_subtype(frame, xlim, ylim, plot_path):
    names = sorted(frame.SubCellType.unique())
    colors = frame.groupby('SubCellType').subtype_color_hex.first().to_dict()
    assert frame.groupby('SubCellType').subtype_color_hex.nunique().eq(1).all()
    counts = frame.SubCellType.value_counts().to_dict()
    anchors = dense_anchors(frame, 'SubCellType', lambda x: x.split('_')[0])
    fig = plt.figure(figsize=(13.2, 8.8), facecolor='white')
    ax = fig.add_axes([.06, .16, .595, .735])
    xy = frame[['UMAP1', 'UMAP2']].to_numpy()
    scatter = ax.scatter(xy[:, 0], xy[:, 1], c=frame.subtype_color_hex,
                         s=1.45, marker='.', alpha=.9, linewidths=0, rasterized=True)
    axes_format(ax, xlim, ylim)
    labels = label_groups(ax, anchors)
    label_path = DATA / 'author_subtype_label_positions.tsv'
    labels.to_csv(label_path, sep='\t', index=False)
    categorical_legend(fig, names, counts, colors)
    fig.text(.06, .959, 'scTCR-covered T cells', fontsize=18, fontweight='bold', color='#152539')
    fig.text(.06, .923, f'{len(frame):,} cells  |  RNA-based re-embedding  |  32 original author subtypes',
             fontsize=10.5, color='#647082')
    fig.text(.06, .070, 'UMAP recomputed on the scTCR-covered RNA T-cell subset; all cells, tissues and time points retained.',
             fontsize=9, color='#536173')
    fig.text(.06, .040, 'c01–c32 denote the original author subtypes. New Leiden clusters are shown separately; no biological relabelling.',
             fontsize=9, color='#536173')
    assert_plot(ax, scatter, xy, xlim, ylim)
    save(fig, '01_scTCR_Tcells_author_subtypes', SUBTYPE, plot_path, len(frame),
         color_definition='Exact original subtype_color_hex; original draw_order', label_data=str(label_path),
         xlim=xlim, ylim=ylim, annotation_mismatches=0, color_mismatches=0)
    return colors, anchors


def render_leiden(frame, xlim, ylim, plot_path):
    categories = sorted(frame[CLUSTER].astype(str).unique(), key=lambda x: int(x))
    frame = frame.copy()
    frame[CLUSTER] = frame[CLUSTER].astype(str)
    cmap_values = np.vstack([mpl.colormaps['tab20'](np.linspace(0, 1, 20)),
                            mpl.colormaps['tab20b'](np.linspace(0, 1, 20)),
                            mpl.colormaps['tab20c'](np.linspace(0, 1, 20))])
    colors = {x: mpl.colors.to_hex(cmap_values[i % len(cmap_values)]) for i, x in enumerate(categories)}
    labels_map = {x: f'L{int(x):02d}' for x in categories}
    counts = frame[CLUSTER].value_counts().to_dict()
    frame['leiden_display_label'] = frame[CLUSTER].map(labels_map)
    frame['leiden_color_hex'] = frame[CLUSTER].map(colors)
    leiden_path = DATA / 'scTCR_Tcells_leiden_r0_8.csv.gz'
    frame[['cell_name', 'UMAP1', 'UMAP2', CLUSTER, 'leiden_display_label', 'leiden_color_hex',
           'SubCellType', 'subtype_color_hex']].to_csv(leiden_path, index=False, compression='gzip')
    pd.crosstab(frame[CLUSTER], frame.SubCellType).to_csv(DATA / 'leiden_r0_8_by_author_subtype_counts.tsv', sep='\t')
    fig = plt.figure(figsize=(11.4, 8.8), facecolor='white')
    ax = fig.add_axes([.07, .16, .685, .735])
    xy = frame[['UMAP1', 'UMAP2']].to_numpy()
    scatter = ax.scatter(xy[:, 0], xy[:, 1], c=frame.leiden_color_hex,
                         s=1.45, marker='.', alpha=.9, linewidths=0, rasterized=True)
    axes_format(ax, xlim, ylim)
    anchors = dense_anchors(frame, CLUSTER, lambda x: labels_map[x])
    labels = label_groups(ax, anchors)
    label_path = DATA / 'leiden_r0_8_label_positions.tsv'
    labels.to_csv(label_path, sep='\t', index=False)
    categorical_legend(fig, categories, counts, colors, left=.80, right=.967, label_title='Leiden r = 0.8',
                       prefix=lambda x: labels_map[x])
    fig.text(.07, .959, 'scTCR-covered T cells: new RNA clusters', fontsize=18, fontweight='bold', color='#152539')
    fig.text(.07, .923, f'{len(frame):,} cells  |  Leiden resolution 0.8  |  {len(categories)} clusters',
             fontsize=10.5, color='#647082')
    fig.text(.07, .070, 'Identical RNA subset embedding and full coordinate range as the author-subtype and bulk TCR panels.',
             fontsize=9, color='#536173')
    fig.text(.07, .040, 'L labels are new computational clusters; original biological subtype labels remain unchanged.',
             fontsize=9, color='#536173')
    assert_plot(ax, scatter, xy, xlim, ylim)
    save(fig, '02_scTCR_Tcells_new_Leiden_r0_8', SUBTYPE, leiden_path, len(frame),
         label_data=str(label_path), xlim=xlim, ylim=ylim, cluster_column=CLUSTER,
         categories=len(categories), new_annotation=False)


def make_norm(source):
    knots = np.array(json.loads(source.color_norm_log10_knots.iloc[0]))
    positions = np.array(json.loads(source.color_norm_positions.iloc[0]))
    vmin, vmax = float(source.color_vmin.iloc[0]), float(source.color_vmax.iloc[0])
    norm = FuncNorm((lambda x: np.interp(x, knots, positions),
                     lambda x: np.interp(x, positions, knots)), vmin=vmin, vmax=vmax, clip=False)
    for col, value in [('color_vmin', vmin), ('color_vmax', vmax)]:
        assert source[col].eq(value).all()
    for val in source.color_norm_log10_knots.unique():
        np.testing.assert_array_equal(json.loads(val), knots)
    for val in source.color_norm_positions.unique():
        np.testing.assert_array_equal(json.loads(val), positions)
    np.testing.assert_array_equal(np.asarray(norm(source.log10_bulk_mean_cpm)), source.color_normalized.to_numpy())
    return norm, knots, positions


def render_bulk(frame, source_path, key, title, num_libraries, dark, xlim, ylim, norm, subtype_names, basic=False):
    bg, fg, soft = (BG_DARK, FG_DARK, '#BAC2D1') if dark else ('#FFFFFF', '#263141', '#647082')
    cmap = ListedColormap(mpl.colormaps['magma'](np.linspace(.30, 1., 256)), name='magma_nebula_030_100') if dark else mpl.colormaps['magma']
    if basic:
        cmap = mpl.colormaps['viridis']
    style = 'basic_viridis' if basic else ('nebula' if dark else 'magma')
    prefix = dict(home='03', external='04', union='05')[key] if not dark else dict(home='06', external='07', union='08')[key]
    if basic:
        prefix = dict(home='11', external='12', union='13')[key]
    point_size = 2.3 if basic else 2.
    name = f'{prefix}_{key}_bulk_TCR_{style}'
    data = frame.copy()
    data['colormap'] = cmap.name
    data['colormap_base'] = 'viridis' if basic else 'magma'
    data['colormap_base_min'] = .3 if dark else 0.
    data['colormap_base_max'] = 1.
    data['background_color'] = bg
    data['umap1_min'], data['umap1_max'] = xlim
    data['umap2_min'], data['umap2_max'] = ylim
    data['point_size_pt2'] = point_size
    data['point_marker'] = '.'
    data['point_alpha'] = 1.
    data['render_dpi'] = DPI
    data['render_width_inches'] = 10.6
    data['render_height_inches'] = 8.6
    data['halo_sizes_pt2'] = '[10.0,4.0]' if dark else '[]'
    data['halo_alphas'] = '[0.025,0.08]' if dark else '[]'
    data['halo_definition'] = 'Decorative same-coordinate scatter only; no smoothing or density estimation' if dark else 'None'
    data['axis_style'] = 'full_subset_limits; equal_aspect; short_direction_arrows'
    xy = data[['UMAP1', 'UMAP2']].to_numpy()
    vals = data.log10_bulk_mean_cpm.to_numpy()
    if basic:
        data['color_normalization'] = 'shared_linear_on_log10_CPM'
        data['color_norm_log10_knots'] = json.dumps([norm.vmin, norm.vmax])
        data['color_norm_positions'] = '[0.0,1.0]'
        data['color_clip'] = False
        data['color_normalized'] = norm(vals)
        data['draw_order'] = 'original_basic_source_order; abundance_ascending'
    rgba = cmap(norm(vals))
    for j, col in enumerate(['color_red', 'color_green', 'color_blue', 'color_alpha']):
        data[col] = rgba[:, j]
    plot_path = DATA / f'{key}_bulk_TCR_{style}.csv.gz'
    data.to_csv(plot_path, index=False, compression='gzip')
    fig = plt.figure(figsize=(10.6, 8.6), facecolor=bg)
    ax = fig.add_axes([.075, .195, .68, .705], facecolor=bg)
    if dark:
        for size, alpha in [(10., .025), (4., .08)]:
            halo = ax.scatter(xy[:, 0], xy[:, 1], c=vals, cmap=cmap, norm=norm,
                             s=size, marker='.', alpha=alpha, linewidths=0, rasterized=True)
            np.testing.assert_array_equal(halo.get_offsets(), xy)
            np.testing.assert_array_equal(halo.get_array(), vals)
    scatter = ax.scatter(xy[:, 0], xy[:, 1], c=vals, cmap=cmap, norm=norm, s=point_size, marker='.',
                         alpha=1., linewidths=0, rasterized=True, zorder=3)
    axes_format(ax, xlim, ylim, dark=dark)
    if basic:
        anchors = pd.read_csv(DATA / f'{key}_magma_label_positions.tsv', sep='\t', float_precision='round_trip')
        anchor_rows = data.set_index('cell_name').loc[anchors.anchor_cell_name]
        np.testing.assert_array_equal(anchor_rows.SubCellType.to_numpy(), anchors.group.to_numpy())
        np.testing.assert_array_equal(anchor_rows[['UMAP1', 'UMAP2']].to_numpy(),
                                      anchors[['anchor_UMAP1', 'anchor_UMAP2']].to_numpy())
    else:
        anchors = dense_anchors(data, 'SubCellType', lambda x: x.split('_')[0])
    assert set(anchors.group) == set(data.SubCellType.unique())
    labels = label_groups(ax, anchors, dark=dark, fontsize=8.1)
    if basic:
        for col in ['anchor_cell_name', 'anchor_UMAP1', 'anchor_UMAP2', 'label_UMAP1', 'label_UMAP2']:
            np.testing.assert_array_equal(labels[col].to_numpy(), anchors.sort_values('group')[col].to_numpy())
    label_path = DATA / f'{key}_{style}_label_positions.tsv'
    labels.to_csv(label_path, sep='\t', index=False)
    cax = fig.add_axes([.815, .330, .023, .400], facecolor=bg)
    cbar = fig.colorbar(scatter, cax=cax)
    cbar.set_ticks(np.arange(-3., 4.))
    cbar.set_ticklabels(['0.001', '0.01', '0.1', '1', '10', '100', '1,000'])
    cbar.minorticks_off()
    cbar.outline.set_visible(False)
    cbar.ax.tick_params(labelsize=8.5, length=3, width=.7, colors=soft, pad=5)
    fig.text(.795, .825, 'Mean bulk TRB CPM', fontsize=10.5, fontweight='bold', color=fg)
    fig.text(.795, .797, 'Shared log scale' if basic else 'Shared piecewise log scale', fontsize=8.6, color=soft)
    fig.text(.795, .286, 'Full abundance range', fontsize=8.6, color=soft)
    fig.text(.795, .262, f'{10**norm.vmin:.6f}–{10**norm.vmax:,.2f} CPM', fontsize=8.2, color=soft)
    fig.text(.795, .218, f'{len(anchors)} author subtypes', fontsize=8.8, color=soft)
    fig.text(.795, .194, 'c labels: see subtype panel', fontsize=8.2, color=soft)
    fig.text(.075, .957, title, fontsize=18, fontweight='bold', color=fg)
    fig.text(.075, .919, f'{len(data):,} matched scTCR-covered T cells (single beta)  |  {num_libraries} bulk libraries',
             fontsize=10.5, color=soft)
    fig.text(.075, .114, 'New RNA subset UMAP; same coordinates in all panels. c labels retain original author subtypes.',
             fontsize=9, color=soft)
    fig.text(.075, .084, 'Exact TRB CDR3 AA + V/J match; aggregated bulk read count ≥ 2/library; matched single-beta cells only.',
             fontsize=8.7, color=soft)
    fig.text(.075, .055, 'CPM averaged over all primary libraries, including absent/below-threshold zeros; no cell values recalculated.',
             fontsize=8.6, color=soft)
    if dark:
        fig.text(.075, .027, 'Decorative halos use the same point coordinates and colors; no smoothing or density interpolation.',
                 fontsize=8.2, color=soft)
    assert_plot(ax, scatter, xy, xlim, ylim, vals)
    np.testing.assert_array_equal(scatter.get_facecolors(), rgba)
    save(fig, name, source_path, plot_path, len(data), cohort=key, style=style, label_data=str(label_path),
         xlim=xlim, ylim=ylim, abundance_mismatches=0, denominator_mismatches=0, key_mismatches=0,
         color_normalization_mismatches=0, color_mismatches=0,
         shared_log_knots=json.loads(data.color_norm_log10_knots.iloc[0]),
         shared_log_positions=json.loads(data.color_norm_positions.iloc[0]))
    return data


def render_overview(subtype, bulk_frames, norm, xlim, ylim, subtype_path):
    fig = plt.figure(figsize=(14.0, 13.6), facecolor='white')
    slots = [[.07, .555, .395, .375], [.535, .555, .395, .375],
             [.07, .135, .395, .375], [.535, .135, .395, .375]]
    panel_metadata = []
    for i, (pos, key) in enumerate(zip(slots, ['subtypes', 'home', 'external', 'union'])):
        ax = fig.add_axes(pos)
        frame = subtype if key == 'subtypes' else bulk_frames[key]
        xy = frame[['UMAP1', 'UMAP2']].to_numpy()
        if key == 'subtypes':
            scatter = ax.scatter(xy[:, 0], xy[:, 1], c=frame.subtype_color_hex, s=1.1, marker='.',
                                 alpha=.9, linewidths=0, rasterized=True)
            title = 'A  Original author subtypes'
        else:
            scatter = ax.scatter(xy[:, 0], xy[:, 1], c=frame.log10_bulk_mean_cpm, s=1.5, marker='.',
                                 cmap='magma', norm=norm, alpha=1., linewidths=0, rasterized=True)
            title = chr(ord('A') + i) + '  ' + next(s[1] for s in SPECS if s[0] == key)
        axes_format(ax, xlim, ylim, arrows=False)
        anchors = dense_anchors(frame, 'SubCellType', lambda x: x.split('_')[0])
        labels = label_groups(ax, anchors, fontsize=7.4)
        labels.to_csv(DATA / f'overview_{key}_label_positions.tsv', sep='\t', index=False)
        ax.set_title(title + f'  ({len(frame):,} cells)', fontsize=11, fontweight='bold', loc='left', pad=9, color='#263141')
        assert_plot(ax, scatter, xy, xlim, ylim)
        panel_metadata.append({'panel': key, 'cells': len(frame)})
    cax = fig.add_axes([.56, .063, .32, .013])
    sm = mpl.cm.ScalarMappable(cmap='magma', norm=norm)
    cbar = fig.colorbar(sm, cax=cax, orientation='horizontal')
    cbar.set_ticks([-3., -2., -1., 0., 1., 3.])
    cbar.set_ticklabels(['0.001', '0.01', '0.1', '1', '10', '1,000'])
    cbar.outline.set_visible(False)
    cbar.minorticks_off()
    cbar.ax.tick_params(labelsize=8, length=2)
    fig.text(.56, .086, 'Mean bulk TRB CPM · shared piecewise log scale', fontsize=9, color='#536173')
    fig.text(.07, .967, 'scTCR-covered T cells: new RNA embedding', fontsize=19, fontweight='bold', color='#152539')
    fig.text(.07, .075, 'c01–c32: original author subtypes; full names in panel 01.', fontsize=8.6, color='#536173')
    fig.text(.07, .050, 'Bulk: matched single-beta cells; exact CDR3 AA + V/J;', fontsize=8.6, color='#536173')
    fig.text(.07, .031, '≥ 2 reads/library; primary-library mean CPM including zeros.', fontsize=8.6, color='#536173')
    save(fig, '09_overview_author_subtypes_and_bulk_TCR', SUBTYPE, subtype_path, len(subtype),
         xlim=xlim, ylim=ylim, subpanels=panel_metadata,
         bulk_plot_data={k: str(DATA / f'{k}_bulk_TCR_magma.csv.gz') for k in bulk_frames})


def render_diagnostics(frame, xlim, ylim, plot_path):
    fields = ['Tissue', 'treatment_stage', 'Patient']
    fields = [x if x in frame.columns else 'Treatment' for x in fields]
    fig = plt.figure(figsize=(18.6, 7.2), facecolor='white')
    cells = len(frame)
    for i, field in enumerate(fields):
        ax = fig.add_axes([.035 + i * .325, .21, .294, .60])
        values = frame[field].fillna('Not available').astype(str)
        names = sorted(values.unique())
        palette = mpl.colormaps['tab20'](np.linspace(0, 1, max(20, len(names))))
        colors = {name: mpl.colors.to_hex(palette[j]) for j, name in enumerate(names)}
        xy = frame[['UMAP1', 'UMAP2']].to_numpy()
        scatter = ax.scatter(xy[:, 0], xy[:, 1], c=values.map(colors), s=1.25, marker='.',
                             alpha=.85, linewidths=0, rasterized=True)
        axes_format(ax, xlim, ylim, arrows=False)
        ax.set_title(field.replace('_', ' '), fontsize=13, fontweight='bold', loc='left', pad=12, color='#263141')
        handles = [mpl.lines.Line2D([0], [0], marker='o', linestyle='none', markersize=5,
                    color=colors[name], label=f'{name} ({(values == name).sum():,})') for name in names]
        ax.legend(handles=handles, loc='upper center', bbox_to_anchor=(.5, -.035), frameon=False,
                  ncol=4 if len(names) > 8 else 2, fontsize=7.4, handletextpad=.25, columnspacing=.65)
        assert_plot(ax, scatter, xy, xlim, ylim)
        pd.DataFrame({'category': names, 'n_cells': [(values == name).sum() for name in names],
                      'color_hex': [colors[name] for name in names]}).to_csv(DATA / f'diagnostic_{field}_counts_colors.tsv', sep='\t', index=False)
    fig.text(.035, .94, 'RNA subset embedding: sample-context diagnostics', fontsize=18, fontweight='bold', color='#152539')
    fig.text(.035, .89, f'{cells:,} scTCR-covered T cells; identical coordinates and full limits across panels', fontsize=10.5, color='#647082')
    fig.text(.035, .035, 'Descriptive diagnostics only. Tissue and treatment remain biological context; visual mixing alone does not establish adequate batch correction.', fontsize=9.5, color='#536173')
    save(fig, '10_diagnostic_tissue_treatment_patient', SUBTYPE, plot_path, cells,
         xlim=xlim, ylim=ylim, diagnostic_fields=fields, diagnostic_only=True)


def main():
    global MANIFEST
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--panels', nargs='+', help='Optional two-digit panel IDs to redraw; merge into the existing manifest.')
    args = parser.parse_args()
    selected = set(args.panels or [])
    wanted = lambda panel_id: not selected or panel_id in selected
    existing_manifest = OUT / 'audit' / 'plot_manifest.json'
    if selected:
        assert existing_manifest.exists(), 'Selective redraw requires the original complete manifest.'
        MANIFEST = json.loads(existing_manifest.read_text())
        MANIFEST['panels'] = [x for x in MANIFEST['panels'] if x['panel'][:2] not in selected]
    assert COORD.exists(), f'Coordinates not ready: {COORD}'
    for directory in (FIGURES, DATA):
        directory.mkdir(parents=True, exist_ok=True)
    record_source(SUBTYPE)
    record_source(COORD)
    params = OUT / 'analysis' / 'parameters.json'
    if params.exists():
        record_source(params)
        MANIFEST['analysis_parameters'] = json.loads(params.read_text())
    coords = pd.read_csv(COORD, sep='\t', float_precision='round_trip', dtype={CLUSTER: str})
    original = pd.read_csv(SUBTYPE, float_precision='round_trip')
    assert len(original) == len(coords) == 369063
    assert set(original.cell_name) == set(coords.cell_name)
    assert coords.cell_name.is_unique
    assert CLUSTER in coords
    frame = join_coordinates(original, coords)
    xlim, ylim = limits(coords)
    subtype_path = DATA / 'scTCR_Tcells_author_subtypes.csv.gz'
    if wanted('01') or not subtype_path.exists():
        frame.to_csv(subtype_path, index=False, compression='gzip')
        pd.DataFrame({'SubCellType': frame.SubCellType, 'subtype_color_hex': frame.subtype_color_hex}).value_counts().rename('n_cells').reset_index().sort_values('SubCellType').to_csv(DATA / 'author_subtype_counts_colors.tsv', sep='\t', index=False)
    if wanted('01'):
        render_subtype(frame, xlim, ylim, subtype_path)
    if wanted('02'):
        render_leiden(frame, xlim, ylim, subtype_path)
    norms = []
    bulk_frames = {}
    for key, title, stem, expected_n, num_libraries in SPECS:
        source_path = BASE / (stem + BULK_SUFFIX + '_contrast_magma_sharedPLog.csv')
        nebula_path = BASE / (stem + BULK_SUFFIX + '_nebula_ovstyle.csv')
        record_source(source_path)
        record_source(nebula_path)
        source = pd.read_csv(source_path, float_precision='round_trip')
        nebula = pd.read_csv(nebula_path, float_precision='round_trip')
        assert len(source) == len(nebula) == expected_n
        assert source.single_beta.all() and source.accepted_distinct_beta.eq(1).all()
        assert source.n_primary_libraries.eq(num_libraries).all()
        assert source.bulk_mean_cpm.gt(0).all()
        assert source.bulk_min_reads_per_library_key.eq(2).all()
        invariant = ['cell_name', 'aa_vj_key', 'nt_vj_key', 'bulk_mean_cpm', 'bulk_mean_frequency',
                     'bulk_sum_cpm_qualified', 'bulk_sum_read_count_qualified', 'bulk_supporting_primary_libraries',
                     'n_primary_libraries', 'SubCellType', 'plot_order', 'color_normalized',
                     'color_norm_log10_knots', 'color_norm_positions', 'single_beta']
        pd.testing.assert_frame_equal(source[invariant], nebula[invariant], check_exact=True)
        norm, knots, positions = make_norm(source)
        if norms:
            np.testing.assert_array_equal(knots, norms[0][0])
            np.testing.assert_array_equal(positions, norms[0][1])
        norms.append((knots, positions))
        new_frame = join_coordinates(source, coords)
        check_subtypes = frame.set_index('cell_name').loc[new_frame.cell_name, 'SubCellType'].to_numpy()
        np.testing.assert_array_equal(check_subtypes, new_frame.SubCellType.to_numpy())
        bulk_frames[key] = new_frame
        new_nebula = join_coordinates(nebula, coords)
        for dark in [False, True]:
            panel_id = (dict(home='06', external='07', union='08') if dark else dict(home='03', external='04', union='05'))[key]
            if wanted(panel_id):
                render_bulk(new_nebula if dark else new_frame,
                            nebula_path if dark else source_path,
                            key, title, num_libraries, dark, xlim, ylim, norm, sorted(frame.SubCellType.unique()))
        basic_path = BASE / (stem + BULK_SUFFIX + '.csv')
        record_source(basic_path)
        basic = pd.read_csv(basic_path, float_precision='round_trip')
        assert len(basic) == expected_n and basic.cell_name.is_unique
        basic_invariant = [c for c in invariant if c not in ['cell_name', 'plot_order', 'color_normalized', 'color_norm_log10_knots', 'color_norm_positions']]
        pd.testing.assert_frame_equal(basic.set_index('cell_name').sort_index()[basic_invariant],
                                      source.set_index('cell_name').sort_index()[basic_invariant], check_exact=True)
        assert basic.colormap.eq('viridis').all()
        assert basic.bulk_mean_cpm.is_monotonic_increasing
        assert basic.color_vmin.eq(norm.vmin).all() and basic.color_vmax.eq(norm.vmax).all()
        basic_norm = Normalize(vmin=norm.vmin, vmax=norm.vmax, clip=False)
        new_basic = join_coordinates(basic, coords)
        if wanted(dict(home='11', external='12', union='13')[key]):
            render_bulk(new_basic, basic_path, key, title, num_libraries, False, xlim, ylim, basic_norm,
                        sorted(frame.SubCellType.unique()), basic=True)
    if wanted('09'):
        render_overview(frame, bulk_frames, norm, xlim, ylim, subtype_path)
    if wanted('10'):
        render_diagnostics(frame, xlim, ylim, subtype_path)
    MANIFEST['checks'] = {
        'all_369063_cells_retained': True, 'unique_cell_names': True,
        'one_to_one_coordinate_join': True, 'all_coordinate_join_mismatches': 0,
        'no_coordinate_filtering_clipping_or_interpolation': True,
        'identical_full_population_limits': {'xlim': xlim, 'ylim': ylim},
        'original_author_labels_and_colors_retained': True,
        'new_leiden_labels_are_computational_only': True,
        'all_bulk_source_values_denominators_keys_preserved': True,
        'all_bulk_panel_abundance_mismatches': 0,
        'all_bulk_source_draw_orders_preserved': True,
        'shared_color_normalization_preserved': True,
        'basic_viridis_uses_original_shared_linear_log_scale': True,
        'alpha_only_or_dual_beta_never_assigned_zero': True,
        'label_anchor_is_observed_cell_in_rendered_group': True,
        'short_label_box_collisions': 0,
        'visual_review_status': 'awaiting_parent_visual_verdict',
    }
    MANIFEST['software'] = {'python': platform.python_version(), 'matplotlib': mpl.__version__,
                            'pandas': pd.__version__, 'numpy': np.__version__}
    MANIFEST['panels'] = sorted(MANIFEST['panels'], key=lambda x: x['panel'])
    MANIFEST['last_redrawn_panels'] = sorted(selected) if selected else 'all'
    MANIFEST['renderer'] = {'path': str(Path(__file__).resolve()), 'sha256': sha(__file__)}
    (DATA / 'plot_qa_manifest.json').write_text(json.dumps(MANIFEST, indent=2))
    (OUT / 'audit').mkdir(exist_ok=True)
    (OUT / 'audit' / 'plot_manifest.json').write_text(json.dumps(MANIFEST, indent=2))
    print('DONE', DATA / 'plot_qa_manifest.json', flush=True)


if __name__ == '__main__':
    main()
