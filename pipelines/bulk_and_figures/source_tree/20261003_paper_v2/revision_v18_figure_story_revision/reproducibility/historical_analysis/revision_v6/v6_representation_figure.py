#!/usr/bin/env python3
"""Style-only S2 re-export from the unchanged historical renderer and tables.

No predictions, fits, metrics, confidence intervals or statistics are computed.
The original four-panel renderer is copied byte-for-byte; exactly its two ROC/PR
line-style expressions and output stem are substituted before calling it.
"""
from __future__ import annotations
from release_runtime import private_literal as _private_literal, release_path as _release_path
import argparse
import csv
import hashlib
import inspect
import json
from pathlib import Path
import runpy
import shutil
import sys
from datetime import datetime, timezone

DEFAULT_ROOT = Path(_release_path('path_38c4da8147aa1485'))
TABLES = ['Figure3_historical_representation_metrics.tsv', 'ROC_curves.tsv',
          'PR_curves.tsv', 'Supplement_representation_bag_coverage_patient.tsv']
LINE_STYLES = {'historical_L2_baseline': (0, (7, 3)),
               'tcrbert_dualtask_aa_mean': '-',
               'sceptr_cdr3_only': '-.',
               'cdr3vec_small': ':',
               'kmer3_cosine_svd': (0, (2, 1))}

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def write_json(path, data):
    Path(path).write_text(json.dumps(data, indent=2, ensure_ascii=False)+'\n')

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--root', type=Path, default=DEFAULT_ROOT)
    ap.add_argument('--verify-only', action='store_true')
    args = ap.parse_args()
    root = args.root.resolve()
    if not str(root).startswith(_release_path('path_4bbe6e708c37abce')):
        raise ValueError('Rendering and verification run on the scientific server.')
    out = root/'figures/representation'
    source = out/'source'
    source.mkdir(parents=True, exist_ok=True)
    provenance_path = out/'source_manifest.tsv'
    if args.verify_only:
        checks = []
        for row in csv.DictReader(provenance_path.open(), delimiter='\t'):
            checks.append({'path':row['snapshot_path'], 'pass':sha(root/row['snapshot_path']) == row['sha256'] and sha(row['source_path']) == row['sha256']})
        export = json.loads((out/'Supplementary_Figure_S2_representations.export.json').read_text())
        checks += [{'path':x['path'], 'pass':sha(x['path'])==x['sha256']} for x in export['files'].values()]
        checks += [{'path':'minimum_font', 'pass':export['font_points_min'] >= 7},
                   {'path':'publication_width', 'pass':export['width_mm']==180},
                   {'path':'canvas', 'pass':not export['out_of_canvas_text']}]
        assert all(x['pass'] for x in checks), checks
        write_json(out/'verification_receipt.json', {'status':'passed','checks':checks,'created_utc':datetime.now(timezone.utc).isoformat(),'model_fits':0,'statistics_recomputed':0})
        print('PASS',len(checks),'unchanged sources, exports and geometry')
        return
    prior = root.parent/'revision_v5'
    rows = []
    files = [(prior/'code/nodal_figures_v3.py', source/'nodal_figures_v3.py'),
             (prior/'code/paper_style.py', source/'paper_style.py')]
    files += [(prior/'nodal/source_tables'/name, out/'source_tables'/name) for name in TABLES]
    files += [(prior/'nodal/figures'/('Supplementary_Figure1_representations'+suffix), source/('reference'+suffix)) for suffix in ['.preview.png','.grayscale.png']]
    for src, target in files:
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src, target)
        assert sha(src) == sha(target)
        rows.append({'source_path':str(src),'snapshot_path':str(target.relative_to(root)), 'bytes':target.stat().st_size,'sha256':sha(target)})
    with provenance_path.open('w') as h:
        w = csv.DictWriter(h,fieldnames=['source_path','snapshot_path','bytes','sha256'],delimiter='\t')
        w.writeheader();w.writerows(rows)
    sys.path.insert(0,str(source))
    namespace = runpy.run_path(str(source/'nodal_figures_v3.py'),run_name='frozen_nodal_renderer')
    original = inspect.getsource(namespace['figure_s1'])
    style_expression = "ls='--' if model=='historical_L2_baseline' else '-'"
    assert original.count(style_expression) == 2
    changed = original.replace(style_expression, 'ls=LINE_STYLES[model]')
    changed = changed.replace("out/'figures/Supplementary_Figure1_representations'", "out/'Supplementary_Figure_S2_representations'")
    assert changed != original
    (source/'original_four_panel_function.py').write_text(original)
    (source/'styled_four_panel_function.py').write_text(changed)
    namespace['LINE_STYLES'] = LINE_STYLES
    exec(compile(changed, str(source/'styled_four_panel_function.py'),'exec'),namespace)
    export = namespace['figure_s1'](out)
    namespace['plt'].close('all')
    assert export['font_points_min'] >= 7 and export['width_mm']==180 and not export['out_of_canvas_text']
    manifest_fields = ['figure_number','title','question','pdf_path','svg_path','png_path','preview_path','grayscale_path','export_path','include_in_manuscript','section','source_tables']
    stem = str((out/'Supplementary_Figure_S2_representations').relative_to(root))
    manifest = dict(figure_number='S2',title='Negative representation-model comparisons',question='TCR-BERT, SCEPTR, CDR3vec and 3mer under their common historical protocol.',
                    pdf_path=stem+'.pdf',svg_path=stem+'.svg',png_path=stem+'.png',preview_path=stem+'.preview.png',grayscale_path=stem+'.grayscale.png',export_path=stem+'.export.json',include_in_manuscript='false',section='supplement',source_tables=';'.join(str(out/'source_tables'/x) for x in TABLES))
    with (out/'figure_manifest_rows.tsv').open('w') as h:
        w=csv.DictWriter(h,fieldnames=manifest_fields,delimiter='\t');w.writeheader();w.writerow(manifest)
    caption = ('### Supplementary Figure 2. Historical representation comparisons.\n\n'
               '(A) ROC and (B) precision–recall curves from the frozen historical matched protocol for the L2 baseline, TCR-BERT, SCEPTR, CDR3vec and 3-mer + SVD. Values in the legends are the original AUROC and AUPRC. Colour and distinct line styles identify the models in both panels. '
               '(C) Paired log-loss increments relative to the L2 baseline, with the original fixed-out-of-fold 95% conditional intervals; positive values favour the representation. '
               '(D) Original held-out representation-bag coverage in 126 patients for unique amino-acid sequences, reads and groups. This figure preserves the historical protocol and does not rank algorithms against the current ridge pipeline. All source tables, curves, metrics, intervals, patient points, axes and panel layout are unchanged; this revision changes only ROC/PR line styles.\n')
    (out/'CAPTION.md').write_text(caption)
    write_json(out/'source_receipt.json',{
        'status':'rendered_pending_actual_visual_review','figure_id':'S2','created_utc':datetime.now(timezone.utc).isoformat(),
        'entrypoint':str(Path(__file__).resolve()),'entrypoint_sha256':sha(__file__),
        'source_manifest_sha256':sha(provenance_path),'renderer_source_sha256':sha(source/'nodal_figures_v3.py'),
        'original_function_sha256':sha(source/'original_four_panel_function.py'),'styled_function_sha256':sha(source/'styled_four_panel_function.py'),
        'renderer_changes':['Two identical ROC/PR linestyle expressions replaced by five explicit model styles.','Output file stem redirected to the owned v6 representation directory.'],
        'scientific_table_changes':0,'new_model_fits':0,'statistics_recomputed':0,'line_styles':{k:str(v) for k,v in LINE_STYLES.items()},
        'figure_manifest_sha256':sha(out/'figure_manifest_rows.tsv'),'export_sha256':sha(out/'Supplementary_Figure_S2_representations.export.json')})
    print(json.dumps({k:export[k] for k in ['width_mm','height_mm','font_points_min','out_of_canvas_text']}))

if __name__=='__main__':
    main()
