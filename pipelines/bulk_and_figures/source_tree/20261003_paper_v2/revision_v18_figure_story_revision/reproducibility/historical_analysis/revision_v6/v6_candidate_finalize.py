#!/usr/bin/env python3
"""Assemble candidate figure namespaces/provenance from completed plots and recorded visual verdicts."""
from release_runtime import private_literal as _private_literal, release_path as _release_path
from pathlib import Path
import argparse,json,hashlib,shutil
import pandas as pd
CAPTIONS=["**Figure 2. Source-stratified evidence for the Q1 and Q2 candidate sequences.** (A) Detection of complete CDR3β amino-acid sequences after summing definition-matching counts within each selected patient or public repertoire and applying count ≥2. Home (137) and External (202) comprise independent patients with one selected library each; horizontal bars show Wilson 95% confidence intervals. Public denominators are libraries, including dependent serial visits and processing contexts, and are shown without patient confidence intervals. All eligible zero-count units remain included. Assay count units and source populations are not harmonized; these fractions do not estimate clinical diagnostic accuracy. Q2 is another nominated candidate. (B) Q1 carriage in all 137 Home patients at summed-read thresholds 1, 2, 5, 10 and 20, separately for complete AA, AA plus assigned TRBV5-5/TRBJ1-4, and the nominated 48-nt sequence. The three definitions are frozen in the candidate definition registry; overlapping curves retain their exact coordinates. Thresholds are count rules, not calibrated assay detection limits. (C) Complete-AA cloneCounts from participant 15's archived primary tumor and serial PBMC repertoires in Le et al. Blood visits appear in chronological categorical order; archival tissue is displayed separately without assuming contemporaneous pairing. The shared tumor/blood NT assigned TRBV5-4/TRBJ1-4 differs from the Home nomination. The 12-week complete-AA sum is 884; its dominant V5-4 row is 867. Counts are not depth-normalized treatment dynamics. All eight peptide/control cultures were Q1-negative. These observations support within-person sequence recurrence, without establishing antigen reactivity, trafficking, treatment response or independent validation of the Home NT target. Raw authenticity and qPCR target/independence gates remain unresolved.",_private_literal('private_5b4884ccaa8cb2a2'),"**Supplementary Figure S7. Historical candidate nomination and current patient-level sensitivity audit.** (A) Two historical nomination routes are shown as reported: a dataset-aware screen yielding 106 sequences and a six-criterion route reporting Q1 and Q2. Both require their six stated conditions. The reported background of 1,367,835 CDR3β amino-acid sequences, per-rule membership and a 106-to-2 transition were not recovered; the diagram does not reconstruct attrition or establish that the two outputs form a sequential filter. Historical statistical and TIL requirements are annotations of the earlier report, not newly confirmed results. (B) Current complete-AA carriage at five fixed summed-read thresholds in all selected Home (137) and External (202) patients, retaining unknown-N patients. (C) Home batch carriage at ≥2 for Q1 AA, Q1 nominated NT and Q2 AA; bars are Wilson 95% confidence intervals. All nominated-Q1-NT carriers occur in batch D. (D) Cumulative candidate reads after ranking all Home patients by each candidate's read sum. Q1's leading patient contributes 96.7% and Q2's 58.7%; solid and dashed lines preserve discrimination in grayscale. Patient prevalence and read concentration measure different features. (E) Descriptive carriage among patients with recorded N0 (70) or N+ (56), with Wilson 95% confidence intervals and exact counts; the 11 unknown-N patients are excluded from this panel only. No new association P value, diagnostic performance or functional claim is inferred. Source-specific technical authenticity and historical qPCR validation remain unresolved."]
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main(root):
    out=root/'candidates';f=out/'figures';s=f/'source';s.mkdir(exist_ok=True);v5=root.parent/'revision_v5'
    provenance=[]
    old_s6=v5/'antigen/figures/Supplementary_Figure_S5.export.json'
    original=json.loads(old_s6.read_text())
    shutil.copy2(old_s6,f/'Supplementary_Figure_S6.source_export.json')
    for source in original['source_tables']:
        p=Path(source);name=p.name.replace('S5_v4_','S6_');target=s/name
        shutil.copy2(p,target)
        provenance.append(dict(figure_number='S6',kind='source_table',original_path=str(p),delivery_path=str(target),original_sha256=sha(p),delivery_sha256=sha(target),change='byte_identical_inherited_source'))
    for ext in ['pdf','svg','png','preview.png','grayscale.png']:
        p=v5/'antigen/figures'/('Supplementary_Figure_S5.'+ext);target=f/('Supplementary_Figure_S6.'+ext)
        # Frozen inherited image is copied only from its original documented source.
        shutil.copy2(p,target)
        provenance.append(dict(figure_number='S6',kind=ext,original_path=str(p),delivery_path=str(target),original_sha256=sha(p),delivery_sha256=sha(target),change='byte_identical; figure_number_in_caption_and_registry_only'))
    inherited=dict(original)
    inherited['source_tables']=[x['delivery_path'] for x in provenance if x['figure_number']=='S6' and x['kind']=='source_table']
    inherited['files']={ext:dict(path=str(f/('Supplementary_Figure_S6.'+ext)),sha256=sha(f/('Supplementary_Figure_S6.'+ext)),bytes=(f/('Supplementary_Figure_S6.'+ext)).stat().st_size) for ext in ['pdf','svg','png']}
    inherited['inherited_export_path']=str(old_s6);inherited['inherited_export_sha256']=sha(old_s6)
    inherited['recorded_visual_verdict_path']=str(f/'Supplementary_Figure_S6.visual_iteration1.json')
    (f/'Supplementary_Figure_S6.export.json').write_text(json.dumps(inherited,indent=2)+'\n')
    old_s7=v5/'figures/Supplementary_Figure_S6.export.json'
    shutil.copy2(old_s7,f/'Supplementary_Figure_S7.source_export.json')
    for p in (v5/'figures/source').glob('S6_*.tsv'):
        target=s/p.name.replace('S6_','S7_',1)
        assert sha(p)==sha(target)
        provenance.append(dict(figure_number='S7',kind='source_table',original_path=str(p),delivery_path=str(target),original_sha256=sha(p),delivery_sha256=sha(target),change='byte_identical_source; Q2_concentration_line_style_changed_only'))
    for p in [v5/'code/figureS6_v4.py',v5/'code/antigen_figures_v4.py']:
        provenance.append(dict(figure_number='S7' if 'S6' in p.name else 'S6',kind='original_plot_entrypoint',original_path=str(p),delivery_path='',original_sha256=sha(p),delivery_sha256='',change='original_code_unchanged'))
    pd.DataFrame(provenance).to_csv(f/'source_provenance.tsv',sep='\t',index=False)
    names=['Figure2','Supplementary_Figure_S6','Supplementary_Figure_S7']
    numbers=['2','S6','S7'];iterations=[3,1,2]
    titles=['Source-stratified candidate sequence evidence','Branch-specific antigen-reference context','Historical nomination and current sensitivity audit']
    questions=['How do complete Q1/Q2 occurrence, exact identity and within-person recurrence differ across sources?','Which exact and similar antigen-reference records are compatible with the observed candidate V/J assignments?','What was reported historically and how do current carriage, batch and concentration summaries change by definition?']
    for stem,caption in zip(names,CAPTIONS):(f/(stem+'.caption.md')).write_text(caption+'\n')
    (f/'captions.md').write_text('\n\n'.join(CAPTIONS)+'\n')
    rows=[];files=[]
    for number,stem,it,title,question in zip(numbers,names,iterations,titles,questions):
        export=json.loads((f/(stem+'.export.json')).read_text())
        assert export['width_mm']==180 and export['height_mm']<=246 and export['font_points_min']>=7 and not export['out_of_canvas_text']
        verdict=json.loads((f/(stem+f'.visual_iteration{it}.json')).read_text());assert verdict['verdict']=='pass' and verdict['score']>=90
        binding=json.loads((f/(stem+f'.visual_binding_iteration{it}.json')).read_text())
        for path,digest in binding['reviewed_files'].items():assert sha(path)==digest
        row=dict(figure_number=number,title=title,question=question)
        for key,ext in [('pdf_path','pdf'),('svg_path','svg'),('png_path','png'),('preview_path','preview.png'),('grayscale_path','grayscale.png'),('export_path','export.json')]:
            p=f/(stem+'.'+ext);row[key]=str(p.relative_to(root))
            files.append(dict(figure_id=stem,figure_number=number,path=str(p),format=ext,bytes=p.stat().st_size,sha256=sha(p),source_tables=';'.join(export['source_tables']),visual_status='pass',visual_iteration=it,visual_score=verdict['score']))
        row.update(include_in_manuscript=True,section='main' if number=='2' else 'supplementary',source_tables=';'.join(export['source_tables']))
        rows.append(row)
    pd.DataFrame(rows).to_csv(f/'figure_manifest_rows.tsv',sep='\t',index=False)
    pd.DataFrame(files).to_csv(f/'figure_manifest.tsv',sep='\t',index=False)
    (out/'FIGURE_DELIVERY_NOTES.md').write_text("""Candidate scientific data are frozen in numeric_freeze.json. Figure 2 adds patient-only Wilson intervals and source/material units; Le2017 counts are not depth-normalized treatment dynamics. Supplementary Figure S6 is a byte-identical inherited antigen-reference graphic with renamed registry/caption only; copied source tables and their original paths/hashes are listed in figures/source_provenance.tsv. Supplementary Figure S7 uses unchanged inherited historical/current source tables; only the Q2 read-concentration line gains a dashed style. Both old v5 figure/code files remain unchanged. Recorded actual colour/grayscale verdicts and reviewed-PNG SHA bindings are required before these figures enter the unified registry. Visual PASS applies to these three standalone figures, not a whole-document render or biological/technical validation.

All computation and plotting were executed on codex-remote. Replay the cached candidate audit with code/v6_candidate_audit.py --root ROOT, main figure with code/v6_candidate_figures.py --root ROOT, and historical figure with code/v6_candidate_history_figure.py --root ROOT. The existing package preserves inherited source tables/assets; code/v6_candidate_finalize.py --root ROOT assembles numbering, captions and source provenance after actual visual review, and refuses mismatched reviewed PNGs. code/v6_candidate_verify.py --root ROOT performs cached numerical/hash checks without a new raw scan or metadata walk. Preserved producing audit code is candidates/provenance/v6_candidate_audit.producing.py; current audit entrypoint differs only in report wording. The bounded metadata inventory stopped at its filename cap; no primary experimental record was recovered and no exhaustive absence claim is made.
""")
    manifest=[]
    for p in sorted(out.rglob('*')):
        if p.is_file() and p.name not in ['delivery_manifest.tsv','delivery_receipt.json']:
            manifest.append(dict(path=str(p.relative_to(root)),bytes=p.stat().st_size,sha256=sha(p),scope='candidate_lane'))
    for p in sorted((root/'code').glob('v6_candidate_*.py')):manifest.append(dict(path=str(p.relative_to(root)),bytes=p.stat().st_size,sha256=sha(p),scope='candidate_lane_code'))
    pd.DataFrame(manifest).to_csv(out/'delivery_manifest.tsv',sep='\t',index=False)
    receipt=dict(status='passed',figures=['2','S6','S7'],actual_colour_and_grayscale_review=True,clinical_or_functional_validation=False,scientific_gate='unresolved',numeric_freeze_sha256=sha(out/'numeric_freeze.json'),delivery_manifest_sha256=sha(out/'delivery_manifest.tsv'),figure_manifest_rows_sha256=sha(f/'figure_manifest_rows.tsv'),verification_receipt_sha256=sha(out/'verification_receipt.json'),entrypoint_sha256=sha(__file__))
    (out/'delivery_receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps(receipt,indent=2))
if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('--root',type=Path,required=True);main(a.parse_args().root.resolve())
