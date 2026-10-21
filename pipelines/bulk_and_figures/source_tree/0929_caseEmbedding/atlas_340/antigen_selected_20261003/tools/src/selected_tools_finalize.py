#!/usr/bin/env python3
from release_runtime import private_literal as _private_literal, release_path as _release_path
from pathlib import Path
import json,hashlib
import pandas as pd
import pyarrow.dataset as ds
import pyarrow.parquet as pq
P=Path(_release_path('path_d0e765bda4af91e5'))
T=P/'tools'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def md(df):
    return '\n'.join(['| '+' | '.join(map(str,df.columns))+' |','|'+'|'.join(['---']*len(df.columns))+'|']+
        ['| '+' | '.join(map(str,row))+' |' for row in df.itertuples(index=False,name=None)])
def main():
    assert not (T/'audit/selected_tools_acceptance.json').exists()
    g=json.loads((T/'audit/gliph_completion.json').read_text())
    r=json.loads((T/'audit/tcrdist_completion.json').read_text())
    repair=json.loads((T/'audit/gliph_report_repair.json').read_text())
    assert g['status']==r['status']==repair['status']=='PASS'
    for receipt in [g,r]:
        for file,h in receipt['output_sha256'].items():assert sha(T/file)==h,file
    assert sha(T/'src/selected_tcrdist_evidence.py')==r['script_sha256']
    assert sha(T/'src/selected_gliph_evidence.py')==repair['current_script_sha256']
    assert sha(T/'report/GLIPH_SELECTED.md')==repair['report_sha256']
    parts=list((T/'results/tcrdist_all_radius24_parts').glob('*.parquet'))
    full=ds.dataset(parts).to_table().to_pandas()
    assert len(full)==r['n_radius24_records']==6807
    assert (full.distance_beta<=24).all()
    assert (full.distance_beta==full.distance_CDR3_weighted+full.distance_V_CDR1_CDR2_CDR2p5).all()
    assert (full.reference_scope=='specificity_annotation').all()
    reps=pd.read_csv(P/'results/representatives.tsv',sep='\t').fillna('')
    assignment=reps[['model','cluster_id','aa']].drop_duplicates()
    allhit=assignment.merge(full,left_on='aa',right_on='query_aa',how='inner')
    rows=[]
    for (model,cid,organism,antigen,epitope,mhc),z in allhit.groupby(
        ['model','cluster_id','source_organism','antigen','epitope','mhc'],dropna=False):
        sim=z[z.complete_AA_exact==False]
        rows.append(dict(model=model,cluster_id=cid,source_organism=organism,antigen=antigen,epitope=epitope,mhc=mhc,
            n_representative_AA_with_any_radius24=z.aa.nunique(),
            n_representative_AA_with_nonexact_radius24=sim.aa.nunique(),
            n_reference_keys=z.reference_key.nunique(),minimum_distance=int(z.distance_beta.min()),
            pmids=';'.join(sorted(set(z.pmid)-{''})),
            scope='representative queries only; reference database coverage/duplication biased; not cluster antigen predominance'))
    ep=pd.DataFrame(rows).sort_values(['model','cluster_id','n_representative_AA_with_nonexact_radius24','minimum_distance'],ascending=[True,True,False,True])
    ep.to_csv(T/'results/tcrdist_representative_antigen_clue_summary.tsv',sep='\t',index=False)
    coverage=[]
    for (model,cid),x in assignment.groupby(['model','cluster_id']):
        hit=allhit[(allhit.model==model)&(allhit.cluster_id==cid)]
        sim=hit[hit.complete_AA_exact==False]
        coverage.append(dict(model=model,cluster_id=cid,n_representative_unique_AA=len(x),
            n_representative_AA_with_any_radius24=hit.aa.nunique(),
            n_representative_AA_with_nonexact_radius24=sim.aa.nunique(),
            n_distinct_ref_epitopes=hit.epitope.nunique(),
            n_distinct_ref_organisms=hit.source_organism.nunique()))
    pd.DataFrame(coverage).to_csv(T/'results/tcrdist_representative_coverage.tsv',sep='\t',index=False)
    q=pd.read_csv(T/'results/tcrdist_Q1_all_radius24_reference_records.tsv',sep='\t').fillna('')
    assert len(q)==2 and q.query_v_gene.eq('TRBV7-2').all()
    assert not q.complete_AA_exact.any()
    qsummary=pd.read_csv(T/'results/tcrdist_query_variant_summary.tsv',sep='\t')
    qsummary=qsummary[qsummary.query_aa==_private_literal('private_45bbaa37a9607bb7')]
    h=pd.read_csv(T/'results/representative_Q1_hamming1_reference_records.tsv',sep='\t').fillna('')
    hq=h[h.query_aa==_private_literal('private_45bbaa37a9607bb7')]
    qc=q[['query_v_gene','query_j_gene','ref_cdr3beta','distance_beta','source_organism','antigen','epitope','mhc','pmid']].copy()
    lines=['# Frozen-cluster sequence-tool follow-up','',
        'GLIPH2 nested frozen Home motifs and representative-only full beta TCRdist supply descriptive evidence. They do not establish binding of an entire Leiden cluster.', '',
        '## GLIPH2','',md(pd.read_csv(T/'results/gliph_selected_coverage_summary.tsv',sep='\t')),'',
        'No exact full CDR3β antigen-reference records with compatible/partially missing V/J were found in these projected motif members. Q1 was not assigned by the 40 frozen Home motif rules. This is limited-rule coverage, not a de novo GLIPH2 run.', '',
        '## TCRdist representatives','',md(pd.DataFrame(coverage)),'',
        'Full metric is 3×CDR3 + V CDR1/CDR2/CDR2.5, radius24. Observed stem-only V uses the minimum across available reference V alleles; this is an optimistic similarity, not an observed allele. Missing/unrecognized/multiple-stem observed V remains unscored. Reference shorthand multi-V alternatives are expanded and retained. J compatibility is shown but J is outside this metric.','',
        'The displayed exact-evidence representatives were intentionally selected from reference-supported AA; their hit rate cannot estimate full-cluster antigen prevalence. Counted reference records are human TRB specificity annotations; McPAS pathology-only records were excluded.','',
        '## Q1','',md(qsummary),'',md(qc),'',
        f'Same-length one-substitution reference hits: {len(hq)}. The supplied nucleotide clone has observed TRBV5-5/TRBJ1-4 according to the independent Q1 audit; no reference lies within radius24 for that V variant (minimum45). Only another observed AA-identical TRBV7-2 variant has two radius24 neighbors, with distinct CMV and SARS-CoV antigen contexts. These are ambiguous low-evidence similarities, not Q1 binding assignments.','',
        _private_literal('private_f1b0270db4fff190')]
    (T/'report/SELECTED_TOOL_EVIDENCE.md').write_text('\n'.join(lines)+'\n')
    complete=dict(status='PASS',scope='Descriptive frozen motif crosswalk and 60 representative selection rows plus Q1; no whole-cluster antigen binding claim',
        GLIPH2_input_result_SHA_verified=True,GLIPH2_report_repaired_without_recomputing=True,
        tcrdist_input_result_SHA_verified=True,tcrdist_distance_decomposition_verified_rows=len(full),
        public_API_checks=len(r['public_API_reproduction_checks']),
        multi_V_parser_regression_checks=['TRBV6-1/6-5/6-6 yields three stems','TRBV7-9 conflicts','TRBV6-5 compatible'],
        Q1_observed_variants=qsummary.to_dict('records'),Q1_radius24_records=2,Q1_hamming1_records=len(hq),
        CPU_threads=8,GPU_used=False,resource_measurement_note='tcrdist peak RSS is main-process RUSAGE_SELF, not sum of short-lived worker RSS',
        tcrdist_peak_main_process_RSS=r['peak_rss_bytes'],tcrdist_elapsed_seconds=r['elapsed_seconds'],
        output_sha256={str(x.relative_to(T)):sha(x) for x in T.rglob('*')
            if x.is_file() and x.suffix!='.log' and not x.name.startswith('selected_tools_acceptance')})
    (T/'audit/selected_tools_acceptance.json').write_text(json.dumps(complete,indent=2)+'\n')
    print(json.dumps({k:v for k,v in complete.items() if k!='output_sha256'},indent=2))
if __name__=='__main__':main()
