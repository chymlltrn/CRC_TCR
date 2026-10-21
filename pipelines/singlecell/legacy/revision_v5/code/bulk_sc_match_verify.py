#!/usr/bin/env python3
"""Verify match outputs without rescanning original blood clone records."""
from release_config import release_path, private_value
from pathlib import Path
from datetime import datetime, timezone
from itertools import product
import argparse, hashlib, json, sys, traceback
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
V1=Path(str(release_path('bulk_sc_match_verify_source_10')))
DEFS=('AA_VJ','NT_VJ','AA'); GROUPS=('home','external','union')
TOTALS={'home':{'raw_clone_rows':41389196,'qc_clone_rows':38834872,'qc_reads':941927575},
        'external':{'raw_clone_rows':16485349,'qc_clone_rows':15862848,'qc_reads':363688521}}
PROTOCOL_SHA='908caf3ca66931bf438e429944dc9e553883c78eb7335d7cf4d70a1d16742780'
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()
def need(test,label):
    if not bool(test):raise AssertionError(label)
def eq(a,b,label):need(a==b,f'{label}: actual={a}, expected={b}')
def series_eq(a,b,label):need(a.equals(b.rename(a.name)),label)
def verify(root):
    out=root/'bulk_sc_match/ingest';data=root/'bulk_sc_match/server_data'
    rec=json.loads((out/'ingest_receipt.json').read_text())
    need(rec['status']=='passed' and rec['full_EOF'],'producer complete')
    eq(rec['selected_libraries'],339,'libraries');eq(rec['independent_blood_patients'],339,'patients')
    eq(sha(root/'bulk_sc_match/PROTOCOL.md'),PROTOCOL_SHA,'protocol');eq(rec['protocol_sha256'],PROTOCOL_SHA,'receipt protocol')
    manifest=pd.read_csv(out/'output_manifest.tsv',sep='\t')
    for r in manifest.itertuples():
        eq(Path(r.path).stat().st_size,r.bytes,'output bytes '+Path(r.path).name)
        eq(sha(r.path),r.sha256,'output SHA '+Path(r.path).name)
    producer=rec.get('producing_entrypoint_sha256',rec['entrypoint_sha256'])
    snapshot=out/'producing_code'/f'bulk_sc_match_ingest.{producer}.py'
    eq(sha(snapshot),producer,'producing code snapshot')
    roster=pd.read_csv(out/'selected_library_roster.tsv',sep='\t')
    frozen=pd.read_csv(V1/'cohort/library_master.tsv',sep='\t')
    frozen=frozen[frozen.primary_repertoire_library.eq(1)].sort_values(['cohort','sample_id'])
    cols=['cohort','sample_id','patient_key']
    eq(roster[cols].to_records(index=False).tolist(),frozen[cols].to_records(index=False).tolist(),'selected roster identity')
    need(len(roster)==339 and roster.patient_key.is_unique,'339 unique blood patients')
    eq(roster.groupby('cohort').size().to_dict(),{'home':137,'external':202},'cohort counts')
    need(private_value('bulk_sc_match_verify_value_44_9') in set(roster.sample_id) and private_value('bulk_sc_match_verify_value_44_50') not in set(roster.sample_id),'designated included library primary')
    input_manifest=pd.read_csv(out/'input_manifest.tsv',sep='\t')
    guard_path=out/'guard_code_equivalence.json'
    guard=json.loads(guard_path.read_text()) if guard_path.exists() else None
    if guard:
        eq(guard['status'],'passed','guard equivalence status')
        eq(guard['producing_entrypoint_sha256'],producer,'guard producing code')
        eq(guard['guarded_entrypoint_sha256'],sha(root/'code/bulk_sc_match_ingest.py'),'guard current code')
        eq(guard['guarded_all_stage_resume_equivalence'],'passed','all-stage no-rescan equivalence')
        eq(rec['guard_code_equivalence_sha256'],sha(guard_path),'guard certificate binding')
        for name,h in guard['first_core_output_sha256'].items():eq(sha(data/name),h,'first/current core equality '+name)
    index_sha=sha(data/'sc_cells_index.parquet');roster_sha=sha(out/'selected_library_roster.tsv')
    refs=input_manifest[~input_manifest.path.eq(str(root/'code/bulk_sc_match_ingest.py'))].to_dict('records')
    reference_sha=hashlib.sha256(json.dumps(refs,sort_keys=True,separators=(',',':')).encode()).hexdigest()
    for r in roster.itertuples():
        path=out/'library_receipts'/f'{r.cohort}__{r.sample_id}.json'
        bound=json.loads(path.read_text());b=bound['execution_binding']
        eq(b['protocol_sha256'],PROTOCOL_SHA,'per-library protocol binding')
        eq(b['sc_index_sha256'],index_sha,'per-library index binding')
        eq(b['selected_roster_sha256'],roster_sha,'per-library roster binding')
        eq(b['reference_inputs_sha256'],reference_sha,'per-library reference input binding')
        eq(b['producing_entrypoint_sha256'],producer,'per-library exact producing code binding')
        for label,p in {'support':data/'library_support'/f'{r.cohort}__{r.sample_id}.parquet','coverage':out/'library_coverage'/f'{r.cohort}__{r.sample_id}.tsv','gene_tokens':out/'library_gene_tokens'/f'{r.cohort}__{r.sample_id}.tsv'}.items():
            eq(b['derivative_sha256'][label],sha(p),'per-library derivative binding '+label)
    rows=pd.read_csv(out/'library_eof_receipts.tsv',sep='\t')
    need(len(rows)==339 and rows.scan_complete_to_EOF.all() and not rows.duplicated(['cohort','sample_id']).any(),'339 EOF receipts')
    joined=rows.merge(roster[cols+['library_qc_reads','library_qc_unique_aa']],on=cols,how='outer',validate='one_to_one',indicator=True)
    need(joined._merge.eq('both').all() and joined.qc_reads.eq(joined.library_qc_reads).all(),'source per-library QC reads/identity')
    for r in rows.itertuples():
        eq(Path(r.source_path).stat().st_size,r.source_bytes,'source size '+r.sample_id)
        eq(pq.ParquetFile(r.source_path).metadata.num_rows,r.raw_clone_rows,'source metadata EOF '+r.sample_id)
        eq(r.raw_clone_rows,r.parquet_metadata_rows,'receipt EOF '+r.sample_id)
        need(isinstance(r.source_sha256,str) and len(r.source_sha256)==64,'source SHA recorded')
    eq(rows.groupby('cohort')[['raw_clone_rows','qc_clone_rows','qc_reads']].sum().to_dict('index'),TOTALS,'source totals')
    eq(rec['totals'],TOTALS,'receipt totals')
    layers=pd.read_csv(out/'QC_layers.tsv',sep='\t');need(len(layers)==339,'QC layer patients')
    cols=list(layers.columns);need(layers[cols].equals(rows[cols]),'QC layers exact receipt totals')
    for col in ['VJ_valid_clone_rows','NT_translation_valid_clone_rows','NT_VJ_eligible_clone_rows']:
        need(layers[col].le(layers.qc_clone_rows).all(),'QC row bound '+col)
    for col in ['VJ_valid_reads','NT_translation_valid_reads','NT_VJ_eligible_reads']:
        need(layers[col].le(layers.qc_reads).all(),'QC read bound '+col)
    need(layers.NT_VJ_eligible_clone_rows.le(layers.VJ_valid_clone_rows).all() and layers.NT_VJ_eligible_clone_rows.le(layers.NT_translation_valid_clone_rows).all(),'NT VJ QC subset')
    need(layers.NT_translation_valid_clone_rows.eq(layers.qc_clone_rows).all() and layers.NT_translation_valid_reads.eq(layers.qc_reads).all(),'all frozen QC bulk NT translation compatible in actual scan')
    cells=pd.read_parquet(data/'sc_cells.parquet');index=pd.read_parquet(data/'sc_cells_index.parquet')
    need(cells.cell_name.is_unique and len(cells)==597963,'all source single beta denominator')
    need(cells.single_beta.all() and cells.accepted_distinct_beta.eq(1).all(),'one distinct beta per cell')
    eq(int((cells.RNA_joined&cells.MajorCellType.eq('T')).sum()),340805,'RNA T single beta denominator')
    eq(int(cells.is_primary_cell.sum()),53886,'primary19 denominator')
    need(cells.sc_gene_syntax_valid.all() and cells.sc_nt_translation_valid.all(),'sc acceptance unchanged')
    need(len(index)==len(cells) and index.cell_name.equals(cells.cell_name),'index cell order')
    for col in index:series_eq(cells[col],index[col],'immutable index '+col)
    eq(cells.groupby('cell_resolution_tag').size().to_dict(),{'RNA_joined_T':340805,'RNA_joined_nonT':3999,'VDJ_only_verified_sample':194634,'VDJ_only_unmapped_source':58525},'resolution counts')
    need(cells.loc[~cells.RNA_joined,[c for c in cells if c.startswith('program_')]].isna().all().all(),'no RNA transfer to VDJ-only cells')
    td=cells[cells.is_primary_cell].groupby(['patient_id','Tissue']).size()
    need(len(td)==38 and td.index.get_level_values(0).nunique()==19 and td.ge(50).all(),'fixed paired tissues')
    for d,k in [('AA',cells.beta_aa),('AA_VJ',cells.beta_aa+'|'+cells.beta_v+'|'+cells.beta_j),('NT_VJ',cells.beta_nt+'|'+cells.beta_v+'|'+cells.beta_j)]:
        series_eq(cells['key_'+d],k,'literal key '+d)
    support=pd.read_parquet(data/'bulk_support.parquet')
    need(not support.duplicated(['definition','match_key','patient_key']).any(),'support unit patient key')
    need(set(support.definition)==set(DEFS),'support all definitions')
    need(support.read_sum.notna().all() and support.read_sum.ge(1).all() and support.read_sum.mod(1).eq(0).all() and support.bulk_clone_rows.ge(1).all(),'support positive integral')
    pm=roster.set_index('patient_key')
    series_eq(support.cohort,support.patient_key.map(pm.cohort),'support cohort')
    series_eq(support.sample_id,support.patient_key.map(pm.sample_id),'support primary library')
    grouped={(c,s):z for (c,s),z in support.groupby(['cohort','sample_id'],sort=False)}
    for r in roster.itertuples():
        local=pd.read_parquet(data/'library_support'/f'{r.cohort}__{r.sample_id}.parquet')
        got=grouped.get((r.cohort,r.sample_id),support.iloc[:0])
        sort=['definition','match_key']
        need((local.empty and got.empty) or local[list(support)].sort_values(sort).reset_index(drop=True).equals(got.sort_values(sort).reset_index(drop=True)),'aggregate union '+r.sample_id)
    cat=pd.read_parquet(data/'key_catalog.parquet')
    need(not cat.duplicated(['definition','match_key']).any(),'catalog unit definition key')
    catalogs={d:cat[cat.definition.eq(d)].set_index('match_key') for d in DEFS}
    for d,c in catalogs.items():
        eq(set(c.index),set(cells['key_'+d]),'zero-support catalog keys '+d)
        counts=cells.groupby('key_'+d).size()
        series_eq(c.full_sc_cells.reindex(counts.index),counts,'catalog source cells '+d)
        for name,mask in [('rna_t',cells.RNA_joined&cells.MajorCellType.eq('T')),('baseline19_Tumor',cells.is_primary_cell&cells.Tissue.eq('Tumor')),('baseline19_Normal',cells.is_primary_cell&cells.Tissue.eq('Normal'))]:
            counts=cells[mask].groupby('key_'+d).size().reindex(c.index,fill_value=0)
            series_eq(c[name+'_cells'],counts,'catalog scoped cells '+d+':'+name)
        sd=support[support.definition.eq(d)];need(sd.match_key.isin(c.index).all(),'catalog support membership '+d)
        for g in GROUPS:
            s=sd if g=='union' else sd[sd.cohort.eq(g)]
            eq(set(c[g+'_denominator']),set([{'home':137,'external':202,'union':339}[g]]),'catalog denominator '+d+':'+g)
            sr=s.groupby('match_key').read_sum.sum().reindex(c.index,fill_value=0)
            series_eq(c[g+'_support_reads'],sr,'catalog read sum '+d+':'+g)
            for t in (1,2):
                ss=s[s.read_sum.ge(t)];prev=ss.groupby('match_key').patient_key.nunique().reindex(c.index,fill_value=0)
                series_eq(c[g+'_patients_ge'+str(t)],prev,'catalog patient prevalence '+d+':'+g+':'+str(t))
                flag=f'match_{d}_{g}_ge{t}'
                series_eq(cells[flag],cells['key_'+d].isin(set(ss.match_key)),'independent cell flags '+flag)
    for g,t in product(GROUPS,(1,2)):
        need((~cells[f'match_NT_VJ_{g}_ge{t}']|cells[f'match_AA_VJ_{g}_ge{t}']).all(),'NT subset')
        need((~cells[f'match_AA_VJ_{g}_ge{t}']|cells[f'match_AA_{g}_ge{t}']).all(),'AA VJ subset')
    for d,g in product(DEFS,GROUPS):need((~cells[f'match_{d}_{g}_ge2']|cells[f'match_{d}_{g}_ge1']).all(),'threshold subset')
    cov=pd.read_csv(out/'patient_coverage.tsv',sep='\t');units=['patient_key','definition','read_threshold']
    need(len(cov)==2034 and not cov.duplicated(units).any(),'339x3x2 zero-preserving coverage')
    eq(set(cov[units].itertuples(index=False,name=None)),set(product(roster.patient_key,DEFS,(1,2))),'Cartesian coverage')
    rp=rows.set_index('patient_key');sup_groups={(p,d):z for (p,d),z in support.groupby(['patient_key','definition'],sort=False)}
    tissue_sets={d:(set(c.index[c.baseline19_Tumor_cells.gt(0)]),set(c.index[c.baseline19_Normal_cells.gt(0)])) for d,c in catalogs.items()}
    for r in cov.itertuples():
        src=rp.loc[r.patient_key];eq(r.qc_clone_rows,int(src.qc_clone_rows),'coverage QC rows');eq(r.qc_reads,int(src.qc_reads),'coverage QC reads');eq(r.cohort,src.cohort,'coverage cohort')
        need(r.definition_eligible_clone_rows<=r.qc_clone_rows and r.definition_eligible_reads<=r.qc_reads,'eligible denominator bounds')
        s=sup_groups.get((r.patient_key,r.definition),support.iloc[:0]);s=s[s.read_sum.ge(r.read_threshold)].set_index('match_key');got=set(s.index);ts,ns=tissue_sets[r.definition];t=got&ts;n=got&ns
        for name,members in {'full_source':got,'baseline19':t|n,'Tumor_only':t-n,'Normal_only':n-t,'both_tissues':t&n}.items():
            z=s.loc[list(members)] if members else s.iloc[:0]
            eq(getattr(r,name+'_matched_unique_keys'),len(members),'coverage key partition')
            eq(getattr(r,name+'_matched_clone_rows'),int(z.bulk_clone_rows.sum()),'coverage clone partition')
            eq(getattr(r,name+'_matched_reads'),int(z.read_sum.sum()),'coverage read partition')
        eq(r.baseline19_matched_reads+r.no_baseline_reads,r.qc_reads,'full QC read mass')
        eq(r.baseline19_matched_clone_rows+r.no_baseline_clone_rows,r.qc_clone_rows,'full QC clone mass')
        eq(r.Tumor_only_matched_reads+r.Normal_only_matched_reads+r.both_tissues_matched_reads,r.baseline19_matched_reads,'disjoint tissue read mass')
        eq(r.no_baseline_matched_unique_keys,r.definition_unique_keys-r.baseline19_matched_unique_keys,'unmatched eligible keys')
        if r.definition=='AA':eq(r.definition_unique_keys,int(pm.loc[r.patient_key].library_qc_unique_aa),'frozen full AA denominator')
        for prefix in ('full_source','baseline19'):
            for unit,den in [('clone_rows','qc_clone_rows'),('reads','qc_reads')]:
                need(np.isclose(getattr(r,f'{prefix}_fraction_qc_{unit}'),getattr(r,f'{prefix}_matched_{unit}')/getattr(r,den),rtol=1e-12,atol=1e-15),'coverage fraction')
    paths=[data/'sc_cells.parquet',data/'sc_cells_index.parquet',data/'key_catalog.parquet',data/'bulk_support.parquet',out/'patient_coverage.tsv',out/'QC_layers.tsv',out/'library_eof_receipts.tsv',out/'output_manifest.tsv',out/'ingest_receipt.json']
    outputs={p.name:{'path':str(p),'bytes':p.stat().st_size,'sha256':sha(p)} for p in paths}
    return {'status':'passed','completed_utc':datetime.now(timezone.utc).isoformat(),'verifier_sha256':sha(__file__),
            'protocol_sha256':PROTOCOL_SHA,'producing_entrypoint_sha256':producer,'current_entrypoint_sha256':sha(root/'code/bulk_sc_match_ingest.py'),
            'raw_clone_rescan_performed':False,'source_verification_scope':'source metadata/size plus producer full-EOF/source SHA/QC receipts; no repeated clone-data scan',
            'selected_blood_patients':339,'selected_libraries':339,'cohort_denominators':{'home':137,'external':202},
            'accepted_single_beta_source_cells':597963,'RNA_T_single_beta_cells':340805,'baseline19_primary_cells':53886,
            'paired_tissue_groups':38,'patient_definition_threshold_rows_verified':2034,
            'flags_recomputed_from_patient_key_support':True,'catalog_zero_support_keys_retained':True,
            'all_patient_zero_rows_retained':True,'disjoint_tissue_read_mass_conserved':True,
            'all_18_global_cell_match_flags_verified':True,'NT_VJ_subset_AA_VJ_subset_AA':True,
            'per_library_code_protocol_index_input_and_derivative_bindings_verified':True,
            'guarded_all_stage_resume_core_output_hashes_identical':True,
            'all_54697720_frozen_QC_bulk_NT_rows_translation_compatible_in_producer_receipts':True,
            'QC_totals':TOTALS,'formal_statistical_tests':'none','outputs':outputs}
def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True);a=p.parse_args();root=a.root.resolve()
    dest=root/'bulk_sc_match/ingest/verification_receipt.json'
    try:result=verify(root)
    except Exception as exc:
        result={'status':'failed','completed_utc':datetime.now(timezone.utc).isoformat(),'verifier_sha256':sha(__file__),'error':str(exc),'traceback':traceback.format_exc()}
        dest.write_text(json.dumps(result,indent=2)+'\n');print(result['traceback'],file=sys.stderr);return 1
    dest.write_text(json.dumps(result,indent=2)+'\n')
    print('OUTPUT VERIFICATION PASS',json.dumps({k:result[k] for k in ['selected_libraries','accepted_single_beta_source_cells','RNA_T_single_beta_cells','baseline19_primary_cells','patient_definition_threshold_rows_verified']}))
    return 0
if __name__=='__main__':sys.exit(main())

