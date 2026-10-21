#!/usr/bin/env python3
"""Read-only join of frozen Home GLIPH2 projection to frozen Leiden labels."""
from release_runtime import private_literal as _private_literal, release_path as _release_path
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[k]='8'
from pathlib import Path
import json, hashlib, time, resource, re
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

ROOT=Path(_release_path('path_99d8c1c6fde2a6b9'))
OUT=ROOT/'antigen_selected_20261003/tools'
Q1=_private_literal('private_45bbaa37a9607bb7')
TARGETS=[('tcrbert',11,'leiden_full_20261001'),('tcrbert',7,'leiden_full_20261001'),('sceptr',5,'leiden_full_sceptr_20261002')]
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(8<<20),b''):h.update(b)
    return h.hexdigest()
def stems(s,prefix):return set(re.findall(prefix+r'[0-9]+(?:-[0-9]+)?',str(s).upper()))
def compatibility(observed,reference,prefix):
    a=stems(observed,prefix);b=stems(reference,prefix)
    return 'missing' if not a or not b else 'compatible' if a&b else 'conflict'
def write(df,name):df.to_csv(OUT/'results'/name,sep='\t',index=False)
def markdown(df):
    return '\n'.join(['| '+' | '.join(map(str,df.columns))+' |','|'+'|'.join(['---']*len(df.columns))+'|']+
        ['| '+' | '.join(map(str,row))+' |' for row in df.itertuples(index=False,name=None)])
def main():
    start=time.perf_counter()
    for d in ['src','config','results','audit','report']:(OUT/d).mkdir(parents=True,exist_ok=True)
    receipt=OUT/'audit/gliph_completion.json'
    assert not receipt.exists(), 'Already complete; retain frozen output'
    inputs={
        'membership':ROOT/'groups/full_home/gliph2_group_membership.parquet',
        'metadata':ROOT/'groups/full_home/group_metadata.tsv',
        'projection_receipt':ROOT/'groups/full_home/gliph2_projection_receipt.json',
        'index':ROOT/'data/atlas_aa_index.parquet',
        'reference':ROOT/'exploratory_extension_20261001/antigen/normalized_reference_records.tsv',
        'reference_parse':ROOT/'exploratory_extension_20261001/antigen/reference_parse_receipt.json',
        'tcrbert_labels':ROOT/'leiden_full_20261001/clusters/primary_labels.npy',
        'sceptr_labels':ROOT/'leiden_full_sceptr_20261002/clusters/primary_labels.npy',
    }
    input_sha={k:sha(v) for k,v in inputs.items()}
    projected=json.loads(inputs['projection_receipt'].read_text())
    assert projected['membership_sha256']==input_sha['membership']
    assert json.loads(inputs['reference_parse'].read_text())['rules']['human_TRB_only'] is True
    g=pd.read_parquet(inputs['membership'])
    assert g.group_id.nunique()==40 and len(g)==42769
    ids=np.sort(g.seq_idx.unique());expected=g[['seq_idx','aa']].drop_duplicates()
    assert expected.seq_idx.is_unique
    found=[]
    for batch in pq.ParquetFile(inputs['index']).iter_batches(batch_size=250000,columns=['seq_idx','aa'],use_threads=True):
        take=pc.is_in(batch.column(0),value_set=pa.array(ids))
        if pc.any(take).as_py():found.append(pa.Table.from_batches([batch]).filter(take).to_pandas())
    idx=pd.concat(found,ignore_index=True)
    check=expected.merge(idx,on='seq_idx',suffixes=('_projected','_atlas'),validate='one_to_one')
    assert len(check)==len(expected) and (check.aa_projected==check.aa_atlas).all()
    ref=pd.read_csv(inputs['reference'],sep='\t',dtype=str).fillna('')
    metadata=pd.read_csv(inputs['metadata'],sep='\t',dtype=str).fillna('')
    meta=metadata[metadata.group_id.isin(g.group_id.unique())].drop_duplicates('group_id')
    assert len(meta)==40
    joins=[];cover=[];summaries=[]
    for model,cid,branch in TARGETS:
        labels=np.load(ROOT/branch/'clusters/primary_labels.npy',mmap_mode='r')
        assert len(labels)==28380380
        selected=g[labels[g.seq_idx.to_numpy()]==cid].copy()
        selected['model']=model; selected['cluster_id']=cid
        joins.append(selected.merge(meta,on='group_id',how='left',validate='many_to_one'))
        n=int((labels==cid).sum()); nAA=selected.aa.nunique()
        for gid,x in selected.groupby('group_id'):
            m=meta[meta.group_id==gid].iloc[0]
            cover.append(dict(model=model,cluster_id=cid,group_id=gid,pattern=m.pattern,
                seed_type=m.type,seed_final_score=m.final_score,n_AA=x.aa.nunique(),n_AA_VJ=len(x),
                full_Leiden_cluster_AA=n,AA_fraction_of_full_cluster=x.aa.nunique()/n))
        summaries.append(dict(model=model,cluster_id=cid,full_Leiden_cluster_AA=n,
            n_frozen_motif_groups=selected.group_id.nunique(),n_projected_AA=nAA,
            n_projected_AA_VJ=len(selected),projected_AA_fraction=nAA/n))
    members=pd.concat(joins,ignore_index=True)
    write(members,'gliph_selected_membership.tsv')
    write(pd.DataFrame(cover).sort_values(['model','cluster_id','n_AA'],ascending=[True,True,False]),'gliph_selected_motif_coverage.tsv')
    write(pd.DataFrame(summaries),'gliph_selected_coverage_summary.tsv')
    joined=members.merge(ref,left_on='aa',right_on='ref_cdr3beta',how='inner')
    joined['v_status']=[compatibility(a,b,'TRBV') for a,b in zip(joined.v_gene,joined.ref_v)]
    joined['j_status']=[compatibility(a,b,'TRBJ') for a,b in zip(joined.j_gene,joined.ref_j)]
    joined['evidence_tier']=np.where((joined.v_status=='conflict')|(joined.j_status=='conflict'),
        'exact_AA_VJ_conflict',np.where((joined.v_status=='compatible')&(joined.j_status=='compatible'),
        'exact_AA_VJ_compatible','exact_AA_partial_VJ_missing'))
    joined.loc[joined.reference_scope!='specificity_annotation','evidence_tier']='pathology_only_not_antigen_specificity'
    write(joined,'gliph_selected_exact_reference_records.tsv')
    good=joined[(joined.reference_scope=='specificity_annotation')&(joined.v_status!='conflict')&(joined.j_status!='conflict')]
    rows=[]
    for (model,cid,gid,organism,antigen,epitope),x in good.groupby(['model','cluster_id','group_id','source_organism','antigen','epitope'],dropna=False):
        rows.append(dict(model=model,cluster_id=cid,group_id=gid,source_organism=organism,
            antigen=antigen,epitope=epitope,n_unique_AA=x.aa.nunique(),
            n_reference_keys=x.reference_key.nunique(),n_databases=x.database.nunique(),
            evidence_tiers=';'.join(sorted(x.evidence_tier.unique())),
            mhc=';'.join(sorted(set(x.mhc)-{''})),pmid=';'.join(sorted(set(x.pmid)-{''}))))
    write(pd.DataFrame(rows,columns=['model','cluster_id','group_id','source_organism','antigen','epitope','n_unique_AA','n_reference_keys','n_databases','evidence_tiers','mhc','pmid']),
        'gliph_selected_motif_antigen_clues.tsv')
    q=g[g.aa==Q1].merge(meta,on='group_id',how='left')
    write(q,'gliph_Q1_membership.tsv')
    outputs={str(x.relative_to(OUT)):sha(x) for x in (OUT/'results').iterdir() if x.is_file()}
    result=dict(status='PASS',scope='Frozen Home-only GLIPH2 seed-derived motif projection crossed to full frozen Leiden clusters; no GLIPH2 rerun; not full-cluster antigen labels',
        input_sha256=input_sha,script_sha256=sha(__file__),index_mapping_verified_AA=len(check),
        n_frozen_motifs=40,n_projection_memberships=42769,n_selected_memberships=len(members),
        n_exact_reference_rows=len(joined),n_compatible_specificity_rows=len(good),
        Q1_in_frozen_projection=len(q)>0,elapsed_seconds=time.perf_counter()-start,
        peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
        max_compute_threads=8,gpu_used=False,output_sha256=outputs,
        immutable_inputs_verified={k:sha(v)==input_sha[k] for k,v in inputs.items()})
    assert all(result['immutable_inputs_verified'].values())
    receipt.write_text(json.dumps(result,indent=2)+'\n')
    lines=['# GLIPH2 nested motif evidence','',result['scope'],'',
        'Only exact complete human CDR3β reference records with compatible or partially missing V/J are counted as antigen clues. Motif sharing is lower evidence and does not assign an antigen to the entire Leiden cluster.',
        '', 'Frozen Home projection has 40 qualifying literal motifs; its limited coverage is reported against all AA in each Leiden cluster.', '',markdown(pd.DataFrame(summaries)),'',
        f'Q1 occurs in frozen Home motif projection: {len(q)>0}. Absence only means not assigned by these 40 frozen rules.','',
        _private_literal('private_4a4471ea66b00c5b')]
    (OUT/'report/GLIPH_SELECTED.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(result,indent=2),flush=True)
if __name__=='__main__':main()
