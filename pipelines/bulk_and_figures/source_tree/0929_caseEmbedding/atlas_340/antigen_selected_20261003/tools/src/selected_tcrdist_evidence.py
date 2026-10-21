#!/usr/bin/env python3
"""Limited representative/Q1 full beta TCRdist against frozen human TRB refs."""
from release_runtime import private_literal as _private_literal, release_path as _release_path
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS','NUMBA_NUM_THREADS'):
    os.environ[k]='8'
from pathlib import Path
import json,hashlib,re,time,resource,importlib.metadata,inspect,warnings
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.parquet as pq
from tcrdist.repertoire import TCRrep
from tcrdist.rep_funcs import _pw
import tcrdist.repertoire as repertoire_module
import tcrdist.rep_funcs as rep_funcs_module
import pwseqdist.metrics as metrics_module
ROOT=Path(_release_path('path_99d8c1c6fde2a6b9'))
OUT=ROOT/'antigen_selected_20261003/tools'
Q1=_private_literal('private_45bbaa37a9607bb7')
RADIUS=24
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(8<<20),b''):h.update(b)
    return h.hexdigest()
def tokens(s,prefix):
    text=str(s).upper()
    found=set(re.findall(prefix+r'[0-9]+(?:-[0-9]+)?(?:\*[0-9A-Za-z]+)?',text))
    # Preserve abbreviated alternatives such as TRBV6-1/6-5/6-6.
    if prefix in text:
        for part in re.split(r'[/,;|+]',text):
            part=part.strip()
            if re.fullmatch(r'[0-9]+(?:-[0-9]+)?(?:\*[0-9A-Za-z]+)?',part):found.add(prefix+part)
    return sorted(found)
def stems(s,prefix):return {x.split('*')[0] for x in tokens(s,prefix)}
def compatibility(a,b,prefix):
    a,b=stems(a,prefix),stems(b,prefix)
    return 'missing' if not a or not b else 'compatible' if a&b else 'conflict'
def hamming(a,b):return sum(x!=y for x,y in zip(a,b)) if len(a)==len(b) else -1
def write(df,name):df.to_csv(OUT/'results'/name,sep='\t',index=False)
def main():
    start=time.perf_counter()
    receipt=OUT/'audit/tcrdist_completion.json'
    assert not receipt.exists(), 'Already completed; preserve result'
    inputs={
        'representatives':ROOT/'antigen_selected_20261003/results/representatives.tsv',
        'observed_nodes':ROOT/'groups/tcrdist_all_nodes.parquet',
        'references':ROOT/'exploratory_extension_20261001/antigen/normalized_reference_records.tsv',
        'reference_parse':ROOT/'exploratory_extension_20261001/antigen/reference_parse_receipt.json',
        'tool_design':OUT/'config/tool_design.json',
    }
    assert all(x.exists() for x in inputs.values())
    input_sha={k:sha(v) for k,v in inputs.items()}
    reps=pd.read_csv(inputs['representatives'],sep='\t').fillna('')
    assert len(reps)==60 and set(zip(reps.model,reps.cluster_id))=={('tcrbert',11),('tcrbert',7),('sceptr',5)}
    query_aa=sorted(set(reps.aa)|{Q1})
    nodes=ds.dataset(inputs['observed_nodes']).to_table(columns=['aa','v_gene','j_gene','cohort_mask','seq_idx'],
        filter=ds.field('aa').isin(query_aa)).to_pandas()
    assert nodes.seq_idx.min()>=0
    assert nodes[['aa','v_gene','j_gene']].drop_duplicates().shape[0]==len(nodes)
    refs=pd.read_csv(inputs['references'],sep='\t',dtype=str).fillna('')
    assert json.loads(inputs['reference_parse'].read_text())['rules']['human_TRB_only'] is True
    specific=refs[refs.reference_scope=='specificity_annotation'].copy().reset_index(drop=True)
    bootstrap=pd.DataFrame({'v_b_gene':['TRBV5-5*01'],'j_b_gene':['TRBJ1-4*01'],'cdr3_b_aa':[Q1],'count':[1]})
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        tr=TCRrep(cell_df=bootstrap,organism='human',chains=['beta'],compute_distances=False,deduplicate=False,cpus=8)
    assert tr.weights_b=={'cdr3_b_aa':3,'pmhc_b_aa':1,'cdr2_b_aa':1,'cdr1_b_aa':1}
    genes={k:g for k,g in tr.all_genes['human'].items() if g.chain=='B' and g.region=='V'}
    gene_names=sorted(genes)
    gi={x:i for i,x in enumerate(gene_names)}
    by_stem={}
    for x in gene_names:by_stem.setdefault(x.split('*')[0],[]).append(x)
    def mapping(v,query=False):
        ts=tokens(v,'TRBV')
        if not ts:return [],'missing_V'
        if query and len({x.split('*')[0] for x in ts})!=1:return [],'ambiguous_multiple_observed_V_stems'
        selected=[];rules=[]
        for t in ts:
            stem=t.split('*')[0]
            if '*' in t and t in genes:selected.append(t);rules.append('reference_exact_allele')
            elif stem in by_stem:
                # MiXCR *00 means unknown allele, and old node table records stems only.
                selected.extend(by_stem[stem]);rules.append('stem_all_available_alleles_minimum_not_observed_allele')
            else:rules.append('unrecognized_V')
        return sorted(set(selected)),';'.join(sorted(set(rules)))
    map_cache={str(v):mapping(v) for v in specific.ref_v.unique()}
    specific['V_options']=specific.ref_v.map(lambda v:';'.join(map_cache[str(v)][0]))
    specific['V_mapping_rule']=specific.ref_v.map(lambda v:map_cache[str(v)][1])
    scorable=specific[specific.V_options!=''].copy().reset_index(drop=True)
    unscorable=specific[specific.V_options==''].copy()
    write(unscorable.groupby(['database','V_mapping_rule']).size().rename('n_reference_records').reset_index(),
        'tcrdist_reference_unscorable_summary.tsv')
    variants=[]
    for row in nodes.itertuples(index=False):
        opts,rule=mapping(row.v_gene,query=True)
        variants.append(dict(aa=row.aa,v_gene=row.v_gene,j_gene=row.j_gene,cohort_mask=row.cohort_mask,
            seq_idx=row.seq_idx,V_options=';'.join(opts),V_mapping_rule=rule,
            V_allele_observed=False,scorable=bool(opts)))
    variants=pd.DataFrame(variants)
    write(variants,'tcrdist_observed_query_VJ_variants.tsv')
    assert Q1 in set(variants.aa)
    vmat=np.zeros((len(genes),len(genes)),dtype=np.int32)
    for field,cdrpos in [('cdr1_b_aa',0),('cdr2_b_aa',1),('pmhc_b_aa',2)]:
        seqs=np.array([genes[x].cdrs_aligned[cdrpos] for x in gene_names])
        vmat+=_pw(tr.metrics_b[field],seqs1=seqs,ncpus=8,**tr.kargs_b[field])
    assert (vmat==vmat.T).all() and np.diag(vmat).sum()==0
    ref_v_raw=list(scorable.ref_v.unique());rvindex={v:i for i,v in enumerate(ref_v_raw)}
    ref_v_group=scorable.ref_v.map(rvindex).to_numpy()
    refs_options=[map_cache[v][0] for v in ref_v_raw]
    unique_ref_aa=np.array(sorted(scorable.ref_cdr3beta.unique()))
    raidx={x:i for i,x in enumerate(unique_ref_aa)}
    ref_aa_index=scorable.ref_cdr3beta.map(raidx).to_numpy()
    pair_cache={}
    def vmin(qv):
        if qv not in pair_cache:
            qopts,_=mapping(qv,query=True);qidx=np.array([gi[x] for x in qopts],dtype=int)
            vals=[];chosen_q=[];chosen_r=[]
            for ropts in refs_options:
                ridx=np.array([gi[x] for x in ropts],dtype=int)
                d=vmat[np.ix_(qidx,ridx)];loc=np.unravel_index(np.argmin(d),d.shape)
                vals.append(int(d[loc]));chosen_q.append(qopts[loc[0]]);chosen_r.append(ropts[loc[1]])
            pair_cache[qv]=(np.array(vals),np.array(chosen_q),np.array(chosen_r))
        return pair_cache[qv]
    parts=OUT/'results/tcrdist_all_radius24_parts';parts.mkdir(exist_ok=True)
    assert not list(parts.iterdir())
    summary=[]; all_small=[]; q1hits=[]; nrecords=0; smoke=[]
    for qi,aa in enumerate(query_aa):
        d3=_pw(tr.metrics_b['cdr3_b_aa'],seqs1=np.array([aa]),seqs2=unique_ref_aa,ncpus=8,
               **tr.kargs_b['cdr3_b_aa'])[0].astype(np.int32)
        assert d3.shape==(len(unique_ref_aa),)
        rows=[]
        for var in variants[variants.aa==aa].itertuples(index=False):
            if not var.scorable:
                summary.append(dict(query_aa=aa,query_v_gene=var.v_gene,query_j_gene=var.j_gene,
                    scorable=False,n_radius24_records=0,n_radius24_ref_AA=0,minimum_distance=np.nan))
                continue
            dv,cq,cr=vmin(var.v_gene)
            total=3*d3[ref_aa_index]+dv[ref_v_group]
            selected=np.where(total<=RADIUS)[0]
            hit=scorable.iloc[selected].copy()
            if len(hit):
                hit.insert(0,'query_aa',aa);hit.insert(1,'query_v_gene',var.v_gene);hit.insert(2,'query_j_gene',var.j_gene)
                hit['query_cohort_mask']=var.cohort_mask;hit['query_seq_idx']=var.seq_idx
                hit['query_V_options']=var.V_options;hit['query_V_mapping_rule']=var.V_mapping_rule
                hit['minimum_over_alleles']=True
                hit['query_minimizing_allele']=cq[ref_v_group[selected]]
                hit['reference_minimizing_allele']=cr[ref_v_group[selected]]
                hit['distance_beta']=total[selected];hit['distance_CDR3_weighted']=3*d3[ref_aa_index[selected]]
                hit['distance_V_CDR1_CDR2_CDR2p5']=dv[ref_v_group[selected]]
                hit['complete_AA_exact']=hit.ref_cdr3beta==aa
                hit['same_length_hamming']=[hamming(aa,x) for x in hit.ref_cdr3beta]
                hit['V_stem_status']=[compatibility(var.v_gene,x,'TRBV') for x in hit.ref_v]
                hit['J_stem_status']=[compatibility(var.j_gene,x,'TRBJ') for x in hit.ref_j]
                hit['evidence_tier']=np.where(hit.complete_AA_exact,
                    np.where(hit.J_stem_status=='conflict','exact_AA_J_conflict_neighborhood_only','exact_AA_minV_compatible_or_missingJ'),
                    'tcrdist_radius24_similarity_clue_not_binding')
                rows.append(hit)
                if aa==Q1:q1hits.append(hit)
                # Nonduplicated nearest clue records for the small user table; full radius retained in parquet parts.
                small=hit.sort_values(['distance_beta','same_length_hamming','reference_key']).drop_duplicates(
                    ['ref_cdr3beta','epitope','mhc','antigen','source_organism']).head(5)
                all_small.append(small)
                if len(smoke)<8:
                    z=hit.iloc[0];smoke.append((aa,z.query_minimizing_allele,z.ref_cdr3beta,z.reference_minimizing_allele,int(z.distance_beta)))
            summary.append(dict(query_aa=aa,query_v_gene=var.v_gene,query_j_gene=var.j_gene,
                scorable=True,n_radius24_records=len(hit),n_radius24_ref_AA=hit.ref_cdr3beta.nunique(),
                minimum_distance=int(total.min())))
        if rows:
            data=pd.concat(rows,ignore_index=True)
            data.to_parquet(parts/f'query_{qi:03d}.parquet',index=False);nrecords+=len(data)
        print(json.dumps(dict(query_index=qi,query_count=len(query_aa),aa=aa,radius24_records=sum(len(x) for x in rows))),flush=True)
    summary=pd.DataFrame(summary)
    write(summary,'tcrdist_query_variant_summary.tsv')
    clues=pd.concat(all_small,ignore_index=True) if all_small else pd.DataFrame()
    write(clues,'tcrdist_representative_nearest_clues.tsv')
    qhits=pd.concat(q1hits,ignore_index=True) if q1hits else clues.iloc[:0].copy()
    write(qhits,'tcrdist_Q1_all_radius24_reference_records.tsv')
    mapping_table=reps[['model','cluster_id','representative_type','rank','aa']].drop_duplicates()
    write(mapping_table.merge(summary,left_on='aa',right_on='query_aa',how='left'),
        'tcrdist_representative_variant_summary.tsv')
    # Same-length 1-AA substitutions are a distinct, V-independent clue; never antigen binding.
    hrows=[]
    for aa in query_aa:
        a=specific[specific.ref_cdr3beta.str.len()==len(aa)].copy()
        a['same_length_hamming']=[hamming(aa,x) for x in a.ref_cdr3beta]
        a=a[a.same_length_hamming==1].copy();a.insert(0,'query_aa',aa)
        a['evidence_tier']='same_length_one_AA_substitution_V_independent_low_evidence'
        if len(a):hrows.append(a)
    write(pd.concat(hrows,ignore_index=True) if hrows else pd.DataFrame(columns=['query_aa','ref_cdr3beta','same_length_hamming','evidence_tier']),
        'representative_Q1_hamming1_reference_records.tsv')
    # Independent public TCRrep full rectangular API reproduces the optimized decomposition.
    api_checks=[]
    for aa,qv,ra,rv,expected in smoke:
        q=pd.DataFrame({'v_b_gene':[qv],'j_b_gene':['TRBJ1-4*01'],'cdr3_b_aa':[aa],'count':[1]})
        r=pd.DataFrame({'v_b_gene':[rv],'j_b_gene':['TRBJ1-4*01'],'cdr3_b_aa':[ra],'count':[1]})
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            qr=TCRrep(cell_df=q,organism='human',chains=['beta'],compute_distances=False,deduplicate=False,cpus=1)
            rr=TCRrep(cell_df=r,organism='human',chains=['beta'],compute_distances=False,deduplicate=False,cpus=1)
        qr.compute_rect_distances(df=qr.clone_df,df2=rr.clone_df,store=False)
        got=int(qr.rw_beta[0,0]);assert got==expected,(got,expected)
        api_checks.append(dict(query_aa=aa,query_allele=qv,reference_AA=ra,reference_allele=rv,expected=expected,public_API_distance=got))
    sources={
        'repertoire_source':Path(repertoire_module.__file__),
        'rep_funcs_source':Path(rep_funcs_module.__file__),
        'pwseqdist_metrics_source':Path(metrics_module.__file__),
        'V_reference_db':Path(repertoire_module.__file__).parent/'db/combo_xcr_2024-03-05.tsv',
    }
    qmin=summary[summary.query_aa==Q1]
    result=dict(status='PASS',scope='60 representative selection rows and Q1; observed AA/V/J variants; human complete beta specificity references; full beta radius24, not antigen binding prediction',
        input_sha256=input_sha,script_sha256=sha(__file__),tool_versions={x:importlib.metadata.version(x) for x in ['tcrdist3','pwseqdist','numpy','pandas','pyarrow']},
        source_sha256={k:sha(v) for k,v in sources.items()},tool_warnings=[str(x.message) for x in caught],
        n_representative_selection_rows=len(reps),n_query_unique_AA=len(query_aa),n_query_observed_AA_VJ=len(variants),
        n_query_scorable_AA_VJ=int(variants.scorable.sum()),n_specificity_reference_records=len(specific),
        n_scorable_specificity_reference_records=len(scorable),n_scorable_reference_AA=len(unique_ref_aa),
        n_radius24_records=nrecords,n_Q1_radius24_records=len(qhits),
        Q1_minimum_distance=int(qmin.minimum_distance.min()) if qmin.scorable.any() else None,
        max_compute_threads=8,gpu_used=False,peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
        elapsed_seconds=time.perf_counter()-start,public_API_reproduction_checks=api_checks,
        partial_missing_V_references_excluded_from_full_distance=True,allele_minima_are_optimistic_similarity=True,
        J_not_part_of_distance=True,unrecognized_observed_V_kept_unscored=True,
        output_sha256={str(x.relative_to(OUT)):sha(x) for x in (OUT/'results').rglob('*') if x.is_file()},
        immutable_inputs_verified={k:sha(v)==input_sha[k] for k,v in inputs.items()})
    assert all(result['immutable_inputs_verified'].values())
    assert result['peak_rss_bytes']<34359738368
    receipt.write_text(json.dumps(result,indent=2)+'\n')
    report=['# Representative and Q1 TCRdist evidence','',result['scope'],'',
        f"{len(query_aa)} query AA / {len(variants)} observed V/J variants; full beta radius fixed at 24. {len(qhits)} Q1 reference-record/observed-VJ matches; minimum full distance {result['Q1_minimum_distance']}.",'',
        'Metric: 3 × CDR3 distance + V-encoded CDR1 + CDR2 + CDR2.5. J concordance is reported but J is not in this distance. Unknown observed allele uses minimum across all available same-stem V alleles, an optimistic similarity, not an observed allele assignment. Missing or unknown V is retained as unscored.','',
        'A neighborhood can contain different epitopes or HLA restrictions. Exact complete AA evidence and distance similarity are separated. No paired α, patient HLA or functional evidence is available to establish binding.','',
        'All radius-24 records are retained as parquet parts; the nearest-clue TSV is a display subset. All Q1 radius-24 records are included in its TSV.','',
        'Methods/API: [tcrdist3 source](https://github.com/kmayerb/tcrdist3/blob/master/tcrdist/repertoire.py), [tcrdist3 paper](https://elifesciences.org/articles/68605).']
    (OUT/'report/TCRDIST_SELECTED.md').write_text('\n'.join(report)+'\n')
    print(json.dumps(result,indent=2),flush=True)
if __name__=='__main__':main()
