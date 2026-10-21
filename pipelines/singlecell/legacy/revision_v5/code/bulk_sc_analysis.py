#!/usr/bin/env python3
"""Frozen patient-level blood-to-tissue matching analysis. Server execution only."""
from release_config import release_path, private_value
import os
for _name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[_name]='4'
import argparse, hashlib, json, platform, time, warnings, tempfile
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.stats import wilcoxon

PROTOCOL_SHA='908caf3ca66931bf438e429944dc9e553883c78eb7335d7cf4d70a1d16742780'
SEED=20261003; BOOTSTRAPS=10000; PERMUTATIONS=1000
MIN_PATIENTS=8; MIN_CELLS=20; MIN_CLONES=5; MIN_PAIRED_CELLS=50
PROGRAMS=('cytotoxic','exhaustion','naive_memory','interferon','residency','proliferation')
IDENTITIES=('AA_VJ','NT_VJ','AA')
DEFAULT_ROOT=Path(str(release_path('bulk_sc_analysis_source_20')))
SOURCE_ROOT=Path(str(release_path('bulk_sc_analysis_source_21')))


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(8*1024*1024),b''):h.update(block)
    return h.hexdigest()


def dump(path,value):
    Path(path).write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')


def save(path,rows):
    frame=rows if isinstance(rows,pd.DataFrame) else pd.DataFrame(rows)
    frame.to_csv(path,sep='\t',index=False,na_rep='NA')
    return frame


def bool_column(frame,column):
    assert column in frame,('missing_boolean',column)
    nonmissing=frame[column].dropna()
    assert all(isinstance(x,(bool,np.bool_)) for x in nonmissing.unique()),('non_boolean',column)
    return frame[column].fillna(False).astype(bool)


def bootstrap_mean(values,rng):
    values=np.asarray(values,dtype=float)
    if len(values)==0:return (np.nan,np.nan)
    means=values[rng.integers(0,len(values),size=(BOOTSTRAPS,len(values)))].mean(axis=1)
    return tuple(np.quantile(means,[.025,.975]))


def summarize(values,patients,rng):
    values=np.asarray(values,dtype=float)
    assert len(values)==len(patients) and len(set(patients))==len(patients)
    assert np.isfinite(values).all()
    low,high=bootstrap_mean(values,rng)
    n=len(values);estimable=n>=MIN_PATIENTS
    if estimable:
        if np.all(values==0):p=1.;status='estimable_all_zero'
        else:
            with warnings.catch_warnings():
                warnings.simplefilter('ignore',UserWarning)
                p=float(wilcoxon(values,zero_method='wilcox',correction=False,method='approx',alternative='two-sided').pvalue)
            status='estimable'
    else:p=np.nan;status='unestimable_fewer_than_8_patients'
    return {'n':n,'mean_difference':float(values.mean()) if n else np.nan,
            'median_difference':float(np.median(values)) if n else np.nan,
            'CI_low':low,'CI_high':high,'CI_target':'mean_patient_difference',
            'CI_method':'patient_percentile_bootstrap','bootstrap_repetitions':BOOTSTRAPS,
            'bootstrap_seed':SEED,'P':p,'P_BH_placeholder':p if estimable else 1.,
            'status':status,'included_patients':';'.join(patients)}


def adjust_families(frame,family_sizes):
    frame=frame.copy();frame['q']=np.nan;frame['q_BH_placeholder']=np.nan;frame['family_size']=0
    for family,size in family_sizes.items():
        idx=frame.index[frame.family==family];assert len(idx)==size,(family,len(idx),size)
        p=frame.loc[idx,'P_BH_placeholder'].to_numpy(dtype=float)
        assert np.isfinite(p).all() and ((p>=0)&(p<=1)).all()
        order=np.argsort(p,kind='stable');adjusted=np.minimum.accumulate((p[order]*size/np.arange(1,size+1))[::-1])[::-1]
        q=np.empty(size);q[order]=np.minimum(1.,adjusted)
        frame.loc[idx,'q_BH_placeholder']=q;frame.loc[idx,'family_size']=size
        good=frame.loc[idx,'P'].notna().to_numpy()
        frame.loc[idx[good],'q']=q[good]
    assert (frame.family_size>0).all()
    return frame


def coverage(cells,patients,definition,cohort,threshold,strict=False):
    label=f'match_{definition}_{cohort}_ge{threshold}'
    selection=cells.loc[cells.strict_paired] if strict else cells
    rows=[];contrasts=[];analysis=f'{definition}_{cohort}_ge{threshold}_'+('strict_alpha_beta_capture' if strict else 'single_beta_capture')
    for patient in patients:
        pair=[]
        for tissue in ('Tumor','Normal'):
            part=selection[(selection.patient_id==patient)&(selection.Tissue==tissue)]
            clone_labels=part.groupby('beta_clone',observed=True)[label].agg(['min','max'])
            assert (clone_labels['min']==clone_labels['max']).all(),('inconsistent_clone_label',analysis,patient,tissue)
            n=len(part);k=len(clone_labels);matched_cells=int(part[label].sum());matched_clones=int(clone_labels['max'].sum())
            row={'analysis':analysis,'definition':definition,'catalog_cohort':cohort,'threshold':threshold,
                 'capture_restriction':'strict_alpha_beta' if strict else 'accepted_single_beta',
                 'patient_id':patient,'tissue':tissue,'n_cells':n,'n_unique_beta_clones':k,
                 'n_matched_cells':matched_cells,'n_matched_clones':matched_clones,
                 'matched_cell_fraction':matched_cells/n if n else np.nan,
                 'matched_clone_fraction':matched_clones/k if k else np.nan}
            pair.append(row)
        eligible=all(r['n_cells']>=MIN_PAIRED_CELLS for r in pair)
        for row in pair:row['eligible_pair']=eligible;row['exclusion']='' if eligible else 'fewer_than_50_accepted_cells_in_at_least_one_tissue';rows.append(row)
        for metric in ('matched_clone_fraction','matched_cell_fraction'):
            contrasts.append({k:pair[0][k] for k in ('analysis','definition','catalog_cohort','threshold','capture_restriction','patient_id')}|
                             {'metric':metric,'tumor':pair[0][metric],'normal':pair[1][metric],
                              'difference':pair[0][metric]-pair[1][metric],'eligible':eligible})
    return rows,contrasts


def phenotype_tables(cells,patients,states):
    """Group eligibility is fixed and zero states retained; cloned cells are not patients."""
    states_rows=[];program_rows=[];eligibility=[]
    primary_label='match_AA_VJ_union_ge2'
    for patient in patients:
        for tissue in ('Tumor','Normal'):
            tissue_cells=cells[(cells.patient_id==patient)&(cells.Tissue==tissue)]
            for context in ('states','programs_conventional_CD8'):
                part=tissue_cells if context=='states' else tissue_cells[tissue_cells.is_cd8 & tissue_cells.positive_expression_qc]
                a=part[part[primary_label]];b=part[~part[primary_label]]
                counts={'n_matched_cells':len(a),'n_unmatched_cells':len(b),
                        'n_matched_clones':a.beta_clone.nunique(),'n_unmatched_clones':b.beta_clone.nunique()}
                eligible=min(counts['n_matched_cells'],counts['n_unmatched_cells'])>=MIN_CELLS and min(counts['n_matched_clones'],counts['n_unmatched_clones'])>=MIN_CLONES
                reason='' if eligible else 'each_group_requires_20_cells_and_5_beta_NTVJ_clones'
                eligibility.append({'patient_id':patient,'tissue':tissue,'context':context,**counts,'eligible':eligible,'exclusion':reason})
                if context=='states':
                    for weighting in ('cell_weighted','clone_equal'):
                        for state in states:
                            scores=[]
                            for group in (a,b):
                                if len(group)==0:value=np.nan
                                elif weighting=='cell_weighted':value=float((group.SubCellType==state).mean())
                                else:value=float(group.assign(_indicator=(group.SubCellType==state).astype(float)).groupby('beta_clone',observed=True)['_indicator'].mean().mean())
                                scores.append(value)
                            states_rows.append({'patient_id':patient,'tissue':tissue,'state':state,'weighting':weighting,**counts,
                                                'matched_fraction':scores[0],'unmatched_fraction':scores[1],'difference':scores[0]-scores[1],
                                                'eligible':eligible,'exclusion':reason})
                else:
                    for weighting in ('cell_weighted','clone_equal'):
                        for program in PROGRAMS:
                            col='program_'+program;scores=[];finite=True
                            for group in (a,b):
                                if len(group)==0:value=np.nan
                                elif not np.isfinite(group[col].to_numpy(dtype=float)).all():finite=False;value=np.nan
                                elif weighting=='cell_weighted':value=float(group[col].mean())
                                else:value=float(group.groupby('beta_clone',observed=True)[col].mean().mean())
                                scores.append(value)
                            ok=eligible and finite
                            program_rows.append({'patient_id':patient,'tissue':tissue,'program':program,'weighting':weighting,**counts,
                                                 'matched_mean_score':scores[0],'unmatched_mean_score':scores[1],'difference':scores[0]-scores[1],
                                                 'eligible':ok,'exclusion':reason if finite else 'nonfinite_fixed_program_score'})
    return pd.DataFrame(states_rows),pd.DataFrame(program_rows),pd.DataFrame(eligibility)


def structural_background(cells,patients,rng):
    """One global AA/V/J label per stratum; sparse fixed tissue-weight maps."""
    keycol='key_AA_VJ';labelcol='match_AA_VJ_union_ge2'
    identity=cells.groupby(keycol,sort=True,observed=True).agg(beta_aa=('beta_aa','first'),beta_v=('beta_v','first'),beta_j=('beta_j','first'),
              matched=(labelcol,'first'),patient_prevalence=('patient_id','nunique'))
    for col in ('beta_aa','beta_v','beta_j',labelcol):assert cells.groupby(keycol,observed=True)[col].nunique().max()==1
    identity['aa_length']=identity.beta_aa.str.len();identity['prevalence_bin']=np.minimum(3,identity.patient_prevalence)
    keys=list(identity.index);key_index={k:i for i,k in enumerate(keys)}
    labels=identity.matched.to_numpy(dtype=np.int8);movable=np.zeros(len(keys),dtype=bool);groups=[]
    for index in identity.groupby(['beta_v','beta_j','aa_length','prevalence_bin'],sort=True,observed=True).indices.values():
        index=np.asarray(index,dtype=int)
        if len(index)>1 and 0<labels[index].sum()<len(index):movable[index]=True;groups.append(index)
    rowkeys=[(p,t) for p in patients for t in ('Tumor','Normal')];row_index={k:i for i,k in enumerate(rowkeys)}
    def weights(frame):
        grouped=frame.groupby(['patient_id','Tissue',keycol],observed=True).size()
        ii=[row_index[(p,t)] for p,t,k in grouped.index];jj=[key_index[k] for p,t,k in grouped.index]
        return sparse.csr_matrix((grouped.to_numpy(dtype=float),(ii,jj)),shape=(len(rowkeys),len(keys)))
    cell_weights=weights(cells);clone_weights=weights(cells.drop_duplicates(['patient_id','Tissue','beta_clone']))
    totals=[np.asarray(m.sum(axis=1)).ravel() for m in (clone_weights,cell_weights)];assert all((t>0).all() for t in totals)
    matrices=(clone_weights,cell_weights)
    def statistic(mask):
        return np.array([(np.asarray(m@mask).ravel()/denom).reshape(len(patients),2).dot(np.array([1.,-1.])).mean() for m,denom in zip(matrices,totals)])
    observed=statistic(labels);null=np.empty((PERMUTATIONS,2),dtype=float)
    for draw in range(PERMUTATIONS):
        permuted=labels.copy()
        for index in groups:permuted[index]=rng.permutation(labels[index])
        null[draw]=statistic(permuted)
    key_cells=np.asarray(cell_weights.sum(axis=0)).ravel()
    exchange={'total_keys':len(keys),'total_matched_keys':int(labels.sum()),'exchangeable_keys':int(movable.sum()),
              'exchangeable_matched_keys':int(labels[movable].sum()),'exchangeable_unmatched_keys':int((1-labels[movable]).sum()),
              'total_cells':int(key_cells.sum()),'total_matched_cells':int(key_cells@labels),
              'exchangeable_cells':int(key_cells[movable].sum()),'exchangeable_matched_cells':int(key_cells[movable]@labels[movable]),
              'total_strata':identity.groupby(['beta_v','beta_j','aa_length','prevalence_bin'],observed=True).ngroups,
              'exchangeable_strata':len(groups),'permutation_seed':SEED,'permutation_repetitions':PERMUTATIONS}
    estimable=exchange['exchangeable_matched_keys']>=20 and exchange['exchangeable_unmatched_keys']>=20 and len(patients)>=MIN_PATIENTS
    exchange.update({'exchangeable_key_fraction':exchange['exchangeable_keys']/exchange['total_keys'] if exchange['total_keys'] else 0.,
                     'exchangeable_matched_key_fraction':exchange['exchangeable_matched_keys']/exchange['total_matched_keys'] if exchange['total_matched_keys'] else 0.,
                     'exchangeable_cell_fraction':exchange['exchangeable_cells']/exchange['total_cells'] if exchange['total_cells'] else 0.,
                     'exchangeable_matched_cell_fraction':exchange['exchangeable_matched_cells']/exchange['total_matched_cells'] if exchange['total_matched_cells'] else 0.})
    results=[];nullrows=[]
    for i,metric in enumerate(('matched_clone_fraction','matched_cell_fraction')):
        upper=int((null[:,i]>=observed[i]).sum());lower=int((null[:,i]<=observed[i]).sum())
        p=min(1.,2*min((1+upper)/(PERMUTATIONS+1),(1+lower)/(PERMUTATIONS+1))) if estimable else np.nan
        low,high=np.quantile(null[:,i],[.025,.975])
        results.append({'metric':metric,'family':'structural_tissue_sharing_background2','n':len(patients),'observed_mean_difference':observed[i],
                        'null_mean':null[:,i].mean(),'null_CI_low':low,'null_CI_high':high,'null_sd':null[:,i].std(ddof=1),
                        'excess_over_null_mean':observed[i]-null[:,i].mean(),'upper_tail_count':upper,'lower_tail_count':lower,'P':p,'P_BH_placeholder':p if estimable else 1.,
                        'status':'estimable' if estimable else 'unestimable_fewer_than_20_exchangeable_keys_in_either_label',**exchange})
        for draw in range(PERMUTATIONS):nullrows.append({'draw':draw+1,'metric':metric,'mean_difference':null[draw,i]})
    identity['exchangeable']=movable;identity['match_key']=identity.index
    return pd.DataFrame(results),pd.DataFrame(nullrows),identity.reset_index(drop=True),exchange


def self_test(out):
    checks=[];rng=np.random.default_rng(SEED)
    a=summarize(np.zeros(8),[f'p{i}' for i in range(8)],rng);assert a['P']==1;checks.append('eight_patients_all_zero_P1')
    a=summarize(np.ones(7),[f'p{i}' for i in range(7)],rng);assert np.isnan(a['P']) and a['P_BH_placeholder']==1;checks.append('seven_patients_unestimable_placeholder')
    f=pd.DataFrame({'family':['f']*3,'P':[.01,.04,np.nan],'P_BH_placeholder':[.01,.04,1.]})
    f=adjust_families(f,{'f':3});assert np.allclose(f.q[:2],[.03,.06]) and np.isnan(f.q.iloc[2]);checks.append('declared_BH_family_preserves_unestimable')
    # A large clone must contribute one clone weight, not its cell multiplicity.
    x=pd.DataFrame({'beta_clone':['large']*10+['singleton'+str(i) for i in range(10)],'SubCellType':['state']*10+['other']*10})
    cell=(x.SubCellType=='state').mean();clone=x.assign(v=(x.SubCellType=='state').astype(float)).groupby('beta_clone').v.mean().mean()
    assert cell==.5 and np.isclose(clone,1/11);checks.append('clone_equal_state_composition_not_cell_or_majority_state')
    # Same key and clone in two tissues retain a global match; denominator is beta clones.
    fixture=[]
    for tissue in ('Tumor','Normal'):
        for key,matched,repeat in [('CASSF|TRBV1|TRBJ1-1',True,3),('CASRF|TRBV1|TRBJ1-1',False,1)]:
            for i in range(repeat):fixture.append({'patient_id':'p','Tissue':tissue,'beta_clone':key,'match_AA_VJ_union_ge2':matched,'strict_paired':True})
    rows,contrasts=coverage(pd.DataFrame(fixture),['p'],'AA_VJ','union',2)
    assert all(r['matched_clone_fraction']==.5 and r['matched_cell_fraction']==.75 for r in rows) and all(r['difference']==0 for r in contrasts)
    checks.append('clone_cell_denominators_and_paired_direction')
    # Formal phenotype groups must satisfy cell AND NT-clone gates, never just cell n.
    fixture=[]
    for patient in ['p'+str(i) for i in range(8)]:
        for tissue in ('Tumor','Normal'):
            for matched in (True,False):
                for clone in range(5):
                    for cell in range(4):
                        fixture.append({'patient_id':patient,'Tissue':tissue,'beta_clone':str(matched)+'_'+str(clone),'SubCellType':'s1' if matched else 's2','is_cd8':True,'positive_expression_qc':True,'match_AA_VJ_union_ge2':matched,**{'program_'+p:2. if matched else 1. for p in PROGRAMS}})
    frame=pd.DataFrame(fixture);st,pr,eligibility=phenotype_tables(frame,['p'+str(i) for i in range(8)],['s1','s2'])
    assert eligibility.eligible.all() and np.isclose(st.loc[st.state=='s1','difference'],1.).all() and np.isclose(pr.difference,1.).all()
    checks.append('twenty_cells_five_clones_gate_and_two_weightings')
    frame.loc[frame.beta_clone.str.endswith('_4'),'beta_clone']=frame.loc[frame.beta_clone.str.endswith('_4'),'beta_clone'].str.replace('_4','_3',regex=False)
    st,pr,eligibility=phenotype_tables(frame,['p'+str(i) for i in range(8)],['s1','s2']);assert not eligibility.eligible.any()
    checks.append('twenty_cells_four_clones_ineligible')
    # One label per AA/V/J key is shared globally, including two NT variants.
    fixture=[]
    for patient in ('p0','p1'):
        for tissue in ('Tumor','Normal'):
            for aa,matched,repeats in [('CASSF',True,4 if tissue=='Tumor' else 1),('CASRF',False,1 if tissue=='Tumor' else 3),('CASAF',False,1)]:
                for cell in range(repeats):
                    fixture.append({'patient_id':patient,'Tissue':tissue,'key_AA_VJ':aa+'|TRBV1|TRBJ1-1','beta_aa':aa,'beta_v':'TRBV1','beta_j':'TRBJ1-1','beta_clone':aa+'_'+str(cell%2)+'|TRBV1|TRBJ1-1','match_AA_VJ_union_ge2':matched,'strict_paired':True})
    frame=pd.DataFrame(fixture);bg,null,keys,exchange=structural_background(frame,['p0','p1'],np.random.default_rng(SEED))
    rows,ct=coverage(frame,['p0','p1'],'AA_VJ','union',2)
    for row in bg.to_dict('records'):
        assert np.isclose(row['observed_mean_difference'],np.mean([r['difference'] for r in ct if r['metric']==row['metric']]))
        assert np.isnan(row['P']) and row['P_BH_placeholder']==1.
    assert exchange['exchangeable_keys']==3 and exchange['exchangeable_matched_keys']==1 and null.groupby('metric').mean_difference.nunique().le(3).all()
    checks.append('global_key_permutation_preserves_NT_variants_and_sparse_P')
    fixed=frame.copy();fixed['beta_v']=fixed.beta_aa;bg,null,keys,exchange=structural_background(fixed,['p0','p1'],np.random.default_rng(SEED))
    assert exchange['exchangeable_keys']==0
    for row in bg.to_dict('records'):assert np.allclose(null.loc[null.metric==row['metric'],'mean_difference'],row['observed_mean_difference'])
    checks.append('singleton_and_one_label_strata_fixed_without_relaxation')
    # Synthetic end-to-end smoke covers receipt construction, every full family,
    # deterministic bootstrap replay, validation and manifest/report generation.
    patients=['s'+str(i).zfill(2) for i in range(19)];states=['c'+str(i).zfill(2) for i in range(1,33)];fixture=[]
    for patient in patients:
        for tissue in ('Tumor','Normal'):
            for matched in (True,False):
                aa='CASSF' if matched else 'CASRF'
                for clone in range(5):
                    for cell in range(6):
                        beta_clone=('NTM' if matched else 'NTU')+str(clone)+'|TRBV1|TRBJ1-1'
                        fixture.append({'cell_name':patient+tissue+str(matched)+str(clone)+str(cell),'patient_id':patient,'Ident':patient+tissue,'Tissue':tissue,'SubCellType':'c01' if matched else 'c02','is_cd8':True,'positive_expression_qc':True,'strict_paired':True,'beta_aa':aa,'beta_v':'TRBV1','beta_j':'TRBJ1-1','beta_clone':beta_clone,'key_AA_VJ':aa+'|TRBV1|TRBJ1-1','key_NT_VJ':beta_clone,'key_AA':aa,**{'program_'+p:2. if matched else 1. for p in PROGRAMS},**{f'match_{d}_{cohort}_ge{threshold}':bool(matched and cohort!='external') for d in IDENTITIES for cohort in ('home','external','union') for threshold in (1,2)}})
    synthetic=pd.DataFrame(fixture);original_loader=globals()['load_inputs']
    try:
        globals()['load_inputs']=lambda root:(synthetic,pd.DataFrame(),patients,states,[root/'bulk_sc_match/PROTOCOL.md',Path(__file__)],{'status':'synthetic_only_no_cohort_outcomes'})
        with tempfile.TemporaryDirectory(prefix='_synthetic_pipeline_',dir=out) as temp:
            run(DEFAULT_ROOT,Path(temp))
            receipt=json.loads((Path(temp)/'verification_receipt.json').read_text());assert receipt['status']=='PASS'
            assert (Path(temp)/'ANALYSIS_REPORT.md').exists() and (Path(temp)/'output_manifest.tsv').exists()
    finally:globals()['load_inputs']=original_loader
    checks.append('synthetic_end_to_end_metadata_validation_and_full_families')
    dump(out/'self_test_receipt.json',{'status':'PASS','checks':checks,'scientific_outcome_tests_run':0,'script_sha256':sha(__file__)})
    print(json.dumps({'self_test':'PASS','checks':len(checks),'scientific_outcome_tests_run':0}))


def inspect(root,out):
    import pyarrow.parquet as pq
    records=[]
    for name in ('sc_cells.parquet','key_catalog.parquet','bulk_support.parquet'):
        path=root/'bulk_sc_match/server_data'/name
        if path.exists():
            pf=pq.ParquetFile(path);records.append({'file':str(path),'rows':pf.metadata.num_rows,'columns':pf.schema_arrow.names,'sha256':sha(path)})
        else:records.append({'file':str(path),'status':'PENDING_INPUT'})
    dump(out/'schema_inspection.json',{'status':'INSPECTED_NO_INFERENCE','files':records,'scientific_outcome_tests_run':0})
    print(json.dumps({'status':'INSPECTED_NO_INFERENCE','files_present':sum('rows' in r for r in records)}))


def load_inputs(root):
    work=root/'bulk_sc_match';protocol=work/'PROTOCOL.md';freeze=work/'PROTOCOL_FREEZE.json'
    assert sha(protocol)==PROTOCOL_SHA and json.loads(freeze.read_text())['sha256']==PROTOCOL_SHA,'protocol changed'
    receipt_path=work/'ingest/ingest_receipt.json';receipt=json.loads(receipt_path.read_text())
    assert str(receipt.get('status','')).upper() in ('PASS','PASSED','COMPLETE'),('ingest_not_complete',receipt.get('status'))
    assert receipt.get('full_EOF') is True and receipt.get('selected_libraries')==339 and receipt.get('independent_blood_patients')==339,'incomplete blood scan'
    assert receipt.get('protocol_sha256')==PROTOCOL_SHA,'producer protocol mismatch'
    outputs={str(Path(r['path']).resolve()):r for r in receipt['outputs']}
    for name in ('sc_cells.parquet','key_catalog.parquet','bulk_support.parquet'):
        path=work/'server_data'/name;record=outputs.get(str(path.resolve()))
        assert record is not None,('producer output SHA absent',str(path))
        assert record['bytes']==path.stat().st_size and record['sha256']==sha(path),('producer output SHA changed',str(path))
    verification_path=work/'ingest/verification_receipt.json'
    verification=json.loads(verification_path.read_text())
    assert str(verification.get('status','')).upper() in ('PASS','PASSED','COMPLETE'),'producer independent verification incomplete'
    assert verification.get('protocol_sha256')==PROTOCOL_SHA,'producer verifier protocol mismatch'
    for name,path in [('sc_cells.parquet',work/'server_data/sc_cells.parquet'),('key_catalog.parquet',work/'server_data/key_catalog.parquet'),('bulk_support.parquet',work/'server_data/bulk_support.parquet'),('ingest_receipt.json',receipt_path)]:
        record=verification['outputs'][name]
        assert Path(record['path']).resolve()==path.resolve() and record['bytes']==path.stat().st_size and record['sha256']==sha(path),('producer verifier SHA mismatch',name)
    cellpath=work/'server_data/sc_cells.parquet';catalogpath=work/'server_data/key_catalog.parquet'
    cells=pd.read_parquet(cellpath);catalog=pd.read_parquet(catalogpath)
    assert cells.cell_name.is_unique and catalog[['definition','match_key']].drop_duplicates().shape[0]==len(catalog)
    pairpath=SOURCE_ROOT/'single_cell/paired_patient_inventory.tsv';pairs=pd.read_csv(pairpath,sep='\t',float_precision='round_trip')
    patients=sorted(pairs.loc[pairs.primary_pair_status=='included','patient_id'].tolist());assert len(patients)==19
    primary=cells.loc[bool_column(cells,'is_primary_pair')&bool_column(cells,'is_primary_cell')].copy()
    assert bool_column(primary,'RNA_joined').all() and (primary.MajorCellType=='T').all() and bool_column(primary,'single_beta').all()
    assert set(primary.patient_id)==set(patients) and set(primary.Tissue)=={'Tumor','Normal'}
    assert len(primary)==int(pairs.loc[pairs.primary_pair_status=='included',['tumor_single_beta_cells','normal_single_beta_cells']].to_numpy().sum()),'primary accepted-cell registration differs'
    for c in ('strict_paired','is_cd8','positive_expression_qc'):primary[c]=bool_column(primary,c)
    for definition in IDENTITIES:
        group=catalog[catalog.definition==definition].set_index('match_key')
        for cohort,denom in (('home',137),('external',202),('union',339)):
            assert (group[cohort+'_denominator']==denom).all(),('blood_denominator',definition,cohort)
            for threshold in (1,2):
                col=f'match_{definition}_{cohort}_ge{threshold}';primary[col]=bool_column(primary,col)
                expected=primary['key_'+definition].map(group[f'{cohort}_patients_ge{threshold}']).fillna(0).to_numpy()>0
                assert np.array_equal(primary[col].to_numpy(),expected),('catalog_cell_match_disagreement',col)
                assert (primary[col]<=primary[f'match_{definition}_{cohort}_ge1']).all()
        for threshold in (1,2):
            assert (primary[f'match_{definition}_union_ge{threshold}']==(primary[f'match_{definition}_home_ge{threshold}']|primary[f'match_{definition}_external_ge{threshold}'])).all()
    for cohort in ('home','external','union'):
        for t in (1,2):assert (primary[f'match_NT_VJ_{cohort}_ge{t}']<=primary[f'match_AA_VJ_{cohort}_ge{t}']).all() and (primary[f'match_AA_VJ_{cohort}_ge{t}']<=primary[f'match_AA_{cohort}_ge{t}']).all()
    for (patient,tissue),part in primary.groupby(['patient_id','Tissue'],observed=True):assert len(part)>=MIN_PAIRED_CELLS
    statepath=SOURCE_ROOT/'single_cell/state_patient_tests.tsv';states=sorted(pd.read_csv(statepath,sep='\t',float_precision='round_trip').subcelltype.unique());assert len(states)==32 and set(primary.SubCellType)<=set(states)
    sourcepaths=[protocol,freeze,receipt_path,verification_path,work/'ingest/SCHEMA.json',cellpath,catalogpath,pairpath,statepath,SOURCE_ROOT/'single_cell/marker_program_dictionary.tsv']
    return primary,catalog,patients,states,sourcepaths,receipt


def run(root,out):
    started=time.monotonic();primary,catalog,patients,states,sourcepaths,ingest=load_inputs(root)
    rng=np.random.default_rng(SEED);covrows=[];contrasts=[];specs=[]
    configs=[('AA_VJ','union',2,False,'primary_coverage2'),('AA_VJ','home',2,False,'sensitivity_home_external4'),
             ('AA_VJ','external',2,False,'sensitivity_home_external4'),('NT_VJ','union',2,False,'sensitivity_NT_VJ2'),
             ('AA','union',2,False,'sensitivity_AA2'),('AA_VJ','union',1,False,'sensitivity_threshold1_2'),
             ('AA_VJ','union',2,True,'sensitivity_strict_alpha_beta_capture2')]
    for definition,cohort,threshold,strict,family in configs:
        rows,ct=coverage(primary,patients,definition,cohort,threshold,strict)
        for r in rows:r['family']=family
        for r in ct:r['family']=family
        covrows+=rows;contrasts+=ct;specs.append({'analysis':rows[0]['analysis'],'family':family,'definition':definition,'catalog_cohort':cohort,'threshold':threshold,'capture_restriction':rows[0]['capture_restriction']})
    cov=save(out/'patient_coverage_metrics.tsv',covrows);ct=save(out/'coverage_patient_contrasts.tsv',contrasts);save(out/'analysis_dictionary.tsv',specs)
    tests=[]
    for spec in specs:
        for metric in ('matched_clone_fraction','matched_cell_fraction'):
            part=ct[(ct.analysis==spec['analysis'])&(ct.metric==metric)&ct.eligible].sort_values('patient_id')
            tests.append({**spec,'metric':metric,**summarize(part.difference.to_numpy(),part.patient_id.tolist(),rng)})
    sizes={name:(4 if name=='sensitivity_home_external4' else 2) for name in set(x['family'] for x in specs)}
    covtests=save(out/'coverage_tests.tsv',adjust_families(pd.DataFrame(tests),sizes))
    state,program,eligibility=phenotype_tables(primary,patients,states)
    save(out/'state_patient_contrasts.tsv',state);save(out/'program_patient_contrasts.tsv',program);save(out/'phenotype_eligibility.tsv',eligibility)
    stests=[];ptests=[]
    for weighting in ('cell_weighted','clone_equal'):
        family='matched_states_'+weighting+'64'
        for tissue in ('Tumor','Normal'):
            for s in states:
                part=state[(state.weighting==weighting)&(state.tissue==tissue)&(state.state==s)&state.eligible].sort_values('patient_id')
                stests.append({'family':family,'state':s,'tissue':tissue,'weighting':weighting,**summarize(part.difference.to_numpy(),part.patient_id.tolist(),rng)})
            pfamily='matched_programs_'+tissue+'_'+weighting+'6'
            for p in PROGRAMS:
                part=program[(program.weighting==weighting)&(program.tissue==tissue)&(program.program==p)&program.eligible].sort_values('patient_id')
                ptests.append({'family':pfamily,'program':p,'tissue':tissue,'weighting':weighting,**summarize(part.difference.to_numpy(),part.patient_id.tolist(),rng)})
    stests=save(out/'state_tests.tsv',adjust_families(pd.DataFrame(stests),{'matched_states_cell_weighted64':64,'matched_states_clone_equal64':64}))
    ptests=save(out/'program_tests.tsv',adjust_families(pd.DataFrame(ptests),{x['family']:6 for x in ptests}))
    background,null,keys,exchange=structural_background(primary,patients,np.random.default_rng(SEED))
    background=save(out/'structural_background_tests.tsv',adjust_families(background,{'structural_tissue_sharing_background2':2}));save(out/'structural_background_null.tsv',null);save(out/'structural_background_key_universe.tsv',keys)
    partitions=[]
    for definition in IDENTITIES:
        key='key_'+definition
        for match_key,part in primary.groupby(key,sort=True,observed=True):
            tissues=set(part.Tissue);partition='shared' if len(tissues)==2 else ('tumor_only' if 'Tumor' in tissues else 'normal_only')
            partitions.append({'definition':definition,'match_key':match_key,'partition':partition,'n_Tumor_cells':int((part.Tissue=='Tumor').sum()),'n_Normal_cells':int((part.Tissue=='Normal').sum()),'n_tissue_patients':part.patient_id.nunique(),
                               **{f'{cohort}_ge{t}':bool(part[f'match_{definition}_{cohort}_ge{t}'].iloc[0]) for cohort in ('home','external','union') for t in (1,2)}})
    partition=save(out/'tissue_reference_key_partitions.tsv',partitions)
    save(out/'tissue_reference_partition_summary.tsv',partition.groupby(['definition','partition'],observed=True).agg(n_keys=('match_key','size'),n_Tumor_cells=('n_Tumor_cells','sum'),n_Normal_cells=('n_Normal_cells','sum'),union_ge2_keys=('union_ge2','sum')).reset_index())
    primary[['cell_name','patient_id','Ident','Tissue','SubCellType','beta_clone','key_AA_VJ','match_AA_VJ_union_ge2','is_cd8','strict_paired']].to_parquet(out/'primary_cell_match_audit.parquet',index=False)
    save(out/'input_manifest.tsv',[{'path':str(p),'sha256':sha(p),'bytes':p.stat().st_size} for p in sourcepaths])
    result={'status':'PASS','completed_utc':datetime.now(timezone.utc).isoformat(),'protocol_sha256':PROTOCOL_SHA,'script_sha256':sha(__file__),'source_ingest_status':ingest.get('status'),
            'n_primary_patients':len(patients),'n_primary_cells':len(primary),'n_primary_samples':primary.Ident.nunique(),'primary_patients':patients,
            'coverage_test_rows':len(covtests),'state_test_rows':len(stests),'program_test_rows':len(ptests),'structural_background_test_rows':len(background),
            'bootstrap_repetitions':BOOTSTRAPS,'bootstrap_seed':SEED,'bootstrap_CI_target':'mean_patient_difference_only','permutations':PERMUTATIONS,'permutation_seed':SEED,
            'formal_min_patients':MIN_PATIENTS,'group_min_cells':MIN_CELLS,'group_min_beta_NTVJ_clones':MIN_CLONES,'declared_families':{**sizes,'matched_states_cell_weighted64':64,'matched_states_clone_equal64':64,**{family:6 for family in ptests.family.unique()},'structural_tissue_sharing_background2':2},
            'exchangeability':exchange,'new_classifier_fits':0,'raw_clone_parquet_rescans':0,'runtime_seconds':time.monotonic()-started,'python':platform.python_version(),'threads_per_library':4,
            'interpretation':['independent_cross_patient_beta_sequence_reference_only','no_tissue_trafficking_or_complete_alpha_beta_identity','structural_background_not_independent_healthy_publicness','technical_PASS_not_biology_validation']}
    dump(out/'analysis_receipt.json',result)
    verify(root,out)
    write_report(out,result,covtests,stests,ptests,background,eligibility)
    refresh_manifest(out)
    dump(out/'readiness.json',{'status':'COMPLETE_ANALYSIS','outcome_tests_run':len(covtests)+len(stests)+len(ptests)+len(background),'new_classifier_fits':0})
    print(json.dumps({k:result[k] for k in ('status','n_primary_patients','n_primary_cells','coverage_test_rows','state_test_rows','program_test_rows','runtime_seconds')}))


def refresh_manifest(out):
    manifest=[{'path':str(p),'sha256':sha(p),'bytes':p.stat().st_size} for p in sorted(out.iterdir()) if p.is_file() and p.name not in ('output_manifest.tsv','readiness.json')]
    save(out/'output_manifest.tsv',manifest)


def verify(root,out):
    """Independent table arithmetic and declared-family/eligibility validation."""
    checks=[]
    def check(name,ok):assert ok,name;checks.append(name)
    cov=pd.read_csv(out/'patient_coverage_metrics.tsv',sep='\t',float_precision='round_trip');ct=pd.read_csv(out/'coverage_patient_contrasts.tsv',sep='\t',float_precision='round_trip')
    check('patient_coverage_rows_19x2x7',len(cov)==19*2*7)
    check('coverage_cell_fractions',np.allclose(cov.matched_cell_fraction,cov.n_matched_cells/cov.n_cells,equal_nan=True))
    check('coverage_clone_fractions',np.allclose(cov.matched_clone_fraction,cov.n_matched_clones/cov.n_unique_beta_clones,equal_nan=True))
    check('paired_contrast_arithmetic',np.allclose(ct.difference,ct.tumor-ct.normal,equal_nan=True))
    check('contrast_unique_unit',not ct.duplicated(['analysis','metric','patient_id']).any())
    for name,expected in (('coverage_tests.tsv',14),('state_tests.tsv',128),('program_tests.tsv',24),('structural_background_tests.tsv',2)):
        table=pd.read_csv(out/name,sep='\t',float_precision='round_trip');check(name+'_complete_rows',len(table)==expected)
        for family,part in table.groupby('family'):
            check(name+'_'+family+'_declared_size',len(part)==int(part.family_size.iloc[0]))
            check(name+'_'+family+'_unestimable_not_reported_Pq',part.loc[part.P.isna(),'q'].isna().all())
            p=part.P_BH_placeholder.to_numpy();order=np.argsort(p,kind='stable');a=np.minimum.accumulate((p[order]*len(p)/np.arange(1,len(p)+1))[::-1])[::-1];q=np.empty(len(p));q[order]=np.minimum(1,a)
            check(name+'_'+family+'_BH_recompute',np.allclose(q,part.q_BH_placeholder))
    for name in ('state_patient_contrasts.tsv','program_patient_contrasts.tsv'):
        frame=pd.read_csv(out/name,sep='\t',float_precision='round_trip')
        expected=(frame.n_matched_cells>=20)&(frame.n_unmatched_cells>=20)&(frame.n_matched_clones>=5)&(frame.n_unmatched_clones>=5)
        if name.startswith('state'):check(name+'_eligibility',np.array_equal(frame.eligible,expected))
        else:check(name+'_eligibility_not_broader_than_thresholds',not (frame.eligible & ~expected).any())
        check(name+'_no_patient_duplication',not frame.duplicated(['patient_id','tissue','weighting','state' if name.startswith('state') else 'program']).any())
    state=pd.read_csv(out/'state_patient_contrasts.tsv',sep='\t',float_precision='round_trip');program=pd.read_csv(out/'program_patient_contrasts.tsv',sep='\t',float_precision='round_trip')
    check('state_complete_rows_19x2x32x2',len(state)==19*2*32*2)
    check('program_complete_rows_19x2x6x2',len(program)==19*2*6*2)
    for group,part in state.groupby(['patient_id','tissue','weighting']):
        for col in ('matched_fraction','unmatched_fraction'):
            finite=part[col].notna()
            check('state_own_group_denominator_'+str(group)+'_'+col,not finite.any() or np.isclose(part.loc[finite,col].sum(),1.))
    rng=np.random.default_rng(SEED)
    coverage_test=pd.read_csv(out/'coverage_tests.tsv',sep='\t',float_precision='round_trip');state_test=pd.read_csv(out/'state_tests.tsv',sep='\t',float_precision='round_trip');program_test=pd.read_csv(out/'program_tests.tsv',sep='\t',float_precision='round_trip')
    # Replay the exact fixed-seed stream order used by run: coverage first,
    # then 32 states and six programs within each weighting/tissue.
    specifications=[('coverage_tests.tsv',ct,['analysis','metric'],coverage_test)]
    for weighting in ('cell_weighted','clone_equal'):
        for tissue in ('Tumor','Normal'):
            specifications.append(('state_tests.tsv',state,['tissue','state','weighting'],state_test[(state_test.weighting==weighting)&(state_test.tissue==tissue)]))
            specifications.append(('program_tests.tsv',program,['tissue','program','weighting'],program_test[(program_test.weighting==weighting)&(program_test.tissue==tissue)]))
    for filename,contrasts,groupcols,tests in specifications:
        for row in tests.to_dict('records'):
            selected=contrasts.eligible.copy()
            for col in groupcols:selected &= contrasts[col]==row[col]
            part=contrasts.loc[selected].sort_values('patient_id');values=part.difference.to_numpy(dtype=float)
            label=filename+'_'+str(tuple(row[c] for c in groupcols))
            check(label+'_n_and_patient_ids',len(part)==row['n'] and ';'.join(part.patient_id)==(row['included_patients'] if pd.notna(row['included_patients']) else ''))
            low,high=bootstrap_mean(values,rng)
            for field,value in [('mean_difference',values.mean() if len(values) else np.nan),('median_difference',np.median(values) if len(values) else np.nan),('CI_low',low),('CI_high',high)]:
                check(label+'_'+field,np.isclose(row[field],value,atol=1e-12,rtol=1e-10,equal_nan=True))
            if len(values)>=MIN_PATIENTS:
                with warnings.catch_warnings():
                    warnings.simplefilter('ignore',UserWarning)
                    expected_p=1. if np.all(values==0) else float(wilcoxon(values,zero_method='wilcox',correction=False,method='approx',alternative='two-sided').pvalue)
                check(label+'_WilcoxonP',np.isclose(row['P'],expected_p,atol=1e-12,rtol=1e-10))
            else:check(label+'_formal_P_missing',pd.isna(row['P']))
    null=pd.read_csv(out/'structural_background_null.tsv',sep='\t',float_precision='round_trip');check('global_null_draws1000x2',len(null)==PERMUTATIONS*2 and null.groupby('metric').draw.nunique().eq(PERMUTATIONS).all())
    bg=pd.read_csv(out/'structural_background_tests.tsv',sep='\t',float_precision='round_trip')
    for row in bg.to_dict('records'):
        values=null.loc[null.metric==row['metric'],'mean_difference'].to_numpy();observed=row['observed_mean_difference']
        main=pd.read_csv(out/'coverage_tests.tsv',sep='\t',float_precision='round_trip');main=main[(main.family=='primary_coverage2')&(main.metric==row['metric'])].iloc[0]
        check(row['metric']+'_observed_equals_primary',np.isclose(observed,main.mean_difference,atol=1e-12,rtol=1e-10))
        check(row['metric']+'_null_moments',np.allclose([row['null_mean'],row['null_sd'],row['null_CI_low'],row['null_CI_high']],[values.mean(),values.std(ddof=1),*np.quantile(values,[.025,.975])],atol=1e-12,rtol=1e-10))
        check(row['metric']+'_null_excess',np.isclose(row['excess_over_null_mean'],observed-values.mean(),atol=1e-12,rtol=1e-10))
        if row['status']=='estimable':
            p=min(1.,2*min((1+int((values>=observed).sum()))/(PERMUTATIONS+1),(1+int((values<=observed).sum()))/(PERMUTATIONS+1)))
            check(row['metric']+'_empiricalP',np.isclose(row['P'],p,atol=1e-12,rtol=1e-10))
        else:check(row['metric']+'_empiricalP_missing',pd.isna(row['P']))
    keys=pd.read_csv(out/'structural_background_key_universe.tsv',sep='\t',float_precision='round_trip')
    check('background_unique_global_keys',keys.match_key.is_unique)
    check('background_prevalence_bins',np.array_equal(keys.prevalence_bin,np.minimum(3,keys.patient_prevalence)))
    movable=np.zeros(len(keys),dtype=bool)
    for index in keys.groupby(['beta_v','beta_j','aa_length','prevalence_bin'],sort=True,observed=True).indices.values():
        index=np.asarray(index,dtype=int);labels=keys.iloc[index].matched.to_numpy(dtype=bool)
        if len(index)>1 and labels.any() and (~labels).any():movable[index]=True
    check('background_exchangeable_strata_exact',np.array_equal(movable,keys.exchangeable))
    check('background_exchangeability_mass',all(row['total_keys']==len(keys) and row['exchangeable_keys']==movable.sum() and row['exchangeable_matched_keys']==int((keys.matched & movable).sum()) for row in bg.to_dict('records')))
    for row in pd.read_csv(out/'input_manifest.tsv',sep='\t',float_precision='round_trip').to_dict('records'):check('input_SHA_'+row['path'],sha(row['path'])==row['sha256'])
    check('protocolSHA_unmodified',sha(root/'bulk_sc_match/PROTOCOL.md')==PROTOCOL_SHA)
    dump(out/'verification_receipt.json',{'status':'PASS','checks':len(checks),'check_names':checks,'scientific_table_reaggregation_only':True,'script_sha256':sha(__file__)})


def markdown_table(frame):
    frame=frame.reset_index() if not isinstance(frame.index,pd.RangeIndex) else frame
    def value(v):
        if pd.isna(v):return 'NA'
        return format(v,'.6g') if isinstance(v,(float,np.floating)) else str(v).replace('|',' / ')
    return '\n'.join(['| '+' | '.join(map(str,frame.columns))+' |','| '+' | '.join(['---']*len(frame.columns))+' |']+['| '+' | '.join(value(v) for v in row)+' |' for row in frame.itertuples(index=False,name=None)])


def write_report(out,receipt,cov,state,program,background,eligibility):
    primary=cov[cov.family=='primary_coverage2']
    lines=['# Full blood-to-tissue matching: fixed-protocol patient analysis','',
           f"Primary dataset: {receipt['n_primary_patients']} baseline CRC patients, {receipt['n_primary_samples']} paired tissue samples and {receipt['n_primary_cells']:,} accepted single-beta RNA T cells. Matching uses union339 patient blood references, complete AA|V|J with >=2 counts within at least one blood patient. Blood and tissue donors are unlinked.",'',
           '## Primary paired coverage','',markdown_table(primary[['metric','n','mean_difference','CI_low','CI_high','P','q','status']]),'',
           '## Conditional structural/tissue-sharing background','',markdown_table(background[['metric','observed_mean_difference','null_mean','null_CI_low','null_CI_high','excess_over_null_mean','P','q','status']]),'',
           'This conditions on measured V/J, CDR3 length and tissue-donor sharing. It is not an independent healthy-population publicness correction or second replication. All key labels are global across both tissues, patients and nucleotide variants. Fixed-label strata remain fixed; exchangeability coverage is in structural_background_tests.tsv.','',
           '## Eligibility and full families','',markdown_table(eligibility.groupby(['tissue','context']).agg(eligible_patients=('eligible','sum'))),'']
    for weighting in ('cell_weighted','clone_equal'):
        s=state[(state.weighting==weighting)&state.P.notna()];p=program[(program.weighting==weighting)&program.P.notna()]
        lines.append(f'{weighting}: states, {len(s)}/64 estimable rows, {int((s.q<.05).sum())} BH64-supported; programs, {len(p)}/12 estimable rows, {int((p.q<.05).sum())} supported across separately declared tissue BH6 families.')
    lines+=['','All 128 state, 24 program, 14 coverage and two background rows are retained, including sparse/unestimable endpoints. Formal patient tests require at least eight eligible independent patients; matched/unmatched groups require at least20 cells and five distinct betaNTVJ clones per group. P=1 placeholders preserve family size only; unestimable reported P/q remain missing.','',
            'Mean and median patient differences are reported; 95% percentile confidence intervals for mean differences use 10,000 patient resamples. Cell/clone counts and bootstrap/permutation draws do not enlarge patient n.','',
            'Program scores are the six existing fixed conventional-CD8 scores. Matched/unmatched score or state associations are transcriptional context, not antigen specificity, within-cell activation, whole-alpha-beta receptor identity, tissue trafficking or causal explanation of nodal-model performance. No nodal classifier or gene list was selected using this extension.','',
            'Patient-level contrasts and eligibility are in coverage_patient_contrasts.tsv, state_patient_contrasts.tsv, program_patient_contrasts.tsv and phenotype_eligibility.tsv. Clone-equal sensitivities average within betaNTVJ clones before equal-clone weighting. All fixed sensitivities are in coverage_tests.tsv and analysis_dictionary.tsv.']
    (out/'ANALYSIS_REPORT.md').write_text('\n'.join(lines)+'\n')


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,default=DEFAULT_ROOT)
    action=parser.add_mutually_exclusive_group(required=True);action.add_argument('--run',action='store_true');action.add_argument('--self-test',action='store_true');action.add_argument('--inspect',action='store_true');action.add_argument('--verify-only',action='store_true')
    args=parser.parse_args();out=args.root/'bulk_sc_match/analysis';out.mkdir(parents=True,exist_ok=True)
    if args.self_test:self_test(out)
    elif args.inspect:inspect(args.root,out)
    elif args.verify_only:verify(args.root,out);refresh_manifest(out);print('verification PASS; manifest refreshed')
    else:run(args.root,out)

if __name__=='__main__':main()
