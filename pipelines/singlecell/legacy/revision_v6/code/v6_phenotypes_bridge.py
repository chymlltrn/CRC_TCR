#!/usr/bin/env python3
"""Frozen v6 phenotype and autologous/independent-bulk bridge; remote only."""
from release_config import release_path, private_value
import os
for name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[name]='4'
import argparse, hashlib, json, platform, warnings
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

DEFAULT_ROOT=Path(str(release_path('v6_phenotypes_bridge_source_13')))
PARENT_SHA='922145f07ff94cecdfc3c575732152d2276d31def5c2ef2beb781ced9a998a20'
ADDENDUM_SHA='40191561f8320d38050da4639ad861a6f50b8f352b5bd5299b6a74c391127ac5'
SEED=20261004; BOOTSTRAPS=10000; MIN_PATIENTS=8
PROGRAMS=('cytotoxic','exhaustion','naive_memory','interferon','residency','proliferation')
TISSUES=('Tumor','Normal'); WEIGHTS=('clone_equal','cell_weighted')
PATIENTS=(private_value('v6_phenotypes_bridge_value_19_10'),private_value('v6_phenotypes_bridge_value_19_16'),private_value('v6_phenotypes_bridge_value_19_22'),private_value('v6_phenotypes_bridge_value_19_28'),private_value('v6_phenotypes_bridge_value_19_34'),private_value('v6_phenotypes_bridge_value_19_40'),private_value('v6_phenotypes_bridge_value_19_46'),private_value('v6_phenotypes_bridge_value_19_52'),private_value('v6_phenotypes_bridge_value_19_58'),private_value('v6_phenotypes_bridge_value_19_64'),private_value('v6_phenotypes_bridge_value_19_70'),private_value('v6_phenotypes_bridge_value_19_76'),private_value('v6_phenotypes_bridge_value_19_82'),private_value('v6_phenotypes_bridge_value_19_88'),private_value('v6_phenotypes_bridge_value_19_94'),private_value('v6_phenotypes_bridge_value_19_100'),private_value('v6_phenotypes_bridge_value_19_106'),private_value('v6_phenotypes_bridge_value_19_112'),private_value('v6_phenotypes_bridge_value_19_118'))
STATES=('c01_CD4_Tn_CCR7','c02_CD4_Tn_SELL','c03_CD4_Tn_NR4A2','c04_CD4_Tcm_ANXA1','c05_CD4_Tcm_GPR183','c06_CD4_Trm_HSPA1A','c07_CD4_Th17_CTSH','c08_CD4_Tfh_CXCL13_IL6ST','c09_CD4_Th1_CXCL13_HAVCR2','c10_CD4_Temra_GZMB','c11_CD4_Treg_FOXP3','c12_CD4_Treg_KLRB1','c13_CD4_Treg_TNFRSF9','c14_CD4_MT','c15_CD8_Tn_CCR7','c16_CD8_Tn_SELL','c17_CD8_Tcm_GPR183','c18_CD8_Tcm_ANXA1','c19_CD8_Tem_CMC1','c20_CD8_Tem_GZMK','c21_CD8_Trm_XCL1','c22_CD8_Trm_HSPA1B','c23_CD8_Tex_LAYN','c24_CD8_Temra_CX3CR1','c25_CD8_Temra_TYROBP','c26_CD8_MAIT_KLRB1','c27_CD8_MAIT_SLC4A10','c28_CD8_IEL_CD160','c29_T_MKI67','c30_gdT_TRDV2','c31_DNT','c32_DPT')
CD8_STATES=STATES[14:25]+(STATES[27],)
CATEGORIES=('both','own_only','bulk_only','neither')


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(8*1024*1024),b''):h.update(block)
    return h.hexdigest()


def dump(path,value):
    Path(path).write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')


def save(path,rows):
    f=rows if isinstance(rows,pd.DataFrame) else pd.DataFrame(rows)
    f.to_csv(path,sep='\t',index=False,na_rep='NA');return f


def boolean(series):
    if pd.api.types.is_bool_dtype(series):return series.fillna(False).astype(bool)
    vals=series.astype(str).str.lower();assert vals.isin(['true','false','1','0']).all(),('invalid_boolean',series.name)
    return vals.isin(['true','1'])


def endpoint_seed(token):
    return (SEED+int(hashlib.sha256(token.encode()).hexdigest()[:8],16))%(2**32)


def summarize(values,patients,token,formal=True):
    x=np.asarray(values,dtype=float);assert len(x)==len(patients) and len(set(patients))==len(patients)
    assert np.isfinite(x).all();n=len(x);seed=endpoint_seed(token)
    if n:
        rng=np.random.default_rng(seed)
        means=x[rng.integers(n,size=(BOOTSTRAPS,n))].mean(axis=1)
        lo,hi=np.quantile(means,[.025,.975]);mean=float(x.mean());median=float(np.median(x))
    else:lo=hi=mean=median=np.nan
    valid=formal and n>=MIN_PATIENTS
    if valid:
        if np.all(x==0):p=1.;status='estimable_all_zero'
        else:
            with warnings.catch_warnings():
                warnings.simplefilter('ignore',UserWarning)
                p=float(wilcoxon(x,zero_method='wilcox',method='approx',correction=False,alternative='two-sided').pvalue)
            status='estimable'
    else:p=np.nan;status='descriptive_only' if not formal else 'unestimable_fewer_than_8_patients'
    return dict(n=n,mean_difference=mean,median_difference=median,CI_low=lo,CI_high=hi,
                CI_target='mean_patient_difference',CI_method='patient_percentile_bootstrap',
                bootstrap_repetitions=BOOTSTRAPS,bootstrap_seed=seed,P=p,P_BH_placeholder=p if valid else 1.,
                status=status,included_patients=';'.join(patients))


def adjust(frame):
    frame=frame.copy();frame['q']=np.nan;frame['q_BH_placeholder']=np.nan
    for family,part in frame.groupby('family',sort=False):
        idx=part.index;p=part.P_BH_placeholder.to_numpy(dtype=float);n=int(part.family_size.iloc[0])
        assert len(part)==n and part.family_size.eq(n).all(),('incomplete_family',family,len(part),n)
        order=np.argsort(p,kind='stable');s=np.minimum.accumulate((p[order]*n/np.arange(1,n+1))[::-1])[::-1]
        q=np.empty(n);q[order]=np.minimum(1,s);frame.loc[idx,'q_BH_placeholder']=q
        good=part.P.notna().to_numpy();frame.loc[idx[good],'q']=q[good]
    return frame


def test_table(contrasts,cols,family_function,size_function,formal_function=lambda x:True):
    rows=[]
    for keys,part in contrasts.groupby(cols,sort=False,dropna=False):
        keys=keys if isinstance(keys,tuple) else (keys,);info=dict(zip(cols,keys))
        ok=part[part.eligible & np.isfinite(part.difference)]
        family=family_function(info);formal=bool(formal_function(info))
        rows.append(info|dict(family=family,family_size=size_function(info))|
                    summarize(ok.difference,ok.patient_id.tolist(),family+'|'+str(keys),formal))
    return adjust(pd.DataFrame(rows))


def group_gate(a,b):
    c=dict(n_group_a_cells=len(a),n_group_b_cells=len(b),n_group_a_clones=a.beta_clone.nunique(),n_group_b_clones=b.beta_clone.nunique())
    good=min(c['n_group_a_cells'],c['n_group_b_cells'])>=20 and min(c['n_group_a_clones'],c['n_group_b_clones'])>=5
    return c,good,'' if good else 'each_group_requires_20_cells_and_5_NTVJ_clones'


def program_means(frame,weight):
    if not len(frame):return {p:np.nan for p in PROGRAMS}
    columns=['program_'+p for p in PROGRAMS]
    assert np.isfinite(frame[columns].to_numpy()).all(),('nonfinite_fixed_program_score',len(frame))
    f=frame[columns] if weight=='cell_weighted' else frame.groupby('beta_clone',observed=True)[columns].mean()
    return {p:float(f['program_'+p].mean()) for p in PROGRAMS}


def state_fractions(frame,weight):
    if not len(frame):return {s:np.nan for s in STATES}
    if weight=='cell_weighted':return {s:float(frame.SubCellType.eq(s).mean()) for s in STATES}
    tab=pd.crosstab(frame.beta_clone,frame.SubCellType).reindex(columns=STATES,fill_value=0)
    frac=tab.div(tab.sum(axis=1),axis=0).mean(axis=0)
    return frac.to_dict()


def expansion(cells,patients):
    x=cells[cells.is_cd8 & cells.positive_expression_qc].copy()
    x['within_cd8_clone_cells']=x.groupby(['patient_id','Tissue','beta_clone'],observed=True).cell_name.transform('size')
    x['expansion_group']=np.where(x.within_cd8_clone_cells>=2,'expanded','singleton')
    x['clone_size_bin']=pd.cut(x.within_cd8_clone_cells,bins=[0,1,4,np.inf],labels=['1','2-4','>=5']).astype(str)
    metrics=[];elig=[];contrasts=[];lookup={}
    for pid in patients:
        gates=[];numbers={}
        for tissue in TISSUES:
            part=x[(x.patient_id==pid)&(x.Tissue==tissue)];a=part[part.expansion_group=='expanded'];b=part[part.expansion_group=='singleton']
            counts,ok,why=group_gate(a,b);gates.append(ok)
            numbers.update({tissue+'_'+k:v for k,v in counts.items()})
            for weight in WEIGHTS:
                am=program_means(a,weight);bm=program_means(b,weight)
                for p in PROGRAMS:
                    row=dict(patient_id=pid,tissue=tissue,program=p,weighting=weight,group_a='expanded',group_b='singleton',
                             group_a_mean=am[p],group_b_mean=bm[p],difference=am[p]-bm[p],eligible=ok,exclusion=why,**counts)
                    metrics.append(row);lookup[pid,tissue,weight,p]=row
        ok=all(gates);elig.append(dict(patient_id=pid,eligible=ok,exclusion='' if ok else 'all_four_groups_require_20_cells_and_5_clones',**numbers))
        for weight in WEIGHTS:
            for p in PROGRAMS:
                t=lookup[pid,'Tumor',weight,p];n=lookup[pid,'Normal',weight,p]
                contrasts.append(dict(patient_id=pid,program=p,weighting=weight,tumor_difference=t['difference'],normal_difference=n['difference'],
                                      difference=t['difference']-n['difference'],eligible=ok,exclusion='' if ok else 'all_four_groups_require_20_cells_and_5_clones'))
    contrast=pd.DataFrame(contrasts)
    tests=test_table(contrast,['program','weighting'],lambda k:'expansion_interaction_'+k['weighting']+'_BH6',lambda k:6)
    return x,pd.DataFrame(metrics),pd.DataFrame(elig),contrast,tests


def ownblood(cells,patients,high_depth):
    states=[];programs=[];elig=[]
    for threshold in (1,2):
        label='ownblood_NT_VJ_ge'+str(threshold)
        for pid in patients:
            for tissue in TISSUES:
                part=cells[(cells.patient_id==pid)&(cells.Tissue==tissue)]
                for context in ('states','programs'):
                    z=part if context=='states' else part[part.is_cd8 & part.positive_expression_qc]
                    a=z[z[label]];b=z[~z[label]];counts,group_ok,why=group_gate(a,b);ok=group_ok and high_depth[pid]
                    if not high_depth[pid]:why='outside_high_depth_baseline_cohort'+(';' + why if why else '')
                    base=dict(patient_id=pid,tissue=tissue,threshold=threshold,context=context,group_a='observed',group_b='unobserved',
                              eligible=ok,high_depth_primary=high_depth[pid],exclusion=why,**counts)
                    elig.append(base)
                    for weight in WEIGHTS:
                        if context=='states':
                            av=state_fractions(a,weight);bv=state_fractions(b,weight)
                            for state in STATES:states.append(base|dict(state=state,weighting=weight,group_a_fraction=av[state],group_b_fraction=bv[state],difference=av[state]-bv[state]))
                        else:
                            av=program_means(a,weight);bv=program_means(b,weight)
                            for p in PROGRAMS:programs.append(base|dict(program=p,weighting=weight,group_a_mean=av[p],group_b_mean=bv[p],difference=av[p]-bv[p]))
    s=pd.DataFrame(states);p=pd.DataFrame(programs)
    st=test_table(s,['state','tissue','threshold','weighting'],lambda k:f"own_NTVJ_ge{k['threshold']}_states_{k['weighting']}_BH64",lambda k:64)
    pt=test_table(p,['program','tissue','threshold','weighting'],lambda k:f"own_NTVJ_ge{k['threshold']}_programs_{k['weighting']}_BH12",lambda k:12)
    return pd.DataFrame(elig),s,st,p,pt


def bridge(cells,patients,high_depth):
    coverage=[];state_rows=[];programs=[];elig=[];members=[];incidence=[]
    for threshold in (1,2):
        label='ownblood_AA_VJ_ge'+str(threshold)
        for pid in patients:
            for tissue in TISSUES:
                z=cells[(cells.patient_id==pid)&(cells.Tissue==tissue)].copy()
                z['category']=np.select([z[label]&z.match_AA_VJ_union_ge2,z[label]&~z.match_AA_VJ_union_ge2,~z[label]&z.match_AA_VJ_union_ge2],CATEGORIES[:3],default='neither')
                for clone,part in z.groupby('beta_clone',observed=True):assert part.category.nunique()==1,('bridge_clone_category_inconsistent',pid,tissue,clone)
                n=len(z);clones=z.beta_clone.nunique()
                for category in CATEGORIES:
                    a=z[z.category==category];k=a.beta_clone.nunique()
                    coverage.append(dict(patient_id=pid,tissue=tissue,own_threshold=threshold,category=category,n_cells=n,n_clones=clones,
                                         category_cells=len(a),category_clones=k,cell_fraction=len(a)/n if n else np.nan,clone_fraction=k/clones if clones else np.nan,
                                         high_depth_primary=high_depth[pid],definition='AA_VJ',bulk_threshold=2))
                    for weight in WEIGHTS:
                        frac=state_fractions(a,weight)
                        for s in STATES:state_rows.append(dict(patient_id=pid,tissue=tissue,own_threshold=threshold,category=category,state=s,
                                                              weighting=weight,n_category_cells=len(a),n_category_clones=k,fraction=frac[s],high_depth_primary=high_depth[pid]))
                cd8=z[z.is_cd8 & z.positive_expression_qc];a=cd8[cd8.category=='own_only'];b=cd8[cd8.category=='bulk_only']
                counts,group_ok,why=group_gate(a,b);ok=group_ok and high_depth[pid]
                if not high_depth[pid]:why='outside_high_depth_baseline_cohort'+(';' + why if why else '')
                base=dict(patient_id=pid,tissue=tissue,own_threshold=threshold,group_a='own_only',group_b='bulk_only',eligible=ok,
                          high_depth_primary=high_depth[pid],exclusion=why,**counts);elig.append(base)
                for weight in WEIGHTS:
                    av=program_means(a,weight);bv=program_means(b,weight)
                    for p in PROGRAMS:programs.append(base|dict(program=p,weighting=weight,group_a_mean=av[p],group_b_mean=bv[p],difference=av[p]-bv[p]))
                members.append(z[['cell_name','patient_id','Tissue','beta_clone','key_AA_VJ',label,'match_AA_VJ_union_ge2','category']].assign(own_threshold=threshold,high_depth_primary=high_depth[pid]))
    p=pd.DataFrame(programs);formal=p[p.weighting=='clone_equal']
    tests=test_table(formal,['program','tissue','own_threshold','weighting'],lambda k:f"bridge_AAVJ_ge{k['own_threshold']}_clone_equal_BH12",lambda k:12)
    descriptions=test_table(p[p.weighting=='cell_weighted'],['program','tissue','own_threshold','weighting'],lambda k:f"bridge_AAVJ_ge{k['own_threshold']}_cell_weighted_description",lambda k:12,lambda k:False)
    for pid in patients:
        for tissue in TISSUES:
            z=cells[(cells.patient_id==pid)&(cells.Tissue==tissue)]
            for cohort,den in [('home',137),('external',202),('union',339)]:
                column='bulk_'+cohort+'_carriers_ge2'
                for metric in ('cell','clone'):
                    part=z if metric=='cell' else z.drop_duplicates('beta_clone')
                    c=part[column].astype(int);bins=np.select([c.eq(0),c.eq(1),c.between(2,4)],['0','1','2-4'],default='>=5')
                    for band in ('0','1','2-4','>=5'):
                        mask=bins==band
                        incidence.append(dict(patient_id=pid,tissue=tissue,catalog_cohort=cohort,catalog_denominator=den,metric=metric,incidence_bin=band,
                                              n_reference_units=len(part),n_units=int(mask.sum()),fraction=float(mask.mean()) if len(part) else np.nan,
                                              mean_key_carrier_proportion=float((c[mask]/den).mean()) if mask.any() else np.nan,high_depth_primary=high_depth[pid]))
    return pd.DataFrame(coverage),pd.DataFrame(state_rows),pd.DataFrame(elig),p,tests,descriptions,pd.concat(members,ignore_index=True),pd.DataFrame(incidence)


def composition(cells,expanded,patients,high_depth):
    contrasts=[];elig=[];weights=[];lookup={}
    for context in ('expansion','own_NTVJ_ge1','bridge_AAVJ_ge1'):
        for pid in patients:
            for tissue in TISSUES:
                z=expanded[(expanded.patient_id==pid)&(expanded.Tissue==tissue)]
                if context=='expansion':a=z[z.expansion_group=='expanded'];b=z[z.expansion_group=='singleton'];an,bn='expanded','singleton'
                elif context=='own_NTVJ_ge1':a=z[z.ownblood_NT_VJ_ge1];b=z[~z.ownblood_NT_VJ_ge1];an,bn='observed','unobserved'
                else:
                    own=z.ownblood_AA_VJ_ge1;bulk=z.match_AA_VJ_union_ge2
                    a=z[own & ~bulk];b=z[~own & bulk];an,bn='own_only','bulk_only'
                counts,group_ok,why=group_gate(a,b);supported=[];state_lookup={}
                for state in CD8_STATES:
                    sa=a[a.SubCellType==state];sb=b[b.SubCellType==state]
                    supported_state=len(sa)>=5 and len(sb)>=5 and sa.beta_clone.nunique()>=2 and sb.beta_clone.nunique()>=2
                    if supported_state:supported.append(state)
                    state_lookup[state]=(sa,sb,supported_state)
                ra=sum(len(state_lookup[s][0]) for s in supported);rb=sum(len(state_lookup[s][1]) for s in supported)
                retained_a=ra/len(a) if len(a) else 0.;retained_b=rb/len(b) if len(b) else 0.
                state_ok=len(supported)>=2 and retained_a>=.5 and retained_b>=.5
                cohort_ok=context=='expansion' or high_depth[pid];ok=group_ok and state_ok and cohort_ok
                reasons=[]
                if not group_ok:reasons.append(why)
                if len(supported)<2:reasons.append('fewer_than_2_common_states')
                if min(retained_a,retained_b)<.5:reasons.append('less_than_half_each_group_cells_retained')
                if not cohort_ok:reasons.append('outside_high_depth_baseline_cohort')
                base=dict(context=context,patient_id=pid,tissue=tissue,weighting='clone_equal',group_a=an,group_b=bn,
                          eligible=ok,exclusion=';'.join(reasons),n_supported_states=len(supported),retained_a_fraction=retained_a,
                          retained_b_fraction=retained_b,high_depth_primary=high_depth[pid],**counts)
                elig.append(base);wa={s:(len(state_lookup[s][0])+len(state_lookup[s][1]))/(ra+rb) if s in supported else 0. for s in CD8_STATES}
                for s in CD8_STATES:
                    sa,sb,good=state_lookup[s]
                    weights.append(dict(context=context,patient_id=pid,tissue=tissue,state=s,state_supported=good,pooled_state_cell_weight=wa[s],
                                        group_a_cells=len(sa),group_b_cells=len(sb),group_a_clones=sa.beta_clone.nunique(),group_b_clones=sb.beta_clone.nunique(),
                                        patient_contrast_eligible=ok))
                am={p:0. for p in PROGRAMS};bm={p:0. for p in PROGRAMS}
                for s in supported:
                    av=program_means(state_lookup[s][0],'clone_equal');bv=program_means(state_lookup[s][1],'clone_equal')
                    for p in PROGRAMS:am[p]+=wa[s]*av[p];bm[p]+=wa[s]*bv[p]
                for p in PROGRAMS:
                    row=base|dict(program=p,group_a_mean=am[p] if supported else np.nan,group_b_mean=bm[p] if supported else np.nan,
                                  difference=am[p]-bm[p] if supported else np.nan)
                    contrasts.append(row);lookup[context,pid,tissue,p]=row
    table=pd.DataFrame(contrasts)
    tests=test_table(table,['context','tissue','program','weighting'],lambda k:f"composition_{k['context']}_{k['tissue']}_BH6",lambda k:6)
    interaction=[]
    for pid in patients:
        for p in PROGRAMS:
            t=lookup['expansion',pid,'Tumor',p];n=lookup['expansion',pid,'Normal',p]
            interaction.append(dict(patient_id=pid,program=p,weighting='clone_equal',tumor_difference=t['difference'],normal_difference=n['difference'],
                                    difference=t['difference']-n['difference'],eligible=t['eligible'] and n['eligible'],
                                    exclusion='' if t['eligible'] and n['eligible'] else 'both_tissues_require_common_state_eligibility'))
    inter=pd.DataFrame(interaction)
    itests=test_table(inter,['program','weighting'],lambda k:'composition_expansion_interaction_BH6',lambda k:6)
    return table,tests,pd.DataFrame(elig),pd.DataFrame(weights),inter,itests


def clone_size_descriptions(x,patients,high_depth):
    rows=[]
    for pid in patients:
        for tissue in TISSUES:
            z=x[(x.patient_id==pid)&(x.Tissue==tissue)].copy()
            z['own_group']=np.where(z.ownblood_NT_VJ_ge1,'observed','unobserved')
            z['bridge_group']=np.select([z.ownblood_AA_VJ_ge1&z.match_AA_VJ_union_ge2,z.ownblood_AA_VJ_ge1&~z.match_AA_VJ_union_ge2,~z.ownblood_AA_VJ_ge1&z.match_AA_VJ_union_ge2],CATEGORIES[:3],default='neither')
            for context,column,groups in [('expansion','expansion_group',('expanded','singleton')),('own_NTVJ_ge1','own_group',('observed','unobserved')),('bridge_AAVJ_ge1','bridge_group',CATEGORIES)]:
                for group in groups:
                    a=z[z[column]==group];clones=a.drop_duplicates('beta_clone')
                    for band in ('1','2-4','>=5'):
                        rows.append(dict(context=context,patient_id=pid,tissue=tissue,group=group,clone_size_bin=band,
                                         group_cells=len(a),group_clones=len(clones),bin_cells=int(a.clone_size_bin.eq(band).sum()),
                                         bin_clones=int(clones.clone_size_bin.eq(band).sum()),
                                         cell_fraction=float(a.clone_size_bin.eq(band).mean()) if len(a) else np.nan,
                                         clone_fraction=float(clones.clone_size_bin.eq(band).mean()) if len(clones) else np.nan,
                                         high_depth_primary=high_depth[pid]))
    return pd.DataFrame(rows)


def load_inputs(root,cohort_receipt):
    assert sha(root/'PROTOCOL_v6.md')==PARENT_SHA,'central_protocol_changed'
    assert sha(root/'phenotypes/PROTOCOL.md')==ADDENDUM_SHA,'lane_protocol_changed'
    freeze=json.loads((root/'phenotypes/PROTOCOL_FREEZE.json').read_text())
    assert freeze['parent_protocol_sha256']==PARENT_SHA and freeze['addendum_sha256']==ADDENDUM_SHA
    assert cohort_receipt is not None,'explicit_final_cohort_receipt_required'
    receipt=json.loads(cohort_receipt.read_text())
    assert str(receipt.get('status','')).lower() in ('pass','passed','complete','completed'),'cohort_receipt_not_pass'
    assert receipt['central_protocol_sha256']==PARENT_SHA,'cohort_protocol_not_bound'
    producer_verifier=root/'cohort/verification_receipt.json'
    vr=json.loads(producer_verifier.read_text())
    assert str(vr.get('status','')).lower() in ('pass','passed'),'cohort_independent_verifier_not_pass'
    assert vr.get('checks') and all(c.get('passed',c.get('pass',False)) or str(c.get('status','')).upper()=='PASS' for c in vr['checks']),'cohort_verifier_has_failed_check'
    for item in receipt['outputs']:
        target=root/item['path']
        assert target.exists() and target.stat().st_size==item['bytes'] and sha(target)==item['sha256'],('producer_output_binding_failed',item['path'])
    cp=root/'cohort/baseline_cells.parquet';rp=root/'cohort/baseline_patient_registry.tsv'
    cells=pd.read_parquet(cp);reg=pd.read_csv(rp,sep='\t')
    assert reg.patient_id.is_unique and set(reg.patient_id)==set(PATIENTS),'fixed19_registry_mismatch'
    assert cells.cell_name.is_unique,'cell_identity_not_unique'
    assert 'high_depth_primary' in reg
    high_depth=dict(zip(reg.patient_id,boolean(reg.high_depth_primary)))
    cells=cells[cells.patient_id.isin(PATIENTS)&cells.Tissue.isin(TISSUES)].copy()
    required=['cell_name','patient_id','Tissue','SubCellType','beta_clone','key_AA_VJ','is_cd8','positive_expression_qc',
              'ownblood_NT_VJ_ge1','ownblood_NT_VJ_ge2','ownblood_AA_VJ_ge1','ownblood_AA_VJ_ge2','match_AA_VJ_union_ge2',
              'bulk_home_carriers_ge2','bulk_external_carriers_ge2']+['program_'+p for p in PROGRAMS]
    assert set(required)<=set(cells.columns),('missing_columns',set(required)-set(cells.columns))
    assert set(cells.SubCellType.dropna())<=set(STATES),'unfrozen_state'
    assert cells.beta_clone.notna().all() and cells.key_AA_VJ.notna().all(),'missing_receptor_identity'
    for c in required:
        if c.startswith(('ownblood_','match_')) or c in ('is_cd8','positive_expression_qc'):cells[c]=boolean(cells[c])
    assert cells.ownblood_NT_VJ_ge2.le(cells.ownblood_NT_VJ_ge1).all()
    assert cells.ownblood_AA_VJ_ge2.le(cells.ownblood_AA_VJ_ge1).all()
    assert cells.ownblood_NT_VJ_ge1.le(cells.ownblood_AA_VJ_ge1).all()
    assert cells.ownblood_NT_VJ_ge2.le(cells.ownblood_AA_VJ_ge2).all()
    assert cells.is_cd8.eq(cells.SubCellType.isin(CD8_STATES)).all(),'conventional_CD8_annotation_changed'
    assert cells.groupby(['patient_id','Tissue','beta_clone'],observed=True).key_AA_VJ.nunique().eq(1).all()
    for c in ('ownblood_NT_VJ_ge1','ownblood_NT_VJ_ge2','ownblood_AA_VJ_ge1','ownblood_AA_VJ_ge2','match_AA_VJ_union_ge2'):
        assert cells.groupby(['patient_id','Tissue','beta_clone'],observed=True)[c].nunique().le(1).all(),('clone_flag_inconsistency',c)
    cells['bulk_union_carriers_ge2']=cells.bulk_home_carriers_ge2+cells.bulk_external_carriers_ge2
    assert cells.bulk_home_carriers_ge2.between(0,137).all() and cells.bulk_external_carriers_ge2.between(0,202).all()
    assert cells.match_AA_VJ_union_ge2.eq(cells.bulk_union_carriers_ge2.gt(0)).all(),'bulk_incidence_label_mismatch'
    assert len(cells)==53886,('baseline_tissue_count_changed',len(cells))
    original_cd8=root.parent/'single_cell/cd8_program_cell_inventory.parquet'
    cached=pd.read_parquet(original_cd8)
    current=cells[cells.is_cd8 & cells.positive_expression_qc].set_index('cell_name')
    old=cached.set_index('cell_name')
    assert set(current.index)==set(old.index),'legacy_expansion_CD8_population_changed'
    columns=['program_'+p for p in PROGRAMS]
    assert np.array_equal(current.loc[old.index,columns].to_numpy(),old[columns].to_numpy()),'frozen_scores_changed'
    return cells,reg,high_depth,[cp,rp,cohort_receipt,producer_verifier,original_cd8,root/'PROTOCOL_v6.md',root/'phenotypes/PROTOCOL.md',root/'phenotypes/PROTOCOL_FREEZE.json',Path(__file__).resolve()]


def run(root,cohort_receipt):
    cells,reg,high_depth,inputs=load_inputs(root,cohort_receipt)
    p=root/'phenotypes';b=root/'bridge';paths=[]
    def write(folder,name,rows):
        path=folder/name;save(path,rows);paths.append(path)
    write(p,'input_manifest.tsv',[dict(path=str(q),bytes=q.stat().st_size,sha256=sha(q)) for q in inputs])
    full_baseline=pd.read_parquet(root/'cohort/baseline_cells.parquet')
    blood=full_baseline[full_baseline.Tissue.eq('Blood')].copy()
    assert len(blood)==int(reg.blood_beta_cells.sum()),'full_autologous_catalog_count_mismatch'
    source_catalog=[]
    for definition,key in [('NT_VJ','key_NT_VJ'),('AA_VJ','key_AA_VJ')]:
        counts=blood.groupby(['patient_id',key],observed=True).cell_name.size().reset_index(name='accepted_blood_cells')
        counts=counts.rename(columns={key:'match_key'});counts['definition']=definition;counts['ge1']=True;counts['ge2']=counts.accepted_blood_cells.ge(2)
        source_catalog.append(counts)
        lookup={(r.patient_id,r.match_key):r.accepted_blood_cells for r in counts.itertuples(index=False)}
        support=np.array([lookup.get((pid,k),0) for pid,k in zip(cells.patient_id,cells[key])])
        for threshold in (1,2):assert np.array_equal(support>=threshold,cells['ownblood_'+definition+'_ge'+str(threshold)].to_numpy()),('autologous_member_label_mismatch',definition,threshold)
    write(p,'ownblood_source_catalog.tsv',pd.concat(source_catalog,ignore_index=True))
    member_path=p/'ownblood_source_members.parquet'
    blood[['cell_name','patient_id','Ident','Tissue','key_NT_VJ','key_AA_VJ','beta_clone']].to_parquet(member_path,index=False);paths.append(member_path)
    del full_baseline,blood
    x,em,ee,ec,et=expansion(cells,PATIENTS)
    write(p,'expansion_patient_metrics.tsv',em);write(p,'expansion_eligibility.tsv',ee)
    write(p,'expansion_patient_contrasts.tsv',ec);write(p,'expansion_tests.tsv',et)
    oe,sc,st,pc,pt=ownblood(cells,PATIENTS,high_depth)
    write(p,'ownblood_eligibility.tsv',oe);write(p,'ownblood_state_patient_contrasts.tsv',sc);write(p,'ownblood_state_tests.tsv',st)
    write(p,'ownblood_program_patient_contrasts.tsv',pc);write(p,'ownblood_program_tests.tsv',pt)
    cov,bs,be,bpc,bpt,bd,members,inc=bridge(cells,PATIENTS,high_depth)
    write(b,'coverage.tsv',cov);write(b,'state_composition.tsv',bs);write(b,'program_eligibility.tsv',be)
    write(b,'program_patient_contrasts.tsv',bpc);write(b,'program_tests.tsv',bpt);write(b,'program_cell_weighted_descriptions.tsv',bd)
    write(b,'bulk_support_frequency.tsv',inc)
    member_path=b/'cell_membership.parquet';members.to_parquet(member_path,index=False);paths.append(member_path)
    cx,ct,ce,cw,ci,cit=composition(cells,x,PATIENTS,high_depth)
    write(p,'composition_patient_contrasts.tsv',cx);write(p,'composition_tests.tsv',ct);write(p,'composition_eligibility.tsv',ce)
    write(p,'composition_weights.tsv',cw);write(p,'composition_expansion_interaction_patient_contrasts.tsv',ci)
    write(p,'composition_expansion_interaction_tests.tsv',cit);write(p,'clone_size_descriptions.tsv',clone_size_descriptions(x,PATIENTS,high_depth))
    own_cols=['cell_name','patient_id','Tissue','beta_clone','key_AA_VJ','ownblood_NT_VJ_ge1','ownblood_NT_VJ_ge2','ownblood_AA_VJ_ge1','ownblood_AA_VJ_ge2','match_AA_VJ_union_ge2']
    q=p/'cell_membership.parquet';cells[own_cols].assign(high_depth_primary=cells.patient_id.map(high_depth)).to_parquet(q,index=False);paths.append(q)
    q=p/'expansion_cell_membership.parquet';x[['cell_name','patient_id','Tissue','beta_clone','SubCellType','within_cd8_clone_cells','expansion_group','clone_size_bin']].to_parquet(q,index=False);paths.append(q)
    write(b,'catalog_source_links.tsv',[dict(source_role='baseline_autologous_flags_and_independent_bulk_flags',path=str(root/'cohort/baseline_cells.parquet'),sha256=sha(root/'cohort/baseline_cells.parquet')),
                                        dict(source_role='original_v5_tissue_key_bulk_catalog',path=str(root.parent/'revision_v5/bulk_sc_match/server_data/key_catalog.parquet'),sha256=sha(root.parent/'revision_v5/bulk_sc_match/server_data/key_catalog.parquet'))])
    report=['# v6 phenotype and independent-reference bridge results','',f'Fixed19 tissue patients; {len(cells):,} accepted-beta RNA-T cells; {sum(high_depth.values())} high-depth baseline blood patients. All patient rows retained; exclusions cannot enlarge n.','',
            'Expansion follows the original conventional-CD8 patient/tissue exact NT|V|J >=2-cell definition. New four-group 20-cell/5-clone eligibility does not rewrite the prior cell-only18/19 results.','',
            'Autologous blood labels use full baseline blood, exact NT|V|J >=1 primary and >=2 sensitivity. Bridge uses AA|V|J on both sides; own single-cell counts and independent bulk counts are distinct measurement units. Neither-observed is not absence.','']
    summaries={}
    for name,table in [('expansion',et),('own_states',st),('own_programs',pt),('bridge_programs',bpt),('composition',ct),('composition_expansion_interaction',cit)]:
        summaries[name]=table.groupby('family',sort=False).apply(lambda z:dict(rows=len(z),estimable=int(z.P.notna().sum()),BH_supported=int(z.q.lt(.05).sum()),minimum_q=float(z.q.min()) if z.q.notna().any() else None),include_groups=False).to_dict() if tuple(map(int,pd.__version__.split('.')[:2]))>=(2,2) else {f:dict(rows=len(z),estimable=int(z.P.notna().sum()),BH_supported=int(z.q.lt(.05).sum()),minimum_q=float(z.q.min()) if z.q.notna().any() else None) for f,z in table.groupby('family',sort=False)}
        report.extend(['## '+name,table.to_markdown(index=False,floatfmt='.6g'),''])
    report.extend(['## Interpretation limits','States/programs are captured transcriptional context. No full-alpha-beta identity, antigen function, tissue trafficking, cancer specificity or causal explanation of nodal models is inferred. Conditional common-state standardization does not establish causal adjustment. Legacy v5 cross-individual results are preserved without rerun.',''])
    q=p/'ANALYSIS_REPORT.md';q.write_text('\n'.join(report));paths.append(q)
    manifest=[dict(path=str(q.relative_to(root)),bytes=q.stat().st_size,sha256=sha(q)) for q in paths]
    save(p/'output_manifest.tsv',manifest)
    receipt=dict(status='passed',completed_utc=datetime.now(timezone.utc).isoformat(),protocol_sha256=PARENT_SHA,addendum_sha256=ADDENDUM_SHA,
                 entrypoint_sha256=sha(Path(__file__)),input_manifest_sha256=sha(p/'input_manifest.tsv'),output_manifest_sha256=sha(p/'output_manifest.tsv'),
                 baseline_patients=len(PATIENTS),baseline_tissue_cells=len(cells),high_depth_patients=sum(high_depth.values()),
                 expansion_CD8_cells=len(x),bootstrap_repetitions=BOOTSTRAPS,seed=SEED,patient_inference_unit=True,families=summaries,outputs=manifest)
    dump(p/'analysis_receipt.json',receipt);print(json.dumps({k:receipt[k] for k in ('status','baseline_patients','baseline_tissue_cells','high_depth_patients','expansion_CD8_cells','families')},indent=2),flush=True)
    verify(root)


def verify(root):
    from statsmodels.stats.multitest import multipletests
    out=root/'phenotypes';receipt=json.loads((out/'analysis_receipt.json').read_text());checks=[]
    def check(name,ok):
        checks.append(dict(name=name,passed=bool(ok)))
        assert ok,name
    check('central_protocol_binding',sha(root/'PROTOCOL_v6.md')==PARENT_SHA)
    check('addendum_protocol_binding',sha(out/'PROTOCOL.md')==ADDENDUM_SHA)
    check('executed_script_binding',sha(Path(__file__))==receipt['entrypoint_sha256'])
    check('output_manifest_binding',sha(out/'output_manifest.tsv')==receipt['output_manifest_sha256'])
    for row in pd.read_csv(out/'input_manifest.tsv',sep='\t').itertuples(index=False):
        check('input_'+Path(row.path).name,sha(row.path)==row.sha256 and Path(row.path).stat().st_size==row.bytes)
    for row in pd.read_csv(out/'output_manifest.tsv',sep='\t').itertuples(index=False):
        q=root/row.path;check('output_'+row.path,sha(q)==row.sha256 and q.stat().st_size==row.bytes)
    coverage=pd.read_csv(root/'bridge/coverage.tsv',sep='\t')
    check('four_category_rows',len(coverage)==19*2*2*4)
    for keys,z in coverage.groupby(['patient_id','tissue','own_threshold']):
        check('bridge_classes_'+str(keys),set(z.category)==set(CATEGORIES))
        check('bridge_cell_closure_'+str(keys),np.isclose(z.cell_fraction.sum(),1) and z.category_cells.sum()==z.n_cells.iloc[0])
        check('bridge_clone_closure_'+str(keys),np.isclose(z.clone_fraction.sum(),1) and z.category_clones.sum()==z.n_clones.iloc[0])
    specs=[('phenotypes/expansion_tests.tsv','phenotypes/expansion_patient_contrasts.tsv',['program','weighting'],12),
           ('phenotypes/ownblood_state_tests.tsv','phenotypes/ownblood_state_patient_contrasts.tsv',['state','tissue','threshold','weighting'],256),
           ('phenotypes/ownblood_program_tests.tsv','phenotypes/ownblood_program_patient_contrasts.tsv',['program','tissue','threshold','weighting'],48),
           ('bridge/program_tests.tsv','bridge/program_patient_contrasts.tsv',['program','tissue','own_threshold','weighting'],24),
           ('phenotypes/composition_tests.tsv','phenotypes/composition_patient_contrasts.tsv',['context','tissue','program','weighting'],36),
           ('phenotypes/composition_expansion_interaction_tests.tsv','phenotypes/composition_expansion_interaction_patient_contrasts.tsv',['program','weighting'],6)]
    for testpath,patientpath,cols,expected in specs:
        t=pd.read_csv(root/testpath,sep='\t');d=pd.read_csv(root/patientpath,sep='\t');d.eligible=boolean(d.eligible)
        check('complete_rows_'+testpath,len(t)==expected)
        for family,z in t.groupby('family'):
            check('complete_family_'+family,len(z)==z.family_size.iloc[0])
            q=multipletests(z.P.fillna(1).to_numpy(),method='fdr_bh')[1]
            check('independent_BH_'+family,np.allclose(q,z.q_BH_placeholder) and np.allclose(q[z.P.notna()],z.loc[z.P.notna(),'q']))
            check('sparse_reported_NA_'+family,z.loc[z.P.isna(),'q'].isna().all())
        for row in t.to_dict('records'):
            part=d
            for c in cols:part=part[part[c]==row[c]]
            check('full19_'+testpath+str(tuple(row[c] for c in cols)),len(part)==19 and part.patient_id.nunique()==19)
            ok=part[part.eligible & np.isfinite(part.difference)];vals=ok.difference.to_numpy()
            check('patient_n_'+testpath+str(tuple(row[c] for c in cols)),len(ok)==row['n'])
            if len(ok):check('patient_mean_'+testpath+str(tuple(row[c] for c in cols)),np.isclose(vals.mean(),row['mean_difference'],rtol=1e-10,atol=1e-12))
            if len(ok)>=8:
                with warnings.catch_warnings():
                    warnings.simplefilter('ignore',UserWarning)
                    pv=1. if np.all(vals==0) else wilcoxon(vals,zero_method='wilcox',method='approx',correction=False).pvalue
                check('patient_Wilcoxon_'+testpath+str(tuple(row[c] for c in cols)),np.isclose(pv,row['P'],rtol=1e-9,atol=1e-12))
            else:check('minimum8_'+testpath+str(tuple(row[c] for c in cols)),pd.isna(row['P']) and pd.isna(row['q']))
    exp=pd.read_parquet(out/'expansion_cell_membership.parquet')
    check('expansion_size_is_grouped_CD8_capture',exp.groupby(['patient_id','Tissue','beta_clone'],observed=True).cell_name.transform('size').eq(exp.within_cd8_clone_cells).all())
    check('expansion_threshold2',exp.expansion_group.eq('expanded').eq(exp.within_cd8_clone_cells.ge(2)).all())
    cw=pd.read_csv(out/'composition_weights.tsv',sep='\t');ce=pd.read_csv(out/'composition_eligibility.tsv',sep='\t')
    for keys,z in cw.groupby(['context','patient_id','tissue']):
        supported=boolean(z.state_supported)
        check('common_states_gate_'+str(keys),(supported==((z.group_a_cells>=5)&(z.group_b_cells>=5)&(z.group_a_clones>=2)&(z.group_b_clones>=2))).all())
        check('common_weights_'+str(keys),np.isclose(z.pooled_state_cell_weight.sum(),1 if supported.any() else 0))
    good=ce[boolean(ce.eligible)]
    check('composition_retained_gate',(good.n_supported_states>=2).all() and (good.retained_a_fraction>=.5).all() and (good.retained_b_fraction>=.5).all())
    members=pd.read_parquet(root/'bridge/cell_membership.parquet')
    for threshold,z in members.groupby('own_threshold'):
        own=z['ownblood_AA_VJ_ge'+str(threshold)].fillna(False);bulk=z.match_AA_VJ_union_ge2
        expected=np.select([own&bulk,own&~bulk,~own&bulk],CATEGORIES[:3],default='neither')
        check('bridge_identity_partition_ge'+str(threshold),np.array_equal(expected,z.category.to_numpy()))
    result=dict(status='passed',verified_utc=datetime.now(timezone.utc).isoformat(),checks=checks,n_checks=len(checks),
                analysis_receipt_sha256=sha(out/'analysis_receipt.json'),output_manifest_sha256=sha(out/'output_manifest.tsv'),
                entrypoint_sha256=sha(Path(__file__)),protocol_sha256=PARENT_SHA,addendum_sha256=ADDENDUM_SHA)
    dump(out/'verification_receipt.json',result);print('V6 PHENOTYPE/BRIDGE VERIFY PASS',len(checks),flush=True)
    return result


def self_test(root):
    checks=[]
    def check(name,ok):assert ok,name;checks.append(name)
    s=summarize([1,2,3],['a','b','c'],'sparse');check('n3_inference_unestimable',np.isnan(s['P']) and s['P_BH_placeholder']==1)
    s=summarize(np.zeros(8),list('abcdefgh'),'zero');check('allzero8_P1',s['P']==1)
    f=adjust(pd.DataFrame([dict(family='f',family_size=2,P=.01,P_BH_placeholder=.01),dict(family='f',family_size=2,P=np.nan,P_BH_placeholder=1.)]))
    check('missing_family_kept',np.isclose(f.q.iloc[0],.02) and np.isnan(f.q.iloc[1]))
    rows=[]
    for pidindex,pid in enumerate(PATIENTS):
        for tissue in TISSUES:
            for clone in range(80):
                size=3 if clone<40 else 1
                for j in range(size):
                    v=dict(cell_name=f'{pid}_{tissue}_{clone}_{j}',patient_id=pid,Tissue=tissue,beta_clone=f'n{pid}_{tissue}_{clone}',
                           key_AA_VJ=f'aa{clone}',SubCellType=CD8_STATES[clone%2],is_cd8=True,positive_expression_qc=True,
                           ownblood_NT_VJ_ge1=clone%2==0,ownblood_NT_VJ_ge2=clone%4==0,ownblood_AA_VJ_ge1=clone%2==0,
                           ownblood_AA_VJ_ge2=clone%4==0,match_AA_VJ_union_ge2=clone%3==0,
                           bulk_home_carriers_ge2=1 if clone%3==0 else 0,bulk_external_carriers_ge2=0,bulk_union_carriers_ge2=1 if clone%3==0 else 0)
                    for k,p in enumerate(PROGRAMS):v['program_'+p]=k/10+clone/100+(tissue=='Tumor')*.1+pidindex*.001
                    rows.append(v)
    cells=pd.DataFrame(rows);high=dict.fromkeys(PATIENTS,True)
    x,em,ee,ec,et=expansion(cells,PATIENTS)
    check('expanded_exact_threshold',x.expansion_group.eq('expanded').eq(x.within_cd8_clone_cells.ge(2)).all())
    check('expansion_complete12',len(et)==12 and len(ec)==19*6*2)
    oe,sc,st,pc,pt=ownblood(cells,PATIENTS,high)
    check('own_complete_families',len(st)==256 and len(pt)==48 and len(sc)==19*2*32*2*2)
    cov,bs,be,bpc,bpt,bd,members,inc=bridge(cells,PATIENTS,high)
    check('bridge_closed',np.allclose(cov.groupby(['patient_id','tissue','own_threshold']).cell_fraction.sum(),1) and np.allclose(cov.groupby(['patient_id','tissue','own_threshold']).clone_fraction.sum(),1))
    check('bridge_complete24_and_descriptive_only',len(bpt)==24 and len(bd)==24 and bd.P.isna().all())
    high[private_value('v6_phenotypes_bridge_value_509_9')]=False
    oe2,*_=ownblood(cells,PATIENTS,high);check('lowblood_patient_retained_not_eligible',len(oe2[oe2.patient_id==private_value('v6_phenotypes_bridge_value_510_112')])==8 and not oe2[oe2.patient_id==private_value('v6_phenotypes_bridge_value_510_151')].eligible.any())
    cx,ct,ce,cw,ci,cit=composition(cells,x,PATIENTS,high)
    check('composition_complete42',len(ct)==36 and len(cit)==6)
    check('no_adaptive_common_state_relaxation',not ce[ce.context=='own_NTVJ_ge1'].eligible.any())
    a=pd.DataFrame(dict(beta_clone=['x']*30+['y']*10,SubCellType=[CD8_STATES[0]]*40,**{'program_'+p:[0.]*30+[1.]*10 for p in PROGRAMS}))
    check('cloneequal_differs_from_cellweight',np.isclose(program_means(a,'clone_equal')['cytotoxic'],.5) and np.isclose(program_means(a,'cell_weighted')['cytotoxic'],.25))
    b=pd.DataFrame(dict(beta_clone=['x']*20+['y']*20));_,good,_=group_gate(a,b);check('20cells_cannot_replace5clones',not good)
    result=dict(status='passed',checks=checks,n_checks=len(checks),entrypoint_sha256=sha(Path(__file__)),seed=SEED,
                synthetic_only=True,no_actual_inference=True,completed_utc=datetime.now(timezone.utc).isoformat())
    dump(root/'phenotypes/self_test_receipt.json',result);print('SELFTEST PASS',len(checks),flush=True)
    return result


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,default=DEFAULT_ROOT)
    group=parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--run',action='store_true');group.add_argument('--verify-only',action='store_true');group.add_argument('--self-test',action='store_true')
    parser.add_argument('--cohort-receipt',type=Path)
    args=parser.parse_args()
    if args.self_test:self_test(args.root)
    elif args.verify_only:verify(args.root)
    else:run(args.root,args.cohort_receipt)

if __name__=='__main__':main()
