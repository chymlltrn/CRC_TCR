#!/usr/bin/env python3
"""Reaggregate compact original clone exports; never mutate original results.

NT-only identities collapse V/J assignments, while NT/V/J identities retain them.
Coding RNA displays are T-to-U transformations, not recovered original RNA reads.
"""
from release_runtime import private_literal as _private_literal, release_path as _release_path
from pathlib import Path
import hashlib, json, re
import numpy as np
import pandas as pd
from Bio.Seq import Seq

B = Path(_release_path('path_1757610fc64073d4'))
R = B / 'revision_v17_figure_qpcr_revision'
O = R / 'analysis/nucleotide_variants'
O.mkdir(parents=True, exist_ok=True)
AA = {'Q1':_private_literal('private_45bbaa37a9607bb7'),'Q2':_private_literal('private_b3445b7fc78bf2f0')}
NOM = {'Q1':_private_literal('private_c287d28aaf0e9d7c'),
       'Q2':_private_literal('private_230496dad7ebf306')}
FWD = _private_literal('private_0205e4363adb0bf7')
PROBE = _private_literal('private_f337b5ef8e62afd0')
REV = _private_literal('private_4327e40ea7f8c48d')
PATHS = {
 'selected_clone_rows':B/'revision_v5/legacy/matching_clone_rows.tsv',
 'selected_patient_roster':B/'revision_v5/legacy/patient_Q1_Q2.tsv',
 'frozen_v16_NT_VJ_summary':B/'revision_v16_core_narrative/manuscript/SourceData/candidates/observed_bulk_NT_VJ_variants.tsv',
 'unified_candidate_units':B/'revision_v16_core_narrative/manuscript/SourceData/candidates/candidate_units.tsv',
 'public_Q1_original_rows':B/'q1/public_query/public_exact_Q1_hit_rows.tsv',
 'public_Q2_original_rows':B/'revision_v5/legacy/public_q2/exact_Q2_hit_rows.tsv',
 'tissue_reference_original_rows':B/'revision_v5/legacy/tissue_reference/exact_candidate_hit_rows.tsv',
 'executed_method_extraction':R/'analysis/qpcr/primary_methodology/full_content.md',
 'planned_protocol_extraction':R/'analysis/qpcr/planned_protocol/full_content.md',
 'executed_method_document':Path(_release_path('path_cb804fa1168cd06e')),
 'prior_raw_audit_receipt':B/'revision_v16_core_narrative/manuscript/SourceData/q1_technical_audit/verification_receipt.json',
}
checks=[]
def check(name, condition, evidence):
    assert bool(condition), name + ': ' + str(evidence)
    checks.append({'check':name,'status':'PASS','evidence':evidence})
def read(key): return pd.read_csv(PATHS[key],sep='\t')
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def save(name,d): d.to_csv(O/name,sep='\t',index=False,na_rep='NA')
def valid_nt(nt,aa):
    return isinstance(nt,str) and len(nt)%3==0 and bool(re.fullmatch('[ACGT]+',nt)) and str(Seq(nt).translate())==aa
def label(sample):
    m=re.fullmatch(r'H2498([A-Z])(\d{3})',str(sample))
    return ''.join(m.groups()) if m else str(sample)
def frac(a,b): return a/b if b else None

h,p,old = read('selected_clone_rows'),read('selected_patient_roster'),read('frozen_v16_NT_VJ_summary')
check('selected patient roster',len(p)==339 and p.patient_key.is_unique and p.cohort.value_counts().to_dict()=={'external':202,'home':137},p.cohort.value_counts().to_dict())
check('unique selected clone rows',not h.duplicated(['cohort','sample_id','clone_id']).any(),len(h))
check('productive canonical QC rows',h[['productive','canonical_aa','qc_pass']].eq(True).all().all(),len(h))
h['candidate']=h.aa.map({v:k for k,v in AA.items()})
check('requested AA and coding NT translation',h.candidate.notna().all() and all(valid_nt(r.nt_cdr3,r.aa) for r in h.itertuples()),len(h))
h=h.merge(p[['patient_key','primary_sample_ids','primary_total_reads']],on='patient_key',validate='many_to_one')
check('positive integral bulk counts',h.read_count.gt(0).all() and h.read_count.eq(h.read_count.astype(int)).all(),'MiXCR assigned counts, not molecules')
check('selected library membership',all(r.sample_id in str(r.primary_sample_ids).split(';') for r in h.itertuples()),_private_literal('private_cae7797d04c0bac5'))
h['patient_label']=h.sample_id.map(label)
for q in AA:
    fresh=h[h.candidate.eq(q)].groupby('patient_key').read_count.sum().reindex(p.patient_key,fill_value=0).to_numpy()
    check(q+' zero-inclusive AA patient counts',np.array_equal(fresh,p[q+'_AA_reads'].to_numpy()),int(sum(fresh)))
g=h.groupby(['candidate','aa','nt_cdr3','v_gene','j_gene'],dropna=False).agg(QC_clone_rows=('clone_id','size'),patients=('patient_key','nunique'),summed_reads=('read_count','sum')).reset_index()
g=g.rename(columns={'candidate':'query','aa':'AA','nt_cdr3':'NT','v_gene':'V','j_gene':'J'})
j=g.merge(old,on=['query','AA','NT','V','J'],how='outer',suffixes=('_new','_old'),indicator=True,validate='one_to_one')
check('frozen 195 NT/V/J summaries reproduced',len(j)==195 and j._merge.eq('both').all() and all(j[c+'_new'].eq(j[c+'_old']).all() for c in ['QC_clone_rows','patients','summed_reads']),len(j))
executed=PATHS['executed_method_extraction'].read_text()
planned=PATHS['planned_protocol_extraction'].read_text()
check('executed method reports Q1 AA and three oligos',all(x in executed for x in [AA['Q1'],FWD,PROBE,REV]),str(PATHS['executed_method_document'])+_private_literal('private_fe0e67fd64625b27'))
check('planned full CDR3 identities verified',all(x in planned for x in NOM.values()),str(PATHS['planned_protocol_extraction']))
rows, patient_rows, summaries=[] ,[],[]
for scope in ['home','external','pooled_home_external']:
    ph=p if scope.startswith('pooled') else p[p.cohort.eq(scope)]
    hs=h[h.patient_key.isin(ph.patient_key)]
    for q in AA:
        z=hs[hs.candidate.eq(q)]
        ac=z.groupby('patient_key').read_count.sum().reindex(ph.patient_key,fill_value=0)
        at=int(ac.sum()); qc=int(ph.primary_total_reads.sum())
        without=ph[~ph.primary_sample_ids.map(label).eq(_private_literal('private_a6e9b00b7bc1ebc5'))]
        aw=int(ac.reindex(without.patient_key).sum())
        dom=ac.idxmax() if at else None
        summaries.append(dict(scope=scope,candidate=q,AA=AA[q],patients_denominator=len(ph),AA_carriers_ge1=int((ac>=1).sum()),AA_carriers_ge2=int((ac>=2).sum()),AA_assigned_reads=at,QC_reads_denominator=qc,distinct_coding_NT=z.nt_cdr3.nunique(),distinct_NT_VJ=len(z[['nt_cdr3','v_gene','j_gene']].drop_duplicates()),distinct_VJ=len(z[['v_gene','j_gene']].drop_duplicates()),selected_clone_rows=len(z),AA_dominant_patient_key=dom,AA_dominant_patient_label=label(ph.set_index('patient_key').loc[dom,'primary_sample_ids']) if dom else None,AA_dominant_patient_reads=int(ac.max()) if at else 0,AA_dominant_patient_read_fraction=frac(int(ac.max()),at) if at else None,**{_private_literal('private_65dfa2568eca636c'): aw}))
        for level,cols in [('coding_NT',['nt_cdr3']),('coding_NT_VJ',['nt_cdr3','v_gene','j_gene'])]:
            grouping=cols[0] if len(cols)==1 else cols
            for ident,v in z.groupby(grouping,dropna=False):
                ident=(ident,) if isinstance(ident,str) else ident
                nt=ident[0]; vcount=v.groupby('patient_key').read_count.sum().reindex(ph.patient_key,fill_value=0)
                vt=int(vcount.sum()); vk=vcount.idxmax(); vd=vcount.reindex(without.patient_key)
                vid=q+'_'+level+'_'+hashlib.sha256('|'.join(map(str,ident)).encode()).hexdigest()[:12]
                row=dict(scope=scope,candidate=q,identity_level=level,variant_id=vid,AA=AA[q],coding_NT_5to3=nt,coding_RNA_display_5to3=nt.replace('T','U'),RNA_display_definition='Coding CDR3 NT T-to-U representation; not recovered original RNA or full transcript',NT_length=len(nt),translated_AA=str(Seq(nt).translate()),V=';'.join(sorted(v.v_gene.unique())),J=';'.join(sorted(v.j_gene.unique())),NT_VJ_branches_for_NT=len(v[['v_gene','j_gene']].drop_duplicates()),selected_clone_rows=len(v),variant_assigned_reads=vt,candidate_AA_reads_denominator=at,pooled_candidate_read_fraction=vt/at,pooled_candidate_read_percent=100*vt/at,all_cohort_QC_reads_denominator=qc,fraction_all_cohort_QC_reads=vt/qc,patients_denominator=len(ph),variant_patients_ge1=int((vcount>=1).sum()),variant_patients_ge2=int((vcount>=2).sum()),variant_prevalence_ge1_percent=100*(vcount>=1).sum()/len(ph),variant_prevalence_ge2_percent=100*(vcount>=2).sum()/len(ph),AA_carriers_ge1_denominator=int((ac>=1).sum()),AA_carriers_ge2_denominator=int((ac>=2).sum()),fraction_variant_ge1_among_AA_ge1=frac(int((vcount>=1).sum()),int((ac>=1).sum())),fraction_variant_ge2_among_AA_ge2=frac(int((vcount>=2).sum()),int((ac>=2).sum())),equal_patient_mean_QC_read_fraction=float((vcount/ph.set_index('patient_key').primary_total_reads).mean()),equal_AA_carrier_mean_variant_fraction=float((vcount[ac>=1]/ac[ac>=1]).mean()),dominant_patient_key=vk,dominant_patient_label=label(ph.set_index('patient_key').loc[vk,'primary_sample_ids']),dominant_patient_reads=int(vcount.max()),dominant_patient_fraction_variant_reads=int(vcount.max())/vt,**{_private_literal('private_21c590a43840c523'): int(vd.sum())},**{_private_literal('private_827b234467a8acd2'): aw},**{_private_literal('private_5f67e5042cefd06c'): frac(int(vd.sum()), aw)},**{_private_literal('private_3bffaf03de092b0e'): int((vd >= 2).sum())},**{_private_literal('private_223046319a02e8b7'): len(without)},**{_private_literal('private_8fe261d63301b0ae'): int((ac.reindex(without.patient_key) >= 2).sum())},nominated_design_CDR3_NT_match=nt==NOM[q],Q1_executed_method_forward_exact_site=FWD in nt if q=='Q1' else None,Q1_executed_method_probe_exact_site=PROBE in nt if q=='Q1' else None,Q1_forward_probe_exact_compatible=(FWD in nt and PROBE in nt) if q=='Q1' else None,reverse_site_assessment='TRBC reverse-primer flank not present in CDR3 export; not assessable',count_unit='MiXCR assigned productive QC reads',patient_presence_rule='Sum this identity within selected patient library; apply >=1 or >=2 counts',evidence_boundary='Coding synonymy does not establish complete receptor identity, antigen specificity or independent assay coverage')
                nd=vcount.drop(index=vk); ad=ac.drop(index=vk)
                row.update(variant_reads_without_its_dominant_patient=int(nd.sum()),candidate_AA_reads_without_variant_dominant_patient_denominator=int(ad.sum()),variant_read_fraction_without_its_dominant_patient=frac(int(nd.sum()),int(ad.sum())),variant_patients_ge2_without_its_dominant_patient=int((nd>=2).sum()),patients_without_variant_dominant_patient_denominator=len(ph)-1)
                rows.append(row)
                if scope!='pooled_home_external':
                    for key,n in vcount[vcount>0].items():
                        patient_rows.append(dict(scope=scope,candidate=q,identity_level=level,variant_id=vid,coding_NT_5to3=nt,V=row['V'],J=row['J'],patient_key=key,patient_label=label(ph.set_index('patient_key').loc[key,'primary_sample_ids']),variant_assigned_reads=int(n),AA_assigned_reads=int(ac[key]),QC_reads=int(ph.set_index('patient_key').loc[key,'primary_total_reads']),variant_present_ge2=n>=2))
d=pd.DataFrame(rows).sort_values(['scope','candidate','identity_level','variant_assigned_reads','coding_NT_5to3'],ascending=[True,True,True,False,True])
d['read_rank_within_cohort_candidate_identity']=d.groupby(['scope','candidate','identity_level']).cumcount()+1
save('bulk_all_variants.tsv',d)
save('bulk_coding_NT_variants.tsv',d[d.identity_level.eq('coding_NT')])
save('bulk_NT_VJ_variants.tsv',d[d.identity_level.eq('coding_NT_VJ')])
save('home_plot_ready_NT.tsv',d[d.identity_level.eq('coding_NT') & d.scope.eq('home')])
save('top10_NT_by_cohort.tsv',d[d.identity_level.eq('coding_NT') & d.read_rank_within_cohort_candidate_identity.le(10)])
save('patient_variant_counts.tsv',pd.DataFrame(patient_rows))
save('cohort_variant_denominators.tsv',pd.DataFrame(summaries))
plot=[]
for scope in ['home','external','pooled_home_external']:
    ph=p if scope.startswith('pooled') else p[p.cohort.eq(scope)]
    for q in AA:
        z=h[h.candidate.eq(q) & h.patient_key.isin(ph.patient_key)]
        nt=d[d.scope.eq(scope) & d.candidate.eq(q) & d.identity_level.eq('coding_NT')]
        if nt.empty: continue
        top=nt.head(3)
        for r in top.to_dict('records'):
            r.update(plot_row_type='single_coding_NT',other_distinct_coding_NT=0,prevalence_definition='This single coding NT >=2 summed assigned reads within selected patient library')
            plot.append(r)
        other=z[~z.nt_cdr3.isin(top.coding_NT_5to3)]
        oc=other.groupby('patient_key').read_count.sum().reindex(ph.patient_key,fill_value=0)
        individual=other.groupby(['patient_key','nt_cdr3']).read_count.sum()
        qualifying=individual[individual>=2].reset_index().patient_key.nunique()
        plot.append(dict(scope=scope,candidate=q,identity_level='aggregated_other_coding_NT',variant_id=q+'_Other_after_top3',AA=AA[q],coding_NT_5to3='Other '+str(other.nt_cdr3.nunique())+' coding NT variants',coding_RNA_display_5to3='Not a single sequence',variant_assigned_reads=int(oc.sum()),candidate_AA_reads_denominator=int(z.read_count.sum()),pooled_candidate_read_fraction=oc.sum()/z.read_count.sum(),pooled_candidate_read_percent=100*oc.sum()/z.read_count.sum(),variant_patients_ge1=int((oc>=1).sum()),variant_patients_ge2=int((oc>=2).sum()),patients_denominator=len(ph),variant_prevalence_ge2_percent=100*(oc>=2).sum()/len(ph),read_rank_within_cohort_candidate_identity=4,plot_row_type='aggregate_other',other_distinct_coding_NT=other.nt_cdr3.nunique(),prevalence_definition='Sum ALL remaining coding NT variants per patient, then apply >=2 total Other counts; this does not imply an individual remaining NT has >=2 counts',patients_with_any_individual_remaining_NT_ge2=qualifying))
save('plot_summary_top3_with_other.tsv',pd.DataFrame(plot))
for key,v in pd.DataFrame(plot).groupby(['scope','candidate']):
    check('top3 plus Other reproduces counts '+str(key),v.variant_assigned_reads.sum()==v.candidate_AA_reads_denominator.iloc[0],int(v.variant_assigned_reads.sum()))
for key,v in d.groupby(['scope','candidate','identity_level']):
    check('read fractions sum to one '+str(key),abs(v.pooled_candidate_read_fraction.sum()-1)<1e-12,float(v.pooled_candidate_read_fraction.sum()))
compat=[]
for scope in ['home','pooled_home_external']:
    ph=p if scope.startswith('pooled') else p[p.cohort.eq('home')]
    for exclusion in ['none',_private_literal('private_a6e9b00b7bc1ebc5')]:
        ph=ph if exclusion=='none' else ph[~ph.primary_sample_ids.map(label).eq(_private_literal('private_a6e9b00b7bc1ebc5'))]
        z=h[h.candidate.eq('Q1') & h.patient_key.isin(ph.patient_key)]
        allc=z.groupby('patient_key').read_count.sum().reindex(ph.patient_key,fill_value=0)
        cz=z[z.nt_cdr3.map(lambda n:FWD in n and PROBE in n)]
        cc=cz.groupby('patient_key').read_count.sum().reindex(ph.patient_key,fill_value=0)
        compat.append(dict(scope=scope,excluded_patient=exclusion,distinct_compatible_coding_NT=cz.nt_cdr3.nunique(),compatible_assigned_reads=int(cc.sum()),Q1_AA_assigned_reads=int(allc.sum()),compatible_pooled_read_fraction=cc.sum()/allc.sum(),compatible_carriers_ge2=int((cc>=2).sum()),Q1_AA_carriers_ge2=int((allc>=2).sum()),patients_denominator=len(ph),compatibility_definition=_private_literal('private_575834fa6c76deab'),complete_assay_coverage_validated=False))
save('Q1_executed_method_oligo_compatibility.tsv',pd.DataFrame(compat))
check('Q1 exact forward/probe compatibility',compat[0]['distinct_compatible_coding_NT']==2 and compat[0]['compatible_assigned_reads']==16659 and compat[0]['compatible_carriers_ge2']==24,compat[0])
check(_private_literal('private_fd75ca947247e42a'),compat[1]['compatible_assigned_reads']==204 and compat[1]['compatible_carriers_ge2']==23,compat[1])

# Public data remain separate by source study and specimen role; denominators are
# source libraries, never independent patients or interchangeable read units.
u=read('unified_candidate_units')
u=u[u.unit.ne('patient_selected_library') & u.count_branch.eq('sequence_proxy_QC') & u.definition.isin(['Q1_AA','Q2_AA'])].copy()
u['candidate']=u.definition.str[:2]
check('public zero-inclusive unit roster',len(u)==772 and not u.duplicated(['library_id','candidate']).any(),{'libraries':u.library_id.nunique(),'candidate_library_rows':len(u)})
check('public count-evaluable libraries',u[u.background_count.gt(0)].library_id.nunique()==385,{'total_libraries':386,'count_evaluable_libraries':385,'unevaluable_is_not_biological_negative':True})
pub=[]
a=read('public_Q1_original_rows')
principal_healthy=a[a.metadata_condition.astype(str).str.casefold().eq('healthy')]
check('all four principal-source healthy Q1 hits are singletons',len(principal_healthy)==4 and principal_healthy.mathematical_integer_cloneCount.eq(1).all(),{'observations':len(principal_healthy),'studies':principal_healthy.study.value_counts().to_dict(),'at_least_two_counts':0})
for r in a.to_dict('records'):
    pub.append(dict(candidate='Q1',study=r['study'],library_id=r['library_id'],donor_key=r['donor_key'],source_row=r['csv_row_number'],nt=r['normalized_NT'],V=r['source_Vregion'],J=r['source_Jregion'],count=r['mathematical_integer_cloneCount'],proxy=r['sequence_proxy_QC_pass'],source_table=str(PATHS['public_Q1_original_rows'])))
a=read('public_Q2_original_rows')
for r in a.to_dict('records'):
    pub.append(dict(candidate='Q2',study=r['canonical_study'],library_id=r['library_id'],donor_key=r['donor_key'],source_row=r['data_row_1based'],nt=r['NNSeq'],V=r['Vregion_normalized'],J=r['Jregion_normalized'],count=r['positive_integral_cloneCount'],proxy=r['sequence_proxy_QC'],source_table=str(PATHS['public_Q2_original_rows'])))
a=read('tissue_reference_original_rows')
for r in a[a.AA.isin(AA.values())].to_dict('records'):
    q={v:k for k,v in AA.items()}[r['AA']]
    pub.append(dict(candidate=q,study=r['study'],library_id=r['library_id'],donor_key=r['donor_key'],source_row=r['data_row_1based'],nt=r['NT'],V=r['V_normalized'],J=r['J_normalized'],count=r['positive_integral_cloneCount'],proxy=r['sequence_proxy_QC'],source_table=str(PATHS['tissue_reference_original_rows'])))
pub=pd.DataFrame(pub)
pub=pub[pub.proxy.eq(True) & pub['count'].notna() & pub['count'].gt(0)]
pub=pub[[valid_nt(r.nt,AA[r.candidate]) for r in pub.itertuples()]].copy()
dupkeys=['candidate','library_id','source_row']
overlap=pub[pub.duplicated(dupkeys,keep=False)]
check('overlapping public export identities and counts agree',all(v[['nt','V','J','count']].drop_duplicates().shape[0]==1 for _,v in overlap.groupby(dupkeys)),{'duplicate_source_rows':len(overlap)})
provenance=pub.groupby(dupkeys).source_table.agg(lambda x:';'.join(sorted(set(x))))
pub=pub.drop_duplicates(dupkeys).drop(columns='source_table').merge(provenance.rename('source_table').reset_index(),on=dupkeys,validate='one_to_one')
pub=pub.merge(u[['library_id','candidate','role','background_count','candidate_count','count_unit']],on=['library_id','candidate'],how='inner',validate='many_to_one')
pub['count']=pub['count'].astype(int)
check('unique public source rows',not pub.duplicated(['candidate','library_id','source_row']).any(),len(pub))
pu=pub.groupby(['library_id','candidate'])['count'].sum().reindex(pd.MultiIndex.from_frame(u[['library_id','candidate']]),fill_value=0).to_numpy()
check('public per-library candidate counts reproduced',np.array_equal(pu,u.candidate_count.to_numpy()),{'rows':len(pub),'libraries':u.library_id.nunique()})
save('public_variant_source_rows.tsv',pub)
ps,pr=[],[]
for key,units in u.groupby(['candidate','study','role']):
    q,study,role=key; ev=units[units.background_count.gt(0)]
    z=pub[pub.candidate.eq(q) & pub.study.eq(study) & pub.role.eq(role)]
    den=int(ev.candidate_count.sum())
    ps.append(dict(candidate=q,study=study,role=role,libraries_all=len(units),count_evaluable_libraries=len(ev),source_donor_labels_evaluable=ev.donor_key.nunique(),candidate_source_counts=den,distinct_coding_NT=z.nt.nunique(),distinct_NT_VJ=len(z[['nt','V','J']].drop_duplicates()),count_unit=';'.join(sorted(ev.count_unit.dropna().unique())),inference_unit='Source library; donor labels not independent patients'))
    for nt,v in z.groupby('nt'):
        vc=v.groupby('library_id')['count'].sum().reindex(ev.library_id,fill_value=0)
        pr.append(dict(candidate=q,study=study,role=role,coding_NT_5to3=nt,coding_RNA_display_5to3=nt.replace('T','U'),RNA_display_definition='Coding CDR3 T-to-U display; processed source sequence, not original RNA',AA=AA[q],V=';'.join(sorted(v.V.dropna().unique())),J=';'.join(sorted(v.J.dropna().unique())),distinct_NT_VJ=len(v[['nt','V','J']].drop_duplicates()),source_clone_rows=len(v),variant_source_counts=int(vc.sum()),candidate_source_counts_denominator=den,pooled_within_source_count_percent=100*vc.sum()/den,variant_libraries_ge1=int((vc>=1).sum()),variant_libraries_ge2=int((vc>=2).sum()),count_evaluable_libraries_denominator=len(ev),library_prevalence_ge1_percent=100*(vc>=1).sum()/len(ev),library_prevalence_ge2_percent=100*(vc>=2).sum()/len(ev),source_donor_labels=v.donor_key.nunique(),donor_labels=';'.join(sorted(v.donor_key.dropna().unique())),count_unit=';'.join(sorted(v.count_unit.dropna().unique())),independent_patient_prevalence=False))
save('public_source_denominators.tsv',pd.DataFrame(ps))
save('public_NT_variants_separate_sources.tsv',pd.DataFrame(pr))
pj=[]
for key,v in pub.groupby(['candidate','study','role','nt','V','J'],dropna=False):
    q,study,role,nt,V,J=key
    ev=u[u.candidate.eq(q) & u.study.eq(study) & u.role.eq(role) & u.background_count.gt(0)]
    vc=v.groupby('library_id')['count'].sum().reindex(ev.library_id,fill_value=0)
    den=int(ev.candidate_count.sum())
    pj.append(dict(candidate=q,study=study,role=role,coding_NT_5to3=nt,coding_RNA_display_5to3=nt.replace('T','U'),AA=AA[q],V=V,J=J,source_clone_rows=len(v),variant_source_counts=int(vc.sum()),candidate_source_counts_denominator=den,pooled_within_source_count_percent=100*vc.sum()/den,variant_libraries_ge1=int((vc>=1).sum()),variant_libraries_ge2=int((vc>=2).sum()),count_evaluable_libraries_denominator=len(ev),source_donor_labels=v.donor_key.nunique(),donor_labels=';'.join(sorted(v.donor_key.dropna().unique())),count_unit=';'.join(sorted(v.count_unit.dropna().unique())),independent_patient_prevalence=False))
save('public_NT_VJ_variants_separate_sources.tsv',pd.DataFrame(pj))
check('healthy public Q1 observations are singletons',pub[pub.candidate.eq('Q1') & pub.role.eq('healthy_blood_descriptive')]['count'].eq(1).all(),int(len(pub[pub.candidate.eq('Q1') & pub.role.eq('healthy_blood_descriptive')])))
save('nominated_CDR3_target_registry.tsv',pd.DataFrame([dict(candidate=q,AA=AA[q],nominated_design_CDR3_NT_5to3=NOM[q],coding_RNA_display_5to3=NOM[q].replace('T','U'),full_NT_provenance='Planned protocol; not independently confirmed executed plasmid insert',executed_oligos_reported=q=='Q1',executed_forward=FWD if q=='Q1' else None,executed_reverse=REV if q=='Q1' else None,executed_probe=PROBE if q=='Q1' else None,full_transcript_available=False) for q in AA]))
manifest=[dict(source_label=k,absolute_path=str(v),bytes=v.stat().st_size,sha256=sha(v)) for k,v in PATHS.items()]
save('input_manifest.tsv',pd.DataFrame(manifest))
readme=_private_literal('private_c0fbf36bb41bdffa')
(O/'README.md').write_text(readme)
def clean(x):
    if isinstance(x,dict): return {k:clean(v) for k,v in x.items()}
    if isinstance(x,list): return [clean(v) for v in x]
    if isinstance(x,(np.integer,)): return int(x)
    if isinstance(x,(np.floating,float)): return None if not np.isfinite(x) else float(x)
    if isinstance(x,np.bool_): return bool(x)
    return x
outputs=[dict(relative_path=str(f.relative_to(R)),bytes=f.stat().st_size,sha256=sha(f)) for f in sorted(O.iterdir()) if f.is_file()]
audit=dict(status='PASS',scope='Fresh compact original-export reaggregation; source studies and patient/library units separate',script_sha256=sha(Path(__file__)),inputs=manifest,cohort_summaries=summaries,executed_Q1_forward_probe_compatibility=compat,checks=checks,unresolved_material_issues=[],scientific_boundaries=['Coding RNA display is not recovered original RNA','Full executed insert and reverse-primer flank unavailable','Compatibility is not assay coverage validation','Low-count cross-sample attribution unresolved in frozen raw audit','Public libraries and source counts are not independent patients or harmonized bulk reads'],ready_to_use_prose={'main':'In the study cohort, Q1 was encoded by 14 distinct CDR3 nucleotide sequences and Q2 by 72. The leading Q1 nucleotide sequence accounted for 96.88% of Q1-assigned reads but was detected at a minimum of two assigned reads in 24 of 137 patients; the leading Q2 sequence accounted for 59.27% of Q2-assigned reads and occurred in 27 of 137 patients. Thus, recurrence of an amino-acid sequence encompasses distinct coding variants, while pooled read dominance and patient prevalence describe different features.','methods_or_supplement':'We report processed coding CDR3 sequences in the 5-prime to 3-prime direction. Coding RNA representations were generated solely by replacing T with U and do not represent recovered original RNA transcripts. Exact forward-primer and probe sites reported in the executed Q1 assay methods were present in two study-cohort coding variants, together accounting for 16,659 of 17,126 Q1-assigned reads and 24 of 53 Q1 AA carriers at the two-count threshold. These site matches do not establish full-amplicon compatibility, because the complete executed insert and the reverse-primer flank were unavailable.','limitation':'Assigned-read support was concentrated in individual carriers, and the earlier raw-data audit left attribution of low-count within-run detections unresolved; coding identity and oligo-site matching therefore do not establish tumor reactivity or validated diagnostic specificity.'},outputs=outputs)
(R/'audit').mkdir(exist_ok=True)
(R/'audit/nucleotide_variants_audit.json').write_text(json.dumps(clean(audit),indent=2,ensure_ascii=False,allow_nan=False)+'\n')
print(json.dumps(clean({'status':'PASS','checks':len(checks),'bulk_variant_rows':len(d),'public_variant_rows':len(pr),'cohort_summaries':summaries,'output_directory':str(O)}),indent=2,ensure_ascii=False,allow_nan=False))
