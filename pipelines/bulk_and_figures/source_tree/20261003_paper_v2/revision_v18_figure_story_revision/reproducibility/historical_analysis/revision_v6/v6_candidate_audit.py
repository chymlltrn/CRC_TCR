#!/usr/bin/env python3
"""Cached, zero-inclusive, source-resolved candidate audit; execute on codex-remote."""
from release_runtime import private_literal as _private_literal, release_path as _release_path
from pathlib import Path
import argparse, hashlib, json, re, os, math, sys
from datetime import datetime, timezone
import pandas as pd
import numpy as np
from scipy.stats import hypergeom
from Bio.Seq import Seq

THRESHOLDS=(1,2,5,10,20)
Q1=_private_literal('private_45bbaa37a9607bb7'); Q2=_private_literal('private_b3445b7fc78bf2f0')
Q1NT=_private_literal('private_c287d28aaf0e9d7c')
DEFS=('Q1_AA','Q1_AA_VJ','Q1_NT','Q2_AA')

def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def clean(x): return '' if pd.isna(x) else str(x).strip()
def truth(x): return str(x).lower() in ('true','1')
def read(p): return pd.read_csv(p,sep='\t',keep_default_na=False)
def write(d,p): d.to_csv(p,sep='\t',index=False)
def wilson(k,n):
    z=1.959963984540054
    v=k/n; a=1+z*z/n; c=(v+z*z/(2*n))/a
    h=z*math.sqrt(v*(1-v)/n+z*z/(4*n*n))/a
    return max(0,c-h),min(1,c+h)

def main(root):
    parent=root.parent; baseline=parent/'revision_v5'; out=root/'candidates'
    out.mkdir(parents=True,exist_ok=True); (out/'inputs').mkdir(exist_ok=True)
    source={
      'patients':baseline/'legacy/patient_Q1_Q2.tsv',
      'principal':baseline/'legacy/reference_library_Q1_Q2.tsv',
      'tissue':baseline/'legacy/tissue_reference/per_repertoire_Q1_Q2.tsv',
      'bulk_hits':baseline/'legacy/matching_clone_rows.tsv',
      'tissue_hits':baseline/'legacy/tissue_reference/exact_candidate_hit_rows.tsv',
      'recurrence':baseline/'legacy/tissue_reference/Le2017_Q1_within_reference_recurrence.tsv',
      'bulk_variants':baseline/'legacy/Q1_Q2_observed_NT_VJ_variants.tsv',
      'antigen':baseline/'antigen/antigen_evidence_long.tsv',
      'qpcr':baseline/'legacy/sources/paper_v1/qpcr_reported_results.tsv',
      'qpcr_inventory':baseline/'legacy/sources/paper_v1/qpcr_source_inventory.tsv',
      'gate':parent/'q1/scientific_gate.json',
      'raw_inventory':parent.parent/'20261003_paper_v1/q1/raw_evidence_inventory.tsv',
      'raw_support':parent.parent/'20261003_paper_v1/q1/FASTQ_selected_support.tsv',
      'raw_gate':parent.parent/'20261003_paper_v1/q1/technical_gate.json',
      'patient_registry':parent.parent/'20261003_paper_v1/cohort/patient_master.tsv',
      'central_protocol':root/'PROTOCOL_v6.md',
      'lane_protocol':out/'PROTOCOL.md',
    }
    inputs=[]
    for label,p in source.items():
        assert p.exists(),p
        inputs.append(dict(label=label,path=str(p),bytes=p.stat().st_size,sha256=sha(p)))
    write(pd.DataFrame(inputs),out/'input_manifest.tsv')
    patients=read(source['patients']); principal=read(source['principal']); tissue=read(source['tissue'])
    assert len(patients)==339 and patients.patient_key.nunique()==339
    assert patients.groupby('cohort').size().to_dict()=={'external':202,'home':137}
    principal=principal[principal.study.isin(['immunoSEQ139','immunoSEQ135','PRJNA300878'])].copy()
    assert len(principal)==256 and principal.library_id.nunique()==256
    assert principal.scan_complete_to_EOF.map(truth).all()
    assert len(tissue)==130 and tissue.library_id.nunique()==130 and tissue.full_EOF.map(truth).all()
    # Later tissue inventory supersedes seven previous IT1208 rows; never double-count.
    earlier=read(source['principal'])
    earlier=earlier[~earlier.study.isin(['immunoSEQ139','immunoSEQ135','PRJNA300878'])].copy()
    overlap=[]
    for _,r in earlier.iterrows():
        run=clean(r.source_run_id) or Path(r.zip_member).stem
        a=tissue[tissue.source_run_id==run]
        assert len(a)==1
        a=a.iloc[0]
        assert float(r.Q1_AA_sequence_proxy_QC_cloneCount)==float(a.Q1_AA_proxy_QC_count)
        assert float(r.Q2_sequence_proxy_QC_cloneCount)==float(a.Q2_AA_proxy_QC_count)
        overlap.append(dict(prior_library_id=r.library_id,source_run_id=run,canonical_library_id=a.library_id,prior_counted=False,canonical_counted_once=True))
    assert len(overlap)==7
    write(pd.DataFrame(overlap),out/'previous_seven_run_overlap.tsv')
    records=[]
    for _,r in patients.iterrows():
        for definition in DEFS:
            records.append(dict(unit_id='patient:'+r.patient_key,unit='patient_selected_library',study='Home' if r.cohort=='home' else 'External',cohort=r.cohort,library_id=r.primary_sample_ids,donor_key=r.patient_key,donor_identity_status='frozen_patient_master',disease='CRC',material='peripheral_blood',role='selected_peripheral_blood',processing='frozen_MiXCR_productive_QC',visit='collection_timing_unverified',cell_subset='unsorted',batch=r.batch,n_binary=r.n_binary,definition=definition,count_branch='frozen_MiXCR_QC',candidate_count=int(r[definition+'_reads']),background_count=int(r.primary_total_reads),count_unit='MiXCR_assigned_QC_reads',independent_patient=True,full_EOF=True,source_file=str(source['patients']),source_sha256=sha(source['patients']),source_role_authority='frozen_selected_patient_master',clinical_comparability='no_same_assay_healthy_controls_verified',notes=_private_literal('private_72c03d3b9a1a2c27')))
    for _,r in principal.iterrows():
        for definition in DEFS:
            for branch in ('source_all','sequence_proxy_QC'):
                stem=definition+'_'
                col=(stem+('source_cloneCount' if branch=='source_all' else 'sequence_proxy_QC_cloneCount')) if definition!='Q2_AA' else ('Q2_source_exact_cloneCount' if branch=='source_all' else 'Q2_sequence_proxy_QC_cloneCount')
                denominator=r.source_total_positive_integral_cloneCount if branch=='source_all' else r.sequence_proxy_QC_total_cloneCount
                assert clean(denominator)!='' and clean(r[col])!=''
                records.append(dict(unit_id='library:'+r.library_id,unit='source_library_not_independent_people',study=r.study,cohort='public_reference',library_id=r.library_id,donor_key=r.donor_key,donor_identity_status=r.donor_identity_status,disease=r.metadata_condition,material=r.metadata_cell_source,role=r.primary_comparison_role,processing=r.assay_protocol,visit='unverified_or_source_repeat',cell_subset=r.metadata_cell_type,batch='source_batch_unverified',n_binary='unverified',definition=definition,count_branch=branch,candidate_count=int(float(r[col])),background_count=int(float(denominator)),count_unit=r.count_unit,independent_patient=False,full_EOF=True,source_file=str(source['principal']),source_sha256=sha(source['principal']),source_role_authority='frozen_source_metadata_labels',clinical_comparability=r.eligibility_for_clinical_specificity,notes=r.independence_note))
    for _,r in tissue.iterrows():
        for definition in DEFS:
            for branch in ('source_all','sequence_proxy_QC'):
                col=definition+('_source_count' if branch=='source_all' else '_proxy_QC_count')
                denominator=r.source_total_positive_integral_cloneCount if branch=='source_all' else r.sequence_proxy_QC_total_cloneCount
                records.append(dict(unit_id='library:'+r.library_id,unit='source_repertoire_not_independent_people',study=r.study,cohort='tissue_blood_reference',library_id=r.library_id,donor_key=r.donor_key,donor_identity_status=r.donor_identity_status,disease=r.cancer_type,material=r.organ,role=r.role,processing=r.processing,visit=r.timepoint,cell_subset=r.cell_subset,batch='source_batch_unverified',n_binary='unverified',definition=definition,count_branch=branch,candidate_count=int(float(r[col])),background_count=int(float(denominator)),count_unit=r.source_cloneCount_unit,independent_patient=False,full_EOF=True,source_file=str(source['tissue']),source_sha256=sha(source['tissue']),source_role_authority=r.source_annotation_authority,clinical_comparability='descriptive_material_resolved_reference_only',notes='source_timepoints_subsets_runs_not_independent_people; '+clean(r.GEO_organ_vs_paper_cancer_discrepancy)))
    units=pd.DataFrame(records)
    assert units.unit_id.nunique()==725 and not units.duplicated(['unit_id','definition','count_branch']).any()
    write(units,out/'candidate_units.tsv')
    definition_registry=pd.DataFrame([
      dict(candidate='Q1',definition='AA',AA=Q1,AA_length=16,NT='',V='',J='',status='frozen_complete_AA_any_NT_VJ'),
      dict(candidate='Q1',definition='AA_VJ',AA=Q1,AA_length=16,NT='',V='TRBV5-5',J='TRBJ1-4',status='frozen_assigned_VJ_branch'),
      dict(candidate='Q1',definition='NT',AA=Q1,AA_length=16,NT=Q1NT,V='',J='',status='frozen_exact_NT_no_extra_VJ_filter'),
      dict(candidate='Q2',definition='AA',AA=Q2,AA_length=14,NT='',V='',J='',status='frozen_complete_AA_any_NT_VJ'),
      dict(candidate='Q2',definition='AA_VJ',AA=Q2,AA_length=14,NT='',V='',J='',status='unresolved_historical_V11_1_vs_V7_9_not_zero'),
      dict(candidate='Q2',definition='NT',AA=Q2,AA_length=14,NT='',V='',J='',status='unrecovered_not_zero'),
    ])
    write(definition_registry,out/'definition_registry.tsv')
    summaries=[]
    by=['study','cohort','disease','material','role','processing','definition','count_branch','unit','count_unit']
    for scope,keys in [('all_visits',by),('by_visit',by+['visit','cell_subset'])]:
        for values,g in units.groupby(keys,dropna=False,sort=True):
            base=dict(zip(keys,values));base.update(aggregation_scope=scope,visit=base.get('visit','all_recorded_visits'),cell_subset=base.get('cell_subset','all_recorded_subsets'))
            for threshold in THRESHOLDS:
                k=int((g.candidate_count>=threshold).sum());n=len(g)
                lo,hi=wilson(k,n) if g.independent_patient.all() else (np.nan,np.nan)
                summaries.append(dict(**base,threshold=threshold,n_units=n,n_carriers=k,carrier_fraction=k/n,n_source_donor_labels=g.donor_key[g.donor_key!=''].nunique(),donor_labels_with_carrier=g.loc[g.candidate_count>=threshold,'donor_key'].nunique(),candidate_sum_count=int(g.candidate_count.sum()),background_sum_count=int(g.background_count.sum()),wilson_lower=lo,wilson_upper=hi,interval_scope='descriptive_patient_only' if g.independent_patient.all() else 'not_estimated_dependent_library_units',clinical_validation=False))
    summary=pd.DataFrame(summaries);write(summary,out/'candidate_source_detection.tsv')
    # Primary patient sensitivity excludes an entire selected [PRIVATE_SUBJECT] patient's library.
    home=patients[patients.cohort=='home'];selected_subject=home[home.primary_sample_ids==_private_literal('private_d160bae090c0c274')]
    assert len(selected_subject)==1
    selected_subject_key=selected_subject.iloc[0].patient_key
    sensitivity=[]
    for cohort,g0 in patients.groupby('cohort'):
        scopes=[('full_selected',g0),(_private_literal('private_e3efc0198a793713'),g0[g0.patient_key!=selected_subject_key])]
        scopes += [('batch_'+batch,g) for batch,g in g0.groupby('batch')]
        for scope,g in scopes:
            for definition in DEFS:
                for t in THRESHOLDS:
                    k=int((g[definition+'_reads'].astype(int)>=t).sum());lo,hi=wilson(k,len(g))
                    sensitivity.append(dict(cohort=cohort,scope=scope,definition=definition,threshold=t,n_patients=len(g),n_carriers=k,carrier_fraction=k/len(g),wilson_lower=lo,wilson_upper=hi,target_assigned_read_sum=int(g[definition+'_reads'].astype(int).sum()),total_QC_assigned_reads=int(g.primary_total_reads.astype(int).sum()),interpretation='descriptive_not_association_or_validated_LOD'))
    sens=pd.DataFrame(sensitivity);write(sens,out/_private_literal('private_a70effc21f3daa7e'))
    concentrations=[]
    for cohort,g in patients.groupby('cohort'):
        for definition in DEFS:
            s=g[definition+'_reads'].astype(int);max_i=s.idxmax();total=int(s.sum())
            concentrations.append(dict(cohort=cohort,definition=definition,total_target_reads=total,max_patient_key=g.loc[max_i,'patient_key'],max_sample=g.loc[max_i,'primary_sample_ids'],max_patient_reads=int(s.loc[max_i]),max_patient_fraction=int(s.loc[max_i])/total if total else np.nan,unit='assigned_reads_not_cells_or_molecules'))
    write(pd.DataFrame(concentrations),out/'read_concentration.tsv')
    # Fixed depths; closed-form exact count thinning, no rescan/random draws.
    thinning=[]
    for _,r in patients.iterrows():
        N=int(r.primary_total_reads)
        for m in (500000,1000000,2000000):
            for definition in DEFS:
                K=int(r[definition+'_reads']);eligible=N>=m
                if not eligible: p=np.nan
                elif K<2:p=0.0
                elif m==N:p=1.0
                else:
                    log0=hypergeom.logpmf(0,N,K,m);log1=hypergeom.logpmf(1,N,K,m)
                    p=float(np.clip(-np.expm1(log0)-np.exp(log1),0,1))
                thinning.append(dict(patient_key=r.patient_key,cohort=r.cohort,batch=r.batch,sample_id=r.primary_sample_ids,definition=definition,total_QC_assigned_reads=N,observed_target_reads=K,target_depth=m,eligible=eligible,probability_target_reads_ge2=p,count_unit='MiXCR_assigned_QC_reads',interpretation='within_cohort_expected_detection_only_not_source_adjusted_disease_effect'))
    th=pd.DataFrame(thinning);write(th,out/'within_cohort_count_thinning_patients.tsv')
    sums=[]
    for (cohort,definition,m),g0 in th.groupby(['cohort','definition','target_depth']):
        groups=[('all',g0),(_private_literal('private_e3efc0198a793713'),g0[g0.patient_key!=selected_subject_key])]
        groups += [('batch_'+batch,g) for batch,g in g0.groupby('batch')]
        for scope,g in groups:
            a=g[g.eligible]
            sums.append(dict(cohort=cohort,definition=definition,target_depth=m,scope=scope,n_original_patients=len(g),n_depth_eligible=len(a),expected_carriers_ge2=float(a.probability_target_reads_ge2.sum()),expected_carrier_fraction=float(a.probability_target_reads_ge2.mean()) if len(a) else np.nan,observed_carriers_ge2_in_eligible=int((a.observed_target_reads>=2).sum()),not_a_new_observed_prevalence=True))
    write(pd.DataFrame(sums),out/'within_cohort_count_thinning_summary.tsv')
    depth=[]
    for values,g in units.drop_duplicates(['unit_id','count_branch']).groupby(['study','disease','role','count_branch','count_unit'],dropna=False):
        depth.append(dict(zip(['study','disease','role','count_branch','count_unit'],values),n_units=len(g),minimum_background_count=int(g.background_count.min()),median_background_count=float(g.background_count.median()),maximum_background_count=int(g.background_count.max()),depths_not_cross_assay_harmonized=True))
    write(pd.DataFrame(depth),out/'source_depth_units.tsv')
    # Candidate-only sequence coordinate/QC audit; not whole-source QC replacement.
    hits=read(source['bulk_hits']);thits=read(source['tissue_hits']);audit=[]
    for authority,d,aa_col,nt_col,v_col,j_col,count_col in [
        ('Home_External_frozen_MiXCR_hits',hits,'aa','nt_cdr3','v_gene','j_gene','read_count'),
        ('material_resolved_public_hits',thits,'AA','NT','V_normalized','J_normalized','positive_integral_cloneCount')]:
        for i,r in d.iterrows():
            aa=clean(r[aa_col]).upper();nt=clean(r[nt_col]).upper()
            nt_valid=bool(re.fullmatch('[ACGT]+',nt)) and len(nt)%3==0
            translated=str(Seq(nt).translate()) if nt_valid else ''
            v=clean(r[v_col]).upper().split('*')[0];j=clean(r[j_col]).upper().split('*')[0]
            audit.append(dict(authority=authority,source_row_1based=i+2,library_id=clean(r.get('sample_id',r.get('library_id',''))),candidate='Q1' if aa==Q1 else 'Q2',AA=aa,AA_length=len(aa),NT=nt,NT_length=len(nt),complete_AA_identity=aa in [Q1,Q2],C_start=aa.startswith('C'),F_end=aa.endswith('F'),canonical_AA=bool(re.fullmatch('[ACDEFGHIKLMNPQRSTVWY]+',aa)),NT_ACGT_inframe=nt_valid,translation_equals_complete_AA=translated==aa,V_assigned=v,J_assigned=j,V_syntax_valid=bool(re.fullmatch(r'TRBV[0-9]+(?:-[0-9]+)?(?:/OR[0-9]+-[0-9]+)?',v)),J_syntax_valid=bool(re.fullmatch(r'TRBJ[12]-[0-9]+P?',j)),count=r[count_col],V_uniqueness='not_asserted_from_first_rank_assignment',coordinate_scope='complete_processed_CDR3_beta_including_source_conserved_flanks_not_raw_alignment_coordinates'))
    sa=pd.DataFrame(audit);write(sa,out/'sequence_coordinate_assignment_audit.tsv')
    assert sa.complete_AA_identity.all() and sa.translation_equals_complete_AA.all()
    (out/'observed_bulk_NT_VJ_variants.tsv').write_bytes(source['bulk_variants'].read_bytes())
    (out/'Le2017_within_reference_recurrence.tsv').write_bytes(source['recurrence'].read_bytes())
    # Preserve annotations, no new database query.
    antigen=read(source['antigen']);antigen=antigen[antigen.candidate_id.isin(['Q1','Q2'])].copy()
    antigen['clinical_function_validated']=False
    antigen['full_complete_AA_identity']=antigen.query_aa==antigen.reference_cdr3beta
    antigen['binding_probability']=np.nan
    write(antigen,out/'branch_specific_antigen_annotations.tsv')
    (out/'qpcr_reported_results.tsv').write_bytes(source['qpcr'].read_bytes())
    (out/'qpcr_source_inventory.tsv').write_bytes(source['qpcr_inventory'].read_bytes())
    # One bounded targeted metadata recovery. Prior read receipts preserved, no payload read.
    rx=re.compile(r'qpcr|rt[-_]?pcr|taqman|primer|probe|plate|standard[_ -]?curve|sample[_ -]?sheet|demux|demultiplex|index[_ -]?(map|sheet)|(^|[_.-])ntc([_.-]|$)|no[_-]?rt|negative[_-]?control',re.I)
    roots=[Path(_release_path('path_df0fb75e17e2f0ab')),Path(_release_path('path_39343c5e69c3d939'))]
    prune={'raw','UID','.git','__pycache__','.venv','venv','node_modules','0929_caseEmbedding'}
    found=[];nfiles=0;truncated=False;limit=50000
    for search_root in roots:
        if not search_root.exists():continue
        for directory,dirs,files in os.walk(search_root):
            rel=Path(directory).relative_to(search_root);level=len(rel.parts)
            dirs[:]=[a for a in dirs if a not in prune and not a.startswith('20261003_') and not a.startswith('revision_') and not a.startswith('env')]
            if level>=6:dirs[:]=[]
            for name in files:
                nfiles+=1
                if nfiles>limit:truncated=True;break
                if not rx.search(name):continue
                p=Path(directory)/name
                if not p.is_file():continue
                st=p.stat();whole_hash=sha(p) if st.st_size<=2*1024**2 else ''
                preview=''
                if p.suffix.lower() in ('.txt','.tsv','.csv','.md','.json','.yaml','.yml','.log','.ini','.xml') and st.st_size<=2*1024**2:
                    try:preview=p.open(errors='replace').read(256*1024)
                    except OSError:pass
                found.append(dict(path=str(p),bytes=st.st_size,sha256=whole_hash,hash_scope='whole_small_file' if whole_hash else 'stat_only',text_preview_bytes=min(st.st_size,256*1024) if preview else 0,filename_candidate_only=True,resolves_technical_or_qpcr_gate=False,note='requires_primary_source_review; filename_or_method_text_is_not_raw_evidence'))
            if truncated:break
        if truncated:break
    write(pd.DataFrame(found,columns=['path','bytes','sha256','hash_scope','text_preview_bytes','filename_candidate_only','resolves_technical_or_qpcr_gate','note']),out/'technical_recovery_inventory.tsv')
    raw=read(source['raw_inventory']);now=[]
    for _,r in raw.iterrows():
        p=Path(r.path);e=p.is_file()
        now.append(dict(**r.to_dict(),current_exists=e,current_bytes=p.stat().st_size if e else 0,current_review='stat_only_no_new_body_scan',prior_index_review='first_original_Illumina_header_and_prefix_receipts_not_full_demultiplex_or_indexed_alignment_audit'))
    write(pd.DataFrame(now),out/'previous_raw_evidence_stat_refresh.tsv')
    support=read(source['raw_support']);(out/'existing_FASTQ_selected_support.tsv').write_bytes(source['raw_support'].read_bytes())
    rg=json.loads(source['raw_gate'].read_text());prefix=[]
    for r in rg['raw_scans']:
        prefix.append(dict(sample_id=r['sample_id'],input_type=r['input_type'],pairs_inspected=r['pairs_inspected'],full_file_coverage=r['full_file_coverage'],exact_supplied_NT_pairs=r['exact_full_48nt_pair_counts'].get(Q1NT,0),prefix_only=True,stream_digest_sha256=r['stream_digest_sha256'],new_v6_scan=False))
    write(pd.DataFrame(prefix),out/'existing_raw_prefix_review_summary.tsv')
    recovery=dict(status='unresolved',new_resolving_raw_or_qpcr_records_recovered=False,roots=list(map(str,roots)),max_depth=6,max_filenames=limit,filenames_examined=nfiles,search_truncated=truncated,filename_candidates=len(found),new_FASTQ_or_alignment_payload_scans=0,previous_prefix_scans=len(prefix),previous_prefix_scope='deterministic_first_250000_pairs_each_original_or_UID; not_EOF_not_representative',indexed_read_trace_complete=False,prior_header_index_review=True,NT_batch_versus_biology_resolved=False,qpcr_actual_primer_and_patient_map_recovered=False,source_limit='bounded_source_roots_and_pruned_analysis/raw_payloads; no_global_absence_claim')
    (out/'technical_recovery_receipt.json').write_text(json.dumps(recovery,indent=2)+'\n')
    # Frozen numeric source for F2.
    display=[]
    filters=[
      ('Home CRC','patients','Home',None,'selected_peripheral_blood'),
      ('External CRC','patients','External',None,'selected_peripheral_blood'),
      ('139 Healthy blood','libraries','immunoSEQ139','Healthy','healthy_blood_descriptive'),
      ('139 CRC PBMC','libraries','immunoSEQ139','ColorectalCancer','external_crc_pbmc_descriptive_within_source'),
      ('Le2017 serial PBMC','libraries; 3 donor labels','immunoSEQ75',None,'serial_PBMC'),
      ('Le2017 archival tumor','libraries; 3 donor labels','immunoSEQ75',None,'archival_primary_tumor'),
      ('Le2017 peptide/control culture','libraries; 1 donor label','immunoSEQ75',None,'PBMC_derived_culture')]
    for label,unit_label,study,disease,role in filters:
        g=units[(units.study==study)&(units.role==role)&units.count_branch.isin(['frozen_MiXCR_QC','sequence_proxy_QC'])]
        # Role itself is the pre-frozen condition/material classification.
        assert len(g)>0,(label,study,role)
        for definition in ('Q1_AA','Q2_AA'):
            a=g[g.definition==definition]
            display.append(dict(source_label=label,unit_label=unit_label,study=study,role=role,candidate=definition[:2],definition='AA',threshold=2,n_units=len(a),n_carriers=int((a.candidate_count>=2).sum()),carrier_fraction=float((a.candidate_count>=2).mean()),n_donor_labels=a.donor_key.nunique(),source_table='candidate_units.tsv',count_units='source_specific_not_cross_assay_harmonized'))
    fig=pd.DataFrame(display);write(fig,out/'Figure2_source_carriage.tsv')
    # Regression/integrity checks cover the requested fixed counts and support nesting.
    def patcount(cohort,d,t):
        return int((patients.loc[patients.cohort==cohort,d+'_reads'].astype(int)>=t).sum())
    assert [patcount('home','Q1_AA',t) for t in THRESHOLDS]==[63,53,32,15,6]
    assert [patcount('external','Q1_AA',t) for t in THRESHOLDS]==[0]*5
    assert patcount('home','Q1_AA_VJ',2)==24 and patcount('home','Q1_NT',2)==24
    assert patcount('home','Q2_AA',2)==99 and patcount('external','Q2_AA',2)==81
    assert patients.Q1_NT_reads.astype(int).le(patients.Q1_AA_VJ_reads.astype(int)).all()
    assert patients.Q1_AA_VJ_reads.astype(int).le(patients.Q1_AA_reads.astype(int)).all()
    assert (fig[(fig.study=='immunoSEQ139')&(fig.candidate=='Q1')].n_carriers==0).all()
    assert fig[(fig.role=='serial_PBMC')&(fig.candidate=='Q1')].n_carriers.iloc[0]==7
    assert fig[(fig.role=='archival_primary_tumor')&(fig.candidate=='Q1')].n_carriers.iloc[0]==1
    assert fig[(fig.role=='PBMC_derived_culture')&(fig.candidate=='Q1')].n_carriers.iloc[0]==0
    assert len(th)==339*4*3 and th.loc[th.eligible,'probability_target_reads_ge2'].between(0,1).all()
    assert patcount('home','Q1_AA',2)-int(selected_subject.Q1_AA_reads.iloc[0]>=2)==52
    freeze=dict(status='passed_cached_reaggregation',generated_utc=datetime.now(timezone.utc).isoformat(),selected_patients=339,Home=137,External=202,principal_public_libraries=256,additional_material_resolved_repertoires=130,unique_total_units=725,prior_seven_runs_counted_again=False,new_raw_source_scans=0,Q1_primary_AA_Home='53/137',Q1_primary_AA_External='0/202',Q1_primary_AA_139Healthy='0/87',Q1_primary_AA_139CRC='0/20',Q2_primary_AA_Home='99/137',Q2_primary_AA_External='81/202',authenticity='unresolved',qpcr='reported_not_raw_verified',disease_diagnostic_inference=False,function_inference=False,producing_script_sha256=sha(Path(__file__)),protocol_sha256=sha(out/'PROTOCOL.md'),central_protocol_sha256=sha(root/'PROTOCOL_v6.md'),checks=['725_zero_inclusive_unique_units','339_fixed_selected_patients','7_prior_runs_overlap_only','full_EOF_cached_sources','frozen_threshold_regression','definition_count_nesting','candidate_cached_NT_translation',_private_literal('private_5ce1ee9ae0939c19'),'thinning_probabilities_eligible_only'])
    (out/'numeric_freeze.json').write_text(json.dumps(freeze,indent=2)+'\n')
    report(out,patients,fig,sens,recovery,source)
    manifest=[]
    for p in sorted(out.glob('*')):
        if p.is_file() and p.name not in ['output_manifest.tsv','audit_receipt.json']:
            manifest.append(dict(path=str(p),bytes=p.stat().st_size,sha256=sha(p)))
    write(pd.DataFrame(manifest),out/'output_manifest.tsv')
    receipt=dict(status='passed',numeric_freeze_sha256=sha(out/'numeric_freeze.json'),input_manifest_sha256=sha(out/'input_manifest.tsv'),output_manifest_sha256=sha(out/'output_manifest.tsv'),script_sha256=sha(Path(__file__)),cached_sources_only=True,scientific_gate='unresolved')
    (out/'audit_receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps(freeze,indent=2))

def report(out,patients,fig,sens,recovery,source):
    q=patients[patients.cohort=='home'];d=q[q.primary_sample_ids==_private_literal('private_d160bae090c0c274')].iloc[0]
    without=sens[(sens.cohort=='home')&(sens.scope==_private_literal('private_e3efc0198a793713'))&(sens.definition=='Q1_AA')&(sens.threshold==2)].iloc[0]
    batch=sens[(sens.cohort=='home')&(sens.scope.str.startswith('batch_'))&(sens.definition=='Q1_AA')&(sens.threshold==2)]
    (out/'CANDIDATE_EVIDENCE_REPORT.md').write_text(f"{_private_literal('private_f2694ea139ba5288')}{int(d.Q1_AA_reads)} of{int(q.Q1_AA_reads.astype(int).sum())} Q1 AA assigned reads ({float(d.Q1_AA_reads) / q.Q1_AA_reads.astype(int).sum() * 100:.4f}%). Excluding that complete patient's library leaves{int(without.n_carriers)}/{int(without.n_patients)} AA carriers at ≥2. Read concentration does not erase recurrent patient carriage; carriage does not establish raw technical authenticity. Batch-specific >=2 AA counts are {', '.join((str(r.scope) + ':' + str(r.n_carriers) + '/' + str(r.n_patients) for r in batch.itertuples()))}{_private_literal('private_a6f1aefd9679b2de')}{recovery['filenames_examined']} filenames under{recovery['roots']}, depth ≤6 and a 50,000-filename cap, pruning analysis/env/rawpayload directories. It found{recovery['filename_candidates']}{_private_literal('private_e119b1550617ebf5')}{out.parent / 'code/v6_candidate_audit.py'} --root {out.parent}\npython3 {out.parent / 'code/v6_candidate_figures.py'} --root {out.parent}\nThe first script reads cached completed source tables plus one bounded metadata inventory; it never scans raw clonotype/FASTQ bodies. numeric_freeze.json and audit_receipt.json bind inputs, protocols, code and outputs. Source inventories/report tables remain authoritative where annotations or definitions are unresolved.\n")

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,required=True)
    a=ap.parse_args();main(a.root.resolve())
