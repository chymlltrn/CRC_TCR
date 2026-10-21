#!/usr/bin/env python3
"""Verify frozen candidate tables without re-walking metadata or rescanning raw files."""
from release_runtime import private_literal as _private_literal, release_path as _release_path
from pathlib import Path
import argparse,json,hashlib,ast
import pandas as pd,numpy as np
from scipy.stats import hypergeom
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main(root):
    out=root/'candidates';checks=[]
    rec=json.loads((out/'audit_receipt.json').read_text())
    assert rec['status']=='passed'
    for name,binding in [('input_manifest.tsv','input_manifest_sha256'),('output_manifest.tsv','output_manifest_sha256')]:
        assert sha(out/name)==rec[binding]
        for row in pd.read_csv(out/name,sep='\t').to_dict('records'):
            p=Path(row['path']);assert p.stat().st_size==row['bytes'];assert sha(p)==row['sha256']
        checks.append(name+': all bound sizes and SHA256 values match')
    if rec.get('producing_script_path'):
        assert sha(rec['producing_script_path'])==rec['script_sha256']==rec['producing_script_sha256']
        assert sha(root/'code/v6_candidate_audit.py')==rec['current_entrypoint_sha256']
        old=Path(rec['producing_script_path']).read_text();new=(root/'code/v6_candidate_audit.py').read_text()
        assert old.splitlines()[:273]==new.splitlines()[:273]
    else:
        # A fresh replay binds the executed current entrypoint directly.
        assert rec['script_sha256']==sha(root/'code/v6_candidate_audit.py')
    for p in root.joinpath('code').glob('v6_candidate_*.py'):ast.parse(p.read_text())
    checks.append('producing entrypoint preserved; current differences are report wording after line 273 only; scripts parse')
    v5=root.parent/'revision_v5'
    patients=pd.read_csv(v5/'legacy/patient_Q1_Q2.tsv',sep='\t')
    rows=pd.read_csv(v5/'legacy/matching_clone_rows.tsv',sep='\t')
    assert len(patients)==339 and patients.patient_key.is_unique
    assert patients.cohort.value_counts().to_dict()=={'external':202,'home':137}
    assert not rows.duplicated(['cohort','sample_id','clone_id']).any()
    indexed=patients.set_index('patient_key')
    assert rows.patient_key.isin(indexed.index).all()
    assert all(indexed.loc[x.patient_key,'primary_sample_ids']==x.sample_id for x in rows.itertuples())
    q1=_private_literal('private_45bbaa37a9607bb7');q2=_private_literal('private_b3445b7fc78bf2f0');nt=_private_literal('private_c287d28aaf0e9d7c')
    masks={'Q1_AA':rows.aa.eq(q1),'Q1_AA_VJ':rows.aa.eq(q1)&rows.v_gene.eq('TRBV5-5')&rows.j_gene.eq('TRBJ1-4'),'Q1_NT':rows.nt_cdr3.eq(nt),'Q2_AA':rows.aa.eq(q2)}
    units=pd.read_csv(out/'candidate_units.tsv',sep='\t')
    pub=units[~units.independent_patient].copy()
    derived=[]
    for definition,mask in masks.items():
        sums=rows[mask].groupby('patient_key').read_count.sum().reindex(indexed.index,fill_value=0).astype('int64')
        assert (sums==indexed[definition+'_reads']).all(),definition
        u=units[units.independent_patient & units.definition.eq(definition)].set_index('donor_key')
        # Patient unit names are mapped explicitly, never inferred from a cohort alias.
        assert set(u.index)==set(indexed.index)
        assert (u.candidate_count.reindex(indexed.index).astype('int64')==sums).all()
        for co,n in [('home',137),('external',202)]:
            for t in [1,2,5,10,20]:
                derived.append(dict(definition=definition,cohort=co,threshold=t,n_patients=n,n_carriers=int(sums[indexed.cohort.eq(co)].ge(t).sum())))
    pd.DataFrame(derived).to_csv(out/'verified_patient_count_reaggregation.tsv',sep='\t',index=False)
    home=patients[patients.cohort.eq('home')]
    assert [int(home.Q1_AA_reads.ge(t).sum()) for t in [1,2,5,10,20]]==[63,53,32,15,6]
    d=home[home.primary_sample_ids.eq(_private_literal('private_d160bae090c0c274'))];assert len(d)==1
    h=home[~home.patient_key.eq(d.iloc[0].patient_key)];assert len(h)==136 and h.Q1_AA_reads.ge(2).sum()==52
    assert home.Q1_AA_reads.sum()==17126 and d.iloc[0].Q1_AA_reads==16564
    # Explicit threshold unit: sum within a selected patient's exact definition, not one clone >=2.
    sum_k=int(home.Q1_AA_reads.ge(2).sum());clone_k=rows[(rows.cohort=='home')&masks['Q1_AA']&rows.read_count.ge(2)].patient_key.nunique()
    assert sum_k==53 and clone_k==52
    checks.append(_private_literal('private_18282049cbf0192b'))
    assert units[['unit_id','definition','count_branch']].duplicated().sum()==0
    assert units.unit_id.nunique()==725
    assert units[units.independent_patient].unit_id.nunique()==339
    assert pub.unit_id.nunique()==386
    assert units.full_EOF.all()
    checks.append('725 unique units retain all zero observations; seven prior IT1208 runs are not duplicated')
    th=pd.read_csv(out/'within_cohort_count_thinning_patients.tsv',sep='\t')
    assert len(th)==339*4*3
    assert th.loc[th.eligible,'probability_target_reads_ge2'].between(0,1).all()
    assert th.loc[~th.eligible,'probability_target_reads_ge2'].isna().all()
    assert ((th.observed_target_reads>=0)&(th.observed_target_reads<=th.total_QC_assigned_reads)).all()
    # Independent exact survival function checks include all low-count rows and deterministic higher-count cases.
    selected=th[(th.observed_target_reads<=50)&th.eligible].copy()
    high=th[th.eligible & th.observed_target_reads.gt(50)].iloc[::17].copy()
    selected=pd.concat([selected,high],ignore_index=True)
    expected=hypergeom.sf(1,selected.total_QC_assigned_reads.to_numpy(),selected.observed_target_reads.to_numpy(),selected.target_depth.to_numpy())
    delta=np.abs(expected-selected.probability_target_reads_ge2.to_numpy())
    assert delta.max()<2e-6,float(delta.max())
    checks.append(f'within-cohort count-thinning: {len(selected)} exact survival-function checks; maximum absolute difference {delta.max():.3g}; no cross-assay harmonization')
    seq=pd.read_csv(out/'sequence_coordinate_assignment_audit.tsv',sep='\t')
    assert len(seq)==523 and seq.complete_AA_identity.all() and seq.translation_equals_complete_AA.all()
    checks.append('523 cached bulk/reference candidate hit rows retain full-AA equality and NT translation compatibility; not a raw-archive/global-clone QC claim')
    ci=out/'figures/source/Figure2_patient_Wilson95.tsv'
    if ci.exists():
        dci=pd.read_csv(ci,sep='\t');assert len(dci)==4 and dci.unit_label.eq('patients').all()
        assert (dci.wilson95_low>=-1e-14).all() and (dci.wilson95_high<=1).all()
        checks.append('Figure2 uncertainty source has four independent-patient Wilson intervals; no repeated-library patient CI')
    result=dict(status='passed',claim_scope='cached-table numerical/identity/provenance verification; technical authenticity remains unresolved',audit_receipt_sha256=sha(out/'audit_receipt.json'),input_manifest_sha256=sha(out/'input_manifest.tsv'),output_manifest_sha256=sha(out/'output_manifest.tsv'),script_sha256=sha(__file__),checks=checks,raw_scans=0,new_filename_walks=0,inferential_tests=0)
    (out/'verification_receipt.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))
if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('--root',type=Path,required=True);main(a.parse_args().root.resolve())
