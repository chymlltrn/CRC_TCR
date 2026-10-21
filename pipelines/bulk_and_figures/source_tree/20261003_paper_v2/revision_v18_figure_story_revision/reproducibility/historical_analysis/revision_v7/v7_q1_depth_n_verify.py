#!/usr/bin/env python3
"""Independent arithmetic and design checks for the candidate/depth lane."""
from release_runtime import private_literal as _private_literal, release_path as _release_path
import os,sys
sys.dont_write_bytecode=True
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS']:os.environ[k]='1'
from pathlib import Path
import argparse,importlib.util,json,hashlib
import numpy as np,pandas as pd
from scipy.stats import hypergeom, fisher_exact
from scipy.optimize import check_grad
from statsmodels.stats.multitest import multipletests
ROOT=Path(_release_path('path_53d7a6a979f86dbd'));OUT=ROOT/'q1_depth_n'
spec=importlib.util.spec_from_file_location('qa',ROOT/'code/v7_q1_depth_n.py');qa=importlib.util.module_from_spec(spec);spec.loader.exec_module(qa)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return pd.read_csv(p,sep='\t')
checks=[]
def check(name,condition,detail=''):
    checks.append({'check':name,'status':'PASS' if bool(condition) else 'FAIL','detail':str(detail)})
    if not condition:raise AssertionError(name+' '+str(detail))

def run(models=False):
    master=read(OUT/'patient_analysis_master.tsv');home=master[(master.cohort=='home')&master.known_N];prim=read(OUT/'N_association_primary.tsv');sens=read(OUT/'N_association_sensitivity.tsv');manifest=read(OUT/'input_manifest.tsv')
    for _,r in manifest.iterrows():check('source_hash_'+r.label,sha(r.path)==r.sha256)
    check('unique_patient_master',master.patient_key.is_unique)
    check('frozen_N_denominator',len(home)==126 and home.n_binary.sum()==56)
    for _,r in prim.iterrows():
        c=(home[r.definition+'_reads']>=2).to_numpy();y=home.n_binary.to_numpy(int);a=int(np.sum(c&(y==1)));b=int(np.sum(~c&(y==1)));c0=int(np.sum(c&(y==0)));d=int(np.sum(~c&(y==0)))
        check('counts_'+r.candidate,(a,c0,a+b,c0+d)==(r.Nplus_carriers,r.N0_carriers,r.Nplus_n,r.N0_n))
        check('fisher_'+r.candidate,abs(fisher_exact([[a,b],[c0,d]]).pvalue-r.fisher_P)<1e-12)
        # Analytical Firth binary-only estimate equals the half-cell adjusted odds ratio.
        X=np.column_stack([np.ones(len(home)),c.astype(float)]);bb,ll,_,_=qa.firth_fit(X,y)
        check('Firth_binary_closed_form_'+r.candidate,abs(np.exp(bb[1])-(a+.5)*(d+.5)/((b+.5)*(c0+.5)))<1e-6)
        X,names,_=qa.design(home,c);rng=np.random.default_rng(29);bet=rng.normal(0,.1,X.shape[1]);err=check_grad(lambda b:qa.firth_objective(b,X,y)[0],lambda b:qa.firth_objective(b,X,y)[1],bet)
        check('Firth_analytic_gradient_'+r.candidate,err<2e-5,err)
        beta,ll,_,_=qa.firth_fit(X,y);check('Firth_point_recompute_'+r.candidate,abs(beta[1]-r.adjusted_log_OR)<1e-7)
        for label,val in [('lower',r.adjusted_OR_low),('upper',r.adjusted_OR_high)]:
            pll=qa.firth_fit(X,y,(1,np.log(val)),beta)[1];deviance=2*(ll-pll)
            check('Firth_profile_endpoint_'+r.candidate+'_'+label,abs(deviance-3.841458820694124)<1e-6,deviance)
    check('primary_BH2_crude',np.allclose(prim.crude_BH_q,multipletests(prim.fisher_P,method='fdr_bh')[1]))
    check('primary_BH2_adjusted',np.allclose(prim.adjusted_BH_q,multipletests(prim.adjusted_P,method='fdr_bh')[1]))
    check('complete_sensitivity_grid',len(sens)==4*5*6 and not sens.duplicated(['scope','definition','threshold']).any())
    family=~sens.is_primary;idx=family&sens.adjusted_P.notna()
    check('sensitivity_fixed118_grid',int(family.sum())==118)
    expected=multipletests(sens.loc[family,'adjusted_P'].fillna(1.),method='fdr_bh')[1]
    check('sensitivity_BH_full_family',np.allclose(sens.loc[idx,'sensitivity_BH_q'],expected[sens.loc[family,'adjusted_P'].notna()]))
    check('sensitivity_BH_missing_restored_NA',sens.loc[family&sens.adjusted_P.isna(),'sensitivity_BH_q'].isna().all())
    check('sensitivity_primary_separate',sens.loc[sens.is_primary,'sensitivity_BH_q'].isna().all())
    check('no_firth_failures',not sens.adjusted_status.eq('fit_failure').any())
    check('all_sparse_rows_NA',sens.loc[sens.adjusted_status.str.startswith('not_estimable'),'adjusted_P'].isna().all())
    grid=pd.read_parquet(OUT/'depth/Home_by_External_capture_grid.parquet');summary=read(OUT/'depth/counterfactual_summary.tsv');targets=read(OUT/'depth/external_target_probabilities.tsv');support=read(OUT/'depth/Home_common_depth_support.tsv');boots=pd.read_parquet(OUT/'depth/counterfactual_bootstrap.parquet')
    check('complete_Home_External_definition_threshold_grid',len(grid)==137*202*4*5)
    check('no_target_upsampling',grid.loc[grid.eligible_no_upsampling,'target_m'].le(grid.loc[grid.eligible_no_upsampling,'source_N']).all())
    check('ineligible_is_NA',grid.loc[~grid.eligible_no_upsampling,'capture_probability'].isna().all())
    check('probability_bounds',grid.capture_probability.dropna().between(0,1).all())
    # Different implementation of the hypergeometric tail checks across actual observed ranges.
    a=grid[grid.eligible_no_upsampling].sample(120,random_state=43)
    ref=hypergeom.sf(a.threshold.to_numpy()-1,a.source_N.to_numpy(),a.source_K.to_numpy(),a.target_m.to_numpy());err=np.max(np.abs(ref-a.capture_probability.to_numpy()));check('scipy_direct_tail_agreement',err<1e-6,err)
    # Exhaustive small finite populations check exact CDF construction, zero carriers and complete sampling.
    for n in [10,30,100]:
        for k in [0,1,2,n//2,n]:
            for m in [1,n//2,n]:
                for t in [1,2,5]:
                    p=qa.capture_probs(np.array([n]),np.array([k]),np.array([m]),t)[0];r=hypergeom.sf(t-1,n,k,m);check(f'hypergeom_small_{n}_{k}_{m}_{t}',abs(p-r)<1e-10)
    for (definition,t),g in grid.groupby(['definition','threshold']):
        arr=g[g.Home_patient_key.isin(support.loc[support.eligible_all_External_depths,'patient_key'])];by=arr.groupby('External_patient_key').capture_probability.mean().sort_index();s=summary[(summary.scope=='Home_all_common_support')&(summary.definition==definition)&(summary.threshold==t)].iloc[0]
        check('expected_recompute_'+definition+'_'+str(t),abs(by.sum()-s.expected_External_carriers)<1e-9)
        check('zero_probability_recompute_'+definition+'_'+str(t),abs(np.log1p(-by).sum()-s.log_probability_zero)<1e-9)
        bt=boots[(boots.scope=='Home_all_common_support')&(boots.definition==definition)&(boots.threshold==t)]
        check('bootstrap_replicate_n_'+definition+'_'+str(t),len(bt)==10000 and bt.repetition.nunique()==10000)
        check('bootstrap_CI_recompute_'+definition+'_'+str(t),abs(bt.expected_External_carriers.quantile(.025)-s.expected_External_carriers_low)<1e-9 and abs(bt.expected_External_carriers.quantile(.975)-s.expected_External_carriers_high)<1e-9)
    # The same finite repertoire cannot become more detectable at a higher read threshold.
    pivot=grid.pivot(index=['definition','Home_patient_key','External_patient_key'],columns='threshold',values='capture_probability').dropna();check('threshold_monotonicity',(np.diff(pivot.to_numpy(),axis=1)<=1e-7).all())
    if models:
        md=OUT/'models';audit=read(md/'fold_audit.tsv');oof=read(md/'oof_predictions.tsv');ens=read(md/'oof_patient_ensemble.tsv');bs=read(md/'full_pipeline_bootstrap.tsv');bc=read(md/'full_pipeline_paired_ci.tsv');status=json.loads((md/'status.json').read_text());check('models_complete',status['state']=='complete')
        check('five_models_same_patients',len(oof)==126*3*5 and len(ens)==126*5)
        check('three_OOF_predictions_per_patient_model',oof.groupby(['patient_key','model']).size().eq(3).all())
        for i,r in audit.iterrows():check('main_fold_group_separation_'+str(i),not(set(r.train_keys.split('|'))&set(r.test_keys.split('|'))))
        check('original_frozen_partition_equivalence',read(md/'original_fold_equivalence.tsv')[['train_equal','test_equal']].all().all())
        check('both_candidates_full200_bootstrap',len(bs)==400 and bs.status.eq('success').all() and bs.groupby('candidate').size().eq(200).all())
        for rep in range(200):
            aa=pd.read_parquet(md/'bootstrap_folds'/f'repetition_{rep:03d}.parquet');check('bootstrap_group_separation_'+str(rep),all(not(set(r.train_keys.split('|'))&set(r.test_keys.split('|'))) for _,r in aa.iterrows()))
        for _,r in bc.iterrows():
            g=bs[bs.candidate==r.candidate];check('bootstrap_model_CI_'+r.candidate+'_'+r.metric,np.allclose([g[r.metric].quantile(.025),g[r.metric].quantile(.975)],[r.ci_low,r.ci_high]))
        check('20_label_permutations',len(read(md/'negative_controls.tsv'))==40 and read(md/'negative_controls.tsv').status.eq('success').all())
    write=pd.DataFrame(checks);write.to_csv(OUT/('verification_all_checks.tsv' if models else 'verification_association_depth_checks.tsv'),sep='\t',index=False)
    receipt={'status':'PASS','checks':len(checks),'scope':'all' if models else 'association_depth','code_sha256':sha(__file__),'analysis_code_sha256':sha(ROOT/'code/v7_q1_depth_n.py'),'input_manifest_sha256':sha(OUT/'input_manifest.tsv'),'results_sha256':{str(p.relative_to(OUT)):sha(p) for p in [OUT/'N_association_primary.tsv',OUT/'N_association_sensitivity.tsv',OUT/'depth/counterfactual_summary.tsv',OUT/'depth/Home_by_External_capture_grid.parquet']}}
    (OUT/('verification_receipt.json' if models else 'association_depth_verification_receipt.json')).write_text(json.dumps(receipt,indent=2));print(json.dumps(receipt),flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--models',action='store_true');a=p.parse_args();run(a.models)
