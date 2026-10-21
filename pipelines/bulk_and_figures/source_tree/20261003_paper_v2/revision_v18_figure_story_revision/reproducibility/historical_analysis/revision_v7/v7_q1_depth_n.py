#!/usr/bin/env python3
"""Patient-level Q1/Q2 association and finite-count depth counterfactual."""
from release_runtime import private_literal as _private_literal, release_path as _release_path
import os
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS']: os.environ[k]='1'
import argparse, hashlib, json, sys, math, warnings, platform
from pathlib import Path
import numpy as np
import pandas as pd
import scipy
from scipy.special import expit, logsumexp
from scipy.stats import fisher_exact, chi2, hypergeom
from scipy.stats.contingency import odds_ratio
from scipy.optimize import minimize, brentq
from statsmodels.stats.multitest import multipletests

ROOT=Path(_release_path('path_53d7a6a979f86dbd'))
OUT=ROOT/'q1_depth_n'; PARENT=ROOT.parent; V1=PARENT.parent/'20261003_paper_v1'; V6=PARENT/'revision_v6'
SEED=20261004; DEFS=('Q1_AA','Q1_AA_VJ','Q1_NT','Q2_AA'); THRS=(1,2,5,10,20)
SOURCES={'candidate_counts':PARENT/'revision_v5/legacy/patient_Q1_Q2.tsv','clinical':PARENT/'cohort/patient_master.tsv','features':V1/'features/patient_features.tsv','v6_reaggregation':V6/'candidates/verified_patient_count_reaggregation.tsv','v6_candidate_units':V6/'candidates/candidate_units.tsv','v6_Le2017':V6/'candidates/Le2017_within_reference_recurrence.tsv','v6_qpcr':V6/'candidates/qpcr_reported_results.tsv','v6_qpcr_inventory':V6/'candidates/qpcr_source_inventory.tsv','v6_definition_registry':V6/'candidates/definition_registry.tsv','original_ridge_code':V1/'code/run_ridge.py','original_ridge_folds':V1/'models/fold_audit.tsv','original_ridge_oof':V1/'models/oof_patient_ensemble.tsv','root_protocol':ROOT/'PROTOCOL_v7.md','lane_protocol':OUT/'PROTOCOL.md'}
def read(p): return pd.read_csv(p,sep='\t')
def write(d,p): Path(p).parent.mkdir(parents=True,exist_ok=True);d.to_csv(p,sep='\t',index=False)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def clean_obj(x):
    if isinstance(x,dict):return {str(k):clean_obj(v) for k,v in x.items()}
    if isinstance(x,(tuple,list)):return [clean_obj(v) for v in x]
    if isinstance(x,(np.integer,)):return int(x)
    if isinstance(x,(np.floating,float)):return float(x) if np.isfinite(x) else None
    if isinstance(x,(np.bool_,)):return bool(x)
    return x

def dump(o,p):Path(p).write_text(json.dumps(clean_obj(o),ensure_ascii=False,indent=2,allow_nan=False))
def bh(p,fixed_family=False):
    a=pd.to_numeric(p,errors='coerce'); out=np.full(len(a),np.nan);mask=np.isfinite(a)
    if fixed_family:
        out=multipletests(np.where(mask,a,1.),method='fdr_bh')[1]
        out[~mask]=np.nan
    elif mask.any():out[mask]=multipletests(a[mask],method='fdr_bh')[1]
    return out

def prepare():
    OUT.mkdir(exist_ok=True);write(pd.DataFrame([{'label':k,'path':str(p),'bytes':p.stat().st_size,'sha256':sha(p)} for k,p in SOURCES.items()]),OUT/'input_manifest.tsv')
    counts=read(SOURCES['candidate_counts']);clinical=read(SOURCES['clinical']);features=read(SOURCES['features'])
    assert counts.patient_key.is_unique and clinical.patient_key.is_unique and features.patient_key.is_unique
    assert set(counts.patient_key)==set(clinical.patient_key)==set(features.patient_key)
    f=features.copy();other=counts.set_index('patient_key').loc[f.patient_key]
    assert np.array_equal(f.total_reads.to_numpy(),other.primary_total_reads.to_numpy())
    for col in ['cohort','batch','n_binary','primary_sample_ids']:
        assert np.array_equal(f[col].fillna('NA').to_numpy(),other[col].fillna('NA').to_numpy()),col
    for definition in DEFS:f[definition+'_reads']=other[definition+'_reads'].to_numpy(int)
    f['primary_total_reads']=f.total_reads
    for d in DEFS:f[d+'_carrier_ge2']=(f[d+'_reads']>=2).astype(int)
    f['known_N']=f.n_binary.isin([0,1]);f['eligible_Home_N']=f.known_N&(f.cohort=='home')
    f['N_exclusion_reason']=np.where(f.eligible_Home_N,'included',np.where(f.cohort!='home','External_descriptive_or_frozen_transport_only','unknown_recorded_N'))
    f[_private_literal('private_044ace73522ed623')]=f.primary_sample_ids.eq(_private_literal('private_d160bae090c0c274'))
    cols=['patient_key','cohort','batch','n_binary','age_years','sex','neoadjuvant','primary_sample_ids','primary_total_reads','unique_aa','log1p_reads','log1p_unique_aa','known_N','eligible_Home_N','N_exclusion_reason',_private_literal('private_044ace73522ed623')]+[d+suffix for d in DEFS for suffix in ['_reads','_carrier_ge2']]
    write(f[cols],OUT/'patient_analysis_master.tsv');write(f,OUT/'model_patient_features.tsv')
    assert f.eligible_Home_N.sum()==126 and f.loc[f.eligible_Home_N,'n_binary'].sum()==56
    assert f.cohort.value_counts().to_dict()=={'external':202,'home':137}
    assert getattr(f, _private_literal('private_044ace73522ed623')).sum()==1
    checks=[]
    re=read(SOURCES['v6_reaggregation'])
    for _,r in re.iterrows():
        g=f[f.cohort==r.cohort];k=(g[str(r.definition)+'_reads']>=r.threshold).sum()
        assert k==r.n_carriers and len(g)==r.n_patients
        checks.append({'definition':r.definition,'cohort':r.cohort,'threshold':r.threshold,'recount_n':len(g),'recount_carriers':int(k),'status':'PASS'})
    write(pd.DataFrame(checks),OUT/'v6_recount_checks.tsv')
    fields=['age_years','sex','neoadjuvant','batch','log1p_reads','log1p_unique_aa']
    miss=[]
    for scope,g in [(_private_literal('private_7e027ebbcb151784'),f[f.eligible_Home_N]),(_private_literal('private_aa2c97c2247fa9b0'),f[f.cohort=='home']),('External202',f[f.cohort=='external'])]:
        for c in fields:miss.append({'scope':scope,'covariate':c,'n':len(g),'missing':int(g[c].isna().sum()),'values':str(g[c].value_counts(dropna=False).to_dict()) if c in ['sex','neoadjuvant','batch'] else ''})
    write(pd.DataFrame(miss),OUT/'covariate_inventory.tsv')
    dump({'state':'inputs_frozen','seed':SEED,'python':sys.version,'numpy':np.__version__,'pandas':pd.__version__,'scipy':scipy.__version__,'platform':platform.platform(),'input_manifest_sha256':sha(OUT/'input_manifest.tsv'),'protocol_sha256':sha(OUT/'PROTOCOL.md')},OUT/'input_receipt.json')
    print('Prepared339patient rows and all frozen recounts PASS',flush=True)

def wilson(k,n):
    z=1.959963984540054;p=k/n;den=1+z*z/n;c=(p+z*z/(2*n))/den;h=z*np.sqrt(p*(1-p)/n+z*z/(4*n*n))/den
    return max(0,c-h),min(1,c+h)

def firth_objective(beta,X,y):
    eta=X@beta;p=expit(eta);w=p*(1-p);info=X.T@(w[:,None]*X)
    sign,ld=np.linalg.slogdet(info)
    if sign<=0 or not np.isfinite(ld):return np.inf,np.zeros(X.shape[1])
    ll=np.sum(y*eta-np.logaddexp(0,eta))+.5*ld
    inv=np.linalg.inv(info);h=w*np.einsum('ij,jk,ik->i',X,inv,X)
    score=X.T@(y-p+h*(.5-p))
    return -ll,-score

def firth_fit(X,y,fixed=None,start=None):
    p=X.shape[1];b=np.zeros(p) if start is None else np.asarray(start).copy()
    free=np.arange(p)
    if fixed is not None:
        j,value=fixed;b[j]=value;free=free[free!=j]
    def fun(v):
        beta=b.copy();beta[free]=v;f,grad=firth_objective(beta,X,y);return f,grad[free]
    fit=minimize(fun,b[free],method='BFGS',jac=True,options={'gtol':1e-8,'maxiter':1500})
    b[free]=fit.x;val,grad=fun(fit.x)
    if not np.isfinite(val) or np.max(np.abs(grad))>2e-5:raise RuntimeError('Firth fit did not converge: '+str(fit.message)+' grad='+str(np.max(np.abs(grad))))
    return b,-float(val),int(fit.nit),float(np.max(np.abs(grad)))

def firth_profile(X,y,j=1):
    beta,ll,nit,grad=firth_fit(X,y);beta0,ll0,_,_=firth_fit(X,y,(j,0),beta)
    lr=max(0,2*(ll-ll0));target=chi2.ppf(.95,1)
    def fn(value):return 2*(ll-firth_fit(X,y,(j,value),beta)[1])-target
    endpoints=[]
    for s in [-1,1]:
        distance=1.;end=beta[j]+s*distance
        while fn(end)<0 and distance<32:
            distance*=1.7;end=beta[j]+s*distance
        if fn(end)<0:endpoints.append(np.nan)
        else:endpoints.append(brentq(fn,min(beta[j],end),max(beta[j],end),xtol=1e-7))
    return {'adjusted_log_OR':beta[j],'adjusted_OR':np.exp(beta[j]),'adjusted_OR_low':np.exp(endpoints[0]),'adjusted_OR_high':np.exp(endpoints[1]),'adjusted_P':chi2.sf(lr,1),'adjusted_LR':lr,'adjusted_fit_iterations':nit,'adjusted_max_abs_gradient':grad,'adjusted_status':'ok','adjusted_CI':'profile_penalized_likelihood'}

def design(g,carrier):
    cols=[np.ones(len(g)),np.asarray(carrier,float)];names=['intercept','candidate_carriage'];excluded=[]
    for c in ['age_years','log1p_reads','log1p_unique_aa']:
        a=g[c].to_numpy(float)
        if np.std(a)<1e-12:excluded.append(c);continue
        cols.append((a-np.mean(a))/np.std(a));names.append(c)
    for c in ['sex','batch','neoadjuvant']:
        vals=g[c].fillna('Missing').astype(str);lev=sorted(vals.unique())
        for l in lev[1:]:cols.append(vals.eq(l).to_numpy(float));names.append(c+'='+l)
        if len(lev)==1:excluded.append(c)
    X=np.column_stack(cols)
    return X,names,excluded

def association(g,definition,threshold,scope):
    carrier=(g[definition+'_reads']>=threshold).to_numpy(int);y=g.n_binary.to_numpy(int)
    a=int(np.sum((y==1)&(carrier==1)));b=int(np.sum((y==1)&(carrier==0)));c=int(np.sum((y==0)&(carrier==1)));d=int(np.sum((y==0)&(carrier==0)))
    table=np.array([[a,b],[c,d]]);n1=a+b;n0=c+d;ci1=wilson(a,n1);ci0=wilson(c,n0)
    row={'scope':scope,'definition':definition,'threshold':threshold,'candidate':'Q1' if definition.startswith('Q1') else 'Q2','n_patients':len(g),'Nplus_n':n1,'Nplus_carriers':a,'Nplus_prevalence':a/n1,'Nplus_prevalence_low':ci1[0],'Nplus_prevalence_high':ci1[1],'N0_n':n0,'N0_carriers':c,'N0_prevalence':c/n0,'N0_prevalence_low':ci0[0],'N0_prevalence_high':ci0[1],'risk_difference':a/n1-c/n0,'fisher_P':fisher_exact(table).pvalue,'unadjusted_OR_sample':fisher_exact(table).statistic,'selection_status':'retrospective_previously_nominated_candidates'}
    oo=odds_ratio(table,kind='conditional');oc=oo.confidence_interval();row.update(unadjusted_OR=oo.statistic,unadjusted_OR_low=oc.low,unadjusted_OR_high=oc.high)
    rng=np.random.default_rng(SEED)
    # Resampling binary patient carriers is exactly binomial resampling within outcome groups.
    rd=rng.binomial(n1,a/n1,10000)/n1-rng.binomial(n0,c/n0,10000)/n0
    row.update(risk_difference_low=np.quantile(rd,.025),risk_difference_high=np.quantile(rd,.975),risk_difference_CI='outcome_stratified_patient_percentile10000')
    complete=g[['age_years','log1p_reads','log1p_unique_aa']].notna().all(axis=1).to_numpy()
    gg=g.loc[complete];cc=carrier[complete];yy=y[complete]
    row.update(adjusted_n=len(gg),covariate_missing_excluded=int((~complete).sum()))
    if min(np.sum(yy==0),np.sum(yy==1))<5 or min(np.sum(cc==0),np.sum(cc==1))<3:
        row.update(adjusted_status='not_estimable_prespecified_sparse_gate');return row
    X,names,dropped=design(gg,cc);row.update(adjusted_covariates='|'.join(names),constant_covariates_dropped='|'.join(dropped),adjusted_parameters=X.shape[1])
    if np.linalg.matrix_rank(X)<X.shape[1]:row.update(adjusted_status='not_estimable_rank_deficient');return row
    try:row.update(firth_profile(X,yy))
    except Exception as e:row.update(adjusted_status='fit_failure',adjusted_error=repr(e))
    return row

def associations():
    f=read(OUT/'patient_analysis_master.tsv');h=f[f.eligible_Home_N].copy();rows=[]
    scopes=[(_private_literal('private_7e027ebbcb151784'),h),(_private_literal('private_044ace73522ed623'),h[~getattr(h, _private_literal('private_044ace73522ed623'))]),('recorded_untreated',h[h.neoadjuvant=='no'])]+[('batch_'+str(b),g) for b,g in h.groupby('batch')]
    for scope,g in scopes:
        for definition in DEFS:
            for threshold in THRS:
                row=association(g,definition,threshold,scope);rows.append(row)
        print('association complete '+scope,flush=True)
    results=pd.DataFrame(rows);mask=(results.scope==_private_literal('private_7e027ebbcb151784'))&(results.threshold==2)&results.definition.isin(['Q1_AA','Q2_AA']);primary=results[mask].copy()
    primary['crude_BH_q']=bh(primary.fisher_P);primary['adjusted_BH_q']=bh(primary.adjusted_P);primary['crude_BH_family']='2_nominated_candidates_primary_AA_ge2';primary['adjusted_BH_family']='2_nominated_candidates_primary_AA_ge2_Firth'
    results['sensitivity_BH_q']=np.nan;results.loc[~mask,'sensitivity_BH_q']=bh(results.loc[~mask,'adjusted_P'],fixed_family=True);results['is_primary']=mask;results['sensitivity_BH_family']='fixed118_nonprimary_tests_NA_P_as1_then_restore_NA_q';results['sensitivity_BH_family_n']=118
    write(primary,OUT/'N_association_primary.tsv');write(results,OUT/'N_association_sensitivity.tsv')
    # External carries no Q1: preserve zero and distinguish an unavailable binary contrast from a null fit.
    ext=f[(f.cohort=='external')&f.known_N]
    descriptions=[]
    for definition in DEFS:
        for threshold in THRS:
            for n,g in ext.groupby('n_binary'):
                k=int((g[definition+'_reads']>=threshold).sum());lo,hi=wilson(k,len(g));descriptions.append({'scope':'External163','definition':definition,'threshold':threshold,'n_binary':n,'n':len(g),'carriers':k,'prevalence':k/len(g),'low':lo,'high':hi})
    write(pd.DataFrame(descriptions),OUT/'External_N_carriage_description.tsv')
    dump({'state':'complete','primary_tests':len(primary),'sensitivity_grid':len(results),'primary_adjusted_BH_n':2,'sensitivity_BH_family_n':int((~mask).sum()),'sensitivity_BH_NA_handling':'P_as1_for_fixed_family_then_restore_NA_q','sensitivity_BH_tested_n':int(results.loc[~mask,'adjusted_P'].notna().sum()),'fit_failures':int(results.adjusted_status.eq('fit_failure').sum()),'NA_endpoints':int(results.adjusted_P.isna().sum()),'all_patient_units_unique':True},OUT/'association_receipt.json')
    print(primary[['candidate','Nplus_carriers','Nplus_n','N0_carriers','N0_n','adjusted_OR','adjusted_OR_low','adjusted_OR_high','adjusted_P','adjusted_BH_q']].to_string(index=False),flush=True)

def capture_probs(N,K,M,t):
    N=np.asarray(N,dtype=np.int64);K=np.asarray(K,dtype=np.int64);M=np.asarray(M,dtype=np.int64)
    eligible=M<=N;out=np.zeros(np.broadcast_shapes(N.shape,K.shape,M.shape),float)
    nn,kk,mm=np.broadcast_arrays(N,K,M);valid=eligible&(kk>=t)
    if valid.any():
        # Summing only 0..threshold-1 is exact and bounded even for deep clone counts.
        terms=np.stack([hypergeom.logpmf(x,nn[valid],kk[valid],mm[valid]) for x in range(t)])
        out[valid]=np.clip(-np.expm1(logsumexp(terms,axis=0)),0,1)
    out[~eligible]=np.nan
    return out

def depth():
    folder=OUT/'depth';folder.mkdir(exist_ok=True);f=read(OUT/'patient_analysis_master.tsv');h=f[f.cohort=='home'].reset_index(drop=True);e=f[f.cohort=='external'].reset_index(drop=True)
    depths=e.primary_total_reads.to_numpy(np.int64);N=h.primary_total_reads.to_numpy(np.int64);common=N>=depths.max();print('depth common support',int(common.sum()),'of',len(h),'maxExternal',int(depths.max()),flush=True)
    write(pd.DataFrame([{'cohort':co,'n':len(g),'min_depth':int(g.primary_total_reads.min()),'q25_depth':g.primary_total_reads.quantile(.25),'median_depth':g.primary_total_reads.median(),'q75_depth':g.primary_total_reads.quantile(.75),'max_depth':int(g.primary_total_reads.max()),'count_unit':'MiXCR_assigned_QC_reads'} for co,g in f.groupby('cohort')]),folder/'depth_distribution.tsv')
    observed=[]
    for co,g in f.groupby('cohort'):
        for definition in DEFS:
            for t in THRS:
                k=int((g[definition+'_reads']>=t).sum());lo,hi=wilson(k,len(g));observed.append({'cohort':co,'definition':definition,'threshold':t,'n_patients':len(g),'carriers':k,'prevalence':k/len(g),'low':lo,'high':hi})
    write(pd.DataFrame(observed),folder/'observed_source_carriage.tsv')
    eligibility=h[['patient_key','primary_sample_ids','batch','primary_total_reads',_private_literal('private_044ace73522ed623')]].copy();eligibility['eligible_all_External_depths']=common;eligibility['eligible_target_depths']=(N[:,None]>=depths).sum(axis=1);write(eligibility,folder/'Home_common_depth_support.tsv')
    summaries=[];targetrows=[];gridrows=[];patients=[];bootstraprows=[]
    rng=np.random.default_rng(SEED);bootstrap_draws={}
    scopes=[('Home_all_common_support',common),(_private_literal('private_d54419bd68c77be2'),common&~getattr(h, _private_literal('private_044ace73522ed623')).to_numpy())]+[('Home_'+str(b)+'_common_support',common&h.batch.eq(b).to_numpy()) for b in sorted(h.batch.unique())]
    for definition in DEFS:
        K=h[definition+'_reads'].to_numpy(np.int64)
        for t in THRS:
            prob=capture_probs(N[:,None],K[:,None],depths[None,:],t)
            for i in range(len(h)):
                patients.append({'definition':definition,'threshold':t,'patient_key':h.loc[i,'patient_key'],'sample_id':h.loc[i,'primary_sample_ids'],'batch':h.loc[i,'batch'],'source_N':N[i],'source_K':K[i],'eligible_all_targets':bool(common[i]),'eligible_targets':int(np.isfinite(prob[i]).sum()),'expected_detection_across_eligible_targets':np.nanmean(prob[i])})
            rr,cc=np.indices(prob.shape)
            gridrows.append(pd.DataFrame({'definition':definition,'threshold':t,'Home_patient_key':h.patient_key.to_numpy()[rr.ravel()],'External_patient_key':e.patient_key.to_numpy()[cc.ravel()],'source_N':N[rr.ravel()],'source_K':K[rr.ravel()],'target_m':depths[cc.ravel()],'eligible_no_upsampling':np.isfinite(prob.ravel()),'capture_probability':prob.ravel()}))
            for scope,mask in scopes:
                a=prob[mask];n=len(a)
                if not n:continue
                assert np.isfinite(a).all()
                pp=a.mean(axis=0);meanpatient=a.mean(axis=1)
                # A single reference-patient bootstrap supplies a coherent empirical population at all target depths.
                if n not in bootstrap_draws:bootstrap_draws[n]=rng.multinomial(n,np.repeat(1/n,n),size=10000)/n
                weights=bootstrap_draws[n];pboot=weights@a;expected=pboot.sum(axis=1);zero_log=np.log1p(-np.clip(pboot,0,1-1e-15)).sum(axis=1);zero=np.exp(zero_log)
                logzero=float(np.log1p(-pp).sum());z=float(np.exp(logzero))
                # Independent resampling of External empirical target depths is a separate uncertainty sensitivity.
                extweights=rng.multinomial(len(e),np.repeat(1/len(e),len(e)),size=10000)
                twoexpected=(pboot*extweights).sum(axis=1);twozero=np.exp((np.log1p(-np.clip(pboot,0,1-1e-15))*extweights).sum(axis=1))
                sm={'scope':scope,'definition':definition,'threshold':t,'n_Home_reference':n,'n_External_targets':len(e),'Home_original_carriers':int((K[mask]>=t).sum()),'Home_original_prevalence':float((K[mask]>=t).mean()),'expected_External_carriers':float(pp.sum()),'expected_External_carriers_low':float(np.quantile(expected,.025)),'expected_External_carriers_high':float(np.quantile(expected,.975)),'expected_External_prevalence':float(pp.mean()),'expected_External_prevalence_low':float(np.quantile(expected,.025)/len(e)),'expected_External_prevalence_high':float(np.quantile(expected,.975)/len(e)),'probability_zero_External_carriers':z,'probability_zero_low':float(np.quantile(zero,.025)),'probability_zero_high':float(np.quantile(zero,.975)),'log_probability_zero':logzero,'two_sample_expected_low':float(np.quantile(twoexpected,.025)),'two_sample_expected_high':float(np.quantile(twoexpected,.975)),'two_sample_zero_low':float(np.quantile(twozero,.025)),'two_sample_zero_high':float(np.quantile(twozero,.975)),'bootstrap_n':10000,'uncertainty':'Home_patient_bootstrap_External_depths_fixed','observed_External_carriers':int((e[definition+'_reads']>=t).sum()),'interpretation':'conditional_on_Home_abundance_transport_and_comparable_assigned_count_capture_not_cause_identification'}
                summaries.append(sm)
                for j in range(len(e)):targetrows.append({'scope':scope,'definition':definition,'threshold':t,'External_patient_key':e.loc[j,'patient_key'],'target_m':depths[j],'n_Home_reference':n,'expected_capture_probability':pp[j]})
                if scope in ['Home_all_common_support',_private_literal('private_d54419bd68c77be2')]:
                    bootstraprows.append(pd.DataFrame({'scope':scope,'definition':definition,'threshold':t,'repetition':np.arange(10000),'expected_External_carriers':expected,'probability_zero':zero,'log_probability_zero':zero_log,'two_sample_expected_carriers':twoexpected,'two_sample_probability_zero':twozero}))
            pp=np.nanmean(prob,axis=0)
            summaries.append({'scope':'pair_eligible_reference_sensitivity','definition':definition,'threshold':t,'n_Home_reference':len(h),'n_External_targets':len(e),'min_Home_reference_per_target':int(np.isfinite(prob).sum(axis=0).min()),'max_Home_reference_per_target':int(np.isfinite(prob).sum(axis=0).max()),'expected_External_carriers':pp.sum(),'expected_External_prevalence':pp.mean(),'probability_zero_External_carriers':float(np.exp(np.log1p(-pp).sum())),'log_probability_zero':float(np.log1p(-pp).sum()),'interpretation':'source_population_can_vary_by_target_depth; no_common_population_CI'})
            print('depth complete',definition,t,flush=True)
    write(pd.DataFrame(summaries),folder/'counterfactual_summary.tsv');write(pd.DataFrame(targetrows),folder/'external_target_probabilities.tsv');write(pd.DataFrame(patients),folder/'Home_patient_capture_probabilities.tsv')
    pd.concat(gridrows,ignore_index=True).to_parquet(folder/'Home_by_External_capture_grid.parquet',index=False);pd.concat(bootstraprows,ignore_index=True).to_parquet(folder/'counterfactual_bootstrap.parquet',index=False)
    # Fixed depth curves include all patients eligible for the requested depth; denominators explicitly vary.
    curves=[]
    for target in [50000,100000,250000,500000,1000000,2000000,4000000]:
        for definition in DEFS:
            k=h[definition+'_reads'].to_numpy(np.int64);p=capture_probs(N,k,np.full(len(h),target,dtype=np.int64),2);a=p[np.isfinite(p)]
            if not len(a):continue
            weights=rng.multinomial(len(a),np.repeat(1/len(a),len(a)),size=10000)/len(a);v=weights@a
            curves.append({'definition':definition,'threshold':2,'target_depth':target,'n_depth_eligible':len(a),'expected_prevalence':a.mean(),'low':np.quantile(v,.025),'high':np.quantile(v,.975),'count_unit':'MiXCR_assigned_QC_reads'})
    write(pd.DataFrame(curves),folder/'within_Home_depth_curves.tsv')
    dump({'state':'complete','matrix_rows':sum(len(x) for x in gridrows),'definitions':DEFS,'thresholds':THRS,'Home_common_support':int(common.sum()),'Home_total':len(h),'External_targets':len(e),'seed':SEED,'bootstrap_repetitions':10000},folder/'receipt.json')

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('phase',choices=['prepare','associations','depth']);args=parser.parse_args()
    {'prepare':prepare,'associations':associations,'depth':depth}[args.phase]()
