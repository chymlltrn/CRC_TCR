#!/usr/bin/env python3
"""Candidate additions to frozen ridge; grouped all-step bootstrap."""
from release_runtime import private_literal as _private_literal, release_path as _release_path
import os,sys
sys.dont_write_bytecode=True
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS']:os.environ[k]='1'
from pathlib import Path
import time, json, hashlib, argparse, importlib.util, traceback
from concurrent.futures import ProcessPoolExecutor,as_completed
import pandas as pd
import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import roc_auc_score,average_precision_score,log_loss,brier_score_loss
ROOT=Path(_release_path('path_53d7a6a979f86dbd'));OUT=ROOT/'q1_depth_n';MD=OUT/'models';V1=ROOT.parent.parent/'20261003_paper_v1'
spec=importlib.util.spec_from_file_location('frozen_ridge',V1/'code/run_ridge.py');rr=importlib.util.module_from_spec(spec);spec.loader.exec_module(rr)
SEEDS=(17,29,43);MODELS=('M1','M1_Q1','M1_Q2')
def write(d,p):Path(p).parent.mkdir(exist_ok=True,parents=True);d.to_csv(p,sep='\t',index=False)
def read(p):return pd.read_csv(p,sep='\t')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def metrics_fast(y,p):
    return {'AUROC':roc_auc_score(y,p),'AUPRC':average_precision_score(y,p),'log_loss':log_loss(y,p),'Brier':brier_score_loss(y,p)}
def delta_fast(y,b,n):
    mb=metrics_fast(y,b);mn=metrics_fast(y,n)
    return {'delta_log_loss':mb['log_loss']-mn['log_loss'],'delta_AUROC':mn['AUROC']-mb['AUROC'],'delta_AUPRC':mn['AUPRC']-mb['AUPRC'],'delta_Brier':mb['Brier']-mn['Brier']}
def blocks_for(f):
    b=rr.feature_blocks(f);out={'M1':b['M1']}
    for q in ['Q1','Q2']:
        col=q+'_AA_carrier_ge2'
        out['M1_'+q]={'numeric':b['M1']['numeric']+[col],'categorical':b['M1']['categorical']}
        out[q+'_only']={'numeric':[col],'categorical':[]}
    return out

def bootstrap_worker(frame,blocks,rep):
    t=time.time()
    try:
        sample=rr.bootstrap_sample(frame,100000+rep);oof,audit=rr.nested_oof(sample,blocks,MODELS,SEEDS,5,3)
        assert audit.patient_overlap.max()==0
        audit.to_parquet(MD/'bootstrap_folds'/f'repetition_{rep:03d}.parquet',index=False)
        ens=rr.ensemble_predictions(oof);rows=[]
        for q in ['Q1','Q2']:
            y,b,n=rr.paired_array(ens,'M1','M1_'+q)
            rows.append({'repetition':rep,'candidate':q,'status':'success','n_rows':len(sample),'n_unique_patients':sample.patient_key.nunique(),'patient_overlap':int(audit.patient_overlap.max()),'max_partition_retry_count':int(audit.partition_retry_count.max()),'elapsed_seconds':time.time()-t,**delta_fast(y,b,n)})
        return rows
    except Exception as exc:return [{'repetition':rep,'candidate':q,'status':'failure','error':repr(exc),'traceback':traceback.format_exc(),'elapsed_seconds':time.time()-t} for q in ['Q1','Q2']]

def negative_worker(frame,blocks,rep):
    shuffled=frame.copy();rng=np.random.default_rng(300000+rep)
    for _,ix in shuffled.groupby('batch').groups.items():shuffled.loc[ix,'n_binary']=rng.permutation(shuffled.loc[ix,'n_binary'].to_numpy())
    try:
        oof,audit=rr.nested_oof(shuffled,blocks,MODELS,SEEDS,5,3);ens=rr.ensemble_predictions(oof);rows=[]
        for q in ['Q1','Q2']:
            y,b,n=rr.paired_array(ens,'M1','M1_'+q);rows.append({'repetition':rep,'candidate':q,'status':'success','patient_overlap':int(audit.patient_overlap.max()),'candidate_AUROC':roc_auc_score(y,n),**delta_fast(y,b,n)})
        return rows
    except Exception as exc:return [{'repetition':rep,'candidate':q,'status':'failure','error':repr(exc)} for q in ['Q1','Q2']]

def frozen_subset_oof(frame,blocks,original_audit):
    frame=frame.reset_index(drop=True);rows=[];audits=[]
    for seed in SEEDS:
        for fold in range(5):
            outer=original_audit[(original_audit.level=='outer')&(original_audit.seed==seed)&(original_audit.outer_fold==fold)].iloc[0]
            train=frame[frame.patient_key.isin(outer.train_keys.split('|'))].copy().reset_index(drop=True);test=frame[frame.patient_key.isin(outer.test_keys.split('|'))].copy()
            inner=[]
            for innerfold in range(3):
                a=original_audit[(original_audit.level=='inner')&(original_audit.seed==seed)&(original_audit.outer_fold==fold)&(original_audit.inner_fold==innerfold)].iloc[0]
                tr=np.flatnonzero(train.patient_key.isin(a.train_keys.split('|')));te=np.flatnonzero(train.patient_key.isin(a.test_keys.split('|')))
                assert len(set(train.iloc[tr].patient_key)&set(train.iloc[te].patient_key))==0
                assert len(np.unique(train.iloc[tr].n_binary))==len(np.unique(train.iloc[te].n_binary))==2
                inner.append((tr,te));audits.append({'level':'inner','seed':seed,'outer_fold':fold,'inner_fold':innerfold,'train_keys':'|'.join(sorted(train.iloc[tr].patient_key)),'test_keys':'|'.join(sorted(train.iloc[te].patient_key)),'patient_overlap':0})
            audits.append({'level':'outer','seed':seed,'outer_fold':fold,'inner_fold':-1,'train_keys':'|'.join(sorted(train.patient_key)),'test_keys':'|'.join(sorted(test.patient_key)),'patient_overlap':0})
            for model in MODELS:
                block=blocks[model];fit=GridSearchCV(rr.make_pipeline(block['numeric'],block['categorical']),{'ridge__C':list(rr.CS)},scoring='neg_log_loss',cv=inner,n_jobs=1,refit=True,error_score='raise').fit(train,train.n_binary.astype(int))
                for idx,p in zip(test.index,fit.predict_proba(test)[:,1]):
                    r=frame.loc[idx];rows.append({'row_id':idx,'patient_key':r.patient_key,'n_binary':int(r.n_binary),'batch':r.batch,'seed':seed,'fold':fold,'model':model,'probability':p,'selected_C':fit.best_params_['ridge__C']})
    return pd.DataFrame(rows),pd.DataFrame(audits)

def main():
    pa=argparse.ArgumentParser();pa.add_argument('--jobs',type=int,default=8);pa.add_argument('--bootstrap',type=int,default=200);pa.add_argument('--skip-primary',action='store_true');args=pa.parse_args()
    MD.mkdir(exist_ok=True);(MD/'bootstrap_folds').mkdir(exist_ok=True);t=time.time();f=read(OUT/'model_patient_features.tsv');home=f[(f.cohort=='home')&f.n_binary.isin([0,1])].reset_index(drop=True);external=f[(f.cohort=='external')&f.n_binary.isin([0,1])].reset_index(drop=True)
    assert len(home)==126 and home.n_binary.sum()==56;assert len(external)==163 and external.n_binary.sum()==54
    blocks=blocks_for(f);original=read(V1/'models/fold_audit.tsv');rr.json_write(MD/'model_spec.json',rr.finite_dict({'models':blocks,'n_Home':126,'n_External':163,'seeds':SEEDS,'outer':5,'inner':3,'C':rr.CS,'bootstrap_requested':args.bootstrap,'bootstrap_duplicate_identity_grouped':True,'selection':'conditional_on_historical_nomination','delta_direction':'baseline_loss_minus_candidate_added_loss_positive_improvement','code_sha256':sha(__file__),'imported_ridge_code_sha256':sha(V1/'code/run_ridge.py')}))
    if not args.skip_primary:
        print('Main OOF start',flush=True);pred,audit=rr.nested_oof(home,blocks,tuple(blocks));ens=rr.ensemble_predictions(pred);write(pred,MD/'oof_predictions.tsv');write(audit,MD/'fold_audit.tsv');write(ens,MD/'oof_patient_ensemble.tsv')
        # Directly compare the saved original patient partitions, not just seed labels.
        columns=['level','seed','outer_fold','inner_fold'];joined=original.merge(audit,on=columns,suffixes=('_old','_new'),validate='one_to_one')
        assert len(joined)==len(original)==len(audit)==60
        joined['train_equal']=joined.train_keys_old==joined.train_keys_new;joined['test_equal']=joined.test_keys_old==joined.test_keys_new
        assert joined.train_equal.all() and joined.test_equal.all();write(joined[columns+['train_equal','test_equal']],MD/'original_fold_equivalence.tsv')
        old=read(V1/'models/oof_patient_ensemble.tsv');old=old[old.model=='M1'].set_index('patient_key');new=ens[ens.model=='M1'].set_index('patient_key').loc[old.index];delta=np.max(np.abs(new.probability-old.probability));assert delta<1e-10,delta
        rr.json_write(MD/'M1_reproduction.json',{'max_absolute_probability_difference':delta,'all_60_outer_inner_partitions_identical':True,'original_OOF_sha256':sha(V1/'models/oof_patient_ensemble.tsv')})
        met=[];metricci=[];increments=[];conditional=[]
        for model,g in ens.groupby('model'):
            met.append({'model':model,'scope':_private_literal('private_c503b7b137033de5'),**rr.metrics(g.n_binary,g.probability)})
            metricci.extend(rr.conditional_metrics_ci(g.n_binary,g.probability,'Home',model,'three_seed_OOF_mean',2000,SEED if 'SEED' in globals() else 20261004))
        for q in ['Q1','Q2']:
            y,b,n=rr.paired_array(ens,'M1','M1_'+q);increments.append({'candidate':q,'baseline':'M1','model':'M1_'+q,'n':len(y),**delta_fast(y,b,n)})
            conditional.extend([{'candidate':q,'baseline':'M1','model':'M1_'+q,**r} for r in rr.conditional_ci(y,b,n,2000,20261004)])
        write(pd.DataFrame(met),MD/'metrics.tsv');write(pd.DataFrame(metricci),MD/'metrics_conditional_ci.tsv');write(pd.DataFrame(increments),MD/'paired_model_increment.tsv');write(pd.DataFrame(conditional),MD/'conditional_paired_ci.tsv')
        print(pd.DataFrame(met).to_string(index=False),flush=True);print(pd.DataFrame(increments).to_string(index=False),flush=True)
        # Prespecified [PRIVATE_SUBJECT] sensitivity retains original partitions after removing one patient.
        sub=home[~getattr(home, _private_literal('private_044ace73522ed623'))];sp,sa=frozen_subset_oof(sub,blocks,original);write(sp,MD/_private_literal('private_deed361bb247a2ee'));write(sa,MD/_private_literal('private_0aaed32975466b1c'));se=rr.ensemble_predictions(sp);rows=[]
        for q in ['Q1','Q2']:
            y,b,n=rr.paired_array(se,'M1','M1_'+q);rows.append({'scope':_private_literal('private_e9fa51b1e2aebc3e'),'candidate':q,'n':len(y),**delta_fast(y,b,n)})
        write(pd.DataFrame(rows),MD/_private_literal('private_912a5adfd3adbca0'))
        # Held-out acquisition batch: no unestimable unseen batch dummy in either arm.
        lp=[];lm=[]
        for batch in sorted(home.batch.unique()):
            tr=home[home.batch!=batch];te=home[home.batch==batch]
            for model in MODELS:
                b={**blocks[model],'categorical':['sex']};fit=rr.tuned_fit(tr,b,17);p=fit.predict_proba(te)[:,1]
                lm.append({'held_out_batch':batch,'model':model,'selected_C':fit.best_params_['ridge__C'],**rr.metrics(te.n_binary,p)})
                lp.extend([{'patient_key':key,'batch':batch,'n_binary':int(y),'model':model,'probability':v} for key,y,v in zip(te.patient_key,te.n_binary,p)])
        write(pd.DataFrame(lp),MD/'lobo_predictions.tsv');write(pd.DataFrame(lm),MD/'lobo_metrics.tsv')
        # Common technical variables; external never enters selection, no absent demographics imputed.
        ep=[];em=[]
        for model in MODELS:
            num=list(rr.TECH)+([] if model=='M1' else [model.split('_')[1]+'_AA_carrier_ge2']);fit=rr.tuned_fit(home,{'numeric':num,'categorical':[]},17);p=fit.predict_proba(external)[:,1]
            em.append({'model':model+'_portable','selected_C':fit.best_params_['ridge__C'],'input':'|'.join(num),'interpretation':'historically_exposed_cross_source_recheck_Q1_all_absent',**rr.metrics(external.n_binary,p)})
            ep.extend([{'patient_key':key,'n_binary':int(y),'model':model+'_portable','probability':v} for key,y,v in zip(external.patient_key,external.n_binary,p)])
        write(pd.DataFrame(ep),MD/'external_predictions.tsv');write(pd.DataFrame(em),MD/'external_metrics.tsv')
    rr.json_write(MD/'status.json',{'state':'bootstrap_running','requested':args.bootstrap,'jobs':args.jobs,'started_unix':t})
    existing=read(MD/'full_pipeline_bootstrap.tsv') if (MD/'full_pipeline_bootstrap.tsv').exists() else pd.DataFrame();done=set(existing.loc[existing.status=='success','repetition']) if len(existing) else set();allrows=existing.to_dict('records') if len(existing) else []
    with ProcessPoolExecutor(max_workers=args.jobs) as ex:
        futures={ex.submit(bootstrap_worker,home,blocks,r):r for r in range(args.bootstrap) if r not in done}
        for fu in as_completed(futures):
            res=fu.result();rep=futures[fu];allrows=[r for r in allrows if r['repetition']!=rep]+res;write(pd.DataFrame(allrows).sort_values(['repetition','candidate']),MD/'full_pipeline_bootstrap.tsv')
            n=len({r['repetition'] for r in allrows if r['status']=='success'});rr.json_write(MD/'status.json',{'state':'bootstrap_running','completed':n,'requested':args.bootstrap,'elapsed_seconds':time.time()-t})
            if n%10==0 or any(r['status']!='success' for r in res):print('Bootstrap',n,'/',args.bootstrap,'last',rep,res[0]['status'],flush=True)
    boot=pd.DataFrame(allrows);ci=[]
    for q,g in boot.groupby('candidate'):
        good=g[g.status=='success']
        for c in ['delta_log_loss','delta_AUROC','delta_AUPRC','delta_Brier']:ci.append({'candidate':q,'metric':c,'ci_low':good[c].quantile(.025),'ci_high':good[c].quantile(.975),'requested':args.bootstrap,'successful':len(good),'CI':'full_pipeline_patient_stratified_percentile','baseline':'M1','model':'M1_'+q})
    write(pd.DataFrame(ci),MD/'full_pipeline_paired_ci.tsv')
    negatives=[]
    with ProcessPoolExecutor(max_workers=args.jobs) as ex:
        futures=[ex.submit(negative_worker,home,blocks,r) for r in range(20)]
        for fu in as_completed(futures):negatives.extend(fu.result())
    write(pd.DataFrame(negatives).sort_values(['repetition','candidate']),MD/'negative_controls.tsv')
    rr.json_write(MD/'status.json',{'state':'complete' if len(boot[boot.status=='success'])==2*args.bootstrap else 'incomplete_bootstrap_failures','bootstrap_successful_pairs':len(boot[boot.status=='success']),'bootstrap_requested_pairs':2*args.bootstrap,'negative_controls':len(negatives),'patient_overlap_max':int(boot.patient_overlap.max()),'elapsed_seconds':time.time()-t})
    print(pd.DataFrame(ci).to_string(index=False),flush=True)
if __name__=='__main__':main()
