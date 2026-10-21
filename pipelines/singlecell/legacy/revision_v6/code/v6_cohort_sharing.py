#!/usr/bin/env python3
"""Server-only v6 cohort, matched-depth donor sharing and descriptive tracking."""
from release_config import release_path, private_value
import os
for name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'): os.environ[name]='2'
import argparse, hashlib, json, platform, sys, time
from pathlib import Path
import numpy as np
import pandas as pd

BASE=Path(str(release_path('v6_cohort_sharing_source_10')))
ROOT=BASE/'revision_v6'
SEED=20261004
DRAWS=200
RESAMPLES=10000
CENTRAL_SHA='922145f07ff94cecdfc3c575732152d2276d31def5c2ef2beb781ced9a998a20'
PROGRAMS=('cytotoxic','exhaustion','naive_memory','interferon','residency','proliferation')
ON=json.loads(private_value('selected_on_visit_mapping_json'))
SOURCE=BASE/'revision_v5/bulk_sc_match/server_data/sc_cells.parquet'
CATALOG=BASE/'revision_v5/bulk_sc_match/server_data/key_catalog.parquet'
CONTEXT=BASE/'literature/GSE236581_sample_context.tsv'
PAIR=BASE/'single_cell/paired_patient_inventory.tsv'

def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()

def dump(p,x):
 Path(p).write_text(json.dumps(x,indent=2,allow_nan=False,default=lambda x:x.item() if hasattr(x,'item') else str(x))+'\n')

def save(p,x):
 pd.DataFrame(x).to_csv(p,sep='\t',index=False,na_rep='NA')

def manifest(folder,name,paths):
 rows=[{'path':str(p.relative_to(ROOT)) if p.is_relative_to(ROOT) else str(p),'bytes':p.stat().st_size,'sha256':sha(p)} for p in paths]
 save(folder/name,rows);return rows

def sub_seed(name):
 return SEED+int(hashlib.sha256(name.encode()).hexdigest()[:8],16)

def bh(p):
 p=np.array(p,float);o=np.argsort(p);out=np.empty(len(p));out[o]=np.minimum.accumulate((p[o]*len(p)/np.arange(1,len(p)+1))[::-1])[::-1];return np.minimum(out,1)

def check_protocol():
 assert sha(ROOT/'PROTOCOL_v6.md')==CENTRAL_SHA,'central protocol changed'

def cohort():
 check_protocol();out=ROOT/'cohort';out.mkdir(exist_ok=True)
 c=pd.read_parquet(SOURCE)
 assert not c.cell_name.duplicated().any()
 mask=(c.RNA_joined & c.MajorCellType.eq('T') & c.single_beta & c.sc_gene_syntax_valid & c.sc_nt_translation_valid & c.clinical_source_mapped)
 g=c.loc[mask].copy();assert g.key_NT_VJ.notna().all()
 assert g.beta_clone.eq(g.key_NT_VJ).all()
 g['sample_id']=g.Ident;g['stage']=g.verified_treatment_stage
 g['strict_paired_key']=g.paired_clone.where(g.strict_paired)
 g['AA_VJ_key']=g.key_AA_VJ
 ids=pd.read_csv(PAIR,sep='\t').query("primary_pair_status=='included'").patient_id.tolist();assert len(ids)==19
 ctx=pd.read_csv(CONTEXT,sep='\t')
 baseline=ctx[(ctx['Patient ID'].isin(ids)) & ctx['Treatment Stage'].eq('Pre') & ctx['Biopsy Site'].isin(['Peripheral blood','Tumor','Adjacent normal tissue'])]
 assert len(baseline)==57 and baseline.groupby('Patient ID').size().eq(3).all()
 b=g[g.Ident.isin(baseline.RNA_Ident_expected)].copy()
 assert b.verified_treatment_stage.eq('Pre').all()
 def add_own(x):
  for visit,part in x.groupby(['patient_id','sampling_stage'],sort=True):
   blood=part[part.Tissue.eq('Blood')]
   for key,tag in [('key_NT_VJ','NT_VJ'),('key_AA_VJ','AA_VJ')]:
    n=blood[key].value_counts()
    for threshold in (1,2): x.loc[part.index,'ownblood_'+tag+'_ge'+str(threshold)]=part[key].map(n).fillna(0).ge(threshold)
  for col in ['ownblood_'+tag+'_ge'+str(k) for tag in ('NT_VJ','AA_VJ') for k in (1,2)]:x[col]=x[col].astype(bool)
  return x
 cat=pd.read_parquet(CATALOG);a=cat[cat.definition.eq('AA_VJ')].set_index('match_key');assert a.index.is_unique
 for label,col in [('home','home_patients_ge2'),('external','external_patients_ge2')]:
  g['bulk_'+label+'_carriers_ge2']=g.key_AA_VJ.map(a[col]).fillna(0).astype(int)
 g['bulk_union_carriers_ge2']=g.bulk_home_carriers_ge2+g.bulk_external_carriers_ge2
 b=g.loc[b.index].copy();b=add_own(b)
 registry=[]
 for pid in ids:
  part=b[b.patient_id.eq(pid)];row={'patient_id':pid,'baseline_pair':True}
  for tissue in ('Blood','Tumor','Normal'):
   x=part[part.Tissue.eq(tissue)];assert x.Ident.nunique()==1
   row[tissue.lower()+'_sample']=x.Ident.iloc[0];row[tissue.lower()+'_beta_cells']=len(x);row[tissue.lower()+'_paired_cells']=int(x.strict_paired.sum());row[tissue.lower()+'_beta_clones']=x.beta_clone.nunique()
  row['high_depth_primary']=row['blood_beta_cells']>=500 and min(row['tumor_beta_cells'],row['normal_beta_cells'])>=50
  row['high_depth_paired']=row['blood_paired_cells']>=500 and min(row['tumor_paired_cells'],row['normal_paired_cells'])>=50
  row['low_depth_beta50']=min(row['blood_beta_cells'],row['tumor_beta_cells'],row['normal_beta_cells'])>=50
  row['tissue500_beta']=row['high_depth_primary'] and min(row['tumor_beta_cells'],row['normal_beta_cells'])>=500
  row['tissue500_paired']=row['high_depth_paired'] and min(row['tumor_paired_cells'],row['normal_paired_cells'])>=500
  row['primary_exclusion_reason']='included' if row['high_depth_primary'] else 'blood_beta_lt500' if row['blood_beta_cells']<500 else 'tissue_beta_lt50'
  registry.append(row)
 reg=pd.DataFrame(registry)
 for flag in ('high_depth_primary','high_depth_paired','low_depth_beta50','tissue500_beta','tissue500_paired'):
  b[flag]=b.patient_id.map(reg.set_index('patient_id')[flag])
 b['baseline_pair']=True
 on_rows=[]
 for pid,stage in ON.items():
  x=ctx[ctx['Patient ID'].eq(pid) & ctx['Treatment Stage'].eq('On') & ctx['Sampling Stage'].eq(stage) & ctx['Biopsy Site'].isin(['Peripheral blood','Tumor','Adjacent normal tissue'])]
  assert len(x)==3 and x['Biopsy Site'].nunique()==3
  # Verify source-based first complete On stage, not first outcome-based visit.
  complete=[]
  for st,z in ctx[ctx['Patient ID'].eq(pid)&ctx['Treatment Stage'].eq('On')].groupby('Sampling Stage'):
   if set(z['Biopsy Site']) >= {'Peripheral blood','Tumor','Adjacent normal tissue'}:complete.append(st)
  order={'I':1,'II':2,'III':3,'IV':4};assert stage==min(complete,key=order.get)
  on_rows.extend(x.to_dict('records'))
 onctx=pd.DataFrame(on_rows)
 l=g[(g.patient_id.isin(ON)) & (g.Ident.isin(baseline.RNA_Ident_expected)|g.Ident.isin(onctx.RNA_Ident_expected))].copy();l=add_own(l)
 l['selected_visit']=np.where(l.verified_treatment_stage.eq('Pre'),'Pre','first_complete_On')
 l['selected_on_stage']=l.patient_id.map(ON)
 lc=[]
 for (pid,visit,tissue),x in l.groupby(['patient_id','selected_visit','Tissue']):
  lc.append({'patient_id':pid,'selected_visit':visit,'sample_id':x.Ident.iloc[0],'Tissue':tissue,'sampling_stage':x.sampling_stage.iloc[0],'treatment_point':x.verified_treatment_point.iloc[0],'regimen':x['Treatment Regimen'].iloc[0],'accepted_beta_cells':len(x),'strict_paired_cells':int(x.strict_paired.sum()),'beta_clones':x.beta_clone.nunique()})
 assert len(lc)==48
 for name,data in [('all_source_t_cells',g),('baseline_cells',b),('longitudinal_cells',l)]: data.to_parquet(out/(name+'.parquet'),index=False)
 save(out/'baseline_patient_registry.tsv',reg);save(out/'longitudinal_sample_registry.tsv',lc)
 save(out/'baseline_sample_registry.tsv',baseline);save(out/'selected_On_sample_context.tsv',onctx)
 (out/'SCHEMA.md').write_text(private_value('v6_cohort_sharing_value_115_30'))
 inputs=manifest(out,'input_manifest.tsv',[SOURCE,CATALOG,CONTEXT,PAIR,ROOT/'PROTOCOL_v6.md',BASE/'single_cell/marker_program_dictionary.tsv'])
 outputs=manifest(out,'output_manifest.tsv',sorted(p for p in out.iterdir() if p.name not in ('output_manifest.tsv','cohort_receipt.json')))
 receipt={'status':'PASS','seed':SEED,'central_protocol_sha256':CENTRAL_SHA,'entrypoint_sha256':sha(Path(__file__)),'all_source_rna_t_beta_cells':len(g),'baseline_cells':len(b),'baseline_patients':19,'primary_patients':int(reg.high_depth_primary.sum()),'paired_primary_patients':int(reg.high_depth_paired.sum()),'low_depth50_patients':int(reg.low_depth_beta50.sum()),'tissue500_beta_patients':int(reg.tissue500_beta.sum()),'tissue500_paired_patients':int(reg.tissue500_paired.sum()),'longitudinal_patients':8,'longitudinal_cells':len(l),'inputs':inputs,'outputs':outputs,'measurement_unit':'accepted RNA-joined author-T cell; beta_nt_vj clone identity','no_inferential_tests_in_cohort':True}
 dump(out/'cohort_receipt.json',receipt);print(json.dumps({k:v for k,v in receipt.items() if k not in ('inputs','outputs')}),flush=True)

def identity_frame(df,identity):
 if identity=='paired':return df[df.strict_paired & df.strict_paired_key.notna()].copy(),'strict_paired_key','strict_paired_key'
 return df,{'NT_VJ':'key_NT_VJ','AA_VJ':'key_AA_VJ','AA':'key_AA'}[identity],'beta_clone'

def build_weights(query,patients,key,clone,query_depth=None,rng=None):
 """K x (tissue,metric,query) contribution table. Each clone counted once."""
 selected={};den=[];keys=[]
 for t in ('Tumor','Normal'):
  for pid in patients:
   x=query[query.patient_id.eq(pid)&query.Tissue.eq(t)]
   if query_depth:x=x.iloc[rng.choice(len(x),query_depth,replace=False)]
   assert len(x)>0 and x[key].notna().all()
   selected[t,pid]=x;keys.extend(x[key].unique())
   den.append({'query_patient':pid,'Tissue':t,'cell_denominator':len(x),'clone_denominator':x[clone].nunique()})
 keys=sorted(set(keys));idx={k:i for i,k in enumerate(keys)};n=len(patients);w=np.zeros((len(keys),2,2,n))
 for ti,t in enumerate(('Tumor','Normal')):
  for i,pid in enumerate(patients):
   x=selected[t,pid];cell=x[key].value_counts();cl=x.drop_duplicates(clone)[key].value_counts()
   for k,c in cell.items():w[idx[k],ti,0,i]=c/len(x)
   for k,c in cl.items():w[idx[k],ti,1,i]=c/x[clone].nunique()
 return w,idx,den

def matching_matrix(df,patients,identity,depth,min_support,draws,seed,query_depth=None):
 df,key,clone=identity_frame(df,identity);rng=np.random.default_rng(seed);n=len(patients)
 blood={p:df[df.patient_id.eq(p)&df.Tissue.eq('Blood')][key].to_numpy() for p in patients}
 query=df.loc[df.Tissue.isin(['Tumor','Normal']) & df.patient_id.isin(patients),list(dict.fromkeys(['patient_id','Tissue',key,clone]))]
 matrix=np.zeros((2,2,n,n));denrows=[]
 fixed=None if query_depth else build_weights(query,patients,key,clone)
 for r in range(draws):
  weights,idx,den=build_weights(query,patients,key,clone,query_depth,rng) if query_depth else fixed
  for z in den:denrows.append({**z,'draw':r})
  for j,pid in enumerate(patients):
   arr=blood[pid] if depth is None else blood[pid][rng.choice(len(blood[pid]),depth,replace=False)]
   kk,nn=np.unique(arr,return_counts=True);ix=[idx[k] for k in kk[nn>=min_support] if k in idx]
   if ix:matrix[:,:,:,j]+=weights[ix].sum(axis=0)/draws
 return matrix,pd.DataFrame(denrows).groupby(['query_patient','Tissue'],as_index=False)[['cell_denominator','clone_denominator']].mean()

def effects(matrix,weights=None):
 """Resample whole original patients; all duplicate identity copies excluded as peers."""
 n=matrix.shape[-1];weights=np.ones(n) if weights is None else np.asarray(weights,float)
 own=np.diagonal(matrix,axis1=2,axis2=3);rowtotal=np.einsum('tmij,j->tmi',matrix,weights)
 peerden=weights.sum()-weights
 valid=(weights>0)&(peerden>0)
 if not np.all(peerden[weights>0]>0):return np.full((2,3),np.nan)
 peers=(rowtotal-own*weights[None,None,:])/peerden[None,None,:]
 e=own-peers;means=np.einsum('tmi,i->tm',e[:,:,valid],weights[valid])/weights[valid].sum()
 return np.stack([means[0],means[1],means[0]-means[1]],axis=1)

def inference(matrix,seed,reps=RESAMPLES):
 n=matrix.shape[-1];obs=effects(matrix);rng=np.random.default_rng(seed)
 null=[];boots=[]
 for _ in range(reps):
  permutation=rng.permutation(n);permuted=matrix[:,:,:,permutation]
  null.append(effects(permuted));w=rng.multinomial(n,np.full(n,1/n));boots.append(effects(matrix,w))
 null=np.array(null);boots=np.array(boots);successful=np.isfinite(boots).all(axis=(1,2))
 assert successful.sum()>=reps*.999
 ci=np.quantile(boots[successful],[.025,.975],axis=0)
 p=(1+(np.abs(null)>=np.abs(obs)[None,:,:]-1e-15).sum(axis=0))/(reps+1)
 return obs,ci,p,null,boots,successful

CONFIGS=[
 ('primary_NT_VJ_500','NT_VJ',500,1,'high_depth_primary',None,True),
 ('AA_VJ_500','AA_VJ',500,1,'high_depth_primary',None,True),
 ('AA_500','AA',500,1,'high_depth_primary',None,True),
 ('paired_500','paired',500,1,'high_depth_paired',None,True),
 ('NT_VJ_50_all19','NT_VJ',50,1,'low_depth_beta50',None,True),
 ('NT_VJ_tissue500','NT_VJ',500,1,'tissue500_beta',500,True),
 ('paired_tissue500','paired',500,1,'tissue500_paired',500,True),
 ('NT_VJ_blood_ge2','NT_VJ',500,2,'high_depth_primary',None,True),
 ('NT_VJ_fullcatalog','NT_VJ',None,1,'low_depth_beta50',None,False)]

def sharing():
 check_protocol();out=ROOT/'sharing';out.mkdir(exist_ok=True);receipt=json.loads((ROOT/'cohort/cohort_receipt.json').read_text());assert receipt['status']=='PASS'
 assert sha(out/'PROTOCOL.md')==json.loads((out/'protocol_receipt.json').read_text())['sha256'],'sharing protocol changed'
 b=pd.read_parquet(ROOT/'cohort/baseline_cells.parquet');reg=pd.read_csv(ROOT/'cohort/baseline_patient_registry.tsv',sep='\t')
 matrices=[];patientsrows=[];stats=[];nullrows=[];elig=[];denrows=[]
 for aid,identity,depth,support,flag,qd,formal in CONFIGS:
  patients=reg.loc[reg[flag],'patient_id'].tolist();n=len(patients)
  print('sharing',aid,n,flush=True)
  seed=sub_seed(aid);m,den=matching_matrix(b,patients,identity,depth,support,DRAWS if depth else 1,seed,qd)
  for row in den.to_dict('records'):denrows.append({'analysis_id':aid,**row})
  for pid in reg.patient_id:elig.append({'analysis_id':aid,'patient_id':pid,'included':pid in patients,'reason':'included' if pid in patients else flag+'=False','n_other_donors':n-1 if pid in patients else 0,'identity':identity,'blood_depth':depth,'query_depth':qd,'support_cells':support,'draws':DRAWS if depth else 1,'seed':seed})
  for ti,t in enumerate(('Tumor','Normal')):
   for mi,metric in enumerate(('cell_fraction','clone_fraction')):
    for i,pid in enumerate(patients):
     selfv=m[ti,mi,i,i];peer=np.delete(m[ti,mi,i],i).mean()
     patientsrows.append({'analysis_id':aid,'patient_id':pid,'Tissue':t,'metric':metric,'self_coverage':selfv,'mean_other_coverage':peer,'self_excess':selfv-peer,'n_peers':n-1})
     for j,bloodpid in enumerate(patients):matrices.append({'analysis_id':aid,'identity':identity,'blood_depth':depth,'query_depth':qd,'min_blood_cells':support,'Tissue':t,'metric':metric,'query_patient':pid,'blood_patient':bloodpid,'coverage':m[ti,mi,i,j],'is_self':pid==bloodpid})
  for mi,metric in enumerate(('cell_fraction','clone_fraction')):
   for i,pid in enumerate(patients):
    selfdiff=m[0,mi,i,i]-m[1,mi,i,i];peerdiff=np.delete(m[0,mi,i]-m[1,mi,i],i).mean()
    patientsrows.append({'analysis_id':aid,'patient_id':pid,'Tissue':'Tumor_minus_Normal','metric':metric,'self_coverage':selfdiff,'mean_other_coverage':peerdiff,'self_excess':selfdiff-peerdiff,'n_peers':n-1})
  if formal:
   obs,ci,p,null,boot,success=inference(m,sub_seed(aid+'_inference'))
   qs=bh(p.flatten()).reshape(2,3)
   for mi,metric in enumerate(('cell_fraction','clone_fraction')):
    for j,contrast in enumerate(('Tumor_self_excess','Normal_self_excess','Tumor_minus_Normal_self_excess')):
     stats.append({'analysis_id':aid,'metric':metric,'contrast':contrast,'n_patients':n,'n_peers':n-1,'mean_effect':obs[mi,j],'ci_low':ci[0,mi,j],'ci_high':ci[1,mi,j],'P':p[mi,j],'q':qs[mi,j],'BH_family':aid+'_six','family_size':6,'permutations':RESAMPLES,'bootstrap_draws':RESAMPLES,'bootstrap_successful':int(success.sum()),'null_mean':null[:,mi,j].mean(),'null_reference_low':np.quantile(null[:,mi,j],.025),'null_reference_high':np.quantile(null[:,mi,j],.975),'seed':sub_seed(aid+'_inference')})
     for r in range(RESAMPLES):nullrows.append({'analysis_id':aid,'draw':r,'metric':metric,'contrast':contrast,'permutation_effect':null[r,mi,j],'patient_block_bootstrap_effect':boot[r,mi,j]})
   np.savez_compressed(out/(aid+'_matrix_resampling.npz'),matrix=m,patients=np.array(patients),permutation_effects=null,bootstrap_effects=boot)
  else:
   obs=effects(m)
   for mi,metric in enumerate(('cell_fraction','clone_fraction')):
    for j,contrast in enumerate(('Tumor_self_excess','Normal_self_excess','Tumor_minus_Normal_self_excess')):stats.append({'analysis_id':aid,'metric':metric,'contrast':contrast,'n_patients':n,'mean_effect':obs[mi,j],'P':np.nan,'q':np.nan,'inference_status':'descriptive_only'})
 save(out/'matching_matrices.tsv',matrices);save(out/'patient_contrasts.tsv',patientsrows);save(out/'statistics.tsv',stats);save(out/'eligibility.tsv',elig);save(out/'query_denominators.tsv',denrows)
 pd.DataFrame(nullrows).to_parquet(out/'resampling_draws.parquet',index=False)
 inputs=manifest(out,'input_manifest.tsv',[ROOT/'cohort/baseline_cells.parquet',ROOT/'cohort/baseline_patient_registry.tsv',ROOT/'cohort/cohort_receipt.json',out/'PROTOCOL.md',Path(__file__)])
 outputs=manifest(out,'output_manifest.tsv',sorted(p for p in out.iterdir() if p.name not in ('output_manifest.tsv','sharing_receipt.json','SHARING_REPORT.md')))
 dump(out/'sharing_receipt.json',{'status':'PASS','seed':SEED,'central_protocol_sha256':CENTRAL_SHA,'entrypoint_sha256':sha(Path(__file__)),'analyses':len(CONFIGS),'formal_six_test_families':8,'primary_family_size':6,'input_manifest':inputs,'output_manifest':outputs,'rare_faction_draws_are_not_patients':True,'peer_exclusion':'all copies of original query donor excluded','permutation_null':'random whole blood-donor identity assignment to fixed query persons; T/N synchronized; exact statistic is differential self excess for T-N, not generic tissue effect'})
 primary=pd.DataFrame(stats).query("analysis_id=='primary_NT_VJ_500'")
 lines=['# Within-person blood–tissue sharing','', 'Status: **PASS**. Seed 20261004. Primary18 patients,17 equal-depth other blood donors per tissue query. Blood500 cells,200 rarefaction draws averaged before donor inference.','',primary.to_csv(sep='\t',index=False),'','Primary match: exact beta NT/V/J. Cell and distinct-beta-clone coverage have accepted-cell and unique NT/V/J denominators. Every identity sensitivity except strict paired retains distinct beta NT/V/J clone denominators; strict paired uses distinct paired alpha/beta identities and paired-cell denominator.','', 'P values test whole donor-label assignments, not independent edges; Tumor-minus-Normal P concerns differential autologous excess. 10000 donor-block bootstrap samples preserve original identities and exclude duplicate self identities from peer means. Full-catalog sharing is descriptive. Sensitivity families each have6 tests and do not replace the primary family.','', 'Nondetection is sampling-dependent. Sharing establishes sequence recurrence; it does not identify migration direction or antigen specificity.']
 (out/'SHARING_REPORT.md').write_text('\n'.join(lines)+'\n')
 print(primary[['metric','contrast','mean_effect','ci_low','ci_high','P','q']].to_string(index=False),flush=True)

def track_visit(l,identity):
 df,key,clone=identity_frame(l,identity);rows=[]
 for (pid,visit),x in df.groupby(['patient_id','selected_visit']):
  blood=set(x[x.Tissue.eq('Blood')][key]);
  for tissue in ('Tumor','Normal'):
   y=x[x.Tissue.eq(tissue)];matched=y[key].isin(blood)
   rows.append({'identity':identity,'patient_id':pid,'visit':visit,'Tissue':tissue,'cell_fraction':matched.mean(),'clone_fraction':y.loc[matched,clone].nunique()/y[clone].nunique(),'cells':len(y),'clones':y[clone].nunique(),'blood_cells':len(x[x.Tissue.eq('Blood')]),'sample_id':y.Ident.iloc[0],'treatment_point':y.verified_treatment_point.iloc[0]})
 return rows

def longitudinal():
 check_protocol();out=ROOT/'longitudinal';out.mkdir(exist_ok=True);l=pd.read_parquet(ROOT/'cohort/longitudinal_cells.parquet')
 assert set(l.patient_id)==set(ON)
 trajectories=[];tracking=[];programrows=[];rare=[];eligible=[]
 for identity in ('NT_VJ','AA_VJ','paired'):
  df,key,clone=identity_frame(l,identity);trajectories.extend(track_visit(l,identity));clone=key
  for pid,x in df.groupby('patient_id'):
   pre=x[x.selected_visit.eq('Pre')];on=x[x.selected_visit.eq('first_complete_On')]
   for src_t in ('Blood','Tumor','Normal'):
    src=pre[pre.Tissue.eq(src_t)];catalog=set(src[clone])
    for dest_t in ('Blood','Tumor','Normal'):
     dest=on[on.Tissue.eq(dest_t)];dc=set(dest[clone]);common=catalog&dc
     tracking.append({'identity':identity,'patient_id':pid,'Pre_source_tissue':src_t,'On_destination_tissue':dest_t,'baseline_clones':len(catalog),'On_clones':len(dc),'reobserved_baseline_clones':len(common),'baseline_fraction_reobserved':len(common)/len(catalog) if catalog else np.nan,'On_cells_with_baseline_clone':int(dest[clone].isin(catalog).sum()),'On_cell_fraction_with_baseline_clone':dest[clone].isin(catalog).mean(),'On_stage':ON[pid]})
   for tissue in ('Blood','Tumor','Normal'):
    a=pre[pre.Tissue.eq(tissue)&pre.is_cd8&pre.positive_expression_qc];z=on[on.Tissue.eq(tissue)&on.is_cd8&on.positive_expression_qc]
    common=set(a[clone])&set(z[clone])
    for k in sorted(common):
     for program in PROGRAMS:
      score='program_'+program
      av=a[a[clone].eq(k)][score].mean();zv=z[z[clone].eq(k)][score].mean()
      programrows.append({'identity':identity,'patient_id':pid,'Tissue':tissue,'clone_key':k,'program':program,'Pre_clone_mean':av,'On_clone_mean':zv,'difference_On_minus_Pre':zv-av,'Pre_clone_cells':int(a[clone].eq(k).sum()),'On_clone_cells':int(z[clone].eq(k).sum()),'On_stage':ON[pid]})
  for pid in ON:
   part=df[df.patient_id.eq(pid)];n=[len(part[part.selected_visit.eq(v)&part.Tissue.eq('Blood')]) for v in ('Pre','first_complete_On')]
   eligible.append({'identity':identity,'patient_id':pid,'Pre_blood_cells':n[0],'On_blood_cells':n[1],'depth500_eligible':min(n)>=500,'reason':'included' if min(n)>=500 else 'at_least_one_visit_blood_lt500'})
  selected=[r['patient_id'] for r in eligible if r['identity']==identity and r['depth500_eligible']]
  if selected:
   for visit in ('Pre','first_complete_On'):
    v=df[df.selected_visit.eq(visit)].copy();m,den=matching_matrix(v,selected,identity,500,1,DRAWS,sub_seed('long_'+identity+'_'+visit))
    for ti,t in enumerate(('Tumor','Normal')):
     for mi,metric in enumerate(('cell_fraction','clone_fraction')):
      for i,pid in enumerate(selected):rare.append({'identity':identity,'patient_id':pid,'visit':visit,'Tissue':t,'metric':metric,'self_coverage':m[ti,mi,i,i],'mean_other_coverage':np.delete(m[ti,mi,i],i).mean(),'self_excess':m[ti,mi,i,i]-np.delete(m[ti,mi,i],i).mean(),'n_peers':len(selected)-1,'blood_depth':500,'draws':DRAWS,'On_stage':ON[pid]})
 save(out/'patient_trajectories.tsv',trajectories);save(out/'clone_reobservation.tsv',tracking);save(out/'depth500_trajectories.tsv',rare);save(out/'depth500_eligibility.tsv',eligible)
 pr=pd.DataFrame(programrows);pr.to_parquet(out/'tracked_clone_programs.parquet',index=False)
 complete=pd.MultiIndex.from_product([('NT_VJ','AA_VJ','paired'),sorted(ON),('Blood','Tumor','Normal'),PROGRAMS],names=['identity','patient_id','Tissue','program']).to_frame(index=False)
 if len(pr):
  summaries=pr.groupby(['identity','patient_id','Tissue','program'],as_index=False).agg(n_tracked_clones=('clone_key','nunique'),mean_Pre=('Pre_clone_mean','mean'),mean_On=('On_clone_mean','mean'),mean_difference=('difference_On_minus_Pre','mean'))
  complete=complete.merge(summaries,on=['identity','patient_id','Tissue','program'],how='left')
 complete['n_tracked_clones']=complete['n_tracked_clones'].fillna(0).astype(int);complete['status']=np.where(complete.n_tracked_clones.gt(0),'descriptive_tracked_clones','no_CD8_clone_reobserved');save(out/'tracked_program_patient_summary.tsv',complete)
 inputs=manifest(out,'input_manifest.tsv',[ROOT/'cohort/longitudinal_cells.parquet',ROOT/'cohort/selected_On_sample_context.tsv',ROOT/'PROTOCOL_v6.md',Path(__file__)])
 outputs=manifest(out,'output_manifest.tsv',sorted(p for p in out.iterdir() if p.name not in ('output_manifest.tsv','longitudinal_receipt.json','LONGITUDINAL_REPORT.md')))
 dump(out/'longitudinal_receipt.json',{'status':'PASS','patients':8,'stage_map':ON,'seed':SEED,'formal_tests':0,'paired_depth500_patients':{i:sum(r['depth500_eligible'] for r in eligible if r['identity']==i) for i in ('NT_VJ','AA_VJ','paired')},'inputs':inputs,'outputs':outputs,'entrypoint_sha256':sha(Path(__file__)),'interpretation':'re-observation, not migration or disappearance; exact paired receptor required for alpha-beta tracking'})
 (out/'LONGITUDINAL_REPORT.md').write_text(private_value('v6_cohort_sharing_value_285_43'))
 print('longitudinal complete; descriptive only',flush=True)

def selftest():
 checks=[]
 def ck(name,condition):
  assert condition,name;checks.append({'check':name,'status':'PASS'})
 m=np.zeros((2,2,4,4));m[:]=np.eye(4)
 ck('identity_synchronised_matrix',np.allclose(effects(m)[:,:2],1))
 ck('T_minus_N_same_matrix_zero',np.allclose(effects(m)[:,2],0))
 ck('bootstrap_duplicate_original_not_peer',np.allclose(effects(m,np.array([2,1,1,0]))[:,:2],1))
 mm=np.ones((2,2,4,4));ck('all_shared_zero_excess',np.allclose(effects(mm),0))
 ck('single_original_bootstrap_unestimable',np.isnan(effects(m,[4,0,0,0])).all())
 p=np.array([.001,.03,.04,.2,.5,1]);ck('BH_full_six',np.allclose(bh(p),[.006,.08,.08,.3,.6,1]))
 df=pd.DataFrame({'patient_id':['P1']*5,'Tissue':['Tumor']*5,'key':['A','A','A','B','B'],'clone':['a','a','b','c','c']})
 df=pd.concat([df,df.assign(Tissue='Normal')]);w,ix,den=build_weights(df,['P1'],'key','clone')
 ck('cell_denominator_five',np.allclose(w[ix['A'],:,0,0],3/5))
 ck('distinct_NT_clones_under_AA_matching',np.allclose(w[ix['A'],:,1,0],2/3))
 ck('denominator_closure',np.allclose(w.sum(axis=0),1))
 kk,nn=np.unique(['A','A','B'],return_counts=True);ck('ge2_cells_within_key_not_total_reads',list(kk[nn>=2])==['A'])
 wgt=np.array([2,1,1,0]);explicit=np.mean([m[:,:,i,i]-sum(wgt[j]*m[:,:,i,j] for j in range(4) if j!=i)/(sum(wgt)-wgt[i]) for i in [0,0,1,2]],axis=0);ck('blockbootstrap_explicit_original_identity_loop',np.allclose(effects(m,wgt)[:,:2],explicit.T))
 # Exact enumeration of donor assignments confirms the mean-zero label null.
 import itertools
 null=np.array([effects(m[:,:,:,p]) for p in itertools.permutations(range(4))]);ck('permutation_null_mean_zero',np.allclose(null.mean(axis=0),0))
 ck('permutation_sync_TN_difference_zero',np.allclose(null[:,:,2],0))
 dump(ROOT/'cohort/self_tests.json',{'status':'PASS','checks':checks,'entrypoint_sha256':sha(Path(__file__))});print('selftests',len(checks),'PASS',flush=True)

def verify():
 checks=[]
 def ck(name,x):assert x,name;checks.append({'check':name,'status':'PASS'})
 b=pd.read_parquet(ROOT/'cohort/baseline_cells.parquet');l=pd.read_parquet(ROOT/'cohort/longitudinal_cells.parquet');reg=pd.read_csv(ROOT/'cohort/baseline_patient_registry.tsv',sep='\t')
 ck('19_unique_baseline_donors',b.patient_id.nunique()==19);ck('baseline_unique_cells',not b.cell_name.duplicated().any());ck('RNA_T_flags',b.RNA_joined.all() and b.MajorCellType.eq('T').all());ck('Pre_only',b.stage.eq('Pre').all());ck('18_primary',reg.high_depth_primary.sum()==18);ck('8_longitudinal',l.patient_id.nunique()==8);ck(private_value('v6_cohort_sharing_value_316_333'),set(l[l.patient_id.eq(private_value('v6_cohort_sharing_value_316_378'))&l.selected_visit.eq('first_complete_On')&l.Tissue.eq('Tumor')].Ident)=={private_value('cohort_sharing_selected_sample_check')})
 for (pid,st),part in pd.concat([b,l]).drop_duplicates('cell_name').groupby(['patient_id','sampling_stage']):
  blood=part[part.Tissue.eq('Blood')]
  for identity,key in [('NT_VJ','key_NT_VJ'),('AA_VJ','key_AA_VJ')]:
   n=blood[key].value_counts()
   for k in (1,2):ck('ownblood_'+pid+'_'+st+'_'+identity+'_'+str(k),part['ownblood_'+identity+'_ge'+str(k)].eq(part[key].map(n).fillna(0).ge(k)).all())
 if (ROOT/'sharing/statistics.tsv').exists():
  stats=pd.read_csv(ROOT/'sharing/statistics.tsv',sep='\t');mats=pd.read_csv(ROOT/'sharing/matching_matrices.tsv',sep='\t');points=pd.read_csv(ROOT/'sharing/patient_contrasts.tsv',sep='\t')
  ck('all_coverage_in_0_1',mats.coverage.between(-1e-12,1+1e-12).all())
  for aid,x in stats[stats.P.notna()].groupby('analysis_id'):
   ck(aid+'_six_tests',len(x)==6);ck(aid+'_BH',np.allclose(bh(x.P),x.q))
   arr=np.load(ROOT/'sharing'/(aid+'_matrix_resampling.npz'));m=arr['matrix'];ck(aid+'_point_estimator',np.allclose(effects(m).flatten(),x.mean_effect))
   null=arr['permutation_effects'];pv=(1+(abs(null)>=abs(effects(m))[None,:,:]-1e-15).sum(axis=0))/(RESAMPLES+1);ck(aid+'_permutation_tail',np.allclose(pv.flatten(),x.P))
   boots=arr['bootstrap_effects'];ci=np.nanquantile(boots,[.025,.975],axis=0);ck(aid+'_blockbootstrap_CI',np.allclose(ci[0].flatten(),x.ci_low) and np.allclose(ci[1].flatten(),x.ci_high))
  ck('full_catalog_no_formal_P',stats.query("analysis_id=='NT_VJ_fullcatalog'").P.isna().all())
 if (ROOT/'longitudinal/longitudinal_receipt.json').exists():ck('longitudinal_zero_tests',json.loads((ROOT/'longitudinal/longitudinal_receipt.json').read_text())['formal_tests']==0)
 for folder,receiptfile in [('cohort','cohort_receipt.json'),('sharing','sharing_receipt.json'),('longitudinal','longitudinal_receipt.json')]:
  p=ROOT/folder/receiptfile
  if p.exists():
   r=json.loads(p.read_text());output=r.get('output_manifest',r.get('outputs',[]))
   for o in output:ck(folder+'_hash_'+o['path'],sha(ROOT/o['path'])==o['sha256'])
 destination=ROOT/'sharing/verification_receipt.json' if (ROOT/'sharing/statistics.tsv').exists() else ROOT/'longitudinal/verification_receipt.json' if (ROOT/'longitudinal/longitudinal_receipt.json').exists() else ROOT/'cohort/verification_receipt.json'
 dump(destination,{'status':'PASS','checks':checks,'n_checks':len(checks),'seed':SEED,'entrypoint_sha256':sha(Path(__file__))});print('verification',len(checks),'PASS',destination,flush=True)

if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('mode',choices=['cohort','sharing','longitudinal','selftest','verify','all']);args=ap.parse_args()
 if args.mode=='all':cohort();selftest();sharing();longitudinal();verify()
 else:globals()[args.mode]()
