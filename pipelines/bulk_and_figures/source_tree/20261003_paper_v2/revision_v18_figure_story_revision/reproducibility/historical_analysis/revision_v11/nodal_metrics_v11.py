#!/usr/bin/env python3
"""Fixed v11 nodal metric audit; remote only; see audit/nodal/PROTOCOL.md."""
from release_runtime import private_literal as _private_literal, release_path as _release_path
import os
for k in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='1'
from pathlib import Path
import json,hashlib,re,importlib.util,sys,time
import numpy as np,pandas as pd
from scipy.stats import mannwhitneyu
from statsmodels.stats.multitest import multipletests
B=Path(_release_path('path_ab9dd0e0159dec18'))
O=B/'revision_v11';T=O/'tables/nodal';A=O/'audit/nodal'
F=B/'revision_v5/nodal/frozen/paper_v1'
AT=Path(_release_path('path_99d8c1c6fde2a6b9'))
SPEC=importlib.util.spec_from_file_location('old_assoc',F/'code/run_associations.py');M=importlib.util.module_from_spec(SPEC);SPEC.loader.exec_module(M)
meta=[
('shannon_entropy','Shannon entropy','nats','repertoire'),
('normalized_shannon','Normalized Shannon diversity','proportion','repertoire'),
('simpson_diversity','Simpson diversity','proportion','repertoire'),
('top1_fraction','Largest CDR3beta AA sequence read fraction','proportion','repertoire'),
('top10_fraction','Top10 CDR3beta AA sequence read fraction','proportion','repertoire'),
('weighted_mean_cdr3_length','Read-weighted CDR3beta length','amino acids','repertoire'),
('unique_aa','CDR3beta AA sequence richness','distinct AA sequences','repertoire'),
('total_reads','Productive-QC read count','MiXCR readCount','technical'),
('input_reads','MiXCR input sequencing read count','MiXCR input counter','technical'),
('alignment_fraction','Successfully aligned / input reads','proportion','technical'),
('trb_alignment_fraction','TRB assigned / successfully aligned reads','proportion','technical'),
('productive_input_fraction','Productive-QC readCount / input reads','proportion','technical')]
dictionary=pd.DataFrame(meta,columns=['metric','label','unit','block'])
dictionary['family']='expanded12'
dictionary.to_csv(T/'metric_dictionary.tsv',sep='\t',index=False)
def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
sources=[]
def track(p,role):sources.append(dict(path=str(p),sha256=digest(p),bytes=Path(p).stat().st_size,role=role))
fp=F/'features/patient_features.tsv';lp=F/'cohort/library_master.tsv'
track(fp,'frozen patient features');track(lp,'frozen library identities');track(A/'PROTOCOL.md','fixed before tests')
p=pd.read_csv(fp,sep='\t');libraries=pd.read_csv(lp,sep='\t').set_index('sample_id')
assert p.patient_key.is_unique and len(p)==339
rows=[]
for i,r in p.iterrows():
 sid=r.primary_sample_ids;l=libraries.loc[sid];assert int(l.primary_repertoire_library)==1 and l.patient_key==r.patient_key
 ap=AT/'data/patient_aa'/f'{r.cohort}__{sid}.parquet';rp=AT/'audit/sample_receipts'/f'{r.cohort}__{sid}.json'
 rec=json.loads(rp.read_text());assert digest(ap)==rec['patient_aa_sha256'];track(ap,'accepted AA read-count table');track(rp,'accepted source receipt')
 aa=pd.read_parquet(ap);counts=aa.read_count.to_numpy(np.int64);assert (counts>0).all() and aa.aa.is_unique
 n=int(counts.sum());k=len(counts);probs=counts/n;h=float(-np.dot(probs,np.log(probs)))
 rr=r.to_dict();rr.update(shannon_entropy=h,clonality=1-r.normalized_shannon,simpson_diversity=1-float(np.dot(probs,probs)),top1_fraction=float(probs.max()))
 assert n==r.total_reads and k==r.unique_aa and abs(h/np.log(k)-r.normalized_shannon)<1e-10
 assert abs(np.sort(counts)[-10:].sum()/n-r.top10_fraction)<1e-10
 assert abs(np.dot(probs,aa.aa.str.len().to_numpy())-r.weighted_mean_cdr3_length)<1e-10
 report=Path(l.mixcr_report_path);text=report.read_text();track(report,'current frozen MiXCR alignment report')
 def val(label):
  hits=re.findall(r'^'+re.escape(label)+r':\s*(\d+)',text,re.M)
  assert len(hits)==1,(sid,label,hits)
  return int(hits[0])
 inp=val('Total sequencing reads');align=val('Successfully aligned reads');trb=val('TRB chains')
 assert 0<n<=inp and 0<trb<=align<=inp
 rr.update(input_reads=inp,aligned_reads=align,trb_aligned_reads=trb,alignment_fraction=align/inp,trb_alignment_fraction=trb/align,productive_input_fraction=n/inp)
 rows.append(rr)
 if i%50==0:print('features',i,flush=True)
d=pd.DataFrame(rows);d.to_csv(T/'patient_metrics.tsv',sep='\t',index=False)
pd.DataFrame(sources).to_csv(A/'source_manifest.tsv',sep='\t',index=False)
def bh(v):return multipletests(np.where(np.isfinite(v),v,1),method='fdr_bh')[1]
def analyses(frame,name,cohort):
 assert frame.patient_key.is_unique
 un=[];ad=[]
 for j,(metric,label,unit,block) in enumerate(meta):
  x=frame.loc[frame.n_binary.eq(0),metric].to_numpy();y=frame.loc[frame.n_binary.eq(1),metric].to_numpy();assert np.isfinite(x).all() and np.isfinite(y).all()
  t=mannwhitneyu(y,x,alternative='two-sided',method='asymptotic',use_continuity=True)
  rng=np.random.default_rng(20261005+j)
  ds=np.median(rng.choice(y,(10000,len(y))),axis=1)-np.median(rng.choice(x,(10000,len(x))),axis=1)
  row=dict(analysis=name,cohort=cohort,metric=metric,label=label,block=block,unit=unit,N0=len(x),Nplus=len(y),median_N0=np.median(x),q25_N0=np.quantile(x,.25),q75_N0=np.quantile(x,.75),median_Nplus=np.median(y),q25_Nplus=np.quantile(y,.25),q75_Nplus=np.quantile(y,.75),median_delta=np.median(y)-np.median(x),median_delta_CIlo=np.quantile(ds,.025),median_delta_CIhi=np.quantile(ds,.975),rank_biserial=2*t.statistic/(len(x)*len(y))-1,P=t.pvalue,method='two-sided Mann-Whitney U asymptotic tie/continuity correction; stratified patient bootstrap median difference 10000',family_size=12)
  un.append(row)
  M.NUMERIC_CONFOUNDERS=tuple((['age_years'] if cohort=='home' else [])+(['log1p_reads'] if block=='repertoire' else []))
  M.CATEGORICAL_CONFOUNDERS=('sex','batch','neoadjuvant') if cohort=='home' else ('batch',)
  work=frame.copy()
  transform='original'
  # skewed counts use log1p only in adjusted logistic; rank test unaffected
  if metric in ['unique_aa','total_reads','input_reads']:
   work[metric]=np.log1p(work[metric]);transform='log1p'
  a=M.adjusted_test(work,metric,name);a.update(cohort=cohort,block=block,label=label,transform=transform,family='expanded12',family_size=12)
  ad.append(a)
 u=pd.DataFrame(un);a=pd.DataFrame(ad);u['q']=bh(u.P);a['q']=bh(a.P)
 return u,a
us=[];ads=[]
for cohort,n0,np_ in [('home',70,56),('external',109,54)]:
 f=d[d.cohort.eq(cohort)&d.n_binary.isin([0,1])].copy();assert (sum(f.n_binary.eq(0)),sum(f.n_binary.eq(1)))==(n0,np_)
 u,a=analyses(f,'known_recorded_N',cohort);us.append(u);ads.append(a);print(cohort,'primary completed',flush=True)
 if cohort=='home':
  for name,sub in [('untreated',f[M.untreated_mask(f)]),(_private_literal('private_044ace73522ed623'),f[~getattr(M, _private_literal('private_5b340947280f76b5'))(f)])]:
   u,a=analyses(sub,name,cohort);us.append(u);ads.append(a)
u=pd.concat(us,ignore_index=True);a=pd.concat(ads,ignore_index=True)
u.to_csv(T/'expanded12_unadjusted.tsv',sep='\t',index=False);a.to_csv(T/'expanded12_adjusted.tsv',sep='\t',index=False)
# mathematical complement uses same inferential test/family; no new P-value
alias=u[u.metric.eq('normalized_shannon')].copy();alias['metric']='clonality';alias['label']='Clonality (1 - normalized Shannon diversity)'
for c in ['median_N0','median_Nplus']:alias[c]=1-alias[c]
for stem in ['N0','Nplus']:
 lo=1-alias['q75_'+stem];hi=1-alias['q25_'+stem];alias['q25_'+stem]=lo;alias['q75_'+stem]=hi
alias['median_delta']=-alias.median_delta;lo=-alias.median_delta_CIhi;hi=-alias.median_delta_CIlo;alias['median_delta_CIlo']=lo;alias['median_delta_CIhi']=hi;alias['rank_biserial']=-alias.rank_biserial
alias['method']='Exact affine-complement alias of normalized Shannon; same P/q, not an additional test'
alias.to_csv(T/'clonality_display_alias.tsv',sep='\t',index=False)
legacy=B/'revision_v5/legacy/sources/local/0506/figures_nature_v4/source_data/r2_mixcr_qc_per_sample.tsv'
old=pd.read_csv(legacy,sep='\t');now=d[d.cohort.eq('home')].copy()
legacyqc=old.merge(now,left_on='sample_id',right_on='primary_sample_ids',suffixes=('_legacy','_current'),validate='one_to_one')
cols=['sample_id','patient_key','n_binary','pN_label','alignment_rate','alignment_fraction','n_clonotypes_TRB','unique_aa','shannon_TRB','shannon_entropy','clonality_TRB','clonality','total_TRB_reads','total_reads_current']
legacyqc[cols].to_csv(A/'legacy_current_QC_reconciliation.tsv',sep='\t',index=False)
# primary three metric tests remain untouched; source as companion rather than reassigning new q
for f in ['Figure2_descriptive_tests_v4.tsv','Figure2_adjusted_effects.tsv','Supplement_VJ_all78.tsv','unified_nodal_evidence.tsv','Figure3_historical_representation_metrics.tsv','Supplement_S5_all_historical_metrics.tsv','Supplement_representation_patient_medians.tsv']:
 src=B/'revision_v5/nodal/source_tables'/f;track(src,'unchanged historical/current primary result')
 pd.read_csv(src,sep='\t').to_csv(T/f,sep='\t',index=False)
pd.DataFrame(sources).to_csv(A/'source_manifest.tsv',sep='\t',index=False)
checks=dict(status='PASS',patients_all=len(d),home_known_N=126,external_known_N=163,metrics=12,clonality='exact normalized Shannon complement; not extra hypothesis',input_hashes_verified=len(sources),current_original_summary_features_reconcile=True,all_unadjusted_tests=len(u),all_adjusted_tests=len(a),adjusted_invalid=int(a.status.ne('ok').sum()),home_BH_significant=u[(u.cohort=='home')&(u.analysis=='known_recorded_N')&(u.q<.05)].metric.tolist(),external_BH_significant=u[(u.cohort=='external')&(u.analysis=='known_recorded_N')&(u.q<.05)].metric.tolist())
(A/'validation.json').write_text(json.dumps(checks,indent=2))
print(json.dumps(checks,indent=2))

