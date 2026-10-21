#!/usr/bin/env python3
"""Frozen reference-level and patient-level antigen-context bridge; remote only."""
from release_config import release_path, private_value
from pathlib import Path
from collections import defaultdict
import hashlib,json,re,sys,platform,itertools,time
import numpy as np
import pandas as pd
from scipy.stats import false_discovery_control

ROOT=Path(str(release_path('v7_antigen_bridge_source_10')))
V6=ROOT.parent/'revision_v6'; V5=ROOT.parent/'revision_v5'
OUT=ROOT/'antigen_bridge'; SEED=20261004
REFSRC=V5/'antigen/inputs/normalized_reference_records.tsv'
ENDPOINTS=['any_reference','viral','human_self','tumor_context']
ANNOTATION_KEYS=['exact_AA_VJ','exact_AA_only','exact_AA_VJ_VDJdb_score_ge1']
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for x in iter(lambda:f.read(1048576),b''):h.update(x)
 return h.hexdigest()
def save(df,name):
 df.to_csv(OUT/name,sep='\t',index=False,na_rep='NA')
def gene(s,prefix):
 s=str(s).strip().split('*')[0]
 return s if re.fullmatch(prefix+r'\d+(?:-\d+)?',s) else ''
def splitwords(x):return sorted(set(y.strip() for y in re.split('[;|]',str(x)) if y.strip()))
def joint(x):return ';'.join(sorted(set(str(y) for y in x if str(y) not in ['', 'nan'])))
def testdiff(x):
 x=np.asarray(x,dtype=float);n=len(x)
 if n==0:return np.nan,np.nan,np.nan,np.nan
 rng=np.random.default_rng(SEED+n)
 boot=x[rng.integers(0,n,(10000,n))].mean(1)
 lo,hi=np.quantile(boot,[.025,.975]);effect=x.mean()
 if n<8:p=np.nan
 elif np.all(x==0):p=1.
 else:
  vals=np.arange(1<<n,dtype=np.uint32)
  sims=np.zeros(len(vals))
  for i,a in enumerate(x):sims+=np.where((vals>>i)&1,a,-a)
  p=float(np.mean(np.abs(sims/n)>=abs(effect)-1e-12))
 return effect,lo,hi,p
def bhframe(df,group):
 df['q']=np.nan
 for _,ids in df.groupby(group,dropna=False).groups.items():
  ids=list(ids);good=[i for i in ids if pd.notna(df.loc[i,'p'])]
  # Keep the prespecified family size even when a sparse endpoint is not testable.
  adjusted=false_discovery_control(df.loc[ids,'p'].fillna(1.).to_numpy())
  if good:df.loc[good,'q']=pd.Series(adjusted,index=ids).loc[good]
 return df

def inventory():
 prev=pd.read_csv(V5/'antigen/inputs/reference_inventory.tsv',sep='\t')
 prev['current_sha256']=prev.source_file.map(sha)
 assert (prev.sha256==prev.current_sha256).all(),'Raw reference drift'
 prev['source_unchanged']=True
 prev['v7_use']=np.where(prev.included,'human_beta_original','inventory_only')
 prev.loc[prev.source_file.str.endswith('EDAR_colorectal_cancer.xlsx'),'v7_use']='CRC_epitope_context_not_receptor_evidence'
 save(prev,'reference_inventory.tsv')
 assert sha(REFSRC)=='6649bacb231110d7d886c13689bb557625f7be088931707d0098388f89c995aa'
 ref=pd.read_csv(REFSRC,sep='\t',dtype=str).fillna('')
 ref['v_gene']=ref.ref_v.map(lambda x:gene(x,'TRBV'));ref['j_gene']=ref.ref_j.map(lambda x:gene(x,'TRBJ'))
 ref['key_AA_VJ']=ref.ref_cdr3beta+'|'+ref.v_gene+'|'+ref.j_gene
 ref['valid_gene_resolution']=ref.v_gene.ne('')&ref.j_gene.ne('')
 ref['specificity_record']=ref.reference_scope.eq('specificity_annotation') & (ref.epitope.ne('')|ref.antigen.ne(''))
 assert ref.ref_cdr3beta.str.fullmatch('C[ACDEFGHIKLMNPQRSTVWY]+[FW]').all()
 crc=pd.read_excel(str(release_path('v7_antigen_bridge_source_66')),header=[0,1],dtype=str).fillna('')
 crc.columns=['|'.join(c) for c in crc.columns]
 save(crc,'CRC_curated_epitope_catalog.tsv')
 peptides=set(crc['Epitope|Name'])-{''}
 ref['CRC_epitope_context']=ref.epitope.map(lambda x:any(y in peptides for y in re.split('[,;|]',x)))
 # Source categories are audited inherited categories; CRC epitope context is separate.
 ref['any_reference']=ref.specificity_record
 ref['viral']=ref.specificity_record & ref.antigen_class.eq('viral')
 ref['human_self']=ref.specificity_record & ref.antigen_class.isin(['human_or_self','autoimmune_or_self'])
 ref['tumor_context']=ref.specificity_record & ref.antigen_class.eq('tumor_associated')
 ref['score_ge1']=ref.database.eq('VDJdb') & pd.to_numeric(ref.database_score,errors='coerce').ge(1)
 cols=['ref_cdr3beta','v_gene','j_gene','epitope','mhc','antigen_class','reference_scope','antigen','source_organism']
 ref['annotation_id']=ref[cols].astype(str).agg('|'.join,axis=1).map(lambda x:hashlib.sha256(x.encode()).hexdigest()[:20])
 save(ref,'reference_source_records.tsv.gz')
 agg={k:'max' for k in ENDPOINTS+['CRC_epitope_context','score_ge1','valid_gene_resolution','specificity_record']}
 agg.update({c:joint for c in ['database','reference_key','reference_id','pmid','source_file','source_rows','assay_method']})
 canon=ref.groupby(['annotation_id']+cols,dropna=False,sort=False).agg(agg).reset_index()
 canon['key_AA_VJ']=canon.ref_cdr3beta+'|'+canon.v_gene+'|'+canon.j_gene
 save(canon,'reference_canonical.tsv.gz')
 save(ref.groupby(['database','reference_scope','antigen_class']).size().rename('n_source_records').reset_index(),'reference_counts.tsv')
 return ref,canon

def annotations(ref,canon):
 b=pd.read_parquet(V6/'cohort/baseline_cells.parquet')
 assert b.cell_name.is_unique
 m=pd.read_parquet(V6/'bridge/cell_membership.parquet');m=m[m.own_threshold.eq(1)].copy()
 assert m.cell_name.is_unique and len(m)==53886
 c=m.merge(b[['cell_name','SubCellType','is_cd8','beta_aa','beta_v','beta_j','program_cytotoxic','program_exhaustion','program_naive_memory','program_interferon','program_residency','program_proliferation']],on='cell_name',validate='one_to_one')
 keys=c[['key_AA_VJ','beta_aa','beta_v','beta_j']].drop_duplicates('key_AA_VJ').copy()
 long=[]
 for identity in ANNOTATION_KEYS:
  rr=canon[canon.specificity_record].copy()
  if identity!='exact_AA_only':rr=rr[rr.valid_gene_resolution]
  if identity.endswith('ge1'):rr=rr[rr.score_ge1]
  group='ref_cdr3beta' if identity=='exact_AA_only' else 'key_AA_VJ'
  g=rr.groupby(group)[ENDPOINTS+['CRC_epitope_context']].max()
  qkey=keys.beta_aa if identity=='exact_AA_only' else keys.key_AA_VJ
  for endpoint in ENDPOINTS+['CRC_epitope_context']:
   col=identity+'__'+endpoint
   keys[col]=qkey.map(g[endpoint]).fillna(False).astype(bool)
  ids=rr.groupby(group).annotation_id.agg(joint)
  keys[identity+'__annotation_ids']=qkey.map(ids).fillna('')
  hits=keys[['key_AA_VJ','beta_aa']].merge(rr,left_on='beta_aa' if identity=='exact_AA_only' else 'key_AA_VJ',right_on=group,suffixes=('','_ref'))
  hits['identity']=identity;long.append(hits)
 hits=pd.concat(long,ignore_index=True);save(hits,'tissue_reference_members.tsv.gz')
 save(keys,'tissue_key_annotations.tsv')
 c=c.merge(keys.drop(columns=['beta_aa','beta_v','beta_j']),on='key_AA_VJ',validate='many_to_one')
 c.to_parquet(OUT/'cell_membership.parquet',index=False)
 save(c.groupby(['patient_id','Tissue','category','high_depth_primary']).agg(n_cells=('cell_name','size'),n_clones=('beta_clone','nunique'),n_AA_VJ=('key_AA_VJ','nunique')).reset_index(),'patient_group_eligibility.tsv')
 return c,keys,hits

def patient_analysis(c):
 rows=[];contrasts=[];standard=[];elig=[]
 primary=c[c.high_depth_primary].copy();assert primary.patient_id.nunique()==18
 for (pid,tis),sub in primary.groupby(['patient_id','Tissue']):
  size=sub.groupby('beta_clone').size();sub=sub.copy();sub['clone_size']=sub.beta_clone.map(size)
  sub['size_band']=pd.cut(sub.clone_size,[0,1,4,np.inf],labels=['1','2-4','>=5']).astype(str)
  counts=sub.groupby('category').agg(n_cells=('cell_name','size'),n_clones=('beta_clone','nunique'))
  aok=all(cat in counts.index and counts.loc[cat,'n_cells']>=20 and counts.loc[cat,'n_clones']>=5 for cat in ['own_only','bulk_only'])
  elig.append({'patient_id':pid,'tissue':tis,'eligible_primary':aok,'own_cells':int(counts.loc['own_only','n_cells']) if 'own_only' in counts.index else 0,'bulk_cells':int(counts.loc['bulk_only','n_cells']) if 'bulk_only' in counts.index else 0,'own_clones':int(counts.loc['own_only','n_clones']) if 'own_only' in counts.index else 0,'bulk_clones':int(counts.loc['bulk_only','n_clones']) if 'bulk_only' in counts.index else 0})
  for metric in ['cell','NT_clone']:
   ss=sub if metric=='cell' else sub.drop_duplicates('beta_clone')
   for identity in ANNOTATION_KEYS:
    for endpoint in ENDPOINTS:
     col=identity+'__'+endpoint
     gg=ss.groupby('category')[col].agg(['mean','sum','size'])
     for cat in ['both','own_only','bulk_only','neither']:
      r=gg.loc[cat] if cat in gg.index else pd.Series({'mean':np.nan,'sum':0,'size':0})
      rows.append({'patient_id':pid,'tissue':tis,'bridge_category':cat,'identity':identity,'metric':metric,'endpoint':endpoint,'n_total':r['size'],'n_annotated':r['sum'],'fraction':r['mean']})
     if all(cat in gg.index for cat in ['own_only','bulk_only']):
      contrasts.append({'patient_id':pid,'tissue':tis,'identity':identity,'metric':metric,'endpoint':endpoint,'eligible':aok,'own_fraction':gg.loc['own_only','mean'],'bulk_fraction':gg.loc['bulk_only','mean'],'difference_pp':100*(gg.loc['own_only','mean']-gg.loc['bulk_only','mean'])})
  # Control library/publicness differences using within-patient matched receptor strata.
  ss=sub.drop_duplicates('beta_clone').copy();ss['length']=ss.beta_aa.str.len();strata=['beta_v','beta_j','length','size_band']
  ss=ss[ss.category.isin(['own_only','bulk_only'])]
  nn=ss.groupby(strata+['category']).size().unstack('category',fill_value=0)
  for cat in ['own_only','bulk_only']:
   if cat not in nn:nn[cat]=0
  nn['weight']=nn[['own_only','bulk_only']].min(axis=1);nn=nn[nn.weight>0]
  match=ss.merge(nn[['weight']],left_on=strata,right_index=True,how='inner')
  matched=match.groupby('category').agg(n_cells=('clone_size','sum'),n_clones=('beta_clone','size'))
  eligible=aok and all(cat in matched.index and matched.loc[cat,'n_cells']>=20 and matched.loc[cat,'n_clones']>=5 for cat in ['own_only','bulk_only'])
  for ep in ENDPOINTS:
   gg=match.groupby(strata+['category'])['exact_AA_VJ__'+ep].mean().unstack('category') if len(match) else pd.DataFrame()
   d=100*np.average(gg['own_only']-gg['bulk_only'],weights=nn.loc[gg.index,'weight']) if len(gg) else np.nan
   standard.append({'patient_id':pid,'tissue':tis,'endpoint':ep,'eligible':eligible,'n_common_strata':len(nn),'matched_weight':nn.weight.sum(),'own_matched_clones':matched.loc['own_only','n_clones'] if 'own_only' in matched.index else 0,'bulk_matched_clones':matched.loc['bulk_only','n_clones'] if 'bulk_only' in matched.index else 0,'difference_pp':d})
 save(pd.DataFrame(rows),'patient_annotation_fractions.tsv');co=pd.DataFrame(contrasts);save(co,'patient_annotation_contrasts.tsv');save(pd.DataFrame(elig),'annotation_eligibility.tsv')
 tests=[]
 for identity,metric,tis,ep in itertools.product(ANNOTATION_KEYS,['cell','NT_clone'],['Tumor','Normal'],ENDPOINTS):
  ss=co[(co.identity==identity)&(co.metric==metric)&(co.tissue==tis)&(co.endpoint==ep)&co.eligible]
  effect,lo,hi,p=testdiff(ss.difference_pp)
  tests.append(dict(identity=identity,metric=metric,tissue=tis,endpoint=ep,contrast='own_only_minus_bulk_only',n_patients=len(ss),effect_pp=effect,ci_low_pp=lo,ci_high_pp=hi,p=p,mean_own_pct=100*ss.own_fraction.mean(),mean_bulk_pct=100*ss.bulk_fraction.mean(),family='primary16' if identity=='exact_AA_VJ' else identity+'_sensitivity16',status='inferential' if len(ss)>=8 else 'descriptive_insufficient_donors'))
 tests=bhframe(pd.DataFrame(tests),'family');save(tests,'annotation_tests.tsv')
 st=pd.DataFrame(standard);save(st,'background_matched_patient_contrasts.tsv');s=[]
 for tis,ep in itertools.product(['Tumor','Normal'],ENDPOINTS):
  ss=st[(st.tissue==tis)&(st.endpoint==ep)&st.eligible];e,l,h,p=testdiff(ss.difference_pp)
  s.append(dict(tissue=tis,endpoint=ep,n_patients=len(ss),effect_pp=e,ci_low_pp=l,ci_high_pp=h,p=p,family='background_matched8',status='inferential' if len(ss)>=8 else 'descriptive_insufficient_donors'))
 save(bhframe(pd.DataFrame(s),'family'),'background_matched_tests.tsv')
 # Descriptive programs and complete32state support: no cell-level statistical inference.
 out=[]
 for (pid,tis,cat),ss in primary.groupby(['patient_id','Tissue','category']):
  total=len(ss)
  for state,n in ss.SubCellType.value_counts().items():
   out.append(dict(patient_id=pid,tissue=tis,bridge_category=cat,state=state,n_cells=n,n_total=total,fraction=n/total))
 save(pd.DataFrame(out),'state_descriptions.tsv')
 refstates=[]
 for (pid,tis,cat,annot),ss in primary.groupby(['patient_id','Tissue','category','exact_AA_VJ__viral']):
  for state in sorted(primary.SubCellType.unique()):
   n=int(ss.SubCellType.eq(state).sum())
   refstates.append(dict(patient_id=pid,tissue=tis,bridge_category=cat,viral_reference=bool(annot),state=state,n_cells=n,n_total=len(ss),fraction=n/len(ss)))
 save(pd.DataFrame(refstates),'reference_state_descriptions.tsv')
 ps=[x for x in primary if x.startswith('program_')]
 clones=primary.groupby(['patient_id','Tissue','category','beta_clone'])[ps+['exact_AA_VJ__any_reference','exact_AA_VJ__viral']].mean().reset_index()
 pr=clones.groupby(['patient_id','Tissue','category','exact_AA_VJ__any_reference'])[ps].agg(['mean','count']).reset_index();pr.columns=['|'.join(x).rstrip('|') if isinstance(x,tuple) else x for x in pr.columns];save(pr,'reference_program_descriptions.tsv')
 return tests

def candidates(ref,canon,keys):
 variants=pd.read_csv(V6/'candidates/observed_bulk_NT_VJ_variants.tsv',sep='\t').fillna('')
 branches=variants[['query','AA','V','J']].drop_duplicates();save(branches,'candidate_observed_branches.tsv')
 results=[];near=[]
 for row in branches.itertuples(index=False):
  rr=canon[canon.ref_cdr3beta.eq(row.AA)]
  strict=rr[rr.v_gene.eq(row.V)&rr.j_gene.eq(row.J)]
  results.append(dict(candidate=row.query,AA=row.AA,V=row.V,J=row.J,exact_AA_reference_annotations=len(rr),exact_AA_VJ_reference_annotations=len(strict),exact_AA_VJ_specificity_annotations=int(strict.specificity_record.sum()),reference_annotation_ids=joint(strict.annotation_id),patient_alpha_available=False,patient_HLA_available=False))
  # Fixed one-substitution bound; all reference annotation rows retained for a matched sequence.
  rr=canon[canon.ref_cdr3beta.str.len().eq(len(row.AA))].copy()
  mask=rr.ref_cdr3beta.map(lambda a:sum(x!=y for x,y in zip(a,row.AA))==1)
  for r in rr[mask].to_dict('records'):
   near.append(dict(candidate=row.query,query_AA=row.AA,query_V=row.V,query_J=row.J,method='completeAA_one_substitution',AA_substitutions=1,VJ_compatible=(r['v_gene']==row.V and r['j_gene']==row.J),**r))
 save(pd.DataFrame(results),'candidate_branch_annotations.tsv');save(pd.DataFrame(near),'candidate_neighbor_members.tsv.gz')
 prior=pd.read_csv(V6/'candidates/branch_specific_antigen_annotations.tsv',sep='\t');save(prior,'candidate_inherited_TCRMatch_tcrdist_source_evidence.tsv')
 hist=V6/'candidates/figures/Supplementary_Figure_S7.source_export.json'
 (OUT/'candidate_filter_trace.md').write_text('''# Historical nomination trace\n\nQ1/Q2 remain historically nominated, not newly selected from the present tissue cohort. Reuse the v6 audited parallel selection records: reported AA background1,367,835; dataset-aware branch106candidates; six-criterion branch reportedQ1/Q2 (CRC P<.01, N P<.01, CRC FDR<.20, TIL support, CRC enrichment, N+ enrichment). Per-filter complete membership/counts were not recovered. N tests were restricted to top2,000, CMH top10,000; these are historical design facts and not the current126-patient inferential family. No new sequential attrition numbers are invented. Source v6 candidates/figures/Supplementary_Figure_S7.source_export.json. Current eligibility, thresholds and patient differences are independently reevaluated by q1_depth_n lane.\n''')

def gliph_input(c):
 d=OUT/'gliph2';d.mkdir(exist_ok=True)
 p=c[c.high_depth_primary].copy();nn=p.groupby(['patient_id','Tissue','beta_clone']).size().rename('n').reset_index();expanded=nn[nn.n>=2]
 x=p.merge(expanded[['patient_id','Tissue','beta_clone']],on=['patient_id','Tissue','beta_clone'],how='inner')
 x=x[['patient_id','key_AA_VJ','beta_aa','beta_v','beta_j']].drop_duplicates(['patient_id','key_AA_VJ']).sort_values(['patient_id','key_AA_VJ'])
 # Each patient/receptor occurs once, no duplicated tumor+normal or clone-size weights.
 inp=pd.DataFrame({'#CDR3b':x.beta_aa,'TRBV':x.beta_v,'TRBJ':x.beta_j,'CDR3c':x.beta_aa,'CDR3p':'NA','subject:condition':x.patient_id+':CRC_tissue','Frequency':1})
 inp.to_csv(d/'expanded_tissue_unweighted.tsv',sep='\t',index=False)
 x.to_csv(d/'foreground_members.tsv',sep='\t',index=False)
 bg=Path(str(release_path('v7_antigen_bridge_source_208')))
 (d/'hla_unavailable.tsv').write_text('')
 params={'out_prefix':'tissue_expanded','hla_file':str(d/'hla_unavailable.tsv'),'cdr3_file':str(d/'expanded_tissue_unweighted.tsv'),'refer_file':str(bg/'ref_CD48_v2.0.reconstructed.txt'),'v_usage_freq_file':str(bg/'ref_V_CD48_v2.0.reconstructed.txt'),'cdr3_length_freq_file':str(bg/'ref_L_CD48_v2.0.reconstructed.txt'),'local_min_pvalue':.001,'p_depth':10000,'global_convergence_cutoff':1,'simulation_depth':10000,'kmer_min_depth':3,'local_min_OVE':10,'algorithm':'GLIPH2','all_aa_interchangeable':1}
 (d/'params.txt').write_text('\n'.join(f'{k}={v}' for k,v in params.items())+'\n')
 (d/'input_receipt.json').write_text(json.dumps({'n_patient_receptor_rows':len(x),'n_unique_AA':x.beta_aa.nunique(),'n_unique_AA_VJ':x.key_AA_VJ.nunique(),'n_patients':x.patient_id.nunique(),'expanded_criterion':'at least2cells in patient/tissue NT clone; one patient/AA-VJ row','parameters':params,'input_sha256':sha(d/'expanded_tissue_unweighted.tsv'),'background_sha256':{k:sha(v) for k,v in params.items() if k in ['refer_file','v_usage_freq_file','cdr3_length_freq_file']},'patient_HLA_unavailable':True,'no_candidate_or_antigen_label_used':True},indent=2))

def main():
 start=time.time();OUT.mkdir(exist_ok=True);ref,canon=inventory();print('Reference',len(ref),len(canon),flush=True)
 c,k,h=annotations(ref,canon);print('Tissue',len(c),len(k),len(h),flush=True)
 t=patient_analysis(c);candidates(ref,canon,k);gliph_input(c)
 paths=[REFSRC,V6/'cohort/baseline_cells.parquet',V6/'bridge/cell_membership.parquet',V6/'candidates/observed_bulk_NT_VJ_variants.tsv',V6/'candidates/branch_specific_antigen_annotations.tsv',Path(__file__),OUT/'PROTOCOL.md']
 save(pd.DataFrame([dict(path=str(p),bytes=p.stat().st_size,sha256=sha(p)) for p in paths]),'input_manifest.tsv')
 receipt={'status':'CORE_COMPLETED','execution_host':platform.node(),'python':sys.version,'seed':SEED,'runtime_seconds':time.time()-start,'n_reference_original_rows':len(ref),'n_reference_collapsed_annotations':len(canon),'n_tissue_AA_VJ':len(k),'n_tissue_cells':len(c),'n_patients_primary':18,'n_primary_tests':16,'n_sensitivity_tests':32,'n_background_tests':8,'complete_testing_families':True,'molecular_docking_run':False,'antigen_binding_claim':False,'main_script_sha256':sha(__file__)}
 (OUT/'core_receipt.json').write_text(json.dumps(receipt,indent=2)+'\n');print(t[t.family.eq('primary16')].to_string(index=False));print(json.dumps(receipt))
if __name__=='__main__':main()
