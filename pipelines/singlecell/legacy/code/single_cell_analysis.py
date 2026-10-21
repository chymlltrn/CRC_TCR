#!/usr/bin/env python3
"""Remote-only frozen scTCR source joins, exact candidates, and patient counts."""
from release_config import release_path, private_value
from single_cell_common import *
import gzip,time,sys
from datetime import datetime,timezone
OUT.mkdir(parents=True,exist_ok=True)
print('prepare start',datetime.now(timezone.utc).isoformat(),flush=True)
m=pd.read_csv(SOURCE/'GSE236581_CRC-ICB_metadata.txt.gz',sep=r'\s+',index_col=0)
v=pd.read_csv(SOURCE/'GSE236581_VDJ_merge.txt.gz',sep=r'\s+',index_col=0)
assert m.index.is_unique and v.cell_name.is_unique
official=ROOT/'literature/sources/Chen2024_official_mmc2.xlsx'
assert official.exists(),'official patient/sample supplement required'
patient=pd.read_excel(official,sheet_name='scRNA-seq patient meta',header=1).dropna(subset=['Patient ID'])
sample=pd.read_excel(official,sheet_name='scRNA-seq sample meta',header=1).dropna(subset=['Sample ID'])
patient=patient.rename(columns={'Patient ID':'patient_id'})
sample=sample.rename(columns={'Sample ID':'clinical_sample_id','Patient ID':'patient_id','Treatment Stage':'treatment_stage','Treatment point':'treatment_point','Biopsy Site':'biopsy_site','Sampling Stage':'sampling_stage'})
assert patient.patient_id.is_unique and sample.clinical_sample_id.is_unique
write(patient,'patient_inventory.tsv')
m['cell_name']=m.index
m['clinical_sample_id']=m.Ident.str.replace(r'^CRC','P',regex=True)
m=m.merge(sample,on='clinical_sample_id',how='left',validate='many_to_one',suffixes=('','_clinical'))
assert m.patient_id.notna().all()
assert (m.Patient==m.patient_id).all()
m=m.merge(patient[['patient_id','Cancer Type','dMMR/pMMR','MSI/MSS','Treatment Regimen']],on='patient_id',validate='many_to_one')
m['is_crc']=m['Cancer Type'].eq('CRC')
m['is_baseline']=m.treatment_stage.eq('Pre') & m.treatment_point.eq('Baseline')
m['eligible_tissue']=((m.Tissue=='Tumor')&m.biopsy_site.eq('Tumor'))|((m.Tissue=='Normal')&m.biopsy_site.eq('Adjacent normal tissue'))
# Print actual clinical labels before asserting allowed join.
print('clinical biopsy labels',sample.biopsy_site.value_counts().to_dict(),flush=True)
if not ((m.Tissue.eq('Tumor') & m.biopsy_site.eq('Tumor')).any()):
 raise RuntimeError('Tumor clinical label differs: inspect source before defining rule')
m['is_cd8']=m.SubCellType.map(cd8)
positions={name:i for i,name in enumerate(v.columns)}
chainrows=[];cellrows=[];set_index=set(m.cell_name)
for k,row in enumerate(v.itertuples(index=False,name=None)):
 cell=row[positions['cell_name']]
 valid={'Alpha':{},'Beta':{}};raw_n={'Alpha':0,'Beta':0}
 for kind,letter in [('Alpha','A'),('Beta','B')]:
  for slot in [1,2]:
   suffix=f'({kind}{slot})'
   aa=row[positions['CDR3'+suffix]];nt=row[positions['CDR3_nt'+suffix]]
   if pd.isna(aa):continue
   raw_n[kind]+=1
   q=accepted(aa,nt,row[positions['V_gene'+suffix]],row[positions['J_gene'+suffix]],row[positions['Full_length'+suffix]],row[positions['nUMI'+suffix]],letter)
   rec=dict(cell_name=cell,chain=kind,slot=slot,aa=clean(aa),nt=clean(nt),v=gene(row[positions['V_gene'+suffix]]),j=gene(row[positions['J_gene'+suffix]]),full_length=str(row[positions['Full_length'+suffix]]),n_umi=row[positions['nUMI'+suffix]],n_read=row[positions['nRead'+suffix]],accepted_proxy=q is not None,rna_joined=cell in set_index)
   chainrows.append(rec)
   if q is not None:valid[kind][q['nt_vj']]=q
 b=list(valid['Beta'].values());a=list(valid['Alpha'].values())
 rec=dict(cell_name=cell,raw_beta_slots=raw_n['Beta'],raw_alpha_slots=raw_n['Alpha'],accepted_distinct_beta=len(b),accepted_distinct_alpha=len(a),single_beta=len(b)==1,strict_paired=len(b)==1 and len(a)==1)
 if len(b)==1:rec.update(beta_aa=b[0]['aa'],beta_nt=b[0]['nt'],beta_v=b[0]['v'],beta_j=b[0]['j'],beta_clone=b[0]['nt_vj'],beta_aa_vj=b[0]['aa_vj'])
 if len(b)==1 and len(a)==1:rec['paired_clone']=a[0]['nt_vj']+'||'+b[0]['nt_vj']
 cellrows.append(rec)
 if (k+1)%100000==0:print('VDJ cells parsed',k+1,flush=True)
chains=pd.DataFrame(chainrows);cell=pd.DataFrame(cellrows)
chains.to_parquet(OUT/'vdj_chain_inventory.parquet',index=False)
cell.to_parquet(OUT/'vdj_cell_chain_audit.parquet',index=False)
m=m.merge(cell,on='cell_name',how='left',validate='one_to_one')
for col in ['single_beta','strict_paired']:m[col]=m[col].fillna(False).astype(bool)
m.to_parquet(OUT/'rna_cell_inventory.parquet',index=False)
t=m[m.MajorCellType.eq('T')].copy()
t.to_parquet(OUT/'cell_inventory.parquet',index=False)
sam=m.groupby(['Ident','clinical_sample_id','patient_id','Cancer Type','Treatment','Tissue','treatment_stage','treatment_point','biopsy_site','eligible_tissue','is_crc','is_baseline'],dropna=False).agg(rna_cells=('cell_name','size'),rna_t_cells=('MajorCellType',lambda s:int(s.eq('T').sum())),rna_joined_vdj_cells=('raw_beta_slots','count'),single_beta_rna_cells=('single_beta','sum'),strict_paired_rna_cells=('strict_paired','sum')).reset_index()
write(sam,'sample_inventory.tsv')
# Exact Q1 raw source chain matches; each scope denominator and unique cells explicit.
beta=chains[chains.chain.eq('Beta')].copy()
definitions={'AA':beta.aa.eq(Q1AA),'AA_VJ':beta.aa.eq(Q1AA)&beta.v.eq('TRBV5-5')&beta.j.eq('TRBJ1-4'),'NT':beta.nt.eq(Q1NT)}
hitrows=[];qsum=[]
rna_names=set(m.cell_name);t_names=set(t.cell_name);primary_names=set(t.loc[t.is_crc&t.is_baseline&t.eligible_tissue,'cell_name'])
for name,hit in definitions.items():
 h=beta.loc[hit].copy();h['definition']=name;hitrows.append(h)
 for scope,names in [('all_source',None),('rna_joined',rna_names),('rna_joined_T',t_names),('crc_baseline_tissue_T',primary_names)]:
  scope_beta=beta if names is None else beta[beta.cell_name.isin(names)]
  for qc in ['source_all','accepted_chain_proxy']:
   hh=h if names is None else h[h.cell_name.isin(names)]
   denom=scope_beta
   if qc=='accepted_chain_proxy':hh=hh[hh.accepted_proxy];denom=denom[denom.accepted_proxy]
   qsum.append(dict(definition=name,scope=scope,qc_scope=qc,beta_bearing_cells=denom.cell_name.nunique(),source_rows=len(denom),q1_cells=hh.cell_name.nunique(),q1_rows=len(hh)))
write(pd.concat(hitrows,ignore_index=True),'q1_exact_hit_rows.tsv')
write(pd.DataFrame(qsum),'q1_exact_query_summary.tsv')
# Pair registry determined from actual QC cell counts, not anticipated n.
base=t[t.is_crc&t.is_baseline&t.eligible_tissue].copy()
rows=[]
for pid in sorted(t.loc[t.is_crc,'patient_id'].unique()):
 pt=base[base.patient_id.eq(pid)]
 a={}
 for tissue in ['Tumor','Normal']:
  x=pt[pt.Tissue.eq(tissue)]
  a[tissue]=dict(rna_t_cells=len(x),single_beta_cells=int(x.single_beta.sum()),strict_paired_cells=int(x.strict_paired.sum()),samples=';'.join(sorted(x.Ident.unique())))
 status='included' if a['Tumor']['single_beta_cells']>=50 and a['Normal']['single_beta_cells']>=50 else 'excluded_missing_tissue_or_lt50_beta'
 rows.append(dict(patient_id=pid,primary_pair_status=status,**{k.lower()+'_'+j:z for k,y in a.items() for j,z in y.items()}))
pairs=pd.DataFrame(rows);write(pairs,'paired_patient_inventory.tsv')
ids=pairs.loc[pairs.primary_pair_status.eq('included'),'patient_id'].tolist()
# Primary cell count metrics and depth matched patient means.
metricsrows=[];rare=[];share=[]
for pid in ids:
 pt=base[base.patient_id.eq(pid)]
 for scheme,flag,key in [('beta_nt_vj','single_beta','beta_clone'),('strict_alpha_beta_nt_vj','strict_paired','paired_clone')]:
  ct={}
  for tissue in ['Tumor','Normal']:
   x=pt[pt.Tissue.eq(tissue)&pt[flag]]
   ct[tissue]=x[key].value_counts()
   z=metrics(ct[tissue].values)
   metricsrows.append(dict(patient_id=pid,tissue=tissue,clone_scheme=scheme,rna_t_cells=int((pt.Tissue==tissue).sum()),capture_fraction=z['tcr_cells']/int((pt.Tissue==tissue).sum()),**z))
  a,b=ct['Tumor'],ct['Normal'];sa,sb=set(a.index),set(b.index);inter=sa&sb
  den=float((a/a.sum()).pow(2).sum()+(b/b.sum()).pow(2).sum()) if a.sum()>0 and b.sum()>0 else np.nan
  dot=sum((a.get(k,0)/a.sum())*(b.get(k,0)/b.sum()) for k in inter) if a.sum()>0 and b.sum()>0 else np.nan
  share.append(dict(patient_id=pid,clone_scheme=scheme,tumor_clones=len(a),normal_clones=len(b),shared_clones=len(inter),jaccard=len(inter)/len(sa|sb) if sa|sb else np.nan,morisita_horn=2*dot/den if den else np.nan,tumor_shared_cell_fraction=a.loc[list(inter)].sum()/a.sum() if a.sum()>0 else np.nan,normal_shared_cell_fraction=b.loc[list(inter)].sum()/b.sum() if b.sum()>0 else np.nan))
  if scheme=='beta_nt_vj':
   target=int(min(a.sum(),b.sum()))
   rng=np.random.default_rng(SEED+int(pid[1:]))
   for tissue,count in ct.items():
    for draw in range(100):
     zz=metrics(rng.multivariate_hypergeometric(count.values,target))
     rare.append(dict(patient_id=pid,tissue=tissue,draw=draw,target_tcr_cells=target,**zz))
write(pd.DataFrame(metricsrows),'patient_pair_metrics.tsv')
rare=pd.DataFrame(rare);write(rare,'rarefaction_draw_metrics.tsv')
write(rare.groupby(['patient_id','tissue','target_tcr_cells'])[['normalized_shannon','top10_fraction','expanded_fraction']].agg(['mean','std']).reset_index().pipe(lambda d:d.set_axis(['_'.join(x).strip('_') if isinstance(x,tuple) else x for x in d.columns],axis=1)),'rarefaction_patient_summary.tsv')
write(pd.DataFrame(share),'patient_sharing_metrics.tsv')
states=sorted(t.SubCellType.unique());assert len(states)==32
state=[]
for pid in ids:
 for tissue in ['Tumor','Normal']:
  x=base[base.patient_id.eq(pid)&base.Tissue.eq(tissue)]
  for s in states:state.append(dict(patient_id=pid,tissue=tissue,subcelltype=s,state_cells=int(x.SubCellType.eq(s).sum()),all_rna_t_cells=len(x),fraction=float(x.SubCellType.eq(s).mean())))
write(pd.DataFrame(state),'patient_state_fractions.tsv')
members=ROOT.parent/'20261003_paper_v1/candidates/frozen17_members.tsv'
families=pd.read_csv(members,sep='\t')
families['key']=families.aa.map(clean)+'|'+families.v_gene.map(gene)+'|'+families.j_gene.map(gene)
keys=families[['group_id','key']].drop_duplicates()
assert keys.group_id.nunique()==17
familyrows=base[base.single_beta][['cell_name','patient_id','Ident','Tissue','SubCellType','beta_aa_vj']].merge(keys,left_on='beta_aa_vj',right_on='key',how='inner')
write(familyrows,'historical17_exact_cell_hits.tsv')
fa=[]
for pid in ids:
 for tissue in ['Tumor','Normal']:
  n=int((base.patient_id.eq(pid)&base.Tissue.eq(tissue)&base.single_beta).sum())
  for group in sorted(keys.group_id.unique()):
   count=familyrows[(familyrows.patient_id==pid)&(familyrows.Tissue==tissue)&(familyrows.group_id==group)].cell_name.nunique()
   fa.append(dict(patient_id=pid,tissue=tissue,group_id=group,matched_cells=count,single_beta_cells=n,fraction=count/n if n else np.nan))
write(pd.DataFrame(fa),'historical17_patient_fractions.tsv')
write(keys.groupby('group_id').size().rename('exact_unique_aa_vj_members').reset_index(),'historical17_dictionary_counts.tsv')
mmr=patient.assign(is_crc=patient['Cancer Type'].eq('CRC'),in_primary_pair=patient.patient_id.isin(ids))
write(mmr.groupby(['is_crc','in_primary_pair','dMMR/pMMR','MSI/MSS'],dropna=False).size().rename('patients').reset_index(),'clinical_mmr_msi_distribution.tsv')
summary=dict(source_patient_count=int(patient.patient_id.nunique()),source_sample_count=int(sample.clinical_sample_id.nunique()),crc_patient_count=int(patient['Cancer Type'].eq('CRC').sum()),primary_paired_patient_count=len(ids),paired_sample_count=int(base[base.patient_id.isin(ids)].Ident.nunique()),paired_ids=ids,primary_tcell_vdj_cells=int(base.loc[base.patient_id.isin(ids),'single_beta'].sum()),primary_all_rna_t_cells=int(base.patient_id.isin(ids).sum()),all_rna_cells=len(m),all_rna_t_cells=len(t),all_source_vdj_cells=len(v),rna_joined_vdj_cells=int(m.raw_beta_slots.notna().sum()),rna_joined_t_vdj_cells=int(t.raw_beta_slots.notna().sum()),unmatched_vdj_cells=int((~v.cell_name.isin(rna_names)).sum()),excluded_non_crc_patients=patient.loc[~patient['Cancer Type'].eq('CRC'),'patient_id'].tolist(),excluded_crc_pairs=pairs.loc[~pairs.primary_pair_status.eq('included'),['patient_id','primary_pair_status']].to_dict('records'),clinical_labels_not_msi_h=True)
(OUT/'cohort_summary.json').write_text(json.dumps(summary,indent=2)+'\n')
inputs=[SOURCE/x for x in ['GSE236581_CRC-ICB_metadata.txt.gz','GSE236581_VDJ_merge.txt.gz','GSE236581_barcodes.tsv.gz','GSE236581_features.tsv.gz','GSE236581_counts.mtx.gz']]+[official,members]
write(pd.DataFrame([dict(path=str(p),bytes=p.stat().st_size,sha256=sha(p),role='author filtered RNA/source VDJ/official clinical/frozen historical candidates') for p in inputs]),'input_manifest.tsv')
receipt=dict(completed_utc=datetime.now(timezone.utc).isoformat(),protocol_sha256=sha(OUT/'PROTOCOL.md'),source_vdj_cells=len(v),source_chain_rows=len(chains),primary_pairs=len(ids),statistics_run=False,productive_flag_available=False,chain_counts=cell[['raw_beta_slots','raw_alpha_slots','accepted_distinct_beta','accepted_distinct_alpha','single_beta','strict_paired']].value_counts().reset_index().rename(columns={'count':'cells'}).to_dict('records'),all_source_q1=qsum)
(OUT/'prepare_receipt.json').write_text(json.dumps(receipt,indent=2,default=int)+'\n')
print('prepare complete',json.dumps(summary),flush=True)
