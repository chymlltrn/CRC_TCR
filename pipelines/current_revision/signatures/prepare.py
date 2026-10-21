import os
from pathlib import Path
import numpy as np, pandas as pd, json, hashlib
from datetime import datetime,timezone
OUT=Path(os.environ['CRC_SIGNATURE_ROOT']).expanduser().resolve()
SRC=Path(os.environ['CRC_SIGNATURE_SOURCE']).expanduser().resolve()
CONFIG=json.loads(Path(os.environ['CRC_SIGNATURE_CONFIG']).read_text())
OUT.mkdir(parents=True,exist_ok=True)
if (OUT/'freeze.json').exists(): raise RuntimeError('Use a fresh signature output root; a freeze already exists')
for d in ['inputs','de','tables','audit','scripts']: (OUT/d).mkdir(exist_ok=True)
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for z in iter(lambda:f.read(8*1024*1024),b''):h.update(z)
 return h.hexdigest()
state_names=['c03_CD4_Tn_NR4A2','c08_CD4_Tfh_CXCL13_IL6ST','c20_CD8_Tem_GZMK','c27_CD8_MAIT_SLC4A10']
protocol={'frozen_utc':datetime.now(timezone.utc).isoformat(),'source_scope':'GSE236581 baseline CRC Tumor all RNA T cells; source non-CRC donors excluded upstream and configured donor exclusions align current paired cohort','states':state_names,'primary':['c03','c20','c27'],'secondary':['c08'],'comparison':'each author state versus all other RNA T cells within patient; no TCR availability filter','minimum_cells_each_side':10,'DE':'edgeR 4.6.2 robust QL legacy=FALSE; paired design donor+condition; TMM; filterByExpr(min.count=10,min.total.count=15,large.n=10,min.prop=0.7)','selection':{'log2FC_gt':0.5,'BH_per_contrast_lt':0.05,'donor_direction_fraction_ge':0.7,'donor_equal_detection_target_ge':0.1,'donor_equal_detection_target_minus_rest_ge':0.05,'ranking':'specificity_score = log2FC * (donor_equal_pct_target - donor_equal_pct_rest), descending; ties log2FC descending, FDR ascending, gene lexical','cap_genes':30,'no_padding_if_under30':True,'minimum_for_release':10},'exclusions':'TCR ^TR[ABDG][CVJD]; IG ^IG[HKL]; MT ^MT-; ribosomal ^RP[SL][0-9]+[A-Z]?[0-9]*(P[0-9]+)?$|^RPLP[0-9]+(P[0-9]+)?$|^RPSA(P[0-9]+)?$|^MRP[SL][0-9]+','detection':'actual raw count>0 per cell, streamed from unique-symbol raw MTX, aggregate donor/state; no pseudobulk-positive surrogate','stability':'paired donor direction fraction; leave-one-donor-out mean effect direction and rank-score subtype specificity, descriptive on training cohort','bulk_limit':'relative expression signature score, not calibrated abundance; c27 vs other T does not establish specificity versus c26 MAIT','outcome_access':'Historical protocol reconstructed for reproducibility; this rerun does not attest to pre-outcome access timing'}
(OUT/'protocol_frozen.json').write_text(json.dumps(protocol,indent=2)+'\n')
m=pd.read_csv(SRC/'pseudobulk_metadata.tsv',sep='\t')
g=pd.read_csv(SRC/'gene_map.tsv',sep='\t')
print('gene_map',g.columns.tolist(),g.head().to_dict('records'),flush=True)
z=np.load(SRC/'pseudobulk_counts.npz',allow_pickle=False);genes=z['genes'].astype(str)
assert len(genes)==36027 and len(set(genes))==len(genes)
assert np.array_equal(z['pseudobulk_ids'].astype(str),m.pseudobulk_id.astype(str))
raw=np.memmap(SRC/'raw_counts.bin',dtype='<i8',mode='r',shape=(len(m),len(genes)))
assert np.array_equal(raw.sum(axis=1),m.metadata_total_counts.to_numpy())
donors=sorted(set(m.patient_id)-set(CONFIG['excluded_donor_ids']))
assert len(donors)==CONFIG['expected_n_donors'] and not set(donors)&set(CONFIG['forbidden_donor_ids'])
rows=[];arrays=[];old_to_new=np.full(len(m),-1,np.int32)
for donor in donors:
 for state in state_names+['other_T_excluding_four']:
  keep=(m.patient_id==donor)&(m.MajorCellType=='T')
  keep &= (m.SubCellType==state) if state!='other_T_excluding_four' else ~m.SubCellType.isin(state_names)
  ids=np.flatnonzero(keep)
  row={'group_index':len(rows),'patient_id':donor,'group':state,'n_cells':int(m.loc[keep,'n_cells'].sum()),'metadata_total_counts':int(m.loc[keep,'metadata_total_counts'].sum()),'source_group_count':len(ids)}
  rows.append(row);arrays.append(np.asarray(raw[ids,:].sum(axis=0)))
  old_to_new[ids]=row['group_index']
cm=np.fromfile(SRC/'cell_group.bin',dtype=np.int32)
new=np.full(len(cm),-1,np.int32);sel=cm>=0;new[sel]=old_to_new[cm[sel]]
new.tofile(OUT/'inputs/cell_group.bin')
np.fromfile(SRC/'gene_map.bin',dtype=np.int32).tofile(OUT/'inputs/gene_map.bin')
assert np.array_equal(np.bincount(new[new>=0],minlength=len(rows)),[x['n_cells'] for x in rows])
meta=pd.DataFrame(rows);meta.to_csv(OUT/'inputs/groups.tsv',sep='\t',index=False)
counts=np.stack(arrays);np.savez_compressed(OUT/'inputs/pseudobulk_19patients_5groups.npz',counts=counts,genes=genes)
pd.Series(genes).to_csv(OUT/'inputs/genes.tsv',sep='\t',index=False,header=False)
elig=[]
for state in state_names:
 samples=[];mat=[]
 for donor in donors:
  ids=meta.index[meta.patient_id==donor];target=ids[meta.loc[ids,'group'].eq(state)];rest=ids[~meta.loc[ids,'group'].eq(state)]
  nt=int(meta.loc[target,'n_cells'].sum());nr=int(meta.loc[rest,'n_cells'].sum());ok=nt>=10 and nr>=10
  elig.append({'state':state,'patient_id':donor,'target_n_cells':nt,'rest_n_cells':nr,'included':ok})
  if not ok:continue
  for condition,ii in [('rest',rest),('target',target)]:
   sid=donor+'__'+condition;samples.append({'sample_id':sid,'patient_id':donor,'condition':condition,'n_cells':int(meta.loc[ii,'n_cells'].sum())});mat.append(counts[ii].sum(axis=0))
 out=OUT/'inputs'/state;out.mkdir(exist_ok=True)
 pd.DataFrame(samples).to_csv(out/'samples.tsv',sep='\t',index=False)
 pd.DataFrame(np.stack(mat).T,index=genes,columns=[x['sample_id'] for x in samples]).rename_axis('gene').to_csv(out/'counts.tsv.gz',sep='\t')
 print(state,'donors',len(samples)//2,'cells',sum(x['n_cells'] for x in samples if x['condition']=='target'),flush=True)
pd.DataFrame(elig).to_csv(OUT/'tables/patient_eligibility.tsv',sep='\t',index=False)
manifest=[]
for p in [SRC/'pseudobulk_metadata.tsv',SRC/'gene_map.tsv',SRC/'pseudobulk_counts.npz',SRC/'raw_counts.bin',SRC/'cell_group.bin',SRC/'gene_map.bin']:
 manifest.append({'path':str(p),'bytes':p.stat().st_size,'sha256':sha(p)})
pd.DataFrame(manifest).to_csv(OUT/'audit/input_manifest.tsv',sep='\t',index=False)
(OUT/'audit/preparation.json').write_text(json.dumps({'donors':donors,'n_donors':19,'all_T_cells':int(meta.n_cells.sum()),'n_genes':len(genes),'matrix_aggregation_counts_match':True,'cell_mapping_counts_match':True,'protocol_sha256':sha(OUT/'protocol_frozen.json'),'source_metadata_scope':'source signature_prepare.py baseline CRC Tumor allRNA groups'},indent=2)+'\n')
