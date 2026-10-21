#!/usr/bin/env python3
"""One remote sparse-matrix pass for frozen marker program scores."""
from release_config import release_path, private_value
from single_cell_common import *
import gzip,gc,time,resource
from datetime import datetime,timezone
from scipy.io import mmread
from threadpoolctl import threadpool_limits
print('RNA start',datetime.now(timezone.utc).isoformat(),flush=True)
t=pd.read_parquet(OUT/'cell_inventory.parquet')
pairs=json.loads((OUT/'cohort_summary.json').read_text())['paired_ids']
features=pd.read_csv(SOURCE/'GSE236581_features.tsv.gz',sep='\t',header=None)
bars=pd.read_csv(SOURCE/'GSE236581_barcodes.tsv.gz',sep='\t',header=None)[0]
assert bars.is_unique
idx=pd.Index(bars).get_indexer(t.cell_name)
assert (idx>=0).all()
gene_indices={g:np.flatnonzero(features[1].eq(g).values) for g in sorted({x for xs in PROGRAMS.values() for x in xs})}
manifest=[]
for name,genes in PROGRAMS.items():
 available=[g for g in genes if len(gene_indices[g])]
 manifest.append(dict(program=name,fixed_genes=';'.join(genes),available_genes=';'.join(available),missing_genes=';'.join(g for g in genes if not len(gene_indices[g])),available_n=len(available),estimable=len(available)>=3,normalization='log1p(CP10K); denominator total all-gene RNA counts'))
write(pd.DataFrame(manifest),'marker_program_dictionary.tsv')
print('mmread opening 4.23GB gz /1.31billion nnz; max4threads',flush=True)
with threadpool_limits(limits=4):
 with gzip.open(SOURCE/'GSE236581_counts.mtx.gz','rb') as f:mat=mmread(f)
 print('mmread done',mat.shape,mat.nnz,'peakGB',resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1e6,flush=True)
 assert mat.shape==(len(features),len(bars)) and mat.nnz==1310816895
 mat=mat.tocsr()
 total=np.asarray(mat.sum(axis=0)).ravel().astype(np.float64)
 selected_rows=sorted({int(i) for g in gene_indices.values() for i in g})
 small=mat[selected_rows,:][:,idx].toarray().astype(np.float64)
 del mat;gc.collect()
print('marker subset extracted',small.shape,'peakGB',resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1e6,flush=True)
rowmap={r:i for i,r in enumerate(selected_rows)}
den=total[idx]
qc=(den>0)&(t.nFeature_RNA.values>0)
gene_expr={}
for g,indices in gene_indices.items():
 if len(indices):
  counts=small[[rowmap[int(i)] for i in indices],:].sum(axis=0)
  gene_expr[g]=np.log1p(np.divide(counts*10000,den,out=np.zeros_like(counts),where=den>0))
for entry in manifest:
 name=entry['program'];gs=entry['available_genes'].split(';') if entry['available_genes'] else []
 t['program_'+name]=np.mean([gene_expr[g] for g in gs],axis=0) if len(gs)>=3 else np.nan
 t.loc[~qc,'program_'+name]=np.nan
t['rna_total_matrix']=den;t['positive_expression_qc']=qc
cols=['cell_name','patient_id','Ident','Tissue','treatment_stage','treatment_point','Cancer Type','is_crc','is_baseline','eligible_tissue','SubCellType','is_cd8','single_beta','strict_paired','beta_clone','paired_clone','rna_total_matrix','positive_expression_qc']+['program_'+x for x in PROGRAMS]
t[cols].to_parquet(OUT/'rna_marker_cell_scores.parquet',index=False)
x=t[t.patient_id.isin(pairs)&t.is_baseline&t.eligible_tissue&t.single_beta&t.is_cd8&t.positive_expression_qc].copy()
x['within_cd8_clone_cells']=x.groupby(['patient_id','Tissue','beta_clone']).cell_name.transform('size')
x['expansion_group']=np.where(x.within_cd8_clone_cells>=2,'expanded','singleton')
x[cols+['within_cd8_clone_cells','expansion_group']].to_parquet(OUT/'cd8_program_cell_inventory.parquet',index=False)
agg=[]
for pid in pairs:
 for tissue in ['Tumor','Normal']:
  for name in PROGRAMS:
   pp=x[x.patient_id.eq(pid)&x.Tissue.eq(tissue)]
   for group in ['expanded','singleton']:
    s=pp.loc[pp.expansion_group.eq(group),'program_'+name].dropna()
    agg.append(dict(patient_id=pid,tissue=tissue,program=name,expansion_group=group,cells=len(s),mean_score=s.mean() if len(s) else np.nan))
write(pd.DataFrame(agg),'patient_cd8_program_means.tsv')
meta=np.asarray(t.nCount_RNA,dtype=float)
receipt=dict(completed_utc=datetime.now(timezone.utc).isoformat(),matrix_rows=len(features),matrix_columns=len(bars),matrix_nnz=1310816895,all_rna_t_score_cells=len(t),conventional_cd8_accepted_beta_baseline_pair_cells=len(x),matrix_metadata_totalcount_exact_matches=int(np.sum(meta==den)),totalcount_compared_cells=len(t),max_absolute_totalcount_difference=float(np.max(np.abs(meta-den))),peak_memory_GB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1e6,threads_limit=4,matrix_source_sha256=sha(SOURCE/'GSE236581_counts.mtx.gz'),protocol_sha256=sha(OUT/'PROTOCOL.md'),source_gene_missing=manifest)
(OUT/'rna_receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
print('RNA complete',json.dumps({k:v for k,v in receipt.items() if k!='source_gene_missing'}),flush=True)
