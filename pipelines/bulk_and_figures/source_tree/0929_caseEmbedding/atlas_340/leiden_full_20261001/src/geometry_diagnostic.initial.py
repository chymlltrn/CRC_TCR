#!/usr/bin/env python3
import os
os.environ['OMP_NUM_THREADS']='8'
os.environ['OPENBLAS_NUM_THREADS']='8'
os.environ['MKL_NUM_THREADS']='8'
import pathlib,hashlib,json,time,datetime,numpy as np,pyarrow.parquet as pq,faiss,psutil
b=pathlib.Path(__file__).resolve().parents[1]; r=b.parent
start=time.monotonic(); faiss.omp_set_num_threads(8)
draw_path=r/'atlas/joint/pca_patient_balanced_draws.parquet'
draws=pq.read_table(draw_path,columns=['seq_idx'])['seq_idx'].to_numpy()
ids=np.random.default_rng(20261001).choice(np.unique(draws),size=50000,replace=False)
np.save(b/'pilot/geometry50k_seq_idx.npy',ids)
mapping=pq.read_table(r/'data/atlas_aa_index.parquet',columns=['bank_source','bank_idx']).take(ids)
raw=np.empty((len(ids),768),np.float32)
for name,p in [('old',r.parent/'embeddings/tcrbert_dualtask_aa_mean/bank_fp16.npy'),('new',r/'embeddings/tcrbert_dualtask_aa_mean/bank_new_fp16.npy')]:
 mask=np.array(mapping['bank_source'].to_pylist())==name
 idx=mapping['bank_idx'].to_numpy()[mask]; bank=np.load(p,mmap_mode='r')
 raw[mask]=bank[idx].astype(np.float32)
del mapping,draws
scores=np.load(r/'atlas/joint/tcrbert_dualtask_aa_mean/pca_scores_fp16.npy',mmap_mode='r')[ids].astype(np.float32)
sceptr=np.load(r/'atlas/joint/sceptr_cdr3_only/pca_scores_fp16.npy',mmap_mode='r')[ids].astype(np.float32)
query=np.random.default_rng(17).choice(len(ids),512,replace=False)
def neighbors(x):
 idx=faiss.IndexFlatL2(x.shape[1]);idx.add(x)
 _,ii=idx.search(x[query],51)
 return np.array([row[row!=own][:50] for row,own in zip(ii,query)])
nn_raw=neighbors(raw);nn_pca=neighbors(scores);nn_scep=neighbors(sceptr)
metrics={}
for k in [15,30,50]:
 a=np.array([len(set(u[:k])&set(v[:k]))/k for u,v in zip(nn_raw,nn_pca)])
 c=np.array([len(set(u[:k])&set(v[:k]))/k for u,v in zip(nn_pca,nn_scep)])
 metrics[str(k)]={'raw768_to_PCA64_neighbor_overlap_mean':float(a.mean()),'overlap_median':float(np.median(a)),'overlap_q05':float(np.quantile(a,.05)),'TCRBERT64_SCEPTR64_neighbor_overlap_mean':float(c.mean())}
receipt={'status':'PASS','scope':'geometry diagnostic on fixed patient-balanced-draw 50000 unique AA; NOT full-bank ANN recall or full Leiden completion','n_pool':50000,'n_queries':512,'no_N_labels_accessed':True,'seed':20261001,'threads':8,'metrics':metrics,'seconds':time.monotonic()-start,'rss_bytes':psutil.Process().memory_info().rss,'draw_path':str(draw_path),'indices_sha256':hashlib.sha256((b/'pilot/geometry50k_seq_idx.npy').read_bytes()).hexdigest(),'interpretation':'PCA64 is the declared graph geometry; projection neighbor overlap is a separate limitation and never guarantees antigen specificity'}
(b/'pilot/geometry50k_receipt.json').write_text(json.dumps(receipt,indent=2))
print(json.dumps(receipt),flush=True)
