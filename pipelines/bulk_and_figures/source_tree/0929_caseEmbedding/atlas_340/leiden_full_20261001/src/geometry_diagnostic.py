#!/usr/bin/env python3
import os
os.environ['OMP_NUM_THREADS']='8'
os.environ['OPENBLAS_NUM_THREADS']='8'
os.environ['MKL_NUM_THREADS']='8'
import pathlib,hashlib,json,time,datetime,numpy as np,pyarrow.parquet as pq,faiss,psutil
b=pathlib.Path(__file__).resolve().parents[1]; r=b.parent
start=time.monotonic(); faiss.omp_set_num_threads(8)
import pyarrow as pa
pa.set_cpu_count(8);pa.set_io_thread_count(4)
print('geometry_diagnostic_started_sorted_bank_IO',flush=True)
draw_path=r/'atlas/joint/pca_patient_balanced_draws.parquet'
draws=pq.read_table(draw_path,columns=['seq_idx'])['seq_idx'].to_numpy()
ids=np.random.default_rng(20261001).choice(np.unique(draws),size=50000,replace=False)
ids_path=b/'pilot/geometry50k_seq_idx.npy'; raw_path=b/'pilot/geometry50k_raw768.npy'
cached=ids_path.exists() and raw_path.exists() and np.array_equal(np.load(ids_path),ids)
if cached:
 raw=np.load(raw_path)
 assert raw.shape==(50000,768) and raw.dtype==np.float32 and np.isfinite(raw).all()
 print('reuse_verified_same_ID_raw768_cache',flush=True)
else:
 np.save(ids_path,ids)
 mapping=pq.read_table(r/'data/atlas_aa_index.parquet',columns=['bank_source','bank_idx']).take(ids)
 raw=np.empty((len(ids),768),np.float32)
 for name,p in [('old',r.parent/'embeddings/tcrbert_dualtask_aa_mean/bank_fp16.npy'),('new',r/'embeddings/tcrbert_dualtask_aa_mean/bank_new_fp16.npy')]:
  mask=np.array(mapping['bank_source'].to_pylist())==name
  print('bank_gather',name,int(mask.sum()),flush=True)
  idx=mapping['bank_idx'].to_numpy()[mask]; bank=np.load(p,mmap_mode='r')
  order=np.argsort(idx); raw[np.flatnonzero(mask)[order]]=bank[idx[order]].astype(np.float32)
 assert set(mapping['bank_source'].to_pylist()) <= {'old','new'}
 assert np.isfinite(raw).all()
 np.save(raw_path,raw)
 del mapping
del draws
def sequential_subset(p):
 print('sequential_PCA_read',str(p),flush=True)
 whole=np.load(p)
 assert whole.shape==(28380380,64) and whole.dtype==np.float16
 result=whole[ids].astype(np.float32); del whole
 assert np.isfinite(result).all()
 return result
scores=sequential_subset(r/'atlas/joint/tcrbert_dualtask_aa_mean/pca_scores_fp16.npy')
sceptr=sequential_subset(r/'atlas/joint/sceptr_cdr3_only/pca_scores_fp16.npy')
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
def source_sha(p):
 h=hashlib.sha256()
 with pathlib.Path(p).open('rb') as f:
  for block in iter(lambda:f.read(16*1024**2),b''):h.update(block)
 return h.hexdigest()
manifest_path=r/'embeddings/bank_manifest.json'; verification_path=r/'embeddings/verification.json'
manifest=json.loads(manifest_path.read_text()); verified=json.loads(verification_path.read_text())
assert verified['status']=='PASS' and verified['n_atlas']==28380380
source_inputs={}
def register(p,expected=None,basis='fresh_complete_file_read_sha256'):
 actual=source_sha(p) if basis=='fresh_complete_file_read_sha256' else expected
 if expected is not None:assert actual==expected,(str(p),actual,expected)
 source_inputs[str(p)]={'sha256':actual,'size':p.stat().st_size,'mtime_ns':p.stat().st_mtime_ns,'hash_basis':basis}
register(draw_path)
register(r/'data/atlas_aa_index.parquet',manifest['index_sha256'])
for model in ['tcrbert_dualtask_aa_mean','sceptr_cdr3_only']:
 proj=json.loads((r/'atlas/joint'/model/'projection_receipt.json').read_text())
 register(r/'atlas/joint'/model/'pca_scores_fp16.npy',proj['scores_sha256'])
for kind in ['old','new']:
 model=manifest['model_banks']['tcrbert_dualtask_aa_mean']; p=pathlib.Path(model[kind+'_bank_path'])
 if kind=='old':
  assert p.stat().st_size==model['old_bank_size_bytes'] and p.stat().st_mtime_ns==model['old_bank_mtime_ns']
 register(p,verified['models']['tcrbert_dualtask_aa_mean'][kind+'_bank_sha256'],basis='prior_frozen_full_verification_receipt; same_bank_path_and_current_stat')
receipt.update(source_inputs=source_inputs,source_code_sha256=source_sha(__file__),source_manifest_sha256=source_sha(manifest_path),verification_receipt_sha256=source_sha(verification_path),raw_sample_sha256=source_sha(raw_path),raw_sample_reused=cached,geometry_change='I/O only; identical sample IDs, Euclidean geometry and k')
(b/'pilot/geometry50k_receipt.json').write_text(json.dumps(receipt,indent=2))
print(json.dumps(receipt),flush=True)
