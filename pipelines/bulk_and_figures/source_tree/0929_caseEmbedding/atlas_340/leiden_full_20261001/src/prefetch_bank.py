import os,pathlib,time,hashlib,json,datetime
os.sched_setaffinity(0,sorted(os.sched_getaffinity(0))[:1])
b=pathlib.Path(__file__).resolve().parents[1]
p=b.parent.parent/'embeddings/tcrbert_dualtask_aa_mean/bank_fp16.npy'
s=time.monotonic();h=hashlib.sha256();count=0
with p.open('rb') as f:
 while True:
  data=f.read(64*1024**2)
  if not data:break
  h.update(data);count+=len(data)
  if count%(1024**3)==0:print(json.dumps({'bytes_read':count,'seconds':time.monotonic()-s}),flush=True)
o={'status':'PASS' if h.hexdigest()=='dea3d54fb77cd4d8ffb8f5c630fd1c99bfae1cd6cf7fbe75f30a3edcfcc993ca' else 'FAIL','source_path':str(p),'source_sha256':h.hexdigest(),'read_only':True,'purpose':'cold original vector bank sequential prefetch plus fresh source verification','bytes_read':count,'seconds':time.monotonic()-s,'max_compute_threads':1,'created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()}
(b/'audit/oldbank_prefetch_receipt.json').write_text(json.dumps(o,indent=2));print(json.dumps(o),flush=True)
