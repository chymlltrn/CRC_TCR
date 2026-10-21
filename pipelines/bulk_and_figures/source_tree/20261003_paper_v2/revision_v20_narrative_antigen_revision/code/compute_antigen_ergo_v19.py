#!/usr/bin/env python3
"""Official ERGO-II weights, safe inference, cached missing-alpha forward.
No training, dependency installation, arbitrary pickle or claims of calibration.
"""
from release_runtime import private_literal as _private_literal, release_path as _release_path
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='4'
from pathlib import Path
import sys,io,pickle,pickletools,json,hashlib,ast,time,argparse
import numpy as np,pandas as pd,torch
from torch import nn
import torch.nn.functional as F
O=Path(sys.argv[1]);B=Path(sys.argv[2]);O.mkdir(parents=True,exist_ok=True);torch.set_num_threads(4)
R=Path(_release_path('path_d64669660f1883f0'))
Q=Path(_release_path('path_65b2261e498be9b9'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
class PrimitiveReader(pickle.Unpickler):
 def find_class(self,module,name):raise pickle.UnpicklingError('All globals denied: '+module+'.'+name)
def train_vocab(p):
 raw=p.read_bytes();assert not any(op.name in ['GLOBAL','STACK_GLOBAL','REDUCE','NEWOBJ','NEWOBJ_EX'] for op,arg,pos in pickletools.genops(raw))
 samples=PrimitiveReader(io.BytesIO(raw)).load();assert isinstance(samples,list) and all(isinstance(z,dict) for z in samples)
 out=[]
 for key in ['va','vb','ja','jb','mhc']:
  vals=sorted({z[key] for z in samples if not pd.isna(z[key])});d={v:i for i,v in enumerate(vals,1)};d['UNK']=0;out.append(d)
 return out,samples
class PaddingAutoencoder(nn.Module):
 def __init__(self,l,d,e):
  super().__init__();self.encoder=nn.Sequential(nn.Linear(l*d,300),nn.ELU(),nn.Dropout(.1),nn.Linear(300,100),nn.ELU(),nn.Dropout(.1),nn.Linear(100,e));self.decoder=nn.Sequential(nn.Linear(e,100),nn.ELU(),nn.Dropout(.1),nn.Linear(100,300),nn.ELU(),nn.Dropout(.1),nn.Linear(300,l*d))
class AE(nn.Module):
 def __init__(self,l,e):super().__init__();self.autoencoder=PaddingAutoencoder(l,21,e)
 def forward(self,x):return self.autoencoder.encoder(x.reshape(len(x),-1))
class LSTM(nn.Module):
 def __init__(self,d,l,drop):super().__init__();self.embedding=nn.Embedding(21,d,padding_idx=0);self.lstm=nn.LSTM(d,l,num_layers=2,batch_first=True,dropout=drop)
 def forward(self,seq,lengths):
  lens,idx=lengths.sort(descending=True);x=self.embedding(seq)[idx];packed=nn.utils.rnn.pack_padded_sequence(x,lens,batch_first=True);result,_=self.lstm(packed);x,lens=nn.utils.rnn.pad_packed_sequence(result,batch_first=True);_,rev=idx.sort();x=x[rev];lens=lens[rev];return torch.stack([x[i,int(l)-1] for i,l in enumerate(lens)])
class Model(nn.Module):
 def __init__(self,h,ds):
  super().__init__();self.tcrb_encoder=AE(28,h.encoding_dim);self.tcra_encoder=AE(34,h.encoding_dim);self.pep_encoder=LSTM(h.aa_embedding_dim,h.lstm_dim,h.dropout);self.v_embedding=nn.Embedding(len(ds[0])+len(ds[1]),h.cat_embedding_dim,padding_idx=0);self.j_embedding=nn.Embedding(len(ds[2])+len(ds[3]),h.cat_embedding_dim,padding_idx=0);self.mhc_embedding=nn.Embedding(len(ds[4]),h.cat_embedding_dim,padding_idx=0);self.hidden_layer1=nn.Linear(751,int(np.sqrt(751)));self.output_layer1=nn.Linear(int(np.sqrt(751)),1);self.hidden_layer2=nn.Linear(951,int(np.sqrt(951)));self.output_layer2=nn.Linear(int(np.sqrt(951)),1);self.relu=nn.LeakyReLU();self.dropout=nn.Dropout(h.dropout);self.use_vj=True;self.use_mhc=True;self.use_t_type=True;self.cat_encoding='embedding'
 def cached(self,tcr,pep,vb,jb):
  n=len(pep);cat=torch.cat([tcr.expand(n,-1),pep,self.v_embedding(torch.tensor([vb])).expand(n,-1),self.j_embedding(torch.tensor([jb])).expand(n,-1),self.mhc_embedding(torch.tensor([0])).expand(n,-1),torch.zeros(n,1)],1);return torch.sigmoid(self.output_layer1(self.relu(self.hidden_layer1(cat)))).ravel()
AA=_private_literal('private_ff4aabaf338f256e');ix={a:i+1 for i,a in enumerate(AA)}
def encoding_tcr(a):
 x=torch.zeros(1,28,21)
 for j,v in enumerate(a):x[0,j,ix[v]-1]=1
 x[0,len(a),20]=1
 return x
def encoding_pep(ps):
 lens=torch.tensor([len(p) for p in ps]);x=torch.zeros(len(ps),int(lens.max()),dtype=torch.long)
 for i,p in enumerate(ps):x[i,:len(p)]=torch.tensor([ix[a] for a in p])
 return x,lens

def main():
 started=time.time();refs=pd.read_csv(R,sep='\t',dtype=str).fillna('');refs=refs[refs.reference_scope.eq('specificity_annotation')].copy();valid=refs.epitope.str.fullmatch('[ACDEFGHIKLMNPQRSTVWY]+');peptides=sorted(refs[valid].epitope.unique());universe=refs[valid].copy();universe.sort_values(['epitope','database','reference_key']).drop_duplicates('epitope').to_csv(O/'scored_peptide_universe.tsv',sep='\t',index=False)
 q=pd.read_csv(Q,sep='\t').rename(columns={'query':'candidate','AA':'query_AA','V':'query_V','J':'query_J'})
 q['query_scope']='observed AA/V/J branch; alpha/MHC/type unknown'
 aaonly=pd.DataFrame([{'candidate':'Q1','query_AA':_private_literal('private_45bbaa37a9607bb7'),'query_V':'UNK','query_J':'UNK','query_scope':'AA only; V/J/alpha/MHC/type unknown'},{'candidate':'Q2','query_AA':_private_literal('private_b3445b7fc78bf2f0'),'query_V':'UNK','query_J':'UNK','query_scope':'AA only; V/J/alpha/MHC/type unknown'}]);q=pd.concat([aaonly,q],ignore_index=True)
 # Execute only the already hash-verified official forward AST; no module imports.
 tree=ast.parse((B/'Trainer.py').read_text());cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='ERGOLightning');fn=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='forward');ns={'torch':torch};exec(compile(ast.Module(body=[fn],type_ignores=[]),'official_Trainer_forward','exec'),ns);official_forward=ns['forward']
 allrows=[];tests=[];receipts=[]
 for dataset,version in [('vdjdb','1veajht'),('mcpas','1meajht')]:
  meta=B/'Models'/('version_'+version)/'meta_tags.csv';args={}
  for z in pd.read_csv(meta).itertuples(index=False):args[z.key]=z.value if z.key in ['dataset','tcr_encoding_model','cat_encoding'] else ast.literal_eval(z.value)
  h=argparse.Namespace(**args);ds,train=train_vocab(B/'Samples'/f'{dataset}_train_samples.pickle');ckpt=next((B/'Models'/('version_'+version)/'checkpoints').glob('*.ckpt'))
  with torch.serialization.safe_globals([argparse.Namespace]):state=torch.load(ckpt,map_location='cpu',weights_only=True)
  model=Model(h,ds);model.load_state_dict(state['state_dict'],strict=True);model.eval();enc=[]
  with torch.inference_mode():
   for k in range(0,len(peptides),128):enc.append(model.pep_encoder(*encoding_pep(peptides[k:k+128])))
   penc=torch.cat(enc);tenc={aa:model.tcrb_encoder(encoding_tcr(aa)) for aa in q.query_AA.unique()}
   for z in q.itertuples(index=False):
    vb=ds[1].get(z.query_V,0);jb=ds[3].get(z.query_J,0);scores=model.cached(tenc[z.query_AA],penc,vb,jb).numpy();assert np.isfinite(scores).all() and (scores>=0).all() and (scores<=1).all()
    out=pd.DataFrame({'candidate':z.candidate,'query_AA':z.query_AA,'query_V':z.query_V,'query_J':z.query_J,'query_scope':z.query_scope,'model':f'ERGO-II_{dataset}','epitope':peptides,'model_score':scores,'encoded_V_index':vb,'encoded_J_index':jb,'encoded_HLA_index':0,'alpha_present':False,'t_cell_type_index':0})
    out=out.sort_values(['model_score','epitope'],ascending=[False,True],kind='stable');out['rank']=np.arange(1,len(out)+1);out.to_csv(O/f'{z.candidate}_{z.query_V}_{z.query_J}_ERGO_{dataset}_all_peptide_scores.tsv.gz',sep='\t',index=False)
    annotated=out.head(10).merge(universe.sort_values(['database','reference_key']).drop_duplicates('epitope'),on='epitope',how='left',validate='one_to_one');annotated['reference_receptor_role']='catalog annotation of target peptide; not sequence transferred by ERGO';annotated['n_catalog_reference_records']=annotated.epitope.map(universe.groupby('epitope').size());allrows.append(annotated)
    # Independent exact official-forward comparison for 3 diverse peptide lengths.
    pp=[peptides[0],peptides[len(peptides)//2],peptides[-1]];xb,xl=encoding_pep(pp);n=len(pp);official=official_forward(model,(None,(encoding_tcr(z.query_AA).expand(n,-1,-1),)),(xb,xl),(torch.zeros(n,dtype=torch.long),torch.full((n,),vb),torch.zeros(n,dtype=torch.long),torch.full((n,),jb),torch.zeros(n,dtype=torch.long)),torch.zeros(n)).ravel().numpy();own=model.cached(tenc[z.query_AA],model.pep_encoder(xb,xl),vb,jb).numpy();err=float(np.max(np.abs(own-official)));assert err<1e-6;tests.append({'model':dataset,'query_AA':z.query_AA,'query_V':z.query_V,'query_J':z.query_J,'maximum_absolute_score_difference':err,'n_pairs':n})
  # Training overlap, a descriptive audit rather than independence assertion.
  exact=[z for z in train if z.get('tcrb') in q.query_AA.unique()];pd.DataFrame(exact).to_csv(O/f'candidate_training_overlap_{dataset}.tsv',sep='\t',index=False)
  receipts.append({'dataset':dataset,'version':version,'checkpoint_path':str(ckpt),'checkpoint_sha256':sha(ckpt),'meta_sha256':sha(meta),'vocabulary_path':str(B/'Samples'/f'{dataset}_train_samples.pickle'),'vocabulary_sha256':sha(B/'Samples'/f'{dataset}_train_samples.pickle'),'weights_only':True,'allowlisted_checkpoint_global':'argparse.Namespace only','training_pickle_globals':'all denied; primitive-only','state_dict_strict_load':True,'n_training_exact_candidate_rows':len(exact),'peptides':len(peptides),'query_scopes':len(q),'forward_validation_comparisons':len(q)*3,'vocab_sizes':[len(x) for x in ds]})
 pd.concat(allrows).to_csv(O/'ERGO_top10_targets_each_scope.tsv',sep='\t',index=False);pd.DataFrame(tests).to_csv(O/'official_forward_parity_checks.tsv',sep='\t',index=False);q.to_csv(O/'ERGO_query_scopes.tsv',sep='\t',index=False)
 r={'status':'PASS','script_sha256':sha(__file__),'reference_path':str(R),'reference_sha256':sha(R),'branch_path':str(Q),'branch_sha256':sha(Q),'official_repo':'https://github.com/IdoSpringer/ERGO-II','official_commit':'85d320ab03ade33460be9a81ea3a51b8d37cd998','official_Trainer_sha256':sha(B/'Trainer.py'),'scorable_canonical_peptide_records':len(universe),'excluded_noncanonical_or_empty_peptide_records':int((~valid).sum()),'unique_scored_peptides':len(peptides),'models':receipts,'torch_version':torch.__version__,'cpu_threads':4,'runtime_seconds':time.time()-started,'score_meaning':'sigmoid model output, uncalibrated ranking; not assay confidence or patient binding probability','missing_alpha_route':'official MLP I','HLA_patient_unknown':'encoded0 UNK in all inputs; reference HLA not imposed','training':False,'installations':False,'outputs':[{'path':p.name,'size_bytes':p.stat().st_size,'sha256':sha(p)} for p in sorted(O.iterdir()) if p.is_file()]};(O/'ERGO_fresh_inference_receipt.json').write_text(json.dumps(r,indent=2));print('ERGO_COMPLETE',json.dumps({k:v for k,v in r.items() if k!='outputs'}),flush=True)
if __name__=='__main__':main()
