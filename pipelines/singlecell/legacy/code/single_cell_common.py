from release_config import release_path, private_value
import json,hashlib,re,os
from pathlib import Path
from collections import Counter
import numpy as np
import pandas as pd
from scipy import stats

ROOT=Path(os.environ.get('CRC_TCR_PAPER_ROOT',str(release_path('single_cell_common_source_8')))).expanduser()
OUT=ROOT/'single_cell'
SOURCE=Path(os.environ.get('GSE236581_SOURCE_ROOT',str(release_path('single_cell_common_source_10')))).expanduser()
SEED=20261003
Q1AA=private_value('single_cell_common_value_12_5')
Q1NT=private_value('single_cell_common_value_13_5')
PROGRAMS={
'cytotoxic':['NKG7','GZMB','PRF1','GNLY','CTSW','GZMH'],
'exhaustion':['PDCD1','LAG3','HAVCR2','TIGIT','TOX','CXCL13'],
'naive_memory':['TCF7','LEF1','CCR7','SELL','IL7R','LTB'],
'interferon':['STAT1','ISG15','IFIT1','IFIT3','MX1','IRF7'],
'residency':['ITGAE','ITGA1','CXCR6','ZNF683','CD69'],
'proliferation':['MKI67','TOP2A','STMN1','TYMS','PCNA']}
AASET=set('ACDEFGHIKLMNPQRSTVWY')
BASES='TCAG'
CODONS={a+b+c:x for (a,b,c),x in zip(((a,b,c) for a in BASES for b in BASES for c in BASES),'FFLLSSSSYY**CC*WLLLLPPPPHHQQRRRRIIIMTTTTNNKKSSRRVVVVAAAADDEEGGGG')}
def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for x in iter(lambda:f.read(8*1024*1024),b''):h.update(x)
 return h.hexdigest()
def write(df,name):df.to_csv(OUT/name,sep='\t',index=False)
def clean(x):
 return '' if pd.isna(x) else str(x).strip().upper()
def gene(x):
 s=clean(x).split('*')[0]
 return '' if ',' in s or ';' in s or ' ' in s else s
def translated(s):
 return ''.join(CODONS.get(s[i:i+3],'?') for i in range(0,len(s),3))
def accepted(aa,nt,v,j,full,umi,chain):
 aa,nt,v,j=clean(aa),clean(nt),gene(v),gene(j)
 try:n=float(umi)
 except (ValueError,TypeError):return None
 if not (aa.startswith('C') and set(aa)<=AASET and nt and set(nt)<=set('ACGT') and len(nt)%3==0 and translated(nt)==aa):return None
 if str(full).strip().lower()!='true' or not(np.isfinite(n) and n>0):return None
 if not(v.startswith('TR'+chain+'V') and j.startswith('TR'+chain+'J')):return None
 return dict(aa=aa,nt=nt,v=v,j=j,nt_vj=nt+'|'+v+'|'+j,aa_vj=aa+'|'+v+'|'+j)
def metrics(counts):
 a=np.asarray(counts,dtype=np.int64);a=a[a>0]
 n=int(a.sum());k=len(a)
 if n==0:return dict(tcr_cells=0,unique_clones=0,normalized_shannon=np.nan,top10_fraction=np.nan,expanded_fraction=np.nan)
 f=a/n;h=-float(np.sum(f*np.log(f)))
 return dict(tcr_cells=n,unique_clones=k,normalized_shannon=h/np.log(k) if k>1 else 0.0,top10_fraction=float(np.sort(a)[-10:].sum()/n),expanded_fraction=float(a[a>=2].sum()/n))
def bh(p):
 x=np.asarray(p,float);out=np.full(len(x),np.nan);ok=np.flatnonzero(np.isfinite(x))
 if len(ok):
  order=ok[np.argsort(x[ok])];q=np.minimum.accumulate((x[order]*len(x)/np.arange(1,len(ok)+1))[::-1])[::-1]
  out[order]=np.minimum(q,1)
 return out
def paired_test(values,seed=SEED):
 x=np.asarray(values,float);x=x[np.isfinite(x)];n=len(x)
 if n<3:return dict(n=n,mean_difference=float(np.mean(x)) if n else np.nan,median_difference=float(np.median(x)) if n else np.nan,ci_low=np.nan,ci_high=np.nan,p_value=np.nan,test_status='insufficient_n')
 rng=np.random.default_rng(seed)
 boot=np.mean(x[rng.integers(0,n,size=(10000,n))],axis=1)
 p=1.0 if np.all(x==0) else float(stats.wilcoxon(x,zero_method='wilcox',correction=False,alternative='two-sided',method='approx').pvalue)
 lo,hi=np.quantile(boot,[.025,.975])
 return dict(n=n,mean_difference=float(x.mean()),median_difference=float(np.median(x)),ci_low=float(lo),ci_high=float(hi),p_value=p,test_status='estimable' if np.any(x!=0) else 'all_zero')
def cd8(s):
 m=re.match(r'c(\d+)_CD8_',str(s))
 return bool(m and (15<=int(m[1])<=25 or int(m[1])==28))
