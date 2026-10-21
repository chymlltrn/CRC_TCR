#!/usr/bin/env python3
"""Full EOF, source-bound blood-to-single-cell beta matching. No inferential tests."""
from release_config import release_path, private_value
from pathlib import Path
from datetime import datetime,timezone
from collections import Counter
import argparse,csv,hashlib,json,re,time,sys
import numpy as np,pandas as pd,pyarrow.parquet as pq
from Bio.Data import CodonTable
V2=Path(str(release_path('bulk_sc_match_ingest_source_9')))
V1=Path(str(release_path('bulk_sc_match_ingest_source_10')))
BULK=Path(str(release_path('bulk_sc_match_ingest_source_11')))
DEFS=('AA_VJ','NT_VJ','AA');GROUPS=('home','external','union')
PROGRAMS=('cytotoxic','exhaustion','naive_memory','interferon','residency','proliferation')
CODON_LUT=np.full(125,ord('X'),dtype=np.uint8);CODON_LUT[124]=0
BASE_LUT=np.full(256,4,dtype=np.uint8)
for i,b in enumerate('ACGT'):BASE_LUT[ord(b)]=i
for c,aa in CodonTable.unambiguous_dna_by_id[1].forward_table.items():
 x=BASE_LUT[np.frombuffer(c.encode(),dtype=np.uint8)];CODON_LUT[int(x[0])*25+int(x[1])*5+int(x[2])]=ord(aa)
for c in CodonTable.unambiguous_dna_by_id[1].stop_codons:
 x=BASE_LUT[np.frombuffer(c.encode(),dtype=np.uint8)];CODON_LUT[int(x[0])*25+int(x[1])*5+int(x[2])]=ord('*')
def now():return datetime.now(timezone.utc).isoformat()
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
def jsonwrite(p,x):Path(p).write_text(json.dumps(x,indent=2,default=str)+'\n')
def clean(x):return x.fillna('').astype(str).str.strip().str.upper()
def gene(x):return clean(x).str.replace(r'\*[A-Z0-9]+$','',regex=True)
def gene_valid(v,j):return v.str.fullmatch(r'TRBV[0-9]+(?:-[0-9]+)?(?:/OR[0-9]+-[0-9]+)?') & j.str.fullmatch(r'TRBJ[0-9]+-[0-9]+')
def nt_compatible(nt,aa):
 """Vectorized standard-code translation of every input row; no length truncation."""
 if not len(nt):return np.zeros(0,dtype=bool)
 lens=nt.str.len().to_numpy();la=aa.str.len().to_numpy();valid=nt.str.fullmatch('[ACGT]+').to_numpy() & (lens%3==0)&(lens==3*la)
 width=int(max(lens.max(initial=0),3));width=((width+2)//3)*3
 n=len(nt);raw=np.asarray(nt.to_numpy(),dtype=f'S{width}').view(np.uint8).reshape(n,width)
 enc=BASE_LUT[raw].reshape(n,width//3,3);code=enc[:,:,0]*25+enc[:,:,1]*5+enc[:,:,2]
 translated=CODON_LUT[code]
 expected=np.asarray(aa.to_numpy(),dtype=f'S{width//3}').view(np.uint8).reshape(n,width//3)
 return valid & np.all(translated==expected,axis=1)
def keys(aa,nt,v,j):return {'AA':aa,'AA_VJ':aa+'|'+v+'|'+j,'NT_VJ':nt+'|'+v+'|'+j}
def tied(hits):
 """Audit only: max-score ties across distinct allele-collapsed genes."""
 if not isinstance(hits,str) or ',' not in hits:return False
 items=re.findall(r'([^,()]+)\(([-+0-9.eE]+)\)',hits)
 try:
  values=[(re.sub(r'\*[A-Za-z0-9]+$','',g.strip().upper()),float(s)) for g,s in items]
  if not values:return False
  maximum=max(s for g,s in values);return len({g for g,s in values if s==maximum})>1
 except ValueError:return False
def load_roster():
 lib=pd.read_csv(V1/'cohort/library_master.tsv',sep='\t');lib=lib[lib.primary_repertoire_library.eq(1)].sort_values(['cohort','sample_id']).copy()
 pat=pd.read_csv(V1/'cohort/patient_master.tsv',sep='\t');assert len(lib)==len(pat)==339 and lib.patient_key.is_unique and pat.patient_key.is_unique
 assert lib.groupby('cohort').size().to_dict()=={'external':202,'home':137}
 assert set(lib.patient_key)==set(pat.patient_key)
 assert private_value('bulk_sc_match_ingest_value_56_8') in set(lib.sample_id) and private_value('bulk_sc_match_ingest_value_56_46') not in set(lib.sample_id)
 for r in lib.itertuples():assert (BULK/f'{r.cohort}__{r.sample_id}.parquet').exists()
 return lib,pat

def sc_index(root,out,data):
 base=V2/'single_cell';audit=pd.read_parquet(base/'vdj_cell_chain_audit.parquet');assert audit.cell_name.is_unique
 x=audit[audit.single_beta].copy();assert x.accepted_distinct_beta.eq(1).all()
 rna=pd.read_parquet(base/'rna_cell_inventory.parquet');assert rna.cell_name.is_unique
 meta=[c for c in rna if not c.startswith('beta_') and c not in ['raw_beta_slots','raw_alpha_slots','accepted_distinct_beta','accepted_distinct_alpha','single_beta','strict_paired','paired_clone']]
 x=x.merge(rna[meta],on='cell_name',how='left',validate='one_to_one',indicator='_rna_merge');x['RNA_joined']=x.pop('_rna_merge').eq('both')
 x['source_sample_prefix']=x.cell_name.str.rsplit('_',n=1).str[0]
 ctx=pd.read_csv(V2/'literature/GSE236581_sample_context.tsv',sep='\t');assert ctx.RNA_Ident_expected.is_unique
 c=ctx.set_index('RNA_Ident_expected')
 for target,source in [('verified_source_patient','Patient ID'),('verified_source_sample','Sample ID'),('verified_biopsy_site','Biopsy Site'),('verified_treatment_stage','Treatment Stage'),('verified_treatment_point','Treatment point'),('verified_cancer_type','Cancer_Type')]:x[target]=x.source_sample_prefix.map(c[source])
 x['clinical_source_mapped']=x.verified_source_sample.notna()
 x['cell_resolution_tag']=np.select([x.RNA_joined & x.MajorCellType.eq('T'),x.RNA_joined,x.clinical_source_mapped],['RNA_joined_T','RNA_joined_nonT','VDJ_only_verified_sample'],default='VDJ_only_unmapped_source')
 pairs=pd.read_csv(base/'paired_patient_inventory.tsv',sep='\t');pairids=set(pairs.loc[pairs.primary_pair_status.eq('included'),'patient_id']);assert len(pairids)==19
 x['is_primary_pair']=x.patient_id.isin(pairids)
 for z in ['is_crc','is_baseline','eligible_tissue','is_cd8','strict_paired']:x[z]=x[z].fillna(False).astype(bool)
 aa=clean(x.beta_aa);nt=clean(x.beta_nt);v=gene(x.beta_v);j=gene(x.beta_j);gv=gene_valid(v,j);nc=nt_compatible(nt,aa)
 x['beta_aa']=aa;x['beta_nt']=nt;x['beta_v']=v;x['beta_j']=j;x['sc_gene_syntax_valid']=gv;x['sc_nt_translation_valid']=nc
 kk=keys(aa,nt,v,j)
 for d in DEFS:x['key_'+d]=kk[d] if d=='AA' else kk[d].where(gv & (nc if d=='NT_VJ' else True),None)
 x['beta_clone']=x.key_NT_VJ
 x['is_primary_cell']=x.is_primary_pair & x.RNA_joined & x.MajorCellType.eq('T') & x.is_crc & x.is_baseline & x.eligible_tissue & x.Tissue.isin(['Tumor','Normal']) & gv & nc
 scores=pd.read_parquet(base/'rna_marker_cell_scores.parquet');assert scores.cell_name.is_unique
 pc=['cell_name','positive_expression_qc']+['program_'+p for p in PROGRAMS]
 x=x.merge(scores[pc],on='cell_name',how='left',validate='one_to_one');x.positive_expression_qc=x.positive_expression_qc.fillna(False).astype(bool)
 assert x.loc[~x.RNA_joined,[f'program_{p}' for p in PROGRAMS]].isna().all().all()
 assert x.cell_name.is_unique
 assert len(x)==597963,'frozen all-source accepted single-beta denominator'
 assert int((x.RNA_joined & x.MajorCellType.eq('T')).sum())==340805,'frozen RNA-T single-beta denominator'
 assert int(x.is_primary_cell.sum())==53886,'frozen baseline19 primary denominator'
 x.to_parquet(data/'sc_cells_index.parquet',index=False)
 den=x.groupby(['cell_resolution_tag'],dropna=False).size().reset_index(name='accepted_single_beta_cells');den.to_csv(out/'sc_scope_denominators.tsv',sep='\t',index=False)
 x[x.is_primary_cell].groupby(['patient_id','Tissue']).size().reset_index(name='accepted_single_beta_cells').to_csv(out/'baseline19_cell_denominators.tsv',sep='\t',index=False)
 assert x[x.is_primary_cell].groupby(['patient_id','Tissue']).size().ge(50).all() and x[x.is_primary_cell].groupby(['patient_id','Tissue']).ngroups==38
 return x

def index_sets(x):
 allsets={d:set(x['key_'+d].dropna()) for d in DEFS};tsets={};nsets={}
 for d in DEFS:
  tsets[d]=set(x.loc[x.is_primary_cell & x.Tissue.eq('Tumor'),'key_'+d].dropna());nsets[d]=set(x.loc[x.is_primary_cell & x.Tissue.eq('Normal'),'key_'+d].dropna())
 return allsets,tsets,nsets

def scan_library(r,allsets,tsets,nsets,out,data,batchsize):
 src=BULK/f'{r.cohort}__{r.sample_id}.parquet';pf=pq.ParquetFile(src);st=src.stat();start=time.monotonic();aggregates={d:Counter() for d in DEFS};rowc={d:Counter() for d in DEFS};tiec={d:Counter() for d in DEFS};unique={d:set() for d in DEFS}
 cnt=Counter();source_patient=set();tok=Counter();cols=['cohort','sample_id','patient_key','clone_id','aa','nt_cdr3','read_count','qc_pass','v_gene','j_gene','all_v_hits_with_score','all_j_hits_with_score']
 for b in pf.iter_batches(batch_size=batchsize,columns=cols):
  z=b.to_pandas();cnt['raw_clone_rows']+=len(z);assert z.cohort.eq(r.cohort).all() and z.sample_id.eq(r.sample_id).all();source_patient.update(z.patient_key.unique());q=z[z.qc_pass].copy();cnt['qc_clone_rows']+=len(q)
  if q.empty:continue
  reads=q.read_count.to_numpy(dtype=np.int64);assert (reads>0).all();cnt['qc_reads']+=int(reads.sum());aa=clean(q.aa);nt=clean(q.nt_cdr3);v=gene(q.v_gene);j=gene(q.j_gene);gv=gene_valid(v,j).to_numpy();nc=nt_compatible(nt,aa);kk=keys(aa,nt,v,j)
  vt=q.all_v_hits_with_score.map(tied).to_numpy();jt=q.all_j_hits_with_score.map(tied).to_numpy();tie=vt|jt
  cnt['VJ_valid_clone_rows']+=int(gv.sum());cnt['VJ_valid_reads']+=int(reads[gv].sum());cnt['NT_translation_valid_clone_rows']+=int(nc.sum());cnt['NT_translation_valid_reads']+=int(reads[nc].sum());cnt['NT_VJ_eligible_clone_rows']+=int((nc&gv).sum());cnt['NT_VJ_eligible_reads']+=int(reads[nc&gv].sum());cnt['rank_cross_gene_tie_rows']+=int(tie.sum());cnt['rank_cross_gene_tie_reads']+=int(reads[tie].sum());cnt['missing_allhit_annotation_rows']+=int((q.all_v_hits_with_score.fillna('').eq('')|q.all_j_hits_with_score.fillna('').eq('')).sum())
  for label,s in [('V',v),('J',j)]:
   for token,n in s.value_counts(dropna=False).items():tok[(label,token)]+=int(n)
  for d in DEFS:
   eligible=np.ones(len(q),dtype=bool) if d=='AA' else gv if d=='AA_VJ' else gv&nc
   k=kk[d].to_numpy();unique[d].update(k[eligible]);m=eligible & kk[d].isin(allsets[d]).to_numpy()
   if m.any():
    a=pd.DataFrame({'match_key':k[m],'read_sum':reads[m],'bulk_clone_rows':1,'rank_tied_rows':tie[m].astype(np.int64)}).groupby('match_key',sort=False).sum()
    aggregates[d].update(a.read_sum.to_dict());rowc[d].update(a.bulk_clone_rows.to_dict());tiec[d].update(a.rank_tied_rows.to_dict())
  del z,q,b
 assert cnt['raw_clone_rows']==pf.metadata.num_rows;assert cnt['qc_reads']==int(r.library_qc_reads),(r.sample_id,cnt['qc_reads'],r.library_qc_reads)
 after=src.stat();assert (st.st_size,st.st_mtime_ns)==(after.st_size,after.st_mtime_ns)
 receipt={'cohort':r.cohort,'sample_id':r.sample_id,'patient_key':r.patient_key,'source_patient_keys':'|'.join(sorted(source_patient)),'source_path':str(src),'source_bytes':st.st_size,'source_sha256':sha(src),'parquet_metadata_rows':pf.metadata.num_rows,'scan_complete_to_EOF':True,'elapsed_seconds':round(time.monotonic()-start,3),**dict(cnt)}
 supports=[];coverage=[]
 for d in DEFS:
  for k,rs in aggregates[d].items():supports.append({'definition':d,'match_key':k,'patient_key':r.patient_key,'cohort':r.cohort,'sample_id':r.sample_id,'read_sum':int(rs),'bulk_clone_rows':int(rowc[d][k]),'rank_tied_rows':int(tiec[d][k])})
  er=cnt['qc_clone_rows'] if d=='AA' else cnt['VJ_valid_clone_rows'] if d=='AA_VJ' else cnt['NT_VJ_eligible_clone_rows'];ed=cnt['qc_reads'] if d=='AA' else cnt['VJ_valid_reads'] if d=='AA_VJ' else cnt['NT_VJ_eligible_reads']
  for threshold in (1,2):
   got={k for k,rs in aggregates[d].items() if rs>=threshold};t=got&tsets[d];n=got&nsets[d];base=t|n;parts={'full_source':got,'baseline19':base,'Tumor_only':t-n,'Normal_only':n-t,'both_tissues':t&n}
   c={'patient_key':r.patient_key,'cohort':r.cohort,'sample_id':r.sample_id,'definition':d,'read_threshold':threshold,'qc_clone_rows':cnt['qc_clone_rows'],'qc_reads':cnt['qc_reads'],'definition_eligible_clone_rows':er,'definition_eligible_reads':ed,'definition_unique_keys':len(unique[d])}
   for name,s in parts.items():
    c[name+'_matched_unique_keys']=len(s);c[name+'_matched_clone_rows']=sum(rowc[d][k] for k in s);c[name+'_matched_reads']=sum(aggregates[d][k] for k in s)
   c['no_baseline_matched_unique_keys']=len(unique[d])-len(base);c['no_baseline_clone_rows']=cnt['qc_clone_rows']-c['baseline19_matched_clone_rows'];c['no_baseline_reads']=cnt['qc_reads']-c['baseline19_matched_reads']
   for name in ('full_source','baseline19'):c[name+'_fraction_qc_clone_rows']=c[name+'_matched_clone_rows']/cnt['qc_clone_rows'] if cnt['qc_clone_rows'] else np.nan;c[name+'_fraction_qc_reads']=c[name+'_matched_reads']/cnt['qc_reads'] if cnt['qc_reads'] else np.nan
   assert sum(c[p+'_matched_reads'] for p in ('Tumor_only','Normal_only','both_tissues'))==c['baseline19_matched_reads']
   coverage.append(c)
 pd.DataFrame(supports,columns=['definition','match_key','patient_key','cohort','sample_id','read_sum','bulk_clone_rows','rank_tied_rows']).to_parquet(data/'library_support'/f'{r.cohort}__{r.sample_id}.parquet',index=False)
 pd.DataFrame(coverage).to_csv(out/'library_coverage'/f'{r.cohort}__{r.sample_id}.tsv',sep='\t',index=False)
 pd.DataFrame([{'cohort':r.cohort,'sample_id':r.sample_id,'gene_type':g,'canonical_token':t,'qc_clone_rows':n} for (g,t),n in tok.items()]).to_csv(out/'library_gene_tokens'/f'{r.cohort}__{r.sample_id}.tsv',sep='\t',index=False)
 jsonwrite(out/'library_receipts'/f'{r.cohort}__{r.sample_id}.json',receipt)
 return receipt

def summarize(x,lib,out,data,root):
 rec=[json.loads((out/'library_receipts'/f'{r.cohort}__{r.sample_id}.json').read_text()) for r in lib.itertuples()];rdf=pd.DataFrame(rec);assert len(rdf)==339 and rdf.scan_complete_to_EOF.all();rdf.to_csv(out/'library_eof_receipts.tsv',sep='\t',index=False)
 qc=['cohort','sample_id','patient_key','raw_clone_rows','qc_clone_rows','qc_reads','VJ_valid_clone_rows','VJ_valid_reads','NT_translation_valid_clone_rows','NT_translation_valid_reads','NT_VJ_eligible_clone_rows','NT_VJ_eligible_reads','rank_cross_gene_tie_rows','rank_cross_gene_tie_reads','missing_allhit_annotation_rows'];rdf[qc].to_csv(out/'QC_layers.tsv',sep='\t',index=False)
 cov=pd.concat([pd.read_csv(out/'library_coverage'/f'{r.cohort}__{r.sample_id}.tsv',sep='\t') for r in lib.itertuples()],ignore_index=True);assert len(cov)==2034;cov.to_csv(out/'patient_coverage.tsv',sep='\t',index=False)
 sup=pd.concat([pd.read_parquet(data/'library_support'/f'{r.cohort}__{r.sample_id}.parquet') for r in lib.itertuples()],ignore_index=True);assert not sup.duplicated(['definition','match_key','patient_key']).any();sup.to_parquet(data/'bulk_support.parquet',index=False)
 catalog=[]
 for d in DEFS:
  z=x[x['key_'+d].notna()];c=z.groupby('key_'+d).size().rename('full_sc_cells').to_frame();c.index.name='match_key'
  for label,m in [('rna_t',z.RNA_joined&z.MajorCellType.eq('T')),('baseline19_Tumor',z.is_primary_cell&z.Tissue.eq('Tumor')),('baseline19_Normal',z.is_primary_cell&z.Tissue.eq('Normal'))]:c[label+'_cells']=z[m].groupby('key_'+d).size().reindex(c.index,fill_value=0)
  sd=sup[sup.definition.eq(d)]
  for g in GROUPS:
   s=sd if g=='union' else sd[sd.cohort.eq(g)]
   for t in (1,2):c[g+'_patients_ge'+str(t)]=s[s.read_sum.ge(t)].groupby('match_key').size().reindex(c.index,fill_value=0)
   c[g+'_support_reads']=s.groupby('match_key').read_sum.sum().reindex(c.index,fill_value=0);c[g+'_denominator']={'home':137,'external':202,'union':339}[g]
  c=c.reset_index();c['definition']=d;catalog.append(c)
  for g in GROUPS:
   for t in (1,2):
    flag=c.set_index('match_key')[g+'_patients_ge'+str(t)].gt(0);x[f'match_{d}_{g}_ge{t}']=x['key_'+d].map(flag).fillna(False).astype(bool)
 cat=pd.concat(catalog,ignore_index=True);cat.to_parquet(data/'key_catalog.parquet',index=False)
 for g in GROUPS:
  for t in (1,2):assert (~x[f'match_NT_VJ_{g}_ge{t}']|x[f'match_AA_VJ_{g}_ge{t}']).all();assert (~x[f'match_AA_VJ_{g}_ge{t}']|x[f'match_AA_{g}_ge{t}']).all()
 x.to_parquet(data/'sc_cells.parquet',index=False)
 tokens=pd.concat([pd.read_csv(out/'library_gene_tokens'/f'{r.cohort}__{r.sample_id}.tsv',sep='\t',keep_default_na=False) for r in lib.itertuples()],ignore_index=True).groupby(['cohort','gene_type','canonical_token'],as_index=False).qc_clone_rows.sum();tokens.to_csv(out/'gene_token_audit.tsv',sep='\t',index=False)
 totals=rdf.groupby('cohort')[['raw_clone_rows','qc_clone_rows','qc_reads']].sum().to_dict(orient='index');assert totals=={'external':{'raw_clone_rows':16485349,'qc_clone_rows':15862848,'qc_reads':363688521},'home':{'raw_clone_rows':41389196,'qc_clone_rows':38834872,'qc_reads':941927575}},totals
 files=[p for p in data.glob('*.parquet')]+[p for p in out.glob('*.tsv') if p.name!='output_manifest.tsv']
 manifest=[{'path':str(p),'bytes':p.stat().st_size,'sha256':sha(p),'mirror_policy':'remote_only' if p.suffix=='.parquet' else 'small_file_local_mirror'} for p in files];pd.DataFrame(manifest).to_csv(out/'output_manifest.tsv',sep='\t',index=False)
 jsonwrite(out/'ingest_receipt.json',{'status':'passed','completed_utc':now(),'remote_only_computation':True,'protocol_sha256':sha(root/'bulk_sc_match/PROTOCOL.md'),'entrypoint_sha256':sha(__file__),'selected_libraries':339,'independent_blood_patients':339,'full_EOF':True,'totals':totals,'sc_accepted_single_beta_cells':len(x),'baseline19_primary_cells':int(x.is_primary_cell.sum()),'sc_cell_resolution_counts':x.cell_resolution_tag.value_counts().to_dict(),'NT_subset_AA_VJ_subset_AA_globally':True,'blood_threshold_unit':'read sum per selected blood patient per identity key; never across patients','first_ranked_gene_ties_retained_and_audited':True,'all_frozen_QC_NT_rows_translation_checked':True,'formal_statistical_tests':'none; downstream analysis module owns inference','outputs':manifest})
 print('FINAL INGEST PASS',json.dumps(totals),flush=True)


def binding_context(source,root,out,data):
 code_sha=sha(__file__)
 snap=out/'producing_code'/f'bulk_sc_match_ingest.{code_sha}.py';snap.parent.mkdir(exist_ok=True)
 if not snap.exists():snap.write_bytes(Path(__file__).read_bytes())
 assert sha(snap)==code_sha
 refs=[{'path':str(p),'bytes':p.stat().st_size,'sha256':sha(p)} for p in source if Path(p).resolve()!=Path(__file__).resolve()]
 ref_sha=hashlib.sha256(json.dumps(refs,sort_keys=True,separators=(',',':')).encode()).hexdigest()
 return {'protocol_sha256':sha(root/'bulk_sc_match/PROTOCOL.md'),'sc_index_sha256':sha(data/'sc_cells_index.parquet'),'selected_roster_sha256':sha(out/'selected_library_roster.tsv'),'reference_inputs_sha256':ref_sha,'current_entrypoint_sha256':code_sha}

def attach_binding(receipt,context,out,data,producing_sha):
 r=dict(receipt);cohort=r['cohort'];sample=r['sample_id']
 paths={'support':data/'library_support'/f'{cohort}__{sample}.parquet','coverage':out/'library_coverage'/f'{cohort}__{sample}.tsv','gene_tokens':out/'library_gene_tokens'/f'{cohort}__{sample}.tsv'}
 r['execution_binding']={k:v for k,v in context.items() if k!='current_entrypoint_sha256'}
 r['execution_binding']['producing_entrypoint_sha256']=producing_sha
 r['execution_binding']['derivative_sha256']={k:sha(p) for k,p in paths.items()}
 jsonwrite(out/'library_receipts'/f'{cohort}__{sample}.json',r)
 return r

def assert_resume_binding(old,context,root,out,data):
 b=old.get('execution_binding')
 if not b:raise RuntimeError('Unbound existing receipt: refusing resume. Preserve original provenance and certify binding first.')
 for k,v in context.items():
  if k!='current_entrypoint_sha256' and b.get(k)!=v:raise RuntimeError('Resume refused: changed '+k)
 producing=b['producing_entrypoint_sha256'];current=context['current_entrypoint_sha256']
 if producing!=current:
  cp=out/'guard_code_equivalence.json'
  if not cp.exists():raise RuntimeError('Resume refused: changed producing code without equivalence certificate')
  c=json.loads(cp.read_text())
  if c.get('status')!='passed' or c.get('producing_entrypoint_sha256')!=producing or c.get('guarded_entrypoint_sha256')!=current:raise RuntimeError('Resume refused: code certificate mismatch')
 cohort=old['cohort'];sample=old['sample_id']
 paths={'support':data/'library_support'/f'{cohort}__{sample}.parquet','coverage':out/'library_coverage'/f'{cohort}__{sample}.tsv','gene_tokens':out/'library_gene_tokens'/f'{cohort}__{sample}.tsv'}
 for k,p in paths.items():
  if not p.exists() or b['derivative_sha256'].get(k)!=sha(p):raise RuntimeError('Resume refused: changed/missing derivative '+k)

def stamp_provenance(root,out,data,context):
 receipts=[json.loads(p.read_text()) for p in (out/'library_receipts').glob('*.json')]
 producing=sorted({r['execution_binding']['producing_entrypoint_sha256'] for r in receipts})
 rp=out/'ingest_receipt.json';r=json.loads(rp.read_text())
 r.update({'producing_entrypoint_sha256':producing[0] if len(producing)==1 else None,'producing_entrypoint_sha256_values':producing,'current_entrypoint_sha256':context['current_entrypoint_sha256'],'sc_index_sha256':context['sc_index_sha256'],'reference_inputs_sha256':context['reference_inputs_sha256'],'producing_input_manifest_sha256':sha(out/'input_manifest.tsv'),'library_receipts_bound_to_code_protocol_index_and_derivatives':True})
 cp=out/'guard_code_equivalence.json'
 if cp.exists():r['guard_code_equivalence_sha256']=sha(cp)
 jsonwrite(rp,r)

def main():
 ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--root',type=Path,required=True);ap.add_argument('--stage',choices=['all','index','scan','summarize'],default='all');ap.add_argument('--resume',action='store_true');ap.add_argument('--batch-size',type=int,default=131072);a=ap.parse_args();root=a.root.resolve();out=root/'bulk_sc_match/ingest';data=root/'bulk_sc_match/server_data'
 for p in [out,data,out/'library_receipts',out/'library_coverage',out/'library_gene_tokens',data/'library_support']:p.mkdir(parents=True,exist_ok=True)
 lib,pat=load_roster();lib.to_csv(out/'selected_library_roster.tsv',sep='\t',index=False)
 source=[V1/'cohort/library_master.tsv',V1/'cohort/patient_master.tsv',V2/'single_cell/vdj_cell_chain_audit.parquet',V2/'single_cell/vdj_chain_inventory.parquet',V2/'single_cell/rna_cell_inventory.parquet',V2/'single_cell/rna_marker_cell_scores.parquet',V2/'single_cell/paired_patient_inventory.tsv',V2/'single_cell/PROTOCOL.md',V2/'code/single_cell_common.py',V2/'literature/GSE236581_sample_context.tsv',root/'bulk_sc_match/PROTOCOL.md',Path(__file__)]
 im=pd.DataFrame([{'path':str(p),'bytes':p.stat().st_size,'sha256':sha(p)} for p in source])
 im.to_csv(out/f'input_manifest.{sha(__file__)}.tsv',sep='\t',index=False)
 if not (out/'input_manifest.tsv').exists():im.to_csv(out/'input_manifest.tsv',sep='\t',index=False)
 if a.stage in ['all','index']:x=sc_index(root,out,data)
 else:x=pd.read_parquet(data/'sc_cells_index.parquet')
 jsonwrite(out/'preflight_receipt.json',{'status':'passed','completed_utc':now(),'selected339':True,'Home137_External202':True,'sc_cell_name_unique':bool(x.cell_name.is_unique),'accepted_single_beta_cells':len(x),'baseline19_pairs':19,'protocol_sha256':sha(root/'bulk_sc_match/PROTOCOL.md'),'no_inference':True})
 if a.stage=='index':return
 context=binding_context(source,root,out,data)
 if a.stage in ['all','scan']:
  sets,ts,ns=index_sets(x);print('INDEX READY',len(x),{d:len(s) for d,s in sets.items()},flush=True)
  for i,r in enumerate(lib.itertuples(),1):
   rp=out/'library_receipts'/f'{r.cohort}__{r.sample_id}.json'
   if a.resume and rp.exists():
    old=json.loads(rp.read_text());assert_resume_binding(old,context,root,out,data);src=BULK/f'{r.cohort}__{r.sample_id}.parquet'
    if old.get('scan_complete_to_EOF') and old['source_bytes']==src.stat().st_size and old['source_sha256']==sha(src):print('RESUME',i,r.sample_id,flush=True);continue
   rr=scan_library(r,sets,ts,ns,out,data,a.batch_size);rr=attach_binding(rr,context,out,data,context['current_entrypoint_sha256']);print('EOF',i,r.cohort,r.sample_id,rr['raw_clone_rows'],rr['qc_clone_rows'],rr['elapsed_seconds'],flush=True)
  if a.stage=='scan':return
 for r in lib.itertuples():assert_resume_binding(json.loads((out/'library_receipts'/f'{r.cohort}__{r.sample_id}.json').read_text()),context,root,out,data)
 summarize(x,lib,out,data,root)
 stamp_provenance(root,out,data,context)
if __name__=='__main__':main()
