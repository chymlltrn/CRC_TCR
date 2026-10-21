#!/usr/bin/env python3
"""Fresh complete Q1/Q2 reference rankings; no binding probability inference."""
from release_runtime import private_literal as _private_literal, release_path as _release_path
import os
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMBA_NUM_THREADS']: os.environ[k]='4'
from pathlib import Path
import re,json,hashlib,time,warnings,importlib.metadata,subprocess,sys
import numpy as np,pandas as pd
from tcrdist.repertoire import TCRrep
from tcrdist.rep_funcs import _pw
O=Path(sys.argv[1]);O.mkdir(parents=True,exist_ok=True)
R=Path(_release_path('path_d64669660f1883f0'))
Q=Path(_release_path('path_65b2261e498be9b9'))
B=Path(_release_path('path_2f6a196b03809909'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def trim(s):return s[1:-1] if s.startswith('C') and s.endswith(('F','W')) else s
def tokens(s):
 found=set(re.findall(r'TRBV[0-9]+(?:-[0-9]+)?(?:\*[0-9A-Za-z]+)?',str(s).upper()))
 for p in re.split('[/,;|+]',str(s).upper()):
  if re.fullmatch('[0-9]+(?:-[0-9]+)?(?:\*[0-9A-Za-z]+)?',p.strip()):found.add('TRBV'+p.strip())
 return sorted(found)
def target_ranks(d,score,ascending):
 # Peptide sequence is target identity; retain best receptor annotation and ties.
 valid=d[d.epitope.ne('')].copy()
 valid=valid.sort_values([score,'epitope','database','reference_key'],ascending=[ascending,True,True,True],kind='stable')
 n=valid.groupby('epitope').size();best=valid.drop_duplicates('epitope').copy()
 opt=valid.groupby('epitope')[score].min() if ascending else valid.groupby('epitope')[score].max()
 tied=valid[valid[score].eq(valid.epitope.map(opt))].groupby('epitope').size()
 best['n_target_reference_records']=best.epitope.map(n);best['n_best_score_ties']=best.epitope.map(tied)
 best['rank']=np.arange(1,len(best)+1);best['dense_score_rank']=best[score].rank(method='dense',ascending=ascending).astype(int)
 return best

def main():
 t=time.time();refs=pd.read_csv(R,sep='\t',dtype=str).fillna('');refs=refs[refs.reference_scope.eq('specificity_annotation')].copy();q=pd.read_csv(Q,sep='\t').fillna('');q.to_csv(O/'candidate_observed_branches.tsv',sep='\t',index=False)
 receipt={'status':'RUNNING','script_sha256':sha(__file__),'reference_path':str(R),'reference_sha256':sha(R),'query_path':str(Q),'query_sha256':sha(Q),'reference_records':len(refs),'reference_AA':refs.ref_cdr3beta.nunique(),'reference_peptides_including_empty':refs.epitope.nunique(),'reference_database_counts':refs.groupby('database').size().to_dict(),'queries':len(q),'score_is_calibrated_probability':False,'commands':[]}
 refs['trimmed_AA']=refs.ref_cdr3beta.map(trim)
 # Safe native kernel domain: canonical AA and trimmed sequence <=30, matching hardcoded array.
 good=refs.trimmed_AA.str.fullmatch('[ACDEFGHIKLMNPQRSTVWY]{1,30}')
 rmatch=refs[good].copy();seqs=sorted(rmatch.trimmed_AA.unique());(O/'TCRMatch_database_sequences.txt').write_text('\n'.join(seqs)+'\n')
 (O/'candidate_AA.txt').write_text(_private_literal('private_77ce437f1ad1b80a'))
 cmd=[str(B/'tcrmatch'),'-i',str(O/'candidate_AA.txt'),'-t','4','-s','0','-d',str(O/'TCRMatch_database_sequences.txt')];receipt['commands'].append(cmd)
 with open(O/'TCRMatch_all_scores.raw.tsv','w') as f,open(O/'TCRMatch_stderr.txt','w') as err:subprocess.run(cmd,stdout=f,stderr=err,check=True,cwd=B)
 scores=pd.read_csv(O/'TCRMatch_all_scores.raw.tsv',sep='\t',dtype={'match_sequence':str})
 assert len(scores)==2*len(seqs),(len(scores),len(seqs));assert scores.groupby('trimmed_input_sequence').match_sequence.nunique().eq(len(seqs)).all()
 rmatch.to_csv(O/'TCRMatch_reference_scoring_records.tsv.gz',sep='\t',index=False)
 tm=[]
 for aa,candidate in [(_private_literal('private_45bbaa37a9607bb7'),'Q1'),(_private_literal('private_b3445b7fc78bf2f0'),'Q2')]:
  ss=scores[scores.trimmed_input_sequence.eq(trim(aa))];mapping=ss.set_index('match_sequence').score
  hit=rmatch.copy();hit['similarity']=hit.trimmed_AA.map(mapping);assert hit.similarity.notna().all();hit.insert(0,'candidate',candidate);hit['query_AA']=aa;hit['query_scope']='CDR3beta AA; V/J not used';hit['complete_AA_exact']=hit.ref_cdr3beta.eq(aa)
  ranked=target_ranks(hit,'similarity',False);ranked.to_csv(O/f'{candidate}_TCRMatch_all_target_ranks.tsv.gz',sep='\t',index=False);tm.append(ranked.head(10))
 pd.concat(tm).to_csv(O/'TCRMatch_top10_targets.tsv',sep='\t',index=False)
 receipt['TCRMatch']={'binary_path':str(B/'tcrmatch'),'binary_sha256':sha(B/'tcrmatch'),'version':'1.3','src_main_sha256':sha(B/'src/main.cpp'),'src_kernel_sha256':sha(B/'src/tcrmatch.cpp'),'reference_scored_records':len(rmatch),'reference_excluded_records':int((~good).sum()),'unique_trimmed_AA':len(seqs),'scoring_pairs':len(scores),'trim_rule':'remove one leading C and terminal F/W only when both present','reference_target_peptides':rmatch[rmatch.epitope.ne('')].epitope.nunique(),'custom_database_sha256':sha(O/'TCRMatch_database_sequences.txt'),'threshold_for_return':0,'interpretation':'normalized kernel similarity, larger closer; no V/J/alpha/MHC, not calibrated confidence'}
 print('TCRMATCH_COMPLETE',receipt['TCRMatch'],flush=True)
 with warnings.catch_warnings():
  warnings.simplefilter('ignore');tr=TCRrep(cell_df=pd.DataFrame({'v_b_gene':['TRBV5-5*01'],'j_b_gene':['TRBJ1-4*01'],'cdr3_b_aa':[_private_literal('private_45bbaa37a9607bb7')],'count':[1]}),organism='human',chains=['beta'],compute_distances=False,deduplicate=False,cpus=4)
 assert tr.weights_b=={'cdr3_b_aa':3,'pmhc_b_aa':1,'cdr2_b_aa':1,'cdr1_b_aa':1}
 genes={k:g for k,g in tr.all_genes['human'].items() if g.chain=='B' and g.region=='V'};names=sorted(genes);ix={x:i for i,x in enumerate(names)};stem={}
 for x in names:stem.setdefault(x.split('*')[0],[]).append(x)
 def mapping(v):
  out=[]
  for x in tokens(v):out.extend([x] if '*' in x and x in genes else stem.get(x.split('*')[0],[]))
  return sorted(set(out))
 cache={x:mapping(x) for x in refs.ref_v.unique()};refs['V_options']=refs.ref_v.map(lambda x:';'.join(cache[x]));unscore=refs[refs.V_options.eq('')];refs=refs[refs.V_options.ne('')].reset_index(drop=True)
 unscore.groupby(['database','ref_v']).size().rename('n_records').reset_index().to_csv(O/'tcrdist_unscorable.tsv',sep='\t',index=False)
 vm=np.zeros((len(names),len(names)),dtype=np.int32)
 for field,pos in [('cdr1_b_aa',0),('cdr2_b_aa',1),('pmhc_b_aa',2)]:vm+=_pw(tr.metrics_b[field],seqs1=np.array([genes[x].cdrs_aligned[pos] for x in names]),ncpus=4,**tr.kargs_b[field])
 assert np.all(vm==vm.T) and np.trace(vm)==0
 refvs=refs.ref_v.unique().tolist();vg=refs.ref_v.map({v:i for i,v in enumerate(refvs)}).to_numpy();raa=np.array(sorted(refs.ref_cdr3beta.unique()));ag=refs.ref_cdr3beta.map({a:i for i,a in enumerate(raa)}).to_numpy()
 top=[];summary=[];smoke=[];combined=[]
 for aa,qq in q.groupby('AA'):
  d3=3*_pw(tr.metrics_b['cdr3_b_aa'],seqs1=np.array([aa]),seqs2=raa,ncpus=4,**tr.kargs_b['cdr3_b_aa'])[0].astype(np.int32)
  for row in qq.itertuples(index=False):
   opts=mapping(row.V)
   if not opts:raise ValueError(row)
   dv=[];cq=[];cr=[]
   for rv in refvs:
    ro=cache[rv];d=vm[np.ix_([ix[v] for v in opts],[ix[v] for v in ro])];loc=np.unravel_index(np.argmin(d),d.shape);dv.append(d[loc]);cq.append(opts[loc[0]]);cr.append(ro[loc[1]])
   total=d3[ag]+np.array(dv)[vg];hit=refs.copy();hit.insert(0,'candidate',row.query);hit['query_AA']=aa;hit['query_V']=row.V;hit['query_J']=row.J;hit['distance_beta']=total;hit['query_minimizing_allele']=np.array(cq)[vg];hit['reference_minimizing_allele']=np.array(cr)[vg]
   hit['complete_AA_exact']=hit.ref_cdr3beta.eq(aa);hit['V_gene_compatible']=hit.ref_v.map(lambda v:row.V in [x.split('*')[0] for x in tokens(v)]);hit['J_gene_compatible']=hit.ref_j.map(lambda j:row.J in re.findall(r'TRBJ\d+-\d+',j));hit['query_scope']='observed AA/V/J branch; allele minimum, J is annotation only'
   ranked=target_ranks(hit,'distance_beta',True);ranked.to_csv(O/f'{row.query}_{row.V}_{row.J}_tcrdist_all_target_ranks.tsv.gz',sep='\t',index=False);top.append(ranked.head(10));combined.append(ranked)
   summary.append(dict(candidate=row.query,AA=aa,V=row.V,J=row.J,minimum_distance=int(total.min()),n_radius24_annotations=int((total<=24).sum()),n_target_peptides=len(ranked)))
   z=ranked.iloc[0];smoke.append((aa,z.query_minimizing_allele,z.ref_cdr3beta,z.reference_minimizing_allele,int(z.distance_beta)))
  pooled=target_ranks(pd.concat([x for x in combined if x.candidate.iloc[0]==qq.iloc[0]['query']],ignore_index=True),'distance_beta',True);pooled.to_csv(O/f'{qq.iloc[0]["query"]}_tcrdist_all_observed_branch_min_target_ranks.tsv.gz',sep='\t',index=False)
 pd.concat(top).to_csv(O/'tcrdist_top10_targets_each_branch.tsv',sep='\t',index=False);pd.DataFrame(summary).to_csv(O/'tcrdist_branch_summary.tsv',sep='\t',index=False)
 checks=[]
 for aa,qv,ra,rv,expected in smoke:
  with warnings.catch_warnings():
   warnings.simplefilter('ignore');a=TCRrep(cell_df=pd.DataFrame({'v_b_gene':[qv],'j_b_gene':['TRBJ1-4*01'],'cdr3_b_aa':[aa],'count':[1]}),organism='human',chains=['beta'],compute_distances=False,deduplicate=False,cpus=1);b=TCRrep(cell_df=pd.DataFrame({'v_b_gene':[rv],'j_b_gene':['TRBJ1-4*01'],'cdr3_b_aa':[ra],'count':[1]}),organism='human',chains=['beta'],compute_distances=False,deduplicate=False,cpus=1)
  a.compute_rect_distances(df=a.clone_df,df2=b.clone_df,store=False);got=int(a.rw_beta[0,0]);assert got==expected;checks.append({'AA':aa,'query_allele':qv,'refAA':ra,'ref_allele':rv,'expected':expected,'public_API':got})
 pd.DataFrame(checks).to_csv(O/'tcrdist_public_API_checks.tsv',sep='\t',index=False)
 receipt['tcrdist']={'scorable_reference_records':len(refs),'unscorable_reference_records':len(unscore),'target_peptides':refs[refs.epitope.ne('')].epitope.nunique(),'versions':{x:importlib.metadata.version(x) for x in ['tcrdist3','pwseqdist','numpy','pandas']},'weights':tr.weights_b,'J_distance_in_metric':False,'allele_policy':'minimum over possible mapped V alleles, optimistic lower bound','public_API_checks':len(checks),'interpretation':'full beta loop distance, smaller closer; no binding probability'}
 receipt['status']='PASS';receipt['runtime_seconds']=time.time()-t;receipt['outputs']=[{'path':p.name,'size_bytes':p.stat().st_size,'sha256':sha(p)} for p in sorted(O.iterdir()) if p.is_file() and p.name!='fresh_reference_rankings_receipt.json'];(O/'fresh_reference_rankings_receipt.json').write_text(json.dumps(receipt,indent=2));print('COMPLETE',json.dumps({k:v for k,v in receipt.items() if k!='outputs'}),flush=True)
if __name__=='__main__':main()
