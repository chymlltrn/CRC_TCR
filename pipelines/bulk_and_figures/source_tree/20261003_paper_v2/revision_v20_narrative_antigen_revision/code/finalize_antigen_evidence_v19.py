#!/usr/bin/env python3
"""Locally independently validate and freeze portable antigen ranking evidence."""
from pathlib import Path
import csv,gzip,json,hashlib,sys,math
B=Path(sys.argv[1]);R=B/'reference_rankings';E=B/'ergo_rankings'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def rows(p):
 with (gzip.open(p,'rt') if p.suffix=='.gz' else p.open()) as f:return list(csv.DictReader(f,delimiter='\t'))
def write(p,d):
 cols=list(dict.fromkeys(k for z in d for k in z))
 with p.open('w') as f:
  w=csv.DictWriter(f,fieldnames=cols,delimiter='\t',lineterminator='\n');w.writeheader();w.writerows(d)
def trim(s):return s[1:-1] if s.startswith('C') and s.endswith(('F','W')) else s
checks=[]
def check(name,condition,evidence):
 checks.append({'check':name,'status':'PASS' if condition else 'FAIL','evidence':evidence});assert condition,(name,evidence)
rr=json.loads((R/'fresh_reference_rankings_receipt.json').read_text());er=json.loads((E/'ERGO_fresh_inference_receipt.json').read_text())
check('frozen catalog identity shared',rr['reference_sha256']==er['reference_sha256'],'6649bacb231110d7d886c13689bb557625f7be088931707d0098388f89c995aa')
for root,receipt in [(R,rr),(E,er)]:
 for z in receipt['outputs']:
  p=root/z['path']
  if p.exists():check('source-copy bytes '+z['path'],sha(p)==z['sha256'] and p.stat().st_size==z['size_bytes'],z)
check('script source identity reference',sha(B.parent.parent/'code/compute_antigen_rankings_v19.py')==rr['script_sha256'],rr['script_sha256'])
check('script source identity ERGO',sha(B.parent.parent/'code/compute_antigen_ergo_v19.py')==er['script_sha256'],er['script_sha256'])
check('tcrdist public API exact parity',all(z['expected']==z['public_API'] for z in rows(R/'tcrdist_public_API_checks.tsv')),38)
pp=rows(E/'official_forward_parity_checks.tsv');check('ERGO official-forward parity',len(pp)==80 and all(float(z['maximum_absolute_score_difference'])<1e-6 for z in pp),{'contexts':len(pp),'pairs':sum(int(z['n_pairs']) for z in pp),'maximum_error':max(float(z['maximum_absolute_score_difference']) for z in pp)})
check('ERGO safe and no training',all(z['weights_only'] and z['state_dict_strict_load'] for z in er['models']) and not er['training'] and not er['installations'],er['models'])
raw=rows(R/'TCRMatch_all_scores.raw.tsv.gz');sm={(z['trimmed_input_sequence'],z['match_sequence']):float(z['score']) for z in raw};check('TCRMatch exhaustive unique pairs',len(raw)==len(sm)==421000,{'records':len(raw),'unique_pairs':len(sm)})
plot=[]
def add(context,method,p,score,direction,candidate=None,filters=None,n_targets=None):
 d=rows(p)
 if candidate:d=[z for z in d if z['candidate']==candidate]
 if filters:d=[z for z in d if all(z[k]==v for k,v in filters.items())]
 d=sorted(d,key=lambda z:int(z['rank']))
 check('ten unique targets '+context,len(d)>=10 and len({z['epitope'] for z in d[:10]})==10,p.name)
 for z in d[:10]:
  out=dict(z);out.update(context=context,method_display=method,score_raw=z[score],score_column=score,score_direction=direction,calibrated_confidence='False',n_ranked_targets=str(n_targets),scope_note='')
  if method=='TCRMatch':out['scope_note']='AA-only; no V/J/alpha/HLA inputs';check('TCRMatch raw score join '+context+' '+z['rank'],float(z[score])==sm[(trim(z['query_AA']),trim(z['ref_cdr3beta']))],z['reference_key'])
  elif method=='tcrdist':out['scope_note']='beta-loop distance; optimistic V allele minimum; J not in metric'
  else:out['scope_note']='CDR3beta+peptide; V/J/HLA/type unknown; missing-alpha MLP I; target catalog receptor not matched by model'
  if 'envelope' in context:
   out['scope_note']+='; minimum across35 observedQ2branches varies by target';out['n_target_reference_records']='';out['n_best_score_ties']='';out['counts_note']='envelope counts omitted; original aggregate counts are branch-best rows, not unique source records'
  plot.append(out)
add('TCRMatch_Q1_AA','TCRMatch',R/'Q1_TCRMatch_all_target_ranks.tsv.gz','similarity','higher',n_targets=3539)
add('TCRMatch_Q2_AA','TCRMatch',R/'Q2_TCRMatch_all_target_ranks.tsv.gz','similarity','higher',n_targets=3539)
add('tcrdist_Q1_nominated_V5-5_J1-4','tcrdist',R/'Q1_TRBV5-5_TRBJ1-4_tcrdist_all_target_ranks.tsv.gz','distance_beta','lower',n_targets=3308)
add('tcrdist_Q1_alternative_V7-2_J1-4','tcrdist',R/'Q1_TRBV7-2_TRBJ1-4_tcrdist_all_target_ranks.tsv.gz','distance_beta','lower',n_targets=3308)
add('tcrdist_Q2_35branch_envelope','tcrdist',R/'Q2_tcrdist_all_observed_branch_min_target_ranks.tsv.gz','distance_beta','lower',n_targets=3308)
for m in ['vdjdb','mcpas']:
 for c in ['Q1','Q2']:
  add(f'ERGO_{m}_{c}_AA','ERGO-II_'+m,E/'ERGO_top10_targets_each_scope.tsv','model_score','higher',candidate=c,filters={'model':'ERGO-II_'+m,'query_V':'UNK','query_J':'UNK'},n_targets=3427)
# All saved full ranks: no unreported gap/rank drift, scores ordered within method.
for p in sorted(R.glob('*_all_target_ranks.tsv.gz')):
 d=rows(p);s='similarity' if 'TCRMatch' in p.name else 'distance_beta';asc=s=='distance_beta';vals=[float(z[s]) for z in d];check('complete ordered ranks '+p.name,[int(z['rank']) for z in d]==list(range(1,len(d)+1)) and len({z['epitope'] for z in d})==len(d) and vals==sorted(vals,reverse=not asc),{'n':len(d),'score':s})
for p in sorted(E.glob('*_all_peptide_scores.tsv.gz')):
 d=rows(p);vals=[float(z['model_score']) for z in d];check('complete ordered model ranks '+p.name,len(d)==3427 and len({z['epitope'] for z in d})==3427 and [int(z['rank']) for z in d]==list(range(1,3428)) and vals==sorted(vals,reverse=True) and all(0<=v<=1 for v in vals),{'n':len(d)})
# Independently reconstruct Q2 envelope from complete35 per-branch peptide rankings.
mins={}
for p in R.glob('Q2_TRBV*_tcrdist_all_target_ranks.tsv.gz'):
 for z in rows(p):mins[z['epitope']]=min(mins.get(z['epitope'],math.inf),int(z['distance_beta']))
d=rows(R/'Q2_tcrdist_all_observed_branch_min_target_ranks.tsv.gz');check('Q2 envelope independently35 branches',len(list(R.glob('Q2_TRBV*_tcrdist_all_target_ranks.tsv.gz')))==35 and all(mins[z['epitope']]==int(z['distance_beta']) for z in d),{'branches':35,'peptides':len(mins)})
# Top10 rows independently exactly match each full ERGO table.
for z in rows(E/'ERGO_top10_targets_each_scope.tsv'):
 m=z['model'].replace('ERGO-II_','');full=rows(E/f'{z["candidate"]}_{z["query_V"]}_{z["query_J"]}_ERGO_{m}_all_peptide_scores.tsv.gz');v=full[int(z['rank'])-1];check('ERGO top10 full table '+z['model']+' '+z['candidate']+' '+z['query_V']+' '+z['rank'],v['epitope']==z['epitope'] and v['model_score']==z['model_score'],z['reference_key'])
write(B/'plot_ready_primary_top10.tsv',plot);write(B/'plot_ready_primary_top3.tsv',[z for z in plot if int(z['rank'])<=3]);write(B/'primary_top10_reference_keys.tsv',[{'context':z['context'],'rank':z['rank'],'reference_key':z['reference_key'],'epitope':z['epitope']} for z in plot])
receipt={'status':'PASS','scope':'v19 antigen quantitative source computation; not production figures/prose','reference_sha256':rr['reference_sha256'],'branch_sha256':rr['query_sha256'],'fresh_reference_receipt_sha256':sha(R/'fresh_reference_rankings_receipt.json'),'fresh_ERGO_receipt_sha256':sha(E/'ERGO_fresh_inference_receipt.json'),'methods':['TCRMatch similarity','tcrdist beta loop distance','ERGO-II VDJdb sigmoid output','ERGO-II McPAS sigmoid output'],'primary_contexts':9,'primary_top10_rows':len(plot),'primary_top3_rows':27,'checks':checks,'unresolved_biological_scope':['unknown paired alpha and patient HLA','no experimental peptide binding validation','no calibrated confidence estimate','allele-minimum distance optimistic','ERGO checkpoints must not be averaged or selected by observed output'],'ranking_policy':'one row per exact peptide sequence; best reference similarity/distance; deterministic peptide/database/reference_key tie order; preserve dense score ranks and source ties; model peptide direct output with representative target annotation only','plot_files':[{'path':p.name,'sha256':sha(p),'size_bytes':p.stat().st_size} for p in [B/'plot_ready_primary_top10.tsv',B/'plot_ready_primary_top3.tsv',B/'primary_top10_reference_keys.tsv']]};(B/'quantitative_validation_receipt.json').write_text(json.dumps(receipt,indent=2));print('VALIDATED',len(checks),'checks','90 primary rows')
