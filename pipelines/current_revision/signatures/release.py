import os
from pathlib import Path
import numpy as np,pandas as pd,json,hashlib,re
from scipy import stats
from datetime import datetime,timezone
OUT=Path(os.environ['CRC_SIGNATURE_ROOT']).expanduser().resolve()
SRC=Path(os.environ['CRC_SIGNATURE_SOURCE']).expanduser().resolve()
if (OUT/'freeze.json').exists(): raise RuntimeError('Use a fresh output root; never overwrite an existing freeze')
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for z in iter(lambda:f.read(8*1024*1024),b''):h.update(z)
 return h.hexdigest()
protocol=json.loads((OUT/'protocol_frozen.json').read_text());states=protocol['states']
m=pd.read_csv(OUT/'inputs/groups.tsv',sep='\t');z=np.load(OUT/'inputs/pseudobulk_19patients_5groups.npz');genes=z['genes'].astype(str);counts=z['counts']
raw=np.fromfile(OUT/'inputs/raw_counts.bin',dtype='<i8').reshape(counts.shape)
det=np.fromfile(OUT/'inputs/raw_detected.bin',dtype='<i4').reshape(counts.shape)
np.testing.assert_array_equal(raw,counts)
assert np.all(det>=0) and np.all(det<=m.n_cells.to_numpy()[:,None]);assert np.array_equal((raw>0),(det>0))
assert len(np.unique(np.fromfile(OUT/'inputs/gene_map.bin',dtype=np.int32)))==len(genes)
stream=json.loads((OUT/'inputs/raw_stream.json').read_text());assert stream['raw_nnz']==1310816895 and stream['selected_total_counts']==int(counts.sum())
(OUT/'audit/raw_detection_verification.json').write_text(json.dumps({'raw_stream':stream,'full_gene_pseudobulk_counts_equal_preexisting_source':True,'detection_counts_bounded_by_group_cell_counts':True,'detection_positive_equivalent_count_positive':True,'unique_source_symbols_verified':True,'all_raw_coordinate_triplets_read':True},indent=2)+'\n')
elig=pd.read_csv(OUT/'tables/patient_eligibility.tsv',sep='\t')
all_markers=[];selected_all=[];detrows=[];summary=[]
for state in states:
 tab=pd.read_csv(OUT/'de'/f'{state}.tsv',sep='\t');np.testing.assert_allclose(stats.false_discovery_control(tab.PValue),tab.FDR,rtol=1e-9,atol=1e-12)
 donorids=elig.loc[elig.state.eq(state)&elig.included,'patient_id'];pts=[];prs=[]
 for donor in donorids:
  ids=m.index[m.patient_id.eq(donor)];ti=ids[m.loc[ids,'group'].eq(state)];ri=ids[~m.loc[ids,'group'].eq(state)]
  pt=det[ti].sum(axis=0)/m.loc[ti,'n_cells'].sum();pr=det[ri].sum(axis=0)/m.loc[ri,'n_cells'].sum();pts.append(pt);prs.append(pr)
  # Per-gene donor detection tables retained after selection below.
 pt=np.stack(pts);pr=np.stack(prs)
 dt=pd.DataFrame({'gene':genes,'pct_target_donor_equal':pt.mean(axis=0),'pct_rest_donor_equal':pr.mean(axis=0),'pct_gap_donor_equal':(pt-pr).mean(axis=0),'detection_gap_positive_donor_fraction':((pt-pr)>0).mean(axis=0)})
 tab=tab.merge(dt,on='gene',validate='one_to_one')
 def excluded(g):
  if re.match(r'^TR[ABDG][CVJD]',g):return 'TCR'
  if re.match(r'^IG[HKL]',g):return 'immunoglobulin'
  if re.match(r'^MT-',g):return 'mitochondrial'
  if re.match(r'^RP[SL][0-9]+[A-Z]?[0-9]*(P[0-9]+)?$|^RPLP[0-9]+(P[0-9]+)?$|^RPSA(P[0-9]+)?$|^MRP[SL][0-9]+',g):return 'ribosomal'
  return ''
 tab['exclusion_reason']=tab.gene.map(excluded)
 tab['specificity_score']=tab.logFC*tab.pct_gap_donor_equal
 tab['selection_pass']=(tab.logFC>.5)&(tab.FDR<.05)&(tab.donor_direction_fraction>=.7)&(tab.pct_target_donor_equal>=.1)&(tab.pct_gap_donor_equal>=.05)&tab.exclusion_reason.eq('')
 sel=tab[tab.selection_pass].sort_values(['specificity_score','logFC','FDR','gene'],ascending=[False,False,True,True]).head(30).copy()
 assert len(sel)>=10,f'{state}: insufficient signature genes {len(sel)}'
 sel['rank']=np.arange(1,len(sel)+1);sel['signature_id']='CRC19_'+state+'_TOP30';sel['analysis_tier']='secondary' if state.startswith('c08_') else 'primary';sel['contrast']='state_vs_all_other_RNA_T';sel['source_scope']='GSE236581_CRC_baseline_Tumor_19_patients';sel['log2FC']=sel.logFC;sel['q_value']=sel.FDR
 tab['selected_rank']=tab.gene.map(sel.set_index('gene')['rank']);tab.to_csv(OUT/'tables'/f'{state}_all_markers.tsv',sep='\t',index=False)
 selected_all.append(sel);all_markers.append(tab)
 gi=pd.Index(genes).get_indexer(sel.gene)
 for i,donor in enumerate(donorids):
  for j,k in enumerate(gi):detrows.append({'state':state,'patient_id':donor,'gene':genes[k],'pct_target':pt[i,k],'pct_rest':pr[i,k],'pct_gap':pt[i,k]-pr[i,k]})
 summary.append({'state':state,'signature_id':sel.signature_id.iloc[0],'analysis_tier':sel.analysis_tier.iloc[0],'paired_donors':len(donorids),'genes_tested':len(tab),'genes_passing_filters':int(tab.selection_pass.sum()),'genes_frozen':len(sel),'minimum_selected_log2FC':float(sel.logFC.min()),'maximum_selected_q':float(sel.FDR.max()),'minimum_selected_donor_support':float(sel.donor_direction_fraction.min()),'target_cells_scope19':int(m.loc[m.group.eq(state),'n_cells'].sum()),'target_cells_eligible_donors':int(elig.loc[elig.state.eq(state)&elig.included,'target_n_cells'].sum())})
selected=pd.concat(selected_all,ignore_index=True)
selected.to_csv(OUT/'frozen_signature_markers.tsv',sep='\t',index=False)
pd.DataFrame(summary).to_csv(OUT/'signature_manifest.tsv',sep='\t',index=False)
pd.DataFrame(detrows).to_csv(OUT/'tables/frozen_gene_patient_detection.tsv',sep='\t',index=False)
with (OUT/'frozen_signatures.gmt').open('w') as f:
 for state,x in selected.groupby('state',sort=False):f.write('\t'.join([x.signature_id.iloc[0],'GSE236581|baseline_CRC_tumor_19|'+x.analysis_tier.iloc[0]+'|state_vs_otherT']+x.gene.tolist())+'\n')
overlap=[]
for a in states:
 for b in states:
  aa=set(selected.loc[selected.state.eq(a),'gene']);bb=set(selected.loc[selected.state.eq(b),'gene']);overlap.append({'state_a':a,'state_b':b,'overlap_genes':len(aa&bb),'jaccard':len(aa&bb)/len(aa|bb),'genes':';'.join(sorted(aa&bb))})
pd.DataFrame(overlap).to_csv(OUT/'tables/signature_overlap.tsv',sep='\t',index=False)
freeze={'frozen_utc':datetime.now(timezone.utc).isoformat(),'status':'RECONSTRUCTED_FROM_ARCHIVED_PROTOCOL','no_TCGA_expression_or_outcomes_accessed':'not attested by rerun','protocol_sha256':sha(OUT/'protocol_frozen.json'),'files':{p.name:sha(p) for p in [OUT/'frozen_signatures.gmt',OUT/'frozen_signature_markers.tsv',OUT/'signature_manifest.tsv']},'n_signatures':4,'n_genes_total':len(selected),'state_vs_otherT_not_c27_vs_c26':True,'selection_rule_unchanged':True}
(OUT/'freeze.json').write_text(json.dumps(freeze,indent=2)+'\n')
print(json.dumps(freeze,indent=2),flush=True)
# Specificity on the scRNA training source only; no external outcome access.
meta=pd.read_csv(SRC/'pseudobulk_metadata.tsv',sep='\t');pb=np.memmap(SRC/'raw_counts.bin',dtype='<i8',mode='r',shape=(len(meta),len(genes)))
scorerows=[];exprrows=[];donors=sorted(set(m.patient_id))
for donor in donors:
 dm=meta[meta.patient_id.eq(donor)]
 groups={s:dm.SubCellType.eq(s)&dm.MajorCellType.eq('T') for s in sorted(meta.loc[meta.MajorCellType.eq('T'),'SubCellType'].unique())}
 groups.update({'nonT_'+x:dm.MajorCellType.eq(x) for x in sorted(meta.loc[~meta.MajorCellType.eq('T'),'MajorCellType'].unique())})
 groups.update({'other_CD4':dm.SubCellType.str.contains('_CD4_')&~dm.SubCellType.isin(states),'other_CD8':dm.SubCellType.str.contains('_CD8_')&~dm.SubCellType.isin(states+['c26_CD8_MAIT_KLRB1','c28_CD8_IEL_CD160']),'other_T_lineages':dm.MajorCellType.eq('T')&~dm.SubCellType.str.contains('_CD[48]_')})
 for group,mask in groups.items():
  ids=dm.index[mask];nc=int(meta.loc[ids,'n_cells'].sum())
  if nc<10:continue
  vec=np.asarray(pb[ids].sum(axis=0));cpm=vec/vec.sum()*1e6
  rank=(stats.rankdata(vec,method='average')-1)/(len(genes)-1)-.5
  for state,x in selected.groupby('state',sort=False):
   gi=pd.Index(genes).get_indexer(x.gene);scorerows.append({'patient_id':donor,'group':group,'n_cells':nc,'state_signature':state,'mean_gene_percentile_score':float(rank[gi].mean())})
  for state,x in selected.groupby('state',sort=False):
   for gene in x.nsmallest(6,'rank').gene:
    k=pd.Index(genes).get_loc(gene);exprrows.append({'patient_id':donor,'group':group,'n_cells':nc,'signature_state':state,'gene':gene,'log2_CPM_plus1':float(np.log2(cpm[k]+1))})
pd.DataFrame(scorerows).to_csv(OUT/'tables/scRNA_signature_specificity_donor_scores.tsv',sep='\t',index=False)
expr=pd.DataFrame(exprrows);expr.to_csv(OUT/'tables/top6_marker_donor_expression.tsv',sep='\t',index=False)
score=pd.DataFrame(scorerows)
ss=score.groupby(['state_signature','group']).agg(mean_score=('mean_gene_percentile_score','mean'),n_donors=('patient_id','nunique'),sd_score=('mean_gene_percentile_score','std')).reset_index()
ss.to_csv(OUT/'tables/scRNA_signature_specificity_summary.tsv',sep='\t',index=False)
checks=[]
for state in states:
 target=ss[(ss.state_signature==state)&(ss.group==state)].iloc[0]
 off=ss[(ss.state_signature==state)&ss.group.str.startswith('nonT_')].sort_values('mean_score',ascending=False).iloc[0]
 checks.append({'state':state,'target_mean_score':target.mean_score,'highest_nonT_group':off.group,'highest_nonT_score':off.mean_score,'target_minus_highest_nonT':target.mean_score-off.mean_score,'independent_validation':False,'c26_specificity_estimable':False if state.startswith('c27') else 'not_targeted'})
pd.DataFrame(checks).to_csv(OUT/'tables/nonT_specificity_checks.tsv',sep='\t',index=False)
print('SPECIFICITY_COMPLETE',pd.DataFrame(checks).to_string(index=False),flush=True)
