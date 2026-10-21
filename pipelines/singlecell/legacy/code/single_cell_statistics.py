#!/usr/bin/env python3
"""Patient-unit inference for frozen families; no cell-pooled tests."""
from release_config import release_path, private_value
from single_cell_common import *
import argparse
from datetime import datetime,timezone
arg=argparse.ArgumentParser();arg.add_argument('--part',choices=['tcr','rna','all'],default='all');args=arg.parse_args()
def family(df):
 df=df.copy();df['family_size']=len(df);df['q_bh']=bh(df.p_value.values)
 return df
def contrasts(frame,idcols,value,label,familyname):
 rows=[]
 for i,keys in enumerate(frame[idcols].drop_duplicates().itertuples(index=False,name=None)):
  mask=np.ones(len(frame),dtype=bool)
  for c,k in zip(idcols,keys):mask &= frame[c].eq(k).values
  d=frame.loc[mask].pivot(index='patient_id',columns='tissue',values=value)
  for tissue in ['Tumor','Normal']:
   if tissue not in d:d[tissue]=np.nan
  differences=d.Tumor-d.Normal
  rows.append(dict(**dict(zip(idcols,keys)),contrast=label,family=familyname,value=value,**paired_test(differences.values,SEED+i),included_patients=';'.join(d.index[differences.notna()].tolist())))
 return family(pd.DataFrame(rows))
if args.part in ['tcr','all']:
 p=pd.read_csv(OUT/'patient_pair_metrics.tsv',sep='\t')
 rr=[]
 for scheme in p.clone_scheme.unique():
  d=p[p.clone_scheme.eq(scheme)]
  block=[]
  for endpoint in ['normalized_shannon','top10_fraction','expanded_fraction']:
   z=contrasts(d.assign(endpoint=endpoint),['endpoint'],endpoint,'Tumor_minus_Normal',f'{scheme}_primary3')
   z['clone_scheme']=scheme;block.append(z)
  rr.append(family(pd.concat(block,ignore_index=True)))
 rare=pd.read_csv(OUT/'rarefaction_draw_metrics.tsv',sep='\t').groupby(['patient_id','tissue'])[['normalized_shannon','top10_fraction','expanded_fraction']].mean().reset_index()
 block=[]
 for endpoint in ['normalized_shannon','top10_fraction','expanded_fraction']:
  z=contrasts(rare.assign(endpoint=endpoint),['endpoint'],endpoint,'Tumor_minus_Normal','depth_matched_beta3');z['clone_scheme']='depth_matched_beta';block.append(z)
 rr.append(family(pd.concat(block,ignore_index=True)))
 write(pd.concat(rr,ignore_index=True),'repertoire_patient_tests.tsv')
 cap=[]
 for scheme in p.clone_scheme.unique():
  z=contrasts(p[p.clone_scheme.eq(scheme)].assign(endpoint='capture_fraction'),['endpoint'],'capture_fraction','Tumor_minus_Normal',f'{scheme}_capture_technical1');z['clone_scheme']=scheme;cap.append(z)
 write(pd.concat(cap,ignore_index=True),'capture_patient_tests.tsv')
 s=pd.read_csv(OUT/'patient_state_fractions.tsv',sep='\t')
 write(contrasts(s,['subcelltype'],'fraction','Tumor_minus_Normal','all_rna_T_subtypes32'),'state_patient_tests.tsv')
 f=pd.read_csv(OUT/'historical17_patient_fractions.tsv',sep='\t')
 write(contrasts(f,['group_id'],'fraction','Tumor_minus_Normal','historical17_exact_transfer_exploratory'),'historical17_patient_tests.tsv')
 print('TCR/state tests complete',flush=True)
if args.part in ['rna','all']:
 a=pd.read_csv(OUT/'patient_cd8_program_means.tsv',sep='\t')
 rows=[]
 for tissue in ['Tumor','Normal']:
  block=[]
  for i,program in enumerate(PROGRAMS):
   d=a[a.tissue.eq(tissue)&a.program.eq(program)].pivot(index='patient_id',columns='expansion_group',values=['cells','mean_score'])
   ok=(d['cells']['expanded']>=20)&(d['cells']['singleton']>=20)
   dif=(d['mean_score']['expanded']-d['mean_score']['singleton']).where(ok)
   row=dict(program=program,tissue=tissue,contrast='Expanded_minus_Singleton_within_conventional_CD8',family=f'{tissue}_CD8_programs6',included_patients=';'.join(d.index[dif.notna()].tolist()),excluded_patients=';'.join(d.index[dif.isna()].tolist()),min_cells_per_group=20,**paired_test(dif.values,SEED+i))
   block.append(row)
  rows.append(family(pd.DataFrame(block)))
 write(pd.concat(rows,ignore_index=True),'program_patient_tests.tsv')
 print('RNA program tests complete',flush=True)
names=['repertoire_patient_tests.tsv','capture_patient_tests.tsv','state_patient_tests.tsv','historical17_patient_tests.tsv','program_patient_tests.tsv']
tables=[]
for n in names:
 if (OUT/n).exists():
  x=pd.read_csv(OUT/n,sep='\t');x['source_table']=n;tables.append(x)
write(pd.concat(tables,ignore_index=True),'patient_tests.tsv')
receipt=dict(completed_utc=datetime.now(timezone.utc).isoformat(),part=args.part,protocol_sha256=sha(OUT/'PROTOCOL.md'),families=pd.concat(tables,ignore_index=True).groupby('family').agg(rows=('p_value','size'),finite_p=('p_value','count'),q_le_05=('q_bh',lambda x:int(x.le(.05).sum()))).reset_index().to_dict('records'),patient_bootstrap_n=10000,seed=SEED,bootstrap_unit='patients; not cells or rarefaction draws',wilcoxon_method='approx',wilcoxon_zero_method='wilcox',continuity_correction=False)
(OUT/'statistics_receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
print(json.dumps(receipt,indent=2),flush=True)
