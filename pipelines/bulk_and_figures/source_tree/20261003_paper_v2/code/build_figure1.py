#!/usr/bin/env python3
"""Server-only cohort flow; all numerical labels derive from frozen source tables."""
from release_runtime import private_literal as _private_literal, release_path as _release_path
from pathlib import Path
import argparse,json,hashlib,csv
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,required=True);a=ap.parse_args();r=a.root
 out=r/'figures';out.mkdir(exist_ok=True);sources=out/'source_tables';sources.mkdir(exist_ok=True)
 pm=r/'cohort/patient_master.tsv';lp=r/'cohort/library_master.tsv';sp=r/'single_cell/cohort_summary.json'
 hp=r.parent/'20261003_reframing_crc_vs_healthy_v1/healthy_source/healthy_library_inventory.tsv'
 d=pd.read_csv(pm,sep='\t');l=pd.read_csv(lp,sep='\t');sc=json.loads(sp.read_text());h=pd.read_csv(hp,sep='\t')
 rows=[]
 def record(scope,metric,value,unit,source):
  rows.append({'scope':scope,'metric':metric,'value':value,'unit':unit,'source_path':str(source),'source_sha256':sha(source)})
 def cohort(name):
  x=d[d.cohort.eq(name)];z={k:int(v) for k,v in {'patients':len(x),'libraries':len(l[l.cohort.eq(name)]),'known_n':x.n_binary.notna().sum(),'n0':x.n_binary.eq(0).sum(),'nplus':x.n_binary.eq(1).sum(),'unknown_n':x.n_binary.isna().sum()}.items()}
  for k,v in z.items():record(name,k,v,'libraries' if k=='libraries' else 'patients',lp if k=='libraries' else pm)
  return z
 home=cohort('home');ext=cohort('external')
 for k in ['source_patient_count','crc_patient_count','primary_paired_patient_count','paired_sample_count','primary_tcell_vdj_cells','primary_all_rna_t_cells']:record('GSE236581',k,sc[k],'cells' if 'cells' in k else 'samples' if 'sample' in k else 'patients',sp)
 def ref(study,condition=None):
  x=h[h.study.eq(study)];x=x[x.metadata_condition.eq(condition)] if condition else x
  n=len(x);nd=x.donor_key.nunique();record(study,str(condition)+'_libraries',n,'libraries',hp);record(study,str(condition)+'_source_donor_labels',nd,'source donor labels',hp);return n,nd
 n139,_=ref('immunoSEQ139','Healthy');c139,_=ref('immunoSEQ139','Colorectal Cancer');n300,d300=ref('PRJNA300878');n135,d135=ref('immunoSEQ135')
 pd.DataFrame(rows).to_csv(sources/'Figure1_cohort_counts.tsv',sep='\t',index=False)
 plt.rcParams.update({'font.family':'DejaVu Sans','font.size':15,'pdf.fonttype':42,'svg.fonttype':'none'})
 fig=plt.figure(figsize=(12,7.3),facecolor='white');axs=[fig.add_axes([.035+i*.325,.07,.3,.86]) for i in range(3)]
 cols=[_private_literal('private_7b2b8c6c640f6ec4'),_private_literal('private_b97b3407f39a4dae'),'#287D8E'];titles=['Peripheral blood','Public TCR references','Paired tissue']
 def card(ax,y,height,title,body,color):
  ax.add_patch(FancyBboxPatch((.015,y),.97,height,boxstyle='round,pad=0.015,rounding_size=0.02',facecolor='#F4F6F8',edgecolor='#D5DADF',lw=1))
  ax.text(.055,y+height-.047,title,va='top',fontsize=15.5,fontweight='bold',color=color)
  ax.text(.055,y+height-.13,body,va='top',fontsize=15,linespacing=1.42,color='#25282C')
 for i,ax in enumerate(axs):
  ax.set(xlim=(0,1),ylim=(0,1));ax.axis('off');ax.text(0,1.035,chr(65+i),fontweight='bold',fontsize=20);ax.text(.085,1.035,titles[i],fontsize=16,fontweight='bold')
 card(axs[0],.625,.34,'Home CRC',f"{home['libraries']} libraries → {home['patients']} patients\n{home['known_n']} with recorded N\n{home['n0']} N0 | {home['nplus']} N+\n{home['unknown_n']} unknown N retained for Q1",cols[0])
 card(axs[0],.22,.34,'External CRC',f"{ext['patients']} patients\n{ext['known_n']} with recorded N\n{ext['n0']} N0 | {ext['nplus']} N+\n{ext['unknown_n']} unknown N retained for Q1",cols[0])
 axs[0].text(.02,.12,'Patient-level N evaluation\nand exact Q1 detection',fontsize=15,fontweight='bold',color=cols[0],va='top',linespacing=1.4)
 card(axs[1],.65,.315,'immunoSEQ139',f'{n139} Healthy/Blood libraries\n{c139} CRC/PBMC libraries\nClinical donor independence\nnot verified',cols[1])
 card(axs[1],.395,.20,'PRJNA300878',f'{n300} sorted libraries\n{d300} donors in 5 twin pairs',cols[1])
 card(axs[1],.14,.20,'immunoSEQ135',f'{n135} libraries from {d135} donors\nSorted subsets / repeat plates',cols[1])
 axs[1].text(.02,.07,'Q1 evaluated by source',fontsize=15,fontweight='bold',color=cols[1],va='top')
 card(axs[2],.65,.315,'GSE236581',f"{sc['source_patient_count']} source patients\nExclude 2 duodenal cancers\n{sc['crc_patient_count']}{_private_literal('private_d65d731d5faa8e37')}",cols[2])
 card(axs[2],.23,.365,'Baseline tissue pairs',f"{sc['primary_paired_patient_count']} patients | {sc['paired_sample_count']} tissue samples\n{sc['primary_all_rna_t_cells']:,} RNA T cells\n{sc['primary_tcell_vdj_cells']:,} accepted single-β cells\nPatient-paired repertoire and\ntranscriptional-state analyses",cols[2])
 axs[2].text(.02,.13,'Previously explored cohort\nNo matched blood participants',fontsize=15,fontweight='bold',color=cols[2],va='top',linespacing=1.4)
 for fmt in ['pdf','svg','png']:fig.savefig(out/f'Figure1.{fmt}',dpi=300,bbox_inches='tight',facecolor='white')
 plt.close(fig)
 (out/'Figure1_receipt.json').write_text(json.dumps({'status':'GENERATED_REQUIRES_VISUAL_REVIEW','inputs':{str(p):sha(p) for p in [pm,lp,sp,hp]},'script_sha256':sha(__file__),'source_rows':len(rows)},indent=2)+'\n')
 print(json.dumps({'figure':str(out/'Figure1.pdf'),'source_rows':len(rows),'cohorts':[home,ext],'paired_patients':sc['primary_paired_patient_count']}))
if __name__=='__main__':main()
