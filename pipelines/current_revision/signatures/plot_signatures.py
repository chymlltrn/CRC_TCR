import os
from pathlib import Path
import pandas as pd,numpy as np,json,hashlib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
OUT=Path(os.environ['CRC_SIGNATURE_ROOT']).expanduser().resolve()
(OUT/'figures').mkdir(exist_ok=True)
frozen=pd.read_csv(OUT/'frozen_signature_markers.tsv',sep='\t');expr=pd.read_csv(OUT/'tables/top6_marker_donor_expression.tsv',sep='\t')
states=['c03_CD4_Tn_NR4A2','c08_CD4_Tfh_CXCL13_IL6ST','c20_CD8_Tem_GZMK','c27_CD8_MAIT_SLC4A10']
colors=['#3B6FB6','#9A6FAD','#DB8651','#32948E']
groups=states+['c26_CD8_MAIT_KLRB1','c28_CD8_IEL_CD160','other_CD4','other_CD8','other_T_lineages','nonT_B','nonT_Epi','nonT_ILC','nonT_Mye','nonT_Stromal']
labels=['c03 CD4 Tn\nNR4A2','c08 CD4 Tfh*\nCXCL13/IL6ST','c20 CD8 Tem\nGZMK','c27 CD8 MAIT\nSLC4A10','c26 CD8 MAIT\nKLRB1','c28 CD8 IEL\nCD160','Other CD4','Other CD8','Other T\nlineages','B','Epithelial','ILC','Myeloid','Stromal']
rows=[]
for state in states:
 for gene in frozen[frozen.state.eq(state)].sort_values('rank').head(6).gene:rows.append((state,gene))
summary=expr.groupby(['signature_state','gene','group']).agg(mean_log2_CPM_plus1=('log2_CPM_plus1','mean'),n_donors=('patient_id','nunique')).reset_index()
lookup=summary.set_index(['signature_state','gene','group'])
a=np.full((len(rows),len(groups)),np.nan);ns=np.zeros(len(groups),int)
for i,(state,gene) in enumerate(rows):
 for j,g in enumerate(groups):
  if (state,gene,g) in lookup.index:
   r=lookup.loc[(state,gene,g)];a[i,j]=r.mean_log2_CPM_plus1;ns[j]=max(ns[j],int(r.n_donors))
z=(a-np.nanmean(a,axis=1)[:,None])/np.nanstd(a,axis=1,ddof=0)[:,None]
source=[]
for i,(state,gene) in enumerate(rows):
 for j,g in enumerate(groups):source.append({'signature_state':state,'gene':gene,'plot_row':i+1,'group':g,'n_donors_ge10':int(ns[j]),'donor_equal_mean_log2_CPM_plus1':a[i,j],'row_zscore':z[i,j],'reference_scale':'log2(raw UMI pseudobulk CPM + 1) then donor equal mean','display_top_n_per_state':6})
pd.DataFrame(source).to_csv(OUT/'figures/Fig_signature_definition_heatmap_source.tsv',sep='\t',index=False)
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'pdf.fonttype':42,'ps.fonttype':42})
fig,ax=plt.subplots(figsize=(12.5,9.6));fig.subplots_adjust(left=.23,right=.90,bottom=.24,top=.88)
cmap=plt.get_cmap('RdBu_r').copy();cmap.set_bad('#E3E5E8')
im=ax.imshow(np.ma.masked_invalid(z),aspect='auto',cmap=cmap,vmin=-2.5,vmax=2.5,interpolation='nearest')
ax.set_xticks(np.arange(len(groups)),[f'{s}\n(n={n})' for s,n in zip(labels,ns)],rotation=45,ha='right',fontsize=8.4)
ax.set_yticks(np.arange(len(rows)),[g for s,g in rows],fontsize=9)
ax.tick_params(axis='both',length=0,pad=5)
for k,(state,col) in enumerate(zip(states,colors)):
 for tick in ax.get_yticklabels()[k*6:(k+1)*6]: tick.set_color(col)
 if k:ax.axhline(k*6-.5,color='white',linewidth=2)
 label=state.split('_')[0]+(' (secondary)' if state.startswith('c08') else '')
 ax.text(-2.35,k*6+2.5,label,rotation=90,va='center',ha='center',color=col,fontsize=10,fontweight='bold')
for s in ax.spines.values():s.set_visible(False)
ax.axvline(8.5,color='white',linewidth=3)
ax.set_title('Frozen subtype signatures from baseline CRC tumor scRNA-seq',loc='left',fontsize=14,pad=33,fontweight='bold')
ax.text(0,1.015,'Top 6 genes per frozen signature  |  19-patient scope; donor groups require ≥10 cells',transform=ax.transAxes,fontsize=10,color='#4C5561')
cax=fig.add_axes([.925,.40,.017,.24]);cb=fig.colorbar(im,cax=cax);cb.set_label('Within-gene z-score',labelpad=7);cb.set_ticks([-2,0,2]);cb.outline.set_visible(False)
fig.text(.23,.055,'Donor-equal mean log₂(CPM + 1), standardized within each gene across displayed groups.\nGray: no donor group meets ≥10 cells. Other CD4/CD8 exclude separately displayed states.\n* c08 is prespecified secondary. Discovery-cohort specificity is descriptive; scores are not calibrated cell fractions.',fontsize=9,color='#46505B',va='bottom',linespacing=1.5)
for ext in ['png','pdf']:fig.savefig(OUT/'figures'/f'Fig_signature_definition_heatmap.{ext}',dpi=220,facecolor='white',bbox_inches='tight')
plt.close(fig)
qa={'expected_genes':24,'actual_rows':len(rows),'columns':len(groups),'source_rows':len(source),'frozen_top6_order_used':True,'no_TCGA_data_used':True,'c26_no_eligible_donor':int(ns[groups.index('c26_CD8_MAIT_KLRB1')])==0,'nan_display':'gray','rendered':True,'visual_inspection':'pending'}
(OUT/'figures/Fig_signature_definition_heatmap_QA.json').write_text(json.dumps(qa,indent=2)+'\n')
print(json.dumps(qa,indent=2))
