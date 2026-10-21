#!/usr/bin/env python3
from release_runtime import private_literal as _private_literal, release_path as _release_path
from pathlib import Path
import pandas as pd,numpy as np,json,hashlib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
B=Path(_release_path('path_ab9dd0e0159dec18'))
O=B/'revision_v11';T=O/'tables/nodal';A=O/'audit/nodal';F=O/'figures/nodal'
d=pd.read_csv(T/'patient_metrics.tsv',sep='\t')
u=pd.read_csv(T/'expanded12_unadjusted.tsv',sep='\t')
a=pd.read_csv(T/'expanded12_adjusted.tsv',sep='\t')
alias=pd.read_csv(T/'clonality_display_alias.tsv',sep='\t')
dictionary=pd.read_csv(T/'metric_dictionary.tsv',sep='\t')
rows=[]
for r in pd.concat([u,alias],ignore_index=True).query("analysis == 'known_recorded_N'").to_dict('records'):
 metric=r['metric'];aa=a[(a.analysis.eq(r['analysis']))&(a.cohort.eq(r['cohort']))&a.feature.eq('normalized_shannon' if metric=='clonality' else metric)].iloc[0]
 scale=100 if r['unit']=='proportion' else 1e-6 if metric in ['total_reads','input_reads'] else 1
 unit='%' if scale==100 else 'million reads' if scale==1e-6 else r['unit']
 def num(x):
  return f'{x*scale:,.0f}' if metric=='unique_aa' else f'{x*scale:.3f}'
 effects=[aa.OR,aa.CIlo,aa.CIhi]
 if metric=='clonality':effects=[1/aa.OR,1/aa.CIhi,1/aa.CIlo]
 rows.append(dict(cohort=r['cohort'],metric=metric,label=r['label'],unit=unit,N0_n=r['N0'],Nplus_n=r['Nplus'],N0_median_IQR=f"{num(r['median_N0'])} [{num(r['q25_N0'])}, {num(r['q75_N0'])}]",Nplus_median_IQR=f"{num(r['median_Nplus'])} [{num(r['q25_Nplus'])}, {num(r['q75_Nplus'])}]",unadjusted_P=r['P'],unadjusted_q_BH12=r['q'],adjusted_OR_CI=f'{effects[0]:.3f} [{effects[1]:.3f}, {effects[2]:.3f}]',adjusted_P=aa.P,adjusted_q_BH12=aa.q,adjustment=aa.covariates,transform=aa.transform,note='Exact complementary display; shares normalized Shannon test' if metric=='clonality' else ''))
pd.DataFrame(rows).to_csv(T/'manuscript_metric_summary.tsv',sep='\t',index=False)
# Preserve exact model evidence and append current prespecified candidate branch
versions=pd.read_csv(T/'unified_nodal_evidence.tsv',sep='\t').query("evidence_type == 'prediction'").copy()
versions['source_table_absolute']=versions.source_table.map(lambda x:'; '.join(str(B/'revision_v5/nodal'/part.strip()) for part in x.split(';')))
versions['recomputed_v11']=False
versions.loc[versions.protocol.eq('current_shared_variable_frozen_transfer'), 'CI_type']='conditional_frozen_external_predictions_patient_stratified_percentile'
v7=B/'revision_v7/q1_depth_n/models';m=pd.read_csv(v7/'metrics.tsv',sep='\t');p=pd.read_csv(v7/'paired_model_increment.tsv',sep='\t');ci=pd.read_csv(v7/'full_pipeline_paired_ci.tsv',sep='\t')
candidate=[]
for r in m.to_dict('records'):
 if r['model']=='M1':continue
 rr=dict(protocol=_private_literal('private_739f3b1c818e52c1'),evidence_type='prediction',model=r['model'],cohort=_private_literal('private_7e027ebbcb151784'),n=126,N0=70,N_positive=56,unit='patient',endpoint='recorded N0/N+',fold_history='patient-grouped nested outer5/inner3; seeds17/29/43; fixed folds shared with current main ridge',baseline_variables='age; sex; batch; log1p reads; log1p unique AA',TCR_input='fixed Q1 or Q2 AA-detection indicator >=2 reads',AUROC=r['AUROC'],AUPRC=r['AUPRC'],log_loss=r['log_loss'],Brier=r['Brier'],source_table=str(v7/'metrics.tsv'),source_table_absolute=str(v7/'metrics.tsv'),source_receipt=str(v7/'fold_audit.tsv'),comparability_limit='Candidates nominated using outcomes historically; exploratory incremental evaluation, not independent candidate validation',recomputed_v11=False)
 if r['model'].startswith('M1_'):
  target=r['model'][-2:];pr=p[p.candidate.eq(target)].iloc[0];cr=ci[ci.candidate.eq(target)&ci.metric.eq('delta_log_loss')].iloc[0]
  rr.update(comparator='M1',baseline_AUROC=float(m[m.model.eq('M1')].AUROC.iloc[0]),delta_AUROC=pr.delta_AUROC,delta_log_loss=pr.delta_log_loss,CI_low=cr.ci_low,CI_high=cr.ci_high,CI_target='delta_log_loss',CI_type=cr.CI,resamples=200)
 candidate.append(rr)
versions=pd.concat([versions,pd.DataFrame(candidate)],ignore_index=True)
versions.to_csv(T/'model_version_audit.tsv',sep='\t',index=False)
# Publication-style compact forest: all12 outcomes; per-cohort adjustments differ as labelled in caption.
labels={'shannon_entropy':'Shannon entropy','normalized_shannon':'Normalized Shannon','simpson_diversity':'Simpson diversity','top1_fraction':'Largest AA sequence','top10_fraction':'Top 10 AA sequences','weighted_mean_cdr3_length':'Mean CDR3β length','unique_aa':'AA sequence richness','total_reads':'Productive read count','input_reads':'Input read count','alignment_fraction':'Alignment fraction','trb_alignment_fraction':'TRB alignment fraction','productive_input_fraction':'Productive / input reads'}
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':7.5,'axes.titlesize':9,'axes.labelsize':8,'pdf.fonttype':42,'ps.fonttype':42,'svg.fonttype':'none','axes.spines.top':False,'axes.spines.right':False})
fig,axes=plt.subplots(1,2,figsize=(7.09,4.5),sharey=True)
for ax,co,col,title,letter in zip(axes,['home','external'],['#0072B2','#D55E00'],['Study cohort (70 N0; 56 N+)','External cohort (109 N0; 54 N+)'],['A','B']):
 sub=a[a.analysis.eq('known_recorded_N')&a.cohort.eq(co)].set_index('feature').loc[dictionary.metric]
 yy=np.arange(len(sub))[::-1]
 ax.axvline(1,color='#9AA4AE',lw=.8,ls='--')
 ax.axhline(4.5,color='#D8DEE3',lw=.65)
 ax.errorbar(sub.OR,yy,xerr=[sub.OR-sub.CIlo,sub.CIhi-sub.OR],fmt='o',mfc='white',mec=col,ecolor=col,ms=4,capsize=2,lw=1)
 ax.set_xscale('log');ax.set_xlim(.25,10);ax.set_xticks([.25,.5,1,2,4]);ax.set_xticklabels(['0.25','0.5','1','2','4'])
 ax.set_yticks(yy);ax.set_yticklabels([labels[v] for v in sub.index]);ax.tick_params(axis='y',length=0)
 ax.set_xlabel('Adjusted odds ratio for N+ (95% CI)')
 ax.set_title(title,pad=20)
 ax.text(-.06,1.10,letter,transform=ax.transAxes,fontweight='bold',fontsize=11)
 for y,qv in zip(yy,sub.q):ax.text(9.4,y,f'{qv:.3f}',ha='right',va='center',fontsize=6.5,color='#46545F')
 ax.text(9.4,11.9,'BH q',ha='right',va='bottom',fontsize=7,color='#46545F')
ax=axes[1];ax.tick_params(axis='y',labelleft=False)
fig.subplots_adjust(left=.22,right=.985,wspace=.12,bottom=.15,top=.85)
stem=F/'Figure_nodal_metrics12_adjusted'
fig.savefig(str(stem)+'.pdf');fig.savefig(str(stem)+'.svg');fig.savefig(str(stem)+'.png',dpi=600);fig.savefig(str(stem)+'.preview.png',dpi=140);plt.close(fig)
(F/'Figure_nodal_metrics12_adjusted.caption.md').write_text("""Expanded patient-level N-status comparisons. Adjusted odds ratios and 95% confidence intervals for 12 fixed repertoire and technical metrics are shown for the study and external CRC cohorts. Each row corresponds to one metric, not a predictive multivariable panel. Counts were log1p-transformed before standardization; other metrics were standardized on their original scale. Study-cohort estimates adjusted for age, sex, recorded neoadjuvant treatment and batch; repertoire metrics additionally adjusted for productive read depth. External estimates adjusted for sequencing pool and, for repertoire metrics, productive read depth. The five technical outcomes below the divider were not additionally adjusted for read depth. BH correction included all 12 outcomes within each cohort. Hollow symbols indicate q >=0.05. AA indicates the full CDR3beta amino-acid sequence; richness and top-sequence fractions use AA-level aggregation. Clonality equals one minus normalized Shannon diversity and is therefore not shown as a second independent hypothesis. These expanded comparisons are supplementary to the unchanged original three-summary and 78-gene tests.\n""")
# Helpful data links for manuscript author
summary=_private_literal('private_fe6baae1ef7ff492')
(A/'NODAL_AUDIT_REPORT.md').write_text(summary)
(A/'abstract_recommendation.md').write_text("Whole-repertoire diversity, clonality and sequence-representation models did not consistently distinguish recorded nodal status.\\n\\nRaw Shannon P=.044 is nominal (BH12 q=.172); it should not be elevated into a significant abstract result. Productive read depth likewise has nominal P=.037 but q=.172.\\n")
print('outputs prepared',len(versions),'model version rows')

