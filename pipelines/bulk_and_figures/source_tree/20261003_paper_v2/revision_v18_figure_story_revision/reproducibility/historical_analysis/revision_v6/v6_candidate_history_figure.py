from pathlib import Path
import sys,json,argparse
import pandas as pd
import numpy as np
from statsmodels.stats.proportion import proportion_confint
from matplotlib.patches import Rectangle, FancyArrowPatch
import hashlib
sys.path.insert(0,str(Path(__file__).parent))
from paper_style import *
a=argparse.ArgumentParser();a.add_argument('--root',type=Path,required=True);V=a.parse_args().root.resolve();R=V.parent;S=V/'candidates/figures/source';S.mkdir(parents=True,exist_ok=True)
P=pd.read_csv(R/'revision_v5/legacy/patient_Q1_Q2.tsv',sep='\t');H=P[P.cohort.eq('home')].copy()
RULES=pd.read_csv(R/'revision_v5/legacy/selection_rule_ledger.tsv',sep='\t');assert len(RULES)==12, 'Both complete historical rule sets are required'
# A expands the reported historical topology; B-E retain their evidence and native plot sizes.
HEIGHT_MM=240
fig=new_figure(HEIGHT_MM)
ax=fig.add_axes([.08,.655,.89,.31]);ax.set_xlim(0,160);ax.set_ylim(0,74.4);ax.axis('off')
panel(ax,'A',x=-.06,y=1.025)
ax.text(0,77.0,'Historical nomination: two parallel reported routes',fontsize=8.5,fontweight='bold',va='baseline')
INK='#29353D';LINE='#AAB4BC';LEFT='#536B79';RIGHT='#776185'
def box(x,y,w,h,fill='white',edge=LINE,lw=.65):
 r=Rectangle((x,y),w,h,facecolor=fill,edgecolor=edge,lw=lw,zorder=2);ax.add_patch(r);return r
def label(x,y,t,size=7.1,color=INK,weight='normal',ha='left',va='center'):
 return ax.text(x,y,t,fontsize=size,color=color,fontweight=weight,ha=ha,va=va,zorder=4,linespacing=1.25)
def arrow(p1,p2,color=LINE,lw=.75):
 ar=FancyArrowPatch(p1,p2,arrowstyle='-|>',mutation_scale=7,lw=lw,color=color,zorder=1);ax.add_patch(ar)
# One reported numerical source, not a recovered shared sequence-membership object.
box(37,62,86,10,'#F5F7F9',edge='#8D99A2',lw=.7)
label(80,69.5,'Reported background • sequence membership unrecovered',7.0,color='#64717B',ha='center')
label(80,65.2,'1,367,835 CDR3β amino acid sequences',9.0,weight='bold',ha='center')
ax.plot([80,80],[62,59],color=LINE,lw=.75,zorder=1)
ax.plot([38,122],[59,59],color=LINE,lw=.75,zorder=1)
arrow((38,59),(38,56.8));arrow((122,59),(122,56.8))
# Equal-width branches do not encode attrition, prevalence or evidence strength.
rows=[
 (0,LEFT,'#F1F5F7','Dataset-aware screen',[
  ('Reference\ncontrast','Meta-FDR ≤ 0.05\nLower in all 3 healthy datasets'),
  ('Prevalence','CRC ≥ 10%\nMaximum healthy ≤ 2%'),
  ('Support','≥2 internal batches\n≥2 external CRC datasets or 1 + TIL')],
  '106','sequences reported'),
 (84,RIGHT,'#F5F2F7','Six-criterion nomination',[
  ('Statistical\nevidence','CRC vs healthy: P < .01¹\nCRC FDR < .20'),
  ('Nodal\nevidence','N+ vs N−: P < .01\nN+ enrichment (top 2,000 tested)'),
  ('Tissue &\ndirection','Detected in CRC TIL\nCRC > merged healthy prevalence')],
  'Q1 + Q2','candidates reported')]
for x,color,tint,title,criteria,result,result_label in rows:
 box(x,49,76,8,tint,edge=color,lw=.75)
 label(x+3,54.6,title,8.0,color=color,weight='bold')
 label(x+3,50.9,'ALL SIX CONDITIONS REQUIRED',7.0,color=color)
 for i,(name,conditions) in enumerate(criteria):
  y=18.0+(2-i)*(31.0/3)
  box(x,y,76,31.0/3,'white',edge='#D9DFE4',lw=.55)
  box(x,y,21,31.0/3,tint,edge='none',lw=0)
  label(x+2.4,y+31.0/6,name,7.0,color=color,weight='bold')
  label(x+23,y+31.0/6,conditions,7.0)
 arrow((x+38,17.6),(x+38,14.5),color=color)
 box(x,5,76,9,tint,edge=color,lw=.75)
 label(x+4,9.4,result,10.5,color=color,weight='bold')
 label(x+27,9.4,result_label,7.5,color=color)
label(0,1.0,'¹ CMH for top 10,000; Fisher otherwise. Per-rule counts and a 106 → 2 transition unrecovered.',7.0,color='#616D76')
# Map v3 axes into v4 while retaining their physical sizes and all data encodings.
def old_axes(rect):
 x,y,w,h=rect
 return fig.add_axes([x,y*211/HEIGHT_MM,w,h*211/HEIGHT_MM])
ax=old_axes([.12,.465,.365,.235]);panel(ax,'B','Fixed count thresholds',x=-.16,y=1.07)
out=[]
for q in ['Q1','Q2']:
 for co,ls,m in [('home','-','o'),('external','--','^')]:
  d=P[P.cohort.eq(co)];xx=[1,2,5,10,20];yy=[d[f'{q}_AA_reads'].ge(t).mean()*100 for t in xx]
  ax.plot(range(5),yy,ls=ls,marker=m,ms=3,color=COLORS[q.lower()],lw=.9,label=f'{q} '+('Home' if co=='home' else 'External'))
  out.extend([dict(candidate=q,cohort=co,threshold=t,n=len(d),carriers=int(d[f'{q}_AA_reads'].ge(t).sum()),fraction=y/100) for t,y in zip(xx,yy)])
ax.set_xticks(range(5),xx);ax.set_xlabel('Minimum summed AA reads');ax.set_ylabel('Patients carrying sequence (%)');ax.set_ylim(-2,100);ax.legend(fontsize=7,loc='upper right');polish(ax)
assert (S/'S7_thresholds.tsv').exists()
ax=old_axes([.665,.465,.295,.235]);panel(ax,'C','Home batch distribution',x=-.20,y=1.07)
out=[]
for i,(co,label,color,mark) in enumerate([('Q1_AA_reads','Q1 AA',COLORS['q1'],'o'),('Q1_NT_reads','Q1 NT',COLORS['q1'],'D'),('Q2_AA_reads','Q2 AA',COLORS['q2'],'s')]):
 for j,(batch,d) in enumerate(H.groupby('batch',sort=True)):
  k=int(d[co].ge(2).sum());n=len(d);lo,hi=proportion_confint(k,n,method='wilson')
  ax.errorbar(k/n*100,j+(i-1)*.19,xerr=np.array([[k/n-lo],[hi-k/n]])*100,fmt=mark,mfc='white' if i==1 else color,mec=color,color=color,ms=3,lw=.7,capsize=1.5,label=label if j==0 else None)
  out.append(dict(batch=batch,definition=label,k=k,n=n,threshold=2,lo=lo,hi=hi,prevalence=k/n))
b=H.groupby('batch').size();ax.set_yticks(range(3),[f'{str(x)[-1]} (n={n})' for x,n in b.items()]);ax.set_ylim(2.5,-.5);ax.set_xlim(-2,102);ax.set_xticks([0,50,100]);ax.set_xlabel('Carriage (%)');ax.legend(loc='upper center',bbox_to_anchor=(.5,-.25),ncol=3,columnspacing=.8,handletextpad=.4,fontsize=7);polish(ax)
assert (S/'S7_batch_carriage.tsv').exists()
ax=old_axes([.12,.10,.365,.235]);panel(ax,'D','Concentration of sequence reads',x=-.16,y=1.07)
out=[]
for q in ['Q1','Q2']:
 d=H.sort_values(f'{q}_AA_reads',ascending=False);vals=d[f'{q}_AA_reads'].to_numpy();cum=np.cumsum(vals)/sum(vals)*100
 ax.plot(np.arange(len(vals)+1),np.r_[0,cum],color=COLORS[q.lower()],ls='-' if q=='Q1' else '--',lw=1,label=q)
 for rank,(_,rr) in enumerate(d.iterrows(),1):out.append(dict(candidate=q,rank=rank,patient_key=rr.patient_key,primary_sample_ids=rr.primary_sample_ids,AA_reads=rr[f'{q}_AA_reads'],cumulative_read_percent=cum[rank-1]))
 ax.text(55,78 if q=='Q1' else 67,f'{q}: top patient {vals[0]/sum(vals):.1%}',color=COLORS[q.lower()],fontsize=7.2)
ax.plot([0,137],[0,100],color='#B5B9BD',ls=':',lw=.7);ax.set_xlim(0,137);ax.set_ylim(0,103);ax.set_xlabel('Patients ranked by matching reads');ax.set_ylabel('Cumulative candidate reads (%)');polish(ax)
assert (S/'S7_read_concentration.tsv').exists()
ax=old_axes([.665,.10,.26,.235]);panel(ax,'E','Recorded-N carriage',x=-.20,y=1.07)
out=[]
for i,(q,col) in enumerate([('Q1','Q1_AA_reads'),('Q2','Q2_AA_reads')]):
 for j,(nn,nlab,color) in enumerate([(0,'N0',COLORS['n0']),(1,'N+',COLORS['nplus'])]):
  d=H[H.n_binary.eq(nn)];k=int(d[col].ge(2).sum());n=len(d);lo,hi=proportion_confint(k,n,method='wilson');y=i+(j-.5)*.18
  ax.errorbar(k/n*100,y,xerr=np.array([[k/n-lo],[hi-k/n]])*100,fmt='o',color=color,ms=3.5,lw=.8,capsize=1.5,label=nlab if i==0 else None)
  ax.text(104,y,f'{k}/{n}',ha='left',va='center',fontsize=7)
  out.append(dict(candidate=q,n_binary=nn,k=k,n=n,threshold=2,lo=lo,hi=hi,prevalence=k/n))
ax.set_yticks([0,1],['Q1 AA','Q2 AA']);ax.set_ylim(1.6,-.6);ax.set_xlim(0,102);ax.set_xticks([0,50,100]);ax.set_xlabel('Carriage (%)');ax.legend(loc='lower right',fontsize=7,ncol=2);polish(ax)
assert (S/'S7_N_carriage.tsv').exists()
export_figure(fig,V/'candidates/figures/Supplementary_Figure_S7',list(S.glob('S7_*.tsv')))
print('S7 v6 exported; unchanged historical/current source tables and dashed Q2 concentration')

