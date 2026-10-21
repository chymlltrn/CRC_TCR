#!/usr/bin/env python3
"""Remote-only v6 overview figures; read frozen results without refitting."""
from release_runtime import private_literal as _private_literal, release_path as _release_path
from pathlib import Path
import argparse
import hashlib
import json
import shutil
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import FancyBboxPatch, Circle
from paper_style import COLORS, new_figure, panel, polish, export_figure

ROOT = Path(_release_path('path_38c4da8147aa1485'))
OLD = ROOT.parent/'revision_v5'
FEATS = ['normalized_shannon', 'top10_fraction', 'weighted_mean_cdr3_length']
MC = {'M0':'#8A9299', 'M1':'#222222', 'M2':'#CC79A7', 'M3':'#6F4C9B'}

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def row(stem, num, title, question, sources):
    return dict(figure_number=num, title=title, question=question,
        **{key:str(stem.with_suffix(ext).relative_to(ROOT)) for key,ext in [
            ('pdf_path','.pdf'),('svg_path','.svg'),('png_path','.png'),
            ('preview_path','.preview.png'),('grayscale_path','.grayscale.png'),('export_path','.export.json')]},
        include_in_manuscript='true' if not num.startswith('S') else 'false',
        section='supplement' if num.startswith('S') else 'main',
        source_tables=';'.join(map(str,sources)))

def inherit():
    registry=pd.read_csv(OLD/'figures/figure_manifest.tsv', sep='\t',dtype=str).set_index('figure_number')
    mapping={'2':'S1','S1':'S2','S2':'S3','S3':'S4','S4':'S5','S7':'S8','S8':'S9','S9':'S10','S10':'S11'}
    out=ROOT/'figures/inherited';out.mkdir(parents=True,exist_ok=True)
    manifest=[];receipts=[]
    for old,new in mapping.items():
        r=registry.loc[old].to_dict();stem=out/f'{new}_preserved'
        source_paths=[]
        for key,ext in [('pdf_path','.pdf'),('svg_path','.svg'),('png_path','.png'),
                        ('preview_path','.preview.png'),('grayscale_path','.grayscale.png'),('export_path','.export.json')]:
            src=OLD/r[key];dst=stem.with_suffix(ext)
            assert src.exists(),src
            shutil.copy2(src,dst)
            source_paths.append(str(src))
            receipts.append(dict(figure_number=new, old_figure_number=old, source=str(src),
                                 output=str(dst),source_sha256=sha(src),output_sha256=sha(dst)))
        manifest.append(row(stem,new,r['title'],r['question'],r['source_tables'].split(';')))
    pd.DataFrame(receipts).to_csv(out/'inheritance_receipt.tsv',sep='\t',index=False)
    pd.DataFrame(manifest).to_csv(out/'figure_manifest_rows.tsv',sep='\t',index=False)
    print(f'Preserved {len(manifest)} supplementary figures with identical bytes.')

def figure1():
    out=ROOT/'figures/overview';out.mkdir(parents=True,exist_ok=True)
    source=out/'source';source.mkdir(exist_ok=True)
    names=['F1_patient_QC.tsv','F1_clinical_patient_data.tsv','F1_clinical_summary.tsv']
    for n in names:shutil.copy2(OLD/'figures/source'/n,source/n)
    shutil.copy2(ROOT/'cohort/baseline_patient_registry.tsv',source/'F1_sc_patient_registry.tsv')
    qc=pd.read_csv(source/names[0],sep='\t')
    clin=pd.read_csv(source/names[1],sep='\t')
    sc=pd.read_csv(source/'F1_sc_patient_registry.tsv',sep='\t').sort_values('patient_id')
    assert len(qc)==339 and len(clin)==137 and len(sc)==19
    assert sc.high_depth_primary.sum()==18
    fig=new_figure(207)
    gs=fig.add_gridspec(3,2,left=.125,right=.96,top=.945,bottom=.115,
        height_ratios=[1.04,1.36,.82],hspace=.69,wspace=.56)
    ax=fig.add_subplot(gs[0,:]);ax.set_axis_off();ax.set_xlim(0,1);ax.set_ylim(0,1)
    panel(ax,'A','Complementary sources of clinical and cellular evidence',x=-.10)
    cards=[(.0,'CRC blood repertoires','Home 137  ·  External 202',COLORS['n0']),
           (.345,'Paired cellular reference','GSE236581  ·  19 patients',COLORS['tumor']),
           (.69,'Public sequence references','Disease and material strata','#6F4C9B')]
    for x,title,sub,c in cards:
        ax.add_patch(FancyBboxPatch((x,.49),.31,.42,boxstyle='round,pad=0.008,rounding_size=.025',
                                  fc='#F5F7F9',ec='#D7DEE3',lw=.7))
        ax.plot([x+.02,x+.29],[.86,.86],c=c,lw=2.4,solid_capstyle='round')
        ax.text(x+.155,.745,title,ha='center',va='center',fontsize=8,fontweight='bold')
        ax.text(x+.155,.59,sub,ha='center',va='center',fontsize=7)
    ax.text(.155,.33,'Candidate recurrence\nRecorded N: 126 / 163',ha='center',va='center',fontsize=7)
    ax.text(.5,.33,'Blood · tumour · adjacent\nRNA-linked TCR identities',ha='center',va='center',fontsize=7)
    ax.text(.845,.33,'Blood · tissue · TIL\nSource-specific denominators',ha='center',va='center',fontsize=7)
    ax.plot([.155,.155,.845,.845],[.22,.1,.1,.22],c='#9AA5AE',lw=.8)
    ax.plot([.5,.5],[.22,.1],c='#9AA5AE',lw=.8)
    for x in [.155,.5,.845]:ax.plot([x,x],[.49,.43],c='#9AA5AE',lw=.8)
    ax.text(.5,.01,'Within-person sharing and cross-cohort sequence support',ha='center',va='top',fontsize=7,
            bbox=dict(facecolor='white',edgecolor='none',pad=3))

    host=fig.add_subplot(gs[1,0]);host.set_axis_off();panel(host,'B','Home clinical composition',x=-.26)
    a=host.inset_axes([0,.60,1,.37]);ages=clin.age_years.to_numpy()
    rng=np.random.default_rng(20261004)
    a.scatter(ages,rng.uniform(-.19,.19,len(ages)),s=7,c=COLORS['n0'],alpha=.48,linewidths=0)
    q25,med,q75=np.quantile(ages,[.25,.5,.75])
    a.plot([q25,q75],[.37,.37],c='#222222',lw=2.7);a.plot(med,.37,'|',c='#222222',ms=9,mew=1.5)
    a.set(xlim=(25,92),ylim=(-.25,.57),xlabel='Age (years)',yticks=[]);a.set_xticks([30,50,70,90]);polish(a)
    a.text(.02,.99,f'Median {med:.0f} (IQR {q25:.0f}–{q75:.0f})',transform=a.transAxes,fontsize=7,va='top')
    a=host.inset_axes([.12,.01,.88,.37])
    bars=[('Sex',[60,77],['Female','Male']),('Neoadjuvant',[126,11],['No record','Recorded']),('Recorded N',[70,56,11],['N0','N+','Unknown'])]
    palette=[COLORS['n0'],COLORS['tumor'],'#ABB2B9']
    for y,(name,counts,labs) in enumerate(bars):
        left=0
        for k,(n,lab) in enumerate(zip(counts,labs)):
            a.barh(y,n,left=left,height=.60,color=palette[k],edgecolor='white',linewidth=.6)
            a.text(left+n/2,y,str(n),ha='center',va='center',fontsize=7,color='white' if k<2 else '#222222')
            left+=n
    a.set(xlim=(0,137),ylim=(2.5,-.5));a.set_yticks(range(3),['Sex\nFemale / Male','Neoadjuvant\nNo / Yes','Recorded N\nN0 / N+ / Unknown'],fontsize=7)
    a.set_xticks([]);a.tick_params(axis='y',length=0)
    for s in a.spines.values():s.set_visible(False)
    ax=fig.add_subplot(gs[1,1])
    for coh,label,c,m in [('home','Home (137)',COLORS['n0'],'o'),('external','External (202)','#6F4C9B','^')]:
        d=qc[qc.cohort.eq(coh)];ax.scatter(d.library_qc_reads,d.library_qc_unique_aa,s=12,c=c,marker=m,alpha=.55,lw=.2,edgecolor='white',label=label)
    ax.set(xscale='log',yscale='log',xlabel='Productive reads',ylabel='Unique CDR3β amino-acid sequences')
    ax.legend(loc='lower right',fontsize=7);panel(ax,'C','Bulk repertoire coverage',x=-.22);polish(ax)
    ax=fig.add_subplot(gs[2,:])
    from matplotlib.colors import LogNorm
    from matplotlib.patches import Rectangle
    vals=sc[['blood_beta_cells','tumor_beta_cells','normal_beta_cells']].to_numpy().T
    hm=ax.imshow(vals,aspect='auto',cmap='Blues',norm=LogNorm(vmin=50,vmax=15000))
    for j in range(vals.shape[1]):
        for i in range(3):
            n=vals[i,j];s=f'{n/1000:.1f}k' if n>=1000 else str(n)
            ax.text(j,i,s,ha='center',va='center',fontsize=7,color='white' if n>=1500 else '#22313D')
    p08=list(sc.patient_id).index(_private_literal('private_33b1c1e2dc1ff0bc'));ax.add_patch(Rectangle((p08-.5,-.5),1,1,fill=False,ec=COLORS['tumor'],lw=1.5))
    ax.set_xticks(range(len(sc)),sc.patient_id,rotation=45,ha='right',fontsize=7)
    ax.set_yticks([0,1,2],['Blood','Tumour','Adjacent'],fontsize=7);ax.tick_params(length=0)
    for s in ax.spines.values():s.set_visible(False)
    panel(ax,'D','Baseline RNA-linked T cells with an accepted single β chain',x=-.10)
    ax.text(0,-.57,_private_literal('private_5bd662e461b575cc'),
            transform=ax.transAxes,fontsize=7,ha='left')
    cax=ax.inset_axes([.76,-.43,.24,.07]);cb=fig.colorbar(hm,cax=cax,orientation='horizontal',ticks=[50,500,5000]);cb.ax.set_xticklabels(['50','500','5,000']);cb.ax.tick_params(labelsize=7,pad=1,length=2)
    stem=out/'Figure1_Study_design'
    sources=[source/n for n in names]+[source/'F1_sc_patient_registry.tsv']
    receipt=export_figure(fig,stem,source_tables=sources);plt.close(fig)
    pd.DataFrame([row(stem,'1','Study design and cohort context','Which cohorts support each clinical or cellular question?',sources)]).to_csv(out/'figure1_manifest_row.tsv',sep='\t',index=False)
    (out/'Figure1_caption.md').write_text(_private_literal('private_31ad41a25e94c0f6'))
    print(json.dumps({'figure':'1','out_of_canvas':receipt['out_of_canvas_text'],'minimum_font':receipt['font_points_min']}))

def figure6():
    src=OLD/'nodal/source_tables'
    files=['Figure2_adjusted_effects.tsv','ROC_curves.tsv','prediction_metrics_recomputed.tsv',
           'Figure3_incremental_log_loss.tsv','Figure3_external_ROC.tsv','Figure3_external_calibration.tsv']
    tables={name:pd.read_csv(src/name,sep='\t') for name in files}
    source=ROOT/'figures/overview/source';source.mkdir(parents=True,exist_ok=True)
    out=ROOT/'figures/overview';out.mkdir(parents=True,exist_ok=True)
    for name in files: shutil.copy2(src/name,source/name)
    assoc=tables[files[0]].set_index('feature').loc[FEATS].reset_index()
    assert assoc.n.eq(126).all() and assoc.N0.eq(70).all() and assoc.Nplus.eq(56).all()
    assert assoc.family_size.eq(3).all()
    met=tables['prediction_metrics_recomputed.tsv'].query(_private_literal('private_7c99a2c5aec51613')).set_index('model')
    cur=tables['ROC_curves.tsv'].query(_private_literal('private_7c99a2c5aec51613'))
    inc=tables['Figure3_incremental_log_loss.tsv'].set_index('model').loc[['M3','M1_sceptr']].reset_index()
    ext=tables['Figure3_external_ROC.tsv'];cal=tables['Figure3_external_calibration.tsv']
    ext_met=pd.read_csv(OLD/'nodal/frozen/paper_v1/models/external_metrics.tsv',sep='\t').set_index('model')
    shutil.copy2(OLD/'nodal/frozen/paper_v1/models/external_metrics.tsv',source/'external_metrics.tsv')
    fig=new_figure(185)
    gs=fig.add_gridspec(3,2,left=.145,right=.965,bottom=.085,top=.945,
                         width_ratios=[1,1],height_ratios=[1,.67,1],hspace=.73,wspace=.48)
    ax=fig.add_subplot(gs[0,0])
    for i,r in assoc.iterrows():
        ax.plot([r.CIlo,r.CIhi],[i,i],color='#555F68',lw=1.1)
        ax.plot(r.OR,i,'o',color='#222222',ms=4.3)
        ax.text(2.16,i,f'q={r.q:.3f}',fontsize=7,ha='left',va='center')
    ax.axvline(1,lw=.7,ls=(0,(3,3)),color='#89939B')
    ax.set_xscale('log');ax.set_xlim(.4,3.4);ax.set_ylim(2.6,-.65)
    ax.set_xticks([.5,1,2],[.5,1,2]);ax.minorticks_off()
    ax.set_yticks([0,1,2],['Shannon','Top-10','CDR3β length'])
    ax.set_xlabel('Adjusted OR per 1 SD')
    panel(ax,'A','Repertoire associations',x=-.28)
    ax.text(0,1.01,'Home: 70 N0, 56 N+',transform=ax.transAxes,fontsize=7,color='#65717A')
    polish(ax)
    ax=fig.add_subplot(gs[0,1])
    for model in ['M0','M2','M1','M3']:
        d=cur[cur.model.eq(model)]
        ax.plot(d.fpr,d.tpr,c=MC[model],lw=1.15 if model in ['M1','M3'] else .7,
                ls='-' if model=='M1' else ('--' if model=='M3' else ':'),label=f'{model}  {met.loc[model,"AUROC"]:.3f}')
    ax.plot([0,1],[0,1],ls=':',c='#9AA2A9',lw=.65)
    ax.set(xlim=(0,1),ylim=(0,1.01),xlabel='False-positive rate',ylabel='True-positive rate')
    ax.legend(loc='lower right',fontsize=7,title='Model / AUROC',title_fontsize=7)
    panel(ax,'B','Home OOF discrimination',x=-.18);polish(ax)
    ax=fig.add_subplot(gs[1,:])
    box=ax.get_position();ax.set_position([.335,box.y0,.425,box.height])
    labels=['Summary + V/J\n200 pipeline resamples','Whole-repertoire SCEPTR\n2,000 fixed-OOF resamples']
    for i,r in inc.iterrows():
        ax.plot([r.CI_low,r.CI_high],[i,i],c=MC['M3'],lw=1.3)
        ax.plot(r.delta_log_loss,i,'o',ms=5,c=MC['M3'],mfc=MC['M3'] if i==0 else 'white')
        ax.text(1.04,i,f'{r.delta_log_loss:+.3f}\n[{r.CI_low:+.3f}, {r.CI_high:+.3f}]',
                transform=ax.get_yaxis_transform(),fontsize=7,va='center',ha='left')
    ax.axvline(0,lw=.7,ls=(0,(3,3)),color='#89939B')
    ax.set_xlim(-.10,.065);ax.set_xticks([-.10,-.05,0,.05]);ax.set_ylim(1.6,-.6);ax.set_yticks([0,1],labels)
    ax.set_xlabel('Δ log loss (M1 − TCR model)')
    fig.text(.047,box.y1+.021,'C',fontsize=10,fontweight='bold')
    fig.text(.145,box.y1+.021,'Increment beyond the clinical and repertoire-load baseline',fontsize=8)
    polish(ax)
    ax=fig.add_subplot(gs[2,0])
    for model,c in [('M1_portable',MC['M1']),('M3_portable',MC['M3'])]:
        d=ext[ext.model.eq(model)];m=ext_met.loc[model]
        ax.plot(d.fpr,d.tpr,c=c,lw=1.15,ls='-' if model.startswith('M1') else '--',label=f'{model[:2]}  {m.AUROC:.3f}')
    ax.plot([0,1],[0,1],ls=':',c='#9AA2A9',lw=.65)
    ax.set(xlim=(0,1),ylim=(0,1.01),xlabel='False-positive rate',ylabel='True-positive rate')
    ax.legend(loc='lower right',title='Model / AUROC',title_fontsize=7)
    panel(ax,'D','External discrimination (n=163)',x=-.28);polish(ax)
    ax=fig.add_subplot(gs[2,1])
    for model,c,marker in [('M1_portable',MC['M1'],'o'),('M3_portable',MC['M3'],'s')]:
        d=cal[cal.model.eq(model)]
        ax.errorbar(d.mean_prediction,d.observed_fraction,yerr=[d.observed_fraction-d.CIlo,d.CIhi-d.observed_fraction],
                    c=c,marker=marker,ms=3,capsize=2,lw=.7,label=f'{model[:2]} (LL {ext_met.loc[model,"log_loss"]:.3f})')
    ax.plot([0,1],[0,1],ls=':',c='#9AA2A9',lw=.65)
    ax.set(xlim=(0,1),ylim=(0,1),xlabel='Mean predicted N+ probability',ylabel='Observed N+ fraction')
    ax.legend(loc='upper left',fontsize=7)
    panel(ax,'E','External calibration',x=-.18);polish(ax)
    stem=out/'Figure6_Nodal_evaluation'
    sources=[source/f for f in files]+[source/'external_metrics.tsv']
    receipt=export_figure(fig,stem,source_tables=sources)
    plt.close(fig)
    pd.DataFrame([row(stem,'6','Clinical discrimination of recorded nodal status',
       'Do repertoire features or sequence representations add stable nodal information?',sources)]).to_csv(out/'figure6_manifest_row.tsv',sep='\t',index=False)
    (out/'Figure6_caption.md').write_text(_private_literal('private_a63a034a50f81e72'))
    print(json.dumps({'figure':'6','out_of_canvas':receipt['out_of_canvas_text'],'minimum_font':receipt['font_points_min']}))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=ROOT);p.add_argument('--figure',choices=['1','6','inherit','all'],default='all');args=p.parse_args()
    ROOT=args.root;OLD=ROOT.parent/'revision_v5'
    if args.figure in ['inherit','all']:inherit()
    if args.figure in ['1','all']:figure1()
    if args.figure in ['6','all']:figure6()
