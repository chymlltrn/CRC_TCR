#!/usr/bin/env python3
"""Render candidate F2 from audited v6 source tables, never raw clonotype scans."""
from release_runtime import private_literal as _private_literal, release_path as _release_path
from pathlib import Path
import argparse,sys,json,hashlib,shutil
import numpy as np,pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from statsmodels.stats.proportion import proportion_confint

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main(root):
    sys.path.insert(0,str(root/'code'))
    from paper_style import new_figure,COLORS,polish,export_figure
    out=root/'candidates';figdir=out/'figures';figdir.mkdir(exist_ok=True)
    receipt=json.loads((out/'audit_receipt.json').read_text());assert receipt['status']=='passed'
    for r in pd.read_csv(out/'output_manifest.tsv',sep='\t').to_dict('records'):
        assert sha(r['path'])==r['sha256'],r['path']
    table=pd.read_csv(out/'Figure2_source_carriage.tsv',sep='\t')
    sens=pd.read_csv(out/_private_literal('private_a70effc21f3daa7e'),sep='\t')
    rec=pd.read_csv(out/'Le2017_within_reference_recurrence.tsv',sep='\t')
    source_dir=figdir/'source';source_dir.mkdir(exist_ok=True)
    ci=table[table.unit_label.eq('patients')].copy()
    ci[['wilson95_low','wilson95_high']]=[proportion_confint(int(k),int(n),method='wilson') for k,n in zip(ci.n_carriers,ci.n_units)]
    ci['interval_unit']='independent selected blood patient'
    ci.to_csv(source_dir/'Figure2_patient_Wilson95.tsv',sep='\t',index=False)
    fig=new_figure(height_mm=210,width_mm=180)
    def title(letter,text,y):
        fig.text(.04,y,letter,fontsize=10,fontweight='bold',ha='left',va='bottom')
        fig.text(.085,y,text,fontsize=8.5,ha='left',va='bottom')
    title('A','Source-stratified complete-AA occurrence (count ≥2)',.956)
    ax=fig.add_axes([.34,.607,.61,.315])
    labels=['Home CRC','External CRC','139 Healthy blood','139 CRC PBMC','Le2017 serial PBMC','Le2017 archival tumor','Le2017 peptide/control culture']
    ylabels=['Home CRC\n137 patients','External CRC\n202 patients','immunoSEQ139 · healthy\n87 libraries','immunoSEQ139 · CRC PBMC\n20 libraries','Le2017 serial PBMC\n24 libraries; 3 participants','Le2017 archival tumor\n3 libraries; 3 participants','Le2017 cultures\n8 libraries; 1 participant']
    for i,label in enumerate(labels):
        y=6-i
        for candidate,off,marker in [('Q1',.16,'o'),('Q2',-.16,'D')]:
            r=table[(table.source_label==label)&(table.candidate==candidate)].iloc[0]
            if r.unit_label=='patients':
                cr=ci[(ci.source_label==label)&(ci.candidate==candidate)].iloc[0]
                ax.errorbar(r.carrier_fraction*100,y+off,xerr=np.maximum(0,np.array([[r.carrier_fraction-cr.wilson95_low],[cr.wilson95_high-r.carrier_fraction]]))*100,fmt='none',ecolor=COLORS[candidate.lower()],elinewidth=.8,capsize=1.5,zorder=2)
            ax.scatter(r.carrier_fraction*100,y+off,s=25,color=COLORS[candidate.lower()],marker=marker,zorder=3,label=candidate if i==0 else None)
            ax.text(94,y+off,f'{int(r.n_carriers)}/{int(r.n_units)}',fontsize=7.5,color=COLORS[candidate.lower()],va='center')
    for y in [4.5,2.5]:ax.axhline(y,color=COLORS['grid'],lw=.65,zorder=0)
    ax.set_xlim(-2,112);ax.set_ylim(-.5,6.6);ax.set_xticks([0,25,50,75]);ax.set_yticks(range(7),list(reversed(ylabels)))
    ax.tick_params(axis='y',length=0,labelsize=7.5)
    ax.set_xlabel('Source-specific detection fraction (%)',labelpad=4)
    ax.text(94,6.65,'Count / units',fontsize=7.2,va='bottom')
    ax.legend(loc='upper left',bbox_to_anchor=(.01,1.115),ncol=2,columnspacing=1.4,handletextpad=.4)
    polish(ax,grid_axis='x');ax.spines['left'].set_visible(False)
    title('B','Q1 resolution and within-patient read-threshold sensitivity',.515)
    bx=fig.add_axes([.12,.345,.81,.135])
    b=sens[(sens.cohort=='home')&(sens.scope=='full_selected')]
    defs=[('Q1_AA','Complete AA',COLORS['q1'],'o','-'),
          ('Q1_AA_VJ','AA + V5-5/J1-4',COLORS['ink'],'s','--'),
          ('Q1_NT','Nominated NT','#D55E00','^',':')]
    for d,label,color,marker,line in defs:
        g=b[b.definition==d].sort_values('threshold');x=np.arange(len(g))
        # Different open marker/line patterns preserve exact overlapping data coordinates.
        bx.plot(x,g.n_carriers,color=color,marker=marker,linestyle=line,ms=4,mfc='white' if d!='Q1_AA' else color,lw=1.1,label=label,zorder=4 if d=='Q1_NT' else 3)
        if d=='Q1_AA':
            for xx,k in zip(x,g.n_carriers):bx.text(xx,k+4,str(int(k)),ha='center',fontsize=7,color=color)
    bx.set_xticks(range(5),['≥1','≥2','≥5','≥10','≥20']);bx.set_xlim(-.15,4.18);bx.set_ylim(-2,74)
    bx.set_yticks([0,20,40,60]);bx.set_ylabel('Home carriers / 137')
    bx.set_xlabel('Summed definition-matching MiXCR reads per patient',labelpad=3)
    bx.legend(loc='upper right',ncol=1,fontsize=7,handlelength=2.2,labelspacing=.35)
    polish(bx,grid_axis='y')
    fig.text(.12,.294,'At ≥2: AA 53/137; AA + V/J 24/137; nominated NT 24/137.',fontsize=7.5)
    title('C','Q1 recurrence in archived tumor and serial blood from one participant',.255)
    ca=fig.add_axes([.12,.085,.13,.125]);cx=fig.add_axes([.34,.085,.59,.125],sharey=ca)
    tumor=rec[rec.source_sample_name=='15_Primary_Tumor'].iloc[0]
    ca.scatter(0,tumor.Q1_AA_proxy_QC_count,s=36,marker='s',color=COLORS['gray'])
    ca.text(0,tumor.Q1_AA_proxy_QC_count*1.6,str(int(tumor.Q1_AA_proxy_QC_count)),ha='center',fontsize=7.5)
    ca.set_xlim(-.55,.55);ca.set_xticks([0],['Archival\nprimary tumor']);ca.set_ylabel('Source cloneCount')
    names=['15_Pre_treatment_PBMC','15_2weeks','15_4weeks','15_12weeks','15_44weeks','15_52weeks','15_72weeks']
    rr=rec.set_index('source_sample_name').loc[names]
    cx.scatter(range(7),rr.Q1_AA_proxy_QC_count,s=27,marker='o',color=COLORS['q1'])
    for i,k in enumerate(rr.Q1_AA_proxy_QC_count):cx.text(i,k*1.55,str(int(k)),ha='center',fontsize=7.2,color=COLORS['q1'])
    cx.set_xticks(range(7),['Pre','2','4','12','44','52','72']);cx.set_xlim(-.4,6.4)
    cx.set_xlabel('Blood visit (weeks after pembrolizumab)',labelpad=3)
    ca.set_yscale('log');ca.set_ylim(10,12500);ca.set_yticks([10,100,1000,10000],['10','100','1,000','10,000'])
    cx.tick_params(axis='y',which='both',left=False,labelleft=False)
    for a in [ca,cx]:polish(a,grid_axis='y')
    cx.spines['left'].set_visible(False)
    fig.text(.12,.024,'Participant 15: shared V5-4/J1-4 NT in tumor and blood; different from Home nominated NT.',fontsize=7.2)
    export=export_figure(fig,figdir/'Figure2',[out/'Figure2_source_carriage.tsv',out/_private_literal('private_a70effc21f3daa7e'),out/'Le2017_within_reference_recurrence.tsv',source_dir/'Figure2_patient_Wilson95.tsv'])
    plt.close(fig)
    assert export['width_mm']==180 and export['height_mm']<=246
    assert export['font_points_min']>=7
    (figdir/'captions.md').write_text(_private_literal('private_cd5fefd6140659d7'))
    manifest=[]
    for ext in ['pdf','svg','png','preview.png','grayscale.png','export.json']:
        p=figdir/('Figure2.'+ext);manifest.append(dict(figure_id='Figure2',logical_id='candidate_evidence',path=str(p),format=ext,bytes=p.stat().st_size,sha256=sha(p),source_tables=_private_literal('private_f6ef0975ee558a9e'),visual_status='pending_actual_review'))
    pd.DataFrame(manifest).to_csv(figdir/'figure_manifest.tsv',sep='\t',index=False)
    print(json.dumps(export,indent=2))
if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('--root',type=Path,required=True)
    main(a.parse_args().root.resolve())
