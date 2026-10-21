#!/usr/bin/env python3
"""Select six supporting evidence units; retain full source tables unchanged."""
from release_config import release_path, private_value
from pathlib import Path
import hashlib
import json
import textwrap
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm

ROOT = release_path('render_v20_root')
DATA = ROOT / 'manuscript/SourceData'
OUT = ROOT / 'analysis/supplement_v20'
FIG = ROOT / 'figures/supplement'
OUT.mkdir(parents=True, exist_ok=True)
FIG.mkdir(parents=True, exist_ok=True)
BLUE, ORANGE, PURPLE = '#0072B2', '#D55E00', '#7A52B3'
PROGRAMS = ['cytotoxic', 'exhaustion', 'naive_memory', 'interferon', 'residency', 'proliferation']
PL = ['Cytotoxicity', 'Exhaustion-associated', 'Naive/memory', 'Interferon response', 'Tissue residency', 'Proliferation']
plt.rcParams.update({'font.family': 'Arial', 'font.size': 8.2, 'axes.titlesize': 9,
    'axes.labelsize': 8.2, 'xtick.labelsize': 8, 'ytick.labelsize': 8,
    'legend.fontsize': 8, 'svg.fonttype': 'none', 'pdf.fonttype': 42,
    'axes.spines.top': False, 'axes.spines.right': False})
SOURCES, CHECKS, RECORDS = {}, [], {}

def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def read(p):
    f = DATA / p
    SOURCES[str(f.relative_to(ROOT))] = sha(f)
    return pd.read_csv(f, sep='\t')

def export(d, name):
    p = OUT / name
    d.to_csv(p, sep='\t', index=False, float_format='%.17g')
    return str(p.relative_to(ROOT))

def test(name, value):
    CHECKS.append({'check': name, 'PASS': bool(value)})
    if not value:
        raise AssertionError(name)

def panel(ax, letter, title):
    ax.set_title(title, loc='left', pad=10)
    ax.text(-.10, 1.08, letter, transform=ax.transAxes, fontweight='bold', fontsize=11, va='bottom')

def pstr(p):
    if not np.isfinite(p):
        return 'NE'
    return f'{p:.3f}' if p >= .001 else f'{p:.1e}'

def finish(fig, n, inputs, caption, components):
    fig.set_size_inches(180/25.4, fig.get_size_inches()[1])
    base = FIG / f'FigureS{n}_v20'
    for suffix in ['svg', 'pdf', 'png']:
        fig.savefig(base.with_suffix('.'+suffix), dpi=240, facecolor='white')
    plt.close(fig)
    cap = OUT / f'FigureS{n}_caption.md'
    cap.write_text(caption.strip()+'\n')
    RECORDS[str(n)] = {'ordinal': n, 'source_tables': inputs, 'caption': str(cap.relative_to(ROOT)),
        'baseline_components': components, 'assets': {ext: {'path': str(base.with_suffix('.'+ext).relative_to(ROOT)),
        'sha256': sha(base.with_suffix('.'+ext))} for ext in ['svg', 'pdf', 'png']},
        'caption_sha256': sha(cap), 'width_mm': 180, 'min_font_pt': 8}



def patient_program(ax,pts,eff,title,letter):
    panel(ax,letter,title)
    pts=pts[pts.eligible.eq(True)&pts.tissue.eq('Tumor')]
    eff=eff[eff.tissue.eq('Tumor')].set_index('program').reindex(PROGRAMS)
    for i,p in enumerate(PROGRAMS):
        a=pts[pts.program.eq(p)].sort_values('patient_id');r=eff.loc[p]
        test(f'{letter} {p} patient mean/source effect',np.isclose(a.difference.mean(),r.mean_difference,atol=1e-12))
        y=i+np.linspace(-.12,.12,len(a));ax.scatter(a.difference,y,c='.65',s=13,zorder=2)
        ax.errorbar(r.mean_difference,i,xerr=[[r.mean_difference-r.CI_low],[r.CI_high-r.mean_difference]],fmt='D',c=BLUE,mfc=BLUE if r.q<.05 else 'white',ms=5,capsize=2,zorder=4)
    ax.axvline(0,c='.6',lw=.8);ax.set(yticks=range(6),yticklabels=PL,xlabel='Patient program-score difference');ax.invert_yaxis()

def S3():
    e1=read('v19/plot_sources/supplement_v19/S5_D_standardized_programs_full_CI.tsv')
    p1=read('v19/plot_sources/supplement_v19/S5_D_patient_standardized_differences.tsv')
    e2=read('v19/plot_sources/supplement_v19/S9_CD_program_effects_full_CI.tsv')
    p2=read('v19/plot_sources/supplement_v19/S9_CD_patient_program_differences.tsv')
    lon=read('longitudinal/depth500_trajectories.tsv');lon=lon[lon.identity.eq('NT_VJ')]
    inputs=[export(e1,'S3A_standardized_program_estimates_full_CI.tsv'),export(p1,'S3A_patient_standardized_programs.tsv'),
        export(e2,'S3B_independent_matched_estimates_full_CI.tsv'),export(p2,'S3B_patient_independent_matched_programs.tsv'),export(lon,'S3CD_longitudinal_excess_sharing.tsv')]
    fig=plt.figure(figsize=(7.087,5.75));gs=fig.add_gridspec(2,2,left=.26,right=.98,bottom=.10,top=.92,hspace=.65,wspace=.72)
    patient_program(fig.add_subplot(gs[0,0]),p1,e1,'State-standardized own-blood\ndetected − undetected (tumor, n=11)','A')
    patient_program(fig.add_subplot(gs[0,1]),p2,e2,'Independent-bulk matched\n− unmatched (tumor, n=19)','B')
    for j,m in enumerate(['cell_fraction','clone_fraction']):
        ax=fig.add_subplot(gs[1,j]);
        for ti,(t,c) in enumerate([('Tumor',BLUE),('Normal',ORANGE)]):
            q=lon[(lon.metric==m)&(lon.Tissue==t)].pivot(index='patient_id',columns='visit',values='self_excess')
            test('S3 seven longitudinal pairs '+m+t,q.shape==(7,2))
            for _,r in q.iterrows():ax.plot([ti*3,ti*3+1],[100*r.Pre,100*r['first_complete_On']],'-o',c=c,lw=.8,ms=3,alpha=.75)
        ax.set(xticks=[0,1,3,4],xticklabels=['Baseline','Later','Baseline','Later'],ylabel='Excess autologous sharing (pp)');ax.axhline(0,c='.6',lw=.8)
        ax.text(.20,1.01,'Tumor',color=BLUE,transform=ax.transAxes,ha='center');ax.text(.80,1.01,'Adjacent-normal',color=ORANGE,transform=ax.transAxes,ha='center')
        panel(ax,'CD'[j],('Cell' if j==0 else 'Clonotype')+' weighting\nSeven paired trajectories')
    caption='''## Supplementary Figure 3. Phenotype specificity and longitudinal sharing.
**A**, Patient-level, common-state-standardized tumor CD8+ program differences between own-blood-detected and undetected clonotypes, using complete baseline blood and ≥1-cell detection. **B**, Tumor CD8+ program differences between independent-bulk-matched and unmatched clonotypes, with ≥2 AA/V/J counts in a bulk participant. Gray points represent patients; diamonds and bars show the mean and 95% patient-bootstrap CI. Filled diamonds denote q < 0.05 in each six-program family. Equal-clonotype weighting is used in both panels. Adjacent-normal estimates, state-level tests and expansion interactions remain in Source Data: tissue. **C–D**, Baseline-to-later excess autologous sharing with cell and clonotype weighting in seven patients with ≥500 qualifying blood cells at both visits. Blood is sampled to 500 cells in 200 draws; NT/V/J identity is required. Later visits and treatment timing differ, so trajectories are descriptive without a longitudinal P value. The plotted quantity is excess sharing, whereas the earlier v19 trajectories displayed direct self-coverage; both underlying columns remain in Source Data: longitudinal.'''
    finish(fig,3,inputs,caption,[{'figure':5,'panels':['D (tumor)']},{'figure':9,'panels':['C']},{'figure':8,'panels':['C','D'],'new_display':'self_excess from the same depth500 trajectory table'}])

def sensitivity_heatmap(ax,d,rows,scale,title,letter):
    combos=[(e,w) for e in ['combined_mean','study_mean','external_mean','positive_carrier_mean'] for w in ['cell_equal','clone_equal']]
    mat=np.full((len(rows),8),np.nan);q=np.full_like(mat,np.nan)
    for i,(t,p,l) in enumerate(rows):
        for j,(e,w) in enumerate(combos):
            r=d[(d.Tissue==t)&(d.endpoint==p)&(d.exposure==e)&(d.weighting==w)].iloc[0]
            mat[i,j]=r.effect;q[i,j]=r.q
    im=ax.imshow(mat,aspect='auto',cmap='RdBu_r',vmin=-scale,vmax=scale)
    for i in range(len(rows)):
        for j in range(8):
            ax.text(j,i,('*' if q[i,j]<.05 else ''),ha='center',va='center',fontweight='bold',fontsize=9)
    ax.set(yticks=range(len(rows)),yticklabels=[r[2] for r in rows],xticks=range(8),xticklabels=['All\ncell','All\nclone','Study\ncell','Study\nclone','External\ncell','External\nclone','Carriers\ncell','Carriers\nclone'])
    ax.tick_params(length=0);panel(ax,letter,title);return im



def ref_matrix(ax,d,rows,title,letter):
    z=np.full((len(rows),2),np.nan)
    for i,e in enumerate(rows):
        for j,c in enumerate(['home','external']):
            r=d[(d.endpoint==e)&(d.cohort==c)].iloc[0];z[i,j]=np.log2(r.odds_ratio) if np.isfinite(r.odds_ratio) else np.nan
    ax.imshow(z,cmap='RdBu_r',vmin=-1,vmax=1,aspect='auto')
    for i,e in enumerate(rows):
        for j,c in enumerate(['home','external']):
            r=d[(d.endpoint==e)&(d.cohort==c)].iloc[0]
            txt=f'{r.odds_ratio:.2f}\nP {pstr(r.p)}' if np.isfinite(r.odds_ratio) else 'NE'
            ax.text(j,i,txt,ha='center',va='center',fontsize=8)
    ax.set(xticks=[0,1],xticklabels=['Study','External'],yticks=range(len(rows)),yticklabels=[e.replace('_',' ').replace('unspecified pathogen','Unspecified') for e in rows]);ax.tick_params(length=0);panel(ax,letter,title)


if __name__=='__main__':
    for fun in [S3]:fun()  # v29 supplementary Figure 2; historical function S3
    (OUT/'plot_receipt.json').write_text(json.dumps({'status':'PASS','plotting_code':{'path':str(Path(__file__)),'sha256':sha(__file__)},'source_inputs':SOURCES,'figures':RECORDS,'checks':CHECKS},indent=2)+'\n')
