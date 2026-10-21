#!/usr/bin/env python3
"""Select six supporting evidence units; retain full source tables unchanged."""
from release_runtime import private_literal as _private_literal, release_path as _release_path
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

ROOT = Path(_release_path('path_052a3da416b1c4ab'))
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

def S1():
    d = read('nodal/expanded12_adjusted.tsv')
    d = d[d.analysis.eq('known_recorded_N')].copy()
    test('S1 expanded adjusted 24 estimates', len(d)==24)
    order = ['shannon_entropy','normalized_shannon','simpson_diversity','top1_fraction','top10_fraction',
        'weighted_mean_cdr3_length','unique_aa','total_reads','input_reads','alignment_fraction',
        'trb_alignment_fraction','productive_input_fraction']
    labels = ['Shannon entropy','Normalized diversity','Simpson diversity','Largest sequence','Top-10 abundance',
        'CDR3β length','AA sequence richness','Productive/QC reads','Input reads','Alignment fraction','Aligned TRB fraction','QC/input read ratio']
    genes = read('nodal/Supplement_VJ_all78.tsv')
    test('S1 complete VJ family', len(genes)==78 and genes.family_size.eq(78).all())
    ev = read('nodal/unified_nodal_evidence.tsv')
    gain = ev[((ev.protocol==_private_literal('private_56f70c522e8a03fa'))&(ev.model=='M3'))|
        (ev.protocol=='current_ridge_full_repertoire_SCEPTR')].copy()
    ext = ev[ev.protocol.eq('current_shared_variable_frozen_transfer')].copy()
    test('S1 gain/external rows', len(gain)==2 and len(ext)==2)
    inputs=[export(d,'S1A_adjusted_12_metrics_full_CI.tsv'),export(genes,'S1B_complete_VJ_family.tsv'),
        export(gain,'S1C_global_increment_CI.tsv'),export(ext,'S1D_portable_global_performance.tsv')]
    fig=plt.figure(figsize=(7.087,6.90));gs=fig.add_gridspec(2,2,height_ratios=[1.9,1],hspace=.52,wspace=.78,
        left=.24,right=.97,bottom=.11,top=.93)
    ax=fig.add_subplot(gs[0,0]);v=d.pivot(index='feature',columns='cohort',values='OR').reindex(order)[['home','external']]
    ax.imshow(np.log2(v),cmap='RdBu_r',vmin=-1.5,vmax=1.5,aspect='auto')
    for i,f in enumerate(order):
        for j,c in enumerate(['home','external']):
            r=d[(d.feature==f)&(d.cohort==c)].iloc[0]
            ax.text(j,i,f'{r.OR:.2f}\nP {pstr(r.P)}',ha='center',va='center',fontsize=8)
    ax.set(xticks=[0,1],xticklabels=['Study','External'],yticks=range(12),yticklabels=labels)
    ax.tick_params(length=0);panel(ax,'A','Adjusted pN+ versus pN0\nOR per standard deviation')
    ax=fig.add_subplot(gs[0,1]);
    for family,c in [('TRBV',BLUE),('TRBJ',ORANGE)]:
        g=genes[genes.gene_family==family];ax.scatter(g.log_OR,-np.log10(g.P),c=c,label=family,s=20,alpha=.8)
    ax.axvline(0,color='.7',lw=.7);ax.axhline(-np.log10(.05),ls='--',lw=.8,c='.5')
    ax.set(xlabel='Adjusted log OR per SD',ylabel='−log10(model P)');ax.legend(frameon=False,loc='upper left')
    panel(ax,'B','All 78 V/J genes\nNominal model P values')
    ax=fig.add_subplot(gs[1,0]);
    for i,(_,r) in enumerate(gain.iterrows()):
        c=BLUE if i==0 else PURPLE
        ax.errorbar(r.delta_log_loss,i,xerr=[[r.delta_log_loss-r.CI_low],[r.CI_high-r.delta_log_loss]],fmt='o',c=c,capsize=3)
    ax.axvline(0,color='.6',lw=.8);ax.set(yticks=[0,1],yticklabels=['Global features','SCEPTR'],ylim=(-.5,1.5),xlabel='Log-loss gain over baseline')
    panel(ax,'C','Incremental performance\nPatient-level 95% intervals')
    ax=fig.add_subplot(gs[1,1]);
    for i,(_,r) in enumerate(ext.iterrows()):
        ax.scatter(r.AUROC,i,c=[BLUE,PURPLE][i],s=45)
        ax.text(r.AUROC+.015,i,f'{r.AUROC:.3f}',va='center')
    ax.axvline(.5,color='.6',ls='--',lw=.8);ax.set(yticks=[0,1],yticklabels=['Baseline','Combined'],ylim=(-.5,1.5),xlim=(.35,.68),xlabel='External AUROC (point estimates)')
    panel(ax,'D','Portable global models\n163 pN-labeled external patients')
    caption='''## Supplementary Figure 1. Repertoire associations and global-model performance.
**A**, Adjusted pN+ versus pN0 odds ratios (ORs) per standard deviation for all 12 repertoire/technical metrics in 126 study and 163 external patients. Cell colors encode log2 OR; displayed P values are model-specific, covariate-adjusted values. Full 95% confidence intervals (CIs), transformations and covariates are in Source Data. **B**, All 78 V/J-gene estimates from the study-cohort model, which additionally adjusts for sequence richness; the dashed line marks nominal P = 0.05. No estimate passed its 78-test correction family. **C**, Baseline-minus-model log-loss gains for global repertoire features and whole-repertoire SCEPTR. Intervals are from 200 full-workflow patient bootstraps for global features and fixed-prediction patient bootstraps for SCEPTR; their different interval targets are retained. Positive gains favor the added model. **D**, External AUROC point estimates for the portable baseline and combined global model. These cross-source rechecks do not provide independent outcome-blind validation. Detailed individual summaries, three-summary models, calibration and representation-specific coverage remain in Source Data: nodal.'''
    finish(fig,1,inputs,caption,[{'figure':1,'panels':['B']},{'figure':2,'panels':['B','D']},{'figure':3,'panels':['C']}])

def S2():
    d=read('sharing/statistics.tsv');s=d[d.contrast.eq('Tumor_minus_Normal_self_excess')].copy()
    corr=read('matching/sharing_program_continuous_tests.tsv');corr=corr[corr.family.eq('primary_NTVJ_cells')]
    test('S2 all nine definitions/two weights',len(s)==18 and len(corr)==12)
    inputs=[export(s,'S2AB_sharing_definition_contrasts_CI.tsv'),export(corr,'S2C_continuous_program_correlations_CI.tsv')]
    labels=['NT/V/J, blood 500','AA/V/J, blood 500','AA only, blood 500','Paired αβ, blood 500','NT/V/J, blood 50','NT/V/J, all 500','Paired αβ, all 500','NT/V/J, blood ≥2','NT/V/J, complete']
    order=d.analysis_id.drop_duplicates().tolist()
    fig=plt.figure(figsize=(7.087,5.55));gs=fig.add_gridspec(2,2,left=.25,right=.96,bottom=.16,top=.92,hspace=.63,wspace=.65)
    for j,(metric,title) in enumerate([('cell_fraction','Cell-weighted sharing'),('clone_fraction','Clonotype-weighted sharing')]):
        ax=fig.add_subplot(gs[0,j]);g=s[s.metric==metric].set_index('analysis_id').reindex(order)
        for i,(_,r) in enumerate(g.iterrows()):
            c=BLUE if i==0 else '.45';ax.errorbar(100*r.mean_effect,i,xerr=[[100*(r.mean_effect-r.ci_low)],[100*(r.ci_high-r.mean_effect)]],fmt='o',c=c,ms=4,capsize=2)
        ax.axvline(0,c='.6',lw=.8);ax.invert_yaxis();ax.set(yticks=range(9),yticklabels=labels if j==0 else [],xlabel='Tumor − adjacent-normal (pp)')
        panel(ax,'AB'[j],title)
    ax=fig.add_subplot(gs[1,:]);
    for j,(t,c) in enumerate([('Tumor',BLUE),('Normal',ORANGE)]):
        g=corr[corr.Tissue.eq(t)].set_index('program').reindex(PROGRAMS)
        for i,(_,r) in enumerate(g.iterrows()):
            ax.errorbar(r.rho,i+(j-.5)*.2,xerr=[[r.rho-r.CI_low],[r.CI_high-r.rho]],fmt='o',c=c,ms=4,capsize=2,label=('Tumor' if t=='Tumor' else 'Adjacent-normal') if i==0 else None)
    ax.axvline(0,c='.6',lw=.8);ax.set(yticks=range(6),yticklabels=PL,xlim=(-1,1),xlabel='Spearman correlation with mean tissue CD8+ program score');ax.invert_yaxis();ax.legend(frameon=False,loc='upper center',bbox_to_anchor=(.5,-.30),ncol=2)
    panel(ax,'C','Across-patient sharing–program correlations (12-test family)')
    caption='''## Supplementary Figure 2. Robustness and scale of autologous blood–tissue sharing.
**A–B**, Tumor-minus-adjacent-normal differences in excess autologous sharing under nine molecular/depth definitions, with cell and nucleotide-defined clonotype weighting. Excess sharing is own-blood detection minus the mean detection with other eligible donors. Points and bars show patient means and 95% bootstrap CIs, in percentage points (pp). The primary analysis samples 500 blood cells and includes 18 patients; paired αβ sampling of all compartments to 500 retains nine. The first eight analyses retain their six-test families; complete-repertoire estimates are descriptive. **C**, Primary across-patient correlations between tissue-cell sharing and clonotype-weighted CD8+ program means in 18 blood-depth-eligible patients. Points and bars show Spearman ρ and patient-bootstrap 95% CIs. The 12 tests use 49,999 patient permutations with add-one P values. These correlations address variation between patients and differ from shared-versus-unshared comparisons within patients. Full tissue-specific excess-sharing contrasts, all test values and four correlation sensitivities remain in Source Data: sharing and matching.'''
    finish(fig,2,inputs,caption,[{'figure':4,'panels':['E','F']},{'figure':6,'panels':['A','B']}])

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

def S4():
    d=read('bulk_abundance/Figure15_sensitivity_plot_source.tsv')
    p=d[d.endpoint_type.eq('program')];s=d[d.endpoint_type.eq('state')&d.endpoint.isin(['c03_CD4_Tn_NR4A2','c20_CD8_Tem_GZMK'])]
    test('S4 complete selected program/state contrasts',len(p)==96 and len(s)==32)
    inputs=[export(p,'S4A_all_program_sensitivities_CI.tsv'),export(s,'S4B_two_state_sensitivities_CI.tsv')]
    fig=plt.figure(figsize=(7.087,5.15));gs=fig.add_gridspec(2,1,height_ratios=[2.5,1],left=.27,right=.89,bottom=.14,top=.92,hspace=.50)
    ax=fig.add_subplot(gs[0]);rows=[(t,p,('T: ' if t=='Tumor' else 'N: ')+l) for t in ['Tumor','Normal'] for p,l in zip(PROGRAMS,PL)]
    im=sensitivity_heatmap(ax,p,rows,.4,'All CD8+ programs: high − low abundance','A');fig.colorbar(im,ax=ax,pad=.03,fraction=.045,label='Program-score difference')
    ax=fig.add_subplot(gs[1]);rows=[(t,p,('T: ' if t=='Tumor' else 'N: ')+l) for t in ['Tumor','Normal'] for p,l in [('c03_CD4_Tn_NR4A2','c03 CD4 Tn NR4A2'),('c20_CD8_Tem_GZMK','c20 CD8 Tem GZMK')]]
    im=sensitivity_heatmap(ax,s,rows,10,'Two primary states: high − low abundance','B');fig.colorbar(im,ax=ax,pad=.03,fraction=.045,label='State difference (pp)')
    caption='''## Supplementary Figure 4. Sensitivity of tissue phenotypes to peripheral-blood bulk TCR abundance.
**A**, High-minus-low program-score contrasts for all six conventional CD8+ programs in tumor (T) and adjacent-normal (N) tissue. **B**, Corresponding contrasts for the two states supported in the primary analysis: c03 CD4 Tn NR4A2 and c20 CD8 Tem GZMK. Columns compare population means across all 339 bulk participants, 137 study participants, 202 external participants, or positive carriers, with cell/equal-clonotype weighting. Each exposure uses its own fixed positive-identity tertiles. Stars mark q < 0.05 within the applicable 12-program or 64-state family across tissues; colors encode patient mean differences. Tests are two-sided paired Wilcoxon comparisons, with ≥8 eligible paired-bin patients. Program and state color scales differ. Primary directions persist but corrected support varies; three-bin Friedman results address omnibus differences without establishing monotonicity. Full 32-state sensitivities, CIs, eligibility and omnibus results remain in Source Data: bulk_abundance.'''
    finish(fig,4,inputs,caption,[{'figure':15,'panels':['A','B (c03/c20)','C (c03/c20)']}])

def S5():
    dep=read('candidates/clinical_models/depth/counterfactual_summary.tsv');dep=dep[dep.scope.eq('Home_all_common_support')&dep.threshold.eq(2)]
    assoc=read('candidates/clinical_models/N_association_primary.tsv');lobo=read('candidates/clinical_models/models/lobo_metrics.tsv');ext=read('candidates/clinical_models/models/external_metrics.tsv')
    test('S5 depth definitions',len(dep)==4);test('S5 batch and portable model counts',len(lobo)==9 and len(ext)==3)
    inputs=[export(dep,'S5A_depth_counterfactual_full_CI.tsv'),export(assoc,'S5B_candidate_nodal_associations_full_CI.tsv'),
        export(lobo,'S5C_batch_held_out_performance.tsv'),export(ext,'S5D_portable_candidate_performance.tsv')]
    fig=plt.figure(figsize=(7.087,5.4));gs=fig.add_gridspec(2,2,left=.19,right=.97,bottom=.12,top=.92,hspace=.65,wspace=.70)
    ax=fig.add_subplot(gs[0,0]);order=['Q1_AA','Q1_AA_VJ','Q1_NT','Q2_AA'];dep=dep.set_index('definition').reindex(order)
    for i,(_,r) in enumerate(dep.iterrows()):
        ax.errorbar(r.expected_External_carriers,i,xerr=[[r.expected_External_carriers-r.expected_External_carriers_low],[r.expected_External_carriers_high-r.expected_External_carriers]],fmt='o',c=BLUE,capsize=2,label='Expected' if i==0 else None)
        ax.scatter(r.observed_External_carriers,i,marker='D',s=30,color='black',label='Observed' if i==0 else None)
    ax.set(yticks=range(4),yticklabels=['Q1 AA','Q1 AA/V/J','Q1 exact NT','Q2 AA'],xlabel='External detections (of 202)',xlim=(-5,140));ax.invert_yaxis();ax.legend(frameon=False,loc='upper right');panel(ax,'A','Depth expectation versus observation\n≥2 assigned counts')
    ax=fig.add_subplot(gs[0,1]);
    for i,(_,r) in enumerate(assoc.iterrows()):
        for j,(pre,c) in enumerate([('unadjusted',BLUE),('adjusted',ORANGE)]):
            val=r[pre+'_OR'];lo=r[pre+'_OR_low'];hi=r[pre+'_OR_high'];ax.errorbar(val,i+(j-.5)*.16,xerr=[[val-lo],[hi-val]],fmt='o',c=c,capsize=2,label='Crude' if j==0 and i==0 else 'Adjusted' if i==0 else None)
    ax.set_xscale('log');ax.axvline(1,c='.6',lw=.8);ax.set(yticks=[0,1],yticklabels=['Q1','Q2'],ylim=(-.5,1.5),xlabel='pN+ versus pN0 OR (95% CI)');panel(ax,'B','Crude / adjusted nodal OR\nBlue / orange, n=126')
    ax=fig.add_subplot(gs[1,0]);names=['M1','M1_Q1','M1_Q2'];labels=['Baseline','+ Q1','+ Q2'];batches=['H2498B','H2498C','H2498D']
    for j,(m,c) in enumerate(zip(names,[BLUE,ORANGE,PURPLE])):
        g=lobo[lobo.model==m].set_index('held_out_batch').reindex(batches);ax.plot(range(3),g.AUROC,'-o',c=c,label=labels[j],ms=4,lw=.8)
    ax.axhline(.5,ls='--',c='.6',lw=.8);ax.set(xticks=range(3),xticklabels=batches,ylim=(.25,.8),ylabel='Held-out AUROC');ax.legend(frameon=False,ncol=1,loc='upper left');panel(ax,'C','Acquisition-batch holdout')
    ax=fig.add_subplot(gs[1,1]);
    for i,(_,r) in enumerate(ext.iterrows()):
        ax.scatter(r.AUROC,i,c=[BLUE,ORANGE,PURPLE][i],s=35);ax.text(r.AUROC+.018,i,f'{r.AUROC:.3f}',va='center')
    ax.axvline(.5,ls='--',c='.6',lw=.8);ax.set(yticks=range(3),yticklabels=labels,ylim=(-.5,2.5),xlim=(.35,.68),xlabel='External AUROC (point estimates)');panel(ax,'D','Portable candidate models\n163 pN-labeled external patients')
    caption=_private_literal('private_4b65269d3e4095d6')
    finish(fig,5,inputs,caption,[{'figure':12,'panels':['C']},{'figure':13,'panels':['E']},{'figure':14,'panels':['E','F']}])

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

def S6():
    d=read('v19/plot_sources/supplement_v19/S11_AB_adjusted_OR_full_CI.tsv');sec=read('v19/plot_sources/supplement_v19/S11_CDE_exact_model_P_and_BH.tsv')
    pts=[]
    for e in ['any_reference','viral','human_self']:
        pts.append(read(f'v19/plot_sources/supplement_v19/S11_{e}_patient_richness.tsv'))
    pts=pd.concat(pts,ignore_index=True)
    test('S6 primary endpoints and patients',len(d)==32 and len(pts)==378)
    inputs=[export(d,'S6AB_primary_antigen_models_full_CI.tsv'),export(sec,'S6C_secondary_models_exact_P_BH.tsv'),export(pts,'S6C_secondary_patient_distributions.tsv')]
    categories=d[d.family.eq('category')].endpoint.drop_duplicates().tolist();pathogens=d[d.family.eq('pathogen')].endpoint.drop_duplicates().tolist()
    fig=plt.figure(figsize=(7.087,6.05));gs=fig.add_gridspec(2,2,height_ratios=[2.1,1],left=.20,right=.98,bottom=.12,top=.92,hspace=.55,wspace=.75)
    ref_matrix(fig.add_subplot(gs[0,0]),d,categories,'Antigen categories\nRead-abundance OR, model P','A')
    ref_matrix(fig.add_subplot(gs[0,1]),d,pathogens,'Pathogen groups\nRead-abundance OR, model P','B')
    sub=gs[1,:].subgridspec(1,3,wspace=.60)
    for j,(e,title) in enumerate([('any_reference','Any reference'),('viral','Viral'),('human_self','Human/self')]):
        ax=fig.add_subplot(sub[0,j]);q=pts[pts.endpoint.eq(e)]
        for n,c in [(0,BLUE),(1,ORANGE)]:
            a=q[q.n_binary.eq(n)].sort_values('patient_key');x=np.random.default_rng(171+j*3+n).uniform(n-.18,n+.18,len(a));y=100*a.richness_fraction
            ax.scatter(x,y,s=9,c=c,alpha=.45,rasterized=False);bp=ax.boxplot([y],positions=[n],widths=.4,showfliers=False,patch_artist=True,medianprops={'color':'black','linewidth':1.2});bp['boxes'][0].set(facecolor='none',edgecolor=c)
        r=sec[sec.endpoint.eq(e)].iloc[0];ax.set(xticks=[0,1],xticklabels=['pN0','pN+'],ylabel='Matched identities (%)' if j==0 else '')
        panel(ax,'C' if j==0 else '',f'{title}\nModel P={pstr(r.p)}; q={r.q:.3f}')
    caption='''## Supplementary Figure 6. Antigen-reference context of lymph node associations.
**A–B**, Covariate-adjusted pN+ versus pN0 ORs per standard deviation of transformed read abundance for ten antigen categories and six pathogen groups, in the study and external cohorts. Colors encode log2 OR; cells report model P values. NE denotes a nonestimable model and is not interpreted as a negative result. Presence/read-abundance correction families comprise 20 category or 12 pathogen tests per cohort; no estimable primary test passed correction. Full presence results, 95% CIs and internal correction placeholders remain in Source Data: antigen. **C**, Individual study-patient matched-identity fractions for any, viral and human/self references (70 pN0, 56 pN+). Boxes show medians and interquartile ranges; whiskers extend to 1.5 times that range. Displayed P values derive from covariate-adjusted models, not the plotted group distributions; q values belong to the separate ten-test secondary richness family. These three distinct P values share a valid BH plateau. Their associations did not reproduce in the external or higher-confidence-reference analyses. Matching requires exact CDR3β AA/V/J identity and does not measure antigen recognition. Candidate-specific organism, peptide, sequence and checkpoint rankings supporting main Figure 5 remain in full in Source Data: antigen.'''
    finish(fig,6,inputs,caption,[{'figure':11,'panels':['A','B','C','D','E']}])

if __name__=='__main__':
    for fun in [S1,S2,S3,S4,S5,S6]:fun()
    (OUT/'plot_receipt.json').write_text(json.dumps({'status':'PASS','plotting_code':{'path':str(Path(__file__).relative_to(ROOT)),'sha256':sha(__file__)},'source_inputs':SOURCES,'figures':RECORDS,'checks':CHECKS},indent=2)+'\n')
