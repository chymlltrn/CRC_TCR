#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Figures for the merged-cohort pN battery (ov.style + Arial, PNG+PDF+CSV each)."""
from release_runtime import private_literal as _private_literal, release_path as _release_path
import os, json
import numpy as np, pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from PIL import Image

import omicverse as ov
ov.style(font_path='/usr/share/fonts/truetype/msttcorefonts/Arial.ttf')

OUT = _release_path('path_6fe8b0cdaf15a381')
FIG = f'{OUT}/figures'
os.makedirs(FIG, exist_ok=True)
PN_NEG, PN_POS = '#0072B2', '#D55E00'
OKABE = ['#0072B2', '#D55E00', '#009E73', '#CC79A7', '#E69F00', '#56B4E9', '#000000']

res = pd.read_csv(f'{OUT}/results/model_performance_full.csv')
oof = pd.read_csv(f'{OUT}/results/oof_predictions_all_arms.csv', index_col=0)
con = pd.read_csv(f'{OUT}/results/contrasts.csv')
tr  = pd.read_csv(f'{OUT}/results/transfer_frozen_definition.csv')
design = pd.read_csv(f'{OUT}/src/merged_design.csv', index_col=0)
y = design['y'].astype(int).values
SK = 'stratLabelBatchCohort'

MODELS_ORDER = ['LASSO_logistic','Ridge_logistic','ElasticNet_logistic','Logistic_unpenalized',
                'RandomForest','HistGradBoost','XGBoost','SVM_linear','SVM_RBF','kNN','GaussianNB','MLP']
ARMS_ORDER = ['CLUSTER3','REPO','CONCAT','CLUSTER3cov','REPOcov','CONCATcov']

def save(fig, name, src):
    for ext in ('png','pdf'):
        fig.savefig(f'{FIG}/{name}.{ext}', dpi=300, bbox_inches='tight')
    plt.close(fig)
    src.to_csv(f'{FIG}/{name}.csv', index=False)
    # QA
    p = f'{FIG}/{name}.png'
    arr = np.asarray(Image.open(p).convert('RGB'))
    nonwhite = float(np.mean(np.any(arr < 245, axis=2)))
    ncol = int(len(np.unique(arr.reshape(-1,3), axis=0)))
    assert os.path.getsize(p) > 5000 and arr.shape[0] > 200 and nonwhite > 0.02 and ncol >= 32,\
        f'QA FAIL {p}: nonwhite={nonwhite}, colors={ncol}'
    print(f'QA PASS {name}: nonwhite={nonwhite:.3f} colors={ncol} size={arr.shape}')

# ---------------- fig01: pooled OOF ROC ----------------
cand = {}
for a in ['CLUSTER3','REPO','CONCAT']:
    sub = res[(res.arm.str.startswith(tuple(MODELS_ORDER))) & (res.arm.str.contains(f'__{a}__{SK}$'))]
    cand[a] = sub.sort_values('auc_pooled_oof').iloc[-1].arm
cand['BASE_depthcohbatch'] = f'BASE_depthcohbatch__logL2__{SK}'
cand['BASE_Tstage'] = f'BASE_Tstage__logL2__{SK}'
fig, ax = plt.subplots(figsize=(4.6, 4.4))
rows = []
for (k, arm), col in zip(cand.items(), OKABE):
    s = oof[arm].values
    fpr, tpr, _ = roc_curve(y, s); a_ = auc(fpr, tpr)
    lab = k if not k in ('CLUSTER3','REPO','CONCAT') else f'{k} ({arm.split("__")[0]})'
    ax.plot(fpr, tpr, color=col, lw=1.8, label=f'{lab}: {a_:.3f}')
    rows.append(dict(curve=k, arm=arm, auc=a_))
ax.plot([0,1],[0,1], color='#999999', lw=0.8, ls='--')
ax.set_xlabel('False positive rate'); ax.set_ylabel('True positive rate')
ax.set_title(_private_literal('private_59cec1d6149c23a2'), fontsize=9)
ax.legend(fontsize=7.5, loc='lower right', frameon=False)
save(fig, 'fig01_roc_comparison', pd.DataFrame(rows))

# ---------------- fig02: 12 models x 6 arms AUC heatmap ----------------
M = np.full((len(MODELS_ORDER), len(ARMS_ORDER)), np.nan)
for i, m in enumerate(MODELS_ORDER):
    for j, a in enumerate(ARMS_ORDER):
        r = res[res.arm == f'{m}__{a}__{SK}']
        if len(r): M[i, j] = r.auc_pooled_oof.iloc[0]
base_cols = {}
for b in ['BASE_depthcohbatch','BASE_Tstage','BASE_depth']:
    r = res[res.arm == f'{b}__logL2__{SK}']
    base_cols[b] = r.auc_pooled_oof.iloc[0] if len(r) else np.nan
Mfull = np.column_stack([M, [base_cols['BASE_depthcohbatch']]*len(M), [base_cols['BASE_Tstage']]*len(M), [base_cols['BASE_depth']]*len(M)])
fig, ax = plt.subplots(figsize=(7.6, 5.2))
im = ax.imshow(Mfull, cmap='RdYlBu_r', vmin=0.45, vmax=0.75, aspect='auto')
xl = ARMS_ORDER + ['BASE depth+\ncohort+batch','BASE T-stage','BASE depth']
ax.set_xticks(range(len(xl))); ax.set_xticklabels(xl, rotation=38, ha='right', fontsize=8)
ax.set_yticks(range(len(MODELS_ORDER))); ax.set_yticklabels(MODELS_ORDER, fontsize=8)
for i in range(Mfull.shape[0]):
    for j in range(Mfull.shape[1]):
        if not np.isnan(Mfull[i,j]):
            ax.text(j, i, f'{Mfull[i,j]:.3f}', ha='center', va='center', fontsize=7.2,
                    color='black')
cb = plt.colorbar(im, ax=ax, fraction=0.03); cb.set_label('Pooled OOF AUC', fontsize=8)
ax.set_title('Merged cohort pooled OOF AUC — 12 methods × feature sets\n(label×batch×cohort stratified; baselines repeated as columns)', fontsize=9)
save(fig, 'fig02_auc_heatmap', pd.DataFrame(M, index=MODELS_ORDER, columns=ARMS_ORDER).reset_index().rename(columns={'index':'model'}))

# ---------------- fig03: forest CLUSTER3 vs REPO ----------------
sub = con[(con.contrast.isin(['CLUSTER3_vs_REPO','CLUSTER3cov_vs_REPOcov'])) & (con.scheme == SK)].copy()
sub['model'] = sub.arm.str.split('__').str[0]
fig, axes = plt.subplots(1, 2, figsize=(8.4, 4.6), sharey=True)
rows=[]
for axi, (tag, ttl) in zip(axes, [('CLUSTER3_vs_REPO','CLUSTER3 − REPO (raw)'),
                                  ('CLUSTER3cov_vs_REPOcov','CLUSTER3cov − REPOcov (+covariates)')]):
    s = sub[sub.contrast == tag].set_index('model').reindex(MODELS_ORDER)
    yy = np.arange(len(MODELS_ORDER))
    axi.axvline(0, color='#555555', lw=0.9, ls='--')
    axi.errorbar(s.delta_auc_pooled, yy, xerr=[s.delta_auc_pooled-s.delta_ci95_lo, s.delta_ci95_hi-s.delta_auc_pooled],
                 fmt='o', ms=4, lw=1.2, capsize=2.5,
                 color=PN_POS if s.delta_auc_pooled.median() > 0 else PN_NEG, ecolor='#50565C')
    for i, m in enumerate(MODELS_ORDER):
        if s.loc[m,'p_holm'] < 0.05:
            axi.text(s.loc[m,'delta_ci95_hi']+0.01, i, '*', va='center', fontsize=10)
    axi.set_yticks(yy); axi.set_yticklabels(MODELS_ORDER if axi is axes[0] else ['']*len(yy), fontsize=8)
    axi.set_xlabel('Δ AUC (pooled OOF, paired bootstrap 95% CI)', fontsize=8.5)
    axi.set_title(ttl + ('   * Holm p<0.05' ), fontsize=9)
    rows.append(s.reset_index().assign(contrast=tag))
fig.suptitle('3-cluster vs full-MiXCR repertoire features — merged cohort, 12 methods\npositive Δ = CLUSTER3 better', fontsize=10)
fig.tight_layout(rect=[0,0,1,0.93])
save(fig, 'fig03_forest_delta_auc', pd.concat(rows))

# ---------------- fig04: frozen-definition transfer ----------------
piv_a = tr[tr.arm.str.split('__').str[-1].isin(['CLUSTER3','REPO','CONCAT','BASE_depthcohbatch'])]
best = {}
for a in ['CLUSTER3','REPO','CONCAT','BASE_depthcohbatch']:
    for d in ['Home2Ext','Ext2Home']:
        s = piv_a[(piv_a.arm.str.endswith('__'+a)) & (piv_a.direction==d)]
        if len(s): best[(a,d)] = s.sort_values('auc_test').iloc[-1]
fig, ax = plt.subplots(figsize=(7.0, 4.2))
labels = ['CLUSTER3','REPO','CONCAT','BASE depth+cohort+batch']
x = np.arange(len(labels)); w = 0.36
rows=[]
for k, (d, off, col) in enumerate([('Home2Ext', -w/2, PN_POS), ('Ext2Home', w/2, PN_NEG)]):
    vals = [best[(a,d)]['auc_test'] if (a,d) in best else np.nan for a in labels]
    los  = [best[(a,d)]['auc_test']-best[(a,d)]['auc_ci95_lo'] if (a,d) in best else np.nan for a in labels]
    his  = [best[(a,d)]['auc_ci95_hi']-best[(a,d)]['auc_test'] if (a,d) in best else np.nan for a in labels]
    models = [best[(a,d)]['arm'].split('__')[0] if (a,d) in best else '' for a in labels]
    ax.bar(x+off, vals, w*0.92, color=col, alpha=0.85,
           yerr=[los,his], capsize=2.5, error_kw=dict(lw=1), label=f'{d} (best of 12 methods)')
    for xi, v, mm in zip(x+off, vals, models):
        ax.text(xi, 0.06, mm, rotation=90, fontsize=6, ha='center', color='white')
    for xi, v in zip(x+off, vals):
        ax.text(xi, v+0.03, f'{v:.3f}', ha='center', fontsize=7.5)
    rows.append(pd.DataFrame(dict(feature_set=labels, direction=d, auc=vals, model=models)))
ax.axhline(0.5, color='#999999', lw=0.8, ls='--')
ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=8.5)
ax.set_ylabel('Test-cohort AUC (95% bootstrap CI)'); ax.set_ylim(0, 0.9)
ax.set_title('Frozen-definition cross-cohort transfer\n(cluster definitions frozen; still selected on pooled labels — see report §leakage)', fontsize=9)
ax.legend(fontsize=8, frameon=False, loc='upper right')
save(fig, 'fig04_transfer', pd.concat(rows))
print('ALL FIGURES DONE')
