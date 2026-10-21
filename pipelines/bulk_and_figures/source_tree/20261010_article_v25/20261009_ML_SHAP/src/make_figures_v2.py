#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Figures for the 20261009_ML_SHAP run (3 scenarios x 2 inputs x 12 ML + SHAP).
ov.style(Arial-resolved) + OmicVerse palette; every figure -> PNG(300dpi)+PDF+source CSV.
fig01 AUC heatmap | fig02 dAUC forest (B vs A) | fig03 ROC best arms + baselines
fig04 SHAP top20 bars | fig05 SHAP beeswarm (merged RF) | fig06 SHAP cross-model heatmap
"""
from release_runtime import private_literal as _private_literal, release_path as _release_path
import os, json, glob
import numpy as np, pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.lines import Line2D

import omicverse as ov

# ---- resolve a real local publication font (avoid silent DejaVu fallback) ----
def resolve_publication_font():
    import subprocess
    cands = []
    try:
        out = subprocess.run(['fc-list'], capture_output=True, text=True, timeout=20).stdout
        for line in out.splitlines():
            p = line.split(':')[0].strip()
            b = os.path.basename(p).lower()
            if 'arial' in b and b.endswith(('.ttf','.otf')): cands.append((0, p))
            elif 'liberationsans' in b and 'regular' in b and b.endswith('.ttf'): cands.append((1, p))
            elif 'helvetica' in b and b.endswith(('.ttf','.otf')): cands.append((2, p))
            elif 'dejavusans' in b and b.endswith('.ttf'): cands.append((9, p))
        cands.sort()
        return cands[0][1] if cands else 'Arial'
    except Exception:
        return 'Arial'
FONT = resolve_publication_font()
print('font resolved ->', FONT)
ov.style(font_path=FONT)

OUT  = _release_path('path_d4455a6432ab939c')
CMV  = _release_path('path_75e8856cd7fc1147')
MERGED = f'{CMV}/pn_ml_models_merged'
FIGD = f'{OUT}/figures'
os.makedirs(FIGD, exist_ok=True)
SK = 'stratLabelBatchCohort'

# OmicVerse-adjacent CB-safe palette (Okabe-Ito)
OI = {"orange":"#E69F00","sky":"#56B4E9","green":"#009E73","yellow":"#F0E442",
      "blue":"#0072B2","verm":"#D55E00","pink":"#CC79A7","black":"#000000"}
CLUSTER_COLOR = {"tcrbert11": OI["blue"], "tcrbert7": OI["verm"], "sceptr5": OI["green"]}

MODELS = ["LASSO_logistic","Ridge_logistic","ElasticNet_logistic","Logistic_unpenalized",
          "RandomForest","HistGradBoost","XGBoost","SVM_linear","SVM_RBF","kNN","GaussianNB","MLP"]
SHORT = {"LASSO_logistic":"LASSO","Ridge_logistic":"Ridge","ElasticNet_logistic":"ElasticNet",
         "Logistic_unpenalized":"LR-unpen","RandomForest":"RF","HistGradBoost":"HistGB","XGBoost":"XGB",
         "SVM_linear":"SVM-lin","SVM_RBF":"SVM-RBF","kNN":"kNN","GaussianNB":"GNB","MLP":"MLP"}
SCEN_LABEL = {"home":"Home (n=126: 70 N0/56 N+)","external":"External (n=163: 109 N0/54 N+)",
              "merged":"Merged (n=289: 179 N0/110 N+)"}
SETS_ORDER = ["REPO","CLUSTERTCR","CLUSTER3","REPOcov","CLUSTERTCRcov","CLUSTER3cov"]
SET_LABEL = {"REPO":"A: REPO 1461","CLUSTERTCR":"B: cluster-TCR 294k","CLUSTER3":"3-cluster agg",
             "REPOcov":"A+cov","CLUSTERTCRcov":"B+cov","CLUSTER3cov":"3-cluster+cov"}

master = pd.read_csv(f'{OUT}/results/master_performance_all_scenarios.csv')
m = master[master.scheme==SK]

def save(fig, name, src_df=None):
    for ext in ('png','pdf'):
        fig.savefig(f'{FIGD}/{name}.{ext}', dpi=300, bbox_inches='tight', pad_inches=0.02)
    plt.close(fig)
    if src_df is not None:
        src_df.to_csv(f'{FIGD}/{name}_source.csv', index=False)
    print('saved', name, flush=True)

# ================= fig01: AUC heatmap (3 scenarios) =================
cols = SETS_ORDER
fig, axes = plt.subplots(1, 3, figsize=(12.6, 5.4), constrained_layout=True)
src_rows = []
for ax, scen in zip(axes, ['home','external','merged']):
    piv = (m[m.scenario==scen].pivot_table(index='model', columns='feature_set',
           values='auc_pooled_oof').reindex(index=MODELS)[cols])
    src_rows.append(piv.assign(scenario=scen))
    im = ax.imshow(piv.values, cmap='viridis', vmin=0.40, vmax=0.80, aspect='auto')
    ax.set_xticks(range(len(cols))); ax.set_xticklabels([SET_LABEL[c] for c in cols], rotation=38, ha='right', fontsize=7.5)
    ax.set_yticks(range(12)); ax.set_yticklabels([SHORT[mm] for mm in MODELS], fontsize=8)
    for i in range(12):
        for j in range(len(cols)):
            v = piv.values[i, j]
            if not np.isnan(v):
                ax.text(j, i, f'{v:.2f}', ha='center', va='center', fontsize=6.6,
                        color='white' if v < 0.66 else 'black')
    bl = m[(m.scenario==scen)&(m.feature_set.str.startswith('BASE'))]
    bls = "; ".join(f"{r.feature_set.replace('BASE_','')}={r.auc_pooled_oof:.2f}" for _, r in bl.iterrows())
    ax.set_title(f"{SCEN_LABEL[scen]}\n{bls}", fontsize=8.5)
    if ax is axes[0]: ax.set_ylabel('Model', fontsize=9)
fig.suptitle('Pooled OOF AUC by scenario × input × model (nested 5×3 CV, label×batch stratified)\n'
             'A = all MiXCR repertoire features · B = per-TCR features of the 3 P<0.05 clusters (tcrbert:11/7, sceptr:5)',
             fontsize=10)
cb = fig.colorbar(im, ax=axes, shrink=0.75, pad=0.015); cb.set_label('pooled OOF AUC', fontsize=8)
src = pd.concat([r.reset_index().melt(id_vars=['model','scenario'], var_name='feature_set', value_name='auc') for r in src_rows])
save(fig, 'fig01_auc_heatmap_all_scenarios', src)

# ================= fig02: ΔAUC forest (B − A) =================
fig, axes = plt.subplots(1, 3, figsize=(12.2, 4.6), constrained_layout=True, sharex=True)
src = []
for ax, scen in zip(axes, ['home','external','merged']):
    c = pd.read_csv(f'{OUT}/results/{scen}/contrasts_clusterTCR.csv')
    c = c[c.scheme==SK]
    raw = c[c.contrast=='CLUSTERTCR_vs_REPO'].set_index('arm')
    cov = c[c.contrast=='CLUSTERTCRcov_vs_REPOcov'].set_index('arm')
    ys = np.arange(12)[::-1]
    for k, mm in enumerate(MODELS):
        y = ys[k]
        for dd, fam, colr, mk in [(raw, 'B−A raw', OI['blue'], 'o'), (cov, 'B−A +cov', OI['verm'], 's')]:
            r = dd.loc[f'{mm}__CLUSTERTCR' if fam.endswith('raw') else f'{mm}__CLUSTERTCRcov']
            ax.errorbar(r.delta_auc_pooled, y, xerr=[[r.delta_auc_pooled-r.delta_ci95_lo],[r.delta_ci95_hi-r.delta_auc_pooled]],
                        fmt=mk, color=colr, ms=4, lw=1.1, capsize=2, alpha=0.9, zorder=3)
            src.append(dict(scenario=scen, model=mm, family=fam, delta_auc=r.delta_auc_pooled,
                            ci_lo=r.delta_ci95_lo, ci_hi=r.delta_ci95_hi, p_boot=r.p_paired_boot, p_holm=r.p_holm))
    ax.axvline(0, color='grey', lw=0.8, ls='--')
    ax.set_yticks(ys); ax.set_yticklabels([SHORT[mm] for mm in MODELS], fontsize=8)
    ax.set_title(SCEN_LABEL[scen], fontsize=9)
    ax.set_xlim(-0.30, 0.30)
    if ax is axes[0]: ax.set_ylabel('Model', fontsize=9)
    if ax is axes[1]: ax.set_xlabel('Δ pooled OOF AUC  (cluster-TCR input − all-MiXCR input)', fontsize=9)
fig.suptitle('Input B vs input A: paired-bootstrap ΔAUC (5000 resamples, 95% CI; Holm within 12-model family — no contrast passes Holm in any scenario)', fontsize=9.5)
fig.legend(handles=[Line2D([],[],marker='o',color=OI['blue'],ls='-',lw=1,label='B−A (raw features)'),
                    Line2D([],[],marker='s',color=OI['verm'],ls='-',lw=1,label='B−A (+depth/cohort/batch/age/sex/T)')],
           loc='upper right', bbox_to_anchor=(0.995,0.985), fontsize=8, frameon=False)
save(fig, 'fig02_deltaAUC_B_vs_A_forest', pd.DataFrame(src))

# ================= fig03: ROC best arms + baselines =================
from sklearn.metrics import roc_curve, auc as sk_auc
fig, axes = plt.subplots(1, 3, figsize=(12.2, 4.4), constrained_layout=True)
src=[]
for ax, scen in zip(axes, ['home','external','merged']):
    d = pd.read_csv(f'{MERGED}/src/merged_design.csv', index_col=0)
    d = d if scen=='merged' else d[d.cohort_external==(0 if scen=='home' else 1)]
    y = d['y'].astype(int).values
    prior = {'home':'pn_ml_models_home','external':'pn_ml_models_external','merged':'pn_ml_models_merged'}[scen]
    oofA = pd.read_csv(f'{CMV}/{prior}/results/oof_predictions_all_arms.csv', index_col=0).loc[d.index]
    oofB = pd.read_csv(f'{OUT}/results/{scen}/oof_clusterTCR.csv', index_col=0).loc[d.index]
    ms = m[m.scenario==scen]
    bestA = ms[(ms.feature_set.isin(['REPO','REPOcov','CONCAT','CONCATcov']))].sort_values('auc_pooled_oof').iloc[-1]
    bestB = ms[(ms.feature_set.isin(['CLUSTERTCR','CLUSTERTCRcov']))].sort_values('auc_pooled_oof').iloc[-1]
    bestC3 = ms[(ms.feature_set.isin(['CLUSTER3','CLUSTER3cov']))].sort_values('auc_pooled_oof').iloc[-1]
    arms = [(f"Input A best: {bestA.feature_set}/{SHORT[bestA.model]}", oofA[f'{bestA.model}__{bestA.feature_set}__{SK}'], OI['blue'], '-'),
            (f"Input B best: {bestB.feature_set}/{SHORT[bestB.model]}", oofB[f'{bestB.model}__{bestB.feature_set}__{SK}'], OI['verm'], '-'),
            (f"3-cluster agg best: {bestC3.feature_set}/{SHORT[bestC3.model]}", oofA[f'{bestC3.model}__{bestC3.feature_set}__{SK}'], OI['green'], '-')]
    for bn, lab in [('BASE_depthcohbatch','depth+batch(+cohort) baseline'), ('BASE_Tstage','T-stage alone')]:
        arm = f'{bn}__logL2__{SK}'
        if arm in oofA.columns: arms.append((f"{lab}", oofA[arm], OI['black'], '--'))
    for lab, s, colr, ls in arms:
        fpr, tpr, _ = roc_curve(y, s.values)
        a = sk_auc(fpr, tpr)
        ax.plot(fpr, tpr, color=colr, ls=ls, lw=1.4, label=f'{lab} — AUC {a:.3f}')
        src.append(dict(scenario=scen, curve=lab, fpr=fpr, tpr=tpr, auc=a))
    ax.plot([0,1],[0,1], color='grey', lw=0.7, ls=':')
    ax.set_xlabel('1 − specificity', fontsize=9); ax.set_ylabel('Sensitivity', fontsize=9)
    ax.set_title(SCEN_LABEL[scen], fontsize=9); ax.legend(fontsize=6.6, loc='lower right', frameon=False)
fig.suptitle('Pooled OOF ROC — best arm per input family vs confounder/T-stage baselines', fontsize=10)
srcdf = pd.concat([pd.DataFrame({'scenario':r['scenario'],'curve':r['curve'],'auc':r['auc'],
                  'fpr':r['fpr'],'tpr':r['tpr']}) for r in src])
save(fig, 'fig03_roc_best_arms', srcdf)

# ================= fig04: SHAP top20 per scenario =================
BEST = {'home':'LASSO_logistic','external':'XGBoost','merged':'RandomForest'}
fig, axes = plt.subplots(1, 3, figsize=(13.0, 5.8), constrained_layout=True)
src=[]
for ax, scen in zip(axes, ['home','external','merged']):
    t = pd.read_csv(f'{OUT}/tables/shap_top20_{scen}_{BEST[scen]}.csv')
    t = t.iloc[::-1].reset_index(drop=True)
    ys = np.arange(len(t))
    cols_ = [CLUSTER_COLOR[c] for c in t.cluster]
    ax.barh(ys, t.cv_mean_abs_shap, color=cols_, height=0.72)
    ax.set_yticks(ys)
    lab = [f"{aa}  [{cl}]" for aa, cl in zip(t.aa, t.cluster)]
    ax.set_yticklabels(lab, fontsize=6.4, family='monospace')
    for i, (v, fs, sg) in enumerate(zip(t.cv_mean_abs_shap, t.freq_selected, t.cv_signed_mean)):
        ax.text(v, i, f'  f={fs:.1f}{"↑" if sg>0 else "↓"}', va='center', fontsize=5.8, color='0.25')
    ax.set_xlabel('CV mean |SHAP| (log-odds, N+ direction)', fontsize=8.5)
    ax.set_title(f"{SCEN_LABEL[scen]}\nbest cluster-TCR model: {BEST[scen]} (pooled OOF AUC "
                 f"{m[(m.scenario==scen)&(m.model==BEST[scen])&(m.feature_set=='CLUSTERTCR')].auc_pooled_oof.iloc[0]:.3f})", fontsize=8.5)
    t2 = t.copy(); t2['rank']=np.arange(1,len(t)+1)[::-1]; src.append(t2)
handles = [Line2D([],[],marker='s',color=CLUSTER_COLOR[k],ls='',label=f'{k} ({"TCR-BERT" if k.startswith("tcrbert") else "SCEPTR"})') for k in ['tcrbert11','tcrbert7','sceptr5']]
fig.legend(handles=handles, loc='upper right', bbox_to_anchor=(0.998,0.99), fontsize=8, frameon=False, title='cluster', title_fontsize=8)
fig.suptitle('Top-20 N+ contributing cluster-TCRs by fold-honest SHAP (selected in k/5 outer folds; ↑/↓ = mean direction toward N+/N0)', fontsize=10)
save(fig, 'fig04_shap_top20_per_scenario', pd.concat(src, ignore_index=True))

# ================= fig05: SHAP beeswarm (merged, RF) =================
payload = pd.read_csv(f'{OUT}/tables/shap_beeswarm_payload_merged_RandomForest.csv')
imp = pd.read_csv(f'{OUT}/tables/shap_feature_importance_merged_RandomForest.csv')
top = imp.sort_values('full_mean_abs_shap', ascending=False).head(15)
feats = list(top.feature)
xb = pd.read_parquet(f'{OUT}/src/X_clusterTCR_289.parquet', columns=feats)
zv = (xb - xb.mean()) / (xb.std() + 1e-12)
rng = np.random.RandomState(0)
fig, ax = plt.subplots(figsize=(7.4, 5.2), constrained_layout=True)
rows=[]
for i, f in enumerate(feats):
    sub = payload[payload.feature==f]
    z = zv[f].reindex(sub.patient_key).values
    jit = rng.uniform(-0.16, 0.16, len(sub))
    sc = ax.scatter(sub.shap.values, i + jit, c=np.clip(z, -2.5, 2.5), cmap='coolwarm', s=7, alpha=0.85, lw=0)
    rows.append(pd.DataFrame({'feature':f,'patient_key':sub.patient_key,'shap':sub.shap,'z_value':z}))
cb = fig.colorbar(sc, ax=ax, shrink=0.8, pad=0.02); cb.set_label('feature value (z, clipped ±2.5)', fontsize=8)
ax.set_yticks(range(len(feats)))
ax.set_yticklabels([f"{n.split('::')[1]} [{n.split('::')[0]}]" for n in feats], fontsize=6.8, family='monospace')
ax.axvline(0, color='grey', lw=0.8, ls='--')
ax.set_xlabel('SHAP value (log-odds, N+ direction) — merged cohort, RandomForest full refit (in-sample view)', fontsize=8.5)
ax.set_title('SHAP beeswarm — top-15 cluster-TCRs, merged cohort', fontsize=10)
save(fig, 'fig05_shap_beeswarm_merged_RF', pd.concat(rows, ignore_index=True))

# ================= fig06: cross-model SHAP heatmap =================
combos = []
for scen in ['home','external','merged']:
    for mdl in glob.glob(f'{OUT}/tables/shap_feature_importance_{scen}_*.csv'):
        mm = os.path.basename(mdl).replace(f'shap_feature_importance_{scen}_','').replace('.csv','')
        df = pd.read_csv(mdl)[['feature','cv_mean_abs_shap']]
        df['col'] = f'{scen[:3]}.{SHORT.get(mm,mm)}'; combos.append(df)
allimp = pd.concat(combos)
cons = pd.read_csv(f'{OUT}/tables/shap_consensus_rank_merged.csv').head(20)
top_feats = list(cons.feature)
mat = allimp[allimp.feature.isin(top_feats)].pivot_table(index='feature', columns='col', values='cv_mean_abs_shap')
mat = mat.reindex(index=top_feats)
matn = mat.div(mat.max(axis=0), axis=1) * 100  # per-model column-normalized %
col_order = [c for c in ['hom.LASSO','hom.RF','hom.XGB','hom.HGB','ext.XGB','ext.RF','ext.HGB','mer.XGB','mer.RF','mer.HGB'] if c in matn.columns]
matn = matn[col_order]
fig, ax = plt.subplots(figsize=(8.6, 6.2), constrained_layout=True)
im = ax.imshow(matn.values, cmap='viridis', aspect='auto', vmin=0, vmax=100)
ax.set_xticks(range(len(col_order))); ax.set_xticklabels(col_order, rotation=38, ha='right', fontsize=8)
ax.set_yticks(range(len(top_feats)))
ax.set_yticklabels([f"{n.split('::')[1]} [{n.split('::')[0]}]" for n in top_feats], fontsize=6.8, family='monospace')
for i in range(matn.shape[0]):
    for j in range(matn.shape[1]):
        v = matn.values[i, j]
        if not np.isnan(v):
            ax.text(j, i, f'{v:.0f}', ha='center', va='center', fontsize=5.8,
                    color='white' if v < 60 else 'black')
cb = fig.colorbar(im, ax=ax, shrink=0.8, pad=0.02); cb.set_label('% of that model\'s max CV mean|SHAP|', fontsize=8)
ax.set_title('Top-20 consensus cluster-TCRs (merged) — SHAP importance across models & scenarios\n(0 = not selected in any outer fold of that model)', fontsize=9.5)
src6 = matn.reset_index().melt(id_vars='feature', var_name='model_col', value_name='pct_of_max')
save(fig, 'fig06_shap_crossmodel_heatmap', src6)

print('ALL FIGURES DONE')
