#!/usr/bin/env python3
"""Freeze verified nodal analyses and render protocol-separated paper figures.

No classifiers are fitted. All calculations must run on the analysis server.
"""
from release_runtime import private_literal as _private_literal, release_path as _release_path
import argparse
import hashlib
import json
from pathlib import Path
import platform
import shutil

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score, log_loss, brier_score_loss, roc_curve


V1_FILES = [
    "cohort/patient_master.tsv", "cohort/library_master.tsv", "cohort/endpoint_audit.tsv",
    "features/patient_features.tsv", "features/patient_features_duplicate_pooled.tsv",
    "features/feature_dictionary.tsv", "features/feature_blocks.json", "features/feature_build_receipt.json",
    "features/patient_index.tsv", "features/sceptr_patient_means.npy", "features/rarefied_summary_features.tsv",
    "models/metrics.tsv", "models/metrics_ci.tsv", "models/paired_comparisons.tsv",
    "models/oof_patient_ensemble.tsv", "models/oof_predictions.tsv", "models/fold_audit.tsv",
    "models/full_pipeline_bootstrap.tsv", "models/full_pipeline_paired_ci.tsv", "models/conditional_paired_ci.tsv",
    "models/bootstrap_initial_failures.tsv", "models/bootstrap_repair_110_fold_audit.tsv",
    "models/primary_partition_equivalence_audit.tsv", "models/final_validation_receipt.json",
    "models/external_metrics.tsv", "models/external_predictions.tsv", "models/external_paired_metrics.json",
    "models/external_conditional_paired_ci.tsv", "models/sceptr_metrics.tsv", "models/sceptr_paired_metrics.json",
    "models/sceptr_conditional_paired_ci.tsv", "models/sceptr_oof_predictions.tsv", "models/sceptr_status.json",
    "models/lobo_metrics.tsv", "models/lobo_paired_metrics.tsv", "models/lobo_predictions.tsv",
    "models/negative_controls.tsv", "models/pooled_duplicate_metrics.tsv", "models/pooled_duplicate_paired_metrics.json",
    "models/model_spec.json", "models/status.json", "models/README.md", "models/final_test_log.txt",
    "association/primary_associations.tsv", "association/sensitivity_associations.tsv",
    "association/batch_interactions.tsv", "association/ordinal_N_trend.tsv", "association/status.json",
    "association/README.md", "legacy/old_s5_primary_summary.tsv",
    "code/run_ridge.py", "code/run_associations.py", "code/build_cohort_features.py", "tests/test_models.py",
]
AUDIT_FILES = [
    "historical_four_arm_recomputed.tsv", "current_ridge_recomputed.tsv", "common125_patients.tsv",
    "library_scope_difference.tsv", "verification_receipt.json", "N_METHODS_AUDIT.md",
    "sources/representation_comparison/model_patient_oof.tsv", "sources/representation_comparison/summary.json",
]
DISPLAY = {"tcrbert_dualtask_aa_mean":"TCR-BERT", "sceptr_cdr3_only":"SCEPTR", "cdr3vec_small":"CDR3vec", "kmer3_cosine_svd":"3-mer + SVD"}
ORDER = list(DISPLAY)
BLUE, ORANGE, GRAY = "#277DA1", _private_literal('private_ffdde0eb0d464546'), "#7A8591"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def freeze(source, relative, out, scope, rows):
    source = Path(source)
    if not source.is_file():
        raise FileNotFoundError(f"Required source missing: {source}")
    target = out/"frozen"/scope/relative
    target.parent.mkdir(parents=True, exist_ok=True)
    checksum = sha(source)
    if target.exists():
        assert sha(target)==checksum, f"Source changed after freezing: {source}"
    else:
        shutil.copy2(source,target)
    assert sha(target)==checksum
    rows.append({"scope":scope,"source_path":str(source),"frozen_path":str(target.relative_to(out)),"bytes":source.stat().st_size,"SHA256":checksum})


def score(y,p):
    return {"AUROC":roc_auc_score(y,p),"AUPRC":average_precision_score(y,p),"log_loss":log_loss(y,p),"Brier":brier_score_loss(y,p)}


def read(out,relative,scope="paper_v1"):
    return pd.read_csv(out/"frozen"/scope/relative,sep="\t")


def verify(out):
    feat=read(out,"features/patient_features.tsv")
    master=read(out,"cohort/patient_master.tsv")
    assert len(feat)==len(master)==339 and feat.patient_key.is_unique and master.patient_key.is_unique
    assert set(feat.patient_key)==set(master.patient_key)
    joined=feat.merge(master[["patient_key","n_binary","primary_sample_ids"]],on="patient_key",validate="one_to_one",suffixes=("","_master"))
    assert np.allclose(joined.n_binary,joined.n_binary_master,equal_nan=True)
    assert (joined.primary_sample_ids==joined.primary_sample_ids_master).all()
    home=feat[(feat.cohort=="home")&feat.n_binary.notna()]
    ext=feat[(feat.cohort=="external")&feat.n_binary.notna()]
    assert len(home)==126 and home.n_binary.sum()==56 and len(ext)==163 and ext.n_binary.sum()==54
    checks=[]
    oof=read(out,"models/oof_patient_ensemble.tsv")
    metrics=read(out,"models/metrics.tsv");metrics=metrics[metrics.prediction=="three_seed_mean"]
    for model in ["M0","M1","M2","M3"]:
        part=oof[oof.model==model]
        assert part.patient_key.is_unique and set(part.patient_key)==set(home.patient_key)
        row=metrics[metrics.model==model].iloc[0]
        for name,value in score(part.n_binary,part.probability).items():
            assert abs(value-row[name])<1e-12
            checks.append({"scope":_private_literal('private_5db4a27593ef5a10'),"model":model,"metric":name,"recomputed":value,"source":row[name],"passed":True})
    extpred=read(out,"models/external_predictions.tsv");extmet=read(out,"models/external_metrics.tsv")
    for model in extmet.model:
        part=extpred[extpred.model==model];row=extmet[extmet.model==model].iloc[0]
        assert part.patient_key.is_unique and set(part.patient_key)==set(ext.patient_key)
        for name,value in score(part.n_binary,part.probability).items():
            assert abs(value-row[name])<1e-12
            checks.append({"scope":"external163","model":model,"metric":name,"recomputed":value,"source":row[name],"passed":True})
    hist=read(out,"historical_four_arm_recomputed.tsv","reframing")
    histpred=read(out,"sources/representation_comparison/model_patient_oof.tsv","reframing")
    for model in ORDER:
        part=histpred[histpred.model==model];row=hist[(hist.model==model)&(hist.population=="historical126")].iloc[0]
        assert part.patient_key.is_unique and set(part.patient_key)==set(home.patient_key)
        for name,value in score(part.recorded_n_binary,part.model_probability).items():
            assert abs(value-row[name])<1e-10
            checks.append({"scope":_private_literal('private_a2d7a4547c61077d'),"model":model,"metric":name,"recomputed":value,"source":row[name],"passed":True})
    common=read(out,"common125_patients.tsv","reframing")
    assert len(common)==125 and common.patient_key.is_unique and common.n_binary.sum()==55
    assert set(home.patient_key)-set(common.patient_key)=={_private_literal('private_283790eb0a2c6510')}
    current_audit=read(out,"current_ridge_recomputed.tsv","reframing")
    for model in ["M0","M1","M2","M3"]:
        part=oof[(oof.model==model)&oof.patient_key.isin(common.patient_key)]
        row=current_audit[(current_audit.model==model)&(current_audit.population=="matched_primary_library125")].iloc[0]
        for name,value in score(part.n_binary,part.probability).items():
            assert abs(value-row[name])<1e-12
            checks.append({"scope":"current_common125","model":model,"metric":name,"recomputed":value,"source":row[name],"passed":True})
    for model in ORDER:
        part=histpred[(histpred.model==model)&histpred.patient_key.isin(common.patient_key)]
        row=hist[(hist.model==model)&(hist.population=="matched_primary_library125")].iloc[0]
        for name,value in score(part.recorded_n_binary,part.model_probability).items():
            assert abs(value-row[name])<1e-10
            checks.append({"scope":"historical_common125","model":model,"metric":name,"recomputed":value,"source":row[name],"passed":True})
    bootstrap=read(out,"models/full_pipeline_bootstrap.tsv")
    assert len(bootstrap)==200 and bootstrap.repetition.is_unique and (bootstrap.status=="success").all() and (bootstrap.patient_overlap==0).all()
    ci=read(out,"models/full_pipeline_paired_ci.tsv")
    for _,r in ci.iterrows():
        assert abs(bootstrap[r.metric].quantile(.025)-r.ci_low)<1e-12
        assert abs(bootstrap[r.metric].quantile(.975)-r.ci_high)<1e-12
    associations=read(out,"association/primary_associations.tsv")
    assert len(associations)==81 and associations.groupby("family").size().to_dict()=={"summary_3":3,"vj_all":78}
    assert (associations.q>=.05).all()
    status=json.loads((out/"frozen/paper_v1/association/status.json").read_text());assert status["skipped"]==[]
    pd.DataFrame(checks).to_csv(out/"source_tables/independent_metric_checks.tsv",sep="\t",index=False)
    return feat,master,home,ext


def table1(out,feat,master,home,ext):
    rows=[]
    columns={"Home_all":feat[feat.cohort=="home"],"Home_known_N":home,"External_all":feat[feat.cohort=="external"],"External_known_N":ext}
    for population,d in columns.items():
        def add(variable,display,observed,missing=0):
            rows.append({"population":population,"variable":variable,"display":display,"n_total":len(d),"n_observed":observed,"n_missing":missing})
        add("Patients",str(len(d)),len(d))
        add("Libraries before patient representation",str(int(d.library_count.sum())),len(d))
        add("Analysis libraries",str(int(d.analysis_library_count.sum())),len(d))
        add("Recorded N known",f"{int(d.n_binary.notna().sum())} ({d.n_binary.notna().mean()*100:.1f}%)",int(d.n_binary.notna().sum()),int(d.n_binary.isna().sum()))
        for n,label in [(0,"N0"),(1,"N+")]:
            count=int((d.n_binary==n).sum());add(f"Recorded {label}",f"{count} ({count/len(d)*100:.1f}%)",count)
        raw=d.n_raw.astype(str).str.replace("N","",regex=False).str.replace(r"\.0$","",regex=True)
        for value in ["0","1","2"]:
            count=int((raw==value).sum());add(f"Recorded N{value} category",f"{count} ({count/len(d)*100:.1f}%)",count)
        for feature,label in [("age_years","Age, years"),("total_reads","Productive QC reads"),("unique_aa","Unique CDR3 AA"),("normalized_shannon","Normalized Shannon"),("top10_fraction","Top-10 read fraction"),("weighted_mean_cdr3_length","Read-weighted CDR3 length, AA")]:
            values=d[feature].dropna();digits=0 if feature in {"age_years","total_reads","unique_aa"} else 3
            display="Not available" if not len(values) else f"{values.median():,.{digits}f} [{values.quantile(.25):,.{digits}f}, {values.quantile(.75):,.{digits}f}]"
            add(label,display,len(values),len(d)-len(values))
        for value,label in [("f","Female"),("m","Male")]:
            count=int((d.sex==value).sum());den=int(d.sex.notna().sum());display="Not available" if not den else f"{count} ({count/den*100:.1f}%)"
            add(label,display,den,len(d)-den)
        for value,label in [("yes","Recorded neoadjuvant therapy"),("no","Recorded no neoadjuvant therapy")]:
            den=int(d.neoadjuvant.isin(["yes","no"]).sum());count=int((d.neoadjuvant==value).sum());display="Not available" if not den else f"{count} ({count/den*100:.1f}%)"
            add(label,display,den,len(d)-den)
        for batch,count in d.batch.value_counts().sort_index().items():
            add(f"Acquisition batch {batch}",f"{count} ({count/len(d)*100:.1f}%)",len(d))
    tidy=pd.DataFrame(rows);tidy.to_csv(out/"source_tables/Table1_descriptive_tidy.tsv",sep="\t",index=False)
    wide=tidy.pivot(index="variable",columns="population",values="display").reindex(tidy.variable.drop_duplicates())
    wide.to_csv(out/"source_tables/Table1_descriptive.tsv",sep="\t")
    # Denominators and missingness are preserved separately, not silently imputed.
    return tidy


def calibration_bins(out):
    pred=read(out,"models/external_predictions.tsv");rows=[]
    for model,d in pred.groupby("model",sort=False):
        d=d.copy();d["bin"]=pd.qcut(d.probability,5,labels=False,duplicates="drop")
        for b,part in d.groupby("bin",sort=True):
            n=len(part);p=part.n_binary.mean();z=1.959963984540054
            centre=(p+z*z/(2*n))/(1+z*z/n);half=z*np.sqrt(p*(1-p)/n+z*z/(4*n*n))/(1+z*z/n)
            rows.append({"model":model,"bin":int(b),"n":n,"N_positive":int(part.n_binary.sum()),"mean_probability":part.probability.mean(),"observed_fraction":p,"Wilson_ci_low":centre-half,"Wilson_ci_high":centre+half})
    table=pd.DataFrame(rows);table.to_csv(out/"source_tables/Figure2D_external_calibration_bins.tsv",sep="\t",index=False)
    assert table.groupby("model").n.sum().eq(163).all()
    return table


def theme():
    plt.rcParams.update({"font.family":"DejaVu Sans","font.size":9.5,"axes.titlesize":10.5,"axes.labelsize":9.5,"axes.linewidth":.6,"xtick.labelsize":8.5,"ytick.labelsize":8.5,"pdf.fonttype":42,"ps.fonttype":42,"svg.fonttype":"none","savefig.facecolor":"white"})


def decorate(ax,letter,title,subtitle):
    for side in ["top","right"]:ax.spines[side].set_visible(False)
    ax.tick_params(width=.6,length=3)
    ax.text(-.20,1.16,letter,transform=ax.transAxes,fontsize=13,fontweight="bold",va="bottom")
    ax.set_title(title,loc="left",pad=25,fontweight="bold")
    ax.text(0,1.045,subtitle,transform=ax.transAxes,fontsize=8.5,color="#555555",va="bottom")


def save(fig,out,name):
    for extension in ["pdf","svg","png"]:
        fig.savefig(out/"figures"/f"{name}.{extension}",dpi=450,bbox_inches="tight",pad_inches=.08)
    plt.close(fig)


def figure2(out):
    theme();fig,axs=plt.subplots(2,2,figsize=(7.5,8.0));fig.subplots_adjust(left=.19,right=.98,bottom=.13,top=.90,hspace=.82,wspace=.85)
    hist=read(out,"historical_four_arm_recomputed.tsv","reframing")
    a=hist[(hist.population=="historical126")&hist.model.isin(ORDER)].set_index("model").loc[ORDER].reset_index()
    a["label"]=a.model.map(DISPLAY);a.to_csv(out/"source_tables/Figure2A_historical_representation.tsv",sep="\t",index=False)
    ax=axs[0,0];y=np.arange(4)[::-1]
    ax.errorbar(a.delta_log_loss,y,xerr=np.vstack([a.delta_log_loss-a.conditional_ci_low,a.conditional_ci_high-a.delta_log_loss]),fmt="o",color=BLUE,capsize=3,ms=4,lw=1)
    arm_labels=[f"{r.label}\nAUROC {r.AUROC:.3f}" for _,r in a.iterrows()]
    ax.axvline(0,color=GRAY,lw=.7,ls="--");ax.set_yticks(y,arm_labels);ax.set_xlim(-.042,.029);ax.set_xticks([-.04,-.02,0,.02]);ax.set_ylim(-.6,3.6);ax.set_xlabel("Δlog-loss vs own L2 baseline")
    decorate(ax,"A","Historical representations",_private_literal('private_a4f5631faaabce54'))
    ax.text(0,-.32,"Baseline AUROC 0.579\nLog-loss 0.681",transform=ax.transAxes,fontsize=8.5,color="#555555")
    metrics=read(out,"models/metrics.tsv");metrics=metrics[metrics.prediction=="three_seed_mean"].set_index("model").loc[["M0","M1","M2","M3"]].reset_index()
    ci=read(out,"models/metrics_ci.tsv");ci=ci[(ci.cohort=="Home")&(ci.metric=="AUROC")&ci.model.isin(["M0","M1","M2","M3"])]
    b=metrics.merge(ci[["model","ci_low","ci_high","ci_type"]],on="model",validate="one_to_one");b.to_csv(out/"source_tables/Figure2B_current_ridge.tsv",sep="\t",index=False)
    ax=axs[0,1];labels=["M0: age, sex","M1: + depth, batch","M2: TCR only","M3: M1 + TCR"]
    for i,(_,r) in zip(y,b.iterrows()):
        color=ORANGE if r.model=="M3" else BLUE
        ax.errorbar(r.AUROC,i,xerr=[[r.AUROC-r.ci_low],[r.ci_high-r.AUROC]],fmt="o",color=color,capsize=3,ms=4,lw=1)
        ax.text(r.AUROC,i+.20,f"{r.AUROC:.3f}",ha="center",va="center",fontsize=8.5)
    ax.axvline(.5,color=GRAY,lw=.7,ls="--");ax.set_yticks(y,labels);ax.set_xlim(.28,.75);ax.set_ylim(-.6,3.6);ax.set_xlabel("AUROC")
    decorate(ax,"B","Current ridge protocol",_private_literal('private_f790cf2954f00e8e'))
    pair=read(out,"models/paired_comparisons.tsv");main=pair[(pair.prediction=="three_seed_mean")&(pair.baseline=="M1")&(pair.new=="M3")].iloc[0]
    full=read(out,"models/full_pipeline_paired_ci.tsv");full=full[full.metric=="delta_log_loss"].iloc[0]
    sp=json.loads((out/"frozen/paper_v1/models/sceptr_paired_metrics.json").read_text());sci=read(out,"models/sceptr_conditional_paired_ci.tsv");sci=sci[sci.metric=="delta_log_loss"].iloc[0]
    c=pd.DataFrame([{"label":"M3 vs M1","estimate":main.delta_log_loss,"ci_low":full.ci_low,"ci_high":full.ci_high,"interval":"200 full-pipeline refits"},{"label":"SCEPTR + M1 vs M1","estimate":sp["delta_log_loss"],"ci_low":sci.ci_low,"ci_high":sci.ci_high,"interval":"2000 fixed-OOF draws"}]);c.to_csv(out/"source_tables/Figure2C_current_increment.tsv",sep="\t",index=False)
    ax=axs[1,0]
    for i,(_,r) in zip([1,0],c.iterrows()):
        color=ORANGE if i==1 else BLUE
        ax.errorbar(r.estimate,i,xerr=[[r.estimate-r.ci_low],[r.ci_high-r.estimate]],fmt="o",color=color,capsize=3,ms=4,lw=1)
        ax.text(.063,i+.21,f"{r.estimate:+.3f}\n[{r.ci_low:+.3f}, {r.ci_high:+.3f}]",ha="right",va="center",fontsize=8.5)
    ax.axvline(0,color=GRAY,lw=.7,ls="--");ax.set_yticks([1,0],["M3 vs M1\nFull pipeline","SCEPTR + M1\nvs M1; conditional"]);ax.set_xlim(-.10,.067);ax.set_xticks([-.1,-.05,0,.05]);ax.set_ylim(-.5,1.6);ax.set_xlabel("Δlog-loss (positive = TCR gain)")
    decorate(ax,"C","Current TCR increment",_private_literal('private_12a465964cbacdca'))
    bins=calibration_bins(out);ax=axs[1,1]
    ax.plot([0,1],[0,1],ls="--",color=GRAY,lw=.8,zorder=0)
    for model,color,marker,label in [("M1_portable",BLUE,"o","Technical baseline"),("M3_portable",ORANGE,"s","Baseline + TCR")]:
        d=bins[bins.model==model]
        ax.errorbar(d.mean_probability,d.observed_fraction,yerr=np.vstack([d.observed_fraction-d.Wilson_ci_low,d.Wilson_ci_high-d.observed_fraction]),color=color,fmt=marker+"-",ms=3.5,lw=.8,capsize=2,label=label)
    ax.set_xlim(0,1);ax.set_ylim(0,1);ax.set_xlabel("Predicted N+ probability");ax.set_ylabel("Observed N+ fraction");ax.legend(loc="upper left",frameon=False,fontsize=8)
    decorate(ax,"D","External portability","External163; quantile calibration")
    ax.text(.03,.04,"AUROC: 0.498 → 0.522\nLog-loss: 1.289 → 0.920",transform=ax.transAxes,fontsize=8,va="bottom",bbox={"facecolor":"white","edgecolor":"none","alpha":.9})
    extp=json.loads((out/"frozen/paper_v1/models/external_paired_metrics.json").read_text());extci=read(out,"models/external_conditional_paired_ci.tsv");extci=extci[extci.metric=="delta_log_loss"].iloc[0]
    ax.text(0,-.38,f"ΔLL {extp['delta_log_loss']:+.3f} [{extci.ci_low:.3f}, {extci.ci_high:.3f}]\nConditional paired 95% CI\nAbsolute calibration remains poor",transform=ax.transAxes,fontsize=8.5,color="#555555")
    save(fig,out,"Figure2")


def supplementary(out):
    theme();fig,grid=plt.subplots(2,2,figsize=(7.5,7.5));axs=grid.ravel();fig.subplots_adjust(left=.18,right=.98,bottom=.11,top=.90,hspace=.82,wspace=.84)
    oof=read(out,"models/oof_patient_ensemble.tsv")
    for model,color,ls in [("M1",BLUE,"-"),("M3",ORANGE,"-"),("M2",GRAY,"--")]:
        d=oof[oof.model==model];fpr,tpr,_=roc_curve(d.n_binary,d.probability);axs[0].plot(fpr,tpr,color=color,ls=ls,lw=1,label=model)
    axs[0].plot([0,1],[0,1],color="#BBBBBB",ls=":",lw=.7);axs[0].set(xlabel="False-positive rate",ylabel="True-positive rate",xlim=(0,1),ylim=(0,1));axs[0].legend(frameon=False,fontsize=8.5,loc="lower right");decorate(axs[0],"A","Current OOF ROC",_private_literal('private_6c0c84b0b9b09bba'))
    lobo=read(out,"models/lobo_metrics.tsv");lobo.to_csv(out/"source_tables/Supplement_N1B_LOBO.tsv",sep="\t",index=False)
    batches=list(lobo.held_out_batch.drop_duplicates());xs=np.arange(len(batches))
    for model,color,offset in [("M1_lobo",BLUE,-.09),("M3_lobo",ORANGE,.09)]:
        d=lobo[lobo.model==model].set_index("held_out_batch").loc[batches];axs[1].scatter(xs+offset,d.AUROC,color=color,s=20,label=model.replace("_lobo",""))
    axs[1].axhline(.5,color=GRAY,ls="--",lw=.7);axs[1].set_xticks(xs,[b.replace("H2498","") for b in batches]);axs[1].set_ylim(0,1);axs[1].set(xlabel="Held-out acquisition batch",ylabel="AUROC");axs[1].legend(frameon=False,fontsize=8.5);decorate(axs[1],"B","Leave-one-batch","Batch omitted from both models")
    a=read(out,"association/primary_associations.tsv");a=a[a.family=="summary_3"].copy();a.to_csv(out/"source_tables/Supplement_N1C_summary_associations.tsv",sep="\t",index=False)
    labels={"normalized_shannon":"Shannon","top10_fraction":"Top-10 fraction","weighted_mean_cdr3_length":"CDR3 length"}
    axs[2].errorbar(a.OR,np.arange(3)[::-1],xerr=np.vstack([a.OR-a.CIlo,a.CIhi-a.OR]),fmt="o",color=BLUE,capsize=3,ms=4,lw=1);axs[2].axvline(1,color=GRAY,lw=.7,ls="--");axs[2].set_yticks(np.arange(3)[::-1],a.feature.map(labels));axs[2].set_xlim(.4,2.05);axs[2].set_xticks([.5,1,1.5,2]);axs[2].set_xlabel("Adjusted OR per feature SD");decorate(axs[2],"C","Summary associations",_private_literal('private_467085283638cb35'))
    trend=read(out,"association/ordinal_N_trend.tsv");trend.to_csv(out/"source_tables/Supplement_N1D_ordinal_trend.tsv",sep="\t",index=False)
    axs[3].errorbar(trend.OR,np.arange(3)[::-1],xerr=np.vstack([trend.OR-trend.CIlo,trend.CIhi-trend.OR]),fmt="o",color=GRAY,capsize=3,ms=4,lw=1);axs[3].axvline(1,color=GRAY,lw=.7,ls="--");axs[3].set_yticks(np.arange(3)[::-1],trend.feature.map(labels));axs[3].set_xlim(.4,2.05);axs[3].set_xticks([.5,1,1.5,2]);axs[3].set_xlabel("Ordinal OR per feature SD");decorate(axs[3],"D","Recorded-N ordinal trend","N0/N1/N2; three-test q = 0.658")
    save(fig,out,"Supplement_N1")
    fig=plt.figure(figsize=(7.5,7.5));gs=fig.add_gridspec(2,2,left=.23,right=.98,bottom=.10,top=.90,hspace=.95,wspace=.85);axs=[fig.add_subplot(gs[0,0]),fig.add_subplot(gs[0,1]),fig.add_subplot(gs[1,:])]
    hist=read(out,"historical_four_arm_recomputed.tsv","reframing");h=hist[(hist.population=="matched_primary_library125")&hist.model.isin(ORDER)].set_index("model").loc[ORDER].reset_index();h.to_csv(out/"source_tables/Supplement_N2A_common125_historical.tsv",sep="\t",index=False)
    axs[0].errorbar(h.delta_log_loss,np.arange(4)[::-1],xerr=np.vstack([h.delta_log_loss-h.conditional_ci_low,h.conditional_ci_high-h.delta_log_loss]),fmt="o",color=BLUE,capsize=3,ms=4,lw=1);axs[0].set_yticks(np.arange(4)[::-1],h.model.map(DISPLAY));axs[0].axvline(0,color=GRAY,ls="--",lw=.7);axs[0].set_xlim(-.05,.03);axs[0].set_xticks([-.04,-.02,0,.02]);axs[0].set_xlabel("Δlog-loss vs own baseline");decorate(axs[0],"A","Historical, common125","70 N0 / 55 N+; fixed OOF scoring")
    cur=read(out,"current_ridge_recomputed.tsv","reframing");cur=cur[(cur.population=="matched_primary_library125")&(cur.model=="M3")];cur.to_csv(out/"source_tables/Supplement_N2B_common125_current.tsv",sep="\t",index=False);r=cur.iloc[0]
    axs[1].errorbar(r.delta_log_loss,0,xerr=[[r.delta_log_loss-r.conditional_ci_low],[r.conditional_ci_high-r.delta_log_loss]],fmt="o",color=ORANGE,capsize=3,ms=4,lw=1);axs[1].set_yticks([0],["M3 vs M1"]);axs[1].axvline(0,color=GRAY,ls="--",lw=.7);axs[1].set_xlim(-.08,.04);axs[1].set_xticks([-.08,-.04,0,.04]);axs[1].set_ylim(-.8,.8);axs[1].set_xlabel("Δlog-loss vs current M1");decorate(axs[1],"B","Current, common125","Same patients; different training")
    old=read(out,"legacy/old_s5_primary_summary.tsv");old.to_csv(out/"source_tables/Supplement_N2C_oldS5_context.tsv",sep="\t",index=False)
    axs[2].errorbar(old.delta_auroc,np.arange(4)[::-1],xerr=np.vstack([old.delta_auroc-old.delta_lo,old.delta_hi-old.delta_auroc]),fmt="o",color=GRAY,capsize=3,ms=4,lw=1);axs[2].axvline(0,color=GRAY,ls="--",lw=.7);axs[2].set_yticks(np.arange(4)[::-1],[_private_literal('private_742ece63f29fe9c2'),"External163","Home → External","External → Home"]);axs[2].set_xlim(-.05,.06);axs[2].set_xlabel("Historical stack ΔAUROC");decorate(axs[2],"C","Old S5: separate context",_private_literal('private_ee39e252b75ee988'))
    save(fig,out,"Supplement_N2")


def documentation(out,feat,home,ext):
    main=read(out,"models/metrics.tsv");main=main[main.prediction=="three_seed_mean"]
    hist=read(out,"historical_four_arm_recomputed.tsv","reframing");current=read(out,"current_ridge_recomputed.tsv","reframing")
    a=read(out,"association/primary_associations.tsv")
    main.to_csv(out/"source_tables/current_primary_metrics.tsv",sep="\t",index=False)
    pd.concat([hist,current],ignore_index=True).to_csv(out/"source_tables/protocol_separated_recomputed_metrics.tsv",sep="\t",index=False)
    definitions=[
        ("Endpoint","recorded N0/N+; clinical/pathological/post-treatment origin unverified; not automatically pN"),
        ("Unit","one patient; unknown N excluded from comparisons and models; libraries linked separately"),
        ("M0","age + sex"),("M1","M0 + log1p productive QC reads + log1p unique AA + acquisition batch"),
        ("M2","normalized Shannon + top-10 read fraction + read-weighted mean CDR3 length + 64 TRBV and 14 TRBJ CLR columns"),
        ("M3","M1 + fixed M2 TCR feature block"),("SCEPTR sensitivity","M1 + full productive-repertoire read-weighted mean of existing 64-D vectors; train-fold scaling and PCA32"),
        ("Portable M1/M3","shared technical features / shared technical + fixed TCR features; no age, sex or Home batch; tuned only on Home"),
        ("LOBO","M1/M3 with batch omitted; cannot estimate a new batch offset"),
        ("Delta log-loss","LL(comparator baseline) minus LL(added TCR model); positive indicates lower log-loss"),
        ("Full-pipeline interval","200 stratified patient draws; all three seeds, preprocessing, inner C selection and outer refits rerun"),
        ("Conditional interval","resample patients from fixed predictions; excludes training and model-selection uncertainty"),
        ("Calibration coefficients","joint logistic regression of N on intercept + logit(prediction); not intercept-only calibration-in-the-large"),
        ("Common125","70 N0/55 N+; removes the duplicate patient's different library representation; fixed predictions only, not retrained"),
        (_private_literal('private_cb6f480a4310d09a'),"69 N0/56 N+; excludes recorded-T-missing patient; different set from common125"),
    ]
    pd.DataFrame(definitions,columns=["term","definition"]).to_csv(out/"definitions_dictionary.tsv",sep="\t",index=False)
    report=_private_literal('private_7813074349acfd98')
    (out/"methods_results.md").write_text(report)
    (out/"REPRODUCE.md").write_text("""# Reproduce nodal figures and source tables

Run on the analysis server with the existing Python runtime (no classifier fit):

```sh
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 /usr/bin/python3 code/nodal_compile.py \\
  --v1 /data/Code/cf_analysis/CRC_TCR/20261003_paper_v1 \\
  --audit /data/Code/cf_analysis/CRC_TCR/20261003_reframing_crc_vs_healthy_v1/n_methods \\
  --out /data/Code/cf_analysis/CRC_TCR/20261003_paper_v2/nodal
```

The source freeze is idempotent: it fails if an existing frozen source no longer matches. Plot panels and TSVs use only frozen copies. To render from a packaged freeze without the original projects, use `--frozen-only` with the same `--out`.

The full v1 pipeline can be rerun into a separate output directory with the copied `frozen/paper_v1/code/run_ridge.py` and `frozen/paper_v1/features/patient_features.tsv`, `patient_features_duplicate_pooled.tsv`, `sceptr_patient_means.npy`, `patient_index.tsv`: use `--bootstrap 200 --jobs 10 --negative-controls 20`, `--pooled-features`, `--sceptr-means`, `--sceptr-index`; keep OMP/OpenBLAS/MKL threads at 1. Never overwrite the frozen results. The completed original result retained one pre-fit partition repair; rerunning the corrected script may directly select the first feasible partition for that draw. `models/model_spec.json` documents seeds17/29/43, grouped outer5/inner3 and C={0.01,0.1,1,10}. The original association CLI uses `--features`, `--output`, `--rarefied-summary`, `--pooled-features`; check the copied script `--help` for exact options. Verify the original 16 tests with unittest discover in the frozen tests directory.

Historical four-arm and common125 outputs are fixed-prediction audit results, not new training. Their source predictions and audit receipts are under frozen/reframing. Re-estimating neural representation models requires their original training/input provenance; this small nodal freeze preserves evaluation evidence and does not pretend to contain every raw repertoire or original network checkpoint. The parent study's reproducibility package links those larger sources.
""")


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--v1",default=_release_path('path_66f74a6f33432568'))
    parser.add_argument("--audit",default=_release_path('path_8fc8a0cf03b2a0bd'))
    parser.add_argument("--out");parser.add_argument("--root",help="Paper-v2 root; output defaults to root/nodal")
    parser.add_argument("--frozen-only",action="store_true")
    args=parser.parse_args()
    if not args.out and not args.root:parser.error("Provide --root or --out")
    out=Path(args.out) if args.out else Path(args.root)/"nodal"
    out.mkdir(parents=True,exist_ok=True)
    for folder in ["source_tables","figures"]:(out/folder).mkdir(exist_ok=True)
    (out/"status.json").write_text(json.dumps({"state":"running","computation":"descriptive and fixed-prediction checks only; no new model fits"},indent=2)+"\n")
    if not args.frozen_only:
        manifest=[]
        for rel in V1_FILES:freeze(Path(args.v1)/rel,rel,out,"paper_v1",manifest)
        for rel in AUDIT_FILES:freeze(Path(args.audit)/rel,rel,out,"reframing",manifest)
        freeze(Path(args.audit).parent/"code/audit_N_comparison.py","audit_N_comparison.py",out,"reframing_code",manifest)
        pd.DataFrame(manifest).to_csv(out/"input_manifest.tsv",sep="\t",index=False)
    for _,row in pd.read_csv(out/"input_manifest.tsv",sep="\t").iterrows():assert sha(out/row.frozen_path)==row.SHA256
    feat,master,home,ext=verify(out);table1(out,feat,master,home,ext);figure2(out);supplementary(out);documentation(out,feat,home,ext)
    receipt={"state":"complete","checks":"PASS","patient_unit":True,"Home_all":137,"Home_known_N":126,"Home_Nplus":56,"External_all":202,"External_known_N":163,"External_Nplus":54,"common125_Nplus":55,"full_pipeline_draws":200,"full_pipeline_success":200,"new_model_fits":0,"frozen_input_files":len(pd.read_csv(out/"input_manifest.tsv",sep="\t")),"runtime":{"python":platform.python_version(),"matplotlib":matplotlib.__version__,"numpy":np.__version__,"pandas":pd.__version__},"script_SHA256":sha(__file__),"figures":["Figure2","Supplement_N1","Supplement_N2"]}
    (out/"verification_receipt.json").write_text(json.dumps(receipt,indent=2)+"\n")
    (out/"status.json").write_text(json.dumps(receipt,indent=2)+"\n")
    print(json.dumps(receipt))


if __name__=="__main__":
    main()
