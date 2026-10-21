#!/usr/bin/env python3
"""Patient-level, outcome-blind cohort and batch separability diagnostics."""
from __future__ import annotations
from release_runtime import private_literal as _private_literal, release_path as _release_path
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, balanced_accuracy_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(_release_path('path_99d8c1c6fde2a6b9'))

def sha(path: Path) -> str:
    h=hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda:f.read(8*1024*1024),b""):
            h.update(block)
    return h.hexdigest()

def bh(p):
    x=np.asarray(p,dtype=float)
    idx=np.argsort(x)
    out=np.empty(len(x),dtype=float)
    out[idx]=np.minimum.accumulate((x[idx]*len(x)/np.arange(1,len(x)+1))[::-1])[::-1]
    return np.clip(out,0,1)

def main():
    base=ROOT/"atlas"
    summary=pd.read_parquet(base/"patient_technical_summary.parquet")
    rare=pd.read_parquet(base/"equal_depth_repeats.parquet")
    gene=pd.read_parquet(base/"patient_vj_usage.parquet")
    if len(summary)!=339 or summary.patient_key.nunique()!=339:
        raise RuntimeError("atlas technical summary not complete")
    if not rare.groupby("patient_key").size().eq(3).all():
        raise RuntimeError("rarefaction repeats missing")
    rare_mean=rare.groupby("patient_key",as_index=False).agg(
        equal_depth_unique_mean=("unique_aa_after","mean"),
        equal_depth_shannon_mean=("shannon_after","mean"))
    d=summary.merge(rare_mean,on="patient_key",validate="one_to_one")
    cols=["total_reads","unique_aa","median_aa_length","reads_weighted_mean_aa_length",
          "equal_depth_unique_mean","equal_depth_shannon_mean"]
    records=[]
    for col in cols:
        home=d.loc[d.cohort.eq("home"),col].to_numpy(dtype=float)
        external=d.loc[d.cohort.eq("external"),col].to_numpy(dtype=float)
        statistic,p=mannwhitneyu(home,external,alternative="two-sided")
        records.append(dict(metric=col,home_median=float(np.median(home)),
                            external_median=float(np.median(external)),
                            home_to_external_median_ratio=float(np.median(home)/np.median(external))
                            if np.median(external)!=0 else None,
                            mann_whitney_u=float(statistic),p=float(p),
                            cliff_delta=float(2*statistic/(len(home)*len(external))-1)))
    diff=pd.DataFrame(records)
    diff["q_bh_descriptive"]=bh(diff.p)
    path=base/"between_cohort_technical.tsv"
    diff.to_csv(path,sep="\t",index=False)
    gene=gene.copy()
    gene["feature"]=gene.gene_type.astype(str)+":"+gene.gene.astype(str)
    top=gene.groupby("feature").reads.sum().nlargest(20).index
    g=gene.loc[gene.feature.isin(top)].pivot_table(index="patient_key",columns="feature",
                                                    values="fraction",aggfunc="sum",fill_value=0)
    X=d.set_index("patient_key")
    numeric=pd.DataFrame(index=X.index)
    numeric["log_total_reads"]=np.log1p(X.total_reads)
    numeric["log_unique_aa"]=np.log1p(X.unique_aa)
    numeric["median_aa_length"]=X.median_aa_length
    numeric["equal_depth_unique_mean"]=X.equal_depth_unique_mean
    numeric=numeric.join(g,how="left").fillna(0)
    if not np.isfinite(numeric.to_numpy(dtype=float)).all():
        raise RuntimeError("nonfinite cohort classifier features")
    y=X.cohort.eq("external").astype(int).to_numpy()
    cv=StratifiedKFold(n_splits=5,shuffle=True,random_state=20260930)
    pipe=make_pipeline(StandardScaler(),LogisticRegression(max_iter=500,solver="liblinear"))
    pred=cross_val_predict(pipe,numeric.to_numpy(dtype=float),y,cv=cv,method="predict_proba")[:,1]
    score=roc_auc_score(y,pred)
    out=dict(status="complete",outcome_blind=True,patients=339,home_patients=int((1-y).sum()),
             external_patients=int(y.sum()),equal_depth_reads=int(rare.equal_reads.iloc[0]),
             cohort_classifier_oof_auc=float(score),
             cohort_classifier_balanced_accuracy=float(balanced_accuracy_score(y,pred>=0.5)),
             classifier_features=list(numeric.columns),
             interpretation="Cohort predictability measures residual processing or biological shift; no claim that the batch effect has been removed.",
             processing_confounding="Home UID deduplication and External SRA/ENA input are cohort-nested and cannot be separated from cohort in these data.",
             n_patients_under_4096_unique_aa=int(d.unique_aa.lt(4096).sum()),
             n_patients_under_4096_after_equal_depth_mean=int(d.equal_depth_unique_mean.lt(4096).sum()),
             between_cohort_table_sha256=sha(path),
             patient_summary_sha256=sha(base/"patient_technical_summary.parquet"),
             equal_depth_sha256=sha(base/"equal_depth_repeats.parquet"),
             vj_usage_sha256=sha(base/"patient_vj_usage.parquet"))
    (base/"batch_report.json").write_text(json.dumps(out,indent=2,sort_keys=True)+"\n")
    print(json.dumps(out,sort_keys=True))

if __name__=="__main__":
    main()
