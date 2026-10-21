#!/usr/bin/env python3
"""Descriptive recorded-N bias of frozen full-AA Leiden clusters.

Patient is the analysis unit. Reads are assigned sequencing reads, not cells.
This script never builds graphs, embeds sequences, changes labels, or adds
association tests. Existing TCRBERT P/q fields are copied as frozen references.
"""
from __future__ import annotations
from release_runtime import private_literal as _private_literal, release_path as _release_path
import os
for name in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS","NUMEXPR_NUM_THREADS"):
    os.environ[name] = "1"
import argparse, hashlib, json, resource, time
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import pandas as pd
import pyarrow as pa
from scipy import sparse
pa.set_cpu_count(1); pa.set_io_thread_count(1)
N_VERTICES = 28_380_380
RSS_LIMIT = 8 * 1024**3
BITS = {("home","1"):1, ("home","0"):2, ("home","unknown"):4,
        ("external","1"):8, ("external","0"):16, ("external","unknown"):32}
def stamp(): return datetime.now(timezone.utc).isoformat()
def sha(path):
    h=hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda:f.read(8<<20),b""): h.update(b)
    return h.hexdigest()
def need(ok,msg):
    if not ok: raise ValueError(msg)
def load(path): return json.loads(Path(path).read_text())
def dump(path,obj):
    p=Path(path);p.parent.mkdir(parents=True,exist_ok=True)
    temp=p.with_name(p.name+".part")
    temp.write_text(json.dumps(obj,indent=2,sort_keys=True,allow_nan=False)+"\n")
    temp.replace(p)
def save(path,frame):
    p=Path(path);p.parent.mkdir(parents=True,exist_ok=True)
    temp=p.with_name(p.name+".part")
    frame.to_csv(temp,sep="\t",index=False,na_rep="NA")
    temp.replace(p)
def memcheck():
    peak=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024
    need(peak <= RSS_LIMIT,"Process exceeded 8 GiB RSS")
    return peak
def source(path, sources):
    path=Path(path)
    d=sha(path);sources[str(path)]={"sha256":d,"size":path.stat().st_size}
    return d
def validate_partition_metadata(counts):
    need(counts.get("status")=="PASS","Formal full cluster report not PASS")
    need(int(counts.get("n_vertices",-1))==N_VERTICES,"Formal report does not cover full AA bank")
    expected={(0.5,17),(1.0,17),(2.0,17),(1.0,29)}
    parts=counts.get("partitions",[])
    observed=[(float(p["resolution"]),int(p["seed"])) for p in parts]
    need(len(observed)==4 and set(observed)==expected,"All four frozen full partitions required")
    need(all(int(p["n_vertices"])==N_VERTICES for p in parts),"Partial partition cannot pass formal gate")
    return parts
def verify_index_alignment(root,sources):
    atlas=root.parent
    specs=[(atlas/"leiden_full_20261001","tcrbert_dualtask_aa_mean"),
           (atlas/"leiden_full_sceptr_20261002","sceptr_cdr3_only")]
    receipts=[]
    for modelroot,modelname in specs:
        path=modelroot/"audit/source_validation.json";v=load(path);source(path,sources)
        need(v.get("status")=="PASS" and int(v["n_vertices"])==N_VERTICES,"Model index source validation not full PASS")
        pp=atlas/"atlas/joint"/modelname/"projection_receipt.json";p=load(pp);source(pp,sources)
        need(p.get("status")=="complete" and p.get("missing")==0 and p.get("nonfinite")==0
             and int(p["n_aa"])==N_VERTICES,"Projection is incomplete")
        need(p["input_index_sha256"]==v["bank_mapping_sha256"],"Projection index SHA mismatch")
        need(v["existing_projection_receipt"]["input_index_sha256"]==p["input_index_sha256"],"Source nested index SHA mismatch")
        receipts.append(v)
    for key in ["bank_mapping_sha256","atlas_union_sha256","bank_mapping_path","atlas_union_path"]:
        need(receipts[0][key]==receipts[1][key],"TCRBERT/SCEPTR seq_idx origin mismatch: "+key)
    for pathkey,hashkey in [("bank_mapping_path","bank_mapping_sha256"),("atlas_union_path","atlas_union_sha256")]:
        need(source(receipts[0][pathkey],sources)==receipts[0][hashkey],"Current seq_idx index source hash differs")
    return {"status":"PASS","bank_mapping_sha256":receipts[0]["bank_mapping_sha256"],
            "atlas_union_sha256":receipts[0]["atlas_union_sha256"],
            "both_projection_input_index_sha256":receipts[0]["bank_mapping_sha256"]}
def gate(root,model,sources):
    runner=load(root/"audit/runner_state.json")
    need(runner.get("status")=="COMPLETE","Model runner is not COMPLETE: "+model)
    final_path=root/("audit/serial_final_acceptance.json" if model=="TCRBERT" else "audit/final_acceptance.json")
    final=load(final_path);need(final.get("status")=="PASS","Model final acceptance is not PASS")
    counts_path=root/"report/full_cluster_counts_by_resolution.json"
    counts=load(counts_path);parts=validate_partition_metadata(counts)
    for p in parts:
        lp=Path(p["labels_path"])
        need(lp.resolve().parent==(root/"clusters").resolve(),"Partition labels outside frozen model root")
        need(source(lp,sources)==p["labels_sha256"],"Formal partition label SHA mismatch")
        partition_labels=np.load(lp,mmap_mode="r")
        need(partition_labels.shape==(N_VERTICES,) and partition_labels.dtype.kind in "iu","Partial partition file")
        sizes=np.bincount(partition_labels)
        need(np.all(sizes>0) and int(sizes.sum())==N_VERTICES and len(sizes)==int(p["n_clusters"]),
             "Formal partition actual cluster counts differ")
        del partition_labels,sizes
    primary=[p for p in parts if float(p["resolution"])==1 and int(p["seed"])==17]
    need(len(primary)==1,"Missing unique frozen primary partition")
    part=primary[0];need(int(part["n_vertices"])==N_VERTICES,"Partial label partition rejected")
    labels_path=Path(part["labels_path"])
    need(labels_path.resolve()==(root/"clusters/primary_labels.npy").resolve(),"Unexpected main labels")
    got=source(labels_path,sources);need(got==part["labels_sha256"],"Main labels SHA differs from formal report")
    receipt_path=root/"clusters/primary_receipt.json"
    receipt=load(receipt_path);need(receipt.get("status")=="PASS","Primary receipt not PASS")
    need(receipt["labels_sha256"]==got,"Primary receipt label hash mismatch")
    need(int(receipt["n_vertices"])==N_VERTICES and int(receipt["n_clusters"])==int(part["n_clusters"]),"Primary receipt partial nodes/counts")
    receipt_sha=source(receipt_path,sources)
    if model=="TCRBERT":
        need(receipt_sha==part["receipt_sha256"],"Formal report primary receipt SHA mismatch")
        need(final["full_labels_sha256"]==got,"Final acceptance primary SHA mismatch")
    else:
        fp=validate_partition_metadata(final)
        need([p for p in fp if float(p["resolution"])==1 and int(p["seed"])==17][0]["labels_sha256"]==got,
             "SCEPTR final acceptance primary label SHA mismatch")
    for p in [root/"audit/runner_state.json",final_path,counts_path,receipt_path]:
        source(p,sources)
    labels=np.load(labels_path,mmap_mode="r")
    need(labels.shape==(N_VERTICES,) and labels.dtype.kind in "iu","Invalid label format")
    nclusters=int(part["n_clusters"])
    need(int(labels.min())==0 and int(labels.max())==nclusters-1,"Non-contiguous labels")
    vertex=np.bincount(labels,minlength=nclusters)
    need(int(vertex.sum())==N_VERTICES and np.all(vertex>0),"Missing AA labels")
    return labels,vertex,got,part
def delta(x,y):
    return ((x[:,None,:]>y[None,:,:]).sum(axis=(0,1)) -
            (x[:,None,:]<y[None,:,:]).sum(axis=(0,1))) / (len(x)*len(y))
def bootstrap(x,y,ix,iy):
    # Whole patients are resampled within recorded-N classes; ties count zero.
    wx=np.stack([np.bincount(z,minlength=len(x)) for z in ix]).astype(float)
    wy=np.stack([np.bincount(z,minlength=len(y)) for z in iy]).astype(float)
    signs=(x[:,None,:]>y[None,:,:]).astype(float)-(x[:,None,:]<y[None,:,:]).astype(float)
    out=np.empty((len(ix),x.shape[1]))
    for g in range(x.shape[1]):
        out[:,g]=np.sum((wx @ signs[:,:,g])*wy,axis=1)/(len(x)*len(y))
    md=np.median(x[ix],axis=1)-np.median(y[iy],axis=1)
    xp=x[ix].mean(axis=1); ym=y[iy].mean(axis=1)
    mean_diff=xp-ym
    score=safe_ratio(xp,xp+ym)
    score_ci=np.full((2,x.shape[1]),np.nan)
    score_valid=np.sum(np.isfinite(score),axis=0)
    for g in range(x.shape[1]):
        if score_valid[g]: score_ci[:,g]=np.quantile(score[np.isfinite(score[:,g]),g],[.025,.975])
    return (np.quantile(out,[.025,.975],axis=0),np.quantile(md,[.025,.975],axis=0),
            np.quantile(mean_diff,[.025,.975],axis=0),score_ci,score_valid)
def qr(x):
    return np.quantile(x,[.25,.5,.75],axis=0)
def source_counts(bitmap,labels,cohort,nclusters):
    plus,minus,unknown=[BITS[(cohort,n)] for n in ["1","0","unknown"]]
    masks={"Nplus_present":lambda a:(a&plus)>0,
           "N0_present":lambda a:(a&minus)>0,
           "Nplus_only_vs_N0":lambda a:((a&plus)>0)&((a&minus)==0),
           "N0_only_vs_Nplus":lambda a:((a&minus)>0)&((a&plus)==0),
           "shared_Nplus_N0":lambda a:((a&plus)>0)&((a&minus)>0),
           "known_union":lambda a:(a&(plus|minus))>0,
           "N_unknown_present":lambda a:(a&unknown)>0,
           "N_unknown_only":lambda a:((a&unknown)>0)&((a&(plus|minus))==0),
           "cohort_union":lambda a:(a&(plus|minus|unknown))>0}
    counts={k:np.zeros(nclusters,dtype=np.int64) for k in masks}
    for start in range(0,len(bitmap),2_000_000):
        b=bitmap[start:start+2_000_000]; lab=labels[start:start+2_000_000]
        for key,fn in masks.items():
            counts[key]+=np.bincount(lab[fn(b)],minlength=nclusters)
    need(np.array_equal(counts["Nplus_only_vs_N0"]+counts["N0_only_vs_Nplus"]+
                        counts["shared_Nplus_N0"],counts["known_union"]),"Source overlap partition failed")
    need(np.array_equal(counts["known_union"]+counts["N_unknown_only"],counts["cohort_union"]),
         "Unknown-source partition failed")
    return counts
def safe_ratio(a,b):
    return np.divide(a,b,out=np.full(np.shape(a),np.nan,dtype=float),where=np.asarray(b)>0)
def median_ratio(a,b):
    a=np.asarray(a,dtype=float);b=np.asarray(b,dtype=float)
    value=safe_ratio(a,b)
    value=np.where((a>0)&(b==0),np.inf,value)
    status=np.select([(a==0)&(b==0),(a>0)&(b==0),(a==0)&(b>0)],
                     ["both_zero_undefined","positive_over_zero_infinite","zero_over_positive"],default="finite")
    return value,status
def summaries(cohort,identity,y,unique,reads,totals,origin,vertex,args):
    sel=(identity.cohort.eq(cohort).to_numpy()) & np.isin(y,["0","1"])
    pos=np.flatnonzero(sel & (y=="1")); neg=np.flatnonzero(sel & (y=="0"))
    up=unique[pos]/totals.total_unique_aa.to_numpy()[pos,None]
    un=unique[neg]/totals.total_unique_aa.to_numpy()[neg,None]
    rp=reads[pos]/totals.total_reads.to_numpy()[pos,None]
    rn=reads[neg]/totals.total_reads.to_numpy()[neg,None]
    rng=np.random.default_rng(args.seed+(0 if cohort=="home" else 1))
    ix=rng.integers(len(pos),size=(args.bootstrap,len(pos)))
    iy=rng.integers(len(neg),size=(args.bootstrap,len(neg)))
    uci,umci,umeanci,uscoreci,uscorevalid=bootstrap(up,un,ix,iy)
    rci,rmci,_,_,_=bootstrap(rp,rn,ix,iy)
    uqpos,uqneg=qr(up),qr(un);rqpos,rqneg=qr(rp),qr(rn)
    df=pd.DataFrame({"model":args.model,"cohort":cohort,"cluster_id":np.arange(len(vertex)),
                     "global_unique_AA":vertex,"n_Nplus":len(pos),"n_N0":len(neg),
                     "n_N_unknown":int(np.sum(identity.cohort.eq(cohort).to_numpy() & (y=="unknown"))),
                     "Nplus_patient_carriers":np.sum(unique[pos]>0,axis=0),
                     "N0_patient_carriers":np.sum(unique[neg]>0,axis=0),
                     "read_fraction_cliff_delta":delta(rp,rn),"unique_fraction_cliff_delta":delta(up,un),
                     "read_fraction_cliff_delta_ci_low":rci[0],"read_fraction_cliff_delta_ci_high":rci[1],
                     "unique_fraction_cliff_delta_ci_low":uci[0],"unique_fraction_cliff_delta_ci_high":uci[1],
                     "read_fraction_median_difference":rqpos[1]-rqneg[1],
                     "read_fraction_median_difference_ci_low":rmci[0],"read_fraction_median_difference_ci_high":rmci[1],
                     "unique_fraction_median_difference":uqpos[1]-uqneg[1],
                     "unique_fraction_median_difference_ci_low":umci[0],"unique_fraction_median_difference_ci_high":umci[1],
                     "read_fraction_mean_Nplus":rp.mean(axis=0),"read_fraction_mean_N0":rn.mean(axis=0),
                     "unique_fraction_mean_Nplus":up.mean(axis=0),"unique_fraction_mean_N0":un.mean(axis=0),
                     "unique_AA_patient_occurrences_Nplus":unique[pos].sum(axis=0),
                     "unique_AA_patient_occurrences_N0":unique[neg].sum(axis=0),
                     "assigned_reads_Nplus":reads[pos].sum(axis=0),"assigned_reads_N0":reads[neg].sum(axis=0)})
    meanup=up.mean(axis=0); meanun=un.mean(axis=0)
    df["unique_fraction_mean_difference"]=meanup-meanun
    df["unique_fraction_mean_difference_ci_low"]=umeanci[0]
    df["unique_fraction_mean_difference_ci_high"]=umeanci[1]
    df["unique_fraction_patient_balanced_Nplus_score"]=safe_ratio(meanup,meanup+meanun)
    df["unique_fraction_patient_balanced_Nplus_score_null"]=0.5
    df["unique_fraction_patient_balanced_Nplus_score_minus_null"]=df.unique_fraction_patient_balanced_Nplus_score-0.5
    df["unique_fraction_patient_balanced_Nplus_score_ci_low"]=uscoreci[0]
    df["unique_fraction_patient_balanced_Nplus_score_ci_high"]=uscoreci[1]
    df["unique_fraction_patient_balanced_Nplus_score_bootstrap_valid_count"]=uscorevalid
    occurrence_plus=unique[pos].sum(axis=0);occurrence_minus=unique[neg].sum(axis=0)
    df["AA_patient_occurrence_Nplus_share"]=safe_ratio(occurrence_plus.astype(float),occurrence_plus+occurrence_minus)
    df["AA_patient_occurrence_Nplus_share_null_class_prevalence"]=len(pos)/(len(pos)+len(neg))
    df["AA_patient_occurrence_Nplus_share_minus_null"]=df.AA_patient_occurrence_Nplus_share-len(pos)/(len(pos)+len(neg))
    for name,p,m in [("unique_fraction",up,un),("read_fraction",rp,rn),
                     ("raw_unique_AA",unique[pos],unique[neg]),("raw_assigned_reads",reads[pos],reads[neg])]:
        qp,qm=qr(p),qr(m)
        for key,q in [("Nplus",qp),("N0",qm)]:
            for index,level in enumerate(["q25","median","q75"]):
                df[name+"_"+level+"_"+key]=q[index]
    for name,cnt in origin.items(): df["distinct_AA_"+name]=cnt
    a=origin["Nplus_present"].astype(float);b=origin["N0_present"].astype(float)
    df["distinct_AA_Nplus_source_incidence_share"]=safe_ratio(a,a+b)
    df["distinct_AA_shared_fraction_of_known_union"]=safe_ratio(origin["shared_Nplus_N0"].astype(float),origin["known_union"])
    baseline=a.sum()/(a.sum()+b.sum())
    df["distinct_AA_source_share_global_baseline"]=baseline
    df["distinct_AA_source_share_minus_global_baseline"]=df.distinct_AA_Nplus_source_incidence_share-baseline
    df["distinct_AA_union_per_class_patient_scaled_difference"]=a/len(pos)-b/len(neg)
    df["read_fraction_median_ratio_Nplus_over_N0"],df["read_fraction_median_ratio_status"]=median_ratio(rqpos[1],rqneg[1])
    df["unique_fraction_median_ratio_Nplus_over_N0"],df["unique_fraction_median_ratio_status"]=median_ratio(uqpos[1],uqneg[1])
    for feature in ["read_fraction","unique_fraction"]:
        df[feature+"_median_Cliff_direction_inconsistent"]=(np.sign(df[feature+"_median_difference"])!=np.sign(df[feature+"_cliff_delta"]))
        df[feature+"_median_Cliff_direction_opposite"]=(df[feature+"_median_difference"]*df[feature+"_cliff_delta"]<0)
    df["read_bias_direction"]=np.select([df.read_fraction_cliff_delta>0,df.read_fraction_cliff_delta<0],
                                        ["Nplus_higher","N0_higher"],default="tie")
    df["read_bias_ci_supports_direction"]=((df.read_fraction_cliff_delta_ci_low>0) |
                                          (df.read_fraction_cliff_delta_ci_high<0))
    df["bias_is_descriptive_not_predictive_contribution"]=True
    # Stable deterministic rank. No labels are used to change the clustering.
    ordered=df.sort_values(["read_fraction_cliff_delta","unique_fraction_cliff_delta","cluster_id"],
                           ascending=[False,False,True])
    ranks={int(g):i+1 for i,g in enumerate(ordered.cluster_id)}
    df["rank_read_fraction_bias"]=df.cluster_id.map(ranks)
    ordered=df.sort_values(["unique_fraction_cliff_delta","read_fraction_cliff_delta","cluster_id"],
                           ascending=[False,False,True])
    df["rank_equal_patient_AA_bias"]=df.cluster_id.map({int(g):i+1 for i,g in enumerate(ordered.cluster_id)})
    ordered=df.sort_values(["read_fraction_median_difference","unique_fraction_median_difference","cluster_id"],
                           ascending=[False,False,True])
    df["rank_median_read_fraction_difference"]=df.cluster_id.map({int(g):i+1 for i,g in enumerate(ordered.cluster_id)})
    ordered=df.sort_values(["distinct_AA_source_share_minus_global_baseline","cluster_id"],ascending=[False,True])
    df["rank_distinct_AA_source_descriptive"]=df.cluster_id.map({int(g):i+1 for i,g in enumerate(ordered.cluster_id)})
    return df.sort_values("rank_read_fraction_bias"),(len(pos),len(neg)),baseline
def copy_effects(frame, root,args,sources):
    # Reuse frozen association family; never compute a new p-value.
    for scale in ["carrier","read_fraction_log_per_sd","unique_fraction_log_per_sd"]:
        for name in ["p","q_bh_all_predefined_tests","q_bh_scale_cohort","odds_ratio","ci_lower","ci_upper","status"]:
            frame["frozen_"+scale+"_"+name]=np.nan if name!="status" else "not_estimated"
    if args.model=="TCRBERT":
        path=root/"stats/associations_all_predefined_tests.parquet";source(path,sources)
        effects=pd.read_parquet(path)
        for scale in ["carrier","read_fraction_log_per_sd","unique_fraction_log_per_sd"]:
            sub=effects.loc[effects.cohort.eq(frame.cohort.iloc[0]) & effects.scale.eq(scale)]
            need(not sub.cluster_id.duplicated().any(),"Duplicated frozen effect rows")
            sub=sub.set_index("cluster_id")
            for name in ["p","q_bh_all_predefined_tests","q_bh_scale_cohort","odds_ratio","ci_lower","ci_upper","status"]:
                frame["frozen_"+scale+"_"+name]=frame.cluster_id.map(sub[name]).fillna("not_in_frozen_test_family") if name=="status" else frame.cluster_id.map(sub[name])
        frame["frozen_association_provenance"]="existing_TCRBERT_primary_frozen_137_test_family"
    else: frame["frozen_association_provenance"]="not_estimated_no_new_tests"
    return frame
def tests():
    x=np.array([[0.,1.,0.],[0.,1.,2.]])
    z=np.array([[0.,0.,1.],[0.,0.,0.]])
    d=delta(x,z)
    need(np.array_equal(d,np.array([0.,1.,0.25])),"Cliff delta tie/direction test failed")
    # AA 1 is shared between N+ and N0 and AA 2 overlaps unknown.
    bm=np.array([1,3,5,4,8,24,32],dtype=np.uint8);labs=np.array([0,0,0,0,1,1,1])
    c=source_counts(bm,labs,"home",2)
    need(c["Nplus_present"].tolist()==[3,0] and c["shared_Nplus_N0"].tolist()==[1,0],"Shared AA test")
    need(c["N_unknown_only"].tolist()==[1,0],"Unknown AA test")
    merged={}
    for rows in [[("AAA",2),("BBB",4)],[("AAA",3),("CCC",5)]]:
        for aa,n in rows:merged[aa]=merged.get(aa,0)+n
    need(len(merged)==3 and sum(merged.values())==14 and merged["AAA"]==5,"Same-patient library union test")
    need(np.median([0,0,5])==0,"Zero-inclusive median test")
    ix=np.array([[0,1],[1,0]]);iy=np.array([[0,1],[1,0]])
    ci,_,mean_ci,score_ci,valid=bootstrap(x,z,ix,iy)
    need(np.allclose(ci[0],d) and np.allclose(ci[1],d),"Bootstrap whole patient test")
    need(np.allclose(mean_ci[0],x.mean(axis=0)-z.mean(axis=0)),"Mean difference bootstrap")
    need(np.isnan(score_ci[0,0]) and valid[0]==0 and np.allclose(score_ci[:,1],1),"Balanced score zero test")
    rat,states=median_ratio(np.array([0,1,0,2]),np.array([0,0,2,1]))
    need(np.isnan(rat[0]) and np.isinf(rat[1]) and np.array_equal(rat[2:],np.array([0.,2.])),
         "Undefined/infinite/finite median ratios")
    need(list(states)==["both_zero_undefined","positive_over_zero_infinite","zero_over_positive","finite"],
         "Median ratio statuses")
    need(56/(56+70)==4/9 and 54/(54+109)==54/163,"AA-patient null class prevalence")
    fixture={"status":"PASS","n_vertices":N_VERTICES,"partitions":[
        {"resolution":r,"seed":sd,"n_vertices":N_VERTICES} for r,sd in [(0.5,17),(1.,17),(2.,17),(1.,29)]]}
    validate_partition_metadata(fixture)
    for altered in [{**fixture,"partitions":fixture["partitions"][:-1]},
                    {**fixture,"partitions":[{**v,"n_vertices":100} if i==0 else v for i,v in enumerate(fixture["partitions"])]}]:
        try:validate_partition_metadata(altered)
        except ValueError:pass
        else:raise AssertionError("Partial partition fixture must fail gate")
    return {"status":"PASS","tests":["ties_and_direction","shared_AA","unknown_AA","same_patient_union_and_reads",
                                     "zero_inclusive_median","within_class_patient_bootstrap","mean_difference_and_balanced_score",
                                     "median_ratio_undefined_infinite","source_null_class_prevalence","four_partition_gate_rejects_partial"]}
def run(args):
    start=time.monotonic();out=args.output_root.resolve()
    for folder in ["src","config","stats","report","audit"]: (out/folder).mkdir(parents=True,exist_ok=True)
    selfcheck=tests()
    selfcheck.update(time_utc=stamp(),script_sha256=sha(__file__))
    dump(out/"audit/selftest.json",selfcheck)
    if args.self_test: print(json.dumps(selfcheck));return
    source_root=args.source_root or args.root.parent/"leiden_full_20261001"
    args.source_root=source_root.resolve()
    sources={}; labels,vertex,labelsha,partition=gate(args.root,args.model,sources)
    index_alignment=verify_index_alignment(args.root,sources)
    final=out/"audit"/(args.model.lower()+"_completion.json")
    design={"version":2,"model":args.model,"endpoint":"recorded N+ versus recorded N0",
            "patient_equal_AA_score":"S=mean(uniqueFraction_Nplus)/(mean(uniqueFraction_Nplus)+mean(uniqueFraction_N0)); equal class means; null0.5; 0/0 NA",
            "patient_equal_AA_score_CI":"same 2000 class-stratified whole-patient draws; percentile95%; undefined bootstrap scores excluded and valid count reported",
            "AA_patient_occurrence_share":"sum cluster uniqueAA occurrences Nplus / sum occurrences Nplus+N0; descriptive; null=nplus/(nplus+n0)",
            "median_ratio_status":"both_zero_undefined NA, positive_over_zero_infinite inf, zero_over_positive0, finite",
            "median_Cliff_direction_flags":"inconsistent if signs differ, including median ties; opposite if product<0",
            "partition":"frozen resolution1 seed17; no outcome-driven regrouping","main_rank":"descending patient read-fraction Cliff delta, tie unique-fraction Cliff delta, then ascending cluster ID",
            "zero_policy":"include absent-cluster zero values for all known-N patients",
            "formal_gate":"runner COMPLETE; final acceptance PASS; all four frozen complete partitions and actual label hashes/counts; primary receipt binding",
            "seq_idx_alignment":"both source_validation PASS and common bank_mapping/atlas_union hashes; both projection input_index SHA equal",
            "distinct_union_scaled_difference":"Nplus distinct group union / nplus patients minus N0 distinct union / n0 patients; NOT mean patient AA count",
            "AA_origin":"exact seq_idx bitmap; shared AA retained explicitly, never assigned a single phenotype",
            "bootstrap":args.bootstrap,"bootstrap_seed":args.seed,"bootstrap_unit":"whole patient stratified within N class; 95% percentile intervals",
            "cohorts":"Home and External separately; External shared-atlas exploratory recheck",
            "P_q":"existing TCRBERT frozen reference only; SCEPTR NA and not estimated",
            "resource":{"threads":1,"rss_limit_bytes":RSS_LIMIT,"gpu":False},
            "read_unit":"MiXCR reads assigned to clonotype; not cell count",
            "sources":["https://mixcr.com/mixcr/reference/mixcr-export/","https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.mannwhitneyu.html"]}
    dump(out/"config"/(args.model.lower()+"_bias_design.json"),design)
    maprec=load(source_root/"stats/patient_mapping_receipt.json")
    need(maprec.get("status")=="PASS" and maprec["n_patients"]==339 and maprec["n_libraries"]==340,"Patient mapping not full PASS")
    source(source_root/"stats/patient_mapping_receipt.json",sources)
    for filename in ["patient_identity.parquet","patient_totals.parquet","patient_cluster_unique_counts.npz","patient_cluster_read_counts.npz","library_mapping_qc.tsv"]:
        got=source(source_root/"stats"/filename,sources)
        need(got==maprec["output_sha256"][filename],"Frozen patient source hash mismatch "+filename)
    identity=pd.read_parquet(source_root/"stats/patient_identity.parquet")
    totals=pd.read_parquet(source_root/"stats/patient_totals.parquet")
    need(identity.patient_key.is_unique and np.array_equal(identity.patient_key,totals.patient_key),"Patient order differs")
    manifest=args.root.parent/"config/cohort_samples.tsv"
    source(manifest,sources)
    metadata=pd.read_csv(manifest,sep="\t",dtype=str,keep_default_na=False)
    need(len(metadata)==340 and metadata.patient_key.nunique()==339,"Cohort library identity mismatch")
    for key,g in metadata.groupby("patient_key",sort=False):
        need(g.recorded_n_binary.nunique()==1,"Same-patient N disagreement: "+key)
    meta=metadata.drop_duplicates("patient_key").set_index("patient_key")
    need(set(meta.index)==set(identity.patient_key),"Metadata patient mismatch")
    raw=meta.loc[identity.patient_key].recorded_n_binary.to_numpy()
    y=np.where(np.isin(raw,["0","1"]),raw,"unknown")
    need(np.sum((identity.cohort.to_numpy()=="home")&(y=="1"))==56 and
         np.sum((identity.cohort.to_numpy()=="home")&(y=="0"))==70 and
         np.sum((identity.cohort.to_numpy()=="external")&(y=="1"))==54 and
         np.sum((identity.cohort.to_numpy()=="external")&(y=="0"))==109,"Known N cohort counts changed")
    duplicate=metadata.groupby("patient_key").size()
    need((duplicate>1).sum()==1 and int(duplicate.max())==2,"Unexpected same-patient libraries")
    same_key=duplicate.index[duplicate>1][0]
    same_samples=set(metadata.loc[metadata.patient_key.eq(same_key)].sample_id)
    need(same_samples=={_private_literal('private_52085b4d67e616fe'),_private_literal('private_db9850ff5b76a6f9')},_private_literal('private_f2bfa252702fa806'))
    nclusters=len(vertex);uc=np.zeros((339,nclusters),dtype=np.int64);rc=uc.copy()
    bitmap=np.zeros(N_VERTICES,dtype=np.uint8)
    for i,p in identity.iterrows():
        path=source_root/"stats/patient_nodes"/(p.patient_key+".parquet")
        receipt_path=path.with_suffix(".receipt.json")
        receipt=load(receipt_path);need(receipt.get("status")=="PASS","Patient node receipt not PASS")
        need(receipt["patient_key"]==p.patient_key and receipt["cohort"]==p.cohort,"Patient receipt identity differs")
        need(receipt["labels_sha256"]==maprec["labels_sha256"],"Source node label order receipt mismatch")
        got=source(path,sources);need(got==receipt["nodes_sha256"],"Patient node file hash mismatch")
        source(receipt_path,sources)
        df=pd.read_parquet(path,columns=["seq_idx","cluster_id","read_count"])
        idx=df.seq_idx.to_numpy(np.int64);rr=df.read_count.to_numpy(np.int64)
        need(len(idx)==len(np.unique(idx)) and len(idx)==int(receipt["n_unique_aa"]),"Patient AA not unique")
        need(len(idx)>0 and idx.min()>=0 and idx.max()<N_VERTICES and np.all(rr>0),"Invalid patient idx/count")
        need(int(rr.sum())==int(receipt["total_reads"]),"Patient read conservation")
        cid=np.asarray(labels[idx])
        if args.model=="TCRBERT": need(np.array_equal(cid,df.cluster_id.to_numpy()),"TCR labels differ from original nodes")
        uc[i]=np.bincount(cid,minlength=nclusters)
        np.add.at(rc[i],cid,rr)
        need(int(uc[i].sum())==int(totals.iloc[i].total_unique_aa) and
             int(rc[i].sum())==int(totals.iloc[i].total_reads),"Mapped patient denominator not conserved")
        bitmap[idx] |= BITS[(p.cohort,y[i])]
        memcheck()
        if (i+1)%50==0:print(json.dumps({"stage":"patient_nodes","completed":i+1,"total":339,"rss_peak":memcheck()}),flush=True)
    need(np.all(bitmap>0),"Some full-atlas AA have no patient origin")
    if args.model=="TCRBERT":
        olduc=sparse.load_npz(source_root/"stats/patient_cluster_unique_counts.npz").toarray()
        oldrc=sparse.load_npz(source_root/"stats/patient_cluster_read_counts.npz").toarray()
        need(np.array_equal(uc,olduc) and np.array_equal(rc,oldrc),"Recomputed matrix differs from frozen source")
    srcrows=[]; frames={}; baselines={}
    for cohort in ["home","external"]:
        origin=source_counts(bitmap,labels,cohort,nclusters)
        f,counts,baseline=summaries(cohort,identity,y,uc,rc,totals,origin,vertex,args)
        frames[cohort]=copy_effects(f,args.root,args,sources);baselines[cohort]=baseline
        for g in range(nclusters):srcrows.append({"model":args.model,"cohort":cohort,"cluster_id":g,
                                                **{k:int(v[g]) for k,v in origin.items()}})
    both=frames["home"][["cluster_id","read_fraction_cliff_delta","unique_fraction_cliff_delta"]].merge(
         frames["external"][["cluster_id","read_fraction_cliff_delta","unique_fraction_cliff_delta"]],
         on="cluster_id",suffixes=("_home","_external"))
    both["read_bias_same_direction"]=(np.sign(both.read_fraction_cliff_delta_home)==
                                      np.sign(both.read_fraction_cliff_delta_external))
    both["read_bias_Nplus_positive_both"]=(both.read_fraction_cliff_delta_home>0)&(both.read_fraction_cliff_delta_external>0)
    long=[]
    for i,p in identity.iterrows():
        for g in range(nclusters):
            long.append({"model":args.model,"cohort":p.cohort,"patient_key":p.patient_key,"recorded_N":y[i],
                         "cluster_id":g,"cluster_unique_AA":int(uc[i,g]),"cluster_assigned_reads":int(rc[i,g]),
                         "patient_total_unique_AA":int(totals.iloc[i].total_unique_aa),
                         "patient_total_assigned_reads":int(totals.iloc[i].total_reads),
                         "cluster_unique_fraction":uc[i,g]/totals.iloc[i].total_unique_aa,
                         "cluster_read_fraction":rc[i,g]/totals.iloc[i].total_reads})
    for cohort,f in frames.items():
        f=f.merge(both[["cluster_id","read_bias_same_direction","read_bias_Nplus_positive_both"]],on="cluster_id",how="left")
        frames[cohort]=f.sort_values("rank_read_fraction_bias")
        save(out/"stats"/(args.model.lower()+"_"+cohort+"_cluster_n_bias.tsv"),frames[cohort])
        for ranking,suffix in [("rank_equal_patient_AA_bias","aa_fraction_rank"),
                               ("rank_median_read_fraction_difference","median_abundance_rank"),
                               ("rank_distinct_AA_source_descriptive","distinct_aa_source_rank")]:
            save(out/"stats"/(args.model.lower()+"_"+cohort+"_"+suffix+".tsv"),f.sort_values(ranking))
    save(out/"stats"/(args.model.lower()+"_patient_cluster_abundance.tsv"),pd.DataFrame(long))
    save(out/"stats"/(args.model.lower()+"_distinct_AA_origin.tsv"),pd.DataFrame(srcrows))
    save(out/"stats"/(args.model.lower()+"_cohort_direction_concordance.tsv"),both)
    lines=["# "+args.model+" 全量 Leiden 群的记录 N 偏向性","",
           "生成时间："+stamp(),"",
           "完整 AA："+format(N_VERTICES,",")+"；冻结主分辨率 1、种子 17；群数："+str(nclusters)+"。",
           "统计单位为患者；Home 56 N+ / 70 N0，External 54 N+ / 109 N0；N 未知患者仅计入 AA 来源覆盖。",
           _private_literal('private_687bb6b20b4641ed'),"",
           "## 计算与排序","",
           "- 主排序为每名患者群 readFraction 的 Cliff delta：P(N+ 丰度大于 N0) − P(N+ 丰度小于 N0)，范围 −1 到 1，正数偏向 N+；并列按 uniqueFraction delta、群 ID 排序。",
           "- 所有中位数、IQR 和 bootstrap 包含未携带群的 0。95% 区间由类别内 2,000 次整患者重抽样得到；描述性区间未作群间多重校正。",
           "- 患者等权 AA 平衡分数 S=mean(U_N+)/(mean(U_N+)+mean(U_N0))，U 为每患者群 uniqueFraction，null=0.5；同时报告 mean(U_N+)−mean(U_N0)、S 的2000次患者 bootstrap 区间。0/0 不定义，标 NA；置信区间排除不定义的bootstrap抽样并报告有效次数。",
           "- 原始 AA–patient 出现占比=sum(cluster uniqueAA_N+)/sum(cluster uniqueAA_N+及N0)，类人数 null=n_N+/(n_N++n_N0)，Home 为56/126，External 为54/163；该比例会受深度影响，仅描述。",
           "- 中位丰度比区分0/0未定义（NA）、正数/0（inf）、0/正数（0）及有限比值；分别标记中位数差与Cliff方向不一致（含中位数并列）和方向相反。",
           "- AA 独立来源使用全库 seq_idx bitmap。N+ 与 N0 共享 AA 单列，绝不将共享 AA 强行赋予唯一标签；来源比例仍受组人数和深度影响，不能当患者级检验。组distinct AA并集/类别患者数的差单列，其值不是平均逐患者AA数量。",
           "- 主表、AA 占比排名、中位丰度排名、独立 AA 来源排名分开交付。原始 MiXCR readCount 不等于细胞数。",
           "- TCRBERT 引用原冻结 137 项 family 的调整 P/BH q；SCEPTR 若没有正式原检验则为 NA，不新增关联检验。","",
           "## 主排名前五群（描述性）","",
           "| 队列 | 群 | read delta | 95% CI | N+ 中位 read% | N0 中位 read% | unique delta |",
           "|---|---:|---:|---|---:|---:|---:|"]
    for cohort,f in frames.items():
        for r in f.head(5).itertuples(index=False):
            lines.append("|"+cohort+"|"+str(r.cluster_id)+"|"+format(r.read_fraction_cliff_delta,".3f")+"|["+
                         format(r.read_fraction_cliff_delta_ci_low,".3f")+", "+format(r.read_fraction_cliff_delta_ci_high,".3f")+"]|"+
                         format(100*r.read_fraction_median_Nplus,".3f")+"|"+format(100*r.read_fraction_median_N0,".3f")+"|"+
                         format(r.unique_fraction_cliff_delta,".3f")+"|")
    lines+=["","## 解释边界","",
            "正向排序与未校正描述性区间不等于独立预测贡献或抗原特异性。联合图含 External，因此外部结果是同图谱探索复核；当前分析没有估计无泄漏折外增益、删群重训或抗原结合。",
            "群 ID 只在各模型内有意义，TCRBERT 和 SCEPTR 的同号群不能直接视为对应群。","",
            "方法来源：[MiXCR export](https://mixcr.com/mixcr/reference/mixcr-export/)；[Mann-Whitney U 统计量](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.mannwhitneyu.html)。"]
    report=out/"report"/(args.model.lower()+"_N_BIAS_REPORT.md")
    report.write_text("\n".join(lines)+"\n")
    bitmap_sha=hashlib.sha256(bitmap.tobytes()).hexdigest()
    source_manifest=out/"audit"/(args.model.lower()+"_source_manifest.json");dump(source_manifest,sources)
    outputs=[p for p in (out/"stats").glob(args.model.lower()+"_*")]+[report,source_manifest,out/"config"/(args.model.lower()+"_bias_design.json")]
    completion={"status":"PASS","model":args.model,"completed_utc":stamp(),"n_clusters":nclusters,"n_vertices":N_VERTICES,
                "n_libraries":340,"n_patients":339,"n_home_known":126,"n_external_known":163,
                "labels_sha256":labelsha,"AA_origin_bitmap_sha256":bitmap_sha,
                "seq_idx_index_alignment":index_alignment,"four_full_partitions_hash_and_count_verified":True,
                "script_sha256":sha(__file__),"bootstrap":args.bootstrap,"seed":args.seed,"source_counts_baseline":baselines,
                "AA_source_overlap_partition_conservation":True,"patient_unique_and_read_conservation":True,
                "source_node_hashes_verified":339,_private_literal('private_48d6cc7c9d5327e9'):True,
                "no_new_association_tests":True,"old_family_reference_only":args.model=="TCRBERT",
                "elapsed_seconds":time.monotonic()-start,"peak_rss_bytes":memcheck(),"threads":1,"gpu_used":False,
                "outputs":{str(p.relative_to(out)):{"sha256":sha(p),"bytes":p.stat().st_size} for p in outputs}}
    dump(final,completion);print(json.dumps(completion),flush=True)
def main():
    p=argparse.ArgumentParser()
    p.add_argument("--root",type=Path,required=True)
    p.add_argument("--output-root",type=Path,required=True)
    p.add_argument("--source-root",type=Path)
    p.add_argument("--model",choices=["TCRBERT","SCEPTR"],required=True)
    p.add_argument("--bootstrap",type=int,default=2000)
    p.add_argument("--seed",type=int,default=20261002)
    p.add_argument("--self-test",action="store_true")
    args=p.parse_args()
    need(args.bootstrap==2000,"Frozen bootstrap count must be 2000")
    try:run(args)
    except Exception as e:
        dump(args.output_root/"audit"/(args.model.lower()+"_failure.json"),
             {"status":"FAIL","time_utc":stamp(),"type":type(e).__name__,"error":str(e),"script_sha256":sha(__file__)})
        raise
if __name__=="__main__":main()
