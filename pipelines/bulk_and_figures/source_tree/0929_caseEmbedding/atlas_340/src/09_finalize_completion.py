#!/usr/bin/env python3
"""Read frozen atlas outputs; add completion report and acceptance record only."""
from release_runtime import private_literal as _private_literal, release_path as _release_path
from pathlib import Path
from datetime import datetime, timezone
import csv, json, hashlib, math, collections

R = Path(_release_path('path_99d8c1c6fde2a6b9'))
A = R/"audit/09_completion_acceptance_20261001.json"
P = R/"report/SCIENTIFIC_COMPLETION_REPORT_20261001.md"

def j(rel): return json.loads((R/rel).read_text())
def rows(rel):
    with (R/rel).open() as f: return list(csv.DictReader(f, delimiter="\t"))
def sha(p):
    h=hashlib.sha256()
    with p.open("rb") as f:
        for b in iter(lambda:f.read(8*1024*1024),b""): h.update(b)
    return h.hexdigest()
def n(v): return float(v) if v else float("nan")
def once(p,text):
    p.parent.mkdir(parents=True,exist_ok=True)
    if p.exists():
        if p.read_text()!=text: raise RuntimeError("Refusing to replace "+str(p))
    else:
        with p.open("x") as f: f.write(text)
def link(label,rel): return "["+label+"]("+str(R/rel)+")"

c={}
lock=j("audit/07_orchestrator_source_lock.json")
c["frozen_sources_17_match"]=len(lock["source_sha256"])==17 and all(sha(R/k)==v for k,v in lock["source_sha256"].items())
c["frozen_runner_match"]=sha(R/"src/07_run_after_pilots.py")==lock["runner_sha256"]
events=[json.loads(x) for x in (R/"logs/07_after_pilots.jsonl").read_text().splitlines()]
ends=[x for x in events if x.get("state")=="end"]
c["runner_complete_no_failed_child"]=events[-1].get("state")=="complete" and events[-1].get("final_gate")=="STOP" and all(x["exit_code"]==0 for x in ends)
qc=j("audit/01_stage1_validation.json")
c["stage1_conservation_rejoin_pass"]=qc["status"]=="PASS" and qc["libraries"]==340 and qc["patients"]==339 and qc["all_sample_aa_rejoin_union"] and qc["home_duplicate_pair_bound"] and qc["new_clone_detail_qc_count_and_read_mass_rechecked"] and sha(R/"audit/01_qc_by_library.tsv")==qc["qc_library_table_sha256"]
emb=j("embeddings/verification.json")
c["prior_full_embedding_finite_validation_pass"]=emb["status"]=="PASS" and emb["n_atlas"]==28380380 and emb["n_new"]+emb["n_reused"]==emb["n_atlas"] and all(v["nonfinite"]==0 for v in emb["models"].values())
scope_status={}
for scope in ["full_home"]+["outer_"+str(i) for i in range(5)]:
    prefix="groups/"+scope+"/"
    v={k:j(prefix+rel)["status"] for k,rel in {
        "gliph2":"gliph2_projection_receipt.json",
        "tcrdist3":"tcrdist_home_projection_receipt.json",
        "integrated":"integrated_projection_receipt.json",
        "burden":"burden_receipt.json"}.items()}
    scope_status[scope]=v
    c[scope+"_dictionary_complete"]=v["gliph2"]==v["tcrdist3"]==v["integrated"]=="complete_home_scope" and v["burden"]=="complete"
for scope in ["home","joint"]:
    for model in ["sceptr_cdr3_only","tcrbert_dualtask_aa_mean"]:
        v=j("atlas/"+scope+"/"+model+"/projection_receipt.json")
        c[scope+"_"+model+"_all_aa_projection"]=v["status"]=="complete" and v["n_aa"]==28380380 and v["missing"]==v["nonfinite"]==0

fold=rows("folds/folds.tsv")
tests={}
for i in range(5):
    f=[x for x in fold if int(x["outer_fold"])==i]
    train={x["patient_key"] for x in f if x["partition"]=="train"}
    test={x["patient_key"] for x in f if x["partition"]=="test"}
    tests[i]=test
    c["outer_"+str(i)+"_split_disjoint"]=not train&test and len(train|test)==len(f)==126
c["outer_tests_unique_126"]=len(set.union(*tests.values()))==sum(len(x) for x in tests.values())==126

assoc=rows("groups/full_home/association_recorded_home.tsv")
eligible=rows("groups/full_home/eligible_groups_recorded_home.tsv")
keys=[(x["method"],x["group_id"]) for x in assoc]
c["complete_test_universe_no_duplicates"]=len(keys)==len(set(keys))==1462 and set(keys)=={(x["method"],x["group_id"]) for x in eligible}
pv=[n(x["adjusted_p"]) for x in assoc]
order=sorted(range(len(pv)),key=lambda i:pv[i])
bh=[0.0]*len(pv)
minimum=1.0
for rank in range(len(pv),0,-1):
    idx=order[rank-1]
    minimum=min(minimum,pv[idx]*len(pv)/rank)
    bh[idx]=minimum
error=max(abs(bh[i]-n(x["adjusted_q_bh_pooled"])) for i,x in enumerate(assoc))
c["pooled_bh_recomputed"]=error<1e-9
c["failed_fit_preserved_as_p1"]=all(n(x["adjusted_p"])==1 for x in assoc if x["adjusted_status"]!="ok")
positive=[x for x in assoc if n(x["adjusted_beta"])>0 and n(x["prevalence_difference"])>0]
stats={"n_tested":len(assoc),"by_method":dict(collections.Counter(x["method"] for x in assoc)),
       "fit_status":dict(collections.Counter(x["adjusted_status"] for x in assoc)),
       "min_p":min(pv),"min_q":min(bh),"bh_max_abs_error":error,
       "n_q_lt_0p05":sum(x<.05 for x in bh),
       "n_nominal":sum(x<.05 for x in pv),
       "n_nominal_positive":sum(n(x["adjusted_p"])<.05 for x in positive),
       "min_positive_p":min(n(x["adjusted_p"]) for x in positive),
       "min_positive_q":min(n(x["adjusted_q_bh_pooled"]) for x in positive)}
oof=rows("prediction/05_patient_oof.tsv")
c["oof_patient_outer_test_alignment"]=len(oof)==len({x["patient_key"] for x in oof})==126 and all(x["patient_key"] in tests[int(x["outer_fold"])] for x in oof)
c["oof_finite"]=all(math.isfinite(n(x[k])) and 0<n(x[k])<1 for x in oof for k in ["baseline_probability","groups_probability"])
c["matched_constant_predictions_verified"]=all(n(x["baseline_probability"])==n(x["groups_probability"])==.5 and n(x["paired_log_loss_improvement"])==0 for x in oof)
pred=j("prediction/05_summary.json")
c["paired_zero_gain_verified"]=pred["favorable_outer_folds"]==0 and all(pred["paired_baseline_minus_groups_log_loss"][k]==0 for k in ["mean","ci_low","ci_high"])
gate=j("results/05_final_candidate_gate.json")
c["final_gate_source_chain"]=all(sha(R/rel)==gate[k] for k,rel in gate["source_paths"].items())
c["valid_scoped_negative_gate"]=gate["status"]=="STOP" and gate["evidence_complete"] and gate["n_candidates"]==stats["n_q_lt_0p05"]==0 and not rows("results/05_final_candidates.tsv")
ext=j("groups/full_home/external_projection_receipt.json")
direction=j("groups/full_home/external_direction.json")
c["external_empty_frozen_projection_explicit"]=ext["empty_candidate_universe"] and not ext["external_n_read"] and ext["n_frozen_groups"]==direction["n_scored_groups"]==0 and direction["n_external_all"]==202 and direction["n_external_N_known"]==163
abl=j("results/05_ablation_receipt.json")
sens=j("results/05_sensitivity_receipt.json")
ant=j("antigen/receipt.json")
c["conditional_ablation_antigen_stop"]=abl["status"]=="STOP_no_q_candidates" and abl["n_ablation_tested"]==0 and ant["status"]=="STOP_gate_not_pass" and ant["n_exact_records"]==0 and sens["n_candidates"]==0 and not rows("antigen/evidence_exact.tsv") and not rows("antigen/enrichment.tsv")
comp=j("representation_comparison/report/summary.json")
c["representation_screen_stop_verified"]=comp["status"]=="complete_screen" and not comp["external_used_for_selection"] and not comp["third_full_bank_allowed_now"] and all(x["n_verified_fits"]==60 for x in comp["verified_model_sources"].values())
if not all(c.values()): raise RuntimeError("Failed checks: "+str([k for k,v in c.items() if not v]))
utc=j("audit/09_completion_acceptance_20261001.json")["verified_utc"] if A.exists() else datetime.now(timezone.utc).isoformat()
labels={"kmer3_cosine_svd":"3-mer + SVD","cdr3vec_small":"CDR3vec_small","sceptr_cdr3_only":"SCEPTR CDR3-only","tcrbert_dualtask_aa_mean":"TCR-BERT CDR3-only"}
table=[]
for name in comp["ranking_by_oof_log_loss"]:
    x=comp["models"][name]; m=x["metrics"]; d=x["paired_l2"]
    table.append(f'| {labels[name]} | {m["log_loss"]:.4f} | {d["mean"]:+.4f} [{d["ci_low"]:+.4f}, {d["ci_high"]:+.4f}] | {m["brier"]:.4f} | {m["auroc"]:.3f} | {m["ap"]:.3f} | {m["calibration_intercept"]:+.3f} / {m["calibration_slope"]:.3f} | 1/5 |')

report=f"# atlas_340：全外周血 TRB 图谱与记录 N+ 分析结案\n\n核验时间（UTC）：{utc}  \n正式串行完成：2026-09-30 22:07:30 UTC（2026-10-01 06:07:30 北京时间）。  \n输出根目录：{R}{_private_literal('private_32915f1f5103aed9')}{link('逐文库 QC', 'audit/01_qc_by_library.tsv')}、{link('计数验收', 'audit/01_stage1_validation.json')}、{link('MiXCR 契约', 'audit/01_source_contract.json')}、{link('批次报告', 'atlas/batch_report.json')}、{link('向量批次诊断', 'atlas/vector_batch_receipt.json')}。\n\n## 1. 全量表示与公平模型筛选\n\n两套全量向量完成并保留：TCR-BERT 768 维、SCEPTR CDR3-only 64 维。每套复用 23,880,428、新增 4,499,952，共覆盖全部 28,380,380 AA；全量验证回执非有限值为 0。完整 AA 精确回联旧/新增银行，原 279 例结果保留。\n\n丰度是独立的 MiXCR readCount 属性。每个克隆型获得一条向量；bulk 克隆型和读数不能解释为一个细胞或细胞数。\n\nCDR3vec_small 通过 50,000 条长度分层成本试测，随后仅编码共同抽样池。该小模型仍属 Transformer，容量/训练目标不同；3-mer + SVD 是非预训练对照。四臂共用患者、每人 4,096 AA 池、2,048 训练及对应推理 bag、丰度、MIL、5 外折/3 内折/3 种子与训练折 PCA/SVD。每臂完成 60 个拟合。External 未用于选型。\n\n结果揭盲前预设的 L2 临床/技术敏感性基线：log loss 0.6813、Brier 0.2441、AUROC 0.579、AP 0.535。患者配对改善定义为同患者基线 log loss 减去模型 log loss，95% CI 为固定患者折条件下 5,000 次患者配对 bootstrap。\n\n| 表示 | Log loss ↓ | 相对 L2 改善及 95% CI ↑ | Brier ↓ | AUROC | AP | 校准截距 / 斜率 | 有利外折 |\n|---|---:|---:|---:|---:|---:|---:|---:|\n{chr(10).join(table)}\n\n四臂区间均跨 0，折/种子稳定性未过门槛。CDR3vec_small 不补全库。抽样池中位 AA/reads 覆盖为 1.65%/1.55%；实际 bag 为 0.83%/0.78%；群捕获率中位 5.73%。这是稀疏筛选，不能替代全组库建群或否定完整组库信息。此比较固定 bag=2,048，不是新增的全 bag 敏感性曲线；实际等深度重抽样重训因初筛失败而未触发。\n\n来源：{link('全量向量验证', 'embeddings/verification.json')}、{link('四表示比较报告', 'representation_comparison/report/INTERPRETATION.md')}、{link('完整比较指标', 'representation_comparison/report/summary.json')}。\n\n## 2. 降维、序列群与覆盖范围\n\nHome 与联合 340 描述图谱的两套 PCA64/全部 AA 投射完成，missing/nonfinite 均为 0；UMAP 仅用于平衡子集展示。Home 与 5 个外层训练集分别冻结独立序列群字典。留出患者只接受投射，5 外折测试集合互斥并恰覆盖 126 人。\n\n| 方案 | Home 结果 | 方法边界 |\n|---|---|---|\n| tcrdist3 | radius=24；2,399 群；24,257,167 可评分 AA×V×J 节点全部检索；6,210,363 明确分配（25.60%） | 平衡种子发现+全 Home 投射；非全对全聚类；713,243 多群歧义节点另列 |\n| GLIPH2 | 6 个 scope 正式运行成功；full Home 112,271 种子群中 40 个可投射 literal motif 群；匹配 46,402/39,068,194 克隆行（0.1188%） | GLIPH2 种子衍生 motif+V/J 投射，非完整原始群成员；其余种子群不进入本检验 |\n| 合并 | 字典 2,439 群；患者负荷初筛保留 2,429 群 | 未归群、歧义和参考不可评分者保留在覆盖分母 |\n| 表示邻域 | 4,384 Home 平衡 AA，top-50 平均重叠 16.44%、Jaccard 0.092 | 描述性一致性较低，未用于提名或调整阈值 |\n\n联合 340 图谱是全量表示的描述图谱，序列群发现范围为 Home。External 本次仅投射冻结的 q 候选全集（为空），没有生成全部 2,439 群的 External 负荷图谱。\n\n来源：{link('tcrdist3 回执', 'groups/full_home/tcrdist_home_projection_receipt.json')}、{link('GLIPH2 回执', 'groups/full_home/gliph2_projection_receipt.json')}、{link('邻域检查', 'atlas/embedding_neighborhood/receipt.json')}、{link('图谱图', 'figures/balanced_embedding_atlas.png')}。\n\n## 3. N+ 关联、贡献与停止门槛\n\n以患者携带状态为主，调整年龄、性别、读深、unique AA、Home 批次、新辅助状态。冻结资格为 N 已知 Home 患者至少 10 名携带者和 10 名非携带者。\n\n- 实际受检 **1,462 群**：tcrdist3 1,427、GLIPH2 35；资格表/结果全集完全一致。\n- **1,458** 个主拟合通过；**4** 个极端/无效拟合按预设 P=1 纳入联合 BH。\n- 名义 P<0.05 **94 群**，其中 N+ 正向 36、负向 58；仅作探索排序。\n- 最小 P={stats['min_p']:.8g}、最小 q={stats['min_q']:.6f}；正向群最小 q={stats['min_positive_q']:.6f}；**q<0.05 为 0**。\n- 独立重算 BH 最大误差 {error:.2g}；未依据 N 结果修改半径、motif 阈值或检验全集。\n- 全表保存未新辅助、AA 占比、reads 占比敏感性拟合及批次方向。候选专属等深度/批次/未治疗稳定性因候选为 0，没有可评对象；不能把候选回执中的 n_repeats=3 当作已做三个候选实验。\n\n固定外折/内折临床技术 L1 基线及加群携带模型均选 C=0.01，所有群系数为 0，所有 OOF 概率为 0.5：\n\n| 指标 | 基线 | 加群 |\n|---|---:|---:|\n| Log loss | 0.693147 | 0.693147 |\n| Brier | 0.250 | 0.250 |\n| AUROC | 0.500 | 0.500 |\n| AP | 0.444 | 0.444 |\n| 校准截距/斜率 | 不可辨识 | 不可辨识 |\n| 患者配对改善 | — | 0，95% CI [0,0]；有利外折 0/5 |\n\n**解释限制：**liblinear L1 同时惩罚截距，one-SE 选出的强惩罚将截距归零。这与冻结实现/调参规则一致，但仅说明当前携带状态模型未显示新增价值，不能推广到所有群丰度分类器或 encoder。[0,0] 来自逐患者概率完全相同，不是生物效应区间。内层验证患者的无标签序列曾参与外层训练字典；外层留出者不参与发现。区间条件于固定折，未重新执行整个发现流程。\n\nExternal 冻结候选=0，原 202 人空候选投射在读取 External N 之前完成；之后登记 163 人可评价标签范围，受评群=0。这不是外部反向或外部验证失败。\n\n删群重训/匹配随机删群：STOP_no_q_candidates，实际测试 **0**。  \n抗原精确匹配/类别富集/序列卡：STOP_gate_not_pass，实际未启动查询。四张抗原表为 0 行不代表查库无命中。本地 VDJdb、IEDB receptor、McPAS 库存在；源码约束文件名，参考文件 SHA 只在 PASS 分支计算，本次未执行实际参考版本哈希锁定或查询。没有群可支持“最多结合哪类抗原”的判断。\n\n来源：{link('完整检验表', 'groups/full_home/association_recorded_home.tsv')}、{link('名义 P 探索表', 'groups/full_home/exploratory_nominal_p_groups.tsv')}、{link('患者 OOF', 'prediction/05_patient_oof.tsv')}、{link('预测摘要', 'prediction/05_summary.json')}、{link('External 回执', 'groups/full_home/external_projection_receipt.json')}、{link('最终候选门槛', 'results/05_final_candidate_gate.json')}、{link('删群停止', 'results/05_ablation_receipt.json')}、{link('抗原停止', 'antigen/receipt.json')}。\n\n## 4. 验收、资源与可复现边界\n\n- 本轮 17 个冻结来源及 runner 哈希全部一致；新增脚本只写本报告和验收记录。\n- 独立复核 340 身份、重复绑定、全部 28,380,380 银行索引、银行 shape/旧 mtime/size、6 scope GLIPH2 输入/导入/退出码及训练字典患者集合。\n- 全量向量有限性/计数守恒采用既有全量 PASS 回执；本轮未重算原始 MiXCR、全部 GLM 或全部向量。BH、OOF 指标、one-SE 和停止链独立重算通过。\n- 当前上限 GPU 14 GiB、CPU 48 线程/RSS 512 GiB。主串行器保留更低的 16 线程/128 GiB 上限；GPU 单任务锁。CDR3vec 50,000 AA 成本试测约 8,336 AA/s、GPU reserved 200 MiB、RSS 0.948 GiB；MIL 拟合峰值 reserved 约 0.092 GiB，仅代表 MIL 拟合，不含全部前处理。\n- 预关联修订披露：此前读取 Home N 哈希用于来源记录，未据 N 选择种子/PCA/群/阈值；应称“没有按结局选择规则”，不能称此前从未访问 N。\n- 原始数据、旧结果、冻结半径/阈值完整保留。External 属于历史外部复核，抗原结合和病理机制均未确立。\n\n{link('本轮验收 JSON', 'audit/09_completion_acceptance_20261001.json')}；{link('正式串行日志', 'logs/07_after_pilots.jsonl')}。\n\n## 5. 本轮交付与后续边界\n\n已交付：340 文库来源/QC、两套全量向量、联合/内部描述图谱、Home 与训练折字典、完整患者检验表、四表示筛选和 OOF，以及可审计的条件停止记录。\n\n本次没有达到“记录 N+ 贡献群”全部门槛。若另行预设下一轮分析，优先核实临床终点/治疗时序、共同上游处理以及非退化的群丰度模型；保留本轮阴性结果，不从 94 个名义 P 群事后指定关键群。\n"
once(P,report)
source_rels=["audit/07_orchestrator_source_lock.json","audit/01_stage1_validation.json",
             "embeddings/verification.json","audit/05_association_receipt.json",
             "groups/full_home/association_recorded_home.tsv","prediction/05_receipt.json",
             "prediction/05_patient_oof.tsv","prediction/05_summary.json",
             "groups/full_home/external_projection_receipt.json","groups/full_home/external_direction.json",
             "results/05_final_candidate_gate.json","results/05_ablation_receipt.json",
             "results/05_sensitivity_receipt.json","antigen/receipt.json",
             "representation_comparison/report/summary.json","logs/07_after_pilots.jsonl"]
receipt={"schema":"atlas340_completion_acceptance_v1","status":"PASS_SCOPED_NEGATIVE_CLOSEOUT",
         "verified_utc":utc,"endpoint":"recorded N+; pathologic pN/ypN unverified",
         "scientific_status":"No candidate meets the frozen contribution gate","n_final_candidates":0,
         "checks":c,"association_recomputed":stats,"scope_statuses":scope_status,
         "runner":{"last_event":events[-1],"n_end_events":len(ends),"nonzero_exits":0},
         "source_sha256":{rel:sha(R/rel) for rel in source_rels},
         "script_sha256":sha(Path(__file__)),"report_path":str(P),"report_sha256":sha(P),
         "independent_audits":["final_stat_audit","final_provenance_audit"],
         "conditional_not_run":["Candidate-specific equal-depth/stability (zero q candidates)",
             "Nonempty External group evaluation (empty frozen candidate universe)",
             "Delete-group and matched-random retraining (zero q candidates)",
             "Antigen version-file hash lock, queries, enrichment and cards (gate STOP)",
             "CDR3vec full-union encoding and actual equal-depth MIL retraining (failed screen)"],
         "limits":["Cohort/upstream processing confounding remains","Primary L1 predictions are constant",
             "Representation screen samples a small fraction of complete repertoire",
             "Inner validation sequences contributed to outer-training unlabeled dictionary",
             "External all-group burden atlas not generated",
             "Full finite/count checks rely on existing PASS receipts; no full refit"]}
once(A,json.dumps(receipt,indent=2,ensure_ascii=False)+"\n")
print(json.dumps({"status":receipt["status"],"n_checks":len(c),"min_q":stats["min_q"],
                  "n_final_candidates":0,"report":str(P),"audit":str(A)},ensure_ascii=False))
