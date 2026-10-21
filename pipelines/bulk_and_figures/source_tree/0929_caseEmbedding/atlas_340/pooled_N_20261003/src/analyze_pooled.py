from release_runtime import private_literal as _private_literal, release_path as _release_path
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[k]='1'
from pathlib import Path
import ast,json,hashlib,datetime,time,resource,warnings
import numpy as np
import pandas as pd
from scipy.stats import chi2,rankdata
import statsmodels.api as sm
import statsmodels as statsmodels_package
from statsmodels.tools.sm_exceptions import PerfectSeparationWarning
started=time.monotonic()
root=Path(__file__).resolve().parents[1]
base=root.parent;prior=base/'leiden_full_20261001';bias=prior/'n_bias_20261002'
design=json.loads((root/'config/design_freeze.json').read_text())
assert design['n_hypotheses']==248
assert not (root/'audit/completion.json').exists(),'Completed branch is immutable'
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for c in iter(lambda:f.read(1048576),b''):h.update(c)
 return h.hexdigest()
def dump(p,d):p.write_text(json.dumps(d,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
source=prior/'src/patient_cluster_stats.py';code=source.read_text();tree=ast.parse(code)
needed={'standardized','glm','effect','bh'};copied={}
for node in tree.body:
 if isinstance(node,ast.FunctionDef) and node.name in needed:
  segment=ast.get_source_segment(code,node);exec(compile(segment,str(source),'exec'),globals());copied[node.name]=hashlib.sha256(segment.encode()).hexdigest()
assert set(copied)==needed
inputs={str(source):sha(source),str(root/'config/design_freeze.json'):sha(root/'config/design_freeze.json')}
frames={};metas=[]
for model,nc in design['models'].items():
 receipt_path=bias/f'audit/{model}_completion.json';receipt=json.loads(receipt_path.read_text());inputs[str(receipt_path)]=sha(receipt_path)
 assert receipt['status']=='PASS' and receipt['n_clusters']==nc and receipt['n_patients']==339
 rel=f'stats/{model}_patient_cluster_abundance.tsv';p=bias/rel;inputs[str(p)]=sha(p)
 assert inputs[str(p)]==receipt['outputs'][rel]['sha256']
 d=pd.read_csv(p,sep='\t',keep_default_na=False)
 assert len(d)==339*nc and not d.duplicated(['patient_key','cluster_id']).any()
 assert sorted(d.cluster_id.unique())==list(range(nc))
 groups=d.groupby('patient_key',sort=True)
 for col in ['cohort','recorded_N','patient_total_unique_AA','patient_total_assigned_reads']:assert (groups[col].nunique()==1).all(),col
 meta=groups[['cohort','recorded_N','patient_total_unique_AA','patient_total_assigned_reads']].first()
 assert np.array_equal(groups.cluster_unique_AA.sum(),meta.patient_total_unique_AA)
 assert np.array_equal(groups.cluster_assigned_reads.sum(),meta.patient_total_assigned_reads)
 assert np.allclose(d.cluster_unique_fraction,d.cluster_unique_AA/d.patient_total_unique_AA,rtol=0,atol=1e-12)
 assert np.allclose(d.cluster_read_fraction,d.cluster_assigned_reads/d.patient_total_assigned_reads,rtol=0,atol=1e-12)
 assert np.allclose(groups.cluster_unique_fraction.sum(),1) and np.allclose(groups.cluster_read_fraction.sum(),1)
 if metas:pd.testing.assert_frame_equal(metas[0],meta)
 metas.append(meta);frames[model]=d
meta=metas[0];known=meta[meta.recorded_N.isin(['0','1'])].copy();keys=known.index;y=known.recorded_N.astype(int).to_numpy()
assert len(keys)==289 and int(y.sum())==110 and int((y==0).sum())==179
assert known.groupby(['cohort','recorded_N']).size().to_dict()=={('home','0'):70,('home','1'):56,('external','0'):109,('external','1'):54}
# Read cohort nuisance dummy encodings. Recompute common depth standardization from raw totals.
cov={}
for c in ['home','external']:
 p=prior/f'stats/{c}_covariate_design.tsv';inputs[str(p)]=sha(p);cov[c]=pd.read_csv(p,sep='\t').set_index('patient_key')
 assert set(cov[c].index)==set(known[known.cohort==c].index)
columns=[('intercept',np.ones(len(keys))),('cohort_external',known.cohort.eq('external').to_numpy(float)),('log1p_total_reads_z',standardized(np.log1p(known.patient_total_assigned_reads.to_numpy(float)))),('log1p_total_unique_AA_z',standardized(np.log1p(known.patient_total_unique_AA.to_numpy(float))))]
for c,prefix in [('home','home_batch_'),('external','technical_batch_')]:
 for col in cov[c].columns:
  if col.startswith(prefix):columns.append((c+'_'+col,cov[c][col].reindex(keys).fillna(0).to_numpy(float)))
kept=[];dropped=[];x=np.empty((len(keys),0))
for name,v in columns:
 candidate=np.column_stack((x,v))
 if np.linalg.matrix_rank(candidate)>x.shape[1]:kept.append(name);x=candidate
 else:dropped.append(name)
assert x.shape[1]==len(kept) and np.linalg.matrix_rank(x)==x.shape[1]
condition=float(np.linalg.cond(x));assert condition<1e10
xp=np.ones((len(keys),1));designs={'naive_pooled_intercept_only':xp,'primary_pooled_cohort_technical_adjusted':x}
nulls={name:glm(y,matrix) for name,matrix in designs.items()}
patient=known.reset_index().rename(columns={'index':'patient_key'});patient.to_csv(root/'stats/pooled_patients.tsv',sep='\t',index=False)
cdf=pd.DataFrame(x,index=keys,columns=kept);cdf.index.name='patient_key';cdf.to_csv(root/'stats/pooled_covariate_design.tsv',sep='\t')
batch=[]
for c in ['home','external']:
 sub=known[known.cohort==c];cc=cov[c].reindex(sub.index);prefix='home_batch_' if c=='home' else 'technical_batch_'
 labels=pd.Series('H2498B' if c=='home' else 'pool1',index=sub.index)
 for col in cc:
  if col.startswith(prefix):labels[cc[col].eq(1)]=col[len(prefix):]
 tab=pd.DataFrame({'cohort':c,'batch':labels,'N':sub.recorded_N.astype(int)}).groupby(['cohort','batch','N']).size().unstack('N',fill_value=0)
 for (cohort,b),r in tab.iterrows():batch.append({'cohort':cohort,'batch':b,'N0':int(r.get(0,0)),'Nplus':int(r.get(1,0))})
pd.DataFrame(batch).to_csv(root/'stats/batch_N_distribution.tsv',sep='\t',index=False)
rows=[]
for model,d in frames.items():
 for cluster in range(design['models'][model]):
  g=d[d.cluster_id==cluster].set_index('patient_key').reindex(keys)
  assert g.index.equals(keys) and len(g)==289
  for scale,count,total,fraction in [('read_fraction_log_per_sd','cluster_assigned_reads','patient_total_assigned_reads','cluster_read_fraction'),('unique_fraction_log_per_sd','cluster_unique_AA','patient_total_unique_AA','cluster_unique_fraction')]:
   raw=np.log((g[count].to_numpy(float)+1)/(g[total].to_numpy(float)+1));z=standardized(raw)
   f=g[fraction].to_numpy(float);ranks=rankdata(f,method='average');u=ranks[y==1].sum()-110*111/2;delta=float(2*u/(110*179)-1)
   cohort_context={}
   for c in ['home','external']:
    cg=g[g.cohort==c];a=cg[cg.recorded_N=='1'][fraction].to_numpy(float);b=cg[cg.recorded_N=='0'][fraction].to_numpy(float)
    rr=rankdata(np.r_[a,b]);uu=rr[:len(a)].sum()-len(a)*(len(a)+1)/2
    cohort_context[c+'_unadjusted_Cliff_delta']=float(2*uu/(len(a)*len(b))-1)
    cohort_context[c+'_median_fraction_Nplus']=float(np.median(a));cohort_context[c+'_median_fraction_N0']=float(np.median(b))
   for name,matrix in designs.items():
    fit=effect(y,matrix,nulls[name],z)
    lo=fit['beta']-1.96*fit['se'];hi=fit['beta']+1.96*fit['se']
    rows.append({'model':model,'cluster_id':cluster,'scale':scale,'design':name,'n_patients':289,'Nplus':110,'N0':179,'feature_log_mean':float(raw.mean()),'feature_log_sd':float(raw.std(ddof=0)),'median_fraction_Nplus':float(np.median(f[y==1])),'median_fraction_N0':float(np.median(f[y==0])),'pooled_unadjusted_Cliff_delta':delta,'beta_CI_low':lo,'beta_CI_high':hi,**fit,**cohort_context})
result=pd.DataFrame(rows);assert len(result)==248 and not result.duplicated(['model','cluster_id','scale','design']).any()
result['q_BH_all_248']=bh(result.p.to_numpy());result['q_BY_all_248']=(result.q_BH_all_248*sum(1/k for k in range(1,249))).clip(upper=1)
result['Pq_CI_Nplus_positive']=(result.status.eq('ok') & result.p.lt(.05) & result.q_BH_all_248.lt(.05) & result.beta_CI_low.gt(0))
result['Pq_CI_N0_positive']=(result.status.eq('ok') & result.p.lt(.05) & result.q_BH_all_248.lt(.05) & result.beta_CI_high.lt(0))
result['cohort_raw_directions_agree']=result.home_unadjusted_Cliff_delta*result.external_unadjusted_Cliff_delta>0
result=result.sort_values(['q_BH_all_248','p','model','scale','cluster_id']).reset_index(drop=True)
result.to_csv(root/'stats/all_248_pooled_tests.tsv',sep='\t',index=False,na_rep='NA')
primary=result[result.design.eq('primary_pooled_cohort_technical_adjusted')];naive=result[result.design.eq('naive_pooled_intercept_only')]
primary.to_csv(root/'stats/primary_adjusted_124_tests.tsv',sep='\t',index=False,na_rep='NA');naive.to_csv(root/'stats/naive_124_tests.tsv',sep='\t',index=False,na_rep='NA')
summary={}
for name,d in [('naive',naive),('primary_adjusted',primary)]:
 for model in design['models']:
  g=d[d.model==model];summary[name+'_'+model]={'tests':len(g),'nominal_P_lt05':int((g.p<.05).sum()),'BH_q_lt05':int((g.q_BH_all_248<.05).sum()),'BY_q_lt05':int((g.q_BY_all_248<.05).sum()),'Nplus_PqCI':int(g.Pq_CI_Nplus_positive.sum()),'N0_PqCI':int(g.Pq_CI_N0_positive.sum()),'min_P':float(g.p.min()),'min_BH_q':float(g.q_BH_all_248.min()),'failed_fits':int(g.status.ne('ok').sum())}
passrows=result[result.Pq_CI_Nplus_positive|result.Pq_CI_N0_positive]
passrows.to_csv(root/'stats/Pq_CI_passed_rows.tsv',sep='\t',index=False,na_rep='NA')
# The existing helper regression guards remain valid without touching old results.
assert np.allclose(bh([.01,.04,1]),[.03,.06,1])
assert effect(y,x,nulls['primary_pooled_cohort_technical_adjusted'],np.zeros(len(y)))['p']==1
assert np.isfinite(result.p).all() and result.p.between(0,1).all() and result.q_BH_all_248.between(0,1).all()
assert (result.q_BH_all_248+1e-15>=result.p).all()
for path,h in inputs.items():assert sha(Path(path))==h,'Source changed during analysis'
lines=['# 合并 Home + External 的记录 N 群关联','',f"生成：{datetime.datetime.now(datetime.timezone.utc).isoformat()}",'','## 设计与范围','',
_private_literal('private_aecd94a4c663c0c5'), '',
'使用冻结主标签（resolution=1, seed=17）：TCR-BERT 32 群、SCEPTR 30 群。每群检验读数占比与 AA 占比两个连续特征；变换沿用 log((群计数+1)/(患者总计数+1))，再按合并患者 SD 标准化。', '',
'直接合并只含截距；主要合并模型调整 cohort、log1p总reads、log1p总unique AA、队列内技术批次。External 的年龄/性别/新辅助治疗全缺，因此不能称完整临床调整。两个方法 × 两个尺度 × 62 群共 **248 个新检验，全部一起 BH 校正**；另附同一组 P 的 BY 敏感性。新 family 独立于既往 family，不改变旧结果。', '',
'沿用原 one-df likelihood-ratio binomial GLM；报告同一模型的 Wald 95% β/OR 区间。β 无效值为 0，OR 无效值为 1；N+ 偏向须 β CI 下界>0。拟合失败保留 P=1、效应 NA。原 Cliff δ 区间没有用作调整模型的 CI。', '',
'## 结果','', '| 方法 | 模型 | P<0.05 | 全248项BH q<0.05 | N+方向且P/q/CI全通过 | N0方向且全通过 | 最小P | 最小q |','|---|---|---:|---:|---:|---:|---:|---:|']
for name in ['naive','primary_adjusted']:
 for model in design['models']:
  z=summary[name+'_'+model];lines.append(f"|{name}|{model}|{z['nominal_P_lt05']}|{z['BH_q_lt05']}|{z['Nplus_PqCI']}|{z['N0_PqCI']}|{z['min_P']:.5g}|{z['min_BH_q']:.5g}|")
lines+=['','### 队列与技术变量调整后的前十项','', '| 模型 | 群 | 尺度 | β [95%CI] | OR [95%CI] | P | BH q |','|---|---:|---|---|---|---:|---:|']
for _,r in primary.head(10).iterrows():lines.append(f"|{r.model}|{int(r.cluster_id)}|{r.scale}|{r.beta:.3f} [{r.beta_CI_low:.3f}, {r.beta_CI_high:.3f}]|{r.odds_ratio:.3f} [{r.ci_lower:.3f}, {r.ci_upper:.3f}]|{r.p:.5g}|{r.q_BH_all_248:.5g}|")
lines+=['','## 解释','',
'合并样本可增加统计信息，但保留 cohort 调整才能减少队列来源、批次和 N 比例差异的混杂。直接合并中出现的低 P/q，不能替代主要调整模型。所有正/负方向和失败项均在完整表中，没有挑选群或重新调整分辨率。', '',
'这是在已查看两队列、共同构建图谱后的新增探索性关联；不是新的外部验证、患者级折外预测贡献或抗原结合证据。分队列原始丰度方向、组成约束和未能调整的临床因素仍须保留；即使合并有阳性，也不能据此声称跨队列稳定。', '',
'- [全部248项 P/q/CI](../stats/all_248_pooled_tests.tsv)',
'- [主要调整模型124项](../stats/primary_adjusted_124_tests.tsv)',
'- [直接合并124项](../stats/naive_124_tests.tsv)',
'- [通过P/q/CI的正负方向条目](../stats/Pq_CI_passed_rows.tsv)',
'- [患者纳入清单](../stats/pooled_patients.tsv)',
'- [批次N分布](../stats/batch_N_distribution.tsv)',
'- [冻结设计](../config/design_freeze.json)','']
(root/'report/POOLED_N_REPORT.md').write_text('\n'.join(lines))
outputs={str(p.relative_to(root)):{'bytes':p.stat().st_size,'sha256':sha(p)} for folder in ['stats','report'] for p in sorted((root/folder).iterdir()) if p.is_file()}
receipt={'status':'PASS','completed_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'inputs':inputs,'outputs':outputs,'source_code_sha256':sha(Path(__file__)),'reused_helper_SHA':copied,'summary':summary,'patients':289,'Nplus':110,'N0':179,'n_hypotheses':248,'patient_count_and_normalization_conservation':True,'same_patient_labels_and_counts_between_models':True,'source_SHA_unchanged_after_run':True,'design_kept':kept,'design_dropped_label_blind':dropped,'design_matrix_rank':int(np.linalg.matrix_rank(x)),'design_matrix_condition_number':condition,'unstable_status_counts':result.status.value_counts().to_dict(),'threads':1,'GPU_used':False,'elapsed_seconds':time.monotonic()-started,'peak_RSS_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,'statsmodels_version':statsmodels_package.__version__}
dump(root/'audit/completion.json',receipt)
print(json.dumps({'status':'PASS','summary':summary,'passed_rows':passrows[['model','cluster_id','scale','design','beta','p','q_BH_all_248']].to_dict(orient='records'),'elapsed_seconds':receipt['elapsed_seconds'],'peak_RSS_bytes':receipt['peak_RSS_bytes']},ensure_ascii=False))
