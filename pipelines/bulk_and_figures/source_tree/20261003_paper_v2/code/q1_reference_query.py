#!/usr/bin/env python3
"""Query only source-verified pretreatment CRC sorted run members on the server."""
from release_runtime import private_literal as _private_literal, release_path as _release_path
from pathlib import Path
import argparse,csv,json,zipfile,io,hashlib,importlib.util,datetime,collections,shutil

def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for block in iter(lambda:f.read(1048576),b''):h.update(block)
 return h.hexdigest()
def rows(p):
 with Path(p).open(newline='') as f:return list(csv.DictReader(f,delimiter='\t'))
def write(p,data,fields=None):
 data=list(data);fields=fields or list(data[0])
 with Path(p).open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=fields,delimiter='\t');w.writeheader();w.writerows(data)
def main():
 ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--root',type=Path,default=Path(_release_path('path_ab9dd0e0159dec18')));args=ap.parse_args()
 q=args.root.resolve()/'q1';probe=q/'reference_probe';out=q/'crc_sorted_reference_query';out.mkdir(exist_ok=True)
 spec=importlib.util.spec_from_file_location('core',q/'inputs/q1_core_frozen.py');core=importlib.util.module_from_spec(spec);spec.loader.exec_module(core)
 inventory=probe/'PRJNA491656_eligibility_inventory.tsv';archive=Path(_release_path('path_7eadc32ff9f83e41'))
 allrows=rows(inventory);eligible=[r for r in allrows if r['query_inclusion']=='reference_only_pretreatment_crc_sorted']
 assert len(eligible)==7 and len({r['metadata_GSM'] for r in eligible})==6 and len({r['donor_label_from_GEO_title'] for r in eligible})==3
 assert all(r['GEO_cancer_type']=='colorectal' and r['GEO_organ']=='peripheral blood' and r['time_from_GEO_title']=='pre' and r['zip_member_exists']=='True' and r['healthy_eligible']=='False' and r['CRC_primary_eligible']=='False' for r in eligible)
 mapping_path=probe/'official_run_mapping/queried_run_GSM_BioProject_reconciliation.tsv';mapping=rows(mapping_path)
 assert {r['run_accession'] for r in mapping}=={Path(r['zip_member']).stem for r in eligible}
 assert all(r['official_TCR_BioProject']=='PRJNA491658' and r['official_GEO']=='GSE120101' and r['mapping_verdict'].startswith('passed') for r in mapping)
 canonical={'official_TCR_BioProject':'PRJNA491658','official_GEO':'GSE120101','official_SRA_study':'SRP162009','local_archive_filename':'PRJNA491656.zip','timing_definition':'before_IT1208_study_treatment; not treatment_naive'}
 eligible=[{**r,**canonical} for r in eligible]
 frozen=out/'frozen_eligible_inventory.tsv';shutil.copyfile(inventory,frozen);inventory_sha=sha(frozen);archive_sha=sha(archive)
 definitions={'AA':core.AA,'AA_VJ':f'{core.AA};{core.V};{core.J}','NT':core.NT}
 rules={'frozen_inventory_sha256':inventory_sha,'frozen_before_query_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
  'eligibility':'GEO explicit cancer type colorectal (exclude dual primary), organ peripheral blood, title pre_Blood_CD4/CD8 with donor label, existing processed ZIP member',
  'analysis_unit':'run CSV; 7 runs, 6 GSM specimens, 3 donor labels; no pooling across run, subset, or time; reference only',
  'canonical_TCR_source':'PRJNA491658 / GSE120101 / SRP162009','archive_umbrella_namespace':'PRJNA491656 / GSE120102','official_run_mapping_sha256':sha(mapping_path),'timing':'before IT1208 study treatment; not treatment-naive',
  'definitions':definitions,'source_all':'exact normalized AA/NT retained even when sequence proxy QC fails; count threshold uses positive mathematical integer source cloneCount',
  'sequence_proxy_QC':'AA canonical 20 letters starts C; NT ACGT and 3-divisible; standard translation agrees with stored AA; positive integral count (821.0 allowed)',
  'productive_flag':'not present in processed CSV; original GEO processing claims productive MiXCR3 filtering, not identical to current proxy QC',
  'assay':'source GEO total RNA, sorted CD4/CD8, Ion Proton; source MiXCR3/VDJtools processing includes coverage normalization to 5x and noise exclusions; local processed CSV stage not proven identical to final paper table',
  'count_unit':'processed source cloneCount; not uniform UMI/template count or assay LOD',
  'no_healthy_or_diagnostic_role':True,'technical_gate':'unresolved'}
 (out/'query_rules_frozen.json').write_text(json.dumps(rules,indent=2)+'\n')
 start=datetime.datetime.now(datetime.timezone.utc);summaries=[];hits=[]
 with zipfile.ZipFile(archive) as z:
  for meta in eligible:
   member=meta['zip_member'];info=z.getinfo(member);counter=collections.Counter();raw=collections.Counter();qc=collections.Counter();reasons=collections.Counter()
   with z.open(member) as stream:
    reader=csv.DictReader(io.TextIOWrapper(stream,encoding='utf-8-sig',newline=''))
    assert {'AASeq','NNSeq','Vregion','Jregion','cloneCount','cloneFraction'}<=set(reader.fieldnames)
    for number,r in enumerate(reader,start=2):
     counter['rows_scanned']+=1;c=core.integral_count(r['cloneCount']);aa,nt,passed,failed=core.public_sequence_qc(r['AASeq'],r['NNSeq'],c);reasons.update(failed)
     if c is not None:counter['positive_integral_count_rows']+=1;counter['source_total_cloneCount']+=c
     if passed:counter['sequence_proxy_QC_rows']+=1;counter['sequence_proxy_QC_total_cloneCount']+=c
     v=r['Vregion'].strip().split('*')[0];j=r['Jregion'].strip().split('*')[0]
     match={'AA':aa==core.AA,'AA_VJ':aa==core.AA and v==core.V and j==core.J,'NT':nt==core.NT}
     for d in definitions:
      if match[d] and c is not None:
       raw[d]+=c
       if passed:qc[d]+=c
     if match['AA'] or match['NT']:
      hits.append({**meta,'clone_query_performed':True,'CSV_row_number':number,'source_AASeq':r['AASeq'],'source_NNSeq':r['NNSeq'],'source_Vregion':r['Vregion'],'source_Jregion':r['Jregion'],'source_cloneCount':r['cloneCount'],'source_cloneFraction':r['cloneFraction'],'mathematical_integer_count':c if c is not None else '',**{'exact_'+d:match[d] for d in definitions},'sequence_proxy_QC_pass':passed,'QC_failure_reasons':';'.join(failed)})
   summary={**meta,'clone_query_performed':True,**{k:counter[k] for k in ['rows_scanned','positive_integral_count_rows','source_total_cloneCount','sequence_proxy_QC_rows','sequence_proxy_QC_total_cloneCount']},'zip_member_size_bytes':info.file_size,_private_literal('private_bb4580532748e494'):f'{info.CRC:08x}','scan_complete_to_EOF':True,'QC_failure_reasons_JSON':json.dumps(dict(reasons),sort_keys=True)}
   for d in definitions:summary[d+'_source_cloneCount']=raw[d];summary[d+'_QC_cloneCount']=qc[d]
   summaries.append(summary)
   print(json.dumps({'completed_run_members':len(summaries),'member':member,'rows':counter['rows_scanned']}),flush=True)
 write(out/'run_library_q1.tsv',summaries)
 hitfields=list(eligible[0])+['CSV_row_number','source_AASeq','source_NNSeq','source_Vregion','source_Jregion','source_cloneCount','source_cloneFraction','mathematical_integer_count','exact_AA','exact_AA_VJ','exact_NT','sequence_proxy_QC_pass','QC_failure_reasons']
 write(out/'exact_Q1_hit_rows.tsv',hits,hitfields)
 stats=[]
 for scope,col in [('source_all','source_cloneCount'),('sequence_proxy_QC','QC_cloneCount')]:
  for d in definitions:
   for t in [1,2,5,10,20]:
    stats.append({'study':'PRJNA491658','local_archive_label':'PRJNA491656',**canonical,'role':'reference_only_pretreatment_crc_sorted','definition':d,'query_scope':scope,'source_count_threshold':t,'n_run_libraries':len(summaries),'n_GSM_specimens':len({r['metadata_GSM'] for r in summaries}),'n_source_donor_labels':len({r['donor_label_from_GEO_title'] for r in summaries}),'detected_run_libraries':sum(r[d+'_'+col]>=t for r in summaries),'total_source_cloneCount':sum(r[d+'_'+col] for r in summaries),'library_pooling':'none','clinical_specificity':'not_estimated','technical_gate':'unresolved'})
 write(out/'role_detection_summary.tsv',stats)
 assert sha(archive)==archive_sha and sha(inventory)==inventory_sha
 receipt={'started_utc':start.isoformat(),'completed_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'entrypoint_sha256':sha(__file__),'frozen_core_sha256':sha(q/'inputs/q1_core_frozen.py'),'frozen_inventory_sha256':inventory_sha,'source_GEO_metadata_sha256':sha(probe/'GSE120101_family.soft.gz'),'archive_sha256_before':archive_sha,'archive_sha256_after':sha(archive),'run_libraries_scanned_to_EOF':len(summaries),'GSM_specimens':6,'source_donor_labels':3,'clone_rows_scanned':sum(r['rows_scanned'] for r in summaries),'exact_hit_rows':len(hits),'unselected_members_not_scanned':89-len(summaries),'old_public_3_ZIPs_rescanned':False,'raw_FASTQ_scanned':False,'output_sha256':{p.name:sha(p) for p in sorted(out.glob('*.tsv'))},'technical_gate':'unresolved','clinical_comparison':'reference_only; advanced cancer trial, sorted subsets, source processing differs from bulk cohorts; no donor pooling or clinical specificity'}
 (out/'query_receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
 report=q/'Q1_FINAL_REPORT.md';txt=report.read_text();mark='\n## GSE120101 / PRJNA491658 有界补充查询（本地存档 PRJNA491656）\n'
 txt=txt.split(mark)[0]+mark+f'\n在逐GSM cancer type、organ、IT1208给药前时间和CD4/CD8分选信息冻结后，仅扫描7个可核验CRC血液run成员至EOF（6个GSM、3位来源donor；GSE120101/PRJNA491658 TCR分支，本地ZIP沿用umbrella491656命名；共{receipt["clone_rows_scanned"]:,}行），其余82成员未扫描。来源为总RNA/Ion Proton、MiXCR3/VDJtools处理，包含原文覆盖度标准化和噪声处理；当前processed cloneCount的具体阶段无法等同于Home MiXCR read或UMI。查询结果见crc_sorted_reference_query/role_detection_summary.tsv与逐库表，source-all和sequence-proxy QC均保留；before IT1208不表示初治或治疗未暴露。这是分选参考，不是新增健康或独立bulk CRC验证。没有合并0006 CD4两run，也没有将CD4/CD8当两个人。真实性仍unresolved。\n'
 report.write_text(txt)
 print(json.dumps({'reference_query_complete':True,'run_libraries':7,'GSM_specimens':6,'donors':3,'exact_hit_rows':len(hits),'rows':receipt['clone_rows_scanned']}))
if __name__=='__main__':main()
