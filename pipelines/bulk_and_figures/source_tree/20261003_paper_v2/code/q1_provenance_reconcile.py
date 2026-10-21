#!/usr/bin/env python3
"""Reconcile queried SRR/GSM accessions with saved official GEO and ENA metadata."""
from release_runtime import private_literal as _private_literal, release_path as _release_path
from pathlib import Path
import argparse,csv,json,gzip,re,hashlib,datetime

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def rows(p):
 with Path(p).open(newline='') as f:return list(csv.DictReader(f,delimiter='\t'))
def write(p,data,fields=None):
 data=list(data);fields=fields or list(data[0])
 with Path(p).open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=fields,delimiter='\t');w.writeheader();w.writerows(data)
def main():
 ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--root',type=Path,default=Path(_release_path('path_ab9dd0e0159dec18')));args=ap.parse_args();q=args.root.resolve()/'q1';probe=q/'reference_probe';out=probe/'official_run_mapping'
 geo=probe/'GSE120101_family.soft.gz';text=gzip.decompress(geo.read_bytes()).decode();records={};current=None
 for line in text.splitlines():
  if line.startswith('^SAMPLE = '):current=line.split(' = ',1)[1];records[current]={'GSM':current}
  elif current and line.startswith('!Sample_title = '):records[current]['title']=line.split(' = ',1)[1]
  elif current and line.startswith('!Sample_characteristics_ch1 = '):
   key,value=line.split(' = ',1)[1].split(': ',1);records[current][key.lower()]=value
  elif current and line.startswith('!Sample_relation = BioSample: '):records[current]['SAMN']=line.rsplit('/',1)[1]
  elif current and line.startswith('!Sample_relation = SRA: '):records[current]['SRX']=line.split('term=',1)[1]
 assert '!Series_relation = BioProject: https://www.ncbi.nlm.nih.gov/bioproject/PRJNA491658' in text
 assert '!Series_relation = SRA: https://www.ncbi.nlm.nih.gov/sra?term=SRP162009' in text
 umbrella=(out/'GSE120102_umbrella_GEO_series.txt').read_text()
 assert '!Series_relation = SuperSeries of: GSE120101' in umbrella and 'PRJNA491656' in umbrella
 with (probe/'PRJNA491656.csv').open() as f:local={r['RunId']:r for r in csv.DictReader(f)}
 eligible=[r for r in rows(probe/'PRJNA491656_eligibility_inventory.tsv') if r['query_inclusion']=='reference_only_pretreatment_crc_sorted']
 ena_receipts=json.loads((out/'ENA_fetch_receipt.json').read_text());ena={r['run']:r for r in ena_receipts};crosswalk=[]
 for r in eligible:
  run=Path(r['zip_member']).stem;m=local[run];g=records[r['metadata_GSM']];e=ena[run];assert 'error' not in e
  er=rows(out/(run+'_ENA_run_metadata.tsv'));assert len(er)==1;er=er[0]
  assert sha(out/(run+'_ENA_run_metadata.tsv'))==e['sha256']
  assert er['run_accession']==run and er['experiment_accession']==m['ExperimentId']==g['SRX']
  assert er['study_accession']=='PRJNA491658' and er['secondary_study_accession']=='SRP162009'
  assert er['sample_alias']==er['experiment_alias']==m['Comment']==g['GSM']
  assert er['sample_accession']==g['SAMN'] and er['secondary_sample_accession']==m['SampleId']
  assert g['cancer type']=='colorectal' and g['organ']=='peripheral blood' and g['title']==r['GEO_title'] and r['time_from_GEO_title']=='pre'
  crosswalk.append({'run_accession':run,'GSM':g['GSM'],'GEO_title':g['title'],'donor_label':r['donor_label_from_GEO_title'],'GEO_cancer_type':g['cancer type'],'GEO_organ':g['organ'],'GEO_cell_type':g['cell type'],'SRX':g['SRX'],'SAMN':g['SAMN'],'SRS':m['SampleId'],'official_TCR_BioProject':'PRJNA491658','official_SRA_study':'SRP162009','official_GEO':'GSE120101','official_umbrella_BioProject':'PRJNA491656','official_umbrella_GEO':'GSE120102','local_archive_filename':'PRJNA491656.zip','local_source_CSV_ProjectId':m['ProjectId'],'mapping_verdict':'passed_exact_SRR_SRX_GSM_SAMN_SRS_correspondence','eligibility_change':False,'timing_definition':'before_IT1208_study_treatment; not treatment_naive','GEO_source_URL':'https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc='+g['GSM'],'ENA_metadata_URL':e['url'],'ENA_metadata_sha256':e['sha256']})
 assert len(crosswalk)==7
 target=out/'queried_run_GSM_BioProject_reconciliation.tsv';write(target,crosswalk)
 receipt={'generated_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'entrypoint_sha256':sha(__file__),'GSE120101_SOFT_sha256':sha(geo),'official_umbrella_GEO_sha256':sha(out/'GSE120102_umbrella_GEO_series.txt'),'crosswalk_sha256':sha(target),'queried_run_accessions':len(crosswalk),'GSM_specimens':len({r['GSM'] for r in crosswalk}),'donor_labels':len({r['donor_label'] for r in crosswalk}),'all_exact_correspondences':'passed','official_TCR_source':'PRJNA491658 / SRP162009 / GSE120101','official_parent':'PRJNA491656 / GSE120102 umbrella; sibling 3SAGE PRJNA491654 / GSE120028','local_archive_filename_preserved':'PRJNA491656.zip','local_ProjectId_interpretation':'umbrella namespace, not direct TCR accession','eligibility_changes':0,'clone_rows_rescanned':0,'numeric_result_changes':False,'timing':'before IT1208 study treatment; advanced trial may have prior treatment','source_urls':['https://www.ncbi.nlm.nih.gov/bioproject/PRJNA491656','https://www.ncbi.nlm.nih.gov/bioproject/PRJNA491658','https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE120101','https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE120102']}
 (out/'provenance_reconciliation_receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
 # Actual results gain canonical provenance, while pre-query frozen inventory remains historical.
 result=q/'crc_sorted_reference_query';changed=[]
 extra={'official_TCR_BioProject':'PRJNA491658','official_GEO':'GSE120101','official_SRA_study':'SRP162009','local_archive_filename':'PRJNA491656.zip','timing_definition':'before_IT1208_study_treatment; not treatment_naive'}
 for filename in ['run_library_q1.tsv','role_detection_summary.tsv','exact_Q1_hit_rows.tsv']:
  p=result/filename
  if not p.exists():continue
  with p.open(newline='') as f:r=csv.DictReader(f,delimiter='\t');fields=r.fieldnames;data=list(r)
  for r in data:
   r.update(extra)
   if filename=='role_detection_summary.tsv':r['study']='PRJNA491658';r['local_archive_label']='PRJNA491656'
  newfields=list(fields)+[k for k in extra if k not in fields]
  if filename=='role_detection_summary.tsv' and 'local_archive_label' not in newfields:newfields.append('local_archive_label')
  write(p,data,newfields);changed.append(p)
 receiptpath=result/'query_receipt.json'
 if receiptpath.exists():
  r=json.loads(receiptpath.read_text());r['official_source_reconciliation']={'receipt_path':str(out/'provenance_reconciliation_receipt.json'),'receipt_sha256':sha(out/'provenance_reconciliation_receipt.json'),'canonical_TCR_source':'PRJNA491658 / GSE120101 / SRP162009','archive_namespace':'PRJNA491656 umbrella','eligible_counts_unchanged':True,'clone_rows_rescanned':0}
  for p in changed:r['output_sha256'][p.name]=sha(p)
  receiptpath.write_text(json.dumps(r,indent=2)+'\n')
 print(json.dumps({'exact_run_mapping_passed':7,'canonical_TCR_source':'PRJNA491658/GSE120101','local_archive_preserved':'PRJNA491656.zip','clone_rows_rescanned':0}))
if __name__=='__main__':main()
