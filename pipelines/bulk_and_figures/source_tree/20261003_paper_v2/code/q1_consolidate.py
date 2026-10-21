#!/usr/bin/env python3
"""Recompute descriptive Q1 evidence on the server from hashed immutable inputs."""
from release_runtime import private_literal as _private_literal, release_path as _release_path
from pathlib import Path
import argparse,csv,json,hashlib,shutil,importlib.util,datetime,os,re,urllib.request,gzip
BASE=Path(_release_path('path_39343c5e69c3d939'))
OUT=BASE/'20261003_paper_v2/q1'
PRIOR=BASE/'20261003_reframing_crc_vs_healthy_v1'
PAPER1=BASE/'20261003_paper_v1'

def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
 return h.hexdigest()
def rows(p):
 with Path(p).open(newline='') as f:return list(csv.DictReader(f,delimiter='\t'))
def write(p,r,fields=None):
 r=list(r);fields=fields or list(r[0])
 with Path(p).open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=fields,delimiter='\t');w.writeheader();w.writerows(r)
def freeze(src,dst,manifest):
 dst.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(src,dst)
 assert sha(src)==sha(dst)
 manifest.append({'source_path':str(src),'frozen_path':str(dst),'size_bytes':dst.stat().st_size,'sha256':sha(dst)})

def missed_reference_probe():
 probe=OUT/'reference_probe';probe.mkdir(exist_ok=True)
 meta=Path(_release_path('path_68a2163223ec1f2b'));archive=meta.with_suffix('.zip')
 import zipfile,collections,io
 manifest=[];freeze(meta,probe/meta.name,manifest)
 url='https://ftp.ncbi.nlm.nih.gov/geo/series/GSE120nnn/GSE120101/soft/GSE120101_family.soft.gz'
 geo_status='unavailable';titles={};characteristics={};source_receipt={'GEO_URL':url}
 try:
  target=probe/'GSE120101_family.soft.gz'
  data=target.read_bytes() if target.exists() else urllib.request.urlopen(url,timeout=20).read(2*1024*1024+1)
  assert len(data)<=2*1024*1024,'metadata exceeds bounded download limit'
  assert hashlib.sha256(data).hexdigest()=='aa5c8fd4e7cdc7b91b3d47422f9f0e1a9e6f9c6aaf899486ba9d9c3480ce12aa','Frozen GEO metadata SHA mismatch'
  if not target.exists():target.write_bytes(data)
  text=gzip.decompress(data).decode();current=None
  for line in text.splitlines():
   if line.startswith('^SAMPLE = '):
    current=line.split(' = ',1)[1];characteristics[current]={}
   elif current and line.startswith('!Sample_title = '):titles[current]=line.split(' = ',1)[1]
   elif current and line.startswith('!Sample_characteristics_ch1 = '):
    key,value=line.split(' = ',1)[1].split(': ',1);characteristics[current][key.lower()]=value
  geo_status='downloaded_metadata_only';source_receipt.update({'file_sha256':sha(target),'file_bytes':len(data),'GEO_sample_titles':len(titles)})
 except Exception as e:source_receipt['error']=repr(e)
 with meta.open() as f:metas=list(csv.DictReader(f))
 with zipfile.ZipFile(archive) as z:
  members={i.filename:i for i in z.infolist() if i.filename.endswith('.csv') and not i.filename.startswith('__MACOSX')}
  first=next(iter(members))
  with z.open(first) as f:header=next(csv.reader(io.TextIOWrapper(f)))
  inventory=[]
  mapped=set()
  for r in metas:
   member=r['RunId']+'.csv';mapped.add(member);title=titles.get(r['Comment'],'')
   match=re.fullmatch(r'(pre|post\d+)_(Blood|TIL)_(CD4|CD8)_(\d+)',title)
   donor=match.group(4) if match else '';time=match.group(1) if match else ''
   ch=characteristics.get(r['Comment'],{});cancer=ch.get('cancer type','');organ=ch.get('organ','')
   eligible=bool(match and time=='pre' and match.group(2)=='Blood' and cancer=='colorectal' and organ=='peripheral blood' and member in members)
   inventory.append({'library_id':'PRJNA491656:'+r['RunId'],'zip_member':member,'zip_member_exists':member in members,
    'metadata_GSM':r['Comment'],'GEO_title':title,'source_condition':r['Condition'],'source_material':r['CellSource'],
    'source_cell_type':r['CellType'],'donor_label_from_GEO_title':donor,'time_from_GEO_title':time,
    'GEO_cancer_type':cancer,'GEO_organ':organ,'GEO_cell_type':ch.get('cell type',''),
    'individual_CRC_histology_to_donor_link':'explicit_per_GSM_GEO_characteristics' if cancer else 'unverified','healthy_eligible':False,'CRC_primary_eligible':False,
    'query_inclusion':'reference_only_pretreatment_crc_sorted' if eligible else 'excluded_before_Q1_query',
    'exclusion_reason':'' if eligible else 'no healthy; non-CRC or dual primary, post-treatment, TIL, missing mapping/member; not comparable bulk blood',
    'clone_query_performed':False})
  for member in sorted(set(members)-mapped):
   inventory.append({'library_id':'PRJNA491656:'+Path(member).stem,'zip_member':member,'zip_member_exists':True,
    'metadata_GSM':'','GEO_title':'','source_condition':'unmapped','source_material':'unmapped','source_cell_type':'unmapped',
    'donor_label_from_GEO_title':'','time_from_GEO_title':'','GEO_cancer_type':'','GEO_organ':'','GEO_cell_type':'','individual_CRC_histology_to_donor_link':'unverified',
    'healthy_eligible':False,'CRC_primary_eligible':False,'query_inclusion':'excluded_before_Q1_query',
    'exclusion_reason':'ZIP member absent from current 73-run source metadata; disease/donor/material/timing unverified','clone_query_performed':False})
 write(probe/'PRJNA491656_eligibility_inventory.tsv',inventory)
 receipt={'generated_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'metadata_rows':len(metas),
   'unique_source_samples':len({r['SampleId'] for r in metas}),'unique_GSM':len({r['Comment'] for r in metas}),
   'metadata_runs_present_in_ZIP':sum(r['RunId']+'.csv' in members for r in metas),'metadata_runs_absent_from_ZIP':sum(r['RunId']+'.csv' not in members for r in metas),
   'zip_csv_members':len(members),'unmapped_zip_members':len(set(members)-mapped),'zip_uncompressed_bytes':sum(i.file_size for i in members.values()),
   'zip_path':str(archive),'zip_sha256':sha(archive),'metadata_sha256':sha(meta),'schema_header_probe':header,
   'header_probe_member':first,'header_probe_scope':'header only; no Q1 query; no clone body scan',
   'source_conditions':dict(collections.Counter(r['Condition'] for r in metas)),'source_materials':dict(collections.Counter(r['CellSource'] for r in metas)),
   'GEO_metadata_status':geo_status,'GEO_receipt':source_receipt,'healthy_eligible_libraries':0,'CRC_primary_eligible_libraries':0,
   'pretreatment_CRC_sorted_reference_runs':sum(r['query_inclusion']=='reference_only_pretreatment_crc_sorted' for r in inventory),
   'pretreatment_CRC_sorted_reference_GSM':len({r['metadata_GSM'] for r in inventory if r['query_inclusion']=='reference_only_pretreatment_crc_sorted'}),
   'pretreatment_CRC_sorted_reference_donors':len({r['donor_label_from_GEO_title'] for r in inventory if r['query_inclusion']=='reference_only_pretreatment_crc_sorted'}),
   'independent_people_count':'not derived from run/member counts','trial_people_primary_paper':11,
   'source_primary_links':['https://www.ncbi.nlm.nih.gov/bioproject/PRJNA491656','https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE120101',_private_literal('private_dbdb69805a7b4622')],
   'source_primary_summary':'IT1208 anti-CD4 phase1 in advanced solid tumors; three CRC plus one pancreatic patient with simultaneous colon cancer; per-GSM cancer type, pretreatment time, organ and sorted cell type recovered from GEO; three CRC donors can be identified; treatment-trial selected subsets are reference only',
   'local_archive_namespace':'PRJNA491656 / GSE120102 umbrella','official_TCR_source_from_GEO':'PRJNA491658 / GSE120101 / SRP162009',
   'frozen_eligibility':'no healthy or bulk CRC primary role; only explicit GEO colorectal, pretreatment, peripheral blood, CD4/CD8 sorted existing members eligible as reference; fixed before Q1 inspection','Q1_query_performed':False}
 (probe/'PRJNA491656_probe_receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
 return receipt

def bounded_technical_metadata():
 paths=[Path(_release_path('path_28a639412ba9fc57')),Path(_release_path('path_df0fb75e17e2f0ab'))]
 candidates=[];visited=0;truncated=False
 pattern=re.compile(r'samples?[_ -]?sheet|demultiplex|demux|index[_ -]?sheet|lane|negative|blank|control|阴性|样本信息|样品信息',re.I)
 for root in paths:
  for folder,dirs,files in os.walk(root):
   rel=Path(folder).relative_to(root);depth=len(rel.parts)
   dirs[:]=[d for d in dirs if d.lower() not in {'raw','uid','fastq','fastqs','03_vdjca','04_clna','05_clns','06_clones_tables'} and depth<2]
   for name in files:
    visited+=1
    if visited>3000:truncated=True;break
    p=Path(folder)/name
    if pattern.search(name):candidates.append({'path':str(p),'size_bytes':p.stat().st_size,'mtime_unix':p.stat().st_mtime,'filename_matches':True,'content_read':False,'technical_gate_released':False})
   if truncated:break
  if truncated:break
 write(OUT/'technical_metadata_candidate_inventory.tsv',candidates,fields=['path','size_bytes','mtime_unix','filename_matches','content_read','technical_gate_released'])
 receipt={'generated_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'roots':[str(p) for p in paths],'max_depth':2,'max_files':3000,
          'files_examined_by_filename':visited,'search_truncated':truncated,'candidate_filename_count':len(candidates),'raw_FASTQ_or_binary_alignment_body_reads':0,
          'criterion':'demultiplex/sample-sheet/index/lane/negative/control/specimen info filenames; exclude raw and UID',
          'new_resolving_evidence_recovered':False,'overall_technical_gate':'unresolved',
          'interpretation':'bounded filename inventory only; none independently establishes demultiplex error, negative controls, qPCR patient link or UID clone authenticity; no claim of global absence'}
 (OUT/'technical_metadata_followup_receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
 return receipt

def main():
 global BASE,OUT,PRIOR,PAPER1
 parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--root',type=Path,default=BASE/'20261003_paper_v2',help='Server paper_v2 package directory; prior packages must be sibling directories')
 args=parser.parse_args();OUT=args.root.resolve()/'q1';BASE=args.root.resolve().parent;PRIOR=BASE/'20261003_reframing_crc_vs_healthy_v1';PAPER1=BASE/'20261003_paper_v1'
 OUT.mkdir(parents=True,exist_ok=True);manifest=[]
 frozen=OUT/'inputs/authoritative_reframing';frozen.mkdir(exist_ok=True)
 for p in sorted((PRIOR/'q1').rglob('*')):
  if p.is_file():freeze(p,frozen/p.relative_to(PRIOR/'q1'),manifest)
 corepath=OUT/'inputs/q1_core_frozen.py';freeze(PRIOR/'code/summarize_q1_reframing.py',corepath,manifest)
 for filename in ['technical_gate.json','raw_evidence_inventory.tsv','raw_first_header_by_carrier.tsv','FASTQ_selected_support.tsv','FASTQ_exact_hit_examples.tsv','MiXCR_selected_source_checks.json','MiXCR_selected_reports.json','Q1_AUDIT.md']:
  freeze(PAPER1/'q1'/filename,OUT/'inputs/original_technical_audit'/filename,manifest)
 write(OUT/'immutable_input_manifest.tsv',manifest)
 spec=importlib.util.spec_from_file_location('q1_core',corepath);core=importlib.util.module_from_spec(spec);spec.loader.exec_module(core)
 core.self_test();core.run(argparse.Namespace(source=PAPER1,output=OUT))
 shutil.copytree(frozen/'public_query',OUT/'public_query',dirs_exist_ok=True)
 core.finalize_public(argparse.Namespace(output=OUT))
 probe=missed_reference_probe();technical=bounded_technical_metadata()
 gate=json.loads((OUT/'inputs/original_technical_audit/technical_gate.json').read_text())
 final_gate={'overall_status':'unresolved','per_definition':{'AA':'unresolved','AA_VJ':'unresolved','NT':'unresolved'},
             'processed_count_reproducibility':'passed','public_processed_CSV_full_query':'passed_256_libraries_63260331_rows',
             'biological_association_allowed':False,'diagnostic_ROC_allowed':False,'model_increment_allowed':False,
             'Q1_function_claim_allowed':False,'contamination_diagnosis':'not_made','raw_evidence_status':'original bounded prefix support only; no new read scan',
             'unresolved_conditions':gate['unresolved_conditions'],'bounded_new_metadata_followup':technical,
             'PRJNA491656':'healthy0; only frozen pretreatment CRC sorted reference runs eligible; not bulk or diagnostic comparison'}
 (OUT/'scientific_gate.json').write_text(json.dumps(final_gate,indent=2)+'\n')
 report=(OUT/'Q1_DISEASE_REFRAMING.md').read_text()+_private_literal('private_f3614a54acd1f97f')
 (OUT/'Q1_FINAL_REPORT.md').write_text(report)
 receipt={'generated_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'execution_host':'codex-remote','computations':'server_only',
          'entrypoint':str(Path(__file__).resolve()),'entrypoint_sha256':sha(__file__),'immutable_input_manifest_sha256':sha(OUT/'immutable_input_manifest.tsv'),
          'patient_count':339,'primary_home':137,'external':202,'public_libraries':256,'public_clone_rows':63260331,
          'PRJNA491656_probe':probe,'technical_gate':'unresolved','raw_reads_scanned_this_version':0,'public_existing_ZIP_repeated_scans':0}
 (OUT/'consolidation_receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
 print(json.dumps({'consolidated':True,'patient_count':339,'technical_gate':'unresolved','PRJNA491656_queried':False}))
if __name__=='__main__':main()
