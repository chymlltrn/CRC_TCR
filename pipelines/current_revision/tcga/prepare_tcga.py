import csv, glob, os, re, hashlib, json
from collections import defaultdict
from pathlib import Path
ROOT=Path(os.environ['CRC_TCGA_ROOT']).expanduser().resolve()
DB=Path(os.environ['CRC_TCGA_DB_ROOT']).expanduser().resolve()
ROOT.mkdir(parents=True,exist_ok=True)
BAD={'','na','nan','null','not reported','unknown','[not available]','[not applicable]','[discrepancy]','not available'}
def read(p):
 if not p or not Path(p).exists(): return []
 with open(p,newline='') as f:return list(csv.DictReader(f))
def norm(x):
 s=str(x or '').strip();return '' if s.lower() in BAD else s

def num(x):
 try:
  v=float(norm(x));return v if v==v else None
 except:return None

def stage(x):
 s=norm(x).upper().replace('STAGE','').strip()
 m=re.fullmatch(r'(IV|III|II|I|[1-4])(?:[ABC][123]?)?',s)
 return {'I':1,'II':2,'III':3,'IV':4,'1':1,'2':2,'3':3,'4':4}.get(m.group(1)) if m else None

def pid(r):return norm(r.get('bcr_patient_barcode') or r.get('submitter_id') or r.get('patient'))[:12]
def vital(r):return norm(r.get('vital_status')).lower()
def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(2**20),b''):h.update(b)
 return h.hexdigest()
def write(name,rows):
 if not rows:return
 keys=list(dict.fromkeys(k for row in rows for k in row))
 with open(ROOT/name,'w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=keys,delimiter='\t');w.writeheader();w.writerows(rows)
paths=sorted(glob.glob(str(DB/'tumor update 2026/*/output_mRNA_lncRNA_expr/*_mrna_expr_tpm.csv')))
clinicaldirs={p.name.replace('_','-'):p for p in (DB/'output-clinical').iterdir() if p.is_dir()}
manifest=[];clinical=[];sources=[];inventory=[]
for path in paths:
 path=Path(path);project=path.name.split('_mrna_expr')[0]; cdir=clinicaldirs[project]
 with open(path) as f:headers=next(csv.reader(f))[1:]
 cfiles={typ:next(iter(cdir.glob('*_'+typ+'.csv')),None) for typ in ['clinical_indexed','clinicalXML_patient','clinicalXML_followUp']}
 records={typ:read(p) for typ,p in cfiles.items()}
 for typ,p in cfiles.items():
  if p:sources.append(dict(project=project,kind=typ,path=str(p),bytes=p.stat().st_size,sha256=sha(p)))
 sources.append(dict(project=project,kind='mRNA_TPM',path=str(path),bytes=path.stat().st_size,sha256=sha(path)))
 rna_path=path.parent/(project+'_clinical.csv');rna=read(rna_path)
 rnamap={r.get('barcode'):r for r in rna}
 rows=[]
 for bc in headers:
  r=rnamap.get(bc,{})
  preservation=norm(r.get('preservation_method'))
  ffpe=norm(r.get('is_ffpe')).lower()
  if 'ffpe' in preservation.lower() or 'formalin' in preservation.lower() or ffpe in ('yes','true'):priority=2
  elif preservation.lower() in ('frozen','fresh','fresh frozen','snap frozen') or ffpe in ('no','false'):priority=0
  else:priority=1
  rows.append(dict(project=project,patient=bc[:12],barcode=bc,sample_type_code=bc[13:15],preservation_method=preservation,is_ffpe=ffpe,preservation_priority=priority,selected=False,exclusion_reason='non_primary_01' if bc[13:15]!='01' else 'duplicate_primary_aliquot'))
 eligible=sorted([r for r in rows if r['sample_type_code']=='01'],key=lambda r:(r['patient'],r['preservation_priority'],r['barcode']))
 seen=set()
 for r in eligible:
  if r['patient'] not in seen:r['selected']=True;r['exclusion_reason']='';seen.add(r['patient'])
 manifest.extend(rows)
 inventory.append(dict(project=project,expression_path=str(path),matrix_samples=len(headers),primary_01_samples=len(eligible),selected_patients=len(seen),bytes=path.stat().st_size))
 by=defaultdict(lambda:defaultdict(list))
 for typ,rr in records.items():
  for r in rr:
   p=pid(r)
   if p.startswith('TCGA-'):by[p][typ].append(r)
 for patient,rr in sorted(by.items()):
  ix=rr.get('clinical_indexed',[]);xp=rr.get('clinicalXML_patient',[]);xf=rr.get('clinicalXML_followUp',[])
  assert len(ix)<=1,(project,patient,'duplicate_indexed')
  assert len(xp)<=1,(project,patient,'duplicate_XMLpatient')
  i=ix[0] if ix else {};x=xp[0] if xp else {};allr=ix+xp+xf
  ist=stage(i.get('ajcc_pathologic_stage'));xst=stage(x.get('stage_event_pathologic_stage'))
  st=ist if ist is not None else xst
  sr='indexed:ajcc_pathologic_stage' if ist is not None else ('XMLpatient:stage_event_pathologic_stage' if xst is not None else '')
  idxdead=vital(i) in ('dead','deceased');xml_dead=any(vital(r) in ('dead','deceased') for r in xp+xf)
  idxdeath=num(i.get('days_to_death')) if idxdead else None
  xdeath=sorted(set(v for r in xp+xf if vital(r) in ('dead','deceased') for v in [num(r.get('days_to_death'))] if v is not None and v>0))
  event=None;time=None;ossource='';osreason='';conflict=''
  if idxdead or xml_dead:
   if idxdeath is not None and idxdeath>0:
    event=1;time=idxdeath;ossource='indexed:days_to_death'
    if any(abs(z-idxdeath)>1e-8 for z in xdeath):conflict='indexed_XML_death_day_conflict_indexed_preferred'
   elif len(xdeath)==1:
    event=1;time=xdeath[0];ossource='XML:unique_days_to_death'
   elif len(xdeath)>1:osreason='conflicting_XML_death_days_without_indexed_resolution'
   else:osreason='known_dead_missing_positive_death_day'
  elif any(vital(r) in ('alive','living') for r in allr):
   ts=[(v,k,typ) for typ,rs in [('indexed',ix),('XMLpatient',xp),('XMLfollowUp',xf)] for r in rs for k in ['days_to_last_follow_up','days_to_last_followup','days_to_last_known_alive'] for v in [num(r.get(k))] if v is not None and v>0]
   if ts:
    t,k,typ=max(ts);event=0;time=t;ossource=typ+':'+k
   else:osreason='alive_missing_positive_followup'
  else:osreason='unknown_vital_status'
  age=num(i.get('age_at_diagnosis'));agesource='indexed:age_at_diagnosis_days'
  if age is not None:age/=365.25
  else:age=num(x.get('age_at_initial_pathologic_diagnosis'));agesource='XMLpatient:age_at_initial_pathologic_diagnosis_years'
  if age is not None and not 0<=age<=120:age=None;agesource='invalid_age'
  gender=norm(i.get('gender')).lower();sexsource='indexed:gender'
  if gender not in ('male','female'):gender=norm(x.get('gender')).lower();sexsource='XMLpatient:gender'
  if gender not in ('male','female'):gender='';sexsource='missing'
  clinical.append(dict(project=project,patient=patient,stage=st,stage_raw_indexed=norm(i.get('ajcc_pathologic_stage')),stage_raw_XML=norm(x.get('stage_event_pathologic_stage')),stage_source=sr,stage_conflict=ist is not None and xst is not None and ist!=xst,OS_time_days=time,OS_event=event,OS_source=ossource,OS_exclusion_reason=osreason,OS_death_day_conflict=conflict,OS_XML_updates_indexed_alive=(vital(i) in ('alive','living') and xml_dead),indexed_vital_status=vital(i),XML_death_days=';'.join(map(str,xdeath)),age=age,age_source=agesource,sex=gender,sex_source=sexsource,n_XML_followups=len(xf),indexed_file=str(cfiles['clinical_indexed'] or ''),XML_patient_file=str(cfiles['clinicalXML_patient'] or ''),XML_followup_file=str(cfiles['clinicalXML_followUp'] or '')))
write('sample_selection_manifest.tsv',manifest);write('clinical_patient_manifest.tsv',clinical);write('input_sources_sha256.tsv',sources);write('cohort_inventory.tsv',inventory)
summary={'projects':len(paths),'selected_primary_patients_by_project':{r['project']:r['selected_patients'] for r in inventory},'total_selected':sum(r['selected_patients'] for r in inventory),'plan_sha256':sha(ROOT/'ANALYSIS_PLAN.md'),'note':'No scores or outcome associations computed in preparation.'}
(ROOT/'preparation_summary.json').write_text(json.dumps(summary,indent=2))
print(json.dumps(summary,indent=2))
