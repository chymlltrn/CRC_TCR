#!/usr/bin/env python3
'Full MiXCR TRB clone ingestion for [PRIVATE_SUBJECT] and original External202.\n\nOnly the target analysis directory is written. Raw clone TSV/CLNS stay read only.\nThe response variable is recorded N, whose pN/ypN prefix remains unverified.\n'
from __future__ import annotations
from release_runtime import private_literal as _private_literal, release_path as _release_path
import argparse
import csv
import hashlib
import json
import os
import subprocess
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

BASE = Path(_release_path('path_39343c5e69c3d939'))
OLD_ROOT = BASE / '0929_caseEmbedding'
ROOT = OLD_ROOT / 'atlas_340'
COHORT = BASE / '0903_pN_multilayer/0928_recordedN_completion_v1/cohort'
HOME_LEDGER = COHORT / 'HOME138_SOURCE_SEQUENCE_LEDGER.tsv'
EXT_LEDGER = COHORT / 'SRP215_SOURCE_SEQUENCE_LEDGER_v2.tsv'
EXT_MANIFEST = BASE / '0828_external_validation/raw_data/11_unified_202_downstream/unified_202_manifest.tsv'
RECOVERED = BASE / '0903_pN_multilayer/0928_recordedN_completion_v1/external/raw_recovery/mixcr'
MIXCR_JAR = BASE / '260818_data/tools/mixcr-4.7.0/mixcr.jar'
HOME_CLINICAL = Path(_release_path('path_d99b31f433ea263e'))
HOME_LIBRARY = Path(_release_path('path_a2730582fc59ec04'))
EXTERNAL_SUPPLEMENT = Path(_release_path('path_4dadd066fc8c160b'))
EXTERNAL_SRA_INFO = Path(_release_path('path_5fd6b340b7dbb8db'))
HOME_YAML = BASE / '260818_data/08_manifests/rna-seq_hs_mixcr470.yaml'
EXTERNAL_YAML = BASE / '0828_external_validation/raw_data/09_manifests/rna-seq_hs_mixcr470.yaml'

FIELDS = ['cloneId','readCount','allVHitsWithScore','allJHitsWithScore','nSeqCDR3','aaSeqCDR3']
CLONE_COLS = ['cohort','sample_id','patient_key','clone_id','aa','nt_cdr3','read_count',
              'productive','canonical_aa','qc_pass','v_gene','j_gene',
              'all_v_hits_with_score','all_j_hits_with_score']
AA_COLS = ['cohort','sample_id','patient_key','aa','read_count']
MANIFEST_COLS = ['cohort','sample_id','patient_key','identity_basis',
                 'recorded_n_binary','original_label_raw','label_scope',
                 'age_years','sex','neoadjuvant','home_batch','technical_batch',
                 'source_run','source_biosample','upstream_processing',
                 'clone_tsv_path','clns_path','mixcr_report_path','mixcr_yaml_path',
                 'source_ledger_path','expected_unique_aa','expected_read_count']

def require(ok, message):
    if not ok:
        raise RuntimeError(message)

def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()

def file_stat(path):
    s = Path(path).stat()
    return {'bytes':s.st_size,'mtime_ns':s.st_mtime_ns}

def atomic_json(path, value):
    temp = path.with_name(path.name + '.partial')
    temp.write_text(json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + '\n')
    os.replace(temp, path)

def read_tsv(path):
    with Path(path).open(newline='') as stream:
        return list(csv.DictReader(stream, delimiter='\t'))

def paths_for(row):
    stem = row['cohort'] + '__' + row['sample_id']
    return (ROOT/'data/clone_detail'/f'{stem}.parquet',
            ROOT/'data/patient_aa'/f'{stem}.parquet',
            ROOT/'audit/sample_receipts'/f'{stem}.json')

def _cell(value):
    if pd.isna(value):
        return ''
    return str(value).strip()

def _write_locked_tsv(path, rows, fields):
    temp=path.with_name(path.name+'.partial')
    with temp.open('w',newline='') as stream:
        writer=csv.DictWriter(stream,delimiter='\t',fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    if path.exists():
        require(path.read_bytes()==temp.read_bytes(),f'locked manifest changed: {path}')
        temp.unlink()
    else:
        os.replace(temp,path)

def _check_mixcr_report(path):
    body=Path(path).read_text(errors='replace')
    require('Version: 4.7.0' in body
            and '--preset rna-seq' in body and '--species hs' in body,
            f'MiXCR report parameter drift: {path}')
    return body

def build_manifest():
    home=read_tsv(HOME_LEDGER)
    external_orig=read_tsv(EXT_MANIFEST)
    external_ledger={r['sample_name']:r for r in read_tsv(EXT_LEDGER)}
    require(len(home)==138 and len(external_orig)==202,
            'original cohort membership count drift')
    require(len(external_ledger)==215,'external source ledger count drift')
    clinical=pd.read_excel(HOME_CLINICAL,sheet_name='Sheet1',
                           header=1,dtype=str).fillna('')
    library=pd.read_excel(HOME_LIBRARY,sheet_name=0,header=0,dtype=str).fillna('')
    require(len(clinical)==138 and len(library)==138,
            'Home clinical/library sheet row count drift')
    require(sha256(HOME_YAML)==sha256(EXTERNAL_YAML),
            'MiXCR YAML differs between cohorts')
    old={r['cohort']+'__'+r['sample_id']:r for r in
         read_tsv(OLD_ROOT/'config/cohort_samples.tsv')}
    rows=[]
    home_subjects={}
    for r in home:
        sid=r['sample_id']
        ci=int(r['clinical_sheet_row'])-3
        li=int(r['library_sheet_row'])-2
        require(0<=ci<len(clinical) and 0<=li<len(library),
                f'Home workbook index invalid: {sid}')
        clin=clinical.iloc[ci]
        lib=library.iloc[li]
        subject=_cell(clin['样品名称'])
        require(subject and subject==_cell(lib['样品名称'])
                and sid==_cell(lib['康测编号']),
                f'Home exact clinical/library join failed: {sid}')
        home_subjects.setdefault((_cell(clin['name']),_cell(clin['age']),
                                  _cell(clin['sex']).lower()),[]).append(sid)
        n_raw=_cell(clin['N'])
        n_binary='' if n_raw in ('','9999','NA') else ('0' if n_raw=='0' else '1'
                    if n_raw in ('1','2') else '')
        require(n_binary==r['pathology_report_N_binary'],
                f'Home N label differs from source ledger: {sid}')
        neo_raw=_cell(clin['Neoadjuvant thearpy']).lower()
        neo='yes' if neo_raw=='y' else 'no' if neo_raw in ('n','no') else 'unknown'
        require(neo==r['neoadjuvant_normalized'],
                f'Home neoadjuvant label differs from source ledger: {sid}')
        key=r['patient_key'] or _private_literal('private_283790eb0a2c6510')
        require(r['patient_key'] or sid in (_private_literal('private_52085b4d67e616fe'),_private_literal('private_db9850ff5b76a6f9')),
                f'Home unmapped patient outside duplicate pair: {sid}')
        age=_cell(clin['age'])
        if age in ('9999','NA'): age=''
        sex=_cell(clin['sex']).lower()
        if sex in ('9999','na'): sex=''
        require((not age or (age.isdigit() and 0<int(age)<120))
                and sex in ('','m','f'),
                f'Home age/sex invalid: {sid}')
        row=dict(cohort='home',sample_id=sid,patient_key=key,
            identity_basis=('exact_source_subject_id_duplicate_pair'
                            if sid in (_private_literal('private_52085b4d67e616fe'),_private_literal('private_db9850ff5b76a6f9'))
                            else 'exact_source_subject_id_deidentified'),
            recorded_n_binary=n_binary,original_label_raw=n_raw,
            label_scope='recorded_N_c_p_yp_unverified',
            age_years=age,sex=sex,neoadjuvant=neo,
            home_batch=sid[:6],technical_batch=sid[:6],
            source_run='',source_biosample='',
            upstream_processing='Home_UID_deduplicated_paired_FASTQ',
            clone_tsv_path=str(BASE/'260818_data/06_clones_tables'/f'{sid}.clones_TRB.tsv'),
            clns_path=str(BASE/'260818_data/05_clns'/f'{sid}.contigs.clns'),
            mixcr_report_path=str(BASE/'260818_data/07_reports'/f'{sid}.align.report.txt'),
            mixcr_yaml_path=str(HOME_YAML),source_ledger_path=str(HOME_LEDGER),
            expected_unique_aa=r['all_productive_unique_aa'],
            expected_read_count=r['all_productive_count'])
        rows.append(row)
    duplicates=sorted(sorted(v) for v in home_subjects.values() if len(v)>1)
    require(duplicates==[[_private_literal('private_52085b4d67e616fe'),_private_literal('private_db9850ff5b76a6f9')]],
            'Home exact subject duplicate structure drift')
    require(clinical.iloc[int(next(x for x in home if x['sample_id']==_private_literal('private_52085b4d67e616fe'))
            ['clinical_sheet_row'])-3]['name']
            ==clinical.iloc[int(next(x for x in home if x['sample_id']==_private_literal('private_db9850ff5b76a6f9'))
            ['clinical_sheet_row'])-3]['name'],
            _private_literal('private_eb2d4490a07d3f78'))
    for r in external_orig:
        sid=r['sample_id']
        require(sid in external_ledger,f'external original sample missing ledger: {sid}')
        ext=external_ledger[sid]
        require(r['output_valid']=='True' and r['biosample']==ext['biosample']
                and r['srr']==ext['run_accession']
                and ext['in_prior_external202']=='yes',
                f'external original source identity mismatch: {sid}')
        n_raw=r['N']
        n_binary='' if n_raw in ('NA','') else ('0' if n_raw=='N0'
                    else '1' if n_raw in ('N1','N2') else '')
        require(n_binary==ext['recorded_N_binary'],
                f'external N label differs from ledger: {sid}')
        key=ext['current_identity_group_key']
        require(key=='external_biosample_'+r['biosample'],
                f'external identity key mismatch: {sid}')
        rows.append(dict(cohort='external',sample_id=sid,patient_key=key,
            identity_basis='study_one_patient_per_library_BioSample',
            recorded_n_binary=n_binary,original_label_raw=n_raw,
            label_scope='recorded_N_c_p_yp_unverified',
            age_years='',sex='',neoadjuvant='unknown',home_batch='',
            technical_batch=sid.split('_',1)[0],
            source_run=r['srr'],source_biosample=r['biosample'],
            upstream_processing='External_SRA_fasterq_or_ENA_paired_FASTQ_UID_equivalence_unverified',
            clone_tsv_path=r['canonical_trb_path'],
            clns_path=r['canonical_clns_path'],
            mixcr_report_path=r['canonical_align_report_path'],
            mixcr_yaml_path=str(EXTERNAL_YAML),
            source_ledger_path=str(EXT_LEDGER),
            expected_unique_aa=ext['all_productive_unique_aa'],
            expected_read_count=ext['all_productive_count']))
    rows.sort(key=lambda r:(r['cohort'],r['sample_id']))
    require(len(rows)==340 and len({(r['cohort'],r['sample_id']) for r in rows})==340,
            'atlas sample count or uniqueness drift')
    require(len({r['patient_key'] for r in rows})==339
            and all(r['patient_key'] for r in rows),
            'patient identity key count drift')
    for cohort,n0,n1,n_missing in [('home',70,57,11),('external',109,54,39)]:
        part=[r for r in rows if r['cohort']==cohort]
        require((sum(r['recorded_n_binary']=='0' for r in part),
                 sum(r['recorded_n_binary']=='1' for r in part),
                 sum(r['recorded_n_binary']=='' for r in part))
                ==(n0,n1,n_missing),f'{cohort} N count drift')
    for r in rows:
        require((not r['expected_unique_aa']
                and not r['expected_read_count']
                and r['sample_id'] in (_private_literal('private_52085b4d67e616fe'),_private_literal('private_db9850ff5b76a6f9')))
                or (r['expected_unique_aa'].isdigit()
                    and r['expected_read_count'].isdigit()),
                f'missing productive count outside Home duplicate pair: {r["sample_id"]}')
        for field in ('clone_tsv_path','clns_path','mixcr_report_path'):
            require(Path(r[field]).is_file(),
                    f'missing MiXCR {field}: {r["sample_id"]}')
        _check_mixcr_report(r['mixcr_report_path'])
        oldrow=old.get(r['cohort']+'__'+r['sample_id'])
        if oldrow is not None:
            require(oldrow['patient_key']==r['patient_key']
                    and oldrow['clone_tsv_path']==r['clone_tsv_path']
                    and oldrow['clns_path']==r['clns_path']
                    and oldrow['recorded_n_binary']==r['recorded_n_binary'],
                    f'old and new sample provenance mismatch: {r["sample_id"]}')
    path=ROOT/'config/cohort_samples.tsv'
    _write_locked_tsv(path,rows,MANIFEST_COLS)
    sources=[HOME_LEDGER,EXT_LEDGER,EXT_MANIFEST,HOME_CLINICAL,HOME_LIBRARY,
             EXTERNAL_SUPPLEMENT,EXTERNAL_SRA_INFO,MIXCR_JAR,HOME_YAML,EXTERNAL_YAML,
             OLD_ROOT/'config/cohort_samples.tsv']
    contract=dict(schema='crc_tcr_atlas340_source_v1',
        home_libraries=138,home_patients=137,
        external_libraries=202,external_patients=202,
        home_known_n_libraries=127,home_known_n_patients=126,
        external_known_n_patients=163,
        endpoint='recorded N, pN/ypN prefix unverified',
        home_duplicate_pair=[_private_literal('private_52085b4d67e616fe'),_private_literal('private_db9850ff5b76a6f9')],
        home_duplicate_evidence='same clinical name, age and sex; different sample identifiers',
        external_identity='study asserts one patient per library; BioSample key',
        mixcr_version='4.7.0',mixcr_preset='analyze rna-seq --species hs',
        mixcr_yaml_sha256=sha256(HOME_YAML),
        upstream_difference='Home UID-deduplicated paired FASTQ; External historical SRA/ENA paired FASTQ, equivalent UID deduplication not documented',
        source_hashes={str(p):sha256(p) for p in sources},
        cohort_manifest_sha256=sha256(path),
        prior_reused_original_list_samples=len(old)-2)
    contract_path=ROOT/'audit/01_source_contract.json'
    if contract_path.exists():
        require(json.loads(contract_path.read_text())==contract,'source contract drift')
    else:
        atomic_json(contract_path,contract)
    return rows

def link_reusable_samples(rows):
    old_rows={r['cohort']+'__'+r['sample_id']:r for r in
              read_tsv(OLD_ROOT/'config/cohort_samples.tsv')}
    linked=0
    for row in rows:
        stem=row['cohort']+'__'+row['sample_id']
        if stem not in old_rows:
            continue
        old=old_rows[stem]
        require(old['patient_key']==row['patient_key'],
                f'reused patient key mismatch: {stem}')
        dests=paths_for(row)
        sources=(OLD_ROOT/'data/clone_detail'/f'{stem}.parquet',
                 OLD_ROOT/'data/patient_aa'/f'{stem}.parquet',
                 OLD_ROOT/'audit/sample_receipts'/f'{stem}.json')
        for source,dest in zip(sources,dests):
            require(source.is_file(),f'old reusable artifact missing: {source}')
            if dest.is_symlink():
                require(dest.resolve()==source.resolve(),
                        f'reused symlink changed: {dest}')
            elif dest.exists():
                require(sha256(dest)==sha256(source),
                        f'reused artifact content mismatch: {dest}')
            else:
                dest.symlink_to(source.resolve())
        require(valid_receipt(row) is not None,
                f'old reusable receipt invalid: {stem}')
        linked+=1
    require(linked==277,f'expected 277 original-list reusable samples, got {linked}')
    return linked

def top_gene(series):
    return (series.str.split(',',n=1).str[0]
            .str.split('(',n=1).str[0]
            .str.split('*',n=1).str[0].fillna(''))

def valid_receipt(row):
    clone,aa,receipt = paths_for(row)
    if not (clone.is_file() and aa.is_file() and receipt.is_file()):
        return None
    try:
        old=json.loads(receipt.read_text())
        if old.get('status')!='PASS':
            return None
        if old['source_tsv_stat']!=file_stat(row['clone_tsv_path']):
            return None
        if old['source_clns_stat']!=file_stat(row['clns_path']):
            return None
        if old['clone_detail_sha256']!=sha256(clone):
            return None
        if old['patient_aa_sha256']!=sha256(aa):
            return None
        return old
    except (KeyError,ValueError,OSError):
        return None

def process_one(row):
    pa.set_cpu_count(1)
    sid,cohort,key=row['sample_id'],row['cohort'],row['patient_key']
    clone_out,aa_out,receipt_out=paths_for(row)
    old=valid_receipt(row)
    if old is not None:
        return old
    stem=f'{cohort}__{sid}'
    source_tsv=Path(row['clone_tsv_path'])
    source_clns=Path(row['clns_path'])
    decoded=ROOT/'tmp'/f'{stem}.decoded.tsv'
    partial=ROOT/'tmp'/f'{stem}.decoded.partial.tsv'
    java_tmp=ROOT/'tmp'/f'{stem}.java'
    java_tmp.mkdir(parents=True,exist_ok=True)
    decoded.unlink(missing_ok=True)
    partial.unlink(missing_ok=True)
    command=['java','-Xmx4g','-XX:ActiveProcessorCount=2',
             f'-Djava.io.tmpdir={java_tmp}','-jar',str(MIXCR_JAR),
             'exportClones','--chains','TRB','--dont-split-files',
             '--drop-default-fields','-cloneId','-readCount',
             '-vHitsWithScore','-jHitsWithScore','-nFeature','CDR3',
             '-aaFeature','CDR3','-isProductive','CDR3',
             str(source_clns),str(partial)]
    log=ROOT/'audit/logs'/f'{stem}.mixcr.log'
    start=time.time()
    env=os.environ.copy()
    env.update(TMPDIR=str(ROOT/'tmp'),XDG_CACHE_HOME=str(ROOT/'tmp'),
               OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',
               MKL_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1')
    with log.open('w') as stream:
        result=subprocess.run(command,stdout=stream,stderr=subprocess.STDOUT,
                              env=env,check=False)
    require(result.returncode==0 and partial.is_file(),
            f'MiXCR export failed {stem}; see {log}')
    os.replace(partial,decoded)
    with source_tsv.open() as stream:
        cols=stream.readline().rstrip('\n').split('\t')
    require(set(FIELDS).issubset(cols),f'source TSV schema {stem}')
    source=pd.read_csv(source_tsv,sep='\t',usecols=FIELDS,dtype=str,
                       keep_default_na=False,low_memory=False)
    export=pd.read_csv(decoded,sep='\t',dtype=str,
                       keep_default_na=False,low_memory=False)
    require(set(FIELDS+['isProductiveCDR3']).issubset(export.columns),
            f'export schema {stem}')
    source=source.sort_values('cloneId',kind='mergesort').reset_index(drop=True)
    export=export.sort_values('cloneId',kind='mergesort').reset_index(drop=True)
    require(len(source)==len(export) and source.cloneId.is_unique,
            f'cloneId duplication or row loss {stem}')
    require(source[FIELDS].equals(export[FIELDS]),
            f'MiXCR six-field cloneId mismatch {stem}')
    clone_id=pd.to_numeric(source.cloneId,errors='coerce')
    require(clone_id.notna().all() and (clone_id>=0).all() and (clone_id%1==0).all(),
            f'invalid cloneId {stem}')
    counts=pd.to_numeric(source.readCount,errors='coerce')
    require(counts.notna().all() and np.isfinite(counts.to_numpy()).all()
            and (counts>=0).all() and (counts<=2**53).all() and (counts%1==0).all(),
            f'invalid readCount {stem}')
    read_count=counts.astype('int64')
    positive_count=read_count.gt(0)
    prod_text=export.isProductiveCDR3.str.lower()
    require(prod_text.isin(['true','false']).all(),f'bad productivity flag {stem}')
    productive=prod_text.eq('true')
    canonical=source.aaSeqCDR3.str.fullmatch('[ACDEFGHIKLMNPQRSTVWY]+').fillna(False)
    in_frame=source.nSeqCDR3.str.len().mod(3).eq(0)
    qc_pass=productive & canonical & in_frame & positive_count
    v_gene=top_gene(source.allVHitsWithScore)
    j_gene=top_gene(source.allJHitsWithScore)
    # AA-level QC does not require V/J. V-aware analyses must separately require
    # v_gene beginning TRBV and j_gene beginning TRBJ for each clone.
    detail=pd.DataFrame({
        'cohort':cohort,'sample_id':sid,'patient_key':key,
        'clone_id':clone_id.astype('int64'),
        'aa':source.aaSeqCDR3,'nt_cdr3':source.nSeqCDR3,
        'read_count':read_count,'productive':productive.to_numpy(dtype=bool),
        'canonical_aa':canonical.to_numpy(dtype=bool),
        'qc_pass':qc_pass.to_numpy(dtype=bool),
        'v_gene':v_gene,'j_gene':j_gene,
        'all_v_hits_with_score':source.allVHitsWithScore,
        'all_j_hits_with_score':source.allJHitsWithScore})[CLONE_COLS]
    aa=(detail.loc[detail.qc_pass,['aa','read_count']]
        .groupby('aa',sort=True,as_index=False)['read_count'].sum())
    aa.insert(0,'patient_key',key)
    aa.insert(0,'sample_id',sid)
    aa.insert(0,'cohort',cohort)
    aa=aa[AA_COLS]
    qc_reads=int(detail.loc[detail.qc_pass,'read_count'].sum())
    require(aa.aa.is_unique and int(aa.read_count.sum())==qc_reads,
            f'AA aggregation mass or uniqueness {stem}')
    if row['expected_unique_aa']:
        require(len(aa)==int(row['expected_unique_aa'])
                and qc_reads==int(row['expected_read_count']),
                f'productive AA count drift {stem}: {len(aa)}/{qc_reads} versus '
                f'{row["expected_unique_aa"]}/{row["expected_read_count"]}')
    clone_temp=clone_out.with_name(clone_out.name+'.partial')
    aa_temp=aa_out.with_name(aa_out.name+'.partial')
    detail.to_parquet(clone_temp,index=False,compression='zstd')
    aa.to_parquet(aa_temp,index=False,compression='zstd')
    require(pq.ParquetFile(clone_temp).metadata.num_rows==len(detail)
            and pq.ParquetFile(aa_temp).metadata.num_rows==len(aa),
            f'Parquet row count {stem}')
    os.replace(clone_temp,clone_out)
    os.replace(aa_temp,aa_out)
    receipt=dict(
        schema='crc_tcr_full_clones_sample_v1',status='PASS',
        cohort=cohort,sample_id=sid,patient_key=key,
        recorded_n_binary=(int(row['recorded_n_binary']) if row['recorded_n_binary'] else None),
        source_tsv_path=str(source_tsv),source_tsv_stat=file_stat(source_tsv),
        source_tsv_sha256=sha256(source_tsv),
        source_clns_path=str(source_clns),source_clns_stat=file_stat(source_clns),
        source_clns_sha256=sha256(source_clns),
        mixcr_command=command,mixcr_export_sha256=sha256(decoded),
        mixcr_log=str(log),six_fields_equal_by_clone_id=True,
        source_clone_rows=int(len(detail)),
        productive_clone_rows=int(productive.sum()),
        nonproductive_clone_rows=int((~productive).sum()),
        productive_noncanonical_rows=int((productive & ~canonical).sum()),
        productive_out_of_frame_rows=int((productive & ~in_frame).sum()),
        source_zero_read_count_rows=int((~positive_count).sum()),
        productive_zero_read_count_rows=int((productive & ~positive_count).sum()),
        qc_clone_rows=int(qc_pass.sum()),qc_read_count=qc_reads,
        unique_aa=int(len(aa)),
        min_aa_length=(int(aa.aa.str.len().min()) if len(aa) else None),
        max_aa_length=(int(aa.aa.str.len().max()) if len(aa) else None),
        clone_detail_path=str(clone_out),
        clone_detail_sha256=sha256(clone_out),
        patient_aa_path=str(aa_out),
        patient_aa_sha256=sha256(aa_out),
        elapsed_s=round(time.time()-start,1),
        filter_rule='MiXCR isProductiveCDR3=true; canonical 20-aa alphabet; '
                    'in-frame nSeqCDR3; positive integer readCount; no count or length cutoff',
        count_unit='MiXCR readCount, assigned reads rather than UMI or cells')
    atomic_json(receipt_out,receipt)
    decoded.unlink(missing_ok=True)
    return receipt

def build_union(rows):
    chunks=[]
    for row in rows:
        _,path,_=paths_for(row)
        table=pq.read_table(path,columns=['aa'])
        chunks.extend(table.column('aa').chunks)
    all_aa=pa.chunked_array(chunks,type=pa.string())
    unique=pc.unique(all_aa)
    require(unique.null_count==0,'null AA in per-sample AA table')
    order=pc.sort_indices(pa.table({'aa':unique}),sort_keys=[('aa','ascending')])
    sorted_aa=pc.take(unique,order)
    n=len(sorted_aa)
    table=pa.table({'seq_idx':pa.array(np.arange(n,dtype=np.int64)),
                    'aa':sorted_aa})
    dest=ROOT/'data/aa_union.parquet'
    temp=dest.with_name(dest.name+'.partial')
    pq.write_table(table,temp,compression='zstd',row_group_size=1_000_000)
    require(pq.ParquetFile(temp).metadata.num_rows==n,
            'global AA union Parquet row count')
    os.replace(temp,dest)
    receipts=[json.loads(paths_for(r)[2].read_text()) for r in rows]
    by_sample={(r['cohort'],r['sample_id']):r for r in rows}
    input_manifest=ROOT/'audit/01_input_sha256.tsv'
    input_temp=input_manifest.with_name(input_manifest.name+'.partial')
    input_fields=['cohort','sample_id','artifact_type','source_path','resolved_path',
                  'bytes','mtime_ns','sha256']
    with input_temp.open('w',newline='') as stream:
        writer=csv.DictWriter(stream,delimiter='\t',fieldnames=input_fields)
        writer.writeheader()
        for rec in receipts:
            for artifact,prefix in [('clone_tsv','source_tsv'),('clns','source_clns')]:
                source=Path(rec[prefix+'_path'])
                stat=rec[prefix+'_stat']
                writer.writerow(dict(cohort=rec['cohort'],sample_id=rec['sample_id'],
                    artifact_type=artifact,source_path=str(source),
                    resolved_path=str(source.resolve()),bytes=stat['bytes'],
                    mtime_ns=stat['mtime_ns'],sha256=rec[prefix+'_sha256']))
            report=Path(by_sample[(rec['cohort'],rec['sample_id'])]['mixcr_report_path'])
            report_stat=file_stat(report)
            writer.writerow(dict(cohort=rec['cohort'],sample_id=rec['sample_id'],
                artifact_type='mixcr_align_report',source_path=str(report),
                resolved_path=str(report.resolve()),bytes=report_stat['bytes'],
                mtime_ns=report_stat['mtime_ns'],sha256=sha256(report)))
    os.replace(input_temp,input_manifest)
    summary=dict(
        schema='crc_tcr_atlas340_ingest_v1',status='complete',
        n_samples=len(rows),home_samples=138,external_samples=202,
        n_patients=len({r['patient_key'] for r in rows}),
        union_unique_aa=n,union_path=str(dest),union_sha256=sha256(dest),
        input_sha256_manifest=str(input_manifest),
        input_sha256_manifest_sha256=sha256(input_manifest),
        input_sha256_manifest_rows=3*len(rows),
        total_qc_clones=sum(r['qc_clone_rows'] for r in receipts),
        total_qc_assigned_reads=sum(r['qc_read_count'] for r in receipts),
        total_productive_noncanonical_rows=sum(r['productive_noncanonical_rows'] for r in receipts),
        total_productive_out_of_frame_rows=sum(r['productive_out_of_frame_rows'] for r in receipts),
        total_source_zero_read_count_rows=sum(r.get('source_zero_read_count_rows',0) for r in receipts),
        total_productive_zero_read_count_rows=sum(r.get('productive_zero_read_count_rows',0) for r in receipts),
        sum_per_sample_unique_aa=sum(r['unique_aa'] for r in receipts),
        patient_aa_columns=AA_COLS,clone_detail_columns=CLONE_COLS,
        source_contract_sha256=sha256(ROOT/'audit/01_source_contract.json'),
        cohort_samples_sha256=sha256(ROOT/'config/cohort_samples.tsv'),
        aa_union_columns=['seq_idx','aa'],
        aa_union_order='AA lexicographic ascending; zero-based contiguous int64 seq_idx',
        label_use='none for sequence QC or AA union')
    atomic_json(ROOT/'audit/01_ingest_summary.json',summary)
    return summary

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--workers',type=int,default=4)
    parser.add_argument('--samples',nargs='*',help='sample IDs for a resumable smoke run')
    args=parser.parse_args()
    require(1<=args.workers<=8,'workers must be 1..8, two Java CPUs each')
    rows=build_manifest()
    print(f'linked_old_samples={link_reusable_samples(rows)}',flush=True)
    if args.samples:
        requested=set(args.samples)
        selected=[r for r in rows if r['sample_id'] in requested]
        require(len(selected)==len(requested),'unknown sample ID in --samples')
    else:
        selected=rows
    pending=[r for r in selected if valid_receipt(r) is None]
    print(f'cohort={len(rows)} selected={len(selected)} pending={len(pending)}',flush=True)
    if pending:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futures={pool.submit(process_one,r):r for r in pending}
            for future in as_completed(futures):
                r=futures[future]
                try:
                    result=future.result()
                except Exception as exc:
                    stem=r['cohort']+'__'+r['sample_id']
                    atomic_json(ROOT/'audit/sample_receipts'/f'{stem}.failure.json',
                        dict(status='FAIL',cohort=r['cohort'],sample_id=r['sample_id'],
                             error_type=type(exc).__name__,error=str(exc)))
                    print(f'{stem}: FAILED {type(exc).__name__}: {exc}',flush=True)
                    continue
                print(f'{r["cohort"]}/{r["sample_id"]}: {result["unique_aa"]:,} AA, '
                      f'{result["qc_clone_rows"]:,} QC clones',flush=True)
    if all(valid_receipt(r) is not None for r in rows):
        print(json.dumps(build_union(rows),ensure_ascii=False,indent=2),flush=True)
    else:
        print('Subset complete; global union awaits all 340 receipts.',flush=True)

if __name__=='__main__':
    main()
