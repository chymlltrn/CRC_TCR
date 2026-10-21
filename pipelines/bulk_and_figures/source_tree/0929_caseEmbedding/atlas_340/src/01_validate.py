#!/usr/bin/env python3
'Independent post-ingestion acceptance for the [PRIVATE_SUBJECT] + original External202 atlas.'
from __future__ import annotations
from release_runtime import private_literal as _private_literal, release_path as _release_path

import csv
import hashlib
import json
import os
import time
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
pa.set_cpu_count(8)

ROOT=Path(_release_path('path_99d8c1c6fde2a6b9'))
OLD=ROOT.parent

def require(ok,msg):
    if not ok:
        raise RuntimeError(msg)

def sha256(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(8*1024*1024),b''):
            h.update(block)
    return h.hexdigest()

def atomic_json(path,record):
    temp=path.with_name(path.name+'.partial')
    temp.write_text(json.dumps(record,ensure_ascii=False,indent=2,sort_keys=True)+'\n')
    os.replace(temp,path)

def main():
    start=time.time()
    with (ROOT/'config/cohort_samples.tsv').open(newline='') as f:
        rows=list(csv.DictReader(f,delimiter='\t'))
    require(len(rows)==340,'manifest must contain 340 libraries')
    require(len({r['patient_key'] for r in rows})==339,'339 patient identity keys expected')
    require(sum(r['cohort']=='home' for r in rows)==138
            and sum(r['cohort']=='external' for r in rows)==202,
            'cohort library count mismatch')
    require(sum(r['cohort']=='home' and r['recorded_n_binary']!='' for r in rows)==127
            and len({r['patient_key'] for r in rows
                     if r['cohort']=='home' and r['recorded_n_binary']!=''})==126,
            'Home known N patient count mismatch')
    bpair=[r for r in rows if r['sample_id'] in (_private_literal('private_52085b4d67e616fe'),_private_literal('private_db9850ff5b76a6f9'))]
    require(len(bpair)==2 and bpair[0]['patient_key']==bpair[1]['patient_key'],
            'Home duplicate patient pair not bound')
    summary_path=ROOT/'audit/01_ingest_summary.json'
    require(summary_path.is_file(),'ingestion summary not yet written')
    summary=json.loads(summary_path.read_text())
    union_path=ROOT/'data/aa_union.parquet'
    require(union_path.is_file(),'global AA union not yet written')
    union=pq.read_table(union_path,columns=['aa'])
    require(union.num_rows==summary['union_unique_aa']
            and sha256(union_path)==summary['union_sha256'],
            'union row count or hash mismatch')
    chunks=[]
    prev=None
    next_idx=0
    parquet=pq.ParquetFile(union_path)
    for rg in range(parquet.num_row_groups):
        tbl=parquet.read_row_group(rg,columns=['seq_idx','aa'])
        aa=tbl.column('aa').combine_chunks()
        ids=tbl.column('seq_idx').to_numpy(zero_copy_only=False)
        require(len(ids)==len(aa) and np.array_equal(
            ids,np.arange(next_idx,next_idx+len(ids),dtype=np.int64)),
            f'union sequence index discontinuity in group {rg}')
        require(aa.null_count==0,'null global AA')
        if len(aa)>1:
            require(pc.all(pc.greater(aa.slice(1),aa.slice(0,len(aa)-1))).as_py(),
                    f'global AA not strictly sorted in group {rg}')
        first,last=aa[0].as_py(),aa[-1].as_py()
        require(prev is None or prev<first,'global AA row group boundary not sorted')
        prev=last
        next_idx+=len(aa)
    require(next_idx==union.num_rows,'union row group coverage mismatch')
    qc_rows=[]
    all_aa_chunks=[]
    total_qc_reads=0
    total_qc_clones=0
    reused=0
    for row in rows:
        stem=row['cohort']+'__'+row['sample_id']
        clone=ROOT/'data/clone_detail'/f'{stem}.parquet'
        aa_path=ROOT/'data/patient_aa'/f'{stem}.parquet'
        receipt_path=ROOT/'audit/sample_receipts'/f'{stem}.json'
        require(clone.is_file() and aa_path.is_file() and receipt_path.is_file(),
                f'missing sample artifacts: {stem}')
        rec=json.loads(receipt_path.read_text())
        require(rec['status']=='PASS' and rec['patient_key']==row['patient_key'],
                f'nonpassing or wrong-patient receipt: {stem}')
        require(rec['source_tsv_path']==row['clone_tsv_path']
                and rec['source_clns_path']==row['clns_path'],
                f'source provenance mismatch: {stem}')
        require(clone.is_symlink()==aa_path.is_symlink()==receipt_path.is_symlink(),
                f'mixed symlink state: {stem}')
        reused+=int(clone.is_symlink())
        require(pq.ParquetFile(clone).metadata.num_rows==rec['source_clone_rows'],
                f'clone detail row count mismatch: {stem}')
        if not clone.is_symlink():
            detail=pq.read_table(clone,columns=['patient_key','qc_pass','read_count','aa'])
            flags=detail.column('qc_pass')
            require(pc.all(pc.equal(detail.column('patient_key'),row['patient_key'])).as_py(),
                    f'new clone detail patient mismatch: {stem}')
            require(int(pc.sum(flags).as_py() or 0)==rec['qc_clone_rows'],
                    f'new clone detail QC-clone count mismatch: {stem}')
            qualified_reads=pc.filter(detail.column('read_count'),flags)
            require(int(pc.sum(qualified_reads).as_py() or 0)==rec['qc_read_count'],
                    f'new clone detail QC-read mass mismatch: {stem}')
            qualified_aa=pc.filter(detail.column('aa'),flags)
            require(qualified_aa.null_count==0,
                    f'new clone detail accepted null AA: {stem}')
        tbl=pq.read_table(aa_path,columns=['cohort','sample_id','patient_key',
                                            'aa','read_count'])
        require(tbl.num_rows==rec['unique_aa'],'per-sample AA count mismatch: '+stem)
        require(pc.all(pc.equal(tbl.column('cohort'),row['cohort'])).as_py()
                and pc.all(pc.equal(tbl.column('sample_id'),row['sample_id'])).as_py()
                and pc.all(pc.equal(tbl.column('patient_key'),row['patient_key'])).as_py(),
                f'sample or patient rejoin mismatch: {stem}')
        aa=tbl.column('aa').combine_chunks()
        require(aa.null_count==0 and
                (len(aa)<2 or pc.all(pc.greater(aa.slice(1),aa.slice(0,len(aa)-1))).as_py()),
                f'AA not strictly sorted/unique: {stem}')
        read_count=tbl.column('read_count')
        require(read_count.null_count==0 and
                pc.all(pc.greater(read_count,0)).as_py(),
                f'nonpositive/null AA count: {stem}')
        require(int(pc.sum(read_count).as_py() or 0)==rec['qc_read_count'],
                f'AA/readCount mass mismatch: {stem}')
        all_aa_chunks.extend(tbl.column('aa').chunks)
        total_qc_reads+=rec['qc_read_count']
        total_qc_clones+=rec['qc_clone_rows']
        qc_rows.append(dict(cohort=row['cohort'],sample_id=row['sample_id'],
            patient_key=row['patient_key'],status='PASS',
            input_mode=('reused_read_only_symlink' if clone.is_symlink() else 'new_QC'),
            recorded_n_binary=row['recorded_n_binary'],
            source_clone_rows=rec['source_clone_rows'],
            productive_clone_rows=rec['productive_clone_rows'],
            qc_clone_rows=rec['qc_clone_rows'],
            qc_assigned_reads=rec['qc_read_count'],unique_aa=rec['unique_aa'],
            source_tsv_sha256=rec['source_tsv_sha256'],
            source_clns_sha256=rec['source_clns_sha256'],
            receipt_path=str(receipt_path)))
    require(reused==277,'expected exactly 277 reused sample partitions')
    require(total_qc_reads==summary['total_qc_assigned_reads']
            and total_qc_clones==summary['total_qc_clones'],
            'sum of sample receipts differs from summary')
    all_aa=pa.chunked_array(all_aa_chunks,type=pa.string())
    require(pc.all(pc.is_in(all_aa,value_set=union.column('aa'))).as_py(),
            'at least one sample AA missing from global union')
    qc_path=ROOT/'audit/01_qc_by_library.tsv'
    tmp=qc_path.with_name(qc_path.name+'.partial')
    with tmp.open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(qc_rows[0]),delimiter='\t')
        writer.writeheader()
        writer.writerows(qc_rows)
    os.replace(tmp,qc_path)
    input_manifest=ROOT/'audit/01_input_sha256.tsv'
    with input_manifest.open(newline='') as f:
        source_rows=list(csv.DictReader(f,delimiter='\t'))
    require(len(source_rows)==1020 and
            sha256(input_manifest)==summary['input_sha256_manifest_sha256'],
            'input hash manifest count/hash mismatch')
    require(summary['cohort_samples_sha256']==sha256(ROOT/'config/cohort_samples.tsv')
            and summary['source_contract_sha256']==sha256(ROOT/'audit/01_source_contract.json'),
            'source or cohort contract hash mismatch')
    validation=dict(schema='crc_tcr_atlas340_stage1_validation_v1',
        status='PASS',libraries=340,patients=339,
        home_known_n_patients=126,external_known_n_patients=163,
        reused_sample_partitions=277,new_sample_partitions=63,
        total_qc_clones=total_qc_clones,total_qc_assigned_reads=total_qc_reads,
        union_unique_aa=union.num_rows,
        sample_aa_rows=len(all_aa),all_sample_aa_rejoin_union=True,
        per_sample_aa_strictly_sorted_unique=True,
        new_clone_detail_qc_count_and_read_mass_rechecked=True,
        union_aa_strictly_sorted_unique=True,
        union_seq_idx_contiguous=True,
        home_duplicate_pair_bound=True,
        source_hash_rows=1020,
        qc_library_table=str(qc_path),qc_library_table_sha256=sha256(qc_path),
        wall_time_s=round(time.time()-start,1))
    atomic_json(ROOT/'audit/01_stage1_validation.json',validation)
    summary['status']='complete'
    summary['schema']='crc_tcr_atlas340_ingest_v1'
    summary['validation_path']=str(ROOT/'audit/01_stage1_validation.json')
    summary['validation_sha256']=sha256(ROOT/'audit/01_stage1_validation.json')
    atomic_json(summary_path,summary)
    print(json.dumps(validation,indent=2))
if __name__=='__main__':
    main()
