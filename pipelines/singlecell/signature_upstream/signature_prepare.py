#!/usr/bin/env python3
"""Build raw all-RNA baseline CRC tumour donor/subtype pseudobulks."""
from release_config import release_path, private_value
import argparse
from datetime import datetime, timezone
import gzip
import hashlib
import json
from pathlib import Path
import subprocess
import tempfile

import numpy as np
import pandas as pd

ROOT = release_path('signature_upstream_root')
OUT = ROOT / 'analysis/signature'
SOURCE = Path(str(release_path('signature_prepare_source_17')))
INVENTORY = Path(str(release_path('signature_prepare_source_18')))


def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024), b''):h.update(b)
    return h.hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')


def stream_test(exe):
    with tempfile.TemporaryDirectory() as tmp:
        p=Path(tmp)
        with gzip.open(p/'x.gz','wt') as f:
            f.write('%%MatrixMarket matrix coordinate integer general\n% duplicate-symbol fixture\n3 3 5\n1 1 2\n2 1 3\n3 2 4\n1 3 5\n3 3 7\n')
        np.array([0,0,1],dtype=np.int32).tofile(p/'gene.bin')
        np.array([0,-1,1],dtype=np.int32).tofile(p/'cell.bin')
        subprocess.run([str(exe),str(p/'x.gz'),str(p/'gene.bin'),str(p/'cell.bin'),'2','2',str(p/'test')],check=True)
        np.testing.assert_array_equal(np.fromfile(p/'test_counts.bin',dtype=np.int64).reshape(2,2), [[5,0],[5,7]])
        np.testing.assert_array_equal(np.fromfile(p/'test_cell_totals.bin',dtype=np.int64), [5,4,12])
        j=json.loads((p/'test_stream.json').read_text())
        assert j['raw_total_counts']==21 and j['selected_total_counts']==17 and j['raw_nnz']==5
    return {'duplicate_symbol_summed':True,'excluded_cells_omitted':True,'row_order_and_total_counts':True}


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    (ROOT/'audit').mkdir(exist_ok=True)
    exe=OUT/'signature_stream'
    subprocess.run(['g++','-O3','-std=c++17',str(Path(__file__).with_name('signature_stream.cpp')),'-lz','-o',str(exe)],check=True)
    tests=stream_test(exe)
    features=pd.read_csv(SOURCE/'GSE236581_features.tsv.gz',sep='\t',header=None,dtype=str)
    bars=pd.read_csv(SOURCE/'GSE236581_barcodes.tsv.gz',sep='\t',header=None,dtype=str)[0]
    meta=pd.read_csv(SOURCE/'GSE236581_CRC-ICB_metadata.txt.gz',sep=r'\s+',index_col=0)
    meta.index.name='cell_name'
    t=pd.read_parquet(INVENTORY)
    assert len(meta)==len(bars)==975275 and len(features)==36027
    assert meta.index.is_unique and bars.is_unique and t.cell_name.is_unique
    np.testing.assert_array_equal(meta.index.to_numpy(), bars.to_numpy())
    assert set(meta.index[meta.MajorCellType.eq('T')])==set(t.cell_name)
    fields=['Ident','patient_id','is_crc','is_baseline','Tissue','treatment_stage','treatment_point','Cancer Type']
    mapping=t[fields].drop_duplicates()
    assert mapping.Ident.is_unique
    for col in ['Patient','Tissue','SubCellType','nCount_RNA','MajorCellType']:
        np.testing.assert_array_equal(meta.loc[t.cell_name,col].to_numpy(),t[col].to_numpy())
    joined=meta.reset_index().merge(mapping.drop(columns='Tissue'),on='Ident',how='left',validate='many_to_one',sort=False)
    assert joined.is_crc.notna().all() and joined.is_baseline.notna().all()
    np.testing.assert_array_equal(joined.cell_name.to_numpy(),bars.to_numpy())
    selected=joined.is_crc & joined.is_baseline & joined.Tissue.eq('Tumor')
    assert not joined.loc[selected,'patient_id'].isin([private_value('signature_prepare_value_71_55'),private_value('signature_prepare_value_71_61')]).any()
    x=joined.loc[selected].copy()
    x['pseudobulk_id']=x.patient_id+'__'+x.SubCellType
    groups=x.groupby(['pseudobulk_id','patient_id','MajorCellType','SubCellType'],sort=True).agg(n_cells=('cell_name','size'),metadata_total_counts=('nCount_RNA','sum'),n_specimens=('Ident','nunique')).reset_index()
    groups['group_index']=np.arange(len(groups),dtype=np.int32)
    idx=pd.Index(groups.pseudobulk_id).get_indexer(x.pseudobulk_id)
    cell_group=np.full(len(joined),-1,dtype=np.int32)
    cell_group[np.flatnonzero(selected)]=idx
    symbols=features[1].str.strip()
    valid=symbols.notna() & symbols.ne('')
    assert valid.all(), 'Unmapped raw genes require explicit count-universe handling'
    genes=pd.Index(sorted(symbols.unique()),name='gene')
    gene_map=genes.get_indexer(symbols).astype(np.int32)
    gene_meta=pd.DataFrame({'raw_row_0based':np.arange(len(features)), 'source_gene_id':features[0],
                            'gene':symbols,'collapsed_gene_index':gene_map})
    gene_meta.to_csv(OUT/'gene_map.tsv',sep='\t',index=False)
    groups.to_csv(OUT/'pseudobulk_metadata.tsv',sep='\t',index=False)
    # No endpoint/response columns are imported or saved into this lane.
    x[['cell_name','Ident','patient_id','MajorCellType','SubCellType','pseudobulk_id','nCount_RNA']].to_parquet(OUT/'training_cell_inventory.parquet',index=False)
    gene_map.tofile(OUT/'gene_map.bin');cell_group.tofile(OUT/'cell_group.bin')
    eligibility=groups.loc[groups.MajorCellType.eq('T')].pivot_table(index='patient_id',columns='SubCellType',values='n_cells',fill_value=0)
    c26=eligibility.get('c26_CD8_MAIT_KLRB1',pd.Series(0,index=eligibility.index))
    c27=eligibility.get('c27_CD8_MAIT_SLC4A10',pd.Series(0,index=eligibility.index))
    gate=pd.DataFrame({'patient_id':eligibility.index,'c26_cells':c26.to_numpy(),'c27_cells':c27.to_numpy(),
                       'other_T_cells':(eligibility.sum(axis=1)-c26-c27).to_numpy()})
    gate['MAIT_cells']=gate.c26_cells+gate.c27_cells
    gate['MAIT_identity_eligible']=(gate.MAIT_cells>=10)&(gate.other_T_cells>=10)
    gate['c27_state_eligible']=(gate.c26_cells>=10)&(gate.c27_cells>=10)
    gate.to_csv(OUT/'donor_eligibility.tsv',sep='\t',index=False)
    print('PREPARED',len(genes),'genes',len(groups),'pseudobulks',len(x),'baseline tumor allRNA cells',flush=True)
    print(gate.to_string(index=False),flush=True)
    prefix=OUT/'raw'
    subprocess.run([str(exe),str(SOURCE/'GSE236581_counts.mtx.gz'),str(OUT/'gene_map.bin'),str(OUT/'cell_group.bin'),str(len(genes)),str(len(groups)),str(prefix)],check=True)
    counts=np.fromfile(OUT/'raw_counts.bin',dtype=np.int64).reshape(len(groups),len(genes)).T
    totals=np.fromfile(OUT/'raw_cell_totals.bin',dtype=np.int64)
    np.testing.assert_array_equal(totals,joined.nCount_RNA.to_numpy(dtype=np.int64))
    np.testing.assert_array_equal(counts.sum(axis=0),groups.metadata_total_counts.to_numpy(dtype=np.int64))
    assert counts.sum()==totals[selected].sum()
    stream=json.loads((OUT/'raw_stream.json').read_text())
    assert stream['raw_rows']==36027 and stream['raw_cells']==975275 and stream['raw_nnz']==1310816895
    np.savez_compressed(OUT/'pseudobulk_counts.npz',counts=counts,genes=genes.to_numpy(dtype=str),pseudobulk_ids=groups.pseudobulk_id.to_numpy(dtype=str))
    pd.DataFrame(counts,index=genes,columns=groups.pseudobulk_id).to_csv(OUT/'pseudobulk_counts.tsv.gz',sep='\t')
    receipt={'completed_utc':datetime.now(timezone.utc).isoformat(),'stream':stream,
             'training_tumor_cells':len(x),'allRNA_T_reference_cells':len(t),'unique_symbols':len(genes),
             'duplicate_symbol_rows_collapsed':len(features)-len(genes),'pseudobulks':len(groups),
             'eligible_donors':{'MAIT_identity':int(gate.MAIT_identity_eligible.sum()),'c27_state':int(gate.c27_state_eligible.sum())},
             'checks':dict(tests,all975275_metadata_total_counts_exact=True,all_pseudobulk_totals_exact=True,
                           full_RNA_T_cellset_exact=True,cell_barcode_order_exact=True,no_endpoint_columns_read=True,
                           selected_counts_h5ad_not_used=True),
             'sources':[{ 'path':str(p),'size':p.stat().st_size,'sha256':sha(p)} for p in [SOURCE/'GSE236581_counts.mtx.gz',SOURCE/'GSE236581_features.tsv.gz',SOURCE/'GSE236581_barcodes.tsv.gz',SOURCE/'GSE236581_CRC-ICB_metadata.txt.gz',INVENTORY]],
             'counts_sha256':sha(OUT/'pseudobulk_counts.npz'),'protocol_sha256':sha(ROOT/'PROTOCOL.md')}
    write_json(ROOT/'audit/signature_aggregation.json',receipt)
    print('AGGREGATION_COMPLETE',json.dumps(receipt['checks']),flush=True)


if __name__=='__main__': main()
