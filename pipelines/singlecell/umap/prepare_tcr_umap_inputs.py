"""Prepare only TCR input/evidence tables for two UMAPs; no embedding, inference, or report.
Source was executed in a shared Python kernel. Use run.py with the required
private paths; this script requires the project working directory.

Output is deliberately library-resolved. No cohort or across-library abundance
aggregation is chosen. All bulk libraries are independently sourced from scRNA.
AA = exact TRB CDR3 amino-acid string, not full receptor clonal identity.
AA_VJ_topcall = AA + cached first V/J gene call (alleles stripped).
AA_VJ_unique_top = AA + V/J, only uniquely highest-scoring gene calls.
NT_VJ_unique_top = nucleotide CDR3 + uniquely highest-scoring V/J genes.
Absence from the positive-only match table means no detected match under that
rule, only for the completely checked libraries in bulk_library_denominators.tsv.
The denominator is all productive/canonical/in-frame/positive-count TRB reads,
not matched reads, sequencing-wide reads, UMIs, scTCR frequency, or cells.
"""
from release_config import release_path, private_value
from pathlib import Path as tcrv_Path
import pandas as tcrv_pd
import numpy as tcrv_np
import hashlib as tcrv_hashlib
import json as tcrv_json
import re as tcrv_re
import time as tcrv_time
import os as tcrv_os

tcrv_root = tcrv_Path(str(release_path('prepare_tcr_umap_inputs_source_25')))
tcrv_out = tcrv_root/'outputs/turn_20261008041153_4d884f4d63dd4cc7a128a681f1081a0f/tcr_match'
tcrv_scdir = tcrv_root/'20261003_paper_v2/single_cell'
tcrv_atlas = tcrv_root/'0929_caseEmbedding/atlas_340'
tcrv_out.mkdir(parents=True, exist_ok=True)
assert tcrv_Path.cwd() == tcrv_root, 'Do not change the working directory.'

def tcrv_sha(tcrv_path):
    tcrv_h = tcrv_hashlib.sha256()
    with open(tcrv_path, 'rb') as tcrv_f:
        for tcrv_block in iter(lambda:tcrv_f.read(8*1024*1024), b''):
            tcrv_h.update(tcrv_block)
    return tcrv_h.hexdigest()

def tcrv_unique_top(tcrv_text):
    """Return gene only when the maximum score has exactly one gene (not allele)."""
    if not isinstance(tcrv_text, str) or not tcrv_text:
        return ''
    tcrv_hits = []
    for tcrv_item in tcrv_text.split(','):
        tcrv_m = tcrv_re.fullmatch(r'([^(),]+)\(([-+0-9.eE]+)\)', tcrv_item.strip())
        if tcrv_m is None:
            return ''
        tcrv_hits.append((tcrv_m.group(1).split('*')[0], float(tcrv_m.group(2))))
    tcrv_best = max(tcrv_score for _,tcrv_score in tcrv_hits)
    tcrv_genes = {tcrv_g for tcrv_g,tcrv_s in tcrv_hits if abs(tcrv_s-tcrv_best)<1e-9}
    return next(iter(tcrv_genes)) if len(tcrv_genes)==1 else ''

tcrv_inputs = tcrv_pd.read_csv(tcrv_scdir/'input_manifest.tsv', sep='\t')
tcrv_outputs = tcrv_pd.read_csv(tcrv_scdir/'output_manifest.tsv', sep='\t')
for tcrv_name in ['GSE236581_CRC-ICB_metadata.txt.gz','GSE236581_VDJ_merge.txt.gz']:
    tcrv_row = tcrv_inputs[tcrv_inputs.path.str.endswith(tcrv_name)].iloc[0]
    assert tcrv_sha(tcrv_row.path)==tcrv_row.sha256, f'Source hash drift: {tcrv_row.path}'
for tcrv_name in ['cell_inventory.parquet','vdj_chain_inventory.parquet']:
    tcrv_rows = tcrv_outputs[tcrv_outputs.relative_path.eq(tcrv_name)]
    assert len(tcrv_rows)==1, f'Missing exact cache provenance: {tcrv_name}'
    assert tcrv_sha(tcrv_scdir/tcrv_name)==tcrv_rows.iloc[0].sha256, f'Cache hash drift: {tcrv_name}'

tcrv_t = tcrv_pd.read_parquet(tcrv_scdir/'cell_inventory.parquet')
tcrv_ch = tcrv_pd.read_parquet(tcrv_scdir/'vdj_chain_inventory.parquet')
tcrv_meta = tcrv_pd.read_csv(str(release_path('prepare_tcr_umap_inputs_source_65')), sep=r'\s+', index_col=0)
tcrv_vnames = tcrv_pd.read_csv(str(release_path('prepare_tcr_umap_inputs_source_66')), sep=r'\s+', usecols=['cell_name','barcode'])
assert tcrv_t.cell_name.is_unique and tcrv_meta.index.is_unique and tcrv_vnames.cell_name.is_unique
assert set(tcrv_t.cell_name)==set(tcrv_meta.index[tcrv_meta.MajorCellType.eq('T')])
for tcrv_col in ['Ident','Patient','Treatment','Tissue','MajorCellType','SubCellType']:
    assert (tcrv_t[tcrv_col].to_numpy()==tcrv_meta.loc[tcrv_t.cell_name,tcrv_col].to_numpy()).all()
assert (tcrv_t.cell_name.str.rsplit('_',n=1).str[0]==tcrv_t.Ident).all()
assert (tcrv_vnames.barcode.str.replace(r'-1$','',regex=True)==tcrv_vnames.cell_name.str.rsplit('_',n=1).str[1]).all()
assert set(tcrv_ch.cell_name)==set(tcrv_vnames.cell_name)
tcrv_acc = tcrv_ch.loc[tcrv_ch.accepted_proxy & tcrv_ch.cell_name.isin(tcrv_t.cell_name)].copy()
tcrv_acc = tcrv_acc.drop_duplicates(['cell_name','chain','nt','v','j'])
tcrv_chcounts = tcrv_acc.groupby(['cell_name','chain']).size().unstack(fill_value=0)
for tcrv_kind,tcrv_col in [('Alpha','accepted_distinct_alpha'),('Beta','accepted_distinct_beta')]:
    assert (tcrv_chcounts[tcrv_kind].reindex(tcrv_t.cell_name,fill_value=0).to_numpy()==tcrv_t[tcrv_col].fillna(0).to_numpy()).all()
assert set(tcrv_chcounts.index)==set(tcrv_t.loc[tcrv_t.raw_beta_slots.notna(),'cell_name'])

tcrv_sc_columns = ['cell_name','Ident','Patient','clinical_sample_id','Treatment','Tissue','MajorCellType','SubCellType','Cancer Type','treatment_stage','treatment_point','is_crc','is_baseline','raw_beta_slots','raw_alpha_slots','accepted_distinct_beta','accepted_distinct_alpha','single_beta','strict_paired']
tcrv_sc_evidence = tcrv_t.loc[tcrv_t.cell_name.isin(tcrv_chcounts.index),tcrv_sc_columns].copy()
tcrv_sc_evidence = tcrv_sc_evidence.merge(tcrv_vnames,on='cell_name',validate='one_to_one')
tcrv_sc_evidence['barcode_sequence'] = tcrv_sc_evidence.cell_name.str.rsplit('_',n=1).str[1]
tcrv_sc_evidence['has_accepted_beta'] = tcrv_sc_evidence.accepted_distinct_beta.gt(0)
tcrv_sc_evidence['has_accepted_alpha'] = tcrv_sc_evidence.accepted_distinct_alpha.gt(0)
assert tcrv_sc_evidence.MajorCellType.eq('T').all()
tcrv_sc_evidence.to_parquet(tcrv_out/'sc_rna_tcr_cell_evidence.parquet', index=False, compression='zstd')

tcrv_beta = tcrv_acc.loc[tcrv_acc.chain.eq('Beta'),['cell_name','slot','aa','nt','v','j','n_umi','n_read']].copy()
tcrv_beta['aa_key'] = tcrv_beta.aa

tcrv_beta['aa_vj_key'] = tcrv_beta.aa+'|'+tcrv_beta.v+'|'+tcrv_beta.j
tcrv_beta['nt_vj_key'] = tcrv_beta.nt+'|'+tcrv_beta.v+'|'+tcrv_beta.j
tcrv_beta = tcrv_beta.merge(tcrv_sc_evidence[['cell_name','Ident','Patient','barcode','barcode_sequence','SubCellType','accepted_distinct_beta','single_beta','strict_paired']],on='cell_name',validate='many_to_one')
assert not tcrv_beta.duplicated(['cell_name','nt_vj_key']).any()
tcrv_beta.to_parquet(tcrv_out/'sc_beta_chain_keys.parquet', index=False, compression='zstd')
tcrv_query_clones = tcrv_beta[['aa','nt','v','j','aa_key','aa_vj_key','nt_vj_key']].drop_duplicates()
assert tcrv_query_clones.nt_vj_key.is_unique
tcrv_query_clones.to_parquet(tcrv_out/'sc_beta_query_clones.parquet', index=False, compression='zstd')
tcrv_aa_queries = set(tcrv_beta.aa_key)
tcrv_aavj_queries = set(tcrv_beta.aa_vj_key)
tcrv_ntvj_queries = set(tcrv_beta.nt_vj_key)

tcrv_lib = tcrv_pd.read_csv(tcrv_root/'20261003_paper_v2/cohort/library_master.tsv',sep='\t',dtype=str)
assert len(tcrv_lib)==340 and not tcrv_lib.duplicated(['cohort','sample_id']).any()
assert len(tcrv_lib[tcrv_lib.primary_repertoire_library.eq('1')])==339
assert tcrv_lib.loc[tcrv_lib.sample_id.eq(private_value('prepare_tcr_umap_inputs_value_108_42')),'primary_repertoire_library'].item()=='1'
assert tcrv_lib.loc[tcrv_lib.sample_id.eq(private_value('prepare_tcr_umap_inputs_value_109_42')),'primary_repertoire_library'].item()=='0'
tcrv_match_parts = []
tcrv_den_rows = []
tcrv_started = tcrv_time.monotonic()
for tcrv_i,tcrv_row in enumerate(tcrv_lib.to_dict('records'),1):
    tcrv_stem = tcrv_row['cohort']+'__'+tcrv_row['sample_id']
    tcrv_rp = tcrv_atlas/'audit/sample_receipts'/f'{tcrv_stem}.json'
    tcrv_rec = tcrv_json.loads(tcrv_rp.read_text())
    tcrv_ap = tcrv_atlas/'data/patient_aa'/f'{tcrv_stem}.parquet'
    tcrv_dp = tcrv_atlas/'data/clone_detail'/f'{tcrv_stem}.parquet'
    assert tcrv_rec['status']=='PASS'
    for tcrv_field in ['cohort','sample_id','patient_key']:
        assert tcrv_rec[tcrv_field]==tcrv_row[tcrv_field], (tcrv_stem,tcrv_field)
    for tcrv_kind,tcrv_field in [('tsv','clone_tsv_path'),('clns','clns_path')]:
        tcrv_p=tcrv_Path(tcrv_row[tcrv_field]); tcrv_stat=tcrv_p.stat()
        assert {'bytes':tcrv_stat.st_size,'mtime_ns':tcrv_stat.st_mtime_ns}==tcrv_rec['source_'+tcrv_kind+'_stat'], f'Raw source stat drift: {tcrv_p}'
    assert tcrv_sha(tcrv_ap)==tcrv_rec['patient_aa_sha256'], f'AA cache hash drift: {tcrv_stem}'
    assert tcrv_sha(tcrv_dp)==tcrv_rec['clone_detail_sha256'], f'Clone cache hash drift: {tcrv_stem}'
    tcrv_aa=tcrv_pd.read_parquet(tcrv_ap,columns=['aa','read_count'])
    tcrv_den=int(tcrv_aa.read_count.sum())
    assert tcrv_den==tcrv_rec['qc_read_count']==int(tcrv_row['library_qc_reads'])
    assert len(tcrv_aa)==tcrv_rec['unique_aa'] and tcrv_aa.aa.is_unique and tcrv_aa.read_count.gt(0).all()
    tcrv_aa_hit=tcrv_aa.loc[tcrv_aa.aa.isin(tcrv_aa_queries)].copy()
    tcrv_detail=tcrv_pd.read_parquet(tcrv_dp,columns=['aa','nt_cdr3','read_count','qc_pass','v_gene','j_gene','all_v_hits_with_score','all_j_hits_with_score'])
    assert int(tcrv_detail.loc[tcrv_detail.qc_pass,'read_count'].sum())==tcrv_den
    tcrv_hit=tcrv_detail.loc[tcrv_detail.qc_pass & tcrv_detail.aa.isin(tcrv_aa_queries)].copy()
    tcrv_recomputed=tcrv_hit.groupby('aa').read_count.sum().sort_index()
    tcrv_expected=tcrv_aa_hit.set_index('aa').read_count.sort_index()
    assert tcrv_recomputed.equals(tcrv_expected), f'AA sum mismatch: {tcrv_stem}'
    tcrv_hit['aa_vj_key']=tcrv_hit.aa+'|'+tcrv_hit.v_gene+'|'+tcrv_hit.j_gene
    tcrv_hit['nt_vj_key']=tcrv_hit.nt_cdr3+'|'+tcrv_hit.v_gene+'|'+tcrv_hit.j_gene
    tcrv_hit['unique_v']=tcrv_hit.all_v_hits_with_score.map(tcrv_unique_top)
    tcrv_hit['unique_j']=tcrv_hit.all_j_hits_with_score.map(tcrv_unique_top)
    tcrv_unamb=(tcrv_hit.unique_v.eq(tcrv_hit.v_gene)&tcrv_hit.unique_v.str.startswith('TRBV')&tcrv_hit.unique_j.eq(tcrv_hit.j_gene)&tcrv_hit.unique_j.str.startswith('TRBJ'))
    tcrv_definition_frames = [
        ('AA',tcrv_aa_hit.rename(columns={'aa':'match_key'})),
        ('AA_VJ_topcall',tcrv_hit.loc[tcrv_hit.aa_vj_key.isin(tcrv_aavj_queries)].groupby('aa_vj_key',as_index=False).read_count.sum().rename(columns={'aa_vj_key':'match_key'})),
        ('AA_VJ_unique_top',tcrv_hit.loc[tcrv_unamb & tcrv_hit.aa_vj_key.isin(tcrv_aavj_queries)].groupby('aa_vj_key',as_index=False).read_count.sum().rename(columns={'aa_vj_key':'match_key'})),
        ('NT_VJ_unique_top',tcrv_hit.loc[tcrv_unamb & tcrv_hit.nt_vj_key.isin(tcrv_ntvj_queries)].groupby('nt_vj_key',as_index=False).read_count.sum().rename(columns={'nt_vj_key':'match_key'}))]
    for tcrv_def,tcrv_match in tcrv_definition_frames:
        if tcrv_match.empty:
            continue
        tcrv_match=tcrv_match[['match_key','read_count']].rename(columns={'read_count':'bulk_read_count'}).copy()
        for tcrv_field in ['cohort','sample_id','patient_key']:
            tcrv_match[tcrv_field]=tcrv_row[tcrv_field]
        tcrv_match['primary_repertoire_library']=int(tcrv_row['primary_repertoire_library'])
        tcrv_match['match_definition']=tcrv_def
        tcrv_match['bulk_productive_read_total']=tcrv_den
        tcrv_match['bulk_frequency']=tcrv_match.bulk_read_count/tcrv_den
        tcrv_match['bulk_cpm']=1e6*tcrv_match.bulk_frequency
        tcrv_match['bulk_count_ge2']=tcrv_match.bulk_read_count.ge(2)
        tcrv_match_parts.append(tcrv_match)
    tcrv_den_rows.append(dict(cohort=tcrv_row['cohort'],sample_id=tcrv_row['sample_id'],patient_key=tcrv_row['patient_key'],primary_repertoire_library=int(tcrv_row['primary_repertoire_library']),bulk_productive_read_total=tcrv_den,bulk_unique_aa=len(tcrv_aa),count_unit=tcrv_rec['count_unit'],filter_rule=tcrv_rec['filter_rule'],source_clone_tsv=tcrv_row['clone_tsv_path'],source_clns=tcrv_row['clns_path'],patient_aa_path=str(tcrv_ap),patient_aa_resolved=str(tcrv_ap.resolve()),clone_detail_path=str(tcrv_dp),clone_detail_resolved=str(tcrv_dp.resolve()),receipt_path=str(tcrv_rp),patient_aa_sha256=tcrv_rec['patient_aa_sha256'],clone_detail_sha256=tcrv_rec['clone_detail_sha256'],cache_hashes_verified=True,raw_source_stat_verified=True,matched_aa_keys=len(tcrv_aa_hit),matched_AA_rows_without_unique_top_vj=int((~tcrv_unamb).sum())))
    if tcrv_i%25==0 or tcrv_i==len(tcrv_lib):
        print(f'Validated/matched {tcrv_i}/{len(tcrv_lib)} libraries; elapsed={tcrv_time.monotonic()-tcrv_started:.1f}s',flush=True)

tcrv_long=tcrv_pd.concat(tcrv_match_parts,ignore_index=True)
tcrv_long=tcrv_long[['cohort','sample_id','patient_key','primary_repertoire_library','match_definition','match_key','bulk_read_count','bulk_productive_read_total','bulk_frequency','bulk_cpm','bulk_count_ge2']]
assert not tcrv_long.duplicated(['cohort','sample_id','match_definition','match_key']).any()
assert tcrv_long.bulk_read_count.gt(0).all() and tcrv_long.bulk_frequency.between(0,1).all()
tcrv_long.to_parquet(tcrv_out/'bulk_exact_matches_by_library.parquet',index=False,compression='zstd')
tcrv_denominators=tcrv_pd.DataFrame(tcrv_den_rows)
tcrv_denominators.to_csv(tcrv_out/'bulk_library_denominators.tsv',sep='\t',index=False)
# Detection summaries only: do not sum/mean/max CPM across libraries or cohorts.
tcrv_detection=tcrv_long.assign(tcrv_primary_ge2=tcrv_long.primary_repertoire_library.eq(1)&tcrv_long.bulk_count_ge2).groupby(['cohort','match_definition','match_key'],sort=False).agg(n_libraries_detected=('sample_id','nunique'),n_primary_libraries_detected=('primary_repertoire_library','sum'),n_primary_libraries_count_ge2=('tcrv_primary_ge2','sum')).reset_index()
tcrv_detection=tcrv_detection.merge(tcrv_denominators.groupby('cohort').agg(total_libraries=('sample_id','nunique'),total_primary_libraries=('primary_repertoire_library','sum')).reset_index(),on='cohort',validate='many_to_one')
tcrv_detection.to_parquet(tcrv_out/'bulk_exact_detection_by_key_cohort.parquet',index=False,compression='zstd')
# Reopen outputs and check the sparse-long join contract.
assert len(tcrv_pd.read_parquet(tcrv_out/'sc_rna_tcr_cell_evidence.parquet'))==len(tcrv_sc_evidence)
assert len(tcrv_pd.read_parquet(tcrv_out/'sc_beta_chain_keys.parquet'))==len(tcrv_beta)
assert len(tcrv_pd.read_parquet(tcrv_out/'bulk_exact_matches_by_library.parquet'))==len(tcrv_long)
print('sc cells',len(tcrv_sc_evidence),'beta-bearing cells',tcrv_beta.cell_name.nunique(),'unique beta NTVJ',len(tcrv_query_clones),'matched long rows',len(tcrv_long),flush=True)
print(tcrv_long.groupby(['cohort','match_definition']).agg(library_key_rows=('match_key','size'),matched_keys=('match_key','nunique')).to_string(),flush=True)
print('Across-library abundance aggregation NOT selected; no cell-level bulk gradient column fabricated.',flush=True)
