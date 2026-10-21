#!/usr/bin/env python3
"""Export complete selected-cluster repertoires and exact-AA scTCR assignments.

Cluster labels are ALWAYS indexed by the explicit seq_idx, never parquet row order.
Primary units are the 340805 single-beta cells in the existing RNA embedding.
The historical strict AAVJ match and its bulk abundance are separate evidence.
"""
from __future__ import annotations
from release_config import release_path, private_value

import argparse
import csv
import gzip
import hashlib
import itertools
import json
import shutil
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq


TARGETS = [('tcrbert', 7, 551367), ('tcrbert', 11, 681884), ('sceptr', 5, 546836)]


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as fh:
        for chunk in iter(lambda: fh.read(8 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def dump_json(value, path):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n')


def export_df(frame, stem):
    frame.to_parquet(str(stem) + '.parquet', index=False, compression='zstd')
    frame.to_csv(str(stem) + '.tsv.gz', sep='\t', index=False, compression='gzip')


def run(root, out):
    start = time.time()
    atlas = root / '0929_caseEmbedding/atlas_340'
    old = root / '20261010_article_v25/20261008_UMAP/recomputed_scTCR_20261009'
    source = {
        'aa_union': atlas / 'data/aa_union.parquet',
        'tcrbert_labels': atlas / 'leiden_full_20261001/clusters/primary_labels.npy',
        'sceptr_labels': atlas / 'leiden_full_sceptr_20261002/clusters/primary_labels.npy',
        'sc_beta_chain_keys': root / 'outputs/turn_20261008041153_4d884f4d63dd4cc7a128a681f1081a0f/tcr_match/sc_beta_chain_keys.parquet',
        'sc_embedding': old / 'plot_data/scTCR_Tcells_author_subtypes.csv.gz',
        'historical_strict_union': old / 'plot_data/union_bulk_TCR_nebula.csv.gz',
        'historical_strict_home': old / 'plot_data/home_bulk_TCR_nebula.csv.gz',
        'historical_strict_external': old / 'plot_data/external_bulk_TCR_nebula.csv.gz',
        'all_248_pooled_tests': atlas / 'pooled_N_20261003/stats/all_248_pooled_tests.tsv',
        'selected_prior_association_context': atlas / 'antigen_selected_20261003/results/selected_prior_association_context.tsv',
    }
    lists = out / 'analysis/tcr_lists'
    mapping = out / 'analysis/mapping'
    audit = out / 'audit'
    for p in [lists, mapping, audit]:
        p.mkdir(parents=True, exist_ok=True)
    sc = pd.read_parquet(source['sc_beta_chain_keys'])
    meta = pd.read_csv(source['sc_embedding'], low_memory=False)
    assert len(meta) == 369063 and meta.cell_name.is_unique
    assert not meta[['UMAP1', 'UMAP2']].isna().any().any()
    single = sc.loc[sc.single_beta].copy()
    assert len(single) == 340805 and single.cell_name.is_unique
    assert single.cell_name.isin(meta.cell_name).all()
    # Explicitly validate every shared source field before retaining the RNA metadata.
    shared = [c for c in sc if c in meta and c != 'cell_name']
    compare = single[['cell_name'] + shared].merge(meta[['cell_name'] + shared], on='cell_name', validate='one_to_one', suffixes=('_chain', '_rna'))
    metadata_check = {}
    for c in shared:
        x, y = compare[c + '_chain'], compare[c + '_rna']
        equal = x.eq(y) | (x.isna() & y.isna())
        metadata_check[c] = int((~equal).sum())
        assert equal.all(), (c, int((~equal).sum()))
    single = single.drop(columns=shared).merge(meta, on='cell_name', how='left', validate='one_to_one')
    strict_cols = ['cell_name', 'aa', 'nt', 'v', 'j', 'aa_vj_key', 'nt_vj_key', 'bulk_mean_cpm', 'bulk_supporting_primary_libraries', 'bulk_sum_read_count_qualified', 'bulk_sum_cpm_qualified', 'n_primary_libraries']
    strict = pd.read_csv(source['historical_strict_union'], usecols=strict_cols)
    assert len(strict) == 98215 and strict.cell_name.is_unique
    assert strict.cell_name.isin(single.cell_name).all()
    receptor_cols = ['aa', 'nt', 'v', 'j', 'aa_vj_key', 'nt_vj_key']
    check = strict[['cell_name'] + receptor_cols].merge(single[['cell_name'] + receptor_cols], on='cell_name', validate='one_to_one', suffixes=('_old', '_new'))
    for c in receptor_cols:
        assert check[c + '_old'].eq(check[c + '_new']).all(), c
    single['strict_aavj_ge2'] = single.cell_name.isin(strict.cell_name)
    single = single.merge(strict.drop(columns=receptor_cols), on='cell_name', how='left', validate='one_to_one')
    for cohort in ['home', 'external']:
        co = pd.read_csv(source['historical_strict_' + cohort], usecols=['cell_name', 'bulk_mean_cpm', 'bulk_supporting_primary_libraries'])
        assert co.cell_name.is_unique
        co = co.rename(columns={c: cohort + '_' + c for c in co if c != 'cell_name'})
        single = single.merge(co, on='cell_name', how='left', validate='one_to_one')
    # Missing bulk values mean no qualified historical match, not an observed zero.
    exact_sc_counts = single.aa.value_counts().to_dict()
    strict_sc_counts = single.loc[single.strict_aavj_ge2, 'aa'].value_counts().to_dict()
    aa_queries = pa.array(sc.aa.drop_duplicates(), type=pa.string())
    labels = {m: np.load(source[m + '_labels'], mmap_mode='r') for m in ['tcrbert', 'sceptr']}
    n = pq.ParquetFile(source['aa_union']).metadata.num_rows
    assert n == 28380380
    for m, arr in labels.items():
        assert arr.shape == (n,) and arr.dtype == np.int32, (m, arr.shape, arr.dtype)
    seen = np.zeros(n, dtype=bool)
    target_counts = {f'{m}_{c}': 0 for m, c, _ in TARGETS}
    writers, handles, csv_writers = {}, {}, {}
    output_schema = pa.schema([('model', pa.string()), ('cluster_id', pa.int32()), ('seq_idx', pa.int64()), ('aa', pa.string()), ('exact_sc_match_count', pa.int32()), ('strict_aavj_match_count', pa.int32())])
    for m, c, _ in TARGETS:
        target = f'{m}_{c}'
        writers[target] = pq.ParquetWriter(lists / (target + '_complete_aa.parquet'), output_schema, compression='zstd')
        handles[target] = gzip.open(lists / (target + '_complete_aa.tsv.gz'), 'wt', newline='')
        csv_writers[target] = csv.writer(handles[target], delimiter='\t', lineterminator='\n')
        csv_writers[target].writerow(output_schema.names)
    overlap_bulk = {tuple(x): 0 for k in [1, 2, 3] for x in itertools.combinations(target_counts, k)}
    found = []
    n_read = 0
    try:
        for batch in pq.ParquetFile(source['aa_union']).iter_batches(batch_size=262144, columns=['seq_idx', 'aa']):
            ids = batch.column(0).to_numpy()
            assert ids.min() >= 0 and ids.max() < n
            assert np.unique(ids).size == ids.size and not seen[ids].any()
            seen[ids] = True
            aa = batch.column(1)
            b_labels = {m: arr[ids] for m, arr in labels.items()}
            masks = {f'{m}_{c}': b_labels[m] == c for m, c, _ in TARGETS}
            for comb in overlap_bulk:
                overlap_bulk[comb] += int(np.logical_and.reduce([masks[t] for t in comb]).sum())
            for m, c, _ in TARGETS:
                target = f'{m}_{c}'
                keep = masks[target]
                seqs = pc.filter(aa, pa.array(keep)).to_pylist()
                selected_ids = ids[keep]
                cnts = [exact_sc_counts.get(x, 0) for x in seqs]
                strict_cnts = [strict_sc_counts.get(x, 0) for x in seqs]
                count = len(seqs)
                table = pa.Table.from_arrays([pa.array([m] * count), pa.array(np.full(count, c, dtype=np.int32)), pa.array(selected_ids), pa.array(seqs, type=pa.string()), pa.array(cnts, type=pa.int32()), pa.array(strict_cnts, type=pa.int32())], schema=output_schema)
                writers[target].write_table(table)
                csv_writers[target].writerows(zip(itertools.repeat(m, count), itertools.repeat(c, count), selected_ids, seqs, cnts, strict_cnts))
                target_counts[target] += count
            # Batch-membership computation keeps the 28M union out of Python memory.
            keep = pc.is_in(aa, value_set=aa_queries).to_numpy(zero_copy_only=False)
            if keep.any():
                found.append(pd.DataFrame({'seq_idx': ids[keep], 'aa': pc.filter(aa, pa.array(keep)).to_pylist(), 'tcrbert_cluster': b_labels['tcrbert'][keep], 'sceptr_cluster': b_labels['sceptr'][keep]}))
            n_read += len(ids)
            if n_read % (262144 * 16) == 0:
                print(f'streamed {n_read:,}/{n:,} AA records; elapsed {time.time()-start:.1f}s', flush=True)
    finally:
        for w in writers.values():
            w.close()
        for h in handles.values():
            h.close()
    assert n_read == n and seen.all()
    for m, c, expected in TARGETS:
        assert target_counts[f'{m}_{c}'] == expected
    lookup = pd.concat(found, ignore_index=True)
    assert lookup.aa.is_unique and lookup.seq_idx.is_unique
    export_df(lookup, mapping / 'sc_receptor_aa_model_lookup')
    single = single.merge(lookup, on='aa', how='left', validate='many_to_one')
    for col in ['seq_idx', 'tcrbert_cluster', 'sceptr_cluster']:
        single[col] = single[col].fillna(-1).astype('int64' if col == 'seq_idx' else 'int32')
    single['bulk_union_exact_aa'] = single.seq_idx.ge(0)
    for m, c, _ in TARGETS:
        single[f'{m}_{c}_exact_aa'] = single[m + '_cluster'].eq(c)
    assert single.loc[single.strict_aavj_ge2, 'bulk_union_exact_aa'].all()
    assert single[['Patient', 'Ident', 'nt_vj_key']].notna().all().all()
    # Counts are within the retained single-beta RNA atlas; paired alpha is not required.
    single['sample_beta_clone_size'] = single.groupby(['Ident', 'nt_vj_key']).cell_name.transform('size').astype('int32')
    single['donor_beta_clone_size'] = single.groupby(['Patient', 'nt_vj_key']).cell_name.transform('size').astype('int32')
    single['mapping_definition'] = 'exact CDR3beta AA to bulk union; cluster label indexed by seq_idx'
    export_df(single, mapping / 'sc_singlebeta_model_assignments')
    long_frames = []
    target_summary = []
    for m, c, expected in TARGETS:
        target = f'{m}_{c}'
        cells = single.loc[single[m + '_cluster'].eq(c)].copy()
        cells.insert(0, 'target', target)
        cells.insert(1, 'model', m)
        cells.insert(2, 'cluster_id', c)
        long_frames.append(cells)
        agg = cells.groupby(receptor_cols, dropna=False).agg(n_sc_cells=('cell_name', 'size'), n_donors=('Patient', 'nunique'), n_samples=('Ident', 'nunique'), n_subtypes=('SubCellType', 'nunique'), n_strict_aavj_cells=('strict_aavj_ge2', 'sum'), max_sample_beta_clone_size=('sample_beta_clone_size', 'max'), max_donor_beta_clone_size=('donor_beta_clone_size', 'max'), donors=('Patient', lambda x: '|'.join(sorted(set(x)))), samples=('Ident', lambda x: '|'.join(sorted(set(x)))), subtypes=('SubCellType', lambda x: '|'.join(sorted(set(x))))).reset_index()
        agg.insert(0, 'target', target)
        agg.insert(1, 'model', m)
        agg.insert(2, 'cluster_id', c)
        export_df(agg.sort_values('n_sc_cells', ascending=False), lists / (target + '_sc_matched_receptors'))
        complete = pq.read_table(lists / (target + '_complete_aa.parquet')).to_pandas()
        assert complete.aa.is_unique and complete.seq_idx.is_unique and len(complete) == expected
        assert int(complete.exact_sc_match_count.sum()) == len(cells)
        assert int(complete.strict_aavj_match_count.sum()) == int(cells.strict_aavj_ge2.sum())
        assert np.array_equal(labels[m][complete.seq_idx.to_numpy()], complete.cluster_id.to_numpy())
        target_summary.append({'target': target, 'model': m, 'cluster_id': c, 'complete_bulk_aa': expected, 'matched_unique_aa': int(cells.aa.nunique()), 'matched_nt_vj_receptors': int(cells.nt_vj_key.nunique()), 'matched_cells': len(cells), 'matched_donors': int(cells.Patient.nunique()), 'matched_samples': int(cells.Ident.nunique()), 'strict_aavj_cells': int(cells.strict_aavj_ge2.sum()), 'strict_aavj_unique_aa': int(cells.loc[cells.strict_aavj_ge2, 'aa'].nunique()), 'crc_baseline_cells': int((cells.is_crc & cells.is_baseline).sum())})
    long = pd.concat(long_frames, ignore_index=True)
    export_df(long, mapping / 'target_matched_cells_long')
    summary_df = pd.DataFrame(target_summary)
    summary_df.to_csv(mapping / 'target_match_summary.tsv', sep='\t', index=False)
    overlap_rows = []
    for comb, count in overlap_bulk.items():
        keep = np.logical_and.reduce([single[t + '_exact_aa'].to_numpy() for t in comb])
        d = single.loc[keep]
        overlap_rows.append({'targets': '|'.join(comb), 'n_targets': len(comb), 'bulk_aa_intersection': count, 'sc_cells_intersection': len(d), 'sc_aa_intersection': int(d.aa.nunique()), 'sc_donors_intersection': int(d.Patient.nunique()), 'strict_aavj_cells_intersection': int(d.strict_aavj_ge2.sum())})
    pd.DataFrame(overlap_rows).to_csv(mapping / 'target_overlap_counts.tsv', sep='\t', index=False)
    # Multi-beta sensitivity is deliberately separate: either accepted beta may match.
    multi = sc.loc[~sc.single_beta].merge(lookup, on='aa', how='left', validate='many_to_one')
    multi['in_rna_embedding'] = multi.cell_name.isin(meta.cell_name)
    sensitivity = []
    for scope, md in [('all_chain_source', multi), ('in_rna_embedding', multi.loc[multi.in_rna_embedding])]:
        for m, c, _ in TARGETS:
            hit = md.loc[md[m + '_cluster'].eq(c)]
            sensitivity.append({'scope': scope, 'target': f'{m}_{c}', 'all_multibeta_cells': int(md.cell_name.nunique()), 'matched_beta_chains': len(hit), 'matched_multibeta_cells_any_beta': int(hit.cell_name.nunique()), 'matched_donors': int(hit.Patient.nunique()), 'matched_aa': int(hit.aa.nunique())})
    pd.DataFrame(sensitivity).to_csv(mapping / 'multibeta_sensitivity_counts.tsv', sep='\t', index=False)
    multi_targets = multi.loc[np.logical_or.reduce([multi[m + '_cluster'].eq(c).to_numpy() for m, c, _ in TARGETS])].copy()
    export_df(multi_targets, mapping / 'multibeta_target_chain_matches_sensitivity')
    # Preserve full test universe and exact selected rows without re-running association.
    for key in ['all_248_pooled_tests', 'selected_prior_association_context']:
        shutil.copy2(source[key], audit / ('mapping_' + key + '.tsv'))
    pooled = pd.read_csv(source['all_248_pooled_tests'], sep='\t')
    tmask = np.logical_or.reduce([(pooled.model.eq(m) & pooled.cluster_id.eq(c)).to_numpy() for m, c, _ in TARGETS])
    pooled.loc[tmask].to_csv(audit / 'mapping_selected_clusters_all_designs_scales.tsv', sep='\t', index=False)
    selected = pooled.loc[tmask & pooled.design.eq('primary_pooled_cohort_technical_adjusted') & pooled.scale.eq('read_fraction_log_per_sd')].copy()
    assert len(selected) == 3 and selected.p.lt(.05).all() and selected.q_BH_all_248.gt(.4).all()
    selected.to_csv(audit / 'mapping_selected_primary_read_fraction_associations.tsv', sep='\t', index=False)
    sources = []
    for key, path in source.items():
        sources.append({'key': key, 'path': str(path), 'bytes': path.stat().st_size, 'sha256': sha256(path)})
    pd.DataFrame(sources).to_csv(audit / 'mapping_source_manifest.tsv', sep='\t', index=False)
    summary = {
        'status': 'PASS', 'sc_chain_rows': len(sc), 'sc_singlebeta_cells': len(single), 'sc_multibeta_cells': int(sc.loc[~sc.single_beta, 'cell_name'].nunique()),
        'sc_embedding_cells': len(meta), 'sc_singlebeta_embedding_absent': 0, 'strict_aavj_ge2_cells': int(single.strict_aavj_ge2.sum()),
        'singlebeta_bulk_union_exact_aa_cells': int(single.bulk_union_exact_aa.sum()), 'singlebeta_not_in_bulk_union_cells': int((~single.bulk_union_exact_aa).sum()),
        'singlebeta_bulk_union_unique_aa': int(single.loc[single.bulk_union_exact_aa, 'aa'].nunique()),
        'bulk_union_rows': n, 'seq_idx_unique_complete': bool(seen.all()), 'label_dtype': 'int32', 'metadata_mismatch_counts': metadata_check,
        'target_summary': target_summary, 'overlap': overlap_rows, 'elapsed_seconds': round(time.time()-start, 2),
        'definitions': {
            'cluster_selection': 'Historically selected 3 clusters: primary pooled cohort/technical adjusted read fraction per SD; nominal P<.05, all-248 BH q>.4. This script does not reselect clusters or rerun their clinical associations.',
            'primary': 'One accepted beta per cell, RNA atlas present; no requirement for paired alpha; all 340805 single-beta cells kept as denominator.',
            'membership': 'Exact CDR3beta amino-acid equality with aa_union, then model label at explicit seq_idx. Does not require V/J equality; it does not establish same paired-alpha-beta clone or antigen specificity.',
            'strict_aavj_ge2': 'Cell appears in historical union nebula: exact beta AA plus V and J; bulk key >=2 reads per library; original source independently checked against beta receptor fields.',
            'bulk_cpm_missing': 'Missing when no historical qualified match; not imputed to zero. CPM means per-primary-library historical bulk abundance, not sc cell clone size.',
            'sample_beta_clone_size': 'Number of retained single-beta RNA atlas cells sharing Ident and nucleotide CDR3beta|V|J.',
            'donor_beta_clone_size': 'Number of retained single-beta RNA atlas cells sharing Patient and nucleotide CDR3beta|V|J, pooling sampled tissues/timepoints; not a paired-alpha-beta clonotype.',
            'overlap': 'TCR-BERT clusters 7 and 11 are disjoint within that partition; SCEPTR cluster 5 can overlap either. Three targets are not independent repertoires.',
            'complete_aa_sc_counts': 'Counts annotate full complete cluster lists; no scRNA matching or strict-AAVJ filtering was used to filter those lists.',
            'multibeta': 'Sensitivity only; a multi-beta cell is a match if any accepted beta matches; excluded from primary mapping.'
        }
    }
    dump_json(summary, audit / 'mapping_summary.json')
    print(summary_df.to_string(index=False), flush=True)
    print(json.dumps({k: summary[k] for k in ['status', 'sc_singlebeta_cells', 'strict_aavj_ge2_cells', 'singlebeta_bulk_union_exact_aa_cells', 'singlebeta_not_in_bulk_union_cells', 'elapsed_seconds']}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, default=Path(str(release_path('build_cluster_mapping_source_257'))))
    parser.add_argument('--out', type=Path, default=Path(str(release_path('build_cluster_mapping_source_258'))))
    args = parser.parse_args()
    run(args.root, args.out)
