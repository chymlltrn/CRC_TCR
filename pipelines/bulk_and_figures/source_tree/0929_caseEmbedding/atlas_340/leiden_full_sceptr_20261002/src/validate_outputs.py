#!/usr/bin/env python3
"""Separate receipt/hash/array acceptance; never reads outcomes or phenotype tables."""
from release_runtime import private_literal as _private_literal, release_path as _release_path
import ast
import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import resource
import sys

ROOT = Path(_release_path('path_fd724b65e07400d1'))


def need(value, message):
    if not value:
        raise ValueError(message)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(16 << 20), b''):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def dump(path, obj):
    path = Path(path)
    tmp = path.with_name(path.name + '.tmp')
    tmp.write_text(json.dumps(obj, indent=2, sort_keys=True, allow_nan=False) + '\n')
    os.replace(tmp, path)


def now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def passing(rel, source=None):
    obj = read(ROOT / rel)
    need(obj.get('status') == 'PASS', 'Receipt is not PASS: ' + rel)
    if source is not None:
        need(obj.get('source_code_sha256') == source, 'Receipt source mismatch: ' + rel)
    return obj


def contract_gate():
    contract = read(ROOT / 'config/comparison_contract.json')
    cfg = read(ROOT / 'config/graph_design.json')
    for key, value in contract['fixed_graph_parameters'].items():
        need(cfg.get(key) == value, 'Comparison parameter changed: ' + key)
    for name, expected in contract['execution_source_sha256'].items():
        need(sha(ROOT / 'src' / name) == expected, 'Frozen code changed: ' + name)
    original = (ROOT / 'audit/reference_full_leiden_graph.py.txt').read_text()
    derived = (ROOT / 'src/full_leiden_graph.py').read_text()
    before = 'ROOT = BASE / "leiden_full_20261001"'
    after = 'ROOT = BASE / "' + ROOT.name + '"'
    need(original.count(before) == 1 and derived == original.replace(before, after, 1),
         'Graph code must differ from audited reference only in output ROOT')
    need(sha(ROOT / 'audit/reference_full_leiden_graph.py.txt') == contract['reference_graph_source_sha256'],
         'Reference source snapshot changed')
    need(sha(ROOT / 'audit/reference_graph_design.json') == contract['reference_design_sha256'],
         'Reference design snapshot changed')
    for name in contract['execution_source_sha256']:
        ast.parse((ROOT / 'src' / name).read_text())
    need(cfg['primary_model'] == 'sceptr_cdr3_only', 'Wrong representation')
    need(cfg['n_vertices'] == 28380380 and cfg['dimension'] == 64, 'Wrong complete bank shape')
    return cfg, contract


def source_check(np, pq, receipt_name='source_preflight.json'):
    cfg, contract = contract_gate()
    projection = read(cfg['projection_receipt_path'])
    fit = read(Path(cfg['projection_receipt_path']).with_name('pca_fit_receipt.json'))
    need(projection['status'] == fit['status'] == 'complete', 'Projection/PCA incomplete')
    need(projection['model'] == fit['model'] == cfg['primary_model'], 'Model receipt mismatch')
    need(projection['n_aa'] == cfg['n_vertices'] and projection['missing'] == projection['nonfinite'] == 0,
         'Embedding coverage gate failed')
    need(projection['scores_sha256'] == cfg['pca_scores_sha256'], 'Embedding receipt hash mismatch')
    need(projection['input_index_sha256'] == cfg['atlas_index_sha256'] == fit['input_index_sha256'],
         'AA-index identity mismatch')
    need(fit['label_blind'] and fit['scope'] == 'joint' and fit['patients'] == 339,
         'PCA scope mismatch')
    hashes = {}
    for key in ['pca_scores', 'atlas_index', 'atlas_union']:
        actual = sha(cfg[key + '_path'])
        need(actual == cfg[key + '_sha256'], 'Actual input hash mismatch: ' + key)
        hashes[key] = actual
    need(sha(cfg['balanced_draws_path']) == fit['draw_sha256'] == contract['balanced_draws_sha256'],
         'Balanced PCA sampling changed')
    scores = np.load(cfg['pca_scores_path'], mmap_mode='r', allow_pickle=False)
    need(scores.shape == (cfg['n_vertices'], cfg['dimension']) and scores.dtype == np.float16,
         'SCEPTR bank shape/dtype mismatch')
    for start in range(0, len(scores), 100000):
        need(np.isfinite(scores[start:start + 100000]).all(), 'Nonfinite SCEPTR score')
    offset = 0
    for batch in pq.ParquetFile(cfg['atlas_index_path']).iter_batches(batch_size=100000, columns=['seq_idx']):
        actual = batch.column(0).to_numpy()
        need(np.array_equal(actual, np.arange(offset, offset + len(actual))), 'AA seq_idx row order mismatch')
        offset += len(actual)
    need(offset == len(scores), 'AA-index row count mismatch')
    need(pq.ParquetFile(cfg['atlas_union_path']).metadata.num_rows == len(scores), 'AA-union count mismatch')
    for rel, expected in contract['shared_sampling_files_sha256'].items():
        need(sha(ROOT / rel) == expected, 'Shared comparison sampling changed: ' + rel)
    for rel, expected in read(ROOT / 'env/vendor_sha256.json').items():
        need(sha(ROOT / 'env/vendor' / rel) == expected, 'Copied dependency changed: ' + rel)
    result = dict(status='PASS', time=now(), model=cfg['primary_model'], n_vertices=len(scores),
                  shape=list(scores.shape), dtype=str(scores.dtype), missing=0, nonfinite=0,
                  input_sha256=hashes, same_seq_idx_and_balanced_draws_as_TCRBERT=True,
                  own_index_and_graph_required=True, phenotype_tables_read=[], gpu_used=False,
                  source_code_sha256=sha(__file__), graph_source_sha256=sha(ROOT / 'src/full_leiden_graph.py'))
    dump(ROOT / 'audit' / receipt_name, result)
    print(json.dumps(result), flush=True)


def neighbor_check(np, ids, distances, self_ids, n, k):
    need(ids.shape == distances.shape == (len(self_ids), k), 'Neighbor shape mismatch')
    need(np.issubdtype(ids.dtype, np.integer), 'Noninteger neighbor IDs')
    need(np.all((ids >= 0) & (ids < n)), 'Out-of-range neighbor')
    need(not np.any(ids == self_ids[:, None]), 'Self neighbor')
    order = np.sort(ids, axis=1)
    need(not np.any(order[:, 1:] == order[:, :-1]), 'Duplicate neighbor')
    need(np.isfinite(distances).all() and np.all(distances >= 0), 'Invalid distances')
    need(np.all(np.diff(distances, axis=1) >= -1e-6), 'Distances not sorted')


def recall_check(np, pq, cfg, graph_source):
    r = passing('audit/fullbank_ann_recall.json', graph_source)
    exact = passing('pilot/exact_fullbank_receipt.json', graph_source)
    query_path = ROOT / 'pilot/global_recall_queries.parquet'
    query = pq.read_table(query_path)
    qi = query['seq_idx'].to_numpy()
    need(len(qi) == cfg['recall_queries'] == 512, 'Wrong full-bank recall query count')
    need(r['exact_database_rows'] == exact['groundtruth_n_vertices'] == cfg['n_vertices'],
         'Recall reference is not the entire 28,380,380-vector bank')
    need(r['query_sha256'] == exact['query_sha256'] == sha(query_path), 'Recall query hash mismatch')
    need(r['source_sha256'] == exact['source_sha256'] == cfg['pca_scores_sha256'], 'Recall bank mismatch')
    for kind in ['indices', 'distances']:
        need(sha(r['exact_' + kind + '_path']) == r['exact_' + kind + '_sha256'] == exact['exact_' + kind + '_sha256'],
             'Exact-truth array hash mismatch')
    true_i = np.load(r['exact_indices_path'], allow_pickle=False)
    true_d = np.load(r['exact_distances_path'], allow_pickle=False)
    neighbor_check(np, true_i, true_d, qi, cfg['n_vertices'], cfg['k'])
    metrics = []
    for row in r['results']:
        for kind in ['indices', 'distances']:
            need(sha(row['approx_' + kind + '_path']) == row['approx_' + kind + '_sha256'], 'ANN audit hash mismatch')
        ai = np.load(row['approx_indices_path'], allow_pickle=False)
        ad = np.load(row['approx_distances_path'], allow_pickle=False)
        neighbor_check(np, ai, ad, qi, cfg['n_vertices'], cfg['k'])
        per = np.array([len(set(a).intersection(b)) / cfg['k'] for a, b in zip(true_i.tolist(), ai.tolist())])
        need(math.isclose(float(per.mean()), row['strict_id_recall_mean'], abs_tol=1e-12), 'Recall recomputation mismatch')
        strata = []
        ls = query['length_stratum'].to_numpy()
        bs = np.array(query['bank_source'].to_pylist())
        for length, source in sorted(set(zip(ls.tolist(), bs.tolist()))):
            mask = (ls == length) & (bs == source)
            strata.append(dict(length_stratum=int(length), bank_source=source, queries=int(mask.sum()),
                               strict_recall=float(per[mask].mean())))
        metrics.append(dict(nprobe=row['nprobe'], strict_recall_mean=float(per.mean()),
                            minimum=float(per.min()), strata=strata))
    eligible = [m for m in metrics if m['strict_recall_mean'] >= cfg['recall_target']]
    need(eligible, 'Strict full-bank ANN recall gate failed')
    selected = min(m['nprobe'] for m in eligible)
    need(selected == r['selected_nprobe'] == cfg['selected_nprobe'], 'Selected search setting differs from recall gate')
    return dict(status='PASS', exact_background_vertices=cfg['n_vertices'], queries=len(qi), k=cfg['k'],
                target=cfg['recall_target'], selected_nprobe=selected, results=metrics,
                criterion='mean strict ID top30 recall >=0.95; per-stratum and minimum also reported',
                audit_receipt_sha256=sha(ROOT / 'audit/fullbank_ann_recall.json'))


def partition_path(cfg, resolution, seed):
    if resolution == cfg['primary_resolution'] and seed == cfg['primary_seed']:
        return ROOT / 'clusters/primary_labels.npy', ROOT / 'clusters/primary_receipt.json'
    stem = 'resolution_' + str(float(resolution)).replace('.', 'p') + '_seed_' + str(seed)
    return ROOT / ('clusters/' + stem + '_labels.npy'), ROOT / ('clusters/' + stem + '_receipt.json')


def finalize(np, pq):
    from sklearn.metrics import adjusted_rand_score
    source_check(np, pq, receipt_name='source_final_check.json')
    cfg, contract = contract_gate()
    need(cfg['frozen'], 'ANN/graph design not frozen')
    graph_source = sha(ROOT / 'src/full_leiden_graph.py')
    design_hash = sha(ROOT / 'config/graph_design.json')
    preflight = passing('audit/source_preflight.json')
    need(preflight['input_sha256']['pca_scores'] == cfg['pca_scores_sha256'], 'Preflight bank mismatch')
    recall = recall_check(np, pq, cfg, graph_source)
    index = passing('graph/index_receipt.json', graph_source)
    need(index['n_vertices'] == cfg['n_vertices'] and index['source_sha256'] == cfg['pca_scores_sha256'], 'Wrong index source/count')
    need(sha(ROOT / 'graph/full.index') == index['index_sha256'], 'Full index hash mismatch')
    knn = passing('graph/knn_receipt.json', graph_source)
    need(knn['completed_rows'] == cfg['n_vertices'] and knn['design_sha256'] == design_hash, 'Incomplete kNN')
    need(knn['index_sha256'] == index['index_sha256'], 'kNN uses another index')
    offset = 0
    for rp in knn['shards_receipts']:
        row = read(rp)
        need(row['status'] == 'PASS' and row['source_code_sha256'] == graph_source, 'Invalid neighbor shard')
        need(row['start'] == offset and row['end'] > offset and row['end'] <= cfg['n_vertices'], 'kNN shard coverage gap')
        need(row['design_sha256'] == design_hash and row['source_sha256'] == cfg['pca_scores_sha256'], 'Shard provenance mismatch')
        need(row['index_sha256'] == index['index_sha256'] and row['nprobe'] == cfg['selected_nprobe'], 'Shard index/probe mismatch')
        for kind in ['indices', 'distances']:
            need(sha(row[kind + '_path']) == row[kind + '_sha256'], 'Shard file hash mismatch')
        ids, dd = np.load(row['indices_path'], mmap_mode='r'), np.load(row['distances_path'], mmap_mode='r')
        neighbor_check(np, ids, dd, np.arange(offset, row['end']), cfg['n_vertices'], cfg['k'])
        offset = row['end']
    need(offset == cfg['n_vertices'], 'Not all nodes have neighbors')
    graph = passing('graph/graph_receipt.json', graph_source)
    need(graph['n_vertices'] == cfg['n_vertices'] and graph['design_sha256'] == design_hash, 'Graph shape/design mismatch')
    for kind in ['edges', 'weights', 'edgelist']:
        need(sha(graph[kind + '_path']) == graph[kind + '_sha256'], 'Graph file hash mismatch: ' + kind)
        need(Path(graph[kind + '_path']).resolve().is_relative_to(ROOT.resolve()), 'Graph artifact outside new root')
    edges, weights = np.load(graph['edges_path'], mmap_mode='r'), np.load(graph['weights_path'], mmap_mode='r')
    need(edges.shape == (graph['n_undirected_edges'], 2) and weights.shape == (len(edges),), 'Graph array shapes disagree')
    seen = np.zeros(cfg['n_vertices'], dtype=bool)
    last = -1
    for start in range(0, len(edges), 1000000):
        e, w = edges[start:start + 1000000], weights[start:start + 1000000]
        need(np.all((e >= 0) & (e < cfg['n_vertices'])) and np.all(e[:, 0] < e[:, 1]), 'Invalid undirected edges')
        key = e[:, 0] * cfg['n_vertices'] + e[:, 1]
        need(int(key[0]) > last and np.all(np.diff(key) > 0), 'Graph edge ordering/duplicates invalid')
        last = int(key[-1])
        need(np.isfinite(w).all() and np.all((w > 0) & (w <= 1)), 'Invalid graph weights')
        seen[e[:, 0]] = True
        seen[e[:, 1]] = True
    isolated = int((~seen).sum())
    need(isolated == graph['n_isolated_vertices'], 'Isolated-vertex count mismatch')
    del seen
    partitions = []
    jobs = [(0.5, 17), (1.0, 17), (2.0, 17), (1.0, 29)]
    for resolution, seed in jobs:
        lp, rp = partition_path(cfg, resolution, seed)
        row = read(rp)
        need(row['status'] == 'PASS' and row['source_code_sha256'] == graph_source, 'Cluster receipt/source mismatch')
        need(row['resolution'] == resolution and row['seed'] == seed and row['n_iterations'] == cfg['full_leiden_iterations'], 'Wrong Leiden parameters')
        need(row['design_sha256'] == design_hash and row['source_sha256'] == cfg['pca_scores_sha256'], 'Cluster design/input mismatch')
        need(row['graph_edges_sha256'] == graph['edges_sha256'] and row['atlas_union_sha256'] == cfg['atlas_union_sha256'], 'Cluster graph/AA mismatch')
        need(sha(lp) == row['labels_sha256'], 'Full labels hash mismatch')
        labels = np.load(lp, mmap_mode='r', allow_pickle=False)
        need(labels.shape == (cfg['n_vertices'],) and np.issubdtype(labels.dtype, np.integer), 'Incomplete labels')
        need(int(labels.min()) == 0 and int(labels.max()) < cfg['n_vertices'], 'Invalid label IDs')
        sizes = np.bincount(labels)
        need(np.all(sizes > 0) and int(sizes.sum()) == cfg['n_vertices'], 'Labels omit vertices or contain ID gaps')
        need(len(sizes) == row['n_clusters'], 'Cluster count differs from actual labels')
        profile = ROOT / 'clusters' / ('sizes_r' + str(resolution).replace('.', 'p') + '_seed' + str(seed) + '.tsv')
        np.savetxt(profile, np.column_stack((np.arange(len(sizes)), sizes)), delimiter='\t', fmt='%d', header='cluster_id\tunique_AA', comments='')
        partitions.append(dict(resolution=resolution, seed=seed, n_vertices=len(labels), n_clusters=len(sizes),
                               singleton_clusters=int((sizes == 1).sum()), min_cluster_size=int(sizes.min()),
                               max_cluster_size=int(sizes.max()), median_cluster_size=float(np.median(sizes)),
                               cluster_size_quantiles={str(v): float(np.quantile(sizes, v)) for v in [0, .25, .5, .75, .9, .99, 1]},
                               labels_path=str(lp), labels_sha256=row['labels_sha256'], size_table_sha256=sha(profile)))
    main = np.load(partition_path(cfg, 1.0, 17)[0], mmap_mode='r')
    repeat = np.load(partition_path(cfg, 1.0, 29)[0], mmap_mode='r')
    ari = float(adjusted_rand_score(main, repeat))
    stability = passing('clusters/primary_seed_stability.json', graph_source)
    need(math.isclose(ari, stability['adjusted_rand_index'], abs_tol=1e-12), 'Seed ARI recomputation mismatch')
    need(stability['primary_labels_sha256'] == sha(partition_path(cfg, 1.0, 17)[0]) and
         stability['repeat_labels_sha256'] == sha(partition_path(cfg, 1.0, 29)[0]), 'Seed ARI label bindings mismatch')
    result = dict(status='PASS', completed_utc=now(), model='sceptr_cdr3_only', n_vertices=cfg['n_vertices'],
                  partitions=partitions, primary_seed_ARI=ari, independent_recall_recomputation=recall,
                  n_isolated_vertices=isolated, source_code_sha256=sha(__file__), graph_source_sha256=graph_source,
                  graph_design_sha256=design_hash, pca_scores_sha256=cfg['pca_scores_sha256'],
                  graph_edges_sha256=graph['edges_sha256'], phenotype_tables_read=[], gpu_used=False,
                  not_estimated=['recorded N+ association', 'predictive contribution', 'antigen binding'])
    dump(ROOT / 'audit/final_acceptance.json', result)
    dump(ROOT / 'report/full_cluster_counts_by_resolution.json', result)
    lines = ['# SCEPTR 全量 Leiden 正式结果', '', '完成：' + now(), '',
             '全部28,380,380条unique CDR3β AA保留；独立索引和图；全库精确近邻召回验收通过。', '',
             '| 分辨率 | 种子 | 实际簇数 | 单例簇 | 最小簇 | 中位数 | 最大簇 |',
             '|---:|---:|---:|---:|---:|---:|---:|']
    for r in partitions:
        lines.append('|{resolution}|{seed}|{n_clusters}|{singleton_clusters}|{min_cluster_size}|{median_cluster_size}|{max_cluster_size}|'.format(**r))
    selected = next(v for v in recall['results'] if v['nprobe'] == recall['selected_nprobe'])
    lines += ['', '主尺度两种子ARI：' + str(ari),
              '严格ID top30平均召回：' + str(selected['strict_recall_mean']) + '；nprobe=' + str(recall['selected_nprobe']),
              '每个长度/来源层的召回率及最低查询召回，见JSON回执。', '',
              '联合Home/External图仅用于探索。簇数不是N+贡献或抗原特异性的证据。']
    (ROOT / 'report/FINAL_COUNTS.md').write_text('\n'.join(lines) + '\n')
    reference = Path(contract['reference_root'])
    reference_result = read(reference / 'report/full_cluster_counts_by_resolution.json')
    need(reference_result['status'] == 'PASS', 'Reference full-atlas acceptance unavailable')
    refs = {(float(r['resolution']), int(r['seed'])): r for r in reference_result['partitions']}
    rows = []
    for r in partitions:
        ref = refs[(r['resolution'], r['seed'])]
        old = np.load(reference / partition_path(cfg, r['resolution'], r['seed'])[0].relative_to(ROOT), mmap_mode='r')
        need(old.shape == main.shape and sha(reference / partition_path(cfg, r['resolution'], r['seed'])[0].relative_to(ROOT)) == ref['labels_sha256'], 'Reference labels changed')
        new = np.load(r['labels_path'], mmap_mode='r')
        rows.append(dict(resolution=r['resolution'], seed=r['seed'], tcrbert_clusters=ref['n_clusters'],
                         sceptr_clusters=r['n_clusters'], full_vertex_cross_model_ARI=float(adjusted_rand_score(old, new))))
    comparisons = dict(status='PASS', n_vertices=cfg['n_vertices'], partitions=rows,
                       same_sampling_and_clustering_parameters=True, independent_indices_and_graphs=True,
                       reference_final_counts_sha256=sha(reference / 'report/full_cluster_counts_by_resolution.json'),
                       interpretation='Partition geometry comparison only; more clusters does not mean better N+ value')
    dump(ROOT / 'report/model_comparison.json', comparisons)
    lines = ['# TCR-BERT / SCEPTR 全量分区比较', '',
             '| 分辨率 | 种子 | TCR-BERT簇数 | SCEPTR簇数 | 全节点跨模型ARI |', '|---:|---:|---:|---:|---:|']
    for r in rows:
        lines.append('|{resolution}|{seed}|{tcrbert_clusters}|{sceptr_clusters}|{full_vertex_cross_model_ARI}|'.format(**r))
    lines += ['', '独立索引与图；同AA、相同PCA平衡抽样、k/权重/Leiden参数。ANN搜索强度按同一全库召回门槛独立锁定。',
              'ARI描述两种表示分区的一致性；本轮未比较患者级N+效应、未推断抗原。']
    (ROOT / 'report/MODEL_COMPARISON.md').write_text('\n'.join(lines) + '\n')
    print(json.dumps(dict(status='PASS', counts=[(r['resolution'], r['seed'], r['n_clusters']) for r in partitions])), flush=True)


def main():
    resource.setrlimit(resource.RLIMIT_AS, (512000000000, 512000000000))
    import numpy as np
    import pyarrow.parquet as pq
    from threadpoolctl import threadpool_limits
    threadpool_limits(limits=4)
    try:
        if sys.argv[1] == 'source':
            source_check(np, pq)
        elif sys.argv[1] == 'finalize':
            finalize(np, pq)
        else:
            raise ValueError('Expected source or finalize')
    except BaseException as e:
        dump(ROOT / ('audit/validation_failure_' + sys.argv[1] + '.json'),
             dict(status='FAIL', error=repr(e), time=now(), source_sha256=sha(__file__)))
        raise


if __name__ == '__main__':
    main()
