#!/usr/bin/env python3
"""Extend sealed CDR3beta AA embedding banks for atlas_340 without copying old banks.

Modes: --prepare, --pilot MODEL, --full MODEL, --verify. Run with old embedding venv.
Index columns: seq_idx, aa, bank_source (old/new), bank_idx. The new bank
is indexed by data/new_aa_union.parquet seq_idx; old bank by old union seq_idx.
"""
from __future__ import annotations
from release_runtime import private_literal as _private_literal, release_path as _release_path

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import resource
import sys
import time

ROOT = Path(_release_path('path_99d8c1c6fde2a6b9'))
OLD = ROOT.parent
OLD_AA = OLD / 'data/aa_union.parquet'
ATLAS_AA = ROOT / 'data/aa_union.parquet'
INDEX = ROOT / 'data/atlas_aa_index.parquet'
NEW_AA = ROOT / 'data/new_aa_union.parquet'
MANIFEST = ROOT / 'embeddings/bank_manifest.json'
PILOT_IDS = ROOT / 'embeddings/pilot_seq_idx.npy'
PILOT_META = ROOT / 'embeddings/pilot_selection.json'
MODELS = ('tcrbert_dualtask_aa_mean', 'sceptr_cdr3_only')
DIMS = dict(zip(MODELS, (768, 64)))


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


base = module('sealed_embed_io', OLD / 'src/02_embed.py')


def need(ok, why):
    if not ok:
        raise RuntimeError(why)


def now():
    return datetime.now(timezone.utc).isoformat()


def rss_gib():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024**2


def parquet_atomic(table, path):
    import pyarrow.parquet as pq
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f'.partial.{os.getpid()}')
    try:
        pq.write_table(table, tmp, compression='zstd', row_group_size=100_000)
        with tmp.open('rb') as stream:
            os.fsync(stream.fileno())
        digest = base.sha256(tmp)
        os.replace(tmp, path)
        return digest
    finally:
        tmp.unlink(missing_ok=True)


def union(path, verify_unique):
    import numpy as np
    import pyarrow as pa
    import pyarrow.compute as pc
    import pyarrow.parquet as pq
    need(path.is_file(), f'Missing union {path}')
    table = pq.read_table(path, columns=['seq_idx', 'aa'])
    n = table.num_rows
    need(n > 0 and table.schema.field('seq_idx').type == pa.int64(),
         f'Invalid union row count/index {path}')
    need(table.schema.field('aa').type in (pa.string(), pa.large_string()),
         f'Invalid AA type {path}')
    ids = table['seq_idx'].to_numpy(zero_copy_only=False)
    need(np.array_equal(ids, np.arange(n, dtype=np.int64)),
         f'Union physical rows and seq_idx disagree: {path}')
    aas = table['aa']
    need(aas.null_count == 0 and bool(pc.all(pc.match_substring_regex(
        aas, '^[ACDEFGHIKLMNPQRSTVWY]+$')).as_py()),
        f'Noncanonical/null AA in {path}')
    if verify_unique:
        need(pc.count_distinct(aas).as_py() == n, f'Duplicate AA in {path}')
    return table, base.sha256(path)


def sealed_old():
    import numpy as np
    import pyarrow.parquet as pq
    receipt = json.loads((OLD / 'mil/bank_integrity.json').read_text())
    need(receipt['status'] == 'complete', 'Old bank integrity receipt incomplete')
    n = pq.ParquetFile(OLD_AA).metadata.num_rows
    need(n == 23_974_109, f'Old AA count changed: {n}')
    old_sha = base.sha256(OLD_AA)
    models = {}
    for model in MODELS:
        folder = OLD / 'embeddings' / model
        meta = json.loads((folder / 'meta.json').read_text())
        stamp = receipt['models'][model]['bank_stamp']
        bank = folder / 'bank_fp16.npy'
        stat = bank.stat()
        arr = np.load(bank, mmap_mode='r', allow_pickle=False)
        need(meta['status'] == 'complete' and meta['n_total'] == n
             and meta['input_sha256'] == old_sha
             and meta['bank_sha256'] == receipt['models'][model]['bank_sha256']
             and stat.st_size == stamp['bytes'] and stat.st_mtime_ns == stamp['mtime_ns']
             and arr.shape == (n, DIMS[model]) and arr.dtype == np.float16,
             f'Sealed {model} bank changed')
        models[model] = dict(old_bank_path=str(bank), old_bank_sha256=meta['bank_sha256'],
                             old_bank_size_bytes=stat.st_size,
                             old_bank_mtime_ns=stat.st_mtime_ns,
                             old_model_fingerprint=meta['model_fingerprint'],
                             old_model_spec=meta['model_spec'], dimension=DIMS[model],
                             new_bank_path=str(ROOT / 'embeddings' / model / 'bank_new_fp16.npy'))
    return n, old_sha, models


def prepare():
    import numpy as np
    import pyarrow as pa
    import pyarrow.compute as pc
    import pyarrow.parquet as pq
    start = time.time()
    pa.set_cpu_count(16)
    n_old, old_sha, models = sealed_old()
    if MANIFEST.is_file():
        prior = json.loads(MANIFEST.read_text())
        need(ATLAS_AA.is_file(), f'Missing atlas union {ATLAS_AA}')
        need(prior['atlas_union_sha256'] == base.sha256(ATLAS_AA)
             and prior['old_union_sha256'] == old_sha
             and prior['n_atlas'] == pq.ParquetFile(ATLAS_AA).metadata.num_rows
             and INDEX.is_file() and NEW_AA.is_file()
             and prior['index_sha256'] == base.sha256(INDEX)
             and prior['new_union_sha256'] == base.sha256(NEW_AA),
             'Prepared atlas provenance changed; refusing to overwrite')
        return prior
    atlas, atlas_sha = union(ATLAS_AA, True)
    n = atlas.num_rows
    lengths = pc.utf8_length(atlas['aa'])
    long = pc.greater(lengths, 62)
    long_n = int(pc.sum(pc.cast(long, pa.int64())).as_py() or 0)
    length_qc = dict(min=int(pc.min(lengths).as_py()), max=int(pc.max(lengths).as_py()),
                     over_62_count=long_n,
                     over_62_examples=atlas['aa'].filter(long).slice(0, 20).to_pylist(),
                     rule='TCR-BERT max 64 tokens including CLS/SEP; never truncate')
    base.atomic_json(ROOT / 'embeddings/length_qc.json', length_qc)
    need(long_n == 0, f'{long_n} qualified AA exceed frozen BERT 62-AA capacity')
    old, _ = union(OLD_AA, False)
    hit = pc.fill_null(pc.index_in(atlas['aa'], value_set=old['aa']), -1)
    old_idx = hit.to_numpy(zero_copy_only=False).astype(np.int64, copy=False)
    is_new = old_idx < 0
    n_new = int(is_new.sum())
    bank_idx = old_idx.copy()
    bank_idx[is_new] = np.arange(n_new, dtype=np.int64)
    old_match = pc.equal(atlas['aa'].filter(pa.array(~is_new)),
                         old['aa'].take(pa.array(old_idx[~is_new])))
    need(bool(pc.all(old_match).as_py()), 'Old bank AA mapping disagrees')
    index = pa.table(dict(seq_idx=atlas['seq_idx'], aa=atlas['aa'],
                          bank_source=pc.if_else(pa.array(is_new), 'new', 'old'),
                          bank_idx=pa.array(bank_idx, type=pa.int64())))
    new = pa.table(dict(seq_idx=pa.array(np.arange(n_new, dtype=np.int64)),
                        aa=atlas['aa'].filter(pa.array(is_new))))
    index_sha = parquet_atomic(index, INDEX)
    new_sha = parquet_atomic(new, NEW_AA)
    manifest = dict(status='index_prepared', prepared_utc=now(),
                    atlas_union_path=str(ATLAS_AA), atlas_union_sha256=atlas_sha,
                    old_union_path=str(OLD_AA), old_union_sha256=old_sha,
                    index_path=str(INDEX), index_sha256=index_sha,
                    new_union_path=str(NEW_AA), new_union_sha256=new_sha,
                    n_atlas=n, n_old_bank=n_old, n_reused=n-n_new, n_new=n_new,
                    length_qc=length_qc, model_banks=models,
                    mapping_rule='exact full CDR3beta AA',
                    prepare_elapsed_sec=round(time.time()-start, 2),
                    prepare_peak_rss_gib=round(rss_gib(), 2))
    need(rss_gib() <= 128, 'Stage2 prepare exceeded 128 GiB RSS policy')
    base.atomic_json(MANIFEST, manifest)
    print(json.dumps({k: manifest[k] for k in
                      ('n_atlas', 'n_reused', 'n_new', 'prepare_elapsed_sec',
                       'prepare_peak_rss_gib')}, indent=2), flush=True)
    return manifest


def backend(model, batch):
    meta = json.loads((OLD / 'embeddings' / model / 'meta.json').read_text())
    if model == MODELS[0]:
        author = module('sealed_author_encoder', OLD / 'embeddings' / model / 'run.py')
        spec = author.build_model_spec()
        cls = author.AuthorEncoder('cuda', 'fp16', batch, 12288)
    else:
        author = None
        spec = base.model_spec('sceptr_cdr3_only', base.DEFAULT_BERT, 'fp16')
        cls = base.SceptrCdr3('cuda', batch, 12288)
    need(spec == meta['model_spec'] and base.json_hash(spec) == meta['model_fingerprint'],
         f'Frozen {model} model, package, or script changed')
    return cls, meta['model_fingerprint'], author


def choose_pilot(manifest):
    import numpy as np
    need(manifest['n_new'] > 0, 'No new AA to encode')
    if PILOT_META.is_file():
        receipt = json.loads(PILOT_META.read_text())
        need(receipt['new_union_sha256'] == manifest['new_union_sha256']
             and receipt['requested_rows'] == 50_000 and PILOT_IDS.is_file()
             and receipt['indices_sha256'] == base.sha256(PILOT_IDS),
             'Existing pilot selection changed')
        return np.load(PILOT_IDS, allow_pickle=False), receipt
    ids, _, strata = base.stratified_pilot(NEW_AA, manifest['n_new'], 50_000, 20260930)
    ids_sha = base.atomic_npy(PILOT_IDS, ids)
    receipt = dict(new_union_sha256=manifest['new_union_sha256'],
                   requested_rows=50_000, selected_rows=len(ids),
                   indices_sha256=ids_sha, strata=strata, seed=20260930)
    base.atomic_json(PILOT_META, receipt)
    return ids, receipt


def recompute_old(model, encoder):
    import numpy as np
    import pyarrow as pa
    import pyarrow.parquet as pq
    ids = np.sort(np.random.default_rng(20260930).choice(23_974_109, 64, False))
    aa = pq.read_table(OLD_AA, columns=['aa'])['aa'].take(pa.array(ids)).to_pylist()
    got = encoder.embed(aa).astype(np.float32)
    ref = np.load(OLD / 'embeddings' / model / 'bank_fp16.npy', mmap_mode='r',
                  allow_pickle=False)[ids].astype(np.float32)
    den = np.linalg.norm(got, axis=1) * np.linalg.norm(ref, axis=1)
    need(np.all(den > 0), 'Old-bank recomputation zero vector')
    cos = np.einsum('ij,ij->i', got, ref) / den
    rel = np.linalg.norm(got-ref, axis=1)/np.linalg.norm(ref, axis=1)
    need(float(cos.min()) >= .999 and float(rel.max()) <= .02,
         f'{model} encoder disagrees with old bank: {cos.min()}, {rel.max()}')
    return dict(rows=64, seed=20260930,
                indices_sha256=hashlib.sha256(ids.tobytes()).hexdigest(),
                cosine_min=float(cos.min()), relative_l2_max=float(rel.max()))


def pilot(model, batch):
    import numpy as np
    import pyarrow as pa
    import pyarrow.parquet as pq
    manifest = prepare()
    ids, selected = choose_pilot(manifest)
    out = ROOT / 'embeddings' / model
    out.mkdir(parents=True, exist_ok=True)
    path = out / 'pilot.json'
    if path.is_file():
        receipt = json.loads(path.read_text())
        need(receipt['status'] == 'complete'
             and receipt['new_union_sha256'] == manifest['new_union_sha256']
             and receipt['pilot_indices_sha256'] == selected['indices_sha256']
             and receipt['model_fingerprint'] == manifest['model_banks'][model]['old_model_fingerprint'],
             'Existing pilot provenance differs')
        print(f'Existing {model} pilot verified', flush=True)
        return
    aa = pq.read_table(NEW_AA, columns=['aa'])['aa'].take(pa.array(ids)).to_pylist()
    with base.gpu_lock(OLD / '.gpu.lock', model, 'atlas_340_pilot'):
        base.preflight_gpu('cuda')
        enc, fingerprint, author = backend(model, batch)
        enc.torch.cuda.reset_peak_memory_stats()
        t0 = time.time()
        vec = enc.embed(aa)
        sec = time.time()-t0
        need(vec.shape == (len(ids), DIMS[model]) and np.isfinite(vec).all(),
             'Pilot vectors wrong shape/nonfinite')
        peak = enc.torch.cuda.max_memory_reserved()/1024**2
        old_match = recompute_old(model, enc)
        precision = (author.precision_check(aa, vec, 12288) if author else
                     base.precision_comparison('sceptr_cdr3_only', base.DEFAULT_BERT,
                                               aa, vec, 12288, 20260930))
        receipt = dict(status='complete', utc=now(), model=model,
                       model_fingerprint=fingerprint,
                       new_union_sha256=manifest['new_union_sha256'],
                       pilot_indices_sha256=selected['indices_sha256'],
                       pilot_rows=len(ids), initial_batch_size=batch,
                       effective_batch_size=enc.batch_size,
                       encode_sec=round(sec, 2), rows_per_sec=round(len(ids)/sec, 2),
                       estimated_full_hours=round(manifest['n_new']/(len(ids)/sec)/3600, 2),
                       gpu_peak_reserved_mib=round(peak, 1), gpu_peak_limit_mib=12288,
                       precision_comparison=precision,
                       old_bank_recomputation=old_match,
                       nonstandard_junction_warnings=(enc.warning_counter.count if not author else 0),
                       peak_rss_gib=round(rss_gib(), 2),
                       script_sha256=base.sha256(Path(__file__)))
        base.atomic_json(path, receipt)
    print(json.dumps(receipt, indent=2, sort_keys=True), flush=True)


def publish(out, n, dim, rows, input_sha, fingerprint):
    import numpy as np
    final = out / 'bank_new_fp16.npy'
    temp = out / f'bank_new_fp16.npy.partial.{os.getpid()}'
    try:
        bank = np.lib.format.open_memmap(temp, 'w+', dtype=np.float16, shape=(n, dim))
        for start in range(0, n, rows):
            end = min(start+rows, n)
            need(base.verify_shard(out, start, end, dim, input_sha, fingerprint),
                 f'Missing/invalid shard {start}:{end}')
            shard = np.load(out/'shards'/f'{base.shard_name(start, end)}.npy',
                            mmap_mode='r', allow_pickle=False)
            need(np.isfinite(shard).all(), f'Nonfinite shard {start}:{end}')
            bank[start:end] = shard
        bank.flush()
        del bank
        with temp.open('rb') as handle:
            os.fsync(handle.fileno())
        digest = base.sha256(temp)
        os.replace(temp, final)
        return digest
    finally:
        temp.unlink(missing_ok=True)


def full(model, batch, rows):
    import numpy as np
    manifest = prepare()
    need(rows in (50_000, 100_000), 'Shard rows must be 50k or 100k')
    out = ROOT/'embeddings'/model
    pilot_receipt = json.loads((out/'pilot.json').read_text())
    need(pilot_receipt['status'] == 'complete'
         and pilot_receipt['new_union_sha256'] == manifest['new_union_sha256']
         and pilot_receipt['gpu_peak_reserved_mib'] <= 12288
         and pilot_receipt['old_bank_recomputation']['cosine_min'] >= .999,
         '50k technical pilot not passed')
    n = manifest['n_new']
    fingerprint = manifest['model_banks'][model]['old_model_fingerprint']
    meta_path = out/'meta.json'
    if meta_path.is_file():
        meta = json.loads(meta_path.read_text())
        need(meta['new_union_sha256'] == manifest['new_union_sha256']
             and meta['model_fingerprint'] == fingerprint
             and meta['n_new'] == n and meta['shard_rows'] == rows,
             'Existing full-run provenance differs')
        if meta['status'] == 'complete':
            bank = out/'bank_new_fp16.npy'
            need(bank.is_file() and base.sha256(bank) == meta['bank_sha256'],
                 'Published new bank checksum differs')
            print(f'Complete new bank verified: {bank}', flush=True)
            return
    else:
        meta = dict(status='running', utc_started=now(), model=model,
                    new_union_path=str(NEW_AA),
                    new_union_sha256=manifest['new_union_sha256'],
                    model_fingerprint=fingerprint,
                    old_bank_sha256=manifest['model_banks'][model]['old_bank_sha256'],
                    n_new=n, dimension=DIMS[model], shard_rows=rows,
                    initial_batch_size=batch, gpu_peak_limit_mib=12288,
                    script_sha256=base.sha256(Path(__file__)))
        base.atomic_json(meta_path, meta)
    (out/'shards').mkdir(exist_ok=True)
    started = time.time()
    skipped = done = warnings = 0
    max_peak = 0.0
    with base.gpu_lock(OLD/'.gpu.lock', model, 'atlas_340_full'):
        enc = None
        for start, end, aas in base.iter_shards(NEW_AA, rows, n):
            if base.verify_shard(out, start, end, DIMS[model],
                                 manifest['new_union_sha256'], fingerprint):
                skipped += 1
                receipt = json.loads((out/'shards'/f'{base.shard_name(start, end)}.json').read_text())
                warnings += receipt['nonstandard_junction_warnings']
                max_peak = max(max_peak, receipt['gpu_peak_reserved_mib'])
                done = end
                continue
            if enc is None:
                base.preflight_gpu('cuda')
                enc, observed_fingerprint, _ = backend(model, batch)
                need(observed_fingerprint == fingerprint, 'Frozen model drift')
                enc.torch.cuda.reset_peak_memory_stats()
            before = enc.warning_counter.count if model == MODELS[1] else 0
            vectors = enc.embed(aas)
            need(vectors.shape == (end-start, DIMS[model])
                 and np.isfinite(vectors).all(), 'Nonfinite/incomplete shard')
            delta = enc.warning_counter.count-before if model == MODELS[1] else 0
            warnings += delta
            peak = enc.torch.cuda.max_memory_reserved()/1024**2
            max_peak = max(max_peak, peak)
            base.write_shard(out, start, end, vectors,
                             manifest['new_union_sha256'], fingerprint,
                             round(peak, 1), delta)
            done = end
            print(f'{model}: {done}/{n}; peak {peak:.0f} MiB; '
                  f'elapsed {time.time()-started:.1f}s', flush=True)
    need(done == n and base.sha256(NEW_AA) == manifest['new_union_sha256'],
         'New union changed or shards incomplete')
    meta.update(status='shards_complete', n_done=done,
                shards_verified_on_resume=skipped,
                nonstandard_junction_warnings_total=warnings,
                encoding_elapsed_sec=round(time.time()-started, 2),
                gpu_peak_reserved_mib=round(max_peak, 1),
                effective_batch_size=enc.batch_size if enc else batch,
                peak_rss_gib=round(rss_gib(), 2))
    base.atomic_json(meta_path, meta)
    digest = publish(out, n, DIMS[model], rows, manifest['new_union_sha256'], fingerprint)
    bank = out/'bank_new_fp16.npy'
    meta.update(status='complete', utc_complete=now(), bank_path=str(bank),
                bank_sha256=digest, bank_size_bytes=bank.stat().st_size,
                bank_mtime_ns=bank.stat().st_mtime_ns)
    base.atomic_json(meta_path, meta)
    print(f'Published {model} new bank: {bank}', flush=True)


def verify():
    import numpy as np
    import pyarrow.compute as pc
    import pyarrow.parquet as pq
    manifest = prepare()
    index = pq.read_table(INDEX)
    need(index.num_rows == manifest['n_atlas'], 'Index row count changed')
    new = pc.equal(index['bank_source'], 'new').to_numpy(zero_copy_only=False)
    ids = index['bank_idx'].to_numpy(zero_copy_only=False)
    old_n = int((~new).sum())
    new_n = int(new.sum())
    need(old_n == manifest['n_reused'] and new_n == manifest['n_new']
         and np.all(ids[~new] >= 0) and np.all(ids[~new] < manifest['n_old_bank'])
         and np.array_equal(ids[new], np.arange(new_n, dtype=np.int64)),
         'Index source counts/positions wrong')
    model_receipts = {}
    for model in MODELS:
        meta = json.loads((ROOT/'embeddings'/model/'meta.json').read_text())
        bank = ROOT/'embeddings'/model/'bank_new_fp16.npy'
        need(meta['status'] == 'complete' and meta['n_done'] == new_n
             and meta['new_union_sha256'] == manifest['new_union_sha256']
             and bank.is_file() and base.sha256(bank) == meta['bank_sha256'],
             f'New bank incomplete or checksum changed: {model}')
        arr = np.load(bank, mmap_mode='r', allow_pickle=False)
        need(arr.shape == (new_n, DIMS[model]) and arr.dtype == np.float16,
             f'New bank shape/dtype wrong: {model}')
        for start in range(0, new_n, 100_000):
            need(np.isfinite(arr[start:start+100_000]).all(),
                 f'New bank nonfinite: {model}:{start}')
        model_receipts[model] = dict(new_bank_path=str(bank),
                                     new_bank_sha256=meta['bank_sha256'],
                                     old_bank_sha256=manifest['model_banks'][model]['old_bank_sha256'],
                                     n_reused=old_n, n_new=new_n, nonfinite=0)
    receipt = dict(status='PASS', utc=now(), atlas_union_sha256=manifest['atlas_union_sha256'],
                   atlas_index_sha256=manifest['index_sha256'],
                   n_atlas=manifest['n_atlas'], n_reused=old_n, n_new=new_n,
                   models=model_receipts, peak_rss_gib=round(rss_gib(), 2))
    base.atomic_json(ROOT/'embeddings/verification.json', receipt)
    manifest['status'] = 'complete'
    manifest['verified_utc'] = receipt['utc']
    base.atomic_json(MANIFEST, manifest)
    print(json.dumps(receipt, indent=2, sort_keys=True), flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    group = p.add_mutually_exclusive_group(required=True)
    group.add_argument('--prepare', action='store_true')
    group.add_argument('--pilot', choices=MODELS)
    group.add_argument('--full', choices=MODELS)
    group.add_argument('--verify', action='store_true')
    p.add_argument('--batch-size', type=int, default=256)
    p.add_argument('--shard-rows', type=int, default=50_000)
    args = p.parse_args()
    model = args.pilot or args.full
    expected = (Path(_release_path('path_70a00081c087c53f'))
                if model == MODELS[0] else OLD/'envs/embedding/bin/python')
    need(Path(sys.executable).resolve() == expected.resolve(),
         f'Use frozen embedding environment {expected}')
    need(1 <= args.batch_size <= 512, 'Batch size must be 1..512')
    policy = json.loads((OLD/'config/resource_policy.json').read_text())
    need(policy['gpu_peak_mib_target'] == 12288, 'GPU resource policy changed')
    for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'RAYON_NUM_THREADS'):
        os.environ[key] = '4'
    os.environ['TOKENIZERS_PARALLELISM'] = 'true'
    if args.prepare:
        prepare()
    elif args.pilot:
        pilot(args.pilot, args.batch_size)
    elif args.full:
        full(args.full, args.batch_size, args.shard_rows)
    else:
        verify()


if __name__ == '__main__':
    try:
        main()
    except Exception as exc:
        print(f'ATLAS EMBEDDING FAILED: {type(exc).__name__}: {exc}',
              file=sys.stderr, flush=True)
        raise
