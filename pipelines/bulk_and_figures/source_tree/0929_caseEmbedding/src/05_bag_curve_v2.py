#!/usr/bin/env python3
"""Isolated post hoc MIL bag-size curve. Never rewrites the frozen 05_mil outputs."""
from __future__ import annotations
from release_runtime import private_literal as _private_literal, release_path as _release_path
import argparse
import hashlib
import importlib.util
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import psutil
import torch

ROOT = Path(_release_path('path_1304218e86d2e596'))
SPEC = importlib.util.spec_from_file_location('frozen_mil05', Path(__file__).with_name('05_mil.py'))
old = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = old
SPEC.loader.exec_module(old)
_ACTIVE_SEED = 0
_MAX_PREFIX = 32768


def paired_draw_v2(n: int, bag_size: int, patient: str, epoch: int, draw: int,
                   scope: str, seed: int = 0) -> np.ndarray:
    """Use one fixed random order so every smaller bag is a literal prefix."""
    if n <= 0 or bag_size <= 0 or bag_size > _MAX_PREFIX:
        raise ValueError('Invalid pool or bag size for draw-v2')
    rng = np.random.default_rng(old.stable_seed('draw-v2-fixed-prefix', patient,
                                                epoch, draw, scope, seed))
    order = rng.choice(n, size=min(_MAX_PREFIX, n), replace=False)
    pick = order[:min(bag_size, n)].astype(np.int64, copy=False)
    if len(pick) != min(bag_size, n) or len(np.unique(pick)) != len(pick):
        raise AssertionError('Draw-v2 duplicated or truncated a patient bag')
    return pick


def bag_tensors_v2(keys, index, bank, bag_size, epoch, draw, scope, device,
                   exclude=None, *, seed=None):
    seed = _ACTIVE_SEED if seed is None else seed
    x = np.zeros((len(keys), bag_size, bank.shape[1]), np.float32)
    count = np.zeros((len(keys), bag_size), np.float32)
    mask = np.zeros((len(keys), bag_size), np.bool_)
    for row, key in enumerate(keys):
        seq_idx, reads = index.get(key)
        if exclude and key in exclude:
            keep = ~np.isin(seq_idx, np.fromiter(exclude[key], dtype=np.int64))
            seq_idx, reads = seq_idx[keep], reads[keep]
        pick = paired_draw_v2(len(seq_idx), bag_size, key, epoch, draw, scope, seed)
        size = len(pick)
        x[row, :size] = np.asarray(bank[seq_idx[pick]], dtype=np.float32)
        count[row, :size] = np.log1p(reads[pick]).astype(np.float32)
        mask[row, :size] = True
        if int(mask[row].sum()) != min(bag_size, len(seq_idx)):
            raise AssertionError('Bag mask does not cover the requested patient AA')
    return tuple(torch.from_numpy(z).to(device) for z in (x, count, mask))


def validate_fold_scope(train, validation, home_keys, external_keys):
    tr, va, home, ext = map(set, (train, validation, home_keys, external_keys))
    if not tr or not va or len(tr) != len(train) or len(va) != len(validation):
        raise ValueError('Empty or repeated patient in fold')
    if tr & va or not (tr | va) <= home or (tr | va) & ext or home & ext:
        raise ValueError('Patient fold leakage or external participant')
    return True


def config(root):
    path = root / 'config/bag_curve_v2.json'
    d = json.loads(path.read_text())
    if (d.get('version') != 2 or d['draw_policy']['name'] != 'draw-v2-fixed-prefix' or
        d['draw_policy']['max_prefix_size'] != _MAX_PREFIX or
        tuple(d['bag_sizes']) != (1024, 2048, 4096, 8192, 16384)):
        raise ValueError('Bag curve config differs from locked v2 contract')
    return d


def source_hashes(root, model, outer, inner, bank):
    name = f'outer_{outer}' + ('' if inner is None else f'_inner_{inner}')
    return {
        'config': old.sha256(root / 'config/bag_curve_v2.json'),
        'runner': old.sha256(Path(__file__)),
        'frozen_mil': old.sha256(Path(old.__file__)),
        'folds': old.sha256(root / 'folds/folds.tsv'),
        'home_covariates': old.sha256(root / 'config/home_clinical_covariates.tsv'),
        'patient_index_receipt': old.sha256(root / 'mil/patient_index/receipt.json'),
        'bank_integrity_receipt': old.sha256(root / 'mil/bank_integrity.json'),
        'embedding_meta': old.sha256(root / 'embeddings' / model / 'meta.json'),
        'pca': old.sha256(root / 'pca' / model / f'outer_{outer}' /
                          ('' if inner is None else f'inner_{inner}') / 'pca.npz'),
        'projection_meta': old.sha256(root / 'mil/projections' / model / f'{name}_d64.json'),
        'projection_path': str(Path(bank.filename).resolve()),
    }


def projection(root, model, outer, inner, device):
    raw = old.load_bank(root, model)
    spec = old.pca_spec(root, model, outer, inner, raw.shape[1])
    if spec.raw_fallback or spec.dim != 64:
        raise RuntimeError('Curve requires sealed 64D fold-local PCA projection')
    bank = old.project_bank(root, model, outer, inner, raw, spec, device)
    return bank


def fit_path(root, outer, inner, model, bag, seed):
    scope = 'final' if inner is None else f'inner_{inner}'
    folder = root / 'mil/bag_curve_v2/fits' / f'outer_{outer}' / scope / model / f'bag_{bag}' / f'seed_{seed}'
    return folder / 'prediction.tsv', folder / 'receipt.json'


def fingerprint(hashes, outer, inner, model, bag, seed, train, test, labels,
                fixed_epochs, epoch_source_hashes):
    payload = {'source_hashes': hashes, 'outer': outer, 'inner': inner, 'model': model,
               'bag': bag, 'seed': seed, 'train': sorted((k, labels[k]) for k in train),
               'test': sorted((k, labels[k]) for k in test),
               'fixed_epochs': fixed_epochs, 'epoch_source_hashes': epoch_source_hashes,
               'draw_policy': 'draw-v2-fixed-prefix-32768', 'version': 2}
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def coverage(index, keys, bag, seed, epoch, scope):
    rows = []
    for key in keys:
        idx, reads = index.get(key)
        pick = paired_draw_v2(len(idx), bag, key, epoch, 0, scope, seed)
        rows.append((len(pick) / len(idx), float(reads[pick].sum() / reads.sum()), len(idx)))
    return rows


def run_fit(root, cfg, folds, labels, home_keys, external_keys, index, bank,
            outer, inner, model_name, bag, seed):
    global _ACTIVE_SEED
    tr, test = old.split_keys(folds, outer, inner)
    validate_fold_scope(tr, test, home_keys, external_keys)
    pred_path, receipt_path = fit_path(root, outer, inner, model_name, bag, seed)
    hashes = source_hashes(root, model_name, outer, inner, bank)
    epoch_sources = None
    if inner is None:
        paths = [fit_path(root, outer, j, model_name, bag, seed)[1] for j in cfg['inner_folds']]
        receipts = [json.loads(p.read_text()) for p in paths]
        if any(r.get('status') != 'complete' or r.get('draw_policy') != 'draw-v2-fixed-prefix-32768' for r in receipts):
            raise RuntimeError('Inner source epochs not complete')
        fixed_epochs = max(1, int(np.median([int(r['best_epoch']) for r in receipts])))
        epoch_sources = [old.sha256(p) for p in paths]
    else:
        fixed_epochs = None
    fprint = fingerprint(hashes, outer, inner, model_name, bag, seed, tr, test,
                         labels, fixed_epochs, epoch_sources)
    if pred_path.exists() and receipt_path.exists():
        saved = json.loads(receipt_path.read_text())
        if (saved.get('status') == 'complete' and saved.get('fingerprint') == fprint and
            saved.get('prediction_sha256') == old.sha256(pred_path)):
            frame = old.valid_saved_prediction(pred_path, test, labels, model_name,
                                               bag, seed, outer, inner)
            if frame is not None and frame.coverage_fraction.notna().all():
                return 'resumed'
        raise RuntimeError(f'Existing fit invalid; preserve and investigate: {receipt_path}')
    runtime = cfg['runtime']
    device = runtime['device']
    _ACTIVE_SEED = seed
    if inner is not None:
        fit, stop = old.early_stop_split(tr, labels, outer, inner, seed)
        validate_fold_scope(fit, stop, home_keys, external_keys)
        early_model, early_receipt = old.train_mil(fit, labels, index, bank, bag, seed, device,
             scope=f'curve-outer{outer}-inner{inner}-earlystop-seed{seed}', val_keys=stop,
             max_epochs=runtime['max_epochs'], patience=runtime['patience'],
             patient_batch=runtime['patient_batch'], mc_val=runtime['early_stop_mc_bags'],
             max_gpu_gib=runtime['max_gpu_allocated_gib'])
        fixed_epochs = int(early_receipt['best_epoch'])
        del early_model
        torch.cuda.empty_cache()
    else:
        early_receipt = None
    model_obj, receipt = old.train_mil(tr, labels, index, bank, bag, seed, device,
        scope=f'curve-outer{outer}-' + ('final' if inner is None else f'inner{inner}') + f'-seed{seed}',
        fixed_epochs=fixed_epochs, patient_batch=runtime['patient_batch'],
        max_gpu_gib=runtime['max_gpu_allocated_gib'])
    begin = time.monotonic()
    inference_scope = f'curve-outer{outer}-' + ('test' if inner is None else f'inner{inner}-val')
    pred, sd, draws, unresolved = old.predict_adaptive(model_obj, test, index,
                                                        bank, bag, device, scope=inference_scope)
    infer_seconds = time.monotonic() - begin
    cov = coverage(index, test, bag, seed, -1, inference_scope)
    frame = pd.DataFrame({'patient_key': test, 'label': [labels[k] for k in test],
        'prediction': pred, 'mc_sd': sd, 'mc_draws': draws, 'outer_fold': outer,
        'inner_fold': inner if inner is not None else np.nan, 'model': model_name,
        'bag_size': bag, 'seed': seed, 'coverage_fraction': [v[0] for v in cov],
        'coverage_read_fraction': [v[1] for v in cov], 'n_unique_aa': [v[2] for v in cov]})
    if not np.isfinite(frame[['prediction','mc_sd','coverage_fraction','coverage_read_fraction']]).all().all():
        raise ValueError('Nonfinite predictions or coverage')
    pred_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(pred_path, sep='\t', index=False)
    old.atomic_json(receipt_path, {**receipt, 'status':'complete', 'fingerprint':fprint,
        'draw_policy':'draw-v2-fixed-prefix-32768', 'source_hashes':hashes,
        'prediction_sha256':old.sha256(pred_path), 'fixed_epochs':fixed_epochs,
        'early_stop_receipt':early_receipt, 'epoch_source_hashes':epoch_sources,
        'inference_seconds':round(infer_seconds,3), 'mc_draws':draws,
        'mc_unresolved_sd_gt_0_05':unresolved, 'peak_cpu_rss_gib':old.check_rss(),
        'scope':'final' if inner is None else f'inner_{inner}',
        'n_test':len(test), 'metrics':old.metrics(frame.label,frame.prediction)})
    del model_obj
    torch.cuda.empty_cache()
    return 'trained'


def run_pilot(root, cfg, folds, labels, home_keys, external_keys, index):
    pilot = cfg['pilot']; outer=pilot['outer_fold']; inner=pilot['inner_fold']; seed=pilot['seed']
    tr, val = old.split_keys(folds, outer, inner)
    validate_fold_scope(tr, val, home_keys, external_keys)
    large = [k for k in tr if int(index.stats.loc[k,'unique_aa']) >= max(pilot['bag_sizes'])]
    group = {y: sorted([k for k in large if labels[k]==y]) for y in (0,1)}
    fit = group[0][:pilot['fit_patients']//2] + group[1][:pilot['fit_patients']//2]
    stop = [k for k in group[0] if k not in fit][:pilot['validation_patients']//2] + [k for k in group[1] if k not in fit][:pilot['validation_patients']//2]
    if len(fit) != pilot['fit_patients'] or len(stop) != pilot['validation_patients']:
        raise RuntimeError('Insufficient full-bag patients for balanced pilot')
    validate_fold_scope(fit, stop, home_keys, external_keys)
    out = root / 'mil/bag_curve_v2/pilot'; out.mkdir(parents=True, exist_ok=True)
    receipts=[]
    for model_name in pilot['models']:
        bank = projection(root, model_name, outer, inner, cfg['runtime']['device'])
        for bag in pilot['bag_sizes']:
            path=out / f'{model_name}_bag_{bag}.json'
            source=source_hashes(root, model_name, outer, inner, bank)
            if path.exists():
                saved=json.loads(path.read_text())
                if saved.get('status')=='pass' and saved.get('source_hashes')==source:
                    receipts.append(saved); continue
                raise RuntimeError(f'Existing pilot changed: {path}')
            global _ACTIVE_SEED
            _ACTIVE_SEED=seed
            for key in fit+stop:
                if len(paired_draw_v2(len(index.get(key)[0]), bag, key, 0, 0, 'train', seed)) != bag:
                    raise AssertionError('Pilot bag not full size')
            model_obj, fit_receipt = old.train_mil(fit, labels, index, bank, bag, seed,
                cfg['runtime']['device'], scope=f'curve-pilot-{model_name}-{bag}',
                val_keys=stop, max_epochs=pilot['epochs'], patience=pilot['epochs'],
                patient_batch=cfg['runtime']['patient_batch'], mc_val=2,
                max_gpu_gib=pilot['max_gpu_allocated_gib'])
            begin=time.monotonic()
            old.predict_mc(model_obj, stop, index, bank, bag, cfg['runtime']['device'],
                           scope=f'curve-pilot-{model_name}', draws=8)
            infer_seconds=time.monotonic()-begin
            record={'status':'pass', 'model':model_name, 'bag_size':bag,
                 'full_bag_validated':True, 'fit_patients':len(fit),'validation_patients':len(stop),
                 'fit_patient_sha256':old.train_patient_hash(fit),
                 'validation_patient_sha256':old.train_patient_hash(stop),
                 'fit':fit_receipt,'inference_seconds':round(infer_seconds,3),
                 'peak_cpu_rss_gib':old.check_rss(), 'source_hashes':source}
            if (fit_receipt['peak_gpu_gib'] > pilot['max_gpu_allocated_gib'] or
                record['peak_cpu_rss_gib'] > pilot['max_cpu_rss_gib']):
                record['status']='fail_resource'
            old.atomic_json(path,record)
            receipts.append(record)
            del model_obj
            torch.cuda.empty_cache()
        del bank
    if any(r['status']!='pass' for r in receipts):
        raise RuntimeError('Large-bag pilot resource gate failed')
    old.atomic_json(out/'summary.json',{'status':'pass','pilot_count':len(receipts),
        'config_sha256':old.sha256(root/'config/bag_curve_v2.json'),
        'runner_sha256':old.sha256(Path(__file__)),
        'peak_gpu_gib':max(r['fit']['peak_gpu_gib'] for r in receipts),
        'total_seconds':sum(r['fit']['seconds']+r['inference_seconds'] for r in receipts)})


def run_curve(root,cfg,folds,labels,home_keys,external_keys,index):
    pilot_path=root/'mil/bag_curve_v2/pilot/summary.json'
    if not pilot_path.exists() or json.loads(pilot_path.read_text()).get('status')!='pass':
        raise RuntimeError('Large-bag pilot must pass before formal curve')
    total=len(cfg['outer_folds'])*len(cfg['models'])*len(cfg['bag_sizes'])*len(cfg['seeds'])*(len(cfg['inner_folds'])+1)
    complete=0
    for outer in cfg['outer_folds']:
        for inner in [*cfg['inner_folds'],None]:
            for model_name in cfg['models']:
                bank=projection(root,model_name,outer,inner,cfg['runtime']['device'])
                for bag in cfg['bag_sizes']:
                    for seed in cfg['seeds']:
                        result=run_fit(root,cfg,folds,labels,home_keys,external_keys,index,bank,
                                       outer,inner,model_name,bag,seed)
                        complete+=1
                        print(json.dumps({'event':'fit','done':complete,'total':total,'outer':outer,
                            'inner':inner,'model':model_name,'bag':bag,'seed':seed,
                            'result':result}),flush=True)
                del bank
                old.check_rss(cfg['runtime']['max_cpu_rss_gib'])
    old.atomic_json(root/'mil/bag_curve_v2/fits_complete.json',{
        'status':'complete','count':complete,'expected_count':total,
        'config_sha256':old.sha256(root/'config/bag_curve_v2.json'),
        'runner_sha256':old.sha256(Path(__file__))})


old.bag_tensors = bag_tensors_v2

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--root',type=Path,default=ROOT)
    parser.add_argument('--phase',choices=['pilot','curve'],required=True)
    args=parser.parse_args()
    root=args.root.resolve(); cfg=config(root)
    old.set_threads(cfg['runtime']['cpu_threads'])
    old.require_primary_bank_receipt(root)
    manifest=old.read_manifest(root)
    labels=old.read_home_labels(root,manifest)
    folds=old.read_folds(root,manifest)
    home_keys=set(labels)
    external_keys=set(manifest.loc[manifest.cohort.eq('external'),'patient_key'])
    index=old.PatientIndex(root,verify_sources=False)
    if not home_keys <= set(index.files):
        raise RuntimeError('Home patient AA index incomplete')
    with old.gpu_slot(root,cfg['runtime']['device'],cfg['runtime']['max_gpu_allocated_gib']):
        if args.phase=='pilot':run_pilot(root,cfg,folds,labels,home_keys,external_keys,index)
        else:run_curve(root,cfg,folds,labels,home_keys,external_keys,index)

if __name__=='__main__':main()
