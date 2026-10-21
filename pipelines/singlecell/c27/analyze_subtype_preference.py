#!/usr/bin/env python3
"""Paired-sample, donor-balanced descriptive receptor/subtype association.

No cells are treated as independent biological replicates. No new P values are
computed. Donor bootstrap preserves each donor's repeated samples and all
subtypes. Model-matched comparator and exact AA membership are explicit.
"""
from release_config import release_path, private_value
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

TARGETS = [('tcrbert', 7, 'TCR-BERT 7'), ('tcrbert', 11, 'TCR-BERT 11'), ('sceptr', 5, 'SCEPTR 5')]
SEED = 20261012


def paired_profiles(frame, model, cluster, subtypes, weighting, min_target=1, min_ref=1):
    """Return donor means of sample-paired subtype compositions plus provenance."""
    col = model + '_cluster'
    x = frame.loc[frame[col].ge(0)].copy()
    x['group'] = np.where(x[col].eq(cluster), 'target', 'reference')
    if weighting == 'clone_weighted':
        valid = x[['nt', 'v', 'j']].notna().all(axis=1) & x.nt.astype(str).str.len().gt(0)
        x = x.loc[valid].copy()
        # A beta NT+V+J clonotype has total weight one within each RNA sample.
        x['weight'] = 1 / x.groupby(['Patient', 'Ident', 'nt_vj_key']).cell_name.transform('size')
        test = x.groupby(['Patient', 'Ident', 'nt_vj_key']).weight.sum()
        np.testing.assert_allclose(test, 1)
    else:
        x['weight'] = 1.
    counts = x.groupby(['Patient', 'Ident', 'group']).size().unstack(fill_value=0)
    counts = counts.reindex(columns=['target', 'reference'], fill_value=0)
    counts['paired'] = counts.target.ge(min_target) & counts.reference.ge(min_ref)
    valid_pairs = counts.index[counts.paired]
    x = x.set_index(['Patient', 'Ident']).loc[lambda z: z.index.isin(valid_pairs)].reset_index()
    if x.empty:
        return None, counts.reset_index(), x
    weighted = x.groupby(['Patient', 'Ident', 'group', 'SubCellType']).weight.sum().unstack(fill_value=0)
    weighted = weighted.reindex(columns=subtypes, fill_value=0)
    proportions = weighted.div(weighted.sum(axis=1), axis=0)
    np.testing.assert_allclose(proportions.sum(axis=1), 1)
    donor = proportions.groupby(['Patient', 'group']).mean()
    t = donor.xs('target', level='group').sort_index()
    r = donor.xs('reference', level='group').reindex(t.index)
    assert t.index.equals(r.index) and not r.isna().any().any()
    return (t, r), counts.reset_index(), x


def estimate(t, r, donor_order, bootstrap_weights):
    present = pd.Index(donor_order).isin(t.index).astype(float)
    tm = t.reindex(donor_order, fill_value=0).to_numpy()
    rm = r.reindex(donor_order, fill_value=0).to_numpy()
    denom = bootstrap_weights @ present
    bt = np.full((len(denom), len(t.columns)), np.nan)
    br = np.full_like(bt, np.nan)
    ok = denom > 0
    bt[ok] = (bootstrap_weights[ok] @ tm) / denom[ok, None]
    br[ok] = (bootstrap_weights[ok] @ rm) / denom[ok, None]
    point_t = t.mean(axis=0).to_numpy()
    point_r = r.mean(axis=0).to_numpy()
    delta = 100 * (point_t - point_r)
    bd = 100 * (bt - br)
    ratio = np.divide(point_t, point_r, out=np.full_like(point_t, np.nan), where=point_r > 0)
    ratio_boot = np.divide(bt, br, out=np.full_like(bt, np.nan), where=br > 0)
    unstable = (~np.isfinite(ratio_boot)).mean(axis=0)
    lo, hi = np.nanquantile(bd, [.025, .975], axis=0)
    rlo = np.full_like(ratio, np.nan)
    rhi = np.full_like(ratio, np.nan)
    stable = (unstable == 0) & (point_t > 0)
    if stable.any():
        rlo[stable], rhi[stable] = np.quantile(ratio_boot[:, stable], [.025, .975], axis=0)
    return dict(target_fraction=point_t, reference_fraction=point_r,
                delta_pp=delta, delta_ci_low=lo, delta_ci_high=hi,
                enrichment_ratio=ratio, ratio_ci_low=rlo, ratio_ci_high=rhi,
                ratio_bootstrap_undefined_fraction=unstable,
                ratio_zero_reference_fraction=np.mean(br == 0, axis=0),
                bootstrap_no_eligible_donor_fraction=np.full_like(point_t, np.mean(~ok)),
                ratio_status=np.where(point_r == 0, 'zero_reference',
                                      np.where(point_t == 0, 'no_target_subtype_observed',
                                               np.where(stable, 'estimated', 'unstable_bootstrap_denominator')))), bd


def self_test():
    # Donor A has nine samples, donor B one. Donors must still each weigh 1/2.
    rows = []
    for donor, ns, sub in [('A', 9, 's1'), ('B', 1, 's2')]:
        for s in range(ns):
            for group, subtype in [(7, sub), (1, 's2')]:
                rows.append(dict(Patient=donor, Ident=f'{donor}_{s}', tcrbert_cluster=group,
                                 SubCellType=subtype, cell_name=f'{donor}_{s}_{group}',
                                 nt='AAA', v='V', j='J', nt_vj_key=f'AAA{group}'))
    data = pd.DataFrame(rows)
    (t, r), counts, _ = paired_profiles(data, 'tcrbert', 7, ['s1', 's2'], 'cell_weighted')
    np.testing.assert_allclose(t.mean(), [.5, .5])
    np.testing.assert_allclose(r.mean(), [0, 1])
    # Clonal duplication changes cell weighting but not clone-weighted composition.
    a = data[data.Patient.eq('B')].copy()
    b = a.iloc[[0]].copy(); b['SubCellType'] = 's1'; b['nt_vj_key'] = 'CCC'
    a = pd.concat([a, pd.concat([b] * 9)], ignore_index=True)
    a['cell_name'] = np.arange(len(a)).astype(str)
    (tc, _), _, _ = paired_profiles(a, 'tcrbert', 7, ['s1', 's2'], 'clone_weighted')
    np.testing.assert_allclose(tc.iloc[0], [.5, .5])
    weights = np.array([[2, 0], [0, 2], [1, 1]])
    e, bd = estimate(t, r, ['A', 'B'], weights)
    assert np.isnan(e['enrichment_ratio'][0]) and e['enrichment_ratio'][1] == .5
    np.testing.assert_allclose(bd, [[100, -100], [0, 0], [50, -50]])
    return {'equal_donor_weight': True, 'clone_weight_total_one': True,
            'zero_reference_not_regularized': True, 'paired_bootstrap': True}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', type=Path, required=True)
    ap.add_argument('--bootstrap', type=int, default=10000)
    args = ap.parse_args()
    out = args.root / 'analysis' / 'subtype_preference'; out.mkdir(parents=True, exist_ok=True)
    audit = args.root / 'audit'; audit.mkdir(exist_ok=True)
    source = args.root / 'analysis' / 'mapping' / 'sc_singlebeta_model_assignments.parquet'
    frame = pd.read_parquet(source)
    assert frame.cell_name.is_unique and frame.single_beta.all()
    assert frame[['Patient', 'Ident', 'SubCellType', 'UMAP1', 'UMAP2']].notna().all().all()
    valid_receptor = (frame[['nt', 'v', 'j']].notna().all(axis=1)
                      & frame[['nt', 'v', 'j']].astype(str).apply(lambda x: x.str.strip().str.len().gt(0)).all(axis=1))
    assert valid_receptor.all(), 'Missing NT/V/J requires an additional same-subset cell-weighted analysis.'
    subtypes = sorted(frame.SubCellType.unique())
    assert len(subtypes) == 32
    scopes = {
        'CRC_all': frame.is_crc,
        'all_atlas': np.ones(len(frame), dtype=bool),
        'CRC_strict_AAVJ': frame.is_crc & frame.strict_aavj_ge2,
        'CRC_baseline': frame.is_crc & frame.is_baseline,
    }
    for tissue in sorted(frame.Tissue.unique()):
        scopes['CRC_' + tissue] = frame.is_crc & frame.Tissue.eq(tissue)
    results, donor_rows, sample_rows, coverage = [], [], [], []
    for scope, mask in scopes.items():
        scoped = frame.loc[mask].copy()
        donor_order = sorted(scoped.Patient.unique())
        if not donor_order:
            continue
        rng = np.random.default_rng(SEED)
        bw = rng.multinomial(len(donor_order), np.full(len(donor_order), 1 / len(donor_order)), args.bootstrap)
        for weighting in ['cell_weighted', 'clone_weighted']:
            family_rows, family_boot = [], []
            for model, cluster, label in TARGETS:
                res, sample, used = paired_profiles(scoped, model, cluster, subtypes, weighting)
                sample['scope'], sample['weighting'], sample['target_cluster'] = scope, weighting, label
                sample_rows.append(sample)
                if res is None:
                    continue
                t, r = res
                effects, boot = estimate(t, r, donor_order, bw)
                n = len(t)
                stat = pd.DataFrame(effects)
                stat['SubCellType'] = subtypes
                stat['scope'], stat['weighting'], stat['target_cluster'] = scope, weighting, label
                stat['model'], stat['cluster_id'], stat['paired_donors'] = model, cluster, n
                stat['paired_samples'] = int(sample.paired.sum())
                stat['target_subtype_donors'] = (t > 0).sum().to_numpy()
                stat['reference_subtype_donors'] = (r > 0).sum().to_numpy()
                stat['positive_delta_donors'] = (t > r).sum().to_numpy()
                stat['target_cells'] = used.group.eq('target').sum()
                stat['reference_cells'] = used.group.eq('reference').sum()
                stat['uncertainty_status'] = 'exploratory_donor_bootstrap' if n >= 6 else 'fewer_than_6_donors_descriptive_only'
                if n < 6:
                    stat[['delta_ci_low', 'delta_ci_high', 'ratio_ci_low', 'ratio_ci_high']] = np.nan
                for group, mat in [('target', t), ('reference', r)]:
                    long = mat.rename_axis('Patient').reset_index().melt(id_vars='Patient', var_name='SubCellType', value_name='fraction')
                    long['scope'], long['weighting'], long['target_cluster'], long['group'] = scope, weighting, label, group
                    donor_rows.append(long)
                coverage.append(dict(scope=scope, weighting=weighting, target_cluster=label,
                                     eligible_donors=n, eligible_samples=int(sample.paired.sum()),
                                     excluded_unpaired_samples=int((~sample.paired).sum()),
                                     target_cells=int(used.group.eq('target').sum()), reference_cells=int(used.group.eq('reference').sum()),
                                     target_unique_NT_VJ_sequences=int(used.loc[used.group.eq('target'), 'nt_vj_key'].nunique()),
                                     target_donor_specific_beta_clones=int(used.loc[used.group.eq('target'), ['Patient', 'nt_vj_key']].drop_duplicates().shape[0]),
                                     target_sample_specific_beta_clone_occurrences=int(used.loc[used.group.eq('target'), ['Patient', 'Ident', 'nt_vj_key']].drop_duplicates().shape[0])))
                family_rows.append(stat); family_boot.append(boot)
            if family_rows:
                family = pd.concat(family_rows, ignore_index=True)
                boots = np.concatenate(family_boot, axis=1)
                eligible = family.paired_donors.ge(6).to_numpy()
                if eligible.any():
                    centered = boots[:, eligible] - family.loc[eligible, 'delta_pp'].to_numpy()
                    valid = np.isfinite(centered).all(axis=1)
                    # Do not silently condition an interval on a heavily reduced
                    # bootstrap distribution when eligible donors are scarce.
                    critical = (np.quantile(np.abs(centered[valid]).max(axis=1), .95)
                                if valid.mean() >= .99 else np.nan)
                    family['delta_simultaneous_ci_low'] = np.where(eligible, np.maximum(-100, family.delta_pp - critical), np.nan)
                    family['delta_simultaneous_ci_high'] = np.where(eligible, np.minimum(100, family.delta_pp + critical), np.nan)
                    family['simultaneous_family_size'] = int(eligible.sum())
                    family['simultaneous_critical_pp'] = critical
                    family['simultaneous_valid_bootstrap_n'] = int(valid.sum())
                    family['simultaneous_valid_bootstrap_fraction'] = float(valid.mean())
                results.append(family)
                print(scope, weighting, 'DONE', flush=True)
    final = pd.concat(results, ignore_index=True)
    final.to_csv(out / 'subtype_enrichment_all_scopes.tsv', sep='\t', index=False)
    primary = final[final.scope.eq('CRC_all') & final.weighting.eq('cell_weighted')]
    primary.to_csv(out / 'subtype_enrichment_primary.tsv', sep='\t', index=False)
    pd.concat(donor_rows).to_csv(out / 'donor_subtype_proportions.tsv.gz', sep='\t', index=False)
    pd.concat(sample_rows).to_csv(out / 'sample_pairing_denominators.tsv.gz', sep='\t', index=False)
    pd.DataFrame(coverage).to_csv(out / 'analysis_denominators.tsv', sep='\t', index=False)
    # Rank every subtype; selection is by effect size, never a fresh P cutoff.
    primary.sort_values(['target_cluster', 'delta_pp'], ascending=[True, False]).to_csv(out / 'ranked_subtype_preferences.tsv', sep='\t', index=False)
    nt_missing = int(frame[['nt', 'v', 'j']].isna().any(axis=1).sum())
    checks = dict(self_tests=self_test(), source=str(source), source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                  input_singlebeta_cells=len(frame), subtypes=len(subtypes), primary_rows=len(primary),
                  missing_NT_VJ_rows=nt_missing, bootstrap=args.bootstrap, seed=SEED,
                  estimator='Within-sample target/reference composition; samples equal within donor; donors equal; mean ratio',
                  paired_sample_minimum='At least one target and one same-model assigned reference cell',
                  primary_scope='CRC; exact CDR3beta AA membership; single accepted beta',
                  clone_proxy='Within RNA sample beta CDR3 NT + V + J, total weight 1 split over cell states; not paired alphabeta identity',
                  ci='Pointwise percentile donor bootstrap; additional approximate 95% simultaneous absolute-deviation band over eligible subtype contrasts per scope and weighting; actual family size recorded, up to 96',
                  inference='Descriptive receptor-associated composition. No new P values, no independent cell tests, no absolute expansion claim.',
                  software={'numpy': np.__version__, 'pandas': pd.__version__})
    assert len(primary) == 96
    (audit / 'subtype_analysis_checks.json').write_text(json.dumps(checks, indent=2) + '\n')
    print(primary.sort_values('delta_pp', ascending=False).groupby('target_cluster', sort=False).head(5)[['target_cluster', 'SubCellType', 'enrichment_ratio', 'delta_pp', 'paired_donors']].to_string(index=False))


if __name__ == '__main__':
    main()
