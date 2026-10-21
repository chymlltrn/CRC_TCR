#!/usr/bin/env python3
"""Summarize frozen, already executed Q1/Q2 reference-method outputs.

This script performs table aggregation and provenance checks only. It does not
query antigen databases, compute receptor distances, load models or predict
binding. ERGO-II resources are explicitly unused and excluded from delivery.
"""
from release_runtime import private_literal as _private_literal, release_path as _release_path
from pathlib import Path
import hashlib
import json
import pandas as pd

BASE = Path(_release_path('path_1757610fc64073d4'))
R17 = Path(__file__).resolve().parents[1]
OUT = R17 / 'analysis/antigen_predictions'
V7 = BASE / 'revision_v7/antigen_bridge'
V5 = BASE / 'revision_v5/antigen'
AUDIT = R17 / 'audit/antigen_predictions_audit.json'
Q1_NT = _private_literal('private_c287d28aaf0e9d7c')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return pd.read_csv(path, sep='\t', keep_default_na=False)


def save(table, name):
    table.to_csv(OUT / name, sep='\t', index=False, na_rep='NA')


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    files = {
        'inherited_reference_evidence': V7 / 'candidate_inherited_TCRMatch_tcrdist_source_evidence.tsv',
        'tcrdist_members': V7 / 'candidate_tcrdist_radius24_members.tsv.gz',
        'tcrdist_branch_summary': V7 / 'candidate_tcrdist_branch_summary.tsv',
        'observed_branches': V7 / 'candidate_observed_branches.tsv',
        'branch_annotations': V7 / 'candidate_branch_annotations.tsv',
        'tcrdist_receipt': V7 / 'candidate_tcrdist_receipt.json',
        'tcrdist_api_checks': V7 / 'candidate_tcrdist_public_API_checks.tsv',
        'gliph_projection': V7 / 'GLIPH2_candidate_pattern_projection.tsv',
        'gliph_receipt': V7 / 'GLIPH2_receipt.json',
        'gliph_input_receipt': V7 / 'gliph2/input_receipt.json',
        'gliph_cluster_summary': V7 / 'GLIPH2_cluster_summary.tsv',
        'q1_tcrmatch_raw': V5 / 'inputs/Q1_TCRMatch_raw.tsv',
        'q1_tcrmatch_receipt': V5 / 'inputs/Q1_TCRMatch_receipt.json',
        'q2_tcrmatch_raw': V5 / 'q2_tcrmatch/matches.raw.tsv',
        'q2_tcrmatch_receipt': V5 / 'q2_tcrmatch/query_receipt.json',
        'tcrmatch_methods': V5 / 'inputs/TCRMatch_methods.md',
        'frozen_query_sequence_definitions': V5 / 'query_definitions_frozen.json',
        'tcrdist_original_code': BASE / 'revision_v7/code/v7_antigen_bridge_neighbors.py',
        'gliph_original_code': BASE / 'revision_v7/code/v7_antigen_bridge_gliph.py',
        'existing_references': BASE / 'revision_v16_core_narrative/manuscript/references_resolved.tsv',
    }
    inherited = read(files['inherited_reference_evidence'])
    inherited['existing_output_path'] = str(files['inherited_reference_evidence'])
    inherited['existing_output_row_1based_with_header'] = inherited.index + 2
    exact = inherited[inherited.match_kind.eq('exact_complete_AA')].copy()
    tm = inherited[inherited.method.eq('TCRMatch')].copy()
    td = read(files['tcrdist_members'])
    td['existing_output_path'] = str(files['tcrdist_members'])
    td['existing_output_row_1based_with_header'] = td.index + 2
    td['score_unit'] = 'weighted beta-loop distance; lower is closer; not binding probability'
    td['existing_radius'] = 24
    td['patient_HLA_known'] = False
    td['paired_patient_alpha_known'] = False
    td['NT_binding_validation'] = False
    branches = read(files['tcrdist_branch_summary'])
    annotations = read(files['branch_annotations'])
    branchq = read(files['observed_branches']).rename(columns={'query': 'candidate'})
    gliph = read(files['gliph_projection'])
    gr = json.loads(files['gliph_receipt'].read_text())
    tr = json.loads(files['tcrdist_receipt'].read_text())
    raw1 = read(files['q1_tcrmatch_raw'])
    raw2 = read(files['q2_tcrmatch_raw'])
    r1 = json.loads(files['q1_tcrmatch_receipt'].read_text())
    r2 = json.loads(files['q2_tcrmatch_receipt'].read_text())
    frozen_queries = json.loads(files['frozen_query_sequence_definitions'].read_text())
    assert frozen_queries['Q1']['designated_NT'] == Q1_NT
    assert frozen_queries['Q1']['designated_NT_observed_VJ'] == 'TRBV5-5/TRBJ1-4'
    assert len(exact) == 6 and len(td) == 805 and len(branches) == 38
    assert len(raw1) == 0 and len(raw2) == 5 and len(gliph) == 0
    assert r1['output_sha256'] == sha(files['q1_tcrmatch_raw'])
    assert r2['output_sha256'] == sha(files['q2_tcrmatch_raw'])
    assert tr['radius'] == 24 and not tr['J_distance_in_metric']
    assert gr['reported_groups'] == 275 and gr['candidate_projection_rows'] == 0
    assert len(read(files['gliph_cluster_summary'])) == 275
    assert td.groupby('candidate').size().to_dict() == {'Q1': 2, 'Q2': 803}
    assert tm[tm.candidate_id.eq('Q2')].score.astype(float).tolist() == raw2.score.astype(float).tolist()
    hit = exact[exact.matched.astype(str).str.lower().eq('true')]
    assert len(hit) == 1
    assert hit.iloc[0].peptide == _private_literal('private_ff1a6ad26dad5d79')
    assert hit.iloc[0].reference_v == 'TRBV7-6' and hit.iloc[0].reference_j == 'TRBJ2-1'
    assert hit.iloc[0].reference_HLA == 'HLA class I'
    assert td[td.candidate.eq('Q1')].query_V.eq('TRBV7-2').all()
    assert not td[td.candidate.eq('Q1')].complete_AA_exact.astype(str).str.lower().eq('true').any()

    source_rows = []
    for role, path in files.items():
        source_rows.append({'role': role, 'path': str(path), 'sha256': sha(path),
                            'v17_action': 'read existing output or documentation; no method rerun'})
    save(pd.DataFrame(source_rows), 'source_provenance.tsv')
    save(inherited[inherited.method.isin(['complete_CDR3beta_reference', 'TCRMatch'])],
         'existing_exact_hamming_tcrmatch_evidence.tsv')
    save(td, 'existing_tcrdist_all_neighbors.tsv')
    save(branches, 'existing_tcrdist_branch_summary.tsv')
    save(annotations, 'existing_exact_branch_annotations.tsv')
    branchq['NT_scope'] = 'AA/V/J branch; no branch-specific NT binding validation'
    branchq['nominated_Q1_assay_branch'] = branchq.candidate.eq('Q1') & branchq.V.eq('TRBV5-5') & branchq.J.eq('TRBJ1-4')
    branchq['nominated_Q1_assay_NT'] = branchq.nominated_Q1_assay_branch.map({True: Q1_NT, False: 'NA'})
    branchq['paired_patient_alpha_available'] = False
    branchq['patient_HLA_available'] = False
    save(branchq, 'candidate_query_scopes.tsv')

    rows = []
    def summary(candidate, method, scope, result, n, score, unit, threshold, target, caveat, source):
        rows.append(dict(candidate=candidate, method=method, query_scope=scope, result=result,
                         n_existing_output_records=n, displayed_score=score, score_unit=unit,
                         existing_analysis_threshold=threshold, target_context=target,
                         interpretation_boundary=caveat, source_path=str(source)))
    for candidate in ['Q1', 'Q2']:
        e = exact[exact.candidate_id.eq(candidate)]
        n = int(e.matched.astype(str).str.lower().eq('true').sum())
        summary(candidate, 'Exact reference lookup', 'complete CDR3beta AA; V/J concordance checked separately',
                'No exact match in VDJdb, IEDB or McPAS' if n == 0 else '1 IEDB exact beta-AA annotation; VDJdb/McPAS none',
                n, 'NA', 'identity, not probability', 'complete AA equality',
                'NA' if n == 0 else _private_literal('private_69b27d9fc82ce4cb'),
                'Q2 annotation is V7-6/J2-1 compatible only; patient alpha/HLA unknown; no functional validation',
                files['inherited_reference_evidence'])
        m = tm[tm.candidate_id.eq(candidate)]
        n = int(m.matched.astype(str).str.lower().eq('true').sum())
        summary(candidate, 'TCRMatch', 'CDR3beta AA trimmed by removal of leading C and one final F/W; V/J not input',
                'No match at similarity >=0.97' if n == 0 else '5 hits: 1 exact + 4 nonidentical trimmed sequence neighbors',
                n, 'NA' if n == 0 else '0.971643–1.000000', 'normalized k-mer kernel similarity; higher is closer', '>=0.97',
                'NA' if n == 0 else 'YFV NS4b; SARS-CoV-2 Spike, orf1ab, ORF7b',
                'Same IEDB reference as exact lookup; repeated exact hit is not independent support; score is not binding probability',
                files['q1_tcrmatch_raw'] if candidate == 'Q1' else files['q2_tcrmatch_raw'])
    for row in branches[branches.candidate.eq('Q1')].itertuples():
        target = _private_literal('private_bff3bf900e21e4ab') if row.n_annotations else 'NA'
        summary('Q1', 'tcrdist3', row.V + '/' + row.J,
                ('2 nonidentical reference neighbors' if row.n_annotations else 'No neighbor within existing radius24'),
                int(row.n_annotations), str(int(row.minimum_distance)), 'minimum weighted beta-loop distance; lower is closer', '<=24', target,
                ('Nominated assay branch' if row.V == 'TRBV5-5' else 'Alternative observed AA/V/J branch') +
                '; J does not enter distance and is checked separately; unknown V allele minimized; no binding inference',
                files['tcrdist_branch_summary'])
    q2 = td[td.candidate.eq('Q2')]
    summary('Q2', 'tcrdist3', '35 observed V/J branches; 34 with radius24 neighbors',
            '803 branch/reference rows; 728 unique reference records; 403 reference beta AAs; 190 peptide strings',
            len(q2), '0–24', 'weighted beta-loop distance; lower is closer', '<=24',
            'Broad reference contexts; full branch-resolved table supplied',
            'Counts overlap across query branches; not 803 experiments; V7-6/J2-1 exact reference distance0; broad neighborhood not target assignment',
            files['tcrdist_members'])
    for candidate in ['Q1', 'Q2']:
        summary(candidate, 'GLIPH2 pattern projection', 'post hoc projection into 275 groups from expanded tissue TCRs in 18 patients',
                'No pattern projection; candidate AA absent from matched tissue dataset', 0, 'NA',
                'group/pattern membership; no peptide-specific model score', 'existing emitted global/local pattern rules', 'NA',
                'Not a direct candidate GLIPH2 antigen screen; no cluster antigen-label transfer; absence does not exclude antigen recognition',
                files['gliph_projection'])
    save(pd.DataFrame(rows), 'method_candidate_summary.tsv')

    targets = []
    for row in pd.concat([hit, tm[tm.matched.astype(str).str.lower().eq('true')]]).itertuples():
        targets.append(dict(candidate=row.candidate_id, method=row.method,
                            query_AA=row.query_aa, query_V=row.query_v, query_J=row.query_j,
                            query_NT_scope='AA query only; does not validate assay NT',
                            match_kind=row.match_kind, reference_CDR3beta=row.reference_cdr3beta,
                            reference_V=row.reference_v, reference_J=row.reference_j,
                            peptide=row.peptide, protein=row.antigen, organism=row.organism,
                            reference_HLA=row.reference_HLA, score=row.score,
                            score_unit=row.score_type, threshold=row.threshold,
                            reference_key=row.reference_key, reference_id=row.reference_id,
                            source_file=row.source_file, source_rows=row.source_rows,
                            existing_output_path=row.existing_output_path,
                            existing_output_row=row.existing_output_row_1based_with_header,
                            scope_note=row.VJ_status + '; no patient alpha/HLA or functional binding evidence'))
    for row in td[td.candidate.eq('Q1')].itertuples():
        targets.append(dict(candidate='Q1', method='tcrdist3', query_AA=row.query_AA,
                            query_V=row.query_V, query_J=row.query_J,
                            query_NT_scope='Alternative V7-2 AA/V/J branch; not nominated V5-5 assay target',
                            match_kind='nonidentical_beta_neighbor', reference_CDR3beta=row.ref_cdr3beta,
                            reference_V=row.ref_v, reference_J=row.ref_j,
                            peptide=row.epitope, protein=row.antigen, organism=row.source_organism,
                            reference_HLA=row.mhc, score=row.distance_beta,
                            score_unit='weighted beta-loop distance', threshold='<=24',
                            reference_key=row.reference_key, reference_id=row.reference_id,
                            source_file=row.source_file, source_rows=row.source_rows,
                            existing_output_path=row.existing_output_path,
                            existing_output_row=row.existing_output_row_1based_with_header,
                            scope_note='Query/reference V and J compatible; nonidentical AA; no binding inference'))
    save(pd.DataFrame(targets), 'figure5_target_evidence.tsv')

    refs = read(files['existing_references'])
    keys = ['TCRMATCH2021', 'TCRDIST3_2021', 'GLIPH2_2020', _private_literal('private_0e46635da4c0e3db'), 'MCPAS2017', _private_literal('private_bb4a9dae21569c81'), 'DASH2017_PAIRED']
    citations = refs[refs.id.isin(keys)].to_dict('records')
    assert len(citations) == len(keys)
    (OUT / 'primary_method_citations.json').write_text(json.dumps(citations, indent=2, ensure_ascii=False) + '\n')
    ergo_manifest = OUT / 'official/ERGO-II/official_download_manifest.json'
    unused = {'method': 'ERGO-II', 'status': 'DOWNLOADED_NOT_SCORED',
              'reason': 'Parent reports automatic safety review rejected scoring and prohibited retry; bounded task uses existing outputs only',
              'checkpoint_loaded': False, 'prediction_computed': False,
              'output_scores_exist': False, 'release_exclude_glob': 'analysis/antigen_predictions/official/ERGO-II/**',
              'repository': 'https://github.com/IdoSpringer/ERGO-II',
              'source_commit': '85d320ab03ade33460be9a81ea3a51b8d37cd998',
              'download_manifest_path': str(ergo_manifest),
              'download_manifest_sha256': sha(ergo_manifest) if ergo_manifest.exists() else None}
    (OUT / 'unused_model_resources_audit.json').write_text(json.dumps(unused, indent=2) + '\n')
    deliverable_names = ['source_provenance.tsv', 'existing_exact_hamming_tcrmatch_evidence.tsv',
                         'existing_tcrdist_all_neighbors.tsv', 'existing_tcrdist_branch_summary.tsv',
                         'existing_exact_branch_annotations.tsv', 'candidate_query_scopes.tsv',
                         'method_candidate_summary.tsv', 'figure5_target_evidence.tsv',
                         'primary_method_citations.json', 'unused_model_resources_audit.json']
    deliverable_names += [n for n in ['method_interpretation_and_figure_notes.md', 'primary_source_evidence_ledger.json'] if (OUT / n).exists()]
    audit = {'status': 'PASS_EXISTING_OUTPUT_DOCUMENTATION', 'scope': 'Aggregation of previously executed outputs only',
             'script_sha256': sha(__file__), 'new_database_queries': 0, 'new_distance_computations': 0,
             'new_binding_model_scores': 0, 'binding_probability_inferred': False,
             'checks': {'source_receipts_and_raw_TCRMatch_hashes_verified': True,
                        'Q1_exact_three_databases_negative': True, 'Q2_exact_IEDB_reference_annotation_one': True,
                        'Q1_TCRMatch_no_hits': True, 'Q2_TCRMatch_five_hits_preserved_exactly': True,
                        'tcrdist_all_805_source_rows_preserved': True, 'all_38_query_branches_preserved': True,
                        'Q1_assay_V5_5_separated_from_V7_2_neighbors': True,
                        'latest_275_GLIPH2_groups_used_instead_of_obsolete_40_pattern_row': True},
             'limitations': ['Large frozen normalized reference table is not present locally; existing source-backlinked outputs and receipts were read; no fresh full-database scan claimed',
                             'Exact lookup and TCRMatch share IEDB evidence; method agreement is not independent experimental validation',
                             'tcrdist J compatibility is separate from weighted beta-loop distance; unknown V alleles minimized',
                             'GLIPH2 result is tissue-group projection, not a direct candidate antigen prediction',
                             'Reference HLA cannot be assigned to study patients; paired alpha and patient HLA unavailable',
                             'Q1 nomination NT and V5-5/J1-4 assay branch are not functionally characterized'],
             'unused_model_resources': unused,
             'inputs': source_rows,
             'release_include_files': [{'path': str(OUT / n), 'sha256': sha(OUT / n)} for n in deliverable_names],
             'release_exclude_globs': ['analysis/antigen_predictions/official/ERGO-II/**']}
    AUDIT.write_text(json.dumps(audit, indent=2, ensure_ascii=False) + '\n')
    print(json.dumps({'status': audit['status'], 'summary_rows': len(rows), 'target_evidence_rows': len(targets),
                      'tcrdist_rows': len(td), 'output_directory': str(OUT)}))


if __name__ == '__main__':
    main()
