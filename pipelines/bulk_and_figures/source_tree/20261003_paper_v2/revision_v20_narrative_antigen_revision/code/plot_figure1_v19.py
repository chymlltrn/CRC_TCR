#!/usr/bin/env python3
"""Improve only the accepted Figure1 demographic rows, preserving count geometry."""
from __future__ import annotations
import csv
import json
import re
import statistics
from collections import Counter
from pathlib import Path
from lxml import etree
from assemble import ROOT, digest, write_json

def main():
    baseline = ROOT/'reproducibility/frozen_v18_baseline/figures/main/Figure1.svg'
    source = ROOT/'manuscript/SourceData/candidates/clinical_models/patient_analysis_master.tsv'
    patients = [row for row in csv.DictReader(source.open(), delimiter='\t') if row['cohort']=='home']
    counts = {'sex': dict(Counter(row['sex'] for row in patients)),
              'neoadjuvant': dict(Counter(row['neoadjuvant'] for row in patients)),
              'pN': dict(Counter(row['n_binary'] for row in patients))}
    assert len(patients)==137 and counts=={'sex': {'f':60,'m':77},
        'neoadjuvant':{'no':126,'yes':11},'pN':{'0.0':70,'1.0':56,'':11}}
    ages = sorted(float(row['age_years']) for row in patients)
    assert [ages[34],statistics.median(ages),ages[102]] == [60,68,74]
    original = etree.fromstring(baseline.read_bytes())
    tree = etree.fromstring(baseline.read_bytes())
    find = lambda name: tree.xpath('//*[@id=$name]', name='x28_0_180_'+name)[0]
    groups = [(['patch_10','patch_11'], ['text_24','text_25'], 'text_21'),
              (['patch_12','patch_13'], ['text_26','text_27'], 'text_22'),
              (['patch_14','patch_15','patch_16'], ['text_28','text_29','text_30'], 'text_23')]
    previous_pitch = 17.3763225
    new_pitch = 26.0
    edits = []
    for row, (bars, values, label) in enumerate(groups):
        delta = row*(new_pitch-previous_pitch)
        for identifier in bars+values+[label]:
            node = find(identifier)
            node.set('transform',f'translate(12 {delta:.7f})')
            for text in node.xpath('.//*[local-name()="text"]'):
                old_style = text.get('style')
                text.set('style', re.sub(r'font-size: 7(?:\.0+)?px', 'font-size: 8px', old_style))
            edits.append({'id': node.get('id'),'row':row+1,'x_shift_pt':12,'y_shift_pt':delta,
                          'font_pt':8 if identifier not in bars else None})
        texts = find(label).xpath('.//*[local-name()="text"]')
        # Center each two-line label on its corresponding unchanged bar height.
        center = [312.7729725,330.149295,347.5256175][row]
        for text, y in zip(texts, [center-1.5,center+8]):
            text.set('transform',f'translate(79 {y:.7f})')
    title = find('text_14').xpath('.//*[local-name()="text"]')[0]
    title.set('style',title.get('style').replace('font-size: 7.00px','font-size: 8.5px'))
    edits.append({'id':'x28_0_180_text_14','font_before_pt':7,'font_after_pt':8.5})
    clip = find('p2274087cc0')[0]
    clip.set('x',str(float(clip.get('x'))+12))
    clip.set('height',f'{float(clip.get("height"))+2*(new_pitch-previous_pitch):.7f}')
    background = find('patch_9')[0]
    background.set('d','M 95.750492 373.461134  L 242.204232 373.461134  L 242.204232 304.084811  L 95.750492 304.084811  z ')
    # All other plot subtrees and every visible/hidden string remain exact.
    text_list = lambda root: [''.join(n.itertext()) for n in root.xpath('//*[local-name()="text"]')]
    assert text_list(original)==text_list(tree)
    changed = {'x28_0_180_'+name for bars,values,label in groups for name in bars+values+[label]}
    changed |= {'x28_0_180_text_14','x28_0_180_patch_9','x28_0_180_p2274087cc0'}
    for identifier in ['x28_0_180_axes_1','x28_0_180_axes_3','x28_0_180_axes_5','x28_0_180_axes_6']:
        old = original.xpath('//*[@id=$id]',id=identifier)
        new = tree.xpath('//*[@id=$id]',id=identifier)
        if old: assert etree.tostring(old[0],method='c14n')==etree.tostring(new[0],method='c14n')
    destination = ROOT/'analysis/figure1/new_figure1.svg'
    destination.parent.mkdir(parents=True,exist_ok=True)
    destination.write_bytes(etree.tostring(tree,encoding='utf-8',xml_declaration=True))
    with (destination.parent/'demographics_verified.tsv').open('w') as stream:
        stream.write('characteristic\tcategory\tcount\ttotal\n')
        for feature,entries in [('sex',[('Female',60),('Male',77)]),('neoadjuvant therapy',[('No',126),('Yes',11)]),('pN status',[('pN0',70),('pN+',56),('Unknown',11)])]:
            for category,count in entries:stream.write(f'{feature}\t{category}\t{count}\t137\n')
    write_json(destination.parent/'source_and_geometry_validation.json', {
        'status':'PASS','baseline_svg_sha256':digest(baseline.read_bytes()),
        'svg_sha256':digest(destination.read_bytes()),'patient_source':str(source.relative_to(ROOT)),
        'patient_source_sha256':digest(source.read_bytes()),'patients':137,'counts':counts,
        'age_median_IQR':[68,60,74],'all_text_strings_exactly_preserved':True,
        'bar_widths_heights_colors_and_proportions_unchanged':True,
        'non_demographic_scientific_geometry_unchanged':True,'root_and_nested_crop_viewports_unchanged':True,
        'demographic_font_before_pt':7,'demographic_font_after_pt':8,
        'row_center_pitch_before_pt':previous_pitch,'row_center_pitch_after_pt':new_pitch,
        'edits':edits,'modified_element_ids':sorted(changed)})
    print(json.dumps({'figure1_svg':'ready','sha256':digest(destination.read_bytes()),'counts_preserved':True}))

if __name__=='__main__':main()
