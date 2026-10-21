#!/usr/bin/env python3
"""Register finalized main or supplementary artwork against the current inventory."""
from __future__ import annotations
import argparse, json, re, shutil
from pathlib import Path
from lxml import etree
from PIL import Image
from pypdf import PdfReader
from assemble import ROOT, digest, write_json
from revision_config import VERSION, expected_registrations

def relative_file(root, value):
    path=Path(value)
    if path.is_absolute() or '..' in path.parts or not (root/path).is_file():
        raise ValueError('Registered input must be an existing packaged relative file: '+str(value))
    return root/path

def owner_freeze_files(receipt):
    """Normalize existing owner receipt schemas without discarding bound inputs."""
    files={}
    def add(relative, checksum):
        if not isinstance(relative,str) or not re.fullmatch(r'[0-9a-f]{64}',str(checksum)):
            raise ValueError('Malformed owner freeze file/hash')
        if relative in files and files[relative]!=checksum:
            raise ValueError('Conflicting owner freeze hashes: '+relative)
        files[relative]=checksum
    source=receipt.get('files',{})
    if isinstance(source,list):
        for item in source:add(item['path'],item['sha256'])
    elif isinstance(source,dict):
        for relative,checksum in source.items():add(relative,checksum)
    else:raise ValueError('Malformed owner freeze files')
    for field in ('source_sha256','output_sha256'):
        for relative,checksum in receipt.get(field,{}).items():add(relative,checksum)
    for field in ('visual_verdict','validation','caption_mapping'):
        item=receipt.get(field)
        if isinstance(item,dict) and 'path' in item:add(item['path'],item['sha256'])
    return files

def prepare(root, kind, ordinal, path):
    spec=json.loads(path.read_text())
    if spec.get('document','manuscript')!=kind or spec.get('display_ordinal')!=ordinal:
        raise ValueError('Registration document/ordinal differs from requested figure')
    components=spec.get('baseline_components',[])
    if not components and not spec.get('new_analysis'):
        raise ValueError('Register baseline components or explicit new-analysis provenance')
    for component in components:
        if not isinstance(component.get('figure'),int) or component['figure']<1:
            raise ValueError('Baseline composition provenance requires positive figure ordinals')
    assets={extension:relative_file(root,spec[extension]) for extension in ('svg','png','pdf')}
    tree=etree.fromstring(assets['svg'].read_bytes())
    viewbox=[float(value) for value in tree.get('viewBox','').replace(',',' ').split()]
    if tree.tag!='{http://www.w3.org/2000/svg}svg' or not tree.get('width') or not tree.get('height') or len(viewbox)!=4 or viewbox[2]<=0 or viewbox[3]<=0:
        raise ValueError('Complete SVG root width/height and positive viewBox required')
    if tree.xpath('//*[local-name()="script"]'): raise ValueError('SVG must be script-free')
    for node in tree.iter():
        for name,value in node.attrib.items():
            if name.rsplit('}',1)[-1]=='href' and not value.startswith(('#','data:')):
                raise ValueError('External SVG dependency: '+value)
    with Image.open(assets['png']) as image: image.verify()
    if len(PdfReader(assets['pdf']).pages)!=1: raise ValueError('Figure PDF requires one complete viewport')
    if not spec.get('source_tables') or not spec.get('plotting_code'): raise ValueError('Source tables and plotting code must be bound')
    dependencies=[relative_file(root,value) for value in spec['source_tables']+spec['plotting_code']]
    for relative,checksum in spec.get('source_sha256',{}).items():
        source=relative_file(root,relative)
        if digest(source.read_bytes())!=checksum: raise ValueError('Owner-declared source hash differs: '+relative)
        dependencies.append(source)
    review=relative_file(root,spec['visual_review']); verdict=json.loads(review.read_text())
    state=verdict.get('status',verdict.get('verdict'))
    if isinstance(state,dict): state=state.get('verdict',state.get('status'))
    checksum=digest(assets['svg'].read_bytes()); bound=checksum in json.dumps(verdict)
    if spec.get('visual_freeze') or not bound:
        freeze=relative_file(root,spec['visual_freeze']) if spec.get('visual_freeze') else path.parent/'final_freeze.json'
        if freeze.is_file():
            receipt=json.loads(freeze.read_text()); files=owner_freeze_files(receipt)
            required=[*assets.values(),review] if spec.get('visual_freeze') else [assets['svg'],review]
            freeze_bound=(receipt.get('status') in {'FROZEN','PASS'} and all(
                files.get(str(item.relative_to(root)))==digest(item.read_bytes()) for item in required) and all(
                digest(relative_file(root,relative).read_bytes())==checksum for relative,checksum in files.items()))
            if spec.get('visual_freeze') and not freeze_bound:
                raise ValueError('Explicit owner freeze must bind current assets and visual review')
            if freeze_bound: dependencies.extend([freeze,*[root/relative for relative in files]])
            bound=bound or freeze_bound
    if str(state).upper()!='PASS' or not bound: raise ValueError('Current visual PASS must bind the exact SVG hash')
    classes=spec.get('section_class','figure-unit tall new-figure').split()
    if not {'figure-unit','new-figure'}.issubset(classes) or set(classes)-{'figure-unit','new-figure','tall','compact','caption-inline'}:
        raise ValueError('Unsupported figure-section classes')
    folder,prefix=('main','Figure') if kind=='manuscript' else ('supplement','FigureS')
    aliases={extension:[f'figures/{folder}/{prefix}{ordinal}.{extension}',f'manuscript/Figure_{"PNG" if extension=="png" else "PDF"}/{prefix}{ordinal}{"" if extension=="png" else "_full"}.{extension}'] for extension in assets}
    hashes={str(item.relative_to(root)):digest(item.read_bytes()) for item in [*dependencies,path,review,*assets.values()]}
    return spec,assets,aliases,hashes,viewbox,' '.join(classes)

def inherited_panel_exports(root, kind, ordinal):
    """Retain obsolete baseline crops as history rather than current aliases."""
    fixture=root/'reproducibility/frozen_v19_baseline/baseline_fixture_manifest.json'
    if not fixture.is_file(): return []
    original=json.loads(fixture.read_text())['asset_sha256']
    prefix='Figure' if kind=='manuscript' else 'FigureS'
    pattern=re.compile(rf'{prefix}{ordinal}_[A-Z]\.(?:svg|pdf)')
    retired=[]
    for relative,checksum in original.items():
        if not relative.startswith('manuscript/Figure_PDF/') or not pattern.fullmatch(Path(relative).name): continue
        source=root/relative
        if not source.exists(): continue
        if digest(source.read_bytes())!=checksum:
            raise ValueError('Unregistered panel export differs from inherited crop: '+relative)
        target_relative='reproducibility/historical_figure_exports/v19/'+relative
        target=root/target_relative
        if target.exists() and digest(target.read_bytes())!=checksum:
            raise ValueError('Historical panel preservation target changed: '+target_relative)
        retired.append({'previous_alias':relative,'historical_alias':target_relative,'sha256':checksum})
    return retired

def caption_refresh(root, ledger, inputs, proof_path):
    """Authorize only the root-approved S2 caption/associated reviewed receipt refresh."""
    if not proof_path:return {},None
    proof_path=relative_file(root,str(proof_path.relative_to(root)))
    proof=json.loads(proof_path.read_text())
    changed=[item for item in proof.get('changes',[]) if item.get('caption_changed')]
    if proof.get('status')!='FROZEN' or proof.get('all_six_captions_final') is not True or len(changed)!=1 or changed[0].get('figure')!='S2' or changed[0].get('artwork_unchanged') is not True:
        raise ValueError('Caption refresh requires the final unchanged-artwork S2 provenance proof')
    key=('supplement',2)
    if key not in inputs or '2' not in ledger['documents']['supplement']:
        raise ValueError('Caption proof requires explicit re-registration of the existing S2')
    previous=ledger['documents']['supplement']['2']; hashes=previous['source_and_output_sha256']
    item=changed[0]; caption=item['caption']; current=digest(relative_file(root,caption).read_bytes())
    if current!=item['final_caption_sha256']:
        raise ValueError('Final caption differs from approved provenance')
    if hashes.get(caption)==current:return {},None
    if hashes.get(caption)!=item['previous_caption_sha256']:
        raise ValueError('Prior caption differs from approved provenance')
    spec=json.loads(inputs[key].read_text()); old_spec=json.loads(relative_file(root,previous['registration_path']).read_text())
    if {k:v for k,v in spec.items() if k!='visual_freeze'}!={k:v for k,v in old_spec.items() if k!='visual_freeze'} or caption not in spec['source_tables'] or not spec.get('visual_freeze'):
        raise ValueError('Caption refresh must preserve the registration and supply an exact owner freeze')
    freeze=relative_file(root,spec['visual_freeze']); receipt=json.loads(freeze.read_text()); files=owner_freeze_files(receipt)
    review=previous['visual_review']; final_review=digest(relative_file(root,review).read_bytes())
    if receipt.get('status') not in {'FROZEN','PASS'} or files.get(caption)!=current or files.get(review)!=final_review:
        raise ValueError('Caption refresh owner freeze must bind final caption and associated visual review')
    allowed={caption:(hashes[caption],current),review:(hashes[review],final_review)}
    record={'figure':'S2','authority':'Root-approved final caption synchronization; artwork, code and numerical inputs unchanged',
        'provenance_path':str(proof_path.relative_to(root)),'provenance_sha256':digest(proof_path.read_bytes()),
        'caption':caption,'caption_previous_sha256':hashes[caption],'caption_final_sha256':current,
        'visual_review':review,'visual_review_previous_sha256':hashes[review],'visual_review_final_sha256':final_review,
        'owner_freeze':spec['visual_freeze'],'owner_freeze_sha256':digest(freeze.read_bytes())}
    return allowed,record

def register(root, inputs, *, caption_proof=None):
    expected=expected_registrations(root); path=root/'audit/figure_registration.json'
    ledger=json.loads(path.read_text()) if path.exists() else {'version':VERSION,'documents':{kind:{}for kind in expected}}
    if ledger.get('version')!=VERSION or set(ledger.get('documents',{}))!=set(expected): raise ValueError('Existing ledger belongs to a different version/inventory')
    allowed,refresh=caption_refresh(root,ledger,inputs,caption_proof)
    for kind,records in ledger['documents'].items():
        if not {int(key) for key in records}.issubset(expected[kind]): raise ValueError('Registered artwork removed from the inventory')
        for item in records.values():
            for relative,checksum in item['source_and_output_sha256'].items():
                current=digest((root/relative).read_bytes()) if (root/relative).is_file() else None
                if current!=checksum and (kind!='supplement' or item.get('display_ordinal')!=2 or allowed.get(relative)!=(checksum,current)):
                    raise ValueError('Previously frozen artwork/source changed: '+relative)
    if not inputs or any(ordinal not in expected[kind] for kind,ordinal in inputs): raise ValueError('Only explicitly registered inventory entries may be installed')
    # Preflight all dependencies before copying any aliases.
    prepared={(kind,ordinal):prepare(root,kind,ordinal,spec) for (kind,ordinal),spec in inputs.items()}
    retired={(kind,ordinal):inherited_panel_exports(root,kind,ordinal) for kind,ordinal in inputs}
    if refresh:
        prior_bytes=path.read_bytes(); prior_sha=digest(prior_bytes)
        archive=root/'audit/figure_registration_history'/f'{prior_sha}.json'
        archive.parent.mkdir(parents=True,exist_ok=True)
        if archive.exists() and archive.read_bytes()!=prior_bytes:raise ValueError('Prior registration archive differs')
        archive.write_bytes(prior_bytes)
        refresh.update({'prior_registration_sha256':prior_sha,'prior_registration_archive':str(archive.relative_to(root))})
        ledger.setdefault('caption_refresh_history',[]).append(refresh)
        prepared['supplement',2][3][refresh['provenance_path']]=refresh['provenance_sha256']
    for (kind,ordinal),(spec,assets,aliases,hashes,viewbox,classes) in prepared.items():
        for extension,targets in aliases.items():
            for relative in targets:
                target=root/relative; target.parent.mkdir(parents=True,exist_ok=True)
                if not target.exists() or digest(target.read_bytes())!=digest(assets[extension].read_bytes()): shutil.copy2(assets[extension],target)
                hashes[relative]=digest(target.read_bytes())
        for entry in retired[kind,ordinal]:
            source=root/entry['previous_alias']; target=root/entry['historical_alias']
            target.parent.mkdir(parents=True,exist_ok=True)
            if target.exists(): source.unlink()
            else: source.rename(target)
            hashes[entry['historical_alias']]=digest(target.read_bytes())
        prior=ledger['documents'][kind].get(str(ordinal),{}).get('retired_baseline_panel_exports',[])
        history={entry['historical_alias']:entry for entry in [*prior,*retired[kind,ordinal]]}
        for relative,entry in history.items(): hashes[relative]=entry['sha256']
        ledger['documents'][kind][str(ordinal)]={'display_ordinal':ordinal,'document':kind,'registration_path':str(inputs[kind,ordinal].relative_to(root)),
            'baseline_components':spec.get('baseline_components',[]),'new_analysis':spec.get('new_analysis'),'viewBox':viewbox,'section_class':classes,
            'source_and_output_sha256':hashes,'visual_review':spec['visual_review'],'full_viewport_embedded_without_reserialization':True,
            'retired_baseline_panel_exports':list(history.values())}
    complete=all({int(key)for key in ledger['documents'][kind]}==ordinals for kind,ordinals in expected.items())
    ledger.update({'status':'REGISTERED_HASH_BOUND'if complete else 'PARTIALLY_REGISTERED_HASH_BOUND','registry_sha256':digest((root/'audit/figure_registry.json').read_bytes())})
    write_json(path,ledger); return ledger

def main():
    cli=argparse.ArgumentParser(description=__doc__);cli.add_argument('--root',type=Path,default=ROOT)
    cli.add_argument('--main',action='append',default=[],metavar='ORDINAL=RELATIVE_JSON')
    cli.add_argument('--supplement',action='append',default=[],metavar='ORDINAL=RELATIVE_JSON')
    cli.add_argument('--refresh-caption-proof',type=Path,help='Root-approved final S2 caption provenance; permits only caption and its bound visual-review refresh')
    args=cli.parse_args();root=args.root.resolve();inputs={}
    for kind,items in [('manuscript',args.main),('supplement',args.supplement)]:
        for value in items:
            ordinal,relative=value.split('=',1);key=kind,int(ordinal)
            if key in inputs:cli.error('Duplicate figure registration')
            spec=Path(relative);inputs[key]=spec.resolve()if spec.is_absolute()else root/spec
            if not inputs[key].is_relative_to(root):cli.error('Registration JSON must be packaged inside this revision')
    if not inputs:cli.error('Provide at least one finalized main or supplement registration')
    proof=args.refresh_caption_proof
    if proof:proof=proof.resolve()if proof.is_absolute()else root/proof
    ledger=register(root,inputs,caption_proof=proof)
    print(json.dumps({'figure_registration':'passed','figures':{kind:sorted(int(key)for key in records)for kind,records in ledger['documents'].items()},'production_assembly_ready':ledger['status']=='REGISTERED_HASH_BOUND'}))

if __name__=='__main__':main()
