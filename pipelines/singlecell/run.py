#!/usr/bin/env python3
"""Dispatch a selected release script; no data are bundled or auto-downloaded."""
import argparse
import json
from pathlib import Path
import runpy
import sys
from release_config import load_config, release_path, private_value

ROOT = Path(__file__).resolve().parent

def entries():
    records = json.loads((ROOT/'source_manifest.json').read_text())
    scripts=[x['release_path'] for x in records if x['release_path'].endswith('.py')]
    names=[Path(path).stem for path in scripts]
    if len(names)!=len(set(names)):
        raise ValueError('Duplicate entry names in source_manifest.json; resolve before dispatch.')
    return dict(zip(names,scripts))

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--list',action='store_true')
    parser.add_argument('--check-config',action='store_true')
    parser.add_argument('entry',nargs='?',choices=sorted(entries()))
    parser.add_argument('arguments',nargs=argparse.REMAINDER)
    args=parser.parse_args()
    if args.list:
        print('\n'.join(f'{key}\t{value}' for key,value in sorted(entries().items())))
        return
    if args.check_config:
        config=load_config()
        schema=json.loads((ROOT/'config.example.json').read_text())
        for key in schema['paths']:release_path(key)
        for key in schema['private_constants']:private_value(key)
        print('Configuration interface PASS; scientific inputs have not been read or verified.')
        return
    if not args.entry:parser.error('choose an entry, --list, or --check-config')
    target=ROOT/entries()[args.entry]
    arguments=args.arguments[1:] if args.arguments[:1]==['--'] else args.arguments
    sys.path.insert(0,str(target.parent))
    sys.argv=[str(target)]+arguments
    runpy.run_path(str(target),run_name='__main__')

if __name__=='__main__':main()
