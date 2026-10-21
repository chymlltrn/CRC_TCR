"""Data-free checks for the release adapter; not a scientific rerun."""
import ast
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import release_config as config

class Interfaces(unittest.TestCase):
    def setUp(self):
        self.old=os.environ.get('CRC_TCR_CONFIG')
        self.tmp=tempfile.TemporaryDirectory()
        self.path=Path(self.tmp.name)/'local.json'
        self.path.write_text(json.dumps({'paths':{'example':str(Path(self.tmp.name)/'input')},'private_constants':{'example':'synthetic'}}))
        os.environ['CRC_TCR_CONFIG']=str(self.path)
        config.load_config.cache_clear()
    def tearDown(self):
        if self.old is None:os.environ.pop('CRC_TCR_CONFIG',None)
        else:os.environ['CRC_TCR_CONFIG']=self.old
        config.load_config.cache_clear();self.tmp.cleanup()
    def test_python_syntax(self):
        for path in ROOT.rglob('*.py'):
            with self.subTest(path=path.relative_to(ROOT)):compile(path.read_text(),str(path),'exec')
    def test_manifest_hashes(self):
        for row in json.loads((ROOT/'source_manifest.json').read_text()):
            self.assertEqual(hashlib.sha256((ROOT/row['release_path']).read_bytes()).hexdigest(),row['release_sha256'])
    def test_schema_contains_no_source_values(self):
        ex=json.loads((ROOT/'config.example.json').read_text())
        self.assertTrue(all(value=='' or value==[] for values in ex.values() for value in values.values()))
    def test_private_value_interface(self):
        self.assertEqual(config.private_value('example'),'synthetic')
        with self.assertRaises(ValueError):config.private_value('missing')
    def test_path_interface(self):
        self.assertTrue(config.release_path('example').is_absolute())
        with self.assertRaises(ValueError):config.release_path('missing')
    def test_reject_data_in_repository(self):
        self.path.write_text(json.dumps({'paths':{'example':str(ROOT/'data')},'private_constants':{}}))
        config.load_config.cache_clear()
        with self.assertRaises(ValueError):config.release_path('example')
    def test_reject_private_config_in_repository(self):
        os.environ['CRC_TCR_CONFIG']=str(ROOT/'config.example.json');config.load_config.cache_clear()
        with self.assertRaises(ValueError):config.load_config()
    def test_listing_does_not_need_data_or_config(self):
        env=os.environ.copy();env.pop('CRC_TCR_CONFIG',None)
        p=subprocess.run([sys.executable,str(ROOT/'run.py'),'--list'],capture_output=True,text=True,env=env)
        self.assertEqual(p.returncode,0,p.stderr)
        self.assertIn('bulk_sc_match_ingest',p.stdout)

if __name__=='__main__':unittest.main()
