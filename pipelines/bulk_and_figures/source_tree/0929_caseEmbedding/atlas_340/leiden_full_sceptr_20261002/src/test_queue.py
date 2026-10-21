#!/usr/bin/env python3
"""Small stdlib-only tests of serialization and predecessor/resource gates."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

spec = importlib.util.spec_from_file_location('sceptr_queue', Path(__file__).with_name('queue_run.py'))
q = importlib.util.module_from_spec(spec)
spec.loader.exec_module(q)


class QueueGateTests(unittest.TestCase):
    def test_missing_state_waits(self):
        with tempfile.TemporaryDirectory() as d:
            self.assertFalse(q.predecessor_status(Path(d))[0])

    def test_running_and_failed_never_start(self):
        with tempfile.TemporaryDirectory() as d:
            b = Path(d)
            (b / 'audit').mkdir()
            for status in ['RUNNING', 'FAILED', 'STOPPED']:
                q.dump(b / 'audit/runner_state.json', {'status': status})
                self.assertFalse(q.predecessor_status(b)[0])

    def test_complete_requires_final_acceptance(self):
        with tempfile.TemporaryDirectory() as d:
            b = Path(d)
            (b / 'audit').mkdir()
            (b / 'report').mkdir()
            q.dump(b / 'audit/runner_state.json', {'status': 'COMPLETE'})
            self.assertFalse(q.predecessor_status(b)[0])
            q.dump(b / 'report/full_cluster_counts_by_resolution.json', {'status': 'FAIL'})
            self.assertFalse(q.predecessor_status(b)[0])
            q.dump(b / 'report/full_cluster_counts_by_resolution.json', {'status': 'PASS'})
            (b / 'report/FINAL_COUNTS.md').write_text('accepted')
            with mock.patch.object(q, 'running_scripts', return_value=[123]):
                self.assertFalse(q.predecessor_status(b)[0])
            with mock.patch.object(q, 'running_scripts', return_value=[]):
                self.assertEqual(q.predecessor_status(b), (True, 'READY'))

    def test_process_match_is_exact_directory(self):
        with tempfile.TemporaryDirectory() as d:
            b = Path(d)
            p = b / '123'
            p.mkdir()
            (p / 'cmdline').write_bytes(b'python3\0/reference/src/serial_run.py\0')
            self.assertEqual(q.running_scripts(Path('/reference'), b), [123])
            self.assertEqual(q.running_scripts(Path('/reference_sceptr'), b), [])

    def test_hash_gate_rejects_edited_sources(self):
        with tempfile.TemporaryDirectory() as d:
            b = Path(d)
            (b / 'src').mkdir()
            f = b / 'src/example.py'
            f.write_text('original')
            c = {'execution_source_sha256': {'example.py': q.sha(f)}}
            with mock.patch.object(q, 'ROOT', b):
                q.source_gate(c)
                f.write_text('changed')
                with self.assertRaises(RuntimeError):
                    q.source_gate(c)

    def test_all_caches_and_vendor_are_in_new_root(self):
        with mock.patch.object(q, 'ROOT', Path('/new_sceptr')):
            e = q.environment(4)
            for k in ['PYTHONPATH', 'TMPDIR', 'XDG_CACHE_HOME', 'MPLCONFIGDIR']:
                self.assertTrue(e[k].startswith('/new_sceptr/'))
            self.assertEqual(e['ATLAS_CPU_THREADS'], '4')
            self.assertEqual(e['PYTHONDONTWRITEBYTECODE'], '1')


if __name__ == '__main__':
    unittest.main()
