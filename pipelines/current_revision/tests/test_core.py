"""No private records: exact rank enumeration and stream detection fixtures."""
import ast
from pathlib import Path
import gzip
import itertools
import json
import math
import subprocess
import tempfile
import unittest
import numpy as np
from scipy.stats import rankdata

CODE = Path(__file__).resolve().parents[1]

class CoreTests(unittest.TestCase):
    def test_exact_rank_ties(self):
        tree = ast.parse((CODE / 'qpcr/build_qpcr_revision.py').read_text())
        fn = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == 'exact_rank_test')
        ns = {'np': np, 'rankdata': rankdata, 'combinations': itertools.combinations, 'math': math}
        exec(compile(ast.Module(body=[fn], type_ignores=[]), '<extracted real implementation>', 'exec'), ns)
        result = ns['exact_rank_test'](np.array([2., 3.]), np.array([0., 0.]))
        self.assertEqual(result['U_CRC'], 4)
        self.assertEqual(result['total_assignments'], 6)
        self.assertAlmostEqual(result['P_two_sided'], 1 / 3)

    def test_stream_counts_and_detected_cells(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            binary = root / 'stream'
            subprocess.run(['g++', '-O3', '-std=c++17', str(CODE / 'signatures/stream_detection.cpp'), '-lz', '-o', str(binary)], check=True)
            with gzip.open(root / 'counts.mtx.gz', 'wt') as f:
                f.write('%%MatrixMarket matrix coordinate integer general\n% synthetic fixture\n3 4 5\n1 1 2\n2 1 1\n1 2 4\n3 3 9\n2 4 7\n')
            np.array([0, 1, 2], dtype=np.int32).tofile(root / 'genes.bin')
            np.array([0, 0, 1, -1], dtype=np.int32).tofile(root / 'cells.bin')
            subprocess.run([str(binary), str(root / 'counts.mtx.gz'), str(root / 'genes.bin'), str(root / 'cells.bin'), '3', '2', str(root / 'raw')], check=True)
            np.testing.assert_array_equal(np.fromfile(root / 'raw_counts.bin', dtype='<i8').reshape(2, 3), [[6, 1, 0], [0, 0, 9]])
            np.testing.assert_array_equal(np.fromfile(root / 'raw_detected.bin', dtype='<i4').reshape(2, 3), [[2, 1, 0], [0, 0, 1]])
            receipt = json.loads((root / 'raw_stream.json').read_text())
            self.assertEqual(receipt['raw_nnz'], 5)
            self.assertEqual(receipt['selected_total_counts'], 16)

if __name__ == '__main__':
    unittest.main()
