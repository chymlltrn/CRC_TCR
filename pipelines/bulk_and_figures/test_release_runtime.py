"""Small configuration-boundary tests; never import or execute scientific pipelines."""
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import release_runtime as runtime


class ConfigurationTests(unittest.TestCase):
    def setUp(self):
        runtime._load_config.cache_clear()

    def test_explicit_path_and_separate_private_file(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / "private.json").write_text(json.dumps({"private_literals": {"example": "synthetic input"}}))
            config = root / "config.json"
            config.write_text(json.dumps({"schema_version": 1, "paths": {"output": str(root / "output")}, "private_literals_file": "private.json"}))
            with patch.dict(os.environ, {"CRC_TCR_BULK_CONFIG": str(config)}):
                self.assertEqual(runtime.release_path("output"), str(root / "output"))
                self.assertEqual(runtime.private_literal("example"), "synthetic input")
                with self.assertRaises(KeyError):
                    runtime.private_literal("missing")

    def test_no_implicit_private_defaults(self):
        with patch.dict(os.environ, {"CRC_TCR_BULK_CONFIG": ""}):
            with self.assertRaises(RuntimeError):
                runtime.private_literal("example")

    def test_relative_path_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            config = Path(folder) / "config.json"
            config.write_text(json.dumps({"schema_version": 1, "paths": {"output": "relative"}, "private_literals": {}}))
            with patch.dict(os.environ, {"CRC_TCR_BULK_CONFIG": str(config)}):
                with self.assertRaises(ValueError):
                    runtime.release_path("output")

    def test_checkout_configuration_rejected_before_read(self):
        with self.assertRaises(ValueError):
            runtime._load_config(str(Path(runtime.__file__)))

    def test_checkout_data_path_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            config = Path(folder) / "config.json"
            config.write_text(json.dumps({"schema_version": 1, "paths": {"output": str(runtime.REPOSITORY_ROOT / "results")}, "private_literals": {}}))
            with patch.dict(os.environ, {"CRC_TCR_BULK_CONFIG": str(config)}):
                with self.assertRaises(ValueError):
                    runtime.release_path("output")

    def test_checkout_private_file_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            config = Path(folder) / "config.json"
            config.write_text(json.dumps({"schema_version": 1, "paths": {}, "private_literals_file": str(Path(runtime.__file__))}))
            with patch.dict(os.environ, {"CRC_TCR_BULK_CONFIG": str(config)}):
                with self.assertRaises(ValueError):
                    runtime.private_literal("example")


if __name__ == "__main__":
    unittest.main()
