"""Integration checks for the installed runtime dependency set."""

import importlib
import unittest
from pathlib import Path

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class ImportTests(unittest.TestCase):
    def test_project_modules_import(self):
        modules = (
            "src.data",
            "src.retrieval",
            "src.verification",
            "src.generator",
            "src.revision",
            "src.evaluation",
            "src.pipeline",
        )
        for module_name in modules:
            with self.subTest(module=module_name):
                importlib.import_module(module_name)

    def test_configuration_loads(self):
        with (PROJECT_ROOT / "config/config.yaml").open(encoding="utf-8") as stream:
            config = yaml.safe_load(stream)

        self.assertEqual(config["experiments"]["device"], "auto")
        self.assertAlmostEqual(config["retrieval"]["fusion"]["dense_weight"], 0.6)


if __name__ == "__main__":
    unittest.main()
