"""The package must work as an installed library, on any machine."""
from __future__ import annotations

import importlib
import pkgutil
import subprocess
import sys
import unittest

import organic_infill_generator


class TestPackage(unittest.TestCase):
    def test_every_module_imports(self):
        for module in pkgutil.walk_packages(organic_infill_generator.__path__, "organic_infill_generator."):
            if module.name == "organic_infill_generator.cadquery_service.cad_converter":
                continue  # needs the optional CAD backend
            if module.name == "organic_infill_generator.cadquery_service":
                continue
            with self.subTest(module=module.name):
                importlib.import_module(module.name)

    def test_the_tool_loads_without_the_cad_backend(self):
        code = (
            "import sys, organic_infill_generator\n"
            "organic_infill_generator.run_pipeline\n"
            "assert 'cadquery' not in sys.modules and 'OCP' not in sys.modules\n"
        )
        subprocess.run([sys.executable, "-c", code], check=True)

    def test_importing_does_not_choose_a_plot_backend_or_configure_logging(self):
        code = (
            "import logging\n"
            "before = list(logging.getLogger().handlers)\n"
            "import organic_infill_generator.pipeline\n"
            "assert logging.getLogger().handlers == before\n"
            "import sys\n"
            "assert 'matplotlib.pyplot' not in sys.modules\n"
        )
        subprocess.run([sys.executable, "-c", code], check=True)


if __name__ == "__main__":
    unittest.main()
