from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
import unittest


MODULE_PATH = Path(__file__).resolve().parents[1] / "src" / "ltx_trainer" / "config.py"
sys.path.insert(0, str(MODULE_PATH.parents[1]))
SPEC = importlib.util.spec_from_file_location("ltx_trainer_config", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC is not None and SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


class BlockSwapConfigTests(unittest.TestCase):
    def test_acceleration_config_defaults_block_swap_to_disabled(self) -> None:
        config = MODULE.AccelerationConfig()

        self.assertEqual(config.blocks_to_swap, 0)
        self.assertFalse(config.use_pinned_memory_for_block_swap)


if __name__ == "__main__":
    unittest.main()
