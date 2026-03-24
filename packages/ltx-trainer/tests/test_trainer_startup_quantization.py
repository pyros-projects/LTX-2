from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
import unittest


MODULE_PATH = Path(__file__).resolve().parents[1] / "src" / "ltx_trainer" / "trainer.py"
sys.path.insert(0, str(MODULE_PATH.parents[1]))
SPEC = importlib.util.spec_from_file_location("ltx_trainer_trainer", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC is not None and SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


class TrainerStartupQuantizationTests(unittest.TestCase):
    def test_keeps_quantized_model_on_device_for_single_process_cuda(self) -> None:
        self.assertTrue(MODULE.should_keep_quantized_model_on_device(world_size="1", cuda_available=True))

    def test_does_not_keep_quantized_model_on_device_for_multi_process(self) -> None:
        self.assertFalse(MODULE.should_keep_quantized_model_on_device(world_size="2", cuda_available=True))

    def test_does_not_keep_quantized_model_on_device_without_cuda(self) -> None:
        self.assertFalse(MODULE.should_keep_quantized_model_on_device(world_size="1", cuda_available=False))


if __name__ == "__main__":
    unittest.main()
