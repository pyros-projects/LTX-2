from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
import unittest


MODULE_PATH = Path(__file__).resolve().parents[1] / "src" / "ltx_trainer" / "quantization.py"
sys.path.insert(0, str(MODULE_PATH.parents[1]))
SPEC = importlib.util.spec_from_file_location("ltx_quantization", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC is not None and SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


class QuantizationRestoreDecisionTests(unittest.TestCase):
    def test_restores_to_original_device_by_default(self) -> None:
        self.assertTrue(MODULE.should_restore_original_device("cpu", "cuda", keep_quantized_model_on_device=False))

    def test_keeps_quantized_model_on_target_device_when_requested(self) -> None:
        self.assertFalse(MODULE.should_restore_original_device("cpu", "cuda", keep_quantized_model_on_device=True))

    def test_offloads_blocks_when_not_keeping_model_on_device(self) -> None:
        self.assertTrue(MODULE.should_offload_after_quantization(False))

    def test_keeps_blocks_on_device_when_requested(self) -> None:
        self.assertFalse(MODULE.should_offload_after_quantization(True))

    def test_skips_adaln_root_modules_for_quantization(self) -> None:
        self.assertTrue(MODULE.should_skip_root_module_quantization("adaln_single"))
        self.assertTrue(MODULE.should_skip_root_module_quantization("prompt_adaln_single"))

    def test_keeps_attention_blocks_eligible_for_quantization(self) -> None:
        self.assertFalse(MODULE.should_skip_root_module_quantization("transformer_blocks"))


if __name__ == "__main__":
    unittest.main()
