from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
import unittest

import torch


MODULE_PATH = Path(__file__).resolve().parents[1] / "src" / "ltx_trainer" / "trainer.py"
sys.path.insert(0, str(MODULE_PATH.parents[1]))
SPEC = importlib.util.spec_from_file_location("ltx_trainer_trainer", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC is not None and SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


class TinyModule(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear = torch.nn.Linear(4, 4)
        self.register_buffer("cpu_buf", torch.ones(2))


class TrainerDebugHelperTests(unittest.TestCase):
    def test_collect_module_device_summary_reports_cpu_params_and_buffers(self) -> None:
        summary = MODULE.collect_module_device_summary(TinyModule())

        self.assertIn("cpu", summary["parameter_devices"])
        self.assertIn("cpu", summary["buffer_devices"])
        self.assertIn("linear", summary["cpu_parameter_modules"])
        self.assertIn("", summary["cpu_buffer_modules"])

    def test_compute_step_timing_scales_microstep_to_optimization_step(self) -> None:
        microstep_seconds, optimization_step_seconds = MODULE.compute_step_timing(
            elapsed_seconds=3.0,
            gradient_accumulation_steps=4,
        )

        self.assertEqual(microstep_seconds, 3.0)
        self.assertEqual(optimization_step_seconds, 12.0)

    def test_collect_expected_swap_cpu_module_prefixes_returns_last_swapped_blocks(self) -> None:
        class FakeBaseTransformer(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.blocks_to_swap = 2
                self.transformer_blocks = torch.nn.ModuleList(
                    [
                        torch.nn.Linear(4, 4),
                        torch.nn.Linear(4, 4),
                        torch.nn.Linear(4, 4),
                    ]
                )

        prefixes = MODULE.collect_expected_swap_cpu_module_prefixes(FakeBaseTransformer())

        self.assertEqual(prefixes, {"transformer_blocks.1", "transformer_blocks.2"})

    def test_filter_unexpected_cpu_modules_excludes_expected_swapped_blocks(self) -> None:
        cpu_modules = [
            "transformer_blocks.1.attn1",
            "transformer_blocks.2.ff",
            "patchify_proj",
            "norm_out",
        ]
        expected_prefixes = {"transformer_blocks.1", "transformer_blocks.2"}

        unexpected = MODULE.filter_unexpected_cpu_modules(cpu_modules, expected_prefixes)

        self.assertEqual(unexpected, ["patchify_proj", "norm_out"])


if __name__ == "__main__":
    unittest.main()
