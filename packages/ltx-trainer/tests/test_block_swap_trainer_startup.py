from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
import unittest

from accelerate import DistributedType
import torch


MODULE_PATH = Path(__file__).resolve().parents[1] / "src" / "ltx_trainer" / "trainer.py"
sys.path.insert(0, str(MODULE_PATH.parents[1]))
SPEC = importlib.util.spec_from_file_location("ltx_trainer_trainer", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC is not None and SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


class FakeBaseTransformer:
    def __init__(self, blocks_to_swap: int = 0) -> None:
        self.blocks_to_swap = blocks_to_swap
        self.calls: list[tuple[str, object]] = []
        self.dtype = torch.bfloat16

    def enable_block_swap(
        self,
        blocks_to_swap: int,
        device: torch.device,
        supports_backward: bool,
        use_pinned_memory: bool = False,
    ) -> None:
        self.calls.append(
            (
                "enable_block_swap",
                blocks_to_swap,
                str(device),
                supports_backward,
                use_pinned_memory,
            )
        )
        self.blocks_to_swap = blocks_to_swap

    def set_gradient_checkpointing(self, enable: bool) -> None:
        self.calls.append(("set_gradient_checkpointing", enable))

    def move_to_device_except_swap_blocks(self, device: torch.device) -> None:
        self.calls.append(("move_to_device_except_swap_blocks", str(device)))

    def prepare_block_swap_before_forward(self) -> None:
        self.calls.append(("prepare_block_swap_before_forward", None))

    def to(self, *args, **kwargs) -> "FakeBaseTransformer":
        self.calls.append(("to", args, kwargs))
        return self


class FakeWrappedTransformer:
    def __init__(self, base_model: FakeBaseTransformer) -> None:
        self._base_model = base_model

    def get_base_model(self) -> FakeBaseTransformer:
        return self._base_model

    def to(self, *args, **kwargs) -> "FakeWrappedTransformer":
        self._base_model.to(*args, **kwargs)
        return self


class FakeFrozenModule:
    def __init__(self) -> None:
        self.calls: list[tuple[str, object]] = []

    def to(self, device: object = None, **kwargs) -> "FakeFrozenModule":
        self.calls.append(("to", device, kwargs))
        return self


class FakeAccelerator:
    def __init__(self) -> None:
        self.distributed_type = DistributedType.NO
        self.device = torch.device("cpu")
        self.prepare_calls: list[object] = []

    def prepare(self, value: object) -> object:
        self.prepare_calls.append(value)
        return value


class BlockSwapTrainerStartupTests(unittest.TestCase):
    def test_should_enable_block_swap_only_for_single_gpu_lora_runs(self) -> None:
        self.assertTrue(MODULE.should_enable_block_swap(DistributedType.NO, "lora", 2))
        self.assertFalse(MODULE.should_enable_block_swap(DistributedType.NO, "lora", 0))
        self.assertFalse(MODULE.should_enable_block_swap(DistributedType.NO, "full", 2))
        self.assertFalse(MODULE.should_enable_block_swap(DistributedType.FSDP, "lora", 2))

    def test_enable_block_swap_configures_base_transformer_for_training(self) -> None:
        base_model = FakeBaseTransformer()
        wrapped_model = FakeWrappedTransformer(base_model)

        MODULE.enable_block_swap_for_training(
            wrapped_model,
            device=torch.device("cpu"),
            blocks_to_swap=3,
            use_pinned_memory=True,
        )

        self.assertEqual(
            base_model.calls,
            [
                ("enable_block_swap", 3, "cpu", True, True),
                ("move_to_device_except_swap_blocks", "cpu"),
                ("prepare_block_swap_before_forward", None),
            ],
        )

    def test_restore_block_swap_residency_reapplies_device_layout(self) -> None:
        base_model = FakeBaseTransformer(blocks_to_swap=2)
        wrapped_model = FakeWrappedTransformer(base_model)

        MODULE.restore_block_swap_residency(wrapped_model, torch.device("cpu"))

        self.assertEqual(
            base_model.calls,
            [
                ("move_to_device_except_swap_blocks", "cpu"),
                ("prepare_block_swap_before_forward", None),
            ],
        )

    def test_prepare_models_for_training_enables_swap_for_non_quantized_single_gpu_run(self) -> None:
        base_model = FakeBaseTransformer()
        wrapped_model = FakeWrappedTransformer(base_model)
        trainer = MODULE.LtxvTrainer.__new__(MODULE.LtxvTrainer)
        trainer._accelerator = FakeAccelerator()
        trainer._config = type(
            "Config",
            (),
            {
                "model": type("ModelCfg", (), {"training_mode": "lora"})(),
                "optimization": type("OptCfg", (), {"enable_gradient_checkpointing": True})(),
                "acceleration": type(
                    "AccelCfg",
                    (),
                    {
                        "quantization": None,
                        "blocks_to_swap": 2,
                        "use_pinned_memory_for_block_swap": False,
                    },
                )(),
            },
        )()
        trainer._transformer = wrapped_model
        trainer._vae_decoder = FakeFrozenModule()
        trainer._vae_encoder = FakeFrozenModule()
        trainer._log_transformer_device_summary = lambda stage: None

        trainer._prepare_models_for_training()

        self.assertEqual(trainer._accelerator.prepare_calls, [wrapped_model])
        self.assertEqual(
            base_model.calls,
            [
                ("set_gradient_checkpointing", True),
                ("enable_block_swap", 2, "cpu", True, False),
                ("move_to_device_except_swap_blocks", "cpu"),
                ("prepare_block_swap_before_forward", None),
                ("move_to_device_except_swap_blocks", "cpu"),
                ("prepare_block_swap_before_forward", None),
            ],
        )

    def test_prepare_models_for_training_preserves_quantized_swap_path(self) -> None:
        base_model = FakeBaseTransformer()
        wrapped_model = FakeWrappedTransformer(base_model)
        trainer = MODULE.LtxvTrainer.__new__(MODULE.LtxvTrainer)
        trainer._accelerator = FakeAccelerator()
        trainer._config = type(
            "Config",
            (),
            {
                "model": type("ModelCfg", (), {"training_mode": "lora"})(),
                "optimization": type("OptCfg", (), {"enable_gradient_checkpointing": False})(),
                "acceleration": type(
                    "AccelCfg",
                    (),
                    {
                        "quantization": "int8-quanto",
                        "blocks_to_swap": 1,
                        "use_pinned_memory_for_block_swap": True,
                    },
                )(),
            },
        )()
        trainer._transformer = wrapped_model
        trainer._vae_decoder = FakeFrozenModule()
        trainer._vae_encoder = None
        trainer._log_transformer_device_summary = lambda stage: None

        trainer._prepare_models_for_training()

        self.assertEqual(trainer._accelerator.prepare_calls, [wrapped_model])
        self.assertEqual(
            base_model.calls,
            [
                ("set_gradient_checkpointing", False),
                ("enable_block_swap", 1, "cpu", True, True),
                ("move_to_device_except_swap_blocks", "cpu"),
                ("prepare_block_swap_before_forward", None),
                ("move_to_device_except_swap_blocks", "cpu"),
                ("prepare_block_swap_before_forward", None),
            ],
        )


if __name__ == "__main__":
    unittest.main()
