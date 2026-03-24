from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import torch


MODULE_PATH = Path(__file__).resolve().parents[1] / "src" / "ltx_trainer" / "trainer.py"
sys.path.insert(0, str(MODULE_PATH.parents[1]))
SPEC = importlib.util.spec_from_file_location("ltx_trainer_trainer", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC is not None and SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


class FakeAccelerator:
    def __init__(self) -> None:
        self.prepare_calls: list[tuple[object, object]] = []

    def prepare(self, optimizer: object, scheduler: object) -> tuple[object, object]:
        self.prepare_calls.append((optimizer, scheduler))
        return optimizer, scheduler


class TrainerOptimizerInitTests(unittest.TestCase):
    def _build_trainer(self, optimizer_args: dict | None = None) -> object:
        trainer = MODULE.LtxvTrainer.__new__(MODULE.LtxvTrainer)
        trainer._accelerator = FakeAccelerator()
        trainer._trainable_params = [torch.nn.Parameter(torch.ones(1))]
        trainer._config = type(
            "Config",
            (),
            {
                "optimization": type(
                    "OptCfg",
                    (),
                    {
                        "learning_rate": 1e-3,
                        "optimizer_type": "adafactor",
                        "optimizer_args": {} if optimizer_args is None else optimizer_args,
                        "scheduler_type": None,
                        "scheduler_params": {},
                        "steps": 1,
                    },
                )(),
            },
        )()
        trainer._create_scheduler = lambda optimizer: ("scheduler", optimizer)
        return trainer

    def test_adafactor_defaults_to_manual_lr_mode(self) -> None:
        trainer = self._build_trainer()

        with patch("transformers.optimization.Adafactor") as mock_adafactor:
            trainer._init_optimizer()

        mock_adafactor.assert_called_once_with(
            trainer._trainable_params,
            lr=1e-3,
            relative_step=False,
            scale_parameter=False,
        )

    def test_adafactor_preserves_explicit_optimizer_args(self) -> None:
        trainer = self._build_trainer(
            optimizer_args={
                "relative_step": False,
                "scale_parameter": True,
                "clip_threshold": 0.5,
            }
        )

        with patch("transformers.optimization.Adafactor") as mock_adafactor:
            trainer._init_optimizer()

        mock_adafactor.assert_called_once_with(
            trainer._trainable_params,
            lr=1e-3,
            relative_step=False,
            scale_parameter=True,
            clip_threshold=0.5,
        )


if __name__ == "__main__":
    unittest.main()
