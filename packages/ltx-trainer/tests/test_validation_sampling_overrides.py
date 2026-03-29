from __future__ import annotations

import importlib.util
from contextlib import nullcontext
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import torch


ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = ROOT / "src"
sys.path.insert(0, str(SRC_ROOT))


def _load_module(name: str, relative_path: str):
    module_path = ROOT / "src" / relative_path
    spec = importlib.util.spec_from_file_location(name, module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    spec.loader.exec_module(module)
    return module


CONFIG_MODULE = _load_module("ltx_trainer_config", "ltx_trainer/config.py")
TRAINER_MODULE = _load_module("ltx_trainer_trainer", "ltx_trainer/trainer.py")
SAMPLER_MODULE = _load_module("ltx_trainer_validation_sampler", "ltx_trainer/validation_sampler.py")


class FakeBaseModel:
    def __init__(self) -> None:
        self.weighted_calls: list[tuple[list[str], list[float], str, str]] = []
        self.set_calls: list[tuple[object, bool]] = []
        self.model = torch.nn.Module()
        self.model.lora_layer = FakeLoRALayer()

    def add_weighted_adapter(
        self,
        adapters: list[str],
        weights: list[float],
        adapter_name: str,
        combination_type: str = "svd",
    ) -> None:
        self.weighted_calls.append((adapters, weights, adapter_name, combination_type))

    def set_adapter(self, adapter_name: object, inference_mode: bool = False) -> None:
        self.set_calls.append((adapter_name, inference_mode))


class FakeLoRALayer(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.scaling = {
            "default": 1.0,
            "__validation_sampling__": 2.0,
        }


class FakePeftTransformer:
    def __init__(self) -> None:
        self.base_model = FakeBaseModel()
        self.peft_config: dict[str, object] = {"default": object()}
        self.active_adapter = "default"
        self.added: list[tuple[str, object, bool]] = []
        self.deleted: list[str] = []
        self.sets: list[str] = []
        self.grad_calls: list[tuple[str, bool]] = []

    def add_adapter(self, adapter_name: str, peft_config: object, low_cpu_mem_usage: bool = False) -> None:
        self.peft_config[adapter_name] = peft_config
        self.added.append((adapter_name, peft_config, low_cpu_mem_usage))

    def delete_adapter(self, adapter_name: str) -> None:
        self.peft_config.pop(adapter_name, None)
        self.deleted.append(adapter_name)

    def set_adapter(self, adapter_name: str) -> None:
        self.active_adapter = adapter_name
        self.sets.append(adapter_name)

    def set_requires_grad(self, adapter_name: str, requires_grad: bool = True) -> None:
        self.grad_calls.append((adapter_name, requires_grad))

    def modules(self):
        return self.base_model.model.modules()


class ValidationSamplingOverrideTests(unittest.TestCase):
    def test_validation_config_accepts_sampling_sigmas_and_lora_path(self) -> None:
        with tempfile.NamedTemporaryFile(suffix=".safetensors") as handle:
            config = CONFIG_MODULE.ValidationConfig(
                prompts=["hello"],
                sample_sigmas=[1.0, 0.9, 0.5, 0.0],
                sampling_lora_weight=handle.name,
                sampling_lora_multiplier=0.6,
            )

        self.assertEqual(config.sample_sigmas, [1.0, 0.9, 0.5, 0.0])
        self.assertEqual(str(config.sampling_lora_weight), handle.name)
        self.assertEqual(config.sampling_lora_multiplier, 0.6)

    def test_validation_config_rejects_increasing_sample_sigmas(self) -> None:
        with self.assertRaises(ValueError):
            CONFIG_MODULE.ValidationConfig(
                prompts=["hello"],
                sample_sigmas=[1.0, 0.5, 0.75, 0.0],
            )

    def test_validation_sampler_resolves_explicit_sigma_list(self) -> None:
        config = SAMPLER_MODULE.GenerationConfig(
            prompt="hello",
            num_inference_steps=30,
            sample_sigmas=[1.0, 0.9, 0.5, 0.0],
        )

        sigmas = SAMPLER_MODULE.ValidationSampler._resolve_sigmas(config, torch.device("cpu"))

        self.assertTrue(torch.equal(sigmas, torch.tensor([1.0, 0.9, 0.5, 0.0], dtype=torch.float32)))

    def test_build_validation_sampling_lora_config_supports_dynamic_rank_and_alpha(self) -> None:
        state_dict = {
            "diffusion_model.transformer_blocks.0.attn1.to_q.lora_A.weight": torch.randn(4, 8),
            "diffusion_model.transformer_blocks.0.attn1.to_q.lora_B.weight": torch.randn(8, 4),
            "diffusion_model.transformer_blocks.0.attn1.to_q.alpha": torch.tensor(16.0),
            "diffusion_model.transformer_blocks.0.attn2.to_v.lora_A.weight": torch.randn(7, 8),
            "diffusion_model.transformer_blocks.0.attn2.to_v.lora_B.weight": torch.randn(8, 7),
            "diffusion_model.transformer_blocks.0.attn2.to_v.alpha": torch.tensor(28.0),
        }

        normalized = TRAINER_MODULE.normalize_external_lora_state_dict(state_dict)
        config = TRAINER_MODULE.build_lora_config_from_state_dict(normalized)

        self.assertCountEqual(
            config.target_modules,
            [
                "transformer_blocks.0.attn1.to_q",
                "transformer_blocks.0.attn2.to_v",
            ],
        )
        self.assertEqual(config.rank_pattern["transformer_blocks.0.attn1.to_q"], 4)
        self.assertEqual(config.rank_pattern["transformer_blocks.0.attn2.to_v"], 7)
        self.assertEqual(config.alpha_pattern["transformer_blocks.0.attn1.to_q"], 16)
        self.assertEqual(config.alpha_pattern["transformer_blocks.0.attn2.to_v"], 28)

    def test_summarize_validation_sampling_lora_state_dict_reports_dynamic_shapes(self) -> None:
        state_dict = {
            "diffusion_model.transformer_blocks.0.attn1.to_q.lora_A.weight": torch.randn(4, 8, dtype=torch.bfloat16),
            "diffusion_model.transformer_blocks.0.attn1.to_q.lora_B.weight": torch.randn(8, 4, dtype=torch.bfloat16),
            "diffusion_model.transformer_blocks.0.attn1.to_q.alpha": torch.tensor(16.0),
            "diffusion_model.transformer_blocks.1.attn2.to_v.lora_A.weight": torch.randn(7, 8, dtype=torch.float16),
            "diffusion_model.transformer_blocks.1.attn2.to_v.lora_B.weight": torch.randn(8, 7, dtype=torch.float16),
            "diffusion_model.transformer_blocks.1.attn2.to_v.alpha": torch.tensor(28.0),
        }

        summary = TRAINER_MODULE.summarize_external_lora_state_dict(
            TRAINER_MODULE.normalize_external_lora_state_dict(state_dict)
        )

        self.assertEqual(summary["target_module_count"], 2)
        self.assertEqual(summary["rank_range"], (4, 7))
        self.assertEqual(summary["alpha_range"], (16, 28))
        self.assertEqual(summary["tensor_dtype_counts"]["torch.bfloat16"], 2)
        self.assertEqual(summary["tensor_dtype_counts"]["torch.float16"], 2)

    def test_validation_sampling_lora_scope_applies_mix_and_restores_default(self) -> None:
        trainer = TRAINER_MODULE.LtxvTrainer.__new__(TRAINER_MODULE.LtxvTrainer)
        trainer._config = type(
            "Config",
            (),
            {
                "model": type("ModelCfg", (), {"training_mode": "lora"})(),
                "validation": type(
                    "ValidationCfg",
                    (),
                    {
                        "sampling_lora_weight": "/tmp/sample.safetensors",
                        "sampling_lora_multiplier": 0.6,
                    },
                )(),
            },
        )()
        transformer = FakePeftTransformer()
        state_dict = {
            "diffusion_model.transformer_blocks.0.attn1.to_q.lora_A.weight": torch.randn(4, 8),
            "diffusion_model.transformer_blocks.0.attn1.to_q.lora_B.weight": torch.randn(8, 4),
            "diffusion_model.transformer_blocks.0.attn1.to_q.alpha": torch.tensor(16.0),
        }

        with (
            patch.object(TRAINER_MODULE, "load_file", return_value=state_dict),
            patch.object(TRAINER_MODULE, "set_peft_model_state_dict") as mock_set_state,
        ):
            with trainer._validation_sampling_lora_scope(transformer):
                self.assertEqual(
                    transformer.active_adapter,
                    ["default", "__validation_sampling__"],
                )
                self.assertEqual(
                    transformer.base_model.model.lora_layer.scaling["__validation_sampling__"],
                    1.2,
                )

        mock_set_state.assert_called_once()
        self.assertIn("__validation_sampling__", transformer.peft_config)
        self.assertEqual(transformer.base_model.weighted_calls, [])
        self.assertEqual(
            transformer.base_model.set_calls,
            [
                (["default", "__validation_sampling__"], True),
                ("default", False),
            ],
        )
        self.assertEqual(transformer.active_adapter, "default")
        self.assertEqual(transformer.base_model.model.lora_layer.scaling["__validation_sampling__"], 2.0)


if __name__ == "__main__":
    unittest.main()
