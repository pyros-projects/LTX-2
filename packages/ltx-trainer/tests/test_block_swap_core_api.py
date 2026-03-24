from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import torch


CORE_SRC = Path(__file__).resolve().parents[2] / "ltx-core" / "src"
sys.path.insert(0, str(CORE_SRC))

MODULE_PATH = CORE_SRC / "ltx_core" / "model" / "transformer" / "model.py"
SPEC = importlib.util.spec_from_file_location("ltx_core_transformer_model", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC is not None and SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


class BlockSwapCoreApiTests(unittest.TestCase):
    @staticmethod
    def _build_model() -> object:
        return MODULE.LTXModel(
            model_type=MODULE.LTXModelType.VideoOnly,
            num_layers=2,
            num_attention_heads=1,
            attention_head_dim=8,
            in_channels=8,
            out_channels=8,
            cross_attention_dim=16,
        )

    def test_model_exposes_block_swap_methods_and_default_state(self) -> None:
        model = self._build_model()

        self.assertEqual(model.blocks_to_swap, 0)
        self.assertIsNone(model.offloader)
        self.assertTrue(callable(model.enable_block_swap))
        self.assertTrue(callable(model.switch_block_swap_for_training))
        self.assertTrue(callable(model.switch_block_swap_for_inference))
        self.assertTrue(callable(model.move_to_device_except_swap_blocks))
        self.assertTrue(callable(model.prepare_block_swap_before_forward))

    def test_enable_block_swap_tracks_swap_state_on_model(self) -> None:
        model = self._build_model()

        model.enable_block_swap(
            blocks_to_swap=1,
            device=torch.device("cpu"),
            supports_backward=True,
        )

        self.assertEqual(model.blocks_to_swap, 1)
        self.assertIsNotNone(model.offloader)
        self.assertFalse(model.offloader.forward_only)

    def test_process_transformer_blocks_schedules_swaps_around_execution(self) -> None:
        model = self._build_model()
        events = []

        class RecordingOffloader:
            def wait_for_block(self, block_idx: int) -> None:
                events.append(("wait", block_idx))

            def submit_move_blocks_forward(self, blocks: object, block_idx: int) -> None:
                del blocks
                events.append(("submit", block_idx))

        class RecordingBlock(torch.nn.Module):
            def __init__(self, idx: int) -> None:
                super().__init__()
                self.idx = idx

            def forward(self, video: object, audio: object, perturbations: object) -> tuple[object, object]:
                del perturbations
                events.append(("block", self.idx))
                return video, audio

        model.blocks_to_swap = 1
        model.offloader = RecordingOffloader()
        model.transformer_blocks = torch.nn.ModuleList([RecordingBlock(0), RecordingBlock(1)])

        video, audio = model._process_transformer_blocks(video="video", audio="audio", perturbations=None)

        self.assertEqual((video, audio), ("video", "audio"))
        self.assertEqual(
            events,
            [
                ("wait", 0),
                ("block", 0),
                ("submit", 0),
                ("wait", 1),
                ("block", 1),
                ("submit", 1),
            ],
        )

    def test_process_transformer_blocks_keeps_swap_order_with_checkpointing(self) -> None:
        model = self._build_model()
        model.train()
        model.set_gradient_checkpointing(True)
        events = []

        class RecordingOffloader:
            def wait_for_block(self, block_idx: int) -> None:
                events.append(("wait", block_idx))

            def submit_move_blocks_forward(self, blocks: object, block_idx: int) -> None:
                del blocks
                events.append(("submit", block_idx))

        class RecordingBlock(torch.nn.Module):
            def __init__(self, idx: int) -> None:
                super().__init__()
                self.idx = idx

            def forward(self, video: object, audio: object, perturbations: object) -> tuple[object, object]:
                del perturbations
                events.append(("block", self.idx))
                return video, audio

        def fake_checkpoint(block: torch.nn.Module, video: object, audio: object, perturbations: object, use_reentrant: bool):
            self.assertFalse(use_reentrant)
            events.append(("checkpoint", getattr(block, "idx", -1)))
            return block(video, audio, perturbations)

        model.blocks_to_swap = 1
        model.offloader = RecordingOffloader()
        model.transformer_blocks = torch.nn.ModuleList([RecordingBlock(0), RecordingBlock(1)])

        with patch("torch.utils.checkpoint.checkpoint", side_effect=fake_checkpoint):
            video, audio = model._process_transformer_blocks(video="video", audio="audio", perturbations=None)

        self.assertEqual((video, audio), ("video", "audio"))
        self.assertEqual(
            events,
            [
                ("wait", 0),
                ("checkpoint", 0),
                ("block", 0),
                ("submit", 0),
                ("wait", 1),
                ("checkpoint", 1),
                ("block", 1),
                ("submit", 1),
            ],
        )


if __name__ == "__main__":
    unittest.main()
