from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
import unittest

import torch


CORE_SRC = Path(__file__).resolve().parents[2] / "ltx-core" / "src"
sys.path.insert(0, str(CORE_SRC))

MODULE_PATH = CORE_SRC / "ltx_core" / "text_encoders" / "gemma" / "embeddings_connector.py"
SPEC = importlib.util.spec_from_file_location("ltx_core_embeddings_connector", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC is not None and SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


class EmbeddingsConnectorBatchingTests(unittest.TestCase):
    def test_connector_handles_batched_attention_masks_with_learnable_registers(self) -> None:
        connector = MODULE.Embeddings1DConnector(
            attention_head_dim=4,
            num_attention_heads=2,
            num_layers=0,
            num_learnable_registers=128,
        )
        hidden_states = torch.randn(2, 128, 8)
        attention_mask = torch.zeros(2, 1, 1, 128, dtype=hidden_states.dtype)

        encoded, encoded_mask = connector(hidden_states, attention_mask)

        self.assertEqual(tuple(encoded.shape), (2, 128, 8))
        self.assertEqual(tuple(encoded_mask.shape), (2, 1, 1, 128))

    def test_connector_handles_batched_split_rope(self) -> None:
        connector = MODULE.Embeddings1DConnector(
            attention_head_dim=4,
            num_attention_heads=2,
            num_layers=1,
            num_learnable_registers=None,
            rope_type=MODULE.LTXRopeType.SPLIT,
        )
        hidden_states = torch.randn(2, 256, 8)
        attention_mask = torch.zeros(2, 1, 1, 256, dtype=hidden_states.dtype)

        encoded, encoded_mask = connector(hidden_states, attention_mask)

        self.assertEqual(tuple(encoded.shape), (2, 256, 8))
        self.assertEqual(tuple(encoded_mask.shape), (2, 1, 1, 256))


if __name__ == "__main__":
    unittest.main()
