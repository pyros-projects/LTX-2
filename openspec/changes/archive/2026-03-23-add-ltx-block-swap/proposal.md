## Why

Single-GPU LoRA training for LTX-2 is currently constrained by transformer residency on the accelerator, even with existing trainer-side quantization and gradient checkpointing. Adding transformer block swapping now unlocks larger and safer single-GPU training configurations, including higher-precision and quantized runs that are currently hard to fit reliably.

## What Changes

- Add single-GPU block swapping for LTX transformer blocks during trainer-side LoRA runs.
- Introduce a model-native block-swap API on the LTX transformer so swap scheduling lives with the block execution loop instead of trainer-side monkey-patching.
- Port a minimal Musubi-style block offloader to manage CPU/GPU movement, forward scheduling, and backward-safe training behavior.
- Add trainer configuration and startup wiring for block swap, including support for quantize-then-swap startup on single GPU.
- Add structural and startup verification for swap configuration, residency, and first-step behavior.

## Capabilities

### New Capabilities
- `single-gpu-transformer-block-swap`: Enables single-GPU trainer runs to swap a configured subset of LTX transformer blocks between CPU and GPU while keeping the non-swapped transformer path and trainer startup flow usable.

### Modified Capabilities

None.

## Impact

- Affected code:
  - `packages/ltx-core/src/ltx_core/model/transformer/model.py`
  - new internal offloading helper module in `ltx-core`
  - `packages/ltx-trainer/src/ltx_trainer/config.py`
  - `packages/ltx-trainer/src/ltx_trainer/trainer.py`
  - trainer tests covering startup, residency, and swap configuration
- APIs:
  - new transformer block-swap methods and trainer config fields
- Dependencies and systems:
  - single-GPU `Accelerate` device placement flow
  - PEFT-wrapped LoRA transformers via `get_base_model()`
  - existing Quanto quantization path, especially packed-weight device movement behavior
