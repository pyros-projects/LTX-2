## Why

Single-GPU block swap was integrated for LTX-2 trainer LoRA runs, but current swap-enabled startup does not preserve the intended CPU/GPU residency layout through `Accelerator.prepare()`. In practice, Accelerate eagerly places the wrapped transformer on CUDA, collapsing the swap layout and making the feature unusable even though the model-native offloader and block-loop scheduling are already present.

## What Changes

- Update swap-enabled trainer startup to follow the Musubi timing pattern more closely by preventing eager device placement for the transformer during `Accelerator.prepare()` when block swap is enabled.
- Rework the post-prepare residency repair path so it operates on the prepared and unwrapped model object that Accelerate actually uses for training.
- Tighten swap-aware device summary and debug helpers so PEFT-wrapped swapped modules are recognized as expected CPU-resident modules instead of being misreported as unexpected.
- Add focused structural verification for swap-enabled prepare sequencing, post-prepare residency repair, and PEFT-prefixed CPU module reporting.

## Capabilities

### New Capabilities

None.

### Modified Capabilities

- `single-gpu-transformer-block-swap`: Swap-enabled startup must preserve intended transformer residency across `Accelerator.prepare()` for single-GPU LoRA runs, and swap-aware debug reporting must correctly classify PEFT-wrapped swapped modules.

## Impact

- Affected code:
  - `packages/ltx-trainer/src/ltx_trainer/trainer.py`
  - `packages/ltx-trainer/tests/test_block_swap_trainer_startup.py`
  - `packages/ltx-trainer/tests/test_trainer_debug_helpers.py`
- APIs:
  - no new public user-facing configuration fields are required
  - trainer-side startup behavior for swap-enabled runs changes to use non-default Accelerate device placement
- Dependencies and systems:
  - single-GPU `Accelerate` prepare/device placement flow
  - PEFT-wrapped LoRA transformer startup and residency inspection
  - existing model-native block-swap APIs in `ltx-core`
