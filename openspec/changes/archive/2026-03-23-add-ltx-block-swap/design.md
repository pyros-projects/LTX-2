## Context

`ltx-core` owns the LTX transformer block list and block execution loop, while `ltx-trainer` owns model loading, quantization, PEFT wrapping, and `Accelerate` preparation. The target change is a cross-cutting single-GPU training feature: swap a configurable subset of `transformer_blocks` between CPU and GPU during trainer-side LoRA runs without relying on trainer monkey-patching.

The current code already gives us the right seams:

- `LTXModel.transformer_blocks` is a single `ModuleList`.
- `_process_transformer_blocks(...)` is the only place where block iteration is scheduled.
- trainer startup already quantizes before `Accelerator.prepare()` and already unwraps PEFT-backed transformers for model-specific methods such as gradient checkpointing.

The main constraints are:

- v1 is single-GPU only.
- v1 must remain compatible with trainer-side LoRA runs.
- v1 should allow quantize-then-swap startup, even with known caveats around packed Quanto weights.
- `Accelerator.prepare()` cannot be allowed to permanently collapse the intended CPU/GPU residency layout.

## Goals / Non-Goals

**Goals:**

- Add a model-native block-swap API to `LTXModel`.
- Reuse a Musubi-style single-list offloading pattern instead of inventing a new scheduler.
- Make block swap available through trainer configuration for single-GPU LoRA training.
- Preserve compatibility with gradient checkpointing and the existing quantization startup sequence.
- Add enough verification to prove startup, residency, and first-step behavior before broader rollout.

**Non-Goals:**

- Multi-GPU or distributed swap scheduling.
- Validation or inference integration in v1.
- Swapping non-transformer modules.
- Audio-specific policy tuning or dynamic swap heuristics.
- Broad refactors of the trainer loading architecture beyond what is required to keep swap-compatible startup.

## Decisions

### 1. Put the block-swap API on `LTXModel`

`LTXModel` will own:

- `enable_block_swap(blocks_to_swap, device, supports_backward, use_pinned_memory=False)`
- `switch_block_swap_for_training()`
- `switch_block_swap_for_inference()`
- `move_to_device_except_swap_blocks(device)`
- `prepare_block_swap_before_forward()`

It will also own the swap state (`blocks_to_swap`, `offloader`).

Rationale:

- The execution loop already lives in `ltx-core`, so scheduling logic belongs there.
- Trainer-side monkey-patching would be fragile around PEFT wrappers, checkpointed execution, and future reuse.

Alternatives considered:

- Trainer-only swap orchestration. Rejected because it would duplicate model knowledge and couple swap behavior to wrapper details.

### 2. Port the minimal Musubi offloader shape into `ltx-core`

We will add a focused helper module in `ltx-core` that ports the minimal `ModelOffloader` behavior needed for:

- preparing initial block residency before forward
- async CPU/GPU movement
- backward-safe training hooks
- optional pinned memory

Rationale:

- Musubi already demonstrates the exact single-list scheduler shape we need.
- Reusing its concepts and method names lowers design risk.

Alternatives considered:

- Building a fresh offloader from scratch. Rejected because the behavior is subtle and already solved well enough by the reference pattern.
- Keeping the helper trainer-local. Rejected for v1 because the model-native API needs an internal helper that is naturally consumed from `ltx-core`.

### 3. Schedule swaps around the block loop, not inside blocks

`_process_transformer_blocks(...)` will wrap each block iteration with:

1. `wait_for_block(block_idx)` before execution when swap is enabled
2. block execution or checkpointed execution
3. `submit_move_blocks_forward(transformer_blocks, block_idx)` after execution when swap is enabled

Rationale:

- The loop is the only place that can coordinate normal execution and gradient-checkpointed execution uniformly.
- Keeping swap orchestration outside `BasicAVTransformerBlock` avoids polluting the block implementation with transport concerns.

Alternatives considered:

- Embedding swap logic in each block. Rejected because it tangles transport and compute and does not help with checkpoint scheduling.

### 4. Keep trainer integration thin and quantize before enabling swap

Trainer startup will:

1. load the transformer on CPU
2. quantize first when configured
3. enable block swap on the base transformer
4. move non-swapped modules to the accelerator device
5. prepare the wrapped model without losing the intended residency layout
6. re-assert swap residency after wrapping and before the first forward

Rationale:

- The existing quantization flow is already blockwise and should stay the owner of quantized weight creation.
- Enabling swap after quantization avoids mixing quantization and residency concerns.

Alternatives considered:

- Enabling swap before quantization. Rejected because it complicates quantization order and invites device-layout interactions during weight packing.

### 5. Add configuration for swap count and pinned-memory behavior

`ltx-trainer` will expose:

- `blocks_to_swap: int = 0`
- `use_pinned_memory_for_block_swap: bool = false`

Rationale:

- `blocks_to_swap=0` preserves the current behavior as the default.
- pinned memory is important enough to expose as an explicit v1 knob because it meaningfully affects the transfer/memory trade-off.

Alternatives considered:

- Hard-coding pinned memory or swap count behavior. Rejected because the workload and hardware trade-offs are too variable.

## Risks / Trade-offs

- Packed-quantized weights may behave poorly across CPU/GPU movement, especially qint4 TinyGemm paths. -> Start with structural and startup validation, document caveats, and keep the swap ordering strictly quantize-then-swap.
- `Accelerator.prepare()` may eagerly move state in ways that fight the desired swap residency. -> Keep model-native `move_to_device_except_swap_blocks(...)` and `prepare_block_swap_before_forward()` as the post-prepare source of truth.
- Backward hooks plus gradient checkpointing can deadlock or stall if scheduling is wrong. -> Keep scheduling at the loop boundary, and verify with first-step tracing before claiming support.
- Swapping only transformer block weights means non-block modules remain resident on GPU. -> Treat this as an explicit v1 trade-off in exchange for lower complexity and clearer correctness boundaries.

## Migration Plan

This change is additive and gated behind new trainer config values. Existing runs remain unchanged with `blocks_to_swap: 0`.

Rollout plan:

1. land model-native APIs and helper logic behind an opt-in configuration
2. wire trainer startup for single-GPU LoRA mode only
3. add structural and startup verification
4. run targeted smoke tests for non-quantized and quantized swap configurations

Rollback strategy:

- disable the feature by setting `blocks_to_swap: 0`
- if needed, remove the trainer wiring while keeping the model API internal and unused

## Open Questions

- Whether `Accelerator.prepare()` needs explicit non-default device placement flags in this codebase or whether post-prepare residency repair is sufficient.
- Whether some quantized modes, especially qint4, should be documented as experimental in v1 even if they pass structural startup checks.
