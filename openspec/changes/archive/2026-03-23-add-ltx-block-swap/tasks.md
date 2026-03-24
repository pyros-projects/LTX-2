## 1. Core block-swap support in `ltx-core`

- [x] 1.1 Add a minimal Musubi-style offloader helper module in `ltx-core` that supports initial block residency preparation, forward swap scheduling, backward-safe hooks, and optional pinned memory.
- [x] 1.2 Extend `LTXModel` with block-swap state and public methods for enabling swap, switching training or inference mode, moving non-swapped modules to device, and preparing swap state before forward.
- [x] 1.3 Update `_process_transformer_blocks(...)` to perform wait-before-execute and schedule-after-execute swap orchestration around both normal and checkpointed block execution.

## 2. Trainer configuration and startup wiring

- [x] 2.1 Add trainer config fields for `blocks_to_swap` and `use_pinned_memory_for_block_swap` with defaults that preserve the current non-swap behavior.
- [x] 2.2 Update trainer model loading and preparation so single-GPU LoRA runs can quantize first, enable swap on the base transformer, and keep non-swapped modules resident on the accelerator device.
- [x] 2.3 Re-assert swap residency after `Accelerator.prepare()` and before the first forward pass so swapped and non-swapped blocks end up on the intended devices.

## 3. Structural verification

- [x] 3.1 Add focused unit or structural tests for the new transformer block-swap API and swap configuration defaults.
- [x] 3.2 Add trainer-side tests for swap-enabled startup decisions, including the single-GPU quantize-then-swap path and residency preparation helpers.
- [x] 3.3 Extend or add debug-oriented verification so first-step tracing can confirm swap-enabled execution does not unexpectedly run resident modules on CPU outside intended transfers.

## 4. Targeted smoke validation and follow-up notes

- [x] 4.1 Run targeted smoke checks for non-quantized and quantized single-GPU LoRA startup with `blocks_to_swap > 0`, and capture any known caveats.
- [x] 4.2 Document v1 limitations and experimental caveats for swap plus quantization modes, especially packed-weight behavior that remains fragile.
