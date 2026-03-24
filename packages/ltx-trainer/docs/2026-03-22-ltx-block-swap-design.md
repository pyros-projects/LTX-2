# LTX Block Swap Design

Date: 2026-03-22

## Goal

Add block swapping to the LTX-2 codebase for single-GPU, trainer-side, LoRA runs.

The immediate target is `ltx-trainer` training runs on one GPU. The desired payoff is to make larger or more aggressive configurations practical, especially combinations like:

- higher precision base + block swap
- trainer quantization + block swap
- future experiments such as TorchAO FP8 + block swap

This is intentionally not a general multi-GPU offloading design and not an inference-first feature.

## Why This Approach

The selected approach is to add block-swap support directly to the LTX transformer model in `ltx-core`, then keep `ltx-trainer` integration thin.

Why:

1. The real transformer block loop lives in `ltx-core`, not in the trainer.
2. Musubi's proven single-list pattern maps well to LTX because LTX has a single `transformer_blocks` `ModuleList`.
3. Trainer-only monkey-patching would be brittle around PEFT wrapping, Accelerate wrapping, and gradient checkpointing.
4. Putting swap control on the model itself makes it easier to reuse later in inference or other tooling if desired.

## Scope

### In Scope

- Single-GPU only
- Trainer-side LoRA runs only
- Swapping transformer blocks between CPU and GPU
- Support for backward-compatible swapping during training
- Allow quantized + swapped runs in v1, with clear caveats

### Out of Scope

- Multi-GPU / distributed swap scheduling
- Validation/inference wiring in v1
- Audio-specific swap policy tuning
- General layer offloading beyond transformer blocks
- A full abstraction for arbitrary module swapping

## Current Code Context

### LTX side

The LTX transformer is a good candidate for Musubi-style swapping:

- `transformer_blocks` are initialized in:
  - `packages/ltx-core/src/ltx_core/model/transformer/model.py`
- the execution loop is:
  - `AVTransformer3DModel._process_transformer_blocks(...)`
- LTX has a single top-level `ModuleList`, not a split double/single layout

Important non-block modules live outside `transformer_blocks` and should remain resident on GPU:

- `patchify_proj`
- `adaln_single`
- `prompt_adaln_single`
- `scale_shift_table`
- `norm_out`
- `proj_out`
- audio-side equivalents where present
- AV cross-AdaLN modules

### Trainer side

Relevant trainer points:

- model load / quantization:
  - `packages/ltx-trainer/src/ltx_trainer/trainer.py`
  - `packages/ltx-trainer/src/ltx_trainer/quantization.py`
- current single-GPU quantization path already has custom handling for packed Quanto tensors
- PEFT wrapping happens before model preparation for training
- `Accelerator.prepare()` currently owns device placement

### Musubi reference pattern

The best reference is Musubi's single-list block swap model pattern:

- Musubi repo root on this machine:
  - `/home/pyro/repos/musubi-tuner`
- offloader utility:
  - `/home/pyro/repos/musubi-tuner/src/musubi_tuner/modules/custom_offloading_utils.py`
- representative single-list model:
  - `/home/pyro/repos/musubi-tuner/src/musubi_tuner/qwen_image/qwen_image_model.py`

The key runtime pattern is:

1. initialize offloader with the block list
2. prepare initial block residency before forward
3. for each block:
   - wait for current block transfer
   - execute block
   - schedule next swap

## Proposed Architecture

### 1. Add a small LTX-native block swap API to the transformer model

Add methods on the transformer model similar to Musubi:

- `enable_block_swap(num_blocks, device, supports_backward, use_pinned_memory=False)`
- `switch_block_swap_for_training()`
- `switch_block_swap_for_inference()` (optional stub in v1, useful for symmetry)
- `move_to_device_except_swap_blocks(device)`
- `prepare_block_swap_before_forward()`

Also add state:

- `blocks_to_swap`
- `offloader`

This should live on the model that owns `transformer_blocks`, not in the trainer.

### 2. Vendor or port a minimal offloading utility into LTX

Do not invent a new scheduler from scratch.

Port the minimal useful subset of Musubi's offloading utility into LTX, ideally into a trainer-local utility module first, or into a clearly named internal helper module if it must be shared.

Needed capabilities:

- prepare block devices before forward
- asynchronous CPU/GPU moves
- backward hook support for training
- pinned memory option

Avoid bringing in unrelated Musubi abstractions.

### 3. Instrument the LTX block loop

Modify `_process_transformer_blocks(...)` in `ltx-core` so each block iteration becomes:

1. if swap enabled: `wait_for_block(block_idx)`
2. run the block
3. if swap enabled: `submit_move_blocks_forward(transformer_blocks, block_idx)`

This must work both in:

- normal execution
- gradient-checkpointed execution

The scheduling should sit around the checkpoint call, not inside the block itself.

### 4. Keep trainer integration thin

In `ltx-trainer`, add config knobs and a small wiring layer:

- `blocks_to_swap: int = 0`
- `use_pinned_memory_for_block_swap: bool = false`

Trainer behavior when `blocks_to_swap > 0`:

1. load transformer in a swap-compatible way
2. if quantization is enabled, quantize first
3. enable block swap on the base transformer
4. move non-swapped modules to accelerator device
5. call `Accelerator.prepare()` with non-default device placement handling
6. call `prepare_block_swap_before_forward()` after wrapping

The preferred pattern is close to how Musubi avoids a full eager move when swap is enabled.

## Quantization Strategy

The user explicitly wants high-payoff v1, which means allowing `swap + quantization` from the start.

### Supported in v1

- non-quantized transformer + block swap
- current trainer quantization + block swap

### Important caveat

Quanto packed weights are already known to be touchy in this repo, especially:

- device-type changes
- TinyGemm packed tensors
- qint4 performance instability

So the implementation should support quantized swapping, but the design should assume:

- `int2`, `int8`, and future FP8-style routes may behave differently
- `int4` may remain slow or fragile even if functionally supported

### Practical ordering

The least-bad ordering is:

1. load model
2. quantize model
3. enable block swap
4. move resident non-swapped modules to GPU
5. prepare with Accelerate

This avoids quantizing already-swapped residency layouts and keeps the swap logic responsible only for block motion.

## PEFT / Wrapper Considerations

LoRA runs wrap the transformer with PEFT.

That means the trainer must be careful to enable swap support on the underlying base transformer, not just the PEFT wrapper.

Likely rule:

- whenever model methods are needed (`set_gradient_checkpointing`, block-swap API), operate on `get_base_model()` if available

The current trainer already does this for gradient checkpointing, so the same pattern should be reused.

## Device Placement Design

The model should support a mode where:

- non-swapped blocks are resident on GPU
- swapped blocks keep weights on CPU but may temporarily move non-weight state as needed
- top-level resident modules stay on GPU

The trainer should avoid forcing a full-model eager move if swap is enabled.

This means `Accelerator.prepare()` likely needs a custom device placement path similar in spirit to Musubi's:

- do not let Accelerate eagerly place everything on the target device first
- prepare the wrapped model
- then re-assert swap residency via:
  - `move_to_device_except_swap_blocks(...)`
  - `prepare_block_swap_before_forward()`

## Verification Plan

The first implementation should be verified in layers:

### 1. Unit / structural verification

- model exposes block-swap API
- enabling swap sets expected state
- `move_to_device_except_swap_blocks()` preserves intended residency

### 2. Startup verification

Single-GPU LoRA run should:

- initialize successfully with `blocks_to_swap > 0`
- pass `Accelerator.prepare()`
- show expected CPU/GPU parameter distribution before first step

### 3. First-step verification

Use the current debug tooling in `trainer.py` to confirm:

- swapped runs execute without CPU surprise outside intended block transfers
- no deadlock around checkpointed blocks
- no packed-tensor device-move stalls

### 4. Practical smoke tests

Smoke test combinations:

- no quantization + block swap
- int8-quanto + block swap
- int2-quanto + block swap

Treat qint4 as experimental even if supported.

## Likely File Touches

Expected main files:

- `packages/ltx-core/src/ltx_core/model/transformer/model.py`
- new block-offload helper module in `ltx-core` or `ltx-trainer`
- `packages/ltx-trainer/src/ltx_trainer/trainer.py`
- `packages/ltx-trainer/src/ltx_trainer/config.py`
- possibly `packages/ltx-trainer/src/ltx_trainer/config_display.py`
- tests for swap state / residency / startup behavior

## Risks

### Highest risk

- interaction between swap and quantized packed tensors

### Medium risk

- Accelerate device placement reintroducing full-model moves
- checkpointed execution and async transfers interacting badly
- PEFT wrapper method dispatch causing swap API to hit the wrong object

### Lower risk

- non-block top-level modules accidentally left on CPU
- validation/inference expectations creeping into v1

## Recommendation for Implementation Order

1. Add offloader helper and model API in `ltx-core`
2. Instrument the LTX block loop
3. Wire trainer config and base-model method calls
4. Make non-quantized swap run first
5. Enable quantized swap path
6. Validate with current startup and first-step debug logs

This sequence still honors the user's requested high-payoff goal, but it keeps failure localization clean.

## Notes for Future Me

- Do not start by monkey-patching trainer forward calls. The real loop is in `ltx-core`.
- Keep v1 focused on the single `transformer_blocks` list. Do not generalize early.
- Reuse Musubi's offloader behavior and naming where it reduces thinking.
- Be very careful with PEFT wrappers: call swap methods on the base transformer.
- Be very careful with quantized tensor device moves. The earlier Quanto work in this repo already exposed that device-type transitions can be pathological.
- If `swap + quant` becomes unstable, do not rip it out immediately. Instead:
  - keep the API shape
  - gate specific quantization modes
  - document supported combinations

## Suggested Next Session Objective

Transfer this note into OpenSpec and then implement:

"Single-GPU trainer-side block swapping for LTX transformer blocks, with initial support for quantized LoRA runs."

## Session Summary So Far

This section is for future me in the next session so the same ground does not get covered twice.

### What Was Added Locally

Helper files created in `packages/ltx-trainer`:

- `scripts/build_image_jsonl_from_txt_pairs.py`
- `tests/test_build_image_jsonl_from_txt_pairs.py`
- `scripts/run_image_lora_24gb_fp8.sh`
- `configs/ltx2_image_lora_24gb_fp8_example.yaml` (local helper file; may be gitignored)

These were created to support image-only LoRA training from common `image.png` + `image.txt` datasets on a 24 GB GPU.

### Dataset / Preprocessing Lessons

The dataset loader expects media paths relative to the metadata file's parent directory. An earlier launcher version wrote the generated JSONL under `_data`, which broke relative path handling for datasets stored elsewhere.

The launcher was updated so the auto-generated metadata file is written inside the dataset root as:

- `.ltx_dataset.jsonl`

Another mismatch also showed up:

- latent files were written into `_data/latents`
- caption condition files were initially not where the trainer expected them

The condition `.pt` files were manually copied into:

- `_data/conditions`

After that, the trainer successfully loaded `208` matched samples.

### Trainer / Quantization Patches Already Made

Local changes were made in:

- `packages/ltx-trainer/src/ltx_trainer/trainer.py`
- `packages/ltx-trainer/src/ltx_trainer/quantization.py`

Tests added locally:

- `tests/test_trainer_startup_quantization.py`
- `tests/test_quantization_device_restore.py`
- `tests/test_trainer_debug_helpers.py`

Main fixes already implemented:

1. Added detailed startup timing and device logging.
2. Added first-step tracing to catch CPU execution and identify slow modules.
3. Fixed a startup bug where `_accelerator` was referenced before initialization.
4. Fixed single-GPU quantized startup so packed Quanto tensors do not get pointlessly bounced across device types before training.
5. Fixed blockwise quantization exclusions so AdaLN-related modules are not quantized into unsafe TinyGemm paths.

### What We Verified About Device Placement

At one point the logs showed:

- `DEVICE SUMMARY  params={'cpu': 2, 'cuda:0': 6488}`
- CPU parameter modules: `['base_model.model']`

This turned out not to be a hidden slow layer issue. The `2` CPU parameters were root-level direct parameters grouped under the base PEFT model path, and after `Accelerator.prepare()` the model became fully CUDA-resident.

The first-step trace later showed:

- `cpu_involved=0`

So the actual executed path for the measured first step was not silently falling back to CPU.

### Quantization Findings

The current repo behavior matters a lot:

- `int2-quanto` is functionally usable and much faster than `int4-quanto` on this stack.
- `int4-quanto` can work, but it was dramatically slower.

Observed user-side result:

- `int4-quanto`: roughly order-of-magnitude slower step time
- `int2-quanto`: about `1s/step` in the shown training progress output

Local investigation in the active LTX environment showed why:

- `qint2` uses the generic `WeightQBitsTensor` path
- `qint4` on BF16 weights uses `TinyGemmWeightQBitsTensor`

This means `int2` and `int4` are not just different bit widths. They hit different backends.

### Why OneTrainer Felt Different

The user's comparison to OneTrainer was valid, but the backends are not comparable:

- OneTrainer's "4-bit" path is BitsAndBytes NF4
- LTX trainer's current "4-bit" path is Quanto qint4 / TinyGemm

So "OneTrainer int4 works great" does not mean Quanto int4 should behave similarly.

### Upstream Issues Already Filed

Issues were opened under the `pyros-projects` GitHub account:

- Optimum-Quanto:
  - `https://github.com/huggingface/optimum-quanto/issues/413`
- LTX-2:
  - `https://github.com/Lightricks/LTX-2/issues/173`

These issues document that `int4-quanto` can be much slower than `int2-quanto` for this style of LoRA training workload.

### Validation Quality Findings

Validation outputs were poor even at step `0`, including single-frame validation.

That strongly suggests:

- the validation/inference stack was already producing bad outputs before learning quality could even be judged

Important conclusion:

- this was not caused by `first_frame_conditioning_p`

For single-frame training, the current first-frame conditioning mask logic is effectively a no-op, so the image training data was not being thrown away.

The more likely explanation is the very aggressive validation stack:

- FP8 base checkpoint
- plus trainer-side low-bit quantization
- plus validation generation through the trainer

That setup is not a trustworthy way to judge whether a concept token is being learned.

### Checkpoint Notes

The local comfy-style file:

- `ltx-2-3-22b-dev_transformer_only_fp8_input_scaled.safetensors`

is transformer-only and not a drop-in `ltx-trainer` training base checkpoint.

It lacks the full checkpoint contents that the trainer loader expects, such as:

- VAE
- vocoder
- text embedding projection

So for trainer-side work, the intended full checkpoints are still the official LTX model files, especially the BF16 base if quality sanity checks are needed.

### Current Practical State

Right now, the repo contains:

- working local image-only training helpers
- improved trainer debug instrumentation
- quantization fixes needed to make low-bit experiments run
- evidence that `int2-quanto` is the currently practical low-bit path on this stack

And the next meaningful engineering step is:

- add block swapping in a model-native way so larger or safer precision modes can become practical on a single GPU

### Recommended OpenSpec Framing

When transferring this into OpenSpec, frame it as:

- single-GPU
- trainer-side
- LoRA-focused
- transformer-block-only swap
- with explicit intent to support quantized runs, but with caution around specific quant backends
