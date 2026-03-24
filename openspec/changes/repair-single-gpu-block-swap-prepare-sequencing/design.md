## Context

Single-GPU trainer-side block swap already exists in `ltx-core` and is wired into `ltx-trainer`, but swap-enabled LoRA startup does not currently preserve the intended residency layout through `Accelerator.prepare()`. The current trainer flow enables swap and prepares initial residency before calling plain `Accelerator.prepare(self._transformer)`, which allows Accelerate to eagerly move the wrapped transformer to the target CUDA device before the post-prepare repair logic runs.

Comparison against Musubi shows two important differences in startup timing:

- Musubi loads the swap-enabled transformer on CPU when block swap is enabled.
- Musubi calls `accelerator.prepare(transformer, device_placement=[False])` for the transformer, then re-applies `move_to_device_except_swap_blocks(...)` and `prepare_block_swap_before_forward()` on the unwrapped model immediately after prepare.

The current LTX trainer also misclassifies PEFT-wrapped swapped CPU modules as unexpected in device summary logs because the expected swap prefixes are generated without accounting for wrapper prefixes such as `base_model.model.`.

Constraints:

- v1 remains single-GPU only.
- The change must preserve the current model-native block-swap API in `ltx-core`.
- Quantize-then-swap startup must remain intact.
- The trainer uses PEFT-wrapped LoRA transformers, so any residency repair or debug inspection must work against wrapped and unwrapped forms consistently.

## Goals / Non-Goals

**Goals:**

- Preserve intended CPU/GPU swap residency across `Accelerator.prepare()` for swap-enabled single-GPU LoRA runs.
- Align transformer prepare timing more closely with Musubi’s working block-swap startup pattern.
- Ensure post-prepare residency repair operates on the actual model object Accelerate uses for training.
- Make swap-aware device summary helpers classify PEFT-prefixed swapped CPU modules correctly.
- Add focused structural verification for prepare sequencing, unwrap-and-restore behavior, and PEFT-aware debug classification.

**Non-Goals:**

- Redesign the `ltx-core` offloader or block-loop scheduling.
- Expand block swap beyond single-GPU LoRA training.
- Add inference or validation-specific swap behavior in this change.
- Rework the existing trainer quantization implementation beyond what is needed to keep swap-compatible startup intact.

## Decisions

### 1. Disable Accelerate device placement for the transformer when block swap is enabled

For swap-enabled runs, the trainer will call Accelerate with transformer-specific non-default device placement behavior equivalent to Musubi’s `device_placement=[False]` pattern.

Rationale:

- This is the clearest known difference between the current LTX startup flow and Musubi’s working training flow.
- Preventing eager placement is better than letting Accelerate collapse residency and trying to reconstruct the intended layout afterward.

Alternatives considered:

- Keep plain `Accelerator.prepare()` and rely only on post-prepare residency repair. Rejected because observed logs show the wrapped transformer becomes fully CUDA-resident and the intended swap layout is lost.
- Move swap enablement until after `Accelerator.prepare()`. Rejected because it breaks the current quantize-then-swap ordering and diverges from the established Musubi pattern.

### 2. Perform residency repair on the prepared and unwrapped model

After swap-enabled prepare, the trainer will apply `move_to_device_except_swap_blocks(...)` and `prepare_block_swap_before_forward()` against the unwrapped prepared model, not only against the pre-prepare wrapper object.

Rationale:

- Musubi explicitly unwraps the model before re-applying block-swap residency after prepare.
- This makes the repair logic target the model instance Accelerate actually uses during training.

Alternatives considered:

- Continue using the current wrapper object directly after prepare. Rejected because wrapper identity and device movement behavior may differ after Accelerate preparation.

### 3. Keep swap enablement on the base transformer before prepare

The trainer will continue enabling swap on the base transformer before prepare, after any quantization startup work has completed.

Rationale:

- The model-native swap API and current quantize-then-swap ordering are still sound.
- The observed failure is in prepare sequencing, not in the existence or ownership of the swap API.

Alternatives considered:

- Re-enable swap from scratch after prepare. Rejected because it introduces more state churn and increases divergence from the working Musubi timing pattern.

### 4. Make swap-aware debug reporting PEFT-prefix aware

The expected swapped CPU prefix calculation will account for PEFT-wrapped module name prefixes so swapped blocks under names such as `base_model.model.transformer_blocks.<idx>` are classified as expected CPU residency.

Rationale:

- Current logs over-report expected swapped CPU parameters as unexpected, which obscures whether the prepare sequencing fix actually works.
- Correct debug classification is needed to validate residency behavior with confidence.

Alternatives considered:

- Leave the existing debug helpers unchanged and inspect raw device summaries manually. Rejected because it makes the feature harder to validate and maintain.

## Risks / Trade-offs

- [Accelerate argument shape differs from Musubi in this trainer setup] -> Mitigation: keep the change narrowly scoped to transformer preparation and add targeted tests around the exact prepare call behavior.
- [Quantized swap-enabled runs may still have backend-specific residency quirks] -> Mitigation: preserve quantize-then-swap ordering and treat this change as prepare-sequencing repair, not a full quantization behavior rewrite.
- [PEFT/Accelerate wrapper interactions may still hide edge cases] -> Mitigation: perform residency repair on the unwrapped prepared model and add structural verification for wrapped naming and restore flow.
- [Debug helpers may still miss uncommon wrapper prefixes] -> Mitigation: normalize around the currently used PEFT wrapper naming and keep the helper logic prefix-based rather than hard-coding a single exact path.

## Migration Plan

1. Update swap-enabled trainer preparation so the transformer uses non-default Accelerate device placement when block swap is enabled.
2. Rework residency restoration to operate on the prepared, unwrapped transformer.
3. Update swap-aware debug helpers to classify PEFT-prefixed swapped CPU modules correctly.
4. Add or adjust trainer tests covering prepare sequencing, residency restore, and debug prefix handling.
5. Validate with a swap-enabled single-GPU LoRA startup run and inspect the post-prepare and post-restore device summaries.

Rollback strategy:

- Remove the non-default transformer prepare path and fall back to the current startup flow.
- Keep the model-native block-swap API intact and disable swap in practice by using `blocks_to_swap: 0` if necessary.

## Open Questions

- Whether the trainer should always unwrap before residency repair, or only in the swap-enabled Accelerate path.
- Whether quantized swap-enabled startup also needs a stricter “load-on-CPU until prepare completes” rule beyond the current repair work.
- Whether additional debug logging should explicitly print `after_restore_block_swap` on live smoke runs as part of the supported validation workflow.
