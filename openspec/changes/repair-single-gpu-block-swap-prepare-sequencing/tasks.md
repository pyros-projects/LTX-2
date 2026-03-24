## 1. Swap-compatible trainer prepare sequencing

- [ ] 1.1 Update swap-enabled trainer preparation so the transformer uses non-default Accelerate device placement behavior instead of the current eager full-model placement path.
- [ ] 1.2 Rework swap-enabled post-prepare residency repair to operate on the prepared and unwrapped transformer object before the first training forward pass.
- [ ] 1.3 Verify that the swap-enabled quantize-then-swap startup order remains intact after the prepare-sequencing changes.

## 2. Swap-aware PEFT residency reporting

- [ ] 2.1 Update expected swapped CPU module prefix detection so PEFT-wrapped transformer block names are classified against the configured swapped block range.
- [ ] 2.2 Adjust unexpected CPU module filtering and device-summary helpers so intentionally swapped PEFT-prefixed modules are no longer misreported as unexpected CPU residency.

## 3. Structural verification

- [ ] 3.1 Extend trainer startup tests to cover swap-enabled transformer preparation with swap-compatible Accelerate device placement.
- [ ] 3.2 Add tests that verify residency repair is re-applied on the prepared and unwrapped transformer for swap-enabled runs.
- [ ] 3.3 Add debug-helper tests that verify PEFT-prefixed swapped module names are excluded from unexpected CPU residency reports.

## 4. Targeted validation

- [ ] 4.1 Run a swap-enabled single-GPU LoRA startup check and confirm the `after_prepare` and `after_restore_block_swap` device summaries reflect the intended residency behavior.
- [ ] 4.2 Capture any remaining caveats for swap-enabled quantized runs if post-fix smoke validation still shows backend-specific residency or transfer issues.
