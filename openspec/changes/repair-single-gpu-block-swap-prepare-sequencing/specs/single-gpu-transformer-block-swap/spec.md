## ADDED Requirements

### Requirement: Swap-aware debug reporting classifies PEFT-wrapped swapped modules correctly
Swap-aware trainer debug helpers SHALL classify CPU-resident parameters and buffers belonging to intentionally swapped transformer blocks as expected residency even when the transformer is wrapped by PEFT or similar wrappers.

#### Scenario: PEFT-prefixed swapped blocks are treated as expected CPU residency
- **WHEN** a swap-enabled LoRA transformer reports CPU modules using wrapper-prefixed names such as `base_model.model.transformer_blocks.<idx>`
- **THEN** swap-aware debug reporting MUST classify those modules as expected CPU residency for the configured swapped block range

## MODIFIED Requirements

### Requirement: Quantized single-GPU LoRA startup remains compatible with block swap
Single-GPU LoRA startup SHALL support enabling block swapping after trainer quantization so that the configured transformer residency can be prepared without requiring a different quantization flow, and SHALL preserve that intended residency layout across `Accelerator.prepare()` for swap-enabled runs.

#### Scenario: Quantized startup enables swap after quantization
- **WHEN** trainer startup uses a supported quantization mode together with block swapping
- **THEN** the transformer MUST be quantized before block swap is enabled

#### Scenario: Swap-enabled startup prevents eager residency collapse during model preparation
- **WHEN** a swap-enabled transformer is prepared for training through `Accelerator.prepare()`
- **THEN** trainer startup MUST use transformer preparation behavior that does not eagerly place all swapped transformer block weights on the accelerator before residency repair runs

#### Scenario: Startup preserves intended residency after model preparation
- **WHEN** a quantized or non-quantized swap-enabled transformer is prepared for training
- **THEN** trainer startup MUST restore the intended CPU/GPU residency layout for swapped and non-swapped transformer blocks on the prepared model before the first training forward pass

### Requirement: Swap-enabled startup is structurally verifiable
The codebase SHALL provide tests or equivalent structural verification for swap configuration, API availability, startup residency behavior, and swap-aware debug classification so that the feature can be implemented incrementally and validated under wrapper-aware trainer startup.

#### Scenario: Structural checks cover swap configuration
- **WHEN** automated tests run for the trainer and transformer startup helpers
- **THEN** they MUST verify the swap configuration defaults and the presence of the swap enablement path

#### Scenario: Structural checks cover startup residency preparation
- **WHEN** automated tests run for swap-enabled startup helpers
- **THEN** they MUST verify that the startup path can prepare non-swapped modules on the accelerator device without requiring all transformer blocks to remain resident there

#### Scenario: Structural checks cover non-default prepare sequencing
- **WHEN** automated tests run for swap-enabled trainer preparation
- **THEN** they MUST verify that the transformer prepare path uses swap-compatible Accelerate device placement behavior and re-applies residency repair on the prepared model

#### Scenario: Structural checks cover PEFT-aware CPU module classification
- **WHEN** automated tests run for swap-aware device summary helpers
- **THEN** they MUST verify that wrapper-prefixed swapped block module names are excluded from unexpected CPU residency reports
