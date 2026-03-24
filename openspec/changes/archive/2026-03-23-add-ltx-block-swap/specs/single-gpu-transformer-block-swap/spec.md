## ADDED Requirements

### Requirement: Trainer configuration can enable transformer block swap
The trainer SHALL provide configuration fields that let single-GPU LoRA runs opt into transformer block swapping without changing the default startup path for runs that do not enable swapping.

#### Scenario: Default training remains unchanged
- **WHEN** a trainer configuration leaves block swapping disabled
- **THEN** trainer startup MUST preserve the existing non-swap transformer preparation path

#### Scenario: Swap is explicitly configured
- **WHEN** a trainer configuration sets a positive number of blocks to swap
- **THEN** trainer startup MUST enable transformer block swapping for the loaded transformer before training begins

### Requirement: LTX transformer exposes a model-native block-swap API
The LTX transformer SHALL expose model-native methods for enabling block swapping, preparing non-swapped modules on the target device, and preparing swap residency before forward execution.

#### Scenario: Swap API is available on the base transformer
- **WHEN** trainer code resolves the underlying LTX transformer from a PEFT-wrapped model
- **THEN** the base transformer MUST provide methods to enable block swap, prepare device placement excluding swapped blocks, and prepare swap state before forward execution

#### Scenario: Swap state is tracked on the model
- **WHEN** block swapping is enabled on the transformer
- **THEN** the transformer MUST retain the configured swap count and offloader state required to run the block loop with swapping enabled

### Requirement: Transformer block execution supports swap-aware scheduling during training
The transformer SHALL coordinate block swapping at the block loop boundary so that each swapped block is available before execution and the next swap operation is scheduled after execution in both normal and checkpointed training paths.

#### Scenario: Non-checkpointed block execution waits before running
- **WHEN** swap-enabled training executes a transformer block without gradient checkpointing
- **THEN** the block loop MUST wait for the current block residency to be ready before executing that block

#### Scenario: Block execution schedules the next swap
- **WHEN** swap-enabled training finishes executing a transformer block
- **THEN** the block loop MUST schedule the next forward swap operation for the configured block list

#### Scenario: Checkpointed execution remains swap-aware
- **WHEN** swap-enabled training executes a transformer block through gradient checkpointing
- **THEN** the same wait-before-execute and schedule-after-execute behavior MUST apply around the checkpointed block call

### Requirement: Quantized single-GPU LoRA startup remains compatible with block swap
Single-GPU LoRA startup SHALL support enabling block swapping after trainer quantization so that the configured transformer residency can be prepared without requiring a different quantization flow.

#### Scenario: Quantized startup enables swap after quantization
- **WHEN** trainer startup uses a supported quantization mode together with block swapping
- **THEN** the transformer MUST be quantized before block swap is enabled

#### Scenario: Startup preserves intended residency after model preparation
- **WHEN** a quantized or non-quantized swap-enabled transformer is prepared for training
- **THEN** trainer startup MUST restore the intended CPU/GPU residency layout for swapped and non-swapped transformer blocks before the first training forward pass

### Requirement: Swap-enabled startup is structurally verifiable
The codebase SHALL provide tests or equivalent structural verification for swap configuration, API availability, and startup residency behavior so that the feature can be implemented incrementally.

#### Scenario: Structural checks cover swap configuration
- **WHEN** automated tests run for the trainer and transformer startup helpers
- **THEN** they MUST verify the swap configuration defaults and the presence of the swap enablement path

#### Scenario: Structural checks cover startup residency preparation
- **WHEN** automated tests run for swap-enabled startup helpers
- **THEN** they MUST verify that the startup path can prepare non-swapped modules on the accelerator device without requiring all transformer blocks to remain resident there
